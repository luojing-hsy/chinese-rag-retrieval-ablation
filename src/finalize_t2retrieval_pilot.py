from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np

from run_t2retrieval_ablation import (
    CLEANING_VERSION,
    DATA_DIR,
    MODEL_CACHE,
    MODEL_NAME,
    PROCESSED_DIR,
    RERANKER_CACHE,
    RERANKER_MODEL_NAME,
    RESULTS_DIR,
    TextCrossEncoder,
    TextEmbedding,
    load_data,
    normalize_vectors,
    write_jsonl,
)


TOP_K = 20
RERANK_BATCH_SIZE = 32
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
SENTENCE_END = re.compile(r"(?<=[。！？!?；;])")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate_pid_rankings(rankings, query_ids, qrels):
    rows = []
    for qid, ranked_ids in zip(query_ids, rankings):
        relevant = qrels[qid]
        hits = [1 if pid in relevant else 0 for pid in ranked_ids]
        recall = lambda k: sum(hits[:k]) / len(relevant)
        first = next((rank for rank, hit in enumerate(hits[:10], start=1) if hit), None)
        mrr10 = 1.0 / first if first else 0.0
        dcg10 = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:10], start=1))
        idcg10 = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
        rows.append(
            {
                "recall@5": recall(5),
                "recall@10": recall(10),
                "recall@20": recall(20),
                "mrr@10": mrr10,
                "ndcg@10": dcg10 / idcg10 if idcg10 else 0.0,
                "precision@10": sum(hits[:10]) / 10,
            }
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def save_ranked_rows(path: Path, method: str, query_ids, query_texts, rankings, scores, score_type: str):
    rows = []
    for qid, query, pids, doc_scores in zip(query_ids, query_texts, rankings, scores):
        rows.append(
            {
                "query_id": qid,
                "query": query,
                "method": method,
                "results": [
                    {"corpus_id": pid, "score": float(score), "score_type": score_type, "rank": rank}
                    for rank, (pid, score) in enumerate(zip(pids, doc_scores), start=1)
                ],
            }
        )
    write_jsonl(path, rows)


def load_saved_top20(method: str, query_ids):
    source = RESULTS_DIR / f"retrievals_{method}.jsonl"
    rows = read_jsonl(source)
    if [row["query_id"] for row in rows] != query_ids:
        raise ValueError(f"Saved {method} rankings do not match the current query sample")
    rankings = [[item["corpus_id"] for item in row["results"][:TOP_K]] for row in rows]
    scores = [[float(item["score"]) for item in row["results"][:TOP_K]] for row in rows]
    return rankings, scores


def load_aligned_a3(path: Path, query_ids, bm25_candidates, candidate_k: int):
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if [row.get("query_id") for row in rows] != query_ids:
        return None
    rankings = [[item["corpus_id"] for item in row.get("results", [])] for row in rows]
    scores = [[float(item["score"]) for item in row.get("results", [])] for row in rows]
    if len(rankings) != len(query_ids):
        return None
    if any(len(ranking) != candidate_k or set(ranking) != set(candidates) for ranking, candidates in zip(rankings, bm25_candidates)):
        return None
    return rankings, scores


def split_into_chunks(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    pieces = [piece.strip() for piece in SENTENCE_END.split(text) if piece.strip()]
    chunks = []
    current = ""
    for piece in pieces:
        if len(piece) <= max_chars:
            if current and len(current) + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
            continue

        if current:
            chunks.append(current)
            current = ""
        start = 0
        while start < len(piece):
            end = start + max_chars
            chunks.append(piece[start:end])
            if end >= len(piece):
                break
            start = end - overlap

    if current:
        chunks.append(current)
    return chunks or [text or " "]


def aggregate_chunk_scores(chunk_vectors, query_vectors, parent_indices, parent_count, top_k):
    rankings = []
    score_lists = []
    started = time.perf_counter()
    for query_vector in query_vectors:
        chunk_scores = np.asarray(chunk_vectors @ query_vector, dtype=np.float32)
        parent_scores = np.full(parent_count, -np.inf, dtype=np.float32)
        np.maximum.at(parent_scores, parent_indices, chunk_scores)
        top = np.argsort(parent_scores)[::-1][:top_k]
        rankings.append(top.astype(np.int64).tolist())
        score_lists.append(parent_scores[top].astype(float).tolist())
    return rankings, score_lists, time.perf_counter() - started


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(DATA_DIR)
    sample_rows = read_jsonl(RESULTS_DIR / "query_sample.jsonl")
    query_ids = [row["query_id"] for row in sample_rows]
    query_texts = [row["query"] for row in sample_rows]
    qrels = {row["query_id"]: set(row["relevant_ids"]) for row in sample_rows}
    candidate_ids = (RESULTS_DIR / "candidate_pool_doc_ids.txt").read_text(encoding="utf-8").splitlines()
    if len(candidate_ids) != 1000 or len(set(candidate_ids)) != 1000:
        raise ValueError("Expected exactly 1,000 unique parent passage IDs in the pilot pool")

    full_doc_by_id = dict(zip(data["doc_ids"], data["documents"]))
    missing = set(candidate_ids) - set(full_doc_by_id)
    if missing:
        raise ValueError(f"{len(missing)} pilot passage IDs are missing from the MTEB corpus")
    doc_by_id = {pid: full_doc_by_id[pid] for pid in candidate_ids}
    candidate_set = set(candidate_ids)
    if any(not qrels[qid] <= candidate_set for qid in query_ids):
        raise ValueError("The pilot candidate pool is missing a relevant passage")

    original_metrics = RESULTS_DIR / "metrics.csv"
    initial_metrics = RESULTS_DIR / "metrics_top50_a0_a2_initial.csv"
    if original_metrics.exists() and not initial_metrics.exists():
        shutil.copy2(original_metrics, initial_metrics)
    old_metrics = {row["method"]: row for row in csv.DictReader(initial_metrics.open(encoding="utf-8-sig"))}

    methods = ["bm25", "dense_bge_small_zh", "hybrid_rrf"]
    rankings_by_method = {}
    scores_by_method = {}
    metric_rows = []
    for method in methods:
        rankings, scores = load_saved_top20(method, query_ids)
        rankings_by_method[method] = rankings
        scores_by_method[method] = scores
        save_ranked_rows(
            RESULTS_DIR / f"retrievals_{method}_top20.jsonl",
            method,
            query_ids,
            query_texts,
            rankings,
            scores,
            "original_retriever",
        )
        old = old_metrics.get(method, {})
        metric_rows.append(
            {
                "method": method,
                "queries": len(query_ids),
                "parent_passages": len(candidate_ids),
                "retrieval_units": len(candidate_ids),
                "top_k": TOP_K,
                "input_rank_depth": 100,
                "index_seconds": float(old.get("index_seconds", 0.0)),
                "query_seconds_total": float(old.get("query_seconds_total", 0.0)),
                "query_ms_mean": float(old.get("query_ms_mean", 0.0)),
                **evaluate_pid_rankings(rankings, query_ids, qrels),
            }
        )
    print("Reused saved A0-A2 rankings and recalculated top-20 metrics", flush=True)

    bm25_candidates = rankings_by_method["bm25"]
    a3_path = RESULTS_DIR / "retrievals_bm25_top20_bge_rerank.jsonl"
    prior_metadata_path = RESULTS_DIR / "run_metadata.json"
    prior_metadata = json.loads(prior_metadata_path.read_text(encoding="utf-8")) if prior_metadata_path.exists() else {}
    cached_a3 = load_aligned_a3(a3_path, query_ids, bm25_candidates, TOP_K)
    if cached_a3:
        a3_rankings, a3_scores = cached_a3
        a3_metadata = prior_metadata.get("a3", {})
        reranker_load_seconds = a3_metadata.get("model_load_seconds")
        reranker_query_seconds = a3_metadata.get("query_scoring_seconds")
        print("Reused saved A3 ranking; query and candidate IDs match", flush=True)
    else:
        reranker_started = time.perf_counter()
        reranker = TextCrossEncoder(
            model_name=RERANKER_MODEL_NAME,
            cache_dir=str(RERANKER_CACHE),
            threads=4,
            cuda=False,
            local_files_only=True,
        )
        reranker_load_seconds = time.perf_counter() - reranker_started
        a3_rankings = []
        a3_scores = []
        with a3_path.open("w", encoding="utf-8", newline="\n") as stream:
            rerank_started = time.perf_counter()
            for index, (qid, query, candidates) in enumerate(zip(query_ids, query_texts, bm25_candidates), start=1):
                candidate_texts = [doc_by_id[pid] for pid in candidates]
                scores = [float(score) for score in reranker.rerank(query, candidate_texts, batch_size=32)]
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                ranked_ids = [candidates[i] for i in order]
                ranked_scores = [scores[i] for i in order]
                a3_rankings.append(ranked_ids)
                a3_scores.append(ranked_scores)
                row = {
                    "query_id": qid,
                    "query": query,
                    "method": "bm25_top20_bge_rerank",
                    "results": [
                        {"corpus_id": pid, "score": score, "score_type": "cross_encoder", "rank": rank}
                        for rank, (pid, score) in enumerate(zip(ranked_ids, ranked_scores), start=1)
                    ],
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if index % 10 == 0:
                    stream.flush()
                    print(f"A3 reranked {index}/{len(query_ids)} queries", flush=True)
        reranker_query_seconds = time.perf_counter() - rerank_started
    a3_metrics = evaluate_pid_rankings(a3_rankings, query_ids, qrels)
    metric_rows.append(
        {
            "method": "bm25_top20_bge_rerank",
            "queries": len(query_ids),
            "parent_passages": len(candidate_ids),
            "retrieval_units": len(candidate_ids),
            "top_k": TOP_K,
            "input_rank_depth": 20,
            "index_seconds": round(reranker_load_seconds, 3) if reranker_load_seconds is not None else None,
            "query_seconds_total": round(reranker_query_seconds, 3) if reranker_query_seconds is not None else None,
            "query_ms_mean": round(1000 * reranker_query_seconds / len(query_ids), 3) if reranker_query_seconds is not None else None,
            **a3_metrics,
        }
    )
    if reranker_query_seconds is None:
        print("A3 rankings reused; original scoring time was not persisted", flush=True)
    else:
        print(f"A3 reranker completed in {reranker_query_seconds:.1f}s", flush=True)

    # A4: re-encode sentence-aware child chunks, then aggregate maximum score to the same parent IDs.
    chunks = []
    chunk_parent_indices = []
    parent_index = {pid: index for index, pid in enumerate(candidate_ids)}
    chunk_prepare_started = time.perf_counter()
    for pid in candidate_ids:
        parent_chunks = split_into_chunks(doc_by_id[pid])
        chunks.extend(parent_chunks)
        chunk_parent_indices.extend([parent_index[pid]] * len(parent_chunks))
    chunk_parent_indices = np.asarray(chunk_parent_indices, dtype=np.int64)
    chunk_prepare_seconds = time.perf_counter() - chunk_prepare_started
    print(f"A4 created {len(chunks)} sentence-aware child chunks from 1,000 parent passages", flush=True)

    embedding_model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_CACHE),
        threads=4,
        cuda=False,
        local_files_only=True,
    )
    chunk_embed_started = time.perf_counter()
    chunk_vector_list = []
    for index, vector in enumerate(embedding_model.embed(chunks, batch_size=64), start=1):
        chunk_vector_list.append(np.asarray(vector, dtype=np.float32))
        if index % 512 == 0:
            print(f"A4 embedded {index}/{len(chunks)} chunks", flush=True)
    chunk_vectors = normalize_vectors(np.vstack(chunk_vector_list))
    chunk_embed_seconds = time.perf_counter() - chunk_embed_started
    query_embedding_started = time.perf_counter()
    query_vectors = normalize_vectors(np.asarray(list(embedding_model.query_embed(query_texts))))
    query_embedding_seconds = time.perf_counter() - query_embedding_started
    a4_rank_indices, a4_scores, a4_search_seconds = aggregate_chunk_scores(
        chunk_vectors, query_vectors, chunk_parent_indices, len(candidate_ids), TOP_K
    )
    a4_rankings = [[candidate_ids[index] for index in row] for row in a4_rank_indices]
    save_ranked_rows(
        RESULTS_DIR / "retrievals_dense_sentence_chunks_parent_max_top20.jsonl",
        "dense_sentence_chunks_parent_max",
        query_ids,
        query_texts,
        a4_rankings,
        a4_scores,
        "max_child_cosine_similarity",
    )
    a4_metrics = evaluate_pid_rankings(a4_rankings, query_ids, qrels)
    metric_rows.append(
        {
            "method": "dense_sentence_chunks_parent_max",
            "queries": len(query_ids),
            "parent_passages": len(candidate_ids),
            "retrieval_units": len(chunks),
            "top_k": TOP_K,
            "input_rank_depth": len(chunks),
            "index_seconds": round(chunk_prepare_seconds + chunk_embed_seconds, 3),
            "query_seconds_total": round(query_embedding_seconds + a4_search_seconds, 3),
            "query_ms_mean": round(1000 * (query_embedding_seconds + a4_search_seconds) / len(query_ids), 3),
            **a4_metrics,
        }
    )

    metrics_path = RESULTS_DIR / "metrics.csv"
    fieldnames = [
        "method", "queries", "parent_passages", "retrieval_units", "top_k", "input_rank_depth",
        "index_seconds", "query_seconds_total", "query_ms_mean", "recall@5", "recall@10",
        "recall@20", "mrr@10", "ndcg@10", "precision@10",
    ]
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    initial_metadata_path = RESULTS_DIR / "run_metadata.json"
    metadata = json.loads(initial_metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "top_k": TOP_K,
            "reported_top_k": TOP_K,
            "candidate_parent_passages": len(candidate_ids),
            "methods_completed": [row["method"] for row in metric_rows],
            "a3": {
                "model": RERANKER_MODEL_NAME,
                "candidate_ids_source": "same BM25 top-20 IDs as A0",
                "only_change": "ordering by cross-encoder scores",
                "local_only": True,
                "model_load_seconds": round(reranker_load_seconds, 3) if reranker_load_seconds is not None else None,
                "query_scoring_seconds": round(reranker_query_seconds, 3) if reranker_query_seconds is not None else None,
                "timing_note": "Timing is omitted if an existing A3 ranking is reused without a saved timing record.",
            },
            "a4": {
                "chunking": "sentence-boundary merge, max 300 chars, overlap 50 for long sentences",
                "parent_pool": len(candidate_ids),
                "child_chunks": len(chunks),
                "aggregation": "max child cosine similarity per parent passage",
                "local_model": MODEL_NAME,
            "chunk_prepare_seconds": round(chunk_prepare_seconds, 3),
            "chunk_embedding_seconds": round(chunk_embed_seconds, 3),
            "query_embedding_seconds": round(query_embedding_seconds, 3),
            "parent_search_seconds": round(a4_search_seconds, 3),
            },
            "metrics_file": "metrics.csv (all methods reported at top-20)",
        }
    )
    initial_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"A4 chunk indexing: {chunk_embed_seconds:.1f}s; search: {a4_search_seconds:.1f}s", flush=True)
    print(f"Wrote unified top-20 metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
