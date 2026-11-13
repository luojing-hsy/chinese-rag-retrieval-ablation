from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_HOME = ROOT / "models" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(HF_HOME / "xet"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import bm25s
import jieba
import numpy as np
import pyarrow.parquet as pq
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


DATA_DIR = ROOT / "data" / "raw" / "t2retrieval"
PROCESSED_DIR = ROOT / "data" / "processed" / "t2retrieval" / "pilot"
RESULTS_DIR = ROOT / "results" / "t2retrieval" / "pilot"
MODEL_CACHE = ROOT / "models" / "fastembed"
RERANKER_CACHE = ROOT / "models" / "fastembed-reranker"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
CLEANING_VERSION = "html-tags-unescape-nfkc-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: str | None) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(data_dir: Path):
    corpus_path = data_dir / "mteb-corpus-dev.parquet"
    queries_path = data_dir / "mteb-queries-dev.parquet"
    qrels_path = data_dir / "mteb-qrels-dev.parquet"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing MTEB dev file: {path}")

    corpus_rows = pq.read_table(corpus_path, columns=["_id", "text"]).to_pylist()
    corpus_rows.sort(key=lambda row: row["_id"])
    doc_ids = [row["_id"] for row in corpus_rows]
    documents = [clean_text(row["text"]) or " " for row in corpus_rows]

    query_rows = pq.read_table(queries_path, columns=["_id", "text"]).to_pylist()
    query_by_id = {row["_id"]: clean_text(row["text"]) for row in query_rows}

    qrel_rows = pq.read_table(qrels_path).to_pylist()
    qrels: dict[str, set[str]] = {}
    scores = set()
    for row in qrel_rows:
        qid = row["query-id"]
        pid = row["corpus-id"]
        qrels.setdefault(qid, set()).add(pid)
        scores.add(row["score"])

    doc_id_set = set(doc_ids)
    missing_queries = set(qrels) - set(query_by_id)
    missing_docs = {pid for relevant in qrels.values() for pid in relevant} - doc_id_set
    if missing_queries or missing_docs:
        raise ValueError(
            f"MTEB ID mismatch: {len(missing_queries)} query IDs and "
            f"{len(missing_docs)} passage IDs are missing"
        )
    if scores != {1}:
        raise ValueError(f"Expected binary MTEB qrels, found scores: {sorted(scores)}")

    return {
        "corpus_path": corpus_path,
        "queries_path": queries_path,
        "qrels_path": qrels_path,
        "doc_ids": doc_ids,
        "documents": documents,
        "query_by_id": query_by_id,
        "qrels": qrels,
        "qrel_rows": qrel_rows,
    }


def sample_queries(query_by_id: dict[str, str], qrels: dict[str, set[str]], count: int, seed: int):
    valid_ids = sorted(set(query_by_id) & set(qrels))
    if count <= 0 or count >= len(valid_ids):
        selected = valid_ids
    else:
        selected = random.Random(seed).sample(valid_ids, count)
        selected.sort()
    return selected, [query_by_id[qid] for qid in selected]


def sample_candidate_corpus(doc_ids, documents, query_ids, qrels, candidate_passages: int, seed: int):
    relevant_ids = set().union(*(qrels[qid] for qid in query_ids))
    if len(relevant_ids) > candidate_passages:
        raise ValueError(
            f"The sampled queries require {len(relevant_ids)} gold passages, "
            f"which exceeds the candidate limit {candidate_passages}. "
            "Reduce --sample-queries or increase --candidate-passages."
        )
    id_to_position = {doc_id: position for position, doc_id in enumerate(doc_ids)}
    negative_pool = [doc_id for doc_id in doc_ids if doc_id not in relevant_ids]
    negative_count = min(candidate_passages - len(relevant_ids), len(negative_pool))
    sampled_negatives = random.Random(seed).sample(negative_pool, negative_count)
    candidate_ids = sorted(relevant_ids | set(sampled_negatives))
    candidate_documents = [documents[id_to_position[doc_id]] for doc_id in candidate_ids]
    return candidate_ids, candidate_documents, len(relevant_ids), negative_count


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors


def load_or_build_document_embeddings(documents, corpus_sha256: str, doc_ids_sha256: str, model, batch_size: int = 64, parallel: int | None = None):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    vector_path = PROCESSED_DIR / "bge-small-zh-v1.5-document-vectors.npy"
    metadata_path = PROCESSED_DIR / "bge-small-zh-v1.5-document-vectors.json"
    expected = {
        "model": MODEL_NAME,
        "corpus_sha256": corpus_sha256,
        "candidate_doc_ids_sha256": doc_ids_sha256,
        "document_count": len(documents),
        "cleaning_version": CLEANING_VERSION,
        "dimension": 512,
    }

    if vector_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            return np.load(vector_path, mmap_mode="r"), float(metadata.get("build_seconds", 0.0)), True

    vector_path.unlink(missing_ok=True)
    started = time.perf_counter()
    vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float32, shape=(len(documents), 512))
    for index, vector in enumerate(model.embed(documents, batch_size=batch_size, parallel=parallel)):
        vectors[index] = normalize_vectors(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        if (index + 1) % 512 == 0:
            print(f"Embedded {index + 1}/{len(documents)} passages", flush=True)
    vectors.flush()
    elapsed = time.perf_counter() - started
    del vectors
    expected["build_seconds"] = elapsed
    metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return np.load(vector_path, mmap_mode="r"), elapsed, False


def topk_dense(query_vectors: np.ndarray, doc_vectors: np.ndarray, k: int):
    all_indices = []
    all_scores = []
    started = time.perf_counter()
    for query_vector in query_vectors:
        scores = np.asarray(doc_vectors @ query_vector, dtype=np.float32)
        candidates = np.argpartition(scores, -k)[-k:]
        candidates = candidates[np.argsort(scores[candidates])[::-1]]
        all_indices.append(candidates.astype(np.int64).tolist())
        all_scores.append(scores[candidates].astype(float).tolist())
    return all_indices, all_scores, time.perf_counter() - started


def rerank_bm25_candidates(reranker, queries, documents, bm25_indices, candidate_k: int, batch_size: int = 32):
    rankings = []
    ranking_scores = []
    started = time.perf_counter()
    for query_index, (query, bm25_row) in enumerate(zip(queries, bm25_indices), start=1):
        candidate_indices = bm25_row[:candidate_k]
        candidate_texts = [documents[index] for index in candidate_indices]
        scores = [float(score) for score in reranker.rerank(query, candidate_texts, batch_size=batch_size)]
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        rankings.append([candidate_indices[index] for index in order])
        ranking_scores.append([scores[index] for index in order])
        if query_index % 10 == 0:
            print(f"Reranked {query_index}/{len(queries)} queries", flush=True)
    return rankings, ranking_scores, time.perf_counter() - started


def rrf_rankings(left_indices, right_indices, k: int, rrf_constant: int = 60):
    rankings = []
    ranking_scores = []
    for left, right in zip(left_indices, right_indices):
        fused = {}
        for ranking in (left, right):
            for rank, doc_index in enumerate(ranking, start=1):
                fused[doc_index] = fused.get(doc_index, 0.0) + 1.0 / (rrf_constant + rank)
        top = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        rankings.append([index for index, _ in top])
        ranking_scores.append([float(score) for _, score in top])
    return rankings, ranking_scores


def evaluate(rankings, doc_ids, query_ids, qrels):
    per_query = []
    for qid, ranking in zip(query_ids, rankings):
        relevant = qrels[qid]
        ranked_ids = [doc_ids[index] for index in ranking]
        hits = [1 if pid in relevant else 0 for pid in ranked_ids]

        def recall(k):
            return sum(hits[:k]) / len(relevant)

        first = next((rank for rank, hit in enumerate(hits[:10], start=1) if hit), None)
        mrr10 = 1.0 / first if first else 0.0
        dcg10 = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:10], start=1))
        idcg10 = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
        per_query.append(
            {
                "recall@5": recall(5),
                "recall@10": recall(10),
                "recall@20": recall(20),
                "recall@50": recall(50) if len(hits) >= 50 else None,
                "mrr@10": mrr10,
                "ndcg@10": dcg10 / idcg10 if idcg10 else 0.0,
                "precision@10": sum(hits[:10]) / 10,
            }
        )
    summary = {}
    for key in per_query[0]:
        values = [row[key] for row in per_query if row[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def save_rankings(path: Path, method: str, query_ids, query_texts, rankings, ranking_scores, doc_ids):
    rows = []
    for qid, query, indices, scores in zip(query_ids, query_texts, rankings, ranking_scores):
        rows.append(
            {
                "query_id": qid,
                "query": query,
                "method": method,
                "results": [
                    {"corpus_id": doc_ids[index], "score": float(score), "rank": rank}
                    for rank, (index, score) in enumerate(zip(indices, scores), start=1)
                ],
            }
        )
    write_jsonl(path, rows)


def run(args):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jieba.setLogLevel(40)

    data = load_data(DATA_DIR)
    query_ids, query_texts = sample_queries(data["query_by_id"], data["qrels"], args.sample_queries, args.seed)
    query_qrels = {qid: data["qrels"][qid] for qid in query_ids}
    full_corpus_count = len(data["doc_ids"])
    doc_ids, documents, positive_doc_count, negative_doc_count = sample_candidate_corpus(
        data["doc_ids"], data["documents"], query_ids, query_qrels, args.candidate_passages, args.seed + 1
    )
    print(
        f"Loaded {full_corpus_count} MTEB passages, {len(data['query_by_id'])} queries, "
        f"{len(data['qrel_rows'])} qrels; evaluating {len(query_ids)} queries over "
        f"{len(doc_ids)} candidate passages ({positive_doc_count} relevant-union + {negative_doc_count} random)"
    )
    print("Building BM25 index", flush=True)

    write_jsonl(
        RESULTS_DIR / "query_sample.jsonl",
        [{"query_id": qid, "query": query_texts[i], "relevant_ids": sorted(query_qrels[qid])} for i, qid in enumerate(query_ids)],
    )
    (RESULTS_DIR / "candidate_pool_doc_ids.txt").write_text("\n".join(doc_ids) + "\n", encoding="utf-8")

    corpus_sha = sha256_file(data["corpus_path"])
    doc_ids_sha = hashlib.sha256("\n".join(doc_ids).encode("utf-8")).hexdigest()
    tokenizer = bm25s.tokenization.Tokenizer(
        lower=True,
        splitter=r"(?u)\b\w+\b",
        stopwords=[],
    )
    segmented_documents = [" ".join(jieba.cut(text, HMM=False)) for text in documents]
    started = time.perf_counter()
    corpus_tokens = tokenizer.tokenize(segmented_documents, update_vocab="if_empty", show_progress=True)
    bm25 = bm25s.BM25(k1=1.5, b=0.75, method="lucene", dtype="float32")
    bm25.index(corpus_tokens, show_progress=True)
    bm25_index_seconds = time.perf_counter() - started

    segmented_queries = [" ".join(jieba.cut(text, HMM=False)) for text in query_texts]
    query_tokens = tokenizer.tokenize(segmented_queries, update_vocab=False, show_progress=True)
    started = time.perf_counter()
    bm25_indices, bm25_scores = bm25.retrieve(query_tokens, k=args.candidate_k, show_progress=True)
    bm25_query_seconds = time.perf_counter() - started
    bm25_indices = np.asarray(bm25_indices).tolist()
    bm25_scores = np.asarray(bm25_scores).astype(float).tolist()

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_CACHE),
        threads=4,
        cuda=False,
        local_files_only=True,
    )
    doc_vectors, embedding_index_seconds, embedding_cache_hit = load_or_build_document_embeddings(
        documents,
        corpus_sha,
        doc_ids_sha,
        model,
        batch_size=args.embedding_batch_size,
    )
    query_embedding_started = time.perf_counter()
    query_vectors = normalize_vectors(np.asarray(list(model.query_embed(query_texts))))
    query_embedding_seconds = time.perf_counter() - query_embedding_started
    dense_indices, dense_scores, dense_search_seconds = topk_dense(query_vectors, doc_vectors, args.candidate_k)
    dense_query_seconds = query_embedding_seconds + dense_search_seconds

    print(f"Loading local cross-encoder {RERANKER_MODEL_NAME}", flush=True)
    reranker_load_started = time.perf_counter()
    reranker = TextCrossEncoder(
        model_name=RERANKER_MODEL_NAME,
        cache_dir=str(RERANKER_CACHE),
        threads=4,
        cuda=False,
        local_files_only=True,
    )
    reranker_load_seconds = time.perf_counter() - reranker_load_started
    rerank_candidate_k = args.rerank_candidate_k
    reranked_indices, reranked_scores, reranker_query_seconds = rerank_bm25_candidates(
        reranker, query_texts, documents, bm25_indices, rerank_candidate_k, batch_size=args.reranker_batch_size
    )
    bm25_top20_indices = [row[:rerank_candidate_k] for row in bm25_indices]
    bm25_top20_scores = [row[:rerank_candidate_k] for row in bm25_scores]
    save_rankings(
        RESULTS_DIR / "retrievals_bm25_top20_no_rerank.jsonl",
        "bm25_top20_no_rerank",
        query_ids,
        query_texts,
        bm25_top20_indices,
        bm25_top20_scores,
        doc_ids,
    )
    save_rankings(
        RESULTS_DIR / "retrievals_bm25_top20_bge_rerank.jsonl",
        "bm25_top20_bge_rerank",
        query_ids,
        query_texts,
        reranked_indices,
        reranked_scores,
        doc_ids,
    )

    fused_indices, fused_scores = rrf_rankings(bm25_indices, dense_indices, args.top_k, args.rrf_constant)
    methods = [
        ("bm25", bm25_indices, bm25_scores[:]),
        ("dense_bge_small_zh", dense_indices, dense_scores),
        ("hybrid_rrf", fused_indices, fused_scores),
    ]
    metric_rows = []
    for method, rankings, scores in methods:
        rankings = [row[: args.top_k] for row in rankings]
        scores = [row[: args.top_k] for row in scores]
        metrics = evaluate(rankings, doc_ids, query_ids, query_qrels)
        metric_rows.append(
            {
                "method": method,
                "queries": len(query_ids),
                "full_mteb_corpus_passages": full_corpus_count,
                "candidate_passages": len(doc_ids),
                "top_k": args.top_k,
                "index_seconds": round(bm25_index_seconds if method == "bm25" else embedding_index_seconds if method == "dense_bge_small_zh" else bm25_index_seconds + embedding_index_seconds, 3),
                "query_seconds_total": round(bm25_query_seconds if method == "bm25" else dense_query_seconds if method == "dense_bge_small_zh" else bm25_query_seconds + dense_query_seconds, 3),
                "query_ms_mean": round(1000 * (bm25_query_seconds if method == "bm25" else dense_query_seconds if method == "dense_bge_small_zh" else bm25_query_seconds + dense_query_seconds) / len(query_ids), 3),
                **metrics,
            }
        )
        save_rankings(RESULTS_DIR / f"retrievals_{method}.jsonl", method, query_ids, query_texts, rankings, scores, doc_ids)

    bm25_top20_metrics = evaluate(bm25_top20_indices, doc_ids, query_ids, query_qrels)
    bm25_top20_query_seconds = bm25_query_seconds * rerank_candidate_k / args.candidate_k
    metric_rows.append(
        {
            "method": "bm25_top20_no_rerank",
            "queries": len(query_ids),
            "full_mteb_corpus_passages": full_corpus_count,
            "candidate_passages": len(doc_ids),
            "top_k": rerank_candidate_k,
            "index_seconds": round(bm25_index_seconds, 3),
            "query_seconds_total": round(bm25_top20_query_seconds, 3),
            "query_ms_mean": round(1000 * bm25_top20_query_seconds / len(query_ids), 3),
            **bm25_top20_metrics,
        }
    )
    rerank_metrics = evaluate(reranked_indices, doc_ids, query_ids, query_qrels)
    rerank_total_query_seconds = bm25_query_seconds + reranker_query_seconds
    metric_rows.append(
        {
            "method": "bm25_top20_bge_rerank",
            "queries": len(query_ids),
            "full_mteb_corpus_passages": full_corpus_count,
            "candidate_passages": len(doc_ids),
            "top_k": rerank_candidate_k,
            "index_seconds": round(bm25_index_seconds + reranker_load_seconds, 3),
            "query_seconds_total": round(rerank_total_query_seconds, 3),
            "query_ms_mean": round(1000 * rerank_total_query_seconds / len(query_ids), 3),
            **rerank_metrics,
        }
    )
    metrics_path = RESULTS_DIR / "metrics.csv"
    fieldnames = list(metric_rows[0])
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    metadata = {
        "dataset": "mteb/T2Retrieval",
        "subset": "dev",
        "dataset_license": "Apache-2.0",
        "candidate_pool_note": "Pilot candidate pool contains exactly the configured total: all qrel-relevant passages for the sampled queries plus deterministic random unjudged passages. Unjudged candidates are treated as non-relevant. This is not the official full MTEB/T2Ranking candidate corpus.",
        "original_t2ranking_alignment": {
            "mteb_query_ids": 22812,
            "original_query_rows": 24832,
            "mteb_qrels_pairs": 118932,
            "original_qrels_pairs_excluding_header": 118932,
            "query_text_quote_differences": 10,
            "full_original_corpus_hash_checked": False,
        },
        "sample_queries": len(query_ids),
        "full_mteb_corpus_passages": full_corpus_count,
        "candidate_passages": len(doc_ids),
        "sampled_relevant_union_passages": positive_doc_count,
        "sampled_random_unjudged_passages": negative_doc_count,
        "candidate_passages_requested": args.candidate_passages,
        "candidate_sample_seed": args.seed + 1,
        "seed": args.seed,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "rrf_constant": args.rrf_constant,
        "model": MODEL_NAME,
        "model_inference": "local FastEmbed ONNX CPU, serial document encoding; local_files_only=True",
        "document_embedding_cache_hit": embedding_cache_hit,
        "reranker_model": RERANKER_MODEL_NAME,
        "reranker_candidate_k": rerank_candidate_k,
        "reranker_load_seconds": round(reranker_load_seconds, 3),
        "reranker_query_seconds": round(reranker_query_seconds, 3),
        "reranker_inference": "local FastEmbed ONNX CPU; local_files_only=True",
        "corpus_file_sha256": corpus_sha,
        "query_file_sha256": sha256_file(data["queries_path"]),
        "qrels_file_sha256": sha256_file(data["qrels_path"]),
        "cleaning_version": CLEANING_VERSION,
        "python": os.sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("fastembed", "bm25s", "jieba", "pyarrow", "numpy", "onnxruntime")
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"BM25 index: {bm25_index_seconds:.1f}s; BM25 query: {bm25_query_seconds:.1f}s")
    print(f"Dense index: {embedding_index_seconds:.1f}s; dense query: {dense_query_seconds:.1f}s; cache_hit={embedding_cache_hit}")
    print(f"Reranker load: {reranker_load_seconds:.1f}s; reranker query: {reranker_query_seconds:.1f}s")
    print(f"Wrote metrics to {metrics_path}")
    for row in metric_rows:
        print(row["method"], {key: round(row[key], 4) if row[key] is not None else None for key in ("recall@5", "recall@10", "recall@20", "recall@50", "mrr@10", "ndcg@10")})


def parse_args():
    parser = argparse.ArgumentParser(description="Run local retrieval ablations on MTEB/T2Retrieval dev.")
    parser.add_argument("--sample-queries", type=int, default=100)
    parser.add_argument("--candidate-passages", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--rerank-candidate-k", type=int, default=20)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
