from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_t2retrieval_ablation import (
    DATA_DIR, MODEL_CACHE, MODEL_NAME, PROCESSED_DIR, RESULTS_DIR, ROOT,
    TextEmbedding, evaluate, load_data, normalize_vectors, save_rankings,
    sha256_file, topk_dense, write_jsonl,
)


OUTPUT = ROOT / "results" / "t2retrieval" / "pooling_ablation"
VECTOR_OUTPUT = PROCESSED_DIR / "pooling_ablation"
SNAPSHOT = MODEL_CACHE / "models--Qdrant--bge-small-zh-v1.5" / "snapshots" / "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
CONFIG = {
    "model": MODEL_NAME,
    "model_snapshot": SNAPSHOT.name,
    "dataset": "mteb/T2Retrieval dev derived pilot",
    "queries": 100, "passages": 1000, "output_k": 20,
    "retrieval": "exact cosine search over all 1000 original parent passages; no frozen top50 reranking",
    "only_change": "pooling applied consistently to query and passage token representations",
    "pooling": {"cls": "last_hidden_state[:,0]", "mean": "attention-mask-weighted mean of last_hidden_state; padding excluded, CLS/SEP included"},
    "normalization": "L2 for both query and passage in both variants",
    "max_length": 512, "truncation": "right", "document_batch_size": 64,
    "query_batch_size": 256, "threads": 4, "device": "CPU",
    "query_instruction": "none; original pilot query processing, not a query-enhancement experiment",
    "forward_pass": "shared token outputs for both pooling variants; fixed local ONNX weights",
    "interpretation": "frozen-model inference pooling comparison; not training-time or architecture ablation",
    "unjudged_policy": "treated as non-relevant, not confirmed negatives",
    "source": "https://huggingface.co/BAAI/bge-small-zh-v1.5#using-huggingface-transformers",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def encode_both(encoder, texts, batch_size, label):
    cls_batches, mean_batches, lengths = [], [], []
    forward_seconds = cls_seconds = mean_seconds = 0.0
    for start in range(0, len(texts), batch_size):
        tick = time.perf_counter()
        context = encoder.onnx_embed(texts[start:start + batch_size])
        forward_seconds += time.perf_counter() - tick
        hidden, attention = context.model_output, context.attention_mask
        assert hidden.ndim == 3 and hidden.shape[:2] == attention.shape and hidden.shape[2] == 512
        assert np.isfinite(hidden).all() and np.isin(attention, (0, 1)).all()
        tick = time.perf_counter()
        cls = normalize_vectors(hidden[:, 0].copy())
        cls_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        mask = attention.astype(np.float32)[..., None]
        counts = mask.sum(axis=1)
        assert np.all(counts > 0) and np.all(counts <= CONFIG["max_length"])
        mean = normalize_vectors((hidden * mask).sum(axis=1) / counts)
        mean_seconds += time.perf_counter() - tick
        # Check masked pooling against a direct mean over non-padding token positions.
        reference = normalize_vectors(hidden[0, attention[0].astype(bool)].mean(axis=0)[None, :])[0]
        assert np.allclose(mean[0], reference, atol=1e-5, rtol=1e-5)
        assert np.isfinite(mean).all() and np.allclose(np.linalg.norm(mean, axis=1), 1, atol=1e-5)
        cls_batches.append(cls)
        mean_batches.append(mean)
        lengths.extend(attention.sum(axis=1).tolist())
        print(f"{label}: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return {"cls": np.vstack(cls_batches), "mean": np.vstack(mean_batches)}, {
        "shared_forward_seconds": forward_seconds,
        "cls_pool_normalize_seconds": cls_seconds, "mean_pool_normalize_seconds": mean_seconds,
    }, lengths


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    VECTOR_OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "experiment_config.json", CONFIG)
    data_started = time.perf_counter()
    data = load_data(DATA_DIR)
    sample_path = RESULTS_DIR / "query_sample.jsonl"
    pool_path = RESULTS_DIR / "candidate_pool_doc_ids.txt"
    old_rank_path = RESULTS_DIR / "retrievals_dense_bge_small_zh_top20.jsonl"
    old_meta_path = RESULTS_DIR / "run_metadata.json"
    old_vectors_path = PROCESSED_DIR / "bge-small-zh-v1.5-document-vectors.npy"
    vector_meta_path = PROCESSED_DIR / "bge-small-zh-v1.5-document-vectors.json"
    sample, old_rankings = read_jsonl(sample_path), read_jsonl(old_rank_path)
    pool = pool_path.read_text(encoding="utf-8").splitlines()
    query_ids = [row["query_id"] for row in sample]
    queries = [row["query"] for row in sample]
    document_lookup = dict(zip(data["doc_ids"], data["documents"]))
    documents = [document_lookup[pid] for pid in pool]
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
    vector_meta = json.loads(vector_meta_path.read_text(encoding="utf-8"))
    assert len(query_ids) == len(set(query_ids)) == CONFIG["queries"]
    assert len(pool) == len(set(pool)) == CONFIG["passages"]
    assert query_ids == [row["query_id"] for row in old_rankings]
    for row in sample:
        assert row["query"] == data["query_by_id"][row["query_id"]]
        assert set(row["relevant_ids"]) == data["qrels"][row["query_id"]] <= set(pool)
    for path, key in ((data["corpus_path"], "corpus_file_sha256"), (data["queries_path"], "query_file_sha256"), (data["qrels_path"], "qrels_file_sha256")):
        assert sha256_file(path) == old_meta[key]
    import hashlib
    assert hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest() == vector_meta["candidate_doc_ids_sha256"]
    data_seconds = time.perf_counter() - data_started

    tick = time.perf_counter()
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE), specific_model_path=str(SNAPSHOT),
                          threads=4, cuda=False, local_files_only=True)
    encoder = model.model
    model_load_seconds = time.perf_counter() - tick
    assert encoder.tokenizer.truncation["max_length"] == CONFIG["max_length"]
    assert encoder.tokenizer.truncation["direction"] == "right"
    print("Loaded local token-output ONNX; encoding both pooling variants in shared passes", flush=True)
    doc_vectors, doc_timing, doc_lengths = encode_both(encoder, documents, CONFIG["document_batch_size"], "passages")
    query_vectors, query_timing, query_lengths = encode_both(encoder, queries, CONFIG["query_batch_size"], "queries")
    cached_cls = np.load(old_vectors_path)
    max_cls_error = float(np.max(np.abs(cached_cls - doc_vectors["cls"])))
    assert np.allclose(cached_cls, doc_vectors["cls"], atol=1e-5, rtol=1e-5)
    for pooling in CONFIG["pooling"]:
        np.save(VECTOR_OUTPUT / f"{pooling}_documents.npy", doc_vectors[pooling])
        np.save(VECTOR_OUTPUT / f"{pooling}_queries.npy", query_vectors[pooling])
    write_json(VECTOR_OUTPUT / "ids.json", {"passage_ids": pool, "query_ids": query_ids, "config": CONFIG})
    metrics, per_query, all_rankings, search_timing = [], [], {}, {}
    for pooling in CONFIG["pooling"]:
        ranks, scores, seconds = topk_dense(query_vectors[pooling], doc_vectors[pooling], CONFIG["output_k"])
        search_timing[pooling] = seconds
        all_rankings[pooling] = ranks
        assert all(len(rank) == len(set(rank)) == CONFIG["output_k"] for rank in ranks)
        summary = evaluate(ranks, pool, query_ids, data["qrels"])
        summary.pop("recall@50")
        metrics.append({"method": f"bge_{pooling}", "queries": len(query_ids), "passages": len(pool), "top_k": 20, **summary})
        save_rankings(OUTPUT / f"retrievals_bge_{pooling}.jsonl", f"bge_{pooling}", query_ids, queries, ranks, scores, pool)
        for qid, rank in zip(query_ids, ranks):
            values = evaluate([rank], pool, [qid], data["qrels"])
            values.pop("recall@50")
            per_query.append({"query_id": qid, "method": f"bge_{pooling}", **values})
    assert [[pool[i] for i in rank] for rank in all_rankings["cls"]] == [[item["corpus_id"] for item in row["results"]] for row in old_rankings]
    with (RESULTS_DIR / "metrics.csv").open(encoding="utf-8-sig") as stream:
        old_metrics = next(row for row in csv.DictReader(stream) if row["method"] == "dense_bge_small_zh")
    metric_names = list(metrics[0])[4:]
    assert all(abs(metrics[0][key] - float(old_metrics[key])) < 1e-12 for key in metric_names)
    baseline_by_qid = {row["query_id"]: row for row in per_query if row["method"] == "bge_cls"}
    paired = [{"query_id": row["query_id"], **{f"delta_{key}": row[key] - baseline_by_qid[row["query_id"]][key] for key in metric_names}}
              for row in per_query if row["method"] == "bge_mean"]
    write_csv(OUTPUT / "metrics.csv", metrics)
    write_csv(OUTPUT / "per_query_metrics.csv", per_query)
    write_csv(OUTPUT / "paired_deltas.csv", paired)
    write_json(OUTPUT / "paired_summary.json", {key: {
        "wins": sum(row[f"delta_{key}"] > 1e-12 for row in paired),
        "ties": sum(abs(row[f"delta_{key}"]) <= 1e-12 for row in paired),
        "losses": sum(row[f"delta_{key}"] < -1e-12 for row in paired),
        "mean_delta": metrics[1][key] - metrics[0][key],
    } for key in metric_names})
    ordered = sorted(paired, key=lambda row: (row["delta_ndcg@10"], row["query_id"]))
    cases = []
    for label, rows in (("ndcg_loss", [row for row in ordered if row["delta_ndcg@10"] < 0][:2]),
                        ("ndcg_gain", [row for row in reversed(ordered) if row["delta_ndcg@10"] > 0][:2])):
        for row in rows:
            qi = query_ids.index(row["query_id"])
            case = {"case_type": label, "query": queries[qi], **row}
            for pooling in CONFIG["pooling"]:
                case[pooling] = [{"rank": rank, "corpus_id": pool[i], "relevant": pool[i] in data["qrels"][row["query_id"]],
                                  "text": documents[i]} for rank, i in enumerate(all_rankings[pooling][qi], 1)]
            cases.append(case)
    write_jsonl(OUTPUT / "case_analysis.jsonl", cases)
    inputs = [sample_path, pool_path, old_rank_path, old_meta_path, old_vectors_path, vector_meta_path, Path(__file__),
              SNAPSHOT / "model_optimized.onnx", SNAPSHOT / "tokenizer.json", SNAPSHOT / "tokenizer_config.json", SNAPSHOT / "config.json"]
    outputs = list(VECTOR_OUTPUT.glob("*.npy")) + [VECTOR_OUTPUT / "ids.json"]
    write_json(OUTPUT / "run_metadata.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
        "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in inputs},
        "vector_output_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in outputs},
        "source_experiment": old_meta,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "onnxruntime", "tokenizers", "fastembed", "pyarrow")},
                    "providers": encoder.model.get_providers()},
        "timing_seconds": {"data_and_validation": data_seconds, "model_load": model_load_seconds,
                           "documents": doc_timing, "queries": query_timing, "search": search_timing,
                           "wall_before_metadata_write": time.perf_counter() - started},
        "timing_scope": "shared backbone forward counted once; pooling includes L2; no separate end-to-end timing claim for each variant; process startup/imports excluded",
        "cls_document_cache_max_absolute_error": max_cls_error,
        "token_lengths_including_specials": {"documents_min_max": [min(doc_lengths), max(doc_lengths)], "queries_min_max": [min(query_lengths), max(query_lengths)]},
        "checks_passed": ["original corpus/query/qrels hashes", "query/qrels/passage ID alignment", "same shared token states and attention masks",
                          "masked mean equals direct non-padding mean", "finite normalized mean vectors", "CLS document vectors match original cache within 1e-5",
                          "CLS top20 IDs exactly match original ranks", "all six CLS aggregate metrics reproduce original baseline", "20 unique output IDs per query"],
        "network_calls": 0, "training": False,
    })
    for row in metrics:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"CLS max vector error: {max_cls_error}; saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
