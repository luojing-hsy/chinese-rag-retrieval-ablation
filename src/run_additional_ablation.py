from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import bm25s
import jieba
import numpy as np

from run_t2retrieval_ablation import MODEL_CACHE, MODEL_NAME, PROCESSED_DIR, RESULTS_DIR, ROOT, TextEmbedding, evaluate, normalize_vectors, sha256_file, topk_dense
from run_pooling_ablation import SNAPSHOT, read_jsonl, write_json
from run_retrieval_extensions import prepare_data, save_evaluation


OUTPUT = ROOT / "results" / "t2retrieval" / "additional_ablation"
VECTOR_OUTPUT = PROCESSED_DIR / "additional_ablation"
CONFIG = {
    "dataset": "mteb/T2Retrieval dev derived pilot", "queries": 100, "passages": 1000, "output_k": 20,
    "bm25_variants": {"lucene": {"k1": 1.5, "b": 0.75}, "bm25l": {"k1": 1.5, "b": 0.75, "delta": 0.5},
                      "bm25+": {"k1": 1.5, "b": 0.75, "delta": 0.5}},
    "bm25_idf": "each method uses its BM25S default IDF; method comparison, not single-component ablation",
    "bm25_preprocessing": "same jieba HMM=False, lowercase, word regex and no stopwords as original A0",
    "bm25_retrieval_depth": 100,
    "model": MODEL_NAME, "model_snapshot": SNAPSHOT.name, "max_length": 512,
    "document_batch_size": 64, "query_batch_size": 256, "threads": 4, "device": "CPU",
    "pooling_ablation": "CLS vs masked Max; exclude padding with -infinity, include non-padding CLS/SEP; apply consistently to query and passage; L2 both, then dot",
    "normalization_ablation": "fixed normalized original CLS queries and same dot-product scorer; only document CLS L2 normalization enabled vs disabled",
    "normalization_warning": "not a redundant normalization toggle inside cosine; raw token CLS vectors are retained before normalization",
    "text_processing": "original clean_text and original queries; no new prefixes, chunking or query expansion",
    "parameter_policy": "fixed before first run, no evaluation-set tuning", "training": False, "network_calls": 0,
    "sources": {"bm25s": "https://github.com/xhluca/bm25s", "bge": "https://huggingface.co/BAAI/bge-small-zh-v1.5"},
}


def lexical_runs(documents, queries):
    tick = time.perf_counter()
    tokenizer = bm25s.tokenization.Tokenizer(lower=True, splitter=r"(?u)\b\w+\b", stopwords=[])
    dt = tokenizer.tokenize([" ".join(jieba.cut(text, HMM=False)) for text in documents], update_vocab="if_empty", show_progress=False)
    qt = tokenizer.tokenize([" ".join(jieba.cut(text, HMM=False)) for text in queries], update_vocab=False, show_progress=False)
    timing = {"shared_tokenization": time.perf_counter() - tick}
    runs = {}
    for method, params in CONFIG["bm25_variants"].items():
        tick = time.perf_counter()
        index = bm25s.BM25(method=method, dtype="float32", **params)
        index.index(dt, show_progress=False)
        timing[f"{method}_index"] = time.perf_counter() - tick
        tick = time.perf_counter()
        ranks, scores = index.retrieve(qt, k=100, show_progress=False)
        timing[f"{method}_search"] = time.perf_counter() - tick
        runs[{"lucene": "bm25_lucene", "bm25l": "bm25l", "bm25+": "bm25plus"}[method]] = (np.asarray(ranks)[:, :20].tolist(), np.asarray(scores)[:, :20].astype(float).tolist())
    return runs, timing


def encode_raw_and_max(encoder, texts, batch_size, label):
    raw, maxima = [], []
    forward_seconds = pooling_seconds = 0.0
    for start in range(0, len(texts), batch_size):
        tick = time.perf_counter()
        context = encoder.onnx_embed(texts[start:start + batch_size])
        forward_seconds += time.perf_counter() - tick
        hidden, mask = context.model_output, context.attention_mask.astype(bool)
        assert hidden.ndim == 3 and hidden.shape[:2] == mask.shape and hidden.shape[2] == 512
        assert mask.any(axis=1).all() and np.isfinite(hidden).all()
        tick = time.perf_counter()
        raw.append(hidden[:, 0].copy())
        pooled = normalize_vectors(np.where(mask[..., None], hidden, -np.inf).max(axis=1))
        pooling_seconds += time.perf_counter() - tick
        reference = normalize_vectors(hidden[0, mask[0]].max(axis=0)[None, :])[0]
        assert np.allclose(reference, pooled[0], atol=1e-6)
        assert np.isfinite(pooled).all() and np.allclose(np.linalg.norm(pooled, axis=1), 1, atol=1e-5)
        maxima.append(pooled)
        print(f"{label}: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(raw), np.vstack(maxima), {"shared_forward": forward_seconds, "extract_raw_and_max_pool_normalize": pooling_seconds}


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    VECTOR_OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "experiment_config.json", CONFIG)
    data, pool, qids, queries, documents, dv, qv, baseline, original = prepare_data()
    preparation = time.perf_counter() - started
    lexical, lexical_timing = lexical_runs(documents, queries)
    previous_bm25 = read_jsonl(RESULTS_DIR / "retrievals_bm25_top20.jsonl")
    assert qids == [row["query_id"] for row in previous_bm25]
    assert [[pool[i] for i in rank] for rank in lexical["bm25_lucene"][0]] == [[item["corpus_id"] for item in row["results"]] for row in previous_bm25]
    with (RESULTS_DIR / "metrics.csv").open(encoding="utf-8-sig") as stream:
        old = next(row for row in csv.DictReader(stream) if row["method"] == "bm25")
    values = evaluate(lexical["bm25_lucene"][0], pool, qids, data["qrels"])
    assert all(abs(values[name] - float(old[name])) < 1e-12 for name in ("recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "precision@10"))
    tick = time.perf_counter()
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE), specific_model_path=str(SNAPSHOT), threads=4, cuda=False, local_files_only=True)
    load_seconds = time.perf_counter() - tick
    assert model.model.tokenizer.truncation["max_length"] == 512
    dr, dm, doc_timing = encode_raw_and_max(model.model, documents, 64, "passages")
    qr, qm, query_timing = encode_raw_and_max(model.model, queries, 256, "queries")
    cls_error = float(np.abs(normalize_vectors(dr.copy()) - dv).max())
    assert np.allclose(normalize_vectors(dr.copy()), dv, atol=1e-5, rtol=1e-5)
    assert np.allclose(normalize_vectors(qr.copy()), qv, atol=1e-5, rtol=1e-5)
    norms = np.linalg.norm(dr, axis=1)
    assert np.ptp(norms) > 1e-5
    max_ranks, max_scores, max_search = topk_dense(qm, dm, 20)
    # topk_dense is exactly document_matrix @ query_vector: it does not divide by norms.
    raw_ranks, raw_scores, raw_search = topk_dense(qv, dr, 20)
    unit_ranks, _, _ = topk_dense(qv, normalize_vectors(dr.copy()), 20)
    assert unit_ranks == baseline[0]
    for name, vector in (("cls_raw_documents", dr), ("cls_raw_queries", qr), ("max_documents", dm), ("max_queries", qm)):
        np.save(VECTOR_OUTPUT / f"{name}.npy", vector)
    write_json(VECTOR_OUTPUT / "ids.json", {"passage_ids": pool, "query_ids": qids, "config": CONFIG})
    runs = {"bge_cls": baseline[:2], **lexical, "bge_max": (max_ranks, max_scores), "bge_cls_document_l2_off": (raw_ranks, raw_scores)}
    save_evaluation(OUTPUT, runs, data, pool, qids, queries, documents)
    sources = [Path(__file__), ROOT / "src/run_retrieval_extensions.py", ROOT / "src/run_t2retrieval_ablation.py",
               RESULTS_DIR / "query_sample.jsonl", RESULTS_DIR / "candidate_pool_doc_ids.txt", RESULTS_DIR / "metrics.csv",
               SNAPSHOT / "model_optimized.onnx", SNAPSHOT / "tokenizer.json", SNAPSHOT / "tokenizer_config.json"]
    write_json(OUTPUT / "run_metadata.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
        "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        "output_vector_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in VECTOR_OUTPUT.glob("*")},
        "source_experiment": original, "runtime": {"python": platform.python_version(), "platform": platform.platform(),
            "providers": model.model.model.get_providers(), "packages": {name: importlib.metadata.version(name) for name in ("numpy", "bm25s", "jieba", "fastembed", "onnxruntime")}},
        "timing_seconds": {"data_and_validation": preparation, "lexical": lexical_timing, "bge_load": load_seconds,
            "documents": doc_timing, "queries": query_timing, "max_search": max_search, "raw_document_dot_search": raw_search,
            "wall_before_metadata_write": time.perf_counter() - started},
        "timing_scope": "shared tokenization and shared BGE forward counted once; includes validation/output in wall, excludes process startup/imports",
        "raw_cls_document_norms": {"min": float(norms.min()), "max": float(norms.max()), "mean": float(norms.mean()), "std": float(norms.std())},
        "normalized_cls_max_error": cls_error,
        "checks_passed": ["original data hashes and ID/qrels alignment", "BGE baseline ranks and six metrics reproduce A1",
            "Lucene BM25 ranks and six metrics reproduce A0", "normalized fresh CLS document/query vectors match cache",
            "masked max excludes padding and equals direct valid-token max", "finite unit max vectors", "document raw norms vary",
            "normalization-on with fixed dot restores baseline", "20 unique in-pool outputs per query"],
    })
    print(f"Saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
