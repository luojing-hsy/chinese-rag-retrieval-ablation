from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata
import json
import platform
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import jieba
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from run_t2retrieval_ablation import (
    DATA_DIR, MODEL_CACHE, MODEL_NAME, PROCESSED_DIR, RESULTS_DIR, ROOT,
    TextEmbedding, evaluate, load_data, normalize_vectors, save_rankings,
    sha256_file, topk_dense, write_jsonl,
)
from run_pooling_ablation import SNAPSHOT, read_jsonl, write_csv, write_json


OUTPUT_ROOT = ROOT / "results" / "t2retrieval" / "retrieval_extensions"
VECTORS = PROCESSED_DIR / "retrieval_extensions"
E5_DIR = ROOT / "models" / "multilingual-e5-small"
COMMON = {
    "dataset": "mteb/T2Retrieval dev derived pilot", "queries": 100, "passages": 1000,
    "output_k": 20, "evaluation": "same original passage-level binary qrels; unjudged treated as non-relevant",
    "model_training": False, "inference_network_calls": 0,
    "parameter_policy": "fixed before first run; no evaluation-set tuning",
}
LOCAL_CONFIG = {
    **COMMON,
    "tfidf": {"analyzer": "jieba HMM=False followed by lowercased word-regex tokens, matching BM25 preprocessing",
              "ngram_range": [1, 1], "norm": "l2", "smooth_idf": True, "sublinear_tf": False,
              "idf_fit": "passages only; queries only transformed", "stop_words": None},
    "html_ablation": {"model": MODEL_NAME, "pooling": "CLS", "max_length": 512,
                      "only_change": "passage HTML tag replacement enabled vs disabled",
                      "query_processing": "unchanged original pilot query and cached CLS vectors",
                      "fixed_steps": "HTML unescape twice, NFKC, whitespace normalization",
                      "cache_policy": "reuse unchanged passage vectors; encode only changed passage texts",
                      "batch_size": 64, "threads": 4, "normalization": "L2", "scoring": "exact cosine over all 1000 passages"},
    "roles": {"tfidf_word": "method comparison", "bge_keep_html": "HTML-removal ablation treatment", "bge_cls": "original A1 baseline"},
}
E5_CONFIG = {
    **COMMON, "model": "intfloat/multilingual-e5-small", "weights": "official unquantized FP32 onnx/model.onnx",
    "query_prefix": "query: ", "passage_prefix": "passage: ", "max_length": 512,
    "pooling": "attention-mask mean, including non-padding special tokens", "normalization": "L2",
    "document_batch_size": 32, "query_batch_size": 32, "threads": 4, "device": "CPU",
    "scoring": "exact cosine over all 1000 passages", "text_cleaning": "original pilot clean_text",
    "role": "pretrained encoder pipeline comparison; tokenizer, prefix and pooling follow E5, not a single-factor ablation",
    "source": "https://huggingface.co/intfloat/multilingual-e5-small",
}


def prepare_data():
    data = load_data(DATA_DIR)
    sample = read_jsonl(RESULTS_DIR / "query_sample.jsonl")
    pool = (RESULTS_DIR / "candidate_pool_doc_ids.txt").read_text(encoding="utf-8").splitlines()
    qids, queries = [row["query_id"] for row in sample], [row["query"] for row in sample]
    lookup = dict(zip(data["doc_ids"], data["documents"]))
    documents = [lookup[pid] for pid in pool]
    original_meta = json.loads((RESULTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    assert len(set(qids)) == len(qids) == 100 and len(set(pool)) == len(pool) == 1000
    for row in sample:
        assert row["query"] == data["query_by_id"][row["query_id"]]
        assert set(row["relevant_ids"]) == data["qrels"][row["query_id"]] <= set(pool)
    for path, key in ((data["corpus_path"], "corpus_file_sha256"), (data["queries_path"], "query_file_sha256"), (data["qrels_path"], "qrels_file_sha256")):
        assert sha256_file(path) == original_meta[key]
    cached = PROCESSED_DIR / "pooling_ablation"
    ids = json.loads((cached / "ids.json").read_text(encoding="utf-8"))
    assert ids["passage_ids"] == pool and ids["query_ids"] == qids
    doc_vectors, query_vectors = np.load(cached / "cls_documents.npy"), np.load(cached / "cls_queries.npy")
    assert doc_vectors.shape == (1000, 512) and query_vectors.shape == (100, 512)
    previous = json.loads((ROOT / "results/t2retrieval/pooling_ablation/run_metadata.json").read_text(encoding="utf-8"))
    for name in ("cls_documents.npy", "cls_queries.npy", "ids.json"):
        path = cached / name
        assert sha256_file(path) == previous["vector_output_sha256"][str(path.relative_to(ROOT))]
    ranks, scores, seconds = topk_dense(query_vectors, doc_vectors, 20)
    saved = read_jsonl(RESULTS_DIR / "retrievals_dense_bge_small_zh_top20.jsonl")
    assert qids == [row["query_id"] for row in saved]
    assert [[pool[i] for i in rank] for rank in ranks] == [[item["corpus_id"] for item in row["results"]] for row in saved]
    return data, pool, qids, queries, documents, doc_vectors, query_vectors, (ranks, scores, seconds), original_meta


def keep_html(text):
    value = html.unescape(html.unescape(text or ""))
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


def run_local(pool, queries, documents, doc_vectors, query_vectors, output):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import save_npz

    timing = {}
    tick = time.perf_counter()
    def segment(text):
        return " ".join(re.findall(r"(?u)\b\w+\b", " ".join(jieba.cut(text, HMM=False)).lower()))
    segmented_documents, segmented_queries = [segment(text) for text in documents], [segment(text) for text in queries]
    timing["tfidf_segmentation"] = time.perf_counter() - tick
    tick = time.perf_counter()
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, norm="l2", smooth_idf=True, sublinear_tf=False)
    matrix = vectorizer.fit_transform(segmented_documents)
    timing["tfidf_document_index"] = time.perf_counter() - tick
    tick = time.perf_counter()
    query_matrix = vectorizer.transform(segmented_queries)
    similarity = (query_matrix @ matrix.T).toarray()
    tfidf_ranks = np.argsort(-similarity, axis=1, kind="stable")[:, :20].tolist()
    tfidf_scores = [[float(similarity[qi, i]) for i in row] for qi, row in enumerate(tfidf_ranks)]
    timing["tfidf_query_transform_and_search"] = time.perf_counter() - tick
    assert len(vectorizer.vocabulary_) == matrix.shape[1] and matrix.shape[0] == 1000
    write_json(output / "tfidf_vocabulary.json", {word: int(index) for word, index in vectorizer.vocabulary_.items()})
    np.save(VECTORS / "tfidf_idf.npy", vectorizer.idf_)
    save_npz(VECTORS / "tfidf_documents.npz", matrix)
    save_npz(VECTORS / "tfidf_queries.npz", query_matrix)
    raw_rows = pq.read_table(DATA_DIR / "mteb-corpus-dev.parquet", columns=["_id", "text"], filters=[("_id", "in", pool)]).to_pylist()
    raw = {row["_id"]: row["text"] for row in raw_rows}
    kept = [keep_html(raw[pid]) or " " for pid in pool]
    changed = [i for i, (clean, keep) in enumerate(zip(documents, kept)) if clean != keep]
    assert set(raw) == set(pool) and changed
    write_jsonl(output / "passage_preprocessing.jsonl", [{"corpus_id": pid, "tags_removed": clean, "tags_kept": keep, "changed": clean != keep}
                                                          for pid, clean, keep in zip(pool, documents, kept)])
    changed_vectors = doc_vectors.copy()
    tick = time.perf_counter()
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE), specific_model_path=str(SNAPSHOT),
                          threads=4, cuda=False, local_files_only=True)
    timing["html_bge_model_load"] = time.perf_counter() - tick
    tick = time.perf_counter()
    count = 0
    for vector, position in zip(model.embed([kept[i] for i in changed], batch_size=64), changed):
        changed_vectors[position] = normalize_vectors(np.asarray(vector)[None, :])[0]
        count += 1
        if count % 128 == 0:
            print(f"HTML treatment: {count}/{len(changed)} passages encoded", flush=True)
    assert count == len(changed) and np.isfinite(changed_vectors).all()
    assert np.allclose(np.linalg.norm(changed_vectors, axis=1), 1, atol=1e-5)
    unchanged = [i for i in range(len(pool)) if i not in set(changed)]
    assert np.array_equal(changed_vectors[unchanged], doc_vectors[unchanged])
    timing["html_changed_document_encoding"] = time.perf_counter() - tick
    html_ranks, html_scores, timing["html_full_pool_search"] = topk_dense(query_vectors, changed_vectors, 20)
    np.save(VECTORS / "bge_keep_html_documents.npy", changed_vectors)
    extra = {"changed_passages": len(changed), "unchanged_passages_reused": len(unchanged),
             "raw_passages_matching_tag_regex": sum(bool(re.search(r"<[^>]*>", raw[pid] or "")) for pid in pool),
             "tfidf_vocabulary_size": len(vectorizer.vocabulary_), "tfidf_nonzero_entries": int(matrix.nnz)}
    return {"tfidf_word": (tfidf_ranks, tfidf_scores), "bge_keep_html": (html_ranks, html_scores)}, timing, extra


def encode_e5(session, tokenizer, texts, batch_size, prefix, label):
    vectors, lengths = [], []
    started = time.perf_counter()
    names = {node.name for node in session.get_inputs()}
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer.encode_batch([prefix + text for text in texts[start:start + batch_size]])
        ids = np.array([row.ids for row in encoded], dtype=np.int64)
        mask = np.array([row.attention_mask for row in encoded], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = session.run(None, feed)[0]
        assert hidden.shape == (*ids.shape, 384) and np.isfinite(hidden).all()
        attention = mask.astype(np.float32)[..., None]
        pooled = normalize_vectors((hidden * attention).sum(axis=1) / attention.sum(axis=1))
        reference = normalize_vectors(hidden[0, mask[0].astype(bool)].mean(axis=0)[None, :])[0]
        assert np.allclose(pooled[0], reference, atol=1e-5)
        vectors.append(pooled)
        lengths.extend(mask.sum(axis=1).tolist())
        print(f"E5 {label}: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(vectors), time.perf_counter() - started, [min(lengths), max(lengths)]


def run_e5(queries, documents):
    provenance = json.loads((E5_DIR / "model_provenance.json").read_text(encoding="utf-8"))
    assert "download_completed_utc" in provenance
    for name, record in provenance["files"].items():
        assert sha256_file(E5_DIR / name) == record["sha256"]
    tick = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(E5_DIR / "onnx/model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(E5_DIR / "tokenizer.json"))
    token_config = json.loads((E5_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
    pad_token = token_config["pad_token"]
    if isinstance(pad_token, dict):
        pad_token = pad_token["content"]
    tokenizer.enable_truncation(max_length=512, direction="right")
    tokenizer.enable_padding(direction="right", pad_id=tokenizer.token_to_id(pad_token), pad_token=pad_token)
    timing = {"model_and_tokenizer_load": time.perf_counter() - tick}
    dv, timing["document_encoding"], dl = encode_e5(session, tokenizer, documents, 32, "passage: ", "passages")
    qv, timing["query_encoding"], ql = encode_e5(session, tokenizer, queries, 32, "query: ", "queries")
    assert np.isfinite(dv).all() and np.isfinite(qv).all()
    assert np.allclose(np.linalg.norm(dv, axis=1), 1, atol=1e-5) and np.allclose(np.linalg.norm(qv, axis=1), 1, atol=1e-5)
    ranks, scores, timing["full_pool_search"] = topk_dense(qv, dv, 20)
    np.save(VECTORS / "e5_documents.npy", dv)
    np.save(VECTORS / "e5_queries.npy", qv)
    return {"e5_small": (ranks, scores)}, timing, {"model_provenance": provenance, "providers": session.get_providers(),
            "input_names": [node.name for node in session.get_inputs()], "output_names": [node.name for node in session.get_outputs()],
            "valid_token_lengths": {"documents": dl, "queries": ql}, "padding_token": pad_token, "padding_id": tokenizer.token_to_id(pad_token)}


def save_evaluation(output, runs, data, pool, qids, queries, documents):
    metrics, per_query, paired, cases = [], [], [], []
    metric_names = ("recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "precision@10")
    for method, (ranks, scores) in runs.items():
        assert len(ranks) == len(qids) and all(len(row) == len(set(row)) == 20 and set(row) <= set(range(1000)) for row in ranks)
        summary = evaluate(ranks, pool, qids, data["qrels"])
        summary.pop("recall@50")
        metrics.append({"method": method, "queries": 100, "passages": 1000, "top_k": 20, **summary})
        save_rankings(output / f"retrievals_{method}.jsonl", method, qids, queries, ranks, scores, pool)
        for qid, rank in zip(qids, ranks):
            values = evaluate([rank], pool, [qid], data["qrels"])
            values.pop("recall@50")
            per_query.append({"query_id": qid, "method": method, **values})
    baseline = {row["query_id"]: row for row in per_query if row["method"] == "bge_cls"}
    with (RESULTS_DIR / "metrics.csv").open(encoding="utf-8-sig") as stream:
        old = next(row for row in csv.DictReader(stream) if row["method"] == "dense_bge_small_zh")
    assert all(abs(metrics[0][name] - float(old[name])) < 1e-12 for name in metric_names)
    for method in runs:
        if method == "bge_cls":
            continue
        rows = [{"query_id": row["query_id"], "method": method, **{f"delta_{key}": row[key] - baseline[row["query_id"]][key] for key in metric_names}}
                for row in per_query if row["method"] == method]
        paired.extend(rows)
        rows.sort(key=lambda row: (row["delta_ndcg@10"], row["query_id"]))
        for label, chosen in (("ndcg_loss", [r for r in rows if r["delta_ndcg@10"] < 0][:2]), ("ndcg_gain", [r for r in reversed(rows) if r["delta_ndcg@10"] > 0][:2])):
            for row in chosen:
                qi = qids.index(row["query_id"])
                case = {"case_type": label, "query": queries[qi], **row}
                for name in ("bge_cls", method):
                    case[name] = [{"rank": rank, "corpus_id": pool[i], "relevant": pool[i] in data["qrels"][row["query_id"]], "text": documents[i]}
                                  for rank, i in enumerate(runs[name][0][qi], 1)]
                cases.append(case)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "per_query_metrics.csv", per_query)
    write_csv(output / "paired_deltas.csv", paired)
    write_jsonl(output / "case_analysis.jsonl", cases)
    write_json(output / "paired_summary.json", {method: {key: {
        "wins": sum(row[f"delta_{key}"] > 1e-12 for row in paired if row["method"] == method),
        "ties": sum(abs(row[f"delta_{key}"]) <= 1e-12 for row in paired if row["method"] == method),
        "losses": sum(row[f"delta_{key}"] < -1e-12 for row in paired if row["method"] == method),
    } for key in metric_names} for method in runs if method != "bge_cls"})
    for row in metrics:
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("local", "e5", "all"), default="all")
    args = parser.parse_args()
    started = time.perf_counter()
    prepared = prepare_data()
    data, pool, qids, queries, documents, dv, qv, baseline, source_meta = prepared
    data_seconds = time.perf_counter() - started
    VECTORS.mkdir(parents=True, exist_ok=True)
    write_json(VECTORS / "ids.json", {"passage_ids": pool, "query_ids": qids})
    for stage in (("local", "e5") if args.stage == "all" else (args.stage,)):
        tick = time.perf_counter()
        output = OUTPUT_ROOT / stage
        output.mkdir(parents=True, exist_ok=True)
        config = LOCAL_CONFIG if stage == "local" else E5_CONFIG
        write_json(output / "experiment_config.json", config)
        if stage == "local":
            treatments, timing, details = run_local(pool, queries, documents, dv, qv, output)
        else:
            treatments, timing, details = run_e5(queries, documents)
        runs = {"bge_cls": baseline[:2], **treatments}
        save_evaluation(output, runs, data, pool, qids, queries, documents)
        sources = [Path(__file__), ROOT / "src/run_t2retrieval_ablation.py", RESULTS_DIR / "query_sample.jsonl",
                   RESULTS_DIR / "candidate_pool_doc_ids.txt", RESULTS_DIR / "metrics.csv",
                   PROCESSED_DIR / "pooling_ablation/cls_documents.npy", PROCESSED_DIR / "pooling_ablation/cls_queries.npy"]
        write_json(output / "run_metadata.json", {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": config, "details": details,
            "source_experiment": source_meta, "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
            "runtime": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "onnxruntime", "tokenizers", "fastembed", "pyarrow", "scikit-learn", "scipy")}},
            "timing_seconds": {"data_load_and_validation_shared": data_seconds, "baseline_cached_search_shared": baseline[2], **timing,
                               "stage_wall_before_metadata_write": time.perf_counter() - tick},
            "timing_scope": "stage excludes model download, process startup/imports; shared data and cached baseline search listed separately",
            "checks_passed": ["raw data hashes and qrels match original", "same 100 query and 1000 passage IDs", "cached CLS vector hashes and IDs match pooling run",
                              "baseline top20 IDs and six metrics exactly reproduce A1", "20 unique in-pool outputs per query", "local inference only; no training"],
        })
        print(f"Saved {stage} experiment to {output}", flush=True)


if __name__ == "__main__":
    main()
