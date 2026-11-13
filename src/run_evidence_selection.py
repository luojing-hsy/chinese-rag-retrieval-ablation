from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "t2retrieval" / "pilot"
PROCESSED = ROOT / "data" / "processed" / "t2retrieval" / "pilot"
OUTPUT = ROOT / "results" / "t2retrieval" / "evidence_selection_top50"
CONFIG = {
    "dataset": "mteb/T2Retrieval dev derived pilot",
    "parent_pool_size": 1000,
    "queries": 100,
    "candidate_k": 50,
    "output_k": 20,
    "model": "BAAI/bge-small-zh-v1.5",
    "model_snapshot": "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59",
    "query_processing": "reuse original saved BGE query cosine scores; no query modification",
    "mmr_gamma": 0.25,
    "dartboard_sigma": 0.1,
    "parameter_policy": "fixed before first run; exploratory settings, no evaluation-set tuning",
    "mmr_formula": "query_cosine - gamma * max cosine to selected; first = highest query cosine",
    "dartboard_formula": "scaled distance=(1-cosine)/2; Gaussian log kernel=-distance^2/(2*sigma^2); greedy logsumexp of query kernel + max selected document kernel",
    "dartboard_initial_choice": "highest query cosine, following paper algorithm",
    "tie_break": "first in original BGE candidate order",
    "controlled_ablation": "dense_top20 vs mmr_gamma_025: only redundancy penalty is enabled",
    "mechanism_comparison": "dartboard_sigma_01 vs dense_top20/MMR; not a single-component ablation",
    "unjudged_policy": "treated as non-relevant; not confirmed negatives",
    "sources": {
        "mmr": "https://doi.org/10.1145/290941.291025",
        "dartboard": "https://arxiv.org/html/2407.12101v2",
        "dartboard_reference": "https://github.com/EmergenceAI/dartboard/blob/main/dartboard.py",
    },
    "implementation": "local NumPy implementation from published equations; no third-party source copied or imported",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mmr_order(relevance, similarities, gamma, k):
    chosen = [int(np.argmax(relevance))]
    redundancy = similarities[chosen[0]].copy()
    while len(chosen) < k:
        scores = relevance - gamma * redundancy
        scores[chosen] = -np.inf
        selected = int(np.argmax(scores))
        chosen.append(selected)
        redundancy = np.maximum(redundancy, similarities[selected])
    return chosen


def dartboard_order(relevance, similarities, sigma, k):
    query_log_kernel = -0.5 * (np.clip((1.0 - relevance) / 2.0, 0, 1) / sigma) ** 2
    document_log_kernel = -0.5 * (np.clip((1.0 - similarities) / 2.0, 0, 1) / sigma) ** 2
    # Gaussian normalization constants are common to all choices and cancel.
    chosen = [int(np.argmax(query_log_kernel))]
    covered = document_log_kernel[chosen[0]].copy()
    while len(chosen) < k:
        possible_coverage = np.maximum(document_log_kernel, covered[None, :])
        objectives = np.logaddexp.reduce(possible_coverage + query_log_kernel[None, :], axis=1)
        objectives[chosen] = -np.inf
        selected = int(np.argmax(objectives))
        chosen.append(selected)
        covered = possible_coverage[selected]
    return chosen


def evaluate(pids, relevant, similarities, order):
    hits = [int(pid in relevant) for pid in pids]
    first = next((rank for rank, hit in enumerate(hits[:10], 1) if hit), None)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    top10_sim = similarities[np.ix_(order[:10], order[:10])]
    return {
        **{f"recall@{k}": sum(hits[:k]) / len(relevant) for k in (5, 10, 20)},
        "mrr@10": 1 / first if first else 0.0,
        "ndcg@10": sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:10], 1)) / ideal,
        "precision@10": sum(hits[:10]) / 10,
        "mean_pair_cosine@10": float(top10_sim[np.triu_indices(10, 1)].mean()),
    }


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "experiment_config.json", CONFIG)
    loading_started = time.perf_counter()
    sample_path = PILOT / "query_sample.jsonl"
    ranking_path = PILOT / "retrievals_dense_bge_small_zh.jsonl"
    previous_top20_path = PILOT / "retrievals_dense_bge_small_zh_top20.jsonl"
    pool_path = PILOT / "candidate_pool_doc_ids.txt"
    vector_path = PROCESSED / "bge-small-zh-v1.5-document-vectors.npy"
    vector_metadata_path = PROCESSED / "bge-small-zh-v1.5-document-vectors.json"
    source_metadata_path = PILOT / "run_metadata.json"
    sample, rankings = read_jsonl(sample_path), read_jsonl(ranking_path)
    pool = pool_path.read_text(encoding="utf-8").splitlines()
    vector_metadata = json.loads(vector_metadata_path.read_text(encoding="utf-8"))
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    vectors = np.load(vector_path).astype(np.float64)
    assert len(sample) == len(rankings) == CONFIG["queries"]
    assert len(set(pool)) == len(pool) == CONFIG["parent_pool_size"]
    assert [row["query_id"] for row in sample] == [row["query_id"] for row in rankings]
    previous_top20 = read_jsonl(previous_top20_path)
    assert [row["query_id"] for row in sample] == [row["query_id"] for row in previous_top20]
    assert all([item["corpus_id"] for item in full["results"][:20]] ==
               [item["corpus_id"] for item in saved["results"]] for full, saved in zip(rankings, previous_top20))
    assert len({row["query_id"] for row in sample}) == len(sample)
    assert vectors.shape == (len(pool), 512) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest() == vector_metadata["candidate_doc_ids_sha256"]
    assert vector_metadata["model"] == source_metadata["model"] == CONFIG["model"]
    assert vector_metadata["corpus_sha256"] == source_metadata["corpus_file_sha256"]
    raw = ROOT / "data" / "raw" / "t2retrieval"
    raw_paths = [raw / f"mteb-{part}-dev.parquet" for part in ("corpus", "queries", "qrels")]
    for path, key in zip(raw_paths, ("corpus_file_sha256", "query_file_sha256", "qrels_file_sha256")):
        assert sha256(path) == source_metadata[key]
    qrels = {}
    for row in pq.read_table(raw_paths[2]).to_pylist():
        assert row["score"] == 1
        qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    pool_set = set(pool)
    for row in sample:
        assert set(row["relevant_ids"]) == qrels[row["query_id"]]
        assert qrels[row["query_id"]] and qrels[row["query_id"]] <= pool_set
    lookup = {pid: i for i, pid in enumerate(pool)}
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    loading_seconds = time.perf_counter() - loading_started

    methods = ("dense_top20", "mmr_gamma_025", "dartboard_sigma_01")
    elapsed = dict.fromkeys(methods, 0.0)
    outputs = {method: [] for method in methods}
    per_query, candidates = [], []
    similarity_seconds = evaluation_seconds = validation_seconds = 0.0
    for sample_row, ranking in zip(sample, rankings):
        qid, query = sample_row["query_id"], sample_row["query"]
        assert query == ranking["query"] and len(ranking["results"]) >= CONFIG["candidate_k"]
        candidate = ranking["results"][:CONFIG["candidate_k"]]
        ids = [item["corpus_id"] for item in candidate]
        relevance = np.array([item["score"] for item in candidate], dtype=np.float64)
        assert len(set(ids)) == len(ids) and set(ids) <= pool_set
        assert np.isfinite(relevance).all() and np.all(relevance[:-1] >= relevance[1:])
        tick = time.perf_counter()
        local_vectors = vectors[[lookup[pid] for pid in ids]]
        similarities = np.clip(local_vectors @ local_vectors.T, -1, 1)
        similarity_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        baseline = list(range(CONFIG["output_k"]))
        elapsed[methods[0]] += time.perf_counter() - tick
        tick = time.perf_counter()
        mmr = mmr_order(relevance, similarities, CONFIG["mmr_gamma"], CONFIG["output_k"])
        elapsed[methods[1]] += time.perf_counter() - tick
        tick = time.perf_counter()
        dartboard = dartboard_order(relevance, similarities, CONFIG["dartboard_sigma"], CONFIG["output_k"])
        elapsed[methods[2]] += time.perf_counter() - tick
        tick = time.perf_counter()
        assert mmr_order(relevance, similarities, 0, CONFIG["output_k"]) == baseline
        validation_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        relevant = qrels[qid]
        ceiling = len(set(ids) & relevant) / len(relevant)
        candidates.append({"query_id": qid, "query": query, "results": candidate, "recall@50": ceiling})
        for method, order in zip(methods, (baseline, mmr, dartboard)):
            assert len(order) == len(set(order)) == CONFIG["output_k"]
            assert set(order) <= set(range(CONFIG["candidate_k"])) and order[0] == 0
            pids = [ids[i] for i in order]
            metrics = evaluate(pids, relevant, similarities, order)
            assert metrics["recall@20"] <= ceiling + 1e-12
            per_query.append({"query_id": qid, "method": method, "candidate_recall@50": ceiling, **metrics})
            outputs[method].append({"query_id": qid, "query": query, "method": method, "results": [
                {"corpus_id": ids[i], "rank": rank, "original_rank": i + 1, "query_cosine": float(relevance[i])}
                for rank, i in enumerate(order, 1)
            ]})
        evaluation_seconds += time.perf_counter() - tick

    metric_keys = list(per_query[0])[2:]
    summaries = []
    for method in methods:
        rows = [row for row in per_query if row["method"] == method]
        summaries.append({"method": method, "queries": len(rows), "candidate_k": 50, "output_k": 20,
                          "selection_seconds_total": elapsed[method], "selection_ms_mean": elapsed[method] * 1000 / len(rows),
                          **{key: float(np.mean([row[key] for row in rows])) for key in metric_keys}})
    with (PILOT / "metrics.csv").open(encoding="utf-8-sig") as stream:
        old_dense = next(row for row in csv.DictReader(stream) if row["method"] == "dense_bge_small_zh")
    for key in ("recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "precision@10"):
        assert math.isclose(summaries[0][key], float(old_dense[key]), abs_tol=1e-12)
    paired = []
    for method in methods[1:]:
        for index, sample_row in enumerate(sample):
            baseline, treatment = per_query[index * 3], per_query[index * 3 + methods.index(method)]
            paired.append({"query_id": sample_row["query_id"], "method": method,
                           **{f"delta_{key}": treatment[key] - baseline[key] for key in metric_keys}})

    write_jsonl(OUTPUT / "candidates_top50.jsonl", candidates)
    for method in methods:
        write_jsonl(OUTPUT / f"retrievals_{method}.jsonl", outputs[method])
    write_csv(OUTPUT / "metrics.csv", summaries)
    write_csv(OUTPUT / "per_query_metrics.csv", per_query)
    write_csv(OUTPUT / "paired_deltas.csv", paired)
    cases = []
    texts = {row["_id"]: row["text"] for row in pq.read_table(raw_paths[0], columns=["_id", "text"]).to_pylist() if row["_id"] in pool_set}
    for method in methods[1:]:
        method_deltas = sorted([row for row in paired if row["method"] == method], key=lambda row: (row["delta_ndcg@10"], row["query_id"]))
        for label, selected in (("largest_ndcg_loss", [row for row in method_deltas if row["delta_ndcg@10"] < 0][:2]),
                                ("largest_ndcg_gain", [row for row in reversed(method_deltas) if row["delta_ndcg@10"] > 0][:1])):
            for delta in selected:
                index = next(i for i, row in enumerate(sample) if row["query_id"] == delta["query_id"])
                case = {"case_type": label, "query": sample[index]["query"], **delta}
                for name in ("dense_top20", method):
                    case[name] = [{**item, "relevant": item["corpus_id"] in qrels[delta["query_id"]], "text": texts[item["corpus_id"]]}
                                  for item in outputs[name][index]["results"]]
                cases.append(case)
    write_jsonl(OUTPUT / "case_analysis.jsonl", cases)
    inputs = [sample_path, ranking_path, previous_top20_path, pool_path, vector_path, vector_metadata_path, source_metadata_path, PILOT / "metrics.csv", Path(__file__)]
    write_json(OUTPUT / "run_metadata.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "source_experiment": source_metadata,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pyarrow.__version__,
                    "platform": platform.platform(), "processor": platform.processor(), "device": "CPU"},
        "timing_seconds": {"load_and_validate_inputs": loading_seconds, "candidate_similarity_shared": similarity_seconds,
                           "selection_by_method": elapsed, "mmr_zero_validation": validation_seconds,
                           "metric_evaluation": evaluation_seconds, "wall_before_metadata_write": time.perf_counter() - started},
        "timing_scope": "cache-based postprocessing only; excludes original embedding/retrieval, process startup/imports; includes separate shared similarity timing",
        "model_inference_calls": 0, "network_calls": 0,
        "checks_passed": ["source file hashes and qrels alignment", "vector ID mapping, dimensions and normalization", "same frozen candidates",
                          "MMR gamma=0 exactly reproduces original top20", "baseline IDs equal saved original top20", "baseline matches six original aggregate metrics",
                          "unique candidate-only outputs", "first result invariant", "recall20 bounded by candidate recall50"],
    })
    for row in summaries:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"Saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
