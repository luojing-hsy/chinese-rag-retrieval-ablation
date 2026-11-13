import csv
import hashlib
import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/t2retrieval/current_summary"
OUT.mkdir(parents=True, exist_ok=True)
PILOT = ROOT / "results/t2retrieval/pilot"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


bindings = [
    ("A0", "对比", "BM25", "pilot/retrievals_bm25_top20.jsonl", "pilot/metrics.csv", "bm25"),
    ("A1", "当前方案", "BGE-CLS", "pilot/retrievals_dense_bge_small_zh_top20.jsonl", "pilot/metrics.csv", "dense_bge_small_zh"),
    ("A2", "对比", "BM25+BGE+RRF", "pilot/retrievals_hybrid_rrf_top20.jsonl", "pilot/metrics.csv", "hybrid_rrf"),
    ("A3", "对比", "BM25+BGE重排", "pilot/retrievals_bm25_top20_bge_rerank.jsonl", "pilot/metrics.csv", "bm25_top20_bge_rerank"),
    ("A5", "对比", "词级TF-IDF", "retrieval_extensions/local/retrievals_tfidf_word.jsonl", "retrieval_extensions/local/metrics.csv", "tfidf_word"),
    ("A7", "对比", "BM25L", "additional_ablation/retrievals_bm25l.jsonl", "additional_ablation/metrics.csv", "bm25l"),
    ("A8", "对比", "BM25+", "additional_ablation/retrievals_bm25plus.jsonl", "additional_ablation/metrics.csv", "bm25plus"),
    ("P-Mean", "池化消融", "BGE-Mean", "pooling_ablation/retrievals_bge_mean.jsonl", "pooling_ablation/metrics.csv", "bge_mean"),
    ("P-Max", "池化消融", "BGE-Max", "additional_ablation/retrievals_bge_max.jsonl", "additional_ablation/metrics.csv", "bge_max"),
    ("N-Off", "文档L2消融", "BGE-CLS文档L2关", "additional_ablation/retrievals_bge_cls_document_l2_off.jsonl", "additional_ablation/metrics.csv", "bge_cls_document_l2_off"),
]
sample_path = PILOT / "query_sample.jsonl"
sample = read_jsonl(sample_path)
qids = [row["query_id"] for row in sample]
qrels = {row["query_id"]: set(row["relevant_ids"]) for row in sample}
sources = [sample_path]
rows = []
for identifier, role, label, ranking_file, metric_file, method in bindings:
    path = ROOT / "results/t2retrieval" / ranking_file
    rankings = read_jsonl(path)
    assert [row["query_id"] for row in rankings] == qids
    values = {f"{metric}@{k}": [] for metric in ("recall", "mrr", "ndcg", "precision") for k in (5, 10, 20)}
    for row in rankings:
        ids = [item["corpus_id"] for item in row["results"]]
        assert len(ids) == len(set(ids)) == 20
        relevant = qrels[row["query_id"]]
        hits = [int(pid in relevant) for pid in ids]
        for k in (5, 10, 20):
            first = next((rank for rank, hit in enumerate(hits[:k], 1) if hit), None)
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
            values[f"recall@{k}"].append(sum(hits[:k]) / len(relevant))
            values[f"precision@{k}"].append(sum(hits[:k]) / k)
            values[f"mrr@{k}"].append(1 / first if first else 0.0)
            values[f"ndcg@{k}"].append(sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:k], 1)) / ideal)
    result = {"id": identifier, "role": role, "method": label, **{key: sum(value) / len(value) for key, value in values.items()}}
    metrics_path = ROOT / "results/t2retrieval" / metric_file
    with metrics_path.open(encoding="utf-8-sig") as stream:
        old = next(row for row in csv.DictReader(stream) if row["method"] == method)
    for key in ("recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "precision@10"):
        assert abs(result[key] - float(old[key])) < 1e-12
    rows.append(result)
    sources.extend([path, metrics_path])
with (OUT / "metrics_all_k.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
comparison_order = ["A5", "A0", "A7", "A8", "A2", "A3", "A1"]
groups = [
    ("方法对比", "comparison_metrics.csv", [next(row for row in rows if row["id"] == identifier) for identifier in comparison_order]),
    ("池化消融", "pooling_ablation_metrics.csv", [row for row in rows if row["id"] in ("A1", "P-Mean", "P-Max")]),
    ("文档L2归一化消融", "normalization_ablation_metrics.csv", [row for row in rows if row["id"] in ("A1", "N-Off")]),
]
display_metrics = ["recall@5", "recall@10", "recall@20", "precision@5", "precision@10", "precision@20", "mrr@5", "ndcg@5"]
for title, filename, selected in groups:
    with (OUT / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "role", "method", *display_metrics], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

base = json.loads((PILOT / "run_metadata.json").read_text(encoding="utf-8"))
pooling = json.loads((ROOT / "results/t2retrieval/pooling_ablation/experiment_config.json").read_text(encoding="utf-8"))
extra = json.loads((ROOT / "results/t2retrieval/additional_ablation/experiment_config.json").read_text(encoding="utf-8"))
audit = json.loads((ROOT / "results/t2retrieval/rrf_depth_audit.json").read_text(encoding="utf-8"))
assert audit["100"]["matching_queries"] == 100
parameters = {
    "display_metrics": display_metrics,
    "display_policy": {"comparison_order": comparison_order, "best": "bold", "second_distinct_best": "underline", "ties": "all marked", "tie_tolerance": 1e-12},
    "scope": "Current formal methods only: A0/A1/A2/A3/A5/A7/A8; pooling CLS/Mean/Max and document L2. Archived variants excluded.",
    "common": {"dataset": "mteb/T2Retrieval dev derived pilot", "queries": 100, "passages": 1000,
               "query_seed": 2026, "candidate_seed": 2027, "relevant_union_passages": 512, "random_unjudged_passages": 488,
               "output_k": 20, "qrels": "binary score=1; unjudged treated as non-relevant; macro average over the same 100 queries",
               "cleaning": "HTML unescape, replace <[^>]*> with space, unescape again, NFKC, collapse whitespace; empty passage becomes single space",
               "query_instruction": None, "training": False, "external_model_calls": False, "device": "CPU"},
    "bge_baseline_defaults": {**{key: pooling[key] for key in ("model", "model_snapshot", "max_length", "truncation", "document_batch_size", "query_batch_size", "threads", "device", "normalization", "query_instruction")},
                              "embedding_dimension": 512, "encoder_layers": 4, "attention_heads": 8,
                              "exception": "N-Off disables document L2 only; other defaults stay fixed"},
    "methods": {
        "A0": {"implementation": "bm25s==0.3.16", "method": "lucene", "idf_method": "lucene", "k1": 1.5, "b": 0.75,
               "delta": "not used", "dtype": "float32", "retrieval_depth": 100, "output_k": 20},
        "A1": {"model": base["model"], "pooling": "CLS", "query_l2": True, "document_l2": True,
               "score": "unit query dot unit passage = cosine", "exact_search_pool": 1000, "output_k": 20},
        "A2": {"inputs": ["A0", "A1"], "input_depth_each": 100, "rrf_constant": 60, "weights": [1, 1],
               "formula": "sum(1/(60+rank)), rank starts at 1; absent item contributes 0", "output_k": 20,
               "audit": "recomputed depth100 matches all100 saved rankings; depth50 matches9/100; saved depth50 is not fusion input depth"},
        "A3": {"initial_candidates": "same original BM25 top20 IDs", "reranker": "BAAI/bge-reranker-base",
               "snapshot": "2cfc18c9415c912f9d8155881c133215df768a70", "max_pair_tokens": 512, "truncation": "right, default longest_first",
               "batch_size": 32, "threads": 4, "device": "CPU", "scoring": "first ONNX output column; descending raw scores, no sigmoid",
               "tie_break": "original BM25 order", "output_k": 20},
        "A5": {"implementation": "scikit-learn==1.9.1 TfidfVectorizer", "ngram_range": [1, 1], "tokenizer": "str.split after shared jieba/regex preprocessing",
               "token_pattern": None, "lowercase": False, "stop_words": None, "use_idf": True, "smooth_idf": True,
               "sublinear_tf": False, "binary": False, "norm": "l2", "min_df": 1, "max_df": 1.0, "max_features": None,
               "dtype": "float64", "fit": "passages only; queries transformed only", "vocabulary_size": 37877,
               "scoring": "L2 TF-IDF dot product", "tie_break": "stable original pool order", "output_k": 20},
        "A7": {"implementation": "bm25s==0.3.16", "method": "bm25l", "idf_method": "bm25l", "k1": 1.5, "b": 0.75, "delta": 0.5,
               "dtype": "float32", "retrieval_depth": 100, "output_k": 20},
        "A8": {"implementation": "bm25s==0.3.16", "method": "bm25+", "idf_method": "bm25+", "k1": 1.5, "b": 0.75, "delta": 0.5,
               "dtype": "float32", "retrieval_depth": 100, "output_k": 20},
    },
    "lexical_preprocessing": {"jieba_version": "0.42.1", "HMM": False, "lowercase": True, "splitter": r"(?u)\b\w+\b", "stopwords": []},
    "ablation": {
        "pooling": {"control": "A1", "P-Mean": "attention-mask mean; exclude padding, include valid CLS/SEP; L2 query and passage",
                    "P-Max": "dimensionwise token max; padding=-infinity, valid CLS/SEP included; L2 query and passage",
                    "fixed": "model weights, tokenizer, queries/passages, cleaning, max length512, normalization and cosine scoring"},
        "document_l2": {"control": "A1 = q_unit @ d_unit", "N-Off": "same original unit CLS query @ raw unnormalized CLS passage",
                        "fixed": "same dot-product operator, CLS pooling, original token states, query vectors and all other settings",
                        "raw_norm_min": 5.032103061676025, "raw_norm_max": 8.003568649291992},
    },
    "environment": (ROOT / "requirements-experiment.txt").read_text(encoding="utf-8").splitlines(),
    "source_file_hashes": {"corpus": base["corpus_file_sha256"], "queries": base["query_file_sha256"], "qrels": base["qrels_file_sha256"]},
}
(OUT / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
lines = ["# 当前正式实验：完整参数与 @5/@10/@20 指标", "", "日期：2026-11-13。指标仅从已保存Top-20排名重新统计；另做无神经模型推理的RRF输入深度审计，没有重新编码。", "",
         "正式方法为A0/A1/A2/A3/A5/A7/A8，消融为CLS/Mean/Max和文档L2；A1是两组消融的共同对照。已移除的A4、E5、HTML及其他归档探索不纳入。", "",
         "## 公共设置", "", "100个问题、1,000段语料，query seed=2026、候选补样seed=2027；512个相关passage并集加488个随机未标注项；统一输出20段。只评检索，不训练、不调用外部模型API。qrels为二值相关性，未标注按非相关，指标按100个query宏平均。", "",
         "文本预处理为HTML实体解码→标签替换为空格→再次解码→NFKC→空白规整；无查询指令、无切块。词法分词为jieba HMM=False、转小写、word正则过滤、无停用词。", "",
         "BGE固定BAAI/bge-small-zh-v1.5，snapshot 46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59，512维、最大512 token、右截断，CPU4线程，文档batch64/query batch256。重排snapshot为2cfc18c9415c912f9d8155881c133215df768a70，pair最大512 token、batch32、CPU4线程，按ONNX原始分数降序。", "",
         "## 方法参数", "", "```json", json.dumps(parameters["methods"], ensure_ascii=False, indent=2), "```", "",
         "## 消融参数", "", "```json", json.dumps(parameters["ablation"], ensure_ascii=False, indent=2), "```", ""]
preview = ["# 对比与消融表格预览", "", "100条query、1,000段passage，最终返回20段。R为Recall，P为Precision；MRR和nDCG仅展示@5，原始其他截断继续保留在metrics_all_k.csv。数值为0–1宏平均，保留四位小数。", ""]
legend = "粗体表示该列最优，下划线表示第二优的不同数值；并列时同时标注。标记基于均值大小，不表示统计显著性。"
preview += [legend, ""]
html_parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>对比与消融表格预览</title>',
              '<style>body{font-family:"Microsoft YaHei",sans-serif;margin:32px;color:#18202a}table{border-collapse:collapse;margin:16px 0 32px;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #d7dde5;text-align:right}th:first-child,td:first-child{text-align:left}th{border-top:2px solid #354052;border-bottom:2px solid #354052}tr:last-child td{border-bottom:2px solid #354052}u{text-underline-offset:3px}strong{font-weight:700}</style>',
              '<h1>对比与消融表格预览</h1><p>100条query、1,000段passage；指标为0–1宏平均。MRR/nDCG仅展示@5。</p>', f'<p>{escape(legend)}</p>']
for title, filename, selected in groups:
    lines += [f"## {title}结果", "", f"[单独的CSV]({filename})。A1为当前方案；出现在消融表中时为共同对照，不能当成新增实验。", "", legend, ""]
    table = ["| 配置 | R@5 | R@10 | R@20 | P@5 | P@10 | P@20 | MRR@5 | nDCG@5 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ranks = {}
    for key in display_metrics:
        best = max(row[key] for row in selected)
        lower = [row[key] for row in selected if best - row[key] > 1e-12]
        ranks[key] = (best, max(lower) if lower else None)
    html_parts += [f'<h2>{escape(title)}</h2><table><thead><tr>', *[f'<th>{escape(label)}</th>' for label in ('配置', 'R@5', 'R@10', 'R@20', 'P@5', 'P@10', 'P@20', 'MRR@5', 'nDCG@5')], '</tr></thead><tbody>']
    for row in selected:
        cells, html_cells = [], []
        for key in display_metrics:
            value = f"{row[key]:.4f}"
            best, second = ranks[key]
            if abs(row[key] - best) <= 1e-12:
                cells.append(f"**{value}**")
                html_cells.append(f"<strong>{value}</strong>")
            elif second is not None and abs(row[key] - second) <= 1e-12:
                cells.append(f"<u>{value}</u>")
                html_cells.append(f"<u>{value}</u>")
            else:
                cells.append(value)
                html_cells.append(value)
        table.append(f"| {row['id']} {row['method']} | " + " | ".join(cells) + " |")
        html_parts += [f'<tr><td>{escape(row["id"] + " " + row["method"])}</td>', *[f'<td>{value}</td>' for value in html_cells], '</tr>']
    html_parts.append('</tbody></table>')
    lines += [*table, ""]
    preview += [f"## {title}", "", *table, ""]
preview += ["配置与参数不变，没有重跑实验；全参数见 [parameters.json](parameters.json)，完整记录见 [SUMMARY.md](SUMMARY.md)。", "",
            "对比组只纳入当前正式方法；池化与归一化消融分别固定其他变量，A1是共同对照。已移除实验不列入。"]
(OUT / "TABLE_PREVIEW.md").write_text("\n".join(preview) + "\n", encoding="utf-8")
html_parts.append('</html>')
(OUT / "TABLE_PREVIEW.html").write_text("\n".join(html_parts) + "\n", encoding="utf-8")
lines += ["## 统计定义与来源", "", "Recall@k=前k相关数/该query全部相关数；Precision@k=前k相关数/k；MRR@k为前k内第一个相关项的倒数名次，否则0；nDCG@k使用二值gain与log2(rank+1)折扣，理想列表长度min(k,相关数)。", "",
          "RRF审计证实每路输入100：100路深度重算与全部100条旧Top-20一致，50路深度只匹配9条。历史文件保存前50名不能当作融合输入深度。审计见 ../rrf_depth_audit.json。", "",
          "所有既有Recall@5/10/20、MRR@10、nDCG@10、Precision@10均与原指标核对一致；新增的其余截断只来自原排名，不能当作新方法或新模型实验。参数全集见 [parameters.json](parameters.json)，未四舍五入指标见 [metrics_all_k.csv](metrics_all_k.csv)。", "",
          "耗时仍按原记录的阶段口径：BM25索引0.203s/检索0.004s；BGE文档46.48s/query编码检索0.160s；RRF两路合计46.684s/0.164s，融合本身未独立测；重排654s且模型加载时间未保存；TF-IDF分词1.510s/索引0.267s/query处理0.008s；BM25L索引0.115s/检索0.004s、BM25+索引0.110s/检索0.006s，共享分词1.600s。", "",
          "CLS/Mean联合脚本主体50.203s；Max和原始CLS向量及BM25变体联合主体51.581s，文档/query共享前向42.450/0.156s。不能把共享时间按每组重复相加，不含权重下载和依赖安装，也不是统一端到端测速。", "",
          "结果仅为MTEB/T2Retrieval dev派生pilot，不是官方完整基准；没有显著性检验或生成准确率。"]
(OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
unique_sources = sorted(set(sources + [ROOT / "results/t2retrieval/rrf_depth_audit.json", ROOT / "requirements-experiment.txt"]))
(OUT / "source_manifest.json").write_text(json.dumps({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in unique_sources}, indent=2), encoding="utf-8")
for row in rows:
    print(row["id"], row["method"], " | ".join(metric + ": " + "/".join(f"{row[f'{metric}@{k}']:.4f}" for k in (5, 10, 20)) for metric in ("recall", "mrr", "ndcg", "precision")))
print("Saved", OUT)
