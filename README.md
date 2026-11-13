# 中文网页段落检索：方法对比与 BGE 表示策略消融

基于 **MTEB/T2Retrieval dev 派生 pilot 子集**的离线中文证据检索实验。固定 100 条 query、1,000 个原始 passage，以原 passage ID 评价检索结果。所有模型均为冻结权重的本地 CPU 推理，不训练、不调用外部模型 API，不报告生成正确率。

## 正式实验与结果

A1（BGE-small-zh-v1.5、CLS、query/document L2）为当前方案；参照方法为 A0 BM25、A2 RRF、A3 BM25 Top-20 加 cross-encoder 重排、A5 词级 TF-IDF、A7 BM25L、A8 BM25+。

以下数值来自已保存排名的汇总，保留四位小数：

| 配置 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| A0 BM25 | 0.7248 | 0.8566 | 0.9034 | 0.9300 | 0.8748 |
| A1 BGE-CLS | 0.8019 | 0.9337 | 0.9584 | 0.9620 | 0.9476 |
| A2 BM25+BGE RRF | 0.7653 | 0.8933 | 0.9269 | 0.9600 | 0.9189 |
| A3 BM25 Top-20 加重排 | 0.7798 | 0.8919 | 0.9034 | 0.9600 | 0.9212 |
| A5 词级 TF-IDF | 0.7154 | 0.8507 | 0.8952 | 0.9181 | 0.8573 |
| A7 BM25L | 0.7320 | 0.8566 | 0.9059 | 0.9253 | 0.8730 |
| A8 BM25+ | 0.7248 | 0.8566 | 0.9068 | 0.9300 | 0.8748 |

受控消融分别比较 CLS/attention-mask Mean/masked Max 池化，以及固定单位 query 和点积计算时的文档 L2 开关。各池化组独立检索同一完整 1,000-passage 池；归一化关闭组使用原始 CLS 文档向量。A3 仅重排 A0 的同一 Top-20，故 Recall@20 相同。RRF 每路输入深度为 100、常数为 60。

完整参数、消融表、@5/@10/@20 指标及来源哈希见 [正式汇总](results/t2retrieval/current_summary/SUMMARY.md)、[完整指标 CSV](results/t2retrieval/current_summary/metrics_all_k.csv) 和 [参数 JSON](results/t2retrieval/current_summary/parameters.json)。

## 仓库内容

- `src/`：检索、重排、池化和归一化消融，以及指标汇总脚本。
- `results/t2retrieval/current_summary/`：当前正式结果入口。
- `results/t2retrieval/pilot/`：初始排名、100 条 query/qrels、1,000 个候选 ID、指标与元数据。
- `results/t2retrieval/pooling_ablation/`、`additional_ablation/`：池化、文档 L2 和 BM25 变体的配置、排名与逐 query 指标。
- `results/t2retrieval/retrieval_extensions/local/`：正式 A5 TF-IDF 的来源记录；同目录的 HTML 变体仅归档。
- `data/README.md`：数据说明与历史运行命令；[来源和许可说明](THIRD_PARTY_NOTICES.md)说明公开文本的归属。

历史 A4、A6 E5、HTML 和 MMR/Dartboard 的代码、数值保留用于追溯，**不纳入正式实验**。旧脚本可能同时计算归档方法；正式范围以当前汇总和 [技术方案](RAG_TECHNICAL_PLAN.md) 为准。已有 PRF/查询增强未在当前记录中保存具体变体及指标，本仓库不补造数值。

## 复现

记录环境为 Windows、Python 3.12.10、ONNX Runtime CPU；直接依赖版本固定在 [requirements-experiment.txt](requirements-experiment.txt)。模型和原始数据不随仓库上传，需先下载公开文件，之后实验离线运行。

仅重算已保存排名的正式指标，无需模型、数据下载或第三方 Python 包：

```powershell
python src/summarize_results.py
```

该命令会覆写 `current_summary/` 中的派生汇总，不重跑编码或检索。

重新执行实验时先安装依赖：

```powershell
uv venv --python 3.12 .venv_experiment
uv pip install --cache-dir .tmp_uv_cache --python .venv_experiment\Scripts\python.exe -r requirements-experiment.txt
```

按 [data/README.md](data/README.md) 下载三个 dev parquet 文件，并核对 `pilot/run_metadata.json` 中的 SHA-256。原下载记录未保存数据集 revision，不能把当前 `main` 当作原版本；文件哈希是本次数据一致性的依据。

模型缓存须包含以下历史 snapshot（Hugging Face 缓存布局）：

| 用途 | 下载仓库 | revision | cache_dir |
|---|---|---|---|
| BGE 向量 | `Qdrant/bge-small-zh-v1.5` | `46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59` | `models/fastembed` |
| 重排 | `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` | `models/fastembed-reranker` |

可在准备阶段使用 `huggingface_hub.snapshot_download`，显式传入表中 `repo_id`、`revision`、`cache_dir`；向量模型需要 `model_optimized.onnx`，重排需要 `onnx/model.onnx`，两者还需要 tokenizer 和 config 文件。不要下载训练数据或改用外部推理服务。模型文件校验值见各实验 `run_metadata.json`。

准备完成后，按依赖顺序运行：

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_t2retrieval_ablation.py --sample-queries 100 --candidate-passages 1000 --seed 2026 --candidate-k 100 --top-k 20
.\.venv_experiment\Scripts\python.exe -u src/finalize_t2retrieval_pilot.py
.\.venv_experiment\Scripts\python.exe -u src/run_pooling_ablation.py
.\.venv_experiment\Scripts\python.exe -u src/run_retrieval_extensions.py --stage local
.\.venv_experiment\Scripts\python.exe -u src/run_additional_ablation.py
.\.venv_experiment\Scripts\python.exe src/summarize_results.py
```

上述历史管线会同时计算归档 A4/HTML；无需运行 E5 下载或推理。重跑会写回原结果目录，建议在单独克隆中进行。向量缓存由脚本重新生成，未上传到 GitHub。历史元数据保留原运行时间和哈希，重跑耗时及浮点末位可能随设备/环境变化；依赖文件只锁定直接依赖，并非完整环境锁文件。

## 数据与结论边界

数据来自 [mteb/T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval)，上游为 [THUIR/T2Ranking](https://github.com/THUIR/T2Ranking)，记录的数据许可为 Apache-2.0。MTEB 可能包含额外处理；其 corpus 为 qrels 相关 passage 的并集，不是原始 230 万段落候选全库。

本次 query seed=2026、补样 seed=2027；池内包含所选 query 的 512 个相关 passage 并集和 488 个随机未标注项，未标注按非相关评价。该设计使问题相关段落均在池内，成绩不能外推为完整 MTEB/T2Ranking 基准。没有显著性检验，不能宣称提出新算法或证明普遍优势。阶段计时也不是统一端到端测速。

课程设计 PDF、本地报告、虚拟环境、模型权重、原始数据和向量缓存未上传。包含网页正文的 `case_analysis.jsonl`、`passage_preprocessing.jsonl` 及词表 `tfidf_vocabulary.json` 也仅留在本地；历史说明中提到这些文件时，不代表它们随仓库分发。仓库提供代码与实际实验记录，不包含课程报告。
