# 实验数据说明

## 正式检索语料

主实验使用 Hugging Face `mteb/T2Retrieval` 的 dev 子集，数据卡标注 Apache-2.0 许可证，并提示该数据可能包含 MMTEB 的额外处理。当前使用的三个文件为 corpus、queries 和 qrels；数据集大小约 159 MB。来源：[数据卡](https://huggingface.co/datasets/mteb/T2Retrieval)。

本地文件位于 `data/raw/t2retrieval/mteb-*-dev.parquet`。`original-queries-dev.tsv` 和 `original-qrels-retrieval-dev.tsv` 来自 [THUIR/T2Ranking](https://github.com/THUIR/T2Ranking)，只用于来源和 ID 对照，不作为额外实验数据。

## 已完成的数据校验

- MTEB dev：22,812 queries、118,605 passages、118,932 qrels；所有 qrel 分数为 1。
- 所有 qrel query ID 均存在于 query 文件，所有 qrel passage ID 均存在于 corpus 文件。
- 原始 T2Ranking dev query 文件有 24,832 行，其中 22,812 个 query ID 有 retrieval qrels；其余 2,020 个不进入 MTEB dev query 集。
- 原始 qrels 去掉表头后，118,932 个 `(qid, pid)` 配对与 MTEB qrels 完全一致。MTEB query 有 10 条引号格式与原始 query 不同。
- MTEB corpus 的 118,605 个 ID 正好覆盖 qrels 中出现的所有相关 passage ID。它是相关 passage 的并集，不是原始 T2Ranking 的 2,303,643 篇候选集合；本实验指标只对应这个 MTEB dev 候选集。

这些结果不能解释为全量 T2Ranking 基准成绩，也不能直接外推到课程材料检索。每个 query 未标注的 passage 按当前检索指标视为非相关项。

## 首轮 pilot 采样

全量本地 CPU 向量化耗时过长，因此首轮 pilot 固定抽取 100 条有 qrels 的 query，并纳入这些 query 的全部相关 passage；再从剩余 corpus 用固定 seed 抽样补足，使候选 passage 总数固定为 1,000。本轮包含 512 个相关 passage ID 和 488 个未标注候选。BM25、稠密检索和 RRF 共用该候选池。结果需标为“MTEB/T2Retrieval dev 派生 pilot 子集”，不能作为完整 MTEB/T2Ranking 成绩。

query 样本、候选 passage ID、检索排名和运行元数据保存在 [`results/t2retrieval/pilot/`](../results/t2retrieval/pilot/)。

## 课程设计 PDF

工作区的 `深度学习-课程设计 2026.pdf` 只用于独立验证 PDF 提取和页码定位，不与 T2Retrieval 指标合并。

## 复现首轮历史 pilot

当前方案A1，正式对比A0/A2/A3/A5（TF-IDF）/A7（BM25L）/A8（BM25+），正式消融CLS/Mean/Max与文档L2。A4、A6 E5和HTML消融已移除。历史脚本和文件仍包含归档变体，正式清单以 `RAG_TECHNICAL_PLAN.md` 为准。

依赖版本见 [`requirements-experiment.txt`](../requirements-experiment.txt)。在 Windows PowerShell 中创建环境并安装依赖：

```powershell
uv venv --python 3.12 .venv_experiment
uv pip install --cache-dir .tmp_uv_cache --python .venv_experiment\Scripts\python.exe -r requirements-experiment.txt
```

首次运行前，需把 BGE-small-zh ONNX 权重下载到本地缓存；这一步只下载模型文件，不进行远端推理：

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'models\huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_XET_CACHE = Join-Path $env:HF_HOME 'xet'
.\.venv_experiment\Scripts\python.exe -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-zh-v1.5', cache_dir='models/fastembed', cuda=False, local_files_only=False)"
```

下载 MTEB dev 的三个 parquet 文件：

```powershell
New-Item -ItemType Directory -Force data/raw/t2retrieval | Out-Null
curl.exe -L --fail -o data/raw/t2retrieval/mteb-corpus-dev.parquet https://huggingface.co/datasets/mteb/T2Retrieval/resolve/main/corpus/dev-00000-of-00001.parquet
curl.exe -L --fail -o data/raw/t2retrieval/mteb-queries-dev.parquet https://huggingface.co/datasets/mteb/T2Retrieval/resolve/main/queries/dev-00000-of-00001.parquet
curl.exe -L --fail -o data/raw/t2retrieval/mteb-qrels-dev.parquet https://huggingface.co/datasets/mteb/T2Retrieval/resolve/main/data/dev-00000-of-00001.parquet
```

执行已缓存模型的离线 pilot：

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_t2retrieval_ablation.py --sample-queries 100 --candidate-passages 1000 --seed 2026 --top-k 20
.\.venv_experiment\Scripts\python.exe -u src/finalize_t2retrieval_pilot.py
```

脚本固定 `local_files_only=True`。第二步会复用已保存的 A0–A2 排序；如 A3 的 query/candidate ID 完全对齐，也会复用已保存的重排结果。top-20 指标、排名和运行元数据写入 `results/t2retrieval/pilot/`。

## 归档：复现轻量证据选择探索（不纳入正式清单）

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_evidence_selection.py
```

直接读取已保存的 BGE Top-50 ID/余弦分数、1,000个passage向量和原query/qrels，不重新编码、不加载模型、不联网。三组共用相同Top-50候选，最终均输出20段：原始相关性排序、MMR gamma=0.25、Dartboard纯向量版 sigma=0.1。参数预先固定，本次没有调参。Top-50选Top-20时Recall@20可以变化；原A0/A3固定Top-20仅换序的实验定义保持不变。

新增代码在 `src/run_evidence_selection.py`，结果独立写入 [`results/t2retrieval/evidence_selection_top50/`](../results/t2retrieval/evidence_selection_top50/SUMMARY.md)，不覆盖原pilot。配置、冻结候选、三组排序、逐query指标、相对基线差值、原文案例和计时均保存。程序会校验输入hash、qrels、向量ID对齐、MMR关闭还原基线、旧Top-20 ID与六项指标完全一致，以及输出唯一性和候选召回上限。当前固定参数下MMR/Dartboard未改善质量，完整数值与限制见实验记录。

## 复现正式消融：BGE 池化策略（2026-11-13）

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_pooling_ablation.py
```

使用相同本地ONNX权重及tokenizer，对原100条query和1,000个passage重新运行一次共享主干前向，同时提取CLS与attention-mask mean表示；不需要新权重。Mean排除padding并包含CLS/SEP，两组都L2归一化，并分别在完整1,000-parent池检索Top-20，不限于原Top-50候选。CLS配置需复现已有基线。

结果写入 [`results/t2retrieval/pooling_ablation/`](../results/t2retrieval/pooling_ablation/SUMMARY.md)，两种query/document向量及ID映射保存在 `data/processed/t2retrieval/pilot/pooling_ablation/`，不覆盖原向量。本次脚本主体约50.20秒，mean作为消融变体保留，最终方法仍用CLS；完整配置、逐query变化、案例、分阶段计时和hash均已保存。该正式消融不训练模型，只比较冻结BGE权重的推理池化规则。

## 历史扩展复现：TF-IDF保留，E5/HTML已归档

本节保留历史运行命令与真实结果，E5和HTML消融不再属于当前正式实验。A5 TF-IDF仍正式保留；执行旧 `local` 会同时运行历史HTML组，不能把这理解为恢复正式清单。

本阶段依赖上一步的CLS query/document缓存；输入ID和向量hash必须与原pilot一致。新增依赖版本已写入 `requirements-experiment.txt`：scikit-learn 1.9.1、SciPy 1.18.1、tokenizers 0.23.3、huggingface-hub 1.33.0。scikit-learn与SciPy核心许可为BSD-3-Clause，二进制包的附加许可保留在安装包中。[scikit-learn](https://github.com/scikit-learn/scikit-learn)、[SciPy](https://github.com/scipy/scipy)

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_retrieval_extensions.py --stage local
.\.venv_experiment\Scripts\python.exe -u src/download_e5_model.py
.\.venv_experiment\Scripts\python.exe -u src/run_retrieval_extensions.py --stage e5
```

`local`运行词级TF-IDF和passage去标签消融，只重新编码666个实际变化的passage；query和334个未变文档向量复用。E5下载脚本只取官方FP32 ONNX和必要配置，固定revision并复用已下载文件，记录许可、获取时间和文件hash；本次revision为 `614241f622f53c4eeff9890bdc4f31cfecc418b3`，模型许可MIT，ONNX文件约470MB。推理脚本不联网，E5用官方query/passage前缀和masked mean；它与BGE属于管线对比，不是单因素消融。

下载后可用 `--stage all`依次重跑两阶段，不重复下载模型。结果独立保存于 [`results/t2retrieval/retrieval_extensions/`](../results/t2retrieval/retrieval_extensions/SUMMARY.md)，原pilot不覆盖；新向量、TF-IDF矩阵、IDF及ID映射位于 `data/processed/t2retrieval/pilot/retrieval_extensions/`。不训练或调用外部模型API。结果是小pilot，E5部分指标略高、部分略低，HTML去标签也没有呈现一致优势，不外推普遍结论。

## 当前新增：BM25L/BM25+对比、Max与文档L2消融

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_additional_ablation.py
```

复用原始数据及CLS缓存校验，BM25S固定k1=1.5、b=0.75、delta=0.5，对应原生IDF；它们是方法对比。BGE一次共享前向提取原始CLS和masked Max，Max排padding并包含有效CLS/SEP。归一化消融固定原CLS单位query和同一点积，只替换文档单位CLS/原始CLS，不是在余弦函数中做无效重复归一化。

原A0和A1的Top-20 ID与六项指标均复现。结果见 [`additional_ablation/`](../results/t2retrieval/additional_ablation/SUMMARY.md)，原始CLS、Max向量及ID映射位于 `data/processed/t2retrieval/pilot/additional_ablation/`；脚本主体约51.58秒，无新依赖、模型下载或训练。池化消融当前为CLS/Mean/Max，其中Mean来自上一池化实验。本轮继续使用CLS与文档L2，所有结果仅为小pilot。
