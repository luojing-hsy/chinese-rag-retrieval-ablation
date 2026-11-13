# RAG 方法对比与 BGE 表示策略消融方案

调研日期：2026-11-12；结果更新：2026-11-13  
目标：围绕中文网页段落证据检索完成离线、无训练、无外部推理API的实验。当前正式七种检索配置与两组BGE表示策略消融均已在100-query/1,000-passage pilot完成；已移除实验只保留历史记录。

当前正式清单：**A1为当前方案；对比实验A0/A2/A3/A5/A7/A8；消融为CLS/Mean/Max池化，以及固定点积下的文档L2归一化开关。**A7/A8为新增BM25L/BM25+，均已运行。A4、A6 E5和HTML消融均已移除；MMR/Dartboard也只归档。原始记录保留追溯，不代表纳入正式清单。

## 结论摘要

**正式实验使用BM25S、FastEmbed、RRF和scikit-learn TF-IDF。**FastEmbed进程内运行本地BGE，BM25S提供BM25、BM25L和BM25+；融合、表示策略及指标由脚本计算，不启用远端推理。[FastEmbed官方模型表](https://qdrant.tech/documentation/embeddings/fastembed/)、[BM25S官方项目](https://github.com/xhluca/bm25s)

FlashRAG 仍是适合研究复现的候选；但本机对 `flashrag-dev==0.1.2` 做依赖预演时解析出 Torch、PEFT、spaCy、FastChat、Streamlit 和 OpenAI SDK 等额外依赖。由于本轮只评估检索，不需要这些组件，暂不安装完整框架，改用轻量本地组件。FastEmbed 与 BM25S 已在 Python 3.12.10 环境安装；BGE-small 中文 ONNX 权重已下载，并确认能在本机生成 512 维 query/passage 向量。

全量 118,605 个 passage 的 CPU 向量化耗时超出课程实验可接受范围：一次试跑约 7 分钟只处理了约 2,000 个文档；当前 ONNX Runtime 只有 CPU/Azure providers。FastEmbed 官方 GPU 安装说明要求 `fastembed-gpu` 与 CUDA 12.x，本机 CUDA Toolkit 为 11.6，因此首轮不改系统级 CUDA 环境，改跑一个可复现小 pilot：抽 100 个 query，候选池总数固定为 1,000，先保留这 100 个 query 的全部 qrel 相关 passage，再用固定 seed 抽样补足；BM25、稠密和 RRF 使用同一候选池。该 pilot 不作为完整 MTEB benchmark 分数。[FastEmbed GPU/并行运行说明](https://qdrant.tech/documentation/fastembed/fastembed-optimize/)

## 候选开源项目

| 项目 | 与本项目相关的官方能力 | 本地运行与实验适配 | 许可证与主要权衡 |
|---|---|---|---|
| [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | 官方将其定位为 RAG 研究与复现工具；README 列出多种检索器、重排器、生成器和压缩组件。自定义语料可用 JSONL；配置支持 BM25、稠密检索、多检索器融合（含 RRF）和重排。问题集可带 `golden_doc_ids`，用于计算文档级 Recall、Precision、F1、MAP。[README](https://github.com/RUC-NLPIR/FlashRAG/blob/main/README.md)、[配置文件](https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/config/basic_config.yaml)、[多检索器与 RRF 用法](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/multi_retriever_usage.md) | 本地模型路径受支持，配置也提供 `hf`、`vllm`、`fschat` 等本地推理框架选项；可完全避开 `openai` 选项。BM25 可用 `bm25s`，稠密索引使用 FAISS。全流程 `SequentialPipeline` 包含生成阶段；若只评估检索，需要直接用 retriever 组件并自行算指标，或写一个很薄的检索评估流程。[配置](https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/config/basic_config.yaml)、[组件用法](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/basic_usage.md)、[索引说明](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/building-index.md) | MIT。[README](https://github.com/RUC-NLPIR/FlashRAG/blob/main/README.md) 写明 Python 3.10+，而当前 `setup.py` 元数据写的是 Python ≥3.9；应锁定具体 tag/commit 后再按该版本安装。稠密检索还要处理 FAISS 安装；vLLM、Pyserini 等为可选依赖，建议避开不需要的部分。框架面向通用研究基准，默认示例会下载较大的数据/模型；本项目应只使用自有小语料和需要的检索组件。[安装说明](https://github.com/RUC-NLPIR/FlashRAG/blob/main/README.md)、[setup.py](https://github.com/RUC-NLPIR/FlashRAG/blob/main/setup.py) |
| [Haystack](https://github.com/deepset-ai/haystack) | `DocumentSplitter` 支持按句子、段落、页、行或 token 切分；`InMemoryBM25Retriever`、`InMemoryEmbeddingRetriever` 可在小语料上做关键词和稠密检索；`DocumentJoiner` 支持 RRF；Sentence Transformers 集成提供本地向量器和交叉编码重排器。[Splitter](https://docs.haystack.deepset.ai/docs/documentsplitter)、[Retriever](https://docs.haystack.deepset.ai/docs/retrievers)、[Joiner](https://docs.haystack.deepset.ai/docs/documentjoiner)、[Sentence Transformers 集成](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/sentence-transformers.md) | 组件可在普通 Python 流程中组合；向量器支持本地模型路径和 `local_files_only=True`，小语料可用内存 Document Store，无须先部署向量数据库。[本地模型参数](https://docs.haystack.deepset.ai/reference/integrations-sentence-transformers)、[内存 Document Store](https://docs.haystack.deepset.ai/docs/inmemorydocumentstore) | 核心库 Apache-2.0，当前项目环境 Python 3.12.10 满足其 `pyproject.toml` 的 Python ≥3.11 要求。[官方 pyproject](https://github.com/deepset-ai/haystack/blob/main/pyproject.toml) 对 T2Ranking passage 再切块需要自定义预处理；检索指标需自行整理或计算。官方当前内存 BM25 代码对中日韩文本默认按单字符切 token，需用本实验的中文查询/网页段落小样本比较，不能仅凭组件名称假定效果好。[官方源码](https://github.com/deepset-ai/haystack/blob/main/haystack/document_stores/in_memory/document_store.py) |
| [LlamaIndex](https://github.com/run-llama/llama_index) | 文档处理和 RAG API 较丰富；支持 Sentence Transformers 本地嵌入、retriever，以及对检索节点做过滤/重排的 postprocessor。[README](https://github.com/run-llama/llama_index/blob/main/README.md)、[本地嵌入文档](https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/module_guides/models/embeddings.md)、[RAG 组件说明](https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/understanding/rag/index.mdx) | 可只安装 core 和选定 integration；本地模型可用，但要显式配置。官方嵌入文档指出，不配置时会回退到 OpenAI embedding，因此本项目必须明确指定本地 embedding 和本地模型组件。[嵌入文档](https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/module_guides/models/embeddings.md) | core 为 MIT。[官方 core pyproject](https://github.com/run-llama/llama_index/blob/main/llama-index-core/pyproject.toml) 集成按包拆分；对本项目，provider/integration 组合和版本管理多于所需，混合检索与指标也需额外组合。更适合之后要扩展成应用时考虑。 |

### 框架选择

**本轮执行使用 FastEmbed + BM25S，不安装完整 FlashRAG。**FastEmbed 用本地 ONNX 权重生成中文 passage/query 向量，BM25S 建立词法检索；RRF 与指标由本项目脚本计算。FlashRAG 保留为研究框架备选，若之后需要生成管线或更多算法再评估。[FastEmbed 本地模型列表](https://qdrant.tech/documentation/embeddings/fastembed/)、[BM25S](https://github.com/xhluca/bm25s)

范围收窄如下：使用 BM25S、jieba、FastEmbed 和 PyArrow；BGE-small 中文 ONNX 权重已放在项目本地缓存。实验运行时固定 `local_files_only=True`，不启用 OpenAI、联网搜索或远端推理端点，不训练或微调模型。

## 可实际使用的背景与数据方案

建议题目：**面向中文网页段落的离线证据检索：方法对比与BGE表示策略消融**。

### 场景

**中文网页证据检索助手**：用户输入一个中文事实问题，系统从本地网页段落库中找出最相关证据，展示段落 ID 和文本；它适用于需要快速定位公开网页依据的检索场景。实验聚焦检索排序质量，不承诺通用问答能力，也不要求生成模型、训练过程或在线服务。T2Ranking 的问题来自搜索日志，文档段落来自搜索引擎结果，适合支撑这个场景。[T2Ranking 官方说明](https://github.com/THUIR/T2Ranking/)

这会把此前的“课程资料助手”收窄并调整为适合数据的公开中文网页检索背景。当前工作区的 `深度学习-课程设计 2026.pdf` 只有三页，可保留为本地 PDF 解析烟测，不作为正式消融语料，也不能把网页检索结果表述成课程规则检索效果。若之后必须主张课程场景，需要另行取得足量且获准使用的课程文件并人工标注文档级相关性。

### 数据集核验与选择

| 数据集 | 官方信息与许可 | 与本实验的适配 | 结论 |
|---|---|---|---|
| **T2Ranking 原始 dev** | THUIR 官方仓库列出 2,303,643 个段落、24,832 个 dev 查询、118,933 条 retrieval dev qrels；字段为段落 `pid/text`、查询 `qid/query` 和 `qid/pid` 相关关系。完整数据另含约 2 亿行训练负例。仓库明确写明数据许可为 Apache-2.0。[官方 README](https://github.com/THUIR/T2Ranking/) | 中文原生检索数据，带 query 与 gold passage ID；检索 qrels 不需要训练。原始 collection 有 230 万段落，完整仓库还包含巨大训练负例，不能整仓克隆。[数据规模与下载说明](https://github.com/THUIR/T2Ranking/) | 权威源、许可明确，但完整 collection 可能超过本课程项目所需规模。不要下载 BM25/hard-negative 训练文件。 |
| **MTEB/T2Retrieval dev** | Hugging Face 卡标注 Apache-2.0，列出 corpus 118,605 条、queries 22,812 条、qrels 118,932 条；总下载约 159 MB，其中 corpus 约 157 MB、queries 约 818 KB、qrels 约 1.15 MB。字段为 `_id/text`、`query-id/corpus-id/score`。[MTEB 数据卡与统计](https://huggingface.co/datasets/mteb/T2Retrieval)、[文件列表](https://huggingface.co/datasets/mteb/T2Retrieval/tree/main) | 规模比原始 230 万段落 collection 小，含 query、passage 和 qrels；适合只评估检索，不需要训练或调用模型 API。MTEB 卡说明该数据可能经 MMTEB 再处理。 | **已下载并选用**。与原始 dev 的 query 行数和 passage 候选集不同，结果只称 MTEB dev 子集结果；本地已核验 qrel ID 覆盖。 |
| **C-MTEB/T2Retrieval 镜像** | Hugging Face 页面列出 corpus 和 qrels；qrels 为 118,932 行，Parquet 文件总计约 158 MB。[C-MTEB corpus](https://huggingface.co/datasets/C-MTEB/T2Retrieval)、[C-MTEB qrels](https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels) | 与 MTEB 镜像的规模看起来相近，但其数据卡显示“More Information needed”，没有独立许可证说明；目前不能确认两套镜像的文件和 ID 完全一致。 | 不作为首选下载源。先与 THUIR 原始数据及 MTEB 镜像做 ID 对照；在许可证与来源链核验完成前，不公开再分发其数据副本。 |
| [MIRACL Chinese](https://github.com/project-miracl/miracl) | 原生多语 IR：中文 dev 为 393 queries、3,928 judgments；中文语料有 4,934,368 段落，压缩后约 733 MB。Hugging Face 卡标注 Apache-2.0，项目说明语料源自 Wikipedia。[topics/qrels 数据卡](https://huggingface.co/datasets/miracl/miracl)、[中文 corpus 文件](https://huggingface.co/datasets/miracl/miracl-corpus/tree/main/miracl-corpus-v1.0-zh) | 标签结构完整，适合原生 retrieval 评测；完整中文语料和索引开销比 MTEB/T2Retrieval 子集大。MIRACL 的加载脚本会读取 corpus；不要只凭 qrels 文件大小推断不需要语料。Wikipedia 上游文本的归属和许可义务仍需随转载内容核验。 | 可作为以后需要大语料复验的选择；当前不下载，也不作为本次小实验主数据。 |
| [DuReader Retrieval](https://github.com/baidu/DuReader/blob/master/DuReader-Retrieval/README.md) | 百度官方说明其含约 809 万段落、2K dev 和 4K test 查询，并采用 MRR@10/Recall@k；README 指向注册后下载，但没有公开声明数据许可证。[官方任务与规模](https://github.com/baidu/DuReader/blob/master/DuReader-Retrieval/README.md) | 原生中文检索 benchmark，但数据规模大、获取有注册步骤，公开许可不清楚。 | 当前不选；若考虑使用，先取得数据使用条款或明确许可。 |

**选择结论：已下载并核验 `mteb/T2Retrieval` 的 dev 子集。**数据卡标注 Apache-2.0；本地文件包含 22,812 条查询、118,605 个 passage 和 118,932 条 qrels，下载约 159 MB。它适合比较 BM25、稠密和混合检索，不需要训练或调用模型 API。实验结果称为“MTEB/T2Retrieval dev 子集上的检索结果”，不能标成原始 T2Ranking 全量官方成绩。[MTEB 数据卡与统计](https://huggingface.co/datasets/mteb/T2Retrieval)

**与原始 T2Ranking 的对应关系已做 ID 级核验，passage 文本未做全量哈希对照。**原始 T2Ranking dev query 文件有 24,832 行，其中 2,020 个 query 没有 retrieval qrels；剩余 22,812 个 query ID 与 MTEB 子集完全对应。原始 qrels 去掉表头后有 118,932 个 `(qid,pid)` 配对，与 MTEB qrels 完全一致；MTEB query 有 10 条在引号格式上与原始 query 不同。全部 118,932 个 qrel passage ID 都存在于 MTEB corpus，且 corpus 的 118,605 个 ID 正好覆盖 qrels 的 unique relevant IDs。MTEB 卡片说明数据可能经过 MMTEB 额外处理；这个候选集不是原始 230 万段落全库，因此不与全库成绩直接比较。[MTEB 数据卡](https://huggingface.co/datasets/mteb/T2Retrieval)、[原始 T2Ranking 统计](https://github.com/THUIR/T2Ranking/blob/main/README.md)

### 本地数据准备与边界

- 只准备一个数据源的 **dev/evaluation 文件**，不下载 train、训练负例或模型 checkpoint。已下载 MTEB dev 的 corpus/query/qrels，以及原始 T2Ranking dev 的 query/qrels TSV，仅用于来源与 ID 对照；未下载原始全量 collection 或训练负例。[数据文件](https://huggingface.co/datasets/mteb/T2Retrieval)
- MTEB 3 个 dev 文件总量约 159 MB；模型缓存与数据均保存在项目目录。运行实验只从本地文件读取，不访问推理端点。
- 统一映射为 `corpus: {id: pid, contents: text}`、`queries: {qid, query}`、`qrels: {qid, pid, score}`。过滤 HTML 占位标记等清理规则必须固定并记录；所有检索变体使用相同的规范化文本和 ID。
- 数据含 query 与相关段落标签，但不提供适合作为生成评价的标准答案。主实验报告检索指标；如另加本地生成，只能固定生成模型和提示，并对引用证据做人工抽查，不调用外部 API，也不训练模型。
- 如果后续将源网页段落或衍生数据集提交到 GitHub，遵循所用数据卡声明并保留来源和许可说明。不要把当前课程 PDF 或未获准公开的教师材料混入公开语料。

## 技术方案

```mermaid
flowchart LR
  A[T2Retrieval dev：corpus、queries、qrels] --> B[本地校验 ID 与许可来源]
  B --> C[固定 passage corpus]
  C --> D1[BM25 / BM25L / BM25+]
  C --> D2[BGE 本地稠密检索]
  C --> D3[TF-IDF]
  D1 --> E[RRF 混合检索对比]
  D2 --> E
  D1 --> F[BM25 Top-20重排对比]
  D1 --> G[Top-20 passage ID]
  D2 --> G
  D3 --> G
  E --> G
  F --> G
  G --> H[Recall@k / MRR / nDCG]
  D2 -. BGE池化消融 .-> J[固定权重：CLS / Mean / Max]
  J --> G
  D2 -. 文档归一化消融 .-> K[固定点积：单位CLS vs 原始CLS]
  K --> G
```

### 组件与控制条件

1. **数据对齐**：MTEB corpus 字段为 `_id/text/title`，queries 为 `_id/text`，qrels 为 `query-id/corpus-id/score`。本地已核验每条 qrel 的 query 和 passage 均能找到；全部 score 为 1，指标基于 passage ID 计算。[MTEB 数据结构与统计](https://huggingface.co/datasets/mteb/T2Retrieval)
2. **文本规范化**：固定 HTML 处理方式，保留标题及正文；移除 `<img>` 等非文本占位符时需记录规则。对所有消融使用完全相同的规范化文本。
3. **检索粒度**：正式实验把官方passage本身作为检索单元，不再纳入切块及parent聚合变体；保留原qrels的passage ID评价。
4. **检索**：BM25S 作为词法基线；FastEmbed 本地 BGE-small 中文 ONNX 模型作稠密对照；混合组以 BM25 + 稠密结果使用 RRF 融合。BM25 固定用 jieba 分词；先清理 HTML 占位标记，再对所有 BM25 变体使用同一预处理规则。
5. **重排**：只在同一组候选 passage 上增加一个本地预训练 cross-encoder 重排器；模型不微调。中文向量模型候选为 `BAAI/bge-small-zh-v1.5`；重排候选为 `BAAI/bge-reranker-base`。固定候选数，并从磁盘本地加载模型权重。[向量模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[重排模型卡](https://huggingface.co/BAAI/bge-reranker-base)
6. **生成**：不是主实验环节。T2Retrieval 只提供 query 和相关 passage qrels，没有标准答案文本；因此不以生成准确率作为本数据集的定量结论。若需要演示端到端输出，可固定一个本地预训练模型和提示词，人工抽查引用是否支持答案，不调用外部 API。

### 正式对比实验与消融实验

A1为当前方案，A0/A2/A3/A5/A7/A8作为对比方法。正式消融有两组：固定BGE权重的池化策略，以及文档CLS的L2归一化。A0/A3不单列消融；A4、E5及HTML消融移除。

| 消融问题 | 对照 | 处理组 | 固定条件 | 变化项与解释范围 |
|---|---|---|---|---|
| 池化策略如何影响BGE检索？ | A1：CLS | attention-mask Mean或masked Max | query、qrels、池、权重、清洗、512-token截断、L2和余弦口径相同 | 只替换query/passage的池化规则，均全池检索20段；推理阶段消融 |
| 文档L2归一化是否影响排序？ | 单位CLS文档向量 | 未归一化原始CLS文档向量 | 固定单位CLS query、相同点积计算、同一原始token输出和其他处理 | 仅改变文档归一化；开启等价A1余弦，关闭保留文档范数作用，不在余弦函数中重复开关L2 |

| 编号 | 方法 | 实验角色 |
|---|---|---|
| A0 | BM25 + jieba | 词法检索对比方法 |
| A1 | BGE-small-zh 整段稠密检索、CLS池化 | 最终方案；也是池化消融的原始配置 |
| A2 | BM25 + BGE + RRF | 混合检索对比方法 |
| A3 | BM25 Top-20 + BGE cross-encoder 重排 | 两阶段检索对比方法 |
| A5 | jieba词级TF-IDF + 余弦检索 | 传统稀疏检索对比；词表和IDF只统计passage |
| A7 | BM25L | 词法评分方法对比；k1=1.5、b=0.75、delta=0.5 |
| A8 | BM25+ | 词法评分方法对比；k1=1.5、b=0.75、delta=0.5 |

所有方法使用相同100条query、qrels和同一1,000个passage池，统一返回Top-20，报告Recall@5/10/20、MRR@10和nDCG@10。A3只重排每条query的BM25 Top-20，因此与A0的Recall@20一致；A1与mean消融变体则分别在全池检索，Recall@20可以变化。

### 正式方法对比结果

首轮pilot固定seed抽100条query，候选池含全部512个相关passage并集及488个随机未标注项。正式表为A0/A1/A2/A3/A5/A7/A8。来源：[首轮指标](results/t2retrieval/pilot/metrics.csv)、[TF-IDF指标](results/t2retrieval/retrieval_extensions/local/metrics.csv)、[新对比指标](results/t2retrieval/additional_ablation/metrics.csv)、[query样本](results/t2retrieval/pilot/query_sample.jsonl)。原始文件可能还含归档变体，正式清单以本节为准。

| 方法 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| A0 BM25 + jieba | 0.7248 | 0.8566 | 0.9034 | 0.9300 | 0.8748 |
| A1 BGE-small-zh 整段稠密检索 | **0.8019** | **0.9337** | **0.9584** | **0.9620** | **0.9476** |
| A2 BM25 + BGE + RRF | 0.7653 | 0.8933 | 0.9269 | 0.9600 | 0.9189 |
| A3 BM25 top-20 + BGE cross-encoder 重排 | 0.7798 | 0.8919 | 0.9034 | 0.9600 | 0.9212 |
| A5 词级TF-IDF | 0.7154 | 0.8507 | 0.8952 | 0.9181 | 0.8573 |
| A7 BM25L | 0.7320 | 0.8566 | 0.9059 | 0.9253 | 0.8730 |
| A8 BM25+ | 0.7248 | 0.8566 | 0.9068 | 0.9300 | 0.8748 |

A3相较A0改善前排次序，但Recall@20不变。A7/A8相对A0的Recall@20略高，前排指标没有一致提升；各变体使用库对应默认IDF，不能单独归因于delta或长度规则。在当前正式表中A1的各项指标最高，继续保留；这不等于全部历史尝试或其他数据上均最优。未做统计显著性检验。

A3对100条query×20段的本地CPU重排评分约654秒；A1对1,000个passage构建向量约46.48秒。全量118,605段的CPU向量化未完成，结果只对应1,000段pilot，不等同官方全候选评估。

A5分词约1.51秒、索引0.27秒、query处理0.008秒。A7/A8此次共享分词1.60秒，索引各0.115/0.110秒，100query检索各0.0040/0.0056秒；这些是局部计时，不包含全部I/O且不是严格时延基准。[新增实验记录](results/t2retrieval/additional_ablation/SUMMARY.md)

### 归档探索（不纳入正式方法清单）

A4、A6 E5及HTML消融已按用户要求移除，数值仅保留在原始档案。MMR/Dartboard也只保留历史记录，不列入主表。PRF和查询增强已做，不重复；不补造当前记录中缺失的具体配置和数值。

### 正式消融：固定 BGE 权重的池化策略（2026-11-13）

固定同一BGE ONNX权重、文本、512-token截断、清洗、归一化和余弦口径，仅比较最后一层CLS、Mean和Max。Mean/Max排padding、包含有效CLS/SEP；Max以负无穷屏蔽padding。query和passage一致应用各自规则，全池检索Top-20。该组是推理阶段池化消融，不证明训练时池化的贡献。[BGE官方CLS用法](https://huggingface.co/BAAI/bge-small-zh-v1.5#using-huggingface-transformers)

这里的Max是在同一文本内部对token表示逐维取最大值，不是切块后parent max聚合，不恢复已移除的A4。

| 池化 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| CLS | 0.801891 | 0.933719 | 0.958444 | 0.962000 | 0.947584 |
| Mean | 0.794058 | 0.921473 | 0.948985 | 0.955833 | 0.930852 |
| Max | 0.764462 | 0.907226 | 0.938636 | 0.939167 | 0.903458 |

CLS的全部Top-20 ID和六项平均指标复现原A1，文档向量最大绝对误差约5.96e-8。Mean的nDCG@10在4条query改善、23条下降、73条相同；Recall@20在6条下降、94条相同。本轮保留CLS，不据此声称mean在其他模型或重新训练后也无效。

新增Max的nDCG@10相对CLS在2条改善、38条下降、60条相同，Recall@20在11条下降。Max及文档原始CLS共享一次BGE前向，文档/query约42.45/0.156秒，两组新表示和BM25变体的脚本主体合计51.58秒；不能分别把该时间计入每一处理组。详见 [新增实验记录](results/t2retrieval/additional_ablation/SUMMARY.md)。

共享文档/query主干前向约42.80/0.14秒，脚本主体约50.20秒；主干只跑一次，不能伪装成两组独立端到端计时。配置、排名、逐query指标、向量hash、校验与案例见 [池化实验记录](results/t2retrieval/pooling_ablation/SUMMARY.md)、[指标](results/t2retrieval/pooling_ablation/metrics.csv)、[元数据](results/t2retrieval/pooling_ablation/run_metadata.json)；复现脚本为 `src/run_pooling_ablation.py`。

### 正式消融：文档CLS的L2归一化（2026-11-13）

固定原CLS单位query、原始token输出及同一 `q_unit @ d` 点积，文档输入分别为单位CLS与未归一化CLS。只改变文档L2，query、清洗、池化和评分操作不变；开启等价A1余弦，关闭让文档范数参与排序。它不同于在余弦函数中开关重复归一化；原始范数必须从token输出保留，不能从单位向量推回。

| 文档L2 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| 开启，A1 | 0.801891 | 0.933719 | 0.958444 | 0.962000 | 0.947584 |
| 关闭 | 0.780129 | 0.927905 | 0.958444 | 0.956429 | 0.933483 |

原始文档范数范围5.0321–8.0036，变量确实存在；开启L2还原原基线。关闭后nDCG@10在6条改善、18条下降、76条相同，Recall@20均实测相同，非固定候选强制。固定参数下保留文档L2更合适，不外推所有数据；完整配置、原始向量及案例见新增实验记录。

### 指标与结果分析

- 主要报告 `Recall@k`、`MRR@10` 和 `nDCG@10`，并说明 MTEB dev 子集与 full T2Ranking dev 的差别。T2Ranking 官方 README 也报告 MRR@10、Recall@1/50；MTEB qrels 的 `score` 为二值 1，不包含原始重排 qrels 的 4 级相关性判断。[T2Ranking 指标与 qrels](https://github.com/THUIR/T2Ranking/)、[MTEB qrels 字段](https://huggingface.co/datasets/mteb/T2Retrieval)
- 不报告本数据集上的生成正确率，因为没有标准答案文本；只在可选演示中人工检查引用 passage 是否支持回答。
- 做失败案例分析：BM25 对精确名词/编号的命中、稠密检索的语义近邻误召回、RRF 对互补结果的融合、重排是否改善 top-10，以及 HTML/重复内容是否影响排序。把结果限制在该 MTEB dev 子集，不外推为全中文网页或课程资料结论。

## 实施顺序与成功标准

1. **已完成**：下载 MTEB dev 的 corpus/query/qrels 和原始 T2Ranking dev 的 query/qrels，仅用于 ID 对照；不下载 train 或 hard negatives。
2. **已完成**：核验 query、qrels、corpus ID 覆盖；本地 BM25S、FastEmbed 和 PyArrow 已安装，BGE-small 中文 ONNX 权重已下载并能在本机生成向量。
3. **已完成正式方法对比**：A0/A1/A2/A3/A5/A7/A8使用相同100条query、qrels和1,000段语料，返回20段，A1保持当前方案。
4. **已完成两组正式消融**：CLS/Mean/Max池化、固定点积下的文档L2开关；已保存平均与逐query指标、案例、原始向量和元数据。A4、A6和HTML消融不纳入正式清单。
5. 当前课程设计 PDF 只用于独立 PDF 解析烟测；不生成课程报告、不发送邮件。

正式内容为A0/A1/A2/A3/A5/A7/A8方法对比、BGE CLS/Mean/Max池化及文档L2消融；不生成课程报告、不发送邮件。结果只对应1,000段pilot，不代表完整MTEB/T2Ranking或课程资料检索表现。
