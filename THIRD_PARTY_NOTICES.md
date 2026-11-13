# 第三方来源与许可说明

## 数据与派生文本

本仓库中的 query 文本、qrels 和 passage ID 来自 [mteb/T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval)，原始来源为 [THUIR/T2Ranking](https://github.com/THUIR/T2Ranking)。原记录声明数据采用 Apache-2.0；许可全文附于 [data/LICENSE-T2Ranking.txt](data/LICENSE-T2Ranking.txt)。数据的来源归属随上游信息保留，本项目不主张这些文本的原创权。

本项目对文本执行 HTML 实体解码、去标签、NFKC 和空白规整，并抽样、计算排名、指标及案例差值。包含网页正文的案例、预处理全文和词表仅保存在本地，不随仓库上传；仓库保留排名 ID、分数和指标等派生实验记录。MTEB 提示可能包含 MMTEB 的额外处理；本项目核验了 query/qrels/passage ID 覆盖，未对原始 T2Ranking 全库文本做全量哈希对照。

原始 parquet 未随仓库分发。原文件 SHA-256、样本数及运行时间保存在 `results/t2retrieval/pilot/run_metadata.json`；历史下载未记录数据集 revision 和独立下载时间，不能追认当前 revision 为原下载版本。2026-11-12 的初始运行已使用这些文件。重新下载应核对原 SHA-256；不匹配时应作为新版本记录，不能声称精确复现。

## 模型与软件

本仓库不分发模型权重和第三方库源码；以已安装版本的公开 API 调用它们。依赖版本见 `requirements-experiment.txt`，实际环境和模型文件哈希见各实验 `run_metadata.json`。使用和另行分发时，应保留各上游安装包、模型卡及其许可文件。

- [FastEmbed](https://github.com/qdrant/fastembed)：0.9.1，安装包元数据声明 Apache-2.0。
- [BM25S](https://github.com/xhluca/bm25s)：0.3.16，安装包元数据声明 MIT。
- [BGE-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)：使用 Qdrant ONNX snapshot `46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59`。
- [BGE-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)：snapshot `2cfc18c9415c912f9d8155881c133215df768a70`。
- [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)：仅归档历史管线，revision 和 MIT 声明保存在 `retrieval_extensions/e5/run_metadata.json`。
- 其余依赖包括 jieba、NumPy、ONNX Runtime、PyArrow、scikit-learn、SciPy、tokenizers 和 huggingface-hub；以各固定版本安装包的许可及附加声明为准。

未克隆或复制完整 RAG 框架源码；方案中提到的 FlashRAG、Haystack、LlamaIndex 是历史选型调研对象。此处的数据许可不自动赋予本项目代码其他许可。
