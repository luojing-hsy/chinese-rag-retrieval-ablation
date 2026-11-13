# TF-IDF 对比与 HTML 去标签消融

后续状态：A5 TF-IDF继续保留，HTML消融已按用户要求归档；本文件保留当次配置和数值，不恢复HTML为正式实验。

日期：2026-11-13。沿用MTEB/T2Retrieval dev派生pilot的100条query、1,000个passage、同一qrels，统一返回20段。只做本地检索，不训练神经模型或调用外部推理API。

## 固定配置

**TF-IDF方法对比**：沿用BM25的jieba `HMM=False`分词、转小写及word正则筛选；词级unigram，平滑IDF、普通TF、L2归一化，余弦排序，无停用词或词表截断。只用1,000个passage建立词表和IDF，query只transform，不使用qrels挑词。实际词表37,877项，文档矩阵186,537个非零元素。并列分数按原passage池次序处理。[scikit-learn官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

**HTML去标签消融**：以A1的BGE-CLS为对照，只关闭passage的 `<[^>]*>` 标签替换步骤；query处理、两次HTML实体解码、NFKC、空白规整、最大512 token、模型权重、L2归一化和余弦评分不变。666段的最终输入实际改变并重新编码，其余334段及全部query向量从已校验的CLS缓存复用。两组分别检索同一完整1,000段池，不冻结原Top-20/50。

模型是否见到标签、以及由此带来的token化与截断变化，是该单一预处理开关的下游效果；不能把差异进一步拆成这些因素的独立贡献。标签正则不等于经过完整HTML语义解析，也不能预设删除一定有利。

## 实测结果

| 方法 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|
| A1 BGE-CLS，去标签开启 | 0.801891 | 0.933719 | 0.958444 | 0.962000 | 0.947584 | 0.473000 |
| TF-IDF词级 | 0.715410 | 0.850658 | 0.895152 | 0.918095 | 0.857271 | 0.426000 |
| BGE-CLS，去标签关闭 | 0.803891 | 0.937481 | 0.958444 | 0.962500 | 0.949907 | 0.474000 |

TF-IDF在该pilot上低于BGE，也略低于原BM25的Recall@20=0.903432、nDCG@10=0.874798。它作为独立传统词法基线保留，不是BGE改进变体。

HTML去标签关闭相对开启：nDCG@10增加0.002323、Recall@10增加0.003762、MRR@10增加0.000500；Recall@20在全部100条query均相同。Recall@20相同是本次观测结果，不是设计强制，因为两组均可在全池独立召回。未做统计显著性检验，不能据小幅平均差异宣称保留标签普遍更好；也不能反向宣称本次证明去标签有益。不据此自动修改用户确定的A1默认配置。

| 相对A1 | nDCG@10 改善/相同/下降 | Recall@20 改善/相同/下降 |
|---|---|---|
| TF-IDF | 5 / 52 / 43 | 1 / 75 / 24 |
| 保留HTML标签 | 5 / 92 / 3 | 0 / 100 / 0 |

## 案例观察

- TF-IDF下降：qid=19303，“中国军装苏联像吗”，BGE第1名命中相关passage `597998`，TF-IDF的Top-20未命中，Recall@20和nDCG@10均下降1。该例符合纯词法方法可能漏掉语义相关证据的现象，不据单例认定具体词汇失配原因。
- TF-IDF改善：qid=408，“仙水有多强”，相关passage `402881` 从BGE第5名升到TF-IDF第1名，nDCG@10增加0.613147，Recall@20不变。
- 保留标签下降：qid=16803，“ps中蒙版的作用以及使用方法”，相关passage `358372` 从基线第10降到第12，nDCG@10下降0.063621，Recall@20不变。
- 保留标签改善：qid=4054，“暗黑之地玩法”，相关passage `646888` 从第11升到第8，nDCG@10增加0.148041，Recall@20不变。

`case_analysis.jsonl`展示同一passage的规范化证据原文及排名；HTML两组真正送入模型的不同文本完整保存在 `passage_preprocessing.jsonl`，不能把案例展示文本误当作保留标签变体的实际输入。

## 运行成本与复现

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_retrieval_extensions.py --stage local
```

本次退出码0。TF-IDF分词约1.509920秒、建索引0.267121秒、query转换及检索0.008226秒；BGE模型加载0.199597秒，666段重编码28.891738秒，全池检索0.008405秒。阶段主体约54.60秒，另有共享数据读取校验5.30秒与基线缓存检索0.009179秒；阶段主体还包括首次库导入、文本/稀疏矩阵读取写入及结果输出。计时不含依赖安装、进程启动或历史A1编码，不是重复测量的严格时延基准。

基线Top-20 ID及六项平均指标复现A1；原始文件hash、query/qrels/passage ID、缓存向量hash均校验一致；20个输出ID唯一且来自同一池。未添加或运行独立测试套件。

`experiment_config.json`记录预先固定配置，`metrics.csv`、`per_query_metrics.csv`、`paired_deltas.csv`和`paired_summary.json`保存均值与逐query变化，`retrievals_*.jsonl`保存原始排序，`tfidf_vocabulary.json`保存词表，`run_metadata.json`保存版本、来源和计时。IDF、稀疏矩阵、保留HTML的文档向量位于 `data/processed/t2retrieval/pilot/retrieval_extensions/`。

本结果不是完整MTEB/T2Ranking成绩；未标注项按非相关评价但不等于确定无关，没有标准答案文本，不报告生成正确率。
