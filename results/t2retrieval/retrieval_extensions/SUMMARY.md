# 历史扩展结果：TF-IDF保留，E5与HTML归档

用户后续已将E5和HTML消融移出正式清单，下表保留当时真实结果，不作为当前方法列表。A5 TF-IDF继续保留；新替代方法与消融见 [新增实验记录](../additional_ablation/SUMMARY.md)，正式方案以 `RAG_TECHNICAL_PLAN.md` 为准。

2026-11-13已完成。固定原100条query、1,000个passage及qrels，统一返回20段。A1继续作为当前方案；新增A5为TF-IDF，A6为E5，另增加HTML去标签消融。不恢复A4或其他归档探索。

| 实验角色 | 配置 | Recall@20 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|
| A1当前方案 | BGE-CLS，去标签开启 | 0.958444 | 0.962000 | 0.947584 |
| A5对比 | jieba词级TF-IDF | 0.895152 | 0.918095 | 0.857271 |
| A6对比 | multilingual-e5-small官方管线 | 0.962294 | 0.965000 | 0.943870 |
| HTML消融 | 同一BGE-CLS，去标签关闭 | 0.958444 | 0.962500 | 0.949907 |

E5的Recall@20/MRR略高，但nDCG@10略低于A1；不能声称单一方法在所有指标上占优。HTML保留标签时的微小前排优势也不说明删除标签有益或保留标签普遍更好。没有统计显著性检验，不自动更改当前方案。TF-IDF作为独立方法对比保留，不是BGE改进尝试。

两项对比使用各自固定检索管线；HTML消融仅改变passage标签替换，query、BGE权重、CLS池化、其他清洗、归一化和评分固定。所有方法从同一全池检索，不锁定旧Top-50。实际去标签开关使666段文本改变，334段及query缓存可复用。

实测主要计算：TF-IDF分词/索引/query处理合计约1.79秒；HTML重编码约28.89秒；E5文档编码约210.72秒、query编码0.64秒。局部计算、阶段总时长、共享数据读取及下载时间不能混为同一口径；均为单次运行记录，不作严格性能基准。

完整配置、逐query指标、配对变化、案例、校验和精确计时：

- [TF-IDF与HTML实验](local/SUMMARY.md)，[指标](local/metrics.csv)，[元数据](local/run_metadata.json)。
- [E5实验](e5/SUMMARY.md)，[指标](e5/metrics.csv)，[元数据](e5/run_metadata.json)。

已执行 `src/run_retrieval_extensions.py --stage local`、`src/download_e5_model.py`、`src/run_retrieval_extensions.py --stage e5`，均成功退出。基线Top-20 ID与六项指标复现，输入hash和标注/ID覆盖校验一致。没有新增训练、外部模型推理或独立测试套件。

新依赖已锁定版本；E5来自官方仓库的FP32 ONNX，revision与所有文件hash记录在元数据。仅下载公开权重，推理在CPU本地运行。结果只对应MTEB/T2Retrieval dev派生pilot，不是完整MTEB/T2Ranking成绩，不报告生成准确率。
