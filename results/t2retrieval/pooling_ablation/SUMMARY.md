# 正式消融：固定 BGE 权重的池化策略

日期：2026-11-13。MTEB/T2Retrieval dev 派生pilot，100条query、1,000个parent passage，最终返回20段。只做本地检索，不训练、不调用外部API或生成模型。

后续已将同一研究问题扩展到Max池化，并增加文档L2消融，见 [新增实验记录](../additional_ablation/SUMMARY.md)。本文件保留当次CLS/Mean两组的真实配置和计时，不把后续计算时间混入。

## 设计与解释范围

固定本地 `BAAI/bge-small-zh-v1.5` ONNX权重、tokenizer、文本清理、原query、输入最大512 token、批大小、CPU线程、L2归一化、余弦评分和qrels。唯一变化是从最后一层token表示提取句向量的方式；两组对query和passage一致应用各自规则。

- **CLS**：取最后一层第一个token的512维表示，再L2归一化；这是 [BGE官方用法](https://huggingface.co/BAAI/bge-small-zh-v1.5#using-huggingface-transformers)。
- **Mean**：按attention mask对最后一层有效token求均值，再L2归一化。padding不计入，CLS/SEP等attention mask为1的特殊token计入。该定义在首次运行前固定，没有试验其他mean变种。

本地ONNX输出为 `last_hidden_state[batch, sequence_length, 512]`，两组使用同一批次的同一份token输出，主干只计算一次。不更换模型，不做查询扩展，不改变长度；也不锁定原BGE Top-50候选。每组都在同一1,000段库中独立计算余弦并检索Top-20，才能观察表示变化对召回的影响。此前MMR/Dartboard实验的“50候选选20”定义保持原样。

本项目将这组实验作为**冻结预训练模型的推理阶段池化策略消融**：CLS为A1原始配置，mean为消融变体。它不能证明训练时CLS优于mean，也不是重新训练的架构或训练损失消融；结论仅适用于当前权重、数据和配置。

## 实测结果

| 方法 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|
| CLS | 0.801891 | 0.933719 | 0.958444 | 0.962000 | 0.947584 | 0.473000 |
| Mean | 0.794058 | 0.921473 | 0.948985 | 0.955833 | 0.930852 | 0.463000 |
| Mean − CLS | -0.007833 | -0.012246 | -0.009460 | -0.006167 | -0.016732 | -0.010000 |

Mean的Recall@20下降约0.946个百分点，nDCG@10下降约1.673个百分点；本轮继续保留CLS和原A1方案。并未进行统计显著性检验，不能声称差异在其他样本上稳定或普遍成立。

| 指标 | Mean改善 | 相同 | Mean下降 |
|---|---:|---:|---:|
| Recall@5 | 0 | 97 | 3 |
| Recall@10 | 1 | 88 | 11 |
| Recall@20 | 0 | 94 | 6 |
| MRR@10 | 1 | 98 | 1 |
| nDCG@10 | 4 | 73 | 23 |

## 案例

以下分析标注和排名，不判定网页内容真实性。

1. **下降，qid=22821，“公墓十七排还是十八排”**：相关passage `304045` 从CLS第1名降到mean第3名；MRR@10由1降到1/3，nDCG@10下降0.5，Recall@20不变。说明池化变化可以损害前排排序，即使相关段落仍被检索到。
2. **下降，qid=9636，“坎地沙坦和倍他乐克”**：相关passage `98394` 从CLS第6名被移出mean的Top-20；nDCG@10下降0.139056，Recall@20下降0.25。该案例显示变化也会影响候选召回，不只是原集合的换序。
3. **改善，qid=4054，“暗黑之地玩法”**：相关passage `646888` 从CLS第11名升到mean第8名；nDCG@10增加0.148041，Recall@20不变。单个改善不代表总体更优。

两组完整原文及排名见 `case_analysis.jsonl`，全部逐query变化见 `paired_deltas.csv`。当前结果与“官方CLS提取规则更适配这份固定权重”相符，但没有单独分析训练机理，不能据此归因于训练阶段某个组件。

## 校验与时间

- CLS文档向量与原缓存的最大绝对误差：`5.960464477539063e-08`，小于预设1e-5容差；全部100条query的Top-20 ID与旧记录完全一致，六项平均指标也一致。
- Mean的attention-mask加权结果与直接取非padding位置求均值一致；向量有限且L2范数为1。
- 原corpus/query/qrels的hash、query和passage ID、原标注、输出唯一性均通过内置校验。有效token数（含特殊token）：passage为6–512，query为7–21。
- 主干前向共享：1,000段文档42.796963秒，100条query 0.143237秒；同一次前向同时服务两种池化。
- CLS池化及归一化合计约0.002410秒，mean约0.372770秒；全库检索分别约0.013077和0.009785秒。这是一次运行的局部计时，不是严格时延基准。
- 数据读取与校验约5.815518秒，模型加载0.214934秒；脚本主体到元数据写入前约50.20秒，含向量保存、结果和案例输出，不含Python启动/导入。不能将共享前向拆成两份独立端到端耗时进行比较。

## 复现与产物

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_pooling_ablation.py
```

本次退出码0；没有添加或运行独立测试套件。权重使用已有本地snapshot `46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59`，无新增下载、网络调用或训练。

- `experiment_config.json`：池化定义及全部固定设置。
- `retrievals_bge_cls.jsonl`、`retrievals_bge_mean.jsonl`：逐query Top-20 ID、余弦分数与次序。
- `metrics.csv`、`per_query_metrics.csv`、`paired_deltas.csv`、`paired_summary.json`：平均指标、逐query结果、变化与计数。
- `case_analysis.jsonl`：改善/下降案例的原文与完整20名。
- `run_metadata.json`：原实验来源、权重/tokenizer/输入/脚本hash、环境、有效token长度和分阶段计时。
- `data/processed/t2retrieval/pilot/pooling_ablation/`：两种池化的文档和query向量及ID映射，供复算使用；不覆盖原缓存。

数据仍是保留全部相关段落并补随机未标注项的小候选池，结果不是官方完整MTEB/T2Ranking成绩；未标注项可能存在真实相关内容。没有标准答案文本，不报告生成正确率。
