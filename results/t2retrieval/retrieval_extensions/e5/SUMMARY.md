# 归档：E5-small 本地编码器管线对比

用户已将E5移出当前正式对比清单，本文件只保留历史运行记录，不恢复其为主实验。

日期：2026-11-13。数据沿用MTEB/T2Retrieval dev派生pilot的100条query、同一1,000个passage和qrels；最终均返回20段。E5作为方法对比，不作为BGE的单组件消融。

## 配置与来源

使用官方 `intfloat/multilingual-e5-small` 的未量化FP32 `onnx/model.onnx`，revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`。模型文件470,268,510字节，许可为上游模型卡声明的MIT。权重、tokenizer、配置和模型卡由下载脚本取得，文件hash和获取时间保存于 `models/multilingual-e5-small/model_provenance.json`，并完整嵌入本次 `run_metadata.json`。[官方模型卡](https://huggingface.co/intfloat/multilingual-e5-small)、[官方ONNX文件目录](https://huggingface.co/intfloat/multilingual-e5-small/tree/main/onnx)

按官方用法：query加 `query: `，passage加 `passage: `；使用自身tokenizer、最大512 token右截断、attention-mask mean pooling、L2归一化和余弦排序。Mean包含非padding的特殊token；padding为 `<pad>`、ID=1。该前缀是E5规定输入格式，不另称一次查询增强实验。模型输出为最后一层token表示，384维。

BGE保持A1原配置，包括原tokenizer、无检索前缀、CLS和512-token上限。两种模型均用原始清洗后的同一passage池和query，独立精确检索Top-20。因此比较的是**各自官方配置下的编码器管线**，不能把分数差异单独归因于模型权重、维度或池化。使用ONNX Runtime CPU、4线程，文档和query批大小均为32。

## 实测结果

| 方法 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|
| A1 BGE-small-zh CLS | 0.801891 | 0.933719 | 0.958444 | 0.962000 | 0.947584 | 0.473000 |
| A6 E5-small | 0.800796 | 0.924195 | 0.962294 | 0.965000 | 0.943870 | 0.468000 |

E5相对BGE的Recall@20增加约0.385个百分点，MRR@10增加0.300个百分点，但nDCG@10下降约0.371个百分点，Recall@10也较低。结果是指标间取舍，不能声称BGE在正式对比中所有指标都最好，也不能声称E5整体胜出。样本较小且没有统计显著性检验，保留用户指定的A1主线，不自动更换最终方案。

| 指标 | E5改善 | 相同 | E5下降 |
|---|---:|---:|---:|
| Recall@5 | 3 | 94 | 3 |
| Recall@10 | 1 | 93 | 6 |
| Recall@20 | 3 | 94 | 3 |
| MRR@10 | 1 | 99 | 0 |
| nDCG@10 | 9 | 83 | 8 |

## 案例

- qid=2364，“吃了一些水果之后头晕眼花大出汗”：E5把相关passage `3496`、`676956` 新召回到第11和16名，Recall@20比BGE提高1/3；但原BGE第2和3名的相关段落在E5降至第12和8名，nDCG@10下降0.246762。这说明召回更多不等于前10名排列更好。
- qid=9160，“12306同一天不同时间段的候补票可以换吗”：BGE原第4名的相关passage `183705` 在E5 Top-20未命中，Recall@20下降0.25、nDCG@10下降0.168128。
- qid=408，“仙水有多强”：相关passage `402881` 从BGE第5名升到E5第2名，nDCG@10增加0.244077，Recall@20不变。

这里只分析标注与排序，不评判网页陈述真实性。完整案例与逐query差值分别见 `case_analysis.jsonl`、`paired_deltas.csv`。

## 耗时与验证

本次模型/tokenizer加载1.227940秒；文档编码210.722932秒，query编码0.639275秒，全池检索0.005197秒；E5阶段主体约214.20秒，另有共享数据读取校验7.29秒。阶段时间不含权重下载、Python启动/导入和历史BGE向量构建；不是多次重复测量的严格时延基准。E5与历史BGE的模型、批大小及输入tokenizer不同，不把耗时比直接解释成单因素加速/减速。

全部推理使用CPUExecutionProvider，没有远端模型调用或训练。BGE基线Top-20 ID和六项指标复现旧结果，原始数据hash、qrels、ID映射和缓存向量hash校验通过；E5的masked mean与直接非padding均值一致，向量有限且单位范数，输出每query均为池内20个唯一ID。有效token数含前缀/特殊token：文档7–512，query 8–22。没有添加或运行独立测试套件。

## 复现

首次只下载公开文件：

```powershell
.\.venv_experiment\Scripts\python.exe -u src/download_e5_model.py
```

本地离线推理：

```powershell
.\.venv_experiment\Scripts\python.exe -u src/run_retrieval_extensions.py --stage e5
```

本次两个命令均成功退出。下载脚本已固定本次revision，后续复用已下载文件，新的工作区也下载同一版本；实验脚本只读本地文件。环境版本与源文件hash记录于 `run_metadata.json`。结果文件包括固定配置、Top-20分数/次序、平均和逐query指标、配对变化与案例。query/document向量及ID映射位于 `data/processed/t2retrieval/pilot/retrieval_extensions/`。

仍沿用pilot候选池和未标注按非相关评价的限制，不是完整MTEB/T2Ranking成绩；无标准答案文本，不报告生成正确率。
