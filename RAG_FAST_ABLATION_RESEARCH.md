# 短耗时 RAG 检索消融：排除已做技术后的建议

调研与修订日期：2026-11-13。

> 当前正式主线：A1为当前方案，A0/A2/A3/A5（TF-IDF）/A7（BM25L）/A8（BM25+）作对比；CLS/Mean/Max池化和固定点积下的文档L2作消融，均已完成。A4、A6 E5、HTML消融及MMR/Dartboard不纳入主表。本文其余内容仅作历史档案，正式清单以 `RAG_TECHNICAL_PLAN.md` 为准。

> **用户已明确确认：本项目做过 PRF 和查询增强。以用户陈述为准，排除这两类；此前“Dense PRF + 查询指令 2×2”的推荐撤回。当前目录不代表全部项目历史，不能因缺记录推断用户没有做过。**

本文记录来源调研和实验建议；2026-11-13 已按用户最终决定完成证据选择实验：候选保留 50，最终输出统一为 20。已新增独立脚本并运行，结果见 [实验记录](results/t2retrieval/evidence_selection_top50/SUMMARY.md)。没有生成课程报告；继续限定本地推理、不训练、不调用外部模型 API。

本轮实际结果：原始 BGE、MMR（gamma=0.25）、Dartboard（sigma=0.1）的 Recall@20 分别为 0.958444、0.953087、0.912375，nDCG@10 分别为 0.947584、0.939112、0.845293。后两者降低了前10段的平均成对向量余弦，但没有改善检索质量，不替换现有 A1。Top-50 的平均 Recall@50 为 0.963333，限制了从这些候选中提升 Recall@20 的空间；原 A0/A3 的固定20候选重排实验仍维持原定义。

同日又按用户要求完成BGE池化方式对比：固定模型权重，CLS与attention-mask mean共享主干输出、各自编码query和passage，在同一1,000段库独立检索Top-20。Mean包含CLS/SEP、排除padding，归一化与余弦评分不变。CLS复现原基线；mean的Recall@20为0.948985、nDCG@10为0.930852，均低于CLS。脚本主体约50.20秒；这是推理表示策略对比，不能解释为训练阶段组件消融。详见 [池化实验记录](results/t2retrieval/pooling_ablation/SUMMARY.md)。

## 优先选择

**本轮已运行“证据集合选择”：固定同一 BGE Top-50 parent 候选，对比普通相关性选择、MMR 去冗余与纯向量 Dartboard。**其中 MMR 惩罚开关构成单组件消融；KNN/MMR/Dartboard 三者整体是选择机制对比，不能混称单组件消融。原始候选文件实际含50名，分数和文档向量直接复用，没有模型加载、查询编码或额外模型推理。

| 方向 | 新增计算与成本判断 | 消融/对比问题 | 主要限制 |
|---|---|---|---|
| MMR 去冗余开关 | 缓存向量的候选间相似度与贪心选择 | 去冗余是否改善证据选择？ | qrels 无子主题标签；可能降低相关性指标 |
| Dartboard 纯余弦版本 | 同一候选的距离矩阵、高斯核与贪心选择 | 信息增益目标与 MMR 的选择行为有何不同？ | 有超参数；属于机制对比；本机耗时未测 |
| 固定句块的 max / mean / top-2 mean 聚合 | 同一子向量分数矩阵算一次，后续仅聚合 | 局部最强信号还是多句支持更适合 parent 相关性？ | 子向量若未保存要重算一次；可能与已做粒度实验接近 |
| 固定两路分数的归一化开关 | 已有分数缩放、加法、排序 | 分数尺度是否影响 BM25+dense 融合？ | 必须固定候选、缺失处理、权重和返回深度 |

上表原为按计算结构作出的成本判断。现已测得100条query的MMR选择合计约0.0077秒，Dartboard选择约0.1530秒，另有三组可共享的候选间相似度计算约0.0079秒；这些是缓存后的局部计算耗时，不含历史向量构建和首次检索，不作为严格时延基准。聚合与归一化仍为未执行备选。

## 1. MMR：最清楚的单组件开关

MMR 同时考虑 query 相关性和相对于已选文档的冗余。PyTerrier 官方提供直接使用已有 doc_vec 和相关性分数的重排，不要求新编码器或生成模型。[原始论文 DOI](https://doi.org/10.1145/290941.291025)、[PyTerrier 官方文档](https://pyterrier.readthedocs.io/en/stable/ext/pyterrier-dr/diversity.html)

对同一个 BGE Top-50 候选，固定 relevance 为 query-passage 余弦分数，固定冗余为 passage 向量余弦：

`score(d) = relevance(q,d) - gamma * max_similarity(d, selected)`。

- M0：gamma=0，关闭冗余惩罚，按相关性选 20 个。
- M1：预先固定 gamma=0.25，开启惩罚。该设定与常见 lambda=0.8 的 MMR 形式排名等价；它是探索设置，不能宣称本任务最优。

只改惩罚开关，其余候选、向量、相关性定义、初始选择与最终深度不变。这是 Top-50 中选择 Top-20，Recall@20 可以改变；若只重排全部同一 Top-20，Recall@20 才必须保持一致。当前 passage 向量和候选分数可复用，无需额外神经推理。

MMR 优化目标含多样性，不保证 nDCG 上升。主表继续报告 Recall/MRR/nDCG；可补充平均成对余弦观察向量冗余，但不能称为事实覆盖或答案正确率。数据没有子主题标注，不直接使用需要该标注的多样性评价。

## 2. Dartboard：更有区别的轻量机制对比

Dartboard 基于相关信息增益选择证据集合；纯余弦版本先选 KNN 候选，再用 query-document 和 document-document 距离、高斯核及贪心选择。作者论文同时包含 cross-encoder/hybrid 变体，本项目仅考虑 D-CS 纯向量版本。[原始论文](https://arxiv.org/html/2407.12101v2)、[作者实现](https://github.com/EmergenceAI/dartboard/blob/main/dartboard.py)

已核对作者源码：`get_dists_cossim` 接收编码回调；`scaled_cosdist`、`lognorm`、`greedy_dartsearch` 的计算可直接接现有缓存向量，无须 LLM。整个模块另含默认 SentenceTransformer 下载及 CUDA cross-encoder 路径，不能照搬整脚本默认执行；只借鉴纯向量部分或显式传已有向量，复用代码前核验仓库许可证并锁定提交号。

固定每 query 候选数 K=50，输出20，单 query 的候选矩阵仅50×50；100 query 共250,000个候选对相似度，每对仍需512维点积，另有贪心选择。上述本机计时仅覆盖此次小样本缓存计算。论文明确没有系统研究运行时，最坏复杂度随候选数呈平方增长。其性能基于其他数据、嵌入和调参，不能移植为本项目保证。[论文方法与局限](https://arxiv.org/html/2407.12101v2)

本轮将 D-CS 与 M0/M1 做机制比较，保存贪心选择次序用于排名指标。Dartboard 的 sigma 预先固定为0.1，没有扫描或按本次成绩修改；该值是探索配置，不代表任务最优。若后续调参，另用调参 query，不在最终评测 query 上挑选最好成绩。它不是无参数算法。

## 3. 固定句块，只改变 parent 聚合

Zhang、Yates 与 Lin 的 ECIR 2021 原论文比较 FirstP、MaxP、SumP、AvgP，支持将“如何把子段分数变为文档分数”作为独立研究变量。其模型和训练设置不同，本项目只借鉴无训练的聚合运算，不搬用其结果。[原论文](https://cs.uwaterloo.ca/~jimmylin/publications/ZhangXinyu_etal_ECIR2021.pdf)

固定当前 A4 的3,374个句块、parent映射、BGE模型、query表示、L2归一化、余弦分数及全部1,000个parent，使用同一完整query×child分数矩阵：

- G0 Max：每个parent取最高子块分数。
- G1 Mean：每个parent取全部子块分数均值。
- 可选G2 Top-2 mean：取最高min(2,子块数)个分数均值；这是本项目预先指定的折中对照，不宣称是原论文提出的最优规则。

这是聚合机制替换的受控对比，与已做的“整段 vs 切块+max”组合处理不同；不能称为移除了聚合。不得各自先截child Top-k再把留下的分数当成全部子块求均值，否则混入截断变量。最后仍按原parent passage qrels评价。

max可能受单个偶然高分影响，mean可能稀释局部强证据；哪个更好必须实测。单子块parent三组完全一致，可按子块数分析差异。

本目录可见A4脚本只把子向量存在内存并保存parent max排名，未见完整子向量/分数矩阵落盘；这不排除用户其他记录已有缓存。若需重建，旧A4记录的句块编码77.813秒仅是历史参考，不是新运行保证。只靠parent max分数不能还原mean，因此不能承诺三个规则都只需读旧排名。

## 4. 固定融合条件，只开关归一化

Bruch、Gai与Ingber的原研究比较分数凸组合和RRF，并分析参数与归一化。其研究含调参结果，不能宣称固定等权在本项目也必胜。[原论文](https://arxiv.org/abs/2210.11934)

OpenSearch官方区分rank-based RRF与score-based normalization，给出min-max、L2计算，并指出返回深度影响归一化统计。[官方归一化文档](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/)、[官方RRF对比](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/rrf/)

固定 `score(d)=0.5*BM25(d)+0.5*dense(d)`：

- F0：原始两路分数，归一化关闭。
- F1：每query、每路先min-max归一化，再相同等权相加。
- 可选F2：仅将min-max替换为L2，属于归一化机制替换。

固定分词、BM25参数、dense模型、候选池、返回深度、缺失处理、最终Top-20与权重；F0是诊断分数尺度的对照。哪一路主导须检查分布，不提前断言。

最干净的是在同一个1,000-parent池取得两路完整分数；若仅用两路Top-50并集，须预先固定“未返回按零贡献”和只使用返回项统计的规则，两种口径不混报。核对旧JSONL实际深度，不能仅凭metadata推断。

已有RRF可列为另一项融合机制对比，不能把RRF→分数融合叫作只移除归一化；权重和RRF常数扫参归为参数分析。

## 实施与解释边界

当前主线保留A1及CLS/Mean/Max池化、文档L2消融；A0/A2/A3/A5/A7/A8列为对比。此前证据选择与聚合只作历史调研，不恢复A4、E5或HTML消融。

每组保存配置、候选IDs、原始分数、选择次序、最终parent排名、逐query指标和计时。模型加载、向量编码、矩阵计算/聚合、选择及指标计算分别记录，区分冷启动与缓存复用。

成功标准是变量清楚、结果可复现、成本可核对，包含无增益或负增益。当前小池上BGE nDCG@10约0.9476，新增组件不保证显著提升；不要为制造增益筛query或反复挑最好测试设置。继续保留既有结果与限制，不因目录缺记录再次推荐用户确认做过的技术。
