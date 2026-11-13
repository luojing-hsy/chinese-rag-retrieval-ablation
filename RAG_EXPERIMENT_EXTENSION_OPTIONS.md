# RAG 对比与消融的可选扩展

> 后续状态更新：用户已移除E5和HTML消融，只有TF-IDF继续正式保留。替代新增的BM25L、BM25+、Max池化和文档L2消融已完成，见 `results/t2retrieval/additional_ablation/SUMMARY.md`。下文是当时建议与历史运行依据，不能作为当前正式清单；以 `RAG_TECHNICAL_PLAN.md` 为准。

历史调研与运行日期：2026-11-13。当时批准并完成TF-IDF、E5和HTML实验；后续正式范围已由上方状态说明取代。下文保留当时的设计依据与结果，不再作为新增E5/HTML的建议。历史记录见 [扩展实验记录](results/t2retrieval/retrieval_extensions/SUMMARY.md)。

## 推荐只新增“两项对比 + 一项消融”

| 可选新增 | 类型 | 价值与新增成本 |
|---|---|---|
| TF-IDF词级检索 | 方法对比 | 补上不同于BM25的传统稀疏基线；仅本地统计和稀疏乘法 |
| multilingual-e5-small | 预训练编码器管线对比 | 检查结论是否依赖BGE；要下载并重编码1,100条文本，成本较高 |
| passage HTML去标签开关 | 单组件预处理消融 | query固定，只改变文档去标签步骤；仅重编码受影响passage即可 |

以上三项均已运行。TF-IDF分词、建索引和query处理约1.79秒；HTML仅重编码666段，约28.89秒；E5文档编码约210.72秒。E5的Recall@20/MRR略高于BGE、nDCG略低；保留HTML的nDCG略高、Recall@20相同，不预设改动必然提高质量。精确口径及完整阶段耗时见运行记录；原来按计算结构作出的成本判断不代替实测。

## 对比一：TF-IDF，先词级，字符n-gram作备选

scikit-learn官方TfidfVectorizer支持word、char与char_wb分析器、n-gram、稀疏输出和L2归一化，可用归一化向量点积得到余弦排序。[官方API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

首选沿用BM25的jieba分词及相同清洗，固定词级unigram，不新增停用词、词干化或词表筛选变量。词表和IDF只从同一1,000个passage统计，query只transform，不把query/qrels用于挑词；这是建索引统计，不训练或微调神经模型。中文不能直接依赖默认英文式token_pattern，把整段连续汉字当一个词。

若需要避免分词误差，可另设字符2–3 gram TF-IDF；它会增加词表和稀疏矩阵规模，但不需模型权重。word与char属于表示机制对比，不把它们的全部差异归因于某一个RAG组件，也不同时扫许多n范围。TF-IDF补充的是“不同词法权重”的证据，不保证胜过BM25。

## 对比二：一个代表性小型多语编码器

建议只考察`intfloat/multilingual-e5-small`：官方说明12层、384维，给出中文输入示例；推荐推理为attention-mask均值池化、L2归一化、最长512 token。非英文也必须使用`query: `和`passage: `前缀。上述是该模型的规定用法，不重新安排一次“查询增强实验”。[官方模型卡](https://huggingface.co/intfloat/multilingual-e5-small)

官方文件表的单个`model.safetensors`约471 MB，另需tokenizer/config；不要把仓库多格式总量2.28 GB当作必须下载量，也不要重复下载同内容的PyTorch/ONNX/OpenVINO权重。模型卡标MIT，可下载后显式离线加载，在CPU运行。当前FastEmbed内置模型表未提供该small模型，需要另接官方ONNX或Transformers推理，不能承诺即插即用。[官方文件表](https://huggingface.co/intfloat/multilingual-e5-small/tree/main)

固定100 query、1,000 passage、ID、清洗、输出Top-20与指标。但E5使用自身tokenizer、前缀和mean，BGE保持正式CLS设置，因此这是**预训练编码器推荐管线对比**，不是只换权重的单因素消融。模型体积小于一些大编码器不等于它在本机一定更快；需额外做1,100条编码，不能套用BGE历史46.48秒。没有证据保证其中文pilot成绩高于BGE。

## 消融：只开关passage HTML去标签

现有`clean_text`同时做HTML实体解码、标签替换、NFKC与空白规整，且用于query和passage。不要把整段函数开/关称为单一“去标签消融”。建议固定query处理及passage的其他步骤，仅将passage的标签替换步骤开/关：C0保留、C1去掉；模型、CLS池化、截断、相似度、ID和qrels完全一致。

本轮只读统计：按固定candidate_pool_doc_ids筛出的1,000条原始passage中，666条匹配现有标签正则`<[^>]*>`，说明不是空变量；这不代表已证明666条都是可安全删除的HTML，也不是新检索实验。实施前还需核对最终文本改变数与尖括号内容。两组同样保留现有unescape、NFKC、空白规则和query清洗。去标签后token化或截断内容改变是该处理的下游结果，不另外调整长度“补偿”。复用未变文档和query向量，只编码文本改变的文档；两组独立在完整同一池检索，不能冻结原Top-50。标签过滤可能误删内容，因此不能预设C1更好。

## 其他备选及不建议增加的变量

**BGE Top-20 + 已有reranker：**可以补一个“不同首阶段召回器加同一重排器”的方法对比，保持与A3相同的cross-encoder与Top-20预算。它需要重新评分新候选，不能挪用A3的BM25候选分数。旧A3的100×20评分耗时654秒，约10.9分钟且不含完整加载耗时，故不属于优先短时选项；新时长也不能直接等同654秒。Recall@20对同一BGE候选重排应不变。[模型官方说明](https://huggingface.co/BAAI/bge-reranker-base)

**mean排除CLS/SEP：**现正式mean排padding但含CLS/SEP，可另与仅正文token的mean对照，固定主干token输出、query/passage同样处理。tokenizer提供special_tokens_mask，可避免靠“删首尾位置”误处理padding。[官方tokenizer文档](https://huggingface.co/docs/transformers/main_classes/tokenizer)

这是现有池化消融的小扩展，内容增量有限；目前仅保存池化向量，不能从mean/CLS向量还原全部token表示，若未缓存主干输出仍需一次前向。它也不保证提高成绩。

**文档最大长度128/256/512：**固定query长度512、右截断、CLS、权重和其他配置，仅改变passage token预算；叫作**长度参数敏感性与质量/成本权衡**，不把三档都称为组件消融。记录各档实际被截文档数，短于128 token的文档可能相同；不同截断需重新前向，不能切已有句向量代替。[官方截断文档](https://huggingface.co/docs/transformers/main/pad_truncation)

**不推荐“L2归一化开关”：**当前缓存与FastEmbed输出已经归一化，删外层重复归一化不能制造有效处理。真正余弦会在评分时除范数，前置L2通常不改变排名；对固定query仅缩放query也不改变点积次序。若改为未归一化文档的内积，实际改变向量范数作用和评分口径，需原始向量，不能称为当前余弦管线中直接有效的单开关。

后续已按新的明确设计实现文档L2消融：从token输出保存原始CLS，在两组均固定单位query和同一矩阵点积，只开关文档L2。它不是上述“删重复归一化”的无效操作；当前正式解释严格限定于固定点积下文档范数是否参与评分。

所有建议继续只评检索Recall/MRR/nDCG，报告pilot口径与逐query变化，不调用生成模型或外部评审；无增益也是可解释结果。
