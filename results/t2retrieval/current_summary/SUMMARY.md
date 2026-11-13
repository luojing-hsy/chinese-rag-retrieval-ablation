# 当前正式实验：完整参数与 @5/@10/@20 指标

日期：2026-11-13。指标仅从已保存Top-20排名重新统计；另做无神经模型推理的RRF输入深度审计，没有重新编码。

正式方法为A0/A1/A2/A3/A5/A7/A8，消融为CLS/Mean/Max和文档L2；A1是两组消融的共同对照。已移除的A4、E5、HTML及其他归档探索不纳入。

## 公共设置

100个问题、1,000段语料，query seed=2026、候选补样seed=2027；512个相关passage并集加488个随机未标注项；统一输出20段。只评检索，不训练、不调用外部模型API。qrels为二值相关性，未标注按非相关，指标按100个query宏平均。

文本预处理为HTML实体解码→标签替换为空格→再次解码→NFKC→空白规整；无查询指令、无切块。词法分词为jieba HMM=False、转小写、word正则过滤、无停用词。

BGE固定BAAI/bge-small-zh-v1.5，snapshot 46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59，512维、最大512 token、右截断，CPU4线程，文档batch64/query batch256。重排snapshot为2cfc18c9415c912f9d8155881c133215df768a70，pair最大512 token、batch32、CPU4线程，按ONNX原始分数降序。

## 方法参数

```json
{
  "A0": {
    "implementation": "bm25s==0.3.16",
    "method": "lucene",
    "idf_method": "lucene",
    "k1": 1.5,
    "b": 0.75,
    "delta": "not used",
    "dtype": "float32",
    "retrieval_depth": 100,
    "output_k": 20
  },
  "A1": {
    "model": "BAAI/bge-small-zh-v1.5",
    "pooling": "CLS",
    "query_l2": true,
    "document_l2": true,
    "score": "unit query dot unit passage = cosine",
    "exact_search_pool": 1000,
    "output_k": 20
  },
  "A2": {
    "inputs": [
      "A0",
      "A1"
    ],
    "input_depth_each": 100,
    "rrf_constant": 60,
    "weights": [
      1,
      1
    ],
    "formula": "sum(1/(60+rank)), rank starts at 1; absent item contributes 0",
    "output_k": 20,
    "audit": "recomputed depth100 matches all100 saved rankings; depth50 matches9/100; saved depth50 is not fusion input depth"
  },
  "A3": {
    "initial_candidates": "same original BM25 top20 IDs",
    "reranker": "BAAI/bge-reranker-base",
    "snapshot": "2cfc18c9415c912f9d8155881c133215df768a70",
    "max_pair_tokens": 512,
    "truncation": "right, default longest_first",
    "batch_size": 32,
    "threads": 4,
    "device": "CPU",
    "scoring": "first ONNX output column; descending raw scores, no sigmoid",
    "tie_break": "original BM25 order",
    "output_k": 20
  },
  "A5": {
    "implementation": "scikit-learn==1.9.1 TfidfVectorizer",
    "ngram_range": [
      1,
      1
    ],
    "tokenizer": "str.split after shared jieba/regex preprocessing",
    "token_pattern": null,
    "lowercase": false,
    "stop_words": null,
    "use_idf": true,
    "smooth_idf": true,
    "sublinear_tf": false,
    "binary": false,
    "norm": "l2",
    "min_df": 1,
    "max_df": 1.0,
    "max_features": null,
    "dtype": "float64",
    "fit": "passages only; queries transformed only",
    "vocabulary_size": 37877,
    "scoring": "L2 TF-IDF dot product",
    "tie_break": "stable original pool order",
    "output_k": 20
  },
  "A7": {
    "implementation": "bm25s==0.3.16",
    "method": "bm25l",
    "idf_method": "bm25l",
    "k1": 1.5,
    "b": 0.75,
    "delta": 0.5,
    "dtype": "float32",
    "retrieval_depth": 100,
    "output_k": 20
  },
  "A8": {
    "implementation": "bm25s==0.3.16",
    "method": "bm25+",
    "idf_method": "bm25+",
    "k1": 1.5,
    "b": 0.75,
    "delta": 0.5,
    "dtype": "float32",
    "retrieval_depth": 100,
    "output_k": 20
  }
}
```

## 消融参数

```json
{
  "pooling": {
    "control": "A1",
    "P-Mean": "attention-mask mean; exclude padding, include valid CLS/SEP; L2 query and passage",
    "P-Max": "dimensionwise token max; padding=-infinity, valid CLS/SEP included; L2 query and passage",
    "fixed": "model weights, tokenizer, queries/passages, cleaning, max length512, normalization and cosine scoring"
  },
  "document_l2": {
    "control": "A1 = q_unit @ d_unit",
    "N-Off": "same original unit CLS query @ raw unnormalized CLS passage",
    "fixed": "same dot-product operator, CLS pooling, original token states, query vectors and all other settings",
    "raw_norm_min": 5.032103061676025,
    "raw_norm_max": 8.003568649291992
  }
}
```

## 方法对比结果

[单独的CSV](comparison_metrics.csv)。A1为当前方案；出现在消融表中时为共同对照，不能当成新增实验。

粗体表示该列最优，下划线表示第二优的不同数值；并列时同时标注。标记基于均值大小，不表示统计显著性。

| 配置 | R@5 | R@10 | R@20 | P@5 | P@10 | P@20 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A5 词级TF-IDF | 0.7154 | 0.8507 | 0.8952 | 0.6260 | 0.4260 | 0.2320 | 0.9167 | 0.8516 |
| A0 BM25 | 0.7248 | 0.8566 | 0.9034 | 0.6520 | 0.4300 | 0.2350 | 0.9283 | 0.8786 |
| A7 BM25L | 0.7320 | 0.8566 | 0.9059 | 0.6500 | 0.4300 | 0.2355 | 0.9253 | 0.8778 |
| A8 BM25+ | 0.7248 | 0.8566 | 0.9068 | 0.6520 | 0.4300 | 0.2360 | 0.9283 | 0.8786 |
| A2 BM25+BGE+RRF | 0.7653 | <u>0.8933</u> | <u>0.9269</u> | 0.6840 | <u>0.4550</u> | <u>0.2430</u> | <u>0.9600</u> | 0.9229 |
| A3 BM25+BGE重排 | <u>0.7798</u> | 0.8919 | 0.9034 | <u>0.6960</u> | 0.4530 | 0.2350 | <u>0.9600</u> | <u>0.9344</u> |
| A1 BGE-CLS | **0.8019** | **0.9337** | **0.9584** | **0.7100** | **0.4730** | **0.2510** | **0.9620** | **0.9487** |

## 池化消融结果

[单独的CSV](pooling_ablation_metrics.csv)。A1为当前方案；出现在消融表中时为共同对照，不能当成新增实验。

粗体表示该列最优，下划线表示第二优的不同数值；并列时同时标注。标记基于均值大小，不表示统计显著性。

| 配置 | R@5 | R@10 | R@20 | P@5 | P@10 | P@20 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 BGE-CLS | **0.8019** | **0.9337** | **0.9584** | **0.7100** | **0.4730** | **0.2510** | **0.9620** | **0.9487** |
| P-Mean BGE-Mean | <u>0.7941</u> | <u>0.9215</u> | <u>0.9490</u> | <u>0.7040</u> | <u>0.4630</u> | <u>0.2480</u> | <u>0.9558</u> | <u>0.9362</u> |
| P-Max BGE-Max | 0.7645 | 0.9072 | 0.9386 | 0.6840 | 0.4560 | 0.2440 | 0.9375 | 0.9021 |

## 文档L2归一化消融结果

[单独的CSV](normalization_ablation_metrics.csv)。A1为当前方案；出现在消融表中时为共同对照，不能当成新增实验。

粗体表示该列最优，下划线表示第二优的不同数值；并列时同时标注。标记基于均值大小，不表示统计显著性。

| 配置 | R@5 | R@10 | R@20 | P@5 | P@10 | P@20 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 BGE-CLS | **0.8019** | **0.9337** | **0.9584** | **0.7100** | **0.4730** | **0.2510** | **0.9620** | **0.9487** |
| N-Off BGE-CLS文档L2关 | <u>0.7801</u> | <u>0.9279</u> | **0.9584** | <u>0.7000</u> | <u>0.4690</u> | **0.2510** | <u>0.9550</u> | <u>0.9296</u> |

## 统计定义与来源

Recall@k=前k相关数/该query全部相关数；Precision@k=前k相关数/k；MRR@k为前k内第一个相关项的倒数名次，否则0；nDCG@k使用二值gain与log2(rank+1)折扣，理想列表长度min(k,相关数)。

RRF审计证实每路输入100：100路深度重算与全部100条旧Top-20一致，50路深度只匹配9条。历史文件保存前50名不能当作融合输入深度。审计见 ../rrf_depth_audit.json。

所有既有Recall@5/10/20、MRR@10、nDCG@10、Precision@10均与原指标核对一致；新增的其余截断只来自原排名，不能当作新方法或新模型实验。参数全集见 [parameters.json](parameters.json)，未四舍五入指标见 [metrics_all_k.csv](metrics_all_k.csv)。

耗时仍按原记录的阶段口径：BM25索引0.203s/检索0.004s；BGE文档46.48s/query编码检索0.160s；RRF两路合计46.684s/0.164s，融合本身未独立测；重排654s且模型加载时间未保存；TF-IDF分词1.510s/索引0.267s/query处理0.008s；BM25L索引0.115s/检索0.004s、BM25+索引0.110s/检索0.006s，共享分词1.600s。

CLS/Mean联合脚本主体50.203s；Max和原始CLS向量及BM25变体联合主体51.581s，文档/query共享前向42.450/0.156s。不能把共享时间按每组重复相加，不含权重下载和依赖安装，也不是统一端到端测速。

结果仅为MTEB/T2Retrieval dev派生pilot，不是官方完整基准；没有显著性检验或生成准确率。
