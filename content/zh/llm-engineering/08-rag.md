---
title: "大模型工程（八）：检索增强生成 RAG"
date: 2026-05-03 09:00:00
tags:
  - llm
  - rag
  - embeddings
  - reranking
  - hybrid-retrieval
categories: 大模型工程
series: llm-engineering
series_order: 8
series_title: "大模型工程"
lang: zh-CN
mathjax: true
disableNunjucks: true
description: "切分策略、dense vs sparse vs 混合检索、reranker 选型、2026 年长上下文 vs RAG 的取舍，以及 10 万文档以上才会冒头的失败模式。"
translationKey: "llm-engineering-8"
---
RAG 是 LLM 应用中最容易部署、但工程化最弱的模式。2024 年的常见做法是：用 `text-embedding-3-large` 嵌入所有内容，存到 pgvector，取 top-5 余弦相似度。这招对 1000 个文档和宽容的演示场景还行。但放到 10 万真实文档和挑剔的客户面前，就撑不住了。我希望更多团队在做第二代 RAG 之前能明白这些。

原始 RAG 论文（[Lewis 等，2020][lewis-rag]）提出了一种混合模型。Dense Retriever（DPR）和 Generator（BART）联合训练，优化端到端任务的准确率。到了 2026 年，生产环境中的 RAG 已经大变样。现代系统用冻结的预训练 embedder，搭配独立 reranker 和 decoder-only 生成器。生成器不再针对 retriever 进行训练。不过核心思想没变：把知识和推理分开参数化。这一思路成了主流范式。[Gao 等（2023）RAG 综述][gao-survey] 详细梳理了 2020 年后从 "Naive RAG → Advanced RAG → Modular RAG" 的演变过程。

![大模型工程（八）：检索增强生成 RAG — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/illustration_1.jpg)
## RAG 实际是什么

检索增强生成（RAG）的核心很简单。查询时，从外部语料库提取相关文本，塞进 LLM 的上下文，生成答案。"增强" 部分就是这个 prompt 模板：

```
你是一个助手。仅根据下面的上下文回答用户问题。
如果上下文里没有答案，回答 "我不知道"。

上下文：
{retrieved_chunks}

问题：{user_query}
```

真正有意思的地方在"增强"左边——构建检索器。难点是找到合适的片段。这需要搞定三个子系统：切分、嵌入和排序。
## 切分是隐形杀手

切分方式决定了检索器能找什么。常见 chunk 大小有 256、512 和 1024 个 token。策略如下：

- **固定大小**：每 $N$ 个 token 切一次。简单粗暴，但会破坏语义。
- **句子**：按句子边界切。效果好点，但通常太短。
- **递归字符**：先按 `\n\n`，再按 `\n`，最后按 `. `（LangChain 默认）。这是不错的基线。
- **语义**：用滑动窗口嵌入，相似度下降时切。效果更好，但成本高。
- **late chunking**（[Günther 等，2024](https://arxiv.org/abs/2409.04701)）：用长上下文 embedder 嵌入整篇文档，再在 *embedding 序列* 上切。每个 chunk 的 embedding 包含上下文信息。适合长文档，需要 Jina 或 BGE-M3。

选择方法要看语料库。代码文件按函数或类切。法律文件按条款切。Markdown 文件按标题切。PDF 文件中，表格和图片单独解析，别打断文本流。我踩过的坑大多是因为 "表格中间被切开" 或 "答案跨两个 chunk，单个 chunk 没意义"。

检查 chunk 大小是否合理：选 20 个代表性问题，手动找答案并数 token。如果多数答案在 512 个 token 内，就用 512。如果多数需要 1500 个 token 的上下文（比如法律合同），就用 1500 并设置 200 个 token 的重叠。

```python
# 合理的默认切分器
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document)
```

重叠很重要。没有重叠，从 chunk A 第 799 个字符开始、在 chunk B 第 821 个字符结束的句子会被截断。有了 100 个 token 的重叠，两个 chunk 都能包含完整句子。

### Late chunking：2024 年的突破

传统切分独立嵌入每个 chunk。比如，"Q3 营收增长 12 %" 的 chunk 不带公司或年份信息。Late chunking 改变了顺序：把整篇文档（最多 8K-32K 个 token）输入长上下文 embedder，获取每个 token 的上下文化 embedding，再在 chunk 边界内池化这些 embedding 得到最终结果。这样，"Q3 营收增长 12 %" 的 chunk embedding 就能反映 "Apple Q1 2024 财报" 的上下文。

报告显示，在长文档问答任务中，与传统切分相比，late chunking 提升了 5-15% 的 NDCG，不增加存储成本，索引延迟略有上升。Jina 的 `jina-embeddings-v3` 和 BGE-M3 都原生支持 late chunking。

什么时候用 late chunking？文档足够长时，单个 chunk 的局部上下文容易丢失重要信息，比如研究论文、法律合同、代码库或多部分技术文档。对于短文档语料（如 FAQ 或产品描述），收益较小。
## 嵌入模型选型

嵌入模型把文本转成向量，方便相似度搜索找到语义相关的片段。选哪个模型直接影响质量、成本和延迟。

2026 年顶尖模型如下：

| 模型 | 维度 | MTEB 平均 | 备注 |
|---|---|---|---|
| `text-embedding-3-large`（OpenAI） | 3072 | ~64.6 | 闭源，$0.13/Mtok |
| `Cohere embed-v4` | 1024 | ~67.1 | 多语言表现强 |
| `BGE-M3` | 1024 | ~69.4 | 开源，多语言，dense+sparse+colbert |
| `Qwen3-Embedding-8B` | 4096 | ~74.0 | 开源，MTEB 排名第一 |
| `voyage-3-large` | 1024 | ~70.5 | 闭源，高端 |
| `jina-embeddings-v3` | 1024 | ~67.0 | 开源，多语言，支持 late-chunking |

两个实战要点：

**多语言很重要。** 2022-2023 年的嵌入模型大多用英文训练，中文、日文、阿拉伯语文本检索效果差。如果语料库是多语言的，优先选多语言模型。BGE-M3 和 Qwen3-Embedding 是开源中的佼佼者。

**领域特定模型优于通用模型。** 小型嵌入模型在法律领域的微调版本，通常比最先进的通用模型表现更好。如果你的语料库有明确领域（如医疗、法律、代码或科学），建议评估领域特定模型，或者自己微调。

自托管嵌入模型越来越便宜。单个 L4 GPU 上跑 BGE-M3，每秒能处理约 3000 个片段。对于小于 1000 万片段的语料库，完全不用托管服务。

关于 **维度**：高维嵌入（3K-4K）检索效果好，但存储成本高，索引查找也慢。Matryoshka 表示学习方法（用在 `text-embedding-3-large` 和 Nomic 嵌入中）让不同长度前缀（256、512、1024）都能有效检索。部署时可以灵活调整维度，无需重新训练。对于 100K 文档的语料库，768-1024 维是实用选择。
## 向量索引：HNSW、IVF 与取舍

有了向量表示后，需要一个索引来快速找到最近邻。目前主流算法分两类。

**HNSW**（Hierarchical Navigable Small World，[Malkov & Yashunin, 2018][malkov-hnsw]）：构建多层图结构，每个节点连接 $M$ 个最近邻。查询从顶层入口开始，逐层下降，逐步缩小候选范围。通过 `ef_search` 调整召回率——值越大，召回越高，但查询越慢。索引构建耗时较长（O(N log N)），内存占用约是原始向量的 1.5 倍。不过，百万级数据查询速度能到亚毫秒级别。

HNSW 是 pgvector、Milvus、Qdrant、Weaviate 等数据库的默认选择。对于 1M 到 100M 的数据集，推荐参数：`M=16-32`、`ef_construction=200-400`、`ef_search=50-100`。这些参数能在普通硬件上实现 >95% 召回率，查询延迟低于 5 毫秒。

**IVF**（Inverted File Index）：用 k-means 将向量聚成 $K$ 个中心点。查询时先找最近的几个中心点，再搜索它们对应的倒排表。这种方法内存效率高，没有图结构开销，但召回率和速度的平衡不如 HNSW。FAISS 中广泛使用 IVF，并发展出更复杂的变体，比如 IVF-PQ（在 IVF 上加产品量化以进一步压缩）。

**FAISS**（[Johnson 等, 2017](https://arxiv.org/abs/1702.08734)）是最常用的实现库。它提供了多种基础索引方法（Flat、IVF、HNSW、IVF-PQ、IVF-SQ），还支持组合使用。生产环境中，我通常用 FAISS 做离线批量检索（如重排实验、训练数据生成），在线服务则交给更高层的向量数据库（如 Qdrant、pgvector）。

对于 100K 到 1M 的小规模部署，选哪种方法差别不大，pgvector 的 HNSW 就够了。数据量到 100M 以上，选择主要看内存预算和分片数量。到了 10B 级别，就得靠自定义基础设施和 Google/Meta 级别的工程能力了。
## Dense vs Sparse vs 混合

**Dense 检索**（神经 embedding 的余弦相似度）擅长语义匹配，但对精确匹配（如缩写、ID 和稀有词）效果差。

**Sparse 检索**（BM25 或其变体 SPLADE）擅长精确匹配和稀有词，但同义词或改写场景表现不佳。BM25 是一种概率相关性模型，基于词频 × 逆文档频率，并加入长度归一化打分。公式如下：

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot |d|/\text{avgdl})}$$

标准参数 $k_1 \approx 1.5$，$b \approx 0.75$。BM25 统治词法检索领域 30 年。现代 dense embedder 没能取代它，因为 BM25 在特定 token 查询（如 SKU、错误码、命名实体）上仍是最佳选择。

**混合检索**结合 Dense 和 Sparse，再合并结果。到 2026 年，几乎所有生产 RAG 系统都用混合方式。相比纯 Dense，混合方式在多数基准测试中 NDCG@10 提升 10-30%，成本却很低（BM25 很便宜，而且你已经有了数据块）。

合并公式很关键。简单加权和（`score = 0.5 * dense + 0.5 * bm25`）需要归一化，调参麻烦。主流方法是 **倒数排序融合**（RRF，[Cormack 等，2009][cormack-rrf]）：

```python
def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """rankings：来自不同检索器的排序 doc-id 列表。"""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

RRF 不依赖分数，常数 $k=60$ 非常稳健。先分别运行 Dense 和 Sparse 检索，各取 top-50，用 RRF 合并，再取 top-20 重排序。

RRF 的优势在于基于排名而非分数操作。BM25 分数无界，cosine 分数在 [-1, 1]，RRF 不关心这些差异。任何检索器的排名第一贡献相同，与底层置信度无关。这是组合不同分数分布检索器的最佳不变量。
## Reranking 是被忽略的英雄

![大模型工程（八）：检索增强生成 RAG — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/illustration_2.jpg)


![fig4: reranker pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig4_reranker_pipeline.png)

Embedding 检索快，但精度差。这是有损压缩，没办法。第二阶段用 **cross-encoder reranker** 提升质量。它直接对 `[query, candidate]` 联合编码，输出相关性分数。

Cross-encoder 的成本是 embedding cosine 的 100 倍。几百万 chunk 的检索跑不起。但第一阶段筛出前 20 后，用它完全可行。质量能稳定提升 5-15%。

我常用的模型有这些：

- `BAAI/bge-reranker-v2-m3`：开源，多语言，约 600M 参数，L4 上跑得快。
- `cohere-rerank-v3`：闭源，英文强，$2/1K queries。
- `jina-reranker-v2`：开源，多语言，速度极快。

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query, candidates, top_k=5):
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
```

流程很简单：先用 embed/sparse 检索出 top-50，再用 reranker 筛到 top-5，最后塞进 LLM 上下文。延迟预算花在 reranker 上最值。调 LLM 提示的效果，远不如它明显。
## Late interaction：ColBERT

![fig5: ColBERT late interaction](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig5_colbert_late_interaction.png)

Bi-encoder 快，cross-encoder 准，而 **late interaction**（ColBERT，[Khattab & Zaharia, 2020][khattab-colbert]）正好卡在中间。查询和文档各自独立编码成 per-token 向量，不池化。相似度逐 token 计算：

$$\text{score}(q, d) = \sum_{i} \max_j \langle q_i, d_j \rangle$$

ColBERT 保留了 token 级匹配，稀有词效果尤其好。还能并行处理，效率不错。ColBERTv2（Santhanam 等，2022）和 PLAID（Santhanam 等，2022）通过残差压缩和近似检索，把这套方法推到了百万文档规模。BGE-M3 里也有个 ColBERT 风格的组件，直接用就行。

2025 到 2026 年，高精度检索场景会更需要 late interaction。重排 50 个候选不够，全量交叉编码又太贵，这种技术刚好补上空缺。2024 年更新的 **ColPali** 把 ColBERT 的思路扩展到视觉-语言模型，用来检索文档图像。结果表明，token 级匹配比 OCR-then-embed 流程强不少，PDF 和扫描件检索提升尤其明显。
## Anthropic Contextual Retrieval 与 GraphRAG

2024 年，两项技术突破让 RAG 在大规模生产语料中的表现更上一层楼。

**Anthropic Contextual Retrieval**（2024）：嵌入每个片段前，先生成一段 50 到 100 token 的上下文。这段上下文描述了片段在文档中的位置，比如“这是 Acme Corp 2023 年 Q3 财报中讨论供应链中断的部分”。接着用密集嵌入器和 BM25 对片段进行嵌入，再通过 RRF 混合检索并重新排序。Anthropic 内部测试显示，这种方法让检索失败率降低了 49%。如果加上重新排序，降幅可达 67%。上下文生成是一次性操作，只在索引时完成，不增加查询开销，却显著提升了嵌入的检索适配性。

**GraphRAG**（[微软研究院，2024][graphrag]）：不再把文档简单切分成扁平片段，而是提取实体和关系，构建知识图谱。然后按不同粒度将图谱聚类成社区。查询时，针对跨文档的问题，返回的是社区总结，而不是原始片段。GraphRAG 在回答“全局性”问题时表现突出，比如“这个语料库的主要主题是什么？”这类单个片段无法解决的问题。代价是成本更高：图谱提取需要在索引时调用 LLM，每 1000 份文档大约花费 $1-10。对于跨文档聚合型查询，GraphRAG 是当前的前沿选择；而对于事实检索型查询，混合检索加重新排序依然更具性价比。

下一节会聊聊这些技术在实际生产环境中的踩坑经验。
## 什么时候改用长上下文

第 6 章提到过这个话题。下面是生产环境的决策矩阵：

| 属性 | RAG 更优 | 长上下文更优 |
|---|---|---|
| 语料规模 | 大（>1M token） | 小（<200K token） |
| 查询延迟 | <2秒 | 5-30秒可接受 |
| 单查询成本 | $0.001-0.05 | $0.10-2.00 |
| 来源溯源 | 重要 | 不需要 |
| 问题局部性 | 是（找一个事实） | 否（跨整篇综合） |
| 更新频率 | 是 | 否（每次更改需重新预填充） |

2024 年流行过一句话："长上下文杀死 RAG"。但光看成本，这话就不成立。以 Claude-4.5-Sonnet 的定价为例，输入 100K token 的 prompt 每次查询光输入就要花 $0.30。而 RAG 包括嵌入和 LLM 成本，通常只需 $0.001-0.01。如果每天处理 100 万次查询，RAG 的成本是 $300/天，长上下文则高达 $3000-10000/天。

实际场景中，正确答案往往是 **两者结合**。先用 RAG 找出候选片段，再用长上下文处理这些片段并生成综合提示。这就是大多数 "2026 生产 RAG" 系统的真实模样。
## 10 万文档以上的失败模式

**Embedding 漂移。** 语料格式变了，比如新模板、新词汇。三个月后，新文档的向量邻域不同，检索效果变差。每月在测试集上评估检索质量。如果 MTEB 分数漂移超过 5%，重新嵌入语料。

**Chunk 边界切割。** 一个完整答案被切成两段，单独看都不完整。症状是明确存在的答案召回率低。解决方法：加 100-200 token 的重叠，按章节或段落切分，抽样检查 chunk 是否独立有意义。Late chunking 让 embedding 带上下文信息，部分缓解问题。

**查询分布变化。** 用户开始问一些我没预料到的问题。reranker 训练数据和实际查询类型不匹配，表现下降。解决方法：记录查询，每周采样，人工标注，每季度重新训练 reranker。

**热门 chunk 主导。** 某些 chunk（比如退货政策 FAQ）总被频繁检索，不管相关性如何。这些 chunk 通常靠近簇心。解决方法：惩罚高频 chunk。BM25 IDF 自然能处理；密集检索需要显式加多样性项，比如 MMR。

**只含元数据的 chunk。** 一个 chunk 只有 "Section 4.2.1"，语义上匹配很多查询，但没实际内容。解决方法：过滤掉内容密度低的 chunk。比如，拒绝少于 100 字符或少于 5 个唯一非停用词的 chunk。

**摄入 bug 导致重复 chunk。** 我踩过坑，30% 的 chunk 是近似重复的，因为脚本在一个子集上跑了两次。Top-K 检索返回同一 chunk 的多个版本，LLM 误以为有多个来源确认事实，结果自信地断言错误。解决方法：摄入时去重，可以用 MinHash、SimHash 或简单归一化+哈希。

**Embedding 与文档更新滞后。** 文档在数据库中更新了，但 embedding 没重新计算。检索返回的是旧版本的 chunk。解决方法：为每个文档跟踪 `content_hash`，hash 变化时重新索引。监控索引和源的新鲜度，作为 SLO。

**Reranker 过度自信。** Cross-encoder reranker 在 "相关" 和 "不相关" 标记对上训练，输出分数不是校准概率。跨域场景下，0.9 分不一定比 0.6 分更准。解决方法：别直接用原始分数设阈值，改用查询内的相对排名。
## Eval：不量就什么都没意义

见过太多团队直接上 RAG，连 eval set 都没有。调 prompt、改 chunk 大小折腾几周，完全不知道有没有效果提升。**先搞个 eval set**，哪怕只有 50 道题。手动找黄金 chunk 和黄金答案。接着盯住这几个指标：

- **检索 recall@k**：retriever 找到包含答案的 chunk 了吗？
- **Rerank 后 precision@k**：rerank 的 top-k 里有多少包含答案？
- **答案忠实度**：答案是不是基于检索到的 chunk 推导出来的？别靠 LLM 自己瞎编。
- **答案正确性**：和黄金答案对得上吗？

后两个指标用 LLM-as-judge 跑得挺好，第 10 章细讲。前两个是纯检索指标，计算简单，调试起来最直接。

eval 集大一点（超过 500 题）时，可以用 **RAGAS**（2023 年开源）和 **TruLens**。这些工具默认用 LLM-as-judge 自动跑四个指标。它们是个不错的起点，但别太依赖——调 prompt 取悦 judge 很容易，黄金标准还是得靠人工标注的子集。
## 生产架构推荐

2026 年，部署 100K-1M 文档的 RAG 系统，建议如下：

- **存储**：用 Postgres 加 pgvector 存向量和元数据。加 `pg_trgm` 或 Elasticsearch 实现 BM25。写入单区域，查询用读副本。
- **Embedder**：索引用 BGE-M3，跑在 L4 GPU 上（基本一次性）。查询嵌入用 Cohere 或 OpenAI API，低延迟，不用管基础设施。
- **检索**：dense 和 BM25 混合检索，RRF 合并结果，取 top-50 候选。
- **Reranker**：BGE-reranker-v2-m3 跑在同一个 L4 GPU 上，或者直接用 Cohere Rerank API 省事。从 top-50 缩到 top-5。
- **生成器**：通用场景用 Claude-4.5-Sonnet 或 Qwen3-Max。成本敏感就自托管 Qwen3-32B。
- **评估**：准备 100-200 个手工标注问题，每次部署都跑一遍。recall@10 或 faithfulness 下降超 5%，立刻报警。
- **监控**：记录每个查询、检索结果、生成内容、延迟和成本。抽 1% 数据人工审核。

这套流水线能满足大多数需求：<1s p95 延迟，单次查询 <$0.01。单个 Postgres 实例撑到 1000 万文档没问题，高负载下性能下降也平滑。
## 小结与下一篇

切分的重要性常被低估。选 chunk 大小时，要匹配答案实际出现的位置。语料库多语言？用多语言 embedder。有领域专用 embedder？更好，直接用。混合方法（dense + sparse + RRF）是标配，重排也是必做项。小规模语料和综合任务，长上下文效果好。其他场景，RAG 在成本和时效性上更有优势。建评估集要趁早，别急着迭代第二版。2024 年新进展（Contextual Retrieval、GraphRAG、late chunking）会推动技术边界，但基础原理依然不过时。

下一篇聊**生产规模的 prompting**。Chain-of-thought、self-consistency、prompt caching 经济学，还有 jailbreak/injection 威胁模型。
## 参考文献

- [Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020.][lewis-rag] The original RAG paper.
- [Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," 2023.][gao-survey]
- [Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT," SIGIR 2020.][khattab-colbert]
- [Robertson et al., "Okapi at TREC-3," NIST 1995.][robertson-bm25] The BM25 paper.
- [Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs," IEEE TPAMI 2018.][malkov-hnsw]
- [Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009.][cormack-rrf]
- [Microsoft Research, "GraphRAG: Unlocking LLM Discovery on Narrative Private Data," 2024.][graphrag]
- [Anthropic, "Introducing Contextual Retrieval," 2024.](https://www.anthropic.com/news/contextual-retrieval)
- Günther et al., "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models," 2024.
- Santhanam et al., "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction," NAACL 2022.
- Johnson et al., "Billion-scale similarity search with GPUs (FAISS)," 2017.
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [RAGAS evaluation framework](https://github.com/explodinggradients/ragas)

[lewis-rag]: https://arxiv.org/abs/2005.11401
[gao-survey]: https://arxiv.org/abs/2312.10997
[khattab-colbert]: https://arxiv.org/abs/2004.12832
[robertson-bm25]: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
[malkov-hnsw]: https://arxiv.org/abs/1603.09320
[cormack-rrf]: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
[graphrag]: https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
