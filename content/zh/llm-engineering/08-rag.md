---
title: "大模型工程（八）：RAG 架构与落地"
date: 2026-04-03 09:00:00
tags:
  - LLM
  - RAG
  - Embeddings
  - reranking
  - hybrid-retrieval
categories: 大模型工程
series: llm-engineering
series_order: 8
series_title: "大模型工程"
lang: zh
mathjax: true
disableNunjucks: true
description: "切分策略、dense vs sparse vs 混合检索、reranker 选型、2026 年长上下文 vs RAG 的取舍，以及 10 万文档以上才会冒头的失败模式。"
translationKey: "llm-engineering-8"
---
RAG 是当前 LLM 应用中部署最广泛，但工程实践最不成熟的范式。 2024 年流行的 Demo 套路——用 `text-embedding-3-large` 把所有内容向量化，扔进 pgvector，然后取 cosine 相似度 top-5——应付千篇量级的文档和对答案容错率较高的演示场景尚可。但当处理十万篇真实业务文档，且客户严格要求答案准确性时，该方案便难以胜任。这一章的内容，我希望更多团队在构建第二代 RAG 系统之前就能掌握。

最早的 RAG 论文（[Lewis et al., 2020][lewis-rag]）将检索增强生成定义为稠密检索器（DPR）与生成器（BART）联合训练的混合架构，以优化端到端任务的准确率；而 2026 年的生产级 RAG 已显著偏离这一设计，现代系统普遍采用冻结的预训练 embedding 模型、独立重排序器（reranker）和不与检索器联合训练的仅解码器（decoder-only）生成模型，但其核心思想——将知识存储与推理能力解耦——得以保留并发展为主导范式。[Gao et al. (2023) 的 RAG 综述][gao-survey] 是对 2020 年后演进路线（"Naive RAG → Advanced RAG → Modular RAG"）最全面的概述。

![LLM Engineering (8): Retrieval-Augmented Generation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/illustration_1.png)

## RAG 到底是什么

检索增强生成（Retrieval-augmented generation）：在查询时，从外部语料库检索相关文本片段，并将其注入大语言模型的上下文窗口，从而生成有依据的答案。“增强”的部分体现在 prompt 模板里：

```yaml
You are an assistant. Answer the user's question using only the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{retrieved_chunks}

Question: {user_query}
```

真正的工程难点在于‘增强’环节之前的构建过程，即构建能精准召回相关文本块（chunk）的检索器（retriever），其核心由文本块切分（chunking）、文本嵌入（embedding）和相关性排序（ranking）三大组件构成。

## Chunking 是隐形杀手

![fig1: chunking strategies compared](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/fig1_chunking_strategies.png)

文本块的切分方式直接决定了检索器理论上能够召回的内容边界，常见的 chunk 大小有 256、512、1024 tokens，常见策略包括：

- **Fixed size**：每 $N$ 个 token 切一刀。简单，但会切断语义单元。
- **Sentence**：按句子边界切分。好一些，但往往太碎。
- **Recursive character**：按 `\n\n` 切，切不动再试 `\n`，再试 `. ` 等（LangChain 默认）。不错的基线。
- **Semantic**：嵌入滑动窗口，在嵌入相似度下降处切分。效果更好，成本更高。
- **Late chunking** ([Günther et al., 2024](https://arxiv.org/abs/2409.04701))：用长上下文 embedder 嵌入整个文档，然后切分*嵌入序列*——每个 chunk 的嵌入都融合了周围文档的上下文。最适合长文档，需要长上下文 embedder （Jina, BGE-M3）。

正确答案取决于你的语料库。代码：按函数/类切分。法律：按条款切分。 Markdown：按标题切分。 PDF：单独解析表格和图片，别让它们打断正文流。我在调试中遇到的多数失败案例，根本原因在于：文档在表格内部被截断，或答案跨越两个文本块，而任一文本块均无法独立支撑完整语义。

建议通过合理性验证确定文本块大小：选取 20 个典型问题，在语料库中人工定位答案并统计其 token 数量；如果多数答案可容纳于 512-token chunk，则选用 512；如果多数需要 1500 token 上下文（如法律合同），则选用 1500 并配置 200-token 重叠。

```python
# A reasonable default chunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # tokens-ish for English (using char count as proxy)
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document)
```

重叠（overlap）很关键。若未设置重叠，一个起始于文本块 A 第 799 个 token、终止于文本块 B 第 821 个 token 的句子将被截断。有了 100-token 重叠，两个 chunk 都包含完整的句子。

### Late chunking： 2024 年的突破

 朴素切分（naive chunking）对每个文本块独立进行嵌入，“Q3 营收增长 12%”这类文本块往往缺乏所属公司及年份等关键上下文信息。晚期切分（late chunking）则调整了处理顺序：先将整篇文档（最长支持 8K–32K tokens）输入长上下文 embedding 模型，获得每个 token 的上下文感知嵌入，再按文本块边界对 token 嵌入进行池化，生成最终的文本块嵌入。“Q3 营收增长 12%”的 chunk 嵌入现在能反映周围“Apple 2024 年第一季度财报”的上下文。

在长文档问答任务上的实测效果显示：相较于采用相同文本块边界的朴素切分方法，晚期切分可将 NDCG 指标提升 5–15%，且无需额外存储开销，仅带来轻微的索引延迟增长。 Jina 的 `jina-embeddings-v3` 和 BGE-M3 都原生支持 late chunking。

什么时候 late chunking 有用？只要你的文档长到单个 chunk 内的局部上下文丢失了重要框架信息——例如研究论文、法律合同、代码库、多章节技术文档。对于短文档语料库（如 FAQ 条目、产品描述），收益较小。

## Embedding 模型选择

![fig2: dense vs sparse vs hybrid retrieval](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/fig2_dense_sparse_hybrid.png)

文本嵌入（embedding）将原始文本映射为向量表示，从而支持基于相似度的语义检索，以召回相关的文本块。选择的模型决定了质量、成本和延迟。

2026 年的技术现状：

| Model | Dim | MTEB avg | Notes |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 | ~64.6 | Closed, $0.13/Mtok |
| `Cohere embed-v4` | 1024 | ~67.1 | Multilingual strong |
| `BGE-M3` | 1024 | ~69.4 | Open, multilingual, dense+sparse+colbert |
| `Qwen3-Embedding-8B` | 4096 | ~74.0 | Open, top of MTEB |
| `voyage-3-large` | 1024 | ~70.5 | Closed, premium |
| `jina-embeddings-v3` | 1024 | ~67.0 | Open, multilingual, late-chunking native |

两个实际要点：

**多语言支持很重要。** 2022-2023 一代的 embedder 大多是英文训练的。它们在中文、日文、阿拉伯文上的检索效果很差。如果你的语料库是多语言的，选一个多语言模型——BGE-M3 和 Qwen3-Embedding 是开源界的领头羊。

**领域专用往往胜过通用。** 一个小模型在法律领域微调后，在法律语料库上的表现往往胜过 SOTA 通用 embedder。如果你的语料库有很强的领域属性（医疗、法律、代码、科学），评估领域专用 embedder 或自己微调。

自托管 embedder 越来越便宜。单张 L4 GPU 上的 BGE-M3 每秒能处理约 3000 个 chunk。对于大多数小于 1000 万 chunk 的语料库，你不需要托管服务。

关于**维度**的一点说明：高维嵌入（3K-4K）通常检索效果更好，但存储成本更高，索引查找更慢。 Matryoshka 表示学习方法（用于 `text-embedding-3-large` 和 Nomic 嵌入）训练嵌入时使得各种长度的前缀（256, 512, 1024）都能良好检索，让你能在部署时权衡维度与性能而无需重新训练。对于 10 万文档规模的语料库， 768-1024 维是实际的甜蜜点。

## 向量索引： HNSW、 IVF 与权衡

有了 embedding，你需要一个向量索引来比暴力搜索更快地找到最近邻。两大算法家族主导市场：

**HNSW**（Hierarchical Navigable Small World, [Malkov & Yashunin, 2018][malkov-hnsw]）：构建一个多层图，每个节点连接到大约 $M$ 个最近邻。查询从顶层入口点开始，贪婪地逐层下降，每一步 refining 候选集。召回率可通过 `ef_search` 调节（候选越多 = 召回率越高，查询越慢）。索引构建慢（O(N log N) 图构建）且吃内存（约嵌入存储的 1.5 倍），但在百万级规模下查询延迟低于毫秒。

HNSW 是 pgvector、 Milvus、 Qdrant、 Weaviate 和大多数现代向量数据库的默认选项。对于 100 万 -1 亿向量语料库的推荐参数：`M=16-32`, `ef_construction=200-400`, `ef_search=50-100。这些参数在普通硬件上能提供 >95% 召回率和 <5 ms 查询延迟。

**IVF**（Inverted File Index）：使用 k-means 将嵌入聚类为 $K$ 个质心；查询时，找到最近的几个质心，只搜索它们的倒排列表。内存效率高（无图开销），但召回率与速度的权衡不如 HNSW。用于 FAISS，它是更复杂变体（如 IVF-PQ，在 IVF 基础上进行乘积量化以进一步压缩）的基础。

**FAISS** ([Johnson et al., 2017](https://arxiv.org/abs/1702.08734)) 是主力实现库。它暴露原始索引原语（Flat, IVF, HNSW, IVF-PQ, IVF-SQ）并允许你组合它们。生产团队通常用 FAISS 做离线批量检索（重排序实验、训练数据构建），用更高层的向量数据库（Qdrant, pgvector）做在线服务。

对于 10 万 -100 万向量部署，选择几乎无关紧要——pgvector 里的 HNSW 就够了。到了 1 亿+，选择就变成了内存预算和分片数量的问题。到了 100 亿+，你就进入了定制基础设施领域，需要 Google/Meta 级别的工程能力。

## 密集型 vs 稀疏型 vs 混合型

![fig3: RRF fusion of rankings](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/fig3_rrf_fusion.png)

**Dense retrieval**（神经嵌入上的 cosine）：擅长语义相似度，弱于精确匹配（缩写、 ID、稀有术语）。

**Sparse retrieval**（BM25 或其现代变体如 SPLADE）：擅长精确匹配和稀有术语，弱于同义词/ paraphrase。**BM25** ([Robertson et al., 1995][robertson-bm25]) 是一种概率相关性模型，通过词频 × 逆文档频率并加上长度归一化来评分文档。公式：

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot |d|/\text{avgdl})}$$

其中 $k_1 \approx 1.5$ 和 $b \approx 0.75$ 是标准参数。 BM25 作为主导的 lexical 检索算法已经 30 年了；现代 dense embedder 并没有取代它，因为对于依赖特定 token 的查询（产品 SKU、错误代码、命名实体），它仍然是同类最佳。

**Hybrid retrieval** 结合两者，然后合并。 2026 年几乎每个生产级 RAG 系统都是混合的——相比纯 dense 优势巨大（大多数基准测试上 NDCG@10 提升 10-30%），成本很小（BM25 很便宜，你已经有 chunk 了）。

合并公式很重要。简单的加权求和（`score = 0.5 * dense + 0.5 * bm25`）需要分数归一化且很难调。主导模式是**Reciprocal Rank Fusion**（RRF, [Cormack et al., 2009][cormack-rrf]）：

```python
def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """rankings: list of ranked doc-id lists from different retrievers."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

RRF 在分数意义上是无参数的；常数 $k=60$ 经验证非常稳健。独立运行 dense 和 sparse 检索，各取 top-50， RRF 合并，再取 top-20。然后重排序。

为什么 RRF 有效：它操作的是排名而不是分数，所以它不在乎 BM25 分数是无界正数而 cosine 分数在 [-1, 1] 之间。任何 retriever 的 Rank-1 对最终分数的贡献相同，无论底层 retriever 如何缩放其置信度。这是结合具有非常不同分数分布的 retriever 的正确不变量。
## 重排序是被低估的英雄

![LLM Engineering (8): Retrieval-Augmented Generation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/illustration_2.png)


![fig4: reranker pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/fig4_reranker_pipeline.png)

Embedding 检索快是快，但精度有限（毕竟是有损压缩）。第二阶段的 **cross-encoder reranker** 能直接对每个候选项和 query 打分：把 `[query, candidate]` 一起编码，输出一个相关性分数。

Cross-encoder 算一对数据的成本是 embedding cosine 的 100 倍，所以拿它去百万级 chunk 里做检索不现实。但如果是第一阶段检索出来的 top-20 候选，这点开销完全承受得起，而且 一致地 能把质量提升 5-15 %。

我推荐的模型：

- `BAAI/bge-reranker-v2-m3`（开源，多语言，~600M 参数， L4 上跑得飞快）
- `cohere-rerank-v3`（闭源，英文强项， $2/1K queries）
- `jina-reranker-v2`（开源，多语言，速度极快）

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query, candidates, top_k=5):
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
```

现在的流水线变成了： embed/sparse 检索 top-50 → rerank 到 top-5 → 塞进 LLM 上下文。 Reranker 是值得消耗 latency budget 的地方；它带来的质量提升，比你去微调 LLM prompt 要稳定得多。

## 晚期交互： ColBERT

![fig5: ColBERT late interaction](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/fig5_colbert_late_interaction.png)

介于 bi-encoder 的速度和 cross-encoder 的质量之间，有个方案叫 **late interaction**（ColBERT, [Khattab & Zaharia, 2020][khattab-colbert]）。 Query 和 document 分别编码成 per-token 向量（不做 pooling）。相似度按 token 逐个计算：

$$\text{score}(q, d) = \sum_{i} \max_j \langle q_i, d_j \rangle$$

ColBERT 保留了 token 级别的匹配能力（对稀有词很友好），同时支持并行计算。 ColBERTv2 (Santhanam et al., 2022) 和 PLAID (Santhanam et al., 2022) 通过残差压缩和近似检索，让它在百万文档规模上变得可行。 BGE-M3 里也包含了一个 ColBERT 风格的组件，免费能用。

2025-2026 年的生产环境里， late interaction 会出现在那些对精度要求极高、 rerank 50 个候选还不够、但全量 cross-encoding 又太贵的场景。 2024 年更新的 **ColPali** 把 ColBERT 的晚期交互原则扩展到了视觉 - 语言模型，用于文档图像检索，发现 token 级匹配比“先 OCR 再 embed"的流水线在 PDF/扫描件检索上效果提升巨大。

## Anthropic 的上下文检索与 GraphRAG

2024 年有两个进展，把生产级语料上的 RAG 质量又推了一把。

**Anthropic Contextual Retrieval** (2024)：在 embed 每个 chunk 之前，先 prepend 一段模型生成的 50-100 token 上下文，解释这个 chunk 在文档里的位置（“本 chunk 来自 Acme Corp 2023 年 Q3 财报，讨论供应链中断的部分”）。这些带上下文的 chunk 再用 dense embedder 和 BM25 分别编码，用 RRF 混合检索，最后 rerank。 Anthropic 报告说，在他们内部基准测试上，检索失败率降低了 49 %，如果结合 reranking 能达到 67 %。这招的 trick 在于，上下文生成是 indexing 阶段的一次性成本（不是每 query 都生成），但 dramatically 提升了 embedding 对检索的适应性。

**GraphRAG** ([Microsoft Research, 2024][graphrag])：不把文档当成扁平的 chunk，而是在 indexing 阶段把实体和关系提取成知识图谱。把图谱聚类成不同粒度的社区。查询时，针对那些跨越多个文档的问题，检索社区摘要（而不是原始 chunk）。对于“这个语料库的主要主题是什么？”这种没有任何单个 chunk 能回答的“全局”问题， GraphRAG 表现优于 naive RAG。代价是更高：图谱提取需要在 indexing 时调用 LLM （~$1-10 per 1000 documents）。对于查询天然需要跨文档聚合的语料， GraphRAG 是当前的 frontier；对于事实检索类查询， hybrid + rerank 在成本 - 质量比上依然胜出。

## 什么时候该用长上下文 instead

![fig6: long context vs RAG trade-off](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/08-rag/fig6_context_vs_rag.png)

第 6 章介绍过这个，这里是生产环境的决策矩阵：

| Property | RAG wins | Long context wins |
|---|---|---|
| Corpus size | Big (>1M tokens) | Small (<200K tokens) |
| Query latency budget | <2s | 5-30s ok |
| Cost per query | $0.001-0.05 | $0.10-2.00 |
| Source attribution | Important | Not needed |
| Question is local | Yes (find one fact) | No (synthesize across whole doc) |
| Updates frequent | Yes | No (re-prefill on every change) |

2024 年有个流行的说法——“长上下文杀死 RAG"——单从成本角度看就是错的。按 Claude-4.5-Sonnet 的定价（$3/Mtok input），100K-token 的 prompt 在生成前就要 $0.30 per query。 RAG 通常只要 $0.001-0.01 per query，包含 embedding 和 LLM 成本。对于一个每天服务 1M queries 的产品，那是 $300/天 vs 长上下文的 $3000-10000/天。

正确的答案往往是 **都要**：用 RAG 找到候选 chunk，然后把候选集 + 一个合成 prompt 一起塞进长上下文。这才是大多数“2026 生产级 RAG”系统的真实面貌。

## 10 万 + 文档规模下的失败模式

**Embedding 漂移。** 你的语料库过了三个月格式变了（新模板，不同词汇）。新文档检索效果不好，因为它们的向量邻域变了。防御：每月在一个 held-out 测试集上重新评估检索质量。当 MTEB 风格的分数漂移 >5 % 时，重新 embed 整个语料。

**Chunk 边界切割。** 一个连贯的答案被切到了两个 chunk 里；单独看哪个 chunk 都没信息量。症状：答案明明在语料里，但检索 recall 很低。防御：使用重叠（100-200 tokens），尊重文档边界（章节、段落）的结构化 chunking，并通过采样验证“这个 chunk 单独看是否讲得通”。 Late chunking 通过让每个 chunk 的 embedding 携带上下文信息，部分解决了这个问题。

**查询分布漂移。** 用户开始问你没预料到的问题。 Reranker 是在不同类型的查询上训练的，表现下滑。防御：记录查询，每周采样，人工标注，每季度重训 reranker。

**Hot chunks 主导。** 少数 chunk （比如关于退货政策的 FAQ）不管相关性如何，每个查询都被检索出来。它们嵌在聚类中心附近。防御：惩罚被过度检索的 chunk （BM25 IDF 天然会做这个； dense 检索需要显式的多样性项，比如 MMR）。

**只包含元数据的 Chunk。** 一个 chunk 只有"4.2.1 节”，语义上匹配很多查询但没答案。防御：过滤低于最小内容密度的 chunk （比如拒绝 <100 个字母数字字符或 <5 个独特非停用词的 chunk）。

** ingestion bug 导致的重复 Chunk。** 我有一次遇到一个语料库， 30 % 的 chunk 是近似重复的，因为 ingestion 脚本在子集上跑了两次。 Top-K 检索返回了同一个 chunk 的 5 个版本； LLM 以为有 5 个来源确认了一个事实， confidently  asserted it。防御： ingestion 时去重（MinHash, SimHash, 或简单的 normalize+hash）。

**Embedding  stale-vs-document 更新滞后。** 源数据库里的文档更新了； embedding 还没重算。检索返回的是旧版本的 chunk。防御：跟踪每个文档的 `content_hash`，重索引任何 hash 变化的 chunk，监控索引 vs 源的新鲜度作为 SLO。

**Reranker 过度自信。** Cross-encoder reranker 是在标记为"relevant"或"irrelevant"的配对上训练的——它输出一个分数，但这不是校准过的概率。在跨域场景下， 0.9 分的 confident 不一定比 0.6 分更准确。防御：不要绝对地阈值化原始 reranker 分数；在单个 query 内使用相对排名。

## 评估：不测量就等于白做

我见过太多团队上线 RAG 却没有 eval set。然后他们花几周时间迭代 prompt 和 chunk 大小，却没有任何信号表明情况在好转。**先建一个 eval set**，哪怕只有 50 个问题。人工找到黄金标准的 chunk 和黄金标准的答案。然后跟踪：

- **Retrieval recall@k**：检索器是否返回了包含答案的 chunk？
- **Reranking precision@k**： rerank 后的 top-k 中，有多少比例包含答案？
- **Answer faithfulness**：答案是否源自检索到的 chunk （而不是 LLM 的先验知识）？
- **Answer correctness**：是否与黄金答案匹配？

后两项用 LLM-as-judge 没问题；第 10 章会详细讲。前两项是纯检索指标，容易计算，对调试最 actionable。

对于更大的 eval set （>500 个问题），像 **RAGAS** (RAG Assessment, 2023 开源) 和 **TruLens** 这样的工具可以用 LLM-as-judge 默认值自动化这四个指标的流水线。它们是个合理的起点，但它们计算的指标可以通过 tuning prompt 来迎合裁判——黄金标准的人工标注子集依然是唯一完全可信的信号。

## 生产架构建议

对于 2026 年典型的 10 万 -100 万文档 RAG 部署：

- **Storage**： Postgres 配 pgvector 存向量和元数据。加 `pg_trgm` 或 Elasticsearch 做 BM25。单区域写入，查询用 read replicas。
- **Embedder**： BGE-M3 自托管在 L4 GPU 上用于 indexing （一次性-ish），查询 embedding 用 Cohere 或 OpenAI API （低延迟，无 infra 负担）。
- **Retrieval**： hybrid dense + BM25， RRF 合并， top-50 候选。
- **Reranker**： BGE-reranker-v2-m3 自托管在同一块 L4 上（或为了简单用 Cohere Rerank API）， top-50 → top-5。
- **Generator**：通用场景用 Claude-4.5-Sonnet 或 Qwen3-Max；成本敏感场景用自托管 Qwen3-32B。
- **Eval**： 100-200 个人工标注问题，每次部署都跑，如果 recall@10 或 faithfulness 下降 >5 % 就报警。
- **Monitoring**：记录每个 query、检索结果、生成内容、延迟、成本。采样 1 % 供人工审查。

这个流水线在大多数工作负载下能提供 <1s p95 延迟，单次查询成本 <$0.01，单个 Postgres 实例能扩展到 ~1000 万文档，并且在高负载下能优雅降级。

## 总结与下一章

Chunking 是最被低估的旋钮；挑选 chunk 大小要匹配你的答案实际存放的位置。如果语料是多语言的，就用多语言 embedder；如果有领域 embedder 就用领域的。始终走 hybrid （dense + sparse + RRF）路线，始终要 rerank。小语料和合成任务长上下文赢；其他所有情况 RAG 在成本和新鲜度上赢。在构建第二次迭代之前先建好 eval set。 2024 年的进展（Contextual Retrieval, GraphRAG, late chunking）推动了 frontier；但基础原则依然适用。

下一章：**生产规模的 prompt 工程**。 Chain-of-thought， self-consistency， prompt 缓存经济学，以及 jailbreak/injection 威胁模型。
## 参考文献

- [Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020.][lewis-rag] RAG 领域的开山之作。
- [Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," 2023.][gao-survey]
- [Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT," SIGIR 2020.][khattab-colbert]
- [Robertson et al., "Okapi at TREC-3," NIST 1995.][robertson-bm25] BM25 算法的原始论文。
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