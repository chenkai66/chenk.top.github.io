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
RAG 是当前 LLM 应用中部署最广泛，却工程实践最不成熟的范式。2024 年流行的 Demo 套路——用 `text-embedding-3-large` 把所有内容向量化，扔进 pgvector，再取 cosine 相似度 top-5——在千篇量级文档和对答案容错率较高的演示场景下尚可应付；但一旦面对十万级真实业务文档，且客户对答案准确性有严格要求时，这套方案便难以为继。本章内容，正是我希望更多团队在构建第二代 RAG 系统前就能掌握的关键认知。

最早的 RAG 论文（[Lewis et al., 2020][lewis-rag]）将检索增强生成定义为一种混合模型：稠密检索器（DPR）与生成器（BART）联合训练，使检索目标直接优化端到端任务的准确率。而到了 2026 年，生产级 RAG 已与 Lewis 的原始设计相去甚远——现代系统普遍采用冻结的预训练嵌入模型、独立的重排序器（reranker），以及不与检索器联合训练的仅解码器（decoder-only）生成模型。尽管如此，其核心思想——将知识存储与推理能力解耦——不仅得以保留，更发展为主导范式。[Gao et al. (2023) 的 RAG 综述][gao-survey] 是对 2020 年后演进路径（“Naive RAG → Advanced RAG → Modular RAG”）最全面的梳理。

![LLM 工程（8）：检索增强生成 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/illustration_1.png)

---

## RAG 到底是什么

检索增强生成（Retrieval-augmented generation）的核心逻辑是：在查询时，从外部语料库中检索相关文本片段，将其注入大语言模型的上下文窗口，从而生成有依据的答案。“增强”的部分体现在 prompt 模板中：

```yaml
You are an assistant. Answer the user's question using only the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{retrieved_chunks}

Question: {user_query}
```

真正的工程难点其实位于“增强”之前——即如何构建一个能精准召回相关文本块（chunk）的检索器。这涉及三大子系统：文本块切分（chunking）、嵌入（embedding）和排序（ranking）。

## Chunking 是隐形杀手

![图1：分块策略比较](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig1_chunking_strategies.png)

你如何将文档切分为文本块，直接决定了检索器理论上能够找到什么内容。常见的 chunk 大小包括 256、512 和 1024 tokens，主流策略如下：

- **固定大小（Fixed size）**：每 $N$ 个 token 切一刀。简单直接，但容易切断语义单元。
- **按句子切分（Sentence）**：在句子边界处分割。效果稍好，但往往过于碎片化。
- **递归字符切分（Recursive character）**：优先按 `

` 切，不行再试 `
`，再不行用 `. ` 等（LangChain 默认策略）。这是一个不错的基线方案。
- **语义切分（Semantic）**：对滑动窗口进行嵌入，在嵌入相似度显著下降处切分。效果更好，但计算成本更高。
- **晚期切分（Late chunking）**（[Günther et al., 2024](https://arxiv.org/abs/2409.04701)）：先用长上下文嵌入模型处理整篇文档（最长支持 8K–32K tokens），再对*嵌入序列*进行切分——每个 chunk 的嵌入都融合了周围文档的上下文信息。该方法最适合长文档，但需要支持长上下文的嵌入模型（如 Jina 或 BGE-M3）。

最佳策略取决于你的语料特性：代码应按函数或类切分，法律文本按条款划分，Markdown 按标题组织，PDF 则需单独解析表格和图片，避免打断正文流。我调试过的多数失败案例，根源往往是“文档在表格中间被截断”或“答案横跨两个 chunk，但任一 chunk 单独看都语义不全”。

建议通过合理性验证确定 chunk 大小：随机选取 20 个典型问题，在语料库中人工定位答案并统计其 token 数。若大多数答案可容纳于 512-token chunk，则选用 512；若多数需要 1500 tokens 上下文（如法律合同），则使用 1500 并配置 200-token 重叠。

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

重叠（overlap）至关重要。若无重叠，一个起始于 chunk A 第 799 位、结束于 chunk B 第 821 位的句子会被截断；而设置 100-token 重叠后，两个 chunk 都能包含完整句子。

### Late chunking：2024 年的突破

朴素切分（naive chunking）对每个 chunk 独立嵌入，“Q3 营收增长 12%”这类片段缺乏所属公司和年份等关键上下文。晚期切分（late chunking）则反转流程：先将整篇文档输入长上下文嵌入模型，获得每个 token 的上下文化嵌入，再按 chunk 边界对 token 嵌入进行池化，生成最终 chunk 嵌入。如此一来，“Q3 营收增长 12%”的嵌入便能反映“Apple 2024 年第一季度财报”这一上下文。

在长文档问答任务上的实测表明：相比采用相同边界的朴素切分，晚期切分可在不增加存储开销的前提下，将 NDCG 指标提升 5–15%，仅带来轻微的索引延迟增长。Jina 的 `jina-embeddings-v3` 和 BGE-M3 均原生支持该技术。

那么何时该用 late chunking？只要你的文档足够长，以至于单个 chunk 内的局部上下文丢失了关键框架信息——例如研究论文、法律合同、代码仓库或多章节技术文档。而对于短文档语料（如 FAQ 条目、产品描述），收益则相对有限。

## Embedding 模型选择

![图2：密集 vs 稀疏 vs 混合检索](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig2_dense_sparse_hybrid.png)

嵌入模型将文本映射为向量，使相似度搜索能召回语义相关的 chunk。所选模型直接决定检索质量、成本与延迟。

2026 年的技术格局如下：

| Model | Dim | MTEB avg | Notes |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 | ~64.6 | 闭源，$0.13/Mtok \|
| `Cohere embed-v4` | 1024 | ~67.1 | 多语言能力强 |
| `BGE-M3` | 1024 | ~69.4 | 开源、多语言、支持密集+稀疏+ColBERT |
| `Qwen3-Embedding-8B` | 4096 | ~74.0 | 开源，MTEB 排名第一 |
| `voyage-3-large` | 1024 | ~70.5 | 闭源，高端商用 |
| `jina-embeddings-v3` | 1024 | ~67.0 | 开源、多语言、原生支持 late chunking |

两个关键实践建议：

**多语言支持至关重要**。2022–2023 年的多数嵌入模型仅在英文上训练，在中文、日文或阿拉伯文上的检索效果显著下降。若你的语料包含多语言内容，务必选择多语言模型——BGE-M3 和 Qwen3-Embedding 是当前开源领域的佼佼者。

**领域专用模型往往优于通用模型**。一个在法律领域微调的小型嵌入模型，常能在法律语料上击败最先进的通用模型。如果你的语料具有强领域属性（如医疗、法律、代码或科研），应评估领域专用模型或自行微调。

自托管嵌入模型的成本正持续降低。单张 L4 GPU 上的 BGE-M3 可达约 3000 chunks/秒的吞吐量。对于少于 1000 万 chunks 的语料库，通常无需依赖托管服务。

关于**维度**的补充说明：高维嵌入（3K–4K）通常检索效果更优，但会增加存储开销和索引查询延迟。Matryoshka 表示学习（用于 `text-embedding-3-large` 和 Nomic 嵌入）通过训练使不同长度前缀（如 256、512、1024）均具备良好检索能力，允许你在部署时灵活权衡维度与性能，无需重新训练。对于 10 万文档规模的语料库，768–1024 维是性价比最高的选择。

## 向量索引：HNSW、IVF 与权衡

获得嵌入后，需借助向量索引加速近邻搜索，避免暴力比对。当前主流算法分为两类：

**HNSW**（Hierarchical Navigable Small World, [Malkov & Yashunin, 2018][malkov-hnsw]）通过构建多层图结构实现高效检索：每个节点连接约 $M$ 个最近邻，查询从顶层入口点开始，逐层贪婪下降并精炼候选集。召回率可通过 `ef_search` 调节（候选越多，召回越高，但查询越慢）。其索引构建较慢（O(N log N) 图构造）且内存占用高（约为嵌入存储的 1.5 倍），但在百万级规模下查询延迟可低于毫秒。

HNSW 已成为 pgvector、Milvus、Qdrant、Weaviate 等现代向量数据库的默认选项。针对 100 万至 1 亿向量的语料库，推荐参数为 `M=16–32`、`ef_construction=200–400`、`ef_search=50–100`，可在普通硬件上实现 >95% 召回率与 <5 ms 查询延迟。

**IVF**（Inverted File Index）则先通过 k-means 将嵌入聚类为 $K$ 个质心，查询时仅搜索最近几个质心对应的倒排列表。该方法内存效率高（无图结构开销），但召回-速度权衡不如 HNSW。FAISS 库以此为基础，衍生出 IVF-PQ（结合乘积量化进一步压缩）等高级变体。

**FAISS**（[Johnson et al., 2017](https://arxiv.org/abs/1702.08734)）是行业标准实现库，提供 Flat、IVF、HNSW、IVF-PQ 等底层索引原语，并支持灵活组合。生产团队通常用 FAISS 处理离线批量任务（如重排序实验、训练数据构建），而在线服务则选用 Qdrant 或 pgvector 等高层向量数据库。

对于 10 万至 100 万向量的部署场景，索引选择影响甚微——pgvector 内置的 HNSW 已足够。当规模达到 1 亿以上时，决策重点转向内存预算与分片数量；而超过 100 亿向量，则需进入定制基础设施领域，依赖 Google 或 Meta 级别的工程能力。

## 密集型 vs 稀疏型 vs 混合型

![图3：RRF 排名融合](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig3_rrf_fusion.png)

**密集检索（Dense retrieval）**（基于神经嵌入的 cosine 相似度）擅长捕捉语义相似性，但在精确匹配（如缩写、ID、罕见术语）上表现较弱。

**稀疏检索（Sparse retrieval）**（如 BM25 或其现代变体 SPLADE）则在精确匹配和罕见词上表现优异，但难以处理同义词或 paraphrase。**BM25**（[Robertson et al., 1995][robertson-bm25]）是一种经典的概率相关性模型，通过词频 × 逆文档频率并结合长度归一化进行评分，其公式为：
$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot |d|/\text{avgdl})}
$$
其中 $k_1 \approx 1.5$、$b \approx 0.75$ 为标准参数。过去 30 年，BM25 一直是词法检索的主流算法，即便在密集嵌入兴起后仍未被取代——因为它在依赖特定 token 的查询（如产品 SKU、错误代码、命名实体）上仍是同类最佳。

**混合检索（Hybrid retrieval）** 结合两者优势后再融合。2026 年几乎所有生产级 RAG 系统均采用混合方案——相比纯密集检索，其在多数基准测试上可将 NDCG@10 提升 10–30%，而额外成本极低（BM25 计算廉价，且 chunk 数据已存在）。

融合方式至关重要。简单的加权求和（`score = 0.5 * dense + 0.5 * bm25`）需对分数归一化，且调参困难。当前主流方案是**倒数排名融合（Reciprocal Rank Fusion, RRF）**（[Cormack et al., 2009][cormack-rrf]）：

```python
def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """rankings: list of ranked doc-id lists from different retrievers."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

RRF 在分数层面无需参数，经验常数 $k=60$ 表现稳健。具体流程为：分别运行密集与稀疏检索，各取 top-50 结果，经 RRF 融合后取 top-20，再送入重排序阶段。

RRF 之所以有效，在于它基于排名而非原始分数进行融合，因此不受 BM25（无界正数）与 cosine（[-1, 1]）分数分布差异的影响。无论底层检索器如何缩放置信度，任一检索器的 Rank-1 对最终得分贡献相同——这正是融合异构检索器所需的不变性。

## 重排序是被低估的英雄

![LLM 工程（8）：检索增强生成 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/illustration_2.png)

![图4：重排序器流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig4_reranker_pipeline.png)

嵌入检索虽快，但精度有限（本质是有损压缩）。第二阶段的 **cross-encoder 重排序器** 通过联合编码 `[query, candidate]` 直接输出相关性分数，显著提升精度。

Cross-encoder 单次计算成本约为嵌入 cosine 的 100 倍，无法用于百万级 chunk 的初筛，但对 top-20 候选进行重排序则成本可控，且能稳定提升 5–15% 的质量。

推荐模型包括：

- `BAAI/bge-reranker-v2-m3`（开源、多语言、约 6 亿参数、L4 GPU 上高效）
- `cohere-rerank-v3`（闭源、英文优化、$2/1K queries）
- `jina-reranker-v2`（开源、多语言、推理极快）

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query, candidates, top_k=5):
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
```

完整流水线变为：嵌入/稀疏检索 top-50 → 重排序至 top-5 → 注入 LLM 上下文。重排序器是值得投入延迟预算的关键环节——其带来的质量提升远比微调 LLM prompt 更可靠。

## 晚期交互：ColBERT

![图5：ColBERT 后期交互](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig5_colbert_late_interaction.png)

介于 bi-encoder 的速度与 cross-encoder 的质量之间，**晚期交互（late interaction）**（ColBERT, [Khattab & Zaharia, 2020][khattab-colbert]）提供了一种折中方案：query 与 document 分别编码为 per-token 向量（不进行池化），相似度通过 token 级匹配计算：
$$
\text{score}(q, d) = \sum_{i} \max_j \langle q_i, d_j \rangle
$$
ColBERT 既保留了 token 级匹配能力（利于罕见词检索），又支持并行计算。ColBERTv2 与 PLAID（Santhanam et al., 2022）通过残差压缩和近似检索，使其在百万文档规模上具备可行性。BGE-M3 也内置了 ColBERT 风格组件，可免费使用。

在 2025–2026 年的生产系统中，晚期交互正逐步应用于高精度检索场景——当重排序 50 个候选仍不足，但全量 cross-encoding 成本过高时。2024 年提出的 **ColPali** 更将该原理扩展至视觉-语言模型，用于文档图像检索，证明 token 级匹配在 PDF/扫描件检索上显著优于“先 OCR 再嵌入”的传统流程。

## Anthropic 的上下文检索与 GraphRAG

2024 年两项进展进一步提升了 RAG 在生产级语料上的表现。

**Anthropic Contextual Retrieval**（2024）在嵌入每个 chunk 前，先 prepend 一段模型生成的 50–100 token 上下文，说明该 chunk 在文档中的位置（例如：“本 chunk 来自 Acme Corp 2023 年 Q3 财报，讨论供应链中断的部分”）。这些带上下文的 chunk 随后通过密集嵌入与 BM25 分别检索，经 RRF 融合后再重排序。Anthropic 报告称，其内部基准测试中检索失败率降低 49%，结合重排序后更达 67%。关键在于，上下文生成是一次性索引开销（非每次查询），却大幅提升了嵌入对检索任务的适配性。

**GraphRAG**（[Microsoft Research, 2024][graphrag]）则摒弃扁平 chunk 思路，在索引阶段提取实体与关系构建知识图谱，并聚类为多粒度社区。查询时，针对跨文档的“全局性”问题（如“该语料库的主要主题是什么？”），直接检索社区摘要而非原始 chunk。GraphRAG 在此类任务上显著优于朴素 RAG，但代价更高：图谱构建需在索引阶段调用 LLM（约 $1–10/1000 文档）。对于天然需跨文档聚合的语料，GraphRAG 代表当前前沿；而对于事实检索类查询，hybrid + rerank 仍在成本-质量比上占优。

## 什么时候该用长上下文 instead

![图6：长上下文与 RAG 的权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig6_context_vs_rag.png)

[第 6 章](/zh/llm-engineering/06-long-context/)已介绍过长上下文技术，此处提供生产环境的决策矩阵：

| 属性 | RAG 胜出 | 长上下文胜出 |
|---|---|---|
| 语料库大小 | 大（>1M tokens） | 小（<200K tokens） |
| Query latency budget | <2s | 5–30s 可接受 |
| Cost per query | $0.001–0.05 \mid$0.10–2.00 |
| 来源归属 | 重要 | 不需要 |
| 问题局部性 | 是（定位单一事实） | 否（综合全文） |
| 更新频繁 | 是 | 否（每次变更需重填充） |

2024 年曾流行“长上下文杀死 RAG”的说法，但从成本角度看显然站不住脚。以 Claude-4.5-Sonnet 定价（$3/Mtok input）计算，100K-token prompt 仅输入成本就达 $0.30/query；而 RAG 通常只需 $0.001–0.01/query（含嵌入与 LLM 成本）。对日均 100 万查询的产品而言，二者日成本分别为 $300 与 $3000–10000。

更合理的方案往往是**两者结合**：先用 RAG 定位候选 chunk，再将候选集与合成 prompt 一同送入长上下文模型。这正是多数“2026 生产级 RAG”系统的实际架构。

## 万+ 文档规模下的失败模式

**嵌入漂移（Embedding drift）**：语料格式在三个月后变更（新模板、新词汇），导致新文档因向量分布偏移而难以被召回。防御措施：每月在 held-out 测试集上评估检索质量，当 MTEB 类指标漂移 >5% 时，重新嵌入整个语料库。

**Chunk 边界切割**：连贯答案被拆至两个 chunk，任一 chunk 单独均无意义。症状表现为：答案明确存在于语料中，但检索召回率低。防御手段包括：设置 100–200 tokens 重叠、采用尊重文档结构（章节、段落）的切分策略，并抽样验证“chunk 是否能独立成义”。Late chunking 通过让嵌入携带上下文信息，部分缓解此问题。

**查询分布漂移**：用户开始提出未预料的问题，导致基于旧查询类型训练的重排序器性能下降。防御方案：记录查询日志，每周抽样并人工标注，每季度重训重排序器。

**热门 chunk 主导（Hot chunks dominating）**：少数 chunk（如退货政策 FAQ）因嵌入靠近聚类中心，被无关查询频繁召回。防御方法：对过度检索的 chunk 施加惩罚（BM25 的 IDF 天然具备此特性；密集检索需显式引入多样性机制如 MMR）。

**仅含元数据的 Chunk**：如“4.2.1 节”这类 chunk 语义匹配广泛但无实质内容。防御策略：过滤内容密度过低的 chunk（如字母数字字符 <100 或独特非停用词 <5）。

**因 ingestion bug 导致的重复 Chunk**：曾遇一案例，30% chunks 因脚本重复运行而近似重复。Top-K 检索返回同一内容的多个副本，LLM 误判为多方确认而自信输出错误答案。防御措施：在 ingestion 阶段去重（MinHash、SimHash 或 normalize+hash）。

**嵌入更新滞后（Embedding stale-vs-document update lag）**：源文档已更新，但嵌入未重算，导致检索返回旧版本。防御方案：为每文档维护 `content_hash`，仅当 hash 变化时重索引，并将索引-源新鲜度纳入 SLO 监控。

**重排序器过度自信（Reranker overconfidence）**：Cross-encoder 重排序器输出的分数未经校准，在跨域场景下高分未必更准。防御建议：避免基于绝对分数设阈值，而应关注单次查询内的相对排名。

## 评估：不测量就等于白做

我见过太多团队在无评估集的情况下上线 RAG，随后耗费数周调整 prompt 与 chunk 大小，却无任何指标证明改进有效。**务必先构建评估集**，哪怕仅有 50 个问题。人工标注黄金标准 chunk 与答案，然后追踪以下指标：

- **Retrieval recall@k**：检索器是否返回了包含答案的 chunk？
- **Reranking precision@k**：重排序后 top-k 中，含答案 chunk 的比例？
- **Answer faithfulness**：答案是否严格基于检索内容（而非 LLM 先验知识）？
- **Answer correctness**：是否与黄金答案一致？

后两项可用 LLM-as-judge 评估（详见[第 10 章](/zh/llm-engineering/10-evaluation/)）；前两项为纯检索指标，易于计算且对调试最具指导性。

对于更大规模评估集（>500 问题），**RAGAS**（RAG Assessment，2023 开源）与 **TruLens** 等工具可基于 LLM-as-judge 自动化四指标 pipeline。它们是合理起点，但其指标可能被 prompt 调优“欺骗”——人工标注的黄金子集仍是唯一完全可信的信号。

## 生产架构建议

针对 2026 年典型的 10 万–100 万文档 RAG 部署：

- **存储**：Postgres + pgvector 存储向量与元数据；搭配 `pg_trgm` 或 Elasticsearch 支持 BM25。单区域写入，查询使用读副本。
- **嵌入器**：索引阶段用 L4 GPU 自托管 BGE-M3（一次性开销）；查询嵌入调用 Cohere 或 OpenAI API（低延迟、免运维）。
- **检索**：混合密集 + BM25，RRF 融合，取 top-50 候选。
- **重排序**：L4 GPU 自托管 BGE-reranker-v2-m3（或简化使用 Cohere Rerank API），top-50 → top-5。
- **生成器**：通用场景用 Claude-4.5-Sonnet 或 Qwen3-Max；成本敏感场景用自托管 Qwen3-32B。
- **评估**：维护 100–200 个人工标注问题，每次部署必跑，若 recall@10 或 faithfulness 下降 >5% 则告警。
- **监控**：记录每条查询的检索结果、生成内容、延迟与成本；抽样 1% 供人工复核。

该 pipeline 在多数负载下可实现 <1s p95 延迟与 <$0.01/query 成本，单 Postgres 实例可扩展至约 1000 万文档，并具备优雅降级能力。

## 总结

Chunking 是最被低估的调节旋钮——chunk 大小必须匹配答案的实际分布位置。若语料为多语言，务必选用多语言嵌入模型；若有领域模型，则优先使用。始终采用 hybrid（dense + sparse + RRF）方案，并坚持重排序。长上下文适用于小语料与综合任务；其余场景 RAG 在成本与新鲜度上更具优势。在启动第二轮迭代前，务必先建立评估集。2024 年的新进展（Contextual Retrieval、GraphRAG、late chunking）虽拓展了技术边界，但基础原则依然适用。

下一章：**生产规模的 prompt 工程**。涵盖 Chain-of-thought、self-consistency、prompt 缓存经济学，以及 jailbreak/injection 威胁模型。

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
