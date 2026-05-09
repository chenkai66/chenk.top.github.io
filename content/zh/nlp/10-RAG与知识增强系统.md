---
title: "自然语言处理（十）：RAG与知识增强系统"
date: 2025-11-15 09:00:00
tags:
  - NLP
  - RAG
  - LLM
  - 向量数据库
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "从第一性原理构建生产级 RAG 系统：retrieve-then-generate 概率分解、向量索引（FAISS / Milvus / Chroma / Weaviate / Pinecone）、稠密+稀疏混合检索与 RRF 融合、Cross-encoder 重排、切块策略、查询改写、HyDE，以及 Self-RAG / Corrective-RAG。"
disableNunjucks: true
series_order: 10
translationKey: "nlp-10"
polished_by_qwen_max: true
---
一个冻结了知识的语言模型就像是个“自信的骗子”。它无法读取昨天的事故报告、公司的 wiki 页面，或者今天早上刚发布的更新日志，因此当你向它提问时，它会编造出一个语法上毫无瑕疵但事实完全错误的答案。**检索增强生成（RAG）** 的出现打破了这一僵局，其核心思想是将“记忆”与“推理”分离：让大语言模型（LLM）保持小巧且稳定，同时把那些容易变化的知识存储在一个可以随时更新的外部系统中。在生成回答之前，先从外部存储中检索相关信息，并将其作为上下文条件输入模型。

理念只需一段话就能概括，而工程实现则是本文的重点。一个真正可用的 RAG 系统涉及大约十几个关键参数——分块大小、Embedding 模型的选择、索引类型、$k$ 值、混合权重策略、重排序深度、Prompt 模板设计、引用格式规范、拒答策略等——而且这些参数之间往往相互影响。接下来，我们将逐一探讨每个参数背后的数学原理、权衡取舍以及可运行的代码示例。


<!-- wanx-hero -->
![自然语言处理（十）：RAG与知识增强系统 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/rag-knowledge-enhancement/illustration_1.jpg)
## 你将学到什么
![自然语言处理（十）：RAG与知识增强系统 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/rag-knowledge-enhancement/illustration_2.jpg)

- 概率分解公式 $P(y\mid q)=\sum_d P(d\mid q)P(y\mid q,d)$ 的含义，以及每一项对性能的具体影响
- ANN 索引（如 HNSW、IVF-PQ、ScaNN）为何要在召回率和延迟之间做权衡，以及在什么场景下值得牺牲召回换取速度
- 稠密检索、稀疏检索和混合检索的区别，为什么 **Reciprocal Rank Fusion (RRF)** 比简单的线性加权更有效
- 两阶段检索-重排序流程中，Cross-encoder 是如何带来 +12 nDCG 的提升的
- 切块时如何根据主题边界划分，而不是简单按字节数切割
- 查询改写、问题分解技术，以及 **HyDE**（假设性文档嵌入）的工作原理和应用场景
- **Self-RAG 和 Corrective RAG**：如何让模型自主决定是否使用检索结果，从而提升决策质量
- 如何根据数据规模、延迟要求和运维预算选择合适的向量数据库
## 前置知识

![NLP (10): RAG 与知识增强系统 —— 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/illustration_2.jpg)

- 熟悉 Embedding 的概念以及余弦相似度在语义表示中的应用（[第 2 篇：词向量与语言模型](/en/nlp/word-embeddings-lm/)）
- 了解 Decoder-only 架构的语言模型及其提示词设计方法（[第 6 篇：GPT](/en/nlp/gpt-generative-models/)）
- 能够熟练使用 Python，并对 TF-IDF 的基本功能有大致印象

---
## 1. RAG 的概率分解

![RAG 端到端流程图，展示离线索引与在线查询路径](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig1_rag_pipeline.png)

RAG 系统通过从少量检索到的文档 $\mathcal{D}_k$ 中进行边缘化计算，来回答查询 $q$：

$$P(y \mid q) \;=\; \sum_{d \in \mathcal{D}_k}\; \underbrace{P(d \mid q)}_{\text{retriever}}\; \cdot \; \underbrace{P(y \mid q, d)}_{\text{generator}}$$

在实际应用中，我们通常用两种方法来近似这个求和过程。一种是成本低且主流的方式（在 LangChain 中称为 **stuff**），它将 top-$k$ 文档拼接到 LLM 的上下文中，利用注意力机制隐式完成边缘化。另一种是理论上更严谨但代价更高的方式（**Fusion-in-Decoder**，Atlas 和原始 RAG 论文采用的方法），它对每个 $(q,d)$ 对单独编码，并在解码器中进行融合。对于大多数生产环境来说，使用 stuff 方法、设置 $k\in[3,8]$ 并搭配一个高质量的重排器，通常是最佳选择。

这种分解方式让 RAG 具备了独特的吸引力：

- **检索器** $P(d\mid q)$ —— 更新成本低。几分钟内即可重建文档索引，推理时无需 GPU。
- **生成器** $P(y\mid q,d)$ —— 固定不变。同一个 LLM 可以服务于不同领域。
- **引用能力** —— 每个回答都能追溯到具体的文档片段，既可向用户展示，也可用于审计或过滤幻觉内容。

### 一个极简但可靠的流水线

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# 1. 加载与分块——注意分块重叠，详见第 5 节
loader = DirectoryLoader("./docs", glob="**/*.md")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=64,
    separators=["\n\n", "\n", "。", ". ", " ", ""],
)
chunks = splitter.split_documents(loader.load())

# 2. 嵌入与索引构建——bge-small 是 2024 年的强力默认选项
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectordb = FAISS.from_documents(chunks, embedder)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

# 3. 强制基于上下文回答并允许拒答的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "仅根据上下文作答，并以 [i] 标注引用来源。"
     "如果上下文不足以回答问题，请回复『我不知道』。"),
    ("human", "上下文：\n{context}\n\n问题：{question}"),
])

def fmt(docs):
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = ({"context": retriever | fmt, "question": RunnablePassthrough()}
         | prompt | llm)

print(chain.invoke("上个季度 OAuth 流程做了哪些改动？").content)
```

三个细节让这套流水线更加可靠：
- **分隔符层级**：确保段落和句子不会被随意切断。
- **`temperature=0`**：消除生成中的随机性，使检索质量成为唯一的变量。
- **拒答机制**：通过简单的规则避免模型生成不可靠的回答，是最经济的幻觉防护手段。

### 什么时候 *不适合* 使用 RAG

| 场景 | 更合适的工具 |
|---|---|
| 风格适配（用品牌语气撰写内容） | 微调 / DPO |
| 稳定且狭窄的任务（如情感分析、NER） | 微调 |
| 模型已经擅长的推理链条 | 更优的 Prompt 设计 |
| 知识量小到能直接写入 system prompt | Prompt 工程 |
| 表格数据、数值事实、精确计算 | 工具调用 / 函数调用 |

RAG 的真正优势在于处理**规模大、变化快、需要审计**的知识场景：例如每周更新的文档、10 GB 的合规语料库，或者任何需要“展示来源”的场合。
## 2. Embedding 空间与 ANN 权衡

![Embedding 空间中的向量相似性及 FAISS 索引族的召回率与延迟权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig2_vector_similarity.png)

双塔编码器 $E_\theta:\text{text}\to\mathbb{R}^d$ 将查询和文档映射到同一个空间，检索任务本质上就是在该空间中基于余弦相似度寻找最近邻：

$$\operatorname{sim}(q,d) \;=\; \frac{E_\theta(q)\cdot E_\theta(d)}{\lVert E_\theta(q)\rVert \, \lVert E_\theta(d)\rVert}.$$

精确搜索的复杂度是 $O(N d)$ 每次查询——在 $10^5$ 规模时还能接受，到了 $10^7$ 就显得吃力，而 $10^9$ 则几乎不可行。**近似最近邻（ANN）** 索引通过牺牲一定的精度来换取延迟的数量级降低。生产环境中最常用的两类索引如下：

- **HNSW**（分层可导航小世界图）—— 基于多层近邻图的结构。从顶层入口开始贪心搜索，通常只需 $O(\log N)$ 次跳跃即可收敛。可调参数包括：$M$（图的度数，影响内存占用）和 $\textit{efSearch}$（束宽，用于权衡召回率与延迟）。这是大多数 RAG 场景下的最佳选择。
- **IVF-PQ** —— 结合粗粒度 $k$-means 聚类（IVF 桶）和残差的乘积量化（Product Quantization）。这种方法可以实现 8× 到 32× 的内存压缩，但会损失几个百分点的召回率。当索引无法完全装入内存时，它是首选方案。

图的右半部分展示了 1 M × 768-d 数据集上的帕累托前沿：HNSW 和 ScaNN 位于左上角，表明它们在高召回和低延迟之间取得了良好平衡；如果数据规模只有几十万，`Flat` 精确检索仍然可行；而当内存极度受限时，PQ-only 是最后的选择。

### 2024 年之后如何选择 Embedding 模型

| 模型 | 维度 | 特点 | 备注 |
|---|---|---|---|
| `text-embedding-3-large`（OpenAI） | 3072（支持截断） | 多语言能力强，可通过 Matryoshka 截断至 256 维 | 仅提供 API 接口 |
| `bge-large-en-v1.5`（BAAI） | 1024 | 在 MTEB 基准测试中表现顶尖，权重开源 | 支持本地部署 |
| `bge-m3` | 1024 | 单模型同时支持稠密、稀疏和多向量表示，覆盖 100+ 种语言 | 开源多语言模型中的佼佼者 |
| `nomic-embed-text-v1.5` | 768 | 支持长上下文（8K tokens），兼容 Matryoshka 截断 | Apache 2.0 许可 |
| `all-MiniLM-L6-v2` | 384 | 比大模型快 5–10 倍，MTEB 得分略低约 3 分 | 适合对延迟敏感的场景 |

**根据实际瓶颈选择模型**，而不是盲目追求排行榜名次。如果内存有限，可以使用 Matryoshka 截断技术将维度降到 256；如果是多语言场景，`bge-m3` 几乎无可匹敌；如果你有领域内的标注数据，微调一个 384 维的小模型往往能超越通用的 1024 维模型——因为余弦几何会更好地适配你的术语体系，冗余维度会被压缩，召回率也会随之提升。
## 3. 混合检索：稠密 + 稀疏 + RRF

![混合检索：通过 RRF 融合稠密检索与 BM25](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig3_hybrid_retrieval.png)

稠密检索擅长理解语义改写，但在三类查询上表现不如 BM25：罕见命名实体（如 `CVE-2024-3094`）、精确标识符（如 `订单 #482915`），以及长度不足以覆盖 Embedding 模型有效感受野的短查询。稀疏检索则恰恰相反：它对字面匹配敏感，但对同义词或语义变化无能为力。

**BM25** 的评分公式如下，用于计算查询 $q$ 和文档 $d$ 的相关性：

$$\operatorname{BM25}(q,d) \;=\; \sum_{t \in q} \operatorname{IDF}(t)\cdot
\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\!\left(1-b+b\,\frac{|d|}{\overline{|d|}}\right)}$$

其中 $k_1\!\approx\!1.2$，$b\!\approx\!0.75$。IDF 部分奖励稀有词的匹配，而长度归一化部分则避免长文档因篇幅优势占据主导地位。

```python
from rank_bm25 import BM25Okapi
import jieba

class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi([self._tok(d) for d in corpus])

    def _tok(self, text):
        return list(jieba.cut(text.lower()))  # 中文需要分词，不能直接 split()

    def search(self, query, k=20):
        scores = self.bm25.get_scores(self._tok(query))
        idx = scores.argsort()[::-1][:k]
        return [(self.corpus[i], float(scores[i])) for i in idx]
```

### 为什么 RRF 比线性混合更优

简单地用 $\alpha\cdot s_\text{dense} + (1-\alpha)\cdot s_\text{sparse}$ 进行线性加权，要求两种分数处于相同的量纲范围，但实际上它们相差甚远——BM25 分数无上限，而余弦相似度被限制在 $[-1,1]$ 区间内。**Reciprocal Rank Fusion**（Cormack 等，2009）巧妙地避开了这一问题，直接基于 *排名* 进行融合：

$$\operatorname{RRF}(d) \;=\; \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}, \qquad k \!=\! 60.$$

常数 $k=60$ 的作用是削弱单一检索器顶部结果的影响，因此一个文档如果在两个列表中都排第一，会比只在一个列表中排第一的文档更具优势。图中右下角的面板展示了在异构查询集上的典型提升效果：仅使用 BM25 ≈ 54，仅使用稠密检索 ≈ 62，使用 RRF 融合 ≈ 71，而 RRF 融合后再用 Cross-encoder 重排序 ≈ 78。

```python
def rrf_fuse(rankings, k=60, top_k=10):
    """rankings: 来自多个检索器的有序 doc_id 列表。"""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
```

在 RAG 系统中，混合检索很少成为性能瓶颈，但它却是最稳定、成本最低的召回率提升手段之一，通常能带来 +5 到 +10 的召回点提升。实际操作时，可以并行运行稠密检索和 BM25，通过 RRF 融合结果，并将前 50 个候选文档送入重排序模块进一步优化。
## 4. 重排：精度大头来自这里

![Bi-encoder 与 Cross-encoder 的对比，以及两阶段检索的质量-延迟曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig4_reranking.png)

Bi-encoder 分别对查询和文档进行编码，因此文档向量可以提前计算并建立索引。这种方式天生速度快，代价是模型无法同时看到查询和文档的 token。而 **Cross-encoder** 则将两者拼接在一起 —— `[CLS] q [SEP] d [SEP]` —— 并通过完整的 Transformer 模型处理这对输入，最终输出一个相关性分数。联合注意力机制能够捕捉否定词、问句的不同表述方式以及词级别的交互关系，这些都是 Bi-encoder 难以做到的。不过，Cross-encoder 的代价是每次查询需要进行 $O(k)$ 次联合推理，因此它通常只适用于小规模候选集。

**两阶段流程** 是目前的标准做法：先用 Bi-encoder（或混合方法）筛选出前 50–100 个候选结果，再用 Cross-encoder 进行重排序，最终选出前 5 个。右图展示了 MS MARCO 开发集上的典型曲线：单次重排序能提升 12 点 nDCG，候选集规模翻倍后还能再提升 1.5 点，如果预算允许使用基于 LLM 的列表级重排序，还能再提升 2.5 点。

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

def retrieve_and_rerank(query, hybrid_retriever, top_k_retrieve=50, top_k_final=5):
    candidates = hybrid_retriever.search(query, k=top_k_retrieve)
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs, batch_size=32)
    order = scores.argsort()[::-1][:top_k_final]
    return [candidates[i] for i in order]
```

**实践建议。** 选择一个训练数据分布接近你领域需求的重排序模型：如果是多语言通用场景，推荐 `bge-reranker-v2-m3`；如果是英文网页风格查询，可以选择 `ms-marco-MiniLM-L-12-v2`；如果有几千条标注数据，不妨微调一个领域专用的重排序模型。注意限制 `max_length` 参数，因为更长的文本片段不仅不会带来额外收益，反而会拖慢速度。尽量在 GPU 上批量处理候选对。最重要的是，**一定要测试效果** —— 重排序的收益与召回率密切相关，因此在真实流量上运行 A/B 测试，验证其延迟成本是否值得。
## 5. 切块：沉默的精度泄漏点

![固定切块、递归切块与语义切块，以及质量与大小的权衡曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig5_chunking.png)

用户查询的目标是“信息”，而不是“字节”，但我们的索引只能处理切块（chunk）。目前主流的切块策略有三种：

- **固定大小**（$N$ tokens，无重叠）—— 简单直接，结果确定，但在边界场景下表现糟糕。能够回答问题的信息往往跨越了句子边界，而这种切块方式会将句子硬生生切断。
- **递归切块 + 重叠** —— 这是 LangChain 的默认方法。首先尝试按分隔符层级 `["\n\n", "\n", "。", ". ", " ", ""]` 进行分割；如果某个切块仍然过长，则递归到更细粒度的层级。64–128 token 的重叠设计是关键：任何靠近边界的句子都会出现在两个切块中，从而避免检索时遗漏重要信息。
- **语义切块** —— 对每个句子进行 Embedding，计算相邻句子之间的余弦距离，并在距离出现显著波动的地方进行切割。话题的切换自然成为切块的边界，因此每个切块内部的主题一致性更强。虽然构建索引的速度较慢，但在处理长篇文本时效果显著提升。

右下角的图表总结了实际中的权衡：检索命中率（Hit@5）和答案忠实度（faithfulness）都在 256–512 token 的范围内达到峰值。切块过小会导致答案被分割得支离破碎；切块过大则会使向量表示变得稀释（chunk 向量的余弦值变成噪声较大的平均值）。对于代码或结构化文档，可以将切块大小扩展到 768–1024 token，因为这些内容的最小语义单元通常更大。

### 父子切块：小块嵌入，大块返回

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_split = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_split  = RecursiveCharacterTextSplitter(chunk_size=400,  chunk_overlap=40)

retriever = ParentDocumentRetriever(
    vectorstore=Chroma(embedding_function=embedder),
    docstore=InMemoryStore(),
    child_splitter=child_split,
    parent_splitter=parent_split,
)
retriever.add_documents(raw_docs)
```

这种方法的核心思想是：用 400-token 的小切块嵌入来保证**精度**（小窗口的余弦相似度更敏锐），同时返回 2000-token 的大切块作为**上下文**（让模型能够看到完整的段落信息，从而更好地生成答案）。这是少数几乎不增加成本，却能稳定提升检索和生成质量的 RAG 技巧之一。
## 6. 查询优化
用户输入的查询，通常和索引系统期待的形式有很大差距。这里有三个实用技巧值得掌握：

**查询改写。** 利用一次简短的 LLM 调用，将问题重新表述为更紧凑、更适合检索的形式。例如，把 *"昨天谁处理了那个 OAuth 的事情？"* 改成 *"OAuth 2.0 access token 撤销事件，2024-04-23"*。这种方式不仅速度快、成本低，还能显著提升对话式查询的召回效果。

**多查询 / 问题分解。** 面对复杂问题时，可以生成 $n$ 种不同的改写或子问题，分别进行检索，然后将结果合并并重新排序。RAG-Fusion 就是基于这种多查询策略，再结合 RRF（Reciprocal Rank Fusion）来优化结果。

```python
def multi_query(question, llm, n=4):
    prompt = (f"请用 {n} 种不同方式改写以下问题，以尽可能提高文档召回率。"
              f"每种改写占一行。\n\nQ: {question}")
    return [q.strip() for q in llm.invoke(prompt).content.split("\n") if q.strip()]
```

**HyDE —— 假设性文档嵌入。** 让 LLM 根据问题生成一个“看似合理”的答案，然后用这个假设的答案生成嵌入向量并进行检索。虽然听起来有些反直觉，但效果出奇地好：LLM 生成的答案即使不完全准确，也往往与真实答案处于相似的语义空间中，因此它的嵌入向量比原始问题更接近相关文档。这种方法的代价是一次额外的 LLM 调用，但在针对技术性语料库的简短、模糊查询中，收益尤为显著。

```python
def hyde_retrieve(question, llm, vectordb, k=6):
    hypothetical = llm.invoke(f"请用一段话回答以下问题：{question}").content
    return vectordb.similarity_search(hypothetical, k=k)
```
## 7. Self-RAG 与 Corrective RAG
![Self-RAG / Corrective RAG 控制流程，包含反思标记](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig6_self_rag.png)

传统的 RAG 方法在检索时是**无条件的**，这种方式既低效又存在风险：闲聊场景下根本不需要检索，而不相关的检索结果反而可能误导模型。为了解决这一问题，**Self-RAG**（Asai 等，2024）和 **Corrective RAG**（Yan 等，2024）将检索决策权交给了模型本身，通过引入“反思标记”（reflection tokens）来实现：

| 标记       | 决策内容                                   |
|------------|------------------------------------------|
| `[Retrieve]` | 是否需要进行检索？                       |
| `[ISREL]`    | 每个召回的文档片段是否与问题相关？（逐文档判断） |
| `[ISSUP]`    | 生成的答案是否有证据支持？               |
| `[ISUSE]`    | 答案是否对用户有实际帮助？               |

整个控制流程可以看作一个小型决策图：首先发出 `[Retrieve]` 标记，根据相关性评分进行分支判断；如果所有召回的文档片段都不相关，则进入纠正分支 —— 改写查询、调用网页搜索、重新检索并重新评估。最后的 `[ISSUP]` 和 `[ISUSE]` 闸门确保答案基于保留下来的证据生成，而不是单纯依赖模型的先验知识。实验表明，Self-RAG 在长文本问答基准测试中相比同规模的非反思型 RAG 提升了 5 到 9 个百分点。

你并不需要专门微调一个 Self-RAG 模型来使用这种模式。同样的控制逻辑可以通过普通大语言模型（LLM）结合几次结构化调用来实现，只是会带来一定的延迟开销：

```python
def self_rag(query, vectordb, llm, web_search):
    if "no" in llm.invoke(
        f"这个问题需要外部知识吗？yes/no：{query}"
    ).content.lower():
        return llm.invoke(query).content, []

    docs = vectordb.similarity_search(query, k=8)
    grades = [
        "yes" in llm.invoke(
            f"该段落是否与问题相关？\n"
            f"Q: {query}\nP: {d.page_content[:600]}\nyes/no："
        ).content.lower()
        for d in docs
    ]
    kept = [d for d, g in zip(docs, grades) if g]

    if not kept:                                                # 纠正分支
        rewritten = llm.invoke(f"改写为搜索查询：{query}").content
        kept = web_search(rewritten, k=5)

    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(kept, 1))
    answer = llm.invoke(
        f"只用上下文回答，并以 [i] 标注引用。\n\n{context}\n\nQ: {query}"
    ).content
    return answer, kept
```

这种方法的代价是显而易见的 —— 它比普通的 RAG 方法多出了 3 到 4 倍的大语言模型调用次数。然而，在那些“答错的后果比答慢更严重”的场景中，这种额外开销是完全值得的。
## 8. 向量数据库的实践选型
![向量数据库能力雷达图与单节点吞吐对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig7_vectordb.png)

**FAISS** 是一个库，而不是传统意义上的数据库。它追求极致的速度，但不提供持久化、过滤功能，也不支持并发处理，非常适合嵌入到其他服务中使用，或者用于百万级别的离线实验。

**Chroma** 是最容易上手的选择：只需 `pip install` 即可安装，支持数据落盘存储，并且能无缝融入你的技术栈。单节点场景下，它的容量上限大约在 $10^5$ 到 $10^6$ 条向量之间。

**Milvus** 是开源领域的重量级选手，支持分布式架构和多租户模式，内置 HNSW、IVF-PQ 和 DiskANN 等索引算法，同时支持标量过滤和原生的混合检索（dense + sparse）。如果你需要处理 $10^7$ 条以上的向量，并且对运维能力有较高要求（如副本管理、滚动升级、可观测性等），Milvus 是不二之选。

**Weaviate** 在相同规模下与 Milvus 展开竞争，主打一流的混合检索能力和简洁的 schema 设计。相比 Milvus，Weaviate 的运维复杂度稍低，但在原始吞吐性能上略逊一筹。

**Pinecone** 是托管服务的最佳代表：零运维、API 设计优秀，但价格相对较高。如果你的团队更关注开发效率，而非基础设施成本，那么 Pinecone 是一个值得考虑的选择。

左侧的雷达图是对各数据库能力的主观总结，右侧的柱状图则展示了在 recall ≥ 0.95 的条件下，单节点的 QPS 性能量级。这些数据仅供参考，实际选型时还需要结合你的语料库、查询模式以及过滤条件进行测试。

### 决策树

- **< 1 万条向量，快速原型开发** → 使用内存版 FAISS 或 Chroma。
- **$10^5$–$10^6$ 条向量，单节点部署，简单过滤需求** → 使用支持持久化的 Chroma 或 Weaviate。
- **$10^7$+ 条向量，多租户支持，混合检索，私有化部署** → 使用 Milvus。
- **任意规模，无运维资源，可接受云调用延迟** → 使用 Pinecone。
- **仅需针对静态语料进行 top-$k$ 检索** → 使用支持落盘存储的 FAISS。
## 9. 评估一个 RAG 系统

RAG 系统有两种主要的失效模式：检索效果差和生成内容差。为了准确评估系统性能，必须分别对这两部分进行测量。

**检索指标**（需要标注好的 query→相关 chunk 对应关系）：

- **Hit@k** —— 检查前 $k$ 个结果中是否包含至少一个相关 chunk。虽然简单粗暴，但非常实用，适合快速判断。
- **MRR** —— 计算第一个相关 chunk 的排名倒数的平均值。这个指标对排名位置非常敏感，适合用来衡量排序质量。
- **nDCG@k** —— 归一化的折扣累计增益。当相关性是分级而非二元（如“相关”或“不相关”）时，这是最合适的指标。

**生成指标**（使用 `gpt-4o` 或 `claude-sonnet` 作为裁判模型）：

- **Faithfulness** —— 答案中的每一句话是否都能在检索到的上下文中找到依据？这是检测模型“幻觉”的关键。
- **Answer Relevance** —— 答案是否真正解决了用户提出的问题？
- **Context Relevance** —— 检索到的内容是否对生成答案有实际帮助？

目前业内常用的工具是 **RAGAS** 和 **TruLens**，它们封装了这些评估逻辑，省去了手动设计 prompt 的麻烦。每次调整 Prompt、更换 Embedding 模型或修改分块策略时，建议用一组包含 50–200 条查询的黄金测试集重新跑一遍评估。如果没有这样的闭环流程，你根本无法确定昨天所谓的“优化”到底有没有效果。
## 10. FAQ

**我的 RAG 系统太慢了，该从哪里入手优化？**  
首先要做性能分析（profile）。查询嵌入的时间通常在个位数毫秒级别，HNSW 检索也是个位数毫秒，而第 90 百分位的延迟几乎总是卡在 LLM 的调用上。优化的第一步是缓存重复的查询请求，接着尝试压缩提示词（父子分块法在这里很有帮助），然后可以考虑换一个更轻量的重排序器（reranker），最后才去调整索引。

**我的 RAG 系统总是在“幻觉”，怎么办？**  
按照以下三层逐步解决：  
(a) 收紧提示词设计，要求模型必须引用文档内容并允许拒绝回答；  
(b) 引入交叉编码器（cross-encoder reranker），确保上下文的相关性；  
(c) 升级到 Self-RAG，并加入 `[ISSUP]` 门控机制。如果经过这三步仍然存在幻觉问题，那很可能是你的检索器（retriever）召回的内容质量太差 —— 回头重新审视分块策略和 Embedding 模型的选择。

**是否需要微调 Embedding 模型？**  
如果你有 ≥ 5K 条标注好的 query-passage 对，并且应用场景明确（如法律、生物医学或内部术语），那么值得微调。提升 5–10 个百分点的召回率会对下游任务产生显著的累积效应。但如果只有少量样本，就别折腾了 —— `bge-large` 在通用英语场景下已经非常优秀。

**选择密集检索（dense）还是混合检索（hybrid）？**  
永远优先从密集检索开始。当用户抱怨某些精确匹配的查询结果找不到时，再引入 BM25 + RRF。这种改动的边际开发成本不过一个下午，但召回率的提升却是长期受益的。

**如何保持索引的实时性？**  
对于小规模语料库，可以在写入时直接生成嵌入（embed-on-write）；对于其他场景，则采用定期批量重建的方式。需要注意的是，删除操作是个坑 —— 大多数近似最近邻（ANN）索引仅支持逻辑删除（tombstone），因此建议规划一个重建周期（大多数团队每周一次即可），并将其流程化、自动化，让它变得无趣但可靠。
## 11. 推理成本：RAG 真正贵在哪里

一次 RAG 回答不仅仅是调用一个 LLM 那么简单。在典型的生产环境中，回答用户一个问题的延迟分布大致如下：

| 阶段 | p50 延迟 | 主要瓶颈 |
|------|---------|---------|
| Query embedding | 15--30 ms | 小型编码器（bge-small，110M 参数）的一次前向计算 |
| 向量检索（top-50，HNSW，1000 万文档） | 8--20 ms | 内存带宽，而非算力 |
| BM25 检索 | 5--15 ms | 倒排索引查询，受 IO 限制 |
| RRF 融合 | <1 ms | 列表合并操作 |
| Reranker（top-50 → top-5，cross-encoder） | 80--200 ms | 50 次 cross-encoder 前向计算，支持批处理 |
| LLM 生成（4K 上下文，200 token 答案） | 1500--4000 ms | 解码自回归，真正的性能瓶颈 |

reranker 是最容易被忽视的优化环节。以 bge-reranker-base 为例，在单张 A10 GPU 上正确使用批处理时，处理 50 对样本大约需要 150 毫秒；但如果串行调用同样的代码，则可能耗时 2 秒。**务必记住：批处理是关键**。

LLM 的调用时间比其他所有阶段加起来还要慢一个数量级。这意味着：**缩短答案长度比加速检索更重要**。除非有特殊需求，建议将 `max_tokens` 限制在 300 以内。对于最终的答案合成步骤，可以考虑使用更小的模型——只要检索到的上下文质量足够好，Llama-3-8B 或 Qwen-2.5-7B 已经能够很好地完成事实性合成任务。

从成本角度来看，GPT-4o-mini 的输入价格为 $0.15/M token，输出价格为 $0.6/M token。假设每次查询使用 4K 输入和 200 token 输出，单次查询的成本为 **$0.0007**。如果每天处理 100 万次查询，日成本为 $700，年成本则高达 $25 万。相比之下，使用 4 张 A10 GPU 自托管一个 7B 参数的模型，月成本约为 $400（包括硬件折旧和电力消耗）。在这种情况下，每日查询量达到 5 万次即可实现盈亏平衡。
## 12. 生产 RAG 真正的失败模式

以下是我在实际部署的系统中遇到并调试过的五种失败模式，以及对应的解决方案。

**失败 1：答案正确，但引用了错误的内容块**  
大语言模型（LLM）有时会“编造”看似合理的引用来源。解决方法是要求模型在生成答案之前，先逐字引用支持结论的具体句子（例如提示词可以设计为：“请先引用原文中的句子，然后基于此进行解释”）。这种方法能够将引用错误的发生率降低约 70%。

**失败 2：检索结果偏向 FAQ 页面，而非实际文档**  
FAQ 页面通常内容简短、信息密集，与用户提问的嵌入距离非常接近，因此容易排在更长、更详细的原始文档前面。解决办法是将 FAQ 单独建立索引，并设置规则：只有当没有其他文档得分超过某个阈值时，才回退到 FAQ。另一种方法是在融合阶段对 FAQ 的内容块进行降权处理。

**失败 3：用户提问“比较 X 和 Y”，但只检索到与 X 相关的内容**  
查询的密集嵌入（dense embedding）容易被提及次数更多的实体主导，导致另一部分信息被忽略。解决方法是对查询进行分解——使用一个小规模的 LLM 将“比较 X 和 Y”拆分为两个独立的检索任务，分别获取相关内容后再合并结果。

**失败 4：多语言查询只返回单一语言的结果**  
例如，用户用中文提问时，系统可能只检索到中文文档，即使英文文档实际上更相关。解决方法是采用多语言嵌入模型（如 bge-m3 或 multilingual-e5-large），并在执行 BM25 检索之前，将查询翻译成语料库中占主导地位的语言。

**失败 5：重排序器（reranker）将正确答案排到了后面**  
Cross-encoder 类型的重排序器通常在 MS MARCO 数据集上训练，而该数据集的问题风格较为固定。对于领域外的输入（如特别长的问题、对话式提问或包含代码片段的查询），打分效果往往不理想。解决方法是使用 500 到 2000 条领域内的查询-文档对对重排序器进行微调。这一过程成本较低，单 GPU 训练大约只需 2 小时，且几乎总是值得投入。