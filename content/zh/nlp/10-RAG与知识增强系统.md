---
title: "自然语言处理（十）：RAG 与知识增强系统"
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
一个知识被冻结的语言模型就像一个“自信的骗子”：它读不了昨天的事故报告、公司的 Wiki 页面，也看不到今天早上刚发布的补丁说明。一旦你提问，它就会生成语法完美但事实错误的答案。**检索增强生成（RAG）** 打破了这一僵局，其核心在于将“记忆”与“推理”分离——让大语言模型（LLM）保持小巧稳定，而把易变的知识放进可随时更新的外部存储中。生成答案前，先检索相关证据，并将其作为条件输入模型。

理念一句话就能说清，工程实现才是本文的重点。一个真正落地的 RAG 系统通常有十几个可调参数：分块大小、Embedding 模型、索引类型、$k$、混合权重、重排序深度、提示模板、引用格式、拒答策略……而且它们彼此耦合。接下来，我们将逐一剖析每个参数背后的数学原理、权衡取舍，并附上可运行的代码。

<!-- wanx-hero -->
![自然语言处理（十）：RAG与知识增强系统 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/illustration_1.png)
## 你将学到什么
![自然语言处理（十）：RAG与知识增强系统 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/illustration_2.png)

- 概率分解公式 $P(y\mid q)=\sum_d P(d\mid q)P(y\mid q,d)$ 的含义，以及每一项的实际代价
- 为何 ANN 索引（HNSW、IVF-PQ、ScaNN）要在召回率和延迟之间权衡，以及何时值得为此付费
- 稠密检索、稀疏检索与混合检索的区别，以及为什么 **Reciprocal Rank Fusion (RRF)** 比线性加权更优
- 两阶段“检索-重排序”流程中，Cross-encoder 如何带来 +12 nDCG 的提升
- 如何按主题边界而非字节数进行切块
- 查询改写、问题分解，以及 **HyDE**（假设性文档嵌入）的原理
- **Self-RAG 与 Corrective RAG**：如何让模型自主决定是否使用检索结果
- 如何根据规模、延迟和运维预算选择合适的向量数据库

## 前置知识

![NLP (10): RAG 与知识增强系统 —— 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/illustration_2.png)

- 熟悉 Embedding 与余弦相似度的语义几何（[第 2 篇：词向量与语言模型](/zh/nlp/02-词向量与语言模型)）
- 了解 Decoder-only 语言模型与提示设计（[第 6 篇：GPT](/zh/nlp/06-gpt与生成式语言模型)）
- 能熟练使用 Python，并对 TF-IDF 有模糊记忆

---

## 1. RAG 的概率分解

![RAG 端到端流程图，展示离线索引与在线查询路径](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig1_rag_pipeline.png)

RAG 系统通过在检索到的文档集合 $\mathcal{D}_k$ 上边缘化来回答查询 $q$：

$$P(y \mid q) \;=\; \sum_{d \in \mathcal{D}_k}\; \underbrace{P(d \mid q)}_{\text{retriever}}\; \cdot \; \underbrace{P(y \mid q, d)}_{\text{generator}}$$

实践中，我们通常用两种方式近似该求和。主流且低成本的方法（LangChain 中称为 **stuff**）将 top-$k$ 文档拼接到 LLM 上下文中，让注意力机制隐式完成边缘化。理论上更严谨但昂贵的方法（**Fusion-in-Decoder**，见 Atlas 和原始 RAG 论文）则对每个 $(q,d)$ 对单独编码，并在解码器中融合。对大多数生产系统而言，使用 stuff 方法（$k\in[3,8]$）并搭配优质重排序器是最佳选择。

这种分解使 RAG 具备三大优势：

- **检索器**（$P(d\mid q)$）—— 更新成本低：几分钟即可重建索引，推理无需 GPU。
- **生成器**（$P(y\mid q,d)$）—— 冻结不变：同一 LLM 可服务多个领域。
- **可溯源性**—— 每个主张都能追溯到具体片段，便于展示、审计或过滤幻觉。

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

三个细节至关重要：**分隔符层级**确保段落和句子完整；**`temperature=0`** 消除生成随机性，使检索成为唯一变量；**系统提示中的拒答条款**是最廉价的幻觉防护。

### 何时 *不应* 使用 RAG

| 场景 | 更佳方案 |
|---|---|
| 风格适配（按品牌语气写作） | 微调 / DPO |
| 稳定窄域任务（情感分析、NER） | 微调 |
| 模型已擅长的推理链 | 更优提示 |
| 知识可塞进系统提示 | 提示工程 |
| 表格/数值事实、精确计算 | 工具调用 |

RAG 在知识**体量大、频繁变动或需审计溯源**时表现最佳：如每周更新的文档、10 GB 合规语料库，或任何要求“展示来源”的场景。

---

## 2. Embedding 空间与 ANN 权衡

![Embedding 空间中的向量相似性及 FAISS 索引族的召回率与延迟权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig2_vector_similarity.png)

双塔编码器 $E_\theta:\text{text}\to\mathbb{R}^d$ 将查询与文档映射至同一空间，检索即基于余弦相似度的最近邻搜索：

$$\operatorname{sim}(q,d) \;=\; \frac{E_\theta(q)\cdot E_\theta(d)}{\lVert E_\theta(q)\rVert \, \lVert E_\theta(d)\rVert}.$$

精确搜索复杂度为 $O(N d)$ 每查询——$10^5$ 规模尚可，$10^7$ 已吃力，$10^9$ 则不可行。**近似最近邻（ANN）** 索引以精度换数量级延迟下降。生产中最常用的两类：

- **HNSW**（分层可导航小世界图）—— 多层邻近图结构，从顶层入口贪心下降，$O(\log N)$ 步收敛。可调参数：$M$（图度数，影响内存）与 $\textit{efSearch}$（束宽，权衡召回与延迟）。对多数 RAG 场景是甜点。
- **IVF-PQ** —— 粗粒度 $k$-means 聚类（IVF 桶）加残差乘积量化。内存压缩达 8×–32×，但损失若干召回点。适用于索引无法全载入内存的场景。

图右半部展示了 1M × 768-d 语料上的帕累托前沿：HNSW 与 ScaNN 位于左上角；若仅数十万向量，`Flat` 精确搜索仍可行；内存极度受限时，PQ-only 是最后选择。

### 2024 年后如何选择 Embedding 模型

| 模型 | 维度 | 优势 | 备注 |
|---|---|---|---|
| `text-embedding-3-large`（OpenAI） | 3072（可截断） | 强大多语言支持，Matryoshka 截断至 256 | 仅 API |
| `bge-large-en-v1.5`（BAAI） | 1024 | MTEB 顶尖，开源权重 | 可本地部署 |
| `bge-m3` | 1024 | 单模型集成稠密+稀疏+多向量，支持 100+ 语言 | 开源多语言首选 |
| `nomic-embed-text-v1.5` | 768 | 长上下文（8K tokens），Matryoshka 支持 | Apache 2.0 |
| `all-MiniLM-L6-v2` | 384 | 快 5–10 倍，MTEB 低约 3 分 | 延迟敏感场景 |

**按瓶颈选模型，而非排行榜**。内存紧张？用 Matryoshka 截断至 256 维。多语言？`bge-m3` 几乎无敌。若有领域标注对，微调 384 维模型常优于通用 1024 维模型——余弦几何会适配你的术语，冗余维度坍缩，召回跃升。

---

## 3. 混合检索：稠密 + 稀疏 + RRF

![混合检索：通过 RRF 融合稠密检索与 BM25](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig3_hybrid_retrieval.png)

稠密检索理解释义，但在三类查询上败给 BM25：罕见命名实体（`CVE-2024-3094`）、精确标识符（`订单 #482915`），以及短于 Embedding 模型有效感受野的查询。稀疏检索则相反：字面匹配强，但对同义词盲。

**BM25** 对查询 $q$ 与文档 $d$ 的评分公式为：

$$\operatorname{BM25}(q,d) \;=\; \sum_{t \in q} \operatorname{IDF}(t)\cdot
\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\!\left(1-b+b\,\frac{|d|}{\overline{|d|}}\right)}$$

其中 $k_1\!\approx\!1.2$，$b\!\approx\!0.75$。IDF 项奖励稀有词匹配，长度归一化项防止长文档主导。

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

### 为何 RRF 优于线性混合

朴素方法 $\alpha\cdot s_\text{dense} + (1-\alpha)\cdot s_\text{sparse}$ 要求分数同尺度，但 BM25 无界而余弦在 $[-1,1]$。**Reciprocal Rank Fusion**（Cormack 等，2009）绕过校准问题，直接基于 *排名* 融合：

$$\operatorname{RRF}(d) \;=\; \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}, \qquad k \!=\! 60.$$

$k=60$ 抑制单一检索器顶部结果的影响，故在两个列表均排第一的文档优于仅在一个列表排第一者。图右下角显示典型提升：BM25-only ≈ 54，稠密-only ≈ 62，RRF ≈ 71，RRF + Cross-encoder 重排序 ≈ 78。

```python
def rrf_fuse(rankings, k=60, top_k=10):
    """rankings: 来自多个检索器的有序 doc_id 列表。"""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
```

混合检索很少成为 RAG 瓶颈，却是最稳、最便宜的 +5 至 +10 召回点来源。并行运行稠密与 BM25，用 RRF 融合，送 50 个候选给重排序器。

---

## 4. 重排序：精度大头所在

![Bi-encoder 与 Cross-encoder 的对比，以及两阶段检索的质量-延迟曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig4_reranking.png)

Bi-encoder **分别**编码查询与文档，故文档向量可预计算索引。速度快，代价是模型从未同时看到两者 token。**Cross-encoder** 则拼接为 `[CLS] q [SEP] d [SEP]`，用完整 Transformer 处理，输出单一相关性 logit。联合注意力能捕捉否定、问句变体及词级交互，Bi-encoder 难以企及。代价是每查询 $O(k)$ 次联合前向，故仅适用于小候选池。

**两阶段流水线**是标准方案：Bi-encoder（或混合）取 top 50–100，Cross-encoder 重排至最终 5 个。右图显示 MS MARCO dev 典型曲线：单次重排 +12 nDCG，深度翻倍再 +1.5，若用列表级 LLM 重排还能 +2.5。

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

**实践建议**：选接近你领域的重排序器——多语言通用选 `bge-reranker-v2-m3`，英文网页查询选 `ms-marco-MiniLM-L-12-v2`，若有数千标注对则微调领域专用模型。限制 `max_length`（长文本不增益反拖慢）。GPU 上批量处理候选对。最重要的是 **实测**——重排序收益与检索召回叠加，上线前务必 A/B 测试验证延迟成本是否值得。

---

## 5. 切块：沉默的精度泄漏点

![固定切块、递归切块与语义切块，以及质量与大小的权衡曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig5_chunking.png)

查询目标是“信息”而非“字节”，但索引只认切块。主流策略有三：

- **固定大小**（$N$ tokens，无重叠）—— 简单确定，但边界处表现极差。回答所需信息常跨句子边界，却被硬切。
- **递归切块 + 重叠** —— LangChain 默认。尝试分隔符层级 `["

", "
", ". ", " ", ""]`；若仍过大则递归下一层。64–128 token 重叠是关键：边界附近句子出现在两块中，避免检索遗漏。
- **语义切块** —— 对每句嵌入，算相邻句余弦距离，在突变处分割。话题切换成自然边界，块内更连贯。构建稍慢，但对散文效果显著更好。

右下角图总结实证权衡：检索 Hit@5 与答案忠实度均在 256–512 token 达峰。块太小致答案碎片化；太大则嵌入稀释（余弦成噪声平均）。代码或结构化文档可推至 768–1024 token，因其语义单元更大。

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

检索器用 400-token 子块嵌入保**精度**（小窗口余弦更锐利），但返回 2000-token 父块给 LLM 供**上下文**（模型需周边段落才能作答）。这是少数几乎零成本却稳提检索与生成质量的 RAG 技巧。

---

## 6. 查询优化

用户输入的查询，往往不是索引期待的形式。三个实用技巧：

**查询改写**：一次简短 LLM 调用将问题转为更稠密、检索友好的形式。如 *“昨天谁处理了 OAuth 那事？”* → *“OAuth 2.0 access token 撤销事件，2024-04-23”*。快速廉价，显著提升对话式查询召回。

**多查询 / 分解**：对复杂问题，生成 $n$ 种改写或子问题，分别检索后合并候选再重排。RAG-Fusion 即多查询 + RRF。

```python
def multi_query(question, llm, n=4):
    prompt = (f"请用 {n} 种不同方式改写以下问题，以尽可能提高文档召回率。"
              f"每种改写占一行。\n\nQ: {question}")
    return [q.strip() for q in llm.invoke(prompt).content.split("\n") if q.strip()]
```

**HyDE —— 假设性文档嵌入**：让 LLM 为查询写个“合理答案”，再用此答案嵌入检索。反直觉但有效：幻觉答案与真实答案同处语义邻域，其嵌入比原问题更近相关文档。代价是一次额外 LLM 调用，在技术语料库的短模糊查询上收益最大。

```python
def hyde_retrieve(question, llm, vectordb, k=6):
    hypothetical = llm.invoke(f"请用一段话回答以下问题：{question}").content
    return vectordb.similarity_search(hypothetical, k=k)
```

---

## 7. Self-RAG 与 Corrective RAG

![Self-RAG / Corrective RAG 控制流程，包含反思标记](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig6_self_rag.png)

传统 RAG **无条件检索**，既浪费（闲聊无需检索）又危险（无关证据误导模型）。**Self-RAG**（Asai 等，2024）与 **Corrective RAG**（Yan 等，2024）将检索变为模型自主决策，通过反思标记表达：

| 标记 | 决策 |
|---|---|
| `[Retrieve]` | 是否应检索？ |
| `[ISREL]` | 每个召回块是否相关？（逐文档） |
| `[ISSUP]` | 生成答案是否有证据支持？ |
| `[ISUSE]` | 答案对用户是否有用？ |

控制流是小图：发 `[Retrieve]`，按相关性分级分支；若**无**块被评为相关，则转入纠正动作——改写查询、调网页搜索、重检、重评。最终 `[ISSUP]` / `[ISUSE]` 闸门确保答案基于保留证据，而非模型先验。Self-RAG 在长问答基准上比同规模非反思 RAG 提升 5–9 点。

无需专用微调模型即可用此模式。相同控制流可在普通 LLM 上运行，只需几次结构化调用，代价是延迟：

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

代价真实存在——LLM 调用达传统 RAG 的 3–4 倍——但当答错代价高于答慢时，完全值得。

---

## 8. 向量数据库实战选型

![向量数据库能力雷达图与单节点吞吐对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG与知识增强系统/fig7_vectordb.png)

**FAISS** 是库非数据库。纯速度，无持久化、无过滤、无并发——适合嵌入服务或百万级离线实验。

**Chroma** 最易上手：pip install，落盘持久化，快速集成。单节点上限约 $10^5$–$10^6$ 向量/集合。

**Milvus** 是开源重量级：分布式、多租户，原生支持 HNSW/IVF-PQ/DiskANN、标量过滤、混合（稠密+稀疏）搜索。需真实运维（副本、升级、可观测性）且向量超 $10^7$ 时选用。

**Weaviate** 在同规模竞争，主打一流混合搜索与简洁 schema。运维略易 Milvus，吞吐略低。

**Pinecone** 是托管选项：零运维、API 优秀、价格高。当工程时间是瓶颈而非基础设施成本时选用。

左图雷达为主观总结；右图是 recall ≥ 0.95 下单节点 QPS 量级。仅作起点——务必在自有语料、查询、过滤选择性下实测。

### 决策树

- **< 1 万向量，原型** → 内存 FAISS 或 Chroma。
- **$10^5$–$10^6$，单节点，简单过滤** → 持久化 Chroma 或 Weaviate。
- **$10^7$+，多租户，混合搜索，私有部署** → Milvus。
- **任意规模，无运维人力，可接受云延迟** → Pinecone。
- **仅需静态语料 top-$k$** → 落盘 FAISS。

---

## 9. 评估 RAG 系统

RAG 有两大失效模式——检索差与生成差——必须分开评估。

**检索指标**（需标注 query→相关 chunk 对）：

- **Hit@k** —— top-$k$ 中是否有相关块？粗但可操作。
- **MRR** —— 首个相关块排名倒数的平均。对位置敏感。
- **nDCG@k** —— 归一化折扣累计增益。相关性分级时最适用。

**生成指标**（用 `gpt-4o` 或 `claude-sonnet` 作裁判）：

- **Faithfulness** —— 答案每主张是否均有上下文支持？抓幻觉。
- **Answer relevance** —— 答案是否解答问题？
- **Context relevance** —— 召回块是否真有用？

标准工具是 **RAGAS** 与 **TruLens**；二者封装提示，免重造轮子。每次改提示、换嵌入、调切块，都应在 50–200 查询黄金集上重跑评估。无此闭环，无法判断昨日“优化”是否真有效。

---

## 10. 常见问题

**RAG 太慢，从哪优化？** 先 profiling。查询嵌入个位数 ms；HNSW 检索个位数 ms；90 分位延迟几乎总在 LLM 调用。缓存相同查询，缩小提示（父子切块有帮助），考虑更小重排序器，最后才动索引。

**RAG 幻觉怎么办？** 三层依次解决：(a) 提示收紧，要求引用并允许拒答；(b) 加 Cross-encoder 重排序保上下文相关；(c) 升级 Self-RAG 加 `[ISSUP]` 闸门。若仍幻觉，则检索器召回垃圾——回头查切块与嵌入选择。

**要微调嵌入模型吗？** 若有 ≥5K 标注 query-passage 对且领域明确（法律、生物医学、内部术语），则值得。5–10 点召回提升会下游累积。若仅少量样本，则不必——`bge-large` 在通用英语已极佳。

**稠密还是混合？** 总从稠密开始。用户抱怨缺精确匹配查询那天，再加 BM25 + RRF。边际工程成本一下午，召回提升永久。

**如何保索引新鲜？** 小语料 embed-on-write；其余定期批量重建。删除是坑——多数 ANN 索引仅支持墓碑，故规划重建周期（多数团队周更即可），并使其无聊化。

---

## 11. 推理成本：RAG 真正贵在哪

RAG 响应不止 LLM 调用。典型生产流水线回答一问的延迟分布：

| 阶段 | p50 延迟 | 主导因素 |
|-------|-------------|----------------|
| 查询嵌入 | 15–30 ms | 小编码器（bge-small，110M 参数）单次前向 |
| 向量搜索（top-50，HNSW，10M 文档） | 8–20 ms | 内存带宽，非计算 |
| BM25 检索 | 5–15 ms | 倒排索引查找，IO 瓶颈 |
| RRF 融合 | <1 ms | 列表合并 |
| 重排序器（top-50 → top-5，Cross-encoder） | 80–200 ms | 50 次 Cross-encoder 前向，可批处理 |
| LLM 生成（4K 上下文，200 token 答案） | 1500–4000 ms | 解码自回归，绝对瓶颈 |

重排序器是团队常忘优化处。bge-reranker-base 在单 A10 GPU 上批处理 50 对约 150 ms；串行调用则需 2 秒。务必批处理。

LLM 调用比其他所有阶段慢一个数量级，意味着：**缩短答案长度优先于加速检索**。除非必要，`max_tokens` 限 300。最终合成步可用更小模型（Llama-3-8B 或 Qwen-2.5-7B 在上下文优质时足矣）。

成本上，GPT-4o-mini 输入 $0.15/M token，输出 $0.6/M token，4K 输入 + 200 输出时单次 **$0.0007**。日 100 万查询即 $700/天，$25 万/年。自托管 7B 模型于 4×A10 GPU 月成本约 $400（含折旧电费）——盈亏平衡点约 5 万查询/天。

---

## 12. 生产 RAG 真正的失败模式

调试过的五种失败模式及修复：

**失败 1：答案对但引错块**。LLM 幻觉合理来源。修复：强制模型在生成前逐字引用支持句（“先引原文句，再解释”）。引用幻觉降 ~70%。

**失败 2：检索返 FAQ 而非实际文档**。FAQ 短密，嵌入极近用户问，压过长详细源文档。修复：FAQ 单独索引，仅当无源文档超阈值时回退；或融合时降权 FAQ 块。

**失败 3：用户问“比较 X 和 Y”仅检 X 相关块**。查询稠密嵌入被提及多的实体主导。修复：查询分解——小 LLM 拆“比较 X 和 Y”为两次检索再合并。

**失败 4：多语言查询返单语结果**。中文问仅返中文文档，即使英文更相关。修复：用多语言嵌入器（bge-m3, multilingual-e5-large），BM25 前将查询译为语料主导语言。

**失败 5：重排序器降级正确答案**。Cross-encoder 重排序器训于 MS MARCO，风格特定。域外问（极长、对话式、含代码）打分差。修复：用 500–2000 域内 query-doc 对微调重排序器。成本低（单 GPU ~2 小时），几乎总值得。
