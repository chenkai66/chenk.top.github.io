---
title: "自然语言处理（十）：RAG与知识增强系统"
date: 2025-09-20 09:00:00
tags:
  - NLP
  - RAG
  - LLM
  - 向量数据库
categories: 自然语言处理
series:
  name: "自然语言处理"
  part: 10
  total: 12
lang: zh-CN
mathjax: true
description: "从第一性原理构建生产级 RAG 系统：retrieve-then-generate 概率分解、向量索引（FAISS / Milvus / Chroma / Weaviate / Pinecone）、稠密+稀疏混合检索与 RRF 融合、Cross-encoder 重排、切块策略、查询改写、HyDE，以及 Self-RAG / Corrective-RAG。"
disableNunjucks: true
series_order: 10
---

被冻结知识的语言模型是个自信的撒谎者：它读不到昨天的故障报告、读不到公司 wiki、读不到今早刚合入的 patch notes，于是当你提问，它会给出一个语法完美但事实错误的答案。**检索增强生成（RAG）** 通过把"记忆"和"推理"分开来打破这个困局：让 LLM 保持小且稳定，把易变的知识放到一个可以随时更新的外部存储里，生成之前先把相关证据检索出来作为上下文。

理念只有一段，工程是这篇文章的剩余部分。一个真实 RAG 系统大概有十几个旋钮——切块大小、Embedding 模型、索引类型、$k$、混合权重、重排深度、Prompt 模板、引用格式、拒答策略——而它们大多互相耦合。下文逐一讲清楚：数学、权衡、可运行的代码。

## 你将学到什么

- 概率分解 $P(y\mid q)=\sum_d P(d\mid q)P(y\mid q,d)$ 以及每一项的代价
- 为什么 ANN 索引（HNSW、IVF-PQ、ScaNN）用召回换延迟，以及何时值得这笔交易
- 稠密 vs 稀疏 vs 混合检索，以及为什么 **Reciprocal Rank Fusion (RRF)** 优于线性混合
- 两阶段 retrieve-then-rerank：Cross-encoder 带来的 +12 nDCG 来自哪里
- 尊重话题边界、而不是字节计数的切块策略
- 查询改写、问题分解、**HyDE**（假设性文档嵌入）
- **Self-RAG / Corrective RAG**：把检索变成模型自己掌控的决策
- 如何按规模、延迟和运维预算选择向量数据库

## 前置知识

- Embedding 与"含义"的余弦几何（[第 2 篇：词向量与语言模型](/zh/自然语言处理-二-词向量与语言模型/)）
- Decoder-only 语言模型与提示词工程（[第 6 篇：GPT](/zh/自然语言处理-六-GPT与生成式语言模型/)）
- 熟练使用 Python；隐约记得 TF-IDF 是干什么的

---

## 1. RAG 的概率分解

![RAG 端到端流水线，包含离线索引与在线查询两条路径](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%B3%BB%E7%BB%9F/fig1_rag_pipeline.png)

RAG 系统通过对一小撮被检索文档 $\mathcal{D}_k$ 做边缘化来回答查询 $q$：

$$
P(y \mid q) \;=\; \sum_{d \in \mathcal{D}_k}\; \underbrace{P(d \mid q)}_{\text{retriever}}\; \cdot \; \underbrace{P(y \mid q, d)}_{\text{generator}}
$$

实践中我们用两种方式近似这个求和。便宜且占主导的方式（在 LangChain 里叫 **stuff**）把 top-$k$ 文档拼到 LLM 上下文里，让 attention 隐式地完成边缘化。原则上更优但代价更高的方式（**Fusion-in-Decoder**，Atlas 与原版 RAG 论文用的是这种）对每个 $(q,d)$ 对单独编码，在 decoder 里做融合。对于绝大多数生产系统，stuff + $k\in[3,8]$ + 一个好的重排器就是正确答案。

这个分解之所以有吸引力：

- **检索器** $P(d\mid q)$ —— 更新便宜。几分钟就能重建一个文档集的索引；推理时不需要 GPU。
- **生成器** $P(y\mid q,d)$ —— 冻结。同一个 LLM 可以服务任意领域。
- **引用** —— 每条声明都能追到一个 chunk，可以展示给用户、可以被审计、可以用来过滤幻觉。

### 一个最小但诚实的流水线

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# 1. 加载与切块——overlap 不是可选项，见第 5 节
loader = DirectoryLoader("./docs", glob="**/*.md")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=64,
    separators=["\n\n", "\n", "。", ". ", " ", ""],
)
chunks = splitter.split_documents(loader.load())

# 2. 嵌入与建索引——bge-small 是 2024+ 的强默认
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectordb = FAISS.from_documents(chunks, embedder)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

# 3. 强制接地、允许拒答的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "只能依据上下文作答，并以 [i] 标注引用。"
     "若上下文不足以回答，回复『我不知道』。"),
    ("human", "上下文：\n{context}\n\n问题：{question}"),
])

def fmt(docs):
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = ({"context": retriever | fmt, "question": RunnablePassthrough()}
         | prompt | llm)

print(chain.invoke("上个季度 OAuth 流程做了哪些改动？").content)
```

三个细节挣得了它们的存在感。**分隔符层级**保证段落和句子不会被切断；**`temperature=0`** 消除一个方差源，让"检索质量"成为唯一的旋钮；**拒答条款**是你能拿到的最便宜的幻觉防线。

### 什么时候 *不要* 用 RAG

| 场景 | 更好的工具 |
|---|---|
| 风格适配（用我们品牌的语气写） | 微调 / DPO |
| 稳定且窄的任务（情感、NER） | 微调 |
| 模型已经做得很好的推理链 | 更好的 Prompt |
| 知识量小到能塞进 system prompt | Prompt 工程 |
| 表格 / 数值事实、精确计算 | Tool / function calling |

RAG 的高光场景是知识**量大、易变、可审计**：每周变化的文档、10 GB 的合规语料、任何"请展示来源"是需求一部分的场合。

---

## 2. Embedding 空间与 ANN 权衡

![Embedding 空间中的相似度检索，及 FAISS 各索引族的召回-延迟权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%B3%BB%E7%BB%9F/fig2_vector_similarity.png)

一个双塔编码器 $E_\theta:\text{text}\to\mathbb{R}^d$ 把查询和文档映射到同一空间，检索就是该空间里的余弦最近邻：

$$
\operatorname{sim}(q,d) \;=\; \frac{E_\theta(q)\cdot E_\theta(d)}{\lVert E_\theta(q)\rVert \, \lVert E_\theta(d)\rVert}.
$$

精确搜索每次查询代价 $O(N d)$ —— $10^5$ 还能扛，$10^7$ 已经痛苦，$10^9$ 完全不行。**近似最近邻（ANN）** 索引以放弃精确换数量级的延迟。生产中占主导的两族：

- **HNSW**（分层可导航小世界图）—— 一张多层近邻图。从顶层入口贪心下降，$O(\log N)$ 跳收敛。可调项：$M$（图度数，决定内存）、$\textit{efSearch}$（束宽，决定召回-延迟权衡）。绝大多数 RAG 工作负载的甜点。
- **IVF-PQ** —— 粗 $k$-means 聚类（IVF 桶）+ 残差的 Product Quantization。8× 到 32× 的内存压缩，代价是几个点的召回。索引装不进内存时用它。

图右侧展示了 1 M × 768-d 语料的 Pareto 前沿：HNSW 与 ScaNN 占据左上角，精确 `Flat` 在几十万规模仍可用，PQ-only 是内存极度紧张时的兜底。

### 2024+ 的 Embedding 模型怎么选

| 模型 | 维度 | 优势 | 备注 |
|---|---|---|---|
| `text-embedding-3-large`（OpenAI） | 3072（可截断） | 多语言强，Matryoshka 可截到 256 | 仅 API |
| `bge-large-zh-v1.5`（智源） | 1024 | 中文 MTEB 顶尖，权重开放 | 本地可跑 |
| `bge-m3` | 1024 | 单模型 dense + sparse + multi-vector，100+ 语言 | 多语言开源最佳 |
| `nomic-embed-text-v1.5` | 768 | 长上下文（8K tokens），Matryoshka | Apache 2.0 |
| `all-MiniLM-L6-v2` | 384 | 快 5–10×，MTEB 低约 3 个点 | 延迟敏感场景 |

**按瓶颈选**，不是按榜单。内存紧 → Matryoshka 截到 256 维；多语言 → `bge-m3` 难有对手；如果你有领域内的标注 query-passage 对，**针对你的语料微调一个 384-d 模型，常常打过 1024-d 的通用模型**——余弦几何会去贴合你的术语，冗余维度坍塌，召回随之上升。

---

## 3. 混合检索：稠密 + 稀疏 + RRF

![BM25 与稠密检索通过 RRF 融合的混合检索](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%B3%BB%E7%BB%9F/fig3_hybrid_retrieval.png)

稠密检索懂改述但在三类查询上输给 BM25：罕见命名实体（`CVE-2024-3094`）、精确标识符（`订单 #482915`）、短到不足以填满 Embedding 模型有效感受野的查询。稀疏检索正好相反：字面强、同义盲。

**BM25** 给查询 $q$ 与文档 $d$ 的打分：

$$
\operatorname{BM25}(q,d) \;=\; \sum_{t \in q} \operatorname{IDF}(t)\cdot
\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\!\left(1-b+b\,\frac{|d|}{\overline{|d|}}\right)}
$$

其中 $k_1\!\approx\!1.2$，$b\!\approx\!0.75$。IDF 项奖励稀有词命中，长度归一化项防止长文档自动占优。

```python
from rank_bm25 import BM25Okapi
import jieba

class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi([self._tok(d) for d in corpus])

    def _tok(self, text):
        return list(jieba.cut(text.lower()))   # 中文必须分词，不能 split()

    def search(self, query, k=20):
        scores = self.bm25.get_scores(self._tok(query))
        idx = scores.argsort()[::-1][:k]
        return [(self.corpus[i], float(scores[i])) for i in idx]
```

### 为什么 RRF 优于线性混合

朴素的 $\alpha\cdot s_\text{dense} + (1-\alpha)\cdot s_\text{sparse}$ 要求两种分数处在可比的尺度上，但它们偏偏不是 —— BM25 无界，余弦在 $[-1,1]$。**Reciprocal Rank Fusion**（Cormack 等，2009）干脆绕开"刻度对齐"问题，只用 *排名*：

$$
\operatorname{RRF}(d) \;=\; \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}, \qquad k \!=\! 60.
$$

常数 $k=60$ 抑制了任意单一检索器顶端结果的贡献，所以"在两个列表里都排第 1"的文档会胜出"只在一个列表里排第 1"的。图中右下面板展示了在异构查询集上的典型提升：BM25-only ≈ 54，dense-only ≈ 62，RRF ≈ 71，RRF + Cross-encoder rerank ≈ 78。

```python
def rrf_fuse(rankings, k=60, top_k=10):
    """rankings: 来自多个检索器的有序 doc_id 列表。"""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
```

混合检索很少是 RAG 系统的瓶颈，但它稳定地是你能找到的、最便宜的 +5 到 +10 召回点。dense 与 BM25 并行跑，RRF 融合，把 50 个候选送给重排器。

---

## 4. 重排：精度大头来自这里

![Bi-encoder vs Cross-encoder，及两阶段检索的质量-延迟曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%B3%BB%E7%BB%9F/fig4_reranking.png)

Bi-encoder 把查询和文档**分别**编码，所以文档向量可以离线预算并建索引——速度白得；代价是模型从未同时看到 query 与 document 的 token。**Cross-encoder** 把它们拼接 —— `[CLS] q [SEP] d [SEP]` —— 让整个 Transformer 跑过这一对，输出一个相关性 logit。联合 attention 能捕捉否定、问句改述、词级交互，这些都是 bi-encoder 的盲区。代价是每次查询要跑 $O(k)$ 次联合前向，所以 cross-encoder 只在小候选池上用得起。

**两阶段流水线**是标准答案：bi-encoder（或混合）拿到 top 50–100，cross-encoder 重排到最终 5。右图给出 MS MARCO dev 的典型曲线：单次重排 +12 nDCG，候选数翻倍再 +1.5，能负担一个 listwise LLM rerank 还能再 +2.5。

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

**实操要点。** 选一个训练分布贴近你领域的 reranker —— 多语言通用用 `bge-reranker-v2-m3`，英文 web 风格用 `ms-marco-MiniLM-L-12-v2`，有几千条领域标注就微调一个领域 reranker。`max_length` 要封顶（更长的 passage 没收益只让你慢）。GPU 上批处理。最重要的是**测一下** —— reranker 的收益与召回率耦合，所以在你的真实流量上跑 A/B，再为它的延迟成本辩护。

---

## 5. 切块：沉默的精度泄漏点

![固定 / 递归 / 语义切块，及切块大小与质量的权衡曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%B3%BB%E7%BB%9F/fig5_chunking.png)

查询的目标是"信息"，而我们的索引只认识"chunk"。三种主流策略：

- **固定大小**（$N$ tokens，无 overlap）—— 简单、确定，但在边界场景下**很糟**。能回答问题的信息常常横跨一个被切断的句子。
- **递归 + overlap** —— LangChain 的默认。按分隔符层级 `["\n\n", "\n", "。", ". ", " ", ""]` 尝试，若仍超长就递归到更细一级。64–128 token 的 overlap 让每个边界附近的句子都出现在两个 chunk 里，检索就不会因为切边漏掉它。
- **语义切块** —— 对每个句子求 Embedding，计算相邻句子的余弦距离，在距离尖峰处切。话题切换变成 chunk 边界，每块内部主题一致。建索慢，但在长文上质量提升明显。

右下面板总结了经验权衡：检索 Hit@5 与回答 faithfulness 都在 256–512 token 处达到峰值。更小 → 答案被切碎；更大 → chunk 向量被稀释（变成有噪的平均）。代码或结构化文档可以提到 768–1024，因为有意义的最小单元更大。

### 父子切块：嵌入小、返回大

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

检索器拿 400-token 的 child 嵌入做**精度**（小窗口的余弦更锐利），返回 2000-token 的 parent 给 LLM 做**上下文**（让模型看到能真正回答问题的相邻段落）。它是少数几乎零成本、却稳定提升检索与生成质量的 RAG 技巧之一。

---

## 6. 查询优化

用户敲的查询，几乎从来不是索引期待的查询。三个值得知道的技巧：

**查询改写。** 一次短的 LLM 调用把问题改写得更密、更适合检索。*"昨天清掉的那个 OAuth 啥来着"* → *"OAuth 2.0 access token 撤销事件，2024-04-23"*。便宜、快、对话式查询的召回明显提升。

**Multi-Query / 问题分解。** 复杂问题先生成 $n$ 个改写或子问题，每个分别检索，候选取并集再丢给重排器。RAG-Fusion 就是 multi-query + RRF。

```python
def multi_query(question, llm, n=4):
    prompt = (f"把下面的问题用 {n} 种不同方式改写以最大化文档召回。"
              f"每行一个。\n\nQ: {question}")
    return [q.strip() for q in llm.invoke(prompt).content.split("\n") if q.strip()]
```

**HyDE —— 假设性文档嵌入。** 让 LLM 先写一个"看起来合理的答案"，再用这个伪答案做嵌入和检索。反直觉但出奇有效：被幻觉出的答案与真实答案处在同一邻域，所以它的 embedding 比"问题"的 embedding 更接近相关文档。代价是多一次 LLM 调用；收益在"短而模糊的问题 × 技术语料"上最大。

```python
def hyde_retrieve(question, llm, vectordb, k=6):
    hypothetical = llm.invoke(f"用一段话回答：{question}").content
    return vectordb.similarity_search(hypothetical, k=k)
```

---

## 7. Self-RAG 与 Corrective RAG

![Self-RAG / Corrective RAG 控制流，含反思 token](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%B3%BB%E7%BB%9F/fig6_self_rag.png)

朴素 RAG **无条件**检索，既浪费（闲聊不需要检索）又危险（不相关的证据反而会带偏模型）。**Self-RAG**（Asai 等，2024）与 **Corrective RAG**（Yan 等，2024）把"检索与否"交还给模型决定，通过 reflection token 表达：

| Token | 决策 |
|---|---|
| `[Retrieve]` | 是否需要检索？ |
| `[ISREL]` | 每个被召回的 chunk 是否相关？（逐文档） |
| `[ISSUP]` | 生成的答案是否真的被证据支撑？ |
| `[ISUSE]` | 答案对用户是否有用？ |

控制流就是一个小图：发出 `[Retrieve]`，按相关性打分分支；若**没有任何**召回 chunk 相关，落入 corrective 分支 —— 改写查询、调用 web 搜索、重检索、重新打分。最终的 `[ISSUP]` / `[ISUSE]` 闸门强制要求"答案接地于保留下的证据"，而不是"答案与模型先验一致"。Self-RAG 在长文本 QA 基准上相对同规模无反思 RAG 报告了 +5 到 +9 点的提升。

不需要专门微调一个 Self-RAG 模型也能用这套模式 —— 同样的控制流可以用普通 LLM 的若干次结构化调用搭出来，代价是延迟：

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

    if not kept:                                                # corrective 分支
        rewritten = llm.invoke(f"改写为搜索查询：{query}").content
        kept = web_search(rewritten, k=5)

    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(kept, 1))
    answer = llm.invoke(
        f"只用上下文回答，并以 [i] 标注引用。\n\n{context}\n\nQ: {query}"
    ).content
    return answer, kept
```

代价是真实的 —— 比 vanilla RAG 多 3–4× LLM 调用 —— 在"答错的代价 > 答慢的代价"的场合是值得的。

---

## 8. 向量数据库的实践选型

![向量数据库能力雷达图与单节点吞吐对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/10-RAG%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%B3%BB%E7%BB%9F/fig7_vectordb.png)

**FAISS** 是库不是数据库。纯速度、无持久化、无过滤、无并发故事 —— 适合嵌入到另一个服务里、或者百万级离线实验。

**Chroma** 是上手最快的：pip install、落盘持久化、不打断你其他栈的工作。单节点上限大约 $10^5$–$10^6$ 向量。

**Milvus** 是开源重量级：分布式、多租户、支持 HNSW / IVF-PQ / DiskANN、标量过滤、原生混合（dense + sparse）检索。需要真正的运维（副本、滚动升级、可观测性）且规模在 $10^7$+ 时用它。

**Weaviate** 在同一规模上与 Milvus 竞争，主打一流的混合检索和清爽的 schema 模型。运维稍简单，原始吞吐稍低。

**Pinecone** 是托管选项：零运维、API 优秀、价格高。当工程时间是瓶颈、不是基础设施成本是瓶颈时，选它。

左侧雷达是主观汇总，右侧条形图是 recall ≥ 0.95 下的单节点 QPS 量级 —— 都只是起点，最终要在你自己的语料、自己的查询、自己的过滤选择性下测。

### 决策树

- **< 1 万向量、原型** → 内存版 FAISS 或 Chroma。
- **$10^5$–$10^6$、单节点、简单过滤** → Chroma 持久化 或 Weaviate。
- **$10^7$+、多租户、混合检索、私有化** → Milvus。
- **任意规模、没有运维人头、能接受云调用延迟** → Pinecone。
- **只关心静态语料的 top-$k$** → 落盘的 FAISS。

---

## 9. 评估一个 RAG 系统

RAG 有两种失败模式 —— 检索差和生成差 —— 必须分开测。

**检索指标**（需要 query→相关 chunk 的标注对）：

- **Hit@k** —— top-$k$ 里出现任意一个相关 chunk 即算命中。粗粒度但可执行。
- **MRR** —— 第一个相关 chunk 排名倒数的平均。对位置敏感。
- **nDCG@k** —— 折扣累计增益，归一化后。当相关性是分级的，用它。

**生成指标**（用 `gpt-4o` / `claude-sonnet` 当裁判）：

- **Faithfulness** —— 答案中的每条声明是否被检索到的上下文支撑？专门抓幻觉。
- **Answer Relevance** —— 答案是否真的回答了问题？
- **Context Relevance** —— 检索到的 chunk 是否真的有用？

标准工具是 **RAGAS** 与 **TruLens**，它们封装了上面这些 prompt，你不用自己再推一遍。每次改 Prompt、换 Embedding、调 chunker，都跑一组 50–200 query 的 golden set。没有这个回路，你根本判断不了昨天的"改进"是不是改进。

---

## 10. FAQ

**RAG 太慢，从哪查起？** 先 profile。Query 嵌入个位数 ms、HNSW 检索个位数 ms、P90 几乎一定卡在 LLM 调用。先缓存命中相同的查询、压缩 Prompt（父子切块在这里很有用），再考虑换更小的 reranker，最后才动索引。

**RAG 在幻觉。** 三层药方按顺序上：(a) 收紧 Prompt：要求引用、允许拒答；(b) 加一个 Cross-encoder reranker，确保上下文真的相关；(c) 升到 Self-RAG，加 `[ISSUP]` 闸门。三层都加完还在幻觉，说明你的 retriever 在召回垃圾 —— 回到切块和 Embedding 选择。

**要不要微调 Embedding 模型？** 有 ≥ 5K 标注 query-passage 对、且领域明确（法律、生物医学、内部黑话）—— 要。5–10 个点的召回提升会一路放大到下游。只有几十条样本——别动，`bge-large` 在通用中英文上已经够强。

**dense 还是 hybrid？** 永远先上 dense。哪天用户抱怨"明明文档里有，搜不出来"那个精确匹配查询时，加上 BM25 + RRF。边际工程成本是一个下午，召回收益是永久的。

**索引怎么保鲜？** 小语料 embed-on-write，其他场景定期批量重建。删除是个坑 —— 多数 ANN 索引只支持 tombstone，规划一个重建周期（多数团队每周一次足够），把它做成无聊的事。

---

## 系列导航

| 部分 | 主题 | 链接 |
|------|------|------|
| 9 | 大语言模型架构深度解析 | [← 上一篇](/zh/自然语言处理-九-大语言模型架构深度解析/) |
| **10** | **RAG 与知识增强系统（本文）** | |
| 11 | 多模态大模型 | [下一篇 →](/zh/自然语言处理-十一-多模态大模型/) |
