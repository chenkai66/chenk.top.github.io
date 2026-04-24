---
title: "NLP (10): RAG and Knowledge Enhancement Systems"
date: 2025-09-20 09:00:00
tags:
  - NLP
  - RAG
  - LLM
  - Vector Database
  - Embeddings
categories: Natural Language Processing
series: NLP
part: 10
total_parts: 12
lang: en
mathjax: true
description: "Build production-grade RAG systems from first principles: the retrieve-then-generate decomposition, vector indexes (FAISS / Milvus / Chroma / Weaviate / Pinecone), dense+sparse hybrid retrieval with RRF, cross-encoder reranking, chunking strategies, query rewriting, HyDE, and Self-RAG / Corrective-RAG."
disableNunjucks: true
series_order: 10
---

A frozen language model is a confident liar. It cannot read yesterday's incident report, your company wiki, or the patch notes that shipped this morning, so when you ask, it confabulates an answer that is grammatically perfect and factually wrong. **Retrieval-Augmented Generation (RAG)** breaks the deadlock by separating *memory* from *reasoning*: keep the LLM small and stable, and put the volatile knowledge in an external store that you can update at any time. Before generating, retrieve the relevant evidence and condition the model on it.

The idea is one paragraph. The engineering is the rest of this article. A real RAG system has roughly a dozen knobs — chunk size, embedding model, index type, $k$, hybrid weighting, reranker depth, prompt template, citation format, refusal policy — and most of them interact. We walk through each one with the math, the trade-offs, and code that runs.

## What you will learn

- The probabilistic decomposition $P(y\mid q)=\sum_d P(d\mid q)P(y\mid q,d)$ and what each term costs you
- Why ANN indexes (HNSW, IVF-PQ, ScaNN) trade recall for latency, and when to pay the price
- Dense vs sparse vs hybrid retrieval, and why **Reciprocal Rank Fusion** beats linear blending
- Two-stage retrieve-then-rerank with cross-encoders: where the +12 nDCG comes from
- Chunking that respects topic boundaries, not byte counts
- Query rewriting, decomposition, and **HyDE** (hypothetical document embeddings)
- **Self-RAG** and **Corrective RAG**: turning retrieval into a decision the model owns
- How to pick a vector database for your scale, latency, and ops budget

## Prerequisites

- Embeddings and the cosine geometry of meaning ([Part 2: Word Embeddings](/en/nlp-word-embeddings-lm/))
- Decoder-only LMs and prompting ([Part 6: GPT](/en/nlp-gpt-generative-models/))
- Comfort with Python and a vague memory of what TF-IDF did

---

## 1. The RAG decomposition

![End-to-end RAG pipeline showing offline indexing and online query path](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/fig1_rag_pipeline.png)

A RAG system answers a query $q$ by marginalising over a small set of retrieved documents $\mathcal{D}_k$:

$$
P(y \mid q) \;=\; \sum_{d \in \mathcal{D}_k}\; \underbrace{P(d \mid q)}_{\text{retriever}}\; \cdot \; \underbrace{P(y \mid q, d)}_{\text{generator}}
$$

In practice we approximate the sum two ways. The cheap and dominant approach (**stuff** in LangChain terms) concatenates the top-$k$ documents into the LLM context and lets attention do the marginalisation implicitly. The principled but expensive approach (**Fusion-in-Decoder**, used in Atlas and the original RAG paper) encodes each $(q,d)$ pair separately and fuses in the decoder. For most production systems, stuff with $k\in[3,8]$ and a good reranker is the right answer.

This factorisation is what makes RAG attractive:

- **Retriever** ($P(d\mid q)$) — cheap to update. Reindex a document set in minutes; no GPU needed at inference.
- **Generator** ($P(y\mid q,d)$) — frozen. The same LLM serves every domain.
- **Citations** — every claim traces back to a chunk. You can show it to a user, audit it, or use it to filter hallucinations.

### A minimal but honest pipeline

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# 1. Load and chunk — note the overlap, see section 5
loader = DirectoryLoader("./docs", glob="**/*.md")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(loader.load())

# 2. Embed and index — bge-small is a strong default in 2024+
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectordb = FAISS.from_documents(chunks, embedder)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

# 3. Prompt that forces grounding and refuses when unsure
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY from the context. Cite sources as [i]. "
     "If the context is insufficient, reply 'I don't know.'"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def fmt(docs):
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = ({"context": retriever | fmt, "question": RunnablePassthrough()}
         | prompt | llm)

print(chain.invoke("What changed in the OAuth flow last quarter?").content)
```

Three details earn their keep. The **separator hierarchy** in the splitter keeps paragraphs and sentences whole. The **`temperature=0`** removes one source of variance so retrieval becomes the only knob. The **refusal clause** in the system prompt is the cheapest hallucination guard you have.

### When *not* to use RAG

| Situation | Better tool |
|---|---|
| Stylistic adaptation (write in our brand voice) | Fine-tuning / DPO |
| Stable, narrow task (sentiment, NER) | Fine-tuning |
| Reasoning chains the model already does well | Better prompting |
| Knowledge that fits in the system prompt | Prompt engineering |
| Tabular / numeric facts, exact arithmetic | Tools / function calling |

RAG shines when the knowledge is **large, volatile, or auditable**: docs that change weekly, a 10 GB compliance corpus, anything where "show me the source" is part of the requirement.

---

## 2. Embedding space and ANN trade-offs

![Vector similarity in embedding space and recall vs latency for FAISS index families](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/fig2_vector_similarity.png)

A bi-encoder $E_\theta:\text{text}\to\mathbb{R}^d$ maps query and document into the same space. Retrieval is then a nearest-neighbour search under cosine similarity:

$$
\operatorname{sim}(q,d) \;=\; \frac{E_\theta(q)\cdot E_\theta(d)}{\lVert E_\theta(q)\rVert \, \lVert E_\theta(d)\rVert}.
$$

Exact search costs $O(N d)$ per query — fine at $10^5$, painful at $10^7$, impossible at $10^9$. Approximate Nearest Neighbour (ANN) indexes give up exactness for orders of magnitude in latency. The two families that dominate production:

- **HNSW** (Hierarchical Navigable Small World) — a multi-layer proximity graph. Greedy descent from a top-layer entry point converges in $O(\log N)$ hops. Tunables: $M$ (graph degree, memory) and $\textit{efSearch}$ (beam width, recall vs latency). Sweet spot for most RAG workloads.
- **IVF-PQ** — coarse $k$-means quantisation (IVF buckets) plus product quantisation of residuals. Massive memory compression (8× to 32×) at the cost of a few recall points. Use when the index does not fit in RAM.

The right panel of the figure shows the Pareto frontier on a 1 M × 768-d corpus: HNSW and ScaNN sit on the upper-left, exact `Flat` is fine if you only have a few hundred thousand vectors, and PQ-only is the fallback when memory is the binding constraint.

### Choosing an embedding model in 2024+

| Model | Dim | Strengths | Notes |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 (truncatable) | Strong multilingual, Matryoshka-truncatable to 256 | API only |
| `bge-large-en-v1.5` (BAAI) | 1024 | SOTA-class on MTEB, open weights | Run locally |
| `bge-m3` | 1024 | Dense + sparse + multi-vector in one model, 100+ languages | Best open multi-lingual |
| `nomic-embed-text-v1.5` | 768 | Long-context (8 K tokens), Matryoshka | Apache 2.0 |
| `all-MiniLM-L6-v2` | 384 | 5–10× faster, ~3 pt MTEB lower | Latency-critical |

**Pick by the binding constraint**, not the leaderboard. If RAM is tight, use Matryoshka truncation and store 256-d vectors. If your corpus is multilingual, `bge-m3` is hard to beat. If you have labelled in-domain pairs, fine-tuning a 384-d model often beats a 1024-d generic one — the cosine geometry adapts to your terminology, redundant terms collapse, and recall jumps.

---

## 3. Hybrid retrieval: dense + sparse with RRF

![Hybrid retrieval combining dense and BM25 via Reciprocal Rank Fusion](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/fig3_hybrid_retrieval.png)

Dense retrieval understands paraphrase but loses to BM25 on three classes of query: rare named entities (`CVE-2024-3094`), exact identifiers (`order #482915`), and queries shorter than the embedding model's effective receptive field. Sparse retrieval is the opposite: literal but blind to synonyms.

**BM25** scores a document $d$ for query $q$ as

$$
\operatorname{BM25}(q,d) \;=\; \sum_{t \in q} \operatorname{IDF}(t)\cdot
\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\!\left(1-b+b\,\frac{|d|}{\overline{|d|}}\right)}
$$

with $k_1\!\approx\!1.2$, $b\!\approx\!0.75$. The IDF term rewards rare matches; the length-normalisation term prevents long documents from dominating.

```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi([self._tok(d) for d in corpus])

    def _tok(self, text):
        # In production: tokenize properly (spaCy / language-specific)
        return text.lower().split()

    def search(self, query, k=20):
        scores = self.bm25.get_scores(self._tok(query))
        idx = scores.argsort()[::-1][:k]
        return [(self.corpus[i], float(scores[i])) for i in idx]
```

### Why RRF beats linear blending

The naïve approach $\alpha\cdot s_\text{dense} + (1-\alpha)\cdot s_\text{sparse}$ requires the two scores to be on commensurate scales, which they emphatically are not — BM25 scores are unbounded, cosine sits in $[-1,1]$. **Reciprocal Rank Fusion** (Cormack et al., 2009) sidesteps the calibration problem by working on *ranks*:

$$
\operatorname{RRF}(d) \;=\; \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}, \qquad k \!=\! 60.
$$

The constant $k=60$ damps the contribution of any single retriever's top result, so a document that lands at rank 1 in *both* lists outranks one that is rank 1 in only one. The bottom-right panel of the figure shows the typical lift on heterogeneous query mixes: BM25-only ≈ 54, dense-only ≈ 62, RRF ≈ 71, RRF + cross-encoder rerank ≈ 78.

```python
def rrf_fuse(rankings, k=60, top_k=10):
    """rankings: list of [doc_id, ...] in ranked order from each retriever."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
```

Hybrid retrieval is rarely the *bottleneck* in a RAG system, but it is reliably the cheapest +5 to +10 recall points you will find. Run dense and BM25 in parallel, fuse with RRF, send 50 candidates to the reranker.

---

## 4. Reranking: where most of the precision lives

![Bi-encoder vs cross-encoder, and the quality-latency curve of two-stage retrieval](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/fig4_reranking.png)

A bi-encoder embeds the query and the document **separately**, so the document vectors can be precomputed and indexed. Speed comes for free; the cost is that the model never sees query and document tokens together. A **cross-encoder** concatenates them — `[CLS] q [SEP] d [SEP]` — and runs the full Transformer over the pair, producing a single relevance logit. Joint attention catches negations, paraphrases of the question form, and term-level interactions that bi-encoders miss. The price is $O(k)$ joint passes per query, so cross-encoders are only viable on a small candidate pool.

The **two-stage pipeline** is the standard answer: bi-encoder (or hybrid) gets the top 50–100, cross-encoder reranks to the final 5. The right-hand chart shows the typical curve on MS MARCO dev: +12 nDCG from a single rerank pass, an extra +1.5 from doubling depth, and another +2.5 if you can afford a listwise LLM rerank.

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

**Practical notes.** Pick a reranker trained on data close to your domain — `bge-reranker-v2-m3` for multilingual general use, `ms-marco-MiniLM-L-12-v2` for English web-style queries, a fine-tuned domain reranker if you have a few thousand labelled pairs. Cap `max_length` (long passages do not help, they just slow you down). Batch the pairs on GPU. And **measure** — the reranker's gain compounds with retrieval recall, so run an A/B with and without it before defending its latency cost.

---

## 5. Chunking: the silent precision leak

![Fixed, recursive, and semantic chunking, with the quality-vs-size trade-off curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/fig5_chunking.png)

A query targets *information*, not *bytes*, but our indexes only know about chunks. Three strategies dominate:

- **Fixed-size** ($N$ tokens, no overlap) — trivial, deterministic, and *terrible* on boundary cases. The information that answers a question often spans a sentence boundary that the chunker just walked through.
- **Recursive** with overlap — the LangChain default. Try a separator hierarchy `["\n\n", "\n", ". ", " ", ""]`; if a chunk is still too big, recurse to the next level. The 64–128 token overlap is what makes it work: any sentence near a boundary appears in two chunks, so retrieval cannot whiff on it.
- **Semantic** — embed every sentence, take cosine distances between adjacent sentences, split at the spikes. Topical shifts become chunk boundaries, so each chunk is internally coherent. Slower to build, materially better on prose.

The bottom-right panel summarises the empirical trade-off: retrieval Hit@5 and answer faithfulness both peak in the 256–512 token range. Smaller chunks fragment the answer; larger chunks dilute the embedding (the cosine of the chunk vector becomes a noisy average). For code or structured docs, push to 768–1024 tokens because the meaningful units are larger.

### Parent–child chunking: index small, return large

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

The retriever embeds 400-token children for **precision** (small windows give sharp cosines) but returns the 2000-token parent to the LLM for **context** (the surrounding paragraphs the model needs to actually answer). It is one of the few RAG tricks that costs almost nothing and reliably improves both retrieval and generation quality.

---

## 6. Query optimisation

The query the user types is rarely the query the index expects. Three techniques worth knowing:

**Query rewriting.** A short LLM call rephrases the question into something denser and more retrieval-friendly. *"who cleared the OAuth thing yesterday"* → *"OAuth 2.0 access token revocation incident, 2024-04-23"*. Fast, cheap, materially better recall on conversational queries.

**Multi-query / decomposition.** For complex questions, generate $n$ rewordings or sub-questions, retrieve for each, and union the candidates before reranking. RAG-Fusion is just multi-query + RRF.

```python
def multi_query(question, llm, n=4):
    prompt = (f"Rewrite the question {n} different ways to maximise "
              f"document recall. One per line.\n\nQ: {question}")
    return [q.strip() for q in llm.invoke(prompt).content.split("\n") if q.strip()]
```

**HyDE — Hypothetical Document Embeddings.** Ask the LLM to write a *plausible answer* to the query, then embed and retrieve with that. Counter-intuitive but surprisingly effective: the hallucinated answer lives in the same neighbourhood as the real ones, so its embedding is closer to the relevant documents than the question's was. The cost is one extra LLM call; the win is largest on short, vague queries against a technical corpus.

```python
def hyde_retrieve(question, llm, vectordb, k=6):
    hypothetical = llm.invoke(f"Write a one-paragraph answer to: {question}").content
    return vectordb.similarity_search(hypothetical, k=k)
```

---

## 7. Self-RAG and Corrective RAG

![Self-RAG / Corrective RAG control flow with reflection tokens](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/fig6_self_rag.png)

Vanilla RAG retrieves *unconditionally*, which is wasteful (chit-chat needs no retrieval) and dangerous (irrelevant evidence misleads the model). **Self-RAG** (Asai et al., 2024) and **Corrective RAG** (Yan et al., 2024) make retrieval a decision the model owns, expressed via reflection tokens:

| Token | Decision |
|---|---|
| `[Retrieve]` | Should I retrieve at all? |
| `[ISREL]` | Is each retrieved chunk relevant? (per-doc) |
| `[ISSUP]` | Is the generated answer actually supported by the evidence? |
| `[ISUSE]` | Is the answer useful to the user? |

The control flow is a small graph: emit `[Retrieve]`, branch on relevance gradings, and if **no** retrieved chunk grades as relevant, fall through to a corrective action — rewrite the query, hit a web search, re-retrieve, and re-grade. The final `[ISSUP]` / `[ISUSE]` gate enforces that the answer is grounded in the kept evidence, not just consistent with the model's prior. Self-RAG reports +5 to +9 points across long-form QA benchmarks over a non-reflective RAG of the same size.

You do not need a fine-tuned Self-RAG model to use the pattern. The same control flow runs on a stock LLM with a few additional structured calls, at the cost of latency:

```python
def self_rag(query, vectordb, llm, web_search):
    if "no" in llm.invoke(
        f"Does this need external knowledge? yes/no: {query}"
    ).content.lower():
        return llm.invoke(query).content, []

    docs = vectordb.similarity_search(query, k=8)
    grades = [
        "yes" in llm.invoke(
            f"Is this passage relevant to the question?\n"
            f"Q: {query}\nP: {d.page_content[:600]}\nyes/no:"
        ).content.lower()
        for d in docs
    ]
    kept = [d for d, g in zip(docs, grades) if g]

    if not kept:                                              # Corrective branch
        rewritten = llm.invoke(f"Rewrite for web search: {query}").content
        kept = web_search(rewritten, k=5)

    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(kept, 1))
    answer = llm.invoke(
        f"Answer using ONLY the context, citing as [i].\n\n{context}\n\nQ: {query}"
    ).content
    return answer, kept
```

The cost is real — 3–4× the LLM calls of vanilla RAG — and it is worth it whenever wrong answers are more expensive than slow ones.

---

## 8. Vector databases in practice

![Capability radar and indicative single-node throughput across vector DBs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rag-knowledge-enhancement/fig7_vectordb.png)

**FAISS** is a library, not a database. Pure speed, no persistence, no filtering, no concurrency story — perfect for embedding it inside another service or for offline experiments at million scale.

**Chroma** is the easiest thing to start with: pip install, persist to disk, get on with the rest of your stack. Tops out at ~$10^5$–$10^6$ vectors per collection on a single node.

**Milvus** is the open-source heavyweight: distributed, multi-tenant, supports HNSW / IVF-PQ / DiskANN, scalar filtering, and hybrid (dense + sparse) search natively. Run it when you need real ops — replication, upgrades, observability — at $10^7$+ vectors.

**Weaviate** competes on the same scale with first-class hybrid search and a clean schema model. Slightly easier to operate than Milvus, slightly less raw throughput.

**Pinecone** is the managed option: zero ops, great APIs, premium price. Pick it when engineering time is the binding constraint, not infra cost.

The radar in the left panel is a subjective summary; the right panel is an order-of-magnitude single-node QPS at recall ≥ 0.95. Treat both as a starting point — measure on your own corpus, your own queries, your own filter selectivity.

### Decision tree

- **< 10 K vectors, prototype** → in-memory FAISS or Chroma.
- **$10^5$–$10^6$, single node, simple filters** → Chroma persistent or Weaviate.
- **$10^7$+, multi-tenant, hybrid search, on-prem** → Milvus.
- **Any scale, no ops headcount, latency budget for cloud calls** → Pinecone.
- **You only care about top-$k$ over a static corpus** → FAISS, persisted to disk.

---

## 9. Evaluating a RAG system

A RAG system has two failure modes — bad retrieval and bad generation — and you must measure them separately.

**Retrieval metrics** (need labelled query→relevant-chunk pairs):

- **Hit@k** — does any relevant chunk appear in the top-$k$? Coarse but actionable.
- **MRR** — average of $1/\operatorname{rank}$ of the first relevant chunk. Sensitive to position.
- **nDCG@k** — discounted cumulative gain, normalised. The right metric when relevance is graded.

**Generation metrics** (LLM-as-judge with `gpt-4o` or `claude-sonnet`):

- **Faithfulness** — is every claim in the answer supported by the retrieved context? Catches hallucinations.
- **Answer relevance** — does the answer address the question?
- **Context relevance** — were the retrieved chunks actually useful?

The standard tooling here is **RAGAS** and **TruLens**; both wrap the prompts so you do not have to re-derive them. Run a 50–200 query golden set on every prompt change, every embedding swap, every chunker tweak. Without this loop you cannot tell whether yesterday's "improvement" actually improved anything.

---

## 10. FAQ

**My RAG is slow. Where do I start?** Profile first. Embedding the query is single-digit ms; HNSW retrieval is single-digit ms; the 90th percentile is almost always in the LLM call. Cache identical queries, shrink your prompt (parent–child chunking helps here), and consider a smaller reranker before you touch the index.

**My RAG hallucinates.** Three layers, in order: (a) tighten the prompt to require citations and allow refusal, (b) add a cross-encoder reranker so the context is actually relevant, (c) move to Self-RAG with an `[ISSUP]` gate. If hallucination persists after all three, your retriever is bringing back garbage — go back to chunking and embedding choice.

**Should I fine-tune the embedding model?** Yes if you have ≥ 5 K labelled query-passage pairs and a clear domain (legal, biomedical, internal jargon). The 5–10 point recall lift compounds with everything downstream. No if you only have a handful of examples — `bge-large` is already excellent on general English.

**Dense or hybrid?** Always start with dense. Add BM25 + RRF the day a user complains about a missing exact-match query. The marginal eng cost is one afternoon; the recall win is permanent.

**How do I keep the index fresh?** Embed-on-write for small corpora, scheduled batch reindex for everything else. Deletes are the trap — most ANN indexes only support tombstones, so plan a rebuild cadence (weekly is fine for most teams) and make it boring.

---

## Series Navigation

- **Previous**: [Part 9 — Deep Dive into LLM Architecture](/en/nlp-llm-architecture-deep-dive/)
- **Next**: [Part 11 — Multimodal NLP](/en/nlp-multimodal-nlp/)
- [View all 12 parts in the NLP series](/tags/NLP/)
