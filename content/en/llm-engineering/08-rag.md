---
title: "LLM Engineering (8): Retrieval-Augmented Generation"
date: 2026-04-03 09:00:00
tags:
  - LLM
  - RAG
  - Embeddings
  - reranking
  - hybrid-retrieval
categories: LLM Engineering
series: llm-engineering
series_order: 8
series_title: "LLM Engineering"
lang: en
mathjax: true
disableNunjucks: true
description: "Chunking strategies, dense vs sparse vs hybrid retrieval, reranker selection, the long-context-vs-RAG tradeoff in 2026, and the failure modes that show up at 100K+ documents."
translationKey: "llm-engineering-8"
---

RAG is the most over-deployed and under-engineered pattern in LLM applications. The 2024 demo loop — embed everything with `text-embedding-3-large`, dump into pgvector, top-5 cosine — works for 1000 documents and a forgiving demo. It does not survive 100K real documents and a customer who notices when the answer is wrong. This chapter is what I wish more teams knew before they built their second generation of RAG.

The original RAG paper ([Lewis et al., 2020][lewis-rag]) framed retrieval-augmented generation as a hybrid model: a dense retriever (DPR) trained jointly with a generator (BART) so the retrieval objective optimized end-task accuracy. Production RAG in 2026 doesn't look much like Lewis's RAG — modern systems use frozen pre-trained embedders, separate rerankers, and decoder-only generators that don't train against the retriever. But the core insight (parameterize knowledge separately from reasoning) survived and became the dominant paradigm. The [Gao et al. (2023) RAG survey][gao-survey] is the best comprehensive overview of the post-2020 evolution into "Naive RAG → Advanced RAG → Modular RAG."

![LLM Engineering (8): Retrieval-Augmented Generation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/illustration_1.png)

## What RAG actually is

Retrieval-augmented generation: at query time, fetch relevant text from an external corpus, stuff it into the LLM's context, generate a grounded answer. The "augmented" part is the prompt template:

```yaml
You are an assistant. Answer the user's question using only the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{retrieved_chunks}

Question: {user_query}
```

The interesting engineering is on the left of "augmented" — building a retriever that finds the right chunks. Three sub-systems: chunking, embedding, and ranking.

## Chunking is the silent killer

![fig1: chunking strategies compared](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig1_chunking_strategies.png)

How you split documents into chunks determines what a retriever can possibly find. Common chunk sizes: 256, 512, 1024 tokens. Common strategies:

- **Fixed size**: split every $N$ tokens. Simple, breaks semantic units.
- **Sentence**: split by sentence boundaries. Better, often too small.
- **Recursive character**: split on `\n\n`, then `\n`, then `. `, etc. (LangChain's default). Good baseline.
- **Semantic**: embed sliding windows, split where embedding similarity drops. Better, more expensive.
- **Late chunking** ([Günther et al., 2024](https://arxiv.org/abs/2409.04701)): embed the whole document with a long-context embedder, then chunk the *embedding sequence* — each chunk's embedding incorporates context from the surrounding document. Best for long documents, requires a long-context embedder (Jina, BGE-M3).

The right answer depends on your corpus. Code: split by function/class. Legal: split by clause. Markdown: split by heading. PDFs: parse tables and figures separately, don't let them interrupt prose flow. Most failures I've debugged trace to "the document was split mid-table" or "the answer spans two chunks but neither chunk alone makes sense."

A chunk-size sanity check: pick 20 representative questions, manually find the answer in your corpus, count tokens. If most answers fit in a 512-token chunk, use 512. If most answers need 1500 tokens of context (e.g. legal contracts), use 1500 with 200-token overlap.

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

The overlap matters. Without overlap, a sentence that starts at position 799 in chunk A and ends at position 821 in chunk B is broken. With 100-token overlap, both chunks contain the full sentence.

### Late chunking: a 2024 breakthrough

Naive chunking embeds each chunk independently. The chunk for "Q3 revenue grew 12 %" carries no information about which company or which year. Late chunking flips the order: feed the entire document (up to 8K-32K tokens) to a long-context embedder, get per-token contextualized embeddings, then pool token embeddings within chunk boundaries to get final chunk embeddings. The chunk embedding for "Q3 revenue grew 12 %" now reflects the surrounding "Apple's Q1 2024 earnings report" context.

Reported gains on long-document QA: 5-15 % NDCG improvement vs naive chunking with the same chunk boundaries, no extra storage cost, slight increase in indexing latency. Jina's `jina-embeddings-v3` and BGE-M3 both support late chunking natively.

When does late chunking help? Whenever your documents are long enough that local context inside a single chunk loses important framing — research papers, legal contracts, code repositories, multi-section technical documentation. For short-document corpora (FAQ entries, product descriptions), the benefit is smaller.

## Embedding model choice

![fig2: dense vs sparse vs hybrid retrieval](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig2_dense_sparse_hybrid.png)

Embeddings turn text into vectors so similarity search can find semantically related chunks. The model you pick determines quality, cost, and latency.

State of the art in 2026:

| Model | Dim | MTEB avg | Notes |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 | ~64.6 | Closed, $0.13/Mtok |
| `Cohere embed-v4` | 1024 | ~67.1 | Multilingual strong |
| `BGE-M3` | 1024 | ~69.4 | Open, multilingual, dense+sparse+colbert |
| `Qwen3-Embedding-8B` | 4096 | ~74.0 | Open, top of MTEB |
| `voyage-3-large` | 1024 | ~70.5 | Closed, premium |
| `jina-embeddings-v3` | 1024 | ~67.0 | Open, multilingual, late-chunking native |

Two practical points:

**Multilingual matters.** Most of the 2022-2023 generation embedders were English-trained. They retrieve poorly on Chinese, Japanese, Arabic. If your corpus is multilingual, pick a multilingual model — BGE-M3 and Qwen3-Embedding are the open leaders.

**Domain-specific often beats general.** A legal-domain fine-tune of a small embedder often beats the SOTA general embedder on a legal corpus. If your corpus has a strong domain (medical, legal, code, scientific), evaluate domain-specific embedders or fine-tune your own.

Self-hosting an embedder is increasingly cheap. BGE-M3 on a single L4 GPU does ~3000 chunks/sec. For most corpora <10M chunks, you don't need a managed service.

A note on **dimensionality**: higher-dim embeddings (3K-4K) generally retrieve better but cost more storage and slower index lookups. The Matryoshka representation learning approach (used in `text-embedding-3-large` and Nomic embeddings) trains the embedding so prefixes of various lengths (256, 512, 1024) all retrieve well, letting you trade dim for performance at deployment time without retraining. For 100K-doc corpora, 768-1024 dim is the practical sweet spot.

## Vector indexing: HNSW, IVF, and the tradeoffs

Once you have embeddings, you need a vector index to find nearest neighbors faster than brute force. Two algorithm families dominate:

**HNSW** (Hierarchical Navigable Small World, [Malkov & Yashunin, 2018][malkov-hnsw]): build a multi-layer graph where each node is connected to roughly $M$ nearest neighbors. Queries start at a top-layer entry point and greedily descend through layers, refining the candidate set at each step. Recall is tunable via `ef_search` (more candidates = higher recall, slower query). Index build is slow (O(N log N) graph constructions) and memory-heavy (~1.5x the embedding storage), but query is sub-millisecond at million-scale.

HNSW is the default in pgvector, Milvus, Qdrant, Weaviate, and most modern vector databases. Recommended parameters for 1M-100M vector corpora: `M=16-32`, `ef_construction=200-400`, `ef_search=50-100`. These give >95 % recall at <5 ms query latency on commodity hardware.

**IVF** (Inverted File Index): cluster the embeddings into $K$ centroids using k-means; at query time, find the closest few centroids and search only their inverted lists. Good for memory efficiency (no graph overhead) but worse recall-vs-speed than HNSW. Used in FAISS, where it's the foundation of more sophisticated variants like IVF-PQ (product quantization on top of IVF for additional compression).

**FAISS** ([Johnson et al., 2017](https://arxiv.org/abs/1702.08734)) is the workhorse implementation library. It exposes raw indexing primitives (Flat, IVF, HNSW, IVF-PQ, IVF-SQ) and lets you compose them. Production teams typically use FAISS for offline batch retrieval (re-ranking experiments, training data construction) and a higher-level vector DB (Qdrant, pgvector) for online serving.

For 100K-1M vector deployments, the choice barely matters — HNSW in pgvector is fine. At 100M+, the choice becomes about memory budget and shard count. At 10B+, you're in custom-infrastructure territory and Google/Meta-grade engineering.

## Dense vs sparse vs hybrid

![fig3: RRF fusion of rankings](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig3_rrf_fusion.png)

**Dense retrieval** (cosine on neural embeddings): great at semantic similarity, weak at exact-match (acronyms, IDs, rare terms).

**Sparse retrieval** (BM25 or its modern variants like SPLADE): great at exact-match and rare terms, weak at synonym/paraphrase. **BM25** ([Robertson et al., 1995][robertson-bm25]) is a probabilistic relevance model that scores documents by term frequency × inverse document frequency with length normalization. The formula:
$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot |d|/\text{avgdl})}$$
with $k_1 \approx 1.5$ and $b \approx 0.75$ as standard parameters. BM25 has been the dominant lexical retrieval algorithm for 30 years; modern dense embedders did not displace it because it remains best-in-class for queries that hinge on specific tokens (product SKUs, error codes, named entities).

**Hybrid retrieval** combines both, then merges. Almost every production RAG system in 2026 is hybrid — the win over pure dense is large (10-30 % NDCG@10 on most benchmarks), the cost is small (BM25 is cheap, you already have the chunks).

The merge formula matters. Simple weighted sum (`score = 0.5 * dense + 0.5 * bm25`) requires score normalization and is finicky. The dominant pattern is **Reciprocal Rank Fusion** (RRF, [Cormack et al., 2009][cormack-rrf]):

```python
def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """rankings: list of ranked doc-id lists from different retrievers."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

RRF is parameter-free in the score sense; the constant $k=60$ is empirically robust. Run dense and sparse retrieval independently, take top-50 from each, RRF-merge, take top-20. Then rerank.

Why RRF works: it operates on ranks rather than scores, so it doesn't care that BM25 scores are unbounded positive numbers and cosine scores are in [-1, 1]. Rank-1 from any retriever contributes the same to the final score, regardless of how the underlying retriever scaled its confidence. This is the right invariant for combining retrievers with very different score distributions.

## Reranking is the unsung hero

![LLM Engineering (8): Retrieval-Augmented Generation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/illustration_2.png)


![fig4: reranker pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig4_reranker_pipeline.png)

Embedding retrieval is fast but imprecise (it's a lossy compression). A second-stage **cross-encoder reranker** scores each candidate against the query directly: jointly encode `[query, candidate]`, output a relevance score.

Cross-encoders are 100x more expensive per pair than embedding cosine, so you can't use them for retrieval over millions of chunks. But for the top-20 candidates from first-stage retrieval, they're affordable and they consistently improve quality 5-15 %.

Models I'd use:

- `BAAI/bge-reranker-v2-m3` (open, multilingual, ~600M params, fast on L4)
- `cohere-rerank-v3` (closed, English-strong, $2/1K queries)
- `jina-reranker-v2` (open, multilingual, very fast)

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query, candidates, top_k=5):
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
```

The pipeline becomes: embed/sparse retrieve top-50 → rerank to top-5 → stuff into LLM context. The reranker is the place to spend latency budget; it consistently moves the quality needle more than tweaking the LLM prompt.

## Late interaction: ColBERT

![fig5: ColBERT late interaction](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig5_colbert_late_interaction.png)

Between the speed of bi-encoders and the quality of cross-encoders sits **late interaction** (ColBERT, [Khattab & Zaharia, 2020][khattab-colbert]). The query and document are independently encoded into per-token vectors (not pooled). Similarity is computed token-by-token:
$$\text{score}(q, d) = \sum_{i} \max_j \langle q_i, d_j \rangle$$
ColBERT preserves token-level matching (good for rare terms) while staying parallelizable. ColBERTv2 (Santhanam et al., 2022) and PLAID (Santhanam et al., 2022) make it feasible at million-document scale through residual compression and approximate retrieval. BGE-M3 includes a ColBERT-style component you can use for free.

Late interaction is appearing in 2025-2026 production stacks for high-precision retrieval where reranking 50 candidates isn't enough but full cross-encoding all candidates is too expensive. The 2024 update **ColPali** extended ColBERT's late-interaction principle to vision-language models for document image retrieval, finding that token-level matching dramatically improves PDF/scan retrieval over OCR-then-embed pipelines.

## Anthropic's Contextual Retrieval and GraphRAG

Two 2024 advances pushed RAG quality further on production-scale corpora.

**Anthropic Contextual Retrieval** (2024): before embedding each chunk, prepend a model-generated 50-100 token context that explains where the chunk fits in the document ("This chunk is from the Q3 2023 earnings report of Acme Corp, in the section discussing supply chain disruptions"). The contextualized chunks are then embedded with both a dense embedder and BM25, hybrid retrieved with RRF, and reranked. Anthropic reported a 49 % reduction in retrieval failures on their internal benchmarks, rising to 67 % when combined with reranking. The trick is that the context generation is a one-time cost during indexing (not per-query) and dramatically improves the embedding's fitness for retrieval.

**GraphRAG** ([Microsoft Research, 2024][graphrag]): instead of treating documents as flat chunks, extract entities and relationships into a knowledge graph during indexing. Cluster the graph into communities at multiple granularities. At query time, retrieve community summaries (rather than raw chunks) for queries that span multiple documents. GraphRAG outperforms naive RAG on "global" questions like "What are the main themes in this corpus?" that no individual chunk can answer. The cost is higher: graph extraction requires LLM calls during indexing (~$1-10 per 1000 documents). For corpora where queries naturally aggregate across documents, GraphRAG is the current frontier; for fact-retrieval queries, hybrid + rerank still wins on cost-quality.

## When to use long context instead

![fig6: long context vs RAG trade-off](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/08-rag/fig6_context_vs_rag.png)

Chapter 6 introduced this; here's the production decision matrix:

| Property | RAG wins | Long context wins |
|---|---|---|
| Corpus size | Big (>1M tokens) | Small (<200K tokens) |
| Query latency budget | <2s | 5-30s ok |
| Cost per query | $0.001-0.05 | $0.10-2.00 |
| Source attribution | Important | Not needed |
| Question is local | Yes (find one fact) | No (synthesize across whole doc) |
| Updates frequent | Yes | No (re-prefill on every change) |

The fashionable claim from 2024 — "long context kills RAG" — turned out to be wrong on cost grounds alone. A 100K-token prompt at Claude-4.5-Sonnet pricing ($3/Mtok input) is $0.30 per query before generation. RAG is typically $0.001-0.01 per query including embedding and LLM cost. For a product serving 1M queries/day, that's $300/day vs $3000-10000/day in long context.

The right answer is often **both**: RAG to find candidate chunks, then long-context the candidate set + a synthesis prompt. This is what most "2026 production RAG" systems actually look like.

## Failure modes at 100K+ documents

**Embedding drift.** Your corpus changed format three months in (new template, different vocabulary). The new docs aren't retrieved well because their vector neighborhoods are different. Defense: re-evaluate retrieval quality monthly on a held-out test set. Re-embed the corpus when MTEB-style scores drift >5 %.

**Chunk boundary cutting.** A single coherent answer is split across two chunks; neither chunk alone is informative. Symptoms: low retrieval recall on questions whose answer is unambiguously in the corpus. Defense: use overlap (100-200 tokens), structured chunking that respects document boundaries (sections, paragraphs), and validation by sampling chunks for "does this chunk make sense standalone?" Late chunking partially addresses this by having each chunk's embedding carry contextual information.

**Distribution shift in queries.** Users start asking questions you didn't anticipate. The reranker, trained on different query types, underperforms. Defense: log queries, sample weekly, hand-label, retrain reranker quarterly.

**Hot chunks dominating.** A few chunks (e.g., the FAQ on returns policy) get retrieved for every other query, regardless of relevance. They're embedded near the cluster centroid. Defense: penalize over-retrieved chunks (BM25 IDF naturally does this; dense retrieval needs explicit diversity terms like MMR).

**Chunks containing only metadata.** A chunk that's just "Section 4.2.1" matches a lot of queries semantically but contains no answer. Defense: filter chunks below a minimum content density (e.g., reject chunks with <100 alphanumeric chars or <5 unique non-stopwords).

**Duplicate chunks from ingestion bugs.** I once had a corpus where 30 % of chunks were near-duplicates because the ingestion script ran twice on a subset. Top-K retrieval returned 5 versions of the same chunk; LLM thought it had 5 sources confirming a fact and confidently asserted it. Defense: dedup at ingestion (MinHash, SimHash, or simple normalize+hash).

**Embedding stale-vs-document update lag.** A document is updated in the source-of-truth database; the embedding hasn't been re-computed. Retrieval returns the old version's chunks. Defense: track a `content_hash` per document, re-index any chunk whose hash changed, monitor index-vs-source freshness as a SLO.

**Reranker overconfidence.** Cross-encoder rerankers are trained on pairs labeled "relevant" or "irrelevant" — they output a score but it's not a calibrated probability. A reranker confident at score 0.9 is not necessarily more accurate than at 0.6 in cross-domain situations. Defense: don't threshold on raw reranker scores in absolute terms; use relative ranks within a query.

## Eval: nothing else matters if you don't measure

I have seen too many teams ship RAG without an eval set. Then they iterate on prompts and chunk sizes for weeks with no signal that anything is improving. **Build an eval set first**, even a 50-question one. Manually find the gold-standard chunks and gold-standard answers. Then track:

- **Retrieval recall@k**: did the retriever return the chunk(s) containing the answer?
- **Reranking precision@k**: of the top-k after reranking, what fraction contain the answer?
- **Answer faithfulness**: does the answer follow from the retrieved chunks (not the LLM's prior)?
- **Answer correctness**: matches the gold answer?

LLM-as-judge for the last two works fine; chapter 10 details that. The first two are pure retrieval metrics, easy to compute, and the most actionable for debugging.

For larger eval sets (>500 questions), tools like **RAGAS** (RAG Assessment, open-source 2023) and **TruLens** automate the four-metric pipeline with LLM-as-judge defaults. They're a reasonable starting point but the metrics they compute can be gamed by tuning prompts to satisfy the judge — gold-standard human-labeled subsets remain the only fully trustworthy signal.

## Production architecture recommendations

For a typical 100K-1M document RAG deployment in 2026:

- **Storage**: Postgres with pgvector for vectors and metadata. Add `pg_trgm` or Elasticsearch for BM25. Single-region write, read replicas for query.
- **Embedder**: BGE-M3 self-hosted on an L4 GPU for indexing (one-time-ish), Cohere or OpenAI API for query embedding (low latency, no infra).
- **Retrieval**: hybrid dense + BM25, RRF merge, top-50 candidates.
- **Reranker**: BGE-reranker-v2-m3 self-hosted on the same L4 (or Cohere Rerank API for simplicity), top-50 → top-5.
- **Generator**: Claude-4.5-Sonnet or Qwen3-Max for general; self-hosted Qwen3-32B for cost-sensitive.
- **Eval**: 100-200 hand-labeled questions, run on every deployment, alert if recall@10 or faithfulness drops >5 %.
- **Monitoring**: log every query, retrieval results, generation, latency, cost. Sample 1 % for human review.

This pipeline serves <1s p95 latency at <$0.01 per query for most workloads, scales to ~10M documents on a single Postgres instance, and degrades gracefully under load.

## What's Next

Chunking is the most under-appreciated knob; pick chunk sizes that match where your answers actually live. Use a multilingual embedder if your corpus is multilingual; use a domain embedder if you have one. Always go hybrid (dense + sparse + RRF) and always rerank. Long context wins on small corpora and synthesis tasks; RAG wins on cost and freshness for everything else. Build an eval set before you build a second iteration. The 2024 advances (Contextual Retrieval, GraphRAG, late chunking) move the frontier; the basics still apply.

Next chapter: **prompting at production scale**. Chain-of-thought, self-consistency, prompt caching economics, and the jailbreak/injection threat model.

## References

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
