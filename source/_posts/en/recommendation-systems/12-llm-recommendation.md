---
title: "Recommendation Systems (12): Large Language Models and Recommendation"
date: 2024-05-13 09:00:00
tags:
  - Recommendation Systems
  - LLM
  - Large Language Models
categories: Recommendation Systems
series:
  name: "Recommendation Systems"
  part: 12
  total: 16
lang: en
mathjax: true
description: "How LLMs reshape recommendation: enhancers (P5, M6Rec), predictors (TallRec, GenRec), and agents (LlamaRec, ChatREC). Hybrid pipelines, cold-start wins, prompt design, and the cost/quality Pareto frontier."
disableNunjucks: true
---
A user opens a movie app and types: *"Something like Inception, but less depressing."* A traditional recommender — collaborative filtering, two-tower DNN, even DIN — sees zero useful tokens here. It has no `like` button to count, no co-watch graph to traverse, no user ID with history. The query has to be turned into IDs before the system can do anything.

A Large Language Model has the opposite problem: it has *too much* world knowledge but doesn't know who this user is. It knows Inception is a Christopher Nolan film with non-linear narrative and a hopeful-but-ambiguous ending; it knows what "depressing" means in cinema; it can name twenty films that fit. But it can't tell you which of those twenty the *current* user has already seen, rated badly, or left half-watched.

The interesting question for 2023–2026 is not "LLM or traditional?" — it's **how to compose them**. This article walks through the three composition patterns that have actually shipped at scale, with code and the hard tradeoffs.

## What You Will Learn

- The **three roles** an LLM can play in a recommender: enhancer, predictor, agent — and which papers ship which
- **Prompt anatomy** for ranking tasks (TallRec / P5 style) and why low temperature matters
- How **LLM-enriched item descriptions** sharpen embedding clusters and fix cold-start
- The **hybrid pipeline** every production team converges to: ANN → DNN ranker → LLM reranker
- **ChatREC-style conversational** flow with multi-turn state and tool use
- The **cost/quality Pareto frontier**: when fine-tuned 7B beats GPT-4, and where the sweet spot lives
- Working **Python** for prompt-based ranking, hybrid reranking, and conversational state

## Prerequisites

- Embeddings, ranking, and the retrieval/ranking split ([Part 1](/en/recommendation-systems-1-fundamentals/), [Part 4](/en/recommendation-systems-4-ctr-prediction/))
- Familiarity with an LLM SDK (OpenAI, Anthropic, vLLM); no fine-tuning required
- Basic PyTorch for the optional encoder examples

---

## The Three Roles of LLMs in Recommendation

![Three roles of LLMs in recommendation: enhancer, predictor, agent](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/12-llm-recommendation/fig1_three_roles.png)

The literature looks chaotic — P5, M6-Rec, KAR, TallRec, BIGRec, GenRec, LlamaRec, ChatREC — but every one of them slots into exactly one of three roles. The role determines latency, cost, training cost, and what the LLM is even doing in the loop.

### Role 1 — Enhancer (offline feature generation)

The LLM never touches a live request. Offline, it reads each item's title, description, and reviews and emits richer features: structured attributes, dense semantic vectors, augmented descriptions, or pseudo-labels. Those features are then fed into a perfectly traditional CTR / CF stack.

This is the role from the earliest serious LLM-in-RecSys papers: **P5** (Geng et al., RecSys 2022) unified five recommendation tasks under a single text-to-text T5; **M6-Rec** (Cui et al., 2022) used Alibaba's M6 to generate item descriptions and behavioral prompts; **KAR** (Xi et al., 2024) distills LLM reasoning into reusable feature vectors. The serving stack stays sub-50ms because the LLM cost is amortized across millions of requests.

**Use it when**: You already have a working ranker but cold-start items, sparse text, or hard-to-tag categories are hurting recall.

### Role 2 — Predictor (direct ranking or generation)

The LLM is the ranker. You hand it the user's history and a candidate list (or no candidates at all) and it outputs a yes/no judgment, a score, or the next item directly.

Two flavors: **discriminative** (TallRec, BIGRec — fine-tune Llama-2-7B with LoRA on `(history, item) → yes/no`) and **generative** (GenRec — generate the next item title). **TallRec** (Bao et al., RecSys 2023) showed that a LoRA-tuned 7B beats much larger zero-shot LLMs and matches strong sequential baselines with only a few hundred training examples. **LlamaRec** (Yue et al., 2023) added a verbalizer head so the LLM ranks an entire candidate set in one forward pass instead of K separate calls.

**Use it when**: You have rich textual content per item, your data is small (cold-start domain, new vertical), or you need explanations bundled with rankings.

### Role 3 — Agent (orchestrator with tools)

The LLM doesn't just rank — it plans. It maintains conversation state, decides when to call retrieval, when to ask a clarifying question, when to filter by year or price, and when to hand off to a downstream model. **ChatREC** (Gao et al., 2023) is the canonical pattern; **RecAgent** (Wang et al., 2023) and **InteRecAgent** push this further with simulated users for evaluation.

**Use it when**: You're building a conversational surface (chatbot, voice, search-with-followup), or your users genuinely cannot articulate what they want in one shot.

The cost ranking is roughly: enhancer ≪ predictor ≪ agent. So is the flexibility ranking. Most production systems start as enhancers, graduate to predictors as a reranking layer, and only adopt agents for specific surfaces.

---

## Prompt-Based Ranking

![Prompt template anatomy for LLM-as-recommender](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/12-llm-recommendation/fig2_prompt_template.png)

A prompt-based recommender is the entry point: no training, just an API key and a well-shaped prompt. Five sections matter, in this order.

```python
def build_recommendation_prompt(history, target, examples=None):
    """TallRec / P5-style ranking prompt."""
    # 1. Role + task: pin the LLM into a narrow behavior
    role = (
        "You are a movie recommender. Given the user's viewing history, "
        "predict whether they will like the target movie."
    )

    # 2. User history (chronological, with signal)
    hist_lines = "\n".join(
        f"  - {item['title']} ({'liked' if item['liked'] else 'disliked'})"
        for item in history[-20:]   # truncate to fit context
    )

    # 3. Candidate / target item with attributes
    target_line = (
        f'Target movie: "{target["title"]}" '
        f'({target["genre"]}, {target["year"]}, dir. {target["director"]})'
    )

    # 4. Output format (constrained — single token if possible)
    output_spec = 'Answer with exactly one token: "Yes" or "No".'

    # 5. Few-shot examples (optional; biggest accuracy lever after format)
    fewshot = ""
    if examples:
        fewshot = "\n".join(
            f"[Example {i+1}]\nHistory: {ex['history']}\n"
            f"Target: {ex['target']}\nAnswer: {ex['answer']}\n"
            for i, ex in enumerate(examples)
        )

    return f"""{role}

{fewshot}
History:
{hist_lines}

{target_line}

{output_spec}"""
```

**Three implementation rules** that sound trivial and are not:

1. **Constrain the output token-by-token.** A single `Yes`/`No` is parseable; "I think the user might enjoy this because..." is not. For ranking K items, ask for `[3, 1, 5, 2, 4]` — never free-form prose. Use logit bias or grammar-constrained decoding when the SDK supports it.
2. **Keep temperature ≤ 0.3.** Recommendation is not creative writing. High temperature yields different rankings on identical inputs — a debugging nightmare.
3. **Truncate history.** LLMs degrade past ~50 items and cost scales linearly with prompt tokens. Newest items first; summarize older ones if needed.

### Zero-shot vs few-shot vs fine-tuned

| Approach | Setup cost | Per-call cost | Accuracy on cold-start | Accuracy on warm-start |
|---|---|---|---|---|
| Zero-shot (GPT-4o) | Hours | High ($0.01+/call) | Strong | Mediocre |
| Few-shot (3–5 examples) | Hours | Higher (more tokens) | Strong | Better |
| Fine-tuned (TallRec, LoRA on 7B) | Days + GPUs | Low ($0.0005/call) | Strong | Strong |

The TallRec result is striking: with **256 fine-tuning examples**, a LoRA-tuned Llama-7B matches or beats zero-shot GPT-4 at a fraction of the per-call cost. If you have any labeled interaction data, fine-tune.

---

## LLM as Enhancer: Sharper Embeddings, Better Cold-Start

![Item embeddings before and after LLM description enhancement](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/12-llm-recommendation/fig3_embedding_quality.png)

The cheapest and lowest-risk way to put an LLM into a production stack: use it offline to expand thin item text into rich descriptions, then re-encode with your existing sentence encoder.

A movie title `"Tenet"` carries almost no signal for a sentence encoder. Asking GPT-4 to expand it into:

> "A 2020 sci-fi action thriller directed by Christopher Nolan. Features inverted entropy as a plot device, exploring themes of free will and predestination. Tone is cerebral and atmospheric, with elaborate practical action sequences. Audience: viewers who liked Inception, Interstellar, or other puzzle-box narratives. Pace is brisk; emotional core is friendship and sacrifice."

…produces an embedding that lands much closer to its true semantic neighbors. In offline tests on MovieLens-style data, silhouette coefficients on genre clusters typically jump from ~0.2 (raw titles) to ~0.65 (LLM-enhanced) — the same effect P5 and KAR report.

```python
from openai import OpenAI

client = OpenAI()


def enhance_item(item):
    """Offline: expand sparse item text into rich semantic description."""
    prompt = f"""Write an 80-word description of this item for a
recommendation system. Cover: themes, tone, mood, target audience,
similar items, what kind of user would love it.

Title: {item['title']}
Existing tags: {', '.join(item.get('tags', []))}
Year: {item.get('year', 'unknown')}

Description:"""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=200,
    ).choices[0].message.content


def build_enhanced_index(items, encoder):
    """Run once per item; cache forever (until item changes)."""
    enhanced_texts = [enhance_item(it) for it in items]
    return encoder.encode(enhanced_texts, batch_size=64)
```

The economics work because **enhancement is offline and one-shot per item**. A million-item catalog at $0.0002 per call is $200 total, paid once. Compare to using an LLM on every live request at the same cost: $200 per million queries, *recurring*.

### Where the offline-only constraint hurts

LLM enhancement only generates *item-side* features that are stable. It cannot:

- React to a *new* item's first hour of interactions
- Capture user-specific signal (taste, mood right now, current session intent)
- Reflect breaking trends ("everyone is suddenly watching X because of Y")

For those you need a live LLM call — or, more realistically, a hybrid pipeline.

---

## The Hybrid Pipeline (the architecture you actually ship)

![Hybrid pipeline: ANN retrieval, DNN ranker, LLM reranker](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/12-llm-recommendation/fig4_hybrid_pipeline.png)

Every production team that ships LLM-powered recommendations converges on the same shape:

1. **Catalog** (10⁶–10⁸ items)
2. **ANN retrieval** with Faiss/HNSW — ~10ms, ~$0 — narrows to ~1,000 candidates
3. **CF or deep ranker** (DIN, DCN-V2, two-tower) — ~30ms, ~$0 — narrows to ~50
4. **LLM reranker** (GPT-4o or fine-tuned 7B) — ~1–2s, ~$0.005–0.01 — picks top 10
5. **User** sees top 10 + LLM-generated explanation

The funnel ratio is critical: **5–7 orders of magnitude** of pruning happen *before* the expensive LLM call. The LLM only ever sees ~50 items, never millions. This single architectural choice is what makes LLM recommendation viable.

```python
class HybridRecommender:
    """ANN -> DNN -> LLM. The pipeline every team converges on."""

    def __init__(self, ann_index, dnn_ranker, llm_reranker):
        self.ann = ann_index            # Faiss / HNSW
        self.ranker = dnn_ranker        # DIN / DCN-V2 / two-tower
        self.llm = llm_reranker         # GPT-4o or fine-tuned 7B

    def recommend(self, user_id, query_embedding, k=10):
        # Stage 1: ANN — millions to ~1000 in ~10ms
        cand_ids = self.ann.search(query_embedding, top_k=1000)

        # Stage 2: DNN ranker — 1000 to ~50 in ~30ms
        scored = self.ranker.score_batch(user_id, cand_ids)
        top50 = sorted(scored, key=lambda x: -x['score'])[:50]

        # Stage 3: LLM rerank — 50 to top-10 in ~1.5s
        # Critical: prompt fits in context, cost stays bounded
        return self.llm.rerank(user_id, top50, k=k)
```

The DNN ranker stage is what people new to LLMs underestimate. You cannot skip from ANN directly to LLM rerank: ANN's 1,000 candidates still contain too much noise (popular-but-irrelevant items dominate), and stuffing 1,000 items into a prompt either blows the context window or buries the signal. The DNN gives you a clean top-50 with *learned interaction patterns* the LLM doesn't have.

---

## Cold-Start: Where LLMs Actually Win

![Cold-start: LLM zero-shot vs CF as user interactions grow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/12-llm-recommendation/fig5_cold_start.png)

The LLM-vs-CF question has a sharp answer when you plot accuracy against user history length:

- **0–5 interactions** (cold): LLM zero-shot wins by 4–7×. CF has nothing to work with; the LLM's world knowledge is all the signal there is.
- **10–50 interactions** (warming up): the gap closes fast. CF starts to find genuine collaborative signal.
- **100+ interactions** (warm): CF wins on pure ranking accuracy. The LLM's world knowledge is now noise compared to the user's actual revealed preferences.

The corollary: **route by user state**.

```python
def route_recommend(user_id, query, cf_model, llm_model):
    """Use the model appropriate for this user's data density."""
    n_interactions = cf_model.interaction_count(user_id)

    if n_interactions < 5:
        # Cold: LLM uses world knowledge from query alone
        return llm_model.recommend(query)
    elif n_interactions < 50:
        # Warming: hybrid blend
        cf_recs = cf_model.recommend(user_id, k=50)
        return llm_model.rerank(query, cf_recs, k=10)
    else:
        # Warm: CF is now strictly better; LLM only for explanation
        return cf_model.recommend(user_id, k=10)
```

This is also why "we tried LLMs and they didn't beat our DCN" is a common false negative: the team almost certainly tested on warm users where CF was always going to win.

---

## Conversational Recommendation: ChatREC-Style

![Conversational recommendation flow with dialog state and tool use](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/12-llm-recommendation/fig6_chat_flow.png)

A conversational recommender is fundamentally different: it's a **planner**, not a one-shot ranker. The LLM maintains dialog state, decides when to retrieve, when to clarify, and when to recommend. The recommendation step itself often delegates back to a retrieval + DNN ranker stack.

```python
class ChatREC:
    """LLM as planner; retrieval and ranking are tools it calls."""

    def __init__(self, llm, retriever, ranker):
        self.llm = llm
        self.retriever = retriever      # ANN search
        self.ranker = ranker            # DNN
        self.state = {
            "history": [],              # message log
            "preferences": {},          # accumulated taste
            "candidates": [],           # current shortlist
        }

    def turn(self, user_message):
        self.state["history"].append({"role": "user", "content": user_message})

        # The LLM decides what to do next: clarify / retrieve / recommend / explain
        plan = self.llm.plan(
            history=self.state["history"],
            current_prefs=self.state["preferences"],
            current_candidates=self.state["candidates"],
        )

        if plan.action == "extract_preference":
            self.state["preferences"].update(plan.extracted_prefs)
            return self.llm.acknowledge(plan.extracted_prefs)

        if plan.action == "retrieve":
            query = self.llm.build_query(self.state["preferences"])
            cand = self.retriever.search(query, top_k=200)
            self.state["candidates"] = self.ranker.rank(cand, top_k=20)
            return self.llm.present(self.state["candidates"][:5])

        if plan.action == "filter":
            self.state["candidates"] = [
                c for c in self.state["candidates"]
                if plan.filter_fn(c)
            ]
            return self.llm.present(self.state["candidates"][:5])

        if plan.action == "explain":
            return self.llm.explain(plan.target_item, self.state["preferences"])

        return self.llm.respond(self.state["history"])  # general fallback
```

The hard parts are not in this code — they're in the planner prompt (which actions to expose, how to keep the LLM from hallucinating items not in `candidates`) and in the state management (when to discard preferences the user contradicted, when to refresh candidates after a filter).

---

## The Cost/Quality Pareto Frontier

![Cost vs quality Pareto frontier across LLM-RecSys methods](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/12-llm-recommendation/fig7_cost_quality.png)

When you map every published method onto (cost per 1K requests, NDCG lift over CF baseline), three things become clear:

1. **Returns diminish sharply above $1 per 1K requests.** The jump from a fine-tuned 7B (~$0.4) to GPT-4 generative (~$25) buys you ~2.4 NDCG points. The jump from CF baseline to fine-tuned 7B buys you ~8.6 points at less than 100× the latency cost.
2. **Fine-tuned 7B is the production sweet spot.** TallRec / LlamaRec deliver near-GPT-4 accuracy at ~1% of GPT-4's per-call cost. If you can afford 4× A100 hours once for LoRA training, you save the rest of the year on inference.
3. **Conversational agents are a different product, not a different point on the curve.** Their cost is high not because the model is expensive but because every session is multi-turn — you're paying 5–10× the requests for the same recommendation outcome. Justify them by the conversational UX itself, not by a ranking-metric improvement.

### Practical cost levers

| Lever | Typical savings | Caveat |
|---|---|---|
| Cache identical prompts | 30–60% | Only helps if queries repeat |
| Truncate history to 20 items | 20–40% | Loses long-tail preferences |
| Rerank top-20 instead of top-50 | 50% | Small accuracy hit (~1 NDCG pt) |
| Fine-tune 7B instead of GPT-4 | 90%+ | Needs labeled data + GPU time |
| Route warm users to CF only | 40–80% | Depends on user mix |

The single biggest lever, by far, is **route by user state** (previous section). For a typical platform 70%+ of requests come from warm users where CF is strictly better; routing them away from the LLM saves real money for free.

---

## Evaluation: What to Measure

Standard ranking metrics still apply (NDCG@K, Recall@K, MRR), but LLM systems need additional axes:

```python
import numpy as np


def evaluate_llm_recommender(predictions, ground_truth, llm_calls, latencies):
    """Standard metrics + LLM-specific operational metrics."""
    # --- Quality ---
    k = 10
    hits = [set(p[:k]) & set(gt) for p, gt in zip(predictions, ground_truth)]
    precision = np.mean([len(h) / k for h in hits])
    recall = np.mean([len(h) / max(len(gt), 1) for h, gt in zip(hits, ground_truth)])

    def ndcg(pred, gt, k=10):
        gt_set = set(gt)
        dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(pred[:k]) if p in gt_set)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt_set))))
        return dcg / idcg if idcg > 0 else 0.0

    ndcg_score = np.mean([ndcg(p, gt) for p, gt in zip(predictions, ground_truth)])

    # --- Operational ---
    p50_latency = float(np.percentile(latencies, 50))
    p99_latency = float(np.percentile(latencies, 99))
    cost_per_request = float(np.mean(llm_calls)) * 0.005   # $/call rough estimate

    return {
        "precision@10": precision,
        "recall@10": recall,
        "ndcg@10": ndcg_score,
        "p50_latency_s": p50_latency,
        "p99_latency_s": p99_latency,
        "cost_per_request_usd": cost_per_request,
    }
```

The operational metrics matter more than people expect. A model that wins NDCG by 2 points but adds 1.5s to p50 latency is a net loss for most consumer products — users churn faster than ranking quality improves.

For conversational systems, also track:

- **Turns to satisfaction**: how many turns until the user accepts a recommendation
- **Recommendation diversity per session**: are you collapsing to the same five items every conversation
- **Hallucination rate**: how often the LLM recommends an item not in the candidate set

---

## Frequently Asked Questions

### When should I use an LLM at all?

When at least one is true: (a) you have rich textual content per item, (b) you have a serious cold-start problem, (c) you need user-facing explanations, (d) you're building a conversational surface. If none of those apply, a well-tuned DCN-V2 or DIN will outperform any LLM at lower cost.

### Should I use GPT-4 or fine-tune a 7B?

Almost always fine-tune. **TallRec** and **LlamaRec** show that LoRA-tuned 7B models match or beat GPT-4 zero-shot on recommendation, at ~1% the inference cost. The only reason to stay on GPT-4 is if you have zero training data or your domain shifts faster than you can re-tune.

### How do I avoid hallucinated items?

Two layers: (1) constrain the LLM's output to indices into the candidate set (`[3, 1, 5, 2, 4]`, not titles), and (2) hard-validate the output against the candidate list before returning. If the LLM emits an unknown index, fall back to the DNN ranker's order.

### How big should the candidate set passed to the LLM be?

Empirically, 20–50. Below 20 you don't give the LLM enough to rerank meaningfully; above 50 the prompt gets long, slow, and expensive, and the model starts losing track of items in the middle ("lost in the middle" effect).

### Do I need RAG for recommendation?

Sometimes. RAG (retrieval-augmented generation) helps when the LLM needs *factual* knowledge about items beyond what's in the prompt — release dates, specifications, current availability. For pure ranking from a known catalog, the catalog itself in the prompt is the retrieval; you don't need a separate vector DB.

### What about latency for real-time feeds?

Real-time feeds (TikTok-style infinite scroll, news) generally cannot afford a synchronous LLM call per impression. The patterns that work: (a) pre-compute LLM rerankings during off-peak and cache, (b) use the LLM only for the *first* page and DNN for subsequent pages, (c) run the LLM on the offline batch path for next-day recommendations.

### How do I handle multilingual catalogs?

LLMs handle this naturally — translation and cross-lingual reasoning come built-in. The catch is that fine-tuning on English-only interaction data will collapse the multilingual capability. Either keep some non-English examples in your fine-tune mix, or fine-tune a multilingual base model (Qwen, Llama-3 multilingual, mT5).

---

## Conclusion

LLMs are not replacing recommendation systems — they're filling specific gaps the previous generation could not. The three roles map cleanly to the three gaps:

- **Enhancer** fills the *content understanding* gap. Run it offline, cache forever, get cleaner embeddings and stronger cold-start.
- **Predictor** fills the *cold-start and small-data* gap. Fine-tune a 7B, use it to rerank the top 20–50 from a traditional pipeline.
- **Agent** fills the *interaction* gap. Reserve it for genuinely conversational surfaces; pay the latency tax knowingly.

The architecture every production team converges on is the hybrid pipeline: ANN retrieval → DNN ranker → LLM reranker → optional explanation. It works because the LLM only sees the top ~50 items, never the catalog. Cost stays bounded; quality gets the LLM's semantic lift.

Start there. Measure operational metrics alongside ranking metrics from day one. Route warm users away from the LLM to keep the bill sane. Fine-tune as soon as you have a few hundred labeled examples.

---

## References

1. Geng, S., Liu, S., Fu, Z., Ge, Y., & Zhang, Y. (2022). [**Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)**](https://arxiv.org/abs/2203.13366). RecSys 2022.
2. Cui, Z., Ma, J., Zhou, C., Zhou, J., & Yang, H. (2022). [**M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems**](https://arxiv.org/abs/2205.08084). arXiv:2205.08084.
3. Bao, K., Zhang, J., Zhang, Y., Wang, W., Feng, F., & He, X. (2023). [**TallRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation**](https://arxiv.org/abs/2305.00447). RecSys 2023.
4. Ji, J., Li, Z., Xu, S., Hua, W., Ge, Y., Tan, J., & Zhang, Y. (2024). [**GenRec: Large Language Model for Generative Recommendation**](https://arxiv.org/abs/2307.00457). ECIR 2024.
5. Yue, Z., Rabhi, S., Moreira, G. de S. P., Wang, D., & Oldridge, E. (2023). [**LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking**](https://arxiv.org/abs/2311.02089). arXiv:2311.02089.
6. Gao, Y., Sheng, T., Xiang, Y., Xiong, Y., Wang, H., & Zhang, J. (2023). [**Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System**](https://arxiv.org/abs/2303.14524). arXiv:2303.14524.
7. Xi, Y., Liu, W., Lin, J., Zhu, J., Chen, B., Tang, R., Zhang, W., Zhang, R., & Yu, Y. (2024). [**Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models (KAR)**](https://arxiv.org/abs/2306.10933). RecSys 2024.
8. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). [**Lost in the Middle: How Language Models Use Long Contexts**](https://arxiv.org/abs/2307.03172). TACL.

---

## Series Navigation

This article is **Part 12** of the 16-part Recommendation Systems series.

| Previous | | Next |
|:---------|:-:|-----:|
| [Part 11: Contrastive Learning](/en/recommendation-systems-11-contrastive-learning/) | [All Parts](/tags/Recommendation-Systems/) | [Part 13: Coming Soon](/tags/Recommendation-Systems/) |
