---
title: "LLM Engineering (12): Production — Deployment, Monitoring, Cost"
date: 2026-04-07 09:00:00
tags:
  - LLM
  - production
  - Deployment
  - Monitoring
  - cost
  - autoscaling
categories: LLM Engineering
series: llm-engineering
series_order: 12
series_title: "LLM Engineering"
lang: en
mathjax: false
disableNunjucks: true
description: "Serving stack choices in detail, autoscaling LLMs, latency budgets, prompt+completion cost tracking, multi-model routing, FrugalGPT cascading, observability you need from day one, and the on-call patterns that work."
translationKey: "llm-engineering-12"
---

This is the last chapter. The previous ones covered building the model, the prompt, the retrieval, and the evaluation. This chapter focuses on maintaining it without breaking the bank. Production LLM serving is more like running a high-traffic web service than classical ML serving, except each web request costs money and can take up to two minutes.

I'll focus more on numbers here than in earlier chapters. In production, the difference between a profitable feature and a money pit often boils down to a 2-5x cost factor that no one is tracking. The most useful skill to develop is back-of-the-envelope cost arithmetic for LLM workloads. The numbers below are accurate as of late 2025 / early 2026; verify them against current pricing before committing.

![LLM Engineering (12): Production — Deployment, Monitoring, Cost — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/illustration_1.png)

## The serving stack, end to end

![fig1: end-to-end serving stack architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/fig1_stack_architecture.png)

A production LLM-app stack typically has these layers:

```
[CDN / WAF]
   ↓
[API Gateway]   ← rate limiting, auth, request normalization
   ↓
[App Server]    ← prompt assembly, RAG retrieval, tool execution, agent loop
   ↓
[LLM Gateway]   ← model routing, fallback, cost accounting, prompt caching
   ↓
[LLM Service]   ← vLLM/SGLang/managed API
   ↓
[Observability] ← logs, metrics, traces, eval runs
```

Most of the engineering work happens in the app server and LLM gateway. The app server handles the business logic, while the LLM gateway makes a portfolio of models act as a single service.

Build the LLM gateway as a separate service from the start. You'll need it for:

- **Multi-model routing**: send some requests to a small fast model, others to a large slow one based on classifier.
- **Fallback**: when the primary provider returns 5xx, retry with a secondary.
- **Cost tracking**: every request logged with prompt tokens, completion tokens, model, $-cost.
- **Prompt caching wrapper**: even if your provider supports caching, your app code shouldn't have to think about cache keys.
- **A/B testing**: route a configurable fraction of traffic to a new model variant.
- **Quotas / circuit breakers**: kill requests when a single user is consuming disproportionate cost.

Building this from scratch takes a few weeks. Open-source options cover much of the space:

- **LiteLLM** — pure-Python proxy with 100+ provider integrations, drop-in OpenAI-compatible endpoint, decent built-in cost tracking. The fastest way to get a usable gateway running.
- **OpenRouter** — managed gateway with a single API across providers and a built-in model marketplace. Higher per-token cost than going direct to providers, but handles failover and pricing arbitrage automatically.
- **Cloudflare AI Gateway** — managed proxy at the CDN edge, with caching, rate limits, and analytics. Cheap and operationally trivial; gives up some flexibility.
- **BentoML / Bento Cloud** — heavier framework, useful when you also need to host self-trained models behind the same gateway.
- **Portkey, Langfuse Gateway** — newer entrants with strong observability stories.

Choose one early and plan to switch later. The gateway is one of the few places where vendor lock-in is a real issue (you write a lot of code against its contract), so keep the contract small.

## Self-host vs managed API

A perennial decision. The honest answer for 2026:

**Use managed APIs when:**

- You need frontier-model quality (GPT-5, Claude-4.5, Gemini-3) and don't have the GPUs to host them.
- Volume is below ~1B tokens/month — managed is usually cheaper at this scale.
- Latency requirements are loose (>500 ms TTFT acceptable).
- You can't dedicate engineering to GPU operations.

**Self-host when:**

- Volume is above 1B tokens/month and the open model fits your quality bar (Qwen3-32B+, LLaMA-3.3-70B+).
- Strict data residency requirements.
- You need <100 ms TTFT consistently.
- You have specific fine-tuning needs.

The break-even point is roughly 1B tokens/month for a Qwen3-32B-class workload. Below that, managed APIs beat self-hosting on cost when you account for engineering time. Above that, self-hosting on dedicated GPUs (rented or owned) wins by 3-10x on cost.

A common mistake is under-utilizing self-hosted GPUs. A 4xH100 deployment running at 30% utilization is more expensive per token than the OpenAI API. Aim for sustained throughput above 70%.

A practical hybrid pattern that gained popularity in 2025-2026: **self-host the bulk of cheap traffic on smaller open models and route the challenging 5-10% to managed frontier APIs.** This captures most of the cost savings of self-hosting while preserving access to frontier quality where it matters. The routing decision is made by a small classifier (see the chapter on multi-model routing below).

## Multi-model routing and FrugalGPT

![LLM Engineering (12): Production — Deployment, Monitoring, Cost — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/illustration_2.png)

Chen et al. (2023, *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*) formalized the routing-then-cascading pattern. The key insight: most LLM queries are easy and can be handled by a small, cheap model. Only a small fraction truly need the frontier model. If you can distinguish these efficiently, you can reduce costs by 5-10x with minimal quality loss.

FrugalGPT proposes three patterns:

1. **Prompt adaptation** — compress prompts, drop redundant context, batch related requests.
2. **LLM cascade** — try the cheapest model first, ask it for a confidence signal, fall back to a more expensive model if confidence is low.
3. **Model routing** — a learned classifier that picks the right model upfront.

The cascade is the most commonly deployed in practice. Pseudocode:

```python
def cascade(question, models=[CHEAP, MID, EXPENSIVE]):
    for m in models:
        answer, confidence = m.generate_with_confidence(question)
        if confidence > THRESHOLDS[m]:
            return answer
    return answer  # last model's output
```

The hard part is the confidence signal. Options:

- **Self-reported confidence** — ask the model "rate your confidence 1-5." Lightly correlated with correctness but noisy.
- **Logprob-based** — use the logprob of the answer span. Reasonable but only works when you control sampling.
- **Sample agreement** — sample the cheap model 3-5 times; if all agree, accept. If they disagree, escalate.
- **Task-specific verifier** — for code, "does it pass the public test?"; for math, "is the answer in a sane range?"

Embedding-based routing (Hu et al., 2024, *RouteLLM: Learning to Route LLMs with Preference Data*) trains a classifier on (query, best-model) pairs from offline preference data to predict which model to use. RouteLLM-trained routers achieve 95% of GPT-4 quality on MT-Bench at 26-44% of the cost.

In production the routing decision often collapses to two questions: "is this query simple?" (route cheap) and "is this query a known-hard pattern?" (route expensive). The middle ground gets the default. A 50-question heuristic classifier captures 80 % of the savings; learned routers capture maybe 90 %.

## Caching tiers

Three caches matter, and they stack:

1. **Provider-side prompt cache** (chapter 9) — the KV-cache reuse for repeated prompt prefixes. Run by Anthropic / OpenAI / DeepSeek / Google. 90 % cost reduction on the cached portion.
2. **Application-side semantic cache** — when two user queries are *semantically similar*, serve the cached answer. Tools: GPTCache (the original, FAISS-based), Redis Semantic Cache, MemCache + custom embedding lookup. 30-70 % hit rate for FAQ-style workloads, near 0 % for open-ended chat.
3. **Result cache for deterministic functions** — if the LLM output feeds into a downstream pipeline (e.g., classification, extraction) and inputs are exact-match, just store the output keyed on the input hash. The LLM never gets called twice for the same input.

GPTCache is worth a closer look. The pattern: embed each user query with a small embedding model, compare against the cache (cosine similarity), and serve the cached answer if similarity > threshold (typically 0.95). The threshold is the dial — higher means fewer false positives but lower hit rate. For a customer-support bot answering common questions, a 0.92 threshold can deliver 50 % cost reduction; for a code-generation assistant where every query is unique, semantic cache is ~useless.

The composability matters: prompt cache reduces the cost of *each* call; semantic cache reduces the *number* of calls. They multiply.

## Autoscaling LLM workloads

![fig4: autoscaling LLM workloads](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/fig4_autoscaling.png)

Autoscaling LLMs is harder than autoscaling stateless web servers. Three reasons:

1. **Loading a model takes minutes**. A new vLLM replica with a 70B model needs 2-4 minutes to load weights and warm up. Autoscaling in response to a traffic spike is too slow if the spike lasts only 10 minutes.
2. **GPU is paid by the hour, not the request**. Spinning a fresh GPU instance for a 5-minute spike wastes 55 minutes of GPU.
3. **Continuous batching is non-linear in load**. A vLLM server at 50 % load and 90 % load have very different latency profiles; the safe operating point is much lower than CPU services suggest.

Practical patterns:

- **Pre-warm on schedule**: if traffic predictably spikes at 9am, pre-scale at 8:50.
- **Conservative scale-up, eager scale-down**: scale up at 60 % utilization, scale down at 30 %. (Reverse of typical web service.)
- **Buffered fallback**: when self-hosted is at capacity, overflow to a managed API. Higher unit cost but absorbs spikes without provisioning waste.
- **Multi-model floor**: keep at least one replica of each model online even at low traffic. Cold-start cost is too high.
- **Queue with backpressure**: when the server is overloaded, queue requests with a per-user fairness scheduler and return a 429 with a `Retry-After` header rather than crashing the latency for everyone.

vLLM supports `--max-num-seqs` (max in-flight requests) and `--max-num-batched-tokens` (max token throughput per step). Tune these to match your latency target rather than maximizing throughput.

## Latency budget breakdown

![fig2: latency budget breakdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/fig2_latency_breakdown.png)

A user-facing chat product has roughly this latency budget:

| Component | Budget (ms) | Notes |
|---|---|---|
| Network in | 50 | Geographic |
| API gateway / auth | 10 | Should be fast |
| App server logic | 50 | RAG embedding, tool dispatch |
| RAG retrieval | 100 | Vector DB + reranker |
| LLM gateway overhead | 5 | Just routing |
| LLM TTFT (queue + prefill) | 300 | The model itself |
| Network out | 50 | Same as in |
| **Total to first token** | **~565 ms** | Below 1s target |
| LLM ITL (decode) | 25 | Per-token, sustains 40 tok/s |
| Network buffering | 20 | Smoothing |

Two latency metrics matter most: **TTFT** (Time To First Token) and **ITL** (Inter-Token Latency). TTFT is what the user feels as "did anything happen?"; ITL determines the sustained read speed once tokens are streaming. Industry rules of thumb in 2026:

- Chat: TTFT < 800 ms is "instant," < 2 s is "acceptable"
- Voice: TTFT < 300 ms required for natural conversation
- Code completion: TTFT < 200 ms or users abandon
- Batch / agentic: TTFT can be 5-30 s; users tolerate it

Once first token is out, the user sees streaming and tolerates ~40-60 tok/s for 5-10 seconds. Total acceptable response time depends on length: 200 tokens at 40 tok/s = 5s plus 600 ms TTFT = 5.6 s end-to-end.

Where to spend latency budget:

- **Reranking**: 50-100 ms is well-spent (chapter 8).
- **Speculative decoding**: -50 to -100 ms ITL (chapter 5), good investment.
- **Tool calls**: a 200-ms tool blocks the model. Parallelize where possible, cache aggressively.
- **Reasoning / thinking**: a thinking model adds 1-10 s before first user-visible token. Reserve for tasks that need it.

For latency-critical paths, *measure where you actually spend time*. The breakdown is rarely what you'd guess. I've seen a "slow" feature where the LLM was 200 ms but the RAG reranker was 1.2 s; another where the LLM was fine but the JSON serialization downstream was 800 ms. Trace before you optimize.

## Cost tracking from day one

![fig3: cost per request breakdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/fig3_cost_per_request.png)

Per-request cost accounting is non-negotiable. Without it, you cannot:

- Identify cost-bloat from a single user or feature.
- A/B test cost-vs-quality changes.
- Forecast spend before invoices arrive.
- Catch a runaway agent loop before it spends $1000.

Every request should be logged with at minimum:

```python
{
    "request_id": "req_xxx",
    "user_id": "user_xxx",
    "endpoint": "/chat",
    "model": "claude-4-5-sonnet-20250901",
    "prompt_tokens": 4321,
    "cached_tokens": 3000,    # of those, came from cache
    "completion_tokens": 215,
    "total_cost_usd": 0.0152,
    "latency_ms": {
        "ttft": 350,
        "total": 5430,
    },
    "tools_called": ["search", "fetch_doc"],
    "ts": 1714123456789,
}
```

Aggregate by user, by feature, by model. Alert when:

- A single user exceeds 10x median spend in 1 hour.
- A feature's cost-per-call doubles week-over-week.
- A new model's cost-per-resolved-conversation exceeds the old model's despite quality being equal.

A useful arithmetic exercise. Suppose your product makes 10K LLM calls/day, average 4K input + 500 output tokens, on Claude-4.5 Sonnet:

- Per call: $4 \cdot 4 + 15 \cdot 0.5 = 16 + 7.5 = 23.5$ thousand-tok-cents = $0.0235.
- Per day: $235.
- Per month: $7050.

Now add prompt caching (3K of the 4K input cached):

- Per call: $4 \cdot 1 + 0.30 \cdot 3 + 15 \cdot 0.5 = 4 + 0.9 + 7.5 = 12.4$ thousand-tok-cents = $0.0124.
- Per month: $3720, savings of $3330 (47 %).

Now add a 30 % semantic cache hit rate:

- Per month: $3720 \cdot 0.7 = $2604, total savings of $4446 (63 %).

Now add cascading where 60 % of queries are handled by Qwen3-32B (self-hosted, $0.10 input + $0.30 output per Mtok):

- Cheap path per call: $0.10 \cdot 4 + 0.30 \cdot 0.5 = 0.55$ thousand-tok-cents = $0.00055.
- Mixed monthly: $0.6 \cdot 7000 \cdot 0.00055 + 0.4 \cdot 7000 \cdot 0.0124 = $2.31 + $34.72 ... per day. Times 30 days = $1110/month, 84 % cheaper than baseline.

These savings compound. Without explicit cost tracking, none of them are visible.

## Observability beyond cost

![fig5: observability dashboard](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/fig5_observability_dashboard.png)

Standard observability (traces, logs, metrics) plus LLM-specific:

- **Per-model latency and error rates**, broken out by model.
- **Token-throughput per replica** for self-hosted (a regression here often signals model load issues).
- **Quality drift**: sample 1 % of requests, run them through your eval set's grading flow, plot pass rate over time.
- **Prompt-injection detection**: rate of detected injection attempts per user.
- **Refusal rate**: model refused fraction. Sudden changes signal either an attack or a model regression.
- **Cache hit rates**: prompt cache, semantic cache. Hit rate dropping signals either a prompt-template change broke caching or traffic shifted.

Tools that work in 2026:

- **Langfuse** (open source) — best for LLM-specific tracing, generous free tier, easy self-host.
- **Helicone** — managed LLM observability with cost tracking; strong out of the box.
- **Phoenix (Arize)** — open-source LLM observability with a focus on retrieval/RAG quality metrics.
- **OpenLLMetry** — OpenTelemetry-based, plugs into existing observability stacks (Datadog, Grafana, Honeycomb).
- **Datadog + custom dashboards** — works if you already have Datadog and want a single pane of glass.

The cheapest pattern that works: structured-log every request, ship to ClickHouse or Snowflake, build a dashboard. You don't need a managed observability product to get 80 % of the value, but you do need the discipline to log the right fields.

## On-call patterns that work

![fig6: on-call escalation flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/fig6_oncall_escalation.png)

Things that page me at 3am and the runbook for each:

**Spike in 5xx from LLM provider.** Check provider status page. Failover to secondary provider if available. If primary returns to normal in <5 min, no further action; if longer, page provider account manager. Check that failover didn't cause a cost spike on the secondary (sometimes the secondary is 2x more expensive).

**Latency p99 > 30s.** Usually a runaway tool call or a stuck request. Check if a single user/feature dominates. Often resolved by killing stuck requests and adding tighter timeouts.

**Cost spike**: sudden 10x increase in $/hour. Check user table — almost always one user with a runaway agent. Rate-limit them, investigate root cause. The classic case is an agent that calls itself recursively without a step limit.

**Refusal rate jump**: model started refusing 30 % more requests overnight. Either a model rollout regressed or there's a coordinated attack. Roll back model, investigate.

**RAG quality drop**: faithfulness score dropped 10 %. Usually an embedding/index version mismatch or new content with different format. Check ingestion pipeline.

**OOM on self-hosted GPU**: KV cache exhausted. Either add capacity or lower per-request `max_tokens`. Don't restart blindly; the next batch will OOM the same way.

**Provider deprecation notice**: a model version we depend on will be removed in 90 days. Immediately schedule a migration test against the recommended successor, A/B for at least 2 weeks, document the diff.

A good runbook reduces mean-time-to-recovery from 30 min to 3 min. Write the runbook *before* you need it.

## Common failure modes worth naming

A non-exhaustive list of production incidents I've seen or read about:

- **Provider regional outage.** A whole AZ goes down. Mitigation: multi-region failover at the gateway; for managed APIs, a secondary provider on standby.
- **Rate-limit cascade.** Provider rate-limits one of your customers; your app retries aggressively and eats the global quota. Mitigation: per-user rate limits before the gateway; exponential backoff with jitter on retries; circuit-breaker per provider.
- **Prompt template regression.** A "harmless" change to the system prompt invalidates the prompt cache (chapter 9), tripling your bill overnight. Mitigation: cache-aware diff in CI; alert on cache hit rate drop.
- **Model deprecation surprise.** OpenAI announces gpt-4-turbo-0613 will be retired in 60 days; your eval set was never run against the replacement. Mitigation: subscribe to provider deprecation feeds; quarterly migration drills.
- **Silent quality drift.** A managed model receives an undocumented update that regresses on your specific use case. Mitigation: run your eval set against production traffic samples weekly; compare to historical baseline; alert on >5 % drop.
- **Agent loop runaway.** An agent invokes a tool, receives an error, retries, gets a different error, retries with the model now arguing with itself, racks up $500. Mitigation: per-conversation step limits; per-conversation cost limits; alert on cost > $X per session.

## Migration playbook

Migrating between models is the most underrated production skill. The pattern that works:

1. **Shadow traffic for 3-7 days.** New model receives a copy of every request but its output is logged, not returned. Compare quality offline.
2. **Canary 1 % for 3-7 days.** New model serves 1 % of real traffic; monitor latency, error rate, cost, refusal rate, satisfaction. No regression → continue.
3. **Ramp 5 → 25 → 50 → 100 %** over 1-2 weeks. Each step gated on monitoring SLAs.
4. **Keep the old model warm for 30 days post-cutover.** Rollback flag is one config change away.
5. **Decommission only after 30 days of clean operation.**

Skipping shadow + canary is the most common cause of "we shipped a new model and customer complaints spiked." It's a process discipline, not a technology.

## Cost by deployment shape (rough numbers, late 2025)

| Setup | $/Mtok input | $/Mtok output | Notes |
|---|---|---|---|
| Claude-4.5-Sonnet API | $3 | $15 | Strong quality |
| GPT-4o API | $2.50 | $10 | Strong quality |
| Qwen3-Max API | $1.40 | $5.60 | Strong, cheaper |
| Gemini-2.5-Pro API | $2.50 | $10 | Strong, long context |
| DeepSeek-V3 API | $0.14 | $0.28 | Cheap, strong on math/code |
| Self-host Qwen3-32B FP8 (1xH100) | $0.10 | $0.30 | At 70 % util |
| Self-host LLaMA-3.3-70B FP8 (2xH100) | $0.30 | $0.90 | At 70 % util |
| Self-host Qwen3-235B-A22B (8xH100) | $0.50 | $1.50 | At 70 % util |
| Open-router pooled small models | $0.15 | $0.50 | Cheapest |
| Anthropic Batch API (50 % discount) | $1.50 | $7.50 | 24h SLA |
| OpenAI Batch API (50 % discount) | $1.25 | $5 | 24h SLA |

Output is consistently 3-5x more expensive than input because output is sequential decode (memory-bound). For applications that produce long outputs (summaries, reports, code), shifting to a model with cheaper output economics often beats optimizing the model itself.

The Batch API discount (50 % off for 24-hour-SLA workloads) is one of the most underused cost levers. If your workload tolerates next-day completion (overnight reports, bulk content moderation, eval sets, dataset labeling), batching cuts spend in half. The migration is usually a one-line API change.

## Final recipe for a cost-effective production LLM product

If I were starting a production LLM product in mid-2026:

1. **Default to a managed API** for the first 6-12 months. Pick based on your latency / quality / language needs.
2. **Build the LLM gateway** from day one even if it just wraps one provider. You will need it.
3. **Track cost per request** to two decimal places (in $0.0001). Aggregate weekly.
4. **Build a 200-question eval set** before iterating on prompts.
5. **Layer retrieval + reranker** if you have any external knowledge to ground on.
6. **Cache aggressively**: prompt caching at provider, semantic cache for repeated questions in your app, result cache for deterministic flows.
7. **Add tool use** when single-shot prompts can't solve real user requests.
8. **Cascade to cheap models** for the easy 60-80 % of queries, route hard queries to frontier.
9. **Self-host when you cross 1B tokens/month** with consistent traffic.
10. **Quantize to FP8** on H100s, AWQ-INT4 on A100s/L40s.
11. **Use batch API** for any workload that tolerates a 24-hour SLA.
12. **Add the safety stack** before you have to.

This is enough to ship and grow without the technical debt that kills most LLM products.

## Series wrap-up

Twelve chapters on what it takes to build a modern LLM product end-to-end:

1. Architectures
2. Tokenization
3. Pretraining
4. Post-training
5. Inference optimization
6. Long context
7. Function calling
8. RAG
9. Prompting
10. Evaluation
11. Safety
12. Production

If you only remember three things: data quality dominates everything (chapters 3 & 4), tokenization and KV cache pay the bills (chapters 2 & 5), and you cannot ship without an eval set and cost tracking (chapters 10 & 12). Everything else is execution.

For follow-up reading, the [NLP series](/en/nlp/) covers the foundations more deeply, the [Aliyun Bailian series](/en/aliyun-bailian/) shows the same patterns through a specific cloud platform, and the [Aliyun PAI series](/en/aliyun-pai/) covers the training-serving infrastructure on Alibaba Cloud.

Thanks for reading this far. Build something that works.

## References

- Chen, L. et al. (2023). *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*. https://arxiv.org/abs/2305.05176
- Hu, Q. et al. (2024). *RouteLLM: Learning to Route LLMs with Preference Data*. https://arxiv.org/abs/2406.18665
- Bang, Y. et al. (2023). *GPTCache: An Open-Source Semantic Cache for LLM Applications*. NLP-OSS @ EMNLP 2023. https://arxiv.org/abs/2311.04205
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP 2023. https://arxiv.org/abs/2309.06180
- LiteLLM project. *LiteLLM: Call all LLM APIs using the OpenAI format*. https://github.com/BerriAI/litellm
- Cloudflare (2024). *Cloudflare AI Gateway*. https://developers.cloudflare.com/ai-gateway/
- OpenRouter (2025). *OpenRouter: A unified interface for LLMs*. https://openrouter.ai/
- Langfuse project. *Langfuse: Open-source LLM engineering platform*. https://langfuse.com/
- OpenLLMetry project. *OpenLLMetry: Open-source observability for LLM applications*. https://www.traceloop.com/openllmetry
- Anthropic (2024). *Message Batches API*. https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
- OpenAI (2024). *Batch API*. https://platform.openai.com/docs/guides/batch
- Anyscale (2023). *Reproducing GPT-2 and Llama-2 inference cost analysis*. https://www.anyscale.com/blog/llm-inference-cost-analysis
- Together AI (2024). *Together Inference Engine performance and cost benchmarks*. https://www.together.ai/blog/together-inference-engine-2
- Fireworks AI (2024). *FireAttention serving stack*. https://fireworks.ai/blog/fire-attention-serving-stack
- HuggingFace TGI project. *Text Generation Inference*. https://github.com/huggingface/text-generation-inference
