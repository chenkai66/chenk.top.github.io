---
title: "LLM Workflows and Application Architecture: Enterprise Implementation Guide"
date: 2024-07-16 09:00:00
tags:
  - Workflow
  - Architecture
  - RAG
  - LLM
categories: Large Language Models
lang: en
description: "From a single API call to a production LLM platform — workflow patterns, RAG, model routing, deployment, cost levers, observability, and enterprise integration, with the trade-offs that actually matter."
disableNunjucks: true
---

Most LLM tutorials end where the interesting work begins. They show you how to call a chat completion endpoint, attach a vector store, and wrap the whole thing in a Streamlit demo. None of that is wrong, but none of it is what breaks at 3 a.m. when 10,000 users hit your service at once and every other answer is a hallucination.

This article is about everything that comes after the demo. It is opinionated on purpose: production LLM systems are mostly plain distributed systems with one non-deterministic component bolted on, and most of the engineering effort goes into containing that non-determinism. We will work through seven dimensions — application architecture, workflow patterns, the RAG-vs-fine-tune decision, deployment topology, cost, observability, and enterprise integration — keeping each one short, concrete, and grounded in the levers that actually move the needle.

## What you will learn

- How to layer an LLM application so that only one layer is probabilistic
- Four workflow patterns (chain, branch, loop, parallel) and when each is the wrong choice
- A decision rule for prompts vs RAG vs fine-tuning that survives contact with reality
- A production deployment topology with semantic cache, LLM gateway, and a model fleet
- Six cost levers in the order you should pull them
- An observability stack that adds the four LLM-specific signals beginners forget
- The six enterprise boxes that block most pilots from going live

## Prerequisites

Comfortable with REST APIs, async Python or Node, basic Docker, and at least one LLM provider SDK. You do not need to have run a vector database in production — we will keep the engineering recipes self-contained.

---

## 1. The LLM application stack

![LLM application stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/llm-workflows-architecture/fig1_application_stack.png)

The single most useful mental model for building LLM products is to treat them as ordinary five-layer applications where exactly one layer happens to be probabilistic. From top to bottom:

1. **Experience layer** — web, mobile, IDE plugins, chat platforms. Streaming UX lives here, not in the model.
2. **Agent / orchestration layer** — the planner, tool router, memory, and guardrails. This is where most of your code goes.
3. **Retrieval and context layer** — RAG, semantic cache, session store, and the prompt builder that assembles the final message list.
4. **Model serving layer** — an LLM gateway in front of one or more model providers, with routing, fallback, and streaming.
5. **Tools and data plane** — SQL, search, code execution, third-party APIs, vector databases. These are deterministic and can be tested like normal services.

Cutting across all five layers is the usual cross-cutting column: authentication and authorisation, audit and PII redaction, rate limits, observability, cost accounting, and evaluation. The reason this layering matters is that it tells you where to spend engineering effort. A team that puts retry logic in the experience layer, prompt templates in the model layer, and rate limits everywhere is not going to ship anything maintainable. Push concerns down to the layer that owns them, and the model layer stays small.

A second consequence of the layering: most reliability work is not LLM work. The model is a black box you call over HTTP. The interesting questions — *did the right tool get called, with the right arguments, against the right tenant's data, within the right budget?* — all live in the orchestration and retrieval layers, which are deterministic Python or TypeScript and can be unit-tested.

## 2. Workflow patterns: pick the smallest one that works

![Workflow orchestration patterns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/llm-workflows-architecture/fig2_workflow_patterns.png)

Anthropic's [*Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents) (2024) makes a point worth internalising: most of what people call an "agent" is better served by a small fixed workflow. Workflows are predictable, cheap, and easy to evaluate. Agents — systems where the LLM dynamically picks its own next step — are the right answer only when the task space is genuinely open-ended.

There are four patterns you actually need.

**Chain.** A sequential pipeline: extract, transform, summarise. Use when the steps are deterministic and the order is known. The implementation is a list of functions. Don't reach for a framework.

**Branch.** Classify the input with a small model, then route to the appropriate handler — FAQ to a tiny LM, code questions to a tool-using path, hard reasoning to a frontier model. This single pattern is responsible for most of the cost wins you can get without changing model providers; in our own deployments, an LLM router in front of a 3-tier model fleet typically reclaims 60–80 % of spend versus always calling the largest model.

**Loop.** Generate, critique, revise. The critique can be an LLM-as-judge, a regex check, a unit-test runner for code, or a SQL dry-run. Loops only work if you bound them — pick a maximum iteration count (3 is usually enough), a hard token cap, and a stop condition that does not require another model call.

**Parallel.** Fan out to N workers, then reduce. This is the right shape for chunked summarisation, self-consistency voting, and multi-source synthesis. Two warnings: parallel calls multiply your cost by N, and the reduce step usually needs the largest model in the fleet, because synthesis is harder than generation.

A useful test before adding any pattern: *can a non-LLM service do this step?* If a regex, a SQL query, or a function call works, use it. Every LLM call is a place latency, cost, and non-determinism leak in.

## 3. RAG vs fine-tuning vs prompt engineering

![Decision tree: prompts, RAG, fine-tuning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/llm-workflows-architecture/fig3_rag_vs_finetuning.png)

This is the most-asked question in any LLM design review, and it has a simple answer: pick the cheapest tool that closes the gap. The gap has two flavours and the diagnosis takes one minute.

If the model produces the **wrong style or format** but the facts are roughly right, you have a *behaviour gap*. Start with prompt engineering — few-shot examples, chain-of-thought scaffolds, structured output constraints. Schulhoff et al.'s [*The Prompt Report*](https://arxiv.org/abs/2406.06608) (2024) catalogues the techniques that have actually held up across model generations. Only after prompts are exhausted should you fine-tune. Fine-tuning fixes style cheaply once you have a few thousand labelled examples and a stable schema; LoRA and DPO have made this affordable, but it is still slower to iterate on than a prompt.

If the model is **missing knowledge** — your product, your customers, your last quarter's outage — you have a *knowledge gap*, and RAG is the default answer. RAG wins on three properties prompts and fine-tuning cannot match: it cites sources (auditable answers), it refreshes in minutes (dump documents, re-embed, done), and it scales to corpora that no fine-tune could memorise. The original [Lewis et al. RAG paper](https://arxiv.org/abs/2005.11401) (NeurIPS 2020) is still the cleanest formulation; Karpukhin et al.'s [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (EMNLP 2020) gives the bi-encoder retriever you will actually deploy.

The combinatorial case — RAG **plus** fine-tuning — is real but rare. It is worth the cost only when the corpus is stable, the domain is narrow, and quality matters more than iteration speed. Examples: medical Q&A in a single specialty, internal tooling with a fixed schema, regulated financial summaries. Fine-tune the retriever or the reader, keep RAG for freshness, accept the operational complexity.

A few practical notes:

- The naive RAG you build in an afternoon will score poorly on anything but the easiest queries. The order in which to add complexity is roughly: better chunking → hybrid search (dense + BM25) → cross-encoder reranking → query rewriting / HyDE. Each adds latency and cost, so add only on measured failure modes.
- Long-context models do not eliminate RAG. Liu et al.'s [*Lost in the Middle*](https://arxiv.org/abs/2307.03172) (TACL 2024) showed that even at 128k tokens, models attend to the start and end of context far more than the middle. Retrieval still wins on precision and on cost.
- Fine-tuning a frontier model is mostly unavailable. Fine-tune open-weight models (Llama, Qwen, Mistral) for behaviour, and call frontier APIs for the cases that need them. The branch pattern from §2 is what makes this practical.

## 4. Production deployment topology

![Production deployment](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/llm-workflows-architecture/fig4_production_deploy.png)

The shape of a production LLM service is not unusual; what is unusual is which boxes do the heavy lifting. Reading left to right:

**Edge.** A CDN with a WAF and an aggressive rate limit. LLM endpoints are uniquely attractive to abuse — every request costs you real money — so the edge limit should be tighter than for normal APIs and should be enforced per identity, not per IP.

**API gateway.** Authentication, per-tenant quota, request-level audit, and the routing rules that decide which version of which app a request hits. Quota at this layer should be denominated in tokens or dollars, not requests, because a single 100k-token query can cost more than 1,000 short queries.

**Application servers.** Stateless FastAPI or Node services behind a horizontal pod autoscaler. They orchestrate the workflow from §2, but they do not call the model directly — they call the LLM gateway.

**Semantic cache.** Sitting just before the LLM gateway, this is the highest-leverage box on the diagram. Two layers: an exact-match cache keyed on the canonicalised request, and a semantic cache that embeds the user's question and returns a previously-cached answer if cosine similarity exceeds a threshold (0.95 is a defensible default). In production, hit ratios of 30–60 % are routine for support, FAQ, and analytics workloads; for open-ended creative work, expect 5–10 %.

**LLM gateway.** Owns model selection, fallback (when the primary 5xxs or rate-limits, the gateway downgrades to a secondary), per-request token budgeting, and streaming. Putting this concern in one service rather than scattered across application code is what lets you change model providers in an afternoon instead of a quarter.

**Model fleet.** Three tiers: a small local model on your own GPU pool for the highest-volume / lowest-difficulty traffic; an open-weight mid-tier for general use; and a frontier API for the hard cases. A code or tool sandbox sits beside the models — never run model-generated code in your application process.

**Async track.** A message queue with workers handles everything that does not need a synchronous response: document ingestion and embedding, scheduled evals, cost and usage rollups, batch generation, retraining jobs.

**Telemetry.** All boxes emit OpenTelemetry traces and Prometheus metrics. We come back to this in §6.

A common mistake is to skip the LLM gateway and let application code talk directly to model providers. This works until the day you need to swap a provider, add a fallback, or enforce a token budget — at which point every service needs to change. Build the gateway on day one even if it has only one provider behind it.

## 5. Cost optimisation

![Cost optimisation levers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/llm-workflows-architecture/fig5_cost_optimization.png)

LLM cost is the single most common cause of pilots that work in dev but get killed in finance review. There are six levers; pull them in this order, because each builds on the previous.

1. **Prompt compression.** Shorter prompts cost less in three ways: input tokens, time-to-first-token, and (because attention is quadratic) infrastructure load. Strip boilerplate from system prompts, use IDs instead of names where you can, and drop conversation history older than the model can effectively use.
2. **Small-model routing.** The branch pattern from §2 — classify each request, send only the hard ones to the expensive model. The classification itself can be done by a 7B model for fractions of a cent.
3. **Semantic cache.** Discussed in §4. Hit ratio is what determines its value; instrument it from day one.
4. **Batching.** OpenAI, Anthropic, and most providers offer batch APIs at roughly half the price of synchronous calls, with a 24-hour SLA. Anything not user-facing — overnight summarisation, embedding refresh, eval runs — should go through the batch endpoint.
5. **INT8 / INT4 quantisation.** For self-hosted models, [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., ICLR 2023) and similar techniques cut memory and latency by roughly 2× with negligible quality loss on most tasks. This is the lever to pull when GPU cost dominates.
6. **Negotiate or self-host.** Once you know your traffic shape, committed-spend discounts and self-hosted open-weight models become real options. The break-even versus a frontier API is usually around 50–100 M tokens per month per workload.

The right side of the figure shows the quality / cost frontier. Three observations from production traffic: small local models can hit 60–65 % of frontier quality for general traffic and close to 95 % on narrow tasks; INT8 quantisation moves you down-and-slightly-left (cheaper, almost as good); and adding RAG on top of any tier is the largest quality jump available short of switching providers. Plot your own version of this chart for your own evals and use it for routing decisions.

A warning about benchmarks. Public leaderboards (MMLU, MT-Bench, Arena) generalise badly to specific products. Build a small task-specific eval (50–500 prompts with reference answers or rubric scores), re-run it whenever you change the model or the prompt, and trust it more than any public number.

## 6. Observability

![Observability stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/llm-workflows-architecture/fig6_observability.png)

Standard observability tooling — Prometheus, OpenTelemetry, Loki / Tempo / Grafana — works for LLM services with one extension: you need four LLM-specific signals on top of the [Google SRE golden signals](https://sre.google/sre-book/monitoring-distributed-systems/) (latency, traffic, errors, saturation).

**Cost per request** is the fifth golden signal. Track it as a histogram, alert on the p95, and break it down by tenant, endpoint, and model. Cost is the metric that turns into a Slack message from finance.

**Hallucination rate.** Run an LLM-as-judge sample (1–5 % of traffic is plenty) that asks a stronger model whether the response is supported by the retrieved context. The absolute number is noisy; the trend over time is not.

**Groundedness / citation hit rate.** For RAG systems, what fraction of factual claims in the answer can be traced back to a retrieved chunk? This is a leading indicator of user-facing hallucinations and is cheap to compute.

**Guardrail trips.** How often does the prompt-injection filter, the PII redactor, or the moderation classifier fire? A sudden spike usually means an attacker found a new bypass; a gradual rise usually means user behaviour shifted and your filters need retuning.

**Cache hit ratio and fallback rate** round out the picture. Cache hit ratio tells you whether the semantic cache is earning its keep. Fallback rate per provider tells you when an upstream is degrading before users complain.

Distributed tracing is non-optional for any non-trivial workflow. The [Dapper paper](https://research.google/pubs/pub36356/) (Sigelman et al., 2010) is still the right mental model. A typical trace for a RAG request has spans for `retrieve`, `rerank`, `compose_prompt`, `llm.generate`, and any tool calls — without these spans, "the request was slow" is unactionable.

One pattern worth adopting: hash prompts and responses before logging them. You get the operational visibility you need (which prompt template, which response length, which user) without the compliance liability of storing raw user text indefinitely. When you do need the raw content for a debugging session, gate it behind a separate, audited path.

## 7. Enterprise integration

![Enterprise integration patterns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/llm-workflows-architecture/fig7_enterprise_patterns.png)

The honest summary of why enterprise pilots stall: it is rarely model quality. It is the six boxes around the model. Each maps to a procurement question that has to have a defensible answer.

**Identity.** SSO via SAML or OIDC, automated user lifecycle via SCIM. Without this, every customer onboarding is a multi-week project.

**Authorisation.** Role-based access control for application features, attribute-based access control for data. For RAG, this means document-level filters that are enforced at the retrieval layer — never trust the model to respect "do not mention document X." If a tenant cannot see a document, the retriever must not return it.

**Data residency.** Regional routing for both storage and inference. EU customers want EU-hosted models and EU-hosted vector stores, full stop. This is a topology decision that is painful to retrofit, so make it before the first European customer lands.

**Bring-your-own-key and bring-your-own-model.** Encryption keys managed in the customer's KMS; optional self-hosted model endpoints that route through your platform but never let plaintext data leave the customer VPC. The LLM gateway from §4 is what makes BYO-model practical.

**Audit and DLP.** An immutable audit log of who did what, plus an inline data-loss-prevention pass that redacts PII on the way in and on the way out. Subject-access-request tooling (find and delete all data for a given user) is a legal requirement under GDPR and several US state laws; build the data model so this is a SQL query, not a spelunking expedition.

**Compliance.** SOC 2 Type II is table stakes for B2B SaaS. ISO 27001, HIPAA, and FedRAMP open specific markets. None of these are technical projects in isolation — they are evidence-collection projects that the architecture has to support. The good news: if the previous five boxes are solid, the audits are mostly paperwork.

A practical sequencing tip: do not try to ship all six on day one. Build SSO, RBAC, and audit logging before the first paying customer; add data residency and BYO-key before the first regulated customer; pursue formal certifications when a deal demands them.

## A short note on what we deliberately skipped

This article does not include 600 lines of FastAPI scaffolding for a "complete RAG system." Those exist in dozens of repositories and they age badly. What ages well is the decision framework: which layer owns this concern, which workflow pattern fits this task, which lever moves cost the most for the least quality loss. The code is the easy part; the engineering judgement is the hard part, and it is what we have tried to compress here.

If you want concrete reference implementations, the official LangChain, LlamaIndex, and Anthropic cookbooks are kept current in a way that no static article can be. Skim them for code; come back here for the trade-offs.

## Closing

A production LLM application is a normal distributed system that calls a non-deterministic function. Most of the work is on the deterministic side: getting the right context to the model, routing the right request to the right model, and observing what comes back well enough to know when something has changed.

The patterns in this article are the ones that have survived contact with real traffic, real customers, and real procurement processes. They are not the only way to build, but they are a reasonable default, and most teams are better off starting from a reasonable default than reinventing one. Build the smallest workflow that works, instrument it from day one, and resist the urge to add complexity until the metrics tell you it is needed.

## References

- Lewis, P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. <https://arxiv.org/abs/2005.11401>
- Karpukhin, V. et al. *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020. <https://arxiv.org/abs/2004.04906>
- Liu, N. F. et al. *Lost in the Middle: How Language Models Use Long Contexts*. TACL 2024. <https://arxiv.org/abs/2307.03172>
- Schulhoff, S. et al. *The Prompt Report: A Systematic Survey of Prompting Techniques*. 2024. <https://arxiv.org/abs/2406.06608>
- Anthropic. *Building Effective Agents*. 2024. <https://www.anthropic.com/research/building-effective-agents>
- Frantar, E. et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023. <https://arxiv.org/abs/2210.17323>
- Sigelman, B. H. et al. *Dapper, a Large-Scale Distributed Systems Tracing Infrastructure*. Google Tech Report, 2010. <https://research.google/pubs/pub36356/>
- Beyer, B. et al. *Site Reliability Engineering*. O'Reilly / Google, 2016 — golden signals chapter. <https://sre.google/sre-book/monitoring-distributed-systems/>
- Gao, L. et al. *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. ACL 2023. <https://arxiv.org/abs/2212.10496>
- Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. <https://arxiv.org/abs/2106.09685>
