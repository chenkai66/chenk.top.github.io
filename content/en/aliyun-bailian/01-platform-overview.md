---
title: "Aliyun Bailian (1): Platform Overview and First Request"
date: 2026-02-25 09:00:00
tags:
  - Aliyun Bailian
  - DashScope
  - LLM
categories: Aliyun Bailian
lang: en
mathjax: false
series: aliyun-bailian
series_title: "Aliyun Bailian Practical Guide"
series_order: 1
series_total: 5
description: "A practitioner's tour of Alibaba Cloud Bailian (DashScope) — what's actually in the model catalog, the two endpoint flavors, the async task pattern, and a working sample request to ground the rest of the series."
disableNunjucks: true
translationKey: "aliyun-bailian-1"
---

If you ship anything that touches Chinese-language users, sooner or later you will end up calling a Bailian model. Qwen-Max is the cheapest sane way to get GPT-4-class Chinese understanding, the Wanxiang video models are the only production-grade text-to-video API I can buy with a Chinese invoice, and Qwen-TTS-Flash is the only TTS that handles Cantonese and Sichuanese without sounding like a customs announcement. After about a year of running these in production for an AI-marketing platform, this series is what I wish someone had handed me on day one.

This first article is the lay of the land: what Bailian actually is, the model families you'll touch, the two endpoint flavors, and a hello-world request in both styles so the rest of the series doesn't have to re-explain it.

![Aliyun Bailian (1): Platform Overview and First Request — Chapter overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/01-platform-overview/illustration_1.png)

---

## What is Bailian, what is DashScope?

The naming is genuinely confusing because Alibaba renamed things mid-flight. The official "DashScope" docs frame it from the API side; the "Bailian" docs frame it from the console side. Same product, two names.

![Bailian (console) vs DashScope (API)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/01-platform-overview/fig1_bailian_dashscope_split.png)

You will see both names in docs, sometimes in the same paragraph. Treat them as console-vs-API. When someone says "deploy a Bailian app" they mean the console; when they say "DashScope error", they mean the API returned a non-200.

## The model catalog you actually care about

Bailian hosts a hundred-something models. In a year of production work I have only ever paid for these:

| Family | Representative model_id | Use it for |
|---|---|---|
| Qwen LLM (text) | `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwen3-max`, `qwen3-coder-plus` | Chat, reasoning, tool use, code |
| Qwen-Omni (multimodal) | `qwen3-omni-flash`, `qwen3.5-omni-plus` | Video / audio / image understanding |
| Qwen-VL (visual) | `qwen3-vl-plus` | Image-only understanding (cheaper than Omni) |
| Wanxiang (video) | `wan2.5-t2v-plus`, `wan2.5-i2v-plus` | Text-to-video, image-to-video |
| Qwen-TTS | `qwen3-tts-flash` | Speech synthesis, 40+ voices |
| Embeddings | `text-embedding-v3`, `text-embedding-v4` | Vector search |

Anything not in that table is either deprecated, a thin variant, or a research preview. If you stick to these you will not get burned by a model going EOL six weeks after you launch.

## Pricing model in one paragraph

Token-metered for LLMs (input vs output priced separately, output is 2-4x more expensive), per-second-of-audio for TTS, per-second-of-video for Wanxiang, per-call for embeddings. There is a free tier per model — usually 1M tokens or 100 generations — that resets when a new model launches, which means you can prototype almost anything for free if you don't mind hopping versions. Production traffic should go through a dedicated API key with a budget alert; I have eaten a 4-figure bill exactly once because someone left a debug loop on overnight.

## API keys: do not commit them

Get a key from the console under **API-KEY** in the left nav. There is one default workspace key plus per-workspace keys; for any production project, make a workspace key so you can rotate it without nuking dev. Then:

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

Every snippet in this series reads `os.environ['DASHSCOPE_API_KEY']`. Do not hardcode keys, do not commit `.env` files, and please put the key behind your secrets manager in production. The DashScope team does revoke leaked keys, but only after they appear in a public crawl, which is too late.

## The two endpoints: OpenAI-compatible vs DashScope native

This is the single most important fact in this article. Per the Qwen API reference, **every Bailian text/multimodal model is reachable through two different HTTP surfaces.**

![Two HTTP surfaces, one model catalog](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/01-platform-overview/fig2_two_endpoints.png)

### OpenAI-compatible endpoint

Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1` (mainland), or `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (Singapore).

This speaks OpenAI's wire protocol. You point the official `openai` Python SDK at it and 95% of your existing OpenAI code works untouched. This is what I use by default.

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a senior backend engineer."},
        {"role": "user", "content": "Explain idempotency keys in two sentences."},
    ],
)
print(resp.choices[0].message.content)
```

### DashScope native endpoint

Base URL: `https://dashscope.aliyuncs.com/api/v1/...` — different paths per modality (`/services/aigc/text-generation/generation`, `/services/aigc/multimodal-generation/generation`, `/services/aigc/text2video/...`).

This is Alibaba's own wire protocol — different request shape, different field names (`input.messages` instead of `messages`, `parameters` block, etc.). You use the `dashscope` SDK, or raw HTTP.

```python
import os
import dashscope

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

resp = dashscope.Generation.call(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a senior backend engineer."},
        {"role": "user", "content": "Explain idempotency keys in two sentences."},
    ],
    result_format="message",
)
print(resp.output.choices[0].message.content)
```

### When to use which

My rule, refined the hard way:

- **OpenAI-compatible**: default for plain chat, function calling, JSON mode, streaming. Lets you A/B against GPT-4 by changing one line.
- **DashScope native**: required for Wanxiang video, required for Qwen-TTS, recommended for Qwen-Omni multimodal (the compat layer drops some video parameters), required if you want async task patterns, required for the latest features that haven't been backported to the compat shim.

A common trap: someone reads "OpenAI-compatible" and assumes *all* models work that way. They don't. `wan2.5-t2v-plus` is native-only. `qwen3-tts-flash` is native-only. Article 4 and 5 will hammer this point.

> **Tip:** When you hit a 400 with a message like `parameter X is not supported`, your first move should be to check whether you're calling a native-only model through the compat endpoint. About half the "Bailian is broken" tickets I've debugged were exactly this.

## Three concepts that recur everywhere

### model_id

Every call is keyed by a string like `qwen-plus` or `wan2.5-t2v-plus`. There is no version number — Alibaba ships new weights under the same id and tells you in the changelog. If you need pinning, use the dated alias (`qwen-plus-2025-09-11`) listed in the model card. For anything customer-facing, **pin the date**; I have seen the unversioned alias change tone overnight.

### Async tasks

Anything that takes more than ~30 seconds (video gen, large-batch embedding, long-form TTS) is async. The pattern is always:

![Async task pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/01-platform-overview/fig3_async_task_pattern.png)

1. POST to the create endpoint with header `X-DashScope-Async: enable`.
2. Get back a `task_id`.
3. Poll `GET /api/v1/tasks/{task_id}` until status is `SUCCEEDED` or `FAILED`.
4. Download the output URL within 24 hours — they expire.

Article 4 has a complete polling implementation with backoff.

### Streaming

LLMs and Qwen-Omni support SSE streaming. For Qwen3 with `enable_thinking=True`, streaming is **mandatory** — non-streaming will reject the call. For Qwen-Omni, streaming is mandatory *period* (more on this in article 3). Get comfortable with `stream=True` early; you will use it more than you expect.

## A complete first request

![Aliyun Bailian (1): Platform Overview and First Request — Chapter summary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/01-platform-overview/illustration_2.png)

Save this as `hello_bailian.py` and run it. If it prints a sentence, your account, key, and network are all good and you can move on to article 2.

```python
import os
from openai import OpenAI

def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY in your environment first.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": "In one paragraph, what is Bailian?"},
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()

if __name__ == "__main__":
    main()
```

If you want to confirm the native endpoint also works, swap in the `dashscope` snippet from earlier. Both should succeed; both will be billed identically.

## Console vs SDK vs OpenAI-compatible — when to use each

There are actually three places you can drive Bailian from, and the choice matters more than the docs make it sound. Spending two months mixing them in the same product taught me this the slow way.

The **console** at `bailian.console.aliyun.com` is for two things: provisioning (creating workspaces, API keys, RAM grants, applications) and diagnostics (looking at the per-request log to find out why a model returned garbage). It is not for runtime traffic. The "playground" inside the console is convenient for prompt iteration but it ignores some of the parameters your SDK call will use, so a prompt that works in the playground can still misbehave when you actually ship it. I treat the playground as "the model is reachable", not "this prompt is correct".

The **DashScope native SDK** (`pip install dashscope`) is the right choice when you need anything Alibaba-specific: async tasks, video / TTS / Wanxiang, the latest model parameters, the workspace-id header for cross-workspace billing, or the `X-DashScope-DataInspection` debug header. The native SDK exposes `parameters` blocks with fields the compat shim drops (`incremental_output`, `seed` for image gen, `enable_search` for the web-grounded models). It is also the only path for batch inference and for the `RemoteService` model deployments.

The **OpenAI-compatible endpoint** is the right choice when your code is written against the OpenAI SDK and you want to A/B Bailian against OpenAI by changing one line. The compat layer is a thin translator. It speaks `chat.completions.create`, `embeddings.create`, supports `tools` / `tool_choice` / `response_format` / `stream`, and accepts `extra_body` for the few Qwen-specific knobs (`enable_thinking`, `enable_search`). It does not speak Wanxiang, TTS, or any async pattern. If you need those, you must drop down to the native SDK for that one call — nothing stops you from using both clients in the same app.

My rule of thumb after a year:

- New project, mostly LLM, want provider-agnostic code → OpenAI SDK pointed at the compat endpoint.
- Mixed bag (LLM + Omni + video + TTS) → native SDK for everything except the chat-completion path, which I keep on the compat endpoint to keep the messaging code portable.
- Production debugging → native SDK exclusively, because the error envelope is richer (`code` + `message` + `request_id`, where compat gives you only the OpenAI-shaped error).

## Region routing and RAM scoping you actually have to know

Bailian has two regions that matter: **Beijing** and **Hangzhou** (with a Singapore "international" endpoint that is functionally a third region). The default `dashscope.aliyuncs.com` resolves to whichever region your account was originally provisioned in. This is the source of more "but it worked yesterday" reports than any other single thing.

Concretely, the trap is that some models are region-pinned. `wan2.5-t2v-plus` lives in the Beijing pool. `qwen3-omni-flash` is multi-region. `text-embedding-v4` rolled out to Hangzhou first and Beijing six weeks later. If your account routes to a region the model is not yet in, you get a perfectly valid 404 with a message like `Model not exist`, which you will spend an hour trying to fix in the prompt before realizing it is a routing issue.

Two ways to resolve this:

```python
# Force a region by hitting the regional URL directly
client = OpenAI(
    api_key=key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # mainland default
    # base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Singapore
)

# Or with the native SDK, set the workspace id header — workspaces are region-scoped
import dashscope
dashscope.api_key = key
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
```

For RAM, the gotcha is that the API key is tied to a workspace, and the workspace lives under a main account. A RAM sub-account does not automatically inherit access to the parent's Bailian workspaces. You have to either grant `AliyunBailianFullAccess` (lazy) or write a custom policy scoped to the workspace ARN. The policy resource format is `acs:bailian:*:*:workspace/<workspace_id>` — the workspace id is in the URL bar of the console, not in any obvious "details" page. RAM users who can hit the console but get `Forbidden.RAM` on API calls are almost always missing this policy.

For production I run a dedicated RAM sub-account per environment (dev / staging / prod), each with one workspace, each with one rotating API key managed by a secrets store. Cross-environment leakage is then an explicit policy action, not an accident.

## The quota reality nobody quotes you upfront

The marketing page says "high concurrency, low latency". The actual numbers, as of my last quota audit, are very specific and very low for free-tier accounts. The three knobs are:

- **RPM** (requests per minute) — how many requests you can fire in any 60-second window.
- **TPM** (tokens per minute) — total in-plus-out tokens metered.
- **TPD** (tokens per day) — daily ceiling, resets at 00:00 Beijing time.

Defaults I have measured for a fresh workspace, no quota application:

| Model | RPM | TPM | Concurrent tasks (async) |
|---|---|---|---|
| `qwen-turbo` | 500 | 500K | n/a |
| `qwen-plus` | 200 | 200K | n/a |
| `qwen-max` | 60 | 100K | n/a |
| `qwen3-omni-flash` | 60 | 100K | n/a |
| `wan2.5-t2v-plus` | n/a | n/a | 5 |
| `qwen3-tts-flash` | 100 | 50K | n/a |

These are not contractual; Alibaba revises them quietly. The pattern that matters: free-tier defaults are fine for development and a 100-DAU prototype, and they will throttle the moment a single ad tweet sends real traffic. The error you get is `Throttling.RateQuota` with HTTP 429 — your retry wrapper should treat 429 as backoff-and-retry, not as fatal.

To raise the quota, go to the console under **API-KEY → 限流配置** (rate config) and submit a justification. Approval is usually under 24 hours for reasonable asks (10× current limit). Asking for 100× of `qwen-max` without revenue numbers gets denied — bring your projected QPS, your average prompt length, and your business case. I have done this four times and only been refused once.

## Pricing per million tokens — the version that's actually current

The pricing page restructures every quarter and the numbers below will drift, but the *ratios* are stable enough to plan with. As of my last audit (2026-04), in RMB per million tokens:

| Model | Input | Output | Cached input | Notes |
|---|---|---|---|---|
| `qwen-turbo` | 0.3 | 0.6 | 0.15 | Cheap workhorse for classification |
| `qwen-plus` | 0.8 | 2.0 | 0.4 | Daily driver |
| `qwen-max` | 20 | 60 | 10 | Reasoning, code review |
| `qwen3-max` | 24 | 96 | 12 | Latest, more expensive than qwen-max |
| `qwen3-coder-plus` | 4 | 16 | 2 | Code-specialist, mid-tier price |
| `qwen3-omni-flash` | 2.5 (text) / 12 (vision) | 5 | n/a | Vision tokens count separately |
| `text-embedding-v4` | 0.7 (per million tokens) | n/a | n/a | Per-call billing |

Two things to internalize:

- **Output is 2-4× input**. A function-calling agent that loops 3 times will spend most of its bill on the assistant's output tokens, not the user prompt. Trim system prompts last, output first.
- **Cached input is half-price**. Bailian rolled out implicit prompt caching in late 2025 — same prompt prefix sent within ~5 minutes hits the cache. You don't have to do anything to opt in, but you do see the savings only on `usage.cached_tokens` if you log it. For a RAG app with a long fixed system prompt, cache hit rate is typically 60-80% and the bill drops accordingly.

For Wanxiang (per second of video) and Qwen-TTS (per second of audio), the right unit is per-finished-asset, not per-token. A 720p 5-second clip is around 1.5 RMB; a 60-second TTS narration is around 0.6 RMB. Both are cheap enough that the bottleneck is human review, not API spend.


## What's Next

Article 2 is a deep dive into the Qwen LLM family — model selection by latency and cost, function calling, JSON mode, and the `enable_thinking` parameter that has personally cost me about four hours of debugging. Article 3 covers Qwen-Omni's streaming requirement and a real video-understanding example. Article 4 is the Wanxiang video pipeline end-to-end, and article 5 is Qwen-TTS for multilingual narration.

If you only read one of them, read article 2 — it has the highest density of "the docs do not tell you this" content.
