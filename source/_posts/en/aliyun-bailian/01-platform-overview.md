---
title: "Aliyun Bailian (1): Platform Overview and First Request"
date: 2026-04-20 09:00:00
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
description: "A practitioner's tour of Alibaba Cloud Bailian (DashScope) — what's actually in the model catalog, the two endpoints you'll use, and a working sample request to ground the rest of the series."
disableNunjucks: true
---

If you ship anything that touches Chinese-language users, sooner or later you will end up calling a Bailian model. Qwen-Max is the cheapest sane way to get GPT-4-class Chinese understanding, the Wanxiang video models are the only production-grade text-to-video API I can buy with a Chinese invoice, and Qwen-TTS-Flash is the only TTS that handles Cantonese and Sichuanese without sounding like a customs announcement. After about a year of running these in production for an AI-marketing platform, this series is what I wish someone had handed me on day one.

This first article is the lay of the land: what Bailian actually is, the model families you'll touch, the two endpoint flavors, and a hello-world request in both styles so the rest of the series doesn't have to re-explain it.

## What is Bailian, what is DashScope?

The naming is genuinely confusing because Alibaba renamed things mid-flight. Here is the truth:

- **Bailian (百炼)** is the **product**: the console at `bailian.console.aliyun.com` where you manage API keys, deploy fine-tuned models, build RAG apps, and look at billing.
- **DashScope** is the **API surface** behind it. Every HTTP call hits `dashscope.aliyuncs.com`. The Python SDK is literally `pip install dashscope`.

You will see both names in docs, sometimes in the same paragraph. Treat them as console-vs-API. When someone says "deploy a Bailian app" they mean the console; when they say "DashScope error", they mean the API returned a non-200.

## The model catalog you actually care about

Bailian hosts a hundred-something models. In a year of production work I have only ever paid for these:

| Family | Representative model_id | Use it for |
|---|---|---|
| Qwen LLM (text) | `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwen3-max`, `qwen3-coder-plus` | Chat, reasoning, tool use, code |
| Qwen-Omni (multimodal) | `qwen3-omni-flash`, `qwen3.5-omni-plus` | Video / audio / image understanding |
| Wanxiang (video) | `wan2.5-t2v-plus`, `wan2.5-i2v-plus` | Text-to-video, image-to-video |
| Qwen-TTS | `qwen3-tts-flash` | Speech synthesis, 40+ voices |
| Embeddings | `text-embedding-v3`, `text-embedding-v4` | Vector search |
| OpenSearch | (separate product) | Hybrid retrieval, web search |

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

This is the single most important fact in this article. **Every Bailian text/multimodal model is reachable through two different HTTP surfaces.**

### OpenAI-compatible endpoint

Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`

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

Base URL: `https://dashscope.aliyuncs.com/api/v1/`

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

1. POST to the create endpoint with header `X-DashScope-Async: enable`.
2. Get back a `task_id`.
3. Poll `GET /api/v1/tasks/{task_id}` until status is `SUCCEEDED` or `FAILED`.
4. Download the output URL within 24 hours — they expire.

Article 4 has a complete polling implementation with backoff.

### Streaming

LLMs and Qwen-Omni support SSE streaming. For Qwen3 with `enable_thinking=True`, streaming is **mandatory** — non-streaming will reject the call. For Qwen-Omni, streaming is mandatory *period* (more on this in article 3). Get comfortable with `stream=True` early; you will use it more than you expect.

## A complete first request

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

## What's next

Article 2 is a deep dive into the Qwen LLM family — model selection by latency and cost, function calling, JSON mode, and the `enable_thinking` parameter that has personally cost me about four hours of debugging. Article 3 covers Qwen-Omni's streaming requirement and a real video-understanding example. Article 4 is the Wanxiang video pipeline end-to-end, and article 5 is Qwen-TTS for multilingual narration.

If you only read one of them, read article 2 — it has the highest density of "the docs do not tell you this" content.
