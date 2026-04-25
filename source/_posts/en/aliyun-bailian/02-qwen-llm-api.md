---
title: "Aliyun Bailian (2): The Qwen LLM API in Production"
date: 2026-02-26 09:00:00
tags:
  - Aliyun Bailian
  - DashScope
  - LLM
  - Qwen
categories: Aliyun Bailian
lang: en
mathjax: false
series: aliyun-bailian
series_title: "Aliyun Bailian Practical Guide"
series_order: 2
description: "Picking the right Qwen model, the enable_thinking parameter that will burn you in non-streaming mode, function calling, JSON mode, and the error patterns I actually see in production."
disableNunjucks: true
---

A user pasted a 30-page contract into our app and asked for a structured summary. The first version called `qwen-turbo` and produced confident nonsense. The second called `qwen-max` and cost 30x more for marginal accuracy. The third called `qwen3-max` with `enable_thinking=True`, returned in 12 seconds with the right answer at half the price of `qwen-max`, and would have worked the first time if I had known one rule about how Qwen3 handles streaming. This article is about the dozen-or-so things in the Qwen LLM family that make the difference between "it works on my laptop" and "it works on Black Friday."

## Pick the right model first, optimize prompts second

Bailian's pricing page lists about thirty Qwen variants. In a year of production work I have used these five and that is enough:

| model_id | When to use | Approx input/output ¥/1M | Latency for 500 tokens out |
|---|---|---|---|
| `qwen-turbo` | High-throughput classification, simple extraction | 0.3 / 0.6 | ~600ms |
| `qwen-plus` | The default — chat, summarization, light reasoning | 0.8 / 2 | ~1.2s |
| `qwen-max` | Hardest reasoning, long contracts, when you cannot afford to be wrong | 2.4 / 9.6 | ~3s |
| `qwen3-max` | New default for hard reasoning; cheaper than qwen-max with `thinking` | 2.0 / 6 | ~3-12s with thinking |
| `qwen3-coder-plus` | Code generation, anything with diff/patch/AST | 1.0 / 4 | ~2s |

The rest of the catalog is either deprecated, smaller (covered by Turbo), or specialized in ways most apps never need. **Default to `qwen-plus`.** Move up to `qwen3-max` only when you have an eval that proves Plus is not enough. Move down to `qwen-turbo` only when you have measured the cost actually mattering.

Context length is generous across the family — 128K tokens for plus/max, 1M for `qwen-turbo-longcontext` if you need it — but long context is expensive linearly and slow super-linearly. If you are stuffing 80K tokens of RAG context into every call, you are doing retrieval wrong, not picking the wrong model.

## The OpenAI-compatible client is your default

Everything in this article uses the OpenAI SDK against the compat endpoint unless I call out otherwise.

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

Hold this `client` for the lifetime of the process. The SDK is thread-safe and pools connections; constructing it per call adds 50-100ms of TLS handshake.

## Streaming basics

For anything user-facing, stream. Time-to-first-token is what users perceive as "fast"; total latency is what your dashboards measure. They are different problems.

```python
stream = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Write a 4-line haiku about TLS."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

Two things bite people:

1. The last chunk has `delta.content == None` and a `finish_reason`. Always check `if delta:`.
2. If you want token counts in streaming mode, pass `stream_options={"include_usage": True}`. Without it the final chunk has no `usage` field and you will not know what you spent.

## The enable_thinking trap (Qwen3 family)

This is the bug I cost myself half a day on. Qwen3 models (`qwen3-max`, `qwen3-coder-plus`, etc.) have a `enable_thinking` parameter that turns on a chain-of-thought reasoning pass before the final answer. It is the reason `qwen3-max` can hit `qwen-max` accuracy at lower price — but there is a hard rule:

> **`enable_thinking=True` requires `stream=True`. In non-streaming mode, the call fails.**

Why: the model emits reasoning tokens (often thousands of them) wrapped in a `<thinking>` block before the answer. The compat layer expects you to consume them progressively. In non-streaming mode the buffer behavior isn't defined and the API rejects the request rather than silently truncate.

Wrong:

```python
# This will return a 400 with a message about thinking + stream
resp = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "Solve: ..."}],
    extra_body={"enable_thinking": True},
)
```

Right:

```python
stream = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "Solve: ..."}],
    stream=True,
    stream_options={"include_usage": True},
    extra_body={"enable_thinking": True},
)

answer_parts = []
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    # The compat layer surfaces reasoning under delta.reasoning_content
    if getattr(delta, "reasoning_content", None):
        # Optional: log it, render in a "thinking" UI panel, or drop it
        pass
    if delta.content:
        answer_parts.append(delta.content)

answer = "".join(answer_parts)
```

If you have a non-streaming code path you cannot easily change — for example, a sync queue worker — set `enable_thinking=False` explicitly. The default in the compat shim is "off" but I have seen it differ between SDK versions, so be explicit.

> **Tip:** When you see Qwen3 quality scores in benchmarks, they almost always assume `thinking` is on. If you A/B and find Qwen3 underperforming Qwen-2.5, check which mode you are actually in. The difference on hard reasoning tasks is huge.

## Function calling (tool use)

Wire-compatible with OpenAI's tool-call format. Define tools, the model decides whether to call them, you execute them and feed the result back.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the status of an order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, e.g. 'ORD-123'"},
                },
                "required": ["order_id"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "Where is order ORD-7781?"},
]

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
)

choice = resp.choices[0]
if choice.message.tool_calls:
    call = choice.message.tool_calls[0]
    # Execute it for real
    result = {"order_id": "ORD-7781", "status": "shipped", "eta": "2026-04-26"}

    messages.append(choice.message)
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": str(result),
    })

    final = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
    )
    print(final.choices[0].message.content)
```

Behavioral notes from production:

- Qwen models are eager to call tools — sometimes they will call one when a plain answer would do. If you don't want that on a given turn, pass `tool_choice="none"`.
- Parallel tool calls are supported on `qwen-plus` and up. The response will have multiple entries in `tool_calls`. Don't assume length 1.
- If your tool's JSON schema is sloppy (missing `description`, vague enum), Qwen's argument-filling gets sloppier. Treat schemas as part of your prompt.

## JSON mode

For structured extraction, ask for JSON and validate it.

```python
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "Reply with a JSON object only. No prose."},
        {"role": "user", "content": "Extract product name and price from: 'iPhone 17 Pro at ¥9999'"},
    ],
    response_format={"type": "json_object"},
)

import json
data = json.loads(resp.choices[0].message.content)
```

`response_format={"type": "json_object"}` works on `qwen-plus`, `qwen-max`, `qwen3-max`. It forces well-formed JSON but does *not* enforce a schema. For schema-strict output, either use function calling (the schema lives on the tool definition) or validate with `pydantic` after.

In my code I always run a `pydantic.parse_obj` after JSON mode — once every few thousand calls a model will produce JSON that satisfies the parser but has the wrong field names. Catch it loud, retry once, then alert.

## Error handling that actually works

The errors you will see, in order of frequency:

| HTTP / type | Meaning | What to do |
|---|---|---|
| `429 Throttling.RateQuota` | You hit RPM/TPM limit | Exponential backoff with jitter, then look at quota |
| `400 InvalidParameter` | Schema-level bug — bad role, bad model_id, conflict like thinking+nonstreaming | Fix the request; do not retry |
| `400 DataInspectionFailed` | Content moderation tripped on input or output | Sanitize input or rephrase prompt; do not retry the same payload |
| `500 InternalError` | Their problem | Retry with backoff, max 3 |
| `503 ModelOverloaded` | Spike on this specific model | Retry with backoff; consider failover model |
| Timeout | Long output, network glitch, or long thinking | Increase client timeout to 120s for thinking models |

Production-grade wrapper:

```python
import time
import random
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

def chat_with_retry(client, **kwargs):
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            sleep = (2 ** attempt) + random.random()
            time.sleep(sleep)
        except APITimeoutError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(1 + attempt)
        except APIError as e:
            # 4xx that aren't rate-limit are not retryable
            if 400 <= e.status_code < 500 and e.status_code != 429:
                raise
            time.sleep(1 + attempt)
```

Set the client-level timeout high enough for the worst case:

```python
client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=120.0,  # default is 600 in newer SDKs but worth being explicit
)
```

## When to use the DashScope native client instead

For pure chat, the compat client is fine. Switch to native (`dashscope.Generation.call`) when:

- You need parameters the compat layer doesn't expose — e.g., `search_options` for the built-in web search retrieval, or `incremental_output=False` semantics for partial results.
- You want a single SDK across LLM, Wanxiang, and TTS (saves your dependency tree one entry).
- You're calling from a region where the native endpoint has lower latency.

Native streaming feels uglier but is no harder:

```python
import dashscope
from dashscope import Generation

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

responses = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
    result_format="message",
    stream=True,
    incremental_output=True,
)

for resp in responses:
    chunk = resp.output.choices[0].message.content
    print(chunk, end="", flush=True)
```

`incremental_output=True` is the native equivalent of "give me deltas, not cumulative text" — without it, every chunk contains everything-so-far and your UI will jitter.

## A small checklist before you ship

- [ ] Pinned the dated alias of your model (`qwen-plus-2025-09-11`, not `qwen-plus`) for anything customer-facing.
- [ ] Set a per-key budget alert in the Bailian console.
- [ ] Wrap calls with the retry helper above; route non-retryable errors to a dead-letter log.
- [ ] If using Qwen3 with thinking, you are on a streaming code path.
- [ ] Logging the `usage` field every call so you can rebuild cost-per-request from logs.
- [ ] Have a fallback model: if `qwen-max` 503s, you fail over to `qwen-plus` rather than to a 500.

That last one matters more than people think. Bailian regional outages are rare but have happened. Your users do not care.

Next up: Qwen-Omni. The streaming requirement is even stricter, the request shape is different, and the use cases — video understanding in particular — are some of the most interesting things you can build on this platform.
