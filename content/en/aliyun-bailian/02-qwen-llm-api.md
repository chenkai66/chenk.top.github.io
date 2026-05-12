---
title: "Aliyun Bailian (2): The Qwen LLM API in Production"
date: 2026-02-26 09:00:00
tags:
  - Aliyun Bailian
  - Qwen
  - LLM
  - Function Calling
  - Streaming
categories: Aliyun Bailian
lang: en
mathjax: false
series: aliyun-bailian
series_title: "Aliyun Bailian Practical Guide"
series_order: 2
description: "Picking a Qwen variant by latency and cost, function calling done right, JSON mode without tears, and the enable_thinking + streaming requirement that the docs gloss over."
disableNunjucks: true
translationKey: "aliyun-bailian-2"
---

This article in the series covers most of the production wins. The other models are interesting, but the LLMs are what every product I've shipped on Bailian calls every minute of every day. The official Qwen API reference is dense and complete; this article is the readable companion that guides you through it.

![Aliyun Bailian (2): The Qwen LLM API in Production — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/02-qwen-llm-api/illustration_1.png)

## Pick the right Qwen variant for the workload

The Qwen family is large. Most teams overspend by defaulting to `qwen-max` everywhere. Most teams underspend on quality by defaulting to `qwen-turbo`. The right answer is "match variant to job":

![Qwen model family](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/02-qwen-llm-api/fig1_qwen_family.png)

My production rules of thumb:

- **`qwen-turbo`** — classification, intent detection, short summarization, anything you call >10× per user request. It's the cheapest sane Qwen and surprisingly good at extraction.
- **`qwen-plus`** — daily driver for chat, RAG synthesis, multi-step reasoning. The cost-vs-quality knee.
- **`qwen-max` / `qwen3-max`** — code review, complex reasoning, anything where being wrong costs more than being slow.
- **`qwen3-coder-plus`** — every code task. It is meaningfully better at code than general `qwen-plus` even at the same parameter scale.
- **`qwen3-vl-plus` / `qwen3-omni-flash`** — image / video / audio in. Article 3 is dedicated to this.

> **Tip:** A common mistake is using `qwen-max` for embedding-style classification. Don't. Use `qwen-turbo` with a tight system prompt and you'll cut cost 10× with no quality loss on tasks where you only need a label.

## What actually goes over the wire

Whether you use the OpenAI compat layer or DashScope native, the core of a chat-completion request remains the same: a model ID, a messages array, and a parameter block.

![Chat completion request flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/02-qwen-llm-api/fig2_request_flow.png)

The fields you'll touch most often:

- **`messages`** — an array of `{role, content}`. Role is `system` / `user` / `assistant` / `tool`. The official docs note that for multimodal models `content` can be an array of typed parts (text, image_url, input_audio, video_url) — see article 3.
- **`temperature`** — 0.0-2.0. I use 0.0 for extraction / classification, 0.2-0.4 for default chat, 0.7+ only for creative writing. The official docs default is around 0.7, which is too high for most agentic uses.
- **`top_p`** — leave it at default unless you know exactly why you want to change it. Tweaking both `temperature` and `top_p` at once is a recipe for confusion.
- **`max_tokens`** (compat) / **`parameters.max_tokens`** (native) — this is the *output* token cap, not total. Set it. Otherwise a runaway can cost you.
- **`stream`** — toggle SSE streaming. See below.
- **`response_format={"type": "json_object"}`** — JSON mode. Strongly recommended over "please return JSON" prompting.
- **`tools`** / **`tool_choice`** — function calling.

## Function calling: the round trip

The Qwen function-calling protocol is the OpenAI tool-calls protocol. Two LLM calls plus your code in the middle:

![Function calling round-trip](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/02-qwen-llm-api/fig3_function_calling.png)

A complete worked example — a tiny agent that can look up the weather:

```python
import json, os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

def call_weather(city):
    # Real impl: call your API. Stub here.
    return {"city": city, "temp_c": 22, "conditions": "sunny"}

messages = [{"role": "user", "content": "Should I bring an umbrella to Shanghai?"}]
resp = client.chat.completions.create(
    model="qwen-plus", messages=messages, tools=tools,
)
msg = resp.choices[0].message

if msg.tool_calls:
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = call_weather(**args)
        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
    final = client.chat.completions.create(model="qwen-plus", messages=messages)
    print(final.choices[0].message.content)
else:
    print(msg.content)
```

Three things that bite:

- **`messages.append(msg)` is required** between the first response and the tool result. The model needs to see its own tool_call message in the history, otherwise the second call returns a 400 about an "orphan tool result".
- **`tool_choice="auto"`** is the default. Force a specific tool with `tool_choice={"type": "function", "function": {"name": "..."}}` when you must — useful for the first call in a workflow.
- **`parallel_tool_calls=True`** is supported. Use it when you have independent tools — the model will return multiple `tool_calls` in one shot.

## JSON mode

For structured output, don't rely on prompting. Use:

```python
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "Return JSON: {\"sentiment\": \"positive|negative|neutral\"}"},
        {"role": "user", "content": "I love this product."},
    ],
    response_format={"type": "json_object"},
)
data = json.loads(resp.choices[0].message.content)
```

Two caveats from production:

- The model will sometimes wrap JSON in ```` ```json ```` fences anyway. Defensive parsing with `json.loads` after stripping fences is wise.
- For *structured* JSON (Pydantic schema) use the function-calling pattern instead. It's stricter and the failure modes are easier to debug.

## enable_thinking and the streaming trap

Qwen3 series models support **`enable_thinking=True`** — it asks the model to produce a reasoning chain before the final answer. Quality goes up, especially on reasoning-heavy tasks. **But you must use streaming.** Non-stream returns a 400.

![enable_thinking + streaming](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/02-qwen-llm-api/fig4_thinking_streaming.png)

Practical pattern — collect the reasoning into a side log and stream the answer to your UI:

```python
stream = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "If a clock loses 5 minutes a day, by how much will it be off after 10 days?"}],
    extra_body={"enable_thinking": True},
    stream=True,
)

reasoning, answer = [], []
for chunk in stream:
    delta = chunk.choices[0].delta
    # Qwen3 streams the reasoning chain in delta.reasoning_content
    rc = getattr(delta, "reasoning_content", None)
    if rc:  reasoning.append(rc)
    if delta.content: answer.append(delta.content)

print("ANSWER:", "".join(answer))
print("(reasoning hidden,", sum(len(r) for r in reasoning), "chars)")
```

I forward `reasoning` to my logging system and never to the user. Three reasons: (1) it leaks chain-of-thought IP if customers ever see it, (2) it confuses non-technical readers, (3) it doubles your visible response length.

## Async chat (rare but useful)

If you have a very long-running chat (e.g. a 30k-token RAG synthesis), you can submit it async with `X-DashScope-Async: enable` and poll, same pattern as Wanxiang. The Qwen API reference documents this under "Asynchronous calling". I use it for cron-batch summarization jobs that don't need an immediate user-facing response.

## Cost controls that actually work

- **Always set `max_tokens`.** Default cap of "model max" means a runaway loop costs you a fortune.
- **Use a workspace key per environment.** Set a hard daily budget on the prod key in the console under the workspace.
- **Log token counts.** `usage.prompt_tokens` and `usage.completion_tokens` are in every response. Aggregate them weekly and you'll spot the prompt that bloated by 3x without anyone noticing.
- **Cache identical prompts at your edge.** DashScope does not currently expose prompt caching the way Anthropic does — so cache yourself for high-volume identical-prefix patterns.

## Token counting: DashScope vs tiktoken, and the CJK bloat

The single biggest surprise for teams coming from the OpenAI ecosystem: **`tiktoken` lies about Qwen token counts**. The Qwen tokenizer is not BPE-compatible with `cl100k_base` or `o200k_base`. If you size your context budget with `tiktoken.encoding_for_model("gpt-4o")`, you'll be off by 20-40% on Chinese, 5-10% on English. I lost a Friday night to a RAG pipeline that was "definitely under the 32k context" by tiktoken count and was actually 41k by Qwen's count.

The right move is to use the official Qwen tokenizer locally:

```python
# pip install transformers tiktoken
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

def count_qwen_tokens(text: str) -> int:
    return len(tok.encode(text))

print(count_qwen_tokens("阿里云百炼 DashScope SDK"))   # 9
# tiktoken o200k_base on the same string gives 11
```

The tokenizer ships in the Hugging Face repo for any Qwen model — `Qwen2.5-7B`, `Qwen3-7B`, etc. share a tokenizer family, so loading any of them gives you a count that matches what DashScope will charge you within ±1 token. After the fact, `usage.prompt_tokens` and `usage.completion_tokens` in the response are authoritative; trust those over local estimates when you have them.

The CJK bloat problem is real and worth pricing in. A typical Chinese sentence runs about 1.5 tokens per character on Qwen. English runs about 0.25 tokens per character (1 token per ~4 chars). So a 1000-character Chinese RAG context costs you 1500 tokens; the same in English costs you 250. When you size context windows, **plan in tokens, not in characters**, and use the local tokenizer. I've watched a "use 100k chars of context" plan turn into "we need a 150k context model" exactly once.

## Streaming with backpressure: drain pattern, partial JSON

Naïve streaming code looks like the snippet from earlier — iterate the stream, append, done. That works for a CLI demo. In a production HTTP service you have two extra problems: backpressure (your downstream is slower than the model produces) and partial parsing (the user wants structured output but your buffer is mid-token).

**Backpressure**: when you forward stream chunks to a slow client (a mobile browser on 4G), the chunks pile up in your process memory until either you OOM or the upstream connection times out. The fix is to drain the upstream into a bounded queue and apply pushback to your client connection:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

async def relay(send_to_client):
    stream = await client.chat.completions.create(
        model="qwen-plus", messages=msgs, stream=True,
    )
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=32)

    async def producer():
        async for chunk in stream:
            d = chunk.choices[0].delta.content
            if d:
                await queue.put(d)        # blocks when queue full → backpressure
        await queue.put(None)

    async def consumer():
        while True:
            d = await queue.get()
            if d is None: break
            await send_to_client(d)        # if client is slow, queue fills, producer blocks

    await asyncio.gather(producer(), consumer())
```

The queue size (32 here) is the depth of in-flight buffer you allow. Smaller means more responsive backpressure but slightly choppier delivery. 32 is what I've found works for an SSE-to-WebSocket relay over the public internet.

**Partial JSON**: if your output is JSON and you want to render fields as they arrive (live-updating a form, for example), you can't just `json.loads` until the stream ends. The trick is a streaming JSON parser like `json-stream` or `partial_json_parser`:

```python
# pip install partial-json-parser
from partial_json_parser import loads as partial_loads

buffer = ""
last_render = None
for chunk in stream:
    buffer += chunk.choices[0].delta.content or ""
    try:
        partial = partial_loads(buffer)   # parses what it can, fills missing with None
        if partial != last_render:
            render_to_ui(partial)
            last_render = partial
    except Exception:
        continue   # not even a valid prefix yet, skip
```

This unlocks "the form is filling in front of the user" UX without waiting for the model to finish. I use this for the structured-extraction endpoints in our marketing tool — perceived latency drops from 4 seconds to under 500ms even though wall-clock is identical.

## Function calling deep dive: multi-round, parallel, the tool_choice="auto" trap

![Aliyun Bailian (2): The Qwen LLM API in Production — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/02-qwen-llm-api/illustration_2.png)

The basic round-trip from the original article handles the simple case. Real agents loop. The pattern is a `while` loop until the model stops emitting `tool_calls`:

```python
def run_agent(user_msg, tools, dispatch, max_rounds=8):
    messages = [{"role": "user", "content": user_msg}]
    for round_n in range(max_rounds):
        resp = client.chat.completions.create(
            model="qwen-plus", messages=messages, tools=tools,
            parallel_tool_calls=True,
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        # Parallel: dispatch all tool calls in one shot
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = dispatch(call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    raise RuntimeError("agent did not converge")
```

Three things the docs do not flag clearly:

**The `tool_choice="auto"` infinite-loop trap.** With `auto`, the model decides whether to call a tool or answer. On Qwen-Plus I have repeatedly seen it call `get_weather` four rounds in a row for the same city, each time deciding the previous answer wasn't enough. The fix is either (a) add a strict `max_rounds` cap (which I do), (b) after round 3 force `tool_choice="none"` so the model must answer, or (c) detect identical tool calls and short-circuit. Option (b) is what I use in production — once the agent has had 3 chances to tool-up, it gets one final answer-only round.

**Parallel tool calls return in one assistant message.** With `parallel_tool_calls=True`, `msg.tool_calls` is a list of multiple calls. You must append the *single* assistant message containing all of them, then append one `tool` message per call, then make the next request. If you append per-call tool messages without the assistant message between them, you get the orphan-tool-result 400.

**Tool argument schemas: keep them flat and required.** Qwen's tool calling is meaningfully less reliable on deeply nested arguments than GPT-4. A tool with `{type: object, properties: {filter: {type: object, properties: {date: {...}, region: {...}}}}}` will get the date right and forget the region about 15% of the time. Flatten to top-level required params and you get to >99% reliability. I learned this the hard way after a week of "the model is dumb today" complaints that were really "your schema is too nested".

## enable_thinking nuances: when CoT helps, when it hurts, what it costs

`enable_thinking=True` is sold as "free quality boost". It is not free, and it is not always a boost. After running it in production for six months on different workloads, my taxonomy:

**Where it helps:**
- Multi-step reasoning (math word problems, logic puzzles, code-execution traces).
- "Did the user request X *and* Y *but not* Z" classification with multiple constraints.
- Code review with multiple files in context.
- Anything where you'd want to write down intermediate steps yourself.

**Where it hurts (or is wasted):**
- Pure extraction ("pull these 5 fields from this text"). The reasoning chain just rephrases the input.
- Short factual lookups. The model thinks for 2 seconds about a one-word answer.
- High-temperature creative writing. Reasoning collapses style toward neutral.
- Tool-calling agents. The reasoning content interferes with the tool-call decision in subtle ways — I've seen calls drop the `tool_choice` signal entirely when thinking is on.

**Latency cost:** TTFT (time to first byte of the answer) goes from ~400ms to ~1.5-3 seconds because the reasoning chain has to finish before the answer streams. Total tokens roughly double — you pay for the reasoning content even though you don't show it to the user. For a chat UI the perceived "is it working" gap during the thinking pause is bad UX; I either show a "thinking…" spinner or stream the reasoning content into a collapsible side panel.

**How I decide:** if the task is the kind I'd want to write notes on before answering, I enable thinking. If I'd just blurt out the answer, I leave it off. For agentic loops I leave it off on every round except the final synthesis.

## Long context: cache hit rate and truncation strategy

Qwen-Plus has a 128k context window. Qwen-Max goes to 32k by default with 1M-token long-context variants. Just because the window is big doesn't mean you should fill it.

The implicit prompt cache I mentioned in chapter 1 has one critical property: **it caches by exact prefix match**. If your system prompt is identical for every user but the user message changes, the system-prompt prefix is cached. If you put dynamic data in the *middle* of the prompt (e.g. "today is {date}"), every variant breaks the cache. The fix is to keep all dynamic content at the *end* of the messages array — put dynamic dates / user IDs / timestamps in the user message, not in the system prompt.

I instrument cache hit rate per endpoint:

```python
total = sum(usage.prompt_tokens for usage in window)
cached = sum(getattr(usage, "cached_tokens", 0) for usage in window)
hit_rate = cached / total
print(f"cache hit rate over last {len(window)} calls: {hit_rate:.1%}")
```

A well-structured RAG endpoint with a stable system prompt hits 70-80%. An endpoint that interpolates request-specific metadata into the system prompt hits 0%. The bill difference is a factor of 2 on input tokens.

For truncation when you do exceed the window: the safe pattern is "preserve the system prompt and the most recent user/assistant pair, then sliding-window the middle". I keep the first system message and the last 6 messages verbatim, and summarize anything in between with a cheap `qwen-turbo` call when the conversation crosses a threshold. The summary goes back into the messages array as a synthetic system message (`"role": "system", "content": "Earlier in this conversation: ..."`). Quality loss is small for chat-style workloads, dramatic for code-context workloads where you can't lossy-compress the file content — for those, prefer a longer-context model over summarization.


## What's next

Article 3 is **Qwen-Omni** — the multimodal sibling. The big differences are: streaming is *required* (not optional), the content array gets typed parts for image / audio / video, and you have to think about pixel budgets and frame rates. It's the highest-leverage capability in Bailian if your product touches non-text content.
