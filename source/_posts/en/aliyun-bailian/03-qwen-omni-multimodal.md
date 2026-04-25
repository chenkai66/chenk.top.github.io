---
title: "Aliyun Bailian (3): Qwen-Omni for Video, Audio, and Image Understanding"
date: 2026-02-27 09:00:00
tags:
  - Aliyun Bailian
  - DashScope
  - Multimodal
  - Qwen-Omni
categories: Aliyun Bailian
lang: en
mathjax: false
series: aliyun-bailian
series_title: "Aliyun Bailian Practical Guide"
series_order: 3
description: "Calling qwen3-omni-flash and qwen3.5-omni-plus on real videos, the mandatory streaming + include_usage rule that the docs bury, and a complete example that pulls structured fields out of a 2-minute product demo."
disableNunjucks: true
---

A merchant uploads a 90-second product video to our platform and expects the system to write a Chinese product description, an English ad headline, and three TikTok hooks. The old pipeline transcribed audio with one API, captioned frames with a second, then asked an LLM to fuse them. Three round trips, two reconciliation bugs a week, and the model never knew that the speaker was holding a different SKU than the one on the box. Qwen-Omni does the whole thing in one call. It is also the most fragile API in this series if you call it the way you'd call a normal LLM. This article walks through the rules.

## Two models, pick by latency budget

| model_id | Strength | Per-call latency for ~1min video | Notes |
|---|---|---|---|
| `qwen3-omni-flash` | Fast, cheap, good enough for most extraction | 8-15s | My default |
| `qwen3.5-omni-plus` | Better fine-grained understanding, longer reasoning | 15-30s | When extraction quality matters more than throughput |

Both accept video, audio, image, and text in the same `messages` array. Both have the same gotchas. I will use `qwen3-omni-flash` in all the examples; switching is one string change.

## The mandatory streaming rule

Read this paragraph twice.

> **Qwen-Omni REQUIRES `stream=True` AND `stream_options={"include_usage": True}`. Non-streaming calls error out. Calls without `include_usage` also error out.**

The first time I hit this I assumed the second requirement was a doc typo. It isn't. The model emits perception tokens (audio embeddings, frame embeddings) interleaved with text tokens, and the server-side accounting needs the usage hook to know when to close the stream. Without `include_usage`, you get a 400 with a vague "stream_options required" message.

This means there is no "synchronous Qwen-Omni" code path. Every call is a streaming call. Plan your worker pool around that.

## The endpoint and the request shape

Qwen-Omni lives at the multimodal generation endpoint:

```
POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
```

You can use the OpenAI-compatible endpoint for it too, and that's what I do — the request shape is just OpenAI's content-block format with `video_url`, `audio_url`, and `image_url` types.

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

stream = client.chat.completions.create(
    model="qwen3-omni-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://example.oss-cn-beijing.aliyuncs.com/demo.mp4"}},
                {"type": "text", "text": "Describe what the speaker is holding."},
            ],
        }
    ],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if not chunk.choices:
        # The usage-only final chunk has empty choices — skip it
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

Two patterns to remember:

1. **`content` is a list of typed blocks**, not a string. Mix and match `text`, `image_url`, `video_url`, `audio_url`. Order matters — put the media first if you want the question to refer back to it.
2. **The final chunk has an empty `choices` array** because it carries `usage` only. The `if not chunk.choices` skip is mandatory or you get an `IndexError`.

## How to host the media

Qwen-Omni accepts three sources, in order of preference:

| Source | Limit | Notes |
|---|---|---|
| HTTPS URL (e.g. OSS) | None practical | Best. Use OSS with a short-lived signed URL. |
| OSS internal URL | None practical | Lowest latency if your code runs in the same Alibaba region. |
| base64-encoded data URL | ~10MB request body | Painful for video. Use only for small images. |

For a typical 90-second 720p video that is 30-60MB, you must use a URL. I host every media asset on OSS in `cn-beijing` with a 15-minute signed URL, and pass the signed URL straight into the `video_url` block. If your storage is somewhere else, that works too — the model fetches it server-side, so you need a publicly-resolvable URL.

A common mistake: putting an OSS URL with `Content-Disposition: attachment` headers. The model's fetcher respects them and won't decode the media. Set `Content-Type: video/mp4` (or whatever) and don't set Content-Disposition.

## A complete real example: extract structured info from a product demo

Here is the pipeline that replaced our three-API mess. Single call, structured output, full code.

```python
import os
import json
from openai import OpenAI

PROMPT = """You are watching a product demo video. Extract:
1. product_name (string)
2. key_features (list of 3 strings, in the speaker's language)
3. target_audience (string)
4. tone (one of: energetic, calm, technical, lifestyle)
5. cta_suggestions (list of 3 short call-to-action lines, in English)

Reply with a single JSON object. No prose."""

def extract_product_info(video_url: str) -> dict:
    client = OpenAI(
        api_key=os.environ['DASHSCOPE_API_KEY'],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=90.0,
    )

    stream = client.chat.completions.create(
        model="qwen3-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    parts = []
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            parts.append(delta.content)

    raw = "".join(parts).strip()
    # The model sometimes wraps JSON in ```json fences
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()

    data = json.loads(raw)
    data["_usage"] = usage.model_dump() if usage else None
    return data

if __name__ == "__main__":
    info = extract_product_info("https://your-oss.oss-cn-beijing.aliyuncs.com/demo.mp4?Expires=...&Signature=...")
    print(json.dumps(info, ensure_ascii=False, indent=2))
```

Things this code does that mine didn't on the first attempt:

- Buffers the whole stream before parsing JSON. You cannot incrementally `json.loads` a partial document.
- Strips the ```json fence the model sometimes adds even when you ask for "no prose."
- Captures `usage` from the final chunk so you can attribute cost per video.
- Uses a 90-second client timeout. The default 60s is sometimes too tight for `qwen3.5-omni-plus`.

For 90-second videos at 720p, this returns in 8-12 seconds on `qwen3-omni-flash`. For the same video on `qwen3.5-omni-plus`, expect 18-25 seconds.

## Audio and image are the same shape

Same `content` list, different block type:

```python
content = [
    {"type": "audio_url", "audio_url": {"url": "https://.../voicemail.mp3"}},
    {"type": "text", "text": "Summarize this voicemail and identify the action items."},
]
```

```python
content = [
    {"type": "image_url", "image_url": {"url": "https://.../receipt.jpg"}},
    {"type": "text", "text": "Extract merchant, total, and date as JSON."},
]
```

You can pass multiple media blocks in one message — for example, three product images plus a question. The cost scales roughly linearly with media duration / image count.

## Latency, cost, and throughput planning

Real numbers from one of our worker pools, processing 30-90 second videos with `qwen3-omni-flash`:

- p50 latency: 11s
- p95 latency: 22s
- p99 latency: 38s
- Effective throughput per worker: ~5 videos/min if you don't block on output
- Cost per video: roughly ¥0.08-0.20 depending on length

Two architecture notes:

1. **Don't share an HTTP client across blocking workers.** Streaming holds the connection open; if you have 16 workers and a single client with default pool size 10, you will deadlock. Either give each worker its own client or raise the pool size.
2. **Pre-sign OSS URLs at job submission time, not at job start.** If your queue lag spikes, a URL signed at submit-time will still be valid 10 minutes later. A URL signed at start-time will be valid even if the job runs immediately. Pick the side of the trade-off that matches your queue.

> **Tip:** If you're tempted to chunk a 5-minute video into five 1-minute calls and merge — don't, unless the merge is trivial. The model loses cross-segment context. Either upload the whole thing (Qwen-Omni handles up to ~10 minutes) or genuinely change the prompt to be segment-local ("describe this minute only").

## Errors specific to Omni

| Error | Cause | Fix |
|---|---|---|
| `stream_options is required` | You forgot `stream_options={"include_usage": True}` | Add it. |
| `stream is required` | You called non-streaming | Switch to `stream=True`. |
| `media fetch failed` | URL not publicly resolvable, wrong content-type, or expired signed URL | Check the URL with curl from outside your VPC. |
| `media duration exceeds limit` | Single media > model's max | Split or downsample. |
| `media decode failed` | Codec the model doesn't handle (rare, usually exotic webm) | Re-encode to H.264 mp4. |

The "media fetch failed" one is the daily annoyance. My one-liner debug step is `curl -I "$url"` from a Mac on residential internet — if that doesn't return 200 with a sane content-type, the model can't read it either.

## What I'd build next

Now that one call gives you a transcript, visual context, and reasoning, the obvious follow-ups are: real-time meeting summaries (chunk audio), product Q&A from a video catalog (RAG over Omni outputs), and accessibility narration (Omni reads the video, then article 5's TTS reads the description aloud). Speaking of which — article 4 is the Wanxiang video pipeline, which is what generates the videos that Omni then describes. Closes the loop nicely.
