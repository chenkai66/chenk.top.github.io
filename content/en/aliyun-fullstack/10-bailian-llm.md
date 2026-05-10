---
title: "Alibaba Cloud Full Stack (10): Bailian and DashScope — The LLM Layer"
date: 2026-05-07 09:00:00
tags:
  - Alibaba Cloud
  - Bailian
  - DashScope
  - Qwen
  - LLM
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 10
description: "The complete LLM toolkit on Alibaba Cloud: Qwen model family, DashScope API (OpenAI-compatible), Wanxiang image/video generation, Qwen TTS, async task patterns, and fine-tuning. Build a multi-modal AI pipeline."
disableNunjucks: true
translationKey: "aliyun-fullstack-10"
---

When I first needed an LLM API for a production app in China, my options were limited and expensive. Most international providers had no mainland endpoint, billing required a foreign credit card, and latency from calling US-based APIs was 800ms+ before a single token came back. Then Qwen showed up on DashScope with an OpenAI-compatible endpoint, and suddenly building AI products in China became as straightforward as anywhere else. Same SDK, same request shape, same streaming protocol -- just a different `base_url` and a key from the Bailian console. I have been running production workloads against it for over a year now, and this article is the comprehensive walkthrough I wish I had on day one.

This is not a shallow overview. By the end you will understand the full model catalog, know how to call every modality (text, image, video, audio, embeddings), handle the async task pattern that trips up every team at least once, and have a working multi-modal pipeline that generates an article, illustrates it, and narrates it -- all from Python.

![Bailian LLM](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/10-bailian-llm/cover.png)

## Bailian vs DashScope: what is what

The naming confuses everyone, including Alibaba's own docs sometimes. Here is the truth:

**Bailian (百炼)** is the product platform. It lives at `bailian.console.aliyun.com`. This is where you manage API keys, browse the model catalog, launch fine-tuning jobs, build RAG applications, create prompt templates, evaluate model performance, and check billing. Think of it as the control plane.

**DashScope** is the API service. Every HTTP call hits `dashscope.aliyuncs.com`. The Python SDK is `pip install dashscope`. When your code calls a model, it is talking to DashScope. When you look at your bill or deploy a fine-tuned model, you are using Bailian.

In practice: you open Bailian to get your API key and configure things, then you write code against DashScope to actually use the models.

### How this maps to AWS

| Concept | Alibaba Cloud | AWS equivalent |
|---|---|---|
| Model marketplace + management console | **Bailian** | Bedrock console + SageMaker Studio |
| Model inference API | **DashScope** | Bedrock Runtime API |
| Fine-tuning platform | **Bailian fine-tuning** | Bedrock Custom Models / SageMaker Training |
| Agent builder | **Bailian Agent** | Bedrock Agents |
| Prompt engineering studio | **Bailian Prompt Lab** | Bedrock Playground |
| RAG service | **Bailian Knowledge Base** | Bedrock Knowledge Bases |

The key difference from AWS: on Alibaba Cloud, Qwen is a first-party model family built by the same company. On AWS, every model (Claude, Llama, Mistral) is third-party. This means Qwen models get features faster on DashScope than anywhere else, pricing is aggressive because there is no middleman margin, and the Chinese-language quality is unmatched because Qwen was trained with Chinese as a first-class language, not an afterthought.

For a deep dive into the Bailian platform itself, see our dedicated [Bailian series](/en/aliyun-bailian/01-platform-overview/).

## The Qwen model family

Qwen is not one model. It is a family of models spanning text, vision, audio, code, math, and multimodal understanding. Here is what matters for production:

![Qwen model family overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/10-bailian-llm/10_model_family.png)

### Text generation models

| model_id | Context | Best for | Input / Output (CNY per 1M tokens) |
|---|---|---|---|
| `qwen-turbo` | 128K | High-throughput classification, simple extraction, cheap batch jobs | 0.3 / 0.6 |
| `qwen-plus` | 128K | The default -- chat, summarization, translation, light reasoning | 0.8 / 2.0 |
| `qwen-max` | 128K | Hardest reasoning, legal/medical accuracy, when you cannot afford errors | 2.4 / 9.6 |
| `qwen3-max` | 128K | New default for hard reasoning; cheaper than qwen-max with thinking mode | 2.0 / 6.0 |
| `qwen3-coder-plus` | 128K | Code generation, diff/patch, AST manipulation | 1.0 / 4.0 |
| `qwen-turbo-longcontext` | 1M | Massive documents where 128K is not enough | 0.6 / 2.0 |

**My rule:** Default to `qwen-plus`. Move up to `qwen3-max` only when you have an eval proving Plus is not accurate enough. Move down to `qwen-turbo` only when cost actually matters at your volume. The `qwen3-max` model with `enable_thinking=True` can match `qwen-max` accuracy at lower price, but requires streaming -- more on that later.

### Multimodal and specialized models

| model_id | Modality | What it does | Pricing |
|---|---|---|---|
| `qwen3-omni-flash` | Video + Audio + Image + Text | Fast multimodal understanding (my default) | Per-token, varies by input type |
| `qwen3.5-omni-plus` | Video + Audio + Image + Text | Higher quality, longer reasoning, audio output | Per-token |
| `text-embedding-v3` | Text → Vector | 1024-dim embeddings for RAG and search | 0.7 / 1M tokens |
| `text-embedding-v4` | Text → Vector | Newer, marginally better on benchmarks | 0.7 / 1M tokens |
| `wan2.5-t2v-plus` | Text → Video | 5-second video generation from prompt | Per-second of video |
| `wan2.5-i2v-plus` | Image → Video | 5-second video from starting frame | Per-second of video |
| `qwen3-tts-flash` | Text → Audio | Speech synthesis, 40+ voices, dialect support | 0.8 CNY / 10K characters |

Each of these modalities has its own API pattern and its own set of gotchas. The rest of this article covers them one by one.

## DashScope API: OpenAI-compatible

This is the single most important thing to understand about DashScope: it provides an OpenAI-compatible endpoint. You can use the official OpenAI Python SDK with a two-line configuration change:

![DashScope API comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/10-bailian-llm/10_api_comparison.png)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

That is it. Every `client.chat.completions.create()` call, every streaming pattern, every function-calling schema you already know from OpenAI works here. The SDK is thread-safe and pools connections -- construct it once and hold it for the lifetime of the process. Constructing a new client per call adds 50-100ms of TLS handshake overhead.

### What works through the OpenAI-compatible endpoint

| Feature | Supported? | Notes |
|---|---|---|
| Chat completions | Yes | All Qwen text models |
| Streaming | Yes | Standard SSE protocol |
| Function calling / tools | Yes | Same schema as OpenAI |
| JSON mode | Yes | `response_format={"type": "json_object"}` |
| Vision (image input) | Yes | Via content blocks with `image_url` |
| Embeddings | Yes | `client.embeddings.create()` |
| Qwen-Omni (multimodal) | Yes | Video/audio/image content blocks |
| TTS | **No** | DashScope native API only |
| Image generation (Wanxiang) | **No** | DashScope native API only |
| Video generation (Wanxiang) | **No** | DashScope native API only |

The pattern is: anything that fits the OpenAI request/response shape goes through the compat endpoint. Anything async (video, image generation) or with a non-standard response format (TTS audio streams) uses the DashScope native API.

### The two endpoints side by side

| Endpoint | URL | SDK | Use for |
|---|---|---|---|
| **OpenAI-compatible** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `openai` Python SDK | Text, embeddings, vision, Omni |
| **DashScope native** | `https://dashscope.aliyuncs.com/api/v1/services/aigc/...` | `dashscope` Python SDK or raw HTTP | TTS, image gen, video gen |

I default to the OpenAI-compatible endpoint for everything it supports. The request shapes are familiar, the error handling is documented to death on the OpenAI side, and switching to another provider later is a one-line `base_url` change.

The Qwen LLM API is covered extensively in [Bailian Part 2: Qwen LLM API](/en/aliyun-bailian/02-qwen-llm-api/).

## Text generation deep dive

Let me walk through the patterns you will use daily.

### Basic chat completion

```python
response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that writes product descriptions."},
        {"role": "user", "content": "Write a 50-word description for a portable Bluetooth speaker."},
    ],
    temperature=0.7,
    max_tokens=200,
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
```

### Streaming

For anything user-facing, stream. Time-to-first-token is what users perceive as "fast." Total latency is what your dashboards measure. They are different problems.

```python
stream = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "user", "content": "Explain serverless computing in 3 sentences."},
    ],
    stream=True,
    stream_options={"include_usage": True},
)

full_response = ""
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        full_response += delta
        print(delta, end="", flush=True)

# The last chunk with include_usage=True contains token counts
```

Two things bite people: the last chunk has `delta.content == None` with a `finish_reason`, so always check `if delta:`. And if you want token counts in streaming mode, you must pass `stream_options={"include_usage": True}` -- without it the final chunk has no `usage` field and you will not know what you spent.

### The enable_thinking trap (Qwen3 family)

This is the bug I cost myself half a day on. Qwen3 models (`qwen3-max`, `qwen3-coder-plus`) have an `enable_thinking` parameter that activates chain-of-thought reasoning. It is powerful -- `qwen3-max` with thinking can match `qwen-max` accuracy at lower price -- but there is a hard rule:

> **`enable_thinking=True` requires `stream=True`. Non-streaming calls will fail.**

```python
# This works
stream = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "What is 127 * 389?"}],
    stream=True,
    extra_body={"enable_thinking": True},
)

reasoning = ""
answer = ""
for chunk in stream:
    delta = chunk.choices[0].delta
    # Thinking tokens come first, then the answer
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning += delta.reasoning_content
    if delta.content:
        answer += delta.content
```

```python
# This FAILS with a 400 error
response = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "What is 127 * 389?"}],
    extra_body={"enable_thinking": True},
    # Missing stream=True!
)
```

### Structured output (JSON mode)

When you need the model to return structured data -- product attributes, extracted entities, classification results -- use JSON mode:

```python
response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {
            "role": "system",
            "content": "Extract product attributes. Return JSON with keys: name, category, price_range, target_audience.",
        },
        {
            "role": "user",
            "content": "The AirPods Max are premium over-ear headphones by Apple, retailing at $549, aimed at audiophiles and professionals.",
        },
    ],
    response_format={"type": "json_object"},
)

import json
data = json.loads(response.choices[0].message.content)
# {"name": "AirPods Max", "category": "headphones", "price_range": "premium", "target_audience": "audiophiles and professionals"}
```

JSON mode is more reliable than just asking for JSON in the prompt. Without it, models occasionally add markdown fences or explanatory text around the JSON. With it, the output is always parseable. But it is not a schema validator -- if you need strict schema conformance, validate after parsing.

### Function calling

DashScope supports OpenAI-style function calling, which is how you build tool-using agents:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Shanghai'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "What is the weather like in Beijing today?"}],
    tools=tools,
    tool_choice="auto",
)

# The model returns a tool_call instead of a text response
tool_call = response.choices[0].message.tool_calls[0]
print(f"Function: {tool_call.function.name}")
print(f"Arguments: {tool_call.function.arguments}")
# Function: get_weather
# Arguments: {"city": "Beijing", "unit": "celsius"}
```

You then execute the function yourself, feed the result back as a `tool` message, and let the model generate the final response. The pattern is identical to OpenAI's function calling -- same JSON schema, same message flow.

### Multi-turn conversation

Maintaining conversation history is just appending messages to the array:

```python
messages = [
    {"role": "system", "content": "You are a cloud architecture advisor."},
]

def chat(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        temperature=0.7,
    )
    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg

# Turn 1
print(chat("I need to host a Python API with about 200 req/hour."))
# Turn 2 -- the model remembers the context
print(chat("Would serverless be cheaper than ECS for that?"))
# Turn 3
print(chat("What about cold starts?"))
```

Watch your token count. Every turn sends the full history as input tokens. For long conversations, implement a sliding window or summarization strategy. I typically cap at 20 turns and summarize the first 15 into a single system message when the limit is hit.

The key parameters worth tuning:

| Parameter | Default | Range | What it controls |
|---|---|---|---|
| `temperature` | 1.0 | 0.0 - 2.0 | Randomness. 0.0 for deterministic, 0.7-0.9 for creative |
| `top_p` | 1.0 | 0.0 - 1.0 | Nucleus sampling. Lower = more focused |
| `max_tokens` | Model-dependent | 1 - 8192 | Maximum output length |
| `stop` | None | List of strings | Stop generation at these sequences |
| `presence_penalty` | 0.0 | -2.0 - 2.0 | Penalize repeating topics |
| `frequency_penalty` | 0.0 | -2.0 - 2.0 | Penalize repeating exact tokens |

> **My defaults for production:** `temperature=0.3` for extraction and classification (you want consistency), `temperature=0.7` for creative writing and chat (you want variety), `max_tokens` always set explicitly (never rely on the default -- it varies by model and you do not want a surprise 8K-token response eating your budget).

## Embeddings

Embeddings turn text into vectors, which is the foundation of RAG (retrieval-augmented generation), semantic search, clustering, and deduplication. DashScope offers `text-embedding-v3` and the newer `text-embedding-v4`.

![Embedding and RAG pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/10-bailian-llm/10_embedding_pipeline.png)

```python
response = client.embeddings.create(
    model="text-embedding-v3",
    input="Alibaba Cloud provides elastic compute services through ECS.",
)

vector = response.data[0].embedding
print(f"Dimensions: {len(vector)}")  # 1024
print(f"First 5 values: {vector[:5]}")
```

### Batch embedding

For efficiency, embed multiple texts in a single call (up to 25 texts per batch, each up to 2048 tokens):

```python
texts = [
    "ECS is Alibaba Cloud's virtual machine service.",
    "OSS provides object storage similar to AWS S3.",
    "Function Compute is a serverless execution engine.",
    "PolarDB is a cloud-native distributed database.",
    "DashScope is the API service for Qwen models.",
]

response = client.embeddings.create(
    model="text-embedding-v3",
    input=texts,
)

vectors = [item.embedding for item in response.data]
print(f"Embedded {len(vectors)} texts, each {len(vectors[0])} dimensions")
```

### Using embeddings for semantic search

The typical pattern: embed your knowledge base offline, store vectors in a vector database (or OpenSearch, which we covered in [Part 9: OpenSearch](/en/aliyun-fullstack/09-opensearch/)), then at query time embed the user's question and find the nearest neighbors.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Embed the query
query = "How do I attach a disk to a virtual machine?"
query_response = client.embeddings.create(
    model="text-embedding-v3",
    input=query,
)
query_vector = query_response.data[0].embedding

# Compare against our document vectors
similarities = [
    (texts[i], cosine_similarity(query_vector, vectors[i]))
    for i in range(len(vectors))
]
similarities.sort(key=lambda x: x[1], reverse=True)

for text, score in similarities[:3]:
    print(f"  {score:.4f}  {text}")
# 0.8234  ECS is Alibaba Cloud's virtual machine service.
# 0.6891  OSS provides object storage similar to AWS S3.
# ...
```

In production, do not compute cosine similarity in Python loops. Use OpenSearch's vector search or a dedicated vector database like Milvus. The code above is for understanding the concept.

## Wanxiang: image and video generation

Wanxiang is DashScope's generative media family. It covers text-to-image, image-to-video, and text-to-video. All media generation uses the DashScope native API (not the OpenAI-compatible endpoint) and follows an async task pattern.

![Wanxiang async generation pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/10-bailian-llm/10_wanxiang_pipeline.png)

![Async task pattern for media generation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/10-bailian-llm/10_async_pattern.png)

### The async task pattern

Every Wanxiang call follows the same three-step dance:

1. **Create the task.** POST with header `X-DashScope-Async: enable`. You get a `task_id` immediately.
2. **Poll.** GET `/api/v1/tasks/{task_id}` until `task_status` is `SUCCEEDED` or `FAILED`.
3. **Download.** The success response includes a URL. **Download within 24 hours** -- after that the URL returns 404 and your media is gone forever.

The 24-hour expiry is the single biggest operational footgun. I have seen multiple teams -- mine included -- lose work because they polled, logged the URL, then failed to download because of an unrelated bug, then noticed the next day. Treat the URL the way you would treat a one-time download link: download immediately, store to your own OSS, never assume it will be there tomorrow.

### Text-to-video example

```python
import os
import time
import requests

API_KEY = os.environ["DASHSCOPE_API_KEY"]
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "X-DashScope-Async": "enable",
}

def create_video_task(prompt: str, size: str = "1280*720", duration: int = 5) -> str:
    """Submit a text-to-video generation task. Returns task_id."""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    payload = {
        "model": "wan2.5-t2v-plus",
        "input": {"prompt": prompt},
        "parameters": {"size": size, "duration": duration},
    }
    resp = requests.post(url, json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["output"]["task_id"]


def poll_task(task_id: str, max_wait: int = 600) -> dict:
    """Poll until task completes. Returns the full output dict."""
    url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    elapsed = 0
    interval = 5
    while elapsed < max_wait:
        resp = requests.get(url, headers=headers)
        result = resp.json()
        status = result["output"]["task_status"]
        
        if status == "SUCCEEDED":
            return result["output"]
        elif status == "FAILED":
            raise RuntimeError(f"Task failed: {result['output'].get('message', 'unknown')}")
        
        time.sleep(interval)
        elapsed += interval
        interval = min(interval * 1.5, 30)  # Exponential backoff, cap at 30s
    
    raise TimeoutError(f"Task {task_id} did not complete within {max_wait}s")


def download_video(video_url: str, output_path: str):
    """Download the video before the 24-hour expiry."""
    resp = requests.get(video_url, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


# Usage
task_id = create_video_task(
    prompt="A drone shot flying over Shanghai's Pudong skyline at sunset, cinematic, 4K quality",
    size="1280*720",
    duration=5,
)
print(f"Task submitted: {task_id}")

output = poll_task(task_id)
video_url = output["video_url"]
print(f"Video ready: {video_url}")

download_video(video_url, "shanghai_sunset.mp4")
print("Downloaded to shanghai_sunset.mp4")
```

### Image-to-video

Same pattern, different model and input:

```python
def create_i2v_task(prompt: str, image_url: str, duration: int = 5) -> str:
    """Image-to-video: animate a starting frame."""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    payload = {
        "model": "wan2.5-i2v-plus",
        "input": {
            "prompt": prompt,
            "img_url": image_url,
        },
        "parameters": {"duration": duration},
    }
    resp = requests.post(url, json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["output"]["task_id"]
```

Both models cap at 5 seconds. If you need 10 seconds, make two clips and stitch them -- use the last frame of the first clip as the `img_url` input for the second.

For the full Wanxiang video deep dive, see [Bailian Part 4: Wanxiang Video Generation](/en/aliyun-bailian/04-wanxiang-video-generation/).

### Text-to-image

Image generation uses a slightly different endpoint but the same async pattern:

```python
def create_image_task(prompt: str, size: str = "1024*1024") -> str:
    """Submit a text-to-image generation task."""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
    payload = {
        "model": "wanx2.1-t2i-plus",
        "input": {"prompt": prompt},
        "parameters": {"size": size, "n": 1},
    }
    resp = requests.post(url, json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["output"]["task_id"]
```

Poll with the same `poll_task()` function. The success response contains `output.results[0].url` instead of `output.video_url` -- small inconsistency, just adapt.

## Qwen TTS: text-to-speech

Qwen TTS is the part that trips up everyone who assumes "if Qwen LLM works through the OpenAI client, TTS must too."

> **Qwen-TTS does NOT work via the OpenAI-compatible endpoint. It is DashScope-native only.**

You cannot point the `openai` SDK's `audio.speech.create` at the compat URL and have it work. There is no compat shim for TTS. Use the `dashscope` SDK or raw HTTP.

### The simplest call

```python
import os
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

synth = SpeechSynthesizer(model="qwen3-tts-flash", voice="Cherry")
audio_bytes = synth.call("Welcome to the product demo. Today we will show you three new features.")

with open("demo_narration.mp3", "wb") as f:
    f.write(audio_bytes)
```

### Voice selection

The model supports 40+ voices. Here are the ones I actually use:

| Voice | Gender | Character | Best for |
|---|---|---|---|
| Cherry | Female | Warm, natural, positive | Product demos, tutorials |
| Serena | Female | Gentle, calm | Meditation, soft narration |
| Ethan | Male | Warm, energetic | Marketing videos |
| Andre | Male | Deep, steady, magnetic | Professional narration |
| Neil | Male | News anchor style | Reports, announcements |
| Maia | Female | Intellectual, gentle | Educational content |
| Stella | Female | Sweet, youthful | Social media content |
| Bellona | Female | Loud, powerful | Calls to action |

Voice names are case-sensitive. `Cherry` works, `cherry` does not.

### Streaming TTS for real-time playback

For long text or real-time applications, stream the audio:

```python
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

synth = SpeechSynthesizer(
    model="qwen3-tts-flash",
    voice="Ethan",
    format="mp3",
    sample_rate=24000,
)

# Streaming callback
chunks = []
def on_audio(data):
    chunks.append(data)

synth.streaming_call(
    text="This is a longer piece of text that will be synthesized incrementally. "
         "Each chunk of audio is delivered as soon as it is ready, "
         "reducing time-to-first-audio for the user.",
    callback=on_audio,
)

with open("streamed_output.mp3", "wb") as f:
    for chunk in chunks:
        f.write(chunk)
```

### Language and dialect coverage

This is where Qwen TTS genuinely has no competition. Beyond Mandarin and English, it supports Cantonese, Sichuanese, Shanghainese, Northeast dialect, Japanese, and Korean -- with voices that sound native, not like a tourist reading a phrasebook. I have not found another TTS API that handles Cantonese this well at this price.

For the full TTS deep dive including voice cloning and instruct mode, see [Bailian Part 5: Qwen TTS](/en/aliyun-bailian/05-qwen-tts-voice/).

## Fine-tuning on Bailian

Fine-tuning is the nuclear option. Before you reach for it, ask whether prompt engineering, few-shot examples, or RAG can solve your problem. In my experience, 80% of "we need to fine-tune" conversations end with "actually, a better system prompt fixed it."

![Bailian platform overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/10-bailian-llm/10_bailian_platform.png)

### When fine-tuning actually makes sense

| Scenario | Why fine-tuning helps | Alternative to try first |
|---|---|---|
| Domain-specific jargon the model consistently gets wrong | Training data teaches the correct terminology | Few-shot examples in the prompt |
| Consistent output format (e.g., always return XML with specific tags) | Fine-tuning bakes the format into the model weights | JSON mode + structured prompt |
| Cost reduction at high volume | Fine-tuned `qwen-turbo` can match `qwen-plus` quality for your specific task | Measure whether the cost difference actually matters |
| Latency reduction | Smaller fine-tuned model runs faster | Prompt compression, shorter system prompt |
| Tone/style consistency | The model learns your brand voice | Detailed style guide in system prompt |

### Preparing training data

Bailian expects JSONL format with the standard chat completion structure:

```jsonl
{"messages": [{"role": "system", "content": "You are a product description writer for electronics."}, {"role": "user", "content": "Write a description for: Sony WH-1000XM5 headphones"}, {"role": "assistant", "content": "Premium wireless noise-cancelling headphones with 30-hour battery life..."}]}
{"messages": [{"role": "system", "content": "You are a product description writer for electronics."}, {"role": "user", "content": "Write a description for: Apple AirPods Pro 2"}, {"role": "assistant", "content": "True wireless earbuds with adaptive noise cancellation..."}]}
```

Rules for good training data:

- **Minimum 50 examples**, 200-500 is the sweet spot. More than 1000 rarely helps unless your domain is very diverse.
- **Consistent system prompt** across all examples -- the model learns the system prompt as part of the task definition.
- **High-quality outputs only** -- every assistant response should be exactly what you want the model to produce. One bad example can teach one bad habit.
- **Diverse inputs** -- do not repeat the same question with minor variations. Cover the full range of inputs you expect in production.
- **Validate JSONL** before uploading. One malformed line and the whole job fails silently.

```python
import json

def validate_training_data(filepath: str) -> tuple[int, list[str]]:
    """Validate JSONL training data. Returns (count, errors)."""
    errors = []
    count = 0
    with open(filepath, "r") as f:
        for i, line in enumerate(f, 1):
            count += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {i}: invalid JSON")
                continue
            
            if "messages" not in data:
                errors.append(f"Line {i}: missing 'messages' key")
                continue
            
            roles = [m["role"] for m in data["messages"]]
            if roles[-1] != "assistant":
                errors.append(f"Line {i}: last message must be 'assistant', got '{roles[-1]}'")
            if "user" not in roles:
                errors.append(f"Line {i}: missing 'user' message")
    
    return count, errors

count, errors = validate_training_data("training_data.jsonl")
print(f"Total examples: {count}")
if errors:
    for e in errors:
        print(f"  ERROR: {e}")
else:
    print("All examples valid")
```

### Launching a fine-tuning job

Fine-tuning is done through the Bailian console or the API:

1. **Upload training data** to the Bailian console under Data Management
2. **Create a fine-tuning job**: select base model (e.g., `qwen-turbo`), point to your dataset, configure hyperparameters
3. **Monitor training**: the console shows loss curves and training progress
4. **Deploy**: once training completes, deploy the model to get a custom `model_id`

Via the API (using the `dashscope` SDK):

```python
import dashscope
from dashscope import FineTune

# Create fine-tuning job
job = FineTune.create(
    model="qwen-turbo",
    training_file_ids=["file-abc123"],  # Upload files first via the console
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 4,
        "learning_rate_multiplier": 1.0,
    },
)
print(f"Job ID: {job.output.job_id}")
print(f"Status: {job.output.status}")

# Check status
status = FineTune.get(job.output.job_id)
print(f"Status: {status.output.status}")
# PENDING → RUNNING → SUCCEEDED
```

### Cost comparison: fine-tuned small vs large with prompting

This is the math that decides whether fine-tuning is worth it:

| Approach | Model | Input cost/1M | Output cost/1M | Typical prompt tokens | Monthly cost at 1M requests |
|---|---|---|---|---|---|
| Prompt engineering | `qwen-plus` | 0.8 | 2.0 | 800 (long system prompt + few-shot) | ~2,240 CNY |
| Prompt engineering | `qwen-max` | 2.4 | 9.6 | 800 | ~7,680 CNY |
| Fine-tuned | `qwen-turbo` (custom) | ~0.6 | ~1.2 | 200 (short prompt, no few-shot needed) | ~360 CNY |

The fine-tuned turbo model costs roughly 6x less than prompt-engineered plus and 21x less than max -- because the prompt is shorter (no few-shot examples needed, the behavior is baked in) and the per-token price of turbo is lower. But fine-tuning itself costs money (training compute) and time (preparing data, validating quality, monitoring for drift). It is worth it only above roughly 100K requests/month for a specific, well-defined task.

## Solution: multi-modal AI pipeline

Let me put it all together. Here is a complete pipeline that takes a topic, generates an article draft, creates an illustration, and produces a voice narration -- all orchestrated in Python.

![Multi-modal AI pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/10-bailian-llm/10_multimodal_flow.png)

```python
"""
Multi-modal AI content pipeline.
Generates: article (Qwen) + illustration (Wanxiang) + narration (Qwen TTS).
"""

import os
import json
import time
import requests
from openai import OpenAI

# -- Config --
API_KEY = os.environ["DASHSCOPE_API_KEY"]

# OpenAI-compat client for text
text_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# DashScope native headers for media generation
NATIVE_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "X-DashScope-Async": "enable",
}
POLL_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# -- Step 1: Generate article --
def generate_article(topic: str) -> str:
    """Generate a short article using Qwen."""
    response = text_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technology writer. Write concise, engaging articles "
                    "with a clear structure: introduction, 2-3 key points, conclusion. "
                    "Keep it under 300 words."
                ),
            },
            {"role": "user", "content": f"Write an article about: {topic}"},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content


# -- Step 2: Generate illustration prompt --
def generate_image_prompt(article: str) -> str:
    """Ask the model to describe an illustration for the article."""
    response = text_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given an article, write a text-to-image prompt for an illustration. "
                    "The prompt should describe a clean, modern, editorial-style image. "
                    "Return ONLY the image prompt, nothing else."
                ),
            },
            {"role": "user", "content": article},
        ],
        temperature=0.5,
        max_tokens=100,
    )
    return response.choices[0].message.content


# -- Step 3: Generate image --
def generate_image(prompt: str) -> str:
    """Submit image generation task and return the image URL."""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
    payload = {
        "model": "wanx2.1-t2i-plus",
        "input": {"prompt": prompt},
        "parameters": {"size": "1024*1024", "n": 1},
    }
    resp = requests.post(url, json=payload, headers=NATIVE_HEADERS)
    resp.raise_for_status()
    task_id = resp.json()["output"]["task_id"]
    
    # Poll
    output = poll_task(task_id)
    return output["results"][0]["url"]


# -- Step 4: Generate narration --
def generate_narration(text: str, output_path: str):
    """Generate TTS narration using DashScope native API."""
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer
    
    dashscope.api_key = API_KEY
    synth = SpeechSynthesizer(model="qwen3-tts-flash", voice="Ethan")
    audio_bytes = synth.call(text)
    
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return output_path


# -- Shared: poll task --
def poll_task(task_id: str, max_wait: int = 300) -> dict:
    """Poll a DashScope async task until completion."""
    url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    elapsed = 0
    interval = 5
    while elapsed < max_wait:
        resp = requests.get(url, headers=POLL_HEADERS)
        result = resp.json()
        status = result["output"]["task_status"]
        if status == "SUCCEEDED":
            return result["output"]
        elif status == "FAILED":
            raise RuntimeError(f"Task failed: {result['output'].get('message')}")
        time.sleep(interval)
        elapsed += interval
        interval = min(interval * 1.5, 30)
    raise TimeoutError(f"Task {task_id} timed out after {max_wait}s")


# -- Orchestrator --
def run_pipeline(topic: str):
    """Run the full multi-modal content pipeline."""
    print(f"=== Topic: {topic} ===\n")
    
    # Step 1: Article
    print("[1/4] Generating article...")
    article = generate_article(topic)
    print(f"Article: {len(article)} chars\n")
    with open("article.md", "w") as f:
        f.write(article)
    
    # Step 2: Image prompt
    print("[2/4] Generating image prompt...")
    image_prompt = generate_image_prompt(article)
    print(f"Image prompt: {image_prompt}\n")
    
    # Step 3: Illustration
    print("[3/4] Generating illustration (this takes 30-60s)...")
    image_url = generate_image(image_prompt)
    print(f"Image URL: {image_url}\n")
    
    # Download image
    img_resp = requests.get(image_url)
    with open("illustration.png", "wb") as f:
        f.write(img_resp.content)
    
    # Step 4: Narration
    print("[4/4] Generating voice narration...")
    # Use just the intro paragraph for narration demo
    intro = article.split("\n\n")[0]
    generate_narration(intro, "narration.mp3")
    print("Narration saved to narration.mp3\n")
    
    print("=== Pipeline complete ===")
    print("  article.md        - Written article")
    print("  illustration.png  - AI-generated illustration")
    print("  narration.mp3     - Voice narration of intro")


if __name__ == "__main__":
    run_pipeline("The future of serverless computing on Alibaba Cloud")
```

This is about 120 lines of Python. It calls three different DashScope capabilities (text generation via OpenAI-compat, image generation via native async, TTS via native sync) and produces three output files. In production, you would add error handling, retry logic, and parallel execution (image and TTS can run concurrently since they are independent). But the bones are here.

Multimodal capabilities including video understanding are covered in [Bailian Part 3: Qwen-Omni](/en/aliyun-bailian/03-qwen-omni-multimodal/).

## API rate limits and error handling

Before you go to production, know the rate limits:

| Model family | Default RPM (requests/min) | Default TPM (tokens/min) | Can be raised? |
|---|---|---|---|
| `qwen-turbo` | 500 | 500K | Yes, via ticket |
| `qwen-plus` | 300 | 300K | Yes |
| `qwen-max` | 120 | 120K | Yes |
| `qwen3-max` | 120 | 120K | Yes |
| `text-embedding-v3` | 500 | 500K | Yes |
| `wan2.5-t2v-plus` | 20 | N/A | Yes |
| `qwen3-tts-flash` | 180 | N/A | Yes |

When you hit a limit, DashScope returns HTTP 429 with a `Retry-After` header. Handle it:

```python
import time
from openai import RateLimitError

def call_with_retry(func, max_retries=3):
    """Retry on rate limit with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt + 1
            print(f"Rate limited. Waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)


# Usage
result = call_with_retry(
    lambda: text_client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello"}],
    )
)
```

### Common error codes

| HTTP status | DashScope code | Meaning | Fix |
|---|---|---|---|
| 400 | `InvalidParameter` | Bad request body | Check your request against the docs |
| 401 | `InvalidApiKey` | Wrong or expired API key | Regenerate key in Bailian console |
| 404 | `ModelNotFound` | Model ID typo or model not available | Check exact `model_id` string |
| 429 | `Throttling` | Rate limit exceeded | Backoff and retry, or request quota increase |
| 500 | `InternalError` | Server-side issue | Retry after 5-10 seconds |

### Budget alerts

Set a budget alert in the Bailian console. I have eaten a four-figure bill exactly once because someone left a debug loop running overnight. The alert would have caught it in 30 minutes instead of 8 hours.

```bash
# Quick check: your current month's usage via CLI
curl -s "https://dashscope.aliyuncs.com/compatible-mode/v1/models" \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" | python3 -m json.tool
```

## Putting it in context: the full-stack picture

Here is where DashScope sits in a typical Alibaba Cloud architecture:

| Layer | Service | Article |
|---|---|---|
| Compute | ECS, Function Compute | [Part 2](/en/aliyun-fullstack/02-ecs-compute/), [Part 8](/en/aliyun-fullstack/08-serverless/) |
| Networking | VPC, SLB | [Part 3](/en/aliyun-fullstack/03-vpc-networking/) |
| Search & Retrieval | OpenSearch + embeddings | [Part 9](/en/aliyun-fullstack/09-opensearch/) |
| **AI / LLM** | **DashScope (this article)** | **Part 10** |
| Storage | OSS (for media assets) | [Part 1](/en/aliyun-fullstack/01-ecosystem-map/) |

A typical AI application flow:

1. User sends a request to your API (running on ECS or Function Compute)
2. Your app embeds the query using `text-embedding-v3` via DashScope
3. You search OpenSearch for relevant context using those embeddings
4. You call `qwen-plus` with the retrieved context + user query via DashScope
5. The response streams back to the user
6. If media is needed, you call Wanxiang (async) and store results on OSS

Every piece of this stack is covered in this series. DashScope is the brain; the other services are the body.

## Key takeaways

1. **Bailian is the console, DashScope is the API.** You configure on Bailian, you code against DashScope. Do not confuse them.

2. **Use the OpenAI-compatible endpoint as your default.** `base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"` with the `openai` SDK covers text, embeddings, vision, and multimodal. Only drop to native API for TTS, image gen, and video gen.

3. **Default to `qwen-plus`.** Move up to `qwen3-max` (with thinking) only when evals prove Plus is not enough. Move down to `qwen-turbo` only when cost matters at your volume.

4. **Qwen3 thinking requires streaming.** `enable_thinking=True` without `stream=True` is a hard error. This catches everyone once.

5. **TTS is DashScope-native only.** Do not try the OpenAI-compat endpoint for `qwen3-tts-flash`. It will 404.

6. **All media generation is async.** Submit task, poll, download within 24 hours. The 24-hour URL expiry is the most common production incident.

7. **Fine-tuning is the last resort.** Try prompt engineering, few-shot examples, and RAG first. Fine-tune only when you have 100K+ monthly requests for a specific, well-defined task where a smaller model with training data can match a larger model with a long prompt.

8. **Set budget alerts.** Do it now, before someone leaves a debug loop running overnight.

## What's next

[Part 11](/en/aliyun-fullstack/11-security/) covers security on Alibaba Cloud: RAM policies, KMS for key management, Security Center, and WAF. Every API key, every DashScope call, every OSS bucket from this article needs proper security -- and that is where we are headed next.
