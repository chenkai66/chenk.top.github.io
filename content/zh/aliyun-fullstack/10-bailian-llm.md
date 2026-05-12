---
title: "阿里云全栈实战（十）：DashScope 与大模型层"
date: 2026-05-07 09:00:00
tags:
  - Alibaba Cloud
  - Bailian
  - DashScope
  - Qwen
  - LLM
  - Cloud Computing
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 10
description: "The complete LLM toolkit on Alibaba Cloud: Qwen model family, DashScope API (OpenAI-compatible), Wanxiang image/video generation, Qwen TTS, async task patterns, and fine-tuning. Build a multi-modal AI pipeline."
disableNunjucks: true
translationKey: "aliyun-fullstack-10"
---
早年在国内开发生产级 LLM 应用时，可选方案极少且成本高昂：国际大厂要么未在中国内地部署服务端点（endpoint），要么计费需绑定境外信用卡；若调用其美国 API，首 token 延迟普遍超过 800ms。后来 Qwen 接入 DashScope 并提供 OpenAI 兼容接口，国内开发 AI 产品的体验因此与海外接轨。 SDK 一样，请求结构一样，流式协议也一样——只要改个 `base_url`，再从百炼控制台拿个 Key 就行。该方案已在生产环境稳定运行一年以上。本文系统梳理了我初上手时最急需的实战经验。

本文不是泛泛而谈的概览。你将厘清完整的模型目录，掌握文本、图像、视频、音频、embeddings 等所有模态的调用方法，理解各团队高频遭遇的异步任务模式，并动手实现端到端的多模态流水线——生成文章、配图和语音合成，全程基于 Python。


## Bailian vs DashScope：到底啥是啥

这两个名称容易混淆，阿里云官方文档中的界定也不够清晰。简要说明如下。

**Bailian (百炼)** 是产品平台。地址在 `bailian.console.aliyun.com`。在这里你管理 API Key、浏览模型目录、启动微调任务、搭建 RAG 应用、创建提示词模板、评估模型表现以及查看账单。可将其理解为控制平面。

**DashScope** 是 API 服务。所有 HTTP 请求都打到 `dashscope.aliyuncs.com`。 Python SDK 是 `pip install dashscope`。代码调用模型时是在跟 DashScope 对话；查账单或部署微调模型时，用的是 Bailian。

实际流程是先在 Bailian 获取 API Key 并配置环境变量，然后用代码调用 DashScope 模型接口。

### 对应到 AWS 是怎么个概念

| 概念 | 阿里云 | AWS 对应物 |
|---|---|---|
| 模型市场 + 管理控制台 | **Bailian** | Bedrock console + SageMaker Studio |
| 模型推理 API | **DashScope** | Bedrock Runtime API |
| 微调平台 | **Bailian fine-tuning** | Bedrock Custom Models / SageMaker Training |
| Agent 构建器 | **Bailian Agent** | Bedrock Agents |
| 提示词工程工作室 | **Bailian Prompt Lab** | Bedrock Playground |
| RAG 服务 | **Bailian Knowledge Base** | Bedrock Knowledge Bases |

与 AWS 的关键区别在于：在阿里云平台上，Qwen 是阿里自研的第一方模型家族，而在 AWS 上，所有模型（Claude、Llama、Mistral）都是第三方的。这意味着 Qwen 模型在 DashScope 上功能迭代更快、定价更优、中文能力业界领先——训练始终以中文为首要语言，而非后期适配。

想深入了解 Bailian 平台本身，可以看我们的专门系列 [Bailian 系列](/zh/aliyun-bailian/01-platform-overview/)。

## Qwen 模型家族

Qwen 不是一个模型，而是一个家族。覆盖文本、视觉、音频、代码、数学和多模态理解。生产环境里值得关注的有这些：

![Qwen 模型家族概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/10-bailian-llm/10_model_family.png)

### 文本生成模型

| model_id | Context | 适用场景 | 输入 / 输出 (CNY per 1M tokens) |
|---|---|---|---|
| `qwen-turbo` | 128K | 高吞吐分类、简单提取、廉价批量任务 | 0.3 / 0.6 |
| `qwen-plus` | 128K | 默认首选 -- 聊天、总结、翻译、轻度推理 | 0.8 / 2.0 |
| `qwen-max` | 128K | 高难度推理、法律/医疗准确性、容错率低的场景 | 2.4 / 9.6 |
| `qwen3-max` | 128K | 高难度推理新默认；开启 thinking 模式比 qwen-max 更便宜 | 2.0 / 6.0 |
| `qwen3-coder-plus` | 128K | 代码生成、 diff/patch、 AST 操作 | 1.0 / 4.0 |
| `qwen-turbo-longcontext` | 1M | 128K 装不下的超大文档 | 0.6 / 2.0 |

**我的原则：** 默认选 `qwen-plus`。只有当评估证明 Plus 准确度不够时，再升级到 `qwen3-max`。只有当你的体量下成本真的成为瓶颈时，再降级到 `qwen-turbo`。开启 `enable_thinking=True` 的 `qwen3-max` 模型能以更低价格达到 `qwen-max` 的准确度，但需要流式输出——后面会细说。

### 多模态和专用模型

| model_id | 模态 | 功能 | 定价 |
|---|---|---|---|
| `qwen3-omni-flash` | 视频 + 音频 + 图像 + 文本 | 快速多模态理解（我的默认选） | 按 token，随输入类型变化 |
| `qwen3.5-omni-plus` | 视频 + 音频 + 图像 + 文本 | 更高质量、更长推理、音频输出 | 按 token |
| `text-embedding-v3` | 文本 → 向量 | 1024 维 embeddings，用于 RAG 和搜索 | 0.7 / 1M tokens |
| `text-embedding-v4` | 文本 → 向量 | 更新版本，基准测试表现略好 | 0.7 / 1M tokens |
| `wan2.5-t2v-plus` | 文本 → 视频 | 根据提示生成 5 秒视频 | 按视频秒数 |
| `wan2.5-i2v-plus` | 图像 → 视频 | 根据起始帧生成 5 秒视频 | 按视频秒数 |
| `qwen3-tts-flash` | 文本 → 音频 | 语音合成， 40+ 音色，支持方言 | 0.8 CNY / 1K 字符 |

每种模态都有自己的 API 模式和坑。文章剩下的部分会逐一拆解。

## DashScope API： OpenAI 兼容

这是关于 DashScope 最重要的一点：它提供了 OpenAI 兼容的 endpoint。只需修改两行配置，就能直接使用官方的 OpenAI Python SDK。

![DashScope API comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/10-bailian-llm/10_api_comparison.png)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

就这么简单。你之前从 OpenAI 那里熟悉的所有 `client.chat.completions.create()` 调用、流式模式、函数调用 schema，在这里都能用。 SDK 是线程安全的且会连接池化——在进程生命周期内构造一次客户端就够了。每次调用都新建 client 会增加 50-100ms 的 TLS 握手开销。

### OpenAI 兼容 endpoint 支持哪些功能

| 功能 | 支持？ | 备注 |
|---|---|---|
| Chat completions | Yes | 所有 Qwen 文本模型 |
| Streaming | Yes | 标准 SSE 协议 |
| Function calling / tools | Yes |  schema 与 OpenAI 相同 |
| JSON mode | Yes | `response_format={"type": "json_object"}` |
| Vision (图像输入) | Yes | 通过带 `image_url` 的内容块 |
| Embeddings | Yes | `client.embeddings.create()` |
| Qwen-Omni (多模态) | Yes | 视频/音频/图像内容块 |
| TTS | **No** | 仅限 DashScope 原生 API |
| 图像生成 (Wanxiang) | **No** | 仅限 DashScope 原生 API |
| 视频生成 (Wanxiang) | **No** | 仅限 DashScope 原生 API |

规律很简单：任何符合 OpenAI 请求/响应形状的都走兼容 endpoint，而任何异步任务（如视频、图像生成）或响应格式非标准（如 TTS 音频流）的都走 DashScope 原生 API。

### 两个 endpoint 对比

| Endpoint | URL | SDK | 适用场景 |
|---|---|---|---|
| **OpenAI-compatible** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `openai` Python SDK | 文本、 embeddings、视觉、 Omni |
| **DashScope native** | `https://dashscope.aliyuncs.com/api/v1/services/aigc/...` | `dashscope` Python SDK 或 raw HTTP | TTS、图像生成、视频生成 |

我默认所有支持的功能都走 OpenAI 兼容 endpoint。请求结构熟悉，错误处理在 OpenAI 那边文档详尽，以后想切换 provider 也只需要改一行 `base_url`。

Qwen LLM API 的详细内容覆盖在 [Bailian 第二部分：Qwen LLM API](/zh/aliyun-bailian/02-qwen-llm-api/)。
## 文本生成深入解析

咱们来聊聊你每天都会用到的几种模式。

### 基础聊天补全

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

### 流式输出

只要面向用户，必须用流式。用户感知的“快”是首字延迟（Time-to-first-token），而仪表盘监控的是总延迟。这是两个不同的问题。

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

这里有两个坑特别容易踩：最后一个 chunk 的 `delta.content` 是 `None`，但会带有 `finish_reason`，所以务必加上 `if delta:` 判断。另外，如果你想在流式模式下拿到 Token 用量，必须传 `stream_options={"include_usage": True}` —— 少了这个参数，最后一个 chunk 里就没有 `usage` 字段，你根本不知道这次调用花了多少 Token。

### enable_thinking 陷阱（Qwen3 系列）

这个 bug 让我白白浪费了半天时间。 Qwen3 模型（`qwen3-max`, `qwen3-coder-plus`）有个 `enable_thinking` 参数，用来激活思维链推理。这功能很强 —— 开启 thinking 的 `qwen3-max` 能在更低成本下达到 `qwen-max` 的准确率 —— 但有条硬规矩：

> **`enable_thinking=True` 必须配合 `stream=True`。非流式调用会直接失败。**

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

### 结构化输出（JSON 模式）

当你需要模型返回结构化数据 —— 比如产品属性、实体抽取、分类结果 —— 直接用 JSON 模式：

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

JSON 模式比直接在 Prompt 里要求返回 JSON 靠谱得多。不开这个模式，模型偶尔会给 JSON 包上 markdown 代码框，或者前后加点解释性文字。开了之后，输出保证能直接解析。但它不是 schema 验证器 —— 如果你需要严格的 schema 合规，解析完还得自己再校验一遍。

### 函数调用

DashScope 支持 OpenAI 风格的函数调用，这也是构建 Tool-using Agent 的标准方式：

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

拿到结果后你自己执行函数，把结果作为 `tool` 消息喂回去，再让模型生成最终回复。这套流程和 OpenAI 的函数调用完全一致 —— JSON schema 一样，消息流转也一样。

### 多轮对话

维护对话历史其实就是往数组里追加消息：

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

盯着点 Token 用量。每一轮对话都会把完整的历史记录作为输入 Token 发过去。对话长了之后，得实现滑动窗口或者摘要策略。我通常限制在 20 轮，一旦超限，就把前 15 轮总结成一条 system 消息塞进去。

几个关键参数的调优参考：

| Parameter | Default | Range | What it controls |
|---|---|---|---|
| `temperature` | 1.0 | 0.0 - 2.0 | 随机性。 0.0 用于确定性任务， 0.7-0.9 用于创意生成 |
| `top_p` | 1.0 | 0.0 - 1.0 | 核采样。值越低输出越聚焦 |
| `max_tokens` | Model-dependent | 1 - 8192 | 最大输出长度 |
| `stop` | None | List of strings | 遇到这些序列时停止生成 |
| `presence_penalty` | 0.0 | -2.0 - 2.0 | 惩罚重复的话题 |
| `frequency_penalty` | 0.0 | -2.0 - 2.0 | 惩罚重复的具体 Token |

> **我的生产环境默认值：** 抽取和分类任务用 `temperature=0.3`（你要的是稳定性），创意写作和聊天用 `temperature=0.7`（你要的是多样性），`max_tokens` 永远显式设置（别依赖默认值 —— 不同模型默认值不一样，你肯定不想意外冒出一个 8K Token 的回复吃掉预算）。

## 嵌入向量

Embeddings 把文本变成向量，这是 RAG （检索增强生成）、语义搜索、聚类和去重的基础。 DashScope 提供 `text-embedding-v3` 和更新的 `text-embedding-v4`。

![嵌入向量与 RAG 流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/10-bailian-llm/10_embedding_pipeline.png)

```python
response = client.embeddings.create(
    model="text-embedding-v3",
    input="Alibaba Cloud provides elastic compute services through ECS.",
)

vector = response.data[0].embedding
print(f"Dimensions: {len(vector)}")  # 1024
print(f"First 5 values: {vector[:5]}")
```

### 批量嵌入

为了效率，单次调用可以嵌入多个文本（每批最多 25 条，每条最多 2048 Token）：

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

### 用 Embeddings 做语义搜索

典型流程：离线嵌入你的知识库，把向量存进向量数据库（或者 OpenSearch，我们在 [第 9 部分：OpenSearch](/zh/aliyun-fullstack/09-opensearch/) 里讲过），查询时嵌入用户问题，然后找最近邻。

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

生产环境别在 Python 循环里算余弦相似度。直接用 OpenSearch 的向量搜索功能，或者专门的向量数据库比如 Milvus。上面的代码只是为了帮你理解概念。
## Wanxiang: image and video generation

万相（Wanxiang）是 DashScope 旗下的生成式媒体家族。它覆盖文生图、图生视频和文生视频。所有媒体生成都走 DashScope 原生 API （不是 OpenAI 兼容接口），并且遵循异步任务模式。

![万象异步生成流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/10-bailian-llm/10_wanxiang_pipeline.png)

![Async task pattern for media generation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/10-bailian-llm/10_async_pattern.png)

### The async task pattern

每次调用万相，都是同样的三步走：

1. **Create the task.** POST 请求带上 header `X-DashScope-Async: enable`。你立刻就能拿到一个 `task_id`。
2. **Poll.** GET `/api/v1/tasks/{task_id}` 直到 `task_status` 变成 `SUCCEEDED` 或者 `FAILED`。
3. **Download.** 成功响应里会包含一个 URL。**必须在 24 小时内下载**——过期后 URL 直接返回 404，媒体文件永久丢失。

24 小时过期是运维上最大的坑。我见过好几个团队——包括我自己——因为轮询后记录了 URL，结果因为别的 bug 没及时下载，第二天才发现链接废了。要把这个 URL 当成一次性下载链接处理：立刻下载，存到自己的 OSS，别指望它明天还在。

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

模式一样，模型和输入参数不同：

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

两个模型都限制在 5 秒。想要 10 秒？生成两段 clips 然后拼接——把第一段最后一帧作为第二段的 `img_url` 输入。

万相视频生成的完整深度解析，见 [Bailian Part 4: Wanxiang Video Generation](/zh/aliyun-bailian/04-wanxiang-video-generation/)。

### Text-to-image

生图用的 endpoint 稍微不一样，但异步模式没变：

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

轮询还用那个 `poll_task()` 函数。成功返回里是 `output.results[0].url` 而不是 `output.video_url` —— 这点小不一致，适配一下就行。

## Qwen TTS: text-to-speech

Qwen TTS 是容易让人踩坑的地方。很多人想当然觉得“既然 Qwen LLM 能通过 OpenAI 客户端跑， TTS 肯定也行”。

> **Qwen-TTS 不走 OpenAI 兼容接口。仅限 DashScope 原生。**

你不能指着 `openai` SDK 的 `audio.speech.create` 去调兼容 URL，行不通。 TTS 没有兼容层。要么用 `dashscope` SDK，要么直接调 HTTP。

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

模型支持 40 多种音色。这是我实际在用的几个：

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

音色名区分大小写。`Cherry` 能用，`cherry` 不行。

### Streaming TTS for real-time playback

文本比较长或者要做实时应用，就得流式传输音频：

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

这里是 Qwen TTS 真正没有对手的地方。除了普通话和英语，它还支持粤语、四川话、上海话、东北话、日语和韩语——而且听起来像 native speaker，不像游客读短语手册。我没见过别的 TTS API 能把粤语处理得这么好，还是这个价格。

Qwen TTS 的完整深度解析包括声音克隆和 instruct 模式，见 [Bailian Part 5: Qwen TTS](/zh/aliyun-bailian/05-qwen-tts-voice/)。
## 在百炼上进行微调

微调是最后的杀手锏。在决定用它之前，先问问自己：提示词工程、 Few-shot 示例或者 RAG 能不能解决问题？根据我的经验， 80% 喊着“我们需要微调”的讨论，最后都以“其实换个更好的 System Prompt 就搞定了”收场。

![百炼平台概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/10-bailian-llm/10_bailian_platform.png)

### 什么时候才真的需要微调

| 场景 | 为什么微调有用 | 优先尝试的替代方案 |
|---|---|---|
| 模型总是搞错领域专有名词 | 训练数据能教会模型正确的术语 | 在 Prompt 中加入 Few-shot 示例 |
| 输出格式必须严格一致（例如始终返回带特定标签的 XML） | 微调能把格式固化到模型权重里 | JSON mode + 结构化 Prompt |
| 大规模调用时的成本优化 | 微调后的 `qwen-turbo` 在特定任务上能达到 `qwen-plus` 的效果 | 测算成本差异是否真的显著 |
| 降低延迟 | 更小的微调模型运行更快 | 压缩 Prompt，缩短 System Prompt |
| 语气风格保持一致 | 模型能学会你的品牌语调 | 在 System Prompt 中写入详细的风格指南 |

### 准备训练数据

百炼要求 JSONL 格式，结构跟标准的 Chat Completion 一样：

```jsonl
{"messages": [{"role": "system", "content": "You are a product description writer for electronics."}, {"role": "user", "content": "Write a description for: Sony WH-1000XM5 headphones"}, {"role": "assistant", "content": "Premium wireless noise-cancelling headphones with 30-hour battery life..."}]}
{"messages": [{"role": "system", "content": "You are a product description writer for electronics."}, {"role": "user", "content": "Write a description for: Apple AirPods Pro 2"}, {"role": "assistant", "content": "True wireless earbuds with adaptive noise cancellation..."}]}
```

写好训练数据的几条规矩：

- **最少 50 条示例**， 200-500 条是甜蜜点。除非你的领域特别杂，否则超过 1000 条提升不大。
- **所有示例的 System Prompt 必须一致** —— 模型会把 System Prompt 当作任务定义的一部分来学习。
- **只要高质量输出** —— 每条 Assistant 回复都必须是你想要模型产出的完美结果。一条坏数据就能教坏模型。
- **输入要多样化** —— 别拿同一个问题变着花样重复。要覆盖生产环境中可能遇到的所有输入情况。
- **上传前务必验证 JSONL** —— 只要有一行格式错了，整个任务就会静默失败。

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

### 启动微调任务

微调可以通过百炼控制台或者 API 进行：

1. **上传训练数据**：在百炼控制台的数据管理模块上传
2. **创建微调任务**：选择基座模型（例如 `qwen-turbo`），指向你的数据集，配置超参数
3. **监控训练**：控制台会显示 Loss 曲线和训练进度
4. **部署**：训练完成后，部署模型以获得自定义的 `model_id`

通过 API （使用 `dashscope` SDK）：

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

### 成本对比：微调小模型 vs 提示词工程大模型

算笔账就知道微调值不值了：

| 方案 | 模型 | 输入成本/1M | 输出成本/1M | 典型 Prompt Token 数 | 每月百万次调用成本 |
|---|---|---|---|---|---|
| 提示词工程 | `qwen-plus` | 0.8 | 2.0 | 800 (长 System Prompt + Few-shot) | ~2,240 CNY |
| 提示词工程 | `qwen-max` | 2.4 | 9.6 | 800 | ~7,680 CNY |
| 微调后 | `qwen-turbo` (自定义) | ~0.6 | ~1.2 | 200 (短 Prompt，无需 Few-shot) | ~360 CNY |

微调后的 Turbo 模型成本比用提示词工程的 Plus 模型低 6 倍，比 Max 模型低 21 倍。原因在于提示词更短（不需要 Few-shot 示例，行为已经固化在权重里），而且 Turbo 的单价更低。但微调本身也要花钱（训练算力）和时间（准备数据、验证质量、监控漂移）。只有当特定明确任务的月调用量超过 10 万次时，才值得这么做。

## 解决方案：多模态 AI 流水线

我把这些整合一下。下面是一个完整的流水线：输入一个主题，生成文章草稿，创建插图，再产出语音解说——全部用 Python 编排。

![多模态 AI 流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/10-bailian-llm/10_multimodal_flow.png)

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

大概 120 行 Python 代码。它调用了 DashScope 的三种能力（兼容 OpenAI 接口的文本生成、原生异步接口画图、原生同步接口 TTS），产出三个文件。生产环境还得加错误处理、重试逻辑和并行执行（画图和 TTS 互不依赖，可以并发）。但骨架都在这儿了。

多模态能力包括视频理解等内容，详见 [百炼系列第三篇：Qwen-Omni](/zh/aliyun-bailian/03-qwen-omni-multimodal/)。
## API 限流与错误处理

上线前得先把限流摸清楚：

| 模型系列 | 默认 RPM (requests/min) | 默认 TPM (tokens/min) | 能提额吗？ |
|---|---|---|---|
| `qwen-turbo` | 500 | 500K | 可以，提工单 |
| `qwen-plus` | 300 | 300K | 可以 |
| `qwen-max` | 120 | 120K | 可以 |
| `qwen3-max` | 120 | 120K | 可以 |
| `text-embedding-v3` | 500 | 500K | 可以 |
| `wan2.5-t2v-plus` | 20 | N/A | 可以 |
| `qwen3-tts-flash` | 180 | N/A | 可以 |

一旦触限， DashScope 会返回 HTTP 429 并带上 `Retry-After` 头。这么处理：

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

### 常见错误码

| HTTP 状态 | DashScope 代码 | 含义 | 解决办法 |
|---|---|---|---|
| 400 | `InvalidParameter` | 请求体有误 | 对照文档检查请求参数 |
| 401 | `InvalidApiKey` | API key 错误或过期 | 去百炼控制台重新生成 key |
| 404 | `ModelNotFound` | 模型 ID 拼写错误或模型不可用 | 核对 `model_id` 字符串 |
| 429 | `Throttling` | 超出限流 | 指数退避重试，或申请提升配额 |
| 500 | `InternalError` | 服务端问题 | 5-10 秒后重试 |

### 预算告警

去百炼控制台设个预算告警。我就吃过一次亏，有人忘了关调试循环跑了一整夜，账单直接四位数。要是当时有告警， 30 分钟就能发现，而不是 8 小时后。

```bash
# Quick check: your current month's usage via CLI
curl -s "https://dashscope.aliyuncs.com/compatible-mode/v1/models" \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" | python3 -m json.tool
```

## 放在全局看：全栈架构视角

看看 DashScope 在典型的阿里云架构里处在什么位置：

| 层级 | 服务 | 文章 |
|---|---|---|
| Compute | ECS, Function Compute | [Part 2](/zh/aliyun-fullstack/02-ecs-compute/), [Part 8](/zh/aliyun-fullstack/08-serverless/) |
| Networking | VPC, SLB | [Part 3](/zh/aliyun-fullstack/03-vpc-networking/) |
| Search & Retrieval | OpenSearch + embeddings | [Part 9](/zh/aliyun-fullstack/09-opensearch/) |
| **AI / LLM** | **DashScope (本文)** | **Part 10** |
| Storage | OSS (用于媒体资产) | [Part 1](/zh/aliyun-fullstack/01-ecosystem-map/) |

典型的 AI 应用流程：

1. 用户请求发到你的 API （跑在 ECS 或 Function Compute 上）
2. 你的应用通过 DashScope 调用 `text-embedding-v3` 嵌入查询
3. 用这些嵌入向量去 OpenSearch 搜相关上下文
4. 通过 DashScope 调用 `qwen-plus`，带上检索到的上下文和用户查询
5. 响应流式回传给用户
6. 如果需要媒体内容，异步调用 Wanxiang 并把结果存到 OSS

这个栈里的每一块本系列都会讲到。 DashScope 是大脑，其他服务是躯干。

## 核心要点

1. **百炼是控制台， DashScope 是 API。** 配置在百炼，代码对接 DashScope。别搞混了。

2. **默认用 OpenAI 兼容端点。** `base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"` 配合 `openai` SDK 能搞定文本、嵌入、视觉和多模态。只有 TTS、生图和生视频才切原生 API。

3. **默认选 `qwen-plus`。** 只有评测证明 Plus 不够用时再上 `qwen3-max`（带思考）。只有当体量上来成本敏感时再降级到 `qwen-turbo`。

4. **Qwen3 思考模式必须流式输出。** `enable_thinking=True` 不加 `stream=True` 会直接报错。这点坑过不少人。

5. **TTS 只能用 DashScope 原生接口。** 别拿 OpenAI 兼容端点调 `qwen3-tts-flash`，会 404。

6. **所有媒体生成都是异步的。** 提交任务、轮询、 24 小时内下载。链接 24 小时过期是生产环境最常见的事故。

7. **微调是最后手段。** 先试提示词工程、 Few-shot 和 RAG。只有当你每月有 10 万 + 请求量，且任务明确，小模型微调能媲美大模型长 prompt 时，再考虑微调。

8. **设预算告警。** 现在就设，别等有人忘了关调试循环跑了一整夜。

## 下一篇

[Part 11](/zh/aliyun-fullstack/11-security/) 讲阿里云上的安全： RAM 策略、 KMS 密钥管理、安全中心和 WAF。本文涉及的每个 API Key、每次 DashScope 调用、每个 OSS Bucket 都需要妥善的安全保障——这也是我们接下来的方向。