---
title: "阿里云百炼实战（二）：Qwen 文本大模型在生产环境的用法"
date: 2026-02-26 09:00:00
tags:
  - 阿里云百炼
  - Qwen
  - LLM
  - Function Calling
  - Streaming
categories: 阿里云百炼
lang: zh-CN
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 2
description: "怎么按延迟和成本挑 Qwen 变体、function calling 写对、JSON mode 不再哭，以及 enable_thinking 必须配流式这条文档没明说的事。"
disableNunjucks: true
translationKey: "aliyun-bailian-2"
---
在这个系列文章中，本篇聚焦于实际生产中最能带来价值的内容。其他模型固然各有亮点，但真正让我在百炼平台上发布的每个产品都离不开的，是那些每天每分钟都在调用的大语言模型（LLM）。官方的 Qwen API 文档内容详实且全面，但稍显复杂；本文则从中提炼出一条清晰易懂的路径，帮助读者更好地理解和使用。

![阿里云百炼实战（二）：Qwen 大模型 API 在生产环境中的应用 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/illustration_1.jpg)
## 根据任务选择合适的 Qwen 变体

Qwen 系列模型种类繁多，但很多团队在使用时容易陷入两个极端：要么一股脑全用 `qwen-max`，导致成本居高不下；要么为了省钱全都选 `qwen-turbo`，结果质量跟不上。其实，正确的做法是“根据具体任务匹配合适的变体”。

![Qwen 模型家族](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/fig1_qwen_family.png)

以下是我在实际生产中总结的一些经验：

- **`qwen-turbo`** — 适合处理分类、意图识别、短文本摘要等任务，尤其是那些单次用户请求中需要调用超过 10 次的场景。这是 Qwen 家族中最便宜的选择，但在信息抽取类任务上表现非常出色，性价比极高。
- **`qwen-plus`** — 日常主力模型，适用于对话系统、RAG（检索增强生成）综合任务以及多步推理等场景。它在成本和质量之间找到了一个很好的平衡点。
- **`qwen-max` / `qwen3-max`** — 适合对精度要求极高的任务，比如代码审查、复杂逻辑推理等。这些场景下，回答错误的代价远高于回答速度慢的成本。
- **`qwen3-coder-plus`** — 所有与代码相关的任务都推荐使用这个变体。即使参数规模相同，它在代码生成和理解方面的表现也明显优于通用型的 `qwen-plus`。
- **`qwen3-vl-plus` / `qwen3-omni-flash`** — 专为图像、视频、音频等多模态输入设计。这部分内容会在第三篇文章中详细展开。

> **小贴士：** 一个常见的误区是用 `qwen-max` 来处理 embedding 类型的分类任务。其实完全没必要。使用 `qwen-turbo` 并搭配一个精简的 system prompt，不仅能把成本降低 10 倍，还能在只需要标签输出的任务中保持几乎相同的质量。
## 请求实际发送了什么内容

无论你是通过 OpenAI 兼容接口还是 DashScope 原生接口发起请求，一个聊天补全（chat-completion）的核心内容都是一样的：模型 ID、消息数组（messages），以及参数配置。

![Chat completion 请求流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/fig2_request_flow.png)

以下是开发者最常接触的几个关键字段：

- **`messages`** — 这是一个由 `{role, content}` 组成的数组。其中 `role` 可以是 `system`、`user`、`assistant` 或 `tool`。官方文档提到，对于多模态模型，`content` 还可以是一个包含多种类型内容的数组（例如文本、图片链接、音频输入、视频链接等），具体细节可以参考第三篇文章。
- **`temperature`** — 范围为 0.0 到 2.0。如果是做信息抽取或分类任务，我会将它设为 0.0；如果是普通的对话场景，通常设置为 0.2 到 0.4；而需要创意写作时，才建议调到 0.7 或更高。官方文档的默认值大约是 0.7，但对大多数智能体（agent）应用场景来说，这个值可能偏高。
- **`top_p`** — 如果你不确定为什么要调整这个参数，那就保持默认值即可。同时调整 `temperature` 和 `top_p` 很容易导致混乱，不建议这么做。
- **`max_tokens`**（兼容模式） / **`parameters.max_tokens`**（原生模式）— 这个参数限制的是*输出*的 token 数量，而不是总 token 数。务必设置一个合理的上限，否则模型可能会生成过多内容，导致费用失控。
- **`stream`** — 用于启用 SSE 流式传输。更多细节见下文。
- **`response_format={"type": "json_object"}`** — 开启 JSON 模式。相比在提示词中写“请返回 JSON”，这种方式更加可靠，强烈推荐。
- **`tools`** / **`tool_choice`** — 用于函数调用（function calling）。
## 函数调用：一来一回的完整流程

Qwen 的函数调用协议与 OpenAI 的工具调用协议完全一致。整个过程可以概括为两次大模型调用，中间插入一段你的代码逻辑：

![函数调用流程图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/fig3_function_calling.png)

下面是一个完整的可运行示例——一个能够查询天气的小型智能体：

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
        "description": "获取某城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

def call_weather(city):
    # 实际场景中，这里会调用你的 API。此处仅作模拟。
    return {"city": city, "temp_c": 22, "conditions": "晴天"}

messages = [{"role": "user", "content": "去上海需要带伞吗？"}]
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

在实际使用中，有三个容易忽略但非常重要的细节需要注意：

- **必须在第一次响应和工具结果之间调用 `messages.append(msg)`。**  
  模型需要从对话历史中看到自己生成的 `tool_call` 消息，否则第二次调用时会返回 400 错误，提示“孤立的工具结果”（orphan tool result）。

- **默认情况下，`tool_choice="auto"`。**  
  如果需要强制指定某个工具，可以通过 `tool_choice={"type": "function", "function": {"name": "..."}}` 来实现。这种设置在工作流的首次调用中尤为常见。

- **支持 `parallel_tool_calls=True`。**  
  当你有多个独立的工具需要并行调用时，可以启用此选项。启用后，模型会在一次调用中返回多个 `tool_calls`，从而提升效率。
## JSON 模式

如果需要生成结构化的输出，不要单纯依赖提示词来要求模型返回 JSON。推荐直接使用以下方式：

```python
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": '请返回 JSON 格式：{"sentiment": "positive|negative|neutral"}'},
        {"role": "user", "content": "我非常喜欢这个产品。"},
    ],
    response_format={"type": "json_object"},
)
data = json.loads(resp.choices[0].message.content)
```

在实际生产环境中，有两点需要注意：

- 模型有时仍然会用 ```` ```json ```` 包裹返回的 JSON 内容。为了确保解析的稳定性，建议先去掉这些围栏，再用 `json.loads` 进行解析。
- 如果你需要更严格的结构化输出（例如配合 Pydantic Schema），建议改用函数调用模式。这种方式约束更强，调试起来也更加方便。
## 启用思维链与流式输出的注意事项

Qwen3 系列模型支持 **`enable_thinking=True`**，这个参数会让模型在生成最终答案之前先输出一段推理过程。对于需要复杂推理的任务，这种机制能够显著提升结果的质量。不过需要注意的是，**必须启用流式输出**，否则会直接返回 400 错误。

![启用思维链 + 流式输出](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/fig4_thinking_streaming.png)

实际使用时，推荐的做法是将推理过程记录到日志中，而只把最终答案逐步流式地展示给用户界面：

```python
stream = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "如果一只表每天慢 5 分钟，10 天后会差多少？"}],
    extra_body={"enable_thinking": True},
    stream=True,
)

reasoning, answer = [], []
for chunk in stream:
    delta = chunk.choices[0].delta
    rc = getattr(delta, "reasoning_content", None)
    if rc:  reasoning.append(rc)
    if delta.content: answer.append(delta.content)

print("最终答案:", "".join(answer))
print("(推理过程已隐藏，共", sum(len(r) for r in reasoning), "字符)")
```

我通常会把 `reasoning` 部分发送到日志系统，而不会展示给终端用户。这样做的原因有三点：(1) 如果用户看到推理过程，可能会泄露 chain-of-thought 的知识产权；(2) 推理内容对非技术背景的用户来说难以理解，容易造成困惑；(3) 展示推理链会让响应内容的长度翻倍，影响用户体验。
## 异步聊天（不常见但很有用）

如果你需要处理一个非常长的对话任务，比如生成包含 30k token 的 RAG 结果，可以通过设置 `X-DashScope-Async: enable` 来异步提交请求，并通过轮询获取结果。这种方式和万相的处理模式类似。在 Qwen API 文档的“异步调用”部分可以找到相关说明。我通常会用这种方法来执行一些定时的批量摘要任务，这些任务并不需要立即返回用户可见的结果。
## 真正有效的成本控制方法

- **务必设置 `max_tokens`。** 默认值是“模型最大”，如果不设限，一旦出现失控循环，可能会让你付出高昂代价。
- **为每个环境单独配置工作空间 key。** 在控制台中为生产环境的 key 设置严格的每日预算，避免意外超支。
- **记录 token 使用量。** 每次 API 响应都会包含 `usage.prompt_tokens` 和 `usage.completion_tokens`。每周汇总一次数据，就能及时发现那些不知不觉膨胀了三倍的 prompt。
- **自行缓存高频重复请求。** 目前 DashScope 并未像 Anthropic 那样提供内置的 prompt 缓存功能——因此，对于高 QPS 的同前缀请求场景，建议在边缘层自行实现缓存逻辑。
## Token 计数：DashScope 与 tiktoken 的差异，以及 CJK 膨胀问题

对于从 OpenAI 生态迁移到阿里云的团队来说，最大的意外莫过于：**`tiktoken` 在计算 Qwen 的 token 数时并不准确。** Qwen 的分词器（tokenizer）与 `cl100k_base` 或 `o200k_base` 并不兼容，无法直接套用 BPE 算法。如果你用 `tiktoken.encoding_for_model("gpt-4o")` 来估算上下文长度，中文内容的误差可能达到 20%-40%，而英文内容的误差则在 5%-10% 左右。我就曾经因为这个问题，在一个周五晚上折腾到深夜：明明用 tiktoken 算出来“绝对不超过 32k 上下文”的 RAG 流水线，结果用 Qwen 的分词器一算，竟然超到了 41k。

正确的做法是使用 Qwen 官方提供的本地分词器：

```python
# pip install transformers tiktoken
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

def count_qwen_tokens(text: str) -> int:
    return len(tok.encode(text))

print(count_qwen_tokens("阿里云百炼 DashScope SDK"))   # 9
# 同样的字符串，tiktoken 的 o200k_base 算出来是 11
```

无论你用的是 `Qwen2.5-7B` 还是 `Qwen3-7B`，这些模型在 Hugging Face 仓库中都自带分词器，并且共享同一个分词器家族。加载任意一个模型的分词器，得到的 token 计数都会和 DashScope 实际计费的数值相差不超过 ±1 token。当然，最权威的还是 API 响应中的 `usage.prompt_tokens` 和 `usage.completion_tokens`，如果能拿到这些数据，就以它们为准。

CJK 膨胀问题确实存在，必须在预算中考虑进去。一般来说，中文句子平均每字占用约 1.5 个 token，而英文每字仅需约 0.25 个 token（也就是大约 4 个字符才消耗 1 个 token）。这意味着，一个包含 1000 字的中文 RAG 上下文会消耗 1500 个 token，而同样长度的英文内容只需 250 个 token。因此，在规划上下文窗口时，**一定要按 token 来计算，而不是按字符数估算**，并且务必使用本地分词器进行校验。我亲眼见过一个“计划使用 100k 字符上下文”的项目，最终发现需要支持 150k token 的模型——一次教训就够了。
## 流式处理与背压：Drain 模式与部分 JSON 解析

最简单的流式代码就像我们之前提到的那样——遍历流、拼接内容，然后结束。这种实现方式在命令行工具（CLI）的演示中完全够用。但在生产环境的 HTTP 服务中，你还需要面对两个额外的问题：**背压**（下游处理速度跟不上模型输出速度）和**部分解析**（用户需要结构化输出，但缓冲区的数据可能只包含不完整的 token）。

### 背压问题

当你将流式数据转发给一个慢速客户端时（比如运行在 4G 网络上的手机浏览器），数据块会在你的进程内存中堆积，最终可能导致内存耗尽（OOM）或上游连接超时。解决这个问题的关键是使用一个有界队列来“吸收”上游数据，并通过反向压力控制客户端的接收速度：

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
                await queue.put(d)  # 队列满时阻塞，形成背压
        await queue.put(None)

    async def consumer():
        while True:
            d = await queue.get()
            if d is None: break
            await send_to_client(d)  # 如果客户端较慢，队列会填满，producer 将被阻塞

    await asyncio.gather(producer(), consumer())
```

这里的队列大小（32）决定了允许的飞行缓冲区深度。较小的队列可以让背压更灵敏，但可能会导致数据下发略微卡顿。经过实际测试，32 是一个适合在公网环境下进行 SSE 到 WebSocket 转发的值。

### 部分 JSON 解析

如果输出是 JSON 格式，并且你希望在数据到达时逐步渲染（例如动态填充表单），就不能等到整个流结束再调用 `json.loads`。这时可以借助流式 JSON 解析器，比如 `json-stream` 或 `partial_json_parser`，来实现边接收边解析的功能：

```python
# pip install partial-json-parser
from partial_json_parser import loads as partial_loads

buffer = ""
last_render = None
for chunk in stream:
    buffer += chunk.choices[0].delta.content or ""
    try:
        partial = partial_loads(buffer)  # 尽可能解析当前内容，缺失的部分用 None 填充
        if partial != last_render:
            render_to_ui(partial)
            last_render = partial
    except Exception:
        continue  # 数据还不足以构成有效前缀，跳过本次处理
```

这种方式能够实现“表单在用户眼前逐步填充”的交互体验，而无需等待模型完成所有输出。我们在营销工具的结构化抽取接口中采用了这种方法，用户的感知延迟从 4 秒降低到了 500 毫秒以内，尽管实际的总耗时并没有变化。
## 深入解析 Function Calling：多轮对话、并行调用与 `tool_choice="auto"` 的陷阱

![阿里云百炼实战（二）：Qwen 文本大模型在生产环境的用法 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/illustration_2.jpg)

上一节提到的简单 round-trip 只能应对最基础的场景。然而，实际应用中的智能体（agent）往往需要通过循环来完成任务。这种模式通常表现为一个 `while` 循环，直到模型不再生成 `tool_calls` 为止。

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
        # 并行调用：一次性分发所有工具调用
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = dispatch(call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    raise RuntimeError("智能体未能收敛")
```

文档中没有明确提及的三个关键点：

**`tool_choice="auto"` 导致的死循环问题。** 当设置为 `auto` 时，模型会自行决定是否调用工具或直接回答问题。在使用 Qwen-Plus 时，我多次观察到它对同一个城市连续调用了 4 次 `get_weather`，每次都认为之前的回答不够充分。解决方法有三种：(a) 设置严格的 `max_rounds` 上限（这是我常用的做法）；(b) 在第 3 轮之后强制将 `tool_choice` 设为 `"none"`，迫使模型直接给出答案；(c) 检测重复的工具调用并提前终止。在生产环境中，我选择了方案 (b)——给智能体 3 次调用工具的机会，最后必须给出一个直接的答案。

**并行工具调用的结果会包含在一条 assistant 消息中返回。** 当启用 `parallel_tool_calls=True` 时，`msg.tool_calls` 会是一个包含多个工具调用的列表。你需要先追加这条包含所有工具调用的 assistant 消息，然后为每个工具调用分别追加一条 `tool` 消息，最后再发起下一轮请求。如果直接逐个追加工具消息而没有插入对应的 assistant 消息，系统会报错 400，提示 orphan tool result。

**工具参数的 schema 应尽量扁平化，并且字段应设为 required。** Qwen 的工具调用功能在处理深层嵌套参数时，可靠性明显低于 GPT-4。例如，一个工具的 schema 定义为 `{type: object, properties: {filter: {type: object, properties: {date: {...}, region: {...}}}}}`，其中 date 字段通常能正确生成，但 region 字段大约有 15% 的概率会被遗漏。如果将参数拍平到顶层，并将其标记为 required，可靠性可以提升至 99% 以上。这个教训是我经历了一周“今天模型怎么这么笨”的抱怨后才总结出来的——其实问题出在 schema 太过复杂和嵌套。
## enable_thinking 的细节：什么时候有用，什么时候没用，代价是什么

`enable_thinking=True` 被宣传为“免费提升质量”的功能。但其实它既不免费，也不总是能提升效果。在生产环境中针对不同任务运行了半年后，我总结了一些经验：

**适合开启的场景：**
- 涉及多步推理的任务，比如数学应用题、逻辑谜题、代码执行路径分析。
- 需要处理多个约束条件的分类问题，例如“用户是否同时要求 X *和* Y *但不要* Z”。
- 代码审查，特别是需要结合多个文件上下文的情况。
- 任何你自己会想先列个提纲或打个草稿再回答的问题。

**不适合甚至有害的场景：**
- 纯粹的信息抽取任务（比如“从这段文本中提取 5 个字段”）。在这种情况下，推理链只是简单地复述输入内容，并没有实际帮助。
- 简单的事实查询。模型可能会花两秒钟思考一个单字答案，显得效率低下。
- 高温（high-temperature）创意写作。推理过程会让风格趋于平淡，失去原本的创造力。
- Tool-calling agent（工具调用代理）。推理内容可能会以微妙的方式干扰工具调用决策——我曾遇到过开启 thinking 后直接丢失 `tool_choice` 信号的情况。

**延迟成本：**  
TTFT（首字节响应时间）从约 400 毫秒增加到 1.5-3 秒，因为推理链必须先完成，答案才能开始流式传输。总的 token 数量大约翻倍——即使推理内容不会展示给用户，你依然需要为此付费。在聊天界面中，这种“思考暂停”会让用户感觉系统卡住了，体验很差。我的解决办法是，要么显示一个“正在思考…”的加载动画，要么将推理内容流式输出到一个可折叠的侧边栏。

**如何判断是否开启：**  
如果这个任务是我自己也会先写点笔记再回答的类型，那就开启 thinking；如果是我可以脱口而出回答的，就关闭。对于多轮对话的 Agent，我通常在每一轮都关闭 thinking，只在最终综合回答时开启。
## 长上下文处理：缓存命中率与截断策略

Qwen-Plus 的上下文窗口支持高达 128k，而 Qwen-Max 默认是 32k，同时提供支持 1M tokens 的长上下文版本。不过，上下文窗口大并不意味着你应该把它塞得满满的。

在第一章提到的隐式 prompt 缓存机制中，有一个关键特性：**它基于精确的前缀匹配进行缓存**。如果你的系统提示（system prompt）对所有用户都是一样的，只有用户输入（user message）不同，那么系统提示部分会被缓存下来。但如果你在 prompt 的*中间*插入了动态内容（比如“今天是 {date}”），每次生成的变体都会导致缓存失效。解决办法是把所有动态内容移到 messages 数组的*末尾*——例如，将动态日期、用户 ID 或时间戳放在用户消息中，而不是嵌入系统提示里。

为了监控缓存效率，我会为每个接口（endpoint）统计缓存命中率：

```python
total = sum(usage.prompt_tokens for usage in window)
cached = sum(getattr(usage, "cached_tokens", 0) for usage in window)
hit_rate = cached / total
print(f"最近 {len(window)} 次调用的缓存命中率: {hit_rate:.1%}")
```

一个结构良好的 RAG 接口，如果系统提示保持稳定，其缓存命中率通常能达到 70%-80%。但如果把请求特定的元数据（metadata）拼接到系统提示中，命中率会直接降到 0%。这种情况下，输入 token 的费用可能会相差两倍。

当上下文长度超出窗口限制时，推荐的安全策略是：“保留系统提示和最近的一轮用户/助手对话，中间部分采用滑动窗口截断”。我的做法是保留第一条系统消息和最后 6 条对话原文，当中间内容超过某个阈值时，使用成本较低的 `qwen-turbo` 对超出部分进行总结。总结结果会以合成的系统消息形式插入到 messages 数组中（格式如 `"role": "system", "content": "此前对话内容: ..."`）。对于聊天类任务，这种方法的质量损失很小；但对于代码上下文类任务，影响可能较大——因为代码文件内容无法通过有损压缩来处理。针对这类场景，建议直接使用支持更长上下文的模型，而不是依赖总结。
## 接下来的内容

第三篇将聚焦于 **Qwen-Omni** —— 这是通义千问家族中负责多模态任务的成员。与之前介绍的内容相比，它有几个显著特点：首先，流式处理不再是可选项，而是*必须*采用的方式；其次，内容数组会根据图像、音频或视频等不同模态进行类型化处理；此外，你还需要特别关注像素预算和帧率等细节。如果你的产品涉及非文本内容（比如图片、音频或视频），那么 Qwen-Omni 无疑是百炼平台上最具价值的能力之一，能够为你带来极高的技术杠杆效应。