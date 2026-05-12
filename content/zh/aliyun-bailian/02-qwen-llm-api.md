---
title: "阿里云百炼（二）：Qwen API 生产接入"
date: 2026-02-26 09:00:00
tags:
  - Aliyun Bailian
  - Qwen
  - LLM
  - Function Calling
  - Streaming
categories: 阿里云百炼
lang: zh
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 2
description: "怎么按延迟和成本挑 Qwen 变体、function calling 写对、JSON mode 不再哭，以及 enable_thinking 必须配流式这条文档没明说的事。"
disableNunjucks: true
translationKey: "aliyun-bailian-2"
---
这个系列的干货集中在本篇。尽管其他模型有趣，我在生产环境中几乎只用 Qwen。虽然官方文档详尽但庞杂，本文将为你提炼出一条最短、最省、最稳的落地路径。

![Aliyun Bailian (2): The Qwen LLM API in Production — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/illustration_1.png)

## 选对适合工作负载的 Qwen  variant

Qwen 家族很大。很多团队默认全用 `qwen-max`，结果钱花多了；有的默认全用 `qwen-turbo`，结果质量差了。正确的策略是‘按需选型’：

![Qwen model family](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/fig1_qwen_family.png)

我在生产环境里的经验法则：

- **`qwen-turbo`** — 分类、意图识别、短文本摘要，或者任何单次用户请求里调用超过 10 次的场景。这是成本最低的 Qwen 变体，在分类、意图识别等信息抽取任务中表现出色。
- **`qwen-plus`** — 日常聊天、 RAG 合成、多步推理的主力模型。性价比最高的主力模型。
- **`qwen-max` / `qwen3-max`** — 代码 Review、复杂推理，任何“出错代价高于慢一点代价”的场景。
- **`qwen3-coder-plus`** — 所有代码任务。哪怕参数量级一样，它写代码也比通用的 `qwen-plus` 强出一截。
- **`qwen3-vl-plus` / `qwen3-omni-flash`** — 输入图片、视频、音频。第三篇文章专门讲这个。

> **提示：** 一个常见错误是用 `qwen-max` 做 embedding 式的分类。不建议这样做。用 `qwen-turbo` 配上紧凑的 system prompt，成本可降低约 90%，而在只需要标签的任务上质量毫无损失。

## 实际上 wire 上传的是什么

无论你使用 OpenAI 兼容层还是 DashScope 原生接口，chat-completion 请求的核心都是一个模型 ID、一个 messages 数组和一个参数块。

![Chat completion request flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/fig2_request_flow.png)

你最常调整的参数包括：

- **`messages`** — `{role, content}` 的数组。 Role 是 `system` / `user` / `assistant` / `tool`。官方文档提到，多模态模型的 `content` 可以是 typed parts 的数组（text, image_url, input_audio, video_url）— 见第三篇。
- **`temperature`** — 0.0-2.0。信息提取和分类任务使用 0.0，日常对话使用 0.2–0.4，仅在创意写作等需要高发散性的场景才使用 0.7 以上。官方文档将默认值设为约 0.7，但在多数 Agent 场景中该值偏高。
- **`top_p`** — 除非你清楚为什么要改，否则留默认。同时 tweaking `temperature` 和 `top_p` 只会让你更困惑。
- **`max_tokens`** (compat) / **`parameters.max_tokens`** (native) — 这是 *输出* token 上限，不是总数。必须显式设置。否则可能触发失控生成（runaway generation），导致意外的高额费用。
- **`stream`** — 开关 SSE streaming。见下文。
- **`response_format={"type": "json_object"}`** — JSON mode。强烈建议使用，优于“please return JSON”这样的提示。
- **`tools`** / **`tool_choice`** — 函数调用。

## 函数调用：往返过程

Qwen 的函数调用协议兼容 OpenAI 的 tool-calls 协议，整个过程包含两次 LLM 调用，中间由你的业务代码衔接：

![Function calling round-trip](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/fig3_function_calling.png)

一个完整的工作示例：一个支持天气查询的轻量级 Agent。

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

三个容易踩的坑：

- **`messages.append(msg)` 是必须的**，在第一次响应和 tool result 之间。模型需要在对话历史中看到自己生成的 tool_call 消息，否则第二次调用会报 400，说是"orphan tool result"。
- **`tool_choice="auto"`** 是默认值。当你必须强制指定某个 tool 时，用 `tool_choice={"type": "function", "function": {"name": "..."}}` — 在工作流的第一次调用里很有用。
- **`parallel_tool_calls=True`** 是支持的。当你有独立的 tools 时用这个 — 模型会一次性返回多个 `tool_calls`。

## JSON mode

想要结构化输出，不要依赖 prompt。直接使用：

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

生产环境里的两个注意点：

- 模型有时候还是会用 ```` ```json ```` 把 JSON 包起来。 stripping fences 之后再用 `json.loads` 防御性解析是明智的。
- 对于结构化的 JSON（如 Pydantic schema），建议使用函数调用模式。这种方式更严格，且失败模式更容易调试。

## enable_thinking 和 streaming 陷阱

Qwen3 系列模型支持 **`enable_thinking=True`** — 它让模型在最终答案之前先生成推理链。质量会提升，尤其是重推理的任务。**但你必须用 streaming。** 若未启用流式响应（streaming）， API 将直接返回 400 错误。

![enable_thinking + streaming](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/fig4_thinking_streaming.png)

实用模式 — 把推理内容收集到侧边日志，把答案 stream 到你的 UI：

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

我把 `reasoning` 转发给我的日志系统，绝不给用户看。三个原因：(1) 如果客户看到会泄露 chain-of-thought IP，(2) 可能让非技术读者困惑，(3) 这会使用户可见的响应长度增加一倍。

## 异步聊天（少见但有用）

如果你有运行时间很长的聊天（比如 30k token 的 RAG 合成），可以用 `X-DashScope-Async: enable` 提交异步任务然后 poll，模式跟 Wanxiang 一样。 Qwen API 参考文档在"Asynchronous calling"下记录了这点。我用它来做 cron-batch 摘要任务，这种任务不需要立即给用户响应。

## 真正管用的成本控制

- **永远设 `max_tokens`。** 默认上限是"model max"，意味着一个 runaway loop 能让你花一大笔钱。
- **每个环境用独立的 workspace key。** 在控制台 workspace 下给 prod key 设硬性的每日预算。
- **记录 token 数。** 每个响应里都有 `usage.prompt_tokens` 和 `usage.completion_tokens`。每周聚合一次，你就能发现那个没人注意但 bloated 了 3 倍的 prompt。
- **在你的 edge 缓存相同的 prompt。** DashScope 目前不像 Anthropic 那样暴露 prompt caching — 所以对于高 volume 的 identical-prefix 模式，自己缓存。

## Token 计数： DashScope vs tiktoken，以及 CJK 膨胀

从 OpenAI 生态过来的团队最大的 surprise：**`tiktoken` 关于 Qwen token 数的说法是假的**。 Qwen tokenizer 跟 `cl100k_base` 或 `o200k_base` 不兼容 BPE。如果你用 `tiktoken.encoding_for_model("gpt-4o")` 来 sizing 你的 context budget，中文会偏 20-40%，英文会偏 5-10%。我曾因此在周五晚间调试至深夜：一条 RAG pipeline 按 tiktoken 估算‘明确低于 32k 上下文’，但实际使用 Qwen tokenizer 计算却达到 41k。

正确的做法是在本地用官方 Qwen tokenizer：

```python
# pip install transformers tiktoken
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

def count_qwen_tokens(text: str) -> int:
    return len(tok.encode(text))

print(count_qwen_tokens("阿里云百炼 DashScope SDK"))   # 9
# tiktoken o200k_base on the same string gives 11
```

Tokenizer 打包在任何 Qwen 模型的 Hugging Face repo 里 — `Qwen2.5-7B`, `Qwen3-7B` 等共享一个 tokenizer 家族，所以加载任何一个都能给你一个跟 DashScope 计费误差在 ±1 token 以内的计数。事后来看，响应里的 `usage.prompt_tokens` 和 `usage.completion_tokens` 是权威的；当你有它们时，信它们别信本地估算。

CJK 膨胀问题确实存在，需要计入成本。在 Qwen 上，典型的中文字符每个字大约占用 1.5 个 token，而英文每个字符大约占用 0.25 个 token（每 4 个字符 1 个 token）。因此，1000 字符的中文 RAG context 需要 1500 个 token，而同样长度的英文只需 250 个 token。规划上下文窗口时，应按 token 计划而非字符，并使用本地 tokenizer。我曾见过一个“使用 100k 字符的上下文”的计划最终变成了“我们需要一个 150k token 的上下文模型”。
## 流式传输与背压： drain 模式与部分 JSON

之前那种 naive 的流式代码，跑个 CLI demo 没问题。但要是上生产环境的 HTTP 服务，立马暴露两个问题：背压（backpressure，下游处理速度跟不上模型生成速度）和部分解析（partial parsing，用户想要结构化输出，但你的 buffer 里 token 还没收完）。

**背压力**：当你把流式数据块转发给慢客户端（比如 4G 网络下的手机浏览器），这些数据块会在进程内存里堆积，直到 OOM 或者上游连接超时。解决办法是把上游数据 drain 到一个有界队列，并对客户端连接施加反压：

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

这里的 queue size （这里设了 32）就是你允许的 in-flight 缓冲深度。设得越小，背压响应越灵敏，但交付可能会稍显卡顿。 32 是我在公网上做 SSE-to-WebSocket 中继实测下来的经验值。

**部分 JSON**：如果输出是 JSON，且你想边收边渲染（比如实时更新表单），就不能等到流结束再 `json.loads`。 trick 是用流式 JSON 解析器，比如 `json-stream` 或 `partial_json_parser`：

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

这能实现“表单在用户眼前自动填充”的 UX，无需等待模型生成完毕。我们的营销工具里结构化提取接口就在用这个——感知延迟从 4 秒降到了 500 毫秒以内，虽然物理耗时没变。

## Function Calling 深挖：多轮、并行与 tool_choice="auto" 的坑

![Aliyun Bailian (2): The Qwen LLM API in Production — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/02-qwen-llm-api/illustration_2.png)

原文里那个基础往返流程只能应付简单场景。真正的 Agent 是循环执行的。模式就是一个 `while` 循环，直到模型不再 emit `tool_calls`：

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

文档里有三点没讲清楚：

**`tool_choice="auto"` 的死循环陷阱**。设为 `auto` 时，模型自己决定是调工具还是直接回答。在 Qwen-Plus 上，我多次遇到它为了同一个城市连续四轮调用 `get_weather`，每次都觉得上次的答案不够。解决办法要么 (a) 加严格的 `max_rounds` 上限（我在用），(b) 第三轮后强制 `tool_choice="none"` 逼模型回答，要么 (c) 检测重复调用并短路。生产环境我选 (b)——给 Agent 三次调工具的机会，最后一次必须只回答。

**并行工具调用返回在同一个 assistant message 里**。开了 `parallel_tool_calls=True`，`msg.tool_calls` 就是一个包含多个调用的列表。你必须先 append 那个包含所有调用的 *单个* assistant message，然后为每个调用 append 一个 `tool` message，再发起下次请求。如果你在 assistant message 之前就直接 append 每个调用的 tool message，会报 orphan-tool-result 400 错误。

**工具参数 Schema：保持扁平且必填**。 Qwen 的工具调用在处理深层嵌套参数时，可靠性明显不如 GPT-4。如果一个工具的 schema 是 `{type: object, properties: {filter: {type: object, properties: {date: {...}, region: {...}}}}}`，它能写对 date，但有 15% 的概率忘掉 region。把参数扁平化到顶层必填项，可靠性能升到 >99%。这是我踩了一周坑才发现的——之前以为是“模型今天变笨了”，其实是"schema 嵌套太深”。

## enable_thinking 的细节：何时有益，何时有害，代价多少

`enable_thinking=True` 被宣传为“免费的质量提升”。既不免费，也不总是提升。在生产环境跑了半年不同负载后，我的分类如下：

**有益场景：**
- 多步推理（数学应用题、逻辑谜题、代码执行 trace）。
- 多约束分类（“用户是否请求了 X *且* Y *但非* Z"）。
- 多文件上下文的代码 Review。
- 任何你自己回答前也想写点中间步骤的任务。

**有害场景（或浪费）：**
- 纯提取（“从文本里抽这 5 个字段”）。推理链只是在复述输入。
- 简短事实查询。模型为一个单词的答案思考 2 秒。
- 高温度创意写作。推理会把风格拉向中性。
- 工具调用 Agent。推理内容会微妙地干扰工具调用决策——我见过开启 thinking 后调用直接丢了 `tool_choice` 信号。

**延迟代价：** TTFT （首字节时间）从 ~400ms 升到 ~1.5-3 秒，因为推理链必须在答案流式输出前完成。总 Token 数大概翻倍——即使你不把推理内容展示给用户，也得为此付费。对于聊天 UI，思考暂停期间的“卡住”感体验很差；我要么显示“思考中…" spinner，要么把推理内容流式输出到可折叠的侧边栏。

**我的决策原则：** 如果这任务换作我自己回答前也想写点笔记，我就开启 thinking。如果可以直接脱口而出，就关掉。对于 Agent 循环，除了最后的合成步骤，其他轮次我都关掉。

## 长上下文：缓存命中率与截断策略

Qwen-Plus 有 128k 上下文窗口。 Qwen-Max 默认 32k，也有 1M token 的长上下文变种。窗口大不代表就要填满。

我在第一章提到的隐式 prompt 缓存有个关键特性：**它基于精确前缀匹配**。如果每个用户的 system prompt 完全一致但 user message 不同， system prompt 前缀会被缓存。如果你在 prompt *中间* 放动态数据（比如“今天是 {date}"），每个变体都会破坏缓存。解决办法是把所有动态内容放在 messages 数组的 *末尾*——把动态日期/用户 ID/时间戳放在 user message 里，别放 system prompt。

我会监控每个端点的缓存命中率：

```python
total = sum(usage.prompt_tokens for usage in window)
cached = sum(getattr(usage, "cached_tokens", 0) for usage in window)
hit_rate = cached / total
print(f"cache hit rate over last {len(window)} calls: {hit_rate:.1%}")
```

结构良好的 RAG 端点配合稳定的 system prompt，命中率能达到 70-80%。如果把请求特定的元数据插进 system prompt，命中率就是 0%。账单上的输入 Token 费用会差两倍。

当确实超过窗口需要截断时：安全模式是“保留 system prompt 和最近的 user/assistant  pair，中间部分滑动窗口”。我保留第一条 system message 和最后 6 条消息原样，当对话超过阈值时，用便宜的 `qwen-turbo` 调用总结中间内容。总结内容作为合成 system message 放回 messages 数组（`"role": "system", "content": "Earlier in this conversation: ..."`）。对于聊天类负载，质量损失很小；但对于代码上下文负载，不能对有损压缩文件内容——这种情况下，优选更长上下文的模型而不是总结。

## 下一篇预告

第三篇是 **Qwen-Omni** —— 多模态兄弟模型。主要区别在于：流式是 *必须* 的（非可选）， content 数组需要为图片/音频/视频使用 typed parts，你还得考虑 pixel budgets 和帧率。如果你的产品涉及非文本内容，这是 Bailian 里杠杆率最高的能力。