---
title: "阿里云百炼实战（二）：Qwen 文本大模型在生产环境的用法"
date: 2026-02-26 09:00:00
tags:
  - 阿里云百炼
  - DashScope
  - 大模型
  - Qwen
categories: 阿里云百炼
lang: zh-CN
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 2
description: "怎么选 Qwen 模型、enable_thinking 在非流式下必踩的坑、工具调用、JSON 模式，以及生产环境真实见到的错误模式。"
disableNunjucks: true
---

用户把一份 30 页的合同直接粘进我们应用，要一份结构化摘要。第一版用 `qwen-turbo` 跑，输出一本正经的胡说八道。第二版换成 `qwen-max`，准确率边际提升、成本翻 30 倍。第三版换成 `qwen3-max` + `enable_thinking=True`，12 秒返回正确结果，价格只有 `qwen-max` 的一半——只要早知道 Qwen3 的流式规则，第一次就能跑对。这一篇就是 Qwen 文本大模型族里那十几条决定生死的细节。

## 先选对模型，再去调 prompt

百炼的定价页列了三十多个 Qwen 变种。一年生产实践下来，我真用过的就这五个，够了：

| model_id | 适用场景 | 输入/输出大约 ¥/百万 token | 输出 500 token 的延迟 |
|---|---|---|---|
| `qwen-turbo` | 高吞吐分类、简单抽取 | 0.3 / 0.6 | ~600ms |
| `qwen-plus` | 默认款，对话、摘要、轻量推理 | 0.8 / 2 | ~1.2s |
| `qwen-max` | 最难的推理、长合同、绝对不能错的场景 | 2.4 / 9.6 | ~3s |
| `qwen3-max` | 推理新默认款；开 thinking 后比 qwen-max 更便宜 | 2.0 / 6 | 开 thinking 后 ~3-12s |
| `qwen3-coder-plus` | 代码生成、所有涉及 diff/patch/AST 的任务 | 1.0 / 4 | ~2s |

剩下的要么已弃用、要么是 Turbo 已经覆盖的小模型、要么是大多数应用永远用不到的特化版。**默认就用 `qwen-plus`。** 只在 eval 证明 Plus 不够时才升 `qwen3-max`，只在测过成本真的卡脖子时才降到 `qwen-turbo`。

上下文长度全族都很慷慨——plus/max 都是 128K token，需要的话 `qwen-turbo-longcontext` 还能上 1M——但长上下文成本是线性涨、延迟是超线性涨。如果你每次都往请求里塞 80K token 的 RAG 上下文，那是检索做错了，不是模型选错了。

## 默认走 OpenAI 兼容 SDK

本文所有代码默认用 OpenAI SDK 走兼容接口，特殊情况会另行说明。

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

`client` 在进程生命周期内复用就行。SDK 是线程安全的、复用连接池；每次重建会平白多花 50-100ms 的 TLS 握手。

## 流式基础

只要面向用户，就开流式。用户感知到的"快"是首 token 时间，监控里看到的"慢"是总延迟，这两件事是不同的问题。

```python
stream = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "写一首四句关于 TLS 的俳句。"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

两个常见的坑：

1. 最后一个 chunk 的 `delta.content` 是 `None`，但带 `finish_reason`。一定要 `if delta:` 判断。
2. 流式下想拿到 token 用量，必须传 `stream_options={"include_usage": True}`。不传的话最后一个 chunk 没有 `usage` 字段，你就不知道这次花了多少钱。

## enable_thinking 的坑（Qwen3 系列）

这就是亲手调试半天的那个 bug。Qwen3 系列模型（`qwen3-max`、`qwen3-coder-plus` 等）有一个 `enable_thinking` 参数，开了会让模型先做一段链式推理再输出最终答案——这正是 `qwen3-max` 能在更低价位达到 `qwen-max` 准确率的原因。但有一条硬规则：

> **`enable_thinking=True` 强制要求 `stream=True`。非流式调用必报错。**

为什么：模型会先输出大段 reasoning token（动辄几千个）放在一个 `<thinking>` 块里，然后再吐答案。兼容层期望你边到边消费，非流式下缓冲行为没定义，所以服务端宁可直接报错也不允许你静默截断。

错误写法：

```python
# 这样会返回 400，错误信息提到 thinking + stream
resp = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "求解：..."}],
    extra_body={"enable_thinking": True},
)
```

正确写法：

```python
stream = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "求解：..."}],
    stream=True,
    stream_options={"include_usage": True},
    extra_body={"enable_thinking": True},
)

answer_parts = []
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    # 兼容层把 reasoning 内容放在 delta.reasoning_content 里
    if getattr(delta, "reasoning_content", None):
        # 想保留就 log，想展示就放在 UI 的 "思考中" 面板，不想要就丢掉
        pass
    if delta.content:
        answer_parts.append(delta.content)

answer = "".join(answer_parts)
```

如果你确实有一个改不动的非流式代码路径——比如同步队列 worker——那就**显式**把 `enable_thinking=False` 写出来。兼容层默认是关，但我在不同 SDK 版本上见过差异，写明确比较稳。

> **实战提示：** 看到 Qwen3 在各种榜单上的高分，几乎都是默认开了 thinking 的成绩。如果你 A/B 发现 Qwen3 居然不如 Qwen-2.5，先确认自己用的是哪种模式。在硬推理任务上这两种模式的差距巨大。

## 工具调用（Function Calling）

线协议跟 OpenAI 完全一致：定义工具、模型决定调不调、你执行后把结果回灌进去。

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "根据订单号查询订单状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，例如 'ORD-123'"},
                },
                "required": ["order_id"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "我的 ORD-7781 现在到哪了？"},
]

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
)

choice = resp.choices[0]
if choice.message.tool_calls:
    call = choice.message.tool_calls[0]
    # 真实业务逻辑
    result = {"order_id": "ORD-7781", "status": "已发货", "eta": "2026-04-26"}

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

生产中体会到的几条：

- Qwen 模型很爱调工具——有时一句普通回答就能解决，它也要去调一下。本轮不希望它调，传 `tool_choice="none"`。
- `qwen-plus` 及以上支持并行工具调用，响应里 `tool_calls` 可能多于一条，别假定长度为 1。
- 工具的 JSON schema 写得糊（缺 `description`、enum 模糊），Qwen 填参的质量也跟着糊。把 schema 当成 prompt 的一部分来写。

## JSON 模式

做结构化抽取就直接要 JSON，然后校验。

```python
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "只输出一个 JSON 对象。不要任何说明文字。"},
        {"role": "user", "content": "从 '淘宝官方 iPhone 17 Pro 售价 ¥9999' 中提取商品名和价格。"},
    ],
    response_format={"type": "json_object"},
)

import json
data = json.loads(resp.choices[0].message.content)
```

`response_format={"type": "json_object"}` 在 `qwen-plus`、`qwen-max`、`qwen3-max` 上都生效。它能保证 JSON 语法合法，但**不保证 schema 合规**。要严格 schema，要么用工具调用（schema 写在工具定义里），要么用 `pydantic` 做后置校验。

我自己代码里 JSON 模式后必跟一次 `pydantic.parse_obj`——几千次调用里总会有那么一两次，模型给你一个语法合法但字段名错误的 JSON。捕到就大声报警，重试一次，再失败就告警。

## 真正能用的错误处理

按出现频率排序，你会见到的错误：

| HTTP / 类型 | 含义 | 处理 |
|---|---|---|
| `429 Throttling.RateQuota` | RPM/TPM 超了 | 指数退避带抖动，看是否要扩配额 |
| `400 InvalidParameter` | schema 级 bug——错 role、错 model_id、thinking 与非流式冲突等 | 修请求，**不要**重试 |
| `400 DataInspectionFailed` | 输入或输出触发内容审核 | 改 prompt 或脱敏，**不要**用同样负载重试 |
| `500 InternalError` | 阿里的问题 | 退避重试，最多 3 次 |
| `503 ModelOverloaded` | 该模型短时高峰 | 退避重试，必要时切备用模型 |
| 超时 | 输出过长、网络抖动、或 thinking 时间长 | 客户端超时调到 120s（thinking 模型必备） |

生产级封装：

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
            # 非限流的 4xx 不可重试
            if 400 <= e.status_code < 500 and e.status_code != 429:
                raise
            time.sleep(1 + attempt)
```

客户端超时配到能容下最坏情况：

```python
client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=120.0,  # 新版 SDK 默认是 600，但显式写出来更稳
)
```

## 什么时候改用原生 SDK

普通对话用兼容 SDK 就够。下面这些情况改用原生（`dashscope.Generation.call`）：

- 需要兼容层没暴露的参数——比如内置联网搜索的 `search_options`、增量输出的细节控制等。
- 想用一套 SDK 同时调 LLM、万相、TTS（少一个依赖）。
- 调用方在的区域，原生接口延迟更低。

原生流式写法稍微丑一点，但不难：

```python
import dashscope
from dashscope import Generation

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

responses = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你好"}],
    result_format="message",
    stream=True,
    incremental_output=True,
)

for resp in responses:
    chunk = resp.output.choices[0].message.content
    print(chunk, end="", flush=True)
```

`incremental_output=True` 是原生侧"只给增量、不给累积"的开关——不开的话每个 chunk 都是从头到当前的全量文本，UI 会狂闪。

## 上线前一份小清单

- [ ] 面向客户的链路锁死带日期的别名（`qwen-plus-2025-09-11`，而不是 `qwen-plus`）。
- [ ] 在百炼控制台为这个 Key 设了预算告警。
- [ ] 调用都过上面那个重试封装；不可重试错误进死信。
- [ ] 用 Qwen3 + thinking 的链路一定是流式。
- [ ] 每次调用都把 `usage` 字段记日志，方便日后用日志重算成本。
- [ ] 有备用模型：`qwen-max` 503 时降级到 `qwen-plus`，而不是给用户一个 500。

最后一条比想象中重要。百炼的区域级故障不常见但确实发生过，用户不会关心是不是阿里挂了。

下一篇是 Qwen-Omni。流式要求更严、请求结构不同、视频理解可能是这个平台上最有意思的能力之一。
