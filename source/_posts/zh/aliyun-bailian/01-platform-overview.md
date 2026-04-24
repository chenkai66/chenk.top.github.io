---
title: "阿里云百炼实战（一）：平台总览与第一个请求"
date: 2026-04-20 09:00:00
tags:
  - 阿里云百炼
  - DashScope
  - 大模型
categories: 阿里云百炼
lang: zh-CN
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 1
description: "从一线工程师视角梳理百炼/DashScope 的真实模型矩阵、两套调用入口的取舍，以及一个能跑通的最小示例，作为后续四篇文章的共同基础。"
disableNunjucks: true
---

只要你的产品需要服务中文用户，迟早会接到一个百炼模型。Qwen-Max 是目前性价比最高的中文 GPT-4 级理解能力，万相是国内唯一能开发票走采购流程的可量产文生视频，Qwen-TTS-Flash 是市面上唯一把粤语和川渝方言读得像本地人的合成。把这套东西在 AI Marketing 平台上跑了一年之后，这个系列就是我希望刚入门时有人塞给我的那本手册。

第一篇先把地图画清楚：百炼到底是什么、有哪些真正值得用的模型家族、两套接口的取舍、以及一个 hello world，让后面四篇可以直接跳到正题。

## 百炼是什么，DashScope 又是什么

命名上的混乱是真实存在的，因为阿里在中途改过一次品牌：

- **百炼** 是**产品**：控制台地址 `bailian.console.aliyun.com`，在那里管理 API Key、部署微调模型、搭 RAG 应用、看账单。
- **DashScope** 是百炼背后的**接口体系**。所有 HTTP 调用打到 `dashscope.aliyuncs.com`，Python SDK 直接 `pip install dashscope`。

你会在文档里看到两个名字反复出现，记住一个口诀就够了：控制台叫百炼，接口叫 DashScope。看到 "DashScope error" 就是 API 返回非 200。

## 真正用得到的模型矩阵

百炼上挂着上百个模型，这一年里真正掏过钱的只有这五类：

| 模型族 | 代表 model_id | 用途 |
|---|---|---|
| Qwen 文本大模型 | `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwen3-max`, `qwen3-coder-plus` | 对话、推理、工具调用、代码 |
| Qwen-Omni 多模态 | `qwen3-omni-flash`, `qwen3.5-omni-plus` | 视频/音频/图像理解 |
| 万相视频 | `wan2.5-t2v-plus`, `wan2.5-i2v-plus` | 文生视频、图生视频 |
| Qwen-TTS 语音合成 | `qwen3-tts-flash` | 语音合成，40+ 音色 |
| 文本向量 | `text-embedding-v3`, `text-embedding-v4` | 向量检索 |
| OpenSearch | （独立产品） | 混合检索、联网搜索 |

不在这个表里的，要么已经被弃用，要么是上面几个的薄变种，要么是研究预览。坚持用上面这些，最起码不会上线两个月就被通知模型 EOL。

## 计费一句话讲清

文本大模型按 token 计费（输入和输出分开定价，输出大约是输入的 2-4 倍）；TTS 按音频秒数；万相按视频秒数；Embedding 按调用次数。每个模型都有免费额度（通常 100 万 token 或 100 次生成），新模型上线时还会重置一轮，意味着只要愿意跨版本切换，原型阶段几乎不花钱。生产流量必须走单独的 API Key 并设预算告警；我自己被一个开了一夜的调试循环坑过四位数账单，正好一次。

## API Key：千万别 commit 进仓库

控制台左侧 **API-KEY** 申请。除了默认 workspace key，还可以建 workspace 级 Key——上线项目一定要建独立 Key，方便单独轮换。然后：

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

后续所有代码都从 `os.environ['DASHSCOPE_API_KEY']` 读。不要硬编码、不要把 `.env` 提交、生产环境请用密钥管理服务。DashScope 团队确实会吊销公开泄露的 Key，但通常是在它被爬虫扫到之后，那时候已经晚了。

## 两套接口：OpenAI 兼容 vs DashScope 原生

这是本文最重要的一段。**所有百炼文本/多模态模型都同时挂在两套 HTTP 接口上。**

### OpenAI 兼容接口

Base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`

走的是 OpenAI 协议。把官方 `openai` Python SDK 的 `base_url` 一改，95% 的存量代码原地能跑。日常默认就用这个。

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
        {"role": "system", "content": "你是一名资深后端工程师。"},
        {"role": "user", "content": "用两句话解释幂等键。"},
    ],
)
print(resp.choices[0].message.content)
```

### DashScope 原生接口

Base URL：`https://dashscope.aliyuncs.com/api/v1/`

阿里自家的协议——请求结构、字段名都不一样（`input.messages` 取代 `messages`、还有独立的 `parameters` 块等等）。用 `dashscope` SDK 或者直接拼 HTTP。

```python
import os
import dashscope

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

resp = dashscope.Generation.call(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一名资深后端工程师。"},
        {"role": "user", "content": "用两句话解释幂等键。"},
    ],
    result_format="message",
)
print(resp.output.choices[0].message.content)
```

### 选哪个

血泪总结后的口诀：

- **OpenAI 兼容**：默认。聊天、工具调用、JSON 模式、流式都够用，还能改一行就跟 GPT-4 做 A/B。
- **DashScope 原生**：万相视频必须用、Qwen-TTS 必须用、Qwen-Omni 多模态推荐用（兼容层会丢掉一些视频参数）；用到异步任务必须用；想第一时间用上新功能也得用——兼容层永远是落后的子集。

最常见的坑：看到 "OpenAI 兼容" 几个字，就以为**所有**模型都能这么调。错。`wan2.5-t2v-plus` 只有原生接口，`qwen3-tts-flash` 只有原生接口。这一点在第四篇和第五篇会反复强调。

> **实战提示：** 当你拿到一个 400，错误信息类似 `parameter X is not supported`，第一反应应该是检查这个模型是不是只支持原生接口。我排过的"百炼是不是挂了"工单里，差不多一半就是这个原因。

## 三个反复出现的概念

### model_id

每次调用都用一个字符串标识模型，比如 `qwen-plus` 或 `wan2.5-t2v-plus`。**model_id 上没有版本号**——阿里会用同一个 id 上线新权重，只在 changelog 里告诉你。需要锁版本就用模型卡片里列出的带日期别名（如 `qwen-plus-2025-09-11`）。任何面向客户的功能，**强烈建议锁日期版本**；我亲眼见过不锁版本的别名一夜之间换了一版调性。

### 异步任务

所有耗时超过约 30 秒的任务（视频生成、大批量 embedding、长篇 TTS）都是异步的，模式永远是：

1. POST 到创建接口，请求头加 `X-DashScope-Async: enable`。
2. 返回 `task_id`。
3. 轮询 `GET /api/v1/tasks/{task_id}`，直到 status 变成 `SUCCEEDED` 或 `FAILED`。
4. **24 小时内**下载结果 URL——过期就没了。

第四篇会给一份带退避的完整轮询代码。

### 流式

文本模型和 Qwen-Omni 都支持 SSE 流式。Qwen3 开了 `enable_thinking=True` 时，流式是**强制**的，非流式直接报错。Qwen-Omni 是**整体强制**流式（详见第三篇）。早点习惯 `stream=True`，你会用得比想象中频繁。

## 一份能直接跑的入门代码

存成 `hello_bailian.py` 跑一下，能打印出一段话就说明账号、Key、网络都通了，可以进入第二篇。

```python
import os
from openai import OpenAI

def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("先在环境变量里设 DASHSCOPE_API_KEY。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": "用一段话介绍阿里云百炼是什么。"},
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

想顺便确认原生接口也通，把前面那段 `dashscope` 代码也跑一遍。两条都应该成功，计费完全相同。

## 后面讲什么

第二篇深挖 Qwen 文本大模型族——按延迟和成本怎么选模型、工具调用、JSON 模式，以及那个亲手让我调试了四个小时的 `enable_thinking` 参数。第三篇讲 Qwen-Omni 的强制流式规则和一个真实的视频理解示例。第四篇是万相视频从创建到下载的完整链路，第五篇是 Qwen-TTS 的多语言旁白方案。

只来得及读一篇的话，建议读第二篇——文档里"没告诉你"的密度最高。
