---
title: "阿里云百炼（一）：平台概览与第一个请求"
date: 2026-02-25 09:00:00
tags:
  - Aliyun Bailian
  - DashScope
  - LLM
categories: 阿里云百炼
lang: zh
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 1
description: "一个工程师视角的阿里云百炼（DashScope）导览——模型目录里真正能用的那几个、两种 endpoint 形态、异步任务模式，外加一个 入门示例 请求把后续文章的基础铺好。"
disableNunjucks: true
translationKey: "aliyun-bailian-1"
---
只要你的产品面向中文用户，迟早都得调用百炼（Bailian）的模型。Qwen-Max 是当前中文理解能力对标 GPT-4 且性价比最优的 LLM；万相（Wanxiang）是迄今唯一支持中文发票生成且已在生产环境稳定落地的 text-to-video API；Qwen-TTS-Flash 则是目前唯一在粤语与四川话合成中自然度高、无机械感的 TTS 模型。在 AI 营销平台跑了一年生产流量后，我希望入职第一天就能拿到这份指南。

第一篇先摸底：百炼是什么、会遇到哪些模型家族、两种接口风格的区别，以及各自的 Hello World，这样后面的文章就不用反复解释基础概念了。

![Aliyun Bailian (1): Platform Overview and First Request — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/illustration_1.png)

## 百炼是什么，DashScope 又是什么？

命名体系较为混乱：阿里曾将产品名从 DashScope 统一更改为‘百炼’。官方 'DashScope' 文档是从 API 角度讲的；‘百炼’ 文档是从控制台角度讲的。其实是同一个产品，两个名字。

![Bailian (console) vs DashScope (API)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/fig1_bailian_dashscope_split.png)

两套文档中名称混用频繁，甚至同一段落内交替出现——本质是‘控制台操作’与‘API 集成’两种使用视角的差异。例如，‘部署百炼应用’指的是在控制台完成配置，而‘DashScope 报错’通常指 API 返回非 200 状态码。

## 你真正关心的模型清单

百炼平台上托管了一百多个模型。在生产环境跑了一年后，我只为下面这些付过钱：

| Family | Representative model_id | Use it for |
|---|---|---|
| Qwen LLM (text) | `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwen3-max`, `qwen3-coder-plus` | Chat, reasoning, tool use, code |
| Qwen-Omni (multimodal) | `qwen3-omni-flash`, `qwen3.5-omni-plus` | Video / audio / image understanding |
| Qwen-VL (visual) | `qwen3-vl-plus` | Image-only understanding (cheaper than Omni) |
| Wanxiang (video) | `wan2.5-t2v-plus`, `wan2.5-i2v-plus` | Text-to-video, image-to-video |
| Qwen-TTS | `qwen3-tts-flash` | Speech synthesis, 40+ voices |
| Embeddings | `text-embedding-v3`, `text-embedding-v4` | Vector search |

表中未列出的模型，通常属于以下三类之一：已弃用（deprecated）、仅提供轻量封装的变体，或处于研究预览阶段的版本。坚持使用表中所列模型，就能避免因模型上线六周后突然终止支持（EOL）而措手不及。

## 一句话讲清计费模式

LLM 按 Token 计费（输入输出分开算，输出贵 2-4 倍），TTS 按音频秒数，万相按视频秒数，Embeddings 按调用次数。每个模型都有免费额度，通常是 100 万 Token 或 100 次生成，新模型发布时会重置。这意味着，只要接受模型版本可能自动更新，几乎所有功能都可用于免费原型验证。生产流量必须使用独立的 API Key，并配置预算告警；我曾因调试循环未及时终止、整夜持续运行，收到过一笔四位数账单。

## API Key：千万别提交到代码库

在控制台左侧导航栏 **API-KEY** 下获取密钥。有一个默认工作空间 Key 和按工作空间分配的 Key；任何生产项目都要创建工作空间 Key，这样轮换时不会影响开发环境。然后：

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

本系列所有代码片段都读取 `os.environ['DASHSCOPE_API_KEY']`。不要硬编码 Key，不要提交 `.env` 文件，生产环境请把 Key 放进 secrets manager。 DashScope 团队确实会撤销已泄露的密钥，但这一操作通常发生在密钥被公开爬虫捕获之后，此时损失往往已无法挽回。

## 两种接口：OpenAI 兼容 vs DashScope 原生

这是本文最关键的前提：根据 Qwen API 参考文档，所有百炼文本及多模态模型均支持两种 HTTP 接口方式。

![Two HTTP surfaces, one model catalog](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/fig2_two_endpoints.png)

### OpenAI 兼容接口

Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`（大陆），或 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`（新加坡）。

它讲 OpenAI 的通信协议。把官方 `openai` Python SDK 指向它，你现有 95% 的 OpenAI 代码不用改就能跑。这也是我的默认选择。

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

### DashScope 原生接口

Base URL: `https://dashscope.aliyuncs.com/api/v1/...` —— 不同模态路径不同（`/services/aigc/text-generation/generation`, `/services/aigc/multimodal-generation/generation`, `/services/aigc/text2video/...`）。

这是阿里自家的通信协议——请求结构不同，字段名不同（`input.messages` 而不是 `messages`，有 `parameters` 块等）。你用 `dashscope` SDK 或者 raw HTTP。

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

### 什么时候用哪个

这些规则均来自实际踩坑经验：

- **OpenAI 兼容**：默认用于普通聊天、函数调用、 JSON 模式、流式输出。改一行代码就能跟 GPT-4 做 A/B 测试。
- **DashScope 原生**：万相视频必须用， Qwen-TTS 必须用，推荐用于 Qwen-Omni 多模态（兼容层会丢掉一些视频参数），需要异步任务模式必须用，想要还没回迁到兼容层的最新特性必须用。

有个常见的坑：有人看到 “OpenAI 兼容” 就以为 *所有* 模型都支持。并不是。`wan2.5-t2v-plus` 仅原生。`qwen3-tts-flash` 仅原生。第 4 和第 5 篇文章会重点强调这点。

> **Tip:** 当你遇到 400 错误且消息类似 `parameter X is not supported` 时，第一步应该检查你是不是通过兼容接口调用了仅原生支持的模型。我调试过的 “百炼坏了” 的工单里，有一半都是这个问题。

## 三个到处出现的概念

### model_id

每次调用都靠一个字符串 keyed，比如 `qwen-plus` 或 `wan2.5-t2v-plus`。没有版本号——阿里会在同一 model_id 下发布新版本模型权重，并在变更日志（changelog）中说明。如果需要固定版本，使用模型卡片里列出的日期别名（`qwen-plus-2025-09-11`）。所有面向客户的功能，**必须使用带日期的模型别名**；我曾因未固定版本，导致模型更新后输出风格突变。

### 异步任务

任何耗时超过 ~30 秒的操作（视频生成、大批量 embedding、长表单 TTS）都是异步的。模式永远是：

![Async task pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/fig3_async_task_pattern.png)

1. POST 到创建接口， header 带 `X-DashScope-Async: enable`。
2. 拿到 `task_id`。
3. 轮询 `GET /api/v1/tasks/{task_id}` 直到状态变为 `SUCCEEDED` 或 `FAILED`。
4. 24 小时内下载输出 URL——会过期。

第 4 篇文章里有完整的带退避策略的轮询实现。

### 流式输出

LLM 和 Qwen-Omni 支持 SSE 流式。对于开启 `enable_thinking=True` 的 Qwen3，流式是 **强制** 的——非流式会拒绝调用。对于 Qwen-Omni，流式也是强制的 *无条件*（第 3 篇详述）。早点习惯 `stream=True`；你会比预期更频繁地使用它。

## 一个完整的第一个请求

![Aliyun Bailian (1): Platform Overview and First Request — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/illustration_2.png)

保存为 `hello_bailian.py` 并运行。如果打印出一句话，说明你的账户、 Key 和网络都没问题，可以进入第 2 篇了。

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

如果你想确认原生接口也能用，换上前面的 `dashscope` 代码片段。两者都应该成功；计费也完全一样。

## 控制台 vs SDK vs OpenAI 兼容——什么时候用哪个

实际上有三个地方可以驱动百炼，选择的重要性比文档说的要大得多。在项目中同时混用这三种方式长达两个月后，我才逐步理清各自的适用边界。

`bailian.console.aliyun.com` 上的 **控制台** 只做两件事： provisioning （创建工作空间、 API Key、 RAM 授权、应用）和诊断（查看单次请求日志，找出模型为什么返回垃圾）。它不适用于运行时的生产流量分发与调度。控制台中的 Playground 便于提示词快速迭代，但会忽略部分 SDK 调用所需的参数，因此在 Playground 中可用的提示词，上线后仍可能失败。我把 playground 当成 “模型可达” 的标志，而不是 “这个提示词正确” 的保证。

**DashScope 原生 SDK**（`pip install dashscope`）是当你需要任何阿里特有功能时的正确选择：异步任务、视频/TTS/万相、最新模型参数、跨工作空间计费的 workspace-id header，或者 `X-DashScope-DataInspection` 调试 header。原生 SDK 暴露了兼容层会丢掉的 `parameters` 块字段（`incremental_output`，图像生成的 `seed`，联网模型的 `enable_search`）。它也是批量推理和 `RemoteService` 模型部署的唯一路径。

**OpenAI 兼容接口** 是当你的代码基于 OpenAI SDK 编写，且想通过改一行代码就能对百炼和 OpenAI 做 A/B 测试时的正确选择。兼容层是个薄翻译器。它讲 `chat.completions.create`，`embeddings.create`，支持 `tools` / `tool_choice` / `response_format` / `stream`，并接受 `extra_body` 来传递少数 Qwen 特有旋钮（`enable_thinking`, `enable_search`）。它不讲万相、 TTS 或任何异步模式。如果需要那些，你必须为那个调用降级到原生 SDK—— 你完全可以同时在同一个应用中使用两种客户端。

跑了一年后的经验法则：

- 新项目，主要是 LLM，想要供应商无关的代码 → OpenAI SDK 指向兼容接口。
- 大杂烩（LLM + Omni + 视频 + TTS）→ 除了 chat-completion 路径用兼容接口以保持消息代码可移植外，其他全部用原生 SDK。
- 生产调试 →  exclusively 原生 SDK，因为错误信封更丰富（`code` + `message` + `request_id`，而兼容层只给你 OpenAI 形状的错误）。
## 区域路由和 RAM 权限，这些你真正得搞清楚

百炼主要就两个区：**北京**和**杭州**（还有个新加坡“国际”节点，功能上算第三个区）。默认的 `dashscope.aliyuncs.com` 会解析到你账号最初开通时所在的区域。这正是‘昨天还正常，今天就报错’类问题高频出现的根本原因。

具体坑在哪？有些模型是绑死区域的。`wan2.5-t2v-plus` 只在北京池子里。`qwen3-omni-flash` 多区域可用。`text-embedding-v4` 先上的杭州，过了六周才到北京。如果你的账号路由到了一个模型还没部署的区域，你会收到一个看似合法的 404，提示 `Model not exist`。此时无需反复调试 prompt，而应首先排查区域路由问题。

解决办法有两个：

```python
# Force a region by hitting the regional URL directly
client = OpenAI(
    api_key=key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # mainland default
    # base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Singapore
)

# Or with the native SDK, set the workspace id header — workspaces are region-scoped
import dashscope
dashscope.api_key = key
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
```

再说 RAM，坑在于 API Key 是绑在工作空间（workspace）上的，而工作空间属于主账号。 RAM 子账号不会自动继承主账号的百炼工作空间权限。你要么直接给 `AliyunBailianFullAccess`（省事但粗放），要么写个自定义策略绑定到工作空间 ARN。策略资源格式是 `acs:bailian:*:*:workspace/<workspace_id>` —— 注意 workspace id 在控制台 URL 栏里，不在什么明显的“详情”页。那些能登控制台但 API 调用报 `Forbidden.RAM` 的，基本都是缺了这个策略。

生产环境我这么搞：每个环境（dev / staging / prod）单独建一个 RAM 子账号，对应一个工作空间，配一个由 secrets store 管理的轮转 API Key。这样跨环境泄露就是显式的策略动作，而不是意外事故。

## 配额真相，没人会提前告诉你

营销页面写的是“高并发、低延迟”。实际数字呢？根据我上次审计配额的情况，免费档的具体数值非常低。主要看三个指标：

- **RPM** (requests per minute) — 任意 60 秒窗口内能发多少请求。
- **TPM** (tokens per minute) — 计入的输入加输出 Token 总数。
- **TPD** (tokens per day) — 每日上限，北京时间 00:00 重置。

这是我测过的新建工作空间默认值，没申请过配额提升：

| 模型 | RPM | TPM | 并发任务 (异步) |
|---|---|---|---|
| `qwen-turbo` | 500 | 500K | n/a |
| `qwen-plus` | 200 | 200K | n/a |
| `qwen-max` | 60 | 100K | n/a |
| `qwen3-omni-flash` | 60 | 100K | n/a |
| `wan2.5-t2v-plus` | n/a | n/a | 5 |
| `qwen3-tts-flash` | 100 | 50K | n/a |

这些不是合同承诺，阿里会悄悄调整。重点在于：免费档默认值够你开发和做个百级日活的原型，但只要一条广告推文带来真实流量，立马限流。报错是 `Throttling.RateQuota` 配 HTTP 429 —— 你的重试包装器得把 429 当成退避重试，别当致命错误处理。

想提配额，去控制台 **API-KEY → 限流配置** 提交理由。合理的请求（比如当前限制的 10 倍）通常 24 小时内批。要是没营收数据就想把 `qwen-max` 提 100 倍，肯定被拒——带上你的预估 QPS、平均 prompt 长度和业务案例。我提过四次，只被拒过一次。

## 每百万 Token 计价——这才是当前有效的版本

计价页面每季度都在变，下面的数字肯定会漂移，但*比例*足够稳定，用来做规划没问题。截至我上次审计（2026-04），人民币每百万 Token：

| 模型 | 输入 | 输出 | 缓存输入 | 备注 |
|---|---|---|---|---|
| `qwen-turbo` | 0.3 | 0.6 | 0.15 | 分类任务的廉价主力 |
| `qwen-plus` | 0.8 | 2.0 | 0.4 | 日常主力 |
| `qwen-max` | 20 | 60 | 10 | 推理、代码审查 |
| `qwen3-max` | 24 | 96 | 12 | 最新，比 qwen-max 贵 |
| `qwen3-coder-plus` | 4 | 16 | 2 | 代码专用，中档价格 |
| `qwen3-omni-flash` | 2.5 (文本) / 12 (视觉) | 5 | n/a | 视觉 Token 单独计费 |
| `text-embedding-v4` | 0.7 (每百万 Token) | n/a | n/a | 按调用计费 |

两点得刻在脑子里：

- **输出价格是输入的 2-4 倍**。一个循环 3 次的 Function Calling Agent，账单大头在助手的输出 Token，不在用户 prompt。优化成本时，最后再砍系统 prompt，优先砍输出。
- **缓存输入半价**。百炼在 2025 年底上了隐式 prompt 缓存——大概 5 分钟内发送相同的 prompt 前缀就能命中缓存。不用特意开通，但得日志里看 `usage.cached_tokens` 才能见到省下的钱。对于带长固定系统 prompt 的 RAG 应用，缓存命中率通常有 60-80%，账单相应下降。

万相（按视频秒数）和 Qwen-TTS （按音频秒数），计价单位是成品资产，不是 Token。 720p 5 秒 clips 大概 1.5 元； 60 秒 TTS  narration 大概 0.6 元。都便宜到瓶颈在于人工审核，而不是 API 花费。

## 接下来写什么

第二篇深挖 Qwen LLM 家族——按延迟和成本选模型、 Function Calling、 JSON 模式，还有那个让我 personally 调试了四个小时的 `enable_thinking` 参数。第三篇讲 Qwen-Omni 的流式要求和真实的视频理解案例。第四篇是万相视频管线端到端实战，第五篇是用 Qwen-TTS 做多语言 narration。

如果只读一篇，读第二篇——那里“文档里没写但你必须知道”的内容密度最高。