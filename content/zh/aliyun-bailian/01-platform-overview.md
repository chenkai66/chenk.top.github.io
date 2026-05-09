---
title: "阿里云百炼实战（一）：平台总览与第一个请求"
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
description: "一个工程师视角的阿里云百炼（DashScope）导览——模型目录里真正能用的那几个、两种 endpoint 形态、异步任务模式，外加一个 hello-world 请求把后续文章的基础铺好。"
disableNunjucks: true
translationKey: "aliyun-bailian-1"
---
如果你的产品需要服务中文用户，那么迟早有一天你会用到阿里云的百炼平台。Qwen-Max 是目前性价比最高、能够达到 GPT-4 级别中文理解能力的模型；万相则是国内唯一支持开具正规发票的生产级文生视频 API；而 Qwen-TTS-Flash 更是目前唯一一个能自然处理粤语和四川话的语音合成工具，听起来不像冷冰冰的海关广播。在某 AI 营销平台上线并运行这些模型一年后，我深刻体会到，这个系列文章正是我希望当初有人能直接递到我手里的入门指南。

作为开篇，本文将带你快速了解百炼平台的整体情况：它究竟是什么？有哪些核心模型家族值得关注？两种 endpoint 的区别是什么？最后，我们会通过一个简单的“Hello World”请求示例，为后续内容打下基础，避免重复解释。

![阿里云百炼实战（一）：平台总览与第一个请求 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/illustration_1.jpg)
## 百炼和 DashScope 到底是什么？

这两个名字确实让人有点摸不着头脑，主要是因为阿里云在产品发展过程中调整了命名策略。简单来说，官方的 "DashScope" 文档是从 API 的角度来描述的，而 "百炼" 文档则是从控制台的角度切入的。其实它们是同一个产品，只是分别对应不同的使用场景。

![百炼（控制台）与 DashScope（API）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/fig1_bailian_dashscope_split.png)

在文档中，你可能会在同一段文字里看到这两个名字交替出现。可以把它们理解为“控制台 vs API”的关系。比如，当有人说“部署一个百炼应用”时，指的通常是通过控制台完成的操作；而提到“DashScope 报错”时，则意味着调用 API 时返回了非 200 的状态码。
## 你真正需要关注的模型清单

百炼平台上虽然列出了上百个模型，但在过去一年的实际生产环境中，我只用过以下这些模型，并且它们完全满足了我的需求：

| 模型系列 | 代表 model_id | 适用场景 |
|---|---|---|
| Qwen LLM（文本） | `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwen3-max`, `qwen3-coder-plus` | 对话、逻辑推理、工具使用、代码生成 |
| Qwen-Omni（多模态） | `qwen3-omni-flash`, `qwen3.5-omni-plus` | 视频、音频、图像的理解与处理 |
| Qwen-VL（视觉） | `qwen3-vl-plus` | 图像理解（比 Omni 系列更便宜） |
| 万相（视频生成） | `wan2.5-t2v-plus`, `wan2.5-i2v-plus` | 文本生成视频、图像生成视频 |
| Qwen-TTS | `qwen3-tts-flash` | 语音合成，支持 40 多种音色 |
| 向量模型 | `text-embedding-v3`, `text-embedding-v4` | 向量检索与相似度计算 |

如果某个模型没有出现在这张表里，那它要么已经停止维护，要么是某个主流模型的变体，要么还处于研究阶段。只要专注于使用这些模型，就能避免上线后不久因模型停止支持（EOL）而踩坑。
## 计费模式一句话说明白

大语言模型（LLM）按 token 数收费，输入和输出分别计费，其中输出的价格是输入的 2 到 4 倍。语音合成（TTS）按音频时长收费，万相则按视频秒数计费，而向量嵌入服务按调用次数收费。每个模型都提供免费额度，通常是 100 万 token 或者 100 次生成任务，而且每当新版本上线时，免费额度会重新刷新。这意味着如果你不介意切换版本，基本上可以一直免费试用各种功能原型。不过，生产环境的流量建议绑定独立的 API key，并设置预算告警；我就曾经因为有人忘记关闭调试循环，一晚上吃了一笔四位数的账单。
## API 密钥：千万别提交到代码库

登录控制台，在左侧导航栏找到 **API-KEY**，即可获取密钥。系统会提供一个默认工作空间的密钥，同时每个工作空间也可以生成专属密钥。对于生产环境项目，建议为每个工作空间单独创建密钥，这样即使需要轮换密钥，也不会影响开发环境。接下来，执行以下命令来设置环境变量：

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

本系列教程中的所有代码示例都会从 `os.environ['DASHSCOPE_API_KEY']` 中读取密钥值。切记不要将密钥硬编码到代码中，也不要将包含密钥的 `.env` 文件提交到代码仓库。在生产环境中，请务必使用密钥管理工具来存储和管理密钥。虽然 DashScope 团队会在发现密钥泄露后将其撤销，但通常是在密钥被公开爬虫抓取后才会采取行动，而这时可能已经造成了严重后果。
## 两个调用入口：OpenAI 兼容模式 vs DashScope 原生模式

这篇文章的核心要点来了。根据 Qwen API 文档的说明，**百炼平台上的每一个文本或多模态模型都可以通过两种不同的 HTTP 接口访问。**

![两种 HTTP 接口，共享同一份模型目录](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/fig2_two_endpoints.png)

### OpenAI 兼容模式入口

基础 URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`（中国大陆），或 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`（新加坡）。

这个接口遵循 OpenAI 的通信协议。只要把官方的 `openai` Python SDK 指向这个地址，95% 的现有 OpenAI 代码无需修改即可直接运行。这也是我日常开发中的默认选择。

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
        {"role": "system", "content": "你是一位资深后端工程师。"},
        {"role": "user", "content": "用两句话解释幂等性 key 的作用。"},
    ],
)
print(resp.choices[0].message.content)
```

### DashScope 原生模式入口

基础 URL：`https://dashscope.aliyuncs.com/api/v1/...`——根据模态不同，路径也会有所区分（例如 `/services/aigc/text-generation/generation`、`/services/aigc/multimodal-generation/generation`、`/services/aigc/text2video/...`）。

这是阿里云自研的通信协议，请求结构和字段名称都与 OpenAI 不同（比如使用 `input.messages` 而非 `messages`，还额外增加了 `parameters` 参数块等）。你可以通过 `dashscope` SDK 调用，也可以直接使用原生 HTTP 请求。

```python
import os
import dashscope

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

resp = dashscope.Generation.call(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一位资深后端工程师。"},
        {"role": "user", "content": "用两句话解释幂等性 key 的作用。"},
    ],
    result_format="message",
)
print(resp.output.choices[0].message.content)
```

### 如何选择合适的调用方式？

这是我踩了无数坑总结出来的经验法则：

- **OpenAI 兼容模式**：适用于普通对话、函数调用（function calling）、JSON 模式、流式输出（streaming）等场景。如果你想在 GPT-4 和 Qwen 之间快速切换进行 A/B 测试，只需修改一行代码即可。
- **DashScope 原生模式**：万相视频生成必选、Qwen-TTS 必选、Qwen-Omni 多模态推荐使用（兼容层会丢失部分视频参数）、需要异步任务模式时必选、最新功能尚未同步到兼容层时也必须使用原生模式。

一个常见的误区是：看到“OpenAI 兼容模式”就以为所有模型都能通过这种方式调用。其实不然。比如 `wan2.5-t2v-plus` 只能通过原生模式调用，`qwen3-tts-flash` 同样如此。后续第四、第五篇文章会反复强调这一点。

> **小贴士：** 如果遇到 400 错误，提示类似 `parameter X is not supported`，第一件事就是检查是否尝试通过兼容模式调用了一个仅支持原生模式的模型。在我处理过的“百炼出问题了”的工单中，大约一半都是这个原因导致的。
## 三个贯穿始终的核心概念

### model_id

每次调用模型时，都会使用一个类似 `qwen-plus` 或 `wan2.5-t2v-plus` 的字符串作为标识。这里没有传统意义上的版本号——阿里云会在同一个 ID 下更新模型权重，并通过 changelog 告知用户具体的变化。如果你需要固定某个版本，可以使用模型卡片中提供的日期别名（例如 `qwen-plus-2025-09-11`）。在任何面向客户的应用场景中，**务必锁定日期别名**；我曾经见过未锁定的默认别名在一夜之间风格大变的情况。

### 异步任务

对于耗时较长的任务（比如视频生成、大批量 embedding 计算、长文本 TTS 等，通常超过约 30 秒），DashScope 采用异步模式处理。其基本流程如下：

![异步任务模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/fig3_async_task_pattern.png)

1. 向创建任务的接口发送 POST 请求，并在请求头中添加 `X-DashScope-Async: enable`。
2. 接口会返回一个 `task_id`。
3. 使用 `GET /api/v1/tasks/{task_id}` 轮询任务状态，直到状态变为 `SUCCEEDED` 或 `FAILED`。
4. 在 24 小时内下载输出文件的 URL，因为链接会过期。

第四篇文章提供了一个完整的轮询实现示例，包含退避策略。

### 流式传输

无论是 LLM 还是 Qwen-Omni，都支持 SSE（Server-Sent Events）流式传输。对于 Qwen3，当启用 `enable_thinking=True` 时，流式传输是**强制要求**——非流式调用会被直接拒绝。而对于 Qwen-Omni，流式传输则是**无条件强制**的（更多细节请参考第三篇文章）。建议尽早熟悉 `stream=True` 参数，因为它的使用频率可能会超出你的预期。
## 第一个请求的完整流程

![阿里云百炼实战（一）：平台概览与第一个请求 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/01-platform-overview/illustration_2.jpg)

将以下代码保存为 `hello_bailian.py`，运行后如果能成功打印出一句话，说明你的账号、API Key 和网络都没有问题，可以继续阅读下一篇内容。

```python
import os
from openai import OpenAI

def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 DASHSCOPE_API_KEY。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": "用一段话简单介绍一下百炼是什么？"},
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

如果你想验证原生接口是否同样可用，可以替换为之前提到的 `dashscope` 代码片段再试一次。两种方式都应该能够成功运行，并且计费方式完全一致。
## 控制台、SDK 和 OpenAI 兼容接口——分别在什么场景下使用？

实际上，百炼可以通过三种方式来驱动，而选择哪种方式的影响比文档描述的要大得多。我花了两个月时间在一个项目里混用这三种方式，才慢慢摸索出它们的最佳使用场景。

**控制台**（`bailian.console.aliyun.com`）主要用于两件事：资源开通（创建工作空间、生成 API Key、配置 RAM 权限、部署应用）和问题诊断（查看每条请求的日志，定位模型返回异常的原因）。它并不适合处理运行时流量。控制台里的“Playground”对调试 Prompt 很有帮助，但它会忽略 SDK 调用中的一些参数，因此在 Playground 里能正常工作的 Prompt 上线后可能仍然会出问题。我把 Playground 看作是“模型是否可达”的探针，而不是“Prompt 是否正确”的验证工具。

**DashScope 原生 SDK**（`pip install dashscope`）是处理阿里云特有功能的首选。比如异步任务、视频处理、TTS、万相、最新模型参数、跨工作空间计费所需的 `workspace-id` Header，以及调试用的 `X-DashScope-DataInspection` Header。原生 SDK 提供了兼容层无法支持的参数字段（如 `incremental_output`、图像生成的 `seed`、联网模型的 `enable_search`）。此外，批量推理和 `RemoteService` 模型部署也只能通过原生 SDK 实现。

**OpenAI 兼容接口**则适合这样的场景：你的代码已经基于 OpenAI SDK 编写，希望通过修改一行代码就能在百炼和 OpenAI 之间进行 A/B 测试。兼容层是一个轻量级的翻译器，支持 `chat.completions.create`、`embeddings.create`，并兼容 `tools` / `tool_choice` / `response_format` / `stream` 等特性，同时允许通过 `extra_body` 传递一些 Qwen 特有的参数（如 `enable_thinking` 和 `enable_search`）。不过，它不支持万相、TTS 或任何异步模式。如果需要这些功能，可以针对特定调用切换到原生 SDK——同一个应用中同时使用两种客户端完全没有问题。

经过一年的实践，我总结了一些经验规则：

- **新项目，以 LLM 为主，希望保持多供应商兼容性** → 使用 OpenAI SDK 并指向兼容接口。
- **多模态混合项目（LLM + Omni + 视频 + TTS）** → 主要使用原生 SDK，但将聊天补全（chat completion）部分保留在兼容接口上，以便消息处理代码更具可移植性。
- **生产环境排错** → 完全切换到原生 SDK，因为它的错误信息更丰富（包含 `code`、`message` 和 `request_id`，而兼容接口只提供 OpenAI 样式的错误结构）。
## 你必须了解的 Region 路由与 RAM 权限管理

在使用百炼时，有两个主要的 region 需要注意：**北京** 和 **杭州**（此外还有一个功能上等同于第三个 region 的新加坡“国际”endpoint）。默认情况下，`dashscope.aliyuncs.com` 会解析到你账号最初创建时所在的 region。这往往是导致“昨天还能用，今天不行了”这类问题的头号原因。

具体来说，问题的根源在于某些模型是 region 锁定的。例如，`wan2.5-t2v-plus` 只在北京的资源池中提供；`qwen3-omni-flash` 则支持多 region；而 `text-embedding-v4` 最初上线时只部署在杭州，六周后才扩展到北京。如果你的账号路由到了一个尚未部署目标模型的 region，就会收到一个看似正常的 404 错误，提示 `Model not exist`。通常你会花上一个小时尝试调整 prompt，最后才发现问题出在路由配置上。

以下是两种解决方法：

```python
# 直接指定 region URL，强制请求走特定 region
client = OpenAI(
    api_key=key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 大陆默认
    # base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # 新加坡
)

# 使用原生 SDK 时，通过 workspace id header 指定——工作空间是按 region 绑定的
import dashscope
dashscope.api_key = key
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
```

在 RAM 权限管理方面，有一个常见的坑需要注意：API key 是绑定到某个工作空间的，而工作空间又隶属于主账号。RAM 子账号并不会自动继承主账号下百炼工作空间的访问权限。为了解决这个问题，你可以选择简单粗暴地授予 `AliyunBailianFullAccess` 权限，或者编写一个基于 workspace ARN 的自定义策略。策略中的 Resource 格式为 `acs:bailian:*:*:workspace/<workspace_id>`，其中 `<workspace_id>` 可以从控制台的 URL 栏中找到，而不是在任何显眼的“详情”页面中。如果某个 RAM 用户能够正常访问控制台，但在调用 API 时收到 `Forbidden.RAM` 错误，几乎可以肯定是缺少这条策略。

在生产环境中，我通常会为每个环境（开发、测试、生产）分别创建一个独立的 RAM 子账号，每个子账号对应一个工作空间，并且每个工作空间的 API key 都由密钥管理器托管并定期轮换。这样一来，跨环境的权限泄漏就不再是意外事件，而是需要显式配置策略才能发生的行为。
## 配额的真相：没人提前告诉你的那些细节

营销页面上写着“高并发、低延迟”，听起来很诱人。但实际的配额数字，至少在我最近一次审核时发现，对于免费账户来说，不仅非常具体，而且相当低。关键的三个指标是：

- **RPM**（每分钟请求数）——即在任意 60 秒内可以发送的请求数量。
- **TPM**（每分钟 token 数）——包括输入和输出在内的总 token 消耗。
- **TPD**（每日 token 上限）——按北京时间凌晨 0 点重置的日限额。

以下是我在一个全新的工作空间中（未申请额外配额）测得的默认值：

| 模型 | RPM | TPM | 异步并发任务数 |
|---|---|---|---|
| `qwen-turbo` | 500 | 500K | 不适用 |
| `qwen-plus` | 200 | 200K | 不适用 |
| `qwen-max` | 60 | 100K | 不适用 |
| `qwen3-omni-flash` | 60 | 100K | 不适用 |
| `wan2.5-t2v-plus` | 不适用 | 不适用 | 5 |
| `qwen3-tts-flash` | 100 | 50K | 不适用 |

需要注意的是，这些数字并非合同承诺，阿里云可能会随时调整。真正重要的是：这些免费配额足够用来开发和测试一个日活用户量（DAU）在 100 左右的原型，但如果突然有一条爆款推文带来了真实流量，系统就会立刻开始限流。此时你会收到 `Throttling.RateQuota` 错误，并返回 HTTP 429 状态码——你的重试逻辑需要将 429 视为“退避并重试”的信号，而不是直接判定为致命错误。

如果需要提高配额，可以通过控制台的 **API-KEY → 限流配置** 页面提交申请，并附上合理的理由。一般来说，只要请求合理（比如当前限额的 10 倍以内），审批通常会在 24 小时内完成。但如果你直接申请将 `qwen-max` 的配额提升 100 倍，却拿不出营收数据或业务预测，大概率会被拒绝。建议提前准备好你的预测 QPS、平均 prompt 长度以及清晰的商业需求说明。我个人已经提交过四次申请，其中只被拒过一次，所以只要理由充分，成功率还是很高的。
## Qwen 系列模型最新定价一览

阿里云的定价页面每季度都会调整，以下列出的具体数字可能会有变动，但各部分的比例相对稳定，可以作为规划依据。截至我最近一次核对（2026 年 4 月），以下是各模型的费用（单位：元/百万 tokens）：

| 模型 | 输入 | 输出 | 缓存输入 | 备注 |
|---|---|---|---|---|
| `qwen-turbo` | 0.3 | 0.6 | 0.15 | 性价比之选，适合分类任务 |
| `qwen-plus` | 0.8 | 2.0 | 0.4 | 日常开发主力 |
| `qwen-max` | 20 | 60 | 10 | 推理与代码审查利器 |
| `qwen3-max` | 24 | 96 | 12 | 最新版本，性能更强，价格也更高 |
| `qwen3-coder-plus` | 4 | 16 | 2 | 专为代码场景优化，中档价位 |
| `qwen3-omni-flash` | 2.5（文本）/ 12（视觉） | 5 | 不适用 | 视觉类 token 单独计费 |
| `text-embedding-v4` | 0.7（每百万 tokens） | 不适用 | 不适用 | 按调用次数计费 |

需要牢记的两个关键点：

- **输出成本是输入的 2-4 倍。** 如果你的应用是一个会多次循环调用的 function-calling agent，那么大部分费用会花在 assistant 的输出 tokens 上，而不是用户的输入 prompt。因此，优化时应优先减少输出内容，再考虑精简系统提示。
- **缓存输入享受半价优惠。** 百炼平台在 2025 年底推出了隐式 prompt 缓存功能——如果相同前缀的输入在约 5 分钟内重复发送，就会命中缓存。这项功能无需额外配置，但只有在你记录了 `usage.cached_tokens` 后才能看到具体节省。对于带有长固定系统提示的 RAG 应用，缓存命中率通常能达到 60%-80%，从而显著降低账单。

至于万相（按视频秒数计费）和 Qwen-TTS（按音频秒数计费），它们的合理计费单位是“按生成资产”而非“按 token”。例如，一段 720p、5 秒的视频大约花费 1.5 元；一段 60 秒的 TTS 语音大约花费 0.6 元。这些服务的价格已经低到让人工审核成为主要瓶颈，而非 API 调用成本。
## 接下来的内容

第二篇文章将深入探讨 Qwen LLM 系列模型——包括如何根据延迟和成本选择合适的模型、function calling 的使用技巧、JSON 模式的正确打开方式，以及那个让我足足调试了四个小时的 `enable_thinking` 参数。第三篇文章聚焦于 Qwen-Omni 的流式处理要求，并通过一个真实的视频理解案例来展示其应用。第四篇则完整解析万相视频处理的端到端流程，而第五篇会介绍如何利用 Qwen-TTS 实现多语言配音。

如果时间有限，只读一篇的话，推荐第二篇——它包含了大量“文档里没提到”的实用内容，绝对值得一读。