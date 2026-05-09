---
title: "大模型工程（十二）：生产 —— 部署、监控、成本"
date: 2026-05-06 15:00:00
tags:
  - llm
  - production
  - deployment
  - monitoring
  - cost
  - autoscaling
categories: 大模型工程
series: llm-engineering
series_order: 12
series_title: "大模型工程"
lang: zh
mathjax: false
disableNunjucks: true
description: "服务栈选型细化、给 LLM 做 autoscaling、延迟预算、prompt+completion 成本跟踪、多模型路由、FrugalGPT 级联、第一天就要的可观测性，以及能用的 on-call 模式。"
translationKey: "llm-engineering-12"
---
这是最后一章。前面讲了模型、prompt、检索和评估的构建方法。这一章聊聊怎么让系统持续运行，还不烧光预算。

生产环境下的 LLM 服务更像高流量 Web 服务，不像传统 ML 推理。区别在于，每个请求可能花几美元，还可能卡两分钟。

这章我会多讲数字。原因很简单：生产环境中，功能是赚钱还是烧钱，往往差个 2 到 5 倍成本。这个差异通常没人注意。学会快速估算 LLM 工作负载成本是最有用的技能。

下面的数据截至 2025 年底到 2026 年初。用之前记得核对最新价格。

![LLM Engineering (12): Production — Deployment, Monitoring, Cost — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/illustration_1.jpg)
## 端到端服务栈

![fig1: 端到端服务栈架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig1_stack_architecture.png)

生产级 LLM 应用的服务栈通常分这几层：

```
[CDN / WAF]
   ↓
[API 网关]    ← 限流、认证、请求标准化
   ↓
[应用服务器]  ← 提示组装、RAG 检索、工具执行、智能体循环
   ↓
[LLM 网关]    ← 模型路由、回退机制、成本核算、提示缓存
   ↓
[LLM 服务]    ← vLLM/SGLang/托管 API
   ↓
[可观测性]    ← 日志、指标、追踪、评估运行
```

应用服务器和 LLM 网关是工程的核心。前者处理业务逻辑，后者让多个模型像一个服务那样工作。

从一开始就该把 LLM 网关设计成独立服务。它需要支持以下功能：

- **多模型路由**：根据分类器，部分请求发给小型快速模型，其他发给大型慢速模型。
- **回退机制**：主供应商返回 5xx 错误时，切换到备用供应商重试。
- **成本跟踪**：记录每个请求的提示词 token 数、生成 token 数、模型名称和成本。
- **提示缓存封装**：即使供应商支持缓存，应用代码也不用操心缓存键。
- **A/B 测试**：按比例将流量路由到新模型变体。
- **配额管理 / 熔断机制**：单用户消耗过多资源时，直接终止请求。

从零开始搭建这个网关要花几周时间。不过开源社区已经有不少选项：

- **LiteLLM**：纯 Python 实现，支持 100 多个供应商集成，提供 OpenAI 兼容接口，内置成本跟踪功能。最快的方式。
- **OpenRouter**：托管式网关，单一 API 跨供应商，内置模型市场。每 token 成本略高，但自动处理故障转移和价格套利。
- **Cloudflare AI Gateway**：部署在 CDN 边缘，支持缓存、限流和分析。便宜且运维简单，但灵活性稍差。
- **BentoML / Bento Cloud**：重型框架，适合在同一网关后托管自定义训练模型。
- **Portkey、Langfuse Gateway**：新兴玩家，观测能力很强。

尽早选一个方案，同时做好后期替换的准备。网关是少数容易导致供应商锁定的地方。我会写大量代码与网关接口绑定，所以尽量保持接口简洁。
## 自托管 vs 托管 API

一个老生常谈的问题。2026 年的实话是这样：

**什么时候用托管 API？**

- 我需要 GPT-5、Claude-4.5、Gemini-3 这种前沿模型的质量，但没 GPU。
- 每月 token 量低于 10 亿——这种规模托管更便宜。
- 延迟要求宽松，TTFT 超过 500 毫秒也能接受。
- 没有工程师专门负责 GPU 运维。

**什么时候选自托管？**

- 每月 token 量超过 10 亿，且开源模型够用（如 Qwen3-32B+、LLaMA-3.3-70B+）。
- 数据驻留有严格要求。
- 需要稳定实现 TTFT 小于 100 毫秒。
- 有特定微调需求。

对于 Qwen3-32B 级别的负载，盈亏平衡点大约是每月 10 亿 token。低于这个量，算上工程成本，托管 API 更划算。高于这个量，用专用 GPU（租或买）自托管的成本优势能达到 3 到 10 倍。

常见错误是 GPU 利用率太低。比如 4xH100 设备跑在 30% 利用率时，单 token 成本比 OpenAI API 还贵。目标是让吞吐率持续高于 70%。

2025 到 2026 年，一种流行的做法是：**用小规模开源模型自托管大部分低成本流量，把最难的 5%-10% 流量交给托管的前沿 API。** 这样既能享受自托管的成本优势，又能在关键场景保留前沿模型的质量。路由决策由一个小分类器完成（详见后面的多模型路由章节）。
## 多模型路由和 FrugalGPT

![LLM Engineering (12): Production — Deployment, Monitoring, Cost — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/12-production/illustration_2.jpg)

Chen 等人在 2023 年的论文《FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance》中提出了路由再级联模式。核心观察很简单：大多数 LLM 请求都很简单，小模型或低成本模型就能搞定；只有少数请求需要大模型出马。如果能快速判断请求类型，成本可以降 5 到 10 倍，质量几乎不受影响。

FrugalGPT 提出了三种方法：

1. **Prompt 优化**——压缩提示词，去掉冗余上下文，合并相关请求。
2. **LLM 级联**——先用最便宜的模型试一下，获取置信度信号。如果置信度低，就切换到更贵的模型。
3. **模型路由**——训练一个分类器，直接选对的模型。

实际生产中，级联用得最多。伪代码如下：

```python
def cascade(question, models=[CHEAP, MID, EXPENSIVE]):
    for m in models:
        answer, confidence = m.generate_with_confidence(question)
        if confidence > THRESHOLDS[m]:
            return answer
    return answer  # 返回最后一个模型的输出
```

难点在于置信度信号怎么搞。常见方法有这些：

- **自报告置信度**——让模型自己打分，比如“请为你的答案信心打 1 到 5 分”。这种方法有点用，但噪声很大。
- **基于 logprob**——用答案片段的 logprob 值。合理，但只适合你能控制采样的场景。
- **采样一致性**——对便宜模型采样 3 到 5 次。如果结果一致，就接受；不一致，就交给更贵的模型。
- **任务专用验证器**——代码任务检查“是否通过公开测试”；数学问题判断“答案是否在合理范围内”。

Hu 等人在 2024 年的论文《RouteLLM: Learning to Route LLMs with Preference Data》中提出基于 embedding 的路由方法。他们用离线偏好数据中的 (query, best-model) 对训练分类器，提前预测该用哪个模型。RouteLLM 训练的路由器在 MT-Bench 上以 26%-44% 的成本达到了 GPT-4 95% 的质量。

生产环境里，路由决策通常归结为两个问题：“这个请求简单吗？”（交给便宜模型）和“这个请求是已知的复杂模式吗？”（交给昂贵模型）。中间情况走默认路径。一个包含 50 条规则的启发式分类器能捕捉 80% 的节省潜力，训练好的路由器则能捕捉约 90%。

下一节会聊我在生产环境中踩过的坑和血泪经验。
## 缓存层

三种缓存很重要，还能叠加使用。

1. **供应商侧的 prompt 缓存**（第 9 章）。复用重复 prompt 前缀的 KV 缓存。Anthropic、OpenAI、DeepSeek 和 Google 都在用。缓存部分成本能降 90%。
2. **应用侧的语义缓存**。两个用户查询语义相似时，直接返回缓存答案。工具包括 GPTCache（基于 FAISS）、Redis Semantic Cache、MemCache 加自定义 embedding 查找。FAQ 类工作负载命中率 30%-70%，开放式对话几乎为 0。
3. **确定性函数的结果缓存**。LLM 输出送入下游流水线（如分类、抽取）且输入完全相同时，用输入哈希值存储输出。相同输入不会触发第二次 LLM 调用。

GPTCache 值得细看。它的逻辑很简单：用小 embedding 模型编码用户查询，和缓存对比余弦相似度。如果相似度超阈值（通常 0.95），就返回缓存答案。阈值是关键参数。调高阈值减少误判，但命中率也会下降。比如，客服机器人回答常见问题，阈值设 0.92 可以省 50% 成本。但代码生成助手每个查询都独特，语义缓存基本没用。

缓存组合效果显著。prompt 缓存降低单次调用成本，语义缓存减少调用次数。两者叠加优化效果翻倍。
## 给 LLM 做 autoscaling

![fig4: LLM 工作负载弹性扩缩容](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig4_autoscaling.png)

给 LLM 做自动扩缩容比无状态的 Web 服务难得多。原因有三点。

第一，模型加载太慢。一个 70B 参数的模型，新启动的 vLLM 实例需要 2 到 4 分钟加载权重并预热。如果流量高峰只持续 10 分钟，扩容根本来不及。

第二，GPU 按小时计费。为 5 分钟的流量高峰启动新 GPU 实例，会浪费 55 分钟的资源。

第三，连续批处理的负载非线性。vLLM 在 50% 和 90% 负载下的延迟表现差异很大。安全运行点远低于传统 CPU 服务的建议值。

实际操作中，我总结了一些方法。

定时预热。如果流量高峰通常在早上 9 点，提前在 8:50 扩容。

保守扩容，积极缩容。利用率到 60% 时扩容，降到 30% 时缩容。这和 Web 服务的策略相反。

备用回退方案。自托管服务达到容量上限时，把溢出请求转到托管 API。单价高一些，但能应对流量高峰，避免资源浪费。

多模型保底。即使低流量时段，也要确保每个模型至少一个副本在线。冷启动成本太高，不能忽视。

带背压的请求队列。服务器过载时，用公平调度器按用户排队处理请求。返回带 `Retry-After` 头的 429 状态码，而不是让所有用户延迟崩溃。

vLLM 提供两个重要参数：`--max-num-seqs`（最大并发请求数）和 `--max-num-batched-tokens`（每步最大 token 吞吐量）。调整这些参数时，优先满足延迟目标，而不是追求吞吐量最大化。
## 延迟预算拆解

![fig2: 延迟预算拆解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig2_latency_breakdown.png)

一个面向用户的聊天产品，延迟预算大致如下：

| 组件 | 预算（ms） | 备注 |
|---|---|---|
| 网络输入 | 50 | 地理位置影响 |
| API 网关 / 认证 | 10 | 要求快速响应 |
| 应用服务器逻辑 | 50 | 包括 RAG embedding、工具调度 |
| RAG 检索 | 100 | 向量数据库 + 重排序 |
| LLM 网关开销 | 5 | 仅路由操作 |
| LLM TTFT（队列 + 预填充） | 300 | 模型本身的处理时间 |
| 网络输出 | 50 | 和输入一致 |
| **首 token 总计** | **~565 ms** | 控制在 1 秒目标内 |
| LLM ITL（解码） | 25 | 每个 token，支持 40 tok/s |
| 网络缓冲 | 20 | 平滑输出 |

两个最关键的延迟指标是：**TTFT** 和 **ITL**。TTFT 决定用户是否觉得“有反应”。ITL 影响流式输出的持续速度。2026 年的行业经验如下：

- 聊天：TTFT < 800 ms 是“即时”，< 2 秒是“可接受”。
- 语音：TTFT < 300 ms 才能自然对话。
- 代码补全：TTFT > 200 ms 用户会放弃。
- 批处理 / 智能体任务：TTFT 可放宽到 5-30 秒，用户通常能容忍。

首 token 输出后，用户看到流式结果。5-10 秒内，40-60 tok/s 的速度可以接受。总响应时间取决于输出长度。比如 200 个 token，按 40 tok/s 计算，需要 5 秒。加上 600 ms 的 TTFT，端到端总共 5.6 秒。

延迟预算应该花在哪里？

- **重排序（Reranking）**：50-100 ms 值得投入（见第 8 篇）。
- **推测解码（Speculative decoding）**：减少 50-100 ms 的 ITL（见第 5 篇），值得投资。
- **工具调用（Tool calls）**：200 ms 的工具会阻塞模型。尽量并行化，或者积极缓存。
- **推理 / 思考（Reasoning / Thinking）**：思考型模型会在输出首个可见 token 前增加 1-10 秒。只在必要时使用。

对于延迟敏感的路径，实际测量时间花在哪。拆解结果往往出乎意料。我见过一个“慢”的功能，LLM 本身只用了 200 ms，但 RAG 的重排序花了 1.2 秒。另一个案例中，LLM 表现正常，但下游的 JSON 序列化拖了 800 ms。优化之前，先做好追踪分析。
## 第一天就开始成本跟踪

![fig3: 每请求成本拆解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig3_cost_per_request.png)

每请求的成本核算，必不可少。没有它，我没法做到以下几点：

- 发现某个用户或功能导致的成本激增。
- 对成本与质量的调整做 A/B 测试。
- 提前预测账单支出。
- 及时发现失控的 agent 循环，避免花掉 1000 美元。

每个请求至少要记录这些信息：

```python
{
    "request_id": "req_xxx",
    "user_id": "user_xxx",
    "endpoint": "/chat",
    "model": "claude-4-5-sonnet-20250901",
    "prompt_tokens": 4321,
    "cached_tokens": 3000,    # 其中来自缓存的部分
    "completion_tokens": 215,
    "total_cost_usd": 0.0152,
    "latency_ms": {
        "ttft": 350,
        "total": 5430,
    },
    "tools_called": ["search", "fetch_doc"],
    "ts": 1714123456789,
}
```

按用户、功能和模型聚合数据。触发告警的场景包括：

- 单用户 1 小时内花费超中位数 10 倍。
- 某功能单次调用成本周环比翻倍。
- 新模型在质量相当的情况下，对话解决成本高于旧模型。

举个算术练习。假设我的产品每天调用 LLM 1 万次，平均输入 4K token，输出 500 token，使用的是 Claude-4.5 Sonnet：

- 每次调用：$4 \cdot 4 + 15 \cdot 0.5 = 16 + 7.5 = 23.5$ 千 tok 分 = $0.0235。
- 每天：$235。
- 每月：$7050。

加入 prompt 缓存（4K 输入中有 3K 被缓存）：

- 每次调用：$4 \cdot 1 + 0.30 \cdot 3 + 15 \cdot 0.5 = 4 + 0.9 + 7.5 = 12.4$ 千 tok 分 = $0.0124。
- 每月：$3720，节省 $3330（47 %）。

再加入 30 % 的语义缓存命中率：

- 每月：$3720 \cdot 0.7 = $2604，总共节省 $4446（63 %）。

最后加入级联处理，其中 60 % 的查询由 Qwen3-32B 处理（自托管，输入 $0.10 + 输出 $0.30 每 Mtok）：

- 低成本路径每次调用：$0.10 \cdot 4 + 0.30 \cdot 0.5 = 0.55$ 千 tok 分 = $0.00055。
- 混合每日成本：$0.6 \cdot 7000 \cdot 0.00055 + 0.4 \cdot 7000 \cdot 0.0124 = $2.31 + $34.72 = $37/天。乘以 30 天 = $1110/月，比基线便宜 84 %。

这些节省是叠加的。没有显式成本跟踪，根本看不到这些优化效果。
## 成本之外的可观测性

![fig5: 可观测性仪表盘](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig5_observability_dashboard.png)

标准可观测性包括 trace、log 和 metric，再加上 LLM 专用指标：

- 统计每个模型的延迟和错误率，按模型分开。
- 自托管时关注每个副本的 token 吞吐量。吞吐量下降通常说明负载有问题。
- 抽取 1% 的请求跑评估集评分流程，画出通过率随时间的变化曲线，观察质量漂移。
- 检测 Prompt 注入，统计每个用户的注入尝试频率。
- 记录模型拒绝的比例。拒绝率突然变化，可能是攻击或模型退化的信号。
- 监控 prompt 缓存和语义缓存的命中率。命中率下降可能是因为模板改动或流量模式变化。

到 2026 年依然好用的工具：

- **Langfuse**（开源）。最适合 LLM 的 tracing 工具，免费版功能够用，自托管也很容易。
- **Helicone**。托管式 LLM 可观测性工具，带成本跟踪，开箱即用。
- **Phoenix（Arize）**。开源工具，专注于检索和 RAG 质量指标。
- **OpenLLMetry**。基于 OpenTelemetry，能无缝接入 Datadog、Grafana、Honeycomb 等现有平台。
- **Datadog + 自定义仪表盘**。如果你已经在用 Datadog，想统一界面查看数据，这是个不错的选择。

最省钱的方案是记录每个请求的结构化日志，发送到 ClickHouse 或 Snowflake，再搭建一个仪表盘。不用买托管产品也能拿到 80% 的价值，但前提是坚持记录关键字段。
## 能用的 on-call 模式

![fig6: on-call 升级流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig6_oncall_escalation.png)

凌晨 3 点把我叫醒的问题，以及对应的处理步骤：

**LLM 供应商 5xx 错误激增。** 先看供应商的状态页面。有备用供应商就切过去。主供应商 5 分钟内恢复就不管；超过 5 分钟，联系客户经理。记得检查备用供应商的成本，有时贵一倍。

**延迟 p99 > 30 秒。** 一般是工具调用失控或请求卡住。先查是不是某个用户或功能占了大头。杀掉卡住的请求，收紧超时限制，问题通常能解决。

**成本突然暴涨。** 每小时花费涨了 10 倍。查用户表，基本是一个用户的 agent 出问题。先限流，再找根本原因。最常见的问题是 agent 没设递归步数限制，导致无限循环调用自己。

**拒绝率突然上升。** 模型一夜之间多拒绝 30% 的请求。要么是模型更新出问题，要么有人搞协同攻击。先回滚模型，再调查。

**RAG 质量下降。** faithfulness 分数掉了 10%。通常是 embedding 或 index 版本不匹配，或者新内容格式变了。检查数据摄入管道。

**自托管 GPU 内存不足（OOM）。** KV 缓存耗尽了。要么加容量，要么降低 `max_tokens` 参数。别急着重启服务，下一批请求还是会 OOM。

**供应商弃用通知。** 我们依赖的模型版本 90 天内会被移除。立刻安排迁移测试，针对新版本做至少 2 周 A/B 测试，记录差异。

好的 runbook 能把平均恢复时间从 30 分钟缩短到 3 分钟。一定要在问题发生前写好。
## 值得命名的常见失败模式

以下是我见过或读到的一些生产事故，列个不完全清单：

- **供应商区域故障。** 整个可用区（AZ）挂了。解决办法：网关层做多区域切换；托管 API 准备备用供应商。
- **限流级联效应。** 供应商对某个客户限流，我的应用疯狂重试，耗光全局配额。解决办法：网关前加每用户限流；重试用带抖动的指数退避；每个供应商配熔断机制。
- **Prompt 模板问题。** 改系统 Prompt 时以为“无害”，结果缓存失效（第 9 章），账单一夜翻三倍。解决办法：CI 加入缓存感知的差异检查；缓存命中率下降就告警。
- **模型弃用意外。** OpenAI 宣布 gpt-4-turbo-0613 六十天后退役，但我没用替代模型跑过评估集。解决办法：订阅供应商的弃用通知；每季度演练迁移。
- **静默质量下降。** 托管模型悄悄更新，特定场景性能变差。解决办法：每周用生产流量样本跑评估集；对比历史基线；质量下降超 5% 就告警。
- **Agent 循环失控。** Agent 调工具出错，重试遇到新错误，模型开始自我争辩，最后花了 $500。解决办法：限制每次对话的步骤数和成本；单次会话超 $X 就告警。
## 迁移 playbook

在生产环境里，模型迁移是个被严重低估的技能。分享一个我踩过的坑和总结的方法。

1. **影子流量跑 3-7 天。** 新模型接收所有请求的副本，输出只记录不返回。离线对比质量差异。
2. **灰度发布 1%，持续 3-7 天。** 新模型处理 1% 的真实流量。监控延迟、错误率、成本、拒绝率和满意度。没问题就继续。
3. **逐步放量：5% → 25% → 50% → 100%。** 整个过程 1-2 周。每一步都必须通过 SLA 验证。
4. **切换后旧模型热备 30 天。** 回滚只需改一个配置。
5. **确保 30 天无异常再下线旧模型。**

跳过影子流量和灰度发布，最容易导致“上线新模型后客户投诉激增”。这是流程纪律问题，不是技术问题。
## 按部署形态的成本（约值，2025 年底）

| 配置 | $/Mtok 输入 | $/Mtok 输出 | 备注 |
|---|---|---|---|
| Claude-4.5-Sonnet API | $3 | $15 | 质量优秀 |
| GPT-4o API | $2.50 | $10 | 质量优秀 |
| Qwen3-Max API | $1.40 | $5.60 | 质量好，价格低 |
| Gemini-2.5-Pro API | $2.50 | $10 | 质量好，支持长上下文 |
| DeepSeek-V3 API | $0.14 | $0.28 | 便宜，数学和代码能力强 |
| 自托管 Qwen3-32B FP8（1xH100） | $0.10 | $0.30 | GPU 利用率 70% |
| 自托管 LLaMA-3.3-70B FP8（2xH100） | $0.30 | $0.90 | GPU 利用率 70% |
| 自托管 Qwen3-235B-A22B（8xH100） | $0.50 | $1.50 | GPU 利用率 70% |
| Open-router 池化小模型 | $0.15 | $0.50 | 最便宜的选择 |
| Anthropic Batch API（5 折） | $1.50 | $7.50 | 24 小时 SLA |
| OpenAI Batch API（5 折） | $1.25 | $5 | 24 小时 SLA |

输出成本一般是输入的 3 到 5 倍。原因很简单，输出是顺序解码，受内存限制。如果任务需要生成长内容，比如总结、报告或代码，换个输出成本低的模型，效果可能比优化模型本身更好。

Batch API 的折扣是个被低估的省钱利器。工作负载接受 24 小时 SLA 的话，费用直接减半。典型场景包括隔夜报告、批量审核、评估集生成或数据标注。迁移也很简单，改一行 API 调用代码就行。
## 性价比生产 LLM 产品的最终配方

如果我在 2026 年中启动一个生产级 LLM 产品，我会这么做：

1. 前 6 到 12 个月直接用托管 API。根据延迟、质量和语言需求选一个合适的。
2. 第一天就搭建 LLM 网关，哪怕只对接一个供应商。迟早会用到。
3. 记录每次请求的成本，精确到小数点后两位（$0.0001 级别）。每周汇总一次。
4. 先准备一个 200 道题的评估集，再优化提示词。
5. 如果有外部知识库，加上检索和重排序（reranker）功能。

6. 尽可能多地缓存。在供应商端缓存提示词。对应用内重复问题做语义缓存。对确定性流程的结果做缓存。
7. 单次提示无法解决真实用户请求时，引入工具使用（tool use）。
8. 简单查询交给便宜模型处理，覆盖 60%-80% 的流量。复杂查询交给高性能模型。
9. 月均 token 超过 10 亿且流量稳定时，切换到自托管。

10. H100 上用 FP8 量化，A100 和 L40 上用 AWQ-INT4 量化。
11. 对于能接受 24 小时 SLA 的任务，直接用批量 API（batch API）。
12. 提前搭建安全栈，别等到出问题才动手。

这套方案足够支撑产品上线和增长，同时避免了拖垮大多数 LLM 产品的技术债务。
## 系列收尾

12 篇文章，讲透如何从零打造一个现代 LLM 产品：

1. 架构  
2. Tokenization  
3. 预训练  
4. Post-training  
5. 推理优化  
6. 长上下文  
7. Function calling  
8. RAG  
9. Prompting  
10. 评估  
11. 安全  
12. 生产  

记住三点就够了：数据质量是王道（第 3、4 篇）。Tokenization 和 KV cache 是命根子（第 2、5 篇）。没有 eval set 和成本跟踪，别提上线（第 10、12 篇）。其他全是执行细节。

想深入研究的话，[NLP 系列](/zh/nlp/) 讲基础理论更扎实。[阿里云百炼系列](/zh/aliyun-bailian/) 借助具体云平台展示这些模式。[阿里云 PAI 系列](/zh/aliyun-pai/) 聚焦阿里云的训练和服务基础设施。

感谢你读到这里。动手去建一个真正能用的东西吧。
## 参考资料

- Chen, L. 等（2023）。*FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*。https://arxiv.org/abs/2305.05176
- Hu, Q. 等（2024）。*RouteLLM: Learning to Route LLMs with Preference Data*。https://arxiv.org/abs/2406.18665
- Bang, Y. 等（2023）。*GPTCache: An Open-Source Semantic Cache for LLM Applications*。NLP-OSS @ EMNLP 2023。https://arxiv.org/abs/2311.04205
- Kwon, W. 等（2023）。*Efficient Memory Management for Large Language Model Serving with PagedAttention*（vLLM）。SOSP 2023。https://arxiv.org/abs/2309.06180
- LiteLLM 项目。*LiteLLM: Call all LLM APIs using the OpenAI format*。https://github.com/BerriAI/litellm
- Cloudflare（2024）。*Cloudflare AI Gateway*。https://developers.cloudflare.com/ai-gateway/
- OpenRouter（2025）。*OpenRouter: A unified interface for LLMs*。https://openrouter.ai/
- Langfuse 项目。*Langfuse: Open-source LLM engineering platform*。https://langfuse.com/
- OpenLLMetry 项目。*OpenLLMetry: Open-source observability for LLM applications*。https://www.traceloop.com/openllmetry
- Anthropic（2024）。*Message Batches API*。https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
- OpenAI（2024）。*Batch API*。https://platform.openai.com/docs/guides/batch
- Anyscale（2023）。*Reproducing GPT-2 and Llama-2 inference cost analysis*。https://www.anyscale.com/blog/llm-inference-cost-analysis
- Together AI（2024）。*Together Inference Engine performance and cost benchmarks*。https://www.together.ai/blog/together-inference-engine-2
- Fireworks AI（2024）。*FireAttention serving stack*。https://fireworks.ai/blog/fire-attention-serving-stack
- HuggingFace TGI 项目。*Text Generation Inference*。https://github.com/huggingface/text-generation-inference
