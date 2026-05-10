---
title: "大模型工程（十二）：生产 —— 部署、监控、成本"
date: 2026-05-06 15:00:00
tags:
  - LLM
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
这是最后一章了。前面我们聊了模型、Prompt、检索、评估，这一章只关心两件事：让服务跑稳，别破产。生产环境的 LLM 服务，其实更像高流量 Web 服务，而不是传统 ML 服务——只不过每个 Web 请求都要烧钱，而且可能卡上两分钟。

这一章我会多摆点数据。原因很简单：在生产环境，一个功能到底赚钱还是亏钱，往往就差在那没人盯的 2-5 倍成本系数上。最有用的技能就是手算 LLM 负载的成本账。下面的数据截止到 2025 年底/2026 年初，真要用之前记得核对最新定价。

![LLM Engineering (12): Production — Deployment, Monitoring, Cost — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/illustration_1.png)

## 端到端的服务栈

![fig1: end-to-end serving stack architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig1_stack_architecture.png)

生产环境的 LLM 应用栈通常长这样：

```
[CDN / WAF]
   ↓
[API Gateway]   ← rate limiting, auth, request normalization
   ↓
[App Server]    ← prompt assembly, RAG retrieval, tool execution, agent loop
   ↓
[LLM Gateway]   ← model routing, fallback, cost accounting, prompt caching
   ↓
[LLM Service]   ← vLLM/SGLang/managed API
   ↓
[Observability] ← logs, metrics, traces, eval runs
```

App Server 和 LLM Gateway 是工程重心。App Server 处理业务逻辑；LLM Gateway 则要让一堆模型表现得像单个服务。

从第一天起就把 LLM Gateway 建成独立服务。你需要它来做这些事：

- **多模型路由**：根据分类器，把某些请求发给小而快的模型，其他的发给大而慢的。
- **降级（Fallback）**：主提供商返回 5xx 时，自动重试备用提供商。
- **成本追踪**：记录每个请求的 prompt tokens、completion tokens、模型和美元成本。
- **Prompt 缓存包装**：即使提供商支持缓存，你的业务代码也不该关心 cache key 怎么生成。
- **A/B 测试**：把可配置比例的流量路由到新的模型 variant。
- **配额/熔断**：当单个用户消耗成本 disproportionate 时，直接切断请求。

从头构建这个大概需要几周工程量。开源选项覆盖了不少场景：

- **LiteLLM** — 纯 Python 代理，集成 100+ 提供商，提供 drop-in OpenAI 兼容端点，内置成本追踪不错。最快上手方案。
- **OpenRouter** — 托管网关，单一 API 对接所有提供商，内置模型市场。单 token 成本比直连提供商高，但自动处理 failover 和价格套利。
- **Cloudflare AI Gateway** — CDN 边缘的托管代理，带缓存、限流和分析。便宜且运维 trivial，但牺牲了一些灵活性。
- **BentoML / Bento Cloud** — 更重的框架，适合同时需要在这个网关后托管自训模型的场景。
- **Portkey, Langfuse Gateway** — 新入局者，观测性故事讲得比较好。

早点选一个，但也做好以后换掉的准备。Gateway 是少数真正存在厂商锁定的地方（你会对着网关的契约写大量代码）；所以要把这个契约保持精简。

## 自建 vs 托管 API

这是个老生常谈的问题。2026 年的实话是这样：

**使用托管 API 当：**

- 你需要 frontier-model 质量（GPT-5, Claude-4.5, Gemini-3）且没有 GPU 资源托管它们。
- 用量低于 ~10 亿 tokens/月 — 这个规模下托管通常更便宜。
- 延迟要求宽松（>500 ms TTFT 可接受）。
- 你无法 dedicate 工程师去搞 GPU 运维。

**自建当：**

- 用量高于 10 亿 tokens/月 且开源模型符合质量要求（Qwen3-32B+, LLaMA-3.3-70B+）。
- 有严格的数据驻留要求。
- 需要持续 <100 ms TTFT。
- 有特定的 fine-tuning 需求。

盈亏平衡点大概在每月 10 亿 tokens（Qwen3-32B 级别 workload）。低于这个数，算上工程时间，托管 API 比自建划算。高于这个数， dedicated GPUs（租或买）自建成本能低 3-10 倍。

常见错误：自建 GPU 利用率太低。4xH100 部署如果利用率只有 30%，单 token 成本比 OpenAI API 还贵。目标得是持续吞吐量 >70%。

2025-2026 流行的一种实用混合模式：**自建小模型扛大部分便宜流量，把难的 5-10% 路由给托管 frontier API。** 这样既拿到了自建的大部分成本优势，又在关键地方保留了 frontier 质量。路由决策由一个小分类器完成（见下文多模型路由章节）。

## 多模型路由与 FrugalGPT

![LLM Engineering (12): Production — Deployment, Monitoring, Cost — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/illustration_2.png)

这种先路由再级联的模式，Chen 等人在 2023 年的 *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance* 里 formalize 了。核心观察：大多数 LLM 查询很简单，小/便宜模型就能搞定；只有一小部分真需要 frontier model。如果你能低成本区分这两者，就能在不怎么损失质量的情况下削减 5-10 倍支出。

FrugalGPT 提出了三种模式：

1. **Prompt adaptation** — 压缩 prompts，去掉冗余 context，batch 相关请求。
2. **LLM cascade** — 先试最便宜的模型，让它给个置信度信号，如果置信度低再 fallback 到更贵的模型。
3. **Model routing** — 用 learned classifier  upfront 挑选合适的模型。

Cascade 是生产环境部署最多的。伪代码：

```python
def cascade(question, models=[CHEAP, MID, EXPENSIVE]):
    for m in models:
        answer, confidence = m.generate_with_confidence(question)
        if confidence > THRESHOLDS[m]:
            return answer
    return answer  # last model's output
```

难点在于置信度信号。选项有：

- **自报置信度** — 问模型“给你的回答打个 1-5 分的置信度”。和正确率轻相关，但噪声大。
- **基于 Logprob** — 用 answer span 的 logprob。合理，但只有你能控制 sampling 时才管用。
- **采样一致性** — 采样便宜模型 3-5 次；如果全一致就接受。不一致就升级。
- **任务特定 verifier** — 代码类“过公共测试了吗？”；数学类“答案在合理范围内吗？”

基于 Embedding 的路由（Hu et al., 2024, *RouteLLM: Learning to Route LLMs with Preference Data*）在离线偏好数据上训练 (query, best-model) 对的分类器，学习 upfront 预测用哪个模型。RouteLLM 训练的路由器在 MT-Bench 上能达到 GPT-4 95% 的质量，成本只有 26-44%。

在生产环境，路由决策往往简化为两个问题：“这查询简单吗？”（路由给便宜的）和“这是已知难点模式吗？”（路由给贵的）。中间地带走默认模型。50 问 heuristic 分类器能拿到 80% 的节省；learned routers 大概能拿到 90%。

## 缓存层级

三层缓存很重要，而且它们是叠加的：

1. **提供商侧 Prompt 缓存**（第 9 章）— 针对重复 prompt prefix 的 KV-cache 复用。Anthropic / OpenAI / DeepSeek / Google 都在跑。缓存部分成本降低 90%。
2. **应用侧语义缓存** — 当两个用户查询 *语义相似* 时，直接服缓存答案。工具：GPTCache（原创，基于 FAISS）、Redis Semantic Cache、MemCache + 自定义 embedding  lookup。FAQ 类 workload 命中率 30-70%，开放式聊天接近 0%。
3. **确定性函数的结果缓存** — 如果 LLM 输出 feeds into 下游 pipeline（比如分类、提取）且输入完全匹配，直接把输出存下来，key 用输入 hash。相同输入绝不调用第二次 LLM。

GPTCache 值得细看。模式：用小 embedding 模型嵌入每个用户查询，跟缓存比（cosine similarity），如果相似度 > 阈值（通常 0.95）就服缓存答案。阈值就是旋钮 — 越高假阳性越少但命中率越低。对于回答常见问题的客服机器人，0.92 阈值能带来 50% 成本降低；对于每个查询都独特的代码生成助手，语义缓存基本 ~没用。

可组合性很重要：Prompt 缓存降低 *每次* 调用的成本；语义缓存减少调用 *次数*。两者是相乘关系。

## LLM 负载的自动扩容

![fig4: autoscaling LLM workloads](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig4_autoscaling.png)

LLM 的自动扩容比无状态 Web 服务器难得多。三个原因：

1. **加载模型需要几分钟**。带 70B 模型的新 vLLM  replica 需要 2-4 分钟加载权重和预热。如果流量峰值只持续 10 分钟，靠扩容响应太慢。
2. **GPU 按小时计费，不是按请求**。为了 5 分钟的峰值启 fresh GPU 实例，会浪费 55 分钟的 GPU 钱。
3. **Continuous batching 负载非线性**。50% 负载和 90% 负载的 vLLM 服务器延迟 profile 差别很大；安全操作点比 CPU 服务建议的低得多。

实用模式：

- **定时预热**：如果流量在 9 点 predictable 峰值，8:50 提前扩容。
- **保守扩容，激进缩容**：60% 利用率时扩容，30% 时缩容。（跟典型 Web 服务反过来。）
- **缓冲降级**：自建满容量时，overflow 到托管 API。单位成本高但能吸收峰值而不浪费 provisioning。
- **多模型保底**：即使流量低，每个模型也至少保持一个 replica 在线。冷启动成本太高。
- **带背压的队列**：服务器过载时，用 per-user 公平调度器 queue 请求，返回带 `Retry-After` 头的 429，而不是让所有人的 latency 崩掉。

vLLM 支持 `--max-num-seqs`（最大并发请求数）和 `--max-num-batched-tokens`（每步最大 token 吞吐）。调优这些参数以匹配你的延迟目标，而不是最大化吞吐。
## 延迟预算拆解

![fig2: latency budget breakdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig2_latency_breakdown.png)

面向用户的聊天产品，延迟预算大概长这样：

| Component | Budget (ms) | Notes |
|---|---|---|
| Network in | 50 | Geographic |
| API gateway / auth | 10 | Should be fast |
| App server logic | 50 | RAG embedding, tool dispatch |
| RAG retrieval | 100 | Vector DB + reranker |
| LLM gateway overhead | 5 | Just routing |
| LLM TTFT (queue + prefill) | 300 | The model itself |
| Network out | 50 | Same as in |
| **Total to first token** | **~565 ms** | Below 1s target |
| LLM ITL (decode) | 25 | Per-token, sustains 40 tok/s |
| Network buffering | 20 | Smoothing |

最关键的两个延迟指标：**TTFT**（Time To First Token）和 **ITL**（Inter-Token Latency）。TTFT 决定了用户感觉“有反应了吗？”；ITL 决定了 token 流式输出后的持续阅读速度。2026 年的行业经验值：

- 聊天：TTFT < 800 ms 算“即时”，< 2 s 算“可接受”
- 语音：TTFT < 300 ms 才能实现自然对话
- 代码补全：TTFT < 200 ms，否则用户直接放弃
- 批量 / 代理任务：TTFT 可以放宽到 5-30 s；用户能容忍

首 token 出来后，用户看到流式输出，通常能接受 5-10 秒内 ~40-60 tok/s 的速度。总响应时间取决于长度：200 tokens 按 40 tok/s 算 = 5s，加上 600 ms TTFT = 端到端 5.6 s。

延迟预算花在哪：

- **Reranking**：花 50-100 ms 很值（第 8 章）。
- **Speculative decoding**：ITL 降低 50 到 100 ms（第 5 章），好投资。
- **Tool calls**：一个 200 ms 的工具调用会卡住模型。能并行就并行，缓存要激进。
- **Reasoning / thinking**：思考型模型会在首 token 可见前增加 1-10 s。只留给需要深思的任务。

对于延迟敏感的路径，*得实测时间花在哪了*。实际情况往往跟你想的不一样。我见过一个“慢”功能，LLM 只花了 200 ms，但 RAG reranker 花了 1.2 s；还有一个 LLM 没问题，但下游 JSON 序列化花了 800 ms。先链路追踪，再优化。

## 从第一天就开始跟踪成本

![fig3: cost per request breakdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig3_cost_per_request.png)

按请求核算成本是没得商量的底线。没这个，你没法：

- 找出单个用户或功能导致的成本膨胀。
- 做成本 vs 质量的 A/B 测试。
- 在账单到来前预测支出。
- 在失控的 Agent 循环花掉 1000 美元前抓住它。

每个请求至少记录以下字段：

```python
{
    "request_id": "req_xxx",
    "user_id": "user_xxx",
    "endpoint": "/chat",
    "model": "claude-4-5-sonnet-20250901",
    "prompt_tokens": 4321,
    "cached_tokens": 3000,    # of those, came from cache
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

按用户、功能、模型聚合数据。出现以下情况要报警：

- 单个用户 1 小时内支出超过中位数的 10 倍。
- 某功能的单次调用成本周环比翻倍。
- 新模型的单次解决对话成本高于旧模型，尽管质量相当。

算笔账很有必要。假设你的产品每天调用 1 万次 LLM，平均 4K 输入 + 500 输出 tokens，使用 Claude-4.5 Sonnet：

- 单次调用：$4 \cdot 4 + 15 \cdot 0.5 = 16 + 7.5 = 23.5$ thousand-tok-cents = $0.0235。
- 每天：$235。
- 每月：$7050。

现在加上提示词缓存（4K 输入中有 3K 命中缓存）：

- 单次调用：$4 \cdot 1 + 0.30 \cdot 3 + 15 \cdot 0.5 = 4 + 0.9 + 7.5 = 12.4$ thousand-tok-cents = $0.0124。
- 每月：$3720，节省 $3330 (47 %)。

现在加上 30% 的语义缓存命中率：

- 每月：$3720 \cdot 0.7 = $2604，总节省 $4446 (63 %)。

现在加上级联路由，60% 的查询由 Qwen3-32B 处理（自托管，每 Mtok 输入 $0.10 + 输出 $0.30）：

- 便宜路径单次：$0.10 \cdot 4 + 0.30 \cdot 0.5 = 0.55$ thousand-tok-cents = $0.00055。
- 混合每月：$0.6 \cdot 7000 \cdot 0.00055 + 0.4 \cdot 7000 \cdot 0.0124 = $2.31 + $34.72 ... 每天。乘以 30 天 = $1110/月，比基线便宜 84 %。

这些节省是复利的。不显式跟踪成本，这些都看不见。

## 超越成本的可观测性

![fig5: observability dashboard](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig5_observability_dashboard.png)

标准可观测性（链路、日志、指标）加上 LLM 特有的：

- **分模型延迟和错误率**，按模型细分。
- **每副本 Token 吞吐量**，针对自托管（这里回退通常意味着模型负载问题）。
- **质量漂移**：采样 1% 的请求，跑一遍评估集的 grading 流程，绘制随时间变化的通过率。
- **提示词注入检测**：每用户检测到的注入尝试率。
- **拒绝率**：模型拒绝的比例。突然变化意味着攻击或模型回退。
- **缓存命中率**：提示词缓存、语义缓存。命中率下降意味着提示词模板变更破坏了缓存，或流量发生了偏移。

2026 年好用的工具：

- **Langfuse**（开源）— 最适合 LLM 特定链路追踪，免费额度大方，自托管容易。
- **Helicone** — 托管型 LLM 可观测性，带成本跟踪；开箱即用性强。
- **Phoenix (Arize)** — 开源 LLM 可观测性，专注检索/RAG 质量指标。
- **OpenLLMetry** — 基于 OpenTelemetry，可插入现有可观测性栈（Datadog, Grafana, Honeycomb）。
- **Datadog + 自定义仪表盘** — 如果你已有 Datadog 且想要单一视图，这个可行。

最省钱且有效的方案：结构化记录每个请求，投喂到 ClickHouse 或 Snowflake，建个仪表盘。不需要托管的可观测性产品也能拿到 80% 的价值，但得有纪律性去记录正确的字段。

## 行之有效的 On-call 模式

![fig6: on-call escalation flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/12-production/fig6_oncall_escalation.png)

凌晨 3 点把我叫醒的事，以及对应的处理手册：

**LLM 提供商 5xx 激增。** 查提供商状态页。如果有备用提供商，切换过去。如果主提供商在 <5 分钟内恢复，无需进一步操作；如果更久，联系提供商客户经理。检查故障切换是否导致备用提供商成本激增（有时备用贵 2 倍）。

**延迟 p99 > 30s。** 通常是失控的工具调用或卡住的请求。检查是否单个用户/功能占主导。通常杀掉卡住的请求并添加更严格的超时就能解决。

**成本激增**：$/小时 突然增加 10 倍。查用户表——几乎总是一个用户有失控的 Agent。限流，查根因。经典案例是 Agent 递归调用自己且没有步数限制。

**拒绝率跳变**：模型一夜之间多拒绝了 30% 的请求。要么模型发布回退了，要么有协同攻击。回滚模型，调查。

**RAG 质量下降**：faithfulness 分数下降 10%。通常是 embedding/index 版本不匹配，或新内容格式不同。检查 ingestion  pipeline。

**自托管 GPU OOM**：KV cache 耗尽。要么加容量，要么降低每请求 `max_tokens`。别盲目重启；下一批请求照样 OOM。

**提供商弃用通知**：我们依赖的模型版本将在 90 天内移除。立即安排针对推荐继任者的迁移测试，A/B 至少 2 周，记录差异。

好的处理手册能把 MTTR 从 30 分钟降到 3 分钟。手册得在需要之前写好。

## 值得命名的常见故障模式

我在生产环境见过或听说过的事故清单，不全但典型：

- **提供商区域宕机。** 整个 AZ 挂了。缓解：网关层多区域故障切换；对于托管 API，备用提供商待命。
- **限流级联。** 提供商限流了你的一个客户；你的应用 aggressive 重试吃掉了全局配额。缓解：网关前每用户限流；重试带指数退避和抖动；每提供商熔断器。
- **提示词模板回退。** 系统提示词“无害”变更导致提示词缓存失效（第 9 章），账单一夜翻三倍。缓解：CI 中感知缓存的 diff；缓存命中率下降报警。
- **模型弃用 surprise。** OpenAI 宣布 gpt-4-turbo-0613 将在 60 天内退役；你的评估集从未在替代品上跑过。缓解：订阅提供商弃用源；每季度迁移演练。
- **静默质量漂移。** 托管模型收到未文档化的更新，在你的特定用例上回退。缓解：每周用评估集跑生产流量样本；对比历史基线；下降 >5% 报警。
- **Agent 循环失控。** Agent 调用工具，收到错误，重试，得到不同错误，重试时模型开始自我争论，累计花费 500 美元。缓解：每对话步数限制；每对话成本限制；每会话成本 > $X 报警。

## 迁移 playbook

模型迁移是被低估的生产技能。有效的模式：

1. **Shadow 流量 3-7 天。** 新模型接收每个请求的副本，但只记录输出，不返回。离线对比质量。
2. **Canary 1% 持续 3-7 天。** 新模型服务 1% 的真实流量；监控延迟、错误率、成本、拒绝率、满意度。无回退 → 继续。
3. **逐步放量 5 → 25 → 50 → 100%**，耗时 1-2 周。每步都卡在监控 SLA 上。
4. **切换后保留旧模型热备 30 天。** 回滚开关只需改一个配置。
5. **30 天 clean 运行后才下线。**

跳过 shadow + canary 是“上新模型后客诉飙升”的最常见原因。这是流程纪律，不是技术问题。

## 不同部署形态的成本（2025 年末粗略数据）

| Setup | $/Mtok input | $/Mtok output | Notes |
|---|---|---|---|
| Claude-4.5-Sonnet API | $3 | $15 | Strong quality |
| GPT-4o API | $2.50 | $10 | Strong quality |
| Qwen3-Max API | $1.40 | $5.60 | Strong, cheaper |
| Gemini-2.5-Pro API | $2.50 | $10 | Strong, long context |
| DeepSeek-V3 API | $0.14 | $0.28 | Cheap, strong on math/code |
| Self-host Qwen3-32B FP8 (1xH100) | $0.10 | $0.30 | At 70 % util |
| Self-host LLaMA-3.3-70B FP8 (2xH100) | $0.30 | $0.90 | At 70 % util |
| Self-host Qwen3-235B-A22B (8xH100) | $0.50 | $1.50 | At 70 % util |
| Open-router pooled small models | $0.15 | $0.50 | Cheapest |
| Anthropic Batch API (50 % discount) | $1.50 | $7.50 | 24h SLA |
| OpenAI Batch API (50 % discount) | $1.25 | $5 | 24h SLA |

输出通常比输入贵 3-5 倍，因为输出是串行 decode（内存受限）。对于产生长输出的应用（摘要、报告、代码），切换到输出经济学更便宜的模型，往往比优化模型本身更有效。

Batch API 折扣（24 小时 SLA  workload 打五折）是最被低估的省钱杠杆。如果你的 workload 能容忍次日完成（隔夜报告、批量内容审核、评估集、数据集标注）， batching 能把支出砍半。迁移通常只需改一行 API 代码。
## 打造高性价比生产级 LLM 产品的最终方案

要是让我在 2026 年年中重新启动一个生产级的 LLM 产品，我会这么做：

1. **前 6-12 个月，默认直接用托管 API**。按你的延迟、质量和语言需求来选供应商。
2. **从第一天起就构建 LLM Gateway**，哪怕它只是包了一层单一供应商。后面你一定会需要它。
3. **精确追踪每次请求的成本**，保留两位小数（比如 $0.0001）。每周汇总一次。
4. **在调整 Prompt 之前，先准备好包含 200 个问题的评估集**。
5. **只要涉及外部知识库，就加上检索 + 重排序（reranker）层**。
6. **缓存策略要激进**：供应商处的 Prompt 缓存、应用层针对重复问题的语义缓存、确定性流程的结果缓存。
7. **单次 Prompt 搞不定用户真实需求时，就上工具调用（tool use）**。
8. **60-80% 的简单查询路由到便宜模型**，难啃的骨头再交给前沿模型。
9. **当月 token 用量突破 10 亿且流量稳定时，再考虑自建托管**。
10. **H100 上用 FP8 量化**，A100/L40s 上用 AWQ-INT4。
11. **任何能接受 24 小时 SLA 的任务，都用 Batch API**。
12. **安全栈（safety stack）要提前部署**，别等出了事再加。

做到这些，足够让你上线产品并持续增长，同时避开那些搞死大多数 LLM 产品的技术债。

## 系列总结

这十二篇文章讲透了从头到尾构建现代 LLM 产品所需的一切：

1. 架构
2. 分词
3. 预训练
4. 后训练
5. 推理优化
6. 长上下文
7. 函数调用
8. RAG
9. Prompting
10. 评估
11. 安全
12. 生产落地

如果只让你记住三件事：数据质量决定一切（第 3 & 4 章），分词和 KV cache 是成本大头（第 2 & 5 章），没有评估集和成本追踪就别想上线（第 10 & 12 章）。剩下的全是执行细节。

想深入阅读的话，[NLP 系列](/zh/nlp/) 会更深入地讲解基础，[Aliyun Bailian 系列](/zh/aliyun-bailian/) 展示了特定云平台上的相同模式，而 [Aliyun PAI 系列](/zh/aliyun-pai/) 则涵盖了阿里云上的训练 - 服务基础设施。

感谢读到这儿。去构建一些真正能跑起来的东西吧。

## 参考文献

- Chen, L. et al. (2023). *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*. https://arxiv.org/abs/2305.05176
- Hu, Q. et al. (2024). *RouteLLM: Learning to Route LLMs with Preference Data*. https://arxiv.org/abs/2406.18665
- Bang, Y. et al. (2023). *GPTCache: An Open-Source Semantic Cache for LLM Applications*. NLP-OSS @ EMNLP 2023. https://arxiv.org/abs/2311.04205
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP 2023. https://arxiv.org/abs/2309.06180
- LiteLLM project. *LiteLLM: Call all LLM APIs using the OpenAI format*. https://github.com/BerriAI/litellm
- Cloudflare (2024). *Cloudflare AI Gateway*. https://developers.cloudflare.com/ai-gateway/
- OpenRouter (2025). *OpenRouter: A unified interface for LLMs*. https://openrouter.ai/
- Langfuse project. *Langfuse: Open-source LLM engineering platform*. https://langfuse.com/
- OpenLLMetry project. *OpenLLMetry: Open-source observability for LLM applications*. https://www.traceloop.com/openllmetry
- Anthropic (2024). *Message Batches API*. https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
- OpenAI (2024). *Batch API*. https://platform.openai.com/docs/guides/batch
- Anyscale (2023). *Reproducing GPT-2 and Llama-2 inference cost analysis*. https://www.anyscale.com/blog/llm-inference-cost-analysis
- Together AI (2024). *Together Inference Engine performance and cost benchmarks*. https://www.together.ai/blog/together-inference-engine-2
- Fireworks AI (2024). *FireAttention serving stack*. https://fireworks.ai/blog/fire-attention-serving-stack
- HuggingFace TGI project. *Text Generation Inference*. https://github.com/huggingface/text-generation-inference