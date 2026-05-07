---
title: "用 Terraform 给 AI Agent 上云（七）：可观测、SLS 看板与成本告警"
date: 2026-03-24 09:00:00
tags:
  - Terraform
  - 阿里云
  - SLS
  - ARMS
  - CloudMonitor
  - AI Agent
categories: Terraform
lang: zh-CN
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 7
description: "日志进 SLS、Trace 进 ARMS、指标进 CloudMonitor——全部用 HCL 配，新环境天生带观测。真实救过我项目的四条告警，再加上 SLS 驱动的成本看板，发薪日之前告诉你哪个 Agent 在烧预算。"
disableNunjucks: true
translationKey: "terraform-agents-7"
---

Agent 是非确定的、多步的、调昂贵 API 的。这组合意味着如果你不在第一天 instrument 它，事后没法 debug。本篇用 Terraform 打通三条管道——日志、Trace、指标——汇成一个统一看板，再叠四条真正在生产环境救过我项目的告警。

读完之后你拥有一个钉钉群，账单爆掉之前、延迟挂掉之前、错误率飙升之前、某个 Agent 自循环之前，它都会先 ping 你。

![用 Terraform 给 AI Agent 上云（七）：可观测、SLS 看板与成本告警 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/illustration_1.jpg)

## 三条核心数据流

![三种信号、三条管道：日志、链路追踪、指标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig1_obs_pipeline.png)

三种类型的可观测性信号，对应三个阿里云服务，最终都会汇总到 SLS，方便我们以人类可读的方式进行分析和查看：

- **日志** —— Agent 的标准输出（stdout/stderr）→ ECS 实例上的 Logtail 采集器 → SLS Logstore
- **链路追踪** —— Agent 代码中集成的 OpenTelemetry SDK → ARMS APM（兼容 OpenTelemetry）
- **指标** —— CloudMonitor agent 收集的主机指标 + Agent 代码中自定义的业务指标 → CloudMonitor → 可选地通过管道转发至 SLS

不要只盯着“日志”或者“指标”，这三者缺一不可：

- 日志能告诉你：“Agent 到底干了什么？”
- 链路追踪能回答：“时间都消耗在哪些环节？”
- 指标则帮助你判断：“当前的现象是否比平时更频繁发生？”
## 第 1 步：SLS 项目和 logstore

所有可观测的东西都从一个 SLS 项目开始。每个环境一个项目是对的；每个 Agent 一个就太碎了。

```hcl
resource "alicloud_log_project" "agents" {
  name        = "agents-${terraform.workspace}"
  description = "agents-${terraform.workspace} 的日志和指标"
  tags = {
    Environment = terraform.workspace
    ManagedBy   = "terraform"
  }
}

locals {
  logstores = {
    "agent-runs"        = { ttl = 30, shard_count = 4 }
    "gateway-requests"  = { ttl = 90, shard_count = 4 }
    "ecs-syslog"        = { ttl = 14, shard_count = 2 }
    "ack-cluster"       = { ttl = 30, shard_count = 4 }
    "audit"             = { ttl = 365, shard_count = 2 }
  }
}

resource "alicloud_log_store" "this" {
  for_each = local.logstores

  project          = alicloud_log_project.agents.name
  name             = each.key
  shard_count      = each.value.shard_count
  retention_period = each.value.ttl
  auto_split       = true
  max_split_shard_count = 16

  encrypt_conf {
    enable      = true
    encrypt_type = "default"
    user_cmk_info {
      cmk_key_id = module.vpc.kms_keys["logs"]
      arn        = module.vpc.kms_keys["logs"]
      region_id  = "cn-shanghai"
    }
  }
}
```

五个 logstore 覆盖实际需要：

- `agent-runs`——每个 Agent 每一步（消防水管）
- `gateway-requests`——每次 LLM API 调用一行，带 model、tokens、latency、cost
- `ecs-syslog`——ECS 实例底层 OS 日志
- `ack-cluster`——Kubernetes 事件和 Pod 日志（仅 ACK 时）
- `audit`——Terraform 的每次变更，留 1 年合规

`audit` 留一年是因为它体积小，几年后"3 月 12 日是谁改了 prod ALB"会问到。

## 第 2 步：从 ECS 推日志

Logtail 是阿里云官方日志采集器。在 cloud-init 里装（加进第四篇的 `cloud-init.sh`）：

```bash
# 装 Logtail
wget http://logtail-release-cn-shanghai.oss-cn-shanghai.aliyuncs.com/linux64/logtail.sh
chmod +x logtail.sh && ./logtail.sh install cn-shanghai
service ilogtaild start

# 给这台机打标，挂到 SLS machine group
echo "${sls_user_id}::${sls_machine_group}" > /etc/ilogtail/user_log_config.json
```

Logtail 配置——抓哪些文件、怎么解析——是 Terraform resource：

```hcl
resource "alicloud_log_machine_group" "agent" {
  project       = alicloud_log_project.agents.name
  name          = "agent-runtime-machines"
  identify_type = "userdefined"
  identify_list = ["agent-runtime-${terraform.workspace}"]
  topic         = "agent-runs"
}

resource "alicloud_logtail_config" "agent" {
  project      = alicloud_log_project.agents.name
  logstore     = alicloud_log_store.this["agent-runs"].name
  input_type   = "file"
  log_sample   = <<-SAMPLE
    {"ts":"2026-03-24T09:15:23Z","agent":"research","step":"plan","tokens":420,"latency_ms":1200}
  SAMPLE
  name         = "agent-runs-collector"
  output_type  = "LogService"

  input_detail = jsonencode({
    logType        = "json_log"
    logPath        = "/var/log/agents"
    filePattern    = "*.log"
    localStorage   = true
    enableRawLog   = false
    timeKey        = "ts"
    timeFormat     = "%Y-%m-%dT%H:%M:%S%z"
    discardUnmatch = false
    maxDepth       = 10
  })
}

resource "alicloud_logtail_attachment" "agent" {
  project              = alicloud_log_project.agents.name
  logtail_config_name  = alicloud_logtail_config.agent.name
  machine_group_name   = alicloud_log_machine_group.agent.name
}
```

现在任何打了标的机器上 `/var/log/agents/*.log` 都会流进 SLS，作为 JSON 按字段可查。Agent 代码只 `logger.info(json.dumps({...}))`，其余自动。

## 第三步：通过 OpenTelemetry 将 Trace 数据接入 ARMS

在分布式追踪方面，ARMS APM 完全兼容 OpenTelemetry。使用 Terraform 的部分非常简单，只需创建一个 ARMS 实例和对应的环境配置：

```hcl
resource "alicloud_arms_environment" "agents" {
  environment_name      = "agents-${terraform.workspace}"
  bind_resource_id      = module.vpc.vpc_id
  environment_type      = "CS"             # cloud service
  environment_sub_type  = "ECS"
  payment_type          = "POSTPAY"
}
```

Agent 的代码实现基于标准的 OpenTelemetry，完全不需要依赖阿里云特定的 SDK：

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(
        endpoint=os.environ["ARMS_OTLP_ENDPOINT"],
        headers={"Authentication": os.environ["ARMS_LICENSE_KEY"]},
    ))
)

tracer = trace.get_tracer("research-agent")

with tracer.start_as_current_span("research_loop") as span:
    span.set_attribute("agent.name", "research-agent")
    span.set_attribute("session.id", session_id)
    # ... Agent 的具体逻辑 ...
```

上述代码中用到的两个环境变量 `ARMS_OTLP_ENDPOINT` 和 `ARMS_LICENSE_KEY` 都来自 ARMS。其中，`ARMS_OTLP_ENDPOINT` 可以在 ARMS 控制台找到，而 `ARMS_LICENSE_KEY` 则需要从你的账号信息中获取。这两个变量可以通过 Terraform 的输出绑定到 cloud-init 模板中。

最终的效果是：在 ARMS 中，你可以清晰地看到类似“这次 Agent 运行耗时 12 秒，其中有 9 秒花在了第三次调用 qwen-max 上”的详细信息。这种级别的可观测性，能够真正改变你设计和优化 Agent 的方式。
## 第 4 步：通过 CloudMonitor 收集指标

安装 cloud-monitor agent 后，CloudMonitor 会自动采集主机级别的指标（如 CPU、内存、网络等）。如果你使用的是 ACK 节点池，`install_cloud_monitor` 参数已经帮你完成了这一步。而对于 ECS 实例，则可以通过 cloud-init 添加以下脚本：

```bash
wget http://cms-agent-cn-shanghai.oss-cn-shanghai.aliyuncs.com/release/cms_go_agent_install.sh
chmod +x cms_go_agent_install.sh && ./cms_go_agent_install.sh
```

对于应用级别的自定义指标，比如“research-agent 消耗的 token”，可以将其以结构化字段的形式写入 SLS 日志，然后通过 SLS 查询来设置告警。阿里云推荐的模式是将 SLS 作为指标（SLS-as-metrics）来使用；虽然 CloudMonitor 的自定义指标功能也可以实现类似需求，但从 Terraform 配置的角度来看，操作起来会显得不够简洁和直观。
## 第 5 步：成本仪表盘

![用 Terraform 部署 AI Agent（七）：可观测性、SLS 仪表盘与成本告警 —— 视觉化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/illustration_2.jpg)

接下来的部分非常有趣。每当有 LLM 请求到达网关时，网关都会在 `gateway-requests` 中记录一行日志，包含以下字段：

```json
{
  "ts": "2026-03-24T09:15:23Z",
  "agent": "research-agent",
  "model": "qwen-max",
  "input_tokens": 1820,
  "output_tokens": 412,
  "latency_ms": 1230,
  "cost_cny": 0.087
}
```

SLS 支持直接运行 SQL 查询。比如，想查看“每个 Agent 的每日成本”，可以使用以下查询语句：

```sql
* | SELECT date_trunc('day', __time__) AS day,
          agent,
          SUM(cost_cny) AS daily_cost
  FROM log
  GROUP BY day, agent
  ORDER BY day, daily_cost DESC
```

我们可以通过 Terraform 来创建这个仪表盘，代码如下：

```hcl
resource "alicloud_log_dashboard" "cost" {
  project_name   = alicloud_log_project.agents.name
  dashboard_name = "agent-cost-overview"
  display_name   = "Agent 成本概览"

  char_list = jsonencode([
    {
      title = "按 Agent 查看每日成本"
      type  = "line"
      query = "* | SELECT date_trunc('day', __time__) AS day, agent, SUM(cost_cny) AS cost FROM log GROUP BY day, agent ORDER BY day"
      logstore = alicloud_log_store.this["gateway-requests"].name
      display = { xAxis = ["day"], yAxis = ["cost"], yKey = "agent" }
    },
    {
      title = "近 24 小时按模型统计 Token 使用量"
      type  = "pie"
      query = "* | SELECT model, SUM(input_tokens + output_tokens) AS tokens FROM log WHERE __time__ > now() - INTERVAL '24' HOUR GROUP BY model"
      logstore = alicloud_log_store.this["gateway-requests"].name
    },
    {
      title = "按 Agent 查看 P95 延迟"
      type  = "line"
      query = "* | SELECT date_trunc('hour', __time__) AS hour, agent, approx_percentile(latency_ms, 0.95) AS p95 FROM log GROUP BY hour, agent ORDER BY hour"
      logstore = alicloud_log_store.this["gateway-requests"].name
    }
  ])
}
```

完成配置后，打开 SLS 控制台，你就能看到一个实时更新的仪表盘：

![按类别堆叠的每日成本](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig2_cost_dashboard.png)

这个仪表盘能帮你回答一个每个月都会被问到的问题：“到底哪个 Agent 在烧我的预算？”
## 第 6 步：四条告警

四条告警在我交付过的多个 Agent stack 里挣得了它们的位置：

![第一天就该配的四条告警](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig3_alert_rules.png)

### 告警 1：成本上限

```hcl
resource "alicloud_log_alert" "cost_ceiling" {
  project_name   = alicloud_log_project.agents.name
  alert_name     = "daily-cost-ceiling"
  alert_displayname = "每日 LLM 花费 > ¥800"

  query_list {
    chart_title = "today_cost"
    logstore    = alicloud_log_store.this["gateway-requests"].name
    query       = "* | SELECT SUM(cost_cny) AS today FROM log WHERE __time__ > to_unixtime(date_trunc('day', now()))"
    start       = "-1m"
    end         = "now"
    time_span_type = "Truncated"
  }

  condition  = "today > 800"
  schedule_interval = "5m"
  notify_threshold  = 1
  throttling        = "30m"

  notification_list {
    type           = "DingTalk"
    service_uri    = var.dingtalk_webhook
    content        = "今日 LLM 成本 ¥${"{{today}}"} 超 ¥800 预算。看 SLS 成本看板。"
  }

  severity_configurations {
    severity      = 8
    eval_condition = { condition = "today > 800" }
  }
}
```

如果当日 LLM 花费过 ¥800，每 30 分钟响一次。按你的真实预算调阈值。throttling 重要——不设的话每 5 分钟响一次，团队会把群消息免打扰。

### 告警 2：延迟

```hcl
resource "alicloud_log_alert" "latency" {
  project_name   = alicloud_log_project.agents.name
  alert_name     = "agent-step-latency"
  alert_displayname = "p95 Agent step > 8s"

  query_list {
    chart_title = "p95_step"
    logstore    = alicloud_log_store.this["agent-runs"].name
    query       = "* | SELECT approx_percentile(latency_ms, 0.95) / 1000.0 AS p95s FROM log WHERE __time__ > now() - INTERVAL '5' MINUTE"
    start       = "-5m"
    end         = "now"
  }

  condition         = "p95s > 8"
  schedule_interval = "1m"
  notify_threshold  = 3
  throttling        = "15m"

  notification_list {
    type        = "DingTalk"
    service_uri = var.dingtalk_webhook
    content     = "Agent p95 step 延迟 ${"{{p95s}}"}s——用户发现之前去看。"
  }
}
```

`notify_threshold = 3` 意味着连续三分钟超阈值才响——把单次慢 LLM 调用的噪声压住。

### 告警 3：错误率

形状一样，query 是 `SUM(IF(status >= 500, 1, 0)) * 1.0 / COUNT(*) AS err_ratio`，condition `err_ratio > 0.02`。throttling 短一些（5 分钟）因为错误通常是真正持续的事件。

### 告警 4：token 漏（失控循环）

```hcl
resource "alicloud_log_alert" "token_spike" {
  project_name   = alicloud_log_project.agents.name
  alert_name     = "token-anomaly"
  alert_displayname = "Tokens/min > 24h 滚动均值的 2 倍"

  query_list {
    chart_title = "current"
    logstore    = alicloud_log_store.this["gateway-requests"].name
    query       = "* | SELECT SUM(input_tokens + output_tokens) AS now_tpm FROM log WHERE __time__ > now() - INTERVAL '1' MINUTE"
    start       = "-1m"
    end         = "now"
  }

  query_list {
    chart_title = "baseline"
    logstore    = alicloud_log_store.this["gateway-requests"].name
    query       = "* | SELECT AVG(per_min) AS baseline FROM (SELECT date_trunc('minute', __time__) AS m, SUM(input_tokens + output_tokens) AS per_min FROM log WHERE __time__ > now() - INTERVAL '24' HOUR GROUP BY m)"
    start       = "-24h"
    end         = "now"
  }

  condition         = "now_tpm > 2 * baseline"
  schedule_interval = "1m"
  notify_threshold  = 2
  throttling        = "10m"

  notification_list {
    type        = "DingTalk"
    service_uri = var.dingtalk_webhook
    content     = "Token 消耗 ${"{{now_tpm}}"} tpm vs 24h 均 ${"{{baseline}}"}。可能 Agent 失控。"
  }
}
```

这是最值回票价的一条。一个停止条件有 bug 的 Agent 一夜之间能烧 ¥10,000 token；这条告警 2 分钟内就抓到，给你时间杀进程。

## 为什么钉钉？

国内多数工程团队默认钉钉。SLS 原生支持钉钉 webhook。也能扇出到邮件、短信、（通过 webhook）Slack/Teams/飞书。挑你团队凌晨两点会看的那个。

## 那么 ARMS 的告警功能呢？

ARMS 自带告警功能，特别适合用于追踪级别的条件（例如“任何包含超过 30 个 span 的 trace”）。不过对于上面提到的四种告警场景，SLS 已经完全能够胜任，而且这样可以避免将告警逻辑分散到两个系统中。只有在 SLS 无法满足特定需求时，才建议使用 ARMS 的告警功能。
## 成本分析

可观测性确实会带来一定的成本，通常占到整体账单的 10%-15% 左右。不过，相比于因未能及时发现成本异常而导致的损失，这笔投入是非常值得的。

以下是主要服务的成本估算：

- **SLS**：每 GB 数据摄入约 ¥0.35，存储费用为 ¥0.15/GB。对于一个中等流量的 Agent 堆栈，假设每天摄入约 5 GB 数据，则每月的数据摄入费用约为 ¥50，而保留 30 天的数据存储费用则在 ¥20 左右。
- **ARMS APM**：单个环境（最多支持 1 亿次 Span）的月费用约为 ¥600。
- **CloudMonitor**：标准监控指标免费，但自定义指标的费用为 ¥0.005/指标/天。

综合来看，如果需要对一个真实的生产环境 Agent 堆栈实现全面的可观测性，建议每月预算 ¥1000-1500。相比因错过一次成本失控告警而导致的潜在损失，这样的投入显得非常划算。
## 接下来的内容

第八篇文章将带你完成一次端到端的完整演练。我们会把第 2 到第 7 篇文章中介绍的所有模块——包括 `vpc-baseline`、`compute`、`storage`、`gateway` 和 `observability`——整合到一个名为 `research-agent-stack` 的项目中，并通过一次 `terraform apply` 命令让整个项目运行起来。你会看到真实的执行输出、实际的时间消耗，以及完整的模块依赖关系图（DAG）。最后提供的初始代码仓库可以直接 fork，供你进一步探索和使用。
## 应急响应时常用的 SLS 查询模式

告警响了，告诉你**有问题**。接下来的 10 分钟，你得在 SLS 查询框里搞清楚**具体是什么问题**。两年的 Agent 运维经验下来，我整理了一份个人笔记，里面收藏了大约 12 条查询。其中有 6 条我觉得特别值得分享，因为它们具有通用性。

### 查询 1：过去一小时最耗资源的 Agent

```sql
* | SELECT agent,
           COUNT(*) AS calls,
           SUM(input_tokens + output_tokens) AS tokens,
           SUM(cost_cny) AS cost,
           AVG(latency_ms) AS avg_ms,
           approx_percentile(latency_ms, 0.95) AS p95_ms
    FROM log
    WHERE __time__ > now() - INTERVAL '1' HOUR
    GROUP BY agent
    ORDER BY cost DESC
    LIMIT 20
```

这是我用来快速定位“谁最耗资源”的面板。成本飙升告警一触发，跑这条查询，通常 2 秒内就能揪出罪魁祸首。

### 查询 2：按状态码分析错误分布

```sql
status >= 400 |
  SELECT date_trunc('minute', __time__) AS minute,
         status,
         model,
         COUNT(*) AS errors,
         arbitrary(error_message) AS sample_msg
  FROM log
  WHERE __time__ > now() - INTERVAL '30' MINUTE
  GROUP BY minute, status, model
  ORDER BY minute DESC, errors DESC
```

几秒钟就能区分是“DashScope 抛了 500 错误”还是“Agent 发送了无效请求”。`arbitrary(error_message)` 会随机抽一条错误信息，省得你再去深挖。

### 查询 3：每步 Token 消耗分布，排查失控循环

```sql
* | SELECT session_id,
           agent,
           COUNT(*) AS step_count,
           SUM(input_tokens + output_tokens) AS total_tokens,
           MAX(step_index) AS final_step
    FROM log
    WHERE __time__ > now() - INTERVAL '1' HOUR
    GROUP BY session_id, agent
    HAVING step_count > 50 OR total_tokens > 500000
    ORDER BY total_tokens DESC
```

如果某个会话的步骤超过 50 步，或者 Token 总量超过 50 万，基本可以判定是失控的循环。失控告警触发后，用这条查询锁定具体的 session ID，然后 SSH 登录服务器，直接 `kill -9` 干掉问题进程，最后再复盘分析。

### 查询 4：按阶段拆解延迟分布

如果你的 Agent 按阶段（如 planning、retrieval、LLM call、tool exec、reflection）记录了结构化的时间数据，可以用以下查询：

```sql
* | SELECT phase,
           agent,
           approx_percentile(phase_ms, 0.5) AS p50,
           approx_percentile(phase_ms, 0.95) AS p95,
           approx_percentile(phase_ms, 0.99) AS p99,
           COUNT(*) AS samples
    FROM log
    WHERE __time__ > now() - INTERVAL '1' HOUR
      AND phase IS NOT NULL
    GROUP BY phase, agent
    ORDER BY p95 DESC
```

这是我用来追踪“时间都去哪儿了”的看板。它能快速发现某个工具突然变慢了 3 倍（通常是上游 API 出了问题），或者 retrieval 阶段开始拖后腿（可能是因为向量索引需要重新构建）。

### 查询 5：单次会话成本排行榜

```sql
* | SELECT session_id,
           agent,
           MIN(__time__) AS started,
           MAX(__time__) AS ended,
           SUM(cost_cny) AS session_cost
    FROM log
    WHERE __time__ > now() - INTERVAL '24' HOUR
    GROUP BY session_id, agent
    HAVING session_cost > 5
    ORDER BY session_cost DESC
    LIMIT 50
```

这是我的“昂贵会话”排行榜。¥5 是一个很实用的阈值——超过这个值的会话要么是合理的长对话，要么就是有 bug 的循环。每周检查一下排名前 10 的会话，提前发现问题，避免它们演变成明天的事故。

### 查询 6：多阶段 Agent 的漏斗分析

对于那些包含多个阶段（如 `start → plan → tool_call → reflect → answer`）的 Agent，可以用以下查询来分析每个阶段的会话留存情况：

```sql
* | SELECT phase,
           COUNT(DISTINCT session_id) AS sessions,
           COUNT(DISTINCT session_id) * 100.0 /
             FIRST_VALUE(COUNT(DISTINCT session_id)) OVER (ORDER BY phase) AS pct_of_start
    FROM log
    WHERE __time__ > now() - INTERVAL '24' HOUR
    GROUP BY phase
    ORDER BY phase
```

如果在 `tool_call` 阶段突然出现大量流失，说明工具对很多用户失效了。这和“LLM 挂了”或者“planner 太笨”是完全不同的问题，需要单独处理。

把这些查询保存为 SLS 的 Saved Query，可以通过 `alicloud_log_savedsearch` 资源定义：

```hcl
resource "alicloud_log_savedsearch" "top_offenders" {
  project_name      = alicloud_log_project.agents.name
  saved_search_name = "top_offending_agents_1h"
  search_query      = "* | SELECT agent, COUNT(*) AS calls, SUM(cost_cny) AS cost FROM log WHERE __time__ > now() - INTERVAL '1' HOUR GROUP BY agent ORDER BY cost DESC LIMIT 20"
  logstore          = alicloud_log_store.this["gateway-requests"].name
  display_name      = "Top offending agents (1h)"
  topic             = "incident-response"
}
```

这样，未来你在 SLS 控制台的搜索栏里就能轻松找到这些查询——凌晨三点被叫起来处理问题的你，一定会感谢现在未雨绸缪的自己。
## 告警疲劳：如何让告警频道保持高价值信息流

文章中提到的四种告警类型方向是对的，但如果不去优化配置，它们很快就会沦为噪音。以下是三条我用来确保钉钉告警频道始终有用的实践经验：

### 规则 1：务必设置限流和去重

每条告警至少需要配置 `throttling = "30m"`。如果不做限流，一个持续一小时的问题可能会触发 12 条消息。工程师不堪其扰，直接把频道静音，最终导致告警失去作用。

对于非简单场景的告警，还需要设置 `notify_threshold`（连续触发次数阈值）：

```hcl
condition         = "p95s > 8"
notify_threshold  = 3   # 连续 3 个分钟窗口触发
throttling        = "15m"
```

通过这种方式，将“单分钟异常 → 立即通知”调整为“持续异常超过五分钟 → 通知”。这样可以避免因某次慢 LLM 调用引发的短暂尖峰打扰团队，只有真正持续性的问题才会触发告警。

### 规则 2：按严重级别分发告警

并不是所有告警都需要发送到同一个频道。我的做法是根据严重性分成三类：

- `#agent-incidents` —— 最高优先级，直接通知相关人员，P0/P1 级别问题，支持电话回拨兜底。
- `#agent-warnings` —— 次优先级，仅发送到钉钉，P2 级别问题，要求在工作时间内确认处理。
- `#agent-info` —— 最低优先级，仅记录日志，不发送通知，用于趋势分析和问题追踪。

每条告警都明确映射到对应的频道：

```hcl
locals {
  alert_routes = {
    "daily-cost-ceiling"  = { severity = 8, channel = "incidents" }
    "token-anomaly"       = { severity = 8, channel = "incidents" }
    "agent-step-latency"  = { severity = 6, channel = "warnings" }
    "error-rate-2pct"     = { severity = 6, channel = "warnings" }
    "drift-detected"      = { severity = 4, channel = "info" }
  }
}

resource "alicloud_log_alert" "this" {
  for_each = local.alert_routes
  notification_list {
    type        = "DingTalk"
    service_uri = lookup({
      "incidents" = var.dingtalk_incidents_webhook
      "warnings"  = var.dingtalk_warnings_webhook
      "info"      = var.dingtalk_info_webhook
    }, each.value.channel)
  }
}
```

这样一来，成本失控的告警会立刻通知我，而延迟 P95 的告警则可以等到周一再查看。

### 规则 3：每条告警都附带 Runbook 链接

每次钉钉告警消息都会包含一个指向内部 Wiki 页面的链接，页面内容包括以下关键信息：
1. 告警的具体含义是什么？
2. 初步排查需要执行的三条诊断查询（通常从上文提到的六条中挑选）。
3. 常见的可能原因有哪些？
4. 如果 30 分钟内无法解决，应该联系谁进行升级处理？

```hcl
content = <<-MSG
  ${each.key} fired: ${"{{ severity }}"}

  Current value: ${"{{ value }}"}
  Threshold: ${each.value.threshold}

  Runbook: https://wiki.internal/agents/runbooks/${each.key}
  Dashboard: https://sls.console.aliyun.com/.../dashboard/agent-cost-overview
MSG
```

Runbook 链接的作用是将“为什么这条告警找我？”转化为“我知道接下来该做什么”。这种设计能够显著缩短平均修复时间（MTTR），大约减少一半。同时，Runbook 应像代码一样维护——当告警逻辑发生变化时，同步更新相关文档。
## SLO：告警之上的更高层次

告警关注的是**当下**，而 SLO（Service Level Objectives，服务级别目标）则关注的是**趋势**。我们需要重点跟踪以下三个指标：

1. **可用性**：在 30 秒内返回非 5xx 错误的 Agent 请求占比。目标是：过去滚动 30 天内达到 99.5%。
2. **延迟**：端到端 p95 的 Agent 响应时间。目标是：控制在 6 秒以内。
3. **单次会话成本**：每次会话的中位成本（单位：¥）。目标是：客服 Agent 控制在 ¥0.30 以内，研究 Agent 控制在 ¥1.50 以内。

每个 SLO 都有一个对应的**预算**（即在未达标之前还能承受的余量）。当月预算剩余超过 70% 时，可以正常发布新功能；当预算降至 30% 时，就需要暂停新功能开发，集中精力提升系统稳定性；如果预算耗尽甚至低于 0%，说明已经未达标——这将成为季度复盘的重点议题。

我们可以通过 SLS 的仪表盘来实现 SLO 的跟踪，具体配置如下：

```hcl
resource "alicloud_log_dashboard" "slo" {
  project_name   = alicloud_log_project.agents.name
  dashboard_name = "agent-slo-tracking"

  char_list = jsonencode([
    {
      title = "Availability budget remaining (rolling 30d)"
      type  = "single"
      query = <<-Q
        * | SELECT (1 - SUM(IF(status >= 500, 1, 0)) * 1.0 / COUNT(*)) - 0.995 AS budget
            FROM log WHERE __time__ > now() - INTERVAL '30' DAY
      Q
      logstore = alicloud_log_store.this["gateway-requests"].name
    },
  ])
}
```

每周团队站会时，务必查看这个仪表盘。这是我见过最有效的工具之一，能够将“我们感觉系统还行”这种模糊的感觉转化为“本月错误预算已消耗 65%——优先修复重试逻辑”的明确行动指引。它帮助我们将运维的主观感受转化为客观数据，真正做到用数据驱动决策。
