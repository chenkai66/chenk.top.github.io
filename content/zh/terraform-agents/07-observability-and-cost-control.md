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

Agent 是非确定的、多步的、调昂贵 API 的。这组合意味着：如果你不在第一天给它 instrument，事后就别想 debug。本篇用 Terraform 打通三条管道——日志、Trace、指标——汇成一个统一看板，再叠六条真实救过场的 SLS 查询和四条真正在生产环境救过我项目的告警。

读完之后你拥有一个钉钉群：账单爆掉之前、延迟挂掉之前、错误率飙升之前、某个 Agent 自循环之前，它都会先 ping 你。最后再用 SLO 把"感觉"变成"数据"。

## 三条核心数据流

![三种信号、三条管道：日志、链路追踪、指标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig1_obs_pipeline.png)

三种可观测信号，对应三个阿里云服务，最终都汇到日志服务（SLS）以人类可读的方式查看：

- 日志 —— Agent 的 stdout/stderr → ECS 上的 Logtail 采集器 → SLS Logstore
- 链路追踪 —— Agent 代码里的 OpenTelemetry SDK → 应用实时监控服务 ARMS APM（兼容 OpenTelemetry）
- 指标 —— 云监控 CloudMonitor 的主机指标 + Agent 代码里的自定义业务指标 → CloudMonitor → 可选转发到 SLS

不要只挑"日志"或者"指标"，三者缺一不可：

- 日志回答："Agent 到底干了什么？"
- Trace 回答："时间花在了哪一步？"
- 指标回答："这事儿是不是比平时频繁？"

最便宜的错误是只 `print()` 到 stdout 上线。最贵的错误是三条管道都通了，但直到第一次事故才打开看板——结果发现 `agent` 字段时不时是 `null`，因为 SDK 没把上下文传下去。第一天就接好，每周戳一下，确保它真的在工作。

## 第 1 步：SLS 项目和 Logstore

所有可观测的东西都从一个 SLS 项目开始。一个环境一个项目是对的；一个 Agent 一个就太碎了。

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
    "agent-runs"        = { ttl = 30,  shard_count = 4 }
    "gateway-requests"  = { ttl = 90,  shard_count = 4 }
    "ecs-syslog"        = { ttl = 14,  shard_count = 2 }
    "ack-cluster"       = { ttl = 30,  shard_count = 4 }
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

- `agent-runs` —— 每个 Agent 每一步（消防水管）
- `gateway-requests` —— 每次 LLM API 调用一行，带 model、tokens、latency、cost
- `ecs-syslog` —— ECS 实例底层 OS 日志
- `ack-cluster` —— Kubernetes 事件和 Pod 日志（仅用 ACK 时）
- `audit` —— Terraform 的每次变更，留 1 年合规

`audit` 留一年是因为它体积小，几年后"3 月 12 日是谁改了 prod ALB"会问到。`gateway-requests` 的 90 天是我调得最多的一个——5 GB/天的量短到能把存储压在 ¥30/月以内，长到能做季度环比成本分析而不用上 Hive 任务。

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

现在任何打了标的机器上 `/var/log/agents/*.log` 都会流进 SLS，作为 JSON 按字段可查。Agent 代码只要 `logger.info(json.dumps({...}))`，其余自动。关键是字段 schema 要早早定下来——`ts`、`agent`、`session_id`、`step`、`phase`、`tokens`、`latency_ms`、`cost_cny`、`status`、`error_message` 这十个字段固定，写进规范，PR 评审时拒绝乱加字段。每一个没规范的字段，都是凌晨三点你写不出的查询。

## 第 3 步：通过 OpenTelemetry 把 Trace 接入 ARMS

链路追踪这边，ARMS APM 兼容 OpenTelemetry。Terraform 这边很简单，只需创建 ARMS 实例和环境：

```hcl
resource "alicloud_arms_environment" "agents" {
  environment_name      = "agents-${terraform.workspace}"
  bind_resource_id      = module.vpc.vpc_id
  environment_type      = "CS"             # cloud service
  environment_sub_type  = "ECS"
  payment_type          = "POSTPAY"
}
```

Agent 代码用标准 OpenTelemetry，完全不依赖阿里云特定 SDK：

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
    # ... Agent 业务逻辑 ...
```

两个环境变量都来自 ARMS：`ARMS_OTLP_ENDPOINT` 在 ARMS 控制台找，`ARMS_LICENSE_KEY` 在账号信息里。两者都用 Terraform output 接进 cloud-init 模板。

回报是：在 ARMS 里你能看到"这次 Agent 跑了 12 秒，其中 9 秒是第三次调 qwen-max"。这种粒度的可观测性会真正改变你设计 Agent 的方式。我第一次在生产 session 上看到完整 span 树的那一周，就改了 planner——用户输入很短时跳过 retrieval，直接把 p95 砍了 30%。

## 第 4 步：通过 CloudMonitor 收集指标

装好 cloud-monitor agent 后，CloudMonitor 自动采集主机级别指标（CPU、内存、网络）。ACK 节点池的 `install_cloud_monitor` 参数已经替你做了；ECS 实例则在 cloud-init 里加：

```bash
wget http://cms-agent-cn-shanghai.oss-cn-shanghai.aliyuncs.com/release/cms_go_agent_install.sh
chmod +x cms_go_agent_install.sh && ./cms_go_agent_install.sh
```

应用级自定义指标——比如"research-agent 消耗的 token"——以结构化字段的形式写进 SLS 日志，再用 SLS 查询出告警就行。SLS-as-metrics 是阿里云推荐的模式；CloudMonitor 自定义指标也能做，但从 Terraform 配置角度看更繁琐，而且告警会被分散到两个控制台。挑一个。我除了主机 CPU/内存/磁盘以外，全选 SLS。

## 第 5 步：成本看板

每次 LLM 请求都打到网关，网关在 `gateway-requests` 写一行日志，字段如下：

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

SLS 直接对这些跑 SQL。"按 Agent 看每日成本"的查询：

```sql
* | SELECT date_trunc('day', __time__) AS day,
          agent,
          SUM(cost_cny) AS daily_cost
  FROM log
  GROUP BY day, agent
  ORDER BY day, daily_cost DESC
```

用 Terraform 配看板：

```hcl
resource "alicloud_log_dashboard" "cost" {
  project_name   = alicloud_log_project.agents.name
  dashboard_name = "agent-cost-overview"
  display_name   = "Agent 成本概览"

  char_list = jsonencode([
    {
      title = "按 Agent 看每日成本"
      type  = "line"
      query = "* | SELECT date_trunc('day', __time__) AS day, agent, SUM(cost_cny) AS cost FROM log GROUP BY day, agent ORDER BY day"
      logstore = alicloud_log_store.this["gateway-requests"].name
      display = { xAxis = ["day"], yAxis = ["cost"], yKey = "agent" }
    },
    {
      title = "近 24 小时按模型看 Token 用量"
      type  = "pie"
      query = "* | SELECT model, SUM(input_tokens + output_tokens) AS tokens FROM log WHERE __time__ > now() - INTERVAL '24' HOUR GROUP BY model"
      logstore = alicloud_log_store.this["gateway-requests"].name
    },
    {
      title = "按 Agent 看 P95 延迟"
      type  = "line"
      query = "* | SELECT date_trunc('hour', __time__) AS hour, agent, approx_percentile(latency_ms, 0.95) AS p95 FROM log GROUP BY hour, agent ORDER BY hour"
      logstore = alicloud_log_store.this["gateway-requests"].name
    }
  ])
}
```

打开 SLS 控制台就有一个实时看板：

![按类别堆叠的每日成本](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig2_cost_dashboard.png)

这个看板回答的是每个月都有人问的那个问题——"到底哪个 Agent 在烧我的预算？"——通常问的人还不知道 Agent 是什么。

## 救场用的六条 SLS 查询

看板回答长期问题。事故响应需要临时查询。两年 Agent 运维下来，我个人笔记里收藏了大约 12 条 SLS 查询。其中 6 条值得分享，因为它们跨 stack 通用。

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

我用来快速定位"谁最耗资源"的面板。成本飙升告警一响，跑这条，2 秒内就揪出罪魁祸首。

### 查询 2：按状态码看错误分布

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

几秒内就能区分"DashScope 在抛 500"还是"Agent 发了非法请求"。`arbitrary(error_message)` 随机抽一条出来，省得再往里钻。

### 查询 3：按步骤看 Token 分布，找失控循环

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

超过 50 步或 50 万 token 的 session 基本都是循环。失控告警响了，用这条锁定具体 session ID。SSH 上机器 `kill -9` 干掉，然后慢慢复盘。

### 查询 4：按阶段拆解延迟

如果 Agent 按阶段（planning、retrieval、LLM call、tool exec、reflection）记录了结构化耗时：

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

我的"时间都去哪儿了"看板。能抓到某个工具突然慢了 3 倍（通常是上游 API 变慢）、或者 retrieval 开始拖后腿（向量索引该重建了）。

### 查询 5：单 session 成本排行榜

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

"昂贵 session"排行榜。¥5 是个实用阈值——超过这个的 session 要么是合理长对话，要么就是 bug 循环。每周看 Top 10，提前发现模式，免得变成明天的事故。

### 查询 6：多阶段 Agent 的漏斗分析

对带 `start → plan → tool_call → reflect → answer` 阶段的 Agent，看每个阶段的 session 留存：

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

如果 `tool_call` 那一阶段突然大量流失，说明工具对很多用户挂了。这和"LLM 挂了"或"planner 太笨"是完全不同的问题，需要单独处理。

把这些存成 SLS Saved Query，用 `alicloud_log_savedsearch` 资源定义，控制台搜索栏就能找到：

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

凌晨三点被叫起来的你会感谢现在的自己。其余五条照样配——一个晚上的 Terraform 工作，第一次被 oncall 时就回本。

## 第 6 步：四条告警

查询回答"是什么"，告警回答"什么时候"。四条告警在我交付过的多个 Agent stack 里挣得了它们的位置：

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

当日 LLM 花费过 ¥800 的话每 30 分钟响一次。按你的真实预算调阈值。throttling 重要——不设的话每 5 分钟响一次，团队一周内就会把群消息免打扰。

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

`notify_threshold = 3` 意思是连续三分钟超阈值才响——把单次慢 LLM 调用的噪声压住。

### 告警 3：错误率

形状一样，query 是 `SUM(IF(status >= 500, 1, 0)) * 1.0 / COUNT(*) AS err_ratio`，condition `err_ratio > 0.02`。throttling 短一些（5 分钟），因为错误通常是真正持续的事件，不是瞬时抖动。如果 2% 的请求是 5xx 并且持续了 5 分钟，那就是事故，不是 glitch。

### 告警 4：Token 漏（失控循环）

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

这是值回票价的一条，已经救过我两次。一个停止条件有 bug 的 Agent 一夜之间能烧 ¥10,000 token；这条告警 2 分钟内就抓到，给你时间杀进程。第一次它在我生产 stack 上响，直接挽回了大约 ¥6,400——一个工具返回了 malformed JSON，把 planner 一直送回第 1 步。

## 告警疲劳：让告警群保持高信噪比的三条规则

上面四条告警的"类型"是对的。如果不调，它们很快就变噪音。三条规则让我的钉钉告警群多年保持有用。

### 规则 1：throttle + 去重，永远

每条告警最少 `throttling = "30m"`。不限流的话，一个持续 1 小时的问题会产生 12 条消息。工程师把群免打扰，告警就废了。

非平凡告警还要设 `notify_threshold`（连续触发次数阈值）：

```hcl
condition         = "p95s > 8"
notify_threshold  = 3   # 连续 3 个分钟窗口
throttling        = "15m"
```

这把"一分钟异常 → 立即通知"调成"持续五分钟 → 通知"。单次慢 LLM 调用引起的瞬时尖峰不打扰人；持续问题才打扰。

### 规则 2：按严重等级分发

不是所有告警都进同一个群。我的配法是三档：

- `#agent-incidents` —— 寻人（含电话兜底），P0/P1
- `#agent-warnings` —— 仅钉钉，P2，工作时间内确认
- `#agent-info` —— 只记录、不通知，做趋势分析

每条告警显式映射：

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

这样成本失控告警会立刻找我；延迟 P95 告警可以等到周一再看。不同 webhook 是把这事做对的最便宜方式——比自己写一个路由服务便宜得多。

### 规则 3：每条告警都带 Runbook 链接

钉钉消息里塞一个内部 Wiki 链接，页面写清楚四件事：

1. 这条告警是什么意思
2. 头三条诊断查询（从上面六条里挑）
3. 常见原因
4. 30 分钟内搞不定该找谁升级

```hcl
content = <<-MSG
  ${each.key} fired: ${"{{ severity }}"}

  Current value: ${"{{ value }}"}
  Threshold: ${each.value.threshold}

  Runbook: https://wiki.internal/agents/runbooks/${each.key}
  Dashboard: https://sls.console.aliyun.com/.../dashboard/agent-cost-overview
MSG
```

Runbook 链接把"为啥找我"变成"我知道该干什么"。MTTR 大约能砍一半。Runbook 像代码一样维护——告警逻辑变了同步改，每季度清理一次过时的。

## SLO：告警之上的元层

告警关注当下，SLO（Service Level Objectives）关注趋势。任何上线的 Agent 我都跟三条 SLO：

1. 可用性：30 秒内返回非 5xx 的 Agent 请求占比。目标：滚动 30 天 99.5%。
2. 延迟：端到端 p95 响应时间。目标：6 秒以内。
3. 单 session 成本：每次会话的中位 ¥。目标：客服 Agent ¥0.30 以内，研究 Agent ¥1.50 以内。

每个 SLO 有一个预算——还没"破"之前能消耗的余量。月预算剩 70% 以上时，新功能随便发；剩 30% 时停发新 Agent 功能、专心做稳定性；低于 0% 就是已经破 SLO，进季度复盘——不是"明天修一下"。

把 SLO 也用 SLS 看板做出来：

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
    # ... latency budget, cost budget ...
  ])
}
```

每周团队站会必看这个看板。这是我见过最好用的工具，能把"我们感觉系统还行"变成"本月错误预算消耗 65%——优先修重试逻辑"。把运维直觉变成运维数据，这就是工程团队和救火团队的区别。

## 路由选择：钉钉与 ARMS-side 告警

国内大部分工程团队默认钉钉，SLS 原生支持钉钉 webhook。也能扇出到邮件、短信、（通过 webhook）Slack/Teams/飞书。挑你团队凌晨两点会看的那个——这是唯一标准。

ARMS 自带告警，适合 trace 级别的条件（"任何包含 > 30 个 span 的 trace"或"`llm.model = qwen-max` 且耗时 > 5s 的 span"）。但上面四条告警 SLS 这边就够，避免把告警逻辑分散到两个控制台。只有 SLS 表达不了的需求才用 ARMS 告警——通常是没法压成扁平日志查询的 span 树形态条件。

## 成本核算

可观测有真实成本，通常占整体账单 10-15%：

- SLS：摄入 ~¥0.35/GB + 存储 ¥0.15/GB。中等流量 Agent stack 每天摄入 ~5 GB → 每月摄入 ¥50，30 天保留存储 ¥20
- ARMS APM：~¥600/月，单环境最多 1 亿 span
- CloudMonitor：标准指标免费，自定义指标 ¥0.005/指标/天

预算 ¥1000-1500/月做生产 Agent stack 的全套可观测性。比错过一次成本失控告警便宜得多——单是 token 漏告警在我项目里就抵了两年的 SLS 钱。

## 接下来

第八篇是端到端走查。我们把第 2-7 篇的所有模块——vpc-baseline、compute、storage、gateway、observability——组装成一个 `research-agent-stack` 项目，看它通过一次 `terraform apply` 立起来。真实的 apply 输出、真实的耗时、完整的模块 DAG。最后给一个可以直接 fork 的初始仓库。
