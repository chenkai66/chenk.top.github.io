---
title: "Terraform 实战（七）：可观测与成本告警"
date: 2026-03-24 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - SLS
  - ARMS
  - CloudMonitor
  - AI Agents
categories: Terraform
lang: zh
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 7
description: "日志进 SLS、Trace 进 ARMS、指标进 CloudMonitor——全部用 HCL 配，新环境天生带观测。真实救过我项目的四条告警，再加上 SLS 驱动的成本看板，发薪日之前告诉你哪个 Agent 在烧预算。"
disableNunjucks: true
translationKey: "terraform-agents-7"
---
Agent 是非确定性的，多步骤的，还要调用昂贵的 API。这几样加在一起，意味着除非你第一天就把埋点做好，否则出事之后根本没法调试。这篇文章通过 Terraform 打通三条管线——日志、链路追踪、指标——汇聚到一个统一仪表盘，配上六个能解决真实故障的 SLS 查询，再配置四个真会响、且在 production 救过我项目的告警。

走完这套流程，你将拥有一个钉钉群：账单爆炸前会 ping 你，延迟飙升前会 ping 你，错误率异常前会 ping 你，或者某个 Agent 开始死循环前也会 ping 你——再加上能把运维直觉变成运维数据的 SLO 预算。

## 三条管线

![Three signals, three pipelines: logs, traces, metrics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig1_obs_pipeline.png)

三种信号类型，对应三个阿里云服务，最后都汇聚到 SLS 方便人眼查看：

- **Logs** — agent stdout/stderr → ECS 上的 Logtail agent → SLS Logstore
- **Traces** — agent 代码里的 OpenTelemetry SDK → ARMS APM（兼容 OpenTelemetry）
- **Metrics** — CloudMonitor agent 采集的主机指标 + agent 代码上报的自定义指标 → CloudMonitor → 可选 piping 到 SLS

别只选“只要日志”或者“只要指标”。三者你都需要：

- 日志回答"agent 做了什么？”
- 链路追踪回答“时间花哪儿了？”
- 指标回答“这事儿是不是比平时发生得更频繁？”

这里最省钱的错误做法是只靠 `print()` 输出到 stdout 就上线。最昂贵的错误做法是三者都上了，但直到第一次出事故才打开仪表盘——这时候你才发现因为 SDK 没传递上下文，`agent` 字段有时候是 `null`。第一天就把管线接通，然后每周戳一下，确保它真能工作。

## Step 1: SLS project 和 logstores

所有可观测性相关的工作都始于一个 SLS project。每个环境一个是对的；每个 agent 一个就太细了。

```hcl
resource "alicloud_log_project" "agents" {
  name        = "agents-${terraform.workspace}"
  description = "Logs and metrics for agents-${terraform.workspace}"
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

五个 logstores 覆盖实际需求：

- `agent-runs` — 每个 agent 每次运行的每一步（数据流）
- `gateway-requests` — 每次 LLM API 调用一行记录，包含 model、tokens、延迟、成本
- `ecs-syslog` — ECS 实例底层的操作系统日志
- `ack-cluster` — Kubernetes 事件和 pod 日志（仅当使用 ACK 时）
- `audit` — Terraform 做的每一次变更，保留一年用于合规

`audit` 存一年是因为它数据量小，而且几年后当有人问"3 月 12 日谁改了 prod ALB"时你会需要它。`gateway-requests` 的 90 天窗口是我调整最频繁的——短到足以让存储成本控制在 5 GB/天 约¥30/月 以内，又长到足以不做 Hive 任务就能做季度成本趋势分析。

## Step 2: 从 ECS 投递日志

Logtail agent 是阿里云官方的日志采集器。通过 cloud-init 安装（加到第 4 篇文章的 `cloud-init.sh` 里）：

```bash
# Install Logtail
wget http://logtail-release-cn-shanghai.oss-cn-shanghai.aliyuncs.com/linux64/logtail.sh
chmod +x logtail.sh && ./logtail.sh install cn-shanghai
service ilogtaild start

# Tag this machine for the SLS machine group
echo "${sls_user_id}::${sls_machine_group}" > /etc/ilogtail/user_log_config.json
```

Logtail 配置——采集哪些文件、如何解析——是一个 Terraform resource：

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

现在，任何 tagged 机器上匹配 `/var/log/agents/*.log` 的文件都会以 JSON 格式流入 SLS，所有字段都可查询。Agent 代码只需做 `logger.info(json.dumps({...}))`，剩下的自动完成。关键在于对 schema 保持纪律——尽早确定 `ts`, `agent`, `session_id`, `step`, `phase`, `tokens`, `latency_ms`, `cost_cny`, `status`, `error_message`，写好文档，然后拒绝那些 emit 临时字段的 PR。每一个未文档化的字段都是凌晨三点你写不出查询语句的坑。

## Step 3: 通过 OpenTelemetry → ARMS 做链路追踪

对于 traces，ARMS APM 兼容 OpenTelemetry。Terraform 这边很简单——开通一个 ARMS 实例和环境：

```hcl
resource "alicloud_arms_environment" "agents" {
  environment_name      = "agents-${terraform.workspace}"
  bind_resource_id      = module.vpc.vpc_id
  environment_type      = "CS"             # cloud service
  environment_sub_type  = "ECS"
  payment_type          = "POSTPAY"
}
```

Agent 代码使用标准的 OpenTelemetry——不需要阿里云特有的东西：

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
    # ... agent work ...
```

两个环境变量来自 ARMS——`ARMS_OTLP_ENDPOINT` 在 ARMS 控制台，`ARMS_LICENSE_KEY` 来自你的账号。通过 Terraform outputs 把这两个变量 wire 进 cloud-init 模板。

好处在于：在 ARMS 里你能看到“这次 agent 运行花了 12 秒；其中 9 秒是第三次调用 qwen-max 的 LLM 请求”。这种可见性真正会改变你构建 agent 的方式。我第一次看到真实生产会话的 span 树时，我重写了 planner，当用户消息很短时跳过 retrieval——同一周 p95 就省了约 30%。

## Step 4: 用 CloudMonitor 做指标

一旦安装了 cloud-monitor agent——ACK 节点池上的 `install_cloud_monitor` 标志已经做了这件事——CloudMonitor 会自动捕获主机级指标（CPU、内存、网络）。对于 ECS，加到 cloud-init：

```bash
wget http://cms-agent-cn-shanghai.oss-cn-shanghai.aliyuncs.com/release/cms_go_agent_install.sh
chmod +x cms_go_agent_install.sh && ./cms_go_agent_install.sh
```

对于自定义应用指标——比如"research-agent 消耗的 tokens"——把它们作为带结构化字段的 SLS 日志条目 emit 出来，然后通过 SLS 查询告警。SLS-as-metrics 是阿里云主推的模式；CloudMonitor 自定义指标也能用，但从 Terraform 接入更笨重，而且你会被迫把告警拆分到两个控制台。选一个就行。我选 SLS 处理所有事情，除了主机 CPU/内存/磁盘。

## Step 5: 成本仪表盘

精彩的部分来了。每个 LLM 请求都会 hitting gateway，gateway 会向 `gateway-requests` 记录每一行请求，字段如下：

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

SLS 可以在这些数据上跑 SQL。查询“每个 agent 的每日成本”：

```sql
* | SELECT date_trunc('day', __time__) AS day,
          agent,
          SUM(cost_cny) AS daily_cost
  FROM log
  GROUP BY day, agent
  ORDER BY day, daily_cost DESC
```

通过 Terraform 配置仪表盘：

```hcl
resource "alicloud_log_dashboard" "cost" {
  project_name   = alicloud_log_project.agents.name
  dashboard_name = "agent-cost-overview"
  display_name   = "Agent cost overview"

  char_list = jsonencode([
    {
      title = "Daily cost by agent"
      type  = "line"
      query = "* | SELECT date_trunc('day', __time__) AS day, agent, SUM(cost_cny) AS cost FROM log GROUP BY day, agent ORDER BY day"
      logstore = alicloud_log_store.this["gateway-requests"].name
      display = { xAxis = ["day"], yAxis = ["cost"], yKey = "agent" }
    },
    {
      title = "Tokens by model (last 24h)"
      type  = "pie"
      query = "* | SELECT model, SUM(input_tokens + output_tokens) AS tokens FROM log WHERE __time__ > now() - INTERVAL '24' HOUR GROUP BY model"
      logstore = alicloud_log_store.this["gateway-requests"].name
    },
    {
      title = "p95 latency by agent"
      type  = "line"
      query = "* | SELECT date_trunc('hour', __time__) AS hour, agent, approx_percentile(latency_ms, 0.95) AS p95 FROM log GROUP BY hour, agent ORDER BY hour"
      logstore = alicloud_log_store.this["gateway-requests"].name
    }
  ])
}
```

打开 SLS 控制台，你就有了一个实时仪表盘：

![Stacked daily cost by category](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig2_cost_dashboard.png)

这个仪表盘就是“哪个 agent 在烧我的预算？”这个问题的答案——这个问题每个月都会有人问你，通常是个不知道 agent 是什么的人。
## 六个真正派得上用场的 SLS 查询

仪表盘适合回答长期问题，处理故障得靠即席查询。搞了两年 Agent 运维，我个人笔记里 pinned 了大概 12 条 SLS 查询。其中六条值得分享，因为它们在不同技术栈里都通用。

### Query 1: 过去一小时最“肇事”的 Agent

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

这是我的“谁在搞事情”快速看板。成本 spike 告警响了，跑一下这个，两秒钟就能揪出罪魁祸首。

### Query 2: 按状态码追踪错误

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

几秒钟就能分清是"DashScope 在报 500"还是"Agent 发了坏请求”。`arbitrary(error_message)` 随便抓一条样例，省得你再钻进去查。

### Query 3: 单步 Token 分布，揪出死循环

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

步数超过 50 或者 Token 超过 50 万的会话，基本就是死循环。Runaway 告警响了，用这条查具体 session ID。SSH 连上去 `kill -9` 掉肇事者，事后复盘再慢慢聊。

### Query 4: 按阶段拆解延迟

如果你的 Agent 会 emit 每个阶段的结构化 timing（planning, retrieval, LLM call, tool exec, reflection）：

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

这是我的“时间都去哪了”看板。能抓出某个 tool 突然变慢 3 倍（通常上游 API 慢了），或者 retrieval 开始 dominating（向量索引该重建了）。

### Query 5: 单次会话成本排行榜

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

“昂贵会话”看板。单次会话 5 块钱是个有用的阈值——高于这个要么是长对话（正常），要么是 bug 循环（不正常）。每周 inspect 前 10 名，把模式 catch 住，别等变成明天的故障。

### Query 6: 多步 Agent 的流失漏斗

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

对于 `start → plan → tool_call → reflect → answer` 这种流程，能看到多少会话走到了每一步。`tool_call` 处突然 drop-off 说明 tool 对大量用户失败——这和"LLM 坏了”或"planner 太蠢”不一样。

通过 `alicloud_log_savedsearch` 把这些存成 SLS Saved Queries，控制台搜索栏直接能搜到：

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

凌晨 3 点的你自己会感谢现在的你。其他五条也一样配——花一个晚上写 Terraform，第一次被 page 的时候就回本了。

## 步骤 6：四个必配告警

查询回答 *what*，告警告诉你是 *when*。我交付过的多个 Agent 栈里，这四个告警真正派上了用场：

![Four alerts you should provision on day one](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig3_alert_rules.png)

### 告警 1：成本封顶

```hcl
resource "alicloud_log_alert" "cost_ceiling" {
  project_name   = alicloud_log_project.agents.name
  alert_name     = "daily-cost-ceiling"
  alert_displayname = "Daily LLM spend > ¥800"

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
    content        = "Daily LLM cost ¥${"{{today}}"} exceeds ¥800 budget. Check the SLS cost dashboard."
  }

  severity_configurations {
    severity      = 8
    eval_condition = { condition = "today > 800" }
  }
}
```

当天 LLM 花费超过 800 块就触发。阈值按你实际预算调。Throttling 很关键——不然每 5 分钟报一次，一周内团队就把渠道静音了。

### 告警 2：延迟

```hcl
resource "alicloud_log_alert" "latency" {
  project_name   = alicloud_log_project.agents.name
  alert_name     = "agent-step-latency"
  alert_displayname = "p95 agent step > 8s"

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
    content     = "Agent p95 step latency is ${"{{p95s}}"}s — investigate before users notice."
  }
}
```

`notify_threshold = 3` 意味着连续三分钟超标才报——过滤掉偶发的 LLM 慢调用噪音。

### 告警 3：错误率

形状一样，查询是 `SUM(IF(status >= 500, 1, 0)) * 1.0 / COUNT(*) AS err_ratio`，条件 `err_ratio > 0.02`。Throttling 短一点（5 分钟），因为错误通常是持续事件，不是 transient blip。如果 2% 请求是 5xx 且持续五分钟，那是故障，不是 glitch。

### 告警 4：Token 泄漏（死循环）

```hcl
resource "alicloud_log_alert" "token_spike" {
  project_name   = alicloud_log_project.agents.name
  alert_name     = "token-anomaly"
  alert_displayname = "Tokens/min > 2x rolling 24h avg"

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
    content     = "Token consumption ${"{{now_tpm}}"} tpm vs 24h avg ${"{{baseline}}"}. Possible runaway agent."
  }
}
```

这个告警回本了两次。Stop condition 有 bug 的 Agent 一晚上能烧掉 1 万块 Token；这个 alert 2 分钟内就能 catch 住，给你时间 kill 掉 offender。第一次在我生产环境触发时，大概省了 6400 块——一个 malformed JSON tool response 让 planner 永远回到第一步。
## 报警疲劳：我保持频道高信噪比的几条原则

上面提到的四种报警类型没问题，但如果不调优，它们很快就会变成噪音。靠三条原则，我的钉钉频道这几年一直挺管用。

### 原则一：始终做节流和去重

每个报警至少设置 `throttling = "30m"`。不开节流，一个持续 1 小时的问题能刷出 12 条消息。工程师直接把频道静音，报警就废了。

对于重要报警，还要设 `notify_threshold`（发送前连续触发次数）：

```hcl
condition         = "p95s > 8"
notify_threshold  = 3   # 3 consecutive minute-windows
throttling        = "15m"
```

这样就把“某一分钟出错就呼人”变成了“连续五分钟出错才呼人”。单个 LLM 调用慢引起的分钟级抖动不会把人吵醒，持续性问题才会。

### 原则二：按 severity 分级路由

别把所有报警都扔进同一个频道。我这边分了三类：

- `#agent-incidents` — 呼人，P0/P1，包含电话 fallback
- `#agent-warnings` — 仅钉钉，P2，工作时间响应就行
- `#agent-info` — 只追加日志，无通知，用于趋势分析

每个报警显式映射：

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
  # ... map severity into the notification block ...
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

现在，成本失控报警会直接呼我；p95 延迟报警可以等到周一再看。按 severity 配不同 webhook 是最省钱的方案——比专门写个路由服务便宜多了。

### 原则三：每个报警都得有 Runbook 链接

钉钉消息里带上 Wiki 页面链接，文档包含：

1. 这个报警意味着什么
2. 前三条诊断查询该怎么跑（从上面六条里选）
3. 常见原因
4. 如果 30 分钟内搞不定，该联系谁升级

```hcl
content = <<-MSG
  ${each.key} fired: ${"{{ severity }}"}

  Current value: ${"{{ value }}"}
  Threshold: ${each.value.threshold}

  Runbook: https://wiki.internal/agents/runbooks/${each.key}
  Dashboard: https://sls.console.aliyun.com/.../dashboard/agent-cost-overview
MSG
```

有了 Runbook 链接，“为啥呼我”就变成了“我知道该干嘛”。MTTR 大概能砍掉一半。像管理代码一样管理 Runbook——报警变了就 review，每季度清理过期的。

## 把 SLO 作为报警之上的元层

报警抓的是*当下*，SLO（Service Level Objectives）抓的是*趋势*。任何生产环境的 agent 我都盯这三个指标：

1. **Availability**：30 秒内返回非 5xx 的 agent 请求百分比。目标：30 天滚动 99.5%。
2. **Latency**：p95 端到端 agent 响应时间。目标：6 秒以内。
3. **Cost-per-session**：会话中位数成本。目标：支持 agent 低于 ¥0.30，研究 agent 低于 ¥1.50。

每个 SLO 都有个*预算*——也就是出错容忍度。当月预算还剩 70% 以上，随便发版；降到 30%，停止新功能，专注稳定性；低于 0%，说明 SLO  breached——这是季度复盘的事，不是“明天修好”就行。

把 SLO 追踪配成另一个 SLS Dashboard：

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

团队周会上每周看这个看板。这是我知道最好的工具，能把“大概还行”变成“本月错误预算用了 65%——优先修重试逻辑”。把运维直觉变成运维数据，这就是工程团队和救火团队的区别。

## 路由选择：钉钉和 ARMS 侧报警

在国内，钉钉是大多数工程团队的默认聊天工具，SLS 也原生支持钉钉 webhook。你也可以分发到邮件、短信，或者（通过 webhook） Slack/Teams/Lark。选那个凌晨 2 点团队还会看的渠道——这是唯一标准。

ARMS 自带报警也有用，适合 trace 级别的条件（比如“任何超过 30 个 span 的 trace"或"`llm.model = qwen-max` 且 duration > 5s 的 span"）。对于上面那四个报警，SLS 侧就够了，避免把报警逻辑拆到两个控制台。只有 SLS 表达不了需求时才用 ARMS 报警——通常是指那些没法简化为扁平日志查询的 span-tree 形状条件。

## 成本多少

可观测性是有真金白银成本的——通常占其余账单的 10-15%：

- **SLS**：摄入 ~¥0.35/GB + 存储 ¥0.15/GB。中等流量的 agent 栈摄入 ~5 GB/天 → 摄入 ¥50/月，30 天保留 ¥20/月
- **ARMS APM**：~¥600/月，含 1 个环境，最多 1 亿 spans
- **CloudMonitor**：标准指标免费，自定义指标 ¥0.005/天/个

真正的生产 agent 栈，全量可观测性预算每月 ¥1000-1500。比起漏掉一次成本失控报警，这很便宜——光 token 泄漏报警这一项，在我的项目里就省回了两年 SLS 费用。

## 接下来

第 8 篇是端到端 walkthrough。我们把第 2 到 7 篇的所有模块——vpc-baseline、compute、storage、gateway、observability——组合成一个 `research-agent-stack` 项目，单次 `terraform apply` 看它跑起来。真实的 apply 输出，真实的耗时，完整的 module DAG。结尾的 starter repo 随便你 fork。