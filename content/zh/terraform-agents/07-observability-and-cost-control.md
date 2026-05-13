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
Agent 具备非确定性、多步骤执行特性，并频繁调用高成本 API。这三者叠加意味着：若未在上线首日完成可观测性埋点，故障发生后将极难定位与调试。本文通过 Terraform 打通日志、链路追踪和指标三条管线，全部汇聚至统一仪表盘，并配套六个可直接用于排查真实故障的 SLS 查询，以及四个已在生产环境中成功拦截事故的钉钉告警。

配置完成后，你将拥有一个高信噪比的钉钉告警群——当账单异常激增、延迟持续飙升、错误率显著偏离基线，或 Agent 出现死循环迹象时，系统会即时通知；搭配 SLO 预算看板，还能将模糊的运维直觉转化为可度量的数据。

## 三条管线

![三个信号，三个管道：日志、跟踪、指标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig1_obs_pipeline.png)

三种信号类型对应三个阿里云服务，最终都汇聚到 SLS，便于人类友好地查看：

- **Logs** — Agent 的 stdout/stderr → ECS 上的 Logtail agent → SLS Logstore
- **Traces** — Agent 代码中的 OpenTelemetry SDK → ARMS APM（兼容 OpenTelemetry）
- **Metrics** — CloudMonitor agent 采集的主机指标 + Agent 代码上报的自定义指标 → CloudMonitor → 可选地导入 SLS

不要只接日志或只接指标——三者缺一不可：

- 日志回答“Agent 做了什么？”
- 链路追踪回答“时间花哪儿了？”
- 指标回答“这事是不是比平时更频繁？”

最省钱的错误做法是仅靠 `print()` 输出到 stdout 就上线；而最昂贵的错误则是三者都上了，却直到第一次事故才打开仪表盘——结果发现因 SDK 未正确传递上下文，`agent` 字段有时为 `null`。务必在第一天就打通全部管线，并坚持每周验证其可用性。

## Step 1: SLS Project 和 Logstores

所有可观测性工作都始于一个 SLS Project。每个环境配一个是对的，每个 Agent 配一个则过于细化。

![带有实时指标仪表板的云监控指挥中心](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/wanxiang_monitoring.png)


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

五个 Logstore 足以覆盖实际需求：

- `agent-runs` — 记录每个 Agent 每次运行的每一步（数据洪流）
- `gateway-requests` — 每次 LLM API 调用一行记录，包含 model、tokens、延迟、成本等字段
- `ecs-syslog` — ECS 实例底层的操作系统日志
- `ack-cluster` — Kubernetes 事件和 Pod 日志（仅在使用 ACK 时启用）
- `audit` — Terraform 所有变更记录，保留一年以满足合规要求

`audit` 存储一年是因为数据量小，且几年后当有人问“3 月 12 日谁改了 prod ALB”时你会急需它。`gateway-requests` 的 90 天保留窗口是我调整最频繁的——短到足以将存储成本控制在每天 5 GB、每月约 ¥30 以内，又长到无需 Hive 作业即可完成季度成本趋势分析。

## Step 2: 从 ECS 投递日志

Logtail Agent 是阿里云官方的日志采集器，可通过 cloud-init 安装（添加到第 4 篇文章的 `cloud-init.sh` 中）：

```bash
# Install Logtail
wget http://logtail-release-cn-shanghai.oss-cn-shanghai.aliyuncs.com/linux64/logtail.sh
chmod +x logtail.sh && ./logtail.sh install cn-shanghai
service ilogtaild start

# Tag this machine for the SLS machine group
echo "${sls_user_id}::${sls_machine_group}" > /etc/ilogtail/user_log_config.json
```

Logtail 的配置——包括采集哪些文件及如何解析——是一个 Terraform 资源：

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

现在，任何带标签的机器上匹配 `/var/log/agents/*.log` 的文件都会以 JSON 格式自动流入 SLS，所有字段均可查询。Agent 代码只需执行 `logger.info(json.dumps({...}))`，其余流程全自动。关键在于对 schema 保持纪律：尽早确定字段如 `ts`、`agent`、`session_id`、`step`、`phase`、`tokens`、`latency_ms`、`cost_cny`、`status`、`error_message`，写好文档，并拒绝那些随意新增字段的 PR。每个未文档化的字段，都可能在凌晨三点阻碍你写出有效的诊断查询。

## Step 3: 通过 OpenTelemetry → ARMS 做链路追踪

ARMS APM 兼容 OpenTelemetry，Terraform 配置非常简洁——只需开通一个 ARMS 实例和环境：

```hcl
resource "alicloud_arms_environment" "agents" {
  environment_name      = "agents-${terraform.workspace}"
  bind_resource_id      = module.vpc.vpc_id
  environment_type      = "CS"             # cloud service
  environment_sub_type  = "ECS"
  payment_type          = "POSTPAY"
}
```

Agent 代码使用标准 OpenTelemetry，无需任何阿里云特有逻辑：

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

两个环境变量来自 ARMS：`ARMS_OTLP_ENDPOINT` 在 ARMS 控制台获取，`ARMS_LICENSE_KEY` 来自账号信息。通过 Terraform outputs 将二者注入 cloud-init 模板。

回报立竿见影：在 ARMS 中你能看到“本次 Agent 运行耗时 12 秒，其中 9 秒花在第三次调用 qwen-max 上”。这种可见性真正改变了我构建 Agent 的方式。首次在真实生产会话中看到 span 树后，我立即重构了 planner——对简短用户消息跳过检索环节，当周 p95 延迟因此下降约 30%。

## Step 4: 用 CloudMonitor 做指标

CloudMonitor 在安装 cloud-monitor agent 后会自动捕获主机级指标（CPU、内存、网络）。ACK 节点池已通过 `install_cloud_monitor` 标志完成此操作；对于 ECS，则需加入 cloud-init：

```bash
wget http://cms-agent-cn-shanghai.oss-cn-shanghai.aliyuncs.com/release/cms_go_agent_install.sh
chmod +x cms_go_agent_install.sh && ./cms_go_agent_install.sh
```

对于自定义应用指标（如“research-agent 消耗的 tokens”），建议将其作为结构化字段写入 SLS 日志，再通过 SLS 查询触发告警。这是阿里云主推的“SLS-as-metrics”模式。虽然 CloudMonitor 也支持自定义指标，但从 Terraform 接入更笨重，且会导致告警分散在两个控制台。建议统一选择其一——我除主机 CPU/内存/磁盘外，其余指标均走 SLS。

## Step 5: 成本仪表盘

精彩部分来了：每个 LLM 请求都会经过 Gateway，而 Gateway 会向 `gateway-requests` 写入一行记录，包含如下字段：

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

SLS 支持对这些数据执行 SQL 查询，例如“各 Agent 的每日成本”：

```sql
* | SELECT date_trunc('day', __time__) AS day,
          agent,
          SUM(cost_cny) AS daily_cost
  FROM log
  GROUP BY day, agent
  ORDER BY day, daily_cost DESC
```

通过 Terraform 配置该仪表盘：

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

打开 SLS 控制台，即可看到实时仪表盘：

![按类别堆叠的每日成本](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig2_cost_dashboard.png)

这个仪表盘直接回答“哪个 Agent 在烧我的预算？”——这个问题每月都会被提出，提问者通常是不熟悉 Agent 架构的相关方（如财务或业务团队）。

## 六个真正派得上用场的 SLS 查询

仪表盘适合回答长期趋势问题，而事故排查依赖即席查询。经过两年 Agent 生产运维实践，我在个人笔记中沉淀了约 12 条高频 SLS 查询。以下六条因其跨栈通用性值得分享。

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

这是我的“谁在搞事情”快速看板。成本突增告警触发后，运行此查询，两秒内通常就能锁定罪魁祸首。

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

几秒内即可区分是“DashScope 返回 500”还是“Agent 发送了无效请求”。`arbitrary(error_message)` 随机提取一条错误示例，省去钻取日志的麻烦。

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

步数超过 50 或 Token 超过 50 万的会话，基本就是死循环。Runaway 告警触发后，用此查询定位具体 session ID，SSH 登录后 `kill -9` 终止进程，再从容复盘。

### Query 4: 按阶段拆解延迟

如果你的 Agent 会记录各阶段的结构化耗时（planning、retrieval、LLM call、tool exec、reflection）：

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

这是我的“时间都去哪了”看板。能快速发现某个工具突然变慢三倍（通常因上游 API 性能下降），或检索阶段开始主导耗时（向量索引需重建）。

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

“昂贵会话”看板。单次会话 ¥5 是一个实用阈值——高于此值要么是长对话（合理），要么是 Bug 循环（异常）。建议每周检查 Top 10，提前捕捉模式，避免演变为明日事故。

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

对于 `start → plan → tool_call → reflect → answer` 这类流程，该查询展示各阶段的会话留存率。若 `tool_call` 阶段突然大量流失，说明工具对多数用户失败——这与“LLM 故障”或“planner 低效”性质不同。

通过 `alicloud_log_savedsearch` 将这些查询保存为 SLS Saved Queries，即可在控制台搜索栏直接调用：

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

凌晨三点被叫醒的你，定会感谢此刻认真配置的自己。其余五条同理：仅需一个晚上的 Terraform 编写，就能在首次生产告警时收回投入。

## Step 6: 四个必配告警

查询回答 *what*，告警揭示 *when*。在我交付的多个 Agent 栈中，以下四个告警真正发挥了作用：

![第一天应配置的四个警报](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/07-observability-and-cost-control/fig3_alert_rules.png)

### 告警 1: 成本封顶

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

若当日 LLM 花费已超 ¥800，则每 30 分钟触发一次。阈值应根据实际预算调整。节流（throttling）至关重要——否则告警每 5 分钟刷屏一次，团队一周内就会静音频道。

### 告警 2: 延迟

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

`notify_threshold = 3` 表示连续三分钟超标才告警，有效过滤偶发的 LLM 慢调用噪音。

### 告警 3: 错误率

查询逻辑为 `SUM(IF(status >= 500, 1, 0)) * 1.0 / COUNT(*) AS err_ratio`，条件为 `err_ratio > 0.02`。节流设为 5 分钟，因为错误通常是持续事件而非瞬时抖动。若 2% 请求返回 5xx 且持续五分钟，那就是事故，不是小故障。

### 告警 4: Token 泄漏（死循环）

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

该告警已两次回本。存在缺陷的停止条件可能导致 Agent 一夜烧掉 ¥10,000；此告警能在 2 分钟内捕获异常，为你争取终止进程的时间。首次在我生产环境触发时，大约节省了 ¥6,400——根源是一个格式错误的 JSON 工具响应，导致 planner 无限循环回到第一步。

## 告警疲劳：保持频道高信噪比的三条原则

上述四类告警类型合理，但若不调优，很快会沦为噪音。依靠以下三条原则，我的钉钉频道多年来始终保持高价值。

### 原则一：始终节流与去重

每个告警至少设置 `throttling = "30m"`。否则，一个持续 1 小时的问题会生成 12 条消息，工程师很快会静音频道，告警机制随之失效。

对于关键告警，还需设置 `notify_threshold`（连续触发次数）：

```hcl
condition         = "p95s > 8"
notify_threshold  = 3   # 3 consecutive minute-windows
throttling        = "15m"
```

这将“单分钟异常即告警”转变为“连续五分钟异常才通知”，避免因单次 LLM 调用延迟而误报，确保只有持续性问题才会打扰团队。

### 原则二：按严重性分级路由

不要将所有告警塞进同一频道。我的设置分为三级：

- `#agent-incidents` — 触发人工响应，P0/P1 级别，含电话兜底
- `#agent-warnings` — 仅钉钉通知，P2 级别，工作时间处理即可
- `#agent-info` — 仅追加日志，无通知，用于趋势分析

每个告警显式映射到对应频道：

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

如今，成本失控告警会直接呼叫我，而 p95 延迟告警可等到周一处理。为不同严重性配置独立 Webhook 是最经济的方案——远比自建路由服务划算。

### 原则三：每个告警必须附带 Runbook 链接

钉钉消息中需包含 Wiki 页面链接，文档应涵盖：

1. 告警含义
2. 前三条诊断查询（从上述六条中选取）
3. 常见原因
4. 若 30 分钟内无法解决，应联系谁升级

```hcl
content = <<-MSG
  ${each.key} fired: ${"{{ severity }}"}

  Current value: ${"{{ value }}"}
  Threshold: ${each.value.threshold}

  Runbook: https://wiki.internal/agents/runbooks/${each.key}
  Dashboard: https://sls.console.aliyun.com/.../dashboard/agent-cost-overview
MSG
```

Runbook 链接将“为何呼我”转化为“我知道该做什么”，平均修复时间（MTTR）可缩短约一半。应像管理代码一样维护 Runbook：告警规则变更时同步 Review，每季度清理过期内容。

## 把 SLO 作为告警之上的元层

告警捕捉 *当下*，SLO（Service Level Objectives）则监控 *趋势*。我对任何生产 Agent 都跟踪以下三项：

1. **Availability**：30 秒内返回非 5xx 的请求占比，目标为滚动 30 天达 99.5%
2. **Latency**：端到端 p95 响应时间，目标低于 6 秒
3. **Cost-per-session**：会话成本中位数，支持型 Agent 目标低于 ¥0.30，研究型低于 ¥1.50

每个 SLO 都有 *预算*——即容错空间。当月预算剩余 70% 以上时，可自由发布新功能；降至 30% 时，应暂停新功能，专注稳定性；若跌破 0%，即 SLO 违约——这属于季度复盘事项，而非“明日修复”范畴。

将 SLO 追踪配置为另一个 SLS Dashboard：

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

团队周会应定期查看此看板。这是我所知最佳工具，能将“大概还行”转化为“本月错误预算已用 65%——优先修复重试逻辑”。将运维直觉转化为运维数据，正是工程团队与救火团队的本质区别。

## 路由选择：钉钉与 ARMS 侧告警

在中国，钉钉是多数工程团队的默认沟通工具，SLS 也原生支持钉钉 Webhook。你也可分发至邮件、短信，或通过 Webhook 接入 Slack/Teams/Lark。唯一标准是：选择团队凌晨 2 点仍会查看的渠道。

ARMS 自带告警适用于 Trace 级条件（如“任一 Trace 超过 30 个 Span”或“`llm.model = qwen-max` 且耗时 > 5s 的 Span”）。但对于前述四类告警，SLS 侧已足够，避免将告警逻辑分散到两个控制台。仅当 SLS 无法表达需求时（通常涉及 Span 树形结构条件）才使用 ARMS 告警。

## 成本多少

可观测性确有成本——通常占其余账单的 10–15%：

- **SLS**：摄入约 ¥0.35/GB + 存储 ¥0.15/GB。中等流量 Agent 栈日均摄入 5 GB → 摄入 ¥50/月，30 天存储 ¥20/月
- **ARMS APM**：约 ¥600/月（含 1 个环境，最多 1 亿 Spans）
- **CloudMonitor**：标准指标免费，自定义指标 ¥0.005/天/个

真实生产 Agent 栈的全量可观测性预算约为 ¥1000–1500/月。相比漏掉一次成本失控事故，这非常划算——仅 Token 泄漏告警一项，在我的项目中就已省回两年 SLS 费用。

## 接下来

第 8 篇是端到端实战。我们将第 2 至 7 篇的所有模块——vpc-baseline、compute、storage、gateway、observability——组合成一个 `research-agent-stack` 项目，通过单次 `terraform apply` 一键拉起。包含真实 apply 输出、实际耗时及完整模块 DAG。文末的 starter repo 可供你自由 fork。
