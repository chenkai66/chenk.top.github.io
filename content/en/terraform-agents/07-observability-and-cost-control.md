---
title: "Terraform for AI Agents (7): Observability, SLS Dashboards, and Cost Alarms"
date: 2026-03-24 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - SLS
  - ARMS
  - CloudMonitor
  - AI Agents
categories: Terraform
lang: en
mathjax: false
series: terraform-agents
series_title: "Terraform for AI Agents on Alibaba Cloud"
series_order: 7
description: "Logs to SLS, traces to ARMS, metrics to CloudMonitor — all provisioned in HCL so a new env comes pre-instrumented. The four alarms that actually catch real incidents and the SLS-driven cost dashboard that tells you which agent is burning your budget before payday."
disableNunjucks: true
translationKey: "terraform-agents-7"
---

Agents are non-deterministic, multi-step, and call expensive APIs. This combination means you can't debug them after the fact unless you instrumented them from the start. This article sets up three pipelines through Terraform — logs, traces, and metrics — into a unified dashboard, adds six SLS queries to solve real incidents, and sets up four alarms that have actually fired and saved my projects in production.

By the end, you'll have a DingTalk channel that alerts you before the bill explodes, latency increases, the error rate spikes, or an agent starts looping on itself — plus SLO budgets that turn operational feelings into data.

---

## The three pipelines

![Three signals, three pipelines: logs, traces, metrics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/07-observability-and-cost-control/fig1_obs_pipeline.png)

Three signal types, three Aliyun services, all converging on SLS for human-friendly viewing:

- **Logs** — agent stdout/stderr → Logtail agent on the ECS → SLS Logstore
- **Traces** — OpenTelemetry SDK in the agent code → ARMS APM (which is OpenTelemetry-compatible)
- **Metrics** — host metrics from CloudMonitor agent + custom metrics from agent code → CloudMonitor → optionally piped to SLS

Don't pick "just logs" or "just metrics". You need all three:

- Logs answer "what did the agent do?"
- Traces answer "where did the time go?"
- Metrics answer "is this happening more often than usual?"

The cheapest mistake here is shipping with only `print()` to stdout. The most expensive is shipping with all three but never opening the dashboard until the first incident — at which point you discover your `agent` field is sometimes `null` because the SDK didn't propagate context. Wire it on day one, then poke it weekly so you know it works.

## Step 1: SLS Project and Logstores

Everything observability-related starts with one SLS project. One per environment is right; one per agent is too granular.

![Cloud monitoring command center with real-time metrics dashboards](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/07-observability-and-cost-control/wanxiang_monitoring.png)


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

Five logstores cover the practical needs:

- `agent-runs` — every step of every agent run (the firehose)
- `gateway-requests` — one row per LLM API call, with model, tokens, latency, cost
- `ecs-syslog` — the underlying OS logs from the ECS instances
- `ack-cluster` — Kubernetes events and pod logs (only if using ACK)
- `audit` — every change Terraform makes, retained for a year for compliance

The `audit` store has a year retention because it's small and you need it years later when "who changed the prod ALB on March 12" comes up. The 90-day window on `gateway-requests` is the one I tune most often — short enough to keep storage under ¥30/mo at 5 GB/day, long enough to do quarter-over-quarter cost trending without a Hive job.

## Step 2: Ship Logs from ECS

The Logtail agent is the official Aliyun-side log collector. Install it via cloud-init (add to `cloud-init.sh` from article 4):

```bash
# Install Logtail
wget http://logtail-release-cn-shanghai.oss-cn-shanghai.aliyuncs.com/linux64/logtail.sh
chmod +x logtail.sh && ./logtail.sh install cn-shanghai
service ilogtaild start

# Tag this machine for the SLS machine group
echo "${sls_user_id}::${sls_machine_group}" > /etc/ilogtail/user_log_config.json
```

The Logtail config — what files to tail and how to parse them — is a Terraform resource:

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

Now any file matching `/var/log/agents/*.log` on any tagged machine flows into SLS as JSON, queryable by every field. The agent code just does `logger.info(json.dumps({...}))` and the rest is automatic. The trick is being disciplined about the schema — settle on `ts`, `agent`, `session_id`, `step`, `phase`, `tokens`, `latency_ms`, `cost_cny`, `status`, `error_message` early, document it, and reject PRs that emit ad-hoc fields. Every undocumented field is a query you can't write at 3am.

## Step 3: Traces via OpenTelemetry → ARMS

For traces, ARMS APM is OpenTelemetry-compatible. The Terraform side is small — provision an ARMS instance and an environment:

```hcl
resource "alicloud_arms_environment" "agents" {
  environment_name      = "agents-${terraform.workspace}"
  bind_resource_id      = module.vpc.vpc_id
  environment_type      = "CS"             # cloud service
  environment_sub_type  = "ECS"
  payment_type          = "POSTPAY"
}
```

The agent code uses standard OpenTelemetry — nothing Aliyun-specific:

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

The two env vars come from ARMS — `ARMS_OTLP_ENDPOINT` is in the ARMS console, `ARMS_LICENSE_KEY` from your account. Wire both via Terraform outputs into the cloud-init template.

The reward: in ARMS you can see "this agent run took 12s; 9s of it was the third LLM call to qwen-max." That's the kind of visibility that actually changes how you build agents. The first time I saw a span tree from a real production session I rewrote the planner to skip retrieval when the user message was short — saved ~30% on p95 the same week.

## Step 4: Metrics with CloudMonitor

CloudMonitor catches the host-level metrics automatically (CPU, memory, network) once you install the cloud-monitor agent — which the `install_cloud_monitor` flag on the ACK node pool already does. For ECS, add to cloud-init:

```bash
wget http://cms-agent-cn-shanghai.oss-cn-shanghai.aliyuncs.com/release/cms_go_agent_install.sh
chmod +x cms_go_agent_install.sh && ./cms_go_agent_install.sh
```

For custom application metrics — "tokens consumed by research-agent" — emit them as SLS log entries with structured fields, then alert via SLS query. SLS-as-metrics is the pattern Aliyun pushes; CloudMonitor custom metrics work too but are clunkier to wire from Terraform, and you end up splitting your alerts across two consoles. Pick one. I pick SLS for everything except host CPU/memory/disk.

## Step 5: The Cost Dashboard

Here's where it gets interesting. Every LLM request hits the gateway, and the gateway logs one row per request to `gateway-requests` with fields like:

![Cloud cost optimization with resource right-sizing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/07-observability-and-cost-control/wanxiang_cost_optimization.png)


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

SLS can run SQL over these. The query for "daily cost by agent":

```sql
* | SELECT date_trunc('day', __time__) AS day,
          agent,
          SUM(cost_cny) AS daily_cost
  FROM log
  GROUP BY day, agent
  ORDER BY day, daily_cost DESC
```

Provision the dashboard via Terraform:

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

Open the SLS console, and you have a live dashboard:

![Stacked daily cost by category](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/07-observability-and-cost-control/fig2_cost_dashboard.png)

The dashboard is the answer to "which agent is burning my budget?" — a question you will be asked monthly, usually by someone who doesn't know what an agent is.

## Six SLS Queries That Earn Their Keep

A dashboard answers the long-running questions. Incidents need ad-hoc ones. After two years of agent ops, I have ~12 SLS queries pinned in a personal notes file. Six are worth sharing because they generalize across stacks.

### Query 1: Top Offending Agents in the Last Hour

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

My "who is doing the most" quick board. Run it on the cost-spike alarm and the offender is usually visible in two seconds.

### Query 2: Error Trace by Status Code

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

Distinguishes between "DashScope is throwing 500s" and "agents are sending bad requests" within seconds. The `arbitrary(error_message)` plucks one example so you don't have to drill in.

### Query 3: Token-Per-Step Distribution to Find Runaway Loops

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

Sessions with > 50 steps or > 500k tokens are usually loops. The runaway alarm fires; this query identifies the specific session ID. SSH to the box and `kill -9` the offender, then post-mortem in calm.

### Query 4: Latency Breakdown by Phase

If your agent emits structured timing per phase (planning, retrieval, LLM call, tool exec, reflection):

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

My "where did the time go" board. Catches when a tool starts taking 3x longer (usually because the upstream API got slower), or when retrieval starts dominating because the vector index needs reindexing.

### Query 5: cost-per-session leaderboard

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

The "expensive sessions" board. ¥5 per session is a useful threshold — anything above means either a long conversation (legitimate) or a buggy loop (not). Inspect the top 10 weekly to catch patterns before they become tomorrow's incident.

### Query 6: drop-off funnel for multi-step agents

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

For an agent with phases like `start → plan → tool_call → reflect → answer`, this shows how many sessions reach each phase. A sudden drop-off at `tool_call` means the tool is failing for many users — different from "the LLM is broken" or "the planner is dumb".

Save these as SLS Saved Queries via `alicloud_log_savedsearch` so they're discoverable from the console search bar:

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

Your future self at 3am will thank you. Repeat for the other five — it's a one-evening Terraform exercise that pays back the first time you're paged.

## Step 6: the four alarms

Queries answer the *what*. Alarms surface the *when*. Four alarms have earned their keep across multiple agent stacks I've shipped:

![Four alerts you should provision on day one](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/07-observability-and-cost-control/fig3_alert_rules.png)

### Alarm 1: cost ceiling

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

Fires once per 30 minutes if the day's LLM spend so far is above ¥800. Tune the threshold to your real budget. The throttling matters — without it, the alert fires every 5 minutes and the team mutes the channel within a week.

### Alarm 2: latency

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

`notify_threshold = 3` means three consecutive minutes above the threshold before firing — kills the noise from one-off slow LLM calls.

### Alarm 3: error rate

Same shape, query is `SUM(IF(status >= 500, 1, 0)) * 1.0 / COUNT(*) AS err_ratio`, condition `err_ratio > 0.02`. Throttling is shorter (5 minutes) because errors are usually a real ongoing event, not a transient blip. If 2% of requests are 5xx and that holds for five minutes, you have an incident, not a glitch.

### Alarm 4: token leak (runaway loop)

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

This is the one that has paid for itself, twice. An agent with a buggy stop condition can burn ¥10,000 in tokens overnight; this alert catches it within 2 minutes and gives you time to kill the offender. The first time it fired on a real prod stack of mine, it saved roughly ¥6,400 — a malformed JSON tool response was sending the planner back to step 1 forever.

## Alarm fatigue: the rules I follow to keep the channel signal-rich

The four alarms above are the right *types*. They become noise quickly if you don't tune them. Three rules have kept my DingTalk channel useful for years.

### Rule 1: throttle and de-dupe, always

Every alarm gets `throttling = "30m"` minimum. Without throttling, a sustained 1-hour issue produces 12 messages. Engineers mute the channel. The alarm stops working.

For non-trivial alarms, also set `notify_threshold` (the consecutive-firings count before sending):

```hcl
condition         = "p95s > 8"
notify_threshold  = 3   # 3 consecutive minute-windows
throttling        = "15m"
```

This converts "one bad minute → page" to "five bad minutes → page". Single-minute spikes from one slow LLM call don't wake people up; sustained problems do.

### Rule 2: severity-tiered routing

Not every alarm goes to the same channel. My setup has three:

- `#agent-incidents` — pages humans, P0/P1, includes phone fallback
- `#agent-warnings` — DingTalk only, P2, expected to be ack'd in business hours
- `#agent-info` — append-only log, no notification, for trend analysis

Map each alarm explicitly:

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

Now the cost-runaway alarm pages me; the latency p95 alarm waits until I look at it Monday. Different webhooks per severity is the cheapest way to get this right — cheaper than building a router service.

### Rule 3: every alarm has a runbook URL

The DingTalk message includes a link to a wiki page that documents:

1. What this alarm means
2. The first three diagnostic queries to run (from the six above)
3. Common causes
4. The escalation contact if it's not fixable in 30 min

```hcl
content = <<-MSG
  ${each.key} fired: ${"{{ severity }}"}

  Current value: ${"{{ value }}"}
  Threshold: ${each.value.threshold}

  Runbook: https://wiki.internal/agents/runbooks/${each.key}
  Dashboard: https://sls.console.aliyun.com/.../dashboard/agent-cost-overview
MSG
```

The runbook URL turns "why is this paging me" into "I know what to do". Cuts mean-time-to-resolution roughly in half. Maintain runbooks like code — review them when alarms change, prune stale ones quarterly.

## SLOs as the meta-layer above alarms

Alarms catch *now*. SLOs (Service Level Objectives) catch *trends*. I track three for any agent in production:

1. **Availability**: % of agent requests that returned a non-5xx within 30s. Target: 99.5% rolling 30 days.
2. **Latency**: p95 end-to-end agent response time. Target: under 6 seconds.
3. **Cost-per-session**: median ¥-per-session. Target: under ¥0.30 for the support agent, under ¥1.50 for the research agent.

Each SLO has a *budget* — the slack you have before missing it. When the budget is at 70%+ for the month, you're shipping freely. When it drops to 30%, you stop shipping new agent features and focus on stability. Below 0%, you've missed the SLO — that's a quarterly review item, not a "fix it tomorrow".

Provision SLO tracking as another SLS dashboard:

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

Look at this dashboard weekly in the team standup. It's the single best tool I know for converting "we're vaguely okay" into "we're 65% through this month's error budget — let's prioritise the retry logic fix." Turns ops feel into ops data, which is the difference between an engineering team and a panic team.

## Routing choices: DingTalk and ARMS-side alerts

In China, DingTalk is the team-chat default for most engineering orgs and SLS supports DingTalk webhooks natively. You can also fan out to email, SMS, and (via webhook) Slack/Teams/Lark. Pick whatever your team checks at 2am — that's the only criterion that matters.

ARMS has its own alerting too, useful for trace-level conditions ("any trace with > 30 spans" or "spans where `llm.model = qwen-max` and duration > 5s"). For the four alarms above, SLS-side is enough and avoids splitting your alerting story across two consoles. Use ARMS alerts only when SLS can't express what you need — usually that means span-tree shape conditions you can't reduce to a flat log query.

## What it costs

Observability has a real cost — usually 10-15% of the rest of your bill:

- **SLS**: ~¥0.35/GB ingested + ¥0.15/GB stored. A medium-traffic agent stack ingests ~5 GB/day → ¥50/mo for ingest, ¥20/mo for 30-day retention
- **ARMS APM**: ~¥600/mo for 1 environment with up to 100M spans
- **CloudMonitor**: free for standard metrics, ¥0.005 per custom metric per day

Budget ¥1000-1500/mo for full observability on a real production agent stack. Cheap compared to one missed cost-runaway alarm — the token-leak alert alone has paid for two years of SLS in my projects.

## What's Next

Article 8 is the end-to-end walkthrough. We compose every module from articles 2-7 — vpc-baseline, compute, storage, gateway, observability — into one `research-agent-stack` project and watch it come up with a single `terraform apply`. Real apply output, real timing, the full module DAG. The starter repo at the end is yours to fork.
