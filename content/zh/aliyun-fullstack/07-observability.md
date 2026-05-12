---
title: "阿里云全栈实战（七）：SLS 打造可观测性体系"
date: 2026-05-04 09:00:00
tags:
  - Alibaba Cloud
  - SLS
  - CloudMonitor
  - ARMS
  - Observability
  - Cloud Computing
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 7
description: "Build full-stack observability: SLS for log collection and querying, CloudMonitor for metrics and alerts, ARMS for distributed tracing. Set up a complete monitoring stack for a production web application."
disableNunjucks: true
translationKey: "aliyun-fullstack-7"
---
我职业生涯中最严重的一次生产事故，排查花了整整三个小时。当时有个 Node.js 服务间歇性返回 502，大约 5% 的请求受影响，而我手头没有任何工具。没有集中式日志（每台 ECS 实例都有自己的 `/var/log/`，我得一台台 SSH 上去看）。没有监控大盘（只能在终端里跑 `top` 和 `df -h`）。没有链路追踪（只能靠加 `console.log` 时间戳来猜是哪个下游调用卡住了）。三个小时后，我发现原因是有一个被遗忘的 cron 任务占着连接不放，导致 RDS 连接池在高负载下耗尽。修复只需两行代码，但由于缺乏可观测性，排查过程耗时三小时，异常艰难。

这个教训看似简单，代价却极高：可观测性不是系统稳定后的‘锦上添花’，而是上线前必须就绪的基础设施——理想情况下，甚至应在编写第一行代码前完成搭建，因为它直接决定了日志格式、请求 ID 透传机制与依赖库埋点策略。事后补建往往需要全面重构，而前置建设则可以自然融入开发流程。


这篇文章将完整介绍阿里云上的可观测性栈： SLS 负责日志， CloudMonitor 负责指标， ARMS 负责链路追踪。读完这篇，本系列一直在构建的生产 Web 应用将拥有一套可用的监控设置。 ECS 实例来自 [Part 2](/zh/aliyun-fullstack/02-ecs-compute/)，网络架构来自 [Part 3](/zh/aliyun-fullstack/03-vpc-networking/)。如果想用 Terraform 部署 这些监控资源，参考 [Terraform Part 7: Observability and Cost Control](/zh/terraform-agents/07-observability-and-cost-control/)。

## The Three Pillars of Observability

行业已经共识了三个信号，组合起来就能看清系统到底在发生什么：

![The three pillars of observability](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_three_pillars.png)

**Logs** 告诉你发生了什么。日志行会说"14:32:07，用户 abc123 请求了 /api/orders，因为数据库连接 30 秒超时而返回 500"。日志是离散事件，带时间戳且结构化。它是出事后的 forensic evidence。

**Metrics** 告诉你正在发生什么。指标会说"/api/orders 的 P99 延迟现在是 2.3 秒，应用层 CPU 利用率 78%， RDS 连接池耗尽 90%"。指标是数值时间序列。它是你在大盘上盯着的生命体征，用来在用户报障前发现问题。

**Traces** 告诉你为什么发生。追踪会说“这个特定请求在 API 网关花了 15ms，在订单服务花了 200ms，等待数据库查询花了 1800ms，序列化响应花了 50ms"。追踪跟随单个请求穿越多个服务。它是分布式系统里的 X 光，揭示哪个组件是瓶颈。

三者缺一不可：指标告诉你出事了（如错误率飙升），日志告诉你哪里出了问题（如数据库超时错误），追踪则揭示原因（如 orders 表上某条查询因索引被误删而执行全表扫描）。

在阿里云上，映射关系很清晰：

| Pillar | Alibaba Cloud Service | AWS Equivalent | What It Does |
|---|---|---|---|
| **Logs** | SLS (Simple Log Service) | CloudWatch Logs + OpenSearch | Log collection, indexing, querying, analytics |
| **Metrics** | CloudMonitor | CloudWatch Metrics | Infrastructure and custom metrics, alerting |
| **Traces** | ARMS (Application Real-Time Monitoring) | X-Ray + CloudWatch APM | APM, distributed tracing, service topology |

这三个服务是互通的。 CloudMonitor 可以基于 SLS 查询结果触发告警。 ARMS 追踪能关联到 SLS 日志条目。 SLS 大盘可以拉取 CloudMonitor 指标数据。集成度虽不如 Datadog 那种统一平台丝滑，但不用第三方工具也能覆盖 90% 的需求。

## SLS: Simple Log Service

SLS 是阿里云可观测性的核心组件。尽管名称含 ‘Simple’，它实则功能完备：集日志采集、存储、索引、查询、可视化与告警于一体。你可以将其理解为 AWS CloudWatch Logs 与 Elasticsearch 的融合体，并内置了 SQL 查询引擎。

![SLS log collection pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_sls_pipeline.png)

### Core Concepts

SLS 把 everything 组织成两层：

**Project** -- 顶层容器，通常每个环境或应用一个。 Project 是 Region 级别的。项目内的所有 Logstore、 Dashboard、告警共享同一个计费账户和访问控制。

**Logstore** -- 项目内的日志数据表。每个 Logstore 有自己的 schema、保留周期和索引配置。通常每个日志源创建一个 Logstore：一个给 nginx 访问日志，一个给应用日志，一个给系统日志。

```
SLS Project: prod-webapp
├── Logstore: nginx-access-log     (30-day retention)
├── Logstore: app-log              (30-day retention)
├── Logstore: system-log           (7-day retention)
├── Logstore: slow-query-log       (90-day retention)
├── Dashboard: ops-overview
├── Dashboard: error-analysis
└── Alert: high-error-rate
```

可通过 CLI 创建 Project 和 Logstore：

```bash
# Create the SLS project
aliyun sls CreateProject \
  --body '{
    "projectName": "prod-webapp",
    "description": "Production web application logs"
  }' \
  --endpoint cn-hangzhou.log.aliyuncs.com

# Create the nginx access log logstore
aliyun sls CreateLogStore \
  --project prod-webapp \
  --body '{
    "logstoreName": "nginx-access-log",
    "ttl": 30,
    "shardCount": 2,
    "autoSplit": true,
    "maxSplitShard": 8
  }' \
  --endpoint cn-hangzhou.log.aliyuncs.com

# Create the application log logstore
aliyun sls CreateLogStore \
  --project prod-webapp \
  --body '{
    "logstoreName": "app-log",
    "ttl": 30,
    "shardCount": 2,
    "autoSplit": true,
    "maxSplitShard": 8
  }' \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

`shardCount` 决定写入吞吐量。每个 shard 处理 5 MB/s 写入和 10 MB/s 读取。两个 shard 给你 10 MB/s 写入能力。开启 `autoSplit` 后，当写入压力超过阈值， SLS 会自动增加 shard，直到 `maxSplitShard`。

### SLS vs AWS: What Is Different

如果你来自 AWS 背景，这个映射值得厘清，因为 SLS 不是 1:1 的 CloudWatch Logs 等价物：

| Capability | SLS | AWS |
|---|---|---|
| Log collection agent | Logtail (SLS-native) | CloudWatch Agent |
| Full-text search | Built-in, sub-second latency | CloudWatch Logs Insights (slower) |
| SQL analytics | Full SQL syntax on log data | CloudWatch Logs Insights (limited SQL) |
| Dashboards | Built into SLS | CloudWatch Dashboards (separate) |
| Long-term storage | Built-in tiered storage | Export to S3 + Athena |
| Schema-on-read | Yes, with indexing | Partially (Insights) |
| Real-time streaming | Built-in consumer groups | Kinesis Data Streams (separate) |

最显著的区别在于： SLS 将日志存储、搜索与分析集成于单一服务；而 AWS 生态通常需组合使用 CloudWatch Logs （采集）、 S3 （长期存储）、 OpenSearch （搜索）和 Athena （SQL 分析）。 SLS 在一个地方全干了。代价是厂商锁定： SLS 查询语法并非跨云通用标准。

### Log Query Syntax

SLS 支持三种查询模式，掌握这三种模式，可显著提升查询效率。

**Full-text search** -- 直接输关键词。 SLS 搜所有索引字段。

```
ERROR
```

返回所有包含"ERROR"的日志行。

**Key-value search** -- 用字段名加操作符精确过滤。

```
status >= 500 and request_method: POST
```

返回 HTTP 状态码 500 及以上且请求方法是 POST 的条目。冒号 `:` 是包含操作符；`>=` 是数值比较。

**SQL analytics** -- 搜索表达式后加 pipe `|` 然后写标准 SQL。

```
status >= 500 | SELECT 
  date_format(__time__, '%H:%i') as time_bucket,
  count(*) as error_count,
  approx_distinct(client_ip) as affected_users
GROUP BY time_bucket
ORDER BY time_bucket
```


![SLS 查询语法：搜索过滤管道接 SQL 分析](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_sls_query_syntax.png)

找出所有 5xx 错误，然后按分钟分组，展示错误计数和受影响唯一用户数随时间的变化。`__time__` 是内置日志时间戳。`approx_distinct` 是 HyperLogLog 近似值——对高基数字段来说快且省内存。

下面是我日常用的真实查询：

```
# Error rate over the last hour (as percentage)
* | SELECT 
  round(count_if(status >= 500) * 100.0 / count(*), 2) as error_rate_pct,
  count(*) as total_requests
WHERE __time__ > unix_timestamp() - 3600

# P50, P90, P99 latency by endpoint
* | SELECT 
  request_uri,
  approx_percentile(request_time, 0.50) as p50_ms,
  approx_percentile(request_time, 0.90) as p90_ms,
  approx_percentile(request_time, 0.99) as p99_ms,
  count(*) as request_count
GROUP BY request_uri
ORDER BY request_count DESC
LIMIT 20

# Top 10 client IPs by request volume (bot detection)
* | SELECT 
  client_ip,
  count(*) as requests,
  count_if(status >= 400) as errors,
  round(count_if(status >= 400) * 100.0 / count(*), 1) as error_pct
GROUP BY client_ip
ORDER BY requests DESC
LIMIT 10

# Slow requests (>2 seconds)
request_time > 2 | SELECT 
  request_uri,
  request_time,
  status,
  client_ip,
  __time__
ORDER BY request_time DESC
LIMIT 50
```

### Enabling Indexes

SLS 默认不为字段建立索引。如需使用键值对查询或 SQL 分析，须预先配置字段索引。未配置字段索引时，仅支持全文关键词搜索；若未启用全文索引，则无法执行任何搜索。

```bash
aliyun sls CreateIndex \
  --project prod-webapp \
  --logstore nginx-access-log \
  --body '{
    "line": {
      "token": [",", " ", "\"", "\\t", ";", "=", "(", ")", "[", "]", "{", "}", "?", "@", "&", "/", ":", "'"'"'"]
    },
    "keys": {
      "client_ip": {"type": "text", "doc_value": true},
      "request_method": {"type": "text", "doc_value": true},
      "request_uri": {"type": "text", "doc_value": true, "token": ["/", "?", "&", "="]},
      "status": {"type": "long", "doc_value": true},
      "body_bytes_sent": {"type": "long", "doc_value": true},
      "request_time": {"type": "double", "doc_value": true},
      "upstream_response_time": {"type": "double", "doc_value": true},
      "http_user_agent": {"type": "text", "doc_value": true}
    }
  }' \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

`line` 部分启用全文索引，指定分词符。`keys` 部分定义字段级索引。设 `doc_value: true` 启用该字段的 SQL 分析。每个索引字段都占存储，只索引你真要查的字段。

> **Cost note:** Indexing roughly doubles your storage cost. For high-volume logs where you only need full-text search, skip per-field indexing and rely on the `line` index. For access logs where you run SQL dashboards, per-field indexing is worth the cost.
## 配置 Logtail

Logtail 是 SLS 官方日志采集 agent，运行于 ECS 实例，负责监控日志文件、按配置解析并投递至 SLS。它轻量（通常占用 50–100 MB 内存， CPU 使用率 <1%）、可靠（通过本地缓冲应对网络中断），且与 SLS 深度集成。

![Logtail 采集器部署架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_logtail_architecture.png)

### 安装

只要 ECS 实例和 SLS 在同一个 region，一条命令就能搞定：

```bash
# Download and install Logtail (cn-hangzhou region)
wget http://logtail-release-cn-hangzhou.oss-cn-hangzhou-internal.aliyuncs.com/linux64/logtail.sh -O logtail.sh
chmod 755 logtail.sh
sudo ./logtail.sh install cn-hangzhou

# Verify installation
sudo /etc/init.d/ilogtaild status
```

安装脚本会自动识别你是走 VPC 内网还是公网，并配置对应的 endpoint。走 VPC 内网是免费的——同 region 内日志投递不收流量费。

装好后，得在 SLS 里建个 machine group，用来标记哪些实例该收哪些采集配置：

```bash
aliyun sls CreateMachineGroup \
  --project prod-webapp \
  --body '{
    "groupName": "prod-app-servers",
    "machineIdentifyType": "ip",
    "groupType": "",
    "groupAttribute": {
      "externalName": "",
      "groupTopic": ""
    },
    "machineList": [
      "10.0.10.5",
      "10.0.10.6",
      "10.0.11.5",
      "10.0.11.6"
    ]
  }' \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

如果是弹性伸缩组， IP 会变，这时候别用 IP 标识，改用用户自定义标识。在每个实例上创建文件 `/etc/ilogtail/user_defined_id`，里面写上组标识比如 `prod-app-servers`，然后把 `machineIdentifyType` 设成 `userdefined`。

### 采集 Nginx 访问日志

最常见的采集场景就是解析 Nginx 访问日志，而且最好用自定义格式。首先，配置 Nginx 写入结构化日志：

```nginx
# /etc/nginx/nginx.conf
log_format structured '$remote_addr - $remote_user [$time_local] '
                      '"$request_method $request_uri $server_protocol" '
                      '$status $body_bytes_sent '
                      '"$http_referer" "$http_user_agent" '
                      '$request_time $upstream_response_time';

access_log /var/log/nginx/access.log structured;
```

接着配个 Logtail 采集配置来解析这个格式：

```json
{
  "configName": "nginx-access-config",
  "inputType": "file",
  "inputDetail": {
    "logType": "common_reg_log",
    "logPath": "/var/log/nginx",
    "filePattern": "access.log*",
    "topicFormat": "none",
    "timeFormat": "%d/%b/%Y:%H:%M:%S",
    "regex": "([\\d.]+) - (\\S+) \\[(\\S+ \\S+)\\] \"(\\w+) (\\S+) (\\S+)\" (\\d+) (\\d+) \"([^\"]*)\" \"([^\"]*)\" ([\\d.]+) ([\\d.-]+)",
    "key": [
      "client_ip", "remote_user", "time_local",
      "request_method", "request_uri", "protocol",
      "status", "body_bytes_sent",
      "http_referer", "http_user_agent",
      "request_time", "upstream_response_time"
    ]
  },
  "outputType": "LogService",
  "outputDetail": {
    "projectName": "prod-webapp",
    "logstoreName": "nginx-access-log"
  }
}
```

用 CLI 应用配置：

```bash
aliyun sls CreateConfig \
  --project prod-webapp \
  --body @nginx-access-config.json \
  --endpoint cn-hangzhou.log.aliyuncs.com

# Bind the config to the machine group
aliyun sls ApplyConfigToMachineGroup \
  --project prod-webapp \
  --groupName prod-app-servers \
  --configName nginx-access-config \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

一分钟内日志就开始流入了。你可以在 SLS 控制台查，或者用 CLI 验证：

```bash
aliyun sls GetLogs \
  --project prod-webapp \
  --logstore nginx-access-log \
  --from $(date -d '5 minutes ago' +%s) \
  --to $(date +%s) \
  --query '*' \
  --line 5 \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

![Logtail 采集配置与机器组的绑定模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_logtail_binding.png)


### 采集应用日志（推荐 JSON 格式）

采应用日志的话，我强烈建议直接用 JSON 格式。这样就不用写正则解析了，不容易出错，字段索引也是自动的。

配置你的应用输出 JSON 日志。这是个用 pino 的 Node.js 例子：

```javascript
// logger.js
const pino = require('pino');

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level(label) {
      return { level: label };
    }
  },
  timestamp: () => `,"timestamp":"${new Date().toISOString()}"`,
  base: {
    service: 'order-service',
    env: process.env.NODE_ENV,
    hostname: require('os').hostname()
  }
});

module.exports = logger;
```

打出来的日志长这样：

```json
{"level":"info","timestamp":"2026-05-20T08:15:32.456Z","service":"order-service","env":"production","hostname":"app-01","msg":"order created","orderId":"ORD-12345","userId":"USR-789","amount":129.99,"latencyMs":45}
```

JSON 日志的 Logtail 配置简单多了，根本不需要正则：

```json
{
  "configName": "app-json-log-config",
  "inputType": "file",
  "inputDetail": {
    "logType": "json_log",
    "logPath": "/var/log/app",
    "filePattern": "*.log",
    "topicFormat": "none",
    "timeKey": "timestamp",
    "timeFormat": "%Y-%m-%dT%H:%M:%S"
  },
  "outputType": "LogService",
  "outputDetail": {
    "projectName": "prod-webapp",
    "logstoreName": "app-log"
  }
}
```

### 采集系统日志

像 syslog、 journald 这种系统级事件， Logtail 原生就支持：

```json
{
  "configName": "syslog-config",
  "inputType": "file",
  "inputDetail": {
    "logType": "common_reg_log",
    "logPath": "/var/log",
    "filePattern": "syslog*",
    "topicFormat": "none",
    "regex": "(\\w+ \\d+ \\d+:\\d+:\\d+) (\\S+) (\\S+): (.*)",
    "key": ["timestamp", "hostname", "program", "message"],
    "timeFormat": "%b %d %H:%M:%S"
  },
  "outputType": "LogService",
  "outputDetail": {
    "projectName": "prod-webapp",
    "logstoreName": "system-log"
  }
}
```

## 构建仪表盘

没人看的仪表盘比没有更糟糕，因为它给你一种虚假的安全感。关键是要围绕故障排查时真正会问的问题去建，而不是堆砌那些看起来厉害的指标。

![SLS 仪表盘布局](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_dashboard_layout.png)

### 五个核心面板

每个生产环境的 Web 应用，主仪表盘上只需要这五个面板：

| 面板 | SLS 查询 | 它能告诉你什么 |
|---|---|---|
| QPS 趋势 | `* \| SELECT date_trunc('minute', __time__) as t, count(*)/60.0 as qps GROUP BY t ORDER BY t` | 流量模式——是流量突增导致的问题，还是流量跌了（上游故障）？ |
| 错误率 | `* \| SELECT date_trunc('minute', __time__) as t, round(count_if(status>=500)*100.0/count(*),2) as err_pct GROUP BY t ORDER BY t` | 错误率是否升高？任何超过 0.1% 的情况都值得调查。 |
| P99 延迟 | `* \| SELECT date_trunc('minute', __time__) as t, approx_percentile(request_time, 0.99) as p99 GROUP BY t ORDER BY t` | 服务是否变慢？ P99 能捕捉到平均值掩盖的长尾延迟。 |
| 顶部接口 | `* \| SELECT request_uri, count(*) as cnt, approx_percentile(request_time, 0.50) as p50 GROUP BY request_uri ORDER BY cnt DESC LIMIT 10` | 流量去哪了？哪些接口慢？ |
| 状态码分布 | `* \| SELECT status, count(*) as cnt GROUP BY status ORDER BY cnt DESC` | 有没有异常的 4xx/5xx 模式？ |

![五个核心仪表盘面板示意](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_dashboard_panels_mockup.png)


### 创建仪表盘

SLS 仪表盘是用 JSON 定义的。这是个精简版但能直接用的运维仪表盘：

```json
{
  "dashboardName": "ops-overview",
  "displayName": "Production Ops Overview",
  "charts": [
    {
      "title": "QPS (Requests per Second)",
      "type": "line",
      "search": {
        "logstore": "nginx-access-log",
        "topic": "",
        "query": "* | SELECT date_trunc('minute', __time__) as time, round(count(*)/60.0, 1) as qps GROUP BY time ORDER BY time",
        "start": "-1h",
        "end": "now"
      },
      "display": {
        "xAxis": ["time"],
        "yAxis": ["qps"],
        "height": 300
      }
    },
    {
      "title": "Error Rate (%)",
      "type": "line",
      "search": {
        "logstore": "nginx-access-log",
        "topic": "",
        "query": "* | SELECT date_trunc('minute', __time__) as time, round(count_if(status >= 500) * 100.0 / count(*), 2) as error_rate GROUP BY time ORDER BY time",
        "start": "-1h",
        "end": "now"
      },
      "display": {
        "xAxis": ["time"],
        "yAxis": ["error_rate"],
        "height": 300
      }
    },
    {
      "title": "P50 / P90 / P99 Latency (seconds)",
      "type": "line",
      "search": {
        "logstore": "nginx-access-log",
        "topic": "",
        "query": "* | SELECT date_trunc('minute', __time__) as time, approx_percentile(request_time, 0.50) as p50, approx_percentile(request_time, 0.90) as p90, approx_percentile(request_time, 0.99) as p99 GROUP BY time ORDER BY time",
        "start": "-1h",
        "end": "now"
      },
      "display": {
        "xAxis": ["time"],
        "yAxis": ["p50", "p90", "p99"],
        "height": 300
      }
    },
    {
      "title": "Status Code Distribution",
      "type": "pie",
      "search": {
        "logstore": "nginx-access-log",
        "topic": "",
        "query": "* | SELECT CASE WHEN status >= 200 AND status < 300 THEN '2xx' WHEN status >= 300 AND status < 400 THEN '3xx' WHEN status >= 400 AND status < 500 THEN '4xx' WHEN status >= 500 THEN '5xx' END as code_class, count(*) as cnt GROUP BY code_class ORDER BY cnt DESC",
        "start": "-1h",
        "end": "now"
      },
      "display": {
        "xAxis": ["code_class"],
        "yAxis": ["cnt"],
        "height": 300
      }
    },
    {
      "title": "Top 10 Endpoints by Volume",
      "type": "table",
      "search": {
        "logstore": "nginx-access-log",
        "topic": "",
        "query": "* | SELECT request_uri, count(*) as requests, round(avg(request_time), 3) as avg_latency, approx_percentile(request_time, 0.99) as p99_latency, count_if(status >= 500) as errors GROUP BY request_uri ORDER BY requests DESC LIMIT 10",
        "start": "-1h",
        "end": "now"
      },
      "display": {
        "height": 300
      }
    }
  ]
}
```

用 CLI 创建：

```bash
aliyun sls CreateDashboard \
  --project prod-webapp \
  --body @ops-dashboard.json \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

> **实战建议：** 先用 SLS 控制台的可视化编辑器交互式的把图表调好，再导出 JSON 定义去做版本控制。手写仪表盘 JSON 太折磨人了。控制台的查询 explorer 能让你在提交到仪表盘面板前，先即时测试 SLS 查询语句。
## CloudMonitor: 基础设施监控与告警

SLS 负责日志， CloudMonitor 负责指标——也就是那些追踪基础设施健康状态的数值时间序列。所有阿里云资源默认都开着 CloudMonitor。只要你创建了 ECS 实例、 RDS 数据库或者 SLB 负载均衡，监控立马就开始收集基础指标。

### 内置指标

CloudMonitor 开箱即用，每台 ECS 实例都会收集这些指标：

| 指标 | 描述 | 采集间隔 |
|---|---|---|
| `CPUUtilization` | CPU 使用率百分比 | 60 秒 |
| `MemoryUsedPercent` | 内存使用率百分比 | 60 秒 |
| `DiskReadBPS` / `DiskWriteBPS` | 磁盘 I/O 吞吐量 | 60 秒 |
| `DiskReadIOPS` / `DiskWriteIOPS` | 磁盘 I/O 操作数 | 60 秒 |
| `InternetInRate` / `InternetOutRate` | 网络吞吐量 | 60 秒 |
| `IntranetInRate` / `IntranetOutRate` | VPC 内部网络吞吐量 | 60 秒 |
| `disk_usage_percent` | 磁盘空间使用率（需 agent） | 60 秒 |
| `load_5m` | 5 分钟负载平均值（需 agent） | 60 秒 |

前六个指标直接从 Hypervisor 层面拿，不用装 agent。最后两个得在实例上装 CloudMonitor agent。跟 Logtail 一起装就行：

```bash
# Install CloudMonitor agent
ARGUS_VERSION=3.5.9
REGION_ID=cn-hangzhou

wget "http://cms-download.aliyun.com/cms-go-client/$ARGUS_VERSION/cms-go-client.linux-amd64.tar.gz"
tar xzf cms-go-client.linux-amd64.tar.gz
sudo ./cms-go-client.linux-amd64/cloudmonitor --install
sudo ./cms-go-client.linux-amd64/cloudmonitor --start

# Verify
sudo ./cms-go-client.linux-amd64/cloudmonitor --status
```

其他服务不用装 agent 也能提供指标：

| 服务 | 关键指标 |
|---|---|
| **RDS** | CPU、内存、连接数、 IOPS、磁盘使用率、每秒慢查询数 |
| **SLB** | 活跃连接数、新建连接数、 QPS、健康主机数、延迟 |
| **OSS** | 请求数、带宽、可用性、首字节延迟 |
| **Redis (Tair)** | CPU、内存使用率、连接数、 QPS、命中率、驱逐数 |
| **NAT Gateway** | 活跃连接数、带宽、包速率 |

### 自定义指标

有些应用层的指标 CloudMonitor 不会自动收集，得通过 API 自己推自定义指标：

```bash
# Push a custom metric: order processing latency
aliyun cms PutCustomMetric \
  --MetricList '[
    {
      "groupId": 12345,
      "metricName": "order_processing_latency",
      "dimensions": {
        "service": "order-service",
        "env": "production"
      },
      "time": "'$(date +%s%3N)'",
      "type": 0,
      "values": {
        "Average": 245,
        "Maximum": 1830,
        "Minimum": 12,
        "SampleCount": 150,
        "Sum": 36750
      }
    }
  ]'
```

写代码的时候有个经验：别每请求推一次，把自定义指标攒一批，按固定节奏（比如每 60 秒）推上去：

```javascript
// metrics-reporter.js
const Core = require('@alicloud/pop-core');

const client = new Core({
  accessKeyId: process.env.ALIBABA_CLOUD_ACCESS_KEY_ID,
  accessKeySecret: process.env.ALIBABA_CLOUD_ACCESS_KEY_SECRET,
  endpoint: 'https://metrics.cn-hangzhou.aliyuncs.com',
  apiVersion: '2019-01-01'
});

class MetricsBuffer {
  constructor(groupId, flushIntervalMs = 60000) {
    this.groupId = groupId;
    this.buffer = {};
    setInterval(() => this.flush(), flushIntervalMs);
  }

  record(metricName, value, dimensions = {}) {
    const key = `${metricName}:${JSON.stringify(dimensions)}`;
    if (!this.buffer[key]) {
      this.buffer[key] = { metricName, dimensions, values: [] };
    }
    this.buffer[key].values.push(value);
  }

  async flush() {
    const metricList = [];
    for (const [key, entry] of Object.entries(this.buffer)) {
      if (entry.values.length === 0) continue;
      const sorted = entry.values.sort((a, b) => a - b);
      metricList.push({
        groupId: this.groupId,
        metricName: entry.metricName,
        dimensions: entry.dimensions,
        time: Date.now().toString(),
        type: 0,
        values: {
          Average: sorted.reduce((a, b) => a + b, 0) / sorted.length,
          Maximum: sorted[sorted.length - 1],
          Minimum: sorted[0],
          SampleCount: sorted.length,
          Sum: sorted.reduce((a, b) => a + b, 0)
        }
      });
    }
    this.buffer = {};
    if (metricList.length === 0) return;

    await client.request('PutCustomMetric', {
      MetricList: JSON.stringify(metricList)
    });
  }
}

module.exports = MetricsBuffer;
```

### 事件监控

CloudMonitor 还能盯系统事件——也就是那些发生在资源上、不属于常规指标采集的事情。比如 ECS 实例重启、磁盘报错、计划维护、安全警报。这些是离散事件，不是连续的时间序列。

重点盯这几个事件：

| 事件 | 含义 | 建议操作 |
|---|---|---|
| `Instance:SystemFailure.Reboot` | 阿里云因宿主机故障重启了你的实例 | 检查应用是否干净恢复 |
| `Disk:Stalled` | 磁盘 I/O 停滞，可能是存储后端问题 | 监控是否有数据损坏 |
| `Instance:PerformanceLimited` | 突发型实例（t 系列）耗尽了 CPU 积分 | 升级实例类型或切换到非突发型 |
| `SecurityGroup:AuthorizeFailed` | 连接被安全组规则拦截 | 确认这是预期行为还是配置错误 |

订阅事件接收通知：

```bash
aliyun cms PutEventRule \
  --RuleName ecs-system-events \
  --EventPattern '{
    "product": "ECS",
    "eventTypeList": [
      "StatusNotification",
      "SystemFailure.Reboot",
      "SystemMaintenance.Reboot",
      "Disk:Stalled"
    ]
  }' \
  --ContactGroups '["ops-team"]' \
  --State ENABLED
```
## 告警配置

告警是连接可观测性和行动的桥梁。对的告警能在错误率飙升时凌晨 3 点把你叫醒。错的告警也会凌晨 3 点把你叫醒，只不过是因为定时备份时 CPU 短暂冲到了 81%， 30 秒后又回去了。把告警阈值设准了是一门艺术，不过下面这几条经验法则我一直用着挺顺手。

![告警配置与通知流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_alert_flow.png)

### 告警设计原则

1. **告警症状，别告警原因。** 告警“错误率 > 1%"，别告警"CPU > 80%"。 CPU 高只有导致用户能感知的影响时才算问题。错误率本身就是用户能感知的影响。
2. **使用持续阈值。** 别单凭一个数据点就发告警。要求条件持续 3-5 分钟，过滤掉瞬时毛刺。
3. **正好设三个 severity 级别。** Critical （立刻打电话叫人）、 Warning （几小时内需要排查）、 Info （记录留档）。超过三个级别，没人搞得清每个级别到底啥意思。
4. **已知维护期间静音。** 没什么比明明提前通知了要部署，告警却还在狂发更破坏信任的了。

![基于症状告警 vs 基于原因告警](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_alert_symptoms_vs_causes.png)


### 设置告警规则

下面是每个生产系统都该有的四条告警规则：

**1. 高错误率（Critical）**

```bash
# SLS Alert: Error rate exceeds 1% for 5 minutes
aliyun sls CreateAlert \
  --project prod-webapp \
  --body '{
    "name": "high-error-rate",
    "displayName": "High Error Rate (>1%)",
    "description": "5xx error rate exceeds 1% over 5 minutes",
    "schedule": {
      "type": "FixedRate",
      "interval": "1m"
    },
    "configuration": {
      "type": "default",
      "queryList": [
        {
          "project": "prod-webapp",
          "logstore": "nginx-access-log",
          "query": "* | SELECT round(count_if(status >= 500) * 100.0 / count(*), 2) as error_rate, count(*) as total WHERE __time__ > unix_timestamp() - 300",
          "timeSpanType": "Custom",
          "start": "-5m",
          "end": "now"
        }
      ],
      "condition": "error_rate > 1 && total > 100",
      "threshold": 1,
      "noDataFire": false,
      "sendResolved": true,
      "notificationList": [
        {
          "type": "DingTalk",
          "serviceUri": "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN",
          "content": "CRITICAL: Error rate is ${error_rate}% (${total} requests in last 5min)"
        }
      ]
    }
  }' \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

`total > 100` 这个条件是为了防止低流量期的误报。如果只进了 3 个请求却挂了 1 个，数值上看是 33% 错误率——看着吓人，实际没啥意义。

**2. 持续高 CPU （Warning）**

```bash
# CloudMonitor Alert: CPU > 80% for 5 minutes
aliyun cms PutResourceMetricRule \
  --RuleId cpu-high-warning \
  --RuleName "High CPU Utilization" \
  --Namespace acs_ecs_dashboard \
  --MetricName CPUUtilization \
  --ContactGroups '["ops-team"]' \
  --Resources '[{"instanceId":"i-bp1xxxxxxxxx"}]' \
  --Escalations.Warn.Statistics Average \
  --Escalations.Warn.ComparisonOperator GreaterThanOrEqualToThreshold \
  --Escalations.Warn.Threshold 80 \
  --Escalations.Warn.Times 5 \
  --Period 60
```

`Times: 5` 意味着条件必须在连续 5 个评估周期内都为真（间隔 60 秒，共 5 分钟）。流量突发导致的短暂 CPU 飙升不会触发这条告警。

**3. 磁盘空间不足（Warning）**

```bash
aliyun cms PutResourceMetricRule \
  --RuleId disk-space-warning \
  --RuleName "Low Disk Space" \
  --Namespace acs_ecs_dashboard \
  --MetricName diskusage_utilization \
  --ContactGroups '["ops-team"]' \
  --Resources '[{"instanceId":"i-bp1xxxxxxxxx"}]' \
  --Escalations.Warn.Statistics Maximum \
  --Escalations.Warn.ComparisonOperator GreaterThanOrEqualToThreshold \
  --Escalations.Warn.Threshold 80 \
  --Escalations.Warn.Times 3 \
  --Period 60 \
  --Escalations.Critical.Statistics Maximum \
  --Escalations.Critical.ComparisonOperator GreaterThanOrEqualToThreshold \
  --Escalations.Critical.Threshold 90 \
  --Escalations.Critical.Times 1 \
  --Period 60
```

这里设了两个升级级别：磁盘使用率 80% 持续时 warn，达到 90% 立刻 critical。磁盘写满是我见过最容易预防、却也最常见的宕机原因。

**4. 数据库慢查询（Warning）**

```bash
# SLS Alert: P99 query time exceeds 1 second
aliyun sls CreateAlert \
  --project prod-webapp \
  --body '{
    "name": "slow-queries",
    "displayName": "Slow Database Queries",
    "description": "P99 query time exceeds 1 second over 10 minutes",
    "schedule": {
      "type": "FixedRate",
      "interval": "5m"
    },
    "configuration": {
      "type": "default",
      "queryList": [
        {
          "project": "prod-webapp",
          "logstore": "slow-query-log",
          "query": "* | SELECT approx_percentile(query_time, 0.99) as p99_query_time, count(*) as slow_query_count WHERE __time__ > unix_timestamp() - 600",
          "timeSpanType": "Custom",
          "start": "-10m",
          "end": "now"
        }
      ],
      "condition": "p99_query_time > 1",
      "threshold": 1,
      "noDataFire": false,
      "sendResolved": true,
      "notificationList": [
        {
          "type": "DingTalk",
          "serviceUri": "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN",
          "content": "WARNING: P99 query time is ${p99_query_time}s (${slow_query_count} slow queries in last 10min)"
        }
      ]
    }
  }' \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

### 联系人组和通知渠道

CloudMonitor 通过联系人组来路由告警通知。建个组，加上通知渠道：

```bash
# Create a contact group
aliyun cms PutContactGroup \
  --ContactGroupName ops-team \
  --ContactNames '["engineer-a", "engineer-b", "on-call"]' \
  --Describe "Production operations team"

# Create a contact with DingTalk and email
aliyun cms PutContact \
  --ContactName engineer-a \
  --Channels '{
    "Mail": "engineer-a@company.com",
    "DingWebHook": "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN"
  }' \
  --Describe "Engineer A - primary on-call"
```

支持的通知渠道：

| Channel | Use Case |
|---|---|
| **Email** | 非紧急警告，每日汇总 |
| **DingTalk webhook** | 团队可见的告警，事故协调 |
| **SMS** | 需要立即关注的 Critical 告警 |
| **Phone call** | 生产不可用的严重程度（慎用） |
| **Webhook (HTTP)** | 对接 PagerDuty、 Slack 或自定义系统 |

![告警严重级别到通知渠道的路由](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_alert_severity_routing.png)


> **静音期：** 对于计划内的维护窗口，在告警规则上设置静音期来抑制通知。这比直接禁用告警更好，因为告警依然会触发并记录事件——你只是不会为了已知的事情被吵醒而已。

## ARMS：应用实时监控服务

ARMS 补上了可观测性的第三块拼图：链路追踪（traces）。 SLS 告诉你发生了什么， CloudMonitor 告诉你系统层面的影响，而 ARMS 告诉你问题究竟出在应用的哪个位置。

![ARMS 分布式链路追踪](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_arms_traces.png)

### ARMS 能做什么

ARMS 是一个 APM （应用性能监控）平台，提供以下功能：

- **分布式追踪** -- 跨服务、数据库、缓存和消息队列追踪请求。精确看到时间花在哪了。
- **服务拓扑** -- 自动发现的服务通信地图。一眼看清依赖关系、调用量和错误率。
- **异常诊断** -- 自动捕获和聚合异常，包含堆栈跟踪、频率和受影响用户。
- **慢事务分析** -- 下钻具体的慢请求，查看完整调用链，包括数据库查询和外部 API 调用。

ARMS 支持自动埋点（automatic instrumentation）的语言包括：

| Language | Agent Type | What Gets Instrumented |
|---|---|---|
| Java | ByteBuddy agent | Spring, Dubbo, gRPC, JDBC, Redis, HTTP clients |
| Node.js | npm package | Express, Koa, MySQL, Redis, HTTP, gRPC |
| Python | pip package | Django, Flask, SQLAlchemy, Redis, requests |
| Go | SDK | net/http, gRPC, database/sql, go-redis |
| PHP | Extension | Laravel, ThinkPHP, MySQLi, cURL |

所谓自动埋点，意味着你不需要修改应用代码。 Agent 会拦截框架层的调用，自动生成 trace spans。你只需要在启动命令里加上 Agent， trace 就会出现。

### 安装 ARMS Agent （Node.js）

对于我们一直在 ECS 实例上运行的 Node.js 应用：

```bash
# Install the ARMS Node.js agent
npm install @alicloud/china-arms-apm --save

# Set environment variables
export ARMS_APP_NAME=order-service
export ARMS_LICENSE_KEY=your-license-key-from-console
export ARMS_REGION_ID=cn-hangzhou
```

在应用入口文件的最顶端添加 agent require，必须放在任何其他 import 之前：

```javascript
// app.js - this MUST be the first line
require('@alicloud/china-arms-apm').default({ appName: 'order-service' });

// everything else follows
const express = require('express');
const app = express();
// ... rest of your application
```

Java 应用更简单——只需加一个 JVM 参数：

```bash
java -javaagent:/path/to/arms-agent.jar \
     -Darms.appName=order-service \
     -Darms.licenseKey=your-license-key \
     -jar your-application.jar
```

### 查看链路

Agent 运行后， ARMS 开始为每个进入的请求生成 trace。每个 trace 由 spans 组成——每个操作（HTTP 调用、数据库查询、缓存查找）对应一个 span。这些 spans 形成一棵树，展示完整的请求生命周期。

典型的 API 请求 trace 长这样：

```sql
Trace: abc-123-def (total: 234ms)
├── [order-service] POST /api/orders                       0-234ms
│   ├── [order-service] MySQL: SELECT * FROM users         12-18ms
│   ├── [order-service] Redis: GET user:789:cart            19-21ms
│   ├── [order-service] MySQL: INSERT INTO orders           22-89ms
│   ├── [order-service] HTTP: POST payment-service/charge   90-210ms
│   │   ├── [payment-service] POST /charge                  95-205ms
│   │   │   ├── [payment-service] MySQL: SELECT balance     100-105ms
│   │   │   ├── [payment-service] HTTP: POST alipay.com     106-195ms
│   │   │   └── [payment-service] MySQL: UPDATE balance     196-203ms
│   └── [order-service] Redis: DEL user:789:cart            211-213ms
```

从这个 trace 你能看出，支付服务调用支付宝花了 89ms——这是你优化不了的外部依赖。数据库 INSERT 花了 67ms——如果平时没这么久，这就值得排查。总耗时 234ms 对于 checkout 流程来说可以接受，但如果是 2340ms，你就能精确知道该看哪个 span。

### 链路与日志关联

真正的威力在于把 ARMS 链路和 SLS 日志条目关联起来。当链路显示某个数据库查询很慢时，你想看到对应的应用日志来理解上下文——是谁触发的、传了什么参数、查询计划是什么。

通过在日志输出中包含 trace ID 来启用链路 - 日志关联：

```javascript
// middleware to inject trace ID into logs
app.use((req, res, next) => {
  const traceId = req.headers['eagleeye-traceid'] || 'no-trace';
  req.logger = logger.child({ traceId });
  next();
});

// In your route handlers, use req.logger
app.post('/api/orders', async (req, res) => {
  req.logger.info({ userId: req.user.id }, 'creating order');
  // ... business logic
  req.logger.info({ orderId: order.id, latencyMs: elapsed }, 'order created');
});
```

现在在 SLS 里，你可以搜索特定 trace 关联的所有日志：

```
traceId: abc-123-def
```

在 ARMS 中，每个 trace span 也能链回对应的 SLS 日志条目。这种双向链接才是让生产问题排查变快的关键。

![通过 traceId 实现 trace 与日志的双向关联](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_trace_log_correlation.png)

## 解决方案：全栈可观测性搭建

咱们把前面几篇的内容串起来，直接上一套完整的搭建流程。这里默认你已经在 SLB 负载均衡后面跑了 ECS 实例，后端挂着 RDS 数据库——也就是本系列前几篇文章里提到的那个架构。

### 第一步：在所有 ECS 实例上安装 Agent

搞个 cloud-init 脚本或者 Ansible 实战手册，把两个 Agent 都装到每台应用服务器上：

```bash
#!/bin/bash
# install-observability.sh
# Run on each ECS instance in the app tier

set -euo pipefail

REGION=cn-hangzhou
SLS_PROJECT=prod-webapp

echo "=== Installing Logtail ==="
wget -q http://logtail-release-${REGION}.oss-${REGION}-internal.aliyuncs.com/linux64/logtail.sh -O /tmp/logtail.sh
chmod 755 /tmp/logtail.sh
sudo /tmp/logtail.sh install ${REGION}

# Set machine group identity
sudo mkdir -p /etc/ilogtail
echo "prod-app-servers" | sudo tee /etc/ilogtail/user_defined_id

# Restart Logtail to pick up identity
sudo /etc/init.d/ilogtaild restart

echo "=== Installing CloudMonitor Agent ==="
ARGUS_VERSION=3.5.9
wget -q "http://cms-download.aliyun.com/cms-go-client/${ARGUS_VERSION}/cms-go-client.linux-amd64.tar.gz" -O /tmp/cms-agent.tar.gz
cd /tmp && tar xzf cms-agent.tar.gz
sudo /tmp/cms-go-client.linux-amd64/cloudmonitor --install
sudo /tmp/cms-go-client.linux-amd64/cloudmonitor --start

echo "=== Installing ARMS Node.js Agent ==="
cd /opt/app
npm install @alicloud/china-arms-apm --save

echo "=== Verifying ==="
sudo /etc/init.d/ilogtaild status
sudo /tmp/cms-go-client.linux-amd64/cloudmonitor --status

echo "All agents installed successfully."
```

### 第二步：配置日志采集

把 Logtail 配置应用到所有日志源上：

```bash
# Create SLS project and logstores
aliyun sls CreateProject --body '{"projectName":"prod-webapp","description":"Production logs"}' \
  --endpoint cn-hangzhou.log.aliyuncs.com

for STORE in nginx-access-log app-log system-log slow-query-log; do
  aliyun sls CreateLogStore --project prod-webapp \
    --body "{\"logstoreName\":\"${STORE}\",\"ttl\":30,\"shardCount\":2,\"autoSplit\":true,\"maxSplitShard\":8}" \
    --endpoint cn-hangzhou.log.aliyuncs.com
done

# Create machine group (user-defined identity)
aliyun sls CreateMachineGroup --project prod-webapp \
  --body '{
    "groupName":"prod-app-servers",
    "machineIdentifyType":"userdefined",
    "groupAttribute":{"externalName":"prod-app-servers","groupTopic":""},
    "machineList":["prod-app-servers"]
  }' \
  --endpoint cn-hangzhou.log.aliyuncs.com

# Apply collection configs (nginx, app JSON, syslog)
aliyun sls CreateConfig --project prod-webapp --body @nginx-access-config.json \
  --endpoint cn-hangzhou.log.aliyuncs.com
aliyun sls ApplyConfigToMachineGroup --project prod-webapp \
  --groupName prod-app-servers --configName nginx-access-config \
  --endpoint cn-hangzhou.log.aliyuncs.com

aliyun sls CreateConfig --project prod-webapp --body @app-json-log-config.json \
  --endpoint cn-hangzhou.log.aliyuncs.com
aliyun sls ApplyConfigToMachineGroup --project prod-webapp \
  --groupName prod-app-servers --configName app-json-log-config \
  --endpoint cn-hangzhou.log.aliyuncs.com

# Create indexes for nginx logs
aliyun sls CreateIndex --project prod-webapp --logstore nginx-access-log \
  --body @nginx-index.json --endpoint cn-hangzhou.log.aliyuncs.com
```

### 第三步：设置云监控告警

别等出事了再救火，先把告警配好：

```bash
# Create contact group
aliyun cms PutContactGroup \
  --ContactGroupName ops-team \
  --ContactNames '["primary-oncall"]'

# ECS alerts
for INSTANCE_ID in i-bp1xxxxxxxxx i-bp1yyyyyyyyy; do
  # CPU alert
  aliyun cms PutResourceMetricRule \
    --RuleId "cpu-high-${INSTANCE_ID}" \
    --RuleName "High CPU - ${INSTANCE_ID}" \
    --Namespace acs_ecs_dashboard \
    --MetricName CPUUtilization \
    --ContactGroups '["ops-team"]' \
    --Resources "[{\"instanceId\":\"${INSTANCE_ID}\"}]" \
    --Escalations.Warn.Statistics Average \
    --Escalations.Warn.ComparisonOperator GreaterThanOrEqualToThreshold \
    --Escalations.Warn.Threshold 80 \
    --Escalations.Warn.Times 5 \
    --Period 60

  # Disk alert
  aliyun cms PutResourceMetricRule \
    --RuleId "disk-high-${INSTANCE_ID}" \
    --RuleName "Low Disk - ${INSTANCE_ID}" \
    --Namespace acs_ecs_dashboard \
    --MetricName diskusage_utilization \
    --ContactGroups '["ops-team"]' \
    --Resources "[{\"instanceId\":\"${INSTANCE_ID}\"}]" \
    --Escalations.Warn.Statistics Maximum \
    --Escalations.Warn.ComparisonOperator GreaterThanOrEqualToThreshold \
    --Escalations.Warn.Threshold 80 \
    --Escalations.Warn.Times 3 \
    --Escalations.Critical.Statistics Maximum \
    --Escalations.Critical.ComparisonOperator GreaterThanOrEqualToThreshold \
    --Escalations.Critical.Threshold 90 \
    --Escalations.Critical.Times 1 \
    --Period 60
done

# RDS alerts
aliyun cms PutResourceMetricRule \
  --RuleId rds-connections-high \
  --RuleName "RDS Connection Pool Exhaustion" \
  --Namespace acs_rds_dashboard \
  --MetricName ConnectionUsage \
  --ContactGroups '["ops-team"]' \
  --Resources '[{"instanceId":"rm-bp1xxxxxxxxx"}]' \
  --Escalations.Warn.Statistics Average \
  --Escalations.Warn.ComparisonOperator GreaterThanOrEqualToThreshold \
  --Escalations.Warn.Threshold 80 \
  --Escalations.Warn.Times 3 \
  --Period 60

# SLB health check alert
aliyun cms PutResourceMetricRule \
  --RuleId slb-unhealthy-hosts \
  --RuleName "SLB Unhealthy Backend Servers" \
  --Namespace acs_slb_dashboard \
  --MetricName UnhealthyServerCount \
  --ContactGroups '["ops-team"]' \
  --Resources '[{"instanceId":"lb-bp1xxxxxxxxx"}]' \
  --Escalations.Critical.Statistics Maximum \
  --Escalations.Critical.ComparisonOperator GreaterThanThreshold \
  --Escalations.Critical.Threshold 0 \
  --Escalations.Critical.Times 2 \
  --Period 60
```

### 第四步：配置 SLS 大盘和告警

```bash
# Create the ops dashboard
aliyun sls CreateDashboard --project prod-webapp \
  --body @ops-dashboard.json \
  --endpoint cn-hangzhou.log.aliyuncs.com

# Create SLS alerts for error rate and slow queries
aliyun sls CreateAlert --project prod-webapp \
  --body @high-error-rate-alert.json \
  --endpoint cn-hangzhou.log.aliyuncs.com

aliyun sls CreateAlert --project prod-webapp \
  --body @slow-queries-alert.json \
  --endpoint cn-hangzhou.log.aliyuncs.com
```

### 第五步：验证整套系统是否运转正常

```bash
# 1. Check Logtail is collecting logs
aliyun sls GetLogs --project prod-webapp --logstore nginx-access-log \
  --from $(date -d '5 minutes ago' +%s) --to $(date +%s) \
  --query '*' --line 3 --endpoint cn-hangzhou.log.aliyuncs.com

# 2. Check CloudMonitor agent metrics
aliyun cms DescribeMetricLast \
  --Namespace acs_ecs_dashboard \
  --MetricName CPUUtilization \
  --Dimensions '[{"instanceId":"i-bp1xxxxxxxxx"}]'

# 3. Generate test traffic and verify dashboard
for i in $(seq 1 100); do
  curl -s -o /dev/null -w "%{http_code}\n" https://your-app.com/api/health
done

# 4. Trigger a test alert (temporarily lower threshold)
aliyun cms PutResourceMetricRule \
  --RuleId cpu-high-test \
  --RuleName "Test CPU Alert" \
  --Namespace acs_ecs_dashboard \
  --MetricName CPUUtilization \
  --ContactGroups '["ops-team"]' \
  --Resources '[{"instanceId":"i-bp1xxxxxxxxx"}]' \
  --Escalations.Warn.Statistics Average \
  --Escalations.Warn.ComparisonOperator GreaterThanOrEqualToThreshold \
  --Escalations.Warn.Threshold 1 \
  --Escalations.Warn.Times 1 \
  --Period 60

# Wait for the test alert, then delete it
# aliyun cms DeleteMetricRules --Id '["cpu-high-test"]'
```

### 完整架构

走完上面这五步，你的可观测性栈大概长这样：

```
                        ┌───────────────────────────────────────────┐
                        │              SLS Dashboard                │
                        │  QPS | Errors | Latency | Top Endpoints  │
                        └──────────┬───────────────┬────────────────┘
                                   │               │
                    ┌──────────────┴───┐    ┌──────┴──────────┐
                    │   SLS Alerts     │    │ CloudMonitor     │
                    │  error rate >1%  │    │  CPU > 80%       │
                    │  slow queries    │    │  disk > 80%      │
                    └──────┬───────────┘    │  RDS conn > 80%  │
                           │               │  SLB unhealthy   │
                    ┌──────┴───────┐       └──────┬───────────┘
                    │              │               │
              ┌─────▼─────┐  ┌────▼──────┐  ┌────▼──────────┐
              │ SLS       │  │ SLS       │  │ DingTalk /    │
              │ Logstore  │  │ Logstore  │  │ Email / SMS   │
              │ nginx     │  │ app       │  │ Webhook       │
              └─────▲─────┘  └────▲──────┘  └───────────────┘
                    │             │
              ┌─────┴─────────────┴──────────────────────┐
              │              Logtail Agent                │
              │    /var/log/nginx/    /var/log/app/       │
              ├──────────────────────────────────────────┤
              │              ARMS Agent                   │
              │    Traces → ARMS Console                 │
              ├──────────────────────────────────────────┤
              │           CloudMonitor Agent              │
              │    CPU, Memory, Disk → CloudMonitor       │
              └──────────────────────────────────────────┘
                         ECS Instance (App Tier)
```

## 成本

可观测性是要花钱的，而且这钱容易不知不觉就超了。下面是一个小型生产环境（2 台 ECS，中等流量）的真实成本估算：

![可观测性成本分解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/07-observability/07_cost_model.png)

| 组件 | 免费额度 | 典型月度成本 |
|---|---|---|
| SLS 写入 | 500 MB/day | 50-200 CNY (取决于日志量) |
| SLS 存储 | 包含在写入中 | 包含 |
| SLS 索引 | 包含 | 大约是存储成本的 2 倍 |
| CloudMonitor | 基础指标免费 | 内置指标 0 元；自定义指标 10-50 CNY |
| ARMS | 15 天免费试用 | 100-500 CNY (取决于 Trace 量) |

成本优化建议：

- **设置合理的保留周期。** 访问日志通常不需要保留超过 30 天。系统日志留 7 天就够了。慢查询日志可以留 90 天。把保留期从 90 天降到 30 天，存储成本能省 66%。
- **只索引你真正要查询的字段。** 每个 indexed field 都会让该字段的存储翻倍。如果你从来不在 SQL 里查 `http_user_agent`，就别给它建字段索引。
- **ARMS 使用采样。** 高流量应用里，只 trace 10% 的请求，而不是 100%。这样照样能抓到异常，但成本只有原来的 1/10。
- **存储前先聚合。** 对于那些只需要 5-minute 粒度的指标，直接在应用层聚合好再推数据，别把每个请求的 datapoint 都推上去。
## 核心要点

1. **上线前就把可观测性搞定，别等之后。** 事后补救的成本——重构日志格式、补充链路追踪、重做仪表盘——永远比从一开始就做好要高。把 Logtail、 CloudMonitor 插件和 ARMS 插件直接写进实例初始化脚本里，一步到位。

2. **三大支柱互补，不是冗余。** 指标告诉你出事了（仪表盘上错误率飙升），日志告诉你啥事（应用日志里数据库超时），追踪告诉你为啥（某条查询路径因为缺失索引跑了 3 秒）。想高效排查线上问题，这三个缺一不可。

3. **把 SLS 当成你的瑞士军刀。** 采集、搜索、 SQL 分析、仪表盘、告警，一个服务全搞定。熟练掌握查询语法，尤其是 `search | SQL` 模式，左边全文检索，右边分析计算。仪表盘只要配好五个核心面板（QPS、错误率、 P99 延迟、 Top 接口、状态码分布），就能覆盖 80% 的事故排查场景。

4. **针对症状告警，别针对原因。** "5 分钟内错误率 > 1%"比"CPU > 80%"有价值得多。一定要设持续阈值（比如连续 3-5 个数据点），别让瞬时抖动把你搞出告警疲劳。计划维护期间记得设静音期。

5. **从最小可行监控栈开始。** Logtail 收 nginx 和应用日志， CloudMonitor 看 ECS/RDS/SLB 自带指标，四条告警规则（错误率、 CPU、磁盘、 DB 连接），一个运维仪表盘。随着业务增长，再慢慢加 ARMS 追踪、自定义指标这些高级货。第一天就追求完美可观测性没必要，能在站点挂掉时把你叫醒才是正经事。

下一篇我们聊容器，上 ACK 和 SAE。到时候你会庆幸自己先搞好了可观测性，因为如果没有集中式日志，调试一个行为异常的 Kubernetes 集群，那滋味可真不好受。