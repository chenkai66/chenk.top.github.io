---
title: "阿里云全栈实战（八）：Serverless 与事件驱动"
date: 2026-05-05 09:00:00
tags:
  - Alibaba Cloud
  - Function Compute
  - EventBridge
  - Serverless
  - Cloud Computing
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 8
description: "Go serverless on Alibaba Cloud: Function Compute triggers, runtimes, cold starts, and pricing. EventBridge for event routing. API Gateway integration. Build an event-driven image processing pipeline."
disableNunjucks: true
translationKey: "aliyun-fullstack-8"
---
第一次看到函数计算的账单——1 万次请求仅需 0.03 元时，我彻底重构了原有架构。之前为了一个每小时大概 200 次请求的 API，我常年跑着一台 2 核 ECS，每月 490 元。同样的工作量搬到函数计算，每月不到 5 元。不是每天 5 块，是每月总共 5 块。成本差距如此悬殊，那个周末我就将所有无需常驻进程的业务，全部从 ECS 迁移至函数计算。

Serverless 并非没有服务器，而是将服务器运维完全交由平台托管。你写个函数，定义好触发条件，剩下的 provisioning、扩缩容、打补丁、下线，平台全包。代码跑多少毫秒收多少钱。没流量不收钱。五分钟内若涌入百万级请求，平台可瞬时扩容至百万实例。你甚至无需调整任何容量配置。

本文将介绍阿里云 Serverless 的两个核心组件：函数计算（FC，执行引擎）和事件总线（EventBridge，事件路由层）。读完后，我们将动手搭建一条完整的事件驱动图像处理流水线：当文件上传至 OSS 后，自动触发图片缩放、添加水印和生成缩略图。


## 什么时候该用 Serverless（什么时候不该用）

Serverless 并非银弹。它有明确的适用边界，提前厘清这一点，可避免数月后被迫回迁至 ECS。

### 适合 Serverless 的场景

| 使用场景 | 为什么合适 | 例子 |
|---|---|---|
| **事件处理** | 天然无状态，由外部事件触发 | 处理 OSS 上传，解析 SLS 日志 |
| **Webhooks** | 流量低频且不可预测 | GitHub webhook 处理，支付回调 |
| **定时任务** | 每天/每小时跑一次，其余时间空闲 | 日报生成，数据清理 |
| **突发流量 API 后端** |  burst 之间缩容到零能省钱 | 营销活动 API，季节性电商 |
| **数据转换** | 短生命周期，CPU 密集，易并行 | ETL 流水线，格式转换 |
| **聊天机器人后端** | 请求 - 响应模式，负载波动大 | 钉钉机器人，Slack 集成 |

### 不适合 Serverless 的场景

| 使用场景 | 为什么不行 | 更好的替代方案 |
|---|---|---|
| **长运行进程 (>15 分钟)** | FC 执行超时限制 15 分钟 | ECS, 容器服务 (ACK) |
| **GPU 任务** | FC 不支持 GPU | ECS GPU 实例，PAI-EAS |
| **低延迟要求 (<10ms)** | 冷启动增加 100ms-2s | 常驻进程 ECS |
| **WebSocket / 长连接** | 函数是请求 - 响应模式 | ECS, 带会话保持的 ALB |
| **高稳态吞吐** | 持续运行在 ECS 上更便宜 | 包年包月 ECS |
| **大内存状态** | 函数无状态，最大 3 GiB 内存 | ECS, Redis |

![函数计算 vs Serverless 应用引擎 vs ECS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_fc_sae_ecs.png)

比较价格之前，先把三种计算原语并排放在一起看。FC、SAE、ECS 在颗粒度 / 生命周期这条轴上各占一个位置，我见过的多数架构失误，都是把错的工作负载放到了错的原语上。

### 成本交叉点

大家最爱问：流量大到什么程度 ECS 比函数计算更便宜？

算笔账，假设 Python 函数 512 MiB 内存，平均执行 200ms：

| 每月请求数 | FC 成本 (CNY) | 等价 ECS 成本 (CNY) | 胜出者 |
|---|---|---|---|
| 10,000 | ~0.03 | ~490 (c7.large) | FC 便宜 16,000 倍 |
| 100,000 | ~0.30 | ~490 | FC 便宜 1,600 倍 |
| 1,000,000 | ~3.00 | ~490 | FC 便宜 160 倍 |
| 10,000,000 | ~30 | ~490 | FC 便宜 16 倍 |
| 100,000,000 | ~300 | ~490 | FC |
| 500,000,000 | ~1,500 | ~490 | ECS |
| 1,000,000,000 | ~3,000 | ~490 | ECS 便宜 6 倍 |

在这个配置下，交叉点大概在每月 2-3 亿次请求。低于这个数，函数计算完胜。高于这个数，专用 ECS 实例更便宜——但你也得自己处理扩缩容、打补丁和高可用。

> **我的经验法则：** 如果函数平均每秒持续请求少于 100 次（大约每月 2.5 亿次），Serverless 几乎肯定更便宜且运维更简单。如果持续超过这个量，评估一下 ECS 或容器方案。

## 函数计算 FC 基础

函数计算是阿里云的 Serverless 执行服务。对标 AWS Lambda。核心概念直接映射：

![函数计算架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_fc_architecture.png)

| FC 概念 | 是什么 | AWS 等价物 |
|---|---|---|
| **Service** | 相关函数的逻辑分组。共享 VPC 配置、日志配置、NAS 挂载和角色。 | (无直接等价；Lambda 用标签/前缀) |
| **Function** | 执行单元。你的代码 + 配置。 | Lambda function |
| **Trigger** | 调用函数的事件源。 | Event source mapping / trigger |
| **Layer** | 与函数代码分开打包的共享依赖。 | Lambda layer |
| **Custom domain** | 将自有域名映射到 HTTP 触发函数。 | API Gateway custom domain |
| **Alias / Version** | 带有流量切换别名 的不可变快照。 | Lambda versions and aliases |

### 执行模型

请求进来时，FC 这么处理：

1. **检查温热实例。** 如果最近调用过的函数实例还活着（保温约 5-15 分钟），请求直接路由过去。无冷启动。
2. **冷启动（如需）。** 下载代码包，初始化运行时，运行初始化代码（模块级导入，DB 连接池）。耗时 100ms 到 2s，取决于运行时和包大小。
3. **执行 handler。** 带着事件 payload 跑你的函数。从这一刻开始计费。
4. **返回响应。** 实例保持温热状态等待后续请求。

几个关键限制得知道：

| 限制 | 值 |
|---|---|
| 最大执行时间 | 15 分钟 (异步调用 86,400s) |
| 最大内存 | 32 GiB |
| 最大代码包 (直接上传) | 100 MiB (压缩后) |
| 最大代码包 (OSS 引用) | 500 MiB (未压缩) |
| 每个函数最大 Layers | 5 |
| 最大并发实例 (默认) | 每个函数 300 |
| 最大 Payload (同步) | 32 MiB |
| 最大 Payload (异步) | 128 KiB |
| 临时磁盘 (`/tmp`) | 10 GiB |

![FC 请求生命周期：热路径 vs 冷启动](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_lifecycle.png)

热路径与冷路径之间的差距，就是 Serverless 性能故事的全部。热路径只要几毫秒；冷路径在你的 handler 真正开始执行之前，要先跑完四个额外步骤。

### 运行时

FC 支持这些托管运行时：

| 运行时 | 版本 | 冷启动 (典型) | 备注 |
|---|---|---|---|
| **Python** | 3.9, 3.10 | 200-400ms | 最流行。生态好。 |
| **Node.js** | 14, 16, 18 | 150-300ms | 冷启动快。适合 API 后端。 |
| **Java** | 8, 11, 17 | 1-3s | 冷启动最慢。用 GraalVM 或 SnapStart。 |
| **Go** | 1.x (自定义运行时) | 50-150ms | 最快。单二进制，无运行时初始化。 |
| **PHP** | 7.2 | 200-400ms | 遗留支持。 |
| **C#** | .NET Core 6 | 500ms-1s | 中等。 |
| **Custom Runtime** | 任意 (Docker) | 可变 | 完全控制。适合 ML 模型，系统库。 |
| **Custom Container** | 任意 Docker 镜像 | 1-10s | 包最大。需要拉取镜像。 |

> **如果你在乎冷启动：** Go 是最快的运行时，优势明显，因为函数编译成单个静态二进制文件，无需运行时初始化。Python 是实际上的 sweet spot——冷启动适中，库生态巨大，写得顺手。除非你用 GraalVM 原生编译，否则对延迟敏感的函数别选 Java。

## 写你的第一个函数

咱们从零构建一个函数。用 Python 3.10 和 Serverless Devs (`s`) CLI，这是 FC 开发的标准工具链。

### 安装 Serverless Devs

```bash
# Install the s CLI
npm install -g @serverless-devs/s

# Configure credentials
s config add \
  --AccountID <your-account-id> \
  --AccessKeyID <your-ak> \
  --AccessKeySecret <your-sk> \
  -a default
```

### 项目结构

```
image-processor/
  ├── s.yaml              # Serverless Devs config
  ├── code/
  │   ├── index.py        # Function handler
  │   └── requirements.txt
  └── README.md
```

### Handler

这是最简单的函数——一个 HTTP 触发的 hello world：

```python
# code/index.py
import json
import logging

logger = logging.getLogger()

def handler(event, context):
    """
    FC handler for HTTP trigger.
    
    Args:
        event: The HTTP request body (bytes).
        context: FC context object with request ID, credentials, 
                 function name, memory limit, etc.
    
    Returns:
        dict with statusCode, headers, and body for HTTP response.
    """
    logger.info(f"Request ID: {context.request_id}")
    logger.info(f"Function: {context.function.name}")
    logger.info(f"Memory limit: {context.function.memory_size} MiB")
    
    # Parse the event (HTTP trigger sends the request body)
    try:
        body = json.loads(event)
    except (json.JSONDecodeError, TypeError):
        body = {}
    
    name = body.get("name", "World")
    
    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "message": f"Hello, {name}!",
            "request_id": context.request_id
        })
    }
    
    return response
```

拆解一下 handler 签名：

- **`event`**：输入 payload。HTTP 触发器收到的是 HTTP 请求 body（bytes）。OSS 触发器收到的是描述上传文件的 JSON 对象。形状取决于触发器类型。
- **`context`**：丰富的对象，包含请求 ID（用于日志关联）、函数元数据（名称、内存、超时）、服务元数据、凭证（如果函数需要调用其他阿里云 API，这里是临时 STS 凭证），以及区域信息。
- **返回值**：HTTP 触发器返回包含 `statusCode`、`headers` 和 `body` 的 dict。非 HTTP 触发器可以返回任何 JSON 可序列化的值。

### Serverless Devs 配置

```yaml
# s.yaml
edition: 3.0.0
name: image-processor
access: default

resources:
  hello:
    component: fc3
    props:
      region: cn-beijing
      functionName: 入门示例
      description: "Simple hello world function"
      runtime: python3.10
      handler: index.handler
      memorySize: 256
      timeout: 30
      code: ./code
      triggers:
        - triggerName: http-trigger
          triggerType: http
          triggerConfig:
            authType: anonymous
            methods:
              - GET
              - POST
```

### 部署与测试

```bash
# Deploy the function
cd image-processor
s deploy

# Invoke locally (for development)
s local invoke -e '{"name": "Alibaba Cloud"}'

# Invoke remotely (the deployed function)
s invoke -e '{"name": "Alibaba Cloud"}'

# Get the HTTP trigger URL
s info
# Output includes: url: https://入门示例-xxxx.cn-beijing.fcapp.run
```

用 curl 测一下：

```bash
# Test the deployed function
curl -X POST \
  https://入门示例-xxxx.cn-beijing.fcapp.run \
  -H "Content-Type: application/json" \
  -d '{"name": "Serverless"}'

# Response:
# {"message": "Hello, Serverless!", "request_id": "1-6789abcd-..."}
```

### 查看日志

FC 把所有 `print()` 和 `logging` 输出送到简单日志服务 SLS。你可以在 FC 控制台看日志，或者通过 CLI 查询：

```bash
# View recent function logs
s logs --tail

# View logs for a specific request
s logs --request-id "1-6789abcd-..."
```
## 触发器：什么唤醒了你的函数

触发器才是让 Serverless 变成事件驱动架构的核心，而不只是个“便宜托管”。每种触发器类型都把函数连到了不同的事件源上。下面我把每种触发器类型都列出来，配上配置示例。

![函数计算触发器类型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_trigger_types.png)

### HTTP 触发器

最简单的触发器。你的函数直接拥有一个 HTTP 端点。

```bash
# Create via CLI
aliyun fc CreateTrigger \
  --functionName my-api \
  --triggerName http-trigger \
  --triggerType http \
  --triggerConfig '{
    "authType": "anonymous",
    "methods": ["GET", "POST", "PUT", "DELETE"]
  }'
```

函数会收到完整的 HTTP 请求，然后返回 HTTP 响应。这是在 Function Compute 上构建 REST API 的基础。

### OSS 触发器

当 OSS Bucket 里的对象被创建、修改或删除时触发。这是大多数 Serverless 数据流水线的 backbone。

```bash
# Trigger on file uploads to a specific prefix
aliyun fc CreateTrigger \
  --functionName image-processor \
  --triggerName oss-upload \
  --triggerType oss \
  --triggerConfig '{
    "bucketName": "my-upload-bucket",
    "events": ["oss:ObjectCreated:PutObject", "oss:ObjectCreated:PostObject"],
    "filter": {
      "key": {
        "prefix": "uploads/",
        "suffix": ".jpg"
      }
    }
  }'
```

事件 payload 长这样：

```json
{
  "events": [
    {
      "eventName": "ObjectCreated:PutObject",
      "eventSource": "acs:oss",
      "eventTime": "2026-05-22T08:30:00.000Z",
      "region": "cn-beijing",
      "oss": {
        "bucket": {
          "name": "my-upload-bucket"
        },
        "object": {
          "key": "uploads/photo-001.jpg",
          "size": 2048576
        }
      }
    }
  ]
}
```

OSS 作为我们流水线的存储层，我在 [第 4 部分](/zh/aliyun-fullstack/04-oss-storage/) 里详细讲过。

### 定时触发器 (cron)

按计划运行函数。支持 cron 表达式或 rate 表达式。

```bash
# Run every day at 02:00 UTC+8
aliyun fc CreateTrigger \
  --functionName daily-report \
  --triggerName daily-cron \
  --triggerType timer \
  --triggerConfig '{
    "cronExpression": "0 0 2 * * *",
    "enable": true,
    "payload": "{\"report_type\": \"daily_summary\"}"
  }'

# Run every 5 minutes
aliyun fc CreateTrigger \
  --functionName health-checker \
  --triggerName every-5min \
  --triggerType timer \
  --triggerConfig '{
    "cronExpression": "@every 5m",
    "enable": true
  }'
```

定时触发器就是 Serverless 版的 crontab。再也不用为了每天跑一次脚本而养着一台 ECS 实例了。

### SLS 触发器 (日志事件)

近实时处理 Simple Log Service 的日志条目。

```bash
aliyun fc CreateTrigger \
  --functionName log-analyzer \
  --triggerName sls-trigger \
  --triggerType log \
  --triggerConfig '{
    "sourceConfig": {
      "logstore": "access-log"
    },
    "jobConfig": {
      "maxRetryTime": 3,
      "triggerInterval": 60
    },
    "logConfig": {
      "project": "my-log-project",
      "logstore": "fc-error-log"
    },
    "functionParameter": {
      "alert_threshold": 100
    },
    "enable": true
  }'
```

### MNS 触发器 (消息队列)

消费 Message Service 队列或主题中的消息。

```bash
aliyun fc CreateTrigger \
  --functionName order-processor \
  --triggerName mns-queue \
  --triggerType mns_topic \
  --triggerConfig '{
    "topicName": "order-events",
    "region": "cn-beijing",
    "filterTag": "new-order"
  }'
```

### EventBridge 触发器

最灵活的选项。后面我会深入讲 EventBridge，先看触发器配置：

![EventBridge event routing pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_event_pipeline.png)

```bash
aliyun fc CreateTrigger \
  --functionName event-handler \
  --triggerName eb-trigger \
  --triggerType eventbridge \
  --triggerConfig '{
    "triggerEnable": true,
    "asyncInvocationType": false,
    "eventRuleFilterPattern": "{\"source\":[\"acs.oss\"],\"type\":[\"oss:BucketCreated\"]}"
  }'
```

### 触发器对比

| 触发器 | 延迟 | 使用场景 | 最大批量 |
|---|---|---|---|
| HTTP | 同步，<100ms | API 端点 | 1 请求 |
| OSS | 异步，1-5s | 文件处理 | 1 事件 |
| Timer | N/A | 定时任务 | N/A |
| SLS | 1-60s | 日志分析 | 可配置 |
| MNS | 1-5s | 消息处理 | 1 消息 |
| EventBridge | 1-5s | 复杂事件路由 | 1 事件 |

## 冷启动与性能

冷启动是 Serverless 被讨论最多的缺点。要想在生产环境用好，必须搞清楚成因以及怎么缓解。

![函数计算冷启动分析](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_cold_start.png)

### 冷启动的成因

当 FC 需要创建一个新的函数实例时，就会发生冷启动。流程如下：

1. **下载代码**（小包 10-100ms，大包 1-5s）
2. **启动运行时**（取决于语言，50-500ms）
3. **运行初始化代码**（模块级 import、连接 setup）
4. **执行 handler**

前 3 步是冷启动的开销。第 4 步是 warm invocation 时的计费项。冷启动时，这四步都要计费。

### 测量冷启动

在函数里加些计时代码，就能测出差别：

```python
# code/index.py
import time
import json

# Module-level code runs during cold start
INIT_TIME = time.time()
print(f"Cold start initialization at {INIT_TIME}")

# Expensive imports happen here
import numpy as np
from PIL import Image

INIT_DURATION = time.time() - INIT_TIME
print(f"Init took {INIT_DURATION:.3f}s")

def handler(event, context):
    start = time.time()
    
    # Your business logic here
    result = {"cold_start_init_ms": round(INIT_DURATION * 1000)}
    
    duration = time.time() - start
    result["handler_ms"] = round(duration * 1000)
    
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }
```

### 不同运行时的冷启动基准

我在 `cn-beijing` 区域的 FC 上测了一个最小化函数（hello world + 一个 import）的冷启动数据：

| 运行时 | 冷启动 (p50) | 冷启动 (p99) | 温调用 (p50) |
|---|---|---|---|
| Go 1.x | 60ms | 150ms | 1ms |
| Node.js 18 | 180ms | 400ms | 3ms |
| Python 3.10 | 250ms | 600ms | 5ms |
| C# .NET 6 | 400ms | 900ms | 8ms |
| Java 17 | 1,200ms | 3,000ms | 5ms |
| Custom Container | 2,000ms | 8,000ms | 5ms |

Java 的冷启动耗时主要卡在 JVM 启动上。如果非要用 Java，可以用 GraalVM native image 编译，能把冷启动压到 200-400ms，但构建过程比较麻烦。

### 缓解策略

![FC 并发：典型一天内 on-demand 与预留对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_concurrency.png)

这就是典型一天里的并发曲线。On-demand 自动扩缩跟着请求速率走，但每次冲峰都要付出冷启动代价；预留实例用一条最低基线让实例常热，代价是夜里空跑也要计费。下面三种缓解策略，本质上都是在这两个失效模式之间做权衡。

**1. 预留实例 (Provisioned concurrency)**

始终保持一定数量的实例处于温热状态。你得为闲置时间付费，但冷启动彻底没了。

```bash
# Keep 5 instances always warm
aliyun fc PutProvisionConfig \
  --functionName image-processor \
  --qualifier LATEST \
  --target 5
```

成本方面：你得一直为 5 个实例付费（大约 0.000110592 CNY/GiB-second）。对于 512 MiB 的函数，大概是 0.055 CNY/s，5 个实例一个月约 143 CNY。对延迟敏感的 API 来说这钱花得值；如果是批处理任务，那就大材小用了。

**2. 定时预热**

如果流量可预测（比如每天早上 9 点有高峰），可以用定时触发器在高峰前几分钟调用一下函数：

```python
# pre-warm function (called by timer trigger)
def handler(event, context):
    """
    Do nothing — the point is to create warm instances.
    """
    return {"statusCode": 200, "body": "warm"}
```

```yaml
# Timer trigger: warm up at 8:55 AM every weekday
triggers:
  - triggerName: pre-warm
    triggerType: timer
    triggerConfig:
      cronExpression: "0 55 8 ? * MON-FRI"
      enable: true
```

**3. 减小包大小**

部署包每多一个字节，冷启动时间就多一点。代码下载阶段往往是耗时大户。

```bash
# Bad: 150 MiB package with all of scipy/numpy/pandas
pip install -r requirements.txt -t ./code/

# Good: only install what you need
pip install Pillow requests -t ./code/ --no-cache-dir

# Better: use layers for shared dependencies
# (covered in the next section)
```

**4. 优化初始化代码**

尽可能把耗时操作移出模块作用域：

```python
# Bad: connects to DB on every cold start, even if the request
# doesn't need it
import pymysql
conn = pymysql.connect(host="...", db="...")  # Runs on cold start

def handler(event, context):
    cursor = conn.cursor()
    ...

# Good: lazy initialization
_conn = None

def get_connection():
    global _conn
    if _conn is None or not _conn.open:
        _conn = pymysql.connect(host="...", db="...")
    return _conn

def handler(event, context):
    conn = get_connection()
    cursor = conn.cursor()
    ...
```

**5. 选对运行时**

如果冷启动是你的首要顾虑，优先级顺序是：Go > Node.js > Python > C# > Java。如果开发效率比冷启动更重要，Python 是实际上的赢家。
## 层与自定义运行时

### 层：共享依赖

![函数层架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_layer_architecture.png)

Layer 本质上就是个 zip 压缩包，里面装着依赖库、自定义运行时或者其他依赖项。Layer 是有版本控制的，而且可以在多个函数之间共享。它主要解决了两个问题：

1. **减小包大小。** 把那些庞大的依赖库（比如 numpy、Pillow 等）挪到 Layer 里。你的函数代码包就能保持轻量 = 部署更快，冷启动也更快。
2. **依赖共享。** 如果有十个函数都要用 Pillow，它们可以引用同一个 Layer，而不是把 Pillow 打包十次。

#### 创建 Layer

```bash
# Create a directory matching the runtime's expected path
mkdir -p layer/python/lib/python3.10/site-packages

# Install dependencies into the layer
pip install Pillow numpy requests \
  -t layer/python/lib/python3.10/site-packages \
  --no-cache-dir

# Zip it
cd layer && zip -r ../image-processing-layer.zip . && cd ..

# Create the layer in FC
aliyun fc CreateLayerVersion \
  --layerName image-processing-deps \
  --code '{"zipFile":"'$(base64 -w0 image-processing-layer.zip)'"}'  \
  --compatibleRuntime '["python3.10"]' \
  --description "Pillow + numpy + requests for image processing"
```

#### 在函数中使用 Layer

```yaml
# s.yaml
resources:
  image-processor:
    component: fc3
    props:
      region: cn-beijing
      functionName: image-processor
      runtime: python3.10
      handler: index.handler
      memorySize: 1024
      timeout: 120
      code: ./code
      layers:
        - acs:fc:cn-beijing:123456789:layers/image-processing-deps/versions/1
```

这样一来，你的函数代码里就只剩业务逻辑，沉重的依赖都交给 Layer 提供。部署时间能从几分钟缩短到几秒钟。

### 自定义运行时

当托管运行时满足不了需求时——比如你需要特定的系统库、编译好的二进制文件，或者一个 ML 模型——这时候就得用 Custom Runtime 或者 Custom Container。

#### 自定义运行时（HTTP 服务器模式）

你的代码以 HTTP 服务器的形式运行。FC 会启动你的二进制文件或脚本，然后把请求发给 `localhost:9000`。你可以用任何语言或框架。

```dockerfile
# Dockerfile for a Rust-based function
FROM rust:1.78-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/my-function /app/bootstrap
EXPOSE 9000
CMD ["/app/bootstrap"]
```

```yaml
# s.yaml for custom container
resources:
  rust-function:
    component: fc3
    props:
      region: cn-beijing
      functionName: rust-processor
      runtime: custom-container
      handler: not-used
      memorySize: 512
      timeout: 60
      customContainerConfig:
        image: registry.cn-beijing.aliyuncs.com/my-repo/rust-function:latest
        port: 9000
```

#### 什么时候该用自定义容器

- **ML 推理：** 把模型权重（通过 NAS 挂载最大支持 10 GiB）和推理框架打包在一起。
- **系统依赖：** 比如视频处理需要的 FFmpeg，图片处理的 ImageMagick，生成 PDF 用的 wkhtmltopdf。
- **不支持的语言：** Rust、C++，或者任何能编译成二进制的语言。

不过代价是冷启动时间。自定义容器在冷启动时需要拉取镜像，这可能会花掉 2-10 秒。对于延迟敏感的业务，记得用上预留实例（provisioned concurrency）。

## EventBridge

EventBridge 充当的是事件路由层的角色。如果说 Function Compute 是运行代码的引擎，那 EventBridge 就是总机，决定哪个事件触发哪段代码。这相当于 AWS 的 Amazon EventBridge（以前叫 CloudWatch Events）。

### 核心概念

| 概念 | 是什么 | 示例 |
|---|---|---|
| **事件总线 (Event bus)** | 接收事件的通道。默认总线接收所有阿里云服务事件。自定义总线用于你自己的事件。 | `default`, `my-app-bus` |
| **事件源 (Event source)** | 事件的来源。内置源（OSS、ECS、RDS）或自定义源（你的应用）。 | `acs.oss`, `my.app` |
| **事件规则 (Event rule)** | 过滤器 + 目标映射。“当事件匹配这个模式时，把它发给那个目标。” | "OSS PutObject 在 uploads/ -> FC 函数" |
| **事件目标 (Event target)** | 匹配后的事件投递到哪里。FC、MNS、SLS、HTTP 端点、另一个总线。 | `acs:fc:cn-beijing:123456:functions/process-image` |

### 事件格式

EventBridge 里的每个事件都遵循 CloudEvents 1.0 规范：

```json
{
  "id": "a]b-1234-efgh-5678",
  "source": "acs.oss",
  "type": "oss:ObjectCreated:PutObject",
  "specversion": "1.0",
  "datacontenttype": "application/json",
  "time": "2026-05-22T08:30:00Z",
  "subject": "acs:oss:cn-beijing:123456789:my-bucket",
  "data": {
    "region": "cn-beijing",
    "bucket": {
      "name": "my-upload-bucket"
    },
    "object": {
      "key": "uploads/photo-001.jpg",
      "size": 2048576,
      "eTag": "abc123..."
    }
  }
}
```

这种标准化格式的好处是，你的事件处理代码不需要关心每个阿里云服务的具体事件格式——它们都遵循同一个结构。

### 内置事件源

EventBridge 会自动接收来自这些阿里云服务的事件（部分列表）：

| 服务 | 事件类型 | 示例触发场景 |
|---|---|---|
| **OSS** | 对象创建、删除、访问 | 文件上传处理 |
| **ECS** | 实例状态变更、磁盘事件 | 故障自动修复 |
| **RDS** | 实例创建、故障转移、高 CPU | 数据库告警 |
| **容器服务** | Pod 事件、部署变更 | 部署追踪 |
| **SLS** | 触发告警 | 事件响应 |
| **ActionTrail** | API 调用（审计日志） | 安全监控 |
| **云监控** | 告警状态变更 | 自定义告警流水线 |
| **MNS** | 消息发布 | 消息路由 |

### 创建事件规则

事件规则是 EventBridge 的核心。它们利用 JSON 模式匹配来过滤事件，并把匹配到的事件路由到目标。

```bash
# Create a custom event bus
aliyun eventbridge CreateEventBus \
  --EventBusName my-app-bus \
  --Description "Custom events for my application"

# Create a rule: route OSS uploads to a Function Compute function
aliyun eventbridge CreateRule \
  --EventBusName default \
  --RuleName oss-upload-to-fc \
  --Description "Route OSS uploads to image processor" \
  --FilterPattern '{
    "source": ["acs.oss"],
    "type": ["oss:ObjectCreated:PutObject", "oss:ObjectCreated:PostObject"],
    "subject": ["acs:oss:cn-beijing:*:my-upload-bucket"]
  }' \
  --Targets '[{
    "Id": "fc-image-processor",
    "Type": "acs.fc.function",
    "Endpoint": "acs:fc:cn-beijing:123456789:functions/image-processor",
    "ParamList": [
      {
        "ResourceKey": "Body",
        "Form": "ORIGINAL"
      }
    ]
  }]'
```

### 事件模式匹配

过滤模式支持多种匹配运算符：

```json
{
  "source": ["acs.oss"],
  "type": [{"prefix": "oss:ObjectCreated"}],
  "data": {
    "object": {
      "size": [{"numeric": [">", 1048576]}],
      "key": [{"prefix": "uploads/"}, {"suffix": ".jpg"}]
    }
  }
}
```

这个模式能匹配：来自 OSS 的事件，其中对象被创建，对象大小超过 1 MiB，key 以 `uploads/` 开头且以 `.jpg` 结尾。

可用运算符如下：

| 运算符 | 语法 | 匹配条件 |
|---|---|---|
| 精确匹配 | `["value1", "value2"]` | 字段等于列出的任意值 |
| 前缀 | `[{"prefix": "abc"}]` | 字段以 "abc" 开头 |
| 后缀 | `[{"suffix": ".jpg"}]` | 字段以 ".jpg" 结尾 |
| 数值 | `[{"numeric": [">", 100]}]` | 字段 > 100 |
| IP 地址 | `[{"cidr": "10.0.0.0/8"}]` | IP 在 CIDR 范围内 |
| 存在性 | `[{"exists": true}]` | 字段存在 |
| 非 | `[{"anything-but": "error"}]` | 字段不是 "error" |

### 事件转换

在把事件投递给目标之前，你可以转换事件负载。当你的目标函数期望的格式与原始 CloudEvent 不同时，这招很好用：

```bash
# Create a rule with event transformation
aliyun eventbridge CreateRule \
  --EventBusName default \
  --RuleName transform-example \
  --FilterPattern '{"source": ["acs.oss"]}' \
  --Targets '[{
    "Id": "fc-target",
    "Type": "acs.fc.function",
    "Endpoint": "acs:fc:cn-beijing:123456789:functions/my-func",
    "ParamList": [
      {
        "ResourceKey": "Body",
        "Form": "TEMPLATE",
        "Template": "{\"bucket\":\"$.data.bucket.name\",\"key\":\"$.data.object.key\",\"size\":$.data.object.size}",
        "Value": "{\"bucket\":\"$.data.bucket.name\",\"key\":\"$.data.object.key\",\"size\":\"$.data.object.size\"}"
      }
    ]
  }]'
```

这样一来，你的函数接收到的不再是完整的 CloudEvent，而是：

```json
{
  "bucket": "my-upload-bucket",
  "key": "uploads/photo-001.jpg",
  "size": 2048576
}
```

### 自定义事件

你的应用也可以发布事件到 EventBridge，供其他服务消费：

```python
from alibabacloud_eventbridge.client import Client
from alibabacloud_eventbridge.models import CloudEvent

client = Client(
    access_key_id="<ak>",
    access_key_secret="<sk>",
    endpoint="cn-beijing.eventbridge.aliyuncs.com"
)

event = CloudEvent(
    source="my.application",
    type="order.created",
    data='{"order_id": "ORD-12345", "amount": 99.99, "currency": "CNY"}',
    subject="acs:my-app:cn-beijing:123456789:orders/ORD-12345"
)

client.put_events(event_bus_name="my-app-bus", event_list=[event])
```

这样你就能构建完全事件驱动的架构，不同的微服务通过 EventBridge 通信，而不是直接调用 API——这是解耦的最佳实践。

### EventBridge 对比直接触发器

你可能会想：FC 本来就有直接的 OSS 触发器，为什么还要用 EventBridge？

| 特性 | 直接 FC 触发器 | EventBridge |
|---|---|---|
| 设置复杂度 | 简单 | 配置较多 |
| 过滤能力 | 基础（前缀/后缀） | 丰富的模式匹配 |
| 扇出 (Fan-out) | 每个触发器对应 1 个函数 | 每条规则对应多个目标 |
| 死信队列 | 无 | 有 |
| 重试策略 | 有限 | 可配置 |
| 跨服务路由 | 无（仅限 OSS 到 FC） | 任意源到任意目标 |
| 事件转换 | 无 | 有 |
| 审计追踪 | 有限 | 完整事件历史 |

**我的建议：** 对于简单的单函数场景（比如“调整每张上传图片的大小”），直接用直接触发器。当你需要复杂路由、多个目标或者跨服务事件流时，再上 EventBridge。
## API Gateway + Function Compute

要想构建生产级的 REST API，光靠 FC 自带的 HTTP Trigger 还不够。得把 API Gateway 架在函数前面，因为它能提供鉴权、限流、请求校验以及 API 版本管理这些关键能力，而这些都是 FC 原生触发器欠缺的。

![API Gateway 与函数计算集成](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_api_gateway.png)

### Creating an API backed by Function Compute

下面直接用 CLI 命令演示如何创建一个后端挂载 Function Compute 的 API。

```bash
# Step 1: Create an API Group
aliyun cloudapi CreateApiGroup \
  --RegionId cn-beijing \
  --GroupName "my-serverless-api" \
  --Description "Serverless API backed by FC"

# Step 2: Create an API definition
aliyun cloudapi CreateApi \
  --RegionId cn-beijing \
  --GroupId <group-id> \
  --ApiName "GetUser" \
  --Visibility PUBLIC \
  --AuthType APP \
  --RequestConfig '{
    "RequestProtocol": "HTTPS",
    "RequestHttpMethod": "GET",
    "RequestPath": "/users/[userId]",
    "RequestMode": "MAPPING"
  }' \
  --ServiceConfig '{
    "ServiceProtocol": "FunctionCompute",
    "FunctionComputeConfig": {
      "RegionId": "cn-beijing",
      "FunctionName": "get-user",
      "Qualifier": "LATEST",
      "RoleArn": "acs:ram::123456789:role/api-gateway-fc"
    }
  }' \
  --ResultType JSON \
  --ResultSample '{"user_id": "123", "name": "Test User"}'
```

### Custom domains

想把自有域名映射到 API 上，操作如下：

```bash
# Bind a custom domain to your API Group
aliyun cloudapi SetDomain \
  --RegionId cn-beijing \
  --GroupId <group-id> \
  --DomainName api.example.com \
  --CertificateName my-cert \
  --CertificateBody "$(cat cert.pem)" \
  --CertificatePrivateKey "$(cat key.pem)"
```

### Rate limiting and throttling

限流策略也是通过 API Gateway 配置的，比如设定一个默认的 throttling policy：

```bash
# Create a throttling policy
aliyun cloudapi CreateTrafficControl \
  --RegionId cn-beijing \
  --TrafficControlName "default-limit" \
  --ApiDefault 1000 \
  --UserDefault 100 \
  --AppDefault 200 \
  --TrafficControlUnit MINUTE \
  --Description "1000 req/min total, 100 per user, 200 per app"
```

### CORS configuration

如果前端浏览器要调用你的 API，CORS 头必不可少。这部分逻辑直接在函数代码里处理就行：

```python
def handler(event, context):
    # Handle CORS preflight
    http_method = event.get("httpMethod", "GET")
    
    cors_headers = {
        "Access-Control-Allow-Origin": "https://example.com",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400"
    }
    
    if http_method == "OPTIONS":
        return {
            "statusCode": 204,
            "headers": cors_headers,
            "body": ""
        }
    
    # Normal request handling
    result = {"message": "Hello from API"}
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            **cors_headers
        },
        "body": json.dumps(result)
    }
```
## 解决方案：事件驱动的图片处理流水线

咱们把前面聊的都串起来，搞个能直接上生产环境的系统。架构大概长这样：

```
  User uploads image to OSS
           │
           ▼
    ┌──────────────┐
    │     OSS      │  (source bucket: upload-images)
    │  PutObject   │
    └──────┬───────┘
           │  CloudEvent
           ▼
    ┌──────────────┐
    │  EventBridge │  (filter: *.jpg, *.png, > 10KB)
    │  Event Rule  │
    └──────┬───────┘
           │  Invoke
           ▼
    ┌──────────────┐
    │   Function   │  (image-processor)
    │   Compute    │  - Resize to 1200px, 600px, 150px
    │              │  - Add watermark
    │              │  - Generate WebP variants
    └──────┬───────┘
           │  PutObject
           ▼
    ┌──────────────┐
    │     OSS      │  (target bucket: processed-images)
    │   /resized/  │
    │   /thumbs/   │
    │   /webp/     │
    └──────────────┘
```

![事件驱动的图片处理流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_image_pipeline.png)

下面这张图是整条流水线的全貌。源 bucket 上一次 PutObject，会在目标 bucket 里扇出 6 个衍生文件——三种尺寸 × 两种格式——整条链路 2-3 秒走完，全程没有一台服务器需要你运维。

### 第一步：函数代码

核心逻辑都在这个 Python 文件里。注意看 `get_oss_client`，这里直接用 FC 提供的临时 STS 凭证，不用硬编码 AccessKey，安全得多。水印逻辑我也写进去了，自动根据图片大小调整字体，避免小图上水印太大。

```python
# code/index.py
import json
import logging
import os
import io
from urllib.parse import unquote

import oss2
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET", "upload-images")
TARGET_BUCKET = os.environ.get("TARGET_BUCKET", "processed-images")
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "https://oss-cn-beijing-internal.aliyuncs.com")
WATERMARK_TEXT = os.environ.get("WATERMARK_TEXT", "chenk.top")

# Resize configurations
SIZES = {
    "large": 1200,
    "medium": 600,
    "thumb": 150
}

def get_oss_client(context, bucket_name):
    """Create OSS client using FC's temporary STS credentials."""
    creds = context.credentials
    auth = oss2.StsAuth(
        creds.access_key_id,
        creds.access_key_secret,
        creds.security_token
    )
    return oss2.Bucket(auth, OSS_ENDPOINT, bucket_name)

def add_watermark(image, text):
    """Add a semi-transparent text watermark to the bottom-right corner."""
    draw = ImageDraw.Draw(image)
    
    # Use a size proportional to the image
    font_size = max(20, image.width // 30)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    
    # Calculate position (bottom-right with padding)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding = 20
    x = image.width - text_width - padding
    y = image.height - text_height - padding
    
    # Draw with semi-transparent white
    draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)
    
    return image

def resize_image(image, max_width):
    """Resize image maintaining aspect ratio."""
    if image.width <= max_width:
        return image.copy()
    
    ratio = max_width / image.width
    new_height = int(image.height * ratio)
    return image.resize((max_width, new_height), Image.LANCZOS)

def process_image(source_client, target_client, object_key):
    """Download, process, and upload image variants."""
    logger.info(f"Processing: {object_key}")
    
    # Download the original image
    result = source_client.get_object(object_key)
    image_data = result.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGBA for watermark support
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Extract the filename without extension
    base_name = os.path.splitext(os.path.basename(object_key))[0]
    results = []
    
    for size_name, max_width in SIZES.items():
        # Resize
        resized = resize_image(image, max_width)
        
        # Add watermark (skip for thumbnails)
        if size_name != "thumb":
            resized = add_watermark(resized, WATERMARK_TEXT)
        
        # Convert to RGB for JPEG output
        rgb_image = resized.convert("RGB")
        
        # Save as JPEG
        jpeg_buffer = io.BytesIO()
        rgb_image.save(jpeg_buffer, format="JPEG", quality=85, optimize=True)
        jpeg_buffer.seek(0)
        
        jpeg_key = f"resized/{size_name}/{base_name}.jpg"
        target_client.put_object(jpeg_key, jpeg_buffer.getvalue())
        results.append({"key": jpeg_key, "format": "jpeg", "size": size_name})
        logger.info(f"  Uploaded: {jpeg_key}")
        
        # Save as WebP
        webp_buffer = io.BytesIO()
        resized.save(webp_buffer, format="WebP", quality=80)
        webp_buffer.seek(0)
        
        webp_key = f"webp/{size_name}/{base_name}.webp"
        target_client.put_object(webp_key, webp_buffer.getvalue())
        results.append({"key": webp_key, "format": "webp", "size": size_name})
        logger.info(f"  Uploaded: {webp_key}")
    
    return results

def handler(event, context):
    """
    EventBridge trigger handler.
    Receives CloudEvent when an image is uploaded to OSS.
    """
    logger.info(f"Event received: {event}")
    
    # Parse the CloudEvent from EventBridge
    evt = json.loads(event)
    
    # Extract bucket and object key from the event
    data = evt.get("data", {})
    bucket_name = data.get("bucket", {}).get("name", SOURCE_BUCKET)
    object_key = unquote(data.get("object", {}).get("key", ""))
    
    if not object_key:
        logger.error("No object key in event")
        return {"statusCode": 400, "body": "Missing object key"}
    
    # Validate file extension
    ext = os.path.splitext(object_key)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        logger.info(f"Skipping non-image file: {object_key}")
        return {"statusCode": 200, "body": "Skipped: not an image"}
    
    # Create OSS clients
    source_client = get_oss_client(context, bucket_name)
    target_client = get_oss_client(context, TARGET_BUCKET)
    
    # Process the image
    try:
        results = process_image(source_client, target_client, object_key)
        
        response = {
            "statusCode": 200,
            "body": json.dumps({
                "source": object_key,
                "processed": results,
                "count": len(results)
            })
        }
        logger.info(f"Processed {len(results)} variants for {object_key}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing {object_key}: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "source": object_key})
        }
```

### 第二步：依赖管理

`requirements.txt` 很简单，主要是 Pillow 处理图片，oss2 操作存储。

```txt
# code/requirements.txt
Pillow>=10.0.0
oss2>=2.18.0
```

### 第三步：Serverless Devs 配置

用 `s.yaml` 定义函数和触发器。这里我配了 EventBridge 触发器，只监听 OSS 的上传事件。注意 `eventRuleFilterPattern`，别什么文件都触发，只处理图片。

```yaml
# s.yaml
edition: 3.0.0
name: image-pipeline
access: default

resources:
  image-processor:
    component: fc3
    props:
      region: cn-beijing
      functionName: image-processor
      description: "Event-driven image processing pipeline"
      runtime: python3.10
      handler: index.handler
      memorySize: 1024
      timeout: 120
      code: ./code
      layers:
        - acs:fc:cn-beijing:123456789:layers/image-processing-deps/versions/1
      environmentVariables:
        SOURCE_BUCKET: upload-images
        TARGET_BUCKET: processed-images
        OSS_ENDPOINT: https://oss-cn-beijing-internal.aliyuncs.com
        WATERMARK_TEXT: "chenk.top"
      role: acs:ram::123456789:role/fc-oss-access
      triggers:
        - triggerName: eb-oss-upload
          triggerType: eventbridge
          triggerConfig:
            triggerEnable: true
            asyncInvocationType: true
            eventRuleFilterPattern: >
              {
                "source": ["acs.oss"],
                "type": [
                  "oss:ObjectCreated:PutObject",
                  "oss:ObjectCreated:PostObject",
                  "oss:ObjectCreated:CompleteMultipartUpload"
                ]
              }
```

### 第四步：Terraform 基础设施

基础设施即代码，全量自动化。这段 Terraform 脚本会把 OSS Bucket、RAM 角色、函数计算和 EventBridge 规则全部创建好。我把环境变量也写进了 resource 里，方便多环境管理。

```hcl
# main.tf — Infrastructure for the image processing pipeline

terraform {
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.220"
    }
  }
}

provider "alicloud" {
  region = "cn-beijing"
}

# --- OSS Buckets ---

resource "alicloud_oss_bucket" "source" {
  bucket = "upload-images-${var.env}"
  acl    = "private"

  cors_rule {
    allowed_origins = ["https://example.com"]
    allowed_methods = ["PUT", "POST"]
    allowed_headers = ["*"]
    max_age_seconds = 3600
  }

  lifecycle_rule {
    id      = "cleanup-uploads"
    enabled = true
    prefix  = "uploads/"

    expiration {
      days = 30
    }
  }
}

resource "alicloud_oss_bucket" "target" {
  bucket = "processed-images-${var.env}"
  acl    = "private"

  lifecycle_rule {
    id      = "archive-old"
    enabled = true

    transition {
      days          = 90
      storage_class = "IA"
    }
  }
}

# --- RAM Role for Function Compute ---

resource "alicloud_ram_role" "fc_role" {
  name     = "fc-image-processor-role"
  document = jsonencode({
    Version   = "1"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = {
          Service = ["fc.aliyuncs.com"]
        }
      }
    ]
  })
}

resource "alicloud_ram_policy" "fc_oss_policy" {
  policy_name = "fc-oss-readwrite"
  policy_document = jsonencode({
    Version   = "1"
    Statement = [
      {
        Action = [
          "oss:GetObject",
          "oss:PutObject",
          "oss:ListObjects"
        ]
        Effect   = "Allow"
        Resource = [
          "acs:oss:*:*:upload-images-${var.env}/*",
          "acs:oss:*:*:processed-images-${var.env}/*"
        ]
      }
    ]
  })
}

resource "alicloud_ram_role_policy_attachment" "fc_oss" {
  role_name   = alicloud_ram_role.fc_role.name
  policy_name = alicloud_ram_policy.fc_oss_policy.policy_name
  policy_type = "Custom"
}

# --- Function Compute ---

resource "alicloud_fcv3_function" "image_processor" {
  function_name = "image-processor"
  description   = "Event-driven image processing pipeline"
  handler       = "index.handler"
  runtime       = "python3.10"
  memory_size   = 1024
  timeout       = 120
  role          = alicloud_ram_role.fc_role.arn

  code {
    zip_file = filebase64("${path.module}/code.zip")
  }

  environment_variables = {
    SOURCE_BUCKET  = alicloud_oss_bucket.source.bucket
    TARGET_BUCKET  = alicloud_oss_bucket.target.bucket
    OSS_ENDPOINT   = "https://oss-cn-beijing-internal.aliyuncs.com"
    WATERMARK_TEXT  = "chenk.top"
  }
}

# --- EventBridge Rule ---

resource "alicloud_event_bridge_rule" "oss_to_fc" {
  event_bus_name = "default"
  rule_name      = "oss-upload-to-image-processor"
  description    = "Route OSS uploads to image processor function"

  filter_pattern = jsonencode({
    source  = ["acs.oss"]
    type    = [
      "oss:ObjectCreated:PutObject",
      "oss:ObjectCreated:PostObject",
      "oss:ObjectCreated:CompleteMultipartUpload"
    ]
    subject = ["acs:oss:cn-beijing:*:${alicloud_oss_bucket.source.bucket}"]
  })

  targets {
    target_id = "fc-image-processor"
    type      = "acs.fc.function"
    endpoint  = alicloud_fcv3_function.image_processor.arn

    param_list {
      resource_key = "Body"
      form         = "ORIGINAL"
    }
  }
}

# --- Variables ---

variable "env" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# --- Outputs ---

output "source_bucket" {
  value = alicloud_oss_bucket.source.bucket
}

output "target_bucket" {
  value = alicloud_oss_bucket.target.bucket
}

output "function_name" {
  value = alicloud_fcv3_function.image_processor.function_name
}
```

### 第五步：部署与测试

直接跑命令验证。先用 Terraform 创建资源，再用 Serverless Devs 部署函数。

```bash
# Deploy infrastructure with Terraform
cd terraform
terraform init
terraform plan -var="env=dev"
terraform apply -var="env=dev" -auto-approve

# Package and deploy the function
cd ../image-pipeline
pip install -r code/requirements.txt -t code/ --no-cache-dir
s deploy

# Test: upload an image to the source bucket
aliyun oss cp test-image.jpg \
  oss://upload-images-dev/uploads/test-image.jpg \
  --endpoint oss-cn-beijing.aliyuncs.com

# Wait a few seconds for the pipeline to process
sleep 5

# Verify the processed images exist
aliyun oss ls oss://processed-images-dev/resized/ \
  --endpoint oss-cn-beijing.aliyuncs.com

aliyun oss ls oss://processed-images-dev/webp/ \
  --endpoint oss-cn-beijing.aliyuncs.com

# Check function logs
s logs --tail
```

上传一张图，吐出六个文件。整套流水线跑下来大概 2-3 秒，单张成本大概 0.0001 元。就算处理 10,000 张图片，也就 1 块钱左右。

```
resized/large/test-image.jpg      (1200px wide, watermarked)
resized/medium/test-image.jpg     (600px wide, watermarked)
resized/thumb/test-image.jpg      (150px wide, no watermark)
webp/large/test-image.webp        (1200px wide, watermarked)
webp/medium/test-image.webp       (600px wide, watermarked)
webp/thumb/test-image.webp        (150px wide, no watermark)
```

### 第六步：监控与异常处理

![FC 异步调用：重试策略与死信队列](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_async_dlq.png)

异步调用自带托管重试队列。配置好重试预算和 `onFailure` 目的地，任何耗尽重试的事件都会落到 MNS 等待人工介入，而不是被静默丢弃。

真要上生产，死信队列和报警机制不能少。万一函数执行失败，事件不能丢，得进 DLQ 等着人工排查。

```bash
# Create an MNS queue for failed events
aliyun mns CreateQueue \
  --QueueName image-processor-dlq \
  --DelaySeconds 0 \
  --MaximumMessageSize 65536 \
  --MessageRetentionPeriod 1209600 \
  --VisibilityTimeout 60

# Configure async invocation with retry and DLQ
aliyun fc PutAsyncInvokeConfig \
  --functionName image-processor \
  --qualifier LATEST \
  --destinationConfig '{
    "onFailure": {
      "destination": "acs:mns:cn-beijing:123456789:/queues/image-processor-dlq/messages"
    }
  }' \
  --maxAsyncRetryAttempts 2 \
  --maxAsyncEventAgeInSeconds 3600
```
## Function Compute 计费

FC 的计费主要看三个维度。搞清楚这些，免得账单出来吓一跳。

![函数计算计费：三个独立维度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_pricing.png)

FC 的账单沿三条独立轴线累加：函数被调用了多少次，消耗了多少 memory × time，以及多少数据离开了云。免费额度对大多数业余项目和中小企业的工作负载都绰绰有余。

![Serverless 与传统服务器成本对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/08-serverless/08_cost_crossover.png)

| 维度 | 单价 (cn-beijing) | 免费额度 (每月) |
|---|---|---|
| **调用次数** | 每 10,000 次调用 0.0133 CNY | 1,000,000 次调用 |
| **执行时长** | 每 GiB-秒 0.000110592 CNY | 400,000 GiB-秒 |
| **公网出流量** | 每 GiB 0.50 CNY | 1 GiB |

免费额度挺大方的。咱们算笔实战账：

**场景：** 一个 API 函数，内存 512 MiB，平均执行 200ms，每月调用 500 万次。

```
Invocations:
  5,000,000 - 1,000,000 (free) = 4,000,000 billable
  4,000,000 / 10,000 * 0.0133 = 5.32 CNY

Execution duration:
  5,000,000 * 0.2s * 0.5 GiB = 500,000 GiB-seconds
  500,000 - 400,000 (free) = 100,000 billable
  100,000 * 0.000110592 = 11.06 CNY

Total: 5.32 + 11.06 = 16.38 CNY/month
```

一个月 16 块钱，扛 500 万次 API 调用。换 ECS 来处理同样的量，成本得翻 30 到 50 倍。

> **预留实例会额外计费。** 如果你保持 5 个 512 MiB 内存的实例常热：5 * 0.5 GiB * 86,400 s/day * 30 days * 0.000110592 = ~717 CNY/month。预留实例只值得用在那些对延迟极其敏感、无法接受冷启动的函数上。

至于配合 Serverless 函数做 AI 处理的 Bailian API，可以参考我们的 [Bailian 系列文章](/zh/aliyun-bailian/02-qwen-llm-api/)。

## FC 对比 AWS Lambda：实战视角

如果你正在评估 Function Compute 和 Lambda，下面这些差异才是实战中真正关键的：

| 特性 | Function Compute (FC) | AWS Lambda |
|---|---|---|
| 最大超时时间 | 15 min (sync), 24h (async) | 15 min |
| 最大内存 | 32 GiB | 10 GiB |
| 最大包大小 | 500 MiB (OSS) | 250 MiB (S3) |
| 临时存储 | 10 GiB | 10 GiB |
| 预留并发 | Yes | Yes |
| 容器支持 | Yes (Custom Container) | Yes (Container Image) |
| VPC 访问 | Yes (service-level config) | Yes (function-level config) |
| Layers | Yes (5 per function) | Yes (5 per function) |
| 免费额度 | 1M invocations + 400K GiB-s | 1M invocations + 400K GiB-s |
| GPU 支持 | No | No (standard), Yes (Bedrock) |
| SnapStart (Java) | No | Yes |
| ARM 支持 | No | Yes (Graviton2) |
| 价格 | 国内区域便宜 ~15% | 全球标准定价 |

两者功能其实差不多。FC 的优势在于国内区域价格更低，内存上限更高（32 GiB 对 10 GiB），还有 Service 抽象用来 grouping 相关函数。Lambda 的优势则是 ARM 支持（算力成本更低）、Java 的 SnapStart，以及更成熟的扩展生态和集成。

## 核心建议

**Serverless 并非万能。** 它擅长事件驱动、突发流量、短任务。长运行进程、GPU 任务、超低延迟场景它搞不定。得清楚你的流量模式拐点在哪。

**新 API 优先上 Function Compute。** 除非你第一天就知道需要持久连接或持续高吞吐，否则 FC 更便宜、简单、运维负担轻。以后随时能迁到 ECS，反过来可就难了。

**冷启动可控。** 选 Go 或 Python 冷启动快。用 Layer 控制包大小。预留实例只给 latency-critical 路径用。predictable traffic 可以用定时触发预热。

**用 EventBridge 解耦架构。** 别搞点对点集成（比如 OSS 触发直接调 FC），把事件路由经过 EventBridge。过滤、转换、扇出、死信队列、审计轨迹全都有。多配这点东西，等你需要给同一个事件加第二个消费者时就回本了。

**生产环境 API 配上 API Gateway。** FC 自带的 HTTP 触发器做内部服务或原型没问题。对外服务的话，前面挡一层 API Gateway 做认证、限流和监控。

**图片处理流水线是个模板。** 这个模式——OSS 上传触发 EventBridge 触发 Function Compute 写回 OSS——适用于任何文件处理流程：PDF 生成、视频转码、数据导入、日志解析。换掉处理逻辑，就是一套新流水线。

系列下一篇，我们聊容器编排 Container Service for Kubernetes (ACK)——适合那些超出 Serverless 承载但仍需云原生运维的工作负载。如果你想找基础设施即代码的方式部署 FC 函数，本文解决方案部分展示的 Terraform 集成是推荐的起点。