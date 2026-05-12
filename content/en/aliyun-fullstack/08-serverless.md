---
title: "Alibaba Cloud Full Stack (8): Serverless — Function Compute and EventBridge"
date: 2026-05-05 09:00:00
tags:
  - Alibaba Cloud
  - Function Compute
  - EventBridge
  - Serverless
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 8
description: "Go serverless on Alibaba Cloud: Function Compute triggers, runtimes, cold starts, and pricing. EventBridge for event routing. API Gateway integration. Build an event-driven image processing pipeline."
disableNunjucks: true
translationKey: "aliyun-fullstack-8"
---

The first time I saw a Function Compute bill that was 0.03 CNY for handling 10,000 requests, I started rethinking my entire architecture. I had been running a 2-vCPU ECS instance 24/7 to serve an API that processed maybe 200 requests per hour, paying around 490 CNY/month. The same workload on Function Compute cost under 5 CNY/month. Not 5 CNY per day — 5 CNY per month. The math was so lopsided that I spent the next weekend migrating everything that did not need a persistent process off ECS and onto functions.

Serverless does not mean there are no servers. It means you stop thinking about servers. You write a function, you define what triggers it, and the platform handles provisioning, scaling, patching, and decommissioning. You pay only for the milliseconds your code actually runs. No traffic, no charge. A million requests in five minutes, a million function instances spin up. You never touch a capacity slider.

This article covers the two serverless building blocks on Alibaba Cloud: Function Compute (the execution engine) and EventBridge (the event routing layer). By the end, we will build a complete event-driven image processing pipeline that resizes, watermarks, and generates thumbnails — triggered automatically when files land in OSS.


## When serverless makes sense (and when it doesn't)

Serverless is not a universal solution. It is a tool with a specific sweet spot, and understanding that sweet spot upfront saves you from a painful migration back to ECS three months later.

### Good fits for serverless

| Use case | Why it works | Example |
|---|---|---|
| **Event processing** | Naturally stateless, triggered by external events | Process OSS uploads, parse SLS logs |
| **Webhooks** | Infrequent, unpredictable traffic | GitHub webhook handler, payment callbacks |
| **Scheduled jobs** | Run once per hour/day, idle the rest | Daily report generation, data cleanup |
| **API backends with bursty traffic** | Scale-to-zero between bursts saves money | Marketing campaign API, seasonal e-commerce |
| **Data transformation** | Short-lived, CPU-bound, embarrassingly parallel | ETL pipelines, format conversion |
| **Chatbot backends** | Request-response pattern, variable load | DingTalk bot, Slack integration |

### Bad fits for serverless

| Use case | Why it fails | Better alternative |
|---|---|---|
| **Long-running processes (>15 min)** | FC has a 15-minute execution timeout | ECS, Container Service (ACK) |
| **GPU workloads** | No GPU support in FC | ECS GPU instances, PAI-EAS |
| **Low-latency requirements (<10ms)** | Cold starts add 100ms-2s | ECS with persistent process |
| **WebSocket / persistent connections** | Functions are request-response | ECS, ALB with sticky sessions |
| **High steady-state throughput** | Continuous execution is cheaper on ECS | ECS with reserved instances |
| **Large in-memory state** | Functions are stateless, max 3 GiB memory | ECS, Redis |

![Function Compute vs Serverless App Engine vs ECS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_fc_sae_ecs.png)

Before comparing prices, line up the three compute primitives side by side. FC, SAE, and ECS each occupy a distinct point on the granularity / lifecycle spectrum, and most architecture mistakes I see come from picking the wrong one for the workload.

### The cost crossover point

The question everyone asks: at what traffic level does ECS become cheaper than Function Compute?

Here is the math for a Python function with 512 MiB memory, 200ms average execution time:

| Monthly requests | FC cost (CNY) | ECS equivalent (CNY) | Winner |
|---|---|---|---|
| 10,000 | ~0.03 | ~490 (c7.large) | FC by 16,000x |
| 100,000 | ~0.30 | ~490 | FC by 1,600x |
| 1,000,000 | ~3.00 | ~490 | FC by 160x |
| 10,000,000 | ~30 | ~490 | FC by 16x |
| 100,000,000 | ~300 | ~490 | FC |
| 500,000,000 | ~1,500 | ~490 | ECS |
| 1,000,000,000 | ~3,000 | ~490 | ECS by 6x |

The crossover happens around 200-300 million requests per month for this configuration. Below that, Function Compute wins decisively. Above that, a dedicated ECS instance is cheaper — but you also need to handle scaling, patching, and availability yourself.

> **My rule of thumb:** If your function averages fewer than 100 requests per second sustained (roughly 250 million/month), serverless is almost certainly cheaper and operationally simpler. If you consistently exceed that, evaluate ECS or a container solution.

## Function Compute (FC) fundamentals

Function Compute is Alibaba Cloud's serverless execution service. The AWS equivalent is Lambda. The core concepts map directly:

![Function Compute architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_fc_architecture.png)

| FC concept | What it is | AWS equivalent |
|---|---|---|
| **Service** | A logical grouping of related functions. Shares VPC config, log config, NAS mount, and role. | (No direct equivalent; Lambda uses tags/prefixes) |
| **Function** | The unit of execution. Your code + configuration. | Lambda function |
| **Trigger** | An event source that invokes the function. | Event source mapping / trigger |
| **Layer** | Shared dependencies packaged separately from function code. | Lambda layer |
| **Custom domain** | Map your own domain to HTTP-triggered functions. | API Gateway custom domain |
| **Alias / Version** | Immutable snapshots with traffic-shifting aliases. | Lambda versions and aliases |

### The execution model

When a request arrives, FC does the following:

1. **Check for a warm instance.** If a function instance from a recent invocation is still alive (kept warm for ~5-15 minutes), route the request there. No cold start.
2. **Cold start (if needed).** Download the code package, initialize the runtime, run your initialization code (module-level imports, DB connection pools). This takes 100ms to 2s depending on runtime and package size.
3. **Execute the handler.** Run your function with the event payload. You are billed from this point.
4. **Return the response.** The instance stays warm for subsequent requests.

Key limits to know:

| Limit | Value |
|---|---|
| Max execution time | 15 minutes (86,400s for async invocation) |
| Max memory | 32 GiB |
| Max code package (direct upload) | 100 MiB (compressed) |
| Max code package (OSS reference) | 500 MiB (uncompressed) |
| Max layers per function | 5 |
| Max concurrent instances (default) | 300 per function |
| Max payload (sync) | 32 MiB |
| Max payload (async) | 128 KiB |
| Temp disk (`/tmp`) | 10 GiB |

![FC request lifecycle: warm path vs cold start](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_lifecycle.png)

The gap between the warm path and the cold path is the entire performance story of serverless. The warm path costs a few milliseconds; the cold path runs four extra steps before your handler even starts.

### Runtimes

FC supports these managed runtimes:

| Runtime | Versions | Cold start (typical) | Notes |
|---|---|---|---|
| **Python** | 3.9, 3.10 | 200-400ms | Most popular. Great ecosystem. |
| **Node.js** | 14, 16, 18 | 150-300ms | Fast cold start. Good for API backends. |
| **Java** | 8, 11, 17 | 1-3s | Slowest cold start. Use GraalVM or SnapStart. |
| **Go** | 1.x (custom runtime) | 50-150ms | Fastest. Single binary, no runtime init. |
| **PHP** | 7.2 | 200-400ms | Legacy support. |
| **C#** | .NET Core 6 | 500ms-1s | Moderate. |
| **Custom Runtime** | Any (Docker) | Varies | Full control. For ML models, system libs. |
| **Custom Container** | Any Docker image | 1-10s | Largest packages. Requires image pull. |

> **If cold start matters to you:** Go is the fastest runtime by a wide margin because functions compile to a single static binary with no runtime initialization. Python is the practical sweet spot — moderate cold start, massive library ecosystem, easy to write. Java should be your last choice for latency-sensitive functions unless you use GraalVM native compilation.

## Writing your first function

Let's build a function from scratch. We will use Python 3.10 and the Serverless Devs (`s`) CLI, which is the standard toolchain for FC development.

### Install Serverless Devs

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

### Project structure

```
image-processor/
  ├── s.yaml              # Serverless Devs config
  ├── code/
  │   ├── index.py        # Function handler
  │   └── requirements.txt
  └── README.md
```

### The handler

Here is the simplest possible function — an HTTP-triggered hello world:

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

Let's break down the handler signature:

- **`event`**: The input payload. For HTTP triggers, this is the HTTP request body as bytes. For OSS triggers, it is a JSON object describing the uploaded file. The shape depends on the trigger type.
- **`context`**: A rich object containing the request ID (for log correlation), function metadata (name, memory, timeout), service metadata, credentials (temporary STS credentials if your function needs to call other Alibaba Cloud APIs), and region information.
- **Return value**: For HTTP triggers, return a dict with `statusCode`, `headers`, and `body`. For non-HTTP triggers, you can return any JSON-serializable value.

### The Serverless Devs configuration

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
      functionName: hello-world
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

### Deploy and test

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
# Output includes: url: https://hello-world-xxxx.cn-beijing.fcapp.run
```

Test it with curl:

```bash
# Test the deployed function
curl -X POST \
  https://hello-world-xxxx.cn-beijing.fcapp.run \
  -H "Content-Type: application/json" \
  -d '{"name": "Serverless"}'

# Response:
# {"message": "Hello, Serverless!", "request_id": "1-6789abcd-..."}
```

### Viewing logs

FC sends all `print()` and `logging` output to Simple Log Service (SLS). You can view logs in the FC console, or query them via CLI:

```bash
# View recent function logs
s logs --tail

# View logs for a specific request
s logs --request-id "1-6789abcd-..."
```

## Triggers: what invokes your function

Triggers are what make serverless event-driven rather than just "cheap hosting." Each trigger type connects your function to a different event source. Here is every trigger type with configuration examples.

![Function Compute trigger types](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_trigger_types.png)

### HTTP trigger

The simplest trigger. Your function gets an HTTP endpoint.

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

The function receives the full HTTP request and returns an HTTP response. This is the foundation for building REST APIs on Function Compute.

### OSS trigger

Fires when objects are created, modified, or deleted in an OSS bucket. This is the backbone of most serverless data pipelines.

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

The event payload looks like this:

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

OSS, the storage layer for our pipeline, is covered in detail in [Part 4](/en/aliyun-fullstack/04-oss-storage/).

### Timer trigger (cron)

Run functions on a schedule. Uses cron expressions or rate expressions.

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

Timer triggers are the serverless replacement for crontab. No more keeping a dedicated ECS instance alive just to run a script once a day.

### SLS trigger (log events)

Process log entries from Simple Log Service in near real-time.

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

### MNS trigger (message queue)

Consume messages from Message Service queues or topics.

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

### EventBridge trigger

The most flexible option. We cover EventBridge in depth later in this article, but here is the trigger setup:

![EventBridge event routing pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_event_pipeline.png)

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

### Trigger comparison

| Trigger | Latency | Use case | Max batch |
|---|---|---|---|
| HTTP | Sync, <100ms | API endpoints | 1 request |
| OSS | Async, 1-5s | File processing | 1 event |
| Timer | N/A | Scheduled tasks | N/A |
| SLS | 1-60s | Log analysis | Configurable |
| MNS | 1-5s | Message processing | 1 message |
| EventBridge | 1-5s | Complex event routing | 1 event |

## Cold starts and performance

Cold starts are the most discussed drawback of serverless. Understanding what causes them and how to mitigate them is essential for production use.

![Function Compute cold start analysis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_cold_start.png)

### What causes a cold start

A cold start happens when FC needs to create a new function instance. The sequence is:

1. **Download code** (10-100ms for small packages, 1-5s for large ones)
2. **Start the runtime** (50-500ms depending on language)
3. **Run initialization code** (your module-level imports, connection setup)
4. **Execute the handler**

Steps 1-3 are the cold start overhead. Step 4 is what you are billed for in a warm invocation. During a cold start, you are billed for all four steps.

### Measuring cold starts

Add timing to your function to measure the difference:

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

### Cold start benchmarks by runtime

I measured cold starts for a minimal function (hello world + one import) across runtimes on FC in `cn-beijing`:

| Runtime | Cold start (p50) | Cold start (p99) | Warm invocation (p50) |
|---|---|---|---|
| Go 1.x | 60ms | 150ms | 1ms |
| Node.js 18 | 180ms | 400ms | 3ms |
| Python 3.10 | 250ms | 600ms | 5ms |
| C# .NET 6 | 400ms | 900ms | 8ms |
| Java 17 | 1,200ms | 3,000ms | 5ms |
| Custom Container | 2,000ms | 8,000ms | 5ms |

Java's cold start is dominated by JVM startup. If you must use Java, GraalVM native image compilation can bring cold starts down to 200-400ms, but the build process is complex.

### Mitigation strategies

![FC concurrency: on-demand vs provisioned over a day](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_concurrency.png)

This is what concurrency looks like over a typical day. On-demand scaling tracks request rate but pays for it with cold starts on every spike; provisioned concurrency keeps a baseline of warm instances at the cost of idle time during the night. The three mitigation strategies below trade off these two failure modes.

**1. Provisioned concurrency**

Keep a set number of instances warm at all times. You pay for the idle time, but cold starts are eliminated.

```bash
# Keep 5 instances always warm
aliyun fc PutProvisionConfig \
  --functionName image-processor \
  --qualifier LATEST \
  --target 5
```

The cost: you pay for 5 instances continuously (about 0.000110592 CNY/GiB-second). For a 512 MiB function, that is roughly 0.055 CNY/s, or about 143 CNY/month for 5 instances. Worth it for latency-sensitive APIs; overkill for batch processing.

**2. Scheduled pre-warming**

If your traffic is predictable (e.g., spikes at 9:00 AM), use a timer trigger to invoke the function a few minutes before the spike:

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

**3. Minimize package size**

Every byte you add to your deployment package adds to cold start time. The code download phase is often the largest contributor.

```bash
# Bad: 150 MiB package with all of scipy/numpy/pandas
pip install -r requirements.txt -t ./code/

# Good: only install what you need
pip install Pillow requests -t ./code/ --no-cache-dir

# Better: use layers for shared dependencies
# (covered in the next section)
```

**4. Optimize initialization code**

Move expensive operations out of module scope when possible:

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

**5. Choose the right runtime**

If cold start is your primary concern, the priority order is: Go > Node.js > Python > C# > Java. If development speed matters more than cold start, Python is the practical winner.

## Layers and custom runtimes

### Layers: shared dependencies

![Function layers architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_layer_architecture.png)

A layer is a zip archive containing libraries, a custom runtime, or other dependencies. Layers are versioned and can be shared across multiple functions. They solve two problems:

1. **Package size reduction.** Move large dependencies (numpy, Pillow, etc.) into a layer. Your function code stays small = faster deploys and faster cold starts.
2. **Dependency sharing.** Ten functions that all use Pillow can reference the same layer instead of bundling Pillow ten times.

#### Creating a layer

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

#### Using a layer in your function

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

Now your function code only contains your business logic, and the layer provides the heavy dependencies. Deploy time drops from minutes to seconds.

### Custom runtimes

When the managed runtimes are not enough — you need a specific system library, a compiled binary, or an ML model — use a Custom Runtime or Custom Container.

#### Custom Runtime (HTTP server mode)

Your code runs as an HTTP server. FC starts your binary/script and sends requests to `localhost:9000`. You can use any language or framework.

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

#### When to use custom containers

- **ML inference:** Bundle your model weights (up to 10 GiB via NAS mount) and inference framework.
- **System dependencies:** FFmpeg for video processing, ImageMagick for image manipulation, wkhtmltopdf for PDF generation.
- **Non-supported languages:** Rust, C++, or any language that compiles to a binary.

The trade-off is cold start time. Custom containers need to pull the image on cold start, which can take 2-10 seconds. Use provisioned concurrency for latency-sensitive workloads.

## EventBridge

EventBridge is the event routing layer. If Function Compute is the engine that runs your code, EventBridge is the switchboard that decides which code runs in response to which event. The AWS equivalent is Amazon EventBridge (formerly CloudWatch Events).

### Core concepts

| Concept | What it is | Example |
|---|---|---|
| **Event bus** | A channel that receives events. Default bus receives all Alibaba Cloud service events. Custom buses for your own events. | `default`, `my-app-bus` |
| **Event source** | Where events come from. Built-in sources (OSS, ECS, RDS) or custom (your application). | `acs.oss`, `my.app` |
| **Event rule** | A filter + target mapping. "When event matches this pattern, send it to that target." | "OSS PutObject in uploads/ -> FC function" |
| **Event target** | Where matched events are delivered. FC, MNS, SLS, HTTP endpoint, another bus. | `acs:fc:cn-beijing:123456:functions/process-image` |

### The event format

Every event in EventBridge follows the CloudEvents 1.0 specification:

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

This standardized format means your event processing code does not need to know the specifics of each Alibaba Cloud service's event format — they all follow the same structure.

### Built-in event sources

EventBridge automatically receives events from these Alibaba Cloud services (partial list):

| Service | Event types | Example trigger |
|---|---|---|
| **OSS** | Object created, deleted, accessed | File upload processing |
| **ECS** | Instance state change, disk events | Auto-remediation on failure |
| **RDS** | Instance created, failover, high CPU | Database alerting |
| **Container Service** | Pod events, deployment changes | Deployment tracking |
| **SLS** | Alert triggered | Incident response |
| **ActionTrail** | API calls (audit log) | Security monitoring |
| **Cloud Monitor** | Alarm state change | Custom alerting pipeline |
| **MNS** | Message published | Message routing |

### Creating event rules

Event rules are the core of EventBridge. They filter events using JSON pattern matching and route matched events to targets.

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

### Event pattern matching

The filter pattern supports several matching operators:

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

This pattern matches: events from OSS where an object was created, the object is larger than 1 MiB, the key starts with `uploads/` and ends with `.jpg`.

Available operators:

| Operator | Syntax | Matches |
|---|---|---|
| Exact match | `["value1", "value2"]` | Field equals any listed value |
| Prefix | `[{"prefix": "abc"}]` | Field starts with "abc" |
| Suffix | `[{"suffix": ".jpg"}]` | Field ends with ".jpg" |
| Numeric | `[{"numeric": [">", 100]}]` | Field > 100 |
| IP address | `[{"cidr": "10.0.0.0/8"}]` | IP is in CIDR range |
| Exists | `[{"exists": true}]` | Field is present |
| Not | `[{"anything-but": "error"}]` | Field is not "error" |

### Event transformation

Before delivering events to a target, you can transform the event payload. This is useful when your target function expects a different format than the raw CloudEvent:

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

Instead of receiving the full CloudEvent, your function receives:

```json
{
  "bucket": "my-upload-bucket",
  "key": "uploads/photo-001.jpg",
  "size": 2048576
}
```

### Custom events

Your application can publish events to EventBridge for other services to consume:

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

This lets you build fully event-driven architectures where different microservices communicate through EventBridge rather than direct API calls — loose coupling at its best.

### EventBridge vs direct triggers

You might wonder: why use EventBridge when FC already has direct OSS triggers?

| Feature | Direct FC trigger | EventBridge |
|---|---|---|
| Setup complexity | Simple | More configuration |
| Filtering | Basic (prefix/suffix) | Rich pattern matching |
| Fan-out | 1 function per trigger | Multiple targets per rule |
| Dead letter queue | No | Yes |
| Retry policy | Limited | Configurable |
| Cross-service routing | No (OSS-to-FC only) | Any source to any target |
| Event transformation | No | Yes |
| Audit trail | Limited | Full event history |

**My recommendation:** Use direct triggers for simple, single-function scenarios (e.g., "resize every uploaded image"). Use EventBridge when you need complex routing, multiple targets, or cross-service event flows.

## API Gateway + Function Compute

For building proper REST APIs, API Gateway sits in front of your functions and provides features that FC's built-in HTTP trigger does not offer: authentication, rate limiting, request validation, and API versioning.

![API Gateway and Function Compute integration](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_api_gateway.png)

### Creating an API backed by Function Compute

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

Map your own domain to the API:

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

If your API is called from a browser, you need CORS headers. Handle this in your function:

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

## Solution: event-driven image processing pipeline

Let's bring everything together into a production-quality system. The architecture:

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

![Event-driven image processing pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_image_pipeline.png)

Here is the full pipeline at a glance. One PutObject in the source bucket fans out into six derivative files in the target bucket — three sizes × two formats — and the entire flow runs in 2-3 seconds without a single server to manage.

### Step 1: The function code

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

### Step 2: Dependencies

```txt
# code/requirements.txt
Pillow>=10.0.0
oss2>=2.18.0
```

### Step 3: Serverless Devs configuration

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

### Step 4: Terraform infrastructure

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

### Step 5: Deploy and test

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

Expected output after uploading one image:

```
resized/large/test-image.jpg      (1200px wide, watermarked)
resized/medium/test-image.jpg     (600px wide, watermarked)
resized/thumb/test-image.jpg      (150px wide, no watermark)
webp/large/test-image.webp        (1200px wide, watermarked)
webp/medium/test-image.webp       (600px wide, watermarked)
webp/thumb/test-image.webp        (150px wide, no watermark)
```

Six output files from one upload. The whole pipeline runs in about 2-3 seconds and costs roughly 0.0001 CNY per image. Processing 10,000 images costs about 1 CNY.

### Step 6: Add monitoring and error handling

![FC async invocation: retry policy and dead letter queue](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_async_dlq.png)

Async invocations get a managed retry queue for free. Configure the retry budget and an `onFailure` destination, and any event that exhausts retries lands in MNS for triage instead of being silently dropped.

For production, add a dead letter queue and alerting:

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

## Function Compute pricing

FC pricing has three dimensions. Understanding them avoids bill shock.

![Function Compute pricing — three independent dimensions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_pricing.png)

FC bills along three independent axes: how often the function is invoked, how much memory × time it consumes, and how much data leaves the cloud. The free tier covers most hobby and small-business workloads outright.

![Serverless vs server cost crossover](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/08-serverless/08_cost_crossover.png)

| Dimension | Rate (cn-beijing) | Free tier (monthly) |
|---|---|---|
| **Invocations** | 0.0133 CNY per 10,000 invocations | 1,000,000 invocations |
| **Execution duration** | 0.000110592 CNY per GiB-second | 400,000 GiB-seconds |
| **Public network outbound** | 0.50 CNY per GiB | 1 GiB |

The free tier is generous. Let's calculate a real workload:

**Scenario:** An API function with 512 MiB memory, 200ms average execution, called 5 million times per month.

```yaml
Invocations:
  5,000,000 - 1,000,000 (free) = 4,000,000 billable
  4,000,000 / 10,000 * 0.0133 = 5.32 CNY

Execution duration:
  5,000,000 * 0.2s * 0.5 GiB = 500,000 GiB-seconds
  500,000 - 400,000 (free) = 100,000 billable
  100,000 * 0.000110592 = 11.06 CNY

Total: 5.32 + 11.06 = 16.38 CNY/month
```

Sixteen CNY per month for five million API calls. An ECS instance to handle the same load would cost 30-50x more.

> **Provisioned concurrency adds cost.** If you keep 5 instances warm with 512 MiB memory: 5 * 0.5 GiB * 86,400 s/day * 30 days * 0.000110592 = ~717 CNY/month. Provisioned concurrency is worth it only for latency-critical functions where cold starts are unacceptable.

For the Bailian API used with serverless functions for AI-powered processing, see our [Bailian series](/en/aliyun-bailian/02-qwen-llm-api/).

## FC vs AWS Lambda: a practical comparison

If you are evaluating Function Compute against Lambda, here are the differences that matter in practice:

| Feature | Function Compute (FC) | AWS Lambda |
|---|---|---|
| Max timeout | 15 min (sync), 24h (async) | 15 min |
| Max memory | 32 GiB | 10 GiB |
| Max package size | 500 MiB (OSS) | 250 MiB (S3) |
| Temp storage | 10 GiB | 10 GiB |
| Provisioned concurrency | Yes | Yes |
| Container support | Yes (Custom Container) | Yes (Container Image) |
| VPC access | Yes (service-level config) | Yes (function-level config) |
| Layers | Yes (5 per function) | Yes (5 per function) |
| Free tier | 1M invocations + 400K GiB-s | 1M invocations + 400K GiB-s |
| GPU support | No | No (standard), Yes (Bedrock) |
| SnapStart (Java) | No | Yes |
| ARM support | No | Yes (Graviton2) |
| Pricing | ~15% cheaper in China regions | Standard global pricing |

The products are functionally similar. FC's advantages are lower pricing in Chinese regions, higher memory ceiling (32 GiB vs 10 GiB), and the Service abstraction for grouping related functions. Lambda's advantages are ARM support (lower cost per compute), SnapStart for Java, and a more mature ecosystem of extensions and integrations.

## Key takeaways

**Serverless is not always the answer.** It excels at event-driven, bursty, short-lived workloads. It fails at long-running processes, GPU tasks, and ultra-low-latency requirements. Know the crossover point for your traffic pattern.

**Start with Function Compute for new APIs.** Unless you know you need persistent connections or sustained high throughput from day one, FC is cheaper, simpler, and operationally lighter. You can always migrate to ECS later — the reverse migration is harder.

**Cold starts are manageable.** Choose Go or Python for fast cold starts. Use layers to keep packages small. Use provisioned concurrency only for latency-critical paths. Pre-warm with timer triggers for predictable traffic patterns.

**EventBridge decouples your architecture.** Instead of point-to-point integrations (OSS trigger directly calls FC), route events through EventBridge. You get filtering, transformation, fan-out, dead letter queues, and an audit trail. The extra configuration pays for itself the first time you need to add a second consumer for the same event.

**Use API Gateway for production APIs.** FC's built-in HTTP trigger is fine for internal services and prototypes. For anything public-facing, put API Gateway in front for authentication, rate limiting, and monitoring.

**The image processing pipeline is a template.** The pattern — OSS upload triggers EventBridge triggers Function Compute writes back to OSS — applies to any file processing workflow: PDF generation, video transcoding, data import, log parsing. Swap the processing logic and you have a new pipeline.

Next in this series, we tackle container orchestration with Container Service for Kubernetes (ACK) — for workloads that outgrow serverless but still need cloud-native operations. If you are looking for the infrastructure-as-code approach to deploying FC functions, the Terraform integration shown in this article's solution section is the recommended starting point.
