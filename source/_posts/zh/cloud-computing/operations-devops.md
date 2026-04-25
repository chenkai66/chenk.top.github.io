---
title: "运维与 DevOps 实践"
date: 2023-05-26 09:00:00
tags:
  - 云计算
  - DevOps
  - SRE
  - 基础设施即代码
  - 监控
  - CI/CD
categories: 云计算
series:
  name: "云计算"
  part: 6
  total: 8
lang: zh-CN
mathjax: false
description: "工程师视角的 DevOps 实战：能把控质量的 CI/CD、可复现的 Terraform 基础设施、Prometheus + Grafana 监控、ELK / EFK 日志、SRE 错误预算，以及凌晨三点能撑住的运维习惯。"
disableNunjucks: true
series_order: 7
---

2017 年 GitLab 丢了六个小时的数据库状态。一位疲惫的工程师在事故处理中对错了服务器跑了 `rm -rf`。备份流程其实已经悄悄坏了几个月，但没人发现，因为没人在做恢复演练。教训不是"用 `rm` 要小心"。教训是：运维是一个**系统**——工具、运行手册、监控、自动化，以及围绕这一切的仪式。系统健康时，任何一个疲惫工程师都搞不挂生产；系统腐烂时，每一次深夜抢救都离灾难一个按键。

本文讲的就是怎么把这个系统建起来。在代码触达用户前把质量挡住的 CI/CD；让"生产环境"成为一个 Git 提交而不是雪花服务器的 IaC；能把噪声和信号分开的监控；真正能搜的日志；以及把救火工程化的 SRE 实践——错误预算、SLO、无指责复盘。

## 你将学到

- CI/CD 流水线：阶段、质量门、回滚，以及一份完整的 GitHub Actions 例子
- 用 Terraform 做基础设施即代码：工作流、状态管理、模块模式
- Prometheus + Grafana + Alertmanager 监控：抓取模型、PromQL、告警规则
- 集中化日志架构（EFK / ELK）：采集器、缓冲、处理器、保留分层
- 不抖动的弹性伸缩
- 不需要重写应用的成本优化
- SRE 实践：SLI / SLO / 错误预算、无指责复盘、GitOps

## 前置知识

- Linux 命令行熟练
- Git 和基本 CI/CD 概念
- 建议先阅读本系列前 5 篇

---

## 1. CI/CD 流水线：发布的"系统记录"

![CI/CD 流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig1_cicd_pipeline.png)

现代 CI/CD 流水线不只是"自动化"。它是代码进入生产**唯一被允许**的路径——也因此成了每一次发布的系统记录：谁、发了什么、跑过哪些测试、对哪一版基础设施、之后发生了什么。运维栈其余每一块都挂在这条主干上。

### 1.1 八个阶段

| 阶段 | 目的 | 失效模式 |
|------|------|----------|
| Commit | 通过 push / merge 触发 | 不会失败，只是事件 |
| Build | 编译、打包、镜像 | 可复现性（基础镜像锁定、依赖锁定） |
| 单元测试 | 逻辑层面的快速反馈 | flaky 测试侵蚀信任，要积极隔离 |
| 安全扫描 | SAST、依赖 CVE、镜像扫描 | 噪声大，按仓库分级阈值 |
| 部署到 Staging | 制品第一次真的跑起来 | Staging 与 Prod 配置漂移 |
| Smoke / E2E | 跨服务契约 | 太慢就被人跳过 |
| 部署到 Prod | 金丝雀 -> 全量 | 一次性铺开、缺自动回滚 |
| 验证 | 部署后 SLO 检查 | "肉眼"验证，没量化 |

### 1.2 一份真实的 GitHub Actions 流水线

```yaml
name: deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: "pip" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest --cov=app --cov-report=xml --cov-fail-under=80
      - uses: codecov/codecov-action@v4

  security:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: fs
          severity: HIGH,CRITICAL
          exit-code: 1

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    permissions: { id-token: write, contents: read }
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::111122223333:role/github-deploy
          aws-region: us-east-1
      - uses: aws-actions/amazon-ecr-login@v2
      - run: |
          docker build -t $ECR/app:${{ github.sha }} .
          docker push $ECR/app:${{ github.sha }}
        env: { ECR: 111122223333.dkr.ecr.us-east-1.amazonaws.com }

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production         # GitHub 人工审批门
    permissions: { id-token: write, contents: read }
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::111122223333:role/github-deploy
          aws-region: us-east-1
      - run: aws ecs update-service --cluster prod --service app
                                    --force-new-deployment
      - run: aws ecs wait services-stable --cluster prod --services app

  verify:
    needs: deploy
    runs-on: ubuntu-latest
    steps:
      - run: ./scripts/check_slo.sh    # 失败时通过重新部署回滚
```

让这份流水线从"在我机器上行"变成"凌晨三点也行"的三个细节：

- **OIDC，不用长期凭证。**`id-token: write` 让 GitHub 颁发短期 AWS Token，你完全不需要存 `AWS_ACCESS_KEY_ID`。
- **生产环境的人工审批门**（`environment: production`），Staging 自动通过。
- **部署后的验证步骤**：检查 SLO，新版本更差就触发自动回滚。

### 1.3 部署策略

- **滚动更新**：每次替换 N 个 Pod。默认选项；K8s / ECS 帮你做健康检查。
- **蓝绿**：并行起新版本，把 LB 指针一切。回滚瞬间完成；占用计算翻倍。
- **金丝雀**：1% 流量给新版本，看指标；扩到 5%、25%、100%。任何"出了问题就疼"的服务都该用。
- **特性开关**：代码先暗发，按人群打开。把"部署"和"发布"解耦。

金丝雀 + 特性开关是金标准——你能**独立**控制铺开和暴露。

## 2. 基础设施即代码：Terraform

![Terraform 工作流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig2_iac_terraform.png)

IaC 的意义不在于抽象的"自动化"。意义是：**生产环境就是一个 Git 提交**。能 diff、能 review、能回滚、能复现——这些事在手搓环境里不可能做到，无论 wiki 写得多好。

### 2.1 Terraform 工作流

```
HCL 文件 -> terraform init   -> 下载 provider
         -> terraform plan   -> diff 期望状态 vs 实际状态
         -> terraform apply  -> 调用云 API 收敛
         -> terraform.tfstate（我们认知中的世界状态）
```

`plan` 是你真正要生活在里面的那一步。它告诉你**变更发生之前**会变什么。Code review 是对着 plan 输出审，不只是看 HCL。

### 2.2 一份完整的生产模块

```hcl
# modules/web-service/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "name"           { type = string }
variable "environment"    { type = string }
variable "instance_type"  { type = string, default = "t3.medium" }
variable "min_size"       { type = number, default = 2 }
variable "max_size"       { type = number, default = 10 }
variable "subnet_ids"     { type = list(string) }
variable "vpc_id"         { type = string }

locals {
  full_name = "${var.environment}-${var.name}"
  tags = {
    Environment = var.environment
    Service     = var.name
    ManagedBy   = "terraform"
  }
}

resource "aws_security_group" "web" {
  name_prefix = "${local.full_name}-"
  vpc_id      = var.vpc_id
  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port = 0; to_port = 0; protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.tags
}

resource "aws_launch_template" "web" {
  name_prefix   = "${local.full_name}-"
  image_id      = data.aws_ami.al2023.id
  instance_type = var.instance_type
  vpc_security_group_ids = [aws_security_group.web.id]
  iam_instance_profile { name = aws_iam_instance_profile.web.name }
  metadata_options { http_tokens = "required" }     # 强制 IMDSv2
  tag_specifications {
    resource_type = "instance"
    tags          = local.tags
  }
}

resource "aws_autoscaling_group" "web" {
  name_prefix         = "${local.full_name}-"
  vpc_zone_identifier = var.subnet_ids
  min_size            = var.min_size
  max_size            = var.max_size
  desired_capacity    = var.min_size
  health_check_type   = "ELB"
  health_check_grace_period = 90
  launch_template { id = aws_launch_template.web.id, version = "$Latest" }
  target_group_arns = [aws_lb_target_group.web.arn]
  instance_refresh {
    strategy = "Rolling"
    preferences { min_healthy_percentage = 90 }
  }
  dynamic "tag" {
    for_each = local.tags
    content { key = tag.key, value = tag.value, propagate_at_launch = true }
  }
}

output "asg_name"      { value = aws_autoscaling_group.web.name }
output "target_group"  { value = aws_lb_target_group.web.arn }
```

值得注意：

- **用 `name_prefix` 而不是 `name`**——Terraform 可以"先建后毁"（`create_before_destroy` 风格）。
- **强制 IMDSv2**（`http_tokens = "required"`）——直接堵死 Capital One 那条 SSRF 偷凭证链。
- **ASG 上的 instance refresh**：launch template 一改，自动滚动替换实例。
- **tag 从一个 `locals` 传播下去**——成本归属对得上，审计也对得上。

### 2.3 远程状态与锁

本地 state 是个大坑。两个工程师同时 `apply` 会把彼此的世界观写花。用带锁的远程后端：

```hcl
terraform {
  backend "s3" {
    bucket         = "company-tfstate"
    key            = "prod/web-service.tfstate"
    region         = "us-east-1"
    dynamodb_table = "tfstate-lock"   # 提供锁
    encrypt        = true
    kms_key_id     = "alias/terraform-state"
  }
}
```

按环境、按服务分 state。一个巨大的 state 文件慢且危险；一个资源一个 state 文件无法管理。

### 2.4 Terraform vs Ansible vs 云原生选项

| 工具 | 适合做什么 | 语言 | 多云 |
|------|-----------|------|------|
| Terraform | 资源开通 | HCL（声明式） | 是 |
| Ansible | 资源内部配置 | YAML（半过程式） | 是 |
| CloudFormation / ARM / Deployment Manager | 单云、深度集成 | YAML / JSON | 否 |
| Pulumi / CDK | 跟 Terraform 一样但用真实代码 | TS / Python / Go | 是 |

成熟的模式是：云资源用 Terraform，应用配置用容器镜像，Ansible 只留给边缘上还活着的遗留机器。

## 3. 用 Prometheus + Grafana + Alertmanager 做监控

![监控栈](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig3_monitoring_stack.png)

### 3.1 三大支柱——以及它们为什么不够

指标、日志、链路。多数团队从指标起步、加日志、打算加链路。三大支柱**必要但不充分**：缺的那一块是**关联**。一次指标尖刺、一条错误日志、一段异常 trace 是同一起事故；如果你不能一键在三者之间切换，你只是数据多，并不可观测。

### 3.2 一份能撑规模的 Prometheus 配置

```yaml
global:
  scrape_interval:     15s
  evaluation_interval: 15s
  external_labels:
    cluster:     production
    region:      us-east-1

rule_files:
  - "rules/recording.yml"
  - "rules/alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: kubernetes-pods
    kubernetes_sd_configs: [{ role: pod }]
    relabel_configs:
      # 只抓取通过注解显式开启的 Pod
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: "true"
      # 尊重自定义路径 / 端口注解
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__,
                        __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      # 把有用的标签提到时序上
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
```

让 Prometheus 在规模下活下来的三个模式：

- **服务发现，不是静态目标。**K8s 里 Pod 来去自如，静态目标分分钟过期。
- **opt-in 抓取。**没有 `prometheus.io/scrape: "true"` 注解，不应该把整个集群每个 sidecar 都抓一遍。
- **Recording rules** 把昂贵查询提前算好。每次仪表盘加载不再重算，预聚合后查询便宜很多。

### 3.3 你真正会用的 PromQL

```promql
# 节点 CPU 利用率（剔除 idle）
100 - avg by (instance) (
  irate(node_cpu_seconds_total{mode="idle"}[5m])
) * 100

# 每个服务的请求速率
sum by (service) (rate(http_requests_total[5m]))

# 错误率（占总数比例）
sum by (service) (rate(http_requests_total{status=~"5.."}[5m]))
/
sum by (service) (rate(http_requests_total[5m]))

# 来自 histogram 的 P95 延迟
histogram_quantile(0.95,
  sum by (le, service) (rate(http_request_duration_seconds_bucket[5m])))

# Apdex（excellent < 100ms、satisfactory < 500ms）
(
  sum by (service) (rate(http_request_duration_seconds_bucket{le="0.1"}[5m]))
  +
  sum by (service) (rate(http_request_duration_seconds_bucket{le="0.5"}[5m]))
) / 2
/ sum by (service) (rate(http_request_duration_seconds_count[5m]))
```

值得记的形状：**对 histogram 桶做 rate** 算延迟、**对 count 做 rate 并按 status 切片** 算错误率。几乎每一个有意义的应用指标都归结为这两类。

### 3.4 尊重 on-call 的告警规则

```yaml
groups:
- name: slo-burn-rate
  rules:
  - alert: ErrorBudgetBurnFast
    # 1 小时窗口内以 14.4 倍正常速率燃烧：约 2 天烧完月度预算
    expr: |
      (
        sum by (service) (rate(http_requests_total{status=~"5.."}[1h]))
        /
        sum by (service) (rate(http_requests_total[1h]))
      ) > 0.014
      and
      (
        sum by (service) (rate(http_requests_total{status=~"5.."}[5m]))
        /
        sum by (service) (rate(http_requests_total[5m]))
      ) > 0.014
    for: 2m
    labels: { severity: page }
    annotations:
      summary: "{{ $labels.service }} 错误预算 14x 速率燃烧"
      runbook: "https://runbooks/internal/slo-burn"

  - alert: ErrorBudgetBurnSlow
    # 6 小时窗口 6 倍正常速率：约 5 天烧完月度预算
    expr: |
      (
        sum by (service) (rate(http_requests_total{status=~"5.."}[6h]))
        /
        sum by (service) (rate(http_requests_total[6h]))
      ) > 0.006
    for: 15m
    labels: { severity: ticket }
```

来自 Google SRE Workbook 的模式：基于**错误预算的燃烧速率**告警，而不是裸阈值。快烧 page（立刻响应）+ 慢烧 ticket（下个工作日处理），消灭两类典型误报：很快自愈的小抖动、还没影响用户但已经在慢漏的趋势。

## 4. 集中化日志

![日志架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig4_logging_architecture.png)

### 4.1 流水线

```
应用 stdout -> 采集器（Fluent Bit / Filebeat）-> 缓冲（Kafka）
            -> 处理器（Logstash / Fluentd）-> 存储（Elasticsearch / Loki）
            -> 仪表盘（Kibana / Grafana）
```

多数团队会跳过缓冲层，然后后悔。没有缓冲，下游 Elasticsearch 一抖动就反压，采集器顶不住，日志直接丢。中间塞一个 Kafka，采集器以满速写入，处理器按自己的节奏追上。

### 4.2 应用层结构化日志

纯文本日志在规模下不可搜。第一天就发 JSON：

```python
import json, logging, sys, time
import contextvars

request_id = contextvars.ContextVar("request_id", default="-")

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "module":  record.module,
            "line":    record.lineno,
            "request_id": request_id.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # 允许 extra={"user_id": ...} 富化字段
        for k, v in record.__dict__.items():
            if k not in payload and not k.startswith("_") and k not in (
                "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name", "message",
                "asctime",
            ):
                payload[k] = v
        return json.dumps(payload, default=str)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
```

之后 `logger.info("order placed", extra={"order_id": 123, "amount": 99.99})` 就能在 Kibana 用 `level:INFO AND order_id:123` 直接查到。

### 4.3 保留分层与成本控制

日志体量长得比预算快。把存储分层：

| 分层 | 年龄 | 后端 | 用途 |
|------|------|------|------|
| 热 | 0–7 天 | SSD 节点 ES | 实时排障 |
| 温 | 7–30 天 | HDD 节点 | 近期调查 |
| 冷 | 30–90 天 | S3 + 可搜索快照 | 合规、临时查询 |
| 归档 | 90 天–7 年 | Glacier | 法律保全 |

ILM（索引生命周期管理）自动迁移。最大的成本杀手是**别索引你从不查询的字段**——在 mapping 里把它们标 `enabled: false`。

### 4.4 永远不要写进日志的内容

- 密码、API Key、JWT（连前缀也不要）。
- 完整信用卡号、SSN、原始生物特征。
- 没有保留策略的个人数据。
- 认证端点的请求体。

**一次日志事故就是一次数据泄露。**

## 5. 不抖动的弹性伸缩

### 5.1 三种风味

- **响应式**：根据观察到的负载（CPU、请求速率、队列深度）伸缩。默认必开。
- **定时**：在可预测的高峰前预扩（双 11 上午 9 点、凌晨批量任务）。
- **预测式**：基于历史数据用 ML 外推。当扩容耗时较长、流量爬坡很快时有用。

三种结合最好。预测式平滑均值，定时处理已知尖峰，响应式抓住意外。

### 5.2 一份正确的 K8s HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: api-hpa, namespace: production }
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 30
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 65 } }
    - type: Pods
      pods:
        metric: { name: http_requests_per_second }
        target: { type: AverageValue, averageValue: "100" }
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0          # 立即响应
      policies:
        - type: Percent, value: 100, periodSeconds: 30
        - type: Pods,    value: 4,   periodSeconds: 30
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300        # 缩容前等 5 分钟
      policies:
        - type: Percent, value: 25, periodSeconds: 60
```

不对称是关键：**扩得快、缩得慢**。下一波流量来之前过早缩容，下一次洪水就会落在过小的舰队上。

### 5.3 失效模式

- **抖动**：扩容触发指标下降，触发缩容，又触发扩容。解：稳定窗口、滞回。
- **踩踏**：50 个新 Pod 一起对数据库初始化同一份 JIT 缓存，把库压垮。解：温池、慢爬坡、连接限流。
- **min 太低**：凌晨三点你缩到 2 个 Pod；一个崩了；剩下那个在恢复中被打爆。`minReplicas` 至少要能扛掉一个节点。
- **没设 max**：跑飞的逻辑扩到 500 个 Pod 把账户烧穿。**永远设 `maxReplicas`。**

## 6. 成本优化

| 策略 | 典型节省 | 投入 | 风险 |
|------|---------|------|------|
| 规格回归（right-sizing） | 20–40% | 低 | 低，挑非高峰时段 |
| Savings Plans / RI（1–3 年） | 30–70% | 中 | 锁定，只锁基线部分 |
| Spot / 抢占式 | 最高 90% | 中 | 中断，仅适合容错负载 |
| 非生产环境定时关机 | 50–70% | 低 | dev/staging 几乎零风险 |
| 存储分层（S3 IA、Glacier） | 50–80% | 低 | 取回延迟 |
| Region 选择 | 最高 40% | 低 | 延迟、合规 |
| 出口流量优化（CDN、对等） | 视场景，常很大 | 中 | 无 |

一个抓最常见浪费——闲置 EC2——的脚本，每月跑一次值得：

```python
import boto3
from datetime import datetime, timedelta, timezone

ec2, cw = boto3.client("ec2"), boto3.client("cloudwatch")

def idle_instances(threshold_pct: float = 5.0, lookback_days: int = 14) -> list[dict]:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    out = []
    pages = ec2.get_paginator("describe_instances").paginate(
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}])
    for page in pages:
        for r in page["Reservations"]:
            for i in r["Instances"]:
                m = cw.get_metric_statistics(
                    Namespace="AWS/EC2",
                    MetricName="CPUUtilization",
                    Dimensions=[{"Name": "InstanceId", "Value": i["InstanceId"]}],
                    StartTime=start, EndTime=end,
                    Period=3600, Statistics=["Average"],
                )["Datapoints"]
                if not m:
                    continue
                avg = sum(p["Average"] for p in m) / len(m)
                if avg < threshold_pct:
                    out.append({
                        "id":   i["InstanceId"],
                        "type": i["InstanceType"],
                        "tag_name": next((t["Value"] for t in i.get("Tags", [])
                                          if t["Key"] == "Name"), "-"),
                        "avg_cpu": round(avg, 1),
                    })
    return out

if __name__ == "__main__":
    for inst in idle_instances():
        print(inst)
```

后续比脚本本身重要：把报告排进定时任务，按 `Owner` tag 路由给资源所有者，连续三个月还在的实例必须给出书面说明。

## 7. SRE：错误预算驱动工程优先级

![错误预算](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig5_error_budget.png)

Google SRE 那本书把复杂主题压成四个概念：

- **SLI**（Service Level Indicator）：跟用户体验挂钩的指标。可用性、延迟、新鲜度。
- **SLO**（Service Level Objective）：在某个窗口下 SLI 的目标。例如"30 天内 99.9% 的请求成功"。
- **错误预算** = `1 - SLO`。每个窗口内**约定可接受**的不可靠量。99.9% 在 30 天窗口下是 43 分钟。
- **燃烧速率**：预算被消耗的速度。

真正的创新在于**用错误预算来做决策**：

- **预算充裕（>50%）** -> 发新功能、敢冒险。
- **预算谨慎（20–50%）** -> 风险变更放慢，优先做稳定性工作。
- **预算紧张（<20%）** -> 功能冻结，强制做稳定性冲刺。
- **预算耗尽** -> 除了修复，谁也不能发；SRE 团队有权否决发布。

这让"可靠性"成为**共同**议题。预算用完时，产品不能再要功能；预算还很多时，SRE 也不能要 100% 可用。**让数字仲裁。**

### 7.1 SLO recording rules

```yaml
groups:
- name: slo
  interval: 30s
  rules:
  - record: slo:requests:rate5m
    expr: sum by (service) (rate(http_requests_total[5m]))

  - record: slo:errors:rate5m
    expr: sum by (service) (rate(http_requests_total{status=~"5.."}[5m]))

  - record: slo:availability:ratio_5m
    expr: 1 - (slo:errors:rate5m / slo:requests:rate5m)

  - record: slo:budget:remaining
    expr: |
      1 - (
        (1 - avg_over_time(slo:availability:ratio_5m[30d]))
        / (1 - 0.999)
      )
```

`slo:budget:remaining` 就是放在高管仪表盘的那个数字。它穿过 50%、20%、0% 时，对应的策略自动启动。

### 7.2 其他能复利累加的 SRE 实践

- **降低 toil。**追踪重复运维耗时，定上限（Google 的做法是 50%）。超出就用代码替代。
- **无指责复盘。**关注系统，不针对个人。问的是"什么条件让这个错误造成了伤害？"——而不是"是谁干的？"。行动项有人有截止日期；没落地的行动项本身就是事件。
- **Game day。**每季度有控制地故意搞坏一件事。告警响了吗？手册有用吗？有人及时发现了吗？
- **容量规划。**预测需求 + 余量；提前供给；演练扩容流程，用到时才真的能用。

## 8. GitOps：Git 是事实来源

```
开发 commit -> CI 构建镜像 -> CI 在配置仓库里更新 manifest
                                    |
                          ArgoCD / Flux 监听仓库
                                    |
                          把集群 reconcile 到 git 状态
```

GitOps 通过删掉一类**能力**（人能直接 `kubectl apply`），删掉了一类**错误**。集群自己 reconcile 到配置仓库里写的东西，唯一改集群的方式就是改 Git。

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata: { name: web, namespace: argocd }
spec:
  project: default
  source:
    repoURL:        https://github.com/company/k8s-config
    targetRevision: main
    path:           apps/web/overlays/production
  destination:
    server:    https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune:    true       # 删掉 git 中不再存在的资源
      selfHeal: true       # 把人手改的东西自动改回来
    retry:
      limit: 5
      backoff: { duration: 30s, maxDuration: 5m, factor: 2 }
```

这一切免费送你的属性：

- **回滚** = `git revert`。
- **审计** = `git log`。
- **Staging 与 Prod 的差异** = `git diff`。
- **灾难恢复** = "把 ArgoCD 指向同一个仓库，拉一个新集群"。

## 9. 运维检查单

**流水线**
- [ ] 每个变更都通过流水线进生产；不存在手工 `apply`。
- [ ] 云端鉴权用 OIDC，没有长期 secret。
- [ ] 质量门让构建失败；任何人都不能用 admin 越过。
- [ ] SLO 异常自动回滚。

**基础设施**
- [ ] 所有资源在 Terraform / 等价 IaC 里定义。
- [ ] 远程 state 带锁；每个服务每个环境一份 state 文件。
- [ ] 每个 PR 都贴出 `terraform plan`，review 看 plan 不只是看 HCL。
- [ ] 至少每天跑一次漂移检测。

**监控**
- [ ] 指标、日志、链路全跑通。
- [ ] 每个服务都有四个黄金信号（延迟、流量、错误、饱和度）的仪表盘。
- [ ] 告警基于燃烧速率，不基于裸阈值。
- [ ] 每条告警都有 runbook 链接；没有 runbook 就不告警。

**日志**
- [ ] 每个服务都发 JSON 结构化日志。
- [ ] Request ID 端到端透传。
- [ ] 配置保留分层；旧索引自动滚出。
- [ ] 写入前敏感字段已经脱敏。

**SRE**
- [ ] 每个服务发布 SLO，并有高管签字。
- [ ] 团队仪表盘上能看到错误预算。
- [ ] On-call 轮值有清晰的升级路径。
- [ ] SEV-1 / SEV-2 事故 5 个工作日内出复盘。

**成本**
- [ ] 每个资源都打 tag；成本仪表盘按团队拆分。
- [ ] 非生产环境业务时间外自动关机。
- [ ] 每月复审闲置 / 过大资源。

每一条没勾的，第一次咬人时都会让你赔上一周事故响应和一块高管信任。补齐它们的工作量很小，收益很大；唯一的障碍只是"还没紧急到必须停下手头活去做"。

---

## 系列导航

| 篇 | 主题 |
|----|------|
| 1 | [基础与架构体系](/zh/cloud-computing-fundamentals/) |
| 2 | [虚拟化技术深度解析](/zh/cloud-computing-virtualization/) |
| 3 | [存储系统与分布式架构](/zh/cloud-computing-storage-systems/) |
| 4 | [网络架构与 SDN](/zh/cloud-computing-networking-sdn/) |
| 5 | [安全与隐私保护](/zh/cloud-computing-security-privacy/) |
| **6** | **运维与 DevOps 实践（当前）** |
| 7 | [云原生与容器技术](/zh/cloud-computing-cloud-native-containers/) |
| 8 | [多云与混合架构](/zh/cloud-computing-multi-cloud-hybrid/) |
