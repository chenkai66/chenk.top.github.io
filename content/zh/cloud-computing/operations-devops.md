---
title: "云计算（七）：运维与 DevOps 实践"
date: 2023-05-26 09:00:00
tags:
  - Cloud Computing
  - DevOps
  - SRE
  - Infrastructure as Code
  - Monitoring
  - CI/CD
categories: 云计算
series: cloud-computing
lang: zh
mathjax: false
description: "工程师视角的 DevOps 实战：能把控质量的 CI/CD、可复现的 Terraform 基础设施、Prometheus + Grafana 监控、ELK / EFK 日志、SRE 错误预算，以及凌晨三点能撑住的运维习惯。"
disableNunjucks: true
series_order: 7
translationKey: "cloud-computing-7"
polished_by_qwen_max: true
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/illustration_1.png)

2017 年 GitLab 丢了六个小时的数据库状态。一位疲惫的工程师在事故处理中对错了服务器跑了 `rm -rf`。备份流程其实已经悄悄坏了几个月，但没人发现，因为没人在做恢复演练。教训不是"用 `rm` 要小心"。教训是：运维是一个**系统**——工具、运行手册、监控、自动化，以及围绕这一切的仪式。系统健康时，任何疲惫的工程师都无法导致生产环境崩溃；系统腐烂时，每一次深夜抢救都可能引发灾难。

本文将介绍如何构建这个系统，包括在代码触达用户前确保质量的 CI/CD、通过 IaC 将“生产环境”变成 Git 提交而非雪花服务器、区分噪声和信号的监控、可搜索的日志，以及将救火工程化的 SRE 实践——如错误预算、 SLO 和无指责复盘。

## 我会学到
- CI/CD 流水线：阶段、质量门、回滚，附完整 GitHub Actions 示例
- 使用 Terraform 实现基础设施即代码：工作流、状态管理、模块模式
- Prometheus + Grafana + Alertmanager 监控：抓取模型、 PromQL、告警规则
- 集中化日志架构（EFK / ELK）：采集器、缓冲、处理器、保留分层
- 响应真实负载且不抖动的弹性伸缩
- 无需重写应用的成本优化
- SRE 实践： SLI / SLO / 错误预算、无指责复盘、 GitOps

## 前置知识

- Linux 命令行熟练
- Git 和基本 CI/CD 概念
- 建议先阅读本系列前 5 篇

---

## 1. CI/CD 流水线：发布的"系统记录"

![CI/CD 流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig1_cicd_pipeline.png)

现代 CI/CD 流水线不仅是自动化，还是代码进入生产的唯一路径和每次发布的系统记录——记录谁发布了什么、经过哪些测试、使用哪个版本的基础设施及其后续情况。运维栈的其他部分都依赖这条主干。

### 1.1 八个阶段

| 阶段 | 目的 | 失效模式 |
|------|------|----------|
| Commit | push / merge 触发 | 不会失败，只是事件 |
| Build | 编译、打包、镜像 | 基础镜像锁定、依赖锁定 |
| 单元测试 | 逻辑快速反馈 | flaky 测试侵蚀信任，需隔离 |
| 安全扫描 | SAST、依赖 CVE、镜像扫描 | 噪声大，按仓库分级 |
| 部署到 Staging | 制品首次运行 | Staging 与 Prod 配置漂移 |
| Smoke / E2E | 跨服务契约 | 测试太慢会被跳过 |
| 部署到 Prod | 金丝雀 -> 全量 | 一次性铺开、缺自动回滚 |
| 验证 | 部署后 SLO 检查 | "肉眼"验证，未量化 |

### 1.2 一份真实的 GitHub Actions 流水线

```yaml
name: deploy
on:
  push:
    branches: [main]

permissions:
  id-token: write          # 使用 OIDC 实现云平台身份认证
  contents: read
  packages: write

env:
  REGISTRY: ghcr.io
  IMAGE:    ghcr.io/${{ github.repository }}

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: 设置 Go 环境
        uses: actions/setup-go@v5
        with: { go-version-file: go.mod }

      - name: 代码静态检查（Lint）
        uses: golangci/golangci-lint-action@v4

      - name: 单元测试
        run: go test -race -coverprofile=coverage.out ./...

      - name: 覆盖率门禁
        run: |
          pct=$(go tool cover -func=coverage.out | tail -1 | awk '{print $3}' | tr -d '%')
          echo "覆盖率：${pct}%"
          if (( $(echo "$pct < 80" | bc -l) )); then
            echo "::error::覆盖率 ${pct}% 低于 80% 门槛"
            exit 1
          fi

      - name: 构建并推送镜像
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login $REGISTRY -u ${{ github.actor }} --password-stdin
          docker build -t $IMAGE:${{ github.sha }} -t $IMAGE:latest .
          docker push $IMAGE --all-tags

  security-scan:
    needs: build-test
    runs-on: ubuntu-latest
    steps:
      - name: Trivy 镜像安全扫描
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.IMAGE }}:${{ github.sha }}
          severity: CRITICAL,HIGH
          exit-code: 1

      - name: SAST（源码审计）—— 使用 Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: p/owasp-top-ten

  deploy-staging:
    needs: security-scan
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::111122223333:role/deploy-staging
          aws-region: us-east-1

      - name: 部署至 ECS Staging 环境
        run: |
          aws ecs update-service --cluster staging --service web --force-new-deployment

      - name: 等待服务稳定
        run: aws ecs wait services-stable --cluster staging --services web

  smoke-test:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - name: 健康检查
        run: |
          for i in {1..10}; do
            status=$(curl -s -o /dev/null -w '%{http_code}' https://staging.example.com/healthz)
            if [ "$status" = "200" ]; then echo "Healthy"; exit 0; fi
            sleep 5
          done
          echo "::error::Staging 环境健康检查在 50 秒后失败"
          exit 1

      - name: API 合约测试
        run: |
          npm ci
          npx newman run tests/postman/smoke.json --environment tests/postman/staging.json

  deploy-prod:
    needs: smoke-test
    runs-on: ubuntu-latest
    environment: production       # 需人工审批
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::111122223333:role/deploy-prod
          aws-region: us-east-1

      - name: 金丝雀发布（10% 流量）
        run: |
          aws ecs update-service --cluster prod --service web \
            --deployment-configuration '{
              "deploymentCircuitBreaker": {"enable": true, "rollback": true},
              "maximumPercent": 200,
              "minimumHealthyPercent": 100
            }' \
            --force-new-deployment

      - name: 验证金丝雀 SLO（5 分钟）
        run: |
          sleep 300
          # 通过 CloudWatch 检查 error_rate；若超过 1%，自动回滚
```

有三点设计决策值得强调：  
第一，**使用 OIDC 替代长期有效的密钥**——`id-token: write` 权限配合 `role-to-assume`，使 GitHub Actions 可直接扮演 IAM 角色，彻底消除 AWS 访问密钥在 GitHub 中的硬编码或明文存储。  
第二，**生产环境部署需人工审批**——在 staging 验证通过后，必须由运维或负责人手动点击确认，才能触发 prod 部署，形成关键的人为把关节点。  
第三，**金丝雀发布具备自动回滚能力**——若新版本在前 5 分钟内错误率突破 1%，系统将立即终止流量导入并回退至旧版本，无需等待人工响应。

### 1.3 不同部署策略对比

并非所有服务都适合金丝雀发布。选择哪种策略，取决于流量规模、回滚代价，以及你检测异常发布的速度。

| 策略 | 工作原理 | 回滚速度 | 适用场景 |
|------|-----------|------------|-------------|
| **滚动更新（Rolling）** | 分批替换实例（如每次替换 20%） | 分钟级（暂停发布 + 重推旧镜像） | 无状态服务，且已配置健康检查 |
| **蓝绿部署（Blue/Green）** | 并行运行两套完整环境，通过 DNS 或负载均衡器切换流量 | 秒级（一键切回） | 有状态服务、数据库变更、或要求零停机与瞬时回滚 |
| **金丝雀发布（Canary）** | 将一小部分真实流量导向新版本，逐步放大 | 秒级（快速摘除金丝雀实例） | 高流量服务，且能快速采集并判断 SLO （如错误率、延迟） |
| **功能开关（Feature Flags）** | 代码先上线但默认关闭，再按用户群/百分比/条件动态启用 | 毫秒级（服务端开关即刻生效） | 用户可见功能、 A/B 测试、渐进式能力开放 |
| **重建式（Recreate）** | 先停全部旧实例，再启全部新实例 | 较慢（需完整重启） | 开发/测试环境、批处理任务、单实例有状态服务 |

实践中，这些策略常组合使用：以金丝雀方式部署服务；该金丝雀版本内部又通过功能开关，仅向 1% 用户开放新逻辑；而承载功能开关能力的服务本身，也采用蓝绿部署保障其高可用与可回滚性。
![部署策略对比：滚动更新、蓝绿、金丝雀](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig2_deployment_strategies.png)

滚动更新逐个替换 v1 Pod；蓝绿在负载均衡层一次切换；金丝雀先放小流量再扩大。三种方式各有适用场景，实际项目里经常组合使用。

---

## 2. 使用 Terraform 实现基础设施即代码（IaC）

基础设施即代码（IaC）意味着生产环境通过版本受控的代码文件来定义，而不是依赖某位工程师上周二在控制台里点出来的临时状态。 Terraform 是目前最主流的 IaC 工具，因为它与云厂商无关、采用声明式语法，并提供关键的预览步骤（`plan`），可在任何变更实际发生前清晰展示「将要创建 / 修改 / 销毁哪些资源」。

### 2.1 核心工作流

```
terraform init      # 下载 Provider 插件，配置远程后端
terraform plan      # 预览：即将创建 / 修改 / 销毁的资源清单
terraform apply     # 执行变更，使真实环境与代码保持一致
terraform destroy   # 彻底销毁全部资源（仅限非生产环境！）
```

真正日常高频使用的其实是 `plan` —— 它让你在任何变更落地前就看清影响范围。代码评审应基于 `plan` 的输出结果，而不仅是 HCL 源码本身。

```bash
terraform plan -out=tfplan -input=false
terraform show -no-color tfplan > plan.txt
terraform apply -input=false tfplan
```

### 2.2 状态管理（State Management）

Terraform 的状态文件（`terraform.tfstate`）是连接 HCL 代码与云上真实资源的唯一映射。一旦状态管理出错，就会引发环境漂移、并发冲突，甚至误删关键基础设施。

必须遵守以下原则：

- **强制使用带锁机制的远程后端**： AWS 推荐 S3 + DynamoDB， GCP 推荐 GCS，跨云场景可选 Terraform Cloud。**绝对禁止**将 state 文件提交至 Git —— 它可能包含密钥，且极易引发合并冲突。
- **每个服务、每个环境独享一个 state 文件**：例如 `services/web/prod/terraform.tfstate`。这能有效控制故障爆炸半径，并支持多环境并行部署。
- **状态文件静态加密**： S3 服务端加密（SSE）为最低要求。
- **严格限制 state 访问权限**：开发人员可在本地执行 `plan`，但 `apply` 必须由 CI/CD 流水线统一触发和执行。

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "company-terraform-state"
    key            = "services/web/prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```
![Terraform 工作流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig_terraform_pipeline_zh.png)

`plan` 是关键一步。它提前告诉你变更内容。 Code review 看的是 plan 输出，不只是 HCL。

### 2.3 完整的生产级模块

模块是可复用、可测试的基础设施单元。设计良好的模块应封装一种服务模式，使团队在使用时无需了解底层实现细节。

```hcl
# modules/ecs-service/main.tf
variable "name"          { type = string }
variable "environment"   { type = string }
variable "image"         { type = string }
variable "cpu"           { type = number, default = 256 }
variable "memory"        { type = number, default = 512 }
variable "desired_count" { type = number, default = 2 }
variable "health_path"   { type = string, default = "/healthz" }

resource "aws_ecs_task_definition" "this" {
  family                   = "${var.name}-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = var.name
    image     = var.image
    essential = true
    portMappings = [{ containerPort = 8080, protocol = "tcp" }]
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8080${var.health_path} || exit 1"]
      interval    = 15
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

resource "aws_ecs_service" "this" {
  name            = "${var.name}-${var.environment}"
  cluster         = data.aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
}
```

### 2.4 漂移检测

即使采用基础设施即代码（IaC），实际环境仍可能与代码定义发生偏离：例如，有人为快速修复线上故障而直接在控制台操作，却忘了将变更同步回代码；又或配置被手动覆盖而未走 CI/CD 流水线。定期执行漂移检查，可在问题累积前及时发现并干预。

```bash
#!/bin/bash
# drift-check.sh -- 通过 cron 或调度流水线每日执行
set -euo pipefail

SERVICES="web api worker"
ENVS="staging prod"

for svc in $SERVICES; do
  for env in $ENVS; do
    dir="services/${svc}/${env}"
    cd "$dir"
    terraform init -backend=true -input=false > /dev/null
    if ! terraform plan -detailed-exitcode -input=false > /dev/null 2>&1; then
      echo "DRIFT DETECTED in ${dir}"
      terraform plan -no-color -input=false > "/tmp/drift-${svc}-${env}.txt"
      curl -X POST "$SLACK_WEBHOOK" -d "{\"text\":\"Drift detected in ${dir}.\"}"
    fi
    cd - > /dev/null
  done
done
```

`-detailed-exitcode` 参数使 `terraform plan` 在检测到差异时返回退出码 `2`，该行为被用于触发后续告警逻辑。
---

## 3. 使用 Prometheus、 Grafana 和 Alertmanager 实现监控

任何生产系统都离不开可观测性的三大支柱：**指标（Metrics）**（随时间变化的数值）、**日志（Logs）**（带上下文的事件）和 **链路追踪（Traces）**（跨服务的请求路径）。本节聚焦指标监控；日志部分将在下一节展开。

### 3.1 Prometheus 的拉取模型（Scrape Model）

Prometheus 采用主动**拉取（Pull）**方式从目标服务采集指标，而非依赖服务主动推送（Push）。该模型具备两大优势：  
- 支持多实例拉取同一目标，天然适配高可用（HA）部署；  
- 若服务崩溃，仅表现为“不再被拉取”，不会残留无效的推送连接，更健壮。

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: "kubernetes-pods"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      # 仅拉取带有 annotation prometheus.io/scrape=true 的 Pod
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      # 将 annotation prometheus.io/path 映射为 metrics 路径
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      # 将 annotation prometheus.io/port 与 Pod IP 组合成地址：${pod_ip}:${port}
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port, __meta_kubernetes_pod_ip]
        action: replace
        target_label: __address__
        regex: (.+);(.+)
        replacement: $2:$1
```

### 3.2 四大黄金信号（Golden Signals，使用 PromQL 表达）

Google SRE 手册提出，每个服务都应持续关注以下四个核心指标——即“四大黄金信号”。 Prometheus 可通过简洁的 PromQL 快速实现：

| 信号 | 关注点 | PromQL 示例 |
|------|--------|-------------|
| **延迟（Latency）** | 成功请求的耗时分布 | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{status!~"5.."}[5m]))` |
| **流量（Traffic）** | 单位时间内的请求数（QPS） | `sum(rate(http_requests_total[5m])) by (service)` |
| **错误（Errors）** | 失败请求占总请求的比例 | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |
| **饱和度（Saturation）** | 服务资源使用程度（如内存/ CPU 占比） | `container_memory_working_set_bytes / container_spec_memory_limit_bytes` |

### 3.3 应用程序埋点（Instrumentation）

每个服务需暴露 `/metrics` 接口供 Prometheus 抓取。以 Go 为例，只需少量代码即可完成基础埋点：

```go
package main

import (
    "net/http"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    httpDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "Duration of HTTP requests.",
            Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
        },
        []string{"method", "route", "status"},
    )
    httpTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests.",
        },
        []string{"method", "route", "status"},
    )
)

func init() {
    prometheus.MustRegister(httpDuration, httpTotal)
}

func instrumentHandler(route string, next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        rw := &responseWriter{ResponseWriter: w, statusCode: 200}
        next(rw, r)
        duration := time.Since(start).Seconds()
        status := http.StatusText(rw.statusCode)
        httpDuration.WithLabelValues(r.Method, route, status).Observe(duration)
        httpTotal.WithLabelValues(r.Method, route, status).Inc()
    }
}

func main() {
    http.Handle("/metrics", promhttp.Handler())
    http.HandleFunc("/api/orders", instrumentHandler("/api/orders", handleOrders))
    http.ListenAndServe(":8080", nil)
}
```
![监控的四个黄金信号：延迟、流量、错误、饱和度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig3_four_golden_signals.png)

Google SRE 把可观测性归结到四个信号——延迟、流量、错误、饱和度。覆盖这四个，你就能回答：服务慢吗、忙吗、坏了吗、满了吗。 Prometheus 用 PromQL 表达它们都是几行的事。

### 3.4 不无故打扰人的告警规则

目标是：零误报告警。每一次触发的告警，都必须需要人工介入；否则，它就该是一个仪表盘面板，而不是一条告警（page）。

```yaml
# alerts/slo.yml
groups:
  - name: slo-burn-rate
    rules:
      # 多时间窗口、多燃烧速率告警（源自 Google SRE 手册的经典模式）
      # 快速燃烧：1 小时内错误预算燃烧速率达 14.4 倍，且 6 小时内达 6 倍
      - alert: HighErrorBurnRate_Critical
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[1h]))
            / sum(rate(http_requests_total[1h]))
          ) > (14.4 * 0.001)
          and
          (
            sum(rate(http_requests_total{status=~"5.."}[6h]))
            / sum(rate(http_requests_total[6h]))
          ) > (6 * 0.001)
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "错误预算燃烧过快——预计 < 1 天耗尽"
          runbook: "https://wiki.internal/runbooks/high-error-rate"
          dashboard: "https://grafana.internal/d/slo-overview"

      # 缓慢燃烧：1 天内达 3 倍，且 3 天内达 1 倍
      - alert: HighErrorBurnRate_Warning
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[1d]))
            / sum(rate(http_requests_total[1d]))
          ) > (3 * 0.001)
          and
          (
            sum(rate(http_requests_total{status=~"5.."}[3d]))
            / sum(rate(http_requests_total[3d]))
          ) > (1 * 0.001)
        for: 15m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "错误预算缓慢燃烧——预计 < 10 天耗尽"
          runbook: "https://wiki.internal/runbooks/high-error-rate"

      - alert: HighLatency_P99
        expr: |
          histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))
          > 2.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.service }} 的 P99 延迟超过 2 秒"
          runbook: "https://wiki.internal/runbooks/high-latency"
```

Google SRE 手册中提出的「多时间窗口 + 多燃烧速率」告警模式，是你能对告警系统做出的最重要改进。它不再简单地监控「错误率 > 1%」这类易受瞬时抖动干扰的阈值，而是聚焦于「错误预算的实际消耗速率」。例如： 2% 的错误率持续 30 秒，对月度错误预算几乎毫无影响；但 0.5% 的错误率若持续 3 天，则足以彻底耗尽整个月的预算。燃烧速率模型能同时捕获这两种关键场景。

### 3.5 Alertmanager 路由配置

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  receiver: "default-slack"
  group_by: ["alertname", "service"]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

  routes:
    - match:
        severity: critical
      receiver: "pagerduty-critical"
      repeat_interval: 15m

    - match:
        severity: warning
      receiver: "slack-warnings"
      repeat_interval: 4h

receivers:
  - name: "pagerduty-critical"
    pagerduty_configs:
      - routing_key: "<pagerduty-integration-key>"
        severity: critical

  - name: "slack-warnings"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/xxx"
        channel: "#alerts-warnings"
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: "default-slack"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/xxx"
        channel: "#alerts-default"
```
---

## 4. 集中式日志栈（EFK / ELK）

指标告诉你**出了问题**，日志则告诉你**为什么出问题**。一个集中式日志栈会从所有服务中采集、处理、索引并长期保留日志，统一存入一个可全文检索的存储中心。

### 4.1 架构概览

当前主流的两种日志栈是 **ELK**（Elasticsearch、 Logstash、 Kibana）和 **EFK**（Elasticsearch、 Fluentd / Fluent Bit、 Kibana）。二者核心区别在于日志采集器（shipper）：  
- Logstash 基于 JVM，功能强大但资源开销大；  
- Fluent Bit 用 C 编写，轻量高效，非常适合以 DaemonSet 方式在 Kubernetes 中部署。

```
+-----------+     +------------+     +---------------+     +---------+
| 应用 Pod  | --> | Fluent Bit | --> | Elasticsearch | --> | Kibana  |
| (stdout)  |     | (DaemonSet)|     | (3 节点高可用) |     | (查询界面) |
+-----------+     +------------+     +---------------+     +---------+
                        |
                        v
                  +------------+
                  |   Kafka    |  （可选：用于应对日志洪峰的缓冲层）
                  +------------+
```

### 4.2 结构化日志

提升日志可检索性的最关键实践，就是**统一使用 JSON 格式输出日志**。结构化日志是一条可直接查询的文档；而非结构化日志只是一段字符串，必须依赖正则表达式解析，效率低且不可靠。

```python
import structlog
import uuid

# 配置 structlog 输出 JSON 格式日志
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
)

log = structlog.get_logger()

def handle_request(request):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger = log.bind(
        request_id=request_id,
        method=request.method,
        path=request.path,
        user_id=request.user.id if request.user else None,
    )

    logger.info("request_started")
    try:
        result = process(request)
        logger.info("request_completed", status=200, duration_ms=result.duration_ms)
        return result
    except ValidationError as e:
        logger.warning("validation_failed", error=str(e), status=400)
        raise
    except Exception as e:
        logger.error("request_failed", error=str(e), status=500, exc_info=True)
        raise
```

该代码生成的日志样例如下：

```json
{
  "timestamp": "2024-03-15T14:30:22.123Z",
  "level": "info",
  "event": "request_completed",
  "request_id": "abc-123",
  "method": "POST",
  "path": "/api/orders",
  "user_id": "u_456",
  "status": 200,
  "duration_ms": 42
}
```
![集中式日志管道：从应用到 Elasticsearch / Kibana](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig4_logging_pipeline.png)

应用 stdout -> Fluent Bit DaemonSet -> （可选） Kafka 缓冲 -> （可选） Logstash 富化 -> Elasticsearch -> Kibana 查询。每多一跳就多一份延迟，但也多一份韧性。规模小时可以省掉 Kafka 和 Logstash，等日志量超过采集器再加。

### 4.3 Kubernetes 中的 Fluent Bit 配置

```ini
# fluent-bit.conf
[SERVICE]
    Flush         5
    Daemon        Off
    Log_Level     info
    Parsers_File  parsers.conf

[INPUT]
    Name              tail
    Path              /var/log/containers/*.log
    Parser            cri
    Tag               kube.*
    Mem_Buf_Limit     50MB
    Skip_Long_Lines   On
    Refresh_Interval  10

[FILTER]
    Name                kubernetes
    Match               kube.*
    Kube_URL            https://kubernetes.default.svc:443
    Kube_Tag_Prefix     kube.var.log.containers.
    Merge_Log           On
    Keep_Log            Off
    K8S-Logging.Parser  On

[FILTER]
    Name    modify
    Match   kube.*
    # 在写入索引前脱敏敏感字段
    Remove  password
    Remove  authorization
    Remove  cookie
    Remove  x-api-key

[OUTPUT]
    Name            es
    Match           kube.*
    Host            elasticsearch.logging.svc.cluster.local
    Port            9200
    Logstash_Format On
    Logstash_Prefix k8s-logs
    Retry_Limit     5
    tls             On
    tls.verify      On
```

### 4.4 日志保留分层策略

日志的存储与索引成本较高，采用分层策略可在可检索性与成本之间取得平衡。

| 层级 | 保留时长 | 存储介质 | 典型用途 |
|------|----------|----------|----------|
| **热层（Hot）** | 0–7 天 | SSD 支持的 Elasticsearch | 实时调试、故障响应 |
| **温层（Warm）** | 7–30 天 | HDD 支持的 Elasticsearch | 近期调查、合规性查询 |
| **冷层（Cold）** | 30–90 天 | S3 / GCS + Elasticsearch 快照 | 审计、低频检索 |
| **冻结层（Frozen）** | 90 天 – 7 年 | S3 Glacier / Archive | 合规性长期留存（如 HIPAA、 PCI） |

通过 Elasticsearch 的索引生命周期管理（ILM）自动执行层级迁移：

```json
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": { "max_size": "50gb", "max_age": "1d" },
          "set_priority": { "priority": 100 }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "shrink": { "number_of_shards": 1 },
          "forcemerge": { "max_num_segments": 1 },
          "set_priority": { "priority": 50 }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "searchable_snapshot": { "snapshot_repository": "s3-logs" },
          "set_priority": { "priority": 0 }
        }
      },
      "delete": { "min_age": "365d", "actions": { "delete": {} } }
    }
  }
}
```
---

## 5. 弹性伸缩：响应真实负载

自动伸缩听起来很简单：负载升高时扩容，负载下降时缩容。但在实际落地中，要真正做好，需要选对伸缩信号、调优响应速度，并避免反复震荡（flapping）。

### 5.1 伸缩信号选择

| 信号 | 适用场景 | 注意事项 |
|------|----------|-----------|
| **CPU 使用率** | 通用型指标；推荐作为默认选择 | 流量突刺型业务易引发震荡 |
| **内存使用率** | 内存敏感型服务（如缓存、 JVM 应用） | 内存释放滞后（受 GC 影响），缩容响应慢 |
| **请求速率（RPS）** | Web 服务，且单请求资源开销较稳定 | 需额外搭建自定义指标采集与上报链路 |
| **队列深度** | 异步工作节点、批处理任务 | 应基于 *队列增长速率* 而非绝对深度触发伸缩 |
| **自定义业务指标** | 当上述指标均无法反映真实用户体验时（如支付成功率、首屏加载时长） | 需投入可观的埋点与监控体系建设成本 |

### 5.2 基于自定义指标的 Kubernetes HPA 配置

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web
  minReplicas: 3
  maxReplicas: 50
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60    # 扩容前等待 1 分钟确认趋势
      policies:
        - type: Percent
          value: 100                     # 单次最多扩容 100%（即翻倍）
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300   # 缩容前等待 5 分钟冷静期
      policies:
        - type: Percent
          value: 10                      # 每轮最多缩容 10%
          periodSeconds: 60
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
```

这种「快扩慢缩」的设计是有意为之：  
- **快速扩容**（60 秒内翻倍）可及时应对突发流量，保障用户体验；  
- **保守缩容**（5 分钟冷静期 + 每分钟最多缩 10%）则有效规避“锯齿震荡”——即因过度缩容导致 CPU 突升、触发再次扩容、继而反复循环的问题。

### 5.3 预测式伸缩（Predictive Scaling）

对于具备明显周期性规律的工作负载（例如每日正午流量高峰的电商网站），纯响应式伸缩总是滞后的：等 CPU 达到 70% 时，用户早已感知到延迟。 AWS 的预测式伸缩（Predictive Scaling）和 GCP 的定时伸缩（Scheduled Scaling）正是为此而生：

```bash
# AWS Auto Scaling 预测策略配置示例
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name web-asg \
  --policy-name predictive-scaling \
  --policy-type PredictiveScaling \
  --predictive-scaling-configuration '{
    "MetricSpecifications": [{
      "TargetValue": 70,
      "PredefinedMetricPairSpecification": {
        "PredefinedMetricType": "ASGCPUUtilization"
      }
    }],
    "Mode": "ForecastAndScale",
    "SchedulingBufferTime": 300
  }'
```

其中 `SchedulingBufferTime: 300` 表示系统会在预测到流量高峰前 **5 分钟** 启动新实例，确保它们完成初始化、预热及健康检查，从容承接真实流量。
![HPA 非对称伸缩：快速扩容、慢速缩容以避免抖动](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig5_autoscaling_curve.png)

扩容快（60 秒翻倍）保护用户体验，缩容慢（每分钟 10%、 5 分钟冷却）避免锯齿抖动。这种非对称是刻意设计的。

---

## 6. 不重写应用也能优化云成本

云账单总会让人措手不及。好消息是：大多数云账单中 30–50% 的费用，无需修改一行应用代码即可削减——关键在于资源规格调优、启停调度和承诺型购买。

### 6.1 成本优化优先级金字塔

自上而下逐层推进；越靠上的层级，收益越高、实施难度越低：

1. **关停闲置资源**：持续运行的开发/测试环境、孤立的 EBS 卷、未绑定的弹性 IP、后端无目标实例的负载均衡器。
2. **非生产环境定时启停**：开发集群无需凌晨三点在线。按工作日每天运行 10 小时计算，可节省 65% 费用。
3. **精准规格调优（Rightsizing）**：多数实例 CPU 和内存长期低于 20%，却配置了 2–4 倍冗余容量。利用 CloudWatch 或 Cloud Monitoring 数据识别低利用率实例。
4. **对容错型负载启用 Spot 实例 / 预 emptible 实例**： CI 构建节点、批处理任务、开发环境等场景适用，成本可降 60–90%。
5. **对稳态生产负载采用预留实例（RI）或 Savings Plans**： 1 年期“零预付”承诺即可节省 30–40%，风险极低。
6. **存储分层归档**：将访问频次低的数据迁移至更经济的存储层（如 S3 IA、 Glacier、 Archive）。

### 6.2 自动化成本管控脚本

```bash
# shutdown-nonprod.sh —— cron: 0 19 * * 1-5（工作日每晚 7 点）
#!/bin/bash
set -euo pipefail

ENVS=("dev" "staging" "qa")

for env in "${ENVS[@]}"; do
  echo "正在关闭 ${env} 环境的 ECS 服务..."
  for svc in $(aws ecs list-services --cluster "$env" --query 'serviceArns[]' --output text); do
    aws ecs update-service --cluster "$env" --service "$svc" --desired-count 0
  done

  echo "正在停止 ${env} 环境的 RDS 实例..."
  for db in $(aws rds describe-db-instances --query "DBInstances[?TagList[?Key=='Environment'&&Value=='${env}']].DBInstanceIdentifier" --output text); do
    aws rds stop-db-instance --db-instance-identifier "$db" || true
  done
done

echo "非生产环境已关闭，时间：$(date)"
```

```bash
# startup-nonprod.sh —— cron: 0 8 * * 1-5（工作日每天早 8 点）
#!/bin/bash
set -euo pipefail

ENVS=("dev" "staging" "qa")

for env in "${ENVS[@]}"; do
  echo "正在启动 ${env} 环境的 RDS 实例..."
  for db in $(aws rds describe-db-instances --query "DBInstances[?TagList[?Key=='Environment'&&Value=='${env}']].DBInstanceIdentifier" --output text); do
    aws rds start-db-instance --db-instance-identifier "$db" || true
  done

  echo "正在恢复 ${env} 环境的 ECS 服务..."
  for svc in $(aws ecs list-services --cluster "$env" --query 'serviceArns[]' --output text); do
    aws ecs update-service --cluster "$env" --service "$svc" --desired-count 2
  done
done

echo "非生产环境已启动，时间：$(date)"
```

### 6.3 成本分摊标签策略

标签是唯一能将云支出精准归属到团队的方式。没有标签，月度账单只是一串无法归因、无法追责的数字。

| 标签键（Tag key） | 示例值 | 用途 |
|------------------|--------|------|
| `Environment` | `prod`, `staging`, `dev` | 用于非生产环境的启停调度过滤 |
| `Team` | `platform`, `payments`, `ml` | 按团队维度进行成本分摊 |
| `Service` | `web`, `api`, `worker` | 统计各微服务的成本消耗 |
| `CostCenter` | `CC-1234` | 与财务系统中的成本中心映射 |
| `ManagedBy` | `terraform`, `manual` | 快速识别未纳入 IaC 管理的资源 |

通过 SCP （AWS）或组织政策（GCP）强制打标：

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "RequireTags",
    "Effect": "Deny",
    "Action": ["ec2:RunInstances", "rds:CreateDBInstance", "s3:CreateBucket"],
    "Resource": "*",
    "Condition": {
      "Null": {
        "aws:RequestTag/Environment": "true",
        "aws:RequestTag/Team": "true",
        "aws:RequestTag/Service": "true"
      }
    }
  }]
}
```
---

## 7. SRE 实践：从救火到工程

站点可靠性工程（SRE）是一种将运维工作视为软件工程问题的实践方法。其核心工具包括服务等级目标（SLO）、错误预算（Error Budget）和无指责复盘（Blameless Postmortem）。

### 7.1 SLI、 SLO 与错误预算

- **SLI**（Service Level Indicator，服务等级指标）：对服务行为的量化度量。例如：“300 毫秒内完成的请求占比。”  
- **SLO**（Service Level Objective，服务等级目标）： SLI 的目标值。例如：“滚动 30 天窗口内， 99.9% 的请求在 300 毫秒内完成。”  
- **错误预算**（Error Budget）： SLO 的补集。若 SLO 为 99.9%，则错误预算为 0.1% —— 即每月最多允许约 43 分钟的服务不可用时间。

```yaml
# SLO 定义（以 Prometheus 实践为例）
# 99.9% 可用性 = 0.1% 错误预算 = 每月 43.2 分钟

# 可用性 SLI
- record: sli:availability:ratio_rate5m
  expr: |
    sum(rate(http_requests_total{status!~"5.."}[5m]))
    / sum(rate(http_requests_total[5m]))

# 延迟 SLI（响应时间 < 300ms 的请求占比）
- record: sli:latency:ratio_rate5m
  expr: |
    sum(rate(http_request_duration_seconds_bucket{le="0.3"}[5m]))
    / sum(rate(http_request_duration_seconds_count[5m]))

# 剩余错误预算（30 天窗口）
- record: error_budget:remaining
  expr: |
    1 - (
      (1 - avg_over_time(sli:availability:ratio_rate5m[30d]))
      / (1 - 0.999)
    )
```

### 7.2 值班机制与升级流程

一个健康的值班轮值机制应具备以下特征：

| 维度 | 良好实践 | 不良实践 |
|------|----------|----------|
| 轮值周期 | 1 周 | 1 个月（易导致倦怠） |
| 团队规模 | 6–8 人（每人每 6–8 周轮值一次） | 2–3 人（长期高频值班） |
| 每班告警数 | 0–2 条可操作告警 | 每班超 10 条（告警疲劳，关键告警易被忽略） |
| 交接方式 | 每次换班提供书面交接文档 | 仅口头交接：“一切正常” |
| 补偿机制 | 调休或额外值班津贴 | “这是岗位职责的一部分” |
| 升级路径 | 明确层级：主责人 → 备岗人 → 工程经理 → 技术副总裁 | “随便打给谁，有人接就行” |
![30 天 SLO 窗口的错误预算燃尽曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig6_error_budget.png)

错误预算是管理工具，不只是指标。预算充足时团队放手发版；预算见底时冻结新功能、专攻稳定性。把『快 vs. 稳』的争论换成一个所有人都看得到的数字。

### 7.3 无指责复盘（Blameless Postmortem）

将事故归咎于某个人的复盘，只会教会其他人如何掩盖自己的错误。以下为无指责复盘（Blameless Postmortem）模板：

```markdown
## 事故报告：[标题]
**日期**：YYYY-MM-DD  
**持续时间**：X 小时 Y 分钟  
**严重等级**：SEV-1 / SEV-2 / SEV-3  
**撰写人**：[姓名]  
**评审人**：[姓名列表]

### 概述  
一句话说明发生了什么；一句话说明对用户的影响；一句话说明如何恢复。例如：数据库连接池耗尽导致核心 API 错误率飙升，影响约 5% 的用户请求，持续 25 分钟；通过扩容连接池并重启服务完成修复。

### 时间线（所有时间均为 UTC）
- 14:00 — 监控告警触发：错误率 > 5%  
- 14:03 — 值班工程师响应并启动排查  
- 14:15 — 定位根因：数据库连接池耗尽  
- 14:20 — 实施缓解措施：扩大连接池容量、重启服务  
- 14:25 — 错误率回落至基线水平  
- 14:30 — 正式宣告事故结束  

### 根因分析  
连接池初始配置为 50 个连接。两天前上线的新批处理任务，在执行期间平均持有连接达 30 秒（远超预期的 100ms），在业务高峰时段迅速耗尽全部连接。

### 影响范围  
- 高错误率持续 25 分钟（峰值达 12%）  
- 约 3,200 次请求失败  
- 无数据丢失  

### 做得好的地方  
- 监控告警在问题发生后 3 分钟内准确触发  
- 值班工程师具备完整权限，可独立完成诊断与处置  

### 待改进之处  
- 新批处理任务未针对共享数据库开展负载测试  
- 连接池缺乏熔断机制（circuit breaker）  

### 改进项（Action Items）  
| 改进项 | 负责人 | 优先级 | 截止日期 |  
|--------|--------|--------|----------|  
| 在 Grafana 仪表盘中增加连接池关键指标（如活跃连接数、等待队列长度） | Alice | P1 | 2024-02-01 |  
| 对所有批处理任务开展共享资源（尤其是数据库）的压测验证 | Bob | P1 | 2024-02-15 |  
| 为数据库连接池实现熔断机制（例如 Hystrix 或 Resilience4j 集成） | Carol | P2 | 2024-03-01 |  
| 编写《连接池耗尽》标准化应急手册（Runbook），纳入内部知识库 | Dave | P2 | 2024-02-01 |  
```

**改进项是复盘的唯一目的。没有明确 Action Items 的复盘，只是一则故事，而非工程改进。**
![事故响应时间线：检测、分诊、修复、复盘](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig7_incident_timeline.png)

从告警触发到恢复基线再到复盘行动项落地——MTTR 是发布速度之外最值得追踪的运维指标。无指责复盘的核心是写下『如何避免下次』，而不是『是谁干的』。

---

## 8. GitOps：以 Git 作为唯一真实源

![基于 ArgoCD/Flux 的 GitOps](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig_gitops_argocd_zh.png)

GitOps 删掉了一类能力（直接 `kubectl apply`），也删掉了一类错误。集群自动 reconcile 到配置仓库的内容，改集群的唯一方法是改 Git。

### 8.1 ArgoCD Application 清单

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
      prune:    true       # 删掉 git 中移除的资源
      selfHeal: true       # 自动修复手动改动
    retry:
      limit: 5
      backoff: { duration: 30s, maxDuration: 5m, factor: 2 }
```

免费获得的特性：

- **回滚** = `git revert`。
- **审计** = `git log`。
- **Staging 和 Prod 差异** = `git diff`。
- **灾难恢复** = "新集群指向同一个仓库， ArgoCD 自动同步"。

### 8.2 GitOps 仓库结构

```
k8s-config/
  apps/
    web/
      base/
        deployment.yaml
        service.yaml
        hpa.yaml
        kustomization.yaml
      overlays/
        staging/
          kustomization.yaml      # 补丁：副本数、资源限制
          ingress-patch.yaml      # 预发布环境域名
        production/
          kustomization.yaml      # 补丁：更高副本数、更多资源
          ingress-patch.yaml      # 生产环境域名
    api/
      base/
        ...
      overlays/
        ...
  platform/
    monitoring/
      prometheus/
      grafana/
    logging/
      fluent-bit/
      elasticsearch/
```

该基于 Kustomize 的目录结构实现了「一次定义、多处复用」：通用配置集中在 `base/`，环境差异化通过 `overlays/` 中的补丁实现。任何修改 `apps/web/overlays/production/` 的 PR 都属于生产变更，需接受与应用代码同等严格的设计评审与代码审查。

### 8.3 GitOps 与传统 CI/CD 对比

| 维度 | 传统 CI/CD | GitOps |
|------|-------------|--------|
| 部署触发方式 | 流水线主动推送至集群 | 集群主动从 Git 拉取状态（Pull 模式） |
| 漂移检测 | 依赖人工检查或缺失检测能力 | 持续比对并自动修复（自愈） |
| 回滚操作 | 重新运行历史流水线版本 | 执行 `git revert` 或切换 Git 分支/提交 |
| 权限控制 | CI 系统需具备集群管理员权限 | 仅 ArgoCD 需集群访问权限，其余角色通过 Git 权限隔离 |
| 审计追踪 | 依赖 CI 日志（可能被轮转清理） | 完整保留在 Git 历史中（不可篡改、永久留存） |
| 多集群管理 | 每个集群需独立维护一套流水线 | 单一代码仓库 + 多个 ArgoCD 实例，统一管控 |
---

## 9. 故障排查指南：当问题发生时

每位运维工程师都需要一套应对凌晨三点告警的快速决策流程。以下是一套经过实战验证的实用方法。

### 9.1 黄金五分钟响应流程

```bash
# 1. 具体现象是什么？先看仪表盘。
open https://grafana.internal/d/overview

# 2. 是单个服务异常，还是全局性故障？
kubectl get pods -A | grep -v Running
kubectl top nodes
kubectl top pods --sort-by=cpu -A | head -20

# 3. 最近有哪些变更？
kubectl rollout history deployment/web -n production
git log --oneline --since="2 hours ago" -- apps/web/

# 4. 检查资源压力
kubectl describe nodes | grep -A5 "Conditions:"
df -h                           # 磁盘使用率
free -m                         # 内存使用情况
ss -tlnp                        # 监听端口与连接数

# 5. 查看受影响服务的近期日志
kubectl logs -l app=web -n production --tail=100 --since=5m
```

### 9.2 常见故障模式速查表

| 现象 | 最可能原因 | 排查方式 | 解决方案 |
|------|-------------|-----------|-----------|
| 发布后 5xx 错误激增 | 代码缺陷或配置错误 | `kubectl rollout undo deployment/web` | 立即回滚，离线深入分析 |
| 延迟缓慢上升 | 内存泄漏、连接泄漏 | 检查堆内存 / Goroutine 数量、连接池指标 | 重启 Pod，修复泄漏点 |
| 突发性 100% 错误率 | 依赖服务宕机（数据库、缓存、外部 API） | 检查依赖服务健康检查端点 | 启用熔断机制，配置降级逻辑 |
| Pod 持续崩溃重启（CrashLoopBackOff） | 内存溢出（OOM）、缺失配置、健康检查失败 | `kubectl describe pod <name>`，`kubectl logs <name> --previous` | 修正资源配置/限制，增加内存配额 |
| 节点状态为 NotReady | 磁盘压力、网络中断、 kubelet 崩溃 | `kubectl describe node`，`journalctl -u kubelet` | 驱逐节点负载并替换节点 |
| DNS 解析失败 | CoreDNS 过载或配置异常 | `kubectl logs -l k8s-app=kube-dns -n kube-system` | 扩容 CoreDNS 实例，检查 `ndots` 配置 |

### 9.3 数据库故障排查清单

```bash
# PostgreSQL：当前正在执行的慢查询（>5 秒）
SELECT pid, now() - pg_stat_activity.query_start AS duration,
       query, state, wait_event_type, wait_event
FROM pg_stat_activity
WHERE state != 'idle'
  AND (now() - pg_stat_activity.query_start) > interval '5 seconds'
ORDER BY duration DESC;

# PostgreSQL：按连接状态统计连接数
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state;

# PostgreSQL：表膨胀检查（死元组超 1 万）
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
       n_dead_tup,
       n_live_tup,
       ROUND(n_dead_tup * 100.0 / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC;

# MySQL：当前活跃查询
SHOW FULL PROCESSLIST;

# MySQL：InnoDB 引擎状态（锁、死锁、缓冲池）
SHOW ENGINE INNODB STATUS\G

# Redis：内存与键空间统计
redis-cli INFO memory
redis-cli INFO keyspace
redis-cli --bigkeys          # 定位大 Key
redis-cli SLOWLOG GET 10     # 获取最近 10 条慢命令
```
---

## 10. 运维检查单

**流水线**
- [ ] 变更全走流水线进生产，不手工 `apply`。
- [ ] 云端鉴权用 OIDC，不用长期 secret。
- [ ] 质量门让构建失败， admin 也不能绕过。
- [ ] SLO 异常触发自动回滚。

**基础设施**
- [ ] 所有资源用 Terraform / 等价 IaC 定义。
- [ ] 远程 state 带锁，服务和环境各一份 state 文件。
- [ ] 每个 PR 都贴 `terraform plan`， review 看 plan 不看 HCL。
- [ ] 漂移检测每天至少跑一次。

**监控**
- [ ] 指标、日志、链路都通。
- [ ] 每个服务配四个黄金信号（延迟、流量、错误、饱和度）仪表盘。
- [ ] 告警基于燃烧速率，不用裸阈值。
- [ ] 每条告警带 runbook 链接，没 runbook 就不告警。

**日志**
- [ ] 服务全发 JSON 结构化日志。
- [ ] Request ID 端到端透传。
- [ ] 配置保留分层，旧索引自动滚出。
- [ ] 写入前脱敏敏感字段。

**SRE**
- [ ] 每个服务发布 SLO，高管签字确认。
- [ ] 团队仪表盘显示错误预算。
- [ ] On-call 轮值有文档化的升级路径。
- [ ] SEV-1 / SEV-2 事故 5 天内复盘。

**成本**
- [ ] 资源全打 tag，成本仪表盘按团队拆分。
- [ ] 非生产环境业务时间外自动关机。
- [ ] 每月复审闲置或过大资源。

每一条没勾的，第一次出问题都会让我赔上一周事故响应和一块高管信任。补齐它们工作量小，收益大；唯一障碍是"还没紧急到必须做"。
