---
title: "Cloud Computing (7): Cloud Operations and DevOps Practices"
date: 2023-05-26 09:00:00
tags:
  - Cloud Computing
  - DevOps
  - SRE
  - Infrastructure as Code
  - Monitoring
  - CI/CD
categories: Cloud Computing
series: cloud-computing
lang: en
mathjax: false
description: "A working DevOps engineer's guide: CI/CD pipelines that gate quality, Terraform for reproducible infrastructure, Prometheus + Grafana monitoring, ELK/EFK logging, SRE error budgets, and the operational habits that keep services up at 3 AM."
disableNunjucks: true
series_order: 7
translationKey: "cloud-computing-7"
---
![Chapter concept illustration](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/illustration_1.png)

In 2017 GitLab lost six hours of database state. An engineer, exhausted, ran `rm -rf` on the wrong server during an incident. The backup procedures had silently been broken for months; nobody noticed because no one was restoring from backups. The lesson is not "be careful with rm". The lesson is that operations is a *system* — tools, runbooks, monitoring, automation, and the rituals around them. When the system is healthy, no single tired engineer can take down production. When the system is rotten, every late-night fix is one keystroke from disaster.

This article is about building that system. CI/CD that gates quality before code reaches users. Infrastructure as code so that "the production environment" is a Git revision, not a snowflake server. Monitoring that distinguishes signal from noise. Logs you can actually search. And the SRE practices — error budgets, SLOs, blameless postmortems — that turn ad-hoc firefighting into engineering.

## What You Will Learn

- CI/CD pipelines: stages, quality gates, rollback, and a complete GitHub Actions example
- Infrastructure as Code with Terraform: the workflow, state management, and module patterns
- Monitoring with Prometheus + Grafana + Alertmanager: scrape model, PromQL, alerting rules
- Centralised logging architecture (EFK / ELK): shippers, buffers, processors, retention tiers
- Auto-scaling that responds to real load without flapping
- Cost optimisation that does not require rewriting your application
- SRE practices: SLI / SLO / error budgets, blameless postmortems, GitOps

## Prerequisites

- Comfort on the Linux command line
- Git and basic CI/CD concepts
- Parts 1-5 of this series recommended

---

## The CI/CD Pipeline as the System of Record

![CI/CD Pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig1_cicd_pipeline.png)

A modern CI/CD pipeline is not just "automation". It is the *only* way code is allowed to reach production, which makes it the system of record for every release: who shipped what, with which tests passing, against which infrastructure version, and what happened next. Every other piece of the operations stack hangs off this spine.

### 1 The eight stages

| Stage | Purpose | Failure mode |
|-------|---------|--------------|
| Commit | Trigger via push or merge | None — this is just an event |
| Build | Compile, package, image | Reproducibility (pin base images, lock dependencies) |
| Unit tests | Fast feedback on logic | Flakiness erodes trust — quarantine flaky tests aggressively |
| Security scan | SAST, dependency CVEs, image scan | Noise; tune severity gates per repo |
| Deploy staging | First time the new artefact runs | Config drift between staging and prod |
| Smoke / e2e | Cross-service contracts | Slow tests cause people to skip them |
| Deploy prod | Canary -> wider rollout | All-at-once rollouts; lack of automated rollback |
| Verify | SLO check post-deploy | Verification by eyeball; not measured |

### 2 A real GitHub Actions pipeline

```yaml
name: deploy
on:
  push:
    branches: [main]

permissions:
  id-token: write          # OIDC for cloud auth
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

      - name: Set up Go
        uses: actions/setup-go@v5
        with: { go-version-file: go.mod }

      - name: Lint
        uses: golangci/golangci-lint-action@v4

      - name: Unit tests
        run: go test -race -coverprofile=coverage.out ./...

      - name: Coverage gate
        run: |
          pct=$(go tool cover -func=coverage.out | tail -1 | awk '{print $3}' | tr -d '%')
          echo "Coverage: ${pct}%"
          if (( $(echo "$pct < 80" | bc -l) )); then
            echo "::error::Coverage ${pct}% is below 80% threshold"
            exit 1
          fi

      - name: Build and push image
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login $REGISTRY -u ${{ github.actor }} --password-stdin
          docker build -t $IMAGE:${{ github.sha }} -t $IMAGE:latest .
          docker push $IMAGE --all-tags

  security-scan:
    needs: build-test
    runs-on: ubuntu-latest
    steps:
      - name: Trivy image scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.IMAGE }}:${{ github.sha }}
          severity: CRITICAL,HIGH
          exit-code: 1           # fail the build on critical CVEs

      - name: SAST with Semgrep
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

      - name: Deploy to ECS staging
        run: |
          aws ecs update-service \
            --cluster staging \
            --service web \
            --force-new-deployment \
            --task-definition web:$(aws ecs describe-task-definition \
              --task-definition web --query 'taskDefinition.revision')

      - name: Wait for stable
        run: aws ecs wait services-stable --cluster staging --services web

  smoke-test:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - name: Health check
        run: |
          for i in {1..10}; do
            status=$(curl -s -o /dev/null -w '%{http_code}' https://staging.example.com/healthz)
            if [ "$status" = "200" ]; then echo "Healthy"; exit 0; fi
            sleep 5
          done
          echo "::error::Staging health check failed after 50s"
          exit 1

      - name: API contract tests
        run: |
          npm ci
          npx newman run tests/postman/smoke.json \
            --environment tests/postman/staging.json \
            --reporters cli,junit \
            --reporter-junit-export results.xml

  deploy-prod:
    needs: smoke-test
    runs-on: ubuntu-latest
    environment: production       # requires manual approval
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::111122223333:role/deploy-prod
          aws-region: us-east-1

      - name: Canary deploy (10% traffic)
        run: |
          aws ecs update-service --cluster prod --service web \
            --deployment-configuration '{
              "deploymentCircuitBreaker": {"enable": true, "rollback": true},
              "maximumPercent": 200,
              "minimumHealthyPercent": 100
            }' \
            --force-new-deployment

      - name: Verify canary SLOs (5 min)
        run: |
          sleep 300
          error_rate=$(aws cloudwatch get-metric-statistics \
            --namespace Custom/Web --metric-name ErrorRate \
            --start-time $(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S) \
            --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
            --period 300 --statistics Average \
            --query 'Datapoints[0].Average' --output text)
          if (( $(echo "$error_rate > 1.0" | bc -l) )); then
            echo "::error::Error rate ${error_rate}% exceeds 1% - triggering rollback"
            aws ecs update-service --cluster prod --service web \
              --task-definition web:$(( $(aws ecs describe-services --cluster prod \
                --services web --query 'services[0].taskDefinition' --output text \
                | grep -o '[0-9]*$') - 1 ))
            exit 1
          fi
```

Three design decisions worth calling out. First, **OIDC replaces long-lived secrets** — the `id-token: write` permission and `role-to-assume` mean no AWS access keys live in GitHub. Second, **the production environment requires manual approval**, creating a human checkpoint between staging success and prod deploy. Third, **the canary has automated rollback** — if the error rate exceeds 1% in the first five minutes, the pipeline rolls back without waiting for a human to wake up.

### 3 Deployment strategies compared

Not every service can afford a canary. The right strategy depends on traffic volume, rollback cost, and how quickly you can detect a bad deployment.

| Strategy | How it works | Rollback speed | When to use |
|----------|-------------|----------------|-------------|
| **Rolling** | Replace instances N at a time | Minutes (stop rollout, redeploy old version) | Stateless services with health checks |
| **Blue/green** | Run two full environments, switch DNS/LB | Seconds (flip back) | Stateful services, databases, when you need instant rollback |
| **Canary** | Route a fraction of traffic to new version | Seconds (drain canary) | High-traffic services where you can measure error rates quickly |
| **Feature flags** | Deploy code dark, enable incrementally | Instant (toggle flag) | User-facing features, A/B tests, gradual rollouts |
| **Recreate** | Stop all old, start all new | Slow (full redeploy) | Dev/test environments, batch workers, stateful singletons |

In practice you combine them. Deploy with canary; the canary itself uses feature flags to expose new behaviour to 1% of users; the feature flag service has its own blue/green deployment.

---

![Deployment strategies compared (rolling, blue-green, canary)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig2_deployment_strategies.png)


## Infrastructure as Code with Terraform

![Terraform Workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig_terraform_pipeline_en.png)

Infrastructure as code (IaC) means the production environment is described in version-controlled files, not in the memory of the person who clicked through the console last Tuesday. Terraform is the most widely adopted tool for this because it is provider-agnostic, declarative, and has a preview step (`plan`) that shows exactly what will change before anything changes.

### 1 The core workflow

```text
terraform init      # download providers, configure backend
terraform plan      # preview: what will be created / changed / destroyed
terraform apply     # make reality match the code
terraform destroy   # tear it all down (non-prod only!)
```

`plan` is the part you actually live in. It tells you what will change *before* anything changes. Code review happens against the plan output, not just the HCL.

```bash
# Generate a plan file for CI
terraform plan -out=tfplan -input=false

# Show it in human-readable form for PR comments
terraform show -no-color tfplan > plan.txt

# Apply only the reviewed plan (no re-planning)
terraform apply -input=false tfplan
```

### 2 State management

Terraform's state file (`terraform.tfstate`) is the map between your HCL code and the real resources in the cloud. Get state management wrong and you get drift, conflicts, and destroyed infrastructure.

Rules:

- **Remote backend with locking.** S3 + DynamoDB on AWS, GCS on GCP, Terraform Cloud anywhere. Never commit state to Git — it contains secrets and causes merge conflicts.
- **One state file per service per environment.** `services/web/prod/`, `services/web/staging/`, `services/api/prod/`. This limits blast radius and parallelises applies.
- **State encryption at rest.** S3 server-side encryption is the minimum.
- **State access restricted to CI.** Humans should run `plan` locally but `apply` only through the pipeline.

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

### 3 A complete production module

Modules are reusable, testable units of infrastructure. A well-designed module encapsulates a service pattern so that teams consume it without knowing the details.

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
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.this.name
        awslogs-region        = data.aws_region.current.name
        awslogs-stream-prefix = var.name
      }
    }
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

  network_configuration {
    subnets          = data.aws_subnets.private.ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.this.arn
    container_name   = var.name
    container_port   = 8080
  }
}

output "service_url" {
  value = "https://${aws_lb.this.dns_name}"
}
```

### 4 Drift detection

Even with IaC, reality drifts. Someone clicks through the console to fix a production incident and forgets to backport the change. A scheduled drift check catches this before it compounds.

```bash
#!/bin/bash
# drift-check.sh -- run daily via cron or scheduled pipeline
set -euo pipefail

SERVICES="web api worker"
ENVS="staging prod"

for svc in $SERVICES; do
  for env in $ENVS; do
    dir="services/${svc}/${env}"
    echo "=== Checking ${dir} ==="
    cd "$dir"
    terraform init -backend=true -input=false > /dev/null
    if ! terraform plan -detailed-exitcode -input=false > /dev/null 2>&1; then
      echo "DRIFT DETECTED in ${dir}"
      terraform plan -no-color -input=false > "/tmp/drift-${svc}-${env}.txt"
      # Send to Slack / PagerDuty
      curl -X POST "$SLACK_WEBHOOK" -d "{\"text\":\"Drift detected in ${dir}. See attached plan.\"}"
    else
      echo "No drift in ${dir}"
    fi
    cd - > /dev/null
  done
done
```

The `-detailed-exitcode` flag makes `terraform plan` return exit code 2 when changes are detected, which is what drives the conditional alert.

---

## Monitoring with Prometheus, Grafana, and Alertmanager

Every production system needs three pillars of observability: **metrics** (numbers over time), **logs** (events with context), and **traces** (request paths across services). This section covers metrics; the next covers logs.

### 1 The Prometheus scrape model

Prometheus *pulls* metrics from your services, rather than having services push to it. This has two advantages: you can scrape a target from multiple Prometheus instances for HA, and a crashed service simply stops being scraped rather than leaving a dangling push connection.

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
      # Only scrape pods with annotation prometheus.io/scrape=true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port, __meta_kubernetes_pod_ip]
        action: replace
        target_label: __address__
        regex: (.+);(.+)
        replacement: $2:$1
```

### 2 The four golden signals

Google's SRE book defines four signals that every service should measure. Prometheus makes them straightforward.

| Signal | What to measure | PromQL example |
|--------|----------------|----------------|
| **Latency** | Duration of successful requests | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{status!~"5.."}[5m]))` |
| **Traffic** | Requests per second | `sum(rate(http_requests_total[5m])) by (service)` |
| **Errors** | Fraction of failed requests | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |
| **Saturation** | How full the service is | `container_memory_working_set_bytes / container_spec_memory_limit_bytes` |

![The four golden signals of monitoring (latency, traffic, errors, saturation)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig3_four_golden_signals.png)


### 3 Instrumenting your application

Every service should expose a `/metrics` endpoint. In Go, this is a few lines:

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

### 4 Alerting rules that do not wake people up for nothing

The goal is zero false-positive pages. Every alert that fires should require human action. If it does not, it should be a dashboard panel, not a page.

```yaml
# alerts/slo.yml
groups:
  - name: slo-burn-rate
    rules:
      # Multi-window, multi-burn-rate alerting (Google SRE Workbook pattern)
      # Fast burn: 14.4x in 1h AND 6x in 6h
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
          summary: "Error budget burning fast -- will exhaust in < 1 day"
          runbook: "https://wiki.internal/runbooks/high-error-rate"
          dashboard: "https://grafana.internal/d/slo-overview"

      # Slow burn: 3x in 1d AND 1x in 3d
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
          summary: "Error budget burning slowly -- will exhaust in < 10 days"
          runbook: "https://wiki.internal/runbooks/high-error-rate"

      - alert: HighLatency_P99
        expr: |
          histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))
          > 2.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency above 2s for {{ $labels.service }}"
          runbook: "https://wiki.internal/runbooks/high-latency"
```

The multi-window, multi-burn-rate pattern from the Google SRE Workbook is the single most important improvement you can make to alerting. Instead of alerting on "error rate > 1%", which fires on any brief spike, it alerts on "the rate at which we are consuming our error budget". A 2% error rate for 30 seconds barely moves the monthly budget; a 0.5% error rate sustained for 3 days consumes it entirely. The burn-rate approach catches both.

### 5 Alertmanager routing

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

## Centralised Logging with EFK / ELK

Metrics tell you *that* something is wrong. Logs tell you *why*. A centralised logging stack collects, processes, indexes, and retains logs from every service in one searchable store.

### 1 Architecture overview

The two dominant stacks are **ELK** (Elasticsearch, Logstash, Kibana) and **EFK** (Elasticsearch, Fluentd/Fluent Bit, Kibana). The difference is the shipper: Logstash is JVM-based and powerful but heavy; Fluent Bit is C-based, lightweight, and runs well as a DaemonSet.

```text
+-----------+     +------------+     +---------------+     +---------+
| App pods  | --> | Fluent Bit | --> | Elasticsearch | --> | Kibana  |
| (stdout)  |     | (DaemonSet)|     | (3-node HA)   |     | (query) |
+-----------+     +------------+     +---------------+     +---------+
                        |
                        v
                  +------------+
                  |   Kafka    |  (optional buffer for high volume)
                  +------------+
```

### 2 Structured logging

The single most impactful thing you can do for log searchability is to log in JSON. A structured log entry is a queryable document; an unstructured one is a string you have to regex-parse.

```python
import structlog
import uuid

# Configure structlog for JSON output
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),

![Centralised logging pipeline from sources to Elasticsearch and Kibana](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig4_logging_pipeline.png)

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

This produces log lines like:

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

### 3 Fluent Bit configuration for Kubernetes

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
    # Scrub sensitive fields before indexing
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

### 4 Retention tiers

Logs are expensive to store and index. A tiered approach balances searchability with cost.

| Tier | Duration | Storage | Use case |
|------|----------|---------|----------|
| **Hot** | 0-7 days | SSD-backed Elasticsearch | Active debugging, incident response |
| **Warm** | 7-30 days | HDD-backed Elasticsearch | Recent investigations, compliance queries |
| **Cold** | 30-90 days | S3 / GCS with Elasticsearch snapshot | Audit, rare lookups |
| **Frozen** | 90 days - 7 years | S3 Glacier / Archive | Compliance retention (HIPAA, PCI) |

Automate the transitions with Elasticsearch Index Lifecycle Management (ILM):

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

## Auto-Scaling: Responding to Real Load

Auto-scaling sounds simple: add capacity when load increases, remove it when load decreases. In practice, getting it right requires choosing the right signal, tuning the response speed, and preventing flapping.

### 1 Scaling signals

| Signal | When to use | Watch out for |
|--------|-------------|---------------|
| **CPU utilisation** | General-purpose; good default | Spiky workloads cause oscillation |
| **Memory utilisation** | Memory-bound services (caches, JVM) | Slow to release — GC delays downsizing |
| **Request rate (RPS)** | Web services with predictable per-request cost | Needs custom metrics pipeline |
| **Queue depth** | Async workers, batch processors | Must scale on *rate of growth*, not absolute depth |
| **Custom business metric** | When none of the above correlates with user experience | Requires instrumentation effort |

### 2 Kubernetes HPA with custom metrics

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
      stabilizationWindowSeconds: 60    # wait 1 min before scaling up
      policies:
        - type: Percent
          value: 100                     # double at most
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300   # wait 5 min before scaling down
      policies:
        - type: Percent
          value: 10                      # remove at most 10% per period
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

The asymmetric scale-up/scale-down behaviour is deliberate. Scaling up fast (double in 60 seconds) protects user experience during traffic spikes. Scaling down slowly (10% per minute, after a 5-minute cooldown) prevents the "sawtooth" pattern where the autoscaler repeatedly scales down too aggressively, triggers high CPU, scales back up, and oscillates.

### 3 Predictive scaling

For workloads with predictable patterns (e.g., a retail site that peaks every day at noon), reactive scaling is always late. By the time CPU hits 70%, users are already experiencing latency. AWS predictive scaling and GCP scheduled scaling address this:

```bash
# AWS Auto Scaling predictive policy
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

The `SchedulingBufferTime` of 300 seconds means instances are launched 5 minutes before the predicted spike, so they are warmed up and passing health checks by the time traffic arrives.

---


![Asymmetric HPA behaviour: fast scale-up, slow scale-down to prevent flapping](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig5_autoscaling_curve.png)

## Cost Optimisation Without Rewriting Your Application

Cloud bills surprise everyone eventually. The good news is that 30-50% of most cloud bills can be cut without changing application code — it is about rightsizing, scheduling, and commitment.

### 1 The cost optimisation hierarchy

Work from top to bottom; each layer has higher impact and lower effort than the one below it:

1. **Turn off what you are not using.** Idle dev/staging environments running 24/7, orphaned EBS volumes, unattached Elastic IPs, load balancers fronting zero targets.
2. **Schedule non-production.** Dev clusters do not need to run at 3 AM. Save 65% by running them 10 hours/day, 5 days/week.
3. **Rightsize.** Most instances are 2-4x oversized. Use CloudWatch / Cloud Monitoring data to find instances where CPU and memory never exceed 20%.
4. **Use Spot/Preemptible for fault-tolerant workloads.** CI runners, batch jobs, dev environments. Savings of 60-90%.
5. **Commit with Reserved Instances / Savings Plans.** For steady-state production, 1-year no-upfront commitments save 30-40% with minimal risk.
6. **Storage tiering.** Move infrequently accessed data to cheaper tiers (S3 Infrequent Access, Glacier, Archive).

### 2 Automated cost controls

```bash
# shutdown-nonprod.sh -- cron: 0 19 * * 1-5 (7 PM weekdays)
#!/bin/bash
set -euo pipefail

ENVS=("dev" "staging" "qa")

for env in "${ENVS[@]}"; do
  echo "Shutting down ${env} ECS services..."
  for svc in $(aws ecs list-services --cluster "$env" --query 'serviceArns[]' --output text); do
    aws ecs update-service --cluster "$env" --service "$svc" --desired-count 0
  done

  echo "Stopping ${env} RDS instances..."
  for db in $(aws rds describe-db-instances --query "DBInstances[?TagList[?Key=='Environment'&&Value=='${env}']].DBInstanceIdentifier" --output text); do
    aws rds stop-db-instance --db-instance-identifier "$db" || true
  done
done

echo "Non-prod environments shut down at $(date)"
```

```bash
# startup-nonprod.sh -- cron: 0 8 * * 1-5 (8 AM weekdays)
#!/bin/bash
set -euo pipefail

ENVS=("dev" "staging" "qa")

for env in "${ENVS[@]}"; do
  echo "Starting ${env} RDS instances..."
  for db in $(aws rds describe-db-instances --query "DBInstances[?TagList[?Key=='Environment'&&Value=='${env}']].DBInstanceIdentifier" --output text); do
    aws rds start-db-instance --db-instance-identifier "$db" || true
  done

  echo "Restoring ${env} ECS services..."
  for svc in $(aws ecs list-services --cluster "$env" --query 'serviceArns[]' --output text); do
    aws ecs update-service --cluster "$env" --service "$svc" --desired-count 2
  done
done

echo "Non-prod environments started at $(date)"
```

### 3 Tagging strategy for cost allocation

Tags are the only way to attribute cloud spend to teams. Without them, the monthly bill is one big number nobody can act on.

| Tag key | Example value | Purpose |
|---------|--------------|---------|
| `Environment` | `prod`, `staging`, `dev` | Filter non-prod for scheduling |
| `Team` | `platform`, `payments`, `ml` | Cost allocation per team |
| `Service` | `web`, `api`, `worker` | Cost per microservice |
| `CostCenter` | `CC-1234` | Finance mapping |
| `ManagedBy` | `terraform`, `manual` | Identify unmanaged resources |

Enforce tags via SCP (AWS) or Organisation Policy (GCP):

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

## SRE Practices: From Firefighting to Engineering

Site Reliability Engineering is the practice of treating operations as a software engineering problem. The key tools are SLOs, error budgets, and blameless postmortems.

### 1 SLI, SLO, and error budgets

- **SLI** (Service Level Indicator): a quantitative measure of service behaviour. "The proportion of requests that complete in under 300ms."
- **SLO** (Service Level Objective): a target for the SLI. "99.9% of requests complete in under 300ms over a rolling 30-day window."
- **Error budget**: the inverse of the SLO. If your SLO is 99.9%, your error budget is 0.1% — you are allowed 43 minutes of downtime per month.

The error budget is a management tool, not just a metric. When the budget is healthy, teams ship features fast. When the budget is low, teams freeze features and focus on reliability. This replaces the perennial "move fast vs. be stable" argument with a number everyone can see.

```yaml
# SLO definition (as applied in Prometheus)
# 99.9% availability = 0.1% error budget = 43.2 min/month

# Availability SLI
- record: sli:availability:ratio_rate5m
  expr: |
    sum(rate(http_requests_total{status!~"5.."}[5m]))
    / sum(rate(http_requests_total[5m]))

# Latency SLI (proportion of requests < 300ms)
- record: sli:latency:ratio_rate5m
  expr: |
    sum(rate(http_request_duration_seconds_bucket{le="0.3"}[5m]))
    / sum(rate(http_request_duration_seconds_count[5m]))

# Error budget remaining (30-day window)
- record: error_budget:remaining
  expr: |
    1 - (
      (1 - avg_over_time(sli:availability:ratio_rate5m[30d]))
      / (1 - 0.999)
    )
```

### 2 On-call and escalation

![Error budget burndown over a 30-day SLO window](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig6_error_budget.png)


A healthy on-call rotation looks like this:

| Aspect | Good | Bad |
|--------|------|-----|
| Rotation length | 1 week | 1 month (burnout) |
| Team size | 6-8 people (one week every 6-8 weeks) | 2-3 people (constant on-call) |
| Pages per shift | 0-2 actionable | 10+ (alert fatigue, important pages get missed) |
| Handoff | Written handoff document at shift change | "Nothing happened" verbal handoff |
| Compensation | Time off in lieu or on-call pay | "It's part of the job" |
| Escalation | Clear path: primary -> secondary -> engineering manager -> VP | "Call whoever answers" |

### 3 Blameless postmortems

A postmortem that names someone as the cause is a postmortem that teaches everyone else to hide their mistakes. The template:

```markdown
## Incident Report: [Title]
**Date**: YYYY-MM-DD
**Duration**: X hours Y minutes
**Severity**: SEV-1 / SEV-2 / SEV-3
**Author**: [name]
**Reviewers**: [names]

### Summary
One paragraph: what happened, what was the user impact, how was it resolved.

### Timeline (all times UTC)
- 14:00 - Monitoring alert fires: error rate > 5%
- 14:03 - On-call engineer acknowledges, begins investigation
- 14:15 - Root cause identified: database connection pool exhausted
- 14:20 - Mitigation applied: increased pool size, restarted service
- 14:25 - Error rate returns to baseline
- 14:30 - Incident declared resolved

### Root cause
The connection pool was sized for 50 connections. A new batch job,
deployed two days prior, held connections for 30 seconds instead of
the expected 100ms, exhausting the pool during peak traffic.

### Impact
- 25 minutes of elevated error rates (peak 12%)
- ~3,200 failed requests
- No data loss

### What went well
- Alert fired within 3 minutes of the problem starting
- On-call engineer had the right access to diagnose and mitigate

### What went poorly
- The batch job was not load-tested against the shared database
- The connection pool had no circuit breaker

### Action items
| Action | Owner | Priority | Due date |
|--------|-------|----------|----------|
| Add connection pool metrics to dashboard | Alice | P1 | 2024-02-01 |
| Load-test batch jobs against shared resources | Bob | P1 | 2024-02-15 |
| Implement connection pool circuit breaker | Carol | P2 | 2024-03-01 |
| Add runbook for connection pool exhaustion | Dave | P2 | 2024-02-01 |
```


![Incident response timeline: detect, triage, mitigate, postmortem](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig7_incident_timeline.png)

The action items are the entire point. A postmortem without action items is just a story.

---

## GitOps: Git as the Single Source of Truth

![GitOps with ArgoCD/Flux](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig_gitops_argocd_en.png)

GitOps removes a whole class of mistakes by removing a whole class of capabilities. Nobody runs `kubectl apply` against production. The cluster reconciles itself to whatever is in the config repo, and the only way to change the cluster is to change Git.

### 1 ArgoCD Application manifest

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
      prune:    true       # delete resources removed from git
      selfHeal: true       # revert manual changes back to git state
    retry:
      limit: 5
      backoff: { duration: 30s, maxDuration: 5m, factor: 2 }
```

The properties this gives you for free:

- **Rollback** is `git revert`.
- **Audit** is `git log`.
- **Diff** between staging and prod is `git diff`.
- **Disaster recovery** is "point ArgoCD at the same repo from a fresh cluster".

### 2 Repository structure for GitOps

```text
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
          kustomization.yaml      # patches: replica count, resource limits
          ingress-patch.yaml      # staging domain
        production/
          kustomization.yaml      # patches: higher replicas, more resources
          ingress-patch.yaml      # prod domain
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

This Kustomize-based layout gives you DRY base manifests with environment-specific patches. A PR that changes `apps/web/overlays/production/` is a production change — it gets the same review scrutiny as application code.

### 3 GitOps vs. traditional CI/CD

| Aspect | Traditional CI/CD | GitOps |
|--------|-------------------|--------|
| Deployment trigger | Pipeline pushes to cluster | Cluster pulls from Git |
| Drift detection | Manual / none | Continuous (self-heal) |
| Rollback | Re-run old pipeline | `git revert` |
| Access control | CI system needs cluster admin | Only ArgoCD needs cluster access |
| Audit trail | CI logs (may rotate) | Git history (permanent) |
| Multi-cluster | Duplicate pipeline per cluster | One repo, multiple ArgoCD instances |

---

## Troubleshooting Guide: When Things Go Wrong

Every operations engineer needs a mental decision tree for the 3 AM page. Here is a practical one.

### 1 The first five minutes

```bash
# 1. What is the symptom? Check the dashboard.
open https://grafana.internal/d/overview

# 2. Is it one service or everything?
kubectl get pods -A | grep -v Running
kubectl top nodes
kubectl top pods --sort-by=cpu -A | head -20

# 3. What changed recently?
kubectl rollout history deployment/web -n production
git log --oneline --since="2 hours ago" -- apps/web/

# 4. Check resource pressure
kubectl describe nodes | grep -A5 "Conditions:"
df -h                           # disk
free -m                         # memory
ss -tlnp                        # open ports / connection counts

# 5. Read the logs for the affected service
kubectl logs -l app=web -n production --tail=100 --since=5m
```

### 2 Common failure patterns

| Symptom | Likely cause | Investigation | Fix |
|---------|-------------|---------------|-----|
| 5xx spike after deploy | Bad code / config | `kubectl rollout undo deployment/web` | Rollback, investigate offline |
| Gradual latency increase | Memory leak, connection leak | Check heap/goroutine metrics, connection pool counters | Restart pods, fix leak |
| Sudden 100% errors | Dependency down (DB, cache, external API) | Check dependency health endpoints | Circuit breaker, fallback |
| Pod crash loop | OOM, missing config, failed health check | `kubectl describe pod <name>`, `kubectl logs <name> --previous` | Fix config/limits, increase memory |
| Nodes not ready | Disk pressure, network, kubelet crash | `kubectl describe node`, `journalctl -u kubelet` | Drain and replace node |
| DNS resolution failures | CoreDNS overloaded | `kubectl logs -l k8s-app=kube-dns -n kube-system` | Scale CoreDNS, check ndots |

### 3 Database troubleshooting checklist

```bash
# PostgreSQL: slow queries right now
SELECT pid, now() - pg_stat_activity.query_start AS duration,
       query, state, wait_event_type, wait_event
FROM pg_stat_activity
WHERE state != 'idle'
  AND (now() - pg_stat_activity.query_start) > interval '5 seconds'
ORDER BY duration DESC;

# PostgreSQL: connection count by state
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state;

# PostgreSQL: table bloat check
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
       n_dead_tup,
       n_live_tup,
       ROUND(n_dead_tup * 100.0 / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC;

# MySQL: current queries
SHOW FULL PROCESSLIST;

# MySQL: InnoDB status (locks, deadlocks, buffer pool)
SHOW ENGINE INNODB STATUS\G

# Redis: memory and key stats
redis-cli INFO memory
redis-cli INFO keyspace
redis-cli --bigkeys          # find large keys
redis-cli SLOWLOG GET 10     # recent slow commands
```

---

## The Operations Checklist

**Pipeline**
- [ ] Every change reaches production via the pipeline; no manual `apply`.
- [ ] OIDC for cloud auth, no long-lived secrets.
- [ ] Quality gates fail builds; nobody bypasses with admin override.
- [ ] Automated rollback on SLO breach.

**Infrastructure**
- [ ] All resources defined in Terraform / equivalent IaC.
- [ ] Remote state with locking; one state file per service per environment.
- [ ] `terraform plan` posted on every PR; review the plan, not just the HCL.
- [ ] Drift detection runs at least daily.

**Monitoring**
- [ ] Metrics, logs, traces all flowing.
- [ ] Dashboards exist for every service with the four golden signals (latency, traffic, errors, saturation).
- [ ] Alerts based on burn rate, not raw thresholds.
- [ ] Every alert has a runbook URL; no runbook -> no alert.

**Logging**
- [ ] JSON structured logs from every service.
- [ ] Request ID propagated end-to-end.
- [ ] Retention tiers configured; old indices roll off automatically.
- [ ] Sensitive fields scrubbed before write.

**SRE**
- [ ] SLOs published per service, with executive sign-off.
- [ ] Error budget visible on the team dashboard.
- [ ] On-call rotation with documented escalation.
- [ ] Postmortems for all SEV-1 / SEV-2 incidents within 5 business days.

**Cost**
- [ ] Tags on every resource; cost dashboard split by team.
- [ ] Auto-shutdown of non-prod outside business hours.
- [ ] Monthly review of idle / oversized resources.

The pattern: every box on this list is something that, if missing, will cost you a week of incident response and a chunk of executive trust the first time it bites. The work to add them is small; the savings are large; the only obstacle is the day it becomes urgent enough to stop putting off.
