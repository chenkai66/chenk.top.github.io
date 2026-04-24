---
title: "Cloud Operations and DevOps Practices"
date: 2024-09-06 09:00:00
tags:
  - Cloud Computing
  - DevOps
  - SRE
  - Infrastructure as Code
  - Monitoring
  - CI/CD
categories: Cloud Computing
series:
  name: "Cloud Computing"
  part: 6
  total: 8
lang: en
mathjax: false
description: "A working DevOps engineer's guide: CI/CD pipelines that gate quality, Terraform for reproducible infrastructure, Prometheus + Grafana monitoring, ELK/EFK logging, SRE error budgets, and the operational habits that keep services up at 3 AM."
---

In 2017 GitLab lost six hours of database state. An engineer, exhausted, ran `rm -rf` on the wrong server during an incident. The backup procedures had silently been broken for months; nobody noticed because no one was restoring from backups. The lesson is not "be careful with rm". The lesson is that operations is a *system* - tools, runbooks, monitoring, automation, and the rituals around them. When the system is healthy, no single tired engineer can take down production. When the system is rotten, every late-night fix is one keystroke from disaster.

This article is about building that system. CI/CD that gates quality before code reaches users. Infrastructure as code so that "the production environment" is a Git revision, not a snowflake server. Monitoring that distinguishes signal from noise. Logs you can actually search. And the SRE practices - error budgets, SLOs, blameless postmortems - that turn ad-hoc firefighting into engineering.

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

## 1. The CI/CD Pipeline as the System of Record

![CI/CD Pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig1_cicd_pipeline.png)

A modern CI/CD pipeline is not just "automation". It is the *only* way code is allowed to reach production, which makes it the system of record for every release: who shipped what, with which tests passing, against which infrastructure version, and what happened next. Every other piece of the operations stack hangs off this spine.

### 1.1 The eight stages

| Stage | Purpose | Failure mode |
|-------|---------|--------------|
| Commit | Trigger via push or merge | None - this is just an event |
| Build | Compile, package, image | Reproducibility (pin base images, lock dependencies) |
| Unit tests | Fast feedback on logic | Flakiness erodes trust - quarantine flaky tests aggressively |
| Security scan | SAST, dependency CVEs, image scan | Noise; tune severity gates per repo |
| Deploy staging | First time the new artefact runs | Config drift between staging and prod |
| Smoke / e2e | Cross-service contracts | Slow tests cause people to skip them |
| Deploy prod | Canary -> wider rollout | All-at-once rollouts; lack of automated rollback |
| Verify | SLO check post-deploy | Verification by eyeball; not measured |

### 1.2 A real GitHub Actions pipeline

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
    environment: production         # GitHub manual approval gate
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
      - run: ./scripts/check_slo.sh    # rolls back via re-deploy on failure
```

Three details that turn this from "works on my machine" into "works at 3 AM":

- **OIDC, not long-lived credentials.** `id-token: write` lets GitHub mint short-lived AWS tokens; you never store an `AWS_ACCESS_KEY_ID` secret.
- **Manual approval gate** (`environment: production`) for prod, with auto-approve for staging.
- **Verification step** that checks an SLO after the deploy and triggers an automated rollback if the new version is worse.

### 1.3 Deployment strategies

- **Rolling**: replace pods N at a time. Good default; works because Kubernetes / ECS handle health checks.
- **Blue/green**: stand up the new version in parallel, switch a load-balancer pointer. Instant rollback; expensive in compute.
- **Canary**: send 1% of traffic to the new version, watch metrics, expand to 5%, 25%, 100%. The right answer for any service where a regression hurts.
- **Feature flags**: ship the code dark, enable for cohorts. Decouples deploy from release.

A canary plus a feature flag is the gold standard - you control rollout *and* exposure independently.

## 2. Infrastructure as Code: Terraform

![Terraform Workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig2_iac_terraform.png)

The point of IaC is not "automation" in the abstract. It is that **the production environment is a Git revision**. You can diff it, review it, roll it back, and reproduce it - which is impossible with a hand-built environment, no matter how good your wiki is.

### 2.1 The Terraform workflow

```
HCL files  ->  terraform init   ->  download providers
            -> terraform plan   ->  diff desired vs actual state
            -> terraform apply  ->  call cloud APIs to converge
            -> terraform.tfstate (record of the world as we believe it)
```

`plan` is the part you actually live in. It tells you what will change *before* anything changes. Code review happens against the plan output, not just the HCL.

### 2.2 A complete production module

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
  metadata_options { http_tokens = "required" }     # IMDSv2 enforced
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

Things to notice:

- **`name_prefix` rather than `name`** - lets Terraform create the new resource before destroying the old one (`create_before_destroy` style).
- **IMDSv2 enforced** via `http_tokens = "required"` - blocks the SSRF -> credential-theft pattern that broke Capital One.
- **Instance refresh on the ASG** automatically rolls instances when the launch template changes.
- **Tags propagated** from a single `locals` block - cost allocation works, audits work.

### 2.3 Remote state and locking

Local state is a footgun. Two engineers running `apply` simultaneously will corrupt each other's view of the world. Use a remote backend with locking:

```hcl
terraform {
  backend "s3" {
    bucket         = "company-tfstate"
    key            = "prod/web-service.tfstate"
    region         = "us-east-1"
    dynamodb_table = "tfstate-lock"   # provides the lock
    encrypt        = true
    kms_key_id     = "alias/terraform-state"
  }
}
```

Per-environment, per-service state files. One huge state file is slow and dangerous; one tiny state file per resource is unmanageable.

### 2.4 Terraform vs Ansible vs the cloud-native option

| Tool | Best for | Language | Multi-cloud |
|------|---------|----------|-------------|
| Terraform | Provisioning resources | HCL (declarative) | Yes |
| Ansible | Configuring inside resources | YAML (procedural-ish) | Yes |
| CloudFormation / ARM / Deployment Manager | Single-cloud, deep integration | YAML / JSON | No |
| Pulumi / CDK | Same as Terraform but in real code | TS / Python / Go | Yes |

The mature pattern is Terraform for cloud resources, container images for application configuration, and Ansible only for the legacy boxes that survive on the edges.

## 3. Monitoring with Prometheus, Grafana, Alertmanager

![Monitoring Stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig3_monitoring_stack.png)

### 3.1 The three pillars - and why they are not enough

Metrics, logs, traces. Most teams start with metrics, add logs, intend to add traces. The pillars are *necessary but not sufficient*: the missing piece is **correlation**. A metric spike, a log error and a trace anomaly are the same incident; if you cannot pivot between them in one click, you are not observable, you just have lots of data.

### 3.2 Prometheus configuration that scales

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
      # Only scrape pods that opt in via annotation
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: "true"
      # Honour custom path / port annotations
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
      # Lift useful labels onto the time series
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
```

Three patterns that make Prometheus survive at scale:

- **Service discovery, not static targets.** In Kubernetes, pods come and go; the static target list goes stale within minutes.
- **Opt-in scraping.** Without the `prometheus.io/scrape: "true"` annotation, you would scrape every sidecar in the cluster.
- **Recording rules** for expensive queries that get rendered on every dashboard load. Compute once on ingest, query the cheap pre-aggregated series.

### 3.3 The PromQL queries you actually use

```promql
# CPU utilisation per node (excluding idle time)
100 - avg by (instance) (
  irate(node_cpu_seconds_total{mode="idle"}[5m])
) * 100

# Request rate per service
sum by (service) (rate(http_requests_total[5m]))

# Error rate as a fraction of total
sum by (service) (rate(http_requests_total{status=~"5.."}[5m]))
/
sum by (service) (rate(http_requests_total[5m]))

# P95 latency from a histogram
histogram_quantile(0.95,
  sum by (le, service) (rate(http_request_duration_seconds_bucket[5m])))

# Apdex score (excellent < 100ms, satisfactory < 500ms)
(
  sum by (service) (rate(http_request_duration_seconds_bucket{le="0.1"}[5m]))
  +
  sum by (service) (rate(http_request_duration_seconds_bucket{le="0.5"}[5m]))
) / 2
/ sum by (service) (rate(http_request_duration_seconds_count[5m]))
```

The shape worth memorising: **rate over a histogram bucket** for latency, **rate of count, sliced by status** for error rate. Almost every meaningful application metric reduces to one of these two patterns.

### 3.4 Alert rules that respect on-call

```yaml
groups:
- name: slo-burn-rate
  rules:
  - alert: ErrorBudgetBurnFast
    # Burning 14.4x normal rate over 1h: would consume monthly budget in ~2 days
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
      summary: "{{ $labels.service }} burning error budget 14x normal"
      runbook: "https://runbooks/internal/slo-burn"

  - alert: ErrorBudgetBurnSlow
    # Burning 6x normal over 6h: would consume monthly budget in ~5 days
    expr: |
      (
        sum by (service) (rate(http_requests_total{status=~"5.."}[6h]))
        /
        sum by (service) (rate(http_requests_total[6h]))
      ) > 0.006
    for: 15m
    labels: { severity: ticket }
```

The pattern, from the Google SRE workbook: alert on **error budget burn rate**, not on raw thresholds. Combine a fast-burn page (immediate response) with a slow-burn ticket (next business day). This eliminates two whole classes of false positives: brief blips that resolve themselves, and slow leaks that are not yet customer-visible.

## 4. Centralised Logging

![Logging Architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig4_logging_architecture.png)

### 4.1 The pipeline

```
app stdout -> shipper (Fluent Bit / Filebeat) -> buffer (Kafka)
           -> processor (Logstash / Fluentd) -> store (Elasticsearch / Loki)
           -> dashboard (Kibana / Grafana)
```

The buffer is the part most teams skip and then regret. Without it, a downstream Elasticsearch outage causes back-pressure that crashes shippers, which causes log loss. With Kafka in the middle, shippers write at line rate and processors catch up at their own pace.

### 4.2 Structured logging from the application

Plain-text logs are unsearchable at scale. Emit JSON from day one:

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
        # Allow extra={"user_id": ...} to enrich
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

Now `logger.info("order placed", extra={"order_id": 123, "amount": 99.99})` produces a record you can query as `level:INFO AND order_id:123` in Kibana.

### 4.3 Retention tiers and cost control

Log volume grows faster than budgets. Tier the storage:

| Tier | Age | Backend | Use |
|------|-----|---------|-----|
| Hot | 0-7 days | SSD-backed Elasticsearch | Live debugging |
| Warm | 7-30 days | HDD-backed nodes | Recent investigations |
| Cold | 30-90 days | S3 with searchable snapshots | Compliance, ad-hoc |
| Archive | 90 days - 7 years | Glacier | Legal hold |

Index Lifecycle Management (ILM) automates the moves. The single biggest cost-killer is to *not* index fields you never query - mark them as `enabled: false` in the mapping.

### 4.4 What to never log

- Passwords, API keys, JWTs (yes, even the prefix).
- Full credit-card numbers, SSNs, raw biometrics.
- Personal data without a documented retention policy.
- Request bodies of authentication endpoints.

A logging accident is a data breach.

## 5. Auto-scaling Without the Pain

### 5.1 Three flavours

- **Reactive**: scale on observed load (CPU, request rate, queue depth). Default; always have it on.
- **Scheduled**: pre-scale before predictable peaks (Black Friday at 9 AM, daily batch at midnight).
- **Predictive**: ML-based extrapolation from historical patterns. Useful when scale-up takes minutes and traffic ramps fast.

Combine all three. Predictive smooths the average; scheduled handles known spikes; reactive catches surprises.

### 5.2 Kubernetes HPA, done right

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
      stabilizationWindowSeconds: 0          # respond instantly
      policies:
        - type: Percent, value: 100, periodSeconds: 30
        - type: Pods,    value: 4,   periodSeconds: 30
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300        # wait 5 min before shrinking
      policies:
        - type: Percent, value: 25, periodSeconds: 60
```

The asymmetry is the point: **scale up fast, scale down slow**. A premature scale-down right before another traffic burst causes the next request flood to land on a too-small fleet.

### 5.3 Failure modes

- **Flapping**: scale-up triggers a metric drop, which triggers scale-down, which triggers scale-up. Cure: stabilisation windows, hysteresis.
- **Stampede**: 50 new pods all initialise the same JIT cache against the database, knocking it over. Cure: warm pools, slower ramp, connection limits.
- **Min-too-low**: at 3 AM you scaled to 2 pods; one crashes; the other is overwhelmed during recovery. Always set `minReplicas` to survive losing one node.
- **Max-not-set**: a runaway loop scales to 500 pods and bankrupts you. Always set `maxReplicas`.

## 6. Cost Optimisation

| Strategy | Typical saving | Effort | Risk |
|----------|---------------|--------|------|
| Right-sizing | 20-40% | Low | Low - resize off-hours |
| Savings Plans / RIs (1-3y) | 30-70% | Medium | Lock-in; commit only to baseline |
| Spot / preemptible | up to 90% | Medium | Interruption - only fault-tolerant work |
| Auto-shutdown of non-prod | 50-70% | Low | None for dev/staging |
| Storage tiering (S3 IA, Glacier) | 50-80% | Low | Retrieval latency |
| Region selection | up to 40% | Low | Latency, residency |
| Egress reduction (CDN, peering) | varies, often huge | Medium | None |

A short script that finds the most common waste - idle EC2 instances - is worth running monthly:

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

The follow-up matters more than the script: schedule the report, route it to the team that owns each instance (via the `Owner` tag), and require a written justification for any instance that survives three consecutive monthly reports.

## 7. SRE: Error Budgets Drive Engineering Priority

![Error Budget](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig5_error_budget.png)

The Google SRE book reduces a complex topic to four ideas:

- **SLI** (Service Level Indicator): a metric that tracks user-facing quality. Availability, latency, freshness.
- **SLO** (Service Level Objective): a target for the SLI over a window. "99.9% of requests over 30 days."
- **Error budget** = `1 - SLO`. The agreed-upon amount of unreliability per window. For 99.9% over 30 days that is 43 minutes.
- **Burn rate**: how fast the budget is being spent.

The real innovation is what you do with the budget:

- **Budget healthy (>50%)** -> ship features, take risks.
- **Budget caution (20-50%)** -> slow risky changes, prioritise reliability work.
- **Budget critical (<20%)** -> feature freeze, mandatory reliability sprint.
- **Budget exhausted** -> nobody ships anything that is not a fix; the SRE team can veto deployments.

This makes reliability a *shared* concern. Product cannot demand more features when the budget is gone; SRE cannot demand 100% availability when the budget is healthy. The number arbitrates.

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

`slo:budget:remaining` is the number you put on the executive dashboard. When it crosses 50%, 20%, 0%, the policy kicks in.

### 7.2 Other SRE practices that compound

- **Toil reduction.** Track time spent on repetitive operational work. Cap it (Google: 50%). Above the cap, write code instead.
- **Blameless postmortems.** Focus on the system, not the human. The question is "what conditions allowed this mistake to cause harm?" - never "who did it?". Action items have owners and ship dates; un-shipped action items are themselves an incident.
- **Game days.** Once a quarter, deliberately break something in a controlled window. Did the alerts fire? Did the runbook work? Did anyone notice in time?
- **Capacity planning.** Project demand + headroom; provision in advance; rehearse the scale-out so it actually works when needed.

## 8. GitOps: Git as the Source of Truth

```
Developer commit -> CI builds image -> CI updates manifest in config repo
                                                 |
                                       ArgoCD / Flux watches repo
                                                 |
                                        Reconciles cluster to git state
```

GitOps removes a whole class of mistakes by removing a whole class of capabilities. Nobody runs `kubectl apply` against production. The cluster reconciles itself to whatever is in the config repo, and the only way to change the cluster is to change Git.

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

## 9. The Operations Checklist

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

---

## Series Navigation

| Part | Topic |
|------|-------|
| 1 | [Fundamentals and Architecture](/en/cloud-computing-fundamentals/) |
| 2 | [Virtualization Technology Deep Dive](/en/cloud-computing-virtualization/) |
| 3 | [Storage Systems and Distributed Architecture](/en/cloud-computing-storage-systems/) |
| 4 | [Network Architecture and SDN](/en/cloud-computing-networking-sdn/) |
| 5 | [Security and Privacy Protection](/en/cloud-computing-security-privacy/) |
| **6** | **Operations and DevOps Practices (you are here)** |
| 7 | [Cloud-Native and Container Technologies](/en/cloud-computing-cloud-native-containers/) |
| 8 | [Multi-Cloud and Hybrid Architecture](/en/cloud-computing-multi-cloud-hybrid/) |
