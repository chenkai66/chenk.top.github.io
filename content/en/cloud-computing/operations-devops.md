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
![Chapter concept illustration](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/illustration_1.jpg)

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

![Terraform Workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig_terraform_pipeline_en.png)

`plan` is the part you actually live in. It tells you what will change *before* anything changes. Code review happens against the plan output, not just the HCL.

### 2.2 A complete production module

![GitOps with ArgoCD/Flux](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/operations-devops/fig_gitops_argocd_en.png)

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
