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
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/illustration_1.jpg)

2017 年 GitLab 丢了六个小时的数据库状态。一位疲惫的工程师在事故处理中对错了服务器跑了 `rm -rf`。备份流程其实已经悄悄坏了几个月，但没人发现，因为没人在做恢复演练。教训不是"用 `rm` 要小心"。教训是：运维是一个**系统**——工具、运行手册、监控、自动化，以及围绕这一切的仪式。系统健康时，任何一个疲惫工程师都搞不挂生产；系统腐烂时，每一次深夜抢救都离灾难一个按键。

本文讲的就是怎么把这个系统建起来。在代码触达用户前把质量挡住的 CI/CD；让"生产环境"成为一个 Git 提交而不是雪花服务器的 IaC；能把噪声和信号分开的监控；真正能搜的日志；以及把救火工程化的 SRE 实践——错误预算、SLO、无指责复盘。

## 我会学到

- CI/CD 流水线：阶段、质量门、回滚，附完整 GitHub Actions 示例
- Terraform 实现基础设施即代码：工作流、状态管理、模块模式
- Prometheus + Grafana + Alertmanager 监控：抓取模型、PromQL、告警规则
- 集中化日志架构（EFK / ELK）：采集器、缓冲、处理器、保留分层
- 弹性伸缩，响应真实负载且不抖动
- 无需重写应用的成本优化
- SRE 实践：SLI / SLO / 错误预算、无指责复盘、GitOps
## 前置知识

- Linux 命令行熟练
- Git 和基本 CI/CD 概念
- 建议先阅读本系列前 5 篇

---

## 1. CI/CD 流水线：发布的"系统记录"

![CI/CD 流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig1_cicd_pipeline.png)

现代 CI/CD 流水线不只是自动化。它是代码进入生产的唯一路径，也是每次发布的系统记录：谁发了什么、跑过哪些测试、对哪版基础设施、后续如何。运维栈的其他部分都依赖这条主干。

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

![Terraform 工作流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig_terraform_pipeline_zh.png)

`plan` 是关键一步。它提前告诉你变更内容。Code review 看的是 plan 输出，不只是 HCL。

### 2.2 一份完整的生产模块

![基于 ArgoCD/Flux 的 GitOps](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/operations-devops/fig_gitops_argocd_zh.png)

GitOps 删掉了一类能力（直接 `kubectl apply`），也删掉了一类错误。集群自动 reconcile 到配置仓库的内容，改集群的唯一方法是改 Git。

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
- **灾难恢复** = "新集群指向同一个仓库，ArgoCD 自动同步"。
## 9. 运维检查单

**流水线**
- [ ] 变更全走流水线进生产，不手工 `apply`。
- [ ] 云端鉴权用 OIDC，不用长期 secret。
- [ ] 质量门让构建失败，admin 也不能绕过。
- [ ] SLO 异常触发自动回滚。

**基础设施**
- [ ] 所有资源用 Terraform / 等价 IaC 定义。
- [ ] 远程 state 带锁，服务和环境各一份 state 文件。
- [ ] 每个 PR 都贴 `terraform plan`，review 看 plan 不看 HCL。
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
