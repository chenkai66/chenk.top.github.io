---
title: "云计算（三）：云原生与容器技术"
date: 2023-03-11 09:00:00
tags:
  - Cloud Computing
  - Cloud Native
  - Docker
  - Kubernetes
  - Microservices
categories: 云计算
series: cloud-computing
lang: zh
mathjax: false
description: "从第一性原理理解云原生：容器在内核层面到底做了什么、Kubernetes 怎样工作、Service Mesh 何时值得引入、整套技术栈在生产环境如何拼装。"
disableNunjucks: true
series_order: 3
translationKey: "cloud-computing-3"
polished_by_qwen_max: true
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/illustration_1.jpg)

从单体架构到云原生，是过去十年软件工程最重要的范式转变。容器和 Kubernetes 这个标题广为人知，但更值得讲的是：**为什么是这套技术栈赢了？每一层究竟在做什么？哪些接缝决定了你的平台是丝滑还是迷宫？**

本文按层级走完整个云原生栈。从背后的架构动因开始，深入容器在 Linux 内核层面的真实形态，向上到 Kubernetes 编排，剖析 Service Mesh 在何时值得它的复杂度，最后讲清 Helm 打包与 GitOps 交付。所有示例都是生产可用的：可直接复用的 Dockerfile、真实 manifest，以及生产环境里，真正重要的取舍。

## 你将学到

- 12 要素应用方法论，每条要素的设计动机
- 容器内部：namespace、cgroup、union 文件系统、镜像分层
- Docker 实践：多阶段构建、安全加固、Compose 本地开发
- K8s 架构：控制平面通过协调循环驱动 Worker 节点
- 工作负载原语：Pod、Service、Deployment、StatefulSet、DaemonSet、Job
- 网络：CNI 插件、NetworkPolicy、Ingress，Istio 回本场景
- 存储：PV/PVC 动态供给，`ReadWriteMany` 的代价
- Helm 打包、Release 历史、回滚机制
- 微服务模式：熔断、Saga、API 网关
- ArgoCD GitOps 交付与运维纪律
## 前置知识

- 熟悉 Linux 命令行和基础网络（路由、DNS、TCP）
- 理解 HTTP/REST，知道 Web 应用和数据库如何交互
- 建议先读本系列前 6 篇，重点是[虚拟化](/zh/cloud-computing/virtualization/)、[网络](/zh/cloud-computing/networking-sdn/)、[运维与 DevOps](/zh/cloud-computing/operations-devops/)
## 云原生：变了什么、为什么变

云原生不等于"把东西放到云上跑"。一台 lift-and-shift 上云的虚拟机在云上，但不是云原生。CNCF 的定义很精准：

> 云原生技术让组织能在公有云、私有云和混合云等动态环境中构建和运行可扩展应用。容器、服务网格、微服务、不可变基础设施和声明式 API 是其代表。

这句话背后是三个核心理念：

1. **不可变基础设施。** 服务器不是宠物，而是牲口。新版本永远是新镜像，不在原地修改。这消除了配置漂移，而配置漂移是生产事故的主要来源。
2. **声明式 API。** 描述**期望状态**（"我要 3 个 v1.4 副本，每个 500 MB 内存"），平台负责匹配。命令式脚本——"先做第一步，再做第二步"——假设不对就断了。
3. **每一层松耦合。** 服务独立、部署独立、故障独立、扩缩容独立。代价是更多部件；收益是没有单点能拖垮全局。

### 单体 vs 微服务：取舍摆在桌面上

![单体 vs 微服务架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig1_microservices_vs_monolith.png)

上图展示结构差异，但重点在这四个数字：

| 维度 | 单体 | 微服务 |
|---|---|---|
| **部署单元** | 1 个二进制 | N 个独立服务 |
| **扩缩容单元** | 整个应用 | 每个服务独立 |
| **技术栈** | 单一语言/运行时 | 多语言按服务选 |
| **故障爆炸半径** | 100% | 1 个服务（配合熔断） |

微服务不绝对更好。它用简单性换独立性：付出分布式复杂度（网络故障、最终一致性、分布式追踪、契约版本管理），换来部署、扩缩容、故障隔离的自由。**判断标准**：团队小到两个披萨能喂饱、发布节奏按月计，写一个结构良好的单体几乎肯定更对。引入微服务的临界点是：团队协调成本主导工程时间。

### 12 要素应用：一份生存指南

[12 要素方法论](https://12factor.net/)（Heroku，2011）早于 Kubernetes，但已成为容器化服务默认遵守的运维契约。每条要素都是为了让某种故障模式**不可能发生**：

| 编号 | 要素 | 为什么重要 |
|---|---|---|
| 1 | **代码库** -- 一份代码、多种部署 | 同一份代码、不同配置 ⇒ 可靠晋升路径 |
| 2 | **依赖** -- 显式声明并隔离 | "在我机器上能跑"成为不可能 |
| 3 | **配置** -- 存于环境变量 | 同一镜像跑遍 dev/staging/prod |
| 4 | **后端服务** -- 视为附加资源 | 换数据库只改 URL，不重构代码 |
| 5 | **构建/发布/运行** -- 严格分离 | Release 不可变、可回滚 |
| 6 | **进程** -- 无状态、无共享 | 任意副本都能服务任意请求 |
| 7 | **端口绑定** -- 自包含 | 无需外部服务器（Tomcat、IIS） |
| 8 | **并发** -- 通过进程模型扩展 | 默认水平扩缩容 |
| 9 | **易处理** -- 快速启动、优雅关闭 | 自动扩缩容与滚动更新可用 |
| 10 | **开发/生产对等** -- 环境尽量一致 | 缩小生产惊喜面 |
| 11 | **日志** -- 写到 stdout 的事件流 | 平台聚合，应用不直接写文件 |
| 12 | **管理进程** -- 一次性，同环境 | 数据库迁移不另起一套栈 |

违反某条要素有时是对的（要素 6 在有状态系统中很难），但**每次违反都是我清楚自己背上的债**。
## 容器：到底是什么

很多人觉得“容器就是轻量虚拟机”。这个理解在关键点上**错了**。**容器不是虚拟化，而是进程隔离。** 容器就是一个 Linux 进程（或进程树），内核对它“撒谎”，让它以为系统是另一个样子。

实现这一点靠三个 Linux 内核特性：

1. **namespace** -- 给进程独立的资源视图（PID、网络、挂载、UTS、IPC、用户、cgroup）。在 PID namespace 里，容器看到自己是 PID 1，看不到外面的进程。
2. **cgroups（v2）** -- 限制资源使用（CPU、内存、IO、PID 数）。设了 `--memory=512m`，超限就杀进程。
3. **联合文件系统**（现在默认 overlay2） -- 把只读镜像层堆起来，上面加一个可写薄层，实现写时复制。

没了。**容器共享宿主机内核**，没有 hypervisor，没有第二个操作系统。代价是：启动 ~50 ms（虚机要 ~30 s），开销 ~5 MB（虚机要 ~500 MB），单机密度上百（虚机几十）。

### 镜像分层：让构建变快的缓存

![Docker 镜像分层](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig2_docker_layers.png)

每条 Dockerfile 指令生成一层。层通过 union FS 堆叠，相同层在镜像间、宿主机间去重。**这就是为什么两个共享基础镜像的镜像，最终大小可能只差几兆，即使基础镜像是几个 G。**

实战中有两条经验：

**1. 按缓存复用顺序写 Dockerfile。** 几乎不变的（系统包、语言运行时）放前面；每次提交都变的（应用代码）放后面。命中缓存的构建是秒级，冷构建是分钟级。

![GitOps 部署流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig_gitops_pipeline_zh.png)

核心是 **GitOps**：集群状态由 Git 定义。ArgoCD（或 Flux）持续对齐集群与 Git 仓。两大优势：

1. **审计链。** 每次变更都有提交记录。凌晨两点谁改了 prod？`git blame`。
2. **灾难恢复。** 集群挂了？从 manifest 仓 `kubectl apply` 即可恢复。

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata: { name: web, namespace: argocd }
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/k8s-manifests
    path: apps/web/overlays/prod
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: prod
  syncPolicy:
    automated: { prune: true, selfHeal: true }
    syncOptions: [CreateNamespace=true]
```

`selfHeal: true` 表示"有人手动 `kubectl edit` 资源，ArgoCD 会自动改回"。这就是 GitOps 的纪律——集群状态以 Git 为准，不看终端操作。
## 生产环境真正有用的命令

```bash
# 集群概况
kubectl cluster-info
kubectl get nodes -o wide

# 查看运行分布
kubectl get pods -A -o wide
kubectl top pods -A                   # CPU/内存实际值

# 调试问题 Pod
kubectl describe pod <name> -n <ns>   # 事件、状态、调度
kubectl logs <name> -n <ns> -f        # 实时日志
kubectl logs <name> -n <ns> --previous  # 崩溃容器的日志
kubectl exec -it <name> -n <ns> -- sh

# 最近集群事件（排查"为啥出问题"的关键）
kubectl get events -A --sort-by='.lastTimestamp' | tail -30

# 扩缩容与更新
kubectl scale deploy/web --replicas=5
kubectl rollout restart deploy/web    # 强制重新发布，用于加载新 Secret
```

最实用的组合：`kubectl describe`（状态、事件、调度） + `kubectl logs --previous`（刚崩溃容器的内部情况）。
## 生产就绪检查表

宣布工作负载生产就绪前，确认以下事项：

- [ ] 使用多阶段 Dockerfile，非 root 用户，distroless 或最小化基础镜像
- [ ] 镜像按 digest 固定（或至少使用不可变 tag），CI 中完成签名和扫描
- [ ] 每个容器都设置 resource requests 和 limits
- [ ] 配置 liveness 和 readiness 探针，readiness 控制流量，liveness 控制重启
- [ ] 设置 PodDisruptionBudget，避免集群维护时低于 `minAvailable`
- [ ] 流量波动时配置 HorizontalPodAutoscaler
- [ ] NetworkPolicy 默认拒绝，显式允许必要流量
- [ ] Secret 存储在外部系统（Vault、AWS Secrets Manager、External Secrets Operator），不使用明文 Secret
- [ ] 日志输出到 stdout，结构化为 JSON，聚合到中央系统
- [ ] 暴露 Prometheus 格式指标，并配置监控面板
- [ ] 接入分布式追踪（OpenTelemetry）
- [ ] 备份已测试，尤其是 StatefulSet
- [ ] 常见故障模式有对应 runbook

满足这些条件的工作负载并非无懈可击，但剩下的问题会是有趣的，而非丢人的。
