---
title: "云原生与容器技术"
date: 2024-09-07 09:00:00
tags:
  - 云原生
  - Docker
  - 云计算
  - 微服务
  - Kubernetes
categories: 云计算
series:
  name: "云计算"
  part: 7
  total: 8
lang: zh-CN
mathjax: false
description: "从第一性原理理解云原生：容器在内核层面到底做了什么、Kubernetes 怎样工作、Service Mesh 何时值得引入、整套技术栈在生产环境如何拼装。"
---
从单体架构到云原生，是过去十年软件工程最重要的范式转变。容器和 Kubernetes 这个标题广为人知，但更值得讲的是：**为什么是这套技术栈赢了？每一层究竟在做什么？哪些接缝决定了你的平台是丝滑还是迷宫？**

本文按层级走完整个云原生栈。从背后的架构动因开始，深入容器在 Linux 内核层面的真实形态，向上到 Kubernetes 编排，剖析 Service Mesh 在何时值得它的复杂度，最后讲清 Helm 打包与 GitOps 交付。所有示例都是生产可用的：可直接复用的 Dockerfile、真实 manifest，以及在生产环境中真正重要的取舍。

## 你将学到

- 12 要素应用方法论，以及**每条要素背后的设计动机**
- 从内部理解容器：namespace、cgroup、union 文件系统、镜像分层
- Docker 生产实践：多阶段构建、安全加固、Compose 用于本地开发
- Kubernetes 架构：控制平面如何通过协调循环驱动 Worker 节点
- 工作负载原语：Pod、Service、Deployment、StatefulSet、DaemonSet、Job
- 网络：CNI 插件、NetworkPolicy、Ingress，以及 Istio 何时真正回本
- 存储：PV/PVC 动态供给，以及 `ReadWriteMany` 的真实代价
- Helm 打包、Release 历史、回滚的真实工作机制
- 微服务模式：熔断、Saga、API 网关
- 基于 ArgoCD 的 GitOps 交付与它强制带来的运维纪律

## 前置知识

- 熟悉 Linux 命令行与基础网络（路由、 DNS 、 TCP）
- 理解 HTTP/REST 与 Web 应用、数据库的协作方式
- 推荐先读本系列前 6 篇，特别是[虚拟化](/zh/cloud-computing-virtualization/)、[网络](/zh/cloud-computing-networking-sdn/)、[运维与 DevOps](/zh/cloud-computing-operations-devops/)

---

## 云原生：变了什么、为什么变

云原生不等于"把东西放到云上跑"。一台被 lift-and-shift 上云的虚拟机在云上，但并不"云原生"。CNCF 的定义说得很精确：

> 云原生技术使组织能够在公有云、私有云和混合云这类现代动态环境中构建和运行可扩展应用。容器、服务网格、微服务、不可变基础设施和声明式 API 是其代表性技术。

这句话背后真正在起作用的是三个理念：

1. **不可变基础设施。** 服务器不是要打补丁照看的宠物，而是可以替换的牲口。新版本永远是新镜像，绝不在原地修改。这一刀切掉了所有的"配置漂移"，而配置漂移是生产事故的最大来源之一。
2. **声明式 API。** 你描述**期望状态**（"我要 3 个 v1.4 副本，每个 500 MB 内存"），平台负责让现实匹配期望。反面是命令式脚本——"先做第一步，再做第二步"——一旦现实和脚本的假设不一致就会断裂。
3. **每一层都松耦合。** 服务独立、部署独立、故障独立、扩缩容独立。代价是更多移动部件；收益是没有任何单一部件能拖垮全局。

### 单体 vs 微服务：把取舍摆在桌面上

![单体 vs 微服务架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig1_microservices_vs_monolith.png)

上图展示了结构差异，但真正的故事在这四个数字里：

| 维度 | 单体 | 微服务 |
|---|---|---|
| **部署单元** | 1 个二进制 | N 个独立服务 |
| **扩缩容单元** | 整个应用 | 每个服务独立 |
| **技术栈** | 单一语言/运行时 | 多语言按服务选 |
| **故障爆炸半径** | 100% | 1 个服务（配合熔断） |

微服务并不绝对更好，它用简单性换取独立性：你付出分布式系统的复杂度（网络故障、最终一致性、分布式追踪、契约版本管理），换来部署、扩缩容、故障隔离的自由。**判断标准**：如果团队小到能用两个披萨喂饱、发布节奏是按月计的，**写一个结构良好的单体几乎肯定是更对的选择**。引入微服务的临界点是：团队之间的协调成本开始主导工程时间。

### 12 要素应用：一份生存指南

[12 要素方法论](https://12factor.net/)（Heroku ， 2011 ）早于 Kubernetes ，但已经成为容器化服务被默认期待遵守的运维契约。每一条要素都是为了让某种特定的故障模式**变得不可能**：

| 编号 | 要素 | 为什么重要 |
|---|---|---|
| 1 | **代码库** -- 一份代码、多种部署 | 同一份代码、不同配置 ⇒ 可靠的环境晋升路径 |
| 2 | **依赖** -- 显式声明并隔离 | "在我机器上能跑"成为不可能 |
| 3 | **配置** -- 存于环境变量 | 同一镜像跑遍 dev/staging/prod |
| 4 | **后端服务** -- 视为附加资源 | 换数据库只改 URL ，不重构代码 |
| 5 | **构建/发布/运行** -- 严格分离 | Release 不可变、可回滚 |
| 6 | **进程** -- 无状态、无共享 | 任意副本都能服务任意请求 |
| 7 | **端口绑定** -- 自包含 | 无需依赖外部服务器（Tomcat 、 IIS） |
| 8 | **并发** -- 通过进程模型扩展 | 默认水平扩缩容 |
| 9 | **易处理** -- 快速启动、优雅关闭 | 自动扩缩容与滚动更新可用 |
| 10 | **开发/生产对等** -- 环境尽量一致 | 缩小生产惊喜面 |
| 11 | **日志** -- 写到 stdout 的事件流 | 平台聚合，应用不直接写文件 |
| 12 | **管理进程** -- 一次性，同环境 | 数据库迁移不另起一套栈 |

违反某条要素有时是对的（要素 6 在有状态系统中真的很难），但**每次违反都是你应该清楚自己背上的债**。

## 容器：到底是什么

常见的心智模型是"容器就是轻量虚拟机"。这个模型在重要细节上**是错的**。**容器不是虚拟化，是进程隔离。** 一个容器就是一个 Linux 进程（或进程树），只不过内核被指示去对它撒谎，让它以为系统长成另一个样子。

干这件事的是三个 Linux 内核特性：

1. **namespace（命名空间）** -- 给进程一份独立的资源视图（PID 、 network 、 mount 、 UTS 、 IPC 、 user 、 cgroup）。在 PID namespace 内，容器看见自己是 PID 1 ，看不见 namespace 外的进程。
2. **cgroups（v2）** -- 强制资源配额（CPU 、内存、 IO 、 PID 数）。设了 `--memory=512m` ，超限时内核就杀掉进程。
3. **联合文件系统**（今天默认是 overlay2） -- 把只读镜像层堆叠起来，在最上面挂一个每容器独有的可写薄层，实现写时复制语义。

仅此而已。**容器共享宿主机内核**，没有 hypervisor ，没有第二个操作系统。代价是：启动 ~50 ms（虚机要 ~30 s），开销 ~5 MB（虚机要 ~500 MB），单机密度上百（虚机几十）。

### 镜像分层：让构建变快的缓存

![Docker 镜像分层](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig2_docker_layers.png)

每条 Dockerfile 指令产生一个新层。层通过 union FS 堆叠；相同层在镜像之间、宿主机之间会被去重。**这就是为什么两个共享同一基础的良好结构镜像，最终大小可能只差几兆，即使基础镜像有几个 G 。**

由此衍生两条实战准则：

**1. 按缓存复用顺序写 Dockerfile 。** 几乎不变的（系统包、语言运行时）放最前；每次提交都变的（应用代码）放最后。命中缓存的构建是秒级，冷构建是分钟级。

```dockerfile
FROM python:3.12-slim

# 第 1 层：系统依赖（约一月一变）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# 第 2 层：Python 依赖（约一周一变）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 第 3 层：应用代码（每次提交都变）
COPY . /app
WORKDIR /app
CMD ["gunicorn", "-w", "4", "app:app"]
```

如果你先写 `COPY . .` ，那每次代码改动都会让依赖层失效，整个 pip install 重做一遍。出货的差距是 **10 倍构建时间**。

**2. 镜像大小比你想的更重要。** 小镜像拉得快、启动快、攻击面小，在 Knative 、 KEDA 这类工具下冷启动也快。多阶段构建让你在重镜像里编译，在轻镜像里出货：

```dockerfile
# Builder 阶段：完整工具链（约 800 MB）
FROM golang:1.22-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/server ./cmd/server

# Runtime 阶段：只有二进制（约 10 MB）
FROM gcr.io/distroless/static-debian12
COPY --from=builder /out/server /server
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/server"]
```

最终镜像里只有编译好的二进制——没有 shell ，没有包管理器，没有源码。这是安全和运维双重收益。

### 生产级 Dockerfile 实践清单

| 实践 | 为什么 |
|---|---|
| **固定 tag，永远不用 `latest`** | 可复现构建；`latest` 是会动的目标 |
| **以非 root 身份运行**（`USER 1000`） | 容器逃逸难度大幅上升 |
| **每容器一个进程** | 用 supervisord 之类的 init 系统会向 K8s 隐藏故障 |
| **使用 `.dockerignore`** | `node_modules/` 、 `.git/` 、 `.env` 永远不该进入构建上下文 |
| **合并相关 `RUN`** | 每个 `RUN` 是一层；后面单独 `RUN` 清理并不会缩小镜像 |
| **使用 `HEALTHCHECK`** | 告诉运行时进程**就绪**，而非仅"还活着" |
| **签名并扫描镜像** | CI 中用 `cosign` 签名、 `trivy`/`grype` 扫描 CVE |

### Docker Compose ：本地开发的正确工具

Compose 用于本地开发和小规模单机部署。它**不是**生产编排器——没有自愈、没有滚动更新、没有跨主机水平扩缩。但是用来"在我的笔记本上把 API 、数据库、 Redis 都拉起来"，没有比它更好的：

```yaml
services:
  web:
    build: .
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://postgres:dev@db:5432/myapp
      REDIS_URL: redis://redis:6379
    depends_on:
      db: { condition: service_healthy }

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: dev
      POSTGRES_DB: myapp
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 5
    volumes: [pgdata:/var/lib/postgresql/data]

  redis:
    image: redis:7-alpine

volumes:
  pgdata:
```

`depends_on` 配 `condition: service_healthy` 是被低估的特性——它意味着 web 只在数据库**真正可以接受连接**之后才启动，而不是容器一存在就启动。

## Kubernetes ：赢得编排之战的那一个

Kubernetes（ K8s ）源自 Google 内部的 Borg 系统，已经把容器编排之战赢得彻底——在大多数对话里，"容器编排"和" Kubernetes "已经是同义词。理解它**怎么工作**——而不是只会照抄 YAML ——是从"会用"走到"能设计平台"的关键。

### 架构：控制平面与 Worker 节点

![Kubernetes 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig3_kubernetes_architecture.png)

**控制平面**是大脑。四个核心组件：

| 组件 | 职责 |
|---|---|
| **kube-apiserver** | 唯一直接与 etcd 通信的组件。其他组件都通过它访问。HTTPS 上的 REST/gRPC 。 |
| **etcd** | 集群唯一可信源。强一致的分布式 KV 存储。**etcd 没了，集群就没了。 一定要备份。** |
| **kube-scheduler** | 决定新 Pod 调度到哪个节点（依据资源请求、 taint 、亲和性等）。 |
| **kube-controller-manager** | 跑各种协调循环——Deployment 、 ReplicaSet 、 Node 等控制器。 |

每个 **Worker 节点**运行：

| 组件 | 职责 |
|---|---|
| **kubelet** | 节点代理。从 API Server 接收 Pod 规约，让容器运行时跑起来，把状态汇报回去。 |
| **kube-proxy** | 配置 iptables/IPVS 规则，让 Service IP 路由到正确的 Pod 。 |
| **容器运行时** | 真正跑容器的（containerd 或 CRI-O ；现在已经不直接用 Docker） |

### 协调循环：Kubernetes 的核心思想

Kubernetes 中**所有**控制器都是同一个循环：

```
while True:
    desired = read_desired_state_from_api_server()
    actual  = observe_actual_state()
    if desired != actual:
        take_action_to_close_the_gap()
```

你声明要 3 个副本。 ReplicaSet 控制器观察到只有 2 个，于是再创建 1 个。某个节点挂了， 1 个 Pod 不可用。控制器观察到只有 2 个在跑，于是再创建 1 个。**你从未告诉它该做什么，你只告诉它你想要什么。**

这就是为什么 Kubernetes 感觉像在自愈——并不是有个独立的"自愈"功能，而是**整个系统就是一个自愈循环**构造出来的。这也解释了它有时为什么神秘：如果现实跟期望不一致，一定有什么地方在静默地重试。

### Pod 、 Service 、 Deployment ：日常主力

**Pod** 是最小可部署单元——一个或多个共享网络与存储的容器。**实际上"一个 Pod 一个容器"是默认值**；例外是当 sidecar （日志代理、 service mesh proxy ）需要与主应用共享网络/卷时。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web
  labels: { app: web }
spec:
  containers:
    - name: web
      image: nginx:1.27-alpine
      ports: [{ containerPort: 80 }]
      resources:
        requests: { memory: "64Mi", cpu: "100m" }   # 保证值
        limits:   { memory: "128Mi", cpu: "500m" }  # 上限
      readinessProbe:
        httpGet: { path: /, port: 80 }
        periodSeconds: 5
      livenessProbe:
        httpGet: { path: /healthz, port: 80 }
        initialDelaySeconds: 30
```

**资源 requests 与 limits 是承重的区别。** requests 是调度器用来选节点的、内核**保证**给你的；limits 是硬上限，超过就被内核杀掉（内存）或限流（CPU）。requests 设太低会导致节点过度认领，引发 OOM ；limits 设太低会引发限流，看起来像应用变慢。

**探针决定流量与生命周期。** Readiness 控制流量（"我准备好接请求了吗？"）；Liveness 控制重启（"我是否还活着？"）。**很多生产事故来自配错的 liveness probe ：它不停重启一个其实没问题、只是慢了点的 Pod 。**

**Service** 在一组 Pod 前提供稳定的虚拟 IP 和 DNS 名。 Pod 来来去去， Service IP 不变。

```yaml
apiVersion: v1
kind: Service
metadata: { name: web }
spec:
  selector: { app: web }
  ports: [{ port: 80, targetPort: 80 }]
  type: ClusterIP    # 仅集群内；对外用 LoadBalancer
```

**Deployment** 管理无状态工作负载的滚动更新与回滚。它拥有一个 ReplicaSet ， ReplicaSet 拥有 Pod 。改镜像时， Deployment 创建新 ReplicaSet ，按你定义的策略迁移流量：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: web }
spec:
  replicas: 3
  selector: { matchLabels: { app: web } }
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }   # 零停机
  template:
    metadata: { labels: { app: web } }
    spec:
      containers:
        - name: web
          image: ghcr.io/myorg/web:v1.4.2
          # ...
```

`maxUnavailable: 0` 意为"更新过程中绝不降低容量"——更安全。`maxSurge: 1` 意为"一次只多起 1 个 Pod "——限制 rollout 的激进程度。

```bash
kubectl set image deployment/web web=ghcr.io/myorg/web:v1.4.3
kubectl rollout status deployment/web        # 等待 rollout 完成
kubectl rollout history deployment/web       # 查看所有 revision
kubectl rollout undo deployment/web          # 回滚到上一版本
```

### 何时需要 StatefulSet 、 DaemonSet 、 Job

绝大多数工作负载都是 Deployment 。例外值得知道：

- **StatefulSet** -- 有序、有名的 Pod （`db-0` 、 `db-1` 、 `db-2`），带稳定的持久化卷。用于数据库、消息队列，任何需要"身份"的场景。更新更慢——一次一个 Pod ，按顺序。
- **DaemonSet** -- 每个节点一个 Pod 。用于日志采集器、监控代理、 CSI 驱动、 CNI 插件。
- **Job** -- 跑到完成。用于数据库迁移、批处理。
- **CronJob** -- 按计划跑的 Job 。用于备份、定期报表。

### 托管 Kubernetes ：理性默认选择

自建控制平面是可能的（`kubeadm` 、 `kops` 、 `kubespray`），但很少是对的——你要自己背上 etcd 备份、证书轮转、版本升级、安全补丁。托管服务把这些重量挑走：

```bash
# AWS EKS
eksctl create cluster --name prod --region us-west-2 --nodes 3

# Google GKE
gcloud container clusters create prod --num-nodes 3 --region us-central1

# Azure AKS
az aks create --name prod --resource-group rg --node-count 3
```

控制平面定价大约 73-150 美元/月，再加上你选的 worker 节点成本。**对绝大多数组织，这点费用远小于它节省的工程时间。**

## 网络：从 CNI 到 Service Mesh

Kubernetes 网络分三层，每层有自己的原语：

| 关注点 | 原语 | 实现 |
|---|---|---|
| **Pod 间连通** | 扁平 L3 网络 | CNI 插件（Calico 、 Cilium 、 Flannel） |
| **L4 访问策略** | NetworkPolicy | 由 CNI 强制实施 |
| **L7 流量管理** | Ingress / Service Mesh | NGINX 、 Istio 、 Linkerd |

### CNI 插件如何选

每个 Pod 拿到自己的 IP 。让这件事真正运转起来的是 Container Network Interface （CNI）插件。

| 插件 | 路线 | 何时选 |
|---|---|---|
| **Flannel** | VXLAN overlay ，无策略 | 仅实验/开发 |
| **Calico** | BGP 路由 + iptables 策略 | 生产环境的默认主力 |
| **Cilium** | eBPF ，内核级策略 + L7 可见性 | 性能 + 安全；现代默认 |

Cilium 越来越成为新集群的答案：它绕开 iptables（在大规模集群下会成为瓶颈）、强制 L7 策略（HTTP method/path 级），还顺带提供流量可见性。

### NetworkPolicy ：默认拒绝才是目标

默认情况下，**任何 Pod 都能访问任何其他 Pod 。** 这很方便也很糟糕。NetworkPolicy 修复它：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: api-policy, namespace: prod }
spec:
  podSelector: { matchLabels: { app: api } }
  policyTypes: [Ingress, Egress]
  ingress:
    - from:
        - podSelector: { matchLabels: { app: frontend } }
      ports: [{ protocol: TCP, port: 8080 }]
  egress:
    - to:
        - podSelector: { matchLabels: { app: postgres } }
      ports: [{ protocol: TCP, port: 5432 }]
    - to:                              # 放行 DNS
        - namespaceSelector: {}
          podSelector: { matchLabels: { k8s-app: kube-dns } }
      ports: [{ protocol: UDP, port: 53 }]
```

**真正回本的纪律：先在 namespace 里下一个默认拒绝策略，再显式放行。** 被攻陷的 Pod 就无法再轻易扫描整个集群。

### Service Mesh ：sidecar 何时值这个钱

![Service Mesh （Istio）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig4_service_mesh_istio.png)

Service Mesh （Istio 、 Linkerd）在每个 Pod 注入一个代理（Envoy）作为 sidecar 。所有服务间流量都经过代理，于是你**无需改应用代码**就拿到 mTLS 、重试、超时、流量切分、黄金指标可观测性。

```yaml
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata: { name: reviews }
spec:
  hosts: [reviews]
  http:
    - route:
        - destination: { host: reviews, subset: v1 }
          weight: 90
        - destination: { host: reviews, subset: v2 }
          weight: 10        # 10% 流量灰度到 v2
---
apiVersion: networking.istio.io/v1
kind: DestinationRule
metadata: { name: reviews }
spec:
  host: reviews
  trafficPolicy:
    outlierDetection:        # 熔断
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
  subsets:
    - name: v1
      labels: { version: v1 }
    - name: v2
      labels: { version: v2 }
```

代价：每个 Pod 多一个 sidecar （约 50 MB 内存、约 5 ms 延迟开销）、要管理一个控制平面、学习曲线更陡。**诚实的经验法则**：服务数 < 10 个，YAML 比问题本身还复杂；服务数 50+ 、工程师在应用代码里花真金白银的时间处理重试/超时/mTLS ，则 Mesh 是明牌的胜利。

### Ingress ：前门

集群外的 HTTP 流量进来，配 Ingress （加上 NGINX 、 Traefik 或云上 ALB/GLB 之类的控制器）通常比每服务一个 LoadBalancer 简单：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts: [app.example.com]
      secretName: web-tls
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service: { name: web, port: { number: 80 } }
```

配上 `cert-manager` ，证书自动续期。再加 ExternalDNS ，整套 DNS+TLS+路由都成了声明式。

## 存储：把持久卷做对

Pod 是短暂的，数据不是。 Kubernetes 的存储抽象把"我要什么样的存储"（PVC）、"怎么供给"（StorageClass）、"底层是什么"（provisioner）解耦：

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata: { name: fast-ssd }
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
  iops: "3000"
allowVolumeExpansion: true
reclaimPolicy: Retain                # PVC 删除时不删卷
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: pgdata }
spec:
  accessModes: ["ReadWriteOnce"]     # 一个节点挂载
  storageClassName: fast-ssd
  resources: { requests: { storage: 100Gi } }
```

**真正重要的访问模式：**

- `ReadWriteOnce`（RWO） -- 单节点读写挂载。块存储（EBS 、 PD 、 Azure Disk）。**默认且最快。**
- `ReadWriteMany`（RWX） -- 多节点读写挂载。需要网络文件系统（EFS 、 Filestore 、 Azure Files 、 CephFS）。更慢，且一致性模型由文件系统决定。**除非你真的需要可写共享状态，否则避免。**
- `ReadOnlyMany`（ROX） -- 分发静态内容。

**`reclaimPolicy: Retain` 是生产数据的安全默认。** 用 `Delete` 时，删 PVC 会**静默销毁卷**。用 `Retain` 时，你必须手动确认——这条规则救过无数个周五下午。

数据库跑在 K8s 上，**优先用 Operator** （Postgres Operator 、 MongoDB Operator）而不是自己撸 StatefulSet 。它们处理了那些容易低估的失败模式（主备切换、备份、滚动升级）。

## Helm ：把打包做对

服务一多， manifest 就开始爆炸。 Helm 把模板化的 manifest 打成版本化、可复用、可参数化的"chart"。

![Helm Chart 与 Release 历史](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig5_helm_charts.png)

一个 chart 是一组模板加一份 `values.yaml` 默认值。安装时，按环境覆盖 values ：

```bash
helm install web ./charts/web -n prod -f values.prod.yaml
helm upgrade web ./charts/web -n prod -f values.prod.yaml --atomic
helm rollback web 3 -n prod                  # 回到 revision 3
helm history web -n prod                     # 完整 release 历史
```

**`--atomic` 是被严重低估的开关**：升级失败（任何 post-install hook 或就绪检查没过）时， Helm 自动回滚。这一个 flag ，把 Helm 从"模板"提升成了"事务化的发布管理"。

### 何时**不要**用 Helm

Helm 模板是 Go template + Sprig 函数。简单 chart 还行，复杂 chart 那种"YAML 嵌在模板字符串里"的写法会变得很痛苦（空白符 bug 、 schema 校验只在安装时才发现）。**正在崛起的替代品：**

- **Kustomize**（已内置 kubectl） -- 不模板，靠 patch 与 overlay 。小变体场景更简单。
- **CUE / KCL / Pkl** -- 带真正 schema 的强类型配置语言。前期成本高，后期翻车少很多。

发第三方软件（数据库、监控栈），Helm 是事实上的通用语言。**你自己的应用**，可以认真评估上述替代品。

## 在生产中存活的微服务模式

成熟系统里反复出现的模式：

**API 网关** -- 客户端的统一入口（鉴权、限流、路由、响应整形）。Kong 、 Envoy Gateway 或云上托管网关。让客户端代码简单，安全策略集中。

**熔断（Circuit Breaker）** -- 下游不稳定时停止调用，让调用方快速失败，给下游恢复空间。Service Mesh （Istio outlierDetection ）透明处理；Resilience4j 、 Hystrix 这类库在进程内做。

**Saga** -- 不要分布式锁的分布式事务。每一步都有补偿动作；如果第 4 步失败，按相反顺序执行 1-3 的补偿。两种风格：编排（一个 coordinator 驱动步骤）和编舞（服务发事件互相驱动）。**编排更易推理；编舞更易扩展。**

**Outbox 模式** -- "DB 写入 + 事件发出"原子化。把事件写到与业务同一事务里的 outbox 表；另一个进程读 outbox 并发出。**解决了所有事件驱动系统迟早会被咬到的"双写问题"。**

**每服务自有数据库** -- 每个服务拥有自己的数据；不共享 DB 。代价是真实的（join 变成 API 调用），但**共享数据库通过 schema 把服务耦合在一起，悄无声息地把微服务的意义抹掉了**。

## CI/CD 与 GitOps

Kubernetes 现代交付管线长这样：

```
git push -> CI（测试、构建、扫描、签名、推镜像）
         -> CI 更新 manifest 仓（image tag bump）
         -> ArgoCD 检测到差异并同步集群
```

关键转变是 **GitOps** ：集群状态由 Git 定义。 ArgoCD （或 Flux）持续把集群与 Git 仓对齐。两个大胜利：

1. **审计链。** 每一次变更都是一次提交。"谁在凌晨两点改了 prod ？"`git blame` 。
2. **灾难恢复。** 集群没了？从 manifest 仓 `kubectl apply` 回来即可。

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

`selfHeal: true` 意味着"如果有人手动 `kubectl edit` 了某资源， ArgoCD 会把它改回来"。**这就是 GitOps 强制的纪律——集群状态以 Git 为准，而不是某人终端为准。**

## 生产环境真正用得上的命令

```bash
# 集群概览
kubectl cluster-info
kubectl get nodes -o wide

# 看什么在哪儿跑
kubectl get pods -A -o wide
kubectl top pods -A                   # CPU/内存实际值

# 调试故障 Pod
kubectl describe pod <name> -n <ns>   # 事件、状态、调度
kubectl logs <name> -n <ns> -f        # 实时日志
kubectl logs <name> -n <ns> --previous  # 上一个崩溃容器的日志
kubectl exec -it <name> -n <ns> -- sh

# 集群最近事件（"为什么会这样"的金矿）
kubectl get events -A --sort-by='.lastTimestamp' | tail -30

# 扩缩容与发布
kubectl scale deploy/web --replicas=5
kubectl rollout restart deploy/web    # 强制重启 rollout，用来让新 Secret 生效
```

最有用的一对组合：`kubectl describe`（状态、事件、调度决策） + `kubectl logs --previous`（刚崩掉的容器里发生了什么）。

## 生产就绪检查表

在宣布"工作负载生产就绪"之前：

- [ ] 多阶段 Dockerfile 、非 root 用户、 distroless 或最小化基础镜像
- [ ] 镜像按 digest 钉死（或至少不可变 tag），CI 中签名并扫描
- [ ] 每个容器都设置了 resource requests **和** limits
- [ ] Liveness **和** readiness 探针都有（readiness 控流量，liveness 控重启）
- [ ] 配置 PodDisruptionBudget ，集群维护不会让你跌破 `minAvailable`
- [ ] 流量有变化的就配 HorizontalPodAutoscaler
- [ ] NetworkPolicy 默认拒绝 + 显式放行
- [ ] Secret 放在外部存储（Vault 、 AWS Secrets Manager 、 External Secrets Operator），而非明文 Secret
- [ ] 日志写 stdout 、结构化（JSON），聚合到中央系统
- [ ] 暴露指标（Prometheus 格式）并有监控面板
- [ ] 接入分布式追踪（OpenTelemetry）
- [ ] 备份已测试（特别是 StatefulSet）
- [ ] 常见故障模式有 runbook

满足这套清单的工作负载并非不可破——但**留下来的故障模式将是有趣的，不是丢人的**。

---

## 系列导航

| 编号 | 主题 |
|------|-------|
| 1 | [基础与架构](/zh/cloud-computing-fundamentals/) |
| 2 | [虚拟化技术深入](/zh/cloud-computing-virtualization/) |
| 3 | [存储系统与分布式架构](/zh/cloud-computing-storage-systems/) |
| 4 | [网络架构与 SDN](/zh/cloud-computing-networking-sdn/) |
| 5 | [安全与隐私保护](/zh/cloud-computing-security-privacy/) |
| 6 | [运维与 DevOps 实践](/zh/cloud-computing-operations-devops/) |
| **7** | **云原生与容器技术（你在这）** |
| 8 | [多云与混合云架构](/zh/cloud-computing-multi-cloud-hybrid/) |
