---
title: "云计算（三）：云原生与容器技术"
date: 2023-03-11 09:00:00
tags:
  - 云原生
  - Docker
  - 云计算
  - 微服务
  - Kubernetes
categories: 云计算
series: cloud-computing
lang: zh-CN
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

如果先写 `COPY . .`，每次代码改动都会让依赖层失效，pip install 重做一遍。差距是 **10 倍构建时间**。

**2. 镜像大小很重要。** 小镜像拉得快、启动快、攻击面小，在 Knative、KEDA 下冷启动也快。多阶段构建让你在重镜像里编译，在轻镜像里出货：

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

最终镜像只有编译好的二进制——没有 shell，没有包管理器，没有源码。这是安全和运维的双重收益。

### 生产级 Dockerfile 实践清单

| 实践 | 为什么 |
|---|---|
| **固定 tag，不用 `latest`** | 可复现构建；`latest` 是会动的目标 |
| **非 root 身份运行**（`USER 1000`） | 容器逃逸难度大幅上升 |
| **每容器一个进程** | 用 supervisord 会向 K8s 隐藏故障 |
| **使用 `.dockerignore`** | `node_modules/`、`.git/`、`.env` 不该进入构建上下文 |
| **合并相关 `RUN`** | 每个 `RUN` 是一层；后面单独清理不会缩小镜像 |
| **使用 `HEALTHCHECK`** | 告诉运行时进程**就绪**，而非仅“还活着” |
| **签名并扫描镜像** | CI 中用 `cosign` 签名、`trivy`/`grype` 扫描 CVE |

### Docker Compose：本地开发的正确工具

Compose 适合本地开发和小规模单机部署。它**不是**生产编排器——没有自愈、没有滚动更新、没有跨主机扩缩。但用来“在笔记本上拉起 API、数据库、Redis”，没有比它更好的：

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

`depends_on` 配 `condition: service_healthy` 是被低估的特性——它确保 web 只在数据库**真正可用**后启动，而不是容器一存在就启动。
## Kubernetes：赢得编排之战的那一个

Kubernetes（K8s）起源于 Google 的 Borg 系统，彻底赢下了容器编排之争。如今，“容器编排”和“K8s”几乎成了同义词。理解它**如何工作**——而不仅仅是抄 YAML——是从“会用”到“能设计平台”的关键。

### 架构：控制平面与 Worker 节点

![Kubernetes 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig3_kubernetes_architecture.png)

**控制平面**是集群的大脑，四个核心组件如下：

| 组件 | 职责 |
|---|---|
| **kube-apiserver** | 唯一与 etcd 通信的组件，其他组件通过它访问。基于 HTTPS 的 REST/gRPC。 |
| **etcd** | 集群的可信数据源，强一致的分布式 KV 存储。**丢了 etcd，就丢了集群。必须备份。** |
| **kube-scheduler** | 决定新 Pod 调度到哪个节点，依据资源请求、taint、亲和性等。 |
| **kube-controller-manager** | 运行各种控制器循环，如 Deployment、ReplicaSet、Node 等。 |

每个 **Worker 节点**运行以下组件：

| 组件 | 职责 |
|---|---|
| **kubelet** | 节点代理，接收 API Server 的 Pod 规约，调用容器运行时启动容器，并汇报状态。 |
| **kube-proxy** | 配置 iptables/IPVS 规则，确保 Service IP 路由到正确 Pod。 |
| **容器运行时** | 真正运行容器的组件，如 containerd 或 CRI-O，Docker 已不再直接使用。 |

### 协调循环：Kubernetes 的核心思想

Kubernetes 中所有控制器都运行同一个循环：

```
while True:
    desired = read_desired_state_from_api_server()
    actual  = observe_actual_state()
    if desired != actual:
        take_action_to_close_the_gap()
```

声明需要 3 个副本，ReplicaSet 控制器发现只有 2 个在运行，于是创建 1 个。某个节点挂了，1 个 Pod 不可用，控制器再次发现只有 2 个在运行，于是再创建 1 个。**你没告诉它做什么，只告诉它你想要什么。**

这就是 Kubernetes 自愈的原因——没有单独的“自愈”功能，整个系统就是**一个自愈循环**。也是它有时显得神秘的原因：现实与期望不一致时，总有什么地方在静默重试。

### Pod、Service、Deployment：日常主力

**Pod** 是最小可部署单元，包含一个或多个共享网络与存储的容器。**实际中“一个 Pod 一个容器”是默认值**，例外是 sidecar（日志代理、service mesh proxy）需要与主应用共享网络或卷时。

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

**资源 requests 和 limits 是承重的区别。** requests 是调度器选节点的依据，内核**保证**分配；limits 是硬上限，超出会被内核杀掉（内存）或限流（CPU）。requests 设低了会导致节点过度认领，引发 OOM；limits 设低了会导致限流，看起来像应用变慢。

**探针决定流量与生命周期。** Readiness 控制流量（“我准备好接请求了吗？”），Liveness 控制重启（“我还活着吗？”）。**很多生产事故来自配错的 liveness probe，它不停重启一个其实没问题、只是慢了点的 Pod。**

**Service** 在一组 Pod 前提供稳定的虚拟 IP 和 DNS 名。Pod 来来去去，Service IP 不变。

```yaml
apiVersion: v1
kind: Service
metadata: { name: web }
spec:
  selector: { app: web }
  ports: [{ port: 80, targetPort: 80 }]
  type: ClusterIP    # 仅集群内；对外用 LoadBalancer
```

**Deployment** 管理无状态工作负载的滚动更新与回滚。它拥有一个 ReplicaSet，ReplicaSet 拥有 Pod。改镜像时，Deployment 创建新 ReplicaSet，按策略迁移流量：

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

`maxUnavailable: 0` 表示“更新过程中绝不降低容量”，更安全。`maxSurge: 1` 表示“一次只多起 1 个 Pod”，限制 rollout 的激进程度。

```bash
kubectl set image deployment/web web=ghcr.io/myorg/web:v1.4.3
kubectl rollout status deployment/web        # 等待 rollout 完成
kubectl rollout history deployment/web       # 查看所有 revision
kubectl rollout undo deployment/web          # 回滚到上一版本
```

### 何时需要 StatefulSet、DaemonSet、Job

绝大多数工作负载都是 Deployment。例外值得了解：

- **StatefulSet** -- 有序、有名的 Pod（`db-0`、`db-1`、`db-2`），带稳定持久化卷。用于数据库、消息队列等需要“身份”的场景。更新更慢，一次一个 Pod，按顺序。
- **DaemonSet** -- 每个节点一个 Pod。用于日志采集器、监控代理、CSI 驱动、CNI 插件。
- **Job** -- 跑到完成。用于数据库迁移、批处理。
- **CronJob** -- 按计划跑的 Job。用于备份、定期报表。

### 托管 Kubernetes：理性默认选择

自建控制平面可行（`kubeadm`、`kops`、`kubespray`），但很少是正确的选择——你要自己负责 etcd 备份、证书轮转、版本升级、安全补丁。托管服务帮你挑走这些重量：

```bash
# AWS EKS
eksctl create cluster --name prod --region us-west-2 --nodes 3

# Google GKE
gcloud container clusters create prod --num-nodes 3 --region us-central1

# Azure AKS
az aks create --name prod --resource-group rg --node-count 3
```

控制平面定价约 73-150 美元/月，加上 worker 节点成本。**对绝大多数组织，这点费用远小于它节省的工程时间。**
## 网络：从 CNI 到 Service Mesh

Kubernetes 网络分三层，每层有自己的原语：

| 关注点 | 原语 | 实现 |
|---|---|---|
| **Pod 间连通** | 扁平 L3 网络 | CNI 插件（Calico、Cilium、Flannel） |
| **L4 访问策略** | NetworkPolicy | CNI 强制实施 |
| **L7 流量管理** | Ingress / Service Mesh | NGINX、Istio、Linkerd |

### 如何选择 CNI 插件

每个 Pod 都有自己的 IP。Container Network Interface (CNI) 插件是关键。

| 插件 | 路线 | 适用场景 |
|---|---|---|
| **Flannel** | VXLAN overlay，无策略 | 仅实验/开发 |
| **Calico** | BGP 路由 + iptables 策略 | 生产环境默认 |
| **Cilium** | eBPF，内核级策略 + L7 可见性 | 性能 + 安全，现代首选 |

Cilium 是新集群的优选：绕开 iptables（大规模瓶颈），支持 L7 策略（HTTP method/path 级），还提供流量可见性。

### NetworkPolicy：默认拒绝才是目标

默认情况下，任何 Pod 都能访问其他 Pod。这方便但不安全。NetworkPolicy 解决问题：

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

真正有效的做法：先在 namespace 下默认拒绝策略，再显式放行。被攻陷的 Pod 无法轻易扫描集群。

### Service Mesh：sidecar 是否值得

![Service Mesh （Istio）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig4_service_mesh_istio.png)

Service Mesh（Istio、Linkerd）在每个 Pod 注入代理（Envoy）作为 sidecar。所有服务间流量经过代理，无需改代码即可实现 mTLS、重试、超时、流量切分和黄金指标可观测性。

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

代价：每个 Pod 多一个 sidecar（约 50 MB 内存、5 ms 延迟）、需管理控制平面、学习曲线陡峭。经验法则：服务数 < 10，YAML 比问题复杂；服务数 > 50，工程师花时间处理重试/超时/mTLS，则 Mesh 是明确优势。

### Ingress：前门

外部 HTTP 流量进集群，用 Ingress（NGINX、Traefik 或云上 ALB/GLB 控制器）比每服务一个 LoadBalancer 更简单：

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

配上 `cert-manager`，证书自动续期。加 ExternalDNS，整套 DNS+TLS+路由都声明式搞定。
## 存储：持久卷的正确实践

Pod 短暂，数据长存。K8s 存储抽象解耦了需求（PVC）、供给方式（StorageClass）和底层实现（provisioner）。

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
  accessModes: ["ReadWriteOnce"]     # 单节点挂载
  storageClassName: fast-ssd
  resources: { requests: { storage: 100Gi } }
```

**关键访问模式：**

- `ReadWriteOnce`（RWO） -- 单节点读写。块存储（EBS、PD、Azure Disk）。默认最快。
- `ReadWriteMany`（RWX） -- 多节点读写。需网络文件系统（EFS、Filestore、Azure Files、CephFS）。较慢，一致性依赖文件系统。非必要不推荐。
- `ReadOnlyMany`（ROX） -- 分发静态内容。

**`reclaimPolicy: Retain` 是生产环境的安全选择。** 用 `Delete` 会静默删除卷，而 `Retain` 需手动操作——这能避免很多麻烦。

数据库跑在 K8s 上，优先选 Operator（Postgres Operator、MongoDB Operator），别自己写 StatefulSet。它们处理了主备切换、备份、滚动升级等复杂问题。
## Helm：正确打包的方式

服务一多，manifest 文件迅速膨胀。Helm 将模板化的 manifest 打包成版本化、可复用、参数化的 "chart"。

![Helm Chart 与 Release 历史](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/cloud-native-containers/fig5_helm_charts.png)

一个 chart 是模板目录加 `values.yaml` 默认值文件。安装时按环境覆盖 values：

```bash
helm install web ./charts/web -n prod -f values.prod.yaml
helm upgrade web ./charts/web -n prod -f values.prod.yaml --atomic
helm rollback web 3 -n prod                  # 回到 revision 3
helm history web -n prod                     # 查看完整 release 历史
```

**`--atomic` 是被低估的选项**：升级失败（post-install hook 或就绪检查未通过）时，Helm 自动回滚。这一个 flag 把 Helm 从“模板工具”变成了“事务化发布管理工具”。

### 何时不用 Helm

Helm 模板基于 Go template 和 Sprig 函数。简单 chart 没问题，复杂场景下 YAML 嵌在模板字符串里容易出问题（空白符 bug、schema 校验只能在安装时发现）。一些替代方案正在崛起：

- **Kustomize**（内置在 kubectl 中） -- 不用模板，靠 patch 和 overlay 实现。适合小改动。
- **CUE / KCL / Pkl** -- 带 schema 的强类型配置语言。前期投入高，但后期问题少。

发布第三方软件（数据库、监控栈），Helm 是通用选择。自己的应用可以评估上述替代品。
## 在生产中存活的微服务模式

成熟系统里常见的模式：

**API 网关** -- 客户端统一入口，负责鉴权、限流、路由和响应整形。用 Kong、Envoy Gateway 或云托管网关。简化客户端代码，集中管理安全策略。

**熔断（Circuit Breaker）** -- 下游故障时停止调用，快速失败，给下游恢复时间。Service Mesh（Istio outlierDetection）透明处理；Resilience4j、Hystrix 在进程内实现。

**Saga** -- 分布式事务，不要分布式锁。每步有补偿动作；第 4 步失败就反向执行 1-3 的补偿。两种方式：编排（coordinator 驱动步骤）和编舞（服务发事件驱动）。编排易推理，编舞易扩展。

**Outbox 模式** -- "DB 写入 + 事件发出"原子化。事件写到业务事务里的 outbox 表；另起进程读取并发布。解决事件驱动系统的双写问题。

**每服务自有数据库** -- 每个服务独占数据，不共享 DB。代价明显（join 变 API 调用），但共享数据库通过 schema 耦合服务，悄悄抹杀了微服务的意义。
## CI/CD 与 GitOps

Kubernetes 现代交付管线如下：

```
git push -> CI（测试、构建、扫描、签名、推镜像）
         -> CI 更新 manifest 仓（image tag bump）
         -> ArgoCD 检测差异并同步集群
```

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
