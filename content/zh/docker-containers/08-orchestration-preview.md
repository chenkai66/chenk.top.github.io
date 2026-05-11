---
title: "Docker 与容器（八）：超越 Docker —— Kubernetes、Swarm 及未来演进"
date: 2023-06-30 09:00:00
tags:
  - Docker
  - Containers
  - Kubernetes
  - Docker Swarm
  - Orchestration
categories:
  - Docker and Containers
series: docker-containers
lang: zh
description: "单机版 Docker 在规模化时必然失效。本文预览容器编排技术——Docker Swarm 以简洁见长，Kubernetes 则覆盖一切需求，并勾勒更广阔的云原生生态全景。"
disableNunjucks: true
series_order: 8
translationKey: "docker-containers-8"
---

本系列此前所有内容均围绕**单机版 Docker**展开：即仅在一台机器上运行容器。这种方式非常适合开发、小型项目以及流量适中的应用。但一旦你的服务需要在服务器宕机时持续可用、应对突发流量高峰，或实现零停机部署更新，单机版 Docker 的局限性便立刻显现。而**容器编排（Container Orchestration）**正是为解决这些问题而生——其中，Kubernetes 已成为事实上的行业标准。

## 为何单机版 Docker 不够用？

设想一下你的 Docker 主机意外宕机时会发生什么：

| 问题 | 单机版 Docker | 启用编排后 |
|---------|-------------------|-------------------|
| 服务器崩溃 | 所有容器终止，需手动重启 | 容器自动重新调度至健康节点 |
| 流量激增 | 手动执行 `--scale` 扩容 | 基于指标的自动扩缩容 |
| 部署更新 | `docker compose down && up`（必然停机） | 滚动更新（Rolling Update），零停机 |
| 服务发现 | 自定义网络 DNS（仅限单机） | 全集群 DNS + 负载均衡 |
| 密钥轮换 | 重启容器并注入新环境变量 | 滚动式密钥轮换，无需重启容器 |
| 资源分配 | 依赖人工估算内存是否充足 | 调度器智能放置容器，优化资源利用 |
| 监控 | `docker stats`（仅限单机） | 全集群指标采集与告警 |
| 存储 | 本地卷（主机宕机即丢失数据） | 支持复制的持久化卷（Persistent Volumes） |

这些是**运维层面的问题，而非 Docker 本身的问题**。Docker 完美地完成了它被设计的任务：在单台主机上运行容器。而编排器则在此之上，增加了**跨多主机协同调度**这一关键能力层。

## Docker Swarm：通往编排的极简路径

Docker Swarm 是 Docker 内置的编排方案。如果你已熟悉 `docker compose`，那么你实际上已经掌握了 Swarm 的 80%。它使用相同的 YAML 格式和高度相似的命令。

### 初始化 Swarm 集群

```bash
# 在首节点执行（该节点将成为 manager）
docker swarm init --advertise-addr 192.168.1.10
```

```
Swarm initialized: current node (abc123def456) is now a manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token SWMTKN-1-0123456789abcdef-worker-token 192.168.1.10:2377

To add a manager to this swarm, run:

    docker swarm join-token manager
```

```bash
# 在其他节点上执行（作为 worker 加入）
docker swarm join --token SWMTKN-1-0123456789abcdef-worker-token 192.168.1.10:2377
```

```
This node joined a swarm as a worker.
```

```bash
# 查看集群状态
docker node ls
```

```
ID                           HOSTNAME   STATUS   AVAILABILITY   MANAGER STATUS   ENGINE VERSION
abc123def456 *               manager1   Ready    Active         Leader           24.0.6
def456abc789                 worker1    Ready    Active                          24.0.6
ghi789def012                 worker2    Ready    Active                          24.0.6
```

仅需四条命令，一个三节点集群即告建成。这正是 Swarm 的核心吸引力所在。

### 部署服务（Services）

Swarm 引入了“服务（Service）”的概念——服务是对容器运行方式的声明式定义，Swarm 负责维持指定数量的副本（replicas）：

```bash
# 创建一个含 3 个副本的服务
docker service create \
    --name web \
    --replicas 3 \
    --publish 80:80 \
    --update-delay 10s \
    --update-parallelism 1 \
    nginx:alpine

# 查看服务列表
docker service ls
```

```
ID             NAME   MODE         REPLICAS   IMAGE          PORTS
a1b2c3d4e5f6   web    replicated   3/3        nginx:alpine   *:80->80/tcp
```

```bash
# 查看各副本实际部署位置
docker service ps web
```

```
ID             NAME    IMAGE          NODE       DESIRED STATE   CURRENT STATE           
b2c3d4e5f6a7   web.1   nginx:alpine   manager1   Running         Running 30 seconds ago
c3d4e5f6a7b8   web.2   nginx:alpine   worker1    Running         Running 30 seconds ago
d4e5f6a7b8c9   web.3   nginx:alpine   worker2    Running         Running 30 seconds ago
```

Swarm 将三个副本均匀分布于全部三个节点。它还提供内置负载均衡：集群中任意节点均可通过端口 80 接收流量，Swarm 会自动将请求路由至运行该服务的容器。

### 滚动更新（Rolling Updates）

```bash
# 更新镜像（按 10 秒间隔逐个滚动更新）
docker service update --image nginx:1.25-alpine web
```

```
web
overall progress: 3 out of 3 tasks
1/3: running   [==================================================>]
2/3: running   [==================================================>]
3/3: running   [==================================================>]
verify: Service converged
```

Swarm 每次仅更新一个容器，并在每个更新之间等待 10 秒。若新容器健康检查失败，Swarm 将自动回滚。

### 部署 Stack（在 Swarm 中运行 Compose 文件）

你可以直接将 `docker-compose.yml` 文件部署到 Swarm：

```yaml
# docker-compose.yml
services:
  web:
    image: myapp:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    ports:
      - "8080:8080"
    networks:
      - app-net

  redis:
    image: redis:7-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
    volumes:
      - redis-data:/data
    networks:
      - app-net

networks:
  app-net:
    driver: overlay

volumes:
  redis-data:
```

```bash
# 部署整个 stack
docker stack deploy -c docker-compose.yml myapp

# 查看 stack 下的服务
docker stack services myapp
```

```
ID             NAME           MODE         REPLICAS   IMAGE             PORTS
a1b2c3d4e5f6   myapp_web      replicated   3/3        myapp:latest      *:8080->8080/tcp
b2c3d4e5f6a7   myapp_redis    replicated   1/1        redis:7-alpine
```

```bash
# 删除整个 stack
docker stack rm myapp
```

### Swarm 的 Secrets 与 Configs

Swarm 原生支持密钥（Secrets）和配置文件（Configs）：

```bash
# 创建密钥
echo "supersecret" | docker secret create db_password -

# 创建配置
docker config create nginx_conf ./nginx.conf

# 在服务中使用它们
docker service create \
    --name api \
    --secret db_password \
    --config source=nginx_conf,target=/etc/nginx/nginx.conf \
    myapp
```

在容器内部，密钥以文件形式挂载在 `/run/secrets/` 目录下：

```bash
cat /run/secrets/db_password
# 输出: supersecret
```

### Swarm 的适用场景

当满足以下任一条件时，Swarm 是一个合理的选择：
- 团队规模小（< 5 名工程师）
- 集群规模小（< 10 个节点）
- 希望获得编排能力，但不愿承担 Kubernetes 的陡峭学习曲线
- 已在使用 Docker Compose，希望平滑迁移
- 无需自动扩缩容、自定义调度器或 CNCF 生态体系

## Kubernetes：行业标准编排平台

Kubernetes（K8s）是当前占据主导地位的容器编排平台。它比 Swarm 更复杂，但能力也强大得多。主流云厂商均提供托管 Kubernetes 服务（如 EKS、GKE、AKS、ACK），从而免除了用户自行管理控制平面（Control Plane）的运维负担。

### 架构概览

Kubernetes 集群包含两类节点：

**控制平面（Control Plane，即 master）组件：**

| 组件 | 角色 |
|-----------|------|
| `kube-apiserver` | 所有组件与用户交互的 REST API 入口 |
| `etcd` | 分布式键值存储，保存整个集群的状态 |
| `kube-scheduler` | 决定新 Pod 应调度到哪个工作节点 |
| `kube-controller-manager` | 运行各类控制器（Deployment、ReplicaSet、Node 等） |
| `cloud-controller-manager` | 与云厂商 API 集成（可选） |

**工作节点（Worker Node）组件：**

| 组件 | 角色 |
|-----------|------|
| `kubelet` | 节点代理，负责管理本节点上的 Pod，并与 API Server 通信 |
| `kube-proxy` | 网络代理，实现 Service 的流量路由 |
| 容器运行时（Container Runtime） | 运行容器（如 containerd、CRI-O —— **不再依赖 Docker daemon**） |

其架构逻辑如下（文字描述，非图示）：

```
控制平面：
  API Server ←→ etcd（集群状态存储）
      ↑
  Scheduler + Controller Manager（监听 API，做出调度与控制决策）

工作节点（N 个）：
  kubelet ←→ API Server（上报状态，接收指令）
  kube-proxy（管理 iptables/IPVS 规则，实现 Service 路由）
  containerd（运行容器）
```

### Kubernetes 核心对象

Kubernetes 中的一切皆为**声明式对象（Declarative Object）**：你描述期望的状态（desired state），Kubernetes 则持续努力使其变为现实。

#### Pod

最小的可部署单元。一个 Pod 包含一个或多个共享网络与存储的容器：

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  containers:
    - name: app
      image: myapp:v1.0
      ports:
        - containerPort: 8080
      resources:
        requests:
          memory: "128Mi"
          cpu: "250m"
        limits:
          memory: "256Mi"
          cpu: "500m"
      livenessProbe:
        httpGet:
          path: /health
          port: 8080
        initialDelaySeconds: 10
        periodSeconds: 30
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 10
```

你极少会直接创建 Pod；通常使用更高层级的对象（如 Deployment）来间接管理。

#### Deployment

Deployment 管理一组完全相同的 Pod，并负责处理滚动更新：

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: app
          image: myapp:v1.0
          ports:
            - containerPort: 8080
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
          resources:
            requests:
              memory: "128Mi"
              cpu: "250m"
            limits:
              memory: "256Mi"
              cpu: "500m"
```

```bash
# 应用该 Deployment
kubectl apply -f deployment.yaml

# 查看 Deployment 状态
kubectl get deployments
```

```
NAME     READY   UP-TO-DATE   AVAILABLE   AGE
my-app   3/3     3            3           60s
```

```bash
# 扩容 Deployment
kubectl scale deployment my-app --replicas=5

# 更新镜像（触发滚动更新）
kubectl set image deployment/my-app app=myapp:v2.0

# 观察发布过程
kubectl rollout status deployment/my-app
```

```
Waiting for deployment "my-app" rollout to finish: 2 out of 3 new replicas have been updated...
Waiting for deployment "my-app" rollout to finish: 1 old replicas are pending termination...
deployment "my-app" successfully rolled out
```

```bash
# 若出错，一键回滚
kubectl rollout undo deployment/my-app
```

#### Service

Service 为一组 Pod 提供稳定的访问端点（DNS 名称与 IP 地址）：

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
```

Service 类型对比：

| 类型 | 可访问性 | 典型用例 |
|------|-------------|----------|
| `ClusterIP` | 仅集群内部可访问 | 微服务间内部调用 |
| `NodePort` | 通过 `<NodeIP>:<NodePort>` 外部访问 | 开发测试、简单暴露 |
| `LoadBalancer` | 通过云厂商负载均衡器外部访问 | 生产环境 Web 服务 |
| `ExternalName` | DNS CNAME 指向外部服务 | 访问外部数据库等 |

#### ConfigMap 与 Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"
  config.yaml: |
    server:
      port: 8080
      timeout: 30s

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  # base64 编码后的值
  url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0Bob3N0OjU0MzIvZGI=
  password: c3VwZXJzZWNyZXQ=
```

在 Pod 中引用它们：

```yaml
spec:
  containers:
    - name: app
      env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: LOG_LEVEL
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
      volumeMounts:
        - name: config
          mountPath: /app/config
  volumes:
    - name: config
      configMap:
        name: app-config
```

### 必备 kubectl 命令速查

```bash
# 获取资源列表
kubectl get pods                    # 列出所有 Pod
kubectl get pods -o wide            # 显示更多细节（所在节点、IP）
kubectl get deployments             # 列出所有 Deployment
kubectl get services                # 列出所有 Service
kubectl get all                     # 列出所有资源类型

# 详细信息与事件（describe）
kubectl describe pod my-app-abc123
kubectl describe deployment my-app

# 日志（logs）
kubectl logs my-app-abc123              # 查看 Pod 日志
kubectl logs my-app-abc123 -f           # 实时跟踪日志（follow）
kubectl logs my-app-abc123 -c sidecar   # 查看多容器 Pod 中特定容器的日志
kubectl logs -l app=my-app              # 查看所有带指定 label 的 Pod 日志

# 执行命令（exec）
kubectl exec -it my-app-abc123 -- bash
kubectl exec my-app-abc123 -- cat /app/config.yaml

# 端口转发（port-forward，无需 Service 即可本地访问 Pod）
kubectl port-forward my-app-abc123 8080:8080

# 应用/删除配置
kubectl apply -f deployment.yaml
kubectl delete -f deployment.yaml

# 调试（debug）
kubectl get events --sort-by=.metadata.creationTimestamp
kubectl top pods                        # 查看 Pod 资源使用率（需 metrics-server）
kubectl top nodes
```

## Kubernetes vs Docker Swarm 对比

| 特性 | Docker Swarm | Kubernetes |
|---------|-------------|------------|
| 部署复杂度 | 分钟级 | 小时级（托管服务大幅简化） |
| 学习曲线 | 低（Docker CLI 知识可复用） | 陡峭（新概念多，YAML 配置繁重） |
| 扩缩容 | 手动（`docker service scale`） | 手动 + Horizontal Pod Autoscaler（HPA） |
| 滚动更新 | 内置，简单易用 | 内置，高度可配置 |
| 服务发现 | Docker 内置 DNS | CoreDNS + Service |
| 负载均衡 | 内置（Routing Mesh） | Service + Ingress Controllers |
| 密钥管理 | Docker Secrets | Kubernetes Secrets（+ 外部集成） |
| 存储 | Docker Volumes | PersistentVolumes + StorageClasses + CSI Drivers |
| 网络 | Overlay Networks | CNI 插件（Calico、Cilium、Flannel 等） |
| 健康检查 | `HEALTHCHECK` 指令 | Liveness / Readiness / Startup Probes |
| 包管理 | 无 | Helm Charts |
| 社区与生态 | 小，呈萎缩趋势 | 庞大，CNCF 生态繁荣 |
| 可托管服务 | 极少 | EKS、GKE、AKS、ACK 等众多选择 |
| 最佳适用场景 | 小团队、简单部署 | 大规模生产、微服务架构 |

### 何时你 *不需要* 编排？

并非所有应用都需要 Kubernetes。请诚实地评估自身需求：

| 你的现状 | 推荐方案 |
|---------------|---------------|
| 单台服务器，服务极少 | Docker Compose |
| 小团队，< 5 个服务 | Docker Compose 或 Swarm |
| 单机上需零停机部署 | Docker Compose（配合滚动重启策略） |
| 无服务器（Serverless）工作负载 | 云函数（Lambda、Cloud Run） |
| 批处理任务 | Docker Compose 或单机调度器 |
| 多地域、高可用要求 | Kubernetes（托管版） |
| 微服务架构 | Kubernetes |
| 合规性强制要求编排 | Kubernetes |
| 工程师团队 > 10 人 | Kubernetes |

一个常见误区是：为一个仅运行在每月 $20 VPS 上的三服务应用仓促引入 Kubernetes。即使采用托管服务，Kubernetes 带来的运维开销也远超其收益——除非你已达到一定规模。

若你仍运行在单台主机上，但希望获得更优的部署体验，可考虑以下工具：

- **Docker Compose** + 简单 CI/CD 流水线  
- **Kamal**（Basecamp 出品）——面向裸金属服务器的零停机部署工具  
- **Dokku**——自托管 PaaS（类似私有 Heroku）  
- **Coolify**——开源、可自托管的 Heroku/Netlify/Vercel 替代方案  

## 云原生（Cloud-Native）生态系统

Kubernetes 催生了一个庞大的工具生态。以下是其中最重要的一些：

### 包管理：Helm

Helm 是 Kubernetes 的包管理器。“Chart” 将一个应用所需的所有 YAML 文件打包封装：

```bash
# 安装一个 Helm Chart（例如 PostgreSQL）
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install my-postgres bitnami/postgresql --set auth.postgresPassword=secret

# 列出已安装的 Release
helm list

# 升级 Release
helm upgrade my-postgres bitnami/postgresql --set auth.postgresPassword=newsecret

# 卸载
helm uninstall my-postgres
```

Helm Charts 就像 Docker Images 的升级版——它是整个应用栈的预打包、版本化、可共享单元。

### 服务网格（Service Mesh）：Istio 与 Linkerd

服务网格为微服务间通信增加可观测性、安全性和流量管理能力：

| 特性 | 无服务网格 | 启用服务网格 |
|---------|---------------------|-------------------|
| 服务间 mTLS | 手动证书管理 | 自动、透明启用 |
| 流量拆分 | 应用层实现 | 声明式（如 80/20 灰度发布） |
| 重试策略 | 每个服务自行编码 | 按路由粒度可配置 |
| 可观测性 | 各服务需自行埋点 | 自动分布式追踪与指标采集 |
| 访问控制 | 应用层鉴权 | 基于策略（YAML）的统一管控 |

Istio 功能丰富但复杂；Linkerd 更轻量、更简单。二者均非必需——除非你拥有 10+ 个相互调用的微服务，且对流量控制有精细化要求。

### GitOps：ArgoCD 与 Flux

GitOps 将 Git 仓库视为集群状态的唯一可信源（Source of Truth）：

1. 你向 Git 提交变更（例如更新 Deployment YAML 中的镜像 tag）  
2. ArgoCD 检测到变更，并自动同步集群状态以匹配 Git  
3. 集群最终收敛至你所声明的期望状态  

```bash
# ArgoCD 监听 Git 仓库并自动应用变更
argocd app create my-app \
    --repo https://github.com/myorg/myapp-k8s.git \
    --path kubernetes/ \
    --dest-server https://kubernetes.default.svc \
    --dest-namespace production
```

优势：
- 每次变更均可审计（Git 历史记录）  
- 回滚 = `git revert`  
- 生产环境无需手动执行 `kubectl apply`  
- 集群状态始终可通过 Git 100% 重建  

### 监控与可观测性（Monitoring & Observability）

| 工具 | 用途 | 采集内容 |
|------|---------|----------|
| Prometheus | 指标采集与告警 | CPU、内存、请求速率、自定义业务指标 |
| Grafana | 可视化与仪表盘 | 展示 Prometheus 数据（及其他数据源） |
| Jaeger / Zipkin | 分布式追踪 | 请求在微服务间的完整调用链路 |
| Fluentd / Fluent Bit | 日志聚合 | 容器日志 → 中央存储系统 |
| Elasticsearch + Kibana | 日志存储与搜索 | 全文可检索的日志索引 |

Kubernetes 社区公认的开源可观测性“黄金栈”是：**Prometheus + Grafana + Fluentd（或 Fluent Bit）+ Jaeger**，当然也存在诸多替代方案。

### Kubernetes 中的容器安全

| 工具 | 用途 |
|------|---------|
| Trivy | 镜像漏洞扫描 |
| Falco | 运行时安全监控（检测异常容器行为） |
| OPA/Gatekeeper | 策略强制执行（例如：“禁止容器以 root 用户运行”） |
| cert-manager | 自动 TLS 证书管理（对接 Let's Encrypt） |
| Kyverno | Kubernetes 原生策略引擎 |

## 从 Docker 到生产环境：典型演进路径

一个团队从个人项目成长为生产级服务，其基础设施与部署方式的演进路径通常如下：

| 阶段 | 基础设施 | 部署方式 |
|-------|---------------|------------|
| 1. 本地开发 | 笔记本上的 Docker Compose | `docker compose up` |
| 2. 单服务器部署 | VPS 上的 Docker Compose | `git pull && docker compose up -d` |
| 3. CI/CD 流水线 | Docker Compose + GitHub Actions | 向 main 分支推送即自动部署 |
| 4. 多服务器部署 | Docker Swarm 或托管 Kubernetes | `docker stack deploy` 或 `kubectl apply` |
| 5. 大规模生产 | 托管 Kubernetes（EKS/GKE/AKS/ACK） | Helm + ArgoCD |
| 6. 多地域部署 | 托管 Kubernetes + 服务网格 | GitOps + 流量管理 |

大多数团队永远无需走到第 5 或第 6 阶段。不要因为“听起来很酷”就跃升至第 5 阶段——只有当当前阶段遇到的实际问题，其复杂度已明确超出当前工具的能力边界时，才应迈出这一步。

## 本系列核心要点回顾

回望全部八篇文章，以下原则最为根本：

**容器本质是进程，而非虚拟机。** 它们共享宿主机内核，依靠命名空间（namespaces）与控制组（cgroups）实现隔离。理解这一点，将从根本上塑造你对安全性、性能及调试的认知。

**镜像是分层结构。** 分层缓存驱动构建性能；Dockerfile 指令顺序至关重要；多阶段构建（multi-stage build）可清晰分离构建期依赖与运行时依赖。

**网络与卷是连接的纽带。** 自定义桥接网络提供基于 DNS 的服务发现；命名卷（named volumes）使数据持久化独立于容器生命周期。

**Compose 是开发者的接口。** 一份 YAML 文件即可替代数十条 `docker run` 命令。它可版本控制、可共享、可确定性复现。

**安全是“显式开启”的选项。** Docker 默认优先考虑便利性。以非 root 用户运行、裁剪 Linux Capabilities、使用只读文件系统、扫描镜像漏洞——这些都必须由你主动、明确地配置。

**编排是一个光谱（spectrum）。** Docker Compose 适用于单机，Swarm 适用于简单多机，Kubernetes 适用于大规模生产。**选择能解决你当下真实问题的最简单工具。**

容器生态虽日新月异，但本系列所阐述的基础原理却极为稳固：Linux 命名空间自 2013 年起未变；OCI 镜像格式早已标准化；Kubernetes API 对象多年保持稳定。掌握这些根基，你便能从容驾驭其上构建的所有新兴工具。