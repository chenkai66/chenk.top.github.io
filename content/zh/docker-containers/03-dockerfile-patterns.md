---
title: "Docker 与容器（三）：Dockerfile 最佳实践 —— 从初学者到生产环境"
date: 2023-06-20 09:00:00
tags:
  - Docker
  - Containers
  - Dockerfile
  - Multi-stage Build
  - DevOps
categories:
  - Docker and Containers
series: docker-containers
lang: zh
description: "Dockerfile 定义了镜像的构建方式。一个朴素的 Dockerfile 和一个优化后的 Dockerfile 之间，镜像大小和构建时间可能相差 10 倍。"
disableNunjucks: true
series_order: 3
translationKey: "docker-containers-3"
---

大多数教程只展示一个 5 行的 Dockerfile 就匆匆略过。结果你部署到生产环境后才发现：镜像体积高达 1.2 GB，哪怕只改一行代码也要花 8 分钟构建，而安全团队更是在报告中反复指出——那些你甚至不知道自己安装过的软件包存在严重漏洞。编写一份优秀的 Dockerfile 是一项关键技能，它会在每次 CI 流水线运行时为你持续带来回报。

## 每一条 Dockerfile 指令详解

我们逐条梳理你将用到的所有指令，并辅以具体示例。

![Dockerfile best practices](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/03-best-practices.png)


### FROM — 起点镜像

每个 Dockerfile 都必须以 `FROM` 开头。它指定了所有后续指令所基于的基础镜像。

```dockerfile
# 使用特定版本（推荐）
FROM python:3.11-slim

# 使用极简基础镜像
FROM alpine:3.18

# 使用 scratch（空镜像），适用于静态编译的二进制文件
FROM scratch

# 多个 FROM 语句 = 多阶段构建（后文详述）
FROM golang:1.21 AS builder
# ...
FROM alpine:3.18
# ...
```

**最佳实践**：务必锁定具体版本。`FROM python:latest` 在新 Python 版本发布且你的代码尚未兼容时，将直接导致构建失败。

| 基础镜像选择 | 大小 | 适用场景 |
|-------------------|------|----------|
| `python:3.11` | ~900 MB | 需要构建工具（如 gcc 等） |
| `python:3.11-slim` | ~150 MB | 标准生产环境镜像 |
| `python:3.11-alpine` | ~50 MB | 对镜像尺寸极度敏感；但需注意 musl 兼容性问题 |
| `ubuntu:22.04` | ~78 MB | 需要 apt 及 Ubuntu 特有软件包 |
| `alpine:3.18` | ~7 MB | 极致精简、非语言专用基础镜像 |
| `scratch` | 0 MB | Go / Rust 等静态编译的二进制程序 |

### RUN — 执行构建时命令

`RUN` 在构建过程中于镜像内执行命令。每条 `RUN` 指令都会生成一个新的镜像层。

```dockerfile
# Shell 形式（在 `/bin/sh -c` 中运行）
RUN apt-get update && apt-get install -y curl

# Exec 形式（无 shell 解析）
RUN ["apt-get", "update"]

# 多行命令用 `&&` 连接，确保单层提交
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*
```

**关键模式**：用 `&&` 链式执行命令，并在同一 `RUN` 中完成清理。若在一条 `RUN` 中 `apt-get install`，又在另一条 `RUN` 中 `rm -rf /var/lib/apt/lists/*`，则包列表仍会保留在第一层中——镜像层是累加的，**不可撤销**。

```dockerfile
# BAD：3 层，清理无效（列表仍存在于 Layer 1）
RUN apt-get update
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*

# GOOD：1 层，列表在该层提交前即被清除
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
```

### COPY 和 ADD — 将文件复制进镜像

```dockerfile
# COPY：从构建上下文复制文件到镜像中
COPY requirements.txt /app/requirements.txt
COPY . /app/

# COPY 支持 --chown 设置属主属组
COPY --chown=appuser:appuser . /app/

# ADD：功能类似 COPY，但额外支持：
#   - 自动解压 tar 归档
#   - 支持 URL（但请勿使用 —— 应改用 RUN + curl）
ADD archive.tar.gz /app/
```

**最佳实践**：除非明确需要自动解压 tar 包，否则一律使用 `COPY`。它更显式、更可预测。

### WORKDIR — 设置工作目录

```dockerfile
# 为后续指令设置工作目录
WORKDIR /app

# 等价于 mkdir -p && cd（不存在时自动创建）
WORKDIR /app/src

# 这三行：
RUN mkdir -p /app && cd /app && npm install
# 更优雅地写作：
WORKDIR /app
RUN npm install
```

### ENV 和 ARG — 变量定义

```dockerfile
# ENV：设置环境变量（在运行时容器中持久存在）
ENV NODE_ENV=production
ENV APP_PORT=8080

# ARG：构建时变量（**不在运行时容器中可用**）
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# ARG 值可在构建时覆盖
# docker build --build-arg PYTHON_VERSION=3.12 .
```

核心区别：

| 特性 | ENV | ARG |
|---------|-----|-----|
| 构建期间可用 | 是 | 是 |
| 运行时可用 | 是 | 否 |
| 在 `docker inspect` 中可见 | 是 | 否（但缓存在镜像层中） |
| 构建时可覆盖 | 否（应使用 ARG 实现） | 是（通过 `--build-arg`） |
| 跨构建阶段持久化 | 是（限于同一阶段内） | 否（每个 `FROM` 后重置） |

**安全警告**：`ENV` 和 `ARG` **均不应存放密钥**。构建参数会被记录在镜像历史中（`docker history`）。请改用 Docker Secrets 或通过 `--secret` 在构建时挂载密钥。

### EXPOSE — 声明端口

```dockerfile
# 声明容器监听的端口（仅文档用途）
EXPOSE 80
EXPOSE 443
EXPOSE 8080/tcp
EXPOSE 8443/udp
```

`EXPOSE` **不会真正发布端口**。它只是元数据注释，供工具读取。运行容器时仍需显式指定 `-p 8080:80`。你可以把它理解为一条“机器可读的注释”。

### CMD 和 ENTRYPOINT — 容器启动行为

这两条指令共同定义容器启动时执行的内容。理解其差异至关重要。

```dockerfile
# CMD：默认命令（可被 `docker run` 完全覆盖）
CMD ["python3", "app.py"]

# ENTRYPOINT：固定可执行程序（`docker run` 参数将追加至其后）
ENTRYPOINT ["python3", "app.py"]

# 组合用法：ENTRYPOINT 是可执行程序，CMD 提供默认参数
ENTRYPOINT ["python3"]
CMD ["app.py"]
```

`docker run` 行为对比：

| Dockerfile | `docker run image` | `docker run image bash` |
|-----------|-------------------|------------------------|
| `CMD ["python3", "app.py"]` | 运行 `python3 app.py` | 运行 `bash`（CMD 被替换） |
| `ENTRYPOINT ["python3", "app.py"]` | 运行 `python3 app.py` | 运行 `python3 app.py bash`（参数追加） |
| `ENTRYPOINT ["python3"]` + `CMD ["app.py"]` | 运行 `python3 app.py` | 运行 `python3 bash`（CMD 被替换） |

**Shell 形式 vs Exec 形式：**

```dockerfile
# Exec 形式（推荐）—— 直接运行，能正确接收信号
CMD ["python3", "app.py"]

# Shell 形式 —— 实际运行为 `/bin/sh -c "python3 app.py"`
# Shell 包裹了你的进程，因此 SIGTERM 发送给 sh，而非你的应用
CMD python3 app.py
```

对 `CMD` 和 `ENTRYPOINT`，**始终使用 Exec 形式**（`["executable", "arg1", "arg2"]`）。Shell 形式会将你的进程包裹在 `/bin/sh -c` 中，导致信号处理异常，影响优雅关闭。

### USER — 以非 root 用户运行

```dockerfile
# 创建非 root 用户并切换
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Alpine 系统写法
RUN addgroup -S appuser && adduser -S appuser -G appuser
USER appuser
```

为何这关乎安全？我们将在第 7 篇文章中深入探讨。

### HEALTHCHECK — 容器健康检查

```dockerfile
# 检查 Web 服务是否响应
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 禁用从基础镜像继承的健康检查
HEALTHCHECK NONE
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `--interval` | 30s | 健康检查间隔时间 |
| `--timeout` | 30s | 单次检查最大超时时间 |
| `--start-period` | 0s | 首次检查前的宽限期 |
| `--retries` | 3 | 连续失败多少次后标记为 “unhealthy” |

Docker 根据健康检查结果将容器标记为 `healthy`、`unhealthy` 或 `starting`。编排系统（Docker Compose、Kubernetes）据此决定是否向该容器路由流量。

## 初学者版 vs 优化版：真实案例对比

我们来构建一个 Flask 应用。应用代码如下：

![Build context](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/03-build-context.png)


```python
# app.py
from flask import Flask, jsonify
import redis

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

@app.route('/')
def hello():
    count = cache.incr('hits')
    return jsonify(message='Hello from Docker!', visits=count)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

```
# requirements.txt
flask==3.0.0
redis==5.0.1
gunicorn==21.2.0
```

### 初学者版 Dockerfile

```dockerfile
FROM python:3.11

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
```

构建并查看镜像大小：

```bash
docker build -t flask-naive -f Dockerfile.naive .
docker images flask-naive
```

```
REPOSITORY    TAG       IMAGE ID       CREATED          SIZE
flask-naive   latest    a1b2c3d4e5f6   10 seconds ago   1.02GB
```

一个仅含 3 个依赖的 Flask 应用，镜像竟达 **1.02 GB**。问题在于：

1. 使用 `python:3.11`（完整 Debian + 构建工具）—— 占用 900+ MB  
2. `COPY .` 复制了全部内容（包括 `.git`、`__pycache__`、`.env` 等）  
3. 缺少 `.dockerignore`  
4. 以 root 用户运行  
5. 使用开发服务器（`python app.py`）  
6. 无健康检查  
7. `pip install` 在任意文件变更时都会重新执行（破坏缓存）

### 优化版 Dockerfile

```dockerfile
FROM python:3.11-slim AS base

# 阻止 Python 写入 .pyc 文件，并启用无缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 优先安装依赖（仅当 requirements.txt 变更时才重建）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# 复制应用代码
COPY --chown=appuser:appuser . .

# 切换至非 root 用户
USER appuser

EXPOSE 5000

# 使用生产级 WSGI 服务器
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

配套的 `.dockerignore` 文件：

```
.git
.gitignore
__pycache__
*.pyc
*.pyo
.env
.venv
venv
*.md
.dockerignore
Dockerfile*
docker-compose*
.pytest_cache
.coverage
htmlcov
```

构建并对比：

```bash
docker build -t flask-optimized -f Dockerfile.optimized .
docker images | grep flask
```

```
REPOSITORY        TAG       IMAGE ID       CREATED          SIZE
flask-optimized   latest    f6e5d4c3b2a1   5 seconds ago    167MB
flask-naive       latest    a1b2c3d4e5f6   2 minutes ago    1.02GB
```

| 指标 | 初学者版 | 优化版 | 提升 |
|--------|-------|-----------|-------------|
| 镜像大小 | 1.02 GB | 167 MB | 缩小 6 倍 |
| 是否以 root 运行 | 是 | 否 | 安全加固 |
| 代码变更后是否全量重装 pip | 是 | 仅重构建 COPY 层 | 节省数分钟 |
| Web 服务器 | Flask 开发服务器 | Gunicorn | 生产就绪 |
| 健康检查 | 无 | 有 | 编排友好 |
| 泄露文件 | `.git`, `.env` 等 | 干净 | 避免密钥泄露 |

## `.dockerignore` 文件


![Dockerfile optimization journey from bloated to slim contain](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/03-dockerfile-optimization-journey-from-bloated-to-slim-contain.jpg)

`.dockerignore` 的作用类似于 `.gitignore`，但针对 Docker 构建上下文。当你执行 `docker build .` 时，Docker 会将整个目录（即“构建上下文”）发送给守护进程。若无 `.dockerignore`，所有文件都将被上传。

![Layer optimization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/03-layer-optimization.png)


```
# .dockerignore

# 版本控制
.git
.gitignore

# Python 构建产物
__pycache__
*.pyc
*.pyo
*.egg-info
dist
build
.eggs

# 虚拟环境
.venv
venv
env

# IDE 和编辑器文件
.vscode
.idea
*.swp
*.swo
*~

# 环境与密钥
.env
.env.*
*.pem
*.key

# Docker 文件（防止递归上下文）
Dockerfile*
docker-compose*
.dockerignore

# 测试与 CI
.pytest_cache
.coverage
htmlcov
.tox
.mypy_cache

# 文档
*.md
docs/
LICENSE
```

若缺失 `.dockerignore`，一个含 500 MB `.git` 目录和 200 MB `node_modules` 的项目，每次构建都会向守护进程发送 700 MB 数据——即使这些文件根本不会进入最终镜像。

## 层缓存（Layer Caching）：为何指令顺序至关重要

Docker 会对每一层进行缓存。若某条指令未变更（且其之前所有层均已缓存），Docker 将复用缓存层。但一旦某层缓存失效，其后所有层都必须重建。

![Base image size comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/03-image-size-comparison.png)


这正是指令顺序如此关键的原因：

```dockerfile
# BAD 顺序：任何代码变更都会使 pip install 缓存失效
COPY . /app
RUN pip install -r requirements.txt  # 每次任意文件变更都重建

# GOOD 顺序：仅当 requirements.txt 变更时才重建 pip install
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt  # 仅依赖变更时重建
COPY . /app                          # 仅代码变更时重建此层
```

通用原则：按**变更频率由低到高**排列指令。

```dockerfile
FROM python:3.11-slim          # 变更极少（基础镜像更新）
RUN apt-get update && ...      # 变更极少（系统包）
COPY requirements.txt .        # 变更偶尔（依赖）
RUN pip install -r ...         # 变更偶尔
COPY . .                       # 变更频繁（你的代码）
CMD [...]                      # 变更极少
```

你可以在构建日志中验证缓存行为：

```bash
# 首次构建（无缓存）
docker build -t myapp .
```

```
[+] Building 45.2s (10/10) FINISHED
 => [1/5] FROM python:3.11-slim                                      0.0s
 => [2/5] WORKDIR /app                                               0.0s
 => [3/5] COPY requirements.txt .                                    0.1s
 => [4/5] RUN pip install --no-cache-dir -r requirements.txt        42.3s
 => [5/5] COPY . .                                                   0.2s
```

```bash
# 第二次构建（仅修改 app.py，pip install 命中缓存）
docker build -t myapp .
```

```
[+] Building 1.8s (10/10) FINISHED
 => [1/5] FROM python:3.11-slim                                      0.0s
 => CACHED [2/5] WORKDIR /app                                        0.0s
 => CACHED [3/5] COPY requirements.txt .                              0.0s
 => CACHED [4/5] RUN pip install --no-cache-dir -r requirements.txt   0.0s
 => [5/5] COPY . .                                                   0.2s
```

因 `requirements.txt` 未变，节省了 42 秒。

## 多阶段构建（Multi-Stage Builds）

多阶段构建是 Docker 最强大的特性之一。它允许你在大型构建镜像（含编译器、构建工具等）中完成构建，再仅将最终产物复制到轻量级运行时镜像中。

![Multi-stage build](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/03-multi-stage-build.png)


### Python 示例

```dockerfile
# 阶段 1：在完整镜像中构建依赖
FROM python:3.11 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 阶段 2：仅含运行时所需最小镜像
FROM python:3.11-slim

WORKDIR /app

# 从 builder 阶段复制已安装的包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

USER nobody
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Go 示例（极致尺寸压缩）

Go 编译为静态二进制文件，从而实现最小镜像：

```dockerfile
# 阶段 1：构建
FROM golang:1.21 AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

# 阶段 2：运行时（scratch = 空镜像）
FROM scratch

COPY --from=builder /app/server /server
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

EXPOSE 8080
ENTRYPOINT ["/server"]
```

一个 Go HTTP 服务的镜像大小对比：

```bash
docker images | grep go-server
```

```
REPOSITORY        TAG          IMAGE ID       CREATED          SIZE
go-server         single       a1b2c3d4e5f6   10 seconds ago   845MB
go-server         multistage   b2c3d4e5f6a7   5 seconds ago    12.4MB
```

**845 MB vs 12.4 MB**。多阶段构建产出的镜像缩小了 68 倍——因为它丢弃了整个 Go 工具链，最终镜像中仅保留编译好的二进制文件和 CA 证书。该模式同样适用于 Node.js（用 `node:20-alpine` 构建，仅复制 `node_modules` 和应用代码至运行时阶段）及任何其他语言。

## 构建参数（ARG）vs 环境变量（ENV）

这是常见混淆点。以下是一个实用示例：

```dockerfile
# 构建时配置（通过 ARG）
ARG NODE_ENV=production
ARG APP_VERSION=unknown

# 构建时使用（决定安装哪些依赖）
RUN if [ "$NODE_ENV" = "development" ]; then \
        npm install; \
    else \
        npm ci --only=production; \
    fi

# 运行时配置（通过 ENV）
ENV NODE_ENV=${NODE_ENV}
ENV APP_VERSION=${APP_VERSION}

# 此时 NODE_ENV 在构建期和运行期均可用
```

不同参数构建：

```bash
# 生产构建
docker build --build-arg NODE_ENV=production --build-arg APP_VERSION=v2.3.1 -t myapp:prod .

# 开发构建（包含 devDependencies）
docker build --build-arg NODE_ENV=development -t myapp:dev .
```

## 常见模式与反模式


![Multi stage build factory assembly line producing optimized](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/03-multi-stage-build-factory-assembly-line-producing-optimized-.jpg)

### 模式：锁定软件包版本

```dockerfile
# BAD：每日构建结果不一致
RUN apt-get install -y curl

# GOOD：可重现
RUN apt-get install -y curl=7.88.1-10+deb12u4

# GOOD（pip）
RUN pip install flask==3.0.0 gunicorn==21.2.0

# GOOD（npm）
COPY package-lock.json .
RUN npm ci  # 使用 lockfile 确保精确版本
```

### 模式：最小化层数

```dockerfile
# BAD：4 层完成单一逻辑操作
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git
RUN apt-get clean

# GOOD：1 层
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
```

### 模式：使用 `COPY --link` 提升缓存效率

Docker BuildKit 支持 `--link`，使 `COPY` 不依赖于先前层：

```dockerfile
COPY --link requirements.txt .
```

### 反模式：将密钥作为构建参数传入

```dockerfile
# 绝对禁止！密码将永久保存在镜像历史中
ARG DB_PASSWORD
RUN echo "password=$DB_PASSWORD" > /app/config

# 正确做法：使用 BuildKit secrets
RUN --mount=type=secret,id=db_password \
    cat /run/secrets/db_password > /app/config
```

构建命令：

```bash
docker build --secret id=db_password,src=./db_password.txt -t myapp .
```

## 构建与打标签（Tagging）

```bash
# 基础构建
docker build -t myapp .

# 指定 Dockerfile 构建
docker build -f Dockerfile.prod -t myapp:prod .

# 多标签构建
docker build -t myapp:v2.3.1 -t myapp:latest .

# 指定平台构建
docker build --platform linux/amd64 -t myapp .

# 强制不使用缓存（全量重建）
docker build --no-cache -t myapp .
```

## 快速参考：Dockerfile 指令汇总表

| 指令 | 用途 | 是否创建层？ |
|------------|---------|---------------|
| `FROM` | 设置基础镜像 | 是（基础层） |
| `RUN` | 构建时执行命令 | 是 |
| `COPY` | 从构建上下文复制文件 | 是 |
| `ADD` | 复制文件（支持 tar 解压） | 是 |
| `WORKDIR` | 设置工作目录 | 是（若目录不存在则创建） |
| `ENV` | 设置环境变量 | 否（元数据） |
| `ARG` | 定义构建时变量 | 否（元数据） |
| `EXPOSE` | 声明端口 | 否（元数据） |
| `CMD` | 默认命令 | 否（元数据） |
| `ENTRYPOINT` | 固定命令 | 否（元数据） |
| `USER` | 设置后续指令用户 | 否（元数据） |
| `HEALTHCHECK` | 定义健康检查 | 否（元数据） |
| `LABEL` | 添加键值对元数据 | 否（元数据） |
| `VOLUME` | 创建挂载点 | 否（元数据） |
| `STOPSIGNAL` | 设置停止信号 | 否（元数据） |
| `SHELL` | 设置默认 shell | 否（元数据） |

## 下一步

你现在已能编写出体积小、安全性高、构建速度快的 Dockerfile。但孤立的容器价值有限——它需要与外部世界通信，并持久化数据。下一篇文章将讲解 Docker 网络（容器如何与彼此及宿主机通信）和卷（Volume，如何让数据在容器重启后依然存在）。这些是构建多容器应用（如 Docker Compose）前必须掌握的核心基石。