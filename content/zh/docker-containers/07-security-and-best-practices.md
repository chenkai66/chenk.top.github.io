---
title: "Docker 与容器（七）：安全——运行容器时不必交出全部权限"
date: 2023-06-22 09:00:00
tags:
  - Docker
  - Containers
  - Security
  - DevSecOps
  - Best Practices
categories: Docker and Containers
series: docker-containers
lang: zh
description: "容器提供隔离性，而非安全性。默认的 Docker 配置以 root 身份运行进程，并赋予其完整的 Linux capabilities。本文介绍如何为生产环境加固容器。"
disableNunjucks: true
series_order: 7
translationKey: "docker-containers-7"
---

Docker 默认配置优先便利性而非安全性：开箱即用时容器以 root （UID 0）运行、拥有大量 Linux capabilities、且根文件系统默认可写。开发环境或可容忍，但生产环境极其危险——若存在容器逃逸（container escape）漏洞且容器以 root 运行，攻击者将直接接管宿主机。让我们来修复这个问题。

## 威胁模型

实施加固前，先明确防御对象：

![安全层](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-security-layers.png)


1. **存在漏洞的应用代码**：你的应用存在缺陷（如远程代码执行 RCE、路径遍历、服务端请求伪造 SSRF），攻击者可在容器内获得代码执行能力  
2. **存在漏洞的依赖项**：镜像中某个库存在已知 CVE 漏洞  
3. **容器逃逸**：攻击者利用内核或运行时漏洞突破容器边界  
4. **供应链攻击**：使用了恶意基础镜像或软件包  
5. **密钥泄露**：凭据通过环境变量、镜像历史记录或日志意外暴露  
6. **横向移动**：攻击者从一个容器跳转至其他容器或宿主机  

每种加固技术对应一项或多项威胁，目标是构建纵深防御（defense in depth）：单一手段无法确保安全，但多层防护可显著抬高攻击成本。

## 以非 root 用户身份运行

默认情况下， Docker 容器进程以 root （UID 0）运行——该身份与宿主机 root 相同（启用 user namespaces 除外）；一旦容器逃逸，攻击者即获得宿主机 root 权限。

![无根容器](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-rootless-container.png)


### 在 Dockerfile 中指定

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 以 root 安装依赖（系统包安装需要 root）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 复制应用文件并设置所有权
COPY --chown=appuser:appuser . .

# 切换至非 root 用户，后续所有指令及运行时均以此用户执行
USER appuser

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

在基于 Alpine 的镜像中，语法略有不同：

```dockerfile
FROM python:3.11-alpine

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup -S appuser && adduser -S appuser -G appuser
COPY --chown=appuser:appuser . .
USER appuser

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

### 运行时覆盖

即使 Dockerfile 中未指定用户，也可在运行时强制覆盖：

```bash
# 以指定 UID:GID 运行
docker run --user 1000:1000 myapp

# 以 "nobody" 用户运行
docker run --user nobody myapp
```

### 验证当前用户

```bash
# 检查容器当前运行用户
docker exec my-container id
# 输出：uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)

# 对比默认容器（无 USER 指令）
docker exec default-container id
# 输出：uid=0(root) gid=0(root) groups=0(root)
```

### 常见非 root 用户陷阱

以非 root 用户运行可能破坏某些假设 root 权限的功能：

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| 无法绑定 < 1024 的端口 | 绑定端口 80 时报 `Permission denied` | 使用 8080+ 端口，并通过 `-p 80:8080` 映射 |
| 无法向目录写入 | `/var/log` 下报 `Permission denied` | `RUN mkdir -p /var/log/app && chown appuser /var/log/app` |
| 运行时无法安装包 | `apt-get` 失败 | 所有包应在 `USER` 指令前完成构建阶段安装 |
| 无法读取挂载的文件 | 卷挂载后报 `Permission denied` | 使容器 UID/GID 与宿主机匹配，或使用命名卷（named volumes） |
| 包管理器需 root 权限 | `npm`/`pip` 失败 | `pip` 使用 `--user` 标志，或在切换用户前完成安装 |

## 只读文件系统


![具有多层防御的容器安全堡垒](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/07-container-security-fortress-with-multiple-defense-layers.jpg)

只读根文件系统可阻断攻击者篡改二进制、注入恶意软件、修改配置等行为：

![镜像漏洞扫描](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-image-scanning.png)


```bash
# 以只读根文件系统运行
docker run --read-only myapp
```

大多数应用仍需向某些路径写入（临时文件、缓存、 PID 文件）。可使用 `tmpfs` 提供可写区域：

```bash
# 根文件系统只读，同时允许 `/tmp` 和 `/var/run` 写入
docker run --read-only \
    --tmpfs /tmp:size=100m \
    --tmpfs /var/run:size=1m \
    myapp
```

在 Docker Compose 中：

```yaml
services:
  api:
    image: myapp
    read_only: true
    tmpfs:
      - /tmp:size=100m
      - /var/run:size=1m
      - /app/cache:size=50m
```

请在开发阶段就用 `--read-only` 测试应用。若崩溃，错误信息会指出尝试写入的路径 —— 然后为其添加 `tmpfs`。

```bash
# 查看应用试图写入的位置
docker run --read-only myapp 2>&1 | grep "Read-only file system"
# 输出：OSError: [Errno 30] Read-only file system: '/app/logs/app.log'
# 解决方案：添加 --tmpfs /app/logs:size=50m
```

## Linux 功能


![以非特权用户运行的无根容器安全视图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/07-rootless-container-running-as-unprivileged-user-security-vis.jpg)

Linux capabilities 将 root 权限拆分为约 40 种独立特权，而 Docker 默认授予容器其中多项——往往远超应用实际需求。

![Linux 能力管理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-capability-model.png)


Docker 容器默认拥有的 capabilities：

| Capability | 权限 | 是否必需？ |
|-----------|------|------------|
| `CHOWN` | 更改文件属主 | 极少 |
| `DAC_OVERRIDE` | 绕过文件权限检查 | 极少 |
| `FSETID` | 设置 SUID/SGID 位 | 几乎从不 |
| `FOWNER` | 绕过文件属主权限检查 | 极少 |
| `MKNOD` | 创建特殊文件 | 几乎从不 |
| `NET_RAW` | 使用原始套接字（ping、抓包） | 有时需要 |
| `SETGID` | 设置组 ID | 有时（初始化脚本） |
| `SETUID` | 设置用户 ID | 有时（初始化脚本） |
| `SETFCAP` | 设置文件 capabilities | 几乎从不 |
| `SETPCAP` | 设置进程 capabilities | 几乎从不 |
| `NET_BIND_SERVICE` | 绑定 < 1024 的端口 | 仅当需监听 80/443 时 |
| `SYS_CHROOT` | 使用 chroot | 几乎从不 |
| `KILL` | 向其他进程发送信号 | 有时 |
| `AUDIT_WRITE` | 写入内核审计日志 | 极少 |

遵循最小权限原则（Principle of Least Privilege）：先全部禁用 capabilities，再按需逐个启用。

```bash
# 移除全部 capabilities，仅添加必要项
docker run \
    --cap-drop ALL \
    --cap-add NET_BIND_SERVICE \
    myapp

# 需绑定端口 80 的 Web 服务器
docker run \
    --cap-drop ALL \
    --cap-add NET_BIND_SERVICE \
    -p 80:80 \
    nginx

# 大多数应用无需任何 capability
docker run \
    --cap-drop ALL \
    myapp
```

在 Docker Compose 中：

```yaml
services:
  api:
    image: myapp
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

### 检查 capabilities

```bash
# 查看运行中容器拥有的 capabilities
docker exec my-container cat /proc/1/status | grep Cap

# 解码十六进制 capability 掩码
docker exec my-container capsh --decode=00000000a80425fb
```

## 密钥管理（Secrets Management）

密钥（API keys、数据库密码、 TLS 证书）是容器化应用最典型的安全失守环节。

### 错误的密钥处理方式

```dockerfile
# 绝对禁止：密钥作为构建参数（会保留在镜像历史中）
ARG DB_PASSWORD=supersecret
RUN echo "password=$DB_PASSWORD" >> /app/config

# 绝对禁止：在 Dockerfile 中通过 ENV 设置密钥
ENV API_KEY=sk-12345abcde

# 绝对禁止：将密钥文件 COPY 进镜像
COPY credentials.json /app/credentials.json
```

以上三种方式均会使密钥暴露给任意可拉取该镜像的用户：

```bash
# 构建参数在 history 中可见
docker history myapp
# 显示：ARG DB_PASSWORD=supersecret

# 环境变量在 inspect 中可见
docker inspect myapp --format '{{json .Config.Env}}'
# 显示：["API_KEY=sk-12345abcde"]

# 文件可从镜像中提取
docker create --name extract myapp
docker cp extract:/app/credentials.json .
```

### 环境变量（适用于多数场景）

运行时传入环境变量（而非在 Dockerfile 中定义）是最常用的方式：

```bash
docker run -e DB_PASSWORD=secret -e API_KEY=sk-12345 myapp
```

或使用文件：

```bash
docker run --env-file .env myapp
```

`.env` 文件绝不可提交至版本控制系统（务必加入 `.gitignore`）。

**环境变量的风险：**  
- 可通过 `docker inspect` 查看  
- 对容器内所有进程（含子进程）可见  
- 可能被意外记录（调试输出中的 `env | sort`、错误上报工具）  
- 在容器内可通过 `/proc/<pid>/environ` 查看  

### Docker BuildKit 密钥（用于构建时密钥）

BuildKit 支持在构建期间挂载密钥，且不会将其保存到任何镜像层中：

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.11-slim

# 构建时挂载密钥 —— 不会保存到任何镜像层
RUN --mount=type=secret,id=pip_extra_index \
    pip install --no-cache-dir \
    --extra-index-url $(cat /run/secrets/pip_extra_index) \
    -r requirements.txt
```

```bash
# 使用密钥构建
DOCKER_BUILDKIT=1 docker build \
    --secret id=pip_extra_index,src=./pip_index_url.txt \
    -t myapp .
```

密钥仅在 `RUN` 指令执行期间可用，不会写入镜像或任何层。

### Docker Swarm 密钥（用于运行时密钥）

若使用 Docker Swarm，密钥是一等公民：

```bash
# 创建密钥
echo "supersecretpassword" | docker secret create db_password -

# 在服务中使用密钥
docker service create \
    --name api \
    --secret db_password \
    myapp
```

在容器内，密钥以文件形式挂载至 `/run/secrets/db_password`。相比环境变量更安全，因为：  
- 它是 tmpfs 挂载（永不落盘）  
- 仅对显式声明依赖的服务可见  
- 可在不重启服务的前提下轮换密钥  

### 运行时挂载文件（非 Swarm 场景）

对于非 Swarm 部署，可通过 bind mounts 实现类似安全级别：

```bash
docker run \
    -v /secure/path/credentials.json:/run/secrets/credentials.json:ro \
    myapp
```

`:ro` 标志确保只读。结合 `--tmpfs /tmp` 和 `--read-only`，可防止密钥被复制到容器内其他位置。

## 使用 Trivy 进行镜像扫描

Trivy 是一款漏洞扫描器，可将容器镜像与已知 CVE 数据库进行比对：

```bash
# 安装 Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# 扫描镜像
trivy image myapp:latest
```

```
myapp:latest (debian 12.1)
===========================
Total: 45 (UNKNOWN: 0, LOW: 25, MEDIUM: 12, HIGH: 6, CRITICAL: 2)

+-------------------+------------------+----------+-------------------+-------------------+
|      LIBRARY      |  VULNERABILITY   | SEVERITY | INSTALLED VERSION |   FIXED VERSION   |
+-------------------+------------------+----------+-------------------+-------------------+
| libssl3           | CVE-2023-XXXXX   | CRITICAL | 3.0.9-1           | 3.0.11-1          |
| libcurl4          | CVE-2023-YYYYY   | CRITICAL | 7.88.1-10         | 7.88.1-10+deb12u4 |
| python3.11        | CVE-2023-ZZZZZ   | HIGH     | 3.11.4            | 3.11.5            |
+-------------------+------------------+----------+-------------------+-------------------+

Python (requirements.txt)
==========================
Total: 3 (HIGH: 2, MEDIUM: 1)

+-------------------+------------------+----------+-------------------+-------------------+
|      LIBRARY      |  VULNERABILITY   | SEVERITY | INSTALLED VERSION |   FIXED VERSION   |
+-------------------+------------------+----------+-------------------+-------------------+
| requests          | CVE-2023-AAAAA   | HIGH     | 2.28.0            | 2.31.0            |
| flask             | CVE-2023-BBBBB   | MEDIUM   | 2.2.0             | 2.3.3             |
+-------------------+------------------+----------+-------------------+-------------------+
```

Trivy 同时扫描操作系统包和应用依赖（pip、 npm、 gem 等）。

```bash
# 仅扫描 CRITICAL 和 HIGH 级别漏洞
trivy image --severity CRITICAL,HIGH myapp:latest

# 发现任意漏洞即失败（适用于 CI）
trivy image --exit-code 1 --severity CRITICAL myapp:latest

# 扫描 Dockerfile（构建前检查基础镜像）
trivy config Dockerfile

# 扫描本地文件系统
trivy fs --security-checks vuln,secret ./
```

### 在 CI 中集成 Trivy

```yaml
# GitHub Actions 示例
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'myapp:${{ github.sha }}'
    format: 'table'
    exit-code: '1'
    severity: 'CRITICAL,HIGH'
```

## 最小化基础镜像

镜像中文件越少，攻击面越小。对比以下基础镜像：

| 基础镜像 | 大小 | 包数量 | Shell | 安全态势 |
|----------|------|--------|-------|-----------|
| `ubuntu:22.04` | 78 MB | ~100 | bash | 攻击面大 |
| `debian:bookworm-slim` | 75 MB | ~80 | bash | 略小 |
| `alpine:3.18` | 7 MB | ~15 | sh | 小，使用 musl libc |
| `distroless/base` | 20 MB | ~5 | 无 | 极简，无 shell 访问 |
| `distroless/static` | 2 MB | ~2 | 无 | 仅含静态二进制文件 |
| `scratch` | 0 MB | 0 | 无 | 绝对最小 |

### Distroless 镜像

Google 的 distroless 镜像仅包含你的应用及其运行时依赖 —— 无 shell、无包管理器、无多余工具：

```dockerfile
# 使用 distroless 的多阶段构建
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY . .

FROM gcr.io/distroless/python3-debian12
WORKDIR /app
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/lib/python3.11/site-packages
COPY --from=builder /app .
ENTRYPOINT ["python3", "app.py"]
```

优势：  
- 无 shell 意味着 `docker exec bash` 失效 —— 攻击者无法获取交互式 shell  
- 无包管理器意味着攻击者无法安装工具  
- 文件越少，潜在 CVE 越少  

缺点：调试更困难（无法 `exec` 进入容器）。可参考上一篇文章中的“临时调试容器”技巧。

### Scratch 镜像（Go、 Rust）

对于静态编译语言，可直接使用 `scratch`（真正空镜像）：

```dockerfile
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /server .

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /server /server
EXPOSE 8080
ENTRYPOINT ["/server"]
```

最终镜像仅含一个二进制文件（加 CA 证书）。攻击面：几乎为零。

## Docker 内容信任（Docker Content Trust, DCT）

Docker 内容信任（DCT）使用数字签名验证镜像真实性：

```bash
# 启用内容信任
export DOCKER_CONTENT_TRUST=1

# 此后 pull/push 均需签名
docker pull nginx:latest
# 仅当镜像已签名时才成功

# 推送已签名镜像（需预先配置签名密钥）
docker push myrepo/myapp:v1.0
# Docker 将提示输入签名口令
```

DCT 基于 The Update Framework （TUF）管理密钥与签名。启用后：  
- `docker pull` 验证镜像是否由可信发布者签名  
- `docker push` 使用你的密钥对镜像签名  
- 未签名镜像将被拒绝  

这可防范因镜像仓库被入侵而导致的恶意镜像替换类供应链攻击。

## 资源限制

若无资源限制，容器可无限消耗 CPU、内存和磁盘 I/O，从而挤占其他容器及宿主机资源：

```bash
# 内存限制（超出即被 OOM kill）
docker run --memory 512m myapp

# 内存 + swap 限制
docker run --memory 512m --memory-swap 1g myapp

# CPU 限制（最多占用 0.5 个 CPU 核心）
docker run --cpus 0.5 myapp

# CPU 权重（相对权重，默认 1024）
docker run --cpu-shares 512 myapp

# 组合限制
docker run \
    --memory 512m \
    --memory-swap 512m \
    --cpus 1.0 \
    --pids-limit 100 \
    myapp
```

在 Docker Compose 中：

```yaml
services:
  api:
    image: myapp
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 128M
```

| 资源 | 参数 | 效果 |
|------|------|------|
| 内存 | `--memory 512m` | 硬限制，超出触发 OOM kill |
| 内存 + Swap | `--memory-swap 1g` | 总内存+swap 限制 |
| CPU | `--cpus 0.5` | 硬限制：单核的 50% |
| CPU 权重 | `--cpu-shares 512` | 相对权重（软限制） |
| 进程数 | `--pids-limit 100` | 最大进程数（防 fork bomb） |
| 磁盘 I/O | `--device-read-bps /dev/sda:1mb` | 磁盘带宽限制 |

`--pids-limit` 常被忽视，但可有效防御 fork bomb 攻击：

```bash
# 无 --pids-limit 时，fork bomb 可导致宿主机崩溃
# 有该限制时，容器最多运行 100 个进程
docker run --pids-limit 100 myapp
```

## 安全选项

### Seccomp 配置文件

Seccomp （Secure Computing Mode）用于过滤容器可调用的系统调用。 Docker 默认 seccomp 配置文件屏蔽了约 60 个高危 syscall：

![Seccomp 系统调用过滤](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-seccomp-profile.png)


```bash
# 使用默认 seccomp 配置文件（自动启用）
docker run myapp

# 使用自定义 seccomp 配置文件
docker run --security-opt seccomp=/path/to/profile.json myapp

# 禁用 seccomp（生产环境切勿使用）
docker run --security-opt seccomp=unconfined myapp
```

### AppArmor 与 SELinux

Docker 自动应用 AppArmor （Ubuntu/Debian）或 SELinux （RHEL/CentOS）配置文件：

```bash
# 查看 AppArmor 配置文件
docker inspect my-container --format '{{.AppArmorProfile}}'
# 输出：docker-default

# 使用自定义 AppArmor 配置文件
docker run --security-opt apparmor=my-custom-profile myapp
```

### 禁止新特权（No New Privileges）

防止容器内进程通过 setuid 二进制文件等方式获取新特权：

```bash
docker run --security-opt no-new-privileges:true myapp
```

在 Docker Compose 中：

```yaml
services:
  api:
    image: myapp
    security_opt:
      - no-new-privileges:true
```

## 安全最佳实践清单

| 实践 | 优先级 | 实施方式 |
|------|--------|-----------|
| 以非 root 用户运行 | 关键 | Dockerfile 中 `USER appuser` |
| 使用具体镜像标签 | 关键 | `FROM python:3.11.5-slim`，禁用 `latest` |
| 扫描镜像 CVE | 关键 | CI 流水线中执行 `trivy image myapp` |
| 移除全部 capabilities | 高 | `--cap-drop ALL --cap-add <所需>` |
| 使用只读文件系统 | 高 | `--read-only --tmpfs /tmp` |
| 设置内存限制 | 高 | `--memory 512m` |
| 使用 `.dockerignore` | 高 | 排除 `.git`、`.env`、密钥等 |
| 镜像中不存放密钥 | 关键 | 使用运行时环境变量、挂载文件或 Docker secrets |
| 使用多阶段构建 | 高 | 构建工具不进入生产镜像 |
| 启用 `no-new-privileges` | 中 | `--security-opt no-new-privileges:true` |
| 使用最小化基础镜像 | 中 | Alpine、 distroless 或 scratch |
| 锁定依赖版本 | 中 | lockfiles、精确版本号 |
| 设置 PID 限制 | 中 | `--pids-limit 100` |
| 启用内容信任 | 中 | `DOCKER_CONTENT_TRUST=1` |
| 添加健康检查 | 中 | `HEALTHCHECK CMD curl -f http://localhost/health` |
| 限制网络暴露 | 中 | 使用自定义网络，不暴露非必要端口 |
| 审计镜像历史 | 低 | `docker history --no-trunc myapp` |
| 使用只读卷 | 低 | `-v config:/app/config:ro` |

## 加固版 Docker Compose 示例

综合运用上述所有实践：

```yaml
services:
  api:
    build:
      context: ./api
      target: production
    read_only: true
    tmpfs:
      - /tmp:size=100m,mode=1777
      - /var/run:size=1m
    user: "1000:1000"
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          memory: 128M
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""]
      interval: 30s
      timeout: 5s
      retries: 3
    environment:
      DATABASE_URL: ${DATABASE_URL}
    ports:
      - "8000:8000"
    networks:
      - frontend
      - backend
    restart: unless-stopped
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  postgres:
    image: postgres:16-alpine
    read_only: true
    tmpfs:
      - /tmp
      - /var/run/postgresql
    user: "999:999"
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    deploy:
      resources:
        limits:
          memory: 1G
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    networks:
      - backend
    restart: unless-stopped

networks:
  frontend:
  backend:
    internal: true  # 无外部访问 —— 仅该网络内容器可通信

volumes:
  postgres-data:
```

注意 `backend` 网络设为 `internal: true` —— 该网络上的容器无法访问互联网，若数据库容器被攻破，可大幅缩小影响范围。

## 下一步

你现在已掌握如何加固单个容器：非 root 用户、最小化 capabilities、只读文件系统、镜像扫描、资源限制等。但安全只是挑战之一 —— 规模化才是另一难题。当单台宿主机不再足够时该怎么办？当你需要自动故障转移、滚动更新、跨多台机器的服务发现时又该如何？最后一篇文章将预览容器编排：面向简单性的 Docker Swarm，面向大规模的 Kubernetes，以及何时你其实根本不需要编排。