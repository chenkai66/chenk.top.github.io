---
title: "Docker 与容器（一）：为何需要容器——虚拟机未能解决的问题"
date: 2023-06-16 09:00:00
tags:
  - Docker
  - Containers
  - Virtualization
  - Linux
  - DevOps
categories: Docker and Containers
series: docker-containers
lang: zh
description: "容器解决了虚拟机虽能应对却代价高昂的‘在我机器上能跑’问题。本文将详解容器的本质、它与虚拟机的根本区别，并带你运行第一个容器。"
disableNunjucks: true
series_order: 1
translationKey: "docker-containers-1"
---

每位开发者都听过那句经典吐槽：“在我机器上是能跑的。”虚拟机虽然解决了这个问题，但代价高昂：动辄数 GB 的内存占用、数分钟启动时间以及为每个应用单独部署一整套操作系统。

而容器则通过不复制内核，仅隔离关键组件来实现应用级隔离。

## 真正的问题所在

设想部署一个 Python Web 应用：你需要 Python 3.11、特定版本的 pip 包、某个特定版本的 `libssl`，以及若干系统级配置。而你的同事的应用却依赖 Python 3.9 和一个与之冲突的 `libssl` 版本。预发布环境运行的是 Ubuntu 20.04，而生产环境却是 Amazon Linux 2。

![VMs vs containers comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-vm-vs-container.png)


虚拟机通过为每个应用提供一套完整操作系统来解决该问题，但这带来了巨大的资源开销：每个虚拟机都包含

- 一个完整内核（数百 MB）
- 各类系统守护进程（如 init、 syslog、 cron —— 这些你的应用根本不需要）
- 多份重复的共享库副本
- 自身独立的内存管理开销

容器则通过共享宿主机内核，仅隔离文件系统、进程树、网络协议栈和资源限制等关键组件，以更轻量的方式实现同等隔离。

## 容器究竟是什么？

容器不是“轻量级虚拟机”——这个类比便于理解，但容易引发误导。本质上，容器就是（或一组）普通 Linux 进程，仅靠三项内核机制实现隔离：

![Container architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-container-architecture.png)


### 命名空间（Namespaces）—— 进程“能看到什么”

Linux 命名空间将内核资源进行分区，使一组进程只能看到某一套资源，而另一组进程则看到另一套。命名空间有多种类型：

![Namespaces and cgroups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-namespace-cgroup.png)


| 命名空间 | 隔离内容 | 效果 |
|-----------|----------|--------|
| `pid` | 进程 ID | 容器内仅可见自身进程；容器内 PID 1 并非宿主机上的 PID 1 |
| `net` | 网络协议栈 | 容器拥有自己的 IP 地址、路由表和端口 |
| `mnt` | 挂载点 | 容器拥有自己独立的文件系统视图 |
| `uts` | 主机名 | 容器可设置自己的主机名 |
| `ipc` | 进程间通信（IPC） | 共享内存与信号量被隔离 |
| `user` | 用户/组 ID | 容器内的 root 可映射为宿主机上的非 root 用户 |
| `cgroup` | Cgroup 根目录 | 容器仅可见自身的 cgroup 层级结构 |

当你执行 `docker run nginx` 时， Docker 会创建一组新的命名空间，并在其中启动 nginx 进程。该进程在宿主机上仍是一个普通进程（你可用 `ps aux` 查看），但从容器内部视角看，它仿佛独占整台机器。

### 控制组（Cgroups）—— 进程“能用多少”

控制组（cgroups）用于限制并统计资源使用量。命名空间控制的是“可见性”，而 cgroups 控制的是“消耗量”：

| 资源 | Cgroup 控制器 | 限制内容 |
|----------|-------------------|----------------|
| CPU | `cpu`, `cpuacct` | CPU 时间、份额、配额 |
| 内存 | `memory` | RAM 使用量、 swap、 OOM 行为 |
| 磁盘 I/O | `blkio` | 对块设备的读写速率 |
| 网络 | `net_cls`, `net_prio` | 流量分类与优先级 |
| 进程数（PIDs） | `pids` | 最大进程数量 |

没有 cgroups，失控容器可能耗尽宿主机全部内存，导致系统崩溃；借助 cgroups，则可明确声明：该容器最多使用 512MB 内存和 0.5 个 CPU 核心。

### 联合文件系统（Union Filesystem）—— 文件系统如何工作

容器采用分层式文件系统。它不是为每个容器复制一份完整的操作系统文件系统，而是利用联合文件系统（现代 Linux 上常用 OverlayFS）将多个只读层叠加在一起。底层为只读层（多个容器可共享），每个容器在其上拥有一层轻量级的可写层。

这正是容器启动极快的原因：无需复制整个文件系统，只需指向已存在的只读层，并附加一个全新的空可写层。

## 容器 vs 虚拟机


![Shipping containers stacked on a digital cargo ship represen](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/01-shipping-containers-stacked-on-a-digital-cargo-ship-represen.jpg)

二者在架构层面存在根本差异。以下是对两种技术栈的对比（可想象为一张自底向上的示意图）：

![Resource overhead comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-resource-comparison.png)


**虚拟机栈（自底向上）：**  
```
硬件 → 宿主机操作系统 → 虚拟机监控器（Hypervisor） → [客户操作系统 + 二进制/库 + 应用] × N
```

**容器栈（自底向上）：**  
```
硬件 → 宿主机操作系统（共享内核） → 容器运行时 → [二进制/库 + 应用] × N
```

核心区别在于：虚拟机虚拟化硬件，容器虚拟化操作系统——前者各自运行独立内核，后者共享宿主机内核。

以下是具体对比：

| 特性 | 虚拟机 | 容器 |
|---------------|----------------|-----------|
| 启动时间 | 30–60 秒 | < 1 秒 |
| 磁盘占用 | 每台 VM 1–20 GB | 每个容器 10–500 MB |
| 内存开销 | 512 MB – 数 GB | 几乎为零（共享内核） |
| 隔离级别 | 硬件级（强） | 进程级（良好，但非完美） |
| 内核 | 每台 VM 独立内核 | 共享宿主机内核 |
| 操作系统支持 | 任意 OS （如 Linux 上运行 Windows） | 必须与宿主机同内核家族 |
| 密度 | 单台宿主机典型承载 10–20 台 VM | 单台宿主机可承载数百个容器 |
| 实时迁移 | 支持 | 不原生支持（需编排器实现） |
| 性能 | 硬件虚拟化下接近原生 | 原生（无虚拟化层） |

**何时仍应选择虚拟机？**  
- 当你需要强安全隔离（如多租户云环境）  
- 当你需要不同内核（例如在 Linux 宿主机上运行 Windows 工作负载）  
- 当合规性要求强制规定必须进行硬件级隔离  

## 容器生态体系： OCI、 Docker、 containerd、 Podman


![Virtual machine vs container architecture split screen compa](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/01-virtual-machine-vs-container-architecture-split-screen-compa.jpg)

Docker 让容器广为人知，但它绝非唯一玩家。理解整个生态体系可以避免后续混淆。

![Container isolation layers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-isolation-layers.png)


### OCI 标准

开放容器倡议（Open Container Initiative, OCI）定义了两项规范：

1. **镜像规范（Image Spec）**：定义容器镜像的结构（分层、清单 manifest、配置等）  
2. **运行时规范（Runtime Spec）**：定义如何运行容器（生命周期、配置格式等）

任何遵循这些规范的工具均可构建可在任意平台运行的镜像。这也是为什么你可以用 Docker 构建镜像、推送到任意镜像仓库、再用 Podman 运行它的原因。

### 容器运行时栈

该栈呈分层结构， Docker 位于顶层：

```
docker CLI → dockerd（Docker 守护进程） → containerd → runc → Linux 内核
```

| 组件 | 角色 | 是否可单独使用？ |
|-----------|------|----------------------|
| `runc` | OCI 运行时 —— 实际创建命名空间与 cgroups | 可以，但属底层操作 |
| `containerd` | 管理容器生命周期、镜像拉取与存储 | 可以（Kubernetes 直接使用） |
| `dockerd` | Docker 守护进程 —— 提供构建、网络、卷等功能 | 可以（绝大多数用户所用） |
| `docker` CLI | 面向用户的命令行工具 | 与 dockerd 通信 |

### Docker vs Podman

Podman 是一款无守护进程（daemonless）的容器引擎，不依赖守护进程（dockerd），而是直接运行容器。其命令语法几乎完全一致：

```bash
# Docker
docker run -d -p 8080:80 nginx

# Podman（相同语法）
podman run -d -p 8080:80 nginx
```

关键差异如下：

| 特性 | Docker | Podman |
|---------|--------|--------|
| 守护进程 | 是（dockerd） | 否（无守护进程） |
| 是否需要 root 权限 | 默认需要（支持 rootless 模式） | 默认即 rootless |
| Compose 支持 | docker compose | podman-compose （或兼容方案） |
| Swarm 集群 | 内置 | 不支持 |
| Kubernetes YAML 支持 | 无原生支持 | `podman generate kube` |
| systemd 集成 | 需手动配置 | 原生支持 |

对于学习和大多数生产场景， Docker 仍是事实标准。而当你关注 rootless 运行或原生 systemd 集成时， Podman 则更具价值。

## 容器简史

容器并非横空出世，其发展历经数十年演进：

| 年份 | 技术 | 功能 |
|------|-----------|-------------|
| 1979 | `chroot` | 更改进程根目录 —— 首次实现文件系统隔离 |
| 2000 | FreeBSD Jails | 结合文件系统、进程与网络隔离 |
| 2001 | Linux VServer | 将 Linux 分割为多个虚拟私有服务器 |
| 2004 | Solaris Zones | 具备资源控制的完整操作系统级虚拟化 |
| 2006 | Cgroups （Google） | 资源限制与计量 —— 后合并入 Linux 内核 |
| 2008 | LXC | 将 cgroups 与 namespaces 结合，形成 Linux 容器 |
| 2013 | Docker | 以简洁 CLI 和镜像格式让容器变得易用 |
| 2014 | Kubernetes | Google 开源其容器编排系统 |
| 2015 | OCI | 容器镜像与运行时的行业标准 |
| 2017 | containerd | 从 Docker 中剥离，成为独立运行时（CNCF 项目） |

Docker 的贡献并非发明了新技术（命名空间与 cgroups 在 Docker 之前早已存在），而在于提升开发者体验：简洁的 CLI、可移植的镜像格式，以及用于共享镜像的公共仓库（Docker Hub）。

## 安装 Docker

### Linux （Ubuntu/Debian）

请勿从发行版默认仓库安装 Docker，这些包通常严重过时。请使用 Docker 官方仓库：

```bash
# 卸载旧版本
sudo apt-get remove docker docker-engine docker.io containerd runc

# 安装前置依赖
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg

# 添加 Docker 官方 GPG 密钥
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 添加仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker 引擎
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 将当前用户加入 docker 组（避免每次命令都需 sudo）
sudo usermod -aG docker $USER
newgrp docker
```

### Linux （CentOS/RHEL/Fedora）

```bash
# 卸载旧版本
sudo yum remove docker docker-client docker-client-latest docker-common \
  docker-latest docker-latest-logrotate docker-logrotate docker-engine

# 安装前置依赖并添加仓库
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# 安装 Docker 引擎
sudo yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 启动并启用 Docker
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户加入 docker 组
sudo usermod -aG docker $USER
```

### macOS

macOS 上的标准方案是 Docker Desktop。 Docker 无法在 macOS 上原生运行，它实际上运行于 Docker Desktop 托管的一个轻量级 Linux 虚拟机中。

1. 从 [docker.com](https://www.docker.com/products/docker-desktop/) 下载 Docker Desktop  
2. 打开 `.dmg` 文件，将 Docker 拖拽至 Applications 文件夹  
3. 启动 Docker Desktop —— 它将请求权限以安装网络组件  
4. 等待菜单栏中的 Docker 图标显示 “Docker Desktop is running”

或者使用 Homebrew：

```bash
brew install --cask docker
```

然后从 Applications 中打开 Docker Desktop 完成安装。

### 验证安装

```bash
docker version
```

预期输出：

```yaml
Client: Docker Engine - Community
 Version:           24.0.6
 API version:       1.43
 Go version:        go1.20.7
 Git commit:        ed223bc
 Built:             Mon Sep  4 12:31:44 2023
 OS/Arch:           linux/amd64
 Context:           default

Server: Docker Engine - Community
 Engine:
  Version:          24.0.6
  API version:      1.43 (minimum version 1.12)
  Go version:       go1.20.7
  Git commit:       1a79695
  Built:            Mon Sep  4 12:31:44 2023
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          1.6.24
  GitCommit:        61f9fd88f79f081d64d6fa3bb1a0dc71ec870523
 runc:
  Version:          1.1.9
  GitCommit:        v1.1.9-0-gccaecfc
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
```

注意其中 “Client” 与 “Server” 两部分：客户端是 CLI 工具；服务端是实际管理容器的守护进程。在 macOS 上，服务端运行于 Docker Desktop 的虚拟机中。

## 运行你的第一个容器

```bash
docker run hello-world
```

输出：

```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
c1ec31eb5944: Pull complete
Digest: sha256:4bd78111b6914a99dbc560e6a20eab57ff6655aea4a80c50b0c5491968cbc2e6
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more interesting, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

让我们拆解发生了什么：

1. `docker run` 命令指示 Docker 客户端运行一个容器  
2. 客户端将请求发送给 Docker 守护进程（dockerd）  
3. 守护进程在本地查找 `hello-world:latest` 镜像 —— 未找到  
4. 守护进程从 Docker Hub 拉取该镜像（你可看到分层下载过程）  
5. 守护进程基于该镜像创建容器  
6. 容器运行其程序（打印欢迎消息），随后退出  

## 探索 Docker 信息

```bash
docker info
```

该命令揭示了 Docker 安装的详细配置：

```yaml
Client: Docker Engine - Community
 Version:    24.0.6
 Context:    default
 Debug Mode: false
 Plugins:
  buildx: Docker Buildx (Docker Inc.)
    Version:  v0.11.2
  compose: Docker Compose (Docker Inc.)
    Version:  v2.21.0

Server:
 Containers: 1
  Running: 0
  Paused: 0
  Stopped: 1
 Images: 1
 Server Version: 24.0.6
 Storage Driver: overlay2
  Backing Filesystem: extfs
  Supports d_type: true
  Using metacopy: false
  Native Overlay Diff: true
  userxattr: false
 Logging Driver: json-file
 Cgroup Driver: systemd
 Cgroup Version: 2
 Plugins:
  Volume: local
  Network: bridge host ipvlan macvlan null overlay
  Log: awslogs fluentd gcplogs gelf journald json-file local logentries splunk syslog
 Swarm: inactive
 Runtimes: io.containerd.runc.v2 runc
 Default Runtime: runc
 Init Binary: docker-init
 containerd version: 61f9fd88f79f081d64d6fa3bb1a0dc71ec870523
 runc version: v1.1.9-0-gccaecfc
 init version: de40ad0
 Security Options:
  apparmor
  seccomp
   Profile: builtin
  cgroupns
 Kernel Version: 5.15.0-82-generic
 Operating System: Ubuntu 22.04.3 LTS
 OSType: linux
 Architecture: x86_64
 CPUs: 4
 Total Memory: 7.748GiB
 Name: docker-host
 ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
 Docker Root Dir: /var/lib/docker
 Debug Mode: false
 Experimental: false
 Insecure Registries:
  127.0.0.0/8
 Live Restore Enabled: false
```

需重点关注的关键项：

- **Storage Driver: overlay2** —— 当前使用的联合文件系统驱动  
- **Cgroup Driver: systemd** —— 资源限制的管理方式  
- **Docker Root Dir: /var/lib/docker** —— 所有镜像、容器与卷的存储位置  
- **Security Options** —— AppArmor 与 seccomp 配置文件已启用  
- **Runtimes: runc** —— 底层 OCI 运行时  

## 运行一个更有趣的例子

让我们运行一个交互式的 Ubuntu 容器：

```bash
docker run -it ubuntu bash
```

你现在已进入容器内部。开始探索吧：

```bash
# 查看主机名 —— 显示为容器 ID
hostname
# 输出：a3f8b2c1d4e5

# 查看进程列表 —— 仅有 bash 和 ps 在运行
ps aux
# 输出：
# USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
# root         1  0.0  0.0   4624  3840 pts/0    Ss   14:30   0:00 bash
# root        10  0.0  0.0   7060  1536 pts/0    R+   14:30   0:00 ps aux

# 查看操作系统版本
cat /etc/os-release
# 输出：
# PRETTY_NAME="Ubuntu 22.04.3 LTS"
# NAME="Ubuntu"
# ...

# 查看文件系统 —— 这是一个精简版 Ubuntu
ls /
# 输出：bin boot dev etc home lib lib64 media mnt opt proc root run sbin srv sys tmp usr var

# 退出容器
exit
```

注意 PID 1 是 `bash` —— 在容器内， bash 就是 init 进程。而在宿主机上，该进程拥有完全不同的 PID。这正是命名空间隔离的直观体现。

退出后，容器停止但并未被删除：

```bash
docker ps -a
```

```
CONTAINER ID   IMAGE         COMMAND   CREATED          STATUS                     PORTS   NAMES
a3f8b2c1d4e5   ubuntu        "bash"    2 minutes ago    Exited (0) 30 seconds ago          hopeful_nobel
b7c9e1f2a3d4   hello-world   "/hello"  5 minutes ago    Exited (0) 5 minutes ago           festive_darwin
```

## 运行后台容器

让我们以后台模式运行 nginx：

```bash
docker run -d -p 8080:80 --name my-nginx nginx
```

参数解析：

- `-d` —— 后台运行（detached mode）  
- `-p 8080:80` —— 将宿主机 8080 端口映射到容器 80 端口  
- `--name my-nginx` —— 为容器指定一个易记名称  
- `nginx` —— 使用的镜像  

```bash
# 检查是否正在运行
docker ps
```

```
CONTAINER ID   IMAGE   COMMAND                  CREATED         STATUS         PORTS                  NAMES
c5d6e7f8a9b0   nginx   "/docker-entrypoint.…"   5 seconds ago   Up 4 seconds   0.0.0.0:8080->80/tcp   my-nginx
```

```bash
# 测试访问
curl http://localhost:8080
```

```html
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
...
```

```bash
# 查看日志
docker logs my-nginx
```

```
/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
/docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
...
2023/09/10 14:35:00 [notice] 1#1: start worker processes
2023/09/10 14:35:00 [notice] 1#1: start worker process 29
172.17.0.1 - - [10/Sep/2023:14:35:15 +0000] "GET / HTTP/1.1" 200 615 "-" "curl/7.81.0" "-"
```

```bash
# 清理
docker stop my-nginx
docker rm my-nginx
```

## 核心命令速查表

| 命令 | 用途 |
|---------|---------|
| `docker run IMAGE` | 基于镜像创建并启动容器 |
| `docker ps` | 列出正在运行的容器 |
| `docker ps -a` | 列出所有容器（含已停止） |
| `docker stop CONTAINER` | 优雅地停止容器 |
| `docker rm CONTAINER` | 删除已停止的容器 |
| `docker images` | 列出已下载的镜像 |
| `docker rmi IMAGE` | 删除镜像 |
| `docker pull IMAGE` | 仅下载镜像（不运行） |
| `docker exec -it CONTAINER bash` | 在运行中的容器内开启 shell |
| `docker logs CONTAINER` | 查看容器输出日志 |
| `docker inspect CONTAINER` | 查看容器详细元数据（JSON 格式） |

## 下一步

现在你已了解容器的本质及其与虚拟机的根本区别，完成了 Docker 安装、运行了首个容器，并掌握了基础命令。但我们略过了一个关键点：当你执行 `docker run nginx` 时， Docker 下载了一个“镜像”。**镜像到底是什么？这些分层是如何组织的？为什么拉取 Ubuntu 时只下载了一个小层，而拉取 nginx 却下载了多个层？** 下一篇文章将深入探讨镜像与分层模型——这正是容器快速启动与节省空间的基石。