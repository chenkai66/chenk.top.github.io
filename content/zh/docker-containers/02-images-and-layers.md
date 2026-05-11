---
title: "Docker 与容器（二）：镜像与分层——`docker pull` 到底下载了什么？"
date: 2023-06-18 09:00:00
tags:
  - Docker
  - Containers
  - Images
  - Registry
  - OCI
categories:
  - Docker and Containers
series: docker-containers
lang: zh
description: "Docker 镜像并非单一的巨型文件，而是由多个只读分层（layers）堆叠而成，且这些分层可在不同容器间共享。理解分层机制，是实现快速构建与精简镜像的关键。"
disableNunjucks: true
series_order: 2
translationKey: "docker-containers-2"
---

第一次运行 `docker pull ubuntu` 时，我本以为会下载一整套操作系统。结果它几秒就完成了，体积仅 77 MB —— 对一个 Linux 发行版而言，这小得不可思议。其中的奥秘正是「分层」（layers）；而理解分层的工作原理，将彻底改变你构建和分发容器的方式。

## 镜像 vs 容器

在深入分层机制之前，我们先厘清一个常令初学者困惑的基础概念。

![Union filesystem](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-union-fs.png)


**镜像（Image）** 是一个只读模板，包含构建容器所需的全部内容：文件系统、环境变量、默认命令及元数据。你可以把它类比为面向对象编程中的「类定义（class definition）」。

**容器（Container）** 是由镜像创建的正在运行（或已停止）的实例。它拥有镜像的一切，外加一层可写层（writable layer），以及运行时状态（如网络配置、进程 ID 等）。你可以把它类比为从类实例化出的「对象（object）」。

```bash
# 镜像（模板）
docker images
```

```
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
nginx        latest    61395b4c586d   2 weeks ago   187MB
ubuntu       22.04     c6b84b685f35   3 weeks ago   77.8MB
```

```bash
# 由镜像创建的容器（实例）
docker ps -a
```

```
CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS                     NAMES
a1b2c3d4e5f6   nginx          "/docker-entrypoint.…"   10 minutes ago   Up 10 minutes              web1
b2c3d4e5f6a7   nginx          "/docker-entrypoint.…"   8 minutes ago    Up 8 minutes               web2
c3d4e5f6a7b8   ubuntu:22.04   "bash"                   5 minutes ago    Exited (0) 3 minutes ago   test
```

注意：两个容器（`web1` 和 `web2`）均基于同一 `nginx` 镜像运行。它们共享相同的只读分层，但各自拥有独立的可写层。`web1` 中的修改不会影响 `web2`，二者也均不影响原始镜像。

## 分层模型（Layer Model）

每个 Docker 镜像均由一组分层堆叠而成。每一层代表一组文件系统变更 —— 文件的新增、修改或删除。这些分层具有以下特性：

![Docker image layer stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-layer-stack.png)


1. **只读性（Read-only）**：一旦创建，分层永不更改；
2. **内容寻址（Content-addressable）**：通过其内容的 SHA256 哈希值唯一标识；
3. **可共享（Shared）**：若两个镜像使用相同基础层，则该层在磁盘上仅存储一份；
4. **堆叠式（Stacked）**：联合文件系统（union filesystem）将所有分层合并为一个统一、连贯的视图。

下面以一个典型 Dockerfile 为例，直观说明分层如何工作：

```dockerfile
FROM ubuntu:22.04          # 第 1 层：Ubuntu 基础文件系统
RUN apt-get update         # 第 2 层：更新后的软件包列表
RUN apt-get install -y curl # 第 3 层：curl 二进制文件及其依赖
COPY app.py /app/app.py    # 第 4 层：你的应用文件
CMD ["python3", "/app/app.py"]  # 仅元数据（不生成新分层）
```

每条修改文件系统的指令都会创建一个新分层。`CMD` 指令仅设置元数据，不改动任何文件，因此不产生新分层。

当容器从此镜像启动时，Docker 会在最上方额外添加一层：

```
[可写容器层]  ← 运行时的修改发生于此
[第 4 层：COPY app.py]      ← 只读
[第 3 层：apt install curl]  ← 只读
[第 2 层：apt update]        ← 只读
[第 1 层：ubuntu:22.04]      ← 只读
```

若你在容器内修改了底层分层中的某个文件，联合文件系统将采用 **写时复制（copy-on-write）** 机制：先将该文件复制到可写层，再修改副本；原始只读分层中的文件保持不变。

## 拉取镜像：`docker pull` 到底下载了什么？

我们来追踪 `docker pull nginx` 的实际执行过程：

![Layer sharing between containers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-layer-sharing.png)


```bash
docker pull nginx
```

```
Using default tag: latest
latest: Pulling from library/nginx
a2abf6c4d29d: Pull complete
a9edb18cadd1: Pull complete
589b7251471a: Pull complete
186b1aaa4aa6: Pull complete
b4df32aa5a72: Pull complete
a0bcbecc962e: Pull complete
Digest: sha256:0d17b565c37bcbd895e9d92315a05c1c3c9a29f762b011a10c54a66cd53c9b31
Status: Downloaded newer image for nginx:latest
docker.io/library/nginx:latest
```

共下载了六个分层。每行 `Pull complete` 对应一个独立分层（真实终端中你会看到并行下载的进度条）。`Digest` 字段是镜像清单（manifest）的 SHA256 哈希值，它唯一标识了这一组特定分层的组合。

现在拉取另一个共享相同基础层的镜像：

```bash
docker pull nginx:alpine
```

```
Using default tag: latest
alpine: Pulling from library/nginx
59bf1c3509f3: Already exists
8d6ba530f648: Pull complete
5288d7ad7a7f: Pull complete
39e51c61c033: Pull complete
ee6f71c6f4a8: Pull complete
Digest: sha256:6a2a8c246fa1c0ee9c9af9e41f51f14b4cc0e0f20a0bfa9e7f0e5e4f25abf2c3
Status: Downloaded newer image for nginx:alpine
docker.io/library/nginx:alpine
```

注意 `59bf1c3509f3: Already exists` —— Docker 识别出本地已存在该分层（很可能与 Alpine 基础镜像共享），因而跳过下载。这就是分层共享（layer sharing）的实际体现：它同时节省带宽与磁盘空间。

## 检查镜像分层


![Build cache mechanism](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-build-cache.png)

### `docker history`

`docker history` 命令展示镜像中每一层的来源指令及其大小：

```bash
docker history nginx
```

```
IMAGE          CREATED       CREATED BY                                      SIZE      COMMENT
61395b4c586d   2 weeks ago   /bin/sh -c #(nop)  CMD ["nginx" "-g" "daemon…   0B
<missing>      2 weeks ago   /bin/sh -c #(nop)  STOPSIGNAL SIGQUIT           0B
<missing>      2 weeks ago   /bin/sh -c #(nop)  EXPOSE 80                    0B
<missing>      2 weeks ago   /bin/sh -c #(nop)  ENTRYPOINT ["/docker-entr…   0B
<missing>      2 weeks ago   /bin/sh -c #(nop) COPY file:caec368f5a54f70a…   4.62kB
<missing>      2 weeks ago   /bin/sh -c #(nop) COPY file:01e75c6dd0ce317d…   3.02kB
<missing>      2 weeks ago   /bin/sh -c #(nop) COPY file:7b307b62e82255f0…   298B
<missing>      2 weeks ago   /bin/sh -c set -x     && addgroup --system -…   61.1MB
<missing>      2 weeks ago   /bin/sh -c #(nop)  ENV PKG_RELEASE=1~bookworm   0B
<missing>      2 weeks ago   /bin/sh -c #(nop)  ENV NJS_VERSION=0.8.0        0B
<missing>      2 weeks ago   /bin/sh -c #(nop)  ENV NGINX_VERSION=1.25.2     0B
<missing>      2 weeks ago   /bin/sh -c #(nop)  LABEL maintainer=NGINX Do…   0B
<missing>      2 weeks ago   /bin/sh -c #(nop)  CMD ["bash"]                 0B
<missing>      2 weeks ago   /bin/sh -c #(nop) ADD file:756183bba9c7f4593…   74.8MB
```

自下而上阅读（最老的层在底部）：

1. `ADD file:756...` — 74.8 MB — Debian 基础文件系统；
2. 大块 `set -x && addgroup...` — 61.1 MB — nginx 安装；
3. 多个 `COPY` 指令 — 各几 KB — 配置文件；
4. `ENV`, `EXPOSE`, `CMD` 等 — 0 字节 — 纯元数据。

`IMAGE` 列中的 `<missing>` 表示这些中间层没有自己的镜像标签。只有最终层（顶部）拥有镜像 ID `61395b4c586d`。

### `docker image inspect`

获取 JSON 格式的详细元数据：

```bash
docker image inspect nginx --format '{{json .RootFS}}' | python3 -m json.tool
```

```json
{
    "Type": "layers",
    "Layers": [
        "sha256:2edcec3590a4ec7f40cf0743c15d78fb39d8326bc029073b41ef9727da6c851f",
        "sha256:e379e8aedd4d72bb4c529a4ca07a4e4d230b5a1d3f7ea8d4e0cfbbe85a1c0e10",
        "sha256:b8d6e692a25e11b0d32c5c3dd544b71b1085ddc1fddad08e68cbd7fda7f70221",
        "sha256:f1db227348d0a5e0b99b15a096d930d1a69db7474a1847acbc31f05e4ef8df8c",
        "sha256:32ce5f6a5106cc637d09a98289782edf47c32cb082dc475dd43d5f9f0b1e5867",
        "sha256:d874fd2bc83bb3322b566df739681fbd2248c58d3369cb0e7b48b2e8a4e97a52"
    ]
}
```

这些是各分层的内容寻址 SHA256 哈希值。Docker 正是依靠它们判断某一分层是否已在本地存在。

## 镜像命名与镜像仓库（Registry）

Docker 镜像遵循如下命名约定：

![Image registry workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-image-registry.png)


```
[registry/][namespace/]repository[:tag][@digest]
```

示例：

| 完整名称 | 仓库（Registry） | 命名空间（Namespace） | 仓库（Repository） | 标签（Tag） |
|-----------|----------|-----------|------------|-----|
| `nginx` | docker.io（隐式） | library（隐式） | nginx | latest（隐式） |
| `nginx:1.25` | docker.io | library | nginx | 1.25 |
| `ubuntu:22.04` | docker.io | library | ubuntu | 22.04 |
| `myuser/myapp:v2` | docker.io | myuser | myapp | v2 |
| `gcr.io/project/app:prod` | gcr.io | project | app | prod |
| `ghcr.io/owner/repo:sha-abc123` | ghcr.io | owner | repo | sha-abc123 |
| `registry.example.com:5000/team/svc:latest` | registry.example.com:5000 | team | svc | latest |

关键规则：

- **省略仓库地址** 默认为 `docker.io`（即 Docker Hub）；
- **省略标签（tag）** 默认为 `latest`（仅为惯例，并不保证是最新的版本）；
- **Docker Hub 上的官方镜像** 无命名空间（例如 `nginx`, `ubuntu`, `python`）；
- **用户镜像** 必须有命名空间（例如 `myuser/myapp`）；
- **摘要（digest）**（`@sha256:...`）是不可变的 —— 标签可被重新指向不同镜像，但摘要永久固定。

### Docker Hub

Docker Hub 是默认的公共镜像仓库。当你运行 `docker pull nginx` 时，Docker 会连接 `registry-1.docker.io` 下载镜像。

```bash
# 从 CLI 搜索 Docker Hub
docker search python --limit 5
```

```
NAME                  DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
python                Python is an interpreted, interactive, objec…   9283      [OK]
pypy                  PyPy is a fast, compliant alternative implem…   380       [OK]
circleci/python       Python is an interpreted, interactive, objec…   55
cimg/python           The CircleCI Python Docker Convenience Image    10
bitnami/python        Bitnami Python Docker Image                     25                   [OK]
```

### 私有镜像仓库（Private Registries）

你可以自建私有仓库，或使用云服务商提供的托管服务：

```bash
# 登录私有仓库
docker login registry.example.com

# 为私有仓库打标签
docker tag myapp:latest registry.example.com/team/myapp:v1.0

# 推送至私有仓库
docker push registry.example.com/team/myapp:v1.0

# 从私有仓库拉取
docker pull registry.example.com/team/myapp:v1.0
```

常见私有镜像仓库：

| 仓库 | 提供方 |
|----------|----------|
| Amazon ECR | AWS |
| Google Artifact Registry | GCP |
| Azure Container Registry | Azure |
| GitHub Container Registry (ghcr.io) | GitHub |
| Docker Hub（私有仓库） | Docker |
| Harbor | 自托管（CNCF 项目） |
| JFrog Artifactory | JFrog |

## 镜像大小：为何重要？

镜像大小直接影响：

- **拉取耗时（Pull time）**：越大越慢，拖慢部署；
- **构建耗时（Build time）**：大分层上传更慢；
- **磁盘占用（Disk space）**：每个节点需本地存储镜像；
- **安全攻击面（Security surface）**：文件越多，潜在漏洞越多；
- **冷启动延迟（Cold start）**：Serverless 平台（如 AWS Lambda、Cloud Run）对大镜像响应更慢。

对比几种基础镜像大小：

```bash
docker pull ubuntu:22.04
docker pull debian:bookworm-slim
docker pull alpine:3.18
docker pull gcr.io/distroless/static-debian12
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

```
REPOSITORY                          TAG                 SIZE
ubuntu                              22.04               77.8MB
debian                              bookworm-slim       74.8MB
alpine                              3.18                7.34MB
gcr.io/distroless/static-debian12   latest              2.45MB
```

| 基础镜像 | 大小 | Shell | 包管理器 | 使用场景 |
|-----------|------|-------|-----------------|----------|
| `ubuntu:22.04` | 77.8 MB | bash | apt | 开发、调试、熟悉度高 |
| `debian:bookworm-slim` | 74.8 MB | bash | apt | 生产环境（官方镜像常用） |
| `alpine:3.18` | 7.34 MB | sh | apk | 极致轻量生产环境、尺寸敏感场景 |
| `distroless/static` | 2.45 MB | 无 | 无 | 仅支持静态编译二进制（如 Go） |
| `scratch` | 0 MB | 无 | 无 | 最小化（Go 二进制等） |

Alpine 比 Ubuntu 小约 10 倍，Distroless 更小约 30 倍。代价是：越小的镜像，调试工具越少。你无法对 distroless 容器执行 `docker exec -it container bash`，因为其中根本不存在 `bash`。

我们将在下一篇关于 Dockerfile 的文章中深入探讨优化策略。

## 导出与导入镜像

### `docker save` / `docker load`

这两个命令操作镜像 tar 归档文件，适用于无镜像仓库时的镜像迁移：

```bash
# 将镜像保存为 tar 文件
docker save nginx:latest -o nginx-latest.tar

# 查看文件大小
ls -lh nginx-latest.tar
# 输出：-rw------- 1 user user 188M Sep 14 10:30 nginx-latest.tar

# 在另一台机器上加载镜像
docker load -i nginx-latest.tar
```

```
Loaded image: nginx:latest
```

tar 文件包含所有分层（各自为独立 tar 文件）及清单（manifest）：

```bash
tar tf nginx-latest.tar | head -20
```

```
2edcec3590a4ec7f40cf0743c15d78fb39d8326bc029073b41ef9727da6c851f/
2edcec3590a4ec7f40cf0743c15d78fb39d8326bc029073b41ef9727da6c851f/VERSION
2edcec3590a4ec7f40cf0743c15d78fb39d8326bc029073b41ef9727da6c851f/json
2edcec3590a4ec7f40cf0743c15d78fb39d8326bc029073b41ef9727da6c851f/layer.tar
e379e8aedd4d72bb4c529a4ca07a4e4d230b5a1d3f7ea8d4e0cfbbe85a1c0e10/
e379e8aedd4d72bb4c529a4ca07a4e4d230b5a1d3f7ea8d4e0cfbbe85a1c0e10/VERSION
e379e8aedd4d72bb4c529a4ca07a4e4d230b5a1d3f7ea8d4e0cfbbe85a1c0e10/json
e379e8aedd4d72bb4c529a4ca07a4e4d230b5a1d3f7ea8d4e0cfbbe85a1c0e10/layer.tar
manifest.json
repositories
```

每个目录对应一个分层，每个 `layer.tar` 包含该分层的文件系统变更。

### `docker export` / `docker import`

这两个命令作用于**容器（而非镜像）**，输出扁平化的文件系统快照：

```bash
# 创建容器并做些修改
docker run --name test-export ubuntu:22.04 bash -c "echo 'hello' > /data.txt"

# 导出容器的文件系统
docker export test-export -o test-export.tar

# 导入为新镜像
docker import test-export.tar my-custom-ubuntu:v1
```

核心区别：

| 操作 | 作用对象 | 是否保留分层？ | 是否保留元数据？（CMD、ENV 等） |
|-----------|-----------|-------------------|---------------------|
| `save/load` | 镜像 | 是 | 是 |
| `export/import` | 容器 | 否（扁平化为单层） | 否 |

用 `save/load` 在机器间迁移镜像；仅在需要扁平化文件系统快照时才用 `export/import`。

## 镜像标签（Tagging）

标签（tag）是可变的指针，指向某个具体镜像摘要（digest）。你可以自由创建：

```bash
# 为现有镜像打新标签
docker tag nginx:latest my-nginx:v1.0
docker tag nginx:latest my-nginx:production
docker tag nginx:latest registry.example.com/team/nginx:v1.0

# 查看镜像列表 — 注意它们共享同一 IMAGE ID
docker images | grep -E "nginx|my-nginx"
```

```
REPOSITORY                        TAG          IMAGE ID       CREATED       SIZE
nginx                             latest       61395b4c586d   2 weeks ago   187MB
my-nginx                          v1.0         61395b4c586d   2 weeks ago   187MB
my-nginx                          production   61395b4c586d   2 weeks ago   187MB
registry.example.com/team/nginx   v1.0         61395b4c586d   2 weeks ago   187MB
```

四个条目均指向同一镜像（`61395b4c586d`），无数据重复。标签只是指针。

### “latest” 标签陷阱

`latest` 标签对 Docker 来说并无特殊含义，它只是一个**惯例（convention）**，而非机制。Docker 不会自动将 `latest` 指向最新版本。如果有人推送了 `myapp:v2` 却未同步更新 `latest`，那么 `latest` 仍指向旧版本。

最佳实践：**生产环境务必使用明确的标签。**

```bash
# 错误 —— 这到底是什么版本？
docker pull myapp:latest

# 正确 —— 明确且可复现
docker pull myapp:v2.3.1

# 最佳 —— 不可变，永不更改
docker pull myapp@sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4
```

## 清理镜像

镜像会快速累积。以下是释放磁盘空间的方法：

```bash
# 查看磁盘使用情况
docker system df
```

```
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          5         2         450.2MB   312.4MB (69%)
Containers      3         1         12.5kB    12.5kB (100%)
Local Volumes   2         1         256MB     128MB (50%)
Build Cache     15        0         1.2GB     1.2GB
```

```bash
# 删除指定镜像
docker rmi nginx:alpine

# 删除所有未被使用的镜像（未被任何容器引用）
docker image prune

# 删除所有未被运行中容器使用的镜像（激进模式）
docker image prune -a

# 终极清理 —— 删除所有未使用的资源（镜像、容器、卷、网络）
docker system prune -a --volumes
```

`docker system prune -a --volumes` 具有破坏性：它将移除所有已停止的容器、所有未使用的网络、所有未被至少一个运行中容器引用的镜像，以及所有未被至少一个容器使用的卷。请仅在开发机上使用，切勿用于生产环境。

## 检查镜像内部内容

有时你想查看镜像内容，却不想真正运行容器：

```bash
# 创建容器但不启动
docker create --name peek nginx:latest

# 复制文件出来
docker cp peek:/etc/nginx/nginx.conf ./nginx.conf

# 或浏览整个文件系统
docker export peek | tar tf - | head -30
```

```
.dockerenv
bin
boot/
dev/
docker-entrypoint.d/
docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
docker-entrypoint.d/15-local-resolvers.sh
docker-entrypoint.d/20-envsubst-on-templates.sh
docker-entrypoint.d/30-tune-worker-processes.sh
docker-entrypoint.sh
etc/
etc/adduser.conf
etc/alternatives/
...
```

```bash
# 清理
docker rm peek
```

你也可以使用第三方工具（如 `dive`）交互式地逐层浏览：

```bash
# 安装 dive（https://github.com/wagoodman/dive）
# 然后分析镜像
dive nginx:latest
```

`dive` 可显示每层的具体内容，让你看清每层新增/修改/删除了哪些文件，并估算浪费的空间。

## 分层在磁盘上的存储方式

在 Linux 主机上，Docker 将所有数据存于 `/var/lib/docker/` 下。确切结构取决于所用存储驱动（通常为 OverlayFS）：

```bash
# 查看存储驱动信息
docker info --format '{{.Driver}}'
# 输出：overlay2

# 查看分层存储位置
ls /var/lib/docker/overlay2/ | head -5
```

```
2edcec3590a4ec7f40cf0743c15d78fb39d8326bc029073b41ef9727da6c851f
backingFsBlockDev
e379e8aedd4d72bb4c529a4ca07a4e4d230b5a1d3f7ea8d4e0cfbbe85a1c0e10
l
```

`overlay2/` 下每个目录即一个分层。`l/` 目录存放用于分层识别的短符号链接。**切勿直接修改这些文件** —— 请交由 Docker 自动管理。

## 多架构镜像（Multi-Architecture Images）

现代 Docker 镜像常在一个标签下支持多种 CPU 架构：

```bash
docker manifest inspect nginx:latest | python3 -m json.tool | head -30
```

```json
{
    "schemaVersion": 2,
    "mediaType": "application/vnd.oci.image.index.v1+json",
    "manifests": [
        {
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": "sha256:...",
            "size": 1234,
            "platform": {
                "architecture": "amd64",
                "os": "linux"
            }
        },
        {
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": "sha256:...",
            "size": 1234,
            "platform": {
                "architecture": "arm64",
                "os": "linux"
            }
        }
    ]
}
```

当你在 ARM Mac 上执行 `docker pull nginx`，Docker 会自动选择 `arm64` 变体；而在 x86_64 Linux 服务器上则选择 `amd64`。**同一标签，不同二进制** —— 这正是跨平台部署无缝衔接的原因。

## 下一步

你现在已理解：镜像是由多个只读分层堆叠而成；分层可在镜像间共享；容器在此之上叠加一层薄薄的可写层；你也掌握了检查、导出、打标签及清理镜像的方法。

下一步是构建你自己的镜像 —— 即编写 Dockerfile。一个朴素的 Dockerfile 与一个经过优化的 Dockerfile，其差异可能就是：一个 1.5 GB、耗时 10 分钟构建的镜像，与一个 50 MB、30 秒即可完成构建的镜像之间的鸿沟。下一篇将详述每一个 Dockerfile 指令，以及区分开发型与生产型 Dockerfile 的关键模式。