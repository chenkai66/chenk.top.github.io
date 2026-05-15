---
title: "Docker 与容器（四）：网络与卷——容器如何通信与持久化数据"
date: 2023-06-19 09:00:00
tags:
  - Docker
  - Containers
  - Networking
  - Volumes
  - Storage
categories: Docker and Containers
series: docker-containers
lang: zh
description: "容器默认是临时性的——删除后数据即丢失，且运行在隔离的网络中。卷（Volumes）和网络（Networks）是两种关键机制，使容器能接入持久化存储与可通信的外部世界。"
disableNunjucks: true
series_order: 4
series_total: 8
translationKey: "docker-containers-4"
---
容器被刻意设计为相互隔离，这正是其核心价值；但实际应用必须能接收外部连接、与数据库通信，并持久化容器重启后仍需保留的数据。Docker 通过**网络（Networking）**（管理容器间及对外通信）和**卷（Volumes）**（实现数据持久化）来满足这些需求；二者配置是否合理，直接决定了环境是仅供演示（demo）还是可用于生产（deployment）。

---

## Docker 网络

Docker 启动时自动创建宿主机上的虚拟网络基础设施：每个容器拥有独立的网络命名空间（含专属 IP、路由表和网络接口），Docker 负责调度容器间及容器与外部的流量。

![容器 DNS 解析](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/04-dns-resolution.png)

### 网络驱动（Network Drivers）

Docker 支持多种网络驱动，各自适用于不同场景：

| 驱动 | 描述 | 使用场景 | 容器间通信方式 |
|--------|-------------|----------|------------------------|
| `bridge` | 默认驱动。容器位于私有虚拟网络中。 | 单宿主机应用 | 通过 IP 或 DNS（仅限自定义 bridge） |
| `host` | 容器共享宿主机的网络命名空间 | 对性能极度敏感、单容器场景 | 容器直接使用宿主机 IP |
| `overlay` | 基于 VXLAN 的多宿主机网络 | Docker Swarm、多节点集群 | 跨物理宿主机 |
| `none` | 完全不启用网络 | 批处理任务、安全隔离 | 无 |
| `macvlan` | 容器在物理网络上获得 MAC 地址 | 需表现为物理主机的遗留应用 | 直连局域网（LAN） |
| `ipvlan` | 类似 macvlan，但共享宿主机 MAC 地址 | 与 macvlan 类似，交换机兼容性更好 | 直连局域网（LAN） |

绝大多数单宿主机场景均采用 **bridge** 网络，下文将逐一解析各驱动。

### 默认 Bridge 网络

安装 Docker 后，它会自动创建一个名为 `bridge` 的网络（底层由 Linux 桥接设备 `docker0` 实现）：

![桥接网络](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/04-bridge-network.png)

```bash
docker network ls
```

```text
NETWORK ID     NAME      DRIVER    SCOPE
a1b2c3d4e5f6   bridge    bridge    local
b2c3d4e5f6a7   host      host      local
c3d4e5f6a7b8   none      null      local
```

```bash
docker network inspect bridge --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}'
```

```text
172.17.0.0/16
```

所有未显式指定网络的容器都会自动加入默认 bridge 网络。我们来实际验证一下：

```bash
# 在默认 bridge 上运行两个容器
docker run -d --name container-a alpine sleep 3600
docker run -d --name container-b alpine sleep 3600

# 查看它们的 IP 地址
docker inspect container-a --format '{{.NetworkSettings.IPAddress}}'
# 输出：172.17.0.2

docker inspect container-b --format '{{.NetworkSettings.IPAddress}}'
# 输出：172.17.0.3

# 它们可通过 IP 互相访问
docker exec container-a ping -c 2 172.17.0.3
```

```text
PING 172.17.0.3 (172.17.0.3): 56 data bytes
64 bytes from 172.17.0.3: seq=0 ttl=64 time=0.108 ms
64 bytes from 172.17.0.3: seq=1 ttl=64 time=0.090 ms
```

但在默认 bridge 上，DNS 解析**不可用**：

```bash
docker exec container-a ping -c 2 container-b
# 输出：ping: bad address 'container-b'
```

这是默认 bridge 的主动限制——如需服务发现（service discovery），应使用**自定义 bridge 网络**。

```bash
# 清理
docker rm -f container-a container-b
```

### 自定义 Bridge 网络

自定义 bridge 网络提供容器间的自动 DNS 解析——这才是应用程序应采用的方式：

```bash
# 创建自定义网络
docker network create my-app-network

# 在该网络上运行容器
docker run -d --name web --network my-app-network nginx
docker run -d --name api --network my-app-network alpine sleep 3600

# DNS 解析生效
docker exec api ping -c 2 web
```

```text
PING web (172.18.0.2): 56 data bytes
64 bytes from 172.18.0.2: seq=0 ttl=64 time=0.065 ms
64 bytes from 172.18.0.2: seq=1 ttl=64 time=0.078 ms
```

容器名 `web` 可直接解析为其 IP 地址。这正是 Docker 内置 DNS 服务器在起作用：Docker 会在每个加入自定义网络的容器内部、以 `127.0.0.11` 地址运行一个嵌入式 DNS 服务器。

自定义网络还提供天然隔离：不同网络的容器默认互不连通，仅当显式连接时才可通信。

```bash
# 创建另一个网络
docker network create other-network
docker run -d --name isolated --network other-network alpine sleep 3600

# 此命令将失败（因处于不同网络）
docker exec isolated ping -c 2 web
# 输出：ping: bad address 'web'

# 将容器连接到多个网络
docker network connect my-app-network isolated

# 现在可以通信了
docker exec isolated ping -c 2 web
```

```text
PING web (172.18.0.2): 56 data bytes
64 bytes from 172.18.0.2: seq=0 ttl=64 time=0.089 ms
```

```bash
# 清理
docker rm -f web api isolated
docker network rm my-app-network other-network
```

### 对比：默认 Bridge vs 自定义 Bridge

| 特性 | 默认 Bridge | 自定义 Bridge |
|---------|---------------|---------------|
| DNS 解析 | 不支持（仅支持 IP） | 支持（按容器名解析） |
| 自动隔离 | 所有容器默认互通 | 仅显式连接的容器可互通 |
| 运行时连接/断开 | 需重启容器 | 支持实时连接/断开 |
| 环境变量链接（linking） | 依赖过时的 `--link` | 无需（直接使用 DNS） |
| 创建方式 | 自动创建 | `docker network create` |

**结论**：请始终为你的应用程序创建自定义 bridge 网络。默认 bridge 仅用于向后兼容。

### 端口映射（Port Mapping）

容器拥有自己的网络命名空间。要将容器端口暴露给宿主机（进而暴露给外部世界），需进行端口映射：

![端口映射](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/04-port-mapping.png)

```bash
# 将宿主机端口 8080 映射到容器端口 80
docker run -d -p 8080:80 --name web nginx

# 将宿主机端口 3307 映射到容器端口 3306
docker run -d -p 3307:3306 --name db mysql:8

# 绑定到特定宿主机接口
docker run -d -p 127.0.0.1:8080:80 --name local-only nginx

# 映射端口范围
docker run -d -p 8000-8010:8000-8010 --name range-app myapp

# 让 Docker 自动分配随机宿主机端口
docker run -d -p 80 --name random-port nginx

# 查看分配的端口
docker port random-port
# 输出：80/tcp -> 0.0.0.0:32768
```

格式始终为 `HOST:CONTAINER`，含义是“宿主机端口 8080 转发至容器端口 80”。

**`EXPOSE` vs `-p`**：

| 特性 | `EXPOSE 80`（Dockerfile 中） | `-p 8080:80`（`docker run` 中） |
|---------|-------------------------|--------------------------|
| 效果 | 仅为文档说明 | 实际发布该端口 |
| 网络访问 | 无任何效果 | 宿主机可通过端口 8080 访问容器 |
| 是否必需？ | 否 | 是（对外部访问而言） |
| 容器间通信 | 不需要（在自定义网络中直接使用 DNS） | 不需要 |

### Host 网络模式

`host` 网络模式完全移除网络隔离——容器直接使用宿主机的网络协议栈：

```bash
docker run -d --network host --name web nginx

# nginx 现在直接监听宿主机的 80 端口——无需端口映射
curl http://localhost:80
```

这消除了端口映射与 NAT 的开销，带来轻微的网络性能提升。代价是：你失去了端口隔离能力（两个容器无法同时监听 80 端口），且容器可看到宿主机的所有网络接口。

适用场景：
- 需要极致网络性能
- 应用程序需动态绑定大量端口
- 运行监控工具（需观测宿主机网络流量）

### 网络管理命令

```bash
# 列出所有网络
docker network ls

# 创建带选项的网络
docker network create \
    --driver bridge \
    --subnet 172.20.0.0/16 \
    --gateway 172.20.0.1 \
    --ip-range 172.20.240.0/20 \
    my-custom-net

# 检查网络详情（含已连接容器、配置等）
docker network inspect my-custom-net

# 将运行中的容器连接至某网络
docker network connect my-custom-net my-container

# 断开容器与某网络的连接
docker network disconnect my-custom-net my-container

# 删除网络（要求无容器连接）
docker network rm my-custom-net

# 删除所有未使用的网络
docker network prune
```

## Docker 卷（Volumes）

默认情况下，容器内写入的数据全部保存在可写层（writable layer）；容器删除后，这些数据即永久丢失。**卷（Volumes）** 提供了一种独立于容器生命周期之外的持久化存储机制。

![卷类型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/04-volume-types.png)

### 三种挂载类型（Mount Types）

Docker 支持三种将存储挂载进容器的方式：

| 类型 | 语法 | 管理方 | 宿主机位置 | 典型用途 |
|------|--------|-----------|-------------------|----------|
| 命名卷（Named volume） | `-v mydata:/data` | Docker | `/var/lib/docker/volumes/mydata/` | 数据库、应用持久化数据 |
| 绑定挂载（Bind mount） | `-v /host/path:/container/path` | 用户 | 任意用户指定路径 | 开发（实时代码重载） |
| tmpfs 挂载 | `--tmpfs /tmp` | 内核 | 仅内存（RAM） | 敏感数据、缓存 |

### 命名卷（Named Volumes）

命名卷是推荐的数据持久化方案：Docker 自动管理存储位置，并支持跨容器共享。

```bash
# 创建命名卷
docker volume create app-data

# 运行挂载该卷的容器
docker run -d \
    --name writer \
    -v app-data:/data \
    alpine sh -c "echo 'persistent data' > /data/message.txt && sleep 3600"

# 验证数据存在
docker exec writer cat /data/message.txt
# 输出：persistent data

# 删除容器
docker rm -f writer

# 数据仍存在——启动新容器并挂载同一卷
docker run --rm \
    -v app-data:/data \
    alpine cat /data/message.txt
# 输出：persistent data
```

卷生命周期管理命令：

```bash
# 列出所有卷
docker volume ls
```

```text
DRIVER    VOLUME NAME
local     app-data
local     postgres-data
local     redis-data
```

```bash
# 检查卷详情
docker volume inspect app-data
```

```json
[
    {
        "CreatedAt": "2023-09-22T10:00:00Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/app-data/_data",
        "Name": "app-data",
        "Options": {},
        "Scope": "local"
    }
]
```

```bash
# 删除卷（要求无容器正在使用）
docker volume rm app-data

# 删除所有未使用的卷（危险操作——不可逆！）
docker volume prune
```

### 绑定挂载（Bind Mounts）

绑定挂载将宿主机上的特定目录映射进容器。这对开发工作流至关重要：

```bash
# 将当前目录挂载进容器
docker run -d \
    --name dev-server \
    -v $(pwd):/app \
    -p 5000:5000 \
    python:3.11-slim \
    bash -c "cd /app && pip install flask && python app.py"
```

宿主机文件的任何修改均实时同步至容器（反之亦然），实现无需重建镜像的热重载开发。

**命名卷 vs 绑定挂载对比**：

```bash
# 命名卷（Docker 管理位置）
docker run -v mydata:/app/data myapp

# 绑定挂载（用户指定精确路径）
docker run -v /home/user/project/data:/app/data myapp

# 绑定挂载 + 只读标志
docker run -v /home/user/config:/app/config:ro myapp
```

| 特性 | 命名卷 | 绑定挂载 |
|---------|-------------|------------|
| 创建方式 | `docker volume create` 或自动创建 | 宿主机上已存在的目录 |
| 存储位置 | `/var/lib/docker/volumes/...` | 宿主机任意路径 |
| 管理工具 | Docker CLI (`docker volume ...`) | 宿主机文件系统工具 |
| 初始化填充 | 是（首次使用时，容器内容会复制进卷） | 否（宿主机内容直接覆盖容器内容） |
| 权限管理 | Docker 自动处理所有权 | 需手动管理 |
| macOS 性能 | 更优（使用 gRPC FUSE/VirtioFS） | 大型目录树下可能较慢 |
| 备份方式 | `docker run --rm -v mydata:/data -v $(pwd):/backup alpine tar czf /backup/data.tar.gz /data` \|标准宿主机备份工具 \|
| 典型用途 | 生产环境数据持久化 | 开发（实时重载） |

### tmpfs 挂载

tmpfs 挂载仅存在于内存中，**绝不会写入宿主机文件系统**：

```bash
docker run -d \
    --name secure-app \
    --tmpfs /tmp:size=100m,mode=1777 \
    --tmpfs /run/secrets:size=1m,mode=0700 \
    myapp
```

适用场景：
- 不应持久化的临时文件
- 敏感数据（密钥、令牌），禁止落盘
- I/O 密集型缓存（磁盘 I/O 成为瓶颈时）

## 真实案例：MySQL 与持久化数据

![Docker 网络桥连接容器，如同岛屿相连](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/04-docker-network-bridge-connecting-containers-like-islands-con.jpg)

本例展示卷为何至关重要。若不使用卷，删除 MySQL 容器将导致所有数据彻底丢失。

![覆盖网络](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/04-overlay-network.png)

```bash
# 创建网络与卷
docker network create db-network
docker volume create mysql-data

# 运行带持久化存储的 MySQL
docker run -d \
    --name mysql-server \
    --network db-network \
    -v mysql-data:/var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=rootpass \
    -e MYSQL_DATABASE=myapp \
    -e MYSQL_USER=appuser \
    -e MYSQL_PASSWORD=apppass \
    -p 3306:3306 \
    mysql:8.0

# 等待 MySQL 启动（查看日志）
docker logs -f mysql-server 2>&1 | grep -m 1 "ready for connections"
```

```text
2023-09-22T10:05:23.456789Z 0 [System] [MY-010931] [Server] /usr/sbin/mysqld: ready for connections. Version: '8.0.34'  socket: '/var/run/mysqld/mysqld.sock'  port: 3306  MySQL Community Server - GPL.
```

```bash
# 连接并创建一些数据
docker exec -it mysql-server mysql -u appuser -papppass myapp -e "
CREATE TABLE users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100));
INSERT INTO users (name) VALUES ('Alice'), ('Bob'), ('Charlie');
SELECT * FROM users;
"
```

```text
+----+---------+
| id | name    |
+----+---------+
|  1 | Alice   |
|  2 | Bob     |
|  3 | Charlie |
+----+---------+
```

现在模拟一次灾难——删除并重建容器：

```bash
# 删除容器（数据保存在卷中，而非容器内）
docker rm -f mysql-server

# 用同一卷重建容器
docker run -d \
    --name mysql-server \
    --network db-network \
    -v mysql-data:/var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=rootpass \
    -p 3306:3306 \
    mysql:8.0

# 等待启动后检查数据
docker exec -it mysql-server mysql -u appuser -papppass myapp -e "SELECT * FROM users;"
```

```text
+----+---------+
| id | name    |
+----+---------+
|  1 | Alice   |
|  2 | Bob     |
|  3 | Charlie |
+----+---------+
```

数据完好无损！卷（`mysql-data`）独立于容器存在。只要挂载相同的卷，你就可以自由地删除和重建容器。

### 卷备份与恢复

```bash
# 备份：运行临时容器，挂载卷并生成 tar 包
docker run --rm \
    -v mysql-data:/source:ro \
    -v $(pwd):/backup \
    alpine tar czf /backup/mysql-backup-$(date +%Y%m%d).tar.gz -C /source .

# 检查备份文件
ls -lh mysql-backup-*.tar.gz
# 输出：-rw-r--r-- 1 user user 45M Sep 22 11:00 mysql-backup-20230922.tar.gz

# 恢复：创建新卷并解压备份
docker volume create mysql-restored
docker run --rm \
    -v mysql-restored:/target \
    -v $(pwd):/backup \
    alpine tar xzf /backup/mysql-backup-20230922.tar.gz -C /target
```

### 卷权限问题

常见问题：容器进程以特定 UID 运行，而卷中文件归属用户（如 root）不一致。

```bash
# 问题：容器以 UID 1000 运行，但卷文件属主为 root
docker run -v mydata:/data myapp
# 报错：Permission denied writing to /data

# 解决方案 1：在 Dockerfile 中设置所有权
RUN mkdir /data && chown 1000:1000 /data
VOLUME /data

# 解决方案 2：以匹配 UID 运行容器
docker run --user 1000:1000 -v mydata:/data myapp

# 解决方案 3：使用初始化脚本修复权限
# （官方数据库镜像常用模式）
```

## 数据模式指南：何时该用哪种存储？

| 场景 | 存储类型 | 示例 |
|----------|-------------|---------|
| 数据库文件 | 命名卷 | `-v postgres-data:/var/lib/postgresql/data` |
| 应用日志 | 命名卷或绑定挂载 | `-v app-logs:/var/log/app` |
| 配置文件 | 绑定挂载（只读） | `-v ./config:/app/config:ro` |
| 源代码（开发） | 绑定挂载 | `-v $(pwd):/app` \|
| 用户上传文件 | 命名卷 | `-v uploads:/app/uploads` |
| 临时/暂存文件 | tmpfs | `--tmpfs /tmp` |
| 敏感凭据（Secrets） | tmpfs 或 Docker Secrets | `--tmpfs /run/secrets` |
| 容器间共享数据 | 命名卷 | 两个容器挂载同一卷 |
| 构建缓存 | 命名卷 | `-v build-cache:/root/.cache` |

## 综合实践：带网络与持久化的应用

![Docker 卷作为浮动存储晶体连接到容器](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/04-docker-volumes-as-floating-storage-crystals-connected-to-con.jpg)

以下是一个完整示例——Python 应用连接 Redis，二者均运行于自定义网络，且 Redis 数据持久化：

```bash
# 创建基础设施
docker network create app-net
docker volume create redis-data

# 运行带持久化的 Redis
docker run -d \
    --name redis \
    --network app-net \
    -v redis-data:/data \
    redis:7 \
    redis-server --appendonly yes

# 运行应用
docker run -d \
    --name app \
    --network app-net \
    -p 8080:5000 \
    -e REDIS_HOST=redis \
    my-flask-app

# 应用可通过名称 "redis" 访问 Redis（自定义网络 DNS）  
# 外部世界通过 8080 端口访问应用
# Redis 数据在容器重启后依然保留
```

该手动配置虽可行，但冗长且易出错；管理 5 个服务时，复杂度将急剧上升。这正是 **Docker Compose** 要解决的问题。

```bash
# 清理全部资源
docker rm -f redis app
docker network rm app-net
docker volume rm redis-data mysql-data
```

## 检查网络与卷状态

```bash
# 查看容器所有端口映射
docker port my-container
```

```text
80/tcp -> 0.0.0.0:8080
443/tcp -> 0.0.0.0:8443
```

```bash
# 查看容器网络配置
docker inspect my-container --format '{{json .NetworkSettings.Networks}}' | python3 -m json.tool
```

```json
{
    "my-app-network": {
        "IPAMConfig": null,
        "Links": null,
        "Aliases": ["my-container", "a1b2c3d4e5f6"],
        "NetworkID": "abc123...",
        "EndpointID": "def456...",
        "Gateway": "172.18.0.1",
        "IPAddress": "172.18.0.2",
        "IPPrefixLen": 16,
        "MacAddress": "02:42:ac:12:00:02"
    }
}
```

```bash
# 查看容器挂载详情
docker inspect my-container --format '{{json .Mounts}}' | python3 -m json.tool
```

```json
[
    {
        "Type": "volume",
        "Name": "app-data",
        "Source": "/var/lib/docker/volumes/app-data/_data",
        "Destination": "/data",
        "Driver": "local",
        "Mode": "z",
        "RW": true,
        "Propagation": ""
    }
]
```

## 下一步

你现在已掌握如何通过网络让容器彼此及与外部世界通信，也了解了如何借助卷实现数据持久化。但若对 5 个服务逐一执行带 10 个参数的 `docker run` 命令，不仅繁琐易错，更无法与团队成员共享。下一篇文章将介绍 **Docker Compose**——一种声明式方法，只需一个 YAML 文件即可定义并运行多容器应用。本文涵盖的所有内容（网络、卷、端口映射等）都将被简洁地表达在 `docker-compose.yml` 中，它既是基础设施的文档，也是可执行的部署蓝图。
