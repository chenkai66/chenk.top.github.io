---
title: "Docker 与容器（六）：调试与日志——当‘盒子’内部出问题时"
date: 2023-06-21 09:00:00
tags:
  - Docker
  - Containers
  - Debugging
  - Logging
  - Observability
categories: Docker and Containers
series: docker-containers
lang: zh
description: "容器按设计隐藏其内部实现。当系统出现故障时，你需要特定的工具和技巧，在不破坏容器核心隔离特性的前提下，窥探‘盒子’内部发生了什么。"
disableNunjucks: true
series_order: 6
translationKey: "docker-containers-6"
---
正常运行的容器近乎‘隐形’，而一旦出问题，它就立刻变成一个密不透风的‘黑盒’。容器化的核心优势在于隔离，但恰恰是这种隔离，让调试变得棘手——你没法像对待普通服务器那样直接 `ssh` 进去，也无法从宿主机随意浏览容器内部的文件系统。好在 Docker 提供了一整套专用工具，帮助你检查、诊断并理解运行中（甚至已崩溃）容器内部究竟发生了什么。

---

## 查看容器日志

日志是你排查问题的第一道防线。Docker 会自动捕获容器写入 stdout 和 stderr 的所有内容。

![资源监控](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/06-resource-monitoring.png)

### docker logs

```bash

![Logging drivers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/06-log-drivers.png)

# 查看容器全部日志
docker logs my-container

# 实时跟踪日志（类似 tail -f）
docker logs -f my-container

# 显示最后 100 行
docker logs --tail 100 my-container

# 显示指定时间点之后的日志
docker logs --since 2023-09-30T10:00:00 my-container

# 显示过去 30 分钟的日志
docker logs --since 30m my-container

# 每行日志前显示时间戳
docker logs -t my-container
```

带时间戳的输出示例如下：

```text
2023-09-30T10:15:23.456789012Z [INFO] Server starting on port 8080
2023-09-30T10:15:23.567890123Z [INFO] Connected to database at postgres:5432
2023-09-30T10:15:24.678901234Z [INFO] Loading configuration from /app/config.yaml
2023-09-30T10:15:24.789012345Z [WARNING] Cache directory /tmp/cache does not exist, creating
2023-09-30T10:15:25.890123456Z [INFO] Ready to accept connections
2023-09-30T10:16:01.234567890Z [ERROR] Failed to process request: connection refused
2023-09-30T10:16:01.345678901Z Traceback (most recent call last):
2023-09-30T10:16:01.345678901Z   File "/app/handler.py", line 45, in process_request
2023-09-30T10:16:01.345678901Z     response = requests.get(upstream_url, timeout=5)
2023-09-30T10:16:01.345678901Z   File "/usr/local/lib/python3.11/site-packages/requests/api.py", line 73
2023-09-30T10:16:01.345678901Z     return session.request(method=method, url=url, **kwargs)
2023-09-30T10:16:01.345678901Z ConnectionError: ('Connection aborted.', ConnectionRefusedError(111, 'Connection refused'))
```

### 已停止容器的日志

这一点至关重要：即使容器因崩溃而停止，其日志仍会被保留，直到你显式执行 `docker rm` 将其删除。

```bash
# 列出所有已停止的容器
docker ps -a --filter "status=exited"
```

```text
CONTAINER ID   IMAGE       COMMAND            CREATED          STATUS                     NAMES
a1b2c3d4e5f6   myapp:v2    "python app.py"    10 minutes ago   Exited (1) 8 minutes ago   crashed-app
```

```bash
# 查看崩溃日志
docker logs crashed-app
```

```text
Traceback (most recent call last):
  File "/app/app.py", line 12, in <module>
    db = psycopg2.connect(os.environ['DATABASE_URL'])
  File "/usr/local/lib/python3.11/site-packages/psycopg2/__init__.py", line 122
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
psycopg2.OperationalError: could not connect to server: Connection refused
    Is the server running on host "postgres" (172.18.0.2) and accepting
    TCP/IP connections on port 5432?
```

可以看到，该容器以退出码 1（错误）终止，日志显示它无法连接 PostgreSQL。这可能是由于应用启动时数据库尚未就绪（依赖顺序问题），也可能是配置的主机名有误。

### 退出码（Exit codes）

```bash
# 查看退出码
docker inspect crashed-app --format '{{.State.ExitCode}}'
# 输出: 1
```

常见退出码及其含义如下：

| 退出码 | 含义 | 常见原因 |
|--------|------|-----------|
| 0 | 成功 | 正常关闭 |
| 1 | 通用错误 | 应用异常、未捕获的异常 |
| 2 | Shell 内置命令误用 | 脚本语法错误 |
| 126 | 命令不可执行 | Entrypoint 权限不足 |
| 127 | 命令未找到 | CMD/ENTRYPOINT 配置错误或二进制文件缺失 |
| 137 | SIGKILL（128+9） | 被 OOM killer 终止、执行了 `docker kill` 或超时 |
| 139 | SIGSEGV（128+11） | 段错误（Segmentation fault） |
| 143 | SIGTERM（128+15） | 执行了 `docker stop`（优雅关闭） |

其中，退出码 137 尤其值得警惕——它通常意味着容器因超出内存限制，被内核的 OOM killer 强制终止。

```bash
# 检查容器是否被 OOM killer 终止
docker inspect crashed-app --format '{{.State.OOMKilled}}'
# 输出: true
```

## 使用 docker exec 进行交互式调试

`docker exec` 可在运行中的容器内执行任意命令，是交互式调试的首选方式：

![exec 与 attach 的区别](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/06-exec-vs-attach.png)

进入容器后，你就可以自由排查问题了：

```bash
# 在运行中的容器内打开 shell
docker exec -it my-container bash

# 若容器不含 bash（如 Alpine 或 distroless 镜像）
docker exec -it my-container sh

# 执行特定命令
docker exec my-container cat /app/config.yaml

# 以不同用户身份运行
docker exec -u root my-container apt-get update

# 为命令设置环境变量
docker exec -e DEBUG=true my-container python check.py
```

```bash
# 检查环境变量
env | sort

# 检查网络连通性
curl -v http://postgres:5432
ping redis

# 检查 DNS 解析
nslookup postgres
cat /etc/resolv.conf

# 检查运行进程
ps aux

# 检查文件权限
ls -la /app/
stat /app/config.yaml

# 检查磁盘使用量
df -h

# 检查内存使用
free -m
cat /proc/meminfo

# 检查打开的文件与连接（若工具可用）
# (if the tool is available)
ss -tlnp
netstat -tlnp
```

### 当 bash 不可用时

许多精简镜像（如 Alpine 或 distroless）为了减小体积，往往不包含 bash 甚至基本的调试工具：

```bash
# Alpine 默认使用 sh，而非 bash
docker exec -it alpine-container sh

# 临时安装调试工具（Alpine）
docker exec -it alpine-container apk add --no-cache curl bind-tools
```

对于 distroless 这类完全无 shell 的镜像，你根本无法通过 `exec` 进入。此时，推荐使用“调试 sidecar”模式：

```bash
# 运行一个共享目标容器网络命名空间的调试容器
docker run -it --rm \
    --network container:my-distroless-container \
    nicolaka/netshoot \
    bash
```

`nicolaka/netshoot` 镜像集成了几乎所有网络调试工具（curl、nslookup、tcpdump、iptables 等），配合 `--network container:my-distroless-container` 参数，就能让它共享目标容器的网络命名空间，从而实现间接调试。

## docker inspect — 获取完整视图

![容器日志流水线数据流从容器中流出](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/06-container-logging-pipeline-data-streams-flowing-from-contain.jpg)

`docker inspect` 会返回关于容器的完整 JSON 元数据。虽然输出冗长，但几乎囊括了你能想到的所有信息：

![故障排查决策树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/06-troubleshooting.png)

```bash
# 完整输出（非常长）
docker inspect my-container

# 使用 Go 模板提取特定字段
docker inspect my-container --format '{{.State.Status}}'
# 输出: running

docker inspect my-container --format '{{.State.StartedAt}}'
# 输出: 2023-09-30T10:15:23.123456789Z

docker inspect my-container --format '{{.Config.Image}}'
# 输出: myapp:v2

# 网络信息
docker inspect my-container --format '{{range .NetworkSettings.Networks}}IP: {{.IPAddress}} Gateway: {{.Gateway}}{{end}}'
# 输出: IP: 172.18.0.3 Gateway: 172.18.0.1

# 环境变量
docker inspect my-container --format '{{range .Config.Env}}{{println .}}{{end}}'
```

```text
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
DATABASE_URL=postgresql://postgres:secret@postgres:5432/myapp
REDIS_URL=redis://redis:6379
NODE_ENV=production
```

```bash
# 挂载点信息
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
        "Mode": "",
        "RW": true,
        "Propagation": ""
    },
    {
        "Type": "bind",
        "Source": "/home/user/config",
        "Destination": "/app/config",
        "Mode": "ro",
        "RW": false,
        "Propagation": "rprivate"
    }
]
```

```bash
# 端口映射
docker inspect my-container --format '{{json .NetworkSettings.Ports}}' | python3 -m json.tool
```

```json
{
    "8080/tcp": [
        {
            "HostIp": "0.0.0.0",
            "HostPort": "8080"
        }
    ]
}
```

### 实用的 inspect 单行命令

```bash
# 获取容器 IP 地址
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' my-container

# 获取容器 MAC 地址
docker inspect -f '{{range .NetworkSettings.Networks}}{{.MacAddress}}{{end}}' my-container

# 获取容器重启次数
docker inspect -f '{{.RestartCount}}' my-container

# 获取所用镜像
docker inspect -f '{{.Config.Image}}' my-container

# 获取当前运行的命令
docker inspect -f '{{json .Config.Cmd}}' my-container

# 检查容器是否正在运行
docker inspect -f '{{.State.Running}}' my-container
```

## docker stats — 实时资源监控

`docker stats` 提供容器资源消耗的实时动态视图：

```bash
docker stats
```

```text
CONTAINER ID   NAME              CPU %   MEM USAGE / LIMIT     MEM %   NET I/O           BLOCK I/O         PIDS
a1b2c3d4e5f6   myapp-api-1       2.34%   125.4MiB / 512MiB     24.49%  15.2kB / 8.9kB    4.1MB / 0B        12
b2c3d4e5f6a7   myapp-postgres-1  0.45%   89.2MiB / 1GiB        8.71%   12.1kB / 9.8kB    28.5MB / 12.3MB   15
c3d4e5f6a7b8   myapp-redis-1     0.12%   12.5MiB / 256MiB      4.88%   8.4kB / 6.2kB     0B / 524kB        5
d4e5f6a7b8c9   myapp-worker-1    15.67%  234.5MiB / 512MiB     45.80%  5.1kB / 3.2kB     0B / 0B           8
```

关键列说明如下：

| 列名 | 含义 | 关注点 |
|------|------|---------|
| CPU % | 相对于宿主机的 CPU 使用率 | 持续高负载可能成为性能瓶颈 |
| MEM USAGE / LIMIT | 当前内存使用量 / 配置上限 | 接近上限存在 OOM 风险 |
| MEM % | 占配置上限的百分比 | 超过 80% 就需警惕 |
| NET I/O | 网络收发字节数 | 异常流量可能暗示问题 |
| BLOCK I/O | 磁盘读写量 | 高写入量可能是日志泛滥所致 |
| PIDS | 进程数 | 持续增长可能意味着进程泄漏 |

```bash
# 监控特定容器（单次快照）
docker stats my-container --no-stream

# 自定义输出格式
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## docker top — 进程列表

```bash
docker top my-container
```

```text
UID                 PID                 PPID                C                   STIME               TTY                 TIME                CMD
appuser             12345               12330               0                   10:15               ?                   00:00:05            gunicorn: master [app:app]
appuser             12350               12345               2                   10:15               ?                   00:01:30            gunicorn: worker [app:app]
appuser             12351               12345               2                   10:15               ?                   00:01:28            gunicorn: worker [app:app]
appuser             12352               12345               1                   10:15               ?                   00:01:25            gunicorn: worker [app:app]
appuser             12353               12345               2                   10:15               ?                   00:01:32            gunicorn: worker [app:app]
```

注意：PID 和 PPID 列显示的是宿主机上的实际进程 ID。尽管在容器内部主进程始终是 PID 1，但在宿主机上它可能对应 PID 12345——这正是 Linux 命名空间映射机制的体现。

```bash
# 使用 ps 风格格式化输出
docker top my-container -o pid,ppid,user,%cpu,%mem,comm
```

## docker diff — 文件系统变更

`docker diff` 能清晰展示容器可写层相对于原始镜像所发生的文件增删改操作：

```bash
docker diff my-container
```

```text
C /tmp
A /tmp/cache
A /tmp/cache/session_abc123
C /var/log
A /var/log/app.log
C /app
C /app/config.yaml
A /app/data/uploads/image001.png
```

| 前缀 | 含义 |
|------|------|
| A | 新增 —— 该文件在原始镜像中不存在 |
| C | 修改 —— 该文件在镜像中存在但已被修改 |
| D | 删除 —— 该文件在镜像中存在但已被移除 |

这项功能特别适用于：
- 定位应用实际写入数据的位置（是否应改用 volume？）
- 检测是否有意外的文件修改行为
- 理解容器运行过程中累积了哪些状态

## 调试已崩溃容器

当容器启动后立即退出，你就无法使用 `docker exec` 进入。此时可尝试以下策略：

### 策略 1：查看日志

```bash
docker logs crashed-container
```

只要容器尚未被 `docker rm` 删除，此方法对任何已停止容器都有效。

### 策略 2：覆盖默认命令

如果容器在启动阶段就崩溃，可以临时覆盖其启动命令，让它保持运行状态以便调试：

```bash
# 不运行原命令，改为 sleep
docker run -it --name debug-container myapp:v2 bash

# 或完全覆盖 entrypoint
docker run -it --entrypoint bash myapp:v2

# 现在你已进入容器，可自由排查
ls /app/
cat /app/config.yaml
python -c "import psycopg2; print('module found')"
```

### 策略 3：从已停止容器中复制文件

```bash
# 创建容器但不启动
docker create --name debug-container myapp:v2

# 复制文件出来
docker cp debug-container:/app/config.yaml ./debug-config.yaml
docker cp debug-container:/var/log/ ./debug-logs/

# 清理
docker rm debug-container
```

### 策略 4：将已停止容器提交为新镜像

```bash
# 查看已崩溃容器
docker ps -a | grep crashed

# 将其当前状态保存为新镜像
docker commit crashed-container debug-image:latest

# 现在可以探索它
docker run -it debug-image:latest bash
```

## 临时调试容器（Ephemeral Debug Containers）

![容器内部调试：侦探拿着放大镜](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/06-debugging-inside-a-container-detective-with-magnifying-glass.jpg)

有时你需要 `strace`、高级网络工具等应用镜像中没有的调试组件。这时可以单独启动一个调试容器，并让它共享目标容器的命名空间：

![调试工作流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/06-debug-workflow.png)

```bash
# 运行调试容器并共享目标容器的网络命名空间
docker run -it --rm \
    --network container:my-app-container \
    nicolaka/netshoot \
    bash
```

在 `netshoot` 容器内，你就能完全访问目标容器的网络环境：

```bash
# DNS 解析（使用 my-app-container 的网络配置）
nslookup postgres

# TCP 连接测试
nc -zv postgres 5432

# HTTP 请求
curl -v http://localhost:8080/health

# 抓包
tcpdump -i eth0 -n port 5432

# 路由追踪
traceroute postgres
```

不仅如此，你还可以共享其他命名空间，比如 PID 或 IPC：

```bash
# 共享 PID 命名空间（查看目标容器的进程）
docker run -it --rm \
    --pid container:my-app-container \
    ubuntu bash -c "ps aux"
```

## 日志驱动（Log Drivers）

默认情况下，Docker 会将日志以 JSON 格式存储在本地磁盘。但你可以通过配置不同的日志驱动来改变这一行为：

```bash
# 查看当前日志驱动
docker info --format '{{.LoggingDriver}}'
# 输出: json-file
```

### 可用日志驱动

| 驱动 | 目标 | 支持 `docker logs`？ | 使用场景 |
|------|------|---------------------|----------|
| `json-file` | 本地 JSON 文件 | 是 | 默认配置，适合开发环境 |
| `local` | 优化的本地存储 | 是 | 单机生产环境 |
| `syslog` | Syslog 守护进程 | 否 | 传统 Linux 日志体系 |
| `journald` | systemd journal | 是 | 基于 systemd 的主机 |
| `fluentd` | Fluentd 收集器 | 否 | 集中式日志（如 ELK/EFK） |
| `awslogs` | CloudWatch Logs | 否 | AWS 部署 |
| `gcplogs` | Google Cloud Logging | 否 | GCP 部署 |
| `splunk` | Splunk HTTP Event Collector | 否 | 企业级环境 |
| `none` | 丢弃所有日志 | 否 | 高吞吐场景（日志由应用自行处理） |

### 按容器配置日志驱动

```bash
# 为单个容器指定日志驱动
docker run -d \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    --name my-container \
    myapp
```

### 配置日志轮转（生产环境必备）

如果不启用日志轮转，`json-file` 驱动生成的日志会无限增长，最终耗尽磁盘空间：

```bash
# Docker daemon 配置 (/etc/docker/daemon.json)
cat /etc/docker/daemon.json
```

```json
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "5",
        "compress": "true"
    }
}
```

| 选项 | 效果 | 推荐值 |
|------|------|--------|
| `max-size` | 单个日志文件最大尺寸 | 10m - 50m |
| `max-file` | 保留的轮转日志文件数量 | 3 - 5 |
| `compress` | 是否压缩已轮转的日志文件 | true |

缺少这些配置时，一个繁忙的容器可能在短时间内生成 GB 级的日志。

### 在 Docker Compose 中配置

```yaml
services:
  api:
    image: myapp
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## 常见故障模式

下表总结了最常见的容器问题及其应对方法：

| 症状 | 最可能原因 | 诊断命令 | 解决方案 |
|------|-------------|-----------|------------|
| 容器立即退出 | 应用启动时崩溃 | `docker logs container` | 检查配置与依赖项 |
| 退出码 137 | 内存不足（OOM killed） | `docker inspect -f '{{.State.OOMKilled}}'` | 增加 `--memory` 限制，或修复内存泄漏 |
| 退出码 127 | 命令未找到 | `docker inspect -f '{{json .Config.Cmd}}'` | 检查 CMD/ENTRYPOINT 拼写，确认二进制存在 |
| 退出码 126 | 命令权限拒绝 | `docker exec container ls -la /entrypoint.sh` | 用 `chmod +x` 设置入口脚本权限，检查 USER 指令 |
| “地址已在使用” | 端口冲突 | `docker ps`（检查端口映射） | 更换宿主机端口 |
| “连接被拒绝”到某服务 | 服务未就绪或主机名错误 | `docker exec container curl http://service:port` | 检查 `depends_on`、健康检查及网络配置 |
| DNS 解析失败 | 未加入自定义网络 | `docker network inspect bridge` | 使用 `docker network create` 创建自定义网络 |
| 文件权限错误 | 宿主机与容器 UID 不匹配 | `docker exec container id` + `ls -la /path` | 对齐 UID，或改用命名卷（named volumes） |
| macOS 上性能缓慢 | 绑定挂载（bind mount）I/O 开销大 | `docker stats` | 对依赖项使用命名卷，仅对源码使用绑定挂载 |
| 容器循环重启 | CrashLoopBackOff（崩溃 → 重启 → 崩溃） | `docker logs --tail 50 container` | 修复根本崩溃原因，设置 `restart: on-failure` |

## 调试工作流检查清单

当容器无法正常工作时，请按以下顺序排查：

```bash
# 1. 容器是否正在运行？
docker ps -a | grep my-container

# 2. 日志说了什么？
docker logs --tail 100 my-container

# 3. 退出码是多少？
docker inspect my-container --format '{{.State.ExitCode}} OOM:{{.State.OOMKilled}}'

# 4. 资源使用情况如何？
docker stats my-container --no-stream

# 5. 我能否进入容器？
docker exec -it my-container bash  # 或 sh

# 6. 当前运行哪些进程？
docker top my-container

# 7. 文件系统发生了哪些变更？
docker diff my-container

# 8. 完整配置是什么？
docker inspect my-container

# 9. 容器能否访问其依赖服务？
docker exec my-container ping postgres
docker exec my-container curl -v http://api:8000/health

# 10. 网络配置是否正确？
docker network inspect $(docker inspect my-container --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}')
```

## Docker Events — 全局系统活动

`docker events` 会以流式方式输出 Docker daemon 的实时事件：

```bash
docker events
```

```text
2023-09-30T10:30:15.123456789Z container start a1b2c3d4e5f6 (image=myapp:v2, name=api)
2023-09-30T10:30:45.234567890Z container die a1b2c3d4e5f6 (exitCode=1, image=myapp:v2, name=api)
2023-09-30T10:30:46.345678901Z container start a1b2c3d4e5f6 (image=myapp:v2, name=api)
2023-09-30T10:31:16.456789012Z container die a1b2c3d4e5f6 (exitCode=1, image=myapp:v2, name=api)
```

上述输出表明容器正处于重启循环（die → start → die）。你可以通过过滤条件减少噪音：

```bash
# 仅显示容器事件
docker events --filter type=container

# 仅显示特定容器的事件
docker events --filter container=my-container

# 仅显示特定事件类型
docker events --filter event=die --filter event=oom
```

## 接下来

现在，你已经掌握了在容器行为异常时定位问题的方法。但调试本质上是一种被动响应——理想情况下，我们更希望防患于未然。下一篇文章将聚焦**安全性**：如何以非 root 用户运行容器、如何降权（drop capabilities）、如何扫描漏洞，以及如何遵循最佳实践，规避容器化应用中最常见的安全失误。
