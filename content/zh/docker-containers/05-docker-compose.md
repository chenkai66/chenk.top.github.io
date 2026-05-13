---
title: "Docker 与容器（五）：Docker Compose——多容器应用"
date: 2023-06-20 09:00:00
tags:
  - Docker
  - Containers
  - Docker Compose
  - YAML
  - DevOps
categories: Docker and Containers
series: docker-containers
lang: zh
description: "真实应用绝非单个容器。Docker Compose 允许你用一个 YAML 文件声明式地定义多服务架构——网络、卷、依赖关系，一应俱全。"
disableNunjucks: true
series_order: 5
translationKey: "docker-containers-5"
---
前几篇文章介绍了如何使用 `docker run` 运行容器、用 `-p` 参数映射端口、用 `docker network create` 创建网络，以及用 `-v` 挂载卷。现在试想一下：每次启动开发环境时，都要为 Web 服务器、API 后端、数据库、缓存和任务队列分别执行这些操作——这将变得极其繁琐。Docker Compose 将原本需要 20 多条命令完成的工作，简化为一个文件和一条命令：`docker compose up`。

## 为什么需要 Compose

典型的 Web 应用通常由多个服务组成：

![Compose 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-compose-architecture.png)

- 前端（React、Vue 或服务端渲染的 HTML）
- 后端 API（Python、Node.js、Go）
- 数据库（PostgreSQL、MySQL）
- 缓存（Redis、Memcached）
- 任务队列（Celery、Bull）
- 可能还有反向代理（nginx）

每个服务都运行在独立的容器中。若不使用 Compose，启动整套栈意味着要依次运行 `docker network create`、`docker volume create`，再为每个服务单独执行带十几个参数的 `docker run`——至少五条命令，极易出错，且无法以清晰、可版本控制的方式管理。而 Compose 将这一切转化为一份声明式配置文件。

## `docker-compose.yml` 基础

以下是上述手动命令对应的 Compose 等价写法：

![健康检查流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-healthcheck-flow.png)

```yaml
services:
  postgres:
    image: postgres:16
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: myapp
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://postgres:secret@postgres:5432/myapp
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  worker:
    build: ./worker
    environment:
      DATABASE_URL: postgresql://postgres:secret@postgres:5432/myapp
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api

volumes:
  postgres-data:
  redis-data:
```

一个文件，一条启动命令：`docker compose up`；一条停止命令：`docker compose down`。

### Compose 自动完成的工作

当你运行 `docker compose up` 时，Compose 会自动：

1. 为所有服务创建一个专用网络（默认命名为 `<project>_default`）
2. 创建 `volumes:` 部分中声明的命名卷
3. 构建含 `build:` 指令的服务镜像
4. 按依赖顺序启动容器
5. 将服务名作为 DNS 主机名注册（因此 `api` 可直接通过 `postgres` 名称访问数据库）
6. 将所有日志流式输出到终端（除非使用 `-d` 参数后台运行）

项目名默认取自当前目录名。例如 `~/projects/myapp/` 下的项目，会创建名为 `myapp_default` 的网络，容器名为 `myapp-postgres-1`、`myapp-redis-1` 等。

## Compose 文件结构

![微服务生态系统中的互联容器](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/05-microservices-ecosystem-interconnected-containers-forming-a-.jpg)

一个 Compose 文件包含四个顶层键：

![Compose 网络](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-compose-networking.png)

```yaml
services:    # 容器定义（必需）
  web:
    image: nginx
    # ...

networks:    # 自定义网络（可选 — 默认已创建）
  frontend:
  backend:

volumes:     # 命名卷（可选）
  db-data:
  cache-data:

configs:     # 配置文件（仅 Swarm 模式）
  my-config:
    file: ./config.ini

secrets:     # 敏感数据（仅 Swarm 模式）
  db-password:
    file: ./db_password.txt
```

注意：Compose 文件顶部的 `version:` 字段已在 Docker Compose v2 中弃用，可完全省略。旧示例中出现的 `version: "3.8"` 虽无害，但已无必要。

### 服务配置选项详解

每个服务支持大量配置项。以下是最常用的选项：

```yaml
services:
  myservice:
    # 使用的镜像（与 build 互斥）
    image: python:3.11-slim

    # 从 Dockerfile 构建（与 image 互斥）
    build:
      context: ./app           # 构建上下文目录
      dockerfile: Dockerfile   # Dockerfile 名称（默认为 Dockerfile）
      args:                    # 构建参数
        APP_VERSION: "2.0"
      target: production       # 多阶段构建目标

    # 容器名称（默认为 <project>-<service>-<number>）
    container_name: my-custom-name

    # 覆盖默认 command
    command: gunicorn --bind 0.0.0.0:8000 app:app

    # 覆盖 entrypoint
    entrypoint: /app/entrypoint.sh

    # 端口映射
    ports:
      - "8080:80"              # host:container
      - "127.0.0.1:3000:3000"  # 仅绑定到 localhost
      - "9090-9099:8080-8089"  # 端口范围映射

    # 卷挂载
    volumes:
      - db-data:/var/lib/mysql          # 命名卷
      - ./src:/app/src                  # 绑定挂载（相对路径）
      - /host/path:/container/path:ro   # 只读绑定挂载
      - type: tmpfs                     # tmpfs 挂载
        target: /tmp

    # 环境变量
    environment:
      NODE_ENV: production
      DB_HOST: postgres
      # 无值变量 = 从宿主机 shell 继承
      API_KEY:

    # 环境变量文件
    env_file:
      - .env
      - .env.production

    # 服务依赖
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

    # 加入的网络
    networks:
      - frontend
      - backend

    # 重启策略
    restart: unless-stopped    # always, on-failure, no

    # 资源限制（需 deploy 块）
    deploy:
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

    # 健康检查
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

    # 日志配置
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## 完整示例：Python Web 应用 + PostgreSQL + Redis

![Docker Compose 编排指挥多个服务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/05-docker-compose-orchestra-conductor-directing-multiple-servic.jpg)

我们来构建一个完整的应用栈：API 通过 HTTP 接收任务，将其存入 PostgreSQL 并推入 Redis 队列；Worker 异步处理任务。项目结构如下：

```text
myapp/
  docker-compose.yml
  .env
  init.sql
  api/          # Flask 应用（含 Dockerfile）
  worker/       # 后台 Worker（含 Dockerfile）
```

### Compose 文件

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  api:
    build:
      context: ./api
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@postgres:5432/${DB_NAME}
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  worker:
    build:
      context: ./worker
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@postgres:5432/${DB_NAME}
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
```

### 环境变量文件

```bash
# .env（Compose 自动加载）
DB_PASSWORD=supersecret
DB_NAME=myapp
```

`init.sql` 在首次启动时创建 `tasks` 表。当 PostgreSQL 的数据卷为空时，它会自动执行 `/docker-entrypoint-initdb.d/` 目录下的脚本。

## 服务依赖关系

### `depends_on`

`depends_on` 控制服务启动顺序，但需严格区分“已启动”与“已就绪”：

![服务依赖关系](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-service-dependency.png)

```yaml
services:
  api:
    depends_on:
      # 简单形式：等待容器启动（未必就绪）
      redis:
        condition: service_started

      # 健康检查形式：等待容器健康（即服务就绪）
      postgres:
        condition: service_healthy
```

若未配置健康检查，`depends_on` 仅确保容器进程已启动，并不等待其内部服务真正就绪。PostgreSQL 初始化需数秒；若 API 在 PostgreSQL 尚未接受连接时启动，将直接崩溃。

✅ **务必对数据库等有初始化耗时的服务，配合 `condition: service_healthy` 使用 `depends_on`**。这要求被依赖服务必须定义 `healthcheck`。

### Wait-for-It 模式

有时健康检查仍不够可靠，你的应用可能需要主动重试：

```yaml
services:
  api:
    build: ./api
    command: >
      sh -c "
        while ! nc -z postgres 5432; do
          echo 'Waiting for PostgreSQL...'
          sleep 1
        done
        echo 'PostgreSQL is ready'
        gunicorn --bind 0.0.0.0:8000 app:app
      "
    depends_on:
      - postgres
```

或使用专用等待脚本：

```bash
#!/bin/sh
# wait-for-it.sh
set -e

host="$1"
port="$2"
shift 2
cmd="$@"

until nc -z "$host" "$port"; do
  echo "Waiting for $host:$port..."
  sleep 1
done

echo "$host:$port is available, executing command"
exec $cmd
```

```yaml
services:
  api:
    command: ["./wait-for-it.sh", "postgres", "5432", "--", "gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

## 环境变量注入方式

Compose 支持多种环境变量注入机制：

### 直接写在 Compose 文件中

```yaml
services:
  api:
    environment:
      NODE_ENV: production
      API_PORT: "8000"
```

### 从 `.env` 文件加载（自动）

Compose 会自动加载与 Compose 文件同目录下的 `.env` 文件。其中变量可用于 Compose 文件内的插值：

```bash
# .env
DB_PASSWORD=mysecret
DB_NAME=production
```

```yaml
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}  # 从 .env 插值
      POSTGRES_DB: ${DB_NAME}
```

### 通过 `env_file` 加载（注入到容器内）

```yaml
services:
  api:
    env_file:
      - .env           # 默认变量
      - .env.local      # 本地覆盖（不提交至 Git）
```

`.env`（自动加载）与 `env_file:` 的用途截然不同：

| 特性 | `.env`（自动加载） | `env_file:` 指令 |
|------|-------------------|------------------|
| 用途 | Compose 文件内变量替换 | 向容器注入变量 |
| 加载方 | `docker compose` CLI | 容器运行时 |
| 语法 | `KEY=VALUE` | `KEY=VALUE` |
| 在 Compose 文件中可用 | 是（`${VAR}`） \|否 \|
| 在容器内可用 | 仅当同时出现在 `environment:` 中 | 是 |

### 默认值与必需变量

```yaml
services:
  api:
    environment:
      # 未设置时使用默认值
      LOG_LEVEL: ${LOG_LEVEL:-info}

      # 必须设置，否则报错
      API_KEY: ${API_KEY:?API_KEY 必须设置}

      # 未设置时为空字符串
      OPTIONAL: ${OPTIONAL:-}
```

## 核心 Compose 命令速查

```bash
# 启动所有服务（前台模式，显示日志）
docker compose up

# 后台启动（守护进程模式）
docker compose up -d

# 仅启动指定服务
docker compose up -d postgres redis

# 启动前强制重新构建镜像
docker compose up -d --build

# 查看运行中的服务
docker compose ps
```

```text
NAME                  IMAGE                  COMMAND                  SERVICE    CREATED          STATUS                    PORTS
myapp-api-1           myapp-api              "gunicorn --bind 0.0…"   api        2 minutes ago    Up 2 minutes (healthy)    0.0.0.0:8000->8000/tcp
myapp-postgres-1      postgres:16-alpine     "docker-entrypoint.s…"   postgres   2 minutes ago    Up 2 minutes (healthy)    0.0.0.0:5432->5432/tcp
myapp-redis-1         redis:7-alpine         "docker-entrypoint.s…"   redis      2 minutes ago    Up 2 minutes (healthy)    0.0.0.0:6379->6379/tcp
myapp-worker-1        myapp-worker           "python worker.py"       worker     2 minutes ago    Up 2 minutes              
```

```bash
# 查看某服务日志（实时跟踪）
docker compose logs -f api

# 在运行中的服务内执行命令
docker compose exec postgres psql -U postgres -d myapp

# 运行一次性命令（新建临时容器）
docker compose run --rm api python manage.py migrate

# 停止所有服务（容器保留，网络保留）
docker compose stop

# 停止并删除容器、网络
docker compose down

# 停止、删除容器、网络及卷（⚠️破坏性操作）
docker compose down -v

# 仅重建某服务镜像
docker compose build api
```

## 开发 vs 生产：覆盖文件（Override Files）

Compose 支持覆盖文件，可对基础配置进行分层叠加。默认情况下，`docker-compose.override.yml` 会自动叠加在 `docker-compose.yml` 之上：

```yaml
# docker-compose.yml — 基础配置（类生产环境）
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      NODE_ENV: production
    restart: always

  postgres:
    image: postgres:16-alpine
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

```yaml
# docker-compose.override.yml — 开发环境覆盖（自动加载）
services:
  api:
    build:
      context: ./api
      target: development    # 使用多阶段构建的 dev 阶段
    volumes:
      - ./api:/app           # 实时代码热重载
    environment:
      NODE_ENV: development
      DEBUG: "true"
    restart: "no"            # 开发时崩溃不自动重启

  postgres:
    ports:
      - "5432:5432"          # 暴露 DB 端口供本地工具连接
    environment:
      POSTGRES_PASSWORD: devpassword
```

运行 `docker compose up` 时，两份文件自动合并，覆盖文件中的值优先级更高。

如需显式指定环境配置文件，使用 `-f` 参数：

```bash
# 开发环境（默认：docker-compose.yml + docker-compose.override.yml）
docker compose up

# 生产环境（显式指定，跳过 override）
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 测试环境
docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit
```

## 服务扩缩容（Scaling）

Compose 支持对单个服务运行多个实例：

![扩展模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-scaling-pattern.png)

```bash
# 将 worker 扩容至 3 个实例
docker compose up -d --scale worker=3

# 查看结果
docker compose ps
```

```text
NAME                  IMAGE                  SERVICE    STATUS          PORTS
myapp-api-1           myapp-api              api        Up (healthy)    0.0.0.0:8000->8000/tcp
myapp-postgres-1      postgres:16-alpine     postgres   Up (healthy)    0.0.0.0:5432->5432/tcp
myapp-redis-1         redis:7-alpine         redis      Up (healthy)    0.0.0.0:6379->6379/tcp
myapp-worker-1        myapp-worker           worker     Up
myapp-worker-2        myapp-worker           worker     Up
myapp-worker-3        myapp-worker           worker     Up
```

三个 Worker 容器共享同一个 Redis 队列，实现基本的水平扩展。

⚠️ 扩容限制：
- **不可扩容含固定端口映射的服务**（两个容器无法同时绑定 `8000` 端口）
- 对于需暴露端口的服务，应使用端口范围或前置反向代理

```yaml
# 此方案支持扩容：
services:
  worker:
    build: ./worker
    # 无端口映射 — Worker 不接收外部连接

  api:
    build: ./api
    # ❌ 错误：ports: ["8000:8000"]（无法扩容）
    # ✅ 正确：使用反向代理前置，仅内部暴露
    expose:
      - "8000"    # 仅限 Docker 网络内访问
```

## Compose Profiles（服务分组）

Profiles 允许定义可选服务，仅在显式启用时才启动。

```yaml
services:
  api:
    build: ./api
    # 无 profile — 总是启动

  adminer:
    image: adminer
    ports:
      - "8081:8080"
    profiles:
      - debug        # 仅当指定 --profile debug 时启动

  grafana:
    image: grafana/grafana
    profiles:
      - monitoring
```

```bash
# 仅启动核心服务
docker compose up -d

# 启动核心 + 调试工具
docker compose --profile debug up -d

# 启动全部服务
docker compose --profile debug --profile monitoring up -d
```

## Compose Watch（开发热重载）

Docker Compose Watch 可自动将文件变更同步至运行中的容器：

```yaml
services:
  api:
    build: ./api
    develop:
      watch:
        - action: sync           # 将变更文件复制进容器
          path: ./api/src
          target: /app/src
        - action: rebuild         # 依赖文件变更时触发重建
          path: ./api/requirements.txt
```

```bash
docker compose watch
```

- `sync` 动作：复制文件，无需重建镜像
- `rebuild` 动作：依赖文件（如 `requirements.txt`）变更时触发完整镜像重建
- 第三种动作 `sync+restart`：复制文件后重启容器（适用于配置变更）

## 快速参考表

| 命令 | 作用 |
|------|------|
| `docker compose up -d` | 后台启动所有服务 |
| `docker compose down` | 停止并移除容器与网络 |
| `docker compose down -v` | 同时移除卷（⚠️数据丢失） |
| `docker compose ps` | 列出运行中的服务 |
| `docker compose logs -f SERVICE` | 实时跟踪某服务日志 |
| `docker compose exec SERVICE cmd` | 在运行中的容器内执行命令 |
| `docker compose run --rm SERVICE cmd` | 在新容器中运行一次性命令 |
| `docker compose build` | 重建所有镜像 |
| `docker compose up -d --scale SVC=N` | 将服务 `SVC` 扩容至 `N` 个实例 |
| `docker compose config` | 验证并输出最终合并后的配置 |

## 下一步

Docker Compose 很好地解决了“理想路径”问题：定义服务、一键启动、一切正常。但现实是应用总会出错——容器静默崩溃、日志消失于无形、“在我机器上能跑”问题重现……当你无法深入容器内部时，问题便无从排查。下一篇文章将聚焦**调试与日志**：当容器拒绝配合时，如何快速定位故障根源？
