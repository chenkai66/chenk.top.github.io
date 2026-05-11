---
title: "Docker and Containers (5): Docker Compose — Multi-Container Applications"
date: 2023-06-24 09:00:00
tags:
  - Docker
  - Containers
  - Docker Compose
  - YAML
  - DevOps
categories:
  - Docker and Containers
series: docker-containers
lang: en
description: "Real applications aren't single containers. Docker Compose lets you define multi-service architectures in a single YAML file — networks, volumes, dependencies, and all."
disableNunjucks: true
series_order: 5
translationKey: "docker-containers-5"
---

The previous articles taught you how to run containers with `docker run`, pass port mappings with `-p`, create networks with `docker network create`, and mount volumes with `-v`. Now imagine doing that for a web server, an API backend, a database, a cache, and a task queue — every time you start your development environment. Docker Compose replaces those 20+ commands with a single file and a single command: `docker compose up`.

## Why Compose Exists

A typical web application has multiple services:

![Compose architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-compose-architecture.png)


- A frontend (React, Vue, or server-rendered HTML)
- A backend API (Python, Node.js, Go)
- A database (PostgreSQL, MySQL)
- A cache (Redis, Memcached)
- A task queue (Celery, Bull)
- Maybe a reverse proxy (nginx)

Each service is its own container. Without Compose, starting this stack means running `docker network create`, `docker volume create`, and a separate `docker run` with a dozen flags for each service — five or more commands, easy to get wrong, impossible to version control cleanly. Compose turns this into a declarative file.

## docker-compose.yml Basics

Here's the Compose equivalent of the commands above:

![Health check flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-healthcheck-flow.png)


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

One file. One command to start: `docker compose up`. One command to stop: `docker compose down`.

### What Compose Does Automatically

When you run `docker compose up`, Compose:

1. Creates a dedicated network (named `<project>_default`) for all services
2. Creates named volumes declared in the `volumes:` section
3. Builds images for services with `build:` directives
4. Starts containers in dependency order
5. Attaches service names as DNS hostnames (so `api` can reach `postgres` by name)
6. Streams all logs to your terminal (unless `-d` is used)

The project name defaults to the directory name. A project in `~/projects/myapp/` creates a network called `myapp_default`, containers named `myapp-postgres-1`, `myapp-redis-1`, etc.

## Compose File Structure


![Microservices ecosystem interconnected containers forming a](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/05-microservices-ecosystem-interconnected-containers-forming-a-.jpg)

A compose file has four top-level keys:

![Compose networking](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-compose-networking.png)


```yaml
services:    # Container definitions (required)
  web:
    image: nginx
    # ...

networks:    # Custom networks (optional — a default is created)
  frontend:
  backend:

volumes:     # Named volumes (optional)
  db-data:
  cache-data:

configs:     # Configuration files (Swarm mode)
  my-config:
    file: ./config.ini

secrets:     # Sensitive data (Swarm mode)
  db-password:
    file: ./db_password.txt
```

Note: The `version:` key at the top of compose files is deprecated since Docker Compose v2. You can omit it entirely. If you see `version: "3.8"` in old examples, it's harmless but unnecessary.

### Service Configuration Options

Each service supports many options. Here are the most common:

```yaml
services:
  myservice:
    # Image to use (mutually exclusive with build)
    image: python:3.11-slim

    # Build from a Dockerfile (mutually exclusive with image)
    build:
      context: ./app           # Build context directory
      dockerfile: Dockerfile   # Dockerfile name (default: Dockerfile)
      args:                    # Build arguments
        APP_VERSION: "2.0"
      target: production       # Multi-stage build target

    # Container name (default: <project>-<service>-<number>)
    container_name: my-custom-name

    # Override the default command
    command: gunicorn --bind 0.0.0.0:8000 app:app

    # Override the entrypoint
    entrypoint: /app/entrypoint.sh

    # Port mappings
    ports:
      - "8080:80"              # host:container
      - "127.0.0.1:3000:3000"  # bind to localhost only
      - "9090-9099:8080-8089"  # port range

    # Volume mounts
    volumes:
      - db-data:/var/lib/mysql          # Named volume
      - ./src:/app/src                  # Bind mount (relative path)
      - /host/path:/container/path:ro   # Read-only bind mount
      - type: tmpfs                     # tmpfs mount
        target: /tmp

    # Environment variables
    environment:
      NODE_ENV: production
      DB_HOST: postgres
      # Variable without value = pass from host shell
      API_KEY:

    # Environment file
    env_file:
      - .env
      - .env.production

    # Service dependencies
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

    # Networks to join
    networks:
      - frontend
      - backend

    # Restart policy
    restart: unless-stopped    # always, on-failure, no

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

    # Healthcheck
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

    # Logging
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## Full Example: Python Web App + PostgreSQL + Redis


![Docker compose orchestra conductor directing multiple servic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/05-docker-compose-orchestra-conductor-directing-multiple-servic.jpg)

Let's build a complete application stack. The API accepts tasks via HTTP, stores them in PostgreSQL, and queues them in Redis. A worker processes tasks asynchronously. The project structure:

```
myapp/
  docker-compose.yml
  .env
  init.sql
  api/          # Flask app with Dockerfile
  worker/       # Background worker with Dockerfile
```

### The Compose File

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

### The Environment File

```bash
# .env (auto-loaded by Compose)
DB_PASSWORD=supersecret
DB_NAME=myapp
```

The `init.sql` creates the tasks table on first start. PostgreSQL automatically runs scripts in `/docker-entrypoint-initdb.d/` when the data volume is empty.

## Service Dependencies


![Service dependencies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-service-dependency.png)

### depends_on

The `depends_on` key controls startup order. But there's a crucial distinction between "started" and "ready":

```yaml
services:
  api:
    depends_on:
      # Simple form: wait for container to start (not necessarily ready)
      redis:
        condition: service_started

      # Healthcheck form: wait for container to be healthy
      postgres:
        condition: service_healthy
```

Without healthchecks, `depends_on` only ensures the container has started — it doesn't wait for the service inside to be ready. PostgreSQL takes several seconds to initialize. If your API starts before PostgreSQL is accepting connections, it crashes.

**Always combine `depends_on` with `condition: service_healthy` for databases and other services that have initialization time.** This requires a `healthcheck` on the dependency.

### Wait-for-It Patterns

Sometimes healthchecks aren't enough. You might need your application to actively retry:

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

Or use a dedicated wait script:

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

## Environment Variables

Compose supports several ways to inject environment variables:

### Direct in compose file

```yaml
services:
  api:
    environment:
      NODE_ENV: production
      API_PORT: "8000"
```

### From .env file (auto-loaded)

Compose automatically loads `.env` from the same directory as the compose file. Variables are available for interpolation in the compose file:

```bash
# .env
DB_PASSWORD=mysecret
DB_NAME=production
```

```yaml
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}  # Interpolated from .env
      POSTGRES_DB: ${DB_NAME}
```

### From env_file (loaded into container)

```yaml
services:
  api:
    env_file:
      - .env           # Default variables
      - .env.local      # Local overrides (not in git)
```

The `.env` file interpolation and `env_file` serve different purposes:

| Feature | `.env` (auto-loaded) | `env_file:` directive |
|---------|---------------------|----------------------|
| Purpose | Variable substitution in compose file | Inject variables into container |
| Loaded by | `docker compose` CLI | Container runtime |
| Syntax | `KEY=VALUE` | `KEY=VALUE` |
| Available in compose file | Yes (`${VAR}`) | No |
| Available in container | Only if also in `environment:` | Yes |

### Default values and required variables

```yaml
services:
  api:
    environment:
      # Default value if not set
      LOG_LEVEL: ${LOG_LEVEL:-info}

      # Error if not set
      API_KEY: ${API_KEY:?API_KEY must be set}

      # Empty string if not set
      OPTIONAL: ${OPTIONAL:-}
```

## Essential Compose Commands

```bash
# Start all services (foreground — shows logs)
docker compose up

# Start in detached mode (background)
docker compose up -d

# Start specific services only
docker compose up -d postgres redis

# Build images before starting
docker compose up -d --build

# View running services
docker compose ps
```

```
NAME                  IMAGE                  COMMAND                  SERVICE    CREATED          STATUS                    PORTS
myapp-api-1           myapp-api              "gunicorn --bind 0.0…"   api        2 minutes ago    Up 2 minutes (healthy)    0.0.0.0:8000->8000/tcp
myapp-postgres-1      postgres:16-alpine     "docker-entrypoint.s…"   postgres   2 minutes ago    Up 2 minutes (healthy)    0.0.0.0:5432->5432/tcp
myapp-redis-1         redis:7-alpine         "docker-entrypoint.s…"   redis      2 minutes ago    Up 2 minutes (healthy)    0.0.0.0:6379->6379/tcp
myapp-worker-1        myapp-worker           "python worker.py"       worker     2 minutes ago    Up 2 minutes              
```

```bash
# View logs for a specific service (follow mode)
docker compose logs -f api

# Execute a command in a running service
docker compose exec postgres psql -U postgres -d myapp

# Run a one-off command (creates a new container)
docker compose run --rm api python manage.py migrate

# Stop all services (containers stopped, not removed)
docker compose stop

# Stop and remove containers, networks
docker compose down

# Stop, remove containers, networks, AND volumes (DESTRUCTIVE)
docker compose down -v

# Rebuild a specific service's image
docker compose build api
```

## Override Files for Dev vs Production

Compose supports override files that layer on top of the base file. By default, `docker-compose.override.yml` is automatically applied on top of `docker-compose.yml`:

```yaml
# docker-compose.yml — base configuration (production-like)
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
# docker-compose.override.yml — development overrides (auto-loaded)
services:
  api:
    build:
      context: ./api
      target: development    # Use dev stage of multi-stage build
    volumes:
      - ./api:/app           # Live code reload
    environment:
      NODE_ENV: development
      DEBUG: "true"
    restart: "no"            # Don't restart on crash during dev

  postgres:
    ports:
      - "5432:5432"          # Expose DB port for local tools
    environment:
      POSTGRES_PASSWORD: devpassword
```

When you run `docker compose up`, both files are merged. The override values take precedence.

For explicit environment files, use `-f`:

```bash
# Development (default: docker-compose.yml + docker-compose.override.yml)
docker compose up

# Production (explicit files, no override)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Testing
docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit
```

## Scaling Services

Compose can run multiple instances of a service:

![Scaling pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/05-scaling-pattern.png)


```bash
# Scale the worker service to 3 instances
docker compose up -d --scale worker=3

# Check the result
docker compose ps
```

```
NAME                  IMAGE                  SERVICE    STATUS          PORTS
myapp-api-1           myapp-api              api        Up (healthy)    0.0.0.0:8000->8000/tcp
myapp-postgres-1      postgres:16-alpine     postgres   Up (healthy)    0.0.0.0:5432->5432/tcp
myapp-redis-1         redis:7-alpine         redis      Up (healthy)    0.0.0.0:6379->6379/tcp
myapp-worker-1        myapp-worker           worker     Up
myapp-worker-2        myapp-worker           worker     Up
myapp-worker-3        myapp-worker           worker     Up
```

Three worker containers, each pulling from the same Redis queue. This is basic horizontal scaling.

Scaling has limitations:
- You can't scale services with fixed port mappings (two containers can't both bind to port 8000)
- For services with ports, use a port range or a reverse proxy

```yaml
# This works with scaling:
services:
  worker:
    build: ./worker
    # No port mapping — workers don't accept connections

  api:
    build: ./api
    # Don't do: ports: ["8000:8000"] (can't scale)
    # Instead, use a reverse proxy in front
    expose:
      - "8000"    # Internal only — accessible within Docker network
```

## Compose Profiles

Profiles let you define optional services that only start when explicitly activated:

```yaml
services:
  api:
    build: ./api
    # No profile — always starts

  adminer:
    image: adminer
    ports:
      - "8081:8080"
    profiles:
      - debug        # Only starts with --profile debug

  grafana:
    image: grafana/grafana
    profiles:
      - monitoring
```

```bash
# Start only core services
docker compose up -d

# Start core + debug tools
docker compose --profile debug up -d

# Start everything
docker compose --profile debug --profile monitoring up -d
```

## Compose Watch (Development Live Reload)

Docker Compose Watch automatically syncs file changes to running containers:

```yaml
services:
  api:
    build: ./api
    develop:
      watch:
        - action: sync           # Copy changed files into container
          path: ./api/src
          target: /app/src
        - action: rebuild         # Rebuild on dependency changes
          path: ./api/requirements.txt
```

```bash
docker compose watch
```

The `sync` action copies files without rebuilding. The `rebuild` action triggers a full image rebuild when dependency files change. A third action, `sync+restart`, copies files and restarts the container (useful for configuration changes).

## Quick Reference

| Command | Effect |
|---------|--------|
| `docker compose up -d` | Start services in background |
| `docker compose down` | Stop and remove containers/networks |
| `docker compose down -v` | Also remove volumes (data loss) |
| `docker compose ps` | List running services |
| `docker compose logs -f SERVICE` | Follow logs for a service |
| `docker compose exec SERVICE cmd` | Run command in running container |
| `docker compose run --rm SERVICE cmd` | Run one-off command in new container |
| `docker compose build` | Rebuild all images |
| `docker compose up -d --scale SVC=N` | Scale a service to N instances |
| `docker compose config` | Validate and display merged config |

## What's Next

Docker Compose handles the happy path well — define services, start them, they work. But applications break. Containers crash silently, logs vanish into the ether, and that "works on my machine" problem returns when you can't see inside the container. The next article covers debugging and logging: how to figure out what went wrong when a container refuses to cooperate.
