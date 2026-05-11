---
title: "Docker and Containers (7): Security — Running Containers Without Giving Away the Keys"
date: 2023-06-28 09:00:00
tags:
  - Docker
  - Containers
  - Security
  - DevSecOps
  - Best Practices
categories:
  - Docker and Containers
series: docker-containers
lang: en
description: "Containers provide isolation, not security. Default Docker configurations run processes as root with full capabilities. This article shows how to lock containers down for production."
disableNunjucks: true
series_order: 7
translationKey: "docker-containers-7"
---

Docker's default configuration prioritizes convenience over security. Out of the box, containers run as root, have access to a broad set of Linux capabilities, and can write to their entire filesystem. This is fine for development — dangerous for production. A container escape vulnerability combined with a root-privileged container means an attacker owns the host. Let's fix that.

## The Threat Model

Before locking things down, understand what you're defending against:

![Rootless containers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-rootless-container.png)


1. **Vulnerable application code**: Your app has a bug (RCE, path traversal, SSRF) and an attacker gets code execution inside the container
2. **Vulnerable dependencies**: A library in your image has a known CVE
3. **Container escape**: An attacker exploits a kernel or runtime vulnerability to break out of the container
4. **Supply chain attack**: A malicious base image or package is used
5. **Secrets exposure**: Credentials leak through environment variables, image history, or logs
6. **Lateral movement**: An attacker in one container pivots to other containers or the host

Each hardening technique addresses one or more of these threats. The goal is defense in depth — no single measure is sufficient, but layers of hardening make exploitation significantly harder.

## Running as Non-Root

By default, the process inside a Docker container runs as root (UID 0). This root is the same root as on the host (unless user namespaces are enabled). If an attacker escapes the container, they're root on the host.

### In the Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies as root (needed for system packages)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy application files and set ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user for all subsequent instructions and runtime
USER appuser

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

On Alpine-based images, the syntax is slightly different:

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

### At Runtime

Even if the Dockerfile doesn't set a user, you can override at runtime:

```bash
# Run as a specific UID:GID
docker run --user 1000:1000 myapp

# Run as the "nobody" user
docker run --user nobody myapp
```

### Verifying the user

```bash
# Check what user the container is running as
docker exec my-container id
# Output: uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)

# Compare with a default container
docker exec default-container id
# Output: uid=0(root) gid=0(root) groups=0(root)
```

### Common non-root gotchas

Running as non-root can break things that assume root access:

| Problem | Symptom | Solution |
|---------|---------|----------|
| Can't bind to ports < 1024 | Permission denied on port 80 | Use port 8080+ and map with `-p 80:8080` |
| Can't write to directories | Permission denied on /var/log | `RUN mkdir -p /var/log/app && chown appuser /var/log/app` |
| Can't install packages at runtime | apt-get fails | Install everything in build stage before `USER` |
| Can't read mounted files | Permission denied on volumes | Match UID/GID with host, or use named volumes |
| Package managers need root | npm/pip fail | Use `--user` flag for pip, or install before switching user |

## Read-Only Filesystem

A read-only root filesystem prevents an attacker from modifying binaries, planting malware, or changing configuration files:

```bash
# Run with read-only filesystem
docker run --read-only myapp
```

Most applications need to write to some locations (temp files, caches, pid files). Use tmpfs for writable areas:

```bash
# Read-only root with writable /tmp and /var/run
docker run --read-only \
    --tmpfs /tmp:size=100m \
    --tmpfs /var/run:size=1m \
    myapp
```

In Docker Compose:

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

Test your application with `--read-only` in development. If it crashes, the error message will tell you which path it tried to write to — then add a tmpfs for that path.

```bash
# Find where your app writes
docker run --read-only myapp 2>&1 | grep "Read-only file system"
# Output: OSError: [Errno 30] Read-only file system: '/app/logs/app.log'
# Solution: Add --tmpfs /app/logs:size=50m
```

## Linux Capabilities

Linux capabilities divide root's power into ~40 individual privileges. By default, Docker grants containers a subset of these — more than most applications need.

![Linux capability management](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-capability-model.png)


Default capabilities given to Docker containers:

| Capability | Permission | Needed? |
|-----------|-----------|---------|
| `CHOWN` | Change file ownership | Rarely |
| `DAC_OVERRIDE` | Bypass file permission checks | Rarely |
| `FSETID` | Set SUID/SGID bits | Almost never |
| `FOWNER` | Bypass permission checks on file owner | Rarely |
| `MKNOD` | Create special files | Almost never |
| `NET_RAW` | Use raw sockets (ping, packet capture) | Sometimes |
| `SETGID` | Set group ID | Sometimes (init scripts) |
| `SETUID` | Set user ID | Sometimes (init scripts) |
| `SETFCAP` | Set file capabilities | Almost never |
| `SETPCAP` | Set process capabilities | Almost never |
| `NET_BIND_SERVICE` | Bind to ports < 1024 | Only for port 80/443 |
| `SYS_CHROOT` | Use chroot | Almost never |
| `KILL` | Send signals to other processes | Sometimes |
| `AUDIT_WRITE` | Write to kernel audit log | Rarely |

The principle of least privilege: drop all capabilities, then add back only what your application needs.

```bash
# Drop ALL capabilities, add back only what's needed
docker run \
    --cap-drop ALL \
    --cap-add NET_BIND_SERVICE \
    myapp

# A web server that needs to bind to port 80
docker run \
    --cap-drop ALL \
    --cap-add NET_BIND_SERVICE \
    -p 80:80 \
    nginx

# Most applications need nothing
docker run \
    --cap-drop ALL \
    myapp
```

In Docker Compose:

```yaml
services:
  api:
    image: myapp
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

### Checking capabilities

```bash
# See what capabilities a running container has
docker exec my-container cat /proc/1/status | grep Cap

# Decode the hex capability mask
docker exec my-container capsh --decode=00000000a80425fb
```

## Secrets Management

Secrets (API keys, database passwords, TLS certificates) are one of the most common security failures in containerized applications.

### How NOT to handle secrets

```dockerfile
# NEVER: Secrets in build arguments (stored in image history)
ARG DB_PASSWORD=supersecret
RUN echo "password=$DB_PASSWORD" >> /app/config

# NEVER: Secrets in environment variables in the Dockerfile
ENV API_KEY=sk-12345abcde

# NEVER: Secrets COPYed into the image
COPY credentials.json /app/credentials.json
```

All three of these are visible to anyone who has access to the image:

```bash
# Build args are visible in history
docker history myapp
# Shows: ARG DB_PASSWORD=supersecret

# Environment variables are visible in inspect
docker inspect myapp --format '{{json .Config.Env}}'
# Shows: ["API_KEY=sk-12345abcde"]

# Files are extractable from the image
docker create --name extract myapp
docker cp extract:/app/credentials.json .
```

### Environment variables (acceptable for many use cases)

Environment variables at runtime (not in the Dockerfile) are the most common approach:

```bash
docker run -e DB_PASSWORD=secret -e API_KEY=sk-12345 myapp
```

Or with a file:

```bash
docker run --env-file .env myapp
```

The `.env` file should never be committed to version control (add it to `.gitignore`).

**Risks of environment variables:**
- Visible via `docker inspect`
- Available to all processes in the container (including child processes)
- Can be logged accidentally (`env | sort` in debug output, error reporters)
- Visible in `/proc/<pid>/environ` inside the container

### Docker BuildKit secrets (for build-time secrets)

BuildKit can mount secrets during the build without storing them in any layer:

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Mount secret at build time — never stored in a layer
RUN --mount=type=secret,id=pip_extra_index \
    pip install --no-cache-dir \
    --extra-index-url $(cat /run/secrets/pip_extra_index) \
    -r requirements.txt
```

```bash
# Build with the secret
DOCKER_BUILDKIT=1 docker build \
    --secret id=pip_extra_index,src=./pip_index_url.txt \
    -t myapp .
```

The secret is available during the `RUN` instruction but is not stored in the image or any layer.

### Docker Swarm secrets (for runtime secrets)

If you use Docker Swarm, secrets are first-class:

```bash
# Create a secret
echo "supersecretpassword" | docker secret create db_password -

# Use it in a service
docker service create \
    --name api \
    --secret db_password \
    myapp
```

Inside the container, the secret is available as a file at `/run/secrets/db_password`. This is more secure than environment variables because:
- It's a tmpfs mount (never written to disk)
- Only available to services that explicitly request it
- Can be rotated without restarting the service

### Files mounted at runtime

For non-Swarm deployments, you can achieve similar security with bind mounts:

```bash
docker run \
    -v /secure/path/credentials.json:/run/secrets/credentials.json:ro \
    myapp
```

The `:ro` flag makes it read-only. Combine with `--tmpfs /tmp` and `--read-only` to prevent the secret from being copied elsewhere in the container.

## Image Scanning with Trivy

Trivy is a vulnerability scanner that checks container images against known CVE databases:

![Image vulnerability scanning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-image-scanning.png)


```bash
# Install Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scan an image
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

Trivy scans both OS packages and application dependencies (pip, npm, gem, etc.).

```bash
# Scan only for CRITICAL and HIGH severity
trivy image --severity CRITICAL,HIGH myapp:latest

# Fail if any vulnerability is found (useful in CI)
trivy image --exit-code 1 --severity CRITICAL myapp:latest

# Scan a Dockerfile (check the base image before building)
trivy config Dockerfile

# Scan a local filesystem
trivy fs --security-checks vuln,secret ./
```

### Integrate Trivy in CI

```yaml
# GitHub Actions example
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'myapp:${{ github.sha }}'
    format: 'table'
    exit-code: '1'
    severity: 'CRITICAL,HIGH'
```

## Minimal Base Images

The fewer files in your image, the smaller the attack surface. Compare these base images:

| Base Image | Size | Packages | Shell | Security Posture |
|-----------|------|----------|-------|-----------------|
| `ubuntu:22.04` | 78 MB | ~100 | bash | Large attack surface |
| `debian:bookworm-slim` | 75 MB | ~80 | bash | Slightly smaller |
| `alpine:3.18` | 7 MB | ~15 | sh | Small, uses musl libc |
| `distroless/base` | 20 MB | ~5 | None | Minimal, no shell access |
| `distroless/static` | 2 MB | ~2 | None | Only static binaries |
| `scratch` | 0 MB | 0 | None | Absolute minimum |

### Distroless Images

Google's distroless images contain only your application and its runtime dependencies — no shell, no package manager, no unnecessary utilities:

```dockerfile
# Multi-stage build with distroless
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

Benefits:
- No shell means `docker exec bash` doesn't work — attackers can't get an interactive shell
- No package manager means attackers can't install tools
- Fewer files means fewer potential CVEs

Drawback: debugging is harder. You can't exec into the container. Use the ephemeral debug container technique from the previous article.

### Scratch Images (Go, Rust)

For statically compiled languages, you can use `scratch` (literally empty):

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

The final image contains exactly one file (plus CA certificates). Attack surface: almost zero.

## Docker Content Trust

Docker Content Trust (DCT) uses digital signatures to verify image authenticity:

```bash
# Enable content trust
export DOCKER_CONTENT_TRUST=1

# Now pulls and pushes require signatures
docker pull nginx:latest
# Only succeeds if the image is signed

# Push a signed image (requires setting up signing keys)
docker push myrepo/myapp:v1.0
# Docker will prompt for a signing passphrase
```

DCT uses The Update Framework (TUF) to manage keys and signatures. When enabled:
- `docker pull` verifies that the image was signed by a trusted publisher
- `docker push` signs the image with your key
- Unsigned images are rejected

This prevents supply chain attacks where a registry is compromised and images are replaced with malicious ones.

## Resource Limits

Without resource limits, a container can consume unlimited CPU, memory, and disk I/O — starving other containers and the host:

```bash
# Memory limit (container is killed if it exceeds this)
docker run --memory 512m myapp

# Memory limit with swap
docker run --memory 512m --memory-swap 1g myapp

# CPU limit (container gets at most 0.5 CPU cores)
docker run --cpus 0.5 myapp

# CPU shares (relative weight, default 1024)
docker run --cpu-shares 512 myapp

# Combined limits
docker run \
    --memory 512m \
    --memory-swap 512m \
    --cpus 1.0 \
    --pids-limit 100 \
    myapp
```

In Docker Compose:

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

| Resource | Flag | Effect |
|----------|------|--------|
| Memory | `--memory 512m` | Hard limit, OOM-killed if exceeded |
| Memory + Swap | `--memory-swap 1g` | Total memory+swap limit |
| CPU | `--cpus 0.5` | Hard limit: 50% of one core |
| CPU shares | `--cpu-shares 512` | Relative weight (soft limit) |
| PIDs | `--pids-limit 100` | Max number of processes (prevents fork bombs) |
| Disk I/O | `--device-read-bps /dev/sda:1mb` | Disk bandwidth limit |

The `--pids-limit` flag is often overlooked but prevents fork bomb attacks:

```bash
# Without --pids-limit, a fork bomb crashes the host
# With it, the container is limited to 100 processes
docker run --pids-limit 100 myapp
```

## Security Options


![Security layers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-security-layers.png)

### Seccomp Profiles

Seccomp (Secure Computing Mode) filters which system calls a container can make. Docker's default seccomp profile blocks ~60 dangerous syscalls:

![Seccomp syscall filtering](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/07-seccomp-profile.png)


```bash
# Run with the default seccomp profile (automatic)
docker run myapp

# Run with a custom seccomp profile
docker run --security-opt seccomp=/path/to/profile.json myapp

# Disable seccomp (DON'T do this in production)
docker run --security-opt seccomp=unconfined myapp
```

### AppArmor and SELinux

Docker automatically applies AppArmor (Ubuntu/Debian) or SELinux (RHEL/CentOS) profiles:

```bash
# Check the AppArmor profile
docker inspect my-container --format '{{.AppArmorProfile}}'
# Output: docker-default

# Run with a custom AppArmor profile
docker run --security-opt apparmor=my-custom-profile myapp
```

### No New Privileges

Prevents processes inside the container from gaining new privileges (through setuid binaries, for example):

```bash
docker run --security-opt no-new-privileges:true myapp
```

In Docker Compose:

```yaml
services:
  api:
    image: myapp
    security_opt:
      - no-new-privileges:true
```

## Security Best Practices Checklist

| Practice | Priority | Implementation |
|----------|----------|---------------|
| Run as non-root user | Critical | `USER appuser` in Dockerfile |
| Use specific image tags | Critical | `FROM python:3.11.5-slim`, never `latest` |
| Scan images for CVEs | Critical | `trivy image myapp` in CI pipeline |
| Drop all capabilities | High | `--cap-drop ALL --cap-add <needed>` |
| Use read-only filesystem | High | `--read-only --tmpfs /tmp` |
| Set memory limits | High | `--memory 512m` |
| Use .dockerignore | High | Exclude `.git`, `.env`, secrets |
| No secrets in images | Critical | Use runtime env vars, mounted files, or Docker secrets |
| Use multi-stage builds | High | Build tools stay out of production image |
| Enable no-new-privileges | Medium | `--security-opt no-new-privileges:true` |
| Use minimal base images | Medium | Alpine, distroless, or scratch |
| Pin dependency versions | Medium | Lockfiles, exact version pins |
| Set PID limits | Medium | `--pids-limit 100` |
| Enable content trust | Medium | `DOCKER_CONTENT_TRUST=1` |
| Use health checks | Medium | `HEALTHCHECK CMD curl -f http://localhost/health` |
| Limit network exposure | Medium | Use custom networks, don't expose unnecessary ports |
| Audit image history | Low | `docker history --no-trunc myapp` |
| Use read-only volumes | Low | `-v config:/app/config:ro` |

## A Hardened Docker Compose Example

Putting it all together:

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
    internal: true  # No external access — only containers on this network

volumes:
  postgres-data:
```

Notice `backend` is an `internal: true` network — containers on this network cannot reach the internet, limiting the blast radius if the database container is compromised.

## What's Next

You now know how to secure individual containers: non-root users, minimal capabilities, read-only filesystems, image scanning, and resource limits. But security is one challenge — scaling is another. What happens when a single host isn't enough? When you need automatic failover, rolling updates, and service discovery across multiple machines? The final article previews container orchestration: Docker Swarm for simplicity and Kubernetes for scale, and when you might not need either.
