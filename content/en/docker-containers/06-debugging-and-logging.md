---
title: "Docker and Containers (6): Debugging and Logging — When Things Go Wrong Inside a Box"
date: 2023-06-26 09:00:00
tags:
  - Docker
  - Containers
  - Debugging
  - Logging
  - Observability
categories:
  - Docker and Containers
series: docker-containers
lang: en
description: "Containers hide their internals by design. When something breaks, you need specific tools and techniques to see inside the box without breaking the isolation that makes containers useful."
disableNunjucks: true
series_order: 6
translationKey: "docker-containers-6"
---

A container that works is invisible. A container that doesn't work is a black box. The entire point of containerization is isolation — but that same isolation makes debugging harder. You can't just `ssh` into a container or browse its filesystem from the host. Docker provides a specific set of tools for inspecting, diagnosing, and understanding what happens inside running (and crashed) containers.

## Reading Container Logs

Logs are your first line of investigation. Docker captures anything a container writes to stdout and stderr.

### docker logs

```bash
# View all logs from a container
docker logs my-container

# Follow logs in real time (like tail -f)
docker logs -f my-container

# Show the last 100 lines
docker logs --tail 100 my-container

# Show logs since a specific time
docker logs --since 2023-09-30T10:00:00 my-container

# Show logs from the last 30 minutes
docker logs --since 30m my-container

# Show timestamps with each log line
docker logs -t my-container
```

Example output with timestamps:

```
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

### Logs from stopped containers

This is critical — when a container crashes, it stops, but its logs are preserved until the container is removed:

```bash
# List stopped containers
docker ps -a --filter "status=exited"
```

```
CONTAINER ID   IMAGE       COMMAND            CREATED          STATUS                     NAMES
a1b2c3d4e5f6   myapp:v2    "python app.py"    10 minutes ago   Exited (1) 8 minutes ago   crashed-app
```

```bash
# View the crash logs
docker logs crashed-app
```

```
Traceback (most recent call last):
  File "/app/app.py", line 12, in <module>
    db = psycopg2.connect(os.environ['DATABASE_URL'])
  File "/usr/local/lib/python3.11/site-packages/psycopg2/__init__.py", line 122
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
psycopg2.OperationalError: could not connect to server: Connection refused
    Is the server running on host "postgres" (172.18.0.2) and accepting
    TCP/IP connections on port 5432?
```

The container exited with code 1 (error). The logs reveal it couldn't connect to PostgreSQL. Maybe the database wasn't ready when the app started (a dependency ordering issue), or the hostname is wrong.

### Exit codes

```bash
# Check exit code
docker inspect crashed-app --format '{{.State.ExitCode}}'
# Output: 1
```

Common exit codes:

| Exit Code | Meaning | Common Cause |
|-----------|---------|--------------|
| 0 | Success | Normal shutdown |
| 1 | General error | Application error, exception |
| 2 | Shell builtin misuse | Bad script syntax |
| 126 | Command not executable | Permission issue on entrypoint |
| 127 | Command not found | Wrong CMD/ENTRYPOINT, missing binary |
| 137 | SIGKILL (128+9) | OOM killer, `docker kill`, timeout |
| 139 | SIGSEGV (128+11) | Segmentation fault |
| 143 | SIGTERM (128+15) | `docker stop` (graceful shutdown) |

Exit code 137 deserves special attention — it usually means the container was killed by the OOM (Out of Memory) killer because it exceeded its memory limit.

```bash
# Check if a container was OOM-killed
docker inspect crashed-app --format '{{.State.OOMKilled}}'
# Output: true
```

## Interactive Debugging with docker exec

`docker exec` runs a command inside a running container. It's the primary way to interactively debug:

```bash
# Open a shell inside a running container
docker exec -it my-container bash

# If bash isn't available (Alpine, distroless)
docker exec -it my-container sh

# Run a specific command
docker exec my-container cat /app/config.yaml

# Run as a different user
docker exec -u root my-container apt-get update

# Set environment variables for the command
docker exec -e DEBUG=true my-container python check.py
```

Once inside a container, you can investigate:

```bash
# Check environment variables
env | sort

# Check network connectivity
curl -v http://postgres:5432
ping redis

# Check DNS resolution
nslookup postgres
cat /etc/resolv.conf

# Check running processes
ps aux

# Check file permissions
ls -la /app/
stat /app/config.yaml

# Check disk usage
df -h

# Check memory
free -m
cat /proc/meminfo

# Check open files and connections
# (if the tool is available)
ss -tlnp
netstat -tlnp
```

### When bash isn't available

Minimal images (Alpine, distroless) might not have bash or common tools:

```bash
# Alpine uses sh, not bash
docker exec -it alpine-container sh

# Install debugging tools temporarily
docker exec -it alpine-container apk add --no-cache curl bind-tools
```

For distroless images, you can't exec into them at all (no shell). Use a debug sidecar instead:

```bash
# Run a debug container that shares the network namespace
docker run -it --rm \
    --network container:my-distroless-container \
    nicolaka/netshoot \
    bash
```

The `nicolaka/netshoot` image contains every network debugging tool you could want (curl, nslookup, tcpdump, iptables, etc.), and `--network container:my-distroless-container` makes it share the target container's network namespace.

## docker inspect — The Complete Picture

`docker inspect` returns detailed JSON metadata about a container. It's verbose but contains everything:

```bash
# Full output (very long)
docker inspect my-container

# Specific fields with Go template formatting
docker inspect my-container --format '{{.State.Status}}'
# Output: running

docker inspect my-container --format '{{.State.StartedAt}}'
# Output: 2023-09-30T10:15:23.123456789Z

docker inspect my-container --format '{{.Config.Image}}'
# Output: myapp:v2

# Network information
docker inspect my-container --format '{{range .NetworkSettings.Networks}}IP: {{.IPAddress}} Gateway: {{.Gateway}}{{end}}'
# Output: IP: 172.18.0.3 Gateway: 172.18.0.1

# Environment variables
docker inspect my-container --format '{{range .Config.Env}}{{println .}}{{end}}'
```

```
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
DATABASE_URL=postgresql://postgres:secret@postgres:5432/myapp
REDIS_URL=redis://redis:6379
NODE_ENV=production
```

```bash
# Mount points
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
# Port bindings
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

### Useful inspect one-liners

```bash
# Get container's IP address
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' my-container

# Get container's MAC address
docker inspect -f '{{range.NetworkSettings.Networks}}{{.MacAddress}}{{end}}' my-container

# Get container's restart count
docker inspect -f '{{.RestartCount}}' my-container

# Get the image used
docker inspect -f '{{.Config.Image}}' my-container

# Get the command being run
docker inspect -f '{{json .Config.Cmd}}' my-container

# Check if container is running
docker inspect -f '{{.State.Running}}' my-container
```

## docker stats — Real-Time Resource Monitoring

`docker stats` provides a live view of resource consumption:

```bash
docker stats
```

```
CONTAINER ID   NAME              CPU %   MEM USAGE / LIMIT     MEM %   NET I/O           BLOCK I/O         PIDS
a1b2c3d4e5f6   myapp-api-1       2.34%   125.4MiB / 512MiB     24.49%  15.2kB / 8.9kB    4.1MB / 0B        12
b2c3d4e5f6a7   myapp-postgres-1  0.45%   89.2MiB / 1GiB        8.71%   12.1kB / 9.8kB    28.5MB / 12.3MB   15
c3d4e5f6a7b8   myapp-redis-1     0.12%   12.5MiB / 256MiB      4.88%   8.4kB / 6.2kB     0B / 524kB        5
d4e5f6a7b8c9   myapp-worker-1    15.67%  234.5MiB / 512MiB     45.80%  5.1kB / 3.2kB     0B / 0B           8
```

Key columns:

| Column | What It Shows | Watch For |
|--------|--------------|-----------|
| CPU % | CPU usage relative to host | Sustained high CPU = bottleneck |
| MEM USAGE / LIMIT | Current memory / configured limit | Approaching limit = OOM risk |
| MEM % | Percentage of limit used | > 80% is concerning |
| NET I/O | Network bytes in/out | Unexpected traffic patterns |
| BLOCK I/O | Disk read/write | High writes = possible log flooding |
| PIDS | Number of processes | Growing = possible leak |

```bash
# Monitor a specific container
docker stats my-container --no-stream

# Format output
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## docker top — Process Listing

```bash
docker top my-container
```

```
UID                 PID                 PPID                C                   STIME               TTY                 TIME                CMD
appuser             12345               12330               0                   10:15               ?                   00:00:05            gunicorn: master [app:app]
appuser             12350               12345               2                   10:15               ?                   00:01:30            gunicorn: worker [app:app]
appuser             12351               12345               2                   10:15               ?                   00:01:28            gunicorn: worker [app:app]
appuser             12352               12345               1                   10:15               ?                   00:01:25            gunicorn: worker [app:app]
appuser             12353               12345               2                   10:15               ?                   00:01:32            gunicorn: worker [app:app]
```

Notice the PID and PPID columns show host PIDs. Inside the container, the master process would be PID 1, but on the host it's PID 12345. This is namespace mapping.

```bash
# Use ps-style formatting
docker top my-container -o pid,ppid,user,%cpu,%mem,comm
```

## docker diff — Filesystem Changes

`docker diff` shows what files have been added, changed, or deleted in the container's writable layer compared to the image:

```bash
docker diff my-container
```

```
C /tmp
A /tmp/cache
A /tmp/cache/session_abc123
C /var/log
A /var/log/app.log
C /app
C /app/config.yaml
A /app/data/uploads/image001.png
```

| Prefix | Meaning |
|--------|---------|
| A | Added — file doesn't exist in the image |
| C | Changed — file was modified |
| D | Deleted — file existed in the image but was removed |

This is useful for:
- Finding where an application writes data (should it be a volume instead?)
- Detecting unexpected file modifications
- Understanding what state accumulates in a running container

## Debugging Crashed Containers

When a container exits immediately, you can't `docker exec` into it. Here are strategies:

### Strategy 1: Check the logs

```bash
docker logs crashed-container
```

This works for any stopped container (until you `docker rm` it).

### Strategy 2: Override the command

If the container crashes on startup, override the command to keep it running:

```bash
# Instead of the normal command, just sleep
docker run -it --name debug-container myapp:v2 bash

# Or override entrypoint entirely
docker run -it --entrypoint bash myapp:v2

# Now you're inside the container and can investigate
ls /app/
cat /app/config.yaml
python -c "import psycopg2; print('module found')"
```

### Strategy 3: Copy files out of a stopped container

```bash
# Create the container without starting it
docker create --name debug-container myapp:v2

# Copy files out
docker cp debug-container:/app/config.yaml ./debug-config.yaml
docker cp debug-container:/var/log/ ./debug-logs/

# Clean up
docker rm debug-container
```

### Strategy 4: Commit a stopped container as an image

```bash
# The container crashed but still exists
docker ps -a | grep crashed

# Save its state as a new image
docker commit crashed-container debug-image:latest

# Now you can explore it
docker run -it debug-image:latest bash
```

## Ephemeral Debug Containers

Sometimes you need networking tools, strace, or other debugging utilities that aren't in your application image. Run a separate debug container that shares the target's network:

```bash
# Debug container sharing target's network namespace
docker run -it --rm \
    --network container:my-app-container \
    nicolaka/netshoot \
    bash
```

Inside `netshoot`, you have access to the target container's network:

```bash
# DNS resolution (from netshoot, using my-app-container's network)
nslookup postgres

# TCP connection test
nc -zv postgres 5432

# HTTP request
curl -v http://localhost:8080/health

# Packet capture
tcpdump -i eth0 -n port 5432

# Trace route
traceroute postgres
```

You can also share other namespaces:

```bash
# Share PID namespace (see the target container's processes)
docker run -it --rm \
    --pid container:my-app-container \
    ubuntu bash -c "ps aux"
```

## Log Drivers

By default, Docker stores logs as JSON files on disk. You can configure different log drivers:

```bash
# Check current log driver
docker info --format '{{.LoggingDriver}}'
# Output: json-file
```

### Available log drivers

| Driver | Destination | `docker logs` Support | Use Case |
|--------|------------|----------------------|----------|
| `json-file` | Local JSON files | Yes | Default, development |
| `local` | Optimized local storage | Yes | Production single-host |
| `syslog` | Syslog daemon | No | Traditional Linux logging |
| `journald` | systemd journal | Yes | systemd-based hosts |
| `fluentd` | Fluentd collector | No | Centralized logging (ELK/EFK) |
| `awslogs` | CloudWatch Logs | No | AWS deployments |
| `gcplogs` | Google Cloud Logging | No | GCP deployments |
| `splunk` | Splunk HTTP Event Collector | No | Enterprise environments |
| `none` | Discard all logs | No | High-throughput, logs handled internally |

### Configuring log drivers per container

```bash
# Use a specific log driver for one container
docker run -d \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    --name my-container \
    myapp
```

### Configuring log rotation (critical for production)

Without log rotation, `json-file` logs grow unbounded and will eventually fill your disk:

```bash
# Docker daemon configuration (/etc/docker/daemon.json)
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

| Option | Effect | Recommended |
|--------|--------|-------------|
| `max-size` | Maximum size per log file | 10m - 50m |
| `max-file` | Number of rotated log files to keep | 3 - 5 |
| `compress` | Compress rotated files | true |

Without these settings, a busy container can generate gigabytes of logs.

### In Docker Compose

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

## Common Failure Patterns

Here's a diagnostic table for the most frequent container issues:

| Symptom | Likely Cause | Diagnostic Command | Fix |
|---------|-------------|-------------------|-----|
| Container exits immediately | Application crash on startup | `docker logs container` | Check config, dependencies |
| Exit code 137 | Out of memory (OOM killed) | `docker inspect -f '{{.State.OOMKilled}}'` | Increase `--memory` or fix memory leak |
| Exit code 127 | Command not found | `docker inspect -f '{{json .Config.Cmd}}'` | Check CMD/ENTRYPOINT spelling, verify binary exists |
| Exit code 126 | Permission denied on command | `docker exec container ls -la /entrypoint.sh` | `chmod +x` the entrypoint, check USER |
| "Address already in use" | Port conflict | `docker ps` (check port mappings) | Use a different host port |
| "Connection refused" to service | Service not ready or wrong hostname | `docker exec container curl http://service:port` | Check `depends_on`, healthchecks, network |
| DNS resolution failure | Not on custom network | `docker network inspect bridge` | Create custom network with `docker network create` |
| File permission errors | UID mismatch between host and container | `docker exec container id` + `ls -la /path` | Match UIDs or use named volumes |
| Slow performance on macOS | Bind mount I/O overhead | `docker stats` | Use named volumes for dependencies, bind mount only source |
| Container restarts in loop | CrashLoopBackOff (crash → restart → crash) | `docker logs --tail 50 container` | Fix the underlying crash, set `restart: on-failure` |

## Debugging Workflow Checklist

When a container isn't working, follow this sequence:

```bash
# 1. Is the container running?
docker ps -a | grep my-container

# 2. What do the logs say?
docker logs --tail 100 my-container

# 3. What's the exit code?
docker inspect my-container --format '{{.State.ExitCode}} OOM:{{.State.OOMKilled}}'

# 4. What's the resource usage?
docker stats my-container --no-stream

# 5. Can I get inside?
docker exec -it my-container bash  # or sh

# 6. What processes are running?
docker top my-container

# 7. What changed in the filesystem?
docker diff my-container

# 8. What's the full configuration?
docker inspect my-container

# 9. Can the container reach its dependencies?
docker exec my-container ping postgres
docker exec my-container curl -v http://api:8000/health

# 10. Is the network configured correctly?
docker network inspect $(docker inspect my-container --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}')
```

## Docker Events — System-Wide Activity

`docker events` streams real-time events from the Docker daemon:

```bash
docker events
```

```
2023-09-30T10:30:15.123456789Z container start a1b2c3d4e5f6 (image=myapp:v2, name=api)
2023-09-30T10:30:45.234567890Z container die a1b2c3d4e5f6 (exitCode=1, image=myapp:v2, name=api)
2023-09-30T10:30:46.345678901Z container start a1b2c3d4e5f6 (image=myapp:v2, name=api)
2023-09-30T10:31:16.456789012Z container die a1b2c3d4e5f6 (exitCode=1, image=myapp:v2, name=api)
```

This shows the container is in a restart loop (die → start → die). Filter events to reduce noise:

```bash
# Only container events
docker events --filter type=container

# Only events for a specific container
docker events --filter container=my-container

# Only specific event types
docker events --filter event=die --filter event=oom
```

## What's Next

Now you can find out what's wrong when containers misbehave. But debugging is reactive — ideally, you prevent problems before they happen. The next article covers security: running containers as non-root, dropping capabilities, scanning for vulnerabilities, and following best practices that prevent the most common security mistakes in containerized applications.
