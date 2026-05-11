---
title: "Docker and Containers (4): Networking and Volumes — How Containers Talk and Persist"
date: 2023-09-22 09:00:00
tags:
  - Docker
  - Containers
  - Networking
  - Volumes
  - Storage
categories:
  - Docker and Containers
series: docker-containers
lang: en
description: "Containers are ephemeral by default — they lose data when deleted and run in isolated networks. Volumes and networks are the two mechanisms that connect containers to the persistent, communicating world."
disableNunjucks: true
series_order: 4
translationKey: "docker-containers-4"
---

Containers are deliberately isolated. That's the point. But useful applications need to accept connections from the outside world, talk to databases, and store data that survives container restarts. Docker provides two mechanisms for this: networks (how containers communicate) and volumes (how data persists). Getting these right is the difference between a demo and a deployment.

## Docker Networking

When Docker starts, it creates a virtual network infrastructure on the host. Each container gets its own network namespace (its own IP address, routing table, and network interfaces), and Docker manages how traffic flows between containers and the outside world.

### Network Drivers

Docker supports several network drivers, each suited to different use cases:

| Driver | Description | Use Case | Container Communication |
|--------|-------------|----------|------------------------|
| `bridge` | Default. Containers on a private virtual network. | Single-host applications | Via IP or DNS (custom bridge only) |
| `host` | Container shares the host's network namespace | Performance-critical, single container | Container uses host's IP directly |
| `overlay` | Multi-host networking via VXLAN | Docker Swarm, multi-node clusters | Across physical hosts |
| `none` | No networking at all | Batch processing, security isolation | None |
| `macvlan` | Container gets a MAC address on the physical network | Legacy apps that need to appear as physical hosts | Directly on LAN |
| `ipvlan` | Like macvlan but shares host's MAC address | Similar to macvlan, fewer switch issues | Directly on LAN |

For most single-host use cases, you'll use **bridge** networks. Let's explore each relevant driver.

### The Default Bridge Network

When you install Docker, it creates a network called `bridge` (backed by a Linux bridge device called `docker0`):

```bash
docker network ls
```

```
NETWORK ID     NAME      DRIVER    SCOPE
a1b2c3d4e5f6   bridge    bridge    local
b2c3d4e5f6a7   host      host      local
c3d4e5f6a7b8   none      null      local
```

```bash
docker network inspect bridge --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}'
```

```
172.17.0.0/16
```

Every container that doesn't specify a network joins the default bridge. Let's see it in action:

```bash
# Run two containers on the default bridge
docker run -d --name container-a alpine sleep 3600
docker run -d --name container-b alpine sleep 3600

# Check their IP addresses
docker inspect container-a --format '{{.NetworkSettings.IPAddress}}'
# Output: 172.17.0.2

docker inspect container-b --format '{{.NetworkSettings.IPAddress}}'
# Output: 172.17.0.3

# They can reach each other by IP
docker exec container-a ping -c 2 172.17.0.3
```

```
PING 172.17.0.3 (172.17.0.3): 56 data bytes
64 bytes from 172.17.0.3: seq=0 ttl=64 time=0.108 ms
64 bytes from 172.17.0.3: seq=1 ttl=64 time=0.090 ms
```

But DNS doesn't work on the default bridge:

```bash
docker exec container-a ping -c 2 container-b
# Output: ping: bad address 'container-b'
```

This is a deliberate limitation of the default bridge. For service discovery, you need a custom bridge network.

```bash
# Clean up
docker rm -f container-a container-b
```

### Custom Bridge Networks

Custom bridge networks provide automatic DNS resolution between containers — this is what you should use for applications:

```bash
# Create a custom network
docker network create my-app-network

# Run containers on it
docker run -d --name web --network my-app-network nginx
docker run -d --name api --network my-app-network alpine sleep 3600

# DNS resolution works
docker exec api ping -c 2 web
```

```
PING web (172.18.0.2): 56 data bytes
64 bytes from 172.18.0.2: seq=0 ttl=64 time=0.065 ms
64 bytes from 172.18.0.2: seq=1 ttl=64 time=0.078 ms
```

The container name `web` resolves to its IP address. This is Docker's built-in DNS server at work. Docker runs an embedded DNS server at `127.0.0.11` inside each container on a custom network.

Custom networks also provide isolation — containers on different networks can't communicate unless explicitly connected:

```bash
# Create another network
docker network create other-network
docker run -d --name isolated --network other-network alpine sleep 3600

# This will fail — different networks
docker exec isolated ping -c 2 web
# Output: ping: bad address 'web'

# Connect a container to multiple networks
docker network connect my-app-network isolated

# Now it works
docker exec isolated ping -c 2 web
```

```
PING web (172.18.0.2): 56 data bytes
64 bytes from 172.18.0.2: seq=0 ttl=64 time=0.089 ms
```

```bash
# Clean up
docker rm -f web api isolated
docker network rm my-app-network other-network
```

### Comparison: Default Bridge vs Custom Bridge

| Feature | Default Bridge | Custom Bridge |
|---------|---------------|---------------|
| DNS resolution | No (IP only) | Yes (by container name) |
| Automatic isolation | All containers connected | Only explicitly connected containers |
| Connect/disconnect at runtime | Requires container restart | Live connect/disconnect |
| Environment variable linking | Legacy `--link` | Not needed (use DNS) |
| Creation | Automatic | `docker network create` |

**Bottom line**: Always create custom bridge networks for your applications. The default bridge is for backward compatibility.

### Port Mapping

Containers have their own network namespace. To expose a container's port to the host (and thus the outside world), you map ports:

```bash
# Map host port 8080 to container port 80
docker run -d -p 8080:80 --name web nginx

# Map host port 3307 to container port 3306
docker run -d -p 3307:3306 --name db mysql:8

# Map to a specific host interface
docker run -d -p 127.0.0.1:8080:80 --name local-only nginx

# Map a range of ports
docker run -d -p 8000-8010:8000-8010 --name range-app myapp

# Let Docker choose a random host port
docker run -d -p 80 --name random-port nginx

# Check what port was assigned
docker port random-port
# Output: 80/tcp -> 0.0.0.0:32768
```

The format is always `HOST:CONTAINER`. Read it as "host port 8080 forwards to container port 80."

**EXPOSE vs -p**:

| Feature | `EXPOSE 80` (Dockerfile) | `-p 8080:80` (docker run) |
|---------|-------------------------|--------------------------|
| Effect | Documentation only | Actually publishes the port |
| Network access | None | Host can reach container on port 8080 |
| Required for networking? | No | Yes (for external access) |
| Container-to-container | Not needed (use DNS on custom network) | Not needed |

### Host Network Mode

The `host` network mode removes network isolation entirely — the container uses the host's network stack directly:

```bash
docker run -d --network host --name web nginx

# nginx is now directly on port 80 of the host — no port mapping needed
curl http://localhost:80
```

This eliminates the overhead of port mapping and NAT, providing slightly better network performance. The tradeoff: you lose port isolation (two containers can't both listen on port 80), and the container can see all host network interfaces.

Use host networking when:
- You need maximum network performance
- Your application binds to many ports dynamically
- You're running monitoring tools that need to see host network traffic

### Network Management Commands

```bash
# List all networks
docker network ls

# Create a network with options
docker network create \
    --driver bridge \
    --subnet 172.20.0.0/16 \
    --gateway 172.20.0.1 \
    --ip-range 172.20.240.0/20 \
    my-custom-net

# Inspect a network (see connected containers, configuration)
docker network inspect my-custom-net

# Connect a running container to a network
docker network connect my-custom-net my-container

# Disconnect a container from a network
docker network disconnect my-custom-net my-container

# Remove a network (must have no connected containers)
docker network rm my-custom-net

# Remove all unused networks
docker network prune
```

## Docker Volumes

By default, all data written inside a container is stored in its writable layer. When the container is deleted, that data is gone. Volumes provide persistent storage that exists independently of the container lifecycle.

### Three Types of Mounts

Docker supports three ways to mount storage into a container:

| Type | Syntax | Managed By | Location on Host | Use Case |
|------|--------|-----------|-------------------|----------|
| Named volume | `-v mydata:/data` | Docker | `/var/lib/docker/volumes/mydata/` | Databases, persistent app data |
| Bind mount | `-v /host/path:/container/path` | You | Anywhere you specify | Development (live code reload) |
| tmpfs | `--tmpfs /tmp` | Kernel | RAM only | Sensitive data, caches |

### Named Volumes

Named volumes are the recommended way to persist data. Docker manages the storage location, and volumes can be shared between containers:

```bash
# Create a named volume
docker volume create app-data

# Run a container with the volume
docker run -d \
    --name writer \
    -v app-data:/data \
    alpine sh -c "echo 'persistent data' > /data/message.txt && sleep 3600"

# Verify the data exists
docker exec writer cat /data/message.txt
# Output: persistent data

# Remove the container
docker rm -f writer

# The data survives — run a new container with the same volume
docker run --rm \
    -v app-data:/data \
    alpine cat /data/message.txt
# Output: persistent data
```

Volume lifecycle commands:

```bash
# List all volumes
docker volume ls
```

```
DRIVER    VOLUME NAME
local     app-data
local     postgres-data
local     redis-data
```

```bash
# Inspect a volume
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
# Remove a volume (must not be in use by any container)
docker volume rm app-data

# Remove all unused volumes (DANGEROUS — irreversible)
docker volume prune
```

### Bind Mounts

Bind mounts map a specific host directory into the container. They're essential for development workflows:

```bash
# Mount current directory into the container
docker run -d \
    --name dev-server \
    -v $(pwd):/app \
    -p 5000:5000 \
    python:3.11-slim \
    bash -c "cd /app && pip install flask && python app.py"
```

Any changes to files on the host are immediately visible in the container (and vice versa). This enables live development without rebuilding the image.

**Bind mount vs named volume comparison:**

```bash
# Named volume (Docker manages location)
docker run -v mydata:/app/data myapp

# Bind mount (you specify the exact path)
docker run -v /home/user/project/data:/app/data myapp

# Bind mount with read-only flag
docker run -v /home/user/config:/app/config:ro myapp
```

| Feature | Named Volume | Bind Mount |
|---------|-------------|------------|
| Created by | `docker volume create` or auto-created | Pre-existing host directory |
| Location | `/var/lib/docker/volumes/...` | Anywhere on host |
| Managed by | Docker CLI (`docker volume ...`) | Host filesystem tools |
| Pre-populated | Yes (container contents copied to volume on first use) | No (host contents override container) |
| Permission | Docker handles ownership | Must manage manually |
| Performance on macOS | Better (uses gRPC FUSE/VirtioFS) | Can be slow for large trees |
| Backup | `docker run --rm -v mydata:/data -v $(pwd):/backup alpine tar czf /backup/data.tar.gz /data` | Standard host backup tools |
| Use case | Production data persistence | Development (live reload) |

### tmpfs Mounts

tmpfs mounts exist only in memory and are never written to the host filesystem:

```bash
docker run -d \
    --name secure-app \
    --tmpfs /tmp:size=100m,mode=1777 \
    --tmpfs /run/secrets:size=1m,mode=0700 \
    myapp
```

Use tmpfs for:
- Temporary files that shouldn't persist
- Sensitive data (keys, tokens) that shouldn't touch disk
- Write-heavy caches where disk I/O would be a bottleneck

## Real-World Example: MySQL with Persistent Data

This example demonstrates why volumes matter. Without a volume, deleting the MySQL container destroys all your data.

```bash
# Create a network and volume
docker network create db-network
docker volume create mysql-data

# Run MySQL with persistent storage
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

# Wait for MySQL to start (check logs)
docker logs -f mysql-server 2>&1 | grep -m 1 "ready for connections"
```

```
2023-09-22T10:05:23.456789Z 0 [System] [MY-010931] [Server] /usr/sbin/mysqld: ready for connections. Version: '8.0.34'  socket: '/var/run/mysqld/mysqld.sock'  port: 3306  MySQL Community Server - GPL.
```

```bash
# Connect and create some data
docker exec -it mysql-server mysql -u appuser -papppass myapp -e "
CREATE TABLE users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100));
INSERT INTO users (name) VALUES ('Alice'), ('Bob'), ('Charlie');
SELECT * FROM users;
"
```

```
+----+---------+
| id | name    |
+----+---------+
|  1 | Alice   |
|  2 | Bob     |
|  3 | Charlie |
+----+---------+
```

Now, simulate a disaster — delete and recreate the container:

```bash
# Remove the container (data lives in the volume, not the container)
docker rm -f mysql-server

# Recreate with the same volume
docker run -d \
    --name mysql-server \
    --network db-network \
    -v mysql-data:/var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=rootpass \
    -p 3306:3306 \
    mysql:8.0

# Wait for startup, then check data
docker exec -it mysql-server mysql -u appuser -papppass myapp -e "SELECT * FROM users;"
```

```
+----+---------+
| id | name    |
+----+---------+
|  1 | Alice   |
|  2 | Bob     |
|  3 | Charlie |
+----+---------+
```

Data survived. The volume (`mysql-data`) exists independently of the container. You can delete and recreate containers freely as long as you mount the same volume.

### Volume Backup and Restore

```bash
# Backup: run a temporary container that mounts the volume and creates a tar
docker run --rm \
    -v mysql-data:/source:ro \
    -v $(pwd):/backup \
    alpine tar czf /backup/mysql-backup-$(date +%Y%m%d).tar.gz -C /source .

# Check the backup
ls -lh mysql-backup-*.tar.gz
# Output: -rw-r--r-- 1 user user 45M Sep 22 11:00 mysql-backup-20230922.tar.gz

# Restore: create a new volume and extract the backup
docker volume create mysql-restored
docker run --rm \
    -v mysql-restored:/target \
    -v $(pwd):/backup \
    alpine tar xzf /backup/mysql-backup-20230922.tar.gz -C /target
```

### Volume Permissions

A common source of frustration: the container process runs as a specific user, but the volume files are owned by a different user.

```bash
# Problem: Container runs as UID 1000, but volume files are owned by root
docker run -v mydata:/data myapp
# Error: Permission denied writing to /data

# Solution 1: Set ownership in Dockerfile
RUN mkdir /data && chown 1000:1000 /data
VOLUME /data

# Solution 2: Run with matching UID
docker run --user 1000:1000 -v mydata:/data myapp

# Solution 3: Use an init script that fixes permissions
# (common pattern in official database images)
```

## Data Patterns: When to Use What

| Scenario | Storage Type | Example |
|----------|-------------|---------|
| Database files | Named volume | `-v postgres-data:/var/lib/postgresql/data` |
| Application logs | Named volume or bind mount | `-v app-logs:/var/log/app` |
| Configuration files | Bind mount (read-only) | `-v ./config:/app/config:ro` |
| Source code (development) | Bind mount | `-v $(pwd):/app` |
| Uploaded files | Named volume | `-v uploads:/app/uploads` |
| Temporary/scratch files | tmpfs | `--tmpfs /tmp` |
| Secrets | tmpfs or Docker secrets | `--tmpfs /run/secrets` |
| Shared data between containers | Named volume | Two containers mount same volume |
| Build cache | Named volume | `-v build-cache:/root/.cache` |

## Putting It Together: Networked App with Persistence

Here's a complete example — a Python app that connects to Redis, with both on a custom network and Redis data persisted:

```bash
# Create infrastructure
docker network create app-net
docker volume create redis-data

# Run Redis with persistence
docker run -d \
    --name redis \
    --network app-net \
    -v redis-data:/data \
    redis:7 \
    redis-server --appendonly yes

# Run the app
docker run -d \
    --name app \
    --network app-net \
    -p 8080:5000 \
    -e REDIS_HOST=redis \
    my-flask-app

# The app can reach Redis by name "redis" (DNS on custom network)
# The outside world reaches the app on port 8080
# Redis data survives container restarts
```

This manual setup works but is verbose and error-prone. Imagine managing 5 services this way. That's exactly the problem Docker Compose solves.

```bash
# Clean up everything
docker rm -f redis app
docker network rm app-net
docker volume rm redis-data mysql-data
```

## Inspecting Network and Volume State

```bash
# See all port mappings for a container
docker port my-container
```

```
80/tcp -> 0.0.0.0:8080
443/tcp -> 0.0.0.0:8443
```

```bash
# See network settings for a container
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
# See mount details for a container
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

## What's Next

You now know how to connect containers to each other and to the outside world with networks, and how to persist data with volumes. But running `docker run` with 10 flags for each of 5 services is tedious, error-prone, and impossible to share with teammates. The next article introduces Docker Compose — a declarative way to define and run multi-container applications with a single YAML file. Everything we covered in this article (networks, volumes, port mappings) gets expressed in a `docker-compose.yml` that serves as both documentation and executable infrastructure.
