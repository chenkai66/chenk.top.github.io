---
title: "Docker and Containers (1): Why Containers — The Problem VMs Didn't Solve"
date: 2023-06-16 09:00:00
tags:
  - Docker
  - Containers
  - Virtualization
  - Linux
  - DevOps
categories:
  - Docker and Containers
series: docker-containers
lang: en
description: "Containers solve the 'works on my machine' problem that VMs made expensive. Learn what containers actually are, how they differ from VMs, and run your first one."
disableNunjucks: true
series_order: 1
translationKey: "docker-containers-1"
---

Every developer has heard the phrase "it works on my machine." Virtual machines were supposed to fix that, and they did — at the cost of gigabytes of RAM, minutes of boot time, and an entire duplicate operating system per application. Containers asked a different question: what if we could isolate applications without duplicating the kernel?

## The Actual Problem

Consider deploying a Python web application. You need Python 3.11, specific pip packages, a particular version of libssl, and some system-level configuration. Your colleague's app needs Python 3.9 and a conflicting libssl version. The staging server runs Ubuntu 20.04 while production runs Amazon Linux 2.

![VMs vs containers comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-vm-vs-container.png)


Virtual machines solve this by giving each application its own entire operating system. It works, but it's wasteful. Each VM carries:

- A full kernel (hundreds of MB)
- System daemons (init, syslog, cron — none of which your app needs)
- Duplicate copies of shared libraries
- Its own memory management overhead

Containers solve the same isolation problem by sharing the host kernel and isolating only what matters: the filesystem, process tree, network stack, and resource limits.

## What Containers Actually Are

A container is not a lightweight VM. That analogy is convenient but misleading. A container is a regular Linux process (or group of processes) with three kernel features applied to it:

![Container architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-container-architecture.png)


### Namespaces — What the Process Can See

Linux namespaces partition kernel resources so that one set of processes sees one set of resources while another set sees a different set. There are several namespace types:

![Namespaces and cgroups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-namespace-cgroup.png)


| Namespace | Isolates | Effect |
|-----------|----------|--------|
| `pid` | Process IDs | Container sees only its own processes; PID 1 inside is not PID 1 on host |
| `net` | Network stack | Container gets its own IP address, routing table, ports |
| `mnt` | Mount points | Container has its own filesystem view |
| `uts` | Hostname | Container can have its own hostname |
| `ipc` | Inter-process communication | Shared memory and semaphores are isolated |
| `user` | User/group IDs | Root inside container can map to non-root on host |
| `cgroup` | Cgroup root directory | Container sees only its own cgroup hierarchy |

When you run `docker run nginx`, Docker creates a new set of these namespaces and launches the nginx process inside them. The process is still a regular process on the host — you can see it with `ps aux` — but from inside, it thinks it's alone on the machine.

### Cgroups — What the Process Can Use

Control groups (cgroups) limit and account for resource usage. While namespaces control visibility, cgroups control consumption:

| Resource | Cgroup Controller | What It Limits |
|----------|-------------------|----------------|
| CPU | `cpu`, `cpuacct` | CPU time, shares, quota |
| Memory | `memory` | RAM usage, swap, OOM behavior |
| Disk I/O | `blkio` | Read/write rates to block devices |
| Network | `net_cls`, `net_prio` | Traffic classification and priority |
| PIDs | `pids` | Maximum number of processes |

Without cgroups, a runaway container could consume all host memory and crash everything. With cgroups, you can say "this container gets at most 512MB of RAM and 0.5 CPU cores."

### Union Filesystem — How the Filesystem Works

Containers use a layered filesystem. Instead of copying an entire OS filesystem for each container, layers are stacked on top of each other using a union filesystem (OverlayFS on modern Linux). The bottom layers are read-only (shared between containers), and each container gets a thin writable layer on top.

This is why starting a container is fast: there's no filesystem to copy. The container just gets a pointer to existing read-only layers plus a new empty writable layer.

## Containers vs Virtual Machines


![Shipping containers stacked on a digital cargo ship represen](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/01-shipping-containers-stacked-on-a-digital-cargo-ship-represen.jpg)

The architectural difference is fundamental. Here's how the two stacks compare (imagine this as a diagram):

![Resource overhead comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-resource-comparison.png)


**Virtual Machine stack (bottom to top):**
```
Hardware → Host OS → Hypervisor → [Guest OS + Bins/Libs + App] per VM
```

**Container stack (bottom to top):**
```
Hardware → Host OS (shared kernel) → Container Runtime → [Bins/Libs + App] per Container
```

The key difference: VMs virtualize hardware. Containers virtualize the operating system. VMs run their own kernel. Containers share the host kernel.

Here's a concrete comparison:

| Characteristic | Virtual Machine | Container |
|---------------|----------------|-----------|
| Boot time | 30-60 seconds | < 1 second |
| Disk footprint | 1-20 GB per VM | 10-500 MB per container |
| Memory overhead | 512 MB - several GB | Essentially zero (shared kernel) |
| Isolation level | Hardware-level (strong) | Process-level (good, not perfect) |
| Kernel | Own kernel per VM | Shared host kernel |
| OS support | Any OS (Windows on Linux, etc.) | Same kernel family as host |
| Density | 10-20 VMs per host typical | 100s of containers per host |
| Live migration | Supported | Not natively (orchestrators handle it) |
| Performance | Near-native with hardware virtualization | Native (no virtualization layer) |

When do you still want VMs? When you need strong security isolation (multi-tenant cloud), when you need a different kernel (running Windows workloads on Linux hosts), or when your compliance requirements mandate hardware-level separation.

## The Container Ecosystem: OCI, Docker, containerd, Podman

Docker popularized containers, but it's not the only player. Understanding the ecosystem prevents confusion later.

![Container isolation layers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/01-isolation-layers.png)


### The OCI Standard

The Open Container Initiative (OCI) defines two specifications:

1. **Image Spec**: What a container image looks like (layers, manifests, configuration)
2. **Runtime Spec**: How to run a container (lifecycle, configuration format)

Any tool that follows these specs can build images that run anywhere. This is why you can build with Docker, push to any registry, and run with Podman.

### The Container Runtime Stack

The stack has layers, and Docker sits on top:

```
docker CLI → dockerd (Docker daemon) → containerd → runc → Linux kernel
```

| Component | Role | Can You Use It Alone? |
|-----------|------|----------------------|
| `runc` | OCI runtime — actually creates namespaces and cgroups | Yes, but low-level |
| `containerd` | Manages container lifecycle, image pulls, storage | Yes (used by Kubernetes directly) |
| `dockerd` | Docker daemon — adds build, networking, volumes | Yes (what most people use) |
| `docker` CLI | User-facing command-line tool | Talks to dockerd |

### Docker vs Podman

Podman is a daemonless container engine. Instead of talking to a daemon (dockerd), Podman runs containers directly. The commands are almost identical:

```bash
# Docker
docker run -d -p 8080:80 nginx

# Podman (same syntax)
podman run -d -p 8080:80 nginx
```

Key differences:

| Feature | Docker | Podman |
|---------|--------|--------|
| Daemon | Yes (dockerd) | No (daemonless) |
| Root required | Default yes (rootless mode available) | Rootless by default |
| Compose | docker compose | podman-compose (or compatible) |
| Swarm | Built-in | No |
| Kubernetes YAML | No native support | `podman generate kube` |
| Systemd integration | Requires configuration | Native |

For learning and most production use cases, Docker remains the standard. Podman matters when you care about rootless operation or systemd integration.

## A Brief History of Containers

Containers didn't appear overnight. The journey took decades:

| Year | Technology | What It Did |
|------|-----------|-------------|
| 1979 | `chroot` | Changed the root directory for a process — first filesystem isolation |
| 2000 | FreeBSD Jails | Combined filesystem, process, and network isolation |
| 2001 | Linux VServer | Partitioned Linux into virtual private servers |
| 2004 | Solaris Zones | Full OS-level virtualization with resource controls |
| 2006 | Cgroups (Google) | Resource limiting and accounting — merged into Linux kernel |
| 2008 | LXC | Combined cgroups + namespaces into Linux containers |
| 2013 | Docker | Made containers accessible with a simple CLI and image format |
| 2014 | Kubernetes | Google open-sourced their container orchestration system |
| 2015 | OCI | Industry standard for container images and runtimes |
| 2017 | containerd | Extracted from Docker as a standalone runtime (CNCF project) |

Docker's contribution wasn't the technology itself — namespaces and cgroups existed before Docker. Docker's contribution was the developer experience: a simple CLI, a portable image format, and a public registry (Docker Hub) to share images.

## Installing Docker

### Linux (Ubuntu/Debian)

Don't install Docker from your distribution's default repositories — those packages are often outdated. Use Docker's official repository:

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install prerequisites
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group (avoids sudo for every command)
sudo usermod -aG docker $USER
newgrp docker
```

### Linux (CentOS/RHEL/Fedora)

```bash
# Remove old versions
sudo yum remove docker docker-client docker-client-latest docker-common \
  docker-latest docker-latest-logrotate docker-logrotate docker-engine

# Install prerequisites and add repo
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker Engine
sudo yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group
sudo usermod -aG docker $USER
```

### macOS

Docker Desktop is the standard approach for macOS. Docker doesn't run natively on macOS — it runs inside a lightweight Linux VM managed by Docker Desktop.

1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
2. Open the `.dmg` file and drag Docker to Applications
3. Launch Docker Desktop — it will ask for permissions to install networking components
4. Wait for the Docker icon in the menu bar to show "Docker Desktop is running"

Alternatively, use Homebrew:

```bash
brew install --cask docker
```

Then open Docker Desktop from Applications to complete the setup.

### Verify Installation

```bash
docker version
```

Expected output:

```
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

Notice both "Client" and "Server" sections. The client is the CLI tool; the server is the daemon that actually manages containers. On macOS, the server runs inside the Docker Desktop VM.

## Running Your First Container

```bash
docker run hello-world
```

Output:

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

Let's break down what happened:

1. `docker run` told the Docker client to run a container
2. The client sent this request to the Docker daemon (dockerd)
3. The daemon looked for the `hello-world:latest` image locally — didn't find it
4. The daemon pulled the image from Docker Hub (you can see the layer being downloaded)
5. The daemon created a container from the image
6. The container ran its program (which printed the message) and exited

## Exploring Docker Info

```bash
docker info
```

This command reveals your Docker installation's configuration:

```
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

Key details to note:

- **Storage Driver: overlay2** — the union filesystem driver being used
- **Cgroup Driver: systemd** — how resource limits are managed
- **Docker Root Dir: /var/lib/docker** — where all images, containers, and volumes live
- **Security Options** — AppArmor and seccomp profiles are active
- **Runtimes: runc** — the low-level OCI runtime

## Running Something More Interesting

Let's run an interactive Ubuntu container:

```bash
docker run -it ubuntu bash
```

You're now inside a container. Let's explore:

```bash
# Check the hostname — it's the container ID
hostname
# Output: a3f8b2c1d4e5

# Check the process list — only bash and ps are running
ps aux
# Output:
# USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
# root         1  0.0  0.0   4624  3840 pts/0    Ss   14:30   0:00 bash
# root        10  0.0  0.0   7060  1536 pts/0    R+   14:30   0:00 ps aux

# Check the OS release
cat /etc/os-release
# Output:
# PRETTY_NAME="Ubuntu 22.04.3 LTS"
# NAME="Ubuntu"
# ...

# Check the filesystem — it's a minimal Ubuntu
ls /
# Output: bin boot dev etc home lib lib64 media mnt opt proc root run sbin srv sys tmp usr var

# Exit the container
exit
```

Notice PID 1 is `bash`. Inside the container, bash is the init process. On the host, this same process has a completely different PID. That's namespace isolation in action.

After exiting, the container stops but isn't deleted:

```bash
docker ps -a
```

```
CONTAINER ID   IMAGE         COMMAND   CREATED          STATUS                     PORTS   NAMES
a3f8b2c1d4e5   ubuntu        "bash"    2 minutes ago    Exited (0) 30 seconds ago          hopeful_nobel
b7c9e1f2a3d4   hello-world   "/hello"  5 minutes ago    Exited (0) 5 minutes ago           festive_darwin
```

## Running a Background Container

Let's run nginx as a background service:

```bash
docker run -d -p 8080:80 --name my-nginx nginx
```

Breaking down the flags:

- `-d` — detached mode (run in background)
- `-p 8080:80` — map host port 8080 to container port 80
- `--name my-nginx` — give the container a memorable name
- `nginx` — the image to use

```bash
# Check it's running
docker ps
```

```
CONTAINER ID   IMAGE   COMMAND                  CREATED         STATUS         PORTS                  NAMES
c5d6e7f8a9b0   nginx   "/docker-entrypoint.…"   5 seconds ago   Up 4 seconds   0.0.0.0:8080->80/tcp   my-nginx
```

```bash
# Test it
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
# View the logs
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
# Clean up
docker stop my-nginx
docker rm my-nginx
```

## Essential Commands Reference

| Command | Purpose |
|---------|---------|
| `docker run IMAGE` | Create and start a container from an image |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers (including stopped) |
| `docker stop CONTAINER` | Gracefully stop a container |
| `docker rm CONTAINER` | Remove a stopped container |
| `docker images` | List downloaded images |
| `docker rmi IMAGE` | Remove an image |
| `docker pull IMAGE` | Download an image without running it |
| `docker exec -it CONTAINER bash` | Open a shell inside a running container |
| `docker logs CONTAINER` | View container output |
| `docker inspect CONTAINER` | Detailed container metadata (JSON) |

## What's Next

Now you know what containers are and how they differ from VMs. You've installed Docker, run your first container, and seen the basic commands. But we glossed over something important: when you ran `docker run nginx`, Docker downloaded an "image." What exactly is an image? How are those layers structured? Why did only one small layer download when you pulled Ubuntu, but several layers downloaded for nginx? The next article digs into images and the layer model — the foundation that makes containers fast and space-efficient.
