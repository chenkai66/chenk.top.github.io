---
title: "Docker and Containers (2): Images and Layers — What docker pull Actually Downloads"
date: 2023-06-17 09:00:00
tags:
  - Docker
  - Containers
  - Images
  - Registry
  - OCI
categories: Docker and Containers
series: docker-containers
lang: en
description: "Docker images aren't monolithic files — they're stacks of read-only layers shared between containers. Understanding layers is the key to fast builds and small images."
disableNunjucks: true
series_order: 2
translationKey: "docker-containers-2"
---

The first time I ran `docker pull ubuntu` I expected to download an entire operating system. Instead, it finished in seconds and was only 77 MB. That seemed impossibly small for a Linux distribution. The secret is layers — and understanding how they work changes the way you think about building and shipping containers.

---

## Image vs Container

Before diving into layers, let's clarify a fundamental distinction that trips up many beginners.

![Image registry workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-image-registry.png)


An **image** is a read-only template containing the filesystem, environment variables, default command, and metadata needed to create a container. Think of it as a class definition in object-oriented programming.

A **container** is a running (or stopped) instance created from an image. It includes everything the image has, plus a writable layer and runtime state (network, process IDs, etc.). Think of it as an object instantiated from a class.

```bash
# The image (template)
docker images
```

```text
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
nginx        latest    61395b4c586d   2 weeks ago   187MB
ubuntu       22.04     c6b84b685f35   3 weeks ago   77.8MB
```

```bash
# Containers created from images (instances)
docker ps -a
```

```text
CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS                     NAMES
a1b2c3d4e5f6   nginx          "/docker-entrypoint.…"   10 minutes ago   Up 10 minutes              web1
b2c3d4e5f6a7   nginx          "/docker-entrypoint.…"   8 minutes ago    Up 8 minutes               web2
c3d4e5f6a7b8   ubuntu:22.04   "bash"                   5 minutes ago    Exited (0) 3 minutes ago   test
```

Notice two containers (`web1` and `web2`) running from the same `nginx` image. They share the same read-only layers but each has its own writable layer. Changes in `web1` don't affect `web2`, and neither affects the image.

## The Layer Model

Every Docker image is built from a stack of layers. Each layer represents filesystem changes — files added, modified, or deleted. Layers are:

![Docker layer building animation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/docker-02-layer-building.gif)


![Docker image layer stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-layer-stack.png)


1. **Read-only** — once created, a layer never changes
2. **Content-addressable** — identified by a SHA256 hash of their contents
3. **Shared** — if two images use the same base layer, it's stored only once on disk
4. **Stacked** — a union filesystem combines them into a single coherent view

Here's how layers work conceptually. Imagine building an image with this Dockerfile:

```dockerfile
FROM ubuntu:22.04          # Layer 1: Ubuntu base filesystem
RUN apt-get update         # Layer 2: Updated package lists
RUN apt-get install -y curl # Layer 3: curl binary + dependencies
COPY app.py /app/app.py    # Layer 4: Your application file
CMD ["python3", "/app/app.py"]  # Metadata only (no new layer)
```

Each instruction that modifies the filesystem creates a new layer. The `CMD` instruction only sets metadata — it doesn't change any files, so it doesn't create a layer.

When a container starts from this image, Docker adds one more layer on top:

```text
[Writable Container Layer]  ← your runtime changes go here
[Layer 4: COPY app.py]      ← read-only
[Layer 3: apt install curl]  ← read-only
[Layer 2: apt update]        ← read-only
[Layer 1: ubuntu:22.04]      ← read-only
```

If you modify a file from a lower layer inside the container, the union filesystem uses **copy-on-write**: it copies the file up to the writable layer, then modifies the copy. The original in the read-only layer remains unchanged.

## Pulling an Image: What Actually Downloads

Let's trace what happens during `docker pull nginx`:

![Union filesystem](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-union-fs.png)


```bash
docker pull nginx
```

```text
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

Six layers downloaded. Each `Pull complete` line is a separate layer. Docker downloaded them in parallel (you'd see progress bars in a real terminal). The `Digest` is the SHA256 hash of the image manifest — it uniquely identifies this exact combination of layers.

Now pull another image that shares the same base:

```bash
docker pull nginx:alpine
```

```text
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

Notice `59bf1c3509f3: Already exists`. Docker recognized that it already had this layer (shared with another image, likely the Alpine base) and skipped downloading it. This is layer sharing in action — it saves both bandwidth and disk space.

## Inspecting Image Layers


![Layer sharing between containers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-layer-sharing.png)


![Container registry as a futuristic vending machine dispensin](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/02-container-registry-as-a-futuristic-vending-machine-dispensin.jpg)

### docker history

The `docker history` command shows each layer in an image, what instruction created it, and how large it is:

```bash
docker history nginx
```

```text
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

Reading from bottom to top (oldest layer first):

1. `ADD file:756...` — 74.8 MB — the Debian base filesystem
2. The big `set -x && addgroup...` — 61.1 MB — nginx installation
3. Several `COPY` instructions — a few KB each — configuration files
4. `ENV`, `EXPOSE`, `CMD`, etc. — 0 bytes — metadata only

The `<missing>` in the IMAGE column means these intermediate layers don't have their own image tags. Only the final layer (the top one) has the image ID `61395b4c586d`.

### docker image inspect

For detailed metadata in JSON format:

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

These are the content-addressable SHA256 hashes of each layer. Docker uses these to determine if a layer is already present locally.

## Image Naming and Registries

Docker images follow a naming convention:

![Build cache mechanism](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/02-build-cache.png)


```text
[registry/][namespace/]repository[:tag][@digest]
```

Examples:

| Full Name | Registry | Namespace | Repository | Tag |
|-----------|----------|-----------|------------|-----|
| `nginx` | docker.io (implicit) | library (implicit) | nginx | latest (implicit) |
| `nginx:1.25` | docker.io | library | nginx | 1.25 |
| `ubuntu:22.04` | docker.io | library | ubuntu | 22.04 |
| `myuser/myapp:v2` | docker.io | myuser | myapp | v2 |
| `gcr.io/project/app:prod` | gcr.io | project | app | prod |
| `ghcr.io/owner/repo:sha-abc123` | ghcr.io | owner | repo | sha-abc123 |
| `registry.example.com:5000/team/svc:latest` | registry.example.com:5000 | team | svc | latest |

Key rules:

- **Omitting the registry** defaults to `docker.io` (Docker Hub)
- **Omitting the tag** defaults to `latest` (a convention, not a guarantee of being the newest)
- **Official images** on Docker Hub have no namespace (e.g., `nginx`, `ubuntu`, `python`)
- **User images** have a namespace (e.g., `myuser/myapp`)
- **Digests** (`@sha256:...`) are immutable — tags can be moved to point to different images, but digests are permanent

### Docker Hub

Docker Hub is the default public registry. When you run `docker pull nginx`, Docker contacts `registry-1.docker.io` to download the image.

```bash
# Search Docker Hub from the CLI
docker search python --limit 5
```

```text
NAME                  DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
python                Python is an interpreted, interactive, objec…   9283      [OK]
pypy                  PyPy is a fast, compliant alternative implem…   380       [OK]
circleci/python       Python is an interpreted, interactive, objec…   55
cimg/python           The CircleCI Python Docker Convenience Image    10
bitnami/python        Bitnami Python Docker Image                     25                   [OK]
```

### Private Registries

You can run your own registry or use cloud-provided ones:

```bash
# Log in to a private registry
docker login registry.example.com

# Tag an image for a private registry
docker tag myapp:latest registry.example.com/team/myapp:v1.0

# Push to the private registry
docker push registry.example.com/team/myapp:v1.0

# Pull from the private registry
docker pull registry.example.com/team/myapp:v1.0
```

Common private registries:

| Registry | Provider |
|----------|----------|
| Amazon ECR | AWS |
| Google Artifact Registry | GCP |
| Azure Container Registry | Azure |
| GitHub Container Registry (ghcr.io) | GitHub |
| Docker Hub (private repos) | Docker |
| Harbor | Self-hosted (CNCF) |
| JFrog Artifactory | JFrog |

## Image Size: Why It Matters

Image size affects:

- **Pull time** — larger images take longer to download, slowing deployments
- **Build time** — larger layers take longer to push
- **Disk space** — each node stores images locally
- **Security surface** — more files mean more potential vulnerabilities
- **Cold start** — serverless platforms (AWS Lambda, Cloud Run) are slower with bigger images

Let's compare base image sizes:

```bash
docker pull ubuntu:22.04
docker pull debian:bookworm-slim
docker pull alpine:3.18
docker pull gcr.io/distroless/static-debian12
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

```text
REPOSITORY                          TAG                 SIZE
ubuntu                              22.04               77.8MB
debian                              bookworm-slim       74.8MB
alpine                              3.18                7.34MB
gcr.io/distroless/static-debian12   latest              2.45MB
```

| Base Image | Size | Shell | Package Manager | Use Case |
|-----------|------|-------|-----------------|----------|
| `ubuntu:22.04` | 77.8 MB | bash | apt | Development, debugging, familiarity |
| `debian:bookworm-slim` | 74.8 MB | bash | apt | Production (official images use this) |
| `alpine:3.18` | 7.34 MB | sh | apk | Minimal production, size-sensitive |
| `distroless/static` | 2.45 MB | No | No | Statically compiled binaries only |
| `scratch` | 0 MB | No | No | Bare minimum (Go binaries, etc.) |

Alpine is 10x smaller than Ubuntu. Distroless is 30x smaller. The tradeoff: smaller images have fewer debugging tools. You can't `docker exec -it container bash` into a distroless container because bash doesn't exist.

We'll explore optimization strategies in detail in the next article on Dockerfiles.

## Exporting and Importing Images

### docker save / docker load

These commands work with image tar archives — useful for transferring images without a registry:

```bash
# Save an image to a tar file
docker save nginx:latest -o nginx-latest.tar

# Check the file size
ls -lh nginx-latest.tar
# Output: -rw------- 1 user user 188M Sep 14 10:30 nginx-latest.tar

# Load an image from a tar file (on another machine)
docker load -i nginx-latest.tar
```

```text
Loaded image: nginx:latest
```

The tar file contains all layers as separate tar files plus the manifest:

```bash
tar tf nginx-latest.tar | head -20
```

```text
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

Each directory is a layer. Each `layer.tar` contains the filesystem changes for that layer.

### docker export / docker import

These work with containers (not images) and produce a flat filesystem:

```bash
# Create a container and make some changes
docker run --name test-export ubuntu:22.04 bash -c "echo 'hello' > /data.txt"

# Export the container's filesystem
docker export test-export -o test-export.tar

# Import as a new image
docker import test-export.tar my-custom-ubuntu:v1
```

The key difference:

| Operation | Works With | Preserves Layers? | Preserves Metadata? |
|-----------|-----------|-------------------|---------------------|
| `save/load` | Images | Yes | Yes (CMD, ENV, etc.) |
| `export/import` | Containers | No (flattens to single layer) | No |

Use `save/load` for moving images between machines. Use `export/import` only when you need a flat filesystem snapshot.

## Image Tagging

Tags are mutable labels that point to a specific image digest. You can create your own:

```bash
# Tag an existing image with a new name
docker tag nginx:latest my-nginx:v1.0
docker tag nginx:latest my-nginx:production
docker tag nginx:latest registry.example.com/team/nginx:v1.0

# List images — notice they share the same IMAGE ID
docker images | grep -E "nginx|my-nginx"
```

```text
REPOSITORY                        TAG          IMAGE ID       CREATED       SIZE
nginx                             latest       61395b4c586d   2 weeks ago   187MB
my-nginx                          v1.0         61395b4c586d   2 weeks ago   187MB
my-nginx                          production   61395b4c586d   2 weeks ago   187MB
registry.example.com/team/nginx   v1.0         61395b4c586d   2 weeks ago   187MB
```

All four entries point to the same image (`61395b4c586d`). No data is duplicated. Tags are just pointers.

### The "latest" Tag Trap

The `latest` tag is not special to Docker. It's a convention, not a mechanism. Docker does not automatically point `latest` to the newest version. If someone pushes `myapp:v2` without also updating `latest`, then `latest` still points to whatever it was before.

Best practice: always use specific tags in production.

```bash
# Bad — which version is this?
docker pull myapp:latest

# Good — explicit and reproducible
docker pull myapp:v2.3.1

# Best — immutable, can never change
docker pull myapp@sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4
```

## Cleaning Up


![Docker image layers as geological rock strata each layer a d](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/docker-containers/02-docker-image-layers-as-geological-rock-strata-each-layer-a-d.jpg)

Images accumulate quickly. Here's how to reclaim disk space:

```bash
# See disk usage
docker system df
```

```text
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          5         2         450.2MB   312.4MB (69%)
Containers      3         1         12.5kB    12.5kB (100%)
Local Volumes   2         1         256MB     128MB (50%)
Build Cache     15        0         1.2GB     1.2GB
```

```bash
# Remove a specific image
docker rmi nginx:alpine

# Remove all unused images (not referenced by any container)
docker image prune

# Remove ALL images not used by running containers (aggressive)
docker image prune -a

# Nuclear option — remove everything unused (images, containers, volumes, networks)
docker system prune -a --volumes
```

The `docker system prune -a --volumes` command is destructive. It removes all stopped containers, all unused networks, all images without at least one running container, and all volumes not used by at least one container. Use it on development machines, not production.

## Inspecting What's Inside an Image

Sometimes you want to see the contents of an image without running a container:

```bash
# Create a container without starting it
docker create --name peek nginx:latest

# Copy a file out
docker cp peek:/etc/nginx/nginx.conf ./nginx.conf

# Or explore the filesystem
docker export peek | tar tf - | head -30
```

```text
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
# Clean up
docker rm peek
```

You can also use third-party tools like `dive` to interactively browse layers:

```bash
# Install dive (https://github.com/wagoodman/dive)
# Then analyze an image
dive nginx:latest
```

`dive` shows each layer's contents, lets you see which files were added/modified/removed in each layer, and estimates wasted space.

## How Layers Are Stored on Disk

On a Linux host, Docker stores everything under `/var/lib/docker/`. The exact structure depends on the storage driver (usually OverlayFS):

```bash
# View the storage driver info
docker info --format '{{.Driver}}'
# Output: overlay2

# See where layers are stored
ls /var/lib/docker/overlay2/ | head -5
```

```text
2edcec3590a4ec7f40cf0743c15d78fb39d8326bc029073b41ef9727da6c851f
backingFsBlockDev
e379e8aedd4d72bb4c529a4ca07a4e4d230b5a1d3f7ea8d4e0cfbbe85a1c0e10
l
```

Each directory under `overlay2/` is a layer. The `l/` directory contains shortened symbolic links for layer identification. Don't modify these files directly — let Docker manage them.

## Multi-Architecture Images

Modern Docker images often support multiple CPU architectures in a single tag:

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

When you `docker pull nginx` on an ARM Mac, Docker automatically selects the `arm64` variant. On an x86_64 Linux server, it selects `amd64`. Same tag, different binary — this is why cross-platform deployment works seamlessly.

## What's Next

You now understand that images are stacks of read-only layers, that layers are shared between images, and that containers add a thin writable layer on top. You know how to inspect, export, tag, and clean up images.

The next step is building your own images. That means writing Dockerfiles — and the difference between a naive Dockerfile and an optimized one can be the difference between a 1.5 GB image that takes 10 minutes to build and a 50 MB image that builds in 30 seconds. The next article covers every Dockerfile instruction and the patterns that separate development Dockerfiles from production ones.
