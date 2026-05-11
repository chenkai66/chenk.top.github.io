---
title: "Docker and Containers (8): Beyond Docker — Kubernetes, Swarm, and What Comes Next"
date: 2023-06-30 09:00:00
tags:
  - Docker
  - Containers
  - Kubernetes
  - Docker Swarm
  - Orchestration
categories:
  - Docker and Containers
series: docker-containers
lang: en
description: "Single-host Docker breaks down at scale. This article previews container orchestration — Docker Swarm for simplicity, Kubernetes for everything else — and maps out the broader cloud-native ecosystem."
disableNunjucks: true
series_order: 8
translationKey: "docker-containers-8"
---

Everything in this series so far has been single-host Docker: one machine running containers. This works well for development, small projects, and applications with modest traffic. But the moment you need your service to survive a server failure, handle traffic spikes, or deploy updates without downtime, single-host Docker shows its limits. Container orchestration solves these problems — and Kubernetes has become the de facto answer.

## Why Single-Host Docker Isn't Enough

Consider what happens when your Docker host fails:

| Problem | Single-Host Docker | With Orchestration |
|---------|-------------------|-------------------|
| Server crashes | All containers die, manual restart | Containers automatically rescheduled to healthy nodes |
| Traffic spike | Scale up manually with `--scale` | Auto-scaling based on metrics |
| Deployment | `docker compose down && up` (downtime) | Rolling update with zero downtime |
| Service discovery | Custom network DNS (single host only) | Cluster-wide DNS, load balancing |
| Secret rotation | Restart containers with new env vars | Rolling secret rotation, no restart |
| Resource allocation | Hope you have enough RAM | Scheduler places containers optimally |
| Monitoring | `docker stats` on one host | Cluster-wide metrics, alerting |
| Storage | Local volumes (lost if host dies) | Persistent volumes with replication |

These are operational problems, not Docker problems. Docker does exactly what it's designed to do: run containers on a single host. Orchestrators add the multi-host coordination layer.

## Docker Swarm: The Simple Path

Docker Swarm is Docker's built-in orchestration. If you know `docker compose`, you already know 80% of Swarm. It uses the same YAML format and similar commands.

### Initializing a Swarm

```bash
# On the first node (becomes the manager)
docker swarm init --advertise-addr 192.168.1.10
```

```
Swarm initialized: current node (abc123def456) is now a manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token SWMTKN-1-0123456789abcdef-worker-token 192.168.1.10:2377

To add a manager to this swarm, run:

    docker swarm join-token manager
```

```bash
# On other nodes (join as workers)
docker swarm join --token SWMTKN-1-0123456789abcdef-worker-token 192.168.1.10:2377
```

```
This node joined a swarm as a worker.
```

```bash
# Check the cluster
docker node ls
```

```
ID                           HOSTNAME   STATUS   AVAILABILITY   MANAGER STATUS   ENGINE VERSION
abc123def456 *               manager1   Ready    Active         Leader           24.0.6
def456abc789                 worker1    Ready    Active                          24.0.6
ghi789def012                 worker2    Ready    Active                          24.0.6
```

Three-node cluster in four commands. That's Swarm's appeal.

### Deploying Services

Swarm uses the concept of "services" — a service is a definition of how to run containers, and Swarm manages the desired number of replicas:

![Deployment strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/08-deployment-strategies.png)


```bash
# Create a service with 3 replicas
docker service create \
    --name web \
    --replicas 3 \
    --publish 80:80 \
    --update-delay 10s \
    --update-parallelism 1 \
    nginx:alpine

# Check the service
docker service ls
```

```
ID             NAME   MODE         REPLICAS   IMAGE          PORTS
a1b2c3d4e5f6   web    replicated   3/3        nginx:alpine   *:80->80/tcp
```

```bash
# See where replicas are placed
docker service ps web
```

```
ID             NAME    IMAGE          NODE       DESIRED STATE   CURRENT STATE           
b2c3d4e5f6a7   web.1   nginx:alpine   manager1   Running         Running 30 seconds ago
c3d4e5f6a7b8   web.2   nginx:alpine   worker1    Running         Running 30 seconds ago
d4e5f6a7b8c9   web.3   nginx:alpine   worker2    Running         Running 30 seconds ago
```

Swarm distributed the three replicas across all three nodes. It also provides built-in load balancing — any node in the swarm can accept traffic on port 80, and Swarm routes it to a container running the service.

### Rolling Updates

```bash
# Update the image (rolling update with 10s delay between containers)
docker service update --image nginx:1.25-alpine web
```

```
web
overall progress: 3 out of 3 tasks
1/3: running   [==================================================>]
2/3: running   [==================================================>]
3/3: running   [==================================================>]
verify: Service converged
```

Swarm updates one container at a time, waiting 10 seconds between each. If a new container fails its healthcheck, Swarm rolls back automatically.

### Deploying a Stack (Compose in Swarm)

You can deploy a compose file directly to Swarm:

```yaml
# docker-compose.yml
services:
  web:
    image: myapp:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    ports:
      - "8080:8080"
    networks:
      - app-net

  redis:
    image: redis:7-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
    volumes:
      - redis-data:/data
    networks:
      - app-net

networks:
  app-net:
    driver: overlay

volumes:
  redis-data:
```

```bash
# Deploy the stack
docker stack deploy -c docker-compose.yml myapp

# Check stack services
docker stack services myapp
```

```
ID             NAME           MODE         REPLICAS   IMAGE             PORTS
a1b2c3d4e5f6   myapp_web      replicated   3/3        myapp:latest      *:8080->8080/tcp
b2c3d4e5f6a7   myapp_redis    replicated   1/1        redis:7-alpine
```

```bash
# Remove the stack
docker stack rm myapp
```

### Swarm Secrets and Configs

Swarm has native support for secrets and config files:

```bash
# Create a secret
echo "supersecret" | docker secret create db_password -

# Create a config
docker config create nginx_conf ./nginx.conf

# Use them in a service
docker service create \
    --name api \
    --secret db_password \
    --config source=nginx_conf,target=/etc/nginx/nginx.conf \
    myapp
```

Inside the container, secrets appear as files in `/run/secrets/`:

```bash
cat /run/secrets/db_password
# Output: supersecret
```

### When Swarm Makes Sense

Swarm is a good choice when:
- You have a small team (< 5 engineers)
- You have a small cluster (< 10 nodes)
- You want orchestration without the Kubernetes learning curve
- You're already using Docker Compose and want a smooth migration
- You don't need auto-scaling, custom schedulers, or the CNCF ecosystem

## Kubernetes: The Industry Standard

Kubernetes (K8s) is the dominant container orchestration platform. It's more complex than Swarm but dramatically more capable. Most cloud providers offer managed Kubernetes services (EKS, GKE, AKS, ACK), eliminating the operational burden of managing the control plane.

![Kubernetes architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/08-k8s-architecture.png)


### Architecture

Kubernetes clusters have two types of nodes:

**Control Plane (master) components:**

| Component | Role |
|-----------|------|
| `kube-apiserver` | REST API that all components and users interact with |
| `etcd` | Distributed key-value store for all cluster state |
| `kube-scheduler` | Decides which node to place new pods on |
| `kube-controller-manager` | Runs controllers (deployment, replicaset, node, etc.) |
| `cloud-controller-manager` | Integrates with cloud provider APIs (optional) |

**Worker node components:**

| Component | Role |
|-----------|------|
| `kubelet` | Agent that manages pods on the node, communicates with API server |
| `kube-proxy` | Network proxy for service routing |
| Container runtime | Runs containers (containerd, CRI-O — not Docker daemon) |

The architecture looks like this (described, not drawn):

```
Control Plane:
  API Server ←→ etcd (cluster state)
      ↑
  Scheduler + Controller Manager (watch API, make decisions)

Worker Nodes (N):
  kubelet ←→ API Server (reports status, receives instructions)
  kube-proxy (manages iptables/IPVS rules for service routing)
  containerd (runs containers)
```

### Core Kubernetes Objects

Everything in Kubernetes is a declarative object — you describe the desired state, and Kubernetes works to make it real.

#### Pod

The smallest deployable unit. A pod contains one or more containers that share network and storage:

![Pod lifecycle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/08-pod-lifecycle.png)


```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  containers:
    - name: app
      image: myapp:v1.0
      ports:
        - containerPort: 8080
      resources:
        requests:
          memory: "128Mi"
          cpu: "250m"
        limits:
          memory: "256Mi"
          cpu: "500m"
      livenessProbe:
        httpGet:
          path: /health
          port: 8080
        initialDelaySeconds: 10
        periodSeconds: 30
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 10
```

You rarely create pods directly. Instead, you use higher-level objects.

#### Deployment

A Deployment manages a set of identical pods and handles rolling updates:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: app
          image: myapp:v1.0
          ports:
            - containerPort: 8080
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
          resources:
            requests:
              memory: "128Mi"
              cpu: "250m"
            limits:
              memory: "256Mi"
              cpu: "500m"
```

```bash
# Apply the deployment
kubectl apply -f deployment.yaml

# Check the deployment
kubectl get deployments
```

```
NAME     READY   UP-TO-DATE   AVAILABLE   AGE
my-app   3/3     3            3           60s
```

```bash
# Scale the deployment
kubectl scale deployment my-app --replicas=5

# Update the image (triggers rolling update)
kubectl set image deployment/my-app app=myapp:v2.0

# Watch the rollout
kubectl rollout status deployment/my-app
```

```
Waiting for deployment "my-app" rollout to finish: 2 out of 3 new replicas have been updated...
Waiting for deployment "my-app" rollout to finish: 1 old replicas are pending termination...
deployment "my-app" successfully rolled out
```

```bash
# Rollback if something goes wrong
kubectl rollout undo deployment/my-app
```

#### Service

A Service provides a stable endpoint (DNS name and IP) for a set of pods:

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
```

Service types:

| Type | Accessibility | Use Case |
|------|-------------|----------|
| `ClusterIP` | Internal cluster only | Internal microservices |
| `NodePort` | External via `<NodeIP>:<NodePort>` | Development, simple exposure |
| `LoadBalancer` | External via cloud load balancer | Production web services |
| `ExternalName` | DNS CNAME to external service | Accessing external databases |

#### ConfigMap and Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"
  config.yaml: |
    server:
      port: 8080
      timeout: 30s

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  # base64 encoded values
  url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0Bob3N0OjU0MzIvZGI=
  password: c3VwZXJzZWNyZXQ=
```

Use them in a pod:

```yaml
spec:
  containers:
    - name: app
      env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: LOG_LEVEL
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
      volumeMounts:
        - name: config
          mountPath: /app/config
  volumes:
    - name: config
      configMap:
        name: app-config
```

### Essential kubectl Commands

```bash
# Get resources
kubectl get pods                    # List pods
kubectl get pods -o wide            # More details (node, IP)
kubectl get deployments             # List deployments
kubectl get services                # List services
kubectl get all                     # List everything

# Describe (detailed info + events)
kubectl describe pod my-app-abc123
kubectl describe deployment my-app

# Logs
kubectl logs my-app-abc123              # Pod logs
kubectl logs my-app-abc123 -f           # Follow logs
kubectl logs my-app-abc123 -c sidecar   # Specific container in multi-container pod
kubectl logs -l app=my-app              # Logs from all pods with a label

# Execute commands
kubectl exec -it my-app-abc123 -- bash
kubectl exec my-app-abc123 -- cat /app/config.yaml

# Port forwarding (access a pod locally without a service)
kubectl port-forward my-app-abc123 8080:8080

# Apply/delete configurations
kubectl apply -f deployment.yaml
kubectl delete -f deployment.yaml

# Debug
kubectl get events --sort-by=.metadata.creationTimestamp
kubectl top pods                        # Resource usage (requires metrics-server)
kubectl top nodes
```

## Kubernetes vs Docker Swarm

| Feature | Docker Swarm | Kubernetes |
|---------|-------------|------------|
| Setup complexity | Minutes | Hours (managed services simplify this) |
| Learning curve | Low (Docker CLI knowledge transfers) | Steep (new concepts, YAML-heavy) |
| Scaling | Manual (`docker service scale`) | Manual + Horizontal Pod Autoscaler |
| Rolling updates | Built-in, simple | Built-in, highly configurable |
| Service discovery | Docker DNS | CoreDNS, Services |
| Load balancing | Built-in (routing mesh) | Services, Ingress controllers |
| Secret management | Docker secrets | Kubernetes Secrets (+ external integrations) |
| Storage | Docker volumes | PersistentVolumes, StorageClasses, CSI drivers |
| Networking | Overlay networks | CNI plugins (Calico, Cilium, Flannel, etc.) |
| Health checks | HEALTHCHECK instruction | Liveness, readiness, startup probes |
| Package management | None | Helm charts |
| Community/ecosystem | Small, declining | Massive, CNCF ecosystem |
| Managed offerings | Few | EKS, GKE, AKS, ACK, and many more |
| Best for | Small teams, simple deployments | Production at scale, microservices |

### When You DON'T Need Orchestration

Not every application needs Kubernetes. Be honest about your requirements:

![Orchestration tools comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/08-orchestration-comparison.png)


| Your Situation | Recommendation |
|---------------|---------------|
| Single server, few services | Docker Compose |
| Small team, < 5 services | Docker Compose or Swarm |
| Need zero-downtime deploys on single host | Docker Compose with rolling restart |
| Serverless workloads | Cloud Functions (Lambda, Cloud Run) |
| Batch processing | Docker Compose or single-host scheduler |
| Multi-region, high availability | Kubernetes (managed) |
| Microservices architecture | Kubernetes |
| Compliance requires orchestration | Kubernetes |
| Team > 10 engineers | Kubernetes |

A common mistake is adopting Kubernetes for a three-service application that runs on a single $20/month VPS. The operational overhead of Kubernetes (even managed) exceeds the benefit until you're at a certain scale.

If you're running on a single host and want better deployment workflows, look at tools like:

- **Docker Compose** with a simple CI/CD pipeline
- **Kamal** (from Basecamp) — zero-downtime deploys to bare servers
- **Dokku** — a self-hosted PaaS (like a private Heroku)
- **Coolify** — an open source and self-hostable alternative to Heroku/Netlify/Vercel

## The Cloud-Native Ecosystem

Kubernetes spawned an ecosystem of tools. Here's a map of the most important ones:

### Package Management: Helm

Helm is the package manager for Kubernetes. A Helm "chart" bundles all the YAML files for an application:

```bash
# Install a Helm chart (e.g., PostgreSQL)
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install my-postgres bitnami/postgresql --set auth.postgresPassword=secret

# List installed releases
helm list

# Upgrade a release
helm upgrade my-postgres bitnami/postgresql --set auth.postgresPassword=newsecret

# Uninstall
helm uninstall my-postgres
```

Helm charts are like Docker images for entire application stacks — pre-packaged, versioned, and shareable.

### Service Mesh: Istio and Linkerd

A service mesh adds observability, security, and traffic management between microservices:

| Feature | Without Service Mesh | With Service Mesh |
|---------|---------------------|-------------------|
| mTLS between services | Manual certificate management | Automatic, transparent |
| Traffic splitting | Application-level | Declarative (80/20 canary) |
| Retry policies | Code in each service | Configurable per-route |
| Observability | Each service adds instrumentation | Automatic tracing, metrics |
| Access control | Application-level auth | Policy-based (YAML) |

Istio is feature-rich but complex. Linkerd is simpler and lighter. Neither is needed unless you have many (10+) communicating services and need fine-grained traffic control.

### GitOps: ArgoCD and Flux

GitOps treats your Git repository as the source of truth for cluster state:

1. You push a change to Git (e.g., update an image tag in a Deployment YAML)
2. ArgoCD detects the change and syncs the cluster to match
3. The cluster converges to the desired state

```bash
# ArgoCD watches a Git repo and applies changes automatically
argocd app create my-app \
    --repo https://github.com/myorg/myapp-k8s.git \
    --path kubernetes/ \
    --dest-server https://kubernetes.default.svc \
    --dest-namespace production
```

Benefits:
- Every change is auditable (Git history)
- Rollback = `git revert`
- No manual `kubectl apply` in production
- Cluster state is always reproducible from Git

### Monitoring and Observability

| Tool | Purpose | Collects |
|------|---------|----------|
| Prometheus | Metrics collection and alerting | CPU, memory, request rates, custom metrics |
| Grafana | Visualization and dashboards | Displays Prometheus data (and others) |
| Jaeger / Zipkin | Distributed tracing | Request paths across microservices |
| Fluentd / Fluent Bit | Log aggregation | Container logs → central storage |
| Elasticsearch + Kibana | Log storage and search | Searchable log index |

The "standard" open-source observability stack for Kubernetes is Prometheus + Grafana + Fluentd (or Fluent Bit) + Jaeger, though many alternatives exist.

### Container Security in Kubernetes

| Tool | Purpose |
|------|---------|
| Trivy | Image vulnerability scanning |
| Falco | Runtime security monitoring (detect anomalous container behavior) |
| OPA/Gatekeeper | Policy enforcement (e.g., "no containers may run as root") |
| cert-manager | Automatic TLS certificate management (Let's Encrypt) |
| Kyverno | Kubernetes-native policy engine |

## From Docker to Production: A Typical Path

Here's a realistic progression for a team growing from a side project to a production service:

![Docker to Kubernetes mapping](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/docker-containers/08-docker-to-k8s.png)


| Stage | Infrastructure | Deployment |
|-------|---------------|------------|
| 1. Local development | Docker Compose on laptop | `docker compose up` |
| 2. Single server | Docker Compose on VPS | `git pull && docker compose up -d` |
| 3. CI/CD pipeline | Docker Compose + GitHub Actions | Auto-deploy on push to main |
| 4. Multi-server | Docker Swarm or managed K8s | `docker stack deploy` or `kubectl apply` |
| 5. Production at scale | Managed Kubernetes (EKS/GKE/AKS/ACK) | Helm + ArgoCD |
| 6. Multi-region | Managed K8s + service mesh | GitOps + traffic management |

Most teams never need to go past stage 3 or 4. Don't jump to stage 5 because it sounds impressive — jump when the problems at your current stage justify the complexity.

## Key Takeaways from This Series

Looking back across all eight articles, here are the principles that matter most:

**Containers are processes, not VMs.** They share the host kernel and use namespaces + cgroups for isolation. Understanding this shapes how you think about security, performance, and debugging.

**Images are layers.** Layer caching drives build performance. Instruction order matters. Multi-stage builds separate build-time dependencies from runtime.

**Networks and volumes are the connective tissue.** Custom bridge networks provide DNS-based service discovery. Named volumes persist data independently of container lifecycle.

**Compose is the developer's interface.** A single YAML file replaces dozens of `docker run` commands. It's version-controlled, shareable, and deterministic.

**Security is opt-in.** Docker's defaults favor convenience. Running as non-root, dropping capabilities, using read-only filesystems, and scanning images are all things you must explicitly do.

**Orchestration is a spectrum.** Docker Compose for single-host, Swarm for simple multi-host, Kubernetes for production at scale. Choose the simplest tool that solves your actual problems.

The container ecosystem moves fast, but the fundamentals in this series are stable. Namespaces haven't changed since 2013. The OCI image format is settled. Kubernetes API objects have been stable for years. Learn these foundations, and the tools built on top of them will make sense.
