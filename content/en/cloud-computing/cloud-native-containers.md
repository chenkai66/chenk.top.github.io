---
title: "Cloud Computing (3): Cloud-Native and Container Technologies"
date: 2023-03-11 09:00:00
tags:
  - Cloud Computing
  - Cloud Native
  - Docker
  - Kubernetes
  - Microservices
categories: Cloud Computing
series: cloud-computing
lang: en
mathjax: false
description: "Why cloud-native exists, what containers actually do at the kernel level, how Kubernetes really works, when service mesh is worth its weight, and how the whole stack fits together in production."
disableNunjucks: true
series_order: 3
series_total: 8
translationKey: "cloud-computing-3"
---
![Cloud Computing (3): Cloud-Native and Container Technologies — Chapter overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/cloud-native-containers/illustration_1.png)

The shift from monolithic applications to cloud-native architectures is one of the most consequential changes in software engineering this decade. The headline — containers and Kubernetes — is well known. The interesting story is *why* this stack won, what each layer actually does, and where the seams are that determine whether your platform feels effortless or feels like a maze.

This article walks the cloud-native stack from first principles. We start with the architectural shift that motivates everything else, then dig into what a container really is at the Linux kernel level, climb up to Kubernetes orchestration, examine when a service mesh earns its complexity, and finish with packaging and delivery via Helm and GitOps. Examples are deliberately concrete: copy-pastable Dockerfiles, real manifests, and the trade-offs that matter when you run this in production.


---

## What You Will Learn

- The 12-Factor App methodology and *why* each factor exists
- Containers from the inside: namespaces, cgroups, union filesystems, and image layering
- Docker production essentials: multi-stage builds, security, Compose for local dev
- Kubernetes architecture: how the control plane drives worker nodes via the reconciliation loop
- Workload primitives: Pods, Services, Deployments, StatefulSets, DaemonSets, Jobs
- Networking: CNI plugins, NetworkPolicy, Ingress, and when Istio service mesh pays for itself
- Storage: PV/PVC dynamic provisioning and what `ReadWriteMany` actually costs
- Helm packaging, release history, and how rollbacks really work
- Microservices patterns: circuit breakers, sagas, API gateways
- GitOps with ArgoCD and the operational discipline it forces

## Prerequisites

- Comfortable with the Linux command line and basic networking (routing, DNS, TCP)
- Understanding of HTTP/REST and how web apps and databases talk to each other
- Parts 1-6 of this series (especially [Virtualization](/en/cloud-computing/virtualization/), [Networking](/en/cloud-computing/networking-sdn/), and [DevOps](/en/cloud-computing/operations-devops/)) provide useful background

---

## Cloud-Native: What Changed and Why

Cloud-native is not "running stuff in the cloud." A lift-and-shifted VM is in the cloud but not cloud-native. The CNCF definition is precise:

> Cloud-native technologies empower organizations to build and run scalable applications in modern, dynamic environments such as public, private, and hybrid clouds. Containers, service meshes, microservices, immutable infrastructure, and declarative APIs exemplify this approach.

Three ideas do most of the work behind that sentence:

1. **Immutable infrastructure.** Servers are not pets you patch; they are cattle you replace. A new release is a new image, never an in-place edit. This eliminates configuration drift, the source of half of all production incidents.
2. **Declarative APIs.** You describe the *desired* state ("I want 3 replicas of v1.4 with 500 MB memory each") and the platform makes reality match. The opposite — imperative scripts that say "do step 1, then step 2" — breaks the moment reality differs from the script's assumptions.
3. **Loose coupling at every layer.** Services are independent. So are deploys. So are failures. So are scaling decisions. The cost is more moving parts; the benefit is that no single moving part can break everything.

### Monolith vs Microservices: The Trade-off Made Visible

![Monolith vs Microservices Architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/cloud-native-containers/fig1_microservices_vs_monolith.png)

The diagram above shows the structural difference, but the real story is in four numbers:

| Dimension | Monolith | Microservices |
|---|---|---|
| **Deploy unit** | 1 binary | N independent services |
| **Scale unit** | Whole app | Each service independently |
| **Tech stack** | One language/runtime | Polyglot per service |
| **Failure blast radius** | 100% | 1 service (with circuit breakers) |

Microservices are not strictly better. They trade simplicity for independence: you pay with distributed systems complexity (network failures, eventual consistency, distributed tracing, contract versioning) to gain the ability to deploy, scale, and fail independently. **The decision rule:** if your team is small enough to fit in two pizzas and your release cadence is monthly, a well-structured monolith is almost certainly the right answer. The threshold to introduce microservices is when coordination overhead between teams starts dominating engineering time.

### The 12-Factor App: A Survival Guide

The [12-Factor methodology](https://12factor.net/) (Heroku, 2011) predates Kubernetes but has become the default operational contract a containerized service is expected to honor. Each factor exists to make a specific failure mode impossible:

| # | Factor | Why it matters |
|---|---|---|
| 1 | **Codebase** — one repo, many deploys | Same code, different config = reliable promotion path |
| 2 | **Dependencies** — explicitly declared and isolated | "Works on my machine" becomes impossible |
| 3 | **Config** — in environment, not code | Same image runs in dev/staging/prod |
| 4 | **Backing services** — attached resources | Swap a DB by changing a URL, not refactoring |
| 5 | **Build, release, run** — strictly separated | A release is immutable and rollback-able |
| 6 | **Processes** — stateless and share-nothing | Any replica can serve any request |
| 7 | **Port binding** — self-contained | No assumed external server (Tomcat, IIS) |
| 8 | **Concurrency** — scale via process model | Horizontal scaling is the default |
| 9 | **Disposability** — fast startup, graceful shutdown | Auto-scaling and rolling updates work |
| 10 | **Dev/prod parity** — keep environments similar | Production surprises shrink |
| 11 | **Logs** — as event streams to stdout | Platform aggregates, you don't write to files |
| 12 | **Admin processes** — one-off in same env | Migrations don't have a separate stack |

Violating a factor is sometimes the right call (factor 6 is genuinely hard for stateful systems), but each violation is a debt you should know you took on.

## Containers: What They Actually Are

A common mental model is "containers are lightweight VMs." That mental model is wrong in important ways. **Containers are not virtualization; they are process isolation.** A container is just a Linux process (or process tree) where the kernel has been instructed to lie to it about what the system looks like.

Three Linux kernel features do the work:

1. **Namespaces** — give a process its own view of system resources (PID, network, mount, UTS, IPC, user, cgroup). Inside a PID namespace, your container sees itself as PID 1 and cannot see processes outside.
2. **cgroups (v2)** — enforce resource limits (CPU, memory, IO, PIDs). When you set `--memory=512m`, the kernel kills the process if it exceeds that limit.
3. **Union filesystems** (overlay2 today) — stack read-only image layers under a thin writable layer per container, enabling instant copy-on-write filesystem semantics.

That's it. A container shares the host kernel. There is no hypervisor, no second OS. The cost: ~50 ms startup vs ~30 s for a VM, ~5 MB overhead vs ~500 MB, and density of hundreds per host vs tens.

### Image Layers: The Cache That Makes Builds Fast

![Docker Image Layers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/cloud-native-containers/fig2_docker_layers.png)

Every Dockerfile instruction creates a new layer. Layers stack via the union filesystem; identical layers are deduplicated across images and across hosts. This is why two well-structured images that share a base can differ by megabytes even if the base is gigabytes.

Two practical consequences:

**1. Order Dockerfile instructions for cache reuse.** Put things that change rarely (system packages, language runtime) first; put things that change every commit (your app code) last. A cached build is seconds; a cold build is minutes.

![GitOps Deployment Pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/cloud-native-containers/fig_gitops_pipeline_en.png)

The key shift is **GitOps**: the cluster's state is defined by Git. ArgoCD (or Flux) continuously reconciles the cluster against a Git repo. Two big wins:

1. **Audit trail.** Every change is a commit. Want to know who changed prod at 2am? `git blame`.
2. **Disaster recovery.** Cluster gone? `kubectl apply` from the manifest repo and you're back.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata: { name: web, namespace: argocd }
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/k8s-manifests
    path: apps/web/overlays/prod
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: prod
  syncPolicy:
    automated: { prune: true, selfHeal: true }
    syncOptions: [CreateNamespace=true]
```

`selfHeal: true` means "if someone `kubectl edit`s a resource by hand, ArgoCD will revert it." That is the discipline GitOps enforces — the cluster's state is what's in Git, not what's in someone's terminal.

## Operating in Production: The Commands That Matter

```bash
# Cluster overview
kubectl cluster-info
kubectl get nodes -o wide

# What's running where
kubectl get pods -A -o wide
kubectl top pods -A                   # CPU/memory actuals

# Debug a failing pod
kubectl describe pod <name> -n <ns>   # events, status, scheduling
kubectl logs <name> -n <ns> -f        # stream logs
kubectl logs <name> -n <ns> --previous  # logs from the crashed previous container
kubectl exec -it <name> -n <ns> -- sh

# Recent cluster events (the goldmine for "why did this happen")
kubectl get events -A --sort-by='.lastTimestamp' | tail -30

# Scale and rollout
kubectl scale deploy/web --replicas=5
kubectl rollout restart deploy/web    # forces a fresh rollout, useful for picking up new secrets
```

The single most useful pair: `kubectl describe` (status, events, scheduling decisions) and `kubectl logs --previous` (what happened in the container that just crashed).

## Production Checklist

Before declaring a workload production-ready:

- [ ] Multi-stage Dockerfile, non-root user, distroless or minimal base
- [ ] Image pinned by digest (or at least immutable tag), signed, scanned in CI
- [ ] Resource requests *and* limits set on every container
- [ ] Liveness *and* readiness probes (readiness controls traffic, liveness controls restarts)
- [ ] PodDisruptionBudget so cluster maintenance doesn't take you below `minAvailable`
- [ ] HorizontalPodAutoscaler if traffic is variable
- [ ] NetworkPolicy with default-deny + explicit allows
- [ ] Secrets in an external store (Vault, AWS Secrets Manager, External Secrets Operator) not in plain Secrets
- [ ] Logs to stdout, structured (JSON), aggregated to a central system
- [ ] Metrics exposed (Prometheus format) and dashboards exist
- [ ] Distributed tracing instrumented (OpenTelemetry)
- [ ] Backups tested (especially for StatefulSets)
- [ ] Runbook exists for the common failure modes

A workload that ticks all these boxes is not unbreakable — but the failure modes that remain are the interesting ones, not the embarrassing ones.
