"""
Figure generation script for Cloud Computing Part 07: Cloud-Native & Containers.

Generates 5 figures used in both EN and ZH versions of the article. Each
figure teaches one concrete idea cleanly and is sized for blog reading.

Figures:
    fig1_microservices_vs_monolith
        Side-by-side comparison of monolithic and microservices architectures
        with deploy unit, scale unit, and failure blast radius annotations.

    fig2_docker_layers
        Anatomy of a Docker image: union FS layer stack on top of host kernel,
        with read-only image layers and the writable container layer.

    fig3_kubernetes_architecture
        Control plane (API server, etcd, scheduler, controller manager) with
        worker nodes (kubelet, kube-proxy, container runtime, pods) and the
        request flow between them.

    fig4_service_mesh_istio
        Two services with sidecar proxies (Envoy), mTLS data plane between
        them, and control plane (Istiod) issuing config + certs.

    fig5_helm_charts
        Helm chart -> values.yaml -> rendered manifests -> release in cluster,
        with rollback / upgrade history shown as a timeline.

Usage:
    python3 scripts/figures/cloud-computing/07-cloud-native.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "cloud-computing" / "cloud-native-containers"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "cloud-computing" / "cloud-native-containers"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, x, y, w, h, text, *, fc=C_BLUE, ec=None, tc="white",
         fontsize=9, weight="normal", radius=0.04):
    ec = ec or fc
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.1, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=tc, weight=weight)


def _arrow(ax, x1, y1, x2, y2, *, color=C_DARK, lw=1.4, style="->",
           rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: Monolith vs Microservices
# ---------------------------------------------------------------------------
def fig1_microservices_vs_monolith() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))

    # ----- Monolith -----
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Monolithic Architecture",
                 fontsize=13, weight="bold", color=C_DARK, pad=12)

    # Single large app
    _box(ax, 1.2, 2.2, 7.6, 6.0, "", fc=C_LIGHT, ec=C_GRAY, tc=C_DARK,
         radius=0.08)
    ax.text(5.0, 7.7, "Single Deployable Unit",
            ha="center", fontsize=10.5, weight="bold", color=C_DARK)

    modules = [
        ("Auth",    1.7, 5.5, C_BLUE),
        ("Catalog", 4.0, 5.5, C_PURPLE),
        ("Cart",    6.3, 5.5, C_GREEN),
        ("Payment", 1.7, 3.4, C_AMBER),
        ("Orders",  4.0, 3.4, C_BLUE),
        ("Search",  6.3, 3.4, C_PURPLE),
    ]
    for name, x, y, col in modules:
        _box(ax, x, y, 2.0, 1.4, name, fc=col, fontsize=10, weight="bold",
             radius=0.06)

    # Single DB
    _box(ax, 3.0, 0.4, 4.0, 1.2, "Shared Database",
         fc=C_DARK, fontsize=10, weight="bold")
    _arrow(ax, 5.0, 2.2, 5.0, 1.6, color=C_GRAY, lw=1.6)

    # Annotations
    ax.text(5.0, 9.2,
            "Deploy unit: 1   Scale unit: 1   Stack: 1   Blast radius: 100%",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")

    # ----- Microservices -----
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Microservices Architecture",
                 fontsize=13, weight="bold", color=C_DARK, pad=12)

    # API Gateway
    _box(ax, 3.0, 8.2, 4.0, 0.9, "API Gateway",
         fc=C_DARK, fontsize=10, weight="bold", radius=0.06)

    services = [
        ("Auth",    0.4, 5.6, C_BLUE,   "Postgres"),
        ("Catalog", 2.7, 5.6, C_PURPLE, "MongoDB"),
        ("Cart",    5.0, 5.6, C_GREEN,  "Redis"),
        ("Payment", 7.3, 5.6, C_AMBER,  "Postgres"),
    ]
    bottom_services = [
        ("Orders",  1.5, 2.6, C_BLUE,   "Postgres"),
        ("Search",  4.4, 2.6, C_PURPLE, "Elastic"),
        ("Notify",  7.3, 2.6, C_GREEN,  "Kafka"),
    ]
    all_svcs = services + bottom_services

    for name, x, y, col, db in all_svcs:
        _box(ax, x, y, 2.2, 1.0, name, fc=col, fontsize=10, weight="bold",
             radius=0.06)
        _box(ax, x + 0.2, y - 1.05, 1.8, 0.7, db, fc="white", ec=col,
             tc=col, fontsize=8.5, radius=0.05)
        _arrow(ax, x + 1.1, y, x + 1.1, y - 0.35, color=col, lw=1.0)

    # Gateway -> top services
    for _, x, y, col, _ in services:
        _arrow(ax, 5.0, 8.2, x + 1.1, y + 1.0, color=C_GRAY, lw=0.9, rad=0.1)

    # Inter-service hint
    _arrow(ax, 6.1, 5.6, 7.3, 3.6, color=C_GRAY, lw=0.9,
           style="->", rad=0.2)
    ax.text(7.4, 4.6, "async\nevents", fontsize=8, color=C_GRAY, style="italic")

    ax.text(5.0, 9.45,
            "Deploy unit: many   Scale per service   Polyglot   Blast radius: 1 svc",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")

    fig.suptitle("Monolith vs Microservices: Deploy, Scale, and Fail Independently",
                 fontsize=14.5, weight="bold", color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig1_microservices_vs_monolith")


# ---------------------------------------------------------------------------
# Figure 2: Docker container layers (Union FS)
# ---------------------------------------------------------------------------
def fig2_docker_layers() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10)
    ax.axis("off")

    # Host kernel base
    _box(ax, 1.0, 0.4, 10.0, 0.9, "Host Linux Kernel  (namespaces + cgroups + union FS)",
         fc=C_DARK, fontsize=10.5, weight="bold", radius=0.05)

    # Image (read-only) layers
    layers = [
        ("L1: FROM ubuntu:22.04          (base OS, 77 MB)",     1.5, C_BLUE),
        ("L2: RUN apt-get install python3 (50 MB)",             2.4, C_BLUE),
        ("L3: COPY requirements.txt      (1 KB)",               3.3, C_PURPLE),
        ("L4: RUN pip install -r req     (120 MB)",             4.2, C_PURPLE),
        ("L5: COPY ./app                 (5 MB)",               5.1, C_GREEN),
        ("L6: CMD [\"python\", \"app.py\"]  (metadata)",         6.0, C_GREEN),
    ]
    for label, y, col in layers:
        _box(ax, 1.0, y, 10.0, 0.75, label,
             fc=col, fontsize=10, weight="bold", radius=0.04)

    # Writable container layer
    _box(ax, 1.0, 7.0, 10.0, 0.85,
         "Container R/W layer (per-container, ephemeral, COW)",
         fc=C_AMBER, fontsize=10.5, weight="bold", radius=0.04)

    # Side annotations
    ax.annotate("Read-only\nimage layers\n(shared between\ncontainers)",
                xy=(11.05, 4.0), xytext=(11.5, 4.0),
                fontsize=9.5, color=C_BLUE, weight="bold", va="center", ha="left")
    ax.plot([11.0, 11.4], [3.3, 3.3], color=C_BLUE, lw=1.2)
    ax.plot([11.0, 11.4], [6.75, 6.75], color=C_BLUE, lw=1.2)
    ax.plot([11.4, 11.4], [3.3, 6.75], color=C_BLUE, lw=1.2)

    ax.annotate("Writable layer\n(only this is\nper-container)",
                xy=(11.05, 7.4), xytext=(11.5, 7.4),
                fontsize=9.5, color=C_AMBER, weight="bold", va="center", ha="left")

    # Top: three running containers sharing the same image
    for i, x in enumerate([1.5, 5.0, 8.5]):
        _box(ax, x, 8.4, 2.0, 1.0, f"Container {i+1}",
             fc=C_GREEN, ec=C_DARK, fontsize=10, weight="bold", radius=0.05)
        _arrow(ax, x + 1.0, 8.4, x + 1.0, 7.85, color=C_GRAY, lw=1.0)

    ax.set_title("Docker Image Layers: Stacked, Cached, Shared",
                 fontsize=14, weight="bold", color=C_DARK, pad=12, loc="left")
    ax.text(0, -0.4,
            "Cache hit: changing only L5 rebuilds 1 layer; L1-L4 reused across builds and containers.",
            fontsize=9, color=C_GRAY, style="italic", transform=ax.transData)

    fig.tight_layout()
    _save(fig, "fig2_docker_layers")


# ---------------------------------------------------------------------------
# Figure 3: Kubernetes architecture
# ---------------------------------------------------------------------------
def fig3_kubernetes_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 10)
    ax.axis("off")

    # ----- Control plane outer box -----
    _box(ax, 0.4, 5.8, 5.8, 4.0, "", fc="#eff6ff", ec=C_BLUE,
         tc=C_DARK, radius=0.04)
    ax.text(0.7, 9.45, "Control Plane (Master)",
            fontsize=11.5, weight="bold", color=C_BLUE)

    _box(ax, 0.8, 7.8, 2.4, 1.2, "API Server\n(kube-apiserver)",
         fc=C_BLUE, fontsize=9.5, weight="bold")
    _box(ax, 3.4, 7.8, 2.4, 1.2, "etcd\n(cluster state)",
         fc=C_PURPLE, fontsize=9.5, weight="bold")
    _box(ax, 0.8, 6.1, 2.4, 1.2, "Scheduler\n(pod placement)",
         fc=C_GREEN, fontsize=9.5, weight="bold")
    _box(ax, 3.4, 6.1, 2.4, 1.2, "Controller Mgr\n(reconcile loops)",
         fc=C_AMBER, fontsize=9.5, weight="bold")

    # API <-> etcd (only API talks to etcd)
    _arrow(ax, 3.2, 8.4, 3.4, 8.4, color=C_DARK, lw=1.4, style="<->")

    # Scheduler & CM <-> API
    _arrow(ax, 2.0, 7.3, 2.0, 7.8, color=C_DARK, lw=1.2, style="<->")
    _arrow(ax, 4.6, 7.3, 4.6, 7.8, color=C_DARK, lw=1.2, style="<->")

    # ----- User -----
    _box(ax, 0.4, 3.6, 2.0, 0.9, "kubectl / CI",
         fc=C_DARK, fontsize=10, weight="bold")
    _arrow(ax, 1.4, 4.5, 2.0, 7.8, color=C_GRAY, lw=1.2, rad=0.2, style="->")
    ax.text(1.6, 6.0, "REST/gRPC", fontsize=8.5, color=C_GRAY,
            rotation=68, va="center")

    # ----- Worker nodes outer box -----
    _box(ax, 6.6, 0.4, 6.2, 9.4, "", fc="#f5f3ff", ec=C_PURPLE,
         tc=C_DARK, radius=0.04)
    ax.text(6.9, 9.45, "Worker Nodes", fontsize=11.5, weight="bold",
            color=C_PURPLE)

    # Two nodes
    node_y = [5.2, 0.7]
    for idx, ny in enumerate(node_y):
        _box(ax, 6.9, ny, 5.7, 3.9, "",
             fc="white", ec=C_PURPLE, tc=C_DARK, radius=0.03)
        ax.text(7.05, ny + 3.55, f"Node {idx+1}",
                fontsize=10, weight="bold", color=C_PURPLE)

        # System components
        _box(ax, 7.1, ny + 2.55, 1.8, 0.7, "kubelet",
             fc=C_BLUE, fontsize=9, weight="bold", radius=0.04)
        _box(ax, 9.0, ny + 2.55, 1.8, 0.7, "kube-proxy",
             fc=C_GREEN, fontsize=9, weight="bold", radius=0.04)
        _box(ax, 10.9, ny + 2.55, 1.6, 0.7, "containerd",
             fc=C_AMBER, fontsize=9, weight="bold", radius=0.04)

        # Pods
        for j, px in enumerate([7.1, 8.95, 10.8]):
            _box(ax, px, ny + 0.4, 1.6, 1.6, "",
                 fc="#fef3c7", ec=C_AMBER, tc=C_DARK, radius=0.05)
            ax.text(px + 0.8, ny + 1.85, f"Pod {j+1}",
                    ha="center", fontsize=8.5, weight="bold", color=C_DARK)
            # Container inside
            _box(ax, px + 0.2, ny + 0.55, 1.2, 0.85, "container",
                 fc=C_PURPLE, fontsize=7.5, weight="bold", radius=0.03)

        # API Server -> kubelet arrow
        _arrow(ax, 3.2, 8.4, 7.1, ny + 2.9,
               color=C_BLUE, lw=1.0, rad=-0.15, style="->")

    ax.text(4.8, 4.85, "watch / instructions",
            fontsize=8.5, color=C_BLUE, rotation=8, style="italic")

    ax.set_title("Kubernetes Architecture: Control Plane Drives Worker Nodes",
                 fontsize=14, weight="bold", color=C_DARK, pad=10, loc="left")

    fig.tight_layout()
    _save(fig, "fig3_kubernetes_architecture")


# ---------------------------------------------------------------------------
# Figure 4: Service mesh (Istio sidecar)
# ---------------------------------------------------------------------------
def fig4_service_mesh_istio() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10)
    ax.axis("off")

    # Control plane
    _box(ax, 4.0, 8.3, 4.0, 1.2, "Istiod  (Pilot + Citadel + Galley)",
         fc=C_DARK, fontsize=10.5, weight="bold", radius=0.05)
    ax.text(6.0, 9.7, "Control Plane",
            ha="center", fontsize=10.5, weight="bold", color=C_DARK)

    # Two pods (data plane)
    def pod(x, y, name, color):
        _box(ax, x, y, 3.6, 3.0, "", fc="#f1f5f9", ec=C_GRAY,
             tc=C_DARK, radius=0.05)
        ax.text(x + 1.8, y + 2.7, f"Pod  ({name})",
                ha="center", fontsize=10, weight="bold", color=C_DARK)
        # App container
        _box(ax, x + 0.3, y + 1.3, 1.5, 1.1, "App\nContainer",
             fc=color, fontsize=9, weight="bold", radius=0.05)
        # Sidecar (Envoy)
        _box(ax, x + 1.95, y + 1.3, 1.4, 1.1, "Envoy\nSidecar",
             fc=C_AMBER, fontsize=9, weight="bold", radius=0.05)
        # localhost link
        _arrow(ax, x + 1.8, y + 1.85, x + 1.95, y + 1.85,
               color=C_GRAY, lw=1.2, style="<->")
        ax.text(x + 1.85, y + 2.05, "localhost",
                fontsize=7.5, color=C_GRAY, ha="center")
        # Metrics out
        ax.text(x + 1.8, y + 0.6, "metrics  /  traces  /  logs",
                ha="center", fontsize=8, color=C_GRAY, style="italic")
        return x + 1.95 + 0.7, y + 1.85   # sidecar center y

    cx1, cy = pod(0.6, 3.5, "frontend", C_BLUE)
    # second pod
    _box(ax, 7.8, 3.5, 3.6, 3.0, "", fc="#f1f5f9", ec=C_GRAY,
         tc=C_DARK, radius=0.05)
    ax.text(7.8 + 1.8, 3.5 + 2.7, "Pod  (reviews)",
            ha="center", fontsize=10, weight="bold", color=C_DARK)
    _box(ax, 7.8 + 1.95, 3.5 + 1.3, 1.4, 1.1, "Envoy\nSidecar",
         fc=C_AMBER, fontsize=9, weight="bold", radius=0.05)
    _box(ax, 7.8 + 0.3, 3.5 + 1.3, 1.5, 1.1, "App\nContainer",
         fc=C_GREEN, fontsize=9, weight="bold", radius=0.05)
    _arrow(ax, 7.8 + 1.8, 3.5 + 1.85, 7.8 + 1.95, 3.5 + 1.85,
           color=C_GRAY, lw=1.2, style="<->")
    ax.text(7.8 + 1.85, 3.5 + 2.05, "localhost", fontsize=7.5,
            color=C_GRAY, ha="center")
    ax.text(7.8 + 1.8, 3.5 + 0.6, "metrics  /  traces  /  logs",
            ha="center", fontsize=8, color=C_GRAY, style="italic")

    # mTLS data plane between sidecars
    _arrow(ax, 3.95, 5.35, 8.05, 5.35,
           color=C_AMBER, lw=2.2, style="<->")
    ax.text(6.0, 5.65, "mTLS  +  retries  +  timeouts  +  traffic split",
            ha="center", fontsize=10.5, weight="bold", color=C_AMBER)
    ax.text(6.0, 5.05, "(data plane: app code unchanged)",
            ha="center", fontsize=8.5, color=C_GRAY, style="italic")

    # Control plane -> sidecars
    _arrow(ax, 5.0, 8.3, 3.3, 6.5, color=C_DARK, lw=1.2, style="->", rad=-0.15)
    _arrow(ax, 7.0, 8.3, 9.7, 6.5, color=C_DARK, lw=1.2, style="->", rad=0.15)
    ax.text(3.6, 7.5, "xDS config\n+ certs",
            fontsize=8.5, color=C_DARK, ha="center", style="italic")
    ax.text(8.4, 7.5, "xDS config\n+ certs",
            fontsize=8.5, color=C_DARK, ha="center", style="italic")

    # Observability
    _box(ax, 1.0, 0.5, 10.0, 1.5, "", fc="#ecfdf5", ec=C_GREEN,
         tc=C_DARK, radius=0.04)
    for i, (label, x) in enumerate([("Prometheus", 2.0),
                                    ("Grafana", 4.5),
                                    ("Jaeger", 7.0),
                                    ("Kiali", 9.5)]):
        _box(ax, x - 0.7, 0.85, 1.4, 0.85, label,
             fc=C_GREEN, fontsize=9, weight="bold", radius=0.04)
    ax.text(6.0, 1.85, "Observability stack (built-in golden signals)",
            ha="center", fontsize=10, weight="bold", color=C_GREEN)

    ax.set_title("Service Mesh (Istio): Sidecars Carry Cross-cutting Concerns",
                 fontsize=14, weight="bold", color=C_DARK, pad=10, loc="left")

    fig.tight_layout()
    _save(fig, "fig4_service_mesh_istio")


# ---------------------------------------------------------------------------
# Figure 5: Helm chart pipeline + release history
# ---------------------------------------------------------------------------
def fig5_helm_charts() -> None:
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")

    # Chart structure
    _box(ax, 0.3, 0.6, 3.0, 5.0, "", fc="#eff6ff", ec=C_BLUE, tc=C_DARK,
         radius=0.04)
    ax.text(1.8, 5.25, "my-chart/", ha="center", fontsize=10.5,
            weight="bold", color=C_BLUE)
    contents = [
        "Chart.yaml",
        "values.yaml",
        "templates/",
        "  deployment.yaml",
        "  service.yaml",
        "  ingress.yaml",
        "charts/  (deps)",
    ]
    for i, line in enumerate(contents):
        ax.text(0.55, 4.6 - i * 0.55, line, fontsize=9.2,
                color=C_DARK, family="monospace")

    # values.yaml
    _box(ax, 3.8, 2.4, 2.4, 2.0, "values.yaml\n(env-specific\noverrides)",
         fc=C_PURPLE, fontsize=9.5, weight="bold", radius=0.05)

    # Helm engine
    _box(ax, 6.7, 2.4, 2.4, 2.0, "helm template\n(Go templating\n+ Sprig)",
         fc=C_AMBER, fontsize=9.5, weight="bold", radius=0.05)

    # Rendered manifests
    _box(ax, 9.6, 2.4, 2.4, 2.0, "Rendered\nK8s manifests\n(YAML)",
         fc=C_GREEN, fontsize=9.5, weight="bold", radius=0.05)

    # Cluster
    _box(ax, 12.2, 2.4, 1.6, 2.0, "Cluster\n(release)",
         fc=C_DARK, fontsize=9.5, weight="bold", radius=0.05)

    # Arrows
    _arrow(ax, 3.3, 3.4, 3.8, 3.4, color=C_DARK, lw=1.6)
    _arrow(ax, 6.2, 3.4, 6.7, 3.4, color=C_DARK, lw=1.6)
    _arrow(ax, 9.1, 3.4, 9.6, 3.4, color=C_DARK, lw=1.6)
    _arrow(ax, 12.0, 3.4, 12.2, 3.4, color=C_DARK, lw=1.6)

    ax.text(7.0, 1.55,
            "helm install release ./my-chart -f values-prod.yaml",
            ha="center", fontsize=10, color=C_DARK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f1f5f9", ec=C_GRAY))

    ax.set_title("Helm: Templated, Versioned, Reusable Kubernetes Packages",
                 fontsize=14, weight="bold", color=C_DARK, loc="left")

    # ---- Bottom: release history timeline ----
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 14); ax2.set_ylim(0, 4)
    ax2.axis("off")
    ax2.set_title("Release history (helm rollback works on any revision)",
                  fontsize=11.5, weight="bold", color=C_DARK, loc="left")

    revs = [
        (1, "v1.0.0", "install",  C_BLUE,   "deployed"),
        (2, "v1.1.0", "upgrade",  C_GREEN,  "deployed"),
        (3, "v1.2.0", "upgrade",  C_AMBER,  "FAILED"),
        (4, "v1.1.0", "rollback", C_PURPLE, "deployed"),
        (5, "v1.3.0", "upgrade",  C_GREEN,  "deployed"),
    ]
    xs = np.linspace(1.0, 13.0, len(revs))
    ax2.plot(xs, [2.0] * len(xs), color=C_GRAY, lw=2.0, zorder=1)
    for x, (rev, ver, action, col, status) in zip(xs, revs):
        ax2.scatter([x], [2.0], s=380, color=col, edgecolor=C_DARK,
                    linewidth=1.4, zorder=3)
        ax2.text(x, 2.0, str(rev), ha="center", va="center",
                 fontsize=10, weight="bold", color="white")
        ax2.text(x, 2.85, ver, ha="center", fontsize=9.5,
                 weight="bold", color=C_DARK)
        ax2.text(x, 1.15, f"{action}\n[{status}]", ha="center",
                 fontsize=8.5, color=col,
                 weight="bold" if status == "FAILED" else "normal")

    ax2.text(13.5, 2.0, "time -->", fontsize=9, color=C_GRAY,
             style="italic", va="center")

    _save(fig, "fig5_helm_charts")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Cloud Computing Part 07 figures...")
    fig1_microservices_vs_monolith()
    fig2_docker_layers()
    fig3_kubernetes_architecture()
    fig4_service_mesh_istio()
    fig5_helm_charts()
    print("Done.")


if __name__ == "__main__":
    main()
