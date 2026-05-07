#!/usr/bin/env python3
"""Regenerate broken figures detected by VLM audit (2026-05-07).

Each fig fixes the specific bug the VLM flagged:
- fig2_tcp_handshake.png: footnote clipped at bottom -> add bottom margin, wrap text
- fig1_three_compute_patterns.png: text overflows column -> wrap text per column
- fig3_init_apply_loop.png: caption labels overlap between boxes -> shorten + tighten spacing
- fig3_cloud_init_flow.png: caption labels overlap -> wrap captions
- fig5_decision_tree.png: bottom example boxes show overlapping text -> stack vertically
- fig2_loss_functions.png: 'punishes heavily' label overlaps MSE curve -> reposition
- fig1_dag.png: 'Sprinkler' label clipped by node -> shorten to 'Sprink' or shrink font
- fig1_mllm_architecture.png: green output box clips text -> widen box
- fig3_encoder_distilling.png: Layer 4 label clipped by box -> enlarge box
- fig3_rag_vs_finetuning.png: 'outputting' clips, 'no' label overlaps box -> reflow
- fig1_functional_minimization.png: bar label overlaps adjacent bar -> outside-bar labels

Each PNG gets uploaded to BOTH EN and ZH OSS paths.
"""
import os
import subprocess
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Load .env
ENV_PATH = Path("/root/chenk-hugo/.env")
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

OSSUTIL = "/root/.aliyun/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_AK = os.environ["OSS_AK"]
OSS_SK = os.environ["OSS_SK"]
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"

OUT = Path("/tmp/vlm_audit/fixed")
OUT.mkdir(parents=True, exist_ok=True)


def upload(local: Path, key: str) -> bool:
    cmd = [
        OSSUTIL, "cp", "-f",
        "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
        "--region", "cn-beijing",
        str(local), f"oss://{OSS_BUCKET}/{key}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"  UPLOAD FAIL {key}: rc={r.returncode} stderr={r.stderr[-300:]}")
        return False
    print(f"  uploaded {key}")
    return True


def upload_pair(local: Path, en_key: str, zh_key: str) -> int:
    n = 0
    if upload(local, en_key):
        n += 1
    if upload(local, zh_key):
        n += 1
    return n


# =================== FIG: TCP handshake ===================
def make_tcp_handshake() -> Path:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("TCP 3-Way Handshake", fontsize=18, fontweight="bold", pad=14)
    ax.text(50, 92, "Establishing a reliable connection: SYN -> SYN-ACK -> ACK",
            ha="center", fontsize=12, style="italic", color="#555")

    # Two endpoints
    ax.add_patch(FancyBboxPatch((10, 78), 22, 8, boxstyle="round,pad=0.3",
                                fc="#3060d4", ec="#1f3a7a", lw=2))
    ax.text(21, 82, "Client", ha="center", va="center", fontsize=14,
            fontweight="bold", color="white")
    ax.add_patch(FancyBboxPatch((68, 78), 22, 8, boxstyle="round,pad=0.3",
                                fc="#7a3fd0", ec="#3a1a6a", lw=2))
    ax.text(79, 82, "Server", ha="center", va="center", fontsize=14,
            fontweight="bold", color="white")

    # Vertical lifelines
    ax.plot([21, 21], [22, 78], color="#3060d4", lw=1.5, alpha=0.45)
    ax.plot([79, 79], [22, 78], color="#7a3fd0", lw=1.5, alpha=0.45)

    # State labels (left column for client, right for server)
    states = [("CLOSED", "LISTEN", 70),
              ("SYN_SENT", "SYN_RCVD", 50),
              ("ESTABLISHED", "ESTABLISHED", 28)]
    for left, right, y in states:
        ax.text(5, y, left, ha="left", va="center", fontsize=10, fontweight="bold",
                color="#1f3a7a", bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                           ec="#1f3a7a", lw=1))
        ax.text(95, y, right, ha="right", va="center", fontsize=10, fontweight="bold",
                color="#3a1a6a", bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                           ec="#3a1a6a", lw=1))

    # Arrows + labels (positioned ABOVE the line so they don't overlap)
    # 1) SYN  client -> server, y descending from 70 to 60
    ax.annotate("", xy=(79, 60), xytext=(21, 70),
                arrowprops=dict(arrowstyle="->", color="#1f5edb", lw=2.5))
    ax.text(50, 67.5, "(1) SYN  seq = x", ha="center", fontsize=11,
            family="monospace", color="#1f5edb", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f5edb", lw=1.2))

    # 2) SYN-ACK  server -> client, y descending from 50 to 40
    ax.annotate("", xy=(21, 40), xytext=(79, 50),
                arrowprops=dict(arrowstyle="->", color="#7a3fd0", lw=2.5))
    ax.text(50, 47.5, "(2) SYN-ACK  seq = y, ack = x+1", ha="center", fontsize=11,
            family="monospace", color="#5a2faf", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#7a3fd0", lw=1.2))

    # 3) ACK  client -> server, y descending from 30 to 24
    ax.annotate("", xy=(79, 24), xytext=(21, 30),
                arrowprops=dict(arrowstyle="->", color="#1ea05a", lw=2.5))
    ax.text(50, 28.5, "(3) ACK  ack = y+1", ha="center", fontsize=11,
            family="monospace", color="#0e7a3a", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1ea05a", lw=1.2))

    # Footnote (NOT clipped — placed inside axes area with proper margin)
    footnote = ("Why three (not two)? Confirms BOTH directions can send AND receive,\n"
                "defends against half-open state and replayed delayed SYNs.")
    ax.add_patch(FancyBboxPatch((6, 2), 88, 14, boxstyle="round,pad=0.4",
                                fc="#1a1f2e", ec="#1a1f2e", lw=1))
    ax.text(50, 9, footnote, ha="center", va="center", fontsize=11,
            color="white", fontweight="bold")

    out = OUT / "fig2_tcp_handshake.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: three compute patterns ===================
def make_three_compute_patterns() -> Path:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("Three places to run an agent: ECS  ·  ACK  ·  Function Compute",
                 fontsize=17, fontweight="bold", pad=14)
    ax.text(50, 91, "Each maps to a different scale, durability and cost profile",
            ha="center", fontsize=12, color="#555")

    cols = [
        ("ECS", "#3060d4", "#dde6f0",
         "Long-lived VM, pm2 supervisor.\nStateful, predictable, easy to SSH-debug.\n\n"
         "Good for: prototypes,\nsingle-tenant agents, custom\nbinaries that must be there."),
        ("ACK (K8s)", "#7a3fd0", "#ece0f5",
         "Production agent fleet on K8s.\nHorizontal Pod Autoscaler,\nrolling deploys, GPU schedule.\n\n"
         "Good for: multi-tenant\nproduction, 5+ agent kinds,\nSRE who own the infra."),
        ("Function Compute", "#1ea05a", "#dff2e6",
         "Per-invocation, scale-to-zero.\nCold-start 200-800ms,\nmax 24h runtime.\n\n"
         "Good for: webhook-triggered\nagents, scheduled crawlers, glue."),
    ]

    col_w = 26
    gap = 4
    total_w = 3 * col_w + 2 * gap
    x0 = (100 - total_w) / 2
    for i, (title, ec, fc, body) in enumerate(cols):
        x = x0 + i * (col_w + gap)
        ax.add_patch(FancyBboxPatch((x, 30), col_w, 50, boxstyle="round,pad=0.4",
                                    fc=fc, ec=ec, lw=2))
        ax.text(x + col_w / 2, 75, title, ha="center", va="center",
                fontsize=15, fontweight="bold", color=ec)
        ax.text(x + col_w / 2, 53, body, ha="center", va="center",
                fontsize=10.5, color="#333", linespacing=1.45)

    # Decision rule panel
    ax.add_patch(FancyBboxPatch((6, 6), 88, 18, boxstyle="round,pad=0.4",
                                fc="#fdf2d6", ec="#e0a020", lw=1.5))
    ax.text(50, 19.5, "Decision rule of thumb", ha="center", fontsize=13,
            fontweight="bold", color="#c07000")
    ax.text(50, 11.5,
            "QPS < 1 and bursty  →  FC      ·      "
            "QPS 1-50 long-lived  →  ECS      ·      "
            "QPS > 50 multi-agent fleet  →  ACK",
            ha="center", fontsize=11.5, color="#333")

    out = OUT / "fig1_three_compute_patterns.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: terraform init/apply loop ===================
def make_init_apply_loop() -> Path:
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("The five-command loop you'll run hundreds of times",
                 fontsize=17, fontweight="bold", pad=14)
    ax.text(50, 91, "Init once per backend change; plan/apply on every change",
            ha="center", fontsize=12, color="#555")

    cmds = [
        ("terraform fmt", "#9aa3b0", "format\nHCL files"),
        ("terraform init", "#13b8d4", "download\nprovider +\ninit backend"),
        ("terraform validate", "#7a3fd0", "static\nschema\ncheck"),
        ("terraform plan", "#2860dc", "diff desired\nvs real"),
        ("terraform apply", "#1ea05a", "send API\ncalls"),
    ]
    n = len(cmds)
    box_w = 14
    gap = 4
    total = n * box_w + (n - 1) * gap
    x0 = (100 - total) / 2
    y_box = 60
    h = 14
    centers = []
    for i, (cmd, color, cap) in enumerate(cmds):
        x = x0 + i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, y_box), box_w, h, boxstyle="round,pad=0.2",
                                    fc=color, ec=color, lw=1))
        ax.text(x + box_w / 2, y_box + h / 2, cmd, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        # Caption BELOW box, centered, multi-line
        ax.text(x + box_w / 2, y_box - 3, cap, ha="center", va="top",
                fontsize=10, color="#444", linespacing=1.25)
        centers.append((x + box_w + gap / 2, y_box + h / 2))
    # Arrows between boxes
    for i in range(n - 1):
        x_start = x0 + i * (box_w + gap) + box_w
        x_end = x0 + (i + 1) * (box_w + gap)
        ax.annotate("", xy=(x_end, y_box + h / 2), xytext=(x_start, y_box + h / 2),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=2))

    # Failure modes panel
    ax.add_patch(FancyBboxPatch((6, 6), 88, 28, boxstyle="round,pad=0.4",
                                fc="#fdf2d6", ec="#e0a020", lw=1.5))
    ax.text(50, 28, "Failure modes you'll hit on day 1",
            ha="center", fontsize=13, fontweight="bold", color="#c07000")
    failures = [
        "·  Provider download blocked by GFW   →   use mirror or HTTPS_PROXY",
        "·  Backend not initialized   →   re-run terraform init",
        "·  Stale lock after Ctrl-C   →   terraform force-unlock <id>",
        "·  Provider version drift   →   pin in required_providers block",
    ]
    for i, line in enumerate(failures):
        ax.text(50, 22 - i * 3.8, line, ha="center", fontsize=11, color="#333")

    out = OUT / "fig3_init_apply_loop.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: cloud-init flow ===================
def make_cloud_init_flow() -> Path:
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("Bootstrapping an agent on ECS via cloud-init user_data",
                 fontsize=17, fontweight="bold", pad=14)
    ax.text(50, 91, "From bare Ubuntu image to pm2-supervised Python agent in ~90 seconds",
            ha="center", fontsize=12, color="#555")

    steps = [
        ("Image", "#13b8d4", "ubuntu_22_04_x64"),
        ("user_data", "#2860dc", "apt + python3.11\n+ node20"),
        ("clone repo", "#7a3fd0", "git clone\nagent-runtime"),
        ("pip install", "#d63a8c", "uv pip sync"),
        ("pm2 start", "#1ea05a", "pm2 start\necosystem.json"),
    ]
    n = len(steps)
    box_w = 14
    gap = 4
    total = n * box_w + (n - 1) * gap
    x0 = (100 - total) / 2
    y_box = 60
    h = 13
    for i, (label, color, cap) in enumerate(steps):
        x = x0 + i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, y_box), box_w, h, boxstyle="round,pad=0.2",
                                    fc=color, ec=color, lw=1))
        ax.text(x + box_w / 2, y_box + h / 2, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(x + box_w / 2, y_box - 3, cap, ha="center", va="top",
                fontsize=9.5, color="#444", linespacing=1.3)
    for i in range(n - 1):
        x_start = x0 + i * (box_w + gap) + box_w
        x_end = x0 + (i + 1) * (box_w + gap)
        ax.annotate("", xy=(x_end, y_box + h / 2), xytext=(x_start, y_box + h / 2),
                    arrowprops=dict(arrowstyle="->", color="#aaa", lw=2))

    note = (
        "user_data is base64-encoded HCL → bash. Logs to /var/log/cloud-init-output.log.\n"
        "If pm2 is missing, ECS image was rebuilt without it — bake your own image with Packer to skip 60s of apt install on every boot.\n"
        "Idempotency: terraform replace = new instance + new IP. Use lifecycle { create_before_destroy = true } for zero-downtime rotation."
    )
    ax.add_patch(FancyBboxPatch((6, 6), 88, 28, boxstyle="round,pad=0.4",
                                fc="#f5f5f7", ec="#bbb", lw=1))
    ax.text(50, 20, note, ha="center", va="center", fontsize=10.5, color="#333",
            linespacing=1.6)

    out = OUT / "fig3_cloud_init_flow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: two-pointers decision tree ===================
def make_decision_tree() -> Path:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("When to Use Which Two-Pointer Pattern",
                 fontsize=17, fontweight="bold", pad=14)

    # root
    ax.add_patch(FancyBboxPatch((35, 80), 30, 10, boxstyle="round,pad=0.3",
                                fc="#3a4555", ec="#1f2a38", lw=2))
    ax.text(50, 85, "What does the input look like?", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    # branches
    branches = [
        (12, "Linked list", "#3060d4", "Fast / slow pointers",
         ["Linked List Cycle", "Find Middle Node"]),
        (42, "Array (sorted)", "#7a3fd0", "Collision pointers",
         ["Two Sum II / 3Sum", "Container With Most Water"]),
        (72, "Subarray / substring", "#1ea05a", "Sliding window",
         ["Longest Substring", "Min Window Substring"]),
    ]
    edge_labels = [(22, "node refs"), (50, "indices, sorted"), (78, "contiguous range")]
    for (xc, lbl) in edge_labels:
        ax.text(xc, 75, lbl, ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666", lw=1))

    for x, top, color, mid, examples in branches:
        # arrow from root to top
        ax.annotate("", xy=(x + 8, 70), xytext=(50, 80),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
        # top: input type
        ax.add_patch(FancyBboxPatch((x, 60), 16, 10, boxstyle="round,pad=0.3",
                                    fc=color, ec=color, lw=1))
        ax.text(x + 8, 65, top, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        # arrow down
        ax.annotate("", xy=(x + 8, 50), xytext=(x + 8, 60),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
        # mid: pattern
        ax.add_patch(FancyBboxPatch((x, 40), 16, 10, boxstyle="round,pad=0.3",
                                    fc="#e89020", ec="#c07010", lw=1))
        ax.text(x + 8, 45, mid, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        # arrow down
        ax.annotate("", xy=(x + 8, 30), xytext=(x + 8, 40),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
        # examples (stacked vertically inside box)
        ax.add_patch(FancyBboxPatch((x - 2, 12), 20, 18, boxstyle="round,pad=0.3",
                                    fc="white", ec="#999", lw=1))
        for j, ex in enumerate(examples):
            ax.text(x + 8, 24 - j * 6, ex, ha="center", va="center",
                    fontsize=10.5, color="#333")

    out = OUT / "fig5_decision_tree.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: loss functions ===================
def make_loss_functions() -> Path:
    r = np.linspace(-3, 3, 400)
    mse = r ** 2
    mae = np.abs(r)
    delta = 1.0
    huber = np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))

    m = np.linspace(-3, 3, 400)
    zero_one = (m <= 0).astype(float)
    hinge = np.maximum(0, 1 - m)
    ce = np.log2(1 + np.exp(-m))
    expo = np.exp(-m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.set_title("Regression losses (residual r = y - h(x))", fontsize=13, fontweight="bold")
    ax.plot(r, mse, color="#2860dc", lw=2.4, label="MSE  $r^2$")
    ax.plot(r, mae, color="#e89020", lw=2.4, label="MAE  $|r|$")
    ax.plot(r, huber, color="#1ea05a", lw=2.4, ls="--", label=r"Huber  ($\delta=1$)")
    ax.set_xlabel("residual r")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 6.5)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", fontsize=11, frameon=True)
    # 'punishes heavily' annotation — placed in upper-LEFT empty corner, away from MSE curve
    ax.annotate("MSE punishes\nlarge errors heavily",
                xy=(2.3, 5.3), xytext=(-2.7, 5.6),
                fontsize=10, color="#1f3a7a",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2860dc", lw=1),
                arrowprops=dict(arrowstyle="->", color="#2860dc", lw=1.2))

    ax = axes[1]
    ax.set_title("Classification surrogates for the 0-1 loss", fontsize=13, fontweight="bold")
    ax.plot(m, zero_one, color="#888", lw=2.4, label=r"0-1  $\mathbb{1}[m \leq 0]$")
    ax.plot(m, hinge, color="#7a3fd0", lw=2.4, label=r"Hinge  $\max(0, 1-m)$")
    ax.plot(m, ce, color="#2860dc", lw=2.4, label=r"Cross-entropy  $\log_2(1+e^{-m})$")
    ax.plot(m, expo, color="#d63a3a", lw=2.4, ls="--", label=r"Exponential  $e^{-m}$")
    ax.set_xlabel(r"margin $m = y \cdot h(x)$")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 5)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    out = OUT / "fig2_loss_functions.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: Bayesian DAG ===================
def make_dag() -> Path:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("Bayesian network: a DAG factorises the joint into local CPTs",
                 fontsize=15, fontweight="bold", pad=12)

    # Nodes (slightly smaller radius and labels FIT inside; longer label gets smaller font)
    def node(x, y, r, label, color):
        circ = plt.Circle((x, y), r, color=color, ec="#1f2a38", lw=1.5)
        ax.add_patch(circ)
        # auto font size: shorter label = bigger
        fs = 12 if len(label) <= 5 else 11 if len(label) <= 7 else 10
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                fontweight="bold", color="white")

    node(28, 78, 5, "Cloudy", "#3060d4")
    node(15, 50, 5.5, "Sprinkler", "#7a3fd0")  # bigger circle so 9-char label fits
    node(40, 50, 5, "Rain", "#1ea05a")
    node(28, 22, 5.5, "WetGrass", "#e89020")

    # Edges
    for (x1, y1), (x2, y2) in [
        ((28, 73), (15, 55)),
        ((28, 73), (40, 55)),
        ((15, 45), (28, 27)),
        ((40, 45), (28, 27)),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#1f2a38", lw=2))

    # CPT panel
    ax.text(58, 83, "Conditional probability tables", fontsize=12, fontweight="bold")
    cpts = [
        "$P(C) = 0.5$",
        "$P(S=1 \\mid C=1) = 0.10,\\ \\ P(S=1 \\mid C=0) = 0.50$",
        "$P(R=1 \\mid C=1) = 0.80,\\ \\ P(R=1 \\mid C=0) = 0.20$",
        "$P(W=1 \\mid S, R)$: 4 entries",
    ]
    for i, line in enumerate(cpts):
        ax.text(58, 76 - i * 5.5, line, fontsize=11, color="#333")

    # Joint factorisation footer
    ax.add_patch(FancyBboxPatch((10, 4), 80, 10, boxstyle="round,pad=0.3",
                                fc="#f5f5f7", ec="#999", lw=1))
    ax.text(50, 10, r"$P(C,S,R,W) = P(C)\, P(S \mid C)\, P(R \mid C)\, P(W \mid S, R)$",
            ha="center", va="center", fontsize=12)
    ax.text(50, 6, "16 joint entries  ->  1 + 2 + 2 + 4 = 9 free parameters",
            ha="center", va="center", fontsize=10, color="#666", style="italic")

    out = OUT / "fig1_dag.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: MLLM architecture ===================
def make_mllm_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("Multimodal LLM Architecture: Vision Encoder + Projector + LLM Decoder",
                 fontsize=15, fontweight="bold", pad=14)

    def box(x, y, w, h, label, fc, ec, fontcolor="white", fontsize=11, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), label, ha="center",
                va="center", fontsize=fontsize, fontweight="bold", color=fontcolor)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2, sub, ha="center", va="center",
                    fontsize=9, color=fontcolor)

    # Top row: image path
    box(4, 70, 10, 8, "Image", "#eaeaea", "#bbb", fontcolor="#333")
    box(18, 67, 13, 14, "Vision\nEncoder", "#2860dc", "#1f3a7a", sub="ViT-L/14 frozen")
    box(35, 70, 11, 8, "Patch\nTokens", "#eaeaea", "#bbb", fontcolor="#333")
    box(50, 67, 14, 14, "Projector\n(MLP / Q-Former)", "#e89020", "#a06010", sub="TRAINABLE")
    box(68, 70, 11, 8, "Visual\nTokens", "#eaeaea", "#bbb", fontcolor="#333")

    # Arrows top row
    for x1, x2 in [(14, 18), (31, 35), (46, 50), (64, 68)]:
        ax.annotate("", xy=(x2, 74), xytext=(x1, 74),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Middle row: text path
    box(4, 42, 10, 8, "Text\nPrompt", "#eaeaea", "#bbb", fontcolor="#333")
    box(18, 39, 13, 14, "Tokenizer +\nText Embed", "#7a3fd0", "#3a1a6a")
    box(35, 42, 11, 8, "Text\nTokens", "#eaeaea", "#bbb", fontcolor="#333")
    box(50, 39, 14, 14, "Concatenate\n[<img>] || [text]", "#888", "#444")
    for x1, x2 in [(14, 18), (31, 35), (46, 50)]:
        ax.annotate("", xy=(x2, 46), xytext=(x1, 46),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Visual tokens -> Concatenate (curved arrow down)
    ax.annotate("", xy=(57, 53), xytext=(73, 70),
                arrowprops=dict(arrowstyle="->", color="#2860dc", lw=1.5,
                                connectionstyle="arc3,rad=0.3"))

    # Bottom: LLM decoder + output (output box WIDER so text fits)
    box(50, 14, 28, 18, "LLM Decoder\n(Llama / Qwen / Mistral)", "#7a3fd0", "#3a1a6a",
        sub="Frozen or LoRA", fontsize=12)
    # Concatenate -> LLM
    ax.annotate("", xy=(64, 32), xytext=(57, 39),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    # LLM -> Output (output box TO THE LEFT, wider, fits text)
    box(8, 16, 36, 12,
        "Generated Text  ->  caption / answer / reasoning",
        "#1ea05a", "#0e7a3a", fontsize=11)
    ax.annotate("", xy=(44, 22), xytext=(50, 22),
                arrowprops=dict(arrowstyle="<-", color="#333", lw=1.5))

    # Legend
    handles = [
        ("Vision (frozen)", "#2860dc"),
        ("Adapter (trainable)", "#e89020"),
        ("Language model", "#7a3fd0"),
        ("Output", "#1ea05a"),
    ]
    for i, (lbl, c) in enumerate(handles):
        ax.add_patch(plt.Rectangle((6 + i * 18, 4), 2, 3, color=c))
        ax.text(8.5 + i * 18, 5.5, lbl, fontsize=10, va="center")

    out = OUT / "fig1_mllm_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: encoder distilling ===================
def make_encoder_distilling() -> Path:
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("Encoder distilling: progressive sequence compression",
                 fontsize=15, fontweight="bold", pad=12)

    ax.text(7, 91, "ProbSparse self-attention runs INSIDE each layer.",
            fontsize=11, fontweight="bold", color="#333")
    ax.text(7, 87, "Distilling halves the sequence between layers, so memory "
                   "is geometric rather than linear in depth.",
            fontsize=10.5, color="#555")

    layers = [
        ("Layer 1: L = 96", 96, "#2860dc", 80, 65),
        ("Layer 2: L = 48", 48, "#7a3fd0", 60, 50),
        ("Layer 3: L = 24", 24, "#e89020", 40, 35),
        ("Layer 4: L = 12", 12, "#1ea05a", 24, 20),  # ENLARGED width so label fits
    ]
    y = 78
    centers = []
    for label, L, color, w, h in layers:
        x = (100 - w) / 2
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h * 0.6, boxstyle="round,pad=0.2",
                                    fc=color, ec=color, lw=1))
        ax.text(50, y - 1, label, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white")
        # Tick marks INSIDE box (smaller for shorter sequences)
        ticks = "│" * min(L, 60)
        ax.text(50, y - h * 0.22, ticks, ha="center", va="center",
                fontsize=7, color="white", family="monospace")
        centers.append((50, y - h * 0.4, y))
        y -= 22

    # Arrows + labels between layers
    arrow_y_pairs = []
    for i in range(len(layers) - 1):
        y_top = 78 - i * 22 - 6
        y_bot = 78 - (i + 1) * 22 + 6
        ax.annotate("", xy=(50, y_bot), xytext=(50, y_top),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
        ax.text(56, (y_top + y_bot) / 2,
                "Conv1d (k=3, s=2) -> ELU -> MaxPool(k=3, s=2)",
                fontsize=10, style="italic", color="#444", va="center")

    out = OUT / "fig3_encoder_distilling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: RAG vs fine-tuning ===================
def make_rag_vs_finetuning() -> Path:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("Adapt an LLM to your task — pick the cheapest tool that closes the gap",
                 fontsize=14, fontweight="bold", pad=12)

    # Start
    ax.add_patch(FancyBboxPatch((38, 84), 24, 8, boxstyle="round,pad=0.3",
                                fc="white", ec="#333", lw=1.5))
    ax.text(50, 88, "Start: baseline LLM\nis insufficient", ha="center", va="center",
            fontsize=10.5, fontweight="bold")

    # Decision: missing knowledge
    ax.annotate("", xy=(50, 78), xytext=(50, 84),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
    ax.add_patch(FancyBboxPatch((36, 70), 28, 8, boxstyle="round,pad=0.3",
                                fc="#eaf3ff", ec="#2860dc", lw=1.5))
    ax.text(50, 74, "Missing knowledge?", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1f3a7a")

    # Left branch: no
    ax.annotate("", xy=(20, 60), xytext=(38, 70),
                arrowprops=dict(arrowstyle="->", color="#7a3fd0", lw=1.5))
    ax.text(28, 67, "no", fontsize=11, color="#7a3fd0", fontweight="bold")
    ax.add_patch(FancyBboxPatch((8, 52), 24, 8, boxstyle="round,pad=0.3",
                                fc="#f0e5ff", ec="#7a3fd0", lw=1.5))
    ax.text(20, 56, "Wrong style/format?", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#3a1a6a")

    # Prompt engineering
    ax.annotate("", xy=(20, 42), xytext=(20, 52),
                arrowprops=dict(arrowstyle="->", color="#7a3fd0", lw=1.2))
    ax.add_patch(FancyBboxPatch((4, 28), 32, 14, boxstyle="round,pad=0.3",
                                fc="#f7f0ff", ec="#7a3fd0", lw=1.5))
    ax.text(20, 39, "Prompt engineering", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(20, 35.5, "• few-shot · CoT · structured output", ha="center", fontsize=9.5)
    ax.text(20, 32.5, "• cheapest, instant, version-controlled", ha="center", fontsize=9.5)
    ax.text(20, 30, "(start here for style/format gaps)", ha="center", fontsize=8.5,
            style="italic", color="#666")

    # Fine-tuning
    ax.annotate("", xy=(20, 22), xytext=(20, 28),
                arrowprops=dict(arrowstyle="->", color="#e89020", lw=1.2))
    ax.add_patch(FancyBboxPatch((4, 6), 32, 16, boxstyle="round,pad=0.3",
                                fc="#fdf2d6", ec="#e89020", lw=1.5))
    ax.text(20, 19, "Fine-tuning (SFT / LoRA / DPO)", ha="center",
            fontsize=10.5, fontweight="bold")
    ax.text(20, 15.5, "• fixed style, tone, schemas", ha="center", fontsize=9.5)
    ax.text(20, 12.5, "• needs ≥ 10³–10⁴ labelled examples", ha="center", fontsize=9.5)
    ax.text(20, 9.5, "• only after prompts are exhausted", ha="center", fontsize=9.5)

    # Right branch: yes
    ax.annotate("", xy=(80, 60), xytext=(62, 70),
                arrowprops=dict(arrowstyle="->", color="#1ea05a", lw=1.5))
    ax.text(72, 67, "yes", fontsize=11, color="#1ea05a", fontweight="bold")
    ax.add_patch(FancyBboxPatch((68, 52), 24, 8, boxstyle="round,pad=0.3",
                                fc="#dff2e6", ec="#1ea05a", lw=1.5))
    ax.text(80, 56, "Knowledge changes\nfrequently?", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#0e6a2a")

    # RAG (yes)
    ax.annotate("", xy=(80, 42), xytext=(80, 52),
                arrowprops=dict(arrowstyle="->", color="#1ea05a", lw=1.2))
    ax.add_patch(FancyBboxPatch((64, 28), 32, 14, boxstyle="round,pad=0.3",
                                fc="#e8f7ee", ec="#1ea05a", lw=1.5))
    ax.text(80, 39, "RAG", ha="center", fontsize=11, fontweight="bold")
    ax.text(80, 35.5, "• cite sources, refresh in minutes", ha="center", fontsize=9.5)
    ax.text(80, 32.5, "• vector + BM25 hybrid recommended", ha="center", fontsize=9.5)

    # RAG + fine-tune
    ax.annotate("", xy=(80, 22), xytext=(80, 28),
                arrowprops=dict(arrowstyle="->", color="#2860dc", lw=1.2))
    ax.add_patch(FancyBboxPatch((64, 6), 32, 16, boxstyle="round,pad=0.3",
                                fc="#eaf3ff", ec="#2860dc", lw=1.5))
    ax.text(80, 19, "RAG + fine-tune", ha="center", fontsize=11, fontweight="bold")
    ax.text(80, 15.5, "• stable corpus, narrow domain", ha="center", fontsize=9.5)
    ax.text(80, 12.5, "• fine-tune retriever or reader", ha="center", fontsize=9.5)
    ax.text(80, 9.5, "• highest quality, highest cost", ha="center", fontsize=9.5)

    out = OUT / "fig3_rag_vs_finetuning.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== FIG: functional minimization ===================
def make_functional_minimization() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: descent paths
    ax = axes[0]
    ax.set_title("Candidate descent paths", fontsize=12, fontweight="bold")
    x = np.linspace(0, 3, 200)
    paths = [
        ("Straight line  (T = 3.668)", x, 0 + (-1.5 / 3) * x, "#3a4555"),
        ("Steep then flat  (T = 2.991)", x, np.where(x < 1, -1.5 * x, -1.5), "#d63a3a"),
        ("Shallow parabola  (T = 11.960)", x, -1.5 * (x / 3) ** 2, "#e89020"),
        ("Circular arc  (T = 1321.650)", x, -1.5 * np.sin(np.pi * x / 6), "#1ea05a"),
    ]
    # cycloid (minimizer) parametric
    t = np.linspace(0, 2 * np.pi, 200)
    cyc_x = 3 / (2 * np.pi) * (t - np.sin(t)) * (2 * np.pi / (2 * np.pi))
    cyc_x = (3 / (2 * np.pi)) * (t - np.sin(t))
    cyc_y = -(3 / (2 * np.pi)) * (1 - np.cos(t))
    paths.append(("Cycloid (minimizer)  (T = 1.853)", cyc_x, cyc_y, "#7a3fd0"))

    for label, xx, yy, color in paths:
        ax.plot(xx, yy, color=color, lw=2, label=label)

    ax.scatter([0, 3], [0, -1.5], color="black", s=40, zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel(r"y (gravity ↓)")
    ax.invert_yaxis()
    ax.set_xlim(-0.2, 3.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)

    # Right: bar chart with labels OUTSIDE bars
    ax = axes[1]
    ax.set_title(r"Functional values:  $T[y] = \int \sqrt{1 + y'^2}/\sqrt{2gy}\, dx$",
                 fontsize=12, fontweight="bold")
    names = ["Cycloid (minimizer)", "Circular arc", "Shallow parabola",
             "Steep then flat", "Straight line"]
    values = [1.853, 1321.650, 11.960, 2.991, 3.668]
    colors = ["#7a3fd0", "#1ea05a", "#e89020", "#d63a3a", "#3a4555"]
    bars = ax.barh(names, values, color=colors, alpha=0.85)
    ax.set_xlabel("Descent time T[y]  (smaller is better)")
    # Annotate values OUTSIDE bars (or just past) — never overlap adjacent bar
    for bar, val in zip(bars, values):
        w = bar.get_width()
        # Use log scale to show small + large
        ax.text(w * 1.05 + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9.5)
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", alpha=0.25)

    out = OUT / "fig1_functional_minimization.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================== Driver ===================
FIXES = [
    # (gen_func, en_key, zh_key)
    (make_tcp_handshake,
     "posts/en/computer-fundamentals/05-network-power/fig2_tcp_handshake.png",
     "posts/zh/computer-fundamentals/05-network-power/fig2_tcp_handshake.png"),
    (make_three_compute_patterns,
     "posts/en/terraform-agents/04-compute-for-agent-runtime/fig1_three_compute_patterns.png",
     "posts/zh/terraform-agents/04-compute-for-agent-runtime/fig1_three_compute_patterns.png"),
    (make_init_apply_loop,
     "posts/en/terraform-agents/02-provider-and-state-setup/fig3_init_apply_loop.png",
     "posts/zh/terraform-agents/02-provider-and-state-setup/fig3_init_apply_loop.png"),
    (make_cloud_init_flow,
     "posts/en/terraform-agents/04-compute-for-agent-runtime/fig3_cloud_init_flow.png",
     "posts/zh/terraform-agents/04-compute-for-agent-runtime/fig3_cloud_init_flow.png"),
    (make_decision_tree,
     "posts/en/leetcode/two-pointers/fig5_decision_tree.png",
     "posts/zh/leetcode/02-双指针技巧/fig5_decision_tree.png"),
    (make_loss_functions,
     "posts/en/ml-math-derivations/01-Introduction-and-Mathematical-Foundations/fig2_loss_functions.png",
     "posts/zh/ml-math-derivations/01-绪论与数学基础/fig2_loss_functions.png"),
    (make_dag,
     "posts/en/ml-math-derivations/10-Semi-Naive-Bayes-and-Bayesian-Networks/fig1_dag.png",
     "posts/zh/ml-math-derivations/10-半朴素贝叶斯与贝叶斯网络/fig1_dag.png"),
    (make_mllm_architecture,
     "posts/en/standalone/multimodal-llm-downstream-tasks/fig1_mllm_architecture.png",
     "posts/zh/standalone/多模态大模型及下游任务研究/fig1_mllm_architecture.png"),
    (make_encoder_distilling,
     "posts/en/time-series/informer-long-sequence/fig3_encoder_distilling.png",
     "posts/zh/time-series/08-Informer长序列预测/fig3_encoder_distilling.png"),
    (make_rag_vs_finetuning,
     "posts/en/standalone/llm-workflows-architecture/fig3_rag_vs_finetuning.png",
     "posts/zh/standalone/llm工作流与应用架构-企业级实战指南/fig3_rag_vs_finetuning.png"),
    (make_functional_minimization,
     "posts/en/pde-ml/03-Variational-Principles/fig1_functional_minimization.png",
     "posts/zh/pde-ml/03-变分原理与优化/fig1_functional_minimization.png"),
]


def main():
    fixed_ok = []
    failed = []
    upload_count = 0
    for gen, en_key, zh_key in FIXES:
        name = gen.__name__
        try:
            print(f"[gen] {name}")
            local = gen()
            n = upload_pair(local, en_key, zh_key)
            upload_count += n
            if n == 2:
                fixed_ok.append(name)
            else:
                failed.append((name, f"only {n}/2 uploaded"))
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed.append((name, str(e)))
    print(f"\n[summary] fixed={len(fixed_ok)} failed={len(failed)} uploads={upload_count}")
    for n in fixed_ok:
        print(f"  OK   {n}")
    for n, e in failed:
        print(f"  FAIL {n}: {e}")


if __name__ == "__main__":
    main()
