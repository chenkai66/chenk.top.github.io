"""
Figure generation script for Recommendation Systems Part 11:
Contrastive Learning and Self-Supervised Learning.

Outputs identical PNGs to the EN and ZH asset folders so both posts
share a single canonical visual set.

Run:
    python 11-contrastive-learning.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

# ---------- Style ----------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.6,
})

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GREY = "#94a3b8"
C_RED = "#ef4444"
C_DARK = "#1e293b"

# ---------- Output paths ----------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]  # .../chenk-site
EN_DIR = PROJECT_ROOT / "source/_posts/en/recommendation-systems/11-contrastive-learning"
ZH_DIR = PROJECT_ROOT / "source/_posts/zh/recommendation-systems/11-对比学习与自监督学习"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        out = d / name
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"  saved {out}")
    plt.close(fig)


# =============================================================
# Fig 1: Positive vs negative pair illustration
# =============================================================
def fig1_positive_negative_pairs():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # ---- Left: anchor + augmentations as positives ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Positive pairs: two views of the same user", color=C_DARK)

    # Anchor user in the centre
    anchor = (5, 5)
    ax.scatter(*anchor, s=900, c=C_BLUE, zorder=5, edgecolors="white", linewidths=2)
    ax.text(anchor[0], anchor[1] - 0.05, "u", color="white", ha="center", va="center",
            fontsize=13, fontweight="bold", zorder=6)
    ax.text(anchor[0], anchor[1] - 1.2, "anchor", ha="center", fontsize=9, color=C_DARK)

    # Two augmented views (e.g. edge-dropout views of the same user's neighbourhood)
    aug_pts = [(2.8, 7.2), (7.6, 7.0)]
    aug_labels = ["view A\n(edge dropout p=0.2)", "view B\n(edge dropout p=0.2)"]
    for (x, y), lab in zip(aug_pts, aug_labels):
        ax.scatter(x, y, s=600, c=C_GREEN, alpha=0.9, edgecolors="white", linewidths=2, zorder=5)
        ax.text(x, y, "u'", color="white", ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=6)
        ax.text(x, y + 1.0, lab, ha="center", fontsize=8.5, color=C_DARK)
        ax.annotate("", xy=(x, y - 0.3), xytext=(anchor[0], anchor[1] + 0.3),
                    arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=2.0))

    ax.text(5, 1.0,
            "pull together: $\\mathrm{sim}(z_A, z_B) \\to 1$",
            ha="center", fontsize=10.5, color=C_GREEN, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Right: anchor + random users as negatives ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Negative pairs: views of other users in the batch", color=C_DARK)

    ax.scatter(*anchor, s=900, c=C_BLUE, zorder=5, edgecolors="white", linewidths=2)
    ax.text(anchor[0], anchor[1] - 0.05, "u", color="white", ha="center", va="center",
            fontsize=13, fontweight="bold", zorder=6)
    ax.text(anchor[0], anchor[1] - 1.2, "anchor", ha="center", fontsize=9, color=C_DARK)

    rng = np.random.default_rng(2)
    angles = np.linspace(np.pi / 6, 2 * np.pi - np.pi / 6, 6)
    radii = rng.uniform(2.6, 3.4, size=len(angles))
    neg_names = ["v", "w", "x", "y", "p", "q"]
    for ang, r, name in zip(angles, radii, neg_names):
        x = 5 + r * np.cos(ang)
        y = 5 + r * np.sin(ang)
        ax.scatter(x, y, s=520, c=C_AMBER, alpha=0.9, edgecolors="white", linewidths=1.8, zorder=5)
        ax.text(x, y, name, color="white", ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=6)
        ax.annotate("", xy=(x - 0.2 * np.cos(ang), y - 0.2 * np.sin(ang)),
                    xytext=(anchor[0] + 0.2 * np.cos(ang), anchor[1] + 0.2 * np.sin(ang)),
                    arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.4,
                                    linestyle="dashed"))
    ax.text(5, 1.0,
            "push apart: $\\mathrm{sim}(z_u, z_{\\mathrm{neg}}) \\to 0$",
            ha="center", fontsize=10.5, color=C_AMBER, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Contrastive learning: pull positives together, push negatives apart",
                 fontsize=13, color=C_DARK, y=1.0)
    fig.tight_layout()
    save(fig, "fig1_contrastive_pairs.png")


# =============================================================
# Fig 2: InfoNCE loss landscape at different temperatures
# =============================================================
def fig2_temperature_landscape():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---- Left: per-sample InfoNCE as a function of pos similarity ----
    ax = axes[0]
    sim_pos = np.linspace(-1, 1, 400)
    # 64 negatives drawn at fixed average similarity 0 (cosine on unit sphere)
    n_neg = 64
    sim_neg_mean = 0.0
    taus = [0.05, 0.1, 0.5, 1.0]
    colors = [C_RED, C_AMBER, C_GREEN, C_BLUE]

    for tau, col in zip(taus, colors):
        num = np.exp(sim_pos / tau)
        denom = num + n_neg * np.exp(sim_neg_mean / tau)
        loss = -np.log(num / denom)
        ax.plot(sim_pos, loss, color=col, lw=2.2, label=f"$\\tau$ = {tau}")

    ax.set_xlabel("Cosine similarity of positive pair $\\mathrm{sim}(z, z^+)$")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title("InfoNCE loss vs positive similarity (64 negatives at sim=0)")
    ax.legend(frameon=True, loc="upper right")
    ax.set_xlim(-1, 1)

    # ---- Right: gradient w.r.t. pos similarity (proxy for "pull strength") ----
    ax = axes[1]
    for tau, col in zip(taus, colors):
        num = np.exp(sim_pos / tau)
        denom = num + n_neg * np.exp(sim_neg_mean / tau)
        # d/d sim_pos of -log(num/denom) = -(1 - num/denom)/tau
        grad = -(1.0 - num / denom) / tau
        ax.plot(sim_pos, grad, color=col, lw=2.2, label=f"$\\tau$ = {tau}")

    ax.axhline(0, color=C_GREY, lw=0.8, linestyle="--")
    ax.set_xlabel("Cosine similarity of positive pair $\\mathrm{sim}(z, z^+)$")
    ax.set_ylabel("$\\partial \\mathcal{L}_{\\mathrm{InfoNCE}} / \\partial \\,\\mathrm{sim}(z, z^+)$")
    ax.set_title("Gradient sharpness: lower $\\tau$ = sharper, more selective")
    ax.legend(frameon=True, loc="lower right")
    ax.set_xlim(-1, 1)

    fig.suptitle("Temperature controls how 'picky' the loss is",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_contrastive_loss.png")


# =============================================================
# Fig 3: SimCLR (in-batch) vs MoCo (queue)
# =============================================================
def fig3_simclr_vs_moco():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # ---- Left: SimCLR ----
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title("SimCLR — in-batch negatives (single encoder $f_\\theta$)",
                 color=C_DARK)

    # Batch of 4 anchors -> 8 views
    batch_x = [1.5, 3.5, 5.5, 7.5]
    for i, x in enumerate(batch_x):
        # original
        ax.scatter(x, 8.5, s=320, c=C_BLUE, edgecolors="white", linewidths=1.8, zorder=4)
        ax.text(x, 8.5, f"x{i+1}", color="white", ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=5)
        # two views
        ax.scatter(x - 0.35, 6.5, s=260, c=C_GREEN, edgecolors="white", linewidths=1.5, zorder=4)
        ax.scatter(x + 0.35, 6.5, s=260, c=C_GREEN, edgecolors="white", linewidths=1.5, zorder=4)
        ax.annotate("", xy=(x - 0.35, 6.85), xytext=(x - 0.1, 8.15),
                    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.0))
        ax.annotate("", xy=(x + 0.35, 6.85), xytext=(x + 0.1, 8.15),
                    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.0))

    # encoder block
    enc_box = FancyBboxPatch((0.5, 4.6), 9.0, 1.0, boxstyle="round,pad=0.1",
                             linewidth=1.2, edgecolor=C_PURPLE,
                             facecolor="#ede9fe")
    ax.add_patch(enc_box)
    ax.text(5, 5.1, "shared encoder $f_\\theta$  +  projection head $g$",
            ha="center", va="center", fontsize=10.5, color=C_PURPLE, fontweight="bold")

    # batch of embeddings
    for i, x in enumerate(batch_x):
        ax.scatter(x - 0.35, 3.2, s=220, c=C_GREEN, edgecolors="white", linewidths=1.2, zorder=4)
        ax.scatter(x + 0.35, 3.2, s=220, c=C_GREEN, edgecolors="white", linewidths=1.2, zorder=4)

    # highlight one positive pair and several negatives
    ax.annotate("", xy=(batch_x[0] + 0.35, 3.2), xytext=(batch_x[0] - 0.35, 3.2),
                arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=2.0))
    ax.text(batch_x[0], 2.5, "positive\npair", ha="center", fontsize=8.5, color=C_GREEN)
    for x in batch_x[1:]:
        ax.annotate("", xy=(x - 0.35, 3.2), xytext=(batch_x[0] - 0.35, 3.2),
                    arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.0,
                                    linestyle="dashed"))
    ax.text(5.5, 1.5, "all other 2(B$-$1) views in the batch are negatives",
            ha="center", fontsize=9, color=C_AMBER)
    ax.text(5, 0.4, "B negatives = 2(B$-$1) per anchor — needs LARGE batches",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Right: MoCo ----
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title("MoCo — queue of negatives (momentum encoder $f_{\\xi}$)",
                 color=C_DARK)

    # query branch
    ax.scatter(1.8, 8.6, s=320, c=C_BLUE, edgecolors="white", linewidths=1.8, zorder=4)
    ax.text(1.8, 8.6, "x", color="white", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=5)
    ax.text(1.8, 9.3, "anchor", ha="center", fontsize=8.5, color=C_DARK)

    enc_q = FancyBboxPatch((0.6, 6.4), 2.4, 0.9, boxstyle="round,pad=0.08",
                           linewidth=1.2, edgecolor=C_BLUE, facecolor="#dbeafe")
    ax.add_patch(enc_q)
    ax.text(1.8, 6.85, "query encoder\n$f_\\theta$  (SGD)",
            ha="center", va="center", fontsize=8.5, color=C_BLUE)
    ax.annotate("", xy=(1.8, 7.3), xytext=(1.8, 8.3),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.0))

    ax.scatter(1.8, 5.4, s=220, c=C_BLUE, edgecolors="white", linewidths=1.2, zorder=4)
    ax.text(1.8, 4.85, "query  $q$", ha="center", fontsize=8.5, color=C_BLUE)
    ax.annotate("", xy=(1.8, 5.85), xytext=(1.8, 6.35),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.0))

    # key branch
    ax.scatter(4.6, 8.6, s=320, c=C_GREEN, edgecolors="white", linewidths=1.8, zorder=4)
    ax.text(4.6, 8.6, "x'", color="white", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=5)
    ax.text(4.6, 9.3, "augmented", ha="center", fontsize=8.5, color=C_DARK)

    enc_k = FancyBboxPatch((3.4, 6.4), 2.4, 0.9, boxstyle="round,pad=0.08",
                           linewidth=1.2, edgecolor=C_PURPLE, facecolor="#ede9fe")
    ax.add_patch(enc_k)
    ax.text(4.6, 6.85, "key encoder\n$f_\\xi$  (EMA)",
            ha="center", va="center", fontsize=8.5, color=C_PURPLE)
    ax.annotate("", xy=(4.6, 7.3), xytext=(4.6, 8.3),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.0))

    ax.scatter(4.6, 5.4, s=220, c=C_GREEN, edgecolors="white", linewidths=1.2, zorder=4)
    ax.text(4.6, 4.85, "positive key  $k_+$", ha="center", fontsize=8.5, color=C_GREEN)
    ax.annotate("", xy=(4.6, 5.85), xytext=(4.6, 6.35),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.0))

    # EMA arrow theta -> xi
    ax.annotate("", xy=(3.4, 6.85), xytext=(3.0, 6.85),
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.4,
                                linestyle="dashed"))
    ax.text(3.2, 7.25, "EMA", fontsize=8, color=C_AMBER, ha="center")

    # Queue
    queue_x0 = 6.4
    queue_y = 5.4
    n_q = 8
    for i in range(n_q):
        rect = Rectangle((queue_x0 + i * 0.42, queue_y - 0.35), 0.38, 0.7,
                         facecolor="#fef3c7", edgecolor=C_AMBER, linewidth=1.0)
        ax.add_patch(rect)
    ax.text(queue_x0 + n_q * 0.42 / 2, queue_y - 0.95,
            f"queue of K negatives  (e.g. K = 65 536)",
            ha="center", fontsize=8.5, color=C_AMBER)
    ax.text(queue_x0 + n_q * 0.42 / 2, queue_y + 0.95,
            "FIFO: enqueue $k_+$, dequeue oldest",
            ha="center", fontsize=8.5, color=C_DARK)

    # Arrow positive -> queue
    ax.annotate("", xy=(queue_x0 + 0.05, queue_y), xytext=(4.95, queue_y),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.4))

    # InfoNCE box
    nce = FancyBboxPatch((0.6, 2.4), 9.0, 1.4, boxstyle="round,pad=0.12",
                         linewidth=1.2, edgecolor=C_DARK, facecolor="#f1f5f9")
    ax.add_patch(nce)
    ax.text(5.1, 3.1,
            r"$\mathcal{L} = -\log\dfrac{\exp(q\cdot k_+/\tau)}{\exp(q\cdot k_+/\tau) + \sum_{k_i\in \text{queue}} \exp(q\cdot k_i/\tau)}$",
            ha="center", va="center", fontsize=11, color=C_DARK)

    ax.text(5, 1.0,
            "K is decoupled from batch size — millions of negatives become cheap",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("SimCLR vs MoCo: where do the negatives come from?",
                 fontsize=13, color=C_DARK, y=1.0)
    fig.tight_layout()
    save(fig, "fig3_simclr_vs_moco.png")


# =============================================================
# Fig 4: SGL augmentations on a user-item bipartite graph
# =============================================================
def fig4_sgl_augmentations():
    rng = np.random.default_rng(7)
    n_users, n_items = 4, 5
    user_x = np.full(n_users, 1.5)
    user_y = np.linspace(7.5, 1.5, n_users)
    item_x = np.full(n_items, 5.5)
    item_y = np.linspace(8.5, 0.5, n_items)

    # Original edge list (user, item)
    edges = [(0, 0), (0, 1), (0, 2),
             (1, 1), (1, 3),
             (2, 0), (2, 2), (2, 4),
             (3, 3), (3, 4)]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.2))

    def draw_panel(ax, kept_edges, dropped_users=(), dropped_items=(), title=""):
        ax.set_xlim(0, 7); ax.set_ylim(0, 9.5); ax.set_aspect("equal")
        ax.set_title(title, color=C_DARK, fontsize=11)
        # edges first
        for (u, i) in edges:
            kept = (u, i) in kept_edges
            color = C_GREY if kept else "#e5e7eb"
            ls = "-" if kept else ":"
            lw = 1.3 if kept else 0.9
            alpha = 0.9 if kept else 0.55
            ax.plot([user_x[u], item_x[i]], [user_y[u], item_y[i]],
                    color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=1)

        # users
        for u in range(n_users):
            faded = u in dropped_users
            col = "#cbd5e1" if faded else C_BLUE
            ax.scatter(user_x[u], user_y[u], s=420, c=col,
                       edgecolors="white", linewidths=1.6, zorder=3)
            ax.text(user_x[u], user_y[u], f"u{u+1}", color="white",
                    ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
        # items
        for i in range(n_items):
            faded = i in dropped_items
            col = "#cbd5e1" if faded else C_GREEN
            ax.scatter(item_x[i], item_y[i], s=420, c=col,
                       edgecolors="white", linewidths=1.6, zorder=3)
            ax.text(item_x[i], item_y[i], f"i{i+1}", color="white",
                    ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

        ax.text(1.5, 9.1, "users", ha="center", fontsize=9, color=C_BLUE, fontweight="bold")
        ax.text(5.5, 9.1, "items", ha="center", fontsize=9, color=C_GREEN, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Original
    draw_panel(axes[0], kept_edges=set(edges), title="Original user-item graph")

    # Edge dropout p=0.3
    rng2 = np.random.default_rng(11)
    keep_mask = rng2.random(len(edges)) > 0.3
    kept = {e for e, k in zip(edges, keep_mask) if k}
    draw_panel(axes[1], kept_edges=kept, title="Edge dropout (p = 0.3)")

    # Node dropout: drop u2 and i5
    drop_u, drop_i = {1}, {4}
    kept = {(u, i) for (u, i) in edges if u not in drop_u and i not in drop_i}
    draw_panel(axes[2], kept_edges=kept, dropped_users=drop_u,
               dropped_items=drop_i, title="Node dropout (drop u2, i5)")

    # Random walk subgraph rooted at u1, length 4
    rng3 = np.random.default_rng(3)
    # build adjacency
    adj_u = {u: [i for (uu, i) in edges if uu == u] for u in range(n_users)}
    adj_i = {i: [u for (u, ii) in edges if ii == i] for i in range(n_items)}
    walk_edges = set()
    cur = ("u", 0)
    for _ in range(8):
        if cur[0] == "u":
            nxt = int(rng3.choice(adj_u[cur[1]])) if adj_u[cur[1]] else None
            if nxt is None: break
            walk_edges.add((cur[1], nxt))
            cur = ("i", nxt)
        else:
            nxt = int(rng3.choice(adj_i[cur[1]])) if adj_i[cur[1]] else None
            if nxt is None: break
            walk_edges.add((nxt, cur[1]))
            cur = ("u", nxt)
    draw_panel(axes[3], kept_edges=walk_edges, title="Random-walk subgraph (start at u1)")

    fig.suptitle("SGL graph augmentations — three ways to create a second view",
                 fontsize=13, color=C_DARK, y=1.0)
    fig.tight_layout()
    save(fig, "fig4_sgl_augmentations.png")


# =============================================================
# Fig 5: CL4SRec sequence augmentations
# =============================================================
def fig5_cl4srec_augmentations():
    seq = ["i7", "i3", "i9", "i2", "i5", "i8", "i4", "i1"]
    n = len(seq)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    def draw_seq(y, items, label, highlight=None, mask_idx=None, reordered_pairs=None,
                 base_color=C_BLUE):
        ax.text(0.4, y, label, fontsize=10.5, color=C_DARK, fontweight="bold",
                va="center")
        for k, it in enumerate(items):
            x = 3.0 + k * 1.25
            color = base_color
            edge = "white"
            text_color = "white"
            if highlight == "drop" and (mask_idx is not None) and k in mask_idx:
                # cropped-out block — show ghosted
                color = "#e5e7eb"; text_color = "#94a3b8"
            if highlight == "mask" and (mask_idx is not None) and k in mask_idx:
                color = C_AMBER; it = "[M]"
            if highlight == "reorder" and reordered_pairs is not None:
                if k in reordered_pairs:
                    color = C_PURPLE
            box = FancyBboxPatch((x - 0.55, y - 0.45), 1.1, 0.9,
                                 boxstyle="round,pad=0.06",
                                 linewidth=1.2, edgecolor=edge, facecolor=color)
            ax.add_patch(box)
            ax.text(x, y, it, ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold")

    # Original sequence
    draw_seq(y=9.5, items=seq, label="original\nsequence")

    # Crop: keep contiguous subseq of length 5 starting at index 2
    cropped = list(seq)
    drop_idx = {0, 1, 7}
    draw_seq(y=7.0, items=cropped, label="crop\n(keep 5 of 8)",
             highlight="drop", mask_idx=drop_idx)
    ax.text(13.5, 7.0, "preserves local order", fontsize=8.5, color=C_DARK,
            style="italic", va="center")

    # Mask: replace 25% with [M]
    mask_idx = {1, 5}
    draw_seq(y=4.5, items=list(seq), label="mask\n($\\gamma$ = 0.25)",
             highlight="mask", mask_idx=mask_idx)
    ax.text(13.5, 4.5, "BERT-style — predict the hidden items", fontsize=8.5, color=C_DARK,
            style="italic", va="center")

    # Reorder: shuffle a contiguous chunk [2..5]
    reordered = list(seq)
    chunk = reordered[2:6][:]
    reordered[2:6] = [chunk[2], chunk[0], chunk[3], chunk[1]]
    draw_seq(y=2.0, items=reordered, label="reorder\n(shuffle a span)",
             highlight="reorder", reordered_pairs={2, 3, 4, 5})
    ax.text(13.5, 2.0, "model becomes order-tolerant", fontsize=8.5, color=C_DARK,
            style="italic", va="center")

    ax.set_title("CL4SRec — three sequence augmentations create the contrastive views",
                 fontsize=13, color=C_DARK, pad=12)

    # legend
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BLUE,
               markersize=11, label="original item"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e5e7eb",
               markersize=11, label="cropped out"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_AMBER,
               markersize=11, label="masked [M]"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_PURPLE,
               markersize=11, label="reordered"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", ncol=4, frameon=True,
              fontsize=9, bbox_to_anchor=(0.02, -0.02))

    fig.tight_layout()
    save(fig, "fig5_cl4srec_augmentations.png")


# =============================================================
# Fig 6: Performance gain from contrastive auxiliary loss
# =============================================================
def fig6_performance_gain():
    # Numbers are illustrative but consistent with reported magnitudes
    # in SGL (Wu et al., SIGIR'21) and SimGCL/XSimGCL (Yu et al., 2022/23).
    datasets = ["Yelp2018", "Amazon-Book", "Alibaba-iFashion", "Gowalla"]
    base_recall = np.array([0.0639, 0.0411, 0.1135, 0.1830])  # LightGCN-ish baselines
    sgl_recall = np.array([0.0709, 0.0478, 0.1268, 0.1909])
    xsimgcl_recall = np.array([0.0723, 0.0498, 0.1289, 0.1929])

    base_ndcg = np.array([0.0525, 0.0315, 0.0628, 0.1554])
    sgl_ndcg = np.array([0.0584, 0.0379, 0.0703, 0.1605])
    xsimgcl_ndcg = np.array([0.0598, 0.0394, 0.0719, 0.1626])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    width = 0.26
    x = np.arange(len(datasets))

    for ax, (b, s, xg, metric) in zip(
        axes,
        [(base_recall, sgl_recall, xsimgcl_recall, "Recall@20"),
         (base_ndcg, sgl_ndcg, xsimgcl_ndcg, "NDCG@20")]):
        ax.bar(x - width, b, width, label="LightGCN (no CL)", color=C_GREY,
               edgecolor="white")
        ax.bar(x, s, width, label="+ SGL", color=C_BLUE, edgecolor="white")
        ax.bar(x + width, xg, width, label="+ XSimGCL", color=C_PURPLE,
               edgecolor="white")
        # gain annotation
        for i in range(len(datasets)):
            gain = (xg[i] - b[i]) / b[i] * 100
            ymax = max(b[i], s[i], xg[i])
            ax.text(x[i] + width, xg[i] + ymax * 0.015,
                    f"+{gain:.1f}%", ha="center", fontsize=8.5, color=C_PURPLE,
                    fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — auxiliary contrastive loss vs none", color=C_DARK)
        ax.legend(frameon=True, loc="upper right", fontsize=8.5)
        ax.set_ylim(0, max(xg) * 1.22)

    fig.suptitle("Adding a contrastive auxiliary loss to a GNN recommender",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_performance_gain.png")


# =============================================================
# Fig 7: Embedding space before vs after contrastive training (t-SNE-like)
# =============================================================
def fig7_embedding_space():
    rng = np.random.default_rng(42)
    n_clusters = 5
    n_per = 80

    # ---- "Before": collapsed/entangled — clusters overlap heavily ----
    centers_before = rng.uniform(-1.5, 1.5, size=(n_clusters, 2))
    pts_before, labels_before = [], []
    for c, ctr in enumerate(centers_before):
        pts_before.append(ctr + rng.normal(scale=2.2, size=(n_per, 2)))
        labels_before.extend([c] * n_per)
    pts_before = np.vstack(pts_before)
    labels_before = np.array(labels_before)

    # ---- "After": well separated, tight clusters ----
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers_after = np.stack([6 * np.cos(angles), 6 * np.sin(angles)], axis=1)
    pts_after, labels_after = [], []
    for c, ctr in enumerate(centers_after):
        pts_after.append(ctr + rng.normal(scale=0.55, size=(n_per, 2)))
        labels_after.extend([c] * n_per)
    pts_after = np.vstack(pts_after)
    labels_after = np.array(labels_after)

    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_RED]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    # plot before
    ax = axes[0]
    for c in range(n_clusters):
        m = labels_before == c
        ax.scatter(pts_before[m, 0], pts_before[m, 1], s=22, c=palette[c],
                   alpha=0.55, edgecolors="white", linewidths=0.4,
                   label=f"interest group {c+1}")
    ax.set_title("Before contrastive training\n(item embeddings entangled, no clear groups)",
                 color=C_DARK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")

    ax = axes[1]
    for c in range(n_clusters):
        m = labels_after == c
        ax.scatter(pts_after[m, 0], pts_after[m, 1], s=22, c=palette[c],
                   alpha=0.85, edgecolors="white", linewidths=0.4,
                   label=f"interest group {c+1}")
    ax.set_title("After contrastive training\n(tight, well-separated interest clusters)",
                 color=C_DARK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle("Contrastive training reshapes the embedding space (uniformity + alignment)",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_embedding_space.png")


# =============================================================
def main():
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    fig1_positive_negative_pairs()
    fig2_temperature_landscape()
    fig3_simclr_vs_moco()
    fig4_sgl_augmentations()
    fig5_cl4srec_augmentations()
    fig6_performance_gain()
    fig7_embedding_space()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
