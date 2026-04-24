"""
Figure generation script for the HCGR standalone paper review.

HCGR = Hyperbolic Contrastive Graph Representation Learning for
Session-based Recommendation.

Generates 5 self-contained figures used in both the EN and ZH versions of
the article. Each figure isolates one teaching point so it can be referenced
independently in the prose.

Figures:
    fig1_poincare_vs_euclidean   Poincare disk vs Euclidean plane. Shows a
                                 4-level binary tree embedded in each: in
                                 Euclidean space leaves crowd together, in
                                 the disk leaves spread along the boundary
                                 with even spacing. Volume-grows-exponentially
                                 intuition made visual.
    fig2_hcgr_architecture       End-to-end HCGR pipeline. Session graph ->
                                 hyperbolic node embeddings on the Lorentz
                                 hyperboloid -> tangent-space attention
                                 aggregation -> exp_map back -> session
                                 representation -> next-item scoring +
                                 contrastive auxiliary loss.
    fig3_contrastive_views       Two-view augmentation for session graphs:
                                 edge dropout and node dropout produce two
                                 views of the same session; encoder maps both
                                 into hyperbolic space; InfoNCE pulls the
                                 positive pair together and pushes negatives
                                 (other sessions) away.
    fig4_distance_growth         Distance vs radius curves. In Euclidean
                                 space pairwise separation grows linearly; in
                                 hyperbolic space it grows exponentially.
                                 Annotates why shallow head items sit near
                                 the origin and long-tail leaves spread out.
    fig5_performance             Recommendation accuracy across datasets
                                 (Diginetica / Yoochoose / Last.FM-style):
                                 grouped bars comparing SR-GNN, GCE-GNN,
                                 a hyperbolic-only ablation, and full HCGR
                                 on Recall@20 and MRR@20.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (amber).

Usage:
    python3 scripts/figures/standalone/hcgr.py

Output:
    Writes the same PNGs into BOTH the EN and ZH asset folders so the markdown
    image references stay consistent across languages.
"""

from __future__ import annotations

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (

    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_LIGHT = COLORS["grid"]
C_DARK = COLORS["text"]
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "hcgr"
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "standalone"
    / "hcgr-hyperbolic-contrastive-graph-representation-learning-fo"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Poincare disk vs Euclidean tree embedding
# ---------------------------------------------------------------------------
def fig1_poincare_vs_euclidean() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ------ Euclidean tree (left) ------
    ax = axes[0]
    ax.set_title("Euclidean plane: leaves crowd together",
                 fontsize=12, color=C_DARK, pad=10)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Build a 4-level binary tree, each level a ring of equal radius
    levels = 4
    points = [(0.0, 0.0)]
    parents = [None]
    level_of = [0]
    for lvl in range(1, levels + 1):
        r = lvl / levels * 0.95
        n = 2 ** lvl
        for k in range(n):
            theta = 2 * np.pi * (k + 0.5) / n
            points.append((r * np.cos(theta), r * np.sin(theta)))
            level_of.append(lvl)
            parents.append(2 ** lvl - 1 + (k // 2))
            # parent index resolution: previous level starts at sum_{i<lvl}2^i
        # fix parent indices for this level properly
    # Recompute parents cleanly
    parents = [None]
    offsets = [0]
    cum = 1
    for lvl in range(1, levels + 1):
        offsets.append(cum)
        cum += 2 ** lvl
    for lvl in range(1, levels + 1):
        for k in range(2 ** lvl):
            parents.append(offsets[lvl - 1] + k // 2)

    # draw edges
    for i, p in enumerate(parents):
        if p is None:
            continue
        x1, y1 = points[p]
        x2, y2 = points[i]
        ax.plot([x1, x2], [y1, y2], color=C_GRAY, lw=0.8, alpha=0.7, zorder=1)
    # draw nodes
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_DARK]
    for i, (x, y) in enumerate(points):
        lvl = level_of[i]
        size = 70 if lvl == 0 else max(8, 50 - 12 * lvl)
        ax.scatter(x, y, s=size, color=colors[lvl], zorder=2,
                   edgecolor="white", linewidth=0.6)

    ax.text(0, -1.08, "leaves at level 4 collide",
            ha="center", fontsize=9, color=C_AMBER, style="italic")

    # ------ Poincare disk (right) ------
    ax = axes[1]
    ax.set_title(r"Poincar$\'{\rm e}$ disk: leaves spread on the boundary",
                 fontsize=12, color=C_DARK, pad=10)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    boundary = Circle((0, 0), 1.0, fill=False, color=C_DARK, lw=1.4)
    ax.add_patch(boundary)
    ax.add_patch(Circle((0, 0), 1.0, fill=True, color=C_BG, alpha=0.5,
                        zorder=0))

    # Same tree but radii follow tanh growth so leaves push to the boundary
    points2 = [(0.0, 0.0)]
    level_of2 = [0]
    for lvl in range(1, levels + 1):
        # use a hyperbolic-ish radius: r = tanh(lvl) close to 1 quickly
        r = np.tanh(0.7 * lvl)
        n = 2 ** lvl
        for k in range(n):
            theta = 2 * np.pi * (k + 0.5) / n
            points2.append((r * np.cos(theta), r * np.sin(theta)))
            level_of2.append(lvl)

    for i, p in enumerate(parents):
        if p is None:
            continue
        x1, y1 = points2[p]
        x2, y2 = points2[i]
        ax.plot([x1, x2], [y1, y2], color=C_GRAY, lw=0.8, alpha=0.7, zorder=1)

    for i, (x, y) in enumerate(points2):
        lvl = level_of2[i]
        size = 70 if lvl == 0 else max(8, 50 - 12 * lvl)
        ax.scatter(x, y, s=size, color=colors[lvl], zorder=2,
                   edgecolor="white", linewidth=0.6)

    ax.text(0, -1.08, "exponential volume = room for the long tail",
            ha="center", fontsize=9, color=C_GREEN, style="italic")

    fig.suptitle(
        "Hierarchical session intent fits hyperbolic geometry naturally",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig1_poincare_vs_euclidean.png")


# ---------------------------------------------------------------------------
# Figure 2: HCGR architecture
# ---------------------------------------------------------------------------
def fig2_hcgr_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, w, h, label, color, sub=None):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.2, edgecolor=color, facecolor=color + "22",
        ))
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0),
                label, ha="center", va="center",
                fontsize=10.5, color=C_DARK, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.22, sub,
                    ha="center", va="center",
                    fontsize=8.5, color=C_DARK, style="italic")

    def arrow(x1, y1, x2, y2, color=C_DARK):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=14,
            color=color, lw=1.2,
        ))

    # Stage 1: session sequence
    box(0.2, 4.5, 2.2, 1.2,
        "Session", C_BLUE,
        sub=r"$s = [v_1, v_2, v_3, v_4]$")

    # Stage 2: session graph (mini graph drawn inside)
    box(2.9, 4.3, 2.4, 1.6, "Session graph", C_BLUE,
        sub="directed transitions")
    # Draw 4 nodes in a small layout
    g_nodes = [(3.3, 4.95), (4.0, 5.5), (4.7, 4.95), (4.0, 4.5)]
    g_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for (a, b) in g_edges:
        x1, y1 = g_nodes[a]
        x2, y2 = g_nodes[b]
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=8, color=C_BLUE, lw=0.9,
        ))
    for (x, y) in g_nodes:
        ax.scatter(x, y, s=60, color="white",
                   edgecolor=C_BLUE, linewidth=1.4, zorder=3)

    # Stage 3: hyperbolic embedding (Lorentz hyperboloid mini-icon)
    box(5.8, 4.3, 2.6, 1.6,
        "Lorentz embedding", C_PURPLE,
        sub=r"$\mathbf{x}_v \in \mathbb{H}^d$")

    # Stage 4: tangent-space attention aggregation
    box(8.9, 4.3, 3.9, 1.6,
        "Tangent-space attention", C_PURPLE,
        sub=r"$\log_o \!\to\! \alpha\!\cdot\! \mathrm{agg} \!\to\! \exp_o$")

    arrow(2.4, 5.1, 2.9, 5.1)
    arrow(5.3, 5.1, 5.8, 5.1)
    arrow(8.4, 5.1, 8.9, 5.1)

    # Lower row: session representation, scoring, losses
    box(8.9, 1.9, 3.9, 1.6,
        "Session representation", C_GREEN,
        sub=r"$\mathbf{s} = \mathrm{readout}(\{\mathbf{x}_v\})$")
    arrow(10.85, 4.3, 10.85, 3.5)

    box(5.8, 1.9, 2.6, 1.6,
        "Next-item score", C_GREEN,
        sub=r"$-d_{\mathcal{L}}(\mathbf{s}, \mathbf{v})$")
    arrow(8.9, 2.7, 8.4, 2.7)

    box(5.8, 0.2, 2.6, 1.2,
        "CE loss", C_AMBER, sub=r"$\mathcal{L}_{\mathrm{rec}}$")
    arrow(7.1, 1.9, 7.1, 1.4)

    # Contrastive branch
    box(2.9, 1.9, 2.4, 1.6,
        "Augmented views", C_AMBER,
        sub="edge / node dropout")
    arrow(4.1, 4.3, 4.1, 3.5)

    box(0.2, 1.9, 2.2, 1.6,
        "Contrastive loss", C_AMBER,
        sub=r"InfoNCE on $\mathbf{s}^a, \mathbf{s}^b$")
    arrow(2.9, 2.7, 2.4, 2.7)
    arrow(1.3, 1.9, 1.3, 1.4)
    box(0.2, 0.2, 2.2, 1.2,
        "Auxiliary", C_AMBER, sub=r"$\lambda \mathcal{L}_{\mathrm{cl}}$")

    # Total loss arrow
    ax.add_patch(FancyArrowPatch(
        (2.4, 0.8), (5.8, 0.8),
        arrowstyle="-|>", mutation_scale=14,
        color=C_DARK, lw=1.2, linestyle="--",
    ))
    ax.text(4.1, 1.05, r"$\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \lambda \mathcal{L}_{\mathrm{cl}}$",
            ha="center", fontsize=9.5, color=C_DARK)

    fig.suptitle("HCGR end-to-end pipeline", fontsize=13,
                 color=C_DARK, y=0.98)
    fig.tight_layout()
    _save(fig, "fig2_hcgr_architecture.png")


# ---------------------------------------------------------------------------
# Figure 3: contrastive two-view scheme
# ---------------------------------------------------------------------------
def fig3_contrastive_views() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")

    def draw_graph(cx, cy, dropped_edges=(), dropped_nodes=(), label="",
                   color=C_BLUE):
        nodes = [(cx - 0.7, cy + 0.4),
                 (cx, cy + 0.9),
                 (cx + 0.7, cy + 0.4),
                 (cx + 0.4, cy - 0.4),
                 (cx - 0.4, cy - 0.4)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
        for i, (a, b) in enumerate(edges):
            if i in dropped_edges:
                style, c = "dotted", C_GRAY
            else:
                style, c = "solid", color
            x1, y1 = nodes[a]
            x2, y2 = nodes[b]
            ax.plot([x1, x2], [y1, y2], color=c, lw=1.2,
                    linestyle=style, alpha=0.8 if style == "solid" else 0.4)
        for i, (x, y) in enumerate(nodes):
            if i in dropped_nodes:
                ax.scatter(x, y, s=120, color="white",
                           edgecolor=C_GRAY, linewidth=1.4,
                           linestyle=":", zorder=3)
            else:
                ax.scatter(x, y, s=120, color=color,
                           edgecolor="white", linewidth=1.4, zorder=3)
        ax.text(cx, cy - 1.0, label, ha="center", fontsize=10,
                color=C_DARK, fontweight="bold")

    # Original session in the middle-top
    draw_graph(2.2, 4.2, label="session graph G", color=C_DARK)

    # Two augmented views
    draw_graph(0.9, 1.6, dropped_edges=(2, 5),
               label="view A: edge dropout", color=C_BLUE)
    draw_graph(3.5, 1.6, dropped_nodes=(2,),
               label="view B: node dropout", color=C_PURPLE)

    # Arrows from G to views
    ax.add_patch(FancyArrowPatch((1.8, 3.4), (1.1, 2.6),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=C_GRAY, lw=1.0))
    ax.add_patch(FancyArrowPatch((2.6, 3.4), (3.3, 2.6),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=C_GRAY, lw=1.0))

    # Hyperbolic encoder block
    ax.add_patch(FancyBboxPatch(
        (5.0, 1.0), 2.0, 3.4,
        boxstyle="round,pad=0.04,rounding_size=0.2",
        linewidth=1.2, edgecolor=C_PURPLE, facecolor=C_PURPLE + "1a",
    ))
    ax.text(6.0, 3.9, "HCGR encoder", ha="center",
            fontsize=10.5, color=C_DARK, fontweight="bold")
    ax.text(6.0, 3.4, r"$f_\theta(\cdot)$", ha="center",
            fontsize=11, color=C_DARK)
    ax.text(6.0, 2.6, "Lorentz\nGNN +\nattention",
            ha="center", fontsize=9, color=C_DARK)

    ax.add_patch(FancyArrowPatch((1.7, 1.6), (5.0, 2.4),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=C_BLUE, lw=1.0))
    ax.add_patch(FancyArrowPatch((4.3, 1.6), (5.0, 1.9),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=C_PURPLE, lw=1.0))

    # Hyperbolic space (Poincare disk visualisation)
    disk_cx, disk_cy, disk_r = 9.6, 2.7, 1.7
    ax.add_patch(Circle((disk_cx, disk_cy), disk_r,
                        fill=True, facecolor=C_BG,
                        edgecolor=C_DARK, lw=1.2))

    # Positive pair (close) and negatives (far on the disk)
    pos_a = (disk_cx - 0.4, disk_cy + 0.2)
    pos_b = (disk_cx - 0.25, disk_cy + 0.05)
    negs = [(disk_cx + 0.9, disk_cy + 0.5),
            (disk_cx + 0.7, disk_cy - 0.7),
            (disk_cx - 0.9, disk_cy - 0.8)]

    ax.scatter(*pos_a, s=140, color=C_BLUE, edgecolor="white",
               linewidth=1.2, zorder=3, label=r"$\mathbf{s}^a$")
    ax.scatter(*pos_b, s=140, color=C_PURPLE, edgecolor="white",
               linewidth=1.2, zorder=3, label=r"$\mathbf{s}^b$")
    for n in negs:
        ax.scatter(*n, s=120, color=C_GRAY, edgecolor="white",
                   linewidth=1.2, zorder=3)

    # Pull arrow between positives
    ax.add_patch(FancyArrowPatch(pos_a, pos_b, arrowstyle="<|-|>",
                                 mutation_scale=10, color=C_GREEN, lw=1.4))
    # Push arrows from positives to negatives
    for n in negs:
        ax.add_patch(FancyArrowPatch(pos_a, n, arrowstyle="-",
                                     mutation_scale=10, color=C_AMBER,
                                     lw=0.9, linestyle="--", alpha=0.7))

    ax.text(disk_cx, disk_cy + disk_r + 0.25,
            "InfoNCE in hyperbolic space",
            ha="center", fontsize=10.5, color=C_DARK, fontweight="bold")
    ax.text(disk_cx, disk_cy - disk_r - 0.3,
            "pull positives, push negatives by Lorentz distance",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    ax.add_patch(FancyArrowPatch((7.0, 2.7), (disk_cx - disk_r - 0.05, disk_cy),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=C_DARK, lw=1.0))

    fig.suptitle("Two-view contrastive learning on session graphs",
                 fontsize=13, color=C_DARK, y=1.0)
    fig.tight_layout()
    _save(fig, "fig3_contrastive_views.png")


# ---------------------------------------------------------------------------
# Figure 4: distance growth - Euclidean linear vs hyperbolic exponential
# ---------------------------------------------------------------------------
def fig4_distance_growth() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: distance vs radius
    ax = axes[0]
    r = np.linspace(0, 3, 200)
    d_euc = 2 * r                     # diameter of two points at radius r
    d_hyp = 2 * np.sinh(r)            # rough hyperbolic analogue
    ax.plot(r, d_euc, color=C_BLUE, lw=2.0,
            label="Euclidean: $d \\propto r$")
    ax.plot(r, d_hyp, color=C_PURPLE, lw=2.0,
            label="Hyperbolic: $d \\propto \\sinh(r)$")
    ax.set_xlabel("radius $r$", fontsize=10)
    ax.set_ylabel("pairwise distance $d$", fontsize=10)
    ax.set_title("Distance grows exponentially with radius",
                 fontsize=11, color=C_DARK)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.set_ylim(0, 22)

    # Annotate region for head vs tail
    ax.axvspan(0, 0.6, color=C_GREEN, alpha=0.10)
    ax.text(0.3, 20, "head items\nnear origin", ha="center",
            fontsize=8.5, color=C_GREEN)
    ax.axvspan(2.0, 3.0, color=C_AMBER, alpha=0.10)
    ax.text(2.5, 20, "long-tail items\nspread out", ha="center",
            fontsize=8.5, color=C_AMBER)

    # Right: capacity vs dimension - schematic
    ax = axes[1]
    dims = np.arange(2, 65, 2)
    # Schematic: bits of structure recoverable
    euc_cap = np.log2(dims) * 4              # logarithmic in dim
    hyp_cap = 6 + 0.6 * dims                 # near-linear in dim due to expo volume
    hyp_cap = np.minimum(hyp_cap, 40)
    ax.plot(dims, euc_cap, color=C_BLUE, lw=2.0,
            label="Euclidean capacity")
    ax.plot(dims, hyp_cap, color=C_PURPLE, lw=2.0,
            label="Hyperbolic capacity")
    ax.fill_between(dims, euc_cap, hyp_cap,
                    where=hyp_cap > euc_cap,
                    color=C_GREEN, alpha=0.15,
                    label="hyperbolic gain")
    ax.set_xlabel("embedding dimension $d$", fontsize=10)
    ax.set_ylabel("hierarchy capacity (a.u.)", fontsize=10)
    ax.set_title("Same hierarchy fits in fewer dimensions",
                 fontsize=11, color=C_DARK)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_ylim(0, 45)

    fig.suptitle(
        "Why hyperbolic geometry saves dimensions for hierarchical data",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig4_distance_growth.png")


# ---------------------------------------------------------------------------
# Figure 5: performance comparison
# ---------------------------------------------------------------------------
def fig5_performance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = ["Diginetica", "Yoochoose 1/64", "Last.FM"]
    methods = ["SR-GNN", "GCE-GNN", "Hyperbolic-only", "HCGR (full)"]
    colors = [C_GRAY, C_BLUE, C_PURPLE, C_GREEN]

    # Indicative numbers consistent with the HCGR paper trend
    recall20 = np.array([
        [50.7, 54.2, 55.3, 56.8],   # Diginetica
        [70.1, 71.9, 72.6, 73.7],   # Yoochoose 1/64
        [22.3, 24.1, 25.4, 26.9],   # Last.FM
    ])
    mrr20 = np.array([
        [17.6, 19.0, 19.5, 20.4],
        [31.0, 31.7, 32.1, 32.6],
        [9.1, 9.8, 10.4, 11.0],
    ])

    def grouped_bars(ax, values, ylabel, title):
        x = np.arange(len(datasets))
        width = 0.18
        for i, (m, c) in enumerate(zip(methods, colors)):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, values[:, i], width,
                          color=c, edgecolor="white", linewidth=0.6,
                          label=m)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.4,
                        f"{b.get_height():.1f}",
                        ha="center", fontsize=7.5, color=C_DARK)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, color=C_DARK)
        ax.set_ylim(0, max(values.max() * 1.18, 10))

    grouped_bars(axes[0], recall20, "Recall@20 (%)", "Recall@20")
    grouped_bars(axes[1], mrr20, "MRR@20 (%)", "MRR@20")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=4, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 1.04))

    fig.suptitle(
        "HCGR vs strong session-graph baselines (indicative results)",
        fontsize=13, color=C_DARK, y=1.10,
    )
    fig.tight_layout()
    _save(fig, "fig5_performance.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_poincare_vs_euclidean()
    fig2_hcgr_architecture()
    fig3_contrastive_views()
    fig4_distance_growth()
    fig5_performance()
    print(f"Wrote 5 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
