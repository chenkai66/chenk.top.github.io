"""
Figure generation script for the GNN Equivariant Representations paper review.

Generates 5 self-contained figures used in both the EN and ZH versions of the
article. Each figure isolates exactly one teaching point so it can be referenced
independently in the prose.

Figures:
    fig1_permutation_equivariance   The defining property: permuting the input
                                    permutes the output the same way. Two MLP
                                    diagrams stacked, with arrows showing the
                                    same permutation acting on both sides.
    fig2_neural_graph               How an MLP becomes a directed graph. Layers
                                    of neurons become nodes; weight matrices
                                    become typed edges; biases become node
                                    features.
    fig3_gnn_pipeline               End-to-end pipeline: weights -> neural
                                    graph -> message passing -> node embeddings
                                    -> pooling -> downstream head.
    fig4_equivariant_vs_invariant   Side-by-side contrast: invariant pooling
                                    collapses to one vector that ignores
                                    permutation; equivariant node embeddings
                                    permute together with the inputs.
    fig5_generalization_prediction  Application: given trained networks as
                                    graphs, a GNN predicts test accuracy
                                    without running validation. Scatter of
                                    predicted vs true.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (amber).

Usage:
    python3 scripts/figures/standalone/gnn-equivariant.py

Output:
    Writes PNGs into BOTH the EN and ZH asset folders so the markdown image
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#94a3b8"
C_LIGHT = "#e2e8f0"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "gnn-equivariant-representations"
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "standalone"
    / "graph-neural-networks-for-learning-equivariant-representatio"
)


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=10,
         fontweight="normal", text_color="white", alpha=1.0,
         rounding=0.05):
    ec = ec or fc
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.2, facecolor=fc, edgecolor=ec, alpha=alpha,
    )
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha="center", va="center",
                color=text_color, fontsize=fontsize,
                fontweight=fontweight)


def _arrow(ax, xy_from, xy_to, color=C_DARK, lw=1.4,
           style="-|>", connection="arc3,rad=0", alpha=1.0):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw, connectionstyle=connection, alpha=alpha,
    )
    ax.add_patch(arr)


def _neuron(ax, xy, r=0.22, fc=C_BLUE, ec=C_DARK, label=None,
            label_color="white", fontsize=10, lw=1.2):
    c = Circle(xy, r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(c)
    if label is not None:
        ax.text(xy[0], xy[1], label, ha="center", va="center",
                color=label_color, fontsize=fontsize, fontweight="bold",
                zorder=4)


# ---------------------------------------------------------------------------
# Figure 1: Permutation equivariance
# ---------------------------------------------------------------------------
def fig1_permutation_equivariance() -> None:
    """Same MLP, hidden units permuted: the function is unchanged."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    fig.subplots_adjust(wspace=0.18)

    def draw_mlp(ax, hidden_order, title, highlight=False):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, color=C_DARK, fontweight="bold",
                     pad=6)

        # input layer (2)
        in_xy = [(0.8, 3.4), (0.8, 1.6)]
        for i, p in enumerate(in_xy):
            _neuron(ax, p, r=0.26, fc=C_GRAY, ec=C_DARK,
                    label=f"x{i+1}", label_color="white", fontsize=10)

        # hidden layer (3) with given order
        hid_y = [4.0, 2.5, 1.0]
        hid_xy = [(3.0, y) for y in hid_y]
        hid_colors = [C_BLUE, C_PURPLE, C_GREEN]
        for slot, idx in enumerate(hidden_order):
            _neuron(ax, hid_xy[slot], r=0.30, fc=hid_colors[idx],
                    ec=C_DARK, label=f"h{idx+1}", fontsize=10)

        # output layer (1)
        out_xy = (5.2, 2.5)
        _neuron(ax, out_xy, r=0.28, fc=C_AMBER, ec=C_DARK,
                label="y", fontsize=11)

        # edges input -> hidden  (W1, columns permute together with hidden)
        for ix in in_xy:
            for slot in range(3):
                _arrow(ax, ix, hid_xy[slot], color=C_GRAY, lw=0.9,
                       style="-", alpha=0.7)
        # hidden -> output (W2, rows permute with hidden)
        for slot in range(3):
            _arrow(ax, hid_xy[slot], out_xy, color=C_GRAY, lw=0.9,
                   style="-", alpha=0.7)

        if highlight:
            for slot in range(3):
                ax.add_patch(Circle(hid_xy[slot], 0.42, fill=False,
                                    edgecolor=C_AMBER, linewidth=1.6,
                                    linestyle="--", zorder=2))

    draw_mlp(axes[0], hidden_order=[0, 1, 2],
             title="Original MLP:  hidden order [h1, h2, h3]")
    draw_mlp(axes[1], hidden_order=[1, 2, 0],
             title="Permuted MLP:  hidden order [h2, h3, h1]",
             highlight=True)

    # bottom caption
    fig.text(0.5, -0.02,
             r"Permute hidden units AND permute the matching rows of $W_1$ "
             r"and columns of $W_2$  $\Rightarrow$  identical function $f(x)$, "
             r"completely different parameter vector.",
             ha="center", fontsize=11, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                       boxstyle="round,pad=0.5"))

    fig.suptitle(
        "Figure 1. Permutation equivariance: hidden-unit symmetry of an MLP",
        fontsize=12.8, color=C_DARK, y=1.02,
    )
    _save(fig, "fig1_permutation_equivariance")


# ---------------------------------------------------------------------------
# Figure 2: Neural network as a graph
# ---------------------------------------------------------------------------
def fig2_neural_graph() -> None:
    """An MLP rewritten as a directed graph: nodes = neurons, edges = weights."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    fig.subplots_adjust(wspace=0.22)

    # ---- LEFT: MLP weight tensors ----
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("(a) Parameters as tensors", fontsize=12.5,
                 color=C_DARK, fontweight="bold", pad=6)

    # W1: 3x2
    _box(ax, (0.5, 3.0), 1.6, 1.8, "", C_BLUE, alpha=0.85, rounding=0.05)
    ax.text(1.3, 3.9, r"$W_1 \in \mathbb{R}^{3\times 2}$",
            ha="center", va="center", color="white", fontsize=11,
            fontweight="bold")
    ax.text(1.3, 2.7, "weights\ninput -> hidden",
            ha="center", va="center", color=C_DARK, fontsize=9)

    # b1: 3x1
    _box(ax, (2.4, 3.0), 0.55, 1.8, "", C_PURPLE, alpha=0.85, rounding=0.05)
    ax.text(2.68, 3.9, r"$b_1$", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    ax.text(2.68, 2.7, "biases\n(hidden)", ha="center", va="center",
            color=C_DARK, fontsize=9)

    # W2: 1x3
    _box(ax, (3.4, 4.0), 1.8, 0.6, "", C_GREEN, alpha=0.85, rounding=0.05)
    ax.text(4.3, 4.3, r"$W_2 \in \mathbb{R}^{1\times 3}$",
            ha="center", va="center", color="white", fontsize=11,
            fontweight="bold")
    ax.text(4.3, 3.7, "weights  hidden -> output", ha="center",
            va="center", color=C_DARK, fontsize=9)

    # b2
    _box(ax, (3.4, 3.0), 0.55, 0.6, "", C_AMBER, alpha=0.85, rounding=0.05)
    ax.text(3.68, 3.3, r"$b_2$", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")

    ax.text(3.0, 1.6,
            "Flat parameter vector\n"
            r"$\theta = \mathrm{vec}(W_1, b_1, W_2, b_2)$"
            "\nLoses topology and changes\nunder hidden permutation.",
            ha="center", va="center", fontsize=10, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.5"))

    # ---- RIGHT: neural graph ----
    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("(b) Same MLP as a neural graph",
                 fontsize=12.5, color=C_DARK, fontweight="bold", pad=6)

    in_xy = [(0.8, 3.8), (0.8, 1.8)]
    hid_xy = [(3.1, 4.4), (3.1, 2.8), (3.1, 1.2)]
    out_xy = (5.3, 2.8)

    # edges with weight labels (W1 entries)
    w1_vals = [[0.7, -0.2], [0.1, 0.9], [-0.5, 0.4]]
    for j, hp in enumerate(hid_xy):
        for i, ip in enumerate(in_xy):
            _arrow(ax, ip, hp, color=C_BLUE, lw=1.0, style="-|>",
                   alpha=0.85)
            mid = ((ip[0] + hp[0]) / 2, (ip[1] + hp[1]) / 2)
            ax.text(mid[0], mid[1] + 0.08, f"{w1_vals[j][i]:+.1f}",
                    fontsize=7.5, color=C_BLUE, ha="center")

    # W2 edges
    w2_vals = [0.6, -0.3, 0.8]
    for j, hp in enumerate(hid_xy):
        _arrow(ax, hp, out_xy, color=C_GREEN, lw=1.1, style="-|>",
               alpha=0.9)
        mid = ((hp[0] + out_xy[0]) / 2, (hp[1] + out_xy[1]) / 2)
        ax.text(mid[0], mid[1] + 0.1, f"{w2_vals[j]:+.1f}",
                fontsize=7.5, color=C_GREEN, ha="center")

    # nodes
    for i, p in enumerate(in_xy):
        _neuron(ax, p, r=0.26, fc=C_GRAY, ec=C_DARK,
                label=f"x{i+1}", fontsize=9)
    for j, p in enumerate(hid_xy):
        _neuron(ax, p, r=0.28, fc=C_PURPLE, ec=C_DARK,
                label=f"h{j+1}", fontsize=9)
        ax.text(p[0] + 0.35, p[1] - 0.05, f"b={0.1*(j+1):.1f}",
                fontsize=7.5, color=C_PURPLE)
    _neuron(ax, out_xy, r=0.28, fc=C_AMBER, ec=C_DARK,
            label="y", fontsize=10)

    # legend
    ax.text(0.4, 0.4,
            "Node feature: bias\nEdge feature: weight\n"
            "Permuting hidden NODES = relabelling -> graph is the SAME.",
            ha="left", va="bottom", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    fig.suptitle(
        "Figure 2. Neural network as a graph: weights -> typed directed edges",
        fontsize=12.8, color=C_DARK, y=1.02,
    )
    _save(fig, "fig2_neural_graph")


# ---------------------------------------------------------------------------
# Figure 3: GNN architecture pipeline
# ---------------------------------------------------------------------------
def fig3_gnn_pipeline() -> None:
    """End-to-end: weights -> graph -> message passing -> embeddings -> head."""
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    stages = [
        (0.2, C_BLUE,   "1. Trained\nnetwork",
         r"$\theta = \{W_\ell, b_\ell\}$"),
        (2.7, C_PURPLE, "2. Build neural\ngraph",
         "nodes = neurons\nedges = weights"),
        (5.2, C_GREEN,  "3. Message passing\n(L GNN layers)",
         r"$h_v \leftarrow \mathrm{UPD}(h_v,$"
         "\n"
         r"$\;\bigoplus_{u} \mathrm{MSG}(h_u, e_{uv}))$"),
        (8.0, C_AMBER,  "4. Pool to graph\nembedding",
         r"$z_G = \mathrm{POOL}(\{h_v^{(L)}\}_v)$"),
        (10.6, C_BLUE,  "5. Downstream\nhead",
         "regress acc.\nclassify task\nretrieve / merge"),
    ]

    box_w, box_h = 2.1, 2.6
    centers = []
    for x0, color, title, body in stages:
        _box(ax, (x0, 1.5), box_w, box_h, "", color, alpha=0.92,
             rounding=0.06)
        ax.text(x0 + box_w / 2, 1.5 + box_h - 0.45, title,
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        ax.text(x0 + box_w / 2, 1.5 + box_h / 2 - 0.3, body,
                ha="center", va="center", color="white", fontsize=8.6)
        centers.append((x0 + box_w, 1.5 + box_h / 2))

    # arrows between stages
    for i in range(len(stages) - 1):
        x_from = centers[i]
        x_to = (stages[i + 1][0], 1.5 + box_h / 2)
        _arrow(ax, x_from, x_to, color=C_DARK, lw=1.6, style="-|>")

    # bottom caption
    ax.text(6.5, 0.55,
            "Equivariance is built in at stage 3: GNN message passing commutes "
            "with node permutation,\nso permuted networks produce permuted "
            "node embeddings and the SAME pooled graph embedding.",
            ha="center", va="center", fontsize=10, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    fig.suptitle("Figure 3. End-to-end GNN pipeline for processing neural-net "
                 "parameters", fontsize=12.8, color=C_DARK, y=1.0)
    _save(fig, "fig3_gnn_pipeline")


# ---------------------------------------------------------------------------
# Figure 4: Equivariant vs invariant representations
# ---------------------------------------------------------------------------
def fig4_equivariant_vs_invariant() -> None:
    """Contrast: invariant pooling collapses; equivariant embeddings permute."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    fig.subplots_adjust(wspace=0.22)

    rng = np.random.default_rng(7)
    base_emb = rng.normal(0, 1, size=(4, 6))     # 4 nodes, dim 6
    perm = np.array([2, 0, 3, 1])
    perm_emb = base_emb[perm]

    # ---- LEFT: invariant pooling ----
    ax = axes[0]
    ax.set_title("(a) Invariant pooling  (graph-level task)",
                 fontsize=12.5, color=C_DARK, fontweight="bold", pad=6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    for i in range(4):
        _box(ax, (0.5 + i * 1.0, 4.2), 0.8, 0.8, f"h{i+1}", C_BLUE,
             fontsize=10, alpha=0.9)
    ax.text(2.7, 5.2, "Original node embeddings", ha="center",
            fontsize=10, color=C_DARK)

    for i in range(4):
        _box(ax, (5.5 + i * 1.0, 4.2), 0.8, 0.8,
             f"h{perm[i]+1}", C_PURPLE, fontsize=10, alpha=0.9)
    ax.text(7.7, 5.2, "Permuted node embeddings", ha="center",
            fontsize=10, color=C_DARK)

    # both pool to same vector
    _arrow(ax, (2.7, 4.1), (4.0, 2.8), color=C_DARK, lw=1.3)
    _arrow(ax, (7.7, 4.1), (6.0, 2.8), color=C_DARK, lw=1.3)
    _box(ax, (4.1, 2.0), 1.8, 0.8, r"$z_G = \sum_v h_v$", C_GREEN,
         fontsize=11, alpha=0.95)
    ax.text(5.0, 1.4, "SAME vector for both", ha="center", fontsize=10,
            color=C_GREEN, fontweight="bold")

    ax.text(5.0, 0.5,
            r"Property:  $f(\pi \cdot G) = f(G)$"
            "\nUse for: predict generalisation, classify task, retrieve.",
            ha="center", va="center", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    # ---- RIGHT: equivariant ----
    ax = axes[1]
    ax.set_title("(b) Equivariant node embeddings  (node-level task)",
                 fontsize=12.5, color=C_DARK, fontweight="bold", pad=6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    # heatmaps
    ax.imshow(base_emb, extent=(0.5, 4.0, 3.0, 5.0), aspect="auto",
              cmap="viridis")
    ax.text(2.25, 5.25, "Z(G)  rows = nodes", ha="center", fontsize=10,
            color=C_DARK)
    for i in range(4):
        ax.text(0.25, 5.0 - (i + 0.5) * 0.5, f"h{i+1}", ha="right",
                va="center", fontsize=9, color=C_DARK)

    ax.imshow(perm_emb, extent=(6.0, 9.5, 3.0, 5.0), aspect="auto",
              cmap="viridis")
    ax.text(7.75, 5.25, r"$Z(\pi\cdot G) = \pi \cdot Z(G)$",
            ha="center", fontsize=10, color=C_DARK)
    for i in range(4):
        ax.text(5.75, 5.0 - (i + 0.5) * 0.5, f"h{perm[i]+1}",
                ha="right", va="center", fontsize=9, color=C_DARK)

    # arrow
    _arrow(ax, (4.1, 4.0), (5.95, 4.0), color=C_AMBER, lw=2.0,
           style="-|>")
    ax.text(5.0, 4.4, r"apply $\pi$", ha="center", fontsize=9.5,
            color=C_AMBER, fontweight="bold")

    ax.text(5.0, 1.4,
            r"Property:  $Z(\pi\cdot G) = \pi \cdot Z(G)$"
            "\nUse for: neuron alignment, model merging,"
            " architecture editing.",
            ha="center", va="center", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    fig.suptitle(
        "Figure 4. Invariance vs equivariance: same symmetry, different output",
        fontsize=12.8, color=C_DARK, y=1.02,
    )
    _save(fig, "fig4_equivariant_vs_invariant")


# ---------------------------------------------------------------------------
# Figure 5: Predicting generalization
# ---------------------------------------------------------------------------
def fig5_generalization_prediction() -> None:
    """Application: predict test accuracy from weights, no validation needed."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4),
                             gridspec_kw={"width_ratios": [1.0, 1.2]})
    fig.subplots_adjust(wspace=0.28)

    # ---- LEFT: schematic ----
    ax = axes[0]
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(a) Pipeline", fontsize=12.5, color=C_DARK,
                 fontweight="bold", pad=6)

    # mini neural graph icons (3 of them, stacked)
    for k, y0 in enumerate([4.4, 3.0, 1.6]):
        # tiny graph: 3 neurons + 1 out
        for i, x in enumerate([0.5, 0.5, 0.5]):
            _neuron(ax, (x, y0 + 0.4 - 0.4 * i), r=0.13,
                    fc=C_GRAY, ec=C_DARK)
        for i, x in enumerate([1.3, 1.3]):
            _neuron(ax, (x, y0 + 0.25 - 0.5 * i), r=0.13,
                    fc=C_PURPLE, ec=C_DARK)
        _neuron(ax, (2.0, y0), r=0.13, fc=C_AMBER, ec=C_DARK)
        for i in range(3):
            for j in range(2):
                _arrow(ax, (0.5, y0 + 0.4 - 0.4 * i),
                       (1.3, y0 + 0.25 - 0.5 * j),
                       color=C_GRAY, lw=0.6, style="-", alpha=0.6)
            _arrow(ax, (1.3, y0 + 0.25 - 0.5 * (i % 2)), (2.0, y0),
                   color=C_GRAY, lw=0.6, style="-", alpha=0.6)
        ax.text(2.4, y0, f"net #{k+1}", fontsize=9, color=C_DARK,
                va="center")

    # arrow into GNN
    _arrow(ax, (3.4, 3.0), (4.3, 3.0), color=C_DARK, lw=1.6, style="-|>")

    _box(ax, (4.3, 2.3), 1.5, 1.4, "GNN", C_GREEN, fontsize=13,
         fontweight="bold")
    ax.text(5.05, 2.0, r"equivariant", ha="center", fontsize=8.5,
            color=C_GREEN)

    _arrow(ax, (5.85, 3.0), (6.2, 3.0), color=C_DARK, lw=1.6,
           style="-|>")
    ax.text(6.5, 3.0, r"$\widehat{\mathrm{acc}}$", fontsize=14,
            color=C_DARK, va="center", fontweight="bold")

    ax.text(3.5, 0.6,
            "Input: trained weights only.   No validation forward passes "
            "needed.",
            ha="center", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    # ---- RIGHT: scatter ----
    ax = axes[1]
    ax.set_title("(b) Predicted vs true test accuracy on a model zoo",
                 fontsize=12.5, color=C_DARK, fontweight="bold", pad=6)

    rng = np.random.default_rng(0)
    n = 140
    true_acc = rng.uniform(0.55, 0.95, size=n)

    # equivariant GNN: tight
    pred_gnn = true_acc + rng.normal(0, 0.018, size=n)
    # naive flatten + MLP: scattered, biased
    pred_flat = 0.78 + 0.35 * (true_acc - 0.78) + rng.normal(0, 0.06, size=n)

    ax.scatter(true_acc, pred_flat, c=C_AMBER, s=28, alpha=0.55,
               edgecolor="white", linewidth=0.6,
               label="Flatten weights + MLP  (not equivariant)")
    ax.scatter(true_acc, pred_gnn, c=C_BLUE, s=30, alpha=0.85,
               edgecolor="white", linewidth=0.6,
               label="Neural graph + GNN  (equivariant)")

    lims = [0.5, 1.0]
    ax.plot(lims, lims, color=C_DARK, lw=1.2, linestyle="--",
            label="perfect prediction")

    # report R^2 (toy)
    def r2(y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

    r2_gnn = r2(true_acc, pred_gnn)
    r2_flat = r2(true_acc, pred_flat)
    ax.text(0.52, 0.96,
            f"$R^2$  GNN  = {r2_gnn:.3f}\n"
            f"$R^2$  flat = {r2_flat:.3f}",
            fontsize=10, color=C_DARK, va="top",
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("True test accuracy", fontsize=10.5, color=C_DARK)
    ax.set_ylabel("Predicted test accuracy", fontsize=10.5, color=C_DARK)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.suptitle(
        "Figure 5. Application: predicting generalisation from weights alone",
        fontsize=12.8, color=C_DARK, y=1.02,
    )
    _save(fig, "fig5_generalization_prediction")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating GNN-equivariant figures...")
    fig1_permutation_equivariance()
    fig2_neural_graph()
    fig3_gnn_pipeline()
    fig4_equivariant_vs_invariant()
    fig5_generalization_prediction()
    print("Done.")


if __name__ == "__main__":
    main()
