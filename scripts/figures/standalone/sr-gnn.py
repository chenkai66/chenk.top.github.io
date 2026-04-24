"""
Figure generation script for the SR-GNN standalone paper review.

Generates 5 self-contained figures used in both the EN and ZH versions of
the article. Each figure isolates exactly one teaching point so it can be
referenced independently in the prose.

Figures:
    fig1_session_graph         Session graph construction. Click sequence
                               A, B, C, B, D becomes a directed graph with
                               edge weights normalised by source out-degree;
                               the repeated visit to B turns the sequence
                               B->C / C->B into structural cycles a sequence
                               model would forget.
    fig2_architecture          End-to-end SR-GNN architecture. Session graph
                               -> Gated GNN propagation -> per-node
                               embeddings -> attention pooling against the
                               last click -> local + global fusion ->
                               softmax over item catalog.
    fig3_ggnn_update           One step of the gated GNN update. Shows how
                               the in/out adjacency rows produce the
                               aggregated message a_t, which then drives
                               GRU-style update / reset gates against the
                               previous node state h_{t-1}.
    fig4_attention_pooling     Local-vs-global session embedding. Per-item
                               attention weights alpha_i are produced from
                               the last click and the per-item embeddings;
                               local (last click) and global (alpha-weighted
                               sum) are linearly fused into the session
                               vector s.
    fig5_benchmark_perf        Benchmark performance on Yoochoose 1/64,
                               Yoochoose 1/4 and Diginetica. Grouped bars
                               vs POP / Item-KNN / FPMC / GRU4Rec / NARM /
                               STAMP for both Recall@20 and MRR@20.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (amber).

Usage:
    python3 scripts/figures/standalone/sr-gnn.py

Output:
    Writes the same PNGs into BOTH the EN and ZH asset folders so the markdown
    image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good
C_AMBER = "#f59e0b"    # warning / highlight
C_GRAY = "#94a3b8"
C_LIGHT = "#e2e8f0"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "en" / "standalone"
    / "session-based-recommendation-with-graph-neural-networks"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "standalone"
    / "session-based-recommendation-with-graph-neural-networks"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to BOTH EN and ZH asset folders."""
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
           style="-|>", connection="arc3,rad=0", mutation=12):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=mutation,
        color=color, lw=lw, connectionstyle=connection,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: Session graph construction from clicks
# ---------------------------------------------------------------------------
def fig1_session_graph() -> None:
    """Click sequence -> directed weighted graph; emphasise repeated visits."""
    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.18)

    # ---------- Left: click sequence as a timeline ----------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    ax1.set_title("(a) Click sequence  (what an RNN sees)",
                  fontsize=12, color=C_DARK, pad=10)

    seq = ["A", "B", "C", "B", "D"]
    seq_colors = [C_BLUE, C_PURPLE, C_GREEN, C_PURPLE, C_AMBER]
    box_w, box_h = 1.2, 1.2
    gap = 0.55
    start_x = 0.5
    y = 3.4
    for i, (s, c) in enumerate(zip(seq, seq_colors)):
        x = start_x + i * (box_w + gap)
        _box(ax1, (x, y), box_w, box_h, s, c, fontsize=20,
             fontweight="bold", rounding=0.18)
        ax1.text(x + box_w / 2, y - 0.45, f"$t_{{{i+1}}}$",
                 ha="center", fontsize=10, color=C_DARK)
        if i < len(seq) - 1:
            xn = start_x + (i + 1) * (box_w + gap)
            _arrow(ax1, (x + box_w + 0.05, y + box_h / 2),
                   (xn - 0.05, y + box_h / 2),
                   color=C_DARK, lw=1.4)

    ax1.text(5.0, 1.6,
             "RNN/GRU folds the sequence into a single hidden state.\n"
             "When the user returns from C back to B, the explicit\n"
             "transition  B -> C  is gone -- it has been overwritten.",
             ha="center", va="center", fontsize=10, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                       boxstyle="round,pad=0.5"))

    # ---------- Right: directed weighted graph ----------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("(b) Session graph  (what SR-GNN sees)",
                  fontsize=12, color=C_DARK, pad=10)
    ax2.axis("off")

    G = nx.MultiDiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    # raw transitions in the click stream:
    # A->B, B->C, C->B, B->D
    raw_edges = [("A", "B"), ("B", "C"), ("C", "B"), ("B", "D")]
    G.add_edges_from(raw_edges)

    # out-degree-normalised weights
    out_count = {"A": 1, "B": 2, "C": 1, "D": 0}
    edge_weights = {
        ("A", "B"): 1 / out_count["A"],
        ("B", "C"): 1 / out_count["B"],
        ("C", "B"): 1 / out_count["C"],
        ("B", "D"): 1 / out_count["B"],
    }

    pos = {"A": (-1.4, 0.8), "B": (0.0, 0.0),
           "C": (1.4, 0.8), "D": (1.4, -0.9)}

    node_color = {"A": C_BLUE, "B": C_PURPLE, "C": C_GREEN, "D": C_AMBER}
    for n, (x, y) in pos.items():
        ax2.scatter(x, y, s=2200, color=node_color[n],
                    edgecolor=C_DARK, linewidth=1.6, zorder=3)
        ax2.text(x, y, n, ha="center", va="center",
                 fontsize=18, fontweight="bold", color="white",
                 zorder=4)

    def _edge(src, dst, rad=0.0, w=1.0, color=C_DARK, label=None,
              label_offset=(0.0, 0.0)):
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=18,
            color=color, lw=1.6 + w * 0.6,
            shrinkA=22, shrinkB=22,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
        ax2.add_patch(arr)
        if label:
            mx = (x1 + x2) / 2 + label_offset[0]
            my = (y1 + y2) / 2 + label_offset[1]
            ax2.text(mx, my, label, ha="center", va="center",
                     fontsize=9.5, color=color, fontweight="bold",
                     bbox=dict(facecolor="white", edgecolor=C_LIGHT,
                               boxstyle="round,pad=0.18"))

    _edge("A", "B", rad=0.0, w=edge_weights[("A", "B")],
          color=C_BLUE, label=f"w={edge_weights[('A','B')]:.2f}",
          label_offset=(-0.08, 0.18))
    _edge("B", "C", rad=0.18, w=edge_weights[("B", "C")],
          color=C_PURPLE, label=f"w={edge_weights[('B','C')]:.2f}",
          label_offset=(0.0, 0.32))
    _edge("C", "B", rad=0.18, w=edge_weights[("C", "B")],
          color=C_GREEN, label=f"w={edge_weights[('C','B')]:.2f}",
          label_offset=(0.0, -0.32))
    _edge("B", "D", rad=0.0, w=edge_weights[("B", "D")],
          color=C_AMBER, label=f"w={edge_weights[('B','D')]:.2f}",
          label_offset=(0.25, 0.0))

    ax2.set_xlim(-2.4, 2.4)
    ax2.set_ylim(-1.8, 1.7)

    ax2.text(0.0, -1.55,
             r"$w_{u \to v} = \dfrac{\#(u \to v)}{\mathrm{outdeg}(u)}$"
             "    -- B has out-degree 2, so each outgoing edge gets weight 1/2.",
             ha="center", fontsize=10, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                       boxstyle="round,pad=0.35"))

    fig.suptitle("Figure 1. From click stream to session graph: "
                 "every transition is preserved, even after a revisit",
                 fontsize=12.8, color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig1_session_graph")


# ---------------------------------------------------------------------------
# Figure 2: SR-GNN architecture (GGNN + attention pooling)
# ---------------------------------------------------------------------------
def fig2_architecture() -> None:
    """Session graph -> GGNN -> attention pooling -> softmax."""
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(6.5, 6.0, "SR-GNN end-to-end pipeline",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=C_DARK)

    # ---- 1. session graph ----
    g_x, g_y = 0.4, 1.6
    _box(ax, (g_x, g_y), 2.4, 3.0, "", C_BG, ec=C_DARK,
         text_color=C_DARK, alpha=0.6, rounding=0.06)
    ax.text(g_x + 1.2, g_y + 3.15, "Session graph $G_s$",
            ha="center", fontsize=10.5,
            fontweight="bold", color=C_DARK)
    # draw 4 small nodes
    mini_pos = {"A": (g_x + 0.6, g_y + 2.4),
                "B": (g_x + 1.6, g_y + 1.7),
                "C": (g_x + 0.6, g_y + 1.0),
                "D": (g_x + 1.85, g_y + 0.55)}
    mini_color = {"A": C_BLUE, "B": C_PURPLE, "C": C_GREEN, "D": C_AMBER}
    for n, (x, y) in mini_pos.items():
        ax.scatter(x, y, s=420, color=mini_color[n],
                   edgecolor=C_DARK, lw=1.0, zorder=3)
        ax.text(x, y, n, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=4)
    for src, dst, rad in [("A", "B", 0.0), ("B", "C", 0.2),
                          ("C", "B", 0.2), ("B", "D", 0.0)]:
        x1, y1 = mini_pos[src]
        x2, y2 = mini_pos[dst]
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=10, color=C_DARK, lw=1.0,
            shrinkA=10, shrinkB=10,
            connectionstyle=f"arc3,rad={rad}", zorder=2,
        ))

    # arrow -> GGNN
    _arrow(ax, (g_x + 2.4, g_y + 1.5), (3.4, g_y + 1.5),
           color=C_DARK, lw=1.6)

    # ---- 2. Gated GNN propagation ----
    gnn_x, gnn_y, gnn_w, gnn_h = 3.4, 1.6, 2.6, 3.0
    _box(ax, (gnn_x, gnn_y), gnn_w, gnn_h, "", C_BLUE,
         alpha=0.18, ec=C_BLUE, rounding=0.06)
    ax.text(gnn_x + gnn_w / 2, gnn_y + gnn_h + 0.15,
            "Gated GNN (T steps)",
            ha="center", fontsize=10.5, fontweight="bold", color=C_BLUE)

    # show update equations
    ax.text(gnn_x + gnn_w / 2, gnn_y + 2.3,
            r"$a_t = A_s [h_1 \dots h_n]^\top W_a$",
            ha="center", fontsize=10, color=C_DARK)
    ax.text(gnn_x + gnn_w / 2, gnn_y + 1.8,
            r"$z_t = \sigma(W_z a_t + U_z h_{t-1})$",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(gnn_x + gnn_w / 2, gnn_y + 1.35,
            r"$r_t = \sigma(W_r a_t + U_r h_{t-1})$",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(gnn_x + gnn_w / 2, gnn_y + 0.85,
            r"$\tilde h_t = \tanh(W a_t + U(r_t \odot h_{t-1}))$",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(gnn_x + gnn_w / 2, gnn_y + 0.35,
            r"$h_t = (1{-}z_t)\, h_{t-1} + z_t\, \tilde h_t$",
            ha="center", fontsize=9.5, color=C_DARK)

    # arrow -> embeddings
    _arrow(ax, (gnn_x + gnn_w, gnn_y + 1.5), (6.5, gnn_y + 1.5),
           color=C_DARK, lw=1.6)

    # ---- 3. per-node embeddings ----
    emb_x = 6.5
    emb_y = gnn_y + 0.2
    ax.text(emb_x + 0.7, gnn_y + gnn_h + 0.15,
            "Item embeddings", ha="center",
            fontsize=10.5, fontweight="bold", color=C_PURPLE)
    for i, (lab, c) in enumerate(zip(["A", "B", "C", "D"],
                                      [C_BLUE, C_PURPLE, C_GREEN, C_AMBER])):
        _box(ax, (emb_x, emb_y + i * 0.7), 1.4, 0.5,
             rf"$h_{{{lab}}} \in \mathbf{{R}}^d$", c, fontsize=10,
             fontweight="bold", rounding=0.18)

    # arrow -> attention
    _arrow(ax, (emb_x + 1.4, gnn_y + 1.5), (8.6, gnn_y + 1.5),
           color=C_DARK, lw=1.6)

    # ---- 4. local + global pooling ----
    pool_x, pool_y, pool_w, pool_h = 8.6, 1.6, 2.4, 3.0
    _box(ax, (pool_x, pool_y), pool_w, pool_h, "", C_GREEN,
         alpha=0.18, ec=C_GREEN, rounding=0.06)
    ax.text(pool_x + pool_w / 2, pool_y + pool_h + 0.15,
            "Local + global pooling",
            ha="center", fontsize=10.5, fontweight="bold", color=C_GREEN)

    ax.text(pool_x + pool_w / 2, pool_y + 2.4,
            r"$s_l = h_n$  (last click)",
            ha="center", fontsize=10, color=C_DARK)
    ax.text(pool_x + pool_w / 2, pool_y + 1.85,
            r"$\alpha_i = q^\top \sigma(W_1 h_n + W_2 h_i)$",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(pool_x + pool_w / 2, pool_y + 1.35,
            r"$s_g = \sum_i \alpha_i\, h_i$",
            ha="center", fontsize=10, color=C_DARK)
    ax.text(pool_x + pool_w / 2, pool_y + 0.55,
            r"$s_h = W_3\, [\,s_l;\, s_g\,]$",
            ha="center", fontsize=10.5, color=C_DARK,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=C_GREEN,
                      boxstyle="round,pad=0.2"))

    # arrow -> softmax
    _arrow(ax, (pool_x + pool_w, gnn_y + 1.5), (11.3, gnn_y + 1.5),
           color=C_DARK, lw=1.6)

    # ---- 5. softmax over catalog ----
    sf_x, sf_y, sf_w, sf_h = 11.3, 1.6, 1.4, 3.0
    _box(ax, (sf_x, sf_y), sf_w, sf_h, "", C_AMBER,
         alpha=0.18, ec=C_AMBER, rounding=0.06)
    ax.text(sf_x + sf_w / 2, sf_y + sf_h + 0.15,
            "Softmax", ha="center", fontsize=10.5,
            fontweight="bold", color=C_AMBER)
    ax.text(sf_x + sf_w / 2, sf_y + 2.2,
            r"$\hat z_i = s_h^\top v_i$",
            ha="center", fontsize=10, color=C_DARK)
    ax.text(sf_x + sf_w / 2, sf_y + 1.55,
            r"$\hat y = \mathrm{softmax}(\hat z)$",
            ha="center", fontsize=10, color=C_DARK)
    ax.text(sf_x + sf_w / 2, sf_y + 0.6,
            "top-K\nrecommendation",
            ha="center", fontsize=9.5, color=C_AMBER,
            fontweight="bold")

    # caption strip
    ax.text(6.5, 0.9,
            "Loss: cross-entropy of $\\hat y$ against the one-hot next click."
            "   Trainable: item table $V$, GGNN ($W_*$, $U_*$), "
            "pooling ($q$, $W_1$, $W_2$, $W_3$).",
            ha="center", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.35"))

    fig.suptitle("Figure 2. SR-GNN architecture: graph in, ranked items out",
                 fontsize=12.8, color=C_DARK, y=0.02)
    _save(fig, "fig2_architecture")


# ---------------------------------------------------------------------------
# Figure 3: Gated GNN single update step
# ---------------------------------------------------------------------------
def fig3_ggnn_update() -> None:
    """Anatomy of one GGNN message-passing step."""
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.18)

    # ---------- (a) Adjacency rows ----------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("(a) Adjacency rows  $A_s = [A^{(in)} \\,|\\, A^{(out)}]$",
                  fontsize=11.5, color=C_DARK, pad=10)

    # Adjacency matrices for nodes (A, B, C, D) on click A->B->C->B->D
    # In-degree normalised
    A_in = np.array([
        [0.0, 0.0, 0.0, 0.0],   # A
        [1.0, 0.0, 1.0, 0.0],   # B  (incoming from A and from C)
        [0.0, 0.5, 0.0, 0.0],   # C  (incoming from B)
        [0.0, 0.5, 0.0, 0.0],   # D  (incoming from B)
    ])
    A_out = np.array([
        [0.0, 1.0, 0.0, 0.0],   # A -> B
        [0.0, 0.0, 0.5, 0.5],   # B -> C, B -> D
        [0.0, 1.0, 0.0, 0.0],   # C -> B
        [0.0, 0.0, 0.0, 0.0],   # D
    ])
    M = np.concatenate([A_in, A_out], axis=1)

    im = ax1.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax1.text(j, i, f"{v:.1f}", ha="center", va="center",
                     fontsize=10,
                     color="white" if v > 0.45 else C_DARK)
    ax1.set_xticks(range(8))
    ax1.set_xticklabels(["A", "B", "C", "D", "A", "B", "C", "D"],
                        fontsize=10)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(["A", "B", "C", "D"], fontsize=10)
    ax1.set_xlabel("incoming  (left 4)        outgoing  (right 4)",
                   fontsize=10, color=C_DARK)
    ax1.axvline(3.5, color=C_DARK, lw=1.4)

    ax1.text(0.5, -1.6,
             "Row of node $i$ tells the GGNN which neighbours feed it,\n"
             "and which neighbours it feeds in turn.",
             transform=ax1.transData, fontsize=9.5, color=C_DARK,
             ha="left")

    # ---------- (b) GRU-style cell ----------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title("(b) GRU-style update on the aggregated message",
                  fontsize=11.5, color=C_DARK, pad=10)

    # h_{t-1}
    _box(ax2, (0.3, 4.4), 1.6, 0.8, r"$h_{t-1}$", C_GRAY,
         fontsize=12, fontweight="bold", text_color="white", rounding=0.18)
    # a_t
    _box(ax2, (0.3, 2.4), 1.6, 0.8, r"$a_t$ (msg)", C_BLUE,
         fontsize=11, fontweight="bold", rounding=0.18)
    ax2.text(1.1, 1.7, r"$a_t = A_s\,[h_1\dots h_n]^\top W_a + b$",
             ha="center", fontsize=9, color=C_BLUE)

    # Reset gate
    _box(ax2, (3.0, 4.4), 1.5, 0.8, r"$r_t$", C_AMBER,
         fontsize=12, fontweight="bold", rounding=0.18)
    ax2.text(3.75, 5.45, r"$r_t = \sigma(W_r a_t + U_r h_{t-1})$",
             ha="center", fontsize=9, color=C_AMBER)
    # Update gate
    _box(ax2, (3.0, 2.4), 1.5, 0.8, r"$z_t$", C_PURPLE,
         fontsize=12, fontweight="bold", rounding=0.18)
    ax2.text(3.75, 1.7, r"$z_t = \sigma(W_z a_t + U_z h_{t-1})$",
             ha="center", fontsize=9, color=C_PURPLE)

    # arrows from inputs to gates
    for tx, ty in [(3.0, 4.8), (3.0, 2.8)]:
        _arrow(ax2, (1.9, 4.8), (tx, ty), color=C_GRAY, lw=1.0)
        _arrow(ax2, (1.9, 2.8), (tx, ty), color=C_BLUE, lw=1.0)

    # candidate
    _box(ax2, (5.3, 3.4), 1.7, 0.9, r"$\tilde h_t$", C_GREEN,
         fontsize=12, fontweight="bold", rounding=0.18)
    ax2.text(6.15, 4.55,
             r"$\tilde h_t = \tanh(W a_t + U(r_t \odot h_{t-1}))$",
             ha="center", fontsize=9, color=C_GREEN)

    _arrow(ax2, (4.5, 4.8), (5.3, 4.0), color=C_AMBER, lw=1.2)
    _arrow(ax2, (4.5, 2.8), (5.3, 3.7), color=C_PURPLE, lw=1.2,
           connection="arc3,rad=0.18")
    _arrow(ax2, (1.9, 4.8), (5.3, 3.95), color=C_GRAY, lw=1.0,
           connection="arc3,rad=-0.2")
    _arrow(ax2, (1.9, 2.8), (5.3, 3.6), color=C_BLUE, lw=1.0,
           connection="arc3,rad=0.2")

    # output h_t
    _box(ax2, (8.0, 3.4), 1.6, 0.9, r"$h_t$", C_DARK,
         fontsize=13, fontweight="bold", rounding=0.18)
    _arrow(ax2, (7.0, 3.85), (8.0, 3.85), color=C_DARK, lw=1.6)
    ax2.text(8.8, 2.85,
             r"$h_t = (1{-}z_t)\, h_{t-1} + z_t\, \tilde h_t$",
             ha="center", fontsize=9.5, color=C_DARK)

    ax2.text(5, 0.7,
             "Reset gate $r_t$ filters how much past state contributes "
             "to the candidate;\n"
             "update gate $z_t$ blends old state with the new candidate.",
             ha="center", fontsize=9.5, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                       boxstyle="round,pad=0.35"))

    fig.suptitle("Figure 3. One step of the gated GNN: message $a_t$ then "
                 "GRU-style state update",
                 fontsize=12.8, color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig3_ggnn_update")


# ---------------------------------------------------------------------------
# Figure 4: Attention pooling -> session embedding
# ---------------------------------------------------------------------------
def fig4_attention_pooling() -> None:
    """Per-item attention from last click; local + global fusion."""
    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.22)

    # ---------- (a) Attention weights ----------
    ax1 = fig.add_subplot(gs[0, 0])
    items = ["A", "B", "C", "B", "D"]
    embeds = np.array([
        [0.20, 0.80],
        [0.55, 0.65],
        [0.85, 0.40],
        [0.55, 0.65],
        [0.80, 0.20],
    ])
    h_n = embeds[-1]
    raw = -np.linalg.norm(embeds - h_n, axis=1) * 1.4 + np.array(
        [0.10, 0.55, 0.30, 0.55, 0.85])
    alpha = np.exp(raw) / np.exp(raw).sum()

    bars = ax1.bar(items, alpha,
                   color=[C_BLUE, C_PURPLE, C_GREEN, C_PURPLE, C_AMBER],
                   edgecolor=C_DARK, linewidth=0.9)
    for bar, v in zip(bars, alpha):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                 f"{v:.2f}", ha="center", fontsize=9.5,
                 color=C_DARK, fontweight="bold")
    ax1.set_ylim(0, max(alpha) * 1.35)
    ax1.set_ylabel(r"attention weight $\alpha_i$", fontsize=10.5)
    ax1.set_title("(a) Attention anchored on the last click  $h_n = h_D$",
                  fontsize=11.5, color=C_DARK, pad=10)
    ax1.text(0.0, -0.18,
             r"$\alpha_i = q^\top\, \sigma(W_1 h_n + W_2 h_i + c)$",
             transform=ax1.transAxes, fontsize=10.5, color=C_DARK)
    ax1.tick_params(labelsize=10)

    # ---------- (b) local + global fusion ----------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title("(b) Local intent + global context  ->  session vector",
                  fontsize=11.5, color=C_DARK, pad=10)

    # Local
    _box(ax2, (0.3, 3.8), 3.4, 1.2, r"$s_l = h_n$" + "\n(last click)",
         C_AMBER, fontsize=11.5, fontweight="bold",
         text_color="white", rounding=0.15)
    ax2.text(2.0, 3.5, "short-term intent", ha="center", fontsize=9.5,
             color=C_AMBER, fontstyle="italic")

    # Global
    _box(ax2, (0.3, 1.0), 3.4, 1.2,
         r"$s_g = \sum_i \alpha_i h_i$" + "\n(attention sum)",
         C_BLUE, fontsize=11.5, fontweight="bold",
         text_color="white", rounding=0.15)
    ax2.text(2.0, 0.7, "long-term context", ha="center", fontsize=9.5,
             color=C_BLUE, fontstyle="italic")

    # Concat node
    _box(ax2, (4.6, 2.6), 1.6, 1.0, r"$[\,s_l;\, s_g\,]$",
         C_GRAY, fontsize=12, fontweight="bold", rounding=0.18)
    _arrow(ax2, (3.7, 4.4), (4.6, 3.4), color=C_AMBER, lw=1.5)
    _arrow(ax2, (3.7, 1.6), (4.6, 2.8), color=C_BLUE, lw=1.5)

    # Linear projection
    _box(ax2, (7.0, 2.6), 2.6, 1.0,
         r"$s_h = W_3\,[\,s_l;\, s_g\,]$",
         C_GREEN, fontsize=11.5, fontweight="bold",
         text_color="white", rounding=0.18)
    _arrow(ax2, (6.2, 3.1), (7.0, 3.1), color=C_DARK, lw=1.6)

    ax2.text(5.0, 5.55,
             "Local catches the most recent click; global re-weights\n"
             "all visited items by relevance to that last click.",
             ha="center", fontsize=9.5, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                       boxstyle="round,pad=0.35"))

    fig.suptitle("Figure 4. Building the session vector $s_h$ "
                 "from per-item embeddings",
                 fontsize=12.8, color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig4_attention_pooling")


# ---------------------------------------------------------------------------
# Figure 5: Benchmark performance
# ---------------------------------------------------------------------------
def fig5_benchmark_perf() -> None:
    """Recall@20 / MRR@20 on Yoochoose 1/64, 1/4 and Diginetica."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Numbers reported in the original SR-GNN paper (Wu et al., AAAI 2019),
    # Tables 2 and 3. POP / Item-KNN / FPMC / GRU4Rec / NARM / STAMP / SR-GNN
    methods = ["POP", "Item-KNN", "FPMC", "GRU4Rec",
               "NARM", "STAMP", "SR-GNN"]
    colors = [C_GRAY, C_GRAY, C_GRAY, C_GRAY,
              C_PURPLE, C_AMBER, C_BLUE]

    # Recall@20
    yoochoose64_r20 = [6.71, 51.60, 45.62, 60.64,
                      68.32, 68.74, 70.57]
    yoochoose4_r20 = [1.33, 52.31, 51.86, 59.53,
                     69.73, 70.44, 71.36]
    digi_r20 = [0.89, 35.75, 26.53, 29.45,
                49.70, 45.64, 50.73]

    # MRR@20
    yoochoose64_mrr = [1.65, 21.81, 15.01, 22.89,
                      28.63, 29.67, 30.94]
    yoochoose4_mrr = [0.30, 21.70, 17.50, 22.60,
                     29.23, 30.00, 31.89]
    digi_mrr = [0.20, 11.57, 6.95, 8.33,
                16.17, 14.32, 17.59]

    datasets = ["Yoochoose 1/64", "Yoochoose 1/4", "Diginetica"]

    # Group bars by method, x = dataset
    x = np.arange(len(datasets))
    n = len(methods)
    width = 0.115

    def _grouped(ax, vals_per_method, title, ylabel):
        for i, (m, c, vals) in enumerate(zip(methods, colors, vals_per_method)):
            offset = (i - (n - 1) / 2) * width
            edge = C_DARK if m == "SR-GNN" else "none"
            lw = 1.0 if m == "SR-GNN" else 0.0
            bars = ax.bar(x + offset, vals, width=width, label=m,
                          color=c, edgecolor=edge, linewidth=lw,
                          alpha=0.95 if m == "SR-GNN" else 0.85)
            if m == "SR-GNN":
                for xi, v in zip(x, vals):
                    ax.text(xi + offset, v + 0.6, f"{v:.1f}",
                            ha="center", fontsize=8.2,
                            color=C_BLUE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(title, fontsize=11.5, color=C_DARK, pad=10)
        ax.tick_params(axis="y", labelsize=9)

    vals_recall = [
        [yoochoose64_r20[i], yoochoose4_r20[i], digi_r20[i]]
        for i in range(n)
    ]
    vals_mrr = [
        [yoochoose64_mrr[i], yoochoose4_mrr[i], digi_mrr[i]]
        for i in range(n)
    ]

    _grouped(axes[0], vals_recall, "Recall@20  (higher is better)",
             "Recall@20 (%)")
    axes[0].set_ylim(0, 80)
    _grouped(axes[1], vals_mrr, "MRR@20  (higher is better)",
             "MRR@20 (%)")
    axes[1].set_ylim(0, 38)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.04),
               ncol=len(methods), fontsize=9.5, frameon=True)

    fig.suptitle("Figure 5. SR-GNN vs prior session-based baselines on "
                 "Yoochoose and Diginetica",
                 fontsize=12.8, color=C_DARK, y=-0.02)
    fig.tight_layout()
    _save(fig, "fig5_benchmark_perf")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating SR-GNN figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")
    fig1_session_graph()
    fig2_architecture()
    fig3_ggnn_update()
    fig4_attention_pooling()
    fig5_benchmark_perf()
    print("Done.")


if __name__ == "__main__":
    main()
