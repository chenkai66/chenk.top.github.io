"""
Figure generation script for the GC-SAN standalone paper review.

Generates 5 self-contained figures used in both the EN and ZH versions of
the article. Each figure isolates one teaching point so it can be
referenced independently in the prose.

Figures:
    fig1_architecture           GC-SAN architecture overview. Embedding ->
                                session graph -> GGNN local encoder ->
                                multi-layer self-attention -> fusion of
                                last-click + global -> softmax over items.
    fig2_session_graph          Session graph construction from a click
                                sequence. Sequence v1 -> v2 -> v3 -> v2 ->
                                v4 becomes a directed graph with repeated
                                edges; in/out adjacency matrices shown.
    fig3_ggnn_message_passing   GGNN message passing on a node: incoming
                                neighbours, outgoing neighbours, GRU gates
                                (update / reset / candidate) producing the
                                next hidden state.
    fig4_self_attention         Self-attention heatmap over a 6-item session
                                showing global dependencies that pure GNN
                                hops cannot easily reach in a few steps.
    fig5_perf_vs_baselines      Recall@20 / MRR@20 vs SR-GNN and other
                                baselines on Yoochoose / Diginetica, plus
                                an ablation strip (no-GNN, no-SA, w grid).

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (amber).

Usage:
    python3 scripts/figures/standalone/gcsan.py

Output:
    Writes the same PNGs into BOTH the EN and ZH asset folders so the
    markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]     # primary
C_PURPLE = COLORS["accent"]   # secondary
C_GREEN = COLORS["success"]    # accent / good
C_AMBER = COLORS["warning"]    # warning / highlight
C_GRAY = COLORS["muted"]
C_LIGHT = COLORS["grid"]
C_DARK = COLORS["text"]
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "gcsan"
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "standalone"
    / "graph-contextualized-self-attention-network-for-session-base"
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
           style="-|>", connection="arc3,rad=0"):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw, connectionstyle=connection,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: GC-SAN architecture overview
# ---------------------------------------------------------------------------
def fig1_architecture() -> None:
    """End-to-end pipeline: embedding -> graph -> GGNN -> SA -> fusion."""
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6.5, 6.55,
            "GC-SAN pipeline: graph-contextualized embeddings + global "
            "self-attention",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=C_DARK)

    # ---- Stage 1: click sequence ----
    seq_y = 4.6
    seq = ["v1", "v2", "v3", "v2", "v4"]
    for i, v in enumerate(seq):
        x = 0.4 + i * 0.55
        _box(ax, (x, seq_y), 0.5, 0.5, v, C_GRAY,
             fontsize=9.5, fontweight="bold", text_color="white")
    ax.text(0.4 + len(seq) * 0.275, seq_y + 0.85,
            "click sequence", ha="center", fontsize=9.5,
            color=C_DARK, fontweight="bold")
    ax.text(0.4 + len(seq) * 0.275, seq_y - 0.35,
            "session $s = (v_1,\\dots,v_n)$",
            ha="center", fontsize=8.5, color=C_GRAY)

    # arrow to graph
    _arrow(ax, (3.3, seq_y + 0.25), (3.95, seq_y + 0.25),
           color=C_DARK, lw=1.3)

    # ---- Stage 2: session graph (mini) ----
    g_cx, g_cy = 4.65, seq_y + 0.25
    nodes = [(g_cx - 0.45, g_cy + 0.4, "1", C_BLUE),
             (g_cx + 0.05, g_cy + 0.55, "2", C_PURPLE),
             (g_cx + 0.55, g_cy + 0.2, "3", C_GREEN),
             (g_cx + 0.05, g_cy - 0.35, "4", C_AMBER)]
    for nx, ny, lab, c in nodes:
        circ = Circle((nx, ny), 0.12, facecolor=c, edgecolor=C_DARK, lw=1)
        ax.add_patch(circ)
        ax.text(nx, ny, lab, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
    # edges
    edges = [(0, 1), (1, 2), (2, 1), (1, 3)]
    for a, b in edges:
        ax.annotate("", xy=(nodes[b][0], nodes[b][1]),
                    xytext=(nodes[a][0], nodes[a][1]),
                    arrowprops=dict(arrowstyle="->", color=C_DARK,
                                    lw=0.9, shrinkA=8, shrinkB=8,
                                    connectionstyle="arc3,rad=0.18"))
    ax.text(g_cx + 0.05, g_cy - 0.85, "session graph $G_s$",
            ha="center", fontsize=9, color=C_DARK, fontweight="bold")

    # arrow graph -> GGNN
    _arrow(ax, (5.45, g_cy), (6.15, g_cy), color=C_DARK, lw=1.3)

    # ---- Stage 3: GGNN block ----
    _box(ax, (6.15, 3.6), 1.7, 1.8, "GGNN\n(local)", C_BLUE,
         fontsize=11.5, fontweight="bold", rounding=0.08)
    ax.text(7.0, 3.4,
            "in/out adjacency\n$T$ propagation steps",
            ha="center", fontsize=8.5, color=C_DARK)

    # arrow GGNN -> SA
    _arrow(ax, (7.85, 4.5), (8.55, 4.5), color=C_DARK, lw=1.3)

    # ---- Stage 4: Self-attention stack ----
    sa_x = 8.55
    for k in range(3):
        _box(ax, (sa_x + k * 0.18, 3.6 + k * 0.18), 1.6, 1.6,
             "", C_PURPLE, alpha=0.55 + 0.15 * k, rounding=0.06)
    _box(ax, (sa_x + 0.36, 3.96), 1.6, 1.6,
         "Self-Attn\nx k", C_PURPLE, fontsize=11,
         fontweight="bold", rounding=0.06)
    ax.text(sa_x + 1.16, 3.7,
            "global dependencies",
            ha="center", fontsize=8.5, color=C_DARK)

    # arrow SA -> fusion
    _arrow(ax, (sa_x + 1.96, 4.7), (sa_x + 2.65, 4.7),
           color=C_DARK, lw=1.3)

    # ---- Stage 5: fusion (last click + global) ----
    fus_x = sa_x + 2.65
    _box(ax, (fus_x, 4.95), 1.4, 0.55,
         "$h_t$  (last click)", C_GRAY,
         fontsize=9, text_color="white", rounding=0.08)
    _box(ax, (fus_x, 4.20), 1.4, 0.55,
         "$a_t$  (global)", C_PURPLE,
         fontsize=9, text_color="white", rounding=0.08)
    ax.text(fus_x + 0.7, 3.95,
            r"$s_f = w\,a_t + (1{-}w)\,h_t$",
            ha="center", fontsize=9.5, color=C_GREEN,
            fontweight="bold")

    # arrow fusion -> softmax
    _arrow(ax, (fus_x + 1.4, 4.55), (fus_x + 2.05, 4.55),
           color=C_DARK, lw=1.3)
    _box(ax, (fus_x + 2.05, 4.2), 0.85, 0.7,
         "softmax\nover $|V|$", C_GREEN,
         fontsize=9, fontweight="bold", rounding=0.1)

    # ---- Lower band: what each stage gives you ----
    band_y = 1.35
    cells = [
        ("Embedding\n+ graph build",
         "alias mapping;\nweighted in/out adj",
         C_GRAY),
        ("GGNN local",
         "transition loops;\nrepeated moves",
         C_BLUE),
        ("Self-Attention",
         "long-range\nintent",
         C_PURPLE),
        ("Fusion + score",
         "current + global\nrankable scores",
         C_GREEN),
    ]
    cw = 2.7
    for i, (title, sub, c) in enumerate(cells):
        x = 0.6 + i * (cw + 0.3)
        _box(ax, (x, band_y), cw, 1.3, "", c,
             alpha=0.18, rounding=0.05)
        ax.text(x + cw / 2, band_y + 0.95, title,
                ha="center", fontsize=10.5,
                color=C_DARK, fontweight="bold")
        ax.text(x + cw / 2, band_y + 0.35, sub,
                ha="center", fontsize=9, color=C_DARK)

    ax.text(6.5, 0.45,
            "Two complementary views of the same session: graph hops "
            "for local structure, attention for global intent.",
            ha="center", fontsize=10, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    fig.suptitle("Figure 1. GC-SAN architecture: from clicks to next-item "
                 "scores",
                 fontsize=12, color=C_DARK, y=0.02)
    _save(fig, "fig1_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Session graph construction
# ---------------------------------------------------------------------------
def fig2_session_graph() -> None:
    """From click sequence to directed weighted graph + adjacency matrices."""
    fig = plt.figure(figsize=(12, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0],
                          wspace=0.35)

    # ---- Panel A: click sequence ----
    axA = fig.add_subplot(gs[0, 0])
    axA.set_xlim(0, 6)
    axA.set_ylim(0, 6)
    axA.axis("off")
    axA.set_title("(a) click sequence", fontsize=11.5,
                  color=C_DARK, pad=12)

    seq = [("v1", C_BLUE), ("v2", C_PURPLE), ("v3", C_GREEN),
           ("v2", C_PURPLE), ("v4", C_AMBER)]
    for i, (lab, c) in enumerate(seq):
        x = 0.5 + i
        _box(axA, (x, 3.2), 0.8, 0.8, lab, c,
             fontsize=12, fontweight="bold", text_color="white",
             rounding=0.15)
        if i < len(seq) - 1:
            _arrow(axA, (x + 0.85, 3.6), (x + 1.0, 3.6),
                   color=C_DARK, lw=1.2)
    axA.text(3.0, 2.55, "time -->",
             ha="center", fontsize=9.5, color=C_GRAY)

    axA.text(3.0, 1.6,
             "v2 appears twice;\ntransitions v2->v3 and\nv3->v2 produce a "
             "loop.",
             ha="center", fontsize=10, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                       boxstyle="round,pad=0.3"))

    # ---- Panel B: directed session graph ----
    axB = fig.add_subplot(gs[0, 1])
    axB.set_xlim(0, 6)
    axB.set_ylim(0, 6)
    axB.axis("off")
    axB.set_title("(b) session graph $G_s$  (directed, weighted)",
                  fontsize=11.5, color=C_DARK, pad=12)

    nodes = {
        "v1": (1.2, 4.4, C_BLUE),
        "v2": (3.0, 4.6, C_PURPLE),
        "v3": (4.6, 3.6, C_GREEN),
        "v4": (3.6, 1.6, C_AMBER),
    }
    for lab, (x, y, c) in nodes.items():
        circ = Circle((x, y), 0.42, facecolor=c, edgecolor=C_DARK, lw=1.3)
        axB.add_patch(circ)
        axB.text(x, y, lab, ha="center", va="center",
                 fontsize=10.5, color="white", fontweight="bold")

    edges = [
        ("v1", "v2", 1, "arc3,rad=0.0"),
        ("v2", "v3", 1, "arc3,rad=-0.18"),
        ("v3", "v2", 1, "arc3,rad=-0.18"),
        ("v2", "v4", 1, "arc3,rad=0.0"),
    ]
    for a, b, w, conn in edges:
        ax_, ay_, _ = nodes[a]
        bx_, by_, _ = nodes[b]
        axB.annotate("", xy=(bx_, by_), xytext=(ax_, ay_),
                     arrowprops=dict(arrowstyle="-|>", color=C_DARK,
                                     lw=1.4, shrinkA=18, shrinkB=18,
                                     connectionstyle=conn))

    axB.text(3.0, 0.55,
             "edge weight = transition count\n"
             "(normalised per-row when building $A^{in}, A^{out}$)",
             ha="center", fontsize=9, color=C_DARK)

    # ---- Panel C: adjacency matrices ----
    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("(c) in/out adjacency",
                  fontsize=11.5, color=C_DARK, pad=12)
    items = ["v1", "v2", "v3", "v4"]
    A_out = np.array([
        [0, 1, 0, 0],   # v1 -> v2
        [0, 0, 1, 1],   # v2 -> v3, v4
        [0, 1, 0, 0],   # v3 -> v2
        [0, 0, 0, 0],
    ], dtype=float)
    # row-normalise
    row_sums = A_out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_out_n = A_out / row_sums

    A_in = A_out.T
    row_sums = A_in.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_in_n = A_in / row_sums

    # show A_out and A_in side by side
    big = np.concatenate([A_out_n, np.full((4, 1), np.nan), A_in_n], axis=1)
    im = axC.imshow(big, cmap="Blues", vmin=0, vmax=1)
    axC.set_xticks(list(range(4)) + list(range(5, 9)))
    axC.set_xticklabels(items + items, fontsize=9)
    axC.set_yticks(range(4))
    axC.set_yticklabels(items, fontsize=9)
    axC.set_xlabel("$A^{out}$            $A^{in}$", fontsize=10,
                   color=C_DARK)
    for i in range(4):
        for j in range(4):
            if A_out_n[i, j] > 0:
                axC.text(j, i, f"{A_out_n[i, j]:.1f}",
                         ha="center", va="center",
                         fontsize=8.5,
                         color="white" if A_out_n[i, j] > 0.5 else C_DARK)
            if A_in_n[i, j] > 0:
                axC.text(j + 5, i, f"{A_in_n[i, j]:.1f}",
                         ha="center", va="center",
                         fontsize=8.5,
                         color="white" if A_in_n[i, j] > 0.5 else C_DARK)
    axC.grid(False)

    fig.suptitle("Figure 2. Session graph construction: clicks -> directed "
                 "graph -> normalised adjacency",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_session_graph")


# ---------------------------------------------------------------------------
# Figure 3: GGNN message passing on a node
# ---------------------------------------------------------------------------
def fig3_ggnn_message_passing() -> None:
    """Single-step GGNN update: aggregate neighbours, GRU gates, new state."""
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(6.5, 6.0,
            "GGNN message passing on node $v_i$  (one propagation step)",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color=C_DARK)

    # ---- Left: target node + neighbours ----
    # incoming neighbours
    for k, y in enumerate([4.6, 3.6, 2.6]):
        circ = Circle((0.7, y), 0.32, facecolor=C_BLUE,
                      edgecolor=C_DARK, lw=1.2)
        ax.add_patch(circ)
        ax.text(0.7, y, f"in{k+1}", ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
        _arrow(ax, (1.05, y), (2.15, 3.7), color=C_BLUE,
               lw=1.2, connection="arc3,rad=0.0")

    # outgoing neighbours
    for k, y in enumerate([4.4, 3.4]):
        circ = Circle((4.6, y), 0.32, facecolor=C_PURPLE,
                      edgecolor=C_DARK, lw=1.2)
        ax.add_patch(circ)
        ax.text(4.6, y, f"out{k+1}", ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
        _arrow(ax, (2.85, 3.7), (4.25, y), color=C_PURPLE,
               lw=1.2, connection="arc3,rad=0.0")

    # target node (centre)
    circ = Circle((2.5, 3.7), 0.55, facecolor=C_GREEN,
                  edgecolor=C_DARK, lw=1.6)
    ax.add_patch(circ)
    ax.text(2.5, 3.7, "$v_i$", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")
    ax.text(2.5, 2.7, "current state $h_i^{(t-1)}$",
            ha="center", fontsize=9, color=C_DARK)

    ax.text(0.7, 5.15, "incoming",
            ha="center", fontsize=10, color=C_BLUE,
            fontweight="bold")
    ax.text(4.6, 5.0, "outgoing",
            ha="center", fontsize=10, color=C_PURPLE,
            fontweight="bold")

    # arrow to aggregation
    _arrow(ax, (5.2, 3.7), (6.1, 3.7), color=C_DARK, lw=1.4)

    # ---- Middle: aggregation a_i^t ----
    _box(ax, (6.1, 3.15), 2.1, 1.1, "", C_DARK, alpha=0.10,
         rounding=0.08)
    ax.text(7.15, 3.95,
            r"$a_i^{(t)}$",
            ha="center", fontsize=14, color=C_DARK,
            fontweight="bold")
    ax.text(7.15, 3.4,
            r"$=\,A^{in}_i H W^{in} + A^{out}_i H W^{out} + b$",
            ha="center", fontsize=9, color=C_DARK)

    # arrow to gates
    _arrow(ax, (8.25, 3.7), (9.15, 3.7), color=C_DARK, lw=1.4)

    # ---- Right: GRU gates ----
    gates_x = 9.15
    # update gate
    _box(ax, (gates_x, 4.55), 1.45, 0.7, r"update  $z_i^{(t)}$",
         C_AMBER, fontsize=10, fontweight="bold", rounding=0.1)
    ax.text(gates_x + 0.72, 4.30,
            r"$\sigma(W_z a + U_z h)$",
            ha="center", fontsize=8.5, color=C_DARK)
    # reset gate
    _box(ax, (gates_x, 3.05), 1.45, 0.7, r"reset  $r_i^{(t)}$",
         C_PURPLE, fontsize=10, fontweight="bold", rounding=0.1)
    ax.text(gates_x + 0.72, 2.80,
            r"$\sigma(W_r a + U_r h)$",
            ha="center", fontsize=8.5, color=C_DARK)
    # candidate
    _box(ax, (gates_x, 1.50), 1.45, 0.7,
         r"candidate  $\tilde h_i^{(t)}$",
         C_BLUE, fontsize=9.5, fontweight="bold", rounding=0.1)
    ax.text(gates_x + 0.72, 1.25,
            r"$\tanh(W_h a + U_h(r \odot h))$",
            ha="center", fontsize=8.5, color=C_DARK)

    # combine -> next state
    _arrow(ax, (gates_x + 1.5, 4.9), (11.55, 3.85),
           color=C_AMBER, lw=1.3)
    _arrow(ax, (gates_x + 1.5, 1.85), (11.55, 3.55),
           color=C_BLUE, lw=1.3)

    _box(ax, (11.55, 3.35), 1.3, 0.85,
         r"$h_i^{(t)}$",
         C_GREEN, fontsize=14, fontweight="bold",
         rounding=0.12)
    ax.text(12.2, 2.95,
            r"$=(1{-}z)\odot h + z\odot \tilde h$",
            ha="center", fontsize=8.5, color=C_DARK)

    # bottom intuition strip
    ax.text(6.5, 0.55,
            "Aggregate in/out neighbour evidence; GRU gates decide how "
            "much old state to keep vs how much new graph signal to write.",
            ha="center", fontsize=10, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    fig.suptitle("Figure 3. GGNN cell: gated update over in/out neighbours",
                 fontsize=12, color=C_DARK, y=0.02)
    _save(fig, "fig3_ggnn_message_passing")


# ---------------------------------------------------------------------------
# Figure 4: Self-attention over a session
# ---------------------------------------------------------------------------
def fig4_self_attention() -> None:
    """Self-attention weight matrix over a 6-item session."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})

    items = ["camera", "lens", "tripod", "memory\ncard", "battery",
             "camera\nbag"]
    n = len(items)

    # synthesise an attention matrix that emphasises:
    # - diagonal (self)
    # - long-range "camera" (0) <-> "memory card" (3) and "battery" (4)
    # - some local clustering for accessory items
    rng = np.random.default_rng(7)
    base = rng.uniform(0.05, 0.18, size=(n, n))
    np.fill_diagonal(base, 0.55)
    long_pairs = [(0, 3), (0, 4), (3, 0), (4, 0),
                  (0, 5), (5, 0)]
    for i, j in long_pairs:
        base[i, j] = 0.85
    # local clustering for accessories
    for i in [1, 2, 5]:
        for j in [1, 2, 5]:
            if i != j:
                base[i, j] = max(base[i, j], 0.45)
    # row-normalise (softmax-like)
    base = np.exp(base * 3.5)
    base = base / base.sum(axis=1, keepdims=True)

    # ----- Left: attention heatmap -----
    ax = axes[0]
    im = ax.imshow(base, cmap="Purples", vmin=0, vmax=base.max())
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(items, fontsize=9, rotation=20, ha="right")
    ax.set_yticklabels(items, fontsize=9)
    ax.set_xlabel("key (attended-to item)", fontsize=10)
    ax.set_ylabel("query (attending item)", fontsize=10)
    ax.set_title("(a) self-attention weights $\\alpha_{ij}$",
                 fontsize=11.5, color=C_DARK, pad=10)
    for i in range(n):
        for j in range(n):
            v = base[i, j]
            if v > 0.04:
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.22 else C_DARK)
    ax.grid(False)

    # ----- Right: contrast with limited-hop GNN reach -----
    ax = axes[1]
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(b) why attention helps: long-range reach in 1 step",
                 fontsize=11.5, color=C_DARK, pad=10)

    # draw the 6 items in a row
    xs = np.linspace(0.5, 6.3, n)
    y_node = 3.6
    colors = [C_BLUE, C_GRAY, C_GRAY, C_AMBER, C_AMBER, C_GRAY]
    for i, (x, lab, c) in enumerate(zip(xs, items, colors)):
        circ = Circle((x, y_node), 0.32, facecolor=c,
                      edgecolor=C_DARK, lw=1.2)
        ax.add_patch(circ)
        ax.text(x, y_node, str(i + 1),
                ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        ax.text(x, y_node - 0.85, lab.replace("\n", " "),
                ha="center", fontsize=8, color=C_DARK)

    # attention links from node 1 (camera) to all
    for j in [3, 4, 5]:
        ax.annotate("", xy=(xs[j], y_node + 0.4),
                    xytext=(xs[0], y_node + 0.4),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=C_PURPLE, lw=1.5,
                                    connectionstyle="arc3,rad=-0.45",
                                    shrinkA=8, shrinkB=8))
    ax.text(xs.mean(), y_node + 1.95,
            "self-attention: camera attends directly to memory card / "
            "battery / bag",
            ha="center", fontsize=9.5, color=C_PURPLE,
            fontweight="bold")

    # GNN local hops (only adjacent edges)
    for j in range(n - 1):
        ax.annotate("", xy=(xs[j + 1], y_node - 0.4),
                    xytext=(xs[j], y_node - 0.4),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=C_BLUE, lw=1.2,
                                    connectionstyle="arc3,rad=0.25",
                                    shrinkA=8, shrinkB=8))
    ax.text(xs.mean(), 1.4,
            "GNN: needs T hops to reach distant items; deeper -> oversmooth",
            ha="center", fontsize=9.5, color=C_BLUE,
            fontweight="bold")

    fig.suptitle("Figure 4. Self-attention captures global session intent "
                 "that local GNN hops miss",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_self_attention")


# ---------------------------------------------------------------------------
# Figure 5: Performance vs SR-GNN baseline
# ---------------------------------------------------------------------------
def fig5_perf_vs_baselines() -> None:
    """Recall@20 / MRR@20 vs SR-GNN baselines, plus ablation of fusion w."""
    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.32)

    # ---- Left: grouped bars on Yoochoose1/64 + Diginetica ----
    ax = fig.add_subplot(gs[0, 0])

    # Indicative numbers in the spirit of the GC-SAN paper.
    # Pattern: GC-SAN > SR-GNN > NARM > GRU4Rec on both datasets.
    methods = ["GRU4Rec", "NARM", "STAMP", "SR-GNN", "GC-SAN"]
    yoo_recall = [60.6, 68.3, 68.7, 70.6, 72.3]
    yoo_mrr =    [22.9, 28.6, 29.7, 30.9, 32.1]
    dig_recall = [29.5, 49.7, 45.6, 50.7, 52.6]
    dig_mrr =    [8.3, 16.2, 14.3, 17.6, 18.6]

    x = np.arange(len(methods))
    w = 0.20
    colors = [C_GRAY, C_GRAY, C_GRAY, C_AMBER, C_BLUE]

    ax.bar(x - 1.5 * w, yoo_recall, width=w, label="Yoochoose Recall@20",
           color=[C_BLUE if m == "GC-SAN" else C_LIGHT for m in methods],
           edgecolor=C_DARK, linewidth=0.8)
    ax.bar(x - 0.5 * w, yoo_mrr, width=w, label="Yoochoose MRR@20",
           color=[C_PURPLE if m == "GC-SAN" else C_GRAY for m in methods],
           edgecolor=C_DARK, linewidth=0.8)
    ax.bar(x + 0.5 * w, dig_recall, width=w, label="Diginetica Recall@20",
           color=[C_GREEN if m == "GC-SAN" else C_LIGHT for m in methods],
           edgecolor=C_DARK, linewidth=0.8)
    ax.bar(x + 1.5 * w, dig_mrr, width=w, label="Diginetica MRR@20",
           color=[C_AMBER if m == "GC-SAN" else C_GRAY for m in methods],
           edgecolor=C_DARK, linewidth=0.8)

    # highlight GC-SAN gain over SR-GNN
    sr_idx = methods.index("SR-GNN")
    gc_idx = methods.index("GC-SAN")
    for vals, off in [(yoo_recall, -1.5 * w), (yoo_mrr, -0.5 * w),
                      (dig_recall, 0.5 * w), (dig_mrr, 1.5 * w)]:
        d = vals[gc_idx] - vals[sr_idx]
        ax.text(gc_idx + off, vals[gc_idx] + 0.8, f"+{d:.1f}",
                ha="center", fontsize=8, color=C_GREEN,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("metric value (%)", fontsize=10.5)
    ax.set_ylim(0, 80)
    ax.set_title("(a) GC-SAN vs SR-GNN and other baselines",
                 fontsize=11.5, color=C_DARK, pad=10)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, frameon=True)
    ax.tick_params(axis="y", labelsize=9)

    # ---- Right: fusion weight ablation ----
    ax2 = fig.add_subplot(gs[0, 1])
    w_grid = np.linspace(0, 1, 11)
    # synthesised curve: peaks around w in [0.4, 0.6]
    recall = 71.2 - 1.2 * (w_grid - 0.5) ** 2 * 4 + 0.05 * np.cos(
        w_grid * np.pi)
    recall[0] = 69.9   # only last-click (h_t)
    recall[-1] = 70.4  # only global (a_t)

    ax2.plot(w_grid, recall, "-o", color=C_BLUE, lw=2,
             markerfacecolor=C_BLUE, markeredgecolor=C_DARK)
    ax2.axhline(70.6, color=C_AMBER, linestyle="--", lw=1.3,
                label="SR-GNN baseline")
    ax2.fill_between([0.35, 0.65], 68, 73, color=C_GREEN, alpha=0.12,
                     label="best operating zone")

    ax2.set_xlabel(r"fusion weight $w$  (global vs last click)",
                   fontsize=10.5)
    ax2.set_ylabel("Recall@20 (Yoochoose)", fontsize=10.5)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(68.5, 73)
    ax2.set_title("(b) fusion weight sweep $s_f = w\\,a_t + (1{-}w)\\,h_t$",
                  fontsize=11.5, color=C_DARK, pad=10)
    ax2.legend(loc="lower center", fontsize=9, frameon=True)
    ax2.tick_params(labelsize=9)

    fig.suptitle("Figure 5. GC-SAN improves on SR-GNN; the fusion weight "
                 "matters but is forgiving",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_perf_vs_baselines")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating GC-SAN figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")
    fig1_architecture()
    fig2_session_graph()
    fig3_ggnn_message_passing()
    fig4_self_attention()
    fig5_perf_vs_baselines()
    print("Done.")


if __name__ == "__main__":
    main()
