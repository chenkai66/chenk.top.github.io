"""
Figure generation script for NLP Part 04: Attention & Transformer.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure teaches one specific idea cleanly, with no clutter.

Figures:
    fig1_attention_heatmap        Cross-attention alignment heatmap on a
                                  small EN -> FR translation example.
    fig2_qkv_computation          Step-by-step scaled dot-product attention:
                                  Q.K^T -> /sqrt(d) -> softmax -> @V.
    fig3_multihead_attention      Multi-head attention architecture: parallel
                                  heads, projection, concat, output.
    fig4_positional_encoding      Sinusoidal positional encoding heatmap +
                                  per-dimension waveforms.
    fig5_causal_mask              Causal mask visualisation: pre-mask scores,
                                  mask pattern, post-softmax weights.
    fig6_transformer_block        Encoder block + decoder block side-by-side
                                  with sub-layers, residuals, LayerNorm.
    fig7_receptive_field          Self-attention vs. CNN vs. RNN receptive
                                  field growth with depth/distance.

Usage:
    python3 scripts/figures/nlp/04-attention-transformer.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "nlp" / "attention-transformer"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "nlp" / "04-注意力机制与Transformer"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Attention weight heatmap (translation alignment)
# ---------------------------------------------------------------------------
def fig1_attention_heatmap() -> None:
    """Cross-attention alignment heatmap for a small EN -> FR example."""
    src = ["The", "cat", "sat", "on", "the", "mat", "."]
    tgt = ["Le", "chat", "s'est", "assis", "sur", "le", "tapis", "."]

    # Hand-crafted attention pattern that mostly tracks word alignment
    # but mixes a little so it looks like a real model.
    n_t, n_s = len(tgt), len(src)
    A = np.zeros((n_t, n_s))
    align = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    for i in range(n_t):
        j = align[i]
        for k in range(n_s):
            A[i, k] = np.exp(-0.9 * (k - j) ** 2)
        # small diffuse component
        A[i] += 0.04
    A = A / A.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(A, cmap="Blues", aspect="auto", vmin=0, vmax=A.max())

    ax.set_xticks(range(n_s))
    ax.set_xticklabels(src, fontsize=11)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(tgt, fontsize=11)
    ax.set_xlabel("Source (English)", fontsize=12, color=C_DARK)
    ax.set_ylabel("Target (French)", fontsize=12, color=C_DARK)
    ax.set_title(
        "Cross-attention alignment: each target word attends to source words",
        fontsize=13, color=C_DARK, pad=12,
    )

    # Annotate the strongest cell per row
    for i in range(n_t):
        j = int(np.argmax(A[i]))
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                color="white" if A[i, j] > 0.4 else C_DARK, fontsize=9, weight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("attention weight", fontsize=10)
    ax.grid(False)

    fig.tight_layout()
    _save(fig, "fig1_attention_heatmap")


# ---------------------------------------------------------------------------
# Figure 2: Scaled dot-product attention pipeline
# ---------------------------------------------------------------------------
def fig2_qkv_computation() -> None:
    """Visualise the four steps: Q.K^T, /sqrt(d), softmax, @V."""
    rng = np.random.default_rng(3)
    n, d = 5, 4

    Q = rng.normal(0, 1, (n, d))
    K = rng.normal(0, 1, (n, d))
    V = rng.normal(0, 1, (n, d))

    scores = Q @ K.T              # (n, n)
    scaled = scores / np.sqrt(d)
    # softmax row-wise
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    attn = e / e.sum(axis=1, keepdims=True)
    out = attn @ V

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    panels = [
        (scores, "1. Scores = Q . K^T", "RdBu_r", True),
        (scaled, f"2. Scale: divide by sqrt(d_k) = {np.sqrt(d):.2f}", "RdBu_r", True),
        (attn,   "3. Softmax (row-wise)", "Blues", False),
        (out,    "4. Output = Attn . V", "PuOr", True),
    ]

    for ax, (M, title, cmap, sym) in zip(axes, panels):
        if sym:
            v = float(np.max(np.abs(M)))
            im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v)
        else:
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, color=C_DARK, pad=8)
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
        ax.set_xticklabels([f"{i+1}" for i in range(M.shape[1])], fontsize=8)
        ax.set_yticklabels([f"{i+1}" for i in range(M.shape[0])], fontsize=8)
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                # text colour: white if dark cell
                bg = abs(val) / (np.max(np.abs(M)) + 1e-9)
                color = "white" if bg > 0.55 else C_DARK
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("query position", fontsize=10)
    axes[2].set_ylabel("query (sums to 1)", fontsize=10)
    axes[2].set_xlabel("key position", fontsize=10)

    fig.suptitle("Scaled dot-product attention, step by step",
                 fontsize=14, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_qkv_computation")


# ---------------------------------------------------------------------------
# Figure 3: Multi-head attention architecture
# ---------------------------------------------------------------------------
def fig3_multihead_attention() -> None:
    """Block diagram: input -> h parallel heads -> concat -> W^O."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_facecolor("white")

    def box(x, y, w, h, label, color, fc=None, fontsize=10, weight="normal"):
        fc = fc or color
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                           linewidth=1.4, edgecolor=color, facecolor=fc, alpha=0.95)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=C_DARK, weight=weight)

    def arrow(x1, y1, x2, y2, color=C_GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="->", mutation_scale=14,
                                     color=color, linewidth=1.3))

    # Inputs
    box(0.3, 5.3, 1.2, 0.7, "Q",  C_BLUE,   "#dbeafe", 12, "bold")
    box(1.7, 5.3, 1.2, 0.7, "K",  C_PURPLE, "#ede9fe", 12, "bold")
    box(3.1, 5.3, 1.2, 0.7, "V",  C_GREEN,  "#d1fae5", 12, "bold")

    # h linear projections per input -> h heads
    n_heads = 4
    head_y = 3.4
    head_w, head_h = 1.6, 0.9
    spacing = 2.1
    head_x = [0.5 + i * spacing for i in range(n_heads)]

    for i, x in enumerate(head_x):
        # show projection lines from Q, K, V into each head
        for sx in (0.9, 2.3, 3.7):
            arrow(sx, 5.28, x + head_w / 2, head_y + head_h, color=C_LIGHT)
        box(x, head_y, head_w, head_h,
            f"head {i+1}\nAttn(QW_{i+1}, KW_{i+1}, VW_{i+1})",
            C_AMBER, "#fef3c7", 9)

    # Concat
    box(1.0, 1.95, 9.0, 0.7, "Concat(head_1, ..., head_h)",
        C_DARK, "#f1f5f9", 11, "bold")
    for x in head_x:
        arrow(x + head_w / 2, head_y, x + head_w / 2, 2.65, color=C_GRAY)

    # Output projection
    box(3.5, 0.6, 4.0, 0.8, "W^O  (linear)", C_PURPLE, "#ede9fe", 11, "bold")
    arrow(5.5, 1.95, 5.5, 1.4, color=C_GRAY)

    # Output arrow
    arrow(5.5, 0.6, 5.5, 0.05, color=C_DARK)
    ax.text(5.7, 0.18, "MultiHead(Q, K, V)", fontsize=10, color=C_DARK, weight="bold")

    # Title and side note
    ax.text(5.5, 6.3, "Multi-Head Attention",
            ha="center", fontsize=15, color=C_DARK, weight="bold")
    ax.text(10.9, 4.3, f"h = {n_heads} parallel heads\nd_k = d_v = d_model / h",
            ha="right", fontsize=10, color=C_GRAY, style="italic")

    _save(fig, "fig3_multihead_attention")


# ---------------------------------------------------------------------------
# Figure 4: Sinusoidal positional encoding
# ---------------------------------------------------------------------------
def fig4_positional_encoding() -> None:
    """Heatmap of PE plus a few per-dimension sinusoids."""
    d_model = 128
    max_len = 100
    pos = np.arange(max_len)[:, None]
    i = np.arange(d_model)[None, :]
    div_term = np.power(10000.0, (2 * (i // 2)) / d_model)
    PE = np.zeros((max_len, d_model))
    PE[:, 0::2] = np.sin(pos / div_term[:, 0::2])
    PE[:, 1::2] = np.cos(pos / div_term[:, 1::2])

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)

    # Left: heatmap
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(PE, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xlabel("encoding dimension i", fontsize=11, color=C_DARK)
    ax1.set_ylabel("position", fontsize=11, color=C_DARK)
    ax1.set_title("Sinusoidal positional encoding (d_model=128)",
                  fontsize=12, color=C_DARK)
    ax1.grid(False)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    cbar.set_label("PE value", fontsize=10)

    # Right: a few dimensions plotted as waveforms
    ax2 = fig.add_subplot(gs[1])
    dims_to_plot = [0, 4, 16, 60]
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    for d, c in zip(dims_to_plot, colors):
        ax2.plot(np.arange(max_len), PE[:, d], color=c, linewidth=1.6,
                 label=f"dim {d}")
    ax2.set_xlabel("position", fontsize=11, color=C_DARK)
    ax2.set_ylabel("PE value", fontsize=11, color=C_DARK)
    ax2.set_title("Low dims oscillate fast, high dims slow",
                  fontsize=12, color=C_DARK)
    ax2.legend(loc="upper right", fontsize=9, frameon=True)
    ax2.set_ylim(-1.15, 1.15)

    fig.tight_layout()
    _save(fig, "fig4_positional_encoding")


# ---------------------------------------------------------------------------
# Figure 5: Causal mask
# ---------------------------------------------------------------------------
def fig5_causal_mask() -> None:
    """Show pre-mask scores, the additive mask, post-softmax weights."""
    rng = np.random.default_rng(11)
    n = 6
    tokens = ["<s>", "The", "cat", "sat", "on", "mat"]

    scores = rng.normal(0, 1.2, (n, n))
    mask_add = np.where(np.arange(n)[:, None] < np.arange(n)[None, :],
                        -np.inf, 0.0)
    masked = scores + mask_add
    e = np.exp(masked - np.nanmax(masked, axis=1, keepdims=True))
    attn = e / e.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # Pre-mask scores
    ax = axes[0]
    v = float(np.max(np.abs(scores)))
    im = ax.imshow(scores, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title("1. Raw scores (Q.K^T / sqrt(d))", fontsize=11, color=C_DARK)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=30, fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Mask pattern
    ax = axes[1]
    mask_show = (mask_add == 0).astype(float)  # 1 = allowed
    ax.imshow(mask_show, cmap="Greens", vmin=0, vmax=1)
    ax.set_title("2. Causal mask (green=allowed, white=blocked)",
                 fontsize=11, color=C_DARK)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=30, fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            sym = "OK" if j <= i else "X"
            color = C_DARK if j <= i else C_GRAY
            ax.text(j, i, sym, ha="center", va="center",
                    fontsize=10, color=color, weight="bold")

    # Post-softmax weights
    ax = axes[2]
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max())
    ax.set_title("3. After softmax: each row sums to 1",
                 fontsize=11, color=C_DARK)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=30, fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if j > i:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=8, color=C_GRAY)
            else:
                v = attn[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v > 0.4 else C_DARK)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Causal masking: position i attends only to positions <= i",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_causal_mask")


# ---------------------------------------------------------------------------
# Figure 6: Encoder block + Decoder block side by side
# ---------------------------------------------------------------------------
def fig6_transformer_block() -> None:
    """Side-by-side encoder and decoder blocks with sub-layers labelled."""
    fig, ax = plt.subplots(figsize=(13, 8.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def block(x, y, w, h, label, color, fc, fontsize=10, weight="normal"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           linewidth=1.5, edgecolor=color, facecolor=fc, alpha=0.95)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=C_DARK, weight=weight)

    def arrow(x1, y1, x2, y2, color=C_DARK, lw=1.4):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="->", mutation_scale=14,
                                     color=color, linewidth=lw))

    # ENCODER (left)
    ex, ew = 0.8, 4.8
    ax.text(ex + ew / 2, 8.55, "Encoder block (x N)",
            ha="center", fontsize=13, color=C_BLUE, weight="bold")
    # Outer box
    outer = FancyBboxPatch((ex - 0.15, 0.8), ew + 0.3, 7.5,
                           boxstyle="round,pad=0.05", linewidth=1.2,
                           edgecolor=C_BLUE, facecolor="#eff6ff", alpha=0.4)
    ax.add_patch(outer)

    block(ex, 1.0, ew, 0.7, "Input embedding + Positional encoding",
          C_GRAY, "#f1f5f9", 10)
    arrow(ex + ew / 2, 1.7, ex + ew / 2, 2.1)

    block(ex, 2.1, ew, 1.0, "Multi-Head Self-Attention", C_BLUE, "#dbeafe", 11, "bold")
    arrow(ex + ew / 2, 3.1, ex + ew / 2, 3.5)
    block(ex, 3.5, ew, 0.55, "Add & LayerNorm", C_GRAY, "#f1f5f9", 9)
    arrow(ex + ew / 2, 4.05, ex + ew / 2, 4.45)

    block(ex, 4.45, ew, 1.0, "Feed-Forward (FFN)", C_GREEN, "#d1fae5", 11, "bold")
    arrow(ex + ew / 2, 5.45, ex + ew / 2, 5.85)
    block(ex, 5.85, ew, 0.55, "Add & LayerNorm", C_GRAY, "#f1f5f9", 9)
    arrow(ex + ew / 2, 6.4, ex + ew / 2, 7.4)

    block(ex, 7.4, ew, 0.7, "Encoder output -> to all decoder layers",
          C_BLUE, "#dbeafe", 10, "bold")

    # Residual arc indicators
    ax.annotate("", xy=(ex + ew + 0.1, 3.75), xytext=(ex + ew + 0.1, 2.1),
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.2,
                                connectionstyle="arc3,rad=0.4"))
    ax.text(ex + ew + 0.35, 2.95, "residual", color=C_AMBER, fontsize=8, rotation=90)
    ax.annotate("", xy=(ex + ew + 0.1, 6.1), xytext=(ex + ew + 0.1, 4.45),
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.2,
                                connectionstyle="arc3,rad=0.4"))
    ax.text(ex + ew + 0.35, 5.3, "residual", color=C_AMBER, fontsize=8, rotation=90)

    # DECODER (right)
    dx, dw = 7.2, 4.8
    ax.text(dx + dw / 2, 8.55, "Decoder block (x N)",
            ha="center", fontsize=13, color=C_PURPLE, weight="bold")
    outer = FancyBboxPatch((dx - 0.15, 0.8), dw + 0.3, 7.5,
                           boxstyle="round,pad=0.05", linewidth=1.2,
                           edgecolor=C_PURPLE, facecolor="#faf5ff", alpha=0.4)
    ax.add_patch(outer)

    block(dx, 1.0, dw, 0.7, "Output embedding + Positional encoding",
          C_GRAY, "#f1f5f9", 10)
    arrow(dx + dw / 2, 1.7, dx + dw / 2, 2.1)

    block(dx, 2.1, dw, 1.0, "Masked Multi-Head Self-Attention",
          C_PURPLE, "#ede9fe", 11, "bold")
    arrow(dx + dw / 2, 3.1, dx + dw / 2, 3.5)
    block(dx, 3.5, dw, 0.45, "Add & LayerNorm", C_GRAY, "#f1f5f9", 9)
    arrow(dx + dw / 2, 3.95, dx + dw / 2, 4.35)

    block(dx, 4.35, dw, 1.0, "Cross-Attention\nQ from decoder, K,V from encoder",
          C_AMBER, "#fef3c7", 10, "bold")
    arrow(dx + dw / 2, 5.35, dx + dw / 2, 5.75)
    block(dx, 5.75, dw, 0.45, "Add & LayerNorm", C_GRAY, "#f1f5f9", 9)
    arrow(dx + dw / 2, 6.2, dx + dw / 2, 6.6)

    block(dx, 6.6, dw, 0.85, "Feed-Forward (FFN)", C_GREEN, "#d1fae5", 11, "bold")
    arrow(dx + dw / 2, 7.45, dx + dw / 2, 7.85)
    block(dx, 7.85, dw, 0.35, "Add & LayerNorm -> Linear -> Softmax",
          C_GRAY, "#f1f5f9", 8)

    # Cross-attention link from encoder output to decoder cross-attn
    ax.add_patch(FancyArrowPatch((ex + ew + 0.05, 7.75),
                                 (dx, 4.85),
                                 arrowstyle="->", mutation_scale=16,
                                 color=C_AMBER, linewidth=1.8,
                                 connectionstyle="arc3,rad=-0.2"))
    ax.text(6.4, 6.6, "K, V from\nencoder",
            ha="center", fontsize=9, color=C_AMBER, style="italic")

    fig.suptitle("Transformer encoder and decoder blocks",
                 fontsize=14, color=C_DARK, y=0.995)
    _save(fig, "fig6_transformer_block")


# ---------------------------------------------------------------------------
# Figure 7: Receptive field comparison
# ---------------------------------------------------------------------------
def fig7_receptive_field() -> None:
    """Visualise how many tokens each architecture can 'see' per step."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6),
                             gridspec_kw={"wspace": 0.3})

    n = 9
    target = 4  # the position whose context we visualise

    # Helper to draw a row of token circles and connections
    def draw_arch(ax, title, edges, subtitle, colour):
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-2.2, 1.5)
        ax.axis("off")
        ax.set_title(title, fontsize=12, color=C_DARK, weight="bold", pad=10)

        # Draw connection arcs
        for src in edges:
            if src == target:
                continue
            rad = 0.5 if src < target else -0.5
            ax.add_patch(FancyArrowPatch((src, 0), (target, 0),
                                         arrowstyle="-", mutation_scale=10,
                                         color=colour, linewidth=1.6, alpha=0.7,
                                         connectionstyle=f"arc3,rad={rad*0.5}"))

        # Tokens
        for i in range(n):
            is_target = (i == target)
            in_field = (i in edges) or is_target
            face = colour if is_target else ("#dbeafe" if in_field else "white")
            edge = colour if in_field else C_GRAY
            ax.add_patch(plt.Circle((i, 0), 0.32, facecolor=face,
                                    edgecolor=edge, linewidth=1.6, zorder=3))
            ax.text(i, 0, str(i + 1), ha="center", va="center",
                    fontsize=10,
                    color="white" if is_target else C_DARK,
                    weight="bold" if is_target else "normal", zorder=4)

        ax.text((n - 1) / 2, -1.6, subtitle, ha="center",
                fontsize=10, color=C_DARK)
        ax.text(target, 0.7, "query", ha="center", fontsize=9,
                color=colour, weight="bold")

    # Self-attention: every token connects to target
    sa_edges = list(range(n))
    draw_arch(axes[0], "Self-Attention",
              sa_edges,
              "Receptive field = N (entire sequence)\nPath length: O(1)",
              C_BLUE)

    # CNN with kernel=3, 1 layer: only neighbours +-1
    cnn_edges = [target - 1, target, target + 1]
    cnn_edges = [e for e in cnn_edges if 0 <= e < n]
    draw_arch(axes[1], "CNN (kernel=3, 1 layer)",
              cnn_edges,
              "Receptive field = kernel size\nNeed log(N) layers to span input",
              C_GREEN)

    # RNN: all positions <= target (sequential), but path length is i
    rnn_edges = list(range(target + 1))
    draw_arch(axes[2], "RNN",
              rnn_edges,
              "Sees full past, but information\nflows through O(N) steps",
              C_AMBER)

    fig.suptitle(
        "Receptive field at the marked query token",
        fontsize=14, color=C_DARK, y=1.02,
    )
    _save(fig, "fig7_receptive_field")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating NLP Part 04 figures...")
    fig1_attention_heatmap()
    fig2_qkv_computation()
    fig3_multihead_attention()
    fig4_positional_encoding()
    fig5_causal_mask()
    fig6_transformer_block()
    fig7_receptive_field()
    print("Done.")


if __name__ == "__main__":
    main()
