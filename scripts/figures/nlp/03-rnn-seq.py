"""
Figure generation script for NLP Part 03: RNN and Sequence Modeling.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single concept cleanly with consistent styling.

Figures:
    fig1_unrolled_rnn         Vanilla RNN unrolled across time steps with
                              recurrent weight sharing made explicit.
    fig2_vanishing_gradient   Gradient norm decay vs. time-step distance for
                              vanilla RNN (vanishing/exploding) vs. LSTM.
    fig3_lstm_gates           LSTM cell architecture with the three gates,
                              cell-state highway, and tensor flow.
    fig4_gru_cell             GRU cell architecture (reset + update gate)
                              shown side-by-side with the simpler topology.
    fig5_bidirectional_rnn    Bidirectional RNN: forward pass, backward pass,
                              and concatenated hidden state per position.
    fig6_seq2seq              Encoder-decoder Seq2Seq with the fixed-size
                              context-vector bottleneck highlighted.
    fig7_loss_curves          Synthetic training-loss curves comparing RNN,
                              LSTM, GRU on a long-dependency task.

Usage:
    python3 scripts/figures/nlp/03-rnn-seq.py

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

C_BLUE = "#2563eb"     # primary (input / forward)
C_PURPLE = "#7c3aed"   # secondary (state / hidden)
C_GREEN = "#10b981"    # accent / success / LSTM-good
C_AMBER = "#f59e0b"    # warning / highlight / problem
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_BG = "#f8fafc"
C_LIGHT_BLUE = "#dbeafe"
C_LIGHT_PURPLE = "#ede9fe"
C_LIGHT_GREEN = "#d1fae5"
C_LIGHT_AMBER = "#fef3c7"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "nlp" / "rnn-sequence-modeling"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "nlp" / "03-RNN与序列建模"

EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders at consistent DPI."""
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def _box(ax, xy, w, h, label, fc, ec=C_DARK, lw=1.4, fs=11, fw="bold"):
    """Draw a rounded rectangle with a centered label."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=C_DARK)


def _arrow(ax, p1, p2, color=C_DARK, lw=1.6, style="->", mut=14, ls="-"):
    """Draw an arrow between two points."""
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=mut,
                        linewidth=lw, color=color, linestyle=ls,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: Vanilla RNN unrolled
# ---------------------------------------------------------------------------
def fig1_unrolled_rnn() -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    n_steps = 5
    x_positions = [1 + 2 * i for i in range(n_steps)]
    labels_x = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
    labels_y = ["$y_1$", "$y_2$", "$y_3$", "$y_4$", "$y_5$"]
    labels_h = ["$h_1$", "$h_2$", "$h_3$", "$h_4$", "$h_5$"]

    # Initial hidden state on the left
    _box(ax, (-0.4, 2.2), 1.0, 0.9, "$h_0$", C_LIGHT_PURPLE, fs=12)

    prev_right = (0.6, 2.65)
    for i, x in enumerate(x_positions):
        # Input
        _box(ax, (x - 0.45, 0.4), 0.9, 0.8, labels_x[i], C_LIGHT_BLUE, fs=12)
        # Hidden cell
        _box(ax, (x - 0.55, 2.2), 1.1, 0.9, labels_h[i], C_LIGHT_PURPLE, fs=13)
        # Output
        _box(ax, (x - 0.45, 4.15), 0.9, 0.8, labels_y[i], C_LIGHT_GREEN, fs=12)

        # Arrows: x -> h (W_x), h -> y (W_y)
        _arrow(ax, (x, 1.2), (x, 2.2), color=C_BLUE)
        ax.text(x + 0.12, 1.7, "$W_x$", fontsize=9, color=C_BLUE)

        _arrow(ax, (x, 3.1), (x, 4.15), color=C_GREEN)
        ax.text(x + 0.12, 3.6, "$W_y$", fontsize=9, color=C_GREEN)

        # Recurrent arrow from previous hidden
        _arrow(ax, prev_right, (x - 0.55, 2.65), color=C_PURPLE, lw=1.8)
        ax.text((prev_right[0] + x - 0.55) / 2, 2.85, "$W_h$",
                fontsize=9, color=C_PURPLE, ha="center")

        prev_right = (x + 0.55, 2.65)

    # Trailing arrow into "..." for arbitrary length
    ax.annotate("", xy=(10.6, 2.65), xytext=(prev_right[0], 2.65),
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.8))
    ax.text(10.75, 2.65, "$\\cdots$", fontsize=16, color=C_PURPLE,
            ha="left", va="center")

    # Title and key equation
    ax.text(5.5, 5.3, "Vanilla RNN unrolled across time",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(5.5, 0.05,
            r"$h_t = \tanh(W_h\,h_{t-1} + W_x\,x_t + b)$   "
            r"$y_t = W_y\,h_t + b_y$",
            ha="center", fontsize=11, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_GRAY, boxstyle="round,pad=0.4"))

    # Note about parameter sharing
    ax.text(5.5, 4.95,
            "Same weights $W_h$, $W_x$, $W_y$ at every step (parameter sharing)",
            ha="center", fontsize=10, color=C_GRAY, style="italic")

    fig.tight_layout()
    save(fig, "fig1_unrolled_rnn")


# ---------------------------------------------------------------------------
# Figure 2: Vanishing / exploding gradient
# ---------------------------------------------------------------------------
def fig2_vanishing_gradient() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    steps = np.arange(0, 50)

    # Left: gradient norm under different lambda regimes
    lam_vanish = 0.85
    lam_stable = 1.00
    lam_explode = 1.10

    g_vanish = lam_vanish ** steps
    g_stable = np.ones_like(steps, dtype=float)
    g_explode = lam_explode ** steps

    ax1.plot(steps, g_vanish, color=C_AMBER, lw=2.4,
             label=r"Vanilla RNN, $\lambda=0.85$  (vanishing)")
    ax1.plot(steps, g_stable, color=C_GREEN, lw=2.4,
             label=r"LSTM cell highway, $\lambda\!\approx\!1$")
    ax1.plot(steps, g_explode, color=C_PURPLE, lw=2.4,
             label=r"Vanilla RNN, $\lambda=1.10$  (exploding)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Distance back through time  $T - t$", fontsize=11)
    ax1.set_ylabel(r"Gradient norm  $\|\partial h_T / \partial h_t\|$",
                   fontsize=11)
    ax1.set_title("Gradient flow over $T-t$ steps",
                  fontsize=12, fontweight="bold", color=C_DARK)
    ax1.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax1.set_ylim(1e-4, 1e2)
    ax1.axhline(1.0, color=C_GRAY, lw=0.8, ls="--")

    # Right: sentence with the long-range dependency highlighted
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis("off")

    words = ["The", "cat", ",", "which", "sat", "on", "the", "mat",
             "and", "purred", ",", "was", "happy"]
    highlight_idx = {1, 11}  # "cat", "was"
    x_w = np.linspace(0.4, 9.6, len(words))
    y_w = 3.3
    for w, x in zip(words, x_w):
        idx = words.index(w) if words.count(w) == 1 else None
        # Use position-based highlight directly
        is_hi = words.index(w) in highlight_idx if words.count(w) == 1 else False
        # safer: rebuild via enumerate
        pass

    for i, (w, x) in enumerate(zip(words, x_w)):
        is_hi = i in highlight_idx
        color = C_AMBER if is_hi else C_DARK
        weight = "bold" if is_hi else "normal"
        ax2.text(x, y_w, w, ha="center", va="center",
                 fontsize=12, color=color, fontweight=weight)

    # Arc from "cat" to "was"
    cat_x = x_w[1]
    was_x = x_w[11]
    arc_y = 4.0
    ax2.annotate("",
                 xy=(was_x, y_w + 0.35), xytext=(cat_x, y_w + 0.35),
                 arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=2,
                                 connectionstyle="arc3,rad=-0.45"))
    ax2.text((cat_x + was_x) / 2, arc_y + 0.35,
             "Subject-verb agreement across 10 tokens",
             ha="center", fontsize=10, color=C_AMBER, style="italic")

    ax2.text(5.0, 1.8,
             "Vanilla RNN cannot propagate this gradient.",
             ha="center", fontsize=11, color=C_DARK)
    ax2.text(5.0, 1.1,
             "Solutions: gated cells (LSTM/GRU), gradient clipping for explosions.",
             ha="center", fontsize=10, color=C_GRAY, style="italic")
    ax2.set_title("Why long-range dependencies break",
                  fontsize=12, fontweight="bold", color=C_DARK, pad=10)

    fig.tight_layout()
    save(fig, "fig2_vanishing_gradient")


# ---------------------------------------------------------------------------
# Figure 3: LSTM gates
# ---------------------------------------------------------------------------
def fig3_lstm_gates() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Cell outer box
    cell = FancyBboxPatch((0.6, 0.6), 10.8, 5.6,
                          boxstyle="round,pad=0.05,rounding_size=0.18",
                          linewidth=1.6, edgecolor=C_DARK,
                          facecolor=C_BG)
    ax.add_patch(cell)

    # Top horizontal "cell-state highway"
    ax.annotate("",
                xy=(11.2, 5.3), xytext=(0.8, 5.3),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=3.0))
    ax.text(0.85, 5.55, "$C_{t-1}$", fontsize=12, color=C_GREEN, fontweight="bold")
    ax.text(11.05, 5.55, "$C_t$", fontsize=12, color=C_GREEN,
            fontweight="bold", ha="right")
    ax.text(6.0, 5.85, "Cell-state highway (additive update)",
            ha="center", fontsize=10, color=C_GREEN, style="italic")

    # Bottom hidden-state line
    ax.annotate("",
                xy=(11.2, 1.3), xytext=(0.8, 1.3),
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=2.0))
    ax.text(0.85, 1.05, "$h_{t-1}$", fontsize=11, color=C_PURPLE, fontweight="bold")
    ax.text(11.05, 1.05, "$h_t$", fontsize=11, color=C_PURPLE,
            fontweight="bold", ha="right")

    # Input x_t entering bottom
    ax.annotate("",
                xy=(2.0, 1.3), xytext=(2.0, 0.05),
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=2.0))
    ax.text(2.0, -0.15, "$x_t$", fontsize=12, color=C_BLUE,
            fontweight="bold", ha="center", va="top")

    # Three gates
    gate_y = 2.6
    gate_h = 1.1
    gate_w = 1.6

    # Forget gate
    _box(ax, (2.6, gate_y), gate_w, gate_h, "Forget\n$f_t = \\sigma$",
         C_LIGHT_AMBER, fs=10, ec=C_AMBER)
    # Input gate
    _box(ax, (5.0, gate_y), gate_w, gate_h, "Input\n$i_t = \\sigma$",
         C_LIGHT_BLUE, fs=10, ec=C_BLUE)
    # Candidate
    _box(ax, (6.9, gate_y), gate_w, gate_h, "Candidate\n$\\tilde{C}_t = \\tanh$",
         C_LIGHT_PURPLE, fs=10, ec=C_PURPLE)
    # Output gate
    _box(ax, (9.0, gate_y), gate_w, gate_h, "Output\n$o_t = \\sigma$",
         C_LIGHT_GREEN, fs=10, ec=C_GREEN)

    # Arrows from gates to highway operations
    # Forget gate multiplies onto highway
    ax.plot([3.4, 3.4], [gate_y + gate_h, 5.3], color=C_AMBER, lw=1.6)
    ax.plot(3.4, 5.3, "o", markersize=12, markerfacecolor="white",
            markeredgecolor=C_AMBER, mew=2)
    ax.text(3.4, 5.3, r"$\times$", ha="center", va="center",
            fontsize=11, color=C_AMBER, fontweight="bold")

    # Input * Candidate combine then add to highway at x=6.0
    ax.plot([5.8, 5.8], [gate_y + gate_h, 4.4], color=C_BLUE, lw=1.6)
    ax.plot([7.7, 7.7], [gate_y + gate_h, 4.4], color=C_PURPLE, lw=1.6)
    ax.plot([5.8, 6.75], [4.4, 4.4], color=C_BLUE, lw=1.6)
    ax.plot([7.7, 6.75], [4.4, 4.4], color=C_PURPLE, lw=1.6)
    ax.plot(6.75, 4.4, "o", markersize=12, markerfacecolor="white",
            markeredgecolor=C_DARK, mew=2)
    ax.text(6.75, 4.4, r"$\times$", ha="center", va="center",
            fontsize=11, color=C_DARK, fontweight="bold")
    ax.plot([6.75, 6.75], [4.4, 5.3], color=C_DARK, lw=1.6)
    ax.plot(6.75, 5.3, "o", markersize=12, markerfacecolor="white",
            markeredgecolor=C_GREEN, mew=2)
    ax.text(6.75, 5.3, r"$+$", ha="center", va="center",
            fontsize=12, color=C_GREEN, fontweight="bold")

    # Output gate -> tanh(C_t) * o_t -> h_t
    ax.plot([9.8, 9.8], [gate_y + gate_h, 4.4], color=C_GREEN, lw=1.6)
    # tanh tap from cell-state line
    ax.plot([10.4, 10.4], [5.3, 4.4], color=C_GREEN, lw=1.6, ls="--")
    ax.text(10.55, 4.85, "tanh", fontsize=9, color=C_GREEN)
    ax.plot([9.8, 10.4], [4.4, 4.4], color=C_GREEN, lw=1.6)
    ax.plot(10.1, 4.4, "o", markersize=12, markerfacecolor="white",
            markeredgecolor=C_GREEN, mew=2)
    ax.text(10.1, 4.4, r"$\times$", ha="center", va="center",
            fontsize=11, color=C_GREEN, fontweight="bold")
    # Down to h_t output
    ax.annotate("", xy=(10.1, 1.3), xytext=(10.1, 4.4),
                arrowprops=dict(arrowstyle="-", color=C_PURPLE, lw=1.6))

    # Inputs into each gate (h_{t-1}, x_t join)
    for gx in [3.4, 5.8, 7.7, 9.8]:
        ax.plot([gx, gx], [1.3, gate_y], color=C_GRAY, lw=1.0, ls=":")

    ax.set_title("LSTM cell: three gates and the additive cell-state highway",
                 fontsize=12.5, fontweight="bold", color=C_DARK, pad=10)

    fig.tight_layout()
    save(fig, "fig3_lstm_gates")


# ---------------------------------------------------------------------------
# Figure 4: GRU cell
# ---------------------------------------------------------------------------
def fig4_gru_cell() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Outer cell
    cell = FancyBboxPatch((0.6, 0.6), 10.8, 4.7,
                          boxstyle="round,pad=0.05,rounding_size=0.18",
                          linewidth=1.6, edgecolor=C_DARK, facecolor=C_BG)
    ax.add_patch(cell)

    # Single hidden-state line (GRU has no separate cell state)
    ax.annotate("", xy=(11.2, 4.4), xytext=(0.8, 4.4),
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=2.6))
    ax.text(0.85, 4.65, "$h_{t-1}$", fontsize=12, color=C_PURPLE, fontweight="bold")
    ax.text(11.05, 4.65, "$h_t$", fontsize=12, color=C_PURPLE,
            fontweight="bold", ha="right")
    ax.text(6.0, 4.95, "Hidden state (no separate cell state)",
            ha="center", fontsize=10, color=C_PURPLE, style="italic")

    # Input x_t bottom
    ax.annotate("", xy=(2.0, 1.4), xytext=(2.0, 0.05),
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=2.0))
    ax.text(2.0, -0.1, "$x_t$", fontsize=12, color=C_BLUE,
            fontweight="bold", ha="center", va="top")
    ax.plot([2.0, 11.0], [1.4, 1.4], color=C_GRAY, lw=1.0, ls=":")

    # Two gates only
    _box(ax, (3.2, 2.2), 1.8, 1.2, "Reset\n$r_t = \\sigma$",
         C_LIGHT_AMBER, fs=11, ec=C_AMBER)
    _box(ax, (5.6, 2.2), 1.8, 1.2, "Update\n$z_t = \\sigma$",
         C_LIGHT_BLUE, fs=11, ec=C_BLUE)
    _box(ax, (8.0, 2.2), 1.8, 1.2, "Candidate\n$\\tilde{h}_t = \\tanh$",
         C_LIGHT_PURPLE, fs=11, ec=C_PURPLE)

    # Reset gate gates h_{t-1} for candidate
    ax.plot([4.1, 4.1], [3.4, 4.4], color=C_AMBER, lw=1.6, ls="--")
    ax.plot(4.1, 4.4, "o", markersize=11, markerfacecolor="white",
            markeredgecolor=C_AMBER, mew=2)
    ax.text(4.1, 4.4, r"$\times$", ha="center", va="center",
            fontsize=10, color=C_AMBER, fontweight="bold")
    ax.annotate("", xy=(8.4, 3.4), xytext=(4.1, 3.7),
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.2))

    # Update gate interpolates between old h and candidate
    ax.plot([6.5, 6.5], [3.4, 4.4], color=C_BLUE, lw=1.6, ls="--")
    ax.plot(6.5, 4.4, "o", markersize=11, markerfacecolor="white",
            markeredgecolor=C_BLUE, mew=2)
    ax.text(6.5, 4.4, "$1\\!-\\!z$", ha="center", va="center",
            fontsize=8, color=C_BLUE, fontweight="bold")

    # Candidate -> combine
    ax.plot([8.9, 8.9], [3.4, 4.4], color=C_PURPLE, lw=1.6)
    ax.plot(8.9, 4.4, "o", markersize=11, markerfacecolor="white",
            markeredgecolor=C_PURPLE, mew=2)
    ax.text(8.9, 4.4, "$z$", ha="center", va="center",
            fontsize=10, color=C_PURPLE, fontweight="bold")

    ax.set_title("GRU cell: two gates, one state — simpler than LSTM",
                 fontsize=12.5, fontweight="bold", color=C_DARK, pad=10)

    # Comparison footer
    ax.text(6.0, -0.6,
            "LSTM: 3 gates + cell state  |  GRU: 2 gates, ~25% fewer params, often comparable accuracy",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    fig.tight_layout()
    save(fig, "fig4_gru_cell")


# ---------------------------------------------------------------------------
# Figure 5: Bidirectional RNN
# ---------------------------------------------------------------------------
def fig5_bidirectional_rnn() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    n = 5
    xs = np.linspace(1.0, 10.0, n)

    # Inputs at the bottom
    tokens = ["He", "said", "not", "very", "good"]
    for x, tok in zip(xs, tokens):
        _box(ax, (x - 0.45, 0.4), 0.9, 0.7, tok,
             C_LIGHT_BLUE, fs=11)

    # Forward layer (purple)
    fy = 2.3
    for i, x in enumerate(xs):
        _box(ax, (x - 0.4, fy), 0.8, 0.7, f"$\\overrightarrow{{h}}_{i+1}$",
             C_LIGHT_PURPLE, fs=11)
        _arrow(ax, (x, 1.1), (x, fy), color=C_BLUE)
        if i > 0:
            _arrow(ax, (xs[i - 1] + 0.4, fy + 0.35),
                   (x - 0.4, fy + 0.35), color=C_PURPLE)

    # Backward layer (amber) — arrows go right-to-left
    by = 3.6
    for i, x in enumerate(xs):
        _box(ax, (x - 0.4, by), 0.8, 0.7, f"$\\overleftarrow{{h}}_{i+1}$",
             C_LIGHT_AMBER, fs=11)
        # Input also feeds backward layer
        _arrow(ax, (x + 0.15, 1.1), (x + 0.15, by),
               color=C_BLUE, lw=1.0, style="->", mut=10, ls=":")
        if i < n - 1:
            _arrow(ax, (xs[i + 1] - 0.4, by + 0.35),
                   (x + 0.4, by + 0.35), color=C_AMBER)

    # Concatenated output
    cy = 4.9
    for i, x in enumerate(xs):
        _box(ax, (x - 0.55, cy), 1.1, 0.7,
             f"$[\\overrightarrow{{h}}_{i+1};\\overleftarrow{{h}}_{i+1}]$",
             C_LIGHT_GREEN, fs=10)
        _arrow(ax, (x - 0.15, fy + 0.7), (x - 0.15, cy), color=C_PURPLE, lw=1.0)
        _arrow(ax, (x + 0.15, by + 0.7), (x + 0.15, cy), color=C_AMBER, lw=1.0)

    ax.set_title("Bidirectional RNN: forward + backward states concatenated per position",
                 fontsize=12, fontweight="bold", color=C_DARK, pad=10)
    ax.text(5.5, 5.85,
            'Each position sees both past and future — e.g. "not" can flip the sentiment of "good".',
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    fig.tight_layout()
    save(fig, "fig5_bidirectional_rnn")


# ---------------------------------------------------------------------------
# Figure 6: Seq2Seq encoder-decoder with bottleneck
# ---------------------------------------------------------------------------
def fig6_seq2seq() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    src = ["I", "love", "deep", "learning"]
    tgt = ["J'", "aime", "l'", "apprentissage", "profond"]

    # Encoder side (left) — purple
    enc_y = 2.0
    enc_x = np.linspace(0.6, 4.5, len(src))
    for i, (x, w) in enumerate(zip(enc_x, src)):
        _box(ax, (x - 0.4, enc_y), 0.8, 0.8, f"$h_{i+1}$",
             C_LIGHT_PURPLE, fs=11)
        _box(ax, (x - 0.4, 0.4), 0.8, 0.7, w, C_LIGHT_BLUE, fs=10)
        _arrow(ax, (x, 1.1), (x, enc_y), color=C_BLUE)
        if i > 0:
            _arrow(ax, (enc_x[i - 1] + 0.4, enc_y + 0.4),
                   (x - 0.4, enc_y + 0.4), color=C_PURPLE)

    ax.text(2.55, 3.05, "Encoder", ha="center", fontsize=11,
            color=C_PURPLE, fontweight="bold")

    # Context vector — the bottleneck
    ctx_x = 5.7
    ctx = FancyBboxPatch((ctx_x - 0.55, enc_y - 0.05), 1.1, 0.9,
                         boxstyle="round,pad=0.05,rounding_size=0.12",
                         linewidth=2.2, edgecolor=C_AMBER,
                         facecolor=C_LIGHT_AMBER)
    ax.add_patch(ctx)
    ax.text(ctx_x, enc_y + 0.4, "$c$", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_AMBER)
    ax.text(ctx_x, 3.05, "Context", ha="center", fontsize=10.5,
            color=C_AMBER, fontweight="bold")
    ax.text(ctx_x, 3.45, "(bottleneck)", ha="center", fontsize=9,
            color=C_AMBER, style="italic")

    _arrow(ax, (enc_x[-1] + 0.4, enc_y + 0.4),
           (ctx_x - 0.55, enc_y + 0.4), color=C_PURPLE, lw=2.0)

    # Decoder side (right) — green
    dec_y = 2.0
    dec_x = np.linspace(7.0, 11.4, len(tgt))
    for i, (x, w) in enumerate(zip(dec_x, tgt)):
        _box(ax, (x - 0.4, dec_y), 0.8, 0.8, f"$s_{i+1}$",
             C_LIGHT_GREEN, fs=11)
        _box(ax, (x - 0.4, 4.3), 0.8, 0.7, w, C_LIGHT_GREEN,
             fs=9.5, ec=C_GREEN)
        _arrow(ax, (x, dec_y + 0.8), (x, 4.3), color=C_GREEN)
        if i > 0:
            _arrow(ax, (dec_x[i - 1] + 0.4, dec_y + 0.4),
                   (x - 0.4, dec_y + 0.4), color=C_GREEN)

    ax.text(9.2, 3.05, "Decoder", ha="center", fontsize=11,
            color=C_GREEN, fontweight="bold")

    _arrow(ax, (ctx_x + 0.55, enc_y + 0.4),
           (dec_x[0] - 0.4, dec_y + 0.4), color=C_AMBER, lw=2.0)

    ax.set_title("Seq2Seq: encoder compresses input into a fixed vector $c$; decoder expands it",
                 fontsize=12, fontweight="bold", color=C_DARK, pad=10)
    ax.text(6.0, 0.05,
            "The fixed-size $c$ becomes a bottleneck for long inputs — the motivation for attention.",
            ha="center", fontsize=9.5, color=C_AMBER, style="italic")

    fig.tight_layout()
    save(fig, "fig6_seq2seq")


# ---------------------------------------------------------------------------
# Figure 7: Loss curves comparing RNN, LSTM, GRU
# ---------------------------------------------------------------------------
def fig7_loss_curves() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    epochs = np.arange(1, 51)
    rng = np.random.default_rng(7)

    # Synthetic curves: vanilla RNN plateaus (long-dependency task), LSTM/GRU converge
    rnn = 2.6 * np.exp(-epochs / 60) + 1.55 + 0.04 * rng.standard_normal(50)
    lstm = 2.6 * np.exp(-epochs / 14) + 0.45 + 0.03 * rng.standard_normal(50)
    gru = 2.6 * np.exp(-epochs / 11) + 0.50 + 0.03 * rng.standard_normal(50)

    ax1.plot(epochs, rnn, color=C_AMBER, lw=2.2, label="Vanilla RNN")
    ax1.plot(epochs, lstm, color=C_GREEN, lw=2.2, label="LSTM")
    ax1.plot(epochs, gru, color=C_BLUE, lw=2.2, label="GRU")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-entropy loss", fontsize=11)
    ax1.set_title("Training loss on a long-dependency task",
                  fontsize=12, fontweight="bold", color=C_DARK)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_ylim(0.3, 3.0)

    # Right: validation accuracy vs sequence length
    seq_lens = np.array([5, 10, 20, 50, 100, 200, 400])
    acc_rnn = np.clip(0.92 - 0.0018 * seq_lens, 0.30, 0.95) - 0.05 * np.log1p(seq_lens / 30)
    acc_lstm = np.clip(0.93 - 0.00035 * seq_lens, 0.55, 0.95)
    acc_gru = np.clip(0.92 - 0.00045 * seq_lens, 0.55, 0.94)

    ax2.plot(seq_lens, acc_rnn, marker="o", color=C_AMBER, lw=2.2, label="Vanilla RNN")
    ax2.plot(seq_lens, acc_lstm, marker="s", color=C_GREEN, lw=2.2, label="LSTM")
    ax2.plot(seq_lens, acc_gru, marker="^", color=C_BLUE, lw=2.2, label="GRU")
    ax2.set_xscale("log")
    ax2.set_xlabel("Sequence length (log scale)", fontsize=11)
    ax2.set_ylabel("Validation accuracy", fontsize=11)
    ax2.set_title("Accuracy vs. sequence length",
                  fontsize=12, fontweight="bold", color=C_DARK)
    ax2.legend(fontsize=10, loc="lower left")
    ax2.set_ylim(0.2, 1.0)

    fig.tight_layout()
    save(fig, "fig7_loss_curves")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Output (EN): {EN_DIR.relative_to(REPO_ROOT)}")
    print(f"Output (ZH): {ZH_DIR.relative_to(REPO_ROOT)}")
    print()

    print("Generating fig1_unrolled_rnn ...")
    fig1_unrolled_rnn()
    print("Generating fig2_vanishing_gradient ...")
    fig2_vanishing_gradient()
    print("Generating fig3_lstm_gates ...")
    fig3_lstm_gates()
    print("Generating fig4_gru_cell ...")
    fig4_gru_cell()
    print("Generating fig5_bidirectional_rnn ...")
    fig5_bidirectional_rnn()
    print("Generating fig6_seq2seq ...")
    fig6_seq2seq()
    print("Generating fig7_loss_curves ...")
    fig7_loss_curves()
    print("\nDone.")


if __name__ == "__main__":
    main()
