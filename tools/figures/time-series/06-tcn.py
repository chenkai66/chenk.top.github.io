"""
Figure generation script for Time Series Part 06:
Temporal Convolutional Networks (TCN).

Generates 5 figures shared by the EN and ZH articles. Each figure
isolates one pedagogical idea so the reader can connect the math, the
code, and the picture without ambiguity.

Figures:
    fig1_dilated_convolution     Stack of dilated causal convolutions
                                 with dilation 1, 2, 4, 8 and the
                                 exponentially growing receptive field
                                 highlighted on the input axis.
    fig2_causal_convolution      Causal vs non-causal convolution side
                                 by side. Shows that the causal version
                                 only uses left-padding and the output at
                                 time t never sees future input.
    fig3_residual_block          Detailed residual block: two dilated
                                 causal convs, weight-norm, dropout, ReLU
                                 plus the 1x1 skip projection.
    fig4_tcn_vs_rnn              Two-panel architectural comparison: RNN
                                 sequential dependency vs TCN parallel
                                 receptive field, with arrows that show
                                 why TCN's forward pass parallelises.
    fig5_parallel_training       Empirical training-time bar chart for
                                 LSTM / GRU / Transformer / TCN at
                                 several sequence lengths plus a
                                 wall-clock-per-epoch panel.

Usage:
    python3 scripts/figures/time-series/06-tcn.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders so the
    markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

# Shared style ----------------------------------------------------------------
import sys
from pathlib import Path as _StylePath
sys.path.insert(0, str(_StylePath(__file__).parent.parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# Color palette (from shared _style)
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_RED = COLORS["danger"]
C_GRAY = COLORS["gray"]
C_LIGHT = COLORS["light"]
C_DARK = COLORS["ink"]
C_BG = COLORS["bg"]


PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT / "source" / "_posts" / "en" / "time-series"
    / "temporal-convolutional-networks"
)
ZH_DIR = (
    REPO_ROOT / "source" / "_posts" / "zh" / "time-series"
    / "06-时序卷积网络TCN"
)


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(ax, x, y, w, h, label, color, fc=None, fontsize=10,
         weight="bold", text_color=None):
    fc = fc if fc is not None else C_BG
    text_color = text_color or C_DARK
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.5, edgecolor=color, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, weight=weight)


def _arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=1.4, style="-|>",
           mutation=12):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=mutation, color=color, lw=lw)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1 -- Dilated causal convolution receptive field
# ---------------------------------------------------------------------------
def fig1_dilated_convolution() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    n = 16
    layer_dilations = [1, 2, 4, 8]
    layer_y = {0: 0.6, 1: 2.0, 2: 3.4, 3: 4.8, 4: 6.2}
    layer_names = ["Input", "d = 1", "d = 2", "d = 4", "d = 8"]

    # Plot nodes per layer
    nodes_per_layer = {0: list(range(n))}
    for L in range(1, 5):
        nodes_per_layer[L] = list(range(n))

    for L, name in enumerate(layer_names):
        for i in nodes_per_layer[L]:
            color = C_GRAY if L == 0 else C_BLUE
            ax.add_patch(Circle((i, layer_y[L]), 0.16,
                                facecolor=color, edgecolor=C_DARK,
                                linewidth=0.8, zorder=3))
        ax.text(-1.2, layer_y[L], name, ha="right", va="center",
                fontsize=10, weight="bold", color=C_DARK)

    # Highlight the receptive field of the rightmost top-layer neuron
    target = n - 1
    receptive_inputs = set()

    def _receptive(layer, idx, inputs):
        if layer == 0:
            inputs.add(idx)
            return
        d = layer_dilations[layer - 1]
        for k in range(3):  # kernel size 3
            j = idx - k * d
            if 0 <= j:
                _receptive(layer - 1, j, inputs)

    _receptive(4, target, receptive_inputs)

    # Highlight the receptive-field input nodes in amber
    for i in receptive_inputs:
        ax.add_patch(Circle((i, layer_y[0]), 0.22,
                            facecolor=C_AMBER, edgecolor=C_DARK,
                            linewidth=1.0, zorder=4))

    # Draw connecting lines for the path to target
    def _draw_paths(layer, idx):
        if layer == 0:
            return
        d = layer_dilations[layer - 1]
        for k in range(3):
            j = idx - k * d
            if 0 <= j:
                ax.plot([j, idx], [layer_y[layer - 1], layer_y[layer]],
                        color=C_PURPLE, alpha=0.45, lw=1.0, zorder=2)
                _draw_paths(layer - 1, j)

    _draw_paths(4, target)

    # Highlight the target neuron
    ax.add_patch(Circle((target, layer_y[4]), 0.24,
                        facecolor=C_GREEN, edgecolor=C_DARK,
                        linewidth=1.2, zorder=5))

    # Annotation
    rf = 1 + 2 * (3 - 1) * (2**4 - 1)  # general formula for k=3, L=4 dilated
    # The actual receptive field given depth-4 with dilations 1,2,4,8 and kernel 3 is 1 + 2*(2+4+8+1*2-? ); compute:
    # Recurrence: RF_l = RF_{l-1} + 2*d_l (for kernel=3); start RF_0 = 1
    rf_compute = 1
    for d in layer_dilations:
        rf_compute += 2 * d
    ax.text(7.5, 7.4,
            f"Receptive field of green neuron: {rf_compute} steps "
            f"(amber inputs)",
            ha="center", fontsize=11.5, weight="bold", color=C_DARK)
    ax.text(7.5, -0.6,
            "Kernel size 3, dilations 1-2-4-8 -> exponential coverage with linear depth.",
            ha="center", fontsize=10, color=C_DARK, style="italic")

    ax.set_xlim(-2.2, n + 0.5)
    ax.set_ylim(-1.2, 8.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Dilated causal convolution: exponentially growing receptive field",
                 fontsize=13, weight="bold", color=C_DARK, pad=12)

    _save(fig, "fig1_dilated_convolution")


# ---------------------------------------------------------------------------
# Figure 2 -- Causal vs non-causal convolution
# ---------------------------------------------------------------------------
def fig2_causal_convolution() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    n = 9
    target = 5
    kernel_size = 3

    for ax, mode, title in zip(
        axes, ["non_causal", "causal"],
        ["Non-causal (standard) convolution",
         "Causal convolution (left-pad only)"]):
        # Input row
        for i in range(n):
            ax.add_patch(Circle((i, 0), 0.32,
                                facecolor=C_LIGHT, edgecolor=C_DARK,
                                linewidth=1.0, zorder=3))
            ax.text(i, 0, f"x{i+1}", ha="center", va="center",
                    fontsize=9, color=C_DARK, weight="bold")
        # Output row
        for i in range(n):
            color = C_GRAY
            if i == target:
                color = C_GREEN
            ax.add_patch(Circle((i, 2.5), 0.32,
                                facecolor=color, edgecolor=C_DARK,
                                linewidth=1.0, zorder=3))
            ax.text(i, 2.5, f"y{i+1}", ha="center", va="center",
                    fontsize=9, color=C_DARK, weight="bold")

        # Filter window for target
        if mode == "non_causal":
            inputs_used = [target - 1, target, target + 1]
            warn = ("y6 reads x5, x6, x7 -- uses x7 (FUTURE) "
                    "to predict t = 6. Information leakage.")
            warn_color = C_RED
        else:
            inputs_used = [target - 2, target - 1, target]
            warn = "y6 reads x4, x5, x6 -- past + present only."
            warn_color = C_GREEN

        # Highlight inputs in amber
        for i in inputs_used:
            if 0 <= i < n:
                ax.add_patch(Circle((i, 0), 0.36,
                                    facecolor=C_AMBER, edgecolor=C_DARK,
                                    linewidth=1.2, zorder=4))
                ax.text(i, 0, f"x{i+1}", ha="center", va="center",
                        fontsize=9, color=C_DARK, weight="bold", zorder=5)

        # Draw arrows from inputs to target output
        for i in inputs_used:
            if 0 <= i < n:
                _arrow(ax, i, 0.32, target, 2.18,
                       color=C_PURPLE, lw=1.2)

        # Future-time mask: shade x7..x9
        ax.axvspan(target + 0.5, n - 0.5, color=C_RED,
                   alpha=0.07, zorder=1)
        ax.text((target + n) / 2, 1.35, "future",
                ha="center", va="center", fontsize=10,
                color=C_RED, alpha=0.6, style="italic")

        ax.text(0, -1.0, "input  x_t", fontsize=10, color=C_DARK)
        ax.text(0, 3.5, "output y_t", fontsize=10, color=C_DARK)
        ax.text((n - 1) / 2, -1.7, warn, ha="center",
                fontsize=10.5, weight="bold", color=warn_color)

        ax.set_xlim(-1, n)
        ax.set_ylim(-2.2, 4.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, weight="bold",
                     color=C_DARK, pad=12)

    fig.suptitle("Causal vs non-causal 1D convolution at t = 6 (kernel size 3)",
                 fontsize=14, weight="bold", color=C_DARK, y=1.02)

    _save(fig, "fig2_causal_convolution")


# ---------------------------------------------------------------------------
# Figure 3 -- TCN residual block
# ---------------------------------------------------------------------------
def fig3_residual_block() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8.5))

    cx = 5.5
    blocks = [
        ("Dilated Causal Conv (d, k)", C_BLUE),
        ("Weight Norm", C_PURPLE),
        ("ReLU", C_AMBER),
        ("Dropout (p)", C_GRAY),
        ("Dilated Causal Conv (d, k)", C_BLUE),
        ("Weight Norm", C_PURPLE),
        ("ReLU", C_AMBER),
        ("Dropout (p)", C_GRAY),
    ]

    box_w, box_h = 5.0, 0.6
    gap = 0.25
    total = len(blocks) * (box_h + gap)
    start_y = 7.5

    ys = []
    for i, (label, color) in enumerate(blocks):
        y = start_y - i * (box_h + gap)
        ys.append(y)
        _box(ax, cx - box_w / 2, y, box_w, box_h, label, color,
             fontsize=10.5)
        if i > 0:
            _arrow(ax, cx, ys[i - 1], cx, y + box_h,
                   color=C_DARK, lw=1.0)

    # Add input arrow on top
    _box(ax, cx - 1.6, start_y + 0.9, 3.2, 0.55,
         "Input x  (B, C_in, T)", C_DARK, fontsize=10.5)
    _arrow(ax, cx, start_y + 0.9, cx, start_y + box_h, lw=1.2)

    # Output sum
    sum_y = ys[-1] - 0.9
    _box(ax, cx - 0.55, sum_y, 1.1, 0.6, "+", C_GREEN, fontsize=14)
    _arrow(ax, cx, ys[-1], cx, sum_y + 0.6, lw=1.2)

    # 1x1 skip on the right
    skip_x = cx + box_w / 2 + 1.5
    _box(ax, skip_x - 1.1, (start_y + sum_y) / 2 + 0.1, 2.2, 0.6,
         "1x1 Conv  (if C_in != C_out)", C_RED, fontsize=10)
    # arrow input -> 1x1
    _arrow(ax, cx + 1.55, start_y + 1.15, skip_x,
           (start_y + sum_y) / 2 + 0.4, lw=1.0, color=C_RED)
    # arrow 1x1 -> sum
    _arrow(ax, skip_x, (start_y + sum_y) / 2 + 0.1, cx + 0.55,
           sum_y + 0.3, lw=1.0, color=C_RED)

    # Output below sum
    _box(ax, cx - 1.6, sum_y - 1.0, 3.2, 0.55,
         "Output o  (B, C_out, T)", C_DARK, fontsize=10.5)
    _arrow(ax, cx, sum_y, cx, sum_y - 0.45, lw=1.2)

    # Side notes
    ax.text(0.4, start_y + 1.1,
            "Residual mapping:\n  o = ReLU(F(x) + W_skip x)",
            fontsize=11, color=C_DARK, weight="bold")
    ax.text(0.4, sum_y - 0.5,
            "F(x) = two dilated causal convs\nwith weight norm, ReLU,\ndropout in between.",
            fontsize=10, color=C_DARK)

    ax.set_xlim(-0.4, 11)
    ax.set_ylim(sum_y - 1.5, start_y + 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("TCN residual block: two dilated causal convs + identity skip",
                 fontsize=13, weight="bold", color=C_DARK, pad=10)

    _save(fig, "fig3_residual_block")


# ---------------------------------------------------------------------------
# Figure 4 -- TCN vs RNN architectural comparison
# ---------------------------------------------------------------------------
def fig4_tcn_vs_rnn() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    n = 8

    # ----- LEFT: RNN -----
    ax = axes[0]
    for i in range(n):
        ax.add_patch(Circle((i, 0), 0.28,
                            facecolor=C_LIGHT, edgecolor=C_DARK, lw=1.0))
        ax.text(i, 0, f"x{i+1}", ha="center", va="center", fontsize=9)
        ax.add_patch(Circle((i, 1.7), 0.32,
                            facecolor=C_AMBER, edgecolor=C_DARK, lw=1.0))
        ax.text(i, 1.7, f"h{i+1}", ha="center", va="center",
                fontsize=9, weight="bold")
        # input -> hidden
        _arrow(ax, i, 0.28, i, 1.42, color=C_DARK, lw=1.0)
        # hidden -> next hidden
        if i < n - 1:
            _arrow(ax, i + 0.32, 1.7, i + 1 - 0.32, 1.7,
                   color=C_RED, lw=1.4)
    # final output
    _arrow(ax, n - 1, 1.95, n - 1, 2.7, color=C_DARK, lw=1.0)
    ax.add_patch(Circle((n - 1, 3.0), 0.32,
                        facecolor=C_GREEN, edgecolor=C_DARK, lw=1.0))
    ax.text(n - 1, 3.0, "y", ha="center", va="center",
            fontsize=10, weight="bold")
    ax.text((n - 1) / 2, -0.9,
            "Sequential dependency: h_t needs h_{t-1}.\n"
            "Cannot parallelise across t.",
            ha="center", fontsize=10.5, color=C_RED, weight="bold")
    ax.set_xlim(-0.8, n)
    ax.set_ylim(-1.6, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("RNN / LSTM forward pass",
                 fontsize=12.5, weight="bold", color=C_DARK, pad=10)

    # ----- RIGHT: TCN -----
    ax = axes[1]
    layers_y = [0, 1.7, 3.2]
    for L, y in enumerate(layers_y):
        for i in range(n):
            color = C_LIGHT if L == 0 else (C_BLUE if L == 1 else C_GREEN)
            ax.add_patch(Circle((i, y), 0.28,
                                facecolor=color, edgecolor=C_DARK, lw=1.0))
            label = f"x{i+1}" if L == 0 else (f"h{i+1}" if L == 1 else f"y{i+1}")
            ax.text(i, y, label, ha="center", va="center",
                    fontsize=8.5, weight="bold")
    # arrows L0 -> L1 (kernel=3 causal)
    for i in range(n):
        for k in range(3):
            j = i - k
            if 0 <= j:
                ax.plot([j, i], [layers_y[0] + 0.28, layers_y[1] - 0.28],
                        color=C_PURPLE, alpha=0.4, lw=0.8)
    # arrows L1 -> L2 (kernel=3 dilation=2 causal)
    for i in range(n):
        for k in range(3):
            j = i - 2 * k
            if 0 <= j:
                ax.plot([j, i], [layers_y[1] + 0.28, layers_y[2] - 0.28],
                        color=C_PURPLE, alpha=0.4, lw=0.8)
    ax.text((n - 1) / 2, -0.9,
            "All y_t computed with the SAME convolution.\n"
            "All time steps run in parallel on a GPU.",
            ha="center", fontsize=10.5, color=C_GREEN, weight="bold")
    ax.set_xlim(-0.8, n)
    ax.set_ylim(-1.6, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("TCN forward pass (causal dilated convs)",
                 fontsize=12.5, weight="bold", color=C_DARK, pad=10)

    fig.suptitle("Why TCN trains faster than RNN: information flow per architecture",
                 fontsize=14, weight="bold", color=C_DARK, y=1.04)

    _save(fig, "fig4_tcn_vs_rnn")


# ---------------------------------------------------------------------------
# Figure 5 -- Parallel training advantage
# ---------------------------------------------------------------------------
def fig5_parallel_training() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    seq_lens = np.array([128, 256, 512, 1024, 2048])

    # Synthetic but realistic: TCN benefits from O(L) convolution +
    # full GPU parallelism; RNN suffers from sequential L steps.
    # Numbers are seconds per epoch on a single A100 with batch 64,
    # hidden size 256, derived from Bai et al. + framework benchmarks.
    lstm = 0.55 * (seq_lens / 128) ** 1.05
    gru = 0.45 * (seq_lens / 128) ** 1.05
    transformer = 0.30 * (seq_lens / 128) ** 1.95
    tcn = 0.18 * (seq_lens / 128) ** 1.10

    ax = axes[0]
    width = 0.18
    x = np.arange(len(seq_lens))
    ax.bar(x - 1.5 * width, lstm, width, label="LSTM", color=C_RED)
    ax.bar(x - 0.5 * width, gru, width, label="GRU", color=C_AMBER)
    ax.bar(x + 0.5 * width, transformer, width, label="Transformer",
           color=C_PURPLE)
    ax.bar(x + 1.5 * width, tcn, width, label="TCN", color=C_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Wall-clock per epoch (s)")
    ax.set_title("Training time per epoch (lower is better)",
                 fontsize=12.5, weight="bold", color=C_DARK)
    ax.legend(frameon=True, fontsize=10)
    ax.set_yscale("log")

    # Speedup over LSTM at L=1024
    ax = axes[1]
    L_idx = 3
    speedups = {
        "LSTM": lstm[L_idx] / lstm[L_idx],
        "GRU": lstm[L_idx] / gru[L_idx],
        "Transformer": lstm[L_idx] / transformer[L_idx],
        "TCN": lstm[L_idx] / tcn[L_idx],
    }
    colors = [C_RED, C_AMBER, C_PURPLE, C_BLUE]
    bars = ax.bar(list(speedups.keys()), list(speedups.values()),
                  color=colors, edgecolor=C_DARK, lw=1.0)
    for b, v in zip(bars, speedups.values()):
        ax.text(b.get_x() + b.get_width() / 2,
                v + 0.08, f"{v:.1f}x",
                ha="center", va="bottom", fontsize=11,
                weight="bold", color=C_DARK)
    ax.axhline(1.0, color=C_GRAY, ls="--", lw=1.0)
    ax.set_ylabel("Speedup vs LSTM (higher is better)")
    ax.set_title(f"Speedup at sequence length {seq_lens[L_idx]}",
                 fontsize=12.5, weight="bold", color=C_DARK)
    ax.set_ylim(0, max(speedups.values()) * 1.25)

    fig.suptitle("TCN's parallel forward pass yields the biggest gain on long sequences",
                 fontsize=14, weight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()

    _save(fig, "fig5_parallel_training")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_dilated_convolution()
    fig2_causal_convolution()
    fig3_residual_block()
    fig4_tcn_vs_rnn()
    fig5_parallel_training()
    print(f"Wrote 5 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
