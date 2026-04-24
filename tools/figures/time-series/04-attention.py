"""
Figures for Time Series Part 04: Attention Mechanism.

Generates seven pedagogical figures used by the EN and ZH posts:
  1. fig1_attention_heatmap.png        - Attention weight heatmap over past steps
  2. fig2_bahdanau_vs_luong.png        - Bahdanau (additive) vs Luong (dot-product) attention
  3. fig3_self_attention_ts.png        - Self-attention applied to a time series window
  4. fig4_complexity_vs_length.png     - Quadratic complexity vs sequence length
  5. fig5_attention_lstm_hybrid.png    - LSTM encoder + attention + decoder hybrid
  6. fig6_multihead_for_time.png       - Multi-head attention specialised for time
  7. fig7_stock_attention_app.png      - Stock-prediction case study with attention overlay

All figures saved to BOTH the EN and ZH asset folders.
Style: seaborn-v0_8-whitegrid, dpi=150, palette = {#2563eb, #7c3aed, #10b981, #f59e0b}.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# Shared style ----------------------------------------------------------------
import sys
from pathlib import Path as _StylePath
sys.path.insert(0, str(_StylePath(__file__).parent.parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Color palette (from shared _style)
BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
ORANGE = COLORS["warning"]
GREY = COLORS["gray"]
LIGHT = COLORS["light"]
DARK = COLORS["ink"]
RED = COLORS["danger"]


DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source/_posts/en/time-series/attention-mechanism"
ZH_DIR = REPO_ROOT / "source/_posts/zh/time-series/04-Attention机制"

# Custom blue heat-map (white -> blue)
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "blue_heat", ["#f8fafc", "#dbeafe", "#93c5fd", "#3b82f6", "#1e3a8a"]
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (EN_DIR, ZH_DIR):
        out = folder / name
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 -- Attention weight heatmap
# ---------------------------------------------------------------------------
def fig1_attention_heatmap() -> None:
    """Heatmap of attention weights (queries vs keys) showing periodic + recent focus."""
    rng = np.random.default_rng(7)
    T = 24  # 24 hourly steps

    # Build a synthetic but realistic attention pattern:
    # (1) recency bias - diagonal band
    # (2) daily periodicity - bright diagonal at offset 24 (wraps for visualisation)
    # (3) anomaly memory - column 5 stays bright (an unusual spike at t=5)
    weights = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            if j > i:  # causal: cannot attend to future
                continue
            recency = np.exp(-0.18 * (i - j))
            periodic = 0.6 * np.exp(-((i - j - 12) ** 2) / (2 * 1.6 ** 2))
            anomaly = 0.4 if j == 5 else 0.0
            weights[i, j] = recency + periodic + anomaly + 0.02 * rng.random()
    # Row-normalise (softmax-like)
    weights = weights / weights.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    im = ax.imshow(weights, cmap=BLUE_CMAP, aspect="auto", origin="lower")

    ax.set_xlabel("Key position (past time step)", fontsize=12)
    ax.set_ylabel("Query position (current time step)", fontsize=12)
    ax.set_title(
        "Attention weights over a 24-step window\n"
        "Recent steps + 12-step periodicity + persistent anomaly at t=5",
        fontsize=13, pad=12,
    )

    ax.set_xticks(range(0, T, 2))
    ax.set_yticks(range(0, T, 2))
    ax.set_xticklabels([f"t-{T-1-i}" if i < T - 1 else "now" for i in range(0, T, 2)],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"q={i}" for i in range(0, T, 2)], fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight", fontsize=11)

    # Annotate the three structural patterns
    ax.annotate("Diagonal:\nrecency", xy=(20, 20), xytext=(13, 22),
                fontsize=10, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.annotate("Off-diagonal:\nperiod 12", xy=(8, 20), xytext=(0.5, 22),
                fontsize=10, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.annotate("Vertical band:\npersistent\nanomaly memory",
                xy=(5, 14), xytext=(8, 6),
                fontsize=10, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))

    fig.tight_layout()
    save(fig, "fig1_attention_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 2 -- Bahdanau (additive) vs Luong (multiplicative) attention
# ---------------------------------------------------------------------------
def fig2_bahdanau_vs_luong() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def draw_panel(ax, title, score_formula, color, kind):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, color=color, fontweight="bold", pad=10)

        # encoder hidden states
        for i, x in enumerate(np.linspace(1.0, 5.5, 4)):
            box = FancyBboxPatch((x - 0.4, 4.6), 0.8, 0.8,
                                 boxstyle="round,pad=0.02",
                                 facecolor=BLUE, edgecolor=DARK, lw=1.0)
            ax.add_patch(box)
            ax.text(x, 5.0, f"$h_{i+1}$", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
        ax.text(3.25, 6.05, "Encoder hidden states", ha="center",
                fontsize=10, color=DARK)

        # decoder state
        ds = FancyBboxPatch((7.4, 4.6), 1.0, 0.8,
                            boxstyle="round,pad=0.02",
                            facecolor=PURPLE, edgecolor=DARK, lw=1.0)
        ax.add_patch(ds)
        ax.text(7.9, 5.0, "$s_t$", ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        ax.text(7.9, 6.05, "Decoder state", ha="center",
                fontsize=10, color=DARK)

        # score box
        sb = FancyBboxPatch((1.8, 2.4), 6.4, 1.0,
                            boxstyle="round,pad=0.05",
                            facecolor="white", edgecolor=color, lw=2.0)
        ax.add_patch(sb)
        ax.text(5.0, 2.9, score_formula, ha="center", va="center",
                fontsize=13, color=color)

        # arrows from encoder + decoder into score box
        for x in np.linspace(1.0, 5.5, 4):
            ax.annotate("", xy=(x, 3.45), xytext=(x, 4.55),
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
        ax.annotate("", xy=(7.5, 3.45), xytext=(7.9, 4.55),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))

        # context output
        cb = FancyBboxPatch((3.8, 0.4), 2.4, 0.9,
                            boxstyle="round,pad=0.02",
                            facecolor=GREEN, edgecolor=DARK, lw=1.0)
        ax.add_patch(cb)
        ax.text(5.0, 0.85, "context $c_t$", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        ax.annotate("", xy=(5.0, 1.35), xytext=(5.0, 2.35),
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5))
        ax.text(5.4, 1.85, "softmax $\\alpha$ then $\\sum \\alpha_i h_i$",
                ha="left", va="center", fontsize=9, color=DARK, style="italic")

        # property strip
        props = {
            "additive": ["Score: small MLP (tanh + projection)",
                         "Slower per step but flexible",
                         "Pre-Transformer, ICLR 2015",
                         "Good when $d$ is large or$Q,K$ live in different spaces"],
            "mult":     ["Score: simple dot product",
                         "Faster (single matmul)",
                         "Used in Transformers (with $1/\\sqrt{d}$)",
                         "Assumes$Q,K$ share the same space"],
        }[kind]
        for k, p in enumerate(props):
            ax.text(0.2, -0.4 - 0.55 * k, "- " + p, ha="left",
                    fontsize=9, color=DARK)

    draw_panel(axes[0], "Bahdanau attention (additive, 2015)",
               "$e_{ti} = v^\\top \\tanh(W_1 h_i + W_2 s_{t-1})$",
               PURPLE, "additive")
    draw_panel(axes[1], "Luong attention (multiplicative, 2015)",
               "$e_{ti} = s_t^\\top W h_i$  (or simply $s_t^\\top h_i$)",
               BLUE, "mult")

    fig.suptitle(
        "Two classic attention scoring functions",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig2_bahdanau_vs_luong.png")


# ---------------------------------------------------------------------------
# Figure 3 -- Self-attention applied to a time series
# ---------------------------------------------------------------------------
def fig3_self_attention_ts() -> None:
    rng = np.random.default_rng(2)
    T = 12
    t = np.arange(T)
    series = (
        np.sin(2 * np.pi * t / 6) * 0.6
        + 0.05 * t
        + 0.12 * rng.standard_normal(T)
    )

    # synthesise an attention row for the last query (q = T-1)
    raw = np.zeros(T)
    for j in range(T):
        recency = np.exp(-0.35 * (T - 1 - j))
        periodic = 0.8 * np.exp(-((T - 1 - j - 6) ** 2) / 1.5)
        raw[j] = recency + periodic
    weights = raw / raw.sum()

    fig = plt.figure(figsize=(13.5, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.45)

    # Top: time series with arcs from query into past
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, series, "-o", color=BLUE, lw=2, markersize=7,
             markerfacecolor="white", markeredgewidth=2, label="$x_t$")
    # highlight the query
    ax1.scatter([T - 1], [series[-1]], s=180, color=ORANGE,
                zorder=5, edgecolor=DARK, lw=1.5, label="query (now)")

    # arcs from query position back to each key position with thickness ~ weight
    for j in range(T - 1):
        w = weights[j]
        if w < 0.005:
            continue
        x0, y0 = T - 1, series[-1]
        x1, y1 = j, series[j]
        height = 0.9 + 0.7 * abs(x0 - x1) / T
        ctrl_x = (x0 + x1) / 2
        ctrl_y = max(y0, y1) + height
        # quadratic Bezier via three points
        ts_arr = np.linspace(0, 1, 60)
        bx = (1 - ts_arr) ** 2 * x0 + 2 * (1 - ts_arr) * ts_arr * ctrl_x + ts_arr ** 2 * x1
        by = (1 - ts_arr) ** 2 * y0 + 2 * (1 - ts_arr) * ts_arr * ctrl_y + ts_arr ** 2 * y1
        ax1.plot(bx, by, color=PURPLE, lw=0.6 + 6.0 * w, alpha=0.55)

    ax1.set_title(
        "Self-attention from the most recent step into the past\n"
        "Arc thickness = attention weight (recency + period-6 peak)",
        fontsize=13, pad=10,
    )
    ax1.set_xlabel("Time step$t$")
    ax1.set_ylabel("$x_t$")
    ax1.set_xticks(range(T))
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_ylim(series.min() - 0.5, series.max() + 2.4)

    # Bottom: bar chart of weights
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(t, weights, color=PURPLE, edgecolor=DARK, lw=0.8)
    bars[-1].set_color(ORANGE)
    bars[T - 1 - 6].set_color(GREEN)
    ax2.set_title("Attention weight$\\alpha_{T-1, j}$", fontsize=12)
    ax2.set_xlabel("Key position$j$")
    ax2.set_ylabel("$\\alpha$")
    ax2.set_xticks(range(T))

    legend_handles = [
        mpatch_color(ORANGE, "Self-attention to itself"),
        mpatch_color(GREEN,  "Period-6 peak"),
        mpatch_color(PURPLE, "Other history"),
    ]
    ax2.legend(handles=legend_handles, fontsize=9, loc="upper left")

    save(fig, "fig3_self_attention_ts.png")


def mpatch_color(color, label):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, edgecolor=DARK, label=label)


# ---------------------------------------------------------------------------
# Figure 4 -- Complexity vs sequence length
# ---------------------------------------------------------------------------
def fig4_complexity_vs_length() -> None:
    n = np.arange(16, 4097, 16)
    d = 64

    rnn_time = n * d ** 2  # O(n d^2) sequential
    attn_time = n ** 2 * d  # O(n^2 d)
    sparse_time = n * np.log2(n) * d  # O(n log n d)
    linear_attn = n * d ** 2  # O(n d^2)

    rnn_mem = n * d
    attn_mem = n ** 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    ax.plot(n, rnn_time, color=BLUE, lw=2.2, label="RNN / LSTM   $O(n \\cdot d^2)$")
    ax.plot(n, attn_time, color=ORANGE, lw=2.4, label="Full attention   $O(n^2 \\cdot d)$")
    ax.plot(n, sparse_time, color=GREEN, lw=2.0, ls="--",
            label="Sparse attention   $O(n \\log n \\cdot d)$")
    ax.plot(n, linear_attn, color=PURPLE, lw=2.0, ls=":",
            label="Linear attention   $O(n \\cdot d^2)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length$n$")
    ax.set_ylabel("FLOPs (relative)")
    ax.set_title("Compute scaling with sequence length", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")

    # Mark cross-over where attention becomes more expensive than RNN
    crossover = d  # n^2 d > n d^2  when  n > d
    ax.axvline(crossover, color=DARK, lw=1.0, ls=":")
    ax.text(crossover * 1.1, rnn_time[len(rnn_time) // 2],
            f"crossover\n$n = d = {d}$", fontsize=9, color=DARK)

    ax = axes[1]
    ax.plot(n, rnn_mem, color=BLUE, lw=2.2, label="RNN hidden state   $O(n \\cdot d)$")
    ax.plot(n, attn_mem, color=ORANGE, lw=2.4,
            label="Attention matrix   $O(n^2)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length$n$")
    ax.set_ylabel("Memory (relative)")
    ax.set_title("Memory scaling: the $O(n^2)$ wall", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")

    # Annotate practical limits
    for length, label in [(512, "BERT"), (2048, "GPT-2"), (4096, "T5-long")]:
        ax.axvline(length, color=GREY, lw=0.7, ls=":")
        ax.text(length, attn_mem.max() * 0.6, label,
                fontsize=8, color=GREY, rotation=90, va="top", ha="right")

    fig.suptitle(
        "Why attention scales quadratically and what bypasses it",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig4_complexity_vs_length.png")


# ---------------------------------------------------------------------------
# Figure 5 -- Attention + LSTM hybrid
# ---------------------------------------------------------------------------
def fig5_attention_lstm_hybrid() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Encoder LSTM cells
    enc_xs = np.linspace(1.0, 6.0, 5)
    for i, x in enumerate(enc_xs):
        # input
        inp = FancyBboxPatch((x - 0.45, 0.6), 0.9, 0.7,
                             boxstyle="round,pad=0.02",
                             facecolor="white", edgecolor=DARK, lw=1.0)
        ax.add_patch(inp)
        ax.text(x, 0.95, f"$x_{i+1}$", ha="center", va="center",
                fontsize=10, color=DARK)

        # LSTM cell
        cell = FancyBboxPatch((x - 0.5, 2.0), 1.0, 0.95,
                              boxstyle="round,pad=0.03",
                              facecolor=BLUE, edgecolor=DARK, lw=1.1)
        ax.add_patch(cell)
        ax.text(x, 2.45, "LSTM", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

        # hidden state output
        h = FancyBboxPatch((x - 0.45, 3.6), 0.9, 0.7,
                           boxstyle="round,pad=0.02",
                           facecolor=LIGHT, edgecolor=DARK, lw=1.0)
        ax.add_patch(h)
        ax.text(x, 3.95, f"$h_{i+1}$", ha="center", va="center",
                fontsize=10, color=DARK)

        ax.annotate("", xy=(x, 1.95), xytext=(x, 1.35),
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0))
        ax.annotate("", xy=(x, 3.55), xytext=(x, 2.95),
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0))
        if i > 0:
            ax.annotate("", xy=(x - 0.5, 2.45), xytext=(enc_xs[i-1] + 0.5, 2.45),
                        arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0))

    ax.text(np.mean(enc_xs), 5.0, "Encoder LSTM (forward in time)",
            ha="center", fontsize=11, color=BLUE, fontweight="bold")

    # Attention block
    attn = FancyBboxPatch((7.5, 3.4), 2.6, 1.6,
                          boxstyle="round,pad=0.05",
                          facecolor=ORANGE, edgecolor=DARK, lw=1.4, alpha=0.85)
    ax.add_patch(attn)
    ax.text(8.8, 4.2, "Attention\n$\\alpha_{ti} = \\text{softmax}(\\text{score}(s_{t-1}, h_i))$",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    # arrows from each h_i to attention
    for i, x in enumerate(enc_xs):
        ax.annotate("", xy=(7.55, 4.0), xytext=(x + 0.4, 4.0),
                    arrowprops=dict(arrowstyle="->", color=GREY,
                                    lw=0.8, alpha=0.6,
                                    connectionstyle="arc3,rad=0.18"))

    # context vector
    ctx = FancyBboxPatch((10.5, 3.6), 1.4, 0.9,
                         boxstyle="round,pad=0.03",
                         facecolor=GREEN, edgecolor=DARK, lw=1.1)
    ax.add_patch(ctx)
    ax.text(11.2, 4.05, "$c_t = \\sum \\alpha_{ti} h_i$",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.annotate("", xy=(10.45, 4.05), xytext=(10.15, 4.2),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))

    # Decoder LSTM cell
    dec = FancyBboxPatch((11.0, 6.0), 1.6, 1.0,
                         boxstyle="round,pad=0.03",
                         facecolor=PURPLE, edgecolor=DARK, lw=1.1)
    ax.add_patch(dec)
    ax.text(11.8, 6.5, "Decoder\nLSTM", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(11.8, 5.95), xytext=(11.2, 4.55),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))

    # Predicted output
    pred = FancyBboxPatch((13.0, 6.05), 0.85, 0.95,
                          boxstyle="round,pad=0.02",
                          facecolor="white", edgecolor=DARK, lw=1.0)
    ax.add_patch(pred)
    ax.text(13.42, 6.5, "$\\hat{y}_t$", ha="center", va="center",
            fontsize=11, color=DARK)
    ax.annotate("", xy=(12.95, 6.5), xytext=(12.6, 6.5),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))

    # caption
    ax.text(7.0, 7.6,
            "Attention + LSTM hybrid: LSTM extracts local temporal features,\n"
            "attention selects which step matters most for each forecast step",
            ha="center", fontsize=12, color=DARK)

    save(fig, "fig5_attention_lstm_hybrid.png")


# ---------------------------------------------------------------------------
# Figure 6 -- Multi-head attention adapted for time
# ---------------------------------------------------------------------------
def fig6_multihead_for_time() -> None:
    """Four heads, each capturing a different temporal pattern."""
    rng = np.random.default_rng(11)
    T = 18

    def make_head(kind):
        w = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                if j > i:
                    continue
                if kind == "local":
                    w[i, j] = np.exp(-0.5 * (i - j))
                elif kind == "long":
                    w[i, j] = np.exp(-0.06 * (i - j))
                elif kind == "periodic":
                    w[i, j] = (
                        np.exp(-0.4 * (i - j))
                        + 1.2 * np.exp(-((i - j - 7) ** 2) / 1.4)
                    )
                elif kind == "anomaly":
                    base = np.exp(-0.3 * (i - j))
                    if j == 4:  # anomaly at t=4
                        base += 0.9
                    w[i, j] = base
                w[i, j] += 0.01 * rng.random()
        return w / w.sum(axis=1, keepdims=True)

    heads = [
        ("Head 1: Local (recency)", make_head("local"), BLUE),
        ("Head 2: Long-range trend", make_head("long"), PURPLE),
        ("Head 3: Periodic (lag 7)", make_head("periodic"), GREEN),
        ("Head 4: Anomaly memory (t=4)", make_head("anomaly"), ORANGE),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, (title, w, col) in zip(axes, heads):
        im = ax.imshow(w, cmap=BLUE_CMAP, aspect="auto", origin="lower")
        ax.set_title(title, fontsize=11, color=col, fontweight="bold")
        ax.set_xlabel("Key$j$", fontsize=10)
        ax.set_xticks([0, T // 2, T - 1])
        ax.set_yticks([0, T // 2, T - 1])
        if ax is axes[0]:
            ax.set_ylabel("Query$i$", fontsize=10)

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("Attention weight", fontsize=10)

    fig.suptitle(
        "Multi-head attention specialised for time series\n"
        "Different heads learn local, long-range, periodic, and anomaly patterns",
        fontsize=13, y=1.03,
    )
    save(fig, "fig6_multihead_for_time.png")


# ---------------------------------------------------------------------------
# Figure 7 -- Stock prediction case study with attention overlay
# ---------------------------------------------------------------------------
def fig7_stock_attention_app() -> None:
    rng = np.random.default_rng(42)
    T = 90
    t = np.arange(T)

    # Synthetic price = trend + weekly cycle + monthly cycle + noise + earnings spike
    price = (
        100.0
        + 0.18 * t
        + 1.6 * np.sin(2 * np.pi * t / 7)
        + 4.0 * np.sin(2 * np.pi * t / 30)
        + 0.9 * rng.standard_normal(T).cumsum() / np.sqrt(T)
    )
    earnings_day = 60
    price[earnings_day:] += 6.0  # earnings beat
    price[earnings_day] += 4.0   # spike

    # Forecast horizon
    horizon = 10
    future_t = np.arange(T, T + horizon)

    # Two predictions: with vs without attention
    no_attn = price[-1] + 0.18 * (future_t - T + 1) + 0.4 * rng.standard_normal(horizon)
    no_attn = no_attn - 1.5  # under-shoots the regime change
    with_attn = (
        price[-1]
        + 0.22 * (future_t - T + 1)
        + 1.3 * np.sin(2 * np.pi * future_t / 30)
        + 0.3 * rng.standard_normal(horizon)
    )

    # Synthetic attention weights from "now" back into the 30 most recent days
    look_back = 30
    raw = np.zeros(look_back)
    for k in range(look_back):
        days_ago = look_back - k  # so k=0 is 30 days ago, k=look_back-1 is yesterday
        recency = np.exp(-0.07 * days_ago)
        earnings_focus = 0.9 if (T - days_ago) == earnings_day else 0.0
        cycle_focus = 0.4 * np.exp(-((days_ago - 30) ** 2) / 4.0)
        raw[k] = recency + earnings_focus + cycle_focus
    weights = raw / raw.sum()

    fig = plt.figure(figsize=(14.5, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.18)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, price, color=BLUE, lw=2, label="Historical price")
    ax1.plot(future_t, no_attn, color=GREY, lw=2, ls="--",
             marker="o", markersize=5, label="No-attention baseline")
    ax1.plot(future_t, with_attn, color=ORANGE, lw=2.4,
             marker="o", markersize=5, label="LSTM + attention")
    ax1.axvline(T - 0.5, color=DARK, ls=":", lw=1)
    ax1.text(T - 0.5, ax1.get_ylim()[1] * 0.99, " forecast horizon",
             ha="left", va="top", color=DARK, fontsize=10)

    # mark earnings event
    ax1.scatter([earnings_day], [price[earnings_day]],
                s=160, color=RED, zorder=5, edgecolor=DARK, lw=1.5)
    ax1.annotate("Earnings beat", xy=(earnings_day, price[earnings_day]),
                 xytext=(earnings_day - 18, price[earnings_day] + 4),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                 color=RED, fontsize=10, fontweight="bold")

    ax1.set_title(
        "Stock-price forecast with attention overlay\n"
        "Attention learns to focus on the recent earnings event and the 30-day cycle",
        fontsize=13, pad=8,
    )
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    bar_t = np.arange(T - look_back, T)
    bars = ax2.bar(bar_t, weights, color=PURPLE, edgecolor=DARK, lw=0.5,
                   width=0.85)
    # highlight earnings-day bar
    ed_idx = np.where(bar_t == earnings_day)[0]
    if len(ed_idx) > 0:
        bars[ed_idx[0]].set_color(RED)
    ax2.set_ylabel("Attention\nweight", fontsize=10)
    ax2.set_xlabel("Day index")
    ax2.set_title("Weights from the forecast query into the past 30 days",
                  fontsize=11)
    ax2.set_xlim(0, T + horizon)

    save(fig, "fig7_stock_attention_app.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Time Series Part 04: Attention Mechanism")
    fig1_attention_heatmap()
    fig2_bahdanau_vs_luong()
    fig3_self_attention_ts()
    fig4_complexity_vs_length()
    fig5_attention_lstm_hybrid()
    fig6_multihead_for_time()
    fig7_stock_attention_app()
    print("Done.")


if __name__ == "__main__":
    main()
