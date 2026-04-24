"""
Figure generation script for Time Series Part 05: Transformer Architecture.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure teaches one specific idea cleanly, with no clutter.

Figures:
    fig1_architecture            Encoder-decoder Transformer adapted for
                                 time series: input embedding, temporal
                                 positional encoding, encoder stack,
                                 decoder stack with causal mask, output
                                 projection.
    fig2_positional_encoding     Sinusoidal positional encoding heatmap +
                                 four representative dimensions, showing
                                 how each position gets a unique frequency
                                 signature.
    fig3_multihead_patterns      Four attention heads over a time series
                                 learning different patterns: local,
                                 periodic (24h), long-range, and a
                                 specific past-event head.
    fig4_lstm_vs_transformer     Forecast quality on a long-context signal:
                                 ground truth vs. LSTM vs. Transformer,
                                 with an MAE bar inset.
    fig5_quadratic_bottleneck    Vanilla self-attention is O(n^2): memory
                                 in GB and FLOPs vs. sequence length, with
                                 sparse / linear / patched alternatives.
    fig6_decoder_only_forecast   Decoder-only autoregressive forecasting:
                                 history rolled forward step by step with
                                 a causal mask.
    fig7_patching                Patching strategy (PatchTST-style): split
                                 a long series into non-overlapping
                                 patches, embed each patch as one token,
                                 huge sequence-length reduction.

Usage:
    python3 scripts/figures/time-series/05-transformer.py

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
C_RED = "#ef4444"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "time-series" / "transformer"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "time-series" / "05-Transformer架构"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, x, y, w, h, text, *, fc, ec=None, fontsize=10, color="white",
         weight="bold"):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.5,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, weight=weight)


def _arrow(ax, x0, y0, x1, y1, color=None, lw=1.5):
    color = color or C_DARK
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
        color=color, lw=lw,
    ))


# ---------------------------------------------------------------------------
# Figure 1: Transformer architecture for time series
# ---------------------------------------------------------------------------
def fig1_architecture() -> None:
    """Encoder-decoder Transformer schematic adapted for time series."""
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title(
        "Transformer for time series: encoder context + decoder forecast",
        fontsize=14, color=C_DARK, pad=10,
    )

    # ---- Encoder column ----
    ex = 1.0
    _box(ax, ex, 0.6, 3.4, 0.7, "History x_{t-L+1 ... t}",
         fc=C_LIGHT, color=C_DARK)
    _box(ax, ex, 1.7, 3.4, 0.7, "Linear input projection (d_model)",
         fc=C_BLUE)
    _box(ax, ex, 2.8, 3.4, 0.7, "+ Temporal positional encoding",
         fc=C_PURPLE)
    # Encoder stack box
    ax.add_patch(FancyBboxPatch(
        (ex, 4.0), 3.4, 3.6, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="white", edgecolor=C_BLUE, linewidth=2,
    ))
    ax.text(ex + 1.7, 7.4, "Encoder x N", ha="center", va="center",
            fontsize=11, weight="bold", color=C_BLUE)
    _box(ax, ex + 0.2, 5.95, 3.0, 0.55, "Multi-head self-attention",
         fc=C_BLUE, fontsize=9)
    _box(ax, ex + 0.2, 5.25, 3.0, 0.55, "Add and LayerNorm",
         fc=C_GRAY, fontsize=9)
    _box(ax, ex + 0.2, 4.55, 3.0, 0.55, "Feed-forward network",
         fc=C_GREEN, fontsize=9)
    _box(ax, ex + 0.2, 3.95, 3.0, 0.45, "Add and LayerNorm",
         fc=C_GRAY, fontsize=9)

    # arrows in encoder column
    for y0, y1 in [(1.3, 1.7), (2.4, 2.8), (3.5, 4.0)]:
        _arrow(ax, ex + 1.7, y0, ex + 1.7, y1)

    # ---- Decoder column ----
    dx = 9.0
    _box(ax, dx, 0.6, 3.4, 0.7, "Decoder input (label + zeros)",
         fc=C_LIGHT, color=C_DARK)
    _box(ax, dx, 1.7, 3.4, 0.7, "Linear input projection (d_model)",
         fc=C_BLUE)
    _box(ax, dx, 2.8, 3.4, 0.7, "+ Temporal positional encoding",
         fc=C_PURPLE)

    ax.add_patch(FancyBboxPatch(
        (dx, 4.0), 3.4, 3.6, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="white", edgecolor=C_AMBER, linewidth=2,
    ))
    ax.text(dx + 1.7, 7.4, "Decoder x M", ha="center", va="center",
            fontsize=11, weight="bold", color=C_AMBER)
    _box(ax, dx + 0.2, 5.95, 3.0, 0.55, "Masked self-attention (causal)",
         fc=C_AMBER, fontsize=9)
    _box(ax, dx + 0.2, 5.25, 3.0, 0.55, "Cross-attention (Q from dec, KV from enc)",
         fc=C_PURPLE, fontsize=8)
    _box(ax, dx + 0.2, 4.55, 3.0, 0.55, "Feed-forward network",
         fc=C_GREEN, fontsize=9)
    _box(ax, dx + 0.2, 3.95, 3.0, 0.45, "Add and LayerNorm",
         fc=C_GRAY, fontsize=9)

    for y0, y1 in [(1.3, 1.7), (2.4, 2.8), (3.5, 4.0)]:
        _arrow(ax, dx + 1.7, y0, dx + 1.7, y1)

    # output
    _box(ax, dx, 7.95, 3.4, 0.55, "Linear -> y_{t+1 ... t+H}",
         fc=C_DARK, fontsize=10)
    _arrow(ax, dx + 1.7, 7.6, dx + 1.7, 7.95)

    # encoder->decoder cross-attention arrow
    _arrow(ax, ex + 3.4, 5.5, dx, 5.5, color=C_PURPLE, lw=2.0)
    ax.text((ex + 3.4 + dx) / 2, 5.85, "memory K, V",
            ha="center", va="center", fontsize=10, color=C_PURPLE,
            weight="bold")

    # encoder output arrow up-out
    _arrow(ax, ex + 1.7, 7.6, ex + 1.7, 8.05)
    ax.text(ex + 1.7, 8.3, "context vectors", ha="center", va="center",
            fontsize=9, color=C_BLUE)

    # caption strip
    ax.text(7.0, 0.1,
            "Encoder reads the lookback window in parallel; decoder generates "
            "the forecast horizon, attending to encoder memory.",
            ha="center", va="bottom", fontsize=9.5, color=C_DARK,
            style="italic")

    fig.tight_layout()
    _save(fig, "fig1_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Sinusoidal positional encoding
# ---------------------------------------------------------------------------
def fig2_positional_encoding() -> None:
    """Sinusoidal positional encoding heatmap + per-dimension waveforms."""
    seq_len = 96      # one day of hourly data + a bit
    d_model = 64

    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    div = np.exp((i // 2) * 2 * (-np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos * div[:, 0::2])
    pe[:, 1::2] = np.cos(pos * div[:, 0::2])

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)

    # heatmap
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                    interpolation="nearest")
    ax0.set_xlabel("position (time step)", fontsize=11, color=C_DARK)
    ax0.set_ylabel("encoding dimension", fontsize=11, color=C_DARK)
    ax0.set_title("Sinusoidal positional encoding: each row is a unique signature",
                  fontsize=12, color=C_DARK, pad=8)
    ax0.grid(False)
    cbar = plt.colorbar(im, ax=ax0, fraction=0.04, pad=0.02)
    cbar.set_label("value", fontsize=9)

    # waveforms
    ax1 = fig.add_subplot(gs[1])
    dims = [0, 8, 24, 48]
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    for d, c in zip(dims, colors):
        ax1.plot(pe[:, d], color=c, lw=1.6, label=f"dim {d}")
    ax1.set_xlabel("position (time step)", fontsize=11, color=C_DARK)
    ax1.set_ylabel("encoding value", fontsize=11, color=C_DARK)
    ax1.set_title("Different dimensions = different frequencies",
                  fontsize=12, color=C_DARK, pad=8)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.set_ylim(-1.15, 1.15)

    fig.suptitle("Position is injected via fixed sinusoids (no parameters required)",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_positional_encoding")


# ---------------------------------------------------------------------------
# Figure 3: Multi-head attention patterns over a time series
# ---------------------------------------------------------------------------
def fig3_multihead_patterns() -> None:
    """Four attention heads, each learning a different temporal pattern."""
    n = 48  # query and key positions
    rng = np.random.default_rng(7)

    def softmax_rows(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    # Head 1: local attention (banded around diagonal)
    qi, kj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    H1 = softmax_rows(-0.5 * (qi - kj) ** 2 + 0.02 * rng.normal(size=(n, n)))

    # Head 2: periodic with period 24 (daily cycle)
    H2 = softmax_rows(
        2.5 * np.cos(2 * np.pi * (qi - kj) / 24)
        - 0.0005 * (qi - kj) ** 2  # mild decay
        + 0.05 * rng.normal(size=(n, n))
    )

    # Head 3: long-range (attend to faraway, with mild diag suppression)
    H3 = softmax_rows(
        0.04 * np.abs(qi - kj) - 1.5 * np.exp(-0.05 * (qi - kj) ** 2)
        + 0.05 * rng.normal(size=(n, n))
    )

    # Head 4: anchored on a specific past event around k=10
    H4 = softmax_rows(
        -0.06 * (kj - 10) ** 2 - 0.02 * (qi - kj) ** 2
        + 0.05 * rng.normal(size=(n, n))
    )

    # apply causal mask (cannot attend to future)
    mask = qi < kj
    for H in (H1, H2, H3, H4):
        H[mask] = 0.0
        H /= H.sum(axis=1, keepdims=True) + 1e-12

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    titles = [
        "Head 1: local (recent steps)",
        "Head 2: periodic (24-step cycle)",
        "Head 3: long-range",
        "Head 4: anchored on past event",
    ]
    for ax, H, t in zip(axes, [H1, H2, H3, H4], titles):
        im = ax.imshow(H, cmap="Blues", aspect="auto", vmin=0,
                       vmax=H.max() * 0.85)
        ax.set_title(t, fontsize=11, color=C_DARK, pad=6)
        ax.set_xlabel("key position (past)", fontsize=10)
        ax.grid(False)
    axes[0].set_ylabel("query position (now)", fontsize=10)

    fig.suptitle(
        "Multi-head attention: each head specialises on a different temporal pattern",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig3_multihead_patterns")


# ---------------------------------------------------------------------------
# Figure 4: Transformer vs LSTM forecast quality
# ---------------------------------------------------------------------------
def fig4_lstm_vs_transformer() -> None:
    """Long-context signal: ground truth vs. LSTM vs. Transformer forecasts."""
    rng = np.random.default_rng(11)
    n_hist, n_fcst = 240, 96
    n = n_hist + n_fcst
    t = np.arange(n)

    # ground truth: weekly + daily seasonality + slow trend + spikes
    daily = np.sin(2 * np.pi * t / 24)
    weekly = 0.6 * np.sin(2 * np.pi * t / (24 * 7))
    trend = 0.0015 * t
    spikes = np.zeros(n)
    for c in [40, 110, 175, 235, 290]:
        spikes[c:c + 4] += rng.uniform(0.6, 1.0)
    truth = daily + weekly + trend + 0.18 * rng.normal(size=n) + spikes

    hist = truth[:n_hist]
    fut = truth[n_hist:]
    tt = t[n_hist:]

    # LSTM forecast: tracks daily but drifts and misses long-range structure
    lstm_pred = (
        0.85 * np.sin(2 * np.pi * tt / 24)
        + 0.30 * np.sin(2 * np.pi * tt / (24 * 7) + 0.4)
        + 0.0011 * tt
        + 0.12 * rng.normal(size=n_fcst)
        + 0.08 * np.sin(2 * np.pi * tt / 9)  # spurious component
    )
    # Transformer forecast: closer to truth, captures both seasonalities
    trans_pred = (
        0.98 * np.sin(2 * np.pi * tt / 24)
        + 0.55 * np.sin(2 * np.pi * tt / (24 * 7))
        + 0.0014 * tt
        + 0.06 * rng.normal(size=n_fcst)
    )

    mae_lstm = np.mean(np.abs(lstm_pred - fut))
    mae_trans = np.mean(np.abs(trans_pred - fut))

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.18)

    ax = fig.add_subplot(gs[0])
    ax.plot(t[:n_hist], hist, color=C_GRAY, lw=1.2, label="history")
    ax.plot(tt, fut, color=C_DARK, lw=1.6, label="ground truth")
    ax.plot(tt, lstm_pred, color=C_AMBER, lw=1.4, ls="--", label=f"LSTM (MAE={mae_lstm:.2f})")
    ax.plot(tt, trans_pred, color=C_BLUE, lw=1.4, label=f"Transformer (MAE={mae_trans:.2f})")
    ax.axvline(n_hist, color=C_GRAY, lw=1, ls=":")
    ax.text(n_hist, ax.get_ylim()[1] * 0.92, " forecast horizon -> ",
            ha="left", va="top", fontsize=9, color=C_GRAY)
    ax.set_xlabel("time step (hours)", fontsize=11, color=C_DARK)
    ax.set_ylabel("value", fontsize=11, color=C_DARK)
    ax.set_title("Forecast on a signal with daily + weekly seasonality",
                 fontsize=12, color=C_DARK, pad=8)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    # MAE bar inset
    ax2 = fig.add_subplot(gs[1])
    models = ["LSTM", "GRU", "Transformer", "Autoformer"]
    mae = [mae_lstm, mae_lstm * 0.96, mae_trans, mae_trans * 0.88]
    colors = [C_AMBER, C_GREEN, C_BLUE, C_PURPLE]
    bars = ax2.bar(models, mae, color=colors, edgecolor="white", lw=1.2)
    for b, v in zip(bars, mae):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                 color=C_DARK)
    ax2.set_ylabel("Mean absolute error", fontsize=10)
    ax2.set_title("Forecast MAE (lower is better)", fontsize=11,
                  color=C_DARK, pad=8)
    ax2.set_ylim(0, max(mae) * 1.25)
    ax2.tick_params(axis="x", labelsize=9, rotation=15)

    fig.tight_layout()
    _save(fig, "fig4_lstm_vs_transformer")


# ---------------------------------------------------------------------------
# Figure 5: Quadratic attention bottleneck
# ---------------------------------------------------------------------------
def fig5_quadratic_bottleneck() -> None:
    """Memory and FLOPs vs. sequence length, with O(n^2) bottleneck."""
    n = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    d_model = 512
    n_heads = 8
    bytes_per = 2  # fp16

    # Vanilla attention: stores n*n attention matrix per head per layer.
    # Per-layer attention memory ~ n_heads * n^2 * bytes
    attn_mem_gb = n_heads * (n.astype(float) ** 2) * bytes_per / (1024 ** 3)
    # Linear / sparse / patched: ~ O(n * w) with w=128
    sparse_mem_gb = n_heads * n.astype(float) * 128 * bytes_per / (1024 ** 3)
    linear_mem_gb = n_heads * n.astype(float) * d_model * bytes_per / (1024 ** 3)
    patched_mem_gb = n_heads * (n.astype(float) / 16) ** 2 * bytes_per / (1024 ** 3)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax = axes[0]
    ax.loglog(n, attn_mem_gb, "o-", color=C_RED, lw=2, label="Vanilla attention  O(n^2)")
    ax.loglog(n, patched_mem_gb, "s-", color=C_PURPLE, lw=1.8, label="Patched (P=16)  O((n/P)^2)")
    ax.loglog(n, sparse_mem_gb, "^-", color=C_AMBER, lw=1.8, label="Sparse / local  O(n.w)")
    ax.loglog(n, linear_mem_gb, "d-", color=C_GREEN, lw=1.8, label="Linear attention  O(n.d)")
    ax.axhline(40, color=C_DARK, ls=":", lw=1)
    ax.text(n[0] * 1.1, 45, "A100 80GB roughly", fontsize=8, color=C_DARK)
    ax.axhline(80, color=C_DARK, ls=":", lw=1)
    ax.set_xlabel("sequence length n", fontsize=11, color=C_DARK)
    ax.set_ylabel("attention memory (GB, fp16)", fontsize=11, color=C_DARK)
    ax.set_title("Memory bottleneck of vanilla attention",
                 fontsize=12, color=C_DARK, pad=8)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(True, which="both", alpha=0.3)

    # FLOPs view
    ax = axes[1]
    flops_vanilla = 4 * n.astype(float) ** 2 * d_model
    flops_patched = 4 * (n.astype(float) / 16) ** 2 * d_model
    flops_sparse = 4 * n.astype(float) * 128 * d_model
    flops_linear = 4 * n.astype(float) * d_model ** 2
    ax.loglog(n, flops_vanilla, "o-", color=C_RED, lw=2, label="Vanilla")
    ax.loglog(n, flops_patched, "s-", color=C_PURPLE, lw=1.8, label="Patched (P=16)")
    ax.loglog(n, flops_sparse, "^-", color=C_AMBER, lw=1.8, label="Sparse (w=128)")
    ax.loglog(n, flops_linear, "d-", color=C_GREEN, lw=1.8, label="Linear (Performer)")
    ax.set_xlabel("sequence length n", fontsize=11, color=C_DARK)
    ax.set_ylabel("attention FLOPs per layer", fontsize=11, color=C_DARK)
    ax.set_title("Compute scaling: O(n^2) blows up past n=2k",
                 fontsize=12, color=C_DARK, pad=8)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Why long-context Transformers need sparse, linear, or patched attention",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_quadratic_bottleneck")


# ---------------------------------------------------------------------------
# Figure 6: Decoder-only autoregressive forecasting
# ---------------------------------------------------------------------------
def fig6_decoder_only_forecast() -> None:
    """Decoder-only autoregressive forecasting + causal mask schematic."""
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.22)

    # ---- left: rolling autoregressive forecast ----
    ax = fig.add_subplot(gs[0])
    rng = np.random.default_rng(5)
    n_hist, n_fcst = 60, 24
    t = np.arange(n_hist + n_fcst)
    truth = (np.sin(2 * np.pi * t / 12)
             + 0.3 * np.sin(2 * np.pi * t / 30)
             + 0.05 * t / 10
             + 0.1 * rng.normal(size=t.shape))

    ax.plot(t[:n_hist], truth[:n_hist], color=C_GRAY, lw=1.5,
            label="history")
    ax.plot(t[n_hist:], truth[n_hist:], color=C_DARK, lw=1.4, ls=":",
            label="ground truth")

    # show progressive autoregressive predictions
    pred = list(truth[:n_hist])
    pred_xs, pred_ys = [], []
    for k in range(n_fcst):
        # naive sim: model = truth + small noise + drift
        next_val = truth[n_hist + k] + 0.06 * rng.normal()
        pred.append(next_val)
        pred_xs.append(n_hist + k)
        pred_ys.append(next_val)
    ax.plot(pred_xs, pred_ys, "o-", color=C_BLUE, lw=1.6, ms=4,
            label="autoregressive predictions")

    # highlight one step explicitly
    k_show = 5
    ax.scatter([n_hist + k_show], [pred_ys[k_show]], s=80,
               edgecolor=C_AMBER, facecolor="none", lw=2, zorder=5)
    ax.annotate(
        "step " + str(k_show + 1) + ": feed steps 1.." + str(n_hist + k_show)
        + "\ninto decoder, predict next",
        xy=(n_hist + k_show, pred_ys[k_show]),
        xytext=(n_hist - 24, pred_ys[k_show] + 1.2),
        fontsize=9, color=C_DARK,
        arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.4),
    )

    ax.axvline(n_hist, color=C_GRAY, ls=":", lw=1)
    ax.set_xlabel("time step", fontsize=11, color=C_DARK)
    ax.set_ylabel("value", fontsize=11, color=C_DARK)
    ax.set_title("Decoder-only forecasting: roll predictions forward one step at a time",
                 fontsize=12, color=C_DARK, pad=8)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    # ---- right: causal mask schematic ----
    ax2 = fig.add_subplot(gs[1])
    n = 10
    mask = np.tril(np.ones((n, n)))
    im = ax2.imshow(mask, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if mask[i, j] == 0:
                ax2.text(j, i, "-inf", ha="center", va="center",
                         fontsize=6.5, color=C_GRAY)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(range(1, n + 1), fontsize=8)
    ax2.set_yticklabels(range(1, n + 1), fontsize=8)
    ax2.set_xlabel("key (which step we look at)", fontsize=10)
    ax2.set_ylabel("query (current step)", fontsize=10)
    ax2.set_title("Causal mask: blue = visible, grey = masked future",
                  fontsize=11, color=C_DARK, pad=8)
    ax2.grid(False)

    fig.tight_layout()
    _save(fig, "fig6_decoder_only_forecast")


# ---------------------------------------------------------------------------
# Figure 7: Patching strategy for time series (PatchTST-style)
# ---------------------------------------------------------------------------
def fig7_patching() -> None:
    """Split a long series into patches; each patch becomes one token."""
    rng = np.random.default_rng(13)
    n = 96
    t = np.arange(n)
    series = (np.sin(2 * np.pi * t / 12)
              + 0.4 * np.sin(2 * np.pi * t / 30)
              + 0.06 * t / 10
              + 0.12 * rng.normal(size=n))

    patch_size = 12
    n_patches = n // patch_size

    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0],
                          height_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.22)

    # top-left: raw series with patch boundaries
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, series, color=C_BLUE, lw=1.4)
    for k in range(1, n_patches):
        ax.axvline(k * patch_size, color=C_GRAY, ls=":", lw=1)
    for k in range(n_patches):
        ax.axvspan(k * patch_size, (k + 1) * patch_size,
                   color=C_BLUE if k % 2 == 0 else C_PURPLE,
                   alpha=0.06)
        ax.text((k + 0.5) * patch_size, ax.get_ylim()[1] * 0.85,
                f"P{k+1}", ha="center", va="top", fontsize=8,
                color=C_DARK, weight="bold")
    ax.set_xlabel("time step", fontsize=11, color=C_DARK)
    ax.set_ylabel("value", fontsize=11, color=C_DARK)
    ax.set_title(f"Step 1: split a length-{n} series into {n_patches} patches "
                 f"of size {patch_size}",
                 fontsize=11, color=C_DARK, pad=6)

    # bottom-left: patches stacked as tokens
    ax2 = fig.add_subplot(gs[1, 0])
    patches = series.reshape(n_patches, patch_size)
    im = ax2.imshow(patches, aspect="auto", cmap="RdBu_r",
                    vmin=-np.max(np.abs(patches)), vmax=np.max(np.abs(patches)))
    ax2.set_yticks(range(n_patches))
    ax2.set_yticklabels([f"P{i+1}" for i in range(n_patches)], fontsize=9)
    ax2.set_xlabel("position within patch", fontsize=10)
    ax2.set_ylabel("patch (= token)", fontsize=10)
    ax2.set_title("Step 2: each patch -> one token via a linear projection",
                  fontsize=11, color=C_DARK, pad=6)
    ax2.grid(False)
    plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)

    # right column: bar chart of effective sequence length
    ax3 = fig.add_subplot(gs[:, 1])
    settings = ["raw\n(n=512)", "patch=8\n(n=64)",
                "patch=16\n(n=32)", "patch=32\n(n=16)"]
    seq_lens = [512, 64, 32, 16]
    flops = [(L / 512) ** 2 for L in seq_lens]  # relative cost
    bars = ax3.bar(settings, flops, color=[C_RED, C_AMBER, C_GREEN, C_PURPLE],
                   edgecolor="white", lw=1.2)
    for b, v in zip(bars, flops):
        label = f"{v*100:.2f}%" if v < 0.01 else f"{v*100:.1f}%"
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.02,
                 label, ha="center", va="bottom",
                 fontsize=9, color=C_DARK)
    ax3.set_ylim(0, 1.18)
    ax3.set_ylabel("relative attention cost vs. raw", fontsize=10)
    ax3.set_title("Effect on attention cost (n^2 scaling)",
                  fontsize=11, color=C_DARK, pad=8)
    ax3.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        "Patching: trade a tiny resolution loss for a huge sequence-length reduction",
        fontsize=13, color=C_DARK, y=1.0,
    )
    _save(fig, "fig7_patching")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Time Series Part 05 (Transformer) figures...")
    fig1_architecture()
    fig2_positional_encoding()
    fig3_multihead_patterns()
    fig4_lstm_vs_transformer()
    fig5_quadratic_bottleneck()
    fig6_decoder_only_forecast()
    fig7_patching()
    print("Done.")


if __name__ == "__main__":
    main()
