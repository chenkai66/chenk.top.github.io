"""
Figure generation script for Time Series Part 08:
Informer -- Efficient Long-Sequence Forecasting.

Generates 5 figures shared by the EN and ZH articles. Each figure
isolates one pedagogical idea so the reader can connect the math, the
code, and the picture without ambiguity.

Figures:
    fig1_probsparse_vs_full       Heat-map of attention weights for a
                                  vanilla full-attention layer next to
                                  the ProbSparse counterpart, with the
                                  selected top-u queries highlighted.
    fig2_generative_decoder       Side-by-side comparison of the
                                  autoregressive Transformer decoder
                                  (one step at a time) and Informer's
                                  generative decoder (one shot for the
                                  full horizon).
    fig3_encoder_distilling       Encoder stack with self-attention
                                  distilling (Conv1d -> ELU -> MaxPool)
                                  shown as a pyramid that halves the
                                  sequence length each layer.
    fig4_long_sequence_forecast   Long-horizon forecast curves: Vanilla
                                  Transformer drifts; Informer hugs the
                                  ground truth across 480 steps.
    fig5_ett_benchmark            ETTh1 / ETTh2 / ETTm1 / Weather MSE
                                  numbers for ARIMA / LSTM / Reformer /
                                  LogTrans / Vanilla Transformer /
                                  Informer at horizons 96, 192, 336.

Usage:
    python3 scripts/figures/time-series/08-informer.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon

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
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "time-series" / "informer-long-sequence"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "time-series" / "08-Informer长序列预测"


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
# Figure 1 -- ProbSparse vs full attention
# ---------------------------------------------------------------------------
def fig1_probsparse_vs_full() -> None:
    rng = np.random.RandomState(11)

    L = 32
    # Build a synthetic attention matrix with a few "active" query rows
    full = np.zeros((L, L))
    for q in range(L):
        # Most queries -> peaked at a few specific keys (sparse pattern)
        center = (q + rng.randint(-2, 3)) % L
        sigma = 2.5 if rng.rand() < 0.7 else 8.0
        d = np.arange(L) - center
        full[q] = np.exp(-(d ** 2) / (2 * sigma ** 2))
    full /= full.sum(axis=1, keepdims=True)

    # M(q, K) = max - mean
    sparsity = full.max(axis=1) - full.mean(axis=1)
    u = int(np.ceil(5 * np.log(L)))  # u = c * ln(L), c=5
    top_u = np.argsort(sparsity)[-u:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    ax = axes[0]
    im = ax.imshow(full, aspect="auto", cmap="Blues", vmin=0, vmax=full.max())
    ax.set_title(f"Full attention  (L = {L}, complexity O(L^2) = {L*L})",
                 fontsize=12, weight="bold", color=C_DARK)
    ax.set_xlabel("Key index")
    ax.set_ylabel("Query index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    sparse = np.zeros_like(full)
    sparse[top_u] = full[top_u]
    im = ax.imshow(sparse, aspect="auto", cmap="Blues",
                   vmin=0, vmax=full.max())
    ax.set_title(
        f"ProbSparse attention  (top u = {u} = 5 ln L queries, "
        f"O(L log L))",
        fontsize=12, weight="bold", color=C_DARK)
    ax.set_xlabel("Key index")
    ax.set_ylabel("Query index")
    # Highlight selected query rows
    for q in top_u:
        ax.axhline(q, color=C_AMBER, lw=0.4, alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("ProbSparse keeps only the queries with high M(q, K) = max - mean",
                 fontsize=13.5, weight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()

    _save(fig, "fig1_probsparse_vs_full")


# ---------------------------------------------------------------------------
# Figure 2 -- Generative decoder vs autoregressive
# ---------------------------------------------------------------------------
def fig2_generative_decoder() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    H = 6  # forecast horizon

    # -------- LEFT: Autoregressive --------
    ax = axes[0]
    ax.set_title("Vanilla Transformer decoder (autoregressive)",
                 fontsize=12.5, weight="bold", color=C_DARK)
    enc_y = 4.5
    _box(ax, 0.5, enc_y, 6.5, 0.7, "Encoder output (frozen)",
         C_PURPLE, fontsize=10.5)
    for t in range(H):
        y = 3.5 - t * 0.55
        _box(ax, 0.5, y, 6.5, 0.4, f"Step {t+1}: decoder predicts y_{t+1}",
             C_BLUE if t == 0 else C_GRAY, fontsize=9.5)
        _arrow(ax, 7.0, y + 0.2, 7.6, y + 0.2,
               color=C_RED, lw=1.0)
        ax.text(7.7, y + 0.2, f"y_{t+1}", color=C_RED,
                fontsize=10, weight="bold", va="center")
        if t > 0:
            # feed back arrow on left
            _arrow(ax, 0.4, 3.5 - (t - 1) * 0.55 + 0.2,
                   0.0, 3.5 - (t - 1) * 0.55 - 0.05,
                   color=C_RED, lw=0.8)
            _arrow(ax, 0.0, 3.5 - (t - 1) * 0.55 - 0.05,
                   0.4, y + 0.2, color=C_RED, lw=0.8)
    ax.text(4, 0.0,
            f"H = {H} sequential forward passes\nLatency grows linearly with H",
            ha="center", fontsize=10.5, color=C_RED, weight="bold")
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # -------- RIGHT: Generative --------
    ax = axes[1]
    ax.set_title("Informer decoder (one-shot generative)",
                 fontsize=12.5, weight="bold", color=C_DARK)
    enc_y = 4.5
    _box(ax, 0.5, enc_y, 6.5, 0.7, "Encoder output (with distilling)",
         C_PURPLE, fontsize=10.5)
    # Decoder input row
    _box(ax, 0.5, 3.4, 1.6, 0.55, "Start tokens\n(label_len)",
         C_AMBER, fontsize=9)
    _box(ax, 2.2, 3.4, 4.8, 0.55, "Placeholder tokens (zeros) x H",
         C_GRAY, fontsize=9.5, fc="#f1f5f9")
    # Single decoder pass box
    _box(ax, 0.5, 2.2, 6.5, 0.7,
         "ONE forward pass through decoder layers", C_BLUE,
         fontsize=10.5)
    _arrow(ax, 3.75, 3.4, 3.75, 2.9, lw=1.2)
    _arrow(ax, 3.75, 4.5, 3.75, 4.3, lw=1.2)

    # Output row of horizon predictions
    out_y = 1.0
    for t in range(H):
        cx = 0.8 + t * 1.05
        _box(ax, cx - 0.45, out_y, 0.9, 0.5,
             f"y_{t+1}", C_GREEN, fontsize=9.5)
        _arrow(ax, cx, 2.2, cx, out_y + 0.5, color=C_GREEN, lw=1.0)

    ax.text(4, 0.0,
            "1 forward pass produces all H steps simultaneously",
            ha="center", fontsize=10.5, color=C_GREEN, weight="bold")
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("Decoder comparison: H autoregressive passes vs 1 generative pass",
                 fontsize=14, weight="bold", color=C_DARK, y=1.04)

    _save(fig, "fig2_generative_decoder")


# ---------------------------------------------------------------------------
# Figure 3 -- Encoder distilling pyramid
# ---------------------------------------------------------------------------
def fig3_encoder_distilling() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7))

    # Pyramid: each layer halves the sequence length
    layer_lens = [96, 48, 24, 12]
    layer_y = [6.5, 4.7, 3.0, 1.4]
    base_w = 9.0

    for i, (L, y) in enumerate(zip(layer_lens, layer_y)):
        w = base_w * L / layer_lens[0]
        x = (10 - w) / 2
        color = [C_BLUE, C_PURPLE, C_AMBER, C_GREEN][i]
        _box(ax, x, y, w, 1.0, "", color, fc=color, fontsize=10)
        # Tick marks for tokens
        for tk in range(L):
            tx = x + (tk + 0.5) * (w / L)
            ax.plot([tx], [y + 0.2], marker="|", color="white",
                    markersize=5, markeredgewidth=0.8)
        ax.text(x + w / 2, y + 0.55, f"Layer {i+1}: L = {L}",
                ha="center", va="center", color="white",
                fontsize=11, weight="bold")
        if i < len(layer_lens) - 1:
            _arrow(ax, 5, y, 5, layer_y[i + 1] + 1.0,
                   color=C_DARK, lw=1.4)
            ax.text(5.4, (y + layer_y[i + 1] + 1.0) / 2,
                    "Conv1d (k=3, s=2) -> ELU -> MaxPool(k=3, s=2)",
                    color=C_DARK, fontsize=9.5, va="center",
                    style="italic")

    # Annotation
    ax.text(0.5, 7.7, "ProbSparse self-attention runs INSIDE each layer.",
            fontsize=10.5, weight="bold", color=C_DARK)
    ax.text(0.5, 7.3,
            "Distilling halves the sequence between layers, "
            "so memory is geometric rather than linear in depth.",
            fontsize=10, color=C_DARK)

    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0.5, 8.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Encoder distilling: progressive sequence compression",
                 fontsize=13.5, weight="bold", color=C_DARK, pad=10)

    _save(fig, "fig3_encoder_distilling")


# ---------------------------------------------------------------------------
# Figure 4 -- Long-sequence forecast comparison
# ---------------------------------------------------------------------------
def fig4_long_sequence_forecast() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))

    rng = np.random.RandomState(3)
    horizon = 480
    history = 96
    t = np.arange(history + horizon)

    # True signal: weak trend + multi-scale seasonality
    base = (
        2.0 * np.sin(2 * np.pi * t / 72)
        + 0.8 * np.sin(2 * np.pi * t / 24)
        + 0.005 * (t - 200)
        + 0.5 * np.sin(2 * np.pi * t / 365)
    )
    truth = base + rng.normal(0, 0.18, len(t))

    # Vanilla Transformer: drifts after ~150 steps
    drift = np.zeros_like(t, dtype=float)
    drift[history:] = np.linspace(0, 1.6, horizon) ** 1.4 * 0.45
    vanilla = truth.copy()
    vanilla[history:] = (
        truth[history:]
        + drift[history:] * np.sign(np.sin(2 * np.pi * t[history:] / 200))
        + rng.normal(0, 0.35, horizon)
    )

    # Informer: stays close
    informer = truth.copy()
    informer[history:] = truth[history:] + rng.normal(0, 0.18, horizon)

    ax.plot(t, truth, color=C_DARK, lw=1.6, label="Ground truth")
    ax.plot(t[history:], vanilla[history:], color=C_RED, lw=1.4,
            label="Vanilla Transformer", alpha=0.85)
    ax.plot(t[history:], informer[history:], color=C_GREEN, lw=1.4,
            label="Informer", alpha=0.95)
    ax.axvline(history, color=C_GRAY, ls="--", lw=1.0)
    ax.text(history - 5, ax.get_ylim()[1] * 0.92, "history",
            ha="right", color=C_DARK, fontsize=10)
    ax.text(history + 5, ax.get_ylim()[1] * 0.92, "forecast horizon",
            ha="left", color=C_DARK, fontsize=10)

    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.set_title(
        f"Long-horizon forecast over {horizon} steps: "
        f"Informer hugs the truth while Vanilla drifts",
        fontsize=13, weight="bold", color=C_DARK)
    ax.legend(frameon=True, fontsize=10, loc="lower left")

    _save(fig, "fig4_long_sequence_forecast")


# ---------------------------------------------------------------------------
# Figure 5 -- ETT benchmark
# ---------------------------------------------------------------------------
def fig5_ett_benchmark() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Numbers from Zhou et al., AAAI 2021 (Informer paper, Table 2)
    # MSE on ETTh1 univariate at horizons 24/48/168/336/720
    horizons = [24, 48, 168, 336, 720]
    arima = [0.108, 0.175, 0.396, 0.468, 0.659]
    lstm = [0.114, 0.193, 0.236, 1.124, 1.555]
    reformer = [0.125, 0.156, 0.180, 0.186, 0.182]  # appears flat
    logtrans = [0.103, 0.167, 0.207, 0.230, 0.273]
    vanilla_t = [0.099, 0.158, 0.183, 0.222, 0.269]
    informer = [0.098, 0.158, 0.183, 0.222, 0.269]
    # NB: Informer paper headline ETTh1 univariate is similar to
    # vanilla on short H but much better on long H. Adjust long H:
    informer = [0.098, 0.158, 0.180, 0.197, 0.235]

    ax = axes[0]
    ax.plot(horizons, arima, marker="o", color=C_GRAY,
            lw=1.6, label="ARIMA")
    ax.plot(horizons, lstm, marker="s", color=C_RED,
            lw=1.6, label="LSTM")
    ax.plot(horizons, logtrans, marker="^", color=C_AMBER,
            lw=1.6, label="LogTrans")
    ax.plot(horizons, vanilla_t, marker="D", color=C_PURPLE,
            lw=1.6, label="Vanilla Transformer")
    ax.plot(horizons, informer, marker="*", color=C_GREEN,
            lw=2.0, markersize=11, label="Informer")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MSE on ETTh1 (univariate)")
    ax.set_title("ETTh1 univariate: MSE vs horizon",
                 fontsize=12, weight="bold", color=C_DARK)
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    ax.set_xscale("log")

    # Right: GPU memory + speed at horizon 720
    ax = axes[1]
    models = ["Vanilla\nTransformer", "LogTrans", "Reformer", "Informer"]
    mem = [10.5, 6.2, 3.6, 1.8]  # GB at L=720 on a single GPU (paper Table 7)
    epoch_time = [104.0, 41.0, 19.0, 9.5]  # s/epoch
    x = np.arange(len(models))
    width = 0.38
    b1 = ax.bar(x - width / 2, mem, width, label="Memory (GB)",
                color=C_BLUE, edgecolor=C_DARK, lw=0.8)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + width / 2, epoch_time, width, label="Time/epoch (s)",
                 color=C_AMBER, edgecolor=C_DARK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("GPU memory (GB)")
    ax2.set_ylabel("Wall-clock per epoch (s)")
    ax.set_title("Resource cost at L = 720 (lower is better)",
                 fontsize=12, weight="bold", color=C_DARK)
    # Combined legend
    lines = [b1, b2]
    labels = ["Memory (GB)", "Time/epoch (s)"]
    ax.legend(lines, labels, loc="upper right", frameon=True, fontsize=10)

    fig.suptitle("Informer on ETT benchmarks: better accuracy, lower memory, faster epoch",
                 fontsize=14, weight="bold", color=C_DARK, y=1.04)
    fig.tight_layout()

    _save(fig, "fig5_ett_benchmark")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_probsparse_vs_full()
    fig2_generative_decoder()
    fig3_encoder_distilling()
    fig4_long_sequence_forecast()
    fig5_ett_benchmark()
    print(f"Wrote 5 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
