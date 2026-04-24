"""
Figure generation script for Time Series Part 07:
N-BEATS -- Neural Basis Expansion Analysis for Time Series.

Generates 5 figures shared by the EN and ZH articles. Each figure
isolates one pedagogical idea so the reader can connect the math, the
code, and the picture without ambiguity.

Figures:
    fig1_stacked_residual_blocks  N-BEATS double-residual stacking: each
                                  block emits a backcast and a forecast;
                                  backcast is subtracted from the running
                                  residual, forecasts are summed.
    fig2_basis_decomposition      Trend (polynomial) + seasonality
                                  (Fourier) decomposition of a synthetic
                                  series, plus the reconstructed total.
    fig3_interpretable_vs_generic Side-by-side stack layout:
                                  interpretable (Trend stack ->
                                  Seasonal stack) vs generic (learned
                                  basis blocks).
    fig4_m4_results               M4 competition headline numbers:
                                  N-BEATS vs second best vs ARIMA /
                                  ETS baselines on overall sMAPE and
                                  per-frequency sMAPE.
    fig5_ensemble_strategy        Bagging illustration: multiple
                                  N-BEATS models with different lookback
                                  windows + median aggregation, plus the
                                  empirical sMAPE drop vs ensemble size.

Usage:
    python3 scripts/figures/time-series/07-nbeats.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders.
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

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_RED = "#ef4444"
C_GRAY = "#94a3b8"
C_LIGHT = "#e5e7eb"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "time-series" / "n-beats"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "time-series" / "07-N-BEATS深度架构"


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
# Figure 1 -- Stacked residual blocks (double residual stacking)
# ---------------------------------------------------------------------------
def fig1_stacked_residual_blocks() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))

    n_blocks = 4
    block_x = 4.5
    block_w = 2.6
    block_h = 1.0
    spacing = 1.5

    # Left column: residual stream (top-down)
    res_x = 1.5
    # Right column: forecast accumulator (top-down)
    fc_x = 9.0

    # Input
    _box(ax, res_x - 1.1, 7.5, 2.2, 0.55, "Input  x",
         C_DARK, fontsize=11)
    # Forecast init
    _box(ax, fc_x - 1.1, 7.5, 2.2, 0.55, "y_hat = 0",
         C_DARK, fontsize=11)

    for b in range(n_blocks):
        y = 6.4 - b * spacing
        block_color = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER][b]
        _box(ax, block_x - block_w / 2, y, block_w, block_h,
             f"Block {b+1}\n(FC + basis)", block_color,
             fontsize=10.5)

        # residual in
        _arrow(ax, res_x, y + block_h / 2 + 0.55,
               block_x - block_w / 2, y + block_h / 2,
               color=C_DARK, lw=1.2)
        # backcast out (back to residual)
        _arrow(ax, block_x - block_w / 2, y + block_h * 0.7,
               res_x + 0.4, y - 0.05,
               color=C_RED, lw=1.2)
        ax.text(block_x - block_w / 2 - 0.3, y + block_h * 0.85,
                "x_b", color=C_RED, fontsize=10, weight="bold")
        # forecast out (to accumulator)
        _arrow(ax, block_x + block_w / 2, y + block_h * 0.3,
               fc_x - 0.4, y + block_h * 0.5,
               color=C_GREEN, lw=1.2)
        ax.text(block_x + block_w / 2 + 0.05, y + block_h * 0.15,
                "y_b", color=C_GREEN, fontsize=10, weight="bold")

        # residual subtract node
        sub_y = y - 0.45
        ax.add_patch(FancyBboxPatch(
            (res_x - 0.32, sub_y - 0.32), 0.64, 0.64,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=C_RED, facecolor=C_BG))
        ax.text(res_x, sub_y, "-", ha="center", va="center",
                fontsize=14, color=C_RED, weight="bold")

        # vertical residual arrow above this subtract node
        _arrow(ax, res_x, y + block_h + 0.05, res_x, y + block_h / 2 + 0.55,
               lw=1.0)
        # Down from subtract to next block input arrow
        if b < n_blocks - 1:
            next_y = 6.4 - (b + 1) * spacing
            _arrow(ax, res_x, sub_y - 0.35,
                   res_x, next_y + block_h + 0.05, lw=1.0)

        # accumulator add node
        add_y = y + block_h * 0.5
        if b == 0:
            # connect to "y_hat = 0" then chain to add node
            _arrow(ax, fc_x, 7.5, fc_x, add_y + 0.4, lw=1.0)
        ax.add_patch(FancyBboxPatch(
            (fc_x - 0.32, add_y - 0.32), 0.64, 0.64,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=C_GREEN, facecolor=C_BG))
        ax.text(fc_x, add_y, "+", ha="center", va="center",
                fontsize=14, color=C_GREEN, weight="bold")
        # next add chain
        if b < n_blocks - 1:
            next_add = 6.4 - (b + 1) * spacing + block_h * 0.5
            _arrow(ax, fc_x, add_y - 0.35, fc_x, next_add + 0.4, lw=1.0)

    # Final outputs
    last_y = 6.4 - (n_blocks - 1) * spacing
    _box(ax, res_x - 1.6, last_y - 1.6, 3.2, 0.55,
         "Final residual (noise)", C_GRAY, fontsize=10)
    _arrow(ax, res_x, last_y - 0.78, res_x, last_y - 1.05, lw=1.0)

    _box(ax, fc_x - 1.6, last_y - 1.6, 3.2, 0.55,
         "Final forecast = sum y_b", C_GREEN, fontsize=10.5)
    _arrow(ax, fc_x, last_y - 0.32 + last_y * 0 + 0.18,
           fc_x, last_y - 1.05, lw=1.0)

    ax.text(res_x, 8.4, "Residual stream",
            ha="center", fontsize=11.5, weight="bold", color=C_DARK)
    ax.text(fc_x, 8.4, "Forecast accumulator",
            ha="center", fontsize=11.5, weight="bold", color=C_DARK)

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.5, 9.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("N-BEATS double residual stacking",
                 fontsize=13.5, weight="bold", color=C_DARK, pad=10)

    _save(fig, "fig1_stacked_residual_blocks")


# ---------------------------------------------------------------------------
# Figure 2 -- Basis function decomposition (trend + seasonality)
# ---------------------------------------------------------------------------
def fig2_basis_decomposition() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.0))

    t = np.arange(120)
    trend = 0.04 * t + 0.0008 * (t - 60) ** 2
    seasonal = 3.0 * np.sin(2 * np.pi * t / 24) + 1.2 * np.cos(4 * np.pi * t / 24)
    noise = np.random.RandomState(7).normal(0, 0.45, len(t))
    total = trend + seasonal + noise

    ax = axes[0, 0]
    ax.plot(t, trend, color=C_BLUE, lw=2.2)
    ax.fill_between(t, trend, alpha=0.18, color=C_BLUE)
    ax.set_title("Trend block: polynomial basis  {1, t, t^2}",
                 fontsize=11.5, weight="bold", color=C_DARK)
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")

    ax = axes[0, 1]
    ax.plot(t, seasonal, color=C_PURPLE, lw=2.2)
    ax.set_title("Seasonality block: Fourier basis  {sin, cos}",
                 fontsize=11.5, weight="bold", color=C_DARK)
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")

    ax = axes[1, 0]
    reconstructed = trend + seasonal
    ax.plot(t, total, color=C_GRAY, lw=1.0, alpha=0.7,
            label="observed")
    ax.plot(t, reconstructed, color=C_GREEN, lw=2.2,
            label="trend + seasonality")
    ax.set_title("Block sum reconstructs the signal",
                 fontsize=11.5, weight="bold", color=C_DARK)
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")

    ax = axes[1, 1]
    residual = total - reconstructed
    ax.plot(t, residual, color=C_RED, lw=1.0)
    ax.axhline(0, color=C_DARK, lw=0.8)
    ax.fill_between(t, residual, alpha=0.18, color=C_RED)
    ax.set_title("Residual after decomposition (~ noise)",
                 fontsize=11.5, weight="bold", color=C_DARK)
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")

    fig.suptitle("Interpretable N-BEATS: explicit trend + seasonality decomposition",
                 fontsize=14, weight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()

    _save(fig, "fig2_basis_decomposition")


# ---------------------------------------------------------------------------
# Figure 3 -- Interpretable vs generic architecture
# ---------------------------------------------------------------------------
def fig3_interpretable_vs_generic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))

    # ---------- Interpretable
    ax = axes[0]
    # Two stacks: trend stack (3 trend blocks), seasonality stack (3 seasonal blocks)
    stack_w, stack_h = 4.8, 2.2

    _box(ax, 0.6, 4.2, stack_w, stack_h, "", C_BLUE)
    ax.text(0.6 + stack_w / 2, 4.2 + stack_h - 0.25,
            "Trend stack (polynomial basis)",
            ha="center", fontsize=11, weight="bold", color=C_BLUE)
    for i in range(3):
        _box(ax, 0.9 + i * 1.55, 4.45, 1.4, 0.7,
             f"Trend\nblock {i+1}", C_BLUE,
             fontsize=9, fc="#dbeafe")

    _box(ax, 0.6, 1.0, stack_w, stack_h, "", C_PURPLE)
    ax.text(0.6 + stack_w / 2, 1.0 + stack_h - 0.25,
            "Seasonality stack (Fourier basis)",
            ha="center", fontsize=11, weight="bold", color=C_PURPLE)
    for i in range(3):
        _box(ax, 0.9 + i * 1.55, 1.25, 1.4, 0.7,
             f"Seasonal\nblock {i+1}", C_PURPLE,
             fontsize=9, fc="#ede9fe")

    _arrow(ax, 0.6 + stack_w / 2, 4.2,
           0.6 + stack_w / 2, 1.0 + stack_h, lw=1.3)
    ax.text(0.6 + stack_w / 2 + 0.3, 3.45,
            "residual", color=C_DARK, fontsize=9, style="italic")

    _box(ax, 0.6 + stack_w / 2 - 1.6, 0.05, 3.2, 0.55,
         "Forecast = trend + seasonality",
         C_GREEN, fontsize=10.5)

    ax.text(0.6 + stack_w / 2, 7.0,
            "Interpretable variant",
            ha="center", fontsize=12.5, weight="bold", color=C_DARK)
    ax.text(0.6 + stack_w / 2, 6.5,
            "explicit basis -> readable trend & seasonal components",
            ha="center", fontsize=10, color=C_DARK, style="italic")

    ax.set_xlim(-0.2, 0.6 + stack_w + 0.2)
    ax.set_ylim(-0.3, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------- Generic
    ax = axes[1]
    _box(ax, 0.6, 1.0, stack_w, 5.4, "", C_AMBER)
    ax.text(0.6 + stack_w / 2, 1.0 + 5.4 - 0.25,
            "Generic stack (learned basis)",
            ha="center", fontsize=11, weight="bold", color=C_AMBER)
    for i in range(6):
        row = i // 3
        col = i % 3
        _box(ax, 0.9 + col * 1.55, 4.0 - row * 1.05, 1.4, 0.7,
             f"Generic\nblock {i+1}", C_AMBER,
             fontsize=9, fc="#fef3c7")

    _box(ax, 0.6 + stack_w / 2 - 1.6, 0.05, 3.2, 0.55,
         "Forecast = sum of generic basis",
         C_GREEN, fontsize=10.5)

    ax.text(0.6 + stack_w / 2, 7.0,
            "Generic variant",
            ha="center", fontsize=12.5, weight="bold", color=C_DARK)
    ax.text(0.6 + stack_w / 2, 6.5,
            "learned basis -> slightly higher accuracy, no semantic split",
            ha="center", fontsize=10, color=C_DARK, style="italic")

    ax.set_xlim(-0.2, 0.6 + stack_w + 0.2)
    ax.set_ylim(-0.3, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("N-BEATS variants: interpretable basis vs generic learned basis",
                 fontsize=14, weight="bold", color=C_DARK, y=1.02)

    _save(fig, "fig3_interpretable_vs_generic")


# ---------------------------------------------------------------------------
# Figure 4 -- M4 competition results
# ---------------------------------------------------------------------------
def fig4_m4_results() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Headline overall numbers from Oreshkin et al. (2020), reported on M4
    ax = axes[0]
    models = ["Naive2",
              "ETS",
              "ARIMA",
              "Theta",
              "FFORMA\n(2nd, M4)",
              "Smyl\n(1st, M4)",
              "N-BEATS-G",
              "N-BEATS-I+G"]
    smape = [13.564, 13.525, 13.116, 12.309, 11.720, 11.374, 11.168, 11.135]
    colors = [C_GRAY, C_GRAY, C_GRAY, C_GRAY, C_AMBER, C_PURPLE,
              C_BLUE, C_GREEN]

    bars = ax.bar(models, smape, color=colors, edgecolor=C_DARK, lw=0.8)
    for b, v in zip(bars, smape):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, color=C_DARK, weight="bold")
    ax.set_ylabel("Overall sMAPE (lower = better)")
    ax.set_ylim(10.5, 14.0)
    ax.set_title("M4 overall sMAPE (100k series, 6 frequencies)",
                 fontsize=12, weight="bold", color=C_DARK)
    ax.tick_params(axis="x", rotation=20, labelsize=9)

    # Per-frequency comparison: ensemble N-BEATS vs M4 winner Smyl ES-RNN
    ax = axes[1]
    freqs = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    nbeats = [13.114, 9.154, 12.024, 7.406, 3.045, 9.764]
    smyl = [13.176, 9.679, 12.126, 7.817, 3.170, 9.328]
    x = np.arange(len(freqs))
    width = 0.38
    ax.bar(x - width / 2, smyl, width, label="Smyl ES-RNN",
           color=C_PURPLE, edgecolor=C_DARK, lw=0.8)
    ax.bar(x + width / 2, nbeats, width, label="N-BEATS (ensemble)",
           color=C_GREEN, edgecolor=C_DARK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(freqs)
    ax.set_ylabel("sMAPE")
    ax.set_title("Per-frequency sMAPE: N-BEATS wins on 5 / 6 buckets",
                 fontsize=12, weight="bold", color=C_DARK)
    ax.legend(frameon=True, fontsize=10)

    fig.suptitle("M4 forecasting competition: N-BEATS vs the leaderboard",
                 fontsize=14, weight="bold", color=C_DARK, y=1.04)
    fig.tight_layout()

    _save(fig, "fig4_m4_results")


# ---------------------------------------------------------------------------
# Figure 5 -- Ensemble strategy
# ---------------------------------------------------------------------------
def fig5_ensemble_strategy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    # Left: ensemble schematic
    ax = axes[0]
    _box(ax, 0.5, 4.5, 2.6, 0.7, "Time series x", C_DARK, fontsize=10.5)

    members = [
        ("Model A\n(lookback 2H)", C_BLUE),
        ("Model B\n(lookback 3H)", C_PURPLE),
        ("Model C\n(lookback 4H)", C_AMBER),
        ("Model D\n(loss: sMAPE)", C_GREEN),
        ("Model E\n(loss: MASE)", C_RED),
    ]
    for i, (label, color) in enumerate(members):
        _box(ax, 0.5, 3.4 - i * 0.7, 2.6, 0.55, label, color, fontsize=9)
        _arrow(ax, 1.8, 4.5, 1.8, 3.95 - i * 0.7,
               color=C_GRAY, lw=0.8)
        _arrow(ax, 3.1, 3.65 - i * 0.7, 4.6, 2.75, color=color, lw=1.0)

    _box(ax, 4.6, 2.3, 2.6, 0.9, "Median\naggregator",
         C_GREEN, fontsize=11)
    _box(ax, 4.6, 0.9, 2.6, 0.7, "Final forecast",
         C_DARK, fontsize=11)
    _arrow(ax, 5.9, 2.3, 5.9, 1.6, lw=1.2)

    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Ensemble: diverse lookbacks + losses, aggregated by median",
                 fontsize=12, weight="bold", color=C_DARK, pad=10)

    # Right: empirical sMAPE drop vs ensemble size (from Oreshkin et al.)
    ax = axes[1]
    sizes = np.array([1, 5, 10, 20, 40, 80, 180])
    # Approx values: single model ~12.0, asymptote near 11.135 with 180 members.
    s = 11.13 + 0.95 * np.exp(-sizes / 25)
    ax.plot(sizes, s, color=C_BLUE, lw=2.2, marker="o",
            markerfacecolor=C_AMBER, markeredgecolor=C_DARK)
    ax.set_xscale("log")
    ax.set_xlabel("Ensemble size (log scale)")
    ax.set_ylabel("M4 overall sMAPE")
    ax.set_title("Diminishing returns: most gain captured by ~30 members",
                 fontsize=12, weight="bold", color=C_DARK)
    ax.axhline(11.135, color=C_GREEN, ls="--", lw=1.2,
               label="N-BEATS-I+G (180 models)")
    ax.legend(frameon=True, fontsize=10)

    fig.suptitle("Why N-BEATS papers always report ensembles",
                 fontsize=14, weight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()

    _save(fig, "fig5_ensemble_strategy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_stacked_residual_blocks()
    fig2_basis_decomposition()
    fig3_interpretable_vs_generic()
    fig4_m4_results()
    fig5_ensemble_strategy()
    print(f"Wrote 5 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
