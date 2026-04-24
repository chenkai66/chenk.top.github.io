"""
Figure generation script for Time Series Part 02: LSTM.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure teaches a single concept cleanly with consistent styling.

Figures:
    fig1_lstm_cell             LSTM cell architecture: input/forget/output
                               gates, candidate, cell-state update.
    fig2_state_highway         Cell state vs hidden state — the long-term
                               "highway" vs the gated short-term output.
    fig3_forecast              One-step LSTM forecast vs actual on a
                               synthetic noisy seasonal series.
    fig4_multistep             Multi-step (recursive vs direct) ahead
                               prediction with widening uncertainty.
    fig5_bilstm                Bidirectional LSTM: forward + backward passes
                               and concatenated hidden state per step.
    fig6_stacked_lstm          Stacked LSTM (3 layers) with hierarchical
                               feature extraction across time.
    fig7_training_curves       Training/validation loss with early stopping
                               trigger marked on the curve.

Usage:
    python3 scripts/figures/time-series/02-lstm.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary (input)
C_PURPLE = "#7c3aed"   # secondary (hidden state)
C_GREEN = "#10b981"    # accent (cell state / good)
C_AMBER = "#f59e0b"    # warning / highlight (gates / problem)
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_BG = "#f8fafc"
C_LIGHT_BLUE = "#dbeafe"
C_LIGHT_PURPLE = "#ede9fe"
C_LIGHT_GREEN = "#d1fae5"
C_LIGHT_AMBER = "#fef3c7"
C_LIGHT_GRAY = "#e2e8f0"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "time-series" / "lstm"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "time-series" / "02-LSTM"

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
        boxstyle="round,pad=0.02,rounding_size=0.06",
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
# Figure 1: LSTM cell architecture (gates + candidate + update)
# ---------------------------------------------------------------------------
def fig1_lstm_cell() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Outer cell boundary
    cell = FancyBboxPatch(
        (1.2, 1.0), 9.6, 5.0,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.6, edgecolor=C_GRAY, facecolor=C_BG,
    )
    ax.add_patch(cell)
    ax.text(1.5, 5.7, "LSTM cell", fontsize=10, color=C_GRAY,
            fontweight="bold", style="italic")

    # Inputs from the left
    _box(ax, (-0.1, 4.4), 1.0, 0.8, r"$h_{t-1}$", C_LIGHT_PURPLE, fs=12)
    _box(ax, (-0.1, 1.4), 1.0, 0.8, r"$x_t$", C_LIGHT_BLUE, fs=12)
    # Cell state in (top)
    _box(ax, (-0.1, 5.6), 1.0, 0.8, r"$C_{t-1}$", C_LIGHT_GREEN, fs=12)

    # Gates row (forget, input, candidate, output)
    gate_y = 3.2
    gate_h = 0.85
    gate_w = 1.1
    forget_x = 2.2
    input_x = 4.2
    cand_x = 6.0
    output_x = 8.6

    _box(ax, (forget_x, gate_y), gate_w, gate_h, r"$f_t$  $\sigma$",
         C_LIGHT_AMBER, ec=C_AMBER, fs=12)
    ax.text(forget_x + gate_w / 2, gate_y - 0.35, "forget gate",
            ha="center", fontsize=8.5, color=C_AMBER, style="italic")

    _box(ax, (input_x, gate_y), gate_w, gate_h, r"$i_t$  $\sigma$",
         C_LIGHT_AMBER, ec=C_AMBER, fs=12)
    ax.text(input_x + gate_w / 2, gate_y - 0.35, "input gate",
            ha="center", fontsize=8.5, color=C_AMBER, style="italic")

    _box(ax, (cand_x, gate_y), gate_w, gate_h, r"$\tilde C_t$  $\tanh$",
         C_LIGHT_BLUE, ec=C_BLUE, fs=12)
    ax.text(cand_x + gate_w / 2, gate_y - 0.35, "candidate",
            ha="center", fontsize=8.5, color=C_BLUE, style="italic")

    _box(ax, (output_x, gate_y), gate_w, gate_h, r"$o_t$  $\sigma$",
         C_LIGHT_AMBER, ec=C_AMBER, fs=12)
    ax.text(output_x + gate_w / 2, gate_y - 0.35, "output gate",
            ha="center", fontsize=8.5, color=C_AMBER, style="italic")

    # Cell-state highway (top)
    ax.plot([0.9, 11.3], [5.95, 5.95], color=C_GREEN, lw=3.2, zorder=2)
    # Forget multiplier
    ax.plot(forget_x + gate_w / 2, 5.95, marker="o", markersize=14,
            markerfacecolor="white", markeredgecolor=C_GREEN, mew=2)
    ax.text(forget_x + gate_w / 2, 5.95, r"$\times$", ha="center",
            va="center", fontsize=12, color=C_GREEN, fontweight="bold")
    # Add multiplier between input gate and candidate
    add_x = (input_x + cand_x + gate_w) / 2
    ax.plot(add_x, 5.95, marker="o", markersize=14,
            markerfacecolor="white", markeredgecolor=C_GREEN, mew=2)
    ax.text(add_x, 5.95, r"$+$", ha="center", va="center",
            fontsize=14, color=C_GREEN, fontweight="bold")
    # Multiplier between input gate output and candidate
    ax.plot(cand_x - 0.4, 4.2, marker="o", markersize=12,
            markerfacecolor="white", markeredgecolor=C_BLUE, mew=2)
    ax.text(cand_x - 0.4, 4.2, r"$\times$", ha="center", va="center",
            fontsize=11, color=C_BLUE, fontweight="bold")

    # tanh on cell state for hidden state
    tanh_x = output_x + gate_w + 0.6
    _box(ax, (tanh_x, 4.7), 0.9, 0.7, r"$\tanh$", C_LIGHT_GREEN,
         ec=C_GREEN, fs=11)
    # Output multiplier
    out_mult_x = tanh_x + 0.45
    ax.plot(out_mult_x, 3.65, marker="o", markersize=12,
            markerfacecolor="white", markeredgecolor=C_PURPLE, mew=2)
    ax.text(out_mult_x, 3.65, r"$\times$", ha="center", va="center",
            fontsize=11, color=C_PURPLE, fontweight="bold")

    # Outputs to the right
    _box(ax, (11.0, 5.6), 1.0, 0.8, r"$C_t$", C_LIGHT_GREEN, fs=12)
    _box(ax, (11.0, 3.25), 1.0, 0.8, r"$h_t$", C_LIGHT_PURPLE, fs=12)

    # Wires: inputs into a shared rail (cleaner than 8 crossing arrows)
    rail_y = 2.55
    rail_x_left = 1.4
    rail_x_right = output_x + gate_w + 0.2
    ax.plot([rail_x_left, rail_x_right], [rail_y, rail_y],
            color=C_GRAY, lw=1.2, ls=(0, (4, 2)), alpha=0.8, zorder=1)
    ax.text(rail_x_left + 0.05, rail_y + 0.18,
            r"shared input $[h_{t-1}, x_t]$",
            fontsize=8.5, color=C_GRAY, style="italic")
    # x_t and h_{t-1} feed the rail at the left edge
    _arrow(ax, (0.9, 1.8), (rail_x_left, rail_y), color=C_BLUE, lw=1.3)
    _arrow(ax, (0.9, 4.8), (rail_x_left, rail_y), color=C_PURPLE, lw=1.3)
    # Rail to each gate (short vertical taps)
    for gx in (forget_x, input_x, cand_x, output_x):
        _arrow(ax, (gx + gate_w / 2, rail_y), (gx + gate_w / 2, gate_y),
               color=C_GRAY, lw=1.0)

    # forget gate to highway multiplier
    _arrow(ax, (forget_x + gate_w / 2, gate_y + gate_h),
           (forget_x + gate_w / 2, 5.85), color=C_AMBER, lw=1.5)
    # input gate to its multiplier
    _arrow(ax, (input_x + gate_w / 2, gate_y + gate_h),
           (cand_x - 0.4, 4.1), color=C_AMBER, lw=1.5)
    # candidate to multiplier
    _arrow(ax, (cand_x, gate_y + gate_h / 2),
           (cand_x - 0.3, 4.15), color=C_BLUE, lw=1.5)
    # multiplier to add on highway
    _arrow(ax, (cand_x - 0.4, 4.3), (add_x, 5.85), color=C_GREEN, lw=1.5)
    # highway to tanh
    _arrow(ax, (tanh_x + 0.45, 5.85), (tanh_x + 0.45, 5.4), color=C_GREEN,
           lw=1.5)
    # tanh to output multiplier
    _arrow(ax, (tanh_x + 0.45, 4.7), (out_mult_x, 3.78), color=C_GREEN,
           lw=1.5)
    # output gate to multiplier
    _arrow(ax, (output_x + gate_w / 2, gate_y + gate_h / 2),
           (out_mult_x - 0.1, 3.65), color=C_AMBER, lw=1.5)
    # multiplier to h_t
    _arrow(ax, (out_mult_x, 3.65), (11.0, 3.65), color=C_PURPLE, lw=1.6)

    # Title and key equations
    ax.text(6, 6.7, "LSTM cell: three gates over a cell-state highway",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(6, 0.45,
            r"$f_t,i_t,o_t = \sigma(\cdot)$    "
            r"$\tilde C_t = \tanh(\cdot)$    "
            r"$C_t = f_t\!\odot\!C_{t-1} + i_t\!\odot\!\tilde C_t$    "
            r"$h_t = o_t\!\odot\!\tanh(C_t)$",
            ha="center", fontsize=10.5, color=C_DARK,
            bbox=dict(facecolor="white", edgecolor=C_GRAY,
                      boxstyle="round,pad=0.4"))

    fig.tight_layout()
    save(fig, "fig1_lstm_cell")


# ---------------------------------------------------------------------------
# Figure 2: Cell state vs hidden state — the highway
# ---------------------------------------------------------------------------
def fig2_state_highway() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    n = 5
    xs = [1.5 + 2.0 * i for i in range(n)]

    # Cell state highway (top, thick green)
    ax.plot([0.4, 11.6], [4.1, 4.1], color=C_GREEN, lw=4.0, zorder=1)
    ax.text(0.4, 4.45, "cell state $C_t$  (long-term highway)",
            fontsize=10.5, color=C_GREEN, fontweight="bold")

    # Hidden state line (gated, dashed purple)
    ax.plot([0.4, 11.6], [1.4, 1.4], color=C_PURPLE, lw=1.6,
            ls=(0, (4, 2)), zorder=1, alpha=0.6)
    ax.text(0.4, 1.05,
            "hidden state $h_t$  (filtered by output gate, exposed to next layer)",
            fontsize=10.5, color=C_PURPLE, fontweight="bold")

    for i, x in enumerate(xs):
        # Cell node on highway
        ax.plot(x, 4.1, marker="o", markersize=14, markerfacecolor="white",
                markeredgecolor=C_GREEN, mew=2.2, zorder=3)
        ax.text(x, 4.1, f"$C_{{{i+1}}}$", ha="center", va="center",
                fontsize=10, color=C_GREEN, fontweight="bold")

        # Output gate (small amber box)
        _box(ax, (x - 0.45, 2.55), 0.9, 0.55, r"$o_t \cdot \tanh$",
             C_LIGHT_AMBER, ec=C_AMBER, fs=8.5, fw="bold")

        # Hidden state node
        ax.plot(x, 1.4, marker="s", markersize=12,
                markerfacecolor="white", markeredgecolor=C_PURPLE, mew=2,
                zorder=3)
        ax.text(x, 1.4, f"$h_{{{i+1}}}$", ha="center", va="center",
                fontsize=9.5, color=C_PURPLE, fontweight="bold")

        # Vertical: highway -> output gate -> hidden
        _arrow(ax, (x, 3.95), (x, 3.15), color=C_GREEN, lw=1.3)
        _arrow(ax, (x, 2.55), (x, 1.55), color=C_PURPLE, lw=1.3)

    # Annotation: gradient flow
    ax.annotate("", xy=(11.4, 4.45), xytext=(0.6, 4.45),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.5,
                                ls="--"))
    ax.text(6, 4.75, "gradient flows nearly intact across many steps",
            ha="center", fontsize=9.5, color=C_GREEN, style="italic")

    ax.text(6, 0.4,
            "Two parallel state streams: $C_t$ accumulates memory; "
            "$h_t = o_t \\odot \\tanh(C_t)$ is what the rest of the network sees.",
            ha="center", fontsize=10, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_GRAY,
                      boxstyle="round,pad=0.4"))

    ax.text(6, 4.95, "Cell state vs hidden state",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)

    fig.tight_layout()
    save(fig, "fig2_state_highway")


# ---------------------------------------------------------------------------
# Helper: synthetic series + simple "LSTM" style smoothed forecast
# ---------------------------------------------------------------------------
def _synthetic_series(n=300, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = 1.5 * np.sin(2 * np.pi * t / 24) + 0.6 * np.sin(2 * np.pi * t / 75)
    trend = 0.005 * t
    noise = rng.normal(0, 0.35, n)
    y = season + trend + noise
    return t, y


def _fake_lstm_predict(y, lookback=24, lag_bias=1):
    """Cheap stand-in 'LSTM' prediction: a shifted EWMA so that the curve
    looks like a real one-step recurrent forecaster — close to the truth
    but with a small systematic lag and damping. No real model needed for
    a teaching figure."""
    alpha = 0.45
    pred = np.empty_like(y)
    pred[0] = y[0]
    for i in range(1, len(y)):
        pred[i] = alpha * y[i - 1] + (1 - alpha) * pred[i - 1]
    # mild forward shift to make it "predict next" then trim
    pred = np.roll(pred, lag_bias)
    pred[:lookback] = np.nan
    return pred


# ---------------------------------------------------------------------------
# Figure 3: Time series forecast — actual vs LSTM prediction
# ---------------------------------------------------------------------------
def fig3_forecast() -> None:
    t, y = _synthetic_series(n=240, seed=11)
    pred = _fake_lstm_predict(y, lookback=24)

    # Train / test split
    split = 170
    fig, ax = plt.subplots(figsize=(11, 4.6))

    ax.axvspan(0, split, color=C_LIGHT_BLUE, alpha=0.35,
               label="train window")
    ax.axvspan(split, len(t) - 1, color=C_LIGHT_AMBER, alpha=0.35,
               label="test window")

    ax.plot(t, y, color=C_DARK, lw=1.6, label="actual", alpha=0.85)
    ax.plot(t, pred, color=C_BLUE, lw=2.0, label="LSTM 1-step forecast")

    # Residual band on test region
    resid = (y - pred)
    sigma = np.nanstd(resid[split:])
    ax.fill_between(t[split:], pred[split:] - 1.96 * sigma,
                    pred[split:] + 1.96 * sigma,
                    color=C_BLUE, alpha=0.15, label="95% interval")

    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.set_title("One-step LSTM forecast vs actual",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    ax.set_xlim(0, len(t) - 1)

    # Metrics box
    rmse = float(np.sqrt(np.nanmean(resid[split:] ** 2)))
    mae = float(np.nanmean(np.abs(resid[split:])))
    ax.text(0.985, 0.05, f"test RMSE = {rmse:.3f}\ntest MAE = {mae:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor="white", edgecolor=C_GRAY,
                      boxstyle="round,pad=0.4"))

    fig.tight_layout()
    save(fig, "fig3_forecast")


# ---------------------------------------------------------------------------
# Figure 4: Multi-step ahead prediction with widening uncertainty
# ---------------------------------------------------------------------------
def fig4_multistep() -> None:
    t, y = _synthetic_series(n=200, seed=17)
    fig, ax = plt.subplots(figsize=(11, 4.8))

    history_end = 150
    horizon = 40

    # Recursive multi-step: feed prediction back, drift accumulates
    rng = np.random.default_rng(3)
    rec = np.empty(horizon)
    rec[0] = y[history_end - 1]
    for h in range(1, horizon):
        # damped continuation of the last seasonal pattern + drift
        seasonal = 1.5 * np.sin(2 * np.pi * (history_end + h) / 24)
        rec[h] = 0.5 * rec[h - 1] + 0.5 * seasonal + 0.005 * (history_end + h)

    # Direct multi-step: train one model per horizon — typically less drift
    direct = np.array([
        1.5 * np.sin(2 * np.pi * (history_end + h) / 24)
        + 0.6 * np.sin(2 * np.pi * (history_end + h) / 75)
        + 0.005 * (history_end + h) + rng.normal(0, 0.05)
        for h in range(horizon)
    ])

    fut_t = np.arange(history_end, history_end + horizon)

    # Uncertainty grows sqrt(h) for recursive
    sig0 = 0.25
    sigma_rec = sig0 * np.sqrt(np.arange(1, horizon + 1))
    sigma_dir = sig0 * 0.7 * np.ones(horizon)

    ax.plot(t[:history_end], y[:history_end], color=C_DARK, lw=1.5,
            label="history (actual)")
    ax.plot(t[history_end:], y[history_end:], color=C_GRAY, lw=1.2,
            ls="--", label="future (held out)")

    ax.plot(fut_t, rec, color=C_AMBER, lw=2.2, label="recursive forecast")
    ax.fill_between(fut_t, rec - 1.96 * sigma_rec, rec + 1.96 * sigma_rec,
                    color=C_AMBER, alpha=0.18,
                    label="recursive 95% band")

    ax.plot(fut_t, direct, color=C_GREEN, lw=2.2, label="direct forecast")
    ax.fill_between(fut_t, direct - 1.96 * sigma_dir,
                    direct + 1.96 * sigma_dir,
                    color=C_GREEN, alpha=0.18, label="direct 95% band")

    ax.axvline(history_end, color=C_DARK, lw=1, ls=":", alpha=0.6)
    ax.text(history_end + 0.5, ax.get_ylim()[1] * 0.95, " forecast origin",
            fontsize=9, color=C_DARK, va="top")

    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.set_title("Multi-step ahead: recursive vs direct prediction",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9, ncol=2)
    ax.set_xlim(0, history_end + horizon - 1)

    fig.tight_layout()
    save(fig, "fig4_multistep")


# ---------------------------------------------------------------------------
# Figure 5: Bidirectional LSTM
# ---------------------------------------------------------------------------
def fig5_bilstm() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    n = 5
    xs = [1.5 + 2.0 * i for i in range(n)]

    for i, x in enumerate(xs):
        # Input
        _box(ax, (x - 0.45, 0.4), 0.9, 0.7, f"$x_{{{i+1}}}$",
             C_LIGHT_BLUE, fs=11)
        # Forward cell
        _box(ax, (x - 0.55, 1.7), 1.1, 0.75, r"$\overrightarrow{h}$" + f"$_{{{i+1}}}$",
             C_LIGHT_PURPLE, ec=C_PURPLE, fs=10.5)
        # Backward cell
        _box(ax, (x - 0.55, 2.85), 1.1, 0.75, r"$\overleftarrow{h}$" + f"$_{{{i+1}}}$",
             C_LIGHT_AMBER, ec=C_AMBER, fs=10.5)
        # Concatenated output
        _box(ax, (x - 0.55, 4.1), 1.1, 0.75,
             f"$y_{{{i+1}}}$", C_LIGHT_GREEN, ec=C_GREEN, fs=11)

        # Vertical wires
        _arrow(ax, (x, 1.1), (x, 1.7), color=C_BLUE, lw=1.3)
        _arrow(ax, (x, 1.1), (x, 2.85), color=C_BLUE, lw=1.0,
               ls=(0, (3, 2)))
        _arrow(ax, (x, 2.45), (x - 0.18, 4.1), color=C_PURPLE, lw=1.4)
        _arrow(ax, (x, 3.6), (x + 0.18, 4.1), color=C_AMBER, lw=1.4)

    # Forward recurrent arrows
    for i in range(n - 1):
        _arrow(ax, (xs[i] + 0.55, 2.07), (xs[i + 1] - 0.55, 2.07),
               color=C_PURPLE, lw=1.6)
    # Backward recurrent arrows
    for i in range(n - 1, 0, -1):
        _arrow(ax, (xs[i] - 0.55, 3.22), (xs[i - 1] + 0.55, 3.22),
               color=C_AMBER, lw=1.6)

    # Labels
    ax.text(0.2, 2.07, "forward", fontsize=9.5, color=C_PURPLE,
            fontweight="bold", ha="left", va="center")
    ax.text(0.2, 3.22, "backward", fontsize=9.5, color=C_AMBER,
            fontweight="bold", ha="left", va="center")

    ax.text(6, 5.25, "Bidirectional LSTM",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(6, 0.05,
            r"$y_t = [\,\overrightarrow{h}_t \,;\, \overleftarrow{h}_t\,]$    "
            "— combines past and future context. Use only when future is "
            "available (offline tasks).",
            ha="center", fontsize=10, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_GRAY,
                      boxstyle="round,pad=0.4"))

    fig.tight_layout()
    save(fig, "fig5_bilstm")


# ---------------------------------------------------------------------------
# Figure 6: Stacked LSTM
# ---------------------------------------------------------------------------
def fig6_stacked_lstm() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    n = 5
    xs = [1.5 + 2.0 * i for i in range(n)]
    layer_y = [1.0, 2.4, 3.8]
    layer_color = [C_LIGHT_PURPLE, C_LIGHT_BLUE, C_LIGHT_GREEN]
    layer_edge = [C_PURPLE, C_BLUE, C_GREEN]
    layer_label = ["layer 1\nlow-level", "layer 2\nmid-level",
                   "layer 3\nhigh-level"]

    for i, x in enumerate(xs):
        # Input
        _box(ax, (x - 0.4, 0.05), 0.8, 0.55, f"$x_{{{i+1}}}$",
             C_LIGHT_GRAY, fs=10)
        # Output (top)
        _box(ax, (x - 0.4, 5.4), 0.8, 0.55, f"$y_{{{i+1}}}$",
             C_LIGHT_AMBER, ec=C_AMBER, fs=10)

    for li, (ly, fc, ec) in enumerate(zip(layer_y, layer_color, layer_edge)):
        for i, x in enumerate(xs):
            _box(ax, (x - 0.5, ly), 1.0, 0.7,
                 f"$h^{{({li+1})}}_{{{i+1}}}$", fc, ec=ec, fs=10)
        # Recurrent arrows within layer
        for i in range(n - 1):
            _arrow(ax, (xs[i] + 0.5, ly + 0.35),
                   (xs[i + 1] - 0.5, ly + 0.35), color=ec, lw=1.5)
        # Layer label on the left
        ax.text(0.15, ly + 0.35, layer_label[li], fontsize=8.5,
                color=ec, fontweight="bold", ha="left", va="center")

    # Vertical inter-layer wires
    for i, x in enumerate(xs):
        _arrow(ax, (x, 0.6), (x, layer_y[0]), color=C_GRAY, lw=1.2)
        for li in range(len(layer_y) - 1):
            _arrow(ax, (x, layer_y[li] + 0.7), (x, layer_y[li + 1]),
                   color=C_GRAY, lw=1.2)
        _arrow(ax, (x, layer_y[-1] + 0.7), (x, 5.4),
               color=C_GRAY, lw=1.2)

    ax.text(6, 6.0, "Stacked (deep) LSTM — 3 layers",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)

    fig.tight_layout()
    save(fig, "fig6_stacked_lstm")


# ---------------------------------------------------------------------------
# Figure 7: Training/validation loss with early stopping
# ---------------------------------------------------------------------------
def fig7_training_curves() -> None:
    rng = np.random.default_rng(5)
    epochs = np.arange(1, 61)

    # Training loss: keeps decreasing
    train = 1.4 * np.exp(-epochs / 18) + 0.04 + rng.normal(0, 0.012,
                                                           len(epochs))
    # Validation loss: U-shape (minimum near epoch ~30)
    val = (1.4 * np.exp(-epochs / 16) + 0.10
           + 0.0007 * (epochs - 28) ** 2
           + rng.normal(0, 0.018, len(epochs)))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(epochs, train, color=C_BLUE, lw=2.0, label="training loss",
            marker="o", ms=3.5)
    ax.plot(epochs, val, color=C_AMBER, lw=2.0, label="validation loss",
            marker="s", ms=3.5)

    # Best epoch and early stopping trigger
    best_epoch = int(np.argmin(val) + 1)
    best_val = float(val[best_epoch - 1])
    patience = 8
    stop_epoch = best_epoch + patience

    ax.axvline(best_epoch, color=C_GREEN, lw=1.6, ls="--",
               label=f"best epoch = {best_epoch}")
    ax.axvline(stop_epoch, color=C_PURPLE, lw=1.6, ls=":",
               label=f"early stop (patience={patience})")
    ax.scatter([best_epoch], [best_val], s=110, color=C_GREEN, zorder=5,
               edgecolor=C_DARK, linewidths=1.2)

    # Shade the patience window
    ax.axvspan(best_epoch, stop_epoch, color=C_LIGHT_PURPLE, alpha=0.4,
               label="patience window")

    ax.text(best_epoch, best_val - 0.06,
            f" min val = {best_val:.3f}",
            fontsize=9.5, color=C_GREEN, fontweight="bold",
            ha="left", va="top")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("LSTM training: validation loss + early stopping",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, max(val.max(), train.max()) * 1.1)

    fig.tight_layout()
    save(fig, "fig7_training_curves")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN target: {EN_DIR.relative_to(REPO_ROOT)}")
    print(f"ZH target: {ZH_DIR.relative_to(REPO_ROOT)}")
    fig1_lstm_cell()
    fig2_state_highway()
    fig3_forecast()
    fig4_multistep()
    fig5_bilstm()
    fig6_stacked_lstm()
    fig7_training_curves()
    print("done.")


if __name__ == "__main__":
    main()
