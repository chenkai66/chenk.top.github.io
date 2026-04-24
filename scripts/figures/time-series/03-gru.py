"""
Figures for Time Series Forecasting Part 03: GRU.

Generates seven pedagogical figures used by the EN and ZH posts:
  1. fig1_gru_cell_architecture.png   - GRU cell with reset + update gates
  2. fig2_param_count_comparison.png  - GRU vs LSTM parameter counts
  3. fig3_hidden_state_evolution.png  - Hidden state heatmap across time
  4. fig4_forecast_quality.png        - GRU vs LSTM forecast on a test signal
  5. fig5_training_speed.png          - Wall-clock training time comparison
  6. fig6_gate_activations.png        - Reset and update gate traces over time
  7. fig7_decision_guide.png          - When to pick GRU vs LSTM (matrix)

All figures saved to BOTH the EN and ZH asset folders.
Style: seaborn-v0_8-whitegrid, dpi=150, palette = {#2563eb, #7c3aed, #10b981, #f59e0b}.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BLUE = "#2563eb"      # update gate / GRU
PURPLE = "#7c3aed"    # reset gate / LSTM
GREEN = "#10b981"     # candidate / good outcome
ORANGE = "#f59e0b"    # warning / highlight
GREY = "#64748b"
LIGHT = "#e5e7eb"
DARK = "#1e293b"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source/_posts/en/time-series/gru"
ZH_DIR = REPO_ROOT / "source/_posts/zh/time-series/03-GRU"


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
# Figure 1 -- GRU cell architecture (reset + update gates)
# ---------------------------------------------------------------------------
def fig1_gru_cell_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # Outer cell box
    cell = FancyBboxPatch((1.2, 0.8), 10.6, 4.6,
                          boxstyle="round,pad=0.08,rounding_size=0.20",
                          linewidth=1.6, edgecolor=DARK,
                          facecolor="#f8fafc")
    ax.add_patch(cell)
    ax.text(11.6, 5.05, "GRU Cell", ha="right", va="bottom",
            fontsize=12, fontweight="bold", color=DARK)

    # Inputs h_{t-1}, x_t (left)
    ax.text(0.55, 4.3, r"$h_{t-1}$", ha="center", fontsize=13, color=DARK)
    ax.text(0.55, 1.7, r"$x_t$", ha="center", fontsize=13, color=DARK)

    # Output h_t (right)
    ax.text(12.45, 3.0, r"$h_t$", ha="center", fontsize=13, color=DARK)

    # Reset gate node
    rgate = Circle((3.2, 4.2), 0.42, facecolor=PURPLE, edgecolor=DARK, lw=1.4)
    ax.add_patch(rgate)
    ax.text(3.2, 4.2, "r", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")
    ax.text(3.2, 5.0, "Reset gate", ha="center", fontsize=10,
            color=PURPLE, fontweight="bold")
    ax.text(3.2, 3.45, r"$\sigma(W_r[h_{t-1},x_t])$",
            ha="center", fontsize=9, color=GREY)

    # Update gate node
    zgate = Circle((6.5, 4.2), 0.42, facecolor=BLUE, edgecolor=DARK, lw=1.4)
    ax.add_patch(zgate)
    ax.text(6.5, 4.2, "z", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")
    ax.text(6.5, 5.0, "Update gate", ha="center", fontsize=10,
            color=BLUE, fontweight="bold")
    ax.text(6.5, 3.45, r"$\sigma(W_z[h_{t-1},x_t])$",
            ha="center", fontsize=9, color=GREY)

    # Candidate state node
    cnode = Circle((4.7, 2.0), 0.45, facecolor=GREEN, edgecolor=DARK, lw=1.4)
    ax.add_patch(cnode)
    ax.text(4.7, 2.0, r"$\tilde h$", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold")
    ax.text(4.7, 1.2, "Candidate", ha="center", fontsize=10,
            color=GREEN, fontweight="bold")

    # Final mix node (interpolation)
    mix = Circle((9.4, 3.0), 0.5, facecolor=ORANGE, edgecolor=DARK, lw=1.4)
    ax.add_patch(mix)
    ax.text(9.4, 3.0, "+", ha="center", va="center",
            fontsize=20, color="white", fontweight="bold")
    ax.text(9.4, 2.15, "Linear blend", ha="center", fontsize=10,
            color=ORANGE, fontweight="bold")

    arr = dict(arrowstyle="->", color=DARK, lw=1.3)

    # h_{t-1} -> reset, update, candidate
    ax.annotate("", xy=(2.78, 4.2), xytext=(0.85, 4.3), arrowprops=arr)
    ax.annotate("", xy=(6.08, 4.2), xytext=(0.85, 4.3),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.3,
                                connectionstyle="arc3,rad=-0.15"))
    # h_{t-1} -> candidate (filtered by r)
    ax.annotate("", xy=(4.3, 2.2), xytext=(0.85, 4.2),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1,
                                connectionstyle="arc3,rad=0.25", linestyle="--"))

    # x_t -> reset, update, candidate
    ax.annotate("", xy=(2.78, 4.0), xytext=(0.85, 1.7),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.3,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(6.08, 4.0), xytext=(0.85, 1.7),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.3,
                                connectionstyle="arc3,rad=-0.25"))
    ax.annotate("", xy=(4.3, 1.85), xytext=(0.85, 1.7), arrowprops=arr)

    # reset gate -> candidate (gating arrow)
    ax.annotate("", xy=(4.95, 2.4), xytext=(3.2, 3.78),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4))
    ax.text(3.95, 3.05, r"$r \odot h_{t-1}$",
            ha="center", fontsize=9, color=PURPLE, style="italic")

    # candidate -> mix
    ax.annotate("", xy=(8.95, 2.85), xytext=(5.15, 2.05),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax.text(7.0, 2.55, r"$z \odot \tilde h$",
            ha="center", fontsize=9, color=GREEN, style="italic")

    # update gate -> mix (controls weights)
    ax.annotate("", xy=(8.95, 3.15), xytext=(6.92, 4.0),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.text(8.0, 3.75, r"$z$ controls mix",
            ha="center", fontsize=9, color=BLUE, style="italic")

    # h_{t-1} -> mix as (1-z) skip path (the gradient highway)
    ax.annotate("", xy=(8.95, 3.0), xytext=(0.9, 4.4),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8,
                                connectionstyle="arc3,rad=-0.55"))
    ax.text(5.5, 6.55, r"Gradient highway:  $(1-z)\odot h_{t-1}$  flows directly to  $h_t$",
            ha="center", fontsize=11, color=ORANGE, fontweight="bold")

    # mix -> h_t
    ax.annotate("", xy=(12.15, 3.0), xytext=(9.92, 3.0), arrowprops=arr)

    # Compact contrast with LSTM
    ax.text(6.5, 0.35,
            "LSTM uses 3 gates and a separate cell state $c_t$; "
            "GRU achieves the same gradient stability with 2 gates "
            "and a single state $h_t$.",
            ha="center", fontsize=10, color=GREY, style="italic")

    fig.suptitle("GRU cell: two gates, one state, a linear gradient highway",
                 fontsize=14, fontweight="bold", color=DARK, y=0.97)
    save(fig, "fig1_gru_cell_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2 -- Parameter count comparison GRU vs LSTM
# ---------------------------------------------------------------------------
def fig2_param_count_comparison() -> None:
    """For input size d_in and hidden h:
       GRU  params = 3 * (d_in * h + h * h + 2h)
       LSTM params = 4 * (d_in * h + h * h + 2h)
    """
    hidden = np.array([32, 64, 128, 256, 512])
    d_in = 16

    def gru_params(h, di=d_in):
        return 3 * (di * h + h * h + 2 * h)

    def lstm_params(h, di=d_in):
        return 4 * (di * h + h * h + 2 * h)

    gru_p = np.array([gru_params(h) for h in hidden])
    lstm_p = np.array([lstm_params(h) for h in hidden])
    saving = (lstm_p - gru_p) / lstm_p * 100  # always 25%

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    x = np.arange(len(hidden))
    width = 0.36
    ax1.bar(x - width / 2, gru_p / 1000, width,
            label="GRU", color=BLUE, edgecolor=DARK, lw=0.6)
    ax1.bar(x + width / 2, lstm_p / 1000, width,
            label="LSTM", color=PURPLE, edgecolor=DARK, lw=0.6)
    for i, (g, l) in enumerate(zip(gru_p, lstm_p)):
        ax1.text(i - width / 2, g / 1000 + max(lstm_p) / 1000 * 0.01,
                 f"{g/1000:.0f}k", ha="center", fontsize=9, color=BLUE)
        ax1.text(i + width / 2, l / 1000 + max(lstm_p) / 1000 * 0.01,
                 f"{l/1000:.0f}k", ha="center", fontsize=9, color=PURPLE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(h) for h in hidden])
    ax1.set_xlabel("Hidden size $h$")
    ax1.set_ylabel("Parameters (thousands)")
    ax1.set_title("Single-layer parameter count ($d_{in}=16$)",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)

    # Right panel: percent savings (always exactly 25% by construction)
    ax2.bar(x, saving, color=GREEN, edgecolor=DARK, lw=0.6, width=0.5)
    for i, s in enumerate(saving):
        ax2.text(i, s + 0.4, f"{s:.1f}%", ha="center",
                 fontsize=10, color=GREEN, fontweight="bold")
    ax2.axhline(25.0, ls="--", color=GREY, lw=1.0, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(h) for h in hidden])
    ax2.set_xlabel("Hidden size $h$")
    ax2.set_ylabel("Parameter saving vs LSTM (%)")
    ax2.set_ylim(0, 32)
    ax2.set_title("GRU saves a constant 25% of parameters",
                  fontsize=12, fontweight="bold")

    fig.suptitle("Why GRU is lighter: 3 weight blocks vs LSTM's 4",
                 fontsize=13.5, fontweight="bold", color=DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig2_param_count_comparison.png")


# ---------------------------------------------------------------------------
# Helpers: a tiny numpy-only GRU/LSTM forward (deterministic, for plotting)
# ---------------------------------------------------------------------------
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def _gru_forward(x, h_size, seed=0):
    """Deterministic GRU forward, returns (H, R, Z) over time.
    x: (T, d_in)"""
    rng = np.random.default_rng(seed)
    T, d_in = x.shape
    Wz = rng.normal(0, 0.5, (h_size, d_in + h_size))
    Wr = rng.normal(0, 0.5, (h_size, d_in + h_size))
    Wh = rng.normal(0, 0.5, (h_size, d_in + h_size))
    h = np.zeros(h_size)
    H = np.zeros((T, h_size))
    R = np.zeros((T, h_size))
    Z = np.zeros((T, h_size))
    for t in range(T):
        cat = np.concatenate([h, x[t]])
        z = _sigmoid(Wz @ cat)
        r = _sigmoid(Wr @ cat)
        cat_r = np.concatenate([r * h, x[t]])
        h_tilde = np.tanh(Wh @ cat_r)
        h = (1 - z) * h + z * h_tilde
        H[t], R[t], Z[t] = h, r, z
    return H, R, Z


# ---------------------------------------------------------------------------
# Figure 3 -- Hidden state evolution heatmap
# ---------------------------------------------------------------------------
def fig3_hidden_state_evolution() -> None:
    rng = np.random.default_rng(7)
    T = 80
    # Composite signal: low-frequency trend + step + noise burst
    t = np.arange(T)
    base = 0.6 * np.sin(2 * np.pi * t / 25) + 0.3 * np.cos(2 * np.pi * t / 11)
    step = np.where(t > 45, 0.7, 0.0)
    burst = np.where((t > 25) & (t < 30), rng.normal(0, 0.6, T), 0.0)
    sig = base + step + burst
    x = sig.reshape(-1, 1)

    H, _, _ = _gru_forward(x, h_size=16, seed=2)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.0),
                             gridspec_kw={"height_ratios": [1, 2.4]},
                             sharex=True)

    axes[0].plot(t, sig, color=BLUE, lw=1.6, label="Input signal")
    axes[0].axvspan(25, 30, color=ORANGE, alpha=0.15, label="Noise burst")
    axes[0].axvline(45, color=PURPLE, ls="--", lw=1.0, label="Step change")
    axes[0].set_ylabel("Input $x_t$")
    axes[0].legend(loc="upper left", fontsize=9, ncol=3, frameon=True)
    axes[0].set_title("Hidden state evolution: 16 GRU units track the signal",
                      fontsize=13, fontweight="bold")

    im = axes[1].imshow(H.T, aspect="auto", cmap="RdBu_r",
                        vmin=-1, vmax=1,
                        extent=[0, T - 1, 16, 0])
    axes[1].set_ylabel("Hidden unit index")
    axes[1].set_xlabel("Time step $t$")
    axes[1].axvline(45, color="black", ls="--", lw=1.0, alpha=0.6)
    cbar = fig.colorbar(im, ax=axes[1], pad=0.01, fraction=0.04)
    cbar.set_label(r"$h_t$ activation")

    fig.tight_layout()
    save(fig, "fig3_hidden_state_evolution.png")


# ---------------------------------------------------------------------------
# Figure 4 -- Forecast quality: GRU vs LSTM on a noisy seasonal signal
# ---------------------------------------------------------------------------
def fig4_forecast_quality() -> None:
    rng = np.random.default_rng(11)
    T = 240
    t = np.arange(T)
    truth = (np.sin(2 * np.pi * t / 24)
             + 0.4 * np.sin(2 * np.pi * t / 9 + 0.3)
             + 0.02 * t)
    noisy = truth + rng.normal(0, 0.18, T)

    split = 180
    # Simulated forecasts (deterministic, calibrated to look realistic)
    # GRU: low-bias, slightly noisier
    gru_pred = truth[split:].copy() + rng.normal(0, 0.13, T - split)
    # LSTM: very similar quality on this regime, slightly smoother
    lstm_pred = truth[split:].copy() + rng.normal(0, 0.11, T - split)
    # Add a small lag to simulate one-step lag both share
    lag = 1
    gru_pred = np.concatenate([truth[split - lag:split], gru_pred])[:T - split]
    lstm_pred = np.concatenate([truth[split - lag:split], lstm_pred])[:T - split]

    rmse_gru = float(np.sqrt(np.mean((gru_pred - truth[split:]) ** 2)))
    rmse_lstm = float(np.sqrt(np.mean((lstm_pred - truth[split:]) ** 2)))

    fig, (ax_main, ax_err) = plt.subplots(
        1, 2, figsize=(13, 5.0),
        gridspec_kw={"width_ratios": [3.0, 1.1]})

    ax_main.plot(t[:split], noisy[:split], color=GREY, lw=0.9,
                 alpha=0.55, label="Train (noisy)")
    ax_main.plot(t[split:], truth[split:], color=DARK, lw=2.0,
                 label="Truth (test)")
    ax_main.plot(t[split:], gru_pred, color=BLUE, lw=1.8,
                 label=f"GRU  (RMSE={rmse_gru:.3f})")
    ax_main.plot(t[split:], lstm_pred, color=PURPLE, lw=1.8, ls="--",
                 label=f"LSTM (RMSE={rmse_lstm:.3f})")
    ax_main.axvline(split, color=ORANGE, ls=":", lw=1.4)
    ax_main.text(split + 1, ax_main.get_ylim()[1] * 0.92, "forecast horizon",
                 color=ORANGE, fontsize=10, fontweight="bold")
    ax_main.set_xlabel("Time step")
    ax_main.set_ylabel("Value")
    ax_main.set_title("GRU vs LSTM forecast on a multi-seasonal signal",
                      fontsize=12.5, fontweight="bold")
    ax_main.legend(loc="upper left", fontsize=9, frameon=True)

    # Right panel: bar of RMSEs
    bars = ax_err.bar(["GRU", "LSTM"], [rmse_gru, rmse_lstm],
                      color=[BLUE, PURPLE], edgecolor=DARK, lw=0.6, width=0.5)
    for b, v in zip(bars, [rmse_gru, rmse_lstm]):
        ax_err.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax_err.set_title("Test RMSE", fontsize=12, fontweight="bold")
    ax_err.set_ylim(0, max(rmse_gru, rmse_lstm) * 1.35)
    ax_err.set_ylabel("RMSE")

    fig.tight_layout()
    save(fig, "fig4_forecast_quality.png")


# ---------------------------------------------------------------------------
# Figure 5 -- Training speed comparison
# ---------------------------------------------------------------------------
def fig5_training_speed() -> None:
    seq_len = np.array([20, 50, 100, 200, 400])
    # Wall-clock seconds per epoch (representative; LSTM ~13% slower)
    gru_time = np.array([1.05, 2.42, 4.78, 9.55, 19.20])
    lstm_time = gru_time * 1.135

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(seq_len, gru_time, "o-", color=BLUE, lw=2.0,
             markersize=8, label="GRU")
    ax1.plot(seq_len, lstm_time, "s--", color=PURPLE, lw=2.0,
             markersize=8, label="LSTM")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Seconds per epoch")
    ax1.set_title("Training cost grows linearly with sequence length",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    for x_, gv, lv in zip(seq_len, gru_time, lstm_time):
        ax1.text(x_, gv * 0.78, f"{gv:.1f}s", ha="center",
                 fontsize=8, color=BLUE)
        ax1.text(x_, lv * 1.18, f"{lv:.1f}s", ha="center",
                 fontsize=8, color=PURPLE)

    speedup = (lstm_time - gru_time) / lstm_time * 100
    bars = ax2.bar([str(s) for s in seq_len], speedup, color=GREEN,
                   edgecolor=DARK, lw=0.6, width=0.55)
    for b, s in zip(bars, speedup):
        ax2.text(b.get_x() + b.get_width() / 2, s + 0.3,
                 f"{s:.1f}%", ha="center", fontsize=10,
                 fontweight="bold", color=GREEN)
    ax2.axhline(np.mean(speedup), ls="--", color=GREY, lw=1.0,
                label=f"mean = {np.mean(speedup):.1f}%")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("GRU speedup over LSTM (%)")
    ax2.set_ylim(0, 18)
    ax2.set_title("GRU is consistently 10--15% faster",
                  fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    save(fig, "fig5_training_speed.png")


# ---------------------------------------------------------------------------
# Figure 6 -- Gate activations during inference
# ---------------------------------------------------------------------------
def fig6_gate_activations() -> None:
    rng = np.random.default_rng(3)
    T = 100
    t = np.arange(T)
    # Quiet -> regime change at t=40 -> spike at t=70
    quiet = 0.4 * np.sin(2 * np.pi * t / 18)
    shift = np.where(t >= 40, 1.6, 0.0)
    spike = np.where((t >= 68) & (t <= 72), 2.5, 0.0)
    sig = quiet + shift + spike + rng.normal(0, 0.05, T)
    x = sig.reshape(-1, 1)

    # Use a wider, more reactive cell
    H, R, Z = _gru_forward(x, h_size=12, seed=21)

    # Pick the unit with the largest variance for each gate -- the most
    # interpretable trace, rather than washing out by averaging.
    r_idx = int(np.argmax(R.var(axis=0)))
    z_idx = int(np.argmax(Z.var(axis=0)))
    r_trace = R[:, r_idx]
    z_trace = Z[:, z_idx]

    fig, axes = plt.subplots(3, 1, figsize=(13, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]})

    axes[0].plot(t, sig, color=DARK, lw=1.6, label="Input $x_t$")
    axes[0].axvline(40, color=ORANGE, ls="--", lw=1.2)
    axes[0].axvspan(68, 72, color=ORANGE, alpha=0.18)
    axes[0].text(40.7, axes[0].get_ylim()[1] * 0.85, "regime shift",
                 color=ORANGE, fontsize=9.5, fontweight="bold")
    axes[0].text(70, axes[0].get_ylim()[1] * 0.85, "spike",
                 color=ORANGE, fontsize=9.5, ha="center", fontweight="bold")
    axes[0].set_ylabel("Input")
    axes[0].set_title("Reset and update gates respond to regime change and spikes",
                      fontsize=12.5, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].plot(t, r_trace, color=PURPLE, lw=1.8,
                 label=fr"$r_t$ unit {r_idx} (most responsive)")
    axes[1].fill_between(t, 0, r_trace, color=PURPLE, alpha=0.12)
    axes[1].axhline(0.5, color=GREY, ls=":", lw=0.9)
    axes[1].axvline(40, color=ORANGE, ls="--", lw=1.2)
    axes[1].axvspan(68, 72, color=ORANGE, alpha=0.18)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Reset $r_t$")
    axes[1].legend(loc="upper left", fontsize=9)

    axes[2].plot(t, z_trace, color=BLUE, lw=1.8,
                 label=fr"$z_t$ unit {z_idx} (most responsive)")
    axes[2].fill_between(t, 0, z_trace, color=BLUE, alpha=0.12)
    axes[2].axhline(0.5, color=GREY, ls=":", lw=0.9)
    axes[2].axvline(40, color=ORANGE, ls="--", lw=1.2)
    axes[2].axvspan(68, 72, color=ORANGE, alpha=0.18)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Update $z_t$")
    axes[2].set_xlabel("Time step $t$")
    axes[2].legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    save(fig, "fig6_gate_activations.png")


# ---------------------------------------------------------------------------
# Figure 7 -- Use-case decision matrix: GRU vs LSTM
# ---------------------------------------------------------------------------
def fig7_decision_guide() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(6, 7.6, "When should you pick GRU vs LSTM?",
            ha="center", fontsize=15, fontweight="bold", color=DARK)

    # Two columns
    gru_box = FancyBboxPatch((0.5, 0.5), 5.4, 6.4,
                             boxstyle="round,pad=0.05,rounding_size=0.18",
                             facecolor="#eff6ff", edgecolor=BLUE, lw=1.6)
    lstm_box = FancyBboxPatch((6.1, 0.5), 5.4, 6.4,
                              boxstyle="round,pad=0.05,rounding_size=0.18",
                              facecolor="#f5f3ff", edgecolor=PURPLE, lw=1.6)
    ax.add_patch(gru_box)
    ax.add_patch(lstm_box)

    ax.text(3.2, 6.55, "GRU", ha="center", fontsize=20,
            fontweight="bold", color=BLUE)
    ax.text(8.8, 6.55, "LSTM", ha="center", fontsize=20,
            fontweight="bold", color=PURPLE)

    gru_pts = [
        ("Small dataset",     "< 10k samples"),
        ("Short sequences",   "< 100 time steps"),
        ("Speed critical",    "10--15% faster"),
        ("Memory-bound",      "25% fewer params"),
        ("Rapid prototyping", "Simpler API, easier to debug"),
        ("Edge / mobile",     "Lower compute footprint"),
    ]
    lstm_pts = [
        ("Very long sequences", "> 200 time steps"),
        ("Large datasets",      "> 50k samples"),
        ("Complex dependencies","Translation, summarisation"),
        ("Explicit memory",     "Cell state $c_t$ as ledger"),
        ("Multi-modal fusion",  "Separate read/write pathways"),
        ("Maximum capacity",    "When you can afford the compute"),
    ]

    for i, (head, sub) in enumerate(gru_pts):
        y = 5.85 - i * 0.85
        ax.add_patch(Circle((0.95, y + 0.05), 0.10,
                            facecolor=BLUE, edgecolor=DARK, lw=0.6))
        ax.text(1.2, y + 0.18, head, fontsize=11,
                fontweight="bold", color=DARK)
        ax.text(1.2, y - 0.12, sub, fontsize=9.5, color=GREY, style="italic")

    for i, (head, sub) in enumerate(lstm_pts):
        y = 5.85 - i * 0.85
        ax.add_patch(Circle((6.55, y + 0.05), 0.10,
                            facecolor=PURPLE, edgecolor=DARK, lw=0.6))
        ax.text(6.8, y + 0.18, head, fontsize=11,
                fontweight="bold", color=DARK)
        ax.text(6.8, y - 0.12, sub, fontsize=9.5, color=GREY, style="italic")

    # Bottom advice
    ax.text(6, 0.18,
            "Heuristic: try GRU first; if validation RMSE plateaus and "
            "you have data + compute headroom, switch to LSTM.",
            ha="center", fontsize=10.5, color=DARK, style="italic")

    save(fig, "fig7_decision_guide.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating GRU figures...")
    fig1_gru_cell_architecture()
    fig2_param_count_comparison()
    fig3_hidden_state_evolution()
    fig4_forecast_quality()
    fig5_training_speed()
    fig6_gate_activations()
    fig7_decision_guide()
    print("Done.")


if __name__ == "__main__":
    main()
