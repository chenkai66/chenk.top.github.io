"""
Figure generation script for Time Series Part 01: Traditional Statistical Models.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure teaches a single concept cleanly and is rendered to BOTH article asset
folders so markdown references stay in sync across languages.

Figures:
    fig1_components             Classical decomposition of a synthetic monthly
                                series into trend, seasonality and residual.
    fig2_ar_ma_arma             Three-panel comparison of AR(1), MA(1) and
                                ARMA(1,1) sample paths showing the qualitative
                                difference in dynamics.
    fig3_acf_pacf               ACF and PACF of an AR(2) and an MA(2) process,
                                illustrating the cut-off / decay rules used to
                                identify p and q.
    fig4_arima_forecast         ARIMA forecast on a held-out test segment with
                                95% confidence band, compared to the actual.
    fig5_sarima_forecast        SARIMA forecast on a series with strong yearly
                                seasonality, demonstrating recovery of the
                                seasonal pattern out of sample.
    fig6_box_jenkins_flow       Schematic of the Box-Jenkins identification
                                -> estimation -> diagnostic-checking loop.
    fig7_holt_winters           Holt-Winters additive forecast (level, trend
                                and seasonal components) on a seasonal series.

Usage:
    python3 scripts/figures/time-series/01-traditional.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Shared style ----------------------------------------------------------------
import sys
from pathlib import Path as _StylePath
sys.path.insert(0, str(_StylePath(__file__).parent.parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
# Color palette (from shared _style)
COLOR_BLUE = COLORS["primary"]
COLOR_PURPLE = COLORS["accent"]
COLOR_GREEN = COLORS["success"]
COLOR_AMBER = COLORS["warning"]


DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "time-series" / "01-traditional-models"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "time-series" / "01-传统模型"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (EN_DIR, ZH_DIR):
        fig.savefig(out_dir / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic data generators (kept reproducible across figures)
# ---------------------------------------------------------------------------

def _seasonal_series(
    n_periods: int = 120,
    seed: int = 7,
    trend: float = 0.25,
    season_amp: float = 6.0,
    noise_std: float = 1.0,
) -> pd.Series:
    """Monthly series with linear trend + 12-month seasonality + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)
    season = season_amp * np.sin(2 * np.pi * t / 12)
    drift = trend * t
    noise = rng.normal(0, noise_std, n_periods)
    values = 20 + drift + season + noise
    index = pd.date_range("2014-01-01", periods=n_periods, freq="MS")
    return pd.Series(values, index=index, name="y")


def _ar_series(phi: list[float], n: int = 300, seed: int = 1) -> np.ndarray:
    """Sample path of an AR(p) process with given coefficients."""
    rng = np.random.default_rng(seed)
    p = len(phi)
    y = np.zeros(n)
    eps = rng.normal(0, 1.0, n)
    for t in range(p, n):
        y[t] = sum(phi[k] * y[t - k - 1] for k in range(p)) + eps[t]
    return y


def _ma_series(theta: list[float], n: int = 300, seed: int = 2) -> np.ndarray:
    """Sample path of an MA(q) process with given coefficients."""
    rng = np.random.default_rng(seed)
    q = len(theta)
    eps = rng.normal(0, 1.0, n + q)
    y = np.zeros(n)
    for t in range(n):
        y[t] = eps[t + q] + sum(theta[k] * eps[t + q - k - 1] for k in range(q))
    return y


def _arma_series(phi: list[float], theta: list[float], n: int = 300, seed: int = 3) -> np.ndarray:
    """Sample path of an ARMA(p, q) process."""
    rng = np.random.default_rng(seed)
    p, q = len(phi), len(theta)
    eps = rng.normal(0, 1.0, n + q)
    y = np.zeros(n)
    for t in range(max(p, q), n):
        ar_part = sum(phi[k] * y[t - k - 1] for k in range(p))
        ma_part = sum(theta[k] * eps[t + q - k - 1] for k in range(q))
        y[t] = ar_part + ma_part + eps[t + q]
    return y


# ---------------------------------------------------------------------------
# Figure 1: Components decomposition (trend + seasonality + residual)
# ---------------------------------------------------------------------------

def fig1_components() -> None:
    series = _seasonal_series(n_periods=120, seed=7)
    decomp = seasonal_decompose(series, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(series.index, series.values, color=COLOR_BLUE, lw=1.5)
    axes[0].set_title("Observed series  $y_t$", loc="left", fontsize=12, fontweight="bold")

    axes[1].plot(decomp.trend.index, decomp.trend.values, color=COLOR_PURPLE, lw=1.8)
    axes[1].set_title("Trend  $T_t$", loc="left", fontsize=12, fontweight="bold")

    axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color=COLOR_GREEN, lw=1.5)
    axes[2].set_title("Seasonality  $S_t$  (period = 12)", loc="left", fontsize=12, fontweight="bold")

    axes[3].plot(decomp.resid.index, decomp.resid.values, color=COLOR_AMBER, lw=1.0)
    axes[3].axhline(0, color="grey", lw=0.6, ls="--")
    axes[3].set_title("Residual  $R_t = y_t - T_t - S_t$", loc="left", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("date")

    fig.suptitle(
        "Classical additive decomposition: $y_t = T_t + S_t + R_t$",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    _save(fig, "fig1_components.png")


# ---------------------------------------------------------------------------
# Figure 2: AR vs MA vs ARMA sample paths
# ---------------------------------------------------------------------------

def fig2_ar_ma_arma() -> None:
    n = 200
    ar = _ar_series([0.85], n=n, seed=11)
    ma = _ma_series([0.9, 0.6], n=n, seed=12)
    arma = _arma_series([0.6], [0.5], n=n, seed=13)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(ar, color=COLOR_BLUE, lw=1.2)
    axes[0].set_title(
        "AR(1):  $y_t = 0.85\\, y_{t-1} + \\varepsilon_t$  "
        "(persistent, slowly mean-reverting)",
        loc="left", fontsize=11, fontweight="bold",
    )
    axes[0].axhline(0, color="grey", lw=0.6, ls="--")

    axes[1].plot(ma, color=COLOR_PURPLE, lw=1.2)
    axes[1].set_title(
        "MA(2):  $y_t = \\varepsilon_t + 0.9\\,\\varepsilon_{t-1} + 0.6\\,\\varepsilon_{t-2}$  "
        "(short memory, finite shock window)",
        loc="left", fontsize=11, fontweight="bold",
    )
    axes[1].axhline(0, color="grey", lw=0.6, ls="--")

    axes[2].plot(arma, color=COLOR_GREEN, lw=1.2)
    axes[2].set_title(
        "ARMA(1,1):  AR persistence + MA shock smoothing combined",
        loc="left", fontsize=11, fontweight="bold",
    )
    axes[2].axhline(0, color="grey", lw=0.6, ls="--")
    axes[2].set_xlabel("time step  $t$")

    fig.suptitle(
        "AR remembers values, MA remembers shocks, ARMA does both",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    _save(fig, "fig2_ar_ma_arma.png")


# ---------------------------------------------------------------------------
# Figure 3: ACF and PACF for AR(2) vs MA(2)
# ---------------------------------------------------------------------------

def fig3_acf_pacf() -> None:
    ar2 = _ar_series([0.6, -0.3], n=500, seed=21)
    ma2 = _ma_series([0.7, 0.4], n=500, seed=22)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))

    plot_acf(ar2, lags=20, ax=axes[0, 0], color=COLOR_BLUE, vlines_kwargs={"colors": COLOR_BLUE})
    axes[0, 0].set_title("AR(2):  ACF decays gradually", fontsize=11, fontweight="bold")

    plot_pacf(ar2, lags=20, ax=axes[0, 1], method="ywm", color=COLOR_PURPLE,
              vlines_kwargs={"colors": COLOR_PURPLE})
    axes[0, 1].set_title("AR(2):  PACF cuts off at lag 2", fontsize=11, fontweight="bold")

    plot_acf(ma2, lags=20, ax=axes[1, 0], color=COLOR_BLUE, vlines_kwargs={"colors": COLOR_BLUE})
    axes[1, 0].set_title("MA(2):  ACF cuts off at lag 2", fontsize=11, fontweight="bold")

    plot_pacf(ma2, lags=20, ax=axes[1, 1], method="ywm", color=COLOR_PURPLE,
              vlines_kwargs={"colors": COLOR_PURPLE})
    axes[1, 1].set_title("MA(2):  PACF decays gradually", fontsize=11, fontweight="bold")

    for ax in axes.flat:
        ax.set_xlabel("lag")

    fig.suptitle(
        "Box-Jenkins rule of thumb: PACF identifies p, ACF identifies q",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    _save(fig, "fig3_acf_pacf.png")


# ---------------------------------------------------------------------------
# Figure 4: ARIMA forecast vs actual on a held-out segment
# ---------------------------------------------------------------------------

def fig4_arima_forecast() -> None:
    # Generate a non-stationary series (random walk + drift) so I(1) is meaningful
    rng = np.random.default_rng(33)
    n = 200
    eps = rng.normal(0, 1.0, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] + 0.05 + 0.6 * eps[t]
    index = pd.date_range("2018-01-01", periods=n, freq="D")
    series = pd.Series(y, index=index)

    train = series.iloc[:-30]
    test = series.iloc[-30:]

    model = ARIMA(train, order=(2, 1, 1)).fit()
    fc = model.get_forecast(steps=30)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(train.index, train.values, color=COLOR_BLUE, lw=1.3, label="Training history")
    ax.plot(test.index, test.values, color=COLOR_AMBER, lw=1.6, label="Actual (held out)")
    ax.plot(mean_fc.index, mean_fc.values, color=COLOR_PURPLE, lw=1.8,
            label="ARIMA(2,1,1) forecast")
    ax.fill_between(
        mean_fc.index,
        ci.iloc[:, 0].values, ci.iloc[:, 1].values,
        color=COLOR_PURPLE, alpha=0.18, label="95% confidence interval",
    )
    ax.axvline(train.index[-1], color="grey", lw=0.7, ls="--")

    ax.set_title(
        "ARIMA(2,1,1) point forecast and uncertainty band",
        fontsize=13, fontweight="bold", loc="left",
    )
    ax.set_xlabel("date")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    _save(fig, "fig4_arima_forecast.png")


# ---------------------------------------------------------------------------
# Figure 5: SARIMA forecast on seasonal series
# ---------------------------------------------------------------------------

def fig5_sarima_forecast() -> None:
    series = _seasonal_series(n_periods=120, seed=51, trend=0.18, season_amp=8.0, noise_std=0.8)
    train = series.iloc[:-24]
    test = series.iloc[-24:]

    model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)

    fc = model.get_forecast(steps=24)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(train.index, train.values, color=COLOR_BLUE, lw=1.3, label="Training history")
    ax.plot(test.index, test.values, color=COLOR_AMBER, lw=1.8, label="Actual (held out)")
    ax.plot(mean_fc.index, mean_fc.values, color=COLOR_PURPLE, lw=1.8,
            label="SARIMA(1,1,1)(1,1,1,12) forecast")
    ax.fill_between(
        mean_fc.index,
        ci.iloc[:, 0].values, ci.iloc[:, 1].values,
        color=COLOR_PURPLE, alpha=0.18, label="95% confidence interval",
    )
    ax.axvline(train.index[-1], color="grey", lw=0.7, ls="--")

    ax.set_title(
        "SARIMA recovers the 12-month seasonal cycle out of sample",
        fontsize=13, fontweight="bold", loc="left",
    )
    ax.set_xlabel("date")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    _save(fig, "fig5_sarima_forecast.png")


# ---------------------------------------------------------------------------
# Figure 6: Box-Jenkins methodology flow
# ---------------------------------------------------------------------------

def fig6_box_jenkins_flow() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        # (x, y, w, h, text, color)
        (0.4, 3.8, 2.6, 1.1,
         "1. Identification\n(plot, ADF, ACF/PACF\n -> guess p, d, q)", COLOR_BLUE),
        (3.7, 3.8, 2.6, 1.1,
         "2. Estimation\n(MLE for AR / MA\ncoefficients)", COLOR_PURPLE),
        (7.0, 3.8, 2.6, 1.1,
         "3. Diagnostic checking\n(residuals white noise?\nLjung-Box, residual ACF)", COLOR_GREEN),
        (3.7, 1.0, 2.6, 1.1,
         "4. Forecasting\n(point + interval)", COLOR_AMBER),
    ]
    for x, y, w, h, text, color in boxes:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.18,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=10.5, color="#111827")

    arrow_kwargs = dict(arrowstyle="-|>", mutation_scale=18, color="#374151", lw=1.4)
    # 1 -> 2
    ax.add_patch(FancyArrowPatch((3.05, 4.35), (3.65, 4.35), **arrow_kwargs))
    # 2 -> 3
    ax.add_patch(FancyArrowPatch((6.35, 4.35), (6.95, 4.35), **arrow_kwargs))
    # 3 -> 4 (down-left)
    ax.add_patch(FancyArrowPatch((7.5, 3.75), (5.4, 2.15), **arrow_kwargs))
    # 3 -> 1 loop back ("if diagnostics fail"), routed cleanly above all boxes
    ax.add_patch(FancyArrowPatch(
        (8.3, 4.95), (1.7, 4.95),
        connectionstyle="arc3,rad=-0.45",
        **arrow_kwargs,
    ))
    ax.text(5.0, 6.35, "if residuals are not white noise -> revise (p, d, q)",
            ha="center", va="center", fontsize=10.5, color="#b45309", style="italic")

    ax.text(5.0, 0.35,
            "Box-Jenkins (1970): treat ARIMA model selection as an iterative loop, "
            "not a one-shot fit.",
            ha="center", va="center", fontsize=10.5, color="#374151", style="italic")

    ax.set_title("Box-Jenkins methodology", fontsize=14, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    _save(fig, "fig6_box_jenkins_flow.png")


# ---------------------------------------------------------------------------
# Figure 7: Holt-Winters exponential smoothing
# ---------------------------------------------------------------------------

def fig7_holt_winters() -> None:
    series = _seasonal_series(n_periods=120, seed=71, trend=0.20, season_amp=7.0, noise_std=1.0)
    train = series.iloc[:-24]
    test = series.iloc[-24:]

    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=12,
        initialization_method="estimated",
    ).fit()
    forecast = model.forecast(24)

    # Pull internal level / trend / season components for the visualisation
    level = pd.Series(model.level, index=train.index)
    trend = pd.Series(model.trend, index=train.index)
    season = pd.Series(model.season, index=train.index)

    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.35)

    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(train.index, train.values, color=COLOR_BLUE, lw=1.3, label="Training history")
    ax_top.plot(test.index, test.values, color=COLOR_AMBER, lw=1.8, label="Actual (held out)")
    ax_top.plot(forecast.index, forecast.values, color=COLOR_PURPLE, lw=1.8,
                label="Holt-Winters forecast")
    ax_top.axvline(train.index[-1], color="grey", lw=0.7, ls="--")
    ax_top.set_title(
        "Holt-Winters additive: level + trend + 12-month seasonality",
        fontsize=13, fontweight="bold", loc="left",
    )
    ax_top.set_ylabel("y")
    ax_top.legend(loc="upper left", frameon=True)

    ax_bot = fig.add_subplot(gs[1])
    ax_bot.plot(level.index, level.values, color=COLOR_BLUE, lw=1.3, label="level  $\\ell_t$")
    ax_bot.plot(trend.index, trend.values + level.values * 0,  # trend on its own scale
                color=COLOR_PURPLE, lw=1.3, label="trend  $b_t$")
    ax_bot_twin = ax_bot.twinx()
    ax_bot_twin.plot(season.index, season.values, color=COLOR_GREEN, lw=1.0,
                     label="seasonal  $s_t$", alpha=0.8)
    ax_bot_twin.set_ylabel("seasonal", color=COLOR_GREEN)
    ax_bot_twin.tick_params(axis="y", labelcolor=COLOR_GREEN)

    ax_bot.set_title(
        "Decomposed smoothing components (level / trend on left, seasonal on right)",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax_bot.set_xlabel("date")
    ax_bot.set_ylabel("level / trend")
    lines1, labels1 = ax_bot.get_legend_handles_labels()
    lines2, labels2 = ax_bot_twin.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    _save(fig, "fig7_holt_winters.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_components()
    fig2_ar_ma_arma()
    fig3_acf_pacf()
    fig4_arima_forecast()
    fig5_sarima_forecast()
    fig6_box_jenkins_flow()
    fig7_holt_winters()
    print(f"Wrote 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
