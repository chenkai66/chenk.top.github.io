#!/usr/bin/env python3
"""
ODE Chapter 01 - Origins and Intuition
Figure generation script.

Generates 7 figures and writes them to BOTH the EN and ZH asset folders:
  - source/_posts/en/ode/01-origins-and-intuition/
  - source/_posts/zh/ode/01-微分方程的起源与直觉/

Run from anywhere:
    python 01-origins.py

Style: seaborn-v0_8-whitegrid, dpi=150
Palette: blue #2563eb, purple #7c3aed, green #10b981, red #ef4444
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint, solve_ivp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
RED = "#ef4444"
GRAY = "#6b7280"
ORANGE = "#f59e0b"

DPI = 150

# Repo root inferred from this file's location: <repo>/scripts/figures/ode/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "ode" / "01-origins-and-intuition"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "ode" / "01-微分方程的起源与直觉"

EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to BOTH EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        path = d / name
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Direction field for dy/dt = -y
# ---------------------------------------------------------------------------

def fig1_direction_field() -> None:
    """Direction field for the prototypical decay equation dy/dt = -y."""
    print("Figure 1: direction field for dy/dt = -y")
    fig, ax = plt.subplots(figsize=(9, 6))

    t_grid = np.linspace(0, 5, 26)
    y_grid = np.linspace(-2.5, 2.5, 21)
    T, Y = np.meshgrid(t_grid, y_grid)

    dT = np.ones_like(T)
    dY = -Y
    M = np.hypot(dT, dY)
    ax.quiver(
        T, Y, dT / M, dY / M, M,
        cmap="Blues", alpha=0.75, scale=35, width=0.0035,
        pivot="middle",
    )

    # Overlay solution curves through several initial conditions.
    t_dense = np.linspace(0, 5, 400)
    initial_conditions = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    for y0 in initial_conditions:
        y = y0 * np.exp(-t_dense)
        ax.plot(t_dense, y, color=PURPLE, linewidth=1.8, alpha=0.85)

    # Highlight the equilibrium y = 0 in red.
    ax.axhline(0, color=RED, linewidth=2.2, linestyle="--",
               label=r"Equilibrium $y=0$ (attractor)")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(r"Direction field for $dy/dt = -y$ with solution curves",
                 fontsize=13)
    ax.set_xlim(0, 5)
    ax.set_ylim(-2.5, 2.5)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig1_direction_field.png")


# ---------------------------------------------------------------------------
# Figure 2: Solution family for cooling coffee
# ---------------------------------------------------------------------------

def fig2_solution_family() -> None:
    """Newton's law of cooling: many initial temperatures, one room."""
    print("Figure 2: solution family for Newton's law of cooling")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    t = np.linspace(0, 60, 400)
    T_env = 20.0
    k = 0.08
    initials = [95, 80, 65, 50, 35, 5, -10]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(initials)))

    for T0, c in zip(initials, cmap):
        T = T_env + (T0 - T_env) * np.exp(-k * t)
        ax.plot(t, T, color=c, linewidth=2.0, label=f"T(0) = {T0}°C")

    ax.axhline(T_env, color=RED, linewidth=2.2, linestyle="--",
               label=f"Room temperature = {T_env}°C")
    ax.fill_between(t, T_env - 0.3, T_env + 0.3, color=RED, alpha=0.15)

    ax.set_xlabel("Time t (minutes)")
    ax.set_ylabel("Temperature T (°C)")
    ax.set_title("Solution family: every initial condition flows to the same equilibrium",
                 fontsize=13)
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax.set_xlim(0, 60)
    fig.tight_layout()
    save(fig, "fig2_solution_family.png")


# ---------------------------------------------------------------------------
# Figure 3: Three real-world cases side by side
# ---------------------------------------------------------------------------

def fig3_three_classics() -> None:
    """Coffee cooling, radioactive decay, harmonic oscillator."""
    print("Figure 3: three classic ODE applications")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) Coffee cooling
    ax = axes[0]
    t = np.linspace(0, 60, 400)
    T_env, k, T0 = 20.0, 0.1, 90.0
    T = T_env + (T0 - T_env) * np.exp(-k * t)
    ax.plot(t, T, color=BLUE, linewidth=2.4, label="T(t)")
    ax.axhline(T_env, color=RED, linestyle="--", linewidth=1.8,
               label="Room 20°C")
    ax.fill_between(t, T, T_env, color=BLUE, alpha=0.10)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(r"Newton cooling: $T' = -k(T - T_{env})$", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9)

    # (b) Radioactive decay
    ax = axes[1]
    t_half = 5730.0  # Carbon-14 in years
    lam = np.log(2) / t_half
    t = np.linspace(0, 4 * t_half, 400)
    N0 = 100.0
    N = N0 * np.exp(-lam * t)
    ax.plot(t / 1000, N, color=PURPLE, linewidth=2.4, label="N(t)")
    for n in (1, 2, 3):
        ax.axvline(n * t_half / 1000, color=GRAY, linestyle=":",
                   linewidth=1.2, alpha=0.7)
        ax.text(n * t_half / 1000, 5 + N0 * 0.5 ** n, f"  {n}×t½",
                fontsize=9, color=GRAY)
    ax.axhline(50, color=RED, linestyle="--", linewidth=1.5,
               label="Half = 50 g")
    ax.set_xlabel("Time (kyr)")
    ax.set_ylabel("Mass remaining (g)")
    ax.set_title(r"Carbon-14 decay: $N' = -\lambda N$", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9)

    # (c) Harmonic oscillator
    ax = axes[2]
    omega = 2 * np.pi
    t = np.linspace(0, 3, 600)

    def spring(state, _t):
        x, v = state
        return [v, -omega ** 2 * x]

    sol = odeint(spring, [1.0, 0.0], t)
    ax.plot(t, sol[:, 0], color=GREEN, linewidth=2.4, label="x(t)")
    ax.plot(t, sol[:, 1] / omega, color=ORANGE, linewidth=1.8, alpha=0.85,
            label=r"$v(t)/\omega$")
    ax.axhline(0, color=RED, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement / scaled velocity")
    ax.set_title(r"Spring oscillator: $x'' + \omega^2 x = 0$", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Three classic differential equations from physics",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_three_classics.png")


# ---------------------------------------------------------------------------
# Figure 4: Linear vs nonlinear ODE comparison
# ---------------------------------------------------------------------------

def fig4_linear_vs_nonlinear() -> None:
    """Side-by-side: linear stays well-behaved, nonlinear blows up."""
    print("Figure 4: linear vs nonlinear")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Linear: y' = -y + sin(t) ----
    ax = axes[0]
    t = np.linspace(0, 12, 600)
    initials = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    cmap = plt.cm.Blues(np.linspace(0.4, 0.9, len(initials)))
    for y0, c in zip(initials, cmap):
        sol = solve_ivp(lambda tt, yy: -yy + np.sin(tt),
                        (0, 12), [y0], t_eval=t, rtol=1e-8)
        ax.plot(sol.t, sol.y[0], color=c, linewidth=1.8,
                label=f"y(0)={y0:+.0f}")
    # Steady-state envelope: y_p = (sin t - cos t)/2
    ax.plot(t, (np.sin(t) - np.cos(t)) / 2, color=RED, linewidth=2.2,
            linestyle="--", label="Steady-state attractor")
    ax.set_title(r"Linear: $y' = -y + \sin t$  (predictable, stable)",
                 fontsize=12)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)

    # ---- Nonlinear: y' = y^2 - t (finite-time blow-up) ----
    ax = axes[1]
    t_eval = np.linspace(0, 4, 600)
    initials = [-1.5, -0.5, 0.0, 0.5, 0.9, 1.05]
    cmap = plt.cm.Purples(np.linspace(0.4, 0.9, len(initials)))
    for y0, c in zip(initials, cmap):
        sol = solve_ivp(lambda tt, yy: yy ** 2 - tt,
                        (0, 4), [y0], t_eval=t_eval,
                        rtol=1e-8, atol=1e-10)
        ax.plot(sol.t, sol.y[0], color=c, linewidth=1.8,
                label=f"y(0)={y0:+.2f}")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_ylim(-3, 6)
    ax.set_xlim(0, 4)
    ax.set_title(r"Nonlinear: $y' = y^2 - t$  (blow-up, sensitive)",
                 fontsize=12)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)

    # Annotate blow-up
    ax.annotate("finite-time\nblow-up",
                xy=(1.6, 5.5), xytext=(2.4, 4.5),
                fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    fig.tight_layout()
    save(fig, "fig4_linear_vs_nonlinear.png")


# ---------------------------------------------------------------------------
# Figure 5: Order of an ODE - first-order vs second-order
# ---------------------------------------------------------------------------

def fig5_order_comparison() -> None:
    """First-order = monotone decay; second-order = oscillation."""
    print("Figure 5: first vs second order")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # First order: y' = -y, family of decays
    ax = axes[0]
    t = np.linspace(0, 5, 400)
    for y0, c in zip([2.0, 1.0, 0.5, -0.5, -1.0, -2.0],
                     plt.cm.Blues(np.linspace(0.4, 0.9, 6))):
        ax.plot(t, y0 * np.exp(-t), color=c, linewidth=2.0,
                label=f"y(0) = {y0:+.1f}")
    ax.axhline(0, color=RED, linewidth=2.0, linestyle="--",
               label="Equilibrium")
    ax.set_title(r"First-order: $y' = -y$ — single initial value", fontsize=12)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)

    # Second order: y'' + y = 0, family of oscillations
    ax = axes[1]
    t = np.linspace(0, 4 * np.pi, 600)
    pairs = [(1.0, 0.0), (1.0, 1.0), (0.5, -0.8), (-0.5, 0.5),
             (1.5, 0.0), (0.0, 1.0)]
    for (x0, v0), c in zip(pairs, plt.cm.Greens(np.linspace(0.4, 0.9, len(pairs)))):
        sol = solve_ivp(lambda tt, s: [s[1], -s[0]], (0, 4 * np.pi),
                        [x0, v0], t_eval=t, rtol=1e-8)
        ax.plot(sol.t, sol.y[0], color=c, linewidth=2.0,
                label=f"x(0)={x0:+.1f}, v(0)={v0:+.1f}")
    ax.axhline(0, color=RED, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_title(r"Second-order: $x'' + x = 0$ — needs position AND velocity",
                 fontsize=12)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend(loc="upper right", fontsize=7.8, ncol=2)

    fig.tight_layout()
    save(fig, "fig5_order_comparison.png")


# ---------------------------------------------------------------------------
# Figure 6: Existence and uniqueness intuition (Picard-Lindelöf)
# ---------------------------------------------------------------------------

def fig6_existence_uniqueness() -> None:
    """Two panels: nice f -> unique solution; bad f -> branching."""
    print("Figure 6: existence and uniqueness")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- Nice f: y' = -y ----
    ax = axes[0]
    t = np.linspace(0, 4, 300)
    for y0, c in zip(np.linspace(-2, 2, 9),
                     plt.cm.Blues(np.linspace(0.3, 0.95, 9))):
        ax.plot(t, y0 * np.exp(-t), color=c, linewidth=1.8)
    ax.scatter([0], [1.0], color=RED, s=110, zorder=5,
               label="Initial condition (1, 1)")
    # Single highlighted unique solution.
    ax.plot(t, np.exp(-t), color=RED, linewidth=3.0,
            label="Unique solution")
    ax.set_title(r"Lipschitz $f$: through every point, exactly ONE curve",
                 fontsize=11.5)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 4)
    ax.set_ylim(-2.2, 2.2)

    # ---- Bad f: y' = 3 y^{2/3}, y(0)=0 has infinitely many solutions ----
    ax = axes[1]
    t = np.linspace(0, 4, 400)
    # Family: y(t) = (t - c)^3 for t >= c, else 0
    for c_shift, color in zip(
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            plt.cm.Purples(np.linspace(0.4, 0.95, 6))):
        y = np.where(t >= c_shift, (t - c_shift) ** 3, 0.0)
        ax.plot(t, y, color=color, linewidth=2.0,
                label=f"branch at c = {c_shift}")
    ax.scatter([0], [0], color=RED, s=140, zorder=5,
               label="Initial condition (0, 0)")
    ax.set_title(r"$y' = 3\,y^{2/3}$: NOT Lipschitz at 0 — many solutions!",
                 fontsize=11.5)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_xlim(0, 4)
    ax.set_ylim(-2, 30)
    ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Picard–Lindelöf: when does the IVP have a unique solution?",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_existence_uniqueness.png")


# ---------------------------------------------------------------------------
# Figure 7: Population growth — exponential vs logistic
# ---------------------------------------------------------------------------

def fig7_exponential_vs_logistic() -> None:
    """Malthus blows up; logistic saturates to carrying capacity."""
    print("Figure 7: exponential vs logistic growth")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Time series ----
    ax = axes[0]
    t = np.linspace(0, 30, 500)
    P0, r, K = 5.0, 0.25, 100.0

    P_exp = P0 * np.exp(r * t)
    P_log = K / (1 + ((K - P0) / P0) * np.exp(-r * t))

    ax.plot(t, P_exp, color=PURPLE, linewidth=2.6,
            label=r"Exponential: $P' = rP$")
    ax.plot(t, P_log, color=BLUE, linewidth=2.6,
            label=r"Logistic: $P' = rP(1 - P/K)$")
    ax.axhline(K, color=RED, linewidth=2.0, linestyle="--",
               label=f"Carrying capacity K = {int(K)}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Population P(t)")
    ax.set_title("Exponential blows up; logistic saturates", fontsize=12)
    ax.set_ylim(0, 200)
    ax.legend(loc="upper left", fontsize=10)

    # ---- Phase line / growth-rate plot ----
    ax = axes[1]
    P = np.linspace(0, 130, 400)
    rate_exp = r * P
    rate_log = r * P * (1 - P / K)

    ax.plot(P, rate_exp, color=PURPLE, linewidth=2.4,
            label=r"$rP$")
    ax.plot(P, rate_log, color=BLUE, linewidth=2.4,
            label=r"$rP(1-P/K)$")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axvline(K, color=RED, linewidth=1.8, linestyle="--",
               label=f"Equilibrium P = K = {int(K)}")
    ax.scatter([0, K], [0, 0], color=RED, s=80, zorder=5)
    ax.fill_between(P, rate_log, 0, where=(rate_log > 0),
                    color=GREEN, alpha=0.15, label="Growth")
    ax.fill_between(P, rate_log, 0, where=(rate_log < 0),
                    color=ORANGE, alpha=0.18, label="Decline")
    ax.set_xlabel("Population P")
    ax.set_ylabel("dP/dt")
    ax.set_title("Phase view: where does growth stop?", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    save(fig, "fig7_exponential_vs_logistic.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Repo root  : {REPO_ROOT}")
    print(f"EN target  : {EN_DIR}")
    print(f"ZH target  : {ZH_DIR}")
    print()

    fig1_direction_field()
    fig2_solution_family()
    fig3_three_classics()
    fig4_linear_vs_nonlinear()
    fig5_order_comparison()
    fig6_existence_uniqueness()
    fig7_exponential_vs_logistic()

    print()
    print("Done. 7 figures written to both EN and ZH asset folders.")


if __name__ == "__main__":
    main()
