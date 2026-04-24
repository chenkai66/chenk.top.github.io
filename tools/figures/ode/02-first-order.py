"""
Figure generation script for ODE Chapter 02:
"First-Order Methods".

Generates 7 figures used in BOTH the EN and ZH versions of the article.
Each figure illustrates ONE first-order technique cleanly, with a consistent
palette and minimal chart-junk so the math idea is the visual idea.

Figures
-------
fig1_separable_solution_curves
    Solution curves of dy/dx = -x/y (concentric circles) overlaid on the
    slope field, showing how separation produces a one-parameter family.

fig2_integrating_factor
    Three panels for y' + 2y = 4: the unmultiplied slope field with one
    solution, the integrating factor mu(x) = e^{2x}, and the product
    d/dx[mu*y] = 4 mu collapsing to a clean total derivative.

fig3_exact_level_curves
    Level curves F(x,y) = x^2 + xy + y^2 = C for an exact equation
    (2x + y) dx + (x + 2y) dy = 0, with the gradient field overlaid to
    confirm orthogonality with the level sets (the solution trajectories).

fig4_bernoulli_transform
    Side-by-side comparison: nonlinear Bernoulli y' + y = y^2 in (x, y)
    space and its linearised twin v' - v = -1 (with v = y^{-1}) in
    (x, v) space, showing how the substitution straightens the family.

fig5_mixing_tank
    Salt mass Q(t) for the canonical 1000 L tank with two scenarios
    (empty start vs. above-equilibrium start), plus a small inset of
    the concentration approaching 2 g/L.

fig6_logistic_growth
    Logistic family P(t) for several P_0 and the inflection lines at
    P = K/2 and P = K, with a second axis showing the dP/dt parabola
    so the maximum-growth-rate point is visually obvious.

fig7_slope_field_comparison
    Two slope fields side by side: linear y' = y - x (parallel-looking
    flow) vs. Bernoulli y' = y - y^3 (three nullclines, two stable, one
    unstable), with representative trajectories integrated by RK45.

Usage
-----
    python3 scripts/figures/ode/02-first-order.py

Output
------
Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
references stay identical across languages. Parent folders are created
if missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import solve_ivp

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_RED = COLORS["danger"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "ode" / "02-first-order-methods"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "ode" / "02-一阶微分方程"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _slope_field(
    ax: plt.Axes,
    f,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    nx: int = 22,
    ny: int = 18,
    color: str = C_GRAY,
    alpha: float = 0.55,
    length: float = 0.55,
) -> None:
    """Draw a normalised slope field of dy/dx = f(x, y)."""
    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    X, Y = np.meshgrid(xs, ys)
    with np.errstate(over="ignore", invalid="ignore"):
        S = f(X, Y)
    U = np.ones_like(S)
    V = S
    N = np.hypot(U, V)
    N = np.where(N == 0, 1, N)
    U /= N
    V /= N
    dx = (xlim[1] - xlim[0]) / nx * length
    dy = (ylim[1] - ylim[0]) / ny * length
    ax.quiver(
        X, Y, U * dx, V * dy,
        angles="xy", scale_units="xy", scale=1.0,
        color=color, alpha=alpha, width=0.0028,
        headwidth=0, headlength=0, headaxislength=0,
        pivot="middle",
    )


def _style_ax(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    if title:
        ax.set_title(title, fontsize=12, color=C_DARK, pad=8, weight="semibold")
    if xlabel:
        ax.set_xlabel(xlabel, color=C_DARK)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_DARK)
    ax.tick_params(labelsize=9, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_color(C_LIGHT)
    ax.grid(True, color=C_LIGHT, linewidth=0.7, zorder=0)


# ---------------------------------------------------------------------------
# Figure 1: Separable equation -- circular solution curves
# ---------------------------------------------------------------------------
def fig1_separable_solution_curves() -> None:
    """dy/dx = -x/y separates to y dy + x dx = 0 -> x^2 + y^2 = C."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    f = lambda x, y: np.where(np.abs(y) < 1e-3, np.nan, -x / np.where(y == 0, 1e-9, y))
    _slope_field(ax, f, (-3.2, 3.2), (-3.2, 3.2), nx=24, ny=24)

    theta = np.linspace(0, 2 * np.pi, 400)
    radii = [0.6, 1.2, 1.8, 2.4, 3.0]
    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_RED]
    for r, c in zip(radii, palette):
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color=c, linewidth=2.2, label=f"$x^2+y^2={r**2:.2f}$")

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    _style_ax(ax,
              title=r"Separable: $\dfrac{dy}{dx}=-\dfrac{x}{y}\ \Rightarrow\ x^2+y^2=C$",
              xlabel="x", ylabel="y")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    _save(fig, "fig1_separable_solution_curves")


# ---------------------------------------------------------------------------
# Figure 2: Integrating factor for y' + 2y = 4
# ---------------------------------------------------------------------------
def fig2_integrating_factor() -> None:
    """Three-panel: slope field + solutions, mu(x), d/dx[mu y] = 4 mu."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    P_const = 2.0
    Q_const = 4.0

    # Panel 1: slope field with several solutions y = 2 + Ce^{-2x}
    ax = axes[0]
    f = lambda x, y: Q_const - P_const * y
    _slope_field(ax, f, (-1.0, 2.5), (-1.0, 5.0), nx=20, ny=18)
    xs = np.linspace(-1.0, 2.5, 300)
    for C, c in zip([-3.0, -1.5, 0.0, 1.5, 3.0],
                    [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_RED]):
        ys = 2.0 + C * np.exp(-2 * xs)
        ax.plot(xs, ys, color=c, linewidth=2.0, label=f"C = {C:+.1f}")
    ax.axhline(2.0, color=C_DARK, linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(2.4, 2.08, "equilibrium $y=2$", color=C_DARK,
            fontsize=9, ha="right", style="italic")
    ax.set_xlim(-1.0, 2.5)
    ax.set_ylim(-1.0, 5.0)
    _style_ax(ax,
              title=r"Solutions of $y' + 2y = 4$",
              xlabel="x", ylabel="y")
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.95)

    # Panel 2: mu(x) = e^{2x}
    ax = axes[1]
    xs = np.linspace(-1.0, 2.0, 300)
    ax.plot(xs, np.exp(P_const * xs), color=C_PURPLE, linewidth=2.6,
            label=r"$\mu(x) = e^{2x}$")
    ax.fill_between(xs, 0, np.exp(P_const * xs),
                    color=C_PURPLE, alpha=0.08)
    ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1.0)
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(0, 60)
    _style_ax(ax,
              title=r"Integrating factor $\mu(x)=e^{\int 2\,dx}$",
              xlabel="x", ylabel=r"$\mu$")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # Panel 3: collapse to total derivative
    ax = axes[2]
    xs = np.linspace(-1.0, 2.0, 300)
    # particular solution with C = -1: y = 2 - e^{-2x}
    y = 2.0 - np.exp(-2 * xs)
    mu = np.exp(2 * xs)
    lhs = mu * y                  # mu * y
    rhs = 2 * mu - 1              # integral of 4 mu = 2 e^{2x} + const ; pick const = -1
    ax.plot(xs, lhs, color=C_BLUE, linewidth=2.4,
            label=r"$\mu(x)\,y(x)$")
    ax.plot(xs, rhs, color=C_GREEN, linewidth=2.4, linestyle="--",
            label=r"$\int 4\mu\,dx = 2e^{2x}+C$")
    ax.set_xlim(-1.0, 2.0)
    _style_ax(ax,
              title=r"After multiplying: $\dfrac{d}{dx}[\mu y] = 4\mu$",
              xlabel="x", ylabel="value")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    _save(fig, "fig2_integrating_factor")


# ---------------------------------------------------------------------------
# Figure 3: Exact equation -- level curves of F(x, y) = C
# ---------------------------------------------------------------------------
def fig3_exact_level_curves() -> None:
    """(2x + y) dx + (x + 2y) dy = 0 has F = x^2 + xy + y^2."""
    fig, ax = plt.subplots(figsize=(7.4, 6.4))

    xs = np.linspace(-3.0, 3.0, 400)
    ys = np.linspace(-3.0, 3.0, 400)
    X, Y = np.meshgrid(xs, ys)
    F = X ** 2 + X * Y + Y ** 2

    levels = [0.5, 1.5, 3.0, 5.0, 8.0, 12.0]
    cs = ax.contour(X, Y, F, levels=levels,
                    colors=[C_BLUE, C_PURPLE, C_GREEN, C_AMBER,
                            C_RED, C_DARK],
                    linewidths=2.0)
    ax.clabel(cs, inline=True, fontsize=8.5, fmt="C=%.1f")

    # Gradient field of F (perpendicular to level curves)
    xs_q = np.linspace(-2.8, 2.8, 14)
    ys_q = np.linspace(-2.8, 2.8, 14)
    Xq, Yq = np.meshgrid(xs_q, ys_q)
    Fx = 2 * Xq + Yq           # = M
    Fy = Xq + 2 * Yq           # = N
    N = np.hypot(Fx, Fy)
    N = np.where(N == 0, 1, N)
    ax.quiver(Xq, Yq, Fx / N, Fy / N,
              color=C_GRAY, alpha=0.55, scale=28, width=0.0035,
              headwidth=4, headlength=5)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    _style_ax(ax,
              title=r"Exact: $(2x{+}y)dx + (x{+}2y)dy = 0\ \Rightarrow\ "
                    r"x^2+xy+y^2 = C$",
              xlabel="x", ylabel="y")

    ax.text(0.02, 0.98,
            "arrows = $\\nabla F$ (perpendicular to solutions)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color=C_DARK, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C_LIGHT, alpha=0.9))

    fig.tight_layout()
    _save(fig, "fig3_exact_level_curves")


# ---------------------------------------------------------------------------
# Figure 4: Bernoulli substitution v = y^{1-n} linearises the equation
# ---------------------------------------------------------------------------
def fig4_bernoulli_transform() -> None:
    """y' + y = y^2  (n=2) -> v = 1/y satisfies v' - v = -1, linear."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # Left: nonlinear Bernoulli in (x, y)
    ax = axes[0]
    f_y = lambda x, y: -y + y ** 2
    _slope_field(ax, f_y, (0.0, 4.0), (0.0, 2.5), nx=22, ny=20)

    # Solutions: y = 1 / (1 + (1/y0 - 1) e^{-x})  comes from v' - v = -1
    xs = np.linspace(0.0, 4.0, 400)
    y0_list = [0.2, 0.5, 0.8, 1.2, 1.6]
    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_RED]
    for y0, c in zip(y0_list, palette):
        v0 = 1.0 / y0
        # v' - v = -1  =>  v(x) = 1 + (v0 - 1) e^{x}
        v = 1.0 + (v0 - 1.0) * np.exp(xs)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(v > 0.05, 1.0 / v, np.nan)
        ax.plot(xs, y, color=c, linewidth=2.2,
                label=f"$y_0 = {y0}$")
    ax.axhline(1.0, color=C_DARK, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(3.95, 1.05, "$y=1$ (unstable)", color=C_DARK,
            fontsize=9, ha="right", style="italic")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 2.5)
    _style_ax(ax,
              title=r"Bernoulli (nonlinear): $y' + y = y^2$",
              xlabel="x", ylabel="y")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    # Right: linearised in (x, v) where v = 1/y
    ax = axes[1]
    f_v = lambda x, v: v - 1.0
    _slope_field(ax, f_v, (0.0, 4.0), (-2.0, 5.0), nx=22, ny=20)

    xs = np.linspace(0.0, 4.0, 400)
    for y0, c in zip(y0_list, palette):
        v0 = 1.0 / y0
        v = 1.0 + (v0 - 1.0) * np.exp(xs)
        ax.plot(xs, v, color=c, linewidth=2.2,
                label=f"$v_0 = {v0:.2f}$")
    ax.axhline(1.0, color=C_DARK, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(3.95, 1.1, "$v=1$ equilibrium", color=C_DARK,
            fontsize=9, ha="right", style="italic")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-2.0, 5.0)
    _style_ax(ax,
              title=r"Substitute $v=1/y$: linear $v' - v = -1$",
              xlabel="x", ylabel="v")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.suptitle(r"Bernoulli $\to$ Linear under $v = y^{1-n}$",
                 fontsize=13, color=C_DARK, weight="semibold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_bernoulli_transform")


# ---------------------------------------------------------------------------
# Figure 5: Mixing tank
# ---------------------------------------------------------------------------
def fig5_mixing_tank() -> None:
    """Q'(t) = 10 - Q/200, Q(0) = 0 vs Q(0) = 3000, both -> 2000 g."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                                  gridspec_kw={"width_ratios": [1.4, 1.0]})

    t = np.linspace(0, 1500, 600)
    Q_a = 2000 * (1 - np.exp(-t / 200))                      # empty start
    Q_b = 2000 + (3000 - 2000) * np.exp(-t / 200)            # over-equilibrium

    ax.plot(t, Q_a, color=C_BLUE, linewidth=2.6,
            label=r"$Q(0)=0$ (filling up)")
    ax.plot(t, Q_b, color=C_RED, linewidth=2.6,
            label=r"$Q(0)=3000$ (washing out)")
    ax.axhline(2000, color=C_GREEN, linestyle="--", linewidth=1.6,
               label=r"equilibrium $Q^\ast = 2000$ g")

    # Mark the time constant tau = 200 min on the filling curve
    tau = 200
    ax.axvline(tau, color=C_GRAY, linestyle=":", linewidth=1.0)
    ax.scatter([tau], [2000 * (1 - np.exp(-1))],
               color=C_BLUE, s=42, zorder=6)
    ax.annotate(r"$\tau = 200$ min$\rightarrow$ 63% filled",
                xy=(tau, 2000 * (1 - np.exp(-1))),
                xytext=(tau + 80, 900),
                fontsize=9.5, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.0))

    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 3200)
    _style_ax(ax,
              title="Salt mass $Q(t)$ in the 1000 L tank",
              xlabel="time t (min)", ylabel="Q (g)")
    ax.legend(loc="center right", fontsize=9.5, framealpha=0.95)

    # Right: concentration vs equilibrium
    c_a = Q_a / 1000.0
    c_b = Q_b / 1000.0
    ax2.plot(t, c_a, color=C_BLUE, linewidth=2.4)
    ax2.plot(t, c_b, color=C_RED, linewidth=2.4)
    ax2.axhline(2.0, color=C_GREEN, linestyle="--", linewidth=1.6)
    ax2.axhline(0.0, color=C_GRAY, linewidth=0.8)
    ax2.text(1450, 2.06, "inflow concentration", color=C_GREEN,
             fontsize=9, ha="right", style="italic")
    ax2.set_xlim(0, 1500)
    ax2.set_ylim(0, 3.3)
    _style_ax(ax2,
              title="Concentration $C(t) = Q/V$ (g/L)",
              xlabel="time t (min)", ylabel="C (g/L)")

    fig.tight_layout()
    _save(fig, "fig5_mixing_tank")


# ---------------------------------------------------------------------------
# Figure 6: Logistic growth
# ---------------------------------------------------------------------------
def fig6_logistic_growth() -> None:
    """P' = r P (1 - P/K), several initial populations."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8),
                                  gridspec_kw={"width_ratios": [1.3, 1.0]})

    K, r = 1000.0, 0.30
    t = np.linspace(0, 30, 500)

    p0_list = [10.0, 100.0, 500.0, 1500.0]
    palette = [C_BLUE, C_PURPLE, C_GREEN, C_RED]
    for P0, c in zip(p0_list, palette):
        P = K / (1 + (K / P0 - 1) * np.exp(-r * t))
        ax.plot(t, P, color=c, linewidth=2.4, label=f"$P_0 = {int(P0)}$")
        # Mark inflection at P = K/2 for the rising curves only
        if P0 < K / 2:
            t_star = np.log(K / P0 - 1) / r
            ax.scatter([t_star], [K / 2], color=c, s=40,
                       edgecolor="white", linewidth=1.2, zorder=6)

    ax.axhline(K, color=C_RED, linestyle="--", linewidth=1.4,
               label=f"K = {int(K)}")
    ax.axhline(K / 2, color=C_AMBER, linestyle=":", linewidth=1.2,
               label=r"inflection $P=K/2$")

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1700)
    _style_ax(ax,
              title=r"Logistic growth: $P' = rP(1-P/K)$",
              xlabel="time t", ylabel="population P")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=1)

    # Right: dP/dt parabola
    P_range = np.linspace(0, 1.5 * K, 300)
    growth = r * P_range * (1 - P_range / K)
    ax2.plot(P_range, growth, color=C_PURPLE, linewidth=2.6)
    ax2.fill_between(P_range, 0, growth,
                     where=(growth > 0), color=C_PURPLE, alpha=0.12)
    ax2.axvline(K / 2, color=C_AMBER, linestyle=":", linewidth=1.2)
    ax2.axvline(K, color=C_RED, linestyle="--", linewidth=1.2)
    ax2.scatter([K / 2], [r * K / 4], color=C_AMBER, s=55,
                edgecolor="white", linewidth=1.4, zorder=6)
    ax2.annotate(r"max rate at $P=K/2$",
                 xy=(K / 2, r * K / 4),
                 xytext=(K / 2 + 60, r * K / 4 + 8),
                 fontsize=9.5, color=C_DARK,
                 arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.0))

    ax2.set_xlim(0, 1.5 * K)
    ax2.set_ylim(-30, r * K / 4 * 1.45)
    _style_ax(ax2,
              title=r"Growth rate $\dot P$ vs. population",
              xlabel="P", ylabel=r"$\dot P$")

    fig.tight_layout()
    _save(fig, "fig6_logistic_growth")


# ---------------------------------------------------------------------------
# Figure 7: Linear vs. Bernoulli slope-field comparison
# ---------------------------------------------------------------------------
def fig7_slope_field_comparison() -> None:
    """Linear y' = y - x  vs. Bernoulli y' = y - y^3 with three nullclines."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    # Left: linear y' = y - x  ->  y = x + 1 + Ce^{x}
    ax = axes[0]
    f1 = lambda x, y: y - x
    _slope_field(ax, f1, (-3.0, 3.0), (-3.0, 3.0), nx=24, ny=22)

    xs = np.linspace(-3.0, 3.0, 300)
    for C, c in zip([-1.5, -0.5, 0.0, 0.5, 1.5],
                    [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_RED]):
        ax.plot(xs, xs + 1 + C * np.exp(xs), color=c, linewidth=2.0)
    # Highlight the singular straight-line solution C = 0: y = x + 1
    ax.plot(xs, xs + 1, color=C_DARK, linewidth=1.4,
            linestyle="--", alpha=0.85, label="$y=x+1$ (no $e^x$)")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    _style_ax(ax,
              title=r"Linear: $y' = y - x$",
              xlabel="x", ylabel="y")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Right: Bernoulli y' = y - y^3, three equilibria y = -1, 0, 1
    ax = axes[1]
    f2 = lambda x, y: y - y ** 3
    _slope_field(ax, f2, (0.0, 6.0), (-1.8, 1.8), nx=26, ny=22)

    # Trajectories from a spread of initial conditions, integrated by RK45
    xs_span = (0.0, 6.0)
    xs_eval = np.linspace(*xs_span, 400)
    ic_list = [-1.6, -1.05, -0.7, -0.3, 0.0, 0.3, 0.7, 1.05, 1.6]
    for y0 in ic_list:
        sol = solve_ivp(lambda x, y: y - y ** 3, xs_span, [y0],
                        t_eval=xs_eval, rtol=1e-7, atol=1e-9)
        if sol.success:
            color = C_GREEN if abs(y0) < 1e-6 else (C_BLUE if y0 > 0 else C_PURPLE)
            ax.plot(sol.t, sol.y[0], color=color, linewidth=1.7, alpha=0.85)

    # Equilibria
    for y_eq, kind, color in [(1.0, "stable", C_RED),
                              (0.0, "unstable", C_AMBER),
                              (-1.0, "stable", C_RED)]:
        ax.axhline(y_eq, color=color, linewidth=1.3, linestyle="--", alpha=0.85)
        ax.text(5.95, y_eq + 0.07,
                f"$y={y_eq:+.0f}$ ({kind})",
                color=color, fontsize=9, ha="right", style="italic")

    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(-1.8, 1.8)
    _style_ax(ax,
              title=r"Bernoulli ($n=3$): $y' = y - y^3$",
              xlabel="x", ylabel="y")

    fig.suptitle("Linear vs. Bernoulli: same first-order family, different geometry",
                 fontsize=13, color=C_DARK, weight="semibold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_slope_field_comparison")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    figures = [
        fig1_separable_solution_curves,
        fig2_integrating_factor,
        fig3_exact_level_curves,
        fig4_bernoulli_transform,
        fig5_mixing_tank,
        fig6_logistic_growth,
        fig7_slope_field_comparison,
    ]
    for fn in figures:
        print(f"[ode/02] Generating {fn.__name__} ...")
        fn()
    print("[ode/02] Done.")


if __name__ == "__main__":
    main()
