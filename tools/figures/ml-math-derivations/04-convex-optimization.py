"""
Figure generation script for ML Math Derivations Part 04:
Convex Optimization Theory.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly. PNGs are written
to BOTH the EN and ZH article asset folders so the same filename works
in both languages.

Figures
-------
fig1_convex_sets         Convex set vs non-convex set: a chord that lies
                         inside vs a chord that pokes outside.
fig2_convex_functions    Convex vs non-convex function as 3D surfaces:
                         single bowl (one global minimum) vs egg-carton
                         (many local minima).
fig3_gd_convex_vs_nonconvex
                         Gradient descent on a convex function (clean
                         monotone descent to the global minimum) vs on
                         a non-convex function (gets stuck at a local
                         minimum), shown on contour plots with paths.
fig4_kkt_conditions      KKT geometry: at the constrained optimum,
                         the negative objective gradient must lie in the
                         cone of active constraint gradients.
fig5_duality_geometry    Primal vs dual: the (g(x), f(x)) image set with
                         a supporting hyperplane whose intercept is the
                         dual value; weak vs strong duality.
fig6_sgd_path            Stochastic gradient descent: noisy zig-zag path
                         around the deterministic GD path on the same
                         landscape.
fig7_learning_rate       Learning rate sweep: too small (slow), just
                         right (clean convergence), too big (oscillates
                         / diverges) on the same convex quadratic.

Usage
-----
    python3 scripts/figures/ml-math-derivations/04-convex-optimization.py

Output
------
Writes PNGs to BOTH the EN and ZH article asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good / convex
C_AMBER = "#f59e0b"    # warning / non-convex / SGD noise
C_RED = "#ef4444"      # error / divergence
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ml-math-derivations/"
    "04-Convex-Optimization-Theory"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ml-math-derivations/"
    "04-凸优化理论"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH EN and ZH asset folders as <name>.png."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png")


# ---------------------------------------------------------------------------
# Figure 1: Convex set vs non-convex set
# ---------------------------------------------------------------------------
def fig1_convex_sets() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # ----- left: convex set (an ellipse) -----
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ex = 2.2 * np.cos(theta)
    ey = 1.4 * np.sin(theta)
    ax.fill(ex, ey, color=C_GREEN, alpha=0.18, edgecolor=C_GREEN, linewidth=2)

    # two interior points + chord that stays inside
    p, q = np.array([-1.4, 0.6]), np.array([1.6, -0.7])
    ax.plot(*zip(p, q), color=C_BLUE, linewidth=2.2)
    ax.scatter(*zip(p, q), s=70, color=C_BLUE, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate("x", p, xytext=(-12, 8), textcoords="offset points",
                fontsize=12, color=C_BLUE, fontweight="bold")
    ax.annotate("y", q, xytext=(6, -14), textcoords="offset points",
                fontsize=12, color=C_BLUE, fontweight="bold")
    mid = 0.5 * (p + q)
    ax.annotate(r"$\lambda x + (1-\lambda) y \in C$",
                mid, xytext=(-50, 22), textcoords="offset points",
                fontsize=11, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1))

    ax.set_title("Convex set: every chord stays inside",
                 color=C_GREEN, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # ----- right: non-convex set (crescent / "C" shape) -----
    ax = axes[1]
    # Build a true crescent: outer half-disc minus a smaller inner half-disc.
    t1 = np.linspace(-np.pi / 2, np.pi / 2, 200)
    outer = np.column_stack([2.0 * np.cos(t1), 1.6 * np.sin(t1)])
    t2 = np.linspace(np.pi / 2, -np.pi / 2, 200)
    # inner arc is shifted left so the "C" opens to the right
    inner = np.column_stack([1.4 * np.cos(t2) - 0.6, 1.2 * np.sin(t2)])
    crescent = np.vstack([outer, inner])
    poly = Polygon(crescent, closed=True, facecolor=C_AMBER, alpha=0.20,
                   edgecolor=C_AMBER, linewidth=2)
    ax.add_patch(poly)

    # Two points inside the crescent, on opposite tips of the "C"
    p2, q2 = np.array([0.4, 1.35]), np.array([0.4, -1.35])
    ax.plot(*zip(p2, q2), color=C_BLUE, linewidth=2.2,
            linestyle="--", dashes=(4, 3))
    ax.scatter(*zip(p2, q2), s=70, color=C_BLUE, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate("x", p2, xytext=(8, 4), textcoords="offset points",
                fontsize=12, color=C_BLUE, fontweight="bold")
    ax.annotate("y", q2, xytext=(8, -10), textcoords="offset points",
                fontsize=12, color=C_BLUE, fontweight="bold")

    # mark a midpoint that escapes the crescent
    out = np.array([0.4, 0.0])
    ax.scatter(*out, s=120, color=C_RED, zorder=6, marker="X",
               edgecolor="white", linewidth=1.5)
    ax.annotate("chord exits the set", out, xytext=(28, 0),
                textcoords="offset points", fontsize=11, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))

    ax.set_title("Non-convex set: chord escapes",
                 color=C_AMBER, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Convex vs Non-Convex Sets",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_convex_sets")


# ---------------------------------------------------------------------------
# Figure 2: Convex vs non-convex function (3D surfaces)
# ---------------------------------------------------------------------------
def fig2_convex_functions() -> None:
    fig = plt.figure(figsize=(13, 6))

    x = np.linspace(-2.2, 2.2, 80)
    y = np.linspace(-2.2, 2.2, 80)
    X, Y = np.meshgrid(x, y)

    # ----- convex bowl -----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    Z1 = 0.5 * (X ** 2 + Y ** 2)
    ax1.plot_surface(X, Y, Z1, cmap="Blues", alpha=0.85, linewidth=0,
                     antialiased=True, edgecolor="none")
    ax1.scatter([0], [0], [0], color=C_GREEN, s=80,
                edgecolor="white", linewidth=1.5, zorder=10)
    ax1.text(0.3, 0.3, 0.7, "global min",
             color=C_GREEN, fontsize=11, fontweight="bold")
    ax1.set_title("Convex: one global minimum",
                  color=C_GREEN, fontsize=13, fontweight="bold", pad=14)
    ax1.set_xlabel("$x_1$", labelpad=2)
    ax1.set_ylabel("$x_2$", labelpad=2)
    ax1.set_zlabel("$f(x)$", labelpad=2)
    ax1.view_init(elev=24, azim=-58)

    # ----- non-convex egg-carton -----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    Z2 = (np.sin(2 * X) * np.cos(2 * Y)
          + 0.15 * (X ** 2 + Y ** 2))
    ax2.plot_surface(X, Y, Z2, cmap="Oranges", alpha=0.85, linewidth=0,
                     antialiased=True, edgecolor="none")
    ax2.scatter([0.78], [-0.78], [-0.85], color=C_GREEN, s=80,
                edgecolor="white", linewidth=1.5, zorder=10)
    ax2.scatter([-0.78], [0.78], [-0.85], color=C_RED, s=80,
                edgecolor="white", linewidth=1.5, zorder=10)
    ax2.text(0.9, -1.3, -0.4, "global min", color=C_GREEN,
             fontsize=11, fontweight="bold")
    ax2.text(-1.6, 0.9, 0.0, "local min", color=C_RED, fontsize=11)
    ax2.set_title("Non-convex: many local minima",
                  color=C_AMBER, fontsize=13, fontweight="bold", pad=14)
    ax2.set_xlabel("$x_1$", labelpad=2)
    ax2.set_ylabel("$x_2$", labelpad=2)
    ax2.set_zlabel("$f(x)$", labelpad=2)
    ax2.view_init(elev=28, azim=-58)

    fig.suptitle("Convex vs Non-Convex Functions",
                 fontsize=15, fontweight="bold", color=C_DARK, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05,
                        wspace=0.05)
    save(fig, "fig2_convex_functions")


# ---------------------------------------------------------------------------
# Figure 3: GD on convex vs non-convex landscapes (contour + paths)
# ---------------------------------------------------------------------------
def fig3_gd_convex_vs_nonconvex() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    # shared grid
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)

    # ---- convex quadratic ----
    ax = axes[0]

    def f1(p):
        return 0.5 * (3 * p[0] ** 2 + p[1] ** 2)

    def g1(p):
        return np.array([3 * p[0], p[1]])

    Z = 0.5 * (3 * X ** 2 + Y ** 2)
    cs = ax.contour(X, Y, Z, levels=[0.2, 0.6, 1.2, 2.0, 3.0, 4.5, 6.0, 8.0],
                    colors=C_GRAY, linewidths=0.8, alpha=0.7)
    ax.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.35)

    # GD path
    p = np.array([-2.6, 2.5])
    path = [p.copy()]
    for _ in range(40):
        p = p - 0.18 * g1(p)
        path.append(p.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "-o", color=C_BLUE, markersize=4,
            linewidth=1.6, markeredgecolor="white", markeredgewidth=0.6,
            label="GD trajectory")
    ax.scatter(0, 0, marker="*", s=240, color=C_GREEN, zorder=10,
               edgecolor="white", linewidth=1.2, label="global min")

    ax.set_title("Convex: GD finds the global minimum",
                 color=C_GREEN, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # ---- non-convex landscape ----
    ax = axes[1]

    def f2(p):
        x_, y_ = p
        return (np.sin(1.3 * x_) * np.cos(1.3 * y_)
                + 0.10 * (x_ ** 2 + y_ ** 2))

    def g2(p):
        x_, y_ = p
        gx = 1.3 * np.cos(1.3 * x_) * np.cos(1.3 * y_) + 0.20 * x_
        gy = -1.3 * np.sin(1.3 * x_) * np.sin(1.3 * y_) + 0.20 * y_
        return np.array([gx, gy])

    Z2 = (np.sin(1.3 * X) * np.cos(1.3 * Y)
          + 0.10 * (X ** 2 + Y ** 2))
    ax.contour(X, Y, Z2, levels=18, colors=C_GRAY, linewidths=0.8, alpha=0.65)
    ax.contourf(X, Y, Z2, levels=22, cmap="Oranges", alpha=0.40)

    # path that gets trapped in a local min; tuned step so iterates stay
    # in-frame and we can clearly see the descent settling into a basin
    p = np.array([-2.2, 2.2])
    path = [p.copy()]
    for _ in range(50):
        p = p - 0.30 * g2(p)
        path.append(p.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "-o", color=C_AMBER, markersize=4,
            linewidth=1.6, markeredgecolor="white", markeredgewidth=0.6,
            label="GD trajectory")
    ax.scatter(path[-1, 0], path[-1, 1], marker="X", s=180, color=C_RED,
               zorder=10, edgecolor="white", linewidth=1.2,
               label="local min (stuck)")

    ax.set_title("Non-convex: GD stalls at a local min",
                 color=C_AMBER, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle("Why convexity matters for gradient descent",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_gd_convex_vs_nonconvex")


# ---------------------------------------------------------------------------
# Figure 4: KKT conditions geometry
# ---------------------------------------------------------------------------
def fig4_kkt_conditions() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.6))

    # Problem:  min (x-2)^2 + (y-2)^2   s.t.  x + y <= 2,   x,y >= 0
    # The unconstrained min (2,2) is infeasible; the constrained optimum
    # is the closest feasible point: the projection onto x+y=2 -> (1, 1).
    x = np.linspace(-0.5, 3.5, 400)
    y = np.linspace(-0.5, 3.5, 400)
    X, Y = np.meshgrid(x, y)
    Z = (X - 2) ** 2 + (Y - 2) ** 2

    # objective contours
    ax.contour(X, Y, Z, levels=[0.05, 0.5, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0],
               colors=C_BLUE, linewidths=0.9, alpha=0.65)

    # feasible region: x+y<=2, x>=0, y>=0  (a triangle)
    feas = np.array([[0, 0], [2, 0], [0, 2]])
    ax.add_patch(Polygon(feas, closed=True, facecolor=C_GREEN,
                         alpha=0.18, edgecolor=C_GREEN, linewidth=1.8,
                         label="feasible region"))

    # constraint boundary line  x + y = 2
    xs = np.linspace(-0.3, 2.6, 50)
    ax.plot(xs, 2 - xs, color=C_GREEN, linewidth=2.2)
    ax.text(2.1, 0.05, r"$g(x) = x_1 + x_2 - 2 = 0$",
            color=C_GREEN, fontsize=10, rotation=-32)

    # unconstrained minimum (infeasible)
    ax.scatter(2, 2, s=110, marker="o", color=C_GRAY,
               edgecolor="white", linewidth=1.2, zorder=8)
    ax.annotate("unconstrained\nminimum (infeasible)",
                (2, 2), xytext=(18, 12), textcoords="offset points",
                fontsize=9, color=C_GRAY)

    # constrained optimum
    xs_, ys_ = 1.0, 1.0
    ax.scatter(xs_, ys_, s=220, marker="*", color=C_RED, zorder=10,
               edgecolor="white", linewidth=1.4)
    ax.annotate(r"$x^\star = (1, 1)$",
                (xs_, ys_), xytext=(-58, -28), textcoords="offset points",
                fontsize=11, color=C_RED, fontweight="bold")

    # gradient of f at x*: nabla f = 2(x-2) -> (-2,-2)
    grad_f = np.array([-2.0, -2.0])
    grad_f_n = grad_f / np.linalg.norm(grad_f) * 0.9
    ax.annotate("", xy=(xs_ + grad_f_n[0], ys_ + grad_f_n[1]),
                xytext=(xs_, ys_),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.0,
                                mutation_scale=18))
    ax.text(xs_ + grad_f_n[0] - 0.05, ys_ + grad_f_n[1] - 0.20,
            r"$-\nabla f(x^\star)$", color=C_BLUE, fontsize=11,
            fontweight="bold")

    # gradient of g at x*: nabla g = (1,1) -> shown going outward
    grad_g = np.array([1.0, 1.0]) / np.sqrt(2) * 0.9
    ax.annotate("", xy=(xs_ + grad_g[0], ys_ + grad_g[1]),
                xytext=(xs_, ys_),
                arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=2.0,
                                mutation_scale=18))
    ax.text(xs_ + grad_g[0] + 0.04, ys_ + grad_g[1] + 0.04,
            r"$\nabla g(x^\star)$", color=C_PURPLE, fontsize=11,
            fontweight="bold")

    # legend / KKT summary
    txt = (r"KKT at $x^\star$:" "\n"
           r"  $\nabla f + \lambda \nabla g = 0$" "\n"
           r"  $g(x^\star) = 0,\ \lambda \geq 0$" "\n"
           r"  $\Rightarrow \lambda = 2$  (active)")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            fontsize=10.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=C_GRAY, alpha=0.95))

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("KKT conditions: gradients align at the optimum",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    fig.tight_layout()
    save(fig, "fig4_kkt_conditions")


# ---------------------------------------------------------------------------
# Figure 5: Primal vs Dual geometric view
# ---------------------------------------------------------------------------
def fig5_duality_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    # ---- left: epigraph view of strong duality (convex problem) ----
    ax = axes[0]
    # Build the (g(x), f(x)) image set for a 1D toy:
    #   min f(x) = (x-1)^2   s.t.  g(x) = x - 0.5 <= 0
    # so feasible x <= 0.5, primal optimum at x* = 0.5, f* = 0.25
    xs = np.linspace(-1.5, 2.5, 400)
    G = xs - 0.5
    F = (xs - 1) ** 2
    # shade epigraph-like region (G, F+t) for t >= 0
    ax.fill_between(G, F, F + 6, color=C_BLUE, alpha=0.12)
    ax.plot(G, F, color=C_BLUE, linewidth=2.2, label=r"image $\{(g(x), f(x))\}$")

    # vertical line g = 0 (constraint boundary)
    ax.axvline(0, color=C_GREEN, linewidth=1.6, linestyle="--", alpha=0.9)
    ax.text(0.04, 4.8, r"$g = 0$", color=C_GREEN, fontsize=10)

    # primal optimum
    ax.scatter(0, 0.25, s=140, color=C_RED, marker="*", zorder=10,
               edgecolor="white", linewidth=1.2)
    ax.text(0.07, 0.10, r"$f^\star = 0.25$", color=C_RED, fontsize=10,
            fontweight="bold")

    # supporting hyperplane:  f + lambda * g = c, slope = -lambda.
    # For this problem dual = 1, so slope = -1; line passes through (0, 0.25).
    line_x = np.linspace(-1.5, 1.5, 50)
    line_y = 0.25 - 1.0 * line_x
    ax.plot(line_x, line_y, color=C_PURPLE, linewidth=2.0,
            label=r"supporting hyperplane (slope $-\lambda$)")
    # dual value = intercept on the f-axis
    ax.scatter(0, 0.25, s=60, color=C_PURPLE, zorder=11)
    ax.annotate(r"$d^\star = f^\star$" "\n(strong duality)",
                xy=(0, 0.25), xytext=(-1.25, 1.6),
                fontsize=10, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1))

    ax.set_xlim(-1.6, 2.2)
    ax.set_ylim(-0.4, 5.5)
    ax.set_xlabel("$g(x)$  (constraint value)")
    ax.set_ylabel("$f(x)$  (objective value)")
    ax.set_title("Convex case: strong duality ($d^\\star = f^\\star$)",
                 color=C_GREEN, fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # ---- right: weak duality / duality gap ----
    ax = axes[1]
    # Non-convex toy: image set is a curved "blob" with a notch above g=0
    t = np.linspace(0, 1, 200)
    # parametric curve for boundary
    gx = -0.6 + 1.6 * t + 0.5 * np.sin(4.0 * t)
    fx = 1.6 - 1.4 * t + 0.9 * np.sin(3.0 * t) + 0.6 * t ** 2
    # close the region: add a high-up arc
    gx2 = gx[::-1]
    fx2 = fx[::-1] + 4.0 + 0.3 * np.sin(5 * t)
    poly_x = np.concatenate([gx, gx2])
    poly_y = np.concatenate([fx, fx2])
    ax.fill(poly_x, poly_y, color=C_AMBER, alpha=0.15,
            edgecolor=C_AMBER, linewidth=1.6,
            label=r"image $\{(g(x), f(x))\}$ (non-convex)")
    ax.plot(gx, fx, color=C_AMBER, linewidth=2.0)

    ax.axvline(0, color=C_GREEN, linewidth=1.6, linestyle="--", alpha=0.9)
    ax.text(0.05, 5.0, r"$g = 0$", color=C_GREEN, fontsize=10)

    # primal optimum: lowest f on the boundary at g=0  (approx)
    # find boundary point near g=0 with min f
    mask = np.abs(gx) < 0.06
    if mask.any():
        idx = np.where(mask)[0][np.argmin(fx[mask])]
    else:
        idx = int(np.argmin(np.abs(gx)))
    f_star = fx[idx]
    ax.scatter(0, f_star, s=160, color=C_RED, marker="*", zorder=10,
               edgecolor="white", linewidth=1.2)
    ax.text(0.08, f_star + 0.05, rf"$f^\star \approx {f_star:.2f}$",
            color=C_RED, fontsize=10, fontweight="bold")

    # dual value: best supporting line from below the image at g=0
    # take a line with slope -lambda tangent to the image; dual value is
    # the intercept. For illustration, draw a line clearly below f_star.
    d_star = f_star - 0.9
    line_x = np.linspace(-1.0, 1.4, 50)
    line_y = d_star - 0.6 * line_x
    ax.plot(line_x, line_y, color=C_PURPLE, linewidth=2.0,
            label="best supporting hyperplane")
    ax.scatter(0, d_star, s=60, color=C_PURPLE, zorder=11)
    ax.text(0.08, d_star - 0.30, rf"$d^\star \approx {d_star:.2f}$",
            color=C_PURPLE, fontsize=10, fontweight="bold")

    # duality gap arrow
    ax.annotate("",
                xy=(-0.22, f_star), xytext=(-0.22, d_star),
                arrowprops=dict(arrowstyle="<->", color=C_DARK, lw=1.4))
    ax.text(-0.65, 0.5 * (f_star + d_star),
            "duality\ngap", fontsize=10, color=C_DARK,
            ha="center", va="center")

    ax.set_xlim(-1.2, 1.8)
    ax.set_ylim(-0.5, 6.0)
    ax.set_xlabel("$g(x)$  (constraint value)")
    ax.set_ylabel("$f(x)$  (objective value)")
    ax.set_title("Non-convex case: $d^\\star \\leq f^\\star$",
                 color=C_AMBER, fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle("Lagrangian duality: primal vs dual",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_duality_geometry")


# ---------------------------------------------------------------------------
# Figure 6: SGD path vs GD path on the same convex landscape
# ---------------------------------------------------------------------------
def fig6_sgd_path() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.6))

    # mildly anisotropic quadratic
    Q = np.diag([4.0, 1.0])
    b = np.zeros(2)

    def f(p):
        return 0.5 * p @ Q @ p

    def g_full(p):
        return Q @ p

    x = np.linspace(-3.0, 3.0, 200)
    y = np.linspace(-3.0, 3.0, 200)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (Q[0, 0] * X ** 2 + Q[1, 1] * Y ** 2)
    ax.contour(X, Y, Z, levels=[0.2, 0.6, 1.2, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0],
               colors=C_GRAY, linewidths=0.8, alpha=0.7)
    ax.contourf(X, Y, Z, levels=22, cmap="Blues", alpha=0.30)

    # deterministic GD path
    p = np.array([-2.7, 2.4])
    gd = [p.copy()]
    for _ in range(35):
        p = p - 0.18 * g_full(p)
        gd.append(p.copy())
    gd = np.array(gd)
    ax.plot(gd[:, 0], gd[:, 1], "-o", color=C_BLUE, markersize=4,
            linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6,
            label="GD (full gradient)")

    # SGD path: same step but add gradient noise
    rng = np.random.default_rng(7)
    p = np.array([-2.7, 2.4])
    sgd = [p.copy()]
    for _ in range(120):
        noise = rng.normal(scale=1.6, size=2)
        p = p - 0.06 * (g_full(p) + noise)
        sgd.append(p.copy())
    sgd = np.array(sgd)
    ax.plot(sgd[:, 0], sgd[:, 1], "-o", color=C_AMBER, markersize=2.8,
            linewidth=1.0, alpha=0.85,
            markeredgecolor="white", markeredgewidth=0.3,
            label="SGD (noisy gradient)")

    ax.scatter(0, 0, marker="*", s=240, color=C_GREEN, zorder=10,
               edgecolor="white", linewidth=1.2, label="optimum")

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("SGD: a noisy walk around the GD path",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig6_sgd_path")


# ---------------------------------------------------------------------------
# Figure 7: Learning rate sweep on a convex quadratic
# ---------------------------------------------------------------------------
def fig7_learning_rate() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # 1D quadratic f(x) = 0.5 * L * x^2,  L = 4
    L = 4.0

    def f(x):
        return 0.5 * L * x * x

    def g(x):
        return L * x

    titles = [
        ("Too small  ($\\alpha = 0.05$)", 0.05, C_AMBER, 25, -3.0),
        ("Just right  ($\\alpha = 0.18$)", 0.18, C_GREEN, 14, -3.0),
        ("Too large  ($\\alpha = 0.55$)", 0.55, C_RED, 6, -1.5),
    ]

    xs = np.linspace(-3.5, 3.5, 400)
    ys = f(xs)

    for ax, (title, alpha, color, n_iter, x0) in zip(axes, titles):
        ax.plot(xs, ys, color=C_GRAY, linewidth=1.5, alpha=0.8)
        ax.axhline(0, color=C_DARK, linewidth=0.6, alpha=0.5)

        x = x0
        path_x = [x]
        for _ in range(n_iter):
            x_new = x - alpha * g(x)
            # cap at frame so the divergent path stays visible
            if abs(x_new) > 3.3:
                path_x.append(np.sign(x_new) * 3.3)
                break
            path_x.append(x_new)
            x = x_new
        path_x = np.array(path_x)
        path_y = f(path_x)

        ax.plot(path_x, path_y, "o", color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        for i in range(len(path_x) - 1):
            ax.annotate("",
                        xy=(path_x[i + 1], path_y[i + 1]),
                        xytext=(path_x[i], path_y[i]),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.4, alpha=0.85))

        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=color, pad=8)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-2, 23)
        ax.set_xlabel("$x$")
        if ax is axes[0]:
            ax.set_ylabel(r"$f(x) = \frac{1}{2} L x^2$")
        ax.scatter([0], [0], s=110, marker="*", color=C_BLUE, zorder=8,
                   edgecolor="white", linewidth=1.0)

    fig.suptitle(r"Learning-rate effect on gradient descent ($L = 4,\ "
                 r"\alpha^\star = 1/L = 0.25$)",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.04)
    fig.tight_layout()
    save(fig, "fig7_learning_rate")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 04 figures (Convex Optimization Theory)...")
    fig1_convex_sets()
    fig2_convex_functions()
    fig3_gd_convex_vs_nonconvex()
    fig4_kkt_conditions()
    fig5_duality_geometry()
    fig6_sgd_path()
    fig7_learning_rate()
    print("Done.")
    print(f"  EN -> {EN_DIR}")
    print(f"  ZH -> {ZH_DIR}")


if __name__ == "__main__":
    main()
