"""
Figure generation script for Linear Algebra Chapter 11:
Matrix Calculus and Optimization.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly.

Figures:
    fig1_gradient_steepest_ascent  Gradient as direction of steepest ascent:
                                   3D bowl on the left, contour map with
                                   gradient arrows on the right.
    fig2_critical_points           Three critical-point types side by side:
                                   minimum (positive-definite Hessian),
                                   maximum (negative-definite),
                                   saddle (indefinite). 3D + Hessian eigenvalue
                                   annotation.
    fig3_gradient_descent_path     Gradient descent trajectory on a contour
                                   plot of an anisotropic quadratic, showing
                                   zig-zag for an ill-conditioned problem.
    fig4_newton_vs_gd              Newton's method vs gradient descent:
                                   Newton uses curvature (Hessian) and reaches
                                   the minimum in one step on a quadratic.
    fig5_convex_vs_nonconvex       Convex vs non-convex 3D surfaces:
                                   bowl (one global min) vs egg-carton
                                   (many local minima).
    fig6_shape_rules_cheatsheet    Visual cheatsheet of vector and matrix
                                   derivative shape rules: input shape, output
                                   shape, gradient shape.
    fig7_backprop_chain_rule       Backpropagation as the chain rule on a
                                   computation graph: forward pass values and
                                   backward pass gradients along the same DAG.

Usage:
    python3 scripts/figures/linear-algebra/11-matrix-calculus-and-optimization.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS, annotate_callout  # noqa: E402, F401
setup_style()



# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# style applied via _style.setup_style()

C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "11-matrix-calculus-and-optimization"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "11-矩阵微积分与优化"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH EN and ZH asset folders as <name>.png."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        path = d / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_3d(ax):
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor(C_LIGHT)
    ax.yaxis.pane.set_edgecolor(C_LIGHT)
    ax.zaxis.pane.set_edgecolor(C_LIGHT)
    ax.tick_params(labelsize=8, colors=C_DARK)
    ax.xaxis.label.set_color(C_DARK)
    ax.yaxis.label.set_color(C_DARK)
    ax.zaxis.label.set_color(C_DARK)


def _style_2d(ax):
    ax.tick_params(labelsize=8, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_color(C_GRAY)
        spine.set_linewidth(0.6)
    ax.grid(True, alpha=0.25, lw=0.6)


# ---------------------------------------------------------------------------
# Fig 1: Gradient = direction of steepest ascent
# ---------------------------------------------------------------------------
def fig1_gradient_steepest_ascent():
    fig = plt.figure(figsize=(13, 5.4))

    # Left: 3D surface + gradient arrow
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    x = np.linspace(-2.2, 2.2, 80)
    y = np.linspace(-2.2, 2.2, 80)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * X**2 + 0.5 * Y**2  # bowl
    ax3d.plot_surface(X, Y, Z, cmap="Blues", alpha=0.7,
                      linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax3d.contour(X, Y, Z, zdir="z", offset=0, levels=8,
                 colors=C_GRAY, linewidths=0.5, alpha=0.7)

    # Pick a point and draw gradient (in xy plane, projected up).
    p = np.array([1.4, 0.9])
    grad = p  # gradient of 0.5(x^2+y^2) is (x, y)
    z_p = 0.5 * (p[0]**2 + p[1]**2)
    ax3d.scatter([p[0]], [p[1]], [z_p], color=C_PURPLE, s=40, zorder=5)
    # Gradient arrow on the floor (z=0).
    g_dir = grad / np.linalg.norm(grad) * 1.0
    ax3d.quiver(p[0], p[1], 0, g_dir[0], g_dir[1], 0,
                color=C_PURPLE, lw=2.2, arrow_length_ratio=0.18)
    ax3d.text(p[0] + g_dir[0] + 0.05, p[1] + g_dir[1] + 0.05, 0.05,
              r"$\nabla f$", color=C_PURPLE, fontsize=12, fontweight="bold")

    ax3d.set_title(r"Surface $f(x,y) = \frac{1}{2}(x^2+y^2)$",
                   fontsize=11, color=C_DARK, fontweight="bold", pad=8)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("f")
    ax3d.view_init(elev=28, azim=-60)
    _style_3d(ax3d)

    # Right: contour + gradient field
    ax = fig.add_subplot(1, 2, 2)
    _style_2d(ax)
    cs = ax.contour(X, Y, Z, levels=10, colors=[C_BLUE], linewidths=0.9,
                    alpha=0.85)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # Gradient arrows on a coarse grid
    xs = np.linspace(-1.8, 1.8, 7)
    ys = np.linspace(-1.8, 1.8, 7)
    Xs, Ys = np.meshgrid(xs, ys)
    Gx, Gy = Xs, Ys
    norms = np.sqrt(Gx**2 + Gy**2) + 1e-9
    Gx_n = Gx / norms * 0.6
    Gy_n = Gy / norms * 0.6
    ax.quiver(Xs, Ys, Gx_n, Gy_n, color=C_PURPLE, alpha=0.85,
              angles="xy", scale_units="xy", scale=1, width=0.005)

    # Highlight perpendicularity at one point
    pt = np.array([1.3, 0.8])
    g = pt / np.linalg.norm(pt) * 0.9
    # Tangent to contour (perpendicular to gradient)
    t = np.array([-g[1], g[0]])
    ax.add_patch(FancyArrowPatch(tuple(pt), tuple(pt + g),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=C_PURPLE, lw=2.2, zorder=5))
    ax.add_patch(FancyArrowPatch(tuple(pt - 0.5 * t), tuple(pt + 0.5 * t),
                                 arrowstyle="-", mutation_scale=10,
                                 color=C_GREEN, lw=2.2, zorder=5))
    ax.text(pt[0] + g[0] + 0.05, pt[1] + g[1] - 0.05,
            r"$\nabla f$ (uphill)", color=C_PURPLE,
            fontsize=10, fontweight="bold")
    ax.text(pt[0] - 0.6 * t[0] - 0.1, pt[1] - 0.6 * t[1] - 0.05,
            "level curve", color=C_GREEN, fontsize=9, fontweight="bold")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_title("Contour view: gradient perpendicular to level curves",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.suptitle("Gradient points in the direction of steepest ascent",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig1_gradient_steepest_ascent")


# ---------------------------------------------------------------------------
# Fig 2: Hessian classifies critical points
# ---------------------------------------------------------------------------
def fig2_critical_points():
    fig = plt.figure(figsize=(14.5, 5.0))
    x = np.linspace(-2, 2, 60)
    y = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(x, y)

    cases = [
        {
            "title": "Local minimum",
            "subtitle": r"$f = x^2 + y^2$",
            "Z": X**2 + Y**2,
            "eigs": "eigs(H) = (2, 2) > 0\n(positive definite)",
            "cmap": "Greens", "color": C_GREEN,
        },
        {
            "title": "Local maximum",
            "subtitle": r"$f = -(x^2 + y^2)$",
            "Z": -(X**2 + Y**2),
            "eigs": "eigs(H) = (-2, -2) < 0\n(negative definite)",
            "cmap": "Reds", "color": C_RED,
        },
        {
            "title": "Saddle point",
            "subtitle": r"$f = x^2 - y^2$",
            "Z": X**2 - Y**2,
            "eigs": "eigs(H) = (2, -2)\n(indefinite)",
            "cmap": "Oranges", "color": C_AMBER,
        },
    ]

    for i, case in enumerate(cases, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(X, Y, case["Z"], cmap=case["cmap"], alpha=0.85,
                        linewidth=0, rstride=2, cstride=2, antialiased=True)
        ax.contour(X, Y, case["Z"],
                   zdir="z",
                   offset=case["Z"].min() - 0.5,
                   levels=8, colors=C_GRAY, linewidths=0.5, alpha=0.7)
        # Mark critical point
        ax.scatter([0], [0], [float(case["Z"][30, 30])],
                   color=case["color"], s=70, zorder=10, edgecolor=C_DARK,
                   linewidth=0.8)
        ax.set_title(f"{case['title']}\n{case['subtitle']}",
                     fontsize=11, color=C_DARK, fontweight="bold", pad=8)
        ax.text2D(0.05, -0.02, case["eigs"], transform=ax.transAxes,
                  fontsize=9, color=case["color"], fontweight="bold",
                  ha="left", va="top")
        ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("f", fontsize=8)
        ax.view_init(elev=28, azim=-58)
        _style_3d(ax)

    fig.suptitle("Hessian eigenvalues classify critical points",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_critical_points")


# ---------------------------------------------------------------------------
# Fig 3: Gradient descent path on a contour plot
# ---------------------------------------------------------------------------
def fig3_gradient_descent_path():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Anisotropic quadratic: f = 0.5 (a x^2 + b y^2)
    A_well = np.diag([2.0, 2.0])           # well conditioned
    A_ill = np.diag([10.0, 1.0])           # ill conditioned

    def run_gd(A, x0, lr, steps):
        x = x0.copy()
        path = [x.copy()]
        for _ in range(steps):
            g = A @ x
            x = x - lr * g
            path.append(x.copy())
        return np.array(path)

    cases = [
        ("Well-conditioned: $\\kappa = 1$",
         A_well, np.array([2.5, 2.0]), 0.18, 30),
        ("Ill-conditioned: $\\kappa = 10$ → zig-zag",
         A_ill, np.array([2.5, 2.0]), 0.18, 50),
    ]

    for ax, (title, A, x0, lr, steps) in zip(axes, cases):
        _style_2d(ax)
        x = np.linspace(-3, 3, 200)
        y = np.linspace(-3, 3, 200)
        X, Y = np.meshgrid(x, y)
        Z = 0.5 * (A[0, 0] * X**2 + A[1, 1] * Y**2)
        ax.contour(X, Y, Z, levels=18, colors=[C_BLUE],
                   linewidths=0.8, alpha=0.7)

        path = run_gd(A, x0, lr, steps)
        ax.plot(path[:, 0], path[:, 1], "-", color=C_AMBER, lw=1.6,
                alpha=0.9, zorder=4)
        ax.scatter(path[:, 0], path[:, 1], s=22, color=C_AMBER,
                   edgecolor=C_DARK, linewidth=0.4, zorder=5)
        # Start and end markers
        ax.scatter([path[0, 0]], [path[0, 1]], s=90, color=C_PURPLE,
                   edgecolor=C_DARK, lw=0.8, zorder=6, label="start")
        ax.scatter([0], [0], s=110, marker="*", color=C_GREEN,
                   edgecolor=C_DARK, lw=0.6, zorder=6, label="minimum")
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, color=C_DARK,
                     fontweight="bold", pad=8)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.suptitle("Gradient descent: condition number controls the path",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig3_gradient_descent_path")


# ---------------------------------------------------------------------------
# Fig 4: Newton vs gradient descent
# ---------------------------------------------------------------------------
def fig4_newton_vs_gd():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    A = np.diag([8.0, 1.0])

    def f(x):
        return 0.5 * (A[0, 0] * x[0]**2 + A[1, 1] * x[1]**2)

    def grad(x):
        return A @ x

    H = A
    H_inv = np.linalg.inv(H)
    x0 = np.array([2.5, 2.5])

    # Gradient descent
    x = x0.copy()
    gd_path = [x.copy()]
    for _ in range(40):
        g = grad(x)
        x = x - 0.2 * g
        gd_path.append(x.copy())
    gd_path = np.array(gd_path)

    # Newton's method (one step for a quadratic suffices)
    x = x0.copy()
    nt_path = [x.copy()]
    for _ in range(3):
        g = grad(x)
        x = x - H_inv @ g
        nt_path.append(x.copy())
    nt_path = np.array(nt_path)

    paths = [
        ("Gradient descent (zig-zag)", gd_path, C_AMBER),
        ("Newton's method (1 step on quadratic)", nt_path, C_PURPLE),
    ]

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (A[0, 0] * X**2 + A[1, 1] * Y**2)

    for ax, (title, path, color) in zip(axes, paths):
        _style_2d(ax)
        ax.contour(X, Y, Z, levels=18, colors=[C_BLUE],
                   linewidths=0.8, alpha=0.7)
        ax.plot(path[:, 0], path[:, 1], "-", color=color, lw=1.8,
                alpha=0.9, zorder=4)
        ax.scatter(path[:, 0], path[:, 1], s=28, color=color,
                   edgecolor=C_DARK, linewidth=0.4, zorder=5)
        ax.scatter([path[0, 0]], [path[0, 1]], s=90, color=C_BLUE,
                   edgecolor=C_DARK, lw=0.8, zorder=6, label="start")
        ax.scatter([0], [0], s=120, marker="*", color=C_GREEN,
                   edgecolor=C_DARK, lw=0.6, zorder=6, label="minimum")
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, color=C_DARK,
                     fontweight="bold", pad=8)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # Footer formula reminder
    fig.text(0.5, -0.02,
             r"GD: $x_{k+1} = x_k - \alpha\,\nabla f$    "
             r"Newton: $x_{k+1} = x_k - H^{-1}\nabla f$",
             ha="center", fontsize=11, color=C_DARK)
    fig.suptitle("Newton uses curvature; jumps to the minimum on a quadratic",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig4_newton_vs_gd")


# ---------------------------------------------------------------------------
# Fig 5: Convex vs non-convex
# ---------------------------------------------------------------------------
def fig5_convex_vs_nonconvex():
    fig = plt.figure(figsize=(13.5, 5.4))

    # Left: convex bowl
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    x = np.linspace(-2.2, 2.2, 80)
    y = np.linspace(-2.2, 2.2, 80)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    ax1.plot_surface(X, Y, Z, cmap="Greens", alpha=0.85,
                     linewidth=0, rstride=2, cstride=2, antialiased=True)
    ax1.scatter([0], [0], [0], color=C_GREEN, s=80, edgecolor=C_DARK,
                lw=0.8, zorder=10)
    ax1.text2D(0.05, 0.93, "Convex: any local min = global min",
               transform=ax1.transAxes, fontsize=10,
               color=C_GREEN, fontweight="bold")
    ax1.set_title(r"Convex   $f = x^2 + y^2$", fontsize=11,
                  color=C_DARK, fontweight="bold", pad=8)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f")
    ax1.view_init(elev=28, azim=-60)
    _style_3d(ax1)

    # Right: non-convex (egg-carton-like)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y) + 0.05 * (X**2 + Y**2)
    ax2.plot_surface(X, Y, Z, cmap="Oranges", alpha=0.9,
                     linewidth=0, rstride=2, cstride=2, antialiased=True)
    # Mark several local minima approximately
    for px, py in [(-1.55, 0.0), (1.55, 0.0), (-1.55, np.pi),
                   (1.55, np.pi), (-1.55, -np.pi), (1.55, -np.pi)]:
        if -3 <= px <= 3 and -3 <= py <= 3:
            zp = np.sin(px) * np.cos(py) + 0.05 * (px**2 + py**2)
            ax2.scatter([px], [py], [zp], color=C_AMBER, s=45,
                        edgecolor=C_DARK, lw=0.6, zorder=10)
    ax2.text2D(0.05, 0.93, "Non-convex: many local minima",
               transform=ax2.transAxes, fontsize=10,
               color=C_AMBER, fontweight="bold")
    ax2.set_title(r"Non-convex   $f = \sin x \cos y + 0.05(x^2+y^2)$",
                  fontsize=11, color=C_DARK, fontweight="bold", pad=8)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("f")
    ax2.view_init(elev=32, azim=-58)
    _style_3d(ax2)

    fig.suptitle("Convexity is what makes optimization easy",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig5_convex_vs_nonconvex")


# ---------------------------------------------------------------------------
# Fig 6: Shape rules cheatsheet
# ---------------------------------------------------------------------------
def fig6_shape_rules_cheatsheet():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Header
    ax.text(5, 5.7, "Shape rules: derivative has the shape of the variable",
            ha="center", fontsize=14, color=C_DARK, fontweight="bold")

    # Column headers: variable | function f | derivative shape
    headers = ["Variable x", "Function f(x)", r"Shape of $\partial f / \partial x$",
               "Example"]
    xpos = [1.0, 3.4, 6.1, 8.5]
    for xp, h in zip(xpos, headers):
        ax.text(xp, 5.0, h, ha="center", fontsize=11,
                color=C_BLUE, fontweight="bold")
    ax.plot([0.4, 9.6], [4.75, 4.75], color=C_GRAY, lw=1)

    rows = [
        # (var label, f desc, shape of derivative, example)
        ("scalar  $x \\in \\mathbb{R}$",
         "scalar  $f(x)$",
         "scalar",
         r"$\frac{d}{dx}x^2 = 2x$"),
        ("vector  $\\vec{x} \\in \\mathbb{R}^n$",
         "scalar  $f(\\vec{x})$",
         r"vector $\nabla f \in \mathbb{R}^n$",
         r"$\nabla(\vec{a}^\top \vec{x}) = \vec{a}$"),
        ("vector  $\\vec{x} \\in \\mathbb{R}^n$",
         "vector  $\\vec{f}(\\vec{x}) \\in \\mathbb{R}^m$",
         r"matrix $J \in \mathbb{R}^{m \times n}$",
         r"$J_{ij} = \partial f_i / \partial x_j$"),
        ("matrix  $X \\in \\mathbb{R}^{m \\times n}$",
         "scalar  $f(X)$",
         r"matrix $\partial f/\partial X \in \mathbb{R}^{m \times n}$",
         r"$\partial\,\mathrm{tr}(AX)/\partial X = A^\top$"),
        ("vector  $\\vec{x} \\in \\mathbb{R}^n$",
         "scalar  $f(\\vec{x})$  (2nd order)",
         r"matrix $H \in \mathbb{R}^{n \times n}$",
         r"$H_{ij} = \partial^2 f /\partial x_i \partial x_j$"),
    ]

    y = 4.25
    for var, fn, sh, ex in rows:
        ax.text(xpos[0], y, var, ha="center", va="center",
                fontsize=10, color=C_DARK)
        ax.text(xpos[1], y, fn, ha="center", va="center",
                fontsize=10, color=C_DARK)
        ax.text(xpos[2], y, sh, ha="center", va="center",
                fontsize=10, color=C_PURPLE, fontweight="bold")
        ax.text(xpos[3], y, ex, ha="center", va="center",
                fontsize=10, color=C_GREEN)
        y -= 0.7
        ax.plot([0.4, 9.6], [y + 0.35, y + 0.35],
                color=C_LIGHT, lw=0.6)

    # Bottom takeaway box
    ax.add_patch(FancyBboxPatch((0.5, 0.15), 9.0, 0.7,
                                boxstyle="round,pad=0.06",
                                facecolor="#fff7ed",
                                edgecolor=C_AMBER, lw=1.2))
    ax.text(5.0, 0.5,
            "Rule of thumb:  $\\partial(\\mathrm{scalar}) / \\partial X$ "
            "has the same shape as $X$.   "
            "$\\partial(\\mathrm{vector}_m) / \\partial(\\mathrm{vector}_n)$ "
            "is an $m \\times n$ matrix.",
            ha="center", va="center", fontsize=11, color=C_DARK)

    save(fig, "fig6_shape_rules_cheatsheet")


# ---------------------------------------------------------------------------
# Fig 7: Backpropagation as chain rule on a computation graph
# ---------------------------------------------------------------------------
def fig7_backprop_chain_rule():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6, 5.6, "Backpropagation = chain rule on a computation graph",
            ha="center", fontsize=14, color=C_DARK, fontweight="bold")

    # Forward / backward labels
    ax.text(0.4, 4.85, "Forward", color=C_BLUE,
            fontsize=12, fontweight="bold")
    ax.text(0.4, 0.95, "Backward", color=C_PURPLE,
            fontsize=12, fontweight="bold")

    # Nodes for forward pass: x  ->  z = Wx+b  ->  a = sigma(z)  ->  L
    fwd_nodes = [
        (1.8, 4.0, r"$\vec{x}$", C_BLUE),
        (4.6, 4.0, r"$\vec{z} = W\vec{x}+\vec{b}$", C_BLUE),
        (7.6, 4.0, r"$\vec{a} = \sigma(\vec{z})$", C_BLUE),
        (10.4, 4.0, r"$L$", C_BLUE),
    ]
    for cx, cy, label, color in fwd_nodes:
        ax.add_patch(FancyBboxPatch((cx - 0.95, cy - 0.45), 1.9, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#eff6ff",
                                    edgecolor=color, lw=1.4))
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=12, color=color, fontweight="bold")

    # Forward arrows
    for i in range(3):
        x1 = fwd_nodes[i][0] + 0.95
        x2 = fwd_nodes[i + 1][0] - 0.95
        ax.add_patch(FancyArrowPatch((x1, 4.0), (x2, 4.0),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=C_BLUE, lw=1.6))

    # Backward nodes (gradients)
    bwd_nodes = [
        (1.8, 1.6, r"$\partial L/\partial \vec{x}=W^\top \delta_z$", C_PURPLE),
        (4.6, 1.6, r"$\delta_z = \delta_a \odot \sigma'(\vec{z})$", C_PURPLE),
        (7.6, 1.6, r"$\delta_a = \partial L/\partial \vec{a}$", C_PURPLE),
        (10.4, 1.6, r"$\partial L/\partial L = 1$", C_PURPLE),
    ]
    for cx, cy, label, color in bwd_nodes:
        ax.add_patch(FancyBboxPatch((cx - 1.15, cy - 0.45), 2.3, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#f5f3ff",
                                    edgecolor=color, lw=1.4))
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

    # Backward arrows (right -> left)
    for i in range(3):
        x1 = bwd_nodes[i + 1][0] - 1.15
        x2 = bwd_nodes[i][0] + 1.15
        ax.add_patch(FancyArrowPatch((x1, 1.6), (x2, 1.6),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=C_PURPLE, lw=1.6))

    # Vertical connectors (each forward node has a matching backward gradient)
    for (cx, _, _, _), (_, _, _, _) in zip(fwd_nodes, bwd_nodes):
        ax.plot([cx, cx], [3.55, 2.05], color=C_GRAY,
                lw=0.8, ls="--", alpha=0.7)

    # Side-note: parameter gradients harvested along the way
    ax.add_patch(FancyBboxPatch((1.0, 0.05), 10.0, 0.7,
                                boxstyle="round,pad=0.06",
                                facecolor="#ecfdf5",
                                edgecolor=C_GREEN, lw=1.2))
    ax.text(6.0, 0.4,
            r"Parameter gradients harvested in the backward pass:   "
            r"$\partial L/\partial W = \delta_z\,\vec{x}^\top$,   "
            r"$\partial L/\partial \vec{b} = \delta_z$",
            ha="center", va="center", fontsize=11, color=C_DARK)

    save(fig, "fig7_backprop_chain_rule")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig1_gradient_steepest_ascent()
    fig2_critical_points()
    fig3_gradient_descent_path()
    fig4_newton_vs_gd()
    fig5_convex_vs_nonconvex()
    fig6_shape_rules_cheatsheet()
    fig7_backprop_chain_rule()
    print("All 7 figures saved to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
