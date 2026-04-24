"""
Figures for Linear Algebra Chapter 02: Linear Combinations and Vector Spaces.

Style: 3Blue1Brown-inspired, clean whitegrid, color palette
  Blue   #2563eb
  Purple #7c3aed
  Green  #10b981
  Amber  #f59e0b

Generates 7 figures and saves them to BOTH the EN and ZH asset folders so the
two language versions stay in sync.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS, annotate_callout  # noqa: E402, F401
setup_style()



# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
# style applied via _style.setup_style()
# plt.rcParams.update({
#     "figure.dpi": 150,
#     "savefig.dpi": 150,
#     "savefig.bbox": "tight",
#     "font.family": "DejaVu Sans",
#     "axes.titlesize": 13,
#     "axes.titleweight": "bold",
#     "axes.labelsize": 11,
#     "legend.fontsize": 10,
# })

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
AMBER = "#f59e0b"
GRAY = "#64748b"
LIGHT = "#e2e8f0"

ROOT = Path(__file__).resolve().parents[3]
EN_DIR = ROOT / "source/_posts/en/linear-algebra/02-linear-combinations-and-vector-spaces"
ZH_DIR = ROOT / "source/_posts/zh/linear-algebra/02-线性组合与向量空间"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / name)
    plt.close(fig)


def arrow(ax, x0, y0, dx, dy, color, lw=2.4, label=None, alpha=1.0):
    ax.annotate(
        "",
        xy=(x0 + dx, y0 + dy),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                        mutation_scale=18),
    )
    if label is not None:
        ax.text(x0 + dx, y0 + dy, "  " + label, color=color,
                fontsize=11, fontweight="bold", va="center")


def style_axes(ax, lim, title=None):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRAY, lw=0.8, alpha=0.5)
    ax.axvline(0, color=GRAY, lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, linestyle="--")
    if title:
        ax.set_title(title, pad=10)


# ---------------------------------------------------------------------------
# Figure 1: Linear combination a*v + b*w spans the plane
# ---------------------------------------------------------------------------
def fig1_linear_combination():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    v = np.array([2.0, 0.5])
    w = np.array([0.5, 1.8])

    # ---- Left: parallelogram construction of one combination ----
    ax = axes[0]
    style_axes(ax, 5, r"Linear combination: $\vec{u} = a\vec{v} + b\vec{w}$")

    a, b = 1.5, 1.2
    av = a * v
    bw = b * w
    u = av + bw

    # Parallelogram
    poly = Polygon([(0, 0), tuple(av), tuple(u), tuple(bw)],
                   closed=True, facecolor=PURPLE, alpha=0.12, edgecolor=PURPLE,
                   lw=1.2, linestyle="--")
    ax.add_patch(poly)

    # Scaled vectors (faded)
    arrow(ax, 0, 0, av[0], av[1], BLUE, lw=2.0, alpha=0.55)
    arrow(ax, av[0], av[1], bw[0], bw[1], GREEN, lw=2.0, alpha=0.55)

    # Original v, w
    arrow(ax, 0, 0, v[0], v[1], BLUE, label=r"$\vec{v}$")
    arrow(ax, 0, 0, w[0], w[1], GREEN, label=r"$\vec{w}$")

    # Result
    arrow(ax, 0, 0, u[0], u[1], PURPLE, lw=2.8,
          label=fr"${a}\vec{{v}}+{b}\vec{{w}}$")

    ax.text(0.02, 0.97,
            f"a = {a},  b = {b}\n"
            r"$\vec{u}=$" + f"({u[0]:.2f}, {u[1]:.2f})",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GRAY, alpha=0.9))

    # ---- Right: lattice of many combinations covers the plane ----
    ax = axes[1]
    style_axes(ax, 5, "All combinations $a\\vec{v}+b\\vec{w}$ tile the plane")

    grid = np.arange(-2.5, 2.6, 0.5)
    pts = np.array([(s * v + t * w) for s in grid for t in grid])
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=PURPLE, alpha=0.55)

    # Sample lattice lines
    for s in grid:
        line = np.array([s * v + t * w for t in [-2.5, 2.5]])
        ax.plot(line[:, 0], line[:, 1], color=BLUE, alpha=0.18, lw=0.8)
    for t in grid:
        line = np.array([s * v + t * w for s in [-2.5, 2.5]])
        ax.plot(line[:, 0], line[:, 1], color=GREEN, alpha=0.18, lw=0.8)

    arrow(ax, 0, 0, v[0], v[1], BLUE, label=r"$\vec{v}$")
    arrow(ax, 0, 0, w[0], w[1], GREEN, label=r"$\vec{w}$")

    save(fig, "fig1_linear_combination.png")


# ---------------------------------------------------------------------------
# Figure 2: Span of one vector (line) vs two non-collinear (plane)
# ---------------------------------------------------------------------------
def fig2_span_visualization():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # ---- (a) span of one vector = line ----
    ax = axes[0]
    style_axes(ax, 4, r"$\mathrm{span}\{\vec{v}\}$: a line through origin")
    v = np.array([1.5, 1.0])
    ts = np.linspace(-2.5, 2.5, 200)
    line = np.outer(ts, v)
    ax.plot(line[:, 0], line[:, 1], color=BLUE, lw=2.4, alpha=0.85)
    arrow(ax, 0, 0, v[0], v[1], BLUE, label=r"$\vec{v}$")
    # sample multiples
    for t in [-1.5, -0.5, 0.5, 1.5]:
        p = t * v
        ax.scatter(p[0], p[1], color=BLUE, s=40, zorder=5)
        ax.annotate(f"{t}"+r"$\vec{v}$", (p[0], p[1]),
                    xytext=(6, -10), textcoords="offset points", fontsize=9)

    # ---- (b) span of two parallel vectors = still a line ----
    ax = axes[1]
    style_axes(ax, 4, r"Two parallel vectors: still a line")
    v1 = np.array([1.5, 1.0])
    v2 = 1.6 * v1  # parallel
    line = np.outer(ts, v1)
    ax.plot(line[:, 0], line[:, 1], color=AMBER, lw=2.4, alpha=0.85)
    arrow(ax, 0, 0, v1[0], v1[1], BLUE, label=r"$\vec{v}_1$")
    arrow(ax, 0, 0, v2[0], v2[1], GREEN, label=r"$\vec{v}_2 = 1.6\vec{v}_1$")
    ax.text(0.02, 0.97, "No new direction\n→ span unchanged",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=AMBER, alpha=0.9))

    # ---- (c) span of two non-collinear vectors = whole plane ----
    ax = axes[2]
    style_axes(ax, 4, r"$\mathrm{span}\{\vec{v},\vec{w}\}=\mathbb{R}^2$")
    v = np.array([1.6, 0.4])
    w = np.array([0.3, 1.4])
    # Fill the visible plane
    grid = np.arange(-3.0, 3.05, 0.4)
    pts = np.array([s * v + t * w for s in grid for t in grid])
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color=PURPLE, alpha=0.45)
    arrow(ax, 0, 0, v[0], v[1], BLUE, label=r"$\vec{v}$")
    arrow(ax, 0, 0, w[0], w[1], GREEN, label=r"$\vec{w}$")

    save(fig, "fig2_span_visualization.png")


# ---------------------------------------------------------------------------
# Figure 3: Linear independence -- 3 vectors in 2D are always dependent
# ---------------------------------------------------------------------------
def fig3_linear_independence():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ---- Left: 2 independent vectors ----
    ax = axes[0]
    style_axes(ax, 4, "Independent: $\\vec{v}_1,\\vec{v}_2$ not parallel")
    v1 = np.array([2.0, 0.6])
    v2 = np.array([0.5, 1.8])
    arrow(ax, 0, 0, v1[0], v1[1], BLUE, label=r"$\vec{v}_1$")
    arrow(ax, 0, 0, v2[0], v2[1], GREEN, label=r"$\vec{v}_2$")
    # Highlight independence with a checkmark badge
    ax.text(0.02, 0.97, "Only $c_1=c_2=0$ gives $c_1\\vec{v}_1+c_2\\vec{v}_2=\\vec{0}$",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5",
                      edgecolor=GREEN, alpha=0.95))

    # ---- Right: 3 vectors in 2D -- always dependent ----
    ax = axes[1]
    style_axes(ax, 4, "3 vectors in $\\mathbb{R}^2$: always dependent")
    v1 = np.array([2.0, 0.4])
    v2 = np.array([0.3, 1.8])
    # v3 is a linear combination of v1, v2
    a, b = 0.8, 0.6
    v3 = a * v1 + b * v2

    # Show construction of v3 as combination
    arrow(ax, 0, 0, a * v1[0], a * v1[1], BLUE, lw=1.8, alpha=0.45)
    arrow(ax, a * v1[0], a * v1[1], b * v2[0], b * v2[1], GREEN, lw=1.8, alpha=0.45)
    # Dashed parallelogram
    poly = Polygon([(0, 0), tuple(a * v1), tuple(v3), tuple(b * v2)],
                   closed=True, facecolor=AMBER, alpha=0.10,
                   edgecolor=AMBER, lw=1.0, linestyle="--")
    ax.add_patch(poly)

    arrow(ax, 0, 0, v1[0], v1[1], BLUE, label=r"$\vec{v}_1$")
    arrow(ax, 0, 0, v2[0], v2[1], GREEN, label=r"$\vec{v}_2$")
    arrow(ax, 0, 0, v3[0], v3[1], AMBER, lw=2.8,
          label=fr"$\vec{{v}}_3={a}\vec{{v}}_1+{b}\vec{{v}}_2$")

    ax.text(0.02, 0.97,
            "Any third vector lies in\n"
            r"$\mathrm{span}\{\vec{v}_1,\vec{v}_2\}=\mathbb{R}^2$"
            "\n→ redundant",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb",
                      edgecolor=AMBER, alpha=0.95))

    save(fig, "fig3_linear_independence.png")


# ---------------------------------------------------------------------------
# Figure 4: Two bases of R^2 -- standard vs rotated
# ---------------------------------------------------------------------------
def fig4_basis_examples():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    target = np.array([3.0, 2.0])

    # ---- Standard basis ----
    ax = axes[0]
    style_axes(ax, 4.5, "Standard basis $\\{\\vec{e}_1, \\vec{e}_2\\}$")
    # grid lines along basis
    for k in range(-4, 5):
        ax.plot([k, k], [-4, 4], color=BLUE, alpha=0.12, lw=0.6)
        ax.plot([-4, 4], [k, k], color=GREEN, alpha=0.12, lw=0.6)

    arrow(ax, 0, 0, 1, 0, BLUE, label=r"$\vec{e}_1$")
    arrow(ax, 0, 0, 0, 1, GREEN, label=r"$\vec{e}_2$")

    # Decompose target
    arrow(ax, 0, 0, target[0], 0, BLUE, lw=1.6, alpha=0.45)
    arrow(ax, target[0], 0, 0, target[1], GREEN, lw=1.6, alpha=0.45)
    arrow(ax, 0, 0, target[0], target[1], PURPLE, lw=2.6,
          label=r"$\vec{u}=3\vec{e}_1+2\vec{e}_2$")
    ax.scatter(*target, color=PURPLE, s=60, zorder=5)
    ax.annotate(r"$(3,2)$", target, xytext=(8, 6),
                textcoords="offset points", fontsize=11, color=PURPLE)

    # ---- Rotated basis ----
    ax = axes[1]
    style_axes(ax, 4.5, "Rotated basis $\\{\\vec{b}_1,\\vec{b}_2\\}$ — same vector, new coords")
    theta = np.deg2rad(30)
    b1 = np.array([np.cos(theta), np.sin(theta)])
    b2 = np.array([-np.sin(theta), np.cos(theta)])

    # grid lines along rotated basis
    for k in range(-4, 5):
        # lines parallel to b2, offset by k*b1
        p0 = k * b1 - 5 * b2
        p1 = k * b1 + 5 * b2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BLUE, alpha=0.12, lw=0.6)
        p0 = -5 * b1 + k * b2
        p1 = 5 * b1 + k * b2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GREEN, alpha=0.12, lw=0.6)

    arrow(ax, 0, 0, b1[0], b1[1], BLUE, label=r"$\vec{b}_1$")
    arrow(ax, 0, 0, b2[0], b2[1], GREEN, label=r"$\vec{b}_2$")

    # Solve target = c1*b1 + c2*b2
    M = np.column_stack([b1, b2])
    c1, c2 = np.linalg.solve(M, target)

    arrow(ax, 0, 0, c1 * b1[0], c1 * b1[1], BLUE, lw=1.6, alpha=0.45)
    arrow(ax, c1 * b1[0], c1 * b1[1], c2 * b2[0], c2 * b2[1], GREEN, lw=1.6, alpha=0.45)
    arrow(ax, 0, 0, target[0], target[1], PURPLE, lw=2.6,
          label=fr"$\vec{{u}}={c1:.2f}\vec{{b}}_1+{c2:.2f}\vec{{b}}_2$")
    ax.scatter(*target, color=PURPLE, s=60, zorder=5)
    ax.annotate(f"({c1:.2f}, {c2:.2f})\nin new basis", target,
                xytext=(8, 6), textcoords="offset points", fontsize=10, color=PURPLE)

    save(fig, "fig4_basis_examples.png")


# ---------------------------------------------------------------------------
# Figure 5: Change of basis -- coordinate grids before / after
# ---------------------------------------------------------------------------
def fig5_change_of_basis():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    point = np.array([2.5, 1.5])

    # ---- Standard basis grid ----
    ax = axes[0]
    style_axes(ax, 4.5, "Same point in standard basis: $(2.5, 1.5)$")
    for k in range(-4, 5):
        ax.plot([k, k], [-4, 4], color=BLUE, alpha=0.18, lw=0.8)
        ax.plot([-4, 4], [k, k], color=GREEN, alpha=0.18, lw=0.8)
    arrow(ax, 0, 0, 1, 0, BLUE, label=r"$\vec{e}_1$")
    arrow(ax, 0, 0, 0, 1, GREEN, label=r"$\vec{e}_2$")
    ax.scatter(*point, color=PURPLE, s=80, zorder=5)
    ax.annotate(r"$P=(2.5, 1.5)$", point, xytext=(8, 8),
                textcoords="offset points", fontsize=11, color=PURPLE,
                fontweight="bold")

    # ---- Sheared basis grid ----
    ax = axes[1]
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.6, 1.0])  # sheared
    M = np.column_stack([b1, b2])
    c1, c2 = np.linalg.solve(M, point)
    style_axes(ax, 4.5,
               f"Same point in sheared basis: $({c1:.2f}, {c2:.2f})$")
    for k in range(-4, 5):
        p0 = k * b1 - 5 * b2
        p1 = k * b1 + 5 * b2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BLUE, alpha=0.18, lw=0.8)
        p0 = -5 * b1 + k * b2
        p1 = 5 * b1 + k * b2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GREEN, alpha=0.18, lw=0.8)
    arrow(ax, 0, 0, b1[0], b1[1], BLUE, label=r"$\vec{b}_1$")
    arrow(ax, 0, 0, b2[0], b2[1], GREEN, label=r"$\vec{b}_2$")
    ax.scatter(*point, color=PURPLE, s=80, zorder=5)
    ax.annotate(f"$P=({c1:.2f}, {c2:.2f})$", point, xytext=(8, 8),
                textcoords="offset points", fontsize=11, color=PURPLE,
                fontweight="bold")

    fig.suptitle("Change of basis: the point doesn't move, the coordinates do",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "fig5_change_of_basis.png")


# ---------------------------------------------------------------------------
# Figure 6: Subspace examples in R^3 -- line and plane through origin
# ---------------------------------------------------------------------------
def fig6_subspaces_3d():
    fig = plt.figure(figsize=(14, 6))

    # ---- Line through origin ----
    ax = fig.add_subplot(121, projection="3d")
    ax.set_title("1D subspace: line through origin\n"
                 r"$\mathrm{span}\{\vec{v}\}$", pad=12, fontweight="bold")
    v = np.array([1.0, 0.7, 0.5])
    ts = np.linspace(-2.5, 2.5, 100)
    line = np.outer(ts, v)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color=BLUE, lw=3, alpha=0.9)
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color=BLUE, lw=2.5,
              arrow_length_ratio=0.15)
    ax.scatter([0], [0], [0], color="red", s=60, zorder=10, label="origin")
    _style_3d(ax)

    # ---- Plane through origin ----
    ax = fig.add_subplot(122, projection="3d")
    ax.set_title("2D subspace: plane through origin\n"
                 r"$\mathrm{span}\{\vec{v},\vec{w}\}$", pad=12, fontweight="bold")
    v = np.array([1.5, 0.2, 0.3])
    w = np.array([0.3, 1.4, 0.4])
    s = np.linspace(-1.5, 1.5, 25)
    t = np.linspace(-1.5, 1.5, 25)
    S, T = np.meshgrid(s, t)
    X = S * v[0] + T * w[0]
    Y = S * v[1] + T * w[1]
    Z = S * v[2] + T * w[2]
    ax.plot_surface(X, Y, Z, alpha=0.35, color=PURPLE, edgecolor=PURPLE,
                    linewidth=0.2, antialiased=True)
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color=BLUE, lw=2.5,
              arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, w[0], w[1], w[2], color=GREEN, lw=2.5,
              arrow_length_ratio=0.15)
    ax.scatter([0], [0], [0], color="red", s=60, zorder=10, label="origin")
    _style_3d(ax)

    save(fig, "fig6_subspaces_3d.png")


def _style_3d(ax):
    lim = 2.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=35)
    # Background panes lighter
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0.0))
        pane.set_edgecolor(GRAY)


# ---------------------------------------------------------------------------
# Figure 7: Affine vs linear -- subspaces must contain the origin
# ---------------------------------------------------------------------------
def fig7_affine_vs_linear():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ---- Linear subspace: line through origin ----
    ax = axes[0]
    style_axes(ax, 4, "Linear subspace $\\checkmark$\nline through origin")
    v = np.array([1.5, 1.0])
    ts = np.linspace(-2.5, 2.5, 200)
    line = np.outer(ts, v)
    ax.plot(line[:, 0], line[:, 1], color=GREEN, lw=2.6)
    ax.scatter(0, 0, color="red", s=110, zorder=5,
               edgecolor="white", linewidth=2)
    ax.annotate("contains $\\vec{0}$", (0, 0), xytext=(12, 12),
                textcoords="offset points", fontsize=11, color="red",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"))
    # Demonstrate closure: u + v stays on line
    u = -1.2 * v
    w = 0.8 * v
    ax.scatter(*u, color=BLUE, s=70, zorder=5)
    ax.scatter(*w, color=BLUE, s=70, zorder=5)
    s_pt = u + w
    ax.scatter(*s_pt, color=PURPLE, s=90, zorder=6)
    ax.annotate("$\\vec{u}$", u, xytext=(-22, -12),
                textcoords="offset points", fontsize=11, color=BLUE)
    ax.annotate("$\\vec{w}$", w, xytext=(8, 6),
                textcoords="offset points", fontsize=11, color=BLUE)
    ax.annotate("$\\vec{u}+\\vec{w}$", s_pt, xytext=(-30, -16),
                textcoords="offset points", fontsize=11, color=PURPLE)

    ax.text(0.02, 0.97,
            "Closed under + and scaling\n→ a subspace",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5",
                      edgecolor=GREEN, alpha=0.95))

    # ---- Affine set: line NOT through origin ----
    ax = axes[1]
    style_axes(ax, 4, "Affine set $\\times$\nline NOT through origin")
    # line: y = x + 1.5
    xs = np.linspace(-3, 3, 200)
    ys = xs + 1.5
    ax.plot(xs, ys, color=AMBER, lw=2.6)
    ax.scatter(0, 0, color="red", s=110, zorder=5,
               edgecolor="white", linewidth=2)
    ax.annotate("origin NOT on line", (0, 0), xytext=(20, -22),
                textcoords="offset points", fontsize=11, color="red",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"))
    # Two points and their sum
    p1 = np.array([1.0, 2.5])
    p2 = np.array([-2.0, -0.5])
    s_pt = p1 + p2  # (-1, 2) — NOT on the line
    ax.scatter(*p1, color=BLUE, s=70, zorder=5)
    ax.scatter(*p2, color=BLUE, s=70, zorder=5)
    ax.scatter(*s_pt, color=PURPLE, s=90, zorder=6)
    ax.annotate("$\\vec{p}_1$", p1, xytext=(8, 6),
                textcoords="offset points", fontsize=11, color=BLUE)
    ax.annotate("$\\vec{p}_2$", p2, xytext=(8, -14),
                textcoords="offset points", fontsize=11, color=BLUE)
    ax.annotate("$\\vec{p}_1+\\vec{p}_2$\n(off the line!)", s_pt,
                xytext=(-95, 10), textcoords="offset points",
                fontsize=10, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE))

    ax.text(0.02, 0.97,
            "Sum leaves the set\n→ NOT a subspace",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb",
                      edgecolor=AMBER, alpha=0.95))

    fig.suptitle("Subspaces must pass through the origin",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "fig7_affine_vs_linear.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    fig1_linear_combination()
    print("  fig1_linear_combination.png")
    fig2_span_visualization()
    print("  fig2_span_visualization.png")
    fig3_linear_independence()
    print("  fig3_linear_independence.png")
    fig4_basis_examples()
    print("  fig4_basis_examples.png")
    fig5_change_of_basis()
    print("  fig5_change_of_basis.png")
    fig6_subspaces_3d()
    print("  fig6_subspaces_3d.png")
    fig7_affine_vs_linear()
    print("  fig7_affine_vs_linear.png")
    print("Done.")


if __name__ == "__main__":
    main()
