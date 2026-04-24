"""
Chapter 05: Linear Systems and Column Space
3Blue1Brown-style figures for the linear algebra series.

Generates 7 figures and saves them to BOTH the EN and ZH asset folders.

Style:
- seaborn-v0_8-whitegrid
- dpi=150
- palette: #2563eb (blue), #7c3aed (violet), #10b981 (emerald), #f59e0b (amber)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon, Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# ---------- Style ----------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#374151",
    "axes.linewidth": 0.9,
    "grid.color": "#d1d5db",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.55,
})

BLUE = "#2563eb"
VIOLET = "#7c3aed"
EMERALD = "#10b981"
AMBER = "#f59e0b"
GRAY = "#6b7280"
DARK = "#111827"
LIGHT = "#f3f4f6"

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "05-linear-systems-and-column-space"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "05-线性方程组与列空间"
)


def save(fig, name: str) -> None:
    """Save the figure into both the EN and ZH folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)
    print(f"  saved: {name}")


def arrow(ax, p0, p1, color, lw=2.4, alpha=1.0, mutation=18):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation,
        color=color, lw=lw, alpha=alpha, zorder=5,
    ))


# ----------------------------------------------------------------------
# Figure 1: Ax = b geometrically -- b in the column space
# ----------------------------------------------------------------------
def fig_ax_equals_b():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: b is reachable (in column space)
    ax = axes[0]
    a1 = np.array([2.5, 1.0])
    a2 = np.array([1.0, 2.0])
    x_coef = np.array([1.2, 1.6])
    b = x_coef[0] * a1 + x_coef[1] * a2

    # span (column space) -- a 2D plane shown as the whole grid background
    ax.add_patch(Rectangle((-5, -5), 10, 10, color=BLUE, alpha=0.06, zorder=0))
    ax.text(-4.6, 4.5, "Column space  C(A) = span(a$_1$, a$_2$) = $\\mathbb{R}^2$",
            color=BLUE, fontsize=10.5, fontweight="bold")

    # parallelogram showing the linear combination
    poly = Polygon([
        (0, 0), (x_coef[0]*a1[0], x_coef[0]*a1[1]),
        (b[0], b[1]), (x_coef[1]*a2[0], x_coef[1]*a2[1])
    ], closed=True, facecolor=AMBER, alpha=0.18, edgecolor=AMBER, lw=1.2)
    ax.add_patch(poly)

    arrow(ax, (0, 0), tuple(a1), BLUE)
    arrow(ax, (0, 0), tuple(a2), VIOLET)
    arrow(ax, (0, 0), tuple(b), EMERALD, lw=3.0)
    # scaled column vectors
    arrow(ax, (0, 0), tuple(x_coef[0]*a1), BLUE, lw=1.4, alpha=0.45)
    arrow(ax, tuple(x_coef[0]*a1), tuple(b), VIOLET, lw=1.4, alpha=0.45)

    ax.text(a1[0]+0.15, a1[1]-0.25, "a$_1$", color=BLUE, fontsize=12, fontweight="bold")
    ax.text(a2[0]-0.55, a2[1]+0.15, "a$_2$", color=VIOLET, fontsize=12, fontweight="bold")
    ax.text(b[0]+0.1, b[1]+0.15, "b = x$_1$a$_1$+x$_2$a$_2$", color=EMERALD,
            fontsize=11, fontweight="bold")

    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.axhline(0, color="#9ca3af", lw=0.7)
    ax.axvline(0, color="#9ca3af", lw=0.7)
    ax.set_title("Solvable:  b ∈ C(A)  →  Ax = b has a solution", color=DARK)

    # --- Right: b is NOT in column space (rank-deficient case)
    ax = axes[1]
    a1 = np.array([2.0, 1.0])
    a2 = np.array([4.0, 2.0])  # parallel to a1 -- column space collapses to a line
    b_off = np.array([-1.0, 2.5])

    # column space is a line through origin
    t = np.linspace(-3, 3, 100)
    line = np.outer(t, a1)
    ax.plot(line[:, 0], line[:, 1], color=BLUE, lw=3.5, alpha=0.35, zorder=1,
            label="C(A)  (only a line)")

    arrow(ax, (0, 0), tuple(a1), BLUE)
    arrow(ax, (0, 0), tuple(a2), VIOLET)
    arrow(ax, (0, 0), tuple(b_off), AMBER, lw=3.0)

    # nearest point on line (projection)
    proj_scale = np.dot(b_off, a1) / np.dot(a1, a1)
    proj = proj_scale * a1
    ax.plot([b_off[0], proj[0]], [b_off[1], proj[1]],
            color=GRAY, lw=1.4, ls="--")
    ax.scatter(*proj, color=GRAY, s=50, zorder=6)

    ax.text(a1[0]+0.15, a1[1]-0.35, "a$_1$", color=BLUE, fontsize=12, fontweight="bold")
    ax.text(a2[0]+0.15, a2[1]+0.05, "a$_2$ = 2a$_1$", color=VIOLET, fontsize=11, fontweight="bold")
    ax.text(b_off[0]-0.4, b_off[1]+0.3, "b ∉ C(A)", color=AMBER, fontsize=12, fontweight="bold")
    ax.text(proj[0]+0.15, proj[1]-0.5, "closest reachable point\n(least-squares)",
            color=GRAY, fontsize=9)

    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.axhline(0, color="#9ca3af", lw=0.7)
    ax.axvline(0, color="#9ca3af", lw=0.7)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_title("No solution:  b ∉ C(A)  →  best we can do is project", color=DARK)

    fig.suptitle("Ax = b geometrically: solvability lives in the column space",
                 fontsize=14, fontweight="bold", color=DARK, y=1.02)
    save(fig, "fig1_ax_equals_b.png")


# ----------------------------------------------------------------------
# Figure 2: Column space as the span of the columns (3D)
# ----------------------------------------------------------------------
def fig_column_space():
    fig = plt.figure(figsize=(13.5, 5.6))

    # Left: rank 1 -- column space is a line
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    a1 = np.array([1.0, 0.5, 0.8])
    t = np.linspace(-2.5, 2.5, 60)
    line = np.outer(t, a1)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color=BLUE, lw=4, alpha=0.55)
    ax.quiver(0, 0, 0, *a1, color=BLUE, lw=2.5, arrow_length_ratio=0.18)
    ax.text(a1[0]*1.2, a1[1]*1.2, a1[2]*1.2, "a$_1$", color=BLUE, fontsize=12,
            fontweight="bold")
    _style_3d(ax, "rank 1 → line")

    # Middle: rank 2 -- column space is a plane
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    a1 = np.array([1.5, 0.0, 0.5])
    a2 = np.array([0.0, 1.5, 0.7])
    s = np.linspace(-1.5, 1.5, 14)
    S, T = np.meshgrid(s, s)
    Xs = S * a1[0] + T * a2[0]
    Ys = S * a1[1] + T * a2[1]
    Zs = S * a1[2] + T * a2[2]
    ax.plot_surface(Xs, Ys, Zs, color=VIOLET, alpha=0.22, edgecolor=VIOLET,
                    linewidth=0.3)
    ax.quiver(0, 0, 0, *a1, color=BLUE, lw=2.5, arrow_length_ratio=0.18)
    ax.quiver(0, 0, 0, *a2, color=EMERALD, lw=2.5, arrow_length_ratio=0.18)
    ax.text(a1[0]*1.15, a1[1], a1[2]*1.2, "a$_1$", color=BLUE, fontsize=12, fontweight="bold")
    ax.text(a2[0], a2[1]*1.15, a2[2]*1.2, "a$_2$", color=EMERALD, fontsize=12, fontweight="bold")
    _style_3d(ax, "rank 2 → plane")

    # Right: rank 3 -- column space is all of R^3
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    a1 = np.array([1.5, 0.0, 0.0])
    a2 = np.array([0.0, 1.5, 0.0])
    a3 = np.array([0.0, 0.0, 1.5])
    # draw a translucent cube to suggest "all of R^3"
    _draw_cube(ax, size=2.0, color=AMBER, alpha=0.10)
    ax.quiver(0, 0, 0, *a1, color=BLUE, lw=2.5, arrow_length_ratio=0.18)
    ax.quiver(0, 0, 0, *a2, color=EMERALD, lw=2.5, arrow_length_ratio=0.18)
    ax.quiver(0, 0, 0, *a3, color=VIOLET, lw=2.5, arrow_length_ratio=0.18)
    ax.text(1.7, 0, 0, "a$_1$", color=BLUE, fontsize=12, fontweight="bold")
    ax.text(0, 1.7, 0, "a$_2$", color=EMERALD, fontsize=12, fontweight="bold")
    ax.text(0, 0, 1.7, "a$_3$", color=VIOLET, fontsize=12, fontweight="bold")
    _style_3d(ax, "rank 3 → all of $\\mathbb{R}^3$")

    fig.suptitle("Column space = span of the columns of A",
                 fontsize=14, fontweight="bold", color=DARK, y=1.02)
    save(fig, "fig2_column_space.png")


def _style_3d(ax, title: str):
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_zlim(-2.2, 2.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, color=DARK, fontsize=12)
    ax.view_init(elev=22, azim=-55)
    # subtle origin marker
    ax.scatter([0], [0], [0], color=DARK, s=18)


def _draw_cube(ax, size, color, alpha):
    s = size / 2.0
    pts = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s],  [s, -s, s],  [s, s, s],  [-s, s, s],
    ])
    faces = [
        [pts[0], pts[1], pts[2], pts[3]],
        [pts[4], pts[5], pts[6], pts[7]],
        [pts[0], pts[1], pts[5], pts[4]],
        [pts[2], pts[3], pts[7], pts[6]],
        [pts[1], pts[2], pts[6], pts[5]],
        [pts[0], pts[3], pts[7], pts[4]],
    ]
    poly = Poly3DCollection(faces, facecolor=color, alpha=alpha,
                            edgecolor=color, linewidths=0.4)
    ax.add_collection3d(poly)


# ----------------------------------------------------------------------
# Figure 3: Null space -- the directions crushed to zero
# ----------------------------------------------------------------------
def fig_null_space():
    fig = plt.figure(figsize=(13.5, 5.6))

    # Left: 2D matrix that crushes a line to the origin
    ax = fig.add_subplot(1, 2, 1)
    # A = [[1,2],[2,4]]  -> null space spanned by (-2, 1)
    null_vec = np.array([-2.0, 1.0])
    t = np.linspace(-2.2, 2.2, 100)
    null_line = np.outer(t, null_vec)
    ax.plot(null_line[:, 0], null_line[:, 1], color=AMBER, lw=4, alpha=0.55,
            label="N(A)  (crushed to 0)")

    # Show several null-space vectors mapping to the origin
    for tt in [-1.6, -0.8, 0.8, 1.6]:
        v = tt * null_vec
        arrow(ax, (0, 0), tuple(v), AMBER, lw=1.2, alpha=0.85, mutation=14)

    # Image direction (column space) -- a line at angle of (1,2)
    col_dir = np.array([1.0, 2.0])
    col_line = np.outer(np.linspace(-2.5, 2.5, 100), col_dir)
    ax.plot(col_line[:, 0], col_line[:, 1], color=BLUE, lw=3, alpha=0.45,
            label="C(A)  (image of A)")

    # show how a non-null vector maps onto C(A) via dashed arrow
    v = np.array([1.5, 1.0])
    Av = (v[0] + 2*v[1]) * np.array([1, 2]) / 5.0  # symbolic projection-style
    arrow(ax, (0, 0), tuple(v), VIOLET, lw=2.0)
    ax.annotate("", xy=Av*2.4, xytext=v,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2, ls="--"))
    arrow(ax, (0, 0), tuple(Av*2.4), EMERALD, lw=2.0)

    ax.text(v[0]+0.1, v[1]+0.1, "x", color=VIOLET, fontsize=12, fontweight="bold")
    ax.text(Av[0]*2.4+0.15, Av[1]*2.4-0.25, "Ax", color=EMERALD,
            fontsize=12, fontweight="bold")
    ax.text(-3.6, 2.2, "A = [[1,2],[2,4]]", color=DARK, fontsize=10,
            family="monospace",
            bbox=dict(facecolor=LIGHT, edgecolor=GRAY, boxstyle="round,pad=0.3"))

    ax.set_xlim(-4, 4); ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axhline(0, color="#9ca3af", lw=0.7); ax.axvline(0, color="#9ca3af", lw=0.7)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_title("Null space: vectors that A maps to 0", color=DARK)

    # Right: 3D -- a plane projects to its 2D shadow; the perpendicular axis is null
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    # plane z = 0 is the column space (image), z-axis is the null space
    s = np.linspace(-1.6, 1.6, 12)
    S, T = np.meshgrid(s, s)
    ax.plot_surface(S, T, np.zeros_like(S), color=BLUE, alpha=0.18,
                    edgecolor=BLUE, linewidth=0.3)
    ax.text(1.7, 0, 0.05, "C(A) = xy-plane", color=BLUE, fontsize=10, fontweight="bold")

    # null direction (z axis) shown as crushed arrows
    for h in [-1.2, -0.6, 0.6, 1.2]:
        ax.quiver(0, 0, 0, 0, 0, h, color=AMBER, lw=1.5,
                  arrow_length_ratio=0.18, alpha=0.85)
    ax.text(0.05, 0.05, 1.6, "N(A) = z-axis\n(crushed to 0)",
            color=AMBER, fontsize=10, fontweight="bold")

    # a generic vector and its image (drop z)
    v = np.array([1.3, 0.9, 1.2])
    ax.quiver(0, 0, 0, *v, color=VIOLET, lw=2.0, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, v[0], v[1], 0, color=EMERALD, lw=2.0,
              arrow_length_ratio=0.15)
    ax.plot([v[0], v[0]], [v[1], v[1]], [0, v[2]], color=GRAY, lw=1.0, ls="--")
    ax.text(v[0]+0.05, v[1]+0.05, v[2]+0.05, "x", color=VIOLET,
            fontsize=12, fontweight="bold")
    ax.text(v[0]+0.05, v[1]+0.05, -0.25, "Ax", color=EMERALD,
            fontsize=12, fontweight="bold")

    _style_3d(ax, "Projection P: $\\mathbb{R}^3 \\to \\mathbb{R}^2$")

    fig.suptitle("Null space N(A) = directions A annihilates",
                 fontsize=14, fontweight="bold", color=DARK, y=1.02)
    save(fig, "fig3_null_space.png")


# ----------------------------------------------------------------------
# Figure 4: Row reduction / Gaussian elimination steps
# ----------------------------------------------------------------------
def fig_gaussian_elimination():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

    matrices = [
        np.array([[1, 2, 1, 2],
                  [3, 8, 1, 12],
                  [0, 4, 1, 2]], dtype=float),
        np.array([[1, 2, 1, 2],
                  [0, 2, -2, 6],
                  [0, 4, 1, 2]], dtype=float),
        np.array([[1, 2, 1, 2],
                  [0, 2, -2, 6],
                  [0, 0, 5, -10]], dtype=float),
        np.array([[1, 0, 0, -11/2],
                  [0, 1, 0, 7/2],
                  [0, 0, 1, -2]], dtype=float),
    ]
    titles = [
        "Step 0\nAugmented [A | b]",
        "Step 1\nR2 ← R2 − 3·R1",
        "Step 2\nR3 ← R3 − 2·R2",
        "Step 3\nReduced row echelon",
    ]
    # which (row,col) cells are pivots in each step (for highlighting)
    pivot_cells = [
        [(0, 0)],
        [(0, 0), (1, 1)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
    ]

    for ax, M, title, pivots in zip(axes, matrices, titles, pivot_cells):
        _draw_aug_matrix(ax, M, pivots, title)

    fig.suptitle("Gaussian elimination: turning a system into a triangular ladder",
                 fontsize=14, fontweight="bold", color=DARK, y=1.05)
    save(fig, "fig4_gaussian_elimination.png")


def _draw_aug_matrix(ax, M, pivot_cells, title):
    rows, cols = M.shape
    cell_w = 1.0
    cell_h = 0.9
    ax.set_xlim(-0.4, cols * cell_w + 0.4)
    ax.set_ylim(-0.4, rows * cell_h + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=DARK, fontsize=11)

    # outer brackets
    bracket_y = (-0.05, rows * cell_h + 0.05)
    ax.plot([-0.18, -0.18, 0.0], [bracket_y[0], bracket_y[1], bracket_y[1]],
            color=DARK, lw=1.4)
    ax.plot([-0.18, 0.0], [bracket_y[0], bracket_y[0]], color=DARK, lw=1.4)
    rb = cols * cell_w
    ax.plot([rb + 0.18, rb + 0.18, rb], [bracket_y[0], bracket_y[1], bracket_y[1]],
            color=DARK, lw=1.4)
    ax.plot([rb + 0.18, rb], [bracket_y[0], bracket_y[0]], color=DARK, lw=1.4)

    # vertical separator (augmented bar) between A and b
    ax.plot([(cols - 1) * cell_w, (cols - 1) * cell_w],
            [0, rows * cell_h], color=GRAY, lw=1.2, ls="--")

    for i in range(rows):
        for j in range(cols):
            x = j * cell_w
            y = (rows - 1 - i) * cell_h
            face = "white"
            edge = "#e5e7eb"
            color = DARK
            weight = "normal"
            if (i, j) in pivot_cells:
                face = AMBER
                edge = AMBER
                color = "white"
                weight = "bold"
            elif j == cols - 1:
                face = LIGHT
            ax.add_patch(Rectangle((x + 0.04, y + 0.05), cell_w - 0.08, cell_h - 0.10,
                                   facecolor=face, edgecolor=edge, linewidth=1.0))
            v = M[i, j]
            txt = f"{v:.2f}".rstrip("0").rstrip(".") if v != 0 else "0"
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", color=color, fontsize=11,
                    fontweight=weight)


# ----------------------------------------------------------------------
# Figure 5: Rank-Nullity theorem -- the "input pie" splits into two parts
# ----------------------------------------------------------------------
def fig_rank_nullity():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- Left: a stacked-bar showing rank + nullity = n for several matrices
    ax = axes[0]
    cases = [
        ("3×3 full rank",      3, 3),
        ("3×3, rank 2",        3, 2),
        ("3×3, rank 1",        3, 1),
        ("3×5, rank 2",        5, 2),
        ("4×4, zero matrix",   4, 0),
    ]
    labels = [c[0] for c in cases]
    ranks = [c[2] for c in cases]
    nulls = [c[1] - c[2] for c in cases]
    ns = [c[1] for c in cases]
    y = np.arange(len(cases))

    ax.barh(y, ranks, color=BLUE, edgecolor="white", height=0.55, label="rank(A) = dim C(A)")
    ax.barh(y, nulls, left=ranks, color=AMBER, edgecolor="white", height=0.55,
            label="nullity(A) = dim N(A)")
    for yi, r, k, n in zip(y, ranks, nulls, ns):
        if r > 0:
            ax.text(r/2, yi, f"{r}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=11)
        if k > 0:
            ax.text(r + k/2, yi, f"{k}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=11)
        ax.text(n + 0.18, yi, f"= {n}", va="center", color=DARK,
                fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("dimension (= n, number of columns)", fontsize=10)
    ax.set_xlim(0, max(ns) + 1.0)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_title("rank(A) + nullity(A) = n", color=DARK)

    # --- Right: the "input pie" diagram
    ax = axes[1]
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")

    # A 3x5 example: rank 2, nullity 3
    rank_frac = 2 / 5
    null_frac = 3 / 5

    # pie for input space R^n
    theta1 = 0
    theta2 = 360 * rank_frac
    theta3 = 360
    wedge1 = mpatches.Wedge((0, 0), 1.0, theta1, theta2,
                            facecolor=BLUE, edgecolor="white", lw=2)
    wedge2 = mpatches.Wedge((0, 0), 1.0, theta2, theta3,
                            facecolor=AMBER, edgecolor="white", lw=2)
    ax.add_patch(wedge1); ax.add_patch(wedge2)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=DARK, lw=1.2))

    ax.text(0.55, 0.32, "row space\n(rank = 2)", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
    ax.text(-0.45, -0.25, "null space\n(nullity = 3)", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
    ax.text(0, 1.2, "Input space  $\\mathbb{R}^n$  (n = 5)",
            ha="center", color=DARK, fontsize=11, fontweight="bold")
    ax.text(0, -1.2, "Every column splits into\n'preserved' + 'crushed'",
            ha="center", color=GRAY, fontsize=9, style="italic")

    fig.suptitle("Rank–Nullity Theorem: every input dimension is either preserved or crushed",
                 fontsize=14, fontweight="bold", color=DARK, y=1.02)
    save(fig, "fig5_rank_nullity.png")


# ----------------------------------------------------------------------
# Figure 6: Underdetermined / Overdetermined / Unique solution
# ----------------------------------------------------------------------
def fig_solution_scenarios():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # ---- Unique: two non-parallel lines intersecting at one point
    ax = axes[0]
    x = np.linspace(-2, 5, 100)
    # x + 2y = 5  -> y = (5 - x)/2
    # 3x - y = 1  -> y = 3x - 1
    ax.plot(x, (5 - x) / 2, color=BLUE, lw=2.5, label="x + 2y = 5")
    ax.plot(x, 3 * x - 1, color=VIOLET, lw=2.5, label="3x − y = 1")
    ax.scatter([1], [2], color=EMERALD, s=110, zorder=5, edgecolor="white", lw=1.5)
    ax.annotate("unique\n(1, 2)", xy=(1, 2), xytext=(2.3, 3.3),
                color=EMERALD, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=EMERALD, lw=1.4))
    ax.set_xlim(-2, 5); ax.set_ylim(-3, 5)
    ax.set_aspect("equal")
    ax.axhline(0, color="#9ca3af", lw=0.7); ax.axvline(0, color="#9ca3af", lw=0.7)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Unique solution\n(2 equations, 2 unknowns, full rank)", color=DARK)

    # ---- Underdetermined: one equation, infinitely many solutions
    ax = axes[1]
    ax.plot(x, (4 - x) / 2, color=BLUE, lw=3, label="x + 2y = 4")
    # show several solution points along the line
    pts = np.array([(0, 2), (2, 1), (4, 0), (-2, 3)])
    ax.scatter(pts[:, 0], pts[:, 1], color=EMERALD, s=70, zorder=5,
               edgecolor="white", lw=1.5)
    # particular + null vector
    arrow(ax, (2, 1), (0, 2), AMBER, lw=2)
    ax.text(0.4, 2.4, "x$_p$ + t·n,  n ∈ N(A)",
            color=AMBER, fontsize=10, fontweight="bold")
    ax.set_xlim(-3, 5); ax.set_ylim(-2, 5)
    ax.set_aspect("equal")
    ax.axhline(0, color="#9ca3af", lw=0.7); ax.axvline(0, color="#9ca3af", lw=0.7)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Infinitely many solutions\n(underdetermined: 1 eq, 2 unknowns)", color=DARK)

    # ---- Overdetermined: three lines with no common point -> no exact solution
    ax = axes[2]
    ax.plot(x, (5 - x) / 2, color=BLUE, lw=2.2, label="x + 2y = 5")
    ax.plot(x, 3 * x - 1, color=VIOLET, lw=2.2, label="3x − y = 1")
    ax.plot(x, x - 1.6, color=EMERALD, lw=2.2, label="x − y = 1.6")
    # least squares "best" point (computed numerically)
    A = np.array([[1, 2], [3, -1], [1, -1]])
    b = np.array([5, 1, 1.6])
    x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    ax.scatter(x_hat[0], x_hat[1], color=AMBER, s=120, zorder=5,
               edgecolor="white", lw=1.5)
    ax.annotate("least-squares\n best fit", xy=(x_hat[0], x_hat[1]),
                xytext=(3.0, 3.5), color=AMBER, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.set_xlim(-2, 5); ax.set_ylim(-3, 5)
    ax.set_aspect("equal")
    ax.axhline(0, color="#9ca3af", lw=0.7); ax.axvline(0, color="#9ca3af", lw=0.7)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("No exact solution\n(overdetermined: 3 eqs, 2 unknowns)", color=DARK)

    fig.suptitle("Three faces of Ax = b: unique, infinitely many, none",
                 fontsize=14, fontweight="bold", color=DARK, y=1.03)
    save(fig, "fig6_solution_scenarios.png")


# ----------------------------------------------------------------------
# Figure 7: LU decomposition geometric meaning
# ----------------------------------------------------------------------
def fig_lu_decomposition():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))

    # original unit square + a sample vector
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    v = np.array([0.7, 0.5])

    # LU example: A = [[2, 1], [4, 3]] = L · U
    # L = [[1, 0], [2, 1]],  U = [[2, 1], [0, 1]]
    L = np.array([[1.0, 0.0], [2.0, 1.0]])
    U = np.array([[2.0, 1.0], [0.0, 1.0]])
    A = L @ U  # = [[2, 1], [4, 3]]

    panels = [
        ("Original\nunit square", np.eye(2), BLUE),
        ("Step 1: U  (shear + scale,\nupper triangular)", U, VIOLET),
        ("Step 2: L  (shear,\nlower triangular)", L, EMERALD),
        ("A = L·U  (final)", A, AMBER),
    ]

    # The transforms are cumulative: identity, then U on the unit square,
    # then L applied to the result -> A on the original.
    cumulative = [np.eye(2), U, L @ U, A]

    for ax, (title, _, color), M in zip(axes, panels, cumulative):
        # background grid showing the transformed lattice
        for k in range(-2, 3):
            line_h = M @ np.array([[-2, 2], [k, k]])
            line_v = M @ np.array([[k, k], [-2, 2]])
            ax.plot(line_h[0], line_h[1], color=color, alpha=0.18, lw=0.9)
            ax.plot(line_v[0], line_v[1], color=color, alpha=0.18, lw=0.9)

        # transformed square
        sq_t = M @ sq
        ax.fill(sq_t[0], sq_t[1], color=color, alpha=0.22, edgecolor=color, lw=2)
        # transformed sample vector
        v_t = M @ v
        arrow(ax, (0, 0), tuple(v_t), DARK, lw=2.0)

        ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5)
        ax.set_aspect("equal")
        ax.axhline(0, color="#9ca3af", lw=0.6); ax.axvline(0, color="#9ca3af", lw=0.6)
        ax.set_title(title, color=DARK, fontsize=11)

    fig.suptitle("LU decomposition: A = L·U is two simple shears in sequence",
                 fontsize=14, fontweight="bold", color=DARK, y=1.04)

    # caption with matrices (avoid LaTeX pmatrix which mathtext doesn't support)
    fig.text(0.5, -0.02,
             "A = [[2, 1], [4, 3]]  =  L · U   "
             "where  L = [[1, 0], [2, 1]],  U = [[2, 1], [0, 1]]   "
             "(Gaussian elimination = factorising A into L and U)",
             ha="center", fontsize=11, color=DARK, family="monospace")
    save(fig, "fig7_lu_decomposition.png")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Generating Chapter 05 figures...")
    fig_ax_equals_b()
    fig_column_space()
    fig_null_space()
    fig_gaussian_elimination()
    fig_rank_nullity()
    fig_solution_scenarios()
    fig_lu_decomposition()
    print(f"\nDone. Saved to:\n  EN: {EN_DIR}\n  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
