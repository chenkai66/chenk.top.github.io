"""
Figure generation script for Linear Algebra Chapter 04: The Secrets of Determinants.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in 3Blue1Brown style.

Figures:
    fig1_determinant_area      Determinant as area scaling factor: unit square
                               transforms into a parallelogram whose shaded
                               area equals |det|.
    fig2_three_determinants    Three transformations side by side with
                               det = 1, det = 2, det = 0.5 - showing how
                               different scalings look.
    fig3_orientation           Negative determinant flips orientation: a CCW
                               labelled basis becomes CW after a reflection.
    fig4_collapse              det = 0 collapses the plane onto a line; both
                               basis vectors land on the same span.
    fig5_volume_3d             3D analogue: unit cube becomes a parallelepiped;
                               its (signed) volume equals det.
    fig6_product_rule          det(AB) = det(A) det(B): two scalings compose
                               into the product, visualised as a chain of
                               parallelograms.
    fig7_cofactor_expansion    Cofactor expansion of a 3x3 matrix: highlight
                               the row, the alternating sign pattern, and the
                               three 2x2 minors that make up the sum.

Usage:
    python3 scripts/figures/linear-algebra/04-the-secrets-of-determinants.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "04-the-secrets-of-determinants"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "04-行列式的秘密"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH EN and ZH asset folders as <name>.png."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        path = d / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _setup_axes(ax, lim=(-3.2, 3.2), aspect=True):
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    if aspect:
        ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, lw=0.6, zorder=0)
    ax.axvline(0, color=C_GRAY, lw=0.6, zorder=0)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_color(C_GRAY)
        spine.set_linewidth(0.6)


def _draw_basis(ax, e1, e2, label_e1=r"$\vec{e}_1$", label_e2=r"$\vec{e}_2$",
                color1=C_BLUE, color2=C_PURPLE, lw=2.4):
    ax.add_patch(FancyArrowPatch((0, 0), tuple(e1),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=color1, lw=lw, zorder=4))
    ax.add_patch(FancyArrowPatch((0, 0), tuple(e2),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=color2, lw=lw, zorder=4))
    # Labels positioned slightly off the arrow tip.
    off1 = 0.18 * np.array([np.sign(e1[0] or 1), np.sign(e1[1] or 1) - 0.3])
    off2 = 0.18 * np.array([np.sign(e2[0] or 1) - 0.3, np.sign(e2[1] or 1)])
    ax.text(e1[0] + off1[0], e1[1] + off1[1], label_e1,
            color=color1, fontsize=11, fontweight="bold")
    ax.text(e2[0] + off2[0], e2[1] + off2[1], label_e2,
            color=color2, fontsize=11, fontweight="bold")


def _parallelogram(e1, e2):
    """Return four corners of the parallelogram spanned by e1, e2."""
    return np.array([(0, 0), e1, np.array(e1) + np.array(e2), e2])


# ---------------------------------------------------------------------------
# Fig 1: Determinant as area scaling factor
# ---------------------------------------------------------------------------
def fig1_determinant_area():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: unit square
    ax = axes[0]
    _setup_axes(ax, lim=(-1.0, 4.0))
    square = _parallelogram((1, 0), (0, 1))
    ax.add_patch(Polygon(square, closed=True,
                         facecolor=C_BLUE, alpha=0.22,
                         edgecolor=C_BLUE, lw=2))
    _draw_basis(ax, (1, 0), (0, 1))
    ax.text(0.5, 0.5, "area = 1",
            ha="center", va="center", fontsize=11, color=C_DARK,
            fontweight="bold")
    ax.set_title("Unit square (input)", fontsize=12, color=C_DARK,
                 fontweight="bold", pad=10)

    # Right: parallelogram after A = [[3,1],[0,2]]
    ax = axes[1]
    _setup_axes(ax, lim=(-1.0, 4.5))
    A = np.array([[3, 1], [0, 2]])
    v1 = A @ np.array([1, 0])
    v2 = A @ np.array([0, 1])
    para = _parallelogram(v1, v2)
    ax.add_patch(Polygon(para, closed=True,
                         facecolor=C_GREEN, alpha=0.25,
                         edgecolor=C_GREEN, lw=2))
    _draw_basis(ax, v1, v2,
                label_e1=r"$A\vec{e}_1=(3,0)$",
                label_e2=r"$A\vec{e}_2=(1,2)$",
                color1=C_BLUE, color2=C_PURPLE)
    det = np.linalg.det(A)
    cx, cy = para.mean(axis=0)
    ax.text(cx, cy, f"area = |det A| = {abs(det):.0f}",
            ha="center", va="center", fontsize=11, color=C_DARK,
            fontweight="bold")
    ax.set_title(r"After $A=[[\,3,\,1\,],[\,0,\,2\,]]$",
                 fontsize=12, color=C_DARK, fontweight="bold", pad=10)

    fig.suptitle("Determinant = factor by which area is scaled",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_determinant_area")


# ---------------------------------------------------------------------------
# Fig 2: Three different determinants (1, 2, 0.5)
# ---------------------------------------------------------------------------
def fig2_three_determinants():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    cases = [
        # name, A, color, lim
        ("Shear: det = 1\n(area unchanged)",
         np.array([[1.0, 0.6], [0.0, 1.0]]), C_BLUE, (-0.6, 2.4)),
        ("Stretch: det = 2\n(area doubles)",
         np.array([[2.0, 0.0], [0.0, 1.0]]), C_GREEN, (-0.6, 2.8)),
        ("Compress: det = 0.5\n(area halved)",
         np.array([[1.0, 0.0], [0.0, 0.5]]), C_AMBER, (-0.6, 2.4)),
    ]

    for ax, (title, A, color, lim) in zip(axes, cases):
        _setup_axes(ax, lim=lim)
        # Faint reference unit square
        ref = _parallelogram((1, 0), (0, 1))
        ax.add_patch(Polygon(ref, closed=True,
                             facecolor="none",
                             edgecolor=C_GRAY, lw=1.2,
                             linestyle="--", zorder=2))
        v1 = A @ np.array([1, 0])
        v2 = A @ np.array([0, 1])
        para = _parallelogram(v1, v2)
        ax.add_patch(Polygon(para, closed=True,
                             facecolor=color, alpha=0.28,
                             edgecolor=color, lw=2, zorder=3))
        _draw_basis(ax, v1, v2,
                    label_e1=r"$A\vec{e}_1$", label_e2=r"$A\vec{e}_2$")
        det = np.linalg.det(A)
        cx, cy = para.mean(axis=0)
        ax.text(cx, cy, f"area = {abs(det):.2f}",
                ha="center", va="center", fontsize=10,
                color=C_DARK, fontweight="bold")
        ax.set_title(title, fontsize=11, color=C_DARK,
                     fontweight="bold", pad=8)

    fig.suptitle("Same shape, different determinants",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_three_determinants")


# ---------------------------------------------------------------------------
# Fig 3: Orientation flip (negative determinant)
# ---------------------------------------------------------------------------
def fig3_orientation():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Helper: draw curved arrow for orientation arc.
    def orient_arc(ax, ccw=True, color=C_GREEN):
        from matplotlib.patches import Arc
        arc = Arc((0, 0), 0.9, 0.9, angle=0,
                  theta1=10 if ccw else 80, theta2=80 if ccw else 10,
                  color=color, lw=2.2, zorder=5)
        ax.add_patch(arc)
        # Arrowhead at end of arc
        if ccw:
            tip = (0.45 * np.cos(np.deg2rad(80)),
                   0.45 * np.sin(np.deg2rad(80)))
            tail = (0.45 * np.cos(np.deg2rad(70)),
                    0.45 * np.sin(np.deg2rad(70)))
        else:
            tip = (0.45 * np.cos(np.deg2rad(10)),
                   0.45 * np.sin(np.deg2rad(10)))
            tail = (0.45 * np.cos(np.deg2rad(20)),
                    0.45 * np.sin(np.deg2rad(20)))
        ax.add_patch(FancyArrowPatch(tail, tip, arrowstyle="-|>",
                                     mutation_scale=12, color=color, lw=2))
        label = "CCW (det > 0)" if ccw else "CW (det < 0)"
        ax.text(0.0, -0.55, label, ha="center", fontsize=10,
                color=color, fontweight="bold")

    # Left: original
    ax = axes[0]
    _setup_axes(ax, lim=(-2.2, 2.2))
    square = _parallelogram((1, 0), (0, 1))
    ax.add_patch(Polygon(square, closed=True,
                         facecolor=C_GREEN, alpha=0.22,
                         edgecolor=C_GREEN, lw=2))
    _draw_basis(ax, (1, 0), (0, 1))
    orient_arc(ax, ccw=True, color=C_GREEN)
    ax.set_title("Original: right-handed basis\n(det = +1)",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)

    # Right: reflection across y-axis A = [[-1,0],[0,1]]
    ax = axes[1]
    _setup_axes(ax, lim=(-2.2, 2.2))
    A = np.array([[-1, 0], [0, 1]])
    v1 = A @ np.array([1, 0])
    v2 = A @ np.array([0, 1])
    para = _parallelogram(v1, v2)
    ax.add_patch(Polygon(para, closed=True,
                         facecolor=C_AMBER, alpha=0.22,
                         edgecolor=C_AMBER, lw=2))
    _draw_basis(ax, v1, v2,
                label_e1=r"$A\vec{e}_1$", label_e2=r"$A\vec{e}_2$")

    # CW arc on the reflected side
    from matplotlib.patches import Arc

    arc = Arc((0, 0), 0.9, 0.9, angle=0, theta1=100, theta2=170,
              color=C_AMBER, lw=2.2, zorder=5)
    ax.add_patch(arc)
    tip = (0.45 * np.cos(np.deg2rad(170)),
           0.45 * np.sin(np.deg2rad(170)))
    tail = (0.45 * np.cos(np.deg2rad(160)),
            0.45 * np.sin(np.deg2rad(160)))
    ax.add_patch(FancyArrowPatch(tail, tip, arrowstyle="-|>",
                                 mutation_scale=12, color=C_AMBER, lw=2))
    ax.text(0.0, -0.55, "CW (det < 0)", ha="center", fontsize=10,
            color=C_AMBER, fontweight="bold")
    ax.set_title("After reflection across y-axis\n(det = -1, area kept,"
                 " orientation flipped)",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)

    fig.suptitle("Sign of det = orientation; "
                 "negative means a mirror flip",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_orientation")


# ---------------------------------------------------------------------------
# Fig 4: det = 0 -> collapse to a line
# ---------------------------------------------------------------------------
def fig4_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: input square (with a small grid of points to track)
    ax = axes[0]
    _setup_axes(ax, lim=(-2.5, 2.5))
    square = _parallelogram((1, 0), (0, 1))
    ax.add_patch(Polygon(square, closed=True,
                         facecolor=C_BLUE, alpha=0.18,
                         edgecolor=C_BLUE, lw=2))
    _draw_basis(ax, (1, 0), (0, 1))
    # Reference points
    pts = np.array([[0.25 * i, 0.25 * j]
                    for i in range(0, 5)
                    for j in range(0, 5)])
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=C_DARK, zorder=5)
    ax.set_title("Input: full 2D plane",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)

    # Right: A = [[1,2],[2,4]] -> det = 0
    ax = axes[1]
    _setup_axes(ax, lim=(-2.5, 5.5))
    A = np.array([[1, 2], [2, 4]])
    v1 = A @ np.array([1, 0])
    v2 = A @ np.array([0, 1])
    # The line they all collapse onto: direction (1,2).
    t = np.linspace(-3, 3, 200)
    line = np.outer(t, np.array([1, 2]))
    ax.plot(line[:, 0], line[:, 1], color=C_AMBER,
            lw=3, alpha=0.7, zorder=2,
            label=r"span = line through $(1,2)$")
    # Both basis images on the line
    _draw_basis(ax, v1, v2,
                label_e1=r"$A\vec{e}_1=(1,2)$",
                label_e2=r"$A\vec{e}_2=(2,4)$")
    pts_t = (A @ pts.T).T
    ax.scatter(pts_t[:, 0], pts_t[:, 1], s=18,
               color=C_DARK, zorder=5)
    ax.text(4.0, 0.6,
            "every point lands\non one line\n(area = 0)",
            ha="center", fontsize=10, color=C_AMBER,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec=C_AMBER, lw=1))
    ax.set_title(r"$A=[[1,2],[2,4]]$,  det = 0 (rank 1)",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle("det = 0: the plane gets crushed onto a lower-dimensional"
                 " subspace",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_collapse")


# ---------------------------------------------------------------------------
# Fig 5: 3D - determinant as volume scaling
# ---------------------------------------------------------------------------
def fig5_volume_3d():
    fig = plt.figure(figsize=(13, 6.5))

    def cube_faces(v1, v2, v3):
        """Return list of 6 quadrilateral faces of the parallelepiped."""
        o = np.zeros(3)
        p1, p2, p3 = np.array(v1), np.array(v2), np.array(v3)
        return [
            [o, p1, p1 + p2, p2],            # bottom (xy)
            [p3, p1 + p3, p1 + p2 + p3, p2 + p3],  # top
            [o, p1, p1 + p3, p3],            # front
            [p2, p1 + p2, p1 + p2 + p3, p2 + p3],  # back
            [o, p2, p2 + p3, p3],            # left
            [p1, p1 + p2, p1 + p2 + p3, p1 + p3],  # right
        ]

    def draw_box(ax, v1, v2, v3, color, alpha=0.25, lw=1.4):
        faces = cube_faces(v1, v2, v3)
        coll = Poly3DCollection(faces, facecolors=color, alpha=alpha,
                                edgecolors=color, linewidths=lw)
        ax.add_collection3d(coll)
        # basis arrows
        for v, c in [(v1, C_BLUE), (v2, C_PURPLE), (v3, C_GREEN)]:
            ax.quiver(0, 0, 0, v[0], v[1], v[2],
                      color=c, lw=2.6, arrow_length_ratio=0.12)

    def style_3d(ax, lim=2.4, title=""):
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_zlim(0, lim)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_zlabel("z", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=13, color=C_DARK,
                     fontweight="bold", pad=10)
        ax.view_init(elev=22, azim=35)

    # Left: unit cube, vol = 1
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    draw_box(ax1, (1, 0, 0), (0, 1, 0), (0, 0, 1), C_BLUE)
    ax1.text(0.5, 0.5, 0.55, "vol = 1", color=C_DARK,
             fontsize=12, fontweight="bold", ha="center")
    style_3d(ax1, lim=2.6, title="Unit cube (vol = 1)")

    # Right: parallelepiped from A
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    A = np.array([[1.5, 0.4, 0.2],
                  [0.0, 1.2, 0.5],
                  [0.3, 0.0, 1.4]])
    v1 = A @ np.array([1, 0, 0])
    v2 = A @ np.array([0, 1, 0])
    v3 = A @ np.array([0, 0, 1])
    draw_box(ax2, v1, v2, v3, C_GREEN, alpha=0.30)
    det = np.linalg.det(A)
    centroid = (v1 + v2 + v3) / 2
    ax2.text(centroid[0], centroid[1], centroid[2] + 0.15,
             f"vol = |det A| = {abs(det):.2f}",
             color=C_DARK, fontsize=12, fontweight="bold", ha="center")
    style_3d(ax2, lim=2.6,
             title=f"Parallelepiped after A  (vol = {abs(det):.2f})")

    fig.suptitle("In 3D: det A = signed volume scaling factor",
                 fontsize=15, color=C_DARK, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.05,
                        wspace=0.10)
    save(fig, "fig5_volume_3d")


# ---------------------------------------------------------------------------
# Fig 6: det(AB) = det(A) det(B)
# ---------------------------------------------------------------------------
def fig6_product_rule():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    A = np.array([[1.5, 0.0], [0.0, 2.0]])  # det = 3
    B = np.array([[1.0, 0.5], [0.0, 1.5]])  # det = 1.5
    AB = A @ B                                # det = 4.5

    cases = [
        ("Step 0: unit square\narea = 1",
         np.eye(2), C_BLUE, 1.0),
        (r"Step 1: apply $B$" + f"\narea = |det B| = {abs(np.linalg.det(B)):.2f}",
         B, C_PURPLE, np.linalg.det(B)),
        (r"Step 2: apply $A$ on top" + f"\narea = |det(AB)| = {abs(np.linalg.det(AB)):.2f}",
         AB, C_GREEN, np.linalg.det(AB)),
    ]

    for ax, (title, M, color, area) in zip(axes, cases):
        _setup_axes(ax, lim=(-0.6, 3.6))
        # faint reference square
        ref = _parallelogram((1, 0), (0, 1))
        ax.add_patch(Polygon(ref, closed=True,
                             facecolor="none", edgecolor=C_GRAY,
                             lw=1.0, linestyle="--", zorder=2))
        v1 = M @ np.array([1, 0])
        v2 = M @ np.array([0, 1])
        para = _parallelogram(v1, v2)
        ax.add_patch(Polygon(para, closed=True,
                             facecolor=color, alpha=0.27,
                             edgecolor=color, lw=2, zorder=3))
        _draw_basis(ax, v1, v2)
        ax.set_title(title, fontsize=11, color=C_DARK,
                     fontweight="bold", pad=8)

    # Suptitle showing the algebra
    fig.suptitle(r"$\det(AB) = \det(A)\cdot\det(B)$:  "
                 r"$3.00 \times 1.50 = 4.50$",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_product_rule")


# ---------------------------------------------------------------------------
# Fig 7: Cofactor expansion of a 3x3
# ---------------------------------------------------------------------------
def fig7_cofactor_expansion():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Master matrix on the left
    M = np.array([[2, 1, 3],
                  [0, 4, 1],
                  [5, 2, 1]])

    def draw_matrix(x, y, mat, highlight_row=None, highlight_col=None,
                    scale=0.7, label_top=None):
        """Draw a small matrix grid with optional row/col highlighting.

        (x, y) is the top-left corner of the bracket.
        """
        n_rows, n_cols = mat.shape
        cell = scale
        # bracket
        ax.plot([x - 0.05, x - 0.05, x + 0.1],
                [y, y - n_rows * cell, y - n_rows * cell],
                color=C_DARK, lw=1.5)
        ax.plot([x + n_cols * cell + 0.05, x + n_cols * cell + 0.05,
                 x + n_cols * cell - 0.1],
                [y, y - n_rows * cell, y - n_rows * cell],
                color=C_DARK, lw=1.5)
        ax.plot([x - 0.05, x + 0.1],
                [y, y], color=C_DARK, lw=1.5)
        ax.plot([x + n_cols * cell + 0.05, x + n_cols * cell - 0.1],
                [y, y], color=C_DARK, lw=1.5)

        for i in range(n_rows):
            for j in range(n_cols):
                cx = x + (j + 0.5) * cell
                cy = y - (i + 0.5) * cell
                bg = None
                if highlight_row is not None and i == highlight_row:
                    bg = "#fde68a"
                if highlight_col is not None and j == highlight_col:
                    bg = "#a7f3d0" if bg is None else "#fbbf24"
                if bg is not None:
                    ax.add_patch(Rectangle((x + j * cell, y - (i + 1) * cell),
                                           cell, cell, facecolor=bg,
                                           alpha=0.55, zorder=1))
                ax.text(cx, cy, str(mat[i, j]),
                        ha="center", va="center",
                        fontsize=11, color=C_DARK, zorder=4,
                        fontweight="bold")
        if label_top is not None:
            ax.text(x + n_cols * cell / 2, y + 0.25, label_top,
                    ha="center", fontsize=10, color=C_DARK,
                    fontweight="bold")

    # Step 1: full 3x3 with row 0 highlighted (top-left area)
    draw_matrix(0.5, 5.5, M, highlight_row=0, scale=0.6,
                label_top="expand along row 1")

    # Sign pattern (below the master matrix)
    ax.text(0.5 + 3 * 0.6 / 2, 3.4,
            r"row signs:  $+\;-\;+$",
            ha="center", fontsize=10, color=C_DARK)

    # Equation row
    eq_y = 1.9
    ax.text(0.0, eq_y, r"$\det = $", fontsize=14,
            color=C_DARK, va="center")

    # Three minor blocks
    minors = [
        # sign, coef, deleted_row, deleted_col, x_offset
        ("+", 2, 0, 0, 1.1),
        ("-", 1, 0, 1, 5.2),
        ("+", 3, 0, 2, 9.3),
    ]
    for sign, coef, dr, dc, x0 in minors:
        ax.text(x0, eq_y, sign, fontsize=18, color=C_DARK,
                va="center", fontweight="bold")
        ax.text(x0 + 0.45, eq_y, str(coef), fontsize=14,
                color=C_BLUE, va="center", fontweight="bold")
        ax.text(x0 + 0.85, eq_y, r"$\cdot$", fontsize=16,
                color=C_DARK, va="center")
        ax.text(x0 + 1.15, eq_y, "det", fontsize=11,
                color=C_DARK, va="center", style="italic")
        keep_rows = [r for r in range(3) if r != dr]
        keep_cols = [c for c in range(3) if c != dc]
        sub = M[np.ix_(keep_rows, keep_cols)]
        # draw the 2x2 to the right, top aligned ~0.6 above eq_y
        draw_matrix(x0 + 1.7, eq_y + 0.65, sub, scale=0.55)

    # Numeric line
    det_val = int(np.linalg.det(M).round())
    sub_dets = []
    for sign, coef, dr, dc, _ in minors:
        keep_rows = [r for r in range(3) if r != dr]
        keep_cols = [c for c in range(3) if c != dc]
        sub = M[np.ix_(keep_rows, keep_cols)]
        sub_dets.append((sign, coef, int(round(np.linalg.det(sub)))))

    parts = []
    for sign, coef, sd in sub_dets:
        parts.append(f"{sign} {coef}({sd})")
    expr = " ".join(parts)
    ax.text(6.5, 0.4,
            f"= {expr}  =  {det_val}",
            ha="center", fontsize=13, color=C_DARK, fontweight="bold")

    # Title
    fig.suptitle("Cofactor (Laplace) expansion of a 3x3 determinant",
                 fontsize=14, color=C_DARK, fontweight="bold", y=0.98)
    fig.tight_layout()
    save(fig, "fig7_cofactor_expansion")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    fig1_determinant_area()
    fig2_three_determinants()
    fig3_orientation()
    fig4_collapse()
    fig5_volume_3d()
    fig6_product_rule()
    fig7_cofactor_expansion()
    print("Generated 7 figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
