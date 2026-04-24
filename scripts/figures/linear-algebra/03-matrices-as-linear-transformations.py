"""
Figure generation script for Linear Algebra Chapter 03:
"Matrices as Linear Transformations".

Generates 7 figures used in BOTH the EN and ZH versions of the article.
The figures follow a unified 3Blue1Brown-inspired aesthetic: a clean
"before / after" grid pair so the reader can see exactly how each linear
transformation deforms the unit grid and the basis vectors.

Figures:
    fig1_unit_grid               The reference unit grid (i-hat, j-hat,
                                 unit square) with no transformation applied.
                                 Anchors the visual vocabulary used by the
                                 rest of the chapter.
    fig2_rotation                Unit grid before vs. after a 30-degree
                                 counter-clockwise rotation. Shows that
                                 lengths and angles are preserved.
    fig3_scaling                 Non-uniform scaling diag(2, 0.5). Shows
                                 area scaling by det = 1 (here actually
                                 det = 1, so area is preserved despite the
                                 anisotropic stretch) - we use diag(2, 1.5)
                                 instead so det = 3 != 1 makes the area
                                 change obvious.
    fig4_shear                   Horizontal shear with k = 1. j-hat tilts
                                 to (1, 1); i-hat unchanged. Useful for the
                                 "italic text" intuition.
    fig5_composition             Matrix multiplication = composition. Shows
                                 (1) original, (2) after A (rotate 45 deg),
                                 (3) after B*A (then scale x by 2).
                                 Makes the right-to-left reading rule visual.
    fig6_identity                The identity transformation. Before and
                                 after grids are identical - nothing moves.
                                 Reinforces what "no transformation" means.
    fig7_singular                A singular matrix [[1,2],[2,4]] collapses
                                 the 2D plane onto a line (the column space).
                                 Shows why det = 0 implies non-invertibility.

Usage:
    python3 scripts/figures/linear-algebra/03-matrices-as-linear-transformations.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay in sync across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"      # i-hat / primary
C_PURPLE = "#7c3aed"    # j-hat / secondary
C_GREEN = "#10b981"     # transformed-square fill / accent
C_AMBER = "#f59e0b"     # highlight / "after" emphasis
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "linear-algebra"
    / "03-matrices-as-linear-transformations"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "linear-algebra"
    / "03-矩阵作为线性变换"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, name: str) -> None:
    """Write the figure to both EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _setup_axis(ax: plt.Axes, lim: float = 3.2, title: str = "") -> None:
    """Prepare a square 2D axis with a clean grid."""
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=C_DARK, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axvline(0, color=C_DARK, linewidth=0.8, alpha=0.6, zorder=1)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xticks(np.arange(-int(lim), int(lim) + 1))
    ax.set_yticks(np.arange(-int(lim), int(lim) + 1))
    ax.tick_params(axis="both", labelsize=8, colors=C_GRAY)
    for spine in ax.spines.values():
        spine.set_color(C_GRAY)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=12, color=C_DARK, pad=8)


def _draw_grid(
    ax: plt.Axes,
    A: np.ndarray | None = None,
    extent: int = 4,
    color: str = C_GRAY,
    alpha: float = 0.45,
    linewidth: float = 0.8,
) -> None:
    """Draw the integer grid, optionally transformed by matrix A.

    The reference grid is a family of horizontal and vertical lines. Each
    line is sampled densely, then every point is mapped through A so that
    we can see how an originally-rectangular grid deforms.
    """
    if A is None:
        A = np.eye(2)
    # Vertical lines x = const
    for x in range(-extent, extent + 1):
        pts = np.array([[x, x], [-extent, extent]])
        out = A @ pts
        ax.plot(
            out[0], out[1],
            color=color, alpha=alpha, linewidth=linewidth, zorder=2,
        )
    # Horizontal lines y = const
    for y in range(-extent, extent + 1):
        pts = np.array([[-extent, extent], [y, y]])
        out = A @ pts
        ax.plot(
            out[0], out[1],
            color=color, alpha=alpha, linewidth=linewidth, zorder=2,
        )


def _draw_unit_square(
    ax: plt.Axes,
    A: np.ndarray | None = None,
    color: str = C_GREEN,
    alpha: float = 0.22,
    edge: bool = True,
) -> None:
    """Draw the (transformed) unit square spanned by i-hat and j-hat."""
    if A is None:
        A = np.eye(2)
    corners = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    out = A @ corners
    poly = Polygon(
        out.T,
        closed=True,
        facecolor=color,
        edgecolor=color if edge else "none",
        alpha=alpha,
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(poly)


def _draw_basis(
    ax: plt.Axes,
    A: np.ndarray | None = None,
    label_i: str = r"$\hat{\imath}$",
    label_j: str = r"$\hat{\jmath}$",
    show_labels: bool = True,
) -> None:
    """Draw the (transformed) basis vectors i-hat and j-hat as arrows."""
    if A is None:
        A = np.eye(2)
    i_hat = A @ np.array([1, 0])
    j_hat = A @ np.array([0, 1])

    for vec, color, label in (
        (i_hat, C_BLUE, label_i),
        (j_hat, C_PURPLE, label_j),
    ):
        arrow = FancyArrowPatch(
            (0, 0), (vec[0], vec[1]),
            arrowstyle="-|>",
            mutation_scale=18,
            color=color,
            linewidth=2.4,
            zorder=5,
        )
        ax.add_patch(arrow)
        if show_labels:
            # Offset the label slightly off the arrow tip
            norm = np.linalg.norm(vec)
            offset = 0.18 if norm > 0 else 0.0
            if norm > 0:
                dx = vec[0] / norm * offset + (0.08 if vec[1] >= 0 else -0.08)
                dy = vec[1] / norm * offset + (0.12 if vec[0] >= 0 else -0.12)
            else:
                dx = dy = 0.15
            ax.text(
                vec[0] + dx, vec[1] + dy, label,
                color=color, fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=6,
            )


def _matrix_caption(A: np.ndarray) -> str:
    """Render a 2x2 matrix as a one-line plain-text string for captions.

    matplotlib's built-in mathtext does not support the LaTeX `pmatrix`
    environment, so we use a compact bracket-and-semicolon notation that
    renders reliably without an external LaTeX install.
    """
    a, b = A[0]
    c, d = A[1]

    def fmt(x: float) -> str:
        if np.isclose(x, round(x)):
            return f"{int(round(x))}"
        return f"{x:.2f}"

    return f"[ {fmt(a)}, {fmt(b)} ;  {fmt(c)}, {fmt(d)} ]"


def _before_after(
    A: np.ndarray,
    suptitle: str,
    after_label: str,
    figsize: tuple[float, float] = (11, 5.2),
) -> plt.Figure:
    """Standard before / after pair with the unit grid + square + basis."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Before
    _setup_axis(axes[0], title="Before")
    _draw_grid(axes[0])
    _draw_unit_square(axes[0])
    _draw_basis(axes[0])
    # After
    _setup_axis(axes[1], title=after_label)
    _draw_grid(axes[1], A=A, color=C_AMBER, alpha=0.55, linewidth=0.9)
    _draw_unit_square(axes[1], A=A, color=C_GREEN, alpha=0.30)
    _draw_basis(axes[1], A=A)
    fig.suptitle(suptitle, fontsize=14, color=C_DARK, y=0.98)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 1: Identity / reference unit grid
# ---------------------------------------------------------------------------
def fig1_unit_grid() -> None:
    """Single-panel reference: the unit grid with i-hat and j-hat."""
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _setup_axis(ax, title="Standard basis and the unit grid")
    _draw_grid(ax)
    _draw_unit_square(ax)
    _draw_basis(ax)
    ax.text(
        0.02, 0.98,
        r"$\hat{\imath} = (1,0),\ \hat{\jmath} = (0,1)$",
        transform=ax.transAxes,
        fontsize=11, color=C_DARK,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor=C_GRAY, alpha=0.9),
    )
    fig.tight_layout()
    _save(fig, "fig1_unit_grid")


# ---------------------------------------------------------------------------
# Figure 2: Rotation
# ---------------------------------------------------------------------------
def fig2_rotation() -> None:
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    fig = _before_after(
        R,
        suptitle="Rotation by 30 deg counter-clockwise",
        after_label=f"After  R = {_matrix_caption(R)}",
    )
    _save(fig, "fig2_rotation")


# ---------------------------------------------------------------------------
# Figure 3: Scaling
# ---------------------------------------------------------------------------
def fig3_scaling() -> None:
    S = np.array([[2.0, 0.0],
                  [0.0, 1.5]])
    fig = _before_after(
        S,
        suptitle="Scaling: stretch x by 2, y by 1.5  (area scales by det = 3)",
        after_label=f"After  S = {_matrix_caption(S)}",
    )
    _save(fig, "fig3_scaling")


# ---------------------------------------------------------------------------
# Figure 4: Shear
# ---------------------------------------------------------------------------
def fig4_shear() -> None:
    H = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    fig = _before_after(
        H,
        suptitle="Horizontal shear: j-hat tilts to (1, 1), i-hat unchanged",
        after_label=f"After  H = {_matrix_caption(H)}",
    )
    _save(fig, "fig4_shear")


# ---------------------------------------------------------------------------
# Figure 5: Composition (matrix multiplication geometrically)
# ---------------------------------------------------------------------------
def fig5_composition() -> None:
    """Three panels: original  -- after A (rotate 45)  --  after B*A (then scale x by 2)."""
    theta = np.deg2rad(45)
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    B = np.array([[2.0, 0.0],
                  [0.0, 1.0]])
    BA = B @ A

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    _setup_axis(axes[0], title="Original")
    _draw_grid(axes[0])
    _draw_unit_square(axes[0])
    _draw_basis(axes[0])

    _setup_axis(axes[1], title=f"Step 1: apply A (rotate 45 deg)\nA = {_matrix_caption(A)}")
    _draw_grid(axes[1], A=A, color=C_AMBER, alpha=0.55, linewidth=0.9)
    _draw_unit_square(axes[1], A=A)
    _draw_basis(axes[1], A=A)

    _setup_axis(axes[2], title=f"Step 2: then apply B (stretch x by 2)\nBA = {_matrix_caption(BA)}")
    _draw_grid(axes[2], A=BA, color=C_AMBER, alpha=0.55, linewidth=0.9)
    _draw_unit_square(axes[2], A=BA)
    _draw_basis(axes[2], A=BA)

    fig.suptitle(
        "Matrix multiplication = composition of transformations  "
        r"(read $BA\vec{v}$ right to left)",
        fontsize=14, color=C_DARK, y=1.0,
    )
    fig.tight_layout()
    _save(fig, "fig5_composition")


# ---------------------------------------------------------------------------
# Figure 6: Identity
# ---------------------------------------------------------------------------
def fig6_identity() -> None:
    I = np.eye(2)
    fig = _before_after(
        I,
        suptitle="Identity transformation: nothing moves",
        after_label=f"After  I = {_matrix_caption(I)}",
    )
    _save(fig, "fig6_identity")


# ---------------------------------------------------------------------------
# Figure 7: Singular matrix (collapses the plane onto a line)
# ---------------------------------------------------------------------------
def fig7_singular() -> None:
    """A singular matrix sends the entire plane onto a 1D line.

    We use S = [[1, 2], [2, 4]] - the second row is 2x the first, so
    det(S) = 0 and rank(S) = 1. Every point is mapped onto the line
    spanned by the column (1, 2).
    """
    S = np.array([[1.0, 2.0],
                  [2.0, 4.0]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # Before
    _setup_axis(axes[0], lim=3.2, title="Before")
    _draw_grid(axes[0])
    _draw_unit_square(axes[0])
    _draw_basis(axes[0])

    # After: grid collapses onto the line y = 2x
    _setup_axis(axes[1], lim=6.5, title=f"After  S = {_matrix_caption(S)}   (det = 0)")
    # Draw the line that is the image of S
    t = np.linspace(-6, 6, 200)
    line_pts = np.outer(np.array([1.0, 2.0]), t)  # span of column (1,2)
    axes[1].plot(
        line_pts[0], line_pts[1],
        color=C_AMBER, linewidth=2.6, alpha=0.9, zorder=2,
        label="Column space (line  y = 2x)",
    )
    # Draw the (collapsed) transformed grid - many lines on top of each other
    _draw_grid(axes[1], A=S, color=C_AMBER, alpha=0.4, linewidth=1.0, extent=3)
    # Transformed basis vectors both land on the line
    _draw_basis(axes[1], A=S)
    # Annotate kernel: vectors (2, -1)*t collapse to 0
    k = np.array([2.0, -1.0])
    arrow = FancyArrowPatch(
        (0, 0), (k[0] * 1.2, k[1] * 1.2),
        arrowstyle="-|>", mutation_scale=14,
        color=C_GRAY, linewidth=1.6,
        linestyle=(0, (4, 3)),
        zorder=4,
    )
    axes[0].add_patch(arrow)
    axes[0].text(
        k[0] * 1.25 + 0.1, k[1] * 1.25 - 0.25,
        "kernel\ndirection",
        color=C_GRAY, fontsize=9, ha="left", va="top",
    )
    axes[1].legend(
        loc="upper left", fontsize=9, frameon=True,
        facecolor="white", edgecolor=C_GRAY,
    )

    fig.suptitle(
        "Singular matrix: the 2D plane is crushed onto a 1D line",
        fontsize=14, color=C_DARK, y=0.98,
    )
    fig.tight_layout()
    _save(fig, "fig7_singular")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Linear Algebra Chapter 03 ...")
    fig1_unit_grid()
    fig2_rotation()
    fig3_scaling()
    fig4_shear()
    fig5_composition()
    fig6_identity()
    fig7_singular()
    print("Done.")


if __name__ == "__main__":
    main()
