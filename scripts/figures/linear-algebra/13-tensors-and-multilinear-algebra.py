"""
Figure generation script for Linear Algebra Chapter 13:
Tensors and Multilinear Algebra.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure focuses on one idea, in 3Blue1Brown style.

Figures:
    fig1_tensor_hierarchy      Scalar (0D), vector (1D), matrix (2D),
                               3rd-order tensor (3D) lined up to show how
                               order grows by stacking the previous one.
    fig2_tensor_as_stack       3rd-order tensor visualised as a stack of
                               frontal slices (matrices) -- "video as
                               sequence of frames".
    fig3_contraction_einsum    Tensor contraction (Einstein summation):
                               matrix-matrix multiply panel + general
                               contraction panel showing the shared index
                               that gets summed away.
    fig4_cp_decomposition      CP decomposition: a tensor as a sum of
                               rank-1 outer-product cubes.
    fig5_tucker_decomposition  Tucker decomposition: small core tensor
                               multiplied by a factor matrix on each mode.
    fig6_mode_n_unfolding      Mode-n unfolding of a 3x4x2 tensor,
                               showing how slices line up as columns of
                               the resulting matrix.
    fig7_image_as_tensor       Application: an image is a 3D tensor of
                               shape (H, W, 3) -- stack of three channel
                               matrices.

Usage:
    python3 scripts/figures/linear-algebra/13-tensors-and-multilinear-algebra.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "13-tensors-and-multilinear-algebra"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "13-张量与多线性代数"
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
# 3D drawing helpers
# ---------------------------------------------------------------------------
def _cube_faces(x0, y0, z0, dx, dy, dz):
    """Return the 6 quadrilateral faces of a cuboid as a list of vertex arrays."""
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
    ]


def _draw_cuboid(ax, x0, y0, z0, dx, dy, dz, color, alpha=0.55, edge=C_DARK,
                 lw=0.8):
    faces = _cube_faces(x0, y0, z0, dx, dy, dz)
    poly = Poly3DCollection(faces, facecolors=color, edgecolors=edge,
                            linewidths=lw, alpha=alpha)
    ax.add_collection3d(poly)


def _setup_3d(ax, lim, elev=22, azim=-55):
    ax.set_xlim(0, lim[0])
    ax.set_ylim(0, lim[1])
    ax.set_zlim(0, lim[2])
    ax.set_box_aspect((lim[0], lim[1], lim[2]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _draw_voxel_grid(ax, nx, ny, nz, base_color, alpha=0.32, edge=C_DARK,
                     lw=0.4, gap=0.05):
    """Draw an nx x ny x nz grid of unit voxels."""
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                _draw_cuboid(ax,
                             i + gap, j + gap, k + gap,
                             1 - 2 * gap, 1 - 2 * gap, 1 - 2 * gap,
                             color=base_color, alpha=alpha,
                             edge=edge, lw=lw)


# ---------------------------------------------------------------------------
# Fig 1: Scalar -> Vector -> Matrix -> 3rd-order tensor
# ---------------------------------------------------------------------------
def fig1_tensor_hierarchy():
    fig = plt.figure(figsize=(13, 3.6))

    # Layout: four panels of equal width
    panel_y = 0.10
    panel_h = 0.66
    panel_w = 0.20
    panel_xs = [0.04, 0.28, 0.52, 0.76]

    # --- Panel 1: Scalar (0D) -------------------------------------------------
    ax1 = fig.add_axes((panel_xs[0], panel_y, panel_w, panel_h))
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 4)
    ax1.set_aspect("equal")
    ax1.set_axis_off()
    box = FancyBboxPatch((1.4, 1.4), 1.2, 1.2,
                         boxstyle="round,pad=0.05",
                         facecolor=C_BLUE, alpha=0.55,
                         edgecolor=C_DARK, lw=1.2)
    ax1.add_patch(box)
    ax1.text(2.0, 2.0, "7", ha="center", va="center",
             fontsize=22, color="white", fontweight="bold")
    ax1.set_title("Scalar  (order 0)\nshape: ()", fontsize=11,
                  color=C_DARK, fontweight="bold", pad=8)
    ax1.text(2.0, 0.4, "a single number",
             ha="center", fontsize=9, color=C_GRAY, style="italic")

    # --- Panel 2: Vector (1D) -------------------------------------------------
    ax2 = fig.add_axes((panel_xs[1], panel_y, panel_w, panel_h))
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 4)
    ax2.set_aspect("equal")
    ax2.set_axis_off()
    vals = [3, 1, 4, 1, 5]
    for i, v in enumerate(vals):
        x = 0.2 + i * 0.92
        box = FancyBboxPatch((x, 1.4), 0.85, 1.2,
                             boxstyle="round,pad=0.02",
                             facecolor=C_BLUE, alpha=0.55,
                             edgecolor=C_DARK, lw=1.0)
        ax2.add_patch(box)
        ax2.text(x + 0.42, 2.0, str(v), ha="center", va="center",
                 fontsize=14, color="white", fontweight="bold")
    ax2.set_title("Vector  (order 1)\nshape: (5,)", fontsize=11,
                  color=C_DARK, fontweight="bold", pad=8)
    ax2.text(2.5, 0.4, "a row of numbers",
             ha="center", fontsize=9, color=C_GRAY, style="italic")

    # --- Panel 3: Matrix (2D) -------------------------------------------------
    ax3 = fig.add_axes((panel_xs[2], panel_y, panel_w, panel_h))
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 4)
    ax3.set_aspect("equal")
    ax3.set_axis_off()
    mat = np.array([[2, 7, 1, 8],
                    [3, 1, 4, 1],
                    [5, 9, 2, 6]])
    for i in range(3):
        for j in range(4):
            x = 0.15 + j * 0.88
            y = 2.7 - i * 0.88
            box = FancyBboxPatch((x, y), 0.82, 0.82,
                                 boxstyle="round,pad=0.02",
                                 facecolor=C_PURPLE, alpha=0.55,
                                 edgecolor=C_DARK, lw=1.0)
            ax3.add_patch(box)
            ax3.text(x + 0.41, y + 0.41, str(mat[i, j]),
                     ha="center", va="center",
                     fontsize=12, color="white", fontweight="bold")
    ax3.set_title("Matrix  (order 2)\nshape: (3, 4)", fontsize=11,
                  color=C_DARK, fontweight="bold", pad=8)
    ax3.text(2.0, 0.4, "a table of numbers",
             ha="center", fontsize=9, color=C_GRAY, style="italic")

    # --- Panel 4: 3rd-order tensor -------------------------------------------
    ax4 = fig.add_axes((panel_xs[3], panel_y - 0.04, panel_w, panel_h + 0.04),
                       projection="3d")
    _setup_3d(ax4, (3.6, 3.6, 2.6), elev=20, azim=-55)
    _draw_voxel_grid(ax4, 3, 3, 2, base_color=C_GREEN, alpha=0.42, gap=0.08)
    ax4.set_title("3rd-order tensor\nshape: (3, 3, 2)", fontsize=11,
                  color=C_DARK, fontweight="bold", pad=0)
    ax4.text2D(0.5, -0.02, "a cube of numbers",
               transform=ax4.transAxes, ha="center",
               fontsize=9, color=C_GRAY, style="italic")

    fig.suptitle("Tensors generalise scalars, vectors, and matrices to "
                 "arbitrary dimensions",
                 fontsize=12.5, color=C_DARK, fontweight="bold", y=0.98)
    save(fig, "fig1_tensor_hierarchy")


# ---------------------------------------------------------------------------
# Fig 2: 3rd-order tensor as a stack of matrices (frontal slices)
# ---------------------------------------------------------------------------
def fig2_tensor_as_stack():
    fig = plt.figure(figsize=(13, 4.5))

    # --- Left: full tensor ----------------------------------------------------
    ax1 = fig.add_axes((0.02, 0.04, 0.34, 0.84), projection="3d")
    _setup_3d(ax1, (4.6, 4.6, 4.2), elev=18, azim=-58)

    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    for k in range(4):
        z = 0.3 + k * 1.0
        for i in range(4):
            for j in range(4):
                _draw_cuboid(ax1,
                             0.3 + i * 1.0, 0.3 + j * 1.0, z,
                             0.85, 0.85, 0.7,
                             color=colors[k], alpha=0.45, lw=0.4)
    ax1.set_title("A 3rd-order tensor  $\\mathcal{X} \\in "
                  "\\mathbb{R}^{4\\times 4\\times 4}$",
                  fontsize=12, color=C_DARK, fontweight="bold", pad=0)

    # --- Right: stack of 4 frontal slices, each a matrix --------------------
    ax2 = fig.add_axes((0.40, 0.10, 0.58, 0.78))
    ax2.set_xlim(0, 18)
    ax2.set_ylim(0, 6)
    ax2.set_aspect("equal")
    ax2.set_axis_off()

    for k in range(4):
        x_off = 0.5 + k * 4.4
        frame = FancyBboxPatch((x_off, 0.6), 4.0, 4.0,
                               boxstyle="round,pad=0.05",
                               facecolor=colors[k], alpha=0.18,
                               edgecolor=colors[k], lw=1.4)
        ax2.add_patch(frame)
        for i in range(4):
            for j in range(4):
                x = x_off + 0.15 + j * 0.92
                y = 0.75 + (3 - i) * 0.92
                cell = FancyBboxPatch((x, y), 0.85, 0.85,
                                      boxstyle="round,pad=0.02",
                                      facecolor=colors[k], alpha=0.55,
                                      edgecolor=C_DARK, lw=0.4)
                ax2.add_patch(cell)
        ax2.text(x_off + 2.0, 5.0,
                 f"$\\mathbf{{X}}_{{::,{k}}}$",
                 ha="center", fontsize=11, color=C_DARK, fontweight="bold")

    # arrow: tensor -> stack
    ax2.annotate("", xy=(0.4, 3.0), xytext=(-1.0, 3.0),
                 arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.5))
    ax2.text(8.7, 0.05, "Each frontal slice is a $4\\times 4$ matrix; "
             "stack them to recover the tensor.",
             ha="center", fontsize=10, color=C_DARK, style="italic")

    fig.suptitle("Visualising a 3D tensor: a stack of 2D matrices",
                 fontsize=13, color=C_DARK, fontweight="bold", y=0.98)
    save(fig, "fig2_tensor_as_stack")


# ---------------------------------------------------------------------------
# Fig 3: Tensor contraction (Einstein summation)
# ---------------------------------------------------------------------------
def fig3_contraction_einsum():
    fig = plt.figure(figsize=(14, 5.4))

    # --- Left: matrix multiplication -----------------------------------------
    ax1 = fig.add_axes((0.04, 0.10, 0.55, 0.78))
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 6)
    ax1.set_aspect("equal")
    ax1.set_axis_off()

    def draw_matrix(x, y, w, h, color, label, dim_top, dim_side):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.04",
                              facecolor=color, alpha=0.55,
                              edgecolor=C_DARK, lw=1.2)
        ax1.add_patch(rect)
        ax1.text(x + w / 2, y + h + 0.35, label, ha="center",
                 fontsize=14, color=C_DARK, fontweight="bold")
        ax1.text(x + w / 2, y - 0.45, dim_top, ha="center",
                 fontsize=10, color=C_GRAY)
        ax1.text(x - 0.35, y + h / 2, dim_side, ha="right", va="center",
                 fontsize=10, color=C_GRAY)

    # A: i x j  (3 x 4)
    draw_matrix(0.7, 1.6, 2.2, 3.0, C_BLUE,
                r"$A_{ij}$", "$j$", "$i$")
    ax1.text(1.8, 0.55, "$3\\times 4$", ha="center", fontsize=9, color=C_GRAY)

    # B: j x k  (4 x 5)
    draw_matrix(4.6, 1.6, 3.0, 2.2, C_PURPLE,
                r"$B_{jk}$", "$k$", "$j$")
    ax1.text(6.1, 0.55, "$4\\times 5$", ha="center", fontsize=9, color=C_GRAY)

    # = sign
    ax1.text(8.4, 2.7, "=", ha="center", va="center", fontsize=24,
             color=C_DARK, fontweight="bold")

    # C: i x k  (3 x 5)
    draw_matrix(9.1, 1.9, 3.0, 2.4, C_GREEN,
                r"$C_{ik}$", "$k$", "$i$")
    ax1.text(10.6, 0.85, "$3\\times 5$", ha="center", fontsize=9, color=C_GRAY)

    # contraction arrow over j
    ax1.annotate("",
                 xy=(5.0, 5.1), xytext=(2.6, 5.1),
                 arrowprops=dict(arrowstyle="<->", color=C_AMBER, lw=2.0))
    ax1.text(3.85, 5.45, "shared index $j$ is summed away",
             ha="center", fontsize=10, color=C_AMBER, fontweight="bold")

    ax1.text(7.0, -0.05,
             r"$C_{ik} = \sum_j A_{ij} B_{jk}$    (matrix multiplication)",
             ha="center", fontsize=12, color=C_DARK)

    ax1.set_title("Familiar case: matrix multiply",
                  fontsize=12, color=C_DARK, fontweight="bold")

    # --- Right: 3D tensor contraction ----------------------------------------
    ax2 = fig.add_axes((0.62, 0.03, 0.36, 0.78), projection="3d")
    _setup_3d(ax2, (7.5, 4.0, 4.0), elev=20, azim=-58)

    # Tensor A: 3 x 3 x 4 cube on the left
    for i in range(3):
        for j in range(3):
            for k in range(4):
                _draw_cuboid(ax2,
                             0.2 + i * 0.55, 0.2 + j * 0.55, 0.2 + k * 0.55,
                             0.5, 0.5, 0.5,
                             color=C_BLUE, alpha=0.32, lw=0.3)
    ax2.text(0.4, 1.0, 3.4, r"$\mathcal{A}_{i_1 i_2 j}$",
             color=C_BLUE, fontsize=12, fontweight="bold")

    # Tensor B: 4 x 3 x 3 cube on the right
    for i in range(4):
        for j in range(3):
            for k in range(3):
                _draw_cuboid(ax2,
                             4.2 + i * 0.55, 0.2 + j * 0.55, 0.2 + k * 0.55,
                             0.5, 0.5, 0.5,
                             color=C_PURPLE, alpha=0.32, lw=0.3)
    ax2.text(4.8, 1.0, 3.0, r"$\mathcal{B}_{j k_1 k_2}$",
             color=C_PURPLE, fontsize=12, fontweight="bold")

    # contraction arrow / label between them
    ax2.text(3.6, 1.0, 4.2, "contract on $j$",
             color=C_AMBER, fontsize=11, fontweight="bold",
             ha="center")

    # General-contraction title placed below the cubes (avoid suptitle clash)
    fig.text(0.80, 0.83,
             "General contraction:  "
             "$\\mathcal{C}_{i_1 i_2 k_1 k_2} = "
             "\\sum_j \\mathcal{A}_{i_1 i_2 j}\\, \\mathcal{B}_{j k_1 k_2}$",
             ha="center", fontsize=11, color=C_DARK, fontweight="bold")

    fig.suptitle("Tensor contraction = pair indices, multiply, sum away",
                 fontsize=13, color=C_DARK, fontweight="bold", y=0.97, x=0.32)
    save(fig, "fig3_contraction_einsum")


# ---------------------------------------------------------------------------
# Fig 4: CP decomposition -- sum of rank-1 tensors
# ---------------------------------------------------------------------------
def fig4_cp_decomposition():
    fig = plt.figure(figsize=(14, 4.4))

    def draw_rank1_block(ax, color, alpha=0.45):
        """Draw a small 4x4x3 rank-1 cube."""
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    _draw_cuboid(ax,
                                 0.1 + i * 0.55, 0.1 + j * 0.55,
                                 0.1 + k * 0.55,
                                 0.5, 0.5, 0.5,
                                 color=color, alpha=alpha, lw=0.3)

    def draw_vector_arrow(ax, p0, p1, color, label):
        ax.plot(*zip(p0, p1), color=color, lw=2.6)
        ax.scatter(*p1, color=color, s=30)
        ax.text(p1[0] + 0.15, p1[1] + 0.15, p1[2] + 0.15,
                label, color=color, fontsize=10, fontweight="bold")

    # Layout: 4 cube panels, with =, +, + symbols overlaid via fig.text
    # left edges of the four panels (each width 0.18):
    panel_w = 0.18
    panel_h = 0.72
    panel_y = 0.10
    panel_xs = [0.04, 0.28, 0.52, 0.76]

    # Panel 1: full tensor
    ax_main = fig.add_axes((panel_xs[0], panel_y, panel_w, panel_h),
                           projection="3d")
    _setup_3d(ax_main, (3.0, 3.0, 2.5), elev=20, azim=-55)
    draw_rank1_block(ax_main, color=C_DARK, alpha=0.28)
    ax_main.set_title(r"$\mathcal{X}$",
                      fontsize=14, color=C_DARK, fontweight="bold", pad=2)

    # equals + plus signs between panels
    sym_y = panel_y + panel_h / 2
    for sx, sym in zip([panel_xs[0] + panel_w, panel_xs[1] + panel_w,
                        panel_xs[2] + panel_w],
                       [r"$\approx$", "+", "+"]):
        fig.text(sx + (panel_xs[1] - panel_xs[0] - panel_w) / 2,
                 sym_y, sym, ha="center", va="center", fontsize=26,
                 color=C_DARK, fontweight="bold")

    # Panels 2..4: rank-1 components
    colors = [C_BLUE, C_PURPLE, C_GREEN]
    for r, color in enumerate(colors):
        ax = fig.add_axes((panel_xs[r + 1], panel_y, panel_w, panel_h),
                          projection="3d")
        _setup_3d(ax, (3.0, 3.0, 2.5), elev=20, azim=-55)

        draw_rank1_block(ax, color=color, alpha=0.30)

        a = np.array([2.5, 0.0, 0.0]) * (0.85 + 0.05 * r)
        b = np.array([0.0, 2.5, 0.0]) * (0.85 + 0.05 * r)
        c = np.array([0.0, 0.0, 2.0]) * (0.85 + 0.05 * r)
        origin = (0.1, 0.1, 0.1)
        draw_vector_arrow(ax, origin,
                          (origin[0] + a[0], origin[1] + a[1], origin[2] + a[2]),
                          C_BLUE, f"$\\mathbf{{a}}_{r+1}$")
        draw_vector_arrow(ax, origin,
                          (origin[0] + b[0], origin[1] + b[1], origin[2] + b[2]),
                          C_PURPLE, f"$\\mathbf{{b}}_{r+1}$")
        draw_vector_arrow(ax, origin,
                          (origin[0] + c[0], origin[1] + c[1], origin[2] + c[2]),
                          C_GREEN, f"$\\mathbf{{c}}_{r+1}$")

        ax.set_title(f"$\\lambda_{{{r+1}}}\\, \\mathbf{{a}}_{r+1} \\circ "
                     f"\\mathbf{{b}}_{r+1} \\circ \\mathbf{{c}}_{r+1}$",
                     fontsize=11, color=C_DARK, fontweight="bold", pad=2)

    fig.suptitle("CP decomposition:  "
                 r"$\mathcal{X} \approx \sum_{r=1}^{R} \lambda_r\, "
                 r"\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r$"
                 "    (sum of rank-1 outer products)",
                 fontsize=13, color=C_DARK, fontweight="bold", y=0.97)
    save(fig, "fig4_cp_decomposition")


# ---------------------------------------------------------------------------
# Fig 5: Tucker decomposition
# ---------------------------------------------------------------------------
def fig5_tucker_decomposition():
    fig = plt.figure(figsize=(14, 4.6))

    # Layout via fig.add_axes for full control
    # Slot widths and positions (in figure fractions):
    #   X cube : approx 0.16
    #   core G : 0.13
    #   A      : 0.07
    #   B      : 0.07
    #   C      : 0.07
    # symbols (=, x1, x2, x3) sit between slots
    panel_y = 0.10
    panel_h = 0.70
    # left edges
    x_x = 0.04
    w_x = 0.16
    x_g = 0.26
    w_g = 0.13
    x_a = 0.46
    w_a = 0.08
    x_b = 0.64
    w_b = 0.08
    x_c = 0.82
    w_c = 0.08

    # X tensor
    ax_x = fig.add_axes((x_x, panel_y, w_x, panel_h), projection="3d")
    _setup_3d(ax_x, (5.5, 5.5, 4.5), elev=20, azim=-55)
    _draw_voxel_grid(ax_x, 5, 5, 4, base_color=C_DARK, alpha=0.22, gap=0.07)
    ax_x.set_title(r"$\mathcal{X}$" + "\n$I\\times J\\times K$",
                   fontsize=11, color=C_DARK, fontweight="bold", pad=0)

    # approx sign between X and G
    fig.text((x_x + w_x + x_g) / 2, panel_y + panel_h / 2, r"$\approx$",
             fontsize=24, ha="center", va="center", color=C_DARK,
             fontweight="bold")

    # Core G
    ax_g = fig.add_axes((x_g, panel_y, w_g, panel_h), projection="3d")
    _setup_3d(ax_g, (3.0, 3.0, 2.4), elev=20, azim=-55)
    _draw_voxel_grid(ax_g, 3, 3, 2, base_color=C_AMBER, alpha=0.55, gap=0.06)
    ax_g.set_title(r"core $\mathcal{G}$" + "\n$P\\times Q\\times R$",
                   fontsize=11, color=C_DARK, fontweight="bold", pad=0)

    # x_1 between G and A
    fig.text((x_g + w_g + x_a) / 2, panel_y + panel_h / 2, r"$\times_1$",
             fontsize=20, ha="center", va="center", color=C_DARK,
             fontweight="bold")

    # Factor A: I x P
    ax_a = fig.add_axes((x_a, panel_y, w_a, panel_h))
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_axis_off()
    rect = FancyBboxPatch((0.18, 0.10), 0.64, 0.78,
                          boxstyle="round,pad=0.02",
                          facecolor=C_BLUE, alpha=0.55,
                          edgecolor=C_DARK, lw=1.2,
                          transform=ax_a.transAxes)
    ax_a.add_patch(rect)
    ax_a.text(0.5, 0.95, r"$\mathbf{A}$", ha="center", fontsize=13,
              color=C_DARK, fontweight="bold")
    ax_a.text(0.5, 0.02, "$I \\times P$", ha="center", fontsize=10,
              color=C_GRAY)

    fig.text((x_a + w_a + x_b) / 2, panel_y + panel_h / 2, r"$\times_2$",
             fontsize=20, ha="center", va="center", color=C_DARK,
             fontweight="bold")

    # Factor B: J x Q
    ax_b = fig.add_axes((x_b, panel_y, w_b, panel_h))
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_axis_off()
    rect = FancyBboxPatch((0.18, 0.18), 0.64, 0.65,
                          boxstyle="round,pad=0.02",
                          facecolor=C_PURPLE, alpha=0.55,
                          edgecolor=C_DARK, lw=1.2,
                          transform=ax_b.transAxes)
    ax_b.add_patch(rect)
    ax_b.text(0.5, 0.95, r"$\mathbf{B}$", ha="center", fontsize=13,
              color=C_DARK, fontweight="bold")
    ax_b.text(0.5, 0.10, "$J \\times Q$", ha="center", fontsize=10,
              color=C_GRAY)

    fig.text((x_b + w_b + x_c) / 2, panel_y + panel_h / 2, r"$\times_3$",
             fontsize=20, ha="center", va="center", color=C_DARK,
             fontweight="bold")

    # Factor C: K x R
    ax_c = fig.add_axes((x_c, panel_y, w_c, panel_h))
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.set_axis_off()
    rect = FancyBboxPatch((0.18, 0.27), 0.64, 0.55,
                          boxstyle="round,pad=0.02",
                          facecolor=C_GREEN, alpha=0.55,
                          edgecolor=C_DARK, lw=1.2,
                          transform=ax_c.transAxes)
    ax_c.add_patch(rect)
    ax_c.text(0.5, 0.95, r"$\mathbf{C}$", ha="center", fontsize=13,
              color=C_DARK, fontweight="bold")
    ax_c.text(0.5, 0.18, "$K \\times R$", ha="center", fontsize=10,
              color=C_GRAY)

    fig.suptitle(r"Tucker decomposition:  "
                 r"$\mathcal{X} \approx \mathcal{G} "
                 r"\times_1 \mathbf{A} \times_2 \mathbf{B} \times_3 \mathbf{C}$"
                 "    (small core + a factor matrix on every mode)",
                 fontsize=12.5, color=C_DARK, fontweight="bold", y=0.97)
    save(fig, "fig5_tucker_decomposition")


# ---------------------------------------------------------------------------
# Fig 6: Mode-n unfolding
# ---------------------------------------------------------------------------
def fig6_mode_n_unfolding():
    fig = plt.figure(figsize=(14, 4.6))

    # --- Left: 3 x 4 x 2 tensor ----------------------------------------------
    ax_t = fig.add_axes((0.02, 0.04, 0.32, 0.84), projection="3d")
    _setup_3d(ax_t, (3.4, 4.4, 2.4), elev=22, azim=-55)

    np.random.seed(0)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, 4 * 2))
    fiber_colors = cmap.reshape(4, 2, 4)

    for i in range(3):
        for j in range(4):
            for k in range(2):
                c = fiber_colors[j, k]
                _draw_cuboid(ax_t,
                             0.2 + i * 1.0, 0.2 + j * 1.0, 0.2 + k * 1.0,
                             0.85, 0.85, 0.85,
                             color=tuple(c[:3]), alpha=0.55, lw=0.5)
    ax_t.set_title(r"Tensor  $\mathcal{X} \in \mathbb{R}^{3\times 4\times 2}$",
                   fontsize=11.5, color=C_DARK, fontweight="bold", pad=0)
    ax_t.text2D(0.5, -0.02,
                "color = one mode-1 fiber  (fix $j, k$, vary $i$)",
                transform=ax_t.transAxes, ha="center", fontsize=9,
                color=C_GRAY, style="italic")

    # --- Middle: arrow + label -----------------------------------------------
    fig.text(0.40, 0.50, r"$\longrightarrow$", ha="center", va="center",
             fontsize=22, color=C_DARK, fontweight="bold")
    fig.text(0.40, 0.58, "mode-1 unfold",
             ha="center", fontsize=10, color=C_DARK, fontweight="bold")

    # --- Right: unfolded matrix 3 x 8 ---------------------------------------
    ax_m = fig.add_axes((0.46, 0.10, 0.52, 0.78))
    ax_m.set_xlim(0, 10)
    ax_m.set_ylim(0, 5)
    ax_m.set_aspect("equal")
    ax_m.set_axis_off()

    cell_w, cell_h = 1.05, 1.05
    x0, y0 = 0.6, 0.7
    for col in range(8):
        j = col % 4
        k = col // 4
        c = fiber_colors[j, k]
        for row in range(3):
            x = x0 + col * cell_w
            y = y0 + (2 - row) * cell_h
            cell = FancyBboxPatch((x, y), cell_w * 0.94, cell_h * 0.94,
                                  boxstyle="round,pad=0.02",
                                  facecolor=tuple(c[:3]), alpha=0.55,
                                  edgecolor=C_DARK, lw=0.5)
            ax_m.add_patch(cell)

    for col in range(8):
        j = col % 4
        k = col // 4
        x = x0 + col * cell_w + cell_w * 0.47
        ax_m.text(x, y0 - 0.35, f"$({j},{k})$",
                  ha="center", fontsize=8, color=C_GRAY)

    outline = Rectangle((x0 - 0.05, y0 - 0.05),
                        8 * cell_w + 0.1, 3 * cell_h + 0.1,
                        fill=False, edgecolor=C_DARK, lw=1.5)
    ax_m.add_patch(outline)

    ax_m.text(5.0, 4.4, r"$\mathbf{X}_{(1)} \in \mathbb{R}^{3 \times 8}$",
              ha="center", fontsize=12, color=C_DARK, fontweight="bold")
    ax_m.text(5.0, 0.1,
              "Each column is one mode-1 fiber of $\\mathcal{X}$",
              ha="center", fontsize=10, color=C_DARK, style="italic")

    fig.suptitle("Mode-1 unfolding: stack the mode-1 fibers as columns",
                 fontsize=13, color=C_DARK, fontweight="bold", y=0.98)
    save(fig, "fig6_mode_n_unfolding")


# ---------------------------------------------------------------------------
# Fig 7: Image as a 3D tensor (H x W x 3)
# ---------------------------------------------------------------------------
def fig7_image_as_tensor():
    fig = plt.figure(figsize=(14, 4.8))

    # Build a small synthetic RGB image
    H, W = 32, 32
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    R = np.clip(0.9 - 0.7 * (xx / W), 0, 1)
    G = np.clip(0.2 + 0.8 * (yy / H), 0, 1)
    Bch = np.clip(0.3 + 0.7 * np.sin(np.hypot(xx - W / 2,
                                              yy - H / 2) / 5), 0, 1)
    img = np.stack([R, G, Bch], axis=-1)

    # --- Panel 1: full RGB image -----------------------------------------------
    ax1 = fig.add_axes((0.04, 0.18, 0.18, 0.66))
    ax1.imshow(img, interpolation="nearest")
    ax1.set_title("Color image", fontsize=11.5,
                  color=C_DARK, fontweight="bold", pad=8)
    ax1.text(W / 2, H + 4,
             f"shape: $({H}, {W}, 3)$",
             ha="center", fontsize=10, color=C_DARK)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # = sign
    fig.text(0.255, 0.50, "=", ha="center", va="center",
             fontsize=24, color=C_DARK, fontweight="bold")

    # --- Panel 2: 3D stack of channels ----------------------------------------
    ax2 = fig.add_axes((0.28, 0.02, 0.42, 0.80), projection="3d")
    _setup_3d(ax2, (W, H, 12), elev=22, azim=-58)
    ax2.set_axis_off()

    channel_imgs = [
        (np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=-1), 0, "R"),
        (np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=-1), 4, "G"),
        (np.stack([np.zeros_like(Bch), np.zeros_like(Bch), Bch], axis=-1), 8, "B"),
    ]

    for slice_img, z, label in channel_imgs:
        step = 2
        for i in range(0, H, step):
            for j in range(0, W, step):
                color = slice_img[i, j]
                if color.sum() < 0.05:
                    continue
                verts = [
                    (j, H - i - step, z),
                    (j + step, H - i - step, z),
                    (j + step, H - i, z),
                    (j, H - i, z),
                ]
                poly = Poly3DCollection([verts], facecolors=[tuple(color)],
                                        edgecolors="none", alpha=0.95)
                ax2.add_collection3d(poly)
        # outline of the plane
        outline = [
            [(0, 0, z), (W, 0, z)],
            [(W, 0, z), (W, H, z)],
            [(W, H, z), (0, H, z)],
            [(0, H, z), (0, 0, z)],
        ]
        ax2.add_collection3d(Line3DCollection(outline,
                                              colors=C_DARK, linewidths=1.0))
        ax2.text(-2, H + 1, z + 0.4, label,
                 fontsize=14, fontweight="bold", color=C_DARK)

    ax2.set_title("3 channels stacked  $\\mathcal{X} \\in "
                  "\\mathbb{R}^{H\\times W\\times 3}$",
                  fontsize=11, color=C_DARK, fontweight="bold", pad=0)

    # --- Panel 3: three channels stacked vertically ---------------------------
    for idx, (ch, name, cmap) in enumerate([(R, "R", "Reds"),
                                            (G, "G", "Greens"),
                                            (Bch, "B", "Blues")]):
        ax = fig.add_axes((0.78, 0.66 - idx * 0.27, 0.18, 0.22))
        ax.imshow(ch, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"channel {name}", fontsize=10, color=C_DARK,
                     fontweight="bold", pad=2)

    fig.suptitle("Application: an image is naturally a 3rd-order tensor",
                 fontsize=13, color=C_DARK, fontweight="bold", y=0.99)
    save(fig, "fig7_image_as_tensor")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_tensor_hierarchy()
    fig2_tensor_as_stack()
    fig3_contraction_einsum()
    fig4_cp_decomposition()
    fig5_tucker_decomposition()
    fig6_mode_n_unfolding()
    fig7_image_as_tensor()
    print("OK -- 7 figures written to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
