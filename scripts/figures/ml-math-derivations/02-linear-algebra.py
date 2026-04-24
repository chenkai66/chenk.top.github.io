"""
Figure generation for ML Math Derivations Part 02: Linear Algebra & Matrix Theory.

Produces seven figures targeted at the *ML practitioner* view of linear algebra
(geometry over algebra; what each tool does to data, not how to row-reduce).

Figures
-------
    fig1_vector_space_subspace      A 2D plane (subspace) inside R^3 - what a
                                    "subspace" actually looks like, with two
                                    spanning vectors and a sample point.
    fig2_linear_dependence          Side-by-side: independent basis (full 2D
                                    parallelogram) vs dependent vectors
                                    (collapsed to a line) - rank visualised.
    fig3_eigendecomposition         A symmetric matrix acting on the unit
                                    circle: eigenvectors are the only
                                    directions that don't rotate; circle
                                    becomes an ellipse aligned with them.
    fig4_svd_three_steps            SVD as rotate-scale-rotate: unit circle ->
                                    rotated by V^T -> scaled by Sigma ->
                                    rotated by U. Four panels in a row.
    fig5_rank_deficiency            Rank-deficient matrix collapses 2D plane
                                    onto a 1D line - everything in the null
                                    space gets "killed".
    fig6_matrix_calculus_shapes     Shape rules cheat-sheet: scalar/vector/
                                    matrix derivative shapes laid out as
                                    coloured blocks with labels.
    fig7_positive_definite_ellipse  Quadratic form x^T A x = 1 drawn as an
                                    ellipse, principal axes = eigenvectors,
                                    semi-axis lengths = 1/sqrt(lambda).

All figures: dpi=150, seaborn-v0_8-whitegrid, palette
{#2563eb blue, #7c3aed purple, #10b981 green, #f59e0b amber}.

Saved into both EN and ZH asset folders so the markdown can reference them
with relative paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#6b7280"

DPI = 150

# ----------------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------------
ROOT = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = ROOT / "source/_posts/en/ml-math-derivations/02-Linear-Algebra-and-Matrix-Theory"
ZH_DIR = ROOT / "source/_posts/zh/ml-math-derivations/02-线性代数与矩阵论"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure into both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png")


# ----------------------------------------------------------------------------
# fig1: Vector space and a 2D subspace inside R^3
# ----------------------------------------------------------------------------
def fig1_vector_space_subspace() -> None:
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # Two spanning vectors of the subspace (a plane through origin)
    v1 = np.array([1.5, 0.4, 0.8])
    v2 = np.array([0.3, 1.4, -0.4])

    # Build plane patch from v1, v2
    s = np.linspace(-1.2, 1.2, 2)
    t = np.linspace(-1.2, 1.2, 2)
    S, T = np.meshgrid(s, t)
    X = S * v1[0] + T * v2[0]
    Y = S * v1[1] + T * v2[1]
    Z = S * v1[2] + T * v2[2]

    # Plane (the subspace W = span{v1, v2})
    verts = [list(zip(X.flatten()[[0, 1, 3, 2]],
                      Y.flatten()[[0, 1, 3, 2]],
                      Z.flatten()[[0, 1, 3, 2]]))]
    poly = Poly3DCollection(verts, alpha=0.22, facecolor=C_BLUE, edgecolor=C_BLUE)
    ax.add_collection3d(poly)

    # Spanning vectors
    for v, c, lbl in [(v1, C_PURPLE, r"$v_1$"), (v2, C_GREEN, r"$v_2$")]:
        ax.quiver(0, 0, 0, *v, color=c, arrow_length_ratio=0.12, linewidth=2.5)
        ax.text(v[0] * 1.15, v[1] * 1.15, v[2] * 1.15, lbl, fontsize=13, color=c)

    # A sample point inside the subspace, written as 0.6 v1 + 0.8 v2
    p = 0.6 * v1 + 0.8 * v2
    ax.scatter(*p, color=C_AMBER, s=70, zorder=5)
    ax.text(p[0] + 0.08, p[1] + 0.08, p[2] + 0.08,
            r"$0.6 v_1 + 0.8 v_2 \in W$", fontsize=11, color=C_AMBER)

    # A point outside the subspace
    out = np.array([0.4, -0.6, 1.6])
    ax.scatter(*out, color=C_GRAY, s=50)
    ax.text(out[0] + 0.05, out[1] + 0.05, out[2] + 0.05,
            r"$\notin W$", fontsize=11, color=C_GRAY)

    # Axes
    L = 2.0
    ax.quiver(0, 0, 0, L, 0, 0, color="k", arrow_length_ratio=0.05, linewidth=0.8)
    ax.quiver(0, 0, 0, 0, L, 0, color="k", arrow_length_ratio=0.05, linewidth=0.8)
    ax.quiver(0, 0, 0, 0, 0, L, color="k", arrow_length_ratio=0.05, linewidth=0.8)
    ax.text(L * 1.05, 0, 0, "x", fontsize=10)
    ax.text(0, L * 1.05, 0, "y", fontsize=10)
    ax.text(0, 0, L * 1.05, "z", fontsize=10)

    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-2, 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-50)
    ax.set_title(r"A 2D subspace $W = \mathrm{span}\{v_1, v_2\}$ inside $\mathbb{R}^3$",
                 fontsize=13)
    ax.grid(False)
    fig.tight_layout()
    save(fig, "fig1_vector_space_subspace")


# ----------------------------------------------------------------------------
# fig2: Linear independence vs dependence
# ----------------------------------------------------------------------------
def fig2_linear_dependence() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Independent
    ax = axes[0]
    a = np.array([2.2, 0.4])
    b = np.array([0.6, 1.8])
    # Parallelogram
    poly = plt.Polygon([[0, 0], a, a + b, b], alpha=0.22, color=C_BLUE)
    ax.add_patch(poly)
    ax.annotate("", xy=a, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=2.5))
    ax.annotate("", xy=b, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2.5))
    ax.text(a[0] + 0.05, a[1] - 0.15, r"$v_1$", fontsize=14, color=C_PURPLE)
    ax.text(b[0] - 0.25, b[1] + 0.05, r"$v_2$", fontsize=14, color=C_GREEN)
    ax.set_title("Linearly independent\n"
                 r"span $= \mathbb{R}^2$ (area > 0)", fontsize=12)
    ax.set_xlim(-1, 3.5); ax.set_ylim(-1, 3)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

    # Dependent
    ax = axes[1]
    a = np.array([2.0, 1.0])
    b = 1.6 * a  # b = 1.6 a -> dependent
    # Just a line segment
    line_end = 2.5 * a
    ax.plot([-line_end[0], line_end[0]], [-line_end[1], line_end[1]],
            color=C_BLUE, lw=1.5, alpha=0.5, linestyle="--",
            label=r"span = line")
    ax.annotate("", xy=a, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=2.5))
    ax.annotate("", xy=b, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=2.5))
    ax.text(a[0] - 0.15, a[1] + 0.20, r"$v_1$", fontsize=14, color=C_PURPLE)
    ax.text(b[0] + 0.10, b[1] - 0.05, r"$v_2 = 1.6\,v_1$", fontsize=12, color=C_AMBER)
    ax.set_title("Linearly dependent\n"
                 r"span = 1D line (parallelogram collapses)", fontsize=12)
    ax.set_xlim(-1, 4.5); ax.set_ylim(-1.5, 3)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.legend(loc="lower right", fontsize=10)

    fig.suptitle("Linear Independence vs Dependence  -  the geometry of rank",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_linear_dependence")


# ----------------------------------------------------------------------------
# fig3: Eigendecomposition geometric meaning
# ----------------------------------------------------------------------------
def fig3_eigendecomposition() -> None:
    # Symmetric matrix - clean eigen-structure
    A = np.array([[2.0, 0.6],
                  [0.6, 1.0]])
    eigvals, eigvecs = np.linalg.eigh(A)  # ascending order

    # Unit circle and its image under A
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: unit circle + eigenvectors
    ax = axes[0]
    ax.plot(circle[0], circle[1], color=C_GRAY, lw=1.5)
    for i, c in enumerate([C_PURPLE, C_GREEN]):
        v = eigvecs[:, i]
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.text(v[0] * 1.15, v[1] * 1.15,
                rf"$q_{i+1}$, $\lambda={eigvals[i]:.2f}$",
                fontsize=11, color=c, ha="center")
    ax.set_title("Before:  unit circle + eigenvectors of $A$", fontsize=12)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

    # Right: image of circle = ellipse, eigenvectors only stretched
    ax = axes[1]
    ax.plot(ellipse[0], ellipse[1], color=C_BLUE, lw=2)
    for i, c in enumerate([C_PURPLE, C_GREEN]):
        v = eigvecs[:, i]
        Av = A @ v
        # Original eigenvector light dashed
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.0,
                                    linestyle="--", alpha=0.45))
        # Av in solid
        ax.annotate("", xy=Av, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.text(Av[0] * 1.10, Av[1] * 1.10,
                rf"$Aq_{i+1}=\lambda_{i+1}q_{i+1}$",
                fontsize=11, color=c, ha="center")
    ax.set_title("After $A$:  circle $\\to$ ellipse;  eigenvectors only scale",
                 fontsize=12)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

    fig.suptitle(r"Eigendecomposition $A = Q\Lambda Q^\top$:  "
                 r"directions that survive the transformation",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_eigendecomposition")


# ----------------------------------------------------------------------------
# fig4: SVD - rotate, scale, rotate
# ----------------------------------------------------------------------------
def fig4_svd_three_steps() -> None:
    A = np.array([[1.6, 0.8],
                  [0.4, 1.2]])
    U, s, Vt = np.linalg.svd(A)
    Sigma = np.diag(s)

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])

    # Two basis vectors to track orientation
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    steps = [
        (circle, e1, e2, "1.  Input: unit circle"),
        (Vt @ circle, Vt @ e1, Vt @ e2, r"2.  Rotate by $V^\top$"),
        (Sigma @ Vt @ circle, Sigma @ Vt @ e1, Sigma @ Vt @ e2,
         r"3.  Scale by $\Sigma$ (axis-aligned ellipse)"),
        (U @ Sigma @ Vt @ circle, U @ Sigma @ Vt @ e1, U @ Sigma @ Vt @ e2,
         r"4.  Rotate by $U$  =  $A x$"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    colors = [C_PURPLE, C_GREEN]
    for ax, (curve, b1, b2, title) in zip(axes, steps):
        ax.plot(curve[0], curve[1], color=C_BLUE, lw=2)
        for v, c in zip([b1, b2], colors):
            ax.annotate("", xy=v, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=c, lw=2.2))
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

    fig.suptitle(r"SVD geometric pipeline:  $A = U\Sigma V^\top$  "
                 r"=  rotate $\to$ scale $\to$ rotate",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    save(fig, "fig4_svd_three_steps")


# ----------------------------------------------------------------------------
# fig5: Rank deficiency - 2D plane collapses to 1D line
# ----------------------------------------------------------------------------
def fig5_rank_deficiency() -> None:
    # rank-1 matrix: every output lies on a single line through origin
    # A = u v^T, here u = (2, 1), v = (1, 1) so column space = span((2,1))
    u = np.array([2.0, 1.0])
    v = np.array([1.0, 1.0])
    A = np.outer(u, v)

    # Sample input points (a grid in R^2)
    grid = np.linspace(-2, 2, 9)
    XX, YY = np.meshgrid(grid, grid)
    pts = np.stack([XX.ravel(), YY.ravel()])
    out = A @ pts

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Input
    ax = axes[0]
    ax.scatter(pts[0], pts[1], color=C_BLUE, s=22, alpha=0.7)
    # Null space direction: A x = 0  ->  v^T x = 0  ->  x perp v
    null_dir = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    ax.plot([-3 * null_dir[0], 3 * null_dir[0]],
            [-3 * null_dir[1], 3 * null_dir[1]],
            color=C_AMBER, lw=2.5, label=r"null space  (gets killed)")
    ax.set_title(r"Input space $\mathbb{R}^2$  -  full 2D grid", fontsize=12)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.legend(loc="lower right", fontsize=10)

    # Output - all collapse onto one line
    ax = axes[1]
    ax.scatter(out[0], out[1], color=C_PURPLE, s=22, alpha=0.7)
    # Column space line: span(u)
    u_n = u / np.linalg.norm(u)
    ax.plot([-7 * u_n[0], 7 * u_n[0]],
            [-7 * u_n[1], 7 * u_n[1]],
            color=C_GREEN, lw=2.5, label=r"column space  $\mathrm{Col}(A)$")
    ax.set_title(r"Output space:  $A x$  lies on a 1D line  "
                 r"(rank $A = 1$)", fontsize=12)
    ax.set_xlim(-7, 7); ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.legend(loc="lower right", fontsize=10)

    fig.suptitle(r"Rank deficiency:  $A = u v^\top$ collapses $\mathbb{R}^2$ "
                 r"onto $\mathrm{span}(u)$",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_rank_deficiency")


# ----------------------------------------------------------------------------
# fig6: Matrix calculus shape rules
# ----------------------------------------------------------------------------
def fig6_matrix_calculus_shapes() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Matrix calculus shape rules  (numerator layout)",
                 fontsize=14, pad=14)

    rules = [
        # (y, label, formula_size, formula, color)
        (5.5, "scalar / scalar", r"$f: \mathbb{R} \to \mathbb{R}$",
         r"$\dfrac{\partial f}{\partial x} \in \mathbb{R}$", C_BLUE),
        (4.2, "scalar / vector", r"$f: \mathbb{R}^n \to \mathbb{R}$",
         r"$\nabla_x f \in \mathbb{R}^{n}$  (column)", C_PURPLE),
        (2.9, "vector / vector", r"$g: \mathbb{R}^n \to \mathbb{R}^m$",
         r"$J_g = \dfrac{\partial g}{\partial x} \in \mathbb{R}^{m \times n}$", C_GREEN),
        (1.6, "scalar / matrix", r"$f: \mathbb{R}^{m \times n} \to \mathbb{R}$",
         r"$\dfrac{\partial f}{\partial X} \in \mathbb{R}^{m \times n}$", C_AMBER),
    ]

    for y, lbl, sig, shape, c in rules:
        # Coloured strip on the left
        ax.add_patch(Rectangle((0.3, y - 0.45), 0.25, 0.9, color=c))
        ax.text(0.8, y, lbl, fontsize=12, va="center", color=c, weight="bold")
        ax.text(3.3, y, sig, fontsize=12, va="center")
        ax.text(6.6, y, shape, fontsize=12, va="center")

    # Footer with the three workhorse formulas
    ax.text(5.5, 0.4,
            r"Workhorses:    "
            r"$\nabla_x (a^\top x) = a$        "
            r"$\nabla_x (x^\top A x) = (A + A^\top) x$        "
            r"$\dfrac{\partial \ln \det X}{\partial X} = X^{-\top}$",
            fontsize=11, ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f4f6",
                      edgecolor=C_GRAY, lw=0.8))

    # Headers
    ax.text(0.8, 6.4, "kind", fontsize=11, color=C_GRAY, weight="bold")
    ax.text(3.3, 6.4, "function signature", fontsize=11, color=C_GRAY, weight="bold")
    ax.text(6.6, 6.4, "derivative shape", fontsize=11, color=C_GRAY, weight="bold")
    ax.plot([0.3, 10.7], [6.1, 6.1], color=C_GRAY, lw=0.8)

    save(fig, "fig6_matrix_calculus_shapes")


# ----------------------------------------------------------------------------
# fig7: Positive-definite matrix as ellipsoid
# ----------------------------------------------------------------------------
def fig7_positive_definite_ellipse() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: positive definite -> ellipse, both eigenvalues positive
    A_pd = np.array([[2.5, 0.7],
                     [0.7, 1.0]])
    # Right: indefinite -> hyperbola (one eigenvalue negative)
    A_ind = np.array([[1.5, 0.0],
                      [0.0, -1.0]])

    grid = np.linspace(-3, 3, 300)
    XX, YY = np.meshgrid(grid, grid)

    def quad(M):
        return M[0, 0] * XX**2 + (M[0, 1] + M[1, 0]) * XX * YY + M[1, 1] * YY**2

    # PD case
    ax = axes[0]
    Z = quad(A_pd)
    ax.contour(XX, YY, Z, levels=[1.0], colors=[C_BLUE], linewidths=2)
    ax.contourf(XX, YY, Z, levels=[0, 1, 4, 9, 16, 25], cmap="Blues", alpha=0.35)

    eigvals, eigvecs = np.linalg.eigh(A_pd)
    for i, c in enumerate([C_PURPLE, C_GREEN]):
        v = eigvecs[:, i]
        # Semi-axis length on x^T A x = 1 ellipse is 1/sqrt(lambda)
        half = 1.0 / np.sqrt(eigvals[i])
        end = v * half
        ax.annotate("", xy=end, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.text(end[0] * 1.20, end[1] * 1.20,
                rf"$\lambda_{i+1}={eigvals[i]:.2f}$,   "
                rf"axis $=1/\sqrt{{\lambda_{i+1}}}={half:.2f}$",
                fontsize=10, color=c, ha="center")
    ax.set_title(r"Positive definite:  $x^\top A x = 1$ is an ellipse",
                 fontsize=12)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

    # Indefinite case (contrast)
    ax = axes[1]
    Z = quad(A_ind)
    ax.contour(XX, YY, Z, levels=[-1, 1], colors=[C_AMBER, C_BLUE], linewidths=2)
    ax.contourf(XX, YY, Z, levels=np.linspace(-9, 9, 11), cmap="RdBu_r", alpha=0.35)
    ax.text(0, -2.2, r"$x^\top A x = -1$  (saddle)",
            fontsize=11, color=C_AMBER, ha="center")
    ax.text(0, 2.2, r"$x^\top A x = +1$",
            fontsize=11, color=C_BLUE, ha="center")
    ax.set_title(r"Indefinite (eigvals mixed sign):  hyperbolas, no minimum",
                 fontsize=12)
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

    fig.suptitle(r"Why ML loves PD matrices:  $x^\top A x$ is a bowl  "
                 r"$\Rightarrow$ unique minimum",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_positive_definite_ellipse")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for ML Math Derivations Part 02 ...")
    print(f"  EN -> {EN_DIR}")
    print(f"  ZH -> {ZH_DIR}")
    fig1_vector_space_subspace()
    fig2_linear_dependence()
    fig3_eigendecomposition()
    fig4_svd_three_steps()
    fig5_rank_deficiency()
    fig6_matrix_calculus_shapes()
    fig7_positive_definite_ellipse()
    print("Done.")


if __name__ == "__main__":
    main()
