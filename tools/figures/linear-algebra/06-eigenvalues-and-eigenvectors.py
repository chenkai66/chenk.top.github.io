"""
Figure generation script for Linear Algebra Chapter 06: Eigenvalues and Eigenvectors.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in 3Blue1Brown style.

Figures:
    fig1_eigen_definition       Definition: a vector that stays on its span
                                after the transformation. One eigenvector
                                survives (only scaled), one ordinary vector
                                gets knocked off its span.
    fig2_eigenvectors_2x2       Eigenvectors of a 2x2 matrix shown in the
                                original grid + transformed grid: the
                                eigen-axes deform but the lines through
                                eigenvectors stay fixed.
    fig3_eigenvalue_scaling     Geometric meaning of eigenvalues: same
                                eigenvector, three different eigenvalues
                                (lambda = 2, 1, -0.5) -- stretch, identity,
                                flip-and-shrink.
    fig4_diagonalization        Diagonalization as a change of basis:
                                P^{-1} -> D -> P. In the eigenbasis, the
                                transformation is just axis-aligned scaling.
    fig5_complex_eigenvalues    Rotation matrix has no real eigenvectors:
                                every direction gets rotated, so the iterates
                                of a vector trace a circle.
    fig6_power_iteration        Power iteration: repeatedly applying A and
                                normalizing converges to the dominant
                                eigenvector, regardless of starting point.
    fig7_pca_preview            PCA preview: principal components are the
                                eigenvectors of the covariance matrix; they
                                point along the spread of a 2D point cloud.

Usage:
    python3 scripts/figures/linear-algebra/06-eigenvalues-and-eigenvectors.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary - input vectors / data
C_PURPLE = "#7c3aed"   # secondary - second eigenvector
C_GREEN = "#10b981"    # accent - eigenvector / surviving span
C_AMBER = "#f59e0b"    # warning - rotated / non-eigen
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "06-eigenvalues-and-eigenvectors"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "06-特征值与特征向量"
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

def arrow(ax, tail, head, color, lw=2.6, alpha=1.0, mutation=18, zorder=5):
    """Draw a clean arrow from tail to head."""
    a = FancyArrowPatch(
        tail, head,
        arrowstyle="-|>",
        mutation_scale=mutation,
        color=color,
        lw=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(a)
    return a


def style_axes(ax, lim, title=None, title_size=12):
    """Apply a clean equal-aspect square frame with thin axis lines."""
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, lw=0.6, zorder=0)
    ax.axvline(0, color=C_GRAY, lw=0.6, zorder=0)
    ax.grid(True, color=C_LIGHT, lw=0.6, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C_LIGHT)
    if title is not None:
        ax.set_title(title, fontsize=title_size, color=C_DARK, pad=8)


def draw_grid(ax, A, lim, color, lw=0.7, n=7, zorder=1):
    """Draw a transformed grid: image of the integer grid under matrix A."""
    pts = np.arange(-n, n + 1)
    for x in pts:
        line = np.array([[x, x], [-n, n]])  # vertical line x=const
        out = A @ line
        ax.plot(out[0], out[1], color=color, lw=lw, alpha=0.7, zorder=zorder)
    for y in pts:
        line = np.array([[-n, n], [y, y]])
        out = A @ line
        ax.plot(out[0], out[1], color=color, lw=lw, alpha=0.7, zorder=zorder)


def draw_span(ax, v, lim, color, lw=1.4, ls="--", alpha=0.85, zorder=2):
    """Draw the line through the origin in the direction of vector v."""
    v = np.asarray(v, dtype=float)
    n = v / np.linalg.norm(v)
    p1 = -3 * lim * n
    p2 = 3 * lim * n
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)


# ---------------------------------------------------------------------------
# Figure 1 -- Eigen definition: stays on its span vs. knocked off
# ---------------------------------------------------------------------------

def fig1_eigen_definition() -> None:
    A = np.array([[2.0, 1.0], [0.0, 3.0]])  # eigenvalues 2 and 3
    # eigenvector for lambda=3: solve (A - 3I)v = 0 -> [-1,1]v=0 -> v=(1,1)
    v_eigen = np.array([1.0, 1.0])
    v_other = np.array([2.0, -0.5])

    Av_eigen = A @ v_eigen          # = 3 * v_eigen
    Av_other = A @ v_other

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    lim = 5.5

    # Left: before transformation
    ax = axes[0]
    style_axes(ax, lim, title="Before  $A$  is applied")
    draw_span(ax, v_eigen, lim, C_GREEN)
    draw_span(ax, v_other, lim, C_AMBER)
    arrow(ax, (0, 0), v_eigen, C_GREEN)
    arrow(ax, (0, 0), v_other, C_AMBER)
    ax.text(v_eigen[0] + 0.2, v_eigen[1] + 0.2, r"$\vec{v}$",
            color=C_GREEN, fontsize=14, fontweight="bold")
    ax.text(v_other[0] + 0.2, v_other[1] - 0.5, r"$\vec{u}$",
            color=C_AMBER, fontsize=14, fontweight="bold")
    ax.text(0, -lim + 0.4,
            "two vectors and the lines they span",
            ha="center", color=C_DARK, fontsize=10)

    # Right: after transformation
    ax = axes[1]
    style_axes(ax, lim, title=r"After  $A$  is applied   ($\lambda = 3$)")
    # original span of v_eigen still drawn to show the eigenvector stayed on it
    draw_span(ax, v_eigen, lim, C_GREEN)
    draw_span(ax, v_other, lim, C_AMBER, alpha=0.35)
    arrow(ax, (0, 0), Av_eigen, C_GREEN, lw=3.0)
    arrow(ax, (0, 0), Av_other, C_AMBER, lw=2.6)
    ax.text(Av_eigen[0] + 0.2, Av_eigen[1] + 0.2,
            r"$A\vec{v} = 3\vec{v}$",
            color=C_GREEN, fontsize=13, fontweight="bold")
    ax.text(Av_other[0] - 0.4, Av_other[1] - 0.6, r"$A\vec{u}$",
            color=C_AMBER, fontsize=13, fontweight="bold")
    ax.text(0, -lim + 0.4,
            r"$\vec{v}$ stayed on its span;  $\vec{u}$ was knocked off",
            ha="center", color=C_DARK, fontsize=10)

    fig.suptitle("Eigenvector  =  a vector whose span survives the transformation",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_eigen_definition")


# ---------------------------------------------------------------------------
# Figure 2 -- Eigenvectors of a 2x2 matrix in original + transformed grid
# ---------------------------------------------------------------------------

def fig2_eigenvectors_2x2() -> None:
    A = np.array([[3.0, 1.0], [0.0, 2.0]])
    # eigenvalues: 3 and 2; eigenvectors: (1,0) for 3, (1,-1) for 2
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, -1.0]) / np.sqrt(2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    lim = 4.2

    # Left: original
    ax = axes[0]
    style_axes(ax, lim, title="Original space")
    draw_grid(ax, np.eye(2), lim, C_LIGHT, n=6)
    draw_span(ax, v1, lim, C_GREEN)
    draw_span(ax, v2, lim, C_PURPLE)
    arrow(ax, (0, 0), 2 * v1, C_GREEN)
    arrow(ax, (0, 0), 2 * v2, C_PURPLE)
    ax.text(2.1, 0.25, r"$\vec{v}_1$", color=C_GREEN, fontsize=13, fontweight="bold")
    ax.text(1.55, -1.95, r"$\vec{v}_2$", color=C_PURPLE, fontsize=13, fontweight="bold")
    ax.text(0, -lim + 0.3,
            "two eigen-directions of $A$",
            ha="center", color=C_DARK, fontsize=10)

    # Right: after transformation by A
    ax = axes[1]
    style_axes(ax, lim, title=r"After applying  $A=[\,3,1\,;\,0,2\,]$")
    draw_grid(ax, A, lim, C_BLUE, lw=0.8, n=4)
    draw_span(ax, v1, lim, C_GREEN)
    draw_span(ax, v2, lim, C_PURPLE)
    Av1 = A @ (2 * v1)   # eigenvalue 3 -> length 6
    Av2 = A @ (2 * v2)   # eigenvalue 2 -> length 4
    arrow(ax, (0, 0), Av1, C_GREEN, lw=3.0)
    arrow(ax, (0, 0), Av2, C_PURPLE, lw=3.0)
    ax.text(2.6, 0.3, r"$A\vec{v}_1 = 3\vec{v}_1$",
            color=C_GREEN, fontsize=12, fontweight="bold")
    ax.text(0.3, -3.4, r"$A\vec{v}_2 = 2\vec{v}_2$",
            color=C_PURPLE, fontsize=12, fontweight="bold")
    ax.text(0, -lim + 0.3,
            "grid is sheared, but eigen-lines stay put",
            ha="center", color=C_DARK, fontsize=10)

    fig.suptitle("Eigenvectors of a $2\\times 2$ matrix:  the lines that the grid pivots around",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_eigenvectors_2x2")


# ---------------------------------------------------------------------------
# Figure 3 -- Eigenvalue meaning: scaling factor (three values side by side)
# ---------------------------------------------------------------------------

def fig3_eigenvalue_scaling() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    lim = 3.3
    v = np.array([1.0, 0.6])
    v = v / np.linalg.norm(v) * 1.5  # length 1.5

    cases = [
        (2.0,  "$\\lambda = 2$\n(stretch)",        C_GREEN),
        (1.0,  "$\\lambda = 1$\n(unchanged)",      C_BLUE),
        (-0.5, "$\\lambda = -0.5$\n(flip & shrink)", C_AMBER),
    ]
    for ax, (lam, label, col) in zip(axes, cases):
        style_axes(ax, lim, title=label)
        draw_span(ax, v, lim, C_GRAY, lw=1.0, alpha=0.6)
        # original vector (light)
        arrow(ax, (0, 0), v, C_DARK, lw=1.8, alpha=0.45)
        ax.text(v[0] + 0.05, v[1] + 0.18, r"$\vec{v}$",
                color=C_DARK, fontsize=12, alpha=0.7)
        # scaled vector
        out = lam * v
        arrow(ax, (0, 0), out, col, lw=3.0)
        ax.text(out[0] + 0.05, out[1] + 0.18, r"$\lambda\vec{v}$",
                color=col, fontsize=13, fontweight="bold")

    fig.suptitle("The eigenvalue is the scaling factor along the eigen-line",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_eigenvalue_scaling")


# ---------------------------------------------------------------------------
# Figure 4 -- Diagonalization: change to eigenbasis, scale, change back
# ---------------------------------------------------------------------------

def fig4_diagonalization() -> None:
    A = np.array([[2.0, 1.0], [1.0, 2.0]])      # eigenvalues 3, 1
    # eigenvectors: (1,1)/sqrt2 (lambda=3), (1,-1)/sqrt2 (lambda=1)
    P = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    D = np.diag([3.0, 1.0])
    Pinv = P.T  # P is orthogonal

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    lim = 3.8

    # Stage 1: P^{-1}  -- enter the eigenbasis
    ax = axes[0]
    style_axes(ax, lim, title=r"Step 1:  $P^{-1}$  enter the eigenbasis")
    draw_grid(ax, np.eye(2), lim, C_LIGHT, n=5)
    draw_grid(ax, Pinv, lim, C_BLUE, lw=0.8, n=4)
    arrow(ax, (0, 0), Pinv @ np.array([1.5, 0]), C_GREEN, lw=2.6)
    arrow(ax, (0, 0), Pinv @ np.array([0, 1.5]), C_PURPLE, lw=2.6)
    ax.text(0, -lim + 0.3,
            "rotate so eigen-axes align with x and y",
            ha="center", color=C_DARK, fontsize=10)

    # Stage 2: D  -- pure axis-aligned scaling
    ax = axes[1]
    style_axes(ax, lim, title=r"Step 2:  $D = \mathrm{diag}(3,1)$  scale")
    draw_grid(ax, np.eye(2), lim, C_LIGHT, n=5)
    draw_grid(ax, D, lim, C_BLUE, lw=0.8, n=2)
    arrow(ax, (0, 0), np.array([3 * 1.5, 0]), C_GREEN, lw=2.6)
    arrow(ax, (0, 0), np.array([0, 1 * 1.5]), C_PURPLE, lw=2.6)
    ax.text(0, -lim + 0.3,
            r"x-axis scaled by 3,  y-axis by 1",
            ha="center", color=C_DARK, fontsize=10)

    # Stage 3: P  -- back to original basis
    ax = axes[2]
    style_axes(ax, lim, title=r"Step 3:  $P$  back to original basis")
    draw_grid(ax, np.eye(2), lim, C_LIGHT, n=5)
    draw_grid(ax, A, lim, C_BLUE, lw=0.8, n=3)
    arrow(ax, (0, 0), A @ np.array([1.5, 0]) * 0.7, C_GREEN, lw=2.6)
    arrow(ax, (0, 0), A @ np.array([0, 1.5]) * 0.7, C_PURPLE, lw=2.6)
    ax.text(0, -lim + 0.3,
            r"$A = P\,D\,P^{-1}$ rebuilt",
            ha="center", color=C_DARK, fontsize=10)

    fig.suptitle("Diagonalization  =  change basis  $\\to$  scale  $\\to$  change back",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_diagonalization")


# ---------------------------------------------------------------------------
# Figure 5 -- Complex eigenvalues = rotation, no real eigenvectors
# ---------------------------------------------------------------------------

def fig5_complex_eigenvalues() -> None:
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    lim = 2.2

    # Left: try several candidate vectors -- all get rotated off their span
    ax = axes[0]
    style_axes(ax, lim, title="No real direction survives a rotation")
    angles = np.linspace(0, np.pi, 6, endpoint=False)
    for a in angles:
        v = 1.5 * np.array([np.cos(a), np.sin(a)])
        Rv = R @ v
        draw_span(ax, v, lim, C_GRAY, lw=0.8, alpha=0.45)
        arrow(ax, (0, 0), v, C_BLUE, lw=1.8, alpha=0.55)
        arrow(ax, (0, 0), Rv, C_AMBER, lw=2.0)
    ax.text(0, -lim + 0.2,
            "every vector (blue) lands off its span (amber)",
            ha="center", color=C_DARK, fontsize=10)

    # Right: iterating R^n on a single vector traces a circle
    ax = axes[1]
    style_axes(ax, lim,
               title=r"Iterating $R$  $\Rightarrow$  circular orbit")
    # full circle
    cs = np.linspace(0, 2 * np.pi, 200)
    ax.plot(1.5 * np.cos(cs), 1.5 * np.sin(cs),
            color=C_LIGHT, lw=1.5, zorder=1)
    v = np.array([1.5, 0.0])
    pts = [v]
    for _ in range(11):
        pts.append(R @ pts[-1])
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "o-",
            color=C_PURPLE, lw=1.5, ms=6, zorder=3)
    arrow(ax, (0, 0), pts[0], C_BLUE, lw=2.4)
    arrow(ax, (0, 0), pts[-1], C_GREEN, lw=2.4)
    ax.text(pts[0, 0] + 0.05, pts[0, 1] - 0.25, r"$\vec{v}_0$",
            color=C_BLUE, fontsize=12, fontweight="bold")
    ax.text(pts[-1, 0] - 0.6, pts[-1, 1] + 0.15, r"$R^{11}\vec{v}_0$",
            color=C_GREEN, fontsize=12, fontweight="bold")
    ax.text(0, -lim + 0.2,
            r"eigenvalues escape to $\mathbb{C}$:  $\lambda = e^{\pm i\theta}$",
            ha="center", color=C_DARK, fontsize=10)

    fig.suptitle("Complex eigenvalues  =  rotation  (no fixed real direction)",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_complex_eigenvalues")


# ---------------------------------------------------------------------------
# Figure 6 -- Power iteration converges to the dominant eigenvector
# ---------------------------------------------------------------------------

def fig6_power_iteration() -> None:
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    # dominant eigenvalue ~ 3.618, eigenvector roughly (1, 1.618)
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigvals))
    dominant = eigvecs[:, idx]
    if dominant[1] < 0:
        dominant = -dominant
    dominant = dominant / np.linalg.norm(dominant)

    starts = [
        np.array([1.0, 0.05]),
        np.array([-0.7, 0.7]),
        np.array([0.3, -1.0]),
    ]
    colors = [C_BLUE, C_AMBER, C_PURPLE]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    lim = 1.35
    style_axes(ax, lim,
               title="Power iteration:  $\\vec{v}_{k+1} = A\\vec{v}_k\\,/\\,\\|A\\vec{v}_k\\|$",
               title_size=12)

    # dominant eigen-line
    draw_span(ax, dominant, lim, C_GREEN, lw=1.6)
    arrow(ax, (0, 0), 1.15 * dominant, C_GREEN, lw=3.0)
    ax.text(dominant[0] * 1.18 + 0.04, dominant[1] * 1.18 + 0.02,
            "dominant\neigenvector",
            color=C_GREEN, fontsize=10, fontweight="bold")

    for v0, col in zip(starts, colors):
        v = v0 / np.linalg.norm(v0)
        traj = [v.copy()]
        for _ in range(10):
            v = A @ v
            v = v / np.linalg.norm(v)
            traj.append(v.copy())
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], "o-",
                color=col, lw=1.4, ms=5, alpha=0.9, zorder=4)
        arrow(ax, (0, 0), traj[0], col, lw=2.0, alpha=0.85)

    # unit circle for context
    cs = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(cs), np.sin(cs), color=C_LIGHT, lw=1.0, zorder=1)

    ax.text(0, -lim + 0.08,
            "any starting direction snaps onto the dominant eigen-line",
            ha="center", color=C_DARK, fontsize=10)

    fig.tight_layout()
    save(fig, "fig6_power_iteration")


# ---------------------------------------------------------------------------
# Figure 7 -- PCA preview: principal components are eigenvectors of cov
# ---------------------------------------------------------------------------

def fig7_pca_preview() -> None:
    rng = np.random.default_rng(7)
    n = 250
    # latent: stretched along the line y = 0.6 x
    base = rng.normal(size=(2, n)) * np.array([[1.8], [0.5]])
    theta = np.deg2rad(28)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    pts = R @ base
    pts = pts - pts.mean(axis=1, keepdims=True)

    cov = np.cov(pts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; reverse to descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    lim = 4.5
    style_axes(ax, lim,
               title="Principal components  =  eigenvectors of the covariance matrix",
               title_size=12)

    ax.scatter(pts[0], pts[1], s=14, color=C_BLUE, alpha=0.55,
               edgecolor="none", zorder=3)

    # principal axes scaled by sqrt(eigenvalue) * 2
    scales = 2.2 * np.sqrt(eigvals)
    pc1 = eigvecs[:, 0] * scales[0]
    pc2 = eigvecs[:, 1] * scales[1]

    # show both directions of each axis
    for v, col, lab in [(pc1, C_GREEN, "PC$_1$  (largest variance)"),
                        (pc2, C_AMBER, "PC$_2$")]:
        ax.plot([-v[0], v[0]], [-v[1], v[1]],
                color=col, lw=1.4, ls="--", alpha=0.6, zorder=4)
        arrow(ax, (0, 0), v, col, lw=3.0, zorder=6)

    ax.text(pc1[0] * 1.05 + 0.1, pc1[1] * 1.05 + 0.1,
            "PC$_1$", color=C_GREEN, fontsize=13, fontweight="bold")
    ax.text(pc2[0] * 1.15 - 0.2, pc2[1] * 1.15 + 0.1,
            "PC$_2$", color=C_AMBER, fontsize=13, fontweight="bold")

    ax.text(0, -lim + 0.25,
            r"$\mathrm{Cov}(X)\,\vec{v}_i = \lambda_i\,\vec{v}_i$    "
            "(eigenvalue $=$ variance along that axis)",
            ha="center", color=C_DARK, fontsize=10)

    fig.tight_layout()
    save(fig, "fig7_pca_preview")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_eigen_definition()
    fig2_eigenvectors_2x2()
    fig3_eigenvalue_scaling()
    fig4_diagonalization()
    fig5_complex_eigenvalues()
    fig6_power_iteration()
    fig7_pca_preview()
    print("All 7 figures saved to:")
    print(f"  {EN_DIR}")
    print(f"  {ZH_DIR}")


if __name__ == "__main__":
    main()
