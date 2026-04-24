"""
Figure generation script for Linear Algebra Chapter 08:
Symmetric Matrices and Quadratic Forms.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches one specific idea cleanly, in a 3Blue1Brown style.

Figures
-------
fig1_symmetric_structure
    A symmetric matrix as a heat map, highlighting that the diagonal is free
    and the off-diagonal pairs (i,j) and (j,i) are mirror images across the
    main diagonal. A non-symmetric matrix is shown alongside for contrast.

fig2_spectral_theorem
    A symmetric matrix's eigenvectors form an orthogonal frame.  Left panel:
    a symmetric A acts on the unit circle, producing an axis-aligned ellipse
    in the eigenvector frame.  Right panel: how A pulls a generic vector x
    apart along the orthogonal eigenvectors q1, q2.

fig3_quadratic_signature
    Three contour plots side by side showing positive definite (bowl),
    indefinite (saddle), and negative definite (hill).  The eigenvector
    arrows are overlaid to make the connection between sign of eigenvalues
    and shape explicit.

fig4_definiteness_zoo
    Four-panel taxonomy: positive definite, positive semidefinite (trough),
    indefinite (saddle), negative definite (hill).  A small badge on each
    panel reports the eigenvalues to make the rule "signs of eigenvalues
    determine the shape" one glance away.

fig5_principal_axes
    The level set x^T A x = 1 is an ellipse whose principal axes are the
    eigenvectors of A.  Semi-axis lengths 1/sqrt(lambda_i) are annotated.
    A second panel shows the standard form after rotating into the eigen-
    basis: the cross-term vanishes.

fig6_rayleigh_quotient
    Visualises R(x) = x^T A x / x^T x as the value sampled on the unit
    circle.  The maximum equals lambda_max and is achieved at the top
    eigenvector; the minimum equals lambda_min.  A polar plot makes the
    extremal directions obvious.

fig7_svd_preview
    Preview of SVD as the natural extension of the spectral theorem to
    non-square / non-symmetric matrices.  Shows: unit circle -> ellipse
    via A, with right singular vectors v_i (input axes) and left singular
    vectors u_i (output axes), plus the sigma_i scaling.

Usage
-----
    python3 scripts/figures/linear-algebra/08-symmetric-matrices-and-quadratic-forms.py

Outputs are written to BOTH the EN and ZH article asset folders so that
markdown references stay in sync.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

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
C_RED = "#dc2626"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "08-symmetric-matrices-and-quadratic-forms"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "08-对称矩阵与二次型"
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
# Common helpers
# ---------------------------------------------------------------------------
def _setup_axes(ax, lim=(-3.2, 3.2), aspect=True, grid=True):
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    if aspect:
        ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, lw=0.6, zorder=0)
    ax.axvline(0, color=C_GRAY, lw=0.6, zorder=0)
    if grid:
        ax.grid(True, alpha=0.25, lw=0.6)
    else:
        ax.grid(False)
    ax.tick_params(labelsize=8, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_color(C_GRAY)
        spine.set_linewidth(0.6)


def _arrow(ax, start, end, color=C_BLUE, lw=2.4, mut=14, zorder=5):
    ax.add_patch(FancyArrowPatch(tuple(start), tuple(end),
                                 arrowstyle="-|>", mutation_scale=mut,
                                 color=color, lw=lw, zorder=zorder))


def _heatmap_matrix(ax, M, title, cmap_pos=C_BLUE, cmap_neg=C_AMBER,
                    show_values=True):
    """Render a small matrix as a colored grid with values overlaid."""
    n = M.shape[0]
    vmax = np.max(np.abs(M)) + 1e-9
    for i in range(n):
        for j in range(n):
            val = M[i, j]
            color = cmap_pos if val >= 0 else cmap_neg
            alpha = min(1.0, 0.18 + 0.65 * abs(val) / vmax)
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                   facecolor=color, alpha=alpha,
                                   edgecolor="white", lw=2))
            if show_values:
                ax.text(j + 0.5, n - 1 - i + 0.5, f"{val:g}",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, color=C_DARK,
                 fontweight="bold", pad=10)


# ---------------------------------------------------------------------------
# Fig 1: Symmetric structure (diagonal + mirrored off-diagonal)
# ---------------------------------------------------------------------------
def fig1_symmetric_structure():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Left: a symmetric matrix
    S = np.array([[3, 1, -2],
                  [1, 4,  0],
                  [-2, 0, 5]], dtype=float)
    _heatmap_matrix(axes[0], S, "Symmetric:  $A = A^{T}$")

    # Annotate mirror pairs
    n = 3
    pairs = [((0, 1), (1, 0)), ((0, 2), (2, 0)), ((1, 2), (2, 1))]
    for (i1, j1), (i2, j2) in pairs:
        x1, y1 = j1 + 0.5, n - 1 - i1 + 0.5
        x2, y2 = j2 + 0.5, n - 1 - i2 + 0.5
        axes[0].annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="<->", color=C_PURPLE,
                                         lw=1.4, alpha=0.7,
                                         connectionstyle="arc3,rad=0.25"))

    # Diagonal callout
    axes[0].plot([0, n], [n, 0], color=C_DARK, lw=1.2,
                 linestyle="--", alpha=0.5)
    axes[0].text(n + 0.05, -0.25, "diagonal is free,\noff-diagonals mirror",
                 fontsize=10, color=C_DARK, ha="right", va="top")

    # Right: non-symmetric
    N = np.array([[3, 1, -2],
                  [4, 4,  0],
                  [1, -3, 5]], dtype=float)
    _heatmap_matrix(axes[1], N, "Not symmetric:  $A \\ne A^{T}$",
                    cmap_pos=C_GRAY, cmap_neg=C_GRAY)
    axes[1].text(n / 2, -0.4, "rotation hidden inside;\neigenvalues may be complex",
                 fontsize=10, color=C_DARK, ha="center", va="top")

    fig.suptitle("Symmetric matrix structure",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    save(fig, "fig1_symmetric_structure")


# ---------------------------------------------------------------------------
# Fig 2: Spectral theorem - orthogonal eigenvectors of a symmetric matrix
# ---------------------------------------------------------------------------
def fig2_spectral_theorem():
    A = np.array([[3.0, 1.0],
                  [1.0, 2.0]])
    eigvals, eigvecs = np.linalg.eigh(A)
    # numpy returns ascending order; flip so eigvecs[:,0] = top eigenvector
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # Left: A maps unit circle to ellipse aligned with eigenvectors
    ax = axes[0]
    _setup_axes(ax, lim=(-3.6, 3.6))
    theta = np.linspace(0, 2 * np.pi, 300)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle
    ax.plot(circle[0], circle[1], color=C_GRAY, lw=1.5,
            linestyle="--", label="unit circle")
    ax.plot(ellipse[0], ellipse[1], color=C_BLUE, lw=2.4,
            label="image $A\\vec{x}$")
    # eigenvector axes scaled by eigenvalues
    for k, color in [(0, C_PURPLE), (1, C_GREEN)]:
        v = eigvecs[:, k]
        lam = eigvals[k]
        _arrow(ax, (0, 0), v, color=color, lw=2.4)
        _arrow(ax, (0, 0), lam * v, color=color, lw=2.0)
        ax.text(1.08 * lam * v[0], 1.08 * lam * v[1],
                f"$\\lambda_{k+1}={lam:.2f}$",
                color=color, fontsize=11, fontweight="bold",
                ha="left" if v[0] >= 0 else "right")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("Symmetric $A$ stretches along orthogonal eigenvectors",
                 fontsize=11.5, color=C_DARK, fontweight="bold", pad=8)

    # Right: spectral decomposition of a vector x = c1 q1 + c2 q2
    ax = axes[1]
    _setup_axes(ax, lim=(-3.6, 3.6))
    q1, q2 = eigvecs[:, 0], eigvecs[:, 1]
    x = 1.6 * q1 + 1.1 * q2
    Ax = A @ x

    # show q1, q2 frame
    _arrow(ax, (0, 0), 2 * q1, color=C_PURPLE, lw=1.6)
    _arrow(ax, (0, 0), 2 * q2, color=C_GREEN, lw=1.6)
    ax.text(2.1 * q1[0], 2.1 * q1[1], "$\\vec{q}_1$",
            color=C_PURPLE, fontsize=12, fontweight="bold")
    ax.text(2.1 * q2[0], 2.1 * q2[1], "$\\vec{q}_2$",
            color=C_GREEN, fontsize=12, fontweight="bold")

    _arrow(ax, (0, 0), x, color=C_DARK, lw=2.2)
    ax.text(x[0] + 0.1, x[1] + 0.1, "$\\vec{x}$",
            color=C_DARK, fontsize=12, fontweight="bold")
    _arrow(ax, (0, 0), Ax, color=C_BLUE, lw=2.4)
    ax.text(Ax[0] + 0.1, Ax[1] + 0.1, "$A\\vec{x}$",
            color=C_BLUE, fontsize=12, fontweight="bold")

    # dotted projections
    c1 = float(x @ q1)
    c2 = float(x @ q2)
    p1 = c1 * q1
    p2 = c2 * q2
    ax.plot([0, p1[0]], [0, p1[1]], color=C_PURPLE, lw=1.0, ls=":")
    ax.plot([p1[0], x[0]], [p1[1], x[1]], color=C_GREEN, lw=1.0, ls=":")

    ax.text(-3.4, -3.3,
            "$A\\vec{x} = \\lambda_1 c_1 \\vec{q}_1 + \\lambda_2 c_2 \\vec{q}_2$",
            fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_GRAY))
    ax.set_title("Action of $A$ in the eigenbasis",
                 fontsize=11.5, color=C_DARK, fontweight="bold", pad=8)

    fig.suptitle("Spectral theorem:  $A = Q\\Lambda Q^{T}$",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    save(fig, "fig2_spectral_theorem")


# ---------------------------------------------------------------------------
# Fig 3: Quadratic form signatures (bowl / saddle / hill) - contours
# ---------------------------------------------------------------------------
def _quad_contour(ax, A, title, cmap, levels=None, show_axes=True):
    g = np.linspace(-2.5, 2.5, 240)
    X, Y = np.meshgrid(g, g)
    Z = A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2
    if levels is None:
        zmax = np.max(np.abs(Z))
        levels = np.linspace(-zmax, zmax, 21)
    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.85)
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.5,
               alpha=0.6)
    eigvals, eigvecs = np.linalg.eigh(A)
    for k in range(2):
        v = 1.7 * eigvecs[:, k]
        _arrow(ax, -v, v, color=C_DARK, lw=1.6, mut=10)
    _setup_axes(ax, lim=(-2.5, 2.5))
    ax.set_title(title, fontsize=11, color=C_DARK,
                 fontweight="bold", pad=8)
    return cs


def fig3_quadratic_signature():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # Positive definite (bowl)
    A_pd = np.array([[2.0, 0.6], [0.6, 1.0]])
    _quad_contour(axes[0], A_pd,
                  "Positive definite (bowl)\n$\\lambda_1>0,\\ \\lambda_2>0$",
                  cmap="Blues")

    # Indefinite (saddle)
    A_in = np.array([[1.5, 0.0], [0.0, -1.5]])
    _quad_contour(axes[1], A_in,
                  "Indefinite (saddle)\n$\\lambda_1>0,\\ \\lambda_2<0$",
                  cmap="PuOr")

    # Negative definite (hill)
    A_nd = np.array([[-2.0, -0.6], [-0.6, -1.0]])
    _quad_contour(axes[2], A_nd,
                  "Negative definite (hill)\n$\\lambda_1<0,\\ \\lambda_2<0$",
                  cmap="Oranges_r")

    fig.suptitle("Quadratic form $Q(\\vec{x}) = \\vec{x}^{T} A \\vec{x}$"
                 ":  signs of eigenvalues set the shape",
                 fontsize=13.5, color=C_DARK, fontweight="bold", y=1.02)
    save(fig, "fig3_quadratic_signature")


# ---------------------------------------------------------------------------
# Fig 4: Definiteness zoo - 4 panels with eigenvalue badges
# ---------------------------------------------------------------------------
def fig4_definiteness_zoo():
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    cases = [
        (np.array([[2.0, 0.5], [0.5, 1.0]]),
         "Positive definite",
         "every direction goes up",
         "Blues"),
        (np.array([[2.0, 0.0], [0.0, 0.0]]),
         "Positive semidefinite",
         "flat along one direction",
         "Greens"),
        (np.array([[1.5, 0.0], [0.0, -1.5]]),
         "Indefinite (saddle)",
         "up one way, down another",
         "PuOr"),
        (np.array([[-2.0, -0.4], [-0.4, -1.0]]),
         "Negative definite",
         "every direction goes down",
         "Oranges_r"),
    ]

    for ax, (A, name, sub, cmap) in zip(axes.ravel(), cases):
        eigvals = np.linalg.eigvalsh(A)
        _quad_contour(ax, A, f"{name}\n{sub}", cmap=cmap)
        badge = (f"$\\lambda_1 = {eigvals[1]:+.2f}$\n"
                 f"$\\lambda_2 = {eigvals[0]:+.2f}$")
        ax.text(0.97, 0.03, badge, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="white", edgecolor=C_GRAY, alpha=0.95))

    fig.suptitle("The four kinds of symmetric matrix",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig4_definiteness_zoo")


# ---------------------------------------------------------------------------
# Fig 5: Principal axes of an ellipse = eigenvectors
# ---------------------------------------------------------------------------
def fig5_principal_axes():
    A = np.array([[2.5, 1.2],
                  [1.2, 1.5]])
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # Left: ellipse x^T A x = 1 in original coords
    ax = axes[0]
    _setup_axes(ax, lim=(-2.0, 2.0))

    # Sample ellipse points: x = R^{-1/2} unit circle in eigenbasis
    theta = np.linspace(0, 2 * np.pi, 300)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    # x^T A x = 1 -> in eigenbasis, lambda_i y_i^2 = 1 -> y_i = cos/sqrt(lam)
    Y = np.vstack([np.cos(theta) / np.sqrt(eigvals[0]),
                   np.sin(theta) / np.sqrt(eigvals[1])])
    X = eigvecs @ Y

    # Light contours for context
    g = np.linspace(-2, 2, 200)
    Xg, Yg = np.meshgrid(g, g)
    Z = A[0, 0] * Xg**2 + 2 * A[0, 1] * Xg * Yg + A[1, 1] * Yg**2
    ax.contour(Xg, Yg, Z, levels=[0.25, 0.5, 1.0, 1.5, 2.0],
               colors=[C_GRAY], linewidths=0.6, alpha=0.5)

    ax.plot(X[0], X[1], color=C_BLUE, lw=2.6,
            label="$\\vec{x}^{T} A \\vec{x} = 1$")

    # Principal axes (eigenvectors)
    for k, color, label in [(0, C_PURPLE, "$\\vec{q}_1$"),
                            (1, C_GREEN, "$\\vec{q}_2$")]:
        v = eigvecs[:, k] / np.sqrt(eigvals[k])
        _arrow(ax, -v, v, color=color, lw=2.0)
        ax.text(1.15 * v[0], 1.15 * v[1],
                f"{label}\n$1/\\sqrt{{\\lambda_{k+1}}}={1/np.sqrt(eigvals[k]):.2f}$",
                color=color, fontsize=10, fontweight="bold")

    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Original coords:  ellipse with tilted axes",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)

    # Right: same ellipse in the eigenbasis - axis-aligned standard form
    ax = axes[1]
    _setup_axes(ax, lim=(-2.0, 2.0))
    ax.plot(Y[0], Y[1], color=C_BLUE, lw=2.6,
            label=f"$\\lambda_1 y_1^2 + \\lambda_2 y_2^2 = 1$")
    # axes
    a1 = 1 / np.sqrt(eigvals[0])
    a2 = 1 / np.sqrt(eigvals[1])
    _arrow(ax, (-a1, 0), (a1, 0), color=C_PURPLE, lw=2.0)
    _arrow(ax, (0, -a2), (0, a2), color=C_GREEN, lw=2.0)
    ax.text(a1 + 0.05, 0.05, f"$1/\\sqrt{{\\lambda_1}}={a1:.2f}$",
            color=C_PURPLE, fontsize=10, fontweight="bold")
    ax.text(0.05, a2 + 0.05, f"$1/\\sqrt{{\\lambda_2}}={a2:.2f}$",
            color=C_GREEN, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Eigenbasis $\\vec{y} = Q^{T}\\vec{x}$:  cross-term gone",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)

    fig.suptitle("Principal axes of an ellipse $=$ eigenvectors of $A$",
                 fontsize=13.5, color=C_DARK, fontweight="bold", y=1.02)
    save(fig, "fig5_principal_axes")


# ---------------------------------------------------------------------------
# Fig 6: Rayleigh quotient on the unit circle
# ---------------------------------------------------------------------------
def fig6_rayleigh_quotient():
    A = np.array([[3.0, 1.0],
                  [1.0, 2.0]])
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, 400)
    x = np.vstack([np.cos(theta), np.sin(theta)])
    R = np.einsum("ij,ji->i", x.T @ A, x)  # x^T A x for unit x

    fig = plt.figure(figsize=(13, 5.6))

    # Left: x^T A x as a function of angle
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.degrees(theta), R, color=C_BLUE, lw=2.4)
    ax1.axhline(eigvals[0], color=C_PURPLE, lw=1.5, ls="--",
                label=f"$\\lambda_{{\\max}} = {eigvals[0]:.2f}$")
    ax1.axhline(eigvals[1], color=C_GREEN, lw=1.5, ls="--",
                label=f"$\\lambda_{{\\min}} = {eigvals[1]:.2f}$")
    # mark the eigenvector angles
    for k, color in [(0, C_PURPLE), (1, C_GREEN)]:
        v = eigvecs[:, k]
        ang = np.degrees(np.arctan2(v[1], v[0])) % 180
        for shift in (0, 180):
            ax1.axvline(ang + shift, color=color, lw=0.8, alpha=0.5)
    ax1.set_xlim(0, 360)
    ax1.set_xticks([0, 90, 180, 270, 360])
    ax1.set_xlabel("direction $\\theta$ (degrees)", fontsize=10)
    ax1.set_ylabel("$R(\\vec{x}) = \\vec{x}^{T} A \\vec{x}$", fontsize=10)
    ax1.set_title("Rayleigh quotient on the unit circle",
                  fontsize=11, color=C_DARK, fontweight="bold", pad=8)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Right: polar / 2D view - color the unit circle by R(x)
    ax2 = fig.add_subplot(1, 2, 2)
    _setup_axes(ax2, lim=(-2.4, 2.4))
    sc = ax2.scatter(x[0], x[1], c=R, cmap="viridis", s=20, zorder=3)
    cbar = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("$R(\\vec{x})$", fontsize=10)
    # eigenvectors
    for k, color, name in [(0, C_PURPLE, "max"),
                           (1, C_GREEN, "min")]:
        v = eigvecs[:, k]
        _arrow(ax2, -1.6 * v, 1.6 * v, color=color, lw=2.2)
        ax2.text(1.7 * v[0], 1.7 * v[1],
                 f"$\\vec{{q}}_{{{name}}}$\n$={eigvals[k]:.2f}$",
                 color=color, fontsize=10, fontweight="bold",
                 ha="center")
    ax2.set_title("Extrema sit on the eigenvectors",
                  fontsize=11, color=C_DARK, fontweight="bold", pad=8)

    fig.suptitle("Rayleigh quotient:  $\\lambda_{\\min} \\leq "
                 "\\frac{\\vec{x}^{T} A \\vec{x}}{\\vec{x}^{T} \\vec{x}} "
                 "\\leq \\lambda_{\\max}$",
                 fontsize=13, color=C_DARK, fontweight="bold", y=1.02)
    save(fig, "fig6_rayleigh_quotient")


# ---------------------------------------------------------------------------
# Fig 7: SVD preview - generalising spectral theorem to any matrix
# ---------------------------------------------------------------------------
def fig7_svd_preview():
    # An asymmetric matrix; SVD still produces orthogonal frames.
    A = np.array([[2.0, 1.0],
                  [0.5, 1.5]])
    U, s, Vt = np.linalg.svd(A)
    # ensure right-handed orientation for nicer plots
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1
        s_signs = 1
    V = Vt.T

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # Left: input - unit circle with right singular vectors v1, v2
    ax = axes[0]
    _setup_axes(ax, lim=(-2.4, 2.4))
    theta = np.linspace(0, 2 * np.pi, 300)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ax.plot(circle[0], circle[1], color=C_BLUE, lw=2.2)
    for k, color in [(0, C_PURPLE), (1, C_GREEN)]:
        v = V[:, k]
        _arrow(ax, (0, 0), v, color=color, lw=2.4)
        ax.text(1.18 * v[0], 1.18 * v[1], f"$\\vec{{v}}_{k+1}$",
                color=color, fontsize=12, fontweight="bold")
    ax.set_title("Input:  unit circle and right singular vectors",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)

    # Right: output - ellipse with left singular vectors u1, u2 scaled by sigma
    ax = axes[1]
    _setup_axes(ax, lim=(-3.4, 3.4))
    ellipse = A @ circle
    ax.plot(ellipse[0], ellipse[1], color=C_BLUE, lw=2.2)
    for k, color in [(0, C_PURPLE), (1, C_GREEN)]:
        u = U[:, k]
        sig = s[k]
        _arrow(ax, (0, 0), sig * u, color=color, lw=2.4)
        ax.text(1.10 * sig * u[0], 1.10 * sig * u[1],
                f"$\\sigma_{k+1}\\vec{{u}}_{k+1}$\n$\\sigma_{k+1}={sig:.2f}$",
                color=color, fontsize=10, fontweight="bold",
                ha="center")
    ax.set_title("Output:  ellipse with left singular vectors $\\sigma_i \\vec{u}_i$",
                 fontsize=11, color=C_DARK, fontweight="bold", pad=8)

    fig.suptitle("SVD preview:  $A = U \\Sigma V^{T}$  generalises "
                 "$A = Q \\Lambda Q^{T}$ to any matrix",
                 fontsize=13, color=C_DARK, fontweight="bold", y=1.02)
    save(fig, "fig7_svd_preview")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_symmetric_structure()
    fig2_spectral_theorem()
    fig3_quadratic_signature()
    fig4_definiteness_zoo()
    fig5_principal_axes()
    fig6_rayleigh_quotient()
    fig7_svd_preview()
    print("Generated 7 figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
