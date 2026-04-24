"""
Figure generation script for Linear Algebra Chapter 07:
Orthogonality and Projections.

Produces 7 figures used in both EN and ZH versions of the article,
each teaching one specific idea cleanly in a 3Blue1Brown-inspired style.

Figures:
    fig1_orthogonal_vectors   Two perpendicular vectors with the
                              right-angle marker and dot product = 0.
    fig2_vector_projection    Geometric projection of b onto a, with the
                              error vector orthogonal to a.
    fig3_subspace_projection  3D projection of a vector onto a plane
                              (subspace). Error vector is normal to the plane.
    fig4_gram_schmidt         Gram-Schmidt step by step (3 input vectors in
                              R^3 turning into an orthogonal basis).
    fig5_orthogonal_complement
                              A line W and its orthogonal complement W^perp;
                              every vector splits uniquely.
    fig6_least_squares        Least squares as projection of b onto Col(A).
                              Data points + best-fit line + residuals.
    fig7_qr_decomposition     QR decomposition geometric meaning: skewed
                              column basis a1,a2 versus orthonormal q1,q2,
                              with R as the change-of-basis upper triangle.

Usage:
    python3 scripts/figures/linear-algebra/07-orthogonality-and-projections.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay consistent across languages.
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
    "07-orthogonality-and-projections"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "07-正交性与投影"
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
    if isinstance(lim, tuple) and len(lim) == 2 and isinstance(lim[0], (int, float)):
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    else:
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
    if aspect:
        ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, lw=0.6, zorder=0)
    ax.axvline(0, color=C_GRAY, lw=0.6, zorder=0)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_color(C_GRAY)
        spine.set_linewidth(0.6)


def _arrow(ax, start, end, color, lw=2.4, mut=14, zorder=4, ls="-"):
    ax.add_patch(FancyArrowPatch(tuple(start), tuple(end),
                                 arrowstyle="-|>", mutation_scale=mut,
                                 color=color, lw=lw, zorder=zorder,
                                 linestyle=ls))


def _right_angle(ax, corner, dir1, dir2, size=0.22, color=C_DARK):
    """Draw a small right-angle marker at `corner` between unit dirs."""
    corner = np.asarray(corner, dtype=float)
    d1 = np.asarray(dir1, dtype=float)
    d2 = np.asarray(dir2, dtype=float)
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    p1 = corner + size * d1
    p2 = corner + size * (d1 + d2)
    p3 = corner + size * d2
    ax.plot([p1[0], p2[0], p3[0]],
            [p1[1], p2[1], p3[1]],
            color=color, lw=1.1, zorder=5)


# ---------------------------------------------------------------------------
# Fig 1: Orthogonal vectors -- right angle, dot product zero
# ---------------------------------------------------------------------------
def fig1_orthogonal_vectors():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Left: orthogonal pair
    ax = axes[0]
    _setup_axes(ax, lim=(-1.0, 4.0))
    u = np.array([3.0, 1.0])
    # Pick v perpendicular to u, same length-ish
    v = np.array([-1.0, 3.0]) * (np.linalg.norm(u) / np.linalg.norm([-1, 3]))
    _arrow(ax, (0, 0), u, C_BLUE)
    _arrow(ax, (0, 0), v, C_PURPLE)
    _right_angle(ax, (0, 0), u, v, size=0.32)
    ax.text(u[0] + 0.15, u[1] - 0.05, r"$\vec{u}=(3,1)$",
            color=C_BLUE, fontsize=12, fontweight="bold")
    ax.text(v[0] - 0.6, v[1] + 0.15, r"$\vec{v}=(-1,3)$",
            color=C_PURPLE, fontsize=12, fontweight="bold")
    ax.text(2.0, -0.6,
            r"$\vec{u}\cdot\vec{v}=3\cdot(-1)+1\cdot 3=0$",
            ha="center", color=C_DARK, fontsize=11)
    ax.set_title(r"Orthogonal:  $\vec{u}\cdot\vec{v}=0\ \Leftrightarrow\ \theta=90^{\circ}$",
                 fontsize=12, color=C_DARK, fontweight="bold", pad=10)

    # Right: not orthogonal -- show angle
    ax = axes[1]
    _setup_axes(ax, lim=(-1.0, 4.0))
    a = np.array([3.0, 0.5])
    b = np.array([1.5, 2.5])
    _arrow(ax, (0, 0), a, C_BLUE)
    _arrow(ax, (0, 0), b, C_PURPLE)
    # Arc for angle
    from matplotlib.patches import Arc

    ang_a = np.degrees(np.arctan2(a[1], a[0]))
    ang_b = np.degrees(np.arctan2(b[1], b[0]))
    ax.add_patch(Arc((0, 0), 1.2, 1.2,
                     angle=0, theta1=ang_a, theta2=ang_b,
                     color=C_AMBER, lw=1.6))
    mid = (ang_a + ang_b) / 2
    ax.text(0.95 * np.cos(np.radians(mid)),
            0.95 * np.sin(np.radians(mid)),
            r"$\theta$", color=C_AMBER, fontsize=12, fontweight="bold")
    ax.text(a[0] + 0.1, a[1] - 0.25, r"$\vec{a}=(3,\,0.5)$",
            color=C_BLUE, fontsize=12, fontweight="bold")
    ax.text(b[0] + 0.1, b[1] + 0.1, r"$\vec{b}=(1.5,\,2.5)$",
            color=C_PURPLE, fontsize=12, fontweight="bold")
    dot = float(a @ b)
    ax.text(2.0, -0.6,
            rf"$\vec{{a}}\cdot\vec{{b}}={dot:.2f}\,\neq 0$  (not orthogonal)",
            ha="center", color=C_DARK, fontsize=11)
    ax.set_title("Generic pair: nonzero dot product",
                 fontsize=12, color=C_DARK, fontweight="bold", pad=10)

    fig.suptitle("Orthogonality: zero dot product = perpendicular",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_orthogonal_vectors")


# ---------------------------------------------------------------------------
# Fig 2: Vector projection of b onto a
# ---------------------------------------------------------------------------
def fig2_vector_projection():
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    _setup_axes(ax, lim=(-0.8, 5.0))

    a = np.array([4.0, 1.0])
    b = np.array([1.5, 3.2])
    proj = (a @ b) / (a @ a) * a
    err = b - proj

    # Dashed line through a (the span)
    t = np.linspace(-0.3, 1.4, 2)
    line = np.outer(t, a)
    ax.plot(line[:, 0], line[:, 1], color=C_GRAY, lw=1.0,
            linestyle="--", zorder=1, label=r"span$\{\vec{a}\}$")

    # Vectors
    _arrow(ax, (0, 0), a, C_BLUE)
    _arrow(ax, (0, 0), b, C_PURPLE)
    _arrow(ax, (0, 0), proj, C_GREEN)
    _arrow(ax, proj, b, C_AMBER, lw=2.0)

    # Right-angle marker at projection foot
    _right_angle(ax, proj, -proj, err, size=0.22)

    # Drop dashed line from b to proj
    ax.plot([b[0], proj[0]], [b[1], proj[1]],
            color=C_AMBER, lw=0.9, linestyle=":", zorder=2)

    # Labels
    ax.text(a[0] + 0.15, a[1] - 0.25, r"$\vec{a}$",
            color=C_BLUE, fontsize=14, fontweight="bold")
    ax.text(b[0] - 0.4, b[1] + 0.2, r"$\vec{b}$",
            color=C_PURPLE, fontsize=14, fontweight="bold")
    ax.text(proj[0] - 0.1, proj[1] - 0.55,
            r"$\hat{\vec{b}}=\mathrm{proj}_{\vec{a}}\vec{b}$",
            color=C_GREEN, fontsize=12, fontweight="bold")
    mid = (proj + b) / 2
    ax.text(mid[0] + 0.15, mid[1],
            r"$\vec{e}=\vec{b}-\hat{\vec{b}}$",
            color=C_AMBER, fontsize=12, fontweight="bold")

    # Formula box
    ax.text(0.05, 0.97,
            r"$\hat{\vec{b}}=\dfrac{\vec{a}\cdot\vec{b}}{\vec{a}\cdot\vec{a}}\,\vec{a}"
            r"\qquad \vec{e}\perp\vec{a}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=C_LIGHT, edgecolor=C_GRAY, lw=0.6))

    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_title("Projection onto a line: closest point + perpendicular error",
                 fontsize=13, color=C_DARK, fontweight="bold", pad=10)
    fig.tight_layout()
    save(fig, "fig2_vector_projection")


# ---------------------------------------------------------------------------
# Fig 3: Projection onto a subspace (plane in R^3)
# ---------------------------------------------------------------------------
def fig3_subspace_projection():
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # Plane W = span(u1, u2)
    u1 = np.array([1.0, 0.0, 0.0])
    u2 = np.array([0.0, 1.0, 0.0])
    n = np.cross(u1, u2)  # normal

    # Plane mesh
    s = np.linspace(-2.2, 2.2, 10)
    t = np.linspace(-2.2, 2.2, 10)
    S, T = np.meshgrid(s, t)
    Xp = S * u1[0] + T * u2[0]
    Yp = S * u1[1] + T * u2[1]
    Zp = S * u1[2] + T * u2[2]
    ax.plot_surface(Xp, Yp, Zp, color=C_BLUE, alpha=0.18,
                    edgecolor=C_BLUE, linewidth=0.2, zorder=1)

    # Vector b above the plane
    b = np.array([1.6, 1.2, 1.8])
    # Projection onto plane (z = 0 plane)
    bhat = b - (b @ n) / (n @ n) * n
    err = b - bhat

    def arr3d(start, end, color, lw=2.0):
        ax.quiver(start[0], start[1], start[2],
                  end[0] - start[0], end[1] - start[1], end[2] - start[2],
                  color=color, lw=lw, arrow_length_ratio=0.12)

    arr3d((0, 0, 0), b, C_PURPLE, lw=2.2)
    arr3d((0, 0, 0), bhat, C_GREEN, lw=2.2)
    arr3d(bhat, b, C_AMBER, lw=2.0)

    # Dashed drop line
    ax.plot([b[0], bhat[0]], [b[1], bhat[1]], [b[2], bhat[2]],
            color=C_AMBER, lw=0.9, linestyle=":")

    # Small right-angle marker at foot in the plane
    sz = 0.22
    p1 = bhat + sz * np.array([1, 0, 0])
    p2 = p1 + sz * np.array([0, 0, 1])
    p3 = bhat + sz * np.array([0, 0, 1])
    ax.plot([p1[0], p2[0], p3[0]],
            [p1[1], p2[1], p3[1]],
            [p1[2], p2[2], p3[2]],
            color=C_DARK, lw=1.1)

    # Labels
    ax.text(b[0] + 0.05, b[1] + 0.05, b[2] + 0.15,
            r"$\vec{b}$", color=C_PURPLE, fontsize=13, fontweight="bold")
    ax.text(bhat[0] + 0.1, bhat[1] - 0.4, bhat[2] - 0.1,
            r"$\hat{\vec{b}}=P\vec{b}$",
            color=C_GREEN, fontsize=12, fontweight="bold")
    ax.text((b[0] + bhat[0]) / 2 + 0.1,
            (b[1] + bhat[1]) / 2 + 0.1,
            (b[2] + bhat[2]) / 2,
            r"$\vec{b}-\hat{\vec{b}}\,\perp\,W$",
            color=C_AMBER, fontsize=11, fontweight="bold")
    ax.text(2.2, 0.0, 0.0, r"$W=\mathrm{Col}(A)$",
            color=C_BLUE, fontsize=12, fontweight="bold")

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-0.5, 2.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-58)
    ax.set_title("Projection onto a subspace: closest point in the plane",
                 fontsize=13, color=C_DARK, fontweight="bold", pad=10)
    fig.tight_layout()
    save(fig, "fig3_subspace_projection")


# ---------------------------------------------------------------------------
# Fig 4: Gram-Schmidt orthogonalization (3 vectors -> orthogonal basis)
# ---------------------------------------------------------------------------
def fig4_gram_schmidt():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # Three input vectors in R^2 to keep the picture readable
    a1 = np.array([3.0, 0.5])
    a2 = np.array([1.4, 2.6])
    a3 = np.array([2.2, -1.4])  # third illustrative vector for last panel

    # Gram-Schmidt
    u1 = a1.copy()
    u2 = a2 - (u1 @ a2) / (u1 @ u1) * u1
    # For panel 3 we'll show that u3 in R^2 must be 0; instead show a 3D third
    # axis by switching the third panel to a different visual (R^3 schematic).

    # ---- Panel 1: take a1 = u1
    ax = axes[0]
    _setup_axes(ax, lim=(-1.0, 4.2))
    _arrow(ax, (0, 0), a1, C_GRAY, lw=1.6, ls="--")
    _arrow(ax, (0, 0), u1, C_BLUE)
    ax.text(a1[0] + 0.1, a1[1] - 0.35, r"$\vec{a}_1$",
            color=C_GRAY, fontsize=12)
    ax.text(u1[0] + 0.1, u1[1] + 0.15, r"$\vec{u}_1=\vec{a}_1$",
            color=C_BLUE, fontsize=12, fontweight="bold")
    ax.set_title("Step 1:  keep the first direction",
                 fontsize=12, color=C_DARK, fontweight="bold", pad=10)

    # ---- Panel 2: subtract projection of a2 onto u1
    ax = axes[1]
    _setup_axes(ax, lim=(-1.0, 4.2))
    proj = (u1 @ a2) / (u1 @ u1) * u1
    _arrow(ax, (0, 0), u1, C_BLUE)
    _arrow(ax, (0, 0), a2, C_PURPLE)
    _arrow(ax, (0, 0), proj, C_GREEN, lw=1.8)
    _arrow(ax, proj, a2, C_AMBER)
    ax.plot([a2[0], proj[0]], [a2[1], proj[1]],
            color=C_AMBER, lw=0.9, linestyle=":")
    _right_angle(ax, proj, -proj, a2 - proj, size=0.2)
    ax.text(u1[0] + 0.1, u1[1] - 0.3, r"$\vec{u}_1$",
            color=C_BLUE, fontsize=12, fontweight="bold")
    ax.text(a2[0] - 0.1, a2[1] + 0.2, r"$\vec{a}_2$",
            color=C_PURPLE, fontsize=12, fontweight="bold")
    ax.text(proj[0] - 0.2, proj[1] - 0.55,
            r"proj$_{\vec{u}_1}\vec{a}_2$",
            color=C_GREEN, fontsize=10)
    mid = (proj + a2) / 2
    ax.text(mid[0] + 0.1, mid[1],
            r"$\vec{u}_2=\vec{a}_2-\mathrm{proj}_{\vec{u}_1}\vec{a}_2$",
            color=C_AMBER, fontsize=10, fontweight="bold")
    ax.set_title("Step 2:  subtract the component along $\\vec{u}_1$",
                 fontsize=12, color=C_DARK, fontweight="bold", pad=10)

    # ---- Panel 3: 3D illustration of three orthogonal axes after GS
    axes[2].remove()
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    # Original (skew) vectors
    A1 = np.array([2.2, 0.4, 0.2])
    A2 = np.array([0.8, 1.8, 0.4])
    A3 = np.array([0.5, 0.6, 1.7])

    # Gram-Schmidt in 3D
    U1 = A1
    U2 = A2 - (U1 @ A2) / (U1 @ U1) * U1
    U3 = A3 - (U1 @ A3) / (U1 @ U1) * U1 - (U2 @ A3) / (U2 @ U2) * U2

    # Normalize for cleaner arrows
    Q1 = U1 / np.linalg.norm(U1) * 2.0
    Q2 = U2 / np.linalg.norm(U2) * 2.0
    Q3 = U3 / np.linalg.norm(U3) * 2.0

    def q3d(v, color, label, off=(0.05, 0.05, 0.05)):
        ax3.quiver(0, 0, 0, v[0], v[1], v[2], color=color,
                   lw=2.4, arrow_length_ratio=0.12)
        ax3.text(v[0] + off[0], v[1] + off[1], v[2] + off[2],
                 label, color=color, fontsize=12, fontweight="bold")

    # Faded original vectors
    for V, lab in [(A1, r"$\vec{a}_1$"),
                   (A2, r"$\vec{a}_2$"),
                   (A3, r"$\vec{a}_3$")]:
        ax3.quiver(0, 0, 0, V[0], V[1], V[2],
                   color=C_GRAY, lw=1.1, arrow_length_ratio=0.1,
                   linestyle="dashed", alpha=0.7)
        ax3.text(V[0] + 0.05, V[1] + 0.05, V[2] + 0.05,
                 lab, color=C_GRAY, fontsize=10)

    q3d(Q1, C_BLUE, r"$\vec{q}_1$")
    q3d(Q2, C_PURPLE, r"$\vec{q}_2$")
    q3d(Q3, C_GREEN, r"$\vec{q}_3$")

    ax3.set_xlim(0, 2.5); ax3.set_ylim(0, 2.5); ax3.set_zlim(0, 2.5)
    ax3.view_init(elev=20, azim=35)
    ax3.set_title("Step 3:  iterate -> orthogonal basis",
                  fontsize=12, color=C_DARK, fontweight="bold", pad=10)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")

    fig.suptitle("Gram-Schmidt:  skewed inputs $\\to$ orthogonal basis",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig4_gram_schmidt")


# ---------------------------------------------------------------------------
# Fig 5: Orthogonal complement
# ---------------------------------------------------------------------------
def fig5_orthogonal_complement():
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    _setup_axes(ax, lim=(-3.5, 3.5))

    # W = span((2,1)) -- a line; W^perp = span((-1,2))
    w_dir = np.array([2.0, 1.0])
    w_dir /= np.linalg.norm(w_dir)
    p_dir = np.array([-1.0, 2.0])
    p_dir /= np.linalg.norm(p_dir)

    # Draw the two lines
    t = np.linspace(-3.5, 3.5, 2)
    Lw = np.outer(t, w_dir)
    Lp = np.outer(t, p_dir)
    ax.plot(Lw[:, 0], Lw[:, 1], color=C_BLUE, lw=2.2,
            label=r"$W=\mathrm{span}\{\vec{w}\}$")
    ax.plot(Lp[:, 0], Lp[:, 1], color=C_PURPLE, lw=2.2,
            label=r"$W^{\perp}$")

    # A vector v to decompose
    v = np.array([1.6, 2.4])
    v_par = (v @ w_dir) * w_dir
    v_perp = v - v_par

    _arrow(ax, (0, 0), v, C_DARK, lw=2.2)
    _arrow(ax, (0, 0), v_par, C_GREEN, lw=2.0)
    _arrow(ax, (0, 0), v_perp, C_AMBER, lw=2.0)

    # Dashed parallelogram showing decomposition
    ax.plot([v_par[0], v[0]], [v_par[1], v[1]],
            color=C_AMBER, lw=0.9, linestyle=":")
    ax.plot([v_perp[0], v[0]], [v_perp[1], v[1]],
            color=C_GREEN, lw=0.9, linestyle=":")

    # Right-angle marker at origin between the two lines
    _right_angle(ax, (0, 0), w_dir, p_dir, size=0.28)

    # Labels
    ax.text(v[0] + 0.1, v[1] + 0.1, r"$\vec{v}$",
            color=C_DARK, fontsize=13, fontweight="bold")
    ax.text(v_par[0] + 0.1, v_par[1] - 0.35,
            r"$\vec{v}_W\in W$",
            color=C_GREEN, fontsize=11, fontweight="bold")
    ax.text(v_perp[0] - 0.85, v_perp[1] + 0.1,
            r"$\vec{v}_{W^{\perp}}\in W^{\perp}$",
            color=C_AMBER, fontsize=11, fontweight="bold")

    # Formula box
    ax.text(0.03, 0.97,
            r"$\mathbb{R}^n=W\oplus W^{\perp}$"
            "\n"
            r"$\vec{v}=\vec{v}_W+\vec{v}_{W^{\perp}}$ (unique)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=C_LIGHT, edgecolor=C_GRAY, lw=0.6))

    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_title("Orthogonal complement: every vector splits uniquely",
                 fontsize=13, color=C_DARK, fontweight="bold", pad=10)
    fig.tight_layout()
    save(fig, "fig5_orthogonal_complement")


# ---------------------------------------------------------------------------
# Fig 6: Least squares as projection onto Col(A)
# ---------------------------------------------------------------------------
def fig6_least_squares():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ---- Left: data + best-fit line + residuals
    ax = axes[0]
    rng = np.random.default_rng(7)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y_true = 0.9 + 1.2 * x
    y = y_true + rng.normal(0, 0.7, size=x.size)

    # Fit
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ beta

    xs = np.linspace(0.5, 7.5, 100)
    ax.plot(xs, beta[0] + beta[1] * xs, color=C_BLUE, lw=2.0,
            label=fr"best-fit  $y={beta[0]:.2f}+{beta[1]:.2f}x$")

    # Residual segments
    for xi, yi, yhi in zip(x, y, y_hat):
        ax.plot([xi, xi], [yi, yhi], color=C_AMBER, lw=1.3, alpha=0.85)

    ax.scatter(x, y, s=55, color=C_PURPLE, zorder=5, label="data")
    ax.scatter(x, y_hat, s=40, marker="x", color=C_GREEN, zorder=5,
               label=r"projection $A\hat{\beta}$")

    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, max(y.max(), y_hat.max()) + 1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.set_title("Data view: minimize sum of squared residuals",
                 fontsize=12, color=C_DARK, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25, lw=0.6)

    # ---- Right: geometric view -- b, projection onto Col(A) plane
    ax2 = axes[1]
    axes[1].remove()
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Col(A) as a tilted plane
    e1 = np.array([1.0, 0.2, 0.1])
    e2 = np.array([0.2, 1.0, -0.1])
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n)

    s = np.linspace(-2.2, 2.2, 10)
    t = np.linspace(-2.2, 2.2, 10)
    S, T = np.meshgrid(s, t)
    Xp = S * e1[0] + T * e2[0]
    Yp = S * e1[1] + T * e2[1]
    Zp = S * e1[2] + T * e2[2]
    ax2.plot_surface(Xp, Yp, Zp, color=C_BLUE, alpha=0.18,
                     edgecolor=C_BLUE, linewidth=0.2)

    b = np.array([0.9, 1.1, 1.6])
    bhat = b - (b @ n) * n

    def arr3d(start, end, color, lw=2.0):
        ax2.quiver(start[0], start[1], start[2],
                   end[0] - start[0], end[1] - start[1], end[2] - start[2],
                   color=color, lw=lw, arrow_length_ratio=0.12)

    arr3d((0, 0, 0), b, C_PURPLE, lw=2.2)
    arr3d((0, 0, 0), bhat, C_GREEN, lw=2.2)
    arr3d(bhat, b, C_AMBER, lw=2.0)
    ax2.plot([b[0], bhat[0]], [b[1], bhat[1]], [b[2], bhat[2]],
             color=C_AMBER, lw=0.9, linestyle=":")

    ax2.text(b[0] + 0.05, b[1] + 0.05, b[2] + 0.15,
             r"$\vec{b}$ (data)",
             color=C_PURPLE, fontsize=12, fontweight="bold")
    ax2.text(bhat[0] + 0.1, bhat[1] - 0.5, bhat[2] - 0.05,
             r"$A\hat{\beta}=\hat{\vec{b}}$",
             color=C_GREEN, fontsize=12, fontweight="bold")
    ax2.text((b[0] + bhat[0]) / 2 + 0.1,
             (b[1] + bhat[1]) / 2 + 0.05,
             (b[2] + bhat[2]) / 2,
             r"residual $\perp$ Col$(A)$",
             color=C_AMBER, fontsize=10, fontweight="bold")
    ax2.text(2.2, 0, 0, r"Col$(A)$",
             color=C_BLUE, fontsize=12, fontweight="bold")

    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_zlim(-0.5, 2.5)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.view_init(elev=22, azim=-55)
    ax2.set_title("Geometric view: projection onto column space",
                  fontsize=12, color=C_DARK, fontweight="bold", pad=10)

    fig.suptitle("Least squares = projection of $\\vec{b}$ onto Col$(A)$",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_least_squares")


# ---------------------------------------------------------------------------
# Fig 7: QR decomposition geometric meaning
# ---------------------------------------------------------------------------
def fig7_qr_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # Original skewed columns of A
    a1 = np.array([3.0, 1.0])
    a2 = np.array([1.5, 2.6])

    # QR via Gram-Schmidt
    q1 = a1 / np.linalg.norm(a1)
    proj = (q1 @ a2) * q1
    u2 = a2 - proj
    q2 = u2 / np.linalg.norm(u2)
    r11 = q1 @ a1
    r12 = q1 @ a2
    r22 = q2 @ a2

    # ---- Left: A's columns (skewed)
    ax = axes[0]
    _setup_axes(ax, lim=(-0.8, 4.0))
    _arrow(ax, (0, 0), a1, C_BLUE)
    _arrow(ax, (0, 0), a2, C_PURPLE)
    ax.text(a1[0] + 0.1, a1[1] - 0.3, r"$\vec{a}_1$",
            color=C_BLUE, fontsize=13, fontweight="bold")
    ax.text(a2[0] - 0.1, a2[1] + 0.2, r"$\vec{a}_2$",
            color=C_PURPLE, fontsize=13, fontweight="bold")
    ax.set_title(r"Columns of $A$ (skewed, not orthogonal)",
                 fontsize=12, color=C_DARK, fontweight="bold", pad=10)

    # ---- Right: orthonormal Q + show R relationship
    ax = axes[1]
    _setup_axes(ax, lim=(-0.8, 4.0))
    # scaled q's so they are visible (length ~ 1)
    _arrow(ax, (0, 0), q1, C_BLUE)
    _arrow(ax, (0, 0), q2, C_GREEN)
    _right_angle(ax, (0, 0), q1, q2, size=0.18)

    # Show a2 as r12*q1 + r22*q2 by drawing the addition
    p1 = r12 * q1
    p2 = p1 + r22 * q2
    ax.plot([0, p1[0]], [0, p1[1]],
            color=C_BLUE, lw=1.2, linestyle="--", alpha=0.7)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=C_GREEN, lw=1.2, linestyle="--", alpha=0.7)
    _arrow(ax, (0, 0), a2, C_PURPLE, lw=1.6)

    ax.text(q1[0] + 0.05, q1[1] - 0.35, r"$\vec{q}_1$",
            color=C_BLUE, fontsize=12, fontweight="bold")
    ax.text(q2[0] - 0.45, q2[1] + 0.05, r"$\vec{q}_2$",
            color=C_GREEN, fontsize=12, fontweight="bold")
    ax.text(a2[0] + 0.1, a2[1] + 0.05, r"$\vec{a}_2$",
            color=C_PURPLE, fontsize=12, fontweight="bold")
    ax.text(p1[0] / 2, p1[1] - 0.3,
            fr"$r_{{12}}={r12:.2f}$",
            color=C_BLUE, fontsize=10)
    ax.text((p1[0] + p2[0]) / 2 + 0.05, (p1[1] + p2[1]) / 2,
            fr"$r_{{22}}={r22:.2f}$",
            color=C_GREEN, fontsize=10)

    # R matrix annotation (matplotlib mathtext lacks \begin{...}, so format manually)
    R_text = (fr"$R=\left[\begin{{matrix}}{r11:.2f} & {r12:.2f}\\ 0 & {r22:.2f}"
              r"\end{matrix}\right]$  (upper triangular)")
    # Fallback: use plain text since some matplotlib versions also lack \begin{matrix}
    R_text = (f"R = [ {r11:.2f}  {r12:.2f} ]\n"
              f"    [ 0.00  {r22:.2f} ]   (upper triangular)")
    ax.text(0.03, 0.97, R_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=C_LIGHT, edgecolor=C_GRAY, lw=0.6))

    ax.set_title(r"Orthonormal $Q$, with $R$ recording the coordinates",
                 fontsize=12, color=C_DARK, fontweight="bold", pad=10)

    fig.suptitle(r"QR decomposition:  $A=QR$  "
                 r"(orthonormal columns + upper triangular)",
                 fontsize=14, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_qr_decomposition")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_orthogonal_vectors()
    fig2_vector_projection()
    fig3_subspace_projection()
    fig4_gram_schmidt()
    fig5_orthogonal_complement()
    fig6_least_squares()
    fig7_qr_decomposition()
    print("All 7 figures generated for chapter 07.")


if __name__ == "__main__":
    main()
