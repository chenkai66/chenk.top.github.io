"""
Figure generation script for Linear Algebra Chapter 09:
Singular Value Decomposition (SVD).

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in 3Blue1Brown style.

Figures:
    fig1_svd_geometry          The three-step decomposition of A = U Sigma V^T:
                               unit circle -> rotate by V^T -> stretch by
                               Sigma -> rotate by U.  Four panels arranged
                               left-to-right with labelled basis vectors.
    fig2_circle_to_ellipse     Detailed view of the circle-to-ellipse story:
                               right singular vectors v1, v2 on the input
                               side map to sigma_i u_i on the output side,
                               revealing the principal axes of the ellipse.
    fig3_eig_vs_svd            Eigenvalues vs singular values for a non-
                               symmetric 2x2 matrix: eigenvectors are not
                               orthogonal and eigenvalues describe invariant
                               directions, while singular vectors ARE
                               orthogonal and singular values describe
                               stretching factors.
    fig4_low_rank_blocks       Low-rank approximation as a sum of rank-1
                               outer products: a 64x64 synthetic image is
                               rebuilt one rank-1 layer at a time
                               (rank 1, 2, 3) plus the singular value bars.
    fig5_image_compression     Image-compression demonstration on a clean
                               procedurally generated grayscale "scene"
                               (rings + stripes + smooth gradient): original
                               vs k = 5, 20, 50, plus the singular-value
                               decay curve and cumulative-energy curve.
    fig6_pseudoinverse         Pseudoinverse via SVD on an overdetermined
                               system y ~ a x + b: A^+ b returns the
                               least-squares line; geometrically it projects
                               b onto the column space of A.
    fig7_pca_via_svd           PCA = SVD of centred data.  An elliptical
                               cloud of 2D points with its mean, the two
                               principal axes (right singular vectors of the
                               centred matrix) drawn proportional to sigma_i,
                               and the projection onto PC1.

Usage:
    python3 scripts/figures/linear-algebra/09-singular-value-decomposition.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon

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
    "09-singular-value-decomposition"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "09-奇异值分解SVD"
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
def arrow(ax, start, end, color, lw=2.2, alpha=1.0, label=None):
    a = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, alpha=alpha, zorder=5,
    )
    ax.add_patch(a)
    if label is not None:
        mid = 0.5 * (np.asarray(start) + np.asarray(end))
        ax.text(*mid, label, color=color, fontsize=10, fontweight="bold",
                ha="center", va="bottom", zorder=6)


def style_axes(ax, lim=2.4, hide_spines=False):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, lw=0.6, alpha=0.7)
    ax.axvline(0, color=C_GRAY, lw=0.6, alpha=0.7)
    ax.tick_params(labelsize=8, colors=C_GRAY)
    if hide_spines:
        for s in ax.spines.values():
            s.set_visible(False)


def unit_circle(n=200):
    t = np.linspace(0, 2 * np.pi, n)
    return np.vstack([np.cos(t), np.sin(t)])


# ===========================================================================
# Figure 1: SVD as three steps - unit circle through V^T, Sigma, U
# ===========================================================================
def fig1_svd_geometry():
    # Choose A so that the story is visually clear.
    theta_v = np.deg2rad(35)        # rotation V^T
    theta_u = np.deg2rad(-20)       # rotation U
    s1, s2 = 2.0, 0.8

    Vt = np.array([[np.cos(theta_v), np.sin(theta_v)],
                   [-np.sin(theta_v), np.cos(theta_v)]])
    S = np.diag([s1, s2])
    U = np.array([[np.cos(theta_u), -np.sin(theta_u)],
                  [np.sin(theta_u), np.cos(theta_u)]])

    circ = unit_circle()
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    # Stage outputs
    s0 = circ                       # input: unit circle
    s1c = Vt @ circ                 # after V^T
    s2c = S @ s1c                   # after Sigma
    s3c = U @ s2c                   # after U  ( = A @ circ )

    e1_0, e2_0 = e1, e2
    e1_1, e2_1 = Vt @ e1, Vt @ e2
    e1_2, e2_2 = S @ e1_1, S @ e2_1
    e1_3, e2_3 = U @ e1_2, U @ e2_2

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    titles = [
        r"Input  ($\mathbb{R}^n$)",
        r"After $V^{\!\top}$  (rotate)",
        r"After $\Sigma$  (stretch)",
        r"After $U$  (rotate)  $= A\,$",
    ]
    stages = [(s0, e1_0, e2_0), (s1c, e1_1, e2_1),
              (s2c, e1_2, e2_2), (s3c, e1_3, e2_3)]

    for ax, title, (curve, b1, b2) in zip(axes, titles, stages):
        ax.fill(curve[0], curve[1], color=C_BLUE, alpha=0.10, zorder=1)
        ax.plot(curve[0], curve[1], color=C_BLUE, lw=2.0, zorder=2)
        arrow(ax, (0, 0), b1, C_GREEN, lw=2.6)
        arrow(ax, (0, 0), b2, C_AMBER, lw=2.6)
        style_axes(ax, lim=2.6)
        ax.set_title(title, fontsize=11, color=C_DARK, pad=8)

    axes[0].text(1.05, -0.2, r"$e_1$", color=C_GREEN, fontsize=11,
                 fontweight="bold")
    axes[0].text(-0.3, 1.05, r"$e_2$", color=C_AMBER, fontsize=11,
                 fontweight="bold")
    axes[3].text(0.05, -2.45, r"semi-axes $= \sigma_1, \sigma_2$",
                 color=C_DARK, fontsize=9.5, ha="left")

    fig.suptitle(
        r"SVD: any linear map factors as  rotate  $\to$  stretch  $\to$  rotate",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig1_svd_geometry")


# ===========================================================================
# Figure 2: circle -> ellipse with v_i / sigma_i u_i annotated
# ===========================================================================
def fig2_circle_to_ellipse():
    # Build a non-symmetric A from a chosen SVD.
    theta_v = np.deg2rad(28)
    theta_u = np.deg2rad(-18)
    s1, s2 = 2.2, 0.9
    V = np.array([[np.cos(theta_v), -np.sin(theta_v)],
                  [np.sin(theta_v), np.cos(theta_v)]])
    U = np.array([[np.cos(theta_u), -np.sin(theta_u)],
                  [np.sin(theta_u), np.cos(theta_u)]])
    A = U @ np.diag([s1, s2]) @ V.T

    v1 = V[:, 0]; v2 = V[:, 1]
    u1 = U[:, 0]; u2 = U[:, 1]

    circ = unit_circle()
    ell = A @ circ

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    # --- Left: input
    ax = axes[0]
    ax.fill(circ[0], circ[1], color=C_BLUE, alpha=0.10)
    ax.plot(circ[0], circ[1], color=C_BLUE, lw=2)
    arrow(ax, (0, 0), v1, C_GREEN, lw=2.6)
    arrow(ax, (0, 0), v2, C_AMBER, lw=2.6)
    ax.text(v1[0]*1.15, v1[1]*1.15, r"$v_1$", color=C_GREEN,
            fontsize=13, fontweight="bold", ha="center")
    ax.text(v2[0]*1.20, v2[1]*1.20, r"$v_2$", color=C_AMBER,
            fontsize=13, fontweight="bold", ha="center")
    style_axes(ax, lim=2.7)
    ax.set_title("Input: unit circle\nright singular vectors $v_1, v_2$",
                 fontsize=11, color=C_DARK)

    # --- Right: output ellipse
    ax = axes[1]
    ax.fill(ell[0], ell[1], color=C_PURPLE, alpha=0.10)
    ax.plot(ell[0], ell[1], color=C_PURPLE, lw=2)
    arrow(ax, (0, 0), s1 * u1, C_GREEN, lw=2.8)
    arrow(ax, (0, 0), s2 * u2, C_AMBER, lw=2.8)
    ax.text((s1 * u1)[0]*1.10, (s1 * u1)[1]*1.10,
            r"$\sigma_1 u_1$", color=C_GREEN,
            fontsize=13, fontweight="bold", ha="center")
    ax.text((s2 * u2)[0]*1.40, (s2 * u2)[1]*1.40 + 0.1,
            r"$\sigma_2 u_2$", color=C_AMBER,
            fontsize=13, fontweight="bold", ha="center")
    style_axes(ax, lim=2.7)
    ax.set_title("Output: ellipse $A\\,(\\text{circle})$\n"
                 "semi-axis lengths $= \\sigma_i$",
                 fontsize=11, color=C_DARK)

    # Big arrow between
    fig.text(0.50, 0.52, r"$A\,v_i = \sigma_i\, u_i$",
             ha="center", va="center", fontsize=14, color=C_DARK,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec=C_GRAY, lw=1))
    fig.suptitle("From unit circle to ellipse: the geometry of SVD",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_circle_to_ellipse")


# ===========================================================================
# Figure 3: eigenvalues vs singular values
# ===========================================================================
def fig3_eig_vs_svd():
    # Non-symmetric matrix with both eigen and SVD interesting.
    A = np.array([[2.0, 1.2],
                  [0.4, 1.6]])
    eigvals, eigvecs = np.linalg.eig(A)
    # Sort by |eigenvalue| descending
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    U, s, Vt = np.linalg.svd(A)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # --- Left: eigenvectors
    ax = axes[0]
    circ = unit_circle()
    ell = A @ circ
    ax.plot(circ[0], circ[1], color=C_GRAY, lw=1, ls="--", alpha=0.7)
    ax.plot(ell[0], ell[1], color=C_PURPLE, lw=1.8, alpha=0.7)
    for i, c in enumerate([C_GREEN, C_AMBER]):
        v = eigvecs[:, i].real
        v = v / np.linalg.norm(v)
        Av = A @ v
        arrow(ax, (0, 0), v, c, lw=2.4)
        arrow(ax, (0, 0), Av, c, lw=2.4, alpha=0.55)
        ax.text(v[0]*1.18, v[1]*1.18, f"$x_{i+1}$",
                color=c, fontsize=12, fontweight="bold", ha="center")
        ax.text(Av[0]*1.08, Av[1]*1.08,
                f"$\\lambda_{i+1}\\,x_{i+1}$",
                color=c, fontsize=10, fontweight="bold", ha="center",
                alpha=0.85)
    style_axes(ax, lim=3.2)
    angle = np.degrees(np.arccos(np.clip(
        np.abs(eigvecs[:, 0].real @ eigvecs[:, 1].real /
               (np.linalg.norm(eigvecs[:, 0]) *
                np.linalg.norm(eigvecs[:, 1]))), -1, 1)))
    ax.set_title("Eigenvectors: invariant directions\n"
                 f"(not orthogonal: angle $\\approx {angle:.0f}^\\circ$)",
                 fontsize=11, color=C_DARK)

    # --- Right: singular vectors
    ax = axes[1]
    ax.plot(circ[0], circ[1], color=C_GRAY, lw=1, ls="--", alpha=0.7)
    ax.plot(ell[0], ell[1], color=C_PURPLE, lw=1.8, alpha=0.7)
    for i, c in enumerate([C_GREEN, C_AMBER]):
        v = Vt[i]
        u = U[:, i]
        arrow(ax, (0, 0), v, c, lw=2.4)
        arrow(ax, (0, 0), s[i] * u, c, lw=2.4, alpha=0.55)
        ax.text(v[0]*1.18, v[1]*1.18, f"$v_{i+1}$",
                color=c, fontsize=12, fontweight="bold", ha="center")
        ax.text((s[i]*u[0])*1.08, (s[i]*u[1])*1.08,
                f"$\\sigma_{i+1}\\,u_{i+1}$",
                color=c, fontsize=10, fontweight="bold", ha="center",
                alpha=0.85)
    style_axes(ax, lim=3.2)
    ax.set_title("Singular vectors: orthogonal in / orthogonal out\n"
                 f"$\\sigma_1={s[0]:.2f},\\;\\sigma_2={s[1]:.2f}$",
                 fontsize=11, color=C_DARK)

    fig.suptitle(
        f"Eigenvalues  vs  singular values  (same matrix "
        f"$A$, $|\\lambda_i|=$ {abs(eigvals[0]):.2f}, {abs(eigvals[1]):.2f})",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig3_eig_vs_svd")


# ===========================================================================
# Helper: build a clean synthetic grayscale "scene"
# ===========================================================================
def build_scene(n=128, rng=None):
    """A grayscale image whose singular values decay quickly enough to
    make low-rank approximation visually meaningful."""
    if rng is None:
        rng = np.random.default_rng(7)
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    # Smooth gradient
    g = 0.5 * (X + 1)
    # Concentric rings
    R = np.sqrt((X + 0.3) ** 2 + (Y - 0.2) ** 2)
    rings = 0.5 + 0.5 * np.cos(8 * np.pi * R)
    # Diagonal stripes
    stripes = 0.5 + 0.5 * np.sin(6 * np.pi * (X + Y))
    # Soft blob
    blob = np.exp(-((X - 0.4) ** 2 + (Y + 0.4) ** 2) / 0.08)

    img = 0.35 * g + 0.30 * rings + 0.25 * stripes + 0.30 * blob
    img = img - img.min()
    img = img / img.max()
    return img


# ===========================================================================
# Figure 4: low-rank approximation as outer-product layers
# ===========================================================================
def fig4_low_rank_blocks():
    img = build_scene(96)
    U, s, Vt = np.linalg.svd(img, full_matrices=False)

    layers = []
    cum = np.zeros_like(img)
    for k in range(3):
        layer = s[k] * np.outer(U[:, k], Vt[k])
        cum = cum + layer
        layers.append((k + 1, layer.copy(), cum.copy()))

    fig = plt.figure(figsize=(13.5, 6.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.30, wspace=0.18)

    # Top row: the rank-1 layers themselves
    for i, (k, layer, _) in enumerate(layers):
        ax = fig.add_subplot(gs[0, i])
        vmax = np.abs(layer).max()
        ax.imshow(layer, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Layer {k}: $\\sigma_{{{k}}}\\, u_{{{k}}} v_{{{k}}}^{{\\top}}$"
                     f"  ($\\sigma={s[k-1]:.2f}$)",
                     fontsize=10, color=C_DARK)
        ax.set_xticks([]); ax.set_yticks([])

    # Top-right: the original
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(img, cmap="gray")
    ax.set_title("Original", fontsize=10, color=C_DARK)
    ax.set_xticks([]); ax.set_yticks([])

    # Bottom row: cumulative reconstructions + singular values
    for i, (k, _, cum_k) in enumerate(layers):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(cum_k, cmap="gray", vmin=0, vmax=1)
        energy = (s[:k] ** 2).sum() / (s ** 2).sum() * 100
        ax.set_title(f"Sum of layers 1..{k}\n"
                     f"energy {energy:.1f}%",
                     fontsize=10, color=C_DARK)
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom-right: singular value bar chart (top 12)
    ax = fig.add_subplot(gs[1, 3])
    nshow = 12
    bars = ax.bar(np.arange(1, nshow + 1), s[:nshow],
                  color=C_GRAY, edgecolor="white")
    for j in range(3):
        bars[j].set_color([C_GREEN, C_AMBER, C_PURPLE][j])
    ax.set_xlabel("index $i$", fontsize=10)
    ax.set_ylabel("$\\sigma_i$", fontsize=10)
    ax.set_title("Singular values", fontsize=10, color=C_DARK)
    ax.tick_params(labelsize=9)

    fig.suptitle(
        "Low-rank approximation: each rank-1 layer adds detail; "
        "early layers carry most of the energy",
        fontsize=13, color=C_DARK, y=0.99,
    )
    save(fig, "fig4_low_rank_blocks")


# ===========================================================================
# Figure 5: image compression demonstration
# ===========================================================================
def fig5_image_compression():
    img = build_scene(128)
    U, s, Vt = np.linalg.svd(img, full_matrices=False)

    ks = [5, 20, 50]
    m, n = img.shape
    total = m * n

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.35, wspace=0.22)

    # Top row: original + 3 reconstructions
    titles = ["Original (rank "
              f"{np.linalg.matrix_rank(img)})\n{total:,} numbers stored"]
    imgs = [img]
    for k in ks:
        rec = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
        stored = k * (m + n + 1)
        ratio = stored / total * 100
        energy = (s[:k] ** 2).sum() / (s ** 2).sum() * 100
        titles.append(f"$k={k}$  ({ratio:.0f}% storage, "
                      f"{energy:.1f}% energy)")
        imgs.append(rec)

    for i, (im, t) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=10, color=C_DARK)
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom-left wide: singular values (log scale)
    ax = fig.add_subplot(gs[1, :2])
    idx = np.arange(1, len(s) + 1)
    ax.semilogy(idx, s, color=C_BLUE, lw=2)
    for k, c in zip(ks, [C_GREEN, C_AMBER, C_PURPLE]):
        ax.axvline(k, color=c, lw=1.4, ls="--", alpha=0.85,
                   label=f"$k={k}$")
    ax.set_xlabel("index $i$", fontsize=10)
    ax.set_ylabel("$\\sigma_i$  (log)", fontsize=10)
    ax.set_title("Singular value spectrum", fontsize=11, color=C_DARK)
    ax.legend(fontsize=9, frameon=True)
    ax.tick_params(labelsize=9)

    # Bottom-right wide: cumulative energy
    ax = fig.add_subplot(gs[1, 2:])
    cum = np.cumsum(s ** 2) / np.sum(s ** 2) * 100
    ax.plot(idx, cum, color=C_PURPLE, lw=2)
    ax.fill_between(idx, 0, cum, color=C_PURPLE, alpha=0.10)
    for k, c in zip(ks, [C_GREEN, C_AMBER, C_PURPLE]):
        ax.axvline(k, color=c, lw=1.4, ls="--", alpha=0.85)
        ax.scatter([k], [cum[k - 1]], color=c, zorder=5, s=40)
        ax.annotate(f"{cum[k-1]:.1f}%", (k, cum[k - 1]),
                    xytext=(8, -12), textcoords="offset points",
                    color=c, fontsize=9, fontweight="bold")
    ax.set_xlabel("number of components $k$", fontsize=10)
    ax.set_ylabel("cumulative energy (%)", fontsize=10)
    ax.set_title("Cumulative energy captured", fontsize=11, color=C_DARK)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=9)

    fig.suptitle(
        "Image compression by rank-$k$ truncation: a few components "
        "already capture most of the picture",
        fontsize=13, color=C_DARK, y=0.99,
    )
    save(fig, "fig5_image_compression")


# ===========================================================================
# Figure 6: pseudoinverse via SVD (least-squares line fit)
# ===========================================================================
def fig6_pseudoinverse():
    rng = np.random.default_rng(3)
    n = 25
    x = np.linspace(-2, 4, n)
    true_a, true_b = 1.2, 0.4
    y = true_a * x + true_b + rng.normal(0, 1.0, n)

    A = np.column_stack([x, np.ones_like(x)])    # n x 2
    # Pseudoinverse via SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    A_pinv = Vt.T @ np.diag(1 / s) @ U.T
    coef = A_pinv @ y                             # [a_hat, b_hat]
    a_hat, b_hat = coef
    y_hat = A @ coef
    residuals = y - y_hat

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # --- Left: data + fit
    ax = axes[0]
    ax.scatter(x, y, color=C_BLUE, s=42, zorder=4, label="data $b_i$")
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
    ax.plot(xs, a_hat * xs + b_hat, color=C_GREEN, lw=2.4,
            label=f"least-squares  $\\hat y = {a_hat:.2f}\\,x + "
                  f"{b_hat:.2f}$")
    for xi, yi, yhi in zip(x, y, y_hat):
        ax.plot([xi, xi], [yi, yhi], color=C_AMBER, lw=1.0, alpha=0.7)
    ax.set_xlabel("$x$", fontsize=10)
    ax.set_ylabel("$y$", fontsize=10)
    ax.set_title(r"Overdetermined  $A\hat x = A^{+} b$",
                 fontsize=11, color=C_DARK)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=9)

    # --- Right: geometry - residual orthogonal to col(A)
    ax = axes[1]
    # Project b onto col(A): b_hat_vec = A coef ; residual r = y - y_hat
    # Show schematic: 3D plane viewed in 2D as a slanted band.
    # Draw vector b, projection onto col(A), residual.
    # Use a stylised diagram (no real data axes).
    ax.set_xlim(-1.2, 5.2); ax.set_ylim(-1.2, 4.2)
    ax.set_aspect("equal")
    # Column space "plane" as a parallelogram
    P = np.array([[0, 0], [4.6, 0.8], [4.6, 1.8], [0, 1.0]])
    poly = Polygon(P, closed=True, facecolor=C_LIGHT, edgecolor=C_GRAY,
                   alpha=0.6)
    ax.add_patch(poly)
    ax.text(4.0, 0.4, r"col$(A)$", color=C_GRAY, fontsize=11)
    # Vector b (data) sticking out
    arrow(ax, (0, 0), (3.6, 3.4), C_BLUE, lw=2.6)
    ax.text(3.7, 3.5, r"$b$", color=C_BLUE, fontsize=13,
            fontweight="bold")
    # Projection (b_hat) onto col(A)
    arrow(ax, (0, 0), (3.6, 1.55), C_GREEN, lw=2.6)
    ax.text(3.7, 1.45, r"$A\hat x = A A^{+} b$", color=C_GREEN,
            fontsize=11, fontweight="bold")
    # Residual
    arrow(ax, (3.6, 1.55), (3.6, 3.4), C_AMBER, lw=2.4)
    ax.text(3.72, 2.4, r"$r = b - A\hat x$", color=C_AMBER,
            fontsize=11, fontweight="bold")
    # Right angle marker at projection
    ax.plot([3.45, 3.45, 3.6], [1.55, 1.7, 1.7], color=C_DARK, lw=1)

    ax.axis("off")
    ax.set_title("Geometry: $A^{+}b$ projects $b$ onto col$(A)$\n"
                 "residual $\\perp$ col$(A)$  (normal equations)",
                 fontsize=11, color=C_DARK)

    fig.suptitle(r"Pseudoinverse via SVD:  $A^{+} = V\,\Sigma^{+}\,U^{\!\top}$"
                 "  gives the minimum-norm least-squares solution",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_pseudoinverse")


# ===========================================================================
# Figure 7: PCA via SVD of centered data
# ===========================================================================
def fig7_pca_via_svd():
    rng = np.random.default_rng(11)
    # Generate a tilted ellipse cloud
    cov = np.array([[3.0, 1.6],
                    [1.6, 1.2]])
    n = 250
    pts = rng.multivariate_normal([2.5, 1.5], cov, size=n)

    mean = pts.mean(axis=0)
    Xc = pts - mean

    # SVD of centered data
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]; pc2 = Vt[1]
    # Variance explained
    var = (s ** 2) / (n - 1)
    ratio = var / var.sum()

    # Length of each axis = standard deviation in that direction
    std1 = np.sqrt(var[0]); std2 = np.sqrt(var[1])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: scatter + principal axes
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], color=C_BLUE, alpha=0.45, s=22,
               edgecolor="white", lw=0.4, label="data")
    ax.scatter([mean[0]], [mean[1]], color=C_DARK, marker="x", s=80, lw=2,
               zorder=5, label="mean")
    # Principal axes, drawn as 2*std arrows from the mean
    arrow(ax, mean, mean + 2.2 * std1 * pc1, C_GREEN, lw=2.8)
    arrow(ax, mean, mean + 2.2 * std2 * pc2, C_AMBER, lw=2.8)
    ax.text(*(mean + 2.4 * std1 * pc1), "PC1", color=C_GREEN,
            fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(*(mean + 2.6 * std2 * pc2), "PC2", color=C_AMBER,
            fontsize=12, fontweight="bold", ha="center", va="center")
    # 1-sigma ellipse
    eigvals = var
    angle = np.degrees(np.arctan2(pc1[1], pc1[0]))
    ell = Ellipse(xy=mean, width=2 * std1, height=2 * std2,
                  angle=angle, facecolor="none",
                  edgecolor=C_PURPLE, lw=1.8, ls="--", alpha=0.8)
    ax.add_patch(ell)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=10)
    ax.set_ylabel("$x_2$", fontsize=10)
    ax.set_title("Centred data with principal axes\n"
                 "axes = right singular vectors of $X_c$",
                 fontsize=11, color=C_DARK)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=9)

    # --- Right: projection on PC1 (1D representation)
    ax = axes[1]
    scores = Xc @ Vt.T          # (n, 2)
    pc1_scores = scores[:, 0]
    ax.hist(pc1_scores, bins=22, color=C_GREEN, alpha=0.65,
            edgecolor="white")
    ax.set_xlabel("score along PC1", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.set_title(f"PC1 captures {ratio[0]*100:.1f}% of variance, "
                 f"PC2 captures {ratio[1]*100:.1f}%",
                 fontsize=11, color=C_DARK)
    ax.tick_params(labelsize=9)

    fig.suptitle("PCA = SVD of centred data: principal directions "
                 "are the right singular vectors $v_i$",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_pca_via_svd")


# ===========================================================================
# Main
# ===========================================================================
def main():
    fig1_svd_geometry()
    fig2_circle_to_ellipse()
    fig3_eig_vs_svd()
    fig4_low_rank_blocks()
    fig5_image_compression()
    fig6_pseudoinverse()
    fig7_pca_via_svd()
    print("Saved 7 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
