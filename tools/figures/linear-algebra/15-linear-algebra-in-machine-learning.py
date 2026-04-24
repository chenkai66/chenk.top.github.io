"""
Figure generation script for Linear Algebra Chapter 15:
Linear Algebra in Machine Learning.

Generates 7 figures shared between the EN and ZH articles. Each figure
isolates a single linear-algebra-driven ML idea so it reads cleanly when
embedded into prose.

Figures:
    fig1_pca_projection         A 2D correlated point cloud, the two principal
                                axes drawn as arrows scaled by sqrt(variance),
                                and the 1D projection onto PC1 shown beneath.
    fig2_regression_projection  Geometry of least squares: y, the column space
                                of X (a tilted plane), and the residual drawn
                                orthogonal to the plane (y_hat is the foot of
                                the perpendicular).
    fig3_svm_margin             Linearly separable two-class data with the
                                max-margin hyperplane, the margin band, and
                                support vectors highlighted.
    fig4_kernel_trick           Two concentric rings in 2D (linearly
                                inseparable) and the same data lifted to 3D
                                with phi(x,y)=(x, y, x^2+y^2): a single plane
                                now separates the classes.
    fig5_matrix_factorization   User x movie rating matrix R with missing
                                cells, factored visually as P @ Q^T; the
                                recovered dense prediction is shown to the
                                right.
    fig6_lda_separation         Two overlapping Gaussian classes in 2D with the
                                PCA direction (max variance) vs the LDA
                                direction (max class separability) drawn, plus
                                the 1D class-conditional projections.
    fig7_word_embeddings        Word vectors for king/queen/man/woman
                                (and a few extras) in 2D, with the parallel
                                analogy arrows king-man and queen-woman that
                                illustrate vec(king)-vec(man)+vec(woman) ~=
                                vec(queen).

Style:
    matplotlib + seaborn-v0_8-whitegrid, dpi=150, palette
    {#2563eb, #7c3aed, #10b981, #f59e0b}.

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.

Usage:
    python3 scripts/figures/linear-algebra/15-linear-algebra-in-machine-learning.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

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
RNG = np.random.default_rng(20240427)

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "15-linear-algebra-in-machine-learning"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "15-机器学习中的线性代数"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH EN and ZH asset folders as <name>.png."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        path = d / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig1_pca_projection
# ---------------------------------------------------------------------------
def fig1_pca_projection() -> None:
    n = 240
    theta = np.deg2rad(28.0)
    base = RNG.normal(size=(n, 2)) * np.array([2.4, 0.6])
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    X = base @ rot.T

    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt          # rows are PC directions
    variances = (S ** 2) / n
    proj = Xc @ components[0]  # 1D coordinates on PC1

    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0],
                          height_ratios=[3.2, 1.0], hspace=0.38, wspace=0.28)

    # Panel 1: data cloud + principal axes
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.scatter(X[:, 0], X[:, 1], s=22, color=C_BLUE, alpha=0.55,
                edgecolor="none", label="Data points")

    mu = X.mean(axis=0)
    scales = 2.4 * np.sqrt(variances)
    colors = [C_AMBER, C_GREEN]
    labels = ["PC1 (max variance)", "PC2"]
    for vec, scale, color, lab in zip(components, scales, colors, labels):
        end = mu + vec * scale
        start = mu - vec * scale
        ax1.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=2.6, mutation_scale=18))
        ax1.plot([], [], color=color, lw=2.6, label=lab)

    ax1.set_aspect("equal")
    ax1.set_xlim(-7, 7)
    ax1.set_ylim(-4.2, 4.2)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_title("PCA: principal axes of a 2D data cloud",
                  fontsize=12, color=C_DARK)
    ax1.legend(loc="upper left", framealpha=0.95)

    # Panel 2: variance bar
    ax2 = fig.add_subplot(gs[0, 1])
    ratios = variances / variances.sum()
    ax2.bar(["PC1", "PC2"], ratios, color=[C_AMBER, C_GREEN], width=0.55)
    for i, r in enumerate(ratios):
        ax2.text(i, r + 0.02, f"{r*100:.1f}%", ha="center",
                 fontsize=11, color=C_DARK)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Explained variance ratio")
    ax2.set_title("Variance captured per component", fontsize=11)

    # Panel 3: 1D projection on PC1
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(proj, np.zeros_like(proj), s=20, color=C_AMBER,
                alpha=0.55, edgecolor="none")
    ax3.axhline(0, color=C_GRAY, lw=1)
    ax3.set_yticks([])
    ax3.set_xlim(-7, 7)
    ax3.set_xlabel("Projected coordinate on PC1")
    ax3.set_title("1D representation after dropping PC2", fontsize=11)

    save(fig, "fig1_pca_projection")


# ---------------------------------------------------------------------------
# fig2_regression_projection
# ---------------------------------------------------------------------------
def fig2_regression_projection() -> None:
    fig = plt.figure(figsize=(8.8, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    # Two columns of X span a 2D plane through the origin.
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.0, 1.0, 0.0])
    grid = np.linspace(-1.6, 1.6, 12)
    G1, G2 = np.meshgrid(grid, grid)
    P = G1[..., None] * a1 + G2[..., None] * a2
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2],
                    color=C_BLUE, alpha=0.18, edgecolor=C_BLUE, linewidth=0.3)

    # Target y is off-plane; projection y_hat lives in the plane.
    y = np.array([1.1, 0.9, 1.4])
    y_hat = np.array([y[0], y[1], 0.0])

    def arrow(start, end, color, lw=2.4, label=None):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=color, lw=lw, label=label)
        ax.scatter(*end, color=color, s=30)

    arrow([0, 0, 0], y, C_PURPLE, label=r"$\mathbf{y}$ (target)")
    arrow([0, 0, 0], y_hat, C_AMBER,
          label=r"$\hat{\mathbf{y}}=\mathbf{X}\hat{\boldsymbol{\beta}}$")
    arrow(y_hat, y, C_GREEN,
          label=r"residual $\mathbf{y}-\hat{\mathbf{y}}$")

    # Right-angle marker between residual and plane.
    n = np.array([0.0, 0.0, 0.12])
    t = (y - y_hat); t = t / np.linalg.norm(t) * 0.18
    foot = y_hat + n * 0.0  # base of marker at y_hat
    p1 = foot + np.array([0.18, 0.0, 0.0])
    p2 = p1 + np.array([0.0, 0.0, 0.18])
    p3 = foot + np.array([0.0, 0.0, 0.18])
    ax.plot([foot[0], p1[0], p2[0], p3[0], foot[0]],
            [foot[1], p1[1], p2[1], p3[1], foot[1]],
            [foot[2], p1[2], p2[2], p3[2], foot[2]],
            color=C_GREEN, lw=1)

    ax.text(*(y + np.array([0.05, 0.05, 0.05])), "y", fontsize=12,
            color=C_PURPLE)
    ax.text(*(y_hat + np.array([0.05, -0.15, -0.2])), r"$\hat{y}$",
            fontsize=12, color=C_AMBER)
    ax.text(0.55, 1.5, 0.02, "col($X$)", fontsize=11, color=C_BLUE)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_zlim(-0.2, 1.8)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.set_title("Least squares = orthogonal projection of $y$ onto col($X$)",
                 fontsize=12, color=C_DARK, pad=12)
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    save(fig, "fig2_regression_projection")


# ---------------------------------------------------------------------------
# fig3_svm_margin
# ---------------------------------------------------------------------------
def fig3_svm_margin() -> None:
    # Two well-separated Gaussian blobs.
    n = 60
    mu_pos = np.array([2.0, 2.0])
    mu_neg = np.array([-2.0, -2.0])
    Xp = RNG.normal(size=(n, 2)) * 0.55 + mu_pos
    Xn = RNG.normal(size=(n, 2)) * 0.55 + mu_neg

    # Hand-pick the max-margin hyperplane: it is the perpendicular bisector
    # of the line connecting the two means (true under equal isotropic noise).
    w = (mu_pos - mu_neg)
    w = w / np.linalg.norm(w)
    b = -0.5 * (mu_pos + mu_neg) @ w  # so w.x + b = 0 at the midpoint

    # Pick three "support vectors": closest point of each class to the line,
    # plus one tied point on the positive side.
    def signed_dist(X):
        return X @ w + b

    sv_pos_idx = np.argsort(signed_dist(Xp))[:1]
    sv_neg_idx = np.argsort(-signed_dist(Xn))[:1]
    sv_pos = Xp[sv_pos_idx]
    sv_neg = Xn[sv_neg_idx]
    margin = (abs(float(signed_dist(sv_pos)[0]))
              + abs(float(signed_dist(sv_neg)[0]))) / 2.0

    fig, ax = plt.subplots(figsize=(7.8, 6.6))

    ax.scatter(Xp[:, 0], Xp[:, 1], s=42, color=C_BLUE, alpha=0.75,
               edgecolor="white", linewidth=0.5, label="Class +1")
    ax.scatter(Xn[:, 0], Xn[:, 1], s=42, color=C_PURPLE, alpha=0.75,
               edgecolor="white", linewidth=0.5, label="Class -1")

    # Draw the hyperplane and margin lines.
    xs = np.linspace(-4.5, 4.5, 200)
    # w[0]*x + w[1]*y + b = 0 -> y = -(w[0]*x + b)/w[1]
    ys0 = -(w[0] * xs + b) / w[1]
    ys_pos = -(w[0] * xs + b - 1) / w[1]   # offset by margin in feature space
    ys_neg = -(w[0] * xs + b + 1) / w[1]
    # Re-derive the visual margin lines in physical units.
    # Move along the unit normal w by +-margin.
    n_vec = w
    line_pts = np.stack([xs, ys0], axis=1)
    upper = line_pts + n_vec * margin
    lower = line_pts - n_vec * margin

    ax.plot(xs, ys0, color=C_DARK, lw=2.2, label="Decision boundary")
    ax.plot(upper[:, 0], upper[:, 1], "--", color=C_AMBER, lw=1.6,
            label="Margin")
    ax.plot(lower[:, 0], lower[:, 1], "--", color=C_AMBER, lw=1.6)
    ax.fill(np.concatenate([upper[:, 0], lower[::-1, 0]]),
            np.concatenate([upper[:, 1], lower[::-1, 1]]),
            color=C_AMBER, alpha=0.10)

    # Highlight support vectors.
    for sv in np.vstack([sv_pos, sv_neg]):
        ax.scatter(sv[0], sv[1], s=230, facecolor="none",
                   edgecolor=C_GREEN, linewidth=2.3, zorder=5)
    ax.scatter([], [], s=230, facecolor="none", edgecolor=C_GREEN,
               linewidth=2.3, label="Support vectors")

    # Annotate the margin width.
    mid = np.array([0.0, -b / w[1]])
    ax.annotate("", xy=mid + n_vec * margin, xytext=mid - n_vec * margin,
                arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=1.8))
    ax.text(*(mid + np.array([0.25, 0.15])), r"margin $=2/\|w\|$",
            color=C_GREEN, fontsize=11)

    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("SVM: maximum-margin hyperplane and support vectors",
                 fontsize=12, color=C_DARK)
    ax.legend(loc="lower right", framealpha=0.95)

    save(fig, "fig3_svm_margin")


# ---------------------------------------------------------------------------
# fig4_kernel_trick
# ---------------------------------------------------------------------------
def fig4_kernel_trick() -> None:
    # Concentric rings dataset.
    n = 140
    r_inner = 0.8 + 0.10 * RNG.normal(size=n)
    r_outer = 2.2 + 0.10 * RNG.normal(size=n)
    theta_in = RNG.uniform(0, 2 * np.pi, size=n)
    theta_out = RNG.uniform(0, 2 * np.pi, size=n)
    Xi = np.stack([r_inner * np.cos(theta_in),
                   r_inner * np.sin(theta_in)], axis=1)
    Xo = np.stack([r_outer * np.cos(theta_out),
                   r_outer * np.sin(theta_out)], axis=1)

    fig = plt.figure(figsize=(13.5, 5.6))

    # Left: original 2D
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(Xi[:, 0], Xi[:, 1], color=C_BLUE, s=28, alpha=0.8,
                edgecolor="white", linewidth=0.4, label="Class A")
    ax1.scatter(Xo[:, 0], Xo[:, 1], color=C_PURPLE, s=28, alpha=0.8,
                edgecolor="white", linewidth=0.4, label="Class B")
    ax1.set_aspect("equal")
    ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3)
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")
    ax1.set_title("Original 2D space: no straight line separates the classes",
                  fontsize=11, color=C_DARK)
    ax1.legend(loc="upper right", framealpha=0.95)

    # Right: lifted 3D with phi(x,y) = (x, y, x^2 + y^2)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    Zi = (Xi ** 2).sum(axis=1)
    Zo = (Xo ** 2).sum(axis=1)
    ax2.scatter(Xi[:, 0], Xi[:, 1], Zi, color=C_BLUE, s=24, alpha=0.85,
                depthshade=False)
    ax2.scatter(Xo[:, 0], Xo[:, 1], Zo, color=C_PURPLE, s=24, alpha=0.85,
                depthshade=False)

    # Separating plane z = c (between the two ring radii squared).
    c = 0.5 * (1.0 ** 2 + 2.2 ** 2)
    g = np.linspace(-3, 3, 14)
    Gx, Gy = np.meshgrid(g, g)
    ax2.plot_surface(Gx, Gy, np.full_like(Gx, c),
                     color=C_AMBER, alpha=0.30, edgecolor=C_AMBER,
                     linewidth=0.3)
    ax2.text(2.2, -2.6, c + 0.4, r"plane $z=c$",
             color=C_AMBER, fontsize=11)

    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    ax2.set_zlabel(r"$z=x_1^2+x_2^2$")
    ax2.set_title(r"After $\phi(x_1,x_2)=(x_1,x_2,x_1^2+x_2^2)$: "
                  r"linearly separable",
                  fontsize=11, color=C_DARK)
    ax2.view_init(elev=18, azim=-62)

    save(fig, "fig4_kernel_trick")


# ---------------------------------------------------------------------------
# fig5_matrix_factorization
# ---------------------------------------------------------------------------
def fig5_matrix_factorization() -> None:
    rng = np.random.default_rng(7)
    m, n, k = 6, 8, 2
    P_true = rng.normal(size=(m, k))
    Q_true = rng.normal(size=(n, k))
    R_true = P_true @ Q_true.T
    # Scale to a 1..5 rating range for display.
    R_true = (R_true - R_true.min()) / (R_true.max() - R_true.min())
    R_true = 1 + 4 * R_true

    mask = rng.random((m, n)) < 0.55
    R_obs = np.where(mask, R_true, np.nan)

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 0.25, 0.7, 0.55, 1.5],
                          wspace=0.35)

    # Observed rating matrix R with NaNs as gray cells.
    ax_R = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("Blues")
    ax_R.imshow(np.where(np.isnan(R_obs), 0, R_obs), cmap=cmap, vmin=1, vmax=5)
    for i in range(m):
        for j in range(n):
            if np.isnan(R_obs[i, j]):
                ax_R.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=C_LIGHT, edgecolor="white"))
                ax_R.text(j, i, "?", ha="center", va="center",
                          color=C_DARK, fontsize=11)
            else:
                ax_R.text(j, i, f"{R_obs[i,j]:.0f}", ha="center", va="center",
                          color="white" if R_obs[i, j] > 3 else C_DARK,
                          fontsize=10)
    ax_R.set_xticks(range(n)); ax_R.set_yticks(range(m))
    ax_R.set_xticklabels([f"M{j+1}" for j in range(n)])
    ax_R.set_yticklabels([f"U{i+1}" for i in range(m)])
    ax_R.set_title(r"Rating matrix $R$ (? = unobserved)",
                   fontsize=11, color=C_DARK)
    ax_R.tick_params(length=0)

    # Equals sign panel.
    ax_eq = fig.add_subplot(gs[0, 1]); ax_eq.axis("off")
    ax_eq.text(0.5, 0.5, r"$\approx$", ha="center", va="center",
               fontsize=34, color=C_DARK)

    # P factor (m x k).
    ax_P = fig.add_subplot(gs[0, 2])
    ax_P.imshow(P_true, cmap="PuOr", aspect="auto")
    ax_P.set_xticks(range(k)); ax_P.set_xticklabels([f"f{j+1}" for j in range(k)])
    ax_P.set_yticks(range(m)); ax_P.set_yticklabels([f"U{i+1}" for i in range(m)])
    ax_P.set_title(r"User factors $P$", fontsize=11, color=C_DARK)
    ax_P.tick_params(length=0)

    # Times sign panel.
    ax_t = fig.add_subplot(gs[0, 3]); ax_t.axis("off")
    ax_t.text(0.5, 0.5, r"$\times$", ha="center", va="center",
              fontsize=28, color=C_DARK)
    ax_t.text(0.5, 0.18, r"$Q^\top$", ha="center", va="center",
              fontsize=14, color=C_DARK)

    # Q^T factor (k x n) and predicted dense matrix.
    ax_pred = fig.add_subplot(gs[0, 4])
    R_pred = P_true @ Q_true.T
    R_pred = 1 + 4 * (R_pred - R_pred.min()) / (R_pred.max() - R_pred.min())
    im = ax_pred.imshow(R_pred, cmap=cmap, vmin=1, vmax=5)
    for i in range(m):
        for j in range(n):
            ax_pred.text(j, i, f"{R_pred[i,j]:.1f}", ha="center", va="center",
                         color="white" if R_pred[i, j] > 3 else C_DARK,
                         fontsize=9)
    ax_pred.set_xticks(range(n)); ax_pred.set_yticks(range(m))
    ax_pred.set_xticklabels([f"M{j+1}" for j in range(n)])
    ax_pred.set_yticklabels([f"U{i+1}" for i in range(m)])
    ax_pred.set_title(r"Recovered $\hat R = P Q^\top$",
                      fontsize=11, color=C_DARK)
    ax_pred.tick_params(length=0)

    fig.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04, label="rating")
    fig.suptitle("Collaborative filtering as low-rank matrix factorization",
                 fontsize=13, color=C_DARK, y=1.02)

    save(fig, "fig5_matrix_factorization")


# ---------------------------------------------------------------------------
# fig6_lda_separation
# ---------------------------------------------------------------------------
def fig6_lda_separation() -> None:
    # Two elongated Gaussian classes whose elongation direction is NOT the
    # direction that separates them. PCA picks the elongation direction;
    # LDA picks the direction that splits the classes.
    n = 140
    cov = np.array([[3.2, 1.7], [1.7, 1.1]])
    L = np.linalg.cholesky(cov)
    Xa = (RNG.normal(size=(n, 2)) @ L.T) + np.array([-1.4, 1.6])
    Xb = (RNG.normal(size=(n, 2)) @ L.T) + np.array([1.4, -1.6])
    X = np.vstack([Xa, Xb])
    y = np.array([0] * n + [1] * n)

    # PCA direction.
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    w_pca = Vt[0]

    # Fisher LDA direction (binary).
    mu_a = Xa.mean(axis=0); mu_b = Xb.mean(axis=0)
    Sa = (Xa - mu_a).T @ (Xa - mu_a)
    Sb = (Xb - mu_b).T @ (Xb - mu_b)
    Sw = Sa + Sb
    w_lda = np.linalg.solve(Sw, mu_b - mu_a)
    w_lda = w_lda / np.linalg.norm(w_lda)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for ax, w, title, color in zip(
        axes,
        [w_pca, w_lda],
        ["PCA direction (max variance)",
         "LDA direction (max class separability)"],
        [C_AMBER, C_GREEN],
    ):
        ax.scatter(Xa[:, 0], Xa[:, 1], color=C_BLUE, s=24, alpha=0.65,
                   edgecolor="white", linewidth=0.4, label="Class A")
        ax.scatter(Xb[:, 0], Xb[:, 1], color=C_PURPLE, s=24, alpha=0.65,
                   edgecolor="white", linewidth=0.4, label="Class B")
        mu = X.mean(axis=0)
        end = mu + w * 6
        start = mu - w * 6
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.6, mutation_scale=18))

        # Project both classes onto the chosen direction and draw a histogram
        # along the bottom of the panel.
        proj_a = (Xa - mu) @ w
        proj_b = (Xb - mu) @ w
        bins = np.linspace(-7, 7, 30)
        h_a, _ = np.histogram(proj_a, bins=bins)
        h_b, _ = np.histogram(proj_b, bins=bins)
        scale = 0.6 / max(h_a.max(), h_b.max(), 1)
        bottom = -7.5
        for h, c in zip([h_a, h_b], [C_BLUE, C_PURPLE]):
            for k, hi in enumerate(h):
                ax.add_patch(Rectangle((bins[k], bottom),
                                       bins[k + 1] - bins[k], hi * scale,
                                       facecolor=c, edgecolor="none",
                                       alpha=0.55))

        ax.axhline(bottom, color=C_GRAY, lw=0.8)
        ax.text(0, bottom - 0.7, "1D projection", ha="center",
                fontsize=10, color=C_GRAY)

        ax.set_aspect("equal")
        ax.set_xlim(-8, 8); ax.set_ylim(-9, 6)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title(title, fontsize=11, color=C_DARK)
        ax.legend(loc="upper right", framealpha=0.95)

    fig.suptitle("PCA vs LDA: variance is not the same as class separability",
                 fontsize=13, color=C_DARK, y=1.02)

    save(fig, "fig6_lda_separation")


# ---------------------------------------------------------------------------
# fig7_word_embeddings
# ---------------------------------------------------------------------------
def fig7_word_embeddings() -> None:
    # Stylised 2D embedding designed so that the gender and royalty axes are
    # roughly orthogonal -- mirrors the qualitative structure of real word2vec
    # embeddings projected into two dimensions.
    words = {
        "man":     np.array([-1.6,  -1.5]),
        "woman":   np.array([ 1.6,  -1.5]),
        "king":    np.array([-1.6,   1.5]),
        "queen":   np.array([ 1.6,   1.5]),
        "boy":     np.array([-1.7,  -2.4]),
        "girl":    np.array([ 1.7,  -2.4]),
        "uncle":   np.array([-2.6,  -0.7]),
        "aunt":    np.array([ 2.6,  -0.7]),
        "prince":  np.array([-1.6,   2.4]),
        "princess":np.array([ 1.6,   2.4]),
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.6))

    royal = {"king", "queen", "prince", "princess"}
    male = {"man", "boy", "uncle", "king", "prince"}
    for w, xy in words.items():
        color = C_AMBER if w in royal else C_BLUE
        marker = "s" if w in male else "o"
        ax.scatter(*xy, s=110, color=color, marker=marker,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(w, xy=xy, xytext=(8, 6), textcoords="offset points",
                    fontsize=11, color=C_DARK)

    # Draw the two parallel "gender" arrows: man -> woman and king -> queen.
    arrow_kw = dict(arrowstyle="-|>", color=C_PURPLE, lw=2.2,
                    mutation_scale=18)
    ax.annotate("", xy=words["woman"], xytext=words["man"],
                arrowprops=arrow_kw)
    ax.annotate("", xy=words["queen"], xytext=words["king"],
                arrowprops=arrow_kw)

    # And the "royalty" arrow: man -> king (parallel to woman -> queen).
    royalty_kw = dict(arrowstyle="-|>", color=C_GREEN, lw=2.2,
                      mutation_scale=18)
    ax.annotate("", xy=words["king"], xytext=words["man"],
                arrowprops=royalty_kw)
    ax.annotate("", xy=words["queen"], xytext=words["woman"],
                arrowprops=royalty_kw)

    # Predicted point: king - man + woman, should land on queen.
    pred = words["king"] - words["man"] + words["woman"]
    ax.scatter(*pred, s=240, facecolor="none", edgecolor=C_GREEN,
               linewidth=2.4, zorder=4)
    ax.annotate("king - man + woman", xy=pred, xytext=(15, -22),
                textcoords="offset points", fontsize=10, color=C_GREEN,
                arrowprops=dict(arrowstyle="-", color=C_GREEN, lw=0.8))

    legend_handles = [
        Line2D([0], [0], color=C_PURPLE, lw=2.2, label="gender direction"),
        Line2D([0], [0], color=C_GREEN, lw=2.2, label="royalty direction"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.95)

    ax.set_xlim(-4, 4); ax.set_ylim(-3.4, 3.4)
    ax.set_aspect("equal")
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    ax.set_title("Word embedding analogy: parallel directions encode "
                 "semantic relations",
                 fontsize=12, color=C_DARK)

    save(fig, "fig7_word_embeddings")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Chapter 15 ...")
    fig1_pca_projection()
    fig2_regression_projection()
    fig3_svm_margin()
    fig4_kernel_trick()
    fig5_matrix_factorization()
    fig6_lda_separation()
    fig7_word_embeddings()
    print("Done.")


if __name__ == "__main__":
    main()
