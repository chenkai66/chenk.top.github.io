"""
Figure generation script for ML Math Derivations Part 08:
Support Vector Machines.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates one geometric or algorithmic idea so the math becomes
visible.

Figures:
    fig1_max_margin             Hard-margin SVM on a linearly separable set:
                                hyperplane, margin band, support vectors
                                highlighted, geometric margin annotated.
    fig2_soft_margin_C          Soft-margin SVM at three values of C
                                (small / medium / large) showing how C
                                trades margin width for slack penalty.
    fig3_kernel_trick_3d        Concentric-ring data lifted by phi(x) =
                                (x1, x2, x1^2 + x2^2): 2D not separable,
                                3D linearly separable by a flat plane.
    fig4_rbf_boundary           RBF-kernel decision boundary on the moons
                                dataset together with support vectors and
                                gamma effect.
    fig5_loss_comparison        0/1 loss vs hinge loss vs logistic loss vs
                                squared loss as functions of margin
                                y * f(x).
    fig6_kkt_geometry           KKT geometry: complementary slackness made
                                visual -- inactive samples (alpha=0),
                                active boundary SVs (0<alpha<C) and bound
                                SVs (alpha=C).
    fig7_smo_step               One SMO step in the (alpha_1, alpha_2)
                                plane: feasibility line, box [0,C]^2,
                                clipping interval [L,H], unconstrained
                                optimum and clipped optimum.

Usage:
    python3 scripts/figures/ml-math-derivations/08-svm.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D proj)

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

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "08-Support-Vector-Machines"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "08-支持向量机"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Maximum margin classifier
# ---------------------------------------------------------------------------
def fig1_max_margin() -> None:
    rng = np.random.default_rng(7)
    # Two well-separated Gaussian blobs
    pos = rng.normal(loc=[2.6, 2.4], scale=0.55, size=(18, 2))
    neg = rng.normal(loc=[-2.0, -1.8], scale=0.55, size=(18, 2))

    # Inject three "support vector" points by hand so the geometry is clean
    pos = np.vstack([pos, [[0.6, 1.4], [1.6, 0.4]]])
    neg = np.vstack([neg, [[-0.4, -0.6]]])

    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))])

    # Train a hard-ish margin SVM
    from sklearn.svm import SVC
    clf = SVC(kernel="linear", C=1e3).fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    margin = 1.0 / np.linalg.norm(w)

    fig, ax = plt.subplots(figsize=(8.4, 6.4))

    # Decision boundary and margin lines
    xs = np.linspace(-4.2, 4.6, 200)
    yb = -(w[0] * xs + b) / w[1]
    yp = -(w[0] * xs + b - 1) / w[1]
    ym = -(w[0] * xs + b + 1) / w[1]

    # Margin band shading
    ax.fill_between(xs, ym, yp, color=C_LIGHT, alpha=0.55,
                    label=f"Margin band (width $= 2/\\Vert w\\Vert$)")
    ax.plot(xs, yb, color=C_DARK, lw=2.4,
            label=r"Decision boundary $w^\top x + b = 0$")
    ax.plot(xs, yp, color=C_BLUE, lw=1.4, ls="--",
            label=r"$w^\top x + b = +1$")
    ax.plot(xs, ym, color=C_PURPLE, lw=1.4, ls="--",
            label=r"$w^\top x + b = -1$")

    # All points
    ax.scatter(pos[:, 0], pos[:, 1], color=C_BLUE, s=70, edgecolor="white",
               linewidth=1.4, zorder=4, label=r"class $+1$")
    ax.scatter(neg[:, 0], neg[:, 1], color=C_PURPLE, s=70, edgecolor="white",
               linewidth=1.4, zorder=4, label=r"class $-1$")

    # Highlight support vectors with a yellow halo
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=260, facecolor="none",
               edgecolor=C_AMBER, linewidth=2.6, zorder=3,
               label="support vectors")

    # Margin arrow (perpendicular from boundary to a SV)
    # Pick a support vector, project onto boundary
    sv0 = sv[np.argmax(sv @ w + b)]  # one with largest f(x)
    proj_t = -(w @ sv0 + b) / (w @ w)
    foot = sv0 + proj_t * w
    arrow = FancyArrowPatch(
        tuple(foot), tuple(sv0),
        arrowstyle="<->", mutation_scale=14,
        color=C_AMBER, lw=2.0, zorder=5,
    )
    ax.add_patch(arrow)
    mid = 0.5 * (foot + sv0)
    ax.annotate(rf"$\gamma = \dfrac{{1}}{{\Vert w\Vert}} \approx {margin:.2f}$",
                xy=tuple(mid), xytext=(mid[0] + 0.6, mid[1] + 0.6),
                fontsize=11.5, color=C_DARK,
                arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.8))

    # w-direction arrow at origin of the boundary
    centre = np.array([0.0, -b / w[1]])
    w_unit = w / np.linalg.norm(w)
    w_arrow = FancyArrowPatch(
        tuple(centre), tuple(centre + 1.2 * w_unit),
        arrowstyle="->", mutation_scale=14,
        color=C_GREEN, lw=2.0, zorder=5,
    )
    ax.add_patch(w_arrow)
    ax.text(centre[0] + 1.3 * w_unit[0], centre[1] + 1.3 * w_unit[1],
            r"$w$", fontsize=12, color=C_GREEN, fontweight="bold")

    ax.set_xlim(-4.2, 4.6)
    ax.set_ylim(-3.6, 4.4)
    ax.set_xlabel(r"$x_1$", fontsize=11)
    ax.set_ylabel(r"$x_2$", fontsize=11)
    ax.set_title("Hard-margin SVM: maximum-margin hyperplane and support vectors",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, "fig1_max_margin")


# ---------------------------------------------------------------------------
# Figure 2: Soft margin -- effect of C
# ---------------------------------------------------------------------------
def fig2_soft_margin_C() -> None:
    rng = np.random.default_rng(3)
    # Overlapping classes so soft margin is meaningful
    n = 40
    pos = rng.normal(loc=[1.2, 1.0], scale=0.95, size=(n, 2))
    neg = rng.normal(loc=[-1.2, -0.8], scale=0.95, size=(n, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n), -np.ones(n)])

    from sklearn.svm import SVC
    Cs = [0.05, 1.0, 50.0]
    titles = [
        r"small $C = 0.05$  (wide margin, many slack)",
        r"moderate $C = 1$  (balanced)",
        r"large $C = 50$  (narrow margin, fewer slack)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)

    for ax, C, title in zip(axes, Cs, titles):
        clf = SVC(kernel="linear", C=C).fit(X, y)
        w, b = clf.coef_[0], clf.intercept_[0]

        xs = np.linspace(-4.0, 4.0, 200)
        yb = -(w[0] * xs + b) / w[1]
        yp = -(w[0] * xs + b - 1) / w[1]
        ym = -(w[0] * xs + b + 1) / w[1]

        ax.fill_between(xs, ym, yp, color=C_LIGHT, alpha=0.55)
        ax.plot(xs, yb, color=C_DARK, lw=2.2)
        ax.plot(xs, yp, color=C_BLUE, lw=1.2, ls="--")
        ax.plot(xs, ym, color=C_PURPLE, lw=1.2, ls="--")

        ax.scatter(pos[:, 0], pos[:, 1], color=C_BLUE, s=42,
                   edgecolor="white", linewidth=1.0, zorder=4)
        ax.scatter(neg[:, 0], neg[:, 1], color=C_PURPLE, s=42,
                   edgecolor="white", linewidth=1.0, zorder=4)
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=160, facecolor="none",
                   edgecolor=C_AMBER, linewidth=2.0, zorder=3)

        margin = 2.0 / np.linalg.norm(w)
        n_sv = len(sv)
        ax.set_title(title, fontsize=11.2, fontweight="bold", pad=8)
        ax.text(0.02, 0.97,
                f"margin $= {margin:.2f}$\n#SV $= {n_sv}$",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C_GRAY, alpha=0.9))
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_xlabel(r"$x_1$", fontsize=10.5)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel(r"$x_2$", fontsize=10.5)

    fig.suptitle(r"Soft-margin SVM: $C$ trades margin width against slack penalty",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_soft_margin_C")


# ---------------------------------------------------------------------------
# Figure 3: Kernel trick -- 2D ring data lifted to 3D
# ---------------------------------------------------------------------------
def fig3_kernel_trick_3d() -> None:
    rng = np.random.default_rng(11)
    n = 90
    # Inner class
    r1 = rng.normal(0.0, 0.35, n)
    t1 = rng.uniform(0, 2 * np.pi, n)
    inner = np.column_stack([r1 * np.cos(t1), r1 * np.sin(t1)])
    # Outer ring class
    r2 = rng.normal(2.2, 0.18, n)
    t2 = rng.uniform(0, 2 * np.pi, n)
    outer = np.column_stack([r2 * np.cos(t2), r2 * np.sin(t2)])

    fig = plt.figure(figsize=(13.5, 5.6))

    # --- Left: 2D input space (not linearly separable) ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(inner[:, 0], inner[:, 1], color=C_BLUE, s=42,
                edgecolor="white", linewidth=0.9, label=r"class $+1$ (inner)")
    ax1.scatter(outer[:, 0], outer[:, 1], color=C_PURPLE, s=42,
                edgecolor="white", linewidth=0.9, label=r"class $-1$ (outer)")
    # Show that any line fails -- draw three random lines
    xs = np.linspace(-3.2, 3.2, 100)
    for slope, intercept, ls in [(0.3, 0.4, ":"), (-0.6, -0.2, "--"),
                                  (1.2, -0.8, "-.")]:
        ax1.plot(xs, slope * xs + intercept, color=C_GRAY, lw=1.0, ls=ls)
    ax1.text(0.0, -3.0,
             "no straight line separates these two classes",
             ha="center", fontsize=10.5, color=C_DARK,
             bbox=dict(boxstyle="round,pad=0.35", fc=C_LIGHT, ec=C_GRAY))
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-3.4, 3.2)
    ax1.set_xlabel(r"$x_1$", fontsize=11)
    ax1.set_ylabel(r"$x_2$", fontsize=11)
    ax1.set_title(r"Input space $\mathbb{R}^2$ (not separable)",
                  fontsize=12, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", fontsize=9.5)
    ax1.set_aspect("equal", adjustable="box")

    # --- Right: 3D feature space phi(x) = (x1, x2, x1^2 + x2^2) ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    z_inner = inner[:, 0] ** 2 + inner[:, 1] ** 2
    z_outer = outer[:, 0] ** 2 + outer[:, 1] ** 2

    # Separating plane z = 2.5 (between inner ~0 and outer ~5)
    plane_xx, plane_yy = np.meshgrid(np.linspace(-3.2, 3.2, 12),
                                      np.linspace(-3.2, 3.2, 12))
    plane_zz = np.full_like(plane_xx, 2.5)
    ax2.plot_surface(plane_xx, plane_yy, plane_zz, color=C_AMBER,
                     alpha=0.18, edgecolor=C_AMBER, linewidth=0.3)

    ax2.scatter(inner[:, 0], inner[:, 1], z_inner,
                color=C_BLUE, s=36, edgecolor="white", linewidth=0.8,
                depthshade=False)
    ax2.scatter(outer[:, 0], outer[:, 1], z_outer,
                color=C_PURPLE, s=36, edgecolor="white", linewidth=0.8,
                depthshade=False)

    ax2.set_xlabel(r"$x_1$", fontsize=10)
    ax2.set_ylabel(r"$x_2$", fontsize=10)
    ax2.set_zlabel(r"$x_1^2 + x_2^2$", fontsize=10)
    ax2.set_title(r"Feature space $\phi(x) = (x_1, x_2, x_1^2 + x_2^2)$"
                  "\n(linearly separable by a flat plane)",
                  fontsize=12, fontweight="bold", pad=8)
    ax2.view_init(elev=18, azim=-58)

    fig.suptitle("The kernel trick: lift to a higher-dim feature space, separate linearly",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    _save(fig, "fig3_kernel_trick_3d")


# ---------------------------------------------------------------------------
# Figure 4: RBF kernel decision boundary
# ---------------------------------------------------------------------------
def fig4_rbf_boundary() -> None:
    from sklearn.datasets import make_moons
    from sklearn.svm import SVC

    X, y_bin = make_moons(n_samples=200, noise=0.22, random_state=4)
    y = np.where(y_bin == 1, 1, -1)

    gammas = [0.3, 2.0, 20.0]
    titles = [
        r"$\gamma = 0.3$  (smooth, possibly underfit)",
        r"$\gamma = 2$  (balanced)",
        r"$\gamma = 20$  (sharp, possibly overfit)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)

    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 240),
                         np.linspace(y_min, y_max, 240))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "svm_div", [C_PURPLE, "#ffffff", C_BLUE], N=256)

    for ax, gamma, title in zip(axes, gammas, titles):
        clf = SVC(kernel="rbf", gamma=gamma, C=10.0).fit(X, y)
        Z = clf.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=24, cmap=cmap, alpha=0.85, vmin=-2.5,
                    vmax=2.5)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                   colors=[C_PURPLE, C_DARK, C_BLUE],
                   linewidths=[1.2, 2.0, 1.2],
                   linestyles=["--", "-", "--"])

        pos_mask = y == 1
        ax.scatter(X[pos_mask, 0], X[pos_mask, 1], color=C_BLUE, s=40,
                   edgecolor="white", linewidth=1.0, zorder=4)
        ax.scatter(X[~pos_mask, 0], X[~pos_mask, 1], color=C_PURPLE, s=40,
                   edgecolor="white", linewidth=1.0, zorder=4)
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=140, facecolor="none",
                   edgecolor=C_AMBER, linewidth=1.8, zorder=3)
        train_acc = clf.score(X, y)
        ax.set_title(title, fontsize=11.2, fontweight="bold", pad=8)
        ax.text(0.02, 0.97,
                f"#SV $= {len(sv)}$\ntrain acc $= {train_acc:.2f}$",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C_GRAY, alpha=0.9))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(r"$x_1$", fontsize=10.5)
    axes[0].set_ylabel(r"$x_2$", fontsize=10.5)

    fig.suptitle(r"RBF-kernel SVM: $\gamma$ controls the bandwidth of each support vector",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_rbf_boundary")


# ---------------------------------------------------------------------------
# Figure 5: Loss comparison -- 0/1, hinge, logistic, squared
# ---------------------------------------------------------------------------
def fig5_loss_comparison() -> None:
    m = np.linspace(-3.0, 3.0, 600)  # margin = y * f(x)
    zero_one = (m < 0).astype(float)
    hinge = np.maximum(0.0, 1.0 - m)
    logistic = np.log1p(np.exp(-m)) / np.log(2)  # /log2 so it passes (0,1)
    squared = (1.0 - m) ** 2  # squared hinge-like

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    ax.plot(m, zero_one, color=C_DARK, lw=2.6, ls="-",
            label=r"$0/1$ loss (true objective, non-convex)")
    ax.plot(m, hinge, color=C_BLUE, lw=2.6,
            label=r"hinge $\max(0, 1 - m)$  (SVM)")
    ax.plot(m, logistic, color=C_PURPLE, lw=2.4,
            label=r"logistic $\log(1 + e^{-m}) / \log 2$")
    ax.plot(m, squared, color=C_GREEN, lw=2.0, ls="--",
            label=r"squared $(1 - m)^2$")

    # Mark hinge kink at m = 1
    ax.axvline(1.0, color=C_AMBER, ls=":", lw=1.2, alpha=0.8)
    ax.scatter([1.0], [0.0], color=C_AMBER, s=70, zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.annotate(r"hinge kink at $m = 1$:"
                "\n"
                r"zero loss only when $y \cdot f(x) \geq 1$",
                xy=(1.0, 0.0), xytext=(1.2, 1.4),
                fontsize=10.5, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    # Shade misclassified region
    ax.axvspan(-3.0, 0.0, color=C_RED, alpha=0.06)
    ax.text(-2.7, 3.6, "misclassified  ($m < 0$)",
            color=C_RED, fontsize=10, fontweight="bold")

    ax.axhline(0, color=C_GRAY, lw=0.8)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.2, 4.2)
    ax.set_xlabel(r"margin  $m = y \cdot f(x)$", fontsize=11.5)
    ax.set_ylabel("loss", fontsize=11.5)
    ax.set_title("Surrogate losses for binary classification: hinge upper-bounds 0/1 and is convex",
                 fontsize=12.2, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)

    fig.tight_layout()
    _save(fig, "fig5_loss_comparison")


# ---------------------------------------------------------------------------
# Figure 6: KKT geometry
# ---------------------------------------------------------------------------
def fig6_kkt_geometry() -> None:
    """Visualise complementary slackness: which samples are SVs and why."""
    rng = np.random.default_rng(0)
    n = 24
    pos = rng.normal(loc=[1.6, 1.4], scale=0.85, size=(n, 2))
    neg = rng.normal(loc=[-1.6, -1.4], scale=0.85, size=(n, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n), -np.ones(n)])

    from sklearn.svm import SVC
    clf = SVC(kernel="linear", C=1.0).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]

    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    xs = np.linspace(-4.0, 4.0, 200)
    yb = -(w[0] * xs + b) / w[1]
    yp = -(w[0] * xs + b - 1) / w[1]
    ym = -(w[0] * xs + b + 1) / w[1]
    ax.fill_between(xs, ym, yp, color=C_LIGHT, alpha=0.55)
    ax.plot(xs, yb, color=C_DARK, lw=2.2)
    ax.plot(xs, yp, color=C_BLUE, lw=1.2, ls="--")
    ax.plot(xs, ym, color=C_PURPLE, lw=1.2, ls="--")

    # Classify each point into one of three KKT regimes
    f = X @ w + b
    margin = y * f
    # alpha values from the classifier (signed via dual_coef = alpha * y)
    alpha = np.zeros(len(X))
    sv_idx = clf.support_
    alpha[sv_idx] = np.abs(clf.dual_coef_[0])

    inactive = alpha == 0
    bound = alpha >= clf.C - 1e-6
    boundary = (~inactive) & (~bound)

    # Inactive: alpha = 0, margin > 1
    ax.scatter(X[inactive, 0], X[inactive, 1],
               c=np.where(y[inactive] == 1, C_BLUE, C_PURPLE),
               s=46, edgecolor="white", linewidth=1.0, zorder=4,
               label=r"$\alpha_i = 0$  (margin $> 1$, inactive)")
    # Boundary support vectors: 0 < alpha < C, margin == 1
    ax.scatter(X[boundary, 0], X[boundary, 1],
               c=np.where(y[boundary] == 1, C_BLUE, C_PURPLE),
               s=110, edgecolor=C_AMBER, linewidth=2.4, zorder=5,
               label=r"$0 < \alpha_i < C$  (on margin, $\xi_i = 0$)")
    # Bound support vectors: alpha = C, margin < 1 or misclassified
    ax.scatter(X[bound, 0], X[bound, 1],
               c=np.where(y[bound] == 1, C_BLUE, C_PURPLE),
               s=110, edgecolor=C_RED, linewidth=2.4, zorder=5, marker="s",
               label=r"$\alpha_i = C$  (margin violator, $\xi_i > 0$)")

    # KKT summary box
    text = (
        r"$\bf{Complementary\ slackness}$" + "\n"
        r"$\alpha_i \, [\, y_i(w^\top x_i + b) - 1 + \xi_i\,] = 0$" + "\n"
        r"$(C - \alpha_i)\, \xi_i = 0$" + "\n\n"
        r"$\Rightarrow$ only support vectors enter $w = \sum_i \alpha_i y_i x_i$"
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=10.2,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=C_GRAY, alpha=0.95))

    ax.set_xlim(-4.0, 4.2)
    ax.set_ylim(-4.0, 4.2)
    ax.set_xlabel(r"$x_1$", fontsize=11)
    ax.set_ylabel(r"$x_2$", fontsize=11)
    ax.set_title("KKT geometry: three regimes of dual variables in soft-margin SVM",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9.4, frameon=True, framealpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, "fig6_kkt_geometry")


# ---------------------------------------------------------------------------
# Figure 7: One SMO step in (alpha_1, alpha_2) plane
# ---------------------------------------------------------------------------
def fig7_smo_step() -> None:
    """One step of SMO: feasibility line, box, clipping, update."""
    C = 4.0
    # Pick a representative case: y1 != y2 (so feasibility line is alpha1 - alpha2 = const)
    # alpha1 - alpha2 = k  =>  L = max(0, -k), H = min(C, C - k)
    # We choose k = -1 so L = 1, H = 4
    k = -1.0
    L = max(0.0, -k)
    H = min(C, C - k)

    # Old alphas on the feasibility line, e.g. (0.5, 1.5) -> alpha1 - alpha2 = -1
    a_old = np.array([0.5, 1.5])
    # Unclipped optimum further along the line, e.g. (3.5, 4.5)
    a_unc = np.array([3.5, 4.5])
    # Clipped: alpha2 in [L, H] = [1, 4], so alpha2_clipped = 4, alpha1 = 4 + k = 3
    a_clipped = np.array([H + k, H])

    fig, ax = plt.subplots(figsize=(7.6, 7.0))

    # Box [0, C]^2
    box = Rectangle((0, 0), C, C, fill=False, edgecolor=C_DARK, linewidth=2.0,
                    label=r"box $[0, C]^2$")
    ax.add_patch(box)

    # Feasibility line alpha1 - alpha2 = k extended
    a2_line = np.linspace(-1.5, C + 1.5, 100)
    a1_line = a2_line + k
    ax.plot(a1_line, a2_line, color=C_GRAY, lw=1.3, ls="--",
            label=r"$y_1 \alpha_1 + y_2 \alpha_2 = \zeta$  ($y_1 \neq y_2$)")

    # Clipping interval [L, H] on the line (in alpha2)
    a2_LH = np.linspace(L, H, 50)
    a1_LH = a2_LH + k
    ax.plot(a1_LH, a2_LH, color=C_BLUE, lw=4.0, alpha=0.6,
            label=rf"feasible segment  $\alpha_2 \in [L, H] = [{L:.0f}, {H:.0f}]$")

    # Old, unconstrained, clipped points
    ax.scatter(*a_old, color=C_DARK, s=120, zorder=5, edgecolor="white",
               linewidth=1.5, label=r"$\alpha^{old}$")
    ax.scatter(*a_unc, color=C_PURPLE, s=120, zorder=5, edgecolor="white",
               linewidth=1.5, marker="X",
               label=r"$\alpha_2^{new, unc} = \alpha_2^{old} + y_2(E_1 - E_2)/\eta$")
    ax.scatter(*a_clipped, color=C_GREEN, s=160, zorder=6, edgecolor="white",
               linewidth=1.6, marker="*",
               label=r"$\alpha^{new}$ after clipping to $[L, H]$")

    # Update arrow
    arrow = FancyArrowPatch(tuple(a_old), tuple(a_clipped),
                            arrowstyle="->", mutation_scale=15,
                            color=C_AMBER, lw=2.0, zorder=4)
    ax.add_patch(arrow)

    # L and H markers on alpha2 axis
    ax.axhline(L, color=C_BLUE, lw=0.8, ls=":", alpha=0.7, xmax=0.55)
    ax.axhline(H, color=C_BLUE, lw=0.8, ls=":", alpha=0.7, xmax=0.55)
    ax.text(-0.45, L, r"$L$", fontsize=11, color=C_BLUE, va="center",
            fontweight="bold")
    ax.text(-0.45, H, r"$H$", fontsize=11, color=C_BLUE, va="center",
            fontweight="bold")
    ax.text(C + 0.05, -0.3, r"$C$", fontsize=11, color=C_DARK,
            fontweight="bold")
    ax.text(-0.4, C + 0.05, r"$C$", fontsize=11, color=C_DARK,
            fontweight="bold")

    ax.set_xlim(-0.8, C + 1.6)
    ax.set_ylim(-0.8, C + 1.6)
    ax.set_xlabel(r"$\alpha_1$", fontsize=12)
    ax.set_ylabel(r"$\alpha_2$", fontsize=12)
    ax.set_title("One SMO step: optimise $(\\alpha_1, \\alpha_2)$ along the feasibility line, then clip to $[L, H]$",
                 fontsize=11.8, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9.0, frameon=True, framealpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, "fig7_smo_step")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")

    print("\n[1/7] fig1_max_margin ...")
    fig1_max_margin()
    print("\n[2/7] fig2_soft_margin_C ...")
    fig2_soft_margin_C()
    print("\n[3/7] fig3_kernel_trick_3d ...")
    fig3_kernel_trick_3d()
    print("\n[4/7] fig4_rbf_boundary ...")
    fig4_rbf_boundary()
    print("\n[5/7] fig5_loss_comparison ...")
    fig5_loss_comparison()
    print("\n[6/7] fig6_kkt_geometry ...")
    fig6_kkt_geometry()
    print("\n[7/7] fig7_smo_step ...")
    fig7_smo_step()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
