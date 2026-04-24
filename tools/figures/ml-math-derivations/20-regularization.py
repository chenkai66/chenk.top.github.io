"""
Figure generation script for ML Math Derivations Part 20:
Regularization and Model Selection (series finale).

Generates 7 figures used in BOTH the EN and ZH versions of the article.
Each figure isolates a single regularization / model-selection concept so
that the underlying mathematics becomes visible at a glance.

Figures:
    fig1_l1_vs_l2_geometry      Classic constraint-region picture: the L2
                                ball (circle) vs. the L1 ball (diamond)
                                intersected with elliptical loss contours.
                                Shows why L1 selects a corner -> sparsity.
    fig2_lasso_path             Coefficient path of LASSO as lambda decreases:
                                features enter the model one at a time and
                                small coefficients stay pinned at zero.
    fig3_kfold_cv               K-fold cross-validation visual: the data is
                                partitioned into K equal folds and each fold
                                takes a turn as the validation set.
    fig4_complexity_curves      Train / validation / test error vs. model
                                complexity, with the bias-variance regions
                                and the optimal-complexity sweet spot.
    fig5_aic_bic_vs_cv          Side-by-side comparison of AIC, BIC and
                                K-fold CV scores selecting polynomial degree.
    fig6_dropout_concept        Dropout as random sub-network sampling: a
                                dense MLP shown together with three sampled
                                thinned networks that share weights.
    fig7_double_descent         Modern double-descent curve: classical U
                                shape, the interpolation peak, and the
                                second descent in the over-parameterised
                                regime.

Usage:
    python3 scripts/figures/ml-math-derivations/20-regularization.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

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
C_RED = COLORS["danger"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "20-Regularization-and-Model-Selection"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "20-正则化与模型选择"
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


def _arrow(ax, p0, p1, *, color=C_DARK, lw=1.3, style="-|>", mut=14, alpha=1.0,
           rad=0.0, ls="-"):
    arr = FancyArrowPatch(
        p0, p1,
        arrowstyle=style,
        mutation_scale=mut,
        color=color, lw=lw, alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=ls,
        zorder=2,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: L1 vs L2 constraint regions (the classic picture)
# ---------------------------------------------------------------------------
def fig1_l1_vs_l2_geometry() -> None:
    """L1 (diamond) vs L2 (circle) constraint regions intersected with the
    elliptical loss contours of an unregularised LS solution off-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))

    # Unregularised LS optimum (some w*)
    w_star = np.array([2.4, 1.6])
    # Hessian of the quadratic loss (positive definite, mildly elongated)
    H = np.array([[1.0, 0.55], [0.55, 1.4]])

    # Loss contours
    grid = np.linspace(-3.5, 3.5, 400)
    X, Y = np.meshgrid(grid, grid)
    pts = np.stack([X - w_star[0], Y - w_star[1]], axis=-1)
    Z = np.einsum("...i,ij,...j->...", pts, H, pts)
    levels = [0.4, 1.2, 2.4, 4.2, 6.5, 9.5, 13.0]

    # Constraint radius t — picked so the intersection is visually clean
    t = 1.5

    # Compute the constrained optimum for L1 and L2 by simple grid search
    def constrained_min(constraint):
        gx = np.linspace(-2.0, 2.0, 1201)
        XX, YY = np.meshgrid(gx, gx)
        pts2 = np.stack([XX - w_star[0], YY - w_star[1]], axis=-1)
        ZZ = np.einsum("...i,ij,...j->...", pts2, H, pts2)
        if constraint == "l1":
            mask = np.abs(XX) + np.abs(YY) <= t + 1e-9
        else:
            mask = XX**2 + YY**2 <= t**2 + 1e-9
        ZZ = np.where(mask, ZZ, np.inf)
        idx = np.unravel_index(np.argmin(ZZ), ZZ.shape)
        return XX[idx], YY[idx]

    w_l2 = constrained_min("l2")
    w_l1 = constrained_min("l1")

    titles = [
        r"$L_2$ constraint:  $\|\mathbf{w}\|_2 \leq t$  (smooth ball)",
        r"$L_1$ constraint:  $\|\mathbf{w}\|_1 \leq t$  (corners on axes)",
    ]

    for ax, kind, title, w_opt in zip(
        axes, ["l2", "l1"], titles, [w_l2, w_l1]
    ):
        # Contours
        ax.contour(X, Y, Z, levels=levels, colors=C_GRAY, linewidths=0.9,
                   alpha=0.85, zorder=1)
        # Outermost contour highlighted
        ax.contour(X, Y, Z, levels=[Z[
            np.argmin(np.abs(grid - w_opt[1])),
            np.argmin(np.abs(grid - w_opt[0]))
        ]], colors=[C_BLUE], linewidths=2.0, zorder=2)

        # Constraint region
        if kind == "l2":
            circ = Circle((0, 0), t, facecolor=C_PURPLE, edgecolor=C_PURPLE,
                          alpha=0.18, lw=2.0, zorder=3)
            ax.add_patch(circ)
            ax.add_patch(Circle((0, 0), t, facecolor="none",
                                edgecolor=C_PURPLE, lw=2.0, zorder=4))
        else:
            diamond = Polygon(
                [(t, 0), (0, t), (-t, 0), (0, -t)],
                closed=True, facecolor=C_PURPLE, edgecolor=C_PURPLE,
                alpha=0.18, lw=2.0, zorder=3,
            )
            ax.add_patch(diamond)
            ax.add_patch(Polygon(
                [(t, 0), (0, t), (-t, 0), (0, -t)],
                closed=True, facecolor="none", edgecolor=C_PURPLE,
                lw=2.0, zorder=4,
            ))

        # Unconstrained optimum
        ax.plot(*w_star, "o", color=C_AMBER, ms=11, zorder=5,
                markeredgecolor=C_DARK, markeredgewidth=1.0)
        ax.annotate(
            r"$\hat{\mathbf{w}}_{\mathrm{LS}}$",
            xy=w_star, xytext=(w_star[0] + 0.25, w_star[1] + 0.30),
            fontsize=12.5, color=C_DARK, fontweight="bold",
        )
        # Constrained solution
        ax.plot(*w_opt, "o", color=C_GREEN, ms=11, zorder=6,
                markeredgecolor=C_DARK, markeredgewidth=1.0)
        label = (r"$\hat{\mathbf{w}}_{\mathrm{ridge}}$" if kind == "l2"
                 else r"$\hat{\mathbf{w}}_{\mathrm{lasso}}$")
        # Label placement
        dx, dy = (-0.55, 0.45) if kind == "l2" else (0.20, 0.45)
        ax.annotate(
            label, xy=w_opt, xytext=(w_opt[0] + dx, w_opt[1] + dy),
            fontsize=12.5, color=C_GREEN, fontweight="bold",
        )

        # Highlight that L1 sits ON an axis (sparse)
        if kind == "l1" and abs(w_opt[0]) < 1e-2:
            ax.annotate(
                r"$w_1=0$  (sparse!)",
                xy=w_opt, xytext=(-2.7, -2.6),
                fontsize=11, color=C_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.2),
            )
        elif kind == "l2":
            ax.text(-3.2, -3.0, "Solution is generically interior\n"
                                "to the open positive orthant\n"
                                r"(no $w_j$ exactly zero)",
                    fontsize=10.0, color=C_PURPLE)

        # Axes
        ax.axhline(0, color=C_DARK, lw=0.6, alpha=0.6)
        ax.axvline(0, color=C_DARK, lw=0.6, alpha=0.6)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$w_1$", fontsize=11.5)
        ax.set_ylabel(r"$w_2$", fontsize=11.5)
        ax.set_title(title, fontsize=12.5, pad=8)

    fig.suptitle(
        "Why $L_1$ Produces Sparsity:  Loss Contours Hit a Corner of the Diamond",
        fontsize=13.5, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig1_l1_vs_l2_geometry")


# ---------------------------------------------------------------------------
# Figure 2: LASSO coefficient path
# ---------------------------------------------------------------------------
def fig2_lasso_path() -> None:
    """Coefficient path of LASSO as the regularisation strength decreases.

    We solve the LASSO with coordinate descent on a small synthetic problem
    where only 3 of 8 features are truly relevant. As lambda shrinks, the
    relevant features "switch on" one at a time while the irrelevant ones
    stay pinned at zero (sparsity)."""
    rng = np.random.default_rng(20)
    N, D = 120, 8
    X = rng.standard_normal((N, D))
    # Decorrelate columns somewhat
    X = (X - X.mean(0)) / X.std(0)
    w_true = np.zeros(D)
    w_true[[0, 2, 5]] = [2.5, -1.8, 1.2]
    y = X @ w_true + rng.standard_normal(N) * 0.7

    # Coordinate-descent LASSO
    def lasso_cd(lam, n_iter=400):
        w = np.zeros(D)
        XtX_diag = (X**2).sum(axis=0)
        r = y.copy()
        for _ in range(n_iter):
            for j in range(D):
                # Remove j-th contribution
                r += X[:, j] * w[j]
                rho = X[:, j] @ r
                # Soft-thresholding
                if rho > lam:
                    w_j = (rho - lam) / XtX_diag[j]
                elif rho < -lam:
                    w_j = (rho + lam) / XtX_diag[j]
                else:
                    w_j = 0.0
                w[j] = w_j
                r -= X[:, j] * w[j]
        return w

    lams = np.logspace(2.4, -1.0, 60)
    paths = np.array([lasso_cd(l) for l in lams])

    fig, ax = plt.subplots(figsize=(10.6, 5.4))

    palette = [C_BLUE, C_GRAY, C_PURPLE, C_GRAY, C_GRAY,
               C_GREEN, C_GRAY, C_GRAY]
    is_relevant = [True, False, True, False, False, True, False, False]

    for j in range(D):
        ax.plot(
            np.log10(lams), paths[:, j],
            color=palette[j],
            lw=2.4 if is_relevant[j] else 1.1,
            alpha=1.0 if is_relevant[j] else 0.55,
            label=(rf"$w_{{{j+1}}}$  (true={w_true[j]:.1f})"
                   if is_relevant[j] else None),
        )
        # Marker at the right edge
        ax.plot(np.log10(lams[-1]), paths[-1, j], "o",
                color=palette[j], ms=5,
                alpha=1.0 if is_relevant[j] else 0.5)

    # True-value reference dots
    for j in range(D):
        if is_relevant[j]:
            ax.axhline(w_true[j], color=palette[j], ls=":", lw=0.9, alpha=0.55)

    ax.axhline(0, color=C_DARK, lw=0.7, alpha=0.7)
    ax.set_xlabel(r"$\log_{10}\lambda$  (regularisation strength)  $\longrightarrow$  weaker reg.",
                  fontsize=11.5)
    ax.set_ylabel(r"Coefficient  $\hat{w}_j(\lambda)$", fontsize=11.5)
    ax.set_title(
        "LASSO Coefficient Path:  Features Enter One at a Time as $\\lambda$ Shrinks",
        fontsize=12.8, pad=10,
    )
    ax.invert_xaxis()  # large lambda on the left (more regularised)
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    # Annotate the "sparsity floor": irrelevant coefficients stay at 0
    ax.annotate(
        "Irrelevant features\nstay pinned at 0",
        xy=(np.log10(lams[10]), 0.05),
        xytext=(np.log10(lams[6]), 1.15),
        fontsize=10.5, color=C_GRAY, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.0),
    )
    ax.annotate(
        "Relevant features\n\"switch on\"",
        xy=(np.log10(lams[35]), paths[35, 0]),
        xytext=(np.log10(lams[42]), 2.7),
        fontsize=10.5, color=C_BLUE, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.0),
    )
    fig.tight_layout()
    _save(fig, "fig2_lasso_path")


# ---------------------------------------------------------------------------
# Figure 3: K-fold cross-validation
# ---------------------------------------------------------------------------
def fig3_kfold_cv() -> None:
    """Visualise K-fold CV: each row is a fold; coloured tile = validation,
    grey tiles = training. K=5 with N=20 samples."""
    K = 5
    N = 20
    fig, ax = plt.subplots(figsize=(10.6, 4.6))

    fold_size = N // K
    cell_w = 1.0
    cell_h = 0.7
    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_RED]

    for k in range(K):
        for i in range(N):
            in_val = (i // fold_size) == k
            color = palette[k] if in_val else C_LIGHT
            edge = C_DARK if in_val else C_GRAY
            r = Rectangle(
                (i * cell_w, -k * (cell_h + 0.18)),
                cell_w * 0.92, cell_h,
                facecolor=color, edgecolor=edge,
                lw=1.2 if in_val else 0.6,
            )
            ax.add_patch(r)
        # Row label
        ax.text(
            -0.4, -k * (cell_h + 0.18) + cell_h / 2,
            f"Fold {k+1}", ha="right", va="center",
            fontsize=11, fontweight="bold", color=palette[k],
        )
        # Right-side metric box
        ax.text(
            N * cell_w + 0.5,
            -k * (cell_h + 0.18) + cell_h / 2,
            rf"$\hat R_{{\mathrm{{val}},{k+1}}}$",
            ha="left", va="center", fontsize=11, color=palette[k],
            fontweight="bold",
        )

    # Top sample indices
    for i in range(N):
        ax.text(
            i * cell_w + cell_w * 0.46, cell_h + 0.18,
            f"{i+1}", ha="center", va="bottom", fontsize=8.5, color=C_DARK,
        )
    ax.text(
        N * cell_w / 2, cell_h + 0.75,
        "Sample index  $i = 1, \\ldots, N$", ha="center",
        fontsize=11, color=C_DARK,
    )

    # Legend tiles below
    base_y = -K * (cell_h + 0.18) - 0.3
    ax.add_patch(Rectangle((0.0, base_y), 0.7, 0.45,
                           facecolor=C_LIGHT, edgecolor=C_GRAY))
    ax.text(0.85, base_y + 0.22, "training fold",
            va="center", fontsize=10.5, color=C_DARK)
    ax.add_patch(Rectangle((3.6, base_y), 0.7, 0.45,
                           facecolor=C_BLUE, edgecolor=C_DARK))
    ax.text(4.45, base_y + 0.22, "validation fold",
            va="center", fontsize=10.5, color=C_DARK)

    # Average formula on the right
    ax.text(
        N * cell_w + 0.5, base_y + 0.22,
        r"$\hat R_{\mathrm{CV}} = \frac{1}{K}\sum_{k=1}^K \hat R_{\mathrm{val},k}$",
        va="center", fontsize=12.5, color=C_DARK,
    )

    ax.set_xlim(-1.2, N * cell_w + 4.0)
    ax.set_ylim(base_y - 0.4, cell_h + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"K-Fold Cross-Validation (K={K}):  Every Sample Validates Exactly Once",
        fontsize=12.8, pad=8,
    )
    fig.tight_layout()
    _save(fig, "fig3_kfold_cv")


# ---------------------------------------------------------------------------
# Figure 4: Train / validation / test error vs model complexity
# ---------------------------------------------------------------------------
def fig4_complexity_curves() -> None:
    """Bias-variance picture: training error decreases monotonically while
    validation/test error has a U shape. Labels for under/overfitting and the
    optimal-complexity sweet spot."""
    cx = np.linspace(0.5, 10, 400)
    # Bias^2 falls with complexity; variance grows; noise is constant
    bias2 = 1.6 / (cx ** 1.4)
    variance = 0.025 * (cx ** 1.6)
    noise = 0.18 * np.ones_like(cx)
    test = bias2 + variance + noise
    train = noise + 1.2 / (cx ** 2.0) - 0.04 * np.minimum(cx, 8) * 0.05

    fig, ax = plt.subplots(figsize=(10.6, 5.4))

    ax.plot(cx, train, color=C_BLUE, lw=2.5, label="Training error")
    ax.plot(cx, test, color=C_AMBER, lw=2.5, label="Test (generalisation) error")
    ax.plot(cx, bias2, color=C_PURPLE, lw=1.5, ls="--",
            label=r"Bias$^2$")
    ax.plot(cx, variance, color=C_GREEN, lw=1.5, ls="--",
            label="Variance")
    ax.axhline(noise[0], color=C_GRAY, lw=1.2, ls=":",
               label=r"Irreducible noise $\sigma^2$")

    # Optimum
    i_opt = int(np.argmin(test))
    ax.axvline(cx[i_opt], color=C_DARK, lw=1.0, ls=":")
    ax.plot(cx[i_opt], test[i_opt], "o", color=C_AMBER, ms=11,
            markeredgecolor=C_DARK, markeredgewidth=1.0, zorder=5)
    ax.annotate(
        "Sweet spot:\nbias = variance",
        xy=(cx[i_opt], test[i_opt]),
        xytext=(cx[i_opt] + 1.4, test[i_opt] + 0.45),
        fontsize=11, color=C_DARK,
        arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.0),
    )

    # Region shading
    ax.axvspan(cx[0], cx[i_opt], alpha=0.06, color=C_PURPLE)
    ax.axvspan(cx[i_opt], cx[-1], alpha=0.06, color=C_GREEN)
    ax.text(cx[i_opt] / 2 + 0.3, test.max() * 0.95,
            "Underfitting\n(high bias)", ha="center",
            fontsize=11.5, color=C_PURPLE, fontweight="bold")
    ax.text((cx[i_opt] + cx[-1]) / 2, test.max() * 0.95,
            "Overfitting\n(high variance)", ha="center",
            fontsize=11.5, color=C_GREEN, fontweight="bold")

    ax.set_xlabel("Model complexity  (capacity, parameters, depth, ...)",
                  fontsize=11.5)
    ax.set_ylabel("Error", fontsize=11.5)
    ax.set_title(
        "The Bias-Variance Tradeoff:  Test Error = Bias$^2$ + Variance + Noise",
        fontsize=12.8, pad=10,
    )
    ax.set_ylim(0, max(test.max(), train.max()) * 1.15)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig4_complexity_curves")


# ---------------------------------------------------------------------------
# Figure 5: AIC / BIC vs CV for selecting polynomial degree
# ---------------------------------------------------------------------------
def fig5_aic_bic_vs_cv() -> None:
    """Compare AIC, BIC, and 5-fold CV for selecting polynomial degree on a
    synthetic regression task with a degree-3 truth."""
    rng = np.random.default_rng(11)
    N = 60
    x = np.linspace(-1, 1, N)
    f_true = 2 * x**3 - 1.2 * x + 0.4
    sigma = 0.35
    y = f_true + rng.standard_normal(N) * sigma

    degrees = np.arange(1, 13)

    def fit_poly(xtr, ytr, d):
        return np.polyfit(xtr, ytr, d)

    def neg_log_lik(y_pred, y_true):
        # Gaussian log-likelihood with MLE sigma
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        sig2_hat = rss / n
        return 0.5 * n * (np.log(2 * np.pi * sig2_hat) + 1)

    aic, bic, cv = [], [], []
    for d in degrees:
        # Full-data fit for AIC/BIC
        coef = fit_poly(x, y, d)
        y_pred = np.polyval(coef, x)
        nll = neg_log_lik(y_pred, y)
        p = d + 1  # +1 for intercept; +1 for sigma omitted (cancels in selection)
        aic.append(2 * nll + 2 * p)
        bic.append(2 * nll + p * np.log(N))

        # 5-fold CV MSE
        K = 5
        idx = np.arange(N)
        rng.shuffle(idx)
        folds = np.array_split(idx, K)
        cv_mse = []
        for k in range(K):
            val = folds[k]
            tr = np.concatenate([folds[i] for i in range(K) if i != k])
            c = fit_poly(x[tr], y[tr], d)
            yv = np.polyval(c, x[val])
            cv_mse.append(np.mean((y[val] - yv) ** 2))
        cv.append(np.mean(cv_mse))

    aic = np.array(aic)
    bic = np.array(bic)
    cv = np.array(cv)

    # Normalise each curve to its minimum for visual comparison
    def norm(z):
        return (z - z.min()) / (z.max() - z.min() + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})

    # --- Left: the three criteria ---
    ax = axes[0]
    ax.plot(degrees, norm(aic), "-o", color=C_BLUE, lw=2.0, ms=6, label="AIC")
    ax.plot(degrees, norm(bic), "-s", color=C_PURPLE, lw=2.0, ms=6, label="BIC")
    ax.plot(degrees, norm(cv), "-^", color=C_GREEN, lw=2.0, ms=6,
            label="5-fold CV MSE")

    for arr, color, marker in [(aic, C_BLUE, "o"), (bic, C_PURPLE, "s"),
                               (cv, C_GREEN, "^")]:
        d_star = degrees[np.argmin(arr)]
        ax.axvline(d_star, color=color, ls=":", lw=1.0, alpha=0.7)
        ax.plot(d_star, 0, marker, color=color, ms=11,
                markeredgecolor=C_DARK, markeredgewidth=1.0, zorder=5)

    ax.axvline(3, color=C_AMBER, lw=1.5, ls="--",
               label="True degree = 3", alpha=0.85)
    ax.set_xlabel("Polynomial degree  $p$", fontsize=11.5)
    ax.set_ylabel("Score (normalised, lower is better)", fontsize=11.5)
    ax.set_title("AIC vs BIC vs Cross-Validation", fontsize=12.5, pad=8)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_xticks(degrees)

    # --- Right: data + chosen fit (BIC pick, usually closest to truth) ---
    ax2 = axes[1]
    d_bic = degrees[np.argmin(bic)]
    coef = fit_poly(x, y, d_bic)
    xx = np.linspace(-1, 1, 400)
    ax2.plot(x, y, "o", color=C_GRAY, ms=5.5, alpha=0.7, label="Data")
    ax2.plot(xx, 2 * xx**3 - 1.2 * xx + 0.4, color=C_AMBER, lw=2.0,
             label=r"Truth (degree 3)")
    ax2.plot(xx, np.polyval(coef, xx), color=C_PURPLE, lw=2.2,
             label=f"BIC pick (degree {d_bic})")
    ax2.set_xlabel(r"$x$", fontsize=11.5)
    ax2.set_ylabel(r"$y$", fontsize=11.5)
    ax2.set_title("Selected Model on the Data", fontsize=12.5, pad=8)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)

    fig.suptitle(
        "Model Selection: BIC Penalises Complexity Most ($p\\,\\log N > 2p$ for $N\\!\\geq\\!8$)",
        fontsize=13.0, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig5_aic_bic_vs_cv")


# ---------------------------------------------------------------------------
# Figure 6: Dropout as random sub-network sampling
# ---------------------------------------------------------------------------
def fig6_dropout_concept() -> None:
    """Show a dense MLP on the left and three dropped-out thinned networks on
    the right, all sharing the same weight matrix."""
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.6))

    layer_sizes = [4, 6, 6, 3]
    layer_x = [0.0, 1.5, 3.0, 4.5]

    def layer_ys(n):
        # Centre-aligned vertical positions
        gap = 0.7
        ys = np.arange(n) * gap
        return ys - ys.mean()

    rng = np.random.default_rng(2026)
    keep_prob = 0.55
    masks_list = [None]
    for _ in range(3):
        masks = []
        for n in layer_sizes:
            if n == layer_sizes[0] or n == layer_sizes[-1]:
                masks.append(np.ones(n, dtype=bool))  # don't drop in/out
            else:
                masks.append(rng.random(n) < keep_prob)
                # ensure at least 2 alive per hidden layer
                if masks[-1].sum() < 2:
                    masks[-1][:2] = True
        masks_list.append(masks)

    titles = [
        "Full network",
        "Thinned sample 1",
        "Thinned sample 2",
        "Thinned sample 3",
    ]
    for ax, masks, title in zip(axes, masks_list, titles):
        # Edges
        for li in range(len(layer_sizes) - 1):
            n0, n1 = layer_sizes[li], layer_sizes[li + 1]
            ys0 = layer_ys(n0)
            ys1 = layer_ys(n1)
            for i in range(n0):
                for j in range(n1):
                    alive_i = True if masks is None else masks[li][i]
                    alive_j = True if masks is None else masks[li + 1][j]
                    on = alive_i and alive_j
                    color = C_DARK if on else C_LIGHT
                    lw = 0.7 if on else 0.3
                    alpha = 0.55 if on else 0.30
                    ax.plot(
                        [layer_x[li], layer_x[li + 1]], [ys0[i], ys1[j]],
                        color=color, lw=lw, alpha=alpha, zorder=1,
                    )
        # Nodes
        layer_colors = [C_BLUE, C_PURPLE, C_PURPLE, C_GREEN]
        for li, n in enumerate(layer_sizes):
            ys = layer_ys(n)
            for i, y in enumerate(ys):
                alive = True if masks is None else masks[li][i]
                color = layer_colors[li] if alive else "white"
                edge = C_DARK if alive else C_GRAY
                lw = 1.2 if alive else 1.0
                circ = Circle((layer_x[li], y), 0.18,
                              facecolor=color, edgecolor=edge, lw=lw, zorder=3)
                ax.add_patch(circ)
                if not alive:
                    # Cross out dropped neuron
                    ax.plot([layer_x[li] - 0.13, layer_x[li] + 0.13],
                            [y - 0.13, y + 0.13],
                            color=C_RED, lw=1.4, zorder=4)
                    ax.plot([layer_x[li] - 0.13, layer_x[li] + 0.13],
                            [y + 0.13, y - 0.13],
                            color=C_RED, lw=1.4, zorder=4)

        ax.set_xlim(-0.6, 5.1)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12.0, pad=4)

    fig.suptitle(
        r"Dropout = Sampling a Random Sub-Network at Every Mini-Batch  "
        r"(keep prob $1-p$, weights shared across all $2^M$ samples)",
        fontsize=12.8, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_dropout_concept")


# ---------------------------------------------------------------------------
# Figure 7: Double descent (modern phenomenon)
# ---------------------------------------------------------------------------
def fig7_double_descent() -> None:
    """The double-descent curve: classical U up to the interpolation peak,
    then a second descent in the over-parameterised regime."""
    # x = parameters / samples ratio
    x = np.linspace(0.05, 5.0, 800)
    N = 1.0  # normalise

    # Classical U test error (peaked towards interpolation threshold)
    u = 0.55 / (x ** 0.7) + 0.04 * x ** 0.6
    # Singularity at x=1 modeled by a Lorentzian-like spike
    spike = 1.4 / ((x - 1.0) ** 2 + 0.012)
    spike = spike / spike.max() * 1.6
    # Modern descent: smooth declining tail beyond x>1
    modern = 0.32 + 0.18 / (x ** 0.7)

    # Combine: classical curve dominates left, modern dominates far right,
    # spike sits at the interpolation threshold.
    blend = 1.0 / (1.0 + np.exp(-(x - 1.0) * 4.5))  # logistic 0->1
    test_err = (1 - blend) * u + blend * modern + spike

    train_err = np.where(x < 1.0, 0.45 * (1 - x), 0.0)
    train_err = np.maximum(train_err - 0.02, 0.0)

    fig, ax = plt.subplots(figsize=(10.8, 5.4))

    ax.plot(x, test_err, color=C_AMBER, lw=2.6, label="Test error")
    ax.plot(x, train_err, color=C_BLUE, lw=2.4, label="Training error")

    # Interpolation threshold
    ax.axvline(1.0, color=C_DARK, lw=1.0, ls=":", alpha=0.85)
    ax.text(
        1.02, ax.get_ylim()[1] * 0.92,
        "Interpolation\nthreshold  ($p=N$)",
        fontsize=10.5, color=C_DARK,
    )

    # Region shading
    ax.axvspan(0.05, 1.0, alpha=0.06, color=C_PURPLE)
    ax.axvspan(1.0, 5.0, alpha=0.06, color=C_GREEN)
    ax.text(0.55, test_err.max() * 0.6,
            "Classical regime\n(under-parameterised)",
            ha="center", fontsize=11, color=C_PURPLE, fontweight="bold")
    ax.text(3.2, test_err.max() * 0.6,
            "Modern regime\n(over-parameterised)",
            ha="center", fontsize=11, color=C_GREEN, fontweight="bold")

    # Annotate first U minimum and second descent
    i_classical = int(np.argmin(u + 0 * spike))
    ax.annotate(
        "Classical U\nminimum",
        xy=(x[i_classical], u[i_classical] * 0.95),
        xytext=(x[i_classical] - 0.05, u[i_classical] + 0.9),
        fontsize=10.5, color=C_PURPLE, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=1.0),
    )
    i_far = -1
    ax.annotate(
        "Second descent:\nbigger model = better!",
        xy=(x[i_far], test_err[i_far]),
        xytext=(x[i_far] - 1.0, test_err[i_far] + 1.2),
        fontsize=10.5, color=C_GREEN, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.0),
    )

    ax.set_xlabel("Model size  /  data size   ($p / N$)", fontsize=11.5)
    ax.set_ylabel("Error", fontsize=11.5)
    ax.set_title(
        "Double Descent: Test Error Spikes at $p=N$, Then Falls Again",
        fontsize=12.8, pad=10,
    )
    ax.set_ylim(0, min(test_err.max() * 1.05, 4.5))
    ax.legend(loc="upper right", frameon=True, fontsize=10.5)
    fig.tight_layout()
    _save(fig, "fig7_double_descent")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 20 (Regularization & Model Selection) figures ...")
    fig1_l1_vs_l2_geometry()
    fig2_lasso_path()
    fig3_kfold_cv()
    fig4_complexity_curves()
    fig5_aic_bic_vs_cv()
    fig6_dropout_concept()
    fig7_double_descent()
    print("Done.")


if __name__ == "__main__":
    main()
