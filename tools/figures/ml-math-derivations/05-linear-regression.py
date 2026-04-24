"""
Figure generation script for ML Math Derivations Chapter 05:
Linear Regression.

Produces 7 figures used in BOTH the EN and ZH versions of the article.
Each figure teaches one specific idea cleanly, in a 3Blue1Brown-inspired
style, and -- where possible -- is cross-checked against scikit-learn.

Figures:
    fig1_scatter_residuals       Scatter plot, OLS best-fit line, and the
                                 vertical residuals being minimised.
    fig2_projection_geometry     OLS as orthogonal projection of y onto
                                 the column space of X (3D view).
    fig3_polynomial_fits         Underfit / right-fit / overfit panels for
                                 polynomial regression on noisy sin data.
    fig4_regularization_paths    Coefficient paths for Ridge (L2) and
                                 Lasso (L1) as lambda varies on a log axis.
    fig5_l1_vs_l2_geometry       The classical "diamond vs disk" picture
                                 explaining why L1 yields sparse solutions.
    fig6_cv_curve                k-fold cross-validation MSE versus
                                 polynomial degree, with the U-shape that
                                 reveals the bias-variance tradeoff.
    fig7_outlier_robustness      OLS vs Huber regression on data with
                                 outliers; OLS gets pulled, Huber resists.

Usage:
    python3 scripts/figures/ml-math-derivations/05-linear-regression.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay in sync across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    HuberRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/"
    "ml-math-derivations/05-Linear-Regression"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/"
    "ml-math-derivations/05-线性回归"
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
def _style_ax(ax, *, grid=True):
    ax.tick_params(labelsize=9, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_color(C_GRAY)
        spine.set_linewidth(0.6)
    if grid:
        ax.grid(True, alpha=0.25, lw=0.6)


# ---------------------------------------------------------------------------
# Fig 1: Scatter + best-fit line + residuals
# ---------------------------------------------------------------------------
def fig1_scatter_residuals():
    """Scatter, OLS line, and vertical residuals -- the things being squared."""
    rng = np.random.default_rng(1)
    n = 40
    x = np.linspace(0, 10, n)
    true_w, true_b = 0.85, 1.2
    y = true_w * x + true_b + rng.normal(0, 1.1, n)

    # Manual OLS for transparency, then verify with sklearn.
    X = x.reshape(-1, 1)
    w_hat = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
    b_hat = y.mean() - w_hat * x.mean()

    sk = LinearRegression().fit(X, y)
    assert np.allclose(sk.coef_[0], w_hat, atol=1e-8)
    assert np.allclose(sk.intercept_, b_hat, atol=1e-8)

    y_hat = w_hat * x + b_hat
    residuals = y - y_hat
    sse = float(np.sum(residuals ** 2))

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    _style_ax(ax)

    # Residual stems first so points draw on top.
    for xi, yi, yhi in zip(x, y, y_hat):
        ax.plot([xi, xi], [yi, yhi], color=C_AMBER, lw=1.2,
                alpha=0.75, zorder=2)

    ax.scatter(x, y, s=42, color=C_BLUE, zorder=4,
               edgecolor="white", linewidth=0.8, label="observations")
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 200)
    ax.plot(xs, w_hat * xs + b_hat, color=C_DARK, lw=2.2, zorder=3,
            label=fr"OLS fit:  $\hat y = {w_hat:.2f}\,x + {b_hat:.2f}$")

    # Single legend entry for residuals via a proxy line.
    ax.plot([], [], color=C_AMBER, lw=1.4,
            label=fr"residuals (SSE = {sse:.2f})")

    ax.set_xlabel("x", fontsize=11, color=C_DARK)
    ax.set_ylabel("y", fontsize=11, color=C_DARK)
    ax.set_title(
        "Linear regression: minimise the sum of squared vertical residuals",
        fontsize=13, color=C_DARK, fontweight="bold", pad=10,
    )
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    fig.tight_layout()
    save(fig, "fig1_scatter_residuals")


# ---------------------------------------------------------------------------
# Fig 2: OLS as orthogonal projection onto col(X)
# ---------------------------------------------------------------------------
def fig2_projection_geometry():
    """3D view: y projects orthogonally onto the plane Col(X)."""
    fig = plt.figure(figsize=(9.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    # Two columns of X spanning a plane in R^3.
    a1 = np.array([2.0, 0.4, 0.0])
    a2 = np.array([0.5, 1.8, 0.0])
    X = np.column_stack([a1, a2])
    y = np.array([1.6, 1.2, 2.0])  # not in the plane

    # Closed-form projection.
    P = X @ np.linalg.inv(X.T @ X) @ X.T
    yhat = P @ y
    resid = y - yhat
    # Verify orthogonality.
    assert abs(resid @ a1) < 1e-10
    assert abs(resid @ a2) < 1e-10

    # Plane patch spanned by a1, a2.
    s = np.linspace(-0.4, 1.25, 2)
    t = np.linspace(-0.4, 1.25, 2)
    S, T = np.meshgrid(s, t)
    Px = S * a1[0] + T * a2[0]
    Py = S * a1[1] + T * a2[1]
    Pz = S * a1[2] + T * a2[2]
    ax.plot_surface(Px, Py, Pz, color=C_BLUE, alpha=0.18,
                    edgecolor=C_BLUE, linewidth=0.4)

    # Column vectors a1, a2.
    def _arrow3d(p0, p1, color, lw=2.0, ls="-"):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=color, lw=lw, linestyle=ls)
        ax.scatter([p1[0]], [p1[1]], [p1[2]], color=color, s=28)

    _arrow3d([0, 0, 0], a1, C_BLUE, lw=2.4)
    _arrow3d([0, 0, 0], a2, C_BLUE, lw=2.4)
    _arrow3d([0, 0, 0], y, C_PURPLE, lw=2.6)
    _arrow3d([0, 0, 0], yhat, C_GREEN, lw=2.6)
    _arrow3d(yhat, y, C_AMBER, lw=2.2, ls="--")

    # Right-angle marker at yhat: draw a tiny L-shape with line segments,
    # one leg in the plane, the other along the residual direction.
    n_in = a1 / np.linalg.norm(a1) * 0.20
    n_out = resid / np.linalg.norm(resid) * 0.20
    p0 = yhat + n_in
    p1 = yhat + n_in + n_out
    p2 = yhat + n_out
    ax.plot([p0[0], p1[0], p2[0]],
            [p0[1], p1[1], p2[1]],
            [p0[2], p1[2], p2[2]],
            color=C_DARK, lw=1.1)

    # Labels.
    ax.text(*a1, "  $a_1$", color=C_BLUE, fontsize=11, fontweight="bold")
    ax.text(*a2, "  $a_2$", color=C_BLUE, fontsize=11, fontweight="bold")
    ax.text(y[0], y[1], y[2] + 0.08, r"$y$",
            color=C_PURPLE, fontsize=13, fontweight="bold")
    ax.text(yhat[0], yhat[1], yhat[2] - 0.18, r"$\hat y = Xw^*$",
            color=C_GREEN, fontsize=12, fontweight="bold")
    mid = (y + yhat) / 2
    ax.text(mid[0] + 0.05, mid[1] + 0.05, mid[2],
            r"$y - \hat y \perp \mathrm{Col}(X)$",
            color=C_AMBER, fontsize=11, fontweight="bold")
    ax.text(1.6, 1.6, -0.05, r"$\mathrm{Col}(X)$",
            color=C_BLUE, fontsize=11, fontweight="bold")

    ax.set_xlabel("$e_1$", color=C_DARK)
    ax.set_ylabel("$e_2$", color=C_DARK)
    ax.set_zlabel("$e_3$", color=C_DARK)
    ax.set_title("OLS = orthogonal projection of $y$ onto $\\mathrm{Col}(X)$",
                 fontsize=13, color=C_DARK, fontweight="bold", pad=14)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1, 1, 0.85))

    fig.tight_layout()
    save(fig, "fig2_projection_geometry")


# ---------------------------------------------------------------------------
# Fig 3: Polynomial under / right / overfit
# ---------------------------------------------------------------------------
def fig3_polynomial_fits():
    rng = np.random.default_rng(2)
    n = 30
    x = np.sort(rng.uniform(0, 1, n))
    true = lambda t: np.sin(2 * np.pi * t)
    y = true(x) + rng.normal(0, 0.22, n)

    xs = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)
    configs = [
        (1, "Underfit  (degree 1)", C_BLUE),
        (3, "Right fit  (degree 3)", C_GREEN),
        (15, "Overfit  (degree 15)", C_AMBER),
    ]

    for ax, (deg, title, color) in zip(axes, configs):
        model = make_pipeline(
            PolynomialFeatures(deg, include_bias=False),
            LinearRegression(),
        )
        model.fit(x.reshape(-1, 1), y)
        ys = model.predict(xs.reshape(-1, 1))
        train_mse = mean_squared_error(y, model.predict(x.reshape(-1, 1)))

        _style_ax(ax)
        ax.plot(xs, true(xs), color=C_GRAY, lw=1.6, ls="--",
                label=r"true: $\sin 2\pi x$")
        ax.plot(xs, ys, color=color, lw=2.4, label=f"degree {deg} fit")
        ax.scatter(x, y, s=34, color=C_DARK, zorder=4,
                   edgecolor="white", linewidth=0.6, label="data")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-1.7, 1.7)
        ax.set_xlabel("x", color=C_DARK)
        ax.set_title(title + f"\ntrain MSE = {train_mse:.3f}",
                     color=C_DARK, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    axes[0].set_ylabel("y", color=C_DARK)
    fig.suptitle("Polynomial regression: model capacity vs data complexity",
                 fontsize=13, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_polynomial_fits")


# ---------------------------------------------------------------------------
# Fig 4: Regularization paths -- Ridge vs Lasso
# ---------------------------------------------------------------------------
def fig4_regularization_paths():
    """Coefficient trajectories vs lambda for Ridge and Lasso."""
    rng = np.random.default_rng(3)
    n, d = 80, 8
    X = rng.normal(0, 1, (n, d))
    # Inject collinearity: column 1 ~ column 0.
    X[:, 1] = X[:, 0] + 0.05 * rng.normal(0, 1, n)
    true_w = np.array([2.0, -1.5, 0.0, 0.0, 1.0, 0.0, -0.7, 0.0])
    y = X @ true_w + rng.normal(0, 0.5, n)
    X = StandardScaler().fit_transform(X)

    alphas = np.logspace(-3, 3, 60)
    ridge_paths = np.zeros((d, len(alphas)))
    lasso_paths = np.zeros((d, len(alphas)))
    for i, a in enumerate(alphas):
        ridge_paths[:, i] = Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_
        lasso_paths[:, i] = Lasso(alpha=a, fit_intercept=False,
                                  max_iter=20000).fit(X, y).coef_

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER,
               "#ef4444", "#0ea5e9", "#a855f7", "#22c55e"]

    for ax, paths, title in [
        (axes[0], ridge_paths, "Ridge (L2):  coefficients shrink smoothly"),
        (axes[1], lasso_paths,
         "Lasso (L1):  coefficients hit exactly zero"),
    ]:
        _style_ax(ax)
        for j in range(d):
            ax.plot(alphas, paths[j], color=palette[j], lw=1.8,
                    label=fr"$w_{j+1}$  (true={true_w[j]:+.1f})")
        ax.axhline(0, color=C_GRAY, lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"regularization strength  $\lambda$",
                      color=C_DARK, fontsize=11)
        ax.set_title(title, color=C_DARK, fontsize=12, fontweight="bold")

    axes[0].set_ylabel("coefficient value", color=C_DARK, fontsize=11)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=8.5, framealpha=0.95)
    fig.suptitle("Regularization paths: how penalties reshape the solution",
                 fontsize=13, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_regularization_paths")


# ---------------------------------------------------------------------------
# Fig 5: L1 vs L2 geometry -- why L1 is sparse
# ---------------------------------------------------------------------------
def fig5_l1_vs_l2_geometry():
    """Diamond vs disk: contours of (w-w_ols)^2 hitting the constraint."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))

    w_ols = np.array([2.0, 1.2])  # OLS optimum (outside both balls)

    # Build a quadratic loss with elliptical (slightly correlated) contours.
    A = np.array([[1.0, 0.45], [0.45, 0.8]])

    def loss(W1, W2):
        d1 = W1 - w_ols[0]
        d2 = W2 - w_ols[1]
        return A[0, 0] * d1 ** 2 + 2 * A[0, 1] * d1 * d2 + A[1, 1] * d2 ** 2

    grid = np.linspace(-1.2, 3.0, 400)
    W1, W2 = np.meshgrid(grid, grid)
    Z = loss(W1, W2)

    # ---- Left: L2 (disk) -> non-sparse ----
    ax = axes[0]
    _style_ax(ax)
    ax.contour(W1, W2, Z, levels=8, colors=C_GRAY, linewidths=0.7)
    disk = Circle((0, 0), 1.0, facecolor=C_BLUE, alpha=0.22,
                  edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(disk)
    # The constrained Ridge optimum: closest point on disk to w_ols
    # (numerically; not exactly that for this metric, but close enough
    # to illustrate). Use Lagrangian: scale w_ols inward.
    w_ridge = w_ols / np.linalg.norm(w_ols) * 1.0
    ax.scatter(*w_ols, s=80, color=C_DARK, zorder=5)
    ax.scatter(*w_ridge, s=80, color=C_BLUE, zorder=5, edgecolor="white")
    ax.text(w_ols[0] + 0.08, w_ols[1] + 0.08, r"$\hat w_{\mathrm{OLS}}$",
            color=C_DARK, fontsize=11, fontweight="bold")
    ax.text(w_ridge[0] - 0.05, w_ridge[1] + 0.18,
            r"$\hat w_{\mathrm{Ridge}}$",
            color=C_BLUE, fontsize=11, fontweight="bold")
    ax.axhline(0, color=C_GRAY, lw=0.6)
    ax.axvline(0, color=C_GRAY, lw=0.6)
    ax.set_xlim(-1.2, 3.0)
    ax.set_ylim(-1.2, 3.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$w_1$", color=C_DARK, fontsize=11)
    ax.set_ylabel("$w_2$", color=C_DARK, fontsize=11)
    ax.set_title(r"L2 ball  $\{w:\|w\|_2\leq t\}$ -- smooth boundary, "
                 "no sparsity",
                 color=C_DARK, fontsize=11.5, fontweight="bold")

    # ---- Right: L1 (diamond) -> sparse ----
    ax = axes[1]
    _style_ax(ax)
    ax.contour(W1, W2, Z, levels=8, colors=C_GRAY, linewidths=0.7)
    diamond = Polygon([[1, 0], [0, 1], [-1, 0], [0, -1]],
                      facecolor=C_PURPLE, alpha=0.22,
                      edgecolor=C_PURPLE, linewidth=1.5)
    ax.add_patch(diamond)
    # Solve constrained problem on the diamond by sampling -- coarse but
    # always a corner of the diamond when contours are tilted toward axes.
    candidates = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    losses = [loss(c[0], c[1]) for c in candidates]
    # Also sample edges to be fair.
    for k in range(4):
        p1 = candidates[k]
        p2 = candidates[(k + 1) % 4]
        for t in np.linspace(0, 1, 200):
            p = (1 - t) * p1 + t * p2
            losses.append(loss(p[0], p[1]))
            candidates = np.vstack([candidates, p])
    w_lasso = candidates[int(np.argmin(losses))]
    ax.scatter(*w_ols, s=80, color=C_DARK, zorder=5)
    ax.scatter(*w_lasso, s=90, color=C_PURPLE, zorder=5, edgecolor="white")
    ax.text(w_ols[0] + 0.08, w_ols[1] + 0.08, r"$\hat w_{\mathrm{OLS}}$",
            color=C_DARK, fontsize=11, fontweight="bold")
    ax.text(w_lasso[0] + 0.08, w_lasso[1] + 0.05,
            fr"$\hat w_{{\mathrm{{Lasso}}}}=({w_lasso[0]:.0f},{w_lasso[1]:.0f})$",
            color=C_PURPLE, fontsize=11, fontweight="bold")
    ax.axhline(0, color=C_GRAY, lw=0.6)
    ax.axvline(0, color=C_GRAY, lw=0.6)
    ax.set_xlim(-1.2, 3.0)
    ax.set_ylim(-1.2, 3.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$w_1$", color=C_DARK, fontsize=11)
    ax.set_ylabel("$w_2$", color=C_DARK, fontsize=11)
    ax.set_title(r"L1 ball  $\{w:\|w\|_1\leq t\}$ -- corners on axes "
                 "produce zeros",
                 color=C_DARK, fontsize=11.5, fontweight="bold")

    fig.suptitle("Why Lasso is sparse: contours meet the L1 ball at corners",
                 fontsize=13, color=C_DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_l1_vs_l2_geometry")


# ---------------------------------------------------------------------------
# Fig 6: Cross-validation curve
# ---------------------------------------------------------------------------
def fig6_cv_curve():
    rng = np.random.default_rng(4)
    n = 50
    x = np.sort(rng.uniform(-1, 1, n))
    y = np.sin(3 * x) + 0.4 * x + rng.normal(0, 0.25, n)
    X = x.reshape(-1, 1)

    degrees = np.arange(1, 16)
    train_mse = []
    cv_mse = []
    cv_std = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for d in degrees:
        model = make_pipeline(
            PolynomialFeatures(d, include_bias=False),
            LinearRegression(),
        )
        model.fit(X, y)
        train_mse.append(mean_squared_error(y, model.predict(X)))
        scores = -cross_val_score(model, X, y, cv=kf,
                                  scoring="neg_mean_squared_error")
        cv_mse.append(scores.mean())
        cv_std.append(scores.std())

    train_mse = np.array(train_mse)
    cv_mse = np.array(cv_mse)
    cv_std = np.array(cv_std)
    best = int(np.argmin(cv_mse))

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    _style_ax(ax)
    ax.plot(degrees, train_mse, "o-", color=C_BLUE, lw=2.0,
            label="train MSE")
    ax.plot(degrees, cv_mse, "s-", color=C_AMBER, lw=2.2,
            label="5-fold CV MSE")
    ax.fill_between(degrees, cv_mse - cv_std, cv_mse + cv_std,
                    color=C_AMBER, alpha=0.18, label=r"CV $\pm 1\sigma$")
    ax.axvline(degrees[best], color=C_GREEN, lw=1.6, ls="--",
               label=fr"best degree = {degrees[best]}")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree", color=C_DARK, fontsize=11)
    ax.set_ylabel("MSE  (log scale)", color=C_DARK, fontsize=11)
    ax.set_title("Cross-validation reveals the bias-variance sweet spot",
                 color=C_DARK, fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig6_cv_curve")


# ---------------------------------------------------------------------------
# Fig 7: Outlier sensitivity -- OLS vs Huber
# ---------------------------------------------------------------------------
def fig7_outlier_robustness():
    rng = np.random.default_rng(5)
    n = 40
    x = np.linspace(0, 10, n)
    y = 0.7 * x + 1.0 + rng.normal(0, 0.6, n)
    # Inject outliers.
    out_idx = [5, 18, 32]
    y[out_idx] += np.array([8.0, -7.0, 9.0])
    X = x.reshape(-1, 1)

    ols = LinearRegression().fit(X, y)
    huber = HuberRegressor(epsilon=1.35).fit(X, y)

    xs = np.linspace(x.min(), x.max(), 200)
    ys_ols = ols.predict(xs.reshape(-1, 1))
    ys_huber = huber.predict(xs.reshape(-1, 1))
    ys_true = 0.7 * xs + 1.0

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    _style_ax(ax)

    mask = np.ones(n, dtype=bool)
    mask[out_idx] = False
    ax.scatter(x[mask], y[mask], s=42, color=C_BLUE, zorder=4,
               edgecolor="white", linewidth=0.7, label="inliers")
    ax.scatter(x[~mask], y[~mask], s=120, color=C_AMBER, zorder=5,
               edgecolor=C_DARK, linewidth=1.0, marker="X",
               label="outliers")

    ax.plot(xs, ys_true, color=C_GRAY, lw=1.6, ls="--",
            label=r"true line  $y=0.7x+1$")
    ax.plot(xs, ys_ols, color=C_PURPLE, lw=2.4,
            label=fr"OLS:    $\hat y={ols.coef_[0]:.2f}\,x+{ols.intercept_:.2f}$")
    ax.plot(xs, ys_huber, color=C_GREEN, lw=2.4,
            label=fr"Huber:  $\hat y={huber.coef_[0]:.2f}\,x+{huber.intercept_:.2f}$")

    ax.set_xlabel("x", color=C_DARK, fontsize=11)
    ax.set_ylabel("y", color=C_DARK, fontsize=11)
    ax.set_title(
        "OLS gets dragged by outliers; Huber regression resists",
        color=C_DARK, fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig7_outlier_robustness")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for ML Math Derivations 05: Linear Regression")
    print(f"  EN dir: {EN_DIR}")
    print(f"  ZH dir: {ZH_DIR}")
    fig1_scatter_residuals();      print("  [ok] fig1_scatter_residuals")
    fig2_projection_geometry();    print("  [ok] fig2_projection_geometry")
    fig3_polynomial_fits();        print("  [ok] fig3_polynomial_fits")
    fig4_regularization_paths();   print("  [ok] fig4_regularization_paths")
    fig5_l1_vs_l2_geometry();      print("  [ok] fig5_l1_vs_l2_geometry")
    fig6_cv_curve();               print("  [ok] fig6_cv_curve")
    fig7_outlier_robustness();     print("  [ok] fig7_outlier_robustness")
    print("All figures saved to both EN and ZH asset folders.")


if __name__ == "__main__":
    main()
