"""
Figure generation script for ML Math Derivations Part 11:
Ensemble Learning (Bagging, Random Forest, AdaBoost, GBDT).

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates one specific idea so the math behind ensembling becomes
visible.

Figures:
    fig1_bagging_diagram        Bagging: bootstrap samples in parallel feed
                                T weak learners; predictions averaged.
    fig2_boosting_diagram       Boosting: sequential weak learners with
                                reweighted samples and weighted vote.
    fig3_rf_decision_boundary   Single deep tree vs Random Forest on a
                                two-moons-style dataset; RF boundary is
                                visibly smoother.
    fig4_bias_variance          Bias-variance decomposition: single learner
                                vs T-member bagging ensemble across many
                                training sets.
    fig5_adaboost_weights       AdaBoost: per-iteration sample weight
                                evolution and training error decay.
    fig6_stacking_diagram       Stacking architecture: heterogeneous base
                                learners feeding a meta-learner.
    fig7_size_vs_error          Test error vs ensemble size for Bagging,
                                Random Forest and AdaBoost.

Usage:
    python3 scripts/figures/ml-math-derivations/11-ensemble.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "11-Ensemble-Learning"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "11-集成学习"
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


def _box(ax, xy, w, h, text, color, text_color="white", fontsize=10,
         fontweight="bold"):
    """Draw a rounded rectangle with centred text."""
    patch = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight=fontweight)


def _arrow(ax, p0, p1, color=C_GRAY, lw=1.6, style="-|>", mutation=14):
    arrow = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mutation,
                            color=color, lw=lw, shrinkA=2, shrinkB=2)
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Figure 1: Bagging diagram (parallel weak learners)
# ---------------------------------------------------------------------------
def fig1_bagging_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.0))

    # Original training set on the far left
    _box(ax, (1.0, 3.0), 1.8, 1.0,
         "Training set\n$\\mathcal{D} = \\{(\\mathbf{x}_i, y_i)\\}_{i=1}^N$",
         C_DARK, fontsize=10.5)

    # Three bootstrap samples
    boot_x = 3.8
    boots_y = [4.7, 3.0, 1.3]
    for i, y in enumerate(boots_y):
        _box(ax, (boot_x, y), 2.3, 0.95,
             f"Bootstrap $\\mathcal{{D}}_{i+1}$\nresample with replacement",
             C_BLUE, fontsize=9.5)
        _arrow(ax, (1.95, 3.0), (boot_x - 1.2, y), color=C_GRAY, lw=1.4)

    # Three base learners
    learner_x = 6.8
    for i, y in enumerate(boots_y):
        _box(ax, (learner_x, y), 2.0, 0.95,
             f"Weak learner $h_{i+1}$\n(e.g. deep tree)",
             C_PURPLE, fontsize=9.5)
        _arrow(ax, (boot_x + 1.2, y), (learner_x - 1.05, y),
               color=C_GRAY, lw=1.4)

    # Ellipsis
    ax.text(boot_x, 0.3, "$\\vdots$", ha="center", fontsize=18, color=C_GRAY)
    ax.text(learner_x, 0.3, "$\\vdots$", ha="center", fontsize=18,
            color=C_GRAY)
    ax.text(boot_x, -0.2, "$T$ samples", ha="center", fontsize=10,
            color=C_DARK)
    ax.text(learner_x, -0.2, "$T$ learners", ha="center", fontsize=10,
            color=C_DARK)

    # Aggregator
    _box(ax, (9.7, 3.0), 1.9, 1.4,
         "Aggregate\n$H(\\mathbf{x}) = \\frac{1}{T}\\sum_t h_t(\\mathbf{x})$\n"
         "or majority vote",
         C_GREEN, fontsize=10)
    for y in boots_y:
        _arrow(ax, (learner_x + 1.05, y), (9.7 - 1.0, 3.0),
               color=C_GRAY, lw=1.4)

    # Output
    _box(ax, (11.7, 3.0), 1.2, 0.9, "$H(\\mathbf{x})$", C_AMBER,
         fontsize=12)
    _arrow(ax, (9.7 + 1.0, 3.0), (11.7 - 0.65, 3.0),
           color=C_GRAY, lw=1.6)

    # Annotation banner
    ax.text(6.4, 5.85,
            "Bagging: train $T$ learners IN PARALLEL on independent "
            "bootstrap samples, then average.\n"
            "Variance drops by $\\approx 1/T$ when learners are nearly "
            "uncorrelated; bias is preserved.",
            ha="center", fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor="none"))

    ax.set_xlim(0, 12.5)
    ax.set_ylim(-0.6, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Bagging: parallel weak learners + averaging",
                 fontsize=13.5, fontweight="bold", pad=8)

    fig.tight_layout()
    _save(fig, "fig1_bagging_diagram")


# ---------------------------------------------------------------------------
# Figure 2: Boosting diagram (sequential weighted learners)
# ---------------------------------------------------------------------------
def fig2_boosting_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    # Three sequential rounds
    centers_x = [2.2, 5.6, 9.0]
    weights_text = [
        "$w_1(i) = 1/N$",
        "$w_2(i) \\propto e^{-\\alpha_1 y_i h_1}$",
        "$w_3(i) \\propto e^{-\\alpha_2 y_i h_2}$",
    ]
    alpha_text = [
        "$\\alpha_1 = \\frac{1}{2}\\ln\\frac{1-\\epsilon_1}{\\epsilon_1}$",
        "$\\alpha_2 = \\frac{1}{2}\\ln\\frac{1-\\epsilon_2}{\\epsilon_2}$",
        "$\\alpha_3 = \\frac{1}{2}\\ln\\frac{1-\\epsilon_3}{\\epsilon_3}$",
    ]

    for i, (cx, w_str, a_str) in enumerate(zip(centers_x, weights_text,
                                               alpha_text)):
        # Weight box (top)
        _box(ax, (cx, 4.6), 2.4, 0.8,
             f"Sample weights\n{w_str}",
             C_BLUE, fontsize=9.5)
        # Learner box (middle)
        _box(ax, (cx, 3.1), 2.4, 0.8,
             f"Train weak learner $h_{i+1}$\non weighted data",
             C_PURPLE, fontsize=9.5)
        # Alpha box (bottom)
        _box(ax, (cx, 1.6), 2.4, 0.8,
             f"Learner weight\n{a_str}",
             C_GREEN, fontsize=9.5)
        # Vertical arrows inside one round
        _arrow(ax, (cx, 4.2), (cx, 3.5), color=C_GRAY, lw=1.3)
        _arrow(ax, (cx, 2.7), (cx, 2.0), color=C_GRAY, lw=1.3)
        # Round label
        ax.text(cx, 5.3, f"Round $t = {i+1}$", ha="center", fontsize=10.5,
                fontweight="bold", color=C_DARK)

    # Horizontal arrows between rounds (carry weights forward)
    for i in range(2):
        _arrow(ax, (centers_x[i] + 1.25, 4.6), (centers_x[i+1] - 1.25, 4.6),
               color=C_AMBER, lw=2.0, mutation=18)
        ax.text((centers_x[i] + centers_x[i+1]) / 2, 4.85,
                "reweight\n(boost\nhard cases)",
                ha="center", fontsize=8.5, color=C_AMBER, fontweight="bold")

    # Final aggregation
    _box(ax, (5.6, 0.2), 6.0, 0.7,
         "Final model: $H(\\mathbf{x}) = "
         "\\mathrm{sign}\\!\\left(\\sum_{t=1}^T \\alpha_t h_t(\\mathbf{x})"
         "\\right)$  --  weighted vote",
         C_DARK, fontsize=11)
    for cx in centers_x:
        _arrow(ax, (cx, 1.2), (cx, 0.55), color=C_GRAY, lw=1.3)

    ax.text(5.6, 6.1,
            "Boosting: train weak learners SEQUENTIALLY. Each round "
            "reweights training samples toward the ones the previous "
            "learner got wrong.",
            ha="center", fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor="none"))

    ax.set_xlim(0, 11.2)
    ax.set_ylim(-0.4, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Boosting: sequential weighted learners (AdaBoost view)",
                 fontsize=13.5, fontweight="bold", pad=8)

    fig.tight_layout()
    _save(fig, "fig2_boosting_diagram")


# ---------------------------------------------------------------------------
# Helpers for tree-style classifiers (kept dependency-free)
# ---------------------------------------------------------------------------
def _two_moons(n=300, noise=0.30, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = np.linspace(0, np.pi, n1)
    X1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    t2 = np.linspace(0, np.pi, n2)
    X2 = np.stack([1 - np.cos(t2), -np.sin(t2) + 0.5], axis=1)
    X = np.vstack([X1, X2]) + rng.normal(scale=noise, size=(n, 2))
    y = np.hstack([np.zeros(n1), np.ones(n2)]).astype(int)
    return X, y


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def _gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - np.sum(p ** 2)


def _build_tree(X, y, depth, max_depth, min_samples=2, feature_subset=None,
                rng=None):
    if (depth >= max_depth or len(y) < min_samples or _gini(y) < 1e-6):
        majority = int(np.argmax(np.bincount(y, minlength=2)))
        return _Node(value=majority)
    n_feat = X.shape[1]
    if feature_subset is not None and feature_subset < n_feat:
        feats = rng.choice(n_feat, size=feature_subset, replace=False)
    else:
        feats = np.arange(n_feat)
    best_gain = -1.0
    best = None
    parent_gini = _gini(y)
    for j in feats:
        # Try a handful of thresholds (quantiles) to keep it fast
        col = X[:, j]
        qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, 9)))
        for t in qs:
            left_mask = col <= t
            n_l = left_mask.sum()
            n_r = len(y) - n_l
            if n_l < 1 or n_r < 1:
                continue
            g = (n_l * _gini(y[left_mask])
                 + n_r * _gini(y[~left_mask])) / len(y)
            gain = parent_gini - g
            if gain > best_gain:
                best_gain = gain
                best = (j, t, left_mask)
    if best is None or best_gain <= 0:
        majority = int(np.argmax(np.bincount(y, minlength=2)))
        return _Node(value=majority)
    j, t, left_mask = best
    left = _build_tree(X[left_mask], y[left_mask], depth + 1, max_depth,
                       min_samples, feature_subset, rng)
    right = _build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth,
                        min_samples, feature_subset, rng)
    return _Node(feature=j, threshold=t, left=left, right=right)


def _predict_tree(node, X):
    if node.value is not None:
        return np.full(len(X), node.value, dtype=int)
    out = np.zeros(len(X), dtype=int)
    mask = X[:, node.feature] <= node.threshold
    out[mask] = _predict_tree(node.left, X[mask])
    out[~mask] = _predict_tree(node.right, X[~mask])
    return out


def _predict_proba_tree(node, X):
    if node.value is not None:
        return np.full(len(X), float(node.value))
    out = np.zeros(len(X))
    mask = X[:, node.feature] <= node.threshold
    out[mask] = _predict_proba_tree(node.left, X[mask])
    out[~mask] = _predict_proba_tree(node.right, X[~mask])
    return out


def _train_random_forest(X, y, n_trees=80, max_depth=8, m=1, seed=0):
    rng = np.random.default_rng(seed)
    trees = []
    n = len(y)
    for t in range(n_trees):
        idx = rng.integers(0, n, size=n)
        tr = _build_tree(X[idx], y[idx], 0, max_depth, 2,
                         feature_subset=m, rng=rng)
        trees.append(tr)
    return trees


def _rf_proba(trees, X):
    return np.mean([_predict_proba_tree(t, X) for t in trees], axis=0)


# ---------------------------------------------------------------------------
# Figure 3: Random Forest decision boundary vs single tree
# ---------------------------------------------------------------------------
def fig3_rf_decision_boundary() -> None:
    X, y = _two_moons(n=320, noise=0.28, seed=7)

    # Single deep tree
    rng = np.random.default_rng(0)
    tree = _build_tree(X, y, 0, max_depth=12, min_samples=2,
                       feature_subset=None, rng=rng)

    # Random forest with feature subset = 1 (out of 2) for visible diversity
    forest = _train_random_forest(X, y, n_trees=80, max_depth=8, m=1, seed=11)

    xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 220),
                         np.linspace(-1.3, 1.7, 220))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    P_tree = _predict_proba_tree(tree, grid).reshape(xx.shape)
    P_rf = _rf_proba(forest, grid).reshape(xx.shape)

    cmap = LinearSegmentedColormap.from_list(
        "bp", ["#dbeafe", "#ffffff", "#ede9fe"], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))

    titles = ["Single decision tree (depth 12)\nblocky, high-variance "
              "boundary",
              "Random Forest (80 trees, $m=1$)\nsmooth, averaged boundary"]
    surfaces = [P_tree, P_rf]
    for ax, P, title in zip(axes, surfaces, titles):
        ax.contourf(xx, yy, P, levels=20, cmap=cmap, alpha=0.95)
        ax.contour(xx, yy, P, levels=[0.5], colors=[C_DARK], linewidths=2.0)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], color=C_BLUE, s=22,
                   edgecolor="white", linewidth=0.7, label="class 0",
                   alpha=0.9)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color=C_PURPLE, s=22,
                   edgecolor="white", linewidth=0.7, label="class 1",
                   alpha=0.9)
        ax.set_xlim(-1.6, 2.6)
        ax.set_ylim(-1.3, 1.7)
        ax.set_xlabel(r"$x_1$", fontsize=11)
        ax.set_ylabel(r"$x_2$", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle("Random Forest smooths the boundary by averaging "
                 "decorrelated trees",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_rf_decision_boundary")


# ---------------------------------------------------------------------------
# Figure 4: Bias-variance: single learner vs bagging ensemble
# ---------------------------------------------------------------------------
def fig4_bias_variance() -> None:
    """1D regression: estimate predictions across many training sets and
    show that bagging shrinks the variance band while leaving the mean
    (bias) almost unchanged."""
    rng_master = np.random.default_rng(2)

    def f_true(x):
        return np.sin(2.0 * x) + 0.3 * x

    x_grid = np.linspace(-3.0, 3.0, 300)
    y_true = f_true(x_grid)

    def sample_dataset(seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3.0, 3.0, size=80)
        y = f_true(x) + rng.normal(scale=0.40, size=x.shape)
        return x, y

    # Tiny regression-tree-like learner: piecewise constant by binning
    def fit_tree(x, y, n_bins=10):
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        means = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (x > edges[b]) & (x <= edges[b + 1])
            means[b] = y[mask].mean() if mask.any() else 0.0
        return edges, means

    def predict_tree(model, xq):
        edges, means = model
        idx = np.clip(np.searchsorted(edges, xq, side="left") - 1,
                      0, len(means) - 1)
        return means[idx]

    n_runs = 120
    n_in_bag = 25  # bagging ensemble size

    single_preds = np.zeros((n_runs, len(x_grid)))
    bag_preds = np.zeros((n_runs, len(x_grid)))

    for r in range(n_runs):
        x, y = sample_dataset(seed=10_000 + r)
        # Single tree
        m = fit_tree(x, y, n_bins=10)
        single_preds[r] = predict_tree(m, x_grid)
        # Bagged ensemble
        rng = np.random.default_rng(20_000 + r)
        bag = np.zeros(len(x_grid))
        for _ in range(n_in_bag):
            idx = rng.integers(0, len(x), size=len(x))
            mb = fit_tree(x[idx], y[idx], n_bins=10)
            bag += predict_tree(mb, x_grid)
        bag_preds[r] = bag / n_in_bag

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    panels = [("Single tree (high variance)", single_preds, C_BLUE),
              (f"Bagging of {n_in_bag} trees (variance shrinks)",
               bag_preds, C_PURPLE)]
    for ax, (title, preds, col) in zip(axes, panels):
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        # Truth
        ax.plot(x_grid, y_true, color=C_DARK, lw=2.2,
                label="true $f(x)$")
        # Variance band
        ax.fill_between(x_grid, mean - std, mean + std, color=col, alpha=0.25,
                        label="$\\pm 1$ std across runs")
        # Mean
        ax.plot(x_grid, mean, color=col, lw=2.2, label="mean prediction")
        # A few individual fits to show the wiggle
        for r in range(0, 12):
            ax.plot(x_grid, preds[r], color=col, lw=0.6, alpha=0.18)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("$x$", fontsize=11)
        ax.set_ylabel("$\\hat f(x)$", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    # Variance numbers in the figure subtitle
    var_single = single_preds.var(axis=0).mean()
    var_bag = bag_preds.var(axis=0).mean()
    fig.suptitle(
        f"Bagging reduces variance: avg pointwise variance "
        f"$\\overline{{\\mathrm{{Var}}}} = {var_single:.3f}$ "
        f"$\\to$ {var_bag:.3f}",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_bias_variance")


# ---------------------------------------------------------------------------
# Figure 5: AdaBoost weight evolution + training error
# ---------------------------------------------------------------------------
def _stump_fit(X, y, w):
    """Fit a 1-feature axis-aligned decision stump minimising weighted error.
    Labels in {-1, +1}."""
    N, d = X.shape
    best = (None, None, None, np.inf)  # (j, thr, polarity, err)
    for j in range(d):
        col = X[:, j]
        ths = np.unique(np.quantile(col, np.linspace(0.05, 0.95, 19)))
        for t in ths:
            for pol in (1, -1):
                pred = np.where(col <= t, pol, -pol)
                err = np.sum(w[pred != y])
                if err < best[3]:
                    best = (j, t, pol, err)
    return best  # j, thr, pol, err


def _stump_predict(stump, X):
    j, t, pol, _ = stump
    return np.where(X[:, j] <= t, pol, -pol)


def fig5_adaboost_weights() -> None:
    rng = np.random.default_rng(4)
    # 2D classification with overlapping (but separable) Gaussians and a
    # handful of GENUINELY hard borderline cases (no label flips, so the
    # training error can actually decay).
    n = 80
    X1 = rng.normal(loc=[-1.4, 0.2], scale=0.65, size=(n // 2, 2))
    X2 = rng.normal(loc=[1.4, -0.1], scale=0.65, size=(n // 2, 2))
    X = np.vstack([X1, X2])
    y = np.hstack([-np.ones(n // 2), np.ones(n // 2)])
    # Drag a few class-+1 points into class -1 territory and vice versa,
    # so they are HARD but not mislabeled.
    hard_idx = rng.choice(n, size=8, replace=False)
    X[hard_idx, 0] *= -0.55  # mirror across vertical axis a bit

    T = 30
    w = np.ones(n) / n
    weights_hist = [w.copy()]
    train_err = []
    F = np.zeros(n)
    for _ in range(T):
        stump = _stump_fit(X, y, w)
        pred = _stump_predict(stump, X)
        eps = np.clip(np.sum(w[pred != y]), 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - eps) / eps)
        w = w * np.exp(-alpha * y * pred)
        w = w / w.sum()
        weights_hist.append(w.copy())
        F += alpha * pred
        train_err.append(np.mean(np.sign(F) != y))

    weights_hist = np.array(weights_hist)  # (T+1, n)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # --- Left: weight evolution heatmap ---
    ax = axes[0]
    final_w = weights_hist[-1]
    order = np.argsort(-final_w)
    Wsorted = weights_hist[:, order].T  # (n, T+1)
    im = ax.imshow(Wsorted, aspect="auto", cmap="magma",
                   extent=(0, T, n, 0))
    ax.set_xlabel("AdaBoost iteration $t$", fontsize=11)
    ax.set_ylabel("training sample (sorted by final weight)", fontsize=11)
    ax.set_title("Sample weights $w_t(i)$ over iterations",
                 fontsize=12.5, fontweight="bold", pad=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("$w_t(i)$", fontsize=10)
    ax.text(T * 0.55, n * 0.10,
            "hard borderline cases\nclimb to large weight",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_DARK,
                      edgecolor="none", alpha=0.7))

    # --- Right: training error decay ---
    ax = axes[1]
    iters = np.arange(1, T + 1)
    ax.plot(iters, train_err, color=C_BLUE, lw=2.4, marker="o", ms=4,
            label="training error of $\\mathrm{sign}(\\sum_t \\alpha_t h_t)$")
    # Theoretical exponential bound using the actual gamma_t observed
    # (gamma_t = 0.5 - eps_t). Recompute via the same fit just to get gammas.
    rng2 = np.random.default_rng(4)
    n2 = n
    w2 = np.ones(n2) / n2
    gammas = []
    for _ in range(T):
        stump = _stump_fit(X, y, w2)
        pred = _stump_predict(stump, X)
        eps = np.clip(np.sum(w2[pred != y]), 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - eps) / eps)
        gammas.append(0.5 - eps)
        w2 = w2 * np.exp(-alpha * y * pred)
        w2 = w2 / w2.sum()
    gammas = np.array(gammas)
    bound = np.exp(-2 * np.cumsum(gammas ** 2))
    ax.plot(iters, bound, color=C_AMBER, lw=1.8, ls="--",
            label=r"upper bound $\exp(-2\sum_t \gamma_t^2)$")
    ax.axhline(0, color=C_GRAY, lw=0.8, ls=":")
    ax.set_xlabel("AdaBoost iteration $t$", fontsize=11)
    ax.set_ylabel("training error", fontsize=11)
    ax.set_ylim(-0.02, max(0.55, max(train_err) + 0.05))
    ax.set_xlim(1, T)
    ax.set_title("Training error decays toward zero",
                 fontsize=12.5, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)

    fig.suptitle("AdaBoost: focus shifts toward hard samples; "
                 "training error vanishes",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_adaboost_weights")


# ---------------------------------------------------------------------------
# Figure 6: Stacking architecture
# ---------------------------------------------------------------------------
def fig6_stacking_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    # Input
    _box(ax, (1.0, 3.0), 1.6, 1.0,
         "Training data\n$(\\mathbf{x}, y)$",
         C_DARK, fontsize=10.5)

    # Layer 1: heterogeneous base learners
    base_y = [5.0, 3.5, 2.0, 0.5]
    base_labels = [
        ("Logistic regression", C_BLUE),
        ("Random Forest", C_PURPLE),
        ("Gradient Boosting", C_GREEN),
        ("k-NN / SVM / ...", C_AMBER),
    ]
    base_x = 4.0
    for y, (lab, col) in zip(base_y, base_labels):
        _box(ax, (base_x, y), 2.2, 0.7, lab, col, fontsize=9.5)
        _arrow(ax, (1.85, 3.0), (base_x - 1.15, y), color=C_GRAY, lw=1.3)

    # Out-of-fold predictions (vector)
    oof_x = 7.0
    _box(ax, (oof_x, 3.0), 2.0, 1.6,
         "Out-of-fold predictions\n"
         "$\\mathbf{z} = (h_1(\\mathbf{x}),\\dots,h_K(\\mathbf{x}))$\n"
         "(meta features)",
         C_LIGHT, text_color=C_DARK, fontsize=9.5)
    for y in base_y:
        _arrow(ax, (base_x + 1.15, y), (oof_x - 1.05, 3.0),
               color=C_GRAY, lw=1.3)

    # Meta-learner
    _box(ax, (9.6, 3.0), 1.8, 1.0,
         "Meta-learner $g(\\mathbf{z})$\n(usually simple,\ne.g. logistic)",
         C_DARK, fontsize=10)
    _arrow(ax, (oof_x + 1.05, 3.0), (9.6 - 0.95, 3.0), color=C_GRAY, lw=1.6)

    # Final prediction
    _box(ax, (10.8, 1.0), 1.4, 0.8,
         "$\\hat y = g(\\mathbf{z})$",
         C_AMBER, fontsize=11.5)
    _arrow(ax, (9.6, 2.5), (10.8, 1.4), color=C_GRAY, lw=1.4)

    # Banner
    ax.text(5.6, 6.0,
            "Stacking: heterogeneous base learners produce a NEW feature "
            "vector that a meta-learner reads.\n"
            "Cross-validation generates the meta features so the "
            "meta-learner does not see in-sample predictions.",
            ha="center", fontsize=10.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor="none"))

    ax.text(base_x, -0.3, "Layer 1 (diverse models)", ha="center",
            fontsize=10, color=C_DARK, fontweight="bold")
    ax.text(9.6, -0.3, "Layer 2 (meta-model)", ha="center",
            fontsize=10, color=C_DARK, fontweight="bold")

    ax.set_xlim(0, 11.8)
    ax.set_ylim(-0.7, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Stacking: a meta-learner consumes base-learner outputs",
                 fontsize=13.5, fontweight="bold", pad=8)

    fig.tight_layout()
    _save(fig, "fig6_stacking_diagram")


# ---------------------------------------------------------------------------
# Figure 7: Ensemble size vs test error
# ---------------------------------------------------------------------------
def fig7_size_vs_error() -> None:
    """Train a Bagging ensemble, a Random Forest and an AdaBoost-of-stumps
    on a synthetic problem, then plot test error as a function of T."""
    rng = np.random.default_rng(13)
    # Synthetic 2D classification (two moons)
    X, y01 = _two_moons(n=600, noise=0.32, seed=17)
    # Train/test split
    perm = rng.permutation(len(X))
    n_train = 400
    Xtr, ytr = X[perm[:n_train]], y01[perm[:n_train]]
    Xte, yte = X[perm[n_train:]], y01[perm[n_train:]]

    T_max = 60
    # ---- Bagging of deep trees ----
    bag_trees = []
    bag_err = []
    rng_b = np.random.default_rng(0)
    bag_pred_sum = np.zeros(len(Xte))
    for t in range(T_max):
        idx = rng_b.integers(0, len(Xtr), size=len(Xtr))
        tree = _build_tree(Xtr[idx], ytr[idx], 0, max_depth=10, min_samples=2,
                           feature_subset=None, rng=rng_b)
        bag_trees.append(tree)
        bag_pred_sum += _predict_proba_tree(tree, Xte)
        avg = bag_pred_sum / (t + 1)
        bag_err.append(np.mean((avg >= 0.5).astype(int) != yte))

    # ---- Random Forest (m=1) ----
    rf_err = []
    rng_r = np.random.default_rng(1)
    rf_pred_sum = np.zeros(len(Xte))
    for t in range(T_max):
        idx = rng_r.integers(0, len(Xtr), size=len(Xtr))
        tree = _build_tree(Xtr[idx], ytr[idx], 0, max_depth=10, min_samples=2,
                           feature_subset=1, rng=rng_r)
        rf_pred_sum += _predict_proba_tree(tree, Xte)
        avg = rf_pred_sum / (t + 1)
        rf_err.append(np.mean((avg >= 0.5).astype(int) != yte))

    # ---- AdaBoost of stumps ----
    y_pm = 2 * ytr - 1
    yte_pm = 2 * yte - 1
    w = np.ones(len(Xtr)) / len(Xtr)
    F_te = np.zeros(len(Xte))
    ada_err = []
    for t in range(T_max):
        stump = _stump_fit(Xtr, y_pm, w)
        pred_tr = _stump_predict(stump, Xtr)
        eps = np.clip(np.sum(w[pred_tr != y_pm]), 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - eps) / eps)
        w = w * np.exp(-alpha * y_pm * pred_tr)
        w = w / w.sum()
        F_te += alpha * _stump_predict(stump, Xte)
        ada_err.append(np.mean(np.sign(F_te) != yte_pm))

    # ---- Single deep tree baseline ----
    rng_s = np.random.default_rng(2)
    single = _build_tree(Xtr, ytr, 0, max_depth=10, min_samples=2,
                         feature_subset=None, rng=rng_s)
    single_err = np.mean(_predict_tree(single, Xte) != yte)

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    iters = np.arange(1, T_max + 1)
    ax.plot(iters, bag_err, color=C_BLUE, lw=2.4,
            label="Bagging (deep trees)")
    ax.plot(iters, rf_err, color=C_PURPLE, lw=2.4,
            label="Random Forest ($m=1$)")
    ax.plot(iters, ada_err, color=C_AMBER, lw=2.4,
            label="AdaBoost (stumps)")
    ax.axhline(single_err, color=C_GRAY, lw=1.6, ls="--",
               label=f"single deep tree = {single_err:.3f}")

    ax.set_xlim(1, T_max)
    ax.set_ylim(0, max(0.30, max(bag_err[0], rf_err[0], ada_err[0]) + 0.02))
    ax.set_xlabel("ensemble size $T$", fontsize=11)
    ax.set_ylabel("test error", fontsize=11)
    ax.set_title("Test error vs ensemble size: more learners is "
                 "(almost) always better",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    ax.annotate("RF benefits most from\nfeature randomisation",
                xy=(T_max, rf_err[-1]), xytext=(T_max - 18, rf_err[-1] - 0.07),
                fontsize=9.5, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1))
    ax.annotate("AdaBoost converges fast\nbut can plateau",
                xy=(T_max, ada_err[-1]),
                xytext=(T_max - 22, ada_err[-1] + 0.05),
                fontsize=9.5, color=C_AMBER,
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1))

    fig.tight_layout()
    _save(fig, "fig7_size_vs_error")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN -> {EN_DIR}")
    print(f"ZH -> {ZH_DIR}")
    print()
    for fn in (
        fig1_bagging_diagram,
        fig2_boosting_diagram,
        fig3_rf_decision_boundary,
        fig4_bias_variance,
        fig5_adaboost_weights,
        fig6_stacking_diagram,
        fig7_size_vs_error,
    ):
        print(f"[render] {fn.__name__}")
        fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
