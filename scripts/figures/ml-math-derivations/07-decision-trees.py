"""
Figure generation script for ML Math Derivations Part 07: Decision Trees.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates one core idea so the math becomes visible.

Figures:
    fig1_tree_structure         Anatomy of a decision tree (root, internal
                                nodes, branches, leaves) on a small toy
                                example with the partition path annotated.
    fig2_impurity_curves        Gini, entropy (in nats and scaled to [0,1])
                                and classification error as a function of
                                p for binary classification, with the
                                Taylor approximation entropy / 2 ~ Gini.
    fig3_decision_boundary      sklearn-trained CART on a 2D non-linear
                                dataset. Shows axis-aligned decision
                                regions and how depth controls expressivity.
    fig4_information_gain       A single split on an informative vs an
                                uninformative feature: parent impurity,
                                child impurities, and the resulting gain
                                bars.
    fig5_overfitting_curve      Train vs test accuracy as max_depth grows
                                on a noisy moons dataset, exposing the
                                classic bias-variance trade-off.
    fig6_pruning                Pre-pruning (max_depth) vs post-pruning
                                (cost-complexity, ccp_alpha) on the same
                                base tree. Two boundary panels plus an
                                accuracy-vs-alpha pruning path.
    fig7_feature_importance     Mean decrease in impurity for each feature
                                on the Iris dataset, verified against
                                scikit-learn's implementation.

Usage:
    python3 scripts/figures/ml-math-derivations/07-decision-trees.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.datasets import load_iris, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "07-Decision-Trees"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "07-决策树"
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
# Helpers
# ---------------------------------------------------------------------------
def gini(p: np.ndarray) -> np.ndarray:
    """Binary Gini index: 2 p (1-p)."""
    return 2.0 * p * (1.0 - p)


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits with safe log."""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def class_error(p: np.ndarray) -> np.ndarray:
    return 1.0 - np.maximum(p, 1.0 - p)


# ---------------------------------------------------------------------------
# Figure 1: Tree structure anatomy
# ---------------------------------------------------------------------------
def fig1_tree_structure() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def node(x, y, label, color, sub=None, w=1.7, h=0.85, text_color="white"):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            linewidth=1.4,
            edgecolor=C_DARK,
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.08, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_color)
        if sub is not None:
            ax.text(x, y - 0.22, sub, ha="center", va="center",
                    fontsize=8.5, color=text_color, alpha=0.92)

    def edge(x1, y1, x2, y2, label, side="left"):
        ax.annotate(
            "",
            xy=(x2, y2 + 0.45),
            xytext=(x1, y1 - 0.45),
            arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=1.5),
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = -0.18 if side == "left" else 0.18
        ax.text(
            mx + dx, my, label, ha="center", va="center",
            fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor="white", edgecolor=C_LIGHT, boxstyle="round,pad=0.18"),
        )

    # Root
    node(5, 5.5, "Petal length", C_BLUE, sub=r"$\leq 2.45$ cm ?")
    # Level 1
    node(2.4, 3.5, "Setosa", C_GREEN, sub="leaf  n=50")
    node(7.6, 3.5, "Petal width", C_BLUE, sub=r"$\leq 1.75$ cm ?")
    # Level 2
    node(5.6, 1.4, "Versicolor", C_AMBER, sub="leaf  n=54")
    node(9.0, 1.4, "Virginica", C_PURPLE, sub="leaf  n=46")

    edge(5, 5.5, 2.4, 3.5, "yes", "left")
    edge(5, 5.5, 7.6, 3.5, "no", "right")
    edge(7.6, 3.5, 5.6, 1.4, "yes", "left")
    edge(7.6, 3.5, 9.0, 1.4, "no", "right")

    # Legend
    ax.text(0.2, 6.2, "Decision tree on Iris (depth 2)",
            fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(0.2, 0.55,
            "Internal node = feature test     Edge = test outcome     "
            "Leaf = predicted class",
            fontsize=10, color=C_DARK)
    ax.text(0.2, 0.15,
            r"Each root-to-leaf path defines a hyper-rectangle "
            r"$R_m$ in feature space; prediction $c_m$ is the majority class.",
            fontsize=9.5, color=C_GRAY, style="italic")

    fig.tight_layout()
    _save(fig, "fig1_tree_structure")


# ---------------------------------------------------------------------------
# Figure 2: Impurity curves (Gini, entropy, classification error)
# ---------------------------------------------------------------------------
def fig2_impurity_curves() -> None:
    p = np.linspace(1e-4, 1 - 1e-4, 600)
    H = entropy_bits(p)            # in bits, max = 1 at p=0.5
    H_half = H / 2.0               # scaled to match Gini max
    G = gini(p)                    # max = 0.5
    E = class_error(p)             # max = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: all three curves ---
    ax = axes[0]
    ax.plot(p, H, color=C_BLUE, lw=2.6, label=r"Entropy  $H(p)$  (bits)")
    ax.plot(p, G, color=C_PURPLE, lw=2.6, label=r"Gini  $2p(1-p)$")
    ax.plot(p, E, color=C_AMBER, lw=2.4, ls="--",
            label=r"Classification error  $1-\max(p,1-p)$")

    ax.axvline(0.5, color=C_GRAY, lw=0.8, ls=":")
    ax.scatter([0.5], [1.0], color=C_BLUE, s=55, zorder=5, edgecolor="white", lw=1.2)
    ax.scatter([0.5], [0.5], color=C_PURPLE, s=55, zorder=5, edgecolor="white", lw=1.2)
    ax.scatter([0.5], [0.5], color=C_AMBER, s=55, zorder=5, edgecolor="white", lw=1.2)
    ax.annotate("max impurity\nat $p = 1/2$", xy=(0.5, 1.0), xytext=(0.62, 0.85),
                fontsize=10, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    ax.set_xlabel(r"positive-class probability  $p$", fontsize=11)
    ax.set_ylabel("impurity", fontsize=11)
    ax.set_title("Three measures of node impurity",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower center", fontsize=9.5, frameon=True)

    # --- Right: Taylor approximation H/2 ~ Gini around 1/2 ---
    ax = axes[1]
    ax.plot(p, H_half, color=C_BLUE, lw=2.6,
            label=r"$H(p) / 2$  (entropy in bits, halved)")
    ax.plot(p, G, color=C_PURPLE, lw=2.6,
            label=r"Gini  $2p(1-p)$")
    diff = H_half - G
    ax.fill_between(p, G, H_half, where=H_half >= G, color=C_BLUE, alpha=0.10,
                    label="gap")
    ax.axvline(0.5, color=C_GRAY, lw=0.8, ls=":")

    # Annotate the second-order match around p=1/2
    ax.annotate("Taylor at $p = 1/2$:\n"
                r"both $\approx \frac{1}{2} - 2(p - 1/2)^2$",
                xy=(0.5, 0.5), xytext=(0.06, 0.78),
                fontsize=10, color=C_DARK,
                bbox=dict(facecolor="white", edgecolor=C_LIGHT, boxstyle="round,pad=0.3"),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    ax.set_xlabel(r"positive-class probability  $p$", fontsize=11)
    ax.set_ylabel("scaled impurity", fontsize=11)
    ax.set_title(r"Why Gini and entropy give similar trees: $H/2 \approx G$",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.legend(loc="lower center", fontsize=9.5, frameon=True)

    fig.suptitle("Impurity functions for binary classification",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_impurity_curves")


# ---------------------------------------------------------------------------
# Figure 3: Axis-aligned decision boundaries on 2D data
# ---------------------------------------------------------------------------
def fig3_decision_boundary() -> None:
    rng = np.random.RandomState(7)
    X, y = make_moons(n_samples=400, noise=0.25, random_state=rng)

    depths = [1, 3, 6]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    cmap_bg = ListedColormap(["#dbeafe", "#ede9fe"])
    cmap_pt = ListedColormap([C_BLUE, C_PURPLE])

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 400),
        np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    for ax, d in zip(axes, depths):
        clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
        Z = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.9, levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors=[C_DARK], linewidths=1.3)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pt, s=18,
                   edgecolor="white", linewidth=0.5)
        train_acc = accuracy_score(y, clf.predict(X))
        ax.set_title(f"max_depth = {d}    leaves = {clf.get_n_leaves()}    "
                     f"train acc = {train_acc:.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$x_1$", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(r"$x_2$", fontsize=10)

    fig.suptitle("Axis-aligned splits build piecewise-constant regions",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_decision_boundary")


# ---------------------------------------------------------------------------
# Figure 4: Information gain at a single split
# ---------------------------------------------------------------------------
def fig4_information_gain() -> None:
    """Two splits side by side: one informative, one not. We show parent
    impurity, child impurities and the resulting weighted gain."""

    rng = np.random.RandomState(0)
    n = 200
    # Parent: 100 positives, 100 negatives -> H = 1 bit, Gini = 0.5
    parent_p = 0.5
    H_parent = entropy_bits(np.array([parent_p]))[0]

    # --- Split A: informative. Left has 90/10, right has 10/90. ---
    nL_A, nR_A = 100, 100
    pL_A, pR_A = 0.9, 0.1
    HL_A = entropy_bits(np.array([pL_A]))[0]
    HR_A = entropy_bits(np.array([pR_A]))[0]
    H_after_A = (nL_A * HL_A + nR_A * HR_A) / (nL_A + nR_A)
    gain_A = H_parent - H_after_A

    # --- Split B: uninformative. Both children remain near 50/50. ---
    nL_B, nR_B = 100, 100
    pL_B, pR_B = 0.55, 0.45
    HL_B = entropy_bits(np.array([pL_B]))[0]
    HR_B = entropy_bits(np.array([pR_B]))[0]
    H_after_B = (nL_B * HL_B + nR_B * HR_B) / (nL_B + nR_B)
    gain_B = H_parent - H_after_B

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    def draw_split(ax, title, pL, pR, nL, nR, gain, color):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6.5)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        # Parent box
        parent = FancyBboxPatch((3.6, 5.0), 2.8, 1.0,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                edgecolor=C_DARK, facecolor=C_LIGHT, lw=1.4)
        ax.add_patch(parent)
        ax.text(5.0, 5.5, f"parent: 100 / 100\n"
                          f"$H = {H_parent:.2f}$ bits",
                ha="center", va="center", fontsize=10, color=C_DARK)

        # Children
        for x, p, nL_, label in [
            (1.8, pL, nL, "left"),
            (8.2, pR, nR, "right"),
        ]:
            n_pos = int(round(p * nL_))
            n_neg = nL_ - n_pos
            H_child = entropy_bits(np.array([p]))[0]
            child = FancyBboxPatch((x - 1.4, 2.4), 2.8, 1.0,
                                   boxstyle="round,pad=0.05,rounding_size=0.15",
                                   edgecolor=C_DARK,
                                   facecolor="white", lw=1.4)
            ax.add_patch(child)
            ax.text(x, 2.9, f"{label}: {n_pos}/{n_neg}\n"
                            f"$H = {H_child:.2f}$ bits",
                    ha="center", va="center", fontsize=10, color=C_DARK)

            # Bar showing class distribution
            bw = 2.4
            bx = x - bw / 2
            ax.barh(1.55, bw * p, height=0.32, left=bx,
                    color=C_BLUE, edgecolor="white")
            ax.barh(1.55, bw * (1 - p), height=0.32, left=bx + bw * p,
                    color=C_PURPLE, edgecolor="white")
            ax.text(x, 1.05, f"p = {p:.2f}", ha="center",
                    fontsize=9, color=C_DARK)

            # Edge
            ax.annotate("", xy=(x, 3.4 + 0.1), xytext=(5.0, 5.0 - 0.05),
                        arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=1.4))

        # Gain banner
        ax.text(5.0, 0.25,
                rf"$IG = H_\mathrm{{parent}} - "
                rf"\sum_v \frac{{|S_v|}}{{|S|}} H(S_v) "
                rf"= {gain:.3f}$ bits",
                ha="center", fontsize=11.5, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.3"))

    draw_split(axes[0], "Split A: informative feature",
               pL_A, pR_A, nL_A, nR_A, gain_A, C_GREEN)
    draw_split(axes[1], "Split B: nearly useless feature",
               pL_B, pR_B, nL_B, nR_B, gain_B, C_AMBER)

    fig.suptitle("Information gain quantifies how much a split reduces uncertainty",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_information_gain")


# ---------------------------------------------------------------------------
# Figure 5: Tree depth vs train/test accuracy (overfitting)
# ---------------------------------------------------------------------------
def fig5_overfitting_curve() -> None:
    X, y = make_moons(n_samples=600, noise=0.32, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)

    depths = list(range(1, 21))
    train_acc, test_acc, n_leaves = [], [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
        train_acc.append(accuracy_score(y_tr, clf.predict(X_tr)))
        test_acc.append(accuracy_score(y_te, clf.predict(X_te)))
        n_leaves.append(clf.get_n_leaves())

    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    best_d = depths[int(np.argmax(test_acc))]
    best_test = test_acc.max()

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(depths, train_acc, color=C_BLUE, lw=2.4, marker="o",
            ms=5.5, label="training accuracy")
    ax.plot(depths, test_acc, color=C_PURPLE, lw=2.4, marker="s",
            ms=5.5, label="test accuracy")

    # Underfit / sweet-spot / overfit shading
    ax.axvspan(0.5, 2.5, color=C_AMBER, alpha=0.10)
    ax.axvspan(best_d - 0.7, best_d + 0.7, color=C_GREEN, alpha=0.12)
    ax.axvspan(10.5, 20.5, color="#fee2e2", alpha=0.55)
    ax.text(1.5, 0.55, "underfit", ha="center", fontsize=10, color=C_AMBER,
            fontweight="bold")
    ax.text(best_d, 0.55, "sweet spot", ha="center", fontsize=10, color=C_GREEN,
            fontweight="bold")
    ax.text(15.5, 0.55, "overfit", ha="center", fontsize=10, color="#dc2626",
            fontweight="bold")

    ax.scatter([best_d], [best_test], s=130, facecolor="white",
               edgecolor=C_PURPLE, lw=2.2, zorder=5)
    ax.annotate(f"best test acc = {best_test:.3f}\nat depth {best_d}",
                xy=(best_d, best_test), xytext=(best_d + 1.5, best_test - 0.06),
                fontsize=10.5, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    ax.set_xlabel("max_depth", fontsize=11)
    ax.set_ylabel("accuracy", fontsize=11)
    ax.set_xticks(depths)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Tree depth vs accuracy: bias on the left, variance on the right",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    # Twin axis: number of leaves
    ax2 = ax.twinx()
    ax2.plot(depths, n_leaves, color=C_GRAY, lw=1.4, ls=":", marker=".",
             label="number of leaves")
    ax2.set_ylabel("number of leaves", color=C_GRAY, fontsize=10)
    ax2.tick_params(axis="y", colors=C_GRAY)
    ax2.grid(False)
    ax2.legend(loc="lower center", fontsize=9, frameon=True)

    fig.tight_layout()
    _save(fig, "fig5_overfitting_curve")


# ---------------------------------------------------------------------------
# Figure 6: Pre-pruning vs post-pruning
# ---------------------------------------------------------------------------
def fig6_pruning() -> None:
    X, y = make_moons(n_samples=500, noise=0.32, random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    # Two boundary panels and one path panel
    fig = plt.figure(figsize=(14.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])

    cmap_bg = ListedColormap(["#dbeafe", "#ede9fe"])
    cmap_pt = ListedColormap([C_BLUE, C_PURPLE])
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 350),
        np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 350),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # --- Pre-pruning: limit max_depth ---
    pre = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
    Zp = pre.predict(grid).reshape(xx.shape)
    pre_train = accuracy_score(y_tr, pre.predict(X_tr))
    pre_test = accuracy_score(y_te, pre.predict(X_te))

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.contourf(xx, yy, Zp, cmap=cmap_bg, alpha=0.9, levels=[-0.5, 0.5, 1.5])
    ax0.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=cmap_pt, s=14,
                edgecolor="white", linewidth=0.4)
    ax0.set_title(f"Pre-prune: max_depth = 4\n"
                  f"leaves = {pre.get_n_leaves()}    "
                  f"train = {pre_train:.2f}    test = {pre_test:.2f}",
                  fontsize=10.5, fontweight="bold")
    ax0.set_xticks([]); ax0.set_yticks([])

    # --- Post-pruning: pick alpha by validation accuracy ---
    full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    path = full.cost_complexity_pruning_path(X_tr, y_tr)
    alphas = path.ccp_alphas[:-1]  # drop the trivial one-leaf tree
    train_scores, test_scores, leaves = [], [], []
    for a in alphas:
        c = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(X_tr, y_tr)
        train_scores.append(accuracy_score(y_tr, c.predict(X_tr)))
        test_scores.append(accuracy_score(y_te, c.predict(X_te)))
        leaves.append(c.get_n_leaves())
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)

    best_idx = int(np.argmax(test_scores))
    best_alpha = alphas[best_idx]
    post = DecisionTreeClassifier(random_state=0,
                                  ccp_alpha=best_alpha).fit(X_tr, y_tr)
    Zq = post.predict(grid).reshape(xx.shape)
    post_train = accuracy_score(y_tr, post.predict(X_tr))
    post_test = accuracy_score(y_te, post.predict(X_te))

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.contourf(xx, yy, Zq, cmap=cmap_bg, alpha=0.9, levels=[-0.5, 0.5, 1.5])
    ax1.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=cmap_pt, s=14,
                edgecolor="white", linewidth=0.4)
    ax1.set_title(f"Post-prune: ccp_alpha = {best_alpha:.4f}\n"
                  f"leaves = {post.get_n_leaves()}    "
                  f"train = {post_train:.2f}    test = {post_test:.2f}",
                  fontsize=10.5, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    # --- Pruning path ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(alphas, train_scores, color=C_BLUE, lw=2.2, marker="o", ms=4,
             label="train")
    ax2.plot(alphas, test_scores, color=C_PURPLE, lw=2.2, marker="s", ms=4,
             label="test")
    ax2.axvline(best_alpha, color=C_GREEN, lw=1.4, ls="--",
                label=rf"best $\alpha$ = {best_alpha:.4f}")
    ax2.scatter([best_alpha], [test_scores[best_idx]], s=110,
                facecolor="white", edgecolor=C_GREEN, lw=2, zorder=5)
    ax2.set_xlabel(r"$\alpha$  (cost-complexity)", fontsize=10.5)
    ax2.set_ylabel("accuracy", fontsize=10.5)
    ax2.set_title("Cost-complexity pruning path",
                  fontsize=11, fontweight="bold", pad=8)
    ax2.legend(loc="lower left", fontsize=9.5, frameon=True)

    fig.suptitle("Pre-pruning stops growth; post-pruning grows then collapses",
                 fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "fig6_pruning")


# ---------------------------------------------------------------------------
# Figure 7: Feature importance on Iris (verified against sklearn)
# ---------------------------------------------------------------------------
def fig7_feature_importance() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = [n.replace(" (cm)", "").replace(" ", "\n")
                     for n in iris.feature_names]

    clf = DecisionTreeClassifier(random_state=0, max_depth=4).fit(X, y)
    imp = clf.feature_importances_
    order = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(10, 5.0))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(imp))]
    colors_sorted = [colors[i] for i in order]
    ax.barh(np.array(feature_names)[order], imp[order],
            color=colors_sorted, edgecolor="white", height=0.62)

    for i, v in enumerate(imp[order]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center",
                fontsize=10.5, color=C_DARK, fontweight="bold")

    ax.set_xlim(0, max(imp) * 1.18)
    ax.set_xlabel("mean decrease in impurity (sklearn-verified)", fontsize=11)
    ax.set_title("Feature importance on Iris  "
                 "(CART, max_depth = 4)",
                 fontsize=12.5, fontweight="bold", pad=10)

    # Side caption with formula
    ax.text(
        0.99, -0.22,
        r"importance$(j) = \sum_{t : \mathrm{splits\ on\ } j} "
        r"\frac{N_t}{N}\,\left(I(t) - \frac{N_{t,L}}{N_t}I(t_L) "
        r"- \frac{N_{t,R}}{N_t}I(t_R)\right)$",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color=C_GRAY,
    )

    fig.tight_layout()
    _save(fig, "fig7_feature_importance")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Writing figures to:\n  EN: {EN_DIR}\n  ZH: {ZH_DIR}\n")
    fig1_tree_structure()
    fig2_impurity_curves()
    fig3_decision_boundary()
    fig4_information_gain()
    fig5_overfitting_curve()
    fig6_pruning()
    fig7_feature_importance()
    print("\nDone.")


if __name__ == "__main__":
    main()
