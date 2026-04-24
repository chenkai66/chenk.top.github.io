"""
Figure generation script for ML Math Derivations Chapter 01:
"Introduction and Mathematical Foundations".

Generates 7 figures used in BOTH the EN and ZH versions of the article.
Each figure conveys one core idea from learning theory cleanly, with
consistent typography, palette and DPI across the series.

Figures:
    fig1_problem_types          Supervised vs unsupervised vs reinforcement
                                learning, illustrated with three small panels
                                showing labeled data, clusters, and an agent
                                trajectory in a grid world.
    fig2_loss_functions         Comparison of MSE, MAE, Cross-Entropy and
                                Hinge losses on the same axes, with
                                annotations highlighting their geometry.
    fig3_bias_variance_tradeoff Train and test error curves vs model
                                complexity, with the bias-dominated and
                                variance-dominated regions shaded.
    fig4_generalization_gap     Train vs test error trajectories during
                                training; the visible gap is the
                                generalization gap.
    fig5_vc_shattering          VC dimension intuition: a linear classifier
                                in 2D can shatter 3 generic points (8
                                labelings shown), but cannot shatter the XOR
                                pattern of 4 points.
    fig6_pac_bounds             PAC sample complexity m as a function of
                                accuracy epsilon and confidence delta, on a
                                log scale, for several hypothesis-space
                                sizes.
    fig7_function_approximation Linear vs polynomial vs neural-style
                                approximation of a non-linear target,
                                showing how inductive bias shapes the
                                solution.

Usage:
    python3 scripts/figures/ml-math-derivations/01-introduction-foundations.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages. Parent folders are created
    if missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

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
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = COLORS["bg"]

DPI = 150

# plt.rcParams.update({
#     "font.family": "DejaVu Sans",
#     "axes.titleweight": "bold",
#     "axes.titlesize": 12,
#     "axes.labelsize": 11,
#     "axes.edgecolor": C_DARK,
#     "axes.linewidth": 0.8,
#     "xtick.color": C_DARK,
#     "ytick.color": C_DARK,
#     "grid.color": C_LIGHT,
#     "grid.linewidth": 0.6,
#     "legend.frameon": False,
#     "legend.fontsize": 10,
# })

# ---------------------------------------------------------------------------
# Output paths -- script writes to BOTH language asset folders
# ---------------------------------------------------------------------------
ROOT = Path("/Users/kchen/Desktop/Project/chenk-site/source/_posts")
EN_DIR = ROOT / "en" / "ml-math-derivations" / "01-Introduction-and-Mathematical-Foundations"
ZH_DIR = ROOT / "zh" / "ml-math-derivations" / "01-绪论与数学基础"

EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIRS = [EN_DIR, ZH_DIR]


def save(fig: plt.Figure, name: str) -> None:
    """Save the same PNG into every output directory."""
    for d in OUT_DIRS:
        path = d / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig1: ML problem types -- supervised, unsupervised, reinforcement
# ---------------------------------------------------------------------------
def fig1_problem_types() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    rng = np.random.default_rng(7)

    # Supervised: two labelled classes with a decision line
    ax = axes[0]
    n = 40
    x1 = rng.normal(loc=[-1.2, -0.6], scale=0.55, size=(n, 2))
    x2 = rng.normal(loc=[1.2, 0.8], scale=0.55, size=(n, 2))
    ax.scatter(x1[:, 0], x1[:, 1], s=34, c=C_BLUE, edgecolors="white",
               linewidths=0.6, label="class A")
    ax.scatter(x2[:, 0], x2[:, 1], s=34, c=C_AMBER, edgecolors="white",
               linewidths=0.6, label="class B")
    xs = np.linspace(-3, 3, 50)
    ax.plot(xs, -0.7 * xs + 0.1, color=C_DARK, lw=2.0,
            label=r"learned $h(x)$")
    ax.set_title("Supervised learning\n(features $\\to$ labels)")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.6, 2.6)
    ax.legend(loc="lower right")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # Unsupervised: three clusters discovered without labels
    ax = axes[1]
    centres = np.array([[-1.4, 0.6], [1.3, 1.1], [0.0, -1.4]])
    cols = [C_BLUE, C_PURPLE, C_GREEN]
    for c, col in zip(centres, cols):
        pts = rng.normal(loc=c, scale=0.35, size=(28, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=34, c=col,
                   edgecolors="white", linewidths=0.6)
    for c, col in zip(centres, cols):
        ax.scatter(*c, s=180, c=col, marker="X",
                   edgecolors=C_DARK, linewidths=1.5)
    ax.set_title("Unsupervised learning\n(structure without labels)")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # Reinforcement: agent on a small grid, trajectory toward goal
    ax = axes[2]
    G = 5
    for i in range(G + 1):
        ax.axhline(i, color=C_LIGHT, lw=0.8)
        ax.axvline(i, color=C_LIGHT, lw=0.8)
    # obstacle
    ax.add_patch(Rectangle((2, 1), 1, 2, facecolor=C_GRAY, alpha=0.45))
    # goal
    ax.add_patch(Rectangle((4, 4), 1, 1, facecolor=C_GREEN, alpha=0.45))
    ax.text(4.5, 4.5, "goal", ha="center", va="center",
            fontsize=10, color=C_DARK, weight="bold")
    # start
    ax.add_patch(Circle((0.5, 0.5), 0.28, facecolor=C_BLUE,
                        edgecolor=C_DARK, lw=1.2))
    ax.text(0.5, -0.3, "agent", ha="center", fontsize=10, color=C_DARK)
    # trajectory
    path = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (1.5, 3.5),
            (3.5, 3.5), (4.5, 4.5)]
    px, py = zip(*path)
    ax.plot(px, py, color=C_AMBER, lw=2.4, marker="o",
            markersize=6, markerfacecolor="white", markeredgecolor=C_AMBER)
    # reward arrows
    ax.annotate("", xy=(4.5, 4.5), xytext=(3.7, 3.7),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2.0))
    ax.set_title("Reinforcement learning\n(actions $\\to$ rewards)")
    ax.set_xlim(-0.4, G + 0.2)
    ax.set_ylim(-0.6, G + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Three families of learning problems",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_problem_types")


# ---------------------------------------------------------------------------
# fig2: Loss function comparison
# ---------------------------------------------------------------------------
def fig2_loss_functions() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: regression losses on residual r = y - h(x)
    ax = axes[0]
    r = np.linspace(-3, 3, 400)
    mse = r ** 2
    mae = np.abs(r)
    huber_delta = 1.0
    huber = np.where(np.abs(r) <= huber_delta,
                     0.5 * r ** 2,
                     huber_delta * (np.abs(r) - 0.5 * huber_delta))
    ax.plot(r, mse, color=C_BLUE, lw=2.4, label=r"MSE  $r^2$")
    ax.plot(r, mae, color=C_AMBER, lw=2.4, label=r"MAE  $|r|$")
    ax.plot(r, huber, color=C_GREEN, lw=2.4, ls="--",
            label=r"Huber  ($\delta{=}1$)")
    ax.set_title("Regression losses (residual $r = y - h(x)$)")
    ax.set_xlabel("residual $r$")
    ax.set_ylabel("loss")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 6)
    ax.legend(loc="upper center")
    ax.annotate("MSE punishes\noutliers heavily",
                xy=(2.4, 5.76), xytext=(0.4, 5.2),
                fontsize=9, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0))

    # Right: classification losses on margin m = y * h(x)
    ax = axes[1]
    m = np.linspace(-3, 3, 400)
    zero_one = (m <= 0).astype(float)
    hinge = np.maximum(0.0, 1.0 - m)
    # Logistic / cross-entropy on margin
    logistic = np.log1p(np.exp(-m)) / np.log(2)
    # Exponential loss (AdaBoost flavour) for context
    expo = np.exp(-m)
    ax.plot(m, zero_one, color=C_GRAY, lw=2.4,
            label=r"0-1  $\mathbb{1}[m \leq 0]$")
    ax.plot(m, hinge, color=C_PURPLE, lw=2.4, label=r"Hinge  $\max(0, 1-m)$")
    ax.plot(m, logistic, color=C_BLUE, lw=2.4,
            label=r"Cross-entropy $\log_2(1{+}e^{-m})$")
    ax.plot(m, expo, color=C_RED, lw=2.0, ls="--",
            label=r"Exponential  $e^{-m}$")
    ax.set_title("Classification surrogates for the 0-1 loss")
    ax.set_xlabel(r"margin  $m = y \cdot h(x)$")
    ax.set_ylabel("loss")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 5)
    ax.axvline(0, color=C_DARK, lw=0.6, alpha=0.5)
    ax.axvline(1, color=C_PURPLE, lw=0.6, ls=":", alpha=0.7)
    ax.text(1.05, 4.6, "hinge\nelbow", color=C_PURPLE, fontsize=9)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save(fig, "fig2_loss_functions")


# ---------------------------------------------------------------------------
# fig3: Bias-variance tradeoff
# ---------------------------------------------------------------------------
def fig3_bias_variance_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    c = np.linspace(0.5, 12, 400)
    bias_sq = 6.0 / (c ** 1.4) + 0.05
    variance = 0.02 * c ** 1.6
    noise = np.full_like(c, 0.4)
    total = bias_sq + variance + noise

    ax.plot(c, bias_sq, color=C_BLUE, lw=2.4, label=r"Bias$^2$")
    ax.plot(c, variance, color=C_AMBER, lw=2.4, label="Variance")
    ax.plot(c, noise, color=C_GRAY, lw=1.6, ls="--",
            label=r"Noise $\sigma^2$ (irreducible)")
    ax.plot(c, total, color=C_DARK, lw=2.8, label="Expected test error")

    # Mark the optimum
    opt_idx = int(np.argmin(total))
    opt_c = c[opt_idx]
    opt_e = total[opt_idx]
    ax.scatter([opt_c], [opt_e], s=120, c=C_GREEN, zorder=5,
               edgecolor=C_DARK, lw=1.2)
    ax.annotate("optimal\ncomplexity",
                xy=(opt_c, opt_e), xytext=(opt_c + 1.5, opt_e + 1.2),
                fontsize=10, color=C_GREEN, weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2))

    # Shade regimes
    ax.axvspan(0.5, opt_c, alpha=0.06, color=C_BLUE)
    ax.axvspan(opt_c, 12, alpha=0.06, color=C_AMBER)
    ax.text(opt_c * 0.45, 5.0, "underfitting\n(high bias)",
            ha="center", color=C_BLUE, fontsize=10, alpha=0.9)
    ax.text(opt_c + (12 - opt_c) * 0.55, 5.0,
            "overfitting\n(high variance)",
            ha="center", color=C_AMBER, fontsize=10, alpha=0.9)

    ax.set_xlabel("model complexity (e.g. polynomial degree, depth, params)")
    ax.set_ylabel("error")
    ax.set_title("The bias-variance tradeoff")
    ax.set_xlim(0.5, 12)
    ax.set_ylim(0, 6)
    ax.legend(loc="upper center", ncol=2)

    fig.tight_layout()
    save(fig, "fig3_bias_variance_tradeoff")


# ---------------------------------------------------------------------------
# fig4: Generalization gap during training
# ---------------------------------------------------------------------------
def fig4_generalization_gap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: train vs test as a function of training-set size m (good case)
    ax = axes[0]
    m = np.logspace(1, 4, 300)
    train_good = 0.18 + 0.6 / np.sqrt(m)        # rises then plateaus
    test_good = 0.22 + 4.5 / np.sqrt(m)         # falls then plateaus
    ax.plot(m, train_good, color=C_BLUE, lw=2.4, label="training error")
    ax.plot(m, test_good, color=C_AMBER, lw=2.4, label="test error")
    ax.fill_between(m, train_good, test_good, color=C_GRAY, alpha=0.18,
                    label="generalization gap")
    ax.set_xscale("log")
    ax.set_xlabel("training-set size $m$ (log scale)")
    ax.set_ylabel("error")
    ax.set_title("Healthy regime: gap shrinks as $m$ grows")
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right")

    # Right: train vs test over training epochs (overfitting case)
    ax = axes[1]
    epoch = np.linspace(0, 100, 400)
    train_of = 1.4 * np.exp(-epoch / 18) + 0.05
    test_of = 0.9 * np.exp(-epoch / 12) + 0.25 + 0.0035 * np.maximum(epoch - 35, 0)
    ax.plot(epoch, train_of, color=C_BLUE, lw=2.4, label="training error")
    ax.plot(epoch, test_of, color=C_AMBER, lw=2.4, label="test error")
    ax.fill_between(epoch, train_of, test_of, where=test_of >= train_of,
                    color=C_RED, alpha=0.15, label="generalization gap")
    # mark the early-stopping point
    es_idx = int(np.argmin(test_of))
    ax.axvline(epoch[es_idx], color=C_GREEN, lw=1.6, ls="--")
    ax.text(epoch[es_idx] + 1.5, 1.35, "early-stopping\noptimum",
            color=C_GREEN, fontsize=10)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("error")
    ax.set_title("Overfitting regime: test error eventually rises")
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save(fig, "fig4_generalization_gap")


# ---------------------------------------------------------------------------
# fig5: VC dimension -- shattering 3 points, failing on XOR
# ---------------------------------------------------------------------------
def fig5_vc_shattering() -> None:
    # 3 points in general position, 8 labelings, plus an XOR failure panel
    pts = np.array([[-1.0, -0.6], [1.0, -0.4], [0.0, 0.9]])

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.25],
                          hspace=0.35, wspace=0.35)

    # 8 labelings: each axis is one labeling pattern
    labelings = []
    for k in range(8):
        bits = [(k >> i) & 1 for i in range(3)]   # 0/1
        labelings.append([1 if b else -1 for b in bits])

    def find_separator(p, y):
        """Return (w, b) for a line w.x + b = 0 that separates labels."""
        # Brute-force over many candidate lines.
        best = None
        rng = np.random.default_rng(0)
        for _ in range(4000):
            theta = rng.uniform(0, np.pi)
            w = np.array([np.cos(theta), np.sin(theta)])
            for b in np.linspace(-2.5, 2.5, 80):
                preds = np.sign(p @ w + b)
                if np.all(preds == y):
                    return w, b
        return best

    panels = [(0, i) for i in range(4)] + [(1, i) for i in range(4)]
    for (r, c), y in zip(panels, labelings):
        ax = fig.add_subplot(gs[r, c])
        sep = find_separator(pts, np.array(y))
        # plot line if found
        if sep is not None:
            w, b = sep
            xs = np.linspace(-1.8, 1.8, 50)
            # w0*x + w1*y + b = 0 -> y = -(w0*x + b)/w1
            if abs(w[1]) > 1e-3:
                ys = -(w[0] * xs + b) / w[1]
                ax.plot(xs, ys, color=C_DARK, lw=1.8)
            else:
                xv = -b / w[0]
                ax.axvline(xv, color=C_DARK, lw=1.8)
        for (px, py), yi in zip(pts, y):
            col = C_BLUE if yi > 0 else C_AMBER
            ax.scatter(px, py, s=160, c=col, edgecolors=C_DARK, lw=1.2,
                       zorder=4)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.5, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"labeling {''.join('+' if v > 0 else '-' for v in y)}",
                     fontsize=10)

    # Right column: XOR failure panel spanning both rows
    ax = fig.add_subplot(gs[:, 4])
    xor_pts = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    xor_lab = [1, 1, -1, -1]
    for (px, py), yi in zip(xor_pts, xor_lab):
        col = C_BLUE if yi > 0 else C_AMBER
        ax.scatter(px, py, s=240, c=col, edgecolors=C_DARK, lw=1.4,
                   zorder=4)
    # Try a few lines and show none work
    for theta in np.linspace(0, np.pi, 7, endpoint=False):
        w = np.array([np.cos(theta), np.sin(theta)])
        xs = np.linspace(-1.8, 1.8, 50)
        if abs(w[1]) > 1e-3:
            ys = -(w[0] * xs) / w[1]
            ax.plot(xs, ys, color=C_GRAY, lw=1.0, alpha=0.55)
    ax.text(0, -1.7, "no line can separate XOR",
            ha="center", color=C_RED, fontsize=11, weight="bold")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("4 points, XOR pattern\n$\\to$ NOT shatterable", fontsize=11)

    fig.suptitle("VC dimension of 2D linear classifiers: shatters 3 points, "
                 "fails on 4 (XOR)",
                 fontsize=13, weight="bold", y=1.0)
    save(fig, "fig5_vc_shattering")


# ---------------------------------------------------------------------------
# fig6: PAC sample complexity bounds
# ---------------------------------------------------------------------------
def fig6_pac_bounds() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: m vs epsilon for several |H|, fixed delta=0.05 (realizable case)
    ax = axes[0]
    eps = np.linspace(0.01, 0.5, 400)
    delta = 0.05
    H_sizes = [10, 1_000, 1_000_000]
    cols = [C_GREEN, C_BLUE, C_PURPLE]
    for H, col in zip(H_sizes, cols):
        m = (np.log(H) + np.log(1 / delta)) / eps
        ax.plot(eps, m, color=col, lw=2.4,
                label=fr"$|\mathcal{{H}}|=10^{{{int(np.log10(H))}}}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"accuracy parameter $\varepsilon$")
    ax.set_ylabel(r"required samples $m$ (log scale)")
    ax.set_title(r"Realizable PAC: $m \geq \frac{1}{\varepsilon}"
                 r"(\ln|\mathcal{H}| + \ln\frac{1}{\delta})$")
    ax.legend(loc="upper right")
    ax.set_xlim(0.01, 0.5)

    # Right: realizable vs agnostic dependence on epsilon, fixed |H|, delta
    ax = axes[1]
    H = 1_000_000
    delta = 0.05
    eps2 = np.linspace(0.01, 0.5, 400)
    m_real = (np.log(H) + np.log(1 / delta)) / eps2
    m_agno = 2 * (np.log(H) + np.log(2 / delta)) / (eps2 ** 2)
    ax.plot(eps2, m_real, color=C_BLUE, lw=2.4,
            label=r"realizable  $\sim 1/\varepsilon$")
    ax.plot(eps2, m_agno, color=C_RED, lw=2.4,
            label=r"agnostic  $\sim 1/\varepsilon^2$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"required samples $m$ (log scale)")
    ax.set_title("Realizable vs agnostic: the price of dropping realizability")
    ax.legend(loc="upper right")
    ax.set_xlim(0.01, 0.5)
    ax.annotate(r"$10\times$ better $\varepsilon$"
                "\n$\\Rightarrow 100\\times$ more samples",
                xy=(0.05, m_agno[np.argmin(np.abs(eps2 - 0.05))]),
                xytext=(0.18, 1e7),
                fontsize=10, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))

    fig.tight_layout()
    save(fig, "fig6_pac_bounds")


# ---------------------------------------------------------------------------
# fig7: Function approximation -- linear vs polynomial vs neural-style
# ---------------------------------------------------------------------------
def fig7_function_approximation() -> None:
    rng = np.random.default_rng(2)
    x_train = np.sort(rng.uniform(-1, 1, 18))
    y_true = lambda x: np.sin(2.5 * x) + 0.3 * x
    noise = rng.normal(0, 0.18, x_train.shape)
    y_train = y_true(x_train) + noise
    x_grid = np.linspace(-1, 1, 400)

    # Linear fit
    A1 = np.vstack([x_train, np.ones_like(x_train)]).T
    w1, *_ = np.linalg.lstsq(A1, y_train, rcond=None)
    y_lin = w1[0] * x_grid + w1[1]

    # Polynomial fit (degree 4 -- well-matched)
    p4 = np.polyfit(x_train, y_train, 4)
    y_poly = np.polyval(p4, x_grid)

    # "Neural" approximation: random ReLU features + ridge regression
    n_feat = 60
    W = rng.normal(0, 2.5, n_feat)
    b = rng.uniform(-1.5, 1.5, n_feat)
    def relu_feats(x):
        z = np.outer(x, W) + b
        return np.maximum(0, z)
    Phi_train = relu_feats(x_train)
    Phi_grid = relu_feats(x_grid)
    lam = 0.5
    A = Phi_train.T @ Phi_train + lam * np.eye(n_feat)
    coeff = np.linalg.solve(A, Phi_train.T @ y_train)
    y_nn = Phi_grid @ coeff

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    titles = [
        ("Linear  (high bias)", y_lin, C_BLUE),
        ("Polynomial deg 4  (balanced)", y_poly, C_PURPLE),
        ("Random ReLU features  (flexible)", y_nn, C_GREEN),
    ]
    for ax, (title, y_fit, col) in zip(axes, titles):
        ax.plot(x_grid, y_true(x_grid), color=C_GRAY, lw=2.0, ls="--",
                label="true $f^\\star(x)$")
        ax.plot(x_grid, y_fit, color=col, lw=2.6, label="fitted $h(x)$")
        ax.scatter(x_train, y_train, s=36, c=C_DARK,
                   edgecolors="white", lw=0.6, zorder=5,
                   label="training data")
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.7, 1.7)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("$y$")

    fig.suptitle("Inductive bias controls how a hypothesis class "
                 "approximates the truth",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_function_approximation")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for ML Math Derivations Chapter 01")
    print(f"Targets: {[str(d) for d in OUT_DIRS]}")
    fig1_problem_types()
    fig2_loss_functions()
    fig3_bias_variance_tradeoff()
    fig4_generalization_gap()
    fig5_vc_shattering()
    fig6_pac_bounds()
    fig7_function_approximation()
    print("Done.")


if __name__ == "__main__":
    main()
