"""
Figure generation script for ML Math Derivations Part 06:
Logistic Regression and Classification.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure teaches a single specific idea cleanly so the math becomes visible.

Figures:
    fig1_sigmoid                Sigmoid function with its tangent at z=0
                                (slope = 1/4) and the derivative curve.
    fig2_decision_boundary      Logistic regression on 2D data: probability
                                heatmap, contours and the linear boundary.
    fig3_loss_landscape         Cross-entropy vs MSE: loss curve and gradient
                                magnitude when the true label is 1, showing
                                the vanishing-gradient pathology of MSE.
    fig4_softmax_simplex        K=3 probability simplex coloured by the
                                argmax class with sample softmax outputs.
    fig5_roc_pr                 ROC and PR curves on a held-out set with the
                                AUC region shaded.
    fig6_confusion_imbalance    Confusion matrix heatmap on an imbalanced
                                problem with derived precision / recall / F1.
    fig7_feature_space          Logistic regression as a linear classifier:
                                weight vector, decision hyperplane and the
                                signed distance interpretation.

Usage:
    python3 scripts/figures/ml-math-derivations/06-logistic-regression.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch, Polygon

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
    / "06-Logistic-Regression-and-Classification"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "06-逻辑回归与分类"
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
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Figure 1: Sigmoid + tangent at 0 + derivative
# ---------------------------------------------------------------------------
def fig1_sigmoid() -> None:
    z = np.linspace(-7, 7, 600)
    s = sigmoid(z)
    ds = s * (1 - s)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # --- Left: sigmoid + tangent at z=0 (slope 1/4) ---
    ax = axes[0]
    ax.plot(z, s, color=C_BLUE, lw=2.6, label=r"$\sigma(z) = 1 / (1 + e^{-z})$")
    # Asymptotes
    ax.axhline(0, color=C_GRAY, lw=0.8, ls=":")
    ax.axhline(1, color=C_GRAY, lw=0.8, ls=":")
    ax.axhline(0.5, color=C_GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color=C_GRAY, lw=0.8, ls="--", alpha=0.6)
    # Tangent at z=0: slope = 1/4, passes through (0, 0.5)
    z_t = np.linspace(-2.6, 2.6, 50)
    tangent = 0.5 + 0.25 * z_t
    ax.plot(z_t, tangent, color=C_AMBER, lw=2.0, ls="--",
            label=r"tangent at $z=0$: slope $= 1/4$")
    ax.scatter([0], [0.5], color=C_AMBER, s=70, zorder=5, edgecolor="white",
               linewidth=1.5)
    ax.annotate(r"$\sigma(0)=\frac{1}{2},\ \sigma'(0)=\frac{1}{4}$",
                xy=(0, 0.5), xytext=(1.3, 0.18),
                fontsize=11, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))
    # Saturation labels
    ax.text(-6.5, 0.04, r"$\sigma \to 0$", fontsize=10, color=C_GRAY)
    ax.text(4.2, 0.93, r"$\sigma \to 1$", fontsize=10, color=C_GRAY)

    ax.set_xlim(-7, 7)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel("z", fontsize=11)
    ax.set_ylabel(r"$\sigma(z)$", fontsize=11)
    ax.set_title("Sigmoid: real line $\\to$ probability",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)

    # --- Right: derivative ---
    ax = axes[1]
    ax.plot(z, ds, color=C_PURPLE, lw=2.6,
            label=r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$")
    ax.fill_between(z, 0, ds, color=C_PURPLE, alpha=0.12)
    ax.axhline(0.25, color=C_AMBER, lw=1.2, ls="--", alpha=0.8)
    ax.scatter([0], [0.25], color=C_AMBER, s=70, zorder=5, edgecolor="white",
               linewidth=1.5)
    ax.annotate(r"max $= 1/4$ at $z=0$",
                xy=(0, 0.25), xytext=(2.0, 0.22),
                fontsize=11, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))
    # Saturation regions
    ax.axvspan(-7, -4, color=C_GRAY, alpha=0.10)
    ax.axvspan(4, 7, color=C_GRAY, alpha=0.10)
    ax.text(-5.5, 0.22, "saturated\n(grad $\\approx 0$)",
            ha="center", fontsize=9, color=C_GRAY)
    ax.text(5.5, 0.22, "saturated\n(grad $\\approx 0$)",
            ha="center", fontsize=9, color=C_GRAY)

    ax.set_xlim(-7, 7)
    ax.set_ylim(-0.01, 0.30)
    ax.set_xlabel("z", fontsize=11)
    ax.set_ylabel(r"$\sigma'(z)$", fontsize=11)
    ax.set_title("Sigmoid derivative: bounded by 1/4",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)

    fig.suptitle("Sigmoid function and its self-expressing derivative",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_sigmoid")


# ---------------------------------------------------------------------------
# Figure 2: Decision boundary on 2D classification data
# ---------------------------------------------------------------------------
def fig2_decision_boundary() -> None:
    rng = np.random.default_rng(1)
    n = 140
    # Two Gaussian blobs with overlap
    mu0 = np.array([-1.4, -0.6])
    mu1 = np.array([1.6, 0.9])
    cov = np.array([[1.3, 0.5], [0.5, 1.0]])
    X0 = rng.multivariate_normal(mu0, cov, size=n)
    X1 = rng.multivariate_normal(mu1, cov, size=n)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])

    # Augment with bias and fit by gradient descent
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(3)
    lr = 0.1
    for _ in range(2000):
        p = sigmoid(Xb @ w)
        w -= lr * Xb.T @ (p - y) / Xb.shape[0]

    # Probability heatmap
    xx, yy = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-4, 4, 300))
    grid = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    P = sigmoid(grid @ w).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    cmap = LinearSegmentedColormap.from_list(
        "bp", ["#dbeafe", "#ffffff", "#ede9fe"], N=256)
    im = ax.contourf(xx, yy, P, levels=30, cmap=cmap, alpha=0.95)
    # Probability contours
    cs = ax.contour(xx, yy, P, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                    colors=[C_GRAY, C_GRAY, C_DARK, C_GRAY, C_GRAY],
                    linewidths=[0.8, 0.8, 2.0, 0.8, 0.8],
                    linestyles=["--", "--", "-", "--", "--"])
    ax.clabel(cs, inline=True, fontsize=9, fmt="%.1f")

    # Data
    ax.scatter(X0[:, 0], X0[:, 1], color=C_BLUE, s=36, edgecolor="white",
               linewidth=0.8, label="class 0", alpha=0.9)
    ax.scatter(X1[:, 0], X1[:, 1], color=C_PURPLE, s=36, edgecolor="white",
               linewidth=0.8, label="class 1", alpha=0.9)

    # Weight vector arrow at boundary midpoint
    w2 = w[:2]
    w2_unit = w2 / np.linalg.norm(w2)
    # midpoint of boundary near origin: solve w0*x + w1*y + b = 0 with x = 0
    mid = np.array([0.0, -w[2] / w[1]])
    arrow = FancyArrowPatch(mid, mid + 1.3 * w2_unit,
                            arrowstyle="-|>", mutation_scale=18,
                            color=C_AMBER, lw=2.2)
    ax.add_patch(arrow)
    ax.text(mid[0] + 1.4 * w2_unit[0], mid[1] + 1.4 * w2_unit[1] + 0.15,
            r"$\mathbf{w}$", fontsize=13, color=C_AMBER, fontweight="bold")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$x_1$", fontsize=11)
    ax.set_ylabel(r"$x_2$", fontsize=11)
    ax.set_title(r"Decision boundary $\mathbf{w}^\top\mathbf{x}+b=0$ "
                 r"with probability contours",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$P(y=1\mid \mathbf{x})$", fontsize=10)

    fig.tight_layout()
    _save(fig, "fig2_decision_boundary")


# ---------------------------------------------------------------------------
# Figure 3: Cross-entropy vs MSE -- loss + gradient
# ---------------------------------------------------------------------------
def fig3_loss_landscape() -> None:
    p = np.linspace(1e-3, 1 - 1e-3, 600)
    # True label y = 1
    ce = -np.log(p)                 # cross-entropy when y = 1
    mse = 0.5 * (p - 1.0) ** 2      # MSE when y = 1
    # Gradients w.r.t. logit z (assuming p = sigmoid(z)):
    # d CE / dz = p - 1
    # d MSE / dz = (p - 1) * p * (1 - p)
    grad_ce = np.abs(p - 1)
    grad_mse = np.abs((p - 1) * p * (1 - p))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))

    # --- Left: loss curves ---
    ax = axes[0]
    ax.plot(p, ce, color=C_BLUE, lw=2.6, label="cross-entropy $-\\log \\hat y$")
    ax.plot(p, mse, color=C_AMBER, lw=2.6,
            label=r"MSE $\frac{1}{2}(\hat y - 1)^2$")
    ax.axvline(1.0, color=C_GRAY, lw=0.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel(r"$\hat y$ (predicted prob., true $y=1$)", fontsize=11)
    ax.set_ylabel("loss", fontsize=11)
    ax.set_title("Loss when the model is wrong: CE explodes, MSE is mild",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.annotate("CE punishes confident\nwrong answers harshly",
                xy=(0.05, -np.log(0.05)), xytext=(0.30, 3.6),
                fontsize=10, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1))

    # --- Right: gradient magnitude wrt logit z ---
    ax = axes[1]
    ax.plot(p, grad_ce, color=C_BLUE, lw=2.6,
            label=r"$|\partial \mathcal{L}_{CE}/\partial z| = |\hat y - 1|$")
    ax.plot(p, grad_mse, color=C_AMBER, lw=2.6,
            label=r"$|\partial \mathcal{L}_{MSE}/\partial z| = |\hat y - 1|\hat y(1-\hat y)$")
    ax.fill_between(p, grad_mse, color=C_AMBER, alpha=0.10)
    # Highlight vanishing region
    ax.axvspan(0, 0.1, color=C_AMBER, alpha=0.08)
    ax.text(0.05, 0.85, "MSE gradient\nvanishes here",
            ha="center", fontsize=9.5, color=C_AMBER)
    ax.annotate("CE gradient is\nlargest when wrong",
                xy=(0.04, 0.96), xytext=(0.32, 0.78),
                fontsize=10, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\hat y$ (predicted prob., true $y=1$)", fontsize=11)
    ax.set_ylabel(r"$|\nabla_z \mathcal{L}|$", fontsize=11)
    ax.set_title("Gradient magnitude: why CE keeps learning, MSE stalls",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)

    fig.suptitle("Cross-entropy vs MSE for classification",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_loss_landscape")


# ---------------------------------------------------------------------------
# Figure 4: Softmax probability simplex (K = 3)
# ---------------------------------------------------------------------------
def fig4_softmax_simplex() -> None:
    """Triangle simplex for K = 3, coloured by argmax region."""
    # Barycentric -> 2D embedding
    A = np.array([0.0, 0.0])              # P_1 corner
    B = np.array([1.0, 0.0])              # P_2 corner
    C = np.array([0.5, np.sqrt(3) / 2])   # P_3 corner

    def bary_to_xy(p1, p2, p3):
        return p1 * A + p2 * B + p3 * C

    # Sample grid inside simplex, colour by argmax
    res = 400
    u = np.linspace(0, 1, res)
    v = np.linspace(0, 1, res)
    UU, VV = np.meshgrid(u, v)
    P1 = 1 - UU - VV
    P2 = UU
    P3 = VV
    inside = (P1 >= 0) & (P2 >= 0) & (P3 >= 0)

    argmax = np.full_like(P1, np.nan)
    stacked = np.stack([P1, P2, P3], axis=-1)
    argmax_idx = np.argmax(stacked, axis=-1).astype(float)
    argmax_idx[~inside] = np.nan

    XY = np.stack([bary_to_xy(P1[i, j], P2[i, j], P3[i, j])
                   for i in range(res) for j in range(res)]).reshape(res, res, 2)

    fig, ax = plt.subplots(figsize=(7.8, 6.6))

    # Plot region as scatter on transformed grid
    cmap = ListedColormap([C_BLUE, C_PURPLE, C_GREEN])
    ax.scatter(XY[..., 0][inside], XY[..., 1][inside],
               c=argmax_idx[inside], cmap=cmap, s=4, alpha=0.32,
               marker="s", edgecolors="none")

    # Triangle edges
    tri = Polygon([A, B, C], closed=True, fill=False,
                  edgecolor=C_DARK, linewidth=2)
    ax.add_patch(tri)

    # Centre = uniform
    centre = bary_to_xy(1 / 3, 1 / 3, 1 / 3)
    ax.scatter(*centre, color=C_DARK, s=80, zorder=5, edgecolor="white",
               linewidth=1.8)
    ax.annotate(r"uniform $(\frac{1}{3},\frac{1}{3},\frac{1}{3})$",
                xy=centre, xytext=(centre[0] + 0.18, centre[1] - 0.05),
                fontsize=10.5, color=C_DARK)

    # A few example softmax outputs
    examples = [
        (np.array([2.5, 0.5, 0.5]), "logits (2.5, 0.5, 0.5)"),
        (np.array([1.0, 2.5, 0.0]), "logits (1.0, 2.5, 0.0)"),
        (np.array([0.0, 0.0, 3.0]), "logits (0.0, 0.0, 3.0)"),
    ]
    for z, lab in examples:
        e = np.exp(z - z.max())
        p = e / e.sum()
        xy = bary_to_xy(p[0], p[1], p[2])
        ax.scatter(*xy, color=C_AMBER, s=110, zorder=5,
                   edgecolor=C_DARK, linewidth=1.4)
        ax.annotate(lab, xy=xy, xytext=(xy[0] + 0.04, xy[1] + 0.05),
                    fontsize=9.5, color=C_DARK,
                    arrowprops=dict(arrowstyle="-", color=C_DARK, lw=0.8))

    # Vertex labels
    ax.text(A[0] - 0.06, A[1] - 0.05, r"$P_1=1$ (class 1)",
            fontsize=11, color=C_BLUE, fontweight="bold")
    ax.text(B[0] - 0.06, B[1] - 0.05, r"$P_2=1$ (class 2)",
            fontsize=11, color=C_PURPLE, fontweight="bold")
    ax.text(C[0] - 0.13, C[1] + 0.04, r"$P_3=1$ (class 3)",
            fontsize=11, color=C_GREEN, fontweight="bold")

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"Softmax probability simplex ($K=3$): every output lives "
                 r"inside the triangle",
                 fontsize=12.5, fontweight="bold", pad=10)

    fig.tight_layout()
    _save(fig, "fig4_softmax_simplex")


# ---------------------------------------------------------------------------
# Figure 5: ROC and PR curves with AUC shading
# ---------------------------------------------------------------------------
def fig5_roc_pr() -> None:
    rng = np.random.default_rng(3)
    # Synthetic scores: positive class higher mean
    n_pos, n_neg = 200, 600
    scores_pos = rng.normal(loc=1.2, scale=1.0, size=n_pos)
    scores_neg = rng.normal(loc=-0.4, scale=1.0, size=n_neg)
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # Sweep thresholds
    thr = np.sort(np.unique(scores))[::-1]
    tpr_list, fpr_list, prec_list, rec_list = [], [], [], []
    for t in thr:
        pred = scores >= t
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        tpr_list.append(tp / max(tp + fn, 1))
        fpr_list.append(fp / max(fp + tn, 1))
        prec_list.append(tp / max(tp + fp, 1))
        rec_list.append(tp / max(tp + fn, 1))
    tpr = np.array([0.0] + tpr_list + [1.0])
    fpr = np.array([0.0] + fpr_list + [1.0])
    prec = np.array([1.0] + prec_list)
    rec = np.array([0.0] + rec_list)

    # AUC by trapezoid
    auc_roc = np.trapz(tpr, fpr)
    # Sort PR by recall ascending for area
    order = np.argsort(rec)
    auc_pr = np.trapz(prec[order], rec[order])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))

    # --- ROC ---
    ax = axes[0]
    ax.fill_between(fpr, 0, tpr, color=C_BLUE, alpha=0.18,
                    label=f"AUC = {auc_roc:.3f}")
    ax.plot(fpr, tpr, color=C_BLUE, lw=2.4)
    ax.plot([0, 1], [0, 1], color=C_GRAY, lw=1.2, ls="--",
            label="random (AUC = 0.5)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false positive rate", fontsize=11)
    ax.set_ylabel("true positive rate (recall)", fontsize=11)
    ax.set_title("ROC curve", fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_aspect("equal")

    # --- PR ---
    ax = axes[1]
    ax.fill_between(rec[order], 0, prec[order], color=C_PURPLE, alpha=0.18,
                    label=f"AP = {auc_pr:.3f}")
    ax.plot(rec[order], prec[order], color=C_PURPLE, lw=2.4)
    baseline = n_pos / (n_pos + n_neg)
    ax.axhline(baseline, color=C_GRAY, lw=1.2, ls="--",
               label=f"baseline = {baseline:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall", fontsize=11)
    ax.set_ylabel("precision", fontsize=11)
    ax.set_title("Precision–Recall curve", fontsize=12.5, fontweight="bold",
                 pad=10)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.set_aspect("equal")

    fig.suptitle("Threshold-free evaluation: ROC and PR with AUC shaded",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_roc_pr")


# ---------------------------------------------------------------------------
# Figure 6: Confusion matrix on imbalanced data
# ---------------------------------------------------------------------------
def fig6_confusion_imbalance() -> None:
    """Show how accuracy lies under class imbalance."""
    # Imbalanced ground truth: 950 negatives, 50 positives
    # Naive model: predicts majority class -> very high accuracy, useless.
    # We compare it with a logistic-regression style model.
    rng = np.random.default_rng(11)
    n_neg, n_pos = 950, 50
    # Naive (predict all negative)
    cm_naive = np.array([
        [n_neg, 0],     # actual negative
        [n_pos, 0],     # actual positive
    ])
    # Trained model: high recall on positives but some FPs
    tp = 42
    fn = n_pos - tp
    fp = 35
    tn = n_neg - fp
    cm_model = np.array([
        [tn, fp],
        [fn, tp],
    ])

    def metrics(cm):
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        acc = (tp + tn) / cm.sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return acc, prec, rec, f1

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    cmap = LinearSegmentedColormap.from_list("bp", ["#ffffff", C_BLUE], N=256)
    titles = ["Naive 'always negative' classifier",
              "Trained logistic regression"]
    cms = [cm_naive, cm_model]

    for ax, cm, title in zip(axes, cms, titles):
        # Use log-ish normalisation by row totals so rare class cell visible
        norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                colour = "white" if norm[i, j] >= 0.5 else C_DARK
                ax.text(j, i, f"{v}", ha="center", va="center",
                        color=colour, fontsize=18, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred. neg.", "pred. pos."])
        ax.set_yticklabels(["actual neg.", "actual pos."])
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
        ax.grid(False)

        acc, prec, rec, f1 = metrics(cm)
        ax.text(0.5, 1.18,
                f"acc = {acc:.3f}    prec = {prec:.3f}    "
                f"rec = {rec:.3f}    F1 = {f1:.3f}",
                transform=ax.transAxes, ha="center", fontsize=10.5,
                color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                          edgecolor="none"))

    fig.suptitle("Class imbalance: accuracy hides what really matters",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, "fig6_confusion_imbalance")


# ---------------------------------------------------------------------------
# Figure 7: Logistic regression as linear classifier in feature space
# ---------------------------------------------------------------------------
def fig7_feature_space() -> None:
    rng = np.random.default_rng(5)
    n = 60
    X0 = rng.normal(loc=[-1.8, -0.6], scale=[0.9, 0.9], size=(n, 2))
    X1 = rng.normal(loc=[1.6, 0.8], scale=[0.9, 0.9], size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])

    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(3)
    for _ in range(2000):
        p = sigmoid(Xb @ w)
        w -= 0.1 * Xb.T @ (p - y) / Xb.shape[0]

    a, b, c = w  # boundary: a*x + b*y + c = 0
    # Boundary line over x range
    x_vals = np.linspace(-5, 5, 200)
    y_line = -(a * x_vals + c) / b

    # Margin lines at p = 0.27 and 0.73 (z = +-1)
    y_low = -(a * x_vals + c - 1) / b
    y_high = -(a * x_vals + c + 1) / b

    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    # Background half-plane shading
    yy_top = np.maximum(y_line, -4)
    ax.fill_between(x_vals, y_line, 4, color=C_PURPLE, alpha=0.07)
    ax.fill_between(x_vals, -4, y_line, color=C_BLUE, alpha=0.07)

    # Margin lines
    ax.plot(x_vals, y_high, color=C_GRAY, lw=1.0, ls=":", alpha=0.8)
    ax.plot(x_vals, y_low, color=C_GRAY, lw=1.0, ls=":", alpha=0.8)
    ax.text(x_vals[-1] - 0.2, y_high[-1] + 0.1, r"$\hat y \approx 0.73$",
            fontsize=9, color=C_GRAY, ha="right")
    ax.text(x_vals[-1] - 0.2, y_low[-1] - 0.4, r"$\hat y \approx 0.27$",
            fontsize=9, color=C_GRAY, ha="right")

    # Boundary
    ax.plot(x_vals, y_line, color=C_DARK, lw=2.4,
            label=r"boundary $\mathbf{w}^\top\mathbf{x}+b=0$ ($\hat y = 0.5$)")

    # Data
    ax.scatter(X0[:, 0], X0[:, 1], color=C_BLUE, s=42, edgecolor="white",
               linewidth=0.9, label="class 0", alpha=0.95)
    ax.scatter(X1[:, 0], X1[:, 1], color=C_PURPLE, s=42, edgecolor="white",
               linewidth=0.9, label="class 1", alpha=0.95)

    # Weight vector arrow from a point on boundary
    base = np.array([0.0, -c / b])
    w2 = np.array([a, b])
    w2_unit = w2 / np.linalg.norm(w2)
    arrow = FancyArrowPatch(base, base + 1.6 * w2_unit,
                            arrowstyle="-|>", mutation_scale=20,
                            color=C_AMBER, lw=2.4)
    ax.add_patch(arrow)
    ax.text(base[0] + 1.7 * w2_unit[0], base[1] + 1.7 * w2_unit[1] + 0.18,
            r"$\mathbf{w}$ (normal vector)",
            fontsize=11, color=C_AMBER, fontweight="bold")

    # Highlight one point + its signed distance line
    pt = np.array([2.6, -1.4])
    d_signed = (a * pt[0] + b * pt[1] + c) / np.linalg.norm(w2)
    foot = pt - d_signed * w2_unit
    ax.scatter(*pt, color=C_AMBER, s=110, zorder=6, edgecolor=C_DARK,
               linewidth=1.4)
    ax.plot([pt[0], foot[0]], [pt[1], foot[1]], color=C_AMBER, lw=1.6,
            ls="--")
    ax.annotate(
        r"signed distance $d = \frac{\mathbf{w}^\top\mathbf{x}+b}{\|\mathbf{w}\|}$",
        xy=((pt[0] + foot[0]) / 2, (pt[1] + foot[1]) / 2),
        xytext=(0.5, -3.4), fontsize=10.5, color=C_DARK,
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$x_1$", fontsize=11)
    ax.set_ylabel(r"$x_2$", fontsize=11)
    ax.set_title("Logistic regression is a linear classifier: weight vector "
                 "= boundary normal",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    fig.tight_layout()
    _save(fig, "fig7_feature_space")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN -> {EN_DIR}")
    print(f"ZH -> {ZH_DIR}")
    print()
    for fn in (
        fig1_sigmoid,
        fig2_decision_boundary,
        fig3_loss_landscape,
        fig4_softmax_simplex,
        fig5_roc_pr,
        fig6_confusion_imbalance,
        fig7_feature_space,
    ):
        print(f"[render] {fn.__name__}")
        fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
