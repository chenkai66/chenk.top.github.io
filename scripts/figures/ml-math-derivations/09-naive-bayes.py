"""
Figure generation script for ML Math Derivations Part 09: Naive Bayes.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates one geometric or probabilistic intuition behind Naive Bayes
so that the math becomes visible.

Figures:
    fig1_class_conditional      Two 2D class-conditional Gaussians with the
                                Gaussian Naive Bayes decision boundary
                                (axis-aligned ellipses, single boundary curve).
    fig2_posterior_surface      The posterior probability P(c=1 | x) as a
                                heat surface with the 0.5 contour, showing
                                how soft confidence behaves around the
                                decision boundary.
    fig3_independence           Side-by-side: a *correlated* class-conditional
                                density vs. its naive (independent) factorised
                                approximation. Same marginals, different joint.
    fig4_bag_of_words           Bag-of-words feature extraction for text
                                classification: documents -> term-frequency
                                matrix -> per-class word probability table.
    fig5_three_variants         Multinomial vs. Gaussian vs. Bernoulli NB:
                                what each variant assumes about a single
                                feature's distribution.
    fig6_laplace_smoothing      Effect of pseudocount alpha on estimated word
                                probabilities, showing how the zero-probability
                                problem is fixed and how alpha trades MLE vs.
                                uniform prior.
    fig7_spam_decision          A worked spam classifier: per-feature log-odds
                                contributions stacked into a final log
                                posterior ratio (waterfall + bar chart).

Usage:
    python3 scripts/figures/ml-math-derivations/09-naive-bayes.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle

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

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "09-Naive-Bayes"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "09-朴素贝叶斯"
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
def gaussian_2d(x, y, mu, sigma):
    """Diagonal Gaussian density on a meshgrid (Naive Bayes assumption)."""
    return (
        np.exp(-0.5 * (((x - mu[0]) / sigma[0]) ** 2 + ((y - mu[1]) / sigma[1]) ** 2))
        / (2 * np.pi * sigma[0] * sigma[1])
    )


def gaussian_2d_full(x, y, mu, cov):
    """Full-covariance 2D Gaussian density on a meshgrid."""
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    dx = x - mu[0]
    dy = y - mu[1]
    q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))


# ---------------------------------------------------------------------------
# Figure 1: Class-conditional Gaussians + decision boundary
# ---------------------------------------------------------------------------
def fig1_class_conditional() -> None:
    rng = np.random.default_rng(7)
    mu0, sig0 = np.array([-1.4, -0.6]), np.array([0.9, 1.1])
    mu1, sig1 = np.array([1.6, 1.0]), np.array([1.0, 0.8])
    prior0, prior1 = 0.5, 0.5

    n = 220
    X0 = rng.normal(mu0, sig0, (n, 2))
    X1 = rng.normal(mu1, sig1, (n, 2))

    xs = np.linspace(-5, 5, 400)
    ys = np.linspace(-4.5, 4.5, 400)
    XX, YY = np.meshgrid(xs, ys)
    p0 = prior0 * gaussian_2d(XX, YY, mu0, sig0)
    p1 = prior1 * gaussian_2d(XX, YY, mu1, sig1)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))

    # Class-conditional contours
    cs0 = ax.contour(XX, YY, p0, levels=5, colors=[C_BLUE], linewidths=1.4,
                     alpha=0.85)
    cs1 = ax.contour(XX, YY, p1, levels=5, colors=[C_AMBER], linewidths=1.4,
                     alpha=0.85)

    # Filled posterior region (which class wins)
    region = (p1 > p0).astype(float)
    cmap_bg = LinearSegmentedColormap.from_list(
        "bg", ["#dbeafe", "#fef3c7"]
    )
    ax.imshow(region, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower",
              cmap=cmap_bg, alpha=0.35, aspect="auto", zorder=0)

    # Decision boundary
    ax.contour(XX, YY, p1 - p0, levels=[0.0], colors=[C_DARK],
               linewidths=2.4)

    # Sample points
    ax.scatter(X0[:, 0], X0[:, 1], s=18, c=C_BLUE, alpha=0.55,
               edgecolor="white", linewidth=0.5, label=r"class $c_0$")
    ax.scatter(X1[:, 0], X1[:, 1], s=18, c=C_AMBER, alpha=0.55,
               edgecolor="white", linewidth=0.5, label=r"class $c_1$")

    # Class means
    ax.scatter(*mu0, s=170, marker="*", c=C_BLUE, edgecolor="white",
               linewidth=1.5, zorder=5)
    ax.scatter(*mu1, s=170, marker="*", c=C_AMBER, edgecolor="white",
               linewidth=1.5, zorder=5)
    ax.annotate(r"$\mu_0$", xy=mu0, xytext=(mu0[0] - 0.25, mu0[1] - 0.7),
                fontsize=11, color=C_DARK)
    ax.annotate(r"$\mu_1$", xy=mu1, xytext=(mu1[0] + 0.25, mu1[1] + 0.4),
                fontsize=11, color=C_DARK)

    # Boundary label
    ax.text(0.05, 3.6, r"decision boundary: $P(c_0|\mathbf{x}) = P(c_1|\mathbf{x})$",
            fontsize=10.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=C_GRAY, alpha=0.92))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel(r"$x^{(1)}$", fontsize=11)
    ax.set_ylabel(r"$x^{(2)}$", fontsize=11)
    ax.set_title("Gaussian Naive Bayes: class-conditional densities and decision boundary",
                 fontsize=12.5, pad=10)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    _save(fig, "fig1_class_conditional")


# ---------------------------------------------------------------------------
# Figure 2: Posterior probability surface
# ---------------------------------------------------------------------------
def fig2_posterior_surface() -> None:
    mu0, sig0 = np.array([-1.4, -0.6]), np.array([0.9, 1.1])
    mu1, sig1 = np.array([1.6, 1.0]), np.array([1.0, 0.8])
    prior0, prior1 = 0.5, 0.5

    xs = np.linspace(-5, 5, 360)
    ys = np.linspace(-4.5, 4.5, 360)
    XX, YY = np.meshgrid(xs, ys)
    p0 = prior0 * gaussian_2d(XX, YY, mu0, sig0)
    p1 = prior1 * gaussian_2d(XX, YY, mu1, sig1)
    posterior1 = p1 / (p0 + p1 + 1e-30)

    fig = plt.figure(figsize=(13, 5.2))

    # --- Left: heatmap of posterior P(c1|x) ---
    ax1 = fig.add_subplot(1, 2, 1)
    cmap = LinearSegmentedColormap.from_list(
        "post", [C_BLUE, "#e0e7ff", "white", "#fef3c7", C_AMBER]
    )
    im = ax1.imshow(posterior1, extent=(xs[0], xs[-1], ys[0], ys[-1]),
                    origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="auto")
    cs = ax1.contour(XX, YY, posterior1,
                     levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                     colors=[C_DARK], linewidths=1.0, alpha=0.7)
    ax1.clabel(cs, inline=True, fontsize=8.5, fmt="%.1f")
    # Highlight decision boundary
    ax1.contour(XX, YY, posterior1, levels=[0.5], colors=[C_DARK], linewidths=2.4)

    cbar = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03)
    cbar.set_label(r"$P(c_1 \mid \mathbf{x})$", fontsize=10.5)
    ax1.set_xlabel(r"$x^{(1)}$", fontsize=11)
    ax1.set_ylabel(r"$x^{(2)}$", fontsize=11)
    ax1.set_title(r"Posterior heatmap: $P(c_1 \mid \mathbf{x})$",
                  fontsize=12, pad=8)

    # --- Right: 1D slice along y=0 showing soft transition ---
    ax2 = fig.add_subplot(1, 2, 2)
    x_line = np.linspace(-5, 5, 500)
    p0_line = prior0 * gaussian_2d(x_line, 0.0, mu0, sig0)
    p1_line = prior1 * gaussian_2d(x_line, 0.0, mu1, sig1)
    post_line = p1_line / (p0_line + p1_line + 1e-30)

    # Likelihoods (rescaled for plotting)
    scale = 1.0 / max(p0_line.max(), p1_line.max())
    ax2.fill_between(x_line, 0, p0_line * scale, color=C_BLUE, alpha=0.18,
                     label=r"likelihood $P(\mathbf{x}|c_0)$")
    ax2.fill_between(x_line, 0, p1_line * scale, color=C_AMBER, alpha=0.18,
                     label=r"likelihood $P(\mathbf{x}|c_1)$")
    ax2.plot(x_line, p0_line * scale, color=C_BLUE, lw=1.6)
    ax2.plot(x_line, p1_line * scale, color=C_AMBER, lw=1.6)

    # Posterior curve
    ax2.plot(x_line, post_line, color=C_PURPLE, lw=2.8,
             label=r"posterior $P(c_1|\mathbf{x})$")
    ax2.axhline(0.5, color=C_GRAY, lw=0.9, ls="--")

    # Decision boundary point
    idx = np.argmin(np.abs(post_line - 0.5))
    ax2.scatter([x_line[idx]], [0.5], s=80, color=C_DARK, zorder=5,
                edgecolor="white", linewidth=1.5)
    ax2.annotate(f"boundary  $x^{{(1)}}\\approx{x_line[idx]:.2f}$",
                 xy=(x_line[idx], 0.5), xytext=(x_line[idx] + 0.4, 0.78),
                 fontsize=10, color=C_DARK,
                 arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"$x^{(1)}$  (slice at $x^{(2)}=0$)", fontsize=11)
    ax2.set_ylabel("probability", fontsize=11)
    ax2.set_title("How likelihood and prior combine into the posterior",
                  fontsize=12, pad=8)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    fig.tight_layout()
    _save(fig, "fig2_posterior")


# ---------------------------------------------------------------------------
# Figure 3: Independence assumption visualisation
# ---------------------------------------------------------------------------
def fig3_independence() -> None:
    mu = np.array([0.0, 0.0])
    cov_true = np.array([[1.0, 0.85], [0.85, 1.0]])  # correlated
    sig_naive = np.sqrt(np.diag(cov_true))           # same marginals

    xs = np.linspace(-3.5, 3.5, 300)
    ys = np.linspace(-3.5, 3.5, 300)
    XX, YY = np.meshgrid(xs, ys)
    p_true = gaussian_2d_full(XX, YY, mu, cov_true)
    p_naive = gaussian_2d(XX, YY, mu, sig_naive)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    for ax, P, title, color in zip(
        axes,
        [p_true, p_naive],
        [r"True $P(x_1,x_2 \mid c)$  (correlated, $\rho=0.85$)",
         r"Naive Bayes $\prod_j P(x_j \mid c)$  (forced independence)"],
        [C_PURPLE, C_GREEN],
    ):
        cs = ax.contourf(XX, YY, P, levels=12,
                         cmap=LinearSegmentedColormap.from_list(
                             "c", ["white", color]))
        ax.contour(XX, YY, P, levels=8, colors=[color], linewidths=0.8,
                   alpha=0.7)

        # Marginal projections (same in both panels by construction)
        # Top marginal of x1
        marg_x = np.exp(-0.5 * (xs / sig_naive[0]) ** 2) / (np.sqrt(2 * np.pi) * sig_naive[0])
        ax.plot(xs, 3.5 - marg_x * 1.7, color=C_DARK, lw=1.4)
        ax.fill_between(xs, 3.5, 3.5 - marg_x * 1.7, color=C_DARK, alpha=0.08)
        # Right marginal of x2
        marg_y = np.exp(-0.5 * (ys / sig_naive[1]) ** 2) / (np.sqrt(2 * np.pi) * sig_naive[1])
        ax.plot(3.5 - marg_y * 1.7, ys, color=C_DARK, lw=1.4)
        ax.fill_betweenx(ys, 3.5, 3.5 - marg_y * 1.7, color=C_DARK, alpha=0.08)

        ax.text(2.05, 3.25, r"$P(x_1)$", fontsize=10, color=C_DARK)
        ax.text(2.55, 2.0, r"$P(x_2)$", fontsize=10, color=C_DARK)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel(r"$x_1$", fontsize=11)
        ax.set_ylabel(r"$x_2$", fontsize=11)
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.set_aspect("equal")

    fig.suptitle("Same marginals, different joint: the cost of the naive assumption",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig3_independence")


# ---------------------------------------------------------------------------
# Figure 4: Bag-of-words for text classification
# ---------------------------------------------------------------------------
def fig4_bag_of_words() -> None:
    docs = [
        ("d1", "spam", "free money win free"),
        ("d2", "spam", "win cash free now"),
        ("d3", "spam", "money money free offer"),
        ("d4", "ham",  "meeting schedule project today"),
        ("d5", "ham",  "project meeting tomorrow"),
        ("d6", "ham",  "schedule meeting today"),
    ]
    vocab = ["free", "money", "win", "cash", "offer",
             "meeting", "schedule", "project", "today", "tomorrow", "now"]

    # Build TF matrix
    tf = np.zeros((len(docs), len(vocab)), dtype=int)
    for i, (_, _, text) in enumerate(docs):
        for w in text.split():
            if w in vocab:
                tf[i, vocab.index(w)] += 1

    # Per-class word probability with Laplace smoothing
    spam_idx = [i for i, d in enumerate(docs) if d[1] == "spam"]
    ham_idx = [i for i, d in enumerate(docs) if d[1] == "ham"]
    alpha = 1.0
    counts_spam = tf[spam_idx].sum(axis=0)
    counts_ham = tf[ham_idx].sum(axis=0)
    p_spam = (counts_spam + alpha) / (counts_spam.sum() + alpha * len(vocab))
    p_ham = (counts_ham + alpha) / (counts_ham.sum() + alpha * len(vocab))

    fig = plt.figure(figsize=(13.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.28)

    # --- Left: TF matrix as a heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    cmap_tf = LinearSegmentedColormap.from_list("tf", ["white", C_BLUE])
    im = ax1.imshow(tf, cmap=cmap_tf, aspect="auto", vmin=0, vmax=tf.max())
    for i in range(tf.shape[0]):
        for j in range(tf.shape[1]):
            v = tf[i, j]
            ax1.text(j, i, str(v) if v > 0 else "·",
                     ha="center", va="center", fontsize=10,
                     color="white" if v >= 2 else C_DARK)
    ax1.set_xticks(range(len(vocab)))
    ax1.set_xticklabels(vocab, rotation=40, ha="right", fontsize=9.5)
    ax1.set_yticks(range(len(docs)))
    ax1.set_yticklabels([f"{d[0]}  ({d[1]})" for d in docs], fontsize=9.5)
    # Color y-tick labels by class
    for tick, d in zip(ax1.get_yticklabels(), docs):
        tick.set_color(C_AMBER if d[1] == "spam" else C_GREEN)
    ax1.set_title("Term-frequency matrix  (rows = documents, cols = vocabulary)",
                  fontsize=11.5, pad=8)
    cb = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
    cb.set_label("count", fontsize=9.5)

    # --- Right: per-class word probabilities ---
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(vocab))
    h = 0.4
    ax2.barh(y - h / 2, p_spam, height=h, color=C_AMBER, alpha=0.85,
             edgecolor="white", label=r"$P(w \mid \mathrm{spam})$")
    ax2.barh(y + h / 2, p_ham, height=h, color=C_GREEN, alpha=0.85,
             edgecolor="white", label=r"$P(w \mid \mathrm{ham})$")
    ax2.set_yticks(y)
    ax2.set_yticklabels(vocab, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("probability  (Laplace smoothed, $\\alpha=1$)", fontsize=10.5)
    ax2.set_title("Estimated $P(w \\mid c)$ per class", fontsize=11.5, pad=8)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
    ax2.set_xlim(0, max(p_spam.max(), p_ham.max()) * 1.15)

    fig.suptitle("Bag-of-words: from raw documents to per-class word distributions",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_bag_of_words")


# ---------------------------------------------------------------------------
# Figure 5: Three Naive Bayes variants
# ---------------------------------------------------------------------------
def fig5_three_variants() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    # --- Multinomial: bar chart of word probabilities (counts) ---
    ax = axes[0]
    words = ["free", "money", "win", "click", "meeting"]
    p_spam = np.array([0.32, 0.28, 0.18, 0.15, 0.07])
    p_ham = np.array([0.06, 0.05, 0.04, 0.05, 0.80])
    x = np.arange(len(words))
    w = 0.4
    ax.bar(x - w / 2, p_spam, w, color=C_AMBER, edgecolor="white",
           label="spam")
    ax.bar(x + w / 2, p_ham, w, color=C_GREEN, edgecolor="white",
           label="ham")
    ax.set_xticks(x)
    ax.set_xticklabels(words, fontsize=10)
    ax.set_ylabel(r"$\theta_{jk} = P(w_j \mid c_k)$", fontsize=10.5)
    ax.set_title("Multinomial NB\n(word-frequency features)",
                 fontsize=11.5, pad=8, color=C_BLUE)
    ax.legend(framealpha=0.95, fontsize=9.5)
    ax.set_ylim(0, 1)

    # --- Gaussian: per-class density curves over a continuous feature ---
    ax = axes[1]
    xs = np.linspace(-3, 8, 400)
    pdf0 = np.exp(-0.5 * ((xs - 1.0) / 1.0) ** 2) / np.sqrt(2 * np.pi)
    pdf1 = np.exp(-0.5 * ((xs - 4.5) / 1.2) ** 2) / (1.2 * np.sqrt(2 * np.pi))
    ax.plot(xs, pdf0, color=C_BLUE, lw=2.4, label=r"$\mathcal{N}(\mu_0, \sigma_0^2)$")
    ax.fill_between(xs, 0, pdf0, color=C_BLUE, alpha=0.18)
    ax.plot(xs, pdf1, color=C_AMBER, lw=2.4, label=r"$\mathcal{N}(\mu_1, \sigma_1^2)$")
    ax.fill_between(xs, 0, pdf1, color=C_AMBER, alpha=0.18)
    ax.axvline(1.0, color=C_BLUE, ls=":", lw=1)
    ax.axvline(4.5, color=C_AMBER, ls=":", lw=1)
    ax.text(1.0, 0.43, r"$\mu_0$", color=C_BLUE, ha="center", fontsize=10)
    ax.text(4.5, 0.36, r"$\mu_1$", color=C_AMBER, ha="center", fontsize=10)
    ax.set_xlabel(r"continuous feature $x^{(j)}$", fontsize=10.5)
    ax.set_ylabel(r"$P(x^{(j)} \mid c_k)$", fontsize=10.5)
    ax.set_title("Gaussian NB\n(continuous features)",
                 fontsize=11.5, pad=8, color=C_PURPLE)
    ax.legend(framealpha=0.95, fontsize=9.5)
    ax.set_ylim(0, 0.5)

    # --- Bernoulli: presence/absence probability bars ---
    ax = axes[2]
    feats = ["free", "money", "win", "click", "meeting"]
    p_present_spam = np.array([0.85, 0.78, 0.62, 0.55, 0.10])
    p_present_ham = np.array([0.08, 0.07, 0.05, 0.06, 0.72])
    x = np.arange(len(feats))
    ax.bar(x - w / 2, p_present_spam, w, color=C_AMBER, edgecolor="white",
           label=r"$P(x_j=1 \mid \mathrm{spam})$")
    ax.bar(x + w / 2, p_present_ham, w, color=C_GREEN, edgecolor="white",
           label=r"$P(x_j=1 \mid \mathrm{ham})$")
    # Overlay absence probability lightly
    ax.bar(x - w / 2, 1 - p_present_spam, w, bottom=p_present_spam,
           color=C_AMBER, alpha=0.18, edgecolor="white")
    ax.bar(x + w / 2, 1 - p_present_ham, w, bottom=p_present_ham,
           color=C_GREEN, alpha=0.18, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(feats, fontsize=10)
    ax.set_ylabel(r"$P(x_j \mid c_k)$", fontsize=10.5)
    ax.set_title("Bernoulli NB\n(presence / absence features)",
                 fontsize=11.5, pad=8, color=C_GREEN)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.95, fontsize=9, loc="upper right")

    fig.suptitle("Three Naive Bayes variants — same Bayes rule, different likelihood",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    _save(fig, "fig5_three_variants")


# ---------------------------------------------------------------------------
# Figure 6: Laplace smoothing effect
# ---------------------------------------------------------------------------
def fig6_laplace_smoothing() -> None:
    # Vocabulary: 8 words; class has total 50 word occurrences.
    # Some words have count=0 -> zero-probability problem.
    vocab = ["free", "money", "win", "deal", "offer",
             "rare", "unseen", "novel"]
    counts = np.array([18, 14, 9, 6, 3, 0, 0, 0])
    N_c = counts.sum()  # 50
    V = len(vocab)
    alphas = [0.0, 0.5, 1.0, 5.0]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # --- Left: probability bars across alphas ---
    ax = axes[0]
    x = np.arange(V)
    width = 0.2
    cmap_alpha = [C_RED, C_AMBER, C_BLUE, C_PURPLE]
    for k, a in enumerate(alphas):
        p = (counts + a) / (N_c + a * V)
        offset = (k - (len(alphas) - 1) / 2) * width
        ax.bar(x + offset, p, width,
               color=cmap_alpha[k], edgecolor="white",
               label=fr"$\alpha={a}$" + (" (MLE)" if a == 0 else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(vocab, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylabel(r"$\hat P(w \mid c)$", fontsize=10.5)
    ax.set_title("How $\\alpha$ trades data fit against uniform prior",
                 fontsize=11.5, pad=8)
    ax.legend(framealpha=0.95, fontsize=9.5)

    # Annotate the zero-probability words
    for j in [5, 6, 7]:
        ax.annotate("MLE = 0",
                    xy=(j - 1.5 * width, 0.005), xytext=(j - 1.7 * width, 0.07),
                    fontsize=8.5, color=C_RED,
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8))

    # --- Right: smoothed probability of a SEEN word vs. UNSEEN word as alpha varies ---
    ax = axes[1]
    a_grid = np.linspace(0, 8, 400)
    p_seen = (18 + a_grid) / (N_c + a_grid * V)         # "free", count=18
    p_zero = (0 + a_grid) / (N_c + a_grid * V)          # "rare", count=0
    p_uniform = np.ones_like(a_grid) / V

    ax.plot(a_grid, p_seen, color=C_BLUE, lw=2.4,
            label=r"frequent word ($\mathrm{count}=18$)")
    ax.plot(a_grid, p_zero, color=C_RED, lw=2.4,
            label=r"unseen word ($\mathrm{count}=0$)")
    ax.plot(a_grid, p_uniform, color=C_GRAY, lw=1.5, ls="--",
            label=r"uniform $1/V$")
    ax.fill_between(a_grid, p_zero, p_uniform, where=p_zero < p_uniform,
                    color=C_GREEN, alpha=0.10,
                    label="prior pulls estimates upward")

    # Mark alpha = 1
    ax.axvline(1.0, color=C_DARK, ls=":", lw=1)
    ax.text(1.05, 0.34, r"$\alpha=1$ (Laplace)", fontsize=9.5, color=C_DARK)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel(r"smoothing parameter $\alpha$", fontsize=10.5)
    ax.set_ylabel(r"$\hat P(w \mid c)$", fontsize=10.5)
    ax.set_title("Estimates shrink toward $1/V$ as $\\alpha$ grows",
                 fontsize=11.5, pad=8)
    ax.legend(framealpha=0.95, fontsize=9.0, loc="upper right")

    fig.suptitle("Laplace smoothing: fixing the zero-probability problem",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_laplace_smoothing")


# ---------------------------------------------------------------------------
# Figure 7: Worked spam classifier with log-odds contributions
# ---------------------------------------------------------------------------
def fig7_spam_decision() -> None:
    # Email features: presence (1) / absence (0) of each word
    feats = ["free", "money", "win", "click", "meeting", "schedule"]
    x = np.array([1, 1, 1, 0, 0, 0])  # the test email

    p_present_spam = np.array([0.85, 0.78, 0.62, 0.55, 0.10, 0.08])
    p_present_ham = np.array([0.08, 0.07, 0.05, 0.06, 0.72, 0.68])

    # Bernoulli log-likelihood per class
    eps = 1e-9

    def loglik(p_present, x):
        return x * np.log(p_present + eps) + (1 - x) * np.log(1 - p_present + eps)

    log_spam = loglik(p_present_spam, x)
    log_ham = loglik(p_present_ham, x)
    contrib = log_spam - log_ham  # per-feature log-odds in favour of spam

    # Priors
    log_prior_ratio = np.log(0.4 / 0.6)  # P(spam)/P(ham)
    total_log_odds = contrib.sum() + log_prior_ratio

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- Left: per-feature log-odds bar chart ---
    ax = axes[0]
    colors = [C_AMBER if c > 0 else C_GREEN for c in contrib]
    bars = ax.barh(range(len(feats)), contrib, color=colors,
                   edgecolor="white", height=0.65)
    ax.set_yticks(range(len(feats)))
    labels = [f"{f}  (x={xi})" for f, xi in zip(feats, x)]
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color=C_DARK, lw=1)

    # Annotate values
    for i, (b, c) in enumerate(zip(bars, contrib)):
        ax.text(c + (0.05 if c >= 0 else -0.05), i,
                f"{c:+.2f}", va="center",
                ha="left" if c >= 0 else "right",
                fontsize=9.5, color=C_DARK)

    ax.set_xlabel(r"per-feature contribution to $\ln\frac{P(\mathrm{spam}|\mathbf{x})}{P(\mathrm{ham}|\mathbf{x})}$",
                  fontsize=10.5)
    ax.set_title("Bernoulli NB on a test email: each word's evidence",
                 fontsize=12, pad=8)

    # Legend swatches
    ax.scatter([], [], s=80, c=C_AMBER, label="evidence for spam")
    ax.scatter([], [], s=80, c=C_GREEN, label="evidence for ham")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)

    # --- Right: waterfall building total log-odds ---
    ax = axes[1]
    steps = ["log prior\nratio"] + feats + ["total\nlog-odds"]
    values = np.concatenate(([log_prior_ratio], contrib, [0.0]))
    cumulative = np.zeros(len(values))
    cumulative[0] = values[0]
    for i in range(1, len(values) - 1):
        cumulative[i] = cumulative[i - 1] + values[i]
    cumulative[-1] = cumulative[-2]

    # Draw bars
    for i in range(len(values)):
        if i == 0:
            base = 0
            top = values[i]
            color = C_BLUE
        elif i == len(values) - 1:
            base = 0
            top = cumulative[-1]
            color = C_PURPLE
        else:
            base = cumulative[i - 1]
            top = cumulative[i]
            color = C_AMBER if values[i] > 0 else C_GREEN
        ax.bar(i, top - base, bottom=base, color=color,
               edgecolor="white", width=0.7)
        # Connector line
        if 0 < i < len(values) - 1:
            ax.plot([i - 1 + 0.35, i - 0.35],
                    [cumulative[i - 1], cumulative[i - 1]],
                    color=C_GRAY, lw=0.9, ls=":")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=9.0, rotation=20, ha="right")
    ax.axhline(0, color=C_DARK, lw=1)

    # Final decision label
    decision = "SPAM" if total_log_odds > 0 else "HAM"
    ax.text(len(values) - 1, cumulative[-1] + 0.4,
            f"log-odds = {total_log_odds:+.2f}\nprediction: {decision}",
            ha="center", fontsize=10.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_PURPLE, alpha=0.95))

    ax.set_ylabel("cumulative log-odds (spam vs. ham)", fontsize=10.5)
    ax.set_title("Stacking evidence: prior + each word $\\to$ final decision",
                 fontsize=12, pad=8)

    fig.suptitle("A worked Naive Bayes spam classification",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_spam_decision")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Naive Bayes figures ...")
    fig1_class_conditional()
    fig2_posterior_surface()
    fig3_independence()
    fig4_bag_of_words()
    fig5_three_variants()
    fig6_laplace_smoothing()
    fig7_spam_decision()
    print("Done.")


if __name__ == "__main__":
    main()
