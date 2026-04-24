"""
Figure generation script for ML Math Derivations Chapter 03:
"Probability Theory and Statistical Inference".

Generates 7 figures used in BOTH the EN and ZH versions of the article.
Each figure visualizes one core idea cleanly and unambiguously.

Figures:
    fig1_distributions          A 2x3 grid of canonical distributions used
                                throughout ML: Normal, Bernoulli, Poisson,
                                Beta, Gamma, and Dirichlet (a single
                                3-simplex contour).
    fig2_bayes_update           Prior -> Likelihood -> Posterior, shown as
                                three coupled panels for a Beta-Bernoulli
                                model with explicit pseudo-count
                                interpretation.
    fig3_mle_vs_map             Three-panel side-by-side comparison of MLE,
                                MAP, and full Bayesian posterior on a coin
                                with little data (n=5) -- the regime in
                                which the three answers diverge most.
    fig4_clt                    Central Limit Theorem: standardized sample
                                means from an exponential distribution
                                converge to N(0,1) as n grows from 1 to 50.
    fig5_confidence_intervals   50 simulated 95% CIs for a normal mean,
                                colored by whether they cover the true
                                parameter; visualizes the frequentist
                                interpretation.
    fig6_hypothesis_errors      Two overlaid normals (H0 and H1) with the
                                Type I (alpha) and Type II (beta) regions
                                shaded; shows the alpha/beta tradeoff.
    fig7_information_theory     Three-panel info theory: entropy of a
                                Bernoulli as a function of p, KL divergence
                                between two Gaussians, and mutual
                                information as the area shared by two
                                distributions.

Usage:
    python3 scripts/figures/ml-math-derivations/03-probability.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages. Parent folders are created
    if missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import stats
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_RED = "#ef4444"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150
RNG = np.random.default_rng(7)

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "03-Probability-Theory-and-Statistical-Inference"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "03-概率论与统计推断"
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


def _style_axes(ax: plt.Axes, *, title: str | None = None,
                xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Consistent typography across panels."""
    if title:
        ax.set_title(title, fontsize=11.5, color=C_DARK, pad=8,
                     weight="semibold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=C_DARK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=C_DARK)
    ax.tick_params(labelsize=8.5, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color=C_LIGHT, linewidth=0.8, zorder=0)


# ---------------------------------------------------------------------------
# Figure 1: Common distributions in ML  (2 x 3 grid)
# ---------------------------------------------------------------------------
def fig1_distributions() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))

    # (0,0) Normal: three variances
    ax = axes[0, 0]
    x = np.linspace(-5, 5, 600)
    for sigma, color, label in [
        (0.6, C_PURPLE, r"$\sigma=0.6$"),
        (1.0, C_BLUE, r"$\sigma=1.0$"),
        (1.8, C_AMBER, r"$\sigma=1.8$"),
    ]:
        ax.plot(x, stats.norm.pdf(x, 0, sigma), color=color, lw=2.2,
                label=label)
    ax.fill_between(x, stats.norm.pdf(x, 0, 1.0), color=C_BLUE, alpha=0.10)
    _style_axes(ax, title=r"Normal  $\mathcal{N}(0,\,\sigma^2)$",
                xlabel="x", ylabel="density")
    ax.legend(frameon=False, fontsize=8.5)

    # (0,1) Bernoulli: stem plot for several p
    ax = axes[0, 1]
    width = 0.22
    for i, (p, color) in enumerate([
        (0.2, C_PURPLE), (0.5, C_BLUE), (0.8, C_AMBER),
    ]):
        ax.bar([0 + (i - 1) * width, 1 + (i - 1) * width], [1 - p, p],
               width=width, color=color, alpha=0.9, label=f"p={p}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["X=0", "X=1"])
    _style_axes(ax, title=r"Bernoulli  $\mathrm{Bern}(p)$",
                ylabel="P(X)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylim(0, 1.0)

    # (0,2) Poisson: bar plot for several lambdas
    ax = axes[0, 2]
    ks = np.arange(0, 18)
    for lam, color, marker in [
        (1, C_PURPLE, "o"), (4, C_BLUE, "s"), (10, C_AMBER, "D"),
    ]:
        ax.plot(ks, stats.poisson.pmf(ks, lam), color=color, lw=1.4,
                marker=marker, markersize=4.5, label=fr"$\lambda={lam}$")
    _style_axes(ax, title=r"Poisson  $\mathrm{Poi}(\lambda)$",
                xlabel="k", ylabel="P(X=k)")
    ax.legend(frameon=False, fontsize=8.5)

    # (1,0) Beta: family of shapes
    ax = axes[1, 0]
    x = np.linspace(0, 1, 400)
    for (a, b), color, label in [
        ((0.5, 0.5), C_AMBER, r"$\alpha=\beta=0.5$"),
        ((2, 2), C_BLUE, r"$\alpha=\beta=2$"),
        ((5, 2), C_PURPLE, r"$\alpha=5,\beta=2$"),
        ((2, 5), C_GREEN, r"$\alpha=2,\beta=5$"),
    ]:
        ax.plot(x, stats.beta.pdf(x, a, b), color=color, lw=2.0,
                label=label)
    _style_axes(ax, title=r"Beta  $\mathrm{Beta}(\alpha,\beta)$",
                xlabel="x", ylabel="density")
    ax.legend(frameon=False, fontsize=8.5)

    # (1,1) Gamma: family
    ax = axes[1, 1]
    x = np.linspace(0, 14, 500)
    for (k, theta), color, label in [
        ((1, 2), C_PURPLE, r"$k=1,\theta=2$"),
        ((2, 2), C_BLUE, r"$k=2,\theta=2$"),
        ((5, 1), C_AMBER, r"$k=5,\theta=1$"),
        ((9, 0.5), C_GREEN, r"$k=9,\theta=0.5$"),
    ]:
        ax.plot(x, stats.gamma.pdf(x, k, scale=theta), color=color, lw=2.0,
                label=label)
    _style_axes(ax, title=r"Gamma  $\mathrm{Gamma}(k,\theta)$",
                xlabel="x", ylabel="density")
    ax.legend(frameon=False, fontsize=8.5)

    # (1,2) Dirichlet on the 3-simplex (contour of one distribution)
    ax = axes[1, 2]
    n = 220
    s = np.linspace(0.001, 0.999, n)
    X, Y = np.meshgrid(s, s)
    Z = 1.0 - X - Y
    mask = Z > 0
    alpha = np.array([3.0, 5.0, 2.0])

    def dirichlet_log_pdf(x1, x2, x3, a):
        coef = gammaln(a.sum()) - gammaln(a).sum()
        return coef + (a[0] - 1) * np.log(x1) + (a[1] - 1) * np.log(x2) + \
            (a[2] - 1) * np.log(x3)

    log_pdf = np.full_like(X, np.nan)
    log_pdf[mask] = dirichlet_log_pdf(X[mask], Y[mask], Z[mask], alpha)
    pdf = np.exp(log_pdf - np.nanmax(log_pdf))

    # Map (x1, x2) onto an equilateral triangle (barycentric -> Cartesian)
    # Vertices of the triangle:
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    Xt = X * v1[0] + Y * v2[0] + Z * v0[0]
    Yt = X * v1[1] + Y * v2[1] + Z * v0[1]

    cs = ax.contourf(Xt, Yt, pdf, levels=18, cmap="Purples")
    # Triangle edges
    tri = np.vstack([v0, v1, v2, v0])
    ax.plot(tri[:, 0], tri[:, 1], color=C_DARK, lw=1.4)
    ax.text(*v0 + np.array([-0.04, -0.05]), r"$x_3$", fontsize=10,
            color=C_DARK, ha="right")
    ax.text(*v1 + np.array([0.04, -0.05]), r"$x_1$", fontsize=10,
            color=C_DARK, ha="left")
    ax.text(*v2 + np.array([0.0, 0.04]), r"$x_2$", fontsize=10,
            color=C_DARK, ha="center")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"Dirichlet on the simplex  $\mathrm{Dir}(3,5,2)$",
                 fontsize=11.5, color=C_DARK, pad=8, weight="semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    fig.suptitle("Common probability distributions in machine learning",
                 fontsize=13.5, weight="semibold", color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig1_distributions")


# ---------------------------------------------------------------------------
# Figure 2: Bayesian update -- prior -> likelihood -> posterior
# ---------------------------------------------------------------------------
def fig2_bayes_update() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)

    p = np.linspace(0, 1, 600)
    a0, b0 = 4, 2          # Prior: a slightly head-leaning belief
    n, k = 12, 4           # Data: 4 heads in 12 flips
    a_post = a0 + k
    b_post = b0 + n - k

    # Panel 1 -- Prior
    ax = axes[0]
    prior = stats.beta.pdf(p, a0, b0)
    ax.plot(p, prior, color=C_PURPLE, lw=2.4, label=fr"$\mathrm{{Beta}}({a0},{b0})$")
    ax.fill_between(p, prior, color=C_PURPLE, alpha=0.15)
    ax.axvline(a0 / (a0 + b0), color=C_PURPLE, ls="--", lw=1.0,
               label=f"prior mean = {a0/(a0+b0):.2f}")
    _style_axes(ax, title="Prior  $P(\\theta)$", xlabel=r"$\theta$",
                ylabel="density")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1)

    # Panel 2 -- Likelihood (binomial with k heads in n)
    ax = axes[1]
    # log-likelihood up to constant
    log_lik = k * np.log(np.clip(p, 1e-12, 1)) + \
        (n - k) * np.log(np.clip(1 - p, 1e-12, 1))
    lik = np.exp(log_lik - log_lik.max())
    ax.plot(p, lik, color=C_BLUE, lw=2.4,
            label=f"data: {k} heads / {n} flips")
    ax.fill_between(p, lik, color=C_BLUE, alpha=0.15)
    ax.axvline(k / n, color=C_BLUE, ls="--", lw=1.0,
               label=f"MLE = {k/n:.2f}")
    _style_axes(ax, title=r"Likelihood  $P(D\mid\theta)$",
                xlabel=r"$\theta$", ylabel="(rescaled)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1)

    # Panel 3 -- Posterior, with prior overlaid for contrast
    ax = axes[2]
    posterior = stats.beta.pdf(p, a_post, b_post)
    ax.plot(p, prior, color=C_PURPLE, lw=1.4, ls="--",
            label="prior", alpha=0.8)
    ax.plot(p, posterior, color=C_GREEN, lw=2.6,
            label=fr"$\mathrm{{Beta}}({a_post},{b_post})$")
    ax.fill_between(p, posterior, color=C_GREEN, alpha=0.18)
    post_mean = a_post / (a_post + b_post)
    ax.axvline(post_mean, color=C_GREEN, ls="--", lw=1.0,
               label=f"posterior mean = {post_mean:.2f}")
    _style_axes(ax, title=r"Posterior  $P(\theta\mid D)$",
                xlabel=r"$\theta$", ylabel="density")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1)

    fig.suptitle("Bayesian update for a Beta–Bernoulli model",
                 fontsize=13.5, weight="semibold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_bayes_update")


# ---------------------------------------------------------------------------
# Figure 3: MLE vs MAP vs Bayesian (small-sample regime)
# ---------------------------------------------------------------------------
def fig3_mle_vs_map() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)

    # Same data for all three panels: only 5 flips, 4 heads
    n, k = 5, 4
    p = np.linspace(0, 1, 600)

    # Panel 1 -- MLE: just the likelihood, point at argmax
    ax = axes[0]
    log_lik = k * np.log(np.clip(p, 1e-12, 1)) + \
        (n - k) * np.log(np.clip(1 - p, 1e-12, 1))
    lik = np.exp(log_lik - log_lik.max())
    p_mle = k / n
    ax.plot(p, lik, color=C_BLUE, lw=2.4, label="likelihood")
    ax.fill_between(p, lik, color=C_BLUE, alpha=0.15)
    ax.axvline(p_mle, color=C_BLUE, ls="--", lw=1.4,
               label=fr"$\hat\theta_{{\rm MLE}}={p_mle:.2f}$")
    _style_axes(ax, title="MLE  —  argmax of likelihood",
                xlabel=r"$\theta$", ylabel="rescaled density")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # Panel 2 -- MAP with Beta(2,2) prior: argmax of posterior
    ax = axes[1]
    a_pri, b_pri = 2, 2
    a_post, b_post = a_pri + k, b_pri + n - k
    posterior = stats.beta.pdf(p, a_post, b_post)
    posterior_n = posterior / posterior.max()
    prior = stats.beta.pdf(p, a_pri, b_pri)
    prior_n = prior / prior.max()
    p_map = (a_post - 1) / (a_post + b_post - 2)
    ax.plot(p, prior_n, color=C_PURPLE, ls="--", lw=1.4,
            label=fr"prior $\mathrm{{Beta}}(2,2)$")
    ax.plot(p, posterior_n, color=C_GREEN, lw=2.4,
            label="posterior")
    ax.fill_between(p, posterior_n, color=C_GREEN, alpha=0.15)
    ax.axvline(p_map, color=C_GREEN, ls="--", lw=1.4,
               label=fr"$\hat\theta_{{\rm MAP}}={p_map:.2f}$")
    _style_axes(ax, title="MAP  —  argmax of posterior",
                xlabel=r"$\theta$")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # Panel 3 -- Full Bayesian: posterior + 95% credible interval
    ax = axes[2]
    ax.plot(p, posterior_n, color=C_AMBER, lw=2.4, label="posterior")
    ax.fill_between(p, posterior_n, color=C_AMBER, alpha=0.15)
    ci_lo, ci_hi = stats.beta.ppf([0.025, 0.975], a_post, b_post)
    mask = (p >= ci_lo) & (p <= ci_hi)
    ax.fill_between(p[mask], posterior_n[mask], color=C_AMBER, alpha=0.45)
    p_mean = a_post / (a_post + b_post)
    ax.axvline(p_mean, color=C_AMBER, ls="--", lw=1.4,
               label=fr"posterior mean $={p_mean:.2f}$")
    ax.text((ci_lo + ci_hi) / 2, 0.05,
            f"95% CrI\n[{ci_lo:.2f}, {ci_hi:.2f}]",
            ha="center", fontsize=9, color=C_DARK,
            bbox=dict(facecolor="white", edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.3"))
    _style_axes(ax, title="Bayesian  —  full posterior",
                xlabel=r"$\theta$")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(f"Three answers from the same data ({k} heads in {n} flips)",
                 fontsize=13.5, weight="semibold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_mle_vs_map")


# ---------------------------------------------------------------------------
# Figure 4: Central Limit Theorem
# ---------------------------------------------------------------------------
def fig4_clt() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))

    # Underlying distribution: Exponential(1)  -- highly skewed
    rate = 1.0
    mu_true = 1.0 / rate
    sigma_true = 1.0 / rate

    sample_sizes = [1, 4, 16, 50]
    n_means = 12000

    # Reference standard normal
    z = np.linspace(-4.2, 4.2, 600)
    sn = stats.norm.pdf(z, 0, 1)

    for ax, n in zip(axes, sample_sizes):
        means = RNG.exponential(scale=1.0 / rate, size=(n_means, n)).mean(axis=1)
        z_means = (means - mu_true) / (sigma_true / np.sqrt(n))
        ax.hist(z_means, bins=46, density=True, color=C_BLUE,
                edgecolor="white", alpha=0.85, range=(-4.2, 4.2))
        ax.plot(z, sn, color=C_AMBER, lw=2.2, label=r"$\mathcal{N}(0,1)$")
        _style_axes(ax, title=f"n = {n}", xlabel=r"$Z_n$",
                    ylabel="density" if n == 1 else None)
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(0, 0.55)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Central Limit Theorem  —  standardized means of "
        "Exponential(1) converge to N(0,1)",
        fontsize=13.5, weight="semibold", color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig4_clt")


# ---------------------------------------------------------------------------
# Figure 5: Confidence intervals -- frequentist coverage
# ---------------------------------------------------------------------------
def fig5_confidence_intervals() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.0))

    mu_true = 0.0
    sigma = 1.0
    n = 30
    n_experiments = 50
    z_crit = stats.norm.ppf(0.975)  # known sigma -> use z

    means = []
    los = []
    his = []
    covers = []
    for _ in range(n_experiments):
        sample = RNG.normal(mu_true, sigma, n)
        m = sample.mean()
        half = z_crit * sigma / np.sqrt(n)
        lo, hi = m - half, m + half
        means.append(m)
        los.append(lo)
        his.append(hi)
        covers.append(lo <= mu_true <= hi)

    means = np.array(means)
    los = np.array(los)
    his = np.array(his)
    covers = np.array(covers)

    for i in range(n_experiments):
        color = C_BLUE if covers[i] else C_RED
        ax.plot([los[i], his[i]], [i, i], color=color, lw=1.8, alpha=0.85)
        ax.plot(means[i], i, "o", color=color, markersize=3.5)

    ax.axvline(mu_true, color=C_DARK, ls="--", lw=1.4,
               label=fr"true $\mu={mu_true}$")
    n_cover = int(covers.sum())
    ax.set_title(
        f"50 simulated 95% CIs   —   "
        f"{n_cover}/{n_experiments} cover the true $\\mu$",
        fontsize=12.5, color=C_DARK, pad=10, weight="semibold",
    )
    ax.set_xlabel("value", fontsize=10, color=C_DARK)
    ax.set_ylabel("experiment", fontsize=10, color=C_DARK)
    ax.tick_params(labelsize=8.5, colors=C_DARK)
    ax.grid(True, color=C_LIGHT, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=C_BLUE, edgecolor=C_BLUE,
              label=r"covers true $\mu$"),
        Patch(facecolor=C_RED, edgecolor=C_RED,
              label=r"misses true $\mu$"),
    ]
    ax.legend(handles=legend_handles + [
        plt.Line2D([], [], color=C_DARK, ls="--", lw=1.4,
                   label=fr"true $\mu={mu_true}$")],
        frameon=False, fontsize=9.5, loc="lower right")

    fig.tight_layout()
    _save(fig, "fig5_confidence_intervals")


# ---------------------------------------------------------------------------
# Figure 6: Hypothesis testing -- Type I and Type II errors
# ---------------------------------------------------------------------------
def fig6_hypothesis_errors() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    # Two competing hypotheses about a mean -- known sigma = 1
    mu0, mu1 = 0.0, 1.6
    sigma = 1.0

    x = np.linspace(-3.5, 5.0, 800)
    f0 = stats.norm.pdf(x, mu0, sigma)
    f1 = stats.norm.pdf(x, mu1, sigma)

    # Decision threshold: reject H0 if X > c  (one-sided test, alpha = 0.05)
    alpha_target = 0.05
    c = stats.norm.ppf(1 - alpha_target, mu0, sigma)

    ax.plot(x, f0, color=C_BLUE, lw=2.2, label=r"$H_0:\;\mu=\mu_0$")
    ax.plot(x, f1, color=C_PURPLE, lw=2.2, label=r"$H_1:\;\mu=\mu_1$")

    # Type I (alpha): under H0, reject region (x > c)
    mask_a = x >= c
    ax.fill_between(x[mask_a], f0[mask_a], color=C_RED, alpha=0.40,
                    label=r"Type I  ($\alpha$)")

    # Type II (beta): under H1, accept region (x <= c)
    mask_b = x <= c
    ax.fill_between(x[mask_b], f1[mask_b], color=C_AMBER, alpha=0.40,
                    label=r"Type II  ($\beta$)")

    ax.axvline(c, color=C_DARK, ls="--", lw=1.4)
    ax.text(c, ax.get_ylim()[1] * 0.92, f"  threshold c = {c:.2f}",
            fontsize=9.5, color=C_DARK, ha="left", va="top")

    # Annotate alpha and beta
    alpha_val = 1 - stats.norm.cdf(c, mu0, sigma)
    beta_val = stats.norm.cdf(c, mu1, sigma)
    ax.text(c + 0.55, 0.06, fr"$\alpha={alpha_val:.3f}$",
            fontsize=10, color=C_RED, weight="semibold")
    ax.text(c - 1.45, 0.06, fr"$\beta={beta_val:.3f}$",
            fontsize=10, color=C_AMBER, weight="semibold")

    _style_axes(ax,
                title=r"Type I vs Type II errors  —  $\alpha/\beta$ tradeoff",
                xlabel="test statistic", ylabel="density")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_xlim(-3.5, 5.0)

    fig.tight_layout()
    _save(fig, "fig6_hypothesis_errors")


# ---------------------------------------------------------------------------
# Figure 7: Information theory -- entropy, KL divergence, mutual information
# ---------------------------------------------------------------------------
def fig7_information_theory() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))

    # Panel 1 -- Bernoulli entropy as a function of p
    ax = axes[0]
    p = np.linspace(1e-4, 1 - 1e-4, 400)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    ax.plot(p, H, color=C_BLUE, lw=2.4)
    ax.fill_between(p, H, color=C_BLUE, alpha=0.12)
    ax.axvline(0.5, color=C_AMBER, ls="--", lw=1.2)
    ax.scatter([0.5], [1.0], color=C_AMBER, s=42, zorder=5)
    ax.text(0.51, 1.02, "max H = 1 bit at p=0.5",
            fontsize=9, color=C_DARK)
    _style_axes(ax, title="Entropy of a Bernoulli",
                xlabel=r"$p$", ylabel=r"$H(p)$  (bits)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)

    # Panel 2 -- KL divergence between two Gaussians (asymmetry)
    ax = axes[1]
    x = np.linspace(-5, 8, 800)
    P = stats.norm.pdf(x, 0, 1.0)
    Q = stats.norm.pdf(x, 2.0, 1.4)
    ax.plot(x, P, color=C_BLUE, lw=2.2, label=r"$P=\mathcal{N}(0,1)$")
    ax.plot(x, Q, color=C_PURPLE, lw=2.2, label=r"$Q=\mathcal{N}(2,1.4^2)$")
    ax.fill_between(x, np.minimum(P, Q), color=C_GREEN, alpha=0.18,
                    label=r"$\min(P,Q)$")

    # Compute KL numerically on a fine grid
    def kl(p, q, dx):
        m = (p > 1e-12) & (q > 1e-12)
        return float(np.sum(p[m] * np.log(p[m] / q[m]) * dx))
    dx = x[1] - x[0]
    kl_pq = kl(P, Q, dx)
    kl_qp = kl(Q, P, dx)
    ax.text(0.02, 0.98,
            fr"$D_{{KL}}(P\|Q)={kl_pq:.3f}$" + "\n" +
            fr"$D_{{KL}}(Q\|P)={kl_qp:.3f}$",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor="white", edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.35"))
    _style_axes(ax, title="KL divergence (asymmetric)",
                xlabel="x", ylabel="density")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # Panel 3 -- Mutual information for a bivariate Gaussian as rho varies
    ax = axes[2]
    rho = np.linspace(-0.99, 0.99, 400)
    # For bivariate Gaussian: I(X;Y) = -0.5 * log(1 - rho^2)  (nats)
    # Convert to bits:
    I = -0.5 * np.log(1 - rho ** 2) / np.log(2)
    ax.plot(rho, I, color=C_PURPLE, lw=2.4)
    ax.fill_between(rho, I, color=C_PURPLE, alpha=0.12)
    for r_mark in [-0.8, 0.0, 0.8]:
        I_mark = -0.5 * np.log(1 - r_mark ** 2 + 1e-12) / np.log(2)
        ax.scatter([r_mark], [I_mark], color=C_AMBER, s=36, zorder=5)
        ax.text(r_mark, I_mark + 0.08, fr"$\rho={r_mark}$",
                ha="center", fontsize=9, color=C_DARK)
    _style_axes(ax,
                title="Mutual information of bivariate Gaussian",
                xlabel=r"correlation  $\rho$",
                ylabel=r"$I(X;Y)$  (bits)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, max(2.5, I.max() * 1.1))

    fig.suptitle(
        "Information theory  —  entropy, divergence, and mutual information",
        fontsize=13.5, weight="semibold", color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig7_information_theory")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for chapter 03 (Probability & Inference)...")
    fig1_distributions()
    fig2_bayes_update()
    fig3_mle_vs_map()
    fig4_clt()
    fig5_confidence_intervals()
    fig6_hypothesis_errors()
    fig7_information_theory()
    print("Done.")


if __name__ == "__main__":
    main()
