"""
Figure generation script for ML Math Derivations Part 13:
EM Algorithm and Gaussian Mixture Models.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates one specific idea so the math becomes visually concrete.

Figures:
    fig1_gmm_clusters       2D mixture data overlaid with sklearn-fit GMM
                            mean markers and 1/2-sigma covariance ellipses
                            for K=3 components.
    fig2_e_step             Soft-assignment heatmap: each pixel of a 2D
                            grid coloured by its responsibilities, with
                            the matrix view of gamma_{ik} for sample
                            points so the row-stochastic structure is
                            obvious.
    fig3_m_step             Before/after one M-step update on a small
                            synthetic problem, showing how component
                            means and covariance ellipses move toward
                            the responsibility-weighted statistics.
    fig4_loglik_monotone    Log-likelihood vs EM iteration on a real
                            run -- monotonically non-decreasing -- with
                            the sklearn final value as a dashed bound.
    fig5_bic_aic            Model selection: BIC and AIC vs number of
                            components K, with the chosen K marked.
    fig6_kmeans_vs_gmm      K-means (hard, spherical) vs GMM (soft,
                            elliptical) on the same anisotropic data.
    fig7_elbo_vs_loglik     ELBO and log-likelihood across iterations,
                            with the KL gap shaded -- KL closes after
                            each E-step then re-opens during the M-step.

Usage:
    python3 scripts/figures/ml-math-derivations/13-em-gmm.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

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

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "13-EM-Algorithm-and-GMM"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "13-EM算法与GMM"
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
def cov_ellipse(ax, mean, cov, color, n_std=(1.0, 2.0), **kw):
    """Draw confidence ellipses for a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for s in n_std:
        w, h = 2.0 * s * np.sqrt(vals)
        e = Ellipse(
            xy=mean,
            width=w,
            height=h,
            angle=angle,
            edgecolor=color,
            facecolor="none",
            lw=2.0,
            alpha=0.85 if s == n_std[0] else 0.45,
            **kw,
        )
        ax.add_patch(e)


def make_mixture_data(seed=7):
    """Three anisotropic Gaussian clusters in 2D."""
    rng = np.random.default_rng(seed)
    means = np.array([[-2.5, -1.0], [1.5, 2.5], [3.0, -2.0]])
    covs = [
        np.array([[1.2, 0.6], [0.6, 0.8]]),
        np.array([[0.8, -0.5], [-0.5, 1.0]]),
        np.array([[1.6, 0.0], [0.0, 0.4]]),
    ]
    weights = np.array([0.4, 0.35, 0.25])
    sizes = (weights * 600).astype(int)
    X_parts, y_parts = [], []
    for k in range(3):
        Xk = rng.multivariate_normal(means[k], covs[k], size=sizes[k])
        X_parts.append(Xk)
        y_parts.append(np.full(sizes[k], k))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], means, covs, weights


# ---------------------------------------------------------------------------
# Figure 1: GMM clusters with covariance ellipses
# ---------------------------------------------------------------------------
def fig1_gmm_clusters() -> None:
    X, y_true, _, _, _ = make_mixture_data()
    gmm = GaussianMixture(
        n_components=3, covariance_type="full", random_state=0, n_init=5
    ).fit(X)
    labels = gmm.predict(X)

    # Reorder so colors stay stable.
    order = np.argsort(gmm.means_[:, 0])
    perm = np.argsort(order)
    labels = perm[labels]
    means = gmm.means_[order]
    covs = gmm.covariances_[order]
    weights = gmm.weights_[order]

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    cmap = ListedColormap([C_BLUE, C_PURPLE, C_GREEN])
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, s=18, alpha=0.55,
               edgecolors="white", linewidths=0.4)

    for k in range(3):
        cov_ellipse(ax, means[k], covs[k], color=PALETTE[k], n_std=(1.0, 2.0))
        ax.scatter(*means[k], marker="X", s=240, color=PALETTE[k],
                   edgecolors=C_DARK, linewidths=1.5, zorder=5)
        ax.annotate(
            rf"$\pi_{k+1}={weights[k]:.2f}$",
            xy=means[k], xytext=(means[k, 0] + 0.4, means[k, 1] + 0.55),
            fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=PALETTE[k], lw=1.2),
        )

    ax.set_title(
        "GMM: three Gaussian components with 1- and 2-sigma covariance ellipses",
        fontsize=12, color=C_DARK,
    )
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "fig1_gmm_clusters")


# ---------------------------------------------------------------------------
# Figure 2: E-step responsibilities (heatmap + matrix view)
# ---------------------------------------------------------------------------
def fig2_e_step() -> None:
    X, _, _, _, _ = make_mixture_data()
    gmm = GaussianMixture(
        n_components=3, covariance_type="full", random_state=0, n_init=5
    ).fit(X)
    order = np.argsort(gmm.means_[:, 0])
    means = gmm.means_[order]
    covs = gmm.covariances_[order]
    weights = gmm.weights_[order]

    # Grid of points.
    x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 220)
    x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 220)
    XX, YY = np.meshgrid(x1, x2)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    pdfs = np.column_stack([
        weights[k] * multivariate_normal.pdf(grid, mean=means[k], cov=covs[k])
        for k in range(3)
    ])
    gamma = pdfs / (pdfs.sum(axis=1, keepdims=True) + 1e-12)
    # Color = responsibility-weighted RGB.
    rgb = np.stack([gamma[:, 0], gamma[:, 1], gamma[:, 2]], axis=1)
    # Map to our palette.
    palette_rgb = np.array([
        [37/255, 99/255, 235/255],
        [124/255, 58/255, 237/255],
        [16/255, 185/255, 129/255],
    ])
    grid_color = rgb @ palette_rgb
    grid_color = grid_color.reshape(XX.shape + (3,))
    grid_color = np.clip(0.45 + 0.55 * grid_color, 0, 1)

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

    # Left: soft-assignment map.
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(grid_color, extent=(x1[0], x1[-1], x2[0], x2[-1]),
              origin="lower", interpolation="bilinear", aspect="auto")
    ax.scatter(X[:, 0], X[:, 1], c=C_DARK, s=8, alpha=0.35, edgecolors="none")
    for k in range(3):
        cov_ellipse(ax, means[k], covs[k], color=PALETTE[k], n_std=(1.0,))
        ax.scatter(*means[k], marker="X", s=180, color=PALETTE[k],
                   edgecolors="white", linewidths=1.4, zorder=5)
    ax.set_title(
        r"E-step: $\gamma_{ik}$ as a soft membership map (color mix $=$ responsibilities)",
        fontsize=11, color=C_DARK,
    )
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")

    # Right: matrix view for some sample points.
    rng = np.random.default_rng(3)
    sample_idx = rng.choice(len(X), size=12, replace=False)
    pdfs_s = np.column_stack([
        weights[k] * multivariate_normal.pdf(X[sample_idx], mean=means[k], cov=covs[k])
        for k in range(3)
    ])
    gamma_s = pdfs_s / pdfs_s.sum(axis=1, keepdims=True)

    ax2 = fig.add_subplot(gs[0, 1])
    cmap = LinearSegmentedColormap.from_list(
        "resp", ["#f8fafc", C_BLUE], N=256,
    )
    im = ax2.imshow(gamma_s, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(gamma_s.shape[0]):
        for k in range(3):
            v = gamma_s[i, k]
            ax2.text(k, i, f"{v:.2f}", ha="center", va="center",
                     color=("white" if v > 0.55 else C_DARK), fontsize=9)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels([r"$k=1$", r"$k=2$", r"$k=3$"])
    ax2.set_yticks(range(gamma_s.shape[0]))
    ax2.set_yticklabels([f"$x_{{{i+1}}}$" for i in range(gamma_s.shape[0])])
    ax2.set_title(
        r"Responsibility matrix $\gamma_{ik}$  (rows sum to 1)",
        fontsize=11, color=C_DARK,
    )
    fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.03,
                 label="responsibility")

    fig.tight_layout()
    _save(fig, "fig2_e_step")


# ---------------------------------------------------------------------------
# Figure 3: One M-step update -- before vs after
# ---------------------------------------------------------------------------
def fig3_m_step() -> None:
    X, _, _, _, _ = make_mixture_data(seed=11)
    rng = np.random.default_rng(0)

    # Deliberately mis-initialise.
    means_init = np.array([[-1.0, -3.0], [-1.0, 3.0], [4.0, 1.0]])
    covs_init = np.array([np.eye(2) * 1.5 for _ in range(3)])
    weights_init = np.ones(3) / 3

    def e_step(means, covs, weights):
        pdfs = np.column_stack([
            weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covs[k])
            for k in range(3)
        ])
        return pdfs / (pdfs.sum(axis=1, keepdims=True) + 1e-12)

    def m_step(gamma):
        N_k = gamma.sum(axis=0)
        weights = N_k / len(X)
        means = (gamma.T @ X) / N_k[:, None]
        covs = []
        for k in range(3):
            d = X - means[k]
            cov = (gamma[:, k, None] * d).T @ d / N_k[k] + 1e-6 * np.eye(2)
            covs.append(cov)
        return means, np.array(covs), weights

    gamma0 = e_step(means_init, covs_init, weights_init)
    means_new, covs_new, weights_new = m_step(gamma0)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.0))
    titles = [
        r"Before M-step: parameters $\boldsymbol{\theta}^{(t)}$",
        r"After M-step: $\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} Q(\boldsymbol{\theta}\mid\boldsymbol{\theta}^{(t)})$",
    ]
    for ax, title, means, covs, weights in zip(
        axes, titles,
        [means_init, means_new],
        [covs_init, covs_new],
        [weights_init, weights_new],
    ):
        # Color points by argmax responsibility under the *current* params.
        gamma = e_step(means, covs, weights)
        labels = gamma.argmax(axis=1)
        for k in range(3):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], color=PALETTE[k],
                       s=18, alpha=0.55, edgecolors="white", linewidths=0.3)
            cov_ellipse(ax, means[k], covs[k], color=PALETTE[k],
                        n_std=(1.0, 2.0))
            ax.scatter(*means[k], marker="X", s=220, color=PALETTE[k],
                       edgecolors=C_DARK, linewidths=1.4, zorder=5)
            ax.annotate(rf"$\pi_{k+1}={weights[k]:.2f}$",
                        xy=means[k],
                        xytext=(means[k, 0] + 0.3, means[k, 1] + 0.5),
                        fontsize=10, color=C_DARK,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=PALETTE[k], lw=1.0))
        ax.set_title(title, fontsize=11, color=C_DARK)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal")
        ax.set_xlim(-6, 7); ax.set_ylim(-5, 6)

    # Draw arrows showing how each mean moved.
    for k in range(3):
        for ax in (axes[1],):
            ax.annotate(
                "",
                xy=means_new[k], xytext=means_init[k],
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.6,
                                alpha=0.85, shrinkA=8, shrinkB=8),
            )

    fig.suptitle(
        r"One EM iteration: weighted means $\mu_k = \sum_i \gamma_{ik}\, x_i / N_k$  pull components toward their data",
        fontsize=12, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig3_m_step")


# ---------------------------------------------------------------------------
# Figure 4: Monotone log-likelihood across iterations
# ---------------------------------------------------------------------------
def _run_em(X, K, max_iter=60, seed=0, return_traj=False):
    """Minimal EM run that records log-likelihood per iteration."""
    rng = np.random.default_rng(seed)
    N, d = X.shape
    idx = rng.choice(N, K, replace=False)
    means = X[idx].copy()
    covs = np.array([np.cov(X.T) + 1e-3 * np.eye(d) for _ in range(K)])
    weights = np.ones(K) / K

    lls = []
    elbos = []
    for _ in range(max_iter):
        # Compute log-likelihood (current params).
        comp = np.column_stack([
            weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covs[k])
            for k in range(K)
        ])
        ll = float(np.sum(np.log(comp.sum(axis=1) + 1e-300)))
        lls.append(ll)
        # E-step:
        gamma = comp / (comp.sum(axis=1, keepdims=True) + 1e-300)
        # ELBO with q = posterior under current params -> ELBO == ll.
        # We instead record ELBO with q = previous-iteration posterior to
        # show the gap; but here the standard EM ELBO equals ll right
        # after E-step. Compute the post-M-step ELBO with frozen q below.
        # M-step:
        N_k = gamma.sum(axis=0)
        weights = N_k / N
        means = (gamma.T @ X) / N_k[:, None]
        for k in range(K):
            diff = X - means[k]
            covs[k] = (gamma[:, k, None] * diff).T @ diff / N_k[k] + 1e-6 * np.eye(d)
        # ELBO with q kept fixed at the just-computed posterior:
        comp_new = np.column_stack([
            weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covs[k])
            for k in range(K)
        ])
        # ELBO = E_q[log p(x,z|theta)] - E_q[log q(z)]
        log_joint = np.log(comp_new + 1e-300)        # (N,K)
        log_q = np.log(gamma + 1e-300)
        elbo = float(np.sum(gamma * (log_joint - log_q)))
        elbos.append(elbo)
    if return_traj:
        return lls, elbos, means, covs, weights
    return lls


def fig4_loglik_monotone() -> None:
    X, _, _, _, _ = make_mixture_data()

    # Several restarts to show different basins.
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    seeds = [0, 1, 2, 3, 4]
    final_vals = []
    for i, s in enumerate(seeds):
        lls = _run_em(X, K=3, max_iter=50, seed=s)
        ax.plot(range(1, len(lls) + 1), lls,
                color=PALETTE[i % len(PALETTE)] if i < 4 else C_GRAY,
                lw=2.0, alpha=0.9, label=f"restart {i+1}  (final $\\ell={lls[-1]:.1f}$)")
        final_vals.append(lls[-1])

    # Reference: sklearn best.
    best_skl = -np.inf
    for s in range(20):
        g = GaussianMixture(n_components=3, covariance_type="full",
                            random_state=s, n_init=1).fit(X)
        best_skl = max(best_skl, float(g.score(X) * len(X)))
    ax.axhline(best_skl, color=C_DARK, ls="--", lw=1.2, alpha=0.7,
               label=f"sklearn best ($n\\_init=20$): $\\ell={best_skl:.1f}$")

    ax.set_xlabel("EM iteration $t$")
    ax.set_ylabel(r"Log-likelihood $\ell(\boldsymbol{\theta}^{(t)})$")
    ax.set_title(
        "EM monotonicity: $\\ell(\\boldsymbol{\\theta}^{(t+1)}) \\geq \\ell(\\boldsymbol{\\theta}^{(t)})$ for every iteration",
        fontsize=11, color=C_DARK,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    _save(fig, "fig4_loglik_monotone")


# ---------------------------------------------------------------------------
# Figure 5: BIC / AIC for choosing K
# ---------------------------------------------------------------------------
def fig5_bic_aic() -> None:
    X, _, _, _, _ = make_mixture_data()
    Ks = list(range(1, 9))
    bics, aics = [], []
    for K in Ks:
        g = GaussianMixture(n_components=K, covariance_type="full",
                            random_state=0, n_init=5).fit(X)
        bics.append(g.bic(X))
        aics.append(g.aic(X))
    bics = np.array(bics); aics = np.array(aics)
    k_bic = Ks[int(np.argmin(bics))]
    k_aic = Ks[int(np.argmin(aics))]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(Ks, bics, "o-", color=C_BLUE, lw=2.4, markersize=8, label="BIC")
    ax.plot(Ks, aics, "s-", color=C_PURPLE, lw=2.4, markersize=8, label="AIC")
    ax.axvline(k_bic, color=C_BLUE, ls="--", alpha=0.5)
    ax.axvline(k_aic, color=C_PURPLE, ls=":", alpha=0.5)
    ax.scatter([k_bic], [bics.min()], color=C_AMBER, s=160, zorder=5,
               edgecolors=C_DARK, linewidths=1.5)
    ax.annotate(
        f"BIC argmin: $K^*={k_bic}$",
        xy=(k_bic, bics.min()),
        xytext=(k_bic + 0.4, bics.min() + (bics.max() - bics.min()) * 0.18),
        fontsize=10, color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_AMBER, lw=1.2),
        arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.4),
    )
    ax.set_xlabel("Number of components $K$")
    ax.set_ylabel("Information criterion (lower is better)")
    ax.set_title(
        "Model selection for GMM: BIC penalises complexity more aggressively than AIC",
        fontsize=11, color=C_DARK,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig5_bic_aic")


# ---------------------------------------------------------------------------
# Figure 6: K-means vs GMM on anisotropic data
# ---------------------------------------------------------------------------
def fig6_kmeans_vs_gmm() -> None:
    rng = np.random.default_rng(2)
    # Anisotropic by stretching after a rotation.
    Xb, yb = make_blobs(n_samples=600, centers=3, cluster_std=0.6,
                        random_state=2)
    transform = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = Xb @ transform

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    gmm = GaussianMixture(n_components=3, covariance_type="full",
                          random_state=0, n_init=5).fit(X)

    # Order both by mean x-coord for stable colors.
    def reorder(centers):
        order = np.argsort(centers[:, 0])
        return order, np.argsort(order)

    order_km, perm_km = reorder(km.cluster_centers_)
    order_gm, perm_gm = reorder(gmm.means_)

    km_labels = perm_km[km.labels_]
    gm_labels = perm_gm[gmm.predict(X)]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.0))
    cmap = ListedColormap([C_BLUE, C_PURPLE, C_GREEN])

    # K-means panel.
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=km_labels, cmap=cmap, s=18,
               alpha=0.55, edgecolors="white", linewidths=0.3)
    for k in range(3):
        c = km.cluster_centers_[order_km[k]]
        ax.scatter(*c, marker="X", s=240, color=PALETTE[k],
                   edgecolors=C_DARK, linewidths=1.5, zorder=5)
        # Spherical "implicit" boundary radius -- average distance.
        mask = km_labels == k
        r = np.linalg.norm(X[mask] - c, axis=1).mean()
        circle = plt.Circle(c, r, fill=False, color=PALETTE[k], lw=1.8,
                            ls="--", alpha=0.7)
        ax.add_patch(circle)
    ax.set_title("K-means: hard assignment, isotropic (spherical) clusters",
                 fontsize=11, color=C_DARK)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")

    # GMM panel.
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c=gm_labels, cmap=cmap, s=18,
               alpha=0.55, edgecolors="white", linewidths=0.3)
    for k in range(3):
        m = gmm.means_[order_gm[k]]
        c = gmm.covariances_[order_gm[k]]
        cov_ellipse(ax, m, c, color=PALETTE[k], n_std=(1.0, 2.0))
        ax.scatter(*m, marker="X", s=240, color=PALETTE[k],
                   edgecolors=C_DARK, linewidths=1.5, zorder=5)
    ax.set_title("GMM: soft responsibilities, full covariance (elliptical)",
                 fontsize=11, color=C_DARK)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")

    fig.suptitle(
        "K-means is the hard, spherical limit of GMM ($\\Sigma_k=\\epsilon I,\\ \\epsilon\\to 0$)",
        fontsize=12, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_kmeans_vs_gmm")


# ---------------------------------------------------------------------------
# Figure 7: ELBO vs log-likelihood (KL gap that closes at every E-step)
# ---------------------------------------------------------------------------
def fig7_elbo_vs_loglik() -> None:
    X, _, _, _, _ = make_mixture_data()

    # Custom EM that records (ll, elbo_pre_E, elbo_post_M) per iteration so
    # we can see the KL gap visually:
    #   - Just before E-step: ELBO_old < ll  (gap = KL[q_old || p(z|x,theta_t)])
    #   - Right after E-step: ELBO == ll      (gap = 0)
    #   - Right after M-step: ELBO_new <= ll_new (gap may reopen)
    rng = np.random.default_rng(0)
    N, d = X.shape
    K = 3
    idx = rng.choice(N, K, replace=False)
    means = X[idx].copy()
    covs = np.array([np.cov(X.T) + 1e-3 * np.eye(d) for _ in range(K)])
    weights = np.ones(K) / K
    gamma = np.full((N, K), 1.0 / K)  # initial q

    iters = 25
    iter_axis = []
    ll_curve = []
    elbo_curve = []
    state_labels = []  # 'pre-E', 'post-E', 'post-M'

    def loglik(means, covs, weights):
        comp = np.column_stack([
            weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covs[k])
            for k in range(K)
        ])
        return float(np.sum(np.log(comp.sum(axis=1) + 1e-300)))

    def elbo(gamma, means, covs, weights):
        comp = np.column_stack([
            weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covs[k])
            for k in range(K)
        ])
        log_joint = np.log(comp + 1e-300)
        log_q = np.log(gamma + 1e-300)
        return float(np.sum(gamma * (log_joint - log_q)))

    t = 0.0
    for it in range(iters):
        # State: pre-E (q is from previous iter, theta unchanged)
        ll = loglik(means, covs, weights)
        eb = elbo(gamma, means, covs, weights)
        iter_axis.append(t); ll_curve.append(ll); elbo_curve.append(eb)
        state_labels.append("pre-E")
        t += 0.5

        # E-step
        comp = np.column_stack([
            weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covs[k])
            for k in range(K)
        ])
        gamma = comp / (comp.sum(axis=1, keepdims=True) + 1e-300)
        eb = elbo(gamma, means, covs, weights)
        iter_axis.append(t); ll_curve.append(ll); elbo_curve.append(eb)
        state_labels.append("post-E")
        t += 0.5

        # M-step
        N_k = gamma.sum(axis=0)
        weights = N_k / N
        means = (gamma.T @ X) / N_k[:, None]
        for k in range(K):
            diff = X - means[k]
            covs[k] = (gamma[:, k, None] * diff).T @ diff / N_k[k] + 1e-6 * np.eye(d)

    iter_axis = np.array(iter_axis)
    ll_curve = np.array(ll_curve)
    elbo_curve = np.array(elbo_curve)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(iter_axis, ll_curve, "-", color=C_BLUE, lw=2.4,
            label=r"$\log p(\mathbf{X}\mid \boldsymbol{\theta}^{(t)})$")
    ax.plot(iter_axis, elbo_curve, "-", color=C_PURPLE, lw=2.4,
            label=r"$\mathrm{ELBO}(q,\boldsymbol{\theta}^{(t)})$")
    ax.fill_between(iter_axis, elbo_curve, ll_curve,
                    color=C_AMBER, alpha=0.18,
                    label=r"$\mathrm{KL}[\,q\,\Vert\, p(\mathbf{z}\mid\mathbf{x},\boldsymbol{\theta})\,]$ gap")

    # Mark a couple of E-steps where the gap closes.
    post_e_idx = [i for i, s in enumerate(state_labels) if s == "post-E"][:6]
    for j, i in enumerate(post_e_idx):
        ax.scatter(iter_axis[i], ll_curve[i], color=C_GREEN,
                   s=70, zorder=5, edgecolors=C_DARK, linewidths=1.0,
                   label="after E-step: KL$=0$" if j == 0 else None)

    ax.set_xlabel("EM iteration (E-step then M-step per integer step)")
    ax.set_ylabel("Objective value")
    ax.set_title(
        "Two views of EM: E-step closes the KL gap, M-step raises both curves",
        fontsize=11, color=C_DARK,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    _save(fig, "fig7_elbo_vs_loglik")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 13 figures (EM Algorithm and GMM)...")
    fig1_gmm_clusters()
    fig2_e_step()
    fig3_m_step()
    fig4_loglik_monotone()
    fig5_bic_aic()
    fig6_kmeans_vs_gmm()
    fig7_elbo_vs_loglik()
    print("Done.")


if __name__ == "__main__":
    main()
