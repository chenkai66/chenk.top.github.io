"""
Figure generation script for ML Math Derivations Part 17:
Dimensionality Reduction and PCA.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates a single specific idea so the geometry of the methods
becomes visible.

Figures:
    fig1_pca_2d_gaussian       PCA on a correlated 2D Gaussian cloud.
                               Shows centred data with PC1/PC2 axes scaled
                               by sqrt(eigenvalue), and rotated coordinates.
    fig2_scree_plot            Scree plot of eigenvalues for a synthetic
                               dataset with low intrinsic rank, with the
                               "elbow" marked.
    fig3_cumulative_variance   Cumulative explained variance ratio with
                               90% / 95% / 99% retention thresholds.
    fig4_reconstruction_vs_k   PCA reconstruction quality vs k on the
                               digits dataset: MSE curve and a 4x6 grid of
                               sample reconstructions for k in {2,8,16,32}.
    fig5_pca_lda_tsne          PCA vs LDA vs t-SNE on the digits dataset:
                               three 2D embeddings of the same data,
                               coloured by class.
    fig6_kernel_pca_swissroll  Linear PCA vs Kernel PCA (RBF) on a swiss
                               roll: the linear projection collapses the
                               manifold while RBF unrolls it.
    fig7_ica_vs_pca            ICA vs PCA on a 2-source mixing problem
                               (sine + sawtooth): PCA recovers
                               variance-aligned axes; ICA recovers the
                               independent sources.

Usage:
    python3 scripts/figures/ml-math-derivations/17-pca.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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
C_BG = COLORS["bg"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "17-Dimensionality-Reduction-and-PCA"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "17-降维与主成分分析"
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
# Figure 1: PCA on a correlated 2D Gaussian cloud
# ---------------------------------------------------------------------------
def fig1_pca_2d_gaussian() -> None:
    rng = np.random.default_rng(7)
    N = 600
    # Build correlated Gaussian via known covariance.
    cov = np.array([[3.2, 1.8], [1.8, 1.4]])
    L = np.linalg.cholesky(cov)
    X = rng.standard_normal((N, 2)) @ L.T
    X = X - X.mean(axis=0)

    pca = PCA(n_components=2).fit(X)
    eigvals = pca.explained_variance_
    W = pca.components_  # rows are PCs
    Z = X @ W.T

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    # --- (a) original space with PC axes ---
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=14, c=C_BLUE, alpha=0.45,
               edgecolors="none")

    # 1- and 2-sigma confidence ellipses for visual reference
    eigvals_full, eigvecs_full = np.linalg.eigh(np.cov(X.T))
    order = np.argsort(eigvals_full)[::-1]
    eigvals_full = eigvals_full[order]
    eigvecs_full = eigvecs_full[:, order]
    angle = np.degrees(np.arctan2(eigvecs_full[1, 0], eigvecs_full[0, 0]))
    for k_sig, alpha in [(1, 0.35), (2, 0.18)]:
        ell = Ellipse(
            (0, 0),
            width=2 * k_sig * np.sqrt(eigvals_full[0]),
            height=2 * k_sig * np.sqrt(eigvals_full[1]),
            angle=angle, facecolor=C_BLUE, edgecolor=C_BLUE,
            alpha=alpha, lw=1.2,
        )
        ax.add_patch(ell)

    # PC arrows scaled by sqrt(eigvalue)
    for i, color, name in [(0, C_AMBER, "PC1"), (1, C_GREEN, "PC2")]:
        v = W[i] * np.sqrt(eigvals[i]) * 2.0
        arr = FancyArrowPatch(
            (0, 0), (v[0], v[1]),
            arrowstyle="-|>", mutation_scale=18,
            color=color, lw=2.4, zorder=5,
        )
        ax.add_patch(arr)
        ax.text(v[0] * 1.18, v[1] * 1.18,
                f"{name}\n$\\lambda={eigvals[i]:.2f}$",
                color=color, fontsize=11, fontweight="bold",
                ha="center", va="center")

    ax.set_xlim(-7, 7)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=11)
    ax.set_ylabel("$x_2$", fontsize=11)
    ax.set_title("(a) Centred data with principal axes",
                 fontsize=12, fontweight="bold")

    # --- (b) rotated coordinates (PC space) ---
    ax = axes[1]
    ax.scatter(Z[:, 0], Z[:, 1], s=14, c=C_PURPLE, alpha=0.45,
               edgecolors="none")
    # Marginal variance bars
    ax.axhline(0, color=C_GRAY, lw=0.8, ls="--")
    ax.axvline(0, color=C_GRAY, lw=0.8, ls="--")
    # 2-sigma horizontal/vertical guides on the PC axes
    s1, s2 = np.sqrt(eigvals)
    ax.plot([-2 * s1, 2 * s1], [0, 0], color=C_AMBER, lw=2.6,
            solid_capstyle="round", alpha=0.85)
    ax.plot([0, 0], [-2 * s2, 2 * s2], color=C_GREEN, lw=2.6,
            solid_capstyle="round", alpha=0.85)
    ax.text(2 * s1 + 0.2, 0.25,
            f"PC1 spread\n$2\\sigma={2 * s1:.2f}$",
            color=C_AMBER, fontsize=10, fontweight="bold")
    ax.text(0.25, 2 * s2 + 0.1,
            f"PC2 spread\n$2\\sigma={2 * s2:.2f}$",
            color=C_GREEN, fontsize=10, fontweight="bold")

    ax.set_xlim(-7, 7)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$z_1$ (PC1 score)", fontsize=11)
    ax.set_ylabel("$z_2$ (PC2 score)", fontsize=11)
    ax.set_title("(b) Same data in PC coordinates",
                 fontsize=12, fontweight="bold")

    fig.suptitle("PCA on a 2D Gaussian cloud: rotation onto axes of "
                 "maximum variance",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_pca_2d_gaussian")


# ---------------------------------------------------------------------------
# Figure 2: Scree plot
# ---------------------------------------------------------------------------
def fig2_scree_plot() -> None:
    rng = np.random.default_rng(11)
    N, D = 400, 20
    # Latent rank-5 signal + small isotropic noise so the scree has a
    # visible elbow.
    k_latent = 5
    Z = rng.standard_normal((N, k_latent))
    A = rng.standard_normal((D, k_latent))
    A *= np.array([3.5, 2.4, 1.6, 1.0, 0.6])
    X = Z @ A.T + 0.18 * rng.standard_normal((N, D))
    X -= X.mean(axis=0)

    pca = PCA().fit(X)
    eig = pca.explained_variance_

    fig, ax = plt.subplots(figsize=(10, 5.2))
    idx = np.arange(1, D + 1)
    bars = ax.bar(idx, eig, color=C_LIGHT, edgecolor=C_DARK, lw=0.8,
                  zorder=3)
    for i in range(k_latent):
        bars[i].set_color(C_BLUE)
    ax.plot(idx, eig, color=C_PURPLE, lw=2.0, marker="o",
            markersize=7, markerfacecolor="white",
            markeredgecolor=C_PURPLE, markeredgewidth=1.8, zorder=4)

    # Mark the elbow.
    elbow = k_latent
    ax.axvline(elbow + 0.5, color=C_AMBER, lw=1.6, ls="--", zorder=2)
    ax.annotate(
        f"elbow at $k = {elbow}$\n(latent dim)",
        xy=(elbow + 0.5, eig[elbow - 1]),
        xytext=(elbow + 3.5, eig[0] * 0.65),
        fontsize=11, color=C_AMBER, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.4),
    )

    ax.set_xlabel("Component index $k$", fontsize=11)
    ax.set_ylabel(r"Eigenvalue $\lambda_k$ (variance along PC$_k$)",
                  fontsize=11)
    ax.set_title("Scree plot: where does signal end and noise begin?",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(idx)
    ax.set_xlim(0.3, D + 0.7)

    fig.tight_layout()
    _save(fig, "fig2_scree_plot")


# ---------------------------------------------------------------------------
# Figure 3: Cumulative explained variance
# ---------------------------------------------------------------------------
def fig3_cumulative_variance() -> None:
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)  # (1797, 64)

    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    D = len(cum)
    idx = np.arange(1, D + 1)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(idx, cum, color=C_BLUE, lw=2.4, zorder=3)
    ax.fill_between(idx, 0, cum, color=C_BLUE, alpha=0.10, zorder=2)

    for thresh, color in [(0.90, C_GREEN), (0.95, C_AMBER),
                          (0.99, C_PURPLE)]:
        k = int(np.searchsorted(cum, thresh) + 1)
        ax.axhline(thresh, color=color, lw=1.2, ls="--", alpha=0.7)
        ax.axvline(k, color=color, lw=1.2, ls="--", alpha=0.7)
        ax.scatter([k], [cum[k - 1]], color=color, s=80, zorder=5,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(
            f"{int(thresh * 100)}% @ $k={k}$",
            xy=(k, cum[k - 1]),
            xytext=(k + 4, thresh - 0.06),
            fontsize=11, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

    ax.set_xlabel("Number of components $k$", fontsize=11)
    ax.set_ylabel("Cumulative explained variance",
                  fontsize=11)
    ax.set_title(
        "Cumulative variance on UCI digits "
        f"($D = {D}$ pixel features)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, D)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, "fig3_cumulative_variance")


# ---------------------------------------------------------------------------
# Figure 4: Reconstruction quality vs k
# ---------------------------------------------------------------------------
def fig4_reconstruction_vs_k() -> None:
    digits = load_digits()
    X = digits.data.astype(float)
    images = digits.images  # (N, 8, 8)
    mu = X.mean(axis=0)
    Xc = X - mu

    Ks = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    mse = []
    pca_full = PCA(n_components=64).fit(Xc)
    Z_full = pca_full.transform(Xc)
    for k in Ks:
        Xr = Z_full[:, :k] @ pca_full.components_[:k] + mu
        mse.append(np.mean((X - Xr) ** 2))

    fig = plt.figure(figsize=(14, 6.6))
    # Layout: column 0 = MSE curve, columns 1..5 = original + 4 recons,
    # rows 0..1 = two sample digits.
    gs = fig.add_gridspec(
        2, 6, width_ratios=[2.6, 1, 1, 1, 1, 1], height_ratios=[1, 1],
        hspace=0.18, wspace=0.18,
    )

    # Left: MSE curve.
    ax_curve = fig.add_subplot(gs[:, 0])
    ax_curve.plot(Ks, mse, color=C_BLUE, lw=2.4, marker="o",
                  markersize=8, markerfacecolor="white",
                  markeredgewidth=1.8)
    for k_show, color in [(2, C_PURPLE), (8, C_GREEN), (16, C_AMBER),
                          (32, C_BLUE)]:
        i = Ks.index(k_show)
        ax_curve.scatter([k_show], [mse[i]], color=color, s=120, zorder=5,
                         edgecolor="white", linewidth=1.8)
        ax_curve.annotate(
            f"$k={k_show}$",
            xy=(k_show, mse[i]),
            xytext=(k_show + 4, mse[i] + max(mse) * 0.05),
            fontsize=11, color=color, fontweight="bold",
        )
    ax_curve.set_xlabel("Number of components $k$", fontsize=11)
    ax_curve.set_ylabel("Mean reconstruction MSE", fontsize=11)
    ax_curve.set_title("Reconstruction error vs $k$ (digits, $D=64$)",
                       fontsize=12, fontweight="bold")
    ax_curve.set_xticks(Ks)

    # Right grid: original + reconstructions for two sample digits.
    sample_idx = [7, 17]
    show_ks = [2, 8, 16, 32]

    def _draw_image(ax, img, title=None, ylabel=None):
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=16)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(C_DARK)
            spine.set_linewidth(0.8)
        if title is not None:
            ax.set_title(title, fontsize=11, color=C_DARK,
                         fontweight="bold")
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=10, color=C_DARK)

    for row, si in enumerate(sample_idx):
        # Original.
        ax = fig.add_subplot(gs[row, 1])
        _draw_image(
            ax, images[si],
            title="original" if row == 0 else None,
            ylabel=f"sample {si}",
        )
        # Reconstructions.
        for col, k in enumerate(show_ks):
            Xr_full = Z_full[:, :k] @ pca_full.components_[:k] + mu
            ax = fig.add_subplot(gs[row, 2 + col])
            _draw_image(
                ax, Xr_full[si].reshape(8, 8),
                title=f"$k={k}$" if row == 0 else None,
            )

    fig.suptitle(
        "PCA reconstruction quality vs number of components",
        fontsize=13, fontweight="bold", y=1.0,
    )
    _save(fig, "fig4_reconstruction_vs_k")


# ---------------------------------------------------------------------------
# Figure 5: PCA vs LDA vs t-SNE on digits
# ---------------------------------------------------------------------------
def fig5_pca_lda_tsne() -> None:
    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target
    Xs = StandardScaler().fit_transform(X)

    Z_pca = PCA(n_components=2).fit_transform(Xs)
    Z_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(Xs, y)
    Z_tsne = TSNE(
        n_components=2, perplexity=30, init="pca",
        learning_rate="auto", random_state=0,
    ).fit_transform(Xs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    cmap = plt.get_cmap("tab10")
    titles = [
        ("PCA", "unsupervised, max variance"),
        ("LDA", "supervised, max class separation"),
        ("t-SNE", "neighbour-preserving, nonlinear"),
    ]
    embeddings = [Z_pca, Z_lda, Z_tsne]
    for ax, (name, sub), Z in zip(axes, titles, embeddings):
        for c in range(10):
            mask = y == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=14,
                       color=cmap(c), alpha=0.75, edgecolors="none",
                       label=str(c))
        ax.set_title(f"{name}\n{sub}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(C_LIGHT)
    leg = axes[-1].legend(
        title="digit", loc="center left",
        bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False,
        markerscale=1.4,
    )
    leg.get_title().set_fontweight("bold")

    fig.suptitle(
        "Same digits, three projections: variance vs class separation "
        "vs neighbourhoods",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    _save(fig, "fig5_pca_lda_tsne")


# ---------------------------------------------------------------------------
# Figure 6: Kernel PCA on a swiss roll
# ---------------------------------------------------------------------------
def fig6_kernel_pca_swissroll() -> None:
    X, t = make_swiss_roll(n_samples=1500, noise=0.05, random_state=2)

    pca_lin = PCA(n_components=2).fit_transform(X)
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04,
                     random_state=0).fit_transform(X)

    fig = plt.figure(figsize=(15, 5.2))
    cmap = plt.get_cmap("Spectral")

    # 3D swiss roll.
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=cmap, s=10,
                 edgecolors="none")
    ax3d.set_title("Swiss roll in $\\mathbb{R}^3$\n(colour = position "
                   "along the manifold)",
                   fontsize=11, fontweight="bold")
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.view_init(elev=12, azim=-72)

    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(pca_lin[:, 0], pca_lin[:, 1], c=t, cmap=cmap, s=12,
               edgecolors="none")
    ax.set_title("Linear PCA\n(folds the roll onto itself)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(kpca[:, 0], kpca[:, 1], c=t, cmap=cmap, s=12,
               edgecolors="none")
    ax.set_title("Kernel PCA, RBF ($\\gamma=0.04$)\n"
                 "(unrolls the manifold)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(
        "Kernel PCA recovers manifold structure that linear PCA destroys",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_kernel_pca_swissroll")


# ---------------------------------------------------------------------------
# Figure 7: ICA vs PCA on source separation
# ---------------------------------------------------------------------------
def fig7_ica_vs_pca() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    time = np.linspace(0, 8, n)
    s1 = np.sin(2 * time)                          # sinusoid
    s2 = np.sign(np.sin(3 * time))                 # square wave
    S = np.column_stack([s1, s2])
    S += 0.04 * rng.standard_normal(S.shape)
    S /= S.std(axis=0)

    # Mixing matrix: non-orthogonal so PCA cannot recover it.
    A = np.array([[1.0, 1.0], [0.5, 2.0]])
    X = S @ A.T

    pca_sources = PCA(n_components=2, whiten=True).fit_transform(X)
    ica_sources = FastICA(n_components=2, random_state=0,
                          whiten="unit-variance").fit_transform(X)

    # Sign / scale alignment with true sources for cleaner display.
    def _align(est, ref):
        out = np.empty_like(est)
        for j in range(est.shape[1]):
            # Match each estimated component to the reference column with
            # the highest absolute correlation, with sign correction.
            corrs = [np.corrcoef(est[:, j], ref[:, k])[0, 1]
                     for k in range(ref.shape[1])]
            k = int(np.argmax(np.abs(corrs)))
            out[:, j] = np.sign(corrs[k]) * est[:, j]
        return out

    pca_sources = _align(pca_sources, S)
    ica_sources = _align(ica_sources, S)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.0), sharex=True)

    axes[0].plot(time, S[:, 0], color=C_BLUE, lw=1.2,
                 label="source 1 (sine)")
    axes[0].plot(time, S[:, 1], color=C_AMBER, lw=1.2,
                 label="source 2 (square)")
    axes[0].set_title("True independent sources $s_1, s_2$",
                      fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9, frameon=False, ncol=2)

    axes[1].plot(time, X[:, 0], color=C_PURPLE, lw=1.0,
                 label="mixture $x_1$")
    axes[1].plot(time, X[:, 1], color=C_GREEN, lw=1.0,
                 label="mixture $x_2$")
    axes[1].set_title("Observed mixtures $\\mathbf{x} = A\\mathbf{s}$",
                      fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9, frameon=False, ncol=2)

    axes[2].plot(time, pca_sources[:, 0], color=C_BLUE, lw=1.0)
    axes[2].plot(time, pca_sources[:, 1], color=C_AMBER, lw=1.0)
    axes[2].set_title(
        "PCA components: orthogonal axes of max variance "
        "(do NOT recover sources)",
        fontsize=11, fontweight="bold",
    )

    axes[3].plot(time, ica_sources[:, 0], color=C_BLUE, lw=1.0)
    axes[3].plot(time, ica_sources[:, 1], color=C_AMBER, lw=1.0)
    axes[3].set_title(
        "ICA components: maximally non-Gaussian "
        "(recover the original sources)",
        fontsize=11, fontweight="bold",
    )
    axes[3].set_xlabel("time", fontsize=10)

    for ax in axes:
        ax.set_yticks([])

    fig.suptitle(
        "Source separation: PCA decorrelates, ICA makes signals "
        "independent",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    _save(fig, "fig7_ica_vs_pca")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 17 figures...")
    fig1_pca_2d_gaussian()
    fig2_scree_plot()
    fig3_cumulative_variance()
    fig4_reconstruction_vs_k()
    fig5_pca_lda_tsne()
    fig6_kernel_pca_swissroll()
    fig7_ica_vs_pca()
    print("Done.")


if __name__ == "__main__":
    main()
