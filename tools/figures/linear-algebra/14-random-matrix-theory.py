"""
Figure generation script for Linear Algebra Chapter 14: Random Matrix Theory.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in 3Blue1Brown style.

Figures:
    fig1_wigner_semicircle     Empirical eigenvalue density of a normalised GOE
                               matrix vs the Wigner semicircle law f(x) =
                               sqrt(4 - x^2)/(2 pi).
    fig2_marchenko_pastur      Empirical density of sample covariance
                               eigenvalues for several aspect ratios gamma vs
                               the Marchenko-Pastur theoretical curves.
    fig3_eigenvalue_repulsion  Nearest-neighbour spacing distribution for GOE
                               eigenvalues (the Wigner surmise) compared with
                               the Poisson distribution that independent
                               variables would give - showing repulsion at
                               s = 0.
    fig4_random_vs_diagonal    Side-by-side scatter of eigenvalues on the real
                               line: a random Wigner matrix (broad, smooth
                               spectrum) vs a deterministic diagonal matrix
                               (clustered points).
    fig5_free_convolution      Free additive convolution of two semicircle
                               laws is again a semicircle with summed
                               variance - a visualisation of the free CLT.
    fig6_covariance_estimation Heatmap of sample vs cleaned vs true covariance
                               for a high-dimensional identity covariance,
                               showing how RMT cleaning recovers structure.
    fig7_spiked_covariance     Spiked model: bulk MP density plus a few outlier
                               eigenvalues for the planted signal directions,
                               with the BBP phase transition marked.

Usage:
    python3 scripts/figures/linear-algebra/14-random-matrix-theory.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary - empirical / data
C_PURPLE = "#7c3aed"   # secondary - second distribution
C_GREEN = "#10b981"    # accent - theory curves
C_AMBER = "#f59e0b"    # warning - thresholds / outliers
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150
RNG = np.random.default_rng(20240426)

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "14-random-matrix-theory"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "14-随机矩阵理论"
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
# Sampling helpers
# ---------------------------------------------------------------------------
def goe_eigenvalues(n: int, repeats: int = 1) -> np.ndarray:
    """Eigenvalues of a normalised real GOE matrix W / sqrt(n).

    Each entry is N(0, 1); we symmetrise as (A + A.T)/2 so the off-diagonal
    variance becomes 1/2 and the diagonal variance is 1. After dividing by
    sqrt(n) the limiting spectrum is the standard semicircle on [-2, 2].
    """
    eigs = []
    for _ in range(repeats):
        a = RNG.standard_normal((n, n))
        w = (a + a.T) / np.sqrt(2 * n)  # variance scaling for std semicircle
        eigs.append(np.linalg.eigvalsh(w))
    return np.concatenate(eigs)


def wishart_eigenvalues(n: int, p: int, repeats: int = 1) -> np.ndarray:
    """Eigenvalues of S = X^T X / n with X of shape (n, p), entries N(0, 1)."""
    eigs = []
    for _ in range(repeats):
        x = RNG.standard_normal((n, p))
        s = x.T @ x / n
        eigs.append(np.linalg.eigvalsh(s))
    return np.concatenate(eigs)


# ---------------------------------------------------------------------------
# Figure 1: Wigner semicircle law
# ---------------------------------------------------------------------------
def fig1_wigner_semicircle() -> None:
    n = 1500
    repeats = 30
    eigs = goe_eigenvalues(n, repeats)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.hist(
        eigs,
        bins=90,
        density=True,
        color=C_BLUE,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label=f"Empirical density (n={n}, {repeats} matrices)",
    )

    x = np.linspace(-2, 2, 600)
    y = np.sqrt(np.maximum(4 - x ** 2, 0)) / (2 * np.pi)
    ax.plot(x, y, color=C_GREEN, linewidth=2.6,
            label=r"Semicircle $\frac{1}{2\pi}\sqrt{4-x^2}$")

    ax.axvline(-2, color=C_AMBER, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(2, color=C_AMBER, linestyle="--", linewidth=1.2, alpha=0.8,
               label="Spectral edge $\\pm 2$")

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(0, 0.40)
    ax.set_xlabel("Eigenvalue $\\lambda$")
    ax.set_ylabel("Density")
    ax.set_title("Wigner semicircle law: GOE eigenvalues",
                 color=C_DARK, fontsize=13)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)

    save(fig, "fig1_wigner_semicircle")


# ---------------------------------------------------------------------------
# Figure 2: Marchenko-Pastur for several aspect ratios
# ---------------------------------------------------------------------------
def mp_density(x: np.ndarray, gamma: float) -> np.ndarray:
    lam_minus = (1 - np.sqrt(gamma)) ** 2
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    inside = (x > lam_minus) & (x < lam_plus)
    out = np.zeros_like(x)
    out[inside] = np.sqrt(
        np.maximum((lam_plus - x[inside]) * (x[inside] - lam_minus), 0)
    ) / (2 * np.pi * gamma * x[inside])
    return out


def fig2_marchenko_pastur() -> None:
    gammas = [0.1, 0.3, 0.6, 1.0]
    n = 1200
    repeats = 12

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=False)
    axes = axes.ravel()

    for ax, gamma in zip(axes, gammas):
        p = max(int(round(gamma * n)), 5)
        eigs = wishart_eigenvalues(n, p, repeats)
        # Drop zero eigenvalues from gamma >= 1 case for visualisation.
        eigs_plot = eigs[eigs > 1e-6]

        lam_minus = (1 - np.sqrt(gamma)) ** 2
        lam_plus = (1 + np.sqrt(gamma)) ** 2
        x_min = max(lam_minus - 0.2, 0)
        x_max = lam_plus + 0.3

        ax.hist(
            eigs_plot,
            bins=70,
            density=True,
            range=(x_min, x_max),
            color=C_BLUE,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4,
            label="Empirical",
        )
        x = np.linspace(x_min + 1e-3, x_max - 1e-3, 500)
        ax.plot(x, mp_density(x, gamma), color=C_GREEN, linewidth=2.4,
                label="MP theory")
        ax.axvline(lam_minus, color=C_AMBER, linestyle="--",
                   linewidth=1.0, alpha=0.85)
        ax.axvline(lam_plus, color=C_AMBER, linestyle="--",
                   linewidth=1.0, alpha=0.85)
        ax.set_title(rf"$\gamma = p/n = {gamma}$",
                     fontsize=11, color=C_DARK)
        ax.set_xlabel(r"Eigenvalue $\lambda$")
        ax.set_ylabel("Density")
        ax.set_xlim(x_min, x_max)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.suptitle("Marchenko-Pastur: sample covariance spectra at four aspect ratios",
                 fontsize=13, color=C_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig2_marchenko_pastur")


# ---------------------------------------------------------------------------
# Figure 3: Eigenvalue repulsion (Wigner surmise vs Poisson)
# ---------------------------------------------------------------------------
def fig3_eigenvalue_repulsion() -> None:
    n = 600
    repeats = 200
    spacings = []
    for _ in range(repeats):
        a = RNG.standard_normal((n, n))
        w = (a + a.T) / np.sqrt(2 * n)
        eigs = np.sort(np.linalg.eigvalsh(w))
        # Use the bulk only (avoid edge effects).
        bulk = eigs[n // 4: 3 * n // 4]
        # Local unfolding: divide by local mean spacing.
        diffs = np.diff(bulk)
        diffs = diffs / np.mean(diffs)
        spacings.append(diffs)
    spacings = np.concatenate(spacings)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.hist(
        spacings,
        bins=70,
        density=True,
        range=(0, 4),
        color=C_BLUE,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label="GOE empirical spacings",
    )
    s = np.linspace(0, 4, 500)
    wigner = (np.pi * s / 2) * np.exp(-np.pi * s ** 2 / 4)
    poisson = np.exp(-s)
    ax.plot(s, wigner, color=C_GREEN, linewidth=2.6,
            label=r"Wigner surmise $\frac{\pi s}{2}e^{-\pi s^2/4}$")
    ax.plot(s, poisson, color=C_AMBER, linewidth=2.2, linestyle="--",
            label=r"Poisson $e^{-s}$ (independent)")

    ax.annotate(
        "Eigenvalues repel:\n$P(s)\\to 0$ as $s\\to 0$",
        xy=(0.05, 0.05), xytext=(1.1, 0.55),
        fontsize=10, color=C_DARK,
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.0),
    )

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Normalised spacing $s$")
    ax.set_ylabel("Density $P(s)$")
    ax.set_title("Eigenvalue repulsion vs independent (Poisson) spacings",
                 color=C_DARK, fontsize=13)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)

    save(fig, "fig3_eigenvalue_repulsion")


# ---------------------------------------------------------------------------
# Figure 4: Random vs deterministic spectrum
# ---------------------------------------------------------------------------
def fig4_random_vs_diagonal() -> None:
    n = 400
    a = RNG.standard_normal((n, n))
    w = (a + a.T) / np.sqrt(2 * n)
    rand_eigs = np.linalg.eigvalsh(w)

    # Deterministic diagonal: eigenvalues come from a small set of clusters.
    clusters = np.array([-1.5, -0.5, 0.7, 1.6])
    counts = [n // 4] * 4
    diag_eigs = np.concatenate([
        c + 0.02 * RNG.standard_normal(k) for c, k in zip(clusters, counts)
    ])

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=True)

    ax1, ax2 = axes
    ax1.scatter(rand_eigs, np.zeros_like(rand_eigs),
                color=C_BLUE, alpha=0.5, s=14)
    x = np.linspace(-2, 2, 400)
    y = np.sqrt(np.maximum(4 - x ** 2, 0)) / (2 * np.pi)
    ax1.plot(x, 0.4 * y / y.max(), color=C_GREEN, linewidth=1.8,
             label="Semicircle envelope")
    ax1.set_yticks([])
    ax1.set_ylim(-0.1, 0.55)
    ax1.set_title("Random Wigner matrix: smooth, predictable bulk",
                  color=C_DARK, fontsize=11.5)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.scatter(diag_eigs, np.zeros_like(diag_eigs),
                color=C_PURPLE, alpha=0.7, s=14)
    for c in clusters:
        ax2.axvline(c, color=C_AMBER, linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.set_yticks([])
    ax2.set_ylim(-0.1, 0.55)
    ax2.set_title("Deterministic spectrum: a few sharp clusters",
                  color=C_DARK, fontsize=11.5)

    ax2.set_xlabel("Eigenvalue $\\lambda$")
    ax2.set_xlim(-2.5, 2.5)
    fig.suptitle("Random spectra spread out; structured spectra cluster",
                 fontsize=13, color=C_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig4_random_vs_diagonal")


# ---------------------------------------------------------------------------
# Figure 5: Free additive convolution of two semicircles
# ---------------------------------------------------------------------------
def fig5_free_convolution() -> None:
    n = 1500
    # Two independent GOEs with different scales.
    a1 = RNG.standard_normal((n, n))
    w1 = (a1 + a1.T) / np.sqrt(2 * n)  # std semicircle on [-2, 2]
    a2 = RNG.standard_normal((n, n))
    w2 = 0.7 * (a2 + a2.T) / np.sqrt(2 * n)  # scaled by 0.7

    eigs1 = np.linalg.eigvalsh(w1)
    eigs2 = np.linalg.eigvalsh(w2)
    # Sum (as random matrices) - large random matrices are asymptotically free,
    # so the spectrum of W1 + W2 should be a semicircle with variance
    # 1^2 + 0.7^2.
    eigs_sum = np.linalg.eigvalsh(w1 + w2)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bins = 70

    ax.hist(eigs1, bins=bins, density=True, color=C_BLUE, alpha=0.45,
            label=r"Spectrum of $W_1$ (radius 2)")
    ax.hist(eigs2, bins=bins, density=True, color=C_PURPLE, alpha=0.45,
            label=r"Spectrum of $W_2 = 0.7\,W'$ (radius 1.4)")
    ax.hist(eigs_sum, bins=bins, density=True, color=C_AMBER, alpha=0.55,
            label=r"Spectrum of $W_1 + W_2$")

    # Theoretical semicircle for the sum: variance = 1 + 0.49 = 1.49,
    # support radius 2 sqrt(1.49).
    sigma_sum = np.sqrt(1.0 + 0.7 ** 2)
    radius = 2 * sigma_sum
    x = np.linspace(-radius, radius, 600)
    y = np.sqrt(np.maximum(radius ** 2 - x ** 2, 0)) / (np.pi * (radius / 2) ** 2 * 2)
    # The standard semicircle of half-width R has density
    #   f(x) = 2/(pi R^2) sqrt(R^2 - x^2)
    y = (2 / (np.pi * radius ** 2)) * np.sqrt(np.maximum(radius ** 2 - x ** 2, 0))
    ax.plot(x, y, color=C_GREEN, linewidth=2.6,
            label=fr"Free CLT prediction: semicircle, radius $2\sqrt{{1.49}}$")

    ax.set_xlim(-3.2, 3.2)
    ax.set_xlabel("Eigenvalue $\\lambda$")
    ax.set_ylabel("Density")
    ax.set_title("Free additive convolution: sum of free semicircles is a semicircle",
                 color=C_DARK, fontsize=12.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    save(fig, "fig5_free_convolution")


# ---------------------------------------------------------------------------
# Figure 6: High-dimensional covariance estimation (sample vs cleaned vs true)
# ---------------------------------------------------------------------------
def fig6_covariance_estimation() -> None:
    n = 200      # samples
    p = 120      # features
    gamma = p / n
    # True covariance is identity.
    x = RNG.standard_normal((n, p))
    s_sample = x.T @ x / n

    # Clean: replace bulk eigenvalues by their mean.
    eigvals, eigvecs = np.linalg.eigh(s_sample)
    sigma_sq = np.mean(eigvals)
    lam_minus = sigma_sq * (1 - np.sqrt(gamma)) ** 2
    lam_plus = sigma_sq * (1 + np.sqrt(gamma)) ** 2
    bulk_mask = (eigvals >= lam_minus) & (eigvals <= lam_plus)
    cleaned_eigs = eigvals.copy()
    cleaned_eigs[bulk_mask] = np.mean(eigvals[bulk_mask])
    s_cleaned = eigvecs @ np.diag(cleaned_eigs) @ eigvecs.T

    s_true = np.eye(p)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    titles = [
        "Sample covariance $S$\n(noisy)",
        "RMT-cleaned $\\tilde{S}$\n(bulk averaged)",
        "True covariance $\\Sigma = I$",
    ]
    mats = [s_sample, s_cleaned, s_true]

    vmax = max(np.abs(s_sample).max(), 1.0)
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=11, color=C_DARK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85,
                 label="Matrix entry")
    fig.suptitle(
        f"High-dim covariance estimation (p={p}, n={n}, "
        rf"$\gamma=${gamma:.2f}): cleaning recovers identity",
        fontsize=12.5, color=C_DARK,
    )

    save(fig, "fig6_covariance_estimation")


# ---------------------------------------------------------------------------
# Figure 7: Spiked covariance model
# ---------------------------------------------------------------------------
def fig7_spiked_covariance() -> None:
    n = 800
    p = 400
    gamma = p / n
    # Spiked population: a few large eigenvalues planted in identity.
    spike_strengths = np.array([5.0, 3.0, 1.7])  # 1.7 is sub-critical
    pop_eigs = np.ones(p)
    pop_eigs[:len(spike_strengths)] = spike_strengths

    # Generate samples with this population covariance.
    sqrt_eigs = np.sqrt(pop_eigs)
    z = RNG.standard_normal((n, p))
    x = z * sqrt_eigs  # rows i.i.d. N(0, diag(pop_eigs))
    s = x.T @ x / n
    eigs = np.linalg.eigvalsh(s)

    lam_minus = (1 - np.sqrt(gamma)) ** 2
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    bbp_threshold = 1 + np.sqrt(gamma)  # critical population spike

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.hist(eigs, bins=80, density=True, color=C_BLUE, alpha=0.55,
            edgecolor="white", linewidth=0.4,
            label="Sample eigenvalues")

    x_grid = np.linspace(lam_minus + 1e-3, lam_plus - 1e-3, 500)
    ax.plot(x_grid, mp_density(x_grid, gamma), color=C_GREEN, linewidth=2.4,
            label="MP bulk (noise floor)")

    # Mark the BBP transition point.
    ax.axvline(lam_plus, color=C_GRAY, linestyle="--", linewidth=1.0,
               label=fr"MP edge $\lambda_+ = {lam_plus:.2f}$")

    # Predicted outlier locations from the spiked model:
    # detectable spikes (s > 1 + sqrt(gamma)) sit at s + gamma * s/(s-1).
    detectable = spike_strengths > bbp_threshold
    for s_val, det in zip(spike_strengths, detectable):
        if det:
            loc = s_val + gamma * s_val / (s_val - 1)
            ax.axvline(loc, color=C_AMBER, linewidth=2.0, alpha=0.9)
            ax.text(loc, 0.32, f"spike\nsignal={s_val:g}\nobs={loc:.2f}",
                    rotation=0, ha="center", va="bottom",
                    fontsize=8.5, color=C_DARK)
        else:
            ax.axvline(s_val, color=C_PURPLE, linewidth=1.6,
                       linestyle=":", alpha=0.9)
            ax.text(s_val, 0.32,
                    f"sub-critical\n{s_val:g} < {bbp_threshold:.2f}",
                    rotation=0, ha="center", va="bottom",
                    fontsize=8.5, color=C_PURPLE)

    ax.set_xlim(0, max(spike_strengths.max(), lam_plus) + 3)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Eigenvalue $\\lambda$")
    ax.set_ylabel("Density")
    ax.set_title(
        rf"Spiked covariance: signal vs noise (p={p}, n={n}, "
        rf"BBP threshold $1+\sqrt{{\gamma}}={bbp_threshold:.2f}$)",
        color=C_DARK, fontsize=12.5,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    save(fig, "fig7_spiked_covariance")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_wigner_semicircle()
    fig2_marchenko_pastur()
    fig3_eigenvalue_repulsion()
    fig4_random_vs_diagonal()
    fig5_free_convolution()
    fig6_covariance_estimation()
    fig7_spiked_covariance()
    print("All figures generated for chapter 14.")


if __name__ == "__main__":
    main()
