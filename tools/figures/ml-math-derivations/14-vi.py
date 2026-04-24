"""
Figure generation script for ML Math Derivations Part 14:
Variational Inference and Variational EM.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates a single idea so the math becomes geometrically obvious.

Figures:
    fig1_elbo_decomposition     log p(x) = ELBO + KL(q || p): a single
                                bar split into the two non-negative
                                pieces, with KL closing as q approaches p.
    fig2_mean_field             A correlated 2D Gaussian posterior and
                                its diagonal mean-field surrogate, side
                                by side, showing variance under-estimation.
    fig3_q_approximates_post    A bimodal target p(theta | D) with the
                                best-in-family Gaussian q(theta) found
                                by minimizing reverse KL: mode-seeking.
    fig4_vi_vs_mcmc             Wall-clock vs error: VI converges fast
                                to a biased plateau, MCMC converges
                                slowly toward zero error.
    fig5_cavi_iterations        Coordinate ascent VI iterations on the
                                correlated Gaussian: ELBO climbs,
                                contours of q tighten onto the target.
    fig6_kl_asymmetry           Reverse vs forward KL on a mixture of
                                Gaussians: zero-forcing (one mode) vs
                                moment-matching (covers both).
    fig7_lda_topics             VI for LDA: per-document topic
                                proportions and per-topic word
                                distributions as stacked bars / heatmap.

Usage:
    python3 scripts/figures/ml-math-derivations/14-vi.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "14-Variational-Inference-and-Variational-EM"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "14-变分推断与变分EM"
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
def gaussian_2d(x, y, mu, cov):
    """Bivariate Gaussian density on a meshgrid."""
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    diff = np.stack([x - mu[0], y - mu[1]], axis=-1)
    quad = np.einsum("...i,ij,...j->...", diff, inv, diff)
    return np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))


def kl_gauss(mu_q, var_q, mu_p, var_p):
    """KL(N(mu_q,var_q) || N(mu_p,var_p)) in nats."""
    return 0.5 * (np.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)


# ---------------------------------------------------------------------------
# Figure 1: ELBO decomposition  log p(x) = ELBO + KL
# ---------------------------------------------------------------------------
def fig1_elbo_decomposition() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))

    # --- Left: stacked bar showing log p(x) = ELBO + KL across iterations ---
    ax = axes[0]
    iters = np.arange(0, 11)
    log_evidence = 4.2  # constant
    kl = 3.2 * np.exp(-0.45 * iters) + 0.05
    elbo = log_evidence - kl

    width = 0.7
    ax.bar(iters, elbo, width, color=C_BLUE, label=r"ELBO $\;\mathcal{L}(q)$",
           edgecolor="white", linewidth=0.8)
    ax.bar(iters, kl, width, bottom=elbo, color=C_AMBER,
           label=r"$\mathrm{KL}(q\|p)$", edgecolor="white", linewidth=0.8)
    ax.axhline(log_evidence, color=C_DARK, lw=1.3, ls="--",
               label=r"$\log p(\mathbf{x})$ (constant)")

    ax.set_xlabel("VI iteration")
    ax.set_ylabel("nats")
    ax.set_title("Maximizing ELBO  $\\Leftrightarrow$  Minimizing KL",
                 color=C_DARK, fontsize=12)
    ax.set_xticks(iters)
    ax.set_ylim(0, log_evidence * 1.18)
    ax.legend(loc="lower right", framealpha=0.95)

    # Annotate gap closing
    ax.annotate("KL gap closes",
                xy=(9, log_evidence - 0.03),
                xytext=(5.2, log_evidence + 0.5),
                fontsize=10, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    # --- Right: schematic of identity as a horizontal bar ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Top bar: log p(x)
    ax.add_patch(Rectangle((0.5, 2.7), 9.0, 0.7,
                           facecolor=C_LIGHT, edgecolor=C_DARK, lw=1.2))
    ax.text(5.0, 3.05, r"$\log p(\mathbf{x})$  (evidence, fixed)",
            ha="center", va="center", fontsize=12, color=C_DARK)

    # Bottom bar: ELBO + KL
    elbo_w = 6.2
    ax.add_patch(Rectangle((0.5, 1.2), elbo_w, 0.7,
                           facecolor=C_BLUE, edgecolor="white", lw=1.0))
    ax.add_patch(Rectangle((0.5 + elbo_w, 1.2), 9.0 - elbo_w, 0.7,
                           facecolor=C_AMBER, edgecolor="white", lw=1.0))
    ax.text(0.5 + elbo_w / 2, 1.55, r"ELBO  $\mathcal{L}(q)$",
            ha="center", va="center", color="white", fontsize=12)
    ax.text(0.5 + elbo_w + (9.0 - elbo_w) / 2, 1.55,
            r"$\mathrm{KL}(q\|p)$",
            ha="center", va="center", color="white", fontsize=11)

    # Equality braces
    ax.annotate("", xy=(0.5, 2.55), xytext=(0.5, 1.95),
                arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=1))
    ax.annotate("", xy=(9.5, 2.55), xytext=(9.5, 1.95),
                arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=1))
    ax.text(5.0, 0.55,
            r"$\log p(\mathbf{x}) \;=\; \mathcal{L}(q) \;+\; "
            r"\mathrm{KL}\!\left(q(\mathbf{z})\,\|\,p(\mathbf{z}\!\mid\!\mathbf{x})\right)$",
            ha="center", va="center", fontsize=13, color=C_DARK)
    ax.text(5.0, 0.05,
            r"$\mathrm{KL}\geq 0\;\Rightarrow\;$ ELBO is a lower bound on the evidence",
            ha="center", va="center", fontsize=10, color=C_GRAY)
    ax.set_title("The variational identity",
                 color=C_DARK, fontsize=12)

    fig.suptitle("Figure 1 — ELBO decomposition of the log evidence",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_elbo_decomposition")


# ---------------------------------------------------------------------------
# Figure 2: Mean-field approximation
# ---------------------------------------------------------------------------
def fig2_mean_field() -> None:
    grid = np.linspace(-3.5, 3.5, 240)
    X, Y = np.meshgrid(grid, grid)

    # True correlated posterior
    rho = 0.85
    cov_true = np.array([[1.0, rho], [rho, 1.0]])
    p = gaussian_2d(X, Y, mu=[0, 0], cov=cov_true)

    # Mean-field: diagonal Gaussian matching the marginals
    var_x = cov_true[0, 0]
    var_y = cov_true[1, 1]
    cov_mf = np.array([[var_x, 0.0], [0.0, var_y]])
    q_mf = gaussian_2d(X, Y, mu=[0, 0], cov=cov_mf)

    # Reverse-KL optimal mean-field (variance shrinks by 1 - rho^2)
    var_opt = 1 - rho ** 2
    cov_rkl = np.array([[var_opt, 0.0], [0.0, var_opt]])
    q_rkl = gaussian_2d(X, Y, mu=[0, 0], cov=cov_rkl)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    # Panel A: true posterior
    ax = axes[0]
    ax.contourf(X, Y, p, levels=14, cmap="Blues", alpha=0.85)
    ax.contour(X, Y, p, levels=6, colors=C_BLUE, linewidths=1.0, alpha=0.9)
    ax.set_title(r"True posterior $p(\mathbf{z}\mid \mathbf{x})$"
                 + f"\n(correlation $\\rho={rho}$)",
                 color=C_DARK, fontsize=11)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
    ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)

    # Panel B: mean-field marginal-matching
    ax = axes[1]
    ax.contourf(X, Y, q_mf, levels=14, cmap="Purples", alpha=0.6)
    ax.contour(X, Y, p, levels=6, colors=C_BLUE, linewidths=1.0, alpha=0.6,
               linestyles="--")
    ax.contour(X, Y, q_mf, levels=6, colors=C_PURPLE, linewidths=1.2)
    ax.set_title(r"Mean-field $q=q_1(z_1)q_2(z_2)$"
                 + "\n(loses correlation; matches marginals)",
                 color=C_DARK, fontsize=11)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
    ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)

    # Panel C: reverse-KL optimal mean-field (under-disperses)
    ax = axes[2]
    ax.contourf(X, Y, q_rkl, levels=14, cmap="Greens", alpha=0.6)
    ax.contour(X, Y, p, levels=6, colors=C_BLUE, linewidths=1.0, alpha=0.6,
               linestyles="--")
    ax.contour(X, Y, q_rkl, levels=6, colors=C_GREEN, linewidths=1.2)
    ax.set_title(r"Reverse-KL optimum $q^\star$"
                 + f"\n(variance $=1-\\rho^2={var_opt:.2f}$, under-disperses)",
                 color=C_DARK, fontsize=11)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
    ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)

    fig.suptitle("Figure 2 — Mean-field assumption $q(\\mathbf{z})=\\prod_j q_j(z_j)$ "
                 "factorizes the joint",
                 fontsize=13, color=C_DARK, y=1.04)
    fig.tight_layout()
    _save(fig, "fig2_mean_field")


# ---------------------------------------------------------------------------
# Figure 3: q(theta) approximating posterior p(theta | D)
# ---------------------------------------------------------------------------
def fig3_q_approximates_post() -> None:
    theta = np.linspace(-5, 7, 1200)

    # Bimodal target posterior (mixture of two Gaussians, unequal weight)
    p1 = np.exp(-0.5 * ((theta + 1.5) / 0.7) ** 2) / (0.7 * np.sqrt(2 * np.pi))
    p2 = np.exp(-0.5 * ((theta - 3.0) / 0.9) ** 2) / (0.9 * np.sqrt(2 * np.pi))
    p = 0.55 * p1 + 0.45 * p2

    # Two candidate Gaussians q
    q_left = np.exp(-0.5 * ((theta + 1.5) / 0.75) ** 2) / (0.75 * np.sqrt(2 * np.pi))
    q_wide = np.exp(-0.5 * ((theta - 0.6) / 2.4) ** 2) / (2.4 * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.fill_between(theta, p, color=C_BLUE, alpha=0.18,
                    label=r"target $p(\theta\mid \mathcal{D})$")
    ax.plot(theta, p, color=C_BLUE, lw=2.4)

    ax.plot(theta, q_left, color=C_PURPLE, lw=2.4,
            label=r"$q^{\star}_{\mathrm{KL}(q\|p)}$  (reverse KL: mode-seeking)")
    ax.plot(theta, q_wide, color=C_AMBER, lw=2.4, ls="--",
            label=r"$q^{\star}_{\mathrm{KL}(p\|q)}$  (forward KL: moment-matching)")

    # Annotate the modes
    ax.annotate("locked onto\nleft mode", xy=(-1.5, q_left.max()),
                xytext=(-3.7, q_left.max() * 0.85),
                fontsize=10, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1))
    ax.annotate("covers both modes\n(places mass between)",
                xy=(0.6, q_wide[np.argmin(np.abs(theta - 0.6))]),
                xytext=(2.6, 0.42),
                fontsize=10, color=C_AMBER,
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1))

    ax.set_xlabel(r"parameter $\theta$")
    ax.set_ylabel("density")
    ax.set_title("Figure 3 — Variational family $q(\\theta)$ approximating an intractable "
                 "posterior $p(\\theta\\mid \\mathcal{D})$",
                 color=C_DARK, fontsize=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xlim(-5, 7)
    ax.set_ylim(0, 0.62)

    fig.tight_layout()
    _save(fig, "fig3_q_approximates_post")


# ---------------------------------------------------------------------------
# Figure 4: VI vs MCMC trade-off
# ---------------------------------------------------------------------------
def fig4_vi_vs_mcmc() -> None:
    rng = np.random.default_rng(0)
    t = np.logspace(-1.2, 2.5, 200)  # wall-clock seconds

    # VI: exponential decay to a biased plateau
    vi_plateau = 0.18
    vi_err = vi_plateau + 1.6 * np.exp(-1.8 * t)

    # MCMC: slow ~ 1/sqrt(t) decay toward zero (after burn-in delay)
    burn_in = 1.0
    mcmc_err = np.where(
        t < burn_in,
        2.2,
        0.95 / np.sqrt(np.clip(t - burn_in + 0.3, 0.1, None)),
    )
    mcmc_err = mcmc_err + 0.04 * rng.standard_normal(len(t))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(t, vi_err, color=C_BLUE, lw=2.6, label="Variational inference")
    ax.plot(t, mcmc_err, color=C_AMBER, lw=2.0, label="MCMC (e.g. HMC / Gibbs)")
    ax.axhline(vi_plateau, color=C_BLUE, lw=1.0, ls="--", alpha=0.6)
    ax.text(t[10], vi_plateau + 0.04,
            "VI bias floor (mean-field gap)",
            color=C_BLUE, fontsize=10)

    ax.annotate("fast convergence\n(deterministic optimization)",
                xy=(0.7, vi_err[np.argmin(np.abs(t - 0.7))]),
                xytext=(2.5, 1.5),
                fontsize=10, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1))
    ax.annotate("slow but unbiased\n(asymptotically exact)",
                xy=(80, mcmc_err[np.argmin(np.abs(t - 80))]),
                xytext=(8, 1.7),
                fontsize=10, color=C_AMBER,
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1))

    ax.set_xscale("log")
    ax.set_xlabel("wall-clock time (s, log scale)")
    ax.set_ylabel(r"posterior approximation error  $\|\hat{p} - p\|$")
    ax.set_title("Figure 4 — VI vs MCMC: speed–accuracy trade-off",
                 color=C_DARK, fontsize=12)
    ax.set_ylim(0, 2.5)
    ax.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout()
    _save(fig, "fig4_vi_vs_mcmc")


# ---------------------------------------------------------------------------
# Figure 5: CAVI iterations
# ---------------------------------------------------------------------------
def fig5_cavi_iterations() -> None:
    """
    CAVI on a 2D Gaussian target with diagonal q.  We iterate the closed-form
    coordinate updates and visualize how q's covariance ellipse tightens
    onto the marginals.  The right panel shows ELBO climbing monotonically.
    """
    # Target: zero-mean Gaussian with strong correlation
    rho = 0.8
    cov_true = np.array([[1.0, rho], [rho, 1.0]])
    prec = np.linalg.inv(cov_true)  # = [[a,b],[b,a]]
    a, b = prec[0, 0], prec[0, 1]

    grid = np.linspace(-3.2, 3.2, 240)
    X, Y = np.meshgrid(grid, grid)
    p = gaussian_2d(X, Y, mu=[0, 0], cov=cov_true)

    # Initialize q = N(m, diag(s2))
    m = np.array([2.0, -1.5])
    s2 = np.array([2.5, 2.5])

    fig = plt.figure(figsize=(13.0, 4.6))
    ax_left = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)

    ax_left.contour(X, Y, p, levels=6, colors=C_BLUE, linewidths=1.0,
                    alpha=0.7, linestyles="--")
    ax_left.set_title("CAVI iterates of the diagonal Gaussian $q$",
                      color=C_DARK, fontsize=11)
    ax_left.set_xlabel(r"$z_1$"); ax_left.set_ylabel(r"$z_2$")
    ax_left.set_aspect("equal"); ax_left.set_xlim(-3.5, 3.5)
    ax_left.set_ylim(-3.5, 3.5)

    elbo_hist = []
    n_iter = 8
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, n_iter + 1))

    def draw_ellipse(ax, m, s2, color, lw=1.5, alpha=0.9):
        th = np.linspace(0, 2 * np.pi, 200)
        ex = m[0] + 2 * np.sqrt(s2[0]) * np.cos(th)
        ey = m[1] + 2 * np.sqrt(s2[1]) * np.sin(th)
        ax.plot(ex, ey, color=color, lw=lw, alpha=alpha)
        ax.scatter(m[0], m[1], color=color, s=24, zorder=5,
                   edgecolor="white", linewidth=0.8)

    def elbo(m, s2):
        # ELBO for diagonal q under zero-mean target Gaussian (closed form,
        # ignoring constants).
        e_quad = a * (m[0] ** 2 + s2[0]) + a * (m[1] ** 2 + s2[1]) \
                 + 2 * b * (m[0] * m[1])
        ent = 0.5 * (np.log(s2[0]) + np.log(s2[1]))
        return -0.5 * e_quad + ent

    draw_ellipse(ax_left, m.copy(), s2.copy(), cmap[0], lw=1.5, alpha=0.6)
    elbo_hist.append(elbo(m, s2))

    for k in range(n_iter):
        # Closed-form CAVI updates for Gaussian with precision = [[a,b],[b,a]]
        # q1: mean = -b/a * m2, var = 1/a
        m_new = m.copy()
        s2_new = s2.copy()
        m_new[0] = -b / a * m[1]
        s2_new[0] = 1.0 / a
        # q2: mean = -b/a * m1, var = 1/a   (use updated m1)
        m_new[1] = -b / a * m_new[0]
        s2_new[1] = 1.0 / a

        m, s2 = m_new, s2_new
        elbo_hist.append(elbo(m, s2))
        draw_ellipse(ax_left, m.copy(), s2.copy(), cmap[k + 1],
                     lw=1.6 + 0.05 * k, alpha=0.92)

    # Final equilibrium ellipse highlighted
    ax_left.text(2.2, 2.7, "iter 0", color=cmap[0], fontsize=9)
    ax_left.text(0.05, 0.9, f"iter {n_iter}", color=cmap[-1], fontsize=10,
                 fontweight="bold")

    # Right: ELBO trajectory
    ax_right.plot(range(len(elbo_hist)), elbo_hist,
                  marker="o", color=C_GREEN, lw=2.2, markersize=6,
                  markerfacecolor="white", markeredgewidth=1.6)
    ax_right.set_xlabel("CAVI iteration")
    ax_right.set_ylabel(r"ELBO $\mathcal{L}(q)$  (up to constant)")
    ax_right.set_title("ELBO climbs monotonically",
                       color=C_DARK, fontsize=11)
    ax_right.axhline(elbo_hist[-1], color=C_GRAY, lw=1, ls=":")
    ax_right.text(len(elbo_hist) - 1, elbo_hist[-1] + 0.08,
                  "stationary point\nof CAVI updates",
                  ha="right", color=C_DARK, fontsize=9)

    fig.suptitle("Figure 5 — Coordinate Ascent Variational Inference (CAVI)",
                 fontsize=13, color=C_DARK, y=1.03)
    fig.tight_layout()
    _save(fig, "fig5_cavi_iterations")


# ---------------------------------------------------------------------------
# Figure 6: KL asymmetry — reverse vs forward KL
# ---------------------------------------------------------------------------
def fig6_kl_asymmetry() -> None:
    """
    Show the qualitatively different behaviour of reverse vs forward KL
    when q is a single Gaussian and p is a mixture of two Gaussians.
    """
    x = np.linspace(-7, 9, 1500)
    p1 = np.exp(-0.5 * ((x + 2.5) / 0.8) ** 2) / (0.8 * np.sqrt(2 * np.pi))
    p2 = np.exp(-0.5 * ((x - 3.5) / 0.9) ** 2) / (0.9 * np.sqrt(2 * np.pi))
    p = 0.5 * p1 + 0.5 * p2

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # --- Left: reverse KL  KL(q || p) ---
    ax = axes[0]
    ax.fill_between(x, p, color=C_BLUE, alpha=0.18, label=r"$p$")
    ax.plot(x, p, color=C_BLUE, lw=2.0)
    # Two reverse-KL local optima: pick one mode each
    q_a = np.exp(-0.5 * ((x + 2.5) / 0.85) ** 2) / (0.85 * np.sqrt(2 * np.pi))
    q_b = np.exp(-0.5 * ((x - 3.5) / 0.95) ** 2) / (0.95 * np.sqrt(2 * np.pi))
    ax.plot(x, q_a, color=C_PURPLE, lw=2.2,
            label=r"$q^\star$ option A")
    ax.plot(x, q_b, color=C_GREEN, lw=2.2, ls="--",
            label=r"$q^\star$ option B")
    ax.set_title(r"Reverse KL  $\;\mathrm{KL}(q\|p)$  "
                 r"— zero-forcing, mode-seeking",
                 color=C_DARK, fontsize=11)
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("density")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 0.55)
    ax.text(-1.1, 0.46, "must avoid\nregions where $p\\!\\approx\\!0$",
            color=C_DARK, fontsize=9, ha="center")

    # --- Right: forward KL  KL(p || q)  → moment matching ---
    ax = axes[1]
    ax.fill_between(x, p, color=C_BLUE, alpha=0.18, label=r"$p$")
    ax.plot(x, p, color=C_BLUE, lw=2.0)
    # Moment-matched Gaussian: same mean and variance as p
    mean_p = 0.5 * (-2.5) + 0.5 * 3.5
    var_p = (0.5 * (0.8 ** 2 + (-2.5 - mean_p) ** 2)
             + 0.5 * (0.9 ** 2 + (3.5 - mean_p) ** 2))
    sd = np.sqrt(var_p)
    q_mm = np.exp(-0.5 * ((x - mean_p) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ax.plot(x, q_mm, color=C_AMBER, lw=2.4,
            label=r"$q^\star$ (moment-matched)")
    ax.set_title(r"Forward KL  $\;\mathrm{KL}(p\|q)$  "
                 r"— mass-covering",
                 color=C_DARK, fontsize=11)
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("density")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 0.55)
    ax.text(0.5, 0.05, "places density\nbetween modes",
            color=C_DARK, fontsize=9, ha="center")

    fig.suptitle("Figure 6 — KL asymmetry shapes the variational solution",
                 fontsize=13, color=C_DARK, y=1.04)
    fig.tight_layout()
    _save(fig, "fig6_kl_asymmetry")


# ---------------------------------------------------------------------------
# Figure 7: VI for LDA topic model
# ---------------------------------------------------------------------------
def fig7_lda_topics() -> None:
    """
    Schematic of variational inference for LDA:
      - per-document topic proportions  theta_d  (Dirichlet posterior)
      - per-topic word distributions     beta_k   (heat map)
    """
    rng = np.random.default_rng(42)
    K = 4  # topics
    D = 8  # docs
    V = 12  # vocab terms

    topic_names = ["ML", "Bio", "Finance", "Sports"]
    topic_colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

    # Construct semi-realistic theta (each doc favors one or two topics)
    theta = np.zeros((D, K))
    for d in range(D):
        primary = d % K
        secondary = (primary + 1 + d // K) % K
        a = np.full(K, 0.4)
        a[primary] = 4.0
        a[secondary] = 1.5
        theta[d] = rng.dirichlet(a)

    # beta: each topic puts mass on a few "characteristic" words
    beta = np.zeros((K, V))
    rng2 = np.random.default_rng(7)
    for k in range(K):
        a = np.full(V, 0.4)
        a[(3 * k) % V] = 5.0
        a[(3 * k + 1) % V] = 3.0
        a[(3 * k + 2) % V] = 2.0
        beta[k] = rng2.dirichlet(a)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8),
                             gridspec_kw={"width_ratios": [1.0, 1.1]})

    # --- Left: stacked bar of per-document topic proportions ---
    ax = axes[0]
    bottoms = np.zeros(D)
    doc_idx = np.arange(D)
    for k in range(K):
        ax.bar(doc_idx, theta[:, k], bottom=bottoms,
               color=topic_colors[k], edgecolor="white", linewidth=0.8,
               label=f"topic {k+1}: {topic_names[k]}")
        bottoms += theta[:, k]
    ax.set_xticks(doc_idx)
    ax.set_xticklabels([f"d{i+1}" for i in doc_idx])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("documents")
    ax.set_ylabel(r"$\mathbb{E}_q[\theta_{d,k}]$  (topic proportion)")
    ax.set_title(r"Per-document topic mix  $q(\theta_d)=\mathrm{Dir}(\gamma_d)$",
                 color=C_DARK, fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2, framealpha=0.95)

    # --- Right: heatmap of per-topic word distributions beta_k ---
    ax = axes[1]
    im = ax.imshow(beta, aspect="auto", cmap="magma_r")
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"topic {k+1}\n{topic_names[k]}" for k in range(K)],
                       fontsize=9)
    ax.set_xticks(range(V))
    ax.set_xticklabels([f"w{i+1}" for i in range(V)], fontsize=8)
    ax.set_xlabel("vocabulary terms")
    ax.set_title(r"Per-topic word distribution  $q(\beta_k)=\mathrm{Dir}(\lambda_k)$",
                 color=C_DARK, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"$\mathbb{E}_q[\beta_{k,v}]$", fontsize=9)

    fig.suptitle("Figure 7 — Variational inference for Latent Dirichlet Allocation",
                 fontsize=13, color=C_DARK, y=1.04)
    fig.tight_layout()
    _save(fig, "fig7_lda_topics")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 14 figures (Variational Inference & Variational EM)...")
    fig1_elbo_decomposition()
    fig2_mean_field()
    fig3_q_approximates_post()
    fig4_vi_vs_mcmc()
    fig5_cavi_iterations()
    fig6_kl_asymmetry()
    fig7_lda_topics()
    print("Done.")


if __name__ == "__main__":
    main()
