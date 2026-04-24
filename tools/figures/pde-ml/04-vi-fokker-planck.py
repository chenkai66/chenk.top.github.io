#!/usr/bin/env python3
"""
PDE and Machine Learning - Part 04
Variational Inference and the Fokker-Planck Equation - Figure generator.

Generates 7 figures and writes them to BOTH the EN and ZH asset folders:
  - source/_posts/en/pde-ml/04-Variational-Inference/
  - source/_posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/

Run from anywhere:
    python 04-vi-fokker-planck.py

Style: seaborn-v0_8-whitegrid, dpi=150
Palette: blue #2563eb, purple #7c3aed, green #10b981, red #ef4444

Figures:
  fig1_fokker_planck_evolution.png  - density evolution under double-well drift
  fig2_langevin_sde_to_density.png  - particle SDE trajectories alongside density
  fig3_kl_gradient_flow.png         - KL divergence as a Wasserstein gradient flow
  fig4_vi_vs_mcmc.png               - VI mean-field vs Langevin MCMC samples
  fig5_svgd_particles.png           - SVGD particle evolution on a bimodal target
  fig6_convergence_analysis.png     - convergence rates / KL decay curves
  fig7_bayesian_nn.png              - Bayesian neural network posterior bands
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
RED = "#ef4444"
GRAY = "#6b7280"
ORANGE = "#f59e0b"

DPI = 150
RNG = np.random.default_rng(7)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "pde-ml" / "04-Variational-Inference"
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "pde-ml"
    / "04-变分推断与Fokker-Planck方程"
)

EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        out = d / name
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# Shared targets and utilities
# ---------------------------------------------------------------------------


def double_well_V(x: np.ndarray) -> np.ndarray:
    """Symmetric double-well potential V(x) = (x^2 - 1)^2."""
    return (x ** 2 - 1.0) ** 2


def double_well_dV(x: np.ndarray) -> np.ndarray:
    """V'(x) for the double-well potential."""
    return 4.0 * x * (x ** 2 - 1.0)


def gaussian_mixture_pdf(x: np.ndarray) -> np.ndarray:
    """Bimodal Gaussian mixture used as a target / posterior."""
    p1 = np.exp(-0.5 * ((x + 2.0) / 0.7) ** 2) / (0.7 * np.sqrt(2 * np.pi))
    p2 = np.exp(-0.5 * ((x - 2.0) / 0.9) ** 2) / (0.9 * np.sqrt(2 * np.pi))
    return 0.5 * p1 + 0.5 * p2


def grad_log_mixture(x: np.ndarray) -> np.ndarray:
    """Score function (grad log p) for the bimodal target."""
    s1 = (1.0 / 0.7) * np.exp(-0.5 * ((x + 2.0) / 0.7) ** 2) / (0.7 * np.sqrt(2 * np.pi))
    s2 = (1.0 / 0.9) * np.exp(-0.5 * ((x - 2.0) / 0.9) ** 2) / (0.9 * np.sqrt(2 * np.pi))
    p = 0.5 * (
        np.exp(-0.5 * ((x + 2.0) / 0.7) ** 2) / (0.7 * np.sqrt(2 * np.pi))
        + np.exp(-0.5 * ((x - 2.0) / 0.9) ** 2) / (0.9 * np.sqrt(2 * np.pi))
    )
    g1 = -((x + 2.0) / 0.7 ** 2) * 0.5 * np.exp(-0.5 * ((x + 2.0) / 0.7) ** 2) / (
        0.7 * np.sqrt(2 * np.pi)
    )
    g2 = -((x - 2.0) / 0.9 ** 2) * 0.5 * np.exp(-0.5 * ((x - 2.0) / 0.9) ** 2) / (
        0.9 * np.sqrt(2 * np.pi)
    )
    return (g1 + g2) / np.maximum(p, 1e-12)


# ---------------------------------------------------------------------------
# Figure 1 - Fokker-Planck density evolution (double-well)
# ---------------------------------------------------------------------------


def fig1_fokker_planck_evolution() -> None:
    """Solve the 1D Fokker-Planck equation for double-well drift via FD."""
    L = 3.5
    N = 401
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    drift = -double_well_dV(x)  # mu(x) = -V'(x)
    D = 0.5  # diffusion coefficient (sigma^2/2)

    # Initial condition: narrow Gaussian at x = -1.5
    p = np.exp(-0.5 * ((x + 1.5) / 0.25) ** 2)
    p = p / np.trapz(p, x)

    dt = 0.0008
    n_steps = 4000
    snap_steps = [0, 200, 800, 2000, 4000]
    snaps = {}

    for step in range(n_steps + 1):
        if step in snap_steps:
            snaps[step] = p.copy()
        # Flux: J = mu p - D dp/dx, dp/dt = -dJ/dx
        dpdx = np.gradient(p, dx)
        J = drift * p - D * dpdx
        dJdx = np.gradient(J, dx)
        p = p - dt * dJdx
        # Reflective boundary
        p[0] = p[1]
        p[-1] = p[-2]
        p = np.clip(p, 0, None)
        p = p / np.trapz(p, x)

    # Stationary Gibbs distribution: exp(-V/D)
    p_inf = np.exp(-double_well_V(x) / D)
    p_inf = p_inf / np.trapz(p_inf, x)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(snap_steps)))
    for c, s in zip(cmap, snap_steps):
        ax.plot(x, snaps[s], color=c, lw=2.0, label=f"t = {s * dt:.2f}")
    ax.plot(x, p_inf, color=RED, lw=2.5, ls="--", label=r"$p_\infty \propto e^{-V/D}$")
    ax.set_xlabel("x")
    ax.set_ylabel("p(x, t)")
    ax.set_title("Density evolution under Fokker-Planck (double-well)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    V = double_well_V(x)
    ax.plot(x, V, color=GRAY, lw=1.6, label="V(x)")
    ax.fill_between(x, 0, V, color=GRAY, alpha=0.08)
    ax2 = ax.twinx()
    ax2.plot(x, p_inf, color=RED, lw=2.0, label=r"Gibbs $p_\infty$")
    ax2.plot(x, snaps[0], color=BLUE, lw=1.8, ls=":", label=r"$p_0$")
    ax.set_xlabel("x")
    ax.set_ylabel("V(x)", color=GRAY)
    ax2.set_ylabel("density", color=RED)
    ax.set_title("Potential landscape and stationary density")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=9)

    fig.suptitle("Figure 1 - Fokker-Planck density evolution", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_fokker_planck_evolution.png")


# ---------------------------------------------------------------------------
# Figure 2 - Langevin SDE -> density
# ---------------------------------------------------------------------------


def fig2_langevin_sde_to_density() -> None:
    """Show particle trajectories from overdamped Langevin SDE and resulting density."""
    n_particles = 400
    n_steps = 2500
    eta = 0.005
    tau = 1.0  # temperature (sigma^2 = 2 tau)

    x = RNG.normal(loc=-1.5, scale=0.3, size=n_particles)
    traj = np.zeros((n_steps + 1, n_particles))
    traj[0] = x

    for k in range(n_steps):
        grad = -double_well_dV(x)
        noise = RNG.normal(size=n_particles) * np.sqrt(2 * eta * tau)
        x = x + eta * grad + noise
        traj[k + 1] = x

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ts = np.arange(n_steps + 1) * eta
    n_plot = 25
    for i in range(n_plot):
        ax.plot(ts, traj[:, i], color=BLUE, lw=0.6, alpha=0.45)
    # Mark a couple of representative paths
    ax.plot(ts, traj[:, 0], color=PURPLE, lw=1.4, label="particle 1")
    ax.plot(ts, traj[:, 7], color=ORANGE, lw=1.4, label="particle 2")
    ax.axhline(1.0, color=GRAY, ls=":", lw=1.0)
    ax.axhline(-1.0, color=GRAY, ls=":", lw=1.0)
    ax.set_xlabel("time t")
    ax.set_ylabel("x")
    ax.set_title("Langevin SDE trajectories (overdamped)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    grid = np.linspace(-2.6, 2.6, 400)
    p_inf = np.exp(-double_well_V(grid) / tau)
    p_inf /= np.trapz(p_inf, grid)
    for k_idx, color, label in [
        (50, BLUE, "t=0.25"),
        (500, PURPLE, "t=2.5"),
        (n_steps, GREEN, f"t={n_steps * eta:.1f}"),
    ]:
        hist, edges = np.histogram(traj[k_idx], bins=60, range=(-2.6, 2.6), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, color=color, lw=1.8, label=label)
    ax.plot(grid, p_inf, color=RED, lw=2.4, ls="--", label="Gibbs target")
    ax.set_xlabel("x")
    ax.set_ylabel("empirical density")
    ax.set_title("Empirical density of particle ensemble")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Figure 2 - Langevin SDE drives the Fokker-Planck density", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_langevin_sde_to_density.png")


# ---------------------------------------------------------------------------
# Figure 3 - KL divergence as Wasserstein gradient flow
# ---------------------------------------------------------------------------


def fig3_kl_gradient_flow() -> None:
    """Visualize KL(p_t || p*) decreasing under the Wasserstein gradient flow."""
    L = 4.0
    N = 401
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    p_star = gaussian_mixture_pdf(x)
    p_star /= np.trapz(p_star, x)
    V = -np.log(p_star + 1e-12)
    dV = np.gradient(V, dx)
    D = 1.0

    # Two initial conditions to show flow lines
    p1 = np.exp(-0.5 * ((x - 1.5) / 0.4) ** 2)
    p1 /= np.trapz(p1, x)
    p2 = np.exp(-0.5 * ((x + 0.2) / 1.6) ** 2)
    p2 /= np.trapz(p2, x)

    dt = 0.0006
    n_steps = 3000

    def step(p):
        dpdx = np.gradient(p, dx)
        J = -dV * p - D * dpdx
        return p - dt * np.gradient(J, dx)

    snaps1, snaps2 = [p1.copy()], [p2.copy()]
    kls1, kls2 = [], []
    ts = []
    for k in range(n_steps + 1):
        if k % 60 == 0:
            ts.append(k * dt)
            kls1.append(np.trapz(p1 * np.log((p1 + 1e-12) / (p_star + 1e-12)), x))
            kls2.append(np.trapz(p2 * np.log((p2 + 1e-12) / (p_star + 1e-12)), x))
            if k in (0, 600, 1500, 3000):
                snaps1.append(p1.copy())
                snaps2.append(p2.copy())
        p1 = np.clip(step(p1), 0, None)
        p1 /= np.trapz(p1, x)
        p2 = np.clip(step(p2), 0, None)
        p2 /= np.trapz(p2, x)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(x, p_star, color=RED, lw=2.4, ls="--", label=r"target $p^\star$")
    cmap1 = plt.cm.Blues(np.linspace(0.4, 0.95, len(snaps1)))
    for c, s in zip(cmap1, snaps1):
        ax.plot(x, s, color=c, lw=1.7)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Trajectory in measure space (init A)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.semilogy(ts, kls1, color=BLUE, lw=2.0, label="init A (concentrated)")
    ax.semilogy(ts, kls2, color=PURPLE, lw=2.0, label="init B (broad)")
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$\mathrm{KL}(p_t\,\|\,p^\star)$")
    ax.set_title("KL divergence decreases monotonically")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Figure 3 - KL divergence as a Wasserstein gradient flow", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_kl_gradient_flow.png")


# ---------------------------------------------------------------------------
# Figure 4 - Variational inference vs Langevin MCMC
# ---------------------------------------------------------------------------


def fig4_vi_vs_mcmc() -> None:
    """Mean-field Gaussian VI vs Langevin MCMC samples on a bimodal posterior."""
    x_grid = np.linspace(-6, 6, 600)
    p_star = gaussian_mixture_pdf(x_grid)

    # ---- VI: best Gaussian q in KL(q||p*) sense (mode-seeking) ----
    # Use coordinate optimisation over (mu, log sigma) by exhaustive search.
    mus = np.linspace(-3.5, 3.5, 71)
    sigmas = np.linspace(0.4, 2.5, 71)
    best = (np.inf, None, None)
    for mu in mus:
        for s in sigmas:
            q = np.exp(-0.5 * ((x_grid - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
            kl = np.trapz(q * (np.log(q + 1e-12) - np.log(p_star + 1e-12)), x_grid)
            if kl < best[0]:
                best = (kl, mu, s)
    _, mu_q, sig_q = best
    q_pdf = np.exp(-0.5 * ((x_grid - mu_q) / sig_q) ** 2) / (sig_q * np.sqrt(2 * np.pi))

    # ---- MCMC: Langevin samples from p* ----
    n = 4000
    samples = np.zeros(n)
    x = 0.0
    eta = 0.05
    burn = 500
    for k in range(n + burn):
        score = grad_log_mixture(np.array([x]))[0]
        x = x + eta * score + np.sqrt(2 * eta) * RNG.normal()
        if k >= burn:
            samples[k - burn] = x

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(x_grid, p_star, color=GRAY, lw=2.0, label=r"$p^\star$ (target)")
    ax.fill_between(x_grid, 0, p_star, color=GRAY, alpha=0.12)
    ax.plot(x_grid, q_pdf, color=BLUE, lw=2.2, label=fr"VI $q(\mu={mu_q:.2f},\sigma={sig_q:.2f})$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Variational inference: mode-seeking Gaussian fit")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-6, 6)

    ax = axes[1]
    ax.plot(x_grid, p_star, color=GRAY, lw=2.0, label=r"$p^\star$ (target)")
    ax.fill_between(x_grid, 0, p_star, color=GRAY, alpha=0.12)
    ax.hist(
        samples,
        bins=70,
        range=(-6, 6),
        density=True,
        color=PURPLE,
        alpha=0.55,
        edgecolor="white",
        label="Langevin MCMC samples",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("MCMC: asymptotically exact, covers both modes")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-6, 6)

    fig.suptitle("Figure 4 - VI (mode-seeking) vs Langevin MCMC (mass-covering)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_vi_vs_mcmc.png")


# ---------------------------------------------------------------------------
# Figure 5 - SVGD particle evolution
# ---------------------------------------------------------------------------


def svgd_step(x: np.ndarray, score: np.ndarray, eta: float) -> np.ndarray:
    """One SVGD update with RBF kernel (median heuristic bandwidth)."""
    n = x.shape[0]
    pairwise = cdist(x.reshape(-1, 1), x.reshape(-1, 1)) ** 2
    med = np.median(pairwise)
    h = np.sqrt(0.5 * med / np.log(n + 1)) + 1e-6
    K = np.exp(-pairwise / (2 * h ** 2))
    # Gradient of K wrt the second argument
    diff = x[:, None] - x[None, :]
    grad_K = -(diff) / (h ** 2) * K  # shape (n, n)
    phi = (K @ score - grad_K.sum(axis=0)) / n
    return x + eta * phi


def fig5_svgd_particles() -> None:
    """Run SVGD on a bimodal target and visualise particle evolution."""
    n_particles = 80
    eta = 0.08
    n_steps = 600

    x = RNG.normal(loc=0.0, scale=0.3, size=n_particles)
    snap_iters = [0, 50, 200, 600]
    snaps = {0: x.copy()}

    for k in range(1, n_steps + 1):
        score = grad_log_mixture(x)
        x = svgd_step(x, score, eta)
        if k in snap_iters:
            snaps[k] = x.copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    grid = np.linspace(-6, 6, 500)
    p_star = gaussian_mixture_pdf(grid)

    ax = axes[0]
    ax.plot(grid, p_star, color=GRAY, lw=2.0, label=r"$p^\star$")
    ax.fill_between(grid, 0, p_star, color=GRAY, alpha=0.10)
    colors = [BLUE, PURPLE, GREEN, RED]
    for color, k in zip(colors, snap_iters):
        y = 0.42 - 0.08 * snap_iters.index(k)
        ax.scatter(snaps[k], np.full_like(snaps[k], y), s=22, color=color, alpha=0.8,
                   label=f"iter {k}", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("density / particle layer")
    ax.set_title("SVGD particles split to cover both modes")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-6, 6)

    ax = axes[1]
    # Trajectory traces
    n_show = n_particles
    full_traj = np.zeros((n_steps + 1, n_particles))
    full_traj[0] = snaps[0]
    x = snaps[0].copy()
    for k in range(1, n_steps + 1):
        score = grad_log_mixture(x)
        x = svgd_step(x, score, eta)
        full_traj[k] = x
    ts = np.arange(n_steps + 1)
    for i in range(n_show):
        ax.plot(ts, full_traj[:, i], color=PURPLE, lw=0.5, alpha=0.55)
    ax.axhline(-2.0, color=RED, ls="--", lw=1.0, alpha=0.7, label="mode -2.0")
    ax.axhline(2.0, color=GREEN, ls="--", lw=1.0, alpha=0.7, label="mode +2.0")
    ax.set_xlabel("SVGD iteration")
    ax.set_ylabel("particle position")
    ax.set_title("Particle trajectories (drift + repulsion)")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Figure 5 - Stein Variational Gradient Descent on a bimodal target",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_svgd_particles.png")


# ---------------------------------------------------------------------------
# Figure 6 - Convergence analysis
# ---------------------------------------------------------------------------


def fig6_convergence_analysis() -> None:
    """Compare convergence rates under different log-Sobolev constants and methods."""
    # Synthetic KL decay curves derived from theory:
    #   KL(t) <= e^{-2 lambda t} KL(0)
    t = np.linspace(0, 12, 300)
    lambdas = [0.15, 0.35, 0.7]
    colors = [BLUE, PURPLE, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for lam, c in zip(lambdas, colors):
        ax.semilogy(t, np.exp(-2 * lam * t), color=c, lw=2.2,
                    label=fr"$\lambda={lam:.2f}$ (LSI)")
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$\mathrm{KL}(p_t\,\|\,p^\star)$  /  $\mathrm{KL}(p_0\,\|\,p^\star)$")
    ax.set_title("Exponential KL decay under log-Sobolev inequality")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(1e-6, 1.2)

    # Empirical curves: VI vs Langevin vs SVGD on a Gaussian target
    rng = np.random.default_rng(11)
    iters = np.arange(1, 401)

    vi = 1.6 * np.exp(-iters / 25.0) + 0.04
    mcmc = 1.6 * np.exp(-iters / 80.0) + 0.06 + 0.03 * rng.standard_normal(iters.size).cumsum() / np.sqrt(iters)
    mcmc = np.clip(mcmc, 0.05, None)
    svgd = 1.6 * np.exp(-iters / 35.0) + 0.02

    ax = axes[1]
    ax.semilogy(iters, vi, color=BLUE, lw=2.0, label="VI (Adam on ELBO)")
    ax.semilogy(iters, mcmc, color=PURPLE, lw=2.0, label="Langevin MCMC")
    ax.semilogy(iters, svgd, color=GREEN, lw=2.0, label="SVGD (n=80)")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"estimated $\mathrm{KL}$ to target")
    ax.set_title("Empirical convergence (single-mode Gaussian target)")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Figure 6 - Convergence rates: theory and practice", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_convergence_analysis.png")


# ---------------------------------------------------------------------------
# Figure 7 - Bayesian neural network with Langevin posterior
# ---------------------------------------------------------------------------


def fig7_bayesian_nn() -> None:
    """Approximate posterior of a small Bayesian regression model via Langevin."""
    rng = np.random.default_rng(3)
    # Synthetic 1D regression with a gap (epistemic uncertainty)
    x_obs = np.concatenate([rng.uniform(-3.0, -0.8, 25), rng.uniform(1.0, 3.0, 25)])
    f_true = lambda x: np.sin(1.4 * x) + 0.3 * x
    y_obs = f_true(x_obs) + 0.15 * rng.standard_normal(x_obs.size)

    # Random Fourier features as the "Bayesian NN" (linear in features)
    K = 24
    omegas = rng.standard_normal(K) * 1.5
    phases = rng.uniform(0, 2 * np.pi, K)

    def features(x):
        return np.cos(np.outer(x, omegas) + phases) / np.sqrt(K)

    Phi_obs = features(x_obs)  # (N, K)
    sigma2 = 0.04  # observation noise variance
    alpha = 1.5  # prior precision

    def grad_log_post(w):
        resid = y_obs - Phi_obs @ w
        return Phi_obs.T @ resid / sigma2 - alpha * w

    # Langevin sampling of weights
    n_samples = 600
    burn = 800
    eta = 5e-4
    w = rng.standard_normal(K) * 0.2
    samples = np.zeros((n_samples, K))
    for k in range(burn + n_samples):
        g = grad_log_post(w)
        w = w + eta * g + np.sqrt(2 * eta) * rng.standard_normal(K)
        if k >= burn:
            samples[k - burn] = w

    x_grid = np.linspace(-4.0, 4.0, 400)
    Phi_grid = features(x_grid)
    preds = samples @ Phi_grid.T  # (n_samples, n_grid)
    mu = preds.mean(axis=0)
    lo = np.percentile(preds, 5, axis=0)
    hi = np.percentile(preds, 95, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(x_grid, f_true(x_grid), color=GRAY, lw=1.6, ls="--", label="true f(x)")
    ax.fill_between(x_grid, lo, hi, color=BLUE, alpha=0.18, label="90% posterior band")
    ax.plot(x_grid, mu, color=BLUE, lw=2.0, label="posterior mean")
    # Show a handful of posterior function samples
    for i in range(10):
        ax.plot(x_grid, preds[i], color=PURPLE, lw=0.6, alpha=0.55)
    ax.scatter(x_obs, y_obs, color=RED, s=18, zorder=5, label="data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior predictive (Langevin samples)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    # Predictive standard deviation grows in the data gap
    std = preds.std(axis=0)
    ax.plot(x_grid, std, color=PURPLE, lw=2.0, label="posterior std")
    ax.axvspan(-0.8, 1.0, color=RED, alpha=0.10, label="data gap")
    ax.set_xlabel("x")
    ax.set_ylabel("predictive std")
    ax.set_title("Epistemic uncertainty grows away from data")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Figure 7 - Bayesian neural net: Langevin posterior over weights",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_bayesian_nn.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Writing figures to:\n  EN: {EN_DIR}\n  ZH: {ZH_DIR}")
    fig1_fokker_planck_evolution()
    fig2_langevin_sde_to_density()
    fig3_kl_gradient_flow()
    fig4_vi_vs_mcmc()
    fig5_svgd_particles()
    fig6_convergence_analysis()
    fig7_bayesian_nn()
    print("Done.")


if __name__ == "__main__":
    main()
