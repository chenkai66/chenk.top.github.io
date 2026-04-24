#!/usr/bin/env python3
"""
PDE + ML Chapter 07 - Diffusion Models and Score Matching
Figure generation script.

Generates 7 figures and writes them to BOTH the EN and ZH asset folders:
  - source/_posts/en/pde-ml/07-Diffusion-Models/
  - source/_posts/zh/pde-ml/07-扩散模型与Score-Matching/

Run from anywhere:
    python 07-diffusion.py

Style: seaborn-v0_8-whitegrid, dpi=150
Palette: blue #2563eb, purple #7c3aed, green #10b981, red #ef4444
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.titleweight": "bold",
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
RED = "#ef4444"
GREY = "#6b7280"
LIGHT = "#e5e7eb"

REPO = Path(__file__).resolve().parents[3]
EN_DIR = REPO / "source/_posts/en/pde-ml/07-Diffusion-Models"
ZH_DIR = REPO / "source/_posts/zh/pde-ml/07-扩散模型与Score-Matching"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bimodal_pdf(x: np.ndarray, mu: float = 2.0, sigma: float = 0.5) -> np.ndarray:
    return 0.5 * norm.pdf(x, -mu, sigma) + 0.5 * norm.pdf(x, mu, sigma)


def vp_marginal(x: np.ndarray, x0: float, sigma0: float, t: float, beta: float = 1.0) -> np.ndarray:
    """Marginal of OU/VP SDE: dX = -0.5*beta*X dt + sqrt(beta) dW.

    Closed form: X_t | X_0 = N(x0 * a, 1 - a^2 + a^2 sigma0^2),
    where a = exp(-0.5 beta t).  (Using std normal stationary.)
    """
    a = np.exp(-0.5 * beta * t)
    mean = a * x0
    var = (1 - a ** 2) + a ** 2 * sigma0 ** 2
    return norm.pdf(x, mean, np.sqrt(var))


def vp_marginal_bimodal(x: np.ndarray, t: float, mu: float = 2.0, sigma0: float = 0.5, beta: float = 1.0) -> np.ndarray:
    return 0.5 * vp_marginal(x, -mu, sigma0, t, beta) + 0.5 * vp_marginal(x, mu, sigma0, t, beta)


# ---------------------------------------------------------------------------
# Figure 1: Forward diffusion process (image-like grid -> noise)
# ---------------------------------------------------------------------------

def fig1_forward_diffusion() -> None:
    """Forward process: VP-SDE noising of a structured 2D pattern across timesteps."""
    rng = np.random.default_rng(0)
    n = 64

    # Structured 'image': two Gaussian blobs + ring, normalised to roughly unit variance.
    yy, xx = np.mgrid[-2:2:n*1j, -2:2:n*1j]
    base = (
        1.6 * np.exp(-((xx + 0.7) ** 2 + (yy + 0.4) ** 2) / 0.18)
        + 1.4 * np.exp(-((xx - 0.7) ** 2 + (yy - 0.4) ** 2) / 0.22)
        - 0.9 * np.exp(-((xx ** 2 + yy ** 2 - 1.4) ** 2) / 0.05)
    )
    base = (base - base.mean()) / base.std()

    timesteps = [0.0, 0.15, 0.4, 0.8, 1.5, 3.0]
    fig, axes = plt.subplots(2, len(timesteps), figsize=(13.2, 5.0))

    # Top row: 'image' (VP-SDE noised samples)
    for ax, t in zip(axes[0], timesteps):
        a = np.exp(-0.5 * 1.0 * t)
        var = 1 - a ** 2
        eps = rng.standard_normal(base.shape)
        img = a * base + np.sqrt(var) * eps
        ax.imshow(img, cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=[-2, 2, -2, 2])
        ax.set_title(f"$t={t:.2f}$" if t > 0 else "$t=0$ (data)")
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom row: matching 1D marginal density evolution (VP-SDE applied to bimodal)
    x = np.linspace(-5, 5, 400)
    for ax, t in zip(axes[1], timesteps):
        p = vp_marginal_bimodal(x, t)
        ax.fill_between(x, p, color=BLUE, alpha=0.25)
        ax.plot(x, p, color=BLUE, lw=2.0, label="$p_t$")
        ax.plot(x, norm.pdf(x, 0, 1), color=RED, ls="--", lw=1.2, label="$\\mathcal{N}(0,1)$")
        ax.set_xlim(-5, 5); ax.set_ylim(0, 0.55)
        ax.set_xticks([-4, 0, 4]); ax.set_yticks([])
        if t == timesteps[0]:
            ax.set_ylabel("density")
            ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "Forward diffusion (VP-SDE):  $d\\mathbf{X}_t = -\\frac{1}{2}\\beta\\mathbf{X}_t\\,dt + \\sqrt{\\beta}\\,d\\mathbf{B}_t$"
        "    -    structure decays into isotropic Gaussian noise",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig1_forward_diffusion")


# ---------------------------------------------------------------------------
# Figure 2: Reverse diffusion (noise -> image) - score-guided denoising
# ---------------------------------------------------------------------------

def fig2_reverse_diffusion() -> None:
    rng = np.random.default_rng(1)

    # Same "image" target as fig1 to keep visual consistency.
    n = 64
    yy, xx = np.mgrid[-2:2:n*1j, -2:2:n*1j]
    target = (
        1.6 * np.exp(-((xx + 0.7) ** 2 + (yy + 0.4) ** 2) / 0.18)
        + 1.4 * np.exp(-((xx - 0.7) ** 2 + (yy - 0.4) ** 2) / 0.22)
        - 0.9 * np.exp(-((xx ** 2 + yy ** 2 - 1.4) ** 2) / 0.05)
    )
    target = (target - target.mean()) / target.std()

    # Reverse-time interpolation x_t = a(t) * target + sqrt(1-a^2) * eps_fixed
    # Using a single eps allows a smooth, deterministic denoising trajectory.
    eps = rng.standard_normal(target.shape)
    times = [3.0, 1.5, 0.8, 0.4, 0.15, 0.0]

    fig, axes = plt.subplots(2, len(times), figsize=(13.2, 5.0))

    for ax, t in zip(axes[0], times):
        a = np.exp(-0.5 * 1.0 * t)
        var = max(1 - a ** 2, 0.0)
        img = a * target + np.sqrt(var) * eps
        ax.imshow(img, cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=[-2, 2, -2, 2])
        ax.set_title(f"$t={t:.2f}$")
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom: reverse-time density evolution toward the bimodal data
    x = np.linspace(-5, 5, 400)
    for ax, t in zip(axes[1], times):
        p = vp_marginal_bimodal(x, t)
        ax.fill_between(x, p, color=PURPLE, alpha=0.25)
        ax.plot(x, p, color=PURPLE, lw=2.0)
        ax.plot(x, bimodal_pdf(x), color=GREEN, ls="--", lw=1.2, label="data $p_0$")
        ax.set_xlim(-5, 5); ax.set_ylim(0, 0.55)
        ax.set_xticks([-4, 0, 4]); ax.set_yticks([])
        if t == times[0]:
            ax.set_ylabel("density")
            ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "Reverse diffusion:  $d\\mathbf{X}_t = [f - g^2 \\nabla\\log p_t]\\,dt + g\\,d\\bar{\\mathbf{B}}_t$"
        "    -    score-guided denoising recovers structure",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_reverse_diffusion")


# ---------------------------------------------------------------------------
# Figure 3: Score function vector field
# ---------------------------------------------------------------------------

def fig3_score_field() -> None:
    """Two-mode 2D Gaussian mixture: density heatmap + true score field."""
    grid = np.linspace(-3.5, 3.5, 220)
    X, Y = np.meshgrid(grid, grid)

    mus = np.array([[-1.3, -0.6], [1.3, 0.6]])
    sig2 = 0.45
    weights = np.array([0.5, 0.5])

    p1 = np.exp(-((X - mus[0, 0]) ** 2 + (Y - mus[0, 1]) ** 2) / (2 * sig2)) / (2 * np.pi * sig2)
    p2 = np.exp(-((X - mus[1, 0]) ** 2 + (Y - mus[1, 1]) ** 2) / (2 * sig2)) / (2 * np.pi * sig2)
    p = weights[0] * p1 + weights[1] * p2

    # score = grad log p = (sum_k w_k p_k * (mu_k - x)/sig2) / p
    g1x = (mus[0, 0] - X) / sig2
    g1y = (mus[0, 1] - Y) / sig2
    g2x = (mus[1, 0] - X) / sig2
    g2y = (mus[1, 1] - Y) / sig2
    sx = (weights[0] * p1 * g1x + weights[1] * p2 * g2x) / p
    sy = (weights[0] * p1 * g1y + weights[1] * p2 * g2y) / p

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    ax = axes[0]
    cf = ax.contourf(X, Y, p, levels=20, cmap="Purples")
    ax.contour(X, Y, p, levels=8, colors="white", linewidths=0.6, alpha=0.6)
    ax.scatter(mus[:, 0], mus[:, 1], color=RED, s=80, zorder=5, edgecolor="white", lw=1.5, label="modes")
    ax.set_title("density  $p(\\mathbf{x})$")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.contour(X, Y, p, levels=8, colors=GREY, linewidths=0.6, alpha=0.5)
    step = 14
    mag = np.sqrt(sx ** 2 + sy ** 2)
    ax.quiver(
        X[::step, ::step], Y[::step, ::step],
        sx[::step, ::step], sy[::step, ::step],
        mag[::step, ::step],
        cmap="viridis", scale=80, width=0.004,
    )
    ax.scatter(mus[:, 0], mus[:, 1], color=RED, s=80, zorder=5, edgecolor="white", lw=1.5)
    ax.set_title("score field  $\\mathbf{s}(\\mathbf{x}) = \\nabla\\log p(\\mathbf{x})$")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)

    fig.suptitle("Score function: arrows point toward high-density regions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig3_score_field")


# ---------------------------------------------------------------------------
# Figure 4: DDPM vs DDIM sampling
# ---------------------------------------------------------------------------

def fig4_ddpm_vs_ddim() -> None:
    """Compare DDPM (stochastic) and DDIM (deterministic) trajectories on a 2D mixture."""
    rng = np.random.default_rng(3)

    mus = np.array([[-1.3, -0.6], [1.3, 0.6]])
    sig2 = 0.45

    def score_fn(x: np.ndarray) -> np.ndarray:
        # x: (N, 2)
        d1 = x - mus[0]
        d2 = x - mus[1]
        e1 = np.exp(-(d1 ** 2).sum(axis=1) / (2 * sig2))
        e2 = np.exp(-(d2 ** 2).sum(axis=1) / (2 * sig2))
        w = e1 / (e1 + e2)
        s = (w[:, None] * (-d1) + (1 - w)[:, None] * (-d2)) / sig2
        return s

    # Reverse VP-SDE on time interval [0, T].  beta=1 constant.
    T = 4.0
    N = 50
    M = 14  # number of trajectories

    # Same starting noise for both samplers, for visual comparability
    x0 = rng.standard_normal((M, 2)) * 1.6

    # Reverse SDE: dX = [-0.5*beta*X - beta * s(X)] dt + sqrt(beta) dW   (running backwards in t)
    dt = T / N
    beta = 1.0

    traj_ddpm = [x0.copy()]
    x = x0.copy()
    for _ in range(N):
        s = score_fn(x)
        drift = (-0.5 * beta * x - beta * s) * (-dt)  # reverse time
        noise = np.sqrt(beta * dt) * rng.standard_normal(x.shape)
        x = x + drift + noise
        traj_ddpm.append(x.copy())
    traj_ddpm = np.array(traj_ddpm)  # (N+1, M, 2)

    # Probability flow ODE (Heun, deterministic)
    N2 = 25
    dt2 = T / N2
    traj_ddim = [x0.copy()]
    x = x0.copy()
    for _ in range(N2):
        s1 = score_fn(x)
        f1 = (-0.5 * beta * x - 0.5 * beta * s1) * (-dt2)
        x_pred = x + f1
        s2 = score_fn(x_pred)
        f2 = (-0.5 * beta * x_pred - 0.5 * beta * s2) * (-dt2)
        x = x + 0.5 * (f1 + f2)
        traj_ddim.append(x.copy())
    traj_ddim = np.array(traj_ddim)

    # Background density
    grid = np.linspace(-3.5, 3.5, 200)
    X, Y = np.meshgrid(grid, grid)
    p1 = np.exp(-((X - mus[0, 0]) ** 2 + (Y - mus[0, 1]) ** 2) / (2 * sig2))
    p2 = np.exp(-((X - mus[1, 0]) ** 2 + (Y - mus[1, 1]) ** 2) / (2 * sig2))
    p = 0.5 * (p1 + p2) / (2 * np.pi * sig2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    for ax, traj, title, color, n_steps in [
        (axes[0], traj_ddpm, f"DDPM (reverse SDE, {N} steps, stochastic)", BLUE, N),
        (axes[1], traj_ddim, f"DDIM (probability-flow ODE, {N2} steps, deterministic)", PURPLE, N2),
    ]:
        ax.contourf(X, Y, p, levels=15, cmap="Greys", alpha=0.55)
        for j in range(M):
            ax.plot(traj[:, j, 0], traj[:, j, 1], color=color, lw=1.0, alpha=0.55)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], color=GREY, s=35, zorder=5, label="$x_T$ (noise)")
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color=RED, s=45, zorder=6, label="$x_0$ (sample)")
        ax.scatter(mus[:, 0], mus[:, 1], color=GREEN, marker="x", s=120, zorder=7, lw=2.5, label="modes")
        ax.set_title(title)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "DDPM vs DDIM: same model, same prior; SDE adds noise per step, ODE follows a deterministic flow",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig4_ddpm_vs_ddim")


# ---------------------------------------------------------------------------
# Figure 5: Score matching loss
# ---------------------------------------------------------------------------

def fig5_score_matching_loss() -> None:
    """Show DSM loss landscape and a 1D learning result."""
    rng = np.random.default_rng(4)

    # 1D mixture target
    mu, s0 = 1.6, 0.5
    x_data = np.concatenate([
        rng.normal(-mu, s0, 4000),
        rng.normal(mu, s0, 4000),
    ])

    # True score on grid
    xg = np.linspace(-4, 4, 400)
    p = bimodal_pdf(xg, mu, s0)
    eps = 1e-9
    log_p = np.log(p + eps)
    score_true = np.gradient(log_p, xg)

    # Learn score with a tiny MLP via DSM (numpy, no torch)
    sigma = 0.4

    def features(x):
        # RBF features
        centers = np.linspace(-3.5, 3.5, 24)
        bw = 0.6
        return np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * bw ** 2))

    Phi_data = features(x_data)
    # closed-form least-squares for score(x_noisy) = -(x_noisy - x)/sigma^2
    rng2 = np.random.default_rng(5)

    # Fixed validation set for a clean loss curve
    val_noise = rng2.normal(0, sigma, x_data.shape)
    x_val = x_data + val_noise
    target_val = -val_noise / (sigma ** 2)
    Phi_val = features(x_val)

    losses = []
    A = np.zeros((Phi_data.shape[1], Phi_data.shape[1]))
    b = np.zeros(Phi_data.shape[1])
    n_rounds = 40
    for _ in range(n_rounds):
        noise = rng2.normal(0, sigma, x_data.shape)
        xn = x_data + noise
        target = -noise / (sigma ** 2)
        Phi_n = features(xn)
        A += Phi_n.T @ Phi_n
        b += Phi_n.T @ target
        w_curr = np.linalg.solve(A + 1e-3 * np.eye(A.shape[0]), b)
        # measure loss on the FIXED validation noise => smooth, monotone curve
        pred_val = Phi_val @ w_curr
        losses.append(np.mean((pred_val - target_val) ** 2))
    w = np.linalg.solve(A + 1e-3 * np.eye(A.shape[0]), b)
    score_learned = features(xg) @ w

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    ax = axes[0]
    ax.semilogy(np.arange(1, len(losses) + 1), losses, color=BLUE, lw=2.0, marker="o", ms=4)
    ax.set_xlabel("DSM training round")
    ax.set_ylabel("$\\mathcal{L}_{\\mathrm{DSM}}$  (log scale)")
    ax.set_title("Denoising score-matching loss converges")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(xg, p * 6, color=GREY, lw=1.5, ls=":", label="data density (scaled)")
    ax.plot(xg, score_true, color=RED, lw=2.4, label="true  $\\nabla\\log p$")
    ax.plot(xg, score_learned, color=GREEN, lw=2.0, ls="--", label="learned  $\\mathbf{s}_\\theta$")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("score")
    ax.set_ylim(-12, 12)
    ax.set_title("Learned score matches the truth")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Score matching:  $\\mathcal{L}_{\\mathrm{DSM}} = \\mathbb{E}\\,\\|\\mathbf{s}_\\theta(\\tilde{x}) + (\\tilde{x}-x)/\\sigma^2\\|^2$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig5_score_matching_loss")


# ---------------------------------------------------------------------------
# Figure 6: PDE -> Diffusion model bridge
# ---------------------------------------------------------------------------

def fig6_pde_diffusion_bridge() -> None:
    """Conceptual bridge: heat / Fokker-Planck PDE  <->  diffusion model."""
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color, fc=None, fontsize=10.5, weight="bold"):
        if fc is None:
            fc = color + "22"
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.18",
            linewidth=2.0, edgecolor=color, facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, color="#111827")

    def arrow(x1, y1, x2, y2, color=GREY, label=None, lx=None, ly=None, style="->"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, mutation_scale=18,
                                     lw=1.6, color=color))
        if label:
            ax.text(lx if lx is not None else (x1 + x2) / 2,
                    ly if ly is not None else (y1 + y2) / 2 + 0.18,
                    label, ha="center", va="bottom",
                    fontsize=9.5, color=color, style="italic")

    # Title
    ax.text(6.0, 7.55, "PDE  $\\longleftrightarrow$  Diffusion Model",
            ha="center", fontsize=14, weight="bold")
    ax.text(6.0, 7.15, "Forward-time stochastic dynamics meet score-based learning",
            ha="center", fontsize=10.5, color=GREY, style="italic")

    # Top row: PDE/SDE side
    box(0.3, 5.3, 3.3, 1.1, "Heat equation\n$\\partial_t u = D\\,\\nabla^2 u$", BLUE)
    box(4.35, 5.3, 3.3, 1.1, "Fokker--Planck\n$\\partial_t p = -\\nabla\\!\\cdot\\!(fp) + \\frac{g^2}{2}\\nabla^2 p$", PURPLE)
    box(8.4, 5.3, 3.3, 1.1, "Forward SDE\n$d\\mathbf{X} = f\\,dt + g\\,d\\mathbf{B}$", GREEN)

    arrow(3.6, 5.85, 4.35, 5.85, color=GREY, label="$f=0,\\,g^2/2=D$")
    arrow(7.65, 5.85, 8.4, 5.85, color=GREY, label="density of $\\mathbf{X}_t$")

    # Middle bridge
    box(2.0, 3.3, 8.0, 1.1,
        "Anderson reverse-time SDE      $d\\mathbf{X}_t = [\\,f - g^2\\,\\nabla\\!\\log p_t(\\mathbf{X}_t)\\,]\\,dt + g\\,d\\bar{\\mathbf{B}}_t$",
        RED, fc="#fee2e2", fontsize=11)
    arrow(6.0, 5.3, 6.0, 4.4, color=RED, label="time reversal", lx=6.7, ly=4.85)
    arrow(6.0, 3.3, 6.0, 2.6, color=RED, label="needs  $\\nabla\\!\\log p_t$", lx=7.05, ly=2.85)

    # Bottom row: ML side
    box(0.3, 0.9, 3.3, 1.5,
        "Score network\n$\\mathbf{s}_\\theta(\\mathbf{x},t) \\approx \\nabla\\!\\log p_t$\n(trained by DSM)",
        BLUE)
    box(4.35, 0.9, 3.3, 1.5,
        "DDPM sampler\nDiscretise reverse SDE\n$\\sim 1000$ steps",
        PURPLE)
    box(8.4, 0.9, 3.3, 1.5,
        "DDIM sampler\nProbability-flow ODE\n$\\sim 25\\!\\!-\\!\\!50$ steps",
        GREEN)

    arrow(3.6, 1.65, 4.35, 1.65, color=GREY)
    arrow(7.65, 1.65, 8.4, 1.65, color=GREY)
    arrow(2.0, 2.4, 2.0, 3.3, color=GREY, label="plug in", lx=1.3, ly=2.85)

    save(fig, "fig6_pde_diffusion_bridge")


# ---------------------------------------------------------------------------
# Figure 7: Latent diffusion / Stable Diffusion architecture
# ---------------------------------------------------------------------------

def fig7_latent_diffusion() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 6.6)
    ax.axis("off")

    def block(x, y, w, h, text, color, fc=None, fs=10):
        if fc is None:
            fc = color + "22"
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                              linewidth=2.0, edgecolor=color, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight="bold", color="#111827")

    def arrow(x1, y1, x2, y2, color=GREY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="->", mutation_scale=18,
                                     lw=1.8, color=color))

    ax.text(6.6, 6.25, "Latent Diffusion (Stable Diffusion) Architecture",
            ha="center", fontsize=13.5, weight="bold")

    # Pixel space (vertical centre y ~ 3.4)
    block(0.2, 3.0, 1.8, 1.2, "Image\n$\\mathbf{x} \\in \\mathbb{R}^{H\\times W\\times 3}$", BLUE)
    arrow(2.0, 3.6, 2.7, 3.6)
    block(2.7, 3.0, 1.7, 1.2, "Encoder\n$\\mathcal{E}$", PURPLE)
    arrow(4.4, 3.6, 5.1, 3.6)

    # Latent space
    block(5.1, 2.7, 3.1, 1.8,
          "Latent  $\\mathbf{z}_0 \\in \\mathbb{R}^{h\\times w\\times c}$\n($\\sim 8\\times$ smaller)",
          GREEN, fc=GREEN + "1f")

    # Diffusion process inside latent (separated above box)
    ax.text(6.65, 5.55, "Diffusion in latent space", ha="center",
            fontsize=10, color=GREY, style="italic")
    arrow(5.4, 5.18, 7.9, 5.18)
    ax.text(6.65, 4.78, "$\\mathbf{z}_0 \\longrightarrow \\mathbf{z}_T$  (forward)",
            ha="center", fontsize=8.8, color=GREY)

    arrow(8.2, 3.6, 8.9, 3.6)
    block(8.9, 3.0, 1.7, 1.2, "Decoder\n$\\mathcal{D}$", PURPLE)
    arrow(10.6, 3.6, 11.3, 3.6)
    block(11.3, 3.0, 1.7, 1.2, "Image\n$\\hat{\\mathbf{x}}$", BLUE)

    # Conditioning + reverse path
    block(2.7, 0.6, 4.0, 1.1,
          "Text prompt  $\\to$  CLIP text encoder  $\\to$  $\\mathbf{c}$",
          RED, fc=RED + "1f", fs=10.5)
    block(7.0, 0.6, 4.5, 1.1,
          "U-Net  $\\boldsymbol{\\epsilon}_\\theta(\\mathbf{z}_t,\\,t,\\,\\mathbf{c})$  with cross-attention",
          PURPLE, fs=10.5)
    arrow(6.7, 1.15, 7.0, 1.15)
    # reverse path: U-Net produces noise estimate that updates latent
    arrow(9.25, 1.7, 9.25, 2.7)
    ax.text(9.4, 2.25, "$\\mathbf{z}_T \\to \\mathbf{z}_0$\n(reverse)",
            ha="left", va="center", fontsize=8.8, color=GREY, style="italic")

    # Footer note
    ax.text(6.6, 0.15,
            "Key idea: train and sample diffusion in a low-dim latent  $\\to$  $\\sim\\!\\!64\\times$ less compute,  same fidelity",
            ha="center", fontsize=10, color=GREY, style="italic")

    fig.tight_layout()
    save(fig, "fig7_latent_diffusion")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    print("Generating figures...")
    fig1_forward_diffusion()
    fig2_reverse_diffusion()
    fig3_score_field()
    fig4_ddpm_vs_ddim()
    fig5_score_matching_loss()
    fig6_pde_diffusion_bridge()
    fig7_latent_diffusion()
    print("Done.")


if __name__ == "__main__":
    main()
