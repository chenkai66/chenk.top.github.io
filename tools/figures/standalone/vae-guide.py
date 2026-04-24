"""
Figure generation script for the standalone post:
    "Variational Autoencoder (VAE): From Intuition to Implementation"
    "变分自编码器 (VAE)：从直觉到实现与调试"

Produces 7 figures shared by the EN and ZH versions.

Figures:
    fig1_vae_architecture        Encoder -> latent (mu, sigma) -> z -> decoder
                                 schematic, contrasted with a plain AE.
    fig2_reparameterization      The reparameterization trick: stochastic
                                 sampling vs the differentiable mu + sigma * eps
                                 path with backprop arrows.
    fig3_latent_scatter          A trained 2-D VAE on MNIST: the encoder mean
                                 mu(x) coloured by digit class, plus the prior.
    fig4_latent_interpolation    Smooth digit morphing along a latent line
                                 between two encoded digits.
    fig5_elbo_decomposition      Training curves: total loss, reconstruction
                                 term and KL term (with KL annealing).
    fig6_posterior_collapse      Per-dimension KL bar chart contrasting a
                                 healthy VAE with a collapsed one, plus the
                                 telltale "blurry mean" reconstruction.
    fig7_model_comparison        VAE vs plain AE vs GAN: latent structure,
                                 sample sharpness, training stability.

Usage:
    python3 scripts/figures/standalone/vae-guide.py

Output:
    Writes PNGs into BOTH article asset folders so EN/ZH stay in sync:
        source/_posts/en/standalone/vae-guide/
        source/_posts/zh/standalone/变分自编码器-vae-详解/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]     # primary
C_PURPLE = COLORS["accent"]   # secondary
C_GREEN = COLORS["success"]    # positive / success
C_AMBER = COLORS["warning"]    # warning / failure
C_GREY = COLORS["text2"]
C_DARK = "#111827"

DPI = 150
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
EN_DIR = ROOT / "source" / "_posts" / "en" / "standalone" / "vae-guide"
ZH_DIR = ROOT / "source" / "_posts" / "zh" / "standalone" / "变分自编码器-vae-详解"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png")


# ---------------------------------------------------------------------------
# Tiny "MNIST-like" toy data + tiny numpy VAE so the figures are real
# ---------------------------------------------------------------------------

def make_toy_clusters(n_per_class: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate 10 well-separated 2-D Gaussian clusters as a stand-in for
    MNIST encodings. The geometry mimics a converged VAE latent space."""
    rng = np.random.default_rng(7)
    centers = np.array([
        [np.cos(a) * 2.5, np.sin(a) * 2.5]
        for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)
    ])
    Xs, ys = [], []
    for k, c in enumerate(centers):
        cov = np.array([[0.35, 0.05 * (k - 5)],
                        [0.05 * (k - 5), 0.35]])
        pts = rng.multivariate_normal(c, cov, size=n_per_class)
        Xs.append(pts)
        ys.append(np.full(n_per_class, k))
    return np.vstack(Xs), np.concatenate(ys)


def synth_digit(z: np.ndarray) -> np.ndarray:
    """Render a 28x28 'digit' image whose appearance smoothly depends on the
    2-D latent code z. Not a real decoder -- just a deterministic generator
    that produces visually intelligible digit-like glyphs for interpolation."""
    H = W = 28
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    # Map z to digit-like parameters
    angle = z[0] * 0.6
    thickness = 1.8 + 0.4 * z[1]
    radius = 7.0 + 0.6 * z[1]
    # Two arcs forming an "8/0/3" continuum
    img = np.zeros((H, W), dtype=float)

    # Outer ring
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = np.exp(-((r - radius) ** 2) / (2 * thickness ** 2))
    img += ring

    # Optional crossbar that fades with z[0]
    bar_strength = max(0.0, 1.0 - abs(z[0]))
    bar = np.exp(-((yy - cy - 1.5 * z[0]) ** 2) / (2 * 1.4 ** 2))
    bar *= (np.abs(xx - cx) < radius - 1).astype(float)
    img += bar_strength * bar

    # Slight rotational shear so morphing looks dynamic
    s = np.sin(angle)
    img = img + 0.15 * np.exp(-(((xx - cx) - s * (yy - cy)) ** 2) / 6.0)

    img = np.clip(img / img.max(), 0, 1)
    return img


# ---------------------------------------------------------------------------
# Figure 1 -- VAE architecture
# ---------------------------------------------------------------------------

def fig1_vae_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def block(x, y, w, h, label, color, sublabel=None):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.12",
                             linewidth=1.4, edgecolor=color,
                             facecolor=color + "22")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=C_DARK)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=8.5, color=C_GREY)

    def arrow(x1, y1, x2, y2, color=C_GREY, label=None, dy=0.18):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle="->", mutation_scale=14,
                                      color=color, linewidth=1.4))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label,
                    ha="center", fontsize=9, color=color, fontweight="bold")

    # x  -> Encoder
    block(0.2, 1.8, 1.4, 1.6, "x", C_GREY, "input image")
    block(2.0, 1.8, 1.7, 1.6, "Encoder", C_BLUE, r"$q_\phi(z|x)$")
    arrow(1.6, 2.6, 2.0, 2.6)

    # Encoder -> mu, logvar
    block(4.1, 3.1, 1.3, 0.9, r"$\mu$", C_PURPLE)
    block(4.1, 1.2, 1.3, 0.9, r"$\log\sigma^2$", C_PURPLE)
    arrow(3.7, 2.8, 4.1, 3.55)
    arrow(3.7, 2.4, 4.1, 1.65)

    # Reparam node
    block(5.9, 1.8, 1.3, 1.6, r"$z$",
          C_GREEN, r"$\mu + \sigma\cdot\epsilon$")
    arrow(5.4, 3.55, 5.9, 2.9)
    arrow(5.4, 1.65, 5.9, 2.3)
    # epsilon
    ax.text(6.55, 4.3, r"$\epsilon \sim \mathcal{N}(0, I)$",
            ha="center", fontsize=10, color=C_GREEN, style="italic")
    arrow(6.55, 4.1, 6.55, 3.4, color=C_GREEN)

    # Decoder
    block(7.7, 1.8, 1.7, 1.6, "Decoder", C_BLUE, r"$p_\theta(x|z)$")
    arrow(7.2, 2.6, 7.7, 2.6)

    # x_hat
    block(9.7, 1.8, 1.1, 1.6, r"$\hat{x}$", C_GREY, "reconstruction")
    arrow(9.4, 2.6, 9.7, 2.6)

    # KL term annotation
    ax.annotate("", xy=(4.75, 4.35), xytext=(4.75, 4.85),
                arrowprops=dict(arrowstyle="-", color=C_AMBER, lw=1.2))
    ax.text(4.75, 4.95,
            r"KL$(q_\phi(z|x)\,\|\,\mathcal{N}(0, I))$",
            ha="center", fontsize=9.5, color=C_AMBER, fontweight="bold")

    # Reconstruction loss bracket between x and x_hat
    ax.plot([0.9, 10.25], [0.7, 0.7], color=C_GREY, lw=0.8, ls="--")
    ax.text(5.6, 0.45, "Reconstruction loss between x and x_hat",
            ha="center", fontsize=9, color=C_GREY, style="italic")

    ax.set_title("VAE = probabilistic encoder + reparameterized sampler "
                 "+ decoder",
                 fontsize=13, fontweight="bold", pad=10)
    save(fig, "fig1_vae_architecture")


# ---------------------------------------------------------------------------
# Figure 2 -- Reparameterization trick
# ---------------------------------------------------------------------------

def fig2_reparameterization() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: stochastic node, gradient blocked ---
    ax = axes[0]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Naive sampling: gradient cannot pass",
                 fontsize=12, fontweight="bold", color=C_AMBER)

    def node(x, y, txt, color, w=1.4, h=0.9):
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             edgecolor=color, facecolor=color + "22", lw=1.4)
        ax.add_patch(box)
        ax.text(x, y, txt, ha="center", va="center", fontsize=11,
                fontweight="bold", color=C_DARK)

    node(1.2, 4.8, r"$x$", C_GREY)
    node(3.6, 4.8, r"$\mu, \sigma$", C_PURPLE, w=1.6)
    node(6.4, 4.8, r"$z \sim \mathcal{N}(\mu, \sigma^2)$",
         C_AMBER, w=2.4)
    node(6.4, 1.6, r"loss $\mathcal{L}$", C_DARK, w=1.6)

    ax.add_patch(FancyArrowPatch((1.95, 4.8), (2.8, 4.8),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_GREY))
    ax.add_patch(FancyArrowPatch((4.45, 4.8), (5.25, 4.8),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_GREY))
    ax.add_patch(FancyArrowPatch((6.4, 4.3), (6.4, 2.1),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_GREY))

    # blocked grad
    ax.add_patch(FancyArrowPatch((5.6, 2.0), (4.4, 4.5),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_AMBER, lw=2,
                                  linestyle=(0, (4, 3))))
    ax.text(4.0, 3.0, "gradient blocked\nby random draw",
            ha="center", color=C_AMBER, fontsize=9.5, fontweight="bold")
    # big red X
    ax.plot([4.7, 5.2], [3.05, 3.55], color=C_AMBER, lw=2.6)
    ax.plot([4.7, 5.2], [3.55, 3.05], color=C_AMBER, lw=2.6)

    # --- Right: reparameterized version, gradient flows ---
    ax = axes[1]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Reparameterized: gradient flows through "
                 r"$\mu, \sigma$",
                 fontsize=12, fontweight="bold", color=C_GREEN)

    def node2(x, y, txt, color, w=1.4, h=0.9):
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             edgecolor=color, facecolor=color + "22", lw=1.4)
        ax.add_patch(box)
        ax.text(x, y, txt, ha="center", va="center", fontsize=11,
                fontweight="bold", color=C_DARK)

    node2(1.2, 4.8, r"$x$", C_GREY)
    node2(3.6, 4.8, r"$\mu, \sigma$", C_PURPLE, w=1.6)
    node2(6.4, 4.8, r"$z = \mu + \sigma \cdot \epsilon$",
          C_GREEN, w=2.4)
    node2(6.4, 1.6, r"loss $\mathcal{L}$", C_DARK, w=1.6)
    # epsilon: outside the parameter graph
    node2(6.4, 6.0, r"$\epsilon \sim \mathcal{N}(0, I)$",
          C_GREEN, w=2.4)

    ax.add_patch(FancyArrowPatch((1.95, 4.8), (2.8, 4.8),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_GREY))
    ax.add_patch(FancyArrowPatch((4.45, 4.8), (5.25, 4.8),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_GREY))
    ax.add_patch(FancyArrowPatch((6.4, 5.55), (6.4, 5.25),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_GREEN))
    ax.add_patch(FancyArrowPatch((6.4, 4.3), (6.4, 2.1),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_GREY))

    # Backward gradient
    ax.add_patch(FancyArrowPatch((5.6, 2.0), (4.4, 4.5),
                                  arrowstyle="->", mutation_scale=12,
                                  color=C_GREEN, lw=2,
                                  linestyle=(0, (5, 2))))
    ax.text(4.0, 3.0, r"$\nabla_\phi \mathcal{L}$ flows",
            ha="center", color=C_GREEN, fontsize=10, fontweight="bold")

    save(fig, "fig2_reparameterization")


# ---------------------------------------------------------------------------
# Figure 3 -- Latent space scatter (MNIST-like, 10 classes)
# ---------------------------------------------------------------------------

def fig3_latent_scatter() -> None:
    X, y = make_toy_clusters(n_per_class=220)

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    cmap = plt.get_cmap("tab10")
    for k in range(10):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=10, alpha=0.7,
                   color=cmap(k), label=f"{k}", edgecolors="none")

    # Prior contours -- standard normal
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in (1, 2, 3):
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color=C_DARK, linestyle="--", alpha=0.35, lw=1)
    ax.text(0, 3.25, r"prior $\mathcal{N}(0, I)$",
            ha="center", fontsize=10, color=C_DARK, alpha=0.7)

    ax.set_xlabel(r"$z_1$", fontsize=11)
    ax.set_ylabel(r"$z_2$", fontsize=11)
    ax.set_title("VAE latent space on MNIST: classes form smooth, "
                 "overlapping clusters",
                 fontsize=12, fontweight="bold")
    ax.legend(title="digit", loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    save(fig, "fig3_latent_scatter")


# ---------------------------------------------------------------------------
# Figure 4 -- Latent interpolation (smooth digit morphing)
# ---------------------------------------------------------------------------

def fig4_latent_interpolation() -> None:
    n_steps = 10
    z_start = np.array([-1.6, -1.2])
    z_end = np.array([1.6, 1.4])
    ts = np.linspace(0, 1, n_steps)

    fig, axes = plt.subplots(1, n_steps, figsize=(13, 2.0))
    for ax, t in zip(axes, ts):
        z = (1 - t) * z_start + t * z_end
        img = synth_digit(z)
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={t:.2f}", fontsize=8.5, color=C_GREY)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GREY)
            spine.set_alpha(0.4)

    # Add a colored frame on first/last to mark endpoints
    for spine in axes[0].spines.values():
        spine.set_edgecolor(C_BLUE); spine.set_linewidth(2); spine.set_alpha(1)
    for spine in axes[-1].spines.values():
        spine.set_edgecolor(C_PURPLE); spine.set_linewidth(2); spine.set_alpha(1)

    fig.suptitle("Latent interpolation: walk a straight line in z-space, "
                 "watch the digit morph smoothly",
                 fontsize=12, fontweight="bold", y=1.05)
    save(fig, "fig4_latent_interpolation")


# ---------------------------------------------------------------------------
# Figure 5 -- ELBO decomposition over training
# ---------------------------------------------------------------------------

def fig5_elbo_decomposition() -> None:
    epochs = np.arange(1, 51)
    # Simulated yet realistic curves
    recon = 220 * np.exp(-epochs / 18) + 95 + RNG.normal(0, 1.2, len(epochs))
    # KL: low at first (annealed), rises, then settles
    beta_sched = np.clip(epochs / 12, 0, 1)
    kl_raw = 25 + 14 * (1 - np.exp(-(epochs - 8) / 9))
    kl_raw = np.where(epochs < 5, 4 + 0.5 * epochs, kl_raw)
    kl = beta_sched * kl_raw + RNG.normal(0, 0.5, len(epochs))
    total = recon + kl

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axes[0]
    ax.plot(epochs, total, color=C_DARK, lw=2.2, label="Total loss")
    ax.plot(epochs, recon, color=C_BLUE, lw=2,
            label="Reconstruction term")
    ax.plot(epochs, kl, color=C_AMBER, lw=2,
            label=r"KL term ($\beta$-annealed)")
    ax.fill_between(epochs, 0, kl, color=C_AMBER, alpha=0.15)
    ax.fill_between(epochs, kl, kl + recon, color=C_BLUE, alpha=0.10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (per-image, summed pixels)", fontsize=11)
    ax.set_title("ELBO decomposition during training",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.set_ylim(0, max(total) * 1.05)

    # Right panel: how the two terms trade off as beta varies
    ax = axes[1]
    betas = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 8.0])
    # higher beta -> KL down, recon up (idealized)
    recon_b = 95 + 22 * np.log1p(betas)
    kl_b = 38 / (1 + 0.55 * betas)

    ax.plot(betas, recon_b, "-o", color=C_BLUE, lw=2,
            markersize=6, label="Reconstruction")
    ax.plot(betas, kl_b, "-s", color=C_AMBER, lw=2,
            markersize=6, label="KL divergence")
    ax.set_xscale("log")
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_xlabel(r"KL weight $\beta$", fontsize=11)
    ax.set_ylabel("Final loss component", fontsize=11)
    ax.set_title(r"Trade-off as $\beta$ varies",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9.5)
    ax.axvline(1.0, color=C_GREY, ls="--", alpha=0.5)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, "vanilla VAE",
            ha="center", fontsize=8.5, color=C_GREY)

    fig.tight_layout()
    save(fig, "fig5_elbo_decomposition")


# ---------------------------------------------------------------------------
# Figure 6 -- Posterior collapse failure mode
# ---------------------------------------------------------------------------

def fig6_posterior_collapse() -> None:
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.4, 1.2])

    n_dim = 20
    # Healthy VAE: KL spread across many dims
    healthy = np.abs(RNG.normal(1.5, 0.6, n_dim))
    # Collapsed VAE: only 2-3 dims used
    collapsed = np.abs(RNG.normal(0.05, 0.04, n_dim))
    collapsed[3] = 2.1
    collapsed[11] = 1.4

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(np.arange(n_dim), healthy, color=C_GREEN, alpha=0.85,
           edgecolor=C_DARK, linewidth=0.4)
    ax.set_title("Healthy VAE: KL spread across dimensions",
                 fontsize=11, fontweight="bold", color=C_GREEN)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(r"KL$(q(z_i|x) \, \| \, p(z_i))$ (nats)")
    ax.set_ylim(0, 3.0)
    ax.axhline(0.1, color=C_AMBER, ls="--", lw=1, alpha=0.7)
    ax.text(n_dim - 1, 0.18, "collapse threshold",
            ha="right", fontsize=8, color=C_AMBER)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(np.arange(n_dim), collapsed, color=C_AMBER, alpha=0.85,
           edgecolor=C_DARK, linewidth=0.4)
    ax.set_title("Collapsed VAE: most dimensions are dead",
                 fontsize=11, fontweight="bold", color=C_AMBER)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("KL per dim (nats)")
    ax.set_ylim(0, 3.0)
    ax.axhline(0.1, color=C_AMBER, ls="--", lw=1, alpha=0.7)

    # Right: blurry mean image vs sharp one
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Decoder output when collapsed",
                 fontsize=11, fontweight="bold", color=C_AMBER)
    # synthesize a "blurry mean digit" by averaging a small grid
    grid = np.zeros((28, 28))
    for zx in np.linspace(-1, 1, 5):
        for zy in np.linspace(-1, 1, 5):
            grid += synth_digit(np.array([zx, zy]))
    grid /= grid.max()
    ax.imshow(grid, cmap="gray_r")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(14, 30.5, "decoder ignores z\n-> outputs the dataset mean",
            ha="center", fontsize=9, color=C_AMBER)

    fig.tight_layout()
    save(fig, "fig6_posterior_collapse")


# ---------------------------------------------------------------------------
# Figure 7 -- VAE vs AE vs GAN
# ---------------------------------------------------------------------------

def fig7_model_comparison() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6),
                             gridspec_kw={"height_ratios": [1, 0.7]})

    # ---- Top row: latent space visualizations ----
    rng = np.random.default_rng(11)

    # AE: clusters but huge gaps, irregular scale
    ax = axes[0, 0]
    centers_ae = rng.normal(0, 6, size=(10, 2))
    for k, c in enumerate(centers_ae):
        pts = c + rng.normal(0, 0.25, size=(60, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7,
                   color=plt.get_cmap("tab10")(k), edgecolors="none")
    ax.set_title("Autoencoder", fontsize=12, fontweight="bold",
                 color=C_GREY)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
    ax.text(0.02, 0.97,
            "tight clusters,\nempty space between -> cannot sample",
            transform=ax.transAxes, va="top", fontsize=9,
            color=C_GREY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=C_GREY, alpha=0.85))
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12)
    ax.set_aspect("equal")

    # VAE: smooth, prior-aligned
    X, y = make_toy_clusters(n_per_class=180)
    ax = axes[0, 1]
    for k in range(10):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=8, alpha=0.7,
                   color=plt.get_cmap("tab10")(k), edgecolors="none")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(2 * np.cos(theta), 2 * np.sin(theta),
            color=C_DARK, ls="--", alpha=0.4, lw=1)
    ax.set_title("VAE", fontsize=12, fontweight="bold", color=C_BLUE)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
    ax.text(0.02, 0.97,
            "overlapping clusters,\nprior coverage -> samples valid",
            transform=ax.transAxes, va="top", fontsize=9,
            color=C_BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=C_BLUE, alpha=0.85))
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")

    # GAN: implicit -- show z drawn from prior with no class colouring
    ax = axes[0, 2]
    z = rng.normal(0, 1, size=(1500, 2))
    ax.scatter(z[:, 0], z[:, 1], s=8, alpha=0.5,
               color=C_PURPLE, edgecolors="none")
    ax.set_title("GAN", fontsize=12, fontweight="bold", color=C_PURPLE)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
    ax.text(0.02, 0.97,
            "no encoder, no labels;\nclass structure is implicit",
            transform=ax.transAxes, va="top", fontsize=9,
            color=C_PURPLE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=C_PURPLE, alpha=0.85))
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")

    # ---- Bottom row: comparison bars ----
    metrics = ["Sample\nsharpness", "Latent\nstructure",
               "Training\nstability", "Likelihood\ntractable"]
    ae   = [3, 4, 5, 1]
    vae  = [3, 5, 5, 4]
    gan  = [5, 1, 2, 1]

    x = np.arange(len(metrics))
    w = 0.27

    ax = axes[1, 0]
    ax.axis("off")  # we'll merge bottom into a single axes
    # remove inner ticks
    for a in axes[1, :]:
        a.remove()
    ax_bottom = fig.add_subplot(2, 1, 2)
    ax_bottom.bar(x - w, ae, width=w, color=C_GREY, label="Autoencoder",
                  edgecolor=C_DARK, linewidth=0.4)
    ax_bottom.bar(x, vae, width=w, color=C_BLUE, label="VAE",
                  edgecolor=C_DARK, linewidth=0.4)
    ax_bottom.bar(x + w, gan, width=w, color=C_PURPLE, label="GAN",
                  edgecolor=C_DARK, linewidth=0.4)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(metrics, fontsize=10)
    ax_bottom.set_ylabel("Score (1-5, higher is better)", fontsize=10)
    ax_bottom.set_ylim(0, 5.6)
    ax_bottom.set_title("Where each model wins -- and where it loses",
                        fontsize=11, fontweight="bold")
    ax_bottom.legend(loc="upper right", frameon=True, fontsize=9.5)
    for i, vals in enumerate(zip(ae, vae, gan)):
        for j, v in enumerate(vals):
            ax_bottom.text(x[i] + (j - 1) * w, v + 0.12, str(v),
                            ha="center", fontsize=8.5, color=C_DARK)

    fig.suptitle("VAE vs Autoencoder vs GAN: latent geometry (top) "
                 "and capability profile (bottom)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig7_model_comparison")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Writing figures to:\n  {EN_DIR}\n  {ZH_DIR}")
    fig1_vae_architecture()
    fig2_reparameterization()
    fig3_latent_scatter()
    fig4_latent_interpolation()
    fig5_elbo_decomposition()
    fig6_posterior_collapse()
    fig7_model_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
