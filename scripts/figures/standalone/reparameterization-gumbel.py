"""
Figure generation for standalone post:
  "Reparameterization Trick & Gumbel-Softmax — A Deep Dive"

Outputs five figures into BOTH the EN and ZH asset folders:
  fig1_reparam_trick.png         Reparameterization trick (VAE-style)
  fig2_gumbel_pdf.png            Gumbel(0,1) distribution: PDF/CDF + samples
  fig3_gumbel_max_trick.png      Gumbel-Max trick: sampling from a categorical
  fig4_gumbel_softmax_temp.png   Gumbel-Softmax with temperature tau effect
  fig5_discrete_pipeline.png     Differentiable discrete sampling pipeline
                                 (gradient flow comparison)

Run:
    python reparameterization-gumbel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "grid.color": "#e2e8f0",
        "grid.alpha": 0.6,
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GREY = "#64748b"
RED = "#ef4444"

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source/_posts/en/standalone/reparameterization-gumbel-softmax"
ZH_DIR = (
    REPO_ROOT
    / "source/_posts/zh/standalone/重参数化详解与gumbel-softmax深入探讨"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (EN_DIR, ZH_DIR):
        fig.savefig(folder / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Reparameterization trick (VAE example: z = mu + sigma * eps)
# ---------------------------------------------------------------------------

def fig1_reparam_trick() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))

    def draw_node(ax, xy, text, color, w=1.5, h=0.75, fc=None):
        fc = fc if fc is not None else "white"
        box = FancyBboxPatch(
            (xy[0] - w / 2, xy[1] - h / 2),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=color,
            facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=10.5)

    def arrow(ax, p, q, color=GREY, style="-|>", lw=1.6, ls="-"):
        ax.add_patch(
            FancyArrowPatch(
                p, q, arrowstyle=style, mutation_scale=14, lw=lw,
                color=color, linestyle=ls,
            )
        )

    # ---------- Left: NAIVE (sample directly) ----------
    ax = axes[0]
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Naive: sample directly  z ~ N(mu, sigma^2)", pad=10)

    draw_node(ax, (1.0, 3.5), "x", BLUE, w=1.0)
    draw_node(ax, (3.2, 3.5), "Encoder", BLUE, w=1.8)
    draw_node(ax, (5.6, 4.6), "mu", PURPLE, w=1.0)
    draw_node(ax, (5.6, 2.4), "sigma", PURPLE, w=1.0)
    # Stochastic sampling node (red)
    draw_node(ax, (8.6, 3.5), "z ~ N(mu, sigma^2)", RED, w=3.0, fc="#fee2e2")
    draw_node(ax, (11.7, 3.5), "Decoder", BLUE, w=1.8)

    arrow(ax, (1.55, 3.5), (2.25, 3.5))
    arrow(ax, (4.15, 3.5), (5.05, 4.4))
    arrow(ax, (4.15, 3.5), (5.05, 2.6))
    arrow(ax, (6.15, 4.4), (7.15, 3.7))
    arrow(ax, (6.15, 2.6), (7.15, 3.3))
    arrow(ax, (10.15, 3.5), (10.75, 3.5))

    # Backward gradient blocked
    arrow(ax, (9.5, 1.4), (7.7, 1.4), color=RED, lw=2.2, ls="--")
    ax.text(8.6, 0.85, "gradient BLOCKED", color=RED, fontsize=10,
            ha="center", style="italic")
    ax.text(8.6, 0.25, "(non-differentiable sampling)",
            color=RED, fontsize=9, ha="center")

    # ---------- Right: REPARAMETERIZED ----------
    ax = axes[1]
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Reparameterized:  z = mu + sigma * epsilon,  eps ~ N(0, I)",
                 pad=10)

    draw_node(ax, (1.0, 3.5), "x", BLUE, w=1.0)
    draw_node(ax, (3.2, 3.5), "Encoder", BLUE, w=1.8)
    draw_node(ax, (5.6, 4.6), "mu", PURPLE, w=1.0)
    draw_node(ax, (5.6, 2.4), "sigma", PURPLE, w=1.0)
    # Noise node (deterministic in graph)
    draw_node(ax, (5.6, 6.2), "eps ~ N(0,I)", RED, w=2.0, fc="#fee2e2")
    draw_node(ax, (8.6, 3.5), "z = mu + sigma * eps", GREEN,
              w=3.0, fc="#d1fae5")
    draw_node(ax, (11.7, 3.5), "Decoder", BLUE, w=1.8)

    arrow(ax, (1.55, 3.5), (2.25, 3.5))
    arrow(ax, (4.15, 3.5), (5.05, 4.4))
    arrow(ax, (4.15, 3.5), (5.05, 2.6))
    arrow(ax, (6.15, 4.4), (7.15, 3.7))
    arrow(ax, (6.15, 2.6), (7.15, 3.3))
    arrow(ax, (5.95, 5.95), (7.25, 3.85), color=GREY, ls=":")
    arrow(ax, (10.15, 3.5), (10.75, 3.5))

    # Backward gradient flows through deterministic path
    arrow(ax, (10.4, 1.3), (3.2, 1.3), color=GREEN, lw=2.2)
    ax.text(6.8, 0.7, "gradient flows through  mu, sigma  (eps is just noise)",
            color=GREEN, fontsize=10, ha="center", style="italic")

    fig.suptitle("Reparameterization Trick: separate randomness from parameters",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig1_reparam_trick.png")


# ---------------------------------------------------------------------------
# Figure 2 — Gumbel(0, 1) distribution: PDF / CDF / samples
# ---------------------------------------------------------------------------

def fig2_gumbel_pdf() -> None:
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    # PDF
    ax = axes[0]
    x = np.linspace(-3, 6, 600)
    pdf = np.exp(-(x + np.exp(-x)))
    ax.plot(x, pdf, color=BLUE, lw=2.4, label="Gumbel(0, 1) PDF")
    ax.fill_between(x, 0, pdf, color=BLUE, alpha=0.12)
    ax.axvline(0, color=GREY, lw=0.8, ls=":")
    # Mode at x = 0; mean at gamma (Euler-Mascheroni)
    ax.axvline(0.5772, color=ORANGE, lw=1.4, ls="--", label="mean = gamma")
    ax.scatter([0], [np.exp(-1)], color=PURPLE, s=40, zorder=5,
               label="mode = 0")
    ax.set_xlabel("g")
    ax.set_ylabel("density")
    ax.set_title("PDF:  f(g) = exp(-(g + e^{-g}))")
    ax.legend(loc="upper right", fontsize=9)

    # CDF + inverse-CDF sampling demo
    ax = axes[1]
    cdf = np.exp(-np.exp(-x))
    ax.plot(x, cdf, color=PURPLE, lw=2.4, label="CDF  F(g) = e^{-e^{-g}}")
    # Show inverse-CDF sampling: u ~ U(0,1) -> g = -log(-log(u))
    u_samples = np.array([0.15, 0.4, 0.7, 0.92])
    g_samples = -np.log(-np.log(u_samples))
    for u, g in zip(u_samples, g_samples):
        ax.plot([x.min(), g], [u, u], color=ORANGE, lw=1.0, ls=":")
        ax.plot([g, g], [0, u], color=ORANGE, lw=1.0, ls=":")
        ax.scatter([g], [u], color=ORANGE, s=28, zorder=5)
    ax.set_xlabel("g")
    ax.set_ylabel("F(g)")
    ax.set_title("Inverse-CDF sampling:  g = -log(-log(u))")
    ax.legend(loc="lower right", fontsize=9)

    # Empirical histogram
    ax = axes[2]
    u = rng.uniform(1e-12, 1 - 1e-12, size=20000)
    samples = -np.log(-np.log(u))
    ax.hist(samples, bins=80, density=True, color=GREEN, alpha=0.5,
            edgecolor="white", label="20k samples via -log(-log u)")
    ax.plot(x, pdf, color=BLUE, lw=2.2, label="theoretical PDF")
    ax.set_xlim(-3, 6)
    ax.set_xlabel("g")
    ax.set_ylabel("density")
    ax.set_title("Empirical histogram matches PDF")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Gumbel(0, 1) distribution: shape, sampling, empirical fit",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    save(fig, "fig2_gumbel_pdf.png")


# ---------------------------------------------------------------------------
# Figure 3 — Gumbel-Max trick (sampling from categorical)
# ---------------------------------------------------------------------------

def fig3_gumbel_max_trick() -> None:
    rng = np.random.default_rng(7)

    K = 5
    classes = [f"c{i+1}" for i in range(K)]
    logits = np.array([1.2, 2.0, 0.5, 1.6, 0.8])
    probs = np.exp(logits) / np.exp(logits).sum()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # Panel A: target categorical distribution
    ax = axes[0]
    bars = ax.bar(classes, probs, color=BLUE, alpha=0.85, edgecolor="white")
    ax.set_ylim(0, max(probs) * 1.25)
    ax.set_ylabel("probability")
    ax.set_title("Target categorical  pi_i = softmax(logits)_i")
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{p:.2f}", ha="center", fontsize=9, color=GREY)

    # Panel B: one Gumbel-Max draw
    ax = axes[1]
    g = -np.log(-np.log(rng.uniform(1e-12, 1 - 1e-12, size=K)))
    perturbed = logits + g
    argmax = int(np.argmax(perturbed))

    x = np.arange(K)
    ax.bar(x - 0.18, logits, width=0.36, color=BLUE, label="logits")
    ax.bar(x + 0.18, perturbed, width=0.36, color=ORANGE,
           label="logits + Gumbel noise")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.axhline(0, color=GREY, lw=0.6)
    # Highlight the argmax bar
    ax.bar(x[argmax] + 0.18, perturbed[argmax], width=0.36,
           color=GREEN, edgecolor=RED, lw=2, label=f"argmax = {classes[argmax]}")
    ax.set_title("One draw:  argmax_i (logits_i + g_i)")
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel C: empirical frequency vs target after many draws
    ax = axes[2]
    N = 8000
    G = -np.log(-np.log(rng.uniform(1e-12, 1 - 1e-12, size=(N, K))))
    draws = np.argmax(logits + G, axis=1)
    freq = np.bincount(draws, minlength=K) / N

    width = 0.36
    ax.bar(np.arange(K) - width / 2, probs, width, color=BLUE,
           alpha=0.85, label="target softmax")
    ax.bar(np.arange(K) + width / 2, freq, width, color=GREEN,
           alpha=0.85, label=f"empirical ({N} draws)")
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(classes)
    ax.set_ylabel("probability")
    ax.set_title("Gumbel-Max recovers softmax (exact in expectation)")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Gumbel-Max trick: argmax over logits + Gumbel noise = softmax sample",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    save(fig, "fig3_gumbel_max_trick.png")


# ---------------------------------------------------------------------------
# Figure 4 — Gumbel-Softmax with temperature tau
# ---------------------------------------------------------------------------

def fig4_gumbel_softmax_temp() -> None:
    rng = np.random.default_rng(42)

    K = 5
    classes = [f"c{i+1}" for i in range(K)]
    logits = np.array([1.2, 2.0, 0.5, 1.6, 0.8])
    probs = np.exp(logits) / np.exp(logits).sum()

    taus = [5.0, 1.0, 0.5, 0.1]
    n_per_tau = 5  # number of independent samples to overlay

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, tau in zip(axes, taus):
        # Average distribution and a few example samples
        for k in range(n_per_tau):
            g = -np.log(-np.log(rng.uniform(1e-12, 1 - 1e-12, size=K)))
            y = np.exp((logits + g) / tau)
            y = y / y.sum()
            ax.plot(np.arange(K), y, color=ORANGE, alpha=0.55,
                    marker="o", lw=1.4, ms=5)

        ax.bar(np.arange(K), probs, color=BLUE, alpha=0.28,
               label="softmax(logits) (target)")
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("y_i")
        ax.set_title(f"tau = {tau}")
        ax.legend(loc="upper right", fontsize=9)

        if tau >= 5:
            ax.text(0.02, 0.95, "high tau:\nnear-uniform, smooth",
                    transform=ax.transAxes, fontsize=9, color=GREY,
                    va="top")
        elif tau <= 0.1:
            ax.text(0.02, 0.95, "low tau:\nnear one-hot, low bias\nbut high variance",
                    transform=ax.transAxes, fontsize=9, color=GREY,
                    va="top")
        elif tau == 1.0:
            ax.text(0.02, 0.95, "tau=1:\nbalanced; smooth with\nsome stochasticity",
                    transform=ax.transAxes, fontsize=9, color=GREY,
                    va="top")
        else:
            ax.text(0.02, 0.95, "tau=0.5:\nsharper; closer to one-hot",
                    transform=ax.transAxes, fontsize=9, color=GREY,
                    va="top")

    fig.suptitle(
        "Gumbel-Softmax samples y = softmax((logits + g) / tau)  for various tau\n"
        "tau small -> discrete (low bias, high variance);  tau large -> smooth "
        "(high bias, low variance)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save(fig, "fig4_gumbel_softmax_temp.png")


# ---------------------------------------------------------------------------
# Figure 5 — Differentiable discrete sampling pipeline + STE comparison
# ---------------------------------------------------------------------------

def fig5_discrete_pipeline() -> None:
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.25)

    # ----- Left: pipeline diagram -----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Differentiable discrete sampling pipeline", pad=10)

    def draw_node(xy, text, color, w=1.7, h=0.8, fc="white"):
        box = FancyBboxPatch(
            (xy[0] - w / 2, xy[1] - h / 2), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.6, edgecolor=color, facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=10)

    def arr(p, q, color=GREY, ls="-", lw=1.6):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>",
                                     mutation_scale=14, lw=lw,
                                     color=color, linestyle=ls))

    draw_node((1.3, 4.2), "Logits\nlogits", BLUE, w=1.6)
    draw_node((3.7, 4.2), "+ Gumbel\nnoise g", PURPLE, w=1.8, fc="#ede9fe")
    draw_node((6.3, 4.2), "/ tau", PURPLE, w=1.4)
    draw_node((8.5, 4.2), "Softmax", BLUE, w=1.6)
    draw_node((11.0, 4.2), "y_soft\n(differentiable)", GREEN,
              w=2.0, fc="#d1fae5")

    arr((2.1, 4.2), (2.8, 4.2))
    arr((4.6, 4.2), (5.6, 4.2))
    arr((7.0, 4.2), (7.7, 4.2))
    arr((9.3, 4.2), (10.0, 4.2))

    # Straight-through branch
    draw_node((8.5, 1.5), "argmax", ORANGE, w=1.6, fc="#fef3c7")
    draw_node((11.0, 1.5), "y_hard\n(one-hot)", ORANGE, w=2.0, fc="#fef3c7")
    arr((8.5, 3.7), (8.5, 1.95))
    arr((9.3, 1.5), (10.0, 1.5))

    # STE: forward hard, backward soft
    arr((11.0, 1.95), (11.0, 3.7), color=GREEN, lw=2.0, ls="--")
    ax.text(11.0, 5.6, "STE:\nforward = y_hard\nbackward = y_soft",
            color=GREEN, fontsize=8.5, ha="center", va="center")

    ax.text(6.0, 0.4,
            "y = softmax((logits + g) / tau);  g_i = -log(-log u_i),  u_i ~ U(0,1)",
            ha="center", fontsize=10, color=GREY, style="italic")

    # ----- Right: gradient variance comparison (REINFORCE vs Gumbel-Softmax) -----
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)

    K = 8
    logits = rng.normal(0, 1.0, size=K)
    probs = np.exp(logits) / np.exp(logits).sum()
    rewards = rng.normal(0, 1.0, size=K)  # synthetic reward per class

    sample_sizes = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    n_trials = 200

    var_reinforce, var_gs = [], []

    for n in sample_sizes:
        rf = []
        gs_ = []
        for _ in range(n_trials):
            # REINFORCE: gradient w.r.t. logit_0 ~ (R - b) * (one_hot(s) - probs)
            samples = rng.choice(K, size=n, p=probs)
            r_vals = rewards[samples]
            grad_rf = np.mean(r_vals * ((samples == 0).astype(float) - probs[0]))
            rf.append(grad_rf)

            # Gumbel-Softmax estimator: differentiate sum_i r_i * y_i wrt logit_0
            tau = 0.5
            g = -np.log(-np.log(rng.uniform(1e-12, 1 - 1e-12, size=(n, K))))
            y = np.exp((logits + g) / tau)
            y = y / y.sum(axis=1, keepdims=True)
            # d/d logits_0 of (sum_i r_i y_i) = (1/tau) * sum_i r_i y_i (delta_{i0} - y_0)
            grad_gs = np.mean(
                (1.0 / tau) * np.sum(rewards * y *
                                     (np.eye(K)[0] - y[:, 0:1]), axis=1)
            )
            gs_.append(grad_gs)

        var_reinforce.append(np.var(rf))
        var_gs.append(np.var(gs_))

    ax.loglog(sample_sizes, var_reinforce, "o-", color=ORANGE, lw=2,
              ms=6, label="REINFORCE (score function)")
    ax.loglog(sample_sizes, var_gs, "s-", color=GREEN, lw=2,
              ms=6, label="Gumbel-Softmax (tau=0.5)")
    ax.set_xlabel("samples per gradient estimate (n)")
    ax.set_ylabel("variance of gradient estimate")
    ax.set_title("Gumbel-Softmax: lower variance than REINFORCE")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, which="both", alpha=0.4)

    fig.suptitle(
        "Application: differentiable discrete sampling end-to-end",
        fontsize=14, fontweight="bold", y=1.02,
    )
    save(fig, "fig5_discrete_pipeline.png")


def main() -> None:
    fig1_reparam_trick()
    fig2_gumbel_pdf()
    fig3_gumbel_max_trick()
    fig4_gumbel_softmax_temp()
    fig5_discrete_pipeline()
    print(f"Wrote figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
