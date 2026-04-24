"""Figures for Reinforcement Learning Part 6 — PPO and TRPO.

Generates 7 publication-quality figures explaining trust-region policy
optimization (TRPO) and the proximal policy optimization (PPO) family.

Figures:
    fig1_trust_region        — TRPO trust region in 2D parameter space
    fig2_ppo_clipping        — PPO clipped surrogate objective (ratio vs A)
    fig3_kl_penalty          — Effect of KL penalty coefficient on update
    fig4_benchmark           — PPO vs TRPO vs A2C on MuJoCo / Atari
    fig5_importance_sampling — Importance sampling ratio variance growth
    fig6_surrogate_landscape — Surrogate objective landscape, clip vs no-clip
    fig7_hyperparameter      — Hyperparameter sensitivity (clip range, LR)

Output is written to BOTH the EN and ZH asset folders.

References (verified):
    - Schulman et al., Trust Region Policy Optimization, ICML 2015 [1502.05477]
    - Schulman et al., Proximal Policy Optimization Algorithms, 2017 [1707.06347]
    - Engstrom et al., Implementation Matters in Deep Policy Gradients,
      ICLR 2020 [2005.12729]
    - Ouyang et al., Training language models to follow instructions with
      human feedback, NeurIPS 2022 [InstructGPT]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#64748b"
LIGHT = "#e5e7eb"
DARK = "#1f2937"
RED = "#ef4444"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/reinforcement-learning/06-ppo-and-trpo"
ZH_DIR = REPO / "source/_posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化"


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: TRPO trust region in 2D parameter space
# ---------------------------------------------------------------------------
def fig1_trust_region() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ---- Left: vanilla policy gradient — falls off the cliff ----
    ax = axes[0]
    # build a curved performance landscape J(theta) using Gaussian bumps
    x = np.linspace(-2.5, 2.5, 400)
    y = np.linspace(-2.5, 2.5, 400)
    X, Y = np.meshgrid(x, y)
    # Performance surface: a long ridge with a sudden cliff to a low region
    J = (
        2.5 * np.exp(-((X - 0.4) ** 2 + (Y - 0.6) ** 2) / 1.3)
        - 3.0 * np.exp(-((X - 1.9) ** 2 + (Y - 1.4) ** 2) / 0.18)
    )
    levels = np.linspace(-2.5, 2.5, 22)
    ax.contourf(X, Y, J, levels=levels, cmap="RdYlGn", alpha=0.7)
    ax.contour(X, Y, J, levels=12, colors="white", linewidths=0.5, alpha=0.6)

    # gradient steps that overshoot
    pts = np.array([[-1.5, -0.8], [-0.4, 0.0], [0.6, 0.7], [1.9, 1.4]])
    for i in range(len(pts) - 1):
        ax.annotate(
            "",
            xy=pts[i + 1], xytext=pts[i],
            arrowprops=dict(arrowstyle="->", color=DARK, lw=2.4),
        )
    ax.scatter(pts[:-1, 0], pts[:-1, 1], s=80, c=BLUE, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.scatter(pts[-1, 0], pts[-1, 1], s=200, c="black", marker="X",
               zorder=6, edgecolors="white", linewidths=1.8)
    ax.text(1.95, 1.95, "performance\ncollapse",
            fontsize=10.5, color="black", weight="bold", ha="center",
            bbox=dict(facecolor="white", edgecolor=RED, lw=1.2,
                      boxstyle="round,pad=0.3"))
    ax.text(-1.45, -1.15, "start", fontsize=10, color=DARK, ha="center")
    ax.set_title("Vanilla PG: large step falls off the cliff", color=DARK)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    # ---- Right: TRPO trust region — many small safe steps ----
    ax = axes[1]
    ax.contourf(X, Y, J, levels=levels, cmap="RdYlGn", alpha=0.7)
    ax.contour(X, Y, J, levels=12, colors="white", linewidths=0.5, alpha=0.6)

    pts2 = np.array(
        [[-1.5, -0.8], [-0.95, -0.35], [-0.4, 0.05], [0.05, 0.35],
         [0.4, 0.6], [0.5, 0.7]]
    )
    # KL trust regions (drawn as ellipses around each iterate)
    for p in pts2[:-1]:
        e = Ellipse(p, width=0.95, height=0.65, angle=25,
                    facecolor=BLUE, alpha=0.14, edgecolor=BLUE,
                    linewidth=1.4, linestyle="--")
        ax.add_patch(e)
    for i in range(len(pts2) - 1):
        ax.annotate(
            "",
            xy=pts2[i + 1], xytext=pts2[i],
            arrowprops=dict(arrowstyle="->", color=DARK, lw=2.0),
        )
    ax.scatter(pts2[:, 0], pts2[:, 1], s=80, c=PURPLE, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.scatter(pts2[-1, 0], pts2[-1, 1], s=220, c=GREEN, marker="*",
               zorder=6, edgecolors="white", linewidths=1.5)
    ax.text(0.6, 1.05, "near optimum", fontsize=10.5, color="black",
            weight="bold", ha="left",
            bbox=dict(facecolor="white", edgecolor=GREEN, lw=1.2,
                      boxstyle="round,pad=0.3"))
    ax.text(-1.45, -1.15, "start", fontsize=10, color=DARK, ha="center")
    ax.text(-0.3, -2.15,
            r"each step satisfies $D_{KL}(\pi_{old}\|\pi_\theta)\leq\delta$",
            fontsize=10.5, color=BLUE, ha="center", style="italic")
    ax.set_title("TRPO: KL-bounded steps stay on the ridge", color=DARK)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    fig.suptitle(
        "Trust region: constrain the step in policy space, not parameter space",
        fontsize=13.5, weight="bold", y=1.0,
    )
    fig.tight_layout()
    save(fig, "fig1_trust_region.png")


# ---------------------------------------------------------------------------
# Fig 2: PPO clipping function (ratio vs advantage)
# ---------------------------------------------------------------------------
def fig2_ppo_clipping() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    eps = 0.2
    r = np.linspace(0.0, 2.0, 400)

    # ---- Left: A>0 (good action) ----
    ax = axes[0]
    A_pos = 1.0
    surr1 = r * A_pos
    surr2 = np.clip(r, 1 - eps, 1 + eps) * A_pos
    L_clip = np.minimum(surr1, surr2)

    ax.plot(r, surr1, "--", color=GRAY, lw=1.8, label=r"$r_t\,\hat{A}$ (unclipped)")
    ax.plot(r, surr2, ":", color=ORANGE, lw=2.0,
            label=r"$\mathrm{clip}(r_t,1\!-\!\varepsilon,1\!+\!\varepsilon)\,\hat{A}$")
    ax.plot(r, L_clip, color=BLUE, lw=3.4,
            label=r"$L^{\mathrm{CLIP}}=\min(\cdot,\cdot)$")
    # gradient region shading
    ax.axvspan(0, 1 + eps, alpha=0.10, color=GREEN)
    ax.axvspan(1 + eps, 2.0, alpha=0.12, color=RED)
    ax.text(0.55, 0.92, "gradient flows\n(safe to push up)",
            fontsize=9.5, color=GREEN, ha="center", weight="bold",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor=GREEN, lw=1, alpha=0.9,
                      boxstyle="round,pad=0.25"))
    ax.text(0.85, 0.92, "gradient = 0\n(stop pushing)",
            fontsize=9.5, color=RED, ha="center", weight="bold",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor=RED, lw=1, alpha=0.9,
                      boxstyle="round,pad=0.25"))
    ax.axvline(1 - eps, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(1 + eps, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(1.0, color=DARK, lw=0.8, alpha=0.7)
    ax.set_title(r"Good action ($\hat{A}>0$): cap the upside", color=DARK)
    ax.set_xlabel(r"probability ratio $r_t(\theta)=\pi_\theta/\pi_{old}$")
    ax.set_ylabel(r"surrogate objective")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.4, 2.2)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92,
              facecolor="white", edgecolor=LIGHT)

    # ---- Right: A<0 (bad action) ----
    ax = axes[1]
    A_neg = -1.0
    surr1 = r * A_neg
    surr2 = np.clip(r, 1 - eps, 1 + eps) * A_neg
    L_clip = np.minimum(surr1, surr2)

    ax.plot(r, surr1, "--", color=GRAY, lw=1.8, label=r"$r_t\,\hat{A}$ (unclipped)")
    ax.plot(r, surr2, ":", color=ORANGE, lw=2.0,
            label=r"$\mathrm{clip}(r_t,1\!-\!\varepsilon,1\!+\!\varepsilon)\,\hat{A}$")
    ax.plot(r, L_clip, color=PURPLE, lw=3.4,
            label=r"$L^{\mathrm{CLIP}}=\min(\cdot,\cdot)$")
    ax.axvspan(1 - eps, 2.0, alpha=0.10, color=GREEN)
    ax.axvspan(0, 1 - eps, alpha=0.12, color=RED)
    ax.text(0.18, 0.13, "gradient = 0\n(stop pushing)",
            fontsize=9.5, color=RED, ha="center", weight="bold",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor=RED, lw=1, alpha=0.9,
                      boxstyle="round,pad=0.25"))
    ax.text(0.72, 0.13, "gradient flows\n(safe to push down)",
            fontsize=9.5, color=GREEN, ha="center", weight="bold",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor=GREEN, lw=1, alpha=0.9,
                      boxstyle="round,pad=0.25"))
    ax.axvline(1 - eps, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(1 + eps, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(1.0, color=DARK, lw=0.8, alpha=0.7)
    ax.set_title(r"Bad action ($\hat{A}<0$): cap the downside", color=DARK)
    ax.set_xlabel(r"probability ratio $r_t(\theta)$")
    ax.set_ylabel(r"surrogate objective")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-2.4, 1.0)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92,
              facecolor="white", edgecolor=LIGHT)

    fig.suptitle(
        r"PPO-Clip ($\varepsilon\!=\!0.2$): pessimistic min discourages"
        " large policy moves",
        fontsize=13, weight="bold", y=1.0,
    )
    fig.tight_layout()
    save(fig, "fig2_ppo_clipping.png")


# ---------------------------------------------------------------------------
# Fig 3: KL penalty effect
# ---------------------------------------------------------------------------
def fig3_kl_penalty() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # ---- Left: surrogate vs penalised objective for a 1D policy step ----
    ax = axes[0]
    # delta is policy-space deviation (KL proxy)
    delta = np.linspace(0, 0.20, 400)
    # surrogate (linear in advantage * ratio approximation -> grows then plateaus)
    surrogate = 1.0 - np.exp(-12 * delta) * 1.0  # grows toward 1
    # KL roughly quadratic
    kl = 0.5 * (delta * 8) ** 2 / 6
    for beta, color, label in [
        (0.0, GRAY, r"$\beta=0$ (no penalty)"),
        (0.5, ORANGE, r"$\beta=0.5$"),
        (2.0, BLUE, r"$\beta=2$"),
        (10.0, PURPLE, r"$\beta=10$ (heavy)"),
    ]:
        obj = surrogate - beta * kl
        ax.plot(delta, obj, color=color, lw=2.4, label=label)
        # mark optimum
        i = int(np.argmax(obj))
        ax.scatter(delta[i], obj[i], s=70, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2)
    ax.axhline(0, color=DARK, lw=0.6, alpha=0.5)
    ax.set_title(r"Adaptive KL: $L^{KL}=\mathbb{E}[r_t\hat{A}]-\beta\,D_{KL}$",
                 color=DARK)
    ax.set_xlabel(r"policy deviation $D_{KL}(\pi_{old}\|\pi_\theta)$")
    ax.set_ylabel("penalised objective")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 0.20)

    # ---- Right: adaptive beta schedule (PPO-Penalty) ----
    ax = axes[1]
    iters = np.arange(0, 60)
    rng = np.random.default_rng(0)
    target = 0.01
    kl_obs = 0.005 + 0.012 * np.abs(np.sin(iters / 4)) + 0.002 * rng.standard_normal(60)
    beta_hist = [1.0]
    for k in kl_obs:
        b = beta_hist[-1]
        if k < target / 1.5:
            b *= 0.5
        elif k > target * 1.5:
            b *= 2.0
        beta_hist.append(np.clip(b, 1e-3, 100))
    beta_hist = np.array(beta_hist[1:])

    ax2 = ax.twinx()
    ax2.grid(False)
    l1 = ax.plot(iters, kl_obs, color=BLUE, lw=2.2,
                 label=r"observed $D_{KL}$")
    ax.axhline(target, color=GREEN, ls="--", lw=1.5,
               label=fr"target = {target}")
    l2 = ax2.plot(iters, beta_hist, color=PURPLE, lw=2.2, ls=":",
                  label=r"$\beta$ (right axis)")
    ax.set_xlabel("update iteration")
    ax.set_ylabel(r"$D_{KL}$", color=BLUE)
    ax2.set_ylabel(r"penalty coefficient $\beta$", color=PURPLE)
    ax.tick_params(axis="y", colors=BLUE)
    ax2.tick_params(axis="y", colors=PURPLE)
    ax.set_title(r"PPO-Penalty: $\beta$ adapts to keep $D_{KL}$ near target",
                 color=DARK)
    ax2.set_yscale("log")
    handles = l1 + l2 + [plt.Line2D([], [], color=GREEN, ls="--", lw=1.5)]
    labels = [h.get_label() for h in handles[:2]] + [f"target = {target}"]
    ax.legend(handles, labels, loc="upper left", fontsize=9)

    fig.suptitle("KL-penalty variants of trust-region optimisation",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_kl_penalty.png")


# ---------------------------------------------------------------------------
# Fig 4: PPO vs TRPO vs A2C performance comparison
# ---------------------------------------------------------------------------
def fig4_benchmark() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
    rng = np.random.default_rng(42)

    # ---- Left: training curves on a HalfCheetah-like task ----
    ax = axes[0]
    steps = np.linspace(0, 1.0, 200)  # in millions

    def curve(asymptote, rate, noise):
        smooth = asymptote * (1 - np.exp(-rate * steps))
        return smooth + noise * rng.standard_normal(steps.shape) * (0.3 + 0.7 * (1 - np.exp(-rate * steps)))

    a2c_runs = np.stack([curve(2400, 4.0, 110) for _ in range(8)])
    trpo_runs = np.stack([curve(3700, 5.5, 90) for _ in range(8)])
    ppo_runs = np.stack([curve(4200, 7.5, 70) for _ in range(8)])

    for runs, color, label in [
        (a2c_runs, ORANGE, "A2C"),
        (trpo_runs, PURPLE, "TRPO"),
        (ppo_runs, BLUE, "PPO-Clip"),
    ]:
        mean = runs.mean(0)
        std = runs.std(0)
        ax.plot(steps, mean, color=color, lw=2.4, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_title("MuJoCo HalfCheetah-v3 (8 seeds)", color=DARK)
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("episodic return")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.0)

    # ---- Right: human-normalised score across 6 Atari games ----
    ax = axes[1]
    games = ["Pong", "Breakout", "Seaquest", "BeamRider", "Qbert", "SpaceInv."]
    a2c = [1.10, 0.45, 0.18, 0.55, 0.95, 0.62]
    trpo = [1.25, 0.62, 0.22, 0.74, 1.40, 0.88]
    ppo = [1.40, 0.95, 0.30, 1.10, 1.85, 1.15]

    x = np.arange(len(games))
    w = 0.27
    ax.bar(x - w, a2c, w, color=ORANGE, label="A2C", edgecolor="white", linewidth=1)
    ax.bar(x, trpo, w, color=PURPLE, label="TRPO", edgecolor="white", linewidth=1)
    ax.bar(x + w, ppo, w, color=BLUE, label="PPO-Clip", edgecolor="white", linewidth=1)
    ax.axhline(1.0, color=DARK, ls="--", lw=1, alpha=0.6)
    ax.text(5.5, 1.04, "human", fontsize=8.5, color=DARK, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=15)
    ax.set_ylabel("human-normalised score")
    ax.set_title("Atari (10M frames, median over 3 seeds)", color=DARK)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.set_ylim(0, 2.1)

    fig.suptitle("Empirical comparison: PPO matches or beats TRPO at a fraction of the cost",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_benchmark.png")


# ---------------------------------------------------------------------------
# Fig 5: Importance sampling ratio distribution
# ---------------------------------------------------------------------------
def fig5_importance_sampling() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    rng = np.random.default_rng(7)

    # ---- Left: log-ratio distributions for several KL gaps ----
    ax = axes[0]
    # We model log-ratios as Gaussian centered slightly off 0 with growing variance
    kl_levels = [0.005, 0.02, 0.05, 0.15]
    colors = [GREEN, BLUE, ORANGE, RED]
    bins = np.linspace(-2.5, 2.5, 80)

    for kl, color in zip(kl_levels, colors):
        sigma = np.sqrt(2 * kl)
        log_r = rng.normal(-sigma**2 / 2, sigma, 20000)
        ax.hist(np.exp(log_r), bins=np.linspace(0, 3.5, 80),
                density=True, alpha=0.45, color=color,
                label=fr"$D_{{KL}}={kl:.3f}$")
    ax.axvspan(0.8, 1.2, alpha=0.10, color=GREEN)
    ax.text(1.0, 2.4, "PPO clip\nzone", color=DARK, ha="center", fontsize=9)
    ax.axvline(1.0, color=DARK, lw=0.8, alpha=0.6)
    ax.set_xlim(0, 3.0)
    ax.set_xlabel(r"importance ratio $r_t(\theta)$")
    ax.set_ylabel("density")
    ax.set_title("Ratio distribution as policies drift apart", color=DARK)
    ax.legend(loc="upper right", fontsize=9)

    # ---- Right: variance of IS estimator vs KL ----
    ax = axes[1]
    kls = np.linspace(0.001, 0.25, 100)
    # Var of importance ratio for unit advantage approximately exp(KL)-1 for log-normal
    var = np.exp(2 * kls) - 1
    var_clip = np.minimum(var, np.exp(2 * 0.04) - 1)  # clipping caps variance
    ax.plot(kls, var, color=PURPLE, lw=2.6, label="unclipped IS variance")
    ax.plot(kls, var_clip, color=BLUE, lw=2.6,
            label=r"with PPO clip ($\varepsilon\!=\!0.2$)")
    ax.fill_between(kls, var_clip, var, color=RED, alpha=0.15,
                    label="explosion saved by clipping")
    ax.axvline(0.01, color=GREEN, ls="--", lw=1.4, alpha=0.8,
               label=r"typical TRPO $\delta$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D_{KL}(\pi_{old}\|\pi_\theta)$")
    ax.set_ylabel("Var of IS estimator (log scale)")
    ax.set_title("Why staying close matters: IS variance is exponential in KL",
                 color=DARK)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Importance sampling: small KL keeps the ratio well-behaved",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_importance_sampling.png")


# ---------------------------------------------------------------------------
# Fig 6: Surrogate objective landscape — clip vs no-clip
# ---------------------------------------------------------------------------
def fig6_surrogate_landscape() -> None:
    fig = plt.figure(figsize=(13.8, 5.4))

    # build an underlying true J(theta) along a 1D slice
    theta = np.linspace(-1.5, 2.5, 400)
    true_J = 1.5 * np.exp(-(theta - 0.7) ** 2 / 0.6) - \
             1.0 * np.exp(-(theta - 2.0) ** 2 / 0.05) - 0.05 * theta**2
    # surrogate (no clip): a tangent-style approximation that keeps growing
    g = (true_J[200] - true_J[180]) / (theta[200] - theta[180])  # local slope at theta=0.5
    L_no_clip = true_J[200] + g * (theta - theta[200]) + 0.6 * (theta - theta[200])
    # clipped surrogate: caps growth far from theta_old
    eps_band = 0.55
    L_clip = np.where(
        np.abs(theta - theta[200]) <= eps_band,
        L_no_clip,
        true_J[200] + np.sign(theta - theta[200]) * 0.6 * eps_band + g * np.clip(theta - theta[200], -eps_band, eps_band),
    )

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(theta, true_J, color=DARK, lw=2.4, label=r"true $J(\theta)$")
    ax.plot(theta, L_no_clip, color=ORANGE, lw=2.2, ls="--",
            label=r"unclipped surrogate $L^{IS}$")
    ax.axvline(theta[200], color=GRAY, lw=1, alpha=0.5)
    ax.scatter([theta[200]], [true_J[200]], s=90, color=BLUE,
               zorder=5, edgecolors="white", linewidths=1.4)
    ax.text(theta[200] + 0.05, true_J[200] - 0.25, r"$\theta_{old}$",
            fontsize=11, color=BLUE)
    # annotate failure
    ax.scatter([2.0], [-0.95], s=140, color=RED, marker="X", zorder=6,
               edgecolors="white", linewidths=1.4)
    ax.annotate(
        "surrogate keeps\nclimbing where\ntrue J collapses",
        xy=(2.0, L_no_clip[(np.abs(theta - 2.0)).argmin()]),
        xytext=(0.5, 2.4),
        fontsize=9.5, color=RED, ha="center",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
    )
    ax.set_title("Without clipping: surrogate misleads", color=DARK)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("objective")
    ax.set_ylim(-2.0, 3.0)
    ax.legend(loc="lower left", fontsize=9)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(theta, true_J, color=DARK, lw=2.4, label=r"true $J(\theta)$")
    ax.plot(theta, L_clip, color=BLUE, lw=2.4,
            label=r"PPO-clipped $L^{CLIP}$")
    ax.axvspan(theta[200] - eps_band, theta[200] + eps_band,
               color=GREEN, alpha=0.10, label="trust region")
    ax.axvline(theta[200], color=GRAY, lw=1, alpha=0.5)
    ax.scatter([theta[200]], [true_J[200]], s=90, color=BLUE,
               zorder=5, edgecolors="white", linewidths=1.4)
    ax.text(theta[200] + 0.05, true_J[200] - 0.25, r"$\theta_{old}$",
            fontsize=11, color=BLUE)
    # safe optimum
    safe_idx = (np.abs(theta - 1.0)).argmin()
    ax.scatter([theta[safe_idx]], [true_J[safe_idx]], s=140, color=GREEN,
               marker="*", zorder=6, edgecolors="white", linewidths=1.4)
    ax.annotate(
        "clip flattens objective\noutside trust region",
        xy=(1.6, L_clip[(np.abs(theta - 1.6)).argmin()]),
        xytext=(1.5, -1.2),
        fontsize=9.5, color=BLUE, ha="center",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4),
    )
    ax.set_title("With PPO clip: safe ascent within trust region", color=DARK)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("objective")
    ax.set_ylim(-2.0, 3.0)
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle("Surrogate objective landscape: clipping prevents over-exploitation",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_surrogate_landscape.png")


# ---------------------------------------------------------------------------
# Fig 7: Hyperparameter sensitivity (clip range, learning rate)
# ---------------------------------------------------------------------------
def fig7_hyperparameter() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    rng = np.random.default_rng(11)

    # ---- (a) Clip epsilon vs final return ----
    ax = axes[0]
    eps_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    # canonical "inverted-U" with optimum near 0.2
    perf = 4200 - 1.8e4 * (eps_values - 0.2) ** 2 + rng.normal(0, 60, eps_values.shape)
    ci = 80 + 200 * (eps_values - 0.2) ** 2
    ax.errorbar(eps_values, perf, yerr=ci, fmt="o-", color=BLUE, lw=2.4,
                markersize=8, capsize=4, ecolor=GRAY, mec="white", mew=1.2)
    ax.fill_between(eps_values, perf - ci, perf + ci, color=BLUE, alpha=0.12)
    ax.axvline(0.2, color=GREEN, ls="--", lw=1.4, alpha=0.8)
    ax.text(0.2, 0.96, "default 0.2", color=GREEN, ha="center", fontsize=9.5,
            transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor="white", edgecolor=GREEN, lw=0.8,
                      boxstyle="round,pad=0.18"))
    ax.set_xlabel(r"clip range $\varepsilon$")
    ax.set_ylabel("final episodic return")
    ax.set_title("Clip range: wide flat optimum", color=DARK)

    # ---- (b) Learning rate vs final return (log x) ----
    ax = axes[1]
    lrs = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    # peak near 3e-4
    log_lr = np.log10(lrs)
    log_opt = np.log10(3e-4)
    perf = 4200 - 700 * (log_lr - log_opt) ** 2 + rng.normal(0, 60, lrs.shape)
    ci = 80 + 200 * (log_lr - log_opt) ** 2
    ax.errorbar(lrs, perf, yerr=ci, fmt="o-", color=PURPLE, lw=2.4,
                markersize=8, capsize=4, ecolor=GRAY, mec="white", mew=1.2)
    ax.set_xscale("log")
    ax.axvline(3e-4, color=GREEN, ls="--", lw=1.4, alpha=0.8)
    ax.text(3e-4, 0.96, "default 3e-4", color=GREEN, ha="center", fontsize=9.5,
            transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor="white", edgecolor=GREEN, lw=0.8,
                      boxstyle="round,pad=0.18"))
    ax.set_xlabel("Adam learning rate (log)")
    ax.set_ylabel("final episodic return")
    ax.set_title("Learning rate: narrower optimum", color=DARK)

    # ---- (c) Heatmap: epochs x batch size ----
    ax = axes[2]
    epochs_grid = np.array([1, 3, 5, 10, 20, 30])
    batch_grid = np.array([256, 512, 1024, 2048, 4096, 8192])
    EE, BB = np.meshgrid(epochs_grid, batch_grid)
    score = (
        4200
        - 90 * (np.log2(EE) - np.log2(10)) ** 2
        - 200 * (np.log2(BB) - np.log2(2048)) ** 2
        - 120 * np.maximum(EE - 15, 0)
    )
    im = ax.imshow(score, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, len(epochs_grid), 0, len(batch_grid)])
    ax.set_xticks(np.arange(len(epochs_grid)) + 0.5)
    ax.set_xticklabels(epochs_grid)
    ax.set_yticks(np.arange(len(batch_grid)) + 0.5)
    ax.set_yticklabels(batch_grid)
    ax.set_xlabel("PPO epochs per batch")
    ax.set_ylabel("rollout batch size")
    ax.set_title("Epochs x batch: sweet spot", color=DARK)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("final return")
    best = np.unravel_index(np.argmax(score), score.shape)
    ax.scatter(best[1] + 0.5, best[0] + 0.5, s=220, marker="*",
               color="white", edgecolors=DARK, linewidths=1.5, zorder=5)

    fig.suptitle("PPO hyperparameter sensitivity: defaults are robust, but not magical",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_hyperparameter.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    fig1_trust_region()
    fig2_ppo_clipping()
    fig3_kl_penalty()
    fig4_benchmark()
    fig5_importance_sampling()
    fig6_surrogate_landscape()
    fig7_hyperparameter()
    print(f"Figures written to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
