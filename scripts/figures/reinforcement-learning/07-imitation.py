"""Figures for Reinforcement Learning Part 7 — Imitation Learning and Inverse RL.

Generates publication-quality figures explaining behavioral cloning,
DAgger, maximum-entropy IRL, and GAIL/AIRL adversarial imitation.

Figures:
    fig1_distribution_shift   — BC vs DAgger: state distributions and recovery
    fig2_expert_vs_learner    — Expert demos vs BC vs DAgger trajectories
    fig3_irl_recovery         — IRL: recovering reward landscape from behavior
    fig4_gail_architecture    — GAIL discriminator-generator block diagram
    fig5_compounding_error    — BC quadratic drift vs DAgger linear bound
    fig6_method_hierarchy     — BC -> DAgger -> IRL -> GAIL -> AIRL ladder
    fig7_sample_efficiency    — Sample efficiency: imitation vs RL on a task

Output is written to BOTH the EN and ZH asset folders.

References (verified):
    - Pomerleau, ALVINN, NIPS 1989.
    - Ross, Gordon & Bagnell, A Reduction of Imitation Learning to No-Regret
      Online Learning, AISTATS 2011 [1011.0686]
    - Ziebart et al., Maximum Entropy Inverse Reinforcement Learning,
      AAAI 2008.
    - Ho & Ermon, Generative Adversarial Imitation Learning, NIPS 2016
      [1606.03476]
    - Fu, Luo & Levine, Learning Robust Rewards with Adversarial Inverse
      Reinforcement Learning (AIRL), ICLR 2018 [1710.11248]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

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

BLUE = "#2563eb"     # expert / reference
PURPLE = "#7c3aed"   # IRL / structural method
GREEN = "#10b981"    # DAgger / corrected behaviour
ORANGE = "#f59e0b"   # BC / drifting behaviour
GRAY = "#64748b"
LIGHT = "#e5e7eb"
DARK = "#1f2937"
RED = "#ef4444"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/reinforcement-learning/07-imitation-learning"
ZH_DIR = REPO / "source/_posts/zh/reinforcement-learning/07-模仿学习与逆强化学习"


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: Distribution shift — BC vs DAgger state coverage
# ---------------------------------------------------------------------------
def fig1_distribution_shift() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    rng = np.random.default_rng(7)

    # Common 2D state space
    xs = np.linspace(-3.5, 3.5, 200)
    ys = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(xs, ys)

    # Expert distribution: a thin tube along a sinusoidal corridor
    def gauss(mx, my, sx, sy):
        return np.exp(-(((X - mx) / sx) ** 2 + ((Y - my) / sy) ** 2))

    centers_x = np.linspace(-3.0, 3.0, 9)
    expert = np.zeros_like(X)
    for cx in centers_x:
        cy = 1.2 * np.sin(0.9 * cx)
        expert += gauss(cx, cy, 0.45, 0.30)
    expert /= expert.max()

    # BC at test time: drifts off the corridor as errors compound
    bc = np.zeros_like(X)
    for cx in centers_x:
        cy = 1.2 * np.sin(0.9 * cx)
        # spread grows with x
        spread = 0.30 + 0.18 * (cx + 3.0)
        bc += 0.85 * gauss(cx, cy, 0.55 + 0.12 * (cx + 3.0), spread)
    # add a drift cloud away from the corridor
    bc += 0.7 * gauss(2.0, -2.0, 1.1, 0.8)
    bc += 0.5 * gauss(1.0, 2.0, 1.3, 0.9)
    bc /= bc.max()

    # DAgger: corridor is widened (training labels collected on visited states)
    dagger = np.zeros_like(X)
    for cx in centers_x:
        cy = 1.2 * np.sin(0.9 * cx)
        dagger += gauss(cx, cy, 0.85, 0.55)
    dagger /= dagger.max()

    titles = [
        ("Expert demonstrations $d_{\\pi^*}$", expert, BLUE),
        ("Behavioral cloning at test $d_{\\pi_\\theta}$", bc, ORANGE),
        ("DAgger: aggregated coverage", dagger, GREEN),
    ]

    for ax, (title, density, color) in zip(axes, titles):
        ax.contourf(X, Y, density, levels=12, cmap="Blues" if color == BLUE
                    else ("Oranges" if color == ORANGE else "Greens"),
                    alpha=0.85)
        # Draw the true corridor as a thin reference line
        cx_line = np.linspace(-3.3, 3.3, 200)
        cy_line = 1.2 * np.sin(0.9 * cx_line)
        ax.plot(cx_line, cy_line, color=DARK, lw=1.3, ls="--", alpha=0.7,
                label="expert corridor")

        # Sample dots
        if color == BLUE:
            n = 80
            x_s = rng.uniform(-3.2, 3.2, n)
            y_s = 1.2 * np.sin(0.9 * x_s) + rng.normal(0, 0.18, n)
        elif color == ORANGE:
            n = 80
            x_s = rng.uniform(-3.2, 3.2, n)
            y_s = 1.2 * np.sin(0.9 * x_s) + rng.normal(0, 0.25 + 0.18 * (x_s + 3) / 6, n)
            # Drift samples off the corridor
            x_drift = rng.uniform(0.0, 3.0, 30)
            y_drift = rng.uniform(-2.3, -1.0, 30)
            x_s = np.concatenate([x_s, x_drift])
            y_s = np.concatenate([y_s, y_drift])
        else:
            n = 110
            x_s = rng.uniform(-3.2, 3.2, n)
            y_s = 1.2 * np.sin(0.9 * x_s) + rng.normal(0, 0.45, n)

        ax.scatter(x_s, y_s, s=8, color=color, alpha=0.75,
                   edgecolors="white", linewidths=0.4)
        ax.set_title(title, color=DARK)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("state dim 1")
        ax.set_ylabel("state dim 2")
        ax.legend(loc="upper left", fontsize=8)

    # annotation arrow on BC plot
    ax = axes[1]
    arrow = FancyArrowPatch((1.6, -0.3), (2.0, -1.7),
                            arrowstyle="->", mutation_scale=18,
                            color=RED, lw=2)
    ax.add_patch(arrow)
    ax.text(2.1, -1.05, "drift into\nun-seen states",
            color=RED, fontsize=9, ha="left", weight="bold")

    fig.suptitle("Distribution shift: BC drifts off-support; DAgger relabels recovery states",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_bc_vs_dagger.png")


# ---------------------------------------------------------------------------
# Fig 2: Expert demonstrations vs learner trajectories
# ---------------------------------------------------------------------------
def fig2_expert_vs_learner() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    rng = np.random.default_rng(3)

    # Reference corridor: a winding path from start to goal
    t = np.linspace(0, 1, 200)
    cx = -4.5 + 9.0 * t
    cy = 1.6 * np.sin(2.4 * t * np.pi)

    # Goal region
    goal = Circle((4.5, 0.0), 0.45, color=GREEN, alpha=0.35,
                  ec=GREEN, lw=2)
    ax.add_patch(goal)
    ax.text(4.5, -0.95, "goal", color=GREEN, ha="center",
            fontsize=10, weight="bold")
    # Start
    ax.scatter([-4.5], [0.0], s=120, color=DARK, zorder=5)
    ax.text(-4.6, 0.45, "start", color=DARK, fontsize=10, weight="bold",
            ha="right")

    # Hazard zone (off-distribution)
    haz = FancyBboxPatch((1.2, -2.6), 3.0, 1.0,
                         boxstyle="round,pad=0.05",
                         linewidth=1.4, edgecolor=RED, facecolor=RED,
                         alpha=0.12)
    ax.add_patch(haz)
    ax.text(2.7, -2.1, "off-distribution / hazard", color=RED,
            ha="center", fontsize=9.5, weight="bold")

    # Expert demonstrations: tight around corridor
    for i in range(6):
        noise = rng.normal(0, 0.10, len(t))
        ax.plot(cx, cy + noise, color=BLUE, lw=1.4, alpha=0.55,
                label="expert" if i == 0 else None)

    # BC learner: starts on, drifts off after midpoint
    for i in range(4):
        drift = np.where(t > 0.45,
                         (t - 0.45) * rng.uniform(2.0, 3.5)
                         * (-1) ** rng.integers(0, 2),
                         0.0)
        noise = rng.normal(0, 0.18, len(t))
        traj = cy + noise + drift
        # one trajectory falls into hazard
        if i == 1:
            traj = cy + noise - 1.2 * np.maximum(t - 0.55, 0) ** 1.4 * 4
        ax.plot(cx, traj, color=ORANGE, lw=1.6, alpha=0.85,
                label="BC learner" if i == 0 else None)

    # DAgger learner: stays close after early correction
    for i in range(3):
        noise = rng.normal(0, 0.16, len(t))
        wobble = 0.25 * np.sin(6 * np.pi * t + i)
        traj = cy + noise + wobble * np.exp(-3 * t)
        ax.plot(cx, traj, color=GREEN, lw=1.7, alpha=0.9,
                label="DAgger learner" if i == 0 else None)

    # Annotations
    ax.annotate("compounding\nerror", xy=(3.5, -2.0), xytext=(3.2, 1.8),
                fontsize=10, color=RED, ha="center", weight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.annotate("DAgger relabels\nrecovery states", xy=(0.5, -0.7),
                xytext=(-1.2, -2.2), fontsize=9.5, color=GREEN, ha="center",
                weight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))

    ax.set_xlim(-5.2, 5.6)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel("position x")
    ax.set_ylabel("position y")
    ax.set_title("Expert vs learner trajectories: BC drifts late, DAgger holds the corridor",
                 color=DARK)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig2_expert_trajectory_comparison.png")


# ---------------------------------------------------------------------------
# Fig 3: Inverse RL — recovering a reward landscape from behaviour
# ---------------------------------------------------------------------------
def fig3_irl_recovery() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    xs = np.linspace(-3, 3, 200)
    ys = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(xs, ys)

    def gauss(cx, cy, s, h=1.0):
        return h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2))

    # True reward: two attractors plus a soft penalty region
    true_r = (gauss(1.8, 1.2, 0.9, 1.0)
              + gauss(-1.5, -1.2, 0.7, 0.85)
              - gauss(0.0, 0.0, 0.55, 0.6))

    # Expert behaviour density: concentrates near reward peaks
    expert_d = np.exp(true_r * 2.2)
    expert_d /= expert_d.max()

    # Recovered reward: noisier but qualitatively similar
    rng = np.random.default_rng(2)
    smooth_noise = (rng.normal(0, 1.0, X.shape))
    # box-blur the noise
    from numpy.lib.stride_tricks import sliding_window_view
    k = 11
    pad = k // 2
    pn = np.pad(smooth_noise, pad, mode="edge")
    blurred = sliding_window_view(pn, (k, k)).mean(axis=(-1, -2))
    recovered = true_r + 0.18 * blurred / (np.abs(blurred).max() + 1e-9)

    panels = [
        ("True reward $r^*(s)$ (unknown)", true_r, "RdBu_r"),
        ("Observed expert behaviour density", expert_d, "Blues"),
        ("IRL: recovered reward $\\hat r(s)$", recovered, "RdBu_r"),
    ]

    for ax, (title, Z, cmap) in zip(axes, panels):
        if cmap == "RdBu_r":
            vmax = max(abs(Z.min()), abs(Z.max()))
            im = ax.contourf(X, Y, Z, levels=18, cmap=cmap,
                             vmin=-vmax, vmax=vmax)
            ax.contour(X, Y, Z, levels=8, colors="white", linewidths=0.4,
                       alpha=0.55)
        else:
            im = ax.contourf(X, Y, Z, levels=18, cmap=cmap)
            ax.contour(X, Y, Z, levels=6, colors="white", linewidths=0.4,
                       alpha=0.55)
        ax.set_title(title, color=DARK)
        ax.set_xlabel("state dim 1")
        ax.set_ylabel("state dim 2")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    # Sample expert trajectories on the middle panel
    ax = axes[1]
    rng2 = np.random.default_rng(5)
    for _ in range(6):
        # Random start, walk uphill on the true reward field
        x0, y0 = rng2.uniform(-2.5, 2.5, 2)
        path_x, path_y = [x0], [y0]
        for _ in range(80):
            # numerical gradient ~ on continuous surface
            gx = (np.exp(-(((x0 + 0.05) - 1.8) ** 2 + (y0 - 1.2) ** 2) / 1.62)
                  - np.exp(-(((x0 - 0.05) - 1.8) ** 2 + (y0 - 1.2) ** 2) / 1.62)) / 0.1
            gy = (np.exp(-((x0 - 1.8) ** 2 + ((y0 + 0.05) - 1.2) ** 2) / 1.62)
                  - np.exp(-((x0 - 1.8) ** 2 + ((y0 - 0.05) - 1.2) ** 2) / 1.62)) / 0.1
            x0 += 0.08 * gx + rng2.normal(0, 0.05)
            y0 += 0.08 * gy + rng2.normal(0, 0.05)
            path_x.append(x0)
            path_y.append(y0)
        ax.plot(path_x, path_y, color=DARK, lw=1.0, alpha=0.7)

    fig.suptitle("Inverse RL: from behaviour, recover a reward whose optimum reproduces the demos",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_irl_recovery.png")


# ---------------------------------------------------------------------------
# Fig 4: GAIL discriminator-generator architecture
# ---------------------------------------------------------------------------
def fig4_gail_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, label, color, fc=None, fontsize=10.5, weight="bold"):
        face = fc if fc is not None else color
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.06",
                           linewidth=1.8, edgecolor=color, facecolor=face,
                           alpha=0.18 if fc is None else 1.0)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                color=DARK, weight=weight)

    def arrow(x1, y1, x2, y2, color=GRAY, lw=1.6, label=None, label_off=(0, 0.2),
              style="->"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, mutation_scale=16,
                            color=color, lw=lw)
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2 + label_off[0],
                    (y1 + y2) / 2 + label_off[1],
                    label, ha="center", fontsize=9, color=color,
                    weight="bold")

    # Expert dataset (top-left)
    box(0.3, 6.0, 2.8, 1.4, "Expert dataset\n$\\mathcal{D} = \\{(s, a)\\}$", BLUE,
        fc="#dbeafe")

    # Policy / Generator (left-middle)
    box(0.3, 3.4, 2.8, 1.6, "Policy $\\pi_\\theta$\n(generator)", ORANGE,
        fc="#fed7aa")

    # Environment (bottom-left)
    box(0.3, 0.6, 2.8, 1.6, "Environment\n$p(s' \\mid s, a)$", GRAY,
        fc=LIGHT)

    # Rollout trajectories
    box(4.4, 3.4, 2.8, 1.6, "Policy rollouts\n$(s, a) \\sim \\pi_\\theta$",
        ORANGE, fc="#fff7ed")
    arrow(3.1, 4.2, 4.4, 4.2, color=ORANGE, label="act", label_off=(0, 0.25))
    arrow(3.1, 1.4, 5.8, 3.4, color=GRAY, style="->", lw=1.2)
    arrow(5.8, 3.4, 3.1, 1.4, color=GRAY, style="->", lw=1.2,
          label="$s \\to a \\to s'$", label_off=(-0.2, -0.2))

    # Discriminator (center-right)
    box(8.2, 4.2, 3.4, 2.0,
        "Discriminator $D_\\phi(s,a)$\nbinary classifier",
        PURPLE, fc="#ede9fe", fontsize=10.5)

    # Inputs into discriminator
    arrow(3.1, 6.7, 8.2, 5.7, color=BLUE,
          label="expert $(s,a)$", label_off=(-1.0, 0.25))
    arrow(7.2, 4.2, 8.2, 4.7, color=ORANGE,
          label="policy $(s,a)$", label_off=(-0.2, -0.5))

    # Reward feedback to policy
    box(8.2, 1.0, 3.4, 1.4,
        "Imitation reward\n$r(s,a) = -\\log D_\\phi(s,a)$",
        GREEN, fc="#d1fae5")
    arrow(9.9, 4.2, 9.9, 2.4, color=GREEN, lw=1.8,
          label="$D_\\phi$", label_off=(0.45, 0))
    arrow(8.2, 1.7, 1.7, 3.4, color=GREEN, lw=1.8,
          label="policy gradient (PPO)", label_off=(-0.4, -0.4))

    # Discriminator update
    box(12.2, 4.2, 1.6, 2.0,
        "BCE loss\n$\\nabla_\\phi$",
        PURPLE, fc="#f3e8ff", fontsize=10)
    arrow(11.6, 5.2, 12.2, 5.2, color=PURPLE, lw=1.6)
    arrow(13.0, 4.2, 11.9, 3.7, color=PURPLE, style="-|>", lw=1.4)
    ax.text(13.3, 3.7, "update\n$\\phi$", color=PURPLE, fontsize=9,
            weight="bold", ha="center")

    # Two-player title
    ax.text(7.0, 7.65,
            "min$_\\theta$ max$_\\phi\\;$ "
            "$\\mathbb{E}_{\\pi^*}[\\log(1-D)] + \\mathbb{E}_{\\pi_\\theta}[\\log D] - \\lambda H(\\pi_\\theta)$",
            ha="center", fontsize=12, color=DARK, weight="bold")

    ax.text(7.0, 0.05,
            "Generator (policy) tries to fool D; D tries to spot policy samples. "
            "At equilibrium, $\\pi_\\theta$ matches the expert occupancy measure.",
            ha="center", fontsize=9.5, color=GRAY, style="italic")

    fig.suptitle("GAIL: adversarial imitation as a two-player game between $\\pi_\\theta$ and $D_\\phi$",
                 fontsize=13.5, weight="bold", y=0.99)
    save(fig, "fig4_gail_architecture.png")


# ---------------------------------------------------------------------------
# Fig 5: Compounding error in BC (cascading drift)
# ---------------------------------------------------------------------------
def fig5_compounding_error() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # ---- (a) Theoretical bound: epsilon * T^2 (BC) vs epsilon * T (DAgger)
    ax = axes[0]
    T = np.arange(1, 201)
    eps = 0.02
    bc_err = np.minimum(eps * T ** 2, T.astype(float))   # cap at horizon
    dagger_err = eps * T

    ax.plot(T, bc_err, color=ORANGE, lw=2.6, label="BC bound $\\;\\varepsilon T^2$")
    ax.plot(T, dagger_err, color=GREEN, lw=2.6,
            label="DAgger bound $\\;\\varepsilon T$")
    ax.fill_between(T, dagger_err, bc_err, color=ORANGE, alpha=0.10)

    # mark where BC saturates
    sat_T = int(np.sqrt(1 / eps))
    ax.axvline(sat_T, color=GRAY, ls=":", lw=1.2)
    ax.text(sat_T + 2, 0.15, f"BC saturates\nat T≈{sat_T}",
            color=GRAY, fontsize=9)

    ax.set_xlabel("trajectory length $T$")
    ax.set_ylabel("expected total error (bounded by 1)")
    ax.set_title("Theoretical error growth (Ross et al., 2011)", color=DARK)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    # ---- (b) Empirical-style cascade: per-step error grows over time
    ax = axes[1]
    rng = np.random.default_rng(11)
    steps = np.arange(0, 200)

    # BC: per-step error grows because state distribution shifts
    bc_step = 0.02 + 0.0008 * steps + rng.normal(0, 0.004, len(steps))
    bc_step = np.clip(bc_step, 0.005, None)
    bc_cum = np.cumsum(bc_step)

    # DAgger: roughly stationary per-step error
    dag_step = 0.02 + rng.normal(0, 0.004, len(steps))
    dag_step = np.clip(dag_step, 0.005, None)
    dag_cum = np.cumsum(dag_step)

    ax.plot(steps, bc_cum, color=ORANGE, lw=2.4, label="BC cumulative error")
    ax.plot(steps, dag_cum, color=GREEN, lw=2.4, label="DAgger cumulative error")
    ax.fill_between(steps, dag_cum, bc_cum, color=ORANGE, alpha=0.12)

    # arrows annotating phases
    ax.annotate("on-distribution\n(per-step error stable)",
                xy=(20, bc_cum[20]), xytext=(40, 4.0),
                fontsize=9, color=DARK, ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.annotate("off-distribution\n(error compounds)",
                xy=(160, bc_cum[160]), xytext=(120, 5.5),
                fontsize=9, color=ORANGE, ha="center", weight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))

    ax.set_xlabel("step within episode")
    ax.set_ylabel("cumulative action error")
    ax.set_title("Empirical cascade: BC drifts further per step", color=DARK)
    ax.legend(loc="upper left")

    fig.suptitle("Compounding error: BC suffers quadratic drift, DAgger stays linear",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_compounding_error.png")


# ---------------------------------------------------------------------------
# Fig 6: Method hierarchy — BC -> DAgger -> IRL -> GAIL -> AIRL
# ---------------------------------------------------------------------------
def fig6_method_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    methods = [
        ("BC",
         "supervised on $(s,a)$",
         "no env, no expert query",
         ORANGE, "#fed7aa"),
        ("DAgger",
         "interactive relabel",
         "linear error growth",
         GREEN, "#d1fae5"),
        ("MaxEnt IRL",
         "recover $r_\\theta(s)$",
         "interpretable, transferable",
         PURPLE, "#ede9fe"),
        ("GAIL",
         "match occupancy via $D_\\phi$",
         "no $r$, scales to vision",
         BLUE, "#dbeafe"),
        ("AIRL",
         "disentangle $r$ from shaping",
         "robust transfer rewards",
         "#0f766e", "#ccfbf1"),
    ]

    n = len(methods)
    box_w = 2.4
    box_h = 1.6
    gap = (15 - n * box_w) / (n + 1)

    centers_x = []
    for i, (name, body, foot, color, face) in enumerate(methods):
        x = gap + i * (box_w + gap)
        y = 3.5
        b = FancyBboxPatch((x, y), box_w, box_h,
                           boxstyle="round,pad=0.06",
                           linewidth=2.0, edgecolor=color, facecolor=face)
        ax.add_patch(b)
        ax.text(x + box_w / 2, y + 1.15, name,
                ha="center", fontsize=13, color=DARK, weight="bold")
        ax.text(x + box_w / 2, y + 0.55, body,
                ha="center", fontsize=9.5, color=DARK)
        ax.text(x + box_w / 2, y - 0.35, foot,
                ha="center", fontsize=8.8, color=color, style="italic",
                weight="bold")
        centers_x.append(x + box_w / 2)

    # Arrows between boxes
    for i in range(n - 1):
        a = FancyArrowPatch(
            (centers_x[i] + box_w / 2 + 0.05, 4.3),
            (centers_x[i + 1] - box_w / 2 - 0.05, 4.3),
            arrowstyle="-|>", mutation_scale=18, color=GRAY, lw=1.8,
        )
        ax.add_patch(a)

    # What each successive arrow buys you
    captions = [
        "+ interactive labels",
        "+ structural reward",
        "+ adversarial training",
        "+ reward disentanglement",
    ]
    for i, cap in enumerate(captions):
        x_mid = (centers_x[i] + centers_x[i + 1]) / 2
        ax.text(x_mid, 4.7, cap, ha="center", fontsize=8.8,
                color=DARK, style="italic")

    # Bottom axis: "what you give up to gain"
    ax.annotate("", xy=(14.5, 1.6), xytext=(0.5, 1.6),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6))
    ax.text(0.5, 1.2, "supervised, simple", color=ORANGE, fontsize=10,
            ha="left", weight="bold")
    ax.text(14.5, 1.2, "adversarial, transferable", color="#0f766e",
            fontsize=10, ha="right", weight="bold")
    ax.text(7.5, 0.7, "increasing assumptions on expert / environment access",
            ha="center", fontsize=9.5, color=GRAY, style="italic")

    # Top axis
    ax.text(0.5, 6.4, "weaker assumptions",
            color=ORANGE, fontsize=10, weight="bold", ha="left")
    ax.text(14.5, 6.4, "stronger guarantees / generalization",
            color="#0f766e", fontsize=10, weight="bold", ha="right")
    ax.annotate("", xy=(14.5, 6.0), xytext=(0.5, 6.0),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6))

    fig.suptitle("Imitation learning ladder: each rung adds capability and assumes more",
                 fontsize=13.5, weight="bold", y=1.0)
    save(fig, "fig6_method_hierarchy.png")


# ---------------------------------------------------------------------------
# Fig 7: Sample efficiency — imitation vs RL learning curves
# ---------------------------------------------------------------------------
def fig7_sample_efficiency() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    rng = np.random.default_rng(19)

    # --- (a) Learning curves: env interactions vs return
    ax = axes[0]
    steps = np.linspace(0, 1.0, 200)  # millions of env steps

    def curve(asymptote, scale, noise=0.02):
        base = asymptote * (1 - np.exp(-steps * scale))
        return base + rng.normal(0, noise * asymptote, len(steps))

    rl_ppo = curve(1.0, 4.0, 0.025)              # slow but reaches asymptote
    rl_ppo = np.clip(rl_ppo, 0, 1.05)
    bc = np.full_like(steps, 0.55) + rng.normal(0, 0.02, len(steps))   # flat (offline)
    dagger = np.where(steps < 0.05, 0.55,
                      0.55 + (0.85 - 0.55) * (1 - np.exp(-(steps - 0.05) * 25)))
    dagger = dagger + rng.normal(0, 0.015, len(steps))
    gail = np.where(steps < 0.02, 0.0,
                    1.0 * (1 - np.exp(-(steps - 0.02) * 12)))
    gail = np.clip(gail + rng.normal(0, 0.025, len(steps)), 0, 1.05)

    ax.plot(steps, rl_ppo, color=GRAY, lw=2.4, label="RL from scratch (PPO)")
    ax.plot(steps, bc, color=ORANGE, lw=2.4, label="Behavioral cloning")
    ax.plot(steps, dagger, color=GREEN, lw=2.4, label="DAgger")
    ax.plot(steps, gail, color=BLUE, lw=2.4, label="GAIL")

    ax.axhline(1.0, color=DARK, ls=":", lw=1.2)
    ax.text(0.02, 1.02, "expert performance", fontsize=9, color=DARK)

    ax.set_xlabel("environment interactions ($\\times 10^6$ steps)")
    ax.set_ylabel("normalized episodic return")
    ax.set_ylim(0, 1.15)
    ax.set_title("Learning curves: imitation jumpstarts learning", color=DARK)
    ax.legend(loc="lower right", fontsize=9.5)

    # --- (b) Demonstrations needed to reach 90% expert performance
    ax = axes[1]
    methods = ["BC", "DAgger", "GAIL", "MaxEnt IRL", "RL\n(no demos)"]
    # Approximate orders of magnitude on a fixed task
    demos = [10000, 800, 1500, 1200, 0]
    env_steps = [0, 5e5, 3e6, 8e6, 5e7]   # env interactions

    width = 0.36
    idx = np.arange(len(methods))

    # Use log scale on a twin axis for env steps
    ax2 = ax.twinx()
    bars1 = ax.bar(idx - width / 2, demos, width=width, color=BLUE, alpha=0.85,
                   edgecolor="white", label="expert demos needed")
    bars2 = ax2.bar(idx + width / 2,
                    [max(s, 1) for s in env_steps], width=width,
                    color=GRAY, alpha=0.85, edgecolor="white",
                    label="env steps to 90% expert")

    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_ylabel("expert demonstrations (log)", color=BLUE)
    ax2.set_ylabel("environment steps (log)", color=GRAY)
    ax.tick_params(axis="y", colors=BLUE)
    ax2.tick_params(axis="y", colors=GRAY)
    ax.set_xticks(idx)
    ax.set_xticklabels(methods)
    ax.set_title("Cost to reach 90% of expert: demos vs env steps",
                 color=DARK)

    # legend (combined)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    # annotate RL
    ax.text(4, 2, "needs no demos\nbut huge env budget",
            ha="center", fontsize=8.5, color=DARK)

    fig.suptitle("Sample efficiency: imitation buys orders-of-magnitude env-step savings",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_sample_efficiency.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_distribution_shift()
    fig2_expert_vs_learner()
    fig3_irl_recovery()
    fig4_gail_architecture()
    fig5_compounding_error()
    fig6_method_hierarchy()
    fig7_sample_efficiency()
    print(f"All figures saved to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
