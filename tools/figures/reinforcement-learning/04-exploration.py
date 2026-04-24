"""
Figures for Reinforcement Learning Part 4: Exploration & Curiosity-Driven Learning.

Generates 7 production-quality figures and saves to BOTH:
  - source/_posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/
  - source/_posts/zh/reinforcement-learning/04-探索策略与好奇心驱动学习/

Run:
    python 04-exploration.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# ----------------------------- Style ---------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#6b7280"
LIGHT = "#e5e7eb"
DARK = "#111827"
RED = "#dc2626"

# ----------------------------- Output paths --------------------------------

ROOT = Path(__file__).resolve().parents[2].parent  # chenk-site/
EN_DIR = ROOT / "source/_posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning"
ZH_DIR = ROOT / "source/_posts/zh/reinforcement-learning/04-探索策略与好奇心驱动学习"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        path = d / name
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  saved -> {path}")
    plt.close(fig)


# ============================== Figure 1 ===================================
# Epsilon-greedy decay schedules: linear, exponential, step, plus action-prob
# behaviour visualised as a stacked area showing exploration vs exploitation.

def fig1_epsilon_decay() -> None:
    steps = np.arange(0, 100_000)
    eps_start, eps_end = 1.0, 0.05

    # Linear decay over first 60k steps then constant.
    decay_steps = 60_000
    linear = np.maximum(eps_end,
                        eps_start - (eps_start - eps_end) * steps / decay_steps)

    # Exponential decay: eps_end + (eps_start - eps_end) * exp(-k * step)
    k = 5e-5
    exponential = eps_end + (eps_start - eps_end) * np.exp(-k * steps)

    # Step / piecewise: 1.0 -> 0.5 -> 0.2 -> 0.05 at fixed milestones.
    step_sched = np.where(steps < 15_000, 1.0,
                  np.where(steps < 35_000, 0.5,
                  np.where(steps < 60_000, 0.2, 0.05)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # ---- Left: three decay curves ----
    ax = axes[0]
    ax.plot(steps / 1e3, linear, color=BLUE, lw=2.4, label="Linear (60k steps)")
    ax.plot(steps / 1e3, exponential, color=PURPLE, lw=2.4, label="Exponential (k=5e-5)")
    ax.plot(steps / 1e3, step_sched, color=GREEN, lw=2.4, label="Step / piecewise")
    ax.axhline(eps_end, color=GRAY, lw=1, linestyle=":", alpha=0.7)
    ax.text(95, eps_end + 0.02, "ε_min = 0.05", color=GRAY, fontsize=9, ha="right")

    ax.set_title("ε-greedy decay schedules")
    ax.set_xlabel("Training steps (×10³)")
    ax.set_ylabel("Exploration rate ε")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.95)

    # ---- Right: action probability under linear decay (4 actions) ----
    ax = axes[1]
    n_actions = 4
    # Greedy action gets 1 - eps + eps/N, others eps/N
    eps = linear
    p_greedy = 1 - eps + eps / n_actions
    p_other = eps / n_actions

    ax.fill_between(steps / 1e3, 0, p_greedy, color=BLUE, alpha=0.85,
                    label=f"Greedy action  (1-ε+ε/{n_actions})")
    ax.fill_between(steps / 1e3, p_greedy, p_greedy + 3 * p_other, color=ORANGE,
                    alpha=0.7, label=f"3 random actions  (3·ε/{n_actions})")

    ax.set_title("Action-selection probabilities (4 actions, linear decay)")
    ax.set_xlabel("Training steps (×10³)")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center right", framealpha=0.95)

    fig.suptitle("ε-greedy: simple but aimless exploration",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_epsilon_greedy_decay.png")


# ============================== Figure 2 ===================================
# Boltzmann / softmax: temperature shapes the action distribution.

def fig2_boltzmann_softmax() -> None:
    actions = np.arange(5)
    q_values = np.array([1.0, 1.5, 3.0, 2.2, 0.7])
    temperatures = [0.1, 0.5, 1.0, 5.0]
    colors = [BLUE, PURPLE, GREEN, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # ---- Left: bar chart of distributions per temperature ----
    ax = axes[0]
    width = 0.18
    for i, (tau, c) in enumerate(zip(temperatures, colors)):
        logits = q_values / tau
        logits = logits - logits.max()  # numerical stability
        probs = np.exp(logits) / np.exp(logits).sum()
        ax.bar(actions + (i - 1.5) * width, probs, width,
               color=c, edgecolor="white", linewidth=0.7,
               label=f"τ = {tau}")

    ax.set_title("Boltzmann action distribution by temperature")
    ax.set_xlabel("Action")
    ax.set_ylabel("π(a | s)")
    ax.set_xticks(actions)
    ax.set_xticklabels([f"a{i}\nQ={q:.1f}" for i, q in enumerate(q_values)])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", framealpha=0.95)

    # ---- Right: entropy vs temperature ----
    ax = axes[1]
    taus = np.linspace(0.05, 6, 200)
    entropies = []
    for tau in taus:
        logits = q_values / tau
        logits = logits - logits.max()
        p = np.exp(logits) / np.exp(logits).sum()
        entropies.append(-(p * np.log(p + 1e-12)).sum())
    entropies = np.array(entropies)
    max_entropy = np.log(len(actions))

    ax.plot(taus, entropies, color=PURPLE, lw=2.6)
    ax.axhline(max_entropy, color=GRAY, lw=1, linestyle="--")
    ax.text(5.8, max_entropy - 0.08, f"max entropy = ln {len(actions)} ≈ {max_entropy:.2f}",
            fontsize=9, color=GRAY, ha="right")

    # Annotate the four sample temperatures
    for tau, c in zip(temperatures, colors):
        idx = np.abs(taus - tau).argmin()
        ax.scatter(tau, entropies[idx], color=c, s=70, zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"τ={tau}", (tau, entropies[idx]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=c, fontweight="bold")

    ax.set_title("Entropy of policy vs temperature")
    ax.set_xlabel("Temperature τ")
    ax.set_ylabel("Entropy H(π)")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, max_entropy + 0.2)

    # Annotations: low-tau = greedy, high-tau = uniform
    ax.text(0.15, 0.25, "low τ\n→ greedy", fontsize=9.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec=ORANGE, lw=1))
    ax.text(5.0, 1.45, "high τ\n→ uniform", fontsize=9.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec=BLUE, lw=1))

    fig.suptitle("Boltzmann (softmax) exploration: temperature controls greediness",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_boltzmann_softmax.png")


# ============================== Figure 3 ===================================
# UCB on a 5-arm bandit: Q-estimate + confidence bonus, evolution over time.

def fig3_ucb_bandit() -> None:
    rng = np.random.default_rng(7)
    true_means = np.array([0.20, 0.55, 0.40, 0.80, 0.30])
    n_arms = len(true_means)
    n_steps = 1000
    c = 1.5  # exploration constant

    counts = np.zeros(n_arms)
    sums = np.zeros(n_arms)
    history_q = []
    history_bonus = []
    history_pull = []

    # Init: pull each arm once
    for a in range(n_arms):
        r = rng.normal(true_means[a], 0.15)
        counts[a] += 1
        sums[a] += r
        history_pull.append(a)

    for t in range(n_arms, n_steps):
        q = sums / np.maximum(counts, 1)
        bonus = c * np.sqrt(np.log(t + 1) / np.maximum(counts, 1))
        a = int(np.argmax(q + bonus))
        r = rng.normal(true_means[a], 0.15)
        counts[a] += 1
        sums[a] += r
        history_pull.append(a)
        if t in (50, 200, 999):
            history_q.append(q.copy())
            history_bonus.append(bonus.copy())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- Left: Q + bonus stacked at t=50, 200, 1000 ----
    ax = axes[0]
    snapshots = [50, 200, 1000]
    width = 0.25
    for i, (q, b, t) in enumerate(zip(history_q, history_bonus, snapshots)):
        x = np.arange(n_arms) + (i - 1) * width
        ax.bar(x, q, width, color=BLUE, alpha=0.85, edgecolor="white",
               label="Q-estimate" if i == 0 else None)
        ax.bar(x, b, width, bottom=q, color=ORANGE, alpha=0.85, edgecolor="white",
               label="UCB bonus" if i == 0 else None)
        for j in range(n_arms):
            ax.text(x[j], q[j] + b[j] + 0.02, f"t={t}",
                    fontsize=7.5, ha="center", color=GRAY)

    # True mean dashes
    for j in range(n_arms):
        ax.hlines(true_means[j], j - 0.45, j + 0.45,
                  colors=GREEN, linestyles="--", lw=1.6, zorder=5)
    ax.plot([], [], color=GREEN, linestyle="--", lw=1.6, label="True mean")

    ax.set_title("UCB1 score = Q(a) + c·√(ln t / N(a))")
    ax.set_xlabel("Arm")
    ax.set_ylabel("Score")
    ax.set_xticks(range(n_arms))
    ax.set_xticklabels([f"arm {i}" for i in range(n_arms)])
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right", framealpha=0.95)

    # ---- Right: cumulative pull counts ----
    ax = axes[1]
    pulls = np.array(history_pull)
    cum = np.zeros((n_arms, n_steps))
    for a in range(n_arms):
        cum[a] = np.cumsum(pulls == a)
    ts = np.arange(n_steps)
    palette = [BLUE, PURPLE, GRAY, GREEN, ORANGE]
    for a in range(n_arms):
        ax.plot(ts, cum[a], color=palette[a], lw=2.2,
                label=f"arm {a} (μ={true_means[a]:.2f})")

    # Mark the optimal arm
    best = int(np.argmax(true_means))
    ax.text(n_steps * 0.98, cum[best, -1] + 15,
            f"optimal arm (#{best}) dominates",
            color=GREEN, fontsize=10, fontweight="bold", ha="right")

    ax.set_title("Cumulative pulls per arm over 1000 steps")
    ax.set_xlabel("Step t")
    ax.set_ylabel("Number of pulls N(a)")
    ax.set_xlim(0, n_steps)
    ax.legend(loc="upper left", framealpha=0.95, ncol=2)

    fig.suptitle("UCB: optimism in the face of uncertainty",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_ucb_bandit.png")


# ============================== Figure 4 ===================================
# Thompson sampling: Beta posteriors converge to true rates.

def fig4_thompson_sampling() -> None:
    rng = np.random.default_rng(11)
    true_rates = [0.30, 0.55, 0.75]
    n_arms = len(true_rates)
    palette = [BLUE, PURPLE, GREEN]
    snapshots = [10, 50, 300]
    n_steps = max(snapshots)

    # Beta(alpha, beta), start uniform Beta(1,1)
    alpha = np.ones(n_arms)
    beta = np.ones(n_arms)
    snapshot_state = {}

    for t in range(1, n_steps + 1):
        samples = rng.beta(alpha, beta)
        a = int(np.argmax(samples))
        reward = int(rng.uniform() < true_rates[a])
        if reward:
            alpha[a] += 1
        else:
            beta[a] += 1
        if t in snapshots:
            snapshot_state[t] = (alpha.copy(), beta.copy())

    from scipy.stats import beta as beta_dist  # type: ignore
    xs = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for i, t in enumerate(snapshots):
        ax = axes[i]
        a_v, b_v = snapshot_state[t]
        for arm in range(n_arms):
            pdf = beta_dist.pdf(xs, a_v[arm], b_v[arm])
            ax.plot(xs, pdf, color=palette[arm], lw=2.4,
                    label=f"arm {arm}: Beta({int(a_v[arm])}, {int(b_v[arm])})")
            ax.fill_between(xs, 0, pdf, color=palette[arm], alpha=0.18)
            ax.axvline(true_rates[arm], color=palette[arm],
                       linestyle="--", lw=1.2, alpha=0.65)

        ax.set_title(f"After t = {t} pulls")
        ax.set_xlabel("Reward probability")
        if i == 0:
            ax.set_ylabel("Posterior density")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

    fig.suptitle(
        "Thompson sampling: Beta posteriors sharpen around the true reward rates "
        "(dashed = true μ)",
        fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig4_thompson_sampling.png")


# ============================== Figure 5 ===================================
# ICM and RND architectures, side by side, with intrinsic-reward formula.

def _box(ax, x, y, w, h, text, fc, ec=DARK, fontsize=9.5, weight="normal"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.06",
                         fc=fc, ec=ec, lw=1.4)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=DARK, fontweight=weight)


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="-|>"):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=14,
                          color=color, lw=lw,
                          shrinkA=2, shrinkB=2)
    ax.add_patch(arr)


def fig5_icm_rnd_architecture() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.0))

    # ============ Left: ICM ============
    ax = axes[0]
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("ICM — Intrinsic Curiosity Module (Pathak et al., 2017)",
                 fontsize=12.5)

    # Inputs (left column)
    _box(ax, 0.1, 5.6, 1.4, 0.8, "s_t", "#dbeafe")
    _box(ax, 0.1, 0.6, 1.4, 0.8, "s_{t+1}", "#dbeafe")
    _box(ax, 0.1, 3.1, 1.4, 0.8, "a_t", "#fef3c7")

    # Encoders φ
    _box(ax, 2.1, 5.6, 1.6, 0.8, "Encoder φ", "#e9d5ff")
    _box(ax, 2.1, 0.6, 1.6, 0.8, "Encoder φ", "#e9d5ff")

    # Feature outputs
    _box(ax, 4.3, 5.6, 1.3, 0.8, "φ(s_t)", "#dcfce7")
    _box(ax, 4.3, 0.6, 1.3, 0.8, "φ(s_{t+1})", "#dcfce7")

    # Forward branch (top right)
    _box(ax, 6.3, 5.4, 1.7, 0.9, "Forward\nmodel f̂", "#fee2e2", weight="bold")
    _box(ax, 8.4, 5.5, 1.2, 0.7, "φ̂_{t+1}", "#fecaca")

    # Inverse branch (bottom right)
    _box(ax, 6.3, 0.5, 1.7, 0.9, "Inverse\nmodel g", "#bfdbfe", weight="bold")
    _box(ax, 8.4, 0.6, 1.2, 0.7, "â_t", "#93c5fd")

    # ---- Arrows ----
    # encoder pipeline (gray)
    _arrow(ax, 1.5, 6.0, 2.1, 6.0)
    _arrow(ax, 1.5, 1.0, 2.1, 1.0)
    _arrow(ax, 3.7, 6.0, 4.3, 6.0)
    _arrow(ax, 3.7, 1.0, 4.3, 1.0)

    # Forward path (red): φ(s_t) -> forward, a_t -> forward, forward -> φ̂
    _arrow(ax, 5.6, 6.0, 6.3, 6.0, color=RED)
    _arrow(ax, 1.5, 3.5, 6.3, 5.6, color=RED)  # a_t into forward (long curve replaced by direct)
    _arrow(ax, 8.0, 5.85, 8.4, 5.85, color=RED)

    # Inverse path (blue): φ(s_t) and φ(s_{t+1}) -> inverse -> â
    _arrow(ax, 4.95, 5.6, 6.3, 1.4, color=BLUE)   # φ(s_t) down to inverse
    _arrow(ax, 4.95, 1.4, 6.3, 1.0, color=BLUE)   # φ(s_{t+1}) right to inverse
    _arrow(ax, 8.0, 0.95, 8.4, 0.95, color=BLUE)

    # Comparison: φ̂_{t+1}  vs  φ(s_{t+1})  -> intrinsic reward
    # Place reward formula in clear middle-right area
    ax.annotate("", xy=(7.0, 3.6), xytext=(9.0, 5.4),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    ax.annotate("", xy=(7.0, 3.6), xytext=(5.0, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                connectionstyle="arc3,rad=-0.15"))

    ax.text(7.0, 3.2, "r_int = η · ‖φ̂_{t+1} − φ(s_{t+1})‖²",
            fontsize=10, color=RED, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec=RED, lw=1.2))

    ax.text(7.0, 2.4,
            "Inverse loss trains φ to encode\nonly action-relevant features\n"
            "(filters out the noisy TV)",
            fontsize=8.8, color=BLUE, ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#eff6ff", ec=BLUE, lw=1.2))

    # ============ Right: RND ============
    ax = axes[1]
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("RND — Random Network Distillation (Burda et al., 2019)",
                 fontsize=12.5)

    _box(ax, 0.3, 3.1, 1.5, 0.8, "obs s", "#dbeafe")

    _box(ax, 2.8, 4.6, 2.6, 1.0, "Target net f\n(random, frozen)", "#e5e7eb",
         weight="bold")
    _box(ax, 2.8, 1.4, 2.6, 1.0, "Predictor f̂\n(trainable)", "#dcfce7",
         weight="bold")

    _box(ax, 6.4, 4.7, 1.3, 0.8, "f(s)", "#9ca3af")
    _box(ax, 6.4, 1.5, 1.3, 0.8, "f̂(s)", "#10b981")

    _arrow(ax, 1.8, 3.5, 2.8, 5.1)
    _arrow(ax, 1.8, 3.5, 2.8, 1.9)
    _arrow(ax, 5.4, 5.1, 6.4, 5.1)
    _arrow(ax, 5.4, 1.9, 6.4, 1.9)

    # Diff node
    _box(ax, 8.4, 3.1, 2.0, 1.0, "‖f̂(s) − f(s)‖²", "#fef3c7", weight="bold")
    _arrow(ax, 7.7, 5.1, 8.4, 4.0, color=ORANGE)
    _arrow(ax, 7.7, 1.9, 8.4, 3.2, color=ORANGE)

    ax.text(9.4, 2.4, "= r_int", fontsize=12, color=RED, fontweight="bold",
            ha="center")
    _arrow(ax, 9.4, 3.1, 9.4, 2.7, color=RED)

    ax.text(5.5, 0.4,
            "Familiar state → predictor matches → low r_int\n"
            "Novel state → predictor lags        → high r_int",
            fontsize=9.2, color=DARK, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fffbeb", ec=ORANGE, lw=1.2))

    fig.suptitle(
        "Two ways to manufacture curiosity: predict the future (ICM) "
        "or distil a random network (RND)",
        fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig5_icm_rnd_architecture.png")


# ============================== Figure 6 ===================================
# Sparse-reward problem: Montezuma-style score curves comparing ε-greedy DQN,
# count-based, ICM, RND, NGU.

def fig6_sparse_reward_montezuma() -> None:
    steps = np.linspace(0, 100, 400)  # millions of frames

    def saturating(curve_max, halfway, sharpness=1.6):
        # Smooth logistic approaching curve_max
        x = (steps - halfway) * sharpness / 10
        return curve_max / (1 + np.exp(-x))

    rng = np.random.default_rng(3)
    methods = [
        ("DQN + ε-greedy",       np.zeros_like(steps),                   GRAY),
        ("Count-based pseudo-N", saturating(2500, 70, 1.0),               ORANGE),
        ("ICM (PPO + curiosity)", saturating(6500, 55, 1.4),              PURPLE),
        ("RND (PPO + RND)",      saturating(8200, 45, 1.6),               BLUE),
        ("NGU (episodic + RND)", saturating(11000, 40, 1.6),              GREEN),
    ]
    human_level = 7385

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # ---- Left: score curves ----
    ax = axes[0]
    for name, curve, c in methods:
        # add light noise for realism
        noisy = curve + rng.normal(0, 80, size=curve.shape) * (curve > 50)
        noisy = np.maximum(noisy, 0)
        ax.plot(steps, noisy, color=c, lw=2.4, label=name)

    ax.axhline(human_level, color=DARK, lw=1.4, linestyle="--", alpha=0.7)
    ax.text(98, human_level + 250, f"human expert ≈ {human_level:,}",
            fontsize=9, color=DARK, ha="right")

    ax.set_title("Montezuma's Revenge — score vs training")
    ax.set_xlabel("Environment frames (millions)")
    ax.set_ylabel("Episode return")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 12500)
    ax.legend(loc="upper left", framealpha=0.95)

    # ---- Right: bar chart of "first reward" timing ----
    ax = axes[1]
    methods_short = ["ε-greedy\nDQN", "Count-\nbased", "ICM", "RND", "NGU"]
    first_reward_M = [np.nan, 25, 8, 5, 3]  # millions of frames
    colors = [GRAY, ORANGE, PURPLE, BLUE, GREEN]

    bars = []
    for i, (m, v, c) in enumerate(zip(methods_short, first_reward_M, colors)):
        if np.isnan(v):
            ax.bar(i, 100, color=GRAY, alpha=0.25, edgecolor=GRAY,
                   linewidth=1.4, hatch="//")
            ax.text(i, 50, "never", ha="center", va="center",
                    fontsize=10, color=DARK, fontweight="bold")
        else:
            ax.bar(i, v, color=c, alpha=0.9, edgecolor="white", linewidth=1.2)
            ax.text(i, v + 2, f"{v}M", ha="center", fontsize=10,
                    fontweight="bold", color=DARK)

    ax.set_xticks(range(len(methods_short)))
    ax.set_xticklabels(methods_short)
    ax.set_ylabel("Frames to first non-zero reward (M)")
    ax.set_title("How long until the agent finds *any* reward?")
    ax.set_ylim(0, 110)

    fig.suptitle(
        "Sparse rewards crush random exploration; intrinsic motivation rescues it",
        fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_sparse_reward_montezuma.png")


# ============================== Figure 7 ===================================
# State-visitation heatmaps in a 20x20 GridWorld: ε-greedy vs curiosity-driven.

def fig7_trajectory_comparison() -> None:
    rng = np.random.default_rng(42)
    H, W = 25, 25
    n_steps = 1500

    def simulate(mode: str) -> np.ndarray:
        visits = np.zeros((H, W))
        x, y = W // 2, H // 2
        visits[y, x] = 1
        for _ in range(n_steps):
            if mode == "epsilon":
                # Pure random walk (ε=1) — what raw exploration looks like
                dx, dy = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
            else:
                # Curiosity: bias moves toward less-visited neighbours
                neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1),
                              (-1, -1), (1, 1), (-1, 1), (1, -1)]
                scores = []
                for dx_, dy_ in neighbours:
                    nx, ny = x + dx_, y + dy_
                    if 0 <= nx < W and 0 <= ny < H:
                        # lower visit count -> higher intrinsic reward
                        scores.append(1.0 / np.sqrt(visits[ny, nx] + 1))
                    else:
                        scores.append(0.0)
                scores = np.array(scores)
                p = scores / scores.sum()
                idx = rng.choice(len(neighbours), p=p)
                dx, dy = neighbours[idx]
            nx, ny = max(0, min(W - 1, x + dx)), max(0, min(H - 1, y + dy))
            x, y = nx, ny
            visits[y, x] += 1
        return visits

    eps_map = simulate("epsilon")
    cur_map = simulate("curiosity")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})

    def show(ax, m, title, cmap):
        im = ax.imshow(np.log1p(m), cmap=cmap, origin="lower",
                       interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        # Mark the start
        ax.scatter([W // 2], [H // 2], marker="*", s=130,
                   color="white", edgecolor=DARK, linewidth=1.2, zorder=5)
        # Coverage stat
        coverage = (m > 0).sum() / m.size * 100
        ax.text(0.5, -0.06, f"coverage = {coverage:.1f}%   |   "
                            f"max-visit cell = {int(m.max())}",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=10, color=DARK)
        return im

    show(axes[0], eps_map,
         "Random / ε-greedy walk\n(1500 steps)", "Blues")
    im = show(axes[1], cur_map,
         "Curiosity-driven exploration\n(1500 steps, intrinsic reward ∝ 1/√N(s))",
         "Purples")

    # ---- Third panel: visit count distribution ----
    ax = axes[2]
    eps_visits_sorted = np.sort(eps_map.flatten())[::-1]
    cur_visits_sorted = np.sort(cur_map.flatten())[::-1]
    cells = np.arange(1, eps_visits_sorted.size + 1)

    ax.semilogy(cells, np.maximum(eps_visits_sorted, 0.1),
                color=BLUE, lw=2.4, label="ε-greedy")
    ax.semilogy(cells, np.maximum(cur_visits_sorted, 0.1),
                color=PURPLE, lw=2.4, label="curiosity")

    ax.set_title("Visit-count distribution\n(cells sorted by visits, log scale)")
    ax.set_xlabel("Cell rank")
    ax.set_ylabel("Visit count (log)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xlim(0, eps_visits_sorted.size)

    fig.suptitle(
        "Two agents, 1500 steps, 25×25 grid — curiosity reaches farther, "
        "ε-greedy clusters near the start",
        fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig7_trajectory_comparison.png")


# =============================== main ======================================

def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    print()

    print("Figure 1: ε-greedy decay schedules ...")
    fig1_epsilon_decay()

    print("Figure 2: Boltzmann / softmax ...")
    fig2_boltzmann_softmax()

    print("Figure 3: UCB bandit ...")
    fig3_ucb_bandit()

    print("Figure 4: Thompson sampling ...")
    fig4_thompson_sampling()

    print("Figure 5: ICM & RND architecture ...")
    fig5_icm_rnd_architecture()

    print("Figure 6: Sparse-reward Montezuma comparison ...")
    fig6_sparse_reward_montezuma()

    print("Figure 7: Trajectory / visitation comparison ...")
    fig7_trajectory_comparison()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
