"""Figures for Reinforcement Learning Part 9 — Multi-Agent RL.

Generates publication-quality figures explaining the foundations and key
algorithms of cooperative, competitive and mixed-motive multi-agent RL,
including value decomposition (VDN/QMIX), counterfactual baselines (COMA),
multi-agent actor-critic (MADDPG), communication topologies, and the
league-training pipeline used by AlphaStar / OpenAI Five.

Figures:
    fig1_scenarios          — Cooperative vs competitive vs mixed payoffs
    fig2_ctde               — Centralized training, decentralized execution
    fig3_qmix               — QMIX mixing network with hypernetwork
    fig4_coma               — COMA counterfactual baseline (credit assignment)
    fig5_maddpg             — MADDPG centralized critic, decentralized actor
    fig6_communication      — Broadcast / sparse / attention topologies
    fig7_league             — AlphaStar / OpenAI Five league-training pipeline

Output is written to BOTH the EN and ZH asset folders.

References (verified):
    - Sunehag et al., Value-Decomposition Networks for Cooperative MARL,
      AAMAS 2018 [1706.05296]
    - Rashid et al., QMIX: Monotonic Value Function Factorisation, ICML 2018
      [1803.11485]
    - Lowe et al., Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
      Environments (MADDPG), NeurIPS 2017 [1706.02275]
    - Foerster et al., Counterfactual Multi-Agent Policy Gradients (COMA),
      AAAI 2018 [1705.08926]
    - Vinyals et al., Grandmaster level in StarCraft II using multi-agent
      reinforcement learning (AlphaStar), Nature 2019
    - OpenAI et al., Dota 2 with Large Scale Deep Reinforcement Learning,
      2019 [1912.06680]
    - Yu et al., The Surprising Effectiveness of PPO in Cooperative MARL
      (MAPPO), NeurIPS 2022 [2103.01955]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

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
EN_DIR = REPO / "source/_posts/en/reinforcement-learning/09-multi-agent-rl"
ZH_DIR = REPO / "source/_posts/zh/reinforcement-learning/09-多智能体强化学习"


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.4, style="-|>", mutation=14):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            color=color,
            lw=lw,
            mutation_scale=mutation,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _box(ax, x, y, w, h, label, fc=BLUE, ec=None, alpha=0.18, fontsize=10,
         text_color=DARK, weight="bold", rounding=0.06):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            facecolor=fc,
            edgecolor=ec if ec is not None else fc,
            alpha=alpha,
            linewidth=1.4,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight=weight,
    )


# ---------------------------------------------------------------------------
# Fig 1: Cooperative vs Competitive vs Mixed scenarios
# ---------------------------------------------------------------------------
def fig1_scenarios() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))

    titles = ["Cooperative\n(team reward)", "Competitive\n(zero-sum)",
              "Mixed-motive\n(general-sum)"]
    colors = [GREEN, RED, ORANGE]
    examples = ["Hide-and-seek\nMulti-robot pickup\nStarCraft micro",
                "Go, Chess, Poker\n1v1 fighting games\nAuctions",
                "Traffic, markets\nNegotiation\nResource sharing"]

    # Use 2x2 payoff matrices as the iconic visual for each setting
    payoffs = [
        # Cooperative: identical rewards
        np.array([[(3, 3), (0, 0)], [(0, 0), (3, 3)]], dtype=object),
        # Zero-sum (matching pennies-like)
        np.array([[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]], dtype=object),
        # Mixed-motive: prisoner's dilemma
        np.array([[(3, 3), (0, 5)], [(5, 0), (1, 1)]], dtype=object),
    ]

    for ax, title, c, ex, P in zip(axes, titles, colors, examples, payoffs):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_title(title, fontsize=12, color=c, weight="bold")

        # Payoff matrix grid
        x0, y0, cell = 0.6, 1.0, 1.4
        # column labels (agent 2)
        ax.text(x0 + cell * 0.5, y0 + cell * 2 + 0.25, "B: a₁",
                ha="center", fontsize=9, color=GRAY, weight="bold")
        ax.text(x0 + cell * 1.5, y0 + cell * 2 + 0.25, "B: a₂",
                ha="center", fontsize=9, color=GRAY, weight="bold")
        ax.text(x0 - 0.3, y0 + cell * 1.5, "A: a₁",
                ha="right", va="center", fontsize=9, color=GRAY, weight="bold")
        ax.text(x0 - 0.3, y0 + cell * 0.5, "A: a₂",
                ha="right", va="center", fontsize=9, color=GRAY, weight="bold")

        for i in range(2):
            for j in range(2):
                rA, rB = P[i, j]
                ax.add_patch(Rectangle((x0 + j * cell, y0 + (1 - i) * cell),
                                       cell, cell, facecolor=c, alpha=0.10,
                                       edgecolor=c, lw=1.3))
                ax.text(x0 + j * cell + cell / 2,
                        y0 + (1 - i) * cell + cell / 2,
                        f"({rA:+d}, {rB:+d})",
                        ha="center", va="center",
                        fontsize=12, color=DARK, weight="bold")

        ax.text(2.0, 0.45, ex, ha="center", va="top",
                fontsize=8.5, color=GRAY, style="italic")

    fig.suptitle("Three Multi-Agent Regimes  (payoffs as (rA, rB))",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_scenarios.png")


# ---------------------------------------------------------------------------
# Fig 2: CTDE (Centralized Training, Decentralized Execution)
# ---------------------------------------------------------------------------
def fig2_ctde() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.5))

    # --- Panel 1: Centralized Training -------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    ax.set_title("Centralized Training\n(uses global state s + all actions)",
                 fontsize=12, color=BLUE, weight="bold")

    # global critic
    _box(ax, 3.0, 6.2, 4.0, 1.2, "Centralised Critic / Mixer\n"
         r"$Q_\mathrm{tot}(s, a_1, a_2, a_3)$",
         fc=BLUE, alpha=0.20, fontsize=10)

    # state arrow
    _box(ax, 0.2, 6.4, 2.2, 0.8, "global state  s", fc=GRAY, alpha=0.20,
         fontsize=10)
    _arrow(ax, 2.4, 6.8, 3.0, 6.8, color=GRAY)

    # three agents
    for i, x in enumerate([1.2, 4.5, 7.8]):
        _box(ax, x, 3.0, 1.8, 1.1, f"Agent {i + 1}\n"
             r"$\pi_i(\cdot|o_i)$", fc=GREEN, alpha=0.20)
        # local obs
        _box(ax, x + 0.1, 1.4, 1.6, 0.7, f"obs $o_{i + 1}$",
             fc=LIGHT, alpha=0.6, fontsize=9, weight="normal")
        _arrow(ax, x + 0.9, 2.1, x + 0.9, 3.0, color=GRAY)
        # action up
        _arrow(ax, x + 0.9, 4.1, x + 0.9, 6.2, color=GREEN)
        ax.text(x + 1.05, 5.1, f"$a_{i + 1}$", color=GREEN, fontsize=10,
                weight="bold")

    # gradient feedback
    ax.annotate("", xy=(2.0, 3.7), xytext=(3.4, 6.2),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(1.6, 4.9, "∇  policy/Q\nupdate", color=ORANGE, fontsize=9,
            weight="bold", ha="center")

    # --- Panel 2: Decentralized Execution ----------------------------------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    ax.set_title("Decentralized Execution\n(each agent uses only its own obs)",
                 fontsize=12, color=PURPLE, weight="bold")

    for i, x in enumerate([1.2, 4.5, 7.8]):
        _box(ax, x, 3.0, 1.8, 1.1, f"Agent {i + 1}\n"
             r"$\pi_i(\cdot|o_i)$", fc=PURPLE, alpha=0.20)
        _box(ax, x + 0.1, 1.4, 1.6, 0.7, f"obs $o_{i + 1}$",
             fc=LIGHT, alpha=0.6, fontsize=9, weight="normal")
        _arrow(ax, x + 0.9, 2.1, x + 0.9, 3.0, color=GRAY)
        _arrow(ax, x + 0.9, 4.1, x + 0.9, 5.3, color=PURPLE)
        ax.text(x + 1.05, 4.7, f"$a_{i + 1}$", color=PURPLE, fontsize=10,
                weight="bold")
        # crossed-out global state
        ax.text(x + 0.9, 6.2, "no global s\nno teammate a", ha="center",
                fontsize=8.5, color=GRAY, style="italic")

    ax.text(5.0, 7.6, "At test time the centralised critic is discarded.",
            ha="center", color=DARK, fontsize=10, weight="bold")

    fig.suptitle("CTDE  —  the dominant paradigm in modern cooperative MARL",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_ctde.png")


# ---------------------------------------------------------------------------
# Fig 3: QMIX architecture
# ---------------------------------------------------------------------------
def fig3_qmix() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.5),
                             gridspec_kw={"width_ratios": [1.4, 1.0]})

    # --- Panel 1: architecture ---------------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    ax.set_title("QMIX: monotonic value factorisation",
                 fontsize=12, color=BLUE, weight="bold")

    # Per-agent Q networks
    agent_x = [0.6, 3.4, 6.2]
    for i, x in enumerate(agent_x):
        _box(ax, x, 1.0, 1.7, 0.8, f"obs $o_{i + 1}$", fc=LIGHT, alpha=0.6,
             fontsize=9, weight="normal")
        _box(ax, x, 2.2, 1.7, 1.0, f"Agent net\n$Q_{i + 1}(o_{i + 1}, \\cdot)$",
             fc=GREEN, alpha=0.20)
        _arrow(ax, x + 0.85, 1.8, x + 0.85, 2.2, color=GRAY)
        # Q value
        _arrow(ax, x + 0.85, 3.2, x + 0.85, 4.4, color=GREEN)
        ax.text(x + 0.95, 3.75, f"$Q_{i + 1}$", color=GREEN, fontsize=10,
                weight="bold")

    # Mixing network (positive weights)
    _box(ax, 0.6, 4.4, 7.3, 1.4,
         "Mixing network (weights $\\geq 0$  →  monotonic)",
         fc=BLUE, alpha=0.20, fontsize=10)

    # global Q
    _arrow(ax, 4.25, 5.8, 4.25, 6.5, color=BLUE)
    _box(ax, 2.85, 6.5, 2.8, 0.9,
         r"$Q_\mathrm{tot}(s, \mathbf{a})$", fc=BLUE, alpha=0.30, fontsize=11)

    # Hypernetwork
    _box(ax, 8.2, 4.0, 1.6, 1.5,
         "Hyper-\nnetwork", fc=PURPLE, alpha=0.25, fontsize=10)
    _box(ax, 8.2, 6.1, 1.6, 0.8, "global state  s", fc=GRAY, alpha=0.20,
         fontsize=9, weight="normal")
    _arrow(ax, 9.0, 6.1, 9.0, 5.5, color=GRAY)
    _arrow(ax, 8.2, 4.75, 7.9, 4.95, color=PURPLE)
    ax.text(7.85, 5.3, "weights\n+ biases", color=PURPLE, fontsize=8.5,
            ha="right", weight="bold")

    # --- Panel 2: monotonicity property ------------------------------------
    ax = axes[1]
    ax.set_title(r"Monotonicity: $\partial Q_\mathrm{tot} / \partial Q_i \geq 0$",
                 fontsize=12, color=PURPLE, weight="bold")
    q1 = np.linspace(0, 1, 60)
    q2 = np.linspace(0, 1, 60)
    Q1, Q2 = np.meshgrid(q1, q2)
    # state-dependent positive weights (illustration)
    w1, w2, b = 1.4, 0.7, 0.2
    Qtot = w1 * Q1 + w2 * Q2 + 0.6 * np.sqrt(np.clip(Q1 * Q2, 0, None)) + b
    cs = ax.contourf(Q1, Q2, Qtot, levels=14, cmap="Blues")
    ax.contour(Q1, Q2, Qtot, levels=8, colors=DARK, linewidths=0.5,
               alpha=0.4)
    fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$Q_\mathrm{tot}$")
    # arg-max consistency arrow
    ax.plot([0.05, 0.95], [0.05, 0.95], color=ORANGE, lw=2.0,
            label="argmax direction")
    ax.scatter([0.95], [0.95], s=140, color=ORANGE, zorder=4,
               edgecolor=DARK, linewidth=1.2)
    ax.text(0.55, 0.30,
            "Greedy on each $Q_i$\n=  greedy on $Q_\\mathrm{tot}$",
            color=DARK, fontsize=9.5, weight="bold")
    ax.set_xlabel(r"$Q_1$"); ax.set_ylabel(r"$Q_2$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("QMIX  —  decomposes a global Q into per-agent Qs while "
                 "preserving the joint argmax",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_qmix.png")


# ---------------------------------------------------------------------------
# Fig 4: COMA counterfactual baseline
# ---------------------------------------------------------------------------
def fig4_coma() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})

    # --- Panel 1: schematic of counterfactual reasoning -------------------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    ax.set_title("COMA counterfactual baseline for agent i",
                 fontsize=12, color=ORANGE, weight="bold")

    # state + others' actions
    _box(ax, 0.4, 5.6, 3.6, 1.3,
         r"state $s$ + others' actions $a_{-i}$",
         fc=GRAY, alpha=0.20, fontsize=10)

    # central critic
    _box(ax, 4.6, 5.6, 5.0, 1.3,
         r"Centralised critic  $Q(s, a_i, a_{-i})$",
         fc=BLUE, alpha=0.20, fontsize=10)
    _arrow(ax, 4.0, 6.25, 4.6, 6.25, color=GRAY)

    # action options
    actions = [r"$a_i^{(1)}$", r"$a_i^{(2)}$", r"$a_i^{(3)}$",
               r"$a_i^{(4)}$"]
    qs = [2.4, 3.8, 1.1, 2.6]
    pis = [0.10, 0.55, 0.05, 0.30]
    chosen = 1  # which action was actually taken
    for k, (a, q, pi) in enumerate(zip(actions, qs, pis)):
        x = 0.6 + k * 2.3
        col = ORANGE if k == chosen else BLUE
        alpha_box = 0.40 if k == chosen else 0.18
        _box(ax, x, 2.8, 1.8, 1.0,
             f"{a}\n$Q={q:.1f}$\n$\\pi={pi:.2f}$",
             fc=col, alpha=alpha_box, fontsize=9.5)
        _arrow(ax, x + 0.9, 3.8, x + 0.9, 5.6, color=col)

    # baseline computation
    baseline = sum(p * q for p, q in zip(pis, qs))
    A = qs[chosen] - baseline
    _box(ax, 1.0, 0.6, 8.0, 1.6,
         (r"baseline = $\sum_{a'} \pi(a'|o_i)\, Q(s, a', a_{-i}) = "
          f"{baseline:.2f}$" + "\n"
          r"counterfactual advantage  $A_i = Q(s, a_i, a_{-i}) - $ baseline"
          f"  $= {qs[chosen]:.1f} - {baseline:.2f} = {A:+.2f}$"),
         fc=ORANGE, alpha=0.18, fontsize=10)

    # --- Panel 2: bar chart of per-action advantage ------------------------
    ax = axes[1]
    ax.set_title("Per-action contribution\n(only the chosen action is "
                 "credited)",
                 fontsize=12, color=PURPLE, weight="bold")
    advs = [q - baseline for q in qs]
    cols = [ORANGE if k == chosen else BLUE for k in range(4)]
    bars = ax.bar(actions, advs, color=cols, alpha=0.80, edgecolor=DARK,
                  linewidth=1.0)
    ax.axhline(0, color=DARK, lw=0.8)
    for b, v in zip(bars, advs):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.06 if v >= 0 else -0.18),
                f"{v:+.2f}", ha="center", fontsize=9.5, weight="bold",
                color=DARK)
    ax.set_ylabel(r"$A_i$ = Q − baseline")
    ax.set_ylim(min(advs) - 0.6, max(advs) + 0.6)
    ax.text(0.02, 0.95,
            "Vanilla shared reward gives the\nsame credit to every agent.\n"
            "COMA isolates each agent's effect.",
            transform=ax.transAxes, va="top",
            fontsize=9, color=DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GRAY, alpha=0.9))

    fig.suptitle("COMA  —  fixes the credit-assignment problem with a "
                 "counterfactual baseline",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_coma.png")


# ---------------------------------------------------------------------------
# Fig 5: MADDPG actor-critic
# ---------------------------------------------------------------------------
def fig5_maddpg() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    ax.set_title("MADDPG  —  one centralised critic per agent, "
                 "decentralised actors",
                 fontsize=13, color=BLUE, weight="bold")

    n = 3
    actor_x = [1.0, 5.5, 10.0]
    critic_x = [1.0, 5.5, 10.0]
    colors = [GREEN, PURPLE, ORANGE]

    # global state strip
    _box(ax, 0.4, 6.6, 13.2, 0.9,
         r"global state $s = (o_1, o_2, o_3)$  +  joint action "
         r"$\mathbf{a} = (a_1, a_2, a_3)$  (training-time only)",
         fc=GRAY, alpha=0.20, fontsize=10)

    # actors
    for i in range(n):
        c = colors[i]
        # observation
        _box(ax, actor_x[i] - 0.1, 0.6, 3.4, 0.7,
             f"local obs $o_{i + 1}$", fc=LIGHT, alpha=0.6, fontsize=9.5,
             weight="normal")
        # actor
        _box(ax, actor_x[i], 1.7, 3.0, 1.2,
             f"Actor $\\mu_{i + 1}(o_{i + 1})$\n(decentralised)",
             fc=c, alpha=0.20)
        _arrow(ax, actor_x[i] + 1.5, 1.3, actor_x[i] + 1.5, 1.7, color=GRAY)
        # action
        _arrow(ax, actor_x[i] + 1.5, 2.9, actor_x[i] + 1.5, 4.0,
               color=c)
        ax.text(actor_x[i] + 1.7, 3.4, f"$a_{i + 1}$",
                color=c, fontsize=10, weight="bold")
        # critic
        _box(ax, critic_x[i], 4.0, 3.0, 2.0,
             f"Critic $Q_{i + 1}(s, a_1, a_2, a_3)$\n"
             f"(centralised, training only)",
             fc=BLUE, alpha=0.18)

        # gradient back to actor
        ax.annotate("", xy=(actor_x[i] + 0.4, 2.9),
                    xytext=(critic_x[i] + 0.4, 4.0),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4,
                                    connectionstyle="arc3,rad=0.35"))
        ax.text(actor_x[i] + 0.05, 3.4, r"$\nabla_\theta J$",
                color=ORANGE, fontsize=9, weight="bold")

        # state into critic
        _arrow(ax, critic_x[i] + 1.5, 6.6, critic_x[i] + 1.5, 6.0, color=GRAY)

    # joint-action sharing arrows between critics
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        ax.annotate("", xy=(critic_x[j], 5.0), xytext=(critic_x[i] + 3.0, 5.0),
                    arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9,
                                    alpha=0.4,
                                    connectionstyle="arc3,rad=-0.15"))

    ax.text(7.0, 7.9,
            "Critics see everyone → environment looks stationary; "
            "actors stay decentralised at test time.",
            ha="center", fontsize=10.5, color=DARK, weight="bold")

    fig.tight_layout()
    save(fig, "fig5_maddpg.png")


# ---------------------------------------------------------------------------
# Fig 6: Communication topologies
# ---------------------------------------------------------------------------
def fig6_communication() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.5))

    # 8 agents arranged in a circle
    n = 8
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    rng = np.random.default_rng(7)

    titles = ["Broadcast\n(every agent → every agent)",
              "Sparse / k-NN\n(each agent talks to 2 neighbours)",
              "Attention\n(soft, learned weights)"]
    colors = [BLUE, GREEN, PURPLE]

    for k, (ax, title, c) in enumerate(zip(axes, titles, colors)):
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.grid(False)
        ax.set_title(title, fontsize=11.5, color=c, weight="bold")

        # draw edges
        edges = []
        weights = []
        if k == 0:
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))
                    weights.append(1.0)
        elif k == 1:
            for i in range(n):
                edges.append((i, (i + 1) % n))
                edges.append((i, (i + 2) % n))
                weights.extend([1.0, 0.7])
        else:
            # attention: dense but heterogeneous weights
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w = float(rng.random())
                    if w > 0.55:
                        edges.append((i, j))
                        weights.append(w)

        for (i, j), w in zip(edges, weights):
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color=c, lw=0.4 + 2.4 * w, alpha=0.18 + 0.55 * w)

        # nodes
        for i, (x, y) in enumerate(pts):
            ax.add_patch(Circle((x, y), 0.15, facecolor=c, alpha=0.85,
                                edgecolor=DARK, lw=1.2))
            ax.text(x, y, str(i + 1), ha="center", va="center",
                    color="white", fontsize=10, weight="bold")

        # cost annotation
        cost = {0: r"O$(n^2)$ messages",
                1: r"O$(nk)$ messages",
                2: "soft routing\nlearned weights"}[k]
        ax.text(0, -1.5, cost, ha="center", fontsize=10, color=DARK,
                weight="bold")

    fig.suptitle("Multi-agent communication topologies "
                 "(scalability ↔ expressiveness)",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_communication.png")


# ---------------------------------------------------------------------------
# Fig 7: AlphaStar / OpenAI Five training pipeline
# ---------------------------------------------------------------------------
def fig7_league() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.5),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # --- Panel 1: league schematic ----------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    ax.set_title("AlphaStar league training",
                 fontsize=12, color=BLUE, weight="bold")

    # Three populations
    _box(ax, 0.4, 6.0, 3.0, 1.6,
         "Main agents\n(target of deployment)", fc=BLUE, alpha=0.25,
         fontsize=10)
    _box(ax, 3.6, 6.0, 3.0, 1.6,
         "Main exploiters\n(find weaknesses of\nMain agents)",
         fc=ORANGE, alpha=0.25, fontsize=10)
    _box(ax, 6.8, 6.0, 3.0, 1.6,
         "League exploiters\n(maintain strategy\ndiversity)",
         fc=PURPLE, alpha=0.25, fontsize=10)

    # Match-making arrows
    _box(ax, 2.6, 3.5, 4.8, 1.4,
         "Prioritised match-making\n(skill-based, anti-cycling)",
         fc=GRAY, alpha=0.20, fontsize=10)
    for x in [1.9, 5.1, 8.3]:
        _arrow(ax, x, 6.0, x * 0.5 + 2.5, 4.9, color=GRAY)

    # Frozen historical opponents pool (replay buffer of past checkpoints)
    _box(ax, 1.2, 1.0, 7.6, 1.4,
         "Frozen past checkpoints  (defeats strategy cycling: "
         "exploits do not vanish)",
         fc=GREEN, alpha=0.22, fontsize=10)

    _arrow(ax, 5.0, 3.5, 5.0, 2.4, color=GREEN)
    _arrow(ax, 5.0, 2.4, 5.0, 3.5, color=GREEN, style="-|>")

    # gradient feedback loop annotation
    ax.text(5.0, 8.1,
            "PFSP + self-play  →  prevents catastrophic forgetting",
            ha="center", color=DARK, fontsize=10, weight="bold")

    # --- Panel 2: scale / compute table -----------------------------------
    ax = axes[1]
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5)
    ax.set_title("Scale of two flagship systems",
                 fontsize=12, color=PURPLE, weight="bold")

    headers = ["", "AlphaStar (SC2, 2019)", "OpenAI Five (Dota 2, 2019)"]
    rows = [
        ["agents / team", "1 controller, 100s units", "5 heroes"],
        ["episode length", "~20 min, 10–20 k steps", "~45 min, ~80 k steps"],
        ["action space", "~10² discrete heads", "~1.7 × 10⁴ per hero"],
        ["self-play horizon", "~200 yr human-equiv.", "~180 yr per day"],
        ["compute", "~10⁴ TPU-days", "~770 PFLOP/s · days"],
        ["final level", "Grandmaster (top 0.15 %)", "Defeated world champions"],
    ]

    # header row
    y = 7.6
    for i, h in enumerate(headers):
        x = 0.2 + i * 3.27
        ax.text(x + 0.1, y, h, fontsize=9.5, weight="bold",
                color=BLUE if i > 0 else DARK)
    ax.plot([0.2, 9.8], [y - 0.25, y - 0.25], color=DARK, lw=0.8)

    for r, row in enumerate(rows):
        yy = y - 0.7 - r * 1.05
        # zebra stripes
        if r % 2 == 0:
            ax.add_patch(Rectangle((0.15, yy - 0.35), 9.7, 0.85,
                                   facecolor=LIGHT, alpha=0.4, edgecolor="none"))
        for i, cell in enumerate(row):
            x = 0.2 + i * 3.27
            ax.text(x + 0.1, yy, cell, fontsize=9, color=DARK,
                    weight="bold" if i == 0 else "normal",
                    va="center")

    fig.suptitle("League training & flagship MARL systems",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_league.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_scenarios()
    fig2_ctde()
    fig3_qmix()
    fig4_coma()
    fig5_maddpg()
    fig6_communication()
    fig7_league()
    print("All figures written to:")
    print(" ", EN_DIR)
    print(" ", ZH_DIR)


if __name__ == "__main__":
    main()
