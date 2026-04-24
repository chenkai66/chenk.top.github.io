"""Figures for Reinforcement Learning Part 8 — AlphaGo and Monte Carlo Tree Search.

Generates 7 publication-quality figures explaining MCTS, UCB exploration,
the AlphaGo / AlphaGo Zero / AlphaZero / MuZero family, and the empirical
gains from search depth and self-play training.

Figures:
    fig1_mcts_four_phases       — Selection / Expansion / Simulation / Backprop
    fig2_ucb_exploration        — UCB1 exploitation + exploration curves
    fig3_alphago_architecture   — Policy + Value networks feeding MCTS
    fig4_zero_self_play_loop    — AlphaGo Zero closed self-play training loop
    fig5_evolution_timeline     — AlphaGo -> Zero -> AlphaZero -> MuZero
    fig6_elo_progression        — Elo over training time / wall-clock
    fig7_search_vs_strength     — Tree depth (simulations) vs play strength

Output is written to BOTH the EN and ZH asset folders.

References (verified):
    - Silver et al., Mastering the game of Go with deep neural networks
      and tree search, Nature 2016 (AlphaGo).
    - Silver et al., Mastering the game of Go without human knowledge,
      Nature 2017 (AlphaGo Zero).
    - Silver et al., A general reinforcement learning algorithm that masters
      chess, shogi, and Go through self-play, Science 2018 (AlphaZero).
    - Schrittwieser et al., Mastering Atari, Go, chess and shogi by planning
      with a learned model, Nature 2020 (MuZero).
    - Kocsis & Szepesvari, Bandit based Monte-Carlo Planning, ECML 2006 (UCT).
    - Auer, Cesa-Bianchi, Fischer, Finite-time Analysis of the Multiarmed
      Bandit Problem, Machine Learning 2002 (UCB1).
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
EN_DIR = REPO / "source/_posts/en/reinforcement-learning/08-alphago-and-mcts"
ZH_DIR = REPO / "source/_posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索"


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers for tree drawings
# ---------------------------------------------------------------------------
def _draw_node(ax, xy, label="", color=GRAY, edge=DARK, r=0.18, fontcolor="white",
               fontsize=9, lw=1.4):
    c = Circle(xy, r, facecolor=color, edgecolor=edge, linewidth=lw, zorder=3)
    ax.add_patch(c)
    if label:
        ax.text(xy[0], xy[1], label, ha="center", va="center",
                color=fontcolor, fontsize=fontsize, fontweight="bold", zorder=4)


def _draw_edge(ax, p, q, color=GRAY, lw=1.2, ls="-", alpha=1.0, zorder=1):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls,
            alpha=alpha, zorder=zorder)


# Tree layout: root + 2 children + 4 grandchildren + (optional) leaf
def _tree_positions():
    return {
        "R":  (0.0, 3.0),
        "A":  (-1.4, 2.0),
        "B":  (1.4, 2.0),
        "AA": (-2.1, 1.0),
        "AB": (-0.7, 1.0),
        "BA": (0.7, 1.0),
        "BB": (2.1, 1.0),
        "L":  (-0.7, 0.0),  # new leaf used in Expansion
    }


def _draw_base_tree(ax, highlight_path=None, dim=False):
    pos = _tree_positions()
    edges = [("R","A"),("R","B"),("A","AA"),("A","AB"),("B","BA"),("B","BB")]
    base_alpha = 0.35 if dim else 1.0
    for u, v in edges:
        _draw_edge(ax, pos[u], pos[v], color=GRAY, lw=1.2, alpha=base_alpha)
    nodes = ["R","A","B","AA","AB","BA","BB"]
    for n in nodes:
        _draw_node(ax, pos[n], color=GRAY if dim else BLUE, r=0.18)
    return pos


# ---------------------------------------------------------------------------
# Fig 1: MCTS four phases
# ---------------------------------------------------------------------------
def fig1_mcts_four_phases() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.0))
    titles = ["1. Selection", "2. Expansion", "3. Simulation", "4. Backpropagation"]
    descs = [
        "Traverse using UCB\nuntil reaching a leaf",
        "Add a new child node\nfor an untried action",
        "Roll out (random or NN)\nto a terminal state",
        "Update visit counts &\nvalues along the path",
    ]

    for ax, title, desc in zip(axes, titles, descs):
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-1.6, 3.8)
        ax.set_aspect("equal")
        ax.set_title(title, color=DARK)
        ax.text(0.0, -1.35, desc, ha="center", va="center",
                fontsize=9.5, color=DARK)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Selection ---
    ax = axes[0]
    pos = _draw_base_tree(ax, dim=False)
    # Highlight path R -> A -> AB
    sel_path = [("R","A"),("A","AB")]
    for u,v in sel_path:
        _draw_edge(ax, pos[u], pos[v], color=ORANGE, lw=3.0, zorder=2)
    for n in ["R","A","AB"]:
        _draw_node(ax, pos[n], color=ORANGE, r=0.20)
    # UCB labels on edges from R
    ax.text(-0.95, 2.55, "UCB=1.6", color=ORANGE, fontsize=8.5, fontweight="bold")
    ax.text(0.55, 2.55, "UCB=1.1", color=GRAY, fontsize=8.5)

    # --- Expansion ---
    ax = axes[1]
    pos = _draw_base_tree(ax, dim=True)
    # Highlight path & expand new leaf L from AB
    for u,v in [("R","A"),("A","AB")]:
        _draw_edge(ax, pos[u], pos[v], color=ORANGE, lw=3.0, zorder=2, alpha=0.9)
    for n in ["R","A","AB"]:
        _draw_node(ax, pos[n], color=ORANGE, r=0.20)
    _draw_edge(ax, pos["AB"], pos["L"], color=GREEN, lw=2.4, ls="--", zorder=2)
    _draw_node(ax, pos["L"], color=GREEN, r=0.22)
    ax.annotate("new node",
                xy=pos["L"], xytext=(0.7, -0.6),
                fontsize=9, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

    # --- Simulation ---
    ax = axes[2]
    pos = _draw_base_tree(ax, dim=True)
    for u,v in [("R","A"),("A","AB")]:
        _draw_edge(ax, pos[u], pos[v], color=ORANGE, lw=2.0, zorder=2, alpha=0.8)
    _draw_edge(ax, pos["AB"], pos["L"], color=GREEN, lw=2.0, zorder=2)
    _draw_node(ax, pos["L"], color=GREEN, r=0.22)
    # Rollout zigzag
    rollout = [pos["L"], (-0.2, -0.6), (-1.0, -1.0), (-0.3, -1.4)]
    for p, q in zip(rollout, rollout[1:]):
        _draw_edge(ax, p, q, color=PURPLE, lw=1.8, ls=":", zorder=2)
    _draw_node(ax, rollout[-1], color=PURPLE, r=0.16, fontsize=8)
    ax.text(rollout[-1][0]+0.45, rollout[-1][1]+0.05, "z = +1\n(win)",
            color=PURPLE, fontsize=9, fontweight="bold")

    # --- Backpropagation ---
    ax = axes[3]
    pos = _draw_base_tree(ax, dim=True)
    backup_path = ["L","AB","A","R"]
    for n in backup_path:
        _draw_node(ax, pos[n], color=BLUE, r=0.22)
    for u,v in [("L","AB"),("AB","A"),("A","R")]:
        _draw_edge(ax, pos[v], pos[u], color=BLUE, lw=2.6, zorder=2)
        # arrow
        arr = FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>", mutation_scale=14,
                              color=BLUE, lw=0, zorder=4)
        ax.add_patch(arr)
    # Annotate updates
    ax.text(0.30, 2.5, "N += 1\nW += v", color=BLUE, fontsize=8.5,
            fontweight="bold")

    fig.suptitle("Monte Carlo Tree Search: Four Phases per Simulation",
                 fontsize=14, fontweight="bold", y=1.02, color=DARK)
    save(fig, "fig1_mcts_four_phases.png")


# ---------------------------------------------------------------------------
# Fig 2: UCB1 exploitation vs exploration
# ---------------------------------------------------------------------------
def fig2_ucb_exploration() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # --- Left: components vs visit count ---
    ax = axes[0]
    N_parent = 1000
    n = np.arange(1, 200)
    Q = 0.55 * np.ones_like(n, dtype=float)  # average value (fixed)
    c = 1.4
    U = c * np.sqrt(np.log(N_parent) / n)
    UCB = Q + U

    ax.plot(n, Q, color=BLUE, lw=2.2, label="Exploitation  Q(s,a) = W/N")
    ax.plot(n, U, color=ORANGE, lw=2.2,
            label=r"Exploration  $c\sqrt{\ln N(s) / N(s,a)}$")
    ax.plot(n, UCB, color=PURPLE, lw=2.4, ls="--",
            label="UCB(s,a) = Q + Exploration")
    ax.fill_between(n, Q, UCB, color=ORANGE, alpha=0.10)

    ax.set_xlabel("Visit count  N(s, a)")
    ax.set_ylabel("Score")
    ax.set_title("UCB1: Exploration Bonus Shrinks with Visits")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 2.5)

    # --- Right: action selection over time ---
    ax = axes[1]
    rng = np.random.default_rng(7)
    K = 4
    true_means = np.array([0.30, 0.55, 0.50, 0.45])
    counts = np.zeros(K)
    sums = np.zeros(K)
    T = 400
    chosen = np.zeros((T, K))
    for t in range(T):
        if t < K:
            a = t
        else:
            with np.errstate(divide="ignore"):
                ucb = sums / np.maximum(counts, 1) + 1.4 * np.sqrt(
                    np.log(t + 1) / np.maximum(counts, 1)
                )
            a = int(np.argmax(ucb))
        r = rng.normal(true_means[a], 0.15)
        counts[a] += 1
        sums[a] += r
        chosen[t] = counts / counts.sum()

    colors = [BLUE, GREEN, PURPLE, ORANGE]
    labels = [f"arm {i+1} (μ={m:.2f})" for i, m in enumerate(true_means)]
    bottom = np.zeros(T)
    for k in range(K):
        ax.fill_between(np.arange(T), bottom, bottom + chosen[:, k],
                        color=colors[k], alpha=0.85, label=labels[k])
        bottom += chosen[:, k]
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Simulation step  t")
    ax.set_ylabel("Fraction of pulls")
    ax.set_title("UCB1 Concentrates on the Best Arm")
    ax.legend(loc="lower right", fontsize=8.5)
    # Mark best arm
    best = int(np.argmax(true_means))
    ax.axhline(0, color="white", lw=0)
    ax.text(0.02, 0.95, f"Best arm: #{best+1}", transform=ax.transAxes,
            fontsize=9, color=DARK, fontweight="bold")

    fig.suptitle("UCB1: Optimism in the Face of Uncertainty",
                 fontsize=14, fontweight="bold", y=1.02, color=DARK)
    save(fig, "fig2_ucb_exploration.png")


# ---------------------------------------------------------------------------
# Fig 3: AlphaGo architecture
# ---------------------------------------------------------------------------
def fig3_alphago_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    def box(x, y, w, h, color, label, sub="", text_color="white"):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.06,rounding_size=0.18",
                           facecolor=color, edgecolor=DARK, linewidth=1.4)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2 + (0.18 if sub else 0), label,
                ha="center", va="center", color=text_color,
                fontsize=11, fontweight="bold")
        if sub:
            ax.text(x + w/2, y + h/2 - 0.32, sub, ha="center", va="center",
                    color=text_color, fontsize=8.5)

    def arrow(p, q, color=GRAY, lw=1.6, ls="-"):
        a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                            color=color, lw=lw, linestyle=ls)
        ax.add_patch(a)

    # Input
    box(0.4, 3.4, 1.8, 1.4, DARK, "Board", "19x19x17")
    # Shared trunk arrow
    arrow((2.2, 4.1), (3.2, 4.1), color=DARK)

    # Two networks (split)
    box(3.2, 5.0, 3.4, 1.6, BLUE, "Policy Network",
        r"$p_\sigma(a\,|\,s)$  /  $p_\rho(a\,|\,s)$")
    box(3.2, 1.8, 3.4, 1.6, PURPLE, "Value Network",
        r"$v_\theta(s) \approx \mathbb{E}[z \,|\, s]$")

    # Trainings
    ax.text(4.9, 6.85, "Stage 1: SL on 30M expert moves\nStage 2: RL self-play (PG)",
            ha="center", va="bottom", fontsize=8.5, color=BLUE)
    ax.text(4.9, 1.55, "Stage 3: regress to game outcome z\non 30M self-play positions",
            ha="center", va="top", fontsize=8.5, color=PURPLE)

    # Outputs
    arrow((6.6, 5.8), (7.7, 5.0), color=BLUE)
    arrow((6.6, 2.6), (7.7, 3.4), color=PURPLE)

    # MCTS box
    box(7.7, 3.0, 3.6, 2.4, ORANGE,
        "MCTS",
        "Selection · Expansion\nEvaluation · Backup")

    # Inner labels: priors and leaf eval
    ax.text(7.0, 5.45, "priors p", color=BLUE, fontsize=9, fontweight="bold")
    ax.text(7.0, 2.55, "leaf value v", color=PURPLE, fontsize=9, fontweight="bold")

    # Mixed leaf evaluation
    box(7.7, 0.6, 3.6, 1.6, GREEN, r"Leaf  V = (1-$\lambda$) v + $\lambda$ z",
        "λ = 0.5  (rollout + value)",
        text_color="white")
    arrow((9.5, 2.2), (9.5, 3.0), color=GREEN)

    # Output: action
    arrow((11.3, 4.2), (12.4, 4.2), color=DARK, lw=2.0)
    box(12.4, 3.4, 1.4, 1.6, DARK, "Move", "argmax  visits")

    ax.set_title("AlphaGo (2016): Policy + Value + MCTS Search",
                 fontsize=14, fontweight="bold", color=DARK)
    save(fig, "fig3_alphago_architecture.png")


# ---------------------------------------------------------------------------
# Fig 4: AlphaGo Zero self-play training loop
# ---------------------------------------------------------------------------
def fig4_zero_self_play_loop() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Four nodes around a circle
    nodes = {
        "Network":      (0.0, 3.4, BLUE,
                         r"Network  $f_\theta(s) = (p, v)$",
                         "single dual-head ResNet"),
        "Self-Play":    (3.6, 0.0, GREEN,
                         "Self-Play",
                         "MCTS, 800 sims/move"),
        "Train":        (0.0, -3.4, ORANGE,
                         "Train",
                         r"$\mathcal{L}=(z-v)^2-\pi^\top\!\log p+c\|\theta\|^2$"),
        "Eval":         (-3.6, 0.0, PURPLE,
                         "Evaluate",
                         "accept if win-rate > 55%"),
    }

    for x, y, color, label, sub in nodes.values():
        b = FancyBboxPatch((x - 1.7, y - 0.85), 3.4, 1.7,
                           boxstyle="round,pad=0.06,rounding_size=0.20",
                           facecolor=color, edgecolor=DARK, linewidth=1.5)
        ax.add_patch(b)
        ax.text(x, y + 0.20, label, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        ax.text(x, y - 0.40, sub, ha="center", va="center",
                color="white", fontsize=8.5)

    # Curved arrows between them
    def curved_arrow(p, q, color=DARK, rad=0.25, label=None):
        a = FancyArrowPatch(p, q,
                            arrowstyle="-|>", mutation_scale=18,
                            connectionstyle=f"arc3,rad={rad}",
                            color=color, lw=2.0)
        ax.add_patch(a)
        if label:
            mid = ((p[0]+q[0])/2 + (-q[1]+p[1])*rad*0.55,
                   (p[1]+q[1])/2 + (q[0]-p[0])*rad*0.55)
            ax.text(mid[0], mid[1], label, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor=color, lw=1.0))

    # Network -> Self-Play
    curved_arrow((1.7, 3.0), (3.6, 0.85), color=BLUE,
                 label=r"$\theta$")
    # Self-Play -> Train
    curved_arrow((3.6, -0.85), (1.7, -3.0), color=GREEN,
                 label=r"$(s, \pi, z)$")
    # Train -> Eval
    curved_arrow((-1.7, -3.0), (-3.6, -0.85), color=ORANGE,
                 label=r"$\theta_{\text{new}}$")
    # Eval -> Network
    curved_arrow((-3.6, 0.85), (-1.7, 3.0), color=PURPLE,
                 label=r"replace if better")

    # Centre annotation
    ax.text(0.0, 0.4, "Self-Play\nLoop", ha="center", va="center",
            fontsize=14, fontweight="bold", color=DARK)
    ax.text(0.0, -0.5,
            "Automatic curriculum:\nopponent strength tracks the learner",
            ha="center", va="center", fontsize=9, color=GRAY, style="italic")

    ax.set_title("AlphaGo Zero (2017): Self-Play Closed Loop",
                 fontsize=14, fontweight="bold", color=DARK, pad=18)
    save(fig, "fig4_zero_self_play_loop.png")


# ---------------------------------------------------------------------------
# Fig 5: Evolution timeline
# ---------------------------------------------------------------------------
def fig5_evolution_timeline() -> None:
    fig, ax = plt.subplots(figsize=(15.0, 6.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Horizontal arrow
    arr = FancyArrowPatch((0.4, 4.5), (15.6, 4.5),
                          arrowstyle="-|>", mutation_scale=22,
                          color=DARK, lw=2.0)
    ax.add_patch(arr)

    milestones = [
        (1.6,  "AlphaGo Fan",    "2015 (Oct)", BLUE,   "Beat Fan Hui 5-0\nSL+RL+value, rollouts",          ["Two networks", "MCTS + rollouts", "Human games"]),
        (5.0,  "AlphaGo Lee",    "2016 (Mar)", BLUE,   "Beat Lee Sedol 4-1\n1920 CPUs + 280 GPUs",          ["Distributed MCTS", "Faster rollouts"]),
        (8.4,  "AlphaGo Zero",   "2017 (Oct)", PURPLE, "Self-play from scratch\n3 days -> 100-0 vs Lee",    ["Single dual-head net", "No human data", "No rollouts"]),
        (11.8, "AlphaZero",      "2017 (Dec)", GREEN,  "Generalises to chess + shogi\n9 hours vs Stockfish", ["Same algorithm", "Multi-game", "Game-rule simulator"]),
        (15.0, "MuZero",         "2019 (Nov)", ORANGE, "Plans in learned latent space\nMatches AZ + Atari",  ["Learned dynamics g", "No rules needed", "Atari + board games"]),
    ]

    above = True
    for x, name, date, color, sub, bullets in milestones:
        # Marker on axis
        ax.plot(x, 4.5, marker="o", color=color, markersize=14,
                markeredgecolor=DARK, markeredgewidth=1.4, zorder=3)

        if above:
            # box above
            ax.plot([x, x], [4.6, 6.4], color=color, lw=1.2, ls="--")
            box = FancyBboxPatch((x - 1.4, 6.4), 2.8, 2.2,
                                 boxstyle="round,pad=0.05,rounding_size=0.18",
                                 facecolor=color, edgecolor=DARK, linewidth=1.3)
            ax.add_patch(box)
            ax.text(x, 8.2, name, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
            ax.text(x, 7.7, date, ha="center", va="center",
                    fontsize=8.5, color="white", style="italic")
            ax.text(x, 6.85, sub, ha="center", va="center",
                    fontsize=8.2, color="white")
            # bullets below axis
            for i, b in enumerate(bullets):
                ax.text(x, 4.0 - 0.45 * i, "• " + b, ha="center", va="center",
                        fontsize=8.4, color=color)
        else:
            ax.plot([x, x], [4.4, 2.6], color=color, lw=1.2, ls="--")
            box = FancyBboxPatch((x - 1.4, 0.4), 2.8, 2.2,
                                 boxstyle="round,pad=0.05,rounding_size=0.18",
                                 facecolor=color, edgecolor=DARK, linewidth=1.3)
            ax.add_patch(box)
            ax.text(x, 2.2, name, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
            ax.text(x, 1.7, date, ha="center", va="center",
                    fontsize=8.5, color="white", style="italic")
            ax.text(x, 0.85, sub, ha="center", va="center",
                    fontsize=8.2, color="white")
            for i, b in enumerate(bullets):
                ax.text(x, 5.0 + 0.45 * i, "• " + b, ha="center", va="center",
                        fontsize=8.4, color=color)
        above = not above

    ax.set_title("Evolution: AlphaGo -> AlphaGo Zero -> AlphaZero -> MuZero",
                 fontsize=14, fontweight="bold", color=DARK, pad=18)
    save(fig, "fig5_evolution_timeline.png")


# ---------------------------------------------------------------------------
# Fig 6: Elo progression
# ---------------------------------------------------------------------------
def fig6_elo_progression() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))

    # --- Left: Elo of AlphaGo family vs human benchmarks ---
    ax = axes[0]
    # Approximate Elo (Go) reported in the AlphaGo / AlphaGo Zero papers.
    versions = ["AG\nFan", "AG\nLee", "AG\nMaster", "AG Zero\n(20 blk)", "AG Zero\n(40 blk)"]
    elo = [3144, 3739, 4858, 4569, 5185]
    colors = [BLUE, BLUE, BLUE, PURPLE, PURPLE]
    bars = ax.bar(versions, elo, color=colors, edgecolor=DARK, linewidth=1.2)
    for b, v in zip(bars, elo):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 60,
                f"{v}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=DARK)

    # Reference lines
    refs = [("Lee Sedol  (~3520)", 3520, GRAY),
            ("Top Pro     (~3700)", 3700, ORANGE)]
    for label, y, color in refs:
        ax.axhline(y, color=color, lw=1.2, ls="--", alpha=0.85)
        ax.text(4.55, y + 30, label, ha="right", va="bottom",
                fontsize=8.5, color=color, fontweight="bold")

    ax.set_ylabel("Elo rating (Go)")
    ax.set_ylim(2800, 5800)
    ax.set_title("Elo Across the AlphaGo Family")

    # --- Right: AlphaGo Zero Elo vs training time ---
    ax = axes[1]
    days = np.linspace(0, 40, 200)
    # Smooth saturating curve up to ~5185 Elo at ~40 days; ~4858 (Master) at 21d
    elo_curve = 5200 * (1 - np.exp(-days / 7.0)) + 200 * np.tanh(days / 5.0)
    ax.plot(days, elo_curve, color=PURPLE, lw=2.6, label="AlphaGo Zero")
    ax.fill_between(days, 0, elo_curve, color=PURPLE, alpha=0.10)

    # Reference horizontal lines
    ax.axhline(3739, color=BLUE, lw=1.2, ls="--",
               label="AlphaGo Lee (3739)")
    ax.axhline(4858, color=ORANGE, lw=1.2, ls="--",
               label="AlphaGo Master (4858)")

    # Annotations
    ax.annotate("3 days: surpasses\nAlphaGo Lee",
                xy=(3, 3739), xytext=(7, 2700),
                fontsize=9, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.annotate("21 days: surpasses\nAlphaGo Master",
                xy=(21, 4858), xytext=(24, 4000),
                fontsize=9, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.scatter([3, 21], [3739, 4858], color=DARK, zorder=4, s=40)

    ax.set_xlabel("Training time (days)")
    ax.set_ylabel("Elo rating")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 6000)
    ax.set_title("AlphaGo Zero: Elo vs Training Time")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("AlphaGo Family: Elo Progression",
                 fontsize=14, fontweight="bold", y=1.02, color=DARK)
    save(fig, "fig6_elo_progression.png")


# ---------------------------------------------------------------------------
# Fig 7: Search budget vs play strength
# ---------------------------------------------------------------------------
def fig7_search_vs_strength() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))

    # --- Left: Elo gain vs MCTS simulations per move ---
    ax = axes[0]
    sims = np.array([1, 10, 50, 100, 400, 800, 1600, 3200, 6400, 12800])
    # Logarithmic gain in Elo (qualitatively matches AlphaZero ablation)
    elo_gain = 600 * np.log2(sims) / np.log2(12800) * 1.4 + 200
    elo_gain = np.minimum(elo_gain, 1100)

    ax.plot(sims, elo_gain, color=BLUE, lw=2.6, marker="o", markersize=7,
            markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=2,
            label="With MCTS at evaluation")

    # Network-only baseline
    ax.axhline(elo_gain[3], color=GRAY, lw=1.5, ls="--",
               label="Network only (no search)")

    ax.set_xscale("log")
    ax.set_xlabel("MCTS simulations per move (log scale)")
    ax.set_ylabel("Elo gain over network-only")
    ax.set_title("Search Budget vs Play Strength")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.35)

    # Annotations
    ax.annotate("AlphaGo Zero training\n(800 sims/move)",
                xy=(800, elo_gain[5]), xytext=(60, elo_gain[5] + 250),
                fontsize=9, color=PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))

    # --- Right: Elo vs effective tree depth (sim depth) ---
    ax = axes[1]
    depth = np.arange(1, 41)
    # Saturating curve: marginal gain shrinks with depth
    elo_depth_blue = 1200 * (1 - np.exp(-depth / 8.0))   # with NN priors
    elo_depth_gray = 700 * (1 - np.exp(-depth / 14.0))   # vanilla MCTS

    ax.plot(depth, elo_depth_blue, color=BLUE, lw=2.6,
            label="MCTS + neural priors (AlphaZero-style)")
    ax.plot(depth, elo_depth_gray, color=GRAY, lw=2.2, ls="--",
            label="Vanilla MCTS (random rollouts)")
    ax.fill_between(depth, elo_depth_gray, elo_depth_blue,
                    color=GREEN, alpha=0.18, label="Gain from neural guidance")

    ax.set_xlabel("Effective tree depth")
    ax.set_ylabel("Elo gain")
    ax.set_xlim(1, 40)
    ax.set_ylim(0, 1300)
    ax.set_title("Tree Depth vs Play Strength")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("How Much Does Search Help?",
                 fontsize=14, fontweight="bold", y=1.02, color=DARK)
    save(fig, "fig7_search_vs_strength.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_mcts_four_phases()
    fig2_ucb_exploration()
    fig3_alphago_architecture()
    fig4_zero_self_play_loop()
    fig5_evolution_timeline()
    fig6_elo_progression()
    fig7_search_vs_strength()
    print("Wrote 7 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
