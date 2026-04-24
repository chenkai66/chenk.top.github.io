"""
Figure generation script for Reinforcement Learning Part 01:
Fundamentals and Core Concepts.

Generates 7 figures shared by the EN and ZH articles. Each figure
isolates a single pedagogical idea so the reader can connect the
math, the code, and the picture without ambiguity.

Figures:
    fig1_agent_environment_loop   The classic agent <-> environment
                                  interaction loop (state, action,
                                  reward) cast as a "rider <-> bicycle"
                                  control diagram.
    fig2_mdp_framework            MDP as a small graph: states, actions,
                                  transition probabilities, rewards.
    fig3_bellman_recursion        Bellman equation as a backup tree:
                                  V(s) decomposed into r + gamma V(s').
    fig4_episode_rewards          Smoothed episode return curves under
                                  three discount factors gamma in
                                  {0.5, 0.9, 0.99} on a tabular task.
    fig5_value_vs_policy          GridWorld V-function heatmap with the
                                  greedy policy arrows overlaid.
    fig6_explore_vs_exploit       Cumulative regret of fixed epsilon
                                  policies on a 10-armed bandit, showing
                                  the explore/exploit tradeoff.
    fig7_discount_effect          Effective horizon 1/(1 - gamma) and
                                  gamma^k decay curves: myopic vs.
                                  farsighted agents.

Usage:
    python3 scripts/figures/reinforcement-learning/01-fundamentals.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders so the
    markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary -- agent / value
C_PURPLE = "#7c3aed"   # secondary -- environment / policy
C_GREEN = "#10b981"    # accent / good outcomes
C_AMBER = "#f59e0b"    # warning / highlight
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "en" / "reinforcement-learning"
    / "01-fundamentals-and-core-concepts"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "reinforcement-learning"
    / "01-基础与核心概念"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (EN_DIR, ZH_DIR):
        out = folder / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1 -- Agent <-> Environment loop (rider <-> bicycle metaphor)
# ---------------------------------------------------------------------------
def fig1_agent_environment_loop() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # Two big rounded boxes.
    agent_box = FancyBboxPatch(
        (0.6, 1.4), 3.2, 2.4,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=2.2, edgecolor=C_BLUE, facecolor="#dbeafe",
    )
    env_box = FancyBboxPatch(
        (6.2, 1.4), 3.2, 2.4,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=2.2, edgecolor=C_PURPLE, facecolor="#ede9fe",
    )
    ax.add_patch(agent_box)
    ax.add_patch(env_box)

    ax.text(2.2, 3.3, "Agent", ha="center", va="center",
            fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(2.2, 2.7, "(rider's brain)", ha="center", va="center",
            fontsize=11, style="italic", color=C_DARK)
    ax.text(2.2, 2.05, r"policy  $\pi(a\,|\,s)$",
            ha="center", va="center", fontsize=12, color=C_BLUE)

    ax.text(7.8, 3.3, "Environment", ha="center", va="center",
            fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(7.8, 2.7, "(bicycle + road)", ha="center", va="center",
            fontsize=11, style="italic", color=C_DARK)
    ax.text(7.8, 2.05, r"dynamics  $P(s'\,|\,s,a)$",
            ha="center", va="center", fontsize=12, color=C_PURPLE)

    # Action arrow: agent -> environment (top).
    a1 = FancyArrowPatch(
        (3.85, 3.45), (6.15, 3.45),
        arrowstyle="->", mutation_scale=22,
        linewidth=2.2, color=C_AMBER,
    )
    ax.add_patch(a1)
    ax.text(5.0, 3.78, r"action  $a_t$  (lean, steer, pedal)",
            ha="center", va="bottom", fontsize=11,
            color=C_AMBER, fontweight="bold")

    # State + reward arrow: environment -> agent (bottom).
    a2 = FancyArrowPatch(
        (6.15, 1.85), (3.85, 1.85),
        arrowstyle="->", mutation_scale=22,
        linewidth=2.2, color=C_GREEN,
    )
    ax.add_patch(a2)
    ax.text(5.0, 1.55,
            r"state  $s_{t+1}$  +  reward  $r_{t+1}$",
            ha="center", va="top", fontsize=11,
            color=C_GREEN, fontweight="bold")
    ax.text(5.0, 1.18, "(tilt angle, speed, did I fall?)",
            ha="center", va="top", fontsize=9.5,
            style="italic", color=C_DARK)

    ax.text(5.0, 4.75,
            "The closed loop of trial, feedback, and improvement",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=C_DARK)
    ax.text(5.0, 0.5,
            "Every decision the rider makes is a tiny RL step:  "
            r"observe $\to$ act $\to$ feel the consequence $\to$ update.",
            ha="center", va="center", fontsize=10.5,
            style="italic", color="#475569")

    save(fig, "fig1_agent_environment_loop")


# ---------------------------------------------------------------------------
# Fig 2 -- MDP as a tiny labelled graph
# ---------------------------------------------------------------------------
def fig2_mdp_framework() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # Three states laid out on a triangle.
    states = {
        "Balanced": (2.0, 4.2),
        "Wobbling": (5.0, 1.6),
        "Fallen":   (8.0, 4.2),
    }
    colors = {"Balanced": C_GREEN, "Wobbling": C_AMBER, "Fallen": "#ef4444"}

    for name, (x, y) in states.items():
        circ = plt.Circle((x, y), 0.55, facecolor="white",
                          edgecolor=colors[name], linewidth=2.6, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=C_DARK, zorder=4)

    def edge(p, q, label, color, rad=0.25, lab_offset=(0, 0)):
        arr = FancyArrowPatch(
            p, q, arrowstyle="->", mutation_scale=18,
            linewidth=1.8, color=color,
            connectionstyle=f"arc3,rad={rad}", zorder=2,
        )
        ax.add_patch(arr)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx + lab_offset[0], my + lab_offset[1], label,
                ha="center", va="center", fontsize=9.8,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="none", alpha=0.9))

    # Action: "steer correction"
    edge((2.55, 4.05), (4.55, 1.85),
         "a=steer\nP=0.7,  r=+1", C_BLUE, rad=-0.18,
         lab_offset=(-0.55, 0.2))
    edge((4.55, 1.45), (2.55, 4.05),
         "a=steer\nP=0.3,  r=-2", C_BLUE, rad=-0.18,
         lab_offset=(0.55, -0.2))

    # Action: "do nothing" -> wobble worsens
    edge((5.55, 1.85), (7.55, 4.05),
         "a=coast\nP=0.6,  r=-5", "#ef4444", rad=-0.18,
         lab_offset=(0.6, -0.1))
    edge((7.55, 4.05), (5.55, 1.85),
         "a=reset\nP=1.0,  r=0", C_GRAY, rad=-0.18,
         lab_offset=(-0.6, 0.1))

    # Self loop on Balanced
    arr = FancyArrowPatch(
        (1.55, 4.55), (1.55, 4.55), arrowstyle="->", mutation_scale=14,
        linewidth=1.6, color=C_GREEN,
        connectionstyle="arc3,rad=2.4", zorder=2,
    )
    ax.add_patch(arr)
    ax.text(0.55, 5.1, "a=hold\nr=+2", ha="center", va="center",
            fontsize=9.5, color=C_GREEN, fontweight="bold")

    ax.text(5.0, 0.55,
            r"MDP $\langle\,\mathcal{S},\,\mathcal{A},\,P,\,R,\,\gamma\,\rangle$ "
            r"  --  edges show $P(s'|s,a)$ and $R(s,a,s')$",
            ha="center", va="center", fontsize=12, color=C_DARK)
    ax.text(5.0, 5.25,
            "A bicycle MDP with 3 states and a few actions",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=C_DARK)

    save(fig, "fig2_mdp_framework")


# ---------------------------------------------------------------------------
# Fig 3 -- Bellman backup diagram
# ---------------------------------------------------------------------------
def fig3_bellman_recursion() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # Root state.
    ax.add_patch(plt.Circle((5.0, 4.6), 0.42,
                            facecolor=C_BLUE, edgecolor=C_DARK,
                            linewidth=1.6, zorder=3))
    ax.text(5.0, 4.6, r"$s$", ha="center", va="center",
            fontsize=15, fontweight="bold", color="white", zorder=4)
    ax.text(5.0, 5.25, r"$V^\pi(s) = ?$", ha="center", va="center",
            fontsize=12, color=C_BLUE, fontweight="bold")

    # Action nodes.
    actions = [(3.0, 3.2, r"$a_1$"), (5.0, 3.2, r"$a_2$"), (7.0, 3.2, r"$a_3$")]
    for (x, y, lab) in actions:
        ax.add_patch(plt.Rectangle((x - 0.32, y - 0.32), 0.64, 0.64,
                                   facecolor=C_PURPLE, edgecolor=C_DARK,
                                   linewidth=1.4, zorder=3))
        ax.text(x, y, lab, ha="center", va="center",
                fontsize=12, color="white", fontweight="bold", zorder=4)
        arr = FancyArrowPatch((5.0, 4.18), (x, y + 0.32),
                              arrowstyle="-", color=C_GRAY,
                              linewidth=1.4)
        ax.add_patch(arr)
        ax.text((5.0 + x) / 2 - 0.2, (4.18 + y + 0.32) / 2 + 0.05,
                r"$\pi(a|s)$", fontsize=9.5, color=C_DARK, alpha=0.7)

    # Next states from middle action.
    nexts = [(3.6, 1.4, r"$s'_1$", "r=+5"),
             (5.0, 1.4, r"$s'_2$", "r=-1"),
             (6.4, 1.4, r"$s'_3$", "r=+2")]
    for (x, y, lab, rew) in nexts:
        ax.add_patch(plt.Circle((x, y), 0.34,
                                facecolor="#dbeafe", edgecolor=C_BLUE,
                                linewidth=1.4, zorder=3))
        ax.text(x, y, lab, ha="center", va="center",
                fontsize=11, color=C_DARK, fontweight="bold", zorder=4)
        arr = FancyArrowPatch((5.0, 2.88), (x, y + 0.34),
                              arrowstyle="->", color=C_AMBER,
                              linewidth=1.4, mutation_scale=12)
        ax.add_patch(arr)
        ax.text(x + 0.05, (2.88 + y + 0.34) / 2,
                rew, fontsize=9, color=C_AMBER, fontweight="bold")

    # Equation strip at the bottom.
    ax.text(5.0, 0.45,
            r"$V^\pi(s) \;=\; \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)"
            r"\,[\,R(s,a,s') + \gamma\, V^\pi(s')\,]$",
            ha="center", va="center", fontsize=13.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor=C_AMBER, linewidth=1.2))

    ax.text(5.0, 5.35,
            "", ha="center")  # spacer
    ax.set_title("Bellman backup: today's value = expected reward + discounted tomorrow",
                 fontsize=13.5, fontweight="bold", color=C_DARK, pad=10)

    save(fig, "fig3_bellman_recursion")


# ---------------------------------------------------------------------------
# Fig 4 -- Episode reward curves under different gamma values
# ---------------------------------------------------------------------------
def _run_qlearning(gamma: float, episodes: int, seed: int) -> np.ndarray:
    """Tiny Q-learning on a 6x6 GridWorld; returns per-episode return."""
    rng = np.random.default_rng(seed)
    H = W = 6
    goal = (5, 5)
    obstacles = {(1, 2), (2, 2), (3, 2), (3, 3)}
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    Q = np.zeros((H, W, 4))
    alpha = 0.2
    eps = 0.15
    returns = np.zeros(episodes)

    for ep in range(episodes):
        s = (0, 0)
        total = 0.0
        for _ in range(80):
            if rng.random() < eps:
                a = int(rng.integers(0, 4))
            else:
                a = int(np.argmax(Q[s[0], s[1]]))
            dr, dc = actions[a]
            nr, nc = s[0] + dr, s[1] + dc
            if not (0 <= nr < H and 0 <= nc < W) or (nr, nc) in obstacles:
                nr, nc = s
            r = 10.0 if (nr, nc) == goal else -1.0
            done = (nr, nc) == goal
            target = r + gamma * np.max(Q[nr, nc]) * (0.0 if done else 1.0)
            Q[s[0], s[1], a] += alpha * (target - Q[s[0], s[1], a])
            total += r
            s = (nr, nc)
            if done:
                break
        returns[ep] = total
    return returns


def _smooth(x: np.ndarray, k: int = 25) -> np.ndarray:
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[k:] - c[:-k]) / k


def fig4_episode_rewards() -> None:
    episodes = 600
    gammas = [0.5, 0.9, 0.99]
    colors = [C_AMBER, C_BLUE, C_PURPLE]

    fig, ax = plt.subplots(figsize=(10, 5.4))

    for g, col in zip(gammas, colors):
        # Average over 5 seeds for a smooth curve.
        runs = np.stack([_run_qlearning(g, episodes, seed=s)
                         for s in range(5)])
        mean = runs.mean(axis=0)
        smoothed = _smooth(mean, 25)
        xs = np.arange(len(smoothed))
        ax.plot(xs, smoothed, color=col, linewidth=2.4,
                label=fr"$\gamma = {g}$")
        # Shaded standard deviation band.
        std_smoothed = _smooth(runs.std(axis=0), 25)
        ax.fill_between(xs, smoothed - std_smoothed,
                        smoothed + std_smoothed,
                        color=col, alpha=0.12)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Smoothed return per episode", fontsize=12)
    ax.set_title(
        "Discount factor controls how fast the agent learns to reach the goal",
        fontsize=13, fontweight="bold", color=C_DARK, pad=10,
    )
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.axhline(0, color=C_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(
        0.02, 0.97,
        "Lower gamma = myopic, slow credit assignment\n"
        "Higher gamma = farsighted, but variance is larger",
        transform=ax.transAxes, fontsize=10,
        va="top", ha="left", color="#475569", style="italic",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor=C_GRAY, alpha=0.85),
    )

    save(fig, "fig4_episode_rewards")


# ---------------------------------------------------------------------------
# Fig 5 -- Value function vs greedy policy
# ---------------------------------------------------------------------------
def fig5_value_vs_policy() -> None:
    H = W = 6
    goal = (5, 5)
    obstacles = {(1, 2), (2, 2), (3, 2), (3, 3)}
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    arrows = ["^", "v", "<", ">"]
    gamma = 0.9
    V = np.zeros((H, W))

    # Value iteration (deterministic transitions).
    for _ in range(200):
        new_V = V.copy()
        for r in range(H):
            for c in range(W):
                if (r, c) == goal or (r, c) in obstacles:
                    continue
                qs = []
                for dr, dc in actions:
                    nr, nc = r + dr, c + dc
                    if (not (0 <= nr < H and 0 <= nc < W)
                            or (nr, nc) in obstacles):
                        nr, nc = r, c
                    rew = 10.0 if (nr, nc) == goal else -1.0
                    done = (nr, nc) == goal
                    qs.append(rew + gamma * V[nr, nc] * (0.0 if done else 1.0))
                new_V[r, c] = max(qs)
        if np.max(np.abs(new_V - V)) < 1e-6:
            V = new_V
            break
        V = new_V

    # Mask obstacles for display.
    V_disp = V.copy()
    for (r, c) in obstacles:
        V_disp[r, c] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # Left: value heatmap.
    ax1 = axes[0]
    cmap = plt.get_cmap("Blues")
    im = ax1.imshow(V_disp, cmap=cmap, origin="upper",
                    vmin=np.nanmin(V_disp), vmax=np.nanmax(V_disp))
    for r in range(H):
        for c in range(W):
            if (r, c) in obstacles:
                ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            facecolor="#1f2937"))
                continue
            txt_color = "white" if V[r, c] > np.nanmean(V_disp) else C_DARK
            ax1.text(c, r, f"{V[r, c]:.1f}", ha="center", va="center",
                     fontsize=10, color=txt_color, fontweight="bold")
    ax1.text(goal[1], goal[0], "G", ha="center", va="center",
             fontsize=15, color=C_AMBER, fontweight="bold")
    ax1.set_title(r"Value function $V^*(s)$",
                  fontsize=12.5, fontweight="bold", color=C_DARK)
    ax1.set_xticks([]); ax1.set_yticks([])
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Right: greedy policy arrows.
    ax2 = axes[1]
    ax2.set_xlim(-0.5, W - 0.5)
    ax2.set_ylim(H - 0.5, -0.5)
    ax2.set_aspect("equal")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_facecolor("#f1f5f9")
    for r in range(H):
        for c in range(W):
            if (r, c) in obstacles:
                ax2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            facecolor="#1f2937"))
                continue
            if (r, c) == goal:
                ax2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            facecolor="#fde68a"))
                ax2.text(c, r, "G", ha="center", va="center",
                         fontsize=15, color=C_AMBER, fontweight="bold")
                continue
            qs = []
            for dr, dc in actions:
                nr, nc = r + dr, c + dc
                if (not (0 <= nr < H and 0 <= nc < W)
                        or (nr, nc) in obstacles):
                    nr, nc = r, c
                rew = 10.0 if (nr, nc) == goal else -1.0
                done = (nr, nc) == goal
                qs.append(rew + gamma * V[nr, nc] * (0.0 if done else 1.0))
            best = int(np.argmax(qs))
            dr, dc = actions[best]
            ax2.annotate("", xy=(c + 0.32 * dc, r + 0.32 * dr),
                         xytext=(c - 0.32 * dc, r - 0.32 * dr),
                         arrowprops=dict(arrowstyle="->",
                                         color=C_PURPLE, lw=2.0))
    # Faint grid lines.
    for k in range(W + 1):
        ax2.axvline(k - 0.5, color="white", lw=1.2)
    for k in range(H + 1):
        ax2.axhline(k - 0.5, color="white", lw=1.2)
    ax2.set_title(r"Greedy policy $\pi^*(s) = \arg\max_a Q^*(s,a)$",
                  fontsize=12.5, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Value tells you 'how good is here'; policy tells you 'where to go next'",
        fontsize=13.5, fontweight="bold", color=C_DARK, y=1.02,
    )
    fig.tight_layout()

    save(fig, "fig5_value_vs_policy")


# ---------------------------------------------------------------------------
# Fig 6 -- Exploration vs exploitation (10-armed bandit, regret)
# ---------------------------------------------------------------------------
def fig6_explore_vs_exploit() -> None:
    rng = np.random.default_rng(0)
    K = 10
    steps = 1000
    runs = 200
    eps_values = [0.0, 0.01, 0.1, 0.3]
    colors = [C_GRAY, C_GREEN, C_BLUE, C_AMBER]

    fig, ax = plt.subplots(figsize=(10, 5.4))

    for eps, col in zip(eps_values, colors):
        cum_regret = np.zeros(steps)
        for _ in range(runs):
            true_means = rng.normal(0.0, 1.0, K)
            best = true_means.max()
            Q = np.zeros(K)
            N = np.zeros(K)
            regret = 0.0
            for t in range(steps):
                if rng.random() < eps:
                    a = int(rng.integers(0, K))
                else:
                    a = int(np.argmax(Q))
                r = rng.normal(true_means[a], 1.0)
                N[a] += 1
                Q[a] += (r - Q[a]) / N[a]
                regret += best - true_means[a]
                cum_regret[t] += regret
        cum_regret /= runs
        ax.plot(np.arange(steps), cum_regret,
                color=col, linewidth=2.2,
                label=fr"$\varepsilon = {eps}$")

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Cumulative regret  (lower = better)", fontsize=12)
    ax.set_title(
        "Exploration vs exploitation on a 10-armed bandit",
        fontsize=13, fontweight="bold", color=C_DARK, pad=10,
    )
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    ax.text(
        0.98, 0.05,
        r"$\varepsilon=0$: pure greedy locks onto the wrong arm"
        "\n"
        r"$\varepsilon=0.3$: keeps wasting pulls on bad arms"
        "\n"
        r"$\varepsilon \approx 0.01$ -- $0.1$: the sweet spot",
        transform=ax.transAxes, fontsize=10,
        va="bottom", ha="right", color="#475569", style="italic",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor=C_GRAY, alpha=0.9),
    )

    save(fig, "fig6_explore_vs_exploit")


# ---------------------------------------------------------------------------
# Fig 7 -- Discount factor effect (myopic vs farsighted)
# ---------------------------------------------------------------------------
def fig7_discount_effect() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Left: gamma^k decay curves.
    ax1 = axes[0]
    ks = np.arange(0, 60)
    gammas = [0.5, 0.7, 0.9, 0.99]
    colors = [C_AMBER, C_GREEN, C_BLUE, C_PURPLE]
    for g, col in zip(gammas, colors):
        ax1.plot(ks, g ** ks, color=col, linewidth=2.4,
                 label=fr"$\gamma = {g}$")
    ax1.axhline(0.1, color=C_GRAY, linestyle="--", linewidth=1.0,
                alpha=0.7)
    ax1.text(58, 0.115, "10% weight",
             ha="right", va="bottom", fontsize=9.5, color=C_GRAY)
    ax1.set_xlabel("Steps into the future  $k$", fontsize=12)
    ax1.set_ylabel(r"Weight on $r_{t+k}$  $= \gamma^k$", fontsize=12)
    ax1.set_title("How much the agent cares about future reward",
                  fontsize=12.5, fontweight="bold", color=C_DARK)
    ax1.legend(loc="upper right", fontsize=10.5, frameon=True)
    ax1.set_ylim(-0.02, 1.02)

    # Right: effective horizon 1/(1-gamma) bar chart.
    ax2 = axes[1]
    gs = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999])
    horizons = 1.0 / (1.0 - gs)
    bar_colors = [C_AMBER, C_AMBER, C_GREEN, C_BLUE,
                  C_BLUE, C_PURPLE, C_PURPLE]
    bars = ax2.bar(range(len(gs)), horizons, color=bar_colors,
                   edgecolor=C_DARK, linewidth=0.8)
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(gs)))
    ax2.set_xticklabels([f"{g}" for g in gs], fontsize=10)
    ax2.set_xlabel(r"Discount factor  $\gamma$", fontsize=12)
    ax2.set_ylabel(r"Effective horizon  $\frac{1}{1-\gamma}$  (log)",
                   fontsize=12)
    ax2.set_title("Myopic vs farsighted: how many steps the agent plans",
                  fontsize=12.5, fontweight="bold", color=C_DARK)
    for b, h in zip(bars, horizons):
        ax2.text(b.get_x() + b.get_width() / 2, h * 1.15,
                 f"{int(round(h))}", ha="center", va="bottom",
                 fontsize=10, color=C_DARK, fontweight="bold")

    # Legend strip explaining colour buckets.
    handles = [
        mpatches.Patch(color=C_AMBER, label="myopic (~few steps)"),
        mpatches.Patch(color=C_GREEN, label="balanced"),
        mpatches.Patch(color=C_BLUE, label="farsighted"),
        mpatches.Patch(color=C_PURPLE, label="very farsighted"),
    ]
    ax2.legend(handles=handles, loc="upper left",
               fontsize=9.5, frameon=True)

    fig.suptitle(
        r"The discount factor $\gamma$ sets the agent's planning horizon",
        fontsize=13.5, fontweight="bold", color=C_DARK, y=1.02,
    )
    fig.tight_layout()

    save(fig, "fig7_discount_effect")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    print()
    print("[1/7] fig1_agent_environment_loop ...")
    fig1_agent_environment_loop()
    print("[2/7] fig2_mdp_framework ...")
    fig2_mdp_framework()
    print("[3/7] fig3_bellman_recursion ...")
    fig3_bellman_recursion()
    print("[4/7] fig4_episode_rewards ...")
    fig4_episode_rewards()
    print("[5/7] fig5_value_vs_policy ...")
    fig5_value_vs_policy()
    print("[6/7] fig6_explore_vs_exploit ...")
    fig6_explore_vs_exploit()
    print("[7/7] fig7_discount_effect ...")
    fig7_discount_effect()
    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
