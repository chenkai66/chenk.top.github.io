"""
Figure generation script for Reinforcement Learning Part 02:
Q-Learning and Deep Q-Networks (DQN).

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly.

Figures:
    fig1_qtable_gridworld         A 4x4 gridworld with Q-values shown as four
                                  triangles per cell (one per action) plus the
                                  greedy policy arrows. Makes "Q-table" tangible.
    fig2_cliff_walking_curves     Q-Learning learning curves on Cliff Walking
                                  for three exploration rates, showing
                                  convergence behaviour.
    fig3_dqn_architecture         Block diagram of the DQN convolutional
                                  network: 84x84x4 frames -> conv stack ->
                                  FC -> per-action Q-values.
    fig4_replay_buffer            Conceptual diagram of the experience replay
                                  buffer: stream-in transitions, FIFO storage,
                                  random mini-batch sample-out.
    fig5_target_network           Time series of the online network and the
                                  lagging target network parameter, showing
                                  how the periodic copy stabilises the target.
    fig6_double_vs_vanilla        Q-value estimates over training steps for
                                  vanilla DQN (overestimates) vs Double DQN
                                  (close to true), illustrating the bias fix.
    fig7_atari_benchmark          Median human-normalised score on Atari for
                                  DQN, Double DQN, Dueling DQN, PER and Rainbow.

Usage:
    python3 scripts/figures/reinforcement-learning/02-q-learning-dqn.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# ---------------------------------------------------------------------------
# Shared aesthetic style (chenk-site)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]      # primary
C_PURPLE = COLORS["accent"]    # secondary
C_GREEN = COLORS["success"]     # accent / good
C_AMBER = COLORS["warning"]     # warning / highlight
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_LIGHT = COLORS["grid"]
C_DARK = COLORS["text"]
C_BG = COLORS["bg"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "reinforcement-learning"
    / "02-q-learning-and-dqn"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "reinforcement-learning"
    / "02-Q-Learning与深度Q网络"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Q-table on a 4x4 gridworld
# ---------------------------------------------------------------------------
def fig1_qtable_gridworld() -> None:
    """Each cell drawn as four triangles (U/R/D/L) coloured by Q-value,
    plus a black arrow indicating the greedy action."""
    n = 4
    goal = (0, 3)
    pit = (2, 1)

    # Build a plausible Q-table by running value iteration on a tiny MDP.
    gamma = 0.9
    rewards = np.full((n, n), -0.04)
    rewards[goal] = 1.0
    rewards[pit] = -1.0
    terminal = np.zeros((n, n), dtype=bool)
    terminal[goal] = True
    terminal[pit] = True

    # Actions: 0=U, 1=R, 2=D, 3=L
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    V = np.zeros((n, n))
    for _ in range(200):
        V_new = V.copy()
        for r in range(n):
            for c in range(n):
                if terminal[r, c]:
                    V_new[r, c] = rewards[r, c]
                    continue
                best = -1e9
                for dr, dc in moves:
                    rr, cc = max(0, min(n - 1, r + dr)), max(0, min(n - 1, c + dc))
                    best = max(best, rewards[r, c] + gamma * V[rr, cc])
                V_new[r, c] = best
        V = V_new

    Q = np.zeros((n, n, 4))
    for r in range(n):
        for c in range(n):
            for a, (dr, dc) in enumerate(moves):
                rr, cc = max(0, min(n - 1, r + dr)), max(0, min(n - 1, c + dc))
                if terminal[r, c]:
                    Q[r, c, a] = rewards[r, c]
                else:
                    Q[r, c, a] = rewards[r, c] + gamma * V[rr, cc]

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    vmin, vmax = -1.0, 1.0
    cmap = plt.cm.RdYlGn

    for r in range(n):
        for c in range(n):
            x, y = c, n - 1 - r
            cx, cy = x + 0.5, y + 0.5
            corners = [
                (x, y),
                (x + 1, y),
                (x + 1, y + 1),
                (x, y + 1),
            ]
            # 4 triangles (U=top, R=right, D=bottom, L=left)
            tris = {
                0: [corners[3], corners[2], (cx, cy)],  # top -> Up
                1: [corners[2], corners[1], (cx, cy)],  # right -> Right
                2: [corners[1], corners[0], (cx, cy)],  # bottom -> Down
                3: [corners[0], corners[3], (cx, cy)],  # left -> Left
            }
            if (r, c) == goal:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=C_GREEN, alpha=0.85))
                ax.text(cx, cy, "GOAL\n+1.0", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")
                continue
            if (r, c) == pit:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=C_RED, alpha=0.85))
                ax.text(cx, cy, "PIT\n-1.0", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")
                continue
            for a, pts in tris.items():
                v = (Q[r, c, a] - vmin) / (vmax - vmin)
                color = cmap(v)
                ax.add_patch(Polygon(pts, facecolor=color, edgecolor="white", linewidth=1.0))
                # value label per triangle
                tx = np.mean([p[0] for p in pts[:2]]) * 0.35 + cx * 0.65
                ty = np.mean([p[1] for p in pts[:2]]) * 0.35 + cy * 0.65
                ax.text(tx, ty, f"{Q[r, c, a]:.2f}", ha="center", va="center",
                        fontsize=7.5, color=C_DARK)
            # Greedy arrow
            best_a = int(np.argmax(Q[r, c]))
            dr, dc = moves[best_a]
            ax.annotate(
                "",
                xy=(cx + 0.28 * dc, cy - 0.28 * dr),
                xytext=(cx - 0.05 * dc, cy + 0.05 * dr),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2.0,
                                mutation_scale=14),
            )

    # Outer grid lines
    for k in range(n + 1):
        ax.plot([0, n], [k, k], color=C_DARK, lw=1.2)
        ax.plot([k, k], [0, n], color=C_DARK, lw=1.2)

    ax.set_xlim(-0.05, n + 0.05)
    ax.set_ylim(-0.05, n + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Q-Table on a 4x4 Gridworld\n"
                 "Each cell stores Q(s, a) for four actions; arrow = greedy choice",
                 fontsize=13, fontweight="bold", pad=12)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Q-value (red = bad, green = good)", fontsize=10)

    plt.tight_layout()
    _save(fig, "fig1_qtable_gridworld")


# ---------------------------------------------------------------------------
# Figure 2: Cliff Walking learning curves
# ---------------------------------------------------------------------------
def fig2_cliff_walking_curves() -> None:
    """Synthesise plausible learning curves for three epsilon values on the
    Cliff Walking task. Reward curve + smoothed mean."""
    rng = np.random.default_rng(42)
    n_episodes = 500

    def curve(asymptote: float, tau: float, noise: float) -> np.ndarray:
        ep = np.arange(n_episodes)
        base = asymptote + (-200 - asymptote) * np.exp(-ep / tau)
        return base + rng.normal(0, noise, size=n_episodes)

    eps_curves = {
        "epsilon = 0.01 (greedy)": curve(-13, 90, 18),
        "epsilon = 0.10 (balanced)": curve(-17, 60, 22),
        "epsilon = 0.50 (over-exploring)": curve(-50, 40, 30),
    }

    def smooth(x, w=20):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    colors = [C_BLUE, C_GREEN, C_AMBER]
    for (label, raw), color in zip(eps_curves.items(), colors):
        ax.plot(raw, color=color, alpha=0.18, lw=1.0)
        sm = smooth(raw, 25)
        ax.plot(np.arange(len(sm)) + 12, sm, color=color, lw=2.4, label=label)

    ax.axhline(-13, color=C_DARK, ls="--", lw=1.0, alpha=0.5)
    ax.text(n_episodes - 5, -10, "Optimal: -13 (shortest safe path)",
            ha="right", va="bottom", fontsize=9, color=C_DARK)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total reward per episode", fontsize=11)
    ax.set_title("Q-Learning on Cliff Walking: Effect of Exploration Rate",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(-220, 5)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig2_cliff_walking_curves")


# ---------------------------------------------------------------------------
# Figure 3: DQN architecture diagram
# ---------------------------------------------------------------------------
def fig3_dqn_architecture() -> None:
    """Block diagram of the Atari DQN: 4 stacked frames -> 3 conv layers
    -> FC 512 -> 18 Q-values, with shapes annotated."""
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def block(x, y, w, h, color, edge=C_DARK, alpha=0.85):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, edgecolor=edge, lw=1.4, alpha=alpha,
        ))

    def arrow(x1, x2, y=3.0):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.8,
                                    mutation_scale=16))

    # Input frames (4 stacked)
    for i in range(4):
        block(0.2 + i * 0.12, 1.6 + i * 0.12, 1.5, 2.6, C_GRAY, alpha=0.6)
    ax.text(1.35, 0.9, "Input: 4 x 84 x 84\n(stacked frames)",
            ha="center", fontsize=9.5, color=C_DARK)

    arrow(2.4, 3.0)

    # Conv1
    block(3.0, 1.2, 1.6, 3.6, C_BLUE, alpha=0.85)
    ax.text(3.8, 5.0, "Conv 8x8\nstride 4\n32 filters", ha="center",
            fontsize=9.5, color=C_DARK, fontweight="bold")
    ax.text(3.8, 0.85, "32 x 20 x 20", ha="center", fontsize=9, color=C_DARK)

    arrow(4.7, 5.3)

    # Conv2
    block(5.4, 1.5, 1.5, 3.0, C_BLUE, alpha=0.7)
    ax.text(6.15, 4.7, "Conv 4x4\nstride 2\n64 filters", ha="center",
            fontsize=9.5, color=C_DARK, fontweight="bold")
    ax.text(6.15, 1.15, "64 x 9 x 9", ha="center", fontsize=9, color=C_DARK)

    arrow(7.0, 7.7)

    # Conv3
    block(7.8, 1.8, 1.4, 2.4, C_BLUE, alpha=0.55)
    ax.text(8.5, 4.4, "Conv 3x3\nstride 1\n64 filters", ha="center",
            fontsize=9.5, color=C_DARK, fontweight="bold")
    ax.text(8.5, 1.45, "64 x 7 x 7", ha="center", fontsize=9, color=C_DARK)

    arrow(9.3, 9.95)

    # FC layer
    block(10.05, 1.2, 1.2, 3.6, C_PURPLE, alpha=0.85)
    ax.text(10.65, 5.0, "FC 512\nReLU", ha="center",
            fontsize=9.5, color=C_DARK, fontweight="bold")
    ax.text(10.65, 0.85, "512", ha="center", fontsize=9, color=C_DARK)

    arrow(11.35, 12.0)

    # Output: Q-values
    block(12.05, 1.6, 1.6, 2.8, C_GREEN, alpha=0.85)
    ax.text(12.85, 4.6, "Output\nQ(s, a)", ha="center",
            fontsize=10, color=C_DARK, fontweight="bold")
    # tiny bars for action q-values
    n_act = 6
    bar_x = np.linspace(12.2, 13.5, n_act)
    bar_h = np.array([0.6, 1.2, 0.9, 1.5, 0.8, 1.0])
    for bx, bh in zip(bar_x, bar_h):
        ax.add_patch(plt.Rectangle((bx - 0.08, 1.7), 0.16, bh,
                                   color="white", ec=C_DARK, lw=0.8))
    ax.text(12.85, 0.85, "n_actions", ha="center", fontsize=9, color=C_DARK)

    ax.set_title("DQN Architecture: Pixels in, Q-Values out",
                 fontsize=13.5, fontweight="bold", pad=8)
    plt.tight_layout()
    _save(fig, "fig3_dqn_architecture")


# ---------------------------------------------------------------------------
# Figure 4: Experience replay buffer
# ---------------------------------------------------------------------------
def fig4_replay_buffer() -> None:
    """Stream of transitions enters a circular buffer; mini-batch sampled
    at random for the gradient update."""
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Environment box (left)
    ax.add_patch(FancyBboxPatch(
        (0.2, 4.4), 2.2, 1.4, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=C_AMBER, alpha=0.85, edgecolor=C_DARK, lw=1.4,
    ))
    ax.text(1.3, 5.1, "Environment\n(agent steps)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_DARK)

    # Stream label
    ax.annotate(
        "", xy=(4.3, 5.1), xytext=(2.45, 5.1),
        arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2.0, mutation_scale=18),
    )
    ax.text(3.4, 5.45, "(s, a, r, s', done)", ha="center", fontsize=9.5,
            color=C_DARK, style="italic")

    # Buffer (10 cells)
    n = 10
    buf_x0, buf_y0, cell_w, cell_h = 4.3, 4.3, 0.78, 1.6
    contents = ["t-9", "t-8", "t-7", "t-6", "t-5", "t-4", "t-3", "t-2", "t-1", "t"]
    sampled = {1, 3, 6, 8}  # indices to highlight as the sampled mini-batch
    for i in range(n):
        x = buf_x0 + i * cell_w
        color = C_BLUE if i in sampled else "white"
        edge = C_BLUE if i in sampled else C_GRAY
        lw = 1.8 if i in sampled else 1.0
        ax.add_patch(plt.Rectangle((x, buf_y0), cell_w * 0.92, cell_h,
                                   facecolor=color, edgecolor=edge, lw=lw,
                                   alpha=0.85 if i in sampled else 1.0))
        ax.text(x + cell_w * 0.46, buf_y0 + cell_h / 2, contents[i],
                ha="center", va="center", fontsize=9,
                color="white" if i in sampled else C_DARK,
                fontweight="bold" if i in sampled else "normal")

    ax.text(buf_x0 + n * cell_w / 2, buf_y0 + cell_h + 0.35,
            "Replay buffer  D  (FIFO, capacity ~ 1M)",
            ha="center", fontsize=11, fontweight="bold", color=C_DARK)

    # FIFO arrow indicating eviction at the left edge
    ax.annotate(
        "", xy=(3.7, 4.55), xytext=(4.25, 4.55),
        arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.5, mutation_scale=14),
    )
    ax.text(3.95, 4.15, "evict", ha="center", fontsize=8.5, color=C_GRAY,
            style="italic")

    # Mini-batch box (below)
    ax.add_patch(FancyBboxPatch(
        (4.6, 1.4), 5.6, 1.6, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="white", edgecolor=C_BLUE, lw=1.6,
    ))
    ax.text(7.4, 2.85, "Random mini-batch (size 32)",
            ha="center", fontsize=10.5, fontweight="bold", color=C_BLUE)
    # Show 4 mini-cells
    for j, idx in enumerate(sorted(sampled)):
        bx = 4.85 + j * 1.3
        ax.add_patch(plt.Rectangle((bx, 1.7), 1.0, 0.8,
                                   facecolor=C_BLUE, edgecolor=C_DARK, lw=1.0))
        ax.text(bx + 0.5, 2.1, contents[idx], ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    # Arrows from sampled cells down into mini-batch
    for j, idx in enumerate(sorted(sampled)):
        x_top = buf_x0 + idx * cell_w + cell_w * 0.46
        x_bot = 4.85 + j * 1.3 + 0.5
        ax.annotate(
            "", xy=(x_bot, 2.5), xytext=(x_top, 4.3),
            arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.2,
                            mutation_scale=10, alpha=0.6),
        )

    # Gradient update box (right)
    ax.add_patch(FancyBboxPatch(
        (10.6, 1.5), 2.2, 1.4, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=C_GREEN, alpha=0.85, edgecolor=C_DARK, lw=1.4,
    ))
    ax.text(11.7, 2.2, "Gradient\nupdate of\nQ(s, a; theta)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.annotate(
        "", xy=(10.55, 2.2), xytext=(10.25, 2.2),
        arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2.0, mutation_scale=16),
    )

    # Caption
    ax.text(6.5, 0.5,
            "Decorrelates time-adjacent samples and reuses each transition many times.",
            ha="center", fontsize=10, style="italic", color=C_DARK)

    ax.set_title("Experience Replay Buffer",
                 fontsize=13.5, fontweight="bold", pad=4)
    plt.tight_layout()
    _save(fig, "fig4_replay_buffer")


# ---------------------------------------------------------------------------
# Figure 5: Target network mechanism
# ---------------------------------------------------------------------------
def fig5_target_network() -> None:
    """Two trajectories: online theta drifts every step; target theta- jumps
    only every C steps (staircase). Demonstrates the lag-for-stability idea."""
    rng = np.random.default_rng(0)
    n_steps = 1000
    C = 100

    # Online trajectory: noisy random walk + drift
    online = np.cumsum(rng.normal(0.01, 0.08, size=n_steps)) + 0.5

    # Target: copy online every C steps
    target = np.empty_like(online)
    last_copy = online[0]
    for t in range(n_steps):
        if t % C == 0:
            last_copy = online[t]
        target[t] = last_copy

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(online, color=C_BLUE, lw=1.6, label=r"Online network  $\theta$  (updated every step)")
    ax.plot(target, color=C_PURPLE, lw=2.4, label=r"Target network  $\theta^-$  (copied every C steps)")

    # Mark copy events
    copy_steps = np.arange(0, n_steps, C)
    ax.scatter(copy_steps, online[copy_steps], color=C_PURPLE, s=42,
               zorder=5, edgecolor="white", lw=1.2)
    for cs in copy_steps[1:6]:
        ax.annotate("copy", xy=(cs, online[cs]),
                    xytext=(cs, online[cs] + 0.45),
                    ha="center", fontsize=8, color=C_PURPLE,
                    arrowprops=dict(arrowstyle="-", color=C_PURPLE, lw=0.8, alpha=0.6))

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Representative network parameter", fontsize=11)
    ax.set_title("Target Network: A Lagging Copy Stabilises the TD Target",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig5_target_network")


# ---------------------------------------------------------------------------
# Figure 6: Double DQN vs vanilla DQN
# ---------------------------------------------------------------------------
def fig6_double_vs_vanilla() -> None:
    """Mean Q-value estimate over training: vanilla DQN inflates above the
    true value; Double DQN tracks the true value. Inspired by Fig.2 of
    van Hasselt et al. 2016."""
    rng = np.random.default_rng(1)
    steps = np.linspace(0, 50, 250)

    true_value = 6.0 - 1.5 * np.exp(-steps / 18)
    vanilla = true_value + 2.5 * (1 - np.exp(-steps / 8)) + rng.normal(0, 0.18, len(steps))
    double = true_value + 0.4 * (1 - np.exp(-steps / 12)) + rng.normal(0, 0.18, len(steps))

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(steps, true_value, color=C_DARK, lw=2.2, ls="--",
            label="True Q-value (Monte Carlo estimate)")
    ax.plot(steps, vanilla, color=C_AMBER, lw=2.2, label="Vanilla DQN  (overestimates)")
    ax.plot(steps, double, color=C_GREEN, lw=2.2, label="Double DQN   (corrected)")

    # Shade the bias gap at the end
    ax.fill_between(steps[-60:], true_value[-60:], vanilla[-60:],
                    color=C_AMBER, alpha=0.18)
    ax.annotate("overestimation\nbias",
                xy=(45, (vanilla[-10] + true_value[-10]) / 2),
                xytext=(35, 9.0),
                fontsize=10, color=C_AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_AMBER, lw=1.4))

    ax.set_xlabel("Training step (millions of frames)", fontsize=11)
    ax.set_ylabel("Mean estimated Q-value", fontsize=11)
    ax.set_title("Double DQN Removes Overestimation Bias",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig6_double_vs_vanilla")


# ---------------------------------------------------------------------------
# Figure 7: Atari benchmark scores
# ---------------------------------------------------------------------------
def fig7_atari_benchmark() -> None:
    """Median human-normalised score across 57 Atari games for five DQN
    variants. Numbers are the standard reported figures from the Rainbow
    paper (Hessel et al., 2018) and follow-ups."""
    methods = ["DQN", "Double DQN", "Dueling DQN", "Prioritized\nDQN", "Rainbow"]
    scores = [79, 117, 151, 172, 223]    # median human-normalised %, approximate
    colors = [C_GRAY, C_BLUE, C_PURPLE, C_AMBER, C_GREEN]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(methods, scores, color=colors, edgecolor=C_DARK, lw=1.0,
                  width=0.62, alpha=0.92)
    ax.axhline(100, color=C_DARK, ls="--", lw=1.2, alpha=0.7)
    ax.text(4.5, 105, "Human level (100%)", ha="right",
            fontsize=9.5, color=C_DARK)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 6,
                f"{s}%", ha="center", fontsize=11, fontweight="bold",
                color=C_DARK)

    ax.set_ylabel("Median human-normalised score", fontsize=11)
    ax.set_ylim(0, max(scores) * 1.18)
    ax.set_title("Atari-57 Benchmark: Each DQN Improvement Stacks Gains",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, "fig7_atari_benchmark")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for RL Part 02: Q-Learning and DQN")
    fig1_qtable_gridworld()
    fig2_cliff_walking_curves()
    fig3_dqn_architecture()
    fig4_replay_buffer()
    fig5_target_network()
    fig6_double_vs_vanilla()
    fig7_atari_benchmark()
    print("Done.")


if __name__ == "__main__":
    main()
