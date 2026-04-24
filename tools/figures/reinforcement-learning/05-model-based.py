"""
Figure generation script for Reinforcement Learning Part 05:
Model-Based RL and World Models.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly.

Figures:
    fig1_mf_vs_mb_loops        Two side-by-side control loops contrasting
                               model-free (env -> policy) and model-based
                               (env -> model -> policy via planning).
    fig2_world_model_vmc       The V/M/C architecture from Ha & Schmidhuber:
                               vision encoder, memory RNN, controller.
    fig3_dyna_q_flow           Dyna-Q: real experience updates Q AND a tabular
                               model; planning replays imagined transitions.
    fig4_mbpo_short_rollouts   MBPO branched short-horizon rollouts: model
                               error grows with rollout length k, MBPO keeps
                               k small (1-5).
    fig5_mpc_planning          Model Predictive Control: shoot N candidate
                               action sequences in the model, pick the best
                               first action, repeat.
    fig6_sample_efficiency     Sample-efficiency curves: model-based methods
                               (MBPO, Dreamer) reach expert performance with
                               ~10x fewer interactions than SAC/PPO.
    fig7_dreamer_latent        Dreamer-style RSSM: deterministic h_t plus
                               stochastic z_t evolved in latent space, with
                               heads predicting reward, value, observation.

Usage:
    python3 scripts/figures/reinforcement-learning/05-model-based.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
C_BLUE = COLORS["primary"]     # primary -- real experience / model-free
C_PURPLE = COLORS["accent"]   # secondary -- model / latent
C_GREEN = COLORS["success"]    # accent -- good outcomes / model-based wins
C_AMBER = COLORS["warning"]    # warning / planning / highlight
C_GRAY = COLORS["muted"]
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
    / "05-model-based-rl-and-world-models"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "reinforcement-learning"
    / "05-Model-Based强化学习与世界模型"
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


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=10, fontweight="bold",
         text_color="white", radius=0.08):
    """Draw a rounded box with centred text."""
    if ec is None:
        ec = fc
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.4, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color)


def _arrow(ax, p1, p2, color=C_DARK, lw=1.6, style="-|>", mutation=18,
           connectionstyle="arc3,rad=0", linestyle="-"):
    arr = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=mutation,
        linewidth=lw, color=color, connectionstyle=connectionstyle,
        linestyle=linestyle,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: Model-Free vs Model-Based control loops
# ---------------------------------------------------------------------------
def fig1_mf_vs_mb_loops() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    # ---- Model-Free ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Model-Free RL", fontsize=13, fontweight="bold",
                 color=C_BLUE, pad=12)
    ax.text(5, 6.4, "Learn directly from real experience",
            ha="center", fontsize=10, color=C_DARK, style="italic")

    _box(ax, (0.5, 3.2), 3.2, 1.4, "Environment", C_DARK, fontsize=11)
    _box(ax, (6.3, 3.2), 3.2, 1.4, "Policy / Q-function", C_BLUE, fontsize=11)

    _arrow(ax, (3.7, 4.4), (6.3, 4.4), color=C_DARK, lw=1.8)
    ax.text(5.0, 4.7, "(s, a, r, s')", ha="center", fontsize=9.5,
            color=C_DARK, fontweight="bold")

    _arrow(ax, (6.3, 3.6), (3.7, 3.6),
           color=C_BLUE, lw=1.8, connectionstyle="arc3,rad=0")
    ax.text(5.0, 3.25, "action a", ha="center", fontsize=9.5,
            color=C_BLUE, fontweight="bold")

    ax.text(5, 1.6, "Every gradient step costs a real interaction.\n"
                    "Atari Pong: ~10M frames. Dota 2: ~45,000 years.",
            ha="center", fontsize=9.5, color=C_GRAY)

    # ---- Model-Based ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Model-Based RL", fontsize=13, fontweight="bold",
                 color=C_GREEN, pad=12)
    ax.text(5, 6.4, "Learn a model, then plan inside it",
            ha="center", fontsize=10, color=C_DARK, style="italic")

    _box(ax, (0.3, 4.6), 2.6, 1.2, "Environment", C_DARK, fontsize=10.5)
    _box(ax, (3.7, 4.6), 2.6, 1.2, "Learned model\n$\\hat{P}(s'|s,a)$",
         C_PURPLE, fontsize=10)
    _box(ax, (7.1, 4.6), 2.6, 1.2, "Policy / Q", C_BLUE, fontsize=10.5)

    _arrow(ax, (2.9, 5.2), (3.7, 5.2), color=C_DARK)
    _arrow(ax, (6.3, 5.2), (7.1, 5.2), color=C_PURPLE)

    # planning loop -- arrow from policy back through model
    _box(ax, (3.7, 2.4), 2.6, 1.2,
         "Planning /\nimagined rollouts", C_AMBER, fontsize=10)
    _arrow(ax, (5.0, 4.6), (5.0, 3.6), color=C_PURPLE, lw=1.6)
    _arrow(ax, (6.3, 3.0), (8.4, 4.6), color=C_AMBER, lw=1.6,
           connectionstyle="arc3,rad=-0.25")
    _arrow(ax, (7.1, 4.6), (5.0, 3.6), color=C_BLUE, lw=1.4,
           linestyle="--", connectionstyle="arc3,rad=0.25")

    ax.text(5, 1.4, "10-100x fewer real interactions.\n"
                    "Caveat: model errors compound.",
            ha="center", fontsize=9.5, color=C_GRAY)

    fig.suptitle("Two paradigms for control",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_mf_vs_mb_loops")


# ---------------------------------------------------------------------------
# Figure 2: World Model V/M/C architecture
# ---------------------------------------------------------------------------
def fig2_world_model_vmc() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.set_title("World Models (Ha & Schmidhuber, 2018):  V -> M -> C",
                 fontsize=13.5, fontweight="bold", pad=12)

    # Pixel observation (left)
    obs_x, obs_y = 0.4, 2.6
    obs = Rectangle((obs_x, obs_y), 1.6, 1.6, facecolor=C_GRAY,
                    edgecolor=C_DARK, linewidth=1.2)
    ax.add_patch(obs)
    # mock pixel grid
    rng = np.random.default_rng(3)
    for i in range(6):
        for j in range(6):
            v = rng.uniform(0.3, 0.95)
            ax.add_patch(Rectangle(
                (obs_x + 0.1 + j * 0.23, obs_y + 0.1 + i * 0.23),
                0.22, 0.22, facecolor=str(v), edgecolor="none"))
    ax.text(obs_x + 0.8, obs_y - 0.35, "observation $o_t$",
            ha="center", fontsize=10, color=C_DARK)

    # V (Vision -- VAE)
    _box(ax, (2.6, 2.7), 1.9, 1.4, "V\nVision (VAE)", C_BLUE, fontsize=11)
    _arrow(ax, (2.0, 3.4), (2.6, 3.4), color=C_DARK)

    # latent z_t
    _box(ax, (5.1, 3.0), 1.0, 0.85, "$z_t$", C_PURPLE, fontsize=11.5,
         radius=0.18)
    _arrow(ax, (4.5, 3.4), (5.1, 3.4), color=C_BLUE)

    # M (Memory -- RNN)
    _box(ax, (6.7, 2.7), 1.9, 1.4, "M\nMemory (RNN)", C_PURPLE, fontsize=11)
    _arrow(ax, (6.1, 3.4), (6.7, 3.4), color=C_PURPLE)

    # hidden state h_t
    _box(ax, (9.2, 3.0), 1.0, 0.85, "$h_t$", C_PURPLE, fontsize=11.5,
         radius=0.18)
    _arrow(ax, (8.6, 3.4), (9.2, 3.4), color=C_PURPLE)

    # C (Controller -- linear policy)
    _box(ax, (10.8, 2.7), 1.9, 1.4, "C\nController", C_GREEN, fontsize=11)
    _arrow(ax, (10.2, 3.4), (10.8, 3.4), color=C_PURPLE)

    # action arrow back
    _arrow(ax, (11.75, 2.7), (1.2, 2.6), color=C_AMBER, lw=1.6,
           connectionstyle="arc3,rad=0.32")
    ax.text(6.5, 1.0, "action $a_t$  (closes the loop with the environment)",
            ha="center", fontsize=10.5, color=C_AMBER, fontweight="bold")

    # parameter counts annotation
    ax.text(3.55, 5.3, "~10M params\n(reconstruction)",
            ha="center", fontsize=9, color=C_BLUE)
    ax.text(7.65, 5.3, "~1M params\n(predict $z_{t+1}$)",
            ha="center", fontsize=9, color=C_PURPLE)
    ax.text(11.75, 5.3, "~867 params\n(linear $a_t$)",
            ha="center", fontsize=9, color=C_GREEN)

    # dream training note
    ax.text(6.5, 0.25,
            "Trained in 'dreams': roll out M from $z_t$, "
            "evolve C without touching the real environment.",
            ha="center", fontsize=10, color=C_DARK, style="italic")

    fig.tight_layout()
    _save(fig, "fig2_world_model_vmc")


# ---------------------------------------------------------------------------
# Figure 3: Dyna-Q flow (real + simulated experience)
# ---------------------------------------------------------------------------
def fig3_dyna_q_flow() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- Left: data flow diagram ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Dyna-Q: real + imagined updates",
                 fontsize=12.5, fontweight="bold", pad=10)

    _box(ax, (0.3, 5.3), 2.6, 1.1, "Environment", C_DARK, fontsize=10.5)
    _box(ax, (3.7, 5.3), 2.6, 1.1, "Real experience\n(s, a, r, s')",
         C_BLUE, fontsize=9.5)
    _box(ax, (7.1, 5.3), 2.6, 1.1, "Q-table", C_GREEN, fontsize=10.5)
    _arrow(ax, (2.9, 5.85), (3.7, 5.85), color=C_DARK)
    _arrow(ax, (6.3, 5.85), (7.1, 5.85), color=C_BLUE)
    ax.text(5.0, 6.55, "1. direct learning",
            ha="center", fontsize=9, color=C_BLUE, fontweight="bold")

    _box(ax, (3.7, 3.4), 2.6, 1.1, "Tabular model\n$M(s,a)\\to(r,s')$",
         C_PURPLE, fontsize=9.5)
    _arrow(ax, (5.0, 5.3), (5.0, 4.5), color=C_BLUE, lw=1.5)
    ax.text(5.6, 4.95, "2. model learning", ha="left", fontsize=9,
            color=C_PURPLE, fontweight="bold")

    _box(ax, (3.7, 1.4), 2.6, 1.1, "Imagined\n(s, a, r, s')",
         C_AMBER, fontsize=9.5)
    _arrow(ax, (5.0, 3.4), (5.0, 2.5), color=C_PURPLE, lw=1.5)
    _arrow(ax, (6.3, 1.95), (8.4, 5.3), color=C_AMBER, lw=1.5,
           connectionstyle="arc3,rad=-0.32")
    ax.text(5.6, 2.95, "3. planning\n($n$ replays)", ha="left", fontsize=9,
            color=C_AMBER, fontweight="bold")

    ax.text(5, 0.4,
            "Each real step yields  1 + n  updates to Q.",
            ha="center", fontsize=10, color=C_DARK, style="italic")

    # ---- Right: convergence curves ----
    ax = axes[1]
    ax.set_title("GridWorld: planning steps speed convergence",
                 fontsize=12.5, fontweight="bold", pad=10)

    rng = np.random.default_rng(11)
    episodes = np.arange(1, 81)

    def curve(steps_per_episode, jitter=0.04):
        # asymptote ~14 steps optimal; convergence rate ~ 1/(1+n)
        target = 14.0
        rate = 0.06 + 0.18 * np.log1p(steps_per_episode)
        baseline = 220.0
        y = target + (baseline - target) * np.exp(-rate * episodes)
        y *= 1 + rng.normal(0, jitter, size=episodes.shape)
        return np.clip(y, target - 1, baseline + 20)

    for n, color, label in [
        (0, C_BLUE, "0 planning steps (Q-Learning)"),
        (5, C_PURPLE, "5 planning steps"),
        (50, C_GREEN, "50 planning steps"),
    ]:
        ax.plot(episodes, curve(n), color=color, lw=2.0, label=label)

    ax.set_xlabel("episode")
    ax.set_ylabel("steps per episode (lower = better)")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig3_dyna_q_flow")


# ---------------------------------------------------------------------------
# Figure 4: MBPO short-horizon branched rollouts
# ---------------------------------------------------------------------------
def fig4_mbpo_short_rollouts() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- Left: branched rollouts diagram ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("MBPO: short branches off the real trajectory",
                 fontsize=12.5, fontweight="bold", pad=8)

    # real trajectory backbone
    real_xs = np.linspace(0.6, 9.4, 7)
    real_y = 1.4
    for x in real_xs:
        ax.scatter(x, real_y, s=110, color=C_BLUE, zorder=3,
                   edgecolor=C_DARK, linewidth=1)
    for i in range(len(real_xs) - 1):
        _arrow(ax, (real_xs[i] + 0.12, real_y),
               (real_xs[i + 1] - 0.12, real_y),
               color=C_BLUE, lw=1.6)
    ax.text(5.0, 0.65, "real trajectory in $\\mathcal{D}_{real}$",
            ha="center", fontsize=10, color=C_BLUE, fontweight="bold")

    # short branched rollouts (k=1..3 steps)
    rng = np.random.default_rng(5)
    for x_start in real_xs[1:-1]:
        prev = (x_start, real_y)
        for step in range(3):
            dx = 0.55 + rng.uniform(-0.05, 0.08)
            dy = 0.55 + 0.15 * step + rng.uniform(-0.05, 0.05)
            nxt = (prev[0] + dx, prev[1] + dy)
            _arrow(ax, prev, nxt, color=C_AMBER, lw=1.2)
            ax.scatter(*nxt, s=55, color=C_AMBER, edgecolor=C_DARK,
                       linewidth=0.6, zorder=3)
            prev = nxt

    ax.text(5.0, 5.6, "imagined branches  (k = 1..5 steps)",
            ha="center", fontsize=10.5, color=C_AMBER, fontweight="bold")
    ax.text(5.0, 5.05,
            "short enough that model error stays bounded",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    # ---- Right: model error vs rollout length ----
    ax = axes[1]
    ax.set_title("Model error grows with rollout length",
                 fontsize=12.5, fontweight="bold", pad=8)

    k = np.arange(0, 21)
    # compounding error model: roughly exponential
    err_single = 0.12 * (1.18 ** k - 1)
    err_ensemble = 0.06 * (1.10 ** k - 1)

    ax.plot(k, err_single, color=C_AMBER, lw=2.2,
            label="single dynamics model", marker="o", markersize=4)
    ax.plot(k, err_ensemble, color=C_GREEN, lw=2.2,
            label="ensemble of 5 (MBPO)", marker="s", markersize=4)

    ax.axvspan(1, 5, color=C_GREEN, alpha=0.10)
    ax.text(3.0, ax.get_ylim()[1] * 0.92 if False else 4.0,
            "MBPO\nsweet spot\n$k$ = 1..5",
            ha="center", fontsize=9.5, color=C_GREEN, fontweight="bold")

    ax.set_xlabel("rollout length $k$")
    ax.set_ylabel("cumulative state prediction error")
    ax.legend(loc="upper left", fontsize=9.5)

    fig.tight_layout()
    _save(fig, "fig4_mbpo_short_rollouts")


# ---------------------------------------------------------------------------
# Figure 5: MPC -- shoot, score, pick first action, repeat
# ---------------------------------------------------------------------------
def fig5_mpc_planning() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-3.4, 3.4)
    ax.set_title("Model Predictive Control:  sample N action sequences, "
                 "execute the best first action",
                 fontsize=12.5, fontweight="bold", pad=10)

    # current state
    ax.scatter(0, 0, s=180, color=C_BLUE, edgecolor=C_DARK,
               linewidth=1.4, zorder=5)
    ax.text(0, -0.55, "current\nstate $s_t$",
            ha="center", fontsize=10, color=C_BLUE, fontweight="bold")

    rng = np.random.default_rng(2)
    H = 8  # horizon
    N = 12  # candidate sequences
    xs = np.linspace(0, 9.5, H + 1)

    best_idx = None
    best_return = -np.inf
    sequences = []
    for n in range(N):
        ys = [0.0]
        # random walk biased differently per sequence
        bias = rng.normal(0, 0.12)
        for h in range(H):
            ys.append(ys[-1] + bias + rng.normal(0, 0.35))
        ys = np.array(ys)
        # pretend "return" = -sum(distance to target line y=2.0)
        ret = -np.sum(np.abs(ys - 2.0))
        sequences.append((ys, ret))
        if ret > best_return:
            best_return = ret
            best_idx = n

    for i, (ys, _ret) in enumerate(sequences):
        if i == best_idx:
            continue
        ax.plot(xs, ys, color=C_GRAY, lw=1.0, alpha=0.55)
        ax.scatter(xs[1:], ys[1:], s=18, color=C_GRAY, alpha=0.55)

    # best sequence
    best_ys = sequences[best_idx][0]
    ax.plot(xs, best_ys, color=C_GREEN, lw=2.6, label="best sequence",
            zorder=4)
    ax.scatter(xs[1:], best_ys[1:], s=42, color=C_GREEN,
               edgecolor=C_DARK, linewidth=0.8, zorder=4)

    # highlight first action of best sequence
    ax.add_patch(FancyArrowPatch(
        (xs[0], best_ys[0]), (xs[1], best_ys[1]),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=3.0, color=C_AMBER, zorder=6,
    ))
    ax.scatter(xs[1], best_ys[1], s=140, color=C_AMBER,
               edgecolor=C_DARK, linewidth=1.4, zorder=6)
    ax.annotate("execute only $a_t$,\nthen replan",
                (xs[1], best_ys[1]),
                xytext=(xs[1] + 0.3, best_ys[1] + 1.2),
                fontsize=10, color=C_AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.2))

    # target line
    ax.axhline(2.0, color=C_PURPLE, linestyle="--", alpha=0.5, lw=1.2)
    ax.text(10.7, 2.0, "goal", color=C_PURPLE, fontsize=10,
            va="center", fontweight="bold")

    ax.set_xlabel("planning horizon (steps into the model)")
    ax.set_ylabel("predicted state")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xticks(np.arange(0, H + 1))

    fig.tight_layout()
    _save(fig, "fig5_mpc_planning")


# ---------------------------------------------------------------------------
# Figure 6: Sample efficiency comparison (model-based wins)
# ---------------------------------------------------------------------------
def fig6_sample_efficiency() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # ---- Left: learning curves ----
    ax = axes[0]
    ax.set_title("MuJoCo HalfCheetah:  return vs. environment steps",
                 fontsize=12.5, fontweight="bold", pad=8)

    steps = np.logspace(3, 6.2, 200)  # 1e3 -> ~1.6e6

    def curve(midpoint, asymptote=11000, slope=2.6, noise=180,
              seed=0):
        rng = np.random.default_rng(seed)
        log_s = np.log10(steps)
        log_mid = np.log10(midpoint)
        y = asymptote / (1 + np.exp(-slope * (log_s - log_mid)))
        y = y + rng.normal(0, noise, size=y.shape)
        return np.clip(y, 0, asymptote * 1.05)

    sac = curve(midpoint=4e5, asymptote=11000, slope=2.4, seed=1)
    ppo = curve(midpoint=8e5, asymptote=8500, slope=2.0, seed=2)
    mbpo = curve(midpoint=4e4, asymptote=11200, slope=2.0, seed=3)
    dreamer = curve(midpoint=6e4, asymptote=10500, slope=1.8, seed=4)

    ax.plot(steps, ppo, color=C_GRAY, lw=2.0, label="PPO (model-free)")
    ax.plot(steps, sac, color=C_BLUE, lw=2.2, label="SAC (model-free)")
    ax.plot(steps, dreamer, color=C_PURPLE, lw=2.2,
            label="Dreamer (model-based)")
    ax.plot(steps, mbpo, color=C_GREEN, lw=2.4, label="MBPO (model-based)")

    ax.set_xscale("log")
    ax.set_xlabel("environment steps (log scale)")
    ax.set_ylabel("episode return")
    ax.legend(loc="lower right", fontsize=9.5)

    # gap annotation
    ax.axvline(1e5, color=C_AMBER, linestyle=":", lw=1.4, alpha=0.7)
    ax.text(1.05e5, 500, "100K steps", rotation=90, fontsize=9,
            color=C_AMBER, va="bottom")

    # ---- Right: bar chart of "steps to reach 9000 return" ----
    ax = axes[1]
    ax.set_title("Steps to reach 9000 return\n(lower is better)",
                 fontsize=12.5, fontweight="bold", pad=8)

    methods = ["PPO", "SAC", "Dreamer", "MBPO"]
    steps_needed = [1.6e6, 6e5, 1.5e5, 8e4]
    colors = [C_GRAY, C_BLUE, C_PURPLE, C_GREEN]

    bars = ax.barh(methods, steps_needed, color=colors,
                   edgecolor=C_DARK, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("environment steps (log)")

    for bar, s in zip(bars, steps_needed):
        if s >= 1e6:
            label = f"{s / 1e6:.1f}M"
        else:
            label = f"{int(s / 1e3)}K"
        ax.text(s * 1.12, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10,
                color=C_DARK, fontweight="bold")

    ax.set_xlim(2e4, 5e6)

    fig.tight_layout()
    _save(fig, "fig6_sample_efficiency")


# ---------------------------------------------------------------------------
# Figure 7: Dreamer-style RSSM latent dynamics
# ---------------------------------------------------------------------------
def fig7_dreamer_latent() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Dreamer (RSSM):  recurrent state + stochastic latent, "
                 "rolled out in imagination",
                 fontsize=13, fontweight="bold", pad=10)

    # Three time slices: t-1, t, t+1
    slices_x = [1.6, 6.0, 10.4]
    slice_labels = ["$t-1$", "$t$", "$t+1$"]

    for sx, lbl in zip(slices_x, slice_labels):
        ax.text(sx, 6.55, lbl, ha="center", fontsize=12,
                fontweight="bold", color=C_DARK)

    # h_t (deterministic) row at y = 5.0
    h_y = 4.7
    z_y = 3.2
    a_y = 1.6

    for sx in slices_x:
        _box(ax, (sx - 0.55, h_y), 1.1, 0.75, "$h$", C_PURPLE,
             fontsize=11, radius=0.18)
        _box(ax, (sx - 0.55, z_y), 1.1, 0.75, "$z$", C_BLUE,
             fontsize=11, radius=0.18)

    # h transitions (deterministic GRU)
    for i in range(len(slices_x) - 1):
        _arrow(ax, (slices_x[i] + 0.55, h_y + 0.37),
               (slices_x[i + 1] - 0.55, h_y + 0.37),
               color=C_PURPLE, lw=2.0)

    # z depends on h (stochastic prior / posterior)
    for sx in slices_x:
        _arrow(ax, (sx, h_y), (sx, z_y + 0.75), color=C_PURPLE,
               lw=1.4, style="-|>")

    # action feeding into next h
    for i in range(len(slices_x) - 1):
        sx = slices_x[i]
        nxt = slices_x[i + 1]
        ax.scatter(sx, a_y, s=160, color=C_AMBER, edgecolor=C_DARK,
                   linewidth=1.0, zorder=4)
        ax.text(sx, a_y - 0.4, f"$a_{{t{['-1', '', '+1'][i]}}}$",
                ha="center", fontsize=10, color=C_AMBER, fontweight="bold")
        _arrow(ax, (sx + 0.1, a_y + 0.15),
               (nxt - 0.55, h_y + 0.1),
               color=C_AMBER, lw=1.4,
               connectionstyle="arc3,rad=-0.18")

    # last action position
    ax.scatter(slices_x[-1], a_y, s=160, color=C_AMBER,
               edgecolor=C_DARK, linewidth=1.0, zorder=4)
    ax.text(slices_x[-1], a_y - 0.4, "$a_{t+1}$",
            ha="center", fontsize=10, color=C_AMBER, fontweight="bold")

    # Heads on the right side: reward, value, observation
    head_x = 12.2
    for y, label, color in [
        (h_y + 0.4, "reward $\\hat{r}$", C_GREEN),
        (z_y + 0.4, "obs $\\hat{o}$", C_BLUE),
    ]:
        _box(ax, (head_x - 0.6, y - 0.35), 1.2, 0.7, label, color,
             fontsize=9.5, radius=0.18)

    _arrow(ax, (slices_x[-1] + 0.55, h_y + 0.55),
           (head_x - 0.6, h_y + 0.55), color=C_GREEN, lw=1.4)
    _arrow(ax, (slices_x[-1] + 0.55, z_y + 0.55),
           (head_x - 0.6, z_y + 0.55), color=C_BLUE, lw=1.4)

    # legend / annotations
    ax.text(0.2, 5.05,
            "deterministic\nrecurrent\nstate", fontsize=9.5,
            color=C_PURPLE, fontweight="bold", va="center", ha="left")
    ax.text(0.2, 3.55,
            "stochastic\nlatent\nvariable", fontsize=9.5,
            color=C_BLUE, fontweight="bold", va="center", ha="left")
    ax.text(0.2, 1.6,
            "actions\nfrom policy", fontsize=9.5,
            color=C_AMBER, fontweight="bold", va="center", ha="left")

    ax.text(6.5, 0.45,
            "Behaviour learning rolls out 15 latent steps from the current "
            "$(h_t, z_t)$ -- no real environment calls.",
            ha="center", fontsize=10, color=C_DARK, style="italic")

    fig.tight_layout()
    _save(fig, "fig7_dreamer_latent")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Reinforcement Learning Part 05 ...")
    fig1_mf_vs_mb_loops()
    fig2_world_model_vmc()
    fig3_dyna_q_flow()
    fig4_mbpo_short_rollouts()
    fig5_mpc_planning()
    fig6_sample_efficiency()
    fig7_dreamer_latent()
    print("Done.")


if __name__ == "__main__":
    main()
