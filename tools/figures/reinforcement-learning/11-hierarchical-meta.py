"""Figures for Reinforcement Learning Part 11 — Hierarchical RL and Meta-RL.

Generates 7 publication-quality figures explaining temporal abstraction
(Options, MAXQ, Feudal) and meta-learning (MAML, RL^2).

Figures:
    fig1_options_framework      — Options as semi-MDP, high-level / low-level
    fig2_feudal_architecture    — Manager-Worker (FuN / HIRO) information flow
    fig3_goal_conditioned       — Goal-conditioned policy and intrinsic reward
    fig4_maml_inner_outer_loop  — MAML two-loop optimisation in parameter space
    fig5_meta_rl_distribution   — Train/test task distribution & adaptation
    fig6_task_decomposition     — Hierarchical task decomposition tree
    fig7_rl_squared             — RL^2 recurrent meta-learner (RNN unroll)

Output is written to BOTH the EN and ZH asset folders.

References (verified):
    - Sutton, Precup, Singh, Between MDPs and semi-MDPs: A framework for
      temporal abstraction in reinforcement learning, AIJ 1999.
    - Dietterich, Hierarchical Reinforcement Learning with the MAXQ Value
      Function Decomposition, JAIR 2000.
    - Vezhnevets et al., FeUdal Networks for Hierarchical RL, ICML 2017
      [1703.01161].
    - Nachum et al., Data-Efficient Hierarchical RL (HIRO), NeurIPS 2018
      [1805.08296].
    - Finn, Abbeel, Levine, Model-Agnostic Meta-Learning for Fast Adaptation
      of Deep Networks (MAML), ICML 2017 [1703.03400].
    - Duan et al., RL^2: Fast Reinforcement Learning via Slow Reinforcement
      Learning, 2016 [1611.02779].
    - Wang et al., Learning to reinforcement learn, 2016 [1611.05763].
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
# Shared aesthetic style (chenk-site)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
ORANGE = COLORS["warning"]
GRAY = COLORS["text2"]
LIGHT = COLORS["grid"]
DARK = COLORS["text"]
RED = COLORS["danger"]

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/reinforcement-learning/11-hierarchical-and-meta-rl"
ZH_DIR = REPO / "source/_posts/zh/reinforcement-learning/11-层次化强化学习与元学习"


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _box(ax, xy, w, h, label, fc, ec=None, fontsize=10, fontweight="bold",
         text_color="white"):
    if ec is None:
        ec = fc
    ax.add_patch(
        FancyBboxPatch(
            xy, w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.4, facecolor=fc, edgecolor=ec,
        )
    )
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color)


def _arrow(ax, src, dst, color=DARK, lw=1.6, style="->", curve=0.0):
    cs = f"arc3,rad={curve}"
    ax.add_patch(
        FancyArrowPatch(
            src, dst,
            arrowstyle=style, color=color, linewidth=lw,
            mutation_scale=14, connectionstyle=cs,
        )
    )


# ---------------------------------------------------------------------------
# Fig 1: Options framework as semi-MDP
# ---------------------------------------------------------------------------
def fig1_options_framework() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))

    # ----- Left: timeline view -----
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Temporal abstraction: primitive actions vs options",
                 loc="left")

    # primitive timeline
    ax.text(0.1, 5.2, "Flat MDP", fontsize=11, fontweight="bold", color=DARK)
    for i in range(12):
        x = 0.4 + i * 0.9
        ax.add_patch(Rectangle((x, 4.2), 0.7, 0.65, facecolor=LIGHT,
                               edgecolor=GRAY, lw=1))
        ax.text(x + 0.35, 4.52, f"a", ha="center", va="center",
                fontsize=8, color=DARK)
    ax.annotate("", xy=(11.4, 3.95), xytext=(0.4, 3.95),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2))
    ax.text(0.4, 3.6, "decisions per timestep", fontsize=8, color=GRAY)

    # option timeline
    ax.text(0.1, 2.7, "Semi-MDP", fontsize=11, fontweight="bold", color=DARK)
    spans = [(0.4, 2.5, BLUE, "$o_1$"),
             (2.9, 3.7, PURPLE, "$o_2$"),
             (6.6, 1.6, GREEN, "$o_3$"),
             (8.2, 3.2, ORANGE, "$o_4$")]
    for x, w, c, lbl in spans:
        ax.add_patch(FancyBboxPatch((x, 1.7), w, 0.7,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     facecolor=c, edgecolor=c, alpha=0.85))
        ax.text(x + w / 2, 2.05, lbl, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(11.4, 1.45), xytext=(0.4, 1.45),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2))
    ax.text(0.4, 1.05, "decisions only at termination $\\beta(s)=1$",
            fontsize=8, color=GRAY)

    ax.text(6.0, 0.45,
            "Options shrink the effective horizon $T \\rightarrow T/\\bar k$",
            ha="center", fontsize=9.5, style="italic", color=DARK)

    # ----- Right: option triple I, pi, beta -----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title(r"An option $o = \langle \mathcal{I}, \pi_o, \beta \rangle$",
                 loc="left")

    _box(ax, (0.5, 4.2), 2.6, 1.2, r"Initiation $\mathcal{I}$", BLUE)
    ax.text(1.8, 3.9, "where can it start?", ha="center", fontsize=8.5,
            color=GRAY)

    _box(ax, (3.7, 4.2), 2.6, 1.2, r"Policy $\pi_o(a|s)$", PURPLE)
    ax.text(5.0, 3.9, "what to do inside", ha="center", fontsize=8.5,
            color=GRAY)

    _box(ax, (6.9, 4.2), 2.6, 1.2, r"Termination $\beta(s)$", GREEN)
    ax.text(8.2, 3.9, "when to stop", ha="center", fontsize=8.5, color=GRAY)

    # high level / low level
    _box(ax, (0.5, 1.3), 9.0, 1.4,
         r"High-level policy  $\mu(o|s)$  —  selects option from $\mathcal{O}$",
         BLUE, fontsize=10.5)
    _box(ax, (0.5, 0.0), 9.0, 1.0,
         r"Low-level execution: roll out $\pi_o$ until $\beta(s)=1$",
         PURPLE, fontsize=10)

    # arrows
    for x in (1.8, 5.0, 8.2):
        _arrow(ax, (x, 4.2), (x, 2.7), color=GRAY, lw=1.0)

    fig.suptitle("Options Framework — Hierarchical Action Abstraction",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "fig1_options_framework.png")


# ---------------------------------------------------------------------------
# Fig 2: Feudal Manager-Worker (FuN / HIRO)
# ---------------------------------------------------------------------------
def fig2_feudal_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Feudal RL — Manager sets goals, Worker reaches them",
                 loc="left")

    # Manager block
    _box(ax, (0.5, 5.2), 12.0, 1.2,
         r"Manager  $g_t = M_\theta(s_t)$  —  acts every $c$ steps, "
         r"low temporal resolution", BLUE, fontsize=11)

    # Worker block
    _box(ax, (0.5, 0.6), 12.0, 1.1,
         r"Worker  $\pi_\phi(a_t \mid s_t, g_t)$  —  acts every step",
         PURPLE, fontsize=11)

    # Timeline of states / sub-goals
    times = np.linspace(1.2, 11.8, 9)
    for i, x in enumerate(times):
        # state circles
        ax.add_patch(Circle((x, 3.4), 0.32, facecolor=LIGHT,
                            edgecolor=GRAY, lw=1.2))
        ax.text(x, 3.4, f"$s_{{{i}}}$", ha="center", va="center", fontsize=9)

    # Manager → goal arrows (every c=4 steps)
    for k, idx in enumerate([0, 4, 8]):
        x = times[idx]
        _arrow(ax, (x, 5.2), (x, 3.75), color=BLUE, lw=1.8)
        ax.text(x + 0.05, 4.55, f"$g_{{{k}}}$", color=BLUE, fontsize=10,
                fontweight="bold")

    # Worker → action arrows
    for i, x in enumerate(times):
        _arrow(ax, (x, 1.7), (x, 3.05), color=PURPLE, lw=1.0)

    # Intrinsic reward arrow
    _arrow(ax, (12.55, 3.4), (12.55, 1.7),
           color=ORANGE, lw=1.6, curve=0.3)
    ax.text(12.7, 2.55, r"$r^{int}_t$" "\n" r"$=\cos(s_{t+c}-s_t,\, g_t)$",
            fontsize=9, color=ORANGE, va="center")

    # Extrinsic reward feedback to manager
    _arrow(ax, (0.5, 3.4), (0.5, 5.2), color=GREEN, lw=1.6, curve=-0.3)
    ax.text(0.05, 4.4, r"$r^{ext}_t$", color=GREEN, fontsize=10,
            fontweight="bold", va="center")

    # Labels
    ax.text(6.5, 6.7, "low-frequency goal-setting",
            ha="center", color=BLUE, fontsize=9.5, style="italic")
    ax.text(6.5, 0.18, "high-frequency motor control",
            ha="center", color=PURPLE, fontsize=9.5, style="italic")

    # HIRO note
    ax.text(6.5, 2.45,
            "HIRO: relabel $g_t \\leftarrow \\arg\\max_{g} \\log \\pi_\\phi(a_{t:t+c}|s_{t:t+c}, g)$"
            "  — keeps Worker training data on-policy",
            ha="center", fontsize=8.8, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7",
                      edgecolor=ORANGE, lw=0.8))

    fig.tight_layout()
    save(fig, "fig2_feudal_architecture.png")


# ---------------------------------------------------------------------------
# Fig 3: Goal-conditioned RL
# ---------------------------------------------------------------------------
def fig3_goal_conditioned() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ----- Left: pi(a|s,g) policy schematic -----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title(r"Goal-conditioned policy $\pi(a \mid s, g)$", loc="left")

    _box(ax, (0.4, 4.6), 2.4, 1.0, r"State $s$", BLUE, fontsize=11)
    _box(ax, (0.4, 3.0), 2.4, 1.0, r"Goal $g$", GREEN, fontsize=11)

    # network as 3 stacked layers
    for i, y in enumerate([4.4, 3.4, 2.4]):
        ax.add_patch(FancyBboxPatch((4.0, y), 2.2, 0.8,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     facecolor=PURPLE, edgecolor=PURPLE, alpha=0.85))
        ax.text(5.1, y + 0.4, f"FC + ReLU", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

    _box(ax, (7.4, 3.5), 2.2, 1.0, r"Action $a$", ORANGE, fontsize=11)

    _arrow(ax, (2.8, 5.1), (4.0, 5.1), color=BLUE)
    _arrow(ax, (2.8, 3.5), (4.0, 3.5), color=GREEN)
    _arrow(ax, (6.2, 4.0), (7.4, 4.0), color=PURPLE)

    # HER note
    ax.text(5.0, 1.3,
            "Hindsight Experience Replay (HER):\n"
            "relabel failed goals with achieved states\n"
            r"$(s_t, a_t, r, s_{t+1}, g) \rightarrow (s_t, a_t, r', s_{t+1}, g'=s_T)$",
            ha="center", fontsize=9, color=DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#dbeafe",
                      edgecolor=BLUE, lw=0.8))

    # ----- Right: 2-D goal landscape -----
    ax = axes[1]
    ax.set_title("Reaching different goals from one policy", loc="left")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # start position
    ax.scatter([0], [0], s=180, color=BLUE, zorder=5,
               edgecolor="white", lw=1.5)
    ax.text(0.05, 0.08, "start", fontsize=9, color=BLUE, fontweight="bold")

    goals = np.array([[0.9, 0.7], [-0.85, 0.6], [0.6, -0.95], [-0.8, -0.7]])
    colors = [PURPLE, GREEN, ORANGE, RED]
    rng = np.random.default_rng(7)
    for g, c in zip(goals, colors):
        # smooth-ish trajectory
        t = np.linspace(0, 1, 25)
        noise = rng.normal(0, 0.04, (25, 2)).cumsum(axis=0) * 0.08
        x = g[0] * t + noise[:, 0] * (1 - t)
        y = g[1] * t + noise[:, 1] * (1 - t)
        ax.plot(x, y, color=c, lw=2.0, alpha=0.9)
        ax.scatter([g[0]], [g[1]], marker="*", s=260, color=c,
                   edgecolor="white", lw=1.5, zorder=5)

    ax.text(0, -1.55,
            "Single network, four goals — generalisation comes from "
            "conditioning on $g$",
            ha="center", fontsize=9.5, style="italic", color=DARK)

    fig.suptitle("Goal-Conditioned RL — Universal Value Function "
                 "Approximators (UVFA)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig3_goal_conditioned.png")


# ---------------------------------------------------------------------------
# Fig 4: MAML inner / outer loop in parameter space
# ---------------------------------------------------------------------------
def fig4_maml_inner_outer_loop() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))

    # ----- Left: parameter-space picture -----
    ax = axes[0]
    ax.set_title(r"MAML in parameter space", loc="left")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")

    # background contours of average loss
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Z = 0.6 * (xx ** 2 + 0.7 * yy ** 2) + 0.4 * np.cos(1.4 * xx) * np.cos(1.4 * yy)
    ax.contour(xx, yy, Z, levels=10, colors=GRAY, alpha=0.35, linewidths=0.8)

    # meta-initialisation theta*
    theta = np.array([0.4, 0.2])
    ax.scatter(*theta, s=240, color=BLUE, edgecolor="white", lw=2, zorder=6)
    ax.annotate(r"$\theta$ (meta-init)", theta + np.array([0.15, 0.18]),
                color=BLUE, fontsize=11, fontweight="bold")

    # task-specific minima
    task_minima = np.array([[ 2.2,  1.6],
                            [-2.0,  1.8],
                            [-1.7, -2.0],
                            [ 2.0, -1.6]])
    task_colors = [PURPLE, GREEN, ORANGE, RED]
    for k, (m, c) in enumerate(zip(task_minima, task_colors)):
        ax.scatter(*m, marker="X", s=160, color=c, edgecolor="white",
                   lw=1.2, zorder=5)
        ax.text(m[0] + 0.12, m[1] + 0.18, f"$\\mathcal{{T}}_{k+1}^*$",
                color=c, fontsize=10.5, fontweight="bold")
        # one-step inner update: theta -> theta - alpha grad ~ direction to m
        adapted = theta + 0.45 * (m - theta)
        ax.annotate("", xy=adapted, xytext=theta,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0))
        ax.scatter(*adapted, s=70, color=c, edgecolor="white", lw=1, zorder=6)

    ax.text(0, -2.8,
            "One inner gradient step from $\\theta$ already lands "
            "near each $\\mathcal{T}_i^*$",
            ha="center", fontsize=9.5, style="italic", color=DARK)

    # ----- Right: two-loop algorithm flow -----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Two-loop optimisation", loc="left")

    _box(ax, (0.5, 5.6), 9.0, 1.0,
         r"Outer loop:  $\theta \leftarrow \theta - \beta \nabla_\theta "
         r"\sum_i \mathcal{L}_{\mathcal{T}_i}(\theta_i')$",
         BLUE, fontsize=11)

    _box(ax, (0.5, 3.6), 9.0, 1.0,
         r"Inner loop (per task $i$):  $\theta_i' = \theta - \alpha "
         r"\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$",
         PURPLE, fontsize=11)

    _box(ax, (0.5, 1.6), 9.0, 1.0,
         r"Sample tasks $\mathcal{T}_i \sim p(\mathcal{T})$",
         GREEN, fontsize=11)

    # arrows down then up
    _arrow(ax, (5.0, 1.6), (5.0, 2.6), color=GRAY)
    _arrow(ax, (5.0, 3.6), (5.0, 4.6), color=GRAY)
    _arrow(ax, (5.0, 5.6), (5.0, 6.6), color=GRAY, style="<-")

    # second-order note
    ax.text(5.0, 0.7,
            r"Meta-gradient sees $\nabla_\theta \theta_i' = "
            r"I - \alpha \nabla^2 \mathcal{L}$  →  Hessian"
            "\nFOMAML drops the second-order term (≈10× faster, "
            "<5% perf loss)",
            ha="center", fontsize=9, color=DARK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef3c7",
                      edgecolor=ORANGE, lw=0.8))

    fig.suptitle("MAML — Learning an Initialisation that Adapts Fast",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig4_maml_inner_outer_loop.png")


# ---------------------------------------------------------------------------
# Fig 5: Meta-RL across task distribution
# ---------------------------------------------------------------------------
def fig5_meta_rl_distribution() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ----- Left: task distribution p(T) -----
    ax = axes[0]
    ax.set_title(r"Task distribution  $p(\mathcal{T})$", loc="left")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("goal x")
    ax.set_ylabel("goal y")

    # plot a ring of training tasks and a few held-out ones
    rng = np.random.default_rng(11)
    n_train = 22
    angles = rng.uniform(0, 2 * np.pi, n_train)
    radii = rng.uniform(0.7, 1.1, n_train)
    tx = radii * np.cos(angles)
    ty = radii * np.sin(angles)
    ax.scatter(tx, ty, s=80, color=BLUE, alpha=0.75,
               edgecolor="white", lw=0.8, label="train tasks")

    # test tasks
    test_x = np.array([0.95, -0.85, 0.0])
    test_y = np.array([0.25, 0.55, -1.05])
    ax.scatter(test_x, test_y, marker="*", s=320, color=ORANGE,
               edgecolor="white", lw=1.5, label="meta-test tasks", zorder=5)

    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=GRAY,
                        ls="--", lw=1.0, alpha=0.6))
    ax.scatter([0], [0], s=120, color=DARK, zorder=4)
    ax.text(0.04, 0.06, "agent", fontsize=9, color=DARK)
    ax.legend(loc="upper right", fontsize=9)

    # ----- Right: adaptation curves -----
    ax = axes[1]
    ax.set_title("Adaptation on a held-out task", loc="left")
    ax.set_xlabel("gradient / rollout steps on new task")
    ax.set_ylabel("episode return")

    x = np.arange(0, 25)

    def curve(start, end, k, noise_scale, seed):
        r = np.random.default_rng(seed)
        base = end - (end - start) * np.exp(-k * x)
        return base + r.normal(0, noise_scale, size=x.shape)

    scratch = curve(-50, -10, 0.05, 1.5, 1)
    pretrain = curve(-30, -5, 0.10, 1.2, 2)
    maml = curve(-20, -2, 0.40, 1.0, 3)
    rl2 = np.full_like(x, -3.0, dtype=float) + np.linspace(-15, 0, len(x)) * \
          np.exp(-0.6 * x)
    rl2 = rl2 + np.random.default_rng(4).normal(0, 0.8, size=x.shape)

    ax.plot(x, scratch, color=GRAY, lw=2.0, label="train from scratch")
    ax.plot(x, pretrain, color=GREEN, lw=2.0, label="multi-task pretrain")
    ax.plot(x, maml, color=BLUE, lw=2.4, label="MAML")
    ax.plot(x, rl2, color=PURPLE, lw=2.4, label=r"RL$^2$ (RNN)")

    ax.axhline(0, color=DARK, ls=":", lw=1, alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_xlim(0, 24)

    ax.text(12, -45,
            "Meta-trained agents reach near-optimal\n"
            "performance within a handful of episodes",
            ha="center", fontsize=9.5, style="italic", color=DARK)

    fig.suptitle("Meta-RL — Generalising Across a Distribution of Tasks",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig5_meta_rl_distribution.png")


# ---------------------------------------------------------------------------
# Fig 6: Hierarchical task decomposition tree
# ---------------------------------------------------------------------------
def fig6_task_decomposition() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    ax.set_title("MAXQ-style task hierarchy — \"make breakfast\"", loc="left")

    # Root
    _box(ax, (5.4, 5.6), 2.2, 0.9, "MakeBreakfast", BLUE, fontsize=10.5)

    # Mid level
    mids = [(0.6, "BrewCoffee", PURPLE),
            (3.6, "FryEggs", PURPLE),
            (6.6, "ToastBread", PURPLE),
            (9.6, "ServeTable", PURPLE)]
    for x, lbl, c in mids:
        _box(ax, (x, 3.6), 2.6, 0.85, lbl, c, fontsize=10)
        _arrow(ax, (6.5, 5.6), (x + 1.3, 4.45), color=GRAY, lw=1.0)

    # Primitives under FryEggs
    leaves = [(3.0, "GoTo(Stove)"),
              (5.0, "Crack(Egg)"),
              (7.0, "Flip"),
              (9.0, "Plate")]
    for x, lbl in leaves:
        _box(ax, (x - 0.85, 1.5), 1.7, 0.75, lbl, GREEN, fontsize=9)
    for x, _ in leaves:
        _arrow(ax, (4.9, 3.6), (x, 2.25), color=GRAY, lw=1.0)

    # Shared primitive note
    _arrow(ax, (1.9, 3.6), (3.0, 2.25), color=ORANGE, lw=1.4,
           style="->", curve=-0.15)
    _arrow(ax, (10.9, 3.6), (9.0, 2.25), color=ORANGE, lw=1.4,
           style="->", curve=0.15)
    ax.text(6.5, 0.85,
            "GoTo(·) and Plate are reused by multiple parents — "
            "MAXQ shares value functions across subtasks",
            ha="center", fontsize=9.5, color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7",
                      edgecolor=ORANGE, lw=0.8))

    # Layer labels
    ax.text(0.1, 6.05, "Root task", fontsize=9, color=GRAY,
            fontweight="bold")
    ax.text(0.1, 4.05, "Subtasks", fontsize=9, color=GRAY,
            fontweight="bold")
    ax.text(0.1, 1.95, "Primitives", fontsize=9, color=GRAY,
            fontweight="bold")

    fig.tight_layout()
    save(fig, "fig6_task_decomposition.png")


# ---------------------------------------------------------------------------
# Fig 7: RL^2 recurrent meta-learner
# ---------------------------------------------------------------------------
def fig7_rl_squared() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title(r"RL$^2$ — RNN unrolled across a meta-trial", loc="left")

    # RNN cells
    cell_x = [1.5, 4.5, 7.5, 10.5]
    for i, x in enumerate(cell_x):
        _box(ax, (x - 0.7, 3.2), 1.4, 1.1, f"RNN", PURPLE, fontsize=11)
        ax.text(x, 4.05, f"$h_{i}$", color="white", fontsize=8.5,
                ha="center", va="bottom", fontweight="bold")

    # hidden-state arrows between cells (carry across episodes)
    for i in range(len(cell_x) - 1):
        _arrow(ax, (cell_x[i] + 0.7, 3.75), (cell_x[i + 1] - 0.7, 3.75),
               color=PURPLE, lw=2.0)

    # inputs at each step
    for i, x in enumerate(cell_x):
        ax.text(x, 1.6,
                f"$(s_{{{i}}},\\, a_{{{i-1 if i>0 else 't-1'}}},\\, "
                f"r_{{{i-1 if i>0 else 't-1'}}},\\, d_{{{i-1 if i>0 else 't-1'}}})$",
                ha="center", fontsize=9, color=DARK)
        _arrow(ax, (x, 1.95), (x, 3.2), color=BLUE, lw=1.2)

    # action / value heads
    for i, x in enumerate(cell_x):
        _arrow(ax, (x, 4.3), (x, 5.3), color=ORANGE, lw=1.2)
        ax.text(x, 5.5, f"$a_{i},\\, v_{i}$", ha="center",
                fontsize=9.5, color=ORANGE, fontweight="bold")

    # episode boundaries
    ax.axvline(6.0, color=GRAY, ls="--", lw=1, ymin=0.05, ymax=0.85)
    ax.axvline(9.0, color=GRAY, ls="--", lw=1, ymin=0.05, ymax=0.85)
    ax.text(3.0, 0.55, "episode 1", color=GRAY, ha="center", fontsize=9)
    ax.text(7.5, 0.55, "episode 2", color=GRAY, ha="center", fontsize=9)
    ax.text(11.0, 0.55, "episode 3 …", color=GRAY, ha="center", fontsize=9)

    # Key insight box
    ax.text(6.75, 6.15,
            "Hidden state $h_t$ acts as task posterior — "
            "the RNN \"learns to learn\" inside its own forward pass",
            ha="center", fontsize=10, color=DARK, style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#dbeafe",
                      edgecolor=BLUE, lw=0.8))

    # Gradient note
    ax.text(6.75, 0.05,
            "Outer optimiser updates RNN weights with PPO/A2C across many "
            "trials drawn from $p(\\mathcal{T})$",
            ha="center", fontsize=9, color=GRAY)

    fig.tight_layout()
    save(fig, "fig7_rl_squared.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)

    fig1_options_framework()
    fig2_feudal_architecture()
    fig3_goal_conditioned()
    fig4_maml_inner_outer_loop()
    fig5_meta_rl_distribution()
    fig6_task_decomposition()
    fig7_rl_squared()

    print(f"Saved 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
