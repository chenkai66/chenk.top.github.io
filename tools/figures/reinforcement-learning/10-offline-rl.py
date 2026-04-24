"""Figures for Reinforcement Learning Part 10 - Offline Reinforcement Learning.

Generates 7 publication-quality figures explaining the core challenges and
algorithms of offline RL: distributional shift, BCQ, CQL, IQL, Decision
Transformer, and the D4RL benchmark.

Figures:
    fig1_online_vs_offline   - Online vs offline RL data-collection contrast
    fig2_distribution_shift  - State / action support mismatch and Q overestimation
    fig3_bcq_architecture    - BCQ: VAE action proposal + perturbation network
    fig4_cql_penalty         - CQL: push down OOD Q-values, keep in-data values
    fig5_iql_expectile       - IQL: asymmetric expectile loss vs MSE
    fig6_decision_transformer - Decision Transformer architecture / RTG conditioning
    fig7_d4rl_benchmark      - D4RL performance comparison across algorithms

Output is written to BOTH the EN and ZH asset folders.

References (verified):
    - Levine et al., Offline Reinforcement Learning: Tutorial, Review, and
      Perspectives on Open Problems, 2020 [2005.01643]
    - Fujimoto et al., Off-Policy Deep RL without Exploration (BCQ),
      ICML 2019 [1812.02900]
    - Kumar et al., Conservative Q-Learning for Offline RL (CQL),
      NeurIPS 2020 [2006.04779]
    - Kostrikov et al., Offline RL with Implicit Q-Learning (IQL),
      ICLR 2022 [2110.06169]
    - Chen et al., Decision Transformer: RL via Sequence Modeling,
      NeurIPS 2021 [2106.01345]
    - Fu et al., D4RL: Datasets for Deep Data-Driven RL, 2020 [2004.07219]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    Ellipse,
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
EN_DIR = REPO / "source/_posts/en/reinforcement-learning/10-offline-reinforcement-learning"
ZH_DIR = REPO / "source/_posts/zh/reinforcement-learning/10-离线强化学习"


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: Online vs Offline RL - data collection paradigms
# ---------------------------------------------------------------------------
def fig1_online_vs_offline() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ---------- Left: Online RL loop ----------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Online RL: interactive trial and error", color=DARK)

    # Agent box
    agent = FancyBboxPatch((0.7, 4.5), 2.4, 1.8,
                           boxstyle="round,pad=0.1", facecolor=BLUE,
                           edgecolor="white", linewidth=2)
    ax.add_patch(agent)
    ax.text(1.9, 5.4, "Agent\n$\\pi_\\theta$", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")

    # Environment box
    env = FancyBboxPatch((6.9, 4.5), 2.4, 1.8,
                         boxstyle="round,pad=0.1", facecolor=GREEN,
                         edgecolor="white", linewidth=2)
    ax.add_patch(env)
    ax.text(8.1, 5.4, "Environment", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")

    # Action arrow
    ax.annotate("", xy=(6.85, 5.7), xytext=(3.15, 5.7),
                arrowprops=dict(arrowstyle="->", lw=2.4, color=DARK))
    ax.text(5.0, 6.0, "action $a_t$", ha="center", color=DARK, fontsize=10.5)

    # Reward / state arrow
    ax.annotate("", xy=(3.15, 5.1), xytext=(6.85, 5.1),
                arrowprops=dict(arrowstyle="->", lw=2.4, color=DARK))
    ax.text(5.0, 4.7, "$r_t, s_{t+1}$", ha="center", color=DARK, fontsize=10.5)

    # Update arrow back to agent (curved)
    ax.annotate(
        "", xy=(1.9, 4.45), xytext=(1.9, 2.6),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=PURPLE,
                        connectionstyle="arc3,rad=0"),
    )
    ax.text(1.9, 2.15, "policy update\nfrom fresh data",
            ha="center", color=PURPLE, fontsize=10, style="italic")

    # Rolling exploration data illustration
    ax.text(5.0, 1.2,
            "Data distribution evolves with the policy.\n"
            "Bad actions can be tried, observed, and corrected.",
            ha="center", fontsize=10.5, color=DARK,
            bbox=dict(facecolor="#f8fafc", edgecolor=GRAY, lw=0.8,
                      boxstyle="round,pad=0.4"))

    # ---------- Right: Offline RL ----------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Offline RL: learn from a fixed log only",
                 color=DARK)

    # Behavior policy box (greyed out, past)
    beta = FancyBboxPatch((0.5, 6.1), 2.6, 1.3,
                          boxstyle="round,pad=0.08", facecolor=GRAY,
                          edgecolor="white", linewidth=2)
    ax.add_patch(beta)
    ax.text(1.8, 6.75, "Behavior $\\pi_\\beta$\n(past, unknown)",
            ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold")

    # Dataset cylinder-ish
    data = FancyBboxPatch((4.0, 5.5), 2.6, 2.0,
                          boxstyle="round,pad=0.08", facecolor=ORANGE,
                          edgecolor="white", linewidth=2)
    ax.add_patch(data)
    ax.text(5.3, 6.5,
            "Dataset $\\mathcal{D}$\n$\\{(s,a,r,s')\\}$",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")

    # Behavior -> Dataset arrow
    ax.annotate("", xy=(3.95, 6.5), xytext=(3.15, 6.75),
                arrowprops=dict(arrowstyle="->", lw=2.0, color=DARK))

    # Agent box
    agent2 = FancyBboxPatch((7.4, 5.5), 2.2, 2.0,
                            boxstyle="round,pad=0.08", facecolor=BLUE,
                            edgecolor="white", linewidth=2)
    ax.add_patch(agent2)
    ax.text(8.5, 6.5, "Agent\n$\\pi_\\theta$",
            ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")

    # Dataset -> Agent arrow
    ax.annotate("", xy=(7.35, 6.5), xytext=(6.65, 6.5),
                arrowprops=dict(arrowstyle="->", lw=2.4, color=DARK))
    ax.text(7.0, 6.95, "minibatches", ha="center", color=DARK, fontsize=10)

    # No-interaction big red cross
    ax.annotate(
        "", xy=(8.5, 4.6), xytext=(8.5, 3.6),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=RED, alpha=0.9),
    )
    ax.text(8.5, 3.25, "no environment\ninteraction allowed",
            ha="center", color=RED, fontsize=10, fontweight="bold")
    # Big crossed-out env
    env2 = FancyBboxPatch((6.9, 1.4), 3.2, 1.6,
                          boxstyle="round,pad=0.08", facecolor=LIGHT,
                          edgecolor=RED, linewidth=2, linestyle="--")
    ax.add_patch(env2)
    ax.text(8.5, 2.2, "Environment", ha="center", va="center",
            color=RED, fontsize=12, fontweight="bold", alpha=0.6)
    ax.plot([6.95, 10.05], [1.45, 2.95], color=RED, lw=2.5, alpha=0.8)
    ax.plot([6.95, 10.05], [2.95, 1.45], color=RED, lw=2.5, alpha=0.8)

    ax.text(2.7, 2.3,
            "Data is frozen.\n"
            "OOD actions can never\nbe verified or corrected.",
            ha="center", fontsize=10.5, color=DARK,
            bbox=dict(facecolor="#fff7ed", edgecolor=ORANGE, lw=0.8,
                      boxstyle="round,pad=0.4"))

    fig.suptitle(
        "Offline RL removes the feedback loop that traditional RL relies on",
        fontsize=13.5, weight="bold", y=1.01,
    )
    fig.tight_layout()
    save(fig, "fig1_online_vs_offline.png")


# ---------------------------------------------------------------------------
# Fig 2: Distribution shift - in-data support vs OOD Q overestimation
# ---------------------------------------------------------------------------
def fig2_distribution_shift() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # ---------- Left: action distributions ----------
    ax = axes[0]
    a = np.linspace(-3, 3, 600)

    # Behavior policy: mixture of two narrow modes
    beh = (0.55 * np.exp(-0.5 * ((a + 0.9) / 0.45) ** 2)
           + 0.45 * np.exp(-0.5 * ((a - 0.7) / 0.4) ** 2))
    beh /= beh.max()

    # Learned policy: drifting toward a high-Q (but uncovered) region
    pi = np.exp(-0.5 * ((a - 1.9) / 0.55) ** 2)

    ax.fill_between(a, 0, beh, color=BLUE, alpha=0.25,
                    label=r"data $\pi_\beta(a|s)$")
    ax.plot(a, beh, color=BLUE, lw=2.2)
    ax.fill_between(a, 0, pi, color=PURPLE, alpha=0.25,
                    label=r"learned $\pi_\theta(a|s)$")
    ax.plot(a, pi, color=PURPLE, lw=2.2, linestyle="--")

    # Highlight OOD region
    ood_mask = (a > 1.3)
    ax.fill_between(a[ood_mask], 0, np.maximum(beh[ood_mask],
                    pi[ood_mask]), color=RED, alpha=0.10)
    ax.axvspan(1.3, 3.0, color=RED, alpha=0.07)
    ax.text(2.15, 0.92, "out-of-distribution\n(OOD) actions",
            color=RED, fontsize=10.5, ha="center", fontweight="bold")
    ax.annotate("", xy=(1.85, 0.78), xytext=(1.85, 0.88),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))

    ax.set_xlabel("action $a$")
    ax.set_ylabel("density (normalized)")
    ax.set_title("Action support mismatch", color=DARK)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1.15)

    # ---------- Right: True Q vs learned Q with OOD overestimation ----------
    ax = axes[1]
    a = np.linspace(-3, 3, 600)
    # True Q-function: peaks inside data
    Q_true = 1.5 * np.exp(-0.5 * ((a + 0.9) / 0.55) ** 2) \
           + 1.6 * np.exp(-0.5 * ((a - 0.7) / 0.5) ** 2)
    # Learned Q: matches inside data, blows up on OOD due to extrapolation
    Q_learned = Q_true.copy()
    extrap_bump = 3.0 * np.exp(-0.5 * ((a - 1.9) / 0.55) ** 2)
    Q_learned = np.where(a > 1.0, Q_true + extrap_bump * (a - 1.0) / 0.9,
                         Q_learned)

    ax.fill_between(a, 0, np.where(np.abs(a + 0.9) < 1.1, 1, 0) * 0.06
                    + np.where(np.abs(a - 0.7) < 1.0, 1, 0) * 0.06,
                    color=BLUE, alpha=0.10, step=None,
                    label="data support")

    ax.plot(a, Q_true, color=GREEN, lw=2.6, label="true $Q^\\pi$")
    ax.plot(a, Q_learned, color=RED, lw=2.6, linestyle="--",
            label="learned $\\hat{Q}$")

    # Mark policy choice
    a_star = a[np.argmax(Q_learned)]
    ax.axvline(a_star, color=PURPLE, lw=1.5, linestyle=":")
    ax.scatter([a_star], [Q_learned.max()], s=120, color=PURPLE,
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.text(a_star + 0.05, Q_learned.max() + 0.15,
            r"$\arg\max_a \hat{Q}$",
            fontsize=11, color=PURPLE, fontweight="bold")

    # Mark gap
    a_gap = 1.9
    idx = np.argmin(np.abs(a - a_gap))
    ax.annotate("",
                xy=(a_gap, Q_learned[idx]), xytext=(a_gap, Q_true[idx]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text(a_gap + 0.15, (Q_learned[idx] + Q_true[idx]) / 2,
            "extrapolation\nerror",
            color=RED, fontsize=10.5, fontweight="bold")

    ax.set_xlabel("action $a$")
    ax.set_ylabel("Q-value")
    ax.set_title("Q-overestimation on OOD actions", color=DARK)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 5.5)

    fig.suptitle(
        "Distributional shift: the policy queries Q where the data is silent",
        fontsize=13.5, weight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig2_distribution_shift.png")


# ---------------------------------------------------------------------------
# Fig 3: BCQ - generative action proposal + perturbation
# ---------------------------------------------------------------------------
def fig3_bcq_architecture() -> None:
    fig = plt.figure(figsize=(14.0, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])

    # ---------- Left: schematic ----------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("BCQ pipeline: stay inside the behavior policy support",
                 color=DARK)

    # State input
    s_box = FancyBboxPatch((0.3, 3.4), 1.5, 1.3,
                           boxstyle="round,pad=0.08", facecolor=GRAY,
                           edgecolor="white", linewidth=2)
    ax.add_patch(s_box)
    ax.text(1.05, 4.05, "state $s$", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")

    # VAE generator G(s) - samples N candidate actions
    vae = FancyBboxPatch((2.4, 5.2), 2.5, 1.5,
                         boxstyle="round,pad=0.08", facecolor=BLUE,
                         edgecolor="white", linewidth=2)
    ax.add_patch(vae)
    ax.text(3.65, 5.95, "VAE $G_\\omega(s)$\nsample $N$ actions",
            ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold")

    # Perturbation network xi(s,a)
    pert = FancyBboxPatch((2.4, 1.2), 2.5, 1.5,
                          boxstyle="round,pad=0.08", facecolor=PURPLE,
                          edgecolor="white", linewidth=2)
    ax.add_patch(pert)
    ax.text(3.65, 1.95,
            "Perturbation\n$\\xi_\\phi(s,a)\\in[-\\Phi,\\Phi]$",
            ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold")

    # Twin Q
    qbox = FancyBboxPatch((6.2, 3.4), 2.5, 1.5,
                          boxstyle="round,pad=0.08", facecolor=GREEN,
                          edgecolor="white", linewidth=2)
    ax.add_patch(qbox)
    ax.text(7.45, 4.15, "Twin $Q_{\\theta_1}, Q_{\\theta_2}$\nargmax",
            ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold")

    # Final action
    aout = FancyBboxPatch((9.0, 3.4), 0.9, 1.3,
                          boxstyle="round,pad=0.08", facecolor=ORANGE,
                          edgecolor="white", linewidth=2)
    ax.add_patch(aout)
    ax.text(9.45, 4.05, "$a^*$", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")

    # Arrows
    ax.annotate("", xy=(2.35, 5.6), xytext=(1.85, 4.4),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))
    ax.annotate("", xy=(2.35, 2.0), xytext=(1.85, 3.7),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))
    ax.annotate("", xy=(3.65, 2.75), xytext=(3.65, 5.15),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2,
                                linestyle="--"))
    ax.text(3.95, 3.95, "perturb each\ncandidate",
            color=DARK, fontsize=9.5, style="italic")
    ax.annotate("", xy=(6.15, 4.15), xytext=(4.95, 4.15),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))
    ax.text(5.55, 4.4, "$N$ candidates", ha="center",
            color=DARK, fontsize=9.5)
    ax.annotate("", xy=(8.95, 4.05), xytext=(8.75, 4.05),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2.4))

    ax.text(5.0, 0.4,
            "Action only ever comes from $G(s)+\\xi$, "
            "never from the full action space.",
            ha="center", color=DARK, fontsize=10.5, style="italic",
            bbox=dict(facecolor="#f1f5f9", edgecolor=GRAY, lw=0.8,
                      boxstyle="round,pad=0.3"))

    # ---------- Right: visualization in action space ----------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Candidate actions stay near data support",
                 color=DARK)

    # Behavior policy support: a banana shape
    rng = np.random.default_rng(7)
    t = rng.uniform(-1.4, 1.4, 250)
    bx = t + rng.normal(0, 0.10, t.size)
    by = -0.6 * t**2 + 0.8 + rng.normal(0, 0.08, t.size)
    ax.scatter(bx, by, s=20, color=BLUE, alpha=0.35,
               label=r"data $(a_1,a_2)\sim\pi_\beta$")

    # VAE samples cluster near data
    n = 30
    tt = rng.uniform(-1.3, 1.3, n)
    vx = tt + rng.normal(0, 0.07, n)
    vy = -0.6 * tt**2 + 0.8 + rng.normal(0, 0.06, n)
    ax.scatter(vx, vy, s=70, color=PURPLE, marker="X",
               label="VAE candidates $G(s)$",
               edgecolors="white", linewidths=0.8)

    # Show perturbation circles around a few
    for i in (3, 12, 22):
        c = Circle((vx[i], vy[i]), 0.18, facecolor="none",
                   edgecolor=PURPLE, lw=1.0, linestyle="--")
        ax.add_patch(c)

    # Argmax pick
    ax.scatter([vx[12]], [vy[12]], s=260, color=ORANGE, marker="*",
               edgecolors=DARK, linewidths=1.4,
               label=r"$a^*=\arg\max_a Q(s,a)$", zorder=10)

    # Show what would be OOD
    ood_pts = np.array([[1.7, -1.0], [-1.6, -1.3], [1.4, 1.6]])
    ax.scatter(ood_pts[:, 0], ood_pts[:, 1], s=110, color=RED,
               marker="X", label="OOD (BCQ avoids)",
               edgecolors="white", linewidths=0.8)

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.8, 2.2)
    ax.set_xlabel("action dim 1")
    ax.set_ylabel("action dim 2")
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    save(fig, "fig3_bcq_architecture.png")


# ---------------------------------------------------------------------------
# Fig 4: CQL - pessimistic Q-values pull down OOD
# ---------------------------------------------------------------------------
def fig4_cql_penalty() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))

    a = np.linspace(-3, 3, 600)
    # In-data Q surface
    Q_true = 1.5 * np.exp(-0.5 * ((a + 0.9) / 0.55) ** 2) \
           + 1.6 * np.exp(-0.5 * ((a - 0.7) / 0.5) ** 2)
    extrap_bump = 3.0 * np.exp(-0.5 * ((a - 1.9) / 0.55) ** 2)
    Q_naive = np.where(a > 1.0, Q_true + extrap_bump * (a - 1.0) / 0.9,
                       Q_true)
    # Behavior data density
    beh = (0.55 * np.exp(-0.5 * ((a + 0.9) / 0.45) ** 2)
           + 0.45 * np.exp(-0.5 * ((a - 0.7) / 0.4) ** 2))
    beh_norm = beh / beh.max()

    # ---------- Left: naive Q ----------
    ax = axes[0]
    ax.fill_between(a, 0, beh_norm * 1.6, color=BLUE, alpha=0.10,
                    label="data density")
    ax.plot(a, Q_true, color=GREEN, lw=2.4, label="true $Q$")
    ax.plot(a, Q_naive, color=RED, lw=2.4, linestyle="--",
            label="naive $\\hat{Q}$ (overestimates OOD)")
    ax.scatter([a[np.argmax(Q_naive)]], [Q_naive.max()],
               s=170, color=RED, marker="X", zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.set_title("Without CQL: argmax flees to OOD", color=DARK)
    ax.set_xlabel("action $a$")
    ax.set_ylabel("Q-value")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 5.5)

    # ---------- Right: CQL pulls OOD down ----------
    ax = axes[1]
    # CQL effective Q: subtract a function that spikes on low-density actions
    penalty = 2.6 * (1.0 - beh_norm)
    Q_cql = Q_naive - penalty
    # Clip floor for visualization
    Q_cql = np.maximum(Q_cql, -0.5)

    ax.fill_between(a, 0, beh_norm * 1.6, color=BLUE, alpha=0.10,
                    label="data density")
    ax.plot(a, Q_true, color=GREEN, lw=2.4, label="true $Q$")
    ax.plot(a, Q_cql, color=PURPLE, lw=2.4,
            label="CQL $\\hat{Q}$ (pessimistic)")
    ax.fill_between(a, Q_cql, Q_naive, color=ORANGE, alpha=0.20,
                    label="CQL penalty")

    a_star_cql = a[np.argmax(Q_cql)]
    ax.scatter([a_star_cql], [Q_cql.max()], s=170, color=PURPLE,
               marker="*", zorder=5, edgecolors="white", linewidths=1.5)
    ax.axvline(a_star_cql, color=PURPLE, lw=1.2, linestyle=":")
    ax.text(a_star_cql + 0.08, Q_cql.max() + 0.2,
            "argmax stays\nin data", color=PURPLE,
            fontsize=10.5, fontweight="bold")

    ax.set_title("With CQL: OOD Q-values are pushed down", color=DARK)
    ax.set_xlabel("action $a$")
    ax.set_ylabel("Q-value")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 5.5)

    # CQL formula annotation
    fig.text(
        0.5, -0.04,
        r"$\mathcal{L}_{\mathrm{CQL}}=\alpha\,[\log\sum_a "
        r"e^{Q(s,a)}-\mathbb{E}_{a\sim\mathcal{D}}Q(s,a)]"
        r"+\mathcal{L}_{\mathrm{TD}}\;\Rightarrow\;"
        r"\hat{Q}^\pi(s,a)\leq Q^\pi(s,a)$",
        ha="center", fontsize=11.5, color=DARK,
    )
    fig.suptitle(
        "CQL: a lower-bound on Q ensures safe argmax",
        fontsize=13.5, weight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig4_cql_penalty.png")


# ---------------------------------------------------------------------------
# Fig 5: IQL - expectile regression
# ---------------------------------------------------------------------------
def fig5_iql_expectile() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # ---------- Left: expectile loss ----------
    ax = axes[0]
    u = np.linspace(-2.2, 2.2, 400)
    for tau, color, label in [
        (0.5, GRAY, r"$\tau=0.5$ (MSE)"),
        (0.7, BLUE, r"$\tau=0.7$"),
        (0.9, PURPLE, r"$\tau=0.9$"),
    ]:
        loss = np.abs(tau - (u < 0).astype(float)) * u**2
        ax.plot(u, loss, lw=2.4, color=color, label=label)

    ax.axvline(0, color=DARK, lw=0.8, linestyle=":")
    ax.set_title(r"Asymmetric expectile loss $L_\tau(u)$", color=DARK)
    ax.set_xlabel(r"residual $u = Q(s,a) - V(s)$")
    ax.set_ylabel("loss")
    ax.legend(fontsize=10)
    ax.text(-1.9, 4.0,
            "penalize\nunder-estimates\nless heavily",
            color=BLUE, fontsize=10, ha="left", style="italic")
    ax.text(0.5, 4.0,
            "penalize\nover-estimates\nas usual",
            color=PURPLE, fontsize=10, ha="left", style="italic")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0, 5)

    # ---------- Right: V approximates max without ever querying max ----------
    ax = axes[1]
    rng = np.random.default_rng(0)
    a = rng.uniform(-2.5, 2.5, 220)
    # True Q surface
    Q = 1.5 * np.exp(-0.5 * ((a + 0.9) / 0.55) ** 2) \
      + 1.6 * np.exp(-0.5 * ((a - 0.7) / 0.5) ** 2)
    Q += rng.normal(0, 0.18, a.size)
    ax.scatter(a, Q, s=22, color=BLUE, alpha=0.55,
               label=r"$(a_i, Q(s,a_i))$ in data")

    # Compute "expectile values" of Q
    def expectile(arr, tau, n_iter=200):
        v = arr.mean()
        for _ in range(n_iter):
            w = np.where(arr > v, tau, 1 - tau)
            v = (w * arr).sum() / w.sum()
        return v

    v_05 = expectile(Q, 0.5)
    v_07 = expectile(Q, 0.7)
    v_09 = expectile(Q, 0.9)
    qmax = Q.max()

    ax.axhline(v_05, color=GRAY, lw=2, linestyle="--",
               label=fr"$V_{{0.5}}={v_05:.2f}$ (mean)")
    ax.axhline(v_07, color=BLUE, lw=2, linestyle="--",
               label=fr"$V_{{0.7}}={v_07:.2f}$")
    ax.axhline(v_09, color=PURPLE, lw=2, linestyle="--",
               label=fr"$V_{{0.9}}={v_09:.2f}$")
    ax.axhline(qmax, color=GREEN, lw=2,
               label=fr"$\max_a Q={qmax:.2f}$ (target)")

    ax.set_title(r"Higher $\tau$ pushes $V$ toward $\max_a Q$ "
                 r"using only in-data $a$", color=DARK)
    ax.set_xlabel("action $a$")
    ax.set_ylabel("Q-value")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.6, 3.2)

    fig.suptitle(
        "IQL: estimate the upper expectile of Q over in-data actions",
        fontsize=13.5, weight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig5_iql_expectile.png")


# ---------------------------------------------------------------------------
# Fig 6: Decision Transformer architecture
# ---------------------------------------------------------------------------
def fig6_decision_transformer() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Decision Transformer: RL as conditional sequence modeling",
                 color=DARK, fontsize=14)

    # Input token row layout: R_t, s_t, a_t triplets
    triplets = [
        (0.5,  "$R_1$", "$s_1$", "$a_1$"),
        (4.5,  "$R_2$", "$s_2$", "$a_2$"),
        (8.5,  "$R_3$", "$s_3$", "$a_3$"),
        (12.5, "$R_4$", "$s_4$", "?"),
    ]
    colors = [ORANGE, BLUE, PURPLE]

    # Draw input tokens
    for x0, R, s, a in triplets:
        for k, (lab, c) in enumerate(zip([R, s, a], colors)):
            box = FancyBboxPatch((x0 + k * 1.1, 1.2), 0.95, 0.95,
                                 boxstyle="round,pad=0.05", facecolor=c,
                                 edgecolor="white", linewidth=1.5)
            ax.add_patch(box)
            ax.text(x0 + k * 1.1 + 0.475, 1.675, lab, ha="center",
                    va="center", color="white", fontsize=11,
                    fontweight="bold")

    # Token-type labels under first triplet
    labels = ["return-to-go", "state", "action"]
    for k, lab in enumerate(labels):
        ax.text(0.5 + k * 1.1 + 0.475, 0.55, lab,
                ha="center", color=colors[k], fontsize=9, fontweight="bold")

    # Causal Transformer block
    tr = FancyBboxPatch((0.4, 3.6), 15.0, 1.8,
                        boxstyle="round,pad=0.08", facecolor=DARK,
                        edgecolor="white", linewidth=2)
    ax.add_patch(tr)
    ax.text(7.9, 4.5, "Causal Transformer (GPT-style)",
            ha="center", va="center", color="white",
            fontsize=13, fontweight="bold")

    # Arrows up from each token
    for x0, _, _, _ in triplets:
        for k in range(3):
            ax.annotate("",
                        xy=(x0 + k * 1.1 + 0.475, 3.55),
                        xytext=(x0 + k * 1.1 + 0.475, 2.2),
                        arrowprops=dict(arrowstyle="->", lw=1.4,
                                        color=GRAY))

    # Output: predict a_4 from final state token
    last_x = 12.5 + 1.1 + 0.475  # state position of last triplet
    ax.annotate("",
                xy=(last_x, 6.6), xytext=(last_x, 5.45),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=GREEN))
    pred = FancyBboxPatch((last_x - 0.95, 6.6), 1.9, 0.9,
                          boxstyle="round,pad=0.06", facecolor=GREEN,
                          edgecolor="white", linewidth=2)
    ax.add_patch(pred)
    ax.text(last_x, 7.05, "predict $\\hat{a}_4$", ha="center",
            va="center", color="white", fontsize=11, fontweight="bold")

    # Note about return-to-go
    ax.text(7.9, 7.5,
            "At test time: pick a target return $R_1$, "
            "then sample $a_t$ autoregressively.",
            ha="center", color=DARK, fontsize=10.5, style="italic",
            bbox=dict(facecolor="#fff7ed", edgecolor=ORANGE, lw=0.8,
                      boxstyle="round,pad=0.3"))

    # Show RTG definition
    ax.text(0.4, 7.5,
            r"$R_t=\sum_{t'\geq t} r_{t'}$",
            ha="left", color=ORANGE, fontsize=12, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig6_decision_transformer.png")


# ---------------------------------------------------------------------------
# Fig 7: D4RL benchmark - representative scores
# ---------------------------------------------------------------------------
def fig7_d4rl_benchmark() -> None:
    """Approximate normalized scores compiled from the original CQL,
    IQL, BCQ and Decision Transformer papers on D4RL MuJoCo locomotion.
    Numbers are illustrative averages (rounded), suitable for teaching.

    Sources:
      - CQL paper Table 1 (NeurIPS 2020)
      - IQL paper Table 1 (ICLR 2022)
      - DT paper Table 2 (NeurIPS 2021)
    """
    methods = ["BC", "BCQ", "CQL", "IQL", "DT"]
    method_colors = [GRAY, ORANGE, BLUE, PURPLE, GREEN]

    # Three D4RL splits, normalized scores (0 = random, 100 = expert)
    datasets = ["medium", "medium-replay", "medium-expert"]

    # Rows: methods, Cols: datasets - hopper-style averaged values
    scores = np.array([
        # BC          BCQ         CQL         IQL         DT
        [54.6,        56.7,       58.0,       66.3,       67.6],   # medium
        [25.6,        42.3,       63.9,       73.1,       58.7],   # m-replay
        [86.0,        74.0,       91.6,       91.5,       86.8],   # m-expert
    ]).T  # transpose -> rows = methods, cols = datasets

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))

    # ---------- Left: grouped bars ----------
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.16
    for i, (m, c) in enumerate(zip(methods, method_colors)):
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, scores[i], width, label=m, color=c,
               edgecolor="white", linewidth=1.2)
        for j, v in enumerate(scores[i]):
            ax.text(x[j] + offset, v + 1.5, f"{v:.0f}",
                    ha="center", fontsize=8.5, color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("D4RL normalized score (avg locomotion)")
    ax.set_title("D4RL MuJoCo: scores by data quality", color=DARK)
    ax.set_ylim(0, 110)
    ax.axhline(100, color=GREEN, lw=1, linestyle=":", alpha=0.6)
    ax.text(2.4, 102, "expert", color=GREEN, fontsize=9, ha="right")
    ax.legend(ncol=5, fontsize=9, loc="upper left")

    # ---------- Right: average across data quality ----------
    ax = axes[1]
    avg = scores.mean(axis=1)
    bars = ax.barh(methods, avg, color=method_colors,
                   edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, avg):
        ax.text(v + 1.0, b.get_y() + b.get_height() / 2,
                f"{v:.1f}", va="center", color=DARK,
                fontsize=10.5, fontweight="bold")

    ax.set_xlabel("average normalized score")
    ax.set_title("Mean across the three splits", color=DARK)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.axvline(scores[0].mean(), color=GRAY, lw=1, linestyle=":")
    ax.text(scores[0].mean() + 0.5, -0.45,
            "BC baseline", color=GRAY, fontsize=9)

    fig.suptitle(
        "Conservative methods dominate when data is suboptimal; "
        "DT is competitive on rich data",
        fontsize=12.5, weight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig7_d4rl_benchmark.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    fig1_online_vs_offline()
    fig2_distribution_shift()
    fig3_bcq_architecture()
    fig4_cql_penalty()
    fig5_iql_expectile()
    fig6_decision_transformer()
    fig7_d4rl_benchmark()
    print(f"Saved 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
