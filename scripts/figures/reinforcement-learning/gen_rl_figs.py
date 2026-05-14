#!/usr/bin/env python3
"""RL series matplotlib figures — 5 dark-theme PNGs + 1 GIF."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter

DARK_BG = "#0d1117"
COL = {"blue":"#58a6ff", "orange":"#f78166", "green":"#7ee787",
       "purple":"#d2a8ff", "yellow":"#f1e05a", "red":"#ff7b72",
       "white":"#e6edf3", "muted":"#8b949e"}

OUT = "/tmp/rl-figs"
os.makedirs(OUT, exist_ok=True)


def setup(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=COL["white"])
    for s in ax.spines.values():
        s.set_color(COL["muted"])


# ============================================================
# Fig 09a: MADDPG architecture (CTDE)
# ============================================================
def fig09_maddpg():
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    # 4 actors
    for i, x in enumerate([12, 32, 52, 72]):
        ax.add_patch(FancyBboxPatch((x, 60), 18, 20,
            boxstyle="round,pad=0.5", fc=COL["blue"], ec="none", alpha=0.85))
        ax.text(x+9, 70, f"Actor {i+1}\n$\\pi_{{{i+1}}}(a_{{{i+1}}}|o_{{{i+1}}})$",
                ha="center", va="center", color=COL["white"], fontsize=10)
    # Centralised critic
    ax.add_patch(FancyBboxPatch((30, 25), 40, 18,
        boxstyle="round,pad=0.5", fc=COL["orange"], ec="none", alpha=0.85))
    ax.text(50, 34, "Centralised critic\n$Q(s, a_1, a_2, a_3, a_4)$",
            ha="center", va="center", color=COL["white"], fontsize=11, fontweight="bold")
    # Arrows from actors to critic
    for x in [21, 41, 61, 81]:
        ax.annotate("", xy=(x-x*0.1+30, 43), xytext=(x, 60),
            arrowprops=dict(arrowstyle="->", color=COL["green"], lw=1.5, alpha=0.8))
    # Env at bottom
    ax.add_patch(FancyBboxPatch((20, 5), 60, 12,
        boxstyle="round,pad=0.5", fc=COL["purple"], ec="none", alpha=0.7))
    ax.text(50, 11, "Environment (joint state $s$, joint action $\\mathbf{a}$, reward $r$)",
            ha="center", va="center", color=COL["white"], fontsize=11)
    # Title
    ax.text(50, 92, "MADDPG: Centralised training, decentralised execution",
            ha="center", color=COL["white"], fontsize=14, fontweight="bold")
    ax.text(50, 87, "training-time only: critic sees all actions", ha="center",
            color=COL["muted"], fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig09_maddpg.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(); print("OK fig09_maddpg")


# ============================================================
# Fig 10a: CQL conservative Q
# ============================================================
def fig10_cql():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    np.random.seed(7)
    # Left: Q-values w/ and w/o CQL
    ax = axes[0]; setup(ax)
    actions = np.linspace(-2, 2, 100)
    q_in_data = np.exp(-(actions - 0.5)**2 / 0.4) * 1.2
    q_naive = q_in_data + 0.6 * np.exp(-(actions + 1.3)**2 / 0.2)  # spurious peak OOD
    q_cql = q_in_data - 0.5 * np.exp(-(actions + 1.3)**2 / 0.2)  # penalised OOD
    ax.fill_between(actions, q_in_data, alpha=0.25, color=COL["green"], label="Behaviour density")
    ax.plot(actions, q_naive, "-", color=COL["red"], lw=2.5, label="Naive Q (overestimates OOD)")
    ax.plot(actions, q_cql, "-", color=COL["blue"], lw=2.5, label="CQL (penalises OOD)")
    ax.axhline(0, ls="--", color=COL["muted"], alpha=0.5)
    ax.set_xlabel("Action $a$", color=COL["white"])
    ax.set_ylabel("$Q(s, a)$", color=COL["white"])
    ax.set_title("Conservative Q-Learning vs Naive Q",
                 color=COL["white"], fontsize=12)
    ax.legend(facecolor=DARK_BG, edgecolor=COL["muted"],
              labelcolor=COL["white"], fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.15)

    # Right: Q-divergence over training (D4RL halfcheetah)
    ax2 = axes[1]; setup(ax2)
    steps = np.arange(0, 100)
    np.random.seed(11)
    naive_q = 50 * np.exp(0.06 * steps) + 5 * np.random.randn(100) * np.linspace(1, 8, 100)
    cql_q = 50 + 30 * (1 - np.exp(-0.06 * steps)) + 2 * np.random.randn(100)
    iql_q = 50 + 25 * (1 - np.exp(-0.05 * steps)) + 1.5 * np.random.randn(100)
    ax2.semilogy(steps, naive_q, "-", color=COL["red"], lw=2, label="Naive offline SAC (diverges)")
    ax2.plot(steps, cql_q, "-", color=COL["blue"], lw=2, label="CQL (bounded)")
    ax2.plot(steps, iql_q, "-", color=COL["green"], lw=2, label="IQL (bounded)")
    ax2.set_xlabel("Training step (k)", color=COL["white"])
    ax2.set_ylabel("Estimated $Q$ (log)", color=COL["white"])
    ax2.set_title("Q-divergence on D4RL halfcheetah-medium",
                  color=COL["white"], fontsize=12)
    ax2.legend(facecolor=DARK_BG, edgecolor=COL["muted"],
               labelcolor=COL["white"], fontsize=9)
    ax2.grid(True, alpha=0.15, which="both")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig10_cql.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(); print("OK fig10_cql")


# ============================================================
# Fig 11a: Options framework + feudal hierarchy
# ============================================================
def fig11_hierarchy():
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    # Manager (top)
    ax.add_patch(FancyBboxPatch((30, 75), 40, 15, boxstyle="round,pad=0.5",
                                 fc=COL["purple"], ec="none", alpha=0.85))
    ax.text(50, 82.5, "Manager (slow, every $k$ steps)\noutputs goal $g_t \\in \\mathbb{R}^d$",
            ha="center", va="center", color=COL["white"], fontsize=11, fontweight="bold")
    # Worker (middle)
    ax.add_patch(FancyBboxPatch((30, 45), 40, 15, boxstyle="round,pad=0.5",
                                 fc=COL["blue"], ec="none", alpha=0.85))
    ax.text(50, 52.5, "Worker (fast, every step)\n$\\pi_W(a_t \\mid s_t, g_t)$",
            ha="center", va="center", color=COL["white"], fontsize=11, fontweight="bold")
    # Env
    ax.add_patch(FancyBboxPatch((30, 15), 40, 15, boxstyle="round,pad=0.5",
                                 fc=COL["orange"], ec="none", alpha=0.7))
    ax.text(50, 22.5, "Environment\n$s_{t+1}, r_t = \\mathcal{P}(s_t, a_t)$",
            ha="center", va="center", color=COL["white"], fontsize=11)
    # Arrows
    ax.annotate("", xy=(50, 60), xytext=(50, 75),
        arrowprops=dict(arrowstyle="->", color=COL["green"], lw=2, mutation_scale=18))
    ax.text(54, 67.5, "goal $g_t$", color=COL["green"], fontsize=10)
    ax.annotate("", xy=(50, 30), xytext=(50, 45),
        arrowprops=dict(arrowstyle="->", color=COL["yellow"], lw=2, mutation_scale=18))
    ax.text(54, 37.5, "action $a_t$", color=COL["yellow"], fontsize=10)
    # State feedback (right side)
    ax.annotate("", xy=(74, 52.5), xytext=(74, 22.5),
        arrowprops=dict(arrowstyle="->", color=COL["red"], lw=2, mutation_scale=18,
                        connectionstyle="arc3,rad=0.3"))
    ax.text(80, 37.5, "state $s_t$", color=COL["red"], fontsize=10)
    # Intrinsic reward dotted
    ax.annotate("", xy=(26, 52.5), xytext=(26, 82.5),
        arrowprops=dict(arrowstyle="->", color=COL["muted"], lw=1.5, ls=":",
                        mutation_scale=14))
    ax.text(20, 67, "intrinsic\nreward\n$-\\|s-g\\|$", color=COL["muted"],
            fontsize=9, ha="right")
    ax.text(50, 95, "Feudal RL: Manager-Worker hierarchy",
            ha="center", color=COL["white"], fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig11_hierarchy.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(); print("OK fig11_hierarchy")


# ============================================================
# Fig 04a: ICM block diagram
# ============================================================
def fig04_icm():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    boxes = [
        (5, 40, 12, 18, "$s_t$", COL["green"], "State $t$"),
        (22, 40, 12, 18, "$\\phi$", COL["blue"], "Encoder"),
        (39, 40, 12, 18, "$\\phi(s_t)$", COL["purple"], "Features"),
        (5, 12, 12, 18, "$s_{t+1}$", COL["green"], "State $t+1$"),
        (22, 12, 12, 18, "$\\phi$", COL["blue"], "Encoder (shared)"),
        (39, 12, 12, 18, "$\\phi(s_{t+1})$", COL["purple"], "Features"),
        (60, 60, 18, 18, "Forward\nmodel $f$", COL["orange"], None),
        (60, 30, 18, 18, "Inverse\nmodel $g$", COL["yellow"], None),
        (84, 60, 12, 18, "$\\hat\\phi_{t+1}$", COL["red"], "Predicted"),
        (84, 30, 12, 18, "$\\hat a_t$", COL["red"], "Predicted action"),
    ]
    for x, y, w, h, label, color, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                     fc=color, ec="none", alpha=0.85))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                color=COL["white"], fontsize=10, fontweight="bold")
        if sub:
            ax.text(x + w/2, y - 3, sub, ha="center", va="top",
                    color=COL["muted"], fontsize=8)
    # Arrows
    arrows = [(17, 49, 22, 49), (34, 49, 39, 49),
              (17, 21, 22, 21), (34, 21, 39, 21),
              (51, 55, 60, 65), (51, 45, 60, 35), (51, 25, 60, 35),
              (78, 65, 84, 65), (78, 35, 84, 35)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=COL["white"], lw=1.5, alpha=0.7))
    # Action input to forward
    ax.text(50, 52, "$a_t$", color=COL["yellow"], fontsize=11, fontweight="bold")
    # Curiosity bonus arrow
    ax.annotate("", xy=(96, 75), xytext=(96, 70),
        arrowprops=dict(arrowstyle="->", color=COL["red"], lw=2))
    ax.text(98, 90, "$r^i_t = \\eta\\|\\hat\\phi_{t+1} - \\phi(s_{t+1})\\|^2$",
            color=COL["red"], fontsize=11, fontweight="bold")
    ax.text(50, 95, "Intrinsic Curiosity Module (ICM)",
            ha="center", color=COL["white"], fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig04_icm.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(); print("OK fig04_icm")


# ============================================================
# Fig overall: RL methods landscape (Art 01 boost too)
# ============================================================
def fig_landscape():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(DARK_BG); setup(ax)
    methods = [
        ("DQN",        2.0, 1.5, COL["blue"]),
        ("PPO",        2.5, 4.5, COL["green"]),
        ("SAC",        3.5, 4.0, COL["green"]),
        ("MADDPG",     4.5, 5.5, COL["orange"]),
        ("QMIX",       5.0, 5.0, COL["orange"]),
        ("CQL",        3.0, 7.5, COL["purple"]),
        ("IQL",        2.5, 7.0, COL["purple"]),
        ("AlphaGo",    7.5, 6.5, COL["red"]),
        ("MuZero",     8.0, 7.0, COL["red"]),
        ("FuN",        6.5, 4.0, COL["yellow"]),
        ("MAML-RL",    7.0, 3.0, COL["yellow"]),
        ("RLHF",       6.0, 8.5, COL["white"]),
    ]
    for name, x, y, c in methods:
        ax.scatter(x, y, s=400, c=c, edgecolors=COL["white"],
                   linewidths=1.5, alpha=0.85, zorder=5)
        ax.annotate(name, (x, y), xytext=(8, 5), textcoords="offset points",
                    color=COL["white"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Sample efficiency / planning depth →",
                  color=COL["white"], fontsize=11)
    ax.set_ylabel("Online → Offline data assumption",
                  color=COL["white"], fontsize=11)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title("RL methods landscape (chenk.top RL series)",
                 color=COL["white"], fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.15)
    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COL["blue"], markersize=12, label="Value-based"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COL["green"], markersize=12, label="Policy gradient"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COL["orange"], markersize=12, label="Multi-agent"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COL["purple"], markersize=12, label="Offline"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COL["red"], markersize=12, label="Search/planning"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COL["yellow"], markersize=12, label="Hierarchical / meta"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COL["white"], markersize=12, label="Alignment"),
    ]
    ax.legend(handles=legend_items, facecolor=DARK_BG, edgecolor=COL["muted"],
              labelcolor=COL["white"], fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_rl_landscape.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(); print("OK fig_rl_landscape")


if __name__ == "__main__":
    fig09_maddpg()
    fig10_cql()
    fig11_hierarchy()
    fig04_icm()
    fig_landscape()
    print("\nAll RL figures generated in", OUT)
