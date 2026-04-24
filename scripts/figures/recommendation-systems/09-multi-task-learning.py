"""Figure generation script for Recommendation Systems Part 9: Multi-Task Learning.

Generates 7 figures used in both EN and ZH versions of the article:
    fig1_shared_bottom.png       - Shared-Bottom architecture
    fig2_mmoe.png                - MMoE: experts + per-task gating
    fig3_ple.png                 - PLE: shared + task-specific experts
    fig4_pareto_frontier.png     - Pareto frontier (CTR vs CVR trade-off)
    fig5_loss_balancing.png      - Uncertainty weighting & GradNorm
    fig6_gradient_conflict.png   - Task gradient conflict illustration
    fig7_esmm.png                - ESMM architecture (CTR x CVR = CTCVR)

Run from anywhere; outputs are written to both the EN and ZH asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Output directories (figures are duplicated to EN and ZH asset folders).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = PROJECT_ROOT / "source/_posts/en/recommendation-systems/09-multi-task-learning"
ZH_DIR = PROJECT_ROOT / "source/_posts/zh/recommendation-systems/09-多任务学习与多目标优化"

# Brand palette
BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
AMBER = "#f59e0b"
INK = "#1f2937"
MUTED = "#6b7280"
LIGHT = "#f3f4f6"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.6,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def save_to_both(fig, name: str) -> None:
    """Save the figure to both the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers for drawing block diagrams.
# ---------------------------------------------------------------------------
def draw_box(ax, x, y, w, h, text, face, edge=None, text_color="white",
             fontsize=10, fontweight="bold"):
    edge = edge or face
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight=fontweight,
    )


def draw_arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.3, style="-|>",
               mutation_scale=12, alpha=1.0):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        color=color,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Figure 1: Shared-Bottom architecture.
# ---------------------------------------------------------------------------
def fig1_shared_bottom():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.set_title("Shared-Bottom: One Trunk, Many Branches", pad=14)

    # Input
    draw_box(ax, 5, 6.3, 3.0, 0.7, "Input Features  x", BLUE)

    # Shared bottom layers
    draw_box(ax, 5, 4.9, 4.4, 0.7, "Shared Bottom (MLP)", PURPLE)
    draw_box(ax, 5, 4.0, 4.4, 0.7, "Shared Representation  h", PURPLE)

    # Three task towers
    for i, (xc, name, color) in enumerate(
        [(2.0, "Task 1\nTower", GREEN), (5.0, "Task 2\nTower", AMBER),
         (8.0, "Task 3\nTower", BLUE)]
    ):
        draw_box(ax, xc, 2.5, 2.0, 1.0, name, color, fontsize=10)
        draw_box(ax, xc, 1.0, 1.6, 0.6, ["CTR", "CVR", "Stay"][i], INK,
                 fontsize=10)
        draw_arrow(ax, 5, 3.65, xc, 3.05, color=MUTED, lw=1.3)
        draw_arrow(ax, xc, 1.95, xc, 1.32, color=MUTED, lw=1.3)

    draw_arrow(ax, 5, 5.95, 5, 5.27, color=MUTED, lw=1.3)
    draw_arrow(ax, 5, 4.55, 5, 4.37, color=MUTED, lw=1.3)

    # Caption
    ax.text(
        5,
        0.25,
        "All tasks are forced to share one bottleneck representation.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=10,
        style="italic",
    )

    save_to_both(fig, "fig1_shared_bottom.png")


# ---------------------------------------------------------------------------
# Figure 2: MMoE -- experts + per-task gating.
# ---------------------------------------------------------------------------
def fig2_mmoe():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.set_axis_off()
    ax.set_title("MMoE: Each Task Owns Its Gate Over Shared Experts", pad=14)

    # Input
    draw_box(ax, 5.5, 6.6, 3.0, 0.6, "Input Features  x", BLUE, fontsize=10)

    # Four experts
    expert_xs = [1.6, 3.7, 5.8, 7.9]
    for i, xc in enumerate(expert_xs):
        draw_box(ax, xc, 5.0, 1.6, 0.7, f"Expert {i+1}", PURPLE, fontsize=10)
        draw_arrow(ax, 5.5, 6.3, xc, 5.4, color=MUTED, lw=1.0, alpha=0.6)

    # Three gates (one per task)
    gate_xs = [2.0, 5.5, 9.0]
    gate_colors = [GREEN, AMBER, BLUE]
    task_names = ["Task 1\n(CTR)", "Task 2\n(CVR)", "Task 3\n(Stay)"]
    for gx, col, name in zip(gate_xs, gate_colors, task_names):
        # gate
        draw_box(ax, gx, 3.1, 1.7, 0.6, f"Gate -> {name.splitlines()[0]}", col,
                 fontsize=9)
        # weighted sum of experts -> tower input
        for ex in expert_xs:
            draw_arrow(ax, ex, 4.65, gx, 3.4, color=col, lw=0.7, alpha=0.45)
        # tower
        draw_box(ax, gx, 1.8, 1.6, 0.7, name, col, fontsize=9)
        draw_arrow(ax, gx, 2.8, gx, 2.15, color=col, lw=1.2)
        # output
        ax.text(gx, 0.85, "y", ha="center", va="center",
                fontsize=11, color=INK, fontweight="bold", style="italic")
        draw_arrow(ax, gx, 1.45, gx, 1.05, color=col, lw=1.2)

    ax.text(
        5.5,
        0.2,
        "Different tasks softly select different expert mixtures -- conflicts can be routed apart.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=10,
        style="italic",
    )

    save_to_both(fig, "fig2_mmoe.png")


# ---------------------------------------------------------------------------
# Figure 3: PLE -- shared + task-specific experts.
# ---------------------------------------------------------------------------
def fig3_ple():
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)
    ax.set_axis_off()
    ax.set_title("PLE: Explicit Split of Shared and Task-Specific Experts", pad=14)

    # Input
    draw_box(ax, 5.5, 7.0, 2.8, 0.55, "Input Features  x", BLUE, fontsize=10)

    # Layer 1: task A experts | shared experts | task B experts
    a1_xs = [1.4, 2.6]
    s_xs = [4.4, 5.5, 6.6]
    b1_xs = [8.4, 9.6]

    for x in a1_xs:
        draw_box(ax, x, 5.6, 1.0, 0.55, "A-Exp", GREEN, fontsize=9)
    for x in s_xs:
        draw_box(ax, x, 5.6, 1.0, 0.55, "Shared", PURPLE, fontsize=9)
    for x in b1_xs:
        draw_box(ax, x, 5.6, 1.0, 0.55, "B-Exp", AMBER, fontsize=9)

    # Gate A (uses A + shared) and Gate B (uses shared + B)
    draw_box(ax, 2.0, 4.3, 2.4, 0.55, "Gate A", GREEN, fontsize=9)
    draw_box(ax, 9.0, 4.3, 2.4, 0.55, "Gate B", AMBER, fontsize=9)

    # Arrows: A experts feed Gate A; shared feed both; B experts feed Gate B
    for x in a1_xs:
        draw_arrow(ax, x, 5.32, 2.0, 4.6, color=GREEN, lw=0.7, alpha=0.7)
    for x in s_xs:
        draw_arrow(ax, x, 5.32, 2.0, 4.6, color=PURPLE, lw=0.6, alpha=0.55)
        draw_arrow(ax, x, 5.32, 9.0, 4.6, color=PURPLE, lw=0.6, alpha=0.55)
    for x in b1_xs:
        draw_arrow(ax, x, 5.32, 9.0, 4.6, color=AMBER, lw=0.7, alpha=0.7)

    # Layer 2 representation per task
    draw_box(ax, 2.0, 3.0, 2.4, 0.55, "Task A Repr", GREEN, fontsize=9)
    draw_box(ax, 9.0, 3.0, 2.4, 0.55, "Task B Repr", AMBER, fontsize=9)
    draw_arrow(ax, 2.0, 4.0, 2.0, 3.27, color=GREEN, lw=1.2)
    draw_arrow(ax, 9.0, 4.0, 9.0, 3.27, color=AMBER, lw=1.2)

    # Towers
    draw_box(ax, 2.0, 1.7, 2.0, 0.6, "Tower A (CTR)", GREEN, fontsize=9)
    draw_box(ax, 9.0, 1.7, 2.0, 0.6, "Tower B (CVR)", AMBER, fontsize=9)
    draw_arrow(ax, 2.0, 2.72, 2.0, 1.97, color=GREEN, lw=1.2)
    draw_arrow(ax, 9.0, 2.72, 9.0, 1.97, color=AMBER, lw=1.2)

    # Outputs
    ax.text(2.0, 0.8, "y_A", ha="center", va="center",
            fontsize=11, color=INK, fontweight="bold")
    ax.text(9.0, 0.8, "y_B", ha="center", va="center",
            fontsize=11, color=INK, fontweight="bold")
    draw_arrow(ax, 2.0, 1.4, 2.0, 1.0, color=GREEN, lw=1.2)
    draw_arrow(ax, 9.0, 1.4, 9.0, 1.0, color=AMBER, lw=1.2)

    # Connect input to all experts
    for x in a1_xs + s_xs + b1_xs:
        draw_arrow(ax, 5.5, 6.7, x, 5.88, color=MUTED, lw=0.5, alpha=0.5)

    ax.text(
        5.5,
        0.2,
        "Task-specific experts shield each task from the other's gradient noise.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=10,
        style="italic",
    )

    save_to_both(fig, "fig3_ple.png")


# ---------------------------------------------------------------------------
# Figure 4: Pareto frontier (CTR vs CVR).
# ---------------------------------------------------------------------------
def fig4_pareto_frontier():
    fig, ax = plt.subplots(figsize=(8.4, 6.2))

    rng = np.random.default_rng(2024)

    # Sample many candidate models in (CTR, CVR) AUC space.
    n = 120
    ctr = rng.uniform(0.62, 0.82, n)
    # Negative correlation with noise to create real trade-off.
    cvr = 1.40 - 1.0 * ctr + rng.normal(0, 0.025, n)
    cvr = np.clip(cvr, 0.55, 0.80)

    # Compute Pareto frontier (maximize both).
    order = np.argsort(-ctr)
    pareto_idx = []
    best_cvr = -np.inf
    for idx in order:
        if cvr[idx] > best_cvr:
            pareto_idx.append(idx)
            best_cvr = cvr[idx]
    pareto_idx = np.array(pareto_idx)
    pareto_idx = pareto_idx[np.argsort(ctr[pareto_idx])]

    # Plot dominated points
    dominated = np.setdiff1d(np.arange(n), pareto_idx)
    ax.scatter(ctr[dominated], cvr[dominated],
               s=42, color=MUTED, alpha=0.45, label="Dominated models",
               edgecolor="white", linewidth=0.6)

    # Plot Pareto-optimal points
    ax.scatter(ctr[pareto_idx], cvr[pareto_idx],
               s=85, color=PURPLE, alpha=0.95, label="Pareto-optimal",
               edgecolor="white", linewidth=0.8, zorder=5)
    ax.plot(ctr[pareto_idx], cvr[pareto_idx],
            color=PURPLE, lw=1.6, alpha=0.55, zorder=4)

    # Mark a few named anchors
    anchor_pts = [
        (ctr[pareto_idx][0], cvr[pareto_idx][0], "CVR-heavy", BLUE),
        (ctr[pareto_idx][len(pareto_idx) // 2],
         cvr[pareto_idx][len(pareto_idx) // 2], "Balanced", GREEN),
        (ctr[pareto_idx][-1], cvr[pareto_idx][-1], "CTR-heavy", AMBER),
    ]
    for x, y, label, col in anchor_pts:
        ax.scatter([x], [y], s=180, color=col, edgecolor="white",
                   linewidth=1.6, zorder=6)
        ax.annotate(label, (x, y), xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=10, fontweight="bold", color=col)

    ax.set_xlabel("CTR AUC", fontweight="bold")
    ax.set_ylabel("CVR AUC", fontweight="bold")
    ax.set_title("Pareto Frontier: CTR vs CVR Trade-off", pad=12)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_xlim(0.60, 0.84)
    ax.set_ylim(0.54, 0.82)

    # Annotation about frontier meaning
    ax.text(
        0.62,
        0.56,
        "Any point under the curve can be improved on both axes.\n"
        "Points on the curve trade one metric for the other.",
        fontsize=9,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT, edgecolor="#d1d5db"),
    )

    save_to_both(fig, "fig4_pareto_frontier.png")


# ---------------------------------------------------------------------------
# Figure 5: Loss balancing (Uncertainty Weighting & GradNorm).
# ---------------------------------------------------------------------------
def fig5_loss_balancing():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rng = np.random.default_rng(7)

    # ---- Left: Uncertainty Weighting -- per-task weights converge as sigma is learned.
    epochs = np.arange(0, 80)
    # Task A is easy (low uncertainty) -> weight grows; Task B is hard -> weight shrinks.
    sigma_a = 1.0 * np.exp(-epochs / 30) + 0.45
    sigma_b = 0.6 + 0.7 * (1 - np.exp(-epochs / 25))
    weight_a = 1.0 / (2 * sigma_a ** 2)
    weight_b = 1.0 / (2 * sigma_b ** 2)
    # Normalize for visualization
    total = weight_a + weight_b
    weight_a_n = weight_a / total
    weight_b_n = weight_b / total

    ax = axes[0]
    ax.plot(epochs, weight_a_n, color=BLUE, lw=2.2, label="Task A (CTR, easier)")
    ax.plot(epochs, weight_b_n, color=AMBER, lw=2.2, label="Task B (CVR, harder)")
    ax.fill_between(epochs, weight_a_n, weight_b_n, color=PURPLE, alpha=0.08)
    ax.set_xlabel("Training epoch", fontweight="bold")
    ax.set_ylabel("Normalized loss weight  1/(2σ²)", fontweight="bold")
    ax.set_title("Uncertainty Weighting: σ is learned", pad=10)
    ax.legend(loc="center right", framealpha=0.95)
    ax.set_ylim(0, 1)

    # ---- Right: GradNorm -- per-task gradient norms aligned to a common rate.
    epochs_g = np.arange(0, 80)
    # Without GradNorm: norms drift apart.
    base_a = 0.9 + 0.05 * np.sin(epochs_g / 6)
    base_b = 0.9 - 0.025 * epochs_g / 80 * 6 + 0.05 * np.cos(epochs_g / 7)
    raw_a = base_a + rng.normal(0, 0.03, len(epochs_g))
    raw_b = 0.4 + 0.02 * epochs_g + rng.normal(0, 0.03, len(epochs_g))

    # With GradNorm: weights adjusted so norms align around target.
    target = 1.0
    gn_a = target + rng.normal(0, 0.04, len(epochs_g))
    gn_b = target + rng.normal(0, 0.05, len(epochs_g))

    ax = axes[1]
    ax.plot(epochs_g, raw_a, color=BLUE, lw=1.4, alpha=0.55,
            label="Task A grad-norm (no balancing)")
    ax.plot(epochs_g, raw_b, color=AMBER, lw=1.4, alpha=0.55,
            label="Task B grad-norm (no balancing)")
    ax.plot(epochs_g, gn_a, color=BLUE, lw=2.2,
            label="Task A grad-norm (GradNorm)")
    ax.plot(epochs_g, gn_b, color=AMBER, lw=2.2,
            label="Task B grad-norm (GradNorm)")
    ax.axhline(target, color=GREEN, ls="--", lw=1.2, alpha=0.8,
               label="Common target rate")
    ax.set_xlabel("Training epoch", fontweight="bold")
    ax.set_ylabel("∥∇L_k∥ on shared params", fontweight="bold")
    ax.set_title("GradNorm: equalize the gradient pull", pad=10)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    save_to_both(fig, "fig5_loss_balancing.png")


# ---------------------------------------------------------------------------
# Figure 6: Gradient conflict illustration.
# ---------------------------------------------------------------------------
def fig6_gradient_conflict():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # ---- Left: 2D loss landscape with two task gradient arrows.
    ax = axes[0]
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    # Two anisotropic loss bowls shifted apart
    L_a = 0.6 * (grid_x + 1.0) ** 2 + 0.25 * (grid_y - 0.6) ** 2
    L_b = 0.25 * (grid_x - 1.2) ** 2 + 0.7 * (grid_y + 0.4) ** 2

    cs1 = ax.contour(grid_x, grid_y, L_a, levels=8,
                     colors=BLUE, alpha=0.55, linewidths=1.0)
    cs2 = ax.contour(grid_x, grid_y, L_b, levels=8,
                     colors=AMBER, alpha=0.55, linewidths=1.0)

    p = np.array([0.0, 0.0])
    # gradients (point downhill -> negative of gradient of L)
    g_a = np.array([-1.2, 0.3])  # task A wants to move left-up-ish
    g_b = np.array([1.0, -0.5])  # task B wants to move right-down

    # Joint update: sum of -grad
    g_sum = g_a + g_b

    ax.annotate("", xy=p + g_a, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4))
    ax.annotate("", xy=p + g_b, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.4))
    ax.annotate("", xy=p + g_sum, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.8))

    ax.scatter([p[0]], [p[1]], color=INK, s=70, zorder=5)

    ax.text(p[0] + g_a[0] - 0.05, p[1] + g_a[1] + 0.20,
            "−∇L_A  (Task A)", color=BLUE, fontsize=10, fontweight="bold",
            ha="right")
    ax.text(p[0] + g_b[0] + 0.05, p[1] + g_b[1] - 0.30,
            "−∇L_B  (Task B)", color=AMBER, fontsize=10, fontweight="bold")
    ax.text(p[0] + g_sum[0] + 0.10, p[1] + g_sum[1] - 0.10,
            "joint step", color=PURPLE, fontsize=10, fontweight="bold")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Shared parameter dim 1", fontweight="bold")
    ax.set_ylabel("Shared parameter dim 2", fontweight="bold")
    ax.set_title("Conflicting gradients on shared parameters", pad=10)
    ax.set_aspect("equal", adjustable="box")

    # ---- Right: cosine similarity of task gradients during training.
    rng = np.random.default_rng(11)
    epochs = np.arange(0, 100)
    cos_sim = 0.45 - 0.35 * np.tanh((epochs - 35) / 12) + rng.normal(0, 0.06, len(epochs))
    cos_sim = np.clip(cos_sim, -0.9, 0.9)

    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=1.0, ls="--", alpha=0.7)
    ax.fill_between(epochs, 0, cos_sim, where=(cos_sim >= 0),
                    color=GREEN, alpha=0.25, label="Aligned (cooperative)")
    ax.fill_between(epochs, 0, cos_sim, where=(cos_sim < 0),
                    color=AMBER, alpha=0.25, label="Conflicting")
    ax.plot(epochs, cos_sim, color=INK, lw=1.7)
    ax.set_xlabel("Training epoch", fontweight="bold")
    ax.set_ylabel("cos(∇L_A, ∇L_B)", fontweight="bold")
    ax.set_title("Tasks shift from aligned to conflicting", pad=10)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    save_to_both(fig, "fig6_gradient_conflict.png")


# ---------------------------------------------------------------------------
# Figure 7: ESMM architecture.
# ---------------------------------------------------------------------------
def fig7_esmm():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.set_title("ESMM: Train CVR on the Entire Impression Space", pad=14)

    # Input
    draw_box(ax, 5.5, 6.4, 3.0, 0.6, "Impression Features  x", BLUE, fontsize=10)

    # Shared embedding
    draw_box(ax, 5.5, 5.3, 4.0, 0.6, "Shared Embedding Layer", PURPLE,
             fontsize=10)
    draw_arrow(ax, 5.5, 6.1, 5.5, 5.6, color=MUTED)

    # Two towers
    draw_box(ax, 2.7, 3.8, 2.4, 0.6, "CTR Tower", GREEN, fontsize=10)
    draw_box(ax, 8.3, 3.8, 2.4, 0.6, "CVR Tower", AMBER, fontsize=10)
    draw_arrow(ax, 5.5, 5.0, 2.7, 4.1, color=MUTED)
    draw_arrow(ax, 5.5, 5.0, 8.3, 4.1, color=MUTED)

    # Outputs
    draw_box(ax, 2.7, 2.5, 2.4, 0.55,
             "pCTR = P(click | imp)", INK, fontsize=10)
    draw_box(ax, 8.3, 2.5, 2.4, 0.55,
             "pCVR = P(buy | click)", INK, fontsize=10)
    draw_arrow(ax, 2.7, 3.5, 2.7, 2.78, color=GREEN)
    draw_arrow(ax, 8.3, 3.5, 8.3, 2.78, color=AMBER)

    # Multiplication
    circle = plt.Circle((5.5, 1.4), 0.32, color=PURPLE, ec="white", lw=1.4)
    ax.add_patch(circle)
    ax.text(5.5, 1.4, "×", ha="center", va="center", color="white",
            fontsize=18, fontweight="bold")
    draw_arrow(ax, 2.7, 2.22, 5.18, 1.55, color=GREEN, lw=1.4)
    draw_arrow(ax, 8.3, 2.22, 5.82, 1.55, color=AMBER, lw=1.4)

    # CTCVR
    draw_box(ax, 5.5, 0.5, 4.6, 0.55,
             "pCTCVR = pCTR × pCVR  (trained on ALL impressions)",
             PURPLE, fontsize=10)
    draw_arrow(ax, 5.5, 1.08, 5.5, 0.78, color=PURPLE, lw=1.4)

    # Side annotations: where loss is computed
    ax.text(0.4, 3.8, "Loss on ALL\nimpressions",
            color=GREEN, fontsize=9, fontweight="bold", ha="left", va="center")
    ax.text(10.6, 3.8, "No direct loss\n(supervised via CTCVR)",
            color=AMBER, fontsize=9, fontweight="bold", ha="right", va="center")

    save_to_both(fig, "fig7_esmm.png")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_shared_bottom()
    fig2_mmoe()
    fig3_ple()
    fig4_pareto_frontier()
    fig5_loss_balancing()
    fig6_gradient_conflict()
    fig7_esmm()
    print("Generated 7 figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
