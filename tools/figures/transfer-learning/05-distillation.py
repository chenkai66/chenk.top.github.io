"""
Figure generation script for Transfer Learning Part 05:
Knowledge Distillation.

Generates 7 figures shared by the EN and ZH articles. Each figure
isolates a single pedagogical idea so the reader can connect the
math, the code, and the picture without ambiguity.

Figures:
    fig1_teacher_student          Teacher -> Student knowledge transfer
                                  diagram with combined loss decomposition
                                  (soft KL term + hard CE term).
    fig2_soft_vs_hard             Soft labels (teacher softmax) versus
                                  one-hot hard labels for a cat image:
                                  same top-1, very different supervision.
    fig3_temperature_effect       Effect of temperature T in {1, 4, 10}
                                  on softmax outputs of identical logits;
                                  shows how T reveals dark knowledge.
    fig4_feature_vs_response      Where the supervision signal enters the
                                  student: only logits (response) versus
                                  intermediate features and attention.
    fig5_compression_curve        Student accuracy vs. parameter count.
                                  Distillation lifts the accuracy/size
                                  Pareto frontier above scratch training.
    fig6_self_distillation        Born-Again Networks: same architecture,
                                  iterated generations, monotonically
                                  improving accuracy until saturation.
    fig7_distilbert_results       DistilBERT compression: 40% smaller,
                                  60% faster, retains ~97% of BERT-base
                                  GLUE score. Bar comparison.

Usage:
    python3 scripts/figures/transfer-learning/05-distillation.py

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
# Shared aesthetic style (chenk-site)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]     # primary -- teacher / soft target
C_PURPLE = COLORS["accent"]   # secondary -- student / response
C_GREEN = COLORS["success"]    # accent / success / distilled
C_AMBER = COLORS["warning"]    # warning / temperature / highlight
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_BG = COLORS["bg"]
C_RED = COLORS["danger"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "en" / "transfer-learning"
    / "05-knowledge-distillation"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "transfer-learning"
    / "05-知识蒸馏"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Teacher-Student architecture and combined loss
# ---------------------------------------------------------------------------
def fig1_teacher_student() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Input
    inp = FancyBboxPatch((0.3, 3.0), 1.4, 1.0,
                         boxstyle="round,pad=0.05",
                         facecolor="#e2e8f0", edgecolor=C_DARK, linewidth=1.2)
    ax.add_patch(inp)
    ax.text(1.0, 3.5, "Input x", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Teacher
    teacher = FancyBboxPatch((2.4, 4.6), 3.2, 1.6,
                             boxstyle="round,pad=0.08",
                             facecolor="#dbeafe", edgecolor=C_BLUE,
                             linewidth=2)
    ax.add_patch(teacher)
    ax.text(4.0, 5.7, "Teacher  T", ha="center", va="center",
            fontsize=13, fontweight="bold", color=C_BLUE)
    ax.text(4.0, 5.15, "large, frozen", ha="center", va="center",
            fontsize=9.5, color=C_DARK, style="italic")
    ax.text(4.0, 4.78, "logits  $z^T$", ha="center", va="center",
            fontsize=10, color=C_DARK)

    # Student
    student = FancyBboxPatch((2.4, 1.0), 3.2, 1.6,
                             boxstyle="round,pad=0.08",
                             facecolor="#ede9fe", edgecolor=C_PURPLE,
                             linewidth=2)
    ax.add_patch(student)
    ax.text(4.0, 2.1, "Student  S", ha="center", va="center",
            fontsize=13, fontweight="bold", color=C_PURPLE)
    ax.text(4.0, 1.55, "small, trainable", ha="center", va="center",
            fontsize=9.5, color=C_DARK, style="italic")
    ax.text(4.0, 1.18, "logits  $z^S$", ha="center", va="center",
            fontsize=10, color=C_DARK)

    # Input arrows
    ax.annotate("", xy=(2.4, 5.4), xytext=(1.7, 3.7),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=C_DARK))
    ax.annotate("", xy=(2.4, 1.8), xytext=(1.7, 3.3),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=C_DARK))

    # Soft target box
    soft = FancyBboxPatch((6.4, 4.6), 2.5, 1.6,
                          boxstyle="round,pad=0.06",
                          facecolor="#fef3c7", edgecolor=C_AMBER,
                          linewidth=1.6)
    ax.add_patch(soft)
    ax.text(7.65, 5.75, "Soft target", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_AMBER)
    ax.text(7.65, 5.25, r"$\sigma(z^T/\tau)$", ha="center",
            va="center", fontsize=12)
    ax.text(7.65, 4.82, "dark knowledge", ha="center", va="center",
            fontsize=9, style="italic", color=C_DARK)

    # Hard target box
    hard = FancyBboxPatch((6.4, 1.0), 2.5, 1.6,
                          boxstyle="round,pad=0.06",
                          facecolor="#fee2e2", edgecolor=C_RED,
                          linewidth=1.6)
    ax.add_patch(hard)
    ax.text(7.65, 2.15, "Hard target", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_RED)
    ax.text(7.65, 1.65, "one-hot  $y$", ha="center", va="center",
            fontsize=11)
    ax.text(7.65, 1.22, "ground truth", ha="center", va="center",
            fontsize=9, style="italic", color=C_DARK)

    # Arrows teacher -> soft, student -> both losses
    ax.annotate("", xy=(6.4, 5.4), xytext=(5.6, 5.4),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_BLUE))
    ax.annotate("", xy=(6.4, 1.8), xytext=(5.6, 1.8),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_PURPLE))
    # student also feeds upward to soft loss
    ax.annotate("", xy=(7.0, 4.6), xytext=(5.0, 2.6),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=C_PURPLE,
                                linestyle=":", alpha=0.7))

    # Combined loss
    loss = FancyBboxPatch((9.3, 2.7), 1.55, 1.6,
                          boxstyle="round,pad=0.06",
                          facecolor="#dcfce7", edgecolor=C_GREEN,
                          linewidth=2)
    ax.add_patch(loss)
    ax.text(10.07, 3.9, "Loss", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_GREEN)
    ax.text(10.07, 3.45, r"$\alpha\tau^2 \mathcal{L}_{\mathrm{KD}}$",
            ha="center", va="center", fontsize=10)
    ax.text(10.07, 3.05, r"$+(1-\alpha)\mathcal{L}_{\mathrm{CE}}$",
            ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(9.3, 3.6), xytext=(8.9, 5.4),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=C_AMBER))
    ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 1.8),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=C_RED))

    # Title & subtitle
    ax.text(5.5, 6.65, "Knowledge Distillation: Teacher  ->  Student",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C_DARK)
    ax.text(5.5, 0.35,
            "Student learns the teacher's full output distribution, "
            "not only the argmax label.",
            ha="center", va="center", fontsize=10, style="italic",
            color=C_GRAY)

    save(fig, "fig1_teacher_student")


# ---------------------------------------------------------------------------
# Figure 2: Soft labels vs hard labels
# ---------------------------------------------------------------------------
def fig2_soft_vs_hard() -> None:
    classes = ["cat", "tiger", "leopard", "dog", "fox", "car", "plane"]
    hard = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Teacher soft probabilities for a cat image
    soft = np.array([0.62, 0.14, 0.10, 0.07, 0.05, 0.012, 0.008])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    x = np.arange(len(classes))

    # Hard
    axes[0].bar(x, hard, color=C_RED, edgecolor=C_DARK, linewidth=0.8,
                alpha=0.85)
    axes[0].set_title("Hard label  (one-hot ground truth)",
                      fontsize=12, fontweight="bold", color=C_DARK)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=25, ha="right")
    axes[0].set_ylabel("Probability", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].text(3.0, 0.55, "entropy = 0\n(no inter-class info)",
                 ha="center", va="center", fontsize=10,
                 color=C_DARK, style="italic",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2",
                           edgecolor=C_RED, linewidth=1))

    # Soft
    bars = axes[1].bar(x, soft, color=C_BLUE, edgecolor=C_DARK,
                       linewidth=0.8, alpha=0.85)
    # Highlight the dark-knowledge bars
    for i in (1, 2, 3, 4):
        bars[i].set_color(C_AMBER)
        bars[i].set_alpha(0.95)
    axes[1].set_title("Soft label  (teacher softmax)",
                      fontsize=12, fontweight="bold", color=C_DARK)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(soft):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=8.5, color=C_DARK)
    axes[1].text(3.0, 0.55,
                 "tigers / leopards / dogs / foxes\nshare features with "
                 "cats\n--> dark knowledge",
                 ha="center", va="center", fontsize=10,
                 color=C_DARK, style="italic",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                           edgecolor=C_AMBER, linewidth=1))

    fig.suptitle("Same prediction, very different supervision",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_soft_vs_hard")


# ---------------------------------------------------------------------------
# Figure 3: Effect of temperature on the softmax distribution
# ---------------------------------------------------------------------------
def fig3_temperature_effect() -> None:
    classes = ["cat", "tiger", "leopard", "dog", "fox", "wolf", "rabbit",
               "car", "plane", "ship"]
    # Hand-picked logits resembling a real classifier output
    z = np.array([6.5, 4.2, 3.5, 2.8, 2.2, 1.9, 0.8, -1.5, -2.0, -2.4])
    temps = [1.0, 4.0, 10.0]
    colors = [C_BLUE, C_PURPLE, C_AMBER]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    x = np.arange(len(classes))

    for ax, T, c in zip(axes, temps, colors):
        zT = z / T
        p = np.exp(zT - zT.max())
        p = p / p.sum()
        ax.bar(x, p, color=c, edgecolor=C_DARK, linewidth=0.7, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(rf"$\tau = {T:g}$", fontsize=13, fontweight="bold",
                     color=c)
        ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
        ax.text(0.97, 0.93, f"entropy = {ent:.2f} nats",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=c, linewidth=1))
        # Annotate top-1
        ax.text(0, p[0] + 0.03, f"{p[0]:.2f}", ha="center", va="bottom",
                fontsize=8.5, color=C_DARK)

    axes[0].set_ylabel("Probability", fontsize=11)
    fig.suptitle(
        "Temperature $\\tau$ controls how much dark knowledge is revealed",
        fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    # Bottom note
    fig.text(0.5, -0.04,
             r"Higher $\tau$ flattens the distribution, exposing the "
             r"teacher's uncertainty over similar classes.",
             ha="center", fontsize=10.5, style="italic", color=C_GRAY)
    fig.tight_layout()
    save(fig, "fig3_temperature_effect")


# ---------------------------------------------------------------------------
# Figure 4: Feature distillation vs response distillation
# ---------------------------------------------------------------------------
def fig4_feature_vs_response() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def draw_stack(x0: float, y0: float, label: str, color: str,
                   widths) -> list:
        """Draw a vertical stack of conv-block boxes; return top-y of each."""
        ys = []
        y = y0
        for w in widths:
            box = FancyBboxPatch((x0 - w / 2, y), w, 0.55,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor=C_DARK,
                                 linewidth=0.9, alpha=0.85)
            ax.add_patch(box)
            ys.append(y + 0.55)
            y += 0.7
        ax.text(x0, y0 - 0.35, label, ha="center", va="top",
                fontsize=11, fontweight="bold", color=C_DARK)
        return ys

    # Teacher: 4 thick blocks, Student: 4 thinner blocks
    teacher_x = 3.0
    student_x = 8.5
    teacher_y0 = 1.5
    student_y0 = 1.5

    t_widths = [1.6, 1.4, 1.2, 1.0]
    s_widths = [1.0, 0.85, 0.7, 0.55]

    t_tops = draw_stack(teacher_x, teacher_y0, "Teacher (large)",
                        "#dbeafe", t_widths)
    s_tops = draw_stack(student_x, student_y0, "Student (small)",
                        "#ede9fe", s_widths)

    # Logit heads
    ax.scatter([teacher_x], [teacher_y0 + 4 * 0.7 + 0.2], s=180,
               color=C_BLUE, zorder=5)
    ax.text(teacher_x, teacher_y0 + 4 * 0.7 + 0.55, "$z^T$",
            ha="center", fontsize=12, color=C_BLUE)
    ax.scatter([student_x], [student_y0 + 4 * 0.7 + 0.2], s=180,
               color=C_PURPLE, zorder=5)
    ax.text(student_x, student_y0 + 4 * 0.7 + 0.55, "$z^S$",
            ha="center", fontsize=12, color=C_PURPLE)

    # Response distillation arrow (top, between logit heads)
    ax.annotate("",
                xy=(student_x - 0.3, teacher_y0 + 4 * 0.7 + 0.2),
                xytext=(teacher_x + 0.3, teacher_y0 + 4 * 0.7 + 0.2),
                arrowprops=dict(arrowstyle="->", lw=2.4, color=C_GREEN))
    ax.text((teacher_x + student_x) / 2, teacher_y0 + 4 * 0.7 + 0.6,
            "Response distillation", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C_GREEN)
    ax.text((teacher_x + student_x) / 2, teacher_y0 + 4 * 0.7 + 0.25,
            r"KL$(\sigma(z^S/\tau) \,\|\, \sigma(z^T/\tau))$",
            ha="center", va="top", fontsize=9.5, color=C_DARK,
            style="italic")

    # Feature distillation arrows (between intermediate blocks)
    for i, (yt, ys) in enumerate(zip(t_tops, s_tops)):
        if i == 3:
            continue  # skip top, already shown as response
        ax.annotate("",
                    xy=(student_x - 0.5, ys - 0.27),
                    xytext=(teacher_x + 0.8, yt - 0.27),
                    arrowprops=dict(arrowstyle="->", lw=1.6,
                                    color=C_AMBER, linestyle="--",
                                    alpha=0.85))
    ax.text((teacher_x + student_x) / 2, 1.1, "Feature distillation",
            ha="center", fontsize=11, fontweight="bold", color=C_AMBER)
    ax.text((teacher_x + student_x) / 2, 0.7,
            r"$\|W_r F^l_S - F^l_T\|^2$  +  attention maps",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")

    ax.text(6.0, 5.55, "Where the supervision signal enters",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)

    save(fig, "fig4_feature_vs_response")


# ---------------------------------------------------------------------------
# Figure 5: Compression curve (accuracy vs model size)
# ---------------------------------------------------------------------------
def fig5_compression_curve() -> None:
    # Made-up but representative numbers; reflect literature trends.
    # Params in millions, accuracy on a CIFAR/ImageNet-like task.
    sizes = np.array([1.0, 2.5, 5.0, 11.0, 21.5, 44.5])  # M params
    scratch = np.array([78.5, 84.2, 88.1, 91.0, 92.8, 93.5])
    distilled = np.array([83.1, 87.6, 90.4, 92.7, 93.8, 94.0])
    teacher_acc = 94.2
    teacher_size = 86.0  # M params

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(sizes, scratch, "-o", color=C_GRAY, linewidth=2.0,
            markersize=8, label="Trained from scratch")
    ax.plot(sizes, distilled, "-o", color=C_GREEN, linewidth=2.4,
            markersize=9, label="Distilled from teacher")
    # Shaded gain region
    ax.fill_between(sizes, scratch, distilled, color=C_GREEN, alpha=0.12)

    # Teacher
    ax.scatter([teacher_size], [teacher_acc], s=240, color=C_BLUE,
               edgecolor=C_DARK, linewidth=1.2, zorder=5,
               label=f"Teacher ({teacher_size:.0f}M)")
    ax.axhline(teacher_acc, color=C_BLUE, linestyle=":", linewidth=1.2,
               alpha=0.6)
    ax.text(teacher_size, teacher_acc + 0.25,
            f"  Teacher  {teacher_acc:.1f}%",
            ha="left", va="bottom", fontsize=10, color=C_BLUE,
            fontweight="bold")

    # Annotate a typical operating point
    idx = 2
    ax.annotate(f"+{distilled[idx] - scratch[idx]:.1f}% from KD",
                xy=(sizes[idx], distilled[idx]),
                xytext=(sizes[idx] + 4, distilled[idx] - 4.5),
                fontsize=10.5, color=C_GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.4))

    ax.set_xscale("log")
    ax.set_xlabel("Student parameters (millions, log scale)", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("Distillation lifts the accuracy / size Pareto frontier",
                 fontsize=13.5, fontweight="bold", color=C_DARK, pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_ylim(75, 95.5)

    save(fig, "fig5_compression_curve")


# ---------------------------------------------------------------------------
# Figure 6: Self-distillation (Born-Again Networks)
# ---------------------------------------------------------------------------
def fig6_self_distillation() -> None:
    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)

    # ---- Left: pipeline diagram ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    gens = ["Gen 1", "Gen 2", "Gen 3", "Gen 4"]
    centers_x = [1.4, 4.0, 6.6, 9.2]
    colors = [C_GRAY, C_BLUE, C_PURPLE, C_GREEN]
    accs = [74.3, 75.2, 75.4, 75.5]

    for cx, label, c, acc in zip(centers_x, gens, colors, accs):
        box = FancyBboxPatch((cx - 0.85, 2.3), 1.7, 1.4,
                             boxstyle="round,pad=0.06",
                             facecolor="white", edgecolor=c, linewidth=2)
        ax.add_patch(box)
        ax.text(cx, 3.3, label, ha="center", va="center", fontsize=11,
                fontweight="bold", color=c)
        ax.text(cx, 2.8, "same\narchitecture", ha="center", va="center",
                fontsize=8.5, style="italic", color=C_DARK)
        ax.text(cx, 1.85, f"{acc:.1f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color=c)

    for i in range(3):
        ax.annotate("",
                    xy=(centers_x[i + 1] - 0.9, 3.0),
                    xytext=(centers_x[i] + 0.9, 3.0),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color=C_AMBER))
        ax.text((centers_x[i] + centers_x[i + 1]) / 2, 3.35,
                "teacher", ha="center", va="bottom", fontsize=8.5,
                color=C_AMBER, fontweight="bold")

    ax.text(5.0, 5.3, "Born-Again Networks", ha="center",
            fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(5.0, 4.75,
            "each generation distills from the previous one",
            ha="center", fontsize=10, style="italic", color=C_GRAY)
    ax.text(5.0, 1.0,
            "Soft labels regularise; iterations form an implicit ensemble.",
            ha="center", fontsize=9.5, style="italic", color=C_DARK)

    # ---- Right: convergence curve ----
    ax2 = fig.add_subplot(gs[0, 1])
    g = np.arange(1, 7)
    acc_curve = np.array([74.3, 75.2, 75.4, 75.5, 75.5, 75.4])
    baseline = 74.3
    ax2.axhline(baseline, color=C_GRAY, linestyle="--", linewidth=1.4,
                label="Baseline (Gen 1)")
    ax2.plot(g, acc_curve, "-o", color=C_GREEN, linewidth=2.4,
             markersize=9, label="Self-distilled")
    for gi, a in zip(g, acc_curve):
        ax2.text(gi, a + 0.07, f"{a:.1f}", ha="center", va="bottom",
                 fontsize=9, color=C_DARK)
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("CIFAR-100 accuracy (%)", fontsize=11)
    ax2.set_title("Saturates after ~3 generations",
                  fontsize=12, fontweight="bold", color=C_DARK)
    ax2.set_xticks(g)
    ax2.set_ylim(73.8, 76.0)
    ax2.legend(loc="lower right", frameon=True, fontsize=9.5)

    save(fig, "fig6_self_distillation")


# ---------------------------------------------------------------------------
# Figure 7: DistilBERT compression results
# ---------------------------------------------------------------------------
def fig7_distilbert_results() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    models = ["BERT-base", "DistilBERT"]
    colors = [C_BLUE, C_GREEN]

    # 1) Parameters
    params = [110, 66]  # millions
    bars = axes[0].bar(models, params, color=colors, edgecolor=C_DARK,
                       linewidth=0.9, width=0.55)
    axes[0].set_title("Parameters (M)", fontsize=12, fontweight="bold",
                      color=C_DARK)
    axes[0].set_ylim(0, 130)
    for b, v in zip(bars, params):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 3, f"{v}M",
                     ha="center", va="bottom", fontsize=11,
                     fontweight="bold")
    axes[0].text(0.5, 0.92, "-40%", transform=axes[0].transAxes,
                 ha="center", fontsize=14, fontweight="bold", color=C_RED,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#fee2e2",
                           edgecolor=C_RED, linewidth=1))

    # 2) Inference latency (ms per query, batch=1, GPU)
    latency = [410, 250]
    bars = axes[1].bar(models, latency, color=colors, edgecolor=C_DARK,
                       linewidth=0.9, width=0.55)
    axes[1].set_title("Inference latency (ms)", fontsize=12,
                      fontweight="bold", color=C_DARK)
    axes[1].set_ylim(0, 480)
    for b, v in zip(bars, latency):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 12, f"{v} ms",
                     ha="center", va="bottom", fontsize=11,
                     fontweight="bold")
    axes[1].text(0.5, 0.92, "60% faster", transform=axes[1].transAxes,
                 ha="center", fontsize=12.5, fontweight="bold",
                 color=C_GREEN,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#dcfce7",
                           edgecolor=C_GREEN, linewidth=1))

    # 3) GLUE score (averaged across tasks)
    glue = [79.5, 77.0]
    bars = axes[2].bar(models, glue, color=colors, edgecolor=C_DARK,
                       linewidth=0.9, width=0.55)
    axes[2].set_title("GLUE score (avg)", fontsize=12, fontweight="bold",
                      color=C_DARK)
    axes[2].set_ylim(70, 84)
    for b, v in zip(bars, glue):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=11,
                     fontweight="bold")
    axes[2].text(0.5, 0.92,
                 f"retains {100 * glue[1] / glue[0]:.0f}%",
                 transform=axes[2].transAxes,
                 ha="center", fontsize=12.5, fontweight="bold",
                 color=C_BLUE,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#dbeafe",
                           edgecolor=C_BLUE, linewidth=1))

    fig.suptitle(
        "DistilBERT: 40% smaller, 60% faster, 97% of BERT-base GLUE score",
        fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_distilbert_results")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_teacher_student()
    fig2_soft_vs_hard()
    fig3_temperature_effect()
    fig4_feature_vs_response()
    fig5_compression_curve()
    fig6_self_distillation()
    fig7_distilbert_results()
    print(f"Saved 7 figures to:\n  EN: {EN_DIR}\n  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
