"""
Figure generation script for Transfer Learning Part 10: Continual Learning.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates a single teaching point and is built to be self-contained.

Figures:
    fig1_catastrophic_forgetting   Accuracy on each old task collapses as new
                                   tasks are learned by a vanilla SGD baseline.
                                   Compares baseline vs EWC vs Replay.
    fig2_ewc_penalty               Loss landscape view of Elastic Weight
                                   Consolidation: a quadratic well anchored at
                                   theta_A* with curvature given by the Fisher
                                   information pulls the optimiser toward a
                                   region jointly good for tasks A and B.
    fig3_replay_buffer             Experience replay pipeline: stream of new
                                   samples mixes with a fixed-size memory
                                   buffer (reservoir sampling) before the
                                   gradient step.
    fig4_cl_scenarios              Task-incremental vs class-incremental vs
                                   domain-incremental learning. Three rows of
                                   stylised batches make the differences
                                   concrete.
    fig5_lwf_distillation          Learning without Forgetting: the frozen old
                                   model produces soft targets that distill
                                   into the new model alongside the new-task
                                   cross-entropy loss.
    fig6_transfer_matrix           Forward and backward transfer visualised on
                                   a task accuracy matrix R[i, j] with the
                                   FWT/BWT diagonal regions highlighted.
    fig7_benchmarks                Average accuracy after T tasks on Permuted
                                   MNIST and Split CIFAR for SGD, EWC, A-GEM,
                                   ER, LwF and a Joint upper bound.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb #7c3aed #10b981 #f59e0b.

Usage:
    python3 scripts/figures/transfer-learning/10-continual.py

Output:
    Writes the same PNGs into BOTH the EN and ZH article asset folders so
    the markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good
C_AMBER = "#f59e0b"    # warning / highlight
C_RED = "#dc2626"      # bad
C_GRAY = "#94a3b8"
C_LIGHT = "#e2e8f0"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "en" / "transfer-learning"
    / "10-continual-learning"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "transfer-learning"
    / "10-持续学习"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to BOTH EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=10, fontweight="normal",
         text_color="white", alpha=1.0, rounding=0.05):
    ec = ec or fc
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.2, facecolor=fc, edgecolor=ec, alpha=alpha,
    )
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight=fontweight)


def _arrow(ax, xy_from, xy_to, color=C_DARK, lw=1.4, style="-|>",
           connection="arc3,rad=0"):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw, connectionstyle=connection,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: Catastrophic forgetting curves
# ---------------------------------------------------------------------------
def fig1_catastrophic_forgetting() -> None:
    """Per-task accuracy vs training stage for SGD vs EWC vs Replay."""
    rng = np.random.default_rng(0)

    T = 5                # number of tasks
    epochs_per_task = 20
    x = np.arange(T * epochs_per_task)

    def simulate(retain_strength: float, plasticity: float = 0.95) -> np.ndarray:
        """Return per-task accuracy curves with controllable forgetting."""
        # accuracy[task_id, step]
        acc = np.zeros((T, len(x)))
        for t in range(T):
            start = t * epochs_per_task
            for s, step in enumerate(x):
                if step < start:
                    acc[t, s] = 10.0  # untrained baseline (10 classes)
                elif step < start + epochs_per_task:
                    # learning curve on this task
                    progress = (step - start + 1) / epochs_per_task
                    target = 92 * plasticity
                    acc[t, s] = 10 + (target - 10) * (1 - np.exp(-3.5 * progress))
                else:
                    # forgetting after task t finished
                    elapsed = step - (start + epochs_per_task - 1)
                    peak = acc[t, start + epochs_per_task - 1]
                    decay = np.exp(-elapsed / (8.0 + 80.0 * retain_strength))
                    floor = 18 + 60 * retain_strength
                    acc[t, s] = floor + (peak - floor) * decay
        # add small noise
        acc += rng.normal(0, 0.6, acc.shape)
        return np.clip(acc, 0, 100)

    sgd = simulate(retain_strength=0.05)
    ewc = simulate(retain_strength=0.55)
    rep = simulate(retain_strength=0.78)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    titles = [
        "Naive SGD: catastrophic forgetting",
        "EWC: regularisation",
        "Experience Replay: rehearsal",
    ]
    data_list = [sgd, ewc, rep]
    edge_colors = [C_RED, C_AMBER, C_GREEN]
    task_colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, "#0ea5e9"]

    for ax, data, title, ec in zip(axes, data_list, titles, edge_colors):
        for t in range(T):
            ax.plot(x, data[t], color=task_colors[t], lw=2.0,
                    label=f"Task {t+1}")
        for t in range(1, T):
            ax.axvline(t * epochs_per_task, color=C_GRAY, lw=0.7,
                       linestyle=":", alpha=0.7)
        # annotate task training windows
        for t in range(T):
            ax.axvspan(t * epochs_per_task,
                       (t + 1) * epochs_per_task,
                       facecolor=task_colors[t], alpha=0.05)
        ax.set_xlabel("Training step (5 tasks x 20 epochs)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", color=ec)
        ax.set_xlim(0, T * epochs_per_task - 1)
        ax.set_ylim(0, 100)

        avg_final = data[:, -1].mean()
        ax.text(0.98, 0.04,
                f"Avg final acc:  {avg_final:.1f}%",
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10,
                color=ec, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=ec, lw=1.0,
                          boxstyle="round,pad=0.3"))

    axes[0].set_ylabel("Accuracy on each task (%)", fontsize=11)
    axes[-1].legend(loc="lower left", fontsize=9, ncol=1, framealpha=0.95)

    fig.suptitle("Catastrophic forgetting on a 5-task sequence",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_catastrophic_forgetting")


# ---------------------------------------------------------------------------
# Figure 2: EWC penalty as a quadratic well in parameter space
# ---------------------------------------------------------------------------
def fig2_ewc_penalty() -> None:
    """Contour view: task A loss + EWC penalty steers update toward joint optimum."""
    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    # Parameter grid
    g = np.linspace(-3.5, 3.5, 400)
    X, Y = np.meshgrid(g, g)

    # Task A optimum
    aA = np.array([-1.4, -0.6])
    # Anisotropic curvature -> Fisher matrix is large in x-direction, small in y
    FxA, FyA = 4.5, 0.6
    LA = 0.5 * (FxA * (X - aA[0]) ** 2 + FyA * (Y - aA[1]) ** 2)

    # Task B optimum
    aB = np.array([1.6, 1.4])
    FxB, FyB = 1.5, 1.5
    LB = 0.5 * (FxB * (X - aB[0]) ** 2 + FyB * (Y - aB[1]) ** 2)

    # Joint optimum balance (illustrative)
    joint = np.array([0.55, 0.7])

    # Plot task A iso-contours (background)
    ax.contour(X, Y, LA, levels=[0.5, 1.5, 3.0, 5.0, 8.0],
               colors=C_BLUE, linewidths=1.0, alpha=0.55)
    # Task B contours
    ax.contour(X, Y, LB, levels=[0.5, 1.5, 3.0, 5.0, 8.0],
               colors=C_PURPLE, linewidths=1.0, alpha=0.55)

    # Filled "EWC well": quadratic anchored at aA with Fisher curvature
    LEWC = LA  # the EWC penalty is exactly LA's quadratic form (Fisher)
    ax.contourf(X, Y, np.clip(LEWC, 0, 6), levels=12,
                cmap="Blues", alpha=0.18)

    # Mark optima
    ax.scatter(*aA, s=180, marker="*", color=C_BLUE, zorder=5,
               edgecolor="white", linewidth=1.5,
               label=r"$\theta_A^{\,*}$ (task A optimum)")
    ax.scatter(*aB, s=180, marker="*", color=C_PURPLE, zorder=5,
               edgecolor="white", linewidth=1.5,
               label=r"$\theta_B^{\,*}$ (task B optimum)")
    ax.scatter(*joint, s=200, marker="X", color=C_GREEN, zorder=6,
               edgecolor="white", linewidth=1.5,
               label="EWC solution (joint)")

    # Naive SGD trajectory: A* -> B*
    naive_path = np.array([
        aA, aA + 0.4 * (aB - aA), aA + 0.7 * (aB - aA), aB,
    ])
    ax.plot(naive_path[:, 0], naive_path[:, 1],
            color=C_RED, lw=2.0, linestyle="--", marker="o",
            markersize=5, alpha=0.9, label="Naive SGD path  (forgets A)")

    # EWC trajectory: A* -> joint
    ewc_path = np.array([
        aA,
        aA + 0.30 * (joint - aA),
        aA + 0.65 * (joint - aA),
        joint,
    ])
    ax.plot(ewc_path[:, 0], ewc_path[:, 1],
            color=C_GREEN, lw=2.4, marker="o", markersize=5,
            label="EWC path  (retains A)")

    # Anisotropy callouts
    ax.annotate("Fisher large\n(stiff direction)",
                xy=(aA[0] + 1.2, aA[1]), xytext=(aA[0] + 2.4, aA[1] - 1.5),
                fontsize=9, color=C_BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0))
    ax.annotate("Fisher small\n(soft direction)",
                xy=(aA[0], aA[1] + 1.0), xytext=(aA[0] - 2.2, aA[1] + 2.0),
                fontsize=9, color=C_BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0))

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"$\theta_1$", fontsize=12)
    ax.set_ylabel(r"$\theta_2$", fontsize=12)
    ax.set_title("EWC penalty: quadratic well around $\\theta_A^{*}$ with "
                 "Fisher curvature\n"
                 r"$\mathcal{L}_B(\theta) + \frac{\lambda}{2}\sum_i F_i\,"
                 r"(\theta_i - \theta^{*}_{A,i})^2$",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "fig2_ewc_penalty")


# ---------------------------------------------------------------------------
# Figure 3: Experience replay pipeline
# ---------------------------------------------------------------------------
def fig3_replay_buffer() -> None:
    """Stream of new samples + memory buffer -> joint mini-batch -> update."""
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Title
    ax.text(6.5, 6.15,
            "Experience Replay:  joint mini-batch from new stream + memory buffer",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C_DARK)

    # ---- New task stream ----
    ax.text(1.0, 5.2, "New task t", ha="center", fontsize=11,
            fontweight="bold", color=C_AMBER)
    new_samples = 6
    for i in range(new_samples):
        _box(ax, (0.25 + i * 0.30, 4.3), 0.26, 0.6, "", C_AMBER,
             alpha=0.85, rounding=0.04)
    ax.text(1.0, 4.0, "stream  $\\mathcal{D}_t$", ha="center",
            fontsize=9, color=C_DARK)

    # ---- Memory buffer ----
    _box(ax, (0.2, 1.6), 2.4, 1.7, "", C_LIGHT, ec=C_DARK,
         alpha=0.7, rounding=0.06)
    ax.text(1.4, 3.05, "Memory buffer  $\\mathcal{M}$  (size N)",
            ha="center", fontsize=10, fontweight="bold", color=C_DARK)
    # populated cells -- mix of past tasks
    past_colors = [C_BLUE, C_PURPLE, C_GREEN, C_BLUE, C_PURPLE,
                   C_GREEN, C_BLUE, C_PURPLE, C_GREEN, C_BLUE]
    for i, c in enumerate(past_colors):
        row, col = i // 5, i % 5
        _box(ax, (0.35 + col * 0.45, 1.85 + row * 0.50),
             0.40, 0.42, "", c, alpha=0.85, rounding=0.04)
    ax.text(1.4, 1.32, "reservoir / class-balanced sampling",
            ha="center", fontsize=8.5, style="italic", color=C_DARK)

    # ---- Sampling arrows ----
    _arrow(ax, (1.0, 4.25), (4.0, 3.9), color=C_AMBER, lw=2.0)
    _arrow(ax, (2.6, 2.5), (4.0, 3.2), color=C_BLUE, lw=2.0)
    ax.text(3.05, 4.15, "B_new", fontsize=8.5, color=C_AMBER,
            fontweight="bold")
    ax.text(3.0, 2.55, "B_mem", fontsize=8.5, color=C_BLUE,
            fontweight="bold")

    # ---- Joint mini-batch ----
    _box(ax, (4.0, 2.7), 2.4, 1.7, "", C_BG, ec=C_DARK,
         alpha=1.0, rounding=0.06)
    ax.text(5.2, 4.15, "joint mini-batch", ha="center", fontsize=10,
            fontweight="bold", color=C_DARK)
    mix = [C_AMBER, C_BLUE, C_AMBER, C_PURPLE, C_AMBER, C_GREEN,
           C_AMBER, C_BLUE, C_AMBER, C_PURPLE]
    for i, c in enumerate(mix):
        row, col = i // 5, i % 5
        _box(ax, (4.15 + col * 0.45, 2.95 + row * 0.40),
             0.40, 0.32, "", c, alpha=0.85, rounding=0.04)
    ax.text(5.2, 2.45,
            r"$\mathcal{L} = \mathcal{L}_{new}(B_{new}) + \alpha \mathcal{L}_{mem}(B_{mem})$",
            ha="center", fontsize=10, color=C_DARK)

    # ---- Model + gradient step ----
    _arrow(ax, (6.4, 3.55), (7.6, 3.55), color=C_DARK, lw=2.0)
    _box(ax, (7.6, 2.7), 2.0, 1.7, "Model\n$f_\\theta$",
         C_PURPLE, alpha=0.9, fontsize=11, fontweight="bold")
    _arrow(ax, (9.6, 3.55), (10.7, 3.55), color=C_DARK, lw=2.0)
    _box(ax, (10.7, 2.7), 2.0, 1.7,
         r"$\theta \leftarrow \theta - \eta \nabla \mathcal{L}$",
         C_GREEN, alpha=0.9, fontsize=10.5, fontweight="bold")

    # ---- Update buffer (write-back) ----
    _arrow(ax, (12.7, 2.7), (12.7, 1.0),
           color=C_GRAY, lw=1.4, style="-|>", connection="arc3,rad=0")
    _arrow(ax, (12.7, 1.0), (2.6, 1.0),
           color=C_GRAY, lw=1.4, style="-|>", connection="arc3,rad=0")
    ax.text(7.5, 0.7, "write back: reservoir-sample new examples into M",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    fig.tight_layout()
    _save(fig, "fig3_replay_buffer")


# ---------------------------------------------------------------------------
# Figure 4: Three CL scenarios
# ---------------------------------------------------------------------------
def fig4_cl_scenarios() -> None:
    """Task-IL vs Class-IL vs Domain-IL -- three rows of stylised batches."""
    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(6.5, 7.20,
            "Three continual-learning scenarios "
            "(van de Ven & Tolias, 2019)",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C_DARK)

    rows = [
        ("Task-IL",
         "task ID known at test time",
         "shared trunk + per-task head",
         C_BLUE,
         [("0/1", C_BLUE), ("2/3", C_PURPLE), ("4/5", C_GREEN), ("6/7", C_AMBER)]),
        ("Domain-IL",
         "same labels, shifted input distribution",
         "single head; test ID unknown",
         C_PURPLE,
         [("clean", C_BLUE), ("rotated", C_PURPLE), ("noisy", C_GREEN), ("blurred", C_AMBER)]),
        ("Class-IL",
         "new classes appear sequentially",
         "single growing head; test ID unknown",
         C_AMBER,
         [("0..1", C_BLUE), ("2..3", C_PURPLE), ("4..5", C_GREEN), ("6..7", C_AMBER)]),
    ]

    row_h = 1.7
    y0 = 5.2
    for r, (name, what, head, color, tasks) in enumerate(rows):
        y = y0 - r * (row_h + 0.25)
        # row label
        _box(ax, (0.1, y - 0.45), 2.3, 1.05, "", color,
             alpha=0.18, rounding=0.05)
        ax.text(1.25, y + 0.30, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(1.25, y - 0.05, what, ha="center", va="center",
                fontsize=8.5, color=C_DARK)
        ax.text(1.25, y - 0.30, head, ha="center", va="center",
                fontsize=8, style="italic", color=C_DARK)

        # Task batches
        for i, (label, c) in enumerate(tasks):
            xpos = 2.95 + i * 2.40
            _box(ax, (xpos, y - 0.45), 2.0, 1.05, "", c,
                 alpha=0.18, rounding=0.05, ec=c)
            ax.text(xpos + 1.0, y + 0.30,
                    f"Task {i+1}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=c)
            ax.text(xpos + 1.0, y - 0.05, label,
                    ha="center", va="center", fontsize=10, color=C_DARK)
            # arrow between tasks
            if i < len(tasks) - 1:
                _arrow(ax, (xpos + 2.05, y + 0.075),
                       (xpos + 2.32, y + 0.075),
                       color=C_GRAY, lw=1.0)

    # Difficulty bar at bottom
    ax.text(6.5, 0.55,
            "difficulty:  Task-IL  <  Domain-IL  <  Class-IL "
            "  (no test-time task ID is the hardest)",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_DARK, lw=1.0,
                      boxstyle="round,pad=0.4"))

    fig.tight_layout()
    _save(fig, "fig4_cl_scenarios")


# ---------------------------------------------------------------------------
# Figure 5: LwF distillation
# ---------------------------------------------------------------------------
def fig5_lwf_distillation() -> None:
    """Frozen old model produces soft targets; new model learns new task + KD."""
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6.5, 6.65,
            "Learning without Forgetting:  knowledge distillation from old model",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C_DARK)

    # Input
    _box(ax, (0.4, 2.7), 1.5, 1.4, "new-task\nbatch  $x$", C_AMBER,
         alpha=0.9, fontsize=10, fontweight="bold")

    # Old (frozen) model
    _box(ax, (2.5, 4.2), 2.6, 1.4, "Old model  $f_{old}$\n(frozen, snapshot)",
         C_GRAY, alpha=0.85, fontsize=10, fontweight="bold")
    # New model
    _box(ax, (2.5, 1.4), 2.6, 1.4, "New model  $f_{new}$\n(trainable)",
         C_PURPLE, alpha=0.9, fontsize=10, fontweight="bold")

    _arrow(ax, (1.9, 3.6), (2.5, 4.7), color=C_DARK, lw=1.6)
    _arrow(ax, (1.9, 3.3), (2.5, 2.1), color=C_DARK, lw=1.6)

    # Outputs
    _box(ax, (5.7, 4.2), 2.6, 1.4,
         "old soft targets\n$\\sigma(z^{old}/T)$",
         C_BLUE, alpha=0.85, fontsize=10, fontweight="bold")
    _box(ax, (5.7, 1.4), 2.6, 1.4,
         "new logits\n$z^{new}$",
         C_PURPLE, alpha=0.9, fontsize=10, fontweight="bold")

    _arrow(ax, (5.1, 4.9), (5.7, 4.9), color=C_DARK, lw=1.6)
    _arrow(ax, (5.1, 2.1), (5.7, 2.1), color=C_DARK, lw=1.6)

    # Losses
    _box(ax, (9.0, 4.2), 3.6, 1.4,
         r"$\mathcal{L}_{KD} = T^2\, \mathrm{KL}\!\left(\sigma(z^{old}/T)\,\Vert\,\sigma(z^{new}_{old\text{-}heads}/T)\right)$",
         C_BLUE, alpha=0.85, fontsize=9.5, fontweight="bold")
    _box(ax, (9.0, 1.4), 3.6, 1.4,
         r"$\mathcal{L}_{CE} = -\sum_c y_c \log \sigma(z^{new}_{new\text{-}heads})_c$",
         C_PURPLE, alpha=0.9, fontsize=9.5, fontweight="bold")

    _arrow(ax, (8.3, 4.9), (9.0, 4.9), color=C_DARK, lw=1.6)
    _arrow(ax, (8.3, 2.1), (9.0, 2.1), color=C_DARK, lw=1.6)

    # Total loss
    _box(ax, (4.8, -0.2), 4.0, 1.0,
         r"$\mathcal{L} = \mathcal{L}_{CE} + \alpha\, \mathcal{L}_{KD}$",
         C_GREEN, alpha=0.95, fontsize=12, fontweight="bold")

    _arrow(ax, (10.8, 4.2), (8.8, 0.8), color=C_BLUE, lw=1.4,
           connection="arc3,rad=-0.25")
    _arrow(ax, (10.8, 1.4), (8.8, 0.8), color=C_PURPLE, lw=1.4,
           connection="arc3,rad=0.25")

    # Note
    ax.text(6.5, -0.7,
            "no old data needed -- only the old model snapshot. "
            "Temperature T softens both distributions.",
            ha="center", va="center", fontsize=9.5, style="italic",
            color=C_DARK)

    fig.tight_layout()
    _save(fig, "fig5_lwf_distillation")


# ---------------------------------------------------------------------------
# Figure 6: Forward / Backward transfer matrix
# ---------------------------------------------------------------------------
def fig6_transfer_matrix() -> None:
    """R[i, j] heatmap with FWT (above diagonal) and BWT (below) regions."""
    rng = np.random.default_rng(7)
    T = 5

    # Construct a plausible R[i, j]: accuracy on task j after training task i
    R = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            if j > i:
                # Forward: zero/few-shot ability before training task j
                R[i, j] = 12 + rng.uniform(0, 12) + 2 * i
            elif j == i:
                R[i, j] = 88 + rng.uniform(-2, 4)
            else:
                # Backward: how well we still do task j after training task i
                forget = 4 * (i - j) + rng.uniform(-2, 6)
                R[i, j] = max(35, R[j, j] - forget)

    # Random baseline (untrained)
    b = np.full(T, 12.0)

    # BWT and FWT (Lopez-Paz & Ranzato 2017)
    BWT = np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)])
    FWT = np.mean([R[j - 1, j] - b[j] for j in range(1, T)])

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    # Heatmap
    im = ax.imshow(R, cmap="YlGnBu", vmin=10, vmax=95, aspect="equal")
    for i in range(T):
        for j in range(T):
            txt_color = "white" if R[i, j] > 55 else C_DARK
            ax.text(j, i, f"{R[i, j]:.0f}",
                    ha="center", va="center",
                    fontsize=11, color=txt_color, fontweight="bold")

    # Highlight diagonal
    for i in range(T):
        rect = Rectangle((i - 0.5, i - 0.5), 1, 1,
                         fill=False, edgecolor=C_DARK, lw=2.0)
        ax.add_patch(rect)

    # FWT region (above diagonal) outline
    for i in range(T):
        for j in range(i + 1, T):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor=C_GREEN, lw=1.5,
                             linestyle="--", alpha=0.85)
            ax.add_patch(rect)
    # BWT region (below diagonal) outline
    for i in range(T):
        for j in range(i):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor=C_AMBER, lw=1.5,
                             linestyle="--", alpha=0.85)
            ax.add_patch(rect)

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"Task {j+1}" for j in range(T)], fontsize=10)
    ax.set_yticklabels([f"after T{i+1}" for i in range(T)], fontsize=10)
    ax.set_xlabel("Evaluated task  $j$", fontsize=11)
    ax.set_ylabel("After training task  $i$", fontsize=11)

    ax.set_title("Transfer matrix  $R_{i,j}$  (% accuracy)\n"
                 "diagonal = on-task,  above = forward transfer,  "
                 "below = backward transfer",
                 fontsize=12, fontweight="bold", color=C_DARK)

    # Annotated metrics box
    txt = (f"BWT  =  $\\frac{{1}}{{T-1}}\\sum_j (R_{{T,j}} - R_{{j,j}})$  "
           f"=  {BWT:+.1f}%\n"
           f"FWT  =  $\\frac{{1}}{{T-1}}\\sum_j (R_{{j-1,j}} - b_j)$  "
           f"=  {FWT:+.1f}%\n"
           f"Avg  =  $\\frac{{1}}{{T}}\\sum_j R_{{T,j}}$  "
           f"=  {R[-1].mean():.1f}%")
    ax.text(1.30, 0.5, txt,
            transform=ax.transAxes,
            ha="left", va="center", fontsize=10.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_DARK, lw=1.0,
                      boxstyle="round,pad=0.6"))

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, shrink=0.7)
    fig.tight_layout()
    _save(fig, "fig6_transfer_matrix")


# ---------------------------------------------------------------------------
# Figure 7: CL benchmark comparison
# ---------------------------------------------------------------------------
def fig7_benchmarks() -> None:
    """Average accuracy on Permuted MNIST and Split CIFAR for several methods."""
    methods = ["SGD\n(naive)", "EWC", "LwF", "A-GEM", "ER\n(replay)", "Joint\n(upper bound)"]
    # Representative numbers from CL literature (illustrative, not exact).
    pmnist = np.array([42.0, 78.5, 70.2, 80.1, 86.4, 96.2])
    pmnist_std = np.array([4.5, 1.8, 2.2, 1.6, 1.2, 0.4])

    scifar = np.array([18.5, 41.8, 38.5, 50.3, 60.7, 78.5])
    scifar_std = np.array([3.2, 2.5, 2.8, 2.1, 1.7, 0.6])

    colors = [C_RED, C_BLUE, C_PURPLE, C_AMBER, C_GREEN, C_GRAY]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for ax, scores, stds, title in zip(
        axes,
        [pmnist, scifar],
        [pmnist_std, scifar_std],
        ["Permuted MNIST  (T = 10 tasks)",
         "Split CIFAR-100  (T = 10, 10 classes/task)"],
    ):
        x = np.arange(len(methods))
        bars = ax.bar(x, scores, yerr=stds, capsize=4,
                      color=colors, edgecolor=C_DARK, linewidth=1.0,
                      alpha=0.92)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.4,
                    f"{score:.1f}",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=C_DARK)
        # Mark Joint upper bound with a dashed line
        ax.axhline(scores[-1], color=C_GRAY, linestyle="--",
                   lw=1.2, alpha=0.7)
        ax.text(len(methods) - 0.5, scores[-1] + 1.4,
                "Joint upper bound",
                ha="right", va="bottom", fontsize=8.5,
                color=C_GRAY, style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9.5)
        ax.set_ylabel("Average accuracy after T tasks (%)",
                      fontsize=10.5)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=12, fontweight="bold", color=C_DARK)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Continual learning benchmarks  (illustrative numbers, single-head class-IL)",
        fontsize=13, fontweight="bold", color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig7_benchmarks")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 10 (Continual Learning) figures...")
    fig1_catastrophic_forgetting()
    fig2_ewc_penalty()
    fig3_replay_buffer()
    fig4_cl_scenarios()
    fig5_lwf_distillation()
    fig6_transfer_matrix()
    fig7_benchmarks()
    print("Done.")


if __name__ == "__main__":
    main()
