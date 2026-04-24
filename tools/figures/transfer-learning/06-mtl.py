"""
Figure generation script for Transfer Learning Part 06: Multi-Task Learning.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates a single conceptual idea so the article can lean on visual
intuition rather than walls of text.

Figures:
    fig1_hard_vs_soft_sharing       Side-by-side architecture diagram
                                    comparing hard parameter sharing
                                    (one shared backbone, T heads) with
                                    soft parameter sharing (T parallel
                                    backbones tied by an L2 penalty).
    fig2_cross_stitch               Schematic of a cross-stitch unit
                                    showing how two task streams mix at
                                    each layer through learned alpha
                                    coefficients.
    fig3_gradient_conflict          2D vector plot of two task gradients
                                    with cosine similarity, the average
                                    direction, and the PCGrad-projected
                                    direction overlaid for comparison.
    fig4_gradnorm                   GradNorm dynamics: per-task loss
                                    curves, relative inverse training
                                    rates, and the resulting weight
                                    schedule converging to a balanced
                                    steady state.
    fig5_uncertainty_weighting      Kendall et al. uncertainty weighting:
                                    how learned sigma reshapes the loss
                                    landscape and prevents one task from
                                    dominating.
    fig6_mtl_vs_single_task         Bar chart comparing single-task and
                                    multi-task accuracy/error across
                                    three NYUv2-style benchmarks for
                                    Uniform / GradNorm / PCGrad / CAGrad.
    fig7_task_affinity              Task affinity heatmap (Taskonomy-
                                    style) plus a derived task grouping
                                    dendrogram showing which tasks belong
                                    in the same shared encoder.

Usage:
    python3 scripts/figures/transfer-learning/06-mtl.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram

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
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_BG = COLORS["bg"]
C_LIGHT = COLORS["grid"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "transfer-learning" / "06-multi-task-learning"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "transfer-learning" / "06-多任务学习"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    """Save a figure to both EN and ZH asset folders, then close it."""
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def box(ax, xy, w, h, label, color, *, text_color="white", fontsize=10, alpha=1.0):
    """Draw a rounded rectangle with a centered text label."""
    patch = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=color, edgecolor="none", alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label,
            ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight="bold")


def arrow(ax, p1, p2, color=C_DARK, lw=1.6, style="-|>", alpha=0.8):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=14,
        color=color, lw=lw, alpha=alpha,
    ))


# ---------------------------------------------------------------------------
# Figure 1: Hard vs Soft Parameter Sharing
# ---------------------------------------------------------------------------
def fig1_hard_vs_soft_sharing():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.0))

    # ----- Hard parameter sharing -----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Hard Parameter Sharing",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=12)

    # Shared backbone (single column of blocks)
    box(ax, (3.8, 1.0), 2.4, 1.0, "Input  x", C_GRAY, fontsize=10)
    backbone_blocks = [
        (2.6, "Shared Block 1"),
        (4.0, "Shared Block 2"),
        (5.4, "Shared Block 3"),
    ]
    for y, label in backbone_blocks:
        box(ax, (3.8, y), 2.4, 1.0, label, C_BLUE, fontsize=10)
    box(ax, (3.6, 6.8), 2.8, 0.8, "Shared Features  G_sh(x)",
        C_PURPLE, fontsize=9)

    # Three task heads, well above the shared features box
    head_xs = [0.5, 4.0, 7.5]
    head_colors = [C_GREEN, C_AMBER, C_PURPLE]
    head_labels = ["Detection", "Segmentation", "Depth"]
    for hx, hc, hl in zip(head_xs, head_colors, head_labels):
        box(ax, (hx, 9.2), 2.0, 0.8, f"Head: {hl}", hc, fontsize=9)
        arrow(ax, (5.0, 7.6), (hx + 1.0, 9.2), color=hc, lw=1.4)

    # Stacked arrows in backbone
    arrow(ax, (5.0, 2.0), (5.0, 2.6))
    arrow(ax, (5.0, 3.6), (5.0, 4.0))
    arrow(ax, (5.0, 5.0), (5.0, 5.4))
    arrow(ax, (5.0, 6.4), (5.0, 6.8))

    ax.text(5.0, 0.4,
            "One backbone, T heads.\nStrongest regularization, fewest parameters.",
            ha="center", va="center", fontsize=9.5, color=C_DARK,
            style="italic")

    # ----- Soft parameter sharing -----
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Soft Parameter Sharing",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=12)

    # Three parallel backbones
    col_xs = [1.0, 5.0, 9.0]
    col_labels = ["Task A backbone", "Task B backbone", "Task C backbone"]
    col_colors = [C_GREEN, C_AMBER, C_PURPLE]
    for cx, cl, cc in zip(col_xs, col_labels, col_colors):
        box(ax, (cx, 1.0), 2.2, 0.8, "Input x", C_GRAY, fontsize=9)
        for i, lbl in enumerate(["Block 1", "Block 2", "Block 3"]):
            box(ax, (cx, 2.2 + 1.3 * i), 2.2, 0.9, lbl, cc,
                alpha=0.85, fontsize=9)
            arrow(ax, (cx + 1.1, 2.0 + 1.3 * i),
                  (cx + 1.1, 2.2 + 1.3 * i), lw=1.0)
        box(ax, (cx, 6.4), 2.2, 0.7, f"Head {cl[5]}", cc, fontsize=9)
        arrow(ax, (cx + 1.1, 6.1), (cx + 1.1, 6.4), lw=1.0)
        ax.text(cx + 1.1, 7.6, cl, ha="center", va="bottom",
                fontsize=9, color=C_DARK)

    # L2 coupling lines between same-level blocks
    for i in range(3):
        y = 2.65 + 1.3 * i
        for j in range(2):
            x1 = col_xs[j] + 2.2
            x2 = col_xs[j + 1]
            ax.plot([x1, x2], [y, y], color=C_AMBER,
                    lw=1.2, ls="--", alpha=0.85)
            ax.text((x1 + x2) / 2, y + 0.15,
                    r"$\lambda\|\theta_i-\theta_j\|^2$",
                    ha="center", va="bottom", fontsize=8,
                    color=C_AMBER, fontweight="bold")

    ax.text(6.0, 0.4,
            "Independent backbones tied by an L2 penalty.\n"
            "Lower negative-transfer risk, more parameters.",
            ha="center", va="center", fontsize=9.5, color=C_DARK,
            style="italic")

    fig.suptitle("Two Ways to Share: Hard vs Soft Parameter Sharing",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    save(fig, "fig1_hard_vs_soft_sharing")


# ---------------------------------------------------------------------------
# Figure 2: Cross-Stitch Network
# ---------------------------------------------------------------------------
def fig2_cross_stitch():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Two parallel networks A (top) and B (bottom)
    layers = ["Conv 1", "Conv 2", "Conv 3", "FC"]
    n_layers = len(layers)
    x_centers = np.linspace(2.0, 12.0, n_layers)
    y_top, y_bot = 6.0, 2.0
    box_w, box_h = 1.4, 0.9

    for i, x in enumerate(x_centers):
        box(ax, (x - box_w / 2, y_top), box_w, box_h, layers[i],
            C_BLUE, fontsize=9)
        box(ax, (x - box_w / 2, y_bot), box_w, box_h, layers[i],
            C_PURPLE, fontsize=9)

    # Forward arrows along each row
    for i in range(n_layers - 1):
        arrow(ax, (x_centers[i] + box_w / 2, y_top + box_h / 2),
              (x_centers[i + 1] - box_w / 2, y_top + box_h / 2),
              color=C_BLUE, lw=1.5)
        arrow(ax, (x_centers[i] + box_w / 2, y_bot + box_h / 2),
              (x_centers[i + 1] - box_w / 2, y_bot + box_h / 2),
              color=C_PURPLE, lw=1.5)

    # Cross-stitch units between layers
    for i in range(n_layers - 1):
        cx = (x_centers[i] + x_centers[i + 1]) / 2
        cy = (y_top + y_bot + box_h) / 2
        circle = mpatches.Circle((cx, cy), 0.45, facecolor=C_AMBER,
                                 edgecolor=C_DARK, lw=1.2, zorder=5)
        ax.add_patch(circle)
        ax.text(cx, cy, r"$\alpha$", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=6)
        # Diagonal mixing arrows
        ax.plot([x_centers[i] + box_w / 2, cx],
                [y_top + box_h / 2, cy + 0.4],
                color=C_AMBER, lw=1.0, ls=":")
        ax.plot([x_centers[i] + box_w / 2, cx],
                [y_bot + box_h / 2, cy - 0.4],
                color=C_AMBER, lw=1.0, ls=":")
        ax.plot([cx, x_centers[i + 1] - box_w / 2],
                [cy + 0.4, y_top + box_h / 2],
                color=C_AMBER, lw=1.0, ls=":")
        ax.plot([cx, x_centers[i + 1] - box_w / 2],
                [cy - 0.4, y_bot + box_h / 2],
                color=C_AMBER, lw=1.0, ls=":")

    # Input/output labels (kept inside the visible area)
    ax.text(0.6, y_top + box_h / 2, "Task A\nInput",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C_BLUE)
    ax.text(0.6, y_bot + box_h / 2, "Task B\nInput",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C_PURPLE)
    ax.text(13.4, y_top + box_h / 2, "Output A",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C_BLUE)
    ax.text(13.4, y_bot + box_h / 2, "Output B",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C_PURPLE)

    # Cross-stitch math at the bottom (matplotlib mathtext: no \begin{pmatrix})
    ax.text(7.0, 0.75,
            r"$\tilde{x}_A^{\,l} = \alpha_{AA}\, x_A^{\,l} + \alpha_{AB}\, x_B^{\,l}"
            r"\qquad \tilde{x}_B^{\,l} = \alpha_{BA}\, x_A^{\,l} + \alpha_{BB}\, x_B^{\,l}$",
            ha="center", va="center", fontsize=11.5, color=C_DARK)
    ax.text(7.0, 0.05,
            "Learned mixing coefficients reveal how much each task borrows.",
            ha="center", va="center", fontsize=9.5,
            color=C_DARK, style="italic")

    ax.set_title("Cross-Stitch Networks: Per-Layer Information Exchange",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=12)
    fig.tight_layout()
    save(fig, "fig2_cross_stitch")


# ---------------------------------------------------------------------------
# Figure 3: Gradient Conflict + PCGrad projection
# ---------------------------------------------------------------------------
def fig3_gradient_conflict():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # ----- Left: cosine similarity intuition (vector plot) -----
    ax = axes[0]
    ax.set_xlim(-1.8, 2.4)
    ax.set_ylim(-2.0, 2.4)
    ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, lw=0.6, zorder=0)
    ax.axvline(0, color=C_GRAY, lw=0.6, zorder=0)

    g1 = np.array([1.6, 0.4])
    g2 = np.array([-1.0, 1.4])
    g_avg = (g1 + g2) / 2
    cos = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))

    def vec(ax, v, color, label, lw=2.5, ls="-", label_offset=(0.18, 0.18)):
        ax.add_patch(FancyArrowPatch(
            (0, 0), tuple(v), arrowstyle="-|>",
            mutation_scale=18, color=color, lw=lw, ls=ls, zorder=4))
        ax.text(v[0] + label_offset[0],
                v[1] + label_offset[1],
                label, color=color, fontsize=11, fontweight="bold")

    vec(ax, g1, C_BLUE, r"$g_1$  (Task 1)", label_offset=(0.05, -0.25))
    vec(ax, g2, C_PURPLE, r"$g_2$  (Task 2)", label_offset=(-0.55, 0.15))
    vec(ax, g_avg, C_AMBER, r"$\bar{g}=(g_1+g_2)/2$", lw=2.0, ls="--",
        label_offset=(0.10, 0.05))

    # PCGrad: project g1 onto normal plane of g2 when conflict
    proj = (np.dot(g1, g2) / np.dot(g2, g2)) * g2
    g1_pc = g1 - proj
    vec(ax, g1_pc, C_GREEN, r"$g_1^{\,PC}$ (PCGrad)", lw=2.0, ls="-",
        label_offset=(0.10, 0.20))

    ax.set_title(rf"Gradient Conflict  ($\cos\theta = {cos:+.2f}$)",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.50, -0.04,
            "Conflict means the naive average $\\bar{g}$\n"
            "barely helps either task. PCGrad removes\n"
            "the conflicting component before averaging.",
            transform=ax.transAxes, va="top", ha="center",
            fontsize=9.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_BG,
                      edgecolor=C_LIGHT))

    # ----- Right: cosine similarity histogram during training -----
    ax = axes[1]
    rng = np.random.default_rng(7)
    # Two clusters tuned so ~42% of updates have negative cosine.
    samples = np.concatenate([
        rng.normal(-0.30, 0.22, 1100),
        rng.normal(0.40, 0.30, 1500),
    ])
    samples = np.clip(samples, -1, 1)

    counts, bins, patches = ax.hist(samples, bins=40, color=C_GRAY,
                                    edgecolor="white")
    for c, p in zip(bins[:-1], patches):
        if c < 0:
            p.set_facecolor(C_AMBER)
        else:
            p.set_facecolor(C_GREEN)

    pct_conflict = (samples < 0).mean() * 100
    ax.axvline(0, color=C_DARK, lw=1.4, ls="--")
    ax.set_xlabel(r"$\cos(\nabla\mathcal{L}_i,\nabla\mathcal{L}_j)$",
                  fontsize=11)
    ax.set_ylabel("Update count", fontsize=11)
    ax.set_title(f"In Practice: {pct_conflict:.0f}% of Updates Conflict",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.text(-0.95, ax.get_ylim()[1] * 0.92,
            "conflicting", color=C_AMBER, fontsize=10, fontweight="bold")
    ax.text(0.95, ax.get_ylim()[1] * 0.92,
            "cooperative", color=C_GREEN, fontsize=10,
            fontweight="bold", ha="right")

    fig.suptitle("Task Gradients: When Do They Fight?",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig3_gradient_conflict")


# ---------------------------------------------------------------------------
# Figure 4: GradNorm dynamics
# ---------------------------------------------------------------------------
def fig4_gradnorm():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    rng = np.random.default_rng(2)
    epochs = np.arange(0, 60)

    # Simulated per-task losses with different scales and convergence rates
    L1 = 1.0 * np.exp(-0.05 * epochs) + 0.05 * rng.normal(size=epochs.size) * np.exp(-0.02 * epochs)
    L2 = 50 * np.exp(-0.02 * epochs) + 1.5 * rng.normal(size=epochs.size) * np.exp(-0.01 * epochs)
    L3 = 0.3 * np.exp(-0.08 * epochs) + 0.02 * rng.normal(size=epochs.size) * np.exp(-0.03 * epochs)
    Ls = [L1, L2, L3]
    names = ["Classification", "Regression (large scale)", "Auxiliary"]
    colors = [C_BLUE, C_PURPLE, C_GREEN]

    # Loss curves (log-scale)
    ax = axes[0]
    for L, n, c in zip(Ls, names, colors):
        ax.plot(epochs, np.maximum(L, 1e-3), color=c, lw=2, label=n)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (log)", fontsize=11)
    ax.set_title("Per-Task Loss (log scale)", fontsize=12,
                 fontweight="bold", color=C_DARK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Relative inverse training rates
    ax = axes[1]
    L0s = [L[0] for L in Ls]
    ratios = [L / L0 for L, L0 in zip(Ls, L0s)]
    ratio_avg = np.mean(np.stack(ratios), axis=0)
    rel_rates = [r / (ratio_avg + 1e-8) for r in ratios]
    for r, n, c in zip(rel_rates, names, colors):
        ax.plot(epochs, r, color=c, lw=2, label=n)
    ax.axhline(1, color=C_DARK, ls="--", lw=1)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(r"$\tilde{r}_t$  (inverse training rate)",
                  fontsize=11)
    ax.set_title(r"$\tilde{r}_t > 1$ means falling behind",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Resulting weight schedule (driven by GradNorm: w_t \propto rel_rate^alpha)
    ax = axes[2]
    alpha = 1.5
    raw = np.stack([np.power(np.maximum(r, 1e-6), alpha) for r in rel_rates])
    weights = raw / raw.sum(axis=0, keepdims=True) * len(Ls)
    # Smooth a bit
    kernel = np.ones(5) / 5
    weights = np.stack([np.convolve(w, kernel, mode="same") for w in weights])
    for w, n, c in zip(weights, names, colors):
        ax.plot(epochs, w, color=c, lw=2, label=n)
    ax.axhline(1, color=C_DARK, ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(r"GradNorm weight $w_t$", fontsize=11)
    ax.set_title("Weights re-balance automatically",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(r"GradNorm: Weights Track $\tilde{r}_t^{\alpha}$ to Equalize Gradient Magnitudes",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_gradnorm")


# ---------------------------------------------------------------------------
# Figure 5: Uncertainty Weighting (Kendall et al.)
# ---------------------------------------------------------------------------
def fig5_uncertainty_weighting():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ----- Left: weighted loss vs sigma for two tasks -----
    ax = axes[0]
    sigma = np.linspace(0.2, 4.0, 300)
    L_task = 2.0  # fixed task loss
    weighted = (1 / (2 * sigma ** 2)) * L_task + np.log(sigma)
    sigma_star = np.sqrt(L_task)
    ax.plot(sigma, weighted, color=C_BLUE, lw=2.4,
            label=r"$\frac{1}{2\sigma^2}\mathcal{L}+\log\sigma$")
    ax.plot(sigma, (1 / (2 * sigma ** 2)) * L_task,
            color=C_PURPLE, lw=1.8, ls="--",
            label=r"weighted loss only")
    ax.plot(sigma, np.log(sigma), color=C_AMBER, lw=1.8,
            ls=":", label=r"$\log\sigma$ regularizer")
    ax.axvline(sigma_star, color=C_GREEN, lw=1.6, ls="--")
    ax.scatter([sigma_star], [(1 / (2 * sigma_star ** 2)) * L_task + np.log(sigma_star)],
               color=C_GREEN, s=80, zorder=5)
    ax.text(sigma_star + 0.1, 1.8,
            rf"$\sigma^*=\sqrt{{\mathcal{{L}}}}\!\approx\!{sigma_star:.2f}$",
            color=C_GREEN, fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Task uncertainty  $\sigma$", fontsize=11)
    ax.set_ylabel("Loss components", fontsize=11)
    ax.set_title(r"$\log\sigma$ Stops $\sigma\!\to\!\infty$ Trivial Minimum",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-2, 8)

    # ----- Right: effective weights vs raw losses for three tasks -----
    ax = axes[1]
    tasks = ["Class.\n(σ≈0.6)", "Regression\n(σ≈3.0)", "Aux\n(σ≈1.0)"]
    raw = np.array([1.0, 50.0, 0.3])
    sigmas = np.array([0.6, 3.0, 1.0])
    eff_w = 1 / (2 * sigmas ** 2)
    eff_loss = eff_w * raw

    x = np.arange(len(tasks))
    w = 0.32
    bars1 = ax.bar(x - w / 2, raw, w, color=C_GRAY,
                   label="Raw loss", edgecolor="white")
    bars2 = ax.bar(x + w / 2, eff_loss, w,
                   color=[C_BLUE, C_PURPLE, C_GREEN],
                   label=r"After $\frac{1}{2\sigma^2}$ weighting",
                   edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Loss contribution (log)", fontsize=11)
    ax.set_title("Uncertainty Weighting Equalizes Contributions",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9, loc="upper right")

    for b, v in zip(bars1, raw):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f"{v:g}",
                ha="center", fontsize=8.5, color=C_DARK)
    for b, v in zip(bars2, eff_loss):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f"{v:.2f}",
                ha="center", fontsize=8.5, color=C_DARK)

    fig.suptitle(r"Kendall et al. (2018): Learn $\sigma_t$ Per Task, Reweight by $1/2\sigma_t^2$",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_uncertainty_weighting")


# ---------------------------------------------------------------------------
# Figure 6: MTL vs Single-Task Performance
# ---------------------------------------------------------------------------
def fig6_mtl_vs_single_task():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    methods = ["Single-task", "Uniform\nMTL", "Uncertainty",
               "GradNorm", "PCGrad", "CAGrad"]
    # Numbers loosely based on NYUv2 / Cityscapes literature
    seg_miou = [38.5, 40.2, 41.1, 41.6, 42.7, 43.2]
    depth_err = [0.65, 0.61, 0.60, 0.59, 0.58, 0.57]   # lower better
    normal_err = [25.5, 24.8, 24.4, 24.0, 23.6, 23.2]   # lower better

    x = np.arange(len(methods))
    w = 0.27
    colors_seg = C_BLUE
    colors_depth = C_PURPLE
    colors_normal = C_GREEN

    # ----- Left: side-by-side bars (different scales -> normalized improvements) -----
    ax = axes[0]
    base_seg, base_depth, base_normal = seg_miou[0], depth_err[0], normal_err[0]
    impr_seg = [(v - base_seg) / base_seg * 100 for v in seg_miou]
    impr_depth = [-(v - base_depth) / base_depth * 100 for v in depth_err]
    impr_normal = [-(v - base_normal) / base_normal * 100 for v in normal_err]

    ax.bar(x - w, impr_seg, w, color=colors_seg, label="Segmentation (mIoU↑)",
           edgecolor="white")
    ax.bar(x, impr_depth, w, color=colors_depth, label="Depth (err↓)",
           edgecolor="white")
    ax.bar(x + w, impr_normal, w, color=colors_normal,
           label="Normals (err↓)", edgecolor="white")
    ax.axhline(0, color=C_DARK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("% improvement vs single-task", fontsize=11)
    ax.set_title("Per-Task Improvement on NYUv2-style Benchmark",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9, loc="upper left")

    # ----- Right: parameter / inference cost comparison -----
    ax = axes[1]
    setups = ["3 separate\nResNet-50s", "Shared encoder\n+ 3 heads"]
    params = [75.0, 31.0]
    fwd_passes = [3, 1]
    xx = np.arange(len(setups))

    ax2 = ax.twinx()
    bars_p = ax.bar(xx - 0.18, params, 0.36, color=C_BLUE,
                    label="Params (M)", edgecolor="white")
    bars_f = ax2.bar(xx + 0.18, fwd_passes, 0.36, color=C_AMBER,
                     label="Forward passes", edgecolor="white")
    ax.set_xticks(xx)
    ax.set_xticklabels(setups, fontsize=10)
    ax.set_ylabel("Parameters (millions)", color=C_BLUE, fontsize=11)
    ax2.set_ylabel("Forward passes per inference",
                   color=C_AMBER, fontsize=11)
    ax.set_title("Compute & Parameter Efficiency",
                 fontsize=12, fontweight="bold", color=C_DARK)

    for b, v in zip(bars_p, params):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}M",
                ha="center", fontsize=10, color=C_BLUE, fontweight="bold")
    for b, v in zip(bars_f, fwd_passes):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v}x",
                 ha="center", fontsize=10, color=C_AMBER,
                 fontweight="bold")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc="upper right", fontsize=9)

    ax.set_ylim(0, max(params) * 1.25)
    ax2.set_ylim(0, 4)

    fig.suptitle("Multi-Task Learning Pays Off on Both Accuracy and Cost",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_mtl_vs_single_task")


# ---------------------------------------------------------------------------
# Figure 7: Task Affinity Heatmap + Grouping Dendrogram
# ---------------------------------------------------------------------------
def fig7_task_affinity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    tasks = ["Detect", "Segment", "Depth",
             "Normals", "Edges", "Pose", "Caption"]
    n = len(tasks)

    # Hand-crafted affinity matrix (symmetric, diag = 1)
    A = np.array([
        [1.00, 0.82, 0.55, 0.45, 0.78, 0.40, 0.25],
        [0.82, 1.00, 0.62, 0.50, 0.85, 0.42, 0.30],
        [0.55, 0.62, 1.00, 0.88, 0.50, 0.35, 0.10],
        [0.45, 0.50, 0.88, 1.00, 0.45, 0.30, 0.05],
        [0.78, 0.85, 0.50, 0.45, 1.00, 0.32, 0.20],
        [0.40, 0.42, 0.35, 0.30, 0.32, 1.00, 0.55],
        [0.25, 0.30, 0.10, 0.05, 0.20, 0.55, 1.00],
    ])

    # ----- Left: heatmap -----
    ax = axes[0]
    im = ax.imshow(A, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tasks, fontsize=10, rotation=35, ha="right")
    ax.set_yticklabels(tasks, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    color="white" if A[i, j] > 0.55 or A[i, j] < 0.30 else C_DARK,
                    fontsize=8.5, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Affinity (gradient cosine / transfer gain)",
                   fontsize=9)
    ax.set_title("Task Affinity Matrix",
                 fontsize=12, fontweight="bold", color=C_DARK)

    # ----- Right: hierarchical clustering dendrogram -----
    ax = axes[1]
    # Convert affinity to distance, run linkage
    D = 1 - A
    np.fill_diagonal(D, 0.0)
    # condensed form
    iu = np.triu_indices(n, k=1)
    cond = D[iu]
    Z = linkage(cond, method="average")

    den = dendrogram(Z, labels=tasks, ax=ax, leaf_rotation=35,
                     color_threshold=0.45,
                     above_threshold_color=C_GRAY)
    ax.set_title("Suggested Task Grouping (avg-linkage)",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_ylabel("Distance  (1 - affinity)", fontsize=11)
    for label in ax.get_xticklabels():
        label.set_fontsize(10)

    # Annotate group regions
    ax.axhline(0.45, color=C_AMBER, lw=1.2, ls="--")
    ax.text(0.02, 0.46, "split threshold", color=C_AMBER,
            transform=ax.get_yaxis_transform(),
            fontsize=9, ha="left", va="bottom", fontweight="bold")

    fig.suptitle("Measure First, Group Second: Don't Share Across Unrelated Tasks",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_task_affinity")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    print("Generating figures...")
    fig1_hard_vs_soft_sharing()
    print("  [1/7] hard vs soft parameter sharing")
    fig2_cross_stitch()
    print("  [2/7] cross-stitch network")
    fig3_gradient_conflict()
    print("  [3/7] gradient conflict + PCGrad projection")
    fig4_gradnorm()
    print("  [4/7] GradNorm dynamics")
    fig5_uncertainty_weighting()
    print("  [5/7] uncertainty weighting")
    fig6_mtl_vs_single_task()
    print("  [6/7] MTL vs single-task")
    fig7_task_affinity()
    print("  [7/7] task affinity + grouping")
    print("Done.")


if __name__ == "__main__":
    main()
