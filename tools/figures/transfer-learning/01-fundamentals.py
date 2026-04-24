"""
Figure generation script for Transfer Learning Part 01: Fundamentals.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly.

Figures:
    fig1_domain_shift          Source vs target domain in 2D feature space:
                               two well-separated Gaussian classes shifted +
                               rotated to visualise covariate shift.
    fig2_taxonomy              Taxonomy tree: inductive / transductive /
                               unsupervised transfer learning, with the
                               distinguishing condition shown on each branch.
    fig3_layer_transferability Layer-wise transferability of CNN features
                               (Yosinski et al. 2014 style): general -> specific
                               curve with frozen vs fine-tuned variants.
    fig4_negative_transfer     When source helps vs hurts: target accuracy as a
                               function of domain divergence, with the
                               positive / neutral / negative-transfer regions.
    fig5_backbone_new_head     Architectural diagram of pretrained backbone
                               with a new task head; frozen vs trainable
                               sections clearly labelled.
    fig6_data_efficiency       Target accuracy vs number of target labels for
                               from-scratch vs transfer; the data-efficiency
                               gap is the value proposition of TL.
    fig7_domain_adaptation     Domain adaptation problem setup: shared encoder
                               aligns labelled source and unlabelled target
                               into a common feature space.

Usage:
    python3 scripts/figures/transfer-learning/01-fundamentals.py

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
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good
C_AMBER = "#f59e0b"    # warning / highlight
C_RED = "#dc2626"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "transfer-learning" / "01-fundamentals-and-core-concepts"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "transfer-learning" / "01-基础与核心概念"


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
# Figure 1: Source vs target domain (covariate shift)
# ---------------------------------------------------------------------------
def fig1_domain_shift() -> None:
    """Two well-separated 2D classes: source clean, target rotated + shifted."""
    rng = np.random.default_rng(42)
    n = 250

    # Source: two Gaussians along the diagonal
    src_a = rng.normal(loc=[-2.0, -2.0], scale=0.55, size=(n, 2))
    src_b = rng.normal(loc=[2.0, 2.0], scale=0.55, size=(n, 2))

    # Target: rotated 45deg, shifted, slightly higher variance
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    tgt_a = (rng.normal(loc=[-1.2, -1.0], scale=0.7, size=(n, 2))) @ R.T + [0.6, -0.4]
    tgt_b = (rng.normal(loc=[1.2, 1.0], scale=0.7, size=(n, 2))) @ R.T + [0.6, -0.4]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)

    for ax, (a, b), title, sub in zip(
        axes,
        [(src_a, src_b), (tgt_a, tgt_b)],
        ["Source domain  $\\mathcal{D}_S$", "Target domain  $\\mathcal{D}_T$"],
        ["abundant labels  ·  $P_S(X)$", "scarce labels  ·  $P_T(X)$"],
    ):
        ax.scatter(a[:, 0], a[:, 1], s=18, color=C_BLUE, alpha=0.65,
                   edgecolor="white", linewidth=0.4, label="class 0")
        ax.scatter(b[:, 0], b[:, 1], s=18, color=C_AMBER, alpha=0.65,
                   edgecolor="white", linewidth=0.4, label="class 1")
        ax.set_title(title, fontsize=13, fontweight="bold", color=C_DARK)
        ax.text(0.5, -0.13, sub, ha="center", va="top",
                transform=ax.transAxes, fontsize=10, color=C_GRAY)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("feature 1")
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.set_aspect("equal")

    axes[0].set_ylabel("feature 2")

    # Big arrow between the two panels showing the shift
    fig.text(0.5, 0.5, "distribution shift\n$P_S(X) \\neq P_T(X)$",
             ha="center", va="center", fontsize=11, color=C_PURPLE,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor=C_PURPLE, linewidth=1.2))

    fig.suptitle("Domain shift: same task, different input distribution",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_domain_shift")


# ---------------------------------------------------------------------------
# Figure 2: Transfer learning taxonomy
# ---------------------------------------------------------------------------
def fig2_taxonomy() -> None:
    """Tree: TL splits by label availability into 3 categories."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=10, weight="bold"):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor=color, edgecolor=C_DARK, linewidth=1.1, alpha=0.95))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="white",
                wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                    lw=1.4, mutation_scale=14))

    # Root
    box(6, 7.2, 4.0, 0.9, "Transfer Learning", C_DARK, fontsize=13)

    # Three categories
    cats = [
        (2.0, 5.2, "Inductive", C_BLUE,
         "$\\mathcal{T}_S \\neq \\mathcal{T}_T$\ntarget has labels",
         "Pretrain + fine-tune\nMulti-task learning",
         "ImageNet -> medical CT"),
        (6.0, 5.2, "Transductive", C_PURPLE,
         "$\\mathcal{T}_S = \\mathcal{T}_T,$  $\\mathcal{D}_S \\neq \\mathcal{D}_T$\ntarget has no labels",
         "Domain adaptation\nSample reweighting",
         "GTA5 sim -> real driving"),
        (10.0, 5.2, "Unsupervised", C_GREEN,
         "no labels in either\ndomain",
         "Self-supervised\nDeep clustering",
         "MoCo / SimCLR"),
    ]

    for x, y, name, color, cond, methods, example in cats:
        box(x, y, 3.0, 0.85, name, color, fontsize=12)
        arrow(6, 6.7, x, 5.7)

        # Condition card
        ax.text(x, y - 0.95, cond, ha="center", va="top",
                fontsize=9.2, color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=color, linewidth=1.0))
        # Methods
        ax.text(x, y - 2.25, "Methods", ha="center", va="top",
                fontsize=9, color=C_GRAY, fontweight="bold")
        ax.text(x, y - 2.55, methods, ha="center", va="top",
                fontsize=9, color=C_DARK)
        # Example
        ax.text(x, y - 3.65, "Example", ha="center", va="top",
                fontsize=9, color=C_GRAY, fontweight="bold")
        ax.text(x, y - 3.95, example, ha="center", va="top",
                fontsize=9, color=color, style="italic")

    ax.set_title("Taxonomy: three flavours of transfer learning",
                 fontsize=14, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig2_taxonomy")


# ---------------------------------------------------------------------------
# Figure 3: Layer-wise transferability
# ---------------------------------------------------------------------------
def fig3_layer_transferability() -> None:
    """Yosinski-style: feature transferability degrades with layer depth."""
    layers = np.array([1, 2, 3, 4, 5, 6, 7])
    layer_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]

    # Baseline: target trained from scratch (flat reference)
    baseline = np.full_like(layers, 0.625, dtype=float)

    # Frozen transfer: drops sharply at deeper layers (specific features)
    frozen = np.array([0.640, 0.638, 0.625, 0.595, 0.555, 0.500, 0.450])
    # Fine-tuned transfer: stays high (recovers specificity)
    finetuned = np.array([0.642, 0.645, 0.648, 0.651, 0.654, 0.652, 0.648])

    fig, ax = plt.subplots(figsize=(11, 5.6))

    ax.plot(layers, baseline, "--", color=C_GRAY, lw=2, label="Target trained from scratch")
    ax.plot(layers, frozen, "o-", color=C_AMBER, lw=2.4, markersize=8,
            label="Transfer + frozen")
    ax.plot(layers, finetuned, "s-", color=C_BLUE, lw=2.4, markersize=8,
            label="Transfer + fine-tuned")

    # Shaded region: general vs specific
    ax.axvspan(0.5, 3.5, color=C_GREEN, alpha=0.07)
    ax.axvspan(3.5, 7.5, color=C_PURPLE, alpha=0.07)
    ax.text(2.0, 0.685, "general features\n(edges, textures)",
            ha="center", fontsize=10, color=C_GREEN, fontweight="bold")
    ax.text(5.5, 0.685, "specific features\n(semantic concepts)",
            ha="center", fontsize=10, color=C_PURPLE, fontweight="bold")

    ax.set_xticks(layers)
    ax.set_xticklabels(layer_names)
    ax.set_xlabel("layer at which the network is split", fontsize=11)
    ax.set_ylabel("target-domain accuracy", fontsize=11)
    ax.set_ylim(0.42, 0.71)
    ax.set_title("Feature transferability across CNN layers",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    fig.tight_layout()
    _save(fig, "fig3_layer_transferability")


# ---------------------------------------------------------------------------
# Figure 4: Negative transfer regions
# ---------------------------------------------------------------------------
def fig4_negative_transfer() -> None:
    """Target accuracy as a function of source-target divergence."""
    div = np.linspace(0, 1.0, 200)

    # From-scratch baseline
    baseline = np.full_like(div, 0.62)

    # Transfer accuracy: high when divergence small, drops as divergence grows
    transfer = 0.92 - 0.55 * div ** 1.3 - 0.05 * div

    fig, ax = plt.subplots(figsize=(11, 5.6))

    ax.plot(div, baseline, "--", color=C_GRAY, lw=2.2,
            label="Train from scratch (no transfer)")
    ax.plot(div, transfer, "-", color=C_BLUE, lw=2.6, label="Transfer learning")

    # Find crossover point
    cross = div[np.argmin(np.abs(transfer - baseline))]

    # Region shading
    ax.fill_between(div, baseline, transfer, where=transfer >= baseline,
                    color=C_GREEN, alpha=0.18, label="Positive transfer")
    ax.fill_between(div, baseline, transfer, where=transfer < baseline,
                    color=C_RED, alpha=0.18, label="Negative transfer")

    # Crossover marker
    ax.axvline(cross, color=C_DARK, linestyle=":", lw=1.2, alpha=0.7)
    ax.scatter([cross], [baseline[0]], color=C_DARK, s=60, zorder=5)
    ax.annotate(f"crossover  d $\\approx$ {cross:.2f}",
                xy=(cross, baseline[0]),
                xytext=(cross + 0.05, baseline[0] - 0.1),
                fontsize=10, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    # Region labels
    ax.text(0.18, 0.85, "source helps\n(positive transfer)",
            fontsize=11, color=C_GREEN, fontweight="bold", ha="center")
    ax.text(0.82, 0.30, "source hurts\n(negative transfer)",
            fontsize=11, color=C_RED, fontweight="bold", ha="center")

    ax.set_xlabel("source-target domain divergence  $d_{\\mathcal{H}\\Delta\\mathcal{H}}$", fontsize=11)
    ax.set_ylabel("target-domain accuracy", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.20, 1.0)
    ax.set_title("Negative transfer: when borrowing knowledge backfires",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(loc="lower left", frameon=True, fontsize=9.5, ncol=2)

    fig.tight_layout()
    _save(fig, "fig4_negative_transfer")


# ---------------------------------------------------------------------------
# Figure 5: Pretrained backbone + new head
# ---------------------------------------------------------------------------
def fig5_backbone_new_head() -> None:
    """Architectural diagram: frozen backbone + new trainable head."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Backbone blocks
    backbone_blocks = [
        ("Conv 1", "edges"),
        ("Conv 2", "textures"),
        ("Conv 3", "patterns"),
        ("Conv 4", "parts"),
        ("Conv 5", "objects"),
    ]

    x_start = 0.5
    block_w, block_h = 1.6, 2.2
    gap = 0.25
    y_center = 3.0

    # Frozen backbone
    for i, (name, sub) in enumerate(backbone_blocks):
        x = x_start + i * (block_w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, y_center - block_h / 2), block_w, block_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=C_BLUE, edgecolor=C_DARK, linewidth=1.0, alpha=0.85))
        ax.text(x + block_w / 2, y_center + 0.35, name,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="white")
        ax.text(x + block_w / 2, y_center - 0.35, sub,
                ha="center", va="center", fontsize=9, color="white",
                style="italic")
        # Lock icon for frozen
        ax.text(x + block_w / 2, y_center - 1.45, "[ frozen ]",
                ha="center", va="center", fontsize=8.5,
                color=C_GRAY, fontweight="bold")

    # New head
    head_x = x_start + 5 * (block_w + gap) + 0.4
    ax.add_patch(FancyBboxPatch(
        (head_x, y_center - block_h / 2), block_w + 0.4, block_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=C_AMBER, edgecolor=C_DARK, linewidth=1.0, alpha=0.95))
    ax.text(head_x + (block_w + 0.4) / 2, y_center + 0.35, "New Head",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white")
    ax.text(head_x + (block_w + 0.4) / 2, y_center - 0.35, "task-specific",
            ha="center", va="center", fontsize=9, color="white", style="italic")
    ax.text(head_x + (block_w + 0.4) / 2, y_center - 1.45, "[ trainable ]",
            ha="center", va="center", fontsize=8.5,
            color=C_AMBER, fontweight="bold")

    # Arrows between blocks
    for i in range(5):
        x1 = x_start + i * (block_w + gap) + block_w
        x2 = x_start + (i + 1) * (block_w + gap) if i < 4 else head_x
        ax.annotate("", xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle="->", color=C_DARK,
                                    lw=1.3, mutation_scale=12))

    # Input image
    ax.text(x_start - 0.3, y_center, "input\nimage", ha="right", va="center",
            fontsize=10, color=C_DARK, fontweight="bold")

    # Output prediction
    final_x = head_x + block_w + 0.4
    ax.text(final_x + 0.3, y_center, "prediction\n(new classes)",
            ha="left", va="center", fontsize=10, color=C_DARK,
            fontweight="bold")

    # Brackets / labels
    ax.text(x_start + 2.5 * (block_w + gap) - 0.3, y_center + 1.7,
            "Pretrained backbone  (transferred from source)",
            ha="center", fontsize=11, color=C_BLUE, fontweight="bold")
    ax.text(head_x + (block_w + 0.4) / 2, y_center + 1.7,
            "New head  (target task)",
            ha="center", fontsize=11, color=C_AMBER, fontweight="bold")

    # Bracket lines
    ax.plot([x_start, x_start + 5 * (block_w + gap) - gap],
            [y_center + 1.45, y_center + 1.45], color=C_BLUE, lw=1.3)
    ax.plot([head_x, head_x + block_w + 0.4],
            [y_center + 1.45, y_center + 1.45], color=C_AMBER, lw=1.3)

    ax.set_title("Pretrained backbone + new head: the canonical recipe",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig5_backbone_new_head")


# ---------------------------------------------------------------------------
# Figure 6: Data efficiency curves
# ---------------------------------------------------------------------------
def fig6_data_efficiency() -> None:
    """Target accuracy vs number of target labels: scratch vs transfer."""
    n_labels = np.array([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000])

    # From-scratch needs lots of data; saturates slowly
    scratch = 0.50 + 0.42 * (1 - np.exp(-n_labels / 3000))
    # Transfer starts much higher, saturates faster
    transfer = 0.74 + 0.20 * (1 - np.exp(-n_labels / 800))

    fig, ax = plt.subplots(figsize=(11, 5.8))

    ax.semilogx(n_labels, scratch, "o-", color=C_GRAY, lw=2.4,
                markersize=8, label="Train from scratch")
    ax.semilogx(n_labels, transfer, "s-", color=C_BLUE, lw=2.4,
                markersize=8, label="Transfer learning")

    ax.fill_between(n_labels, scratch, transfer,
                    where=transfer >= scratch,
                    color=C_GREEN, alpha=0.15, label="data-efficiency gap")

    # Annotate one specific operating point
    n_ref = 100
    s_ref = 0.50 + 0.42 * (1 - np.exp(-n_ref / 3000))
    t_ref = 0.74 + 0.20 * (1 - np.exp(-n_ref / 800))
    ax.scatter([n_ref], [s_ref], color=C_GRAY, s=80, zorder=5,
               edgecolor=C_DARK, linewidth=1)
    ax.scatter([n_ref], [t_ref], color=C_BLUE, s=80, zorder=5,
               edgecolor=C_DARK, linewidth=1)
    ax.annotate(f"+{(t_ref - s_ref) * 100:.0f} pts at 100 labels",
                xy=(n_ref, (s_ref + t_ref) / 2),
                xytext=(300, 0.60), fontsize=10, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1))

    # Equivalent-data annotation
    target_acc = 0.85
    n_scratch_equiv = -3000 * np.log(1 - (target_acc - 0.50) / 0.42)
    n_transfer_equiv = -800 * np.log(1 - (target_acc - 0.74) / 0.20)
    ax.axhline(target_acc, color=C_PURPLE, linestyle=":", lw=1.2, alpha=0.6)
    ax.text(11, target_acc + 0.005, f"to reach 85%: scratch needs ~{int(n_scratch_equiv):,}, "
            f"transfer needs ~{int(n_transfer_equiv):,}",
            fontsize=9.5, color=C_PURPLE, fontweight="bold")

    ax.set_xlabel("number of labelled target examples (log scale)", fontsize=11)
    ax.set_ylabel("target-domain accuracy", fontsize=11)
    ax.set_xlim(8, 12000)
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Performance gain from transfer learning (especially in low-data regimes)",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    fig.tight_layout()
    _save(fig, "fig6_data_efficiency")


# ---------------------------------------------------------------------------
# Figure 7: Domain adaptation problem setup
# ---------------------------------------------------------------------------
def fig7_domain_adaptation() -> None:
    """Shared encoder maps source + target into aligned feature space."""
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Source data box
    ax.add_patch(FancyBboxPatch(
        (0.4, 4.5), 2.6, 1.6,
        boxstyle="round,pad=0.04,rounding_size=0.15",
        facecolor=C_BLUE, edgecolor=C_DARK, linewidth=1.0, alpha=0.92))
    ax.text(1.7, 5.7, "Source data", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(1.7, 5.25, "$X_S, Y_S$", ha="center", va="center",
            fontsize=11, color="white")
    ax.text(1.7, 4.85, "labelled  ·  abundant", ha="center", va="center",
            fontsize=9, color="white", style="italic")

    # Target data box
    ax.add_patch(FancyBboxPatch(
        (0.4, 1.0), 2.6, 1.6,
        boxstyle="round,pad=0.04,rounding_size=0.15",
        facecolor=C_PURPLE, edgecolor=C_DARK, linewidth=1.0, alpha=0.92))
    ax.text(1.7, 2.2, "Target data", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(1.7, 1.75, "$X_T$  ·  $Y_T$ unknown", ha="center", va="center",
            fontsize=11, color="white")
    ax.text(1.7, 1.35, "unlabelled  ·  scarce", ha="center", va="center",
            fontsize=9, color="white", style="italic")

    # Shared encoder
    ax.add_patch(FancyBboxPatch(
        (4.3, 2.5), 2.6, 2.0,
        boxstyle="round,pad=0.04,rounding_size=0.15",
        facecolor=C_GREEN, edgecolor=C_DARK, linewidth=1.0, alpha=0.95))
    ax.text(5.6, 3.85, "Shared encoder", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white")
    ax.text(5.6, 3.4, "$\\phi(\\cdot)$", ha="center", va="center",
            fontsize=14, color="white")
    ax.text(5.6, 2.9, "weights tied", ha="center", va="center",
            fontsize=9, color="white", style="italic")

    # Arrows from data to encoder
    for y in [5.3, 1.8]:
        ax.annotate("", xy=(4.3, 3.5), xytext=(3.0, y),
                    arrowprops=dict(arrowstyle="->", color=C_GRAY,
                                    lw=1.4, mutation_scale=14,
                                    connectionstyle="arc3,rad=0.0"))

    # Aligned feature space (mini scatter)
    rng = np.random.default_rng(11)
    fx, fy = 9.5, 3.5
    radius = 1.4
    src_pts_x = fx + rng.normal(0, 0.55, 50)
    src_pts_y = fy + rng.normal(0, 0.55, 50)
    tgt_pts_x = fx + rng.normal(0, 0.55, 40)
    tgt_pts_y = fy + rng.normal(0, 0.55, 40)

    # Background circle
    circle = plt.Circle((fx, fy), radius * 1.15, color=C_LIGHT,
                        alpha=0.5, zorder=1)
    ax.add_patch(circle)

    ax.scatter(src_pts_x, src_pts_y, s=22, color=C_BLUE, alpha=0.75,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(tgt_pts_x, tgt_pts_y, s=22, color=C_PURPLE, alpha=0.75,
               edgecolor="white", linewidth=0.4, zorder=3)

    ax.text(fx, fy + radius * 1.4, "Aligned feature space",
            ha="center", fontsize=11, fontweight="bold", color=C_DARK)
    ax.text(fx, fy - radius * 1.45,
            "$P(\\phi(X_S)) \\approx P(\\phi(X_T))$",
            ha="center", fontsize=11, color=C_DARK)

    # Arrow from encoder to feature space
    ax.annotate("", xy=(fx - radius * 1.15, fy), xytext=(6.9, 3.5),
                arrowprops=dict(arrowstyle="->", color=C_DARK,
                                lw=1.6, mutation_scale=16))

    # Classifier
    ax.add_patch(FancyBboxPatch(
        (12.0, 2.9), 1.6, 1.2,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=C_AMBER, edgecolor=C_DARK, linewidth=1.0, alpha=0.95))
    ax.text(12.8, 3.7, "Classifier", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white")
    ax.text(12.8, 3.25, "trained on $Y_S$", ha="center", va="center",
            fontsize=8.5, color="white", style="italic")

    # Arrow to classifier
    ax.annotate("", xy=(12.0, 3.5), xytext=(fx + radius * 1.15, fy),
                arrowprops=dict(arrowstyle="->", color=C_DARK,
                                lw=1.4, mutation_scale=14))

    # Loss callout below
    ax.text(7, 0.5,
            "Training objective:   $\\mathcal{L} = \\mathcal{L}_{\\mathrm{task}}(X_S, Y_S) "
            "+ \\lambda \\cdot \\mathrm{MMD}^2(\\phi(X_S), \\phi(X_T))$",
            ha="center", va="center", fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=C_DARK, linewidth=1.0))

    # Legend dots
    ax.scatter([10.6], [6.4], color=C_BLUE, s=40)
    ax.text(10.8, 6.4, "source features", va="center", fontsize=9.5, color=C_DARK)
    ax.scatter([10.6], [6.0], color=C_PURPLE, s=40)
    ax.text(10.8, 6.0, "target features", va="center", fontsize=9.5, color=C_DARK)

    ax.set_title("Domain adaptation: align distributions in a shared latent space",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig7_domain_adaptation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Transfer Learning Part 01...")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")
    print()

    fig1_domain_shift()
    fig2_taxonomy()
    fig3_layer_transferability()
    fig4_negative_transfer()
    fig5_backbone_new_head()
    fig6_data_efficiency()
    fig7_domain_adaptation()

    print("\nDone. 7 figures written to both EN and ZH asset folders.")


if __name__ == "__main__":
    main()
