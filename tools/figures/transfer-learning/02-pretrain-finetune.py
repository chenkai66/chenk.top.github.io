"""
Figure generation script for Transfer Learning Part 02:
Pre-training and Fine-tuning.

Generates 7 figures shared by the EN and ZH articles. Each figure
isolates one pedagogical idea so the reader can connect the math, the
code, and the picture without ambiguity.

Figures:
    fig1_pretrain_finetune_pipeline   Two-stage pipeline diagram:
                                      unlabeled corpus -> self-supervised
                                      pre-training -> base model ->
                                      task-specific fine-tuning.
    fig2_loss_curves_comparison       Train and validation loss for two
                                      regimes (random init vs. pre-trained
                                      init) on a small target dataset:
                                      faster convergence + better minimum.
    fig3_layer_freezing_strategies    Stack of Transformer blocks with
                                      four freezing strategies side by
                                      side: full fine-tune, freeze
                                      bottom, gradual unfreeze, head only.
    fig4_linear_probe_vs_full         Accuracy as a function of target
                                      dataset size for linear probing
                                      vs. full fine-tuning -- the
                                      classic crossover.
    fig5_catastrophic_forgetting      Performance on the source task
                                      degrading while the target task
                                      improves during sequential
                                      fine-tuning, with EWC-style
                                      regularisation flattening the drop.
    fig6_lr_schedules                 Three learning-rate schedules
                                      (constant, warmup + linear decay,
                                      warmup + cosine decay) plus the
                                      discriminative per-layer LR profile.
    fig7_data_size_scaling            Target accuracy vs. dataset size
                                      on log-x axis for from-scratch,
                                      linear probe, full fine-tune, and
                                      LoRA -- pre-training shifts the
                                      curve up and to the left.

Usage:
    python3 scripts/figures/transfer-learning/02-pretrain-finetune.py

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
C_BLUE = COLORS["primary"]      # primary -- pre-trained / target
C_PURPLE = COLORS["accent"]    # secondary -- fine-tune / source
C_GREEN = COLORS["success"]     # accent / good outcome
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
    / "transfer-learning"
    / "02-pre-training-and-fine-tuning"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "transfer-learning"
    / "02-预训练与微调技术"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: pre-train then fine-tune pipeline
# ---------------------------------------------------------------------------
def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, label, color, fc=None, fontsize=10, weight="bold"):
        fc = fc if fc is not None else C_BG
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.6, edgecolor=color, facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=C_DARK, fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2, color=C_DARK, lw=1.8, label=None, lbl_dy=0.18):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=14, color=color, lw=lw)
        ax.add_patch(a)
        if label is not None:
            ax.text((x1 + x2) / 2, max(y1, y2) + lbl_dy, label,
                    ha="center", color=color, fontsize=9, style="italic")

    # Stage banners
    ax.add_patch(FancyBboxPatch(
        (0.2, 4.4), 5.0, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=C_BLUE, alpha=0.12,
    ))
    ax.text(2.7, 4.68, "Stage 1: Self-supervised pre-training",
            ha="center", color=C_BLUE, fontsize=11, weight="bold")

    ax.add_patch(FancyBboxPatch(
        (5.6, 4.4), 5.2, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=C_PURPLE, alpha=0.12,
    ))
    ax.text(8.2, 4.68, "Stage 2: Supervised fine-tuning",
            ha="center", color=C_PURPLE, fontsize=11, weight="bold")

    # Stage 1 boxes
    box(0.3, 2.6, 1.7, 1.3, "Unlabeled\ncorpus\n(TB scale)", C_BLUE)
    box(2.4, 2.6, 1.7, 1.3, "Pretext\ntask\n(MLM, SimCLR)", C_BLUE)
    box(4.5, 1.7, 1.6, 2.2, "Pre-trained\nbase model\n$\\theta_{\\mathrm{pre}}$",
        C_BLUE, fc="#dbeafe")

    # Stage 2 boxes
    box(6.5, 2.9, 1.6, 1.0, "Labeled task\ndata (small)", C_PURPLE)
    box(8.4, 2.9, 1.6, 1.0, "Fine-tuning\nloop", C_PURPLE)
    box(8.4, 0.9, 1.6, 1.4, "Task model\n$\\theta^{*}$", C_PURPLE,
        fc="#ede9fe")

    # Arrows stage 1
    arrow(2.0, 3.25, 2.4, 3.25, C_BLUE)
    arrow(4.1, 3.25, 4.5, 3.25, C_BLUE)

    # Bridge
    arrow(6.1, 2.8, 6.5, 3.2, C_DARK, label="initialise", lbl_dy=0.05)

    # Arrows stage 2
    arrow(8.1, 3.4, 8.4, 3.4, C_PURPLE)
    arrow(9.2, 2.9, 9.2, 2.3, C_PURPLE)

    # Lower captions
    ax.text(2.7, 2.3, "Cheap, abundant", ha="center", color=C_GRAY,
            fontsize=9, style="italic")
    ax.text(8.2, 0.55, "Cheap to train, deploy per task",
            ha="center", color=C_GRAY, fontsize=9, style="italic")

    # Bottom annotation
    ax.text(5.5, 0.15,
            "One expensive pre-training pass amortised across many "
            "downstream tasks.",
            ha="center", color=C_DARK, fontsize=10, style="italic")

    fig.suptitle("Pre-training then fine-tuning pipeline",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_pretrain_finetune_pipeline")


# ---------------------------------------------------------------------------
# Figure 2: loss curves -- from scratch vs. fine-tuning
# ---------------------------------------------------------------------------
def fig2_loss_curves() -> None:
    rng = np.random.default_rng(7)
    epochs = np.arange(1, 41)

    # Synthetic but plausible curves
    scratch_train = 2.30 * np.exp(-epochs / 22.0) + 0.55 + rng.normal(0, 0.018, epochs.size)
    scratch_val = 2.30 * np.exp(-epochs / 26.0) + 0.78 + rng.normal(0, 0.025, epochs.size)
    # mild overfit at the tail
    scratch_val[25:] += np.linspace(0, 0.06, epochs.size - 25)

    ft_train = 1.40 * np.exp(-epochs / 6.0) + 0.18 + rng.normal(0, 0.012, epochs.size)
    ft_val = 1.40 * np.exp(-epochs / 8.0) + 0.32 + rng.normal(0, 0.018, epochs.size)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(epochs, scratch_train, color=C_AMBER, lw=2.2,
            label="From scratch -- train")
    ax.plot(epochs, scratch_val, color=C_AMBER, lw=2.2, linestyle="--",
            label="From scratch -- val")
    ax.plot(epochs, ft_train, color=C_BLUE, lw=2.4,
            label="Fine-tune (pre-trained) -- train")
    ax.plot(epochs, ft_val, color=C_BLUE, lw=2.4, linestyle="--",
            label="Fine-tune (pre-trained) -- val")

    # Annotate: faster + lower
    ax.annotate(
        "Pre-training reaches a lower\nvalidation loss in ~5 epochs",
        xy=(6, ft_val[5]), xytext=(15, 1.55),
        fontsize=10, color=C_BLUE,
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2),
    )
    ax.annotate(
        "From-scratch model still\nclimbing after 40 epochs",
        xy=(35, scratch_val[34]), xytext=(20, 2.3),
        fontsize=10, color=C_AMBER,
        arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.2),
    )

    # Final-loss gap
    ax.hlines(ft_val[-1], epochs[0], epochs[-1], color=C_GREEN,
              linestyle=":", lw=1.2, alpha=0.7)
    ax.text(40.2, ft_val[-1], "  fine-tune floor",
            color=C_GREEN, fontsize=9, va="center")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss curves: from scratch vs. fine-tuning a pre-trained model",
                 fontsize=13, weight="bold")
    ax.set_xlim(1, 44)
    ax.set_ylim(0, 3.0)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    _save(fig, "fig2_loss_curves_comparison")


# ---------------------------------------------------------------------------
# Figure 3: layer freezing strategies
# ---------------------------------------------------------------------------
def fig3_freezing() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-0.4, 7.2)
    ax.axis("off")

    strategies = [
        ("Full fine-tune", [True] * 6),
        ("Freeze bottom\n(top-k unfreeze)", [False, False, False, True, True, True]),
        ("Gradual unfreeze\n(epoch by epoch)", "gradient"),
        ("Linear probe\n(head only)", [False] * 6),
    ]

    block_w = 1.8
    block_h = 0.78
    gap = 0.85

    layer_names = ["Embedding", "Layer 1", "Layer 2",
                   "Layer 3", "Layer 4", "Head"]

    for col, (title, mask) in enumerate(strategies):
        x = 0.4 + col * (block_w + gap)
        # column title
        ax.text(x + block_w / 2, 6.7, title, ha="center",
                color=C_DARK, fontsize=11, weight="bold")

        for i in range(6):
            y = 5.5 - i * (block_h + 0.12)
            if mask == "gradient":
                # interpolate freezing from bottom (frozen) to top (trainable)
                t = i / 5.0
                color = (1 - t) * np.array([148, 163, 184]) / 255 + \
                        t * np.array([37, 99, 235]) / 255
                fc = tuple(color)
                ec = C_DARK
                trainable = t > 0.4
                ec = C_BLUE if trainable else C_GRAY
            else:
                trainable = mask[i]
                fc = "#dbeafe" if trainable else C_LIGHT
                ec = C_BLUE if trainable else C_GRAY

            patch = FancyBboxPatch(
                (x, y), block_w, block_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.6, edgecolor=ec, facecolor=fc,
            )
            ax.add_patch(patch)
            ax.text(x + block_w / 2, y + block_h / 2,
                    layer_names[i], ha="center", va="center",
                    color=C_DARK, fontsize=9)

            if col == 0:
                # Lock / flame icon column
                pass

    # Legend swatches at bottom
    legend_items = [
        (C_BLUE, "#dbeafe", "Trainable (gradients flow)"),
        (C_GRAY, C_LIGHT, "Frozen (no update)"),
    ]
    lx = 1.4
    for ec, fc, label in legend_items:
        patch = FancyBboxPatch(
            (lx, -0.1), 0.4, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.4, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(lx + 0.55, 0.06, label, va="center", color=C_DARK, fontsize=10)
        lx += 4.2

    fig.suptitle("Layer freezing strategies for Transformer fine-tuning",
                 fontsize=13, weight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "fig3_layer_freezing_strategies")


# ---------------------------------------------------------------------------
# Figure 4: linear probing vs. full fine-tuning
# ---------------------------------------------------------------------------
def fig4_linear_vs_full() -> None:
    # Dataset size grid (per class) on log scale
    n = np.logspace(np.log10(8), np.log10(20000), 60)

    # Saturation curves
    def sigmoid_growth(x, lo, hi, mid, k):
        return lo + (hi - lo) / (1 + np.exp(-k * (np.log10(x) - np.log10(mid))))

    # Linear probe: fast off the line (frozen good features), saturates lower
    linear = sigmoid_growth(n, lo=0.45, hi=0.79, mid=80, k=2.2)
    # Full fine-tune: slower start (head has to settle), surpasses linear with
    # enough data, much higher ceiling
    full = sigmoid_growth(n, lo=0.32, hi=0.92, mid=350, k=1.8)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(n, linear, color=C_GREEN, lw=2.6, label="Linear probing (frozen backbone)")
    ax.plot(n, full, color=C_BLUE, lw=2.6, label="Full fine-tuning")
    ax.fill_between(n, linear, full, where=(full > linear),
                    color=C_BLUE, alpha=0.08)
    ax.fill_between(n, linear, full, where=(full < linear),
                    color=C_GREEN, alpha=0.08)

    # Crossover annotation
    cross_idx = np.argmin(np.abs(linear - full))
    ax.scatter([n[cross_idx]], [linear[cross_idx]], color=C_AMBER, s=80, zorder=5)
    ax.annotate(
        f"Crossover ~ {int(n[cross_idx])} samples/class",
        xy=(n[cross_idx], linear[cross_idx]),
        xytext=(n[cross_idx] * 3, linear[cross_idx] - 0.18),
        color=C_AMBER, fontsize=10, weight="bold",
        arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.2),
    )

    ax.text(15, 0.82, "Few labels ->\nlinear probe wins",
            color=C_GREEN, fontsize=10, style="italic")
    ax.text(3500, 0.55, "Many labels ->\nfull fine-tune wins",
            color=C_BLUE, fontsize=10, style="italic")

    ax.set_xscale("log")
    ax.set_xlabel("Target dataset size (samples per class, log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Linear probing vs. full fine-tuning across data regimes",
                 fontsize=13, weight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    _save(fig, "fig4_linear_probe_vs_full")


# ---------------------------------------------------------------------------
# Figure 5: catastrophic forgetting
# ---------------------------------------------------------------------------
def fig5_forgetting() -> None:
    steps = np.linspace(0, 1, 200)

    # Source task accuracy: starts high, drops as we fine-tune on target
    src_naive = 0.92 - 0.55 / (1 + np.exp(-12 * (steps - 0.35)))
    src_ewc = 0.92 - 0.18 / (1 + np.exp(-10 * (steps - 0.45)))
    # Target task accuracy: rises quickly
    tgt_naive = 0.18 + 0.74 / (1 + np.exp(-9 * (steps - 0.30)))
    tgt_ewc = 0.18 + 0.66 / (1 + np.exp(-8 * (steps - 0.35)))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    for ax, title, src, tgt, subtitle in [
        (axes[0], "Naive sequential fine-tuning", src_naive, tgt_naive,
         "Target ↑ but source collapses"),
        (axes[1], "With EWC / replay regularisation", src_ewc, tgt_ewc,
         "Source preserved, target almost as good"),
    ]:
        ax.plot(steps, src, color=C_RED, lw=2.6, label="Source-task accuracy")
        ax.plot(steps, tgt, color=C_BLUE, lw=2.6, label="Target-task accuracy")
        ax.fill_between(steps, src, 0.92, color=C_RED, alpha=0.08)
        ax.set_title(f"{title}\n", fontsize=12, weight="bold")
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
                color=C_GRAY, fontsize=10, style="italic")
        ax.set_xlabel("Fine-tuning progress")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="center right", framealpha=0.95)

    axes[0].set_ylabel("Accuracy")

    # Annotate the forgetting gap on the left subplot
    drop = src_naive[0] - src_naive[-1]
    axes[0].annotate(
        f"Forgetting\ngap = {drop:.0%}",
        xy=(1.0, src_naive[-1]), xytext=(0.55, 0.55),
        color=C_RED, fontsize=10, weight="bold",
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
    )

    fig.suptitle("Catastrophic forgetting during fine-tuning",
                 fontsize=13, weight="bold", y=1.06)
    fig.tight_layout()
    _save(fig, "fig5_catastrophic_forgetting")


# ---------------------------------------------------------------------------
# Figure 6: learning-rate schedules and discriminative LR
# ---------------------------------------------------------------------------
def fig6_lr_schedules() -> None:
    total_steps = 1000
    warmup = 100
    steps = np.arange(total_steps)
    eta_max = 2e-5

    constant = np.full(total_steps, eta_max)

    linear = np.where(
        steps < warmup,
        eta_max * steps / warmup,
        eta_max * np.maximum(0.0, 1.0 - (steps - warmup) / (total_steps - warmup)),
    )

    cosine = np.where(
        steps < warmup,
        eta_max * steps / warmup,
        0.5 * eta_max * (1 + np.cos(np.pi * (steps - warmup) / (total_steps - warmup))),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # --- Left: temporal schedules ---
    ax = axes[0]
    ax.plot(steps, constant * 1e5, color=C_GRAY, lw=2.0,
            linestyle=":", label="Constant")
    ax.plot(steps, linear * 1e5, color=C_AMBER, lw=2.4,
            label="Warmup + linear decay")
    ax.plot(steps, cosine * 1e5, color=C_BLUE, lw=2.4,
            label="Warmup + cosine decay")

    ax.axvspan(0, warmup, color=C_PURPLE, alpha=0.10)
    ax.text(warmup / 2, 2.05, "warmup", ha="center", color=C_PURPLE, fontsize=9)

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Learning rate ($\times 10^{-5}$)")
    ax.set_title("LR schedules for fine-tuning", fontsize=12, weight="bold")
    ax.set_ylim(0, 2.3)
    ax.legend(loc="upper right", framealpha=0.95)

    # --- Right: discriminative per-layer LR ---
    ax = axes[1]
    layers = np.arange(0, 13)  # embedding + 12 transformer blocks (top is rightmost)
    decay = 2.6
    top_lr = 2e-5
    per_layer = top_lr / (decay ** (layers.max() - layers))

    bars = ax.bar(layers, per_layer * 1e5,
                  color=[C_GRAY] * 1 + [C_BLUE] * 11 + [C_PURPLE],
                  edgecolor=C_DARK, linewidth=0.6)
    bars[0].set_color(C_GRAY)
    ax.set_yscale("log")
    ax.set_xlabel("Layer index (0 = embedding, 12 = top)")
    ax.set_ylabel(r"Per-layer LR ($\times 10^{-5}$, log)")
    ax.set_title("Discriminative LR (ULMFiT, $\\xi = 2.6$)",
                 fontsize=12, weight="bold")
    ax.set_xticks(layers)
    ax.text(0.0, per_layer[0] * 1e5 * 1.4, "tiny LR\nfor embeddings",
            color=C_GRAY, fontsize=9, ha="left")
    ax.text(12.0, per_layer[-1] * 1e5 * 1.05, "full LR\nat the head",
            color=C_PURPLE, fontsize=9, ha="right")

    fig.suptitle("Learning-rate strategies: smaller and smarter than pre-training",
                 fontsize=13, weight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "fig6_lr_schedules")


# ---------------------------------------------------------------------------
# Figure 7: performance vs. target dataset size (multi-strategy)
# ---------------------------------------------------------------------------
def fig7_data_size_scaling() -> None:
    n = np.logspace(np.log10(10), np.log10(100000), 80)

    def saturate(x, lo, hi, mid, k):
        return lo + (hi - lo) / (1 + np.exp(-k * (np.log10(x) - np.log10(mid))))

    scratch = saturate(n, lo=0.10, hi=0.88, mid=8000, k=1.7)
    linear_probe = saturate(n, lo=0.55, hi=0.81, mid=120, k=2.0)
    full_ft = saturate(n, lo=0.62, hi=0.93, mid=400, k=1.8)
    lora = saturate(n, lo=0.60, hi=0.91, mid=350, k=1.85)

    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    ax.plot(n, scratch, color=C_AMBER, lw=2.6, label="From scratch")
    ax.plot(n, linear_probe, color=C_GREEN, lw=2.4,
            label="Linear probing")
    ax.plot(n, lora, color=C_PURPLE, lw=2.4, linestyle="--",
            label="LoRA (r = 8)")
    ax.plot(n, full_ft, color=C_BLUE, lw=2.8, label="Full fine-tuning")

    # Highlight low-data win
    ax.axvspan(10, 200, color=C_GREEN, alpha=0.06)
    ax.text(35, 0.20, "low-data zone:\npre-training matters most",
            color=C_GREEN, fontsize=10, style="italic")

    # Pre-training "shift" arrow at fixed accuracy 0.80
    target = 0.80
    n_scratch = n[np.argmin(np.abs(scratch - target))]
    n_full = n[np.argmin(np.abs(full_ft - target))]
    ax.annotate(
        "",
        xy=(n_full, target), xytext=(n_scratch, target),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.6),
    )
    ax.text(np.sqrt(n_full * n_scratch), target + 0.025,
            f"~{int(n_scratch / max(n_full, 1)):d}x less data\nfor the same accuracy",
            ha="center", color=C_DARK, fontsize=9, weight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Target dataset size (labelled samples, log scale)")
    ax.set_ylabel("Target-task accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(10, 100000)
    ax.set_title("Performance vs. target dataset size across fine-tuning regimes",
                 fontsize=13, weight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    _save(fig, "fig7_data_size_scaling")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_pipeline()
    fig2_loss_curves()
    fig3_freezing()
    fig4_linear_vs_full()
    fig5_forgetting()
    fig6_lr_schedules()
    fig7_data_size_scaling()
    print("All 7 figures written to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
