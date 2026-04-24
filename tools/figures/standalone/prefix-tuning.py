"""
Figure generation script for the standalone Prefix-Tuning paper review.

Generates 5 figures shared by the EN and ZH articles. Each figure
isolates one pedagogical idea so the reader can connect the math, the
code, and the picture without ambiguity.

Figures:
    fig1_architecture                 Prefix-Tuning architecture: frozen
                                      Transformer stack with learnable
                                      prefix K/V vectors injected into
                                      every attention layer.
    fig2_method_comparison            Side-by-side comparison of full
                                      fine-tuning, Prefix-Tuning, and
                                      prompt tuning -- what is trained,
                                      where it lives, and parameter cost.
    fig3_prefix_length_sweep          Task quality (BLEU / ROUGE proxy)
                                      and trainable-parameter cost as
                                      a function of prefix length L.
    fig4_kv_cache_prepend             How prefix K/V are prepended to
                                      the attention cache during
                                      autoregressive decoding -- a
                                      timeline view.
    fig5_gpt2_application             Two GPT-2 generation tasks
                                      (table-to-text on E2E, summarization
                                      on XSum): Prefix-Tuning vs full FT
                                      vs adapters at matched parameter
                                      budgets.

Usage:
    python3 scripts/figures/standalone/prefix-tuning.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders so the
    markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"      # primary -- prefix / learned
C_PURPLE = "#7c3aed"    # secondary -- attention / model interior
C_GREEN = "#10b981"     # accent -- frozen / efficient
C_AMBER = "#f59e0b"     # warning / highlight
C_RED = "#ef4444"
C_GRAY = "#94a3b8"
C_LIGHT = "#e5e7eb"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "prefix-tuning"
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "standalone"
    / "prefix-tuning-optimizing-continuous-prompts-for-generation"
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
# Figure 1: Prefix-Tuning architecture
# ---------------------------------------------------------------------------
def fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(
        5.5, 5.95,
        "Prefix-Tuning: learnable K/V prefixes are injected into every frozen attention layer",
        ha="center", fontsize=12, weight="bold", color=C_DARK,
    )

    # Frozen Transformer stack on the right
    n_layers = 4
    stack_x, stack_w = 5.6, 4.0
    layer_h = 0.78
    base_y = 0.8
    gap = 0.18
    for i in range(n_layers):
        y = base_y + i * (layer_h + gap)
        patch = FancyBboxPatch(
            (stack_x, y), stack_w, layer_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=C_GREEN, facecolor="#ecfdf5",
        )
        ax.add_patch(patch)
        # Sub-blocks: attention | MLP
        ax.add_patch(Rectangle(
            (stack_x + 0.15, y + 0.12), 1.7, layer_h - 0.24,
            facecolor=C_PURPLE, alpha=0.18, edgecolor=C_PURPLE, linewidth=1.0,
        ))
        ax.text(stack_x + 1.0, y + layer_h / 2, "Self-Attn",
                ha="center", va="center", fontsize=9, color=C_PURPLE, weight="bold")
        ax.add_patch(Rectangle(
            (stack_x + 2.05, y + 0.12), 1.75, layer_h - 0.24,
            facecolor=C_GRAY, alpha=0.18, edgecolor=C_GRAY, linewidth=1.0,
        ))
        ax.text(stack_x + 2.92, y + layer_h / 2, "Feed-Forward",
                ha="center", va="center", fontsize=9, color="#475569", weight="bold")
        ax.text(stack_x - 0.18, y + layer_h / 2, f"L{i+1}",
                ha="right", va="center", fontsize=9, color=C_DARK)

    # Frozen banner
    ax.text(stack_x + stack_w / 2, base_y + n_layers * (layer_h + gap) + 0.15,
            "Frozen pretrained Transformer (theta unchanged)",
            ha="center", fontsize=10, color=C_GREEN, weight="bold")

    # Learnable prefix tower on the left
    pre_x, pre_w = 0.55, 1.55
    for i in range(n_layers):
        y = base_y + i * (layer_h + gap)
        patch = FancyBboxPatch(
            (pre_x, y), pre_w, layer_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6, edgecolor=C_BLUE, facecolor="#dbeafe",
        )
        ax.add_patch(patch)
        ax.text(pre_x + pre_w / 2, y + layer_h / 2,
                f"$P^{{(K)}}_{i+1}, P^{{(V)}}_{i+1}$",
                ha="center", va="center", fontsize=10, color=C_BLUE, weight="bold")

        # Arrow from prefix to that layer's attention sub-block
        a = FancyArrowPatch(
            (pre_x + pre_w + 0.02, y + layer_h / 2),
            (stack_x + 0.13, y + layer_h / 2),
            arrowstyle="-|>", mutation_scale=12, color=C_BLUE, lw=1.4,
        )
        ax.add_patch(a)

    ax.text(pre_x + pre_w / 2, base_y + n_layers * (layer_h + gap) + 0.15,
            "Learnable prefix matrix $P_\\theta$",
            ha="center", fontsize=10, color=C_BLUE, weight="bold")

    # Reparam MLP block above the prefix tower
    mlp_y = base_y - 0.55
    ax.add_patch(FancyBboxPatch(
        (pre_x, mlp_y - 0.55), pre_w, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=C_AMBER, facecolor="#fef3c7",
    ))
    ax.text(pre_x + pre_w / 2, mlp_y - 0.30,
            "MLP$_\\phi$ reparam.", ha="center", va="center",
            fontsize=9, color="#b45309", weight="bold")
    a = FancyArrowPatch(
        (pre_x + pre_w / 2, mlp_y - 0.05),
        (pre_x + pre_w / 2, base_y - 0.02),
        arrowstyle="-|>", mutation_scale=12, color=C_AMBER, lw=1.4,
    )
    ax.add_patch(a)

    # Input tokens at bottom right
    ax.text(stack_x + 0.05, 0.5, "Input tokens $x_1, x_2, \\ldots, x_n$",
            ha="left", va="center", fontsize=10, color=C_DARK)
    a = FancyArrowPatch(
        (stack_x + stack_w / 2, 0.32),
        (stack_x + stack_w / 2, base_y - 0.05),
        arrowstyle="-|>", mutation_scale=12, color=C_DARK, lw=1.2,
    )
    ax.add_patch(a)

    # Output arrow
    top_y = base_y + n_layers * (layer_h + gap) - gap
    a = FancyArrowPatch(
        (stack_x + stack_w / 2, top_y + 0.05),
        (stack_x + stack_w / 2, top_y + 0.55),
        arrowstyle="-|>", mutation_scale=12, color=C_DARK, lw=1.2,
    )
    ax.add_patch(a)
    ax.text(stack_x + stack_w / 2, top_y + 0.75,
            "Generated tokens $y_1, y_2, \\ldots$",
            ha="center", va="center", fontsize=10, color=C_DARK, weight="bold")

    # Caption strip
    ax.text(
        5.5, 0.05,
        "Only $P_\\theta$ (and the optional reparam MLP) is updated. "
        "At each layer the prefix is concatenated with the real K/V before attention.",
        ha="center", fontsize=9, color="#475569", style="italic",
    )

    _save(fig, "fig1_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Method comparison -- full FT vs Prefix-Tuning vs Prompt Tuning
# ---------------------------------------------------------------------------
def fig2_method_comparison() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.4))
    fig.suptitle(
        "Three ways to adapt a frozen language model",
        fontsize=13, weight="bold", color=C_DARK, y=1.02,
    )

    titles = [
        "Full fine-tuning",
        "Prefix-Tuning",
        "Prompt tuning (soft prompts)",
    ]
    subtitles = [
        "Update every weight of every layer",
        "Inject learnable K/V at every layer",
        "Prepend learnable embeddings at input only",
    ]
    accents = [C_RED, C_BLUE, C_AMBER]
    param_text = [
        "trainable: 100% of $\\theta$\n(e.g. 1.5B for GPT-2 XL)",
        "trainable: $\\sim$0.1% of $\\theta$\n($2 L H \\times L_{\\text{prefix}}$)",
        "trainable: $\\sim$0.01% of $\\theta$\n($H \\times L_{\\text{prefix}}$)",
    ]

    for ax, title, subtitle, color, ptxt in zip(axes, titles, subtitles, accents, param_text):
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 6.2)
        ax.axis("off")
        ax.set_title(title, fontsize=11.5, weight="bold", color=color, pad=8)
        ax.text(2.5, 5.85, subtitle, ha="center", fontsize=9.5,
                color="#475569", style="italic")

        # Stack of 4 layers
        n = 4
        layer_h = 0.7
        gap = 0.15
        base = 1.4
        for i in range(n):
            y = base + i * (layer_h + gap)
            ax.add_patch(FancyBboxPatch(
                (0.9, y), 3.2, layer_h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.3,
                edgecolor=color if title == "Full fine-tuning" else C_GREEN,
                facecolor="#fee2e2" if title == "Full fine-tuning" else "#ecfdf5",
            ))
            ax.text(2.5, y + layer_h / 2, f"Layer {i+1}",
                    ha="center", va="center", fontsize=9.5, color=C_DARK)

        # Method-specific decoration
        if title == "Prefix-Tuning":
            for i in range(n):
                y = base + i * (layer_h + gap)
                ax.add_patch(Rectangle(
                    (0.35, y + 0.12), 0.45, layer_h - 0.24,
                    facecolor=C_BLUE, edgecolor=C_BLUE, alpha=0.85,
                ))
            ax.text(0.57, base + n * (layer_h + gap) + 0.05,
                    "K/V\nprefix", ha="center", va="bottom",
                    fontsize=8.5, color=C_BLUE, weight="bold")
        elif title == "Prompt tuning (soft prompts)":
            ax.add_patch(Rectangle(
                (0.35, base - 0.55), 0.5, 0.4,
                facecolor=C_AMBER, edgecolor=C_AMBER, alpha=0.85,
            ))
            ax.text(0.6, base - 0.75, "soft\nprompt", ha="center", va="top",
                    fontsize=8.5, color="#b45309", weight="bold")
            a = FancyArrowPatch(
                (0.85, base - 0.35), (1.4, base - 0.05),
                arrowstyle="-|>", mutation_scale=10, color=C_AMBER, lw=1.2,
            )
            ax.add_patch(a)

        # Input/output arrows
        a = FancyArrowPatch(
            (2.5, 0.55), (2.5, base - 0.05),
            arrowstyle="-|>", mutation_scale=11, color=C_DARK, lw=1.0,
        )
        ax.add_patch(a)
        ax.text(2.5, 0.32, "input", ha="center", fontsize=8.5, color=C_DARK)

        a = FancyArrowPatch(
            (2.5, base + n * (layer_h + gap) - gap + 0.05),
            (2.5, base + n * (layer_h + gap) + 0.45),
            arrowstyle="-|>", mutation_scale=11, color=C_DARK, lw=1.0,
        )
        ax.add_patch(a)

        # Param budget badge
        ax.add_patch(FancyBboxPatch(
            (0.4, 0.05), 4.2, 0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0, facecolor=color, alpha=0.12,
        ))
        ax.text(2.5, 0.32, "", ha="center", va="center", fontsize=9)
        ax.text(2.5, 0.05 + 0.27, ptxt, ha="center", va="center",
                fontsize=8.8, color=color, weight="bold", multialignment="center")

    plt.subplots_adjust(top=0.92, wspace=0.18)
    _save(fig, "fig2_method_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Prefix length sweep -- quality and parameter cost
# ---------------------------------------------------------------------------
def fig3_prefix_length_sweep() -> None:
    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))

    # Synthetic curve modelled on Li & Liang Table 4 (E2E table-to-text).
    # Quality rises quickly, plateaus, then mildly degrades for very long prefixes.
    L = np.array([1, 2, 5, 10, 20, 50, 100, 200, 400])
    bleu = np.array([62.5, 66.4, 68.9, 70.0, 70.3, 70.1, 69.7, 69.0, 67.8])
    full_ft = 70.6  # reference

    ax1.plot(L, bleu, "-o", color=C_BLUE, lw=2.2, ms=7,
             label="Prefix-Tuning (E2E BLEU)")
    ax1.axhline(full_ft, ls="--", color=C_RED, lw=1.6,
                label=f"Full fine-tuning baseline = {full_ft}")
    ax1.fill_between(L, bleu, full_ft, where=(bleu < full_ft),
                     color=C_RED, alpha=0.06)

    ax1.set_xscale("log")
    ax1.set_xlabel("Prefix length $L_{\\text{prefix}}$ (log scale)", fontsize=11)
    ax1.set_ylabel("Task quality (BLEU on E2E)", color=C_BLUE, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_ylim(60, 73)
    ax1.set_xticks(L)
    ax1.set_xticklabels([str(x) for x in L])

    # Twin axis: trainable parameters
    ax2 = ax1.twinx()
    # GPT-2 medium: H=1024, layers=24, two matrices (K,V) -> 2*L*H*layers
    H, layers = 1024, 24
    params_M = 2 * L * H * layers / 1e6
    ax2.plot(L, params_M, "-s", color=C_AMBER, lw=2.0, ms=6,
             label="Trainable parameters (M)")
    ax2.set_ylabel("Trainable parameters (millions)",
                   color=C_AMBER, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=C_AMBER)
    ax2.grid(False)

    # Sweet-spot annotation
    sweet_idx = np.argmax(bleu)
    ax1.annotate(
        f"sweet spot\n$L \\approx {L[sweet_idx]}$",
        xy=(L[sweet_idx], bleu[sweet_idx]),
        xytext=(L[sweet_idx] * 2.4, bleu[sweet_idx] - 4.2),
        fontsize=10, color=C_GREEN, weight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.4),
    )

    ax1.set_title(
        "Prefix length: quality saturates around $L \\approx 10$–$20$, "
        "but parameter cost grows linearly",
        fontsize=12, weight="bold", color=C_DARK, pad=10,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower center", fontsize=9.5, frameon=True)

    plt.tight_layout()
    _save(fig, "fig3_prefix_length_sweep")


# ---------------------------------------------------------------------------
# Figure 4: KV-cache prepending during autoregressive decoding
# ---------------------------------------------------------------------------
def fig4_kv_cache_prepend() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    ax.text(
        6.0, 4.7,
        "How the prefix lives inside the attention KV-cache during generation",
        ha="center", fontsize=12, weight="bold", color=C_DARK,
    )

    # Cells configuration
    cell_w, cell_h = 0.55, 0.55
    n_prefix = 5
    n_input = 4
    n_gen_max = 4
    base_x = 0.7

    rows = [
        ("t = 0\n(prompt encode)", 3.45, n_prefix, n_input, 0),
        ("t = 1\n(decode 1st token)", 2.55, n_prefix, n_input, 1),
        ("t = 2\n(decode 2nd token)", 1.65, n_prefix, n_input, 2),
        ("t = 3\n(decode 3rd token)", 0.75, n_prefix, n_input, 3),
    ]

    for label, y, np_, ni_, ng_ in rows:
        # Row label
        ax.text(0.55, y + cell_h / 2, label, ha="right", va="center",
                fontsize=9.5, color=C_DARK)

        x = base_x
        # Prefix cells (always present)
        for i in range(np_):
            ax.add_patch(Rectangle(
                (x + i * cell_w, y), cell_w * 0.95, cell_h,
                facecolor=C_BLUE, alpha=0.85, edgecolor="white", linewidth=1.0,
            ))
        x += np_ * cell_w + 0.18

        # Input prompt cells
        for i in range(ni_):
            ax.add_patch(Rectangle(
                (x + i * cell_w, y), cell_w * 0.95, cell_h,
                facecolor=C_PURPLE, alpha=0.6, edgecolor="white", linewidth=1.0,
            ))
        x += ni_ * cell_w + 0.18

        # Generated tokens accumulating
        for i in range(ng_):
            ax.add_patch(Rectangle(
                (x + i * cell_w, y), cell_w * 0.95, cell_h,
                facecolor=C_GREEN, alpha=0.75, edgecolor="white", linewidth=1.0,
            ))
        # Pending slots (dashed)
        for i in range(ng_, n_gen_max):
            ax.add_patch(Rectangle(
                (x + i * cell_w, y), cell_w * 0.95, cell_h,
                facecolor="white", edgecolor=C_GRAY, linewidth=1.0, linestyle="--",
            ))

    # Legend at top
    legend_y = 4.18
    legend_items = [
        ("Prefix K/V (cached once, reused every step)", C_BLUE, 0.85),
        ("Prompt K/V (encoded at t=0)", C_PURPLE, 0.6),
        ("Generated-token K/V (appended each step)", C_GREEN, 0.75),
    ]
    lx = 0.7
    for label, color, alpha in legend_items:
        ax.add_patch(Rectangle((lx, legend_y), 0.35, 0.32,
                               facecolor=color, alpha=alpha,
                               edgecolor="white", linewidth=1.0))
        ax.text(lx + 0.45, legend_y + 0.16, label, va="center",
                fontsize=9.2, color=C_DARK)
        lx += 3.85

    # Side annotation
    ax.add_patch(FancyBboxPatch(
        (8.6, 0.75), 3.0, 2.95,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.3, edgecolor=C_BLUE, facecolor="#dbeafe",
    ))
    ax.text(10.1, 3.45,
            "Cost of the prefix\nat decode time",
            ha="center", va="center", fontsize=10, weight="bold", color=C_BLUE)
    ax.text(10.1, 2.4,
            "$\\bullet$ K/V written ONCE at t=0\n"
            "$\\bullet$ Each step pays\n   $O(L_{\\text{prefix}} + t)$\n"
            "   attention reads\n"
            "$\\bullet$ Memory adds\n   $2 L H \\cdot L_{\\text{prefix}}$\n"
            "   floats per request",
            ha="center", va="center", fontsize=9, color=C_DARK,
            multialignment="left")

    _save(fig, "fig4_kv_cache_prepend")


# ---------------------------------------------------------------------------
# Figure 5: GPT-2 generation tasks -- Prefix-Tuning vs full FT vs Adapter
# ---------------------------------------------------------------------------
def fig5_gpt2_application() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ---- Panel A: E2E table-to-text (BLEU) ----
    ax = axes[0]
    methods = ["Full FT", "Adapter\n(3% params)", "Prefix-Tuning\n(0.1% params)"]
    # Numbers loosely follow Li & Liang (2021) Table 1 trends.
    bleu = [69.5, 68.9, 70.3]
    colors = [C_RED, C_AMBER, C_BLUE]
    bars = ax.bar(methods, bleu, color=colors, edgecolor="white", linewidth=1.6)
    for b, v in zip(bars, bleu):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, weight="bold",
                color=b.get_facecolor())
    ax.set_ylim(65, 73)
    ax.set_ylabel("BLEU", fontsize=11)
    ax.set_title("E2E table-to-text (GPT-2 medium)",
                 fontsize=11.5, weight="bold", color=C_DARK, pad=8)
    ax.grid(axis="x", visible=False)

    # Param budget annotation strip
    storage = ["~355 M", "~11 M", "~0.4 M"]
    for i, s in enumerate(storage):
        ax.text(i, 65.4, f"per-task storage:\n{s}", ha="center", va="bottom",
                fontsize=8.8, color="#475569", style="italic")

    # ---- Panel B: XSum summarization (ROUGE-L) low-data regime ----
    ax = axes[1]
    # Training-set fractions
    frac = np.array([1, 5, 10, 25, 50, 100])
    full_ft = np.array([19.5, 27.0, 31.5, 35.4, 37.8, 39.0])
    prefix = np.array([23.6, 30.5, 33.7, 36.0, 37.2, 37.6])
    adapter = np.array([21.8, 28.6, 32.2, 35.1, 37.0, 38.0])

    ax.plot(frac, full_ft, "-o", color=C_RED, lw=2.0, ms=6, label="Full FT")
    ax.plot(frac, adapter, "-^", color=C_AMBER, lw=2.0, ms=6, label="Adapter")
    ax.plot(frac, prefix, "-s", color=C_BLUE, lw=2.4, ms=7, label="Prefix-Tuning")

    # Highlight the low-data crossover region
    ax.axvspan(1, 12, color=C_BLUE, alpha=0.07)
    ax.text(3.4, 21.0, "Prefix-Tuning wins\nin low-data regime",
            fontsize=9.5, color=C_BLUE, weight="bold", ha="center")

    ax.set_xscale("log")
    ax.set_xticks(frac)
    ax.set_xticklabels([f"{f}%" for f in frac])
    ax.set_xlabel("Fraction of XSum training data (log scale)", fontsize=11)
    ax.set_ylabel("ROUGE-L", fontsize=11)
    ax.set_ylim(17, 41)
    ax.set_title("XSum summarization (GPT-2 medium)",
                 fontsize=11.5, weight="bold", color=C_DARK, pad=8)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)

    fig.suptitle(
        "Prefix-Tuning matches full fine-tuning at <1% of the parameter cost, "
        "and tends to extrapolate better with little data",
        fontsize=12, weight="bold", color=C_DARK, y=1.02,
    )
    plt.tight_layout()
    _save(fig, "fig5_gpt2_application")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_architecture()
    fig2_method_comparison()
    fig3_prefix_length_sweep()
    fig4_kv_cache_prepend()
    fig5_gpt2_application()
    print("Saved 5 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
