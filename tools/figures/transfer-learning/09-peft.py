"""
Figure generation script for Transfer Learning Part 09:
Parameter-Efficient Fine-Tuning (PEFT).

Generates 7 figures used in both EN and ZH versions of the article.
Each figure isolates one teaching point and is built to be self-contained.

Figures:
    fig1_lora_decomposition     LoRA: W = W_0 + (alpha/r) * B*A. Shows the
                                low-rank factorisation visually -- a wide
                                d_in x d_out matrix replaced by a thin B and
                                a flat A whose product approximates the
                                update Delta W.
    fig2_param_count            Trainable parameter count: full fine-tuning
                                vs LoRA at several ranks vs other PEFT
                                methods. Log-scale bar chart making the
                                100x-1000x reduction visceral.
    fig3_adapter_placement      Adapter modules inside a Transformer block.
                                Shows the bottleneck (down-project, GELU,
                                up-project) inserted after attention and FFN
                                with a residual skip.
    fig4_prefix_tuning          Prefix-tuning: m learnable virtual tokens
                                prepended at every layer; only the prefix
                                key/value vectors are trained.
    fig5_prompt_vs_ptuning      Prompt-Tuning vs P-Tuning v2. Prompt-Tuning
                                only tunes input-layer soft prompts; P-Tuning
                                v2 injects trainable prompts at every layer.
    fig6_qlora_stack            QLoRA: 4-bit NF4 quantised frozen base model
                                + paged optimiser + bf16 LoRA adapters.
                                Memory budget for a 65B model on one GPU.
    fig7_glue_comparison        GLUE-style accuracy vs trainable parameter
                                fraction for full FT, Adapter, LoRA, Prefix-
                                Tuning, BitFit. Bubble chart highlighting the
                                Pareto frontier.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb #7c3aed #10b981 #f59e0b.

Usage:
    python3 scripts/figures/transfer-learning/09-peft.py

Output:
    Writes the same PNGs into BOTH the EN and ZH article asset folders so
    the markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
C_BLUE = COLORS["primary"]     # primary
C_PURPLE = COLORS["accent"]   # secondary
C_GREEN = COLORS["success"]    # accent / good
C_AMBER = COLORS["warning"]    # warning / highlight
C_GRAY = COLORS["muted"]
C_LIGHT = COLORS["grid"]
C_DARK = COLORS["text"]
C_BG = COLORS["bg"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "en" / "transfer-learning"
    / "09-parameter-efficient-fine-tuning"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "transfer-learning"
    / "09-参数高效微调"
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
# Figure 1: LoRA low-rank decomposition
# ---------------------------------------------------------------------------
def fig1_lora_decomposition() -> None:
    """W = W0 + (alpha/r) * B * A. Visualise dimensions and parameter ratio."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    d_in, d_out, r = 4096, 4096, 8
    full_params = d_in * d_out
    lora_params = r * (d_in + d_out)
    ratio = lora_params / full_params * 100

    # Title
    ax.text(6, 6.15,
            r"LoRA:  $W = W_0 + \frac{\alpha}{r}\, B\, A$",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C_DARK)

    # ---- W0 (frozen) ----
    w0_x, w0_y, w0_w, w0_h = 0.4, 1.4, 3.6, 3.6
    _box(ax, (w0_x, w0_y), w0_w, w0_h, "", C_GRAY, ec=C_DARK,
         alpha=0.30, rounding=0.08)
    ax.text(w0_x + w0_w / 2, w0_y + w0_h / 2,
            r"$W_0$" + "\n\n(frozen)\n" + f"{d_in}x{d_out}",
            ha="center", va="center", fontsize=12,
            color=C_DARK, fontweight="bold")
    ax.annotate(f"{full_params/1e6:.1f}M params (frozen)",
                xy=(w0_x + w0_w / 2, w0_y - 0.25),
                ha="center", fontsize=9, color=C_DARK)

    # plus sign
    ax.text(4.45, w0_y + w0_h / 2, "+",
            ha="center", va="center", fontsize=22,
            fontweight="bold", color=C_DARK)

    # ---- B (tall, trainable) ----
    b_x, b_y, b_w, b_h = 5.0, 1.4, 0.6, 3.6
    _box(ax, (b_x, b_y), b_w, b_h, "", C_BLUE, alpha=0.85, rounding=0.05)
    ax.text(b_x + b_w / 2, b_y + b_h / 2, "B",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color="white")
    ax.text(b_x + b_w / 2, b_y - 0.25,
            f"{d_out}x{r}", ha="center", fontsize=9, color=C_DARK)
    ax.text(b_x + b_w + 0.05, b_y + b_h + 0.15,
            "init = 0", ha="left", fontsize=8, color=C_BLUE,
            fontweight="bold")

    # multiply
    ax.text(5.95, w0_y + w0_h / 2, "x",
            ha="center", va="center", fontsize=18,
            fontweight="bold", color=C_DARK)

    # ---- A (wide, trainable) ----
    a_x, a_y, a_w, a_h = 6.3, 3.6, 3.6, 0.6
    _box(ax, (a_x, a_y), a_w, a_h, "", C_PURPLE, alpha=0.85, rounding=0.05)
    ax.text(a_x + a_w / 2, a_y + a_h / 2, "A",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color="white")
    ax.text(a_x + a_w / 2, a_y + a_h + 0.25,
            f"{r}x{d_in}", ha="center", fontsize=9, color=C_DARK)
    ax.text(a_x + a_w + 0.05, a_y - 0.05,
            "init ~ N(0, sigma^2)", ha="left", fontsize=8,
            color=C_PURPLE, fontweight="bold")

    # ---- Brace + Delta W explanation ----
    ax.annotate("", xy=(10.05, 1.4), xytext=(10.05, 5.0),
                arrowprops=dict(arrowstyle="-", color=C_DARK, lw=1.2))
    ax.text(10.25, 3.2,
            r"$\Delta W = B A$" + "\n" +
            f"rank <= r = {r}\n" +
            f"trainable: {lora_params/1e3:.1f}K\n" +
            f"= {ratio:.2f}% of W_0",
            ha="left", va="center", fontsize=10, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_GREEN, lw=1.2,
                      boxstyle="round,pad=0.4"))

    # bottom note
    ax.text(6, 0.55,
            "Forward:  h = W_0 x + (alpha/r) * B(A x)        "
            "Inference:  W_merged = W_0 + (alpha/r)*BA  -> zero overhead",
            ha="center", va="center", fontsize=9.5,
            color=C_DARK, style="italic")

    _save(fig, "fig1_lora_decomposition")


# ---------------------------------------------------------------------------
# Figure 2: Trainable parameter count comparison
# ---------------------------------------------------------------------------
def fig2_param_count() -> None:
    """Full FT vs LoRA (several r) vs other PEFT, log-scale bars."""
    fig, ax = plt.subplots(figsize=(11, 5.6))

    # Base model: GPT-3 175B
    base = 175_000  # in millions
    methods = [
        ("Full FT", base, C_AMBER),
        ("Adapter (m=64)", 0.012 * base, C_PURPLE),
        ("LoRA r=16", 0.0024 * base, C_BLUE),
        ("LoRA r=8", 0.0012 * base, C_BLUE),
        ("LoRA r=4", 0.00060 * base, C_BLUE),
        ("Prefix-Tuning", 0.001 * base, C_GREEN),
        ("Prompt-Tuning", 0.00002 * base, C_GREEN),
        ("BitFit", 0.0008 * base, C_GRAY),
    ]
    names = [m[0] for m in methods]
    values = np.array([m[1] for m in methods])  # in millions
    colors = [m[2] for m in methods]

    y = np.arange(len(methods))[::-1]
    bars = ax.barh(y, values, color=colors, alpha=0.88,
                   edgecolor=C_DARK, linewidth=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Trainable parameters (millions, log scale)",
                  fontsize=10, color=C_DARK)
    ax.set_title("Trainable parameters per method  (base: GPT-3, 175B)",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=12)
    ax.set_xlim(1e-3, base * 4)

    # annotate each bar with absolute value and ratio vs full
    for bar, v in zip(bars, values):
        pct = v / base * 100
        if v >= 1000:
            label = f"{v/1000:.1f} B  ({pct:.0f}%)"
        elif v >= 1:
            label = f"{v:.1f} M  ({pct:.3f}%)"
        else:
            label = f"{v*1000:.1f} K  ({pct:.4f}%)"
        ax.text(v * 1.25, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, color=C_DARK)

    # vertical reference line at 1% of full
    ax.axvline(base * 0.01, color=C_GREEN, lw=1.2, ls="--", alpha=0.6)
    ax.text(base * 0.01 * 1.1, len(methods) - 0.4, "1% of full",
            color=C_GREEN, fontsize=9, fontweight="bold")

    ax.grid(True, which="both", axis="x", alpha=0.3)
    _save(fig, "fig2_param_count")


# ---------------------------------------------------------------------------
# Figure 3: Adapter placement inside Transformer block
# ---------------------------------------------------------------------------
def fig3_adapter_placement() -> None:
    """Houlsby-style adapter inserted after attention and FFN sub-layers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2),
                             gridspec_kw=dict(width_ratios=[1.05, 1]))

    # ---- Left: Transformer block with adapters highlighted ----
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    ax.set_title("Transformer block with adapters",
                 fontsize=12, fontweight="bold", color=C_DARK)

    # block sequence (bottom -> top): Norm -> Attn -> Adapter -> Add
    #                                   Norm -> FFN  -> Adapter -> Add
    components = [
        ("LayerNorm",        C_LIGHT, C_DARK,  False),
        ("Multi-Head Attn", C_GRAY,  "white", False),
        ("Adapter",          C_GREEN, "white", True),
        ("Add (residual)",  C_LIGHT, C_DARK,  False),
        ("LayerNorm",        C_LIGHT, C_DARK,  False),
        ("Feed-Forward",    C_GRAY,  "white", False),
        ("Adapter",          C_GREEN, "white", True),
        ("Add (residual)",  C_LIGHT, C_DARK,  False),
    ]
    h = 0.95
    gap = 0.18
    y0 = 0.5
    for i, (name, fc, tc, trainable) in enumerate(components):
        y = y0 + i * (h + gap)
        _box(ax, (1.2, y), 3.6, h, name, fc, ec=C_DARK,
             text_color=tc, fontsize=10,
             fontweight="bold" if trainable else "normal", rounding=0.06)
        if trainable:
            ax.text(4.95, y + h / 2, "trainable",
                    color=C_GREEN, fontsize=9, fontweight="bold",
                    va="center")
        else:
            ax.text(4.95, y + h / 2, "frozen",
                    color=C_GRAY, fontsize=9, va="center")
        if i < len(components) - 1:
            _arrow(ax, (3.0, y + h), (3.0, y + h + gap), color=C_DARK)

    # input/output labels
    ax.text(3.0, 0.25, "input  h", ha="center", fontsize=9.5, color=C_DARK)
    ax.text(3.0, 10.05, "output  h'", ha="center",
            fontsize=9.5, color=C_DARK, fontweight="bold")

    # ---- Right: Adapter internal structure ----
    ax2 = axes[1]
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 10.5)
    ax2.axis("off")
    ax2.set_title("Inside an Adapter (bottleneck)",
                  fontsize=12, fontweight="bold", color=C_DARK)

    # input vector wide
    _box(ax2, (0.6, 8.6), 4.8, 0.8, "h in R^d  (d = 768)",
         C_LIGHT, ec=C_DARK, text_color=C_DARK, fontsize=10)
    _arrow(ax2, (3.0, 8.6), (3.0, 8.0))

    # down-project (narrowing trapezoid)
    _box(ax2, (1.6, 7.0), 2.8, 0.95, "W_down  (d -> m)",
         C_BLUE, fontsize=10, fontweight="bold")
    _arrow(ax2, (3.0, 7.0), (3.0, 6.4))

    # bottleneck
    _box(ax2, (2.2, 5.4), 1.6, 0.95, "z in R^m  (m = 64)",
         C_AMBER, text_color="white", fontsize=10, fontweight="bold")
    _arrow(ax2, (3.0, 5.4), (3.0, 4.8))

    # nonlinearity
    _box(ax2, (2.0, 4.0), 2.0, 0.7, "GELU",
         C_PURPLE, fontsize=10, fontweight="bold")
    _arrow(ax2, (3.0, 4.0), (3.0, 3.4))

    # up-project
    _box(ax2, (1.6, 2.0), 2.8, 0.95, "W_up  (m -> d)",
         C_BLUE, fontsize=10, fontweight="bold")
    _arrow(ax2, (3.0, 2.0), (3.0, 1.4))

    # residual add
    _box(ax2, (1.0, 0.4), 4.0, 0.85,
         "Adapter(h) = h + W_up * GELU(W_down h)",
         C_GREEN, text_color="white", fontsize=10, fontweight="bold")

    # parameter count side note
    ax2.text(5.7, 5.4,
             "Params per adapter\n2 m d  (e.g.\n2*64*768 ~ 98K)",
             ha="left", va="center", fontsize=9, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_GREEN, lw=1.0,
                       boxstyle="round,pad=0.3"))

    plt.tight_layout()
    _save(fig, "fig3_adapter_placement")


# ---------------------------------------------------------------------------
# Figure 4: Prefix-Tuning with learnable virtual tokens
# ---------------------------------------------------------------------------
def fig4_prefix_tuning() -> None:
    """Learnable prefix tokens prepended at every layer of the Transformer."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.set_title("Prefix-Tuning: m learnable virtual tokens at every layer",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)

    n_layers = 4
    n_prefix = 3
    n_real = 5
    cell_w, cell_h = 0.7, 0.6
    gap_x, gap_y = 0.10, 0.55
    x0 = 1.2
    y0 = 0.6

    for L in range(n_layers):
        y = y0 + L * (cell_h + gap_y)
        # layer label
        ax.text(x0 - 0.5, y + cell_h / 2, f"Layer {L+1}",
                ha="right", va="center", fontsize=10, color=C_DARK,
                fontweight="bold")
        # prefix cells (trainable)
        for k in range(n_prefix):
            x = x0 + k * (cell_w + gap_x)
            _box(ax, (x, y), cell_w, cell_h, f"P{k+1}",
                 C_PURPLE, fontsize=9, fontweight="bold", rounding=0.04)
        # divider gap
        sep_x = x0 + n_prefix * (cell_w + gap_x) + 0.15
        ax.plot([sep_x, sep_x], [y - 0.05, y + cell_h + 0.05],
                color=C_GRAY, lw=0.8, ls=":")
        # real input tokens (frozen)
        for k in range(n_real):
            x = sep_x + 0.25 + k * (cell_w + gap_x)
            _box(ax, (x, y), cell_w, cell_h, f"x{k+1}",
                 C_LIGHT, text_color=C_DARK, fontsize=9, rounding=0.04)

    # legend / annotation
    ax.add_patch(Rectangle((9.0, 5.2), 0.5, 0.4,
                           facecolor=C_PURPLE, edgecolor=C_DARK))
    ax.text(9.6, 5.4, "P_i  trainable prefix (key/value)",
            fontsize=10, color=C_DARK, va="center")
    ax.add_patch(Rectangle((9.0, 4.55), 0.5, 0.4,
                           facecolor=C_LIGHT, edgecolor=C_DARK))
    ax.text(9.6, 4.75, "x_i  frozen input tokens",
            fontsize=10, color=C_DARK, va="center")

    ax.text(9.0, 3.6,
            "Prefix params:\n  m * d * L * 2   (key + value)\n"
            "  e.g. m=20, d=1024,\n  L=24  ->  0.98M  (~0.5%)",
            ha="left", va="top", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_GREEN, lw=1.0,
                      boxstyle="round,pad=0.4"))

    ax.text(6.0, 0.05,
            "Reparameterise via small MLP at training time -> drop MLP at inference.",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    _save(fig, "fig4_prefix_tuning")


# ---------------------------------------------------------------------------
# Figure 5: Prompt-Tuning vs P-Tuning v2
# ---------------------------------------------------------------------------
def fig5_prompt_vs_ptuning() -> None:
    """Where soft prompts live: input layer only vs every layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    n_layers = 5
    n_prompt = 3
    n_real = 5
    cell_w, cell_h = 0.55, 0.5
    gap_x, gap_y = 0.08, 0.40

    for ax, title, mode in [
        (axes[0], "Prompt-Tuning  (input layer only)", "input"),
        (axes[1], "P-Tuning v2  (every layer)", "all"),
    ]:
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 6.5)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=C_DARK, pad=8)

        x0 = 1.4
        y0 = 0.5
        for L in range(n_layers):
            y = y0 + L * (cell_h + gap_y)
            ax.text(x0 - 0.45, y + cell_h / 2, f"L{L+1}",
                    ha="right", va="center", fontsize=9, color=C_DARK)
            trainable_layer = (mode == "all") or (L == 0)
            for k in range(n_prompt):
                x = x0 + k * (cell_w + gap_x)
                fc = C_PURPLE if trainable_layer else C_LIGHT
                tc = "white" if trainable_layer else C_DARK
                _box(ax, (x, y), cell_w, cell_h, f"p{k+1}",
                     fc, text_color=tc, fontsize=8, rounding=0.04)
            sep = x0 + n_prompt * (cell_w + gap_x) + 0.05
            ax.plot([sep, sep], [y - 0.04, y + cell_h + 0.04],
                    color=C_GRAY, lw=0.8, ls=":")
            for k in range(n_real):
                x = sep + 0.20 + k * (cell_w + gap_x)
                _box(ax, (x, y), cell_w, cell_h, f"x{k+1}",
                     C_LIGHT, text_color=C_DARK, fontsize=8, rounding=0.04)

        if mode == "input":
            note = ("Trainable: only m*d at input layer\n"
                    "Pros: simplest, scales with model size\n"
                    "Best for: very large models (>10B)")
        else:
            note = ("Trainable: m*d*L at every layer\n"
                    "Pros: matches full FT on small/medium\n"
                    "Best for: any model size & task")
        ax.text(0.05, -0.05, note, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.0, color=C_DARK,
                bbox=dict(facecolor=C_BG, edgecolor=C_BLUE, lw=1.0,
                          boxstyle="round,pad=0.35"))

    plt.tight_layout()
    _save(fig, "fig5_prompt_vs_ptuning")


# ---------------------------------------------------------------------------
# Figure 6: QLoRA stack and memory budget
# ---------------------------------------------------------------------------
def fig6_qlora_stack() -> None:
    """4-bit NF4 quantised base + paged optimiser + bf16 LoRA adapters."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6),
                             gridspec_kw=dict(width_ratios=[1.0, 1.05]))

    # ---- Left: stack diagram ----
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("QLoRA stack",
                 fontsize=12, fontweight="bold", color=C_DARK)

    layers = [
        ("LoRA adapters (bf16, trainable)",  C_BLUE,   "white", True),
        ("Frozen base weights\nNF4 4-bit quantised",
         C_AMBER,  "white", False),
        ("Double quantisation\n(quantise the quant constants)",
         C_PURPLE, "white", False),
        ("Paged optimiser states\n(CPU ↔ GPU paging)",
         C_GREEN,  "white", False),
    ]
    y = 6.2
    for name, fc, tc, _ in layers:
        _box(ax, (0.5, y - 1.1), 5.0, 1.0, name, fc,
             text_color=tc, fontsize=10, fontweight="bold", rounding=0.06)
        y -= 1.4

    # forward dataflow arrow
    ax.annotate("", xy=(0.25, 5.7), xytext=(0.25, 1.4),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.4))
    ax.text(0.05, 3.5, "forward / backward",
            rotation=90, va="center", ha="center",
            fontsize=9, color=C_DARK)

    # ---- Right: memory bar chart for 65B model ----
    ax2 = axes[1]
    methods = ["Full FT\n(fp16)", "LoRA\n(fp16)", "QLoRA\n(NF4 + bf16)"]
    # rough memory budget for a 65B model in GB
    weights = [130, 130, 33]      # base weights
    grads_opt = [520, 8, 4]        # optimiser + grads (Adam = 8x param count fp32)
    activations = [40, 30, 12]     # activations + paging buffers

    x = np.arange(len(methods))
    w = 0.55
    p1 = ax2.bar(x, weights, w, color=C_GRAY, label="Base weights",
                 edgecolor=C_DARK, linewidth=0.6)
    p2 = ax2.bar(x, grads_opt, w, bottom=weights, color=C_AMBER,
                 label="Grads + optimiser", edgecolor=C_DARK, linewidth=0.6)
    p3 = ax2.bar(x, activations, w,
                 bottom=np.array(weights) + np.array(grads_opt),
                 color=C_BLUE, label="Activations",
                 edgecolor=C_DARK, linewidth=0.6)

    totals = np.array(weights) + np.array(grads_opt) + np.array(activations)
    for xi, t in zip(x, totals):
        ax2.text(xi, t + 12, f"{t} GB", ha="center",
                 fontsize=10, fontweight="bold", color=C_DARK)

    # 80GB reference line (single A100/H100)
    ax2.axhline(80, color=C_GREEN, lw=1.4, ls="--")
    ax2.text(len(methods) - 0.5, 86, "single 80 GB GPU",
             color=C_GREEN, fontsize=9, fontweight="bold", ha="right")

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.set_ylabel("GPU memory (GB)", fontsize=10, color=C_DARK)
    ax2.set_title("Memory to fine-tune a 65B model",
                  fontsize=12, fontweight="bold", color=C_DARK, pad=8)
    ax2.set_ylim(0, max(totals) * 1.18)
    ax2.legend(loc="upper right", fontsize=9, frameon=True)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig6_qlora_stack")


# ---------------------------------------------------------------------------
# Figure 7: PEFT methods on GLUE (efficiency vs accuracy)
# ---------------------------------------------------------------------------
def fig7_glue_comparison() -> None:
    """Bubble chart: accuracy vs trainable param fraction.

    Numbers approximate published results for RoBERTa-base on GLUE average.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    # name, trainable_pct (vs full FT), GLUE avg, bubble size, color
    rows = [
        ("Full FT",        100.0,    86.4, 1300, C_AMBER),
        ("BitFit",         0.08,     85.2,  350, C_GRAY),
        ("Adapter (m=64)", 0.94,     86.0,  700, C_PURPLE),
        ("Prefix-Tuning",  0.10,     85.4,  400, C_GREEN),
        ("Prompt-Tuning",  0.01,     82.8,  250, C_GREEN),
        ("LoRA r=8",       0.24,     86.3,  600, C_BLUE),
        ("LoRA r=16",      0.48,     86.5,  750, C_BLUE),
    ]
    # Explicit (label_x, label_y, ha) to avoid bubble overlap.
    label_pos = {
        "Full FT":        (40.0,   86.95, "right"),
        "BitFit":         (0.022,  85.20, "left"),
        "Adapter (m=64)": (3.0,    85.60, "left"),
        "Prefix-Tuning":  (0.022,  85.55, "left"),
        "Prompt-Tuning":  (0.022,  82.80, "left"),
        "LoRA r=8":       (0.40,   85.30, "left"),
        "LoRA r=16":      (1.20,   86.95, "left"),
    }
    for name, x, y, s, c in rows:
        ax.scatter(x, y, s=s, color=c, edgecolor=C_DARK,
                   linewidth=1.0, alpha=0.85, zorder=3)
        lx, ly, ha = label_pos[name]
        # leader line (skip when label sits right next to bubble)
        if name not in ("BitFit", "Prompt-Tuning", "Full FT"):
            ax.plot([x, lx], [y, ly], color=C_GRAY, lw=0.7,
                    alpha=0.6, zorder=2)
        ax.annotate(name, (lx, ly),
                    fontsize=10, color=C_DARK, ha=ha, va="center",
                    fontweight="bold", zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(0.005, 250)
    ax.set_ylim(81.5, 87.5)
    ax.set_xlabel("Trainable parameters (% of full fine-tuning, log scale)",
                  fontsize=10.5, color=C_DARK)
    ax.set_ylabel("GLUE average score", fontsize=10.5, color=C_DARK)
    ax.set_title(
        "PEFT efficiency vs accuracy  (RoBERTa-base, GLUE avg, approx.)",
        fontsize=12.5, fontweight="bold", color=C_DARK, pad=10,
    )

    # Pareto guide line
    ax.axhline(86.4, color=C_AMBER, lw=1.0, ls="--", alpha=0.6)
    ax.text(0.006, 86.55, "full FT baseline",
            color=C_AMBER, fontsize=9, fontweight="bold")

    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig7_glue_comparison")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}\n")
    for fn in [
        fig1_lora_decomposition,
        fig2_param_count,
        fig3_adapter_placement,
        fig4_prefix_tuning,
        fig5_prompt_vs_ptuning,
        fig6_qlora_stack,
        fig7_glue_comparison,
    ]:
        print(f"-- {fn.__name__}")
        fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
