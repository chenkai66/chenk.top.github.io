"""Figures for NLP Part 8 — Fine-tuning and PEFT.

Generates 7 publication-quality figures for the article on
parameter-efficient fine-tuning. Each figure is saved into BOTH the
English and Chinese asset folders.

Figures:
    fig1_full_vs_peft    — Trainable parameter count: full FT vs PEFT methods
    fig2_lora_decomp     — LoRA: W = W_0 + (alpha/r) * B*A decomposition
    fig3_adapter         — Adapter placement inside a Transformer block
    fig4_prefix_prompt   — Prefix-tuning vs Prompt-tuning vs P-Tuning v2
    fig5_qlora           — QLoRA: 4-bit NF4 base + LoRA adapters in fp16
    fig6_memory          — VRAM footprint: full FT vs LoRA vs QLoRA (7B/13B/70B)
    fig7_perf_vs_params  — Performance retention vs trainable parameter ratio

Style: seaborn-v0_8-whitegrid, dpi=150, palette
    #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (orange).

References (verified):
    - Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)
    - Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs (NeurIPS 2023)
    - Houlsby et al., Parameter-Efficient Transfer Learning for NLP (ICML 2019)
    - Li & Liang, Prefix-Tuning (ACL 2021)
    - Lester et al., The Power of Scale for Parameter-Efficient Prompt Tuning (EMNLP 2021)
    - Liu et al., P-Tuning v2 (ACL 2022)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Shared style ----------------------------------------------------------------
import sys
from pathlib import Path as _StylePath
sys.path.insert(0, str(_StylePath(__file__).parent.parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# Color palette (from shared _style)
BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
ORANGE = COLORS["warning"]
GRAY = COLORS["gray"]
LIGHT = COLORS["light"]
DARK = COLORS["ink"]
RED = COLORS["danger"]


REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/nlp/fine-tuning-peft"
ZH_DIR = REPO / "source/_posts/zh/nlp/08-模型微调与PEFT"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders, then close it."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _rounded(ax, x, y, w, h, text, edge, fc, *, fs=10, weight="normal", tc=None):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=edge, facecolor=fc, alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, weight=weight,
        color=tc if tc else DARK,
    )


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="->", mut=14):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=mut, linewidth=lw, color=color,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1 — Full FT vs PEFT: trainable parameter counts (log scale)
# ---------------------------------------------------------------------------
def fig1_full_vs_peft() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # Left: bar chart, trainable params (millions) on a log scale, 7B base
    methods = ["Full FT", "Frozen\n(top layers)", "Adapter", "Prefix-Tuning",
               "Prompt-Tuning", "LoRA r=16", "QLoRA r=16"]
    # Trainable parameters in millions (LLaMA-7B reference)
    params_m = [6738, 800, 35, 4.2, 0.8, 8.4, 8.4]
    colors = [RED, ORANGE, PURPLE, BLUE, BLUE, GREEN, GREEN]

    bars = ax1.barh(methods, params_m, color=colors, alpha=0.85,
                    edgecolor="white", linewidth=1.5)
    ax1.set_xscale("log")
    ax1.set_xlim(0.3, 20000)
    ax1.set_xlabel("Trainable parameters (millions, log scale)")
    ax1.set_title("Trainable parameters — LLaMA-7B base", loc="left")
    ax1.invert_yaxis()
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.grid(False, axis="y")

    for bar, val in zip(bars, params_m):
        pct = val / 6738 * 100
        label = f"{val:,.1f} M  ({pct:.2f}%)" if pct >= 0.01 else f"{val:.2f} M  (<0.01%)"
        ax1.text(val * 1.15, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=9, color=DARK)

    # Right: storage cost — one base + N task adapters
    n_tasks = np.arange(1, 21)
    full_gb = n_tasks * 13.5      # one full 7B model per task in fp16
    lora_gb = 13.5 + n_tasks * 0.034   # base + per-task LoRA (~34 MB)
    qlora_gb = 3.8 + n_tasks * 0.034   # 4-bit base + LoRA

    ax2.plot(n_tasks, full_gb, "-o", color=RED, lw=2.2, ms=5,
             label="Full FT (one model / task)")
    ax2.plot(n_tasks, lora_gb, "-s", color=GREEN, lw=2.2, ms=5,
             label="LoRA (shared fp16 base)")
    ax2.plot(n_tasks, qlora_gb, "-^", color=BLUE, lw=2.2, ms=5,
             label="QLoRA (shared 4-bit base)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of task-specific deployments")
    ax2.set_ylabel("Total disk footprint (GB, log scale)")
    ax2.set_title("Serving N tasks from one base", loc="left")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Full Fine-tuning vs Parameter-Efficient Fine-tuning",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_full_vs_peft.png")


# ---------------------------------------------------------------------------
# Figure 2 — LoRA decomposition diagram
# ---------------------------------------------------------------------------
def fig2_lora_decomp() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    ax.text(5.5, 5.25, r"LoRA: $W = W_0 + \frac{\alpha}{r}\, B A$",
            ha="center", fontsize=15, weight="bold")

    # Frozen W_0 — large square
    w0_x, w0_y, w0_s = 0.4, 1.1, 2.6
    ax.add_patch(Rectangle((w0_x, w0_y), w0_s, w0_s,
                           facecolor=GRAY, edgecolor=DARK, lw=1.6, alpha=0.55,
                           hatch="///"))
    ax.text(w0_x + w0_s / 2, w0_y + w0_s / 2 + 0.15,
            r"$W_0$", ha="center", va="center", fontsize=20, weight="bold")
    ax.text(w0_x + w0_s / 2, w0_y + w0_s / 2 - 0.35,
            "frozen\n(d × k)", ha="center", va="center",
            fontsize=9.5, color=DARK, style="italic")
    ax.text(w0_x + w0_s / 2, w0_y - 0.35,
            "d·k parameters\ne.g. 1024×1024 = 1.05 M",
            ha="center", fontsize=9, color=GRAY)

    # Plus sign
    ax.text(3.5, 2.4, "+", ha="center", va="center",
            fontsize=30, weight="bold", color=DARK)

    # B matrix — tall thin rectangle (d × r)
    b_x, b_y, b_w, b_h = 4.1, 1.1, 0.7, 2.6
    ax.add_patch(Rectangle((b_x, b_y), b_w, b_h,
                           facecolor=GREEN, edgecolor=DARK, lw=1.6, alpha=0.85))
    ax.text(b_x + b_w / 2, b_y + b_h / 2, "B",
            ha="center", va="center", fontsize=18, weight="bold", color="white")
    ax.text(b_x + b_w / 2, b_y - 0.25, "d × r",
            ha="center", fontsize=10, color=DARK)
    ax.text(b_x + b_w / 2, b_y - 0.6, "init: 0",
            ha="center", fontsize=9, color=GREEN, weight="bold")

    # Multiplication dot
    ax.text(5.15, 2.4, "·", ha="center", va="center", fontsize=36,
            weight="bold", color=DARK)

    # A matrix — short wide rectangle (r × k)
    a_x, a_y, a_w, a_h = 5.4, 2.65, 2.6, 0.55
    ax.add_patch(Rectangle((a_x, a_y), a_w, a_h,
                           facecolor=BLUE, edgecolor=DARK, lw=1.6, alpha=0.85))
    ax.text(a_x + a_w / 2, a_y + a_h / 2, "A",
            ha="center", va="center", fontsize=18, weight="bold", color="white")
    ax.text(a_x + a_w / 2, a_y - 0.3, "r × k",
            ha="center", fontsize=10, color=DARK)
    ax.text(a_x + a_w / 2, a_y - 0.6, "init: N(0, σ²)",
            ha="center", fontsize=9, color=BLUE, weight="bold")

    # Equals — result of B·A is d×k but trainable params are tiny
    ax.text(8.25, 2.4, "=", ha="center", va="center",
            fontsize=24, weight="bold", color=DARK)

    # Resulting low-rank ΔW
    dx, dy, ds = 8.6, 1.1, 2.0
    ax.add_patch(Rectangle((dx, dy), ds, ds,
                           facecolor=ORANGE, edgecolor=DARK, lw=1.6, alpha=0.85))
    ax.text(dx + ds / 2, dy + ds / 2 + 0.15, r"$\Delta W$",
            ha="center", va="center", fontsize=18, weight="bold", color="white")
    ax.text(dx + ds / 2, dy + ds / 2 - 0.4,
            "rank ≤ r", ha="center", va="center",
            fontsize=9.5, color="white", style="italic")
    ax.text(dx + ds / 2, dy - 0.35,
            "trainable: r·(d+k)\nr=8 → 16 K params (1.5%)",
            ha="center", fontsize=9, color=DARK)

    # Bottom annotation
    ax.text(5.5, 0.25,
            "Forward pass:  h = x W₀ᵀ + (α/r) · x Aᵀ Bᵀ        "
            "Inference:  W ← W₀ + (α/r) BA  →  zero overhead",
            ha="center", fontsize=10.5, color=DARK, style="italic",
            bbox=dict(facecolor=LIGHT, edgecolor="none", pad=8))

    save(fig, "fig2_lora_decomp.png")


# ---------------------------------------------------------------------------
# Figure 3 — Adapter placement inside a Transformer block
# ---------------------------------------------------------------------------
def fig3_adapter() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.6),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- Left: Transformer block with Adapter modules inserted ----
    axL.set_xlim(0, 5)
    axL.set_ylim(0, 11)
    axL.axis("off")
    axL.set_title("Adapter placement (Houlsby-style)", loc="left", pad=8)

    # block components, bottom to top
    items = [
        ("Multi-Head Self-Attention", BLUE, 1.6, False),
        ("Adapter (down → ReLU → up)", GREEN, 0.9, True),
        ("Add & LayerNorm", GRAY, 0.7, False),
        ("Feed-Forward Network", PURPLE, 1.6, False),
        ("Adapter (down → ReLU → up)", GREEN, 0.9, True),
        ("Add & LayerNorm", GRAY, 0.7, False),
    ]
    y = 0.7
    cx = 2.5
    for text, color, h, trainable in items:
        fc = color if trainable else "white"
        tc = "white" if trainable else DARK
        weight = "bold" if trainable else "normal"
        _rounded(axL, 0.6, y, 3.8, h, text, color, fc,
                 fs=10, weight=weight, tc=tc)
        if y + h < 9.5:
            _arrow(axL, cx, y + h + 0.02, cx, y + h + 0.18, color=DARK, lw=1.4)
        y += h + 0.2

    # input/output arrows
    _arrow(axL, cx, 0.2, cx, 0.65, color=DARK)
    _arrow(axL, cx, y - 0.05, cx, y + 0.45, color=DARK)
    axL.text(cx, 10.55, "h_out", ha="center", fontsize=10, weight="bold")
    axL.text(cx, 0.05, "h_in", ha="center", fontsize=10, weight="bold")

    # legend
    axL.text(0.6, 10.6, "■ trainable", color=GREEN,
             fontsize=10, weight="bold")
    axL.text(2.0, 10.6, "■ frozen", color=GRAY, fontsize=10, weight="bold")

    # ---- Right: Adapter internal bottleneck ----
    axR.set_xlim(0, 6)
    axR.set_ylim(0, 11)
    axR.axis("off")
    axR.set_title("Adapter bottleneck structure", loc="left", pad=8)

    # input
    axR.text(3, 10.4, "x  (d = 768)", ha="center", fontsize=11, weight="bold")
    _arrow(axR, 3, 10.1, 3, 9.5)

    # down-projection trapezoid
    axR.add_patch(plt.Polygon([(0.7, 9.5), (5.3, 9.5),
                               (4.0, 8.2), (2.0, 8.2)],
                              facecolor=GREEN, edgecolor=DARK, lw=1.4, alpha=0.85))
    axR.text(3, 8.85, "Down-projection W_down  (d → m)",
             ha="center", color="white", fontsize=10, weight="bold")
    _arrow(axR, 3, 8.15, 3, 7.5)

    # bottleneck activation
    _rounded(axR, 1.7, 6.7, 2.6, 0.8,
             "ReLU / GELU\nbottleneck m = 64", PURPLE, PURPLE,
             fs=10, weight="bold", tc="white")
    _arrow(axR, 3, 6.65, 3, 6.0)

    # up-projection trapezoid
    axR.add_patch(plt.Polygon([(2.0, 6.0), (4.0, 6.0),
                               (5.3, 4.7), (0.7, 4.7)],
                              facecolor=GREEN, edgecolor=DARK, lw=1.4, alpha=0.85))
    axR.text(3, 5.35, "Up-projection W_up  (m → d)",
             ha="center", color="white", fontsize=10, weight="bold")
    _arrow(axR, 3, 4.65, 3, 4.0)

    # residual sum
    _rounded(axR, 2.1, 3.2, 1.8, 0.8, "+  residual", DARK, LIGHT,
             fs=11, weight="bold")
    _arrow(axR, 3, 3.15, 3, 2.5)
    axR.text(3, 2.2, "h = x + W_up · σ(W_down · x)",
             ha="center", fontsize=10.5, weight="bold", color=DARK)

    # parameter count
    axR.text(3, 1.3,
             "Adapter parameters per layer:\n"
             "2 m d + (m + d)  ≈  100 K  ( ≈ 0.5 % of layer )",
             ha="center", fontsize=9.5, color=GRAY,
             bbox=dict(facecolor=LIGHT, edgecolor="none", pad=6))

    save(fig, "fig3_adapter.png")


# ---------------------------------------------------------------------------
# Figure 4 — Prefix-Tuning vs Prompt-Tuning vs P-Tuning v2
# ---------------------------------------------------------------------------
def fig4_prefix_prompt() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    titles = [
        "Prompt-Tuning\n(soft prompt at input only)",
        "Prefix-Tuning\n(learnable K/V prefix per layer)",
        "P-Tuning v2\n(deep prompts + no MLP reparam)",
    ]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.set_title(title, fontsize=11.5, weight="bold")

    # Common: stack of L Transformer layers shown as 4 boxes
    layer_y = [1.0, 2.4, 3.8, 5.2]
    for ax in axes:
        for y in layer_y:
            _rounded(ax, 2.0, y, 6.0, 0.9, "Transformer layer", GRAY, "white",
                     fs=9.5)
        # input tokens row at top
        for i, tok in enumerate(["x₁", "x₂", "x₃", "x₄", "x₅"]):
            _rounded(ax, 2.0 + i * 1.2, 6.6, 1.0, 0.6, tok, BLUE, "white",
                     fs=9, tc=BLUE, weight="bold")
        ax.text(5.0, 7.45, "Input tokens", ha="center", fontsize=9.5,
                color=BLUE, weight="bold")
        ax.text(5.0, 0.5, "Output", ha="center", fontsize=9.5,
                color=DARK, weight="bold")

    # ---- (1) Prompt-Tuning: prepend soft tokens at input only ----
    ax = axes[0]
    for i, tok in enumerate(["p₁", "p₂"]):
        _rounded(ax, 0.2 + i * 0.85, 6.6, 0.75, 0.6, tok, GREEN, GREEN,
                 fs=9.5, tc="white", weight="bold")
    ax.text(0.95, 7.45, "soft prompts\n(trainable)", ha="center", fontsize=8.5,
            color=GREEN, weight="bold")

    # ---- (2) Prefix-Tuning: K/V prefix injected at every layer ----
    ax = axes[1]
    for y in layer_y:
        _rounded(ax, 0.3, y + 0.15, 1.4, 0.6, "P_K, P_V", GREEN, GREEN,
                 fs=8.5, tc="white", weight="bold")
        _arrow(ax, 1.75, y + 0.45, 1.95, y + 0.45, color=GREEN)
    ax.text(1.0, 6.6, "prefix per\nlayer", ha="center", fontsize=8.5,
            color=GREEN, weight="bold")

    # ---- (3) P-Tuning v2: deep prompts at every layer, no MLP reparam ----
    ax = axes[2]
    for y in layer_y:
        for i in range(2):
            _rounded(ax, 0.2 + i * 0.85, y + 0.15, 0.75, 0.6,
                     f"p", PURPLE, PURPLE, fs=8.5, tc="white", weight="bold")
        _arrow(ax, 1.85, y + 0.45, 1.95, y + 0.45, color=PURPLE)
    ax.text(0.95, 6.6, "deep\nprompts", ha="center", fontsize=8.5,
            color=PURPLE, weight="bold")

    fig.suptitle("Prompt-based PEFT methods",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_prefix_prompt.png")


# ---------------------------------------------------------------------------
# Figure 5 — QLoRA architecture
# ---------------------------------------------------------------------------
def fig5_qlora() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("QLoRA — 4-bit quantized base + LoRA adapters in fp16",
                 loc="left", fontsize=13, weight="bold")

    # Forward arrow flow row
    # Input
    _rounded(ax, 0.2, 2.4, 1.4, 1.0, "x\n(fp16)", BLUE, "white", fs=10,
             weight="bold", tc=BLUE)
    _arrow(ax, 1.65, 2.9, 2.05, 2.9)

    # Frozen 4-bit base weights — large box
    _rounded(ax, 2.1, 1.6, 3.2, 2.6,
             "Frozen base weights\nW₀  (4-bit NF4)\n\nstorage: 0.5 byte/param\n"
             "double-quantized\nconstants",
             GRAY, GRAY, fs=9.5, weight="bold", tc="white")
    # de-quant arrow up
    _arrow(ax, 5.35, 2.9, 5.85, 2.9, color=DARK)
    ax.text(5.6, 3.25, "de-quant\nto bf16", ha="center", fontsize=8.5,
            color=DARK, style="italic")

    # Multiplication node
    _rounded(ax, 5.9, 2.4, 1.0, 1.0, "× W₀ᵀ", DARK, LIGHT,
             fs=10, weight="bold")
    _arrow(ax, 6.95, 2.9, 7.45, 2.9)

    # plus node
    _rounded(ax, 7.5, 2.4, 0.7, 1.0, "+", DARK, LIGHT,
             fs=14, weight="bold")
    _arrow(ax, 8.25, 2.9, 8.75, 2.9)

    # output
    _rounded(ax, 8.8, 2.4, 1.4, 1.0, "h\n(bf16)", BLUE, "white", fs=10,
             weight="bold", tc=BLUE)

    # LoRA branch (above): trainable adapters in bf16
    # Branch from input
    _arrow(ax, 0.9, 3.45, 0.9, 4.7, color=GREEN)
    _rounded(ax, 0.4, 4.7, 1.0, 0.7, "A (bf16)", GREEN, GREEN, fs=9.5,
             weight="bold", tc="white")
    _arrow(ax, 1.45, 5.05, 2.85, 5.05, color=GREEN)
    _rounded(ax, 2.9, 4.7, 1.0, 0.7, "B (bf16)", GREEN, GREEN, fs=9.5,
             weight="bold", tc="white")
    _arrow(ax, 3.95, 5.05, 7.85, 5.05, color=GREEN)
    _arrow(ax, 7.85, 5.05, 7.85, 3.45, color=GREEN)
    ax.text(2.2, 5.55, "LoRA — trainable (bf16)", color=GREEN,
            fontsize=10, weight="bold")
    ax.text(5.5, 5.55, "(α/r) · B A x  →", color=GREEN,
            fontsize=10, weight="bold")

    # Bottom: paged optimizer note
    ax.text(0.2, 0.85,
            "Three QLoRA innovations:\n"
            "  ① 4-bit NormalFloat (NF4) — info-theoretically optimal for "
            "normal-distributed weights\n"
            "  ② Double quantization — quantize the quantization constants "
            "themselves (~0.4 bit/param saved)\n"
            "  ③ Paged optimizers — page AdamW state to CPU on long-sequence "
            "memory spikes",
            fontsize=9.5, color=DARK,
            bbox=dict(facecolor=LIGHT, edgecolor="none", pad=8))

    save(fig, "fig5_qlora.png")


# ---------------------------------------------------------------------------
# Figure 6 — Memory footprint comparison
# ---------------------------------------------------------------------------
def fig6_memory() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # Component breakdown for a 7B model (training memory in GB)
    methods = ["Full FT\n(fp16 + AdamW)",
               "LoRA r=16\n(fp16 base, fp32 adam)",
               "QLoRA r=16\n(NF4 base + LoRA bf16)"]
    weights = [14.0, 14.0, 3.8]
    grad_mem = [14.0, 0.06, 0.06]
    optim = [56.0, 0.12, 0.12]
    activ = [12.0, 12.0, 12.0]

    x = np.arange(len(methods))
    bw = 0.55
    p1 = ax1.bar(x, weights, bw, color=GRAY, label="Model weights",
                 edgecolor="white")
    p2 = ax1.bar(x, grad_mem, bw, bottom=weights, color=ORANGE,
                 label="Gradients", edgecolor="white")
    p3 = ax1.bar(x, optim, bw, bottom=np.array(weights) + np.array(grad_mem),
                 color=PURPLE, label="Optimizer state (AdamW = 8B/param)",
                 edgecolor="white")
    p4 = ax1.bar(x, activ, bw,
                 bottom=np.array(weights) + np.array(grad_mem) + np.array(optim),
                 color=BLUE, label="Activations (batch=1, seq=2048)",
                 edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9.5)
    ax1.set_ylabel("VRAM (GB)")
    ax1.set_title("Training memory breakdown — LLaMA-7B", loc="left")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(0, 105)

    totals = [w + g + o + a for w, g, o, a in
              zip(weights, grad_mem, optim, activ)]
    for xi, t in zip(x, totals):
        ax1.text(xi, t + 2, f"{t:.1f} GB",
                 ha="center", fontsize=10.5, weight="bold", color=DARK)

    # Right: VRAM vs model scale (7B / 13B / 70B)
    sizes = ["7B", "13B", "70B"]
    full_vram = [96, 180, 970]
    lora_vram = [26, 48, 240]
    qlora_vram = [16, 28, 48]

    xs = np.arange(len(sizes))
    bw2 = 0.27
    ax2.bar(xs - bw2, full_vram, bw2, color=RED, label="Full FT",
            edgecolor="white")
    ax2.bar(xs, lora_vram, bw2, color=GREEN, label="LoRA",
            edgecolor="white")
    ax2.bar(xs + bw2, qlora_vram, bw2, color=BLUE, label="QLoRA",
            edgecolor="white")

    # consumer GPU lines
    ax2.axhline(24, color=GRAY, ls="--", lw=1, alpha=0.7)
    ax2.text(2.55, 25, "RTX 4090 / A10 (24 GB)", color=GRAY, fontsize=8.5,
             ha="right")
    ax2.axhline(80, color=DARK, ls="--", lw=1, alpha=0.7)
    ax2.text(2.55, 82, "A100 / H100 (80 GB)", color=DARK, fontsize=8.5,
             ha="right")

    ax2.set_yscale("log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(sizes)
    ax2.set_xlabel("Model size")
    ax2.set_ylabel("Peak VRAM (GB, log scale)")
    ax2.set_title("VRAM required to fine-tune", loc="left")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_ylim(8, 1500)

    for xi, vals in zip(xs, zip(full_vram, lora_vram, qlora_vram)):
        for off, v, c in zip([-bw2, 0, bw2], vals, [RED, GREEN, BLUE]):
            ax2.text(xi + off, v * 1.1, f"{v}", ha="center",
                     fontsize=8.5, color=c, weight="bold")

    fig.suptitle("Memory: full fine-tuning vs LoRA vs QLoRA",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_memory.png")


# ---------------------------------------------------------------------------
# Figure 7 — Performance vs trainable parameter ratio
# ---------------------------------------------------------------------------
def fig7_perf_vs_params() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Trainable parameter ratio (%) vs avg score on GLUE-like benchmark
    # numbers reflect commonly reported PEFT vs FT comparisons
    methods = ["Prompt-Tuning",
               "Prefix-Tuning",
               "BitFit",
               "LoRA r=4",
               "LoRA r=8",
               "Adapter (m=64)",
               "LoRA r=16",
               "LoRA r=64",
               "P-Tuning v2",
               "Full FT"]
    ratios = [0.01, 0.10, 0.08, 0.06, 0.12, 0.50, 0.24, 0.95, 0.20, 100.0]
    scores = [86.5, 88.2, 87.5, 88.8, 89.5, 89.2, 89.8, 90.1, 89.4, 90.4]
    colors = [BLUE, BLUE, ORANGE, GREEN, GREEN, PURPLE,
              GREEN, GREEN, BLUE, RED]

    for r, s, c, m in zip(ratios, scores, colors, methods):
        ax1.scatter(r, s, s=160, color=c, alpha=0.85,
                    edgecolor="white", lw=1.5, zorder=3)
        ha = "left" if r < 5 else "right"
        dx = 1.15 if r < 5 else 0.85
        ax1.annotate(m, (r, s), xytext=(r * dx, s + 0.18),
                     fontsize=9, color=DARK, ha=ha)

    ax1.axhline(scores[-1], color=RED, ls="--", lw=1.2, alpha=0.6)
    ax1.text(0.012, scores[-1] - 0.18, "Full FT baseline",
             color=RED, fontsize=9, style="italic")

    ax1.set_xscale("log")
    ax1.set_xlim(0.005, 300)
    ax1.set_ylim(85.5, 91.0)
    ax1.set_xlabel("Trainable parameters (%, log scale)")
    ax1.set_ylabel("Avg benchmark score")
    ax1.set_title("Performance vs trainable parameter ratio",
                  loc="left")
    ax1.grid(True, alpha=0.3)

    # Right: LoRA rank vs score on three task complexities
    ranks = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    classify = np.array([84.2, 86.8, 88.5, 89.4, 89.8, 90.0, 90.1, 90.1])
    code_gen = np.array([62.0, 67.5, 72.4, 76.2, 79.1, 80.5, 81.2, 81.4])
    domain = np.array([71.5, 74.2, 77.8, 80.6, 82.4, 83.5, 83.9, 84.0])

    ax2.plot(ranks, classify, "-o", color=GREEN, lw=2, ms=6,
             label="Sentiment classification")
    ax2.plot(ranks, code_gen, "-s", color=PURPLE, lw=2, ms=6,
             label="Code generation")
    ax2.plot(ranks, domain, "-^", color=BLUE, lw=2, ms=6,
             label="Domain adaptation (medical)")

    ax2.axvspan(4, 16, color=GREEN, alpha=0.08)
    ax2.text(8, 64, "common\nsweet spot",
             ha="center", fontsize=9, color=GREEN, weight="bold")

    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ranks)
    ax2.set_xticklabels([str(r) for r in ranks])
    ax2.set_xlabel("LoRA rank r")
    ax2.set_ylabel("Task score")
    ax2.set_title("LoRA rank — diminishing returns",
                  loc="left")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("How small can trainable parameters get?",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_perf_vs_params.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating NLP Part 8 figures (Fine-tuning & PEFT) ...")
    fig1_full_vs_peft();    print("  ✓ fig1_full_vs_peft.png")
    fig2_lora_decomp();     print("  ✓ fig2_lora_decomp.png")
    fig3_adapter();         print("  ✓ fig3_adapter.png")
    fig4_prefix_prompt();   print("  ✓ fig4_prefix_prompt.png")
    fig5_qlora();           print("  ✓ fig5_qlora.png")
    fig6_memory();          print("  ✓ fig6_memory.png")
    fig7_perf_vs_params();  print("  ✓ fig7_perf_vs_params.png")
    print(f"\nSaved to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
