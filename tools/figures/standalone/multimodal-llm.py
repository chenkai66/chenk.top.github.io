"""
Figures for the standalone post: Multimodal LLMs and Downstream Tasks.

Generates 7 figures used by both EN and ZH versions. Each figure is rendered
once and saved into both asset folders so the two language versions stay
in sync.

Run:
    python multimodal-llm.py
"""

from __future__ import annotations

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (

    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
ORANGE = COLORS["warning"]
RED = COLORS["danger"]
GREY = COLORS["muted"]
DARK = "#1f2937"
LIGHT = "#f1f5f9"
WHITE = "#ffffff"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/"
    "standalone/multimodal-llm-downstream-tasks"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/"
    "standalone/多模态大模型及下游任务研究"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure into both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def rounded_box(ax, x, y, w, h, text, fc=BLUE, ec=None, tc=WHITE,
                fontsize=11, fontweight="bold", alpha=1.0, pad=0.08):
    if ec is None:
        ec = fc
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        linewidth=1.5, edgecolor=ec, facecolor=fc, alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", color=tc,
            fontsize=fontsize, fontweight=fontweight)


def arrow(ax, p1, p2, color=DARK, lw=1.8, style="->",
          rad=0.0, mutation=18):
    a = FancyArrowPatch(
        p1, p2,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw,
        mutation_scale=mutation,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: Multimodal LLM architecture (Vision Encoder + Projector + LLM)
# ---------------------------------------------------------------------------

def fig1_mllm_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=DPI)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title(
        "Multimodal LLM Architecture: Vision Encoder + Projector + LLM Decoder",
        fontsize=14, fontweight="bold", pad=14, color=DARK,
    )

    # Image input (left, top row)
    rounded_box(ax, 0.3, 4.6, 1.6, 1.0, "Image",
                fc=LIGHT, tc=DARK, fontsize=11)
    ax.text(1.1, 4.45, "(H x W x 3)", ha="center", va="top",
            fontsize=8, color=GREY, style="italic")

    # Vision encoder
    rounded_box(ax, 2.4, 4.4, 2.0, 1.4, "Vision\nEncoder",
                fc=BLUE, fontsize=12)
    ax.text(3.4, 4.25, "ViT-L/14 frozen", ha="center", va="top",
            fontsize=8, color=GREY, style="italic")

    # Patch tokens
    rounded_box(ax, 4.9, 4.6, 1.6, 1.0, "Patch\nTokens",
                fc=LIGHT, tc=DARK, fontsize=10)
    ax.text(5.7, 4.45, "(N_v x d_v)", ha="center", va="top",
            fontsize=8, color=GREY, style="italic")

    # Projector / Adapter (the trainable bottleneck)
    rounded_box(ax, 7.0, 4.4, 2.0, 1.4, "Projector\n(MLP / Q-Former)",
                fc=ORANGE, fontsize=11)
    ax.text(8.0, 4.25, "TRAINABLE", ha="center", va="top",
            fontsize=8, color=ORANGE, fontweight="bold")

    # Visual tokens in LLM space
    rounded_box(ax, 9.5, 4.6, 1.7, 1.0, "Visual\nTokens",
                fc=LIGHT, tc=DARK, fontsize=10)
    ax.text(10.35, 4.45, "(N_v x d_llm)", ha="center", va="top",
            fontsize=8, color=GREY, style="italic")

    # arrows top row
    for x1, x2 in [(1.9, 2.4), (4.4, 4.9), (6.5, 7.0), (9.0, 9.5)]:
        arrow(ax, (x1, 5.1), (x2, 5.1), color=DARK, lw=1.8)

    # Text input (left middle)
    rounded_box(ax, 0.3, 2.4, 1.6, 1.0, "Text\nPrompt",
                fc=LIGHT, tc=DARK, fontsize=11)

    rounded_box(ax, 2.4, 2.4, 2.0, 1.0, "Tokenizer +\nText Embed",
                fc=PURPLE, fontsize=11)

    rounded_box(ax, 4.9, 2.4, 1.6, 1.0, "Text\nTokens",
                fc=LIGHT, tc=DARK, fontsize=10)
    ax.text(5.7, 2.25, "(N_t x d_llm)", ha="center", va="top",
            fontsize=8, color=GREY, style="italic")

    arrow(ax, (1.9, 2.9), (2.4, 2.9), color=DARK, lw=1.8)
    arrow(ax, (4.4, 2.9), (4.9, 2.9), color=DARK, lw=1.8)

    # Concatenate -> LLM
    rounded_box(ax, 7.0, 2.4, 2.0, 1.0,
                "Concatenate\n[<img>] || [text]", fc=GREY, fontsize=10)
    arrow(ax, (6.5, 2.9), (7.0, 2.9), color=DARK, lw=1.8)
    # visual tokens feed in too
    arrow(ax, (10.35, 4.6), (8.0, 3.5), color=BLUE, lw=1.6, rad=-0.3)

    # LLM Decoder (large box, right)
    rounded_box(ax, 7.0, 0.4, 4.2, 1.6,
                "LLM Decoder (Llama / Qwen / Mistral)",
                fc=PURPLE, fontsize=12)
    ax.text(9.1, 0.25, "Frozen or LoRA", ha="center", va="top",
            fontsize=8, color=PURPLE, fontweight="bold", style="italic")

    arrow(ax, (8.0, 2.4), (8.0, 2.0), color=DARK, lw=1.8)

    # Output
    rounded_box(ax, 2.4, 0.4, 3.6, 1.0,
                "Generated Text  ->  caption / answer / reasoning",
                fc=GREEN, fontsize=11)
    arrow(ax, (7.0, 1.2), (6.0, 0.9), color=DARK, lw=1.8)

    # Legend
    legend_handles = [
        mpatches.Patch(color=BLUE, label="Vision (frozen)"),
        mpatches.Patch(color=ORANGE, label="Adapter (trainable)"),
        mpatches.Patch(color=PURPLE, label="Language model"),
        mpatches.Patch(color=GREEN, label="Output"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              ncol=4, frameon=True, fontsize=9)

    save(fig, "fig1_mllm_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2: CLIP / BLIP / LLaVA family comparison
# ---------------------------------------------------------------------------

def fig2_family_comparison() -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=DPI)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "Vision-Language Model Families: CLIP vs BLIP-2 vs LLaVA",
        fontsize=14, fontweight="bold", pad=14, color=DARK,
    )

    col_w = 3.6
    col_x = [0.3, 4.2, 8.1]
    col_titles = ["CLIP (2021)", "BLIP-2 (2023)", "LLaVA (2023)"]
    col_colors = [BLUE, ORANGE, PURPLE]
    col_subtitle = [
        "Dual encoder, contrastive",
        "Frozen LM + Q-Former",
        "Frozen ViT + MLP + Llama",
    ]

    for i, (x, title, c, sub) in enumerate(
            zip(col_x, col_titles, col_colors, col_subtitle)):
        # header
        rounded_box(ax, x, 6.0, col_w, 0.7, title,
                    fc=c, fontsize=12)
        ax.text(x + col_w / 2, 5.7, sub, ha="center", va="top",
                fontsize=9, color=DARK, style="italic")

    # CLIP column
    x = col_x[0]
    rounded_box(ax, x + 0.2, 4.4, 1.4, 0.7, "Image", fc=LIGHT, tc=DARK, fontsize=9)
    rounded_box(ax, x + 2.0, 4.4, 1.4, 0.7, "Text", fc=LIGHT, tc=DARK, fontsize=9)
    rounded_box(ax, x + 0.2, 3.3, 1.4, 0.8, "ViT", fc=BLUE, fontsize=10)
    rounded_box(ax, x + 2.0, 3.3, 1.4, 0.8, "Text Enc", fc=BLUE, fontsize=10)
    arrow(ax, (x + 0.9, 4.4), (x + 0.9, 4.1), color=DARK)
    arrow(ax, (x + 2.7, 4.4), (x + 2.7, 4.1), color=DARK)
    rounded_box(ax, x + 0.2, 2.0, 3.2, 0.9,
                "Contrastive Loss\n(I-T similarity)", fc=GREEN, fontsize=10)
    arrow(ax, (x + 0.9, 3.3), (x + 1.4, 2.9), color=DARK)
    arrow(ax, (x + 2.7, 3.3), (x + 2.2, 2.9), color=DARK)
    ax.text(x + col_w / 2, 1.5,
            "Use:  zero-shot classify\n      retrieval, embed",
            ha="center", va="top", fontsize=9, color=DARK)
    ax.text(x + col_w / 2, 0.45,
            "400M image-text pairs\nNo generation",
            ha="center", va="top", fontsize=8.5, color=GREY, style="italic")

    # BLIP-2 column
    x = col_x[1]
    rounded_box(ax, x + 0.2, 4.4, 1.4, 0.7, "Image", fc=LIGHT, tc=DARK, fontsize=9)
    rounded_box(ax, x + 0.2, 3.3, 1.4, 0.8, "ViT (frozen)", fc=GREY, fontsize=9)
    arrow(ax, (x + 0.9, 4.4), (x + 0.9, 4.1), color=DARK)
    rounded_box(ax, x + 2.0, 3.3, 1.4, 0.8, "Q-Former\n(trainable)", fc=ORANGE, fontsize=9)
    arrow(ax, (x + 1.6, 3.7), (x + 2.0, 3.7), color=DARK)
    rounded_box(ax, x + 0.2, 2.0, 3.2, 0.9,
                "Frozen LLM\n(OPT / Flan-T5)", fc=PURPLE, fontsize=10)
    arrow(ax, (x + 2.7, 3.3), (x + 2.0, 2.9), color=DARK)
    ax.text(x + col_w / 2, 1.5,
            "Use:  VQA, captioning\n      grounded chat",
            ha="center", va="top", fontsize=9, color=DARK)
    ax.text(x + col_w / 2, 0.45,
            "Bridge frozen models\n~188M trainable",
            ha="center", va="top", fontsize=8.5, color=GREY, style="italic")

    # LLaVA column
    x = col_x[2]
    rounded_box(ax, x + 0.2, 4.4, 1.4, 0.7, "Image", fc=LIGHT, tc=DARK, fontsize=9)
    rounded_box(ax, x + 0.2, 3.3, 1.4, 0.8, "CLIP ViT\n(frozen)", fc=GREY, fontsize=9)
    arrow(ax, (x + 0.9, 4.4), (x + 0.9, 4.1), color=DARK)
    rounded_box(ax, x + 2.0, 3.3, 1.4, 0.8, "MLP\nProjector", fc=ORANGE, fontsize=9)
    arrow(ax, (x + 1.6, 3.7), (x + 2.0, 3.7), color=DARK)
    rounded_box(ax, x + 0.2, 2.0, 3.2, 0.9,
                "Llama / Vicuna\n(SFT or LoRA)", fc=PURPLE, fontsize=10)
    arrow(ax, (x + 2.7, 3.3), (x + 2.0, 2.9), color=DARK)
    ax.text(x + col_w / 2, 1.5,
            "Use:  instruction-follow\n      chat over images",
            ha="center", va="top", fontsize=9, color=DARK)
    ax.text(x + col_w / 2, 0.45,
            "GPT-4 generated SFT\nSimple, scalable recipe",
            ha="center", va="top", fontsize=8.5, color=GREY, style="italic")

    save(fig, "fig2_family_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3: Vision-Language tasks (VQA, captioning, grounding, OCR)
# ---------------------------------------------------------------------------

def fig3_vl_tasks() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=DPI)
    fig.suptitle("Vision-Language Downstream Tasks",
                 fontsize=15, fontweight="bold", y=0.995, color=DARK)

    # -- (a) Image Captioning --
    ax = axes[0, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("(a) Image Captioning", fontsize=12,
                 fontweight="bold", color=BLUE)
    # mock image
    rect = Rectangle((0.5, 3.0), 4.0, 3.0, facecolor=LIGHT,
                     edgecolor=DARK, linewidth=1.5)
    ax.add_patch(rect)
    # mock objects in image
    ax.add_patch(Circle((1.7, 4.5), 0.55, color=ORANGE))  # ball
    ax.add_patch(Rectangle((2.7, 3.4), 1.4, 1.0,
                           color=GREEN, alpha=0.8))      # dog body
    ax.add_patch(Circle((4.0, 4.7), 0.35, color=GREEN))  # head
    ax.text(2.5, 6.3, "input image", ha="center", fontsize=9,
            color=GREY, style="italic")

    arrow(ax, (4.6, 4.5), (5.4, 4.5), color=DARK, lw=2)

    rounded_box(ax, 5.5, 3.8, 4.2, 0.9, "MLLM",
                fc=PURPLE, fontsize=11)
    arrow(ax, (7.6, 3.8), (7.6, 3.1), color=DARK, lw=2)
    ax.text(7.6, 2.6,
            '"A green dog is\nplaying with an\norange ball."',
            ha="center", va="top", fontsize=10, color=DARK,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=LIGHT, edgecolor=GREY))
    ax.text(5.0, 0.3, "Metric: BLEU, CIDEr, SPICE",
            ha="center", fontsize=9, color=GREY, style="italic")

    # -- (b) VQA --
    ax = axes[0, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("(b) Visual Question Answering",
                 fontsize=12, fontweight="bold", color=ORANGE)
    rect = Rectangle((0.5, 3.0), 4.0, 3.0, facecolor=LIGHT,
                     edgecolor=DARK, linewidth=1.5)
    ax.add_patch(rect)
    ax.add_patch(Circle((1.7, 4.5), 0.55, color=ORANGE))
    ax.add_patch(Rectangle((2.7, 3.4), 1.4, 1.0,
                           color=GREEN, alpha=0.8))
    ax.add_patch(Circle((4.0, 4.7), 0.35, color=GREEN))
    ax.text(2.5, 2.5,
            'Q: "What color is the ball?"',
            ha="center", fontsize=10, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=LIGHT, edgecolor=ORANGE))

    arrow(ax, (4.6, 4.5), (5.4, 4.5), color=DARK, lw=2)
    rounded_box(ax, 5.5, 3.8, 4.2, 0.9, "MLLM",
                fc=PURPLE, fontsize=11)
    arrow(ax, (7.6, 3.8), (7.6, 3.1), color=DARK, lw=2)
    ax.text(7.6, 2.6, 'A: "Orange."',
            ha="center", va="top", fontsize=11, color=DARK,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=LIGHT, edgecolor=GREEN))
    ax.text(5.0, 0.3, "Metric: VQA-acc, MMBench, MMMU",
            ha="center", fontsize=9, color=GREY, style="italic")

    # -- (c) Visual Grounding --
    ax = axes[1, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("(c) Visual Grounding (REC)",
                 fontsize=12, fontweight="bold", color=GREEN)
    rect = Rectangle((0.5, 2.5), 4.5, 3.5, facecolor=LIGHT,
                     edgecolor=DARK, linewidth=1.5)
    ax.add_patch(rect)
    # 2 dogs and 1 ball
    ax.add_patch(Circle((1.7, 4.0), 0.45, color=GREEN))
    ax.add_patch(Circle((3.7, 4.5), 0.45, color=BLUE))
    ax.add_patch(Circle((2.7, 3.2), 0.35, color=ORANGE))
    # bounding box (predicted)
    bbox = Rectangle((3.15, 4.0), 1.1, 1.05, fill=False,
                     edgecolor=RED, linewidth=2.5, linestyle="--")
    ax.add_patch(bbox)
    ax.text(3.7, 5.25, "predicted bbox", ha="center",
            fontsize=8, color=RED, fontweight="bold")
    ax.text(2.75, 1.9,
            'Query: "the blue dog"',
            ha="center", fontsize=10, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=LIGHT, edgecolor=GREEN))

    arrow(ax, (5.2, 4.3), (6.0, 4.3), color=DARK, lw=2)
    rounded_box(ax, 6.1, 3.7, 3.7, 1.2,
                "Output: (x,y,w,h)\n[3.15, 4.0, 1.1, 1.05]",
                fc=GREEN, fontsize=9)
    ax.text(5.0, 0.5, "Metric: IoU, Acc@0.5",
            ha="center", fontsize=9, color=GREY, style="italic")

    # -- (d) OCR / Document --
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("(d) OCR / Document Understanding",
                 fontsize=12, fontweight="bold", color=PURPLE)
    rect = Rectangle((0.5, 2.5), 4.0, 3.8, facecolor=WHITE,
                     edgecolor=DARK, linewidth=1.5)
    ax.add_patch(rect)
    # mock text lines
    for i, line in enumerate([
            "INVOICE #2026-04",
            "------------------",
            "Item A    USD 12.00",
            "Item B    USD  7.50",
            "------------------",
            "Total     USD 19.50"]):
        ax.text(0.7, 5.9 - i * 0.55, line, fontsize=8.5,
                family="monospace", color=DARK)

    arrow(ax, (4.6, 4.3), (5.4, 4.3), color=DARK, lw=2)
    rounded_box(ax, 5.5, 3.7, 4.2, 1.2,
                'Q: "What is the total?"\nA: "USD 19.50"',
                fc=PURPLE, fontsize=9)
    ax.text(5.0, 0.5, "Metric: ANLS, exact-match (DocVQA)",
            ha="center", fontsize=9, color=GREY, style="italic")

    plt.tight_layout()
    save(fig, "fig3_vl_tasks.png")


# ---------------------------------------------------------------------------
# Figure 4: Cross-modal alignment (contrastive embedding space)
# ---------------------------------------------------------------------------

def fig4_alignment() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=DPI)
    fig.suptitle(
        "Cross-Modal Alignment: contrastive learning shapes a shared space",
        fontsize=14, fontweight="bold", y=1.00, color=DARK,
    )

    # -- left: similarity matrix N x N --
    ax = axes[0]
    n = 6
    rng = np.random.default_rng(7)
    # diagonal high, off-diagonal low
    sim = rng.uniform(-0.2, 0.3, size=(n, n))
    np.fill_diagonal(sim, rng.uniform(0.7, 0.95, size=n))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"T{i+1}" for i in range(n)], fontsize=10)
    ax.set_yticklabels([f"I{i+1}" for i in range(n)], fontsize=10)
    ax.set_xlabel("Text embeddings", fontsize=11, color=DARK)
    ax.set_ylabel("Image embeddings", fontsize=11, color=DARK)
    ax.set_title("Image-Text similarity matrix\n(diagonal = matched pairs)",
                 fontsize=11, color=DARK)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center",
                    va="center", fontsize=8.5,
                    color=(WHITE if abs(sim[i, j]) > 0.45 else DARK))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cos similarity")

    # -- right: 2D embedding space --
    ax = axes[1]
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_title("Shared embedding space (2D projection)",
                 fontsize=11, color=DARK)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.3)

    centers = np.array([
        [-2.0,  1.5],   # cluster 1: dog
        [ 2.0,  1.7],   # cluster 2: cat
        [-1.8, -1.6],   # cluster 3: car
        [ 1.9, -1.4],   # cluster 4: airplane
    ])
    labels = ["dog", "cat", "car", "airplane"]
    colors = [BLUE, PURPLE, GREEN, ORANGE]

    for c, lbl, col in zip(centers, labels, colors):
        # 5 image points
        img_pts = c + rng.normal(0, 0.32, size=(5, 2))
        # 5 text points, slightly offset
        txt_pts = c + np.array([0.25, 0.2]) + rng.normal(0, 0.32, size=(5, 2))
        ax.scatter(img_pts[:, 0], img_pts[:, 1], s=80, color=col,
                   marker="o", edgecolor=DARK, linewidth=0.6,
                   label=f"image: {lbl}" if lbl == "dog" else None)
        ax.scatter(txt_pts[:, 0], txt_pts[:, 1], s=110, color=col,
                   marker="*", edgecolor=DARK, linewidth=0.6,
                   label=f"text: {lbl}" if lbl == "dog" else None)
        # cluster annotation
        ax.text(c[0], c[1] + 0.95, lbl, ha="center", fontsize=10,
                color=col, fontweight="bold")

    # Legend (markers only, single example)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=GREY, markeredgecolor=DARK,
                   markersize=10, label="image embedding"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=GREY, markeredgecolor=DARK,
                   markersize=14, label="text embedding"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True)

    plt.tight_layout()
    save(fig, "fig4_cross_modal_alignment.png")


# ---------------------------------------------------------------------------
# Figure 5: Fine-tuning strategies (Full / LoRA / Projector-only)
# ---------------------------------------------------------------------------

def fig5_finetuning() -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=DPI)
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title(
        "Fine-Tuning Strategies for MLLMs: trade-off between cost and capacity",
        fontsize=14, fontweight="bold", pad=14, color=DARK,
    )

    strategies = [
        ("Full fine-tune", 0.3, RED,
         ["ViT", "Projector", "LLM"], [True, True, True],
         "100% params\nbest fit, max cost\nrisk: catastrophic\nforgetting"),
        ("LoRA on LLM", 4.2, ORANGE,
         ["ViT", "Projector", "LLM"], [False, True, "lora"],
         "~0.5% params\nbest cost/quality\ndefault choice"),
        ("Projector only", 8.1, GREEN,
         ["ViT", "Projector", "LLM"], [False, True, False],
         "<0.1% params\nstage-1 alignment\nlimited reasoning"),
    ]

    for name, x0, color, stack, train, note in strategies:
        # title
        rounded_box(ax, x0, 5.7, 3.5, 0.7, name, fc=color, fontsize=12)
        # 3 stacked components
        for i, (comp, t) in enumerate(zip(stack, train)):
            y = 4.6 - i * 0.95
            if t is True:
                fc, tc, tag = color, WHITE, "train"
            elif t == "lora":
                fc, tc, tag = ORANGE, WHITE, "+ LoRA"
            else:
                fc, tc, tag = GREY, WHITE, "frozen"
            rounded_box(ax, x0 + 0.2, y, 3.1, 0.75,
                        f"{comp}  ({tag})", fc=fc, tc=tc, fontsize=10)
        # note
        ax.text(x0 + 1.75, 1.4, note, ha="center", va="top",
                fontsize=9.5, color=DARK,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=LIGHT, edgecolor=color, linewidth=1.5))

    # Bottom axis: trainable parameter %
    ax.text(6.0, 0.25, "Trainable parameter %:   100%   ->   ~0.5%   ->   <0.1%",
            ha="center", fontsize=10, color=DARK, style="italic")

    save(fig, "fig5_finetuning_strategies.png")


# ---------------------------------------------------------------------------
# Figure 6: Benchmark results (MMBench, MMMU, etc.)
# ---------------------------------------------------------------------------

def fig6_benchmarks() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=DPI)

    # -- (a) Bar chart: model x benchmark --
    ax = axes[0]
    models = ["LLaVA-1.5\n7B", "LLaVA-1.5\n13B", "Qwen-VL\nChat",
              "InternVL\n1.5", "GPT-4V", "GPT-4o"]
    mmbench = [64.3, 67.7, 60.6, 82.2, 75.8, 83.4]
    mmmu    = [35.7, 36.4, 35.9, 45.2, 56.8, 69.1]

    x = np.arange(len(models))
    w = 0.38
    ax.bar(x - w/2, mmbench, w, color=BLUE, label="MMBench (dev)")
    ax.bar(x + w/2, mmmu, w, color=PURPLE, label="MMMU (val)")

    for i, v in enumerate(mmbench):
        ax.text(i - w/2, v + 0.7, f"{v}", ha="center", fontsize=8.5,
                color=BLUE, fontweight="bold")
    for i, v in enumerate(mmmu):
        ax.text(i + w/2, v + 0.7, f"{v}", ha="center", fontsize=8.5,
                color=PURPLE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 95)
    ax.set_title("(a) Benchmark scores: MMBench vs MMMU",
                 fontsize=12, fontweight="bold", color=DARK)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # -- (b) Capability radar (5 dims) --
    ax = axes[1]
    ax.remove()
    ax = fig.add_subplot(1, 2, 2, projection="polar")
    cats = ["Perception", "Reasoning", "OCR", "Math",
            "Knowledge", "Hallucination\n(higher = better)"]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    series = [
        ("LLaVA-1.5-13B", [73, 60, 48, 35, 60, 55], BLUE),
        ("Qwen-VL-Max",   [82, 72, 80, 55, 70, 70], ORANGE),
        ("GPT-4o",        [88, 85, 82, 78, 85, 80], PURPLE),
    ]
    for name, vals, col in series:
        v = vals + vals[:1]
        ax.plot(angles, v, color=col, linewidth=2, label=name)
        ax.fill(angles, v, color=col, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color=GREY)
    ax.set_title("(b) Capability profile across 6 dimensions",
                 fontsize=12, fontweight="bold", color=DARK, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=9)

    plt.tight_layout()
    save(fig, "fig6_benchmarks.png")


# ---------------------------------------------------------------------------
# Figure 7: Production deployment considerations
# ---------------------------------------------------------------------------

def fig7_deployment() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=DPI)
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.set_title(
        "Production Deployment: latency, cost, safety, observability",
        fontsize=14, fontweight="bold", pad=14, color=DARK,
    )

    # Pipeline boxes
    rounded_box(ax, 0.3, 5.4, 1.9, 1.0, "Image\nUpload", fc=LIGHT, tc=DARK, fontsize=10)
    rounded_box(ax, 2.5, 5.4, 2.0, 1.0, "Pre-process\nresize, OCR detect",
                fc=BLUE, fontsize=10)
    rounded_box(ax, 4.8, 5.4, 2.0, 1.0, "Vision Encoder\n(GPU, batched)",
                fc=BLUE, fontsize=10)
    rounded_box(ax, 7.1, 5.4, 2.0, 1.0, "Cache lookup\nimage embedding",
                fc=ORANGE, fontsize=10)
    rounded_box(ax, 9.4, 5.4, 1.9, 1.0, "MLLM Decode\nKV cache",
                fc=PURPLE, fontsize=10)
    rounded_box(ax, 11.5, 5.4, 1.4, 1.0, "Stream\nresponse",
                fc=GREEN, fontsize=10)
    for x in [2.2, 4.5, 6.8, 9.1, 11.3]:
        arrow(ax, (x, 5.9), (x + 0.2, 5.9), color=DARK, lw=1.6)

    # Latency budget bar
    ax.text(6.5, 4.6, "Latency budget (P95, target 1.5 s):",
            fontsize=11, color=DARK, ha="center", fontweight="bold")
    segments = [
        ("Pre",  0.05, BLUE),
        ("Encode", 0.25, BLUE),
        ("Cache", 0.05, ORANGE),
        ("Prefill", 0.45, PURPLE),
        ("Decode (stream)", 0.65, GREEN),
        ("Net", 0.05, GREY),
    ]
    total_w = 12.0
    cur = 0.5
    total_t = sum(s[1] for s in segments)
    for label, t, col in segments:
        w = (t / total_t) * total_w
        ax.add_patch(Rectangle((cur, 3.7), w, 0.6,
                               facecolor=col, edgecolor=DARK, linewidth=1))
        ax.text(cur + w / 2, 4.0, f"{label}\n{int(t*1000)} ms",
                ha="center", va="center", fontsize=8.5,
                color=WHITE, fontweight="bold")
        cur += w

    # Three concern columns
    concerns = [
        ("Cost", BLUE, [
            "$ / image: encode + tokens",
            "Visual tokens dominate prompt",
            "Lever: image resize + KV cache",
            "Lever: batch + speculative decode",
        ]),
        ("Safety", ORANGE, [
            "Prompt injection in images",
            "PII in screenshots / OCR",
            "Lever: input filter + redaction",
            "Lever: output classifier",
        ]),
        ("Observability", PURPLE, [
            "Token / latency / error logs",
            "Per-task quality eval",
            "Lever: shadow traffic + canary",
            "Lever: hallucination detector",
        ]),
    ]
    col_w = 4.0
    for i, (title, col, items) in enumerate(concerns):
        x0 = 0.3 + i * 4.3
        rounded_box(ax, x0, 2.5, col_w, 0.7, title,
                    fc=col, fontsize=12)
        for j, item in enumerate(items):
            ax.text(x0 + 0.15, 2.1 - j * 0.45, "- " + item,
                    fontsize=9, color=DARK, va="top")

    save(fig, "fig7_production_deployment.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_mllm_architecture()
    fig2_family_comparison()
    fig3_vl_tasks()
    fig4_alignment()
    fig5_finetuning()
    fig6_benchmarks()
    fig7_deployment()
    print("OK: all 7 figures saved to:")
    print("  EN:", EN_DIR)
    print("  ZH:", ZH_DIR)


if __name__ == "__main__":
    main()
