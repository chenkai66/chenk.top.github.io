"""
Figure generation script for Transfer Learning Part 08: Multimodal Transfer.

Generates 7 figures used in both EN and ZH versions of the article. Each figure
teaches one specific concept of vision-language transfer learning.

Figures:
    fig1_clip_architecture        CLIP dual-encoder layout. Image encoder
                                  (ViT-style) and text encoder (Transformer)
                                  project to a shared L2-normalised space.
    fig2_contrastive_alignment    Contrastive matching matrix: B image-text
                                  pairs form a B x B similarity grid; the
                                  diagonal (positives) is pulled up, off-diag
                                  (negatives) pushed down.
    fig3_blip_architecture        BLIP / BLIP-2 architecture: image encoder +
                                  Q-Former + LLM, with three losses (ITC, ITM,
                                  ITG) annotated.
    fig4_cross_modal_retrieval    Image->text and text->image retrieval ranked
                                  lists with similarity bars.
    fig5_embedding_tsne           t-SNE-style 2D plot of joint multimodal
                                  embeddings: image points + text points
                                  cluster by semantic concept.
    fig6_vl_tasks                 Vision-language downstream task gallery:
                                  VQA, Captioning, Retrieval, Grounding -
                                  each shown as a mini diagram.
    fig7_fusion_strategies        Early / Late / Cross-attention fusion side
                                  by side, with information-flow arrows.

Usage:
    python3 scripts/figures/transfer-learning/08-multimodal.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

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
C_BLUE = COLORS["primary"]     # image / vision modality
C_PURPLE = COLORS["accent"]   # text / language modality
C_GREEN = COLORS["success"]    # positive / aligned
C_AMBER = COLORS["warning"]    # highlight / attention
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = COLORS["bg"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "transfer-learning" / "08-multimodal-transfer"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "transfer-learning" / "08-多模态迁移"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, x, y, w, h, text, *, fc, ec=None, fontsize=10, fontcolor="white",
         weight="bold", radius=0.04):
    """Helper: draw a rounded box with centered text."""
    if ec is None:
        ec = fc
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=1.5, facecolor=fc, edgecolor=ec, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=fontcolor, weight=weight, zorder=3)


def _arrow(ax, x1, y1, x2, y2, *, color=C_DARK, lw=1.6, style="->", mutation=15):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation,
        color=color, linewidth=lw, zorder=1,
    )
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Figure 1: CLIP dual-encoder architecture
# ---------------------------------------------------------------------------
def fig1_clip_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()

    # Title
    ax.text(5.5, 5.9, "CLIP: Dual-Encoder Vision-Language Pretraining",
            ha="center", fontsize=14, weight="bold", color=C_DARK)

    # Image branch (top)
    ax.text(0.6, 4.7, "Image", fontsize=11, color=C_BLUE, weight="bold")
    # mock image patch grid
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((0.4 + j * 0.35, 3.4 + i * 0.35), 0.32, 0.32,
                                   facecolor=C_BLUE, alpha=0.15 + 0.1 * ((i + j) % 3),
                                   edgecolor=C_BLUE, linewidth=0.6))
    ax.text(0.92, 3.2, "patch tokens", ha="center", fontsize=8, color=C_GRAY, style="italic")

    _arrow(ax, 1.7, 4.05, 2.6, 4.05, color=C_BLUE)
    _box(ax, 2.6, 3.5, 2.0, 1.1, "Image Encoder\n(ViT / ResNet)", fc=C_BLUE, fontsize=10)
    _arrow(ax, 4.6, 4.05, 5.5, 4.05, color=C_BLUE)
    _box(ax, 5.5, 3.6, 1.6, 0.9, r"$\mathbf{v}_i \in \mathbb{R}^d$",
         fc="white", ec=C_BLUE, fontcolor=C_BLUE, fontsize=11)
    _arrow(ax, 7.1, 4.05, 8.0, 4.05, color=C_BLUE)
    ax.text(7.55, 4.2, "L2 norm", ha="center", fontsize=8, color=C_GRAY, style="italic")

    # Text branch (bottom)
    ax.text(0.6, 2.55, "Text", fontsize=11, color=C_PURPLE, weight="bold")
    ax.text(0.92, 1.6, '"a photo of\na golden retriever"',
            ha="center", fontsize=9, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_PURPLE, linewidth=1.2))
    _arrow(ax, 1.7, 1.7, 2.6, 1.7, color=C_PURPLE)
    _box(ax, 2.6, 1.15, 2.0, 1.1, "Text Encoder\n(Transformer)", fc=C_PURPLE, fontsize=10)
    _arrow(ax, 4.6, 1.7, 5.5, 1.7, color=C_PURPLE)
    _box(ax, 5.5, 1.25, 1.6, 0.9, r"$\mathbf{t}_i \in \mathbb{R}^d$",
         fc="white", ec=C_PURPLE, fontcolor=C_PURPLE, fontsize=11)
    _arrow(ax, 7.1, 1.7, 8.0, 1.7, color=C_PURPLE)
    ax.text(7.55, 1.85, "L2 norm", ha="center", fontsize=8, color=C_GRAY, style="italic")

    # Shared embedding space (right)
    _box(ax, 8.0, 1.55, 2.7, 2.85, "", fc=C_BG, ec=C_DARK, fontcolor=C_DARK)
    ax.text(9.35, 4.2, "Shared embedding\nspace", ha="center",
            fontsize=10, color=C_DARK, weight="bold")
    rng = np.random.default_rng(1)
    for _ in range(7):
        x = 8.2 + rng.uniform(0, 2.3)
        y = 1.75 + rng.uniform(0, 1.7)
        ax.scatter(x, y, s=45, color=C_BLUE, alpha=0.7, zorder=4)
        ax.scatter(x + rng.uniform(-0.07, 0.07), y + rng.uniform(-0.07, 0.07),
                   s=45, marker="^", color=C_PURPLE, alpha=0.7, zorder=4)
    ax.text(9.35, 1.65, r"image $\bullet$    text $\blacktriangle$",
            ha="center", fontsize=9, color=C_DARK)

    # Loss annotation
    ax.text(5.5, 0.55,
            r"Symmetric InfoNCE:  $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i})$,  with $\tau = 0.07$",
            ha="center", fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_GRAY, linewidth=1))

    _save(fig, "fig1_clip_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Contrastive image-text alignment matrix
# ---------------------------------------------------------------------------
def fig2_contrastive_alignment() -> None:
    rng = np.random.default_rng(3)
    B = 8
    # base random similarity matrix
    sim = rng.uniform(0.05, 0.35, size=(B, B))
    # boost the diagonal: positives
    for i in range(B):
        sim[i, i] = rng.uniform(0.78, 0.95)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5),
                             gridspec_kw=dict(width_ratios=[1.05, 1]))

    # --- Left: similarity matrix heatmap ---
    ax = axes[0]
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
    for i in range(B):
        for j in range(B):
            v = sim[i, j]
            color = "white" if v > 0.55 else C_DARK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    # highlight diagonal (positives)
    for i in range(B):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor=C_GREEN, linewidth=2.5, zorder=4))

    ax.set_xticks(range(B))
    ax.set_yticks(range(B))
    ax.set_xticklabels([f"$T_{i+1}$" for i in range(B)], fontsize=10)
    ax.set_yticklabels([f"$I_{i+1}$" for i in range(B)], fontsize=10)
    ax.set_xlabel("Text encodings", fontsize=11)
    ax.set_ylabel("Image encodings", fontsize=11)
    ax.set_title("Batch similarity matrix  $S_{ij}=\\mathbf{v}_i^\\top \\mathbf{t}_j$",
                 fontsize=12, weight="bold", color=C_DARK)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("cosine similarity", fontsize=9)

    # --- Right: pull/push intuition ---
    ax = axes[1]
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Contrastive objective in embedding space",
                 fontsize=12, weight="bold", color=C_DARK, pad=10)

    # anchor image
    ax.scatter(0, 0, s=320, color=C_BLUE, zorder=5, edgecolor="white", linewidth=2)
    ax.text(0, -0.55, "image $\\mathbf{v}_i$", ha="center", fontsize=10, color=C_BLUE, weight="bold")

    # positive text close
    ax.scatter(0.85, 0.5, s=260, marker="^", color=C_GREEN, zorder=5,
               edgecolor="white", linewidth=2)
    ax.text(1.35, 0.85, "$\\mathbf{t}_i$ (positive)", fontsize=10, color=C_GREEN, weight="bold")
    _arrow(ax, 0.15, 0.05, 0.7, 0.42, color=C_GREEN, lw=2.2, mutation=18)
    ax.text(0.45, 0.55, "pull", fontsize=9, color=C_GREEN, weight="bold")

    # negatives scattered far
    rng2 = np.random.default_rng(11)
    for _ in range(5):
        ang = rng2.uniform(0.6 * np.pi, 1.9 * np.pi)
        r = rng2.uniform(2.1, 2.8)
        x, y = r * np.cos(ang), r * np.sin(ang)
        ax.scatter(x, y, s=200, marker="^", color=C_PURPLE, alpha=0.55,
                   zorder=4, edgecolor="white", linewidth=1.5)
        _arrow(ax, x * 0.55, y * 0.55, x * 0.95, y * 0.95,
               color=C_AMBER, lw=1.6, mutation=14)
    ax.text(-1.7, -2.4, "negatives $\\mathbf{t}_{j \\neq i}$\n(pushed away)",
            ha="center", fontsize=10, color=C_PURPLE)

    fig.suptitle("Contrastive image-text alignment (InfoNCE)",
                 fontsize=14, weight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_contrastive_alignment")


# ---------------------------------------------------------------------------
# Figure 3: BLIP-2 architecture (Q-Former bridge)
# ---------------------------------------------------------------------------
def fig3_blip_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    ax.set_axis_off()

    ax.text(6, 5.45, "BLIP-2: Bootstrapped Vision-Language Pretraining via Q-Former",
            ha="center", fontsize=13.5, weight="bold", color=C_DARK)

    # Frozen Image Encoder
    _box(ax, 0.4, 2.35, 2.2, 1.4, "Image Encoder\n(frozen ViT)", fc=C_BLUE, fontsize=10.5)
    ax.text(1.5, 2.1, "[frozen]", ha="center", fontsize=9, color=C_GRAY, style="italic")
    # tiny image
    ax.add_patch(Rectangle((0.65, 4.05), 1.7, 1.0, facecolor="white",
                           edgecolor=C_BLUE, linewidth=1.5))
    rng = np.random.default_rng(5)
    for _ in range(20):
        ax.scatter(0.7 + rng.uniform(0, 1.6), 4.1 + rng.uniform(0, 0.9),
                   s=12, color=C_BLUE, alpha=0.4)
    _arrow(ax, 1.5, 4.05, 1.5, 3.75, color=C_BLUE)

    # Q-Former (the bridge)
    _box(ax, 3.5, 1.9, 3.0, 2.3, "Q-Former\n(Querying Transformer)\n\nlearned queries\n+ cross-attention",
         fc=C_AMBER, fontsize=10.5)
    _arrow(ax, 2.6, 3.05, 3.5, 3.05, color=C_BLUE, lw=2)
    ax.text(3.05, 3.25, "image\nfeatures", ha="center", fontsize=8, color=C_DARK)

    # Frozen LLM
    _box(ax, 7.6, 2.35, 2.6, 1.4, "Large LLM\n(frozen)", fc=C_PURPLE, fontsize=10.5)
    ax.text(8.9, 2.1, "[frozen]", ha="center", fontsize=9, color=C_GRAY, style="italic")
    _arrow(ax, 6.5, 3.05, 7.6, 3.05, color=C_AMBER, lw=2)
    ax.text(7.05, 3.25, "soft visual\nprompt", ha="center", fontsize=8, color=C_DARK)

    # Output
    _box(ax, 10.6, 2.65, 1.3, 0.8, "text", fc="white", ec=C_PURPLE, fontcolor=C_PURPLE, fontsize=10)
    _arrow(ax, 10.2, 3.05, 10.6, 3.05, color=C_PURPLE)

    # Three losses (training objectives) - bottom annotation
    losses = [
        ("Image-Text\nContrastive (ITC)", C_BLUE),
        ("Image-Text\nMatching (ITM)", C_GREEN),
        ("Image-grounded\nText Generation (ITG)", C_PURPLE),
    ]
    ax.text(5.0, 1.35, "Stage-1 objectives (Q-Former pretraining):",
            ha="center", fontsize=10, color=C_DARK, weight="bold")
    for k, (label, color) in enumerate(losses):
        _box(ax, 0.6 + k * 3.05, 0.25, 2.7, 0.85, label, fc=color, fontsize=9.5)

    # Stage-2 note
    ax.text(10.6, 0.65,
            "Stage 2:\nconnect to LLM\n(generative loss)",
            ha="center", fontsize=9, color=C_PURPLE,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=C_PURPLE, linewidth=1.2))

    _save(fig, "fig3_blip_architecture")


# ---------------------------------------------------------------------------
# Figure 4: Cross-modal retrieval (image<->text)
# ---------------------------------------------------------------------------
def fig4_cross_modal_retrieval() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    # ---- Left: image -> text retrieval ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.set_axis_off()
    ax.set_title("Image $\\rightarrow$ Text retrieval",
                 fontsize=12.5, weight="bold", color=C_BLUE, pad=8)

    # query image
    ax.add_patch(Rectangle((0.5, 3.4), 2.2, 2.4, facecolor=C_LIGHT,
                           edgecolor=C_BLUE, linewidth=2))
    rng = np.random.default_rng(2)
    for _ in range(35):
        ax.scatter(0.6 + rng.uniform(0, 2.0), 3.5 + rng.uniform(0, 2.2),
                   s=14, color=C_BLUE, alpha=0.45)
    ax.text(1.6, 3.05, "query image", ha="center", fontsize=10, color=C_BLUE, weight="bold")

    # arrow
    _arrow(ax, 2.85, 4.6, 3.7, 4.6, color=C_DARK, lw=1.8)
    ax.text(3.27, 4.85, "encode", ha="center", fontsize=9, color=C_GRAY, style="italic")

    # ranked candidate captions with similarity bars
    captions = [
        ("a golden retriever puppy on grass", 0.92, True),
        ("a dog playing in a park",            0.81, False),
        ("a brown puppy on a lawn",            0.74, False),
        ("a cat on a sofa",                    0.31, False),
        ("a red sports car driving fast",      0.18, False),
    ]
    y0 = 5.7
    for i, (cap, score, hit) in enumerate(captions):
        y = y0 - i * 1.05
        col = C_GREEN if hit else C_GRAY
        ax.text(3.95, y + 0.12, f"{i+1}.  {cap}", fontsize=10, color=C_DARK,
                weight="bold" if hit else "normal")
        # bar
        ax.add_patch(Rectangle((3.95, y - 0.32), 5.6 * score, 0.28,
                               facecolor=col, alpha=0.85))
        ax.add_patch(Rectangle((3.95, y - 0.32), 5.6, 0.28,
                               fill=False, edgecolor=C_GRAY, linewidth=0.6))
        ax.text(9.65, y - 0.18, f"{score:.2f}", fontsize=9, color=C_DARK)

    # ---- Right: text -> image retrieval ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.set_axis_off()
    ax.set_title("Text $\\rightarrow$ Image retrieval",
                 fontsize=12.5, weight="bold", color=C_PURPLE, pad=8)

    # query text
    ax.text(1.7, 4.6, '"a small dog\non green grass"',
            ha="center", fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=C_PURPLE, linewidth=2))
    ax.text(1.7, 3.4, "query text", ha="center", fontsize=10, color=C_PURPLE, weight="bold")
    _arrow(ax, 2.85, 4.6, 3.7, 4.6, color=C_DARK, lw=1.8)
    ax.text(3.27, 4.85, "encode", ha="center", fontsize=9, color=C_GRAY, style="italic")

    # ranked candidate thumbnails (5 small swatches)
    rng2 = np.random.default_rng(8)
    scores = [0.94, 0.86, 0.72, 0.41, 0.22]
    is_hit = [True, False, False, False, False]
    for i in range(5):
        x = 4.0 + i * 1.15
        y = 4.2
        col = C_GREEN if is_hit[i] else C_BLUE
        ax.add_patch(Rectangle((x, y), 1.0, 1.0, facecolor=C_LIGHT,
                               edgecolor=col, linewidth=2))
        for _ in range(15):
            ax.scatter(x + 0.05 + rng2.uniform(0, 0.9),
                       y + 0.05 + rng2.uniform(0, 0.9),
                       s=8, color=C_BLUE, alpha=0.4)
        ax.text(x + 0.5, y - 0.25, f"{scores[i]:.2f}",
                ha="center", fontsize=9, color=C_DARK)
        ax.text(x + 0.5, y + 1.25, f"#{i+1}", ha="center",
                fontsize=10, color=col, weight="bold")

    # metric annotation
    ax.text(7.0, 2.4, "Standard metric: $R@K$\n(is the true match in top-$K$?)",
            ha="center", fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_BG,
                      edgecolor=C_GRAY, linewidth=1))

    fig.suptitle("Cross-modal retrieval with a shared embedding space",
                 fontsize=14, weight="bold", color=C_DARK, y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_cross_modal_retrieval")


# ---------------------------------------------------------------------------
# Figure 5: Multimodal embedding space (t-SNE-like)
# ---------------------------------------------------------------------------
def fig5_embedding_tsne() -> None:
    rng = np.random.default_rng(42)
    # 5 semantic clusters; in each cluster mix image (circle) and text (triangle)
    centers = np.array([
        [-3.5,  3.0],   # dog
        [ 3.5,  3.0],   # car
        [-3.5, -3.0],   # beach
        [ 3.5, -3.0],   # mountain
        [ 0.0,  0.0],   # food
    ])
    labels = ["dog", "car", "beach", "mountain", "food"]
    cluster_colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, "#ec4899"]

    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    for c, (cx, cy), label, color in zip(range(5), centers, labels, cluster_colors):
        # 18 image points per cluster
        img_pts = rng.normal(loc=(cx, cy), scale=0.55, size=(18, 2))
        # 18 text points per cluster (slightly offset)
        txt_pts = rng.normal(loc=(cx + 0.25, cy + 0.15), scale=0.55, size=(18, 2))
        ax.scatter(img_pts[:, 0], img_pts[:, 1], s=110, color=color,
                   alpha=0.65, edgecolor="white", linewidth=1.2,
                   label=f"{label} (image)" if c == 0 else None)
        ax.scatter(txt_pts[:, 0], txt_pts[:, 1], s=110, marker="^",
                   color=color, alpha=0.65, edgecolor="white", linewidth=1.2,
                   label=f"{label} (text)" if c == 0 else None)
        # cluster label
        ax.text(cx, cy + 1.5, label, ha="center", fontsize=12,
                weight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, linewidth=1.2))

    # Draw a few image<->text alignment lines on cluster 'dog' to show pairing
    rng2 = np.random.default_rng(7)
    cx, cy = centers[0]
    for _ in range(3):
        x1, y1 = rng2.normal(loc=(cx, cy), scale=0.4)
        x2, y2 = rng2.normal(loc=(cx + 0.25, cy + 0.15), scale=0.4)
        ax.plot([x1, x2], [y1, y2], color=C_DARK, linewidth=1.0, alpha=0.45,
                linestyle="--", zorder=1)

    # Custom legend (markers only, no color clutter)
    from matplotlib.lines import Line2D


    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GRAY,
               markersize=11, label="image embedding"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_GRAY,
               markersize=11, label="text embedding"),
        Line2D([0], [0], color=C_DARK, linestyle="--", linewidth=1.2,
               label="image-text pair"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              framealpha=0.95, edgecolor=C_GRAY)

    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.set_title("Joint multimodal embedding space (t-SNE projection)\n"
                 "image and text encodings cluster by semantic concept",
                 fontsize=12.5, weight="bold", color=C_DARK)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-6, 6)

    fig.tight_layout()
    _save(fig, "fig5_embedding_tsne")


# ---------------------------------------------------------------------------
# Figure 6: Vision-Language downstream tasks gallery
# ---------------------------------------------------------------------------
def fig6_vl_tasks() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle("Vision-Language downstream tasks",
                 fontsize=14, weight="bold", color=C_DARK, y=1.0)

    rng = np.random.default_rng(2025)

    def _img_swatch(ax, x, y, w, h, color=C_BLUE):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=C_LIGHT,
                               edgecolor=color, linewidth=1.8))
        for _ in range(int(w * h * 25)):
            ax.scatter(x + rng.uniform(0, w), y + rng.uniform(0, h),
                       s=12, color=color, alpha=0.45)

    # --- VQA ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off()
    ax.set_title("Visual Question Answering (VQA)", fontsize=11.5,
                 weight="bold", color=C_BLUE)
    _img_swatch(ax, 0.5, 1.6, 3.2, 3.2)
    ax.text(2.1, 1.1, "image", ha="center", fontsize=9, color=C_BLUE)
    ax.text(5.2, 4.5, "Q: How many people are in the photo?",
            fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_PURPLE, linewidth=1.2))
    ax.text(5.2, 2.8, "A: three",
            fontsize=11, color=C_GREEN, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_GREEN, linewidth=1.5))
    _arrow(ax, 5.0, 4.5, 4.7, 3.9, color=C_GRAY)
    _arrow(ax, 5.5, 3.8, 5.5, 3.2, color=C_GRAY)

    # --- Captioning ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off()
    ax.set_title("Image Captioning", fontsize=11.5,
                 weight="bold", color=C_PURPLE)
    _img_swatch(ax, 0.5, 1.6, 3.2, 3.2, color=C_PURPLE)
    ax.text(2.1, 1.1, "image", ha="center", fontsize=9, color=C_PURPLE)
    _arrow(ax, 3.9, 3.2, 4.7, 3.2, color=C_DARK)
    ax.text(7.4, 3.2,
            '"a young girl in\na red dress\nplaying with a dog"',
            ha="center", fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_PURPLE, linewidth=1.5))
    ax.text(4.3, 3.45, "decode", fontsize=8, color=C_GRAY, style="italic")

    # --- Retrieval ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off()
    ax.set_title("Image-Text Retrieval", fontsize=11.5,
                 weight="bold", color=C_GREEN)
    ax.text(0.4, 4.6, "query:", fontsize=9, color=C_GRAY)
    ax.text(2.5, 4.6, '"a sunset over the ocean"',
            ha="center", fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_GREEN, linewidth=1.2))
    for i in range(5):
        x = 0.4 + i * 1.95
        col = C_GREEN if i == 0 else C_GRAY
        _img_swatch(ax, x, 1.4, 1.7, 1.7, color=col)
        ax.text(x + 0.85, 1.05, f"#{i+1}", ha="center", fontsize=9,
                color=col, weight="bold")
    ax.text(5.0, 0.45, "ranked by cosine similarity",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    # --- Visual Grounding ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off()
    ax.set_title("Visual Grounding / Referring", fontsize=11.5,
                 weight="bold", color=C_AMBER)
    _img_swatch(ax, 0.5, 0.9, 4.2, 4.2, color=C_GRAY)
    # bbox highlight
    ax.add_patch(Rectangle((1.5, 2.2), 1.3, 1.5, fill=False,
                           edgecolor=C_AMBER, linewidth=2.8))
    ax.text(2.15, 1.95, "predicted bbox", ha="center", fontsize=8,
            color=C_AMBER, weight="bold")
    ax.text(7.5, 3.3,
            'expression:\n"the woman wearing\na blue jacket"',
            ha="center", fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_AMBER, linewidth=1.4))
    _arrow(ax, 6.2, 3.0, 5.0, 3.0, color=C_GRAY)

    fig.tight_layout()
    _save(fig, "fig6_vl_tasks")


# ---------------------------------------------------------------------------
# Figure 7: Fusion strategies (early / late / cross-attention)
# ---------------------------------------------------------------------------
def fig7_fusion_strategies() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    fig.suptitle("Multimodal fusion strategies",
                 fontsize=14, weight="bold", color=C_DARK, y=1.02)

    def _setup(ax, title, color):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, weight="bold", color=color, pad=8)

    # ---- Early fusion ----
    ax = axes[0]
    _setup(ax, "Early fusion", C_BLUE)
    _box(ax, 0.3, 5.0, 1.4, 0.7, "image", fc=C_BLUE, fontsize=9.5)
    _box(ax, 4.3, 5.0, 1.4, 0.7, "text",  fc=C_PURPLE, fontsize=9.5)
    _arrow(ax, 1.0, 5.0, 1.7, 4.4, color=C_BLUE)
    _arrow(ax, 5.0, 5.0, 4.3, 4.4, color=C_PURPLE)
    _box(ax, 1.7, 3.7, 2.6, 0.7, "concat raw features", fc=C_AMBER, fontsize=9.5)
    _arrow(ax, 3.0, 3.7, 3.0, 3.1, color=C_DARK)
    _box(ax, 1.7, 2.4, 2.6, 0.7, "single encoder $f$", fc=C_DARK, fontsize=9.5)
    _arrow(ax, 3.0, 2.4, 3.0, 1.8, color=C_DARK)
    _box(ax, 1.7, 1.1, 2.6, 0.7, "joint embedding", fc=C_GREEN, fontsize=9.5)
    ax.text(3.0, 0.55,
            r"$\mathbf{h} = f([\mathbf{v}; \mathbf{t}])$",
            ha="center", fontsize=11, color=C_DARK)

    # ---- Late fusion ----
    ax = axes[1]
    _setup(ax, "Late fusion", C_GREEN)
    _box(ax, 0.3, 5.0, 1.4, 0.7, "image", fc=C_BLUE, fontsize=9.5)
    _box(ax, 4.3, 5.0, 1.4, 0.7, "text",  fc=C_PURPLE, fontsize=9.5)
    _arrow(ax, 1.0, 5.0, 1.0, 4.3, color=C_BLUE)
    _arrow(ax, 5.0, 5.0, 5.0, 4.3, color=C_PURPLE)
    _box(ax, 0.2, 3.0, 1.6, 1.3, "image\nencoder $f_v$", fc=C_BLUE, fontsize=9.5)
    _box(ax, 4.2, 3.0, 1.6, 1.3, "text\nencoder $f_t$", fc=C_PURPLE, fontsize=9.5)
    _arrow(ax, 1.0, 3.0, 2.4, 2.1, color=C_BLUE)
    _arrow(ax, 5.0, 3.0, 3.6, 2.1, color=C_PURPLE)
    _box(ax, 1.9, 1.4, 2.2, 0.7, "combine $g(\\cdot,\\cdot)$", fc=C_AMBER, fontsize=9.5)
    _arrow(ax, 3.0, 1.4, 3.0, 0.85, color=C_DARK)
    ax.text(3.0, 0.45,
            r"$\mathbf{h} = g(f_v(\mathbf{v}), f_t(\mathbf{t}))$",
            ha="center", fontsize=10.5, color=C_DARK)

    # ---- Cross-attention fusion ----
    ax = axes[2]
    _setup(ax, "Cross-attention fusion", C_AMBER)
    _box(ax, 0.3, 5.0, 1.4, 0.7, "image", fc=C_BLUE, fontsize=9.5)
    _box(ax, 4.3, 5.0, 1.4, 0.7, "text",  fc=C_PURPLE, fontsize=9.5)
    _arrow(ax, 1.0, 5.0, 1.0, 4.3, color=C_BLUE)
    _arrow(ax, 5.0, 5.0, 5.0, 4.3, color=C_PURPLE)
    _box(ax, 0.2, 3.0, 1.6, 1.3, "image\nencoder", fc=C_BLUE, fontsize=9.5)
    _box(ax, 4.2, 3.0, 1.6, 1.3, "text\nencoder", fc=C_PURPLE, fontsize=9.5)
    # cross-attention block
    _box(ax, 1.4, 1.5, 3.2, 1.0, "cross-attention\nblocks  (xN)", fc=C_AMBER, fontsize=9.5)
    _arrow(ax, 1.0, 3.0, 2.0, 2.5, color=C_BLUE)
    _arrow(ax, 5.0, 3.0, 4.0, 2.5, color=C_PURPLE)
    # bidirectional arrow inside
    _arrow(ax, 2.2, 2.0, 3.8, 2.0, color="white", lw=2, style="<->", mutation=14)
    _arrow(ax, 3.0, 1.5, 3.0, 0.95, color=C_DARK)
    ax.text(3.0, 0.55,
            r"$\mathrm{Attn}(Q_v, K_t, V_t)\;,\;\mathrm{Attn}(Q_t, K_v, V_v)$",
            ha="center", fontsize=9.5, color=C_DARK)

    # Bottom captions: trade-offs row
    fig.text(0.18, -0.02, "simple, but no pretrained encoders",
             ha="center", fontsize=9, color=C_GRAY, style="italic")
    fig.text(0.5, -0.02, "modular, easy to scale; weak interaction",
             ha="center", fontsize=9, color=C_GRAY, style="italic")
    fig.text(0.83, -0.02, "richest interaction; expensive (ViLBERT, BLIP)",
             ha="center", fontsize=9, color=C_GRAY, style="italic")

    fig.tight_layout()
    _save(fig, "fig7_fusion_strategies")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating multimodal-transfer figures ...")
    fig1_clip_architecture()
    fig2_contrastive_alignment()
    fig3_blip_architecture()
    fig4_cross_modal_retrieval()
    fig5_embedding_tsne()
    fig6_vl_tasks()
    fig7_fusion_strategies()
    print("Done.")


if __name__ == "__main__":
    main()
