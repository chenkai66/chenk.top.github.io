"""
Figure generation script for Transfer Learning Part 07: Zero-Shot Learning.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates one teaching idea so the visual reads cleanly at article
display size.

Figures
-------
fig1_zsl_vs_fsl_vs_supervised
    Three-panel comparison: supervised (many labels), few-shot (few labels),
    zero-shot (semantic description, no labels). Shows label budget and
    inference mechanism in one glance.
fig2_attribute_classification
    Worked DAP example: image -> attribute scores -> nearest class via the
    attribute prototype matrix. Heat map plus arrow flow.
fig3_semantic_embedding_space
    2D semantic embedding space showing seen-class and unseen-class
    prototypes; a query image embedding lands near an unseen prototype,
    illustrating the cross-modal compatibility intuition.
fig4_devise_visual_text_embedding
    DeViSE architecture: CNN visual encoder + Word2Vec text encoder ->
    shared embedding space optimised by hinge ranking loss.
fig5_clip_zero_shot
    CLIP zero-shot classification flow: image embedding compared against
    text embeddings of "a photo of a {class}" prompts via cosine similarity
    and softmax. Includes a small score bar chart.
fig6_gzsl_vs_zsl
    Conventional ZSL vs GZSL: bar groups for seen accuracy, unseen accuracy
    and harmonic mean H, before and after calibration on AwA2.
fig7_benchmark_results
    Method comparison on AwA2 and CUB: DAP / ALE / DeViSE / SAE / f-CLSWGAN
    / CADA-VAE / CLIP. Grouped bars for ZSL acc and GZSL H.

Usage
-----
    python3 scripts/figures/transfer-learning/07-zero-shot.py

Outputs
-------
PNG files written to BOTH the EN and ZH article asset folders so the
markdown references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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
C_LIGHT = COLORS["grid"]
C_DARK = COLORS["text"]

DPI = 150

# Output destinations (English + Chinese article folders).
EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/"
    "transfer-learning/07-zero-shot-learning"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/"
    "transfer-learning/07-零样本学习"
)


def save(fig: plt.Figure, name: str) -> None:
    """Write the figure to both the EN and ZH asset folders."""
    for folder in (EN_DIR, ZH_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: ZSL vs Few-Shot vs Supervised
# ---------------------------------------------------------------------------
def fig1_zsl_vs_fsl_vs_supervised() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    titles = ["Supervised", "Few-Shot (k=5)", "Zero-Shot"]
    counts = [200, 5, 0]
    colors = [C_BLUE, C_PURPLE, C_GREEN]
    notes = [
        "Hundreds of labelled\nexamples per class",
        "A handful of labelled\nexamples per class",
        "No labels — only a\nsemantic description",
    ]
    mechanism = [
        "$\\hat y = \\arg\\max_c P(c|x;\\theta)$",
        "$\\hat y = \\arg\\max_c \\mathrm{sim}(x, S_c)$",
        "$\\hat y = \\arg\\max_c F(x, a_c)$",
    ]

    rng = np.random.default_rng(0)
    for ax, title, n, col, note, mech in zip(axes, titles, counts, colors,
                                             notes, mechanism):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#f8fafc")

        # Sample dots representing labelled images.
        n_show = min(n, 60)
        if n_show > 0:
            xs = rng.uniform(0.5, 9.5, n_show)
            ys = rng.uniform(4.0, 8.5, n_show)
            ax.scatter(xs, ys, s=42, c=col, alpha=0.55, edgecolor="white",
                       linewidth=0.6)
        else:
            # Zero-shot: a stylised text description.
            box = FancyBboxPatch(
                (1.4, 5.6), 7.2, 2.5,
                boxstyle="round,pad=0.18", linewidth=1.4,
                edgecolor=col, facecolor="white",
            )
            ax.add_patch(box)
            ax.text(5.0, 6.85,
                    '"horse-like animal\nwith black and white stripes"',
                    ha="center", va="center", fontsize=10.5,
                    style="italic", color=C_DARK)

        # Bottom info card.
        card = FancyBboxPatch(
            (0.4, 0.6), 9.2, 2.6,
            boxstyle="round,pad=0.15", linewidth=1.0,
            edgecolor=C_LIGHT, facecolor="white",
        )
        ax.add_patch(card)
        label = f"Labels per class: {n}" if n > 0 else "Labels per class: 0"
        ax.text(5.0, 2.55, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col)
        ax.text(5.0, 1.75, note, ha="center", va="center",
                fontsize=9.5, color=C_DARK)
        ax.text(5.0, 0.95, mech, ha="center", va="center",
                fontsize=10, color=C_DARK)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10,
                     color=C_DARK)

    fig.suptitle("Supervision spectrum: from labelled images to semantics",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_zsl_vs_fsl_vs_supervised")


# ---------------------------------------------------------------------------
# Figure 2: Attribute-based classification (DAP)
# ---------------------------------------------------------------------------
def fig2_attribute_classification() -> None:
    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.5], wspace=0.35)

    # ----- panel A: query image (stylised) -----
    axA = fig.add_subplot(gs[0])
    axA.set_xlim(0, 10)
    axA.set_ylim(0, 10)
    axA.set_xticks([])
    axA.set_yticks([])
    for s in axA.spines.values():
        s.set_visible(False)
    axA.set_facecolor("#f8fafc")
    # frame
    axA.add_patch(Rectangle((1, 2), 8, 6, linewidth=1.2,
                            edgecolor=C_GRAY, facecolor="white"))
    # stripes (zebra hint)
    for i in range(8):
        axA.add_patch(Rectangle((1.3 + i * 1.0, 2.3), 0.45, 5.4,
                                facecolor=C_DARK if i % 2 == 0 else "white",
                                alpha=0.85, edgecolor="none"))
    axA.text(5, 1.0, "Query image $x$", ha="center", va="center",
             fontsize=11, fontweight="bold", color=C_DARK)
    axA.text(5, 9.2, "CNN features $\\phi(x)$",
             ha="center", va="center", fontsize=10.5, color=C_BLUE)

    # ----- panel B: predicted attribute scores -----
    axB = fig.add_subplot(gs[1])
    attrs = ["striped", "four-legs", "mane", "wings", "fur", "aquatic",
             "hooves"]
    pred = np.array([0.94, 0.91, 0.78, 0.05, 0.72, 0.04, 0.85])
    bars = axB.barh(attrs, pred,
                    color=[C_GREEN if v > 0.5 else C_GRAY for v in pred],
                    edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, pred):
        axB.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=9, color=C_DARK)
    axB.set_xlim(0, 1.15)
    axB.invert_yaxis()
    axB.set_xlabel("$\\hat a_m(x)$ (attribute probability)", fontsize=10)
    axB.set_title("Attribute classifiers", fontsize=12, fontweight="bold",
                  color=C_DARK)
    axB.set_axisbelow(True)
    axB.grid(axis="x", alpha=0.3)

    # ----- panel C: class prototype matching -----
    axC = fig.add_subplot(gs[2])
    classes = ["zebra", "horse", "tiger", "penguin", "dolphin"]
    proto = np.array([
        [1, 1, 1, 0, 1, 0, 1],   # zebra
        [0, 1, 1, 0, 1, 0, 1],   # horse
        [1, 1, 0, 0, 1, 0, 0],   # tiger
        [0, 0, 0, 1, 0, 1, 0],   # penguin
        [0, 0, 0, 0, 0, 1, 0],   # dolphin
    ])
    sims = proto @ pred / (np.linalg.norm(proto, axis=1)
                           * np.linalg.norm(pred) + 1e-9)
    order = np.argsort(-sims)
    cls_sorted = [classes[i] for i in order]
    sims_sorted = sims[order]
    cols = [C_AMBER if i == 0 else C_BLUE for i in range(len(cls_sorted))]
    bars = axC.barh(cls_sorted, sims_sorted, color=cols,
                    edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, sims_sorted):
        axC.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=9, color=C_DARK)
    axC.set_xlim(0, 1.15)
    axC.invert_yaxis()
    axC.set_xlabel("cosine($\\hat a$, $a_c$)", fontsize=10)
    axC.set_title("Nearest class prototype", fontsize=12, fontweight="bold",
                  color=C_DARK)
    axC.set_axisbelow(True)
    axC.grid(axis="x", alpha=0.3)
    axC.text(sims_sorted[0] / 2, -0.55,
             "predicted: " + cls_sorted[0],
             ha="center", fontsize=10, fontweight="bold", color=C_AMBER)

    fig.suptitle("Direct Attribute Prediction (DAP): two-stage zero-shot "
                 "pipeline", fontsize=13.5, fontweight="bold",
                 color=C_DARK, y=1.02)
    save(fig, "fig2_attribute_classification")


# ---------------------------------------------------------------------------
# Figure 3: Semantic embedding space (seen + unseen)
# ---------------------------------------------------------------------------
def fig3_semantic_embedding_space() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.6))

    # Seen-class prototypes.
    seen = {
        "horse":   (-2.0,  1.4),
        "dog":     (-3.6, -0.8),
        "cat":     (-3.0, -2.2),
        "cow":     (-1.6,  2.7),
        "tiger":   ( 0.4,  0.6),
        "bear":    (-1.0, -2.4),
    }
    # Unseen-class prototypes.
    unseen = {
        "zebra":   (-0.2,  2.1),
        "lion":    ( 1.2,  0.0),
        "panda":   (-2.4, -1.6),
        "leopard": ( 1.8,  1.4),
    }

    rng = np.random.default_rng(42)
    # Visual feature samples drawn around seen classes after f_v projection.
    for (xc, yc) in seen.values():
        pts = rng.normal(loc=[xc, yc], scale=0.32, size=(35, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=C_BLUE, alpha=0.22,
                   edgecolor="none")

    # Plot prototypes.
    for name, (x, y) in seen.items():
        ax.scatter(x, y, s=240, marker="o", color="white",
                   edgecolor=C_BLUE, linewidth=2.2, zorder=4)
        ax.text(x, y - 0.45, name, ha="center", va="top",
                fontsize=10, color=C_BLUE, fontweight="bold")
    for name, (x, y) in unseen.items():
        ax.scatter(x, y, s=240, marker="*", color=C_AMBER,
                   edgecolor=C_DARK, linewidth=1.0, zorder=4)
        ax.text(x, y + 0.55, name, ha="center", va="bottom",
                fontsize=10, color=C_AMBER, fontweight="bold")

    # Query image projection landing near "zebra".
    qx, qy = 0.05, 1.85
    ax.scatter(qx, qy, s=180, marker="X", color=C_PURPLE,
               edgecolor="white", linewidth=1.6, zorder=5)
    ax.annotate("query $\\phi(x)$", xy=(qx, qy), xytext=(2.6, 3.2),
                fontsize=10.5, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.4))

    # Highlight the nearest-prototype match.
    zx, zy = unseen["zebra"]
    ax.plot([qx, zx], [qy, zy], linestyle="--", color=C_AMBER, lw=1.6,
            zorder=3)
    ax.text((qx + zx) / 2 + 0.1, (qy + zy) / 2 + 0.15,
            "nearest unseen\nprototype",
            fontsize=9, color=C_AMBER, ha="left")

    ax.set_xlim(-5.2, 4.0)
    ax.set_ylim(-3.5, 3.8)
    ax.set_xlabel("semantic dim 1", fontsize=10)
    ax.set_ylabel("semantic dim 2", fontsize=10)
    ax.set_title("Shared semantic embedding space: seen prototypes anchor "
                 "the geometry, unseen prototypes inherit it",
                 fontsize=12.5, fontweight="bold", color=C_DARK)

    legend = [
        mpatches.Patch(color=C_BLUE, label="seen class prototype $a_c$"),
        mpatches.Patch(color=C_AMBER, label="unseen class prototype $a_c$"),
        mpatches.Patch(color=C_PURPLE,
                       label="query image embedding $\\phi(x)$"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9.5,
              framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig3_semantic_embedding_space")


# ---------------------------------------------------------------------------
# Figure 4: DeViSE architecture
# ---------------------------------------------------------------------------
def fig4_devise_visual_text_embedding() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    def block(x, y, w, h, label, color, sub=None):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           linewidth=1.4, edgecolor=color,
                           facecolor=color + "22")
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), label,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=color)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.32, sub, ha="center",
                    va="center", fontsize=8.5, color=C_DARK)

    def arrow(x1, y1, x2, y2, color=C_GRAY, label=None, lab_off=(0, 0.25)):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=1.6))
        if label:
            ax.text((x1 + x2) / 2 + lab_off[0],
                    (y1 + y2) / 2 + lab_off[1],
                    label, ha="center", fontsize=9, color=color)

    # Visual branch (top).
    block(0.3, 5.4, 1.8, 1.6, "image\n$x$", C_DARK)
    block(2.6, 5.4, 2.6, 1.6, "CNN", C_BLUE,
          sub="ResNet / Inception")
    block(5.8, 5.4, 2.4, 1.6, "$f_v(\\phi(x))$", C_BLUE,
          sub="visual projection")
    arrow(2.1, 6.2, 2.6, 6.2, C_BLUE)
    arrow(5.2, 6.2, 5.8, 6.2, C_BLUE)

    # Text branch (bottom).
    block(0.3, 1.0, 1.8, 1.6, 'class name\n"zebra"', C_DARK)
    block(2.6, 1.0, 2.6, 1.6, "Word2Vec", C_PURPLE,
          sub="skip-gram embedding")
    block(5.8, 1.0, 2.4, 1.6, "$f_s(a_c)$", C_PURPLE,
          sub="text projection")
    arrow(2.1, 1.8, 2.6, 1.8, C_PURPLE)
    arrow(5.2, 1.8, 5.8, 1.8, C_PURPLE)

    # Shared embedding space.
    block(9.2, 3.1, 2.6, 1.8, "shared\nembedding", C_GREEN,
          sub="$\\mathbb{R}^d$")
    arrow(8.2, 6.2, 9.2, 4.4, C_BLUE)
    arrow(8.2, 1.8, 9.2, 3.6, C_PURPLE)

    # Loss block.
    block(12.3, 3.1, 1.5, 1.8, "hinge\nrank loss", C_AMBER,
          sub="margin pulls\n$F^+$ above $F^-$")
    arrow(11.8, 4.0, 12.3, 4.0, C_AMBER)

    ax.text(7.0, 7.6, "DeViSE: align visual features with word "
            "embeddings in a shared space",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(7.0, 0.25,
            "At inference: compute $F(x,c)=f_v(\\phi(x))^\\top f_s(a_c)$ "
            "for every class, including unseen ones",
            ha="center", fontsize=10, color=C_DARK, style="italic")
    save(fig, "fig4_devise_visual_text_embedding")


# ---------------------------------------------------------------------------
# Figure 5: CLIP zero-shot classification
# ---------------------------------------------------------------------------
def fig5_clip_zero_shot() -> None:
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    def block(x, y, w, h, label, color, sub=None):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           linewidth=1.4, edgecolor=color,
                           facecolor=color + "22")
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2 + (0.2 if sub else 0), label,
                ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=color)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.32, sub, ha="center",
                    va="center", fontsize=8.5, color=C_DARK)

    def arrow(x1, y1, x2, y2, color=C_GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=1.6))

    # Image -> image encoder -> image embedding
    block(0.3, 5.6, 2.0, 1.6, "image", C_DARK)
    block(2.8, 5.6, 2.6, 1.6, "image\nencoder", C_BLUE,
          sub="ViT / ResNet")
    block(6.0, 5.6, 2.2, 1.6, "$I$", C_BLUE, sub="image embedding")
    arrow(2.3, 6.4, 2.8, 6.4, C_BLUE)
    arrow(5.4, 6.4, 6.0, 6.4, C_BLUE)

    # Prompts -> text encoder -> text embeddings
    prompts_label = ('"a photo of a dog"\n"a photo of a cat"\n'
                     '"a photo of a zebra"\n...')
    block(0.3, 0.5, 2.6, 2.6, prompts_label, C_PURPLE,
          sub="prompt template")
    block(3.4, 1.2, 2.4, 1.6, "text\nencoder", C_PURPLE,
          sub="Transformer")
    block(6.4, 1.2, 1.8, 1.6, "$T_1, \\ldots, T_K$", C_PURPLE,
          sub="text embeddings")
    arrow(2.9, 2.0, 3.4, 2.0, C_PURPLE)
    arrow(5.8, 2.0, 6.4, 2.0, C_PURPLE)

    # Cosine similarity + softmax
    block(9.0, 3.2, 2.2, 2.0, "cosine\nsimilarity",
          C_GREEN, sub="$I \\cdot T_k / \\|I\\|\\|T_k\\|$")
    block(11.6, 3.2, 2.2, 2.0, "softmax",
          C_AMBER, sub="$/\\tau$, then $\\arg\\max$")
    arrow(8.2, 6.4, 9.0, 4.6, C_BLUE)
    arrow(8.2, 2.0, 9.0, 3.8, C_PURPLE)
    arrow(11.2, 4.2, 11.6, 4.2, C_GREEN)

    ax.text(7.0, 7.6,
            "CLIP zero-shot: build a classifier on the fly from text prompts",
            ha="center", fontsize=12.5, fontweight="bold", color=C_DARK)

    # Right panel: class probability bars.
    axR = fig.add_subplot(gs[1])
    classes = ["zebra", "horse", "dog", "cat", "tiger"]
    sims = np.array([0.29, 0.23, 0.15, 0.13, 0.20])
    probs = np.exp(sims / 0.07)
    probs = probs / probs.sum()
    cols = [C_AMBER if i == int(np.argmax(probs)) else C_BLUE
            for i in range(len(classes))]
    bars = axR.barh(classes, probs, color=cols, edgecolor="white",
                    linewidth=1.2)
    for bar, p in zip(bars, probs):
        axR.text(p + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{p:.2f}", va="center", fontsize=9.5, color=C_DARK)
    axR.invert_yaxis()
    axR.set_xlim(0, 1.05)
    axR.set_xlabel("$P(c | x)$ via softmax($I \\cdot T_c / \\tau$)",
                   fontsize=10)
    axR.set_title("Predicted class probabilities", fontsize=12,
                  fontweight="bold", color=C_DARK)
    axR.grid(axis="x", alpha=0.3)
    axR.set_axisbelow(True)

    save(fig, "fig5_clip_zero_shot")


# ---------------------------------------------------------------------------
# Figure 6: Conventional ZSL vs GZSL (with calibration)
# ---------------------------------------------------------------------------
def fig6_gzsl_vs_zsl() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    methods = ["Conventional\nZSL", "GZSL\n(no calibration)",
               "GZSL\n(+ bias subtraction)", "GZSL\n(generative: f-CLSWGAN)"]
    seen = [None, 87.6, 70.4, 60.7]
    unseen = [65.0, 12.4, 41.5, 56.1]
    H = [None, 21.7, 52.1, 58.3]

    x = np.arange(len(methods))
    width = 0.26

    # Plot only valid bars per method.
    bars_seen = []
    bars_unseen = []
    bars_h = []
    for i, m in enumerate(methods):
        if seen[i] is not None:
            bars_seen.append(ax.bar(x[i] - width, seen[i], width,
                                    color=C_BLUE,
                                    edgecolor="white", linewidth=1.2))
        bars_unseen.append(ax.bar(x[i], unseen[i], width,
                                  color=C_PURPLE,
                                  edgecolor="white", linewidth=1.2))
        if H[i] is not None:
            bars_h.append(ax.bar(x[i] + width, H[i], width,
                                 color=C_AMBER,
                                 edgecolor="white", linewidth=1.2))

    # Annotate.
    for i in range(len(methods)):
        if seen[i] is not None:
            ax.text(x[i] - width, seen[i] + 1.3, f"{seen[i]:.1f}",
                    ha="center", fontsize=9, color=C_BLUE)
        ax.text(x[i], unseen[i] + 1.3, f"{unseen[i]:.1f}",
                ha="center", fontsize=9, color=C_PURPLE)
        if H[i] is not None:
            ax.text(x[i] + width, H[i] + 1.3, f"{H[i]:.1f}",
                    ha="center", fontsize=9, color=C_AMBER)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Accuracy on AwA2 (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Why GZSL is hard, and how calibration / generation help",
                 fontsize=12.5, fontweight="bold", color=C_DARK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.35)

    legend = [
        mpatches.Patch(color=C_BLUE, label="seen accuracy $S$"),
        mpatches.Patch(color=C_PURPLE, label="unseen accuracy $U$"),
        mpatches.Patch(color=C_AMBER, label="harmonic mean $H$"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=10,
              framealpha=0.95)
    ax.text(0.5, -0.15, "Numbers are illustrative AwA2-style values used to "
            "convey the bias problem.",
            transform=ax.transAxes, ha="center", fontsize=8.5,
            color=C_GRAY, style="italic")

    fig.tight_layout()
    save(fig, "fig6_gzsl_vs_zsl")


# ---------------------------------------------------------------------------
# Figure 7: Benchmark results on AwA2 / CUB
# ---------------------------------------------------------------------------
def fig7_benchmark_results() -> None:
    methods = ["DAP\n(2009)", "ALE\n(2013)", "DeViSE\n(2013)",
               "SAE\n(2017)", "f-CLSWGAN\n(2018)", "CADA-VAE\n(2019)",
               "CLIP\n(2021)"]
    awa_zsl = [46.1, 62.5, 59.7, 54.1, 68.2, 64.0, 88.3]
    awa_h   = [ 7.4, 23.9, 22.4, 14.6, 58.3, 63.9, 79.5]
    cub_zsl = [40.0, 54.9, 52.0, 33.3, 57.3, 60.4, 64.4]
    cub_h   = [ 4.8, 34.4, 32.8, 19.6, 49.7, 53.5, 58.9]

    x = np.arange(len(methods))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True)

    for ax, zsl, h, name in zip(
        axes,
        [awa_zsl, cub_zsl],
        [awa_h, cub_h],
        ["AwA2", "CUB"],
    ):
        ax.bar(x - width / 2, zsl, width, color=C_BLUE,
               edgecolor="white", linewidth=1.2,
               label="Conventional ZSL accuracy")
        ax.bar(x + width / 2, h, width, color=C_AMBER,
               edgecolor="white", linewidth=1.2,
               label="GZSL harmonic mean $H$")
        for i, (a, b) in enumerate(zip(zsl, h)):
            ax.text(i - width / 2, a + 1.0, f"{a:.0f}", ha="center",
                    fontsize=8.5, color=C_BLUE)
            ax.text(i + width / 2, b + 1.0, f"{b:.0f}", ha="center",
                    fontsize=8.5, color=C_AMBER)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9.5)
        ax.set_title(name, fontsize=12.5, fontweight="bold",
                     color=C_DARK)
        ax.set_ylim(0, 100)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.35)

    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    fig.suptitle("Zero-shot benchmarks: discriminative -> generative -> "
                 "vision-language pretraining",
                 fontsize=13.5, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_benchmark_results")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    fig1_zsl_vs_fsl_vs_supervised()
    fig2_attribute_classification()
    fig3_semantic_embedding_space()
    fig4_devise_visual_text_embedding()
    fig5_clip_zero_shot()
    fig6_gzsl_vs_zsl()
    fig7_benchmark_results()
    print(f"Wrote 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
