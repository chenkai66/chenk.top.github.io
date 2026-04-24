"""
Figure generation script for Transfer Learning Part 11:
Cross-Lingual Transfer.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure isolates one teaching point and is built to be self-contained.

Figures:
    fig1_embedding_space         Multilingual embedding space. Anchor concepts
                                 (cat / dog / king / queen / one / two) appear
                                 as clusters whose English/Chinese/French
                                 vectors land close together after alignment.
    fig2_xlmr_architecture       mBERT vs XLM-R: shared subword vocab, 12/24
                                 Transformer layers, MLM (and TLM for XLM)
                                 objectives, multilingual corpus mixing.
    fig3_zero_shot_ner           Zero-shot NER F1: train on English CoNLL,
                                 test on 10 target languages with mBERT vs
                                 XLM-R-base vs XLM-R-large; gap to in-language
                                 supervised oracle highlighted.
    fig4_pivot_strategies        Three pivot strategies for X -> Y when no
                                 direct data: source-pivot, target-pivot,
                                 multi-source ensemble. Diagrammatic.
    fig5_subword_tokenization    BPE / SentencePiece subword splits of the
                                 word "international" / "internationale" /
                                 "internationaler" across English / French /
                                 German -- shows how subword overlap creates
                                 cross-lingual anchors.
    fig6_resource_curve          Pre-training corpus size (log) vs zero-shot
                                 XNLI accuracy per language. High-resource
                                 plateau, low-resource cliff, and the gain
                                 from balanced sampling (alpha = 0.7).
    fig7_translate_vs_align      Translate-train vs Translate-test vs
                                 Zero-shot align: pipeline diagram with
                                 cost / latency / quality bars.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb #7c3aed #10b981 #f59e0b.

Usage:
    python3 scripts/figures/transfer-learning/11-cross-lingual.py

Output:
    Writes the same PNGs into BOTH the EN and ZH article asset folders so
    the markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# Make sure CJK glyphs render. Fall back gracefully on Linux.
plt.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB", "PingFang SC", "Heiti TC", "Songti SC",
    "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

C_BLUE = "#2563eb"     # English / source
C_PURPLE = "#7c3aed"   # Chinese / target-A
C_GREEN = "#10b981"    # French  / target-B
C_AMBER = "#f59e0b"    # warning / oracle
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
    / "11-cross-lingual-transfer"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "transfer-learning"
    / "11-跨语言迁移"
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
# Figure 1: Multilingual embedding space
# ---------------------------------------------------------------------------
def fig1_embedding_space() -> None:
    """Show that semantically equivalent words across languages cluster."""
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    concepts = [
        ("cat",   "猫",   "chat",     ( 1.2,  3.4)),
        ("dog",   "狗",   "chien",    ( 2.1,  2.9)),
        ("king",  "国王", "roi",      (-2.6,  2.0)),
        ("queen", "女王", "reine",    (-2.0,  1.2)),
        ("one",   "一",   "un",       ( 3.1, -1.7)),
        ("two",   "二",   "deux",     ( 3.7, -2.4)),
        ("water", "水",   "eau",      (-2.4, -1.7)),
        ("bread", "面包", "pain",     (-3.0, -2.6)),
    ]

    lang_offset = {
        "en": np.array([ 0.00,  0.00]),
        "zh": np.array([ 0.42, -0.22]),
        "fr": np.array([-0.36,  0.30]),
    }
    lang_color = {"en": C_BLUE, "zh": C_PURPLE, "fr": C_GREEN}
    lang_marker = {"en": "o", "zh": "s", "fr": "^"}

    for en, zh, fr, center in concepts:
        cx, cy = center
        # cluster halo
        ax.add_patch(plt.Circle((cx, cy), 0.65, color=C_LIGHT, alpha=0.55,
                                zorder=0))
        for lang, label in (("en", en), ("zh", zh), ("fr", fr)):
            jitter = rng.normal(0, 0.06, size=2)
            p = np.array([cx, cy]) + lang_offset[lang] + jitter
            ax.scatter(p[0], p[1], s=120, marker=lang_marker[lang],
                       color=lang_color[lang], edgecolor="white",
                       linewidth=1.2, zorder=3)
            ax.text(p[0] + 0.12, p[1] + 0.12, label, fontsize=8.5,
                    color=lang_color[lang], zorder=4)

    # Decorative semantic axes
    ax.annotate("", xy=(4.6, 3.4), xytext=(-4.6, 3.4),
                arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.8,
                                linestyle="--"))
    ax.annotate("", xy=(0, 4.4), xytext=(0, -4.4),
                arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.8,
                                linestyle="--"))
    ax.text(4.55, 3.55, "semantic dim 1", fontsize=8.5, ha="right",
            color=C_GRAY)
    ax.text(0.1, 4.4, "semantic dim 2", fontsize=8.5, color=C_GRAY)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C_BLUE, markersize=9, label="English"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=C_PURPLE, markersize=9, label="中文"),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor=C_GREEN, markersize=9, label="Français"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=10)

    ax.set_title("Multilingual embedding space: equivalent concepts cluster",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=14)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4.6, 4.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)

    _save(fig, "fig1_embedding_space")


# ---------------------------------------------------------------------------
# Figure 2: mBERT / XLM-R architecture
# ---------------------------------------------------------------------------
def fig2_xlmr_architecture() -> None:
    """Side-by-side schematic of mBERT and XLM-R training."""
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 6.65, "Multilingual pretraining: mBERT vs XLM-R",
            ha="center", fontsize=13.5, fontweight="bold", color=C_DARK)

    def column(x0, title, color, params, layers, corpus, sampling, objective):
        # Title bar
        _box(ax, (x0, 5.6), 5.0, 0.55, title, color, fontsize=12,
             fontweight="bold")
        # Corpus
        _box(ax, (x0 + 0.2, 4.55), 4.6, 0.7,
             f"Corpus: {corpus}", C_LIGHT, text_color=C_DARK,
             fontsize=10, alpha=0.9)
        # Sampling
        _box(ax, (x0 + 0.2, 3.75), 4.6, 0.7,
             f"Sampling: {sampling}", C_LIGHT, text_color=C_DARK,
             fontsize=10, alpha=0.9)

        # Transformer stack
        stack_x = x0 + 0.6
        stack_w = 3.8
        stack_y = 1.55
        stack_h = 1.95
        _box(ax, (stack_x, stack_y), stack_w, stack_h, "", color,
             alpha=0.18, rounding=0.06)
        for i in range(min(layers, 6)):
            y = stack_y + 0.18 + i * (stack_h - 0.36) / 6
            _box(ax, (stack_x + 0.25, y), stack_w - 0.5,
                 (stack_h - 0.36) / 6 - 0.04,
                 f"Transformer x {layers // 6 if i == 0 else ''}",
                 color, fontsize=8, alpha=0.55)
        ax.text(stack_x + stack_w / 2, stack_y + stack_h + 0.18,
                f"{layers} Transformer layers - {params}",
                ha="center", fontsize=9.5, color=C_DARK, fontweight="bold")

        # Objective
        _box(ax, (x0 + 0.2, 0.6), 4.6, 0.7, objective, color,
             fontsize=10, alpha=0.9)

    column(0.3, "mBERT (2018)", C_BLUE,
           "110M params",
           12,
           "Wikipedia, 104 langs (~13 GB)",
           "exponential, alpha=0.7 (heuristic)",
           "MLM only")
    column(6.7, "XLM-R (2020)", C_PURPLE,
           "270M / 550M params",
           24,
           "CommonCrawl, 100 langs (2.5 TB)",
           "balanced, p_l ~ n_l^0.7",
           "MLM (TLM optional)")

    # Shared vocab callout at bottom
    _box(ax, (1.0, 0.05), 10.0, 0.45,
         "Shared subword vocabulary (WordPiece 110K / SentencePiece 250K) "
         "creates cross-lingual anchors",
         C_GREEN, fontsize=10, alpha=0.9)

    _save(fig, "fig2_xlmr_architecture")


# ---------------------------------------------------------------------------
# Figure 3: Zero-shot NER F1 across 10 languages
# ---------------------------------------------------------------------------
def fig3_zero_shot_ner() -> None:
    """Train on English CoNLL-2003, evaluate zero-shot on WikiAnn."""
    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    langs = ["English\n(source)", "German", "Dutch", "Spanish",
             "French", "Russian", "Arabic", "Chinese", "Hindi",
             "Swahili"]
    # Representative numbers in line with XTREME / WikiAnn reports
    mbert     = [91.0, 78.2, 82.1, 76.4, 75.3, 65.8, 41.5, 53.1, 60.2, 63.4]
    xlmr_base = [91.5, 81.7, 84.6, 79.2, 78.6, 70.1, 49.3, 60.5, 67.4, 71.0]
    xlmr_lg   = [92.6, 84.4, 86.5, 81.6, 81.0, 73.8, 56.0, 66.2, 72.5, 76.2]
    oracle    = [92.6, 89.3, 90.1, 88.4, 88.1, 85.0, 78.0, 84.5, 84.0, 82.5]

    x = np.arange(len(langs))
    w = 0.21

    ax.bar(x - 1.5 * w, mbert,    w, color=C_BLUE,   label="mBERT")
    ax.bar(x - 0.5 * w, xlmr_base, w, color=C_PURPLE, label="XLM-R-base")
    ax.bar(x + 0.5 * w, xlmr_lg,  w, color=C_GREEN,  label="XLM-R-large")
    ax.bar(x + 1.5 * w, oracle,   w, color=C_AMBER,  alpha=0.85,
           label="In-language oracle")

    ax.axhline(91.0, color=C_BLUE, ls="--", lw=0.8, alpha=0.5)
    ax.text(9.4, 91.4, "English supervised (mBERT)",
            fontsize=8, color=C_BLUE, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=9)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_ylim(30, 100)
    ax.set_title("Zero-shot cross-lingual NER (WikiAnn): "
                 "trained only on English CoNLL",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    ax.legend(loc="lower left", fontsize=9, ncol=4, frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    _save(fig, "fig3_zero_shot_ner")


# ---------------------------------------------------------------------------
# Figure 4: Pivot language strategies
# ---------------------------------------------------------------------------
def fig4_pivot_strategies() -> None:
    """Three strategies for X -> Y when there is no direct labeled data."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    def panel(ax, title, nodes, edges, captions):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=C_DARK,
                     pad=8)
        for (x, y, label, color) in nodes:
            circ = plt.Circle((x, y), 0.55, color=color, ec="white", lw=1.6,
                              zorder=3)
            ax.add_patch(circ)
            ax.text(x, y, label, ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold", zorder=4)
        for (x1, y1, x2, y2, lab, col) in edges:
            _arrow(ax, (x1, y1), (x2, y2), color=col, lw=1.8)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, lab,
                    ha="center", fontsize=8.5, color=col, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=1.4))
        for (x, y, txt) in captions:
            ax.text(x, y, txt, ha="center", fontsize=9, color=C_DARK,
                    style="italic")

    # Strategy 1: source-pivot (translate target -> source for inference)
    nodes1 = [(1.0, 3.6, "X", C_BLUE),
              (3.0, 3.6, "EN", C_AMBER),
              (5.0, 3.6, "Y", C_PURPLE)]
    edges1 = [(1.45, 3.6, 2.55, 3.6, "MT", C_GRAY),
              (3.45, 3.6, 4.55, 3.6, "model", C_GREEN)]
    cap1 = [(3.0, 1.6, "translate target -> English,\nrun English model")]
    panel(axes[0], "Source-pivot (translate-test)", nodes1, edges1, cap1)

    # Strategy 2: target-pivot (translate source -> target, train on it)
    nodes2 = [(1.0, 3.6, "X", C_BLUE),
              (3.0, 3.6, "Y", C_PURPLE),
              (5.0, 3.6, "Y", C_PURPLE)]
    edges2 = [(1.45, 3.6, 2.55, 3.6, "MT data", C_GRAY),
              (3.45, 3.6, 4.55, 3.6, "fine-tune", C_GREEN)]
    cap2 = [(3.0, 1.6, "translate English data -> target,\nfine-tune on it")]
    panel(axes[1], "Target-pivot (translate-train)", nodes2, edges2, cap2)

    # Strategy 3: multi-source ensemble
    nodes3 = [(1.2, 4.3, "EN", C_BLUE),
              (1.2, 3.0, "ES", C_PURPLE),
              (1.2, 1.7, "DE", C_GREEN),
              (4.5, 3.0, "Y", C_AMBER)]
    edges3 = [(1.65, 4.3, 4.05, 3.15, "model_1", C_BLUE),
              (1.65, 3.0, 4.05, 3.00, "model_2", C_PURPLE),
              (1.65, 1.7, 4.05, 2.85, "model_3", C_GREEN)]
    cap3 = [(3.0, 0.6, "ensemble predictions from\nmultiple source-language models")]
    panel(axes[2], "Multi-source ensemble", nodes3, edges3, cap3)

    fig.suptitle("Pivot strategies when source -> target is direct-data-poor",
                 fontsize=13.5, fontweight="bold", color=C_DARK, y=1.02)

    plt.tight_layout()
    _save(fig, "fig4_pivot_strategies")


# ---------------------------------------------------------------------------
# Figure 5: BPE / SentencePiece subword overlap
# ---------------------------------------------------------------------------
def fig5_subword_tokenization() -> None:
    """Show how subwords of cognates overlap across languages."""
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6, 5.6, "Subword tokenization creates cross-lingual anchors",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)

    rows = [
        ("English",  C_BLUE,   "international",
         ["inter", "national"]),
        ("French",   C_GREEN,  "internationale",
         ["inter", "national", "e"]),
        ("German",   C_PURPLE, "internationaler",
         ["inter", "national", "er"]),
        ("Chinese",  C_AMBER,  "国际化",
         ["国", "际", "化"]),
    ]

    # Header
    ax.text(0.6, 4.85, "Language", fontsize=10, fontweight="bold",
            color=C_DARK)
    ax.text(2.3, 4.85, "Word", fontsize=10, fontweight="bold", color=C_DARK)
    ax.text(5.0, 4.85, "Subword pieces (SentencePiece)",
            fontsize=10, fontweight="bold", color=C_DARK)

    shared_tokens = {"inter", "national"}

    for i, (lang, color, word, pieces) in enumerate(rows):
        y = 4.0 - i * 0.85
        # Language tag
        _box(ax, (0.4, y - 0.22), 1.5, 0.5, lang, color, fontsize=10,
             fontweight="bold", alpha=0.9)
        # Word
        ax.text(2.3, y, word, fontsize=11, color=C_DARK)
        # Pieces
        x = 5.0
        for p in pieces:
            is_shared = p in shared_tokens
            fc = C_GREEN if is_shared else C_LIGHT
            tc = "white" if is_shared else C_DARK
            piece_w = max(0.55, 0.25 + 0.20 * len(p))
            _box(ax, (x, y - 0.22), piece_w, 0.5,
                 p, fc, fontsize=10, text_color=tc,
                 fontweight="bold" if is_shared else "normal",
                 alpha=0.95 if is_shared else 0.85)
            x += piece_w + 0.18

    # Legend / footnote
    _box(ax, (5.0, 0.25), 1.0, 0.4, "shared", C_GREEN, fontsize=9)
    ax.text(6.15, 0.45,
            "= subword shared across languages -> "
            "same embedding row -> implicit alignment",
            fontsize=9.5, color=C_DARK, va="center")

    _save(fig, "fig5_subword_tokenization")


# ---------------------------------------------------------------------------
# Figure 6: Resource curve (corpus size vs zero-shot accuracy)
# ---------------------------------------------------------------------------
def fig6_resource_curve() -> None:
    """How does pre-training corpus size correlate with zero-shot perf?"""
    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    # Approximate corpus sizes (GB) for a sample of XLM-R languages
    langs   = ["en", "ru", "es", "de", "fr", "zh", "ar", "vi",
               "hi", "tr", "th", "sw", "ur", "yo"]
    corpus  = np.array([2500, 280, 220, 200, 180,  80,  50,  37,
                          25,  21,  20,  1.5, 1.2, 0.05])  # GB
    # Zero-shot XNLI accuracy with default sampling
    xnli_def = np.array([89.0, 78.0, 80.6, 79.4, 79.7, 76.7, 73.8, 75.4,
                         72.5, 73.0, 71.4, 65.0, 62.1, 53.0])
    # With balanced alpha=0.7 sampling
    xnli_bal = xnli_def + np.array([-0.4, 0.6, 0.5, 0.4, 0.4, 0.7, 1.0, 0.9,
                                     1.4, 1.6, 1.8, 3.5, 4.2, 6.8])

    ax.scatter(corpus, xnli_def, s=120, color=C_BLUE, alpha=0.85,
               edgecolor="white", linewidth=1.2, label="default sampling",
               zorder=3)
    ax.scatter(corpus, xnli_bal, s=120, color=C_GREEN, alpha=0.85,
               edgecolor="white", linewidth=1.2,
               label=r"balanced, $\alpha=0.7$", zorder=3, marker="s")

    # Connect pairs to visualize the gain
    for c, a, b in zip(corpus, xnli_def, xnli_bal):
        ax.plot([c, c], [a, b], color=C_GRAY, lw=0.8, alpha=0.6, zorder=1)

    # Annotate languages
    for c, a, l in zip(corpus, xnli_def, langs):
        ax.annotate(l, (c, a), xytext=(5, -10), textcoords="offset points",
                    fontsize=8, color=C_DARK)

    # Trend line (log-linear)
    xs = np.logspace(np.log10(corpus.min()), np.log10(corpus.max()), 100)
    coef = np.polyfit(np.log10(corpus), xnli_def, 1)
    ax.plot(xs, coef[0] * np.log10(xs) + coef[1], color=C_PURPLE,
            ls="--", lw=1.4, alpha=0.7, label="log-linear fit (default)")

    ax.set_xscale("log")
    ax.set_xlabel("Pre-training corpus size (GB, log scale)", fontsize=11)
    ax.set_ylabel("Zero-shot XNLI accuracy (%)", fontsize=11)
    ax.set_title("High-resource plateau, low-resource cliff: "
                 "balanced sampling helps the tail",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(45, 95)

    _save(fig, "fig6_resource_curve")


# ---------------------------------------------------------------------------
# Figure 7: Translate-based vs alignment-based pipelines
# ---------------------------------------------------------------------------
def fig7_translate_vs_align() -> None:
    """Pipeline diagram + cost / latency / quality bars."""
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.45,
                          wspace=0.35)

    # ---- Top: three pipelines ----
    methods = [
        ("Translate-train",  C_BLUE,
         ["EN labels", "MT -> Y", "fine-tune\nmBERT/XLM-R", "deploy in Y"]),
        ("Translate-test",   C_PURPLE,
         ["EN labels", "fine-tune EN model", "MT Y -> EN at\ninference",
          "deploy"]),
        ("Zero-shot align",  C_GREEN,
         ["EN labels", "fine-tune mBERT/XLM-R\non English",
          "model already aligned", "deploy in Y directly"]),
    ]

    for i, (title, color, steps) in enumerate(methods):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color,
                     pad=10)
        n = len(steps)
        for k, step in enumerate(steps):
            y = 5.0 - k * 1.20
            _box(ax, (1.0, y - 0.45), 8.0, 0.85, step, color,
                 fontsize=10, alpha=0.85 - 0.05 * k)
            if k < n - 1:
                _arrow(ax, (5.0, y - 0.5), (5.0, y - 0.75),
                       color=C_DARK, lw=1.2)

    # ---- Bottom: cost / latency / quality bars ----
    ax_bar = fig.add_subplot(gs[1, :])
    metrics = ["Training cost", "Inference latency", "Target quality",
               "MT dependency"]
    # 0..5 qualitative scale
    tt    = [4, 1, 4, 4]
    tte   = [1, 3, 3, 4]
    align = [1, 1, 3, 0]

    x = np.arange(len(metrics))
    w = 0.25
    ax_bar.bar(x - w, tt,    w, color=C_BLUE,   label="Translate-train")
    ax_bar.bar(x,     tte,   w, color=C_PURPLE, label="Translate-test")
    ax_bar.bar(x + w, align, w, color=C_GREEN,  label="Zero-shot align")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, fontsize=10)
    ax_bar.set_yticks([0, 1, 2, 3, 4, 5])
    ax_bar.set_yticklabels(["none", "low", "", "med", "", "high"],
                           fontsize=9)
    ax_bar.legend(loc="upper right", fontsize=9, frameon=True, ncol=3)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_axisbelow(True)
    ax_bar.set_title("Trade-offs across pipelines",
                     fontsize=11.5, fontweight="bold", color=C_DARK, pad=6)

    fig.suptitle("Translation-based vs alignment-based cross-lingual transfer",
                 fontsize=13.5, fontweight="bold", color=C_DARK, y=0.99)

    _save(fig, "fig7_translate_vs_align")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Part 11: Cross-Lingual Transfer")
    print(f"  EN dir: {EN_DIR}")
    print(f"  ZH dir: {ZH_DIR}")
    fig1_embedding_space()
    fig2_xlmr_architecture()
    fig3_zero_shot_ner()
    fig4_pivot_strategies()
    fig5_subword_tokenization()
    fig6_resource_curve()
    fig7_translate_vs_align()
    print("Done.")


if __name__ == "__main__":
    main()
