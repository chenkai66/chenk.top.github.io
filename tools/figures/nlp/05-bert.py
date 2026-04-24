"""
Figure generation script for NLP Part 05:
BERT and Pre-trained Models.

Generates 7 figures shared by the EN and ZH articles. Each figure
isolates one pedagogical idea so the reader can connect the math, the
code, and the picture without ambiguity.

Figures:
    fig1_bert_architecture       Bidirectional Transformer encoder stack
                                 with input embedding composition (token +
                                 segment + position) and the [CLS]/[SEP]
                                 special tokens highlighted.
    fig2_mlm_corruption          Masked Language Modeling 80/10/10 rule:
                                 a sentence with three masked positions
                                 visualised as MASK / random / unchanged
                                 corruption, plus the prediction head.
    fig3_nsp                     Next Sentence Prediction: positive (IsNext)
                                 vs negative (NotNext) sentence pairs feeding
                                 the [CLS] head, with the 50/50 sampling rule.
    fig4_finetune_pipeline       Pre-train once -> fine-tune for many
                                 downstream tasks: classification, NER, QA,
                                 sentence-pair, sharing the same backbone.
    fig5_variants_comparison     BERT vs RoBERTa vs ALBERT vs ELECTRA on
                                 three axes: parameters, GLUE score,
                                 training-token efficiency. Stacked bars +
                                 annotation of each model's key idea.
    fig6_wordpiece               WordPiece tokenization: how rare or
                                 morphologically rich words are split into
                                 subword pieces, with frequency-driven
                                 vocabulary composition.
    fig7_glue_benchmark          BERT-base vs BERT-large vs prior SOTA on
                                 the eight GLUE tasks, showing the absolute
                                 jump that the 2018 paper reported.

Usage:
    python3 scripts/figures/nlp/05-bert.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders so the
    markdown image references stay consistent across languages.
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
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_RED = COLORS["danger"]
C_GRAY = COLORS["gray"]
C_LIGHT = COLORS["light"]
C_DARK = COLORS["ink"]
C_BG = COLORS["bg"]


PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "nlp"
    / "bert-pretrained-models"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "nlp"
    / "05-BERT与预训练模型"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(ax, x, y, w, h, label, color, fc=None, fontsize=10,
         weight="bold", text_color=None):
    fc = fc if fc is not None else C_BG
    text_color = text_color or C_DARK
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.5, edgecolor=color, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, weight=weight)


def _arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=1.6, style="-|>",
           mutation=12):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=mutation, color=color, lw=lw)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: BERT bidirectional encoder architecture
# ---------------------------------------------------------------------------
def fig1_bert_architecture() -> None:
    fig, (ax_emb, ax_enc) = plt.subplots(
        2, 1, figsize=(11, 7.6),
        gridspec_kw={"height_ratios": [1.0, 1.6], "hspace": 0.35},
    )

    # ---- top: input embedding composition ----
    ax_emb.set_xlim(0, 11)
    ax_emb.set_ylim(0, 4)
    ax_emb.axis("off")
    tokens = ["[CLS]", "the", "river", "bank", "[SEP]", "money", "##s", "[SEP]"]
    n = len(tokens)
    pad = 0.6
    cell_w = (11 - 2 * pad - 1.4) / n
    x0 = pad + 1.4

    # row labels
    rows = [
        ("Token",   "#dbeafe", C_BLUE),
        ("Segment", "#ede9fe", C_PURPLE),
        ("Position","#d1fae5", C_GREEN),
    ]
    seg_ids = ["A", "A", "A", "A", "A", "B", "B", "B"]
    for r_idx, (rname, fc, ec) in enumerate(rows):
        y = 2.6 - r_idx * 0.85
        ax_emb.text(pad, y + 0.25, rname + " emb.", ha="left", va="center",
                    color=ec, fontsize=10, weight="bold")
        for i, tok in enumerate(tokens):
            x = x0 + i * cell_w
            ax_emb.add_patch(FancyBboxPatch(
                (x + 0.05, y), cell_w - 0.1, 0.55,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=1.0, edgecolor=ec, facecolor=fc,
            ))
            if r_idx == 0:
                lbl = tok
            elif r_idx == 1:
                lbl = f"E_{seg_ids[i]}"
            else:
                lbl = f"E_{i}"
            ax_emb.text(x + cell_w / 2, y + 0.27, lbl, ha="center",
                        va="center", color=C_DARK, fontsize=9,
                        weight="bold" if r_idx == 0 else "normal")
        # plus signs between rows
        if r_idx < 2:
            ax_emb.text(x0 - 0.35, y - 0.15, "+", ha="center",
                        color=C_DARK, fontsize=14, weight="bold")

    ax_emb.text(5.5, 3.65, "Input = Token + Segment + Position",
                ha="center", color=C_DARK, fontsize=12, weight="bold")
    ax_emb.text(5.5, 0.3,
                "Two sentences packed together; [CLS] aggregates the "
                "whole sequence for classification heads.",
                ha="center", color=C_GRAY, fontsize=9, style="italic")

    # ---- bottom: encoder stack with bidirectional attention ----
    ax_enc.set_xlim(0, 11)
    ax_enc.set_ylim(0, 5.4)
    ax_enc.axis("off")

    # token boxes at the bottom (input)
    n_tok = 6
    tok_labels = ["[CLS]", "the", "river", "bank", "is", "[SEP]"]
    tw = 0.95
    tx0 = 1.4
    for i, t in enumerate(tok_labels):
        x = tx0 + i * (tw + 0.15)
        _box(ax_enc, x, 0.2, tw, 0.55, t, C_BLUE, fc="#dbeafe", fontsize=9)

    # encoder layer boxes (4 stacked, label "x12")
    layer_y = [1.2, 2.0, 2.8, 3.6]
    for ly in layer_y:
        ax_enc.add_patch(FancyBboxPatch(
            (1.2, ly), 7.4, 0.6,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            linewidth=1.4, edgecolor=C_BLUE, facecolor=C_BG,
        ))
        ax_enc.text(4.9, ly + 0.30,
                    "Multi-Head Self-Attention  +  Feed-Forward",
                    ha="center", va="center", color=C_DARK,
                    fontsize=10, weight="bold")
    ax_enc.text(8.85, 2.4, "x 12\n(BERT-Base)\nx 24\n(BERT-Large)",
                ha="left", va="center", color=C_PURPLE,
                fontsize=10, weight="bold")

    # bidirectional attention arrows on first layer (illustrative)
    arc_y = 1.0
    centres = [tx0 + i * (tw + 0.15) + tw / 2 for i in range(n_tok)]
    focus = 3  # "bank"
    for j, cx in enumerate(centres):
        if j == focus:
            continue
        col = C_AMBER if abs(j - focus) <= 1 else C_GRAY
        _arrow(ax_enc, centres[focus], arc_y - 0.05, cx, arc_y - 0.05,
               color=col, lw=1.0, style="-", mutation=6)
    ax_enc.text(centres[focus], 0.95, "bidirectional attention\n"
                "from \"bank\" to all positions",
                ha="center", color=C_AMBER, fontsize=8.5, style="italic")

    # arrows between layers
    for ly in [0.78, 1.8, 2.6, 3.4]:
        _arrow(ax_enc, 4.9, ly, 4.9, ly + 0.40, color=C_BLUE, lw=1.4)

    # Output: contextual representations
    out_y = 4.5
    for i, t in enumerate(tok_labels):
        x = tx0 + i * (tw + 0.15)
        _box(ax_enc, x, out_y, tw, 0.55, f"h_{i}", C_PURPLE,
             fc="#ede9fe", fontsize=9)
    ax_enc.text(9.0, out_y + 0.27, "contextual\nvectors",
                ha="left", va="center", color=C_PURPLE, fontsize=9,
                style="italic")

    fig.suptitle("BERT: bidirectional Transformer encoder",
                 fontsize=13.5, weight="bold", y=0.995)
    _save(fig, "fig1_bert_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Masked Language Modeling 80/10/10 rule
# ---------------------------------------------------------------------------
def fig2_mlm_corruption() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    tokens = ["the", "cat", "sat", "on", "the", "mat", "and", "purred"]
    n = len(tokens)
    cell_w = 1.05
    x0 = (11 - n * cell_w) / 2

    # input row
    y_in = 5.2
    ax.text(x0 - 0.2, y_in + 0.27, "Input",
            ha="right", va="center", color=C_DARK, fontsize=10,
            weight="bold")
    for i, t in enumerate(tokens):
        _box(ax, x0 + i * cell_w, y_in, cell_w - 0.08, 0.55,
             t, C_GRAY, fc="#f1f5f9", fontsize=10)

    # corruption row -- pick 3 positions out of 8 (~15% would be ~1.2; we
    # show 3 to clearly illustrate all three rules)
    y_cor = 4.0
    ax.text(x0 - 0.2, y_cor + 0.27, "Corrupt 15%",
            ha="right", va="center", color=C_DARK, fontsize=10,
            weight="bold")

    # mapping: index -> (label, color, rule)
    corruption = {
        1: ("[MASK]", C_AMBER, "80%: replace with [MASK]"),
        4: ("dog",    C_RED,   "10%: replace with random token"),
        6: ("and",    C_GREEN, "10%: keep unchanged"),
    }
    for i, t in enumerate(tokens):
        if i in corruption:
            new_t, col, _ = corruption[i]
            _box(ax, x0 + i * cell_w, y_cor, cell_w - 0.08, 0.55,
                 new_t, col, fc=_mix_with_white(col, 0.85), fontsize=10)
        else:
            _box(ax, x0 + i * cell_w, y_cor, cell_w - 0.08, 0.55,
                 t, C_GRAY, fc="#f1f5f9", fontsize=10)

    # vertical arrows from input to corruption
    for i in range(n):
        col = corruption[i][1] if i in corruption else C_GRAY
        _arrow(ax, x0 + i * cell_w + (cell_w - 0.08) / 2, y_in,
               x0 + i * cell_w + (cell_w - 0.08) / 2, y_cor + 0.55,
               color=col, lw=1.3)

    # encoder block
    enc_y = 2.6
    ax.add_patch(FancyBboxPatch(
        (x0, enc_y), n * cell_w - 0.08, 0.7,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=1.6, edgecolor=C_BLUE, facecolor="#dbeafe",
    ))
    ax.text(x0 + (n * cell_w - 0.08) / 2, enc_y + 0.35,
            "BERT encoder (12 or 24 layers)",
            ha="center", va="center", color=C_BLUE, fontsize=11,
            weight="bold")

    # arrow from corruption to encoder
    for i in range(n):
        _arrow(ax, x0 + i * cell_w + (cell_w - 0.08) / 2, y_cor,
               x0 + i * cell_w + (cell_w - 0.08) / 2, enc_y + 0.7,
               color=C_BLUE, lw=1.0)

    # prediction row only at masked positions
    y_pred = 1.4
    ax.text(x0 - 0.2, y_pred + 0.27, "Predict",
            ha="right", va="center", color=C_DARK, fontsize=10,
            weight="bold")
    for i in range(n):
        if i in corruption:
            true_token = tokens[i]
            _box(ax, x0 + i * cell_w, y_pred, cell_w - 0.08, 0.55,
                 true_token, C_BLUE, fc="#dbeafe", fontsize=10)
            _arrow(ax, x0 + i * cell_w + (cell_w - 0.08) / 2, enc_y,
                   x0 + i * cell_w + (cell_w - 0.08) / 2, y_pred + 0.55,
                   color=C_BLUE, lw=1.2)

    # legend / rule explanation
    rules = [
        ("80% [MASK]", C_AMBER, "predict from context"),
        ("10% random", C_RED,   "stay robust to noise"),
        ("10% kept",   C_GREEN, "use context even if unmasked"),
    ]
    for k, (lbl, col, hint) in enumerate(rules):
        x = 0.6 + k * 3.5
        ax.add_patch(FancyBboxPatch(
            (x, 0.2), 3.2, 0.7,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.2, edgecolor=col,
            facecolor=_mix_with_white(col, 0.9),
        ))
        ax.text(x + 0.18, 0.66, lbl, ha="left", va="center",
                color=col, fontsize=10, weight="bold")
        ax.text(x + 0.18, 0.38, hint, ha="left", va="center",
                color=C_DARK, fontsize=9, style="italic")

    fig.suptitle("Masked Language Modeling: the 80 / 10 / 10 rule",
                 fontsize=13.5, weight="bold", y=0.99)
    _save(fig, "fig2_mlm_corruption")


def _mix_with_white(hex_color: str, ratio: float) -> str:
    """Blend hex color toward white. ratio=0 -> color, ratio=1 -> white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * ratio)
    g = int(g + (255 - g) * ratio)
    b = int(b + (255 - b) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Figure 3: Next Sentence Prediction
# ---------------------------------------------------------------------------
def fig3_nsp() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    pairs = [
        {
            "ax": axes[0],
            "title": "Positive pair  (50%)",
            "label": "IsNext",
            "color": C_GREEN,
            "a": "She opened the umbrella",
            "b": "because the rain was heavy.",
        },
        {
            "ax": axes[1],
            "title": "Negative pair  (50%)",
            "label": "NotNext",
            "color": C_RED,
            "a": "She opened the umbrella",
            "b": "Penguins waddle on the ice.",
        },
    ]

    for p in pairs:
        ax = p["ax"]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6.4)
        ax.axis("off")
        ax.set_title(p["title"], fontsize=12, weight="bold",
                     color=p["color"], pad=8)

        # input sequence with [CLS] sentA [SEP] sentB [SEP]
        tokens = (["[CLS]"] + p["a"].split() + ["[SEP]"]
                  + p["b"].split() + ["[SEP]"])
        n = len(tokens)
        cell_w = 9.4 / n
        x0 = 0.3
        y_in = 1.2
        for i, t in enumerate(tokens):
            is_special = t in ("[CLS]", "[SEP]")
            ec = p["color"] if t == "[CLS]" else C_GRAY
            fc = _mix_with_white(p["color"], 0.85) if t == "[CLS]" else "#f1f5f9"
            _box(ax, x0 + i * cell_w, y_in, cell_w - 0.08, 0.55,
                 t, ec, fc=fc, fontsize=8 if not is_special else 8.5,
                 weight="bold" if is_special else "normal")

        # encoder
        enc_y = 2.5
        ax.add_patch(FancyBboxPatch(
            (x0, enc_y), 9.4 - 0.08, 0.7,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            linewidth=1.6, edgecolor=C_BLUE, facecolor="#dbeafe",
        ))
        ax.text(x0 + (9.4 - 0.08) / 2, enc_y + 0.35,
                "BERT encoder", ha="center", va="center",
                color=C_BLUE, fontsize=10, weight="bold")

        # arrows
        for i in range(n):
            _arrow(ax, x0 + i * cell_w + (cell_w - 0.08) / 2, y_in + 0.55,
                   x0 + i * cell_w + (cell_w - 0.08) / 2, enc_y,
                   color=C_GRAY, lw=0.8)

        # CLS hidden vector
        cls_y = 4.0
        _box(ax, 0.3, cls_y, 1.4, 0.6,
             "h_[CLS]", C_PURPLE, fc="#ede9fe", fontsize=10)
        _arrow(ax, x0 + (cell_w - 0.08) / 2, enc_y + 0.7,
               1.0, cls_y, color=C_PURPLE, lw=1.4)

        # head + softmax box
        _box(ax, 2.2, cls_y, 3.0, 0.6,
             "Linear + softmax", C_PURPLE,
             fc="#ede9fe", fontsize=10)
        _arrow(ax, 1.7, cls_y + 0.3, 2.2, cls_y + 0.3, color=C_PURPLE,
               lw=1.4)

        # prediction
        _box(ax, 5.7, cls_y, 1.8, 0.6,
             p["label"], p["color"],
             fc=_mix_with_white(p["color"], 0.8), fontsize=11)
        _arrow(ax, 5.2, cls_y + 0.3, 5.7, cls_y + 0.3,
               color=p["color"], lw=1.6)

        # truth annotation
        ax.text(5, 5.3, "[CLS]'s vector decides if B follows A",
                ha="center", color=C_DARK, fontsize=9.5, style="italic")
        ax.text(5, 5.85,
                "Sample sentence pairs 50% real, 50% random",
                ha="center", color=C_GRAY, fontsize=9, style="italic")

    fig.suptitle("Next Sentence Prediction (NSP)",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_nsp")


# ---------------------------------------------------------------------------
# Figure 4: Fine-tuning pipeline -- one backbone, many tasks
# ---------------------------------------------------------------------------
def fig4_finetune_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # Pre-training side
    ax.add_patch(FancyBboxPatch(
        (0.3, 4.7), 4.0, 1.4,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=1.6, edgecolor=C_BLUE, facecolor="#dbeafe",
    ))
    ax.text(2.3, 5.7, "Stage 1: Pre-training", ha="center",
            color=C_BLUE, fontsize=11.5, weight="bold")
    ax.text(2.3, 5.20,
            "MLM + NSP on BooksCorpus (800M)\n+ English Wikipedia (2.5B)",
            ha="center", color=C_DARK, fontsize=9.5)

    # Backbone
    _box(ax, 1.0, 2.6, 2.6, 1.6,
         "BERT\nbackbone\n$\\theta_{\\mathrm{pre}}$",
         C_BLUE, fc="#bfdbfe", fontsize=11)
    _arrow(ax, 2.3, 4.7, 2.3, 4.2, color=C_BLUE, lw=1.8)

    # Stage 2 banner
    ax.add_patch(FancyBboxPatch(
        (4.6, 4.7), 6.6, 1.4,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=1.6, edgecolor=C_PURPLE, facecolor="#ede9fe",
    ))
    ax.text(7.9, 5.7, "Stage 2: Fine-tuning per task",
            ha="center", color=C_PURPLE, fontsize=11.5, weight="bold")
    ax.text(7.9, 5.20,
            "Add a small head, train end-to-end with lr=2e-5 for 2-4 epochs",
            ha="center", color=C_DARK, fontsize=9.5)

    # Bridge arrow
    _arrow(ax, 3.6, 3.4, 4.6, 3.4, color=C_DARK, lw=1.8)
    ax.text(4.1, 3.6, "copy weights",
            ha="center", color=C_DARK, fontsize=9, style="italic")

    # Four downstream task cards
    tasks = [
        {
            "title": "Sentence\nclassification",
            "head": "[CLS] -> Linear",
            "ex":   "sentiment, spam, NLI",
            "color": C_PURPLE,
        },
        {
            "title": "Token tagging\n(NER, POS)",
            "head": "h_t -> Linear",
            "ex":   "PER / ORG / LOC",
            "color": C_GREEN,
        },
        {
            "title": "Question\nanswering",
            "head": "h_t -> start, end",
            "ex":   "SQuAD spans",
            "color": C_AMBER,
        },
        {
            "title": "Sentence-pair\ntasks",
            "head": "[CLS] -> Linear",
            "ex":   "STS, paraphrase",
            "color": C_BLUE,
        },
    ]
    card_w, card_h = 1.55, 2.2
    gap = 0.10
    total_w = 4 * card_w + 3 * gap
    start_x = 4.6 + (6.6 - total_w) / 2
    for i, t in enumerate(tasks):
        x = start_x + i * (card_w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, 1.5), card_w, card_h,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            linewidth=1.5, edgecolor=t["color"],
            facecolor=_mix_with_white(t["color"], 0.9),
        ))
        ax.text(x + card_w / 2, 3.35, t["title"],
                ha="center", color=t["color"], fontsize=10,
                weight="bold")
        # head box
        ax.add_patch(FancyBboxPatch(
            (x + 0.12, 2.55), card_w - 0.24, 0.45,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.0, edgecolor=t["color"],
            facecolor="white",
        ))
        ax.text(x + card_w / 2, 2.78, t["head"],
                ha="center", color=C_DARK, fontsize=8.5, weight="bold")
        ax.text(x + card_w / 2, 1.95, t["ex"],
                ha="center", color=C_GRAY, fontsize=8.5, style="italic")
        # arrow from backbone area into card
        _arrow(ax, 3.6, 3.4, x + card_w / 2, 3.05,
               color=t["color"], lw=1.0, mutation=8)

    # Bottom note
    ax.text(5.75, 0.5,
            "One pre-trained backbone is reused for many tasks. "
            "Heads are tiny; most parameters come from BERT.",
            ha="center", color=C_DARK, fontsize=10, style="italic")

    fig.suptitle("Fine-tuning BERT for downstream tasks",
                 fontsize=13.5, weight="bold", y=1.0)
    _save(fig, "fig4_finetune_pipeline")


# ---------------------------------------------------------------------------
# Figure 5: BERT vs RoBERTa vs ALBERT vs ELECTRA
# ---------------------------------------------------------------------------
def fig5_variants_comparison() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))

    models = ["BERT", "RoBERTa", "ALBERT", "ELECTRA"]
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

    # ---- (a) parameters (large variants) ----
    params = [340, 355, 235, 335]  # millions, large/xxlarge config
    ax = axes[0]
    bars = ax.bar(models, params, color=colors, edgecolor="white", lw=1.5)
    ax.set_ylabel("Parameters (millions)", fontsize=10.5)
    ax.set_title("(a) Model size (large config)",
                 fontsize=11, weight="bold", color=C_DARK)
    ax.set_ylim(0, max(params) * 1.25)
    for b, v in zip(bars, params):
        ax.text(b.get_x() + b.get_width() / 2, v + 6,
                f"{v}M", ha="center", color=C_DARK,
                fontsize=10, weight="bold")
    ax.grid(axis="x", visible=False)

    # ---- (b) GLUE score ----
    glue = [80.5, 88.5, 89.4, 89.4]  # approx large-model GLUE averages
    ax = axes[1]
    bars = ax.bar(models, glue, color=colors, edgecolor="white", lw=1.5)
    ax.set_ylabel("GLUE average", fontsize=10.5)
    ax.set_title("(b) GLUE benchmark (higher is better)",
                 fontsize=11, weight="bold", color=C_DARK)
    ax.set_ylim(75, 92)
    for b, v in zip(bars, glue):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25,
                f"{v}", ha="center", color=C_DARK,
                fontsize=10, weight="bold")
    ax.grid(axis="x", visible=False)

    # ---- (c) training-token efficiency: GLUE per 1B training tokens ----
    # Pre-training tokens (rough orders of magnitude, billions):
    # BERT 3.3B, RoBERTa 160B, ALBERT 16B, ELECTRA 33B (base reported)
    train_tokens = [3.3, 160, 16, 33]
    eff = [g / t for g, t in zip(glue, train_tokens)]
    ax = axes[2]
    bars = ax.bar(models, eff, color=colors, edgecolor="white", lw=1.5)
    ax.set_ylabel("GLUE / B training tokens", fontsize=10.5)
    ax.set_title("(c) Pre-training data efficiency",
                 fontsize=11, weight="bold", color=C_DARK)
    ax.set_ylim(0, max(eff) * 1.25)
    for b, v in zip(bars, eff):
        ax.text(b.get_x() + b.get_width() / 2, v + max(eff) * 0.03,
                f"{v:.1f}", ha="center", color=C_DARK,
                fontsize=10, weight="bold")
    ax.grid(axis="x", visible=False)

    # bottom annotations as a row of mini cards
    ideas = [
        ("BERT",    "Bidirectional MLM + NSP"),
        ("RoBERTa", "Drop NSP, dynamic mask, more data"),
        ("ALBERT",  "Param sharing + factorised embed"),
        ("ELECTRA", "RTD: train on 100% of tokens"),
    ]
    fig.subplots_adjust(bottom=0.30)
    for i, (m, idea) in enumerate(ideas):
        x = 0.07 + i * 0.235
        fig.text(x, 0.10, m, color=colors[i], fontsize=10.5,
                 weight="bold")
        fig.text(x, 0.04, idea, color=C_DARK, fontsize=9.5,
                 style="italic")

    fig.suptitle("BERT family: same backbone, different recipes",
                 fontsize=13.5, weight="bold", y=1.0)
    _save(fig, "fig5_variants_comparison")


# ---------------------------------------------------------------------------
# Figure 6: WordPiece tokenization
# ---------------------------------------------------------------------------
def fig6_wordpiece() -> None:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 6.4),
        gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.50},
    )

    # ---- top: a few example words split into pieces ----
    ax_top.set_xlim(0, 11)
    ax_top.set_ylim(0, 4.2)
    ax_top.axis("off")
    examples = [
        ("playing",        ["play", "##ing"]),
        ("unbelievable",   ["un", "##bel", "##iev", "##able"]),
        ("transformer",    ["transform", "##er"]),
        ("Tokyo2024",      ["tokyo", "##20", "##24"]),
    ]
    row_h = 0.85
    for r, (word, pieces) in enumerate(examples):
        y = 3.4 - r * row_h
        # original word
        _box(ax_top, 0.3, y, 2.4, 0.55, word, C_GRAY,
             fc="#f1f5f9", fontsize=10.5)
        _arrow(ax_top, 2.7, y + 0.27, 3.3, y + 0.27,
               color=C_DARK, lw=1.2)
        # pieces
        x = 3.3
        for i, p in enumerate(pieces):
            is_continuation = p.startswith("##")
            ec = C_PURPLE if is_continuation else C_BLUE
            fc = "#ede9fe" if is_continuation else "#dbeafe"
            w = max(1.0, 0.18 * len(p) + 0.5)
            _box(ax_top, x, y, w, 0.55, p, ec, fc=fc, fontsize=10.5)
            x += w + 0.10

    ax_top.text(5.5, 3.95,
                "Words -> WordPiece sub-tokens "
                "(\"##\" marks a continuation)",
                ha="center", color=C_DARK, fontsize=11.5, weight="bold")

    # legend
    ax_top.add_patch(Rectangle((0.3, 0.05), 0.35, 0.35,
                               facecolor="#dbeafe", edgecolor=C_BLUE))
    ax_top.text(0.75, 0.22, "word-start piece",
                va="center", color=C_DARK, fontsize=9.5)
    ax_top.add_patch(Rectangle((3.0, 0.05), 0.35, 0.35,
                               facecolor="#ede9fe", edgecolor=C_PURPLE))
    ax_top.text(3.45, 0.22, "##continuation piece",
                va="center", color=C_DARK, fontsize=9.5)
    ax_top.text(6.5, 0.22,
                "Vocabulary size: ~30K (English BERT)",
                va="center", color=C_GRAY, fontsize=9.5, style="italic")

    # ---- bottom: vocabulary composition / OOV resilience ----
    categories = ["Whole-word\npieces", "##continuation\npieces",
                  "Single-char\npieces", "Special tokens"]
    counts = [18000, 11500, 480, 5]  # illustrative
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    bars = ax_bot.barh(categories, counts, color=colors,
                       edgecolor="white", lw=1.5)
    ax_bot.set_xlabel("Number of vocabulary entries", fontsize=10.5)
    ax_bot.set_title("Composition of the BERT-base WordPiece vocabulary "
                     "(approx.)",
                     fontsize=11, weight="bold", color=C_DARK)
    ax_bot.set_xlim(0, max(counts) * 1.18)
    for b, v in zip(bars, counts):
        ax_bot.text(v + max(counts) * 0.01,
                    b.get_y() + b.get_height() / 2,
                    f"{v:,}", va="center", color=C_DARK,
                    fontsize=10, weight="bold")
    ax_bot.grid(axis="y", visible=False)
    ax_bot.invert_yaxis()

    fig.suptitle("WordPiece tokenization: balancing coverage and vocabulary "
                 "size",
                 fontsize=13.5, weight="bold", y=1.0)
    _save(fig, "fig6_wordpiece")


# ---------------------------------------------------------------------------
# Figure 7: GLUE benchmark performance
# ---------------------------------------------------------------------------
def fig7_glue_benchmark() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    # 8 GLUE tasks; numbers reflect figures from the BERT paper (Devlin
    # et al., 2018) for the prior SOTA, BERT-Base, and BERT-Large.
    tasks = ["MNLI-m", "QQP", "QNLI", "SST-2", "CoLA",
             "STS-B", "MRPC", "RTE"]
    prior_sota = [80.6, 66.1, 82.3, 93.2, 35.0, 81.0, 86.0, 61.7]
    bert_base  = [84.6, 71.2, 90.5, 93.5, 52.1, 85.8, 88.9, 66.4]
    bert_large = [86.7, 72.1, 92.7, 94.9, 60.5, 86.5, 89.3, 70.1]

    x = np.arange(len(tasks))
    w = 0.27

    ax.bar(x - w, prior_sota, w, label="Prior SOTA (pre-BERT)",
           color=C_GRAY, edgecolor="white", lw=1.2)
    ax.bar(x,     bert_base,  w, label="BERT-Base (110M)",
           color=C_BLUE, edgecolor="white", lw=1.2)
    ax.bar(x + w, bert_large, w, label="BERT-Large (340M)",
           color=C_PURPLE, edgecolor="white", lw=1.2)

    # arrow annotations highlighting the largest jumps
    for i, t in enumerate(tasks):
        gain = bert_large[i] - prior_sota[i]
        if gain >= 8:
            ax.annotate(f"+{gain:.1f}",
                        xy=(x[i] + w, bert_large[i]),
                        xytext=(x[i] + w, bert_large[i] + 6),
                        ha="center", color=C_GREEN, fontsize=9.5,
                        weight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=C_GREEN,
                                        lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylabel("Score (task-specific metric)", fontsize=10.5)
    ax.set_ylim(0, 110)
    ax.set_title("BERT on GLUE: bidirectional pre-training resets the "
                 "state of the art (2018)",
                 fontsize=12.5, weight="bold", color=C_DARK, pad=10)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(axis="x", visible=False)

    ax.text(len(tasks) - 1, -14,
            "Numbers from Devlin et al., 2018 (BERT paper, GLUE test set).",
            ha="right", color=C_GRAY, fontsize=9, style="italic")

    fig.tight_layout()
    _save(fig, "fig7_glue_benchmark")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_bert_architecture()
    fig2_mlm_corruption()
    fig3_nsp()
    fig4_finetune_pipeline()
    fig5_variants_comparison()
    fig6_wordpiece()
    fig7_glue_benchmark()
    print(f"Saved 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
