"""
Figure generator for NLP Part 02 — Word Embeddings & Language Models.

Generates seven production-quality figures and writes copies into BOTH the EN
and ZH asset folders so the same image is referenced from either language post.

Figures:
  fig1_skipgram_architecture.png    Skip-gram network with the embedding lookup
                                     -> projection -> softmax bottleneck.
  fig2_word_analogy.png             Vector analogy plot: king-man+woman~queen
                                     and Paris-France+Italy~Rome.
  fig3_tsne_clusters.png            t-SNE-style scatter where animals, royalty,
                                     countries and tech words form clusters.
  fig4_glove_factorization.png      Co-occurrence matrix factorised into a word
                                     and a context embedding matrix.
  fig5_lm_perplexity.png            N-gram vs neural LM perplexity as the
                                     training corpus grows.
  fig6_negative_sampling.png        Anchor + positive pulled together, k random
                                     negatives pushed apart.
  fig7_subword_fasttext.png         FastText character n-grams summed into a
                                     single word embedding.

Style contract:
  - matplotlib seaborn-v0_8-whitegrid
  - dpi = 150
  - palette: #2563eb (blue), #7c3aed (violet), #10b981 (emerald), #f59e0b (amber)

Run:
  python 02-word-embeddings.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

# --------------------------------------------------------------------------- #
# Output configuration
# --------------------------------------------------------------------------- #

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/nlp/"
    "word-embeddings-lm"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/nlp/"
    "02-词向量与语言模型"
)

OUT_DIRS = [EN_DIR, ZH_DIR]
for _d in OUT_DIRS:
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

plt.style.use("seaborn-v0_8-whitegrid")

BLUE   = "#2563eb"
VIOLET = "#7c3aed"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
GREY   = "#6b7280"
DARK   = "#111827"
LIGHT  = "#f3f4f6"

PALETTE = [BLUE, VIOLET, GREEN, AMBER]
DPI = 150

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.edgecolor": "#cbd5e1",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})


def _save(fig, name: str) -> None:
    """Save figure into every output directory."""
    for d in OUT_DIRS:
        out = d / name
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name} -> {len(OUT_DIRS)} dirs")


# --------------------------------------------------------------------------- #
# fig1 — Skip-gram architecture
# --------------------------------------------------------------------------- #

def fig1_skipgram_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Layer x positions
    x_in, x_emb, x_proj, x_out = 1.2, 4.0, 7.2, 10.4
    layer_w = 1.6

    # ---- input one-hot column ----
    n_in = 7
    for i in range(n_in):
        y = 0.7 + i * 0.75
        val = "1" if i == 3 else "0"
        face = AMBER if i == 3 else "white"
        rect = Rectangle((x_in - layer_w / 2, y), layer_w, 0.55,
                         edgecolor=GREY, facecolor=face, linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x_in, y + 0.27, val, ha="center", va="center",
                fontsize=9, color=DARK)
    ax.text(x_in, 6.3, "Input one-hot\n$x \\in \\mathbb{R}^V$",
            ha="center", fontsize=10, color=DARK, fontweight="bold")
    ax.text(x_in, 0.25, "target word: brown", ha="center",
            fontsize=9, color=AMBER, style="italic")

    # ---- embedding layer ----
    rect = FancyBboxPatch((x_emb - layer_w / 2, 1.6), layer_w, 3.2,
                          boxstyle="round,pad=0.05",
                          facecolor=BLUE, edgecolor="none", alpha=0.85)
    ax.add_patch(rect)
    ax.text(x_emb, 3.2, "$W_{V \\times d}$", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")
    ax.text(x_emb, 6.3, "Embedding\nlookup", ha="center",
            fontsize=10, color=DARK, fontweight="bold")
    ax.text(x_emb, 1.2, "row 'brown'", ha="center", fontsize=9,
            color=BLUE, style="italic")

    # ---- projection / hidden vector ----
    n_h = 5
    for i in range(n_h):
        y = 1.5 + i * 0.7
        rect = Rectangle((x_proj - 0.45, y), 0.9, 0.5,
                         edgecolor=BLUE, facecolor="white", linewidth=1.1)
        ax.add_patch(rect)
        vals = [0.21, -0.34, 0.78, 0.05, -0.62]
        ax.text(x_proj, y + 0.25, f"{vals[i]:+.2f}", ha="center",
                va="center", fontsize=8.5, color=DARK)
    ax.text(x_proj, 6.3, "$\\mathbf{v}_w \\in \\mathbb{R}^d$",
            ha="center", fontsize=11, color=DARK, fontweight="bold")
    ax.text(x_proj, 0.9, "dense vector\n(d=100-300)", ha="center",
            fontsize=9, color=GREY, style="italic")

    # ---- output softmax over vocab ----
    n_out = 7
    probs = [0.02, 0.31, 0.04, 0.18, 0.27, 0.03, 0.15]
    for i in range(n_out):
        y = 0.7 + i * 0.75
        w = layer_w * (0.25 + probs[i] * 2.2)
        col = GREEN if i in (1, 3, 4, 6) else GREY
        rect = Rectangle((x_out - layer_w / 2, y), w, 0.55,
                         edgecolor=col, facecolor=col, alpha=0.55,
                         linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x_out - layer_w / 2 - 0.1, y + 0.27,
                f"{probs[i]:.2f}", ha="right", va="center", fontsize=8,
                color=DARK)
    ax.text(x_out, 6.3, "Softmax over $V$\n$P(c \\mid w)$",
            ha="center", fontsize=10, color=DARK, fontweight="bold")
    ax.text(x_out, 0.25, "context targets:\nthe / quick / fox / jumps",
            ha="center", fontsize=8.5, color=GREEN, style="italic")

    # ---- arrows between layers ----
    for src, dst in [(x_in + layer_w / 2, x_emb - layer_w / 2),
                     (x_emb + layer_w / 2, x_proj - 0.45),
                     (x_proj + 0.45, x_out - layer_w / 2)]:
        arr = FancyArrowPatch((src, 3.5), (dst, 3.5),
                              arrowstyle="-|>", mutation_scale=14,
                              color=DARK, linewidth=1.3)
        ax.add_patch(arr)

    # softmax bottleneck note
    ax.annotate("Softmax denominator sums over all $V$ words\n"
                "=> replaced by negative sampling",
                xy=(x_out, 6.1), xytext=(8.6, 6.7),
                fontsize=9, color=VIOLET,
                ha="left", va="bottom", fontweight="bold")

    ax.set_title("Skip-gram: predict context words from a target word",
                 fontsize=13, color=DARK, pad=12)
    _save(fig, "fig1_skipgram_architecture.png")


# --------------------------------------------------------------------------- #
# fig2 — Word analogies in embedding space
# --------------------------------------------------------------------------- #

def fig2_word_analogy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # ---- left panel: gender analogy ----
    ax = axes[0]
    pts = {
        "man":   np.array([1.0, 1.0]),
        "woman": np.array([1.4, 2.6]),
        "king":  np.array([3.8, 1.4]),
        "queen": np.array([4.2, 3.0]),
    }
    colors = {"man": BLUE, "woman": VIOLET, "king": BLUE, "queen": VIOLET}

    for w, p in pts.items():
        ax.scatter(*p, s=180, color=colors[w], zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(w, p, xytext=(8, 8), textcoords="offset points",
                    fontsize=12, color=DARK, fontweight="bold")

    # gender vector arrows (woman - man, queen - king)
    for a, b in [("man", "woman"), ("king", "queen")]:
        ax.annotate("", xy=pts[b], xytext=pts[a],
                    arrowprops=dict(arrowstyle="-|>", color=GREEN,
                                    lw=2.0, mutation_scale=16))
    # gender direction label between the two arrows
    ax.text(2.7, 2.55, "gender direction", color=GREEN, fontsize=10,
            ha="center", fontweight="bold", rotation=14)
    ax.text(2.7, 0.45,
            "king - man + woman ~ queen",
            color=AMBER, fontsize=11, ha="center", fontweight="bold")

    ax.set_xlim(0, 5.5); ax.set_ylim(0, 3.7)
    ax.set_xlabel("dimension 1"); ax.set_ylabel("dimension 2")
    ax.set_title("Gender analogy")

    # ---- right panel: capital analogy ----
    ax = axes[1]
    pts = {
        "France": np.array([1.0, 1.0]),
        "Paris":  np.array([1.6, 2.7]),
        "Italy":  np.array([3.6, 1.3]),
        "Rome":   np.array([4.2, 3.0]),
    }
    colors = {"France": BLUE, "Italy": BLUE, "Paris": VIOLET, "Rome": VIOLET}
    for w, p in pts.items():
        ax.scatter(*p, s=180, color=colors[w], zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(w, p, xytext=(8, 8), textcoords="offset points",
                    fontsize=12, color=DARK, fontweight="bold")

    for a, b in [("France", "Paris"), ("Italy", "Rome")]:
        ax.annotate("", xy=pts[b], xytext=pts[a],
                    arrowprops=dict(arrowstyle="-|>", color=GREEN,
                                    lw=2.0, mutation_scale=16))
    ax.text(2.6, 2.55, "capital-of direction", color=GREEN, fontsize=10,
            ha="center", fontweight="bold", rotation=15)
    ax.text(2.6, 0.45,
            "Italy - France + Paris ~ Rome",
            color=AMBER, fontsize=11, ha="center", fontweight="bold")

    ax.set_xlim(0, 5.5); ax.set_ylim(0, 3.7)
    ax.set_xlabel("dimension 1"); ax.set_ylabel("dimension 2")
    ax.set_title("Capital analogy")

    fig.suptitle("Word embeddings encode relations as constant directions",
                 fontsize=13, color=DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_word_analogy.png")


# --------------------------------------------------------------------------- #
# fig3 — t-SNE clusters
# --------------------------------------------------------------------------- #

def fig3_tsne_clusters() -> None:
    rng = np.random.default_rng(7)

    clusters = {
        "Animals":   (np.array([-3.5,  2.2]), BLUE,
                      ["cat", "dog", "horse", "wolf", "tiger",
                       "rabbit", "kitten"]),
        "Royalty":   (np.array([ 3.0,  2.5]), VIOLET,
                      ["king", "queen", "prince", "princess",
                       "throne", "crown"]),
        "Countries": (np.array([-3.0, -2.4]), GREEN,
                      ["France", "Germany", "Italy", "Spain",
                       "Japan", "Brazil"]),
        "Tech":      (np.array([ 3.2, -2.0]), AMBER,
                      ["computer", "software", "internet",
                       "laptop", "data", "network"]),
    }

    fig, ax = plt.subplots(figsize=(10.5, 7))

    for label, (centre, colour, words) in clusters.items():
        # poisson-disc-ish sampling: jitter on a small grid to avoid overlap
        n = len(words)
        thetas = rng.uniform(0, 2 * np.pi, size=n)
        radii = rng.uniform(0.4, 1.15, size=n)
        offsets = np.stack([radii * np.cos(thetas),
                            radii * np.sin(thetas)], axis=1)
        for w, off in zip(words, offsets):
            p = centre + off
            ax.scatter(*p, s=120, color=colour, alpha=0.85, zorder=3,
                       edgecolor="white", linewidth=1.2)
            ax.annotate(w, p, xytext=(7, 5), textcoords="offset points",
                        fontsize=10, color=DARK)

        # cluster halo
        circ = Circle(centre, 1.85, facecolor=colour, alpha=0.10,
                      edgecolor=colour, linewidth=1.5, linestyle="--")
        ax.add_patch(circ)
        ax.text(centre[0], centre[1] + 2.15, label, ha="center",
                color=colour, fontweight="bold", fontsize=11.5)

    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-5, 5)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.set_title("t-SNE projection of trained embeddings: "
                 "semantic neighbours form tight clusters")
    fig.tight_layout()
    _save(fig, "fig3_tsne_clusters.png")


# --------------------------------------------------------------------------- #
# fig4 — GloVe co-occurrence matrix factorisation
# --------------------------------------------------------------------------- #

def fig4_glove_factorization() -> None:
    rng = np.random.default_rng(11)

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[3, 0.4, 1.6, 0.4, 3])

    # ---- co-occurrence matrix X ----
    ax = fig.add_subplot(gs[0])
    words = ["ice", "steam", "water", "solid", "gas", "fashion",
             "cold", "hot"]
    n = len(words)
    # craft semi-realistic block-ish counts
    base = rng.integers(0, 12, size=(n, n))
    pairs = {
        ("ice", "solid"): 90, ("solid", "ice"): 90,
        ("steam", "gas"): 95, ("gas", "steam"): 95,
        ("ice", "water"): 70, ("water", "ice"): 70,
        ("steam", "water"): 75, ("water", "steam"): 75,
        ("ice", "cold"): 80, ("cold", "ice"): 80,
        ("steam", "hot"): 70, ("hot", "steam"): 70,
        ("water", "cold"): 30, ("water", "hot"): 30,
    }
    X = base.astype(float) + np.eye(n) * 5
    for (a, b), v in pairs.items():
        i, j = words.index(a), words.index(b)
        X[i, j] = v
    im = ax.imshow(np.log1p(X), cmap="Blues", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(words, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(words, fontsize=9)
    ax.set_title("Co-occurrence matrix\n$\\log(1 + X_{ij})$")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # ---- approx symbol ----
    ax = fig.add_subplot(gs[1]); ax.axis("off")
    ax.text(0.5, 0.5, "≈", fontsize=34, ha="center", va="center",
            color=DARK, fontweight="bold")

    # ---- W matrix (V x d) ----
    ax = fig.add_subplot(gs[2])
    W = rng.normal(0, 1, size=(n, 4))
    ax.imshow(W, cmap="PuBu", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"d{i+1}" for i in range(4)],
                                                fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(words, fontsize=9)
    ax.set_title("Word vectors\n$W \\in \\mathbb{R}^{V \\times d}$",
                 color=BLUE)
    ax.grid(False)

    # ---- multiply symbol ----
    ax = fig.add_subplot(gs[3]); ax.axis("off")
    ax.text(0.5, 0.5, "×", fontsize=28, ha="center", va="center",
            color=DARK, fontweight="bold")

    # ---- W tilde transpose (d x V) ----
    ax = fig.add_subplot(gs[4])
    Wt = rng.normal(0, 1, size=(4, n))
    ax.imshow(Wt, cmap="BuPu", aspect="auto")
    ax.set_yticks(range(4)); ax.set_yticklabels([f"d{i+1}" for i in range(4)],
                                                fontsize=9)
    ax.set_xticks(range(n)); ax.set_xticklabels(words, rotation=40,
                                                ha="right", fontsize=9)
    ax.set_title("Context vectors\n$\\tilde{W}^{\\top} \\in "
                 "\\mathbb{R}^{d \\times V}$",
                 color=VIOLET)
    ax.grid(False)

    fig.suptitle("GloVe: factorise the global co-occurrence matrix into "
                 "low-rank word and context embeddings",
                 fontsize=12.5, color=DARK, y=1.04)
    fig.tight_layout()
    _save(fig, "fig4_glove_factorization.png")


# --------------------------------------------------------------------------- #
# fig5 — N-gram vs neural LM perplexity
# --------------------------------------------------------------------------- #

def fig5_lm_perplexity() -> None:
    sizes = np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300])  # million tokens
    # Synthetic but representative perplexity curves
    ngram   = 320 * (sizes ** -0.18) + 60
    neural  = 260 * (sizes ** -0.30) + 35
    transf  = 220 * (sizes ** -0.42) + 18

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(sizes, ngram, "o-", color=AMBER, lw=2.2, ms=8,
            label="N-gram + smoothing")
    ax.plot(sizes, neural, "s-", color=BLUE, lw=2.2, ms=8,
            label="Feed-forward neural LM")
    ax.plot(sizes, transf, "^-", color=VIOLET, lw=2.2, ms=8,
            label="Transformer LM")

    ax.set_xscale("log")
    ax.set_xlabel("Training corpus size (million tokens)")
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("Why neural LMs win: perplexity vs. corpus size")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", alpha=0.35)

    # annotation
    ax.annotate("N-gram plateaus:\ncannot share strength\nacross similar words",
                xy=(100, ngram[-2]), xytext=(0.13, 280),
                fontsize=9.5, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2))
    ax.annotate("Neural LMs keep\nimproving with scale\n(via embeddings)",
                xy=(100, transf[-2]), xytext=(0.5, 24),
                fontsize=9.5, color=VIOLET, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.2))

    fig.tight_layout()
    _save(fig, "fig5_lm_perplexity.png")


# --------------------------------------------------------------------------- #
# fig6 — Negative sampling
# --------------------------------------------------------------------------- #

def fig6_negative_sampling() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-4.8, 4.5)
    ax.set_aspect("equal")

    # anchor (target word)
    anchor = np.array([0.0, 0.0])
    ax.scatter(*anchor, s=320, color=AMBER, zorder=5,
               edgecolor="white", linewidth=2)
    ax.annotate("brown\n(target)", anchor, xytext=(-12, -28),
                textcoords="offset points",
                fontsize=11, color=AMBER, fontweight="bold")

    # positive (true context)
    pos = np.array([1.7, 1.1])
    ax.scatter(*pos, s=240, color=GREEN, zorder=5,
               edgecolor="white", linewidth=2)
    ax.annotate("fox\n(positive)", pos, xytext=(12, 4),
                textcoords="offset points",
                fontsize=11, color=GREEN, fontweight="bold")
    ax.annotate("", xy=pos * 0.45, xytext=pos * 0.95,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4,
                                mutation_scale=18))
    ax.text(0.55, 0.95, "pull together", color=GREEN, fontsize=9.5,
            rotation=32, fontweight="bold")

    # negatives
    rng = np.random.default_rng(3)
    neg_words = ["banana", "philosophy", "router", "tide", "ledger"]
    angles = np.linspace(np.pi * 0.65, np.pi * 1.85, len(neg_words))
    radii = rng.uniform(2.7, 3.4, size=len(neg_words))
    for w, a, r in zip(neg_words, angles, radii):
        p = np.array([np.cos(a), np.sin(a)]) * r
        ax.scatter(*p, s=200, color=VIOLET, zorder=5, alpha=0.9,
                   edgecolor="white", linewidth=2)
        # label outside the dot
        label_offset = (np.array([np.cos(a), np.sin(a)]) * 0.55)
        ax.annotate(w, p + label_offset, ha="center",
                    fontsize=10, color=VIOLET, fontweight="bold")
        # push arrow (anchor -> further out)
        far = p * 1.3
        ax.annotate("", xy=far, xytext=p * 1.05,
                    arrowprops=dict(arrowstyle="-|>", color=VIOLET,
                                    lw=1.8, mutation_scale=14,
                                    linestyle=(0, (3, 2))))

    # ring guide
    ring = Circle((0, 0), 3.0, fill=False, color=GREY,
                  linestyle=":", linewidth=1)
    ax.add_patch(ring)

    ax.text(0, -4.4,
            "Loss per (target, context) pair:\n"
            "$\\log\\sigma(\\mathbf{v}_w^{\\top}\\mathbf{v}'_c) "
            "+ \\sum_{i=1}^{k}\\log\\sigma(-\\mathbf{v}_w^{\\top}\\mathbf{v}'_{n_i})$",
            fontsize=10.5, color=DARK, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT,
                      edgecolor=GREY, linewidth=0.8))

    handles = [
        mpatches.Patch(color=AMBER, label="Target word"),
        mpatches.Patch(color=GREEN, label="True context (positive)"),
        mpatches.Patch(color=VIOLET, label="Sampled negatives"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9.5,
              bbox_to_anchor=(1.0, 1.0))

    ax.axis("off")
    ax.set_title("Negative sampling: 1 positive vs. k random negatives "
                 "replaces the full softmax",
                 fontsize=12.5, color=DARK, pad=12)
    fig.tight_layout()
    _save(fig, "fig6_negative_sampling.png")


# --------------------------------------------------------------------------- #
# fig7 — Subword tokenisation (FastText)
# --------------------------------------------------------------------------- #

def fig7_subword_fasttext() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.axis("off")

    # word
    ax.text(6, 6.3, "word: <where>", ha="center", fontsize=14,
            color=DARK, fontweight="bold")

    # n-grams row
    ngrams = ["<wh", "whe", "her", "ere", "re>",
              "<whe", "wher", "here", "ere>", "<where>"]
    n = len(ngrams)
    x_positions = np.linspace(0.7, 11.3, n)
    for x, g in zip(x_positions, ngrams):
        col = AMBER if g == "<where>" else BLUE
        rect = FancyBboxPatch((x - 0.55, 4.4), 1.1, 0.7,
                              boxstyle="round,pad=0.04",
                              facecolor=col, edgecolor="none", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, 4.75, g, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.text(6, 5.4, "character n-grams (n = 3..6) + full word",
            ha="center", fontsize=10, color=GREY, style="italic")

    # plus signs and arrows toward sum vector
    sum_x, sum_y = 6.0, 1.5
    for x in x_positions:
        ax.annotate("", xy=(sum_x, sum_y + 0.6), xytext=(x, 4.35),
                    arrowprops=dict(arrowstyle="-", color=GREY,
                                    lw=0.7, alpha=0.55))

    # sum block
    rect = FancyBboxPatch((sum_x - 1.6, sum_y - 0.4), 3.2, 1.0,
                          boxstyle="round,pad=0.05",
                          facecolor=GREEN, edgecolor="none", alpha=0.9)
    ax.add_patch(rect)
    ax.text(sum_x, sum_y + 0.1,
            "$\\mathbf{v}_{\\text{where}} = "
            "\\sum_{g \\in G(w)} \\mathbf{z}_g$",
            ha="center", va="center", fontsize=12.5, color="white",
            fontweight="bold")

    # OOV side note
    ax.text(0.4, 0.4,
            "OOV power: an unseen word like 'wherever' shares\n"
            "n-grams (whe, here, ere…) and still gets a useful vector.",
            fontsize=10, color=VIOLET, fontweight="bold")

    ax.set_title("FastText: word embedding = sum of subword n-gram embeddings",
                 fontsize=13, color=DARK, pad=10)
    _save(fig, "fig7_subword_fasttext.png")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    print("Generating NLP-02 figures...")
    fig1_skipgram_architecture()
    fig2_word_analogy()
    fig3_tsne_clusters()
    fig4_glove_factorization()
    fig5_lm_perplexity()
    fig6_negative_sampling()
    fig7_subword_fasttext()
    print("Done.")


if __name__ == "__main__":
    main()
