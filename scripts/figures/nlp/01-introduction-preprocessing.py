"""
Figures for NLP Part 01: Introduction and Text Preprocessing.

Generates 7 publication-quality figures and saves them simultaneously into the
English and Chinese asset folders so both posts share identical visuals.

Run:
    python 01-introduction-preprocessing.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Style and output configuration
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#94a3b8",
    "grid.color": "#e2e8f0",
})

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
SLATE = "#475569"
LIGHT = "#f1f5f9"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/nlp/introduction-and-preprocessing"
ZH_DIR = REPO / "source/_posts/zh/nlp/01-NLP入门与文本预处理"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure into both EN and ZH asset folders."""
    for folder in (EN_DIR, ZH_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png")


# ---------------------------------------------------------------------------
# Figure 1 - NLP applications landscape
# ---------------------------------------------------------------------------
def fig_applications_landscape() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6, 7.45, "NLP Application Landscape",
            fontsize=17, fontweight="bold", ha="center", color="#1e293b")
    ax.text(6, 7.0, "Where natural language processing powers everyday products",
            fontsize=10.5, ha="center", color=SLATE, style="italic")

    categories = [
        ("Conversational AI", BLUE,
         ["ChatGPT", "Claude", "Voice assistants", "Customer-service bots"]),
        ("Search and Retrieval", PURPLE,
         ["Google search", "Semantic search", "RAG systems", "Code search"]),
        ("Translation", GREEN,
         ["Google Translate", "DeepL", "Subtitle generation", "Cross-lingual QA"]),
        ("Content Understanding", ORANGE,
         ["Sentiment analysis", "Spam filters", "Topic modeling", "Toxicity detection"]),
        ("Text Generation", "#ec4899",
         ["Summarization", "Code completion", "Marketing copy", "Story writing"]),
        ("Information Extraction", "#06b6d4",
         ["Named entities", "Relation extraction", "Knowledge graphs", "Resume parsing"]),
    ]

    positions = [(0.35, 3.6), (4.15, 3.6), (7.95, 3.6),
                 (0.35, 0.3), (4.15, 0.3), (7.95, 0.3)]
    box_w, box_h = 3.6, 2.9

    for (title, color, items), (x, y) in zip(categories, positions):
        card = FancyBboxPatch((x, y), box_w, box_h,
                              boxstyle="round,pad=0.05,rounding_size=0.18",
                              linewidth=1.6, edgecolor=color,
                              facecolor=color, alpha=0.08)
        ax.add_patch(card)
        header = FancyBboxPatch((x, y + box_h - 0.55), box_w, 0.55,
                                boxstyle="round,pad=0.0,rounding_size=0.18",
                                linewidth=0, facecolor=color, alpha=0.95)
        ax.add_patch(header)
        ax.text(x + box_w / 2, y + box_h - 0.275, title,
                fontsize=11.5, fontweight="bold", ha="center", va="center",
                color="white")
        for i, item in enumerate(items):
            ax.text(x + 0.18, y + box_h - 0.95 - i * 0.45,
                    f"•  {item}", fontsize=9.8, color="#1e293b", va="center")

    plt.tight_layout()
    save(fig, "fig1_applications_landscape")


# ---------------------------------------------------------------------------
# Figure 2 - Text preprocessing pipeline
# ---------------------------------------------------------------------------
def fig_preprocessing_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(6.5, 4.85, "Text Preprocessing Pipeline",
            fontsize=16, fontweight="bold", ha="center", color="#1e293b")
    ax.text(6.5, 4.45, "From raw text to numerical features ready for a model",
            fontsize=10.5, ha="center", color=SLATE, style="italic")

    stages = [
        ("Raw\nText", "<p>Hello\nworld!!!</p>", "#94a3b8"),
        ("Cleaning", "Hello world", BLUE),
        ("Tokenize", "[Hello, world]", PURPLE),
        ("Normalize", "[hello, world]", GREEN),
        ("Stopwords", "[hello, world]", ORANGE),
        ("Vectorize", "[0.3, 0.8, ...]", "#ec4899"),
        ("Model\nReady", "tensor", "#06b6d4"),
    ]

    n = len(stages)
    box_w = 1.45
    gap = (13 - n * box_w) / (n + 1)
    y_box = 1.9
    y_h = 1.55

    centers = []
    for i, (title, payload, color) in enumerate(stages):
        x = gap + i * (box_w + gap)
        centers.append(x + box_w / 2)
        card = FancyBboxPatch((x, y_box), box_w, y_h,
                              boxstyle="round,pad=0.04,rounding_size=0.16",
                              linewidth=1.8, edgecolor=color,
                              facecolor=color, alpha=0.12)
        ax.add_patch(card)
        ax.text(x + box_w / 2, y_box + y_h - 0.32, title,
                fontsize=10.5, fontweight="bold", ha="center", va="center",
                color=color)
        ax.text(x + box_w / 2, y_box + 0.42, payload,
                fontsize=8.5, ha="center", va="center", color="#1e293b",
                family="monospace")

    for i in range(n - 1):
        arrow = FancyArrowPatch((centers[i] + box_w / 2 - 0.1, y_box + y_h / 2),
                                (centers[i + 1] - box_w / 2 + 0.1, y_box + y_h / 2),
                                arrowstyle="-|>", mutation_scale=14,
                                color=SLATE, linewidth=1.6)
        ax.add_patch(arrow)

    notes = [
        ("Strip noise", 1),
        ("Split units", 2),
        ("Lemma / lower", 3),
        ("Drop \"the\", \"is\"", 4),
        ("BoW / TF-IDF\n/ embeddings", 5),
    ]
    for label, idx in notes:
        ax.text(centers[idx], y_box - 0.45, label,
                fontsize=8.5, ha="center", color=SLATE, style="italic")

    plt.tight_layout()
    save(fig, "fig2_preprocessing_pipeline")


# ---------------------------------------------------------------------------
# Figure 3 - BoW vs TF-IDF comparison
# ---------------------------------------------------------------------------
def fig_bow_vs_tfidf() -> None:
    docs_label = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
    vocab = ["machine", "learning", "deep", "vision", "language", "is"]

    bow = np.array([
        [1, 1, 0, 0, 0, 1],
        [1, 2, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
    ], dtype=float)

    n_docs = bow.shape[0]
    df = (bow > 0).sum(axis=0)
    idf = np.log((1 + n_docs) / (1 + df)) + 1
    tf = bow / np.maximum(bow.sum(axis=1, keepdims=True), 1)
    tfidf = tf * idf
    tfidf = tfidf / np.maximum(np.linalg.norm(tfidf, axis=1, keepdims=True), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Bag of Words vs TF-IDF: weighting word importance",
                 fontsize=15, fontweight="bold", y=1.02)

    def heatmap(ax, data, title, cmap, fmt):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(vocab)))
        ax.set_xticklabels(vocab, rotation=30, ha="right")
        ax.set_yticks(range(len(docs_label)))
        ax.set_yticklabels(docs_label)
        ax.set_title(title, fontsize=12)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                rel = value / max(data.max(), 1e-9)
                color = "white" if rel > 0.55 else "#1e293b"
                ax.text(j, i, fmt.format(value), ha="center", va="center",
                        color=color, fontsize=9.5)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.grid(False)

    heatmap(axes[0], bow, "Bag of Words (raw counts)", "Blues", "{:.0f}")
    heatmap(axes[1], tfidf, "TF-IDF (normalized weights)", "Purples", "{:.2f}")

    note = ("\"learning\" appears in every doc → TF-IDF down-weights it.\n"
            "\"vision\" appears only in Doc 4 → TF-IDF lifts it as a discriminator.")
    fig.text(0.5, -0.04, note, ha="center", fontsize=10,
             color=SLATE, style="italic")

    plt.tight_layout()
    save(fig, "fig3_bow_vs_tfidf")


# ---------------------------------------------------------------------------
# Figure 4 - Tokenization variants
# ---------------------------------------------------------------------------
def fig_tokenization_variants() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 6.55, "Three Ways to Tokenize \"unbelievable transformers\"",
            fontsize=15.5, fontweight="bold", ha="center", color="#1e293b")
    ax.text(6, 6.15, "Granularity changes vocabulary size and ability to handle unseen words",
            fontsize=10.3, ha="center", color=SLATE, style="italic")

    rows = [
        ("Character",
         list("unbelievable transformers"),
         BLUE,
         "Tiny vocab (~100). Loses morphology. Sequences grow long."),
        ("Word",
         ["unbelievable", "transformers"],
         GREEN,
         "Compact for known words. Fails on rare words → <UNK>."),
        ("Subword (BPE)",
         ["un", "believ", "able", "transform", "ers"],
         PURPLE,
         "Sweet spot. Splits rare words into reusable pieces. Used by GPT, BERT, Llama."),
    ]

    y_positions = [4.4, 2.7, 1.0]
    for (label, tokens, color, note), y in zip(rows, y_positions):
        ax.text(0.3, y + 0.35, label, fontsize=12.2, fontweight="bold", color=color)
        ax.text(0.3, y - 0.15, note, fontsize=9.5, color=SLATE, style="italic")

        token_widths = [max(0.55, len(t) * 0.18 + 0.35) for t in tokens]
        total = sum(token_widths) + 0.18 * (len(tokens) - 1)
        x_start = 11.6 - total
        x = x_start
        for tok, w in zip(tokens, token_widths):
            box = FancyBboxPatch((x, y - 0.05), w, 0.7,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 linewidth=1.4, edgecolor=color,
                                 facecolor=color, alpha=0.18)
            ax.add_patch(box)
            ax.text(x + w / 2, y + 0.3, tok, ha="center", va="center",
                    fontsize=10, family="monospace", color="#1e293b")
            x += w + 0.18

    # Vocab-size axis at bottom
    ax.annotate("", xy=(11.4, 0.25), xytext=(0.3, 0.25),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.4))
    ax.text(0.3, 0.02, "smaller vocab ← character", fontsize=9, color=SLATE)
    ax.text(11.4, 0.02, "larger vocab → word", fontsize=9, color=SLATE,
            ha="right")
    ax.text(6, 0.02, "subword balances both", fontsize=9, color=PURPLE,
            ha="center", fontweight="bold")

    plt.tight_layout()
    save(fig, "fig4_tokenization_variants")


# ---------------------------------------------------------------------------
# Figure 5 - Zipf distribution of word frequency
# ---------------------------------------------------------------------------
def fig_zipf_distribution() -> None:
    rng = np.random.default_rng(42)
    ranks = np.arange(1, 5001)
    # Zipfian frequency with a small noise term
    base = 1e6 / (ranks ** 1.07)
    noise = rng.normal(loc=0, scale=base * 0.05)
    freq = np.clip(base + noise, 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Zipf's Law: a few words dominate, most appear rarely",
                 fontsize=15, fontweight="bold", y=1.02)

    # Linear plot - top 50 words
    ax1 = axes[0]
    top_n = 50
    bar_colors = [BLUE if r > 10 else ORANGE for r in ranks[:top_n]]
    ax1.bar(ranks[:top_n], freq[:top_n], color=bar_colors, edgecolor="white")
    ax1.set_xlabel("Word rank")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Top 50 words: head of the distribution")
    legend_handles = [
        mpatches.Patch(color=ORANGE, label="Top 10 (often stopwords)"),
        mpatches.Patch(color=BLUE, label="Rank 11-50"),
    ]
    ax1.legend(handles=legend_handles, loc="upper right", frameon=True)

    # Log-log plot
    ax2 = axes[1]
    ax2.loglog(ranks, freq, color=PURPLE, linewidth=2, label="Empirical")
    ax2.loglog(ranks, 1e6 / ranks, color=GREEN, linewidth=1.6,
               linestyle="--", label="$f \\propto 1/r$ (Zipf)")
    ax2.set_xlabel("Word rank (log)")
    ax2.set_ylabel("Frequency (log)")
    ax2.set_title("Log-log: nearly straight line confirms power law")
    ax2.legend(loc="upper right", frameon=True)
    ax2.axvspan(1, 100, alpha=0.08, color=ORANGE)
    ax2.text(8, freq[0] * 0.6, "stopword zone",
             color=ORANGE, fontsize=9.5, fontweight="bold")
    ax2.axvspan(2000, 5000, alpha=0.08, color=GREEN)
    ax2.text(2300, freq[2000] * 4, "long tail\n(rare words)",
             color=GREEN, fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    save(fig, "fig5_zipf_distribution")


# ---------------------------------------------------------------------------
# Figure 6 - One-hot vs distributed representation
# ---------------------------------------------------------------------------
def fig_onehot_vs_distributed() -> None:
    words = ["king", "queen", "man", "woman", "apple"]
    one_hot = np.eye(len(words))

    # Hand-crafted distributed embeddings to make the structure obvious
    distributed = np.array([
        # royalty, gender(female), age, fruit
        [0.92, 0.05, 0.60, 0.05],   # king
        [0.90, 0.95, 0.55, 0.05],   # queen
        [0.10, 0.05, 0.70, 0.05],   # man
        [0.10, 0.95, 0.65, 0.05],   # woman
        [0.05, 0.50, 0.20, 0.95],   # apple
    ])
    dims = ["royalty", "female", "adult", "fruit"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("One-hot vs Distributed Representation",
                 fontsize=15, fontweight="bold", y=1.02)

    ax1 = axes[0]
    im1 = ax1.imshow(one_hot, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks(range(len(words)))
    ax1.set_xticklabels(words)
    ax1.set_yticks(range(len(words)))
    ax1.set_yticklabels(words)
    ax1.set_title("One-hot: sparse, every pair is orthogonal")
    ax1.set_xlabel("Vocabulary index")
    ax1.set_ylabel("Word")
    for i in range(len(words)):
        for j in range(len(words)):
            ax1.text(j, i, int(one_hot[i, j]), ha="center", va="center",
                     color="white" if one_hot[i, j] > 0.5 else "#1e293b",
                     fontsize=10)
    ax1.grid(False)
    ax1.text(2, 5.6, "cosine(king, queen) = 0  →  no semantic signal",
             ha="center", fontsize=9.8, color=SLATE, style="italic")

    ax2 = axes[1]
    im2 = ax2.imshow(distributed, cmap="Purples", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(dims)))
    ax2.set_xticklabels(dims)
    ax2.set_yticks(range(len(words)))
    ax2.set_yticklabels(words)
    ax2.set_title("Distributed: dense vector encodes meaning")
    ax2.set_xlabel("Latent dimension")
    for i in range(len(words)):
        for j in range(len(dims)):
            ax2.text(j, i, f"{distributed[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if distributed[i, j] > 0.55 else "#1e293b",
                     fontsize=9.5)
    ax2.grid(False)
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    ax2.text(1.5, 5.6, "king - man + woman ≈ queen",
             ha="center", fontsize=9.8, color=PURPLE,
             fontweight="bold", style="italic")

    plt.tight_layout()
    save(fig, "fig6_onehot_vs_distributed")


# ---------------------------------------------------------------------------
# Figure 7 - N-gram language models
# ---------------------------------------------------------------------------
def fig_ngram_language_models() -> None:
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0],
                          hspace=0.55, wspace=0.32)

    fig.suptitle("N-gram Language Models",
                 fontsize=15.5, fontweight="bold", y=1.0)

    # Top: sentence with sliding window
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_xlim(0, 13)
    ax_top.set_ylim(0, 4.2)
    ax_top.axis("off")
    ax_top.set_title("Sliding window over: \"the cat sat on the mat\"",
                     fontsize=12, color="#1e293b")

    sentence = ["the", "cat", "sat", "on", "the", "mat"]
    word_w = 1.5
    x_start = 1.5
    for i, w in enumerate(sentence):
        x = x_start + i * (word_w + 0.15)
        box = FancyBboxPatch((x, 2.7), word_w, 0.9,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.5, edgecolor=SLATE,
                             facecolor=LIGHT)
        ax_top.add_patch(box)
        ax_top.text(x + word_w / 2, 3.15, w, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#1e293b")

    # Highlight bigrams and trigrams
    def highlight(start_idx, n, y, color, label):
        x = x_start + start_idx * (word_w + 0.15) - 0.05
        width = n * word_w + (n - 1) * 0.15 + 0.1
        box = FancyBboxPatch((x, y), width, 0.85,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=2.0, edgecolor=color,
                             facecolor=color, alpha=0.18)
        ax_top.add_patch(box)
        ax_top.text(x + width / 2, y + 0.42, label, ha="center", va="center",
                    fontsize=9.5, color=color, fontweight="bold")

    highlight(0, 2, 1.55, BLUE, "bigram: (the, cat)")
    highlight(3, 3, 0.45, PURPLE, "trigram: (on, the, mat)")

    # Bottom-left: bigram probability formula and counts
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.axis("off")
    ax_bl.set_title("Bigram model: predict next word from one previous", fontsize=11.5)
    ax_bl.text(0.04, 0.78,
               r"$P(w_t \mid w_{t-1}) = \dfrac{\mathrm{count}(w_{t-1}, w_t)}{\mathrm{count}(w_{t-1})}$",
               fontsize=14, color=BLUE)
    ax_bl.text(0.04, 0.42,
               "Example: count(the, cat) = 240, count(the) = 1200",
               fontsize=10.5, color="#1e293b")
    ax_bl.text(0.04, 0.22,
               r"$P(\mathrm{cat} \mid \mathrm{the}) = 240 / 1200 = 0.20$",
               fontsize=12, color=GREEN, fontweight="bold")

    # Bottom-right: bar chart - perplexity vs n
    ax_br = fig.add_subplot(gs[1, 1])
    ns = np.array([1, 2, 3, 4, 5])
    perplexity = np.array([950, 180, 110, 95, 92])
    sparsity_risk = np.array([0.05, 0.18, 0.42, 0.68, 0.88])

    bars = ax_br.bar(ns, perplexity, color=BLUE, alpha=0.85,
                     label="Perplexity (lower = better fit)")
    ax_br.set_xlabel("n (context size)")
    ax_br.set_ylabel("Perplexity", color=BLUE)
    ax_br.tick_params(axis="y", labelcolor=BLUE)
    ax_br.set_xticks(ns)
    ax_br.set_title("Bigger n: better fit, worse sparsity", fontsize=11.5)
    for bar, value in zip(bars, perplexity):
        ax_br.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 18, str(value),
                   ha="center", fontsize=9, color=BLUE)

    ax_br2 = ax_br.twinx()
    ax_br2.plot(ns, sparsity_risk, color=ORANGE, marker="o", linewidth=2,
                label="Sparsity risk")
    ax_br2.set_ylabel("Unseen-context risk", color=ORANGE)
    ax_br2.tick_params(axis="y", labelcolor=ORANGE)
    ax_br2.set_ylim(0, 1)
    ax_br2.grid(False)

    plt.tight_layout()
    save(fig, "fig7_ngram_language_models")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Output dirs:\n  EN: {EN_DIR}\n  ZH: {ZH_DIR}")
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    fig_applications_landscape()
    fig_preprocessing_pipeline()
    fig_bow_vs_tfidf()
    fig_tokenization_variants()
    fig_zipf_distribution()
    fig_onehot_vs_distributed()
    fig_ngram_language_models()
    print("Done.")


if __name__ == "__main__":
    main()
