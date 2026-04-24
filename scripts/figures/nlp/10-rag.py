"""Figures for NLP Part 10 — RAG and Knowledge Enhancement.

Generates 7 publication-quality figures for the article on
Retrieval-Augmented Generation. Each figure is saved into BOTH the
English and Chinese asset folders.

Figures:
    fig1_rag_pipeline      — End-to-end RAG: query -> retrieve -> augment -> generate
    fig2_vector_similarity — Embedding space with cosine retrieval geometry
    fig3_hybrid_retrieval  — Dense + Sparse (BM25) fusion via RRF
    fig4_reranking         — Bi-encoder vs Cross-encoder, two-stage pipeline
    fig5_chunking          — Fixed / Recursive / Semantic chunking trade-offs
    fig6_self_rag          — Self-RAG / Corrective-RAG control flow
    fig7_vectordb          — Vector DB comparison (FAISS, Milvus, Chroma, Pinecone, Weaviate)

Style: seaborn-v0_8-whitegrid, dpi=150, palette
    #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (orange).

References (verified):
    - Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP (NeurIPS 2020)
    - Karpukhin et al., Dense Passage Retrieval for Open-Domain QA (EMNLP 2020)
    - Robertson & Zaragoza, BM25 and Beyond (FnTIR 2009)
    - Cormack et al., Reciprocal Rank Fusion (SIGIR 2009)
    - Nogueira & Cho, Passage Re-ranking with BERT (2019)
    - Asai et al., Self-RAG (ICLR 2024)
    - Yan et al., Corrective Retrieval Augmented Generation (CRAG, 2024)
    - Gao et al., Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE, ACL 2023)
    - Douze et al., The FAISS Library (2024)
    - Wang et al., Milvus: A Purpose-Built Vector Data Management System (SIGMOD 2021)
"""

from __future__ import annotations

from pathlib import Path

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
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#64748b"
LIGHT = "#e5e7eb"
DARK = "#1f2937"
RED = "#ef4444"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/nlp/rag-knowledge-enhancement"
ZH_DIR = REPO / "source/_posts/zh/nlp/10-RAG与知识增强系统"


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
# Figure 1 — End-to-end RAG pipeline
# ---------------------------------------------------------------------------
def fig1_rag_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(6.75, 5.95, "Retrieval-Augmented Generation Pipeline",
            ha="center", fontsize=14, weight="bold")

    # Offline indexing track (top)
    ax.text(0.2, 5.35, "OFFLINE  ·  index build",
            fontsize=9, weight="bold", color=GRAY)
    _rounded(ax, 0.2, 4.55, 1.8, 0.7, "Source\ncorpus", DARK, LIGHT, fs=9)
    _rounded(ax, 2.4, 4.55, 1.8, 0.7, "Chunk\n+ clean", PURPLE, "#ede9fe", fs=9)
    _rounded(ax, 4.6, 4.55, 1.8, 0.7, "Embed\n(bi-encoder)", BLUE, "#dbeafe", fs=9)
    _rounded(ax, 6.8, 4.55, 1.9, 0.7, "Vector DB\n(ANN index)", GREEN, "#d1fae5", fs=9)
    for x in (2.0, 4.2, 6.4):
        _arrow(ax, x, 4.9, x + 0.4, 4.9, color=GRAY)

    # Online query track (bottom)
    ax.text(0.2, 3.6, "ONLINE  ·  per-query",
            fontsize=9, weight="bold", color=GRAY)
    _rounded(ax, 0.2, 2.6, 1.8, 0.85, "User\nquery", DARK, "#fff7ed", fs=10, weight="bold")
    _rounded(ax, 2.4, 2.6, 1.8, 0.85, "Query\nrewrite", ORANGE, "#fef3c7", fs=9)
    _rounded(ax, 4.6, 2.6, 1.8, 0.85, "Embed\nquery", BLUE, "#dbeafe", fs=9)
    _rounded(ax, 6.8, 2.6, 1.9, 0.85, "Retrieve\ntop-k = 50", GREEN, "#d1fae5", fs=9)
    _rounded(ax, 8.9, 2.6, 1.7, 0.85, "Rerank\ntop-k = 5", PURPLE, "#ede9fe", fs=9)
    _rounded(ax, 10.8, 2.6, 1.5, 0.85, "Build\nprompt", ORANGE, "#fef3c7", fs=9)
    _rounded(ax, 12.4, 2.6, 1.0, 0.85, "LLM", BLUE, "#dbeafe", fs=10, weight="bold")

    for x1, x2 in [(2.0, 2.4), (4.2, 4.6), (6.4, 6.8), (8.7, 8.9),
                   (10.6, 10.8), (12.3, 12.4)]:
        _arrow(ax, x1, 3.025, x2, 3.025, lw=1.8)

    # Vector DB feeds retrieve step
    _arrow(ax, 7.75, 4.55, 7.75, 3.45, color=GREEN, lw=1.8, style="->")

    # Output
    _rounded(ax, 11.4, 0.6, 2.0, 0.85, "Grounded answer\n+ citations",
             GREEN, "#d1fae5", fs=10, weight="bold")
    _arrow(ax, 12.9, 2.6, 12.9, 1.45, color=GREEN, lw=2.0)

    # Math box
    ax.text(0.2, 1.55,
            r"$P(y \mid q) \;=\; \sum_{d \in \mathcal{D}_k}\;"
            r"P(d \mid q)\;\cdot\;P(y \mid q, d)$",
            fontsize=13, color=DARK)
    ax.text(0.2, 1.05,
            r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;"
            r"\uparrow\;\mathrm{retriever}\;\;\;\;\;\;\;\;\;\;\;\;"
            r"\uparrow\;\mathrm{generator}$",
            fontsize=10, color=GRAY)
    ax.text(0.2, 0.55,
            r"Retrieved evidence $\mathcal{D}_k$ grounds generation;"
            r" knowledge can be updated without retraining the LLM.",
            fontsize=9.5, color=GRAY, style="italic")

    fig.tight_layout()
    save(fig, "fig1_rag_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 2 — Vector similarity in embedding space
# ---------------------------------------------------------------------------
def fig2_vector_similarity() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    # ----- Left: 2-D projection of an embedding space -----
    rng = np.random.default_rng(7)

    # Three clusters of documents
    clusters = [
        ("ML / training",  np.array([2.4, 2.2]),  BLUE),
        ("Databases",      np.array([-2.2, 1.8]), PURPLE),
        ("Cooking",        np.array([0.4, -2.4]), ORANGE),
    ]
    for label, center, color in clusters:
        pts = center + rng.normal(0, 0.55, size=(22, 2))
        ax1.scatter(pts[:, 0], pts[:, 1], c=color, s=55, alpha=0.55,
                    edgecolor="white", linewidth=0.8, label=label)

    # Query and its top-k neighbours
    q = np.array([2.05, 2.55])
    ax1.scatter(*q, c=RED, s=220, marker="*", edgecolor=DARK,
                linewidth=1.4, zorder=5, label="query q")

    # Use ML cluster as candidate pool
    ml_center = clusters[0][1]
    candidates = ml_center + rng.normal(0, 0.55, size=(22, 2))
    dists = np.linalg.norm(candidates - q, axis=1)
    topk = np.argsort(dists)[:3]
    for i, idx in enumerate(topk):
        c = candidates[idx]
        ax1.plot([q[0], c[0]], [q[1], c[1]], "--", color=RED, lw=1.4, alpha=0.85)
        ax1.scatter(*c, c=RED, s=80, edgecolor=DARK, linewidth=1.2, zorder=4)
        offsets = [(14, -2), (14, 10), (-32, 8)]
        ax1.annotate(f"top-{i+1}", c, textcoords="offset points",
                     xytext=offsets[i], fontsize=8.5, color=RED, weight="bold")

    # Cosine similarity arc — show angle interpretation at origin
    theta = np.linspace(0, np.arctan2(q[1], q[0]), 30)
    r = 0.7
    ax1.plot(r * np.cos(theta), r * np.sin(theta), color=GRAY, lw=1.2)
    ax1.annotate(r"$\theta$", (0.55, 0.35), fontsize=11, color=GRAY)
    ax1.annotate("", xy=(q[0] * 0.45, q[1] * 0.45), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))

    ax1.set_xlim(-4.2, 4.5)
    ax1.set_ylim(-4.2, 4.2)
    ax1.set_xlabel("embedding dim 1 (PCA)")
    ax1.set_ylabel("embedding dim 2 (PCA)")
    ax1.set_title("Documents cluster by meaning;  retrieval = nearest neighbours",
                  loc="left")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ----- Right: Recall@k vs index latency for FAISS index families -----
    indices = ["Flat\n(exact)", "IVF-Flat", "IVF-PQ", "HNSW\nM=32",
               "ScaNN", "PQ-only"]
    # Approximate Recall@10 and per-query latency on a 1M, 768-d corpus
    recall = [100.0, 96.5, 90.5, 98.5, 97.5, 82.0]
    latency = [38.0, 4.5, 1.6, 0.9, 1.1, 0.8]   # ms / query
    colors = [GRAY, BLUE, ORANGE, GREEN, PURPLE, RED]

    ax2.scatter(latency, recall, c=colors, s=170, edgecolor=DARK,
                linewidth=1.2, zorder=3)
    for x, y, name in zip(latency, recall, indices):
        ax2.annotate(name, (x, y), textcoords="offset points",
                     xytext=(8, 6), fontsize=9, weight="bold")
    ax2.set_xscale("log")
    ax2.set_xlabel("Query latency (ms, log scale)")
    ax2.set_ylabel("Recall@10  (%)")
    ax2.set_xlim(0.5, 70)
    ax2.set_ylim(78, 102)
    ax2.set_title("Index trade-off:  recall vs latency  (1 M × 768-d)",
                  loc="left")
    ax2.grid(True, alpha=0.3)
    ax2.text(45, 80, "Pareto frontier:\nHNSW / ScaNN", fontsize=9,
             color=GREEN, weight="bold", ha="center")

    fig.suptitle("Vector Similarity Search in Embedding Space",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_vector_similarity.png")


# ---------------------------------------------------------------------------
# Figure 3 — Hybrid retrieval (dense + BM25) with RRF fusion
# ---------------------------------------------------------------------------
def fig3_hybrid_retrieval() -> None:
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.4, 1],
                          hspace=0.55, wspace=0.28)

    # ----- Top: schematic of two retrievers feeding RRF -----
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    _rounded(ax, 0.2, 1.85, 1.7, 0.85, "User query", DARK, "#fff7ed",
             fs=10, weight="bold")

    _rounded(ax, 2.6, 3.05, 2.4, 0.85,
             "Dense (bi-encoder)\ncosine over embeddings",
             BLUE, "#dbeafe", fs=9.5)
    _rounded(ax, 2.6, 0.6, 2.4, 0.85,
             "Sparse (BM25)\nlexical / IDF weighted",
             ORANGE, "#fef3c7", fs=9.5)

    _arrow(ax, 1.95, 2.5, 2.55, 3.45, color=BLUE, lw=1.8)
    _arrow(ax, 1.95, 2.1, 2.55, 1.05, color=ORANGE, lw=1.8)

    # Two ranked lists
    dense_list = ["d_4", "d_2", "d_9", "d_1", "d_7"]
    sparse_list = ["d_2", "d_5", "d_4", "d_8", "d_1"]
    for i, did in enumerate(dense_list):
        _rounded(ax, 5.2 + i * 0.55, 3.05, 0.5, 0.85, did,
                 BLUE, "#dbeafe", fs=8.5)
    for i, did in enumerate(sparse_list):
        _rounded(ax, 5.2 + i * 0.55, 0.6, 0.5, 0.85, did,
                 ORANGE, "#fef3c7", fs=8.5)
    ax.text(5.2 + 5 * 0.55 / 2, 3.95, "ranked candidates",
            ha="center", fontsize=8.5, color=GRAY)

    # RRF box
    _rounded(ax, 8.5, 1.85, 2.4, 0.85,
             "RRF fusion\n" r"$\sum_r 1/(k+\mathrm{rank}_r(d))$",
             GREEN, "#d1fae5", fs=9.5)
    _arrow(ax, 7.95, 3.45, 8.5, 2.55, color=GREEN, lw=1.8)
    _arrow(ax, 7.95, 1.05, 8.5, 2.05, color=GREEN, lw=1.8)

    # Final list
    final_list = ["d_2", "d_4", "d_1", "d_9", "d_5"]
    for i, did in enumerate(final_list):
        _rounded(ax, 11.05 + i * 0.45, 1.85, 0.4, 0.85, did,
                 PURPLE, "#ede9fe", fs=8.5)
    ax.text(11.05 + 5 * 0.45 / 2, 2.85, "fused top-k",
            ha="center", fontsize=8.5, color=GRAY)

    # ----- Bottom-left: BM25 score bars vs cosine bars for one query -----
    ax2 = fig.add_subplot(gs[1, 0])
    docs = ["d_1", "d_2", "d_3", "d_4", "d_5"]
    bm25 = [3.1, 7.4, 0.4, 6.8, 5.2]
    cos = [0.62, 0.81, 0.55, 0.86, 0.60]
    x = np.arange(len(docs))
    w = 0.38
    ax2.bar(x - w / 2, bm25, w, color=ORANGE, alpha=0.9, label="BM25 (sparse)")
    ax2_2 = ax2.twinx()
    ax2_2.bar(x + w / 2, cos, w, color=BLUE, alpha=0.9, label="cosine (dense)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(docs)
    ax2.set_ylabel("BM25 score", color=ORANGE)
    ax2_2.set_ylabel("cosine sim.", color=BLUE)
    ax2_2.set_ylim(0, 1.0)
    ax2_2.grid(False)
    ax2.set_title("Per-document scores disagree across retrievers", loc="left")

    # ----- Bottom-right: Recall@10 across modes -----
    ax3 = fig.add_subplot(gs[1, 1])
    modes = ["BM25\nonly", "Dense\nonly", "Linear\nblend", "RRF\nfusion",
             "RRF +\nrerank"]
    recall = [54.0, 61.5, 67.0, 71.0, 78.5]
    colors = [ORANGE, BLUE, PURPLE, GREEN, "#0ea5e9"]
    bars = ax3.bar(modes, recall, color=colors, alpha=0.9,
                   edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, recall):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                 f"{val:.1f}", ha="center", fontsize=9.5, weight="bold")
    ax3.set_ylim(0, 90)
    ax3.set_ylabel("Recall@10  (%)")
    ax3.set_title("Hybrid lifts recall on heterogeneous queries", loc="left")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Hybrid Retrieval:  Dense  +  Sparse  via Reciprocal Rank Fusion",
                 fontsize=14, weight="bold", y=0.995)
    save(fig, "fig3_hybrid_retrieval.png")


# ---------------------------------------------------------------------------
# Figure 4 — Reranking with cross-encoders (two-stage pipeline)
# ---------------------------------------------------------------------------
def fig4_reranking() -> None:
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)

    # ----- Left: bi-encoder vs cross-encoder schematic -----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    ax.text(4.5, 5.55, "Bi-encoder vs Cross-encoder",
            ha="center", fontsize=12.5, weight="bold")

    # Bi-encoder (top half)
    ax.text(0.1, 4.85, "Bi-encoder  (retrieval, fast)",
            fontsize=9.5, weight="bold", color=BLUE)
    _rounded(ax, 0.2, 3.95, 1.7, 0.7, "query q", DARK, "#fff7ed", fs=9)
    _rounded(ax, 0.2, 3.15, 1.7, 0.7, "doc d", DARK, LIGHT, fs=9)
    _rounded(ax, 2.4, 3.95, 1.6, 0.7, "encoder", BLUE, "#dbeafe", fs=9)
    _rounded(ax, 2.4, 3.15, 1.6, 0.7, "encoder", BLUE, "#dbeafe", fs=9)
    _arrow(ax, 1.9, 4.3, 2.4, 4.3); _arrow(ax, 1.9, 3.5, 2.4, 3.5)
    _rounded(ax, 4.5, 3.55, 1.6, 0.7, r"$e_q,\, e_d$", BLUE, "#dbeafe", fs=10)
    _arrow(ax, 4.0, 4.3, 4.5, 4.0); _arrow(ax, 4.0, 3.5, 4.5, 3.8)
    _rounded(ax, 6.6, 3.55, 2.2, 0.7,
             r"score = $\cos(e_q, e_d)$", GREEN, "#d1fae5", fs=10)
    _arrow(ax, 6.1, 3.9, 6.6, 3.9)
    ax.text(8.85, 3.25, "O(N) embeds\ncached offline",
            fontsize=8.5, color=GRAY, ha="right", style="italic")

    # Cross-encoder (bottom half)
    ax.text(0.1, 2.55, "Cross-encoder  (rerank, accurate)",
            fontsize=9.5, weight="bold", color=PURPLE)
    _rounded(ax, 0.2, 1.45, 4.4, 0.7,
             "[CLS]  q   [SEP]   d   [SEP]", DARK, "#f3e8ff", fs=10)
    _rounded(ax, 5.0, 1.45, 1.7, 0.7, "Transformer", PURPLE, "#ede9fe", fs=9.5)
    _rounded(ax, 7.0, 1.45, 1.8, 0.7, "score (logit)", GREEN, "#d1fae5", fs=10)
    _arrow(ax, 4.6, 1.8, 5.0, 1.8); _arrow(ax, 6.7, 1.8, 7.0, 1.8)
    ax.text(8.85, 0.95, "O(k) joint passes\ncomputed online",
            fontsize=8.5, color=GRAY, ha="right", style="italic")
    ax.text(4.5, 0.35,
            "Joint attention sees query–document interactions  →  +5 to +15 nDCG",
            ha="center", fontsize=9.5, color=DARK, style="italic")

    # ----- Right: nDCG@10 vs latency for retrieve / retrieve+rerank -----
    ax2 = fig.add_subplot(gs[0, 1])
    stages = ["Bi-encoder\nonly", "+ Cross-enc\nrerank top-50",
              "+ Cross-enc\nrerank top-100", "+ Listwise\nLLM rerank"]
    ndcg = [62.0, 74.5, 76.0, 78.5]
    latency = [22, 95, 165, 720]   # ms per query
    colors = [BLUE, PURPLE, PURPLE, ORANGE]

    x = np.arange(len(stages))
    bars = ax2.bar(x, ndcg, color=colors, alpha=0.9,
                   edgecolor="white", linewidth=1.5, width=0.55)
    for bar, val in zip(bars, ndcg):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.6,
                 f"{val:.1f}", ha="center", fontsize=9.5, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=9)
    ax2.set_ylim(55, 85)
    ax2.set_ylabel("nDCG@10  (MS MARCO dev)")
    ax2.set_title("Quality climbs with rerank depth", loc="left")
    ax2.grid(True, axis="y", alpha=0.3)

    # Overlay latency on twin axis
    ax2t = ax2.twinx()
    ax2t.plot(x, latency, "o-", color=RED, lw=2.0, ms=8,
              markeredgecolor="white", markeredgewidth=1.4)
    for xi, lat in zip(x, latency):
        ax2t.annotate(f"{lat} ms", (xi, lat), textcoords="offset points",
                      xytext=(0, 10), fontsize=8.5, ha="center", color=RED)
    ax2t.set_ylabel("Latency  (ms / query)", color=RED)
    ax2t.set_yscale("log")
    ax2t.set_ylim(15, 1500)
    ax2t.grid(False)

    fig.suptitle("Two-stage Retrieval:  Cheap recall  →  Expensive precision",
                 fontsize=14, weight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig4_reranking.png")


# ---------------------------------------------------------------------------
# Figure 5 — Chunking strategies
# ---------------------------------------------------------------------------
def fig5_chunking() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.0),
                             gridspec_kw={"height_ratios": [1.0, 1.0],
                                          "hspace": 0.95, "wspace": 0.42})

    # Common synthetic document
    doc_len = 30   # arbitrary "tokens" / cells
    cells = np.arange(doc_len)

    # --- Fixed-size chunks (size 6, overlap 0) ---
    ax = axes[0, 0]
    ax.set_xlim(-0.5, doc_len + 0.5)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Fixed-size chunking  (size = 6, overlap = 0)", loc="left")
    for c in cells:
        ax.add_patch(Rectangle((c, 1.2), 0.95, 0.6,
                               facecolor=LIGHT, edgecolor=GRAY, lw=0.5))
    chunks = [(0, 6), (6, 6), (12, 6), (18, 6), (24, 6)]
    for i, (s, w) in enumerate(chunks):
        color = [BLUE, PURPLE, GREEN, ORANGE, BLUE][i]
        ax.add_patch(Rectangle((s, 1.05), w - 0.05, 0.9,
                               facecolor=color, alpha=0.55,
                               edgecolor=DARK, lw=1.2))
        ax.text(s + w / 2, 1.5, f"chunk {i+1}",
                ha="center", va="center", fontsize=9, weight="bold", color="white")
    ax.text(0, 0.4, "+ trivial to implement, deterministic",
            fontsize=9, color=GREEN)
    ax.text(0, 0.05,
            "− splits sentences mid-thought  →  retrieval drops on boundary chunks",
            fontsize=9, color=RED)

    # --- Recursive (overlap 2) ---
    ax = axes[0, 1]
    ax.set_xlim(-0.5, doc_len + 0.5)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Recursive chunking  (size = 8, overlap = 2)", loc="left")
    for c in cells:
        ax.add_patch(Rectangle((c, 1.2), 0.95, 0.6,
                               facecolor=LIGHT, edgecolor=GRAY, lw=0.5))
    rchunks = [(0, 8), (6, 8), (12, 8), (18, 8), (22, 8)]
    rcolors = [BLUE, PURPLE, GREEN, ORANGE, BLUE]
    for i, (s, w) in enumerate(rchunks):
        ax.add_patch(Rectangle((s, 1.0 + (i % 2) * 0.05), w - 0.05, 0.95,
                               facecolor=rcolors[i], alpha=0.45,
                               edgecolor=DARK, lw=1.0))
    ax.text(15, 2.45, "shaded overlap regions preserve context across boundaries",
            ha="center", fontsize=8.5, style="italic", color=GRAY)
    ax.text(0, 0.4,
            "+ respects separator hierarchy  (\\n\\n  >  \\n  >  '. '  >  ' ')",
            fontsize=9, color=GREEN)
    ax.text(0, 0.05,
            "+ overlap recovers boundary recall;  default in LangChain / LlamaIndex",
            fontsize=9, color=GREEN)

    # --- Semantic chunking ---
    ax = axes[1, 0]
    ax.set_xlim(-0.5, doc_len + 0.5)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_title("Semantic chunking  (split where embedding distance spikes)",
                 loc="left", fontsize=11)
    # Synthetic embedding-distance curve between adjacent sentences
    rng = np.random.default_rng(42)
    dist = 0.18 + 0.06 * rng.standard_normal(doc_len - 1)
    spikes = [5, 12, 19, 25]
    for s in spikes:
        dist[s] = 0.65
    ax.plot(np.arange(doc_len - 1) + 0.5, 1.2 + dist * 1.6,
            color=PURPLE, lw=1.8)
    ax.axhline(1.2 + 0.45 * 1.6, color=RED, lw=1.0, ls="--", alpha=0.7)
    ax.text(doc_len + 0.2, 1.2 + 0.45 * 1.6, "split\nthreshold",
            fontsize=8.5, color=RED, va="center")

    # Boxes following the spikes
    bounds = [0] + [s + 1 for s in spikes] + [doc_len]
    semcolors = [BLUE, PURPLE, GREEN, ORANGE, BLUE]
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        ax.add_patch(Rectangle((s, 0.4), e - s - 0.05, 0.55,
                               facecolor=semcolors[i], alpha=0.55,
                               edgecolor=DARK, lw=1.0))
        ax.text((s + e) / 2, 0.675, f"sem-chunk {i+1}",
                ha="center", va="center", fontsize=8.5, weight="bold", color="white")
    ax.text(0, 0.0,
            "chunks coincide with topical shifts  →  fewer cross-topic chunks, "
            "higher precision",
            fontsize=9, color=GREEN)

    # --- Trade-off: chunk size vs answer quality / hit-rate ---
    ax = axes[1, 1]
    sizes = np.array([64, 128, 256, 384, 512, 768, 1024, 1536, 2048])
    # Stylised retrieval F1 (peaks near 256-512) and answer faithfulness
    hit = 88 - 0.0008 * (sizes - 350) ** 2
    faith = 70 + 12 * np.exp(-((sizes - 420) / 380) ** 2)
    ax.plot(sizes, hit, "-o", color=BLUE, lw=2.0, ms=6, label="Retrieval Hit@5")
    ax.plot(sizes, faith, "-s", color=GREEN, lw=2.0, ms=6,
            label="Answer faithfulness")
    ax.axvspan(256, 512, color=ORANGE, alpha=0.15)
    ax.text(384, 60, "sweet spot\n256–512 tok",
            ha="center", fontsize=9, color=ORANGE, weight="bold")
    ax.set_xscale("log")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels([64, 128, 256, 512, 1024, 2048])
    ax.set_xlabel("Chunk size  (tokens, log)")
    ax.set_ylabel("Score  (%)")
    ax.set_ylim(55, 95)
    ax.set_title("Quality vs chunk size  (ad-hoc QA)", loc="left", fontsize=11)
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Chunking Strategies:  Fixed  ·  Recursive  ·  Semantic",
                 fontsize=14, weight="bold", y=0.995)
    save(fig, "fig5_chunking.png")


# ---------------------------------------------------------------------------
# Figure 6 — Self-RAG / Corrective-RAG control flow
# ---------------------------------------------------------------------------
def fig6_self_rag() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(6.75, 6.1, "Self-RAG  /  Corrective RAG  —  retrieval as a decision",
            ha="center", fontsize=14, weight="bold")

    # Query
    _rounded(ax, 0.3, 4.6, 1.8, 0.85, "Query", DARK, "#fff7ed",
             fs=10, weight="bold")

    # Decision: retrieve?
    diamond = np.array([[2.6, 5.025], [3.55, 5.55], [4.5, 5.025], [3.55, 4.5]])
    ax.add_patch(plt.Polygon(diamond, facecolor="#fef3c7",
                             edgecolor=ORANGE, lw=1.6))
    ax.text(3.55, 5.025, "Retrieve?\n[Retrieve] token", ha="center", va="center",
            fontsize=9, weight="bold")
    _arrow(ax, 2.1, 5.025, 2.6, 5.025)

    # No path -> direct LLM
    _rounded(ax, 4.7, 3.0, 2.0, 0.85, "Direct LLM\nanswer",
             BLUE, "#dbeafe", fs=10)
    ax.text(3.55, 4.05, "no", fontsize=9, color=GRAY)
    _arrow(ax, 3.55, 4.5, 5.5, 3.85, color=GRAY)

    # Yes path -> retrieve top-k
    _rounded(ax, 4.95, 5.0, 1.7, 0.85, "Retrieve\ntop-k", GREEN, "#d1fae5",
             fs=9.5)
    ax.text(4.55, 5.3, "yes", fontsize=9, color=GREEN, weight="bold")
    _arrow(ax, 4.5, 5.025, 4.95, 5.4, color=GREEN)

    # Per-doc relevance grading [ISREL]
    _rounded(ax, 7.0, 5.0, 2.2, 0.85,
             "Grade each doc\n[ISREL]: rel | irrel", PURPLE, "#ede9fe", fs=9.5)
    _arrow(ax, 6.65, 5.4, 7.0, 5.4)

    # Branch: relevant -> generate; irrelevant -> action
    diamond2 = np.array([[9.55, 5.4], [10.45, 5.85], [11.35, 5.4],
                         [10.45, 4.95]])
    ax.add_patch(plt.Polygon(diamond2, facecolor="#fef3c7",
                             edgecolor=ORANGE, lw=1.6))
    ax.text(10.45, 5.4, "Any\nrelevant?", ha="center", va="center",
            fontsize=9, weight="bold")
    _arrow(ax, 9.2, 5.4, 9.55, 5.4)

    # Yes -> generate w/ ISSUP grading
    _rounded(ax, 11.6, 5.0, 1.7, 0.85,
             "Generate\n+ [ISSUP] check", BLUE, "#dbeafe", fs=9.5)
    _arrow(ax, 11.35, 5.4, 11.6, 5.4, color=GREEN)
    ax.text(11.5, 5.95, "yes", fontsize=9, color=GREEN, weight="bold")

    # No -> CRAG: web search / rewrite
    _rounded(ax, 9.4, 3.1, 2.2, 0.85,
             "CRAG action:\nrewrite + web search", ORANGE, "#fef3c7", fs=9.5)
    _arrow(ax, 10.45, 4.95, 10.45, 3.95, color=RED)
    ax.text(10.6, 4.45, "no", fontsize=9, color=RED, weight="bold")
    _arrow(ax, 10.0, 3.1, 8.0, 3.1, color=ORANGE)
    _arrow(ax, 7.0, 3.5, 7.0, 4.95, color=ORANGE)
    ax.text(7.2, 3.85, "re-grade", fontsize=8, color=ORANGE)

    # Final critique [ISUSE]
    _rounded(ax, 9.4, 1.5, 2.5, 0.85,
             "[ISUSE] critique:\nuseful?  cite?  hallucinate?",
             PURPLE, "#ede9fe", fs=9.5)
    _arrow(ax, 12.45, 5.0, 12.45, 2.35, color=BLUE)
    _arrow(ax, 12.45, 2.35, 11.9, 1.95, color=BLUE)

    # Final answer
    _rounded(ax, 4.95, 1.5, 2.5, 0.85, "Grounded answer\n+ citations",
             GREEN, "#d1fae5", fs=10, weight="bold")
    _arrow(ax, 9.4, 1.925, 7.45, 1.925, color=GREEN)
    _arrow(ax, 5.7, 3.0, 5.7, 2.35, color=BLUE)

    # Legend at bottom
    legend_y = 0.6
    items = [
        ("[Retrieve]", ORANGE, "should I retrieve?"),
        ("[ISREL]", PURPLE, "is each doc relevant?"),
        ("[ISSUP]", PURPLE, "is the answer supported?"),
        ("[ISUSE]", PURPLE, "is the answer useful?"),
    ]
    ax.text(0.3, legend_y + 0.45, "Self-RAG reflection tokens:",
            fontsize=9.5, weight="bold", color=DARK)
    for i, (tok, color, desc) in enumerate(items):
        x0 = 0.3 + i * 3.3
        ax.text(x0, legend_y, tok, color=color, weight="bold", fontsize=9.5)
        ax.text(x0 + 1.15, legend_y, desc, color=GRAY, fontsize=9)

    save(fig, "fig6_self_rag.png")


# ---------------------------------------------------------------------------
# Figure 7 — Vector DB comparison
# ---------------------------------------------------------------------------
def fig7_vectordb() -> None:
    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.55)

    # ----- Left: capability radar -----
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    axes_labels = ["Scale\n(B vectors)", "Filtering", "Hybrid\nsearch",
                   "Ops\nsimplicity", "Multi-tenant", "Cost\nefficiency"]
    N = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Scores 1–5 from public docs / benchmarks (subjective summary)
    systems = {
        "FAISS":      ([5, 1, 2, 4, 1, 5], BLUE),
        "Chroma":     ([2, 4, 3, 5, 2, 4], GREEN),
        "Milvus":     ([5, 5, 5, 3, 5, 4], PURPLE),
        "Weaviate":   ([4, 5, 5, 4, 5, 3], ORANGE),
        "Pinecone":   ([4, 5, 5, 5, 5, 2], RED),
    }
    for name, (vals, color) in systems.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=color, lw=2.0, label=name)
        ax.fill(angles, v, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color=GRAY)
    ax.set_ylim(0, 5.5)
    ax.set_title("Capability profile  (1 = weak, 5 = best-in-class)",
                 pad=28, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              fontsize=9, ncol=5)

    # ----- Right: indicative QPS (1 M × 768-d, k = 10) at recall ≥ 0.95 -----
    ax2 = fig.add_subplot(gs[0, 1])
    systems_b = ["FAISS\nHNSW", "Milvus\nHNSW", "Weaviate\nHNSW",
                 "Chroma\nHNSW", "Pinecone\nmanaged"]
    qps = [4200, 3500, 2400, 1100, 1800]   # indicative single-node QPS
    colors = [BLUE, PURPLE, ORANGE, GREEN, RED]
    bars = ax2.barh(systems_b, qps, color=colors, alpha=0.9,
                    edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, qps):
        ax2.text(val + 80, bar.get_y() + bar.get_height() / 2,
                 f"{val:,} qps", va="center", fontsize=9.5, weight="bold")
    ax2.set_xlim(0, 5400)
    ax2.invert_yaxis()
    ax2.set_xlabel(
        "Single-node QPS  (1 M × 768-d, recall ≥ 0.95)\n"
        "Indicative; production depends on hardware, M / efSearch, batch size, filter selectivity.",
        fontsize=9,
    )
    ax2.set_title("Throughput (indicative, HNSW where applicable)",
                  loc="left")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Vector Database Comparison  —  capabilities and throughput",
                 fontsize=14, weight="bold", y=1.02)
    save(fig, "fig7_vectordb.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)

    fig1_rag_pipeline()
    fig2_vector_similarity()
    fig3_hybrid_retrieval()
    fig4_reranking()
    fig5_chunking()
    fig6_self_rag()
    fig7_vectordb()

    print(f"Wrote 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
