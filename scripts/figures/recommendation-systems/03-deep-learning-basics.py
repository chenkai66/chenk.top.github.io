"""Figure generation for Recommendation Systems Part 3: Deep Learning Foundations.

Outputs identical PNGs to both EN and ZH asset folders.
Run:  python 03-deep-learning-basics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.7,
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
AMBER = "#f59e0b"
SLATE = "#475569"
LIGHT = "#f1f5f9"
INK = "#0f172a"

ROOT = Path(__file__).resolve().parents[3]
OUT_DIRS = [
    ROOT / "source/_posts/en/recommendation-systems/03-deep-learning-basics",
    ROOT / "source/_posts/zh/recommendation-systems/03-深度学习基础模型",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    for d in OUT_DIRS:
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)
    print(f"  saved: {name}")


# ---------------------------------------------------------------------------
# Fig 1 -- Deep vs traditional CF AUC comparison
# ---------------------------------------------------------------------------
def fig1_dl_vs_traditional() -> None:
    models = ["MF", "FM", "Wide & Deep", "DeepFM", "DIN"]
    auc = [0.750, 0.780, 0.810, 0.825, 0.845]
    colors = [SLATE, AMBER, BLUE, PURPLE, GREEN]
    baseline = auc[0]
    deltas = [(v - baseline) / baseline * 100 for v in auc]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]})

    bars = ax1.bar(models, auc, color=colors, edgecolor="white", linewidth=2, width=0.62, zorder=3)
    ax1.set_ylim(0.72, 0.87)
    ax1.set_ylabel("AUC (Criteo / Avazu CTR benchmark)")
    ax1.set_title("Deep models lift CTR ranking above the MF ceiling", color=INK)
    ax1.axhline(baseline, color=SLATE, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax1.text(4.45, baseline + 0.001, "MF baseline", color=SLATE, fontsize=9, ha="right")

    for bar, v, d in zip(bars, auc, deltas):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.003,
            f"{v:.3f}\n+{d:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )

    # Right panel: sketched gain trajectory.
    x = np.arange(len(models))
    ax2.plot(x, auc, "o-", color=PURPLE, linewidth=2.5, markersize=9, zorder=3)
    ax2.fill_between(x, baseline, auc, alpha=0.18, color=PURPLE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.set_ylabel("AUC")
    ax2.set_title("Each new nonlinearity buys a few AUC points", color=INK)
    ax2.set_ylim(0.72, 0.87)
    for i, v in enumerate(auc):
        ax2.annotate(f"{v:.3f}", (i, v), xytext=(6, 6), textcoords="offset points", fontsize=9, color=INK)

    fig.suptitle("Deep Learning vs Traditional CF -- Public CTR Benchmarks", fontsize=14, color=INK, y=1.02)
    save(fig, "fig1_dl_vs_traditional.png")


# ---------------------------------------------------------------------------
# Fig 2 -- NeuMF architecture
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, color, text_color="white", fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)


def _arrow(ax, x1, y1, x2, y2, color=SLATE):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.4, zorder=2)
    ax.add_patch(a)


def fig2_ncf_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("NeuMF: Generalized MF + MLP fused into one objective", color=INK, fontsize=14, pad=14)

    # Inputs
    _box(ax, 1.0, 0.4, 1.6, 0.7, "User u", SLATE)
    _box(ax, 6.4, 0.4, 1.6, 0.7, "Item i", SLATE)

    # Embedding tables -- four total (separate per path)
    _box(ax, 0.3, 1.6, 1.6, 0.8, "GMF\nUser Emb", BLUE)
    _box(ax, 2.1, 1.6, 1.6, 0.8, "MLP\nUser Emb", PURPLE)
    _box(ax, 5.6, 1.6, 1.6, 0.8, "MLP\nItem Emb", PURPLE)
    _box(ax, 7.4, 1.6, 1.6, 0.8, "GMF\nItem Emb", BLUE)

    # Arrows from inputs to embeddings
    for ex in [1.1, 2.9]:
        _arrow(ax, 1.8, 1.1, ex, 1.6)
    for ex in [6.4, 8.2]:
        _arrow(ax, 7.2, 1.1, ex, 1.6)

    # GMF op
    _box(ax, 0.6, 3.2, 2.0, 0.8, "Element-wise\nproduct  pᵤ ⊙ qᵢ", BLUE)
    _arrow(ax, 1.1, 2.4, 1.3, 3.2, BLUE)
    _arrow(ax, 8.2, 2.4, 2.3, 3.2, BLUE)

    # MLP concat
    _box(ax, 3.6, 3.2, 2.8, 0.8, "Concat  [pᵤ ; qᵢ]", PURPLE)
    _arrow(ax, 2.9, 2.4, 4.5, 3.2, PURPLE)
    _arrow(ax, 6.4, 2.4, 5.7, 3.2, PURPLE)

    # MLP layers
    layer_y = [4.5, 5.4, 6.3]
    layer_w = [2.6, 2.0, 1.6]
    layer_labels = ["Dense 128 + ReLU", "Dense 64 + ReLU", "Dense 32 + ReLU"]
    prev_x, prev_w, prev_y = 3.6, 2.8, 4.0
    for y, w, lbl in zip(layer_y, layer_w, layer_labels):
        x = 5.0 - w / 2
        _box(ax, x, y, w, 0.6, lbl, PURPLE)
        _arrow(ax, prev_x + prev_w / 2, prev_y, x + w / 2, y, PURPLE)
        prev_x, prev_w, prev_y = x, w, y + 0.6

    # GMF output flowing up
    _arrow(ax, 1.6, 4.0, 3.6, 7.4, BLUE)

    # Concat + sigmoid
    _box(ax, 3.6, 7.2, 2.8, 0.7, "Concat  [GMF ; MLP]", INK)
    _arrow(ax, 5.0, 6.9, 5.0, 7.2, SLATE)

    _box(ax, 4.1, 8.1, 1.8, 0.65, "σ(h·[…])", GREEN)
    _arrow(ax, 5.0, 7.9, 5.0, 8.1, GREEN)

    # Side annotations
    ax.text(0.2, 4.4, "GMF path\n(generalized\ndot product)", color=BLUE, fontsize=10, fontweight="bold")
    ax.text(7.5, 5.4, "MLP path\n(learned\nnonlinear\ninteraction)", color=PURPLE, fontsize=10, fontweight="bold")
    ax.text(5.0, 8.85, "ŷᵤᵢ -- click probability", color=GREEN, fontsize=10.5, ha="center", fontweight="bold")
    ax.text(5.0, 0.1, "Separate embeddings per path -- a key NeuMF design choice (He et al., 2017)",
            ha="center", color=SLATE, fontsize=9, style="italic")

    save(fig, "fig2_ncf_architecture.png")


# ---------------------------------------------------------------------------
# Fig 3 -- YouTube DNN two-stage pipeline
# ---------------------------------------------------------------------------
def fig3_youtube_dnn() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # ---- Candidate generation tower ----
    ax1.set_title("Stage 1 -- Candidate Generation\n(millions  →  hundreds, fast)", color=BLUE, pad=10)
    feats = [
        ("Watch history\n(mean-pooled emb)", 0.5),
        ("Search tokens\n(mean-pooled emb)", 2.4),
        ("Geo / age / gender", 4.3),
    ]
    for label, x in feats:
        _box(ax1, x, 0.6, 1.5, 0.7, label, SLATE, fontsize=8.5)

    _box(ax1, 1.5, 2.0, 5.0, 0.6, "Concatenate user features", BLUE, fontsize=10)
    for _, x in feats:
        _arrow(ax1, x + 0.75, 1.3, 4.0, 2.0, BLUE)

    layers = [("Dense 1024 + ReLU", 3.0), ("Dense 512 + ReLU", 4.0), ("Dense 256 + ReLU", 5.0)]
    for lbl, y in layers:
        _box(ax1, 2.0, y, 4.0, 0.55, lbl, BLUE, fontsize=10)
    _arrow(ax1, 4.0, 2.6, 4.0, 3.0, BLUE)
    _arrow(ax1, 4.0, 3.55, 4.0, 4.0, BLUE)
    _arrow(ax1, 4.0, 4.55, 4.0, 5.0, BLUE)

    _box(ax1, 2.5, 6.2, 3.0, 0.7, "User vector u ∈ ℝ²⁵⁶", GREEN, fontsize=10)
    _arrow(ax1, 4.0, 5.55, 4.0, 6.2, GREEN)

    _box(ax1, 1.5, 7.6, 5.0, 0.85, "ANN index over video embeddings\n(HNSW / ScaNN, ~few ms)", PURPLE, fontsize=10)
    _arrow(ax1, 4.0, 6.9, 4.0, 7.6, PURPLE)

    _box(ax1, 1.5, 9.0, 5.0, 0.65, "Top-N candidates  (~hundreds)", AMBER, fontsize=10)
    _arrow(ax1, 4.0, 8.45, 4.0, 9.0, AMBER)

    # ---- Ranking tower ----
    ax2.set_title("Stage 2 -- Ranking\n(hundreds  →  K, accurate)", color=PURPLE, pad=10)
    rfeats = [
        ("Impression\nvideo emb", 0.3),
        ("Watched\nvideo embs", 1.9),
        ("Time since\nlast watch", 3.5),
        ("Position /\nlanguage", 5.1),
    ]
    for lbl, x in rfeats:
        _box(ax2, x, 0.6, 1.4, 0.75, lbl, SLATE, fontsize=8.5)

    _box(ax2, 0.5, 2.0, 6.5, 0.6, "Rich feature concatenation", PURPLE, fontsize=10)
    for _, x in rfeats:
        _arrow(ax2, x + 0.7, 1.35, 3.75, 2.0, PURPLE)

    rlayers = [("Dense 1024 + ReLU", 3.0), ("Dense 512 + ReLU", 3.85), ("Dense 256 + ReLU", 4.7), ("Dense 128 + ReLU", 5.55)]
    for lbl, y in rlayers:
        _box(ax2, 1.5, y, 4.5, 0.5, lbl, PURPLE, fontsize=10)
    prev_y = 2.6
    for _, y in rlayers:
        _arrow(ax2, 3.75, prev_y, 3.75, y, PURPLE)
        prev_y = y + 0.5

    _box(ax2, 1.0, 6.8, 5.5, 0.85, "Weighted Logistic Regression\n→ predict expected watch time", GREEN, fontsize=10)
    _arrow(ax2, 3.75, prev_y, 3.75, 6.8, GREEN)

    _box(ax2, 1.5, 9.0, 4.5, 0.65, "Top-K final list", AMBER, fontsize=10)
    _arrow(ax2, 3.75, 7.65, 3.75, 9.0, AMBER)

    fig.suptitle("YouTube DNN -- The two-stage pipeline that runs the internet  (Covington et al., RecSys 2016)",
                 fontsize=13.5, color=INK, y=1.02)
    save(fig, "fig3_youtube_dnn.png")


# ---------------------------------------------------------------------------
# Fig 4 -- Wide & Deep
# ---------------------------------------------------------------------------
def fig4_wide_and_deep() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Wide & Deep -- Memorization meets Generalization (Cheng et al., 2016)",
                 color=INK, fontsize=14, pad=14)

    # ---- Wide side ----
    _box(ax, 0.5, 0.6, 4.5, 0.7, "Sparse features  +  manual cross features  φ(x)", AMBER, fontsize=10)
    _box(ax, 1.0, 1.9, 3.5, 0.8, "Linear layer\nw·x + b", AMBER, fontsize=10.5)
    _arrow(ax, 2.75, 1.3, 2.75, 1.9, AMBER)

    _box(ax, 1.0, 5.6, 3.5, 0.75, "Wide score  ŷʷ", AMBER, fontsize=10.5)
    _arrow(ax, 2.75, 2.7, 2.75, 5.6, AMBER)

    ax.text(2.75, 4.0, "Memorize\nspecific\nco-occurrences", ha="center", color=AMBER,
            fontsize=10, style="italic", fontweight="bold")

    # ---- Deep side ----
    _box(ax, 7.0, 0.6, 4.5, 0.7, "Categorical fields  →  Embeddings", PURPLE, fontsize=10)

    deep_layers = [(7.5, 1.9, "Dense 256 + ReLU"), (7.7, 2.85, "Dense 128 + ReLU"), (7.9, 3.8, "Dense 64 + ReLU")]
    for x, y, lbl in deep_layers:
        w = 11.5 - 2 * x + 4.5
        _box(ax, x, y, 11.5 - x, 0.55, lbl, PURPLE, fontsize=10)
    # arrows
    _arrow(ax, 9.25, 1.3, 9.25, 1.9, PURPLE)
    _arrow(ax, 9.25, 2.45, 9.25, 2.85, PURPLE)
    _arrow(ax, 9.25, 3.4, 9.25, 3.8, PURPLE)

    _box(ax, 7.5, 5.6, 4.0, 0.75, "Deep score  ŷᵈ", PURPLE, fontsize=10.5)
    _arrow(ax, 9.25, 4.35, 9.25, 5.6, PURPLE)

    ax.text(9.25, 4.85, "Generalize\nvia learned\nembeddings", ha="center", color=PURPLE,
            fontsize=10, style="italic", fontweight="bold")

    # ---- Joint head ----
    _box(ax, 4.5, 7.0, 3.0, 0.8, "ŷʷ + ŷᵈ  →  σ", GREEN, fontsize=11)
    _arrow(ax, 2.75, 6.35, 5.0, 7.0, GREEN)
    _arrow(ax, 9.25, 6.35, 7.0, 7.0, GREEN)

    _box(ax, 5.0, 8.1, 2.0, 0.6, "P(click)", GREEN, fontsize=10.5)
    _arrow(ax, 6.0, 7.8, 6.0, 8.1, GREEN)

    # Optimizer note
    ax.text(2.75, 0.15, "FTRL + L1 (sparse, picks features)", ha="center", color=AMBER, fontsize=9, style="italic")
    ax.text(9.25, 0.15, "AdaGrad / Adam (dense, smooth)", ha="center", color=PURPLE, fontsize=9, style="italic")

    save(fig, "fig4_wide_and_deep.png")


# ---------------------------------------------------------------------------
# Fig 5 -- Embedding space visualization (synthetic t-SNE-like)
# ---------------------------------------------------------------------------
def fig5_embedding_space() -> None:
    rng = np.random.default_rng(42)
    clusters = [
        ("Sci-Fi", BLUE, (-5, 4), 0.9, 95),
        ("Action", AMBER, (5, 4), 1.0, 95),
        ("Documentary", GREEN, (-5, -4), 0.85, 80),
        ("Romcom", PURPLE, (5, -4), 0.95, 90),
        ("Horror", "#ef4444", (0, 6.5), 0.8, 70),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 8))
    for name, color, (cx, cy), spread, n in clusters:
        pts = rng.normal(loc=(cx, cy), scale=spread, size=(n, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=42, c=color, alpha=0.72, edgecolors="white", linewidth=0.6, label=name, zorder=3)

    # Bridge / hybrid items between Sci-Fi and Action
    bridge = rng.normal(loc=(0, 4), scale=1.0, size=(28, 2))
    ax.scatter(bridge[:, 0], bridge[:, 1], s=46, c="#06b6d4", alpha=0.78, edgecolors="white",
               linewidth=0.6, marker="D", label="Sci-Fi × Action (hybrid)", zorder=4)

    # Long-tail halo
    tail = rng.normal(loc=(0, 0), scale=4.5, size=(60, 2))
    ax.scatter(tail[:, 0], tail[:, 1], s=18, c=SLATE, alpha=0.35, edgecolors="none", label="Long-tail items", zorder=2)

    # Annotate
    ax.annotate("Tight genre cluster", xy=(-5, 4), xytext=(-9, 7),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1), fontsize=10, color=INK)
    ax.annotate("Hybrid items bridge\nrelated genres", xy=(0, 4), xytext=(3.2, 8.4),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1), fontsize=10, color=INK)
    ax.annotate("Long-tail halo\n(few interactions)", xy=(2.6, -1.8), xytext=(7.5, -1.8),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1), fontsize=10, color=INK)

    ax.set_title("Item embeddings projected to 2D (t-SNE)\nGradients alone discovered the genre structure",
                 color=INK, pad=12)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9, ncol=2)
    ax.set_aspect("equal", adjustable="box")
    save(fig, "fig5_embedding_space.png")


# ---------------------------------------------------------------------------
# Fig 6 -- Training loss + validation AUC curves
# ---------------------------------------------------------------------------
def fig6_training_curves() -> None:
    rng = np.random.default_rng(7)
    epochs = np.arange(1, 51)

    def curve(start, end, decay, noise):
        base = end + (start - end) * np.exp(-decay * epochs)
        return base + rng.normal(0, noise, size=epochs.size)

    loss_mf = curve(0.70, 0.42, 0.07, 0.006)
    loss_ncf = curve(0.68, 0.34, 0.08, 0.006)
    loss_wd = curve(0.66, 0.28, 0.10, 0.005)

    auc_mf = 0.75 + (0.79 - 0.75) * (1 - np.exp(-0.08 * epochs)) + rng.normal(0, 0.003, size=epochs.size)
    auc_ncf = 0.75 + (0.83 - 0.75) * (1 - np.exp(-0.09 * epochs)) + rng.normal(0, 0.003, size=epochs.size)
    auc_wd = 0.75 + (0.852 - 0.75) * (1 - np.exp(-0.11 * epochs)) + rng.normal(0, 0.003, size=epochs.size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for arr, color, label in [(loss_mf, SLATE, "MF"), (loss_ncf, BLUE, "NeuMF"), (loss_wd, PURPLE, "Wide & Deep")]:
        ax1.plot(epochs, arr, color=color, linewidth=2.2, label=label)
        ax1.fill_between(epochs, arr - 0.012, arr + 0.012, color=color, alpha=0.10)
    ax1.set_title("Training loss -- deeper models reach lower minima", color=INK)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE loss")
    ax1.legend(loc="upper right", framealpha=0.95)
    ax1.set_ylim(0.22, 0.78)

    for arr, color, label in [(auc_mf, SLATE, "MF"), (auc_ncf, BLUE, "NeuMF"), (auc_wd, PURPLE, "Wide & Deep")]:
        ax2.plot(epochs, arr, color=color, linewidth=2.2, label=label)
        ax2.fill_between(epochs, arr - 0.005, arr + 0.005, color=color, alpha=0.10)

    # Early-stopping marker
    es_epoch = 38
    ax2.axvline(es_epoch, color=GREEN, linestyle=":", linewidth=1.5)
    ax2.text(es_epoch + 0.5, 0.76, "Early stop\n(val AUC plateau)", color=GREEN, fontsize=9.5, fontweight="bold")

    ax2.set_title("Validation AUC -- gap holds, ranking quality grows", color=INK)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation AUC")
    ax2.legend(loc="lower right", framealpha=0.95)
    ax2.set_ylim(0.745, 0.86)

    fig.suptitle("Training dynamics: MF vs NeuMF vs Wide & Deep  (synthetic, illustrative)",
                 fontsize=13.5, color=INK, y=1.02)
    save(fig, "fig6_training_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 03 figures...")
    fig1_dl_vs_traditional()
    fig2_ncf_architecture()
    fig3_youtube_dnn()
    fig4_wide_and_deep()
    fig5_embedding_space()
    fig6_training_curves()
    print(f"Done. Wrote 6 PNGs to:")
    for d in OUT_DIRS:
        print(f"  {d}")


if __name__ == "__main__":
    main()
