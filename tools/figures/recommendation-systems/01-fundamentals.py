"""
Figure generation script for Recommendation Systems Part 01: Fundamentals.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly.

Figures:
    fig1_user_item_matrix      Sparse user-item rating matrix heatmap (the
                               core data structure of CF; visualises sparsity).
    fig2_paradigm_comparison   Radar/bar chart comparing CF vs. Content vs.
                               Hybrid across 6 dimensions.
    fig3_funnel_architecture   Multi-stage recommendation funnel (millions ->
                               top-10) with latencies and item counts.
    fig4_embedding_space       2D scatter showing users and items in a learned
                               latent space; geometric meaning of MF.
    fig5_evaluation_metrics    Side-by-side visualisation of Precision@K,
                               Recall@K and NDCG on a worked example.
    fig6_cold_start_longtail   Long-tail item popularity distribution with the
                               cold-start zone highlighted.
    fig7_business_impact       Bar chart of cited business-impact stats
                               (Netflix, YouTube, Amazon).

Usage:
    python3 scripts/figures/recommendation-systems/01-fundamentals.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
C_BLUE = COLORS["primary"]     # primary
C_PURPLE = COLORS["accent"]   # secondary
C_GREEN = COLORS["success"]    # accent / good
C_AMBER = COLORS["warning"]    # warning / highlight
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_BG = COLORS["bg"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "recommendation-systems" / "01-fundamentals"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "recommendation-systems" / "01-入门与基础概念"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: User-item rating matrix (sparsity heatmap)
# ---------------------------------------------------------------------------
def fig1_user_item_matrix() -> None:
    """A small sparse user x item rating matrix with missing entries shown."""
    rng = np.random.default_rng(7)
    n_users, n_items = 8, 12
    # Density ~25%
    mask = rng.random((n_users, n_items)) < 0.28
    ratings = np.where(mask, rng.integers(1, 6, size=(n_users, n_items)), np.nan)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("Blues")
    cmap.set_bad(color="#eef2f7")

    im = ax.imshow(ratings, cmap=cmap, vmin=1, vmax=5, aspect="auto")

    # Annotate cells
    for i in range(n_users):
        for j in range(n_items):
            v = ratings[i, j]
            if np.isnan(v):
                ax.text(j, i, "?", ha="center", va="center",
                        color=C_GRAY, fontsize=11, fontweight="bold")
            else:
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color="white" if v >= 3 else C_DARK,
                        fontsize=10, fontweight="bold")

    ax.set_xticks(range(n_items))
    ax.set_yticks(range(n_users))
    ax.set_xticklabels([f"i{j+1}" for j in range(n_items)])
    ax.set_yticklabels([f"u{i+1}" for i in range(n_users)])
    ax.set_xlabel("Items", fontsize=11)
    ax.set_ylabel("Users", fontsize=11)
    ax.set_title("User–Item Rating Matrix: most entries are missing",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(False)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("rating (1–5)", fontsize=10)

    # Density annotation
    density = mask.mean()
    ax.text(1.02, -0.18,
            f"observed density ≈ {density:.0%}    "
            f"(real systems: < 0.1%)",
            transform=ax.transAxes, ha="right", fontsize=10, color=C_DARK)

    _save(fig, "fig1_user_item_matrix")


# ---------------------------------------------------------------------------
# Figure 2: Paradigm comparison (grouped bar)
# ---------------------------------------------------------------------------
def fig2_paradigm_comparison() -> None:
    """Compare CF vs. Content-based vs. Hybrid across six positive dimensions."""
    dims = [
        "Handles new\nitems",
        "Handles new\nusers",
        "Serendipity",
        "Explainability",
        "Works on\nsparse data",
        "Works without\ndomain features",
    ]
    # Subjective 0-5 scores; higher is better on every axis.
    cf =        [1, 1, 5, 2, 1, 5]
    content =   [5, 3, 2, 5, 4, 1]
    hybrid =    [4, 3, 4, 4, 4, 3]

    x = np.arange(len(dims))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w, cf, w, label="Collaborative Filtering",
           color=C_BLUE, edgecolor="white")
    ax.bar(x, content, w, label="Content-Based",
           color=C_PURPLE, edgecolor="white")
    ax.bar(x + w, hybrid, w, label="Hybrid",
           color=C_GREEN, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("strength  (higher is better)", fontsize=10)
    ax.set_title("Three paradigms, complementary strengths",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.set_axisbelow(True)

    _save(fig, "fig2_paradigm_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Multi-stage funnel
# ---------------------------------------------------------------------------
def fig3_funnel_architecture() -> None:
    """Visualise the recall -> rank -> rerank funnel."""
    stages = [
        ("Catalog",            10_000_000, "—",      C_GRAY),
        ("Recall",              2_000,     "< 10 ms", C_BLUE),
        ("Coarse rank",         500,       "< 50 ms", C_PURPLE),
        ("Fine rank",           50,        "< 100 ms", C_GREEN),
        ("Rerank + policy",     10,        "< 20 ms", C_AMBER),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, len(stages) + 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    max_w = 9.0
    min_w = 4.4   # ensure even the narrowest box fits its label
    log_max = np.log10(stages[0][1])
    log_min = np.log10(stages[-1][1])

    for idx, (name, n, lat, color) in enumerate(stages):
        if n > 1:
            t = (np.log10(n) - log_min) / (log_max - log_min)
        else:
            t = 0.0
        w = min_w + t * (max_w - min_w)
        x0 = (10 - w) / 2
        y0 = idx + 0.15
        h = 0.7
        box = FancyBboxPatch((x0, y0), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.18",
                             linewidth=0, facecolor=color, alpha=0.92)
        ax.add_patch(box)
        ax.text(5, y0 + h / 2,
                f"{name}    {_fmt(n)} items"
                + (f"    •    {lat}" if lat != "—" else ""),
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")

        if idx < len(stages) - 1:
            arrow = FancyArrowPatch((5, y0 + h + 0.02),
                                    (5, y0 + h + 0.28),
                                    arrowstyle="-|>", mutation_scale=14,
                                    color=C_DARK, linewidth=1.4)
            ax.add_patch(arrow)

    ax.set_title(r"Recommendation funnel: $10^7 \rightarrow 10$ in under 200 ms",
                 fontsize=13, fontweight="bold", pad=10)

    # Side annotation: model cost grows as candidates shrink
    ax.text(11.0, 0.5, "cheap models\n(ANN search,\n inverted index)",
            fontsize=9, color=C_DARK, ha="left", va="top")
    ax.text(11.0, len(stages) - 0.5, "expensive models\n(deep nets,\n cross features)",
            fontsize=9, color=C_DARK, ha="left", va="bottom")
    ax.annotate("", xy=(10.85, len(stages) - 0.4), xytext=(10.85, 0.6),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.4))

    _save(fig, "fig3_funnel_architecture")


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return f"{n}"


# ---------------------------------------------------------------------------
# Figure 4: Latent embedding space
# ---------------------------------------------------------------------------
def fig4_embedding_space() -> None:
    """Show users and items projected into a 2D latent space."""
    rng = np.random.default_rng(42)

    # Three taste clusters
    centers = np.array([[-2.0, 1.5], [2.0, 1.8], [0.2, -2.0]])
    item_labels = ["Action", "Romance", "Documentary"]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # Items
    for c, label, color in zip(centers, item_labels, PALETTE):
        pts = c + rng.normal(0, 0.45, size=(14, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=70, color=color, alpha=0.55,
                   edgecolor="white", linewidth=1.0,
                   label=f"{label} items")

    # A specific user near the action cluster
    user = np.array([-1.6, 1.0])
    ax.scatter(*user, s=320, marker="*", color=C_DARK, edgecolor="white",
               linewidth=1.5, zorder=5, label="user u")

    # Draw an arrow to the closest item (max dot product ≈ nearest)
    target = centers[0] + np.array([0.3, 0.1])
    ax.annotate("", xy=target, xytext=user,
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.6))
    midpoint = (user + target) / 2
    ax.text(midpoint[0] - 0.1, midpoint[1] + 0.25,
            r"high  $\mathbf{p}_u^\top \mathbf{q}_i$", fontsize=11,
            color=C_DARK, fontweight="bold")

    # Decorations
    ax.axhline(0, color=C_GRAY, lw=0.6, alpha=0.6)
    ax.axvline(0, color=C_GRAY, lw=0.6, alpha=0.6)
    ax.set_xlabel("latent factor 1   (e.g. action ↔ contemplative)", fontsize=10)
    ax.set_ylabel("latent factor 2   (e.g. light ↔ serious)", fontsize=10)
    ax.set_title("Matrix factorization: users and items share a vector space",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    _save(fig, "fig4_embedding_space")


# ---------------------------------------------------------------------------
# Figure 5: Evaluation metrics on a worked example
# ---------------------------------------------------------------------------
def fig5_evaluation_metrics() -> None:
    """Worked example: precision/recall/NDCG visualised on one ranked list."""
    # Ranked list of 10 items; True/False = relevant or not.
    ranked = [True, True, False, True, False, False, True, False, False, False]
    n = len(ranked)
    total_relevant = 5  # ground-truth relevant items in the catalog
    # Note: the bottom curve uses total_relevant as the denominator for
    # recall, so 4 hits in the top-10 means recall plateaus at 0.8.

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5),
                             gridspec_kw={"height_ratios": [1.2, 2]})

    # --- top: the ranked list as coloured boxes ---
    ax = axes[0]
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for k, rel in enumerate(ranked):
        color = C_GREEN if rel else "#e2e8f0"
        rect = FancyBboxPatch((k + 0.05, 0.15), 0.9, 0.7,
                              boxstyle="round,pad=0.01,rounding_size=0.08",
                              linewidth=0, facecolor=color)
        ax.add_patch(rect)
        ax.text(k + 0.5, 0.5, f"{k+1}", ha="center", va="center",
                color="white" if rel else C_DARK, fontsize=11, fontweight="bold")
    ax.set_title("Ranked recommendations  (green = relevant)",
                 fontsize=12, fontweight="bold", loc="left")

    # --- bottom: cumulative metrics curves ---
    ax = axes[1]
    ks = np.arange(1, n + 1)
    hits = np.cumsum(ranked)
    precision = hits / ks
    recall = hits / total_relevant

    # NDCG@k with binary relevance
    gains = np.array([1 if r else 0 for r in ranked], dtype=float)
    discounts = 1.0 / np.log2(ks + 1)
    dcg = np.cumsum(gains * discounts)
    ideal_gains = np.sort(gains)[::-1]
    idcg = np.cumsum(ideal_gains * discounts)
    idcg[idcg == 0] = 1.0
    ndcg = dcg / idcg

    ax.plot(ks, precision, "-o", color=C_BLUE, lw=2, label="Precision@k")
    ax.plot(ks, recall, "-o", color=C_PURPLE, lw=2, label="Recall@k")
    ax.plot(ks, ndcg, "-o", color=C_GREEN, lw=2, label="NDCG@k")

    ax.set_xticks(ks)
    ax.set_xlabel("k (cut-off)", fontsize=10)
    ax.set_ylabel("score", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Same list, three different stories",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    _save(fig, "fig5_evaluation_metrics")


# ---------------------------------------------------------------------------
# Figure 6: Long-tail distribution and cold-start zone
# ---------------------------------------------------------------------------
def fig6_cold_start_longtail() -> None:
    """Power-law popularity distribution with cold-start region highlighted."""
    rng = np.random.default_rng(0)
    n_items = 1000
    rank = np.arange(1, n_items + 1)
    # Zipf-like: count ~ C / rank^0.95
    counts = 50_000 / (rank ** 0.95)
    counts = counts * (1 + 0.05 * rng.standard_normal(n_items))
    counts = np.clip(counts, 1, None)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(rank, counts, color=C_BLUE, alpha=0.18)
    ax.plot(rank, counts, color=C_BLUE, lw=1.4)

    # Head (top 5%)
    head_cut = int(n_items * 0.05)
    ax.fill_between(rank[:head_cut], counts[:head_cut],
                    color=C_AMBER, alpha=0.55, label="Head: top 5% of items")

    # Long tail (bottom 50%)
    tail_start = int(n_items * 0.5)
    ax.fill_between(rank[tail_start:], counts[tail_start:],
                    color=C_PURPLE, alpha=0.45,
                    label="Long tail: bottom 50% of items")

    # Cold-start band (very few interactions)
    cold_start_threshold = 100
    ax.axhline(cold_start_threshold, color=C_GRAY, ls="--", lw=1)
    ax.text(1.1, cold_start_threshold * 1.15,
            "cold-start zone (< 100 interactions)",
            fontsize=9, color=C_DARK)

    # Annotations: share of interactions
    head_share = counts[:head_cut].sum() / counts.sum()
    tail_share = counts[tail_start:].sum() / counts.sum()
    ax.annotate(
        f"top 5% absorb ≈ {head_share:.0%}\nof all interactions",
        xy=(head_cut, counts[head_cut]),
        xytext=(head_cut * 2.0, counts[0] * 0.3),
        fontsize=10, color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=C_AMBER, lw=1),
        arrowprops=dict(arrowstyle="-", color=C_AMBER, lw=1),
    )
    ax.annotate(
        f"long tail (50% of catalog)\nshares only ≈ {tail_share:.0%}",
        xy=(int(n_items * 0.7), counts[int(n_items * 0.7)]),
        xytext=(60, counts[tail_start] * 0.18),
        fontsize=10, color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=C_PURPLE, lw=1),
        arrowprops=dict(arrowstyle="-", color=C_PURPLE, lw=1),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("item rank (most popular → least popular)", fontsize=10)
    ax.set_ylabel("interactions per item (log scale)", fontsize=10)
    ax.set_title("The long tail and the cold-start zone",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower left", framealpha=0.95)

    _save(fig, "fig6_cold_start_longtail")


# ---------------------------------------------------------------------------
# Figure 7: Business impact stats
# ---------------------------------------------------------------------------
def fig7_business_impact() -> None:
    """Cited business-impact numbers across major platforms."""
    labels = [
        "Amazon\nrevenue from\nrecommendations",
        "Netflix\nstreams driven by\nrecommendations",
        "YouTube\nwatch time from\nrecommendations",
        "Spotify\nDiscover Weekly\nstreams (weekly)",
    ]
    # Cited values; conservative figures from public talks / papers.
    values = [35, 75, 70, None]   # percentages where applicable
    extra = ["", "", "", "2.3 B+\n streams / week"]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

    for i, (v, e, c) in enumerate(zip(values, extra, colors)):
        if v is not None:
            ax.bar(i, v, color=c, edgecolor="white", width=0.6)
            ax.text(i, v + 1.5, f"{v}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=C_DARK)
        else:
            ax.bar(i, 60, color=c, alpha=0.25, edgecolor="white",
                   width=0.6, hatch="//")
            ax.text(i, 30, e, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=C_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 95)
    ax.set_ylabel("share of activity (%)", fontsize=10)
    ax.set_title("Why platforms invest: cited business impact",
                 fontsize=13, fontweight="bold", pad=10)

    ax.text(0.5, -0.27,
            "Sources: McKinsey 2013 (Amazon), Netflix tech blog "
            "& Gomez-Uribe & Hunt 2015 (Netflix), Covington et al. "
            "RecSys 2016 (YouTube), Spotify newsroom (Discover Weekly).",
            transform=ax.transAxes, ha="center", fontsize=8.5,
            color=C_GRAY, style="italic")

    _save(fig, "fig7_business_impact")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Recommendation Systems Part 01 …")
    fig1_user_item_matrix()
    fig2_paradigm_comparison()
    fig3_funnel_architecture()
    fig4_embedding_space()
    fig5_evaluation_metrics()
    fig6_cold_start_longtail()
    fig7_business_impact()
    print("Done.")


if __name__ == "__main__":
    main()
