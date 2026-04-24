"""
Figures for Recommendation Systems Part 8: Knowledge Graph-Enhanced Recommendation.

Generates 7 publication-quality figures and saves into BOTH the EN and ZH
asset folders so the Markdown posts can reference them with relative paths.

Figures:
  fig1_knowledge_graph.png       Movie KG example (entity-relation triples).
  fig2_transe_embedding.png      TransE: h + r approximately t in vector space.
  fig3_ripplenet_propagation.png RippleNet preference propagation through hops.
  fig4_kgat_attention.png        KGAT attention over triples.
  fig5_four_problems.png         Four problems KG solves.
  fig6_dark_knight_example.png   The Dark Knight ego-graph.
  fig7_with_vs_without_kg.png    Comparison: with KG vs without.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.style.use("seaborn-v0_8-whitegrid")

# Brand palette
BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#6b7280"
DARK = "#1f2937"
LIGHT = "#f3f4f6"

DPI = 150

OUT_DIRS = [
    Path("/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/recommendation-systems/08-knowledge-graph"),
    Path("/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/recommendation-systems/08-知识图谱增强推荐系统"),
]


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure into every output directory."""
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig1: Movie knowledge graph
# ---------------------------------------------------------------------------
def fig1_knowledge_graph() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))

    G = nx.DiGraph()

    # Movies (blue), People (purple), Genres (green), Collections (orange)
    movies = ["The Dark Knight", "Inception", "The Prestige", "Batman Begins"]
    people = ["Christopher Nolan", "Christian Bale", "Leonardo DiCaprio"]
    genres = ["Action", "Sci-Fi", "Drama"]
    coll = ["Batman Trilogy"]

    for n in movies + people + genres + coll:
        G.add_node(n)

    edges = [
        ("The Dark Knight", "Christopher Nolan", "directed_by"),
        ("The Dark Knight", "Christian Bale", "starred"),
        ("The Dark Knight", "Action", "has_genre"),
        ("The Dark Knight", "Batman Trilogy", "part_of"),
        ("Inception", "Christopher Nolan", "directed_by"),
        ("Inception", "Leonardo DiCaprio", "starred"),
        ("Inception", "Sci-Fi", "has_genre"),
        ("The Prestige", "Christopher Nolan", "directed_by"),
        ("The Prestige", "Christian Bale", "starred"),
        ("The Prestige", "Drama", "has_genre"),
        ("Batman Begins", "Christopher Nolan", "directed_by"),
        ("Batman Begins", "Christian Bale", "starred"),
        ("Batman Begins", "Action", "has_genre"),
        ("Batman Begins", "Batman Trilogy", "part_of"),
    ]
    for h, t, r in edges:
        G.add_edge(h, t, label=r)

    # Manual layout: movies in middle column, people left, genres right
    pos = {
        "The Dark Knight":     (0.0,  1.5),
        "Inception":           (0.0,  0.4),
        "The Prestige":        (0.0, -0.7),
        "Batman Begins":       (0.0, -1.8),
        "Christopher Nolan":   (-2.4, 1.0),
        "Christian Bale":      (-2.4, -0.6),
        "Leonardo DiCaprio":   (-2.4, -1.8),
        "Action":              (2.4,  1.4),
        "Sci-Fi":              (2.4,  0.4),
        "Drama":               (2.4, -0.7),
        "Batman Trilogy":      (2.4, -1.8),
    }

    color_map = {n: BLUE for n in movies}
    color_map.update({n: PURPLE for n in people})
    color_map.update({n: GREEN for n in genres})
    color_map.update({n: ORANGE for n in coll})

    # Draw edges
    for u, v, d in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=12,
            color=GRAY, alpha=0.55, lw=1.1,
            shrinkA=22, shrinkB=22, zorder=1,
        )
        ax.add_patch(arrow)
        # Edge label at midpoint, slightly offset
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, d["label"], fontsize=7, color=DARK,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.8))

    # Draw nodes as rounded boxes
    for n, (x, y) in pos.items():
        c = color_map[n]
        box = FancyBboxPatch((x - 0.55, y - 0.14), 1.1, 0.28,
                             boxstyle="round,pad=0.04",
                             fc=c, ec="white", lw=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, n, fontsize=8.5, color="white",
                ha="center", va="center", fontweight="bold", zorder=4)

    # Legend
    legend_items = [
        mpatches.Patch(color=BLUE, label="Movie (item)"),
        mpatches.Patch(color=PURPLE, label="Person"),
        mpatches.Patch(color=GREEN, label="Genre"),
        mpatches.Patch(color=ORANGE, label="Collection"),
    ]
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=10)

    ax.set_title("Movie Knowledge Graph: entities linked by typed relations",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-2.5, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    save(fig, "fig1_knowledge_graph.png")


# ---------------------------------------------------------------------------
# fig2: TransE embedding
# ---------------------------------------------------------------------------
def fig2_transe_embedding() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: vector arithmetic concept
    ax = axes[0]
    ax.set_title("TransE: $\\mathbf{h} + \\mathbf{r} \\approx \\mathbf{t}$",
                 fontsize=13, fontweight="bold")

    # Three triples sharing the relation "directed_by"
    data = [
        ("Inception", "Nolan",  np.array([0.6, 2.4]), np.array([2.6, 1.4])),
        ("Prestige",  "Nolan",  np.array([0.4, 1.4]), np.array([2.4, 0.4])),
        ("Memento",   "Nolan",  np.array([0.8, 0.4]), np.array([2.8, -0.6])),
    ]
    r_vec = np.array([2.0, -1.0])  # directed_by relation displacement

    for movie, _, h, t in data:
        ax.annotate("", xy=h, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
        ax.annotate("", xy=h + r_vec, xytext=h,
                    arrowprops=dict(arrowstyle="->", color=ORANGE,
                                    lw=1.8, linestyle="--"))
        ax.scatter(*t, s=160, color=PURPLE, zorder=4,
                   edgecolors="white", linewidths=1.5)
        ax.text(h[0] - 0.05, h[1] + 0.18, movie,
                fontsize=9, color=BLUE, ha="right", fontweight="bold")
        ax.text(t[0] + 0.12, t[1], "Nolan", fontsize=9,
                color=PURPLE, va="center", fontweight="bold")

    # Single relation arrow legend (one example)
    ax.text(2.2, -1.6, "$\\mathbf{r}$ = directed_by",
            fontsize=11, color=ORANGE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fef3c7", ec=ORANGE))

    ax.scatter(0, 0, s=80, color=DARK, zorder=4)
    ax.text(-0.05, -0.18, "origin", fontsize=8, color=DARK, ha="right")

    ax.set_xlim(-0.8, 4.2)
    ax.set_ylim(-2.2, 3.2)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_aspect("equal")
    ax.legend(handles=[
        mpatches.Patch(color=BLUE, label="Head entity (movie)"),
        mpatches.Patch(color=ORANGE, label="Relation vector"),
        mpatches.Patch(color=PURPLE, label="Tail entity (Nolan)"),
    ], loc="lower right", fontsize=9, framealpha=0.95)

    # Right: training objective (margin loss)
    ax = axes[1]
    ax.set_title("Training: pull positives close, push negatives apart",
                 fontsize=13, fontweight="bold")

    # Anchor h+r
    anchor = np.array([2.0, 1.5])
    ax.scatter(*anchor, s=260, color=ORANGE, marker="*",
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.text(anchor[0], anchor[1] + 0.35, "$\\mathbf{h}+\\mathbf{r}$",
            fontsize=12, ha="center", fontweight="bold", color=ORANGE)

    # Positive tail (close)
    pos_t = anchor + np.array([0.35, -0.25])
    ax.scatter(*pos_t, s=180, color=GREEN, zorder=4,
               edgecolors="white", linewidths=1.5)
    ax.text(pos_t[0] + 0.1, pos_t[1] - 0.2,
            "valid tail $\\mathbf{t}$", fontsize=10, color=GREEN,
            fontweight="bold")
    ax.annotate("", xy=pos_t, xytext=anchor,
                arrowprops=dict(arrowstyle="-", color=GREEN,
                                lw=2.0, alpha=0.7))

    # Negative tails (pushed away)
    rng = np.random.default_rng(7)
    angles = [0.6, 1.5, 2.4, -1.0, -2.0]
    for ang in angles:
        d = 1.7 + rng.uniform(-0.2, 0.2)
        nt = anchor + d * np.array([np.cos(ang), np.sin(ang)])
        ax.scatter(*nt, s=130, color=GRAY, alpha=0.7, zorder=3,
                   edgecolors="white", linewidths=1.2)
        ax.annotate("", xy=nt, xytext=anchor,
                    arrowprops=dict(arrowstyle="-", color=GRAY,
                                    lw=1.0, alpha=0.4, linestyle=":"))

    # Margin annotation
    margin_circle = plt.Circle(anchor, 1.0, fill=False,
                               edgecolor=BLUE, lw=1.5, linestyle="--", alpha=0.7)
    ax.add_patch(margin_circle)
    ax.text(anchor[0] + 1.05, anchor[1] + 0.6, "margin $\\gamma$",
            fontsize=10, color=BLUE, fontweight="bold")

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.0, 4.0)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_aspect("equal")
    ax.legend(handles=[
        mpatches.Patch(color=GREEN, label="Valid tail (pull in)"),
        mpatches.Patch(color=GRAY, label="Negative tails (push out)"),
    ], loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    save(fig, "fig2_transe_embedding.png")


# ---------------------------------------------------------------------------
# fig3: RippleNet preference propagation
# ---------------------------------------------------------------------------
def fig3_ripplenet_propagation() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Hop 0 (seed), Hop 1, Hop 2 entities
    seed = "Dark Knight"
    hop1 = ["Nolan", "Bale", "Action", "Crime"]
    hop2 = ["Inception", "Prestige", "Batman Begins", "DiCaprio", "Drama"]

    # Concentric layout
    pos = {seed: (0, 0)}
    for i, n in enumerate(hop1):
        ang = np.pi / 2 + i * (2 * np.pi / len(hop1))
        pos[n] = (2.3 * np.cos(ang), 2.3 * np.sin(ang))
    for i, n in enumerate(hop2):
        ang = np.pi / 2 + 0.3 + i * (2 * np.pi / len(hop2))
        pos[n] = (4.4 * np.cos(ang), 4.4 * np.sin(ang))

    edges_h1 = [(seed, n) for n in hop1]
    edges_h2 = [
        ("Nolan", "Inception"),
        ("Nolan", "Prestige"),
        ("Nolan", "Batman Begins"),
        ("Bale", "Prestige"),
        ("Bale", "Batman Begins"),
        ("Action", "Batman Begins"),
        ("Inception", "DiCaprio"),
        ("Prestige", "Drama"),
    ]

    # Ripple circles
    for r, c, lbl in [(2.3, BLUE, "Hop 1"), (4.4, PURPLE, "Hop 2")]:
        circle = plt.Circle((0, 0), r, fill=False,
                            edgecolor=c, lw=1.2, linestyle="--", alpha=0.4)
        ax.add_patch(circle)
        ax.text(0, r + 0.25, lbl, fontsize=10, color=c,
                ha="center", fontweight="bold")

    # Edges
    for u, v in edges_h1:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", color=BLUE,
                                    lw=2.0, alpha=0.6))
    for u, v in edges_h2:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", color=PURPLE,
                                    lw=1.4, alpha=0.5))

    def draw_node(name, color, size=0.45):
        x, y = pos[name]
        circ = plt.Circle((x, y), size, fc=color, ec="white",
                          lw=1.8, zorder=4)
        ax.add_patch(circ)
        ax.text(x, y, name, fontsize=8, color="white",
                ha="center", va="center", fontweight="bold", zorder=5)

    draw_node(seed, ORANGE, 0.55)
    for n in hop1:
        draw_node(n, BLUE)
    for n in hop2:
        draw_node(n, PURPLE, 0.42)

    # Stone-in-pond annotation
    ax.text(0, -5.5,
            "User watched Dark Knight -> ripples reach Nolan/Bale (hop 1)\n"
            "-> reach Inception, Prestige, Batman Begins (hop 2)",
            fontsize=10, ha="center", color=DARK,
            bbox=dict(boxstyle="round,pad=0.5", fc=LIGHT, ec=GRAY))

    ax.legend(handles=[
        mpatches.Patch(color=ORANGE, label="Hop 0: user history"),
        mpatches.Patch(color=BLUE, label="Hop 1: direct neighbors"),
        mpatches.Patch(color=PURPLE, label="Hop 2: 2nd-order neighbors"),
    ], loc="upper right", fontsize=9, framealpha=0.95)

    ax.set_title("RippleNet: user preferences propagate outward like ripples",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-6.2, 5.3)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    save(fig, "fig3_ripplenet_propagation.png")


# ---------------------------------------------------------------------------
# fig4: KGAT attention over triples
# ---------------------------------------------------------------------------
def fig4_kgat_attention() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    center = "Inception"
    nbrs = [
        ("Nolan",        "directed_by",   0.42),
        ("DiCaprio",     "starred",       0.27),
        ("Sci-Fi",       "has_genre",     0.18),
        ("2010",         "released_in",   0.05),
        ("Warner Bros",  "produced_by",   0.08),
    ]

    pos = {center: (0, 0)}
    n = len(nbrs)
    for i, (e, _, _) in enumerate(nbrs):
        ang = np.pi / 2 - i * (2 * np.pi / n)
        pos[e] = (3.2 * np.cos(ang), 3.2 * np.sin(ang))

    # Edges with width and alpha proportional to attention
    for e, r, w in nbrs:
        x1, y1 = pos[center]
        x2, y2 = pos[e]
        lw = 0.6 + 8 * w
        alpha = 0.25 + 0.7 * w
        color = ORANGE if w > 0.2 else GRAY
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", color=color,
                                    lw=lw, alpha=alpha))
        # Relation label
        mx, my = 0.55 * (x1 + x2) / 1, 0.55 * (y1 + y2) / 1
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"{r}\n$\\alpha$={w:.2f}",
                fontsize=8, ha="center", va="center", color=DARK,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=color, lw=1.0, alpha=0.95))

    # Center node
    cc = plt.Circle(pos[center], 0.55, fc=BLUE, ec="white", lw=2, zorder=4)
    ax.add_patch(cc)
    ax.text(*pos[center], center, fontsize=10, color="white",
            ha="center", va="center", fontweight="bold", zorder=5)

    # Neighbor nodes
    for e, _, w in nbrs:
        x, y = pos[e]
        size = 0.42 + 0.25 * w
        c = PURPLE if w > 0.2 else GRAY
        circ = plt.Circle((x, y), size, fc=c, ec="white", lw=1.8, zorder=4)
        ax.add_patch(circ)
        ax.text(x, y, e, fontsize=8.5, color="white",
                ha="center", va="center", fontweight="bold", zorder=5)

    ax.text(0, -4.7,
            "Thicker edges = higher attention. The model learns that director and\n"
            "lead actor matter more than release year for this item.",
            fontsize=10, ha="center", color=DARK,
            bbox=dict(boxstyle="round,pad=0.5", fc=LIGHT, ec=GRAY))

    ax.set_title("KGAT: attention weights over the item's KG neighbors",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.5, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    save(fig, "fig4_kgat_attention.png")


# ---------------------------------------------------------------------------
# fig5: Four problems KG solves
# ---------------------------------------------------------------------------
def fig5_four_problems() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    panels = [
        (axes[0, 0], "Cold start", BLUE,
         "New item has 0 interactions",
         "...but its director, cast, genre\nlink it to known items"),
        (axes[0, 1], "Data sparsity", PURPLE,
         "Interaction matrix is 99%+ zeros",
         "KG fills gaps with semantic edges:\nshared director, genre, studio"),
        (axes[1, 0], "Explainability", GREEN,
         "'Users like you also liked X'",
         "'Recommended because Nolan also\ndirected Dark Knight, which you loved'"),
        (axes[1, 1], "Diversity", ORANGE,
         "Filter bubbles: more of the same",
         "Different relation paths surface\ndifferent kinds of recommendations"),
    ]

    for ax, title, color, problem, solution in panels:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Title bar
        title_box = FancyBboxPatch((0.3, 8.3), 9.4, 1.3,
                                    boxstyle="round,pad=0.05",
                                    fc=color, ec="white", lw=1.5)
        ax.add_patch(title_box)
        ax.text(5, 8.95, title, fontsize=14, color="white",
                ha="center", va="center", fontweight="bold")

        # Problem (red-tinted)
        ax.text(5, 6.5, "Problem", fontsize=10, ha="center",
                color="#dc2626", fontweight="bold")
        ax.text(5, 5.5, problem, fontsize=10.5, ha="center",
                color=DARK, va="center")

        # Arrow
        ax.annotate("", xy=(5, 3.6), xytext=(5, 4.6),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.2))

        # Solution
        ax.text(5, 2.9, "KG solution", fontsize=10, ha="center",
                color=color, fontweight="bold")
        ax.text(5, 1.6, solution, fontsize=10.5, ha="center",
                color=DARK, va="center")

    fig.suptitle("Four problems knowledge graphs solve in recommendation",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    save(fig, "fig5_four_problems.png")


# ---------------------------------------------------------------------------
# fig6: Dark Knight ego graph with reasoning paths
# ---------------------------------------------------------------------------
def fig6_dark_knight_example() -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    nodes = {
        "User":            (-4.5, 0.0, ORANGE, "user"),
        "Dark Knight":     (-2.0, 0.0, BLUE, "movie"),
        "Nolan":           (0.0, 1.6, PURPLE, "person"),
        "Bale":            (0.0, -1.6, PURPLE, "person"),
        "Action":          (0.0, 0.0, GREEN, "genre"),
        "Inception":       (2.5, 2.2, BLUE, "movie"),
        "Prestige":        (2.5, -1.4, BLUE, "movie"),
        "Batman Begins":   (4.5, -0.2, BLUE, "movie"),
    }

    edges = [
        ("User", "Dark Knight", "watched", "solid", DARK),
        ("Dark Knight", "Nolan", "directed_by", "solid", GRAY),
        ("Dark Knight", "Bale", "starred", "solid", GRAY),
        ("Dark Knight", "Action", "has_genre", "solid", GRAY),
        ("Nolan", "Inception", "directed", "dashed", BLUE),
        ("Nolan", "Prestige", "directed", "dashed", BLUE),
        ("Bale", "Prestige", "starred_in", "dashed", BLUE),
        ("Bale", "Batman Begins", "starred_in", "dashed", BLUE),
        ("Action", "Batman Begins", "is_genre_of", "dashed", BLUE),
    ]

    for u, v, lbl, ls, c in edges:
        x1, y1, *_ = nodes[u]
        x2, y2, *_ = nodes[v]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.6,
                                    linestyle=ls, alpha=0.75,
                                    shrinkA=22, shrinkB=22))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, lbl, fontsize=7.5, ha="center", va="center",
                color=DARK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.85))

    for n, (x, y, c, kind) in nodes.items():
        w, h = (1.3, 0.42) if kind != "user" else (1.0, 0.42)
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.04",
                             fc=c, ec="white", lw=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, n, fontsize=9, color="white",
                ha="center", va="center", fontweight="bold", zorder=4)

    # Reasoning path callout
    ax.text(0.5, -3.6,
            "Reasoning path: User -> watched -> Dark Knight -> directed_by -> Nolan\n"
            "-> directed -> Inception   ==> recommend Inception",
            fontsize=10, ha="center", color=DARK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec=BLUE))

    ax.legend(handles=[
        mpatches.Patch(color=DARK, label="Observed interaction"),
        mpatches.Patch(color=GRAY, label="Known KG fact"),
        mpatches.Patch(color=BLUE, label="Inferred path -> recommendation"),
    ], loc="upper right", fontsize=9, framealpha=0.95)

    ax.set_title("Reasoning over a knowledge graph: from one watch to recommendations",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-5.5, 5.8)
    ax.set_ylim(-4.5, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    save(fig, "fig6_dark_knight_example.png")


# ---------------------------------------------------------------------------
# fig7: With KG vs without KG comparison
# ---------------------------------------------------------------------------
def fig7_with_vs_without_kg() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2),
                              gridspec_kw={"width_ratios": [1, 1.05]})

    # Left: bar chart of metrics
    ax = axes[0]
    metrics = ["AUC", "Recall@20", "NDCG@20", "Cold-start\nRecall@20"]
    no_kg = [0.823, 0.456, 0.389, 0.121]
    with_kg = [0.892, 0.561, 0.485, 0.298]

    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w / 2, no_kg, w, label="Without KG (BPR)",
                color=GRAY, edgecolor="white", lw=1.2)
    b2 = ax.bar(x + w / 2, with_kg, w, label="With KG (KGAT)",
                color=BLUE, edgecolor="white", lw=1.2)

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center",
                    va="bottom", fontsize=8.5, color=DARK)

    # Lift annotations
    for i, (a, c) in enumerate(zip(no_kg, with_kg)):
        lift = (c - a) / a * 100
        ax.text(i, max(a, c) + 0.05, f"+{lift:.0f}%",
                ha="center", color=GREEN, fontweight="bold", fontsize=10)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Metric value")
    ax.set_title("Quantitative lift on MovieLens-1M",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: qualitative recommendation list comparison
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Qualitative: recommendations after watching The Dark Knight",
                 fontsize=12, fontweight="bold")

    # Without KG card
    box1 = FancyBboxPatch((0.2, 0.5), 4.6, 8.5,
                           boxstyle="round,pad=0.12",
                           fc="#f9fafb", ec=GRAY, lw=1.5)
    ax.add_patch(box1)
    ax.text(2.5, 8.5, "Without KG", fontsize=12, ha="center",
            color=GRAY, fontweight="bold")
    ax.text(2.5, 7.9, "(collaborative filtering only)",
            fontsize=8.5, ha="center", color=GRAY, style="italic")
    no_kg_recs = [
        ("Avengers: Endgame", "users who liked X"),
        ("Joker", "popular action"),
        ("Iron Man", "popular blockbuster"),
        ("Top Gun", "popular action"),
        ("[new Nolan film]", "no signal -> not shown"),
    ]
    for i, (title, why) in enumerate(no_kg_recs):
        y = 7.1 - i * 1.2
        color = "#dc2626" if "[new" in title else DARK
        ax.text(0.55, y, f"{i+1}.", fontsize=10, color=color, fontweight="bold")
        ax.text(1.0, y, title, fontsize=10, color=color)
        ax.text(1.0, y - 0.35, why, fontsize=8, color=GRAY, style="italic")

    # With KG card
    box2 = FancyBboxPatch((5.2, 0.5), 4.6, 8.5,
                           boxstyle="round,pad=0.12",
                           fc="#eff6ff", ec=BLUE, lw=1.5)
    ax.add_patch(box2)
    ax.text(7.5, 8.5, "With KG", fontsize=12, ha="center",
            color=BLUE, fontweight="bold")
    ax.text(7.5, 7.9, "(KGAT-style reasoning)",
            fontsize=8.5, ha="center", color=BLUE, style="italic")
    with_kg_recs = [
        ("Inception", "same director (Nolan)"),
        ("The Prestige", "Nolan + Bale"),
        ("Batman Begins", "same trilogy + cast"),
        ("Memento", "early Nolan, similar tone"),
        ("[new Nolan film]", "linked via Nolan -> shown!"),
    ]
    for i, (title, why) in enumerate(with_kg_recs):
        y = 7.1 - i * 1.2
        color = GREEN if "[new" in title else DARK
        ax.text(5.55, y, f"{i+1}.", fontsize=10, color=color, fontweight="bold")
        ax.text(6.0, y, title, fontsize=10, color=color, fontweight="bold")
        ax.text(6.0, y - 0.35, why, fontsize=8, color=BLUE, style="italic")

    plt.tight_layout()
    save(fig, "fig7_with_vs_without_kg.png")


def main() -> None:
    print("Generating figures for Part 8: Knowledge Graph...")
    fig1_knowledge_graph();           print("  fig1_knowledge_graph.png")
    fig2_transe_embedding();          print("  fig2_transe_embedding.png")
    fig3_ripplenet_propagation();     print("  fig3_ripplenet_propagation.png")
    fig4_kgat_attention();            print("  fig4_kgat_attention.png")
    fig5_four_problems();             print("  fig5_four_problems.png")
    fig6_dark_knight_example();       print("  fig6_dark_knight_example.png")
    fig7_with_vs_without_kg();        print("  fig7_with_vs_without_kg.png")
    print(f"Saved to:")
    for d in OUT_DIRS:
        print(f"  {d}")


if __name__ == "__main__":
    main()
