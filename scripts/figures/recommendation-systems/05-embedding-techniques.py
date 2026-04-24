"""
Figure generator for Recommendation Systems Part 05 — Embedding & Representation Learning.

Generates 6 production-quality figures and writes copies into BOTH the EN and ZH
asset folders so the same image is referenced from either language post.

Style contract:
  - matplotlib seaborn-v0_8-whitegrid
  - dpi = 150
  - palette: #2563eb (blue), #7c3aed (violet), #10b981 (emerald), #f59e0b (amber)

Run:
  python 05-embedding-techniques.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------- #
# Output configuration
# --------------------------------------------------------------------------- #

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/"
    "recommendation-systems/05-embedding-techniques"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/"
    "recommendation-systems/05-Embedding表示学习"
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
    """Save the figure into every output directory."""
    for d in OUT_DIRS:
        out = d / name
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name} -> {len(OUT_DIRS)} dirs")


# --------------------------------------------------------------------------- #
# Fig 1 — t-SNE embedding space colored by category
# --------------------------------------------------------------------------- #

def fig1_embedding_space() -> None:
    rng = np.random.default_rng(42)

    categories = ["Action", "Romance", "Sci-Fi", "Documentary"]
    colors = PALETTE
    centers = [(-3.5, 2.0), (3.0, 2.5), (-2.0, -3.0), (3.5, -2.5)]
    counts = [55, 50, 45, 40]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    for cat, color, (cx, cy), n in zip(categories, colors, centers, counts):
        # blob with a bit of anisotropy
        cov = rng.uniform(0.7, 1.3, size=(2,))
        x = rng.normal(cx, cov[0], n)
        y = rng.normal(cy, cov[1], n)
        ax.scatter(x, y, s=42, c=color, alpha=0.78,
                   edgecolors="white", linewidths=0.6, label=cat)

        # category centroid label
        ax.text(cx, cy + 0.05, cat,
                fontsize=10.5, fontweight="bold", color=DARK,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=color, lw=1.4, alpha=0.92))

    # Highlight a "query" item and its nearest neighbours
    qx, qy = -3.2, 2.3
    ax.scatter([qx], [qy], s=240, marker="*", c=AMBER,
               edgecolors=DARK, linewidths=1.4, zorder=5,
               label="Query item")
    # draw nearest-neighbour ring
    circle = plt.Circle((qx, qy), 1.6, fill=False,
                        ec=AMBER, lw=1.6, ls="--", alpha=0.85)
    ax.add_patch(circle)
    ax.annotate("k-NN region",
                xy=(qx + 1.6, qy), xytext=(qx + 2.6, qy + 1.4),
                fontsize=9.5, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1))

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title("Item Embedding Space (t-SNE projection, colored by category)",
                 pad=14)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    ax.set_xlim(-7, 7.5)
    ax.set_ylim(-6, 5.5)

    fig.tight_layout()
    _save(fig, "fig1_embedding_space.png")


# --------------------------------------------------------------------------- #
# Fig 2 — Item2Vec skip-gram architecture
# --------------------------------------------------------------------------- #

def fig2_item2vec_skipgram() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=10, fw="bold", tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc)

    def arrow(x1, y1, x2, y2, color=GREY, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="->", mutation_scale=14,
                                     color=color, lw=1.4, linestyle=ls))

    # Center item (input)
    box(0.4, 2.5, 1.4, 1.0, "Center\nitem  i_t", BLUE, BLUE, fs=10)

    # Embedding lookup
    box(2.4, 2.5, 1.7, 1.0, "Embedding\nLookup", VIOLET, VIOLET, fs=10)

    # Embedding vector
    box(4.7, 2.5, 1.5, 1.0, "e(i_t)  in R^d", GREEN, GREEN, fs=10)

    # Context outputs (positive)
    ctx_x, ctx_y = 7.4, 4.4
    box(ctx_x, ctx_y, 1.9, 0.7, "context  i_{t-2}", "#dbeafe", BLUE, fs=9, tc=DARK, fw="normal")
    box(ctx_x, ctx_y - 0.85, 1.9, 0.7, "context  i_{t-1}", "#dbeafe", BLUE, fs=9, tc=DARK, fw="normal")
    box(ctx_x, ctx_y - 1.7, 1.9, 0.7, "context  i_{t+1}", "#dbeafe", BLUE, fs=9, tc=DARK, fw="normal")
    box(ctx_x, ctx_y - 2.55, 1.9, 0.7, "context  i_{t+2}", "#dbeafe", BLUE, fs=9, tc=DARK, fw="normal")

    # Negatives
    box(7.4, 0.55, 1.9, 0.7, "K negatives  ~ P_n^{3/4}",
        "#fef3c7", AMBER, fs=9, tc=DARK, fw="bold")

    # Arrows: input -> lookup -> embedding
    arrow(1.8, 3.0, 2.4, 3.0)
    arrow(4.1, 3.0, 4.7, 3.0)

    # Embedding -> context (positive, solid green)
    for j in range(4):
        cy_box = 4.4 + 0.35 - j * 0.85
        arrow(6.2, 3.0, ctx_x, cy_box, color=GREEN)

    # Embedding -> negatives (dashed amber)
    arrow(6.2, 2.7, 7.4, 1.05, color=AMBER, ls="--")

    # Sequence strip on top
    seq_y = 5.2
    items = ["i_{t-2}", "i_{t-1}", "i_t", "i_{t+1}", "i_{t+2}"]
    seq_colors = ["#dbeafe", "#dbeafe", BLUE, "#dbeafe", "#dbeafe"]
    for k, (it, c) in enumerate(zip(items, seq_colors)):
        x = 1.6 + k * 1.0
        is_center = k == 2
        ax.add_patch(Rectangle((x, seq_y), 0.85, 0.5,
                               fc=c, ec=BLUE, lw=1.4))
        ax.text(x + 0.425, seq_y + 0.25, it,
                ha="center", va="center", fontsize=9,
                fontweight="bold" if is_center else "normal",
                color="white" if is_center else DARK)
    # window bracket
    ax.annotate("", xy=(1.6, seq_y - 0.15), xytext=(6.45, seq_y - 0.15),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
    ax.text(4.0, seq_y - 0.45, "context window  c = 2",
            ha="center", fontsize=9, color=GREY, style="italic")

    # Loss legend
    ax.text(5.0, 0.25,
            "Skip-gram loss:  log sigma( e . e+ )  +  sum log sigma( - e . e- )",
            ha="center", fontsize=10, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=GREY, lw=1))

    ax.set_title("Item2Vec — Skip-gram with Negative Sampling",
                 fontsize=13, color=DARK, pad=10)

    fig.tight_layout()
    _save(fig, "fig2_item2vec_skipgram.png")


# --------------------------------------------------------------------------- #
# Fig 3 — DSSM / Two-Tower architecture
# --------------------------------------------------------------------------- #

def fig3_two_tower() -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def block(x, y, w, h, text, fc, ec, fs=10, tc="white", fw="bold"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc)

    def varrow(x, y1, y2, color=GREY):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2),
                                     arrowstyle="->", mutation_scale=14,
                                     color=color, lw=1.5))

    # ---- User tower (left) ----
    user_x = 1.0
    block(user_x, 0.4, 2.6, 0.7, "User features\n(id, demo, history)",
          "#dbeafe", BLUE, fs=9.5, tc=DARK, fw="bold")
    block(user_x, 1.5, 2.6, 0.7, "Embedding layer", BLUE, BLUE, fs=10)
    block(user_x, 2.6, 2.6, 0.7, "FC 256 + ReLU + BN", BLUE, BLUE, fs=10)
    block(user_x, 3.7, 2.6, 0.7, "FC 128 + ReLU + BN", BLUE, BLUE, fs=10)
    block(user_x, 4.8, 2.6, 0.7, "FC d  ->  L2 normalize", BLUE, BLUE, fs=10)
    block(user_x, 5.95, 2.6, 0.6, "user vector  e_u  in R^d",
          "#eef2ff", BLUE, fs=10, tc=DARK)
    for y1, y2 in [(1.1, 1.5), (2.2, 2.6), (3.3, 3.7),
                   (4.4, 4.8), (5.5, 5.95)]:
        varrow(user_x + 1.3, y1, y2, color=BLUE)

    # ---- Item tower (right) ----
    item_x = 6.4
    block(item_x, 0.4, 2.6, 0.7, "Item features\n(id, category, content)",
          "#ede9fe", VIOLET, fs=9.5, tc=DARK, fw="bold")
    block(item_x, 1.5, 2.6, 0.7, "Embedding layer", VIOLET, VIOLET, fs=10)
    block(item_x, 2.6, 2.6, 0.7, "FC 256 + ReLU + BN", VIOLET, VIOLET, fs=10)
    block(item_x, 3.7, 2.6, 0.7, "FC 128 + ReLU + BN", VIOLET, VIOLET, fs=10)
    block(item_x, 4.8, 2.6, 0.7, "FC d  ->  L2 normalize", VIOLET, VIOLET, fs=10)
    block(item_x, 5.95, 2.6, 0.6, "item vector  e_i  in R^d",
          "#f5f3ff", VIOLET, fs=10, tc=DARK)
    for y1, y2 in [(1.1, 1.5), (2.2, 2.6), (3.3, 3.7),
                   (4.4, 4.8), (5.5, 5.95)]:
        varrow(item_x + 1.3, y1, y2, color=VIOLET)

    # ---- Similarity at the top ----
    block(3.85, 6.2, 2.3, 0.6,
          "score  =  cos(e_u, e_i)",
          GREEN, GREEN, fs=11)

    # arrows from each tower head to similarity box
    ax.add_patch(FancyArrowPatch((user_x + 2.6, 6.25),
                                 (3.85, 6.5),
                                 arrowstyle="->", mutation_scale=14,
                                 color=GREEN, lw=1.6))
    ax.add_patch(FancyArrowPatch((item_x, 6.25),
                                 (6.15, 6.5),
                                 arrowstyle="->", mutation_scale=14,
                                 color=GREEN, lw=1.6))

    # serving annotation
    ax.text(5.0, 0.0,
            "Inference: pre-compute all e_i offline -> online ANN search on e_u",
            ha="center", fontsize=9.5, style="italic", color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=GREY, lw=1))

    # tower headers
    ax.text(user_x + 1.3, 6.85, "USER TOWER",
            ha="center", fontsize=10, color=BLUE, fontweight="bold")
    ax.text(item_x + 1.3, 6.85, "ITEM TOWER",
            ha="center", fontsize=10, color=VIOLET, fontweight="bold")

    ax.set_title("Two-Tower (DSSM) Architecture",
                 fontsize=13, color=DARK, pad=8)

    fig.tight_layout()
    _save(fig, "fig3_two_tower.png")


# --------------------------------------------------------------------------- #
# Fig 4 — Random walk on user-item graph (Node2Vec)
# --------------------------------------------------------------------------- #

def fig4_random_walk() -> None:
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis("off")

    # Node positions (mix of users U and items I)
    nodes = {
        "U1": (1.0, 5.0), "U2": (3.0, 5.5), "U3": (5.0, 5.0),
        "U4": (7.0, 5.5), "U5": (9.0, 5.0),
        "I1": (1.5, 2.5), "I2": (3.5, 2.0), "I3": (5.0, 2.8),
        "I4": (6.5, 2.0), "I5": (8.5, 2.5),
        "I6": (2.0, 0.5), "I7": (4.0, 0.4), "I8": (6.5, 0.4),
        "I9": (8.5, 0.6),
    }

    edges = [
        ("U1", "I1"), ("U1", "I2"), ("U1", "I6"),
        ("U2", "I1"), ("U2", "I2"), ("U2", "I3"), ("U2", "I7"),
        ("U3", "I3"), ("U3", "I4"), ("U3", "I7"),
        ("U4", "I3"), ("U4", "I4"), ("U4", "I5"), ("U4", "I8"),
        ("U5", "I4"), ("U5", "I5"), ("U5", "I9"),
        ("I2", "I3"), ("I3", "I4"), ("I7", "I8"),
    ]

    # draw edges
    for a, b in edges:
        x1, y1 = nodes[a]; x2, y2 = nodes[b]
        ax.plot([x1, x2], [y1, y2], color="#cbd5e1", lw=1.2, zorder=1)

    # nodes
    for name, (x, y) in nodes.items():
        is_user = name.startswith("U")
        c = BLUE if is_user else VIOLET
        ax.scatter(x, y, s=520, c=c, edgecolors="white", lw=1.5, zorder=3)
        ax.text(x, y, name, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=4)

    # The walk: U2 -> I3 -> U4 -> I8 -> I7 -> U3 -> I4
    walk = ["U2", "I3", "U4", "I8", "I7", "U3", "I4"]
    walk_color = AMBER
    for k in range(len(walk) - 1):
        x1, y1 = nodes[walk[k]]
        x2, y2 = nodes[walk[k + 1]]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="->",
                                     mutation_scale=18,
                                     color=walk_color, lw=2.6,
                                     connectionstyle="arc3,rad=0.12",
                                     zorder=2))
        # step label
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.18, f"{k+1}",
                fontsize=9, fontweight="bold", color=walk_color,
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.16",
                          fc="white", ec=walk_color, lw=1.2))

    # legend
    legend_handles = [
        mpatches.Patch(color=BLUE,   label="User node"),
        mpatches.Patch(color=VIOLET, label="Item node"),
        mpatches.Patch(color=AMBER,  label="Biased random walk"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              framealpha=0.95, fontsize=9)

    # caption box
    ax.text(0.05, 0.05,
            "Walk: U2 -> I3 -> U4 -> I8 -> I7 -> U3 -> I4    "
            "(parameters p, q control return vs. exploration)",
            transform=ax.transAxes,
            fontsize=9, color=DARK, style="italic",
            bbox=dict(boxstyle="round,pad=0.35", fc=LIGHT, ec=GREY, lw=1))

    ax.set_title("Node2Vec — Biased Random Walk on a User–Item Graph",
                 fontsize=13, color=DARK, pad=8)

    fig.tight_layout()
    _save(fig, "fig4_random_walk.png")


# --------------------------------------------------------------------------- #
# Fig 5 — ANN search illustration
# --------------------------------------------------------------------------- #

def fig5_ann_search() -> None:
    rng = np.random.default_rng(11)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))

    # ---- Left: brute-force vs ANN, conceptual schematic ----
    ax = axes[0]
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.axis("off")

    # generate point cloud and cluster centers
    centers = np.array([(-3, 2.5), (3.0, 2.5), (-2.5, -2.5),
                        (2.5, -3.0), (0.0, 0.5)])
    pts = []
    for cx, cy in centers:
        pts.append(rng.normal([cx, cy], 0.9, size=(38, 2)))
    pts = np.vstack(pts)

    # cluster Voronoi-ish circles
    for cx, cy in centers:
        ax.add_patch(plt.Circle((cx, cy), 1.9, fc=LIGHT,
                                ec=GREY, lw=1.0, ls="--", alpha=0.6))
        ax.scatter([cx], [cy], s=80, c=AMBER, marker="X",
                   edgecolors=DARK, lw=1, zorder=4)

    # all candidate items
    ax.scatter(pts[:, 0], pts[:, 1], s=22, c=BLUE, alpha=0.55,
               edgecolors="white", linewidths=0.4, zorder=2)

    # query
    qx, qy = -2.7, 2.3
    ax.scatter([qx], [qy], s=240, marker="*", c=GREEN,
               edgecolors=DARK, lw=1.4, zorder=6)
    ax.text(qx + 0.3, qy + 0.4, "query  e_u",
            fontsize=10, fontweight="bold", color=DARK)

    # selected probed cluster (the one closest to query)
    ax.add_patch(plt.Circle((-3, 2.5), 1.9, fc="#dcfce7",
                            ec=GREEN, lw=1.8, alpha=0.55, zorder=1))
    # k-NN inside that cluster
    cluster_mask = ((pts[:, 0] + 3) ** 2 + (pts[:, 1] - 2.5) ** 2) < 1.9 ** 2
    cand = pts[cluster_mask]
    dists = np.linalg.norm(cand - np.array([qx, qy]), axis=1)
    knn = cand[np.argsort(dists)[:5]]
    ax.scatter(knn[:, 0], knn[:, 1], s=80, c=GREEN,
               edgecolors=DARK, lw=1, zorder=5)
    for p in knn:
        ax.plot([qx, p[0]], [qy, p[1]],
                color=GREEN, lw=1.0, alpha=0.75, zorder=3)

    ax.text(-5.7, 5.4, "IVF: probe nearest cluster centroid, search inside",
            fontsize=9.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=1))

    legend_handles = [
        mpatches.Patch(color=BLUE,  label="Item embedding"),
        mpatches.Patch(color=AMBER, label="Cluster centroid"),
        mpatches.Patch(color=GREEN, label="Top-K result"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=9, framealpha=0.95)
    ax.set_title("ANN Search — IVF Index", fontsize=12, color=DARK)

    # ---- Right: latency / recall trade-off ----
    ax2 = axes[1]
    methods = ["Flat\n(exact)", "IVF", "HNSW", "Annoy", "IVFPQ"]
    latency_ms = [120, 9, 3, 6, 4]
    recall_pct = [100, 96, 98, 92, 90]

    bar_color = [GREY, BLUE, GREEN, AMBER, VIOLET]
    bars = ax2.bar(methods, latency_ms, color=bar_color,
                   edgecolor="white", linewidth=1.4, alpha=0.92)
    ax2.set_ylabel("Query latency (ms, lower is better)", color=DARK)
    ax2.set_ylim(0, max(latency_ms) * 1.25)
    ax2.set_axisbelow(True)
    ax2.grid(axis="y", alpha=0.5)

    # recall as line on twin axis
    ax2b = ax2.twinx()
    ax2b.plot(methods, recall_pct, "o-", color=DARK, lw=1.6,
              markersize=7, markerfacecolor="white",
              markeredgewidth=1.6, zorder=5)
    ax2b.set_ylabel("Recall@10 (%, higher is better)", color=DARK)
    ax2b.set_ylim(85, 102)
    ax2b.grid(False)

    # bar labels
    for bar, v in zip(bars, latency_ms):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 3,
                 f"{v} ms", ha="center", fontsize=9,
                 color=DARK, fontweight="bold")
    # recall labels
    for x, v in enumerate(recall_pct):
        ax2b.text(x, v + 0.6, f"{v}%", ha="center",
                  fontsize=8.5, color=DARK)

    ax2.set_title("Latency vs. Recall Trade-off (1 M items, d=128)",
                  fontsize=12, color=DARK)

    fig.suptitle("Approximate Nearest Neighbour Search",
                 fontsize=13, color=DARK, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_ann_search.png")


# --------------------------------------------------------------------------- #
# Fig 6 — Embedding similarity heatmap
# --------------------------------------------------------------------------- #

def fig6_similarity_heatmap() -> None:
    rng = np.random.default_rng(2026)

    # Build a synthetic block-similarity matrix for 4 categories
    items = [
        ("Inception",       "Sci-Fi"),
        ("Interstellar",    "Sci-Fi"),
        ("The Matrix",      "Sci-Fi"),
        ("Dark Knight",     "Action"),
        ("John Wick",       "Action"),
        ("Mad Max",         "Action"),
        ("La La Land",      "Romance"),
        ("Notebook",        "Romance"),
        ("Pride & Prej.",   "Romance"),
        ("Planet Earth",    "Doc"),
        ("Cosmos",          "Doc"),
        ("Free Solo",       "Doc"),
    ]
    cats = [c for _, c in items]
    n = len(items)

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim[i, j] = 1.0
            elif cats[i] == cats[j]:
                sim[i, j] = rng.uniform(0.55, 0.85)
            else:
                sim[i, j] = rng.uniform(-0.15, 0.30)
    # symmetrize
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 1.0)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    # custom blue-white-violet diverging colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "bwv", ["#bfdbfe", "#ffffff", VIOLET], N=256
    )

    im = ax.imshow(sim, cmap=cmap, vmin=-0.2, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([name for name, _ in items],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([name for name, _ in items], fontsize=9)

    # numeric overlay
    for i in range(n):
        for j in range(n):
            v = sim[i, j]
            tc = "white" if v > 0.55 else DARK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=tc)

    # category block borders
    cat_changes = [k for k in range(1, n) if cats[k] != cats[k-1]]
    boundaries = [0] + cat_changes + [n]
    for k in range(len(boundaries) - 1):
        a, b = boundaries[k], boundaries[k+1]
        ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a,
                               fill=False, ec=AMBER, lw=2.0))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", fontsize=10, color=DARK)

    ax.set_title("Item Embedding Similarity Heatmap (block diagonal = same category)",
                 fontsize=12.5, color=DARK, pad=10)

    fig.tight_layout()
    _save(fig, "fig6_similarity_heatmap.png")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    print("Generating figures for Recommendation Systems Part 05 ...")
    fig1_embedding_space()
    fig2_item2vec_skipgram()
    fig3_two_tower()
    fig4_random_walk()
    fig5_ann_search()
    fig6_similarity_heatmap()
    print("All figures written to:")
    for d in OUT_DIRS:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
