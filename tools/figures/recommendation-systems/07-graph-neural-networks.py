"""
Figures for Recommendation Systems Part 7: Graph Neural Networks.

Generates 7 figures used by both EN and ZH posts. Each figure is rendered
once and saved into both asset folders so the two language versions stay
in sync.

Run:
    python 07-graph-neural-networks.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

BLUE = COLORS["primary"]      # users / primary
PURPLE = COLORS["accent"]    # items / secondary
GREEN = COLORS["success"]     # positive / aggregation
ORANGE = COLORS["warning"]    # highlight / target
GREY = COLORS["muted"]
DARK = COLORS["text"]
LIGHT = COLORS["bg"]

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/"
    "recommendation-systems/07-graph-neural-networks"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/"
    "recommendation-systems/07-图神经网络与社交推荐"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure into both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (EN_DIR, ZH_DIR):
        path = folder / name
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: User-item bipartite graph
# ---------------------------------------------------------------------------


def fig1_bipartite_graph() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    users = ["Alice", "Bob", "Carol", "Dave"]
    items = ["Inception", "Interstellar", "Matrix", "Blade Runner", "Dune"]

    edges = [
        ("Alice", "Inception"),
        ("Alice", "Interstellar"),
        ("Alice", "Matrix"),
        ("Bob", "Interstellar"),
        ("Bob", "Matrix"),
        ("Bob", "Blade Runner"),
        ("Carol", "Matrix"),
        ("Carol", "Blade Runner"),
        ("Carol", "Dune"),
        ("Dave", "Inception"),
        ("Dave", "Dune"),
    ]

    pos = {}
    for i, u in enumerate(users):
        pos[u] = (0.0, (len(users) - 1) / 2 - i)
    for j, it in enumerate(items):
        pos[it] = (4.0, (len(items) - 1) / 2 - j)

    # Edges
    for u, it in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[it]
        ax.plot([x0, x1], [y0, y1], color=GREY, lw=1.4, alpha=0.55, zorder=1)

    # Highlight Alice <-> Bob shared neighbours
    shared = [("Alice", "Interstellar"), ("Alice", "Matrix"),
              ("Bob", "Interstellar"), ("Bob", "Matrix")]
    for u, it in shared:
        x0, y0 = pos[u]
        x1, y1 = pos[it]
        ax.plot([x0, x1], [y0, y1], color=ORANGE, lw=2.4, alpha=0.9, zorder=2)

    # User nodes
    for u in users:
        x, y = pos[u]
        ax.scatter(x, y, s=2200, color=BLUE, edgecolor="white",
                   linewidth=2.5, zorder=3)
        ax.text(x, y, u, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=4)

    # Item nodes
    for it in items:
        x, y = pos[it]
        ax.scatter(x, y, s=2400, color=PURPLE, edgecolor="white",
                   linewidth=2.5, marker="s", zorder=3)
        ax.text(x, y, it, ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=4)

    # Labels
    ax.text(0, max(p[1] for p in pos.values()) + 0.9, "Users",
            ha="center", fontsize=14, fontweight="bold", color=BLUE)
    ax.text(4, max(p[1] for p in pos.values()) + 0.9, "Items",
            ha="center", fontsize=14, fontweight="bold", color=PURPLE)

    ax.text(2, -3.2,
            "Edge = interaction (click / watch / rate)\n"
            "Highlighted edges: Alice and Bob share two items "
            "→ collaborative signal",
            ha="center", fontsize=10.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT,
                      edgecolor=GREY, linewidth=1))

    ax.set_xlim(-1.2, 5.2)
    ax.set_ylim(-3.8, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("User-Item Bipartite Graph", fontsize=15,
                 fontweight="bold", pad=16, color=DARK)

    fig.tight_layout()
    save(fig, "fig1_bipartite_graph.png")


# ---------------------------------------------------------------------------
# Figure 2: Message passing
# ---------------------------------------------------------------------------


def fig2_message_passing() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # A 1-hop neighbourhood
    G = nx.Graph()
    centre = "v"
    neighbours = ["u1", "u2", "u3", "u4"]
    G.add_node(centre)
    G.add_nodes_from(neighbours)
    for n in neighbours:
        G.add_edge(centre, n)

    pos = {centre: (0, 0)}
    for i, n in enumerate(neighbours):
        angle = math.pi / 2 + i * (2 * math.pi / len(neighbours))
        pos[n] = (1.4 * math.cos(angle), 1.4 * math.sin(angle))

    titles = ["1. Message", "2. Aggregate", "3. Update"]
    descriptions = [
        "Each neighbour computes\n"
        r"a message $m_u = f(h_u)$",
        "Target node combines messages\n"
        r"$\bar m = \mathrm{AGG}(\{m_u\})$",
        "Target updates its state\n"
        r"$h_v' = \mathrm{UPD}(h_v, \bar m)$",
    ]

    for ax, title, desc, step in zip(axes, titles, descriptions, range(3)):
        # Edges
        for n in neighbours:
            x0, y0 = pos[n]
            ax.plot([x0, 0], [y0, 0], color=GREY, lw=1.5, alpha=0.5, zorder=1)

        # Step-specific arrows
        if step == 0:
            for n in neighbours:
                x0, y0 = pos[n]
                arrow = FancyArrowPatch(
                    (x0 * 0.6, y0 * 0.6), (x0 * 0.25, y0 * 0.25),
                    arrowstyle="->", mutation_scale=18,
                    color=GREEN, lw=2.2, zorder=4,
                )
                ax.add_patch(arrow)
        elif step == 1:
            ax.scatter(0, 0, s=5500, color=GREEN, alpha=0.18, zorder=2)
            ax.text(0, -0.55, "AGG", ha="center", va="center",
                    fontsize=10.5, color=GREEN, fontweight="bold", zorder=5)
        elif step == 2:
            ax.scatter(0, 0, s=5500, color=ORANGE, alpha=0.22, zorder=2)
            ax.text(0, -0.55, "UPDATE", ha="center", va="center",
                    fontsize=10.5, color=ORANGE, fontweight="bold", zorder=5)

        # Neighbour nodes
        for n in neighbours:
            x, y = pos[n]
            ax.scatter(x, y, s=900, color=BLUE, edgecolor="white",
                       linewidth=2, zorder=3)
            ax.text(x, y, n, ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold", zorder=4)

        # Centre node
        ax.scatter(0, 0, s=1300,
                   color=ORANGE if step == 2 else PURPLE,
                   edgecolor="white", linewidth=2.5, zorder=5)
        ax.text(0, 0, "v", ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", zorder=6)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2.2, 2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=DARK, pad=10)
        ax.text(0, -1.85, desc, ha="center", va="center",
                fontsize=10, color=DARK)

    fig.suptitle("One Round of Message Passing",
                 fontsize=15, fontweight="bold", color=DARK, y=1.03)
    fig.tight_layout()
    save(fig, "fig2_message_passing.png")


# ---------------------------------------------------------------------------
# Figure 3: GCN vs GAT vs GraphSAGE
# ---------------------------------------------------------------------------


def fig3_gcn_gat_sage() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5))

    centre = (0, 0)
    n_neighbours = 6
    angles = [math.pi / 2 + k * (2 * math.pi / n_neighbours)
              for k in range(n_neighbours)]
    nb_pos = [(1.4 * math.cos(a), 1.4 * math.sin(a)) for a in angles]

    def draw_base(ax, weights, sampled=None, title="", subtitle=""):
        # Edges
        for i, (x, y) in enumerate(nb_pos):
            if sampled is not None and i not in sampled:
                ax.plot([x, 0], [y, 0], color=GREY, lw=1, alpha=0.25,
                        linestyle="--", zorder=1)
            else:
                w = weights[i]
                ax.plot([x, 0], [y, 0], color=BLUE,
                        lw=0.8 + 5 * w, alpha=0.6, zorder=1)

        # Neighbour nodes
        for i, (x, y) in enumerate(nb_pos):
            faded = sampled is not None and i not in sampled
            color = GREY if faded else BLUE
            alpha = 0.4 if faded else 1.0
            ax.scatter(x, y, s=620, color=color, edgecolor="white",
                       linewidth=1.8, zorder=3, alpha=alpha)
            if not faded:
                ax.text(x * 1.45, y * 1.45, f"{weights[i]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color=DARK, fontweight="bold")

        # Centre
        ax.scatter(*centre, s=1100, color=ORANGE, edgecolor="white",
                   linewidth=2.5, zorder=5)
        ax.text(0, 0, "v", ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", zorder=6)

        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 2.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13.5, fontweight="bold", color=DARK)
        ax.text(0, -1.85, subtitle, ha="center", fontsize=10, color=DARK)

    # GCN: degree-normalised, all neighbours equal here
    gcn_w = [1 / n_neighbours] * n_neighbours
    draw_base(axes[0], gcn_w,
              title="GCN", subtitle="Symmetric normalisation\nall neighbours treated equally")

    # GAT: learned attention weights
    gat_w = np.array([0.35, 0.05, 0.20, 0.10, 0.25, 0.05])
    gat_w = gat_w / gat_w.sum()
    draw_base(axes[1], gat_w.tolist(),
              title="GAT", subtitle="Learned attention\nimportant neighbours weighted higher")

    # GraphSAGE: sample 3 of 6, equal weight on sampled
    sampled = [0, 2, 4]
    sage_w = [1 / len(sampled) if i in sampled else 0
              for i in range(n_neighbours)]
    draw_base(axes[2], sage_w, sampled=sampled,
              title="GraphSAGE", subtitle="Sample fixed-size neighbourhood\nthen aggregate")

    fig.suptitle("Three Aggregation Strategies",
                 fontsize=15, fontweight="bold", color=DARK, y=1.0)
    fig.tight_layout()
    save(fig, "fig3_gcn_gat_sage.png")


# ---------------------------------------------------------------------------
# Figure 4: LightGCN simplification
# ---------------------------------------------------------------------------


def fig4_lightgcn_simplify() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    components = [
        ("Self-loops", True, False),
        ("Linear transform W", True, False),
        ("Nonlinearity ReLU", True, False),
        ("Neighbour aggregation", True, True),
        ("Layer combination", False, True),
    ]

    def draw_panel(ax, keep_func, title, subtitle):
        y = 0.85
        ax.text(0.5, 0.96, title, ha="center", fontsize=14,
                fontweight="bold", color=DARK,
                transform=ax.transAxes)
        ax.text(0.5, 0.91, subtitle, ha="center", fontsize=10,
                color=GREY, transform=ax.transAxes, style="italic")
        for name, in_gcn, in_light in components:
            keep = keep_func(in_gcn, in_light)
            color = GREEN if keep else "#fee2e2"
            text_color = "white" if keep else "#b91c1c"
            edge = GREEN if keep else "#fca5a5"
            box = FancyBboxPatch((0.08, y - 0.05), 0.84, 0.08,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor=edge,
                                 linewidth=1.5,
                                 transform=ax.transAxes)
            ax.add_patch(box)
            mark = "kept" if keep else "removed"
            ax.text(0.12, y - 0.01, name, fontsize=11.5,
                    color=text_color, fontweight="bold",
                    va="center", transform=ax.transAxes)
            ax.text(0.88, y - 0.01, mark, fontsize=10.5,
                    color=text_color, va="center", ha="right",
                    transform=ax.transAxes, style="italic")
            y -= 0.13
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    draw_panel(
        axes[0],
        keep_func=lambda gcn, light: gcn,
        title="Standard GCN",
        subtitle="Many moving parts",
    )
    draw_panel(
        axes[1],
        keep_func=lambda gcn, light: light,
        title="LightGCN",
        subtitle="Strip away everything but aggregation",
    )

    # Arrow between panels
    fig.text(0.51, 0.5, "→", ha="center", va="center",
             fontsize=42, color=ORANGE, fontweight="bold")

    fig.suptitle("LightGCN: Simplicity Wins for Collaborative Filtering",
                 fontsize=15, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_lightgcn_simplify.png")


# ---------------------------------------------------------------------------
# Figure 5: Multi-hop neighbour aggregation
# ---------------------------------------------------------------------------


def fig5_multihop() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    # Centre user
    centre = "U"
    pos = {centre: (0, 0)}

    # 1-hop: items
    hop1 = ["i1", "i2", "i3"]
    for k, n in enumerate(hop1):
        angle = math.pi / 2 + (k - 1) * 0.55
        pos[n] = (2.3 * math.cos(angle), 2.3 * math.sin(angle))

    # 2-hop: other users
    hop2 = ["u_a", "u_b", "u_c", "u_d", "u_e"]
    for k, n in enumerate(hop2):
        angle = math.pi / 2 + (k - 2) * 0.45
        pos[n] = (4.4 * math.cos(angle), 4.4 * math.sin(angle))

    # 3-hop: candidate items
    hop3 = ["j1", "j2", "j3", "j4"]
    for k, n in enumerate(hop3):
        angle = math.pi / 2 + (k - 1.5) * 0.4
        pos[n] = (6.4 * math.cos(angle), 6.4 * math.sin(angle))

    edges = [
        ("U", "i1"), ("U", "i2"), ("U", "i3"),
        ("i1", "u_a"), ("i1", "u_b"),
        ("i2", "u_b"), ("i2", "u_c"),
        ("i3", "u_d"), ("i3", "u_e"),
        ("u_a", "j1"), ("u_b", "j2"), ("u_c", "j2"),
        ("u_d", "j3"), ("u_e", "j4"),
    ]

    for a, b in edges:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        ax.plot([x0, x1], [y0, y1], color=GREY, lw=1.2, alpha=0.45, zorder=1)

    # Hop rings
    for radius, color, label in [
        (2.3, BLUE, "1-hop"),
        (4.4, GREEN, "2-hop"),
        (6.4, PURPLE, "3-hop"),
    ]:
        circle = plt.Circle((0, 0), radius, fill=False,
                            color=color, linestyle="--",
                            linewidth=1.2, alpha=0.45, zorder=0)
        ax.add_patch(circle)
        ax.text(radius + 0.15, 0.3, label, color=color,
                fontsize=10.5, fontweight="bold")

    # Draw nodes
    def node(name, color, marker="o", size=900):
        x, y = pos[name]
        ax.scatter(x, y, s=size, color=color, edgecolor="white",
                   linewidth=2, marker=marker, zorder=3)
        ax.text(x, y, name, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4)

    node("U", ORANGE, size=1300)
    for n in hop1:
        node(n, PURPLE, marker="s")
    for n in hop2:
        node(n, BLUE)
    for n in hop3:
        node(n, PURPLE, marker="s")

    ax.text(0, -7.6,
            "Layer 1: target user collects features from items they touched\n"
            "Layer 2: signal reaches similar users (collaborative filtering)\n"
            "Layer 3: signal reaches items those similar users liked  →  candidate recommendations",
            ha="center", fontsize=10.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT,
                      edgecolor=GREY))

    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-8.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Multi-hop Neighbour Aggregation",
                 fontsize=15, fontweight="bold", color=DARK, pad=10)
    fig.tight_layout()
    save(fig, "fig5_multihop.png")


# ---------------------------------------------------------------------------
# Figure 6: Mini-batch sampling
# ---------------------------------------------------------------------------


def fig6_minibatch_sampling() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    # Build a random graph for both subplots
    rng = np.random.default_rng(7)
    n = 60
    G = nx.random_geometric_graph(n, radius=0.22, seed=11)
    pos = nx.spring_layout(G, seed=11)

    # Left: full graph
    ax = axes[0]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GREY,
                           width=0.8, alpha=0.35)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=BLUE,
                           node_size=110, edgecolors="white", linewidths=1)
    ax.set_title("Full graph (does not fit in memory at billion scale)",
                 fontsize=12.5, fontweight="bold", color=DARK)
    ax.axis("off")

    # Right: sampled subgraph
    ax = axes[1]
    target = 25  # target node
    # Two-hop neighbour sampling
    nb1 = list(G.neighbors(target))
    sampled1 = list(rng.choice(nb1, size=min(4, len(nb1)),
                               replace=False)) if nb1 else []
    sampled2 = []
    for u in sampled1:
        nb2 = [v for v in G.neighbors(u) if v != target]
        if nb2:
            sampled2.extend(rng.choice(nb2, size=min(2, len(nb2)),
                                        replace=False).tolist())

    keep_nodes = set([target] + sampled1 + sampled2)
    keep_edges = []
    for u in sampled1:
        keep_edges.append((target, u))
    for u in sampled1:
        for v in sampled2:
            if G.has_edge(u, v):
                keep_edges.append((u, v))

    # Draw faded full graph
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GREY,
                           width=0.6, alpha=0.12)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=GREY,
                           node_size=70, edgecolors="white",
                           linewidths=0.8, alpha=0.25)

    # Draw sampled subgraph on top
    nx.draw_networkx_edges(G, pos, edgelist=keep_edges, ax=ax,
                           edge_color=BLUE, width=1.8, alpha=0.85)
    sampled1_set = set(sampled1)
    sampled2_set = set(sampled2) - sampled1_set - {target}
    nx.draw_networkx_nodes(G, pos, nodelist=list(sampled1_set), ax=ax,
                           node_color=BLUE, node_size=200,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=list(sampled2_set), ax=ax,
                           node_color=GREEN, node_size=160,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[target], ax=ax,
                           node_color=ORANGE, node_size=380,
                           edgecolors="white", linewidths=2.2)

    ax.set_title("Neighbour sampling: target + 4 hop-1 + ≤2 hop-2 each",
                 fontsize=12.5, fontweight="bold", color=DARK)
    ax.axis("off")

    legend = [
        mpatches.Patch(color=ORANGE, label="Target node"),
        mpatches.Patch(color=BLUE, label="Hop-1 sampled neighbour"),
        mpatches.Patch(color=GREEN, label="Hop-2 sampled neighbour"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True,
              fontsize=9.5)

    fig.suptitle("Mini-batch Neighbour Sampling for Scalable GNN Training",
                 fontsize=15, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_minibatch_sampling.png")


# ---------------------------------------------------------------------------
# Figure 7: Cold-start handling
# ---------------------------------------------------------------------------


def fig7_cold_start() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    # Build small bipartite graph for both panels
    users = [f"u{i}" for i in range(5)]
    items = [f"i{j}" for j in range(5)]
    edges = [
        ("u0", "i0"), ("u0", "i1"),
        ("u1", "i0"), ("u1", "i2"),
        ("u2", "i1"), ("u2", "i2"), ("u2", "i3"),
        ("u3", "i3"), ("u3", "i4"),
        ("u4", "i4"), ("u4", "i2"),
    ]

    pos = {}
    for i, u in enumerate(users):
        pos[u] = (0.0, (len(users) - 1) / 2 - i)
    for j, it in enumerate(items):
        pos[it] = (3.0, (len(items) - 1) / 2 - j)

    new_user = "u_new"
    pos[new_user] = (0.0, -3.5)

    def draw_base(ax, with_new, title, hint):
        # Edges
        for u, it in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[it]
            ax.plot([x0, x1], [y0, y1], color=GREY, lw=1.1, alpha=0.45,
                    zorder=1)
        # Existing users / items
        for u in users:
            x, y = pos[u]
            ax.scatter(x, y, s=750, color=BLUE, edgecolor="white",
                       linewidth=2, zorder=3)
            ax.text(x, y, u, ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")
        for it in items:
            x, y = pos[it]
            ax.scatter(x, y, s=820, color=PURPLE, edgecolor="white",
                       linewidth=2, marker="s", zorder=3)
            ax.text(x, y, it, ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")

        # Cold-start user
        x, y = pos[new_user]
        ax.scatter(x, y, s=900, color=ORANGE, edgecolor="white",
                   linewidth=2.5, zorder=4)
        ax.text(x, y, "new", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")

        if with_new:
            # One interaction
            x1, y1 = pos["i2"]
            ax.plot([x, x1], [y, y1], color=ORANGE, lw=2.4,
                    alpha=0.9, zorder=2)
            # Show GraphSAGE-style aggregation arrow
            ax.annotate("aggregate\nneighbours of i2",
                        xy=(1.6, -1.6), xytext=(1.4, -3.4),
                        fontsize=9.5, color=GREEN,
                        arrowprops=dict(arrowstyle="->", color=GREEN,
                                        lw=1.8))

        ax.text(1.5, -4.6, hint, ha="center", fontsize=10, color=DARK,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT,
                          edgecolor=GREY))
        ax.set_xlim(-1, 4.2)
        ax.set_ylim(-5.2, 3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", color=DARK)

    draw_base(axes[0], with_new=False,
              title="Matrix factorisation",
              hint="No embedding for the new user — cold start")
    draw_base(axes[1], with_new=True,
              title="GraphSAGE-style induction",
              hint="One interaction → aggregate neighbours\n→ embedding on the fly")

    fig.suptitle("Cold-Start Users: Why Graph Models Help",
                 fontsize=15, fontweight="bold", color=DARK, y=1.0)
    fig.tight_layout()
    save(fig, "fig7_cold_start.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating figures for: Recommendation Systems Part 7 (GNN)")
    fig1_bipartite_graph()
    fig2_message_passing()
    fig3_gcn_gat_sage()
    fig4_lightgcn_simplify()
    fig5_multihop()
    fig6_minibatch_sampling()
    fig7_cold_start()
    print("Done.")


if __name__ == "__main__":
    main()
