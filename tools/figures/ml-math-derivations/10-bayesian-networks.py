"""
Figure generation script for ML Math Derivations Part 10:
Semi-Naive Bayes and Bayesian Networks.

Generates 7 didactic figures shared by the EN and ZH articles. Each figure is
designed to expose one specific idea from probabilistic graphical models so
the reader can ground the math in an image.

Figures:
    fig1_dag                       Toy DAG (Sprinkler-Rain-Wet-Slippery) with
                                   the joint factorisation written below.
    fig2_d_separation              The three canonical d-separation patterns
                                   (chain, fork, collider) with the activation
                                   rule under conditioning shown explicitly.
    fig3_nb_vs_tan                 Side-by-side comparison: Naive Bayes star
                                   graph vs Tree-Augmented Naive Bayes (TAN).
    fig4_aode                      AODE = arithmetic mean of d SPODE
                                   sub-models, each picking a super-parent.
    fig5_variable_elimination      Variable elimination on a small chain:
                                   factors collapsing step by step into the
                                   posterior over the query variable.
    fig6_markov_blanket            Markov blanket of a node: parents, children
                                   and co-parents shading vs the rest of the
                                   network.
    fig7_junction_tree             Triangulation -> max-clique -> junction tree
                                   pipeline shown as three small panels.

Usage:
    python3 scripts/figures/ml-math-derivations/10-bayesian-networks.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS, annotate_callout  # noqa: E402, F401
setup_style()



# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# style applied via _style.setup_style()

C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = COLORS["bg"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "10-Semi-Naive-Bayes-and-Bayesian-Networks"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "10-半朴素贝叶斯与贝叶斯网络"
)


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
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_node(ax, xy, label, *, color=C_BLUE, radius=0.32, text_color="white",
              fontsize=12, edgecolor=None, lw=1.5):
    circ = plt.Circle(xy, radius, facecolor=color, edgecolor=edgecolor or color,
                      linewidth=lw, zorder=3)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)


def draw_edge(ax, p, q, *, color=C_DARK, lw=1.6, style="-", shrink=18, alpha=1.0):
    arrow = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                            color=color, lw=lw, linestyle=style,
                            shrinkA=shrink, shrinkB=shrink, alpha=alpha,
                            zorder=2)
    ax.add_patch(arrow)


def panel_title(ax, text, *, y=1.04, fontsize=12):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=C_DARK)


# ---------------------------------------------------------------------------
# Figure 1: A canonical Bayesian network DAG
# ---------------------------------------------------------------------------
def fig1_dag() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-1.3, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Nodes: classic 4-variable example
    pos = {
        "Cloudy":   (1.0, 3.4),
        "Sprinkler": (0.0, 1.7),
        "Rain":     (2.4, 1.7),
        "WetGrass": (1.2, 0.0),
    }
    colors = {
        "Cloudy":   C_BLUE,
        "Sprinkler": C_PURPLE,
        "Rain":     C_GREEN,
        "WetGrass": C_AMBER,
    }
    for name, p in pos.items():
        draw_node(ax, p, name, color=colors[name], radius=0.46, fontsize=10)

    # Directed edges
    edges = [
        ("Cloudy", "Sprinkler"),
        ("Cloudy", "Rain"),
        ("Sprinkler", "WetGrass"),
        ("Rain", "WetGrass"),
    ]
    for u, v in edges:
        draw_edge(ax, pos[u], pos[v], color=C_DARK, lw=1.8, shrink=26)

    # CPT vignette next to Cloudy
    ax.text(3.5, 3.7, "Conditional probability tables", fontsize=10,
            color=C_DARK, fontweight="bold")
    cpt_lines = [
        r"$P(C)=0.5$",
        r"$P(S{=}1\mid C{=}1)=0.10,\ P(S{=}1\mid C{=}0)=0.50$",
        r"$P(R{=}1\mid C{=}1)=0.80,\ P(R{=}1\mid C{=}0)=0.20$",
        r"$P(W{=}1\mid S,R)$: 4 entries",
    ]
    for i, line in enumerate(cpt_lines):
        ax.text(3.5, 3.3 - 0.32 * i, line, fontsize=10, color=C_DARK)

    # Factorisation banner
    factor_box = FancyBboxPatch((0.1, -1.1), 5.9, 0.75,
                                boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=C_BG, edgecolor=C_GRAY, lw=1.2)
    ax.add_patch(factor_box)
    ax.text(3.05, -0.73,
            r"$P(C,S,R,W) = P(C)\,P(S\mid C)\,P(R\mid C)\,P(W\mid S,R)$",
            ha="center", va="center", fontsize=12, color=C_DARK)
    ax.text(3.05, -1.02,
            "16 joint entries -> 1 + 2 + 2 + 4 = 9 free parameters",
            ha="center", va="center", fontsize=9.5, color=C_GRAY,
            style="italic")

    panel_title(ax,
                "Bayesian network: a DAG factorises the joint into local CPTs",
                fontsize=13)
    _save(fig, "fig1_dag")


# ---------------------------------------------------------------------------
# Figure 2: d-separation - chain, fork, collider
# ---------------------------------------------------------------------------
def fig2_d_separation() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    panels = [
        {
            "title": "Chain  $A \\to B \\to C$",
            "edges": [("A", "B"), ("B", "C")],
            "rule_top":   r"unobserved $B$:  $A \not\!\perp C$",
            "rule_bot":   r"observed $B$:    $A \perp C \mid B$",
            "shaded":     "B",
            "intuition":  "Information flows through B; conditioning blocks it.",
        },
        {
            "title": "Fork  $A \\leftarrow B \\to C$",
            "edges": [("B", "A"), ("B", "C")],
            "rule_top":   r"unobserved $B$:  $A \not\!\perp C$",
            "rule_bot":   r"observed $B$:    $A \perp C \mid B$",
            "shaded":     "B",
            "intuition":  "Common cause induces correlation; observing it removes it.",
        },
        {
            "title": "Collider  $A \\to B \\leftarrow C$",
            "edges": [("A", "B"), ("C", "B")],
            "rule_top":   r"unobserved $B$:  $A \perp C$",
            "rule_bot":   r"observed $B$:    $A \not\!\perp C \mid B$  (explaining away)",
            "shaded":     "B",
            "intuition":  "Conditioning on the effect couples the causes.",
        },
    ]

    pos = {"A": (-1.0, 0.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    for ax, panel in zip(axes, panels):
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-2.1, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        panel_title(ax, panel["title"], fontsize=12)

        # Nodes
        for name, p in pos.items():
            color = C_AMBER if name == panel["shaded"] else C_BLUE
            draw_node(ax, p, name, color=color, radius=0.30)
            if name == panel["shaded"]:
                ring = plt.Circle(p, 0.42, facecolor="none",
                                  edgecolor=C_AMBER, lw=1.8, ls="--", zorder=2)
                ax.add_patch(ring)

        for u, v in panel["edges"]:
            draw_edge(ax, pos[u], pos[v], color=C_DARK, lw=2.0, shrink=18)

        ax.text(0.0, -1.0, panel["rule_top"], ha="center", fontsize=10.5,
                color=C_DARK)
        ax.text(0.0, -1.45, panel["rule_bot"], ha="center", fontsize=10.5,
                color=C_PURPLE, fontweight="bold")
        ax.text(0.0, -1.95, panel["intuition"], ha="center", fontsize=9,
                color=C_GRAY, style="italic")

    fig.suptitle("d-separation: three canonical structures",
                 fontsize=13.5, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_d_separation")


# ---------------------------------------------------------------------------
# Figure 3: Naive Bayes vs Tree-Augmented Naive Bayes
# ---------------------------------------------------------------------------
def fig3_nb_vs_tan() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # ----- Naive Bayes (star) -----
    ax = axes[0]
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.6, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_title(ax, "Naive Bayes:  features independent given $Y$", fontsize=12.5)

    Y = (0.0, 1.4)
    feat_pos = [
        (-2.4, -0.6), (-1.2, -1.4), (0.0, -1.7),
        (1.2, -1.4), (2.4, -0.6),
    ]
    feat_labels = [r"$X_1$", r"$X_2$", r"$X_3$", r"$X_4$", r"$X_5$"]
    draw_node(ax, Y, r"$Y$", color=C_PURPLE, radius=0.36, fontsize=12)
    for p, lbl in zip(feat_pos, feat_labels):
        draw_node(ax, p, lbl, color=C_BLUE, radius=0.30, fontsize=11)
        draw_edge(ax, Y, p, color=C_DARK, lw=1.6, shrink=18)

    ax.text(0.0, -2.35,
            r"$P(\mathbf{x},y) = P(y)\prod_{j=1}^{d} P(x_j\mid y)$",
            ha="center", fontsize=11.5, color=C_DARK)

    # ----- TAN -----
    ax = axes[1]
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.6, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_title(ax, "TAN:  features form an augmenting tree", fontsize=12.5)

    draw_node(ax, Y, r"$Y$", color=C_PURPLE, radius=0.36, fontsize=12)
    for p, lbl in zip(feat_pos, feat_labels):
        draw_node(ax, p, lbl, color=C_BLUE, radius=0.30, fontsize=11)
        draw_edge(ax, Y, p, color=C_GRAY, lw=1.0, shrink=18, alpha=0.55)

    # Tree edges among features (max spanning tree)
    tree_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for i, j in tree_edges:
        draw_edge(ax, feat_pos[i], feat_pos[j], color=C_GREEN,
                  lw=2.2, shrink=18)

    ax.text(0.0, -2.35,
            r"$P(\mathbf{x},y) = P(y)\prod_{j=1}^{d} P(x_j\mid y, x_{\pi(j)})$",
            ha="center", fontsize=11.5, color=C_DARK)

    # Legend strip
    fig.text(0.5, 0.015,
             "Class edges (grey, faded) and augmenting tree edges (green) "
             "weighted by conditional mutual information $I(X_j;X_k\\mid Y)$.",
             ha="center", fontsize=9.5, color=C_GRAY, style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "fig3_nb_vs_tan")


# ---------------------------------------------------------------------------
# Figure 4: AODE = average of SPODE sub-models
# ---------------------------------------------------------------------------
def fig4_aode() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))

    Y = (0.0, 1.6)
    feat_pos = [(-1.6, -0.6), (-0.55, -1.2), (0.55, -1.2), (1.6, -0.6)]
    labels = [r"$X_1$", r"$X_2$", r"$X_3$", r"$X_4$"]

    for k in range(3):  # SPODE_1, SPODE_2, SPODE_3
        ax = axes[k]
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.1, 2.4)
        ax.set_aspect("equal")
        ax.axis("off")
        sp = k  # super-parent index
        panel_title(ax, f"SPODE: super-parent = $X_{{{sp+1}}}$", fontsize=11.5)
        draw_node(ax, Y, r"$Y$", color=C_PURPLE, radius=0.32, fontsize=11)

        for i, (p, lbl) in enumerate(zip(feat_pos, labels)):
            color = C_AMBER if i == sp else C_BLUE
            draw_node(ax, p, lbl, color=color, radius=0.28, fontsize=10)
            draw_edge(ax, Y, p, color=C_GRAY, lw=1.1, shrink=16, alpha=0.6)

        # super-parent -> other features
        for i in range(len(feat_pos)):
            if i == sp:
                continue
            draw_edge(ax, feat_pos[sp], feat_pos[i], color=C_AMBER,
                      lw=1.8, shrink=16)

    # Final panel: AODE = average
    ax = axes[3]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "AODE:  average of all eligible SPODEs", fontsize=11.5)
    box = FancyBboxPatch((0.04, 0.18), 0.92, 0.55,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         facecolor=C_BG, edgecolor=C_AMBER, lw=1.5)
    ax.add_patch(box)
    ax.text(0.5, 0.62,
            r"$\hat{P}(y\mid\mathbf{x}) \,\propto\,$"
            r"$\sum_{i:\,n_i\geq m} P(y)\,P(x_i\mid y)$",
            ha="center", fontsize=11, color=C_DARK)
    ax.text(0.5, 0.42,
            r"$\quad\times\,\prod_{j\ne i} P(x_j\mid y, x_i)$",
            ha="center", fontsize=11, color=C_DARK)
    ax.text(0.5, 0.24,
            "(filter unreliable super-parents with $n_i \\geq m$)",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    fig.suptitle("AODE: ensemble of one-dependence estimators (no structure search)",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.04)
    fig.tight_layout()
    _save(fig, "fig4_aode")


# ---------------------------------------------------------------------------
# Figure 5: Variable elimination on a chain
# ---------------------------------------------------------------------------
def fig5_variable_elimination() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # Chain A -> B -> C -> D ; query P(D), eliminate A then B then C
    chain = [("A", -1.5), ("B", -0.5), ("C", 0.5), ("D", 1.5)]
    coords = {n: (x, 0.0) for n, x in chain}

    panels = [
        {
            "title": "Step 0: factors  $P(A)\\,P(B|A)\\,P(C|B)\\,P(D|C)$",
            "alive": ["A", "B", "C", "D"],
            "factor": r"$\phi_1(A,B)\,\phi_2(B,C)\,\phi_3(C,D)$",
            "highlight": None,
        },
        {
            "title": "Step 1: eliminate $A$  ->  $\\tau_1(B)=\\sum_A P(A)\\,P(B|A)$",
            "alive": ["B", "C", "D"],
            "factor": r"$\tau_1(B)\,\phi_2(B,C)\,\phi_3(C,D)$",
            "highlight": "A",
        },
        {
            "title": "Step 2: eliminate $B,C$  ->  posterior $P(D)$",
            "alive": ["D"],
            "factor": r"$P(D) = \sum_C \phi_3(C,D)\sum_B \tau_1(B)\,\phi_2(B,C)$",
            "highlight": "B,C",
        },
    ]

    for ax, panel in zip(axes, panels):
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.4, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        panel_title(ax, panel["title"], fontsize=10.5)

        for name, _ in chain:
            p = coords[name]
            if name in panel["alive"]:
                color = C_GREEN if name == "D" else C_BLUE
                draw_node(ax, p, name, color=color, radius=0.30)
            else:
                # eliminated: faded ghost
                circ = plt.Circle(p, 0.30, facecolor=C_LIGHT,
                                  edgecolor=C_GRAY, ls=":", lw=1.4, zorder=3)
                ax.add_patch(circ)
                ax.text(p[0], p[1], name, ha="center", va="center",
                        color=C_GRAY, fontsize=11, zorder=4)

        # Edges along chain (only between alive consecutive nodes)
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        for u, v in edges:
            color = C_DARK
            alpha = 1.0
            if u not in panel["alive"] or v not in panel["alive"]:
                color = C_GRAY
                alpha = 0.4
            draw_edge(ax, coords[u], coords[v], color=color, lw=1.6,
                      shrink=18, alpha=alpha)

        ax.text(0.0, -1.25, panel["factor"], ha="center", fontsize=10.5,
                color=C_DARK)
        if panel["highlight"]:
            ax.text(0.0, -1.85,
                    f"summed out: ${panel['highlight']}$",
                    ha="center", fontsize=9.5, color=C_AMBER,
                    fontweight="bold")

    fig.suptitle("Variable elimination: marginalise one variable at a time, "
                 "passing intermediate factors $\\tau$",
                 fontsize=12.5, fontweight="bold", color=C_DARK, y=1.03)
    fig.tight_layout()
    _save(fig, "fig5_variable_elimination")


# ---------------------------------------------------------------------------
# Figure 6: Markov blanket
# ---------------------------------------------------------------------------
def fig6_markov_blanket() -> None:
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    pos = {
        "X":   (0.0, 0.0),
        # parents
        "P1":  (-1.6, 1.6),
        "P2":  (1.6, 1.6),
        # children
        "C1":  (-1.0, -1.7),
        "C2":  (1.0, -1.7),
        # co-parents (other parents of the children)
        "S1":  (-2.7, -1.0),
        "S2":  (2.7, -1.0),
        # outsiders (NOT in blanket)
        "U1":  (-2.8, 2.4),
        "U2":  (2.8, 2.4),
        "U3":  (0.0, -2.7),
    }
    in_blanket = {"P1", "P2", "C1", "C2", "S1", "S2"}

    # Soft halo around blanket nodes (and X)
    halo_nodes = list(in_blanket) + ["X"]
    halo_pts = np.array([pos[n] for n in halo_nodes])
    cx, cy = halo_pts.mean(axis=0)
    rmax = np.max(np.linalg.norm(halo_pts - [cx, cy], axis=1)) + 0.7
    halo = plt.Circle((cx, cy), rmax, facecolor=C_AMBER, alpha=0.10,
                      edgecolor=C_AMBER, lw=1.6, ls="--", zorder=1)
    ax.add_patch(halo)
    ax.text(cx, cy + rmax + 0.2, "Markov blanket of $X$",
            ha="center", fontsize=11, color=C_AMBER, fontweight="bold")

    # Edges
    edges = [
        ("P1", "X"), ("P2", "X"),       # parents -> X
        ("X", "C1"), ("X", "C2"),       # X -> children
        ("S1", "C1"), ("S2", "C2"),     # co-parents -> children
        ("U1", "P1"), ("U2", "P2"),     # outside connections
        ("C1", "U3"), ("C2", "U3"),     # children to grand-children outside
    ]
    for u, v in edges:
        color = C_DARK if (u in in_blanket or u == "X") and \
                          (v in in_blanket or v == "X") else C_GRAY
        alpha = 1.0 if color == C_DARK else 0.45
        draw_edge(ax, pos[u], pos[v], color=color, lw=1.5, shrink=18,
                  alpha=alpha)

    # Nodes
    role_color = {
        "X": C_PURPLE,
        "P1": C_BLUE, "P2": C_BLUE,
        "C1": C_GREEN, "C2": C_GREEN,
        "S1": C_AMBER, "S2": C_AMBER,
        "U1": C_GRAY, "U2": C_GRAY, "U3": C_GRAY,
    }
    role_label = {
        "X": "X",
        "P1": "Pa$_1$", "P2": "Pa$_2$",
        "C1": "Ch$_1$", "C2": "Ch$_2$",
        "S1": "Sp$_1$", "S2": "Sp$_2$",
        "U1": "U$_1$", "U2": "U$_2$", "U3": "U$_3$",
    }
    for name, p in pos.items():
        draw_node(ax, p, role_label[name], color=role_color[name],
                  radius=0.34, fontsize=10)

    # Legend
    legend_y = -2.85
    items = [
        (C_PURPLE, "target $X$"),
        (C_BLUE, "parents Pa($X$)"),
        (C_GREEN, "children Ch($X$)"),
        (C_AMBER, "co-parents Sp($X$)"),
        (C_GRAY, "outside the blanket"),
    ]
    x_cursor = -3.2
    for color, txt in items:
        circ = plt.Circle((x_cursor, legend_y), 0.13, color=color)
        ax.add_patch(circ)
        ax.text(x_cursor + 0.22, legend_y, txt, va="center", fontsize=9,
                color=C_DARK)
        x_cursor += 1.4

    panel_title(ax,
                "Markov blanket = parents + children + children's other parents",
                fontsize=12.5)
    _save(fig, "fig6_markov_blanket")


# ---------------------------------------------------------------------------
# Figure 7: Junction tree pipeline
# ---------------------------------------------------------------------------
def fig7_junction_tree() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # ---------- Panel 1: original moralised undirected graph ----------
    ax = axes[0]
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_title(ax, "1. Moralise + triangulate", fontsize=12)

    pos = {
        "A": (-1.0, 1.0),
        "B": (1.0, 1.0),
        "C": (-1.0, -1.0),
        "D": (1.0, -1.0),
        "E": (0.0, 0.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"),
             ("A", "E"), ("B", "E"), ("C", "E"), ("D", "E")]
    # added chord
    chord = ("A", "D")
    edges.append(chord)
    for u, v in edges:
        is_chord = (u, v) == chord
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=C_AMBER if is_chord else C_DARK,
                lw=2.2 if is_chord else 1.5,
                ls="--" if is_chord else "-",
                zorder=2)
    for n, p in pos.items():
        draw_node(ax, p, n, color=C_BLUE, radius=0.22, fontsize=10)
    ax.text(0.0, -1.45, "added chord $A\\!-\\!D$ (amber)", ha="center",
            fontsize=9, color=C_AMBER, style="italic")

    # ---------- Panel 2: maximal cliques ----------
    ax = axes[1]
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-1.8, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_title(ax, "2. Extract maximal cliques", fontsize=12)

    cliques = [
        ("$\\{A,B,E\\}$", 0.6, 0.7, C_BLUE),
        ("$\\{A,D,E\\}$", 2.0, 0.7, C_PURPLE),
        ("$\\{A,C,D\\}$", 0.6, -0.6, C_GREEN),
        ("$\\{B,D,E\\}$", 2.0, -0.6, C_AMBER),
    ]
    for label, x, y, color in cliques:
        box = FancyBboxPatch((x - 0.55, y - 0.32), 1.1, 0.64,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=color, edgecolor=color, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")

    ax.text(2.0, -1.4, "each clique becomes a node of the junction tree",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    # ---------- Panel 3: junction tree with separators ----------
    ax = axes[2]
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(-1.8, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_title(ax, "3. Junction tree (running intersection)", fontsize=12)

    nodes = {
        "ABE": (0.6, 0.9, C_BLUE,   "$\\{A,B,E\\}$"),
        "ADE": (2.0, 0.9, C_PURPLE, "$\\{A,D,E\\}$"),
        "ACD": (0.6, -0.6, C_GREEN, "$\\{A,C,D\\}$"),
        "BDE": (3.4, -0.6, C_AMBER, "$\\{B,D,E\\}$"),
    }
    for key, (x, y, color, label) in nodes.items():
        box = FancyBboxPatch((x - 0.55, y - 0.30), 1.1, 0.6,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=color, edgecolor=color, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")

    # Tree edges with separator labels
    tree = [
        ("ABE", "ADE", "$\\{A,E\\}$"),
        ("ADE", "ACD", "$\\{A,D\\}$"),
        ("ADE", "BDE", "$\\{D,E\\}$"),
    ]
    for u, v, sep in tree:
        x1, y1 = nodes[u][0], nodes[u][1]
        x2, y2 = nodes[v][0], nodes[v][1]
        ax.plot([x1, x2], [y1, y2], color=C_DARK, lw=1.8, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, sep, ha="center", va="center", fontsize=9,
                color=C_DARK,
                bbox=dict(facecolor="white", edgecolor=C_GRAY,
                          boxstyle="round,pad=0.18", lw=0.8))

    ax.text(2.0, -1.55,
            "messages pass between cliques via shared separator variables",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    fig.suptitle("Junction tree algorithm: from DAG to exact inference engine",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_junction_tree")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 10 figures (Semi-Naive Bayes & Bayesian Networks)...")
    fig1_dag()
    fig2_d_separation()
    fig3_nb_vs_tan()
    fig4_aode()
    fig5_variable_elimination()
    fig6_markov_blanket()
    fig7_junction_tree()
    print("Done.")


if __name__ == "__main__":
    main()
