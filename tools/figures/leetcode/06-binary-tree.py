"""
Figure generation script for LeetCode Part 06: Binary Tree Traversal & Construction.

Generates 5 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, no decoration noise.

Figures:
    fig1_tree_anatomy          A labeled binary tree showing root, internal
                               nodes, leaves, depth and height. Establishes
                               the vocabulary used by every later figure.
    fig2_dfs_traversals        Side-by-side preorder / inorder / postorder
                               on the same tree, with numbered visit order
                               on each node. Makes the "where does the root
                               sit" rule obvious.
    fig3_bfs_levelorder        Level-order traversal: queue snapshot at each
                               level + which level each node belongs to.
    fig4_recursive_vs_stack    Recursive call stack vs. explicit iterative
                               stack for inorder traversal, frame-by-frame.
                               Shows they are doing the same work.
    fig5_build_from_pre_in     Recursive split of preorder + inorder arrays
                               into root / left / right slices, with the
                               reconstructed tree on the right.

Usage:
    python3 scripts/figures/leetcode/06-binary-tree.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "leetcode" / "binary-tree-traversal"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "leetcode" / "06-二叉树遍历与构造"


def _save(fig: plt.Figure, name: str) -> None:
    """Write the figure to BOTH the EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared tree data
# ---------------------------------------------------------------------------
# Canonical example tree used through the article:
#            3
#          /   \
#         9    20
#             /  \
#            15   7
#
# preorder  = [3, 9, 20, 15, 7]
# inorder   = [9, 3, 15, 20, 7]
# postorder = [9, 15, 7, 20, 3]
# levelord  = [[3], [9, 20], [15, 7]]

EDGES = [(3, 9), (3, 20), (20, 15), (20, 7)]
POS = {
    3:  (0.0, 3.0),
    9:  (-2.4, 2.0),
    20: (2.4, 2.0),
    15: (1.2, 1.0),
    7:  (3.6, 1.0),
}


def _draw_tree(
    ax: plt.Axes,
    pos: Dict[int, Tuple[float, float]],
    edges: List[Tuple[int, int]],
    node_colors: Optional[Dict[int, str]] = None,
    node_labels: Optional[Dict[int, str]] = None,
    edge_color: str = C_GRAY,
    node_size: int = 1700,
    font_size: int = 13,
    show_default_label: bool = True,
) -> None:
    """Draw a labeled binary tree on the given axes."""
    g = nx.DiGraph()
    g.add_edges_from(edges)
    for n in pos:
        g.add_node(n)

    colors = [node_colors.get(n, C_BLUE) if node_colors else C_BLUE for n in g.nodes()]

    nx.draw_networkx_edges(
        g, pos, ax=ax, edge_color=edge_color, width=1.6, arrows=False,
    )
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=colors, node_size=node_size,
        edgecolors=C_DARK, linewidths=1.2,
    )

    if node_labels is None and show_default_label:
        node_labels = {n: str(n) for n in g.nodes()}
    if node_labels:
        nx.draw_networkx_labels(
            g, pos, labels=node_labels, ax=ax,
            font_size=font_size, font_color="white", font_weight="bold",
        )

    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Figure 1: Tree anatomy (depth / height / root / leaf)
# ---------------------------------------------------------------------------
def fig1_tree_anatomy() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    node_colors = {
        3: C_PURPLE,   # root
        9: C_GREEN,    # leaf
        20: C_BLUE,    # internal
        15: C_GREEN,   # leaf
        7: C_GREEN,    # leaf
    }
    _draw_tree(ax, POS, EDGES, node_colors=node_colors)

    # Node role annotations (positioned to avoid overlap)
    annotations = [
        (3,  ( 0.55,  0.30), "root",            C_PURPLE),
        (9,  (-0.55, -0.45), "leaf",            C_GREEN),
        (20, ( 0.55,  0.30), "internal node",   C_BLUE),
        (15, (-0.45, -0.45), "leaf",            C_GREEN),
        (7,  ( 0.45, -0.45), "leaf",            C_GREEN),
    ]
    for n, (dx, dy), text, color in annotations:
        x, y = POS[n]
        ax.text(x + dx, y + dy, text, fontsize=10, color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec=color, lw=1))

    # Depth labels on the left, level lines for clarity
    for level, y in [(0, 3.0), (1, 2.0), (2, 1.0)]:
        ax.axhline(y, xmin=0.04, xmax=0.96, color=C_LIGHT, lw=1, zorder=0)
        ax.text(-4.4, y, f"depth {level}", fontsize=10,
                color=C_DARK, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc=C_BG, ec=C_GRAY, lw=0.8))

    # Height arrow (right side)
    ax.annotate(
        "", xy=(4.2, 3.0), xytext=(4.2, 1.0),
        arrowprops=dict(arrowstyle="<->", color=C_AMBER, lw=2),
    )
    ax.text(4.5, 2.0, "height = 2\n(longest path\nroot to leaf)",
            fontsize=10, color=C_AMBER, fontweight="bold", va="center")

    ax.set_xlim(-5.0, 6.0)
    ax.set_ylim(0.4, 3.7)
    ax.set_title("Binary tree anatomy: root, internal nodes, leaves, depth, height",
                 fontsize=13, fontweight="bold", pad=12, color=C_DARK)
    plt.tight_layout()
    _save(fig, "fig1_tree_anatomy")


# ---------------------------------------------------------------------------
# Figure 2: DFS traversal orders side-by-side
# ---------------------------------------------------------------------------
def fig2_dfs_traversals() -> None:
    """Pre / In / Post order on the same tree, numbered by visit order."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    orders = [
        ("Preorder  (Root - Left - Right)",  [3, 9, 20, 15, 7]),
        ("Inorder   (Left - Root - Right)",  [9, 3, 15, 20, 7]),
        ("Postorder (Left - Right - Root)",  [9, 15, 7, 20, 3]),
    ]

    for ax, (title, sequence) in zip(axes, orders):
        # Color = visit order via amber gradient (early=light, late=dark)
        cmap = plt.get_cmap("YlOrRd")
        node_colors = {
            n: cmap(0.25 + 0.65 * (i / max(1, len(sequence) - 1)))
            for i, n in enumerate(sequence)
        }
        labels = {n: f"{n}\n#{sequence.index(n) + 1}" for n in sequence}

        _draw_tree(
            ax, POS, EDGES,
            node_colors=node_colors,
            node_labels=labels,
            node_size=2000,
            font_size=10,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", color=C_DARK, pad=8)

        # Sequence strip below the tree
        seq_str = " -> ".join(str(v) for v in sequence)
        ax.text(0.0, 0.15, seq_str, transform=ax.transAxes,
                ha="center", fontsize=11, color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.4", fc=C_BG,
                          ec=C_PURPLE, lw=1.2))
        ax.set_xlim(-4.0, 4.6)
        ax.set_ylim(0.4, 3.7)

    fig.suptitle("Three DFS traversal orders on the same tree (number = visit order)",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.02)
    plt.tight_layout()
    _save(fig, "fig2_dfs_traversals")


# ---------------------------------------------------------------------------
# Figure 3: BFS / level-order traversal
# ---------------------------------------------------------------------------
def fig3_bfs_levelorder() -> None:
    """Tree on left, queue snapshots on right."""
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.18)

    # Left: tree colored by level
    ax_tree = fig.add_subplot(gs[0, 0])
    level_color = {0: C_PURPLE, 1: C_BLUE, 2: C_GREEN}
    node_level = {3: 0, 9: 1, 20: 1, 15: 2, 7: 2}
    node_colors = {n: level_color[lvl] for n, lvl in node_level.items()}

    _draw_tree(ax_tree, POS, EDGES, node_colors=node_colors, node_size=1900)

    for level, y in [(0, 3.0), (1, 2.0), (2, 1.0)]:
        ax_tree.text(
            -4.2, y, f"Level {level}",
            fontsize=11, fontweight="bold",
            color=level_color[level], va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec=level_color[level], lw=1.2),
        )
    ax_tree.set_xlim(-4.6, 4.2)
    ax_tree.set_ylim(0.4, 3.7)
    ax_tree.set_title("Tree colored by level", fontsize=12,
                      fontweight="bold", color=C_DARK, pad=8)

    # Right: queue snapshots at each step
    ax_q = fig.add_subplot(gs[0, 1])
    ax_q.set_axis_off()
    ax_q.set_xlim(0, 10)
    ax_q.set_ylim(0, 10)
    ax_q.set_title("Queue snapshots (FIFO) — process one level per outer iteration",
                   fontsize=12, fontweight="bold", color=C_DARK, pad=8)

    snapshots = [
        ("Init",          [3],          C_PURPLE, "enqueue root"),
        ("After level 0", [9, 20],      C_BLUE,   "pop 3, enqueue children"),
        ("After level 1", [15, 7],      C_GREEN,  "pop 9 and 20, enqueue children"),
        ("After level 2", [],           C_GRAY,   "pop 15 and 7, no children -> done"),
    ]
    box_w, box_h = 1.0, 0.8
    y_top = 8.8
    for i, (label, queue, color, note) in enumerate(snapshots):
        y = y_top - i * 2.1
        ax_q.text(0.2, y + box_h / 2, label, fontsize=10,
                  fontweight="bold", color=C_DARK, va="center")
        # draw boxes for queue elements
        x0 = 2.4
        if not queue:
            ax_q.add_patch(FancyBboxPatch(
                (x0, y), box_w * 1.2, box_h,
                boxstyle="round,pad=0.04",
                fc=C_BG, ec=C_GRAY, lw=1, linestyle="--",
            ))
            ax_q.text(x0 + box_w * 0.6, y + box_h / 2, "empty",
                      ha="center", va="center", fontsize=10, color=C_GRAY)
        else:
            for j, v in enumerate(queue):
                ax_q.add_patch(FancyBboxPatch(
                    (x0 + j * (box_w + 0.15), y), box_w, box_h,
                    boxstyle="round,pad=0.04",
                    fc=color, ec=C_DARK, lw=1.2,
                ))
                ax_q.text(x0 + j * (box_w + 0.15) + box_w / 2,
                          y + box_h / 2, str(v),
                          ha="center", va="center",
                          color="white", fontsize=12, fontweight="bold")
        # note
        ax_q.text(0.2, y - 0.35, note, fontsize=9,
                  color=C_DARK, va="center", style="italic")

    # legend / front-back arrow
    ax_q.annotate("", xy=(2.4, 9.6), xytext=(7.8, 9.6),
                  arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.4))
    ax_q.text(2.4, 9.85, "front (popleft)", fontsize=9, color=C_DARK)
    ax_q.text(7.8, 9.85, "back (append)", fontsize=9, color=C_DARK, ha="right")

    fig.suptitle("BFS / Level-order traversal: visit nodes one level at a time",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.0)
    plt.tight_layout()
    _save(fig, "fig3_bfs_levelorder")


# ---------------------------------------------------------------------------
# Figure 4: Recursive call stack vs. explicit iterative stack
# ---------------------------------------------------------------------------
def fig4_recursive_vs_stack() -> None:
    """Inorder traversal: call-stack frames vs. explicit-stack frames."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))

    # ---------- Left: recursive call stack ----------
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Recursive: implicit call stack",
                 fontsize=12, fontweight="bold", color=C_DARK, pad=8)

    frames = [
        "inorder(3)",
        "  inorder(9)",
        "    inorder(None) -> return",
        "    visit 9",
        "    inorder(None) -> return",
        "  visit 3",
        "  inorder(20)",
        "    inorder(15) ... visit 15 ...",
        "    visit 20",
        "    inorder(7) ... visit 7 ...",
    ]
    y = 9.2
    for i, f in enumerate(frames):
        is_visit = "visit" in f
        color = C_GREEN if is_visit else C_BLUE
        ax.add_patch(FancyBboxPatch(
            (0.6, y - 0.55), 8.8, 0.55,
            boxstyle="round,pad=0.03",
            fc=("#ecfdf5" if is_visit else "#eff6ff"),
            ec=color, lw=1.0,
        ))
        ax.text(0.85, y - 0.28, f, fontsize=10,
                color=C_DARK, va="center", family="monospace")
        y -= 0.75

    ax.text(0.6, 0.4,
            "Each call adds a frame to Python's call stack.\n"
            "Stack depth = tree height h, so space O(h).",
            fontsize=9, color=C_DARK, style="italic")

    # ---------- Right: iterative explicit stack ----------
    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Iterative: explicit stack (inorder)",
                 fontsize=12, fontweight="bold", color=C_DARK, pad=8)

    # Each row shows: stack contents (top on the right) + result so far
    steps = [
        ([3],          [],            "go left from 3, push 3"),
        ([3, 9],       [],            "go left from 9 (None), pop 9"),
        ([3],          [9],           "visit 9; right of 9 is None"),
        ([],           [9],           "pop 3"),
        ([],           [9, 3],        "visit 3; move to right child 20"),
        ([20],         [9, 3],        "go left from 20, push 20"),
        ([20, 15],     [9, 3],        "go left from 15 (None), pop 15"),
        ([20],         [9, 3, 15],    "visit 15; right of 15 is None"),
        ([],           [9, 3, 15],    "pop 20"),
        ([],           [9,3,15,20],   "visit 20; move to right child 7"),
        ([7],          [9,3,15,20],   "go left from 7 (None), pop 7"),
        ([],           [9,3,15,20,7], "visit 7; stack empty -> done"),
    ]

    y = 9.4
    box_w = 0.55
    for stack, result, note in steps:
        # stack boxes
        x0 = 1.2
        ax.text(0.2, y, "stk:", fontsize=9, color=C_DARK,
                fontweight="bold", va="center")
        if not stack:
            ax.add_patch(FancyBboxPatch(
                (x0, y - 0.22), 0.7, 0.44,
                boxstyle="round,pad=0.03",
                fc=C_BG, ec=C_GRAY, lw=0.8, linestyle="--",
            ))
            ax.text(x0 + 0.35, y, "—", ha="center", va="center",
                    fontsize=9, color=C_GRAY)
        else:
            for j, v in enumerate(stack):
                ax.add_patch(FancyBboxPatch(
                    (x0 + j * (box_w + 0.05), y - 0.22),
                    box_w, 0.44,
                    boxstyle="round,pad=0.03",
                    fc=C_BLUE, ec=C_DARK, lw=1,
                ))
                ax.text(x0 + j * (box_w + 0.05) + box_w / 2, y,
                        str(v), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

        # result list
        res_str = "[" + ",".join(str(v) for v in result) + "]"
        ax.text(5.0, y, "res: " + res_str, fontsize=9,
                color=C_GREEN, va="center", family="monospace",
                fontweight="bold")
        # note
        ax.text(0.2, y - 0.45, note, fontsize=8,
                color=C_DARK, style="italic")
        y -= 0.78

    fig.suptitle("Same algorithm, two stacks: recursion vs. explicit iteration (Inorder)",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.0)
    plt.tight_layout()
    _save(fig, "fig4_recursive_vs_stack")


# ---------------------------------------------------------------------------
# Figure 5: Build tree from preorder + inorder
# ---------------------------------------------------------------------------
def fig5_build_from_pre_in() -> None:
    """Show recursive split of arrays into root / L / R + final tree."""
    fig = plt.figure(figsize=(14.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.15)

    # ----- Left: array slicing, three recursion levels -----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Recursive split: preorder gives root, inorder gives left/right boundary",
                 fontsize=12, fontweight="bold", color=C_DARK, pad=8)

    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]

    # cell helper
    def draw_cells(x0, y0, values, highlight=None, label=None,
                   left_split=None, right_split=None):
        cw = 0.65
        for i, v in enumerate(values):
            color = C_LIGHT
            if highlight is not None and i == highlight:
                color = C_PURPLE
            elif left_split is not None and i in left_split:
                color = C_BLUE
            elif right_split is not None and i in right_split:
                color = C_AMBER
            ax.add_patch(FancyBboxPatch(
                (x0 + i * cw, y0), cw, 0.55,
                boxstyle="round,pad=0.02",
                fc=color, ec=C_DARK, lw=1,
            ))
            text_color = "white" if color != C_LIGHT else C_DARK
            ax.text(x0 + i * cw + cw / 2, y0 + 0.27, str(v),
                    ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")
        if label:
            ax.text(x0 - 0.15, y0 + 0.27, label, fontsize=10,
                    color=C_DARK, va="center", ha="right",
                    fontweight="bold", family="monospace")

    # Level 0: full arrays, root = preorder[0] = 3
    y = 8.6
    ax.text(0.2, y + 0.7, "Step 1 — root = preorder[0] = 3",
            fontsize=10, color=C_PURPLE, fontweight="bold")
    draw_cells(2.0, y, preorder, highlight=0, label="preorder")
    draw_cells(2.0, y - 0.85, inorder, highlight=1,
               left_split=[0], right_split=[2, 3, 4], label="inorder")
    ax.text(2.0 + 5 * 0.65 + 0.15, y - 0.55,
            "  left = inorder[:1]   right = inorder[2:]",
            fontsize=9, color=C_DARK, family="monospace")

    # Level 1: left subtree (single 9), right subtree (root 20)
    y = 6.0
    ax.text(0.2, y + 0.7,
            "Step 2 — left=[9] (single leaf); right subtree root = 20",
            fontsize=10, color=C_PURPLE, fontweight="bold")
    # right subtree slice
    draw_cells(2.0, y, [20, 15, 7], highlight=0, label="pre[L]")
    draw_cells(2.0, y - 0.85, [15, 20, 7], highlight=1,
               left_split=[0], right_split=[2], label="in[L]")

    # Level 2: leaves
    y = 3.4
    ax.text(0.2, y + 0.7,
            "Step 3 — left=[15] leaf, right=[7] leaf -> recursion bottoms out",
            fontsize=10, color=C_PURPLE, fontweight="bold")
    draw_cells(2.0, y, [15], highlight=0, label="pre[LL]")
    draw_cells(5.0, y, [7], highlight=0, label="pre[LR]")
    draw_cells(2.0, y - 0.85, [15], highlight=0, label="in[LL]")
    draw_cells(5.0, y - 0.85, [7], highlight=0, label="in[LR]")

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_PURPLE, label="root (current call)"),
        mpatches.Patch(color=C_BLUE,   label="left-subtree slice"),
        mpatches.Patch(color=C_AMBER,  label="right-subtree slice"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
              fontsize=9)

    # ----- Right: reconstructed tree -----
    ax_t = fig.add_subplot(gs[0, 1])
    node_colors = {3: C_PURPLE, 9: C_BLUE, 20: C_AMBER, 15: C_BLUE, 7: C_AMBER}
    _draw_tree(ax_t, POS, EDGES, node_colors=node_colors, node_size=1800)
    ax_t.set_xlim(-4.0, 4.0)
    ax_t.set_ylim(0.4, 3.7)
    ax_t.set_title("Reconstructed tree", fontsize=12,
                   fontweight="bold", color=C_DARK, pad=8)

    fig.suptitle("Build a binary tree from preorder + inorder (LeetCode 105)",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.0)
    plt.tight_layout()
    _save(fig, "fig5_build_from_pre_in")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for LeetCode 06: Binary Tree Traversal & Construction")
    print(f"  EN target: {EN_DIR}")
    print(f"  ZH target: {ZH_DIR}")
    fig1_tree_anatomy()
    fig2_dfs_traversals()
    fig3_bfs_levelorder()
    fig4_recursive_vs_stack()
    fig5_build_from_pre_in()
    print("Done.")


if __name__ == "__main__":
    main()
