"""
Figure generation script for LeetCode Part 08: Backtracking Algorithms.

Generates 5 figures used in both EN and ZH versions of the article. Each
figure teaches one concrete idea about the backtracking template
(choose -> recurse -> un-choose) and writes PNGs to BOTH the EN and ZH
asset folders so the markdown references stay consistent across languages.

Figures:
    fig1_template
        The universal backtracking template visualized as a state machine:
        choose -> recurse -> un-choose, with annotations for path and
        constraint checks. The "spine" of the article.

    fig2_permutations_tree
        Full recursion tree for permute([1,2,3]). Six leaves, with
        used[] state shown at each level. Highlights why we need the
        used array and why we restore state on the way up.

    fig3_combinations_pruning
        Decision tree for combinationSum([2,3,6,7], target=7) showing
        which branches get pruned (remain < 0) versus which reach
        remain == 0. Pruned subtrees are drawn faded with a red cross.

    fig4_nqueens_board
        4-Queens: a valid solution board on the left, and on the right
        the three constraint sets (column, row-col diagonal,
        row+col diagonal) overlaid as lines so the diagonal formula
        becomes obvious.

    fig5_sudoku_step
        One backtracking step on a Sudoku cell: candidate digits 1..9,
        struck-through digits eliminated by row/col/box constraints,
        and the chosen digit highlighted. Shows pruning in action on a
        constraint satisfaction problem.

Usage:
    python3 scripts/figures/leetcode/08-backtracking.py

Output:
    PNGs at 150 dpi, written to:
      source/_posts/en/leetcode/backtracking/
      source/_posts/zh/leetcode/08-回溯算法/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_RED = "#ef4444"
C_GRAY = "#6b7280"
C_LIGHT = "#e5e7eb"
C_BG = "#f9fafb"

DPI = 150

# Output directories: write to BOTH EN and ZH asset folders.
ROOT = Path(__file__).resolve().parents[3]
OUT_EN = ROOT / "source" / "_posts" / "en" / "leetcode" / "backtracking"
OUT_ZH = ROOT / "source" / "_posts" / "zh" / "leetcode" / "08-回溯算法"
OUT_EN.mkdir(parents=True, exist_ok=True)
OUT_ZH.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (OUT_EN, OUT_ZH):
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rounded_box(ax, xy, w, h, label, *, fc, ec, fontsize=10, fontweight="bold",
                color="white"):
    """Draw a rounded rectangle node with a centered label."""
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=1.6, facecolor=fc, edgecolor=ec)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=color)


def arrow(ax, p1, p2, *, color=C_GRAY, lw=1.6, style="-|>", mut=14, ls="-"):
    """Draw an arrow between two points."""
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=mut,
                        linewidth=lw, color=color, linestyle=ls,
                        shrinkA=4, shrinkB=4)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: The universal backtracking template
# ---------------------------------------------------------------------------
def fig1_template() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("The Backtracking Template: choose -> recurse -> un-choose",
                 fontsize=13.5, fontweight="bold", pad=12)

    # Center: state node
    rounded_box(ax, (4.5, 2.7), 2.0, 1.0, "state\n(path, used)",
                fc=C_BG, ec=C_GRAY, color="#111827", fontweight="bold")

    # Step 1: Goal check (top)
    rounded_box(ax, (4.4, 5.0), 2.2, 0.85, "1. is_goal(path)?\nsave path[:]  return",
                fc=C_GREEN, ec=C_GREEN, fontsize=10)
    arrow(ax, (5.5, 5.0), (5.5, 3.7), color=C_GREEN, ls="--")

    # Step 2: Choose (left)
    rounded_box(ax, (0.4, 2.7), 2.4, 1.0, "2. choose\npath.append(c)\nused[c]=True",
                fc=C_BLUE, ec=C_BLUE, fontsize=10)
    arrow(ax, (2.8, 3.2), (4.5, 3.2), color=C_BLUE)

    # Step 3: Recurse (right)
    rounded_box(ax, (8.2, 2.7), 2.4, 1.0, "3. recurse\nbacktrack(...)",
                fc=C_PURPLE, ec=C_PURPLE, fontsize=10)
    arrow(ax, (6.5, 3.2), (8.2, 3.2), color=C_PURPLE)

    # Step 4: Un-choose (bottom)
    rounded_box(ax, (4.4, 0.5), 2.2, 0.95,
                "4. un-choose\npath.pop()\nused[c]=False",
                fc=C_AMBER, ec=C_AMBER, fontsize=10)
    arrow(ax, (5.5, 2.7), (5.5, 1.45), color=C_AMBER)

    # Loop back arrow: un-choose -> next choice
    arrow(ax, (4.4, 0.95), (1.6, 2.7), color=C_AMBER, ls="--")
    ax.text(2.4, 1.55, "next choice", color=C_AMBER, fontsize=9,
            fontweight="bold", style="italic")

    # Constraint pruning (between choose and the state)
    ax.text(3.65, 3.55, "is_valid?", color=C_RED, fontsize=9.5,
            fontweight="bold", ha="center")
    ax.text(3.65, 2.90, "(prune early)", color=C_RED, fontsize=8.5,
            ha="center", style="italic")

    # Side legend
    ax.text(0.4, 0.2,
            "Invariant: every recurse is matched by an un-choose, so the "
            "state is restored on the way up the tree.",
            fontsize=9.5, color="#374151", style="italic")

    save(fig, "fig1_template.png")


# ---------------------------------------------------------------------------
# Figure 2: Permutations recursion tree for [1,2,3]
# ---------------------------------------------------------------------------
def fig2_permutations_tree() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("Permutations Recursion Tree for [1, 2, 3]  (3! = 6 leaves)",
                 fontsize=13.5, fontweight="bold", pad=12)

    # Levels (y coordinates)
    y0, y1, y2, y3 = 6.4, 4.7, 3.0, 1.3
    r = 0.32  # node radius

    def node(x, y, label, *, fc=C_BLUE, ec=C_BLUE, txt="white"):
        c = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=txt, zorder=4)

    def edge(p1, p2, label=None, color=C_GRAY):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1.4, zorder=1)
        if label is not None:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(mx + 0.05, my + 0.02, label, fontsize=8.5,
                    color="#374151", fontweight="bold")

    # Root
    root = (6.5, y0)
    node(*root, "[]", fc="white", ec=C_GRAY, txt="#111827")

    # Level 1: choose 1, 2, or 3
    L1_x = [2.2, 6.5, 10.8]
    L1_lbl = ["[1]", "[2]", "[3]"]
    for x, lbl in zip(L1_x, L1_lbl):
        node(x, y1, lbl)
        edge(root, (x, y1), label=lbl[1])

    # Level 2: each of the 3 nodes has 2 children
    L2_layout = {
        0: [(0.8, "[1,2]", "2"), (3.6, "[1,3]", "3")],
        1: [(5.1, "[2,1]", "1"), (7.9, "[2,3]", "3")],
        2: [(9.4, "[3,1]", "1"), (12.2, "[3,2]", "2")],
    }
    for i, parent_x in enumerate(L1_x):
        for cx, lbl, num in L2_layout[i]:
            node(cx, y2, lbl, fc=C_PURPLE, ec=C_PURPLE)
            edge((parent_x, y1), (cx, y2), label=num)

    # Level 3: leaves (6 permutations)
    L3 = [
        (0.8, "[1,2,3]", "3"),
        (3.6, "[1,3,2]", "2"),
        (5.1, "[2,1,3]", "3"),
        (7.9, "[2,3,1]", "1"),
        (9.4, "[3,1,2]", "2"),
        (12.2, "[3,2,1]", "1"),
    ]
    for cx, lbl, num in L3:
        node(cx, y3, lbl, fc=C_GREEN, ec=C_GREEN)
        # connect to its parent at level 2 (same x)
        edge((cx, y2), (cx, y3), label=num)

    # Annotations
    ax.text(0.05, y0, "depth 0\nused=[F,F,F]", fontsize=9, color=C_GRAY,
            va="center")
    ax.text(0.05, y1, "depth 1", fontsize=9, color=C_GRAY, va="center")
    ax.text(0.05, y2, "depth 2", fontsize=9, color=C_GRAY, va="center")
    ax.text(0.05, y3, "depth 3\n(leaf, save)", fontsize=9, color=C_GREEN,
            va="center", fontweight="bold")

    # Caption
    ax.text(6.5, 0.25,
            "Edges = choices. used[] flips True on the way down and False on "
            "the way up — that is the 'un-choose' step.",
            ha="center", fontsize=10, color="#374151", style="italic")

    save(fig, "fig2_permutations_tree.png")


# ---------------------------------------------------------------------------
# Figure 3: Combination Sum pruning tree
# ---------------------------------------------------------------------------
def fig3_combinations_pruning() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.set_title("Combination Sum: candidates=[2,3,6,7], target=7  "
                 "(green = solution, red X = pruned by remain<0)",
                 fontsize=12.5, fontweight="bold", pad=12)

    r = 0.30

    def node(x, y, label, *, fc, ec, txt="white", alpha=1.0):
        c = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.6,
                   zorder=3, alpha=alpha)
        ax.add_patch(c)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=txt, zorder=4,
                alpha=alpha)

    def edge(p1, p2, label=None, *, color=C_GRAY, alpha=1.0, ls="-"):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1.4,
                zorder=1, alpha=alpha, linestyle=ls)
        if label is not None:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(mx + 0.06, my + 0.02, f"+{label}", fontsize=8.5,
                    color="#374151", fontweight="bold", alpha=alpha)

    # Root: remain = 7
    root = (6.5, 6.6)
    node(*root, "7", fc="white", ec=C_GRAY, txt="#111827")
    ax.text(root[0] + 0.5, root[1] + 0.05, "remain", fontsize=9, color=C_GRAY)

    # Level 1: choose 2, 3, 6, or 7  (start = 0)
    L1 = [(2.0, 5.1, 5, "2"), (5.0, 5.1, 4, "3"),
          (8.0, 5.1, 1, "6"), (11.0, 5.1, 0, "7")]
    for x, y, rem, c in L1:
        if rem == 0:
            node(x, y, str(rem), fc=C_GREEN, ec=C_GREEN)
        else:
            node(x, y, str(rem), fc=C_BLUE, ec=C_BLUE)
        edge(root, (x, y), label=c)

    # Mark [7] as a saved solution
    ax.text(11.0, 4.5, "save [7]", color=C_GREEN, fontsize=9,
            ha="center", fontweight="bold")

    # Level 2 from "2" (start=0, remain=5): try 2,3,6,7 -> remain 3, 2, -1(prune), -2(prune)
    P2 = (2.0, 5.1)
    L2_from_2 = [(0.4, 3.6, 3, "2"), (1.7, 3.6, 2, "3"),
                 (3.0, 3.6, -1, "6"), (4.3, 3.6, -2, "7")]
    for x, y, rem, c in L2_from_2:
        if rem < 0:
            # Pruned
            node(x, y, str(rem), fc=C_RED, ec=C_RED, alpha=0.55)
            edge(P2, (x, y), label=c, color=C_RED, alpha=0.55, ls="--")
            ax.plot(x, y, marker="x", color=C_RED, markersize=14,
                    markeredgewidth=2.5, zorder=5)
        else:
            node(x, y, str(rem), fc=C_BLUE, ec=C_BLUE)
            edge(P2, (x, y), label=c)

    # Level 2 from "3" (start=1, remain=4): try 3,6,7 -> 1, -2, -3
    P3 = (5.0, 5.1)
    L2_from_3 = [(5.7, 3.6, 1, "3"), (6.7, 3.6, -2, "6"),
                 (7.7, 3.6, -3, "7")]
    for x, y, rem, c in L2_from_3:
        if rem < 0:
            node(x, y, str(rem), fc=C_RED, ec=C_RED, alpha=0.55)
            edge(P3, (x, y), label=c, color=C_RED, alpha=0.55, ls="--")
            ax.plot(x, y, marker="x", color=C_RED, markersize=14,
                    markeredgewidth=2.5, zorder=5)
        else:
            node(x, y, str(rem), fc=C_BLUE, ec=C_BLUE)
            edge(P3, (x, y), label=c)

    # Level 2 from "6": only 6,7 reachable -> -5, -6 (both pruned)
    P6 = (8.0, 5.1)
    L2_from_6 = [(8.6, 3.6, -5, "6"), (9.6, 3.6, -6, "7")]
    for x, y, rem, c in L2_from_6:
        node(x, y, str(rem), fc=C_RED, ec=C_RED, alpha=0.55)
        edge(P6, (x, y), label=c, color=C_RED, alpha=0.55, ls="--")
        ax.plot(x, y, marker="x", color=C_RED, markersize=14,
                markeredgewidth=2.5, zorder=5)

    # Level 3: from [2,2] (remain=3) -> try 2,3,6,7 -> 1, 0(SAVE), -3, -4
    P22 = (0.4, 3.6)
    L3_from_22 = [(-0.2, 2.1, 1, "2"), (0.4, 2.1, 0, "3"),
                  (1.0, 2.1, -3, "6"), (1.6, 2.1, -4, "7")]
    # Clamp to visible area: shift entire row right
    L3_from_22 = [(x + 0.6, y, rem, c) for x, y, rem, c in L3_from_22]
    for x, y, rem, c in L3_from_22:
        if rem == 0:
            node(x, y, "0", fc=C_GREEN, ec=C_GREEN)
            edge(P22, (x, y), label=c)
            ax.text(x, y - 0.6, "save [2,2,3]", color=C_GREEN, fontsize=9,
                    ha="center", fontweight="bold")
        elif rem < 0:
            node(x, y, str(rem), fc=C_RED, ec=C_RED, alpha=0.55)
            edge(P22, (x, y), label=c, color=C_RED, alpha=0.55, ls="--")
            ax.plot(x, y, marker="x", color=C_RED, markersize=14,
                    markeredgewidth=2.5, zorder=5)
        else:
            node(x, y, str(rem), fc=C_BLUE, ec=C_BLUE)
            edge(P22, (x, y), label=c)

    # Note on [2,3] (remain=2) -> would need at least 2 more, only 3,6,7 available -> all pruned
    P23 = (1.7, 3.6)
    edge(P23, (2.2, 2.1), label="3", color=C_RED, alpha=0.55, ls="--")
    node(2.2, 2.1, "-1", fc=C_RED, ec=C_RED, alpha=0.55)
    ax.plot(2.2, 2.1, marker="x", color=C_RED, markersize=14,
            markeredgewidth=2.5, zorder=5)

    # Note on [3,3] (remain=1) all children pruned
    P33 = (5.7, 3.6)
    edge(P33, (5.7, 2.1), label="3", color=C_RED, alpha=0.55, ls="--")
    node(5.7, 2.1, "-2", fc=C_RED, ec=C_RED, alpha=0.55)
    ax.plot(5.7, 2.1, marker="x", color=C_RED, markersize=14,
            markeredgewidth=2.5, zorder=5)

    # Caption
    ax.text(6.5, 0.5,
            "Two solutions found: [2,2,3] and [7]. The 'start' index keeps "
            "us going forward in candidates so we never produce [3,2,2] as a duplicate.",
            ha="center", fontsize=10, color="#374151", style="italic")

    save(fig, "fig3_combinations_pruning.png")


# ---------------------------------------------------------------------------
# Figure 4: 4-Queens — solution board + diagonal constraints
# ---------------------------------------------------------------------------
def fig4_nqueens_board() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    fig.suptitle("4-Queens: a valid solution and the three constraint sets",
                 fontsize=13.5, fontweight="bold", y=0.99)

    # ---- Left: chessboard with queens at (0,1),(1,3),(2,0),(3,2)
    ax = axes[0]
    n = 4
    queens = [(0, 1), (1, 3), (2, 0), (3, 2)]
    for r in range(n):
        for c in range(n):
            color = "#f3f4f6" if (r + c) % 2 == 0 else "#9ca3af"
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor=color,
                                   edgecolor="#374151", linewidth=1))
    for r, c in queens:
        ax.add_patch(Circle((c + 0.5, n - 1 - r + 0.5), 0.32,
                            facecolor=C_PURPLE, edgecolor="#1e1b4b",
                            linewidth=1.8, zorder=3))
        ax.text(c + 0.5, n - 1 - r + 0.5, "Q", ha="center", va="center",
                fontsize=18, fontweight="bold", color="white", zorder=4)
    ax.set_xlim(-0.6, n + 0.1)
    ax.set_ylim(-0.6, n + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    # Row / col labels
    for i in range(n):
        ax.text(i + 0.5, -0.3, f"col {i}", ha="center", fontsize=9,
                color=C_GRAY)
        ax.text(-0.3, n - 1 - i + 0.5, f"row {i}", ha="right", va="center",
                fontsize=9, color=C_GRAY)
    ax.set_title("Solution: place queens row by row\n[1, 3, 0, 2]",
                 fontsize=11.5, color="#111827")

    # ---- Right: diagonal constraint visualization on an empty board
    ax = axes[1]
    for r in range(n):
        for c in range(n):
            color = "#f9fafb"
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor=color,
                                   edgecolor="#d1d5db", linewidth=1))
            # Annotate row-col and row+col in each cell
            ax.text(c + 0.5, n - 1 - r + 0.72,
                    f"r-c={r-c}", ha="center", fontsize=7.5,
                    color=C_BLUE)
            ax.text(c + 0.5, n - 1 - r + 0.28,
                    f"r+c={r+c}", ha="center", fontsize=7.5,
                    color=C_AMBER)

    # Highlight the queen at (1,3): its column, its r-c=-2, its r+c=4
    r0, c0 = 1, 3
    # Column 3 (top to bottom)
    ax.add_patch(Rectangle((c0, 0), 1, n, facecolor=C_GREEN,
                           edgecolor="none", alpha=0.15, zorder=0))
    # r - c = -2: cells (0,2),(1,3)
    for r, c in [(0, 2), (1, 3)]:
        ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor=C_BLUE,
                               edgecolor="none", alpha=0.25, zorder=0))
    # r + c = 4: cells (1,3),(2,2),(3,1) ... and (4,0) off-board
    for r, c in [(1, 3), (2, 2), (3, 1)]:
        ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor=C_AMBER,
                               edgecolor="none", alpha=0.25, zorder=0))
    # Draw the queen
    ax.add_patch(Circle((c0 + 0.5, n - 1 - r0 + 0.5), 0.30,
                        facecolor=C_PURPLE, edgecolor="#1e1b4b",
                        linewidth=1.8, zorder=3))
    ax.text(c0 + 0.5, n - 1 - r0 + 0.5, "Q", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white", zorder=4)

    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.6, n + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Three O(1) sets: cols, r-c diag, r+c diag",
                 fontsize=11.5, color="#111827")

    # Legend below
    ax.text(0.0, -0.45, "  column", color=C_GREEN, fontsize=9.5,
            fontweight="bold")
    ax.text(1.4, -0.45, "  r-c diag", color=C_BLUE, fontsize=9.5,
            fontweight="bold")
    ax.text(2.9, -0.45, "  r+c diag", color=C_AMBER, fontsize=9.5,
            fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig4_nqueens_board.png")


# ---------------------------------------------------------------------------
# Figure 5: Sudoku step — candidates eliminated by row/col/box
# ---------------------------------------------------------------------------
def fig5_sudoku_step() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.suptitle("Sudoku: one backtracking step on cell (4, 4)",
                 fontsize=13.5, fontweight="bold", y=0.99)

    # A simple partial board (not a real puzzle, just illustrative).
    # Empty cell '.' is at (4,4).
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    target_r, target_c = 4, 4

    # ---- Left: the board, highlighting target cell + its row/col/box
    ax = axes[0]
    n = 9
    box_r, box_c = (target_r // 3) * 3, (target_c // 3) * 3

    for r in range(n):
        for c in range(n):
            in_row = r == target_r
            in_col = c == target_c
            in_box = (box_r <= r < box_r + 3) and (box_c <= c < box_c + 3)
            if r == target_r and c == target_c:
                fc = C_AMBER
                alpha = 0.45
            elif in_row or in_col or in_box:
                fc = C_BLUE
                alpha = 0.10
            else:
                fc = "white"
                alpha = 1.0
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor=fc,
                                   edgecolor="#9ca3af", linewidth=0.7,
                                   alpha=alpha, zorder=1))
            v = board[r][c]
            if v != 0:
                ax.text(c + 0.5, n - 1 - r + 0.5, str(v),
                        ha="center", va="center", fontsize=13,
                        fontweight="bold", color="#111827", zorder=3)

    # Thick lines for 3x3 boxes
    for k in range(0, n + 1, 3):
        ax.plot([0, n], [k, k], color="#111827", lw=2.0, zorder=2)
        ax.plot([k, k], [0, n], color="#111827", lw=2.0, zorder=2)

    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.2, n + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Empty cell (4, 4): blue = constraints\n(row + col + 3x3 box)",
                 fontsize=11.5, color="#111827")

    # ---- Right: candidate digits 1..9 with eliminations
    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Candidates for (4, 4): strike out anything in the row/col/box",
                 fontsize=11.5, color="#111827")

    # Compute used digits in row, col, box
    used_row = {board[target_r][j] for j in range(n) if board[target_r][j] != 0}
    used_col = {board[i][target_c] for i in range(n) if board[i][target_c] != 0}
    used_box = set()
    for i in range(box_r, box_r + 3):
        for j in range(box_c, box_c + 3):
            if board[i][j] != 0:
                used_box.add(board[i][j])
    used = used_row | used_col | used_box

    # Show 1..9 in a 3x3 grid of bubbles
    pick = 5  # the chosen digit (any unused will do; 5 happens to be free? check)
    candidates = [d for d in range(1, 10) if d not in used]
    if not candidates:
        candidates = [5]
    pick = candidates[0]

    # Layout 3x3 of digit bubbles centered
    for d in range(1, 10):
        i = (d - 1) // 3
        j = (d - 1) % 3
        cx = 1.4 + j * 1.45
        cy = 4.6 - i * 1.45
        in_used = d in used
        if d == pick:
            fc = C_GREEN
            ec = C_GREEN
            txt = "white"
        elif in_used:
            fc = "#fee2e2"
            ec = C_RED
            txt = C_RED
        else:
            fc = "white"
            ec = C_GRAY
            txt = "#111827"
        ax.add_patch(Circle((cx, cy), 0.46, facecolor=fc, edgecolor=ec,
                            linewidth=1.8, zorder=2))
        ax.text(cx, cy, str(d), ha="center", va="center",
                fontsize=15, fontweight="bold", color=txt, zorder=3)
        if in_used and d != pick:
            # Strike-through line
            ax.plot([cx - 0.42, cx + 0.42], [cy - 0.42, cy + 0.42],
                    color=C_RED, lw=2.2, zorder=4)
        # source tag
        if in_used:
            tag = []
            if d in used_row: tag.append("row")
            if d in used_col: tag.append("col")
            if d in used_box: tag.append("box")
            ax.text(cx, cy - 0.85, ",".join(tag), ha="center",
                    fontsize=7.5, color=C_RED)

    # Bottom: which digit was chosen + recurse
    ax.text(3.0, 0.55,
            f"choose {pick}  ->  recurse on next empty cell  ->  "
            f"if dead end, board[4][4] = '.'  (un-choose)",
            ha="center", fontsize=10, color="#374151", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig5_sudoku_step.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_template()
    fig2_permutations_tree()
    fig3_combinations_pruning()
    fig4_nqueens_board()
    fig5_sudoku_step()
    print(f"Wrote 5 figures to:\n  {OUT_EN}\n  {OUT_ZH}")


if __name__ == "__main__":
    main()
