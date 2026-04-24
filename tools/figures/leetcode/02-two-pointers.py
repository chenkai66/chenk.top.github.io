"""
Figure generation for LeetCode Patterns Part 02: Two Pointers.

Five figures, each targeting one specific pattern, designed to make the
*movement story* visible at a glance instead of relying on prose.

Figures
-------
    fig1_collision_pointers   Sorted-array Two Sum II walkthrough: a row of
                              numbered cells with left/right arrows that
                              converge, annotated with the sum decision at
                              each step (too small -> move left; too big ->
                              move right).
    fig2_fast_slow_cycle      Floyd's tortoise and hare on a linked list with
                              a tail cycle. Slow moves 1 step, fast moves 2
                              steps; the meeting point is highlighted and the
                              cycle structure (mu, lambda) is annotated.
    fig3_sliding_window       Variable-length sliding window over a string,
                              showing how the right pointer expands and the
                              left pointer shrinks when the constraint
                              (no duplicates) is violated. Three snapshots
                              stacked vertically.
    fig4_partition_pointers   Dutch National Flag three-way partition: three
                              regions (low / middle / high) with the lo / mid
                              / hi pointers and color blocks for 0, 1, 2.
                              Shows before, during, and final layouts.
    fig5_decision_tree        "When to use which pattern" decision tree -
                              first split on data structure (linked list vs
                              array vs string subarray), then on extra
                              constraints (sorted? cycle? frequency?). Leaf
                              nodes name the pattern and a representative
                              problem.

All figures: dpi=150, seaborn-v0_8-whitegrid, palette
{#2563eb blue, #7c3aed purple, #10b981 green, #f59e0b amber}.

Saved into both EN and ZH asset folders so the markdown can reference them
with relative paths.
"""

from __future__ import annotations

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (

    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["text2"]
C_LIGHT = "#f3f4f6"
C_TEXT = "#111827"

DPI = 150

# ----------------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------------
ROOT = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = ROOT / "source/_posts/en/leetcode/two-pointers"
ZH_DIR = ROOT / "source/_posts/zh/leetcode/02-双指针技巧"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure into both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png")


def _draw_cells(ax, values, y, *, cell_w=0.9, cell_h=0.9,
                highlight=None, highlight_color=C_AMBER, fontsize=13):
    """Draw a row of numbered cells centered around x = i * cell_w."""
    highlight = highlight or {}
    n = len(values)
    for i, v in enumerate(values):
        x = i * cell_w
        face = highlight.get(i, "white")
        edge = highlight_color if i in highlight else C_GRAY
        lw = 2.0 if i in highlight else 1.0
        ax.add_patch(
            Rectangle(
                (x - cell_w / 2, y - cell_h / 2),
                cell_w, cell_h,
                facecolor=face, edgecolor=edge, linewidth=lw,
            )
        )
        ax.text(x, y, str(v), ha="center", va="center",
                fontsize=fontsize, color=C_TEXT, fontweight="bold")
    return n


def _pointer_arrow(ax, x, y_top, label, color, *, length=0.55, fontsize=11):
    """Vertical arrow pointing down to a cell plus a label above it."""
    ax.annotate(
        "", xy=(x, y_top), xytext=(x, y_top + length),
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
    )
    ax.text(x, y_top + length + 0.12, label, ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold")


# ----------------------------------------------------------------------------
# Figure 1 - Collision pointers on a sorted array (Two Sum II walkthrough)
# ----------------------------------------------------------------------------
def fig1_collision_pointers() -> None:
    nums = [2, 3, 5, 7, 8, 11, 15]
    target = 13

    # Pointer state at each step: (left, right, sum, decision)
    steps = [
        (0, 6, 2 + 15, "sum 17 > 13  ->  right--"),
        (0, 5, 2 + 11, "sum 13 == 13  -> found!"),
    ]

    fig, axes = plt.subplots(len(steps), 1, figsize=(9, 4.0), dpi=DPI)
    if len(steps) == 1:
        axes = [axes]

    for ax, (l, r, s, decision) in zip(axes, steps):
        ax.set_xlim(-1.0, len(nums) * 0.9)
        ax.set_ylim(-0.6, 1.6)
        ax.axis("off")

        # Highlight the two pointer cells
        highlight = {l: "#dbeafe", r: "#ede9fe"}
        _draw_cells(ax, nums, y=0.0, highlight=highlight)

        # Pointer arrows
        _pointer_arrow(ax, x=l * 0.9, y_top=0.55, label=f"left={l}", color=C_BLUE)
        _pointer_arrow(ax, x=r * 0.9, y_top=0.55, label=f"right={r}", color=C_PURPLE)

        # Decision banner on the right
        ax.text(
            len(nums) * 0.9 - 0.2, 0.0,
            f"nums[{l}] + nums[{r}] = {s}\n{decision}",
            ha="right", va="center", fontsize=11,
            color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor=C_LIGHT, edgecolor=C_GRAY),
        )

    fig.suptitle(
        f"Collision Pointers: Two Sum II  (target = {target})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig1_collision_pointers")


# ----------------------------------------------------------------------------
# Figure 2 - Fast / slow pointers (Floyd's cycle detection)
# ----------------------------------------------------------------------------
def fig2_fast_slow_cycle() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=DPI)
    ax.set_xlim(-0.8, 10.0)
    ax.set_ylim(-2.4, 2.4)
    ax.axis("off")

    # Layout: 3 nodes on a tail, then 5 nodes on a cycle.
    tail = [(0, 0), (1.4, 0), (2.8, 0)]                   # mu = 3
    center = (5.6, 0)
    radius = 1.6
    cycle_n = 5
    cycle = [
        (center[0] + radius * np.cos(2 * np.pi * k / cycle_n + np.pi),
         center[1] + radius * np.sin(2 * np.pi * k / cycle_n + np.pi))
        for k in range(cycle_n)
    ]

    nodes = tail + cycle
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # Draw edges
    edges = [(i, i + 1) for i in range(len(tail) + cycle_n - 1)]
    edges.append((len(tail) + cycle_n - 1, len(tail)))  # close cycle
    for a, b in edges:
        ax_, ay_ = nodes[a]
        bx_, by_ = nodes[b]
        ax.annotate(
            "", xy=(bx_, by_), xytext=(ax_, ay_),
            arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.4,
                            shrinkA=15, shrinkB=15),
        )

    # Draw nodes
    for (x, y), name in zip(nodes, labels):
        ax.add_patch(Circle((x, y), 0.35, facecolor="white",
                            edgecolor=C_GRAY, linewidth=1.4))
        ax.text(x, y, name, ha="center", va="center",
                fontsize=11, color=C_TEXT, fontweight="bold")

    # Highlight slow and fast at meeting point (E, index 4)
    meet_idx = 4
    mx, my = nodes[meet_idx]
    ax.add_patch(Circle((mx, my), 0.42, facecolor="none",
                        edgecolor=C_AMBER, linewidth=3.0))

    # Slow pointer label (below)
    ax.annotate(
        "slow (1 step)", xy=(mx, my - 0.45), xytext=(mx - 0.6, my - 1.6),
        fontsize=11, color=C_BLUE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.6),
    )
    # Fast pointer label (above)
    ax.annotate(
        "fast (2 steps)", xy=(mx, my + 0.45), xytext=(mx + 0.6, my + 1.6),
        fontsize=11, color=C_PURPLE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.6),
    )

    # Annotate mu (tail length) and lambda (cycle length)
    ax.annotate(
        "", xy=(2.8, -0.7), xytext=(0, -0.7),
        arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=1.6),
    )
    ax.text(1.4, -1.0, r"$\mu = 3$  (tail length)", ha="center", va="top",
            fontsize=11, color=C_GREEN, fontweight="bold")

    ax.text(center[0], center[1], r"$\lambda = 5$",
            ha="center", va="center", fontsize=12, color=C_GREEN,
            fontweight="bold", style="italic")

    ax.set_title(
        "Fast / Slow Pointers: Floyd's Cycle Detection",
        fontsize=14, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    save(fig, "fig2_fast_slow_cycle")


# ----------------------------------------------------------------------------
# Figure 3 - Sliding window (longest substring without repeating chars)
# ----------------------------------------------------------------------------
def fig3_sliding_window() -> None:
    s = list("abcabcbb")
    # (left, right, note)
    snapshots = [
        (0, 2, "expand: window = 'abc', length = 3"),
        (0, 3, "duplicate 'a' at right=3 -> shrink"),
        (1, 3, "shrunk past first 'a', window = 'bca'"),
    ]

    fig, axes = plt.subplots(len(snapshots), 1, figsize=(9, 4.6), dpi=DPI)

    for ax, (l, r, note) in zip(axes, snapshots):
        ax.set_xlim(-0.8, len(s) * 0.9 + 1.5)
        ax.set_ylim(-0.6, 1.8)
        ax.axis("off")

        # Window background
        win_x0 = l * 0.9 - 0.45
        win_x1 = r * 0.9 + 0.45
        ax.add_patch(Rectangle(
            (win_x0, -0.5), win_x1 - win_x0, 1.0,
            facecolor="#fef3c7", edgecolor=C_AMBER, linewidth=1.5,
            zorder=0,
        ))

        # Cells
        highlight = {i: "white" for i in range(l, r + 1)}
        _draw_cells(ax, s, y=0.0, highlight=highlight,
                    highlight_color=C_AMBER, fontsize=14)

        # Pointers
        _pointer_arrow(ax, x=l * 0.9, y_top=0.55, label=f"left={l}", color=C_BLUE)
        _pointer_arrow(ax, x=r * 0.9, y_top=0.55, label=f"right={r}", color=C_PURPLE)

        # Note
        ax.text(
            len(s) * 0.9 + 0.4, 0.0, note,
            ha="left", va="center", fontsize=10.5, color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=C_LIGHT, edgecolor=C_GRAY),
        )

    fig.suptitle(
        "Sliding Window: Longest Substring Without Repeating Characters",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig3_sliding_window")


# ----------------------------------------------------------------------------
# Figure 4 - Partition pointers (Dutch National Flag, 3-way)
# ----------------------------------------------------------------------------
def fig4_partition_pointers() -> None:
    color_map = {0: C_BLUE, 1: C_AMBER, 2: C_PURPLE}

    initial = [2, 0, 2, 1, 1, 0, 2, 0, 1]
    middle  = [0, 0, 2, 1, 1, 2, 2, 0, 1]   # snapshot mid-run
    final   = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    snapshots = [
        ("Initial: unsorted", initial, 0, 0, len(initial) - 1),
        ("Mid-run: lo=2, mid=4, hi=5", middle, 2, 4, 5),
        ("Final: 3-way partitioned", final, 3, 5, 6),  # lo, mid, hi conceptual
    ]

    fig, axes = plt.subplots(len(snapshots), 1, figsize=(9.5, 5.2), dpi=DPI)

    for ax, (title, arr, lo, mid, hi) in zip(axes, snapshots):
        ax.set_xlim(-0.8, len(arr) * 0.9 + 0.5)
        ax.set_ylim(-1.2, 1.8)
        ax.axis("off")

        # Cells colored by value
        for i, v in enumerate(arr):
            x = i * 0.9
            ax.add_patch(Rectangle(
                (x - 0.42, -0.42), 0.84, 0.84,
                facecolor=color_map[v], edgecolor="white",
                linewidth=1.5, alpha=0.85,
            ))
            ax.text(x, 0.0, str(v), ha="center", va="center",
                    fontsize=13, color="white", fontweight="bold")

        # Pointers below the row (only for the mid snapshot - useful info)
        if "Mid-run" in title:
            for px, name, color in [
                (lo * 0.9, "lo", C_BLUE),
                (mid * 0.9, "mid", C_GREEN),
                (hi * 0.9, "hi", C_PURPLE),
            ]:
                ax.annotate(
                    "", xy=(px, -0.45), xytext=(px, -1.0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
                )
                ax.text(px, -1.1, name, ha="center", va="top",
                        fontsize=11, color=color, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold",
                     loc="left", color=C_TEXT)

    # Color legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_BLUE,   label="0 (low)"),
        plt.Rectangle((0, 0), 1, 1, color=C_AMBER,  label="1 (mid)"),
        plt.Rectangle((0, 0), 1, 1, color=C_PURPLE, label="2 (high)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Partition Pointers: Dutch National Flag (3-way partition)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig4_partition_pointers")


# ----------------------------------------------------------------------------
# Figure 5 - Decision tree: when to use which pattern
# ----------------------------------------------------------------------------
def fig5_decision_tree() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=DPI)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def node(x, y, w, h, text, *, color=C_BLUE, text_color="white",
             fontsize=10.5, weight="bold"):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            facecolor=color, edgecolor=color, linewidth=1.2,
        ))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight)

    def edge(x1, y1, x2, y2, label="", *, side="right"):
        ax.annotate(
            "", xy=(x2, y2 + 0.35), xytext=(x1, y1 - 0.35),
            arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.4),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset = 0.25 if side == "right" else -0.25
            ax.text(mx + offset, my, label, ha="center", va="center",
                    fontsize=9.5, color=C_TEXT, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor=C_GRAY,
                              linewidth=0.8))

    # Root
    node(5.5, 6.2, 4.2, 0.7, "What does the input look like?",
         color=C_GRAY, text_color="white")

    # Level 1: data structure split (3 children)
    node(2.0, 4.7, 3.0, 0.7, "Linked list",      color=C_BLUE)
    node(5.5, 4.7, 3.0, 0.7, "Array (sorted)",   color=C_PURPLE)
    node(9.0, 4.7, 3.0, 0.7, "Subarray / substring", color=C_GREEN)

    edge(5.5, 6.2, 2.0, 4.7, "node refs")
    edge(5.5, 6.2, 5.5, 4.7, "indices, sorted")
    edge(5.5, 6.2, 9.0, 4.7, "contiguous range")

    # Level 2: pattern leaves
    node(2.0, 2.8, 3.0, 0.7, "Fast / slow pointers", color=C_AMBER)
    node(5.5, 2.8, 3.0, 0.7, "Collision pointers",   color=C_AMBER)
    node(9.0, 2.8, 3.0, 0.7, "Sliding window",       color=C_AMBER)

    edge(2.0, 4.7, 2.0, 2.8)
    edge(5.5, 4.7, 5.5, 2.8)
    edge(9.0, 4.7, 9.0, 2.8)

    # Level 3: example problems
    node(2.0, 1.1, 3.4, 0.95,
         "Linked List Cycle\nFind Middle Node",
         color="white", text_color=C_TEXT, weight="normal")
    node(5.5, 1.1, 3.4, 0.95,
         "Two Sum II / 3Sum\nContainer With Most Water",
         color="white", text_color=C_TEXT, weight="normal")
    node(9.0, 1.1, 3.4, 0.95,
         "Longest Substring\nMin Window Substring",
         color="white", text_color=C_TEXT, weight="normal")

    # Borders for level 3 (since white fill blends in)
    for x in (2.0, 5.5, 9.0):
        ax.add_patch(FancyBboxPatch(
            (x - 1.7, 0.625), 3.4, 0.95,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            facecolor="white", edgecolor=C_GRAY, linewidth=1.2,
        ))
    for x, txt in [
        (2.0, "Linked List Cycle\nFind Middle Node"),
        (5.5, "Two Sum II / 3Sum\nContainer With Most Water"),
        (9.0, "Longest Substring\nMin Window Substring"),
    ]:
        ax.text(x, 1.1, txt, ha="center", va="center",
                fontsize=9.5, color=C_TEXT)

    edge(2.0, 2.8, 2.0, 1.6)
    edge(5.5, 2.8, 5.5, 1.6)
    edge(9.0, 2.8, 9.0, 1.6)

    ax.set_title(
        "When to Use Which Two-Pointer Pattern",
        fontsize=14, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    save(fig, "fig5_decision_tree")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main() -> None:
    print("Generating Two Pointers figures...")
    fig1_collision_pointers()
    fig2_fast_slow_cycle()
    fig3_sliding_window()
    fig4_partition_pointers()
    fig5_decision_tree()
    print(f"\nDone. Wrote PNGs into:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
