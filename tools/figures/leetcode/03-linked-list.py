"""
LeetCode Article 03: Linked List Operations - Figure Generation
================================================================

Generates 5 production-quality figures visualizing pointer manipulation
for linked list problems.

Figures:
  fig1_insert_delete.png   - Insertion and deletion (pointer rewiring)
  fig2_reverse.png         - Reverse linked list (iterative + recursive)
  fig3_floyd_cycle.png     - Floyd's tortoise and hare cycle detection
  fig4_merge.png           - Merge two sorted lists (dummy node)
  fig5_lru.png             - LRU cache (doubly linked list + hash map)

Style:
  matplotlib seaborn-v0_8-whitegrid, dpi=150
  Palette: #2563eb (blue), #7c3aed (purple), #10b981 (green), #f59e0b (amber)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# -------- Style --------
BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
AMBER = COLORS["warning"]
GRAY = COLORS["text2"]
DARK = "#111827"

DPI = 150

OUT_DIRS = [
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/leetcode/linked-list-operations",
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/leetcode/03-链表操作",
]


def save_fig(fig, name):
    """Save figure to all output directories."""
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  saved: {path}")
    plt.close(fig)


def draw_node(ax, x, y, val, color=BLUE, w=0.9, h=0.7, text_color="white",
              edge=None, alpha=1.0, fontsize=13):
    """Draw a linked list node as a rounded box with a value."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8,
        edgecolor=edge if edge else color,
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(x, y, str(val), ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def draw_arrow(ax, x1, y1, x2, y2, color=DARK, style="-|>", lw=1.6,
               connectionstyle="arc3,rad=0", alpha=1.0):
    """Draw an arrow between two points."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw,
        connectionstyle=connectionstyle, alpha=alpha,
    )
    ax.add_patch(arr)


def draw_null(ax, x, y, fontsize=12):
    """Draw a None / null sink."""
    ax.text(x, y, "None", ha="center", va="center",
            fontsize=fontsize, color=GRAY, style="italic",
            fontweight="bold")


def draw_pointer_label(ax, x, y, label, color, dy=0.55, fontsize=11):
    """Draw a labeled pointer arrow above a node."""
    ax.annotate(
        label,
        xy=(x, y + 0.35), xytext=(x, y + 0.35 + dy),
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold", color=color,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
    )


# ============================================================
# Figure 1: Insert / Delete pointer rewiring
# ============================================================
def fig1_insert_delete():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))

    # ---- Top: Insert ----
    ax = axes[0]
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Insert node 'X' between A and B  (rewire 2 pointers)",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)

    # Before
    ax.text(0.3, 2.4, "Before:", fontsize=11, fontweight="bold", color=GRAY)
    xs = [1.6, 3.4, 5.2]
    labels = ["A", "B", "C"]
    for x, lab in zip(xs, labels):
        draw_node(ax, x, 2.3, lab, color=BLUE)
    for i in range(len(xs) - 1):
        draw_arrow(ax, xs[i] + 0.45, 2.3, xs[i + 1] - 0.45, 2.3, color=DARK)
    draw_arrow(ax, xs[-1] + 0.45, 2.3, xs[-1] + 1.05, 2.3, color=DARK)
    draw_null(ax, xs[-1] + 1.4, 2.3)

    # After
    ax.text(0.3, 0.85, "After:", fontsize=11, fontweight="bold", color=GRAY)
    xs2 = [1.6, 3.4, 5.2, 7.0]
    labels2 = ["A", "X", "B", "C"]
    colors2 = [BLUE, AMBER, BLUE, BLUE]
    for x, lab, c in zip(xs2, labels2, colors2):
        draw_node(ax, x, 0.75, lab, color=c)
    # Old A->B link broken (shown faded)
    draw_arrow(ax, 1.6 + 0.45, 1.05, 3.4 + 0.45, 1.05, color=GRAY,
               style="-", lw=1.0, alpha=0.35,
               connectionstyle="arc3,rad=-0.45")
    ax.text(2.5, 1.55, "old link\nbroken", ha="center", fontsize=8.5,
            color=GRAY, style="italic")
    # New links
    draw_arrow(ax, 1.6 + 0.45, 0.75, 3.4 - 0.45, 0.75, color=GREEN, lw=2.2)
    draw_arrow(ax, 3.4 + 0.45, 0.75, 5.2 - 0.45, 0.75, color=GREEN, lw=2.2)
    draw_arrow(ax, 5.2 + 0.45, 0.75, 7.0 - 0.45, 0.75, color=DARK)
    draw_arrow(ax, 7.0 + 0.45, 0.75, 7.0 + 1.05, 0.75, color=DARK)
    draw_null(ax, 7.0 + 1.4, 0.75)

    # Code snippet
    ax.text(8.2, 1.6, "X.next = A.next   # X -> B\nA.next = X        # A -> X",
            fontsize=10, family="monospace", color=DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3f4f6",
                      edgecolor=GRAY, lw=0.8))

    # ---- Bottom: Delete ----
    ax = axes[1]
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Delete node B  (rewire 1 pointer, free B)",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)

    # Before
    ax.text(0.3, 2.4, "Before:", fontsize=11, fontweight="bold", color=GRAY)
    xs = [1.6, 3.4, 5.2]
    for x, lab in zip(xs, ["A", "B", "C"]):
        draw_node(ax, x, 2.3, lab, color=BLUE)
    for i in range(len(xs) - 1):
        draw_arrow(ax, xs[i] + 0.45, 2.3, xs[i + 1] - 0.45, 2.3, color=DARK)
    draw_arrow(ax, xs[-1] + 0.45, 2.3, xs[-1] + 1.05, 2.3, color=DARK)
    draw_null(ax, xs[-1] + 1.4, 2.3)

    # After
    ax.text(0.3, 0.85, "After:", fontsize=11, fontweight="bold", color=GRAY)
    xs2 = [1.6, 3.4, 5.2]
    # B drawn faded
    draw_node(ax, 1.6, 0.75, "A", color=BLUE)
    draw_node(ax, 3.4, 0.75, "B", color=GRAY, alpha=0.35)
    draw_node(ax, 5.2, 0.75, "C", color=BLUE)
    # New A->C link bypasses B
    draw_arrow(ax, 1.6 + 0.45, 1.1, 5.2 - 0.45, 1.1, color=GREEN, lw=2.2,
               connectionstyle="arc3,rad=-0.35")
    ax.text(3.4, 1.65, "A.next = B.next", ha="center",
            fontsize=10, family="monospace", color=GREEN, fontweight="bold")
    draw_arrow(ax, 5.2 + 0.45, 0.75, 5.2 + 1.05, 0.75, color=DARK)
    draw_null(ax, 5.2 + 1.4, 0.75)

    ax.text(8.2, 1.4, "prev.next = node.next\n# (B is now garbage-collected)",
            fontsize=10, family="monospace", color=DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3f4f6",
                      edgecolor=GRAY, lw=0.8))

    plt.tight_layout()
    save_fig(fig, "fig1_insert_delete.png")


# ============================================================
# Figure 2: Reverse linked list - iterative + recursive
# ============================================================
def fig2_reverse():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2))

    # ---- Top: iterative three-pointer ----
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Iterative reversal: three pointers (prev, curr, next) — O(1) space",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)

    # Show 3 snapshots of state
    snapshots = [
        ("Step 0  (initial)", 5.0,
         [None, 1, 2, 3], None, 0, 1),
        ("Step 1  (curr=1 reversed)", 3.0,
         [1, 2, 3], 1, 1, 2),
        ("Step 2  (curr=2 reversed)", 1.0,
         [2, 1, 3], 2, 2, 3),
    ]
    # Custom rendering per snapshot for clarity
    snapshots_data = [
        # (title_y, prev, curr, list state shown)
        ("Initial:  prev=None, curr=1", 5.0, "list: 1 -> 2 -> 3 -> None"),
        ("After step 1: 1.next=None, prev=1, curr=2", 3.0,
         "broken: None <- 1   |  remaining: 2 -> 3 -> None"),
        ("After step 2: 2.next=1, prev=2, curr=3", 1.0,
         "reversed: None <- 1 <- 2  |  remaining: 3 -> None"),
    ]

    # Step 0
    y = 5.0
    ax.text(0.2, y + 0.55, snapshots_data[0][0], fontsize=10,
            fontweight="bold", color=DARK)
    xs = [2.0, 3.6, 5.2, 6.8]
    vals = ["None", 1, 2, 3]
    cols = [GRAY, BLUE, BLUE, BLUE]
    for x, v, c in zip(xs, vals, cols):
        if v == "None":
            draw_null(ax, x, y)
        else:
            draw_node(ax, x, y, v, color=c)
    for i in range(len(xs) - 1):
        if vals[i] != "None":
            draw_arrow(ax, xs[i] + 0.45, y, xs[i + 1] - 0.45, y, color=DARK)
    draw_arrow(ax, xs[-1] + 0.45, y, xs[-1] + 1.05, y, color=DARK)
    draw_null(ax, xs[-1] + 1.4, y)
    # pointers
    ax.annotate("prev", xy=(xs[0], y + 0.35), xytext=(xs[0], y + 0.95),
                ha="center", color=PURPLE, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.8))
    ax.annotate("curr", xy=(xs[1], y + 0.35), xytext=(xs[1], y + 0.95),
                ha="center", color=AMBER, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))

    # Step 1
    y = 3.0
    ax.text(0.2, y + 0.55, snapshots_data[1][0], fontsize=10,
            fontweight="bold", color=DARK)
    # left fragment: None <- 1
    draw_null(ax, 2.0, y)
    draw_node(ax, 3.0, y, 1, color=GREEN)
    draw_arrow(ax, 3.0 - 0.45, y, 2.0 + 0.3, y, color=GREEN, lw=2.0)
    # right fragment: 2 -> 3 -> None
    draw_node(ax, 5.0, y, 2, color=BLUE)
    draw_node(ax, 6.5, y, 3, color=BLUE)
    draw_arrow(ax, 5.0 + 0.45, y, 6.5 - 0.45, y, color=DARK)
    draw_arrow(ax, 6.5 + 0.45, y, 7.6, y, color=DARK)
    draw_null(ax, 7.95, y)
    ax.annotate("prev", xy=(3.0, y + 0.35), xytext=(3.0, y + 0.95),
                ha="center", color=PURPLE, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.8))
    ax.annotate("curr", xy=(5.0, y + 0.35), xytext=(5.0, y + 0.95),
                ha="center", color=AMBER, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))

    # Step 2
    y = 1.0
    ax.text(0.2, y + 0.55, snapshots_data[2][0], fontsize=10,
            fontweight="bold", color=DARK)
    draw_null(ax, 2.0, y)
    draw_node(ax, 3.0, y, 1, color=GREEN)
    draw_node(ax, 4.5, y, 2, color=GREEN)
    draw_arrow(ax, 3.0 - 0.45, y, 2.0 + 0.3, y, color=GREEN, lw=2.0)
    draw_arrow(ax, 4.5 - 0.45, y, 3.0 + 0.45, y, color=GREEN, lw=2.0)
    draw_node(ax, 6.5, y, 3, color=BLUE)
    draw_arrow(ax, 6.5 + 0.45, y, 7.6, y, color=DARK)
    draw_null(ax, 7.95, y)
    ax.annotate("prev", xy=(4.5, y + 0.35), xytext=(4.5, y + 0.95),
                ha="center", color=PURPLE, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.8))
    ax.annotate("curr", xy=(6.5, y + 0.35), xytext=(6.5, y + 0.95),
                ha="center", color=AMBER, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))

    # legend / code
    code = ("while curr:\n"
            "    nxt = curr.next\n"
            "    curr.next = prev\n"
            "    prev = curr\n"
            "    curr = nxt")
    ax.text(9.0, 3.0, code, fontsize=9.5, family="monospace", color=DARK,
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f3f4f6",
                      edgecolor=GRAY, lw=0.8))

    # ---- Bottom: recursive call stack ----
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.set_title("Recursive reversal: unwinds from tail, rewires on the way back — O(n) stack",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)

    # Stack diagram on left
    frames = [
        ("reverseList(1)", 4.0, BLUE),
        ("reverseList(2)", 3.0, BLUE),
        ("reverseList(3)", 2.0, BLUE),
        ("base: 3.next is None  ->  return 3", 1.0, GREEN),
    ]
    for label, y, c in frames:
        ax.add_patch(FancyBboxPatch((0.2, y - 0.3), 4.4, 0.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=c, alpha=0.18, edgecolor=c, lw=1.5))
        ax.text(0.4, y, label, fontsize=10.5, color=DARK, va="center",
                fontweight="bold")
    # arrows showing call direction
    ax.annotate("", xy=(2.4, 1.45), xytext=(2.4, 3.7),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.8))
    ax.text(2.4, 2.55, "  recurse down", color=PURPLE, fontsize=9.5,
            fontweight="bold", rotation=90, va="center")

    ax.annotate("", xy=(4.55, 3.7), xytext=(4.55, 1.45),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
    ax.text(4.85, 2.55, "unwind: rewire", color=GREEN, fontsize=9.5,
            fontweight="bold", rotation=90, va="center")

    # Right: list before / after
    # Before
    y = 3.6
    ax.text(5.7, y + 0.6, "Before", fontsize=10, fontweight="bold", color=GRAY)
    for i, v in enumerate([1, 2, 3]):
        draw_node(ax, 6.0 + i * 1.3, y, v, color=BLUE)
    for i in range(2):
        draw_arrow(ax, 6.0 + i * 1.3 + 0.45, y, 6.0 + (i + 1) * 1.3 - 0.45, y,
                   color=DARK)
    draw_arrow(ax, 6.0 + 2 * 1.3 + 0.45, y, 9.2, y, color=DARK)
    draw_null(ax, 9.55, y)

    # After
    y = 1.6
    ax.text(5.7, y + 0.6, "After (returned new head = 3)", fontsize=10,
            fontweight="bold", color=GRAY)
    for i, v in enumerate([3, 2, 1]):
        draw_node(ax, 6.0 + i * 1.3, y, v, color=GREEN)
    for i in range(2):
        draw_arrow(ax, 6.0 + i * 1.3 + 0.45, y, 6.0 + (i + 1) * 1.3 - 0.45, y,
                   color=GREEN, lw=2.0)
    draw_arrow(ax, 6.0 + 2 * 1.3 + 0.45, y, 9.2, y, color=GREEN, lw=2.0)
    draw_null(ax, 9.55, y)
    # key trick
    ax.text(11.0, 2.6, "key:\nhead.next.next = head\nhead.next = None",
            fontsize=9, family="monospace", color=DARK, ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor=AMBER, lw=0.8))

    plt.tight_layout()
    save_fig(fig, "fig2_reverse.png")


# ============================================================
# Figure 3: Floyd's cycle detection (tortoise and hare)
# ============================================================
def fig3_floyd_cycle():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Left: cycle structure with annotated distances ----
    ax = axes[0]
    ax.set_xlim(-1, 9)
    ax.set_ylim(-2, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Phase 1: fast/slow meet inside the cycle",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)

    # Tail nodes leading to cycle
    tail_xs = [0.0, 1.3, 2.6]
    for i, x in enumerate(tail_xs):
        draw_node(ax, x, 1.0, chr(ord('A') + i), color=BLUE)
        if i < len(tail_xs) - 1:
            draw_arrow(ax, x + 0.45, 1.0, x + 1.3 - 0.45, 1.0, color=DARK)
    # arrow from tail to cycle entrance
    entrance = (4.0, 1.0)
    draw_arrow(ax, tail_xs[-1] + 0.45, 1.0, entrance[0] - 0.45,
               entrance[1], color=DARK)

    # Cycle (5 nodes around an ellipse)
    cx, cy, rx, ry = 5.5, 1.0, 1.7, 1.3
    n = 5
    angles = np.linspace(np.pi, np.pi - 2 * np.pi, n, endpoint=False)
    cycle_pts = [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in angles]
    cycle_labels = ["E", "F", "G", "H", "I"]
    cycle_colors = [PURPLE, BLUE, BLUE, AMBER, BLUE]  # E=entrance, H=meet
    for (x, y), lab, c in zip(cycle_pts, cycle_labels, cycle_colors):
        draw_node(ax, x, y, lab, color=c, w=0.75, h=0.6, fontsize=11)

    # Arrows along cycle
    for i in range(n):
        x1, y1 = cycle_pts[i]
        x2, y2 = cycle_pts[(i + 1) % n]
        ax.annotate("", xy=(x2 - 0.05 * (x2 - x1), y2 - 0.05 * (y2 - y1)),
                    xytext=(x1 + 0.05 * (x2 - x1), y1 + 0.05 * (y2 - y1)),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.3,
                                    connectionstyle="arc3,rad=0.05"))

    # Tortoise and hare markers
    slow_pos = cycle_pts[3]  # H
    fast_pos = cycle_pts[3]
    ax.annotate("slow + fast meet", xy=(slow_pos[0], slow_pos[1] - 0.45),
                xytext=(slow_pos[0] + 0.6, slow_pos[1] - 1.4),
                ha="center", fontsize=10, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6))
    ax.annotate("entrance", xy=(cycle_pts[0][0] - 0.4, cycle_pts[0][1]),
                xytext=(cycle_pts[0][0] - 1.6, cycle_pts[0][1] + 1.2),
                ha="center", fontsize=10, color=PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.6))

    # distances a, b, c
    ax.text(1.3, 1.6, "a", color=DARK, fontsize=12, fontweight="bold",
            ha="center")
    ax.text(4.6, -0.05, "b", color=DARK, fontsize=12, fontweight="bold",
            ha="center")
    ax.text(4.7, 2.05, "c", color=DARK, fontsize=12, fontweight="bold",
            ha="center")

    # ---- Right: Phase 2 reset ----
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Phase 2: reset slow to head, both step by 1 -> meet at entrance",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)

    # Mathematical identity box
    ax.text(0.3, 4.2,
            "When fast and slow first meet:\n"
            "   slow walked  a + b\n"
            "   fast walked  a + b + k·L   (k full extra cycles)\n"
            "Since fast = 2 · slow:\n"
            "   2(a + b) = a + b + k·L   =>   a = k·L − b = (k−1)L + c",
            fontsize=10, family="monospace", color=DARK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f4f6",
                      edgecolor=GRAY, lw=0.8),
            va="top")

    ax.text(0.3, 1.9,
            "Therefore  walking 'a' steps from head\n"
            "         = walking 'a' steps from meeting point\n"
            "Both pointers, moving 1 step at a time,\n"
            "land on the cycle entrance simultaneously.",
            fontsize=10.5, color=PURPLE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ede9fe",
                      edgecolor=PURPLE, lw=1.0),
            va="top")

    # Mini schematic
    ax.text(8.5, 3.5, "head", fontsize=10, fontweight="bold", color=DARK,
            ha="center")
    draw_node(ax, 8.5, 3.0, "h", color=BLUE, w=0.7, h=0.55, fontsize=11)
    ax.text(11.0, 3.5, "meet", fontsize=10, fontweight="bold", color=AMBER,
            ha="center")
    draw_node(ax, 11.0, 3.0, "m", color=AMBER, w=0.7, h=0.55, fontsize=11)
    ax.text(9.75, 1.6, "entrance", fontsize=10, fontweight="bold",
            color=PURPLE, ha="center")
    draw_node(ax, 9.75, 1.1, "e", color=PURPLE, w=0.7, h=0.55, fontsize=11)
    ax.annotate("", xy=(9.75 - 0.35, 1.4), xytext=(8.5 + 0.35, 2.7),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8))
    ax.annotate("", xy=(9.75 + 0.35, 1.4), xytext=(11.0 - 0.35, 2.7),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))

    plt.tight_layout()
    save_fig(fig, "fig3_floyd_cycle.png")


# ============================================================
# Figure 4: Merge two sorted lists (dummy node)
# ============================================================
def fig4_merge():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Merge two sorted lists with a dummy node "
                 "(O(m+n) time, O(1) space)",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)

    # l1 row
    l1_y = 5.6
    ax.text(0.3, l1_y, "l1:", fontsize=11, fontweight="bold", color=BLUE)
    l1_vals = [1, 2, 4]
    l1_xs = [1.2, 2.6, 4.0]
    for x, v in zip(l1_xs, l1_vals):
        draw_node(ax, x, l1_y, v, color=BLUE)
    for i in range(len(l1_xs) - 1):
        draw_arrow(ax, l1_xs[i] + 0.45, l1_y, l1_xs[i + 1] - 0.45, l1_y,
                   color=DARK)
    draw_arrow(ax, l1_xs[-1] + 0.45, l1_y, l1_xs[-1] + 1.05, l1_y, color=DARK)
    draw_null(ax, l1_xs[-1] + 1.4, l1_y)

    # l2 row
    l2_y = 4.1
    ax.text(0.3, l2_y, "l2:", fontsize=11, fontweight="bold", color=PURPLE)
    l2_vals = [1, 3, 4]
    l2_xs = [1.2, 2.6, 4.0]
    for x, v in zip(l2_xs, l2_vals):
        draw_node(ax, x, l2_y, v, color=PURPLE)
    for i in range(len(l2_xs) - 1):
        draw_arrow(ax, l2_xs[i] + 0.45, l2_y, l2_xs[i + 1] - 0.45, l2_y,
                   color=DARK)
    draw_arrow(ax, l2_xs[-1] + 0.45, l2_y, l2_xs[-1] + 1.05, l2_y, color=DARK)
    draw_null(ax, l2_xs[-1] + 1.4, l2_y)

    # Result row
    res_y = 1.8
    ax.text(0.3, res_y, "result:", fontsize=11, fontweight="bold", color=GREEN)
    # dummy
    draw_node(ax, 1.4, res_y, "D", color=GRAY, w=0.7, h=0.6, fontsize=11)
    ax.text(1.4, res_y - 0.7, "dummy", fontsize=9, color=GRAY,
            ha="center", style="italic")
    merged_vals = [1, 1, 2, 3, 4, 4]
    merged_colors = [BLUE, PURPLE, BLUE, PURPLE, BLUE, PURPLE]
    res_xs = [2.5, 3.7, 4.9, 6.1, 7.3, 8.5]
    prev_x = 1.4
    for x, v, c in zip(res_xs, merged_vals, merged_colors):
        draw_node(ax, x, res_y, v, color=c)
        draw_arrow(ax, prev_x + 0.45, res_y, x - 0.45, res_y, color=GREEN,
                   lw=2.0)
        prev_x = x
    draw_arrow(ax, prev_x + 0.45, res_y, prev_x + 1.05, res_y, color=GREEN,
               lw=2.0)
    draw_null(ax, prev_x + 1.4, res_y)
    ax.annotate("return dummy.next", xy=(2.5, res_y + 0.4),
                xytext=(2.5, res_y + 1.1),
                ha="center", color=GREEN, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))

    # Code box on right
    code = ("dummy = ListNode()\n"
            "cur = dummy\n"
            "while l1 and l2:\n"
            "    if l1.val <= l2.val:\n"
            "        cur.next = l1; l1 = l1.next\n"
            "    else:\n"
            "        cur.next = l2; l2 = l2.next\n"
            "    cur = cur.next\n"
            "cur.next = l1 or l2\n"
            "return dummy.next")
    ax.text(6.0, 5.0, code, fontsize=9.5, family="monospace", color=DARK,
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f3f4f6",
                      edgecolor=GRAY, lw=0.8))

    # Legend for colors
    blue_patch = mpatches.Patch(color=BLUE, label="from l1")
    purple_patch = mpatches.Patch(color=PURPLE, label="from l2")
    green_patch = mpatches.Patch(color=GREEN, label="newly wired link")
    ax.legend(handles=[blue_patch, purple_patch, green_patch],
              loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    save_fig(fig, "fig4_merge.png")


# ============================================================
# Figure 5: LRU cache (doubly linked list + hash map)
# ============================================================
def fig5_lru():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("LRU cache  =  hash map  +  doubly linked list  "
                 "(O(1) get and put)",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)

    # ---- Hash map on top ----
    map_y = 5.6
    ax.text(0.3, map_y + 0.5, "hash map  (key  ->  node)",
            fontsize=11, fontweight="bold", color=PURPLE)
    keys = ["1", "2", "3"]
    map_xs = [1.6, 3.6, 5.6]
    for x, k in zip(map_xs, keys):
        ax.add_patch(FancyBboxPatch((x - 0.55, map_y - 0.3), 1.1, 0.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=PURPLE, alpha=0.18,
                                    edgecolor=PURPLE, lw=1.5))
        ax.text(x, map_y, k, fontsize=12, fontweight="bold",
                color=PURPLE, ha="center", va="center")

    # ---- Doubly linked list on bottom ----
    list_y = 2.5
    # head sentinel ... node1 ... node2 ... node3 ... tail sentinel
    box_xs = [1.0, 3.2, 5.4, 7.6, 9.8]
    box_labels = ["HEAD", "k=1\nv=A", "k=2\nv=B", "k=3\nv=C", "TAIL"]
    box_colors = [GRAY, BLUE, BLUE, BLUE, GRAY]
    for x, lab, c in zip(box_xs, box_labels, box_colors):
        ax.add_patch(FancyBboxPatch((x - 0.7, list_y - 0.55), 1.4, 1.1,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=c, alpha=0.85,
                                    edgecolor=c, lw=1.8))
        ax.text(x, list_y, lab, fontsize=10.5, fontweight="bold",
                color="white", ha="center", va="center")

    # Forward arrows (next)
    for i in range(len(box_xs) - 1):
        ax.annotate("", xy=(box_xs[i + 1] - 0.7, list_y + 0.18),
                    xytext=(box_xs[i] + 0.7, list_y + 0.18),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
    # Back arrows (prev)
    for i in range(len(box_xs) - 1):
        ax.annotate("", xy=(box_xs[i] + 0.7, list_y - 0.22),
                    xytext=(box_xs[i + 1] - 0.7, list_y - 0.22),
                    arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6))

    ax.text(box_xs[0], list_y - 1.05, "most recent", fontsize=9.5,
            color=GREEN, ha="center", fontweight="bold", style="italic")
    ax.text(box_xs[-1], list_y - 1.05, "least recent (evict here)",
            fontsize=9.5, color=AMBER, ha="center", fontweight="bold",
            style="italic")

    # Map -> list pointer arrows
    for i, (mx, lx) in enumerate(zip(map_xs, box_xs[1:4])):
        ax.annotate("", xy=(lx, list_y + 0.6), xytext=(mx, map_y - 0.35),
                    arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.2,
                                    alpha=0.7,
                                    connectionstyle="arc3,rad=0.1"))

    # Operation legend on right
    ops = (
        "get(key):\n"
        "  node = map[key]\n"
        "  move_to_head(node)\n"
        "  return node.value\n"
        "\n"
        "put(key, value):\n"
        "  if key in map:\n"
        "      update + move_to_head\n"
        "  else:\n"
        "      add_to_head(new_node)\n"
        "      if size > capacity:\n"
        "          evict tail.prev"
    )
    ax.text(10.6, 5.0, ops, fontsize=9, family="monospace", color=DARK,
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f3f4f6",
                      edgecolor=GRAY, lw=0.8))

    # Color legend
    ax.text(0.3, 0.6,
            "next pointer (green)   prev pointer (amber)   "
            "map -> node (purple)",
            fontsize=9.5, color=DARK, style="italic")

    plt.tight_layout()
    save_fig(fig, "fig5_lru.png")


# ============================================================
# Main
# ============================================================
def main():
    print("Generating Article 03 figures...")
    fig1_insert_delete()
    fig2_reverse()
    fig3_floyd_cycle()
    fig4_merge()
    fig5_lru()
    print("Done.")


if __name__ == "__main__":
    main()
