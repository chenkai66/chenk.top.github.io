"""
Figure generation script for LeetCode Patterns Part 07: Dynamic Programming Basics.

Generates 6 figures used in both EN and ZH versions of the article. Each figure
isolates a single conceptual idea so readers can build intuition step by step.

Figures:
    fig1_fib_recursion_tree     The recursion tree of fib(6) with overlapping
                                subproblems highlighted - the canonical "why DP
                                matters" picture.
    fig2_dp_fill_order          Side-by-side fill order for two 1D DP tables
                                (Fibonacci and Climbing Stairs), showing how
                                each cell depends on its two predecessors.
    fig3_knapsack_table         Fully filled 2D knapsack DP table for a small
                                instance, with the optimal traceback path
                                drawn on top.
    fig4_lcs_table              Filled LCS table for "ABCBDAB" vs "BDCAB",
                                arrows showing match diagonals and skip
                                transitions, plus the traced subsequence.
    fig5_space_optimization     1D vs 2D DP space usage compared visually:
                                a full 2D grid vs a single rolling row, with
                                memory-cost annotations.
    fig6_dp_categories          A taxonomy of common DP patterns - linear,
                                grid/2D, interval, state machine - each with
                                a tiny worked example and representative
                                problems.

Usage:
    python3 scripts/figures/leetcode/07-dp.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_RED = "#ef4444"
C_GRAY = "#94a3b8"
C_LIGHT = "#e2e8f0"
C_DARK = "#0f172a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.0,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Output paths (write to BOTH EN and ZH asset folders)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
OUT_DIRS = [
    ROOT / "source" / "_posts" / "en" / "leetcode" / "dynamic-programming-basics",
    ROOT / "source" / "_posts" / "zh" / "leetcode" / "07-动态规划入门",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure into every output directory and close it."""
    for d in OUT_DIRS:
        fig.savefig(d / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png x{len(OUT_DIRS)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cell(ax, x, y, w, h, text, fc, ec=None, fontsize=10, fontcolor="white",
          weight="bold"):
    ec = ec or fc
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=fontcolor, weight=weight)


def _arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=1.4, style="-|>", mut=12):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mut,
        color=color, linewidth=lw,
    ))


# ---------------------------------------------------------------------------
# Figure 1: Fibonacci recursion tree (overlapping subproblems)
# ---------------------------------------------------------------------------
def fig1_fib_recursion_tree():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_axis_off()
    ax.set_title(
        "Figure 1. fib(6) Recursion Tree - Same Subproblems Recomputed",
        loc="left", pad=12,
    )

    # (label, x, y, level)
    nodes = [
        ("fib(6)", 6.0, 7.2, 0),
        ("fib(5)", 3.0, 6.0, 1),
        ("fib(4)", 9.0, 6.0, 1),
        ("fib(4)", 1.5, 4.8, 2),
        ("fib(3)", 4.5, 4.8, 2),
        ("fib(3)", 7.5, 4.8, 2),
        ("fib(2)", 10.5, 4.8, 2),
        ("fib(3)", 0.6, 3.6, 3),
        ("fib(2)", 2.4, 3.6, 3),
        ("fib(2)", 3.7, 3.6, 3),
        ("fib(1)", 5.3, 3.6, 3),
        ("fib(2)", 6.7, 3.6, 3),
        ("fib(1)", 8.3, 3.6, 3),
        ("fib(1)", 9.9, 3.6, 3),
        ("fib(0)", 11.1, 3.6, 3),
        ("fib(2)", 0.2, 2.4, 4),
        ("fib(1)", 1.1, 2.4, 4),
        ("fib(1)", 2.0, 2.4, 4),
        ("fib(0)", 2.8, 2.4, 4),
        ("fib(1)", 3.4, 2.4, 4),
        ("fib(0)", 4.1, 2.4, 4),
        ("fib(1)", 6.4, 2.4, 4),
        ("fib(0)", 7.1, 2.4, 4),
        ("fib(1)", -0.2, 1.2, 5),
        ("fib(0)", 0.6, 1.2, 5),
    ]

    # Highlight repeats: fib(4) appears 1x, fib(3) 3x, fib(2) 5x
    repeat_colors = {
        "fib(3)": C_AMBER,
        "fib(2)": C_RED,
        "fib(1)": C_GREEN,
        "fib(0)": C_GREEN,
        "fib(4)": C_PURPLE,
        "fib(5)": C_BLUE,
        "fib(6)": C_BLUE,
    }

    # Edges: parent index -> children indices
    edges = [
        (0, 1), (0, 2),
        (1, 3), (1, 4),
        (2, 5), (2, 6),
        (3, 7), (3, 8),
        (4, 9), (4, 10),
        (5, 11), (5, 12),
        (6, 13), (6, 14),
        (7, 15), (7, 16),
        (8, 17), (8, 18),
        (9, 19), (9, 20),
        (11, 21), (11, 22),
        (15, 23), (15, 24),
    ]

    # Draw edges
    for p, c in edges:
        ax.plot([nodes[p][1], nodes[c][1]],
                [nodes[p][2] - 0.18, nodes[c][2] + 0.18],
                color=C_GRAY, linewidth=0.9, zorder=1)

    # Draw nodes
    for label, x, y, _ in nodes:
        color = repeat_colors.get(label, C_GRAY)
        ax.add_patch(Circle((x, y), 0.28, facecolor=color, edgecolor=color,
                            linewidth=1, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                color="white", weight="bold", zorder=3)

    # Legend / counts
    counts = [
        ("fib(2) computed 5 times", C_RED),
        ("fib(3) computed 3 times", C_AMBER),
        ("fib(1) / fib(0) - base cases", C_GREEN),
    ]
    for i, (txt, c) in enumerate(counts):
        ax.add_patch(Rectangle((0.3, 0.2 + i * 0.5), 0.3, 0.3, facecolor=c,
                               edgecolor=c))
        ax.text(0.7, 0.35 + i * 0.5, txt, fontsize=10, va="center",
                color=C_DARK)

    ax.text(11.8, 0.5,
            "Pure recursion: O(2^n) calls\nMemoized: O(n) calls\n(each subproblem solved once)",
            ha="right", va="center", fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9",
                      edgecolor=C_GRAY))

    save(fig, "fig1_fib_recursion_tree")


# ---------------------------------------------------------------------------
# Figure 2: 1D DP fill order (Fibonacci + Climbing Stairs)
# ---------------------------------------------------------------------------
def fig2_dp_fill_order():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.suptitle("Figure 2. Bottom-Up Fill Order: dp[i] = dp[i-1] + dp[i-2]",
                 fontsize=13, fontweight="bold", x=0.05, ha="left", y=1.02)

    # ---- Left: Fibonacci ----
    ax = axes[0]
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-1.2, 4)
    ax.set_axis_off()
    ax.set_title("Fibonacci  -  fib(0..8)", loc="left", pad=8, fontsize=11)

    fib_vals = [0, 1, 1, 2, 3, 5, 8, 13, 21]
    cell_w, cell_h = 0.9, 0.9
    y0 = 1.2
    for i, v in enumerate(fib_vals):
        if i < 2:
            fc = C_GREEN  # base case
        else:
            fc = C_BLUE
        _cell(ax, i * cell_w, y0, cell_w, cell_h, str(v), fc, fontsize=12)
        ax.text(i * cell_w + cell_w / 2, y0 - 0.35, f"i={i}", ha="center",
                fontsize=9, color=C_DARK)

    # Highlight transition: dp[5] = dp[4] + dp[3]
    src1_x = 3 * cell_w + cell_w / 2
    src2_x = 4 * cell_w + cell_w / 2
    tgt_x = 5 * cell_w + cell_w / 2
    _arrow(ax, src1_x, y0 + cell_h + 0.05, tgt_x - 0.1,
           y0 + cell_h + 0.6, color=C_AMBER, lw=1.6)
    _arrow(ax, src2_x, y0 + cell_h + 0.05, tgt_x - 0.05,
           y0 + cell_h + 0.6, color=C_AMBER, lw=1.6)
    ax.text(tgt_x, y0 + cell_h + 0.95, "dp[5] = dp[3] + dp[4] = 5",
            ha="center", fontsize=10, color=C_AMBER, weight="bold")

    ax.text(0, -0.6,
            "Base cases (green): dp[0]=0, dp[1]=1.  Each later cell needs only its two predecessors.",
            fontsize=9.5, color=C_DARK)

    # ---- Right: Climbing Stairs ----
    ax = axes[1]
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-1.2, 4)
    ax.set_axis_off()
    ax.set_title("Climbing Stairs  -  ways to reach step n", loc="left",
                 pad=8, fontsize=11)

    cs_vals = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    for i, v in enumerate(cs_vals):
        fc = C_GREEN if i < 2 else C_PURPLE
        _cell(ax, i * cell_w, y0, cell_w, cell_h, str(v), fc, fontsize=12)
        ax.text(i * cell_w + cell_w / 2, y0 - 0.35, f"n={i}", ha="center",
                fontsize=9, color=C_DARK)

    # Show transition for n=4
    src1_x = 2 * cell_w + cell_w / 2
    src2_x = 3 * cell_w + cell_w / 2
    tgt_x = 4 * cell_w + cell_w / 2
    _arrow(ax, src1_x, y0 + cell_h + 0.05, tgt_x - 0.1,
           y0 + cell_h + 0.6, color=C_AMBER, lw=1.6)
    _arrow(ax, src2_x, y0 + cell_h + 0.05, tgt_x - 0.05,
           y0 + cell_h + 0.6, color=C_AMBER, lw=1.6)
    ax.text(tgt_x, y0 + cell_h + 0.95, "ways(4) = ways(2) + ways(3) = 5",
            ha="center", fontsize=10, color=C_AMBER, weight="bold")

    ax.text(0, -0.6,
            "Same recurrence, different base cases: dp[0]=dp[1]=1 (one way each).",
            fontsize=9.5, color=C_DARK)

    save(fig, "fig2_dp_fill_order")


# ---------------------------------------------------------------------------
# Figure 3: 0/1 Knapsack 2D DP table with traceback
# ---------------------------------------------------------------------------
def fig3_knapsack_table():
    # Items: weights, values; capacity W=5
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    W = 5
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if w >= weights[i - 1]:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # Traceback
    traceback = set()
    i, w = n, W
    while i > 0 and w > 0:
        traceback.add((i, w))
        if dp[i][w] != dp[i - 1][w]:
            w -= weights[i - 1]
        i -= 1
    traceback.add((0, 0))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(-2, W + 2)
    ax.set_ylim(-1.5, n + 2)
    ax.set_axis_off()
    ax.set_title(
        "Figure 3. 0/1 Knapsack DP Table  -  weights=[2,3,4,5], values=[3,4,5,6], W=5",
        loc="left", pad=12,
    )

    cw, ch = 0.85, 0.7

    # Column headers (capacity)
    for w in range(W + 1):
        ax.text(w * cw + cw / 2, (n + 1) * ch + 0.35, f"w={w}",
                ha="center", fontsize=9, color=C_DARK, weight="bold")
    ax.text(-1.0, (n + 1) * ch + 0.35, "items \\ cap.", fontsize=9,
            color=C_DARK, weight="bold", ha="center")

    # Cells
    for i in range(n + 1):
        # Row header
        if i == 0:
            row_label = "no item"
        else:
            row_label = f"i={i} (w={weights[i-1]}, v={values[i-1]})"
        ax.text(-1.0, (n - i) * ch + ch / 2, row_label, ha="center",
                fontsize=9, color=C_DARK)

        for w in range(W + 1):
            val = dp[i][w]
            if (i, w) in traceback and i > 0:
                fc = C_AMBER
                fontc = "white"
            elif i == 0 or w == 0:
                fc = C_GREEN
                fontc = "white"
            else:
                fc = "#dbeafe"
                fontc = C_DARK
            _cell(ax, w * cw, (n - i) * ch, cw, ch, str(val), fc,
                  fontsize=10, fontcolor=fontc)

    # Arrow marking optimal answer
    opt_x = W * cw + cw / 2
    opt_y = 0 * ch + ch / 2
    _arrow(ax, opt_x + 1.5, opt_y - 0.3, opt_x + 0.4, opt_y, color=C_RED,
           lw=2)
    ax.text(opt_x + 1.6, opt_y - 0.4,
            f"answer = dp[{n}][{W}] = {dp[n][W]}",
            fontsize=10, color=C_RED, weight="bold", va="top")

    # Recurrence box
    ax.text(0, -1.0,
            "dp[i][w] = max( dp[i-1][w],  dp[i-1][w - weight[i]] + value[i] )    "
            "amber = traceback of chosen items (item 2 + item 4)",
            fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor=C_AMBER))

    save(fig, "fig3_knapsack_table")


# ---------------------------------------------------------------------------
# Figure 4: LCS table for "ABCBDAB" vs "BDCAB"
# ---------------------------------------------------------------------------
def fig4_lcs_table():
    s1 = "ABCBDAB"
    s2 = "BDCAB"
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    src = [[None] * (n + 1) for _ in range(m + 1)]  # 'D','U','L'
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                src[i][j] = "D"
            elif dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                src[i][j] = "U"
            else:
                dp[i][j] = dp[i][j - 1]
                src[i][j] = "L"

    # Traceback path cells
    path = set()
    i, j = m, n
    lcs_chars = []
    while i > 0 and j > 0:
        path.add((i, j))
        if src[i][j] == "D":
            lcs_chars.append(s1[i - 1])
            i -= 1
            j -= 1
        elif src[i][j] == "U":
            i -= 1
        else:
            j -= 1
    lcs_str = "".join(reversed(lcs_chars))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-2, n + 2)
    ax.set_ylim(-1.5, m + 3)
    ax.set_axis_off()
    ax.set_title(
        f'Figure 4. LCS Table  -  "{s1}" vs "{s2}"  ->  LCS = "{lcs_str}" (len {dp[m][n]})',
        loc="left", pad=12,
    )

    cw, ch = 0.8, 0.7
    # Column headers
    headers = [""] + list(s2)
    for j, ch_label in enumerate(headers):
        ax.text(j * cw + cw / 2, (m + 1) * ch + 0.35, ch_label,
                ha="center", fontsize=11, color=C_DARK, weight="bold")
    # Row headers
    row_labels = [""] + list(s1)
    for i, ch_label in enumerate(row_labels):
        ax.text(-0.6, (m - i) * ch + ch / 2, ch_label, ha="center",
                fontsize=11, color=C_DARK, weight="bold")

    # Cells
    for i in range(m + 1):
        for j in range(n + 1):
            val = dp[i][j]
            if (i, j) in path:
                if src[i][j] == "D":
                    fc = C_AMBER
                    fontc = "white"
                else:
                    fc = "#fed7aa"
                    fontc = C_DARK
            elif i == 0 or j == 0:
                fc = C_GREEN
                fontc = "white"
            else:
                fc = "#ede9fe"
                fontc = C_DARK
            _cell(ax, j * cw, (m - i) * ch, cw, ch, str(val), fc,
                  fontsize=10, fontcolor=fontc)

    ax.text(0, -1.0,
            "dp[i][j] = dp[i-1][j-1]+1 if s1[i-1]==s2[j-1] else max(dp[i-1][j], dp[i][j-1])    "
            "amber = match (diagonal step), peach = skip",
            fontsize=9.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor=C_AMBER))

    save(fig, "fig4_lcs_table")


# ---------------------------------------------------------------------------
# Figure 5: 1D vs 2D DP space optimization
# ---------------------------------------------------------------------------
def fig5_space_optimization():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Figure 5. Space Optimization: 2D Table -> Rolling 1D Array",
        fontsize=13, fontweight="bold", x=0.05, ha="left", y=1.0,
    )

    rows, cols = 6, 8

    # ---- Left: full 2D table ----
    ax = axes[0]
    ax.set_xlim(-1, cols + 1)
    ax.set_ylim(-1.5, rows + 1)
    ax.set_axis_off()
    ax.set_title(f"Naive: 2D dp[{rows}][{cols}]  -  O(m * n) memory",
                 loc="left", pad=8, fontsize=11)

    cw, ch = 0.7, 0.7
    for i in range(rows):
        for j in range(cols):
            # highlight current row in amber, prev row in blue
            if i == 3:
                fc = C_AMBER
                fontc = "white"
            elif i == 2:
                fc = C_BLUE
                fontc = "white"
            else:
                fc = C_LIGHT
                fontc = C_DARK
            _cell(ax, j * cw, (rows - 1 - i) * ch, cw, ch, "", fc,
                  fontsize=8, fontcolor=fontc)

    ax.text(0, -0.8, "amber = row being filled,  blue = previous row used,  gray = wasted memory",
            fontsize=9, color=C_DARK)
    ax.text(cols * cw / 2, rows * ch + 0.4,
            f"Total cells stored: {rows * cols}",
            ha="center", fontsize=10, color=C_DARK, weight="bold")

    # ---- Right: rolling 1D array ----
    ax = axes[1]
    ax.set_xlim(-1, cols + 1)
    ax.set_ylim(-1.5, rows + 1)
    ax.set_axis_off()
    ax.set_title(f"Optimized: 1D dp[{cols}] (rolling)  -  O(n) memory",
                 loc="left", pad=8, fontsize=11)

    # Show the same table but with previous rows faded out and only one
    # active row highlighted
    for i in range(rows):
        for j in range(cols):
            if i == 3:
                fc = C_AMBER
                fontc = "white"
            else:
                fc = "#f8fafc"
                fontc = "#cbd5e1"
            _cell(ax, j * cw, (rows - 1 - i) * ch, cw, ch, "", fc,
                  fontsize=8, fontcolor=fontc)

    # Annotation arrow showing reuse
    _arrow(ax, cols * cw + 0.3, (rows - 1 - 3) * ch + ch / 2,
           cols * cw + 1.2, (rows - 1 - 3) * ch + ch / 2,
           color=C_RED, lw=1.6)
    ax.text(cols * cw + 1.3, (rows - 1 - 3) * ch + ch / 2,
            "overwrite\nin place", fontsize=9, color=C_RED, weight="bold",
            va="center")

    ax.text(0, -0.8,
            "Only the current row exists in memory; previous values are overwritten as we sweep.",
            fontsize=9, color=C_DARK)
    ax.text(cols * cw / 2, rows * ch + 0.4,
            f"Total cells stored: {cols}  ({rows}x reduction)",
            ha="center", fontsize=10, color=C_GREEN, weight="bold")

    save(fig, "fig5_space_optimization")


# ---------------------------------------------------------------------------
# Figure 6: DP problem categories
# ---------------------------------------------------------------------------
def fig6_dp_categories():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.set_axis_off()
    ax.set_title("Figure 6. A Map of Common DP Patterns", loc="left", pad=12)

    categories = [
        {
            "title": "Linear DP",
            "state": "dp[i] - state over a 1D index",
            "examples": ["Climbing Stairs", "House Robber", "LIS", "Max Subarray"],
            "color": C_BLUE,
            "x": 0.4, "y": 4.2,
        },
        {
            "title": "2D / Grid DP",
            "state": "dp[i][j] - over two indices or two strings",
            "examples": ["LCS", "Edit Distance", "Unique Paths", "Min Path Sum"],
            "color": C_PURPLE,
            "x": 6.7, "y": 4.2,
        },
        {
            "title": "Interval DP",
            "state": "dp[i][j] - over interval [i, j]",
            "examples": ["Burst Balloons", "Matrix Chain Mult.",
                         "Longest Palindromic Subseq.", "Stone Merge"],
            "color": C_AMBER,
            "x": 0.4, "y": 0.3,
        },
        {
            "title": "State Machine DP",
            "state": "dp[i][k] - position + extra state (held/sold, k uses, parity ...)",
            "examples": ["Stock with Cooldown", "Stock with k Transactions",
                         "House Robber II / III", "Knapsack 0/1"],
            "color": C_GREEN,
            "x": 6.7, "y": 0.3,
        },
    ]

    box_w, box_h = 6.0, 3.4
    for cat in categories:
        x, y = cat["x"], cat["y"]
        c = cat["color"]
        # Container
        ax.add_patch(FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="white", edgecolor=c, linewidth=2,
        ))
        # Header bar
        ax.add_patch(FancyBboxPatch(
            (x, y + box_h - 0.7), box_w, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=c, edgecolor=c,
        ))
        ax.text(x + 0.2, y + box_h - 0.35, cat["title"], fontsize=12,
                color="white", weight="bold", va="center")

        # State definition
        ax.text(x + 0.2, y + box_h - 1.1, cat["state"], fontsize=10,
                color=C_DARK, va="top", style="italic")

        # Examples
        ax.text(x + 0.2, y + box_h - 1.7, "Examples:", fontsize=10,
                color=C_DARK, weight="bold", va="top")
        for k, ex in enumerate(cat["examples"]):
            ax.text(x + 0.4, y + box_h - 2.05 - k * 0.32, f"-  {ex}",
                    fontsize=9.5, color=C_DARK, va="top")

    # Center caption
    ax.text(6.5, 7.6,
            "Pick the smallest state that captures every decision still ahead.",
            ha="center", fontsize=11, color=C_DARK, style="italic",
            weight="bold")

    save(fig, "fig6_dp_categories")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures for LeetCode 07: Dynamic Programming Basics ...")
    fig1_fib_recursion_tree()
    fig2_dp_fill_order()
    fig3_knapsack_table()
    fig4_lcs_table()
    fig5_space_optimization()
    fig6_dp_categories()
    print("Done.")
