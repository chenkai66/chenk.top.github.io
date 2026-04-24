"""
Figure generation script for LeetCode Part 04: Sliding Window Technique.

Generates 5 figures used in both EN and ZH versions of the article. Each
figure isolates one teaching moment so the reader can read the prose and
the picture in lockstep.

Figures:
    fig1_fixed_window           Fixed-size window of width k slides across an
                                array; each step shows the running sum, with
                                the leaving / entering element highlighted to
                                explain the increment trick.
    fig2_variable_window        Variable-size window for "longest substring
                                without repeating characters". Window expands
                                until a duplicate appears, then contracts.
    fig3_expand_contract        Time-stepped trace showing how window length
                                changes over right-pointer iterations, with
                                expand / contract phases shaded.
    fig4_deque_max              Sliding window maximum using a monotonic deque.
                                Shows the deque state at each step and why the
                                front of the deque is always the answer.
    fig5_decision_tree          Decision tree: when does sliding window apply,
                                and which flavour (fixed / max-len / min-len /
                                permutation / monotonic-deque) to reach for.

Usage:
    python3 scripts/figures/leetcode/04-sliding-window.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary -- left pointer / fixed-window
C_PURPLE = "#7c3aed"   # secondary -- right pointer / contract
C_GREEN = "#10b981"    # accent / valid window
C_AMBER = "#f59e0b"    # warning / highlight
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG_SOFT = "#f8fafc"
C_RED = "#dc2626"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "leetcode" / "sliding-window"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "leetcode" / "04-滑动窗口技巧"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _no_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def _draw_cell(ax, x, y, w, h, label, *, fill, edge, text_color="white",
               font_size=12, font_weight="bold"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, str(label),
            ha="center", va="center",
            color=text_color, fontsize=font_size, fontweight=font_weight)


def _draw_bracket(ax, x_left, x_right, y, color, label_left="L", label_right="R"):
    """Draw a square bracket below cells with L/R labels."""
    ax.plot([x_left, x_left, x_right, x_right],
            [y + 0.10, y, y, y + 0.10],
            color=color, linewidth=2.0, solid_capstyle="round")
    if abs(x_left - x_right) < 0.01:
        # single cell -- show "L=R" once to avoid overlap
        ax.text(x_left, y - 0.13, "L=R", ha="center", va="top",
                color=C_BLUE, fontsize=10, fontweight="bold")
    else:
        ax.text(x_left, y - 0.13, label_left, ha="center", va="top",
                color=C_BLUE, fontsize=10, fontweight="bold")
        ax.text(x_right, y - 0.13, label_right, ha="center", va="top",
                color=C_PURPLE, fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# Figure 1 -- Fixed-size window: max sum subarray of size k
# ---------------------------------------------------------------------------
def fig1_fixed_window() -> None:
    """Show k=3 window sliding across [2, 1, 5, 1, 3, 2] and the sum trick."""
    arr = [2, 1, 5, 1, 3, 2]
    k = 3
    n = len(arr)
    # window start positions
    starts = list(range(n - k + 1))
    sums = [sum(arr[i:i + k]) for i in starts]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    _no_axis(ax)
    ax.set_xlim(-0.6, n + 0.4)
    ax.set_ylim(-0.4, len(starts) * 1.25 + 1.2)

    cell_w, cell_h = 0.9, 0.75

    ax.text(n / 2, len(starts) * 1.25 + 0.7,
            "Fixed-Size Window (k = 3): Max Sum Subarray",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C_DARK)

    # one row per slide step, top to bottom
    for row, s in enumerate(starts):
        y = (len(starts) - 1 - row) * 1.25 + 0.2
        for j, v in enumerate(arr):
            in_window = s <= j < s + k
            entering = (row > 0 and j == s + k - 1)
            leaving = (row > 0 and j == s - 1)
            if entering:
                fill, edge, txt = C_AMBER, C_AMBER, "white"
            elif leaving:
                fill, edge, txt = "#fee2e2", C_RED, C_RED
            elif in_window:
                fill, edge, txt = C_GREEN, "#047857", "white"
            else:
                fill, edge, txt = C_LIGHT, C_GRAY, C_GRAY
            _draw_cell(ax, j + 0.05, y, cell_w, cell_h, v,
                       fill=fill, edge=edge, text_color=txt, font_size=14)

        # bracket under window
        _draw_bracket(ax, s + 0.5, s + k - 0.5, y - 0.08, C_AMBER)

        # right-side: incremental update annotation
        if row == 0:
            note = f"window = {arr[s:s+k]}  →  sum = {sums[row]}"
        else:
            prev = sums[row - 1]
            cur = sums[row]
            leaving_v = arr[s - 1]
            entering_v = arr[s + k - 1]
            sign = "+" if (entering_v - leaving_v) >= 0 else ""
            note = (f"sum = {prev} − {leaving_v} + {entering_v} = {cur}   "
                    f"(Δ = {sign}{entering_v - leaving_v})")
        ax.text(n + 0.15, y + cell_h / 2, note,
                ha="left", va="center", fontsize=11.5, color=C_DARK,
                family="monospace")

        if sums[row] == max(sums):
            ax.text(-0.15, y + cell_h / 2, "*",
                    ha="right", va="center", fontsize=22,
                    color=C_AMBER, fontweight="bold")

    # legend
    legend_y = -0.05
    legend_items = [
        ("inside window", C_GREEN, "#047857"),
        ("entering →", C_AMBER, C_AMBER),
        ("leaving ←", "#fee2e2", C_RED),
    ]
    x_cur = 0.2
    for txt, fill, edge in legend_items:
        rect = Rectangle((x_cur, legend_y - 0.05), 0.35, 0.22,
                         facecolor=fill, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x_cur + 0.45, legend_y + 0.06, txt, ha="left", va="center",
                fontsize=10.5, color=C_DARK)
        x_cur += 2.0

    ax.text(n + 0.15, legend_y + 0.06,
            "*  = best window so far",
            ha="left", va="center", fontsize=10.5, color=C_DARK)

    fig.tight_layout()
    _save(fig, "fig1_fixed_window")


# ---------------------------------------------------------------------------
# Figure 2 -- Variable-size window: longest substring without repeating chars
# ---------------------------------------------------------------------------
def fig2_variable_window() -> None:
    """Trace s='abcabcbb' showing expand on unique, contract on duplicate."""
    s = "abcabcbb"
    # Each row: (left, right, action, note)
    # Actions: expand / contract
    trace = [
        (0, 0, "expand",   "add 'a'  → window 'a'"),
        (0, 1, "expand",   "add 'b'  → window 'ab'"),
        (0, 2, "expand",   "add 'c'  → window 'abc'   max=3"),
        (0, 3, "duplicate", "see 'a' again — must contract"),
        (1, 3, "contract", "drop 'a' → window 'bca'   max=3"),
        (2, 4, "expand",   "add 'b' duplicate, drop 'b' → 'cab'"),
        (5, 7, "expand",   "later: window 'b' only"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 8.4))
    _no_axis(ax)
    ax.set_xlim(-0.4, len(s) + 5.0)
    ax.set_ylim(-0.4, len(trace) * 1.20 + 1.0)

    cell_w, cell_h = 0.85, 0.75

    ax.text((len(s)) / 2, len(trace) * 1.20 + 0.55,
            "Variable-Size Window: Longest Substring Without Repeating",
            ha="center", va="center", fontsize=14.5, fontweight="bold",
            color=C_DARK)

    for row, (l, r, action, note) in enumerate(trace):
        y = (len(trace) - 1 - row) * 1.20 + 0.15
        for j, ch in enumerate(s):
            inside = l <= j <= r
            if inside:
                if action == "duplicate":
                    fill, edge, txt = "#fee2e2", C_RED, C_RED
                elif action == "contract":
                    fill, edge, txt = C_PURPLE, "#5b21b6", "white"
                else:
                    fill, edge, txt = C_GREEN, "#047857", "white"
            else:
                fill, edge, txt = C_LIGHT, C_GRAY, C_GRAY
            _draw_cell(ax, j + 0.07, y, cell_w, cell_h, ch,
                       fill=fill, edge=edge, text_color=txt, font_size=14)

        # bracket
        _draw_bracket(ax, l + 0.5, r + 0.5, y - 0.05,
                      C_AMBER if action != "duplicate" else C_RED)

        # action tag
        tag_color = {
            "expand": C_GREEN,
            "contract": C_PURPLE,
            "duplicate": C_RED,
        }[action]
        ax.text(len(s) + 0.4, y + cell_h / 2, f"[{action}]",
                ha="left", va="center", fontsize=11, color=tag_color,
                fontweight="bold", family="monospace")
        ax.text(len(s) + 2.0, y + cell_h / 2, note,
                ha="left", va="center", fontsize=11, color=C_DARK,
                family="monospace")

    fig.tight_layout()
    _save(fig, "fig2_variable_window")


# ---------------------------------------------------------------------------
# Figure 3 -- Window length over time (expand vs contract)
# ---------------------------------------------------------------------------
def fig3_expand_contract() -> None:
    """Two-panel: window length curve + which pointer moved at each tick."""
    # Trace for 'abcabcbb' from fig2 expressed at every right-pointer event.
    # We list the (right, left) state after each step.
    states = [
        (0, 0),  # add a
        (1, 0),  # add b
        (2, 0),  # add c
        (3, 1),  # add a -> contract to 1
        (4, 2),  # add b -> contract to 2
        (5, 3),  # add c -> contract to 3
        (6, 3),  # add b -> need contract? 'bcb' has dup b, contract...
        (6, 5),  # contract to 5
        (7, 6),  # add b duplicate -> contract
        (7, 7),  # contract to 7
    ]
    lengths = [r - l + 1 for r, l in states]
    xs = list(range(len(states)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 6.4),
                                   gridspec_kw={"height_ratios": [3, 1.2]},
                                   sharex=True)

    # ---- top: window length curve ----
    ax1.plot(xs, lengths, color=C_BLUE, linewidth=2.4,
             marker="o", markersize=8, markerfacecolor="white",
             markeredgewidth=2, markeredgecolor=C_BLUE, zorder=3)

    # shade expand vs contract regions
    for i in range(1, len(states)):
        prev_len, cur_len = lengths[i - 1], lengths[i]
        if cur_len > prev_len:
            color = C_GREEN
        elif cur_len < prev_len:
            color = C_PURPLE
        else:
            color = C_GRAY
        ax1.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.10, zorder=0)

    # annotate max
    max_idx = lengths.index(max(lengths))
    ax1.annotate(f"max length = {max(lengths)}",
                 xy=(max_idx, lengths[max_idx]),
                 xytext=(max_idx + 0.4, lengths[max_idx] + 0.6),
                 fontsize=11, color=C_AMBER, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.3))

    ax1.set_ylabel("window length\n(right − left + 1)", fontsize=11.5,
                   color=C_DARK)
    ax1.set_ylim(0, max(lengths) + 1.6)
    ax1.set_title("Window Length Over Time — Expand vs Contract Phases",
                  fontsize=14, fontweight="bold", color=C_DARK, pad=10)
    ax1.tick_params(axis="y", labelsize=10, colors=C_DARK)
    ax1.set_xticks(xs)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # legend on top panel
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_GREEN, alpha=0.30,
                      label="expand (right →)"),
        plt.Rectangle((0, 0), 1, 1, color=C_PURPLE, alpha=0.30,
                      label="contract (left →)"),
        plt.Rectangle((0, 0), 1, 1, color=C_GRAY, alpha=0.30,
                      label="hold"),
    ]
    ax1.legend(handles=handles, loc="upper right", frameon=False,
               fontsize=10)

    # ---- bottom: pointer-event strip ----
    _no_axis(ax2)
    ax2.set_xlim(-0.6, len(states) - 0.4)
    ax2.set_ylim(-0.5, 1.1)
    for i, (r, l) in enumerate(states):
        prev = states[i - 1] if i > 0 else (-1, 0)
        moved_right = r != prev[0]
        moved_left = l != prev[1]
        color = C_GREEN if moved_right and not moved_left else (
            C_PURPLE if moved_left and not moved_right else C_BLUE)
        sym = "R+" if moved_right and not moved_left else (
            "L+" if moved_left and not moved_right else "·")
        circ = plt.Circle((i, 0.4), 0.28, color=color, zorder=2)
        ax2.add_patch(circ)
        ax2.text(i, 0.4, sym, ha="center", va="center",
                 color="white", fontsize=10, fontweight="bold")
        ax2.text(i, -0.15, f"L={l}\nR={r}", ha="center", va="top",
                 fontsize=8.5, color=C_DARK, family="monospace")

    ax2.text(-0.55, 0.4, "step:", ha="left", va="center",
             fontsize=10, color=C_DARK, fontweight="bold")
    ax2.set_xlabel("iteration", fontsize=10.5, color=C_DARK, labelpad=2)

    fig.tight_layout()
    _save(fig, "fig3_expand_contract")


# ---------------------------------------------------------------------------
# Figure 4 -- Sliding window maximum using a monotonic deque
# ---------------------------------------------------------------------------
def fig4_deque_max() -> None:
    """LeetCode 239: nums = [1,3,-1,-3,5,3,6,7], k=3."""
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    # Hand-traced deque states (storing values; in code we'd store indices).
    # After processing nums[i] for i in 0..n-1
    # We use a monotonically decreasing deque of values for clarity.
    dq_states = [
        ([1],            None),   # i=0: push 1
        ([3],            None),   # i=1: pop 1<3, push 3
        ([3, -1],        3),      # i=2: push -1 (window full → emit 3)
        ([3, -1, -3],    3),      # i=3: push -3 (window 3,-1,-3 → emit 3)
        ([5],            5),      # i=4: 5 evicts all, emit 5
        ([5, 3],         5),      # i=5: push 3, emit 5
        ([6],            6),      # i=6: 6 evicts 3,5, emit 6
        ([7],            7),      # i=7: 7 evicts 6, emit 7
    ]
    n = len(nums)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    _no_axis(ax)
    ax.set_xlim(-0.5, n + 5.0)
    ax.set_ylim(-0.6, n * 0.9 + 1.4)

    ax.text(n / 2, n * 0.9 + 0.95,
            "Sliding Window Maximum with Monotonic Deque (k = 3)",
            ha="center", va="center", fontsize=14.5, fontweight="bold",
            color=C_DARK)

    cell_w, cell_h = 0.85, 0.7

    for i in range(n):
        y = (n - 1 - i) * 0.9 + 0.05
        # array row
        for j, v in enumerate(nums):
            l = max(0, i - k + 1)
            inside = l <= j <= i
            if j == i:
                fill, edge, txt = C_AMBER, C_AMBER, "white"
            elif inside:
                fill, edge, txt = C_GREEN, "#047857", "white"
            else:
                fill, edge, txt = C_LIGHT, C_GRAY, C_GRAY
            _draw_cell(ax, j + 0.07, y, cell_w, cell_h, v,
                       fill=fill, edge=edge, text_color=txt,
                       font_size=12)

        # deque box
        dq, emit = dq_states[i]
        dq_x = n + 0.5
        ax.text(dq_x - 0.05, y + cell_h / 2, "deque:",
                ha="right", va="center", fontsize=10.5, color=C_DARK,
                fontweight="bold")
        for di, dv in enumerate(dq):
            color = C_BLUE if di == 0 else C_PURPLE
            _draw_cell(ax, dq_x + di * 0.7, y, 0.6, cell_h, dv,
                       fill=color, edge=color, text_color="white",
                       font_size=11)
        # emit
        if emit is not None:
            ax.text(dq_x + 3.2, y + cell_h / 2,
                    f"→ max = {emit}",
                    ha="left", va="center", fontsize=11.5, color=C_AMBER,
                    fontweight="bold", family="monospace")
        else:
            ax.text(dq_x + 3.2, y + cell_h / 2,
                    "(window not full)",
                    ha="left", va="center", fontsize=10, color=C_GRAY,
                    family="monospace", style="italic")

    # caption explaining invariant
    ax.text(n / 2, -0.4,
            "Invariant: deque values are strictly decreasing from front to back. "
            "Front is always the window max.",
            ha="center", va="center", fontsize=10.5, color=C_DARK,
            style="italic")

    fig.tight_layout()
    _save(fig, "fig4_deque_max")


# ---------------------------------------------------------------------------
# Figure 5 -- Decision tree: when to use sliding window
# ---------------------------------------------------------------------------
def fig5_decision_tree() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.2))
    _no_axis(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)

    ax.text(7, 9.0, "When to Reach for Sliding Window — and Which Flavour",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C_DARK)

    def node(x, y, w, h, text, fill, edge, txt_color="white", fs=10.5):
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.5, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                color=txt_color, fontsize=fs, fontweight="bold",
                wrap=True)

    def arrow(x1, y1, x2, y2, label=None, color=C_GRAY, side="right"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->,head_width=4,head_length=6",
                            linewidth=1.4, color=color,
                            connectionstyle="arc3,rad=0.0")
        ax.add_patch(a)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx = 0.25 if side == "right" else -0.25
            ax.text(mx + dx, my, label, fontsize=10, color=color,
                    fontweight="bold",
                    ha="left" if side == "right" else "right",
                    va="center")

    # Root
    node(7, 8.0, 5.4, 0.8,
         "Contiguous subarray / substring?",
         fill="white", edge=C_DARK, txt_color=C_DARK, fs=11.5)

    # Level 2 -- left "no" branch, right "yes" branch
    node(2.3, 6.4, 3.6, 0.8,
         "No → try other patterns\n(prefix sum · DP · two pointers)",
         fill=C_LIGHT, edge=C_GRAY, txt_color=C_DARK, fs=9.5)
    node(9.0, 6.4, 4.4, 0.8,
         "Yes → window size known?",
         fill=C_BLUE, edge=C_BLUE, fs=11)

    arrow(5.5, 7.6, 3.0, 6.85, "no", C_GRAY, side="left")
    arrow(8.5, 7.6, 9.0, 6.85, "yes", C_GREEN, side="right")

    # Level 3 -- under "window size known"
    node(7.0, 4.6, 3.0, 0.8,
         "Fixed k\n(LC 643 · 567 · 438)",
         fill=C_GREEN, edge="#047857", fs=10.5)
    node(11.5, 4.6, 3.4, 0.8,
         "Variable size",
         fill=C_PURPLE, edge="#5b21b6", fs=11)

    arrow(8.2, 6.0, 7.4, 5.05, "fixed", C_GREEN, side="left")
    arrow(10.0, 6.0, 11.2, 5.05, "variable", C_PURPLE, side="right")

    # Level 4 -- under variable
    node(10.0, 2.6, 3.0, 0.85,
         "Looking for\nLONGEST?\n(LC 3 · 159 · 340)",
         fill=C_AMBER, edge="#b45309", fs=10)
    node(13.0, 2.6, 2.0, 0.85,
         "SHORTEST?\n(LC 76 · 209)",
         fill=C_RED, edge="#991b1b", fs=10)

    arrow(11.0, 4.15, 10.3, 3.05, "longest", C_AMBER, side="left")
    arrow(12.0, 4.15, 12.7, 3.05, "shortest", C_RED, side="right")

    # Special branch -- Monotonic deque (separate, lower-left)
    node(3.0, 4.6, 4.6, 0.85,
         "Need MIN / MAX inside window?\n→ Monotonic deque (LC 239)",
         fill=C_DARK, edge=C_DARK, fs=10)
    arrow(7.0, 6.0, 3.5, 5.1, "min/max in window", C_DARK, side="left")

    # Bottom strip — recipe
    node(7.0, 0.85, 12.5, 1.2,
         "Recipe:  expand right while window is OK  ·  contract left while violated\n"
         "Track O(1) state (sum / count map / deque) instead of recomputing each step",
         fill=C_BG_SOFT, edge=C_GRAY, txt_color=C_DARK, fs=10.5)

    fig.tight_layout()
    _save(fig, "fig5_decision_tree")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating sliding-window figures...")
    fig1_fixed_window()
    fig2_variable_window()
    fig3_expand_contract()
    fig4_deque_max()
    fig5_decision_tree()
    print("Done.")


if __name__ == "__main__":
    main()
