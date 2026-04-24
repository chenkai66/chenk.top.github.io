"""
Figure generation script for LeetCode Patterns Article 05: Binary Search.

Generates 5 figures used in both EN and ZH versions of the article. Each
figure teaches a single concept cleanly and is rendered to BOTH article asset
folders so markdown references stay in sync across languages.

Figures:
    fig1_search_space           Halving the search space across iterations
                                of standard binary search.
    fig2_left_right_bound       Side-by-side trace of the left-bound and
                                right-bound templates on an array with
                                duplicates.
    fig3_rotated_array          Search in a rotated sorted array, showing
                                which half is sorted at each iteration.
    fig4_answer_space           Binary search on the answer (Capacity To
                                Ship Packages): feasibility curve and the
                                converging bracket.
    fig5_template_comparison    Closed [l, r] vs half-open [l, r) interval
                                semantics across the three templates.

Usage:
    python3 scripts/figures/leetcode/05-binary-search.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

COLOR_BLUE = "#2563eb"
COLOR_PURPLE = "#7c3aed"
COLOR_GREEN = "#10b981"
COLOR_AMBER = "#f59e0b"
COLOR_GRAY = "#94a3b8"
COLOR_DARK = "#1e293b"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "leetcode" / "binary-search"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "leetcode" / "05-二分查找"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (EN_DIR, ZH_DIR):
        fig.savefig(out_dir / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_array(ax, values, x0, y, *, cell_w=0.55, cell_h=0.55,
                highlight=None, faded=None, found=None,
                label_idx=True, fontsize=10):
    """Draw an array as a row of square cells centered at y."""
    highlight = highlight or {}
    faded = set(faded or [])
    n = len(values)
    for i, v in enumerate(values):
        x = x0 + i * cell_w
        if i == found:
            face, edge, txt_color = COLOR_GREEN, COLOR_DARK, "white"
        elif i in highlight:
            face = highlight[i]
            edge = COLOR_DARK
            txt_color = "white"
        elif i in faded:
            face, edge, txt_color = "#f1f5f9", "#cbd5e1", "#94a3b8"
        else:
            face, edge, txt_color = "white", COLOR_DARK, COLOR_DARK
        rect = Rectangle((x, y - cell_h / 2), cell_w, cell_h,
                         facecolor=face, edgecolor=edge, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, y, str(v),
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=txt_color)
        if label_idx:
            idx_color = "#94a3b8" if i in faded else "#64748b"
            ax.text(x + cell_w / 2, y - cell_h / 2 - 0.18, str(i),
                    ha="center", va="top", fontsize=8, color=idx_color)
    return x0 + n * cell_w


# ---------------------------------------------------------------------------
# Figure 1: Halving the search space
# ---------------------------------------------------------------------------

def fig1_search_space() -> None:
    """Show four iterations of binary search collapsing the search window."""
    nums = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    target = 23
    n = len(nums)

    # Pre-compute the trace.
    trace = []
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        trace.append((left, right, mid))
        if nums[mid] == target:
            break
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    fig, axes = plt.subplots(len(trace), 1, figsize=(11, 1.05 * len(trace) + 1.0))
    if len(trace) == 1:
        axes = [axes]

    cell_w = 0.55
    x0 = 0.4

    for ax, (l, r, m) in zip(axes, trace):
        ax.set_xlim(0, n * cell_w + 1.0)
        ax.set_ylim(-0.7, 0.8)
        ax.set_aspect("equal")
        ax.axis("off")

        faded = [i for i in range(n) if i < l or i > r]
        if nums[m] == target:
            highlight = {}
            found = m
        else:
            color = COLOR_AMBER
            highlight = {m: color}
            found = None
        _draw_array(ax, nums, x0, 0,
                    cell_w=cell_w, cell_h=cell_w,
                    highlight=highlight, faded=faded, found=found)

        # Window bracket.
        bx0 = x0 + l * cell_w
        bx1 = x0 + (r + 1) * cell_w
        ax.plot([bx0, bx1], [cell_w / 2 + 0.18] * 2,
                color=COLOR_BLUE, lw=2.0)
        ax.plot([bx0, bx0], [cell_w / 2 + 0.10, cell_w / 2 + 0.26],
                color=COLOR_BLUE, lw=2.0)
        ax.plot([bx1, bx1], [cell_w / 2 + 0.10, cell_w / 2 + 0.26],
                color=COLOR_BLUE, lw=2.0)
        ax.text((bx0 + bx1) / 2, cell_w / 2 + 0.40,
                f"window size = {r - l + 1}",
                ha="center", va="bottom", fontsize=9, color=COLOR_BLUE)

        status = (f"found  nums[{m}] = {target}"
                  if nums[m] == target
                  else f"nums[{m}]={nums[m]} {'<' if nums[m] < target else '>'} {target}"
                       f"   →   {'left' if nums[m] < target else 'right'} = "
                       f"{m + 1 if nums[m] < target else m - 1}")
        ax.text(x0 - 0.15, 0, f"l={l}\nr={r}\nm={m}",
                ha="right", va="center", fontsize=9, color=COLOR_DARK)
        ax.text(x0 + n * cell_w + 0.05, 0, status,
                ha="left", va="center", fontsize=10,
                color=COLOR_GREEN if nums[m] == target else COLOR_DARK)

    fig.suptitle(f"Binary search for target = {target}: search window halves each step",
                 fontsize=12, fontweight="bold", color=COLOR_DARK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig1_search_space.png")


# ---------------------------------------------------------------------------
# Figure 2: Left-bound vs right-bound template
# ---------------------------------------------------------------------------

def fig2_left_right_bound() -> None:
    """Trace left-bound and right-bound on [1, 2, 5, 5, 5, 5, 8, 9], target=5."""
    nums = [1, 2, 5, 5, 5, 5, 8, 9]
    target = 5
    n = len(nums)

    def _trace_left():
        steps = []
        l, r = 0, n
        while l < r:
            m = (l + r) // 2
            steps.append((l, r, m, "<" if nums[m] < target else ">="))
            if nums[m] < target:
                l = m + 1
            else:
                r = m
        return steps, l

    def _trace_right():
        steps = []
        l, r = 0, n
        while l < r:
            m = (l + r) // 2
            steps.append((l, r, m, "<=" if nums[m] <= target else ">"))
            if nums[m] <= target:
                l = m + 1
            else:
                r = m
        return steps, l - 1

    left_steps, left_ans = _trace_left()
    right_steps, right_ans = _trace_right()

    rows = max(len(left_steps), len(right_steps)) + 1  # +1 for result row
    fig, axes = plt.subplots(rows, 2, figsize=(13, 0.95 * rows + 1.0))

    cell_w = 0.6
    x0 = 0.2

    def _draw_panel(ax, steps, ans, label):
        # Empty fill rows so both columns have equal height.
        return None

    for col, (steps, ans, title, ans_color) in enumerate([
        (left_steps, left_ans, "Left bound  →  first index ≥ target", COLOR_BLUE),
        (right_steps, right_ans, "Right bound  →  last index ≤ target", COLOR_PURPLE),
    ]):
        for r_i in range(rows):
            ax = axes[r_i, col]
            ax.set_xlim(0, n * cell_w + 1.5)
            ax.set_ylim(-0.7, 0.8)
            ax.set_aspect("equal")
            ax.axis("off")
            if r_i == 0:
                ax.text((n * cell_w) / 2 + x0, 0.55, title,
                        ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color=ans_color)
            if r_i < len(steps):
                l, r, m, cmp = steps[r_i]
                faded = [i for i in range(n) if i < l or i >= r]
                _draw_array(ax, nums, x0, 0, cell_w=cell_w, cell_h=cell_w,
                            highlight={m: ans_color}, faded=faded)
                ax.text(x0 - 0.15, 0, f"l={l}\nr={r}\nm={m}",
                        ha="right", va="center", fontsize=9)
                ax.text(x0 + n * cell_w + 0.05, 0,
                        f"nums[{m}]={nums[m]} {cmp} {target}",
                        ha="left", va="center", fontsize=9)
            elif r_i == len(steps):
                _draw_array(ax, nums, x0, 0, cell_w=cell_w, cell_h=cell_w,
                            found=ans)
                ax.text(x0 + n * cell_w + 0.05, 0,
                        f"answer index = {ans}",
                        ha="left", va="center", fontsize=10,
                        fontweight="bold", color=ans_color)

    fig.suptitle(f"nums = {nums},  target = {target}  ·  same loop, different '<' vs '<='",
                 fontsize=12, fontweight="bold", color=COLOR_DARK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig2_left_right_bound.png")


# ---------------------------------------------------------------------------
# Figure 3: Search in rotated sorted array
# ---------------------------------------------------------------------------

def fig3_rotated_array() -> None:
    """Trace search for target=0 in [4,5,6,7,0,1,2]."""
    nums = [4, 5, 6, 7, 0, 1, 2]
    target = 0
    n = len(nums)

    # Trace.
    trace = []
    l, r = 0, n - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            trace.append((l, r, m, "found", None))
            break
        if nums[l] <= nums[m]:
            sorted_side = "left"
            in_range = nums[l] <= target < nums[m]
        else:
            sorted_side = "right"
            in_range = nums[m] < target <= nums[r]
        trace.append((l, r, m, sorted_side, in_range))
        if sorted_side == "left":
            if in_range:
                r = m - 1
            else:
                l = m + 1
        else:
            if in_range:
                l = m + 1
            else:
                r = m - 1

    rows = len(trace)
    fig, axes = plt.subplots(rows, 1, figsize=(11.5, 1.35 * rows + 1.0))
    if rows == 1:
        axes = [axes]

    cell_w = 0.7
    x0 = 0.6

    for ax, step in zip(axes, trace):
        l, r, m, side, in_range = step
        ax.set_xlim(0, n * cell_w + 2.2)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        faded = [i for i in range(n) if i < l or i > r]
        if side == "found":
            _draw_array(ax, nums, x0, 0, cell_w=cell_w, cell_h=cell_w,
                        faded=faded, found=m, fontsize=11)
            ax.text(x0 + n * cell_w + 0.1, 0,
                    f"nums[{m}] = {target}  →  found",
                    ha="left", va="center", fontsize=10,
                    fontweight="bold", color=COLOR_GREEN)
        else:
            highlight = {m: COLOR_AMBER}
            _draw_array(ax, nums, x0, 0, cell_w=cell_w, cell_h=cell_w,
                        highlight=highlight, faded=faded, fontsize=11)
            # Mark which half is sorted.
            if side == "left":
                xs0 = x0 + l * cell_w
                xs1 = x0 + m * cell_w
                colour = COLOR_BLUE
                msg = f"left half [{l},{m - 1}] sorted"
            else:
                xs0 = x0 + (m + 1) * cell_w
                xs1 = x0 + (r + 1) * cell_w
                colour = COLOR_PURPLE
                msg = f"right half [{m + 1},{r}] sorted"
            ax.plot([xs0, xs1], [-cell_w / 2 - 0.20] * 2,
                    color=colour, lw=2.4)
            ax.text((xs0 + xs1) / 2, -cell_w / 2 - 0.40, msg,
                    ha="center", va="top", fontsize=9, color=colour)
            decision = ("target in sorted half  →  search there"
                        if in_range
                        else "target NOT in sorted half  →  search the other half")
            ax.text(x0 + n * cell_w + 0.1, 0.05,
                    f"l={l}, r={r}, m={m}\n{decision}",
                    ha="left", va="center", fontsize=9, color=COLOR_DARK)

    fig.suptitle(f"Rotated array  {nums},  target = {target}  ·  one half is always sorted",
                 fontsize=12, fontweight="bold", color=COLOR_DARK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig3_rotated_array.png")


# ---------------------------------------------------------------------------
# Figure 4: Binary search on the answer (Capacity To Ship Packages)
# ---------------------------------------------------------------------------

def fig4_answer_space() -> None:
    """Capacity to ship packages within D=5 days: monotone feasibility."""
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    D = 5

    def days_needed(cap):
        d, cur = 1, 0
        for w in weights:
            if cur + w > cap:
                d += 1
                cur = w
            else:
                cur += w
        return d

    lo, hi = max(weights), sum(weights)
    caps = list(range(lo, hi + 1))
    days = [days_needed(c) for c in caps]
    feasible = [d <= D for d in days]

    # Binary search trace.
    trace = []
    l, r = lo, hi
    while l < r:
        m = (l + r) // 2
        ok = days_needed(m) <= D
        trace.append((l, r, m, ok))
        if ok:
            r = m
        else:
            l = m + 1
    answer = l

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7),
                                         gridspec_kw={"height_ratios": [3, 2]})

    # Top: days vs capacity, with feasibility shading.
    ax_top.plot(caps, days, color=COLOR_BLUE, lw=2.2,
                marker="o", markersize=4, label="days needed")
    ax_top.axhline(D, color=COLOR_AMBER, lw=1.6, ls="--",
                   label=f"deadline D = {D} days")
    ax_top.fill_between(caps, 0, max(days) + 1,
                        where=feasible, color=COLOR_GREEN, alpha=0.12,
                        label="feasible (≤ D)")
    ax_top.fill_between(caps, 0, max(days) + 1,
                        where=[not f for f in feasible],
                        color=COLOR_AMBER, alpha=0.10,
                        label="infeasible (> D)")
    ax_top.axvline(answer, color=COLOR_GREEN, lw=2.2,
                   label=f"min capacity = {answer}")
    ax_top.set_xlim(lo - 0.5, hi + 0.5)
    ax_top.set_ylim(0, max(days) + 1)
    ax_top.set_xlabel("ship capacity (binary-searched answer)")
    ax_top.set_ylabel("days required")
    ax_top.set_title("Monotone feasibility: bigger capacity, fewer days",
                     fontsize=11, fontweight="bold", color=COLOR_DARK)
    ax_top.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Bottom: shrinking [l, r] bracket per iteration.
    ax_bot.set_xlim(lo - 0.5, hi + 0.5)
    ax_bot.set_ylim(-0.5, len(trace) + 0.5)
    ax_bot.set_yticks(range(len(trace)))
    ax_bot.set_yticklabels([f"iter {i + 1}" for i in range(len(trace))])
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("capacity")
    ax_bot.set_title("Binary-search bracket converging to the minimum",
                     fontsize=11, fontweight="bold", color=COLOR_DARK)
    for i, (l, r, m, ok) in enumerate(trace):
        ax_bot.plot([l, r], [i, i], color=COLOR_GRAY, lw=2.5,
                    solid_capstyle="round")
        ax_bot.scatter([l, r], [i, i], color=COLOR_DARK, s=30, zorder=3)
        ax_bot.scatter([m], [i], color=COLOR_GREEN if ok else COLOR_AMBER,
                       s=85, zorder=4, edgecolor=COLOR_DARK, linewidth=1.0)
        ax_bot.text(r + 0.25, i,
                    f"mid={m}, days={days_needed(m)} "
                    f"{'feasible' if ok else 'too slow'}",
                    va="center", fontsize=8.5,
                    color=COLOR_GREEN if ok else COLOR_AMBER)
    ax_bot.axvline(answer, color=COLOR_GREEN, lw=1.6, ls="--", alpha=0.7)

    fig.suptitle(
        f"Capacity To Ship Packages: weights={weights}, D={D}  →  answer={answer}",
        fontsize=12, fontweight="bold", color=COLOR_DARK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig4_answer_space.png")


# ---------------------------------------------------------------------------
# Figure 5: Closed [l, r] vs half-open [l, r) interval semantics
# ---------------------------------------------------------------------------

def fig5_template_comparison() -> None:
    """Visualise the search interval at each step under the two conventions."""
    n = 8
    nums = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 9

    # Closed [l, r]: right = n - 1, while l <= r, right = mid - 1.
    closed_steps = []
    l, r = 0, n - 1
    while l <= r:
        m = (l + r) // 2
        closed_steps.append((l, r, m, l, r + 1))   # interval [l, r+1)
        if nums[m] == target:
            break
        if nums[m] < target:
            l = m + 1
        else:
            r = m - 1

    # Half-open [l, r): right = n, while l < r, right = mid.
    open_steps = []
    l, r = 0, n
    while l < r:
        m = (l + r) // 2
        open_steps.append((l, r, m, l, r))         # interval [l, r)
        if nums[m] == target:
            break
        if nums[m] < target:
            l = m + 1
        else:
            r = m

    rows = max(len(closed_steps), len(open_steps))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 1.0 * rows + 1.4))
    if rows == 1:
        axes = np.array([axes])

    cell_w = 0.65
    x0 = 0.4

    headers = [
        ("Closed  [l, r]    while l ≤ r,  r = mid − 1", COLOR_BLUE, closed_steps),
        ("Half-open  [l, r)   while l < r,  r = mid", COLOR_PURPLE, open_steps),
    ]

    for col, (title, colour, steps) in enumerate(headers):
        for r_i in range(rows):
            ax = axes[r_i, col]
            ax.set_xlim(0, n * cell_w + 1.6)
            ax.set_ylim(-0.9, 1.1)
            ax.set_aspect("equal")
            ax.axis("off")
            if r_i == 0:
                ax.text(x0 + n * cell_w / 2, 0.85, title,
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color=colour)
            if r_i < len(steps):
                l, r, m, span_l, span_r = steps[r_i]
                faded = [i for i in range(n) if i < span_l or i >= span_r]
                found = m if nums[m] == target else None
                highlight = {} if found is not None else {m: colour}
                _draw_array(ax, nums, x0, 0, cell_w=cell_w, cell_h=cell_w,
                            highlight=highlight, faded=faded, found=found)
                # Interval bracket beneath the array.
                bx0 = x0 + span_l * cell_w
                bx1 = x0 + span_r * cell_w
                y_bracket = -cell_w / 2 - 0.20
                ax.plot([bx0, bx1], [y_bracket, y_bracket], color=colour, lw=2.0)
                ax.text(bx0 - 0.05, y_bracket, "[",
                        ha="right", va="center", color=colour,
                        fontsize=14, fontweight="bold")
                close_char = "]" if col == 0 else ")"
                ax.text(bx1 + 0.05, y_bracket, close_char,
                        ha="left", va="center", color=colour,
                        fontsize=14, fontweight="bold")
                ax.text(x0 - 0.15, 0, f"l={l}\nr={r}",
                        ha="right", va="center", fontsize=9)
                if found is not None:
                    ax.text(x0 + n * cell_w + 0.05, 0,
                            f"found at {m}",
                            ha="left", va="center", fontsize=9.5,
                            fontweight="bold", color=COLOR_GREEN)

    fig.suptitle(f"Two interval conventions, same answer  ·  nums = {nums}, target = {target}",
                 fontsize=12, fontweight="bold", color=COLOR_DARK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig5_template_comparison.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_search_space()
    fig2_left_right_bound()
    fig3_rotated_array()
    fig4_answer_space()
    fig5_template_comparison()
    print(f"Wrote 5 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
