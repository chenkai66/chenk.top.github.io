"""
Figure generation for LeetCode Patterns Article 09: Greedy Algorithms.

Generates 5 figures used in BOTH the EN and ZH articles. Each figure
isolates ONE pedagogical idea cleanly and avoids decoration without
information value.

Figures:
    fig1_interval_scheduling   Activity selection visualised as a Gantt
                               chart sorted by end time. The greedy choice
                               (earliest finish) is highlighted; rejected
                               overlapping intervals are dimmed. Conveys why
                               sorting by END time leaves the most room for
                               the rest of the schedule.

    fig2_jump_game             Jump Game (LC 55) reachability frontier.
                               Two arrays are shown side by side: a
                               reachable case [2,3,1,1,4] and a stuck case
                               [3,2,1,0,4]. For each index i we draw the
                               cone of cells reachable from i and the
                               running max_reach line. The frontier
                               collapses past the zero in the failing case.

    fig3_gas_station           Gas Station (LC 134) cumulative tank
                               diagram. Plots the running tank level
                               sum(gas[i]-cost[i]) around the loop with
                               the index of the global minimum highlighted
                               -- the optimal start is one position after
                               that minimum. Explains the O(n) greedy.

    fig4_exchange_argument     Visual proof template (exchange argument)
                               for why "sort by end time" is optimal.
                               Shows an arbitrary optimal schedule OPT,
                               then swaps OPT's first interval for the
                               earliest-finishing greedy interval and
                               argues compatibility is preserved.

    fig5_greedy_vs_dp          When greedy works and when it fails. Two
                               coin-change instances side by side:
                               greedy is OPTIMAL on the canonical US
                               system {1,5,10,25} for amount 30 but
                               SUBOPTIMAL on {1,3,4} for amount 6 (greedy
                               picks 4+1+1, DP picks 3+3). Demonstrates
                               that greedy correctness is structural, not
                               universal.

Usage:
    python3 scripts/figures/leetcode/09-greedy.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_RED = "#dc2626"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "leetcode" / "09-greedy-algorithms"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "leetcode" / "09-贪心算法"


def _save(fig: plt.Figure, name: str) -> None:
    """Write the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Interval scheduling -- earliest finish time greedy
# ---------------------------------------------------------------------------
def fig1_interval_scheduling() -> None:
    # Hand-picked instance whose optimal schedule is non-trivial.
    # Each tuple is (start, end). They will be sorted by end time before plot.
    intervals = [
        (1, 4),
        (3, 5),
        (0, 6),
        (5, 7),
        (3, 9),
        (5, 9),
        (6, 10),
        (8, 11),
        (8, 12),
        (2, 14),
        (12, 16),
    ]
    intervals.sort(key=lambda iv: iv[1])

    # Run earliest-finish greedy on the sorted list.
    chosen = []
    last_end = -1
    for s, e in intervals:
        if s >= last_end:
            chosen.append((s, e))
            last_end = e
    chosen_set = set(chosen)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bar_h = 0.65
    for row, (s, e) in enumerate(intervals):
        y = len(intervals) - 1 - row
        is_pick = (s, e) in chosen_set
        face = C_GREEN if is_pick else C_LIGHT
        edge = C_GREEN if is_pick else C_GRAY
        text_color = "white" if is_pick else C_GRAY
        ax.add_patch(
            Rectangle(
                (s, y - bar_h / 2),
                e - s,
                bar_h,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.2,
                alpha=0.95 if is_pick else 0.75,
            )
        )
        ax.text(
            (s + e) / 2,
            y,
            f"[{s},{e})",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
            fontweight="bold" if is_pick else "normal",
        )

    # Vertical guides at the chosen end times to make the "earliest finish" idea visible.
    for _, e in chosen:
        ax.axvline(e, color=C_GREEN, linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_xlim(-0.5, 17)
    ax.set_ylim(-0.7, len(intervals) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Intervals (sorted by end time, top = earliest finish)", fontsize=10)
    ax.set_title(
        "Activity Selection: pick the earliest-finishing compatible interval at each step",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Legend.
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=C_GREEN, edgecolor=C_GREEN, label=f"Selected ({len(chosen)} intervals)"),
        Rectangle((0, 0), 1, 1, facecolor=C_LIGHT, edgecolor=C_GRAY, label="Rejected (overlap with a chosen one)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95, fontsize=9)

    fig.tight_layout()
    _save(fig, "fig1_interval_scheduling")


# ---------------------------------------------------------------------------
# Figure 2: Jump Game -- max_reach frontier
# ---------------------------------------------------------------------------
def fig2_jump_game() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    cases = [
        ("Reachable: nums = [2, 3, 1, 1, 4]", [2, 3, 1, 1, 4], True),
        ("Stuck: nums = [3, 2, 1, 0, 4]", [3, 2, 1, 0, 4], False),
    ]

    for ax, (title, nums, ok) in zip(axes, cases):
        n = len(nums)
        # Compute the running max_reach.
        max_reach = 0
        reach_trace = []
        stuck_at = None
        for i in range(n):
            if i > max_reach:
                stuck_at = i
                break
            max_reach = max(max_reach, i + nums[i])
            reach_trace.append(max_reach)

        # Cell row for the array.
        cell_y = 0.25
        cell_h = 0.5
        for i, v in enumerate(nums):
            face = C_LIGHT
            if stuck_at is not None and i >= stuck_at:
                face = "#fee2e2"  # soft red for unreachable
            ax.add_patch(
                Rectangle(
                    (i - 0.4, cell_y - cell_h / 2),
                    0.8,
                    cell_h,
                    facecolor=face,
                    edgecolor=C_DARK,
                    linewidth=1.2,
                )
            )
            ax.text(i, cell_y, str(v), ha="center", va="center", fontsize=12, fontweight="bold", color=C_DARK)
            ax.text(i, cell_y - 0.55, f"i={i}", ha="center", va="center", fontsize=8, color=C_GRAY)

        # Reach-from-i arcs.
        for i, v in enumerate(nums):
            if stuck_at is not None and i >= stuck_at:
                break
            target = min(i + v, n - 1)
            if target == i:
                continue
            arrow = FancyArrowPatch(
                (i, 1.4),
                (target, 1.4),
                arrowstyle="->",
                mutation_scale=14,
                color=C_BLUE,
                linewidth=1.4,
                alpha=0.7,
                connectionstyle="arc3,rad=-0.35",
            )
            ax.add_patch(arrow)

        # max_reach line.
        if reach_trace:
            xs = list(range(len(reach_trace)))
            ax.plot(xs, [r + 0.05 for r in reach_trace], color=C_PURPLE, linewidth=2.5, marker="o", markersize=6, label="max_reach after step i")
            for i, r in enumerate(reach_trace):
                ax.annotate(str(r), (i, r + 0.05), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=C_PURPLE, fontweight="bold")

        # Goal marker.
        ax.axvline(n - 1, color=C_GREEN, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(n - 1, 4.55, f"goal = n-1 = {n-1}", ha="center", fontsize=9, color=C_GREEN, fontweight="bold")

        # Verdict banner.
        verdict_text = "True (goal reached)" if ok else f"False (i={stuck_at} > max_reach={reach_trace[-1] if reach_trace else 0})"
        verdict_color = C_GREEN if ok else C_RED
        ax.text(
            (n - 1) / 2,
            -0.85,
            verdict_text,
            ha="center",
            fontsize=11,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=verdict_color, edgecolor="none"),
        )

        ax.set_xlim(-0.8, n - 0.2)
        ax.set_ylim(-1.4, 5.0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Jump Game: maintain the running max_reach frontier", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_jump_game")


# ---------------------------------------------------------------------------
# Figure 3: Gas Station -- cumulative tank
# ---------------------------------------------------------------------------
def fig3_gas_station() -> None:
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    n = len(gas)
    delta = [g - c for g, c in zip(gas, cost)]

    # Cumulative tank starting from station 0 (loop is closed at end).
    xs = list(range(n + 1))
    tank = [0]
    for d in delta:
        tank.append(tank[-1] + d)

    # The optimal start index = one past the global minimum of the cumulative curve.
    min_idx = int(np.argmin(tank))
    start = min_idx % n  # since tank[0]=0 corresponds to "before station 0"

    fig, ax = plt.subplots(figsize=(11, 5.4))

    # Shade the failing prefix (tank goes negative) vs. the rest.
    ax.fill_between(xs, tank, 0, where=[t < 0 for t in tank], interpolate=True, color="#fee2e2", alpha=0.7, label="Tank empty (cannot start before this)")

    ax.plot(xs, tank, color=C_BLUE, linewidth=2.4, marker="o", markersize=8, zorder=3)
    for x, t in zip(xs, tank):
        ax.annotate(f"{t:+d}", (x, t), textcoords="offset points", xytext=(0, 10 if t >= 0 else -16), ha="center", fontsize=9, color=C_DARK, fontweight="bold")

    # Highlight the global minimum.
    ax.scatter([min_idx], [tank[min_idx]], color=C_RED, s=140, zorder=4, edgecolor="white", linewidth=2, label=f"Global min at index {min_idx}")
    # Highlight the chosen start.
    ax.axvline(start, color=C_GREEN, linestyle="--", linewidth=1.8, alpha=0.85)
    ax.text(start, max(tank) + 0.6, f"start = {start}", ha="center", fontsize=11, color=C_GREEN, fontweight="bold")

    # Station table at the bottom.
    table_y = -3.6
    for i in range(n):
        ax.text(i, table_y, f"gas={gas[i]}\ncost={cost[i]}\nΔ={delta[i]:+d}", ha="center", va="top", fontsize=8.5, color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_GRAY, linewidth=0.8))

    ax.axhline(0, color=C_GRAY, linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"before {i}" if i == 0 else f"after {i-1}" for i in xs], rotation=0, fontsize=8.5)
    ax.set_xlim(-0.5, n + 0.4)
    ax.set_ylim(-5.5, max(tank) + 2.0)
    ax.set_xlabel("Position around the loop", fontsize=11)
    ax.set_ylabel("Cumulative tank = Σ (gas[k] − cost[k])", fontsize=11)
    ax.set_title("Gas Station: optimal start = one past the global minimum of the prefix sum", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    _save(fig, "fig3_gas_station")


# ---------------------------------------------------------------------------
# Figure 4: Exchange argument
# ---------------------------------------------------------------------------
def fig4_exchange_argument() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))

    # Two schedules drawn as Gantt rows: an arbitrary OPT and the post-swap OPT'.
    bar_h = 0.55

    # OPT (top): first chosen interval is X = [1, 7); it finishes late.
    opt = [("X", 1, 7), ("B", 8, 11), ("C", 12, 14)]
    # Greedy alternative G = [2, 5) finishes earliest among compatible firsts.
    greedy = ("G", 2, 5)
    # OPT' after swap: replace X with G; the rest (B, C) still fits because G ends earlier than X.
    opt_prime = [("G", 2, 5), ("B", 8, 11), ("C", 12, 14)]

    # Top row -- OPT.
    y_top = 2.5
    for name, s, e in opt:
        color = C_AMBER if name == "X" else C_BLUE
        ax.add_patch(Rectangle((s, y_top - bar_h / 2), e - s, bar_h, facecolor=color, edgecolor=C_DARK, alpha=0.9))
        ax.text((s + e) / 2, y_top, f"{name} [{s},{e})", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(-0.5, y_top, "OPT", ha="right", va="center", fontsize=11, fontweight="bold", color=C_DARK)

    # Greedy candidate (middle) -- shown in dashed outline.
    y_mid = 1.3
    s, e = greedy[1], greedy[2]
    ax.add_patch(Rectangle((s, y_mid - bar_h / 2), e - s, bar_h, facecolor=C_GREEN, edgecolor=C_GREEN, alpha=0.85))
    ax.text((s + e) / 2, y_mid, f"G [{s},{e})  earliest finish", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(-0.5, y_mid, "Greedy", ha="right", va="center", fontsize=11, fontweight="bold", color=C_GREEN)

    # Swap arrow.
    arrow = FancyArrowPatch((1.5, y_top - 0.4), (3.5, y_mid + 0.4), arrowstyle="->", mutation_scale=18, color=C_RED, linewidth=2)
    ax.add_patch(arrow)
    ax.text(2.5, (y_top + y_mid) / 2 + 0.05, "swap X → G", color=C_RED, fontsize=9.5, fontweight="bold", ha="center")

    # Bottom row -- OPT' (post-swap).
    y_bot = 0.1
    for name, s, e in opt_prime:
        color = C_GREEN if name == "G" else C_BLUE
        ax.add_patch(Rectangle((s, y_bot - bar_h / 2), e - s, bar_h, facecolor=color, edgecolor=C_DARK, alpha=0.9))
        ax.text((s + e) / 2, y_bot, f"{name} [{s},{e})", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(-0.5, y_bot, "OPT'", ha="right", va="center", fontsize=11, fontweight="bold", color=C_DARK)

    # Annotation: same size, still feasible.
    ax.text(15.4, y_bot, "|OPT'| = |OPT|\nstill feasible\n(G ends ≤ X ends)", ha="left", va="center", fontsize=9.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor=C_GREEN))

    ax.set_xlim(-2.0, 22)
    ax.set_ylim(-0.6, 3.4)
    ax.set_xticks(range(0, 16, 2))
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=11)
    ax.set_title("Exchange argument: any OPT can be transformed into a greedy schedule of equal size", fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    _save(fig, "fig4_exchange_argument")


# ---------------------------------------------------------------------------
# Figure 5: Greedy works vs. greedy fails (coin change)
# ---------------------------------------------------------------------------
def fig5_greedy_vs_dp() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    def draw_coins(ax, title, coins, amount, greedy_pick, optimal_pick, greedy_is_optimal):
        # Plot the chosen coin "stack" as horizontal bars.
        ax.set_title(title, fontsize=11.5, fontweight="bold")

        def stack(y, picks, label_prefix, color):
            x = 0
            for c in picks:
                ax.add_patch(Rectangle((x, y - 0.35), c, 0.7, facecolor=color, edgecolor="white", linewidth=1.5))
                ax.text(x + c / 2, y, str(c), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
                x += c
            ax.text(amount + 0.6, y, f"{label_prefix}: {len(picks)} coins", ha="left", va="center", fontsize=10, color=color, fontweight="bold")

        stack(1.5, greedy_pick, "Greedy", C_AMBER if not greedy_is_optimal else C_GREEN)
        stack(0.4, optimal_pick, "DP / optimal", C_BLUE)

        ax.axvline(amount, color=C_GRAY, linestyle="--", linewidth=1)
        ax.text(amount, 2.55, f"target = {amount}", ha="center", fontsize=10, color=C_DARK, fontweight="bold")

        verdict = "Greedy = optimal" if greedy_is_optimal else "Greedy SUBOPTIMAL"
        verdict_color = C_GREEN if greedy_is_optimal else C_RED
        ax.text(
            amount / 2,
            -0.8,
            f"Coins = {coins}     |     {verdict}",
            ha="center",
            fontsize=10.5,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=verdict_color, edgecolor="none"),
        )

        ax.set_xlim(-0.4, amount + 8)
        ax.set_ylim(-1.5, 3.0)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)

    # Case A: canonical US coins -> greedy is optimal.
    draw_coins(
        axes[0],
        "Greedy WORKS:  amount = 30, coins = {1, 5, 10, 25}",
        [1, 5, 10, 25],
        30,
        greedy_pick=[25, 5],
        optimal_pick=[25, 5],
        greedy_is_optimal=True,
    )

    # Case B: pathological set -> greedy fails.
    draw_coins(
        axes[1],
        "Greedy FAILS:  amount = 6, coins = {1, 3, 4}",
        [1, 3, 4],
        6,
        greedy_pick=[4, 1, 1],   # greedy by largest coin
        optimal_pick=[3, 3],     # DP finds 2-coin solution
        greedy_is_optimal=False,
    )

    fig.suptitle(
        "When greedy works depends on the structure of the problem (matroid / exchange property),\n"
        "not on the algorithm being 'simple'.",
        fontsize=12.5,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    _save(fig, "fig5_greedy_vs_dp")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Article 09 (Greedy Algorithms) figures...")
    fig1_interval_scheduling()
    fig2_jump_game()
    fig3_gas_station()
    fig4_exchange_argument()
    fig5_greedy_vs_dp()
    print("Done.")


if __name__ == "__main__":
    main()
