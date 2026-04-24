"""
Figure generation script for LeetCode Patterns Part 10: Stack & Queue.

Generates 5 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly.

Figures:
    fig1_lifo_vs_fifo          Side-by-side comparison of stack (LIFO) and
                               queue (FIFO) operations on the same input
                               sequence so the access discipline becomes
                               visually obvious.
    fig2_valid_parentheses     Step-by-step trace of the stack while validating
                               a bracket string; shows push, match, pop.
    fig3_monotonic_stack       Daily-temperatures style next-greater-element
                               trace; shows how a decreasing-monotone stack
                               resolves answers in amortised O(n).
    fig4_heap_priority_queue   Binary-heap (min-heap) drawn as a tree with the
                               underlying array, illustrating the
                               parent <= child invariant used by priority
                               queues.
    fig5_queue_via_two_stacks  Two-stack queue construction: push goes to the
                               input stack, pop drains input -> output to flip
                               LIFO into FIFO.

Usage:
    python3 scripts/figures/leetcode/10-stack-queue.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good
C_AMBER = "#f59e0b"    # warning / highlight
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "leetcode" / "stack-and-queue"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "leetcode" / "10-栈与队列"


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
# Helpers
# ---------------------------------------------------------------------------
def _draw_cell(ax, x, y, w, h, text, *, color=C_BLUE, edge=C_DARK,
               text_color="white", fontsize=12, lw=1.2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=color, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def _arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=1.6, style="->"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw,
    ))


# ---------------------------------------------------------------------------
# Figure 1: LIFO vs FIFO
# ---------------------------------------------------------------------------
def fig1_lifo_vs_fifo() -> None:
    """Push the same sequence into a stack and a queue; pop and compare."""
    seq = [1, 2, 3, 4]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle("Same input, opposite output: LIFO stack vs FIFO queue",
                 fontsize=14, fontweight="bold", y=1.00)

    # ---------- Stack (vertical) ----------
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Stack (LIFO)  —  push / pop both at the top",
                 fontsize=12, fontweight="bold", color=C_BLUE)

    # Container
    ax.add_patch(Rectangle((1.6, 0.6), 1.8, 4.4,
                           facecolor="white",
                           edgecolor=C_DARK, linewidth=2))
    # Floor only — open top
    ax.plot([1.6, 3.4], [0.6, 0.6], color=C_DARK, lw=2)

    # Stacked elements (1 at bottom, 4 on top)
    for i, v in enumerate(seq):
        _draw_cell(ax, 1.7, 0.7 + i * 0.85, 1.6, 0.78, f"{v}",
                   color=C_BLUE, fontsize=14)

    # Push arrow
    _arrow(ax, 4.6, 4.4, 3.6, 4.0, color=C_GREEN, lw=2)
    ax.text(4.7, 4.55, "push 4", fontsize=11, color=C_GREEN, fontweight="bold")

    # Pop arrow
    _arrow(ax, 3.6, 4.6, 4.6, 5.0, color=C_AMBER, lw=2)
    ax.text(4.7, 5.0, "pop -> 4", fontsize=11, color=C_AMBER, fontweight="bold")

    ax.text(2.5, 0.2, "bottom", ha="center", fontsize=10, color=C_GRAY)
    ax.text(0.6, 5.4, "Push order:  1, 2, 3, 4",
            fontsize=10, color=C_DARK)
    ax.text(0.6, 5.05, "Pop order:    4, 3, 2, 1",
            fontsize=10, color=C_DARK, fontweight="bold")

    # ---------- Queue (horizontal) ----------
    ax = axes[1]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Queue (FIFO)  —  enqueue at rear, dequeue at front",
                 fontsize=12, fontweight="bold", color=C_PURPLE)

    # Open-ended pipe
    y0, h = 2.4, 1.1
    ax.plot([0.6, 7.4], [y0, y0], color=C_DARK, lw=2)
    ax.plot([0.6, 7.4], [y0 + h, y0 + h], color=C_DARK, lw=2)

    for i, v in enumerate(seq):
        _draw_cell(ax, 1.0 + i * 1.4, y0 + 0.05, 1.25, h - 0.1, f"{v}",
                   color=C_PURPLE, fontsize=14)

    # Front / rear labels
    ax.text(1.6, y0 - 0.5, "front", ha="center",
            fontsize=11, color=C_DARK, fontweight="bold")
    ax.text(6.2, y0 - 0.5, "rear", ha="center",
            fontsize=11, color=C_DARK, fontweight="bold")

    # Enqueue arrow (rear)
    _arrow(ax, 7.6, y0 + h / 2, 6.85, y0 + h / 2,
           color=C_GREEN, lw=2)
    ax.text(7.7, y0 + h + 0.15, "enqueue",
            fontsize=11, color=C_GREEN, fontweight="bold")

    # Dequeue arrow (front)
    _arrow(ax, 1.6, y0 + h + 0.6, 0.4, y0 + h + 0.6,
           color=C_AMBER, lw=2)
    ax.text(0.2, y0 + h + 0.75, "dequeue -> 1",
            fontsize=11, color=C_AMBER, fontweight="bold")

    ax.text(0.6, 5.4, "Enqueue order:  1, 2, 3, 4",
            fontsize=10, color=C_DARK)
    ax.text(0.6, 5.05, "Dequeue order:  1, 2, 3, 4",
            fontsize=10, color=C_DARK, fontweight="bold")

    plt.tight_layout()
    _save(fig, "fig1_lifo_vs_fifo")


# ---------------------------------------------------------------------------
# Figure 2: Valid parentheses trace
# ---------------------------------------------------------------------------
def fig2_valid_parentheses() -> None:
    """Trace the stack as we scan the string '({[]})'."""
    s = "({[]})"
    # The matching map mirrors the algorithm.
    pairs = {")": "(", "}": "{", "]": "["}

    # Simulate to get stack snapshots after each step.
    snapshots = []          # list of (index, action, stack-after, ok)
    stack: list[str] = []
    for i, c in enumerate(s):
        if c in pairs:
            top = stack[-1] if stack else None
            if top == pairs[c]:
                stack.pop()
                snapshots.append((i, f"pop  matches", list(stack), True))
            else:
                snapshots.append((i, f"mismatch", list(stack), False))
        else:
            stack.append(c)
            snapshots.append((i, f"push  {c}", list(stack), True))

    n_steps = len(snapshots)

    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(0, n_steps * 1.6 + 1)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f'Valid Parentheses — scanning  "{s}"',
                 fontsize=14, fontweight="bold", pad=10)

    # Top row: the input string with the current pointer highlighted per step.
    ax.text(0.5, 6.5, "step", fontsize=10, color=C_GRAY,
            fontweight="bold", ha="left")
    ax.text(0.5, 5.4, "char", fontsize=10, color=C_GRAY,
            fontweight="bold", ha="left")
    ax.text(0.5, 4.1, "action", fontsize=10, color=C_GRAY,
            fontweight="bold", ha="left")
    ax.text(0.5, 2.6, "stack", fontsize=10, color=C_GRAY,
            fontweight="bold", ha="left")

    for step, (i, action, stk, ok) in enumerate(snapshots):
        x = step * 1.6 + 1.6
        ax.text(x, 6.5, f"{step+1}", ha="center",
                fontsize=11, color=C_DARK, fontweight="bold")

        # current char box
        col = C_GREEN if ok else "#ef4444"
        _draw_cell(ax, x - 0.4, 5.05, 0.8, 0.7, s[i],
                   color=col, fontsize=14)

        # action label
        action_color = C_DARK if ok else "#b91c1c"
        ax.text(x, 4.1, action, ha="center", fontsize=10,
                color=action_color, rotation=0)

        # stack drawing (vertical, base at y=1.1)
        base_y = 1.1
        # box outline
        ax.plot([x - 0.45, x - 0.45], [base_y, base_y + 2.4],
                color=C_GRAY, lw=1)
        ax.plot([x + 0.45, x + 0.45], [base_y, base_y + 2.4],
                color=C_GRAY, lw=1)
        ax.plot([x - 0.45, x + 0.45], [base_y, base_y],
                color=C_DARK, lw=1.5)

        for k, ch in enumerate(stk):
            _draw_cell(ax, x - 0.4, base_y + 0.05 + k * 0.6, 0.8, 0.5, ch,
                       color=C_BLUE, fontsize=12)

    # Final verdict
    final_ok = not stack and all(ok for *_, ok in snapshots)
    verdict = "VALID  -> stack empty" if final_ok else "INVALID"
    ax.text(n_steps * 0.8 + 1, 0.3, verdict,
            ha="center", fontsize=12, fontweight="bold",
            color=C_GREEN if final_ok else "#b91c1c")

    _save(fig, "fig2_valid_parentheses")


# ---------------------------------------------------------------------------
# Figure 3: Monotonic stack — Daily Temperatures
# ---------------------------------------------------------------------------
def fig3_monotonic_stack() -> None:
    """Show the next-greater-element resolution on a small temperature array."""
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    n = len(temps)
    answer = [0] * n
    # We collect (i, popped index list, stack-after)
    trace = []
    stack: list[int] = []
    for i in range(n):
        popped = []
        while stack and temps[i] > temps[stack[-1]]:
            j = stack.pop()
            answer[j] = i - j
            popped.append(j)
        stack.append(i)
        trace.append((i, popped, list(stack)))

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1], hspace=0.45)

    # ---- Top: bar chart with arrows resolving each pop ----
    ax = fig.add_subplot(gs[0])
    bars = ax.bar(range(n), temps, color=C_LIGHT, edgecolor=C_DARK,
                  linewidth=1.2, width=0.7)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"i={i}\n{t}°" for i, t in enumerate(temps)],
                       fontsize=9)
    ax.set_ylim(60, max(temps) + 8)
    ax.set_ylabel("temperature", fontsize=10)
    ax.set_title("Daily Temperatures — monotonic decreasing stack of indices",
                 fontsize=13, fontweight="bold")

    # Highlight every bar that ever gets resolved (answer > 0) in green;
    # bars that are never resolved (answer == 0) remain neutral.
    for k, ans in enumerate(answer):
        if ans > 0:
            bars[k].set_color("#dbeafe")
            bars[k].set_edgecolor(C_BLUE)

    # Draw an arc-arrow from each resolved index to the day that resolved it.
    for j, ans in enumerate(answer):
        if ans == 0:
            continue
        i = j + ans
        ax.annotate(
            "", xy=(i, temps[i] + 0.4), xytext=(j, temps[j] + 0.4),
            arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.8,
                            connectionstyle="arc3,rad=-0.35"),
        )
        ax.text((j + i) / 2, max(temps[j], temps[i]) + 3.0,
                f"+{ans}", ha="center", fontsize=10,
                color=C_GREEN, fontweight="bold")

    # answer row underneath x-axis
    ax.text(-0.9, 60.5, "answer:", fontsize=10, color=C_DARK,
            fontweight="bold")
    for k, a in enumerate(answer):
        ax.text(k, 60.5, f"{a}", ha="center", fontsize=10,
                color=C_GREEN if a > 0 else C_GRAY, fontweight="bold")

    # ---- Bottom: stack snapshots after each step ----
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, n * 2 + 1)
    ax2.set_ylim(0, 4.2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.text(0.05, 3.6, "stack of indices  (top -> right)",
             fontsize=10, color=C_GRAY, fontweight="bold")

    for step, (i, popped, stk) in enumerate(trace):
        x0 = step * 2 + 0.6
        # step header
        ax2.text(x0 + 0.7, 3.05, f"i={i}", ha="center",
                 fontsize=9, color=C_DARK, fontweight="bold")
        if popped:
            ax2.text(x0 + 0.7, 2.55, f"pop {popped}",
                     ha="center", fontsize=8, color=C_AMBER)
        # stack as horizontal cells
        for k, idx in enumerate(stk):
            _draw_cell(ax2, x0 + k * 0.45, 1.4, 0.4, 0.55,
                       f"{idx}", color=C_BLUE, fontsize=10)
        # draw a low underline indicating "stack base"
        ax2.plot([x0 - 0.05, x0 + max(1, len(stk)) * 0.45 + 0.0],
                 [1.35, 1.35], color=C_GRAY, lw=1)

    _save(fig, "fig3_monotonic_stack")


# ---------------------------------------------------------------------------
# Figure 4: Min-heap (priority queue)
# ---------------------------------------------------------------------------
def fig4_heap_priority_queue() -> None:
    """Draw a min-heap as both a complete binary tree and its array layout."""
    heap = [2, 5, 3, 9, 7, 4, 8, 12, 10]
    n = len(heap)

    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1], hspace=0.35)

    # ---- Tree view ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Min-heap — every parent <= its children (priority queue core)",
                 fontsize=13, fontweight="bold")

    # Position nodes by level: level d has 2^d nodes spaced evenly.
    positions: dict[int, tuple[float, float]] = {}
    levels = int(np.floor(np.log2(n))) + 1
    width = 16
    for idx in range(n):
        d = int(np.floor(np.log2(idx + 1)))
        offset = idx + 1 - 2**d
        slots = 2**d
        x = width * (offset + 0.5) / slots
        y = 5.4 - d * 1.6
        positions[idx] = (x, y)

    # Edges first (parent -> child)
    for idx in range(n):
        p = (idx - 1) // 2
        if idx == 0:
            continue
        x1, y1 = positions[p]
        x2, y2 = positions[idx]
        ax.plot([x1, x2], [y1, y2], color=C_GRAY, lw=1.3, zorder=1)

    # Nodes
    for idx, val in enumerate(heap):
        x, y = positions[idx]
        circ = mpatches.Circle((x, y), 0.45, facecolor=C_BLUE,
                               edgecolor=C_DARK, lw=1.4, zorder=2)
        ax.add_patch(circ)
        ax.text(x, y, f"{val}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=3)
        ax.text(x + 0.55, y + 0.45, f"[{idx}]", fontsize=8, color=C_GRAY)

    # Highlight root as the min
    rx, ry = positions[0]
    ax.add_patch(mpatches.Circle((rx, ry), 0.55,
                                 facecolor="none", edgecolor=C_AMBER, lw=2.5))
    ax.text(rx + 1.4, ry + 0.1, "<-  min = peek()", ha="left",
            fontsize=10, color=C_AMBER, fontweight="bold")

    # ---- Array view ----
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, n + 1)
    ax2.set_ylim(0, 2.4)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.text(0.05, 2.0, "array layout  (parent of i  =  (i-1)//2)",
             fontsize=10, color=C_DARK, fontweight="bold")

    for idx, val in enumerate(heap):
        _draw_cell(ax2, idx + 0.4, 0.7, 0.85, 0.85, f"{val}",
                   color=C_PURPLE, fontsize=12)
        ax2.text(idx + 0.82, 0.45, f"{idx}", ha="center",
                 fontsize=9, color=C_GRAY)

    _save(fig, "fig4_heap_priority_queue")


# ---------------------------------------------------------------------------
# Figure 5: Queue using two stacks
# ---------------------------------------------------------------------------
def fig5_queue_via_two_stacks() -> None:
    """Three panels: after pushes, the transfer step, and after a pop."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4))
    fig.suptitle("Queue using two stacks: push -> input, pop -> drain to output",
                 fontsize=14, fontweight="bold", y=1.02)

    panels = [
        ("State A:  push(1), push(2), push(3)",
         [1, 2, 3], [],          None),
        ("State B:  pop() triggers transfer  input -> output",
         [],        [3, 2, 1],   "transfer"),
        ("State C:  output.pop() returns 1   (true FIFO order)",
         [],        [3, 2],      "popped"),
    ]

    for ax, (title, in_stack, out_stack, marker) in zip(axes, panels):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", color=C_DARK)

        # input stack container (left)
        ax.text(1.0, 5.5, "input", fontsize=10, color=C_BLUE,
                fontweight="bold", ha="center")
        ax.plot([0.5, 0.5], [0.6, 4.8], color=C_DARK, lw=1.5)
        ax.plot([1.5, 1.5], [0.6, 4.8], color=C_DARK, lw=1.5)
        ax.plot([0.5, 1.5], [0.6, 0.6], color=C_DARK, lw=2)
        for k, v in enumerate(in_stack):
            _draw_cell(ax, 0.55, 0.7 + k * 0.7, 0.9, 0.62, f"{v}",
                       color=C_BLUE, fontsize=12)

        # output stack container (right)
        ax.text(5.0, 5.5, "output", fontsize=10, color=C_PURPLE,
                fontweight="bold", ha="center")
        ax.plot([4.5, 4.5], [0.6, 4.8], color=C_DARK, lw=1.5)
        ax.plot([5.5, 5.5], [0.6, 4.8], color=C_DARK, lw=2)
        ax.plot([4.5, 5.5], [0.6, 0.6], color=C_DARK, lw=2)
        for k, v in enumerate(out_stack):
            _draw_cell(ax, 4.55, 0.7 + k * 0.7, 0.9, 0.62, f"{v}",
                       color=C_PURPLE, fontsize=12)

        if marker == "transfer":
            _arrow(ax, 1.7, 3.0, 4.3, 3.0, color=C_AMBER, lw=2.4)
            ax.text(3.0, 3.4, "while input:  output.push(input.pop())",
                    ha="center", fontsize=9, color=C_AMBER,
                    fontweight="bold")
        if marker == "popped":
            ax.text(3.0, 3.4, "returned: 1", ha="center",
                    fontsize=11, color=C_GREEN, fontweight="bold")
            _arrow(ax, 4.5, 4.0, 3.0, 4.0, color=C_GREEN, lw=2.0)

    plt.tight_layout()
    _save(fig, "fig5_queue_via_two_stacks")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for LeetCode Part 10: Stack & Queue")
    fig1_lifo_vs_fifo()
    fig2_valid_parentheses()
    fig3_monotonic_stack()
    fig4_heap_priority_queue()
    fig5_queue_via_two_stacks()
    print("Done.")


if __name__ == "__main__":
    main()
