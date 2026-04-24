"""
Figure generation script for Computer Fundamentals Part 01: CPU.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single, specific idea cleanly.

Figures:
    fig1_cpu_architecture        Block diagram of a modern CPU core: ALU,
                                 control unit, register file, L1 I/D caches,
                                 L2, with the data/instruction paths wired up.
    fig2_pipeline_stages         5-stage in-order pipeline (IF/ID/EX/MEM/WB)
                                 unrolled across cycles, showing how four
                                 instructions overlap and the steady-state IPC.
    fig3_cache_hierarchy         Memory pyramid: registers -> L1 -> L2 -> L3
                                 -> DRAM -> SSD -> HDD with measured latencies
                                 in nanoseconds (log scale).
    fig4_cache_misses            The "3C" cache-miss taxonomy: compulsory,
                                 capacity, conflict, with a small worked
                                 example for each on a direct-mapped cache.
    fig5_branch_prediction       2-bit saturating counter state machine and
                                 a workload-vs-accuracy bar chart comparing
                                 static / 1-bit / 2-bit / TAGE-style.
    fig6_ooo_execution           In-order vs out-of-order execution timeline
                                 for the same instruction stream, showing how
                                 a load-use stall is hidden by reordering.
    fig7_multicore_smt           Multi-core + SMT (hyperthreading) layout:
                                 cores, per-core L1/L2, shared L3, ring bus,
                                 and how 2 logical threads share one core's
                                 execution resources.

Usage:
    python3 scripts/figures/computer-fundamentals/01-cpu.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"      # primary  - control / instructions
C_PURPLE = "#7c3aed"    # secondary - data / compute
C_GREEN = "#10b981"     # good / hit / fast
C_AMBER = "#f59e0b"     # warning / miss / hot path
C_RED = "#ef4444"       # bad / stall
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
    ROOT / "source" / "_posts" / "en" / "computer-fundamentals" / "01-cpu",
    ROOT / "source" / "_posts" / "zh" / "computer-fundamentals" / "01-cpu",
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
def _box(ax, x, y, w, h, text, fc, ec=None, fontsize=10, fontcolor="white",
         weight="bold", radius=0.04):
    ec = ec or fc
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=fontcolor, weight=weight)


def _arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=1.4, style="-|>", mut=14):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mut,
        color=color, linewidth=lw,
    ))


# ---------------------------------------------------------------------------
# Figure 1: CPU core architecture block diagram
# ---------------------------------------------------------------------------
def fig1_cpu_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.set_title("Figure 1. CPU Core: Control, Compute, Registers, Cache",
                 loc="left", pad=12)

    # Outer chip boundary
    chip = FancyBboxPatch((0.2, 0.3), 11.6, 6.3,
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          facecolor="#f8fafc", edgecolor=C_GRAY, linewidth=1.5)
    ax.add_patch(chip)
    ax.text(0.5, 6.35, "CPU Core", fontsize=11, color=C_DARK, weight="bold")

    # Front-end: fetch + decode + branch predictor
    _box(ax, 0.7, 4.6, 2.2, 1.1, "Fetch\n(IF)", C_BLUE)
    _box(ax, 0.7, 3.2, 2.2, 1.1, "Decode\n(ID)", C_BLUE)
    _box(ax, 0.7, 1.8, 2.2, 1.1, "Branch\nPredictor", "#1d4ed8")

    # Register file
    _box(ax, 3.5, 3.2, 1.8, 2.5, "Register\nFile\n(x86: 16 GPR\nAVX: 32x512b)",
         C_PURPLE, fontsize=9)

    # Execution units (ALU, FPU, LSU)
    _box(ax, 5.8, 4.8, 1.6, 0.9, "ALU", "#6d28d9", fontsize=10)
    _box(ax, 5.8, 3.7, 1.6, 0.9, "FPU /\nSIMD", "#6d28d9", fontsize=10)
    _box(ax, 5.8, 2.6, 1.6, 0.9, "Load/Store\nUnit", "#6d28d9", fontsize=10)

    # L1 caches (split)
    _box(ax, 7.9, 4.6, 1.6, 1.1, "L1 I-cache\n32 KB", C_GREEN, fontsize=10)
    _box(ax, 7.9, 3.2, 1.6, 1.1, "L1 D-cache\n48 KB", C_GREEN, fontsize=10)

    # L2 + L3 (L3 shared, dashed)
    _box(ax, 9.9, 3.9, 1.7, 1.8, "L2 cache\n1.25 MB\n(per core)", C_AMBER,
         fontsize=10)
    l3 = FancyBboxPatch((0.7, 0.55), 10.9, 1.0,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor="#fde68a", edgecolor=C_AMBER, linewidth=1.2,
                        linestyle="--")
    ax.add_patch(l3)
    ax.text(6.15, 1.05, "L3 cache  -  shared across all cores  -  24 MB",
            ha="center", va="center", fontsize=11, color="#78350f",
            weight="bold")

    # Wires
    # fetch -> decode
    _arrow(ax, 1.8, 4.6, 1.8, 4.3, C_BLUE)
    # branch predictor -> fetch (route along the left edge so it doesn't
    # cross the decode box)
    ax.plot([0.55, 0.55], [2.35, 5.15], color=C_BLUE, linewidth=1.0)
    _arrow(ax, 0.55, 5.15, 0.7, 5.15, C_BLUE, lw=1.0)
    ax.plot([0.55, 0.7], [2.35, 2.35], color=C_BLUE, linewidth=1.0)
    # decode -> register file
    _arrow(ax, 2.9, 3.75, 3.5, 3.75, C_BLUE)
    # register file -> ALU/FPU/LSU
    for y in (5.25, 4.15, 3.05):
        _arrow(ax, 5.3, 4.45, 5.8, y, C_PURPLE)
    # LSU -> L1D
    _arrow(ax, 7.4, 3.05, 7.9, 3.55, C_PURPLE)
    # IF -> L1I  (route up and over so it doesn't cut through ALU)
    ax.plot([1.8, 1.8],   [5.7, 6.1], color=C_GREEN, linewidth=1.2)
    ax.plot([1.8, 8.7],   [6.1, 6.1], color=C_GREEN, linewidth=1.2)
    _arrow(ax, 8.7, 6.1, 8.7, 5.7, C_GREEN, lw=1.2)
    # L1 -> L2
    _arrow(ax, 9.5, 5.15, 9.9, 5.15, C_AMBER)
    _arrow(ax, 9.5, 3.75, 9.9, 4.5, C_AMBER)
    # L2 -> L3
    _arrow(ax, 10.75, 3.9, 10.75, 1.55, C_AMBER, lw=1.2, style="<|-|>")

    # Legend
    handles = [
        mpatches.Patch(color=C_BLUE, label="Control / Front-end"),
        mpatches.Patch(color=C_PURPLE, label="Compute / Registers"),
        mpatches.Patch(color=C_GREEN, label="L1 cache (lowest latency)"),
        mpatches.Patch(color=C_AMBER, label="L2 / L3 cache"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
              bbox_to_anchor=(0.99, 0.99))

    fig.tight_layout()
    save(fig, "fig1_cpu_architecture")


# ---------------------------------------------------------------------------
# Figure 2: 5-stage pipeline
# ---------------------------------------------------------------------------
def fig2_pipeline_stages():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    stages = ["IF", "ID", "EX", "MEM", "WB"]
    stage_color = {
        "IF": C_BLUE, "ID": "#3b82f6", "EX": C_PURPLE,
        "MEM": C_AMBER, "WB": C_GREEN,
    }
    n_inst = 5
    n_cycles = n_inst + len(stages) - 1

    ax.set_xlim(-0.2, n_cycles + 1)
    ax.set_ylim(-0.5, n_inst + 1.2)
    ax.set_axis_off()
    ax.set_title(
        "Figure 2. Classic 5-Stage Pipeline: 5 instructions finish in 9 cycles, not 25",
        loc="left", pad=12)

    # Cycle header
    for c in range(n_cycles):
        ax.text(c + 1.5, n_inst + 0.6, f"Cycle {c+1}",
                ha="center", va="center", fontsize=10,
                color=C_DARK, weight="bold")

    # Instruction labels and stage cells
    for i in range(n_inst):
        y = n_inst - i - 0.5
        ax.text(0.4, y, f"I{i+1}", ha="right", va="center",
                fontsize=11, color=C_DARK, weight="bold")
        for s, name in enumerate(stages):
            x = 1 + i + s
            rect = Rectangle((x, y - 0.4), 1.0, 0.8,
                             facecolor=stage_color[name],
                             edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + 0.5, y, name, ha="center", va="center",
                    fontsize=10, color="white", weight="bold")

    # Steady-state highlight (place the label below the instruction rows
    # so it doesn't sit on top of an EX cell)
    ax.add_patch(Rectangle((5, -0.05), 1.0, n_inst + 0.05,
                           facecolor=C_AMBER, alpha=0.12, edgecolor="none"))
    ax.text(5.5, -0.35, "steady state: 1 instr/cycle",
            ha="center", va="top", fontsize=9, color="#78350f",
            weight="bold")

    # Sequential vs pipelined annotation
    ax.text(n_cycles + 0.7, n_inst - 0.5,
            "Without pipelining:\n5 instr x 5 stages = 25 cycles\n\n"
            "With pipelining:\n4 + 5 = 9 cycles\n"
            "Speedup ~ 2.8x",
            ha="left", va="top", fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9",
                      edgecolor=C_GRAY))

    # Stage legend below
    handles = [mpatches.Patch(color=stage_color[s],
                              label=f"{s} - " + {
                                  "IF": "Instruction Fetch",
                                  "ID": "Instruction Decode",
                                  "EX": "Execute (ALU)",
                                  "MEM": "Memory Access",
                                  "WB": "Write Back",
                              }[s]) for s in stages]
    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(0.0, -0.05), ncol=5, frameon=False, fontsize=9)

    fig.tight_layout()
    save(fig, "fig2_pipeline_stages")


# ---------------------------------------------------------------------------
# Figure 3: Cache hierarchy with latencies
# ---------------------------------------------------------------------------
def fig3_cache_hierarchy():
    fig, ax = plt.subplots(figsize=(11, 6.0))

    # Each row: (name, latency in ns, capacity, color)
    levels = [
        ("Register",    0.3,    "few KB",     C_PURPLE),
        ("L1 cache",    1.0,    "32-64 KB",   C_GREEN),
        ("L2 cache",    4.0,    "256 KB-1 MB", "#34d399"),
        ("L3 cache",   15.0,    "8-64 MB",    C_AMBER),
        ("DRAM (DDR4/5)", 90.0, "8-128 GB",   "#fb923c"),
        ("SSD (NVMe)", 100_000.0,    "0.5-8 TB",   C_RED),
        ("HDD (7200 rpm)", 10_000_000.0, "1-20 TB", "#7f1d1d"),
    ]

    names = [l[0] for l in levels]
    lat = np.array([l[1] for l in levels])
    caps = [l[2] for l in levels]
    colors = [l[3] for l in levels]
    ypos = np.arange(len(levels))[::-1]

    ax.barh(ypos, lat, color=colors, edgecolor="white", linewidth=1.5,
            log=True, height=0.65)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("Access latency (nanoseconds, log scale)", fontsize=11)
    ax.set_xlim(0.1, 5e8)
    ax.set_title("Figure 3. The Memory Hierarchy: 8 Orders of Magnitude "
                 "Between Register and Disk",
                 loc="left", pad=12)

    # Annotate latency + capacity to the right of each bar
    human = ["~0.3 ns", "~1 ns", "~4 ns", "~15 ns",
             "~90 ns", "~100 us", "~10 ms"]
    for y, l, h, c in zip(ypos, lat, human, caps):
        ax.text(l * 1.6, y, f"{h}   |   {c}",
                va="center", ha="left", fontsize=10, color=C_DARK)

    # Vertical reference lines: "1 second of CPU thinking"
    ax.axvline(1, color=C_GRAY, linestyle=":", linewidth=1)
    ax.text(1, len(levels) - 0.3, "1 ns", ha="center",
            fontsize=8, color=C_GRAY)

    ax.set_axisbelow(True)
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    save(fig, "fig3_cache_hierarchy")


# ---------------------------------------------------------------------------
# Figure 4: Cache miss taxonomy (3C)
# ---------------------------------------------------------------------------
def fig4_cache_misses():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    fig.suptitle("Figure 4. The Three Kinds of Cache Misses (3C Model)",
                 x=0.04, ha="left", fontsize=13, weight="bold", y=0.99)

    titles = ["Compulsory (cold)", "Capacity", "Conflict"]
    subs = [
        "First-ever access to a block.\nUnavoidable on cold start.",
        "Working set larger than cache.\nEven a fully-associative cache misses.",
        "Multiple hot blocks map to\nthe same set in a low-associativity cache.",
    ]
    color = [C_BLUE, C_AMBER, C_RED]

    for ax, t, s, c in zip(axes, titles, subs, color):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 6.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#f8fafc", edgecolor=c,
                                    linewidth=1.6))
        ax.text(5, 6.3, t, ha="center", va="center", fontsize=12,
                color=c, weight="bold")
        ax.text(5, 5.4, s, ha="center", va="center", fontsize=9.5,
                color=C_DARK)

    # --- Panel 1: Compulsory --- show empty cache, first access -> miss ---
    ax = axes[0]
    cache_x = 1.2
    for i in range(4):
        y = 1.0 + i * 0.9
        ax.add_patch(Rectangle((cache_x, y), 4.8, 0.7, facecolor="white",
                               edgecolor=C_GRAY, linewidth=1.0))
        ax.text(cache_x + 0.3, y + 0.35, f"set {i}", ha="left", va="center",
                fontsize=9, color=C_GRAY)
    # arrow: address A -> set 2 (empty) -> miss
    ax.text(7.3, 4.0, "addr A", fontsize=10, color=C_DARK, weight="bold")
    _arrow(ax, 7.2, 3.85, 6.0, 2.7 + 0.35, C_BLUE)
    ax.text(8.4, 2.7 + 0.35, "MISS", fontsize=11, color=C_BLUE,
            weight="bold", ha="center", va="center")

    # --- Panel 2: Capacity --- working set larger than cache ---
    ax = axes[1]
    # cache holds 4 blocks; we touch 6
    for i in range(4):
        x = 1.0 + i * 1.2
        ax.add_patch(Rectangle((x, 2.0), 1.0, 0.9,
                               facecolor=C_LIGHT, edgecolor=C_GRAY))
        ax.text(x + 0.5, 2.45, f"B{i}", ha="center", va="center",
                fontsize=10, color=C_DARK, weight="bold")
    ax.text(5.0, 1.6, "cache: 4 blocks", ha="center", fontsize=9, color=C_GRAY)
    # working set
    for i in range(6):
        x = 0.6 + i * 1.4
        col = C_AMBER if i >= 4 else C_GREEN
        ax.add_patch(Rectangle((x, 4.1), 1.1, 0.7,
                               facecolor=col, edgecolor="white"))
        ax.text(x + 0.55, 4.45, f"B{i}", ha="center", va="center",
                fontsize=10, color="white", weight="bold")
    ax.text(5.0, 3.6, "working set: 6 blocks  ->  2 evicted",
            ha="center", fontsize=9.5, color="#78350f", weight="bold")

    # --- Panel 3: Conflict --- direct-mapped collision ---
    ax = axes[2]
    # 4 sets, two blocks both map to set 1
    for i in range(4):
        y = 1.0 + i * 0.9
        ax.add_patch(Rectangle((1.2, y), 4.8, 0.7, facecolor="white",
                               edgecolor=C_GRAY, linewidth=1.0))
        ax.text(1.5, y + 0.35, f"set {i}", ha="left", va="center",
                fontsize=9, color=C_GRAY)
    # two addresses hashing to set 1
    ax.text(7.0, 4.5, "addr X", fontsize=10, color=C_DARK, weight="bold")
    ax.text(7.0, 3.7, "addr Y", fontsize=10, color=C_DARK, weight="bold")
    _arrow(ax, 6.9, 4.4, 6.0, 1.9 + 0.35, C_RED)
    _arrow(ax, 6.9, 3.6, 6.0, 1.9 + 0.35, C_RED)
    ax.text(8.4, 1.9 + 0.35, "CONFLICT\nat set 1", fontsize=10, color=C_RED,
            weight="bold", ha="center", va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig4_cache_misses")


# ---------------------------------------------------------------------------
# Figure 5: Branch prediction
# ---------------------------------------------------------------------------
def fig5_branch_prediction():
    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.25)

    # --- Left: 2-bit saturating counter state machine ---
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.4, 5.3)
    ax.set_ylim(-1.6, 2.2)
    ax.set_axis_off()
    ax.set_title("2-bit Saturating Counter (per-branch state)",
                 loc="left", fontsize=11)

    states = [
        ("00\nStrong\nNot Taken", 0.5,  0.5, C_RED),
        ("01\nWeak\nNot Taken",   1.9,  0.5, "#fb923c"),
        ("10\nWeak\nTaken",       3.3,  0.5, "#84cc16"),
        ("11\nStrong\nTaken",     4.7,  0.5, C_GREEN),
    ]
    for txt, x, y, c in states:
        ax.add_patch(Circle((x, y), 0.55, facecolor=c, edgecolor="white",
                            linewidth=1.5))
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.5,
                color="white", weight="bold")

    # transitions: taken -> right, not taken -> left
    for i in range(3):
        x1 = states[i][1] + 0.55
        x2 = states[i+1][1] - 0.55
        # taken (above)
        _arrow(ax, x1, 0.78, x2, 0.78, C_GREEN, lw=1.4)
        ax.text((x1+x2)/2, 1.1, "T", ha="center", color=C_GREEN,
                fontsize=10, weight="bold")
        # not taken (below)
        _arrow(ax, x2, 0.22, x1, 0.22, C_RED, lw=1.4)
        ax.text((x1+x2)/2, -0.1, "NT", ha="center", color=C_RED,
                fontsize=10, weight="bold")
    # self loops at extremes (labeled curve hint)
    ax.annotate("", xy=(0.05, 0.7), xytext=(0.05, 0.3),
                arrowprops=dict(arrowstyle="-|>", color=C_RED,
                                connectionstyle="arc3,rad=-1.2"))
    ax.text(-0.25, 0.5, "NT", ha="right", color=C_RED,
            fontsize=10, weight="bold")
    ax.annotate("", xy=(5.15, 0.3), xytext=(5.15, 0.7),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN,
                                connectionstyle="arc3,rad=-1.2"))
    ax.text(5.25, 0.5, "T", ha="left", color=C_GREEN,
            fontsize=10, weight="bold")
    ax.text(2.6, -1.15,
            "Predict TAKEN if state >= 10. One stray mispredict\n"
            "doesn't immediately flip the prediction.",
            ha="center", fontsize=9, color=C_DARK)

    # --- Right: predictor accuracy bar chart ---
    ax2 = fig.add_subplot(gs[0, 1])
    methods = ["Static\n(always NT)", "1-bit\ncounter",
               "2-bit\ncounter", "TAGE / perceptron\n(modern)"]
    acc = [55, 82, 93, 98]
    colors = [C_GRAY, "#fb923c", C_AMBER, C_GREEN]
    bars = ax2.bar(methods, acc, color=colors, edgecolor="white",
                   linewidth=1.5)
    for b, v in zip(bars, acc):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%",
                 ha="center", fontsize=10, color=C_DARK, weight="bold")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Branch prediction accuracy (%)", fontsize=10)
    ax2.set_title("Why it matters: a misprediction wastes 15-20 cycles",
                  loc="left", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Figure 5. Branch Prediction Keeps the Pipeline Full",
                 x=0.04, ha="left", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_branch_prediction")


# ---------------------------------------------------------------------------
# Figure 6: Out-of-order execution
# ---------------------------------------------------------------------------
def fig6_ooo_execution():
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
    fig.suptitle("Figure 6. Out-of-Order Execution Hides a Load-Use Stall",
                 x=0.04, ha="left", fontsize=13, weight="bold", y=0.99)

    # Same instruction stream in both panels:
    # I1: LOAD  R1, [mem]   (latency 4 cycles)
    # I2: ADD   R2, R1, R3  (depends on I1)
    # I3: SUB   R4, R5, R6  (independent)
    # I4: XOR   R7, R8, R9  (independent)
    # I5: MUL   R10, R4, R7 (depends on I3, I4)

    instr_labels = [
        "I1  LOAD  R1, [mem]",
        "I2  ADD   R2, R1, R3",
        "I3  SUB   R4, R5, R6",
        "I4  XOR   R7, R8, R9",
        "I5  MUL   R10, R4, R7",
    ]
    in_order = {  # cycle ranges (start, end inclusive)
        0: [(1, 4)],                    # LOAD: 4 cycles
        1: [(5, 5), ("stall", 5, 5)],   # actually waits then executes
        2: [(6, 6)],
        3: [(7, 7)],
        4: [(8, 8)],
    }
    # Simpler: in-order with explicit stall cells
    in_order_cells = [
        ("I1", [("EX", 1, 4)]),
        ("I2", [("STALL", 2, 4), ("EX", 5, 5)]),
        ("I3", [("STALL", 3, 5), ("EX", 6, 6)]),
        ("I4", [("STALL", 4, 6), ("EX", 7, 7)]),
        ("I5", [("STALL", 5, 7), ("EX", 8, 8)]),
    ]
    # OoO: I3 and I4 slip ahead of I2 while LOAD's data is in flight
    ooo_cells = [
        ("I1", [("EX", 1, 4)]),
        ("I2", [("WAIT", 2, 4), ("EX", 5, 5)]),
        ("I3", [("EX", 2, 2)]),
        ("I4", [("EX", 3, 3)]),
        ("I5", [("WAIT", 4, 5), ("EX", 6, 6)]),
    ]

    n_cycles = 9

    def draw(ax, cells, title):
        ax.set_xlim(0.5, n_cycles + 1.5)
        ax.set_ylim(-0.5, len(cells) + 0.5)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([c[0] for c in cells][::-1], fontsize=10)
        ax.set_title(title, loc="left", fontsize=11)
        for i, (_, segs) in enumerate(cells):
            y = len(cells) - i - 1
            for tag, c1, c2 in segs:
                if tag == "EX":
                    color = C_GREEN
                    label = "EX"
                elif tag == "STALL":
                    color = C_RED
                    label = "stall"
                else:
                    color = C_AMBER
                    label = "wait"
                ax.add_patch(Rectangle((c1, y - 0.35), c2 - c1 + 1, 0.7,
                                       facecolor=color, edgecolor="white",
                                       linewidth=1.5))
                ax.text((c1 + c2 + 1) / 2, y, label, ha="center",
                        va="center", fontsize=9, color="white",
                        weight="bold")
        ax.set_xticks(range(1, n_cycles + 1))
        ax.set_xticklabels([f"C{c}" for c in range(1, n_cycles + 1)])
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    draw(axes[0], in_order_cells,
         "In-order: I2 stalls on LOAD, I3-I5 wait behind it -> finish at C8")
    draw(axes[1], ooo_cells,
         "Out-of-order: I3, I4 execute while LOAD is in flight -> finish at C6")

    # Annotation
    axes[1].annotate(
        "2 cycles saved", xy=(6.5, 4.0), xytext=(7.6, 4.6),
        fontsize=10, color=C_DARK, weight="bold",
        arrowprops=dict(arrowstyle="->", color=C_DARK))

    handles = [
        mpatches.Patch(color=C_GREEN, label="EX  - executing"),
        mpatches.Patch(color=C_AMBER, label="WAIT  - waiting on operand"),
        mpatches.Patch(color=C_RED,   label="STALL - blocked behind earlier instr"),
    ]
    axes[0].legend(handles=handles, loc="upper right", frameon=False,
                   fontsize=9, bbox_to_anchor=(1.0, 1.35), ncol=3)

    axes[1].set_xlabel("Cycle", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig6_ooo_execution")


# ---------------------------------------------------------------------------
# Figure 7: Multi-core + SMT (hyperthreading)
# ---------------------------------------------------------------------------
def fig7_multicore_smt():
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.95], wspace=0.22)

    # --- Left: 4-core chip with shared L3 ---
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_axis_off()
    ax.set_title("Multi-core CPU: 4 cores, private L1/L2, shared L3",
                 loc="left", fontsize=11)

    # Chip outline
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 11.6, 7.4,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor="#f8fafc", edgecolor=C_GRAY,
                                linewidth=1.5))
    # 4 cores in a 2x2 grid
    positions = [(0.7, 4.4), (6.3, 4.4), (0.7, 1.5), (6.3, 1.5)]
    for idx, (x, y) in enumerate(positions):
        # core box
        _box(ax, x, y, 5.0, 2.6, "", "#dbeafe", ec=C_BLUE, fontcolor=C_DARK)
        ax.text(x + 0.25, y + 2.3, f"Core {idx}", fontsize=10,
                color=C_BLUE, weight="bold")
        # 2 SMT threads
        _box(ax, x + 0.3, y + 0.3, 1.5, 0.6, "Thread A", C_PURPLE,
             fontsize=9, fontcolor="white")
        _box(ax, x + 0.3, y + 1.05, 1.5, 0.6, "Thread B", "#a78bfa",
             fontsize=9, fontcolor="white")
        # shared execution units inside the core
        _box(ax, x + 2.0, y + 0.3, 1.3, 1.35, "ALU\nFPU\nLSU", "#6d28d9",
             fontsize=8, fontcolor="white")
        # private L1 + L2
        _box(ax, x + 3.5, y + 1.05, 1.3, 0.6, "L1 32KB", C_GREEN,
             fontsize=9, fontcolor="white")
        _box(ax, x + 3.5, y + 0.3, 1.3, 0.6, "L2 1MB", C_AMBER,
             fontsize=9, fontcolor="white")

    # Shared L3 + ring bus across the bottom (drawn as bar above chip floor)
    ax.add_patch(FancyBboxPatch((0.7, 0.55), 10.6, 0.55,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#fde68a", edgecolor=C_AMBER,
                                linewidth=1.2))
    ax.text(6.0, 0.83, "Shared L3 cache (24 MB)  +  ring/mesh interconnect",
            ha="center", va="center", fontsize=10, color="#78350f",
            weight="bold")

    # --- Right: SMT explanation ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.set_axis_off()
    ax2.set_title("SMT (Hyperthreading): 1 core, 2 logical threads",
                  loc="left", fontsize=11)

    # Without SMT
    ax2.text(0.3, 7.4, "Without SMT", fontsize=11, color=C_DARK,
             weight="bold")
    n = 8
    for c in range(n):
        used = c not in (2, 4, 6)  # idle slots
        col = C_PURPLE if used else C_LIGHT
        ax2.add_patch(Rectangle((0.3 + c * 1.05, 6.2), 1.0, 0.8,
                                facecolor=col, edgecolor="white",
                                linewidth=1.2))
    ax2.text(0.3, 5.85, "Idle slots when one thread stalls "
                        "(cache miss / mispredict)",
             fontsize=9, color=C_GRAY)

    # With SMT
    ax2.text(0.3, 4.6, "With SMT (2 threads)", fontsize=11,
             color=C_DARK, weight="bold")
    pattern = ["A", "A", "B", "A", "B", "A", "B", "A"]
    for c, t in enumerate(pattern):
        col = C_PURPLE if t == "A" else "#a78bfa"
        ax2.add_patch(Rectangle((0.3 + c * 1.05, 3.4), 1.0, 0.8,
                                facecolor=col, edgecolor="white",
                                linewidth=1.2))
        ax2.text(0.3 + c * 1.05 + 0.5, 3.8, t, ha="center", va="center",
                 color="white", fontsize=10, weight="bold")
    ax2.text(0.3, 2.7,
             "Thread B fills the bubbles left by Thread A.",
             fontsize=9.5, color=C_DARK, weight="bold")
    ax2.text(0.3, 2.1,
             "Real-world gain: typically +20-30% throughput, NOT 2x.\n"
             "Two threads share one set of execution units.",
             fontsize=9, color=C_DARK)

    handles = [
        mpatches.Patch(color=C_PURPLE, label="Thread A active"),
        mpatches.Patch(color="#a78bfa", label="Thread B active"),
        mpatches.Patch(color=C_LIGHT, label="Idle execution slot"),
    ]
    ax2.legend(handles=handles, loc="lower left",
               bbox_to_anchor=(0.0, -0.02), frameon=False, fontsize=9,
               ncol=3)

    fig.suptitle("Figure 7. Multi-Core and SMT: Two Different Ways to "
                 "Add Parallelism",
                 x=0.04, ha="left", fontsize=13, weight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig7_multicore_smt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating Computer Fundamentals - Part 01 (CPU) figures")
    print(f"Output directories:")
    for d in OUT_DIRS:
        print(f"  {d}")
    print()
    fig1_cpu_architecture()
    fig2_pipeline_stages()
    fig3_cache_hierarchy()
    fig4_cache_misses()
    fig5_branch_prediction()
    fig6_ooo_execution()
    fig7_multicore_smt()
    print("\nDone.")


if __name__ == "__main__":
    main()
