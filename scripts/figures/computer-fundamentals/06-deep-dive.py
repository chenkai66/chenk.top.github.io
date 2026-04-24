"""
Figure generator for Computer Fundamentals Part 06: Deep Dive — Series Finale.

Produces 7 production-quality figures that close the series by zooming
out from individual components to the system as a whole, then forward
to the future of computing:

    1. Full system architecture (CPU + cache + memory + IO + network)
    2. Cross-layer optimization (hardware-aware software stack)
    3. Performance counters (PMU events + simulated `perf top`)
    4. Von Neumann bottleneck vs modern mitigations
    5. Heterogeneous computing (CPU + GPU + TPU + NPU)
    6. Six-chapter series journey map (networkx)
    7. Future trends — chiplets, photonic interconnects, quantum

Outputs PNGs (dpi=150) into the EN and ZH asset directories:
    source/_posts/en/computer-fundamentals/06-deep-dive/
    source/_posts/zh/computer-fundamentals/06-deep-dive/

Style: matplotlib + networkx, seaborn-v0_8-whitegrid; brand palette
    blue   #2563eb  (primary  / control plane)
    purple #7c3aed  (accent   / accelerators)
    green  #10b981  (success  / fast / hot)
    amber  #f59e0b  (warning  / bottlenecks / cold)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle, RegularPolygon)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
AMBER = "#f59e0b"
DARK = "#0f172a"
GRAY = "#64748b"
LIGHT = "#f1f5f9"
RED = "#dc2626"

DPI = 150

REPO = Path(__file__).resolve().parents[3]
OUT_DIRS = [
    REPO / "source/_posts/en/computer-fundamentals/06-deep-dive",
    REPO / "source/_posts/zh/computer-fundamentals/06-deep-dive",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.0,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
})


def _save(fig: plt.Figure, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  wrote {name} -> {len(OUT_DIRS)} dirs")


def _box(ax, x, y, w, h, label, color, sub=None, text_color="white",
         fontsize=10, sub_fontsize=8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=color, edgecolor=DARK, linewidth=1.1, alpha=0.95))
    ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0),
            label, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22,
                sub, ha="center", va="center",
                color=text_color, fontsize=sub_fontsize, alpha=0.95)


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.4, style="->",
           connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=12, color=color, lw=lw,
        connectionstyle=connectionstyle))


# ---------------------------------------------------------------------------
# Figure 1 — Full system architecture (everything together)
# ---------------------------------------------------------------------------

def fig1_full_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")

    # ---------- CPU package (top-left) ----------
    ax.add_patch(FancyBboxPatch((0.4, 5.6), 6.4, 4.0,
                                boxstyle="round,pad=0.04,rounding_size=0.25",
                                facecolor="#eef2ff", edgecolor=BLUE, linewidth=2))
    ax.text(3.6, 9.35, "CPU package", ha="center", color=BLUE,
            fontsize=12, fontweight="bold")

    # cores
    for i, x in enumerate([0.7, 2.3, 3.9, 5.5]):
        _box(ax, x, 7.7, 1.3, 1.0, f"Core {i}", BLUE,
             sub="L1 32 KB\nL2 512 KB", fontsize=9, sub_fontsize=7)
    # L3 shared
    _box(ax, 0.7, 6.6, 6.1, 0.9, "Shared L3 cache  ·  16 MB", PURPLE,
         sub="ring / mesh interconnect", fontsize=10, sub_fontsize=7.5)
    # memory controller / IO die
    _box(ax, 0.7, 5.75, 2.9, 0.7, "Integrated memory ctrl", DARK,
         fontsize=9)
    _box(ax, 3.9, 5.75, 2.9, 0.7, "PCIe / IO die", DARK, fontsize=9)

    # ---------- DRAM (right of CPU) ----------
    for i, y in enumerate([8.1, 7.0, 5.9]):
        _box(ax, 7.6, y, 3.0, 0.9,
             f"DDR5 DIMM {i+1}", GREEN,
             sub="51.2 GB/s  ·  100 ns", fontsize=10, sub_fontsize=7.5)
    ax.text(9.1, 9.4, "Main memory (DRAM)", ha="center",
            color=GREEN, fontsize=11, fontweight="bold")

    # CPU <-> DRAM bus
    for y in [8.55, 7.45, 6.35]:
        _arrow(ax, 6.8, y, 7.6, y, color=GREEN, lw=2.0, style="<->")

    # ---------- GPU / accelerators (top-right) ----------
    ax.add_patch(FancyBboxPatch((11.0, 5.6), 4.6, 4.0,
                                boxstyle="round,pad=0.04,rounding_size=0.25",
                                facecolor="#faf5ff", edgecolor=PURPLE, linewidth=2))
    ax.text(13.3, 9.35, "Accelerators", ha="center", color=PURPLE,
            fontsize=12, fontweight="bold")
    _box(ax, 11.2, 7.9, 4.2, 1.1, "GPU  ·  10 000 cores", PURPLE,
         sub="HBM3  ·  3 TB/s", fontsize=10, sub_fontsize=7.5)
    _box(ax, 11.2, 6.7, 4.2, 1.0, "NPU / TPU", PURPLE,
         sub="INT8 matmul  ·  >100 TOPS", fontsize=10, sub_fontsize=7.5)
    _box(ax, 11.2, 5.75, 4.2, 0.8, "DMA engine", DARK,
         sub="zero-copy to system memory", fontsize=9, sub_fontsize=7)

    # ---------- PCIe fabric (middle horizontal) ----------
    ax.add_patch(FancyBboxPatch((0.4, 4.4), 15.2, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=DARK, edgecolor=DARK))
    ax.text(8.0, 4.85, "PCIe 5.0 fabric  ·  64 GB/s per x16  ·  CXL coherent",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")

    # links from CPU IO die to fabric, and from GPU/accel down
    _arrow(ax, 5.3, 5.75, 5.3, 5.3, color=DARK, lw=1.8, style="<->")
    _arrow(ax, 13.3, 5.6, 13.3, 5.3, color=DARK, lw=1.8, style="<->")

    # ---------- Storage tier (bottom-left) ----------
    ax.add_patch(FancyBboxPatch((0.4, 0.6), 6.4, 3.5,
                                boxstyle="round,pad=0.04,rounding_size=0.25",
                                facecolor="#ecfdf5", edgecolor=GREEN, linewidth=2))
    ax.text(3.6, 3.85, "Storage hierarchy", ha="center",
            color=GREEN, fontsize=12, fontweight="bold")
    _box(ax, 0.7, 2.6, 6.0, 0.9, "NVMe SSD  ·  7 GB/s  ·  100 µs", GREEN,
         sub="hot data, OS, app binaries", fontsize=10, sub_fontsize=7.5)
    _box(ax, 0.7, 1.6, 6.0, 0.9, "SATA SSD / 7.2K HDD  ·  500 MB/s · 5 ms",
         BLUE, sub="warm data, libraries, datasets",
         fontsize=10, sub_fontsize=7.5)
    _box(ax, 0.7, 0.7, 6.0, 0.8, "Object storage  ·  S3 / Glacier  ·  s–h",
         AMBER, sub="cold archive, backups",
         fontsize=10, sub_fontsize=7.5)

    # PCIe to NVMe
    _arrow(ax, 3.6, 4.4, 3.6, 3.5, color=DARK, lw=1.8, style="<->")

    # ---------- IO + network (bottom-right) ----------
    ax.add_patch(FancyBboxPatch((7.6, 0.6), 8.0, 3.5,
                                boxstyle="round,pad=0.04,rounding_size=0.25",
                                facecolor="#fffbeb", edgecolor=AMBER, linewidth=2))
    ax.text(11.6, 3.85, "Peripherals & network",
            ha="center", color=AMBER, fontsize=12, fontweight="bold")
    _box(ax, 7.85, 2.7, 3.7, 0.9, "USB 4 / Thunderbolt", AMBER,
         sub="40 Gbps", fontsize=10, sub_fontsize=7.5)
    _box(ax, 11.7, 2.7, 3.8, 0.9, "Display engine / HDMI 2.1", AMBER,
         sub="48 Gbps", fontsize=10, sub_fontsize=7.5)
    _box(ax, 7.85, 1.65, 3.7, 0.9, "NIC  ·  10 / 25 GbE", BLUE,
         sub="RDMA, kernel-bypass", fontsize=10, sub_fontsize=7.5)
    _box(ax, 11.7, 1.65, 3.8, 0.9, "WiFi 6E / 5G modem", BLUE,
         sub="latency-sensitive", fontsize=10, sub_fontsize=7.5)
    _box(ax, 7.85, 0.7, 7.65, 0.8, "BMC  ·  TPM  ·  power & telemetry",
         GRAY, sub="out-of-band management",
         fontsize=10, sub_fontsize=7.5)

    # PCIe to peripherals/NIC
    _arrow(ax, 11.6, 4.4, 11.6, 3.6, color=DARK, lw=1.8, style="<->")

    # latency annotations on the right edge
    ax.text(15.85, 8.55, "ns",  ha="left", color=GREEN, fontsize=9, fontweight="bold")
    ax.text(15.85, 6.85, "ns",  ha="left", color=PURPLE, fontsize=9, fontweight="bold")
    ax.text(15.85, 4.85, "µs",  ha="left", color="white", fontsize=9, fontweight="bold")
    ax.text(15.85, 3.05, "µs",  ha="left", color=GREEN, fontsize=9, fontweight="bold")
    ax.text(15.85, 1.05, "ms+", ha="left", color=AMBER, fontsize=9, fontweight="bold")

    fig.suptitle("Full System Architecture — How Everything Talks",
                 fontsize=16, fontweight="bold", color=DARK, y=0.99)
    fig.tight_layout()
    _save(fig, "fig1_full_system.png")


# ---------------------------------------------------------------------------
# Figure 2 — Cross-layer optimization (hardware-aware software)
# ---------------------------------------------------------------------------

def fig2_cross_layer_optimization() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 7.0),
                                     gridspec_kw={"width_ratios": [1.0, 1.05]})

    # ---------- Left: layer stack with arrows showing co-design ----------
    ax_l.set_xlim(0, 10); ax_l.set_ylim(0, 10); ax_l.axis("off")
    ax_l.set_title("Software stack ↔ hardware co-design",
                   color=DARK, fontsize=13, pad=10)

    layers = [
        ("Application",        "vector DBs, LLMs, games", GREEN),
        ("Framework / runtime","PyTorch, JVM, V8",        BLUE),
        ("Compiler",           "LLVM, GCC, MLIR",         BLUE),
        ("OS / kernel",        "scheduler, page cache, io_uring", PURPLE),
        ("Driver / firmware",  "GPU, NVMe, NIC",          PURPLE),
        ("ISA",                "x86-64, ARMv9, RISC-V",   AMBER),
        ("Microarchitecture",  "OoO, SMT, branch predictor", AMBER),
        ("Silicon",            "transistors, wires, SRAM", DARK),
    ]
    h = 0.95
    y0 = 9.2
    for i, (name, sub, color) in enumerate(layers):
        y = y0 - i * (h + 0.05)
        _box(ax_l, 0.7, y, 6.6, h, name, color,
             sub=sub, fontsize=10.5, sub_fontsize=8)

    # bidirectional co-design arrow
    _arrow(ax_l, 8.2, 1.0, 8.2, 9.7, color=DARK, lw=2.0, style="<->")
    ax_l.text(8.55, 5.3, "co-design\nfeedback", color=DARK,
              fontsize=10, fontweight="bold", rotation=90,
              ha="center", va="center")

    # examples linking layers
    ax_l.text(0.1, 4.85, "•", color=DARK, fontsize=14)
    ax_l.text(0.4, 0.15, "examples: SIMD intrinsics · cache-blocked GEMM · "
              "huge pages · NUMA pinning · branchless code",
              color=GRAY, fontsize=8.5, style="italic")

    # ---------- Right: matmul speedups bar chart ----------
    ax_r.set_title("Matmul speedups from hardware-aware tricks\n"
                   "(1024×1024 FP32, single core, normalized)",
                   color=DARK, fontsize=12, pad=10)

    techniques = [
        "Naïve\ntriple loop",
        "Loop\nreorder (ikj)",
        "Cache\nblocking",
        "+ SIMD\n(AVX2)",
        "+ FMA\n+ unroll",
        "+ Multi-thread\n(OpenMP)",
    ]
    speedups = [1.0, 3.2, 9.4, 32.0, 58.0, 410.0]
    colors = [AMBER, AMBER, BLUE, BLUE, PURPLE, GREEN]

    y = np.arange(len(techniques))
    bars = ax_r.barh(y, speedups, color=colors,
                     edgecolor=DARK, linewidth=0.8, alpha=0.9)
    ax_r.set_yticks(y); ax_r.set_yticklabels(techniques, fontsize=9.5)
    ax_r.invert_yaxis()
    ax_r.set_xscale("log")
    ax_r.set_xlim(0.7, 800)
    ax_r.set_xlabel("Speedup vs. naïve loop  (log scale)")
    for bar, s in zip(bars, speedups):
        ax_r.text(s * 1.08, bar.get_y() + bar.get_height() / 2,
                  f"{s:g}×", va="center", fontsize=9.5,
                  fontweight="bold", color=DARK)

    ax_r.grid(True, axis="x", which="both", alpha=0.4)
    ax_r.text(1.0, 6.05, "same algorithm, ~400× difference — purely from "
              "cache, vectors, and threads",
              color=GRAY, fontsize=8.5, style="italic")

    fig.suptitle("Cross-Layer Optimization — Why Hardware-Aware Code Wins",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig2_cross_layer.png")


# ---------------------------------------------------------------------------
# Figure 3 — Performance counters (PMU) and `perf top` view
# ---------------------------------------------------------------------------

def fig3_performance_counters() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.6),
                                     gridspec_kw={"width_ratios": [1.0, 1.05]})

    # ---------- Left: PMU event categories radar/bar ----------
    ax_l.set_title("PMU events you actually look at",
                   color=DARK, fontsize=13, pad=8)

    categories = [
        "cycles", "instructions", "branch-misses",
        "L1-dcache-load-misses", "LLC-load-misses",
        "dTLB-load-misses", "stalled-cycles-frontend",
        "stalled-cycles-backend",
    ]
    # representative "hot" workload counts per kilo-instruction
    pki = [1230, 1000, 8.4, 22.0, 4.1, 0.8, 180, 260]
    colors_l = [BLUE, BLUE, AMBER, AMBER, RED, AMBER, PURPLE, PURPLE]

    y = np.arange(len(categories))
    ax_l.barh(y, pki, color=colors_l, edgecolor=DARK,
              linewidth=0.8, alpha=0.9)
    ax_l.set_yticks(y); ax_l.set_yticklabels(categories, fontsize=9.5)
    ax_l.invert_yaxis()
    ax_l.set_xscale("log")
    ax_l.set_xlim(0.3, 3000)
    ax_l.set_xlabel("events per 1 000 instructions  (log)")
    for yi, v in zip(y, pki):
        ax_l.text(v * 1.12, yi, f"{v:g}", va="center",
                  fontsize=9, color=DARK, fontweight="bold")

    # legend tiles
    legend = [
        (BLUE,   "throughput"),
        (AMBER,  "speculation / locality"),
        (RED,    "memory bottleneck"),
        (PURPLE, "stalls"),
    ]
    for i, (c, name) in enumerate(legend):
        ax_l.add_patch(Rectangle((1.0, -1.6 + i * -0.45), 0.6, 0.32,
                                 facecolor=c, edgecolor=DARK))
        ax_l.text(1.8, -1.45 + i * -0.45, name, fontsize=8.5, color=DARK,
                  va="center")

    # ---------- Right: simulated `perf top` panel ----------
    ax_r.set_xlim(0, 10); ax_r.set_ylim(0, 10); ax_r.axis("off")
    ax_r.set_title("`perf top` — what's actually burning CPU",
                   color=DARK, fontsize=13, pad=8)

    # terminal frame
    ax_r.add_patch(FancyBboxPatch((0.2, 0.4), 9.6, 9.0,
                                  boxstyle="round,pad=0.02,rounding_size=0.15",
                                  facecolor=DARK, edgecolor=DARK))
    # title bar
    ax_r.add_patch(Rectangle((0.2, 8.7), 9.6, 0.7, facecolor="#1e293b",
                             edgecolor=DARK))
    ax_r.text(0.5, 9.05, "● ● ●", color="#f1f5f9", fontsize=10,
              fontweight="bold")
    ax_r.text(5.0, 9.05, "perf top — Hz: 4000   Comm: app",
              ha="center", color="#cbd5e1", fontsize=9.5)

    header = "  Overhead  Symbol                                Module"
    ax_r.text(0.45, 8.25, header, color="#94a3b8", fontsize=9.5,
              family="monospace")
    rows = [
        (32.4, "memcpy_avx2",                   "libc.so.6",   GREEN),
        (18.7, "matmul_inner_loop",             "app",         BLUE),
        (11.2, "__lock_acquire",                "kernel",      AMBER),
        ( 8.6, "page_fault_handler",            "kernel",      AMBER),
        ( 6.1, "json_parse",                    "libjson.so",  PURPLE),
        ( 4.3, "tcp_recvmsg",                   "kernel",      BLUE),
        ( 3.0, "memset_avx2",                   "libc.so.6",   GREEN),
        ( 2.4, "sha256_compress",               "libcrypto",   PURPLE),
        ( 1.8, "futex_wait",                    "kernel",      AMBER),
        ( 1.2, "rcu_read_unlock",               "kernel",      GRAY),
    ]
    y = 7.7
    for pct, sym, mod, color in rows:
        ax_r.text(0.45, y, f"  {pct:5.1f}%", color=color, fontsize=9.8,
                  family="monospace", fontweight="bold")
        ax_r.text(2.3, y, sym, color="#e2e8f0", fontsize=9.8,
                  family="monospace")
        ax_r.text(7.2, y, mod, color="#94a3b8", fontsize=9.5,
                  family="monospace")
        y -= 0.55

    # caption
    ax_r.text(5.0, 0.85,
              "top symbol = memcpy_avx2  →  bandwidth-bound, not compute-bound",
              ha="center", color=GREEN, fontsize=9.5,
              fontweight="bold", family="monospace")

    fig.suptitle("Performance Counters — Measure Before You Optimize",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig3_perf_counters.png")


# ---------------------------------------------------------------------------
# Figure 4 — Von Neumann bottleneck vs modern designs
# ---------------------------------------------------------------------------

def fig4_von_neumann_vs_modern() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 7.0))

    # ---------- Left: classic Von Neumann ----------
    ax_l.set_xlim(0, 10); ax_l.set_ylim(0, 10); ax_l.axis("off")
    ax_l.set_title("Classic Von Neumann (1945)",
                   color=AMBER, fontsize=13, pad=8)

    _box(ax_l, 1.0, 7.3, 3.4, 1.6, "CPU", BLUE,
         sub="single ALU\nfetch–decode–exec", fontsize=12, sub_fontsize=8.5)
    _box(ax_l, 5.6, 7.3, 3.4, 1.6, "Memory", GREEN,
         sub="instructions + data\nshared address space",
         fontsize=12, sub_fontsize=8.5)

    # the single skinny bus = the bottleneck
    ax_l.add_patch(FancyBboxPatch((4.4, 7.95), 1.2, 0.35,
                                  boxstyle="round,pad=0.01,rounding_size=0.05",
                                  facecolor=AMBER, edgecolor=DARK))
    ax_l.text(5.0, 8.13, "BUS", ha="center", va="center",
              color="white", fontsize=10, fontweight="bold")
    ax_l.annotate("", xy=(5.6, 8.13), xytext=(4.4, 8.13),
                  arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2.4))

    ax_l.text(5.0, 6.6, "Von Neumann bottleneck:\n"
              "every instruction + every operand\nflows through one narrow channel",
              ha="center", color=AMBER, fontsize=10.5, fontweight="bold")

    # latency wall illustration
    xs = np.linspace(0.6, 9.4, 200)
    cpu = 100 * 1.5 ** (xs - 0.6)        # CPU clock growth
    mem = 100 * 1.07 ** (xs - 0.6)       # memory speed growth
    # rescale into pixel coords
    def _rescale(v):
        return 1.0 + 4.5 * (np.log10(v) - 2) / 2.5
    ax_l.plot(xs, _rescale(cpu), color=BLUE, lw=2.2, label="CPU performance")
    ax_l.plot(xs, _rescale(mem), color=AMBER, lw=2.2, label="DRAM performance")
    ax_l.fill_between(xs, _rescale(mem), _rescale(cpu),
                      color=RED, alpha=0.10)
    ax_l.text(8.6, 5.3, "memory\nwall", color=RED, fontsize=10,
              fontweight="bold", ha="center")
    ax_l.text(0.6, 0.6, "year →", color=GRAY, fontsize=9)
    ax_l.legend(loc="upper left", bbox_to_anchor=(0.02, 0.42),
                fontsize=9, frameon=True)

    # ---------- Right: modern mitigations ----------
    ax_r.set_xlim(0, 10); ax_r.set_ylim(0, 10); ax_r.axis("off")
    ax_r.set_title("Modern designs that route around the wall",
                   color=GREEN, fontsize=13, pad=8)

    _box(ax_r, 0.6, 7.6, 3.0, 1.6, "Core", BLUE,
         sub="OoO + SMT\nbranch predictor",
         fontsize=11, sub_fontsize=8)
    _box(ax_r, 4.0, 8.4, 2.6, 0.8, "L1  (32 KB)", GREEN, fontsize=10)
    _box(ax_r, 4.0, 7.5, 2.6, 0.8, "L2  (512 KB)", GREEN, fontsize=10)
    _box(ax_r, 4.0, 6.6, 2.6, 0.8, "L3  (16 MB)",  PURPLE, fontsize=10)
    _box(ax_r, 7.0, 7.6, 2.6, 1.6, "DRAM", AMBER,
         sub="100 ns", fontsize=11, sub_fontsize=8)

    # arrows between
    for y in [8.8, 7.9, 7.0]:
        _arrow(ax_r, 3.6, y, 4.0, y, color=DARK, style="<->")
    _arrow(ax_r, 6.6, 7.9, 7.0, 8.4, color=DARK, style="<->")
    _arrow(ax_r, 6.6, 7.0, 7.0, 7.9, color=DARK, style="<->")

    # mitigation list
    mitigations = [
        ("Multi-level cache",       GREEN),
        ("Hardware prefetcher",     GREEN),
        ("Out-of-order execution",  BLUE),
        ("SIMD / vector units",     BLUE),
        ("Multi-core + SMT",        PURPLE),
        ("Harvard split I/D L1",    PURPLE),
        ("HBM stacked DRAM",        AMBER),
        ("Compute-in-memory (CIM)", RED),
    ]
    ax_r.text(0.6, 5.4, "Mitigations:", color=DARK, fontsize=11,
              fontweight="bold")
    for i, (m, c) in enumerate(mitigations):
        col = i % 2
        row = i // 2
        x = 0.7 + col * 4.7
        y = 4.7 - row * 0.85
        ax_r.add_patch(Circle((x + 0.15, y + 0.15), 0.13,
                              facecolor=c, edgecolor=DARK))
        ax_r.text(x + 0.45, y + 0.12, m, color=DARK, fontsize=10,
                  va="center")

    ax_r.text(5.0, 0.7,
              "result: effective memory latency drops from ~100 ns to ~2 ns",
              ha="center", color=GREEN, fontsize=10, fontweight="bold")

    fig.suptitle("Von Neumann Bottleneck — Then and Now",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig4_von_neumann.png")


# ---------------------------------------------------------------------------
# Figure 5 — Heterogeneous computing (CPU + GPU + TPU + NPU)
# ---------------------------------------------------------------------------

def fig5_heterogeneous_computing() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.8),
                                     gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---------- Left: chip layout with shared memory fabric ----------
    ax_l.set_xlim(0, 10); ax_l.set_ylim(0, 10); ax_l.axis("off")
    ax_l.set_title("Heterogeneous SoC layout",
                   color=DARK, fontsize=13, pad=8)

    # interconnect (center bus)
    ax_l.add_patch(FancyBboxPatch((0.6, 4.5), 8.8, 1.0,
                                  boxstyle="round,pad=0.02,rounding_size=0.15",
                                  facecolor=DARK, edgecolor=DARK))
    ax_l.text(5.0, 5.0, "Coherent fabric  ·  CXL  ·  shared HBM",
              ha="center", va="center", color="white",
              fontsize=11, fontweight="bold")

    # accelerators top
    _box(ax_l, 0.6, 6.8, 2.0, 2.6, "CPU", BLUE,
         sub="control,\nbranchy code,\nlow latency",
         fontsize=12, sub_fontsize=8.5)
    _box(ax_l, 2.85, 6.8, 2.0, 2.6, "GPU", PURPLE,
         sub="dense FP32/FP16,\ngraphics,\ntraining",
         fontsize=12, sub_fontsize=8.5)
    _box(ax_l, 5.1, 6.8, 2.0, 2.6, "TPU", GREEN,
         sub="systolic matmul,\nINT8/BF16,\ndatacenter",
         fontsize=12, sub_fontsize=8.5)
    _box(ax_l, 7.35, 6.8, 2.0, 2.6, "NPU", AMBER,
         sub="on-device AI,\nlow power,\nINT8",
         fontsize=12, sub_fontsize=8.5)

    # arrows down to bus
    for x in [1.6, 3.85, 6.1, 8.35]:
        _arrow(ax_l, x, 6.8, x, 5.5, color=DARK, lw=1.6, style="<->")

    # bottom: memory + IO
    _box(ax_l, 0.6, 1.4, 4.2, 2.6, "HBM3 stacks", GREEN,
         sub="3 TB/s shared", fontsize=12, sub_fontsize=9)
    _box(ax_l, 5.0, 1.4, 4.4, 2.6, "DDR5 + PCIe",  BLUE,
         sub="host + storage", fontsize=12, sub_fontsize=9)
    _arrow(ax_l, 2.7, 4.5, 2.7, 4.0, color=DARK, lw=1.6, style="<->")
    _arrow(ax_l, 7.2, 4.5, 7.2, 4.0, color=DARK, lw=1.6, style="<->")

    # ---------- Right: throughput vs efficiency scatter ----------
    ax_r.set_title("Compute throughput vs energy efficiency",
                   color=DARK, fontsize=13, pad=8)

    # (TFLOPS-equivalent, perf/W TOPS/W, label, color, size)
    units = [
        ("CPU  AVX-512",  3.0,    0.10,  BLUE),
        ("GPU  H100",     67.0,   2.50,  PURPLE),
        ("TPU  v5p",      459.0,  3.80,  GREEN),
        ("NPU  mobile",   30.0,   8.00,  AMBER),
        ("FPGA",          18.0,   1.20,  GRAY),
        ("ASIC  custom",  900.0,  12.00, RED),
    ]
    for name, tflops, eff, color in units:
        ax_r.scatter(tflops, eff, s=320, color=color,
                     edgecolor=DARK, linewidth=1.2, zorder=3, alpha=0.92)
        ax_r.annotate(name, (tflops, eff), xytext=(8, 6),
                      textcoords="offset points", fontsize=9.5,
                      fontweight="bold", color=DARK)

    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlim(1.0, 3000)
    ax_r.set_ylim(0.05, 30)
    ax_r.set_xlabel("Peak throughput  (TFLOPS / TOPS, log)")
    ax_r.set_ylabel("Energy efficiency  (TOPS per Watt, log)")
    ax_r.grid(True, which="both", alpha=0.4)

    # frontier line
    fx = np.logspace(0, 3, 100)
    fy = 0.04 * fx ** 0.65
    ax_r.plot(fx, fy, "--", color=GRAY, lw=1.2, alpha=0.7)
    ax_r.text(700, 7.5, "specialization\nfrontier", color=GRAY,
              fontsize=9, ha="center", style="italic")

    fig.suptitle("Heterogeneous Computing — Right Tool For Each Job",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig5_heterogeneous.png")


# ---------------------------------------------------------------------------
# Figure 6 — Six-chapter series journey map (networkx)
# ---------------------------------------------------------------------------

def fig6_series_journey() -> None:
    fig, ax = plt.subplots(figsize=(14, 8.4))
    ax.set_xlim(-0.8, 12.4); ax.set_ylim(-0.6, 8.6); ax.axis("off")

    G = nx.DiGraph()

    # six chapters along an S-curve
    chapters = [
        ("01", "Bits, Bytes\n& the CPU",        1.2, 6.8, BLUE),
        ("02", "Memory &\ncache",               4.0, 7.4, BLUE),
        ("03", "Storage —\nHDD, SSD, RAID",     7.0, 7.0, GREEN),
        ("04", "Motherboard\n& GPU",            10.4, 6.0, PURPLE),
        ("05", "Network,\npower, cooling",      9.0, 3.0, AMBER),
        ("06", "Deep dive —\nthe whole system", 4.4, 1.6, RED),
    ]
    for cid, label, x, y, color in chapters:
        G.add_node(cid, pos=(x, y), label=label, color=color)

    edges = [("01", "02"), ("02", "03"), ("03", "04"),
             ("04", "05"), ("05", "06")]
    G.add_edges_from(edges)

    # cross-references that this finale stitches together
    cross = [("01", "06"), ("02", "06"), ("03", "06"),
             ("04", "06"), ("05", "06")]
    G.add_edges_from(cross)

    pos = nx.get_node_attributes(G, "pos")

    # main spine arrows
    for u, v in edges:
        _arrow(ax, pos[u][0], pos[u][1], pos[v][0], pos[v][1],
               color=DARK, lw=2.0, style="->",
               connectionstyle="arc3,rad=0.18")

    # cross-ref arrows (faded)
    for u, v in cross:
        _arrow(ax, pos[u][0], pos[u][1], pos[v][0], pos[v][1],
               color=GRAY, lw=1.0, style="->",
               connectionstyle="arc3,rad=-0.25")

    # nodes
    for cid, _, x, y, color in chapters:
        ax.add_patch(Circle((x, y), 0.85, facecolor=color,
                            edgecolor=DARK, linewidth=1.6, zorder=3))
        ax.text(x, y + 0.18, cid, ha="center", va="center",
                color="white", fontsize=15, fontweight="bold", zorder=4)
        ax.text(x, y - 0.32, "Part", ha="center", va="center",
                color="white", fontsize=8, zorder=4)
        # label below
        label = next(c[1] for c in chapters if c[0] == cid)
        ax.text(x, y - 1.35, label, ha="center", va="top",
                color=DARK, fontsize=10, fontweight="bold")

    # banner
    ax.add_patch(FancyBboxPatch((0.0, 8.0), 12.0, 0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=DARK, edgecolor=DARK))
    ax.text(6.0, 8.3, "Six-Chapter Series Journey  ·  finale stitches them together",
            ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")

    # finale star
    star = RegularPolygon((4.4, 1.6), numVertices=5, radius=1.25,
                          facecolor="none", edgecolor=RED,
                          linewidth=1.4, alpha=0.6, zorder=2)
    ax.add_patch(star)
    ax.text(4.4, -0.25, "★ you are here", ha="center", color=RED,
            fontsize=11, fontweight="bold")

    fig.suptitle("Series Journey Map — From Bits to System",
                 fontsize=16, fontweight="bold", color=DARK, y=0.995)
    fig.tight_layout()
    _save(fig, "fig6_series_journey.png")


# ---------------------------------------------------------------------------
# Figure 7 — Future trends (chiplets, photonic, quantum)
# ---------------------------------------------------------------------------

def fig7_future_trends() -> None:
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95],
                          hspace=0.42, wspace=0.28)

    # ---------- Top-left: chiplets ----------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")
    ax1.set_title("Chiplets — disaggregated dies",
                  color=BLUE, fontsize=12, pad=8)

    # interposer
    ax1.add_patch(FancyBboxPatch((0.6, 0.8), 8.8, 8.4,
                                 boxstyle="round,pad=0.02,rounding_size=0.2",
                                 facecolor="#dbeafe", edgecolor=BLUE,
                                 linewidth=1.6))
    ax1.text(5.0, 9.5, "Si interposer / fan-out package",
             ha="center", color=BLUE, fontsize=8.5, style="italic")
    # tiles
    tiles = [
        (1.2, 5.3, "CCD\n(cores)", PURPLE),
        (4.0, 5.3, "CCD\n(cores)", PURPLE),
        (6.8, 5.3, "IO die\n+ MCs",  BLUE),
        (1.2, 1.6, "GPU tile",       GREEN),
        (4.0, 1.6, "HBM stack",      AMBER),
        (6.8, 1.6, "HBM stack",      AMBER),
    ]
    for x, y, label, color in tiles:
        _box(ax1, x, y, 2.2, 2.4, label, color, fontsize=9.5)
    # bridges
    for (x1, y1), (x2, y2) in [
        ((3.4, 6.5), (4.0, 6.5)),
        ((6.2, 6.5), (6.8, 6.5)),
        ((3.4, 2.8), (4.0, 2.8)),
        ((6.2, 2.8), (6.8, 2.8)),
    ]:
        ax1.plot([x1, x2], [y1, y2], color=DARK, lw=1.6, zorder=2)
    ax1.text(5.0, 0.25, "yield + cost: mix nodes, ship faster",
             ha="center", color=GRAY, fontsize=8.5, style="italic")

    # ---------- Top-middle: photonic interconnect ----------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.set_title("Photonic interconnect — light, not copper",
                  color=PURPLE, fontsize=12, pad=8)

    _box(ax2, 0.4, 4.0, 1.8, 2.0, "die A", BLUE, fontsize=10)
    _box(ax2, 7.8, 4.0, 1.8, 2.0, "die B", BLUE, fontsize=10)
    # waveguide
    ax2.add_patch(Rectangle((2.2, 4.7), 5.6, 0.6,
                            facecolor="#f3e8ff", edgecolor=PURPLE,
                            linewidth=1.2))
    # wavelengths
    waves = [(PURPLE, 4.95), (BLUE, 5.05), (GREEN, 5.15)]
    xs = np.linspace(2.2, 7.8, 200)
    for color, baseline in waves:
        ys = baseline + 0.07 * np.sin((xs - 2.2) * 7)
        ax2.plot(xs, ys, color=color, lw=1.4, alpha=0.85)
    ax2.text(5.0, 6.1, "wavelength-division multiplexing",
             ha="center", color=PURPLE, fontsize=8.5, style="italic")

    # bandwidth comparison bars
    ax_b = ax2.inset_axes([0.05, 0.05, 0.9, 0.32])
    labels = ["Cu trace", "PCIe 5.0", "Optical link"]
    vals = [40, 64, 1600]
    colors_b = [AMBER, BLUE, PURPLE]
    y_pos = np.arange(len(labels))
    bars = ax_b.barh(y_pos, vals, color=colors_b, edgecolor=DARK,
                     linewidth=0.8)
    ax_b.set_yticks(y_pos); ax_b.set_yticklabels(labels)
    ax_b.set_xscale("log"); ax_b.set_xlim(10, 4000)
    ax_b.set_xlabel("Gb/s per lane (log)", fontsize=8)
    ax_b.tick_params(labelsize=8)
    for b, v in zip(bars, vals):
        ax_b.text(v * 1.1, b.get_y() + b.get_height() / 2,
                  f"{v}", va="center", fontsize=8, fontweight="bold")

    # ---------- Top-right: quantum ----------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis("off")
    ax3.set_title("Quantum — qubits, not bits",
                  color=GREEN, fontsize=12, pad=8)

    # qubit lattice
    np.random.seed(7)
    pts = [(2 + 2 * (i % 3), 5 + 2 * (i // 3))
           for i in range(9)]
    for (x, y) in pts:
        ax3.add_patch(Circle((x, y), 0.45, facecolor=GREEN,
                             edgecolor=DARK, linewidth=1.2, alpha=0.85))
        ax3.text(x, y, "|ψ⟩", ha="center", va="center",
                 color="white", fontsize=8.5, fontweight="bold")
    # entanglement edges
    for i, j in [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                 (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]:
        x1, y1 = pts[i]; x2, y2 = pts[j]
        ax3.plot([x1, x2], [y1, y2], color=PURPLE, lw=1.0, alpha=0.6)

    # scaling note
    ax3.add_patch(FancyBboxPatch((0.6, 0.6), 8.8, 3.2,
                                 boxstyle="round,pad=0.02,rounding_size=0.15",
                                 facecolor="#ecfdf5", edgecolor=GREEN,
                                 linewidth=1.2))
    ax3.text(5.0, 2.95, "n qubits  ⇒  2ⁿ amplitudes",
             ha="center", color=GREEN, fontsize=11, fontweight="bold")
    ax3.text(5.0, 2.05, "Shor · Grover · QAOA · VQE",
             ha="center", color=DARK, fontsize=9, style="italic")
    ax3.text(5.0, 1.20, "still: noisy, cryogenic, narrow workloads",
             ha="center", color=AMBER, fontsize=9)

    # ---------- Bottom: timeline ----------
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_xlim(2018, 2040); ax4.set_ylim(-0.5, 4.5)
    ax4.axis("off")
    ax4.set_title("Roadmap — how the next 15 years could unfold",
                  color=DARK, fontsize=13, pad=10)

    # central spine
    ax4.plot([2018, 2040], [2.0, 2.0], color=DARK, lw=2.4, zorder=1)
    for yr in range(2018, 2041, 2):
        ax4.plot([yr, yr], [1.92, 2.08], color=DARK, lw=1.0)
        ax4.text(yr, 1.55, str(yr), ha="center", color=GRAY, fontsize=8.5)

    milestones = [
        (2019, 3.4, BLUE,
         "AMD Zen 2 · first\nmainstream chiplet CPU"),
        (2022, 3.4, GREEN,
         "Apple M1 Ultra · UltraFusion\ndie-to-die bridge"),
        (2024, 0.6, AMBER,
         "Intel + Lightmatter\nco-packaged optics demos"),
        (2026, 3.4, PURPLE,
         "CXL 3.0 pooled memory\nin hyperscale fleets"),
        (2029, 0.6, GREEN,
         "wafer-scale + photonic IO\nin AI training racks"),
        (2032, 3.4, RED,
         "early fault-tolerant\nquantum machines"),
        (2038, 0.6, DARK,
         "post–Moore era:\nspecialization everywhere"),
    ]
    for x, y, color, txt in milestones:
        ax4.add_patch(Circle((x, 2.0), 0.18, facecolor=color,
                             edgecolor=DARK, linewidth=1.2, zorder=3))
        ax4.plot([x, x], [2.0, y - (0.15 if y > 2 else -0.15)],
                 color=color, lw=1.2)
        ax4.add_patch(FancyBboxPatch(
            (x - 1.1, y - 0.45), 2.2, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="white", edgecolor=color, linewidth=1.4))
        ax4.text(x, y, txt, ha="center", va="center",
                 color=DARK, fontsize=8.5, fontweight="bold")

    fig.suptitle("Future Trends — Chiplets · Photonic · Quantum",
                 fontsize=16, fontweight="bold", color=DARK, y=0.995)
    fig.tight_layout()
    _save(fig, "fig7_future_trends.png")


# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating deep-dive (finale) figures to:")
    for d in OUT_DIRS:
        print(f"  - {d}")
    fig1_full_system_architecture()
    fig2_cross_layer_optimization()
    fig3_performance_counters()
    fig4_von_neumann_vs_modern()
    fig5_heterogeneous_computing()
    fig6_series_journey()
    fig7_future_trends()
    print("Done.")


if __name__ == "__main__":
    main()
