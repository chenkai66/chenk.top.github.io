"""
Figure generation script for Computer Fundamentals Chapter 02:
"Memory & High-Speed Cache".

Generates 7 figures used in BOTH the EN and ZH versions of the article.
Each figure teaches one core memory concept cleanly, in a flat, modern,
data-visualization style: generous whitespace, saturated palette, clear
focal points, no decorative chrome.

Figures:
    fig1_memory_hierarchy   The classic memory pyramid: register, L1/L2/L3
                            cache, RAM, SSD, HDD, with size and latency
                            annotated on a log scale.
    fig2_dram_vs_sram       Side-by-side schematic of a DRAM 1T1C cell and
                            an SRAM 6T cell, with cost/speed trade-off
                            summary.
    fig3_virtual_memory     Virtual memory page table translation: virtual
                            address split into VPN + offset, page table
                            lookup, physical frame mapping.
    fig4_tlb                TLB (Translation Lookaside Buffer) flow: hit
                            path versus miss-then-page-walk path, with
                            timing comparison.
    fig5_ecc_correction     ECC memory error correction: parity bits added,
                            single-bit flip detected and corrected via
                            Hamming SEC-DED.
    fig6_numa               NUMA topology: two CPU sockets, local versus
                            remote memory access, latency asymmetry.
    fig7_memory_channels    Memory channel scaling: single, dual, quad
                            channel bandwidth as parallel data lanes.

Usage:
    python3 scripts/figures/computer-fundamentals/02-memory.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages. Parent folders are created
    if missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Polygon

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = "#f8fafc"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/"
    "computer-fundamentals/02-memory"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/"
    "computer-fundamentals/02-memory"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def _strip(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1: Memory hierarchy pyramid
# ---------------------------------------------------------------------------
def fig1_memory_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.2))
    _strip(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    levels = [
        ("Registers",   "~1 KB",       "0.3 ns",  C_DARK,   7.0, 1.4),
        ("L1 Cache",    "32-64 KB",    "1 ns",    C_BLUE,   6.2, 2.0),
        ("L2 Cache",    "256 KB-1 MB", "4 ns",    C_PURPLE, 5.4, 2.6),
        ("L3 Cache",    "8-96 MB",     "15 ns",   C_GREEN,  4.6, 3.4),
        ("DRAM",        "8-128 GB",    "100 ns",  C_AMBER,  3.6, 4.4),
        ("SSD (NVMe)",  "0.5-8 TB",    "100 us",  C_GRAY,   2.4, 5.6),
        ("HDD",         "1-20 TB",     "10 ms",   COLORS["text2"], 1.0, 6.8),
    ]

    cx = 6.0
    for label, cap, lat, color, y, w in levels:
        x0, x1 = cx - w / 2, cx + w / 2
        poly = Polygon(
            [[x0, y - 0.35], [x1, y - 0.35],
             [x1 - 0.07, y + 0.35], [x0 + 0.07, y + 0.35]],
            facecolor=color, edgecolor="white", linewidth=1.6, alpha=0.92,
        )
        ax.add_patch(poly)
        ax.text(cx, y, label, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="white")
        # right side: capacity
        ax.text(cx + w / 2 + 0.25, y + 0.08, cap, ha="left", va="center",
                fontsize=10, color=C_DARK, fontweight="bold")
        ax.text(cx + w / 2 + 0.25, y - 0.18, lat, ha="left", va="center",
                fontsize=9, color=C_GRAY)
        # left side: relative cost / position
        ax.text(cx - w / 2 - 0.25, y, "", ha="right", va="center")

    # Speed arrow (left)
    arr = FancyArrowPatch((1.2, 7.4), (1.2, 0.6),
                          arrowstyle="->", mutation_scale=18,
                          color=C_DARK, linewidth=2)
    ax.add_patch(arr)
    ax.text(0.95, 4.0, "Faster   <-  Speed  ->   Slower",
            rotation=90, ha="center", va="center",
            fontsize=10, color=C_DARK, fontweight="bold")

    # Capacity / cost arrow (right)
    arr2 = FancyArrowPatch((10.8, 0.6), (10.8, 7.4),
                           arrowstyle="->", mutation_scale=18,
                           color=C_DARK, linewidth=2)
    ax.add_patch(arr2)
    ax.text(11.05, 4.0, "Smaller  <-  Capacity / $-per-GB  ->  Larger",
            rotation=90, ha="center", va="center",
            fontsize=10, color=C_DARK, fontweight="bold")

    ax.text(cx, 7.75, "Memory Hierarchy: Speed vs Capacity Trade-off",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)
    ax.text(cx, 0.15, "Each level is roughly 10x larger and 10x slower than the level above",
            ha="center", fontsize=10, color=C_GRAY, style="italic")

    save(fig, "fig1_memory_hierarchy")


# ---------------------------------------------------------------------------
# Figure 2: DRAM vs SRAM cell
# ---------------------------------------------------------------------------
def fig2_dram_vs_sram() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ---- DRAM (1T1C) ----
    ax = axes[0]
    _strip(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_title("DRAM cell  (1 transistor + 1 capacitor)",
                 fontsize=12.5, fontweight="bold", color=C_DARK, pad=14)

    # word line
    ax.plot([1, 9], [6.5, 6.5], color=C_BLUE, linewidth=2.4)
    ax.text(0.85, 6.5, "WL", ha="right", va="center",
            color=C_BLUE, fontsize=11, fontweight="bold")
    # bit line
    ax.plot([3.5, 3.5], [1, 7.5], color=C_PURPLE, linewidth=2.4)
    ax.text(3.5, 7.7, "BL", ha="center", va="bottom",
            color=C_PURPLE, fontsize=11, fontweight="bold")
    # transistor (simple symbol)
    tr = Rectangle((3.0, 5.4), 1.0, 0.9,
                   facecolor=C_LIGHT, edgecolor=C_DARK, linewidth=1.6)
    ax.add_patch(tr)
    ax.text(3.5, 5.85, "T", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_DARK)
    # gate connection from WL
    ax.plot([3.5, 3.5], [6.5, 6.3], color=C_DARK, linewidth=1.4)
    # capacitor below transistor
    ax.plot([3.5, 3.5], [5.4, 4.1], color=C_DARK, linewidth=1.4)
    ax.plot([2.7, 4.3], [4.1, 4.1], color=C_DARK, linewidth=2.6)
    ax.plot([2.9, 4.1], [3.7, 3.7], color=C_DARK, linewidth=2.6)
    ax.text(4.5, 3.9, "C  (stores charge = 1 bit)",
            ha="left", va="center", fontsize=10, color=C_DARK)
    # ground
    ax.plot([3.5, 3.5], [3.7, 3.0], color=C_DARK, linewidth=1.4)
    ax.plot([3.0, 4.0], [3.0, 3.0], color=C_DARK, linewidth=1.6)
    ax.plot([3.2, 3.8], [2.78, 2.78], color=C_DARK, linewidth=1.4)
    ax.plot([3.35, 3.65], [2.55, 2.55], color=C_DARK, linewidth=1.4)

    # info box
    info = ("- Density: HIGH (1T1C ~ 6 F^2)\n"
            "- Cost / bit: LOW\n"
            "- Speed: ~50 ns access\n"
            "- Volatile: needs refresh every ~64 ms\n"
            "- Used as: main memory (DDR)")
    ax.text(5.5, 1.6, info, ha="left", va="bottom", fontsize=9.8,
            color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor=C_BG, edgecolor=C_BLUE, linewidth=1.2))

    # ---- SRAM (6T) ----
    ax = axes[1]
    _strip(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_title("SRAM cell  (6 transistors, cross-coupled latch)",
                 fontsize=12.5, fontweight="bold", color=C_DARK, pad=14)

    # word line
    ax.plot([0.5, 9.5], [6.7, 6.7], color=C_BLUE, linewidth=2.4)
    ax.text(0.35, 6.7, "WL", ha="right", va="center",
            color=C_BLUE, fontsize=11, fontweight="bold")
    # bit lines (BL, BL_bar)
    ax.plot([2.0, 2.0], [1.0, 7.5], color=C_PURPLE, linewidth=2.0)
    ax.plot([8.0, 8.0], [1.0, 7.5], color=C_PURPLE, linewidth=2.0)
    ax.text(2.0, 7.7, "BL",     ha="center", va="bottom",
            color=C_PURPLE, fontsize=10, fontweight="bold")
    ax.text(8.0, 7.7, "BL̅", ha="center", va="bottom",
            color=C_PURPLE, fontsize=10, fontweight="bold")

    # access transistors
    for x in (2.6, 7.4):
        r = Rectangle((x - 0.35, 5.9), 0.7, 0.7,
                      facecolor=C_LIGHT, edgecolor=C_DARK, linewidth=1.4)
        ax.add_patch(r)
    # cross-coupled inverters (two triangles facing each other)
    inv1 = Polygon([[3.6, 5.0], [4.7, 4.5], [3.6, 4.0]],
                   facecolor=C_GREEN, edgecolor=C_DARK, linewidth=1.4, alpha=0.85)
    inv2 = Polygon([[6.4, 5.0], [5.3, 4.5], [6.4, 4.0]],
                   facecolor=C_GREEN, edgecolor=C_DARK, linewidth=1.4, alpha=0.85)
    ax.add_patch(inv1)
    ax.add_patch(inv2)
    # feedback wires
    ax.plot([4.7, 5.0, 5.0, 5.3], [4.5, 4.5, 4.5, 4.5],
            color=C_DARK, linewidth=1.2)
    ax.plot([3.6, 3.2, 3.2, 6.8, 6.8, 6.4],
            [4.5, 4.5, 3.6, 3.6, 4.5, 4.5],
            color=C_DARK, linewidth=1.2)
    # connections to access
    ax.plot([2.6, 2.6, 3.6], [5.9, 4.5, 4.5], color=C_DARK, linewidth=1.2)
    ax.plot([7.4, 7.4, 6.4], [5.9, 4.5, 4.5], color=C_DARK, linewidth=1.2)
    ax.plot([2.0, 2.6], [6.2, 6.2], color=C_DARK, linewidth=1.0)
    ax.plot([7.4, 8.0], [6.2, 6.2], color=C_DARK, linewidth=1.0)

    info = ("- Density: LOW (6T ~ 120 F^2)\n"
            "- Cost / bit: HIGH (~20x DRAM)\n"
            "- Speed: ~1 ns access\n"
            "- Volatile but no refresh needed\n"
            "- Used as: CPU caches (L1/L2/L3)")
    ax.text(0.4, 1.6, info, ha="left", va="bottom", fontsize=9.8,
            color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor=C_BG, edgecolor=C_GREEN, linewidth=1.2))

    fig.suptitle("DRAM vs SRAM: why caches are small but fast",
                 fontsize=14, fontweight="bold", color=C_DARK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig2_dram_vs_sram")


# ---------------------------------------------------------------------------
# Figure 3: Virtual memory page table translation
# ---------------------------------------------------------------------------
def fig3_virtual_memory() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    _strip(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    ax.text(7, 7.55, "Virtual to Physical Address Translation",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)

    # Virtual address (split into VPN + offset)
    ax.text(0.3, 6.6, "Virtual address (CPU view)",
            fontsize=10.5, fontweight="bold", color=C_DARK)
    vpn = Rectangle((0.3, 5.5), 3.5, 0.8,
                    facecolor=C_BLUE, edgecolor="white", linewidth=2)
    off = Rectangle((3.8, 5.5), 1.6, 0.8,
                    facecolor=C_AMBER, edgecolor="white", linewidth=2)
    ax.add_patch(vpn)
    ax.add_patch(off)
    ax.text(2.05, 5.9, "VPN  (virtual page #)",
            ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold")
    ax.text(4.6, 5.9, "Offset",
            ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold")
    ax.text(2.05, 5.25, "20 bits", ha="center", fontsize=9, color=C_GRAY)
    ax.text(4.6, 5.25, "12 bits", ha="center", fontsize=9, color=C_GRAY)

    # Page Table
    ax.text(6.4, 6.6, "Page Table  (in main memory)",
            fontsize=10.5, fontweight="bold", color=C_DARK)
    pt_x, pt_y = 6.4, 1.4
    pt_w, pt_h = 3.2, 4.8
    pt = FancyBboxPatch((pt_x, pt_y), pt_w, pt_h,
                        boxstyle="round,pad=0.04",
                        facecolor=C_BG, edgecolor=C_PURPLE, linewidth=1.6)
    ax.add_patch(pt)
    rows = [
        ("0x000", "0x4A1", True),
        ("0x001", "0x0C7", True),
        ("0x002", "  -- ", False),
        ("0x003", "0x9F2", True),
        ("0x004", "0x271", True),
        ("0x005", "0x88B", True),
    ]
    row_h = pt_h / (len(rows) + 1)
    ax.text(pt_x + pt_w / 2, pt_y + pt_h - row_h * 0.5,
            "VPN          PFN     V",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=C_DARK)
    highlight_idx = 3
    for i, (vp, pf, v) in enumerate(rows):
        y = pt_y + pt_h - row_h * (i + 1.5)
        if i == highlight_idx:
            hl = Rectangle((pt_x + 0.1, y - row_h / 2 + 0.04),
                           pt_w - 0.2, row_h - 0.08,
                           facecolor=C_AMBER, alpha=0.32,
                           edgecolor="none")
            ax.add_patch(hl)
        ax.text(pt_x + 0.55, y, vp, va="center", fontsize=10,
                color=C_DARK, family="monospace")
        ax.text(pt_x + 1.7, y, pf, va="center", fontsize=10,
                color=C_DARK, family="monospace")
        ax.text(pt_x + 2.75, y, "1" if v else "0",
                va="center", fontsize=10,
                color=C_GREEN if v else C_RED, family="monospace",
                fontweight="bold")

    # Arrow VPN -> page table (lookup)
    ax.add_patch(FancyArrowPatch((3.8, 5.9), (pt_x, 5.9 - 1.5),
                                 arrowstyle="->", mutation_scale=18,
                                 color=C_BLUE, linewidth=2,
                                 connectionstyle="arc3,rad=-0.18"))
    ax.text(5.0, 5.55, "lookup", fontsize=9.5,
            color=C_BLUE, style="italic")

    # Physical address output
    ax.text(10.4, 6.6, "Physical address (DRAM)",
            fontsize=10.5, fontweight="bold", color=C_DARK)
    pfn = Rectangle((10.4, 5.5), 2.4, 0.8,
                    facecolor=C_PURPLE, edgecolor="white", linewidth=2)
    off2 = Rectangle((12.8, 5.5), 0.9, 0.8,
                     facecolor=C_AMBER, edgecolor="white", linewidth=2)
    ax.add_patch(pfn)
    ax.add_patch(off2)
    ax.text(11.6, 5.9, "PFN  (physical frame #)",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    ax.text(13.25, 5.9, "Off",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")

    # Arrow PT -> physical address
    ax.add_patch(FancyArrowPatch((pt_x + pt_w, 3.4), (10.4, 5.6),
                                 arrowstyle="->", mutation_scale=18,
                                 color=C_PURPLE, linewidth=2,
                                 connectionstyle="arc3,rad=-0.20"))
    # Offset passthrough arrow
    ax.add_patch(FancyArrowPatch((5.4, 5.9), (12.8, 5.9),
                                 arrowstyle="->", mutation_scale=14,
                                 color=C_AMBER, linewidth=1.4,
                                 connectionstyle="arc3,rad=0.35"))
    ax.text(8.9, 6.95, "offset copied unchanged",
            fontsize=9, color=C_AMBER, style="italic", ha="center")

    # Notes at bottom
    note = ("Each process has its own page table. Page size is typically 4 KB. "
            "If the V (valid) bit is 0, the page is not in RAM -> page fault, "
            "OS loads it from SSD/HDD.")
    ax.text(7, 0.55, note, ha="center", fontsize=9.5,
            color=C_GRAY, style="italic", wrap=True)

    save(fig, "fig3_virtual_memory")


# ---------------------------------------------------------------------------
# Figure 4: TLB (Translation Lookaside Buffer)
# ---------------------------------------------------------------------------
def fig4_tlb() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    _strip(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)

    ax.text(7, 8.05, "TLB: caching virtual-to-physical translations",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)

    # CPU box
    cpu = FancyBboxPatch((0.3, 3.3), 1.7, 1.6,
                         boxstyle="round,pad=0.08",
                         facecolor=C_DARK, edgecolor="white", linewidth=2)
    ax.add_patch(cpu)
    ax.text(1.15, 4.4, "CPU", ha="center", color="white",
            fontsize=12, fontweight="bold")
    ax.text(1.15, 3.85, "issues VA",
            ha="center", color="white", fontsize=8.5)

    # TLB
    tlb = FancyBboxPatch((3.0, 3.0), 2.4, 2.2,
                         boxstyle="round,pad=0.08",
                         facecolor=C_BG, edgecolor=C_GREEN, linewidth=2)
    ax.add_patch(tlb)
    ax.text(4.2, 4.85, "TLB", ha="center", color=C_GREEN,
            fontsize=12, fontweight="bold")
    ax.text(4.2, 4.45, "(64-1024 entries)", ha="center",
            color=C_GRAY, fontsize=9)
    ax.text(4.2, 4.0, "VPN -> PFN", ha="center",
            color=C_DARK, fontsize=10, family="monospace")
    ax.text(4.2, 3.55, "lookup ~1 cycle",
            ha="center", color=C_GREEN, fontsize=9.5, fontweight="bold")

    # Arrow CPU -> TLB
    ax.add_patch(FancyArrowPatch((2.0, 4.1), (3.0, 4.1),
                                 arrowstyle="->", mutation_scale=18,
                                 color=C_DARK, linewidth=1.8))

    # Hit path (top)
    ax.add_patch(FancyArrowPatch((5.4, 4.6), (8.0, 6.4),
                                 arrowstyle="->", mutation_scale=18,
                                 color=C_GREEN, linewidth=2,
                                 connectionstyle="arc3,rad=-0.15"))
    ax.text(6.5, 5.85, "TLB HIT  (~99%)",
            color=C_GREEN, fontsize=10.5, fontweight="bold",
            ha="left")

    hit_box = FancyBboxPatch((8.0, 5.7), 5.4, 1.5,
                             boxstyle="round,pad=0.08",
                             facecolor=C_GREEN, alpha=0.18,
                             edgecolor=C_GREEN, linewidth=1.6)
    ax.add_patch(hit_box)
    ax.text(10.7, 6.75, "Direct DRAM access",
            ha="center", fontsize=11, fontweight="bold", color=C_DARK)
    ax.text(10.7, 6.25, "Total cost: ~1 cycle (TLB) + memory access",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(10.7, 5.9, "<- the fast path",
            ha="center", fontsize=9, style="italic", color=C_GREEN)

    # Miss path (bottom)
    ax.add_patch(FancyArrowPatch((5.4, 3.6), (8.0, 2.0),
                                 arrowstyle="->", mutation_scale=18,
                                 color=C_RED, linewidth=2,
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(6.5, 2.5, "TLB MISS  (~1%)",
            color=C_RED, fontsize=10.5, fontweight="bold", ha="left")

    miss_box = FancyBboxPatch((8.0, 0.7), 5.4, 1.8,
                              boxstyle="round,pad=0.08",
                              facecolor=C_RED, alpha=0.15,
                              edgecolor=C_RED, linewidth=1.6)
    ax.add_patch(miss_box)
    ax.text(10.7, 2.05, "Page-table walk in memory",
            ha="center", fontsize=11, fontweight="bold", color=C_DARK)
    ax.text(10.7, 1.55, "x86-64: 4-level walk = up to 4 mem reads",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(10.7, 1.15, "Total cost: 100-400 ns",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(10.7, 0.85, "Result then inserted into TLB",
            ha="center", fontsize=9, style="italic", color=C_RED)

    # Bottom takeaway
    ax.text(7, 0.15,
            "TLB hit-rate of 99% turns a 100+ ns lookup into ~1 ns -- it is the silent hero of virtual memory.",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    save(fig, "fig4_tlb")


# ---------------------------------------------------------------------------
# Figure 5: ECC memory (Hamming SEC-DED)
# ---------------------------------------------------------------------------
def fig5_ecc_correction() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.0))
    _strip(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    ax.text(7, 6.6, "ECC memory: detect and correct bit flips",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)

    # row 1: write side
    write_bits = list("10110100")  # 8 data bits
    parity_w = ["1", "0", "1", "1", "0"]  # 5 ECC bits (illustrative)

    def draw_byte(x0, y0, bits, color, label):
        ax.text(x0 - 0.25, y0 + 0.45, label, ha="right", va="center",
                fontsize=9.5, color=C_DARK, fontweight="bold")
        for i, b in enumerate(bits):
            r = Rectangle((x0 + i * 0.55, y0), 0.5, 0.85,
                          facecolor=color, edgecolor="white", linewidth=1.6)
            ax.add_patch(r)
            ax.text(x0 + i * 0.55 + 0.25, y0 + 0.42, b,
                    ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold",
                    family="monospace")

    # WRITE
    ax.text(0.3, 5.65, "1. WRITE",
            fontsize=11, fontweight="bold", color=C_DARK)
    draw_byte(2.0, 5.0, write_bits, C_BLUE, "data (8b)")
    draw_byte(8.4, 5.0, parity_w, C_PURPLE, "ECC (5b)")
    ax.text(11.5, 5.42, "stored in DRAM",
            fontsize=10, color=C_GRAY, style="italic")

    # cosmic ray flips bit 4 (index 3)
    ax.add_patch(FancyArrowPatch((6.5, 5.85), (3.65, 5.92),
                                 arrowstyle="->", mutation_scale=18,
                                 color=C_AMBER, linewidth=2,
                                 connectionstyle="arc3,rad=-0.35"))
    ax.text(5.1, 6.15, "cosmic ray / soft error -> flip bit",
            color=C_AMBER, fontsize=9.5, ha="center")

    # READ (corrupted)
    read_bits = write_bits.copy()
    read_bits[3] = "0" if read_bits[3] == "1" else "1"
    ax.text(0.3, 3.55, "2. READ",
            fontsize=11, fontweight="bold", color=C_DARK)

    # draw read bits with the flipped one in red
    ax.text(2.0 - 0.25, 3.32, "data (8b)", ha="right", va="center",
            fontsize=9.5, color=C_DARK, fontweight="bold")
    for i, b in enumerate(read_bits):
        col = C_RED if i == 3 else C_BLUE
        r = Rectangle((2.0 + i * 0.55, 2.9), 0.5, 0.85,
                      facecolor=col, edgecolor="white", linewidth=1.6)
        ax.add_patch(r)
        ax.text(2.0 + i * 0.55 + 0.25, 3.32, b,
                ha="center", va="center",
                color="white", fontsize=11, fontweight="bold",
                family="monospace")
    draw_byte(8.4, 2.9, parity_w, C_PURPLE, "ECC (5b)")

    # ECC decoder box
    dec = FancyBboxPatch((11.4, 2.6), 2.4, 1.5,
                         boxstyle="round,pad=0.08",
                         facecolor=C_GREEN, alpha=0.16,
                         edgecolor=C_GREEN, linewidth=1.6)
    ax.add_patch(dec)
    ax.text(12.6, 3.7, "ECC decoder", ha="center", fontsize=11,
            fontweight="bold", color=C_DARK)
    ax.text(12.6, 3.25, "syndrome != 0", ha="center", fontsize=9.5,
            color=C_DARK, family="monospace")
    ax.text(12.6, 2.9, "-> locate & flip back",
            ha="center", fontsize=9.5, color=C_GREEN, fontweight="bold")
    ax.add_patch(FancyArrowPatch((11.15, 3.32), (11.4, 3.32),
                                 arrowstyle="->", mutation_scale=14,
                                 color=C_DARK, linewidth=1.6))

    # CORRECTED row
    ax.text(0.3, 1.45, "3. CORRECTED",
            fontsize=11, fontweight="bold", color=C_DARK)
    draw_byte(2.0, 0.8, write_bits, C_GREEN, "data (8b)")
    ax.text(7.3, 1.22, "<-  delivered to CPU, bit flip transparently fixed",
            fontsize=10, color=C_GREEN, style="italic")
    ax.add_patch(FancyArrowPatch((4.15, 2.85), (4.15, 1.7),
                                 arrowstyle="->", mutation_scale=14,
                                 color=C_GREEN, linewidth=1.8))

    # Capability summary
    ax.text(7, 0.05,
            "Hamming SEC-DED ECC: corrects any single-bit error, detects any double-bit error per word.",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    save(fig, "fig5_ecc_correction")


# ---------------------------------------------------------------------------
# Figure 6: NUMA architecture
# ---------------------------------------------------------------------------
def fig6_numa() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    _strip(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    ax.text(7, 7.55, "NUMA: Non-Uniform Memory Access",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)

    # Two sockets
    def draw_socket(cx, label, mem_label, color):
        # CPU package
        cpu = FancyBboxPatch((cx - 1.6, 3.4), 3.2, 2.3,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="white",
                             linewidth=2, alpha=0.92)
        ax.add_patch(cpu)
        ax.text(cx, 5.4, label, ha="center", color="white",
                fontsize=12, fontweight="bold")
        # cores grid
        for i in range(4):
            for j in range(2):
                core = Rectangle((cx - 1.4 + i * 0.7, 3.85 + j * 0.5),
                                 0.55, 0.4,
                                 facecolor="white", edgecolor=color,
                                 linewidth=1.0, alpha=0.95)
                ax.add_patch(core)
        ax.text(cx, 3.55, "L3 + memory controller",
                ha="center", fontsize=8.5, color="white", style="italic")

        # local memory bank
        mem = FancyBboxPatch((cx - 1.4, 0.6), 2.8, 1.5,
                             boxstyle="round,pad=0.06",
                             facecolor=C_BG, edgecolor=color,
                             linewidth=1.8)
        ax.add_patch(mem)
        ax.text(cx, 1.55, mem_label, ha="center",
                fontsize=11, fontweight="bold", color=C_DARK)
        ax.text(cx, 1.05, "(local DRAM)",
                ha="center", fontsize=9, color=C_GRAY, style="italic")

        # local arrow
        ax.add_patch(FancyArrowPatch((cx, 3.4), (cx, 2.1),
                                     arrowstyle="<->", mutation_scale=14,
                                     color=color, linewidth=2.0))

    draw_socket(3.0,  "Socket 0  (CPU 0)", "Memory Node 0", C_BLUE)
    draw_socket(11.0, "Socket 1  (CPU 1)", "Memory Node 1", C_PURPLE)

    # Local-access latency labels
    ax.text(3.0, 2.7, "~80 ns",
            ha="center", fontsize=10, color=C_BLUE, fontweight="bold")
    ax.text(11.0, 2.7, "~80 ns",
            ha="center", fontsize=10, color=C_PURPLE, fontweight="bold")

    # Interconnect (UPI / Infinity Fabric)
    ax.add_patch(FancyArrowPatch((4.6, 4.55), (9.4, 4.55),
                                 arrowstyle="<->", mutation_scale=18,
                                 color=C_AMBER, linewidth=2.2))
    ax.text(7.0, 4.85, "UPI / Infinity Fabric",
            ha="center", fontsize=10, color=C_AMBER, fontweight="bold")

    # Remote access dashed line
    ax.add_patch(FancyArrowPatch((4.6, 4.0), (9.6, 1.5),
                                 arrowstyle="->", mutation_scale=14,
                                 color=C_RED, linewidth=1.6,
                                 linestyle="--",
                                 connectionstyle="arc3,rad=-0.18"))
    ax.text(7.0, 2.55, "remote access  ~140 ns  (~1.7x slower)",
            ha="center", fontsize=10, color=C_RED, fontweight="bold")

    # Bottom takeaway
    ax.text(7, 0.15,
            "Operating systems pin memory near the CPU that uses it; "
            "ignoring NUMA can hurt throughput by 30-60%.",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    save(fig, "fig6_numa")


# ---------------------------------------------------------------------------
# Figure 7: Memory channels (single / dual / quad)
# ---------------------------------------------------------------------------
def fig7_memory_channels() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    _strip(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    ax.text(7, 7.6, "Memory Channels: parallel data lanes between CPU and DRAM",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)

    configs = [
        ("Single channel",  1,  C_GRAY,   "25.6 GB/s",  1.4),
        ("Dual channel",    2,  C_BLUE,   "51.2 GB/s",  3.6),
        ("Quad channel",    4,  C_GREEN,  "102 GB/s",   6.4),
    ]

    for title, n, color, bw, cy in configs:
        # CPU box (left)
        cpu = FancyBboxPatch((0.4, cy - 0.6), 1.8, 1.2,
                             boxstyle="round,pad=0.08",
                             facecolor=C_DARK, edgecolor="white", linewidth=1.6)
        ax.add_patch(cpu)
        ax.text(1.3, cy, "CPU", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")

        # title above CPU on first row only? do per row instead
        ax.text(0.4, cy + 1.0, title, fontsize=11.5, fontweight="bold",
                color=color)

        # channels (lanes)
        lane_xs = np.linspace(2.4, 9.2, max(n, 1))
        # actually: draw n parallel lanes between CPU and DRAM
        dram_x = 9.6
        spacing = 0.32
        # vertical positions of lanes around cy
        if n == 1:
            ys = [cy]
        else:
            ys = np.linspace(cy + (n - 1) * spacing / 2,
                             cy - (n - 1) * spacing / 2, n)
        for y in ys:
            ax.add_patch(FancyArrowPatch((2.2, y), (dram_x, y),
                                         arrowstyle="->", mutation_scale=14,
                                         color=color, linewidth=2.4))

        # DRAM modules
        for i, y in enumerate(ys):
            mod = Rectangle((dram_x, y - 0.22), 2.2, 0.45,
                            facecolor=color, edgecolor="white",
                            linewidth=1.4, alpha=0.92)
            ax.add_patch(mod)
            ax.text(dram_x + 1.1, y, f"DIMM {i + 1}",
                    ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")

        # bandwidth label
        ax.text(12.2, cy, bw, fontsize=12, fontweight="bold",
                color=color, va="center")

    # axis-style annotation on the right
    ax.text(13.7, 7.2, "Peak\nbandwidth", ha="center", va="top",
            fontsize=9.5, color=C_DARK, fontweight="bold")

    # bottom note
    ax.text(7, 0.25,
            "Each channel adds an independent 64-bit data path to DRAM. "
            "Bandwidth scales nearly linearly; latency stays roughly the same.",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    save(fig, "fig7_memory_channels")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_memory_hierarchy()
    print("[ok] fig1_memory_hierarchy")
    fig2_dram_vs_sram()
    print("[ok] fig2_dram_vs_sram")
    fig3_virtual_memory()
    print("[ok] fig3_virtual_memory")
    fig4_tlb()
    print("[ok] fig4_tlb")
    fig5_ecc_correction()
    print("[ok] fig5_ecc_correction")
    fig6_numa()
    print("[ok] fig6_numa")
    fig7_memory_channels()
    print("[ok] fig7_memory_channels")
    print(f"\nWrote 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
