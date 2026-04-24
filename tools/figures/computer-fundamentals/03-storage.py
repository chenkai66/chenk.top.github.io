"""
Figure generator for Computer Fundamentals Part 03: Storage Systems.

Produces 7 production-quality figures covering HDD vs SSD physical
structure, storage performance metrics, RAID levels, file system layout,
NVMe vs SATA architecture, interface evolution, and storage tiering.

Outputs PNGs (dpi=150) into the EN and ZH asset directories:
  source/_posts/en/computer-fundamentals/03-storage/
  source/_posts/zh/computer-fundamentals/03-storage/

Style: matplotlib seaborn-v0_8-whitegrid; brand palette
  blue   #2563eb  (primary / SSD / fast)
  purple #7c3aed  (NVMe / accent)
  green  #10b981  (success / hot tier)
  amber  #f59e0b  (warning / HDD / cold)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Wedge

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
AMBER = COLORS["warning"]
DARK = COLORS["text"]
GRAY = COLORS["text2"]
LIGHT = "#f1f5f9"

DPI = 150

REPO = Path(__file__).resolve().parents[3]
OUT_DIRS = [
    REPO / "source/_posts/en/computer-fundamentals/03-storage",
    REPO / "source/_posts/zh/computer-fundamentals/03-storage",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": COLORS["border"],
    "axes.linewidth": 1.0,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.8,
})


def _save(fig: plt.Figure, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  wrote {name} -> {len(OUT_DIRS)} dirs")


# ---------------------------------------------------------------------------
# Figure 1: HDD vs SSD physical structure
# ---------------------------------------------------------------------------

def fig1_hdd_vs_ssd_structure() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.2))

    # ---------- HDD (left) ----------
    ax_l.set_xlim(0, 10); ax_l.set_ylim(0, 10)
    ax_l.set_aspect("equal"); ax_l.axis("off")
    ax_l.set_title("HDD — Mechanical Platters & Heads",
                   color=AMBER, fontsize=14, pad=12)

    # case
    ax_l.add_patch(FancyBboxPatch((0.4, 0.6), 9.2, 8.6,
                                  boxstyle="round,pad=0.05,rounding_size=0.25",
                                  facecolor="#fff7ed", edgecolor=AMBER, linewidth=2))

    # platter stack (concentric circles)
    cx, cy = 5.2, 5.2
    for r, alpha in [(3.4, 0.18), (3.0, 0.30), (2.6, 0.45)]:
        ax_l.add_patch(Circle((cx, cy), r, facecolor=AMBER, alpha=alpha,
                              edgecolor=AMBER, linewidth=1.0))
    # spindle
    ax_l.add_patch(Circle((cx, cy), 0.35, facecolor=DARK, edgecolor=DARK))
    # tracks
    for r in [1.4, 2.0, 2.6, 3.2]:
        ax_l.add_patch(Circle((cx, cy), r, facecolor="none",
                              edgecolor="#fed7aa", linewidth=0.6, linestyle=":"))

    # actuator arm + head
    ax_l.plot([8.7, 5.9], [7.8, 5.9], color=DARK, linewidth=3.5, solid_capstyle="round")
    ax_l.add_patch(Circle((8.7, 7.8), 0.35, facecolor=DARK, edgecolor=DARK))
    ax_l.add_patch(Circle((5.9, 5.9), 0.18, facecolor=COLORS["danger"], edgecolor=DARK))

    # rotation arrow
    arc = mpatches.FancyArrowPatch((cx + 1.7, cy - 1.7), (cx - 1.7, cy - 1.7),
                                   connectionstyle="arc3,rad=-0.4",
                                   arrowstyle="-|>", mutation_scale=18,
                                   color=BLUE, linewidth=2)
    ax_l.add_patch(arc)
    ax_l.text(cx, cy - 2.5, "5 400 / 7 200 / 10 000 RPM",
              ha="center", color=BLUE, fontsize=10, fontweight="bold")

    # labels
    ax_l.annotate("Read/write head", xy=(5.95, 5.95), xytext=(2.0, 2.0),
                  fontsize=10, color=DARK,
                  arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax_l.annotate("Magnetic platter\n(stores bits as N/S domains)",
                  xy=(2.6, 6.5), xytext=(0.7, 8.4),
                  fontsize=10, color=DARK,
                  arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax_l.annotate("Spindle motor", xy=(cx, cy), xytext=(7.6, 2.6),
                  fontsize=10, color=DARK,
                  arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax_l.annotate("Actuator arm", xy=(7.5, 6.9), xytext=(7.0, 8.6),
                  fontsize=10, color=DARK,
                  arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))

    ax_l.text(5.0, 0.05,
              "Random latency ~10 ms   •   100–200 MB/s seq.   •   moving parts",
              ha="center", fontsize=10, color=AMBER, fontweight="bold")

    # ---------- SSD (right) ----------
    ax_r.set_xlim(0, 10); ax_r.set_ylim(0, 10)
    ax_r.set_aspect("equal"); ax_r.axis("off")
    ax_r.set_title("SSD — NAND Flash Cells & Controller",
                   color=BLUE, fontsize=14, pad=12)

    ax_r.add_patch(FancyBboxPatch((0.4, 0.6), 9.2, 8.6,
                                  boxstyle="round,pad=0.05,rounding_size=0.25",
                                  facecolor="#eff6ff", edgecolor=BLUE, linewidth=2))

    # NAND grid
    grid_x0, grid_y0 = 1.0, 1.4
    cell_w, cell_h = 1.0, 0.95
    cols, rows = 6, 4
    for i in range(rows):
        for j in range(cols):
            x = grid_x0 + j * cell_w
            y = grid_y0 + i * cell_h
            ax_r.add_patch(Rectangle((x, y), cell_w * 0.88, cell_h * 0.82,
                                     facecolor=PURPLE, alpha=0.20 + 0.10 * ((i + j) % 3),
                                     edgecolor=PURPLE, linewidth=0.8))
    ax_r.text(grid_x0 + cols * cell_w / 2 - 0.2, grid_y0 + rows * cell_h + 0.15,
              "NAND flash array (pages → blocks → planes → die)",
              ha="center", fontsize=9, color=PURPLE, fontweight="bold")

    # Controller chip
    ax_r.add_patch(FancyBboxPatch((1.1, 6.5), 4.0, 1.7,
                                  boxstyle="round,pad=0.04,rounding_size=0.12",
                                  facecolor=BLUE, edgecolor=DARK, linewidth=1.2))
    ax_r.text(3.1, 7.35, "Controller IC",
              ha="center", color="white", fontsize=11, fontweight="bold")
    ax_r.text(3.1, 6.85, "FTL · wear-levelling · ECC · GC",
              ha="center", color="white", fontsize=8.5)

    # DRAM cache
    ax_r.add_patch(FancyBboxPatch((5.5, 6.5), 3.4, 1.7,
                                  boxstyle="round,pad=0.04,rounding_size=0.12",
                                  facecolor=GREEN, edgecolor=DARK, linewidth=1.2))
    ax_r.text(7.2, 7.35, "DRAM cache",
              ha="center", color="white", fontsize=11, fontweight="bold")
    ax_r.text(7.2, 6.85, "mapping table + write buffer",
              ha="center", color="white", fontsize=8.5)

    # Host interface
    ax_r.add_patch(FancyBboxPatch((1.1, 8.4), 7.8, 0.65,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor=DARK, edgecolor=DARK))
    ax_r.text(5.0, 8.72, "Host interface — SATA / PCIe (NVMe)",
              ha="center", color="white", fontsize=10, fontweight="bold")

    ax_r.text(5.0, 0.05,
              "Random latency ~50–100 µs   •   500 MB/s – 12 GB/s   •   no moving parts",
              ha="center", fontsize=10, color=BLUE, fontweight="bold")

    fig.suptitle("HDD vs SSD — Physical Anatomy",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig1_hdd_vs_ssd.png")


# ---------------------------------------------------------------------------
# Figure 2: Storage performance — throughput, IOPS, latency
# ---------------------------------------------------------------------------

def fig2_performance() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    devices = ["HDD\n7200 RPM", "SATA\nSSD", "NVMe\nGen 3", "NVMe\nGen 4", "NVMe\nGen 5"]
    colors = [AMBER, BLUE, PURPLE, PURPLE, GREEN]

    # Throughput MB/s (sequential read)
    seq = [180, 550, 3500, 7000, 12000]
    ax = axes[0]
    bars = ax.bar(devices, seq, color=colors, edgecolor=DARK, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Sequential read (MB/s, log)")
    ax.set_title("Throughput", color=DARK)
    ax.set_ylim(50, 30000)
    for b, v in zip(bars, seq):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15,
                f"{v:,}", ha="center", fontsize=9, fontweight="bold", color=DARK)

    # Random 4K IOPS
    iops = [120, 95_000, 600_000, 1_000_000, 1_500_000]
    ax = axes[1]
    bars = ax.bar(devices, iops, color=colors, edgecolor=DARK, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("4K random read (IOPS, log)")
    ax.set_title("IOPS — random access", color=DARK)
    ax.set_ylim(50, 5_000_000)
    for b, v in zip(bars, iops):
        label = f"{v/1000:.0f}K" if v >= 1000 else f"{v}"
        if v >= 1_000_000:
            label = f"{v/1_000_000:.1f}M"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.4,
                label, ha="center", fontsize=9, fontweight="bold", color=DARK)

    # Latency (µs)
    lat = [10_000, 100, 50, 30, 20]
    ax = axes[2]
    bars = ax.bar(devices, lat, color=colors, edgecolor=DARK, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Random read latency (µs, log) — lower better")
    ax.set_title("Latency", color=DARK)
    ax.set_ylim(5, 50_000)
    for b, v in zip(bars, lat):
        label = f"{v/1000:.0f} ms" if v >= 1000 else f"{v} µs"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.4,
                label, ha="center", fontsize=9, fontweight="bold", color=DARK)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(True, axis="y", which="both", alpha=0.4)

    fig.suptitle("Storage Performance — Throughput, IOPS, Latency (log scale)",
                 fontsize=14, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_performance.png")


# ---------------------------------------------------------------------------
# Figure 3: RAID levels visualization (0/1/5/6/10)
# ---------------------------------------------------------------------------

def fig3_raid_levels() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    axes = axes.flatten()

    def draw_disk(ax, x, y, blocks, label, w=1.0, h=0.55):
        """Draw a disk with stacked colored blocks."""
        n = len(blocks)
        for i, (txt, color) in enumerate(blocks):
            ax.add_patch(Rectangle((x, y + i * h), w, h * 0.92,
                                   facecolor=color, edgecolor=DARK, linewidth=0.8))
            ax.text(x + w / 2, y + i * h + h * 0.46, txt,
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
        ax.text(x + w / 2, y - 0.25, label, ha="center",
                fontsize=9, fontweight="bold", color=DARK)

    def setup(ax, title, subtitle):
        ax.set_xlim(0, 10); ax.set_ylim(-0.6, 4.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=12, color=DARK, pad=4)
        ax.text(5, 4.0, subtitle, ha="center", fontsize=9, color=GRAY, style="italic")

    # RAID 0 — striping (2 disks)
    ax = axes[0]; setup(ax, "RAID 0 — Striping", "100% capacity · 0 fault tolerance")
    draw_disk(ax, 2.5, 0.5, [("A1", BLUE), ("A3", BLUE), ("A5", BLUE)], "Disk 1")
    draw_disk(ax, 5.5, 0.5, [("A2", PURPLE), ("A4", PURPLE), ("A6", PURPLE)], "Disk 2")

    # RAID 1 — mirroring (2 disks)
    ax = axes[1]; setup(ax, "RAID 1 — Mirroring", "50% capacity · 1-disk tolerance")
    draw_disk(ax, 2.5, 0.5, [("A1", BLUE), ("A2", BLUE), ("A3", BLUE)], "Disk 1")
    draw_disk(ax, 5.5, 0.5, [("A1", GREEN), ("A2", GREEN), ("A3", GREEN)], "Disk 2 (mirror)")

    # RAID 5 — distributed parity (3 disks)
    ax = axes[2]; setup(ax, "RAID 5 — Distributed parity", "(n−1)/n capacity · 1-disk tolerance")
    draw_disk(ax, 1.5, 0.5, [("A1", BLUE), ("B1", BLUE), ("Cp", AMBER)], "Disk 1")
    draw_disk(ax, 4.5, 0.5, [("A2", BLUE), ("Bp", AMBER), ("C1", BLUE)], "Disk 2")
    draw_disk(ax, 7.5, 0.5, [("Ap", AMBER), ("B2", BLUE), ("C2", BLUE)], "Disk 3")

    # RAID 6 — double parity (4 disks)
    ax = axes[3]; setup(ax, "RAID 6 — Dual parity", "(n−2)/n capacity · 2-disk tolerance")
    draw_disk(ax, 0.5, 0.5, [("A1", BLUE), ("Bp", AMBER), ("Cq", PURPLE)], "Disk 1", w=0.9)
    draw_disk(ax, 3.0, 0.5, [("A2", BLUE), ("B1", BLUE), ("Cp", AMBER)], "Disk 2", w=0.9)
    draw_disk(ax, 5.5, 0.5, [("Ap", AMBER), ("B2", BLUE), ("C1", BLUE)], "Disk 3", w=0.9)
    draw_disk(ax, 8.0, 0.5, [("Aq", PURPLE), ("Bq", PURPLE), ("C2", BLUE)], "Disk 4", w=0.9)

    # RAID 10 — mirror+stripe (4 disks)
    ax = axes[4]; setup(ax, "RAID 10 — Mirror + Stripe", "50% capacity · best perf + protection")
    draw_disk(ax, 0.5, 0.5, [("A1", BLUE), ("A3", BLUE), ("A5", BLUE)], "M1-D1", w=0.9)
    draw_disk(ax, 3.0, 0.5, [("A1", GREEN), ("A3", GREEN), ("A5", GREEN)], "M1-D2", w=0.9)
    draw_disk(ax, 5.5, 0.5, [("A2", BLUE), ("A4", BLUE), ("A6", BLUE)], "M2-D1", w=0.9)
    draw_disk(ax, 8.0, 0.5, [("A2", GREEN), ("A4", GREEN), ("A6", GREEN)], "M2-D2", w=0.9)

    # Legend / summary
    ax = axes[5]; ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Choose by workload", fontsize=12, color=DARK, pad=4)
    rows = [
        ("RAID 0",  "Scratch, video edit cache",       BLUE),
        ("RAID 1",  "Boot drive, small NAS",           GREEN),
        ("RAID 5",  "File server, balanced",           PURPLE),
        ("RAID 6",  "Large arrays (≥6 disks)",         AMBER),
        ("RAID 10", "DB, VM, mission critical",        DARK),
    ]
    for i, (lvl, use, col) in enumerate(rows):
        y = 4.2 - i * 0.78
        ax.add_patch(Rectangle((0.4, y - 0.18), 0.5, 0.45,
                               facecolor=col, edgecolor=DARK, linewidth=0.8))
        ax.text(1.1, y + 0.04, lvl, fontsize=10, fontweight="bold", color=DARK)
        ax.text(3.1, y + 0.04, use, fontsize=9.5, color=DARK)

    # Color legend (data vs parity)
    legend_handles = [
        mpatches.Patch(color=BLUE,   label="Data block (primary)"),
        mpatches.Patch(color=GREEN,  label="Mirror copy"),
        mpatches.Patch(color=AMBER,  label="Parity P (XOR)"),
        mpatches.Patch(color=PURPLE, label="Parity Q (Reed–Solomon)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("RAID Levels — Striping, Mirroring & Parity",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, "fig3_raid_levels.png")


# ---------------------------------------------------------------------------
# Figure 4: File system layout — inodes, blocks, journaling
# ---------------------------------------------------------------------------

def fig4_filesystem_layout() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

    # Disk regions ribbon
    regions = [
        ("Boot\nblock",        0.4, 1.2, GRAY),
        ("Super-\nblock",      1.7, 1.2, DARK),
        ("Block group\ndescriptors", 3.1, 1.7, COLORS["text2"]),
        ("Block\nbitmap",      5.0, 1.4, AMBER),
        ("Inode\nbitmap",      6.6, 1.4, AMBER),
        ("Inode table",        8.2, 2.4, PURPLE),
        ("Journal\n(write-ahead log)", 10.8, 2.6, GREEN),
    ]
    y = 6.0; h = 1.1
    for label, x, w, color in regions:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=color, edgecolor=DARK, linewidth=1.0,
                                    alpha=0.9))
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    ax.text(0.4, y + h + 0.55, "On-disk layout (ext4-style)",
            fontsize=11, fontweight="bold", color=DARK)
    ax.annotate("", xy=(13.4, y + h + 0.25), xytext=(0.2, y + h + 0.25),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))
    ax.text(13.5, y + h + 0.25, "LBA →", fontsize=9, color=GRAY, va="center")

    # Inode → data blocks
    ax.add_patch(FancyBboxPatch((0.4, 2.0), 4.6, 3.3,
                                boxstyle="round,pad=0.04,rounding_size=0.1",
                                facecolor="white", edgecolor=PURPLE, linewidth=1.5))
    ax.text(2.7, 5.0, "Inode (256 B)", ha="center", color=PURPLE,
            fontsize=11, fontweight="bold")
    fields = ["mode + uid/gid", "size, atime/mtime/ctime",
              "link count", "12 direct block pointers",
              "1 indirect", "1 double-indirect", "1 triple-indirect"]
    for i, f in enumerate(fields):
        ax.text(0.7, 4.55 - i * 0.35, f"• {f}",
                fontsize=9, color=DARK)

    # Direct blocks
    block_x0 = 6.0; block_y = 3.2
    for i in range(6):
        ax.add_patch(Rectangle((block_x0 + i * 0.85, block_y), 0.75, 0.75,
                               facecolor=BLUE, edgecolor=DARK, linewidth=0.8))
        ax.text(block_x0 + i * 0.85 + 0.375, block_y + 0.375,
                f"D{i}", ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
    ax.text(block_x0 + 6 * 0.85 / 2, block_y + 1.0,
            "Direct data blocks (4 KiB each)",
            ha="center", fontsize=9, color=BLUE, fontweight="bold")

    # Indirect block
    ax.add_patch(Rectangle((11.6, block_y), 1.0, 0.75,
                           facecolor=PURPLE, edgecolor=DARK, linewidth=0.8))
    ax.text(12.1, block_y + 0.375, "IND",
            ha="center", va="center", color="white",
            fontsize=8.5, fontweight="bold")
    ax.text(12.1, block_y - 0.3, "→ 1024 ptrs\n→ 4 MiB",
            ha="center", fontsize=8, color=PURPLE)

    # Arrows inode -> blocks (originate from inode box right edge)
    ax.annotate("", xy=(block_x0, block_y + 0.4), xytext=(5.0, 3.6),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))
    # Arrow first block -> indirect chain illustration
    ax.annotate("", xy=(11.6, block_y + 0.4),
                xytext=(block_x0 + 6 * 0.85, block_y + 0.4),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0,
                                linestyle="--"))

    # Journaling timeline
    j_y = 0.6
    ax.add_patch(FancyBboxPatch((0.4, j_y), 13.0, 1.1,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor="#ecfdf5", edgecolor=GREEN, linewidth=1.5))
    ax.text(0.7, j_y + 0.85, "Journaling (write-ahead) — crash safety",
            color=GREEN, fontsize=10.5, fontweight="bold")
    steps = [
        ("1. Begin TX",      1.5, GREEN),
        ("2. Write metadata\nto journal", 4.0, GREEN),
        ("3. Commit record", 7.0, GREEN),
        ("4. Checkpoint to\nfinal location", 9.6, BLUE),
        ("5. Free journal\nspace", 12.4, GRAY),
    ]
    for i, (s, x, c) in enumerate(steps):
        ax.add_patch(Circle((x, j_y + 0.35), 0.16,
                            facecolor=c, edgecolor=DARK, linewidth=0.8))
        ax.text(x, j_y + 0.35, str(i + 1), ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(x, j_y - 0.15, s.split(". ", 1)[1],
                ha="center", fontsize=8.3, color=DARK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i + 1][1] - 0.18, j_y + 0.35),
                        xytext=(x + 0.18, j_y + 0.35),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    fig.suptitle("File-System Layout — Inodes, Blocks & Journaling (ext4)",
                 fontsize=15, fontweight="bold", color=DARK, y=0.98)
    fig.tight_layout()
    _save(fig, "fig4_filesystem.png")


# ---------------------------------------------------------------------------
# Figure 5: NVMe vs SATA SSD architecture
# ---------------------------------------------------------------------------

def fig5_nvme_vs_sata() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.4))

    def stack(ax, title, color, queues, queue_depth, bus, max_bw):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.set_title(title, color=color, fontsize=14, pad=10)

        layers = [
            ("Application",         9.0, COLORS["border"]),
            ("VFS / file system",   8.0, COLORS["muted"]),
            ("Block layer",         7.0, COLORS["text2"]),
        ]
        for name, y, col in layers:
            ax.add_patch(FancyBboxPatch((1.0, y - 0.4), 8.0, 0.7,
                                        boxstyle="round,pad=0.02,rounding_size=0.05",
                                        facecolor=col, edgecolor=DARK, linewidth=0.8))
            ax.text(5.0, y - 0.05, name, ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")

        # Driver / protocol
        ax.add_patch(FancyBboxPatch((1.0, 5.6), 8.0, 0.8,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=color, edgecolor=DARK, linewidth=1.2))
        ax.text(5.0, 6.0, f"Protocol driver — {title.split(' — ')[0]}",
                ha="center", color="white", fontsize=10.5, fontweight="bold")

        # Command queues
        ax.text(5.0, 4.95, f"Command queues: {queues}  ×  depth {queue_depth}",
                ha="center", fontsize=10, color=DARK, fontweight="bold")
        # draw queues
        n_show = min(queues, 8)
        qw = 8.0 / n_show
        for i in range(n_show):
            x = 1.0 + i * qw
            ax.add_patch(Rectangle((x + 0.05, 4.2), qw - 0.1, 0.55,
                                   facecolor=color, alpha=0.35,
                                   edgecolor=DARK, linewidth=0.6))
        if queues > n_show:
            ax.text(5.0, 4.0, f"… up to {queues:,} queues", ha="center",
                    fontsize=8, color=GRAY, style="italic")

        # Bus
        ax.add_patch(FancyBboxPatch((1.0, 2.7), 8.0, 0.8,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=DARK, edgecolor=DARK))
        ax.text(5.0, 3.1, f"Physical bus: {bus}",
                ha="center", color="white", fontsize=10.5, fontweight="bold")

        # SSD
        ax.add_patch(FancyBboxPatch((1.0, 0.8), 8.0, 1.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="white", edgecolor=color, linewidth=1.5))
        ax.text(5.0, 1.85, "SSD device", ha="center", color=color,
                fontsize=10.5, fontweight="bold")
        ax.text(5.0, 1.35, f"Practical bandwidth ≈ {max_bw}",
                ha="center", color=DARK, fontsize=10)

        # arrow connectors
        for y1, y2 in [(8.6, 8.4), (7.6, 7.4), (6.6, 6.4), (5.6, 5.5),
                       (4.2, 3.5), (2.7, 2.3)]:
            ax.annotate("", xy=(5.0, y2), xytext=(5.0, y1),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    stack(ax_l, "SATA — AHCI",      AMBER,
          queues=1, queue_depth=32,
          bus="SATA III · 6 Gb/s · half-duplex",
          max_bw="550 MB/s")

    stack(ax_r, "NVMe — PCIe",      PURPLE,
          queues=65535, queue_depth=65536,
          bus="PCIe Gen4 ×4 · 64 Gb/s · full-duplex",
          max_bw="7 000 MB/s (Gen4)  •  12 000 MB/s (Gen5)")

    fig.suptitle("NVMe vs SATA SSD — Why NVMe Wins on Parallelism",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig5_nvme_vs_sata.png")


# ---------------------------------------------------------------------------
# Figure 6: Storage interface evolution timeline (IDE → SATA → NVMe)
# ---------------------------------------------------------------------------

def fig6_interface_evolution() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.4))

    interfaces = [
        ("PATA / IDE",      1986, 133,    AMBER),
        ("SATA I",          2003, 150,    AMBER),
        ("SATA II",         2004, 300,    AMBER),
        ("SATA III",        2009, 600,    AMBER),
        ("NVMe PCIe Gen3 ×4", 2014, 4000, BLUE),
        ("NVMe PCIe Gen4 ×4", 2019, 8000, PURPLE),
        ("NVMe PCIe Gen5 ×4", 2022, 16000, GREEN),
    ]
    years = [d[1] for d in interfaces]
    bws   = [d[2] for d in interfaces]
    names = [d[0] for d in interfaces]
    colors = [d[3] for d in interfaces]

    ax.set_yscale("log")
    ax.plot(years, bws, color=DARK, linewidth=1.4, alpha=0.5, zorder=1)
    ax.scatter(years, bws, c=colors, s=240, edgecolor=DARK,
               linewidth=1.2, zorder=3)

    for x, y, name in zip(years, bws, names):
        offset_y = y * 1.7 if name not in ("SATA II",) else y * 0.45
        ax.annotate(f"{name}\n{y:,} MB/s",
                    xy=(x, y), xytext=(x, offset_y),
                    ha="center", fontsize=9, fontweight="bold", color=DARK,
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))

    ax.set_xlim(1983, 2027)
    ax.set_ylim(80, 90_000)
    ax.set_xlabel("Year introduced")
    ax.set_ylabel("Theoretical bandwidth (MB/s, log)")
    ax.set_title("Storage Interface Evolution — 120× bandwidth in 36 years",
                 fontsize=14, color=DARK, pad=10)

    # Era shading
    ax.axvspan(1983, 2009, color=AMBER, alpha=0.06)
    ax.axvspan(2009, 2014, color=BLUE,  alpha=0.05)
    ax.axvspan(2014, 2027, color=PURPLE, alpha=0.05)
    ax.text(1996, 55_000, "Parallel ATA era",  ha="center", color=AMBER,  fontsize=10, fontweight="bold")
    ax.text(2011.5, 55_000, "SATA era",        ha="center", color=BLUE,   fontsize=10, fontweight="bold")
    ax.text(2018, 55_000, "PCIe / NVMe era",   ha="center", color=PURPLE, fontsize=10, fontweight="bold")

    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig6_interface_evolution.png")


# ---------------------------------------------------------------------------
# Figure 7: Storage tiering — hot / warm / cold
# ---------------------------------------------------------------------------

def fig7_storage_tiering() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.4),
                                     gridspec_kw={"width_ratios": [1.2, 1]})

    # ----- Pyramid -----
    ax_l.set_xlim(0, 10); ax_l.set_ylim(0, 10); ax_l.axis("off")
    ax_l.set_title("Storage tiering pyramid", color=DARK, fontsize=13, pad=8)

    tiers = [
        # (label, top_y, bot_y, top_w, bot_w, color, perf, cost, share)
        ("HOT  — DRAM / Optane / NVMe SSD",   9.0, 7.4, 2.4, 4.0, GREEN,
         "<100 µs",     "cost ••••",  "5 %"),
        ("WARM — SATA SSD / 15K SAS",         7.4, 5.6, 4.0, 5.6, BLUE,
         "100 µs–1 ms", "cost •••",   "15 %"),
        ("COOL — 7.2K HDD (RAID6)",           5.6, 3.5, 5.6, 7.5, PURPLE,
         "5–10 ms",     "cost ••",    "30 %"),
        ("COLD — Object storage (S3-IA, Glacier)", 3.5, 1.0, 7.5, 9.4, AMBER,
         "seconds–hours", "cost •",   "50 %"),
    ]
    cx = 5.0
    for label, ty, by, tw, bw, color, perf, cost, share in tiers:
        poly = plt.Polygon(
            [(cx - tw / 2, ty), (cx + tw / 2, ty),
             (cx + bw / 2, by), (cx - bw / 2, by)],
            facecolor=color, edgecolor=DARK, linewidth=1.0, alpha=0.85)
        ax_l.add_patch(poly)
        ax_l.text(cx, (ty + by) / 2 + 0.15, label, ha="center", va="center",
                  color="white", fontsize=10, fontweight="bold")
        ax_l.text(cx, (ty + by) / 2 - 0.32,
                  f"latency {perf}   ·   {cost}   ·   ~{share} of data",
                  ha="center", va="center", color="white", fontsize=8.5)

    # axes annotations
    ax_l.annotate("", xy=(0.6, 9.0), xytext=(0.6, 1.0),
                  arrowprops=dict(arrowstyle="<-", color=DARK, lw=1.2))
    ax_l.text(0.3, 5.0, "faster\n+ pricier", rotation=90, ha="center",
              va="center", color=DARK, fontsize=9, fontweight="bold")
    ax_l.annotate("", xy=(9.7, 1.0), xytext=(9.7, 9.0),
                  arrowprops=dict(arrowstyle="<-", color=DARK, lw=1.2))
    ax_l.text(9.95, 5.0, "larger\n+ cheaper", rotation=270, ha="center",
              va="center", color=DARK, fontsize=9, fontweight="bold")

    # ----- Cost / latency scatter -----
    ax_r.set_title("Latency vs cost (USD per GB-month, log–log)",
                   color=DARK, fontsize=13, pad=8)
    points = [
        ("DRAM",         5e-2, 5.0,    GRAY),
        ("Optane",       1.0,  3.0,    DARK),
        ("NVMe SSD",     1e2,  0.10,   GREEN),
        ("SATA SSD",     1e3,  0.06,   BLUE),
        ("15K SAS HDD",  3e3,  0.04,   BLUE),
        ("7.2K HDD",     1e4,  0.015,  PURPLE),
        ("S3 Standard",  5e4,  0.023,  AMBER),
        ("S3 Glacier",   3e9,  0.004,  AMBER),
    ]
    for name, lat, cost, color in points:
        ax_r.scatter(lat, cost, s=210, color=color, edgecolor=DARK,
                     linewidth=1.0, zorder=3)
        ax_r.annotate(name, (lat, cost), xytext=(8, 6),
                      textcoords="offset points",
                      fontsize=9, fontweight="bold", color=DARK)

    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlabel("Read latency (ns, log)")
    ax_r.set_ylabel("Cost (USD per GB-month, log)")
    ax_r.set_xlim(1e-2, 1e10)
    ax_r.set_ylim(1e-3, 20)
    ax_r.grid(True, which="both", alpha=0.4)

    # tier bands (vertical)
    ax_r.axvspan(1e-2, 1e2,  color=GREEN,  alpha=0.05)
    ax_r.axvspan(1e2,  1e4,  color=BLUE,   alpha=0.05)
    ax_r.axvspan(1e4,  1e6,  color=PURPLE, alpha=0.05)
    ax_r.axvspan(1e6,  1e10, color=AMBER,  alpha=0.05)

    fig.suptitle("Storage Tiering — Hot · Warm · Cool · Cold",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig7_tiering.png")


# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating storage figures to:")
    for d in OUT_DIRS:
        print(f"  - {d}")
    fig1_hdd_vs_ssd_structure()
    fig2_performance()
    fig3_raid_levels()
    fig4_filesystem_layout()
    fig5_nvme_vs_sata()
    fig6_interface_evolution()
    fig7_storage_tiering()
    print("Done.")


if __name__ == "__main__":
    main()
