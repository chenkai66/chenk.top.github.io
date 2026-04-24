"""
Figure generation script for Linux Article 03: Disk Management.

Generates 5 conceptual figures used in both EN and ZH versions of the
article. Each figure is rendered to BOTH article asset folders so the
markdown image references stay in sync across languages.

Figures:
    fig1_filesystem_hierarchy   The standard Linux filesystem hierarchy
                                showing the role of /, /usr, /var, /etc,
                                /home and what typically lives under each.
    fig2_partition_layout       Side-by-side layout of an MBR-partitioned
                                disk vs. a GPT-partitioned disk, with the
                                primary/secondary GPT header redundancy
                                highlighted.
    fig3_lvm_stack              The PV -> VG -> LV stack used by Linux
                                LVM, drawn as three layers showing how
                                physical disks are pooled and then carved
                                back out into resizable logical volumes.
    fig4_mount_table            How a block device + filesystem driver +
                                mount point combine into an entry in the
                                kernel mount table, with a small example
                                of /etc/fstab on the side.
    fig5_filesystem_comparison  Bar/grid comparison of ext4, xfs and
                                btrfs across the dimensions that matter
                                operationally (max file size, shrink
                                support, snapshots, checksumming, etc.).

Usage:
    python3 scripts/figures/linux/03-disk-management.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401  (registers the seaborn style we use)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

COLOR_BLUE = "#2563eb"
COLOR_PURPLE = "#7c3aed"
COLOR_GREEN = "#10b981"
COLOR_AMBER = "#f59e0b"
COLOR_GREY = "#475569"
COLOR_LIGHT = "#e2e8f0"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linux" / "disk-management"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "disk-management"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (EN_DIR, ZH_DIR):
        fig.savefig(out_dir / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _rounded_box(ax, x, y, w, h, *, facecolor, edgecolor=None, alpha=1.0, lw=1.2):
    """Helper to draw a rounded rectangle that we can label."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor or facecolor,
        alpha=alpha,
    )
    ax.add_patch(box)
    return box


# ---------------------------------------------------------------------------
# Figure 1: Linux filesystem hierarchy
# ---------------------------------------------------------------------------

def fig1_filesystem_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Root node at the top
    _rounded_box(ax, 5.2, 6.6, 1.6, 0.9, facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE)
    ax.text(6.0, 7.05, "/", color="white", ha="center", va="center",
            fontsize=18, fontweight="bold")
    ax.text(6.0, 6.35, "root of the entire filesystem tree",
            ha="center", va="center", fontsize=9, color=COLOR_GREY, style="italic")

    children = [
        ("/bin", "essential user\nbinaries (ls, cp)", COLOR_PURPLE),
        ("/etc", "system-wide\nconfiguration", COLOR_AMBER),
        ("/home", "per-user home\ndirectories", COLOR_GREEN),
        ("/usr", "installed software\n(/usr/bin, /usr/lib)", COLOR_PURPLE),
        ("/var", "variable data:\nlogs, spools, caches", COLOR_AMBER),
        ("/tmp", "temporary files\n(often tmpfs)", COLOR_GREY),
    ]

    n = len(children)
    box_w, box_h = 1.55, 1.15
    spacing = (12 - n * box_w) / (n + 1)
    y_box = 2.4

    for i, (name, desc, color) in enumerate(children):
        x = spacing + i * (box_w + spacing)
        _rounded_box(ax, x, y_box, box_w, box_h, facecolor="white",
                     edgecolor=color, lw=1.6)
        ax.text(x + box_w / 2, y_box + box_h - 0.32, name,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=color)
        ax.text(x + box_w / 2, y_box + 0.35, desc,
                ha="center", va="center", fontsize=8, color=COLOR_GREY)

        # Connector from root to child
        ax.plot([6.0, x + box_w / 2], [6.6, y_box + box_h],
                color=COLOR_LIGHT, lw=1.4, zorder=0)

    # Legend / annotation strip at the bottom
    ax.text(6.0, 1.4,
            "Each top-level directory has a stable purpose (FHS).  "
            "Many of them can live on separate block devices and be "
            "mounted independently.",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY)

    ax.text(6.0, 0.55,
            "Common split: /  +  /var (logs)  +  /home (users)  +  /data (app)",
            ha="center", va="center", fontsize=10, color=COLOR_BLUE,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff",
                      edgecolor=COLOR_BLUE, lw=1.0))

    ax.set_title("Linux Filesystem Hierarchy (FHS, simplified)",
                 fontsize=14, fontweight="bold", pad=12, color="#1e293b")

    _save(fig, "fig1_filesystem_hierarchy.png")


# ---------------------------------------------------------------------------
# Figure 2: MBR vs GPT partition layout
# ---------------------------------------------------------------------------

def fig2_partition_layout() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.subplots_adjust(wspace=0.28)

    # ---------- MBR ----------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("MBR  (Master Boot Record)",
                 fontsize=13, fontweight="bold", color=COLOR_AMBER)

    # The disk strip
    disk_y = 2.6
    disk_h = 1.3
    ax.add_patch(Rectangle((0.4, disk_y), 9.2, disk_h, facecolor="white",
                           edgecolor=COLOR_GREY, lw=1.4))

    # MBR sector
    ax.add_patch(Rectangle((0.4, disk_y), 0.6, disk_h, facecolor=COLOR_AMBER,
                           edgecolor=COLOR_AMBER, alpha=0.85))
    ax.text(0.7, disk_y + disk_h / 2, "MBR\n512B",
            ha="center", va="center", fontsize=8, color="white",
            fontweight="bold")

    # 4 primary partitions
    parts = [
        ("Primary 1", COLOR_BLUE),
        ("Primary 2", COLOR_PURPLE),
        ("Primary 3", COLOR_GREEN),
        ("Extended", COLOR_GREY),
    ]
    p_x = 1.1
    p_w = 2.05
    for label, color in parts:
        ax.add_patch(Rectangle((p_x, disk_y), p_w, disk_h, facecolor=color,
                               edgecolor="white", alpha=0.85, lw=1.5))
        ax.text(p_x + p_w / 2, disk_y + disk_h / 2, label,
                ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
        p_x += p_w

    # Logical partitions inside Extended (drawn just below)
    ext_x0 = 1.1 + 3 * p_w
    for i in range(3):
        lx = ext_x0 + 0.1 + i * 0.6
        ax.add_patch(Rectangle((lx, disk_y - 1.05), 0.5, 0.7,
                               facecolor=COLOR_GREY, alpha=0.45, edgecolor="white"))
        ax.text(lx + 0.25, disk_y - 0.7, f"L{i+1}",
                ha="center", va="center", color="white", fontsize=7)
    ax.annotate("logical\npartitions", xy=(ext_x0 + 0.9, disk_y - 0.7),
                xytext=(ext_x0 + 1.9, disk_y - 1.4),
                fontsize=8, color=COLOR_GREY,
                arrowprops=dict(arrowstyle="->", color=COLOR_GREY, lw=0.8))

    # Constraints box
    ax.text(5.0, 5.4,
            "32-bit LBA  -  max disk size 2 TiB",
            ha="center", fontsize=10, color="#7f1d1d", fontweight="bold")
    ax.text(5.0, 4.85,
            "<= 4 primary partitions (or 3 primary + 1 extended)",
            ha="center", fontsize=9.5, color=COLOR_GREY)
    ax.text(5.0, 4.45,
            "single copy of the partition table - corruption is unrecoverable",
            ha="center", fontsize=9.5, color=COLOR_GREY)

    # ---------- GPT ----------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("GPT  (GUID Partition Table)",
                 fontsize=13, fontweight="bold", color=COLOR_BLUE)

    ax.add_patch(Rectangle((0.4, disk_y), 9.2, disk_h, facecolor="white",
                           edgecolor=COLOR_GREY, lw=1.4))

    # Protective MBR + primary GPT header
    ax.add_patch(Rectangle((0.4, disk_y), 0.55, disk_h, facecolor=COLOR_GREY,
                           edgecolor=COLOR_GREY, alpha=0.7))
    ax.text(0.675, disk_y + disk_h / 2, "PMBR",
            ha="center", va="center", color="white", fontsize=7.5,
            fontweight="bold")

    ax.add_patch(Rectangle((0.95, disk_y), 0.7, disk_h, facecolor=COLOR_BLUE,
                           edgecolor=COLOR_BLUE, alpha=0.85))
    ax.text(1.3, disk_y + disk_h / 2, "GPT\nhdr",
            ha="center", va="center", color="white", fontsize=8,
            fontweight="bold")

    # Several partitions
    gpt_parts = [
        ("EFI", COLOR_AMBER),
        ("/boot", COLOR_GREEN),
        ("/", COLOR_BLUE),
        ("/home", COLOR_PURPLE),
        ("data", COLOR_GREY),
        ("...", COLOR_LIGHT),
    ]
    p_x = 1.7
    p_w = (9.6 - 1.7 - 0.8) / len(gpt_parts)
    for label, color in gpt_parts:
        text_color = "white" if color != COLOR_LIGHT else COLOR_GREY
        ax.add_patch(Rectangle((p_x, disk_y), p_w, disk_h, facecolor=color,
                               edgecolor="white", alpha=0.85, lw=1.4))
        ax.text(p_x + p_w / 2, disk_y + disk_h / 2, label,
                ha="center", va="center", color=text_color,
                fontsize=10, fontweight="bold")
        p_x += p_w

    # Backup GPT header at the end
    ax.add_patch(Rectangle((9.0, disk_y), 0.6, disk_h, facecolor=COLOR_BLUE,
                           edgecolor=COLOR_BLUE, alpha=0.85))
    ax.text(9.3, disk_y + disk_h / 2, "GPT\nbak",
            ha="center", va="center", color="white", fontsize=8,
            fontweight="bold")

    # Redundancy arrow
    arr = FancyArrowPatch((1.3, disk_y - 0.05), (9.3, disk_y - 0.05),
                          arrowstyle="<->", mutation_scale=14,
                          color=COLOR_BLUE, lw=1.4)
    ax.add_patch(arr)
    ax.text(5.3, disk_y - 0.5,
            "primary + backup header  =  partition table is recoverable",
            ha="center", fontsize=9, color=COLOR_BLUE, style="italic")

    ax.text(5.0, 5.4,
            "64-bit LBA  -  practically unlimited capacity (ZB scale)",
            ha="center", fontsize=10, color="#064e3b", fontweight="bold")
    ax.text(5.0, 4.85,
            "up to 128 partitions by default, named with GUIDs",
            ha="center", fontsize=9.5, color=COLOR_GREY)
    ax.text(5.0, 4.45,
            "CRC32 checksums on header + table  ->  detect corruption",
            ha="center", fontsize=9.5, color=COLOR_GREY)

    fig.suptitle("Partition Layout: MBR vs GPT",
                 fontsize=15, fontweight="bold", y=1.02, color="#1e293b")

    _save(fig, "fig2_partition_layout.png")


# ---------------------------------------------------------------------------
# Figure 3: LVM stack (PV -> VG -> LV)
# ---------------------------------------------------------------------------

def fig3_lvm_stack() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ---- Layer 1: physical disks / partitions (PV) ----
    pv_y = 0.6
    pv_h = 1.3
    pvs = ["/dev/sdb", "/dev/sdc", "/dev/sdd1"]
    pv_w = 2.6
    pv_gap = 0.5
    total_w = len(pvs) * pv_w + (len(pvs) - 1) * pv_gap
    pv_x0 = (12 - total_w) / 2
    for i, name in enumerate(pvs):
        x = pv_x0 + i * (pv_w + pv_gap)
        _rounded_box(ax, x, pv_y, pv_w, pv_h, facecolor=COLOR_GREY,
                     edgecolor=COLOR_GREY, alpha=0.85)
        ax.text(x + pv_w / 2, pv_y + pv_h / 2 + 0.15, name,
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        ax.text(x + pv_w / 2, pv_y + pv_h / 2 - 0.3, "PV",
                ha="center", va="center", color="white",
                fontsize=9, style="italic")

    ax.text(0.4, pv_y + pv_h / 2, "Physical\nVolumes",
            ha="left", va="center", fontsize=10, color=COLOR_GREY,
            fontweight="bold")
    ax.text(0.4, pv_y - 0.3, "pvcreate",
            ha="left", va="center", fontsize=8.5, color=COLOR_GREY,
            family="monospace")

    # ---- Layer 2: volume group (VG) - one big pool ----
    vg_y = 3.2
    vg_h = 1.4
    vg_x = pv_x0 - 0.2
    vg_w = total_w + 0.4
    _rounded_box(ax, vg_x, vg_y, vg_w, vg_h, facecolor=COLOR_PURPLE,
                 edgecolor=COLOR_PURPLE, alpha=0.85)
    ax.text(vg_x + vg_w / 2, vg_y + vg_h / 2 + 0.2, "vg_data",
            ha="center", va="center", color="white",
            fontsize=14, fontweight="bold")
    ax.text(vg_x + vg_w / 2, vg_y + vg_h / 2 - 0.3,
            "Volume Group  -  pool of extents from all PVs",
            ha="center", va="center", color="white",
            fontsize=10, style="italic")

    ax.text(0.4, vg_y + vg_h / 2, "Volume\nGroup",
            ha="left", va="center", fontsize=10, color=COLOR_PURPLE,
            fontweight="bold")
    ax.text(0.4, vg_y - 0.3, "vgcreate / vgextend",
            ha="left", va="center", fontsize=8.5, color=COLOR_PURPLE,
            family="monospace")

    # Arrows from PVs to VG
    for i in range(len(pvs)):
        x = pv_x0 + i * (pv_w + pv_gap) + pv_w / 2
        arr = FancyArrowPatch((x, pv_y + pv_h),
                              (x, vg_y),
                              arrowstyle="->", mutation_scale=14,
                              color=COLOR_PURPLE, lw=1.4)
        ax.add_patch(arr)

    # ---- Layer 3: logical volumes (LV) ----
    lv_y = 5.6
    lv_h = 1.4
    lvs = [
        ("lv_root",  20, COLOR_BLUE),
        ("lv_var",   30, COLOR_AMBER),
        ("lv_data",  50, COLOR_GREEN),
    ]
    lv_x = vg_x
    total_pct = sum(p for _, p, _ in lvs)
    for name, pct, color in lvs:
        w = vg_w * pct / total_pct - 0.1
        _rounded_box(ax, lv_x, lv_y, w, lv_h, facecolor=color,
                     edgecolor=color, alpha=0.9)
        ax.text(lv_x + w / 2, lv_y + lv_h / 2 + 0.18, name,
                ha="center", va="center", color="white",
                fontsize=11.5, fontweight="bold")
        ax.text(lv_x + w / 2, lv_y + lv_h / 2 - 0.3,
                f"{pct} % of VG",
                ha="center", va="center", color="white", fontsize=9)
        lv_x += w + 0.1

    ax.text(0.4, lv_y + lv_h / 2, "Logical\nVolumes",
            ha="left", va="center", fontsize=10, color=COLOR_BLUE,
            fontweight="bold")
    ax.text(0.4, lv_y - 0.3, "lvcreate / lvextend",
            ha="left", va="center", fontsize=8.5, color=COLOR_BLUE,
            family="monospace")

    # Arrow from VG up to LVs
    arr = FancyArrowPatch((vg_x + vg_w / 2, vg_y + vg_h),
                          (vg_x + vg_w / 2, lv_y),
                          arrowstyle="->", mutation_scale=16,
                          color=COLOR_BLUE, lw=1.6)
    ax.add_patch(arr)

    # Title and tag line
    ax.text(6, 7.6, "LVM stack:  Physical Volume  ->  Volume Group  ->  Logical Volume",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color="#1e293b")
    ax.text(6, 7.2,
            "Decouples filesystems from physical disks - resize, migrate "
            "and snapshot without re-partitioning",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY,
            style="italic")

    _save(fig, "fig3_lvm_stack.png")


# ---------------------------------------------------------------------------
# Figure 4: Mount points and the mount table
# ---------------------------------------------------------------------------

def fig4_mount_table() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Block device box (left)
    _rounded_box(ax, 0.4, 3.0, 2.6, 1.6, facecolor=COLOR_GREY,
                 edgecolor=COLOR_GREY, alpha=0.9)
    ax.text(1.7, 4.05, "/dev/sdb1", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(1.7, 3.55, "block device", ha="center", va="center",
            color="white", fontsize=9, style="italic")
    ax.text(1.7, 3.2, "raw bytes", ha="center", va="center",
            color="white", fontsize=8.5)

    # Filesystem driver box (middle)
    _rounded_box(ax, 4.2, 3.0, 2.6, 1.6, facecolor=COLOR_PURPLE,
                 edgecolor=COLOR_PURPLE, alpha=0.9)
    ax.text(5.5, 4.05, "ext4 / xfs", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(5.5, 3.55, "filesystem driver", ha="center", va="center",
            color="white", fontsize=9, style="italic")
    ax.text(5.5, 3.2, "interprets layout", ha="center", va="center",
            color="white", fontsize=8.5)

    # Mount point box (right)
    _rounded_box(ax, 8.0, 3.0, 2.6, 1.6, facecolor=COLOR_BLUE,
                 edgecolor=COLOR_BLUE, alpha=0.9)
    ax.text(9.3, 4.05, "/mnt/data", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(9.3, 3.55, "mount point", ha="center", va="center",
            color="white", fontsize=9, style="italic")
    ax.text(9.3, 3.2, "directory in tree", ha="center", va="center",
            color="white", fontsize=8.5)

    # Arrows
    for x0, x1 in [(3.0, 4.2), (6.8, 8.0)]:
        arr = FancyArrowPatch((x0, 3.8), (x1, 3.8), arrowstyle="->",
                              mutation_scale=18, color=COLOR_GREY, lw=1.6)
        ax.add_patch(arr)
    ax.text(3.6, 4.1, "open()", ha="center", fontsize=8.5,
            color=COLOR_GREY, family="monospace")
    ax.text(7.4, 4.1, "mount()", ha="center", fontsize=8.5,
            color=COLOR_GREY, family="monospace")

    # mount command at the top
    ax.text(6, 6.5,
            "mount -t ext4 /dev/sdb1 /mnt/data",
            ha="center", va="center", fontsize=14, fontweight="bold",
            family="monospace", color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7",
                      edgecolor=COLOR_AMBER, lw=1.2))
    ax.text(6, 5.85,
            "Three things must agree: a block device, a filesystem driver, "
            "and a mount point.",
            ha="center", va="center", fontsize=10, color=COLOR_GREY,
            style="italic")

    # /etc/fstab table at the bottom
    ax.text(6, 2.3, "/etc/fstab  -  persistent mount table",
            ha="center", fontsize=11, fontweight="bold", color="#1e293b")

    fstab_lines = [
        "# <device>            <mount>     <fs>    <options>           <dump> <pass>",
        "UUID=8f1c-...-3a   /          ext4    defaults,noatime    0      1",
        "UUID=4b2e-...-9d   /home      ext4    defaults            0      2",
        "UUID=12ab-...-77   /data      xfs     defaults            0      2",
        "tmpfs              /tmp       tmpfs   defaults,size=2G    0      0",
    ]
    table_y = 1.85
    line_h = 0.32
    ax.add_patch(Rectangle((0.6, 0.05), 10.8, len(fstab_lines) * line_h + 0.25,
                           facecolor="#f8fafc", edgecolor=COLOR_GREY, lw=1.0))
    for i, line in enumerate(fstab_lines):
        color = COLOR_GREY if i == 0 else "#1e293b"
        weight = "normal" if i == 0 else "normal"
        ax.text(0.85, table_y - i * line_h, line,
                ha="left", va="center", fontsize=8.8,
                family="monospace", color=color, fontweight=weight)

    ax.set_title("Mount Points and the Mount Table",
                 fontsize=14, fontweight="bold", pad=12, color="#1e293b")

    _save(fig, "fig4_mount_table.png")


# ---------------------------------------------------------------------------
# Figure 5: ext4 vs xfs vs btrfs comparison grid
# ---------------------------------------------------------------------------

def fig5_filesystem_comparison() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    fss = ["ext4", "xfs", "btrfs"]
    fs_colors = [COLOR_BLUE, COLOR_PURPLE, COLOR_GREEN]

    rows = [
        ("Default on",            ["Debian / Ubuntu", "RHEL 7+ / Rocky", "openSUSE / Fedora WS"]),
        ("Max file size",         ["16 TiB",          "8 EiB",            "16 EiB"]),
        ("Online grow",           ["yes",             "yes",              "yes"]),
        ("Online shrink",         ["no (offline)",    "no",               "yes"]),
        ("Snapshots",             ["via LVM",         "via LVM",          "native (CoW)"]),
        ("Data checksums",        ["metadata only",   "metadata only",    "data + metadata"]),
        ("Built-in RAID",         ["no",              "no",               "RAID 0/1/10"]),
        ("Best fit",              ["general purpose", "large files / parallel I/O",
                                                                          "snapshots / multi-disk"]),
    ]

    # Table geometry
    col0_x = 0.4
    col0_w = 3.2
    col_w = 2.7
    col_gap = 0.05
    row_h = 0.75
    table_top = 8.0

    # Header row with FS names
    for i, (fs, color) in enumerate(zip(fss, fs_colors)):
        x = col0_x + col0_w + col_gap + i * (col_w + col_gap)
        _rounded_box(ax, x, table_top, col_w, row_h * 1.05,
                     facecolor=color, edgecolor=color, alpha=0.95)
        ax.text(x + col_w / 2, table_top + row_h * 0.5, fs,
                ha="center", va="center", color="white",
                fontsize=15, fontweight="bold")

    # Empty top-left corner
    ax.text(col0_x + col0_w / 2, table_top + row_h * 0.5,
            "dimension",
            ha="center", va="center", color=COLOR_GREY,
            fontsize=10, fontweight="bold", style="italic")

    # Body rows
    for r, (label, values) in enumerate(rows):
        y = table_top - (r + 1) * row_h
        # Label cell
        ax.add_patch(Rectangle((col0_x, y), col0_w, row_h * 0.95,
                               facecolor="#f1f5f9", edgecolor=COLOR_LIGHT,
                               lw=0.8))
        ax.text(col0_x + 0.2, y + row_h * 0.5, label,
                ha="left", va="center", fontsize=10.5, color="#1e293b",
                fontweight="bold")
        for i, (val, color) in enumerate(zip(values, fs_colors)):
            x = col0_x + col0_w + col_gap + i * (col_w + col_gap)
            ax.add_patch(Rectangle((x, y), col_w, row_h * 0.95,
                                   facecolor="white", edgecolor=COLOR_LIGHT,
                                   lw=0.8))
            ax.text(x + col_w / 2, y + row_h * 0.5, val,
                    ha="center", va="center", fontsize=9.8,
                    color=color, fontweight="bold")

    ax.set_title("Filesystem Comparison:  ext4  /  xfs  /  btrfs",
                 fontsize=14, fontweight="bold", pad=12, color="#1e293b")

    _save(fig, "fig5_filesystem_comparison.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_filesystem_hierarchy()
    fig2_partition_layout()
    fig3_lvm_stack()
    fig4_mount_table()
    fig5_filesystem_comparison()
    print("Wrote 5 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
