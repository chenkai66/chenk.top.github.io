"""
Figure generation script for Cloud Computing Part 03: Storage Systems.

Generates 5 figures used by both EN and ZH versions of the article. Each
figure isolates a single concept and is rendered as a clean, presentation-
quality diagram.

Figures:
    fig1_storage_comparison    Object vs Block vs File storage compared on
                               6 dimensions (grouped bars).
    fig2_s3_architecture       Logical layout of an S3-style object store
                               (clients -> regions -> AZs -> partitions ->
                               replicas).
    fig3_cap_theorem           CAP theorem Venn diagram with real systems
                               placed in CP / AP / CA zones.
    fig4_erasure_vs_replication Storage overhead and durability trade-off
                               between 3x replication and erasure codes.
    fig5_distributed_fs        Side-by-side architectural comparison of
                               HDFS and Ceph (master/peer-to-peer).

Usage:
    python3 scripts/figures/cloud-computing/03-storage.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

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
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "cloud-computing" / "storage-systems"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "cloud-computing" / "storage-systems"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, x, y, w, h, label, color, text_color="white", fontsize=10,
         fontweight="bold", radius=0.04, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.01,rounding_size={radius}",
                       linewidth=0, facecolor=color, alpha=alpha)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight=fontweight)


# ---------------------------------------------------------------------------
# Figure 1: Object vs Block vs File storage comparison
# ---------------------------------------------------------------------------
def fig1_storage_comparison() -> None:
    """Compare three storage types across six practical dimensions."""
    dims = [
        "Scalability",
        "Throughput",
        "Latency\n(lower=better)",
        "Cost\nefficiency",
        "Metadata\nrichness",
        "Random\nwrite",
    ]
    # Subjective 1-5 (higher is better; latency inverted so higher == lower latency)
    block = [2, 5, 5, 2, 2, 5]
    file_ = [3, 4, 4, 3, 3, 4]
    obj   = [5, 4, 2, 5, 5, 1]

    x = np.arange(len(dims))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x - w, block, w, label="Block (EBS, Azure Disk)",
           color=C_BLUE, edgecolor="white")
    ax.bar(x,     file_, w, label="File (EFS, Azure Files)",
           color=C_PURPLE, edgecolor="white")
    ax.bar(x + w, obj,   w, label="Object (S3, OSS, GCS)",
           color=C_GREEN, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 5.7)
    ax.set_yticks(range(0, 6))
    ax.set_ylabel("Strength  (higher is better)", fontsize=10)
    ax.set_title("Object vs Block vs File: each shines in a different dimension",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.set_axisbelow(True)

    # Footnote
    ax.text(0.0, -0.18,
            "Read this as: Block wins on latency and random write; "
            "Object wins on scale, cost and metadata; File sits in between.",
            transform=ax.transAxes, ha="left", fontsize=9, color=C_DARK,
            style="italic")

    _save(fig, "fig1_storage_comparison")


# ---------------------------------------------------------------------------
# Figure 2: S3-style object store architecture
# ---------------------------------------------------------------------------
def fig2_s3_architecture() -> None:
    """Logical request path through an S3-style object store."""
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.set_title("Object Store Architecture: from PUT request to durable replicas",
                 fontsize=13, fontweight="bold", pad=12)

    # Layer labels (left)
    layer_labels = [
        (6.2, "Client"),
        (5.0, "Edge / API"),
        (3.6, "Region (multi-AZ)"),
        (2.0, "Storage layer"),
        (0.6, "Replicas / EC shards"),
    ]
    for y, t in layer_labels:
        ax.text(0.05, y + 0.35, t, fontsize=9, color=C_GRAY,
                fontweight="bold", style="italic")

    # Client
    _box(ax, 5.0, 6.05, 2.0, 0.7, "PUT bucket/key", C_DARK, fontsize=10)

    # Edge / API
    _box(ax, 4.0, 4.85, 1.8, 0.75, "Edge POP\n(TLS, auth)", C_BLUE, fontsize=9)
    _box(ax, 6.2, 4.85, 1.8, 0.75, "Request router\n(SigV4)", C_PURPLE, fontsize=9)

    # Region
    for i, (x, lbl) in enumerate([(2.2, "AZ-a\nFront-end"),
                                  (5.1, "AZ-b\nFront-end"),
                                  (8.0, "AZ-c\nFront-end")]):
        _box(ax, x, 3.4, 1.8, 0.75, lbl, C_GREEN, fontsize=9)

    # Storage layer (partition / index service)
    _box(ax, 1.6, 1.85, 3.0, 0.8, "Index service\n(bucket -> partitions)",
         C_AMBER, fontsize=9)
    _box(ax, 5.0, 1.85, 3.0, 0.8, "Placement\n(consistent hash)",
         C_AMBER, fontsize=9)
    _box(ax, 8.4, 1.85, 2.6, 0.8, "Background\nGC / repair",
         C_GRAY, fontsize=9)

    # Replicas
    for i, x in enumerate([1.4, 3.2, 5.0, 6.8, 8.6, 10.4]):
        _box(ax, x, 0.35, 1.4, 0.75,
             f"OSD {i+1}", C_BLUE, fontsize=9, alpha=0.85 - i * 0.05)

    # Arrows
    def arrow(p1, p2, color=C_DARK, lw=1.4, style="->"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style,
                                     mutation_scale=12, color=color, linewidth=lw))

    arrow((6.0, 6.05), (4.9, 5.6))
    arrow((6.0, 6.05), (7.1, 5.6))
    arrow((4.9, 4.85), (3.1, 4.15))
    arrow((4.9, 4.85), (6.0, 4.15))
    arrow((7.1, 4.85), (8.9, 4.15))
    arrow((6.0, 3.4),  (3.1, 2.65))
    arrow((6.0, 3.4),  (6.5, 2.65))
    for x in [1.4, 3.2, 5.0, 6.8, 8.6, 10.4]:
        arrow((6.0, 1.85), (x + 0.7, 1.1), color=C_GRAY, lw=0.9)

    # Annotation: durability line
    ax.text(11.95, 0.05,
            "Durability: 11 nines via 3-way replication or 6+3 erasure coding",
            ha="right", va="bottom", fontsize=9, color=C_DARK, style="italic")

    _save(fig, "fig2_s3_architecture")


# ---------------------------------------------------------------------------
# Figure 3: CAP theorem Venn diagram
# ---------------------------------------------------------------------------
def fig3_cap_theorem() -> None:
    """Three-circle Venn with real systems placed in CP / AP / (CA)."""
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(-1.0, 11.0)
    ax.set_ylim(-0.5, 8.0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.set_title("CAP Theorem: pick two — but partition tolerance is non-negotiable",
                 fontsize=13, fontweight="bold", pad=12)

    cx, cy, r = 5.0, 4.0, 2.6
    # Three circles
    centers = {
        "Consistency":         (cx - 1.5, cy + 1.0, C_BLUE),
        "Availability":        (cx + 1.5, cy + 1.0, C_PURPLE),
        "Partition Tolerance": (cx,       cy - 1.4, C_GREEN),
    }
    for name, (x, y, color) in centers.items():
        ax.add_patch(Circle((x, y), r, facecolor=color, alpha=0.22,
                            edgecolor=color, linewidth=2))
        # Labels just outside
        if name == "Consistency":
            ax.text(x - 1.6, y + 1.5, name, fontsize=12, fontweight="bold",
                    color=color, ha="right")
        elif name == "Availability":
            ax.text(x + 1.6, y + 1.5, name, fontsize=12, fontweight="bold",
                    color=color, ha="left")
        else:
            ax.text(x, y - 2.1, name, fontsize=12, fontweight="bold",
                    color=color, ha="center")

    # Region labels
    ax.text(cx - 2.4, cy - 0.6, "CP",
            fontsize=18, fontweight="bold", color=C_BLUE, ha="center")
    ax.text(cx - 2.4, cy - 1.1,
            "HBase, MongoDB\n(majority writes),\nZooKeeper, etcd",
            fontsize=9, color=C_DARK, ha="center")

    ax.text(cx + 2.4, cy - 0.6, "AP",
            fontsize=18, fontweight="bold", color=C_PURPLE, ha="center")
    ax.text(cx + 2.4, cy - 1.1,
            "Cassandra,\nDynamoDB,\nRiak, CouchDB",
            fontsize=9, color=C_DARK, ha="center")

    ax.text(cx, cy + 1.8, "CA",
            fontsize=14, fontweight="bold", color=C_GRAY, ha="center")
    ax.text(cx, cy + 1.3,
            "single-node RDBMS\n(no real partitions)",
            fontsize=9, color=C_GRAY, ha="center", style="italic")

    # Caveat box
    note = ("In practice every distributed system must tolerate partitions, "
            "so the real choice is CP vs AP. 'CA' only applies to systems "
            "that aren't actually distributed.")
    ax.text(5.0, -0.2, note, ha="center", va="bottom", fontsize=9,
            color=C_DARK, style="italic", wrap=True)

    _save(fig, "fig3_cap_theorem")


# ---------------------------------------------------------------------------
# Figure 4: Erasure coding vs replication
# ---------------------------------------------------------------------------
def fig4_erasure_vs_replication() -> None:
    """Storage overhead vs durability for replication and EC schemes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # ---- Left: overhead bar comparison ----
    ax = axes[0]
    schemes = ["1x\n(none)", "2x\nrepl", "3x\nrepl", "EC 6+3\n(50%)", "EC 10+4\n(40%)", "EC 12+4\n(33%)"]
    overhead = [0, 100, 200, 50, 40, 33]   # % overhead
    colors = [C_GRAY, C_AMBER, C_AMBER, C_GREEN, C_GREEN, C_GREEN]
    bars = ax.bar(schemes, overhead, color=colors, edgecolor="white")
    ax.set_ylabel("Storage overhead (%)", fontsize=10)
    ax.set_title("EC drastically cuts storage overhead", fontsize=11,
                 fontweight="bold", pad=8)
    ax.set_ylim(0, 230)
    for b, v in zip(bars, overhead):
        ax.text(b.get_x() + b.get_width() / 2, v + 6,
                f"{v}%", ha="center", fontsize=10, fontweight="bold",
                color=C_DARK)
    ax.set_axisbelow(True)

    # ---- Right: durability / failure tolerance ----
    ax2 = axes[1]
    labels = ["3x replication\n(tolerate 2 disks)",
              "EC 6+3\n(tolerate 3 shards)",
              "EC 10+4\n(tolerate 4 shards)",
              "EC 12+4\n(tolerate 4 shards)"]
    durability_nines = [6, 11, 13, 13]    # rough orders of magnitude
    repair_cost      = [1, 6, 10, 12]     # relative shards read on rebuild
    x = np.arange(len(labels))
    w = 0.36
    ax2.bar(x - w/2, durability_nines, w, label="Durability (nines)",
            color=C_BLUE, edgecolor="white")
    ax2.bar(x + w/2, repair_cost, w, label="Rebuild I/O cost (relative)",
            color=C_PURPLE, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Higher EC durability comes with higher rebuild cost",
                  fontsize=11, fontweight="bold", pad=8)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax2.set_ylim(0, 16)
    ax2.set_axisbelow(True)

    fig.suptitle("Replication vs Erasure Coding: cheaper storage trades against higher CPU and rebuild traffic",
                 fontsize=12.5, fontweight="bold", y=1.02)

    _save(fig, "fig4_erasure_vs_replication")


# ---------------------------------------------------------------------------
# Figure 5: HDFS vs Ceph distributed file system architecture
# ---------------------------------------------------------------------------
def fig5_distributed_fs() -> None:
    """Side-by-side: HDFS (master-based) vs Ceph (CRUSH peer-to-peer)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))

    # ----- HDFS on the left -----
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("HDFS (master / worker)", fontsize=12, fontweight="bold", pad=10)

    _box(ax, 1.5, 6.6, 3.0, 0.9, "Client", C_DARK, fontsize=11)
    _box(ax, 5.5, 6.6, 3.0, 0.9, "Resource\nManager (YARN)", C_GRAY, fontsize=10)
    _box(ax, 1.5, 4.7, 3.0, 1.1, "NameNode\n(metadata)", C_BLUE, fontsize=11)
    _box(ax, 5.5, 4.7, 3.0, 1.1, "Standby\nNameNode (HA)", C_BLUE, fontsize=11, alpha=0.7)

    for i, x in enumerate([0.5, 3.0, 5.5, 8.0]):
        _box(ax, x, 1.5, 1.7, 1.2, f"DataNode\n{i+1}", C_GREEN, fontsize=10)

    # Replication arrows
    for x_src in [0.5, 3.0, 5.5, 8.0]:
        for x_dst in [0.5, 3.0, 5.5, 8.0]:
            if x_src < x_dst:
                ax.add_patch(FancyArrowPatch(
                    (x_src + 0.85, 1.5), (x_dst + 0.85, 1.5),
                    arrowstyle="-", color=C_GRAY, linewidth=0.6,
                    connectionstyle="arc3,rad=-0.35", alpha=0.5))

    # Vertical control lines
    for p1, p2 in [((3.0, 6.6), (3.0, 5.8)),
                   ((3.0, 4.7), (1.35, 2.7)),
                   ((3.0, 4.7), (3.85, 2.7)),
                   ((3.0, 4.7), (6.35, 2.7)),
                   ((3.0, 4.7), (8.85, 2.7))]:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->",
                                     color=C_DARK, linewidth=1.2,
                                     mutation_scale=10))

    ax.text(5.0, 0.5,
            "Single NameNode = central metadata.\n"
            "128 MB blocks, 3x replication, rack-aware placement.",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    # ----- Ceph on the right -----
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Ceph (peer-to-peer with CRUSH)",
                 fontsize=12, fontweight="bold", pad=10)

    _box(ax, 1.0, 6.6, 2.4, 0.9, "RGW\n(object)", C_BLUE, fontsize=10)
    _box(ax, 3.8, 6.6, 2.4, 0.9, "RBD\n(block)", C_PURPLE, fontsize=10)
    _box(ax, 6.6, 6.6, 2.4, 0.9, "CephFS\n(file)", C_GREEN, fontsize=10)

    # Monitors + MDS
    _box(ax, 0.6, 4.6, 2.6, 1.1, "Monitors\n(cluster map)", C_AMBER, fontsize=10)
    _box(ax, 3.8, 4.6, 2.4, 1.1, "MDS\n(CephFS meta)", C_AMBER, fontsize=10, alpha=0.85)
    _box(ax, 6.8, 4.6, 2.6, 1.1, "Managers\n(metrics, dash)", C_AMBER, fontsize=10, alpha=0.7)

    # OSDs in a ring
    osd_positions = [(1.0, 1.6), (3.0, 1.2), (5.0, 1.0),
                     (7.0, 1.2), (8.6, 1.6)]
    for i, (x, y) in enumerate(osd_positions):
        _box(ax, x, y, 1.2, 1.0, f"OSD\n{i+1}", C_GREEN, fontsize=9)

    # CRUSH peer arcs
    for i in range(len(osd_positions)):
        for j in range(i + 1, len(osd_positions)):
            (x1, y1) = osd_positions[i]; (x2, y2) = osd_positions[j]
            ax.add_patch(FancyArrowPatch(
                (x1 + 0.6, y1 + 0.5), (x2 + 0.6, y2 + 0.5),
                arrowstyle="-", color=C_GRAY, linewidth=0.5,
                connectionstyle="arc3,rad=0.18", alpha=0.55))

    # Vertical lines from interfaces to RADOS layer
    for x_src in [2.2, 5.0, 7.8]:
        ax.add_patch(FancyArrowPatch((x_src, 6.6), (x_src, 5.7),
                                     arrowstyle="->", color=C_DARK,
                                     linewidth=1.2, mutation_scale=10))

    ax.text(5.0, 0.3,
            "No single master. Clients compute placement with CRUSH +\n"
            "the cluster map; OSDs replicate / repair peer-to-peer.",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    fig.suptitle("Distributed File Systems: centralised metadata vs algorithmic placement",
                 fontsize=13, fontweight="bold", y=1.02)

    _save(fig, "fig5_distributed_fs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating cloud-computing/03-storage figures…")
    fig1_storage_comparison()
    fig2_s3_architecture()
    fig3_cap_theorem()
    fig4_erasure_vs_replication()
    fig5_distributed_fs()
    print("Done.")


if __name__ == "__main__":
    main()
