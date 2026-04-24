"""
Figure generator for Computer Fundamentals Part 05: Network & Power.

Produces 7 production-quality figures covering NIC architecture, TCP
3-way handshake, network topologies, PSU 80 PLUS efficiency curves,
datacenter PUE breakdown, component power hierarchy, and UPS / battery
backup architecture.

Outputs PNGs (dpi=150) into the EN and ZH asset directories:
  source/_posts/en/computer-fundamentals/05-network-power/
  source/_posts/zh/computer-fundamentals/05-network-power/

Style: matplotlib seaborn-v0_8-whitegrid; brand palette
  blue   #2563eb  (primary)
  purple #7c3aed  (accent)
  green  #10b981  (success / efficient)
  amber  #f59e0b  (warning / loss)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Wedge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
DARK = "#0f172a"
GRAY = "#64748b"
LIGHT = "#f1f5f9"

DPI = 150

REPO = Path(__file__).resolve().parents[3]
OUT_DIRS = [
    REPO / "source/_posts/en/computer-fundamentals/05-network-power",
    REPO / "source/_posts/zh/computer-fundamentals/05-network-power",
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


def _box(ax, x, y, w, h, label, color, text_color="white", fontsize=10, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=color, edgecolor=DARK, linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold")


# ---------------------------------------------------------------------------
# Figure 1: NIC architecture (PHY, MAC, DMA)
# ---------------------------------------------------------------------------

def fig1_nic_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6.5, 6.6, "Network Interface Card (NIC) Architecture",
            ha="center", fontsize=15, fontweight="bold", color=DARK)
    ax.text(6.5, 6.25, "From cable to CPU memory: layered hardware pipeline",
            ha="center", fontsize=10, color=GRAY, style="italic")

    # Outside world
    _box(ax, 0.3, 2.8, 1.5, 1.2, "Ethernet\nCable\n(RJ-45)", "#475569", fontsize=9)
    ax.annotate("", xy=(2.0, 3.4), xytext=(1.85, 3.4),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.8))

    # PHY layer
    _box(ax, 2.0, 2.5, 2.2, 1.8, "PHY\n(Physical Layer)", BLUE, fontsize=11)
    ax.text(3.1, 2.25, "Encode/decode\n100BASE-TX, 1000BASE-T\nSerDes, magnetics",
            ha="center", fontsize=8, color=DARK)

    ax.annotate("", xy=(4.5, 3.4), xytext=(4.25, 3.4),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.8))

    # MAC layer
    _box(ax, 4.5, 2.5, 2.4, 1.8, "MAC\n(Media Access)", PURPLE, fontsize=11)
    ax.text(5.7, 2.25, "Frame assembly, CRC\nMAC address filter\nFlow control",
            ha="center", fontsize=8, color=DARK)

    ax.annotate("", xy=(7.2, 3.4), xytext=(6.95, 3.4),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.8))

    # Buffer / DMA engine
    _box(ax, 7.2, 2.5, 2.4, 1.8, "DMA Engine\n+ Ring Buffer", GREEN, fontsize=11)
    ax.text(8.4, 2.25, "Descriptor rings\nZero-copy to RAM\nMSI-X interrupts",
            ha="center", fontsize=8, color=DARK)

    ax.annotate("", xy=(9.9, 3.4), xytext=(9.65, 3.4),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.8))

    # Host memory / CPU
    _box(ax, 9.9, 2.5, 2.7, 1.8, "Host RAM\n(skb / mbuf)", AMBER, fontsize=11)
    ax.text(11.25, 2.25, "Kernel network stack\nTCP/IP processing",
            ha="center", fontsize=8, color=DARK)

    # Top: control path
    _box(ax, 4.5, 4.8, 5.1, 0.9, "PCIe Bus  (Gen 3 x4 = 4 GB/s | Gen 4 x4 = 8 GB/s)",
         "#0f172a", fontsize=10)
    ax.annotate("", xy=(5.7, 4.8), xytext=(5.7, 4.3),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))
    ax.annotate("", xy=(8.4, 4.8), xytext=(8.4, 4.3),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))

    # Bottom: feature row
    features = [
        ("Checksum\nOffload", BLUE),
        ("TSO / LRO\nSegmentation", PURPLE),
        ("RSS\nMulti-queue", GREEN),
        ("VLAN\nTagging", AMBER),
        ("SR-IOV\nVirtualization", "#0ea5e9"),
    ]
    fx = 1.0
    fw = 2.2
    for label, color in features:
        _box(ax, fx, 0.4, fw, 1.1, label, color, fontsize=9)
        fx += fw + 0.15

    ax.text(6.5, 0.05, "Modern NICs offload work from the CPU: hardware does what software once did",
            ha="center", fontsize=9, color=GRAY, style="italic")

    _save(fig, "fig1_nic_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2: TCP 3-way handshake
# ---------------------------------------------------------------------------

def fig2_tcp_handshake() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 6.6, "TCP 3-Way Handshake", ha="center",
            fontsize=15, fontweight="bold", color=DARK)
    ax.text(6, 6.25, "Establishing a reliable connection: SYN -> SYN-ACK -> ACK",
            ha="center", fontsize=10, color=GRAY, style="italic")

    # Endpoints
    _box(ax, 0.5, 5.2, 2.5, 0.7, "Client", BLUE, fontsize=12)
    _box(ax, 9.0, 5.2, 2.5, 0.7, "Server", PURPLE, fontsize=12)

    # Vertical timelines
    ax.plot([1.75, 1.75], [0.5, 5.2], color=BLUE, lw=2, alpha=0.4)
    ax.plot([10.25, 10.25], [0.5, 5.2], color=PURPLE, lw=2, alpha=0.4)

    # State labels - client
    states_c = [(4.5, "CLOSED"), (3.7, "SYN_SENT"), (2.0, "ESTABLISHED")]
    for y, s in states_c:
        ax.text(0.85, y, s, ha="right", fontsize=8.5, color=BLUE,
                fontweight="bold", bbox=dict(facecolor="white", edgecolor=BLUE, pad=2))

    states_s = [(4.5, "LISTEN"), (3.0, "SYN_RCVD"), (2.0, "ESTABLISHED")]
    for y, s in states_s:
        ax.text(11.15, y, s, ha="left", fontsize=8.5, color=PURPLE,
                fontweight="bold", bbox=dict(facecolor="white", edgecolor=PURPLE, pad=2))

    # Step 1: SYN
    arr = FancyArrowPatch((1.85, 4.4), (10.15, 3.7),
                          arrowstyle="->", mutation_scale=20, color=BLUE, lw=2.2)
    ax.add_patch(arr)
    ax.text(6.0, 4.25, "(1) SYN  seq = x", ha="center", fontsize=11,
            color=BLUE, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=BLUE, pad=4, boxstyle="round"))

    # Step 2: SYN-ACK
    arr2 = FancyArrowPatch((10.15, 3.0), (1.85, 2.4),
                           arrowstyle="->", mutation_scale=20, color=PURPLE, lw=2.2)
    ax.add_patch(arr2)
    ax.text(6.0, 2.85, "(2) SYN-ACK  seq = y, ack = x+1", ha="center", fontsize=11,
            color=PURPLE, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=PURPLE, pad=4, boxstyle="round"))

    # Step 3: ACK
    arr3 = FancyArrowPatch((1.85, 2.0), (10.15, 1.5),
                           arrowstyle="->", mutation_scale=20, color=GREEN, lw=2.2)
    ax.add_patch(arr3)
    ax.text(6.0, 1.85, "(3) ACK  ack = y+1", ha="center", fontsize=11,
            color=GREEN, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=GREEN, pad=4, boxstyle="round"))

    # Bottom note
    _box(ax, 1.0, 0.05, 10.0, 0.65,
         "Why three (not two)? Confirms both directions can send AND receive — defends against half-open state and replayed SYNs.",
         "#1e293b", fontsize=10)

    _save(fig, "fig2_tcp_handshake.png")


# ---------------------------------------------------------------------------
# Figure 3: Network topologies
# ---------------------------------------------------------------------------

def fig3_network_topologies() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Network Topologies — Trade-offs of Cost, Resilience, and Scale",
                 fontsize=15, fontweight="bold", y=0.995)

    def node(ax, x, y, color=BLUE, r=0.13):
        ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=DARK, lw=1.2, zorder=3))

    def edge(ax, p, q, color=GRAY, lw=1.5, alpha=0.9):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, alpha=alpha, zorder=1)

    # --- Bus ---
    ax = axes[0, 0]
    ax.set_xlim(-0.2, 5.2); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title("Bus", color=DARK)
    ax.plot([0.2, 4.8], [1.5, 1.5], color=DARK, lw=3)
    for x in [0.7, 1.7, 2.7, 3.7, 4.5]:
        ax.plot([x, x], [1.5, 2.2], color=GRAY, lw=1.2)
        node(ax, x, 2.4, BLUE)
    ax.text(2.5, 0.7, "Single shared backbone\nCheap | one break = whole net dies",
            ha="center", fontsize=9, color=DARK)

    # --- Ring ---
    ax = axes[0, 1]
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.axis("off")
    ax.set_title("Ring", color=DARK)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    pts = [(np.cos(a), np.sin(a)) for a in angles]
    for i, p in enumerate(pts):
        node(ax, p[0], p[1], PURPLE)
        edge(ax, p, pts[(i + 1) % len(pts)], color=DARK, lw=1.8)
    ax.text(0, -1.7, "Token passing | predictable latency\nOne node down -> bypass",
            ha="center", fontsize=9, color=DARK)

    # --- Star ---
    ax = axes[0, 2]
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.axis("off")
    ax.set_title("Star", color=DARK)
    node(ax, 0, 0, AMBER, r=0.2)
    ax.text(0, -0.35, "switch", ha="center", fontsize=8, color=DARK)
    for a in np.linspace(0, 2 * np.pi, 8)[:-1]:
        x, y = 1.3 * np.cos(a), 1.3 * np.sin(a)
        edge(ax, (0, 0), (x, y), color=GRAY, lw=1.5)
        node(ax, x, y, BLUE)
    ax.text(0, -1.7, "Today's dominant LAN topology\nHub failure = total loss",
            ha="center", fontsize=9, color=DARK)

    # --- Mesh (full) ---
    ax = axes[1, 0]
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.axis("off")
    ax.set_title("Full Mesh", color=DARK)
    n = 6
    pts = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, n + 1)[:-1]]
    for i in range(n):
        for j in range(i + 1, n):
            edge(ax, pts[i], pts[j], color=GRAY, lw=0.8, alpha=0.6)
    for p in pts:
        node(ax, p[0], p[1], GREEN)
    ax.text(0, -1.7, "n*(n-1)/2 links | maximum redundancy\nExpensive at scale",
            ha="center", fontsize=9, color=DARK)

    # --- Tree ---
    ax = axes[1, 1]
    ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Tree (hierarchical)", color=DARK)
    root = (3, 3.4)
    mids = [(1.5, 2.2), (4.5, 2.2)]
    leaves = [(0.6, 1.0), (1.5, 1.0), (2.4, 1.0),
              (3.6, 1.0), (4.5, 1.0), (5.4, 1.0)]
    node(ax, *root, AMBER, r=0.18)
    for m in mids:
        node(ax, *m, PURPLE, r=0.15)
        edge(ax, root, m, color=DARK, lw=1.6)
    for i, lf in enumerate(leaves):
        node(ax, *lf, BLUE)
        edge(ax, mids[i // 3], lf, color=GRAY, lw=1.3)
    ax.text(3, 0.3, "Classic enterprise core / aggregation / access\nOversubscription at upper layers",
            ha="center", fontsize=9, color=DARK)

    # --- Fat-tree (datacenter) ---
    ax = axes[1, 2]
    ax.set_xlim(0, 7); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Fat-Tree (datacenter)", color=DARK)
    core = [(1.5, 3.4), (3.5, 3.4), (5.5, 3.4)]
    agg = [(1.0, 2.2), (2.5, 2.2), (4.0, 2.2), (5.5, 2.2), (6.5, 2.2)]
    edge_sw = [(0.7, 1.0), (1.7, 1.0), (2.7, 1.0),
               (3.7, 1.0), (4.7, 1.0), (5.7, 1.0), (6.5, 1.0)]
    for c in core:
        node(ax, *c, AMBER, r=0.15)
        for a in agg:
            edge(ax, c, a, color=GRAY, lw=0.7, alpha=0.6)
    for a in agg:
        node(ax, *a, PURPLE, r=0.13)
        for e in edge_sw:
            if abs(a[0] - e[0]) < 1.6:
                edge(ax, a, e, color=GRAY, lw=0.8, alpha=0.7)
    for e in edge_sw:
        node(ax, *e, BLUE)
    ax.text(3.5, 0.3, "Equal bandwidth at every layer\nHyperscale standard (Clos network)",
            ha="center", fontsize=9, color=DARK)

    plt.tight_layout()
    _save(fig, "fig3_network_topologies.png")


# ---------------------------------------------------------------------------
# Figure 4: PSU 80 PLUS efficiency curves
# ---------------------------------------------------------------------------

def fig4_psu_efficiency() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    load = np.array([10, 20, 50, 100])  # % load tested by 80 PLUS
    # Approximate 115V internal spec curves
    curves = {
        "80 PLUS White":    (np.array([0,   80, 80, 80]), "#94a3b8"),
        "Bronze":           (np.array([0,   82, 85, 82]), "#a16207"),
        "Silver":           (np.array([0,   85, 88, 85]), "#9ca3af"),
        "Gold":             (np.array([0,   87, 90, 87]), AMBER),
        "Platinum":         (np.array([0,   90, 92, 89]), PURPLE),
        "Titanium":         (np.array([90,  92, 94, 90]), GREEN),
    }

    # x extended to include 10% for titanium
    x_full = np.array([10, 20, 50, 100])

    for name, (vals, color) in curves.items():
        # interpolate from given pivots — pad first val if missing
        if name != "Titanium":
            yvals = np.array([vals[1] - 2, vals[1], vals[2], vals[3]], dtype=float)
        else:
            yvals = vals.astype(float)
        # smooth spline-ish
        xs = np.linspace(10, 100, 200)
        ys = np.interp(xs, x_full, yvals)
        ax.plot(xs, ys, color=color, lw=2.6, label=name, marker="o",
                markevery=[0, 22, 89, 199], markersize=6)

    # Sweet spot band
    ax.axvspan(40, 70, color=GREEN, alpha=0.10)
    ax.text(55, 79.5, "Efficiency sweet spot\n(40–70 % load)",
            ha="center", fontsize=10, color=GREEN, fontweight="bold")

    ax.set_xlabel("Load on PSU  (% of rated wattage)")
    ax.set_ylabel("Conversion efficiency  (%)")
    ax.set_title("80 PLUS Efficiency Curves — Why You Should Not Buy a 1000 W PSU for a 200 W Build",
                 fontsize=13)
    ax.set_xlim(10, 100)
    ax.set_ylim(78, 96)
    ax.legend(loc="lower right", frameon=True, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.4)

    # Annotation: cost example
    ax.annotate("Example: 400 W draw\nGold (90 %) -> 444 W from wall\nWhite (80 %) -> 500 W from wall\nDelta = 56 W = ~$50/yr (8 h/day, $0.12/kWh)",
                xy=(50, 90), xytext=(15, 82),
                fontsize=9, color=DARK,
                bbox=dict(facecolor="white", edgecolor=BLUE, boxstyle="round,pad=0.4"),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

    plt.tight_layout()
    _save(fig, "fig4_psu_efficiency.png")


# ---------------------------------------------------------------------------
# Figure 5: Datacenter PUE breakdown
# ---------------------------------------------------------------------------

def fig5_datacenter_pue() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: donut showing PUE ~1.5 typical datacenter
    sizes = [67, 25, 5, 3]
    labels = ["IT Equipment\n(servers, storage, network)",
              "Cooling\n(CRAC, chillers, fans)",
              "Power conversion\n(UPS, PDU losses)",
              "Lighting & misc."]
    colors = [BLUE, PURPLE, AMBER, GRAY]
    wedges, _ = ax1.pie(sizes, colors=colors, startangle=90,
                        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax1.text(0, 0.18, "PUE", ha="center", fontsize=14, fontweight="bold", color=DARK)
    ax1.text(0, -0.05, "1.50", ha="center", fontsize=28, fontweight="bold", color=BLUE)
    ax1.text(0, -0.30, "(typical)", ha="center", fontsize=9, color=GRAY)
    ax1.set_title("Where every watt goes  —  Industry-average DC", color=DARK)

    # Custom legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=f"{l}  {s}%")
               for c, l, s in zip(colors, labels, sizes)]
    ax1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=1, frameon=False, fontsize=9)

    # Right: bar chart of PUE values across operators
    operators = ["Average\nenterprise DC", "Industry\naverage", "Hyperscale\n(AWS/Azure)",
                 "Google\n(2024)", "Best-in-class\n(immersion)"]
    pue_vals = [2.0, 1.55, 1.18, 1.10, 1.03]
    bar_colors = [RED, AMBER, BLUE, PURPLE, GREEN]
    bars = ax2.bar(operators, pue_vals, color=bar_colors, edgecolor=DARK, linewidth=1.2)

    ax2.axhline(1.0, color=DARK, ls="--", lw=1.5)
    ax2.text(4.5, 1.02, "Theoretical limit (PUE = 1.0)", ha="right",
             fontsize=9, color=DARK, style="italic")

    for bar, v in zip(bars, pue_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}",
                 ha="center", fontsize=11, fontweight="bold", color=DARK)

    ax2.set_ylabel("Power Usage Effectiveness  (lower = better)")
    ax2.set_title("PUE across operators  —  Total power / IT power", color=DARK)
    ax2.set_ylim(0.9, 2.2)
    ax2.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    _save(fig, "fig5_datacenter_pue.png")


# ---------------------------------------------------------------------------
# Figure 6: Component power hierarchy (idle vs load)
# ---------------------------------------------------------------------------

def fig6_power_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    components = ["NVMe SSD", "DDR5 RAM\n(32 GB)", "Motherboard", "Case fans\n(4x)",
                  "CPU\n(i9-13900K)", "GPU\n(RTX 4090)"]
    idle = [0.5, 4, 12, 4, 25, 18]
    load = [8, 10, 50, 16, 253, 450]

    x = np.arange(len(components))
    width = 0.38

    b1 = ax.bar(x - width / 2, idle, width, label="Idle", color=GREEN,
                edgecolor=DARK, linewidth=1.0)
    b2 = ax.bar(x + width / 2, load, width, label="Full load", color=RED,
                edgecolor=DARK, linewidth=1.0)

    for bar, v in zip(b1, idle):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{v} W",
                ha="center", fontsize=9, color=DARK)
    for bar, v in zip(b2, load):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{v} W",
                ha="center", fontsize=9, color=DARK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=10)
    ax.set_ylabel("Power draw  (Watts)")
    ax.set_title("Where the watts go — idle vs full-load consumption per component",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_ylim(0, 530)
    ax.grid(True, axis="y", alpha=0.4)

    # Annotation: GPU dominates
    ax.annotate("GPU + CPU = ~95 % of peak draw\nSize the PSU around these two.",
                xy=(5, 450), xytext=(2.5, 380),
                fontsize=10, color=DARK,
                bbox=dict(facecolor="white", edgecolor=PURPLE, boxstyle="round,pad=0.4"),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4))

    # Total bar at right
    total_idle = sum(idle)
    total_load = sum(load)
    ax.text(0.98, 0.78,
            f"System total\n  Idle: {total_idle} W\n  Load: {total_load} W\n"
            f"PSU rec.: {int(total_load * 1.3)} W (x1.3)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color=DARK,
            bbox=dict(facecolor=LIGHT, edgecolor=BLUE, boxstyle="round,pad=0.5"))

    plt.tight_layout()
    _save(fig, "fig6_power_hierarchy.png")


# ---------------------------------------------------------------------------
# Figure 7: UPS / battery backup architecture
# ---------------------------------------------------------------------------

def fig7_ups_backup() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(6.5, 7.1, "Online (Double-Conversion) UPS — How the Server Stays Up",
            ha="center", fontsize=15, fontweight="bold", color=DARK)
    ax.text(6.5, 6.75, "AC -> DC -> battery bus -> DC -> AC : load is ALWAYS on the inverter (zero-ms transfer)",
            ha="center", fontsize=10, color=GRAY, style="italic")

    # Utility AC in
    _box(ax, 0.3, 3.5, 1.6, 1.2, "Utility AC\n(grid)", "#1e3a8a", fontsize=10)
    ax.annotate("", xy=(2.3, 4.1), xytext=(1.95, 4.1),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))

    # Rectifier
    _box(ax, 2.3, 3.5, 1.7, 1.2, "Rectifier\nAC -> DC", BLUE, fontsize=10)
    ax.annotate("", xy=(4.4, 4.1), xytext=(4.05, 4.1),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))

    # DC bus junction
    _box(ax, 4.4, 3.5, 1.6, 1.2, "DC bus\n(48 V / HVDC)", "#0f172a", fontsize=10)

    # Battery bank (below DC bus)
    _box(ax, 4.4, 1.4, 1.6, 1.4, "Battery\nbank\n(VRLA / Li-ion)", AMBER,
         text_color=DARK, fontsize=10)
    ax.annotate("", xy=(5.2, 3.4), xytext=(5.2, 2.85),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.2))
    ax.text(5.5, 3.1, "float\ncharge", fontsize=8, color=GREEN, fontweight="bold")

    ax.annotate("", xy=(6.4, 4.1), xytext=(6.05, 4.1),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))

    # Inverter
    _box(ax, 6.4, 3.5, 1.7, 1.2, "Inverter\nDC -> AC", PURPLE, fontsize=10)
    ax.annotate("", xy=(8.5, 4.1), xytext=(8.15, 4.1),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))

    # Static bypass switch (top path)
    _box(ax, 4.4, 5.4, 3.7, 0.8, "Static bypass switch  (millisecond fail-safe)",
         GRAY, fontsize=9)
    ax.annotate("", xy=(4.3, 5.8), xytext=(2.0, 4.7),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4, ls="--"))
    ax.annotate("", xy=(8.5, 4.7), xytext=(8.15, 5.8),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4, ls="--"))

    # PDU
    _box(ax, 8.5, 3.5, 1.6, 1.2, "PDU\n(distribution)", "#0ea5e9", fontsize=10)
    ax.annotate("", xy=(10.6, 4.1), xytext=(10.15, 4.1),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=2))

    # Server racks
    _box(ax, 10.6, 3.0, 2.1, 2.0, "Server racks\n(critical load)", GREEN, fontsize=11)

    # Generator at far left bottom
    _box(ax, 0.3, 1.4, 1.6, 1.4, "Diesel\ngenerator\n(15-30 s startup)", "#7c2d12",
         fontsize=9)
    ax.annotate("", xy=(2.4, 2.1), xytext=(1.95, 2.1),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6, ls="--"))
    ax.text(2.15, 2.4, "long-outage path", fontsize=8, color=DARK, style="italic")

    # ATS
    _box(ax, 2.4, 1.4, 1.6, 1.4, "ATS\nAuto transfer\nswitch", "#475569", fontsize=9)
    ax.annotate("", xy=(4.4, 2.1), xytext=(4.05, 2.1),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6, ls="--"))

    # Runtime info
    info = ("Runtime planning\n"
            "  - VRLA  battery: 5–15 min @ rated load\n"
            "  - Li-ion: 8–20 min, 3x cycle life\n"
            "  - Generator must start before depletion\n"
            "  - N+1 redundancy keeps MTBF high")
    ax.text(0.3, 0.55, info, fontsize=9, color=DARK,
            bbox=dict(facecolor=LIGHT, edgecolor=BLUE, boxstyle="round,pad=0.4"),
            va="bottom", ha="left")

    _save(fig, "fig7_ups_backup.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating Part 05 (Network & Power) figures…")
    fig1_nic_architecture()
    fig2_tcp_handshake()
    fig3_network_topologies()
    fig4_psu_efficiency()
    fig5_datacenter_pue()
    fig6_power_hierarchy()
    fig7_ups_backup()
    print("Done.")


if __name__ == "__main__":
    main()
