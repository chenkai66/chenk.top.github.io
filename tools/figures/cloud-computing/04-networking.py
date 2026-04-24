"""
Figure generation script for Cloud Computing Part 04: Networking & SDN.

Generates 5 figures used by both EN and ZH versions of the article.

Figures:
    fig1_vpc_architecture        Multi-AZ VPC with public/private subnets,
                                 IGW, NAT GW, route tables and security
                                 boundaries.
    fig2_sdn_planes              Control plane vs data plane separation;
                                 OpenFlow southbound, REST northbound.
    fig3_lb_l4_l7                L4 (NLB) vs L7 (ALB) load balancers:
                                 capabilities and request flow.
    fig4_cdn_edge                CDN edge distribution map: client routed
                                 to nearest PoP; cache hit vs miss path.
    fig5_bgp_multi_region        BGP routing across multiple regions; AS
                                 paths and ECMP for resilience.

Usage:
    python3 scripts/figures/cloud-computing/04-networking.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "cloud-computing" / "networking-sdn"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "cloud-computing" / "networking-sdn"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, x, y, w, h, label, color, text_color="white", fontsize=10,
         fontweight="bold", radius=0.04, alpha=1.0, edgecolor=None):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.01,rounding_size={radius}",
                       linewidth=0 if edgecolor is None else 1.2,
                       facecolor=color,
                       edgecolor=edgecolor if edgecolor else color,
                       alpha=alpha)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight=fontweight)


def _arrow(ax, p1, p2, color=C_DARK, lw=1.4, style="->"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style,
                                 mutation_scale=12, color=color, linewidth=lw))


# ---------------------------------------------------------------------------
# Figure 1: VPC architecture
# ---------------------------------------------------------------------------
def fig1_vpc_architecture() -> None:
    """Multi-AZ VPC with public + private subnets, IGW, NAT GW."""
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("VPC Architecture: a multi-AZ network with public and private tiers",
                 fontsize=13, fontweight="bold", pad=12)

    # Internet on top
    _box(ax, 5.5, 8.05, 3.0, 0.7, "Internet", C_DARK, fontsize=10)

    # IGW
    _box(ax, 6.3, 7.05, 1.5, 0.6, "IGW", C_AMBER, fontsize=10)
    _arrow(ax, (7.0, 8.05), (7.0, 7.65))

    # VPC outer rectangle
    ax.add_patch(Rectangle((0.4, 0.4), 13.2, 6.4, fill=False,
                           edgecolor=C_BLUE, linewidth=2.0,
                           linestyle="--"))
    ax.text(0.55, 6.6, "VPC  10.0.0.0/16", fontsize=10, color=C_BLUE,
            fontweight="bold")

    # AZ-A and AZ-B containers
    for i, (x, az) in enumerate([(0.8, "AZ-a"), (7.2, "AZ-b")]):
        ax.add_patch(Rectangle((x, 0.7), 6.0, 5.6, fill=True,
                               facecolor=C_BG, edgecolor=C_GRAY,
                               linewidth=1.0))
        ax.text(x + 0.15, 6.05, az, fontsize=10, color=C_GRAY,
                fontweight="bold", style="italic")

    # Public subnets (label moved to top-left, no overlap with boxes inside)
    _box(ax, 1.0, 4.4, 5.6, 1.5, "", C_BLUE, fontsize=10, alpha=0.18)
    _box(ax, 7.4, 4.4, 5.6, 1.5, "", C_BLUE, fontsize=10, alpha=0.18)
    ax.text(1.15, 5.78, "Public subnet  10.0.1.0/24", fontsize=9.5,
            color=C_BLUE, fontweight="bold")
    ax.text(7.55, 5.78, "Public subnet  10.0.3.0/24", fontsize=9.5,
            color=C_BLUE, fontweight="bold")
    # NAT GW + ALB inside public
    _box(ax, 1.4, 4.55, 1.6, 0.75, "NAT GW", C_AMBER, fontsize=9)
    _box(ax, 4.6, 4.55, 1.7, 0.75, "ALB", C_PURPLE, fontsize=9)
    _box(ax, 7.8, 4.55, 1.6, 0.75, "NAT GW", C_AMBER, fontsize=9)
    _box(ax, 11.0, 4.55, 1.7, 0.75, "ALB", C_PURPLE, fontsize=9)

    # Private subnets
    _box(ax, 1.0, 2.4, 5.6, 1.5, "", C_GREEN, fontsize=10, alpha=0.16)
    _box(ax, 7.4, 2.4, 5.6, 1.5, "", C_GREEN, fontsize=10, alpha=0.16)
    ax.text(1.15, 3.78, "Private subnet  10.0.2.0/24", fontsize=9.5,
            color=C_GREEN, fontweight="bold")
    ax.text(7.55, 3.78, "Private subnet  10.0.4.0/24", fontsize=9.5,
            color=C_GREEN, fontweight="bold")
    _box(ax, 1.6, 2.55, 1.9, 0.75, "App", C_GREEN, fontsize=9)
    _box(ax, 4.1, 2.55, 1.9, 0.75, "App", C_GREEN, fontsize=9)
    _box(ax, 8.0, 2.55, 1.9, 0.75, "App", C_GREEN, fontsize=9)
    _box(ax, 10.5, 2.55, 1.9, 0.75, "App", C_GREEN, fontsize=9)

    # DB subnet (isolated)
    _box(ax, 1.0, 0.85, 12.0, 1.3, "", C_PURPLE, fontsize=10, alpha=0.18)
    ax.text(1.15, 2.05, "Database subnet (no internet route)  10.0.5.0/24",
            fontsize=9.5, color=C_PURPLE, fontweight="bold")
    _box(ax, 5.6, 1.0, 1.5, 0.85, "RDS\nprimary", C_PURPLE, fontsize=9)
    _box(ax, 7.3, 1.0, 1.5, 0.85, "RDS\nstandby", C_PURPLE, fontsize=9, alpha=0.7)

    # Arrows: IGW -> Public ALBs
    _arrow(ax, (6.7, 7.05), (5.45, 5.30), color=C_BLUE)
    _arrow(ax, (7.4, 7.05), (11.85, 5.30), color=C_BLUE)
    # Public ALBs -> Private apps
    _arrow(ax, (5.45, 4.55), (5.05, 3.30), color=C_PURPLE)
    _arrow(ax, (11.85, 4.55), (11.45, 3.30), color=C_PURPLE)
    # Private apps -> DB
    _arrow(ax, (3.5, 2.55), (6.0, 1.85), color=C_GREEN)
    _arrow(ax, (10.5, 2.55), (8.0, 1.85), color=C_GREEN)
    # NAT GW -> IGW (egress)
    _arrow(ax, (2.2, 5.30), (6.7, 7.05), color=C_AMBER)
    _arrow(ax, (8.6, 5.30), (7.4, 7.05), color=C_AMBER)

    # Legend
    legend_items = [(C_BLUE, "Inbound (Internet -> ALB)"),
                    (C_PURPLE, "ALB -> App tier"),
                    (C_GREEN, "App -> DB tier"),
                    (C_AMBER, "Egress via NAT GW")]
    for i, (c, t) in enumerate(legend_items):
        x0 = 0.55 + i * 3.1
        _box(ax, x0, 0.05, 0.3, 0.25, "", c, fontsize=8)
        ax.text(x0 + 0.4, 0.18, t, fontsize=8.5, color=C_DARK, va="center")

    _save(fig, "fig1_vpc_architecture")


# ---------------------------------------------------------------------------
# Figure 2: SDN control plane vs data plane
# ---------------------------------------------------------------------------
def fig2_sdn_planes() -> None:
    """SDN control/data plane separation with N/S APIs."""
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.0); ax.axis("off")
    ax.set_title("SDN: separating the control plane (decisions) from the data plane (forwarding)",
                 fontsize=13, fontweight="bold", pad=12)

    # Application layer (top)
    _box(ax, 1.0, 7.0, 2.2, 0.8, "Routing app", C_PURPLE, fontsize=10)
    _box(ax, 4.0, 7.0, 2.2, 0.8, "Firewall app", C_PURPLE, fontsize=10)
    _box(ax, 7.0, 7.0, 2.2, 0.8, "QoS app", C_PURPLE, fontsize=10)
    _box(ax, 10.0, 7.0, 1.8, 0.8, "Monitoring", C_PURPLE, fontsize=10, alpha=0.7)
    ax.text(0.05, 7.4, "Apps", fontsize=10, color=C_GRAY, fontweight="bold",
            style="italic")

    # Northbound API band
    ax.add_patch(Rectangle((0.6, 5.85), 11.0, 0.4, facecolor=C_AMBER,
                           alpha=0.25, edgecolor=C_AMBER))
    ax.text(6.1, 6.05, "Northbound API  (REST / gRPC)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C_AMBER)

    # Controller (control plane)
    _box(ax, 2.5, 4.0, 7.0, 1.4,
         "SDN Controller\n(topology, path computation, policy)",
         C_BLUE, fontsize=11)
    ax.text(0.05, 4.7, "Control\nplane", fontsize=10, color=C_GRAY,
            fontweight="bold", style="italic")

    # Southbound band
    ax.add_patch(Rectangle((0.6, 3.0), 11.0, 0.4, facecolor=C_AMBER,
                           alpha=0.25, edgecolor=C_AMBER))
    ax.text(6.1, 3.2, "Southbound API  (OpenFlow / NETCONF / P4Runtime)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C_AMBER)

    # Data plane switches
    for i, x in enumerate([0.8, 3.2, 5.6, 8.0, 10.4]):
        _box(ax, x, 1.2, 1.6, 1.4,
             f"Switch\n(flow table {i+1})", C_GREEN, fontsize=9)

    ax.text(0.05, 1.9, "Data\nplane", fontsize=10, color=C_GRAY,
            fontweight="bold", style="italic")

    # Arrows
    for x in [1.6, 4.0, 6.4, 8.8, 11.2]:
        _arrow(ax, (x, 4.0), (x, 2.6), color=C_DARK, lw=1.0)
        _arrow(ax, (x, 5.4), (x, 6.25), color=C_DARK, lw=1.0)

    # Forwarding traffic at the bottom
    _arrow(ax, (1.6, 1.2), (3.2, 1.2), color=C_PURPLE, lw=1.5)
    _arrow(ax, (4.8, 1.2), (5.6, 1.2), color=C_PURPLE, lw=1.5)
    _arrow(ax, (7.2, 1.2), (8.0, 1.2), color=C_PURPLE, lw=1.5)
    _arrow(ax, (9.6, 1.2), (10.4, 1.2), color=C_PURPLE, lw=1.5)
    ax.text(6.0, 0.6,
            "Packets traverse the data plane at line rate; "
            "controller intervenes only on flow setup or policy change.",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    _save(fig, "fig2_sdn_planes")


# ---------------------------------------------------------------------------
# Figure 3: L4 vs L7 load balancers
# ---------------------------------------------------------------------------
def fig3_lb_l4_l7() -> None:
    """Side-by-side feature comparison + simple request flow."""
    fig = plt.figure(figsize=(13.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # ----- Left: capability bars -----
    ax = fig.add_subplot(gs[0, 0])
    dims = [
        "Throughput",
        "Latency\n(lower=better)",
        "Routing\nflexibility",
        "TLS\ntermination",
        "Header /\npath rules",
        "WebSocket\n+ HTTP/2",
    ]
    nlb = [5, 5, 1, 2, 1, 1]   # L4
    alb = [3, 3, 5, 5, 5, 5]   # L7
    x = np.arange(len(dims)); w = 0.36
    ax.bar(x - w/2, nlb, w, label="NLB  (L4)",  color=C_BLUE, edgecolor="white")
    ax.bar(x + w/2, alb, w, label="ALB  (L7)",  color=C_PURPLE, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(dims, fontsize=9)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("Capability  (higher is better)", fontsize=10)
    ax.set_title("L4 wins on raw speed, L7 wins on intelligent routing",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.legend(loc="upper center", frameon=True, framealpha=0.95,
              ncol=2)
    ax.set_axisbelow(True)

    # ----- Right: request flow diagram -----
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Where each LB sits", fontsize=11.5, fontweight="bold", pad=8)

    # Client
    _box(ax, 4.0, 6.8, 2.0, 0.7, "Client", C_DARK, fontsize=10)

    # NLB path
    _box(ax, 0.5, 5.0, 3.5, 0.9, "NLB  (L4)\nTCP / UDP, IP+port",
         C_BLUE, fontsize=9)
    # ALB path
    _box(ax, 6.0, 5.0, 3.5, 0.9, "ALB  (L7)\nHTTP/HTTPS, path/host",
         C_PURPLE, fontsize=9)

    _arrow(ax, (4.6, 6.8), (2.3, 5.9), color=C_BLUE)
    _arrow(ax, (5.4, 6.8), (7.7, 5.9), color=C_PURPLE)

    # Backends
    for i, x in enumerate([0.4, 2.2, 4.0, 5.8, 7.6, 9.0]):
        _box(ax, x, 2.6, 1.4, 0.9, f"Pod {i+1}", C_GREEN, fontsize=9)

    # NLB to backends
    for x in [0.4, 2.2, 4.0]:
        _arrow(ax, (2.3, 5.0), (x + 0.7, 3.5), color=C_BLUE, lw=0.9)
    # ALB to backends
    for x in [4.0, 5.8, 7.6, 9.0]:
        _arrow(ax, (7.7, 5.0), (x + 0.7, 3.5), color=C_PURPLE, lw=0.9)

    ax.text(5.0, 1.7,
            "Use NLB for gaming / IoT / TCP services.\n"
            "Use ALB for web apps, APIs, microservices, gRPC, gRPC-Web.",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    _save(fig, "fig3_lb_l4_l7")


# ---------------------------------------------------------------------------
# Figure 4: CDN edge distribution
# ---------------------------------------------------------------------------
def fig4_cdn_edge() -> None:
    """Schematic world map: edges around continents serving local clients."""
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.set_title("CDN: edge PoPs serve local clients; only cache misses hit the origin",
                 fontsize=13, fontweight="bold", pad=12)

    # Stylised continent blobs (rectangles)
    continents = [
        (0.6,  4.0, 3.0, 1.6, "North America"),
        (4.0,  4.5, 2.6, 1.4, "Europe"),
        (6.8,  4.0, 3.4, 1.8, "Asia"),
        (1.0,  1.6, 2.4, 1.4, "South America"),
        (4.4,  1.4, 2.0, 1.2, "Africa"),
        (10.6, 1.6, 2.6, 1.4, "Oceania"),
    ]
    for x, y, w, h, name in continents:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor=C_BG, edgecolor=C_GRAY,
                                    linewidth=1.2))
        ax.text(x + w/2, y + h - 0.25, name, fontsize=9, color=C_GRAY,
                ha="center", fontweight="bold", style="italic")

    # PoPs (edge nodes) on each continent
    pops = [
        (1.4, 4.4), (2.6, 4.5), (3.2, 5.0),     # NA
        (4.4, 5.2), (5.4, 5.0), (6.2, 5.2),     # EU
        (7.6, 4.6), (8.4, 4.8), (9.2, 5.2), (9.8, 4.6),  # Asia
        (1.6, 2.0), (2.6, 2.4),                 # SA
        (4.8, 1.8), (6.0, 1.8),                 # AF
        (11.2, 2.0), (12.4, 2.4),               # Oceania
    ]
    for x, y in pops:
        ax.add_patch(Circle((x, y), 0.18, facecolor=C_GREEN, edgecolor="white",
                            linewidth=1.5, zorder=5))

    # Origin (top-right)
    _box(ax, 12.0, 6.2, 1.8, 0.9, "Origin\n(us-east-1)", C_PURPLE, fontsize=9)

    # Clients (small dots near PoPs)
    rng = np.random.default_rng(1)
    for x, y in pops:
        for _ in range(3):
            cx = x + rng.uniform(-0.35, 0.35)
            cy = y + rng.uniform(-0.35, 0.35)
            ax.add_patch(Circle((cx, cy), 0.06, facecolor=C_BLUE,
                                edgecolor="none", alpha=0.8, zorder=6))

    # Highlight: cache HIT path (Tokyo -> nearby PoP)
    _arrow(ax, (9.5, 5.0), (9.2, 5.2), color=C_GREEN, lw=1.6)
    ax.text(9.5, 4.55, "20 ms  (cache HIT)", fontsize=9, color=C_GREEN,
            fontweight="bold")

    # Cache MISS path: PoP -> origin -> back
    _arrow(ax, (9.2, 5.2), (12.0, 6.5), color=C_AMBER, lw=1.4,
           style="->")
    ax.text(11.0, 5.9, "miss -> 220 ms", fontsize=9, color=C_AMBER,
            fontweight="bold", style="italic")

    # Footer
    ax.text(7.0, 0.3,
            "Static assets: cache 90%+ at the edge.   "
            "Dynamic API responses: tiered caching + short TTL.",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")

    # Legend
    ax.add_patch(Circle((0.5, 6.95), 0.18, facecolor=C_GREEN,
                        edgecolor="white", linewidth=1.5))
    ax.text(0.78, 6.95, "Edge PoP", fontsize=9, color=C_DARK, va="center")
    ax.add_patch(Circle((2.2, 6.95), 0.07, facecolor=C_BLUE))
    ax.text(2.4, 6.95, "Client", fontsize=9, color=C_DARK, va="center")
    _box(ax, 3.4, 6.78, 0.4, 0.32, "", C_PURPLE, fontsize=8)
    ax.text(3.95, 6.95, "Origin", fontsize=9, color=C_DARK, va="center")

    _save(fig, "fig4_cdn_edge")


# ---------------------------------------------------------------------------
# Figure 5: BGP multi-region routing
# ---------------------------------------------------------------------------
def fig5_bgp_multi_region() -> None:
    """Three regions interconnected via BGP, showing AS-paths and ECMP."""
    fig, ax = plt.subplots(figsize=(13, 7.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("BGP Across Regions: AS-paths, ECMP, and failover for global reach",
                 fontsize=13, fontweight="bold", pad=12)

    # Three regional ASes
    regions = [
        (1.5,  4.0, "us-east-1\nAS 64500"),
        (6.5,  5.5, "eu-west-1\nAS 64501"),
        (11.0, 4.0, "ap-northeast-1\nAS 64502"),
    ]
    centers = []
    for x, y, label in regions:
        ax.add_patch(Circle((x, y), 1.2, facecolor=C_BLUE, alpha=0.18,
                            edgecolor=C_BLUE, linewidth=2))
        ax.text(x, y + 0.1, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=C_BLUE)
        centers.append((x, y))

    # Transit AS in the middle
    _box(ax, 5.7, 1.8, 2.6, 0.9, "Transit  AS 64600\n(Tier-1 backbone)",
         C_PURPLE, fontsize=9.5)

    # BGP peerings (lines + arrowheads)
    peerings = [
        (centers[0], centers[1], C_BLUE),
        (centers[1], centers[2], C_BLUE),
        (centers[0], centers[2], C_GRAY),    # longer / backup
        (centers[0], (7.0, 2.7),  C_PURPLE),
        (centers[1], (7.0, 2.7),  C_PURPLE),
        (centers[2], (7.0, 2.7),  C_PURPLE),
    ]
    for p1, p2, c in peerings:
        ls = ":" if c == C_GRAY else "-"
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-",
                                     color=c, linewidth=1.6,
                                     linestyle=ls,
                                     connectionstyle="arc3,rad=0.05"))

    # Client + flow
    _box(ax, 0.2, 7.0, 1.8, 0.7, "Client (NA)", C_DARK, fontsize=9)
    _box(ax, 12.0, 7.0, 1.8, 0.7, "Service in AP", C_GREEN, fontsize=9)

    # Two candidate AS paths
    ax.add_patch(FancyArrowPatch((1.1, 7.0), (1.5, 5.2),
                                 arrowstyle="->", mutation_scale=12,
                                 color=C_GREEN, linewidth=1.8))
    ax.add_patch(FancyArrowPatch((1.5, 4.0), (6.5, 5.5),
                                 arrowstyle="->", mutation_scale=12,
                                 color=C_GREEN, linewidth=1.8))
    ax.add_patch(FancyArrowPatch((6.5, 5.5), (11.0, 4.0),
                                 arrowstyle="->", mutation_scale=12,
                                 color=C_GREEN, linewidth=1.8))
    ax.add_patch(FancyArrowPatch((11.0, 4.0), (12.9, 7.0),
                                 arrowstyle="->", mutation_scale=12,
                                 color=C_GREEN, linewidth=1.8))

    ax.text(6.5, 6.55, "Preferred AS-path:  64500  64501  64502   (3 hops)",
            ha="center", fontsize=10, fontweight="bold", color=C_GREEN)
    ax.text(10.0, 0.45,
            "Backup path via Transit (AS 64600) takes over on link or region failure.\n"
            "ECMP spreads flows across equal-cost neighbours.",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")

    # BGP attribute callout box (bottom-left, clear of routing path)
    callout = ("BGP route selection  (simplified):\n"
               "  1. highest LOCAL_PREF\n"
               "  2. shortest AS_PATH\n"
               "  3. lowest MED\n"
               "  4. eBGP > iBGP\n"
               "  5. lowest IGP cost to next-hop")
    ax.text(0.3, 0.1, callout, fontsize=9, color=C_DARK,
            family="monospace", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_BG,
                      edgecolor=C_GRAY))

    _save(fig, "fig5_bgp_multi_region")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating cloud-computing/04-networking figures…")
    fig1_vpc_architecture()
    fig2_sdn_planes()
    fig3_lb_l4_l7()
    fig4_cdn_edge()
    fig5_bgp_multi_region()
    print("Done.")


if __name__ == "__main__":
    main()
