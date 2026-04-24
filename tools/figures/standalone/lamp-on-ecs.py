"""
Figure generation for the standalone tutorial:
"LAMP Stack on Alibaba Cloud ECS".

Generates 5 figures used in BOTH the EN and ZH versions of the article.
Each figure communicates a single concrete idea, in editorial print style.

Figures:
    fig1_lamp_architecture        Four-layer stack: Linux base, Apache web
                                  server, PHP runtime, MySQL data store -- with
                                  the responsibility each layer owns.
    fig2_aliyun_ecs_overview      ECS instance anatomy: regions/zones, instance
                                  family choice, vCPU/RAM/disk/bandwidth, plus
                                  the public-IP and security group surface.
    fig3_request_flow             A real HTTP request walked left-to-right:
                                  browser -> security group -> Apache -> PHP
                                  -> MySQL and back, with the failure points
                                  annotated at each hop.
    fig4_security_setup           Defence in depth: cloud security group +
                                  OS firewall + TLS + MySQL hardening, drawn
                                  as concentric rings with the ports exposed
                                  and the threats each ring blocks.
    fig5_deployment_topology      Single-instance LAMP vs three-tier (web /
                                  app / db separated), with cost, complexity
                                  and failure-domain trade-offs.

Usage:
    python3 scripts/figures/standalone/lamp-on-ecs.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (

    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]     # primary
C_PURPLE = COLORS["accent"]   # secondary
C_GREEN = COLORS["success"]    # accent / good
C_AMBER = COLORS["warning"]    # warning / highlight
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG_SOFT = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "lamp-on-ecs"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "standalone" / "lamp-与阿里云服务器详解"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _no_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def _box(ax, x, y, w, h, color, label, sub=None, *, text_color="white",
         alpha=0.95, fontsize=12, sub_fontsize=8.5):
    """Rounded box with a centered label and optional subtitle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=0, facecolor=color, alpha=alpha,
    )
    ax.add_patch(box)
    if sub:
        ax.text(x + w / 2, y + h / 2 + 0.18, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=text_color)
        ax.text(x + w / 2, y + h / 2 - 0.22, sub,
                ha="center", va="center", fontsize=sub_fontsize,
                color=text_color, alpha=0.95)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=text_color)


# ---------------------------------------------------------------------------
# Figure 1 -- LAMP architecture: four-layer stack
# ---------------------------------------------------------------------------
def fig1_lamp_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    _no_axis(ax)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 7.5)

    # Title
    ax.text(5.75, 7.15, "The LAMP Stack",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(5.75, 6.78, "Four layers, each one depends on the layer below",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # The stack (bottom to top). Wider at the bottom -- foundation idea.
    layers = [
        # (label, subtitle, owns_text, color, y0, h, width)
        ("Linux",  "Ubuntu / CentOS / Alibaba Cloud Linux",
         "Process scheduling, file system, networking, users",
         C_DARK,   0.6, 1.05, 8.4),
        ("Apache", "httpd 2.4 -- prefork / event MPM",
         "Listens on :80/:443, parses HTTP, picks the handler",
         C_BLUE,   1.75, 1.05, 7.6),
        ("PHP",    "mod_php or php-fpm via FastCGI",
         "Executes .php scripts, opens DB connection, builds response",
         C_PURPLE, 2.90, 1.05, 6.8),
        ("MySQL",  "MySQL 8 / MariaDB -- InnoDB engine",
         "Persistent storage, transactions, indexes, query plan",
         C_GREEN,  4.05, 1.05, 6.0),
    ]
    cx = 5.75
    for label, sub, owns, color, y0, h, w in layers:
        x0 = cx - w / 2
        rect = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=0, facecolor=color, alpha=0.95,
        )
        ax.add_patch(rect)
        ax.text(x0 + 0.32, y0 + h / 2 + 0.18, label,
                ha="left", va="center", fontsize=15, fontweight="bold", color="white")
        ax.text(x0 + 0.32, y0 + h / 2 - 0.22, sub,
                ha="left", va="center", fontsize=9, color="white", alpha=0.95)
        # Owns text on the right, inside the box
        ax.text(x0 + w - 0.32, y0 + h / 2, owns,
                ha="right", va="center", fontsize=8.5, color="white",
                alpha=0.95, style="italic")

    # Top: a request enters
    ax.annotate("", xy=(cx, 5.6), xytext=(cx, 6.3),
                arrowprops=dict(arrowstyle="-|>", color=C_AMBER, lw=2.2))
    ax.text(cx, 6.45, "HTTP request from the browser",
            ha="center", va="center", fontsize=10, color=C_DARK, fontweight="bold")

    # Side annotation -- abstraction direction
    ax.annotate("", xy=(10.7, 5.10), xytext=(10.7, 0.6),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))
    ax.text(10.95, 2.85, "higher level of abstraction",
            rotation=-90, ha="center", va="center",
            fontsize=10, color=C_DARK, fontweight="bold")

    ax.text(0.55, 0.22, "If a layer below is broken, every layer above breaks with it.",
            ha="left", va="center", fontsize=9.5, style="italic", color=C_GRAY)

    plt.tight_layout()
    _save(fig, "fig1_lamp_architecture")


# ---------------------------------------------------------------------------
# Figure 2 -- Alibaba Cloud ECS overview
# ---------------------------------------------------------------------------
def fig2_aliyun_ecs_overview() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    _no_axis(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)

    ax.text(6, 7.15, "Anatomy of an Alibaba Cloud ECS Instance",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(6, 6.78,
            "What you actually pick when you click Create",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # Outer region container
    region = FancyBboxPatch(
        (0.5, 0.6), 11, 6.0,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.2, edgecolor=C_GRAY, facecolor=C_BG_SOFT,
    )
    ax.add_patch(region)
    ax.text(0.85, 6.32, "Region: cn-hangzhou",
            ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=C_DARK)
    ax.text(11.15, 6.32, "Billing: pay-as-you-go / subscription / spot",
            ha="right", va="center", fontsize=9.5, color=C_GRAY)

    # Three AZs as columns
    az_x = [1.0, 4.7, 8.4]
    az_w = 3.1
    for i, x in enumerate(az_x):
        az_box = FancyBboxPatch(
            (x, 1.0), az_w, 5.0,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.0, edgecolor=C_LIGHT, facecolor="white",
        )
        ax.add_patch(az_box)
        ax.text(x + az_w / 2, 5.78, f"Availability Zone {chr(ord('A') + i)}",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=C_DARK)

    # Put a single ECS instance in AZ-B with full anatomy
    ix, iy, iw, ih = 4.85, 1.25, az_w - 0.3, 4.35
    inst = FancyBboxPatch(
        (ix, iy), iw, ih,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=C_BLUE, facecolor="white",
    )
    ax.add_patch(inst)
    ax.text(ix + iw / 2, iy + ih - 0.32, "ECS instance ecs-g7.large",
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=C_BLUE)

    # Inside the instance: vCPU/RAM/Disk/NIC bars
    rows = [
        ("vCPU",   "2 cores -- Xeon Platinum",          C_BLUE),
        ("Memory", "8 GiB DDR4",                        C_PURPLE),
        ("Disk",   "ESSD PL1 40 GiB system + data",     C_GREEN),
        ("NIC",    "Internal VPC + 5 Mbps public IP",   C_AMBER),
    ]
    for k, (label, sub, color) in enumerate(rows):
        ry = iy + ih - 0.95 - k * 0.78
        bar = FancyBboxPatch(
            (ix + 0.18, ry), iw - 0.36, 0.6,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=0, facecolor=color, alpha=0.92,
        )
        ax.add_patch(bar)
        ax.text(ix + 0.32, ry + 0.30, label,
                ha="left", va="center", fontsize=9.5,
                fontweight="bold", color="white")
        ax.text(ix + iw - 0.32, ry + 0.30, sub,
                ha="right", va="center", fontsize=8.5, color="white", alpha=0.95)

    # AZ-A: instance family choice
    ax.text(az_x[0] + az_w / 2, 5.32, "Instance family",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C_DARK)
    fam_rows = [
        ("g7  general",  "balanced, web tier",      C_BLUE),
        ("c7  compute",  "API, app server",         C_PURPLE),
        ("r7  memory",   "cache, in-memory DB",     C_GREEN),
        ("hfg7 high-clock", "low-latency tasks",    C_AMBER),
        ("ecs.t6  burst", "low-traffic site",       C_GRAY),
    ]
    for k, (label, sub, color) in enumerate(fam_rows):
        fy = 4.85 - k * 0.62
        ax.add_patch(Rectangle((az_x[0] + 0.18, fy), 0.18, 0.40,
                               facecolor=color, edgecolor="none"))
        ax.text(az_x[0] + 0.45, fy + 0.28, label,
                ha="left", va="center", fontsize=8.6,
                fontweight="bold", color=C_DARK)
        ax.text(az_x[0] + 0.45, fy + 0.10, sub,
                ha="left", va="center", fontsize=7.8, color=C_GRAY)

    # AZ-C: networking surface
    ax.text(az_x[2] + az_w / 2, 5.32, "What's exposed",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C_DARK)
    net_rows = [
        ("Public IP / EIP",       "8.x.x.x reachable from the Internet"),
        ("Security Group",        "stateful firewall, port-by-port allow"),
        ("VPC + vSwitch",         "private 192.168.x.x for east-west"),
        ("Console + SSH key",     "operator access to the OS"),
        ("Cloud Monitor agent",   "CPU / mem / disk / TCP metrics"),
    ]
    for k, (label, sub) in enumerate(net_rows):
        ny = 4.85 - k * 0.62
        ax.add_patch(Circle((az_x[2] + 0.32, ny + 0.24), 0.10,
                            facecolor=C_BLUE, edgecolor="none"))
        ax.text(az_x[2] + 0.55, ny + 0.32, label,
                ha="left", va="center", fontsize=8.6,
                fontweight="bold", color=C_DARK)
        ax.text(az_x[2] + 0.55, ny + 0.12, sub,
                ha="left", va="center", fontsize=7.6, color=C_GRAY)

    ax.text(6, 0.32,
            "Region = geography. Zone = blast radius. Instance family = $/perf shape.",
            ha="center", va="center", fontsize=9.5, style="italic", color=C_GRAY)

    plt.tight_layout()
    _save(fig, "fig2_aliyun_ecs_overview")


# ---------------------------------------------------------------------------
# Figure 3 -- HTTP request flow through the stack
# ---------------------------------------------------------------------------
def fig3_request_flow() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)

    ax.text(6.5, 6.18, "How a Request Travels Through the Stack",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(6.5, 5.82,
            "Each hop has its own way of failing -- knowing where helps you debug fast",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # Six hops left to right, two rows -- request on top, response on bottom
    hops = [
        ("Browser",         "GET /index.php",              C_GRAY),
        ("Security\ngroup", "allow tcp:80 from 0.0.0.0/0", C_AMBER),
        ("Apache",          "VirtualHost match\n+ DocumentRoot", C_BLUE),
        ("PHP",             "mod_php / php-fpm\nexecute .php",   C_PURPLE),
        ("MySQL",           "SELECT ... FROM users",       C_GREEN),
        ("Response",        "HTML returned",               C_DARK),
    ]
    n = len(hops)
    margin = 0.5
    avail = 13 - 2 * margin
    bw = (avail - (n - 1) * 0.35) / n
    by = 3.4
    bh = 1.55
    centers = []
    for i, (label, sub, color) in enumerate(hops):
        x0 = margin + i * (bw + 0.35)
        cx = x0 + bw / 2
        centers.append(cx)
        box = FancyBboxPatch(
            (x0, by), bw, bh,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0, facecolor=color, alpha=0.95,
        )
        ax.add_patch(box)
        ax.text(cx, by + bh / 2 + 0.30, label,
                ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="white")
        ax.text(cx, by + bh / 2 - 0.32, sub,
                ha="center", va="center", fontsize=8.4,
                color="white", alpha=0.95)

    # Forward arrows
    for i in range(n - 1):
        x_a = centers[i] + bw / 2 + 0.02
        x_b = centers[i + 1] - bw / 2 - 0.02
        ax.annotate("", xy=(x_b, by + bh - 0.25),
                    xytext=(x_a, by + bh - 0.25),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=C_DARK, lw=1.6, alpha=0.7))
    # Return arrow (one big sweep below)
    ax.annotate("", xy=(centers[0], by + 0.25),
                xytext=(centers[-1], by + 0.25),
                arrowprops=dict(arrowstyle="-|>",
                                color=C_GREEN, lw=1.6, alpha=0.7,
                                connectionstyle="arc3,rad=0.0"))
    ax.text((centers[0] + centers[-1]) / 2, by - 0.25,
            "response travels back along the same path",
            ha="center", va="center", fontsize=9, color=C_GREEN, style="italic")

    # Failure annotations under each hop
    failures = [
        "DNS / TLS",
        "Connection\nrefused",
        "403 / 404\nwrong DocRoot",
        "PHP source\nleaks as text",
        "Access denied\nor timeout",
        "5xx / blank\npage",
    ]
    for cx, msg in zip(centers, failures):
        ax.text(cx, 1.10, msg,
                ha="center", va="center", fontsize=8.4,
                color=C_RED, fontweight="bold")
    ax.text(0.5, 1.95, "Common failure", ha="left", va="center",
            fontsize=9.0, color=C_RED, fontweight="bold")
    # Divider line for failure row
    ax.plot([0.4, 12.6], [1.65, 1.65], color=C_LIGHT, lw=1)

    ax.text(6.5, 0.35,
            "Security group lives in the cloud console. Firewall and Apache live on the OS. They both have to agree.",
            ha="center", va="center", fontsize=9.5, style="italic", color=C_GRAY)

    plt.tight_layout()
    _save(fig, "fig3_request_flow")


# ---------------------------------------------------------------------------
# Figure 4 -- Defence in depth (security setup)
# ---------------------------------------------------------------------------
def fig4_security_setup() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    _no_axis(ax)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 7.6)

    ax.text(5.75, 7.30, "Defence in Depth for a Public LAMP Server",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(5.75, 6.95,
            "Five rings -- if one fails, the next one still has to be picked",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # Concentric rings around (4.0, 3.5)
    cx, cy = 4.0, 3.5
    rings = [
        # (radius, color, label, sub)
        (3.2, C_GRAY,   "Internet",          "untrusted, anything can hit you"),
        (2.7, C_AMBER,  "Security group",    "stateful firewall in the cloud"),
        (2.2, C_BLUE,   "OS firewall (ufw)", "second line of defence"),
        (1.7, C_PURPLE, "TLS / HTTPS",       "encrypts the wire"),
        (1.2, C_GREEN,  "App + DB hardening","strong passwords, least privilege"),
    ]
    for r, color, _label, _sub in rings:
        ax.add_patch(Circle((cx, cy), r,
                            facecolor=color, edgecolor="white",
                            linewidth=2.2, alpha=0.85))
    # Centre target
    ax.add_patch(Circle((cx, cy), 0.55,
                        facecolor=C_DARK, edgecolor="white", linewidth=2))
    ax.text(cx, cy + 0.07, "MySQL",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")
    ax.text(cx, cy - 0.18, "your data",
            ha="center", va="center", fontsize=8.0,
            color="white", alpha=0.9)

    # Labels to the right side
    label_x = 8.0
    label_y_start = 6.2
    label_dy = 0.95
    for k, (_r, color, label, sub) in enumerate(rings):
        ly = label_y_start - k * label_dy
        ax.add_patch(Rectangle((label_x - 0.18, ly - 0.25), 0.18, 0.5,
                               facecolor=color, edgecolor="none"))
        ax.text(label_x + 0.10, ly + 0.10, label,
                ha="left", va="center", fontsize=10,
                fontweight="bold", color=C_DARK)
        ax.text(label_x + 0.10, ly - 0.18, sub,
                ha="left", va="center", fontsize=8.4, color=C_GRAY)

    # Bottom -- which port on which ring
    ax.text(0.5, 0.95, "Port surface (smallest is best)",
            ha="left", va="center", fontsize=10,
            fontweight="bold", color=C_DARK)
    ports = [
        ("22",   "SSH -- key auth, ideally restricted by source CIDR"),
        ("80",   "HTTP -- only as a 301 redirect to HTTPS"),
        ("443",  "HTTPS -- the only port the public ever talks to"),
        ("3306", "MySQL -- never on 0.0.0.0/0, use SSH tunnel"),
    ]
    for k, (p, sub) in enumerate(ports):
        py = 0.55 - k * 0.0  # single line layout, use x-spacing
    # Single horizontal layout
    px = 0.5
    py = 0.40
    for k, (p, sub) in enumerate(ports):
        ax.add_patch(FancyBboxPatch(
            (px + k * 2.75, py), 0.55, 0.40,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=0,
            facecolor=[C_BLUE, C_PURPLE, C_GREEN, C_RED][k],
            alpha=0.95,
        ))
        ax.text(px + k * 2.75 + 0.275, py + 0.20, p,
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")
        ax.text(px + k * 2.75 + 0.65, py + 0.20, sub,
                ha="left", va="center", fontsize=7.8, color=C_DARK)

    plt.tight_layout()
    _save(fig, "fig4_security_setup")


# ---------------------------------------------------------------------------
# Figure 5 -- Single instance vs three-tier deployment
# ---------------------------------------------------------------------------
def fig5_deployment_topology() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.6))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.7)

    ax.text(6.5, 6.40, "Two Topologies for a LAMP Site",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(6.5, 6.05,
            "Same software, different shape -- pick by traffic, not by taste",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # ------ Left: single instance ------
    pane_x, pane_w = 0.4, 5.8
    pane = FancyBboxPatch(
        (pane_x, 0.6), pane_w, 5.0,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.2, edgecolor=C_LIGHT, facecolor=C_BG_SOFT,
    )
    ax.add_patch(pane)
    ax.text(pane_x + pane_w / 2, 5.30, "All-in-one ECS",
            ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=C_DARK)
    ax.text(pane_x + pane_w / 2, 5.00, "Apache + PHP + MySQL on one box",
            ha="center", va="center", fontsize=9.5, color=C_GRAY)

    # Internet -> ECS
    ax.add_patch(FancyBboxPatch(
        (pane_x + 0.5, 3.8), 1.4, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0, facecolor=C_GRAY, alpha=0.85))
    ax.text(pane_x + 1.2, 4.25, "Internet",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")

    box_x = pane_x + 2.5
    ax.add_patch(FancyBboxPatch(
        (box_x, 1.4), 2.7, 3.4,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=C_BLUE, facecolor="white"))
    ax.text(box_x + 1.35, 4.55, "ECS g7.large",
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=C_BLUE)
    inner = [("Apache :80/443", C_BLUE),
             ("PHP runtime", C_PURPLE),
             ("MySQL :3306 localhost", C_GREEN)]
    for k, (lab, col) in enumerate(inner):
        ry = 4.0 - k * 0.78
        ax.add_patch(FancyBboxPatch(
            (box_x + 0.2, ry - 0.30), 2.3, 0.6,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=0, facecolor=col, alpha=0.92))
        ax.text(box_x + 1.35, ry, lab,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white")

    ax.annotate("", xy=(box_x - 0.05, 4.25), xytext=(pane_x + 1.95, 4.25),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.6))

    # Pros/cons
    ax.text(pane_x + 0.35, 1.10,
            "+ cheapest, simplest to run\n"
            "+ low latency: PHP -> DB on localhost\n"
            "- one failure takes the whole site down\n"
            "- DB and web compete for CPU and RAM",
            ha="left", va="top", fontsize=8.7, color=C_DARK)

    # ------ Right: three-tier ------
    pane2_x = 6.7
    pane2_w = 5.9
    pane2 = FancyBboxPatch(
        (pane2_x, 0.6), pane2_w, 5.0,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.2, edgecolor=C_LIGHT, facecolor=C_BG_SOFT,
    )
    ax.add_patch(pane2)
    ax.text(pane2_x + pane2_w / 2, 5.30, "Three-tier on Aliyun",
            ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=C_DARK)
    ax.text(pane2_x + pane2_w / 2, 5.00, "SLB -> ECS web fleet -> RDS for MySQL",
            ha="center", va="center", fontsize=9.5, color=C_GRAY)

    # SLB
    slb_x = pane2_x + 0.3
    ax.add_patch(FancyBboxPatch(
        (slb_x, 3.1), 1.0, 1.6,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0, facecolor=C_AMBER, alpha=0.92))
    ax.text(slb_x + 0.5, 3.90, "SLB\n:443",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="white")

    # Two web ECS
    web_x = pane2_x + 2.0
    for k in range(2):
        wy = 3.7 - k * 1.45
        ax.add_patch(FancyBboxPatch(
            (web_x, wy), 1.7, 1.05,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.2, edgecolor=C_BLUE, facecolor="white"))
        ax.text(web_x + 0.85, wy + 0.72, f"Web {k+1}",
                ha="center", va="center", fontsize=9.8,
                fontweight="bold", color=C_BLUE)
        ax.text(web_x + 0.85, wy + 0.32, "Apache + PHP",
                ha="center", va="center", fontsize=8.4, color=C_DARK)

    # RDS
    rds_x = pane2_x + 4.4
    ax.add_patch(FancyBboxPatch(
        (rds_x, 2.4), 1.3, 2.2,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0, facecolor=C_GREEN, alpha=0.92))
    ax.text(rds_x + 0.65, 3.80, "RDS for\nMySQL",
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="white")
    ax.text(rds_x + 0.65, 2.95, "primary +\nstandby",
            ha="center", va="center", fontsize=8.4,
            color="white", alpha=0.95)

    # Arrows
    ax.annotate("", xy=(web_x - 0.05, 4.20), xytext=(slb_x + 1.05, 4.20),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.4))
    ax.annotate("", xy=(web_x - 0.05, 2.80), xytext=(slb_x + 1.05, 3.20),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.4))
    ax.annotate("", xy=(rds_x - 0.05, 4.20), xytext=(web_x + 1.75, 4.20),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.4))
    ax.annotate("", xy=(rds_x - 0.05, 3.20), xytext=(web_x + 1.75, 2.80),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.4))

    ax.text(pane2_x + 0.35, 1.10,
            "+ web tier scales horizontally behind the SLB\n"
            "+ DB is managed: backups, failover, patching\n"
            "- 3-5x the cost of a single ECS\n"
            "- extra round-trip web -> RDS over VPC",
            ha="left", va="top", fontsize=8.7, color=C_DARK)

    plt.tight_layout()
    _save(fig, "fig5_deployment_topology")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating LAMP-on-ECS figures ...")
    fig1_lamp_architecture()
    fig2_aliyun_ecs_overview()
    fig3_request_flow()
    fig4_security_setup()
    fig5_deployment_topology()
    print("Done.")


if __name__ == "__main__":
    main()
