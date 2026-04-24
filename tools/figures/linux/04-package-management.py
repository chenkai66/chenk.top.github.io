"""
Figure generation script for Linux Article 04: Package Management.

Generates 5 conceptual figures used in both EN and ZH versions of the
article. Each figure is rendered to BOTH article asset folders so the
markdown image references stay in sync across languages.

Figures:
    fig1_package_manager_comparison  Side-by-side comparison of the major
                                     Linux package toolchains (apt/dpkg,
                                     yum-dnf/rpm, pacman, zypper) showing
                                     low-level vs high-level layers, the
                                     package format, and the distros they
                                     ship on.
    fig2_dependency_resolution       A small dependency graph (nginx ->
                                     libssl, libpcre, zlib, ...) drawn
                                     with networkx, illustrating what a
                                     dependency solver actually has to do
                                     and why manual rpm/dpkg installs go
                                     wrong.
    fig3_package_lifecycle           The lifecycle a package goes through
                                     on a typical box: search -> install
                                     -> upgrade / hold / downgrade ->
                                     remove / purge -> autoremove. Drawn
                                     as a circular flow with the relevant
                                     apt and dnf commands on each edge.
    fig4_repository_structure        Anatomy of a Debian-style apt
                                     repository (Release / Packages /
                                     pool/) and the chain of trust from
                                     the signed Release file down to a
                                     .deb on disk.
    fig5_modern_alternatives         Comparison of Snap, Flatpak and
                                     AppImage against the classic distro
                                     package: sandboxing, dependency
                                     bundling, update model and best fit.

Usage:
    python3 scripts/figures/linux/04-package-management.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns  # noqa: F401  (registers the seaborn style we use)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

COLOR_BLUE = COLORS["primary"]
COLOR_PURPLE = COLORS["accent"]
COLOR_GREEN = COLORS["success"]
COLOR_AMBER = COLORS["warning"]
COLOR_GREY = COLORS["text2"]
COLOR_LIGHT = COLORS["grid"]
COLOR_RED = COLORS["danger"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linux" / "package-management"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "package-management"


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
# Figure 1: Package manager comparison
# ---------------------------------------------------------------------------

def fig1_package_manager_comparison() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    families = [
        {
            "title": "Debian family",
            "distros": "Debian / Ubuntu / Mint",
            "fmt": ".deb",
            "low": "dpkg",
            "high": "apt  /  apt-get",
            "color": COLOR_BLUE,
        },
        {
            "title": "Red Hat family",
            "distros": "RHEL / CentOS / Rocky / Fedora",
            "fmt": ".rpm",
            "low": "rpm",
            "high": "dnf  (yum on EL7)",
            "color": COLOR_RED,
        },
        {
            "title": "Arch family",
            "distros": "Arch / Manjaro / EndeavourOS",
            "fmt": ".pkg.tar.zst",
            "low": "pacman",
            "high": "pacman  /  yay (AUR)",
            "color": COLOR_PURPLE,
        },
        {
            "title": "SUSE family",
            "distros": "openSUSE / SLES",
            "fmt": ".rpm",
            "low": "rpm",
            "high": "zypper",
            "color": COLOR_GREEN,
        },
    ]

    n = len(families)
    margin_x = 0.5
    gap = 0.35
    col_w = (12 - 2 * margin_x - (n - 1) * gap) / n
    col_top = 6.7
    col_bot = 1.4

    for i, fam in enumerate(families):
        x = margin_x + i * (col_w + gap)
        # Family card
        _rounded_box(ax, x, col_bot, col_w, col_top - col_bot,
                     facecolor="white", edgecolor=fam["color"], lw=1.8)
        # Header strip
        _rounded_box(ax, x, col_top - 0.95, col_w, 0.95,
                     facecolor=fam["color"], edgecolor=fam["color"])
        ax.text(x + col_w / 2, col_top - 0.42, fam["title"],
                ha="center", va="center", color="white",
                fontsize=12.5, fontweight="bold")
        ax.text(x + col_w / 2, col_top - 0.78, fam["distros"],
                ha="center", va="center", color="white",
                fontsize=8.5, style="italic")

        # Two layers: high level above low level
        # High-level tool box
        _rounded_box(ax, x + 0.2, col_top - 2.5, col_w - 0.4, 0.95,
                     facecolor="white", edgecolor=fam["color"], lw=1.4)
        ax.text(x + col_w / 2, col_top - 1.85, "high-level",
                ha="center", va="center", fontsize=8, color=COLOR_GREY,
                style="italic")
        ax.text(x + col_w / 2, col_top - 2.25, fam["high"],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=fam["color"], family="monospace")

        # Arrow between
        arr = FancyArrowPatch((x + col_w / 2, col_top - 2.55),
                              (x + col_w / 2, col_top - 3.05),
                              arrowstyle="->", mutation_scale=14,
                              color=COLOR_GREY, lw=1.2)
        ax.add_patch(arr)
        ax.text(x + col_w / 2 + 0.15, col_top - 2.8, "calls",
                ha="left", va="center", fontsize=7.5, color=COLOR_GREY,
                style="italic")

        # Low-level tool box
        _rounded_box(ax, x + 0.2, col_top - 4.0, col_w - 0.4, 0.95,
                     facecolor="#f8fafc", edgecolor=COLOR_GREY, lw=1.2)
        ax.text(x + col_w / 2, col_top - 3.35, "low-level",
                ha="center", va="center", fontsize=8, color=COLOR_GREY,
                style="italic")
        ax.text(x + col_w / 2, col_top - 3.75, fam["low"],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=COLORS["text"], family="monospace")

        # Package format pill
        ax.text(x + col_w / 2, col_bot + 0.45,
                fam["fmt"],
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=fam["color"],
                          edgecolor=fam["color"]))

    # Bottom legend strip explaining low vs high
    ax.text(6, 0.65,
            "high-level tools resolve dependencies from configured repos    "
            "low-level tools install one package file, no dependency solving",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY,
            style="italic")

    ax.text(6, 7.55, "Mainstream Linux package toolchains",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=COLORS["text"])
    ax.text(6, 7.15,
            "Same job (search / install / upgrade / remove), different "
            "command names and package formats",
            ha="center", va="center", fontsize=10, color=COLOR_GREY,
            style="italic")

    _save(fig, "fig1_package_manager_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: Dependency resolution graph
# ---------------------------------------------------------------------------

def fig2_dependency_resolution() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.axis("off")

    G = nx.DiGraph()

    # Top-level package the user asked for
    top = "nginx"
    # Direct dependencies (level 1)
    direct = ["libssl3", "libpcre2-8", "zlib1g", "libc6"]
    # Transitive dependencies (level 2)
    transitive = {
        "libssl3":    ["libcrypto3"],
        "libpcre2-8": ["libc6"],
        "zlib1g":     ["libc6"],
    }

    G.add_node(top)
    for d in direct:
        G.add_edge(top, d)
    for parent, kids in transitive.items():
        for k in kids:
            G.add_edge(parent, k)

    # Manual layered layout so the picture reads top -> bottom
    pos = {
        top: (0.0, 2.0),
        "libssl3":    (-3.0, 1.0),
        "libpcre2-8": (-1.0, 1.0),
        "zlib1g":     ( 1.0, 1.0),
        "libc6":      ( 3.0, 1.0),
        "libcrypto3": (-3.0, 0.0),
    }

    # Draw edges (curved a little so they don't all overlap on libc6)
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        # curve depending on horizontal distance
        rad = 0.0 if abs(x1 - x0) < 0.05 else 0.12
        arr = FancyArrowPatch((x0, y0 - 0.18), (x1, y1 + 0.18),
                              arrowstyle="-|>", mutation_scale=14,
                              connectionstyle=f"arc3,rad={rad}",
                              color=COLOR_GREY, lw=1.2)
        ax.add_patch(arr)

    # Draw nodes as boxes coloured by role
    node_styles = {
        top:          (COLOR_BLUE,  "package you asked for"),
        "libssl3":    (COLOR_PURPLE, "shared library"),
        "libpcre2-8": (COLOR_PURPLE, "shared library"),
        "zlib1g":     (COLOR_PURPLE, "shared library"),
        "libc6":      (COLOR_AMBER, "core C library (shared)"),
        "libcrypto3": (COLOR_PURPLE, "shared library"),
    }
    for node, (x, y) in pos.items():
        color, _ = node_styles[node]
        # slightly bigger box for the top package
        w = 1.6 if node == top else 1.4
        h = 0.55 if node == top else 0.45
        _rounded_box(ax, x - w / 2, y - h / 2, w, h,
                     facecolor=color, edgecolor=color, alpha=0.95)
        ax.text(x, y, node, ha="center", va="center", color="white",
                fontsize=11 if node == top else 10,
                fontweight="bold", family="monospace")

    # Annotation: who is shared
    ax.annotate("shared by 3 deps -\ninstalled once",
                xy=(3.0, 1.0), xytext=(4.4, 1.6),
                fontsize=9, color=COLOR_AMBER, style="italic",
                arrowprops=dict(arrowstyle="->", color=COLOR_AMBER, lw=1.2))

    # Side panel: what the solver does
    ax.text(-5.6, 2.1,
            "What the dependency solver does",
            fontsize=11.5, fontweight="bold", color=COLORS["text"])
    bullets = [
        "1. read declared deps of nginx",
        "2. recurse into each dep",
        "3. unify versions (one libc6,",
        "   not three different ones)",
        "4. pick a feasible install order",
        "5. fail loudly if no solution",
        "   exists (conflict / missing)",
    ]
    for i, b in enumerate(bullets):
        ax.text(-5.6, 1.65 - i * 0.27, b,
                fontsize=9.2, color=COLOR_GREY, family="monospace")

    # Failure mode callout
    ax.text(0, -0.4,
            "dpkg -i nginx.deb        ->  fails: 'depends on libssl3 (>= 3.0) but it is not installable'\n"
            "apt install nginx        ->  resolves the whole graph above and installs it in order",
            ha="center", va="center", fontsize=9.5,
            family="monospace", color=COLORS["text"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7",
                      edgecolor=COLOR_AMBER, lw=1.0))

    ax.set_xlim(-6.2, 6.0)
    ax.set_ylim(-1.0, 3.0)

    ax.text(0, 2.78, "Dependency resolution: install one package, pull in a graph",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=COLORS["text"])

    _save(fig, "fig2_dependency_resolution.png")


# ---------------------------------------------------------------------------
# Figure 3: Package lifecycle
# ---------------------------------------------------------------------------

def fig3_package_lifecycle() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.axis("off")

    # 5 stages around a circle
    stages = [
        ("search",    "find candidate package"),
        ("install",   "fetch + resolve deps + unpack"),
        ("upgrade",   "newer version available"),
        ("hold",      "pin to current version"),
        ("remove",    "uninstall (purge to wipe config)"),
    ]
    n = len(stages)
    radius = 3.2
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)

    coords = []
    colors = [COLOR_GREY, COLOR_BLUE, COLOR_GREEN, COLOR_AMBER, COLOR_RED]

    for i, ((stage, sub), angle, color) in enumerate(zip(stages, angles, colors)):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        coords.append((x, y))
        _rounded_box(ax, x - 1.3, y - 0.55, 2.6, 1.1,
                     facecolor=color, edgecolor=color, alpha=0.95)
        ax.text(x, y + 0.18, stage, ha="center", va="center",
                color="white", fontsize=13, fontweight="bold")
        ax.text(x, y - 0.22, sub, ha="center", va="center",
                color="white", fontsize=8.5, style="italic")

    # Curved arrows between stages
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        arr = FancyArrowPatch((x0, y0), (x1, y1),
                              arrowstyle="-|>", mutation_scale=18,
                              connectionstyle="arc3,rad=0.18",
                              color=COLOR_GREY, lw=1.6)
        ax.add_patch(arr)

    # Center: the two command families on each edge
    edge_cmds = [
        ("apt search / dnf search",        (-2.4,  3.1)),
        ("apt install / dnf install",      ( 3.1,  1.5)),
        ("apt upgrade / dnf upgrade",      ( 2.6, -2.2)),
        ("apt-mark hold / dnf versionlock",(-2.6, -2.2)),
        ("apt purge / dnf remove",         (-3.4,  1.4)),
    ]
    for txt, (cx, cy) in edge_cmds:
        ax.text(cx, cy, txt, ha="center", va="center",
                fontsize=8.8, color=COLORS["text"], family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=COLOR_LIGHT, lw=0.8))

    # Center hub
    _rounded_box(ax, -1.4, -0.5, 2.8, 1.0,
                 facecolor="#f8fafc", edgecolor=COLOR_GREY, lw=1.2)
    ax.text(0, 0.15, "package state",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=COLORS["text"])
    ax.text(0, -0.25, "tracked in /var/lib/dpkg  /  rpmdb",
            ha="center", va="center", fontsize=8.5,
            color=COLOR_GREY, family="monospace")

    # Title
    ax.text(0, 4.5, "Lifecycle of a package on one machine",
            ha="center", va="center", fontsize=14.5, fontweight="bold",
            color=COLORS["text"])
    ax.text(0, 4.05,
            "search -> install -> upgrade -> hold -> remove,  "
            "with autoremove cleaning orphan deps",
            ha="center", va="center", fontsize=10,
            color=COLOR_GREY, style="italic")

    _save(fig, "fig3_package_lifecycle.png")


# ---------------------------------------------------------------------------
# Figure 4: Repository structure (apt example) + chain of trust
# ---------------------------------------------------------------------------

def fig4_repository_structure() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ---------- Left: tree of files on the mirror ----------
    ax.text(3.0, 8.4, "Apt repository on the mirror",
            ha="center", fontsize=12.5, fontweight="bold", color=COLORS["text"])
    ax.text(3.0, 8.0, "(Debian / Ubuntu layout)",
            ha="center", fontsize=9, color=COLOR_GREY, style="italic")

    tree_lines = [
        ("dists/", COLOR_BLUE, 0),
        ("jammy/",                       COLOR_BLUE, 1),
        ("Release",                      COLOR_AMBER, 2),
        ("Release.gpg / InRelease",      COLOR_AMBER, 2),
        ("main/",                        COLOR_PURPLE, 2),
        ("binary-amd64/Packages.gz",     COLOR_GREY,   3),
        ("source/Sources.gz",            COLOR_GREY,   3),
        ("universe/  restricted/  ...",  COLOR_PURPLE, 2),
        ("pool/",                        COLOR_BLUE,   0),
        ("main/n/nginx/nginx_1.24.0.deb", COLOR_GREEN, 1),
        ("main/o/openssl/libssl3_3.0.2.deb", COLOR_GREEN, 1),
        ("...", COLOR_GREY, 1),
    ]

    base_y = 7.5
    line_h = 0.42
    for i, (name, color, indent) in enumerate(tree_lines):
        y = base_y - i * line_h
        prefix = "    " * indent
        bullet = "+- " if indent > 0 else ""
        ax.text(0.4, y, prefix + bullet, fontsize=10,
                color=COLOR_GREY, family="monospace")
        ax.text(0.4 + 0.25 * len(prefix) + 0.27 * len(bullet), y, name,
                fontsize=10.5, color=color, family="monospace",
                fontweight="bold" if indent <= 1 else "normal")

    # File role legend
    legend = [
        ("Release",  "checksums + signature for the suite", COLOR_AMBER),
        ("Packages", "index of every .deb in this component", COLOR_GREY),
        ("pool/",    "actual .deb files, deduplicated",       COLOR_GREEN),
    ]
    for i, (k, v, c) in enumerate(legend):
        ax.text(0.4, 1.6 - i * 0.45, k,
                fontsize=10, fontweight="bold", color=c, family="monospace")
        ax.text(1.6, 1.6 - i * 0.45, v,
                fontsize=9.5, color=COLOR_GREY)

    # ---------- Right: chain of trust ----------
    ax.text(8.8, 8.4, "Chain of trust  (apt update -> apt install)",
            ha="center", fontsize=12.5, fontweight="bold", color=COLORS["text"])

    steps = [
        ("apt-key / signed-by",
         "GPG public keys you trust\n(/etc/apt/trusted.gpg.d/)", COLOR_BLUE),
        ("InRelease",
         "signed manifest of the suite\n(checksums of all index files)", COLOR_AMBER),
        ("Packages.gz",
         "index: name, version, deps,\nfilename, SHA256 of the .deb", COLOR_PURPLE),
        ("nginx_1.24.0.deb",
         "downloaded, SHA256 verified,\nthen unpacked by dpkg", COLOR_GREEN),
    ]

    box_x = 6.4
    box_w = 5.2
    box_h = 1.15
    gap = 0.35
    top = 7.6
    for i, (label, sub, color) in enumerate(steps):
        y = top - i * (box_h + gap)
        _rounded_box(ax, box_x, y - box_h, box_w, box_h,
                     facecolor="white", edgecolor=color, lw=1.6)
        # left strip
        _rounded_box(ax, box_x, y - box_h, 0.18, box_h,
                     facecolor=color, edgecolor=color)
        ax.text(box_x + 0.4, y - 0.35, label,
                fontsize=11, fontweight="bold", color=color,
                family="monospace")
        ax.text(box_x + 0.4, y - 0.85, sub,
                fontsize=9, color=COLOR_GREY)

        # Down-arrow between boxes
        if i < len(steps) - 1:
            arr = FancyArrowPatch(
                (box_x + box_w / 2, y - box_h),
                (box_x + box_w / 2, y - box_h - gap),
                arrowstyle="->", mutation_scale=14,
                color=COLOR_GREY, lw=1.2,
            )
            ax.add_patch(arr)

    # Footer: what breaks if any link fails
    ax.text(6, 0.35,
            "If the GPG key is missing -> 'NO_PUBKEY' on apt update.    "
            "If the SHA256 doesn't match -> apt refuses to install the .deb.",
            ha="center", fontsize=9.3, color=COLOR_GREY, style="italic")

    ax.text(6, 8.85, "Anatomy of an apt repository",
            ha="center", fontsize=14.5, fontweight="bold", color=COLORS["text"])

    _save(fig, "fig4_repository_structure.png")


# ---------------------------------------------------------------------------
# Figure 5: Modern alternatives - Snap / Flatpak / AppImage
# ---------------------------------------------------------------------------

def fig5_modern_alternatives() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    formats = ["distro pkg", "Snap", "Flatpak", "AppImage"]
    fmt_colors = [COLOR_GREY, COLOR_BLUE, COLOR_PURPLE, COLOR_GREEN]

    rows = [
        ("Sponsor",
         ["distro vendor", "Canonical", "freedesktop.org", "community"]),
        ("Package format",
         [".deb / .rpm", ".snap (squashfs)", ".flatpak (OSTree)", ".AppImage (single file)"]),
        ("Bundles deps?",
         ["no - shared libs", "yes - in the snap", "yes - via runtime", "yes - in the file"]),
        ("Sandboxed?",
         ["no", "yes (AppArmor)", "yes (bubblewrap)", "no (optional)"]),
        ("Auto-update",
         ["apt/dnf upgrade", "snapd, automatic", "flatpak update", "manual / appimaged"]),
        ("Disk footprint",
         ["small", "large (own libs)", "large (own runtime)", "medium"]),
        ("Best fit",
         ["system tools, services",
          "desktop apps, IoT, servers",
          "desktop apps (GUI)",
          "portable single-file apps"]),
    ]

    col0_x = 0.4
    col0_w = 2.6
    col_w = 2.2
    col_gap = 0.05
    row_h = 0.7
    table_top = 7.5

    # Header
    for i, (fmt, color) in enumerate(zip(formats, fmt_colors)):
        x = col0_x + col0_w + col_gap + i * (col_w + col_gap)
        _rounded_box(ax, x, table_top, col_w, row_h * 1.05,
                     facecolor=color, edgecolor=color, alpha=0.95)
        ax.text(x + col_w / 2, table_top + row_h * 0.5, fmt,
                ha="center", va="center", color="white",
                fontsize=12.5, fontweight="bold")

    ax.text(col0_x + col0_w / 2, table_top + row_h * 0.5,
            "dimension", ha="center", va="center", color=COLOR_GREY,
            fontsize=10, fontweight="bold", style="italic")

    # Body rows
    for r, (label, values) in enumerate(rows):
        y = table_top - (r + 1) * row_h
        ax.add_patch(Rectangle((col0_x, y), col0_w, row_h * 0.95,
                               facecolor="#f1f5f9", edgecolor=COLOR_LIGHT,
                               lw=0.8))
        ax.text(col0_x + 0.2, y + row_h * 0.5, label,
                ha="left", va="center", fontsize=10.2, color=COLORS["text"],
                fontweight="bold")
        for i, (val, color) in enumerate(zip(values, fmt_colors)):
            x = col0_x + col0_w + col_gap + i * (col_w + col_gap)
            ax.add_patch(Rectangle((x, y), col_w, row_h * 0.95,
                                   facecolor="white", edgecolor=COLOR_LIGHT,
                                   lw=0.8))
            ax.text(x + col_w / 2, y + row_h * 0.5, val,
                    ha="center", va="center", fontsize=8.8,
                    color=color, fontweight="bold")

    # Footer summary
    ax.text(6, 1.4,
            "Distro packages share libraries with the rest of the system - "
            "small and tightly integrated, but bound to the distro release.",
            ha="center", fontsize=9.3, color=COLOR_GREY, style="italic")
    ax.text(6, 1.0,
            "Snap / Flatpak / AppImage bundle their own dependencies - "
            "always up-to-date, but trade disk space and integration.",
            ha="center", fontsize=9.3, color=COLOR_GREY, style="italic")
    ax.text(6, 0.45,
            "Rule of thumb:  servers -> distro pkg.  desktop GUI app -> "
            "Flatpak / Snap.  one-shot binary -> AppImage.",
            ha="center", fontsize=10, color=COLOR_BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eff6ff",
                      edgecolor=COLOR_BLUE, lw=1.0))

    ax.text(6, 8.65,
            "Modern alternatives:  Snap  /  Flatpak  /  AppImage  vs  distro packages",
            ha="center", fontsize=14, fontweight="bold", color=COLORS["text"])

    _save(fig, "fig5_modern_alternatives.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_package_manager_comparison()
    fig2_dependency_resolution()
    fig3_package_lifecycle()
    fig4_repository_structure()
    fig5_modern_alternatives()
    print("Wrote 5 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
