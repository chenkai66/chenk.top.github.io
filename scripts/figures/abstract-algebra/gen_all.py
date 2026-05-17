#!/usr/bin/env python3
"""Generate 6 PNG figures for the Abstract Algebra series on chenk.top."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUTPUT_DIR = "/tmp/abstract-algebra-figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG = "#fdfcf9"
COLORS = {
    "red": "#e85d4a",
    "amber": "#f5a834",
    "purple": "#8b5cf6",
    "blue": "#3b82f6",
    "green": "#10b981",
    "gray": "#6b7280",
}


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig01_group_table():
    """Group operation table for Z/4Z (colored cells)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # Z/4Z addition table
    n = 4
    elements = list(range(n))
    table = [[(a + b) % n for b in elements] for a in elements]

    cell_colors = {0: "#e8f5e9", 1: "#e3f2fd", 2: "#fff3e0", 3: "#fce4ec"}

    cell_size = 0.8
    x_start, y_start = 1.5, 3.5

    # Title
    ax.text(3.5, 5.2, r"Addition Table for $\mathbb{Z}/4\mathbb{Z}$",
            fontsize=14, ha="center", fontweight="bold", color="#1a1a1a")

    # Header row
    ax.text(x_start - 0.5, y_start + 0.4, r"$+$", fontsize=12, ha="center", va="center",
            fontweight="bold", color=COLORS["gray"])
    for j, b in enumerate(elements):
        x = x_start + j * cell_size + 0.4
        ax.text(x, y_start + 0.4, str(b), fontsize=12, ha="center", va="center",
                fontweight="bold", color=COLORS["purple"])

    # Table body
    for i, a in enumerate(elements):
        y = y_start - i * cell_size - 0.4
        # Row header
        ax.text(x_start - 0.5, y, str(a), fontsize=12, ha="center", va="center",
                fontweight="bold", color=COLORS["purple"])
        for j, b in enumerate(elements):
            x = x_start + j * cell_size + 0.4
            val = table[i][j]
            rect = FancyBboxPatch((x - 0.35, y - 0.3), 0.7, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor=cell_colors[val], edgecolor="none")
            ax.add_patch(rect)
            ax.text(x, y, str(val), fontsize=12, ha="center", va="center", color="#333")

    # Annotations
    ax.text(3.5, 0.8, r"Cyclic: every element is $1 + 1 + \cdots + 1$",
            fontsize=10.5, ha="center", color=COLORS["gray"])
    ax.text(3.5, 0.3, r"Identity: 0    |    Inverse of $k$: $4 - k$    |    Order: 4",
            fontsize=9.5, ha="center", color=COLORS["gray"])

    ax.set_xlim(0, 7)
    ax.set_ylim(-0.2, 5.8)
    return save(fig, "fig01_group_table.png")


def fig02_kernel_image():
    """Kernel-Image diagram for a homomorphism."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # Title
    ax.text(6, 5.5, r"Homomorphism $\varphi: G \to H$ — Kernel and Image",
            fontsize=14, ha="center", fontweight="bold", color="#1a1a1a")

    # G ellipse (left)
    g_center = (3, 3)
    g_ellipse = mpatches.Ellipse(g_center, 4, 5, facecolor="#ede9fe", edgecolor=COLORS["purple"],
                                  linewidth=2, alpha=0.7)
    ax.add_patch(g_ellipse)
    ax.text(3, 5.3, r"$G$", fontsize=14, ha="center", fontweight="bold", color=COLORS["purple"])

    # Kernel inside G
    ker_ellipse = mpatches.Ellipse((3, 2.2), 2.2, 2.2, facecolor="#fef3c7",
                                    edgecolor=COLORS["amber"], linewidth=1.5, alpha=0.8)
    ax.add_patch(ker_ellipse)
    ax.text(3, 2.2, r"$\ker\varphi$", fontsize=11, ha="center", color=COLORS["amber"],
            fontweight="bold")
    ax.text(3, 1.5, r"$\{g : \varphi(g) = e_H\}$", fontsize=9.5, ha="center", color=COLORS["gray"])

    # H ellipse (right)
    h_center = (9, 3)
    h_ellipse = mpatches.Ellipse(h_center, 4, 5, facecolor="#ecfdf5", edgecolor=COLORS["green"],
                                  linewidth=2, alpha=0.7)
    ax.add_patch(h_ellipse)
    ax.text(9, 5.3, r"$H$", fontsize=14, ha="center", fontweight="bold", color=COLORS["green"])

    # Image inside H
    im_ellipse = mpatches.Ellipse((9, 3.3), 2.5, 3, facecolor="#dbeafe",
                                   edgecolor=COLORS["blue"], linewidth=1.5, alpha=0.8)
    ax.add_patch(im_ellipse)
    ax.text(9, 3.3, r"$\mathrm{im}\,\varphi$", fontsize=11, ha="center", color=COLORS["blue"],
            fontweight="bold")

    # e_H point
    ax.plot(9, 1.8, 'o', color=COLORS["red"], markersize=6)
    ax.text(9.3, 1.8, r"$e_H$", fontsize=9.5, color=COLORS["red"])

    # Arrow from G to H
    ax.annotate("", xy=(6.8, 3.8), xytext=(5.2, 3.8),
                arrowprops=dict(arrowstyle="->, head_width=0.3", color="#333", lw=2))
    ax.text(6, 4.3, r"$\varphi$", fontsize=13, ha="center", color="#333", fontweight="bold")

    # First Isomorphism Theorem
    box = FancyBboxPatch((2.5, -0.3), 7, 0.8, boxstyle="round,pad=0.15",
                          facecolor="#fef9f0", edgecolor=COLORS["amber"], linewidth=1.5)
    ax.add_patch(box)
    ax.text(6, 0.1, r"First Isomorphism Theorem:  $G/\ker\varphi \;\cong\; \mathrm{im}\,\varphi$",
            fontsize=11, ha="center", va="center", color="#333")

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 6)
    return save(fig, "fig02_kernel_image.png")


def fig03_ring_hierarchy():
    """Ring hierarchy: Rings > Commutative > Integral Domain > Field."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(6, 5.7, "Ring Hierarchy", fontsize=14, ha="center", fontweight="bold", color="#1a1a1a")

    # Nested boxes (outermost to innermost)
    levels = [
        (0.5, 0.3, 11, 5.0, "Rings", COLORS["gray"], "#f9fafb",
         r"$(R, +, \cdot)$: abelian group under $+$, associative $\cdot$, distributive"),
        (1.3, 0.6, 9.4, 4.0, "Commutative Rings", COLORS["purple"], "#f5f3ff",
         r"$ab = ba$ for all $a, b$"),
        (2.2, 0.9, 7.6, 3.0, "Integral Domains", COLORS["blue"], "#eff6ff",
         r"No zero divisors: $ab = 0 \Rightarrow a = 0$ or $b = 0$"),
        (3.3, 1.2, 5.4, 2.0, "Fields", COLORS["green"], "#ecfdf5",
         r"Every nonzero element invertible"),
    ]

    for x, y, w, h, label, color, fc, desc in levels:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=fc, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.3, y + h - 0.35, label, fontsize=12, fontweight="bold",
                color=color, va="top")
        ax.text(x + w / 2, y + 0.25, desc, fontsize=9.5, ha="center", color=COLORS["gray"])

    # Examples on the right
    examples = [
        (11.8, 4.5, r"$M_n(\mathbb{R})$", COLORS["gray"]),
        (11.8, 3.5, r"$\mathbb{Z}[x]$", COLORS["purple"]),
        (11.8, 2.5, r"$\mathbb{Z}$", COLORS["blue"]),
        (11.8, 1.5, r"$\mathbb{Q}, \mathbb{R}, \mathbb{F}_p$", COLORS["green"]),
    ]
    for x, y, txt, color in examples:
        ax.text(x, y, txt, fontsize=10.5, ha="center", color=color)

    ax.set_xlim(0, 13)
    ax.set_ylim(-0.2, 6.2)
    return save(fig, "fig03_ring_hierarchy.png")


def fig04_field_extension_tower():
    """Field extension tower diagram."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(5, 6.8, r"Field Extension Tower: $x^3 - 2$ over $\mathbb{Q}$",
            fontsize=14, ha="center", fontweight="bold", color="#1a1a1a")

    # Tower levels (bottom to top)
    levels = [
        (5, 1.0, r"$\mathbb{Q}$", COLORS["gray"]),
        (5, 3.0, r"$\mathbb{Q}(\sqrt[3]{2})$", COLORS["blue"]),
        (5, 5.0, r"$\mathbb{Q}(\sqrt[3]{2}, \omega)$", COLORS["purple"]),
    ]

    for x, y, label, color in levels:
        box = FancyBboxPatch((x - 1.5, y - 0.35), 3, 0.7, boxstyle="round,pad=0.1",
                              facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, fontsize=12, ha="center", va="center", color=color, fontweight="bold")

    # Connecting lines with degree labels
    connections = [
        (5, 1.35, 5, 2.65, r"degree 3", r"min. poly: $x^3 - 2$"),
        (5, 3.35, 5, 4.65, r"degree 2", r"min. poly: $x^2 + x + 1$"),
    ]

    for x1, y1, x2, y2, deg, poly in connections:
        ax.plot([x1, x2], [y1, y2], color="#333", linewidth=1.5, linestyle="-")
        mid_y = (y1 + y2) / 2
        ax.text(x1 + 1.8, mid_y + 0.1, deg, fontsize=11, ha="center", color=COLORS["red"],
                fontweight="bold")
        ax.text(x1 + 1.8, mid_y - 0.3, poly, fontsize=9.5, ha="center", color=COLORS["gray"])

    # Total degree bracket on the left
    ax.annotate("", xy=(2.8, 5.0), xytext=(2.8, 1.0),
                arrowprops=dict(arrowstyle="<->", color=COLORS["amber"], lw=2))
    ax.text(2.0, 3.0, r"$[\,:\mathbb{Q}] = 6$", fontsize=11, ha="center", color=COLORS["amber"],
            fontweight="bold", rotation=90)

    # Note at bottom
    ax.text(5, 0.1, r"Tower Law: $[K:F] = [K:L] \cdot [L:F]$  so  $6 = 3 \times 2$",
            fontsize=10.5, ha="center", color=COLORS["gray"])

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 7.3)
    return save(fig, "fig04_field_tower.png")


def fig05_galois_correspondence():
    """Galois correspondence lattice for Q(sqrt2, sqrt3)/Q."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(6, 6.8, r"Galois Correspondence: $\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$",
            fontsize=14, ha="center", fontweight="bold", color="#1a1a1a")

    # Left side: subgroups (top = small, bottom = big — reversed from fields)
    ax.text(3, 6.2, "Subgroups of $V_4$", fontsize=11, ha="center", color=COLORS["purple"],
            fontweight="bold")

    sg_positions = {
        "{e}": (3, 5.5),
        r"$\langle\sigma\rangle$": (1.5, 3.8),
        r"$\langle\tau\rangle$": (3, 3.8),
        r"$\langle\sigma\tau\rangle$": (4.5, 3.8),
        "$V_4$": (3, 2.0),
    }

    for label, (x, y) in sg_positions.items():
        box = FancyBboxPatch((x - 0.7, y - 0.25), 1.4, 0.5, boxstyle="round,pad=0.08",
                              facecolor="#f5f3ff", edgecolor=COLORS["purple"], linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, fontsize=10, ha="center", va="center", color=COLORS["purple"])

    # Lines for subgroup lattice
    sg_lines = [
        ((3, 5.25), (1.5, 4.05)), ((3, 5.25), (3, 4.05)), ((3, 5.25), (4.5, 4.05)),
        ((1.5, 3.55), (3, 2.25)), ((3, 3.55), (3, 2.25)), ((4.5, 3.55), (3, 2.25)),
    ]
    for (x1, y1), (x2, y2) in sg_lines:
        ax.plot([x1, x2], [y1, y2], color=COLORS["purple"], linewidth=1, alpha=0.6)

    # Right side: intermediate fields (top = big, bottom = small)
    ax.text(9, 6.2, "Intermediate Fields", fontsize=11, ha="center", color=COLORS["green"],
            fontweight="bold")

    field_positions = {
        r"$\mathbb{Q}(\sqrt{2},\sqrt{3})$": (9, 5.5),
        r"$\mathbb{Q}(\sqrt{3})$": (7.5, 3.8),
        r"$\mathbb{Q}(\sqrt{2})$": (9, 3.8),
        r"$\mathbb{Q}(\sqrt{6})$": (10.5, 3.8),
        r"$\mathbb{Q}$": (9, 2.0),
    }

    for label, (x, y) in field_positions.items():
        box = FancyBboxPatch((x - 0.9, y - 0.25), 1.8, 0.5, boxstyle="round,pad=0.08",
                              facecolor="#ecfdf5", edgecolor=COLORS["green"], linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, fontsize=10, ha="center", va="center", color=COLORS["green"])

    # Lines for field lattice
    field_lines = [
        ((9, 5.25), (7.5, 4.05)), ((9, 5.25), (9, 4.05)), ((9, 5.25), (10.5, 4.05)),
        ((7.5, 3.55), (9, 2.25)), ((9, 3.55), (9, 2.25)), ((10.5, 3.55), (9, 2.25)),
    ]
    for (x1, y1), (x2, y2) in field_lines:
        ax.plot([x1, x2], [y1, y2], color=COLORS["green"], linewidth=1, alpha=0.6)

    # Correspondence arrows
    corr_pairs = [
        ((3.7, 5.5), (8.0, 5.5)),
        ((2.2, 3.8), (6.6, 3.8)),
        ((3.7, 3.8), (8.1, 3.8)),
        ((5.2, 3.8), (9.6, 3.8)),
        ((3.7, 2.0), (8.0, 2.0)),
    ]
    for (x1, y1), (x2, y2) in corr_pairs:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["amber"], lw=1.5,
                                    linestyle="dashed"))

    # Legend
    ax.text(6, 1.0, r"$\longleftrightarrow$  Inclusion-reversing bijection",
            fontsize=10.5, ha="center", color=COLORS["amber"])
    ax.text(6, 0.4, r"Bigger subgroup $\leftrightarrow$ smaller fixed field",
            fontsize=9.5, ha="center", color=COLORS["gray"])

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 7.3)
    return save(fig, "fig05_galois_correspondence.png")


def fig06_applications():
    """Applications overview: 3-column layout."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(6, 5.8, "Abstract Algebra in Applications", fontsize=14, ha="center",
            fontweight="bold", color="#1a1a1a")

    columns = [
        {
            "title": "Coding Theory",
            "color": COLORS["blue"],
            "x": 2,
            "items": [
                r"$\mathbb{F}_q$ arithmetic",
                "Reed-Solomon codes",
                "Hamming codes",
                "Polynomial GCD",
            ],
            "algebra": r"Finite fields $\mathbb{F}_{p^n}$"
        },
        {
            "title": "Cryptography",
            "color": COLORS["purple"],
            "x": 6,
            "items": [
                r"$(\mathbb{Z}/n\mathbb{Z})^*$ for RSA",
                "Elliptic curve groups",
                "Discrete logarithm",
                "Lattice-based (rings)",
            ],
            "algebra": r"Groups, Rings, Fields"
        },
        {
            "title": "Physics",
            "color": COLORS["green"],
            "x": 10,
            "items": [
                r"$SU(3) \times SU(2) \times U(1)$",
                "Representation theory",
                "Gauge invariance",
                "Symmetry breaking",
            ],
            "algebra": "Lie groups & algebras"
        },
    ]

    for col in columns:
        x = col["x"]
        color = col["color"]

        # Column header box
        header_box = FancyBboxPatch((x - 1.6, 4.8), 3.2, 0.7, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor="none", alpha=0.15)
        ax.add_patch(header_box)
        ax.text(x, 5.15, col["title"], fontsize=12, ha="center", va="center",
                fontweight="bold", color=color)

        # Items
        for i, item in enumerate(col["items"]):
            y = 4.2 - i * 0.7
            ax.text(x, y, item, fontsize=10.5, ha="center", va="center", color="#333")

        # Bottom algebra note
        alg_box = FancyBboxPatch((x - 1.6, 0.8), 3.2, 0.6, boxstyle="round,pad=0.08",
                                  facecolor="#fef9f0", edgecolor=COLORS["amber"], linewidth=1)
        ax.add_patch(alg_box)
        ax.text(x, 1.1, col["algebra"], fontsize=9.5, ha="center", va="center",
                color=COLORS["amber"])

    # Bottom label
    ax.text(6, 0.2, "Core algebraic structures powering each domain",
            fontsize=9.5, ha="center", color=COLORS["gray"])

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 6.3)
    return save(fig, "fig06_applications.png")


if __name__ == "__main__":
    print("Generating Abstract Algebra figures...")
    fig01_group_table()
    fig02_kernel_image()
    fig03_ring_hierarchy()
    fig04_field_extension_tower()
    fig05_galois_correspondence()
    fig06_applications()
    print("All 6 figures generated in", OUTPUT_DIR)
