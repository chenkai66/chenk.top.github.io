#!/usr/bin/env python3
"""Generate all 6 figures for the Functional Analysis series on chenk.top."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Design tokens
BG_COLOR = "#fdfcf9"
COLORS = {
    "red": "#e85d4a",
    "amber": "#f5a834",
    "purple": "#8b5cf6",
    "blue": "#3b82f6",
    "green": "#10b981",
    "gray": "#6b7280",
}
FONT_MAIN = {"fontsize": 13, "fontfamily": "serif"}
FONT_TITLE = {"fontsize": 15, "fontfamily": "serif", "fontweight": "bold"}
FONT_SMALL = {"fontsize": 10, "fontfamily": "serif"}

OUTPUT_DIR = "/tmp/fa-figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG_COLOR, pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {path}")


def fig01_nested_spaces():
    """Nested spaces: Metric > Normed > Banach, with examples."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 5)

    # Three nested rounded rectangles
    boxes = [
        (0.2, 0.2, 11.5, 4.5, COLORS["blue"], "Metric Spaces", 0.15),
        (1.5, 0.6, 9.0, 3.7, COLORS["purple"], "Normed Spaces", 0.20),
        (3.0, 1.0, 6.0, 2.9, COLORS["green"], "Banach Spaces", 0.25),
    ]

    for x, y, w, h, color, label, alpha in boxes:
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + 0.3, y + h - 0.4, label, color=color,
                fontsize=14, fontfamily="serif", fontweight="bold")

    # Examples in each layer
    # Metric only
    ax.text(0.7, 1.0, "discrete metric spaces\nultrametric spaces",
            **FONT_SMALL, color=COLORS["blue"], style="italic")

    # Normed but not Banach
    ax.text(2.0, 1.5, r"$(c_{00}, \|\cdot\|_2)$" + "\n" + r"$(C[0,1], \|\cdot\|_1)$",
            **FONT_SMALL, color=COLORS["purple"], style="italic")

    # Banach
    ax.text(4.5, 1.8, r"$\ell^p\ (1 \leq p \leq \infty)$" + "\n" +
            r"$L^p(\Omega)$" + "\n" +
            r"$(C[0,1], \|\cdot\|_\infty)$",
            fontsize=12, fontfamily="serif", color=COLORS["green"], style="italic")

    # Hilbert subset callout
    hilbert_box = FancyBboxPatch(
        (5.0, 2.8), 3.2, 0.9,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["red"], alpha=0.15,
        edgecolor=COLORS["red"], linewidth=1.5, linestyle="--"
    )
    ax.add_patch(hilbert_box)
    ax.text(5.3, 3.0, r"Hilbert: $\ell^2,\ L^2,\ H^k$",
            fontsize=11, fontfamily="serif", color=COLORS["red"], fontweight="bold")

    # Arrow annotations
    ax.annotate("completeness", xy=(3.0, 2.4), xytext=(1.8, 2.8),
                fontsize=10, fontfamily="serif", color=COLORS["gray"],
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))
    ax.annotate("linear structure\n+ homogeneity", xy=(1.5, 2.2), xytext=(0.5, 3.5),
                fontsize=10, fontfamily="serif", color=COLORS["gray"],
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))

    save_fig(fig, "fig01_nested_spaces.png")


def fig02_projection():
    """Orthogonal projection onto a subspace (2D geometric illustration)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    ax.set_aspect("equal")

    # Subspace M (a line through origin)
    ax.plot([-0.3, 5.5], [-0.15, 2.75], color=COLORS["blue"], lw=2, alpha=0.6)
    ax.text(5.3, 2.9, r"$M$", fontsize=14, color=COLORS["blue"], fontfamily="serif")

    # Vector x
    x_pos = (3.5, 4.0)
    ax.annotate("", xy=x_pos, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=2.5))
    ax.text(x_pos[0] + 0.1, x_pos[1] + 0.1, r"$x$", fontsize=14,
            color=COLORS["red"], fontfamily="serif", fontweight="bold")

    # Projection P_M x
    proj = (3.0, 1.5)  # on the line y = 0.5x
    ax.annotate("", xy=proj, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["green"], lw=2.5))
    ax.text(proj[0] + 0.15, proj[1] - 0.4, r"$P_M x$", fontsize=14,
            color=COLORS["green"], fontfamily="serif", fontweight="bold")

    # Error vector x - P_M x (dashed, perpendicular)
    ax.annotate("", xy=x_pos, xytext=proj,
                arrowprops=dict(arrowstyle="-|>", color=COLORS["purple"], lw=2,
                               linestyle="dashed"))
    mid = ((x_pos[0] + proj[0]) / 2 + 0.2, (x_pos[1] + proj[1]) / 2)
    ax.text(mid[0], mid[1], r"$x - P_M x$", fontsize=12,
            color=COLORS["purple"], fontfamily="serif")

    # Right angle marker
    size = 0.25
    # Direction along M from proj
    dm = np.array([1, 0.5])
    dm = dm / np.linalg.norm(dm) * size
    # Direction perpendicular (toward x)
    dp = np.array(x_pos) - np.array(proj)
    dp = dp / np.linalg.norm(dp) * size
    corner = np.array(proj)
    p1 = corner + dm
    p2 = corner + dm + dp
    p3 = corner + dp
    ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]],
            color=COLORS["gray"], lw=1.5)

    # Label
    ax.text(0.5, 4.0, "Orthogonal Projection",
            fontsize=15, fontfamily="serif", fontweight="bold", color=COLORS["gray"])
    ax.text(0.5, 3.5, r"$(x - P_M x) \perp M$",
            fontsize=13, fontfamily="serif", color=COLORS["gray"])
    ax.text(0.5, 3.0, r"$\|x - P_M x\| = d(x, M)$",
            fontsize=13, fontfamily="serif", color=COLORS["gray"])

    # Origin dot
    ax.plot(0, 0, "o", color="black", markersize=5)
    ax.text(-0.2, -0.3, "0", fontsize=11, fontfamily="serif")

    save_fig(fig, "fig02_projection.png")


def fig03_operator_norm():
    """Operator norm: unit ball mapping through T."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(BG_COLOR)

    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axhline(0, color=COLORS["gray"], lw=0.5, alpha=0.3)
        ax.axvline(0, color=COLORS["gray"], lw=0.5, alpha=0.3)
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Left: unit ball in X
    theta = np.linspace(0, 2 * np.pi, 100)
    axes[0].fill(np.cos(theta), np.sin(theta), alpha=0.2, color=COLORS["blue"])
    axes[0].plot(np.cos(theta), np.sin(theta), color=COLORS["blue"], lw=2)
    axes[0].set_title(r"Unit ball $B_X$", fontsize=13, fontfamily="serif", pad=10)
    axes[0].text(0, -1.7, r"$\|x\| \leq 1$", ha="center", fontsize=11,
                fontfamily="serif", color=COLORS["blue"])

    # Arrow between
    fig.text(0.37, 0.5, r"$T$", fontsize=18, fontfamily="serif",
             fontweight="bold", ha="center", va="center")
    fig.text(0.37, 0.4, r"$\longrightarrow$", fontsize=20, ha="center", va="center")

    # Middle: image T(B_X) - an ellipse
    a, b = 1.5, 0.7
    axes[1].fill(a * np.cos(theta), b * np.sin(theta), alpha=0.2, color=COLORS["green"])
    axes[1].plot(a * np.cos(theta), b * np.sin(theta), color=COLORS["green"], lw=2)
    axes[1].set_title(r"Image $T(B_X)$", fontsize=13, fontfamily="serif", pad=10)
    # Mark the operator norm
    axes[1].plot([0, a], [0, 0], "--", color=COLORS["red"], lw=1.5)
    axes[1].plot(a, 0, "o", color=COLORS["red"], markersize=8)
    axes[1].text(a * 0.5, 0.2, r"$\|T\|$", fontsize=12, color=COLORS["red"],
                fontfamily="serif", fontweight="bold")

    # Right: ||T|| ball in Y containing image
    axes[2].fill(a * np.cos(theta), a * np.sin(theta), alpha=0.08, color=COLORS["amber"])
    axes[2].plot(a * np.cos(theta), a * np.sin(theta), color=COLORS["amber"],
                lw=2, linestyle="--")
    axes[2].fill(a * np.cos(theta), b * np.sin(theta), alpha=0.2, color=COLORS["green"])
    axes[2].plot(a * np.cos(theta), b * np.sin(theta), color=COLORS["green"], lw=2)
    axes[2].set_title(r"$T(B_X) \subseteq \|T\| \cdot B_Y$", fontsize=13,
                     fontfamily="serif", pad=10)
    axes[2].text(0, -1.8, r"$\|Tx\| \leq \|T\|\,\|x\|$", ha="center",
                fontsize=11, fontfamily="serif", color=COLORS["gray"])

    fig.text(0.64, 0.5, r"$\subseteq$", fontsize=18, ha="center", va="center",
             fontfamily="serif", color=COLORS["gray"])

    plt.tight_layout(pad=2)
    save_fig(fig, "fig03_operator_norm.png")


def fig04_big_four():
    """The Big Four theorems: relationship diagram with boxes and arrows."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    # Four theorem boxes
    theorems = [
        (1, 4, "Hahn-Banach", "Algebraic / Zorn's Lemma\nNo completeness needed",
         COLORS["red"]),
        (7, 4, "Uniform\nBoundedness", "Baire Category\nDomain must be Banach",
         COLORS["blue"]),
        (1, 1, "Open Mapping\n(Banach-Schauder)", "Baire Category\nBoth spaces Banach",
         COLORS["green"]),
        (7, 1, "Closed Graph", "Baire Category\nBoth spaces Banach",
         COLORS["purple"]),
    ]

    for x, y, title, subtitle, color in theorems:
        box = FancyBboxPatch(
            (x, y), 3.8, 1.5,
            boxstyle="round,pad=0.12",
            facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=2
        )
        ax.add_patch(box)
        ax.text(x + 1.9, y + 1.1, title, ha="center", va="center",
                fontsize=12, fontfamily="serif", fontweight="bold", color=color)
        ax.text(x + 1.9, y + 0.35, subtitle, ha="center", va="center",
                fontsize=9, fontfamily="serif", color=COLORS["gray"])

    # Arrows showing relationships
    # Open Mapping <-> Closed Graph (equivalent)
    ax.annotate("", xy=(6.9, 1.75), xytext=(4.9, 1.75),
                arrowprops=dict(arrowstyle="<->", color=COLORS["gray"], lw=2))
    ax.text(5.9, 2.0, "equivalent", ha="center", fontsize=10,
            fontfamily="serif", color=COLORS["gray"], style="italic")

    # Baire -> Open Mapping and Uniform Boundedness
    ax.text(5.9, 3.2, "Baire Category\nTheorem", ha="center", va="center",
            fontsize=11, fontfamily="serif", fontweight="bold",
            color=COLORS["amber"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["amber"],
                     alpha=0.15, edgecolor=COLORS["amber"]))

    ax.annotate("", xy=(4.9, 1.9), xytext=(5.5, 3.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["amber"], lw=1.5))
    ax.annotate("", xy=(7.0, 4.0), xytext=(6.3, 3.4),
                arrowprops=dict(arrowstyle="->", color=COLORS["amber"], lw=1.5))
    ax.annotate("", xy=(7.0, 1.9), xytext=(6.3, 3.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["amber"], lw=1.5))

    # Hahn-Banach: standalone (Zorn)
    ax.text(2.9, 3.5, "Zorn's\nLemma", ha="center", va="center",
            fontsize=10, fontfamily="serif", color=COLORS["red"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["red"],
                     alpha=0.1, edgecolor=COLORS["red"], linestyle="--"))
    ax.annotate("", xy=(2.9, 5.0), xytext=(2.9, 3.8),
                arrowprops=dict(arrowstyle="<-", color=COLORS["red"], lw=1.5,
                               linestyle="dashed"))

    # Common theme at bottom
    ax.text(6, 0.3, "Common theme: qualitative hypotheses  →  quantitative conclusions",
            ha="center", fontsize=12, fontfamily="serif", color=COLORS["gray"],
            style="italic")

    save_fig(fig, "fig04_big_four.png")


def fig05_spectrum():
    """Spectrum decomposition on the complex plane."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Axes
    ax.axhline(0, color=COLORS["gray"], lw=0.8, alpha=0.4)
    ax.axvline(0, color=COLORS["gray"], lw=0.8, alpha=0.4)
    ax.text(2.3, -0.2, r"Re", fontsize=11, fontfamily="serif", color=COLORS["gray"])
    ax.text(0.1, 1.85, r"Im", fontsize=11, fontfamily="serif", color=COLORS["gray"])

    # Spectral radius circle (dashed)
    theta = np.linspace(0, 2 * np.pi, 100)
    r_spec = 1.8
    ax.plot(r_spec * np.cos(theta), r_spec * np.sin(theta), "--",
            color=COLORS["gray"], lw=1, alpha=0.5)
    ax.text(1.85, -1.6, r"$r(T) = \|T\|$", fontsize=10,
            fontfamily="serif", color=COLORS["gray"])

    # Point spectrum (eigenvalues) - discrete dots
    eigenvalues = [1.2, 0.6, 0.3, 0.15, 0.08]
    for ev in eigenvalues:
        ax.plot(ev, 0, "o", color=COLORS["red"], markersize=8, zorder=5)
    ax.plot(0, 0, "o", color=COLORS["red"], markersize=6, zorder=5)
    ax.text(1.2, 0.25, r"$\lambda_1$", fontsize=11, color=COLORS["red"],
            fontfamily="serif")
    ax.text(0.6, 0.25, r"$\lambda_2$", fontsize=11, color=COLORS["red"],
            fontfamily="serif")
    ax.text(0.3, 0.25, r"$\lambda_3$", fontsize=11, color=COLORS["red"],
            fontfamily="serif")

    # Continuous spectrum - a thick arc on unit circle
    t_arc = np.linspace(0.3, 2.8, 80)
    ax.plot(np.cos(t_arc), np.sin(t_arc), color=COLORS["blue"], lw=4, alpha=0.6)
    ax.text(-0.9, 1.3, r"$\sigma_c$", fontsize=13, color=COLORS["blue"],
            fontfamily="serif", fontweight="bold")

    # Residual spectrum - shaded region
    t_res = np.linspace(-0.5, -2.5, 60)
    x_res = 0.5 * np.cos(t_res) - 1.0
    y_res = 0.3 * np.sin(t_res)
    ax.fill(x_res, y_res, color=COLORS["purple"], alpha=0.2)
    ax.plot(x_res, y_res, color=COLORS["purple"], lw=2, alpha=0.6)
    ax.text(-1.3, -0.5, r"$\sigma_r$", fontsize=13, color=COLORS["purple"],
            fontfamily="serif", fontweight="bold")

    # Legend
    legend_y = -1.6
    ax.plot(-2.2, legend_y, "o", color=COLORS["red"], markersize=8)
    ax.text(-2.0, legend_y, r"Point spectrum $\sigma_p$ (eigenvalues)",
            fontsize=10, fontfamily="serif", color=COLORS["red"], va="center")
    ax.plot(-2.2, legend_y - 0.3, "s", color=COLORS["blue"], markersize=8)
    ax.text(-2.0, legend_y - 0.3, r"Continuous spectrum $\sigma_c$",
            fontsize=10, fontfamily="serif", color=COLORS["blue"], va="center")
    ax.plot(-2.2, legend_y - 0.6, "D", color=COLORS["purple"], markersize=8)
    ax.text(-2.0, legend_y - 0.6, r"Residual spectrum $\sigma_r$",
            fontsize=10, fontfamily="serif", color=COLORS["purple"], va="center")

    # Annotation: eigenvalues cluster at 0
    ax.annotate(r"$\lambda_n \to 0$", xy=(0.08, 0), xytext=(0.5, -0.8),
                fontsize=11, fontfamily="serif", color=COLORS["red"],
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2))

    ax.set_title("Spectrum Decomposition of a Bounded Operator",
                fontsize=14, fontfamily="serif", fontweight="bold",
                color=COLORS["gray"], pad=15)

    save_fig(fig, "fig05_spectrum.png")


def fig06_hierarchy():
    """Function space hierarchy: test functions < Schwartz < L^2 < distributions < Sobolev."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)

    # Nested containment from left (smallest) to right (largest)
    spaces = [
        (0.5, 1.0, 2.0, 3.0, COLORS["green"], r"$\mathcal{D}$" + "\nTest functions\n" + r"$C_c^\infty$"),
        (2.8, 0.8, 2.2, 3.4, COLORS["blue"], r"$\mathcal{S}$" + "\nSchwartz\n(rapid decay)"),
        (5.3, 0.6, 2.0, 3.8, COLORS["purple"], r"$L^2$" + "\nSquare-\nintegrable"),
        (7.6, 0.4, 2.2, 4.2, COLORS["amber"], r"$\mathcal{S}'$" + "\nTempered\ndistributions"),
        (10.0, 0.2, 1.7, 4.6, COLORS["red"], r"$\mathcal{D}'$" + "\nAll\ndistributions"),
    ]

    for x, y, w, h, color, label in spaces:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=2
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=11, fontfamily="serif", color=color, fontweight="bold")

    # Arrows showing inclusion
    arrows_x = [2.55, 5.05, 7.35, 9.85]
    for ax_x in arrows_x:
        ax.annotate("", xy=(ax_x + 0.15, 2.5), xytext=(ax_x - 0.15, 2.5),
                    arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))
        ax.text(ax_x, 2.8, r"$\subset$", ha="center", fontsize=14,
                fontfamily="serif", color=COLORS["gray"])

    # Sobolev spaces annotation (spans across)
    ax.plot([3.0, 9.5], [0.15, 0.15], color=COLORS["gray"], lw=1.5, linestyle="--")
    ax.text(6.25, -0.15, r"Sobolev spaces $W^{k,p}$ live here (between $L^p$ and $\mathcal{D}'$)",
            ha="center", fontsize=10, fontfamily="serif", color=COLORS["gray"], style="italic")

    # Title
    ax.text(6, 4.9, "Function Space Hierarchy",
            ha="center", fontsize=14, fontfamily="serif",
            fontweight="bold", color=COLORS["gray"])

    save_fig(fig, "fig06_hierarchy.png")


if __name__ == "__main__":
    fig01_nested_spaces()
    fig02_projection()
    fig03_operator_norm()
    fig04_big_four()
    fig05_spectrum()
    fig06_hierarchy()
    print("\nAll 6 figures generated in", OUTPUT_DIR)
