"""
Figure generation script for Linear Algebra Chapter 01:
"The Essence of Vectors".

Generates 7 figures used in BOTH the EN and ZH versions of the article.
Each figure teaches one geometric idea cleanly, in a 3Blue1Brown-inspired
style: dark axes, saturated arrows, generous white space, tight focal points.

Figures:
    fig1_vector_as_arrow         A 2D vector as a directed arrow from the
                                 origin, with components, length and angle
                                 labelled.
    fig2_vector_addition         Head-to-tail and parallelogram methods,
                                 shown side by side, both producing the same
                                 sum.
    fig3_scalar_multiplication   Stretching, shrinking and reversing a
                                 vector under scalar multiplication.
    fig4_position_vs_free        Position vector (anchored at origin) versus
                                 free vector (translation-invariant) in a
                                 single panel.
    fig5_dot_product_projection  Geometric interpretation of the dot product
                                 as length-of-projection times length-of-b.
    fig6_norm_unit_balls         Unit balls of L1, L2 and L-infinity norms,
                                 each in its own subplot for comparison.
    fig7_three_interpretations   The same 4-tuple shown as: a physics arrow,
                                 a feature vector (CS), and a math column.

Usage:
    python3 scripts/figures/linear-algebra/01-the-essence-of-vectors.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages. Parent folders are created
    if missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Circle, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary vector
C_PURPLE = "#7c3aed"   # secondary vector
C_GREEN = "#10b981"    # accent / sum / projection
C_AMBER = "#f59e0b"    # highlight / warning
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linear-algebra" / "01-the-essence-of-vectors"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linear-algebra" / "01-向量的本质"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _setup_axes(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    grid: bool = True,
    show_axes: bool = True,
    title: str | None = None,
) -> None:
    """Apply a consistent 3Blue1Brown-style axis treatment."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, color=C_LIGHT, linewidth=0.8, zorder=0)
    else:
        ax.grid(False)
    if show_axes:
        ax.axhline(0, color=C_GRAY, linewidth=1.0, zorder=1)
        ax.axvline(0, color=C_GRAY, linewidth=1.0, zorder=1)
    ax.set_xticks(np.arange(int(xlim[0]), int(xlim[1]) + 1))
    ax.set_yticks(np.arange(int(ylim[0]), int(ylim[1]) + 1))
    ax.tick_params(labelsize=9, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, color=C_DARK, pad=10, weight="semibold")


def _arrow(
    ax: plt.Axes,
    tail: tuple[float, float],
    head: tuple[float, float],
    color: str,
    *,
    lw: float = 2.4,
    alpha: float = 1.0,
    zorder: int = 5,
    mutation_scale: float = 18,
) -> FancyArrowPatch:
    """Draw a clean, thick arrow with a filled head."""
    a = FancyArrowPatch(
        tail, head,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        color=color,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(a)
    return a


# ---------------------------------------------------------------------------
# Figure 1: Vector as a directed arrow with components, length and angle
# ---------------------------------------------------------------------------
def fig1_vector_as_arrow() -> None:
    """A single 2D vector v = (4, 3) as an arrow from the origin."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    _setup_axes(ax, (-1, 6), (-1, 5))

    v = np.array([4.0, 3.0])

    # Component decomposition (dashed)
    ax.plot([0, v[0]], [0, 0], color=C_GRAY, linestyle="--", linewidth=1.4, zorder=2)
    ax.plot([v[0], v[0]], [0, v[1]], color=C_GRAY, linestyle="--", linewidth=1.4, zorder=2)

    # Right-angle marker
    ax.add_patch(Rectangle((v[0] - 0.22, 0), 0.22, 0.22,
                           fill=False, edgecolor=C_GRAY, linewidth=1.0, zorder=3))

    # Angle arc
    theta = np.degrees(np.arctan2(v[1], v[0]))
    arc = mpatches.Arc((0, 0), 1.6, 1.6, angle=0, theta1=0, theta2=theta,
                       color=C_AMBER, linewidth=1.8, zorder=3)
    ax.add_patch(arc)
    ax.text(0.95, 0.32, r"$\theta \approx 37^\circ$",
            color=C_AMBER, fontsize=11, weight="bold")

    # The main vector
    _arrow(ax, (0, 0), tuple(v), C_BLUE, lw=3.0, mutation_scale=22)

    # Length label along the vector
    mid = v / 2 + np.array([-0.45, 0.35])
    ax.text(mid[0], mid[1], r"$\|\vec{v}\| = 5$",
            color=C_BLUE, fontsize=13, weight="bold",
            rotation=theta, rotation_mode="anchor")

    # Component labels
    ax.text(v[0] / 2, -0.45, r"$v_x = 4$", color=C_DARK, fontsize=11,
            ha="center", weight="semibold")
    ax.text(v[0] + 0.18, v[1] / 2, r"$v_y = 3$", color=C_DARK, fontsize=11,
            va="center", weight="semibold")

    # Tip annotation
    ax.scatter([v[0]], [v[1]], color=C_BLUE, s=42, zorder=6)
    ax.annotate(r"$\vec{v} = (4,\,3)$",
                xy=(v[0], v[1]), xytext=(v[0] + 0.35, v[1] + 0.35),
                color=C_BLUE, fontsize=12, weight="bold")

    # Origin
    ax.scatter([0], [0], color=C_DARK, s=28, zorder=6)
    ax.text(-0.35, -0.45, "O", color=C_DARK, fontsize=11, weight="bold")

    ax.set_title("A 2D vector: magnitude, direction, components",
                 fontsize=13, color=C_DARK, pad=12, weight="semibold")

    fig.tight_layout()
    _save(fig, "fig1_vector_as_arrow")


# ---------------------------------------------------------------------------
# Figure 2: Vector addition -- head-to-tail vs parallelogram, side by side
# ---------------------------------------------------------------------------
def fig2_vector_addition() -> None:
    """Head-to-tail and parallelogram methods, both yielding a + b."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    a = np.array([3.0, 1.0])
    b = np.array([1.0, 2.5])
    s = a + b

    # ----- Left: head-to-tail -----
    ax = axes[0]
    _setup_axes(ax, (-1, 5), (-1, 5),
                title="Head-to-tail: walk, then walk again")
    _arrow(ax, (0, 0), tuple(a), C_BLUE, lw=2.6)
    _arrow(ax, tuple(a), tuple(a + b), C_PURPLE, lw=2.6)
    _arrow(ax, (0, 0), tuple(s), C_GREEN, lw=3.0, mutation_scale=22)

    ax.text(a[0] / 2 - 0.1, a[1] / 2 - 0.45, r"$\vec{a}$",
            color=C_BLUE, fontsize=14, weight="bold")
    ax.text(a[0] + b[0] / 2 - 0.05, a[1] + b[1] / 2 + 0.15, r"$\vec{b}$",
            color=C_PURPLE, fontsize=14, weight="bold")
    ax.text(s[0] / 2 - 0.55, s[1] / 2 + 0.25, r"$\vec{a}+\vec{b}$",
            color=C_GREEN, fontsize=14, weight="bold")

    # ----- Right: parallelogram -----
    ax = axes[1]
    _setup_axes(ax, (-1, 5), (-1, 5),
                title="Parallelogram: combine forces from one point")

    # Filled parallelogram
    poly = Polygon([(0, 0), tuple(a), tuple(s), tuple(b)],
                   closed=True, facecolor=C_GREEN, alpha=0.10,
                   edgecolor=C_GRAY, linewidth=1.2, linestyle="--", zorder=2)
    ax.add_patch(poly)

    _arrow(ax, (0, 0), tuple(a), C_BLUE, lw=2.6)
    _arrow(ax, (0, 0), tuple(b), C_PURPLE, lw=2.6)
    _arrow(ax, (0, 0), tuple(s), C_GREEN, lw=3.0, mutation_scale=22)

    ax.text(a[0] / 2, a[1] / 2 - 0.5, r"$\vec{a}$",
            color=C_BLUE, fontsize=14, weight="bold")
    ax.text(b[0] / 2 - 0.5, b[1] / 2, r"$\vec{b}$",
            color=C_PURPLE, fontsize=14, weight="bold")
    ax.text(s[0] / 2 + 0.15, s[1] / 2 + 0.15, r"$\vec{a}+\vec{b}$",
            color=C_GREEN, fontsize=14, weight="bold")

    fig.suptitle("Vector addition: two pictures, one answer  "
                 r"$(3,1)+(1,2.5)=(4,3.5)$",
                 fontsize=13, color=C_DARK, weight="semibold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_vector_addition")


# ---------------------------------------------------------------------------
# Figure 3: Scalar multiplication -- stretch, shrink, reverse
# ---------------------------------------------------------------------------
def fig3_scalar_multiplication() -> None:
    """Show c*v for several c with the same base vector v."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    _setup_axes(ax, (-5, 6), (-4, 5))

    v = np.array([2.0, 1.5])

    scalars = [
        (2.0, C_PURPLE, r"$2\vec{v}$"),
        (1.0, C_BLUE,   r"$\vec{v}$"),
        (0.5, C_GREEN,  r"$\frac{1}{2}\vec{v}$"),
        (-1.0, C_AMBER, r"$-\vec{v}$"),
    ]

    # Draw the dashed line of all real multiples of v (the span)
    t = np.linspace(-2.5, 2.5, 2)
    line_xs = t * v[0]
    line_ys = t * v[1]
    ax.plot(line_xs, line_ys, color=C_GRAY, linestyle=":",
            linewidth=1.3, zorder=2,
            label=r"span$(\vec{v})$: all scalar multiples")

    # Draw arrows from longest to shortest so shorter ones aren't hidden
    for c, color, _ in sorted(scalars, key=lambda x: -abs(x[0])):
        end = c * v
        _arrow(ax, (0, 0), tuple(end), color, lw=2.8, mutation_scale=20)

    # Labels (placed independently)
    label_positions = {
        2.0: (4.2, 3.4),
        1.0: (2.05, 1.85),
        0.5: (0.9, 0.45),
        -1.0: (-2.55, -1.65),
    }
    for c, color, txt in scalars:
        x, y = label_positions[c]
        ax.text(x, y, txt, color=color, fontsize=14, weight="bold")

    ax.set_title("Scalar multiplication: stretching, shrinking, flipping",
                 fontsize=13, color=C_DARK, pad=12, weight="semibold")
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    fig.tight_layout()
    _save(fig, "fig3_scalar_multiplication")


# ---------------------------------------------------------------------------
# Figure 4: Position vector vs free vector (translation invariance)
# ---------------------------------------------------------------------------
def fig4_position_vs_free() -> None:
    """Same vector drawn from different basepoints; both equal."""
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    _setup_axes(ax, (-1, 8), (-1, 6))

    direction = np.array([3.0, 2.0])
    bases = [
        np.array([0.0, 0.0]),
        np.array([1.5, 2.5]),
        np.array([4.0, 0.5]),
    ]
    colors = [C_BLUE, C_PURPLE, C_GREEN]
    labels = [
        "anchored at origin\n(position vector)",
        "free vector\nsame direction & length",
        "free vector\nsame direction & length",
    ]

    for base, color, lab in zip(bases, colors, labels):
        head = base + direction
        _arrow(ax, tuple(base), tuple(head), color, lw=2.6, mutation_scale=20)
        ax.scatter([base[0]], [base[1]], color=color, s=34, zorder=6)
        # label near the middle, perpendicular offset
        mid = base + direction / 2
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)
        offset = 0.45
        ax.text(mid[0] + normal[0] * offset,
                mid[1] + normal[1] * offset,
                lab, color=color, fontsize=9.5, ha="center", va="center",
                weight="semibold")

    # Annotate that all three arrows ARE the same vector
    ax.text(4.5, 5.3,
            r"All three arrows represent the same vector  $\vec{v} = (3,\,2)$"
            "\n(translation invariance)",
            color=C_DARK, fontsize=11.5, weight="semibold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=C_BG,
                      edgecolor=C_GRAY, linewidth=1.0))

    ax.set_title("A vector has no basepoint -- only direction and magnitude",
                 fontsize=13, color=C_DARK, pad=12, weight="semibold")

    fig.tight_layout()
    _save(fig, "fig4_position_vs_free")


# ---------------------------------------------------------------------------
# Figure 5: Dot product as projection (geometric interpretation)
# ---------------------------------------------------------------------------
def fig5_dot_product_projection() -> None:
    """Show a, b, the projection of a onto b, and a x b = |a||b|cos(theta)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # Left panel: acute angle, positive dot product
    a = np.array([3.5, 2.5])
    b = np.array([4.0, 0.5])

    def _draw_proj(ax, a, b, title):
        _setup_axes(ax, (-1, 6), (-1.5, 4), title=title)
        # b
        _arrow(ax, (0, 0), tuple(b), C_PURPLE, lw=2.6)
        ax.text(b[0] / 2 + 0.05, b[1] / 2 - 0.5, r"$\vec{b}$",
                color=C_PURPLE, fontsize=14, weight="bold")

        # a
        _arrow(ax, (0, 0), tuple(a), C_BLUE, lw=2.6)
        ax.text(a[0] / 2 - 0.55, a[1] / 2 + 0.15, r"$\vec{a}$",
                color=C_BLUE, fontsize=14, weight="bold")

        # projection scalar and vector
        bb = float(np.dot(b, b))
        ab = float(np.dot(a, b))
        proj = (ab / bb) * b
        # drop perpendicular from tip of a to line of b
        ax.plot([a[0], proj[0]], [a[1], proj[1]],
                color=C_GRAY, linestyle="--", linewidth=1.4, zorder=3)

        # right-angle marker at foot of perpendicular
        # build small square aligned with b
        bhat = b / np.linalg.norm(b)
        nhat = np.array([-bhat[1], bhat[0]])
        # if a is on the upper side, normal should point up
        if np.dot(a - proj, nhat) < 0:
            nhat = -nhat
        sq = 0.22
        sq_pts = [proj,
                  proj - bhat * sq,
                  proj - bhat * sq + nhat * sq,
                  proj + nhat * sq]
        ax.plot([p[0] for p in sq_pts] + [sq_pts[0][0]],
                [p[1] for p in sq_pts] + [sq_pts[0][1]],
                color=C_GRAY, linewidth=1.0, zorder=3)

        # projection arrow (thicker green)
        _arrow(ax, (0, 0), tuple(proj), C_GREEN, lw=3.4, mutation_scale=22,
               zorder=7)
        ax.text(proj[0] / 2, -0.55,
                r"$\mathrm{proj}_{\vec{b}}\,\vec{a}$",
                color=C_GREEN, fontsize=12, weight="bold", ha="center")

        # Angle arc
        theta = np.degrees(np.arctan2(a[1], a[0])) - np.degrees(np.arctan2(b[1], b[0]))
        arc = mpatches.Arc((0, 0), 1.4, 1.4,
                           angle=np.degrees(np.arctan2(b[1], b[0])),
                           theta1=0, theta2=theta,
                           color=C_AMBER, linewidth=1.8)
        ax.add_patch(arc)
        ax.text(0.85, 0.25, r"$\theta$",
                color=C_AMBER, fontsize=12, weight="bold")

        # Formula box
        cos_t = ab / (np.linalg.norm(a) * np.linalg.norm(b))
        ax.text(0.05, 0.95,
                r"$\vec{a}\cdot\vec{b} = \|\vec{a}\|\,\|\vec{b}\|\cos\theta$"
                f"\n$= {ab:.2f}$  ({'positive' if ab > 0 else 'negative'})",
                transform=ax.transAxes, fontsize=11, color=C_DARK,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.45",
                          facecolor=C_BG, edgecolor=C_GRAY, linewidth=1.0))

    _draw_proj(axes[0], a, b,
               "Acute angle  -> dot product > 0 (vectors agree)")

    # Right: obtuse, negative dot product
    a2 = np.array([-2.0, 2.5])
    b2 = np.array([4.0, 0.5])
    _draw_proj(axes[1], a2, b2,
               "Obtuse angle  -> dot product < 0 (vectors disagree)")

    fig.suptitle("Dot product = signed length of projection x length of "
                 r"$\vec{b}$",
                 fontsize=13, color=C_DARK, weight="semibold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_dot_product_projection")


# ---------------------------------------------------------------------------
# Figure 6: Unit balls of L1, L2, L-infinity norms
# ---------------------------------------------------------------------------
def fig6_norm_unit_balls() -> None:
    """Three side-by-side unit balls for three different norms."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    panels = [
        (1, r"$L^1$ (Manhattan): $\sum |x_i| = 1$",
         "Diamond -- promotes sparsity (LASSO)", C_AMBER),
        (2, r"$L^2$ (Euclidean): $\sqrt{\sum x_i^2} = 1$",
         "Circle -- rotation-invariant default", C_BLUE),
        ("inf", r"$L^\infty$ (Max): $\max |x_i| = 1$",
         "Square -- worst-case bound", C_GREEN),
    ]

    theta = np.linspace(0, 2 * np.pi, 720)

    for ax, (p, title, sub, color) in zip(axes, panels):
        _setup_axes(ax, (-1.6, 1.6), (-1.6, 1.6), title=title)
        if p == 1:
            xs = np.array([1, 0, -1, 0, 1])
            ys = np.array([0, 1, 0, -1, 0])
            ax.fill(xs, ys, color=color, alpha=0.18, zorder=2)
            ax.plot(xs, ys, color=color, linewidth=2.6, zorder=4)
        elif p == 2:
            xs = np.cos(theta)
            ys = np.sin(theta)
            ax.fill(xs, ys, color=color, alpha=0.18, zorder=2)
            ax.plot(xs, ys, color=color, linewidth=2.6, zorder=4)
        else:  # inf
            xs = np.array([1, -1, -1, 1, 1])
            ys = np.array([1, 1, -1, -1, 1])
            ax.fill(xs, ys, color=color, alpha=0.18, zorder=2)
            ax.plot(xs, ys, color=color, linewidth=2.6, zorder=4)

        # Mark a few unit points
        ax.scatter([1, 0, -1, 0], [0, 1, 0, -1], color=color, s=28, zorder=5)
        ax.text(0, -1.42, sub, ha="center", va="top",
                fontsize=10.5, color=C_DARK, weight="semibold")

    fig.suptitle("Unit balls: each norm has its own geometric fingerprint",
                 fontsize=13, color=C_DARK, weight="semibold", y=1.04)
    fig.tight_layout()
    _save(fig, "fig6_norm_unit_balls")


# ---------------------------------------------------------------------------
# Figure 7: Three interpretations of the same vector
# ---------------------------------------------------------------------------
def fig7_three_interpretations() -> None:
    """Same numbers, three meanings: physics arrow, CS features, math column."""
    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.85], wspace=0.35)

    # ---- Panel A: physics (3D-ish projection of an arrow) ----
    axA = fig.add_subplot(gs[0, 0])
    _setup_axes(axA, (-1, 5), (-1, 5),
                title="Physics: an arrow in space")
    v = np.array([3.5, 2.5])
    _arrow(axA, (0, 0), tuple(v), C_BLUE, lw=3.0, mutation_scale=22)
    axA.text(v[0] / 2 - 0.6, v[1] / 2 + 0.25,
             r"$\vec{F} = (3.5,\,2.5)$"
             "\nforce, velocity,\nacceleration...",
             color=C_BLUE, fontsize=10.5, weight="semibold")

    # ---- Panel B: CS feature vector (bar chart of features) ----
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("Computer science: a feature vector",
                  fontsize=12, color=C_DARK, pad=10, weight="semibold")
    feats = ["temp", "humid", "press", "wind", "cloud"]
    vals = [25.3, 65.0, 101.3, 15.2, 45.0]   # pressure scaled to fit
    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_GRAY]
    bars = axB.bar(feats, vals, color=palette, edgecolor="white", linewidth=1.5)
    for b, v_ in zip(bars, vals):
        axB.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                 f"{v_:.1f}", ha="center", fontsize=9.5,
                 color=C_DARK, weight="semibold")
    axB.set_ylim(0, 130)
    axB.set_ylabel("value", fontsize=10, color=C_DARK)
    axB.grid(True, axis="y", color=C_LIGHT, linewidth=0.8)
    axB.set_axisbelow(True)
    for spine in ("top", "right"):
        axB.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axB.spines[spine].set_color(C_GRAY)
    axB.tick_params(labelsize=10, colors=C_DARK)

    # ---- Panel C: math column vector (rendered as a tall matrix) ----
    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("Mathematics: a column vector",
                  fontsize=12, color=C_DARK, pad=10, weight="semibold")
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.axis("off")
    column = (
        r"$\vec{w} \;=\;$"
    )
    matrix_lines = ["25.3", "65.0", "1013", "15.2", "  45"]
    axC.text(0.30, 0.55, column, ha="center", va="center",
             fontsize=20, color=C_DARK)
    # Render the column with a tall pair of brackets and stacked numbers.
    bracket_x_left = 0.46
    bracket_x_right = 0.78
    top_y, bot_y = 0.85, 0.20
    axC.plot([bracket_x_left, bracket_x_left - 0.04, bracket_x_left - 0.04, bracket_x_left],
             [top_y, top_y, bot_y, bot_y],
             color=C_DARK, linewidth=1.6, solid_capstyle="round")
    axC.plot([bracket_x_right, bracket_x_right + 0.04, bracket_x_right + 0.04, bracket_x_right],
             [top_y, top_y, bot_y, bot_y],
             color=C_DARK, linewidth=1.6, solid_capstyle="round")
    ys = np.linspace(top_y - 0.08, bot_y + 0.05, len(matrix_lines))
    palette_col = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_GRAY]
    for y, txt, col in zip(ys, matrix_lines, palette_col):
        axC.text(0.62, y, txt, ha="center", va="center",
                 fontsize=15, color=col, weight="semibold")
    axC.text(0.92, 0.55, r"$\in \mathbb{R}^{5}$", ha="center", va="center",
             fontsize=15, color=C_DARK)
    axC.text(0.5, 0.10,
             "an ordered list of\nreal numbers",
             ha="center", va="center", fontsize=10.5,
             color=C_GRAY, style="italic")

    fig.suptitle("Same object, three viewpoints -- this is the power of "
                 "the vector concept",
                 fontsize=13, color=C_DARK, weight="semibold", y=1.02)
    _save(fig, "fig7_three_interpretations")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Linear Algebra Chapter 01...")
    fig1_vector_as_arrow()
    fig2_vector_addition()
    fig3_scalar_multiplication()
    fig4_position_vs_free()
    fig5_dot_product_projection()
    fig6_norm_unit_balls()
    fig7_three_interpretations()
    print("Done.")


if __name__ == "__main__":
    main()
