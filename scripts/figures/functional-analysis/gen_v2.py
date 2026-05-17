#!/usr/bin/env python3
"""
Generate 84 functional analysis figures for chenk.top blog.
12 articles x 7 figures = 84 PNGs at DPI=180.
"""
import os, math, traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch, Wedge, Arc, Ellipse
from matplotlib.patheffects import withSimplePatchShadow
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# ---------- design tokens ----------
BG = "#fdfcf9"
C = {
    "red":    "#e85d4a",
    "amber":  "#f5a834",
    "purple": "#8b5cf6",
    "blue":   "#3b82f6",
    "green":  "#10b981",
    "gray":   "#6b7280",
    "dark":   "#1f2937",
    "light":  "#f3f4f6",
    "lightgray": "#d1d5db",
}
shadow = withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.15)

OUTROOT = "/tmp/fa_figs_v2"

def newfig(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    return fig, ax

def clean_axes(ax, keep_spines=False):
    if not keep_spines:
        for s in ax.spines.values():
            s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

def title(ax, t, sub=None, y=1.02):
    ax.set_title(t, fontsize=14, color=C["dark"], fontweight="bold", pad=10, y=y)
    if sub:
        ax.text(0.5, y-0.07, sub, transform=ax.transAxes, ha="center",
                fontsize=10, color=C["gray"], style="italic")

def box(ax, xy, w, h, text, color, fc=None, fontsize=10, alpha=0.18, fontweight="bold"):
    if fc is None: fc = color
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                       fc=fc, ec=color, lw=1.6, alpha=alpha, path_effects=[shadow])
    ax.add_patch(p)
    p2 = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                        fc="none", ec=color, lw=1.8)
    ax.add_patch(p2)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center",
            fontsize=fontsize, color=C["dark"], fontweight=fontweight)

def arrow(ax, p1, p2, color="#6b7280", lw=1.8, style="-|>", mut=14, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, color=color, lw=lw,
                        mutation_scale=mut, linestyle=ls)
    ax.add_patch(a)

def label(ax, x, y, text, color=None, fontsize=10, weight="normal", ha="center", va="center", bg=None):
    if color is None: color = C["dark"]
    kwargs = dict(ha=ha, va=va, fontsize=fontsize, color=color, fontweight=weight)
    if bg is not None:
        kwargs["bbox"] = dict(facecolor=bg, edgecolor="none", pad=2, alpha=0.9)
    ax.text(x, y, text, **kwargs)

def save(fig, slug, name):
    d = os.path.join(OUTROOT, slug)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".png")
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


# ============================================================
# 01 metric-spaces
# ============================================================
def fa_v2_01_1_metric_axioms():
    fig, ax = newfig((9, 6))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    title(ax, "Metric Space Axioms", "d: X×X → ℝ satisfying four properties")
    items = [
        ("Non-negativity", "d(x,y) ≥ 0", C["blue"], 0.5, 4.5),
        ("Identity",       "d(x,y)=0 ⇔ x=y", C["green"], 5.2, 4.5),
        ("Symmetry",       "d(x,y) = d(y,x)", C["amber"], 0.5, 1.5),
        ("Triangle ineq.", "d(x,z) ≤ d(x,y)+d(y,z)", C["red"], 5.2, 1.5),
    ]
    for name, eq, col, x, y in items:
        box(ax, (x, y), 4.3, 1.6, "", col)
        ax.text(x+0.2, y+1.15, name, fontsize=11, color=col, fontweight="bold")
        ax.text(x+2.15, y+0.5, eq, fontsize=12, color=C["dark"], ha="center", style="italic")
    return save(fig, "01-metric-spaces", "fa_v2_01_1_metric_axioms")

def fa_v2_01_2_open_balls():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), facecolor=BG)
    titles = ["Euclidean (ℓ²)", "Taxicab (ℓ¹)", "Chebyshev (ℓ∞)"]
    cols = [C["blue"], C["amber"], C["purple"]]
    t = np.linspace(0, 2*np.pi, 200)
    for i, (ax, ti, col) in enumerate(zip(axes, titles, cols)):
        ax.set_facecolor(BG)
        ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.axhline(0, color=C["lightgray"], lw=0.6); ax.axvline(0, color=C["lightgray"], lw=0.6)
        if i == 0:
            x, y = np.cos(t), np.sin(t)
        elif i == 1:
            x = np.array([1, 0, -1, 0, 1]); y = np.array([0, 1, 0, -1, 0])
        else:
            x = np.array([1, 1, -1, -1, 1]); y = np.array([1, -1, -1, 1, 1])
        ax.fill(x, y, color=col, alpha=0.22)
        ax.plot(x, y, color=col, lw=2.2)
        ax.plot(0, 0, "o", color=C["dark"], ms=4)
        ax.set_title(ti, fontsize=12, color=C["dark"], fontweight="bold")
        for s in ax.spines.values(): s.set_color(C["lightgray"])
        ax.tick_params(colors=C["gray"], labelsize=8)
    fig.suptitle("Unit Balls B(0,1) under Different Metrics", fontsize=14,
                 color=C["dark"], fontweight="bold", y=1.02)
    return save(fig, "01-metric-spaces", "fa_v2_01_2_open_balls")

def fa_v2_01_3_cauchy_seq():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)
    n = np.arange(1, 25)
    # converging
    a = 1 + (-1)**n / n
    axes[0].set_facecolor(BG)
    axes[0].plot(n, a, "o-", color=C["green"], lw=1.5, ms=5)
    axes[0].axhline(1, color=C["red"], ls="--", lw=1.5, label="limit = 1")
    axes[0].set_title("Cauchy & Convergent in ℝ", fontsize=12, color=C["dark"], fontweight="bold")
    axes[0].legend(frameon=False)
    # Cauchy in Q but not converging in Q
    b = (1 + 1/n)**n  # tends to e (irrational)
    axes[1].set_facecolor(BG)
    axes[1].plot(n, b, "o-", color=C["amber"], lw=1.5, ms=5)
    axes[1].axhline(np.e, color=C["red"], ls="--", lw=1.5, label="e (∉ ℚ)")
    axes[1].set_title("Cauchy in ℚ but no limit in ℚ", fontsize=12, color=C["dark"], fontweight="bold")
    axes[1].legend(frameon=False)
    for ax in axes:
        for s in ax.spines.values(): s.set_color(C["lightgray"])
        ax.tick_params(colors=C["gray"], labelsize=8)
        ax.grid(True, color=C["lightgray"], alpha=0.4, lw=0.5)
        ax.set_xlabel("n", color=C["gray"])
    fig.suptitle("Cauchy Sequences: Convergence Depends on the Space",
                 fontsize=14, color=C["dark"], fontweight="bold", y=1.02)
    return save(fig, "01-metric-spaces", "fa_v2_01_3_cauchy_seq")

def fa_v2_01_4_completion():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Completion: ℚ → ℝ", "Equivalence classes of Cauchy sequences")
    box(ax, (0.5, 1.5), 3, 2, "ℚ\n(rationals)\nincomplete", C["amber"])
    box(ax, (6.5, 1.5), 3, 2, "ℝ\n(reals)\ncomplete", C["green"])
    arrow(ax, (3.6, 2.5), (6.4, 2.5), color=C["blue"], lw=2.2)
    ax.text(5, 3.0, "completion", fontsize=11, color=C["blue"],
            ha="center", fontweight="bold")
    ax.text(5, 2.0, "[(xₙ)] ~ [(yₙ)]\nif d(xₙ,yₙ)→0", fontsize=10,
            color=C["gray"], ha="center", style="italic")
    ax.text(5, 0.7, "every Cauchy sequence becomes convergent in the completion",
            fontsize=10, color=C["dark"], ha="center")
    return save(fig, "01-metric-spaces", "fa_v2_01_4_completion")

def fa_v2_01_5_baire():
    fig, ax = newfig((9, 5.5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    title(ax, "Baire Category Theorem", "Complete metric spaces are not meager")
    box(ax, (0.4, 4.0), 9.2, 1.4, "X complete metric space", C["green"], fontweight="bold")
    box(ax, (0.4, 2.2), 4.2, 1.4, "Uₙ open & dense", C["blue"])
    box(ax, (5.4, 2.2), 4.2, 1.4, "⋂ₙ Uₙ is dense", C["red"])
    arrow(ax, (4.6, 2.9), (5.4, 2.9), color=C["dark"])
    box(ax, (0.4, 0.3), 9.2, 1.4, "X ≠ ⋃ₙ Aₙ with Aₙ nowhere dense (no meager decomposition)",
        C["purple"], fontsize=10)
    return save(fig, "01-metric-spaces", "fa_v2_01_5_baire")

def fa_v2_01_6_fixed_point():
    fig, ax = newfig((8, 6))
    ax.set_facecolor(BG)
    # contraction f(x) = 0.6 x + 1, fixed point at 2.5
    f = lambda x: 0.6*x + 1
    xs = np.linspace(-0.5, 5, 200)
    ax.plot(xs, f(xs), color=C["blue"], lw=2.2, label="y = 0.6x + 1")
    ax.plot(xs, xs, color=C["gray"], lw=1.2, ls="--", label="y = x")
    # cobweb
    x = 0.2
    pts = [(x, 0)]
    for _ in range(8):
        y = f(x); pts.append((x, y)); pts.append((y, y)); x = y
    pts = np.array(pts)
    ax.plot(pts[:,0], pts[:,1], "-", color=C["red"], lw=1.5, alpha=0.8)
    ax.plot(2.5, 2.5, "o", color=C["green"], ms=10, zorder=5)
    ax.annotate("fixed point\nx* = 2.5", (2.5, 2.5), xytext=(3.4, 1.5),
                fontsize=10, color=C["green"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["green"]))
    ax.set_xlim(-0.2, 4.5); ax.set_ylim(-0.2, 4.5)
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=9)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Banach Fixed Point: Contraction Iteration", fontsize=14,
                 color=C["dark"], fontweight="bold")
    return save(fig, "01-metric-spaces", "fa_v2_01_6_fixed_point")

def fa_v2_01_7_compactness():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Compactness: Equivalent Conditions in Metric Spaces")
    items = [
        ("Sequential", "every seq has\nconverg. subseq.", C["blue"]),
        ("Heine-Borel", "every open cover\nhas finite subcover", C["green"]),
        ("Total bdd + complete", "ε-net for all ε > 0\n+ Cauchy ⇒ converg.", C["amber"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        x = 0.4 + i*3.2
        box(ax, (x, 1.3), 2.8, 2.4, "", col)
        ax.text(x+1.4, 3.2, n, fontsize=11, color=col, fontweight="bold", ha="center")
        ax.text(x+1.4, 2.2, eq, fontsize=10, color=C["dark"], ha="center", style="italic")
    ax.text(5, 0.6, "all three equivalent in metric spaces", fontsize=11,
            color=C["red"], ha="center", fontweight="bold")
    return save(fig, "01-metric-spaces", "fa_v2_01_7_compactness")


# ============================================================
# 02 normed-and-banach
# ============================================================
def fa_v2_02_1_unit_balls_lp():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), facecolor=BG)
    ps = [1, 1.5, 2, np.inf]
    titles = ["ℓ¹", "ℓ^{1.5}", "ℓ²", "ℓ∞"]
    cols = [C["red"], C["amber"], C["blue"], C["purple"]]
    t = np.linspace(0, 2*np.pi, 600)
    for ax, p, ti, col in zip(axes, ps, titles, cols):
        ax.set_facecolor(BG); ax.set_aspect("equal")
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.axhline(0, color=C["lightgray"], lw=0.5); ax.axvline(0, color=C["lightgray"], lw=0.5)
        if np.isinf(p):
            x = np.array([1,1,-1,-1,1]); y = np.array([1,-1,-1,1,1])
        elif p == 1:
            x = np.array([1,0,-1,0,1]); y = np.array([0,1,0,-1,0])
        else:
            x = np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/p)
            y = np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/p)
        ax.fill(x, y, color=col, alpha=0.22)
        ax.plot(x, y, color=col, lw=2.2)
        ax.set_title(ti, fontsize=13, color=col, fontweight="bold")
        for s in ax.spines.values(): s.set_color(C["lightgray"])
        ax.tick_params(colors=C["gray"], labelsize=7)
    fig.suptitle("Unit Balls in ℝ² for Different ℓ^p Norms", fontsize=14,
                 color=C["dark"], fontweight="bold", y=1.04)
    return save(fig, "02-normed-and-banach", "fa_v2_02_1_unit_balls_lp")

def fa_v2_02_2_norm_equiv():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Equivalence of Norms on Finite-Dim Space",
          "All norms on ℝⁿ induce the same topology")
    ax.text(5, 3.6, r"$c_1 \|x\|_a \leq \|x\|_b \leq c_2 \|x\|_a$",
            fontsize=16, color=C["dark"], ha="center")
    ax.text(5, 2.7, "for some constants 0 < c₁ ≤ c₂", fontsize=11, color=C["gray"],
            ha="center", style="italic")
    box(ax, (0.6, 0.5), 2.7, 1.4, "‖·‖₁", C["red"])
    box(ax, (3.7, 0.5), 2.7, 1.4, "‖·‖₂", C["blue"])
    box(ax, (6.8, 0.5), 2.7, 1.4, "‖·‖∞", C["purple"])
    return save(fig, "02-normed-and-banach", "fa_v2_02_2_norm_equiv")

def fa_v2_02_3_lp_chain():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Sequence Spaces ℓ^p Inclusion (1 ≤ p ≤ ∞)",
          "ℓ^p ⊊ ℓ^q for p < q (on counting measure)")
    chain = [("ℓ¹", C["red"]), ("ℓ²", C["amber"]), ("ℓ^p", C["green"]),
             ("ℓ^q", C["blue"]), ("ℓ∞", C["purple"])]
    for i, (n, col) in enumerate(chain):
        x = 0.5 + i * 1.95
        box(ax, (x, 2.0), 1.4, 1.4, n, col, fontsize=14)
        if i < 4:
            arrow(ax, (x+1.4, 2.7), (x+1.95, 2.7), color=C["dark"])
            ax.text(x+1.65, 3.0, "⊂", fontsize=12, color=C["gray"], ha="center")
    ax.text(5, 0.8, "ℓ¹ ⊂ ℓ² ⊂ … ⊂ ℓ^∞", fontsize=12, color=C["dark"], ha="center",
            fontweight="bold", style="italic")
    return save(fig, "02-normed-and-banach", "fa_v2_02_3_lp_chain")

def fa_v2_02_4_banach_complete():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Banach Space = Normed + Complete")
    box(ax, (0.5, 3.0), 4, 1.5, "Normed space\n(V, ‖·‖)", C["amber"])
    box(ax, (5.5, 3.0), 4, 1.5, "Complete\n(every Cauchy converges)", C["blue"])
    box(ax, (2.5, 0.5), 5, 1.5, "Banach space", C["green"], fontsize=14)
    arrow(ax, (2.5, 3.0), (4, 2.0), color=C["dark"])
    arrow(ax, (7.5, 3.0), (6, 2.0), color=C["dark"])
    return save(fig, "02-normed-and-banach", "fa_v2_02_4_banach_complete")

def fa_v2_02_5_schauder():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Schauder Basis", "Every x ∈ X has unique series x = Σ aₙ eₙ")
    ax.text(5, 3.5, r"$x \;=\; \sum_{n=1}^{\infty} a_n e_n$", fontsize=18,
            color=C["dark"], ha="center")
    ax.text(5, 2.6, "convergence in norm: ‖x − Σⁿ aₖ eₖ‖ → 0",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    box(ax, (0.6, 0.5), 2.7, 1.3, "Hamel basis\n(algebraic)", C["red"])
    box(ax, (3.7, 0.5), 2.7, 1.3, "Schauder basis\n(topological)", C["green"])
    box(ax, (6.8, 0.5), 2.7, 1.3, "Riesz basis\n(Hilbert)", C["blue"])
    return save(fig, "02-normed-and-banach", "fa_v2_02_5_schauder")

def fa_v2_02_6_equiv_norms():
    fig, ax = newfig((8, 7))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.axhline(0, color=C["lightgray"], lw=0.5); ax.axvline(0, color=C["lightgray"], lw=0.5)
    # ℓ1, ℓ2, ℓ∞ unit balls
    x1 = np.array([1,0,-1,0,1]); y1 = np.array([0,1,0,-1,0])
    t = np.linspace(0, 2*np.pi, 200)
    x2, y2 = np.cos(t), np.sin(t)
    x3 = np.array([1,1,-1,-1,1]); y3 = np.array([1,-1,-1,1,1])
    ax.fill(x3, y3, color=C["purple"], alpha=0.10)
    ax.plot(x3, y3, color=C["purple"], lw=2, label=r"$\|\cdot\|_\infty \leq 1$")
    ax.fill(x2, y2, color=C["blue"], alpha=0.18)
    ax.plot(x2, y2, color=C["blue"], lw=2, label=r"$\|\cdot\|_2 \leq 1$")
    ax.fill(x1, y1, color=C["red"], alpha=0.28)
    ax.plot(x1, y1, color=C["red"], lw=2, label=r"$\|\cdot\|_1 \leq 1$")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Equivalent Norms on ℝ²: Unit Ball Inclusions",
                 fontsize=13, color=C["dark"], fontweight="bold")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.text(0, -1.45, r"$B_1 \subset B_2 \subset B_\infty$",
            ha="center", fontsize=11, color=C["dark"], fontweight="bold")
    return save(fig, "02-normed-and-banach", "fa_v2_02_6_equiv_norms")

def fa_v2_02_7_seq_spaces():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Sequence Spaces c₀, c, ℓ∞ (with sup norm)")
    items = [
        ("c₀", "(xₙ) → 0", C["red"], 0.5),
        ("c",  "(xₙ) converges", C["amber"], 3.7),
        ("ℓ∞", "bounded sequences", C["blue"], 6.9),
    ]
    for n, eq, col, x in items:
        box(ax, (x, 1.5), 2.6, 2.0, "", col)
        ax.text(x+1.3, 2.9, n, fontsize=15, color=col, fontweight="bold", ha="center")
        ax.text(x+1.3, 2.0, eq, fontsize=10, color=C["dark"], ha="center", style="italic")
    ax.text(5, 0.7, "c₀ ⊊ c ⊊ ℓ∞   (all Banach with sup norm)",
            fontsize=11, color=C["dark"], ha="center", fontweight="bold")
    return save(fig, "02-normed-and-banach", "fa_v2_02_7_seq_spaces")


# ============================================================
# 03 hilbert-spaces
# ============================================================
def fa_v2_03_1_inner_product():
    fig, ax = newfig((8, 7))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-0.5, 5); ax.set_ylim(-0.5, 5)
    # vectors
    u = np.array([4, 1.5]); v = np.array([2, 3.5])
    ax.annotate("", xy=u, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=2.5))
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2.5))
    # projection of v onto u
    proj_len = u.dot(v) / u.dot(u)
    p = proj_len * u
    ax.annotate("", xy=p, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=2.0))
    ax.plot([v[0], p[0]], [v[1], p[1]], "--", color=C["gray"], lw=1.5)
    # angle arc
    th_u = np.arctan2(u[1], u[0]); th_v = np.arctan2(v[1], v[0])
    arc = Arc((0,0), 1.4, 1.4, angle=0, theta1=np.degrees(th_u), theta2=np.degrees(th_v),
              color=C["amber"], lw=2)
    ax.add_patch(arc)
    ax.text(0.95, 0.65, "θ", fontsize=13, color=C["amber"], fontweight="bold")
    ax.text(u[0]+0.1, u[1], "u", fontsize=13, color=C["red"], fontweight="bold")
    ax.text(v[0], v[1]+0.1, "v", fontsize=13, color=C["blue"], fontweight="bold")
    ax.text(p[0], p[1]-0.4, r"$\frac{\langle u,v\rangle}{\langle u,u\rangle}u$",
            fontsize=11, color=C["green"], fontweight="bold", ha="center")
    ax.text(2.5, 4.6, r"$\langle u,v\rangle = \|u\|\,\|v\|\cos\theta$",
            fontsize=12, color=C["dark"], ha="center")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.set_title("Inner Product: Geometry", fontsize=14, color=C["dark"], fontweight="bold")
    return save(fig, "03-hilbert-spaces", "fa_v2_03_1_inner_product")

def fa_v2_03_2_orthogonal_proj():
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(8, 6.5), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    # plane M: z = 0 spanned by e1, e2
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 8), np.linspace(-1.5, 2.5, 8))
    zz = 0*xx
    ax.plot_surface(xx, yy, zz, alpha=0.18, color=C["blue"], edgecolor=C["lightgray"])
    v = np.array([1.5, 1.0, 1.8])
    p = np.array([1.5, 1.0, 0])
    ax.quiver(0,0,0, v[0], v[1], v[2], color=C["red"], lw=2, arrow_length_ratio=0.12)
    ax.quiver(0,0,0, p[0], p[1], p[2], color=C["green"], lw=2, arrow_length_ratio=0.15)
    ax.plot([v[0], p[0]], [v[1], p[1]], [v[2], p[2]], "--", color=C["gray"], lw=1.5)
    ax.text(v[0]+0.1, v[1], v[2]+0.05, "x", color=C["red"], fontsize=13, fontweight="bold")
    ax.text(p[0], p[1]+0.1, -0.15, "P_M x", color=C["green"], fontsize=12, fontweight="bold")
    ax.text(2.5, -1.0, 0.05, "M (closed subspace)", color=C["blue"], fontsize=10)
    ax.set_title("Orthogonal Projection onto Closed Subspace", fontsize=13,
                 color=C["dark"], fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.5, 2.5); ax.set_zlim(0, 2.2)
    return save(fig, "03-hilbert-spaces", "fa_v2_03_2_orthogonal_proj")

def fa_v2_03_3_orthonormal_basis():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Orthonormal Basis (Hilbert)",
          r"$\langle e_i, e_j \rangle = \delta_{ij}$ and span dense")
    ax.text(5, 3.2, r"$x = \sum_n \langle x, e_n\rangle\, e_n$",
            fontsize=18, color=C["dark"], ha="center")
    ax.text(5, 2.3, "Fourier coefficients are inner products",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    box(ax, (1.0, 0.4), 3.6, 1.3, "complete\n(span dense)", C["green"])
    box(ax, (5.4, 0.4), 3.6, 1.3, "orthonormal\n(δᵢⱼ)", C["blue"])
    return save(fig, "03-hilbert-spaces", "fa_v2_03_3_orthonormal_basis")

def fa_v2_03_4_riesz():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Riesz Representation Theorem",
          "Every continuous linear functional comes from a vector")
    box(ax, (0.4, 2.0), 3.5, 1.6, "f ∈ H*\ncontinuous linear", C["amber"])
    box(ax, (6.1, 2.0), 3.5, 1.6, "y_f ∈ H\nunique", C["green"])
    arrow(ax, (3.9, 2.8), (6.1, 2.8), color=C["blue"], lw=2)
    ax.text(5, 3.3, r"$f(x) = \langle x, y_f\rangle$",
            fontsize=14, color=C["blue"], ha="center", fontweight="bold")
    ax.text(5, 1.0, "isometric anti-isomorphism H ≅ H*",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "03-hilbert-spaces", "fa_v2_03_4_riesz")

def fa_v2_03_5_parseval():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Parseval Identity", "Norm preserved by Fourier coefficients")
    ax.text(5, 3.4, r"$\|x\|^2 \;=\; \sum_n |\langle x,e_n\rangle|^2$",
            fontsize=18, color=C["dark"], ha="center")
    ax.text(5, 2.5, "energy conservation under orthonormal expansion",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    box(ax, (1.0, 0.6), 3.6, 1.3, "Bessel ≤", C["amber"])
    box(ax, (5.4, 0.6), 3.6, 1.3, "Parseval = (complete)", C["green"])
    return save(fig, "03-hilbert-spaces", "fa_v2_03_5_parseval")

def fa_v2_03_6_l2_basis():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    x = np.linspace(0, 1, 400)
    cols = [C["red"], C["amber"], C["blue"], C["purple"]]
    for k, col in zip([1, 2, 3, 4], cols):
        ax.plot(x, np.sqrt(2)*np.sin(k*np.pi*x), color=col, lw=1.8,
                label=f"√2 sin({k}πx)")
    ax.axhline(0, color=C["lightgray"], lw=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Orthonormal Basis of L²[0,1]: {√2 sin(kπx)}",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("x", color=C["gray"])
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "03-hilbert-spaces", "fa_v2_03_6_l2_basis")

def fa_v2_03_7_gram_schmidt():
    fig, ax = newfig((9, 7))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4)
    v1 = np.array([3, 1.0]); v2 = np.array([1.5, 2.5])
    u1 = v1 / np.linalg.norm(v1)
    proj = (v2.dot(u1)) * u1
    w2 = v2 - proj
    u2 = w2 / np.linalg.norm(w2)
    # original
    ax.annotate("", xy=v1, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=C["gray"], lw=1.6, ls="--"))
    ax.annotate("", xy=v2, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=C["gray"], lw=1.6, ls="--"))
    # u1, u2
    ax.annotate("", xy=u1*1.2, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=2.5))
    ax.annotate("", xy=u2*1.2, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2.5))
    # projection construction
    ax.plot([v2[0], proj[0]], [v2[1], proj[1]], "--", color=C["green"], lw=1.5)
    ax.annotate("", xy=proj, xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.5, ls=":"))
    ax.text(v1[0]+0.1, v1[1]-0.15, "v₁", fontsize=11, color=C["gray"])
    ax.text(v2[0]+0.05, v2[1], "v₂", fontsize=11, color=C["gray"])
    ax.text(u1[0]*1.3+0.05, u1[1]*1.3, "e₁", fontsize=12, color=C["red"], fontweight="bold")
    ax.text(u2[0]*1.3, u2[1]*1.3+0.1, "e₂", fontsize=12, color=C["blue"], fontweight="bold")
    ax.set_title("Gram–Schmidt Process", fontsize=14, color=C["dark"], fontweight="bold")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    return save(fig, "03-hilbert-spaces", "fa_v2_03_7_gram_schmidt")


# ============================================================
# 04 dual-spaces-hahn-banach
# ============================================================
def fa_v2_04_1_dual_geom():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Dual Space V*", "continuous linear functionals f: V → 𝕂")
    box(ax, (0.5, 1.8), 3.5, 1.6, "V\nBanach space", C["blue"])
    box(ax, (6.0, 1.8), 3.5, 1.6, "V*\ndual (Banach)", C["red"])
    arrow(ax, (4.0, 2.8), (6.0, 2.8), color=C["dark"], lw=2)
    ax.text(5, 3.2, "f", fontsize=14, color=C["dark"], ha="center", fontweight="bold")
    ax.text(5, 1.1, r"$\|f\|_* = \sup_{\|x\| \leq 1} |f(x)|$",
            fontsize=12, color=C["gray"], ha="center", style="italic")
    return save(fig, "04-dual-spaces-hahn-banach", "fa_v2_04_1_dual_geom")

def fa_v2_04_2_hb_extension():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Hahn–Banach Extension", "Extend a bounded functional preserving its norm")
    box(ax, (0.5, 1.8), 3.5, 1.6, "M ⊂ V\nf₀: M → 𝕂\nbounded", C["amber"])
    box(ax, (6.0, 1.8), 3.5, 1.6, "f: V → 𝕂\nf|M = f₀\n‖f‖ = ‖f₀‖", C["green"])
    arrow(ax, (4.0, 2.6), (6.0, 2.6), color=C["dark"], lw=2)
    ax.text(5, 3.0, "extend", fontsize=11, color=C["dark"], ha="center", fontweight="bold")
    ax.text(5, 1.0, "no constraint sacrificed: norm is preserved",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "04-dual-spaces-hahn-banach", "fa_v2_04_2_hb_extension")

def fa_v2_04_3_separation():
    fig, ax = newfig((8, 6))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-0.5, 6); ax.set_ylim(-0.5, 5)
    # convex set A (ellipse) and singleton b
    A = Ellipse((1.6, 2.0), 2.4, 1.6, color=C["blue"], alpha=0.3)
    ax.add_patch(A)
    A2 = Ellipse((1.6, 2.0), 2.4, 1.6, fill=False, edgecolor=C["blue"], lw=2)
    ax.add_patch(A2)
    ax.text(1.6, 2.0, "A (convex)", ha="center", color=C["blue"], fontweight="bold")
    B = Circle((4.6, 3.5), 0.18, color=C["red"], zorder=5)
    ax.add_patch(B)
    ax.text(4.7, 3.7, "b", color=C["red"], fontweight="bold", fontsize=12)
    # separating hyperplane
    xs = np.linspace(-0.5, 6, 50)
    ys = -0.5*xs + 4.0
    ax.plot(xs, ys, color=C["green"], lw=2.2, label=r"$\{x: f(x)=\alpha\}$")
    ax.legend(frameon=False, loc="lower left")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.set_title("Geometric Hahn–Banach: Separating Hyperplane",
                 fontsize=13, color=C["dark"], fontweight="bold")
    return save(fig, "04-dual-spaces-hahn-banach", "fa_v2_04_3_separation")

def fa_v2_04_4_reflexive():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Reflexivity: V → V**", "Canonical isometric embedding J: V → V**")
    box(ax, (0.4, 1.8), 2.6, 1.6, "V", C["blue"], fontsize=13)
    box(ax, (3.7, 1.8), 2.6, 1.6, "V*", C["amber"], fontsize=13)
    box(ax, (7.0, 1.8), 2.6, 1.6, "V**", C["red"], fontsize=13)
    arrow(ax, (3.0, 2.6), (3.7, 2.6))
    arrow(ax, (6.3, 2.6), (7.0, 2.6))
    arrow(ax, (1.7, 3.4), (8.3, 3.4), color=C["green"], lw=2)
    ax.text(5, 3.7, "J(x)(f) = f(x)", fontsize=11, color=C["green"], ha="center", fontweight="bold")
    ax.text(5, 0.9, "V reflexive ⇔ J surjective", fontsize=11, color=C["dark"],
            ha="center", style="italic", fontweight="bold")
    return save(fig, "04-dual-spaces-hahn-banach", "fa_v2_04_4_reflexive")

def fa_v2_04_5_weak_strong():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Weak vs Strong Convergence")
    box(ax, (0.4, 1.5), 4.2, 2.0, "", C["red"])
    ax.text(2.5, 3.0, "Strong (norm)", fontsize=12, color=C["red"], ha="center", fontweight="bold")
    ax.text(2.5, 2.1, r"$\|x_n - x\| \to 0$", fontsize=14, color=C["dark"], ha="center")
    box(ax, (5.4, 1.5), 4.2, 2.0, "", C["blue"])
    ax.text(7.5, 3.0, "Weak", fontsize=12, color=C["blue"], ha="center", fontweight="bold")
    ax.text(7.5, 2.1, r"$f(x_n) \to f(x),\;\forall f\in V^*$", fontsize=12,
            color=C["dark"], ha="center")
    arrow(ax, (4.6, 2.5), (5.4, 2.5), color=C["dark"])
    ax.text(5, 0.8, "strong ⇒ weak; converse fails in infinite dim",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    return save(fig, "04-dual-spaces-hahn-banach", "fa_v2_04_5_weak_strong")

def fa_v2_04_6_lp_dual():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Duality of ℓ^p", "(ℓ^p)* ≅ ℓ^q where 1/p + 1/q = 1, 1 ≤ p < ∞")
    pairs = [("ℓ¹", "ℓ∞", C["red"]), ("ℓ²", "ℓ²", C["blue"]), ("ℓ^p", "ℓ^q", C["green"])]
    for i, (a, b, col) in enumerate(pairs):
        x = 0.6 + i * 3.2
        box(ax, (x, 1.6), 1.2, 1.5, a, col, fontsize=14)
        box(ax, (x+1.6, 1.6), 1.2, 1.5, b, col, fontsize=14)
        ax.text(x+1.4, 2.35, "*", fontsize=14, color=C["dark"], ha="center", fontweight="bold")
    ax.text(5, 0.7, "Hölder pairing: ⟨x,y⟩ = Σ xₙ yₙ",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    return save(fig, "04-dual-spaces-hahn-banach", "fa_v2_04_6_lp_dual")

def fa_v2_04_7_supporting():
    fig, ax = newfig((8, 6))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
    # convex C
    th = np.linspace(0, 2*np.pi, 200)
    cx = 1.7 + 1.5*np.cos(th); cy = 2.0 + 1.0*np.sin(th)
    ax.fill(cx, cy, color=C["blue"], alpha=0.25)
    ax.plot(cx, cy, color=C["blue"], lw=2)
    # boundary point
    p = np.array([1.7 + 1.5*np.cos(0.6), 2.0 + 1.0*np.sin(0.6)])
    ax.plot(p[0], p[1], "o", color=C["red"], ms=9, zorder=5)
    # tangent supporting line
    n = np.array([np.cos(0.6)/1.5, np.sin(0.6)/1.0])
    n = n / np.linalg.norm(n)
    t = np.array([-n[1], n[0]])
    line_pts = np.array([p - 3*t, p + 3*t])
    ax.plot(line_pts[:, 0], line_pts[:, 1], color=C["green"], lw=2.2,
            label="supporting hyperplane")
    ax.legend(frameon=False, loc="upper right")
    ax.text(p[0]+0.15, p[1], "x₀", fontsize=11, color=C["red"], fontweight="bold")
    ax.text(1.7, 2.0, "C", color=C["blue"], fontsize=14, ha="center", fontweight="bold")
    ax.set_title("Supporting Hyperplane at Boundary Point",
                 fontsize=13, color=C["dark"], fontweight="bold")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    return save(fig, "04-dual-spaces-hahn-banach", "fa_v2_04_7_supporting")


# ============================================================
# 05 weak-topologies
# ============================================================
def fa_v2_05_1_weak_conv():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    x = np.linspace(0, 1, 800)
    for k, col in zip([2, 8, 24], [C["amber"], C["blue"], C["purple"]]):
        ax.plot(x, np.sin(k*np.pi*x), color=col, lw=1.4, alpha=0.85, label=f"sin({k}πx)")
    ax.axhline(0, color=C["red"], lw=2.5, label="weak limit = 0")
    ax.set_title("eₙ = sin(nπx) → 0 weakly in L²[0,1] (not strongly)",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_xlabel("x", color=C["gray"])
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "05-weak-topologies", "fa_v2_05_1_weak_conv")

def fa_v2_05_2_alaoglu():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Banach–Alaoglu Theorem",
          "The closed unit ball of V* is compact in the weak* topology")
    box(ax, (1.0, 1.5), 3.8, 2.0, "B_{V*} = {f : ‖f‖ ≤ 1}", C["blue"], fontsize=11)
    box(ax, (5.2, 1.5), 3.8, 2.0, "weak*-compact", C["green"], fontsize=12)
    arrow(ax, (4.8, 2.5), (5.2, 2.5), color=C["dark"], lw=2)
    ax.text(5, 0.8, "even though norm-compact only in finite dim",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "05-weak-topologies", "fa_v2_05_2_alaoglu")

def fa_v2_05_3_weak_star():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Weak* Topology on V*",
          "fₙ → f weakly* iff fₙ(x) → f(x) for every x ∈ V")
    ax.text(5, 3.0, r"$f_n \to f$ (weak*) $\;\Leftrightarrow\; f_n(x) \to f(x)\;\forall x\in V$",
            fontsize=14, color=C["dark"], ha="center")
    box(ax, (0.6, 0.6), 2.7, 1.4, "weak*\n(weakest)", C["red"])
    box(ax, (3.7, 0.6), 2.7, 1.4, "weak", C["amber"])
    box(ax, (6.8, 0.6), 2.7, 1.4, "norm\n(strongest)", C["blue"])
    return save(fig, "05-weak-topologies", "fa_v2_05_3_weak_star")

def fa_v2_05_4_compactness():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Compactness: Strong vs Weak vs Weak*")
    items = [
        ("Norm-compact", "infinite-dim:\nnever the unit ball", C["red"]),
        ("Weakly compact", "reflexive ⇒ ball weakly\ncompact (Kakutani)", C["amber"]),
        ("Weak*-compact", "always: Banach–Alaoglu", C["green"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        x = 0.4 + i*3.2
        box(ax, (x, 1.3), 2.8, 2.4, "", col)
        ax.text(x+1.4, 3.2, n, fontsize=11, color=col, fontweight="bold", ha="center")
        ax.text(x+1.4, 2.2, eq, fontsize=10, color=C["dark"], ha="center", style="italic")
    return save(fig, "05-weak-topologies", "fa_v2_05_4_compactness")

def fa_v2_05_5_weak_def():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Weak Topology", "Coarsest topology making every f ∈ V* continuous")
    ax.text(5, 3.3, r"$x_\alpha \to x$ (weak) $\;\Leftrightarrow\; f(x_\alpha)\to f(x),\;\forall f\in V^*$",
            fontsize=13, color=C["dark"], ha="center")
    ax.text(5, 2.4, "neighbourhood basis: U(x; f₁,…,fₙ; ε) = {y : |fᵢ(y−x)|<ε}",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    box(ax, (3.4, 0.8), 3.2, 1.2, "open sets defined by\nfinite collections of f's", C["blue"])
    return save(fig, "05-weak-topologies", "fa_v2_05_5_weak_def")

def fa_v2_05_6_seq_top():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Sequences vs Net Topology", "Weak topology not metrizable in infinite dim")
    box(ax, (0.4, 2.0), 4.2, 1.6, "Sequences\nsuffice in metric space", C["green"])
    box(ax, (5.4, 2.0), 4.2, 1.6, "Need nets in general\nweak topology (∞-dim)", C["red"])
    arrow(ax, (4.6, 2.8), (5.4, 2.8), color=C["dark"])
    ax.text(5, 0.8, "Eberlein–Šmulian: sequential closure works in weak topology of Banach",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, "05-weak-topologies", "fa_v2_05_6_seq_top")

def fa_v2_05_7_metrizability():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Metrizability of Weak/Weak* on Bounded Sets")
    box(ax, (0.4, 2.0), 4.2, 1.6, "V separable ⇒\n(B_{V*}, w*) metrizable", C["blue"])
    box(ax, (5.4, 2.0), 4.2, 1.6, "V* separable ⇒\n(B_V, weak) metrizable", C["amber"])
    ax.text(5, 0.7, "metric defined via countable dense set of test functionals",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, "05-weak-topologies", "fa_v2_05_7_metrizability")


# ============================================================
# 06 bounded-operators
# ============================================================
def fa_v2_06_1_op_norm():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Operator Norm")
    ax.text(5, 3.3, r"$\|T\| \;=\; \sup_{\|x\|\leq 1} \|Tx\|$",
            fontsize=18, color=C["dark"], ha="center")
    ax.text(5, 2.4, "T: V → W bounded ⇔ continuous ⇔ ‖T‖ < ∞",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    box(ax, (1.5, 0.6), 3.0, 1.4, "B(V,W)\nBanach if W Banach", C["green"])
    box(ax, (5.5, 0.6), 3.0, 1.4, "‖T₁T₂‖ ≤ ‖T₁‖‖T₂‖", C["amber"])
    return save(fig, "06-bounded-operators", "fa_v2_06_1_op_norm")

def fa_v2_06_2_three_thms():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Three Pillars (Banach Spaces)")
    items = [
        ("Uniform Bdd (UBP)", "pointwise ⇒ uniform\nbound", C["red"]),
        ("Open Mapping (OMT)", "surjective ⇒ open", C["amber"]),
        ("Closed Graph (CGT)", "graph closed ⇒\nbounded", C["blue"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        x = 0.4 + i*3.2
        box(ax, (x, 1.3), 2.8, 2.4, "", col)
        ax.text(x+1.4, 3.2, n, fontsize=11, color=col, fontweight="bold", ha="center")
        ax.text(x+1.4, 2.2, eq, fontsize=10, color=C["dark"], ha="center", style="italic")
    ax.text(5, 0.6, "all rest on Baire category theorem", fontsize=11,
            color=C["dark"], ha="center", fontweight="bold")
    return save(fig, "06-bounded-operators", "fa_v2_06_2_three_thms")

def fa_v2_06_3_ubp():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Uniform Boundedness Principle")
    ax.text(5, 3.5, r"$\sup_\alpha \|T_\alpha x\| < \infty\;\forall x \;\Rightarrow\; \sup_\alpha \|T_\alpha\| < \infty$",
            fontsize=13, color=C["dark"], ha="center")
    ax.text(5, 2.5, "(Banach–Steinhaus)", fontsize=11, color=C["gray"], ha="center", style="italic")
    box(ax, (1.0, 0.6), 3.6, 1.4, "Pointwise bdd", C["amber"])
    box(ax, (5.4, 0.6), 3.6, 1.4, "Uniformly bdd", C["green"])
    arrow(ax, (4.6, 1.3), (5.4, 1.3), color=C["dark"])
    return save(fig, "06-bounded-operators", "fa_v2_06_3_ubp")

def fa_v2_06_4_open_map():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Open Mapping Theorem",
          "T: V → W bounded surjective (Banach) ⇒ T is open")
    box(ax, (0.5, 1.8), 3.5, 1.6, "open U ⊂ V", C["amber"])
    box(ax, (6.0, 1.8), 3.5, 1.6, "T(U) open in W", C["green"])
    arrow(ax, (4.0, 2.6), (6.0, 2.6), color=C["dark"], lw=2)
    ax.text(5, 1.0, "consequence: bounded bijection ⇒ T⁻¹ bounded",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "06-bounded-operators", "fa_v2_06_4_open_map")

def fa_v2_06_5_closed_graph():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Closed Graph Theorem",
          "T: V → W between Banach. Graph closed ⇒ T bounded")
    ax.text(5, 3.4, r"$\Gamma(T) = \{(x, Tx)\} \subset V \times W$",
            fontsize=13, color=C["dark"], ha="center")
    box(ax, (1.0, 1.0), 3.6, 1.4, "Γ closed", C["amber"])
    box(ax, (5.4, 1.0), 3.6, 1.4, "T bounded", C["green"])
    arrow(ax, (4.6, 1.7), (5.4, 1.7), color=C["dark"])
    return save(fig, "06-bounded-operators", "fa_v2_06_5_closed_graph")

def fa_v2_06_6_examples():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Classical Bounded Operators on ℓ²")
    items = [
        ("Shift S", "(x₁,x₂,…) ↦ (0,x₁,x₂,…)", C["red"]),
        ("Multiplication", "(xₙ) ↦ (aₙ xₙ), bounded a", C["amber"]),
        ("Convolution", "T_a x = a * x in ℓ²(ℤ)", C["blue"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        y = 3.7 - i*1.4
        box(ax, (0.6, y), 9.0, 1.0, "", col)
        ax.text(1.0, y+0.5, n, fontsize=11, color=col, fontweight="bold", ha="left", va="center")
        ax.text(5.0, y+0.5, eq, fontsize=11, color=C["dark"], ha="center", va="center", style="italic")
    return save(fig, "06-bounded-operators", "fa_v2_06_6_examples")

def fa_v2_06_7_inverse():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Bounded Inverse Theorem")
    ax.text(5, 3.5, r"$T \in B(V,W)\text{ bijection (Banach)} \Rightarrow T^{-1} \in B(W,V)$",
            fontsize=12, color=C["dark"], ha="center")
    box(ax, (1.0, 1.0), 3.6, 1.4, "T bijection\nbounded", C["amber"])
    box(ax, (5.4, 1.0), 3.6, 1.4, "T⁻¹ bounded", C["green"])
    arrow(ax, (4.6, 1.7), (5.4, 1.7), color=C["dark"])
    return save(fig, "06-bounded-operators", "fa_v2_06_7_inverse")


# ============================================================
# 07 compact-operators
# ============================================================
def fa_v2_07_1_compact_def():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Compact Operator", "T(B) is relatively compact for B bounded")
    box(ax, (0.5, 1.8), 3.5, 1.6, "B ⊂ V\nbounded", C["amber"])
    box(ax, (6.0, 1.8), 3.5, 1.6, "T(B) ⊂ W\ntotally bounded", C["green"])
    arrow(ax, (4.0, 2.6), (6.0, 2.6), color=C["blue"], lw=2)
    ax.text(5, 3.0, "T", fontsize=14, color=C["blue"], ha="center", fontweight="bold")
    ax.text(5, 1.0, "= every bounded sequence has a Cauchy image-subsequence",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "07-compact-operators", "fa_v2_07_1_compact_def")

def fa_v2_07_2_finite_rank():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Finite-Rank Operators",
          "rank(T) < ∞: T(V) ⊂ finite-dim subspace")
    box(ax, (0.4, 2.0), 4.2, 1.6, "Finite-rank", C["amber"])
    box(ax, (5.4, 2.0), 4.2, 1.6, "Compact (closure)", C["green"])
    arrow(ax, (4.6, 2.8), (5.4, 2.8), color=C["dark"], lw=2)
    ax.text(5, 0.8, "in Hilbert space: K(H) = closure of finite-rank in operator norm",
            fontsize=10, color=C["red"], ha="center", style="italic")
    return save(fig, "07-compact-operators", "fa_v2_07_2_finite_rank")

def fa_v2_07_3_spectral_compact():
    fig, ax = newfig((9, 6))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    # eigenvalues converging to 0
    lam = [1.4, 0.9, 0.55, 0.3, 0.15, 0.08, 0.04]
    ax.plot(lam, [0]*len(lam), "o", color=C["blue"], ms=10, zorder=5)
    ax.plot(0, 0, "X", color=C["red"], ms=14, zorder=6)
    ax.axhline(0, color=C["lightgray"], lw=0.6)
    ax.axvline(0, color=C["lightgray"], lw=0.6)
    for L in lam:
        ax.plot([L, L], [-0.05, 0.05], color=C["dark"], lw=1)
    ax.text(0.05, 0.2, "0 (only limit)", fontsize=10, color=C["red"], fontweight="bold")
    ax.text(1.4, -0.3, "λ₁", fontsize=10, color=C["blue"], ha="center")
    ax.text(0.9, -0.3, "λ₂", fontsize=10, color=C["blue"], ha="center")
    ax.set_title("Spectrum of Compact Self-Adjoint Operator: λₙ → 0",
                 fontsize=13, color=C["dark"], fontweight="bold")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    return save(fig, "07-compact-operators", "fa_v2_07_3_spectral_compact")

def fa_v2_07_4_fredholm():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Fredholm Alternative",
          "T = I − K compact: kernel and cokernel finite, equal dim")
    box(ax, (0.4, 2.0), 4.2, 1.6, "ker(I − K)\nfinite-dim", C["amber"])
    box(ax, (5.4, 2.0), 4.2, 1.6, "coker(I − K)\nfinite-dim", C["blue"])
    ax.text(5, 0.8, "either Tx=0 has only trivial solution, or it has finitely many",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "07-compact-operators", "fa_v2_07_4_fredholm")

def fa_v2_07_5_hilbert_schmidt():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Hilbert–Schmidt Operators",
          r"$\|T\|_{HS}^2 = \sum_{i,j} |\langle Te_j, e_i\rangle|^2 < \infty$")
    ax.text(5, 3.0, "K(x,y) ∈ L²(Ω×Ω) ⇒  Tf(x) = ∫ K(x,y) f(y) dy is HS",
            fontsize=11, color=C["dark"], ha="center")
    box(ax, (1.0, 0.8), 3.6, 1.3, "Trace class\n⊂ HS", C["red"])
    box(ax, (5.4, 0.8), 3.6, 1.3, "HS ⊂ Compact", C["green"])
    return save(fig, "07-compact-operators", "fa_v2_07_5_hilbert_schmidt")

def fa_v2_07_6_eigenvalue_decay():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    n = np.arange(1, 30)
    lam_compact = 1.0 / n
    lam_trace = 1.0 / n**2
    lam_hs = 1.0 / n**1.5
    ax.semilogy(n, lam_compact, "o-", color=C["blue"], lw=1.4, label="compact: 1/n")
    ax.semilogy(n, lam_hs, "s-", color=C["amber"], lw=1.4, label="HS: 1/n^{1.5}")
    ax.semilogy(n, lam_trace, "^-", color=C["red"], lw=1.4, label="trace: 1/n²")
    ax.set_title("Eigenvalue Decay Rates by Operator Class",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("n", color=C["gray"])
    ax.set_ylabel("|λₙ|", color=C["gray"])
    ax.legend(frameon=False)
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.4, lw=0.5)
    return save(fig, "07-compact-operators", "fa_v2_07_6_eigenvalue_decay")

def fa_v2_07_7_compact_examples():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Classical Compact Operators")
    items = [
        ("Integral op", "Tf(x)=∫ K(x,y)f(y)dy, K ∈ L²", C["red"]),
        ("Sobolev embedding", "H^s ↪ L² compact (bdd domain)", C["amber"]),
        ("Multiplication", "(aₙ) → 0 on ℓ², compact mult.", C["blue"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        y = 3.7 - i*1.4
        box(ax, (0.6, y), 9.0, 1.0, "", col)
        ax.text(1.0, y+0.5, n, fontsize=11, color=col, fontweight="bold", ha="left", va="center")
        ax.text(5.0, y+0.5, eq, fontsize=10, color=C["dark"], ha="center", va="center", style="italic")
    return save(fig, "07-compact-operators", "fa_v2_07_7_compact_examples")


# ============================================================
# 08 spectral-theory
# ============================================================
def fa_v2_08_1_spectrum_decomp():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Spectrum Decomposition",
          "σ(T) = σ_p(T) ∪ σ_c(T) ∪ σ_r(T)")
    items = [
        ("Point σ_p", "T − λI not injective\n(eigenvalues)", C["red"]),
        ("Continuous σ_c", "injective, dense range\nbut not surjective", C["amber"]),
        ("Residual σ_r", "injective, range\nnot dense", C["blue"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        x = 0.4 + i*3.2
        box(ax, (x, 1.3), 2.8, 2.4, "", col)
        ax.text(x+1.4, 3.2, n, fontsize=11, color=col, fontweight="bold", ha="center")
        ax.text(x+1.4, 2.2, eq, fontsize=10, color=C["dark"], ha="center", style="italic")
    return save(fig, "08-spectral-theory", "fa_v2_08_1_spectrum_decomp")

def fa_v2_08_2_resolvent():
    fig, ax = newfig((9, 6))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5)
    # spectrum: disk
    sp = Circle((0.4, 0), 1.2, color=C["red"], alpha=0.3, label="σ(T)")
    ax.add_patch(sp)
    sp2 = Circle((0.4, 0), 1.2, fill=False, edgecolor=C["red"], lw=2)
    ax.add_patch(sp2)
    # resolvent set
    ax.add_patch(Rectangle((-3, -2.5), 6, 5, fill=False, edgecolor=C["green"], lw=2))
    ax.text(-2.3, 2.0, "ρ(T) = ℂ \\ σ(T)", color=C["green"], fontsize=12, fontweight="bold")
    ax.text(0.4, 0, "σ(T)", color=C["red"], fontsize=12, fontweight="bold", ha="center")
    ax.axhline(0, color=C["lightgray"], lw=0.5)
    ax.axvline(0, color=C["lightgray"], lw=0.5)
    ax.set_title("Resolvent Set and Spectrum in ℂ",
                 fontsize=13, color=C["dark"], fontweight="bold")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    return save(fig, "08-spectral-theory", "fa_v2_08_2_resolvent")

def fa_v2_08_3_spectral_radius():
    fig, ax = newfig((9, 6))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    # spectrum points
    pts = [(1.4, 0.8), (-0.6, 1.5), (-1.2, -0.4), (0.5, -1.2), (0.2, 0.3)]
    for p in pts:
        ax.plot(p[0], p[1], "o", color=C["red"], ms=8, zorder=5)
    # spectral radius circle
    r = max(np.linalg.norm(p) for p in pts)
    circ = Circle((0,0), r, fill=False, edgecolor=C["blue"], lw=2.5, ls="--")
    ax.add_patch(circ)
    ax.axhline(0, color=C["lightgray"], lw=0.5)
    ax.axvline(0, color=C["lightgray"], lw=0.5)
    ax.text(0, r+0.2, f"r(T) = {r:.2f}", color=C["blue"], fontsize=12,
            ha="center", fontweight="bold")
    ax.text(2, -2.5, r"$r(T) = \lim_n \|T^n\|^{1/n}$",
            color=C["dark"], fontsize=12, ha="right", fontweight="bold")
    ax.set_title("Spectral Radius", fontsize=13, color=C["dark"], fontweight="bold")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    return save(fig, "08-spectral-theory", "fa_v2_08_3_spectral_radius")

def fa_v2_08_4_self_adjoint():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Self-Adjoint Operator", "Spectrum is real, A = A*")
    ax.text(5, 3.4, r"$\langle Ax, y\rangle = \langle x, Ay\rangle\;\Rightarrow\;\sigma(A)\subset\mathbb{R}$",
            fontsize=13, color=C["dark"], ha="center")
    box(ax, (1.0, 1.4), 3.6, 1.5, "A = A*", C["green"])
    box(ax, (5.4, 1.4), 3.6, 1.5, "σ(A) ⊂ ℝ", C["blue"])
    arrow(ax, (4.6, 2.15), (5.4, 2.15), color=C["dark"])
    ax.text(5, 0.7, "physics: observables ↔ self-adjoint operators",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "08-spectral-theory", "fa_v2_08_4_self_adjoint")

def fa_v2_08_5_spectral_thm():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Spectral Theorem",
          "Self-adjoint A ↔ unique projection-valued measure E")
    ax.text(5, 3.4, r"$A = \int_{\sigma(A)} \lambda\, dE(\lambda)$",
            fontsize=18, color=C["dark"], ha="center")
    ax.text(5, 2.5, "diagonalisation in continuous spectrum case",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    box(ax, (1.0, 0.6), 3.6, 1.4, "compact A:\nΣₙ λₙ ⟨·, eₙ⟩ eₙ", C["amber"])
    box(ax, (5.4, 0.6), 3.6, 1.4, "general:\n∫ λ dE", C["green"])
    return save(fig, "08-spectral-theory", "fa_v2_08_5_spectral_thm")

def fa_v2_08_6_func_calc():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Functional Calculus",
          "Define f(A) for measurable f on σ(A)")
    ax.text(5, 3.4, r"$f(A) = \int f(\lambda)\, dE(\lambda)$",
            fontsize=16, color=C["dark"], ha="center")
    items = [("polynomial", C["red"]), ("continuous", C["amber"]),
             ("Borel", C["blue"])]
    for i, (n, col) in enumerate(items):
        x = 0.6 + i * 3.2
        box(ax, (x, 0.7), 2.8, 1.3, n, col)
    arrow(ax, (3.4, 1.35), (3.8, 1.35), color=C["dark"])
    arrow(ax, (6.6, 1.35), (7.0, 1.35), color=C["dark"])
    return save(fig, "08-spectral-theory", "fa_v2_08_6_func_calc")

def fa_v2_08_7_examples():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Spectra of Classical Operators")
    items = [
        ("Mult by x on L²[0,1]", "σ = [0,1] (continuous)", C["red"]),
        ("Right shift on ℓ²", "σ = closed unit disk", C["amber"]),
        ("−Δ on bdd domain", "{λₙ} discrete, λₙ → ∞", C["blue"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        y = 3.7 - i*1.4
        box(ax, (0.6, y), 9.0, 1.0, "", col)
        ax.text(1.0, y+0.5, n, fontsize=11, color=col, fontweight="bold", ha="left", va="center")
        ax.text(6.5, y+0.5, eq, fontsize=10, color=C["dark"], ha="center", va="center", style="italic")
    return save(fig, "08-spectral-theory", "fa_v2_08_7_examples")


# ============================================================
# 09 unbounded-operators
# ============================================================
def fa_v2_09_1_unbounded_domain():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Unbounded Operator with Dense Domain",
          "T: D(T) ⊂ V → W, D(T) dense")
    box(ax, (0.5, 1.6), 3.5, 2.0, "D(T) ⊂ V\ndense (e.g. C_c^∞)", C["amber"])
    box(ax, (6.0, 1.6), 3.5, 2.0, "Range\nin W", C["green"])
    arrow(ax, (4.0, 2.6), (6.0, 2.6), color=C["dark"], lw=2)
    ax.text(5, 0.8, "differential operators are typically unbounded",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "09-unbounded-operators", "fa_v2_09_1_unbounded_domain")

def fa_v2_09_2_closed_op():
    fig, ax = newfig((9, 6))
    ax.set_facecolor(BG)
    ax.set_xlim(-2, 5); ax.set_ylim(-2, 5)
    # graph of operator
    x = np.linspace(0, 4, 100)
    y = x**2 / 4
    ax.plot(x, y, color=C["blue"], lw=2.2, label="Γ(T) = {(x, Tx)}")
    ax.fill_between(x, y, y+0.6, color=C["blue"], alpha=0.1)
    ax.plot([0.5], [0.5**2/4], "o", color=C["red"], ms=8)
    ax.annotate("(xₙ, Txₙ)", (0.5, 0.5**2/4), xytext=(-1, 1),
                fontsize=10, color=C["red"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["red"]))
    ax.set_xlabel("V", color=C["gray"])
    ax.set_ylabel("W", color=C["gray"])
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Closed Operator: Graph Γ(T) Closed in V × W",
                 fontsize=13, color=C["dark"], fontweight="bold")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "09-unbounded-operators", "fa_v2_09_2_closed_op")

def fa_v2_09_3_sym_vs_sa():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Symmetric vs Self-Adjoint", "Domain matters: T ⊂ T*")
    box(ax, (0.4, 2.0), 4.2, 1.6, "Symmetric\nT ⊂ T* (D(T)⊂D(T*))", C["amber"])
    box(ax, (5.4, 2.0), 4.2, 1.6, "Self-adjoint\nT = T* (equal domains)", C["green"])
    arrow(ax, (4.6, 2.8), (5.4, 2.8), color=C["dark"])
    ax.text(5, 0.8, "self-adjoint ⇒ real spectrum + spectral theorem",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "09-unbounded-operators", "fa_v2_09_3_sym_vs_sa")

def fa_v2_09_4_laplacian():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    x = np.linspace(0, 1, 200)
    for n, col in zip([1, 2, 3, 4], [C["red"], C["amber"], C["blue"], C["purple"]]):
        ax.plot(x, np.sqrt(2)*np.sin(n*np.pi*x), color=col, lw=1.6,
                label=f"φ_{n}, λ_{n} = (nπ)²")
    ax.set_title("Eigenfunctions of −Δ on [0,1] (Dirichlet)",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("x", color=C["gray"])
    ax.legend(frameon=False, fontsize=9)
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "09-unbounded-operators", "fa_v2_09_4_laplacian")

def fa_v2_09_5_friedrichs():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Friedrichs Extension",
          "Semi-bounded symmetric operator → canonical self-adjoint")
    box(ax, (0.5, 1.8), 3.5, 1.6, "T ≥ 0\nsymmetric", C["amber"])
    box(ax, (6.0, 1.8), 3.5, 1.6, "T_F\nself-adjoint", C["green"])
    arrow(ax, (4.0, 2.6), (6.0, 2.6), color=C["dark"], lw=2)
    ax.text(5, 3.0, "Friedrichs", fontsize=11, color=C["dark"], ha="center", fontweight="bold")
    ax.text(5, 1.0, "uses associated quadratic form completion",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "09-unbounded-operators", "fa_v2_09_5_friedrichs")

def fa_v2_09_6_essential():
    fig, ax = newfig((9, 6))
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_xlim(-3, 3); ax.set_ylim(-1, 1)
    # discrete eigenvalues + essential spectrum interval
    ax.plot([-2, -1.4, -0.9], [0, 0, 0], "o", color=C["red"], ms=10, zorder=5)
    ax.plot([0, 2], [0, 0], color=C["blue"], lw=8, alpha=0.6, zorder=4)
    ax.text(-1.4, 0.25, "discrete\nσ_d", color=C["red"], fontsize=11,
            ha="center", fontweight="bold")
    ax.text(1.0, 0.25, "essential σ_ess", color=C["blue"], fontsize=11,
            ha="center", fontweight="bold")
    ax.axhline(0, color=C["lightgray"], lw=0.5)
    ax.set_title("Essential Spectrum vs Discrete Spectrum",
                 fontsize=13, color=C["dark"], fontweight="bold")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.set_xlabel("ℝ", color=C["gray"])
    ax.set_yticks([])
    ax.tick_params(colors=C["gray"], labelsize=8)
    return save(fig, "09-unbounded-operators", "fa_v2_09_6_essential")

def fa_v2_09_7_qm_examples():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Quantum Mechanical Operators (Unbounded)")
    items = [
        ("Position X", "(Xψ)(x)=xψ(x), σ = ℝ", C["red"]),
        ("Momentum P", "P = −iℏ d/dx, σ = ℝ", C["amber"]),
        ("Hamiltonian H", "H = −ℏ²Δ/2m + V(x)", C["blue"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        y = 3.7 - i*1.4
        box(ax, (0.6, y), 9.0, 1.0, "", col)
        ax.text(1.0, y+0.5, n, fontsize=11, color=col, fontweight="bold", ha="left", va="center")
        ax.text(6.0, y+0.5, eq, fontsize=11, color=C["dark"], ha="center", va="center", style="italic")
    return save(fig, "09-unbounded-operators", "fa_v2_09_7_qm_examples")


# ============================================================
# 10 semigroups
# ============================================================
def fa_v2_10_1_semigroup_orbit():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    t = np.linspace(0, 4, 400)
    for x0, col, lbl in [(1.0, C["red"], "x₀=1"), (0.5, C["amber"], "x₀=0.5"),
                         (-0.7, C["blue"], "x₀=−0.7")]:
        ax.plot(t, x0 * np.exp(-0.6*t), color=col, lw=2, label=lbl)
    ax.set_title("C₀ Semigroup Orbits: T(t)x₀ = e^{tA}x₀",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("t", color=C["gray"])
    ax.set_ylabel("T(t)x₀", color=C["gray"])
    ax.legend(frameon=False)
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.axhline(0, color=C["lightgray"], lw=0.5)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "10-semigroups", "fa_v2_10_1_semigroup_orbit")

def fa_v2_10_2_generator():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Infinitesimal Generator", "A = lim_{t→0⁺} (T(t)−I)/t")
    ax.text(5, 3.4, r"$\frac{d}{dt} T(t)x = AT(t)x = T(t)Ax$",
            fontsize=14, color=C["dark"], ha="center")
    box(ax, (1.0, 1.4), 3.6, 1.5, "T(t)\nsemigroup", C["amber"])
    box(ax, (5.4, 1.4), 3.6, 1.5, "A\ngenerator", C["green"])
    arrow(ax, (4.6, 2.15), (5.4, 2.15), color=C["dark"])
    ax.text(5, 0.7, "T(t) ↔ A bijection (Hille–Yosida)",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "10-semigroups", "fa_v2_10_2_generator")

def fa_v2_10_3_hille_yosida():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Hille–Yosida Theorem",
          "Generator characterization for contraction semigroups")
    box(ax, (0.4, 1.3), 4.2, 2.4, "A: D(A) ⊂ X → X", C["amber"], fontsize=11)
    box(ax, (5.4, 1.3), 4.2, 2.4, "T(t) contraction\nsemigroup", C["green"], fontsize=11)
    arrow(ax, (4.6, 2.5), (5.4, 2.5), color=C["dark"], lw=2)
    ax.text(5, 0.7, "(0,∞) ⊂ ρ(A), ‖(λI−A)⁻¹‖ ≤ 1/λ, A densely defined and closed",
            fontsize=10, color=C["red"], ha="center", style="italic")
    return save(fig, "10-semigroups", "fa_v2_10_3_hille_yosida")

def fa_v2_10_4_heat_eq():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    x = np.linspace(0, 1, 400)
    u0 = np.where(np.abs(x-0.5) < 0.15, 1.0, 0.0)
    ax.plot(x, u0, color=C["dark"], lw=2.2, label="t = 0 (initial)")
    for t, col in zip([0.005, 0.02, 0.08], [C["red"], C["amber"], C["blue"]]):
        u = np.zeros_like(x)
        for n in range(1, 60):
            cn = 2*np.trapz(u0*np.sin(n*np.pi*x), x)
            u += cn * np.exp(-(n*np.pi)**2 * t) * np.sin(n*np.pi*x)
        ax.plot(x, u, color=col, lw=1.8, label=f"t = {t}")
    ax.set_title("Heat Equation on [0,1]: u(x,t) = e^{tΔ} u₀",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("x", color=C["gray"])
    ax.set_ylabel("u(x,t)", color=C["gray"])
    ax.legend(frameon=False)
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "10-semigroups", "fa_v2_10_4_heat_eq")

def fa_v2_10_5_evolution():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Time Evolution", "u(t) = T(t) u₀ solves du/dt = Au")
    arrow(ax, (1.0, 2.5), (9.0, 2.5), color=C["dark"], lw=2)
    for i, t in enumerate([1.0, 3.0, 5.0, 7.0, 9.0]):
        ax.plot(t, 2.5, "o", color=C["blue"], ms=10, zorder=5)
        ax.text(t, 2.0, f"t={i}", color=C["blue"], ha="center", fontsize=10)
    ax.text(5, 3.4, "T(t+s) = T(t) T(s),  T(0) = I",
            fontsize=12, color=C["dark"], ha="center", fontweight="bold")
    ax.text(5, 1.0, "deterministic flow on Banach space",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "10-semigroups", "fa_v2_10_5_evolution")

def fa_v2_10_6_resolvent_repr():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Resolvent Representation",
          "Laplace transform: R(λ,A) = ∫₀^∞ e^{−λt} T(t) dt")
    ax.text(5, 3.0, r"$(\lambda I - A)^{-1} \;=\; \int_0^\infty e^{-\lambda t} T(t)\,dt$",
            fontsize=14, color=C["dark"], ha="center")
    box(ax, (1.0, 0.8), 3.6, 1.4, "semigroup T(t)", C["amber"])
    box(ax, (5.4, 0.8), 3.6, 1.4, "resolvent R(λ,A)", C["green"])
    arrow(ax, (4.6, 1.5), (5.4, 1.5), color=C["dark"])
    return save(fig, "10-semigroups", "fa_v2_10_6_resolvent_repr")

def fa_v2_10_7_examples():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Classical Semigroups")
    items = [
        ("Translation", "T(t)f(x) = f(x+t)", C["red"]),
        ("Heat", "T(t) = e^{tΔ}", C["amber"]),
        ("Schrödinger", "U(t) = e^{−itH}", C["blue"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        y = 3.7 - i*1.4
        box(ax, (0.6, y), 9.0, 1.0, "", col)
        ax.text(1.0, y+0.5, n, fontsize=11, color=col, fontweight="bold", ha="left", va="center")
        ax.text(6.0, y+0.5, eq, fontsize=11, color=C["dark"], ha="center", va="center", style="italic")
    return save(fig, "10-semigroups", "fa_v2_10_7_examples")


# ============================================================
# 11 distributions-sobolev
# ============================================================
def fa_v2_11_1_test_functions():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    x = np.linspace(-1.5, 1.5, 600)
    bump = np.where(np.abs(x) < 1, np.exp(-1.0 / np.maximum(1 - x**2, 1e-10)), 0)
    bump = np.where(np.abs(x) < 1, bump, 0)
    ax.plot(x, bump, color=C["blue"], lw=2.2, label=r"$\phi(x)=e^{-1/(1-x^2)}\chi_{|x|<1}$")
    ax.fill_between(x, 0, bump, color=C["blue"], alpha=0.18)
    ax.axvline(-1, color=C["red"], ls="--", lw=1, alpha=0.6)
    ax.axvline(1, color=C["red"], ls="--", lw=1, alpha=0.6)
    ax.text(-1, 0.05, "−1", color=C["red"], fontsize=10, ha="right")
    ax.text(1, 0.05, "1", color=C["red"], fontsize=10, ha="left")
    ax.set_title(r"Test Function $\phi \in C_c^\infty(\mathbb{R})$ (Bump)",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("x", color=C["gray"])
    ax.legend(frameon=False, loc="upper right")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "11-distributions-sobolev", "fa_v2_11_1_test_functions")

def fa_v2_11_2_distributions():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    x = np.linspace(-2, 2, 400)
    # Gaussian approx of delta
    for sig, col, lbl in [(0.5, C["amber"], "σ=0.5"), (0.2, C["blue"], "σ=0.2"),
                          (0.08, C["red"], "σ=0.08")]:
        ax.plot(x, np.exp(-x**2/(2*sig**2)) / (sig*np.sqrt(2*np.pi)),
                color=col, lw=1.8, label=lbl)
    ax.annotate("", xy=(0, 4.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["dark"], lw=3))
    ax.text(0.05, 4.6, "δ", fontsize=14, color=C["dark"], fontweight="bold")
    ax.set_title("Dirac δ as Limit of Distributions",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("x", color=C["gray"])
    ax.legend(frameon=False)
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "11-distributions-sobolev", "fa_v2_11_2_distributions")

def fa_v2_11_3_weak_deriv():
    fig, ax = newfig((9, 5.5))
    ax.set_facecolor(BG)
    x = np.linspace(-2, 2, 400)
    f = np.abs(x)
    df_classical = np.where(x>0, 1, np.where(x<0, -1, np.nan))
    ax.plot(x, f, color=C["blue"], lw=2.2, label="f(x) = |x|")
    ax.plot(x, df_classical, color=C["red"], lw=2.2, label="weak derivative sign(x)")
    ax.set_title("Weak Derivative of |x| (Discontinuous Classical)",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("x", color=C["gray"])
    ax.legend(frameon=False)
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    return save(fig, "11-distributions-sobolev", "fa_v2_11_3_weak_deriv")

def fa_v2_11_4_sobolev_chain():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Sobolev Embeddings", "H^s ↪ H^{s'} for s > s'")
    chain = [("H²", C["red"]), ("H¹", C["amber"]), ("L²", C["green"]),
             ("H⁻¹", C["blue"]), ("H⁻²", C["purple"])]
    for i, (n, col) in enumerate(chain):
        x = 0.5 + i * 1.95
        box(ax, (x, 2.0), 1.4, 1.4, n, col, fontsize=14)
        if i < 4:
            arrow(ax, (x+1.4, 2.7), (x+1.95, 2.7), color=C["dark"])
    ax.text(5, 0.8, "H^s(Ω) compactly embedded in H^{s−1}(Ω) on bounded domain",
            fontsize=11, color=C["dark"], ha="center", style="italic")
    return save(fig, "11-distributions-sobolev", "fa_v2_11_4_sobolev_chain")

def fa_v2_11_5_trace():
    fig, ax = newfig((9, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Trace Theorem", "γ: H¹(Ω) → H^{1/2}(∂Ω) bounded")
    box(ax, (0.5, 1.8), 3.5, 1.6, "u ∈ H¹(Ω)", C["amber"])
    box(ax, (6.0, 1.8), 3.5, 1.6, "u|_{∂Ω} ∈ H^{1/2}(∂Ω)", C["green"])
    arrow(ax, (4.0, 2.6), (6.0, 2.6), color=C["dark"], lw=2)
    ax.text(5, 3.0, "γ", fontsize=14, color=C["dark"], ha="center", fontweight="bold")
    ax.text(5, 1.0, "boundary values for non-continuous Sobolev functions",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "11-distributions-sobolev", "fa_v2_11_5_trace")

def fa_v2_11_6_dual_sobolev():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Negative Sobolev Spaces",
          "H^{−s}(Ω) = (H₀^s(Ω))*  contains δ when s > n/2")
    ax.text(5, 3.4, r"$H^{-s} = \overline{L^2}^{\|\cdot\|_{-s}}$", fontsize=14,
            color=C["dark"], ha="center")
    box(ax, (1.0, 1.4), 3.6, 1.5, "H^s_0(Ω)", C["amber"])
    box(ax, (5.4, 1.4), 3.6, 1.5, "H^{−s}(Ω) = dual", C["green"])
    arrow(ax, (4.6, 2.15), (5.4, 2.15), color=C["dark"])
    ax.text(5, 0.7, "δ ∈ H^{−s}(ℝⁿ) for s > n/2",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "11-distributions-sobolev", "fa_v2_11_6_dual_sobolev")

def fa_v2_11_7_examples():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Classical Distributions")
    items = [
        ("Dirac δ", "⟨δ, φ⟩ = φ(0)", C["red"]),
        ("Heaviside H", "H'= δ in distribution sense", C["amber"]),
        ("Principal value", "PV(1/x): integrable cancellation", C["blue"]),
    ]
    for i, (n, eq, col) in enumerate(items):
        y = 3.7 - i*1.4
        box(ax, (0.6, y), 9.0, 1.0, "", col)
        ax.text(1.0, y+0.5, n, fontsize=11, color=col, fontweight="bold", ha="left", va="center")
        ax.text(5.5, y+0.5, eq, fontsize=11, color=C["dark"], ha="center", va="center", style="italic")
    return save(fig, "11-distributions-sobolev", "fa_v2_11_7_examples")


# ============================================================
# 12 applications-pde-qm
# ============================================================
def fa_v2_12_1_lax_milgram():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Lax–Milgram Theorem",
          "Coercive bounded bilinear form ⇒ unique weak solution")
    box(ax, (0.4, 2.5), 4.2, 1.4, "a(u,v) bilinear,\nbounded + coercive", C["amber"])
    box(ax, (5.4, 2.5), 4.2, 1.4, "∀ f ∈ H*: ∃! u\na(u,v)=⟨f,v⟩", C["green"])
    arrow(ax, (4.6, 3.2), (5.4, 3.2), color=C["dark"], lw=2)
    ax.text(5, 1.4, "‖u‖ ≤ (1/α)‖f‖_{H*}", fontsize=12, color=C["red"],
            ha="center", fontweight="bold")
    ax.text(5, 0.7, "foundation for finite element methods",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, "12-applications-pde-qm", "fa_v2_12_1_lax_milgram")

def fa_v2_12_2_variational():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Variational Formulation",
          "−Δu = f ⇔ ∫∇u·∇v = ∫fv ∀v ∈ H¹₀")
    box(ax, (0.4, 2.0), 4.2, 1.6, "Strong form\n−Δu = f, u|∂Ω = 0", C["amber"])
    box(ax, (5.4, 2.0), 4.2, 1.6, "Weak form\na(u,v) = (f,v)", C["green"])
    arrow(ax, (4.6, 2.8), (5.4, 2.8), color=C["dark"], lw=2)
    ax.text(5, 0.8, "weak formulation needs only u ∈ H¹₀ — opens Hilbert toolkit",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "12-applications-pde-qm", "fa_v2_12_2_variational")

def fa_v2_12_3_qm_observables():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Quantum Observables ↔ Self-Adjoint Operators")
    box(ax, (0.4, 2.0), 4.2, 1.6, "Observable A\n(physical measurement)", C["amber"])
    box(ax, (5.4, 2.0), 4.2, 1.6, "Self-adjoint A on H\nA = A*", C["green"])
    arrow(ax, (4.6, 2.8), (5.4, 2.8), color=C["dark"], lw=2)
    ax.text(5, 0.8, "spectrum σ(A) = possible measurement outcomes",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "12-applications-pde-qm", "fa_v2_12_3_qm_observables")

def fa_v2_12_4_stone():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Stone's Theorem",
          "Self-adjoint A ↔ strongly continuous unitary group U(t)")
    box(ax, (0.4, 2.0), 4.2, 1.6, "A self-adjoint", C["amber"])
    box(ax, (5.4, 2.0), 4.2, 1.6, "U(t) = e^{itA}\nunitary group", C["green"])
    arrow(ax, (4.6, 2.8), (5.4, 2.8), color=C["dark"], lw=2)
    ax.text(5, 0.8, "quantum time evolution: ψ(t) = e^{−itH/ℏ} ψ₀",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "12-applications-pde-qm", "fa_v2_12_4_stone")

def fa_v2_12_5_schrodinger():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Schrödinger Equation as Semigroup",
          "iℏ ∂_t ψ = Hψ,  ψ(t) = e^{−itH/ℏ} ψ₀")
    ax.text(5, 3.4, r"$\psi(t) = e^{-itH/\hbar}\,\psi_0$",
            fontsize=18, color=C["dark"], ha="center")
    box(ax, (1.0, 1.4), 3.6, 1.5, "Hamiltonian H\nself-adjoint", C["amber"])
    box(ax, (5.4, 1.4), 3.6, 1.5, "Unitary U(t)\n(by Stone)", C["green"])
    arrow(ax, (4.6, 2.15), (5.4, 2.15), color=C["dark"])
    return save(fig, "12-applications-pde-qm", "fa_v2_12_5_schrodinger")

def fa_v2_12_6_qm_spectrum():
    fig, ax = newfig((9, 6))
    ax.set_facecolor(BG)
    x = np.linspace(-6, 6, 600)
    # Potential well + bound states + scattering
    V = -3 / (1 + (x/1.5)**2)
    ax.plot(x, V, color=C["dark"], lw=2.2, label="V(x)")
    ax.fill_between(x, V, 0, where=V<0, color=C["dark"], alpha=0.05)
    # bound state energies
    bound_E = [-2.4, -1.5, -0.6]
    for i, E in enumerate(bound_E):
        ax.axhline(E, xmin=0.1, xmax=0.9, color=C["red"], lw=1.5, ls="--")
        # rough wavefunction shape
        psi = 0.4*np.exp(-(x/(1.5+i*0.6))**2) * np.cos((i+0.5)*x*0.7) + E
        ax.plot(x, psi, color=C["red"], lw=1.0, alpha=0.8)
        ax.text(5.5, E + 0.06, f"E_{i}", color=C["red"], fontsize=9)
    # continuous spectrum
    ax.axhspan(0, 2.5, alpha=0.15, color=C["blue"])
    ax.text(0, 1.7, "continuous spectrum (scattering)",
            color=C["blue"], ha="center", fontsize=10, fontweight="bold")
    ax.text(0, -0.2, "bound states", color=C["red"], ha="center",
            fontsize=10, fontweight="bold")
    ax.set_title("Quantum Spectrum: Bound States + Scattering",
                 fontsize=13, color=C["dark"], fontweight="bold")
    ax.set_xlabel("x", color=C["gray"]); ax.set_ylabel("E", color=C["gray"])
    ax.legend(frameon=False, loc="lower right")
    for s in ax.spines.values(): s.set_color(C["lightgray"])
    ax.tick_params(colors=C["gray"], labelsize=8)
    ax.grid(True, color=C["lightgray"], alpha=0.3, lw=0.5)
    ax.set_ylim(-3.2, 2.5)
    return save(fig, "12-applications-pde-qm", "fa_v2_12_6_qm_spectrum")

def fa_v2_12_7_uncertainty():
    fig, ax = newfig((10, 5))
    clean_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    title(ax, "Heisenberg Uncertainty",
          "Non-commuting self-adjoint operators bound joint variance")
    ax.text(5, 3.4, r"$\Delta A \cdot \Delta B \;\geq\; \frac{1}{2} |\langle [A,B]\rangle|$",
            fontsize=16, color=C["dark"], ha="center")
    box(ax, (1.0, 1.4), 3.6, 1.5, "Position X\nMomentum P", C["amber"])
    box(ax, (5.4, 1.4), 3.6, 1.5, "ΔX · ΔP ≥ ℏ/2", C["green"])
    arrow(ax, (4.6, 2.15), (5.4, 2.15), color=C["dark"])
    ax.text(5, 0.7, "operator-theoretic root: [X,P] = iℏI",
            fontsize=11, color=C["red"], ha="center", style="italic")
    return save(fig, "12-applications-pde-qm", "fa_v2_12_7_uncertainty")


# ============================================================
# main
# ============================================================
ALL_FIGS = [
    fa_v2_01_1_metric_axioms, fa_v2_01_2_open_balls, fa_v2_01_3_cauchy_seq,
    fa_v2_01_4_completion, fa_v2_01_5_baire, fa_v2_01_6_fixed_point, fa_v2_01_7_compactness,
    fa_v2_02_1_unit_balls_lp, fa_v2_02_2_norm_equiv, fa_v2_02_3_lp_chain,
    fa_v2_02_4_banach_complete, fa_v2_02_5_schauder, fa_v2_02_6_equiv_norms, fa_v2_02_7_seq_spaces,
    fa_v2_03_1_inner_product, fa_v2_03_2_orthogonal_proj, fa_v2_03_3_orthonormal_basis,
    fa_v2_03_4_riesz, fa_v2_03_5_parseval, fa_v2_03_6_l2_basis, fa_v2_03_7_gram_schmidt,
    fa_v2_04_1_dual_geom, fa_v2_04_2_hb_extension, fa_v2_04_3_separation,
    fa_v2_04_4_reflexive, fa_v2_04_5_weak_strong, fa_v2_04_6_lp_dual, fa_v2_04_7_supporting,
    fa_v2_05_1_weak_conv, fa_v2_05_2_alaoglu, fa_v2_05_3_weak_star,
    fa_v2_05_4_compactness, fa_v2_05_5_weak_def, fa_v2_05_6_seq_top, fa_v2_05_7_metrizability,
    fa_v2_06_1_op_norm, fa_v2_06_2_three_thms, fa_v2_06_3_ubp,
    fa_v2_06_4_open_map, fa_v2_06_5_closed_graph, fa_v2_06_6_examples, fa_v2_06_7_inverse,
    fa_v2_07_1_compact_def, fa_v2_07_2_finite_rank, fa_v2_07_3_spectral_compact,
    fa_v2_07_4_fredholm, fa_v2_07_5_hilbert_schmidt, fa_v2_07_6_eigenvalue_decay, fa_v2_07_7_compact_examples,
    fa_v2_08_1_spectrum_decomp, fa_v2_08_2_resolvent, fa_v2_08_3_spectral_radius,
    fa_v2_08_4_self_adjoint, fa_v2_08_5_spectral_thm, fa_v2_08_6_func_calc, fa_v2_08_7_examples,
    fa_v2_09_1_unbounded_domain, fa_v2_09_2_closed_op, fa_v2_09_3_sym_vs_sa,
    fa_v2_09_4_laplacian, fa_v2_09_5_friedrichs, fa_v2_09_6_essential, fa_v2_09_7_qm_examples,
    fa_v2_10_1_semigroup_orbit, fa_v2_10_2_generator, fa_v2_10_3_hille_yosida,
    fa_v2_10_4_heat_eq, fa_v2_10_5_evolution, fa_v2_10_6_resolvent_repr, fa_v2_10_7_examples,
    fa_v2_11_1_test_functions, fa_v2_11_2_distributions, fa_v2_11_3_weak_deriv,
    fa_v2_11_4_sobolev_chain, fa_v2_11_5_trace, fa_v2_11_6_dual_sobolev, fa_v2_11_7_examples,
    fa_v2_12_1_lax_milgram, fa_v2_12_2_variational, fa_v2_12_3_qm_observables,
    fa_v2_12_4_stone, fa_v2_12_5_schrodinger, fa_v2_12_6_qm_spectrum, fa_v2_12_7_uncertainty,
]

if __name__ == "__main__":
    os.makedirs(OUTROOT, exist_ok=True)
    ok, fail = 0, []
    for fn in ALL_FIGS:
        try:
            p = fn()
            ok += 1
            print(f"OK {p}")
        except Exception as e:
            fail.append((fn.__name__, str(e)))
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\nTotal: {len(ALL_FIGS)}, OK: {ok}, FAIL: {len(fail)}")
    for name, err in fail:
        print(f"  - {name}: {err}")
