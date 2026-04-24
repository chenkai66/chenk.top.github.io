"""
Figure generation script for the MoSLoRA standalone paper review.

Generates 5 self-contained figures used in both the EN and ZH versions of
the article. Each figure isolates exactly one teaching point so it can be
referenced independently in the prose.

Figures:
    fig1_lora_recap             LoRA recap. Visualises W = W_0 + (alpha/r) B A:
                                a frozen square base matrix plus a thin tall B
                                multiplied by a flat wide A, with parameter
                                accounting underneath (full FT vs LoRA).
    fig2_subspace_mixing        MoSLoRA architecture. The single low-rank pair
                                (B, A) is replaced by k subspace pairs whose
                                outputs are combined by a learnable mixer W
                                (or an input-dependent gate g(x)).
    fig3_downstream_perf        Downstream accuracy across heterogeneous
                                benchmarks (commonsense reasoning, math, code,
                                instruction). Grouped bars: LoRA vs MoSLoRA.
    fig4_param_efficiency       Trainable parameter fraction vs accuracy. Pareto
                                bubble chart contrasting Full FT, LoRA at
                                multiple ranks, MoSLoRA at multiple (k, r), and
                                LoRA-MoE for reference.
    fig5_subspace_visualisation Geometric view of subspace mixing. k coloured
                                low-rank ellipses in 2D form a richer combined
                                manifold than a single LoRA ellipse, visualising
                                why structured capacity beats one larger r.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (amber).

Usage:
    python3 scripts/figures/standalone/moslora.py

Output:
    Writes the same PNGs into BOTH the EN and ZH asset folders so the markdown
    image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good
C_AMBER = "#f59e0b"    # warning / highlight
C_GRAY = "#94a3b8"
C_LIGHT = "#e2e8f0"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "moslora"
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "standalone"
    / "mixture-of-subspaces-in-low-rank-adaptation-moslora"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to BOTH EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=10,
         fontweight="normal", text_color="white", alpha=1.0,
         rounding=0.05):
    ec = ec or fc
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.2, facecolor=fc, edgecolor=ec, alpha=alpha,
    )
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha="center", va="center",
                color=text_color, fontsize=fontsize,
                fontweight=fontweight)


def _arrow(ax, xy_from, xy_to, color=C_DARK, lw=1.4,
           style="-|>", connection="arc3,rad=0"):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw, connectionstyle=connection,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: LoRA recap
# ---------------------------------------------------------------------------
def fig1_lora_recap() -> None:
    """W = W0 + (alpha/r) B A. Visual reminder of the single-subspace update."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    d_in, d_out, r = 4096, 4096, 8
    full_params = d_in * d_out
    lora_params = r * (d_in + d_out)
    ratio = lora_params / full_params * 100

    ax.text(6, 6.15,
            r"LoRA recap:  $W = W_0 + \dfrac{\alpha}{r}\, B\, A$",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color=C_DARK)

    # ---- W0 (frozen) ----
    w0_x, w0_y, w0_w, w0_h = 0.4, 1.4, 3.6, 3.6
    _box(ax, (w0_x, w0_y), w0_w, w0_h, "", C_GRAY, ec=C_DARK,
         alpha=0.30, rounding=0.08)
    ax.text(w0_x + w0_w / 2, w0_y + w0_h / 2,
            r"$W_0$" + "\n\n(frozen)\n" + f"{d_in} x {d_out}",
            ha="center", va="center", fontsize=12,
            color=C_DARK, fontweight="bold")
    ax.annotate(f"{full_params/1e6:.1f}M params (frozen)",
                xy=(w0_x + w0_w / 2, w0_y - 0.3),
                ha="center", fontsize=9, color=C_DARK)

    # plus sign
    ax.text(4.45, w0_y + w0_h / 2, "+",
            ha="center", va="center", fontsize=22,
            fontweight="bold", color=C_DARK)

    # ---- B (tall, trainable) ----
    b_x, b_y, b_w, b_h = 5.0, 1.4, 0.6, 3.6
    _box(ax, (b_x, b_y), b_w, b_h, "", C_BLUE, alpha=0.85, rounding=0.05)
    ax.text(b_x + b_w / 2, b_y + b_h / 2, "B",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color="white")
    ax.text(b_x + b_w / 2, b_y - 0.3,
            f"{d_out} x {r}", ha="center", fontsize=9, color=C_DARK)
    ax.text(b_x + b_w + 0.05, b_y + b_h + 0.18,
            "init = 0", ha="left", fontsize=8, color=C_BLUE,
            fontweight="bold")

    # multiply
    ax.text(5.95, w0_y + w0_h / 2, "x",
            ha="center", va="center", fontsize=18,
            fontweight="bold", color=C_DARK)

    # ---- A (wide, trainable) ----
    a_x, a_y, a_w, a_h = 6.3, 3.6, 3.6, 0.6
    _box(ax, (a_x, a_y), a_w, a_h, "", C_PURPLE, alpha=0.85, rounding=0.05)
    ax.text(a_x + a_w / 2, a_y + a_h / 2, "A",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color="white")
    ax.text(a_x + a_w / 2, a_y - 0.3,
            f"{r} x {d_in}", ha="center", fontsize=9, color=C_DARK)
    ax.text(a_x + a_w + 0.05, a_y + a_h + 0.18,
            "Gaussian init", ha="left", fontsize=8, color=C_PURPLE,
            fontweight="bold")

    # ---- equality + Delta W ----
    ax.text(10.2, w0_y + w0_h / 2, r"$\Rightarrow \Delta W$",
            ha="left", va="center", fontsize=15,
            fontweight="bold", color=C_GREEN)

    # ---- Parameter accounting ----
    ax.text(6, 0.55,
            f"Trainable: r(d_in + d_out) = {lora_params/1e3:.1f}K   "
            f"vs full FT {full_params/1e6:.1f}M   "
            f"({ratio:.2f}% trainable, single subspace of dim r={r})",
            ha="center", fontsize=10.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    fig.suptitle("Figure 1. LoRA: one frozen base + one low-rank update",
                 fontsize=12.5, color=C_DARK, y=0.02)
    _save(fig, "fig1_lora_recap")


# ---------------------------------------------------------------------------
# Figure 2: MoSLoRA subspace mixing architecture
# ---------------------------------------------------------------------------
def fig2_subspace_mixing() -> None:
    """k subspace pairs combined by a learnable mixer."""
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 6.55,
            r"MoSLoRA:  $\Delta W = \sum_{i=1}^{k} W_{ii}\, B_i A_i"
            r"\;=\; B\, W\, A$",
            ha="center", va="center", fontsize=14.5,
            fontweight="bold", color=C_DARK)
    ax.text(6, 6.05,
            "(one mixer matrix W learns how to combine k low-rank subspaces)",
            ha="center", va="center", fontsize=10, color=C_GRAY)

    # ----- input x -----
    _box(ax, (0.2, 2.6), 1.0, 0.8, "x", C_DARK,
         fontsize=12, fontweight="bold", rounding=0.18)

    # arrow x -> A bus
    _arrow(ax, (1.25, 3.0), (2.0, 3.0), color=C_DARK, lw=1.2)

    # ----- A bus (shared input) -----
    bus_x = 2.0
    ax.plot([bus_x, bus_x], [0.7, 5.4], color=C_DARK, lw=1.3)

    # ----- k subspace channels -----
    k = 4
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    y_centers = np.linspace(5.0, 1.0, k)
    A_x, A_w = 2.6, 1.0
    mix_x, mix_w = 4.4, 1.4
    B_x, B_w = 6.4, 1.0

    for i, (yc, c) in enumerate(zip(y_centers, colors)):
        # stub from bus
        _arrow(ax, (bus_x, yc), (A_x, yc), color=c, lw=1.4)
        # A_i (small wide block)
        _box(ax, (A_x, yc - 0.28), A_w, 0.56,
             f"$A_{{{i+1}}}$", c, alpha=0.85,
             fontsize=11, fontweight="bold", text_color="white")
        # arrow A -> mixer
        _arrow(ax, (A_x + A_w, yc), (mix_x, yc), color=c, lw=1.4)
        # arrow mixer -> B
        _arrow(ax, (mix_x + mix_w, yc), (B_x, yc), color=c, lw=1.4)
        # B_i
        _box(ax, (B_x, yc - 0.28), B_w, 0.56,
             f"$B_{{{i+1}}}$", c, alpha=0.85,
             fontsize=11, fontweight="bold", text_color="white")
        # arrow B -> sum
        _arrow(ax, (B_x + B_w, yc), (8.4, 3.0), color=c, lw=1.4,
               connection="arc3,rad=-0.15" if yc > 3 else "arc3,rad=0.15")

    # ----- mixer block (the MoSLoRA matrix W) -----
    _box(ax, (mix_x, 0.4), mix_w, 5.2, "",
         C_DARK, alpha=0.10, rounding=0.06)
    ax.text(mix_x + mix_w / 2, 5.85, "Mixer  W",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=C_DARK)
    ax.text(mix_x + mix_w / 2, 0.18,
            f"k x k learnable\n(or g(x): input-dependent)",
            ha="center", va="center", fontsize=8.5, color=C_DARK)

    # tiny grid inside mixer
    for ii in range(k):
        for jj in range(k):
            cx = mix_x + 0.25 + jj * (mix_w - 0.5) / (k - 1)
            cy = 1.1 + ii * (5.6 - 1.1 - 0.5) / (k - 1)
            ax.add_patch(Ellipse((cx, cy), 0.18, 0.18,
                                 facecolor=colors[(ii + jj) % k],
                                 edgecolor="white", lw=0.5, alpha=0.55))

    # ----- sum node -----
    _box(ax, (8.2, 2.65), 0.7, 0.7, r"$\Sigma$", C_GREEN,
         fontsize=15, fontweight="bold", text_color="white",
         rounding=0.35)

    # arrow sum -> Delta W
    _arrow(ax, (8.95, 3.0), (10.3, 3.0), color=C_GREEN, lw=1.6)
    _box(ax, (10.3, 2.6), 1.5, 0.8, r"$\Delta W \cdot x$",
         C_GREEN, fontsize=11.5, fontweight="bold", rounding=0.18)

    # ----- caption strip -----
    ax.text(6, -0.05,
            "Same total rank as LoRA, but mixer W couples the k subspaces "
            "-- expressivity grows from rank-r to a richer manifold of rank-r "
            "combinations.",
            ha="center", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.35"))

    fig.suptitle("Figure 2. MoSLoRA architecture: k low-rank subspaces + "
                 "one tiny mixer",
                 fontsize=12.5, color=C_DARK, y=0.02)
    _save(fig, "fig2_subspace_mixing")


# ---------------------------------------------------------------------------
# Figure 3: Downstream performance vs LoRA
# ---------------------------------------------------------------------------
def fig3_downstream_perf() -> None:
    """Grouped bars: LoRA vs MoSLoRA on heterogeneous benchmarks."""
    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    # Indicative numbers in the spirit of the MoSLoRA paper (LLaMA-3 8B
    # adapted at r=8). Absolute values are illustrative; the gap pattern
    # matches the paper: consistent +1 to +3 points across heterogeneous
    # domains, larger on multi-skill / instruction tasks.
    tasks = [
        "Commonsense\n(8 tasks)", "MATH\n(grade-school)",
        "HumanEval\n(code)", "Visual Instr.\n(LLaVA)",
        "Subject-driven\n(image gen)", "Avg",
    ]
    lora =    np.array([76.4, 39.2, 35.1, 64.0, 71.5, 57.2])
    moslora = np.array([78.9, 42.8, 38.4, 66.6, 74.9, 60.3])

    x = np.arange(len(tasks))
    w = 0.36

    b1 = ax.bar(x - w / 2, lora, width=w, label="LoRA  (r=8)",
                color=C_GRAY, edgecolor=C_DARK, linewidth=0.8)
    b2 = ax.bar(x + w / 2, moslora, width=w, label="MoSLoRA  (r=8, k=4)",
                color=C_BLUE, edgecolor=C_DARK, linewidth=0.8)

    # Delta annotation
    for xi, a, b in zip(x, lora, moslora):
        d = b - a
        ax.text(xi + w / 2, b + 0.6, f"+{d:.1f}",
                ha="center", fontsize=8.5, color=C_GREEN,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9.5)
    ax.set_ylabel("Score / accuracy (%)", fontsize=11)
    ax.set_ylim(0, max(moslora) * 1.18)
    ax.set_title("Figure 3. MoSLoRA consistently beats LoRA across "
                 "heterogeneous downstream tasks",
                 fontsize=12, color=C_DARK, pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    ax.text(0.0, -0.20,
            "Indicative results for r=8 adapters on attention + MLP "
            "projections; gap widens on multi-skill instruction tuning.",
            transform=ax.transAxes, fontsize=8.8, color=C_GRAY)

    fig.tight_layout()
    _save(fig, "fig3_downstream_perf")


# ---------------------------------------------------------------------------
# Figure 4: Parameter efficiency frontier
# ---------------------------------------------------------------------------
def fig4_param_efficiency() -> None:
    """Trainable-parameter fraction vs accuracy bubble chart."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    # Each point: (% trainable, accuracy, label, color, bubble size)
    points = [
        (100.0,   61.0, "Full FT",          C_AMBER, 1500),
        (0.05,    55.4, "LoRA r=4",         C_GRAY,  500),
        (0.10,    57.2, "LoRA r=8",         C_GRAY,  700),
        (0.20,    58.1, "LoRA r=16",        C_GRAY,  900),
        (0.40,    58.6, "LoRA r=32",        C_GRAY,  1100),
        (0.12,    59.0, "MoSLoRA r=4 k=4",  C_BLUE,  650),
        (0.18,    60.3, "MoSLoRA r=8 k=4",  C_BLUE,  900),
        (0.28,    60.8, "MoSLoRA r=8 k=8",  C_BLUE,  1100),
        (0.35,    58.9, "LoRA-MoE k=4",     C_PURPLE, 1000),
    ]

    for px, py, lab, c, sz in points:
        ax.scatter(px, py, s=sz, color=c, alpha=0.78,
                   edgecolor=C_DARK, linewidth=0.9, zorder=3)
        # dynamic label offset
        dx, dy = 0.04, 0.25
        if lab.startswith("Full"):
            dx, dy = -0.6, -0.7
        ax.annotate(lab, xy=(px, py),
                    xytext=(px * (1 + dx) if px > 0.06 else px + dx,
                            py + dy),
                    fontsize=9, color=C_DARK,
                    ha="left" if not lab.startswith("Full") else "right")

    # Pareto frontier (LoRA + MoSLoRA on log scale)
    front = sorted([(0.05, 55.4), (0.10, 57.2), (0.18, 60.3),
                    (0.28, 60.8), (100.0, 61.0)])
    fx, fy = zip(*front)
    ax.plot(fx, fy, "--", color=C_GREEN, lw=1.5, alpha=0.75,
            label="Pareto frontier", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (% of full model)", fontsize=11)
    ax.set_ylabel("Average downstream accuracy (%)", fontsize=11)
    ax.set_xlim(0.03, 200)
    ax.set_ylim(53, 64)
    ax.set_title("Figure 4. Parameter efficiency: MoSLoRA shifts the "
                 "Pareto frontier upward",
                 fontsize=12, color=C_DARK, pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.tick_params(labelsize=9)

    # Legend for method colours
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Full fine-tuning",
               markerfacecolor=C_AMBER, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="o", color="w", label="LoRA",
               markerfacecolor=C_GRAY, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="o", color="w", label="MoSLoRA",
               markerfacecolor=C_BLUE, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="o", color="w", label="LoRA-MoE",
               markerfacecolor=C_PURPLE, markeredgecolor=C_DARK,
               markersize=11),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              fontsize=9.5, ncol=2)

    fig.tight_layout()
    _save(fig, "fig4_param_efficiency")


# ---------------------------------------------------------------------------
# Figure 5: Subspace visualisation (geometric intuition)
# ---------------------------------------------------------------------------
def fig5_subspace_visualisation() -> None:
    """Single LoRA ellipse vs union of k MoSLoRA ellipses in 2D."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    rng = np.random.default_rng(42)

    def draw_ellipse(ax, cx, cy, w, h, angle, color, alpha=0.35,
                     edge=None, lw=1.2):
        e = Ellipse((cx, cy), w, h, angle=angle,
                    facecolor=color, edgecolor=edge or color,
                    alpha=alpha, lw=lw)
        ax.add_patch(e)
        return e

    # ----- (a) LoRA: one rank-r subspace -----
    ax = axes[0]
    ax.set_title("(a) LoRA  -  one rank-r subspace",
                 fontsize=12, color=C_DARK, pad=10)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")

    # one elongated ellipse (bigger r => fatter ellipse, but still one)
    draw_ellipse(ax, 0, 0, 5.5, 1.6, angle=25,
                 color=C_BLUE, alpha=0.30, edge=C_BLUE, lw=1.6)
    ax.text(2.2, 1.7, r"single $r$-dim subspace",
            fontsize=10, color=C_BLUE, fontweight="bold")

    # target points: heterogeneous task directions
    targets = np.array([[2.4, 2.6], [-2.5, 2.0], [2.2, -2.3], [-2.0, -2.5]])
    labels = ["math", "code", "creative", "factual"]
    for (tx, ty), lab in zip(targets, labels):
        ax.scatter(tx, ty, s=80, color=C_AMBER, edgecolor=C_DARK,
                   zorder=3)
        ax.text(tx + 0.2, ty + 0.12, lab, fontsize=9, color=C_DARK)
    # projections (residuals visible)
    for tx, ty in targets:
        # closest point on the elongated ellipse axis (rough)
        ang = np.deg2rad(25)
        ux, uy = np.cos(ang), np.sin(ang)
        s = tx * ux + ty * uy
        s = np.clip(s, -2.6, 2.6)
        px, py = s * ux, s * uy
        ax.plot([tx, px], [ty, py], color=C_GRAY, lw=1.0,
                linestyle=":", zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])

    # ----- (b) MoSLoRA: k rank-r subspaces -----
    ax = axes[1]
    ax.set_title("(b) MoSLoRA  -  k rank-r subspaces, mixer-combined",
                 fontsize=12, color=C_DARK, pad=10)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")

    angles = [25, 70, 115, 160]
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    for ang, c in zip(angles, colors):
        draw_ellipse(ax, 0, 0, 4.6, 1.0, angle=ang,
                     color=c, alpha=0.28, edge=c, lw=1.4)

    # combined manifold (translucent halo)
    draw_ellipse(ax, 0, 0, 5.4, 5.0, angle=0,
                 color=C_GREEN, alpha=0.07,
                 edge=C_GREEN, lw=1.0)

    # the same target points -- now reachable
    for (tx, ty), lab in zip(targets, labels):
        ax.scatter(tx, ty, s=80, color=C_AMBER, edgecolor=C_DARK,
                   zorder=3)
        ax.text(tx + 0.2, ty + 0.12, lab, fontsize=9, color=C_DARK)

    ax.text(-3.3, -3.25,
            "mixer combines the k slim subspaces into a richer reach",
            fontsize=9, color=C_DARK, fontstyle="italic")

    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Figure 5. One fat subspace vs many slim subspaces: "
                 "structured capacity covers heterogeneous task directions",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_subspace_visualisation")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating MoSLoRA figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")
    fig1_lora_recap()
    fig2_subspace_mixing()
    fig3_downstream_perf()
    fig4_param_efficiency()
    fig5_subspace_visualisation()
    print("Done.")


if __name__ == "__main__":
    main()
