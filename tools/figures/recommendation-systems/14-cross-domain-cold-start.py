"""Figures for Recommendation Systems Part 14 — Cross-Domain & Cold-Start.

Generates 7 publication-quality figures explaining cold-start scenarios,
cross-domain transfer (EMCDR / PTUPCDR), meta-learning (MAML / MeLU),
bandit exploration vs exploitation (UCB), the cold-to-warm performance
progression, and content-based fallback.

Output is written to BOTH the EN and ZH asset folders so this script
is the single source of truth. Run from any working directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#64748b"
LIGHT = "#e5e7eb"
DARK = "#1f2937"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/recommendation-systems/14-cross-domain-cold-start"
ZH_DIR = REPO / "source/_posts/zh/recommendation-systems/14-跨域推荐与冷启动解决方案"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=10, fontweight="bold", tc="white"):
    ec = ec or fc
    ax.add_patch(
        FancyBboxPatch(
            xy, w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5, facecolor=fc, edgecolor=ec,
        )
    )
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, color=tc,
    )


def _arrow(ax, p1, p2, color=DARK, lw=1.6, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2,
            arrowstyle=style, mutation_scale=14,
            linewidth=lw, color=color,
        )
    )


# ---------------------------------------------------------------------------
# Figure 1 — Cold-start scenarios (3 quadrants)
# ---------------------------------------------------------------------------
def fig1_cold_start_scenarios() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    # ---- Panel A: New User ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("New User Cold-Start", color=BLUE, pad=12)

    # Item grid (warm items)
    rng = np.random.default_rng(7)
    item_xs = np.linspace(1.2, 8.8, 6)
    item_ys = np.linspace(1.0, 6.0, 4)
    for x in item_xs:
        for y in item_ys:
            ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64,
                                   facecolor=LIGHT, edgecolor="white", lw=1.2))
    # Existing user with rich history (filled cells)
    for x, y in zip(rng.choice(item_xs, 7, replace=True),
                    rng.choice(item_ys, 7, replace=True)):
        ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64,
                               facecolor=GREEN, edgecolor="white", lw=1.2))
    # New user row (top, empty)
    for x in item_xs:
        ax.add_patch(Rectangle((x - 0.32, 7.5), 0.64, 0.64,
                               facecolor="white", edgecolor=BLUE, lw=1.5,
                               linestyle="--"))
    ax.text(0.55, 7.82, "?", ha="center", va="center",
            fontsize=18, color=BLUE, fontweight="bold")
    ax.text(0.55, 6.32, "warm", ha="center", va="center", fontsize=8, color=GRAY)
    ax.text(5, 9.25, "Items →", ha="center", fontsize=9, color=GRAY)
    ax.text(0.5, 4.0, "Users\n↑", ha="center", va="center",
            fontsize=9, color=GRAY, rotation=0)
    ax.text(5, 0.2, "No clicks, no ratings, no history",
            ha="center", fontsize=9, style="italic", color=DARK)

    # ---- Panel B: New Item ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("New Item Cold-Start", color=PURPLE, pad=12)

    item_xs = np.linspace(1.2, 7.4, 5)
    item_ys = np.linspace(1.0, 7.0, 5)
    for x in item_xs:
        for y in item_ys:
            ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64,
                                   facecolor=LIGHT, edgecolor="white", lw=1.2))
    rng = np.random.default_rng(11)
    for x, y in zip(rng.choice(item_xs, 12, replace=True),
                    rng.choice(item_ys, 12, replace=True)):
        ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64,
                               facecolor=GREEN, edgecolor="white", lw=1.2))
    # New item column (right, empty)
    for y in item_ys:
        ax.add_patch(Rectangle((8.4, y - 0.32), 0.64, 0.64,
                               facecolor="white", edgecolor=PURPLE, lw=1.5,
                               linestyle="--"))
    ax.text(8.72, 7.95, "?", ha="center", va="center",
            fontsize=18, color=PURPLE, fontweight="bold")
    ax.text(8.72, 8.55, "new", ha="center", va="center",
            fontsize=8, color=PURPLE)
    ax.text(5, 9.25, "Items →", ha="center", fontsize=9, color=GRAY)
    ax.text(5, 0.2, "Catalog entry with zero interactions",
            ha="center", fontsize=9, style="italic", color=DARK)

    # ---- Panel C: New System ----
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("New System Cold-Start", color=ORANGE, pad=12)

    item_xs = np.linspace(1.2, 8.8, 6)
    item_ys = np.linspace(1.0, 7.5, 5)
    for x in item_xs:
        for y in item_ys:
            ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64,
                                   facecolor="white", edgecolor=ORANGE,
                                   lw=1.0, linestyle=":"))
    # Sparse interactions
    rng = np.random.default_rng(3)
    for x, y in zip(rng.choice(item_xs, 4, replace=True),
                    rng.choice(item_ys, 4, replace=True)):
        ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64,
                               facecolor=ORANGE, edgecolor="white", lw=1.2,
                               alpha=0.7))
    ax.text(5, 9.25, "Items →", ha="center", fontsize=9, color=GRAY)
    ax.text(5, 0.2, "Brand-new platform — sparse everywhere",
            ha="center", fontsize=9, style="italic", color=DARK)

    fig.suptitle("Three Faces of Cold-Start",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig1_cold_start_scenarios.png")


# ---------------------------------------------------------------------------
# Figure 2 — Cross-domain transfer architecture
# ---------------------------------------------------------------------------
def fig2_cross_domain_transfer() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Source domain block
    ax.add_patch(FancyBboxPatch(
        (0.3, 1.0), 4.2, 6.0,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=2, facecolor="#eff6ff", edgecolor=BLUE,
    ))
    ax.text(2.4, 6.55, "Source Domain", ha="center",
            fontsize=12, fontweight="bold", color=BLUE)
    ax.text(2.4, 6.15, "(data-rich)", ha="center",
            fontsize=9, color=GRAY, style="italic")
    ax.text(2.4, 5.5, "Movies / Books / Amazon Electronics",
            ha="center", fontsize=8.5, color=DARK)

    _box(ax, (0.7, 4.2), 3.4, 0.7, "User-Item Interactions  R^S", BLUE, fontsize=9)
    _box(ax, (0.7, 3.2), 3.4, 0.7, "Source Embeddings  U^S, V^S", BLUE, fontsize=9)
    _box(ax, (0.7, 2.2), 3.4, 0.7, "Source Predictor  f^S", BLUE, fontsize=9)
    ax.text(2.4, 1.4, "Millions of ratings", ha="center",
            fontsize=8, color=GRAY, style="italic")

    # Bridge block (mapping function)
    ax.add_patch(FancyBboxPatch(
        (5.2, 2.5), 3.6, 3.0,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=2, facecolor="#faf5ff", edgecolor=PURPLE,
    ))
    ax.text(7.0, 5.10, "Bridge Function", ha="center",
            fontsize=12, fontweight="bold", color=PURPLE)
    ax.text(7.0, 4.70, r"$f_\phi: \mathcal{U}^S \to \mathcal{U}^T$",
            ha="center", fontsize=11, color=PURPLE)
    ax.text(7.0, 4.10, "EMCDR: shared MLP", ha="center", fontsize=8.5, color=DARK)
    ax.text(7.0, 3.70, "PTUPCDR: meta network", ha="center", fontsize=8.5, color=DARK)
    ax.text(7.0, 3.30, "DARec: adversarial", ha="center", fontsize=8.5, color=DARK)
    ax.text(7.0, 2.85, "(per-user personalized)", ha="center",
            fontsize=8, color=GRAY, style="italic")

    # Target domain block
    ax.add_patch(FancyBboxPatch(
        (9.5, 1.0), 4.2, 6.0,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=2, facecolor="#ecfdf5", edgecolor=GREEN,
    ))
    ax.text(11.6, 6.55, "Target Domain", ha="center",
            fontsize=12, fontweight="bold", color=GREEN)
    ax.text(11.6, 6.15, "(data-sparse, cold-start)", ha="center",
            fontsize=9, color=GRAY, style="italic")
    ax.text(11.6, 5.5, "Books / Music / New Category",
            ha="center", fontsize=8.5, color=DARK)

    _box(ax, (9.9, 4.2), 3.4, 0.7, "Few or No Interactions  R^T", GREEN, fontsize=9)
    _box(ax, (9.9, 3.2), 3.4, 0.7, "Mapped User Embedding", GREEN, fontsize=9)
    _box(ax, (9.9, 2.2), 3.4, 0.7, "Target Predictor  f^T", GREEN, fontsize=9)
    ax.text(11.6, 1.4, "Cold-start users / items",
            ha="center", fontsize=8, color=GRAY, style="italic")

    # Arrows
    _arrow(ax, (4.5, 3.55), (5.2, 3.55), color=PURPLE, lw=2.0)
    _arrow(ax, (8.8, 3.55), (9.5, 3.55), color=PURPLE, lw=2.0)
    ax.text(7.0, 1.5, "Knowledge transferred via shared user space",
            ha="center", fontsize=9.5, fontweight="bold", color=PURPLE)

    ax.set_title("Cross-Domain Recommendation: Source → Bridge → Target",
                 fontsize=13, pad=14)
    save(fig, "fig2_cross_domain_transfer.png")


# ---------------------------------------------------------------------------
# Figure 3 — EMCDR vs PTUPCDR architecture
# ---------------------------------------------------------------------------
def fig3_emcdr_ptupcdr() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # ---- EMCDR ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("EMCDR (Man et al., IJCAI 2017)",
                 color=BLUE, pad=12, fontsize=12)

    _box(ax, (0.5, 5.8), 2.4, 0.9, "u_i^S (source\nembedding)", BLUE, fontsize=9)
    _box(ax, (0.5, 1.5), 2.4, 0.9, "u_i^T (target\nembedding)", GREEN, fontsize=9)

    # Single shared MLP
    _box(ax, (4.2, 3.4), 2.6, 1.6,
         "Shared MLP\n$f_\\phi(\\cdot)$\n(global mapping)",
         PURPLE, fontsize=10)

    _box(ax, (8.0, 3.7), 1.6, 1.0, "$\\hat u_i^T$", PURPLE, fontsize=11)

    _arrow(ax, (2.9, 6.25), (4.2, 4.7), color=BLUE)
    _arrow(ax, (2.9, 1.95), (4.2, 3.7), color=GREEN, style="<|-")
    _arrow(ax, (6.8, 4.2), (8.0, 4.2), color=PURPLE)

    ax.text(5.0, 0.8,
            "Train mapping on overlap users:\n"
            r"$\min_\phi \sum_{i \in U_o} \|f_\phi(u_i^S) - u_i^T\|^2$",
            ha="center", fontsize=9, color=DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3f4f6", edgecolor="none"))
    ax.text(5.0, 7.5, "One mapping for all users", ha="center",
            fontsize=9, color=GRAY, style="italic")

    # ---- PTUPCDR ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("PTUPCDR (Zhu et al., WSDM 2022)",
                 color=PURPLE, pad=12, fontsize=12)

    # Source-side characteristic
    _box(ax, (0.4, 6.0), 2.6, 0.9,
         "User behavior in S\n{v_1, v_2, …, v_n}", BLUE, fontsize=8.5)
    _box(ax, (0.4, 4.6), 2.6, 0.8,
         "Characteristic\nencoder", BLUE, fontsize=9)

    # Meta network → personalized bridge
    _box(ax, (4.0, 4.8), 2.6, 1.4,
         "Meta Network\n(generates $\\phi_i$)",
         ORANGE, fontsize=9.5)

    _box(ax, (4.0, 2.4), 2.6, 1.2,
         "Personalized\nBridge $f_{\\phi_i}$",
         PURPLE, fontsize=9.5)

    _box(ax, (0.4, 2.4), 2.6, 1.2,
         "u_i^S\n(source emb)", BLUE, fontsize=9)

    _box(ax, (7.6, 2.7), 2.0, 1.0,
         "$\\hat u_i^T$\n(per-user)", PURPLE, fontsize=10)

    # Arrows
    _arrow(ax, (3.0, 6.45), (4.0, 5.8), color=BLUE)
    _arrow(ax, (1.7, 6.0), (1.7, 5.4), color=BLUE)
    _arrow(ax, (1.7, 4.6), (1.7, 3.6), color=BLUE)
    _arrow(ax, (5.3, 4.8), (5.3, 3.6), color=ORANGE, lw=2.0)
    _arrow(ax, (3.0, 3.0), (4.0, 3.0), color=BLUE)
    _arrow(ax, (6.6, 3.0), (7.6, 3.0), color=PURPLE)

    ax.text(5.0, 1.4,
            "Mapping function is conditioned on each user's behavior\n"
            r"$\hat u_i^T = f_{\phi_i}(u_i^S)$  with  $\phi_i = h_\theta(\{v_j\})$",
            ha="center", fontsize=8.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3f4f6", edgecolor="none"))
    ax.text(5.0, 7.5, "Per-user personalized bridge", ha="center",
            fontsize=9, color=GRAY, style="italic")

    fig.suptitle("Two Generations of Cross-Domain Bridges",
                 fontsize=13, fontweight="bold", y=1.00)
    save(fig, "fig3_emcdr_ptupcdr.png")


# ---------------------------------------------------------------------------
# Figure 4 — Meta-Learning (MAML) for cold-start adaptation
# ---------------------------------------------------------------------------
def fig4_maml_landscape() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # ---- Left: MAML loss landscape with task-specific minima ----
    ax = axes[0]
    ax.set_title("MAML: A Good Initialization in Parameter Space",
                 fontsize=12, pad=10)

    # Build a 2D landscape with three task minima
    grid = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(grid, grid)
    centers = [(-1.6, 1.2), (1.5, 1.4), (0.0, -1.7)]
    Z = np.zeros_like(X)
    for cx, cy in centers:
        Z += -np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 1.4)
    Z += 0.05 * (X ** 2 + Y ** 2)

    cs = ax.contourf(X, Y, Z, levels=18, cmap="Blues_r", alpha=0.8)
    ax.contour(X, Y, Z, levels=10, colors="white", alpha=0.5, linewidths=0.6)

    # Meta-init point
    meta_x, meta_y = 0.0, 0.3
    ax.plot(meta_x, meta_y, marker="*", markersize=22,
            color=ORANGE, markeredgecolor="white", markeredgewidth=1.5,
            zorder=5, label="Meta-init  $\\theta$")

    # Adaptation arrows to each task
    colors = [BLUE, PURPLE, GREEN]
    labels = ["User A  $\\theta'_A$", "User B  $\\theta'_B$", "User C  $\\theta'_C$"]
    for (cx, cy), c, lab in zip(centers, colors, labels):
        # Two short gradient steps
        dx, dy = cx - meta_x, cy - meta_y
        p1 = (meta_x + 0.45 * dx, meta_y + 0.45 * dy)
        p2 = (meta_x + 0.85 * dx, meta_y + 0.85 * dy)
        _arrow(ax, (meta_x, meta_y), p1, color=c, lw=2.2)
        _arrow(ax, p1, p2, color=c, lw=2.2)
        ax.plot(cx, cy, marker="o", markersize=10, color=c,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.text(cx, cy + 0.45, lab, ha="center", fontsize=9, color=c,
                fontweight="bold")

    ax.set_xlabel("$\\theta_1$")
    ax.set_ylabel("$\\theta_2$")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")

    # ---- Right: inner / outer loop schematic ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Inner & Outer Loop (MeLU / MAML)",
                 fontsize=12, pad=10)

    # Inner loop box
    ax.add_patch(FancyBboxPatch(
        (0.4, 4.2), 9.2, 3.2,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.8, facecolor="#eff6ff", edgecolor=BLUE,
    ))
    ax.text(5.0, 7.0, "Inner loop  (per task / user)",
            ha="center", fontsize=10, fontweight="bold", color=BLUE)
    ax.text(5.0, 6.5,
            r"$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{T_i}(f_\theta, \mathcal{S}_i)$",
            ha="center", fontsize=10, color=DARK)
    ax.text(5.0, 5.7,
            "K=1–5 gradient steps on the user's support set",
            ha="center", fontsize=8.5, color=GRAY, style="italic")
    ax.text(5.0, 5.0,
            "MeLU keeps decision layer adaptive, embedding layer frozen",
            ha="center", fontsize=8.5, color=DARK)

    # Outer loop box
    ax.add_patch(FancyBboxPatch(
        (0.4, 0.5), 9.2, 3.2,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.8, facecolor="#faf5ff", edgecolor=PURPLE,
    ))
    ax.text(5.0, 3.3, "Outer loop  (across tasks)",
            ha="center", fontsize=10, fontweight="bold", color=PURPLE)
    ax.text(5.0, 2.8,
            r"$\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{T_i}(f_{\theta'_i}, \mathcal{Q}_i)$",
            ha="center", fontsize=10, color=DARK)
    ax.text(5.0, 2.0,
            "Aggregate query-set losses across many users",
            ha="center", fontsize=8.5, color=GRAY, style="italic")
    ax.text(5.0, 1.3,
            "Shifts initialization toward fast-adapting parameters",
            ha="center", fontsize=8.5, color=DARK)

    save(fig, "fig4_maml_meta_learning.png")


# ---------------------------------------------------------------------------
# Figure 5 — UCB exploration vs exploitation
# ---------------------------------------------------------------------------
def fig5_ucb_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # ---- Left: UCB confidence bounds for 4 arms ----
    ax = axes[0]
    rng = np.random.default_rng(2024)
    arms = ["Item A", "Item B", "Item C", "Item D"]
    true_means = [0.62, 0.55, 0.71, 0.48]
    pulls = [40, 25, 5, 15]
    estimated = [tm + rng.normal(0, 0.02) for tm in true_means]
    t = sum(pulls)

    colors = [BLUE, PURPLE, GREEN, ORANGE]
    for i, (m, n, name, c) in enumerate(zip(estimated, pulls, arms, colors)):
        ucb = m + np.sqrt(2 * np.log(t) / n)
        # Estimate bar
        ax.barh(i, m, color=c, alpha=0.85, edgecolor="white", height=0.55)
        # Uncertainty whisker
        ax.errorbar(m, i, xerr=[[0], [ucb - m]], color=DARK,
                    capsize=6, lw=1.5, capthick=1.5)
        # Marker for UCB tip
        ax.plot(ucb, i, marker="D", markersize=8, color=DARK,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        ax.text(0.02, i, f"  {name}  (n={n})", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(ucb + 0.02, i, f"UCB={ucb:.2f}", va="center",
                fontsize=8.5, color=DARK)

    ax.set_yticks([])
    ax.set_xlim(0, 1.7)
    ax.set_xlabel("Estimated reward  +  uncertainty bonus")
    ax.set_title("UCB1: pick the arm with highest upper bound",
                 fontsize=12, pad=10)
    ax.axvline(0, color=GRAY, lw=0.8)

    # ---- Right: cumulative regret over time ----
    ax = axes[1]
    horizon = 1500
    t = np.arange(1, horizon + 1)
    # Random
    regret_random = 0.18 * t
    # Greedy (gets stuck on suboptimal)
    regret_greedy = 0.05 * t
    # eps-greedy
    regret_epsg = 0.012 * t + 8
    # UCB (logarithmic)
    regret_ucb = 6 * np.log(t + 1) + 4
    # Thompson
    regret_thom = 5 * np.log(t + 1) + 3

    ax.plot(t, regret_random, color=GRAY, lw=2, label="Random", linestyle=":")
    ax.plot(t, regret_greedy, color=ORANGE, lw=2, label="Pure exploit (greedy)")
    ax.plot(t, regret_epsg, color=GREEN, lw=2.2, label="ε-greedy (ε=0.1)")
    ax.plot(t, regret_ucb, color=BLUE, lw=2.5, label="UCB1")
    ax.plot(t, regret_thom, color=PURPLE, lw=2.5, label="Thompson sampling")

    ax.set_xlabel("Rounds  $t$")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("UCB & Thompson achieve $O(\\log t)$ regret",
                 fontsize=12, pad=10)
    ax.set_ylim(0, 200)
    ax.legend(loc="upper left", fontsize=9)

    save(fig, "fig5_ucb_exploration.png")


# ---------------------------------------------------------------------------
# Figure 6 — Performance vs interaction count (cold→warm)
# ---------------------------------------------------------------------------
def fig6_cold_to_warm() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))

    n = np.arange(0, 101)

    # Pure CF (fails until enough data)
    cf = np.where(n < 5, 0.18,
                  0.18 + 0.45 * (1 - np.exp(-(n - 5) / 35)))

    # Content-based (flat, decent floor)
    content = 0.42 + 0.05 * (1 - np.exp(-n / 60))

    # Meta-learning (MeLU): jumps up after a few interactions
    meta = 0.30 + 0.30 * (1 - np.exp(-n / 6))

    # Cross-domain (PTUPCDR): strong from very start due to source knowledge
    cdr = 0.50 + 0.18 * (1 - np.exp(-n / 25))

    # Hybrid production stack: best of all
    hybrid = np.maximum.reduce([cf, content, meta, cdr]) + 0.04
    hybrid = np.clip(hybrid, 0, 0.78)

    ax.plot(n, cf, color=GRAY, lw=2.2, label="Pure collaborative filtering")
    ax.plot(n, content, color=ORANGE, lw=2.2, label="Content-based fallback")
    ax.plot(n, meta, color=PURPLE, lw=2.4, label="Meta-learning (MeLU)")
    ax.plot(n, cdr, color=BLUE, lw=2.4, label="Cross-domain (PTUPCDR)")
    ax.plot(n, hybrid, color=GREEN, lw=3.0, label="Hybrid production stack")

    # Phase shading
    ax.axvspan(0, 3, color=ORANGE, alpha=0.08)
    ax.axvspan(3, 20, color=PURPLE, alpha=0.08)
    ax.axvspan(20, 100, color=GREEN, alpha=0.08)

    ax.text(1.5, 0.74, "Bootstrap", ha="center", fontsize=9,
            color=ORANGE, fontweight="bold")
    ax.text(11, 0.74, "Few-shot / meta", ha="center", fontsize=9,
            color=PURPLE, fontweight="bold")
    ax.text(60, 0.74, "Warm: full CF", ha="center", fontsize=9,
            color=GREEN, fontweight="bold")

    ax.set_xlabel("Number of user interactions  $n$")
    ax.set_ylabel("Recommendation quality (NDCG@10)")
    ax.set_title("From Cold to Warm: Which Method Wins When?",
                 fontsize=13, pad=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.1, 0.82)
    ax.legend(loc="lower right", fontsize=9.5)

    save(fig, "fig6_cold_to_warm.png")


# ---------------------------------------------------------------------------
# Figure 7 — Content-based fallback for cold-start
# ---------------------------------------------------------------------------
def fig7_content_fallback() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # New item box (left)
    ax.add_patch(FancyBboxPatch(
        (0.3, 3.2), 3.2, 2.6,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=2, facecolor="#fef3c7", edgecolor=ORANGE,
    ))
    ax.text(1.9, 5.3, "New Item", ha="center",
            fontsize=11, fontweight="bold", color=ORANGE)
    ax.text(1.9, 4.85, "(zero ratings)", ha="center",
            fontsize=8.5, color=GRAY, style="italic")
    ax.text(1.9, 4.3, "title, genre,\ntags, description,\nimage, text",
            ha="center", fontsize=8.5, color=DARK)

    # Feature extractor
    _box(ax, (4.2, 3.6), 2.6, 1.8,
         "Feature Encoder\n(BERT / CLIP /\nTF-IDF)",
         PURPLE, fontsize=9.5)

    # Item embedding
    _box(ax, (7.5, 3.9), 2.0, 1.2, "Content\nEmbedding  $e_i$",
         BLUE, fontsize=9.5)

    # Warm catalog with cosine similarity
    rng = np.random.default_rng(42)
    catalog_x = 11.5
    catalog_centers = [(catalog_x, 6.5), (catalog_x, 5.4), (catalog_x, 4.3),
                       (catalog_x, 3.2), (catalog_x, 2.1)]
    sims = [0.91, 0.84, 0.71, 0.58, 0.42]
    for (cx, cy), s in zip(catalog_centers, sims):
        size = 0.65
        # Color intensity by similarity
        alpha = 0.3 + 0.7 * s
        ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size,
                               facecolor=GREEN, alpha=alpha,
                               edgecolor="white", lw=1.2))
        ax.text(cx + 0.6, cy, f"sim = {s:.2f}", va="center",
                fontsize=9, color=DARK)

    ax.text(catalog_x, 7.4, "Warm catalog", ha="center",
            fontsize=10, fontweight="bold", color=GREEN)
    ax.text(catalog_x, 1.2, "Borrow ratings\nfrom top-K nearest",
            ha="center", fontsize=8.5, color=GRAY, style="italic")

    # Arrows
    _arrow(ax, (3.5, 4.5), (4.2, 4.5), color=DARK)
    _arrow(ax, (6.8, 4.5), (7.5, 4.5), color=DARK)
    _arrow(ax, (9.5, 4.5), (10.8, 4.5), color=DARK)

    # Predicted rating
    ax.add_patch(FancyBboxPatch(
        (5.0, 0.5), 4.0, 1.4,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        linewidth=1.5, facecolor="#ecfdf5", edgecolor=GREEN,
    ))
    ax.text(7.0, 1.45, "Predicted rating for cold-start item",
            ha="center", fontsize=9.5, fontweight="bold", color=GREEN)
    ax.text(7.0, 0.95,
            r"$\hat r_{u,i} = \frac{\sum_{j \in N_K(i)} \mathrm{sim}(i,j) \cdot r_{u,j}}"
            r"{\sum_{j \in N_K(i)} \mathrm{sim}(i,j)}$",
            ha="center", fontsize=11, color=DARK)

    ax.set_title("Content-Based Fallback: Bridging Cold Items to Warm Catalog",
                 fontsize=13, pad=14)
    save(fig, "fig7_content_fallback.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_cold_start_scenarios()
    fig2_cross_domain_transfer()
    fig3_emcdr_ptupcdr()
    fig4_maml_landscape()
    fig5_ucb_curves()
    fig6_cold_to_warm()
    fig7_content_fallback()
    print("[OK] All 7 figures written to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
