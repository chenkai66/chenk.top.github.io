"""
Figure generation script for the paper2repo standalone paper review.

Generates 5 self-contained figures used in both the EN and ZH versions of
the article. Each figure isolates one teaching point and is referenced
once in the prose.

Figures:
    fig1_system_architecture    End-to-end pipeline. Two text encoders feed
                                two GCN towers (paper-side citation graph,
                                repo-side association graph) which are tied
                                by a cosine-alignment constraint on the
                                bridged pairs. Output is a shared embedding
                                space for ranking.
    fig2_heterogeneous_graph    Joint heterogeneous graph view: paper nodes
                                connected by citations, repo nodes connected
                                by co-star and tag-overlap edges, and user
                                nodes whose star edges induce the repo-side
                                association graph. Bridged paper-repo pairs
                                are highlighted.
    fig3_embedding_objectives   The two training forces. (a) WARP ranking
                                pulls each paper toward its positive repo
                                and pushes negatives below the margin; (b)
                                the alignment constraint pulls bridged
                                paper-repo embeddings together so the two
                                towers share a metric space.
    fig4_recommendation_flow    Inference flow: encode a query paper -> GCN
                                propagation -> dot-product against all repo
                                embeddings -> top-K shortlist with similarity
                                bars.
    fig5_evaluation_results     Grouped bars: paper2repo vs the seven
                                baselines (NSCR, KGCN, CDL, NCF, LINE, MF,
                                BPR) on HR@10, MAP@10, MRR@10. Highlights
                                the consistent margin and the larger gap on
                                HR@10.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (amber).

Usage:
    python3 scripts/figures/standalone/paper2repo.py

Output:
    Writes the same PNGs into BOTH the EN and ZH asset folders so the
    markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # papers / primary
C_PURPLE = "#7c3aed"   # repos / secondary
C_GREEN = "#10b981"    # bridged / good
C_AMBER = "#f59e0b"    # users / highlight
C_GRAY = "#94a3b8"
C_LIGHT = "#e2e8f0"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "en" / "standalone"
    / "paper2repo-github-repository-recommendation"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "standalone"
    / "paper2repo-github-repository-recommendation-for-academic-pap"
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
# Figure 1: System architecture
# ---------------------------------------------------------------------------
def fig1_system_architecture() -> None:
    """Two text encoders + two GCN towers tied by a cosine constraint."""
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(6.5, 6.85,
            "paper2repo: cross-platform alignment of papers and repositories",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=C_DARK)

    # --- Left tower: papers ---
    _box(ax, (0.3, 5.2), 2.4, 0.7,
         "Paper abstracts", C_BLUE, alpha=0.85,
         fontsize=10.5, fontweight="bold", rounding=0.12)
    _box(ax, (0.3, 4.1), 2.4, 0.7,
         "CNN text encoder", C_BLUE, alpha=0.55,
         fontsize=10, fontweight="bold", text_color=C_DARK,
         rounding=0.10)
    _box(ax, (0.3, 2.7), 2.4, 1.0,
         "Citation graph\n(paper - paper)", C_BLUE, alpha=0.30,
         fontsize=10, fontweight="bold", text_color=C_DARK,
         rounding=0.10)
    _box(ax, (0.3, 1.3), 2.4, 0.9,
         "GCN tower\n(paper side)", C_BLUE, alpha=0.85,
         fontsize=10.5, fontweight="bold", rounding=0.10)
    _box(ax, (0.3, 0.2), 2.4, 0.7,
         r"paper embeddings  $h^p$", C_DARK,
         fontsize=10.5, fontweight="bold", rounding=0.12)

    # arrows down (paper tower)
    for y_top, y_bot in [(5.2, 4.8), (4.1, 3.7), (2.7, 2.2), (1.3, 0.9)]:
        _arrow(ax, (1.5, y_top), (1.5, y_bot), color=C_BLUE, lw=1.3)

    # --- Right tower: repos ---
    _box(ax, (10.3, 5.2), 2.4, 0.7,
         "Repo description + tags", C_PURPLE, alpha=0.85,
         fontsize=10.5, fontweight="bold", rounding=0.12)
    _box(ax, (10.3, 4.1), 2.4, 0.7,
         "CNN text encoder", C_PURPLE, alpha=0.55,
         fontsize=10, fontweight="bold", text_color=C_DARK,
         rounding=0.10)
    _box(ax, (10.3, 2.7), 2.4, 1.0,
         "Association graph\n(co-star + tag overlap)",
         C_PURPLE, alpha=0.30,
         fontsize=10, fontweight="bold", text_color=C_DARK,
         rounding=0.10)
    _box(ax, (10.3, 1.3), 2.4, 0.9,
         "GCN tower\n(repo side)", C_PURPLE, alpha=0.85,
         fontsize=10.5, fontweight="bold", rounding=0.10)
    _box(ax, (10.3, 0.2), 2.4, 0.7,
         r"repo embeddings  $h^r$", C_DARK,
         fontsize=10.5, fontweight="bold", rounding=0.12)

    # arrows down (repo tower)
    for y_top, y_bot in [(5.2, 4.8), (4.1, 3.7), (2.7, 2.2), (1.3, 0.9)]:
        _arrow(ax, (11.5, y_top), (11.5, y_bot), color=C_PURPLE, lw=1.3)

    # --- Center: alignment constraint ---
    _box(ax, (4.4, 3.4), 4.2, 1.5, "", C_GREEN, alpha=0.12,
         ec=C_GREEN, rounding=0.06)
    ax.text(6.5, 4.55,
            "Alignment constraint  (bridged pairs)",
            ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=C_GREEN)
    ax.text(6.5, 4.05,
            r"$h^{p}_{i}\!\cdot h^{r}_{i} \;\geq\; 1 - \delta$",
            ha="center", va="center", fontsize=13,
            color=C_DARK)
    ax.text(6.5, 3.62,
            r"normalized embeddings  ($\|h\|_2 = 1$)",
            ha="center", va="center", fontsize=9,
            color=C_GRAY, fontstyle="italic")

    # arrows from each tower into the constraint box
    _arrow(ax, (2.7, 4.45), (4.4, 4.15), color=C_GREEN, lw=1.4,
           connection="arc3,rad=0.05")
    _arrow(ax, (10.3, 4.45), (8.6, 4.15), color=C_GREEN, lw=1.4,
           connection="arc3,rad=-0.05")

    # --- Bottom: WARP ranking + shared metric space ---
    _box(ax, (4.4, 1.3), 4.2, 0.9,
         "WARP ranking loss\n(paper vs positive repo vs negatives)",
         C_AMBER, alpha=0.85, fontsize=10.5,
         fontweight="bold", rounding=0.10)
    _arrow(ax, (2.7, 1.75), (4.4, 1.75), color=C_AMBER, lw=1.4)
    _arrow(ax, (10.3, 1.75), (8.6, 1.75), color=C_AMBER, lw=1.4)

    _box(ax, (4.4, 0.2), 4.2, 0.7,
         "Shared embedding space  ->  rank repos by  "
         r"$h^{p}\!\cdot h^{r}$",
         C_DARK, fontsize=10.5, fontweight="bold", rounding=0.12)
    _arrow(ax, (6.5, 1.3), (6.5, 0.9), color=C_DARK, lw=1.5)

    # legend
    handles = [
        Line2D([0], [0], marker="s", color="w", label="Paper side",
               markerfacecolor=C_BLUE, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="s", color="w", label="Repo side",
               markerfacecolor=C_PURPLE, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="s", color="w", label="Alignment",
               markerfacecolor=C_GREEN, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="s", color="w", label="Ranking",
               markerfacecolor=C_AMBER, markeredgecolor=C_DARK,
               markersize=11),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, 0.06), ncol=4,
              frameon=True, fontsize=9.5)

    fig.suptitle("Figure 1. paper2repo system architecture: two text-encoder "
                 "+ GCN towers tied by a cosine constraint",
                 fontsize=12, color=C_DARK, y=0.005)
    _save(fig, "fig1_system_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Heterogeneous graph (papers + repos + users)
# ---------------------------------------------------------------------------
def fig2_heterogeneous_graph() -> None:
    """Joint heterogeneous graph: papers, repos, users + bridged pairs."""
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 7.5)
    ax.axis("off")

    ax.text(6, 7.15,
            "Heterogeneous graph that paper2repo consumes",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=C_DARK)

    rng = np.random.default_rng(7)

    # --- Paper nodes (left cluster, citation graph) ---
    G_p = nx.barabasi_albert_graph(8, 2, seed=3)
    pos_p = nx.spring_layout(G_p, seed=11, k=0.9)
    paper_xy = {}
    for i, (x, y) in pos_p.items():
        # remap to left region
        px = 0.8 + (x + 1) * 1.6
        py = 1.4 + (y + 1) * 1.9
        paper_xy[i] = (px, py)
    for u, v in G_p.edges():
        ax.plot(*zip(paper_xy[u], paper_xy[v]),
                color=C_BLUE, lw=1.0, alpha=0.55, zorder=1)
    for i, (px, py) in paper_xy.items():
        ax.add_patch(Circle((px, py), 0.20,
                            facecolor=C_BLUE, edgecolor=C_DARK,
                            lw=0.9, zorder=3))
        ax.text(px, py, "P", color="white", fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=4)

    ax.text(2.4, 5.95, "Paper citation graph",
            ha="center", fontsize=10.5, color=C_BLUE, fontweight="bold")

    # --- Repo nodes (right cluster, association graph) ---
    G_r = nx.barabasi_albert_graph(8, 2, seed=5)
    pos_r = nx.spring_layout(G_r, seed=23, k=0.9)
    repo_xy = {}
    for i, (x, y) in pos_r.items():
        rx = 8.6 + (x + 1) * 1.6
        ry = 1.4 + (y + 1) * 1.9
        repo_xy[i] = (rx, ry)
    for u, v in G_r.edges():
        ax.plot(*zip(repo_xy[u], repo_xy[v]),
                color=C_PURPLE, lw=1.0, alpha=0.55, zorder=1)
    for i, (rx, ry) in repo_xy.items():
        ax.add_patch(Circle((rx, ry), 0.20,
                            facecolor=C_PURPLE, edgecolor=C_DARK,
                            lw=0.9, zorder=3))
        ax.text(rx, ry, "R", color="white", fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=4)

    ax.text(10.2, 5.95,
            "Repo association graph\n(co-star + tag overlap)",
            ha="center", fontsize=10.5, color=C_PURPLE, fontweight="bold")

    # --- User nodes (bottom strip) inducing co-star edges ---
    user_x = np.linspace(7.6, 12.0, 4)
    for i, ux in enumerate(user_x):
        ax.add_patch(Circle((ux, 0.25), 0.16,
                            facecolor=C_AMBER, edgecolor=C_DARK,
                            lw=0.9, zorder=3))
        ax.text(ux, 0.25, "U", color="white", fontsize=7,
                ha="center", va="center", fontweight="bold", zorder=4)

    # star edges: each user stars 2-3 repos (dashed amber)
    star_pairs = [
        (0, [0, 1, 4]),
        (1, [1, 2, 5]),
        (2, [3, 5, 6]),
        (3, [4, 6, 7]),
    ]
    for u_idx, repos in star_pairs:
        ux = user_x[u_idx]
        for r_idx in repos:
            rx, ry = repo_xy[r_idx]
            ax.plot([ux, rx], [0.25, ry],
                    color=C_AMBER, lw=0.7, alpha=0.45,
                    linestyle=":", zorder=1)

    ax.text(9.8, -0.45, "Users star repos -> induces co-star edges",
            ha="center", fontsize=9.5, color=C_AMBER, fontstyle="italic")

    # --- Bridged paper-repo pairs (the supervision) ---
    bridged = [(0, 0), (3, 4), (5, 6)]
    for p_idx, r_idx in bridged:
        px, py = paper_xy[p_idx]
        rx, ry = repo_xy[r_idx]
        ax.annotate(
            "", xy=(rx - 0.22, ry), xytext=(px + 0.22, py),
            arrowprops=dict(
                arrowstyle="<|-|>", color=C_GREEN, lw=2.0,
                connectionstyle="arc3,rad=0.08"),
            zorder=2,
        )

    ax.text(6, 4.2, "Bridged paper-repo pair",
            ha="center", fontsize=10.5, color=C_GREEN, fontweight="bold")
    ax.text(6, 3.85,
            r"(supervises the alignment:  $h^p \cdot h^r \to 1$)",
            ha="center", fontsize=9.5, color=C_GREEN, fontstyle="italic")

    # legend
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Paper",
               markerfacecolor=C_BLUE, markeredgecolor=C_DARK, markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Repository",
               markerfacecolor=C_PURPLE, markeredgecolor=C_DARK,
               markersize=10),
        Line2D([0], [0], marker="o", color="w", label="User",
               markerfacecolor=C_AMBER, markeredgecolor=C_DARK,
               markersize=10),
        Line2D([0], [0], color=C_BLUE, lw=2, label="Citation"),
        Line2D([0], [0], color=C_PURPLE, lw=2, label="Co-star / tag"),
        Line2D([0], [0], color=C_AMBER, lw=2, linestyle=":",
               label="Star event"),
        Line2D([0], [0], color=C_GREEN, lw=2.4, label="Bridged pair"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=7,
              frameon=True, fontsize=9)

    fig.suptitle("Figure 2. The joint graph: two structural views plus a "
                 "thin bridge of labeled paper-repo pairs",
                 fontsize=11.8, color=C_DARK, y=0.005)
    _save(fig, "fig2_heterogeneous_graph")


# ---------------------------------------------------------------------------
# Figure 3: Embedding objectives - WARP ranking + alignment constraint
# ---------------------------------------------------------------------------
def fig3_embedding_objectives() -> None:
    """Two side-by-side panels visualising the two training forces."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # ---------- (a) WARP ranking ----------
    ax = axes[0]
    ax.set_title("(a) WARP ranking force",
                 fontsize=12, color=C_DARK, pad=10)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")

    # unit circle (normalized embedding sphere)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta),
            color=C_GRAY, lw=0.8, linestyle="--", alpha=0.6)
    ax.text(0, 2.25, r"$\|h\|_2 = 1$  (cosine = dot product)",
            ha="center", fontsize=9, color=C_GRAY, fontstyle="italic")

    # paper anchor
    p = np.array([2.0, 0.0])
    ax.scatter(*p, s=160, color=C_BLUE, edgecolor=C_DARK,
               zorder=4, label="paper  $h^p$")
    ax.text(p[0] + 0.15, p[1] + 0.12, r"$h^{p}$",
            fontsize=12, color=C_BLUE, fontweight="bold")

    # positive repo (close)
    pos = np.array([1.85, 0.75])
    ax.scatter(*pos, s=140, color=C_GREEN, edgecolor=C_DARK,
               zorder=4)
    ax.text(pos[0] + 0.1, pos[1] + 0.12, r"$h^{r^+}$",
            fontsize=12, color=C_GREEN, fontweight="bold")
    ax.annotate("", xy=pos, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2.0))

    # negatives (far) - some violate margin
    negs = np.array([
        [-1.2, 1.6],
        [0.4, -1.95],
        [-1.95, -0.5],
        [1.5, -1.3],
        [1.0, 1.8],
    ])
    for i, n in enumerate(negs):
        # inside-margin negatives get pushed away
        sim_neg = n @ p / 4  # crude
        violates = sim_neg > -0.05
        col = C_AMBER if violates else C_GRAY
        ax.scatter(*n, s=110, color=col, edgecolor=C_DARK, zorder=3)
        ax.text(n[0] + 0.08, n[1] + 0.12, r"$h^{r^-}$",
                fontsize=9.5, color=col)
        if violates:
            ax.annotate("", xy=n + (n - p) * 0.15, xytext=n,
                        arrowprops=dict(arrowstyle="-|>",
                                        color=C_AMBER, lw=1.4,
                                        alpha=0.85))

    # margin band around the positive direction
    ang_pos = np.arctan2(pos[1], pos[0])
    margin = 0.55  # radians-ish for visual
    arc = np.linspace(ang_pos - margin, ang_pos + margin, 60)
    ax.plot(2.05 * np.cos(arc), 2.05 * np.sin(arc),
            color=C_GREEN, lw=2.2, alpha=0.8)
    ax.text(0, -2.4,
            r"pull positive in, push violators out by margin $\gamma$",
            ha="center", fontsize=10, color=C_DARK, fontstyle="italic")
    ax.set_xticks([])
    ax.set_yticks([])

    # ---------- (b) Alignment constraint ----------
    ax = axes[1]
    ax.set_title("(b) Cross-tower alignment constraint",
                 fontsize=12, color=C_DARK, pad=10)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")

    # Two ellipsoidal "spaces" before alignment (faint)
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((-1.5, 0.4), 2.6, 1.6, angle=20,
                         facecolor=C_BLUE, alpha=0.12,
                         edgecolor=C_BLUE, lw=1.0))
    ax.add_patch(Ellipse((1.6, -0.3), 2.4, 1.7, angle=-15,
                         facecolor=C_PURPLE, alpha=0.12,
                         edgecolor=C_PURPLE, lw=1.0))
    ax.text(-1.5, 1.55, "paper space",
            ha="center", fontsize=9, color=C_BLUE, fontweight="bold")
    ax.text(1.6, -1.4, "repo space",
            ha="center", fontsize=9, color=C_PURPLE, fontweight="bold")

    # Bridged pairs - drawn before-and-after
    pairs = [
        ((-2.0, 0.8), (2.4, -0.3)),
        ((-1.0, 0.0), (1.4, -0.7)),
        ((-1.7, -0.2), (1.9, 0.2)),
    ]
    for (px, py), (rx, ry) in pairs:
        ax.scatter(px, py, s=110, color=C_BLUE,
                   edgecolor=C_DARK, zorder=4)
        ax.scatter(rx, ry, s=110, color=C_PURPLE,
                   edgecolor=C_DARK, zorder=4)
        ax.annotate("", xy=(rx, ry), xytext=(px, py),
                    arrowprops=dict(arrowstyle="<|-|>",
                                    color=C_GREEN, lw=1.8, alpha=0.85,
                                    connectionstyle="arc3,rad=0.06"))

    ax.text(0, 2.1,
            r"$h^{p}_{i}\!\cdot h^{r}_{i}\;\geq\;1 - \delta$",
            ha="center", fontsize=13, color=C_DARK)
    ax.text(0, -2.4,
            "shrink the gap between bridged paper-repo embeddings",
            ha="center", fontsize=10, color=C_DARK, fontstyle="italic")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Figure 3. Two training forces in paper2repo: ranking "
                 "(WARP) + alignment (constrained GCN)",
                 fontsize=12.5, color=C_DARK, y=1.01)
    fig.tight_layout()
    _save(fig, "fig3_embedding_objectives")


# ---------------------------------------------------------------------------
# Figure 4: Recommendation flow at inference time
# ---------------------------------------------------------------------------
def fig4_recommendation_flow() -> None:
    """Query paper -> embedding -> dot product -> top-K repos."""
    fig = plt.figure(figsize=(12.0, 5.6))

    # Two areas: pipeline strip (top) + ranking bars (bottom)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0],
                          hspace=0.45)
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_xlim(0, 13)
    ax_top.set_ylim(0, 3.2)
    ax_top.axis("off")

    # pipeline boxes
    _box(ax_top, (0.2, 1.1), 2.2, 1.0,
         "Query paper\n(abstract)", C_BLUE, alpha=0.85,
         fontsize=10.5, fontweight="bold", rounding=0.10)
    _arrow(ax_top, (2.5, 1.6), (3.2, 1.6), color=C_DARK, lw=1.4)
    _box(ax_top, (3.2, 1.1), 2.2, 1.0,
         "CNN encoder\n+ GCN propagation", C_BLUE, alpha=0.55,
         fontsize=10, fontweight="bold", text_color=C_DARK,
         rounding=0.10)
    _arrow(ax_top, (5.5, 1.6), (6.3, 1.6), color=C_DARK, lw=1.4)
    _box(ax_top, (6.3, 1.1), 2.2, 1.0,
         r"paper embedding $h^p$",
         C_DARK, fontsize=10.5, fontweight="bold", rounding=0.10)
    _arrow(ax_top, (8.6, 1.6), (9.4, 1.6), color=C_DARK, lw=1.4)
    _box(ax_top, (9.4, 1.1), 3.4, 1.0,
         r"dot product vs all $h^r$" + "\n" + "(precomputed index)",
         C_PURPLE, alpha=0.85, fontsize=10.5, fontweight="bold",
         rounding=0.10)

    ax_top.text(6.5, 2.85,
                "Inference path: encode once on the paper side, look up "
                "against all repo embeddings",
                ha="center", fontsize=11, color=C_DARK,
                fontweight="bold")
    ax_top.text(6.5, 0.55,
                "Ranking is a single matrix-vector product against the "
                "repository index.",
                ha="center", fontsize=9.5, color=C_GRAY, fontstyle="italic")

    # bottom: top-K bars
    ax_bot = fig.add_subplot(gs[1])
    repos = [
        "tensorflow/models",
        "huggingface/transformers",
        "facebookresearch/fairseq",
        "pyg-team/pytorch_geometric",
        "deepmind/sonnet",
        "rusty1s/pytorch_scatter",
        "openai/baselines",
        "scikit-learn/scikit-learn",
    ]
    sims = np.array([0.92, 0.88, 0.83, 0.79, 0.74, 0.68, 0.61, 0.55])
    colors = [C_GREEN if s >= 0.80 else C_PURPLE for s in sims]

    y = np.arange(len(repos))[::-1]
    ax_bot.barh(y, sims, color=colors, edgecolor=C_DARK, linewidth=0.8,
                alpha=0.88)
    for yi, s in zip(y, sims):
        ax_bot.text(s + 0.012, yi, f"{s:.2f}",
                    va="center", fontsize=9.5, color=C_DARK)

    ax_bot.set_yticks(y)
    ax_bot.set_yticklabels(repos, fontsize=9.5)
    ax_bot.set_xlim(0, 1.05)
    ax_bot.set_xlabel(r"cosine similarity  $h^p \cdot h^r$",
                      fontsize=10.5)
    ax_bot.set_title("Top-K shortlist (illustrative): green = above shortlist "
                     "threshold",
                     fontsize=11, color=C_DARK, pad=8)
    ax_bot.tick_params(axis="x", labelsize=9)

    fig.suptitle("Figure 4. Recommendation flow: from a query abstract to a "
                 "ranked repo shortlist",
                 fontsize=12.5, color=C_DARK, y=1.02)
    _save(fig, "fig4_recommendation_flow")


# ---------------------------------------------------------------------------
# Figure 5: Evaluation results vs baselines
# ---------------------------------------------------------------------------
def fig5_evaluation_results() -> None:
    """Grouped bars: paper2repo vs 7 baselines on HR@10, MAP@10, MRR@10."""
    fig, ax = plt.subplots(figsize=(12, 5.8))

    # Indicative numbers in the spirit of the paper2repo paper.
    # Absolute values are illustrative; the ordering and the size of the
    # gap on HR@10 follow the trend reported in the paper.
    methods = ["BPR", "MF", "LINE", "NCF", "CDL", "KGCN", "NSCR",
               "paper2repo"]
    hr10  = np.array([0.082, 0.094, 0.118, 0.142, 0.171, 0.198, 0.224, 0.286])
    map10 = np.array([0.041, 0.048, 0.061, 0.074, 0.091, 0.108, 0.122, 0.158])
    mrr10 = np.array([0.063, 0.071, 0.089, 0.108, 0.131, 0.151, 0.172, 0.218])

    x = np.arange(len(methods))
    w = 0.26

    base_color = [C_GRAY] * (len(methods) - 1) + [C_BLUE]
    edge_color = [C_DARK] * len(methods)

    bars1 = ax.bar(x - w, hr10, width=w, label="HR@10",
                   color=base_color, edgecolor=edge_color, linewidth=0.8)
    bars2 = ax.bar(x, map10, width=w, label="MAP@10",
                   color=[c if c != C_BLUE else C_PURPLE
                          for c in base_color],
                   edgecolor=edge_color, linewidth=0.8, alpha=0.92)
    bars3 = ax.bar(x + w, mrr10, width=w, label="MRR@10",
                   color=[c if c != C_BLUE else C_GREEN
                          for c in base_color],
                   edgecolor=edge_color, linewidth=0.8, alpha=0.92)

    # value labels on paper2repo bars
    for b, v in [(bars1[-1], hr10[-1]),
                 (bars2[-1], map10[-1]),
                 (bars3[-1], mrr10[-1])]:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=9,
                color=C_DARK, fontweight="bold")

    # delta over best baseline (NSCR)
    deltas = [
        (bars1[-1], hr10[-1] - hr10[-2]),
        (bars2[-1], map10[-1] - map10[-2]),
        (bars3[-1], mrr10[-1] - mrr10[-2]),
    ]
    for b, d in deltas:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.022,
                f"+{d:.3f}", ha="center", fontsize=8.5,
                color=C_GREEN, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, max(hr10) * 1.30)
    ax.set_title("Figure 5. paper2repo vs seven cross-domain baselines: "
                 "consistent margin, larger gap on HR@10",
                 fontsize=12, color=C_DARK, pad=12)

    handles = [
        Line2D([0], [0], marker="s", color="w", label="HR@10",
               markerfacecolor=C_BLUE, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="s", color="w", label="MAP@10",
               markerfacecolor=C_PURPLE, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="s", color="w", label="MRR@10",
               markerfacecolor=C_GREEN, markeredgecolor=C_DARK,
               markersize=11),
        Line2D([0], [0], marker="s", color="w", label="Baselines",
               markerfacecolor=C_GRAY, markeredgecolor=C_DARK,
               markersize=11),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True,
              fontsize=9.5, ncol=2)

    ax.text(0.0, -0.16,
            "Indicative reproduction of the trend reported in the "
            "paper2repo paper (32K papers / 7.5K repos / 2.1K bridged).",
            transform=ax.transAxes, fontsize=8.8, color=C_GRAY)

    fig.tight_layout()
    _save(fig, "fig5_evaluation_results")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating paper2repo figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")
    fig1_system_architecture()
    fig2_heterogeneous_graph()
    fig3_embedding_objectives()
    fig4_recommendation_flow()
    fig5_evaluation_results()
    print("Done.")


if __name__ == "__main__":
    main()
