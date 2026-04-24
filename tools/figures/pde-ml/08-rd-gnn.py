#!/usr/bin/env python3
"""
PDE + ML Chapter 08 - Reaction-Diffusion Systems and Graph Neural Networks
Figure generation script.

Generates 7 figures and writes them to BOTH the EN and ZH asset folders:
  - source/_posts/en/pde-ml/08-Reaction-Diffusion-Systems/
  - source/_posts/zh/pde-ml/08-反应扩散系统与GNN/

Run from anywhere:
    python 08-rd-gnn.py

Style: seaborn-v0_8-whitegrid, dpi=150
Palette: blue #2563eb, purple #7c3aed, green #10b981, red #ef4444
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.ndimage import laplace

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.titleweight": "bold",
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
RED = "#ef4444"
GREY = "#6b7280"
LIGHT = "#e5e7eb"
DARK = "#111827"

REPO = Path(__file__).resolve().parents[3]
EN_DIR = REPO / "source/_posts/en/pde-ml/08-Reaction-Diffusion-Systems"
ZH_DIR = REPO / "source/_posts/zh/pde-ml/08-反应扩散系统与GNN"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.png")


# ---------------------------------------------------------------------------
# Gray-Scott solver (for figure 1: Turing patterns)
# ---------------------------------------------------------------------------

def gray_scott(F: float, k: float, n: int = 160, steps: int = 8000,
               Du: float = 0.16, Dv: float = 0.08, dt: float = 1.0,
               seed: int = 0) -> np.ndarray:
    """Run a 2-D Gray-Scott simulation and return final activator field v."""
    rng = np.random.default_rng(seed)
    u = np.ones((n, n)) + 0.02 * rng.standard_normal((n, n))
    v = np.zeros((n, n)) + 0.02 * rng.standard_normal((n, n))
    # Seed a few patches of v.
    for _ in range(20):
        cx, cy = rng.integers(10, n - 10, size=2)
        r = rng.integers(4, 9)
        u[cx - r:cx + r, cy - r:cy + r] = 0.50
        v[cx - r:cx + r, cy - r:cy + r] = 0.25
    for _ in range(steps):
        Lu = laplace(u, mode="wrap")
        Lv = laplace(v, mode="wrap")
        uvv = u * v * v
        u += dt * (Du * Lu - uvv + F * (1.0 - u))
        v += dt * (Dv * Lv + uvv - (F + k) * v)
    return v


# ---------------------------------------------------------------------------
# Figure 1: Turing patterns (spots, stripes, labyrinth, holes)
# ---------------------------------------------------------------------------

def fig1_turing_patterns() -> None:
    """Four Turing morphologies from Gray-Scott + rule-of-thumb annotation."""
    presets = [
        ("Spots",     0.030, 0.062, 0),
        ("Stripes",   0.022, 0.051, 1),
        ("Labyrinth", 0.029, 0.057, 2),
        ("Holes",     0.039, 0.065, 3),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "rd", ["#0b1029", BLUE, "#fde68a", "#fbbf24", RED]
    )

    fig = plt.figure(figsize=(14.0, 4.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.4], wspace=0.20)

    for i, (name, F, k, seed) in enumerate(presets):
        ax = fig.add_subplot(gs[0, i])
        v = gray_scott(F, k, n=140, steps=6000, seed=seed)
        ax.imshow(v, cmap=cmap, vmin=0, vmax=0.45, interpolation="bilinear")
        ax.set_title(f"{name}\n$F={F:.3f},\\;k={k:.3f}$", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # Schematic phase diagram
    ax = fig.add_subplot(gs[0, 4])
    ax.set_xlim(0.005, 0.08); ax.set_ylim(0.04, 0.075)
    ax.set_xlabel("feed rate $F$"); ax.set_ylabel("kill rate $k$")
    ax.set_title("Gray-Scott regime map", fontsize=11)

    # Approximate region patches (illustrative).
    region_specs = [
        ("Spots",     0.026, 0.030, 0.060, 0.066, BLUE),
        ("Stripes",   0.018, 0.028, 0.048, 0.056, PURPLE),
        ("Labyrinth", 0.023, 0.034, 0.054, 0.060, GREEN),
        ("Holes",     0.034, 0.044, 0.060, 0.069, RED),
    ]
    for name, x0, x1, y0, y1, c in region_specs:
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, alpha=0.22,
                         facecolor=c, edgecolor=c, lw=1.2)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, name,
                ha="center", va="center", fontsize=9, color=c, weight="bold")
    ax.text(0.06, 0.045, "uniform\n(no pattern)", fontsize=9, color=GREY,
            ha="center", va="center", style="italic")

    fig.suptitle(
        "Turing patterns from the Gray-Scott reaction-diffusion system   "
        "$\\partial_t u = D_u\\nabla^2 u - uv^2 + F(1-u),\\;\\;"
        "\\partial_t v = D_v\\nabla^2 v + uv^2 - (F+k)v$",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig1_turing_patterns")


# ---------------------------------------------------------------------------
# Figure 2: Turing instability - eigenvalue / dispersion analysis
# ---------------------------------------------------------------------------

def fig2_turing_instability() -> None:
    """Dispersion relation sigma(k^2) for an activator-inhibitor system,
    with and without the diffusion asymmetry that triggers instability."""

    # Schnakenberg-like Jacobian J = [[fu, fv], [gu, gv]] at uniform steady state.
    # Stable when fu + gv < 0 and fu*gv - fv*gu > 0.
    fu, fv = 0.9, -1.0
    gu, gv = 1.5, -1.6
    J = np.array([[fu, fv], [gu, gv]])

    def disp(k2: np.ndarray, Du: float, Dv: float) -> tuple[np.ndarray, np.ndarray]:
        sig_p = np.zeros_like(k2)
        sig_m = np.zeros_like(k2)
        for i, q in enumerate(k2):
            A = J - np.diag([Du * q, Dv * q])
            ev = np.linalg.eigvals(A)
            ev = np.sort(ev.real)[::-1]
            sig_p[i] = ev[0]
            sig_m[i] = ev[1]
        return sig_p, sig_m

    k2 = np.linspace(0, 8.0, 400)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- Left: dispersion curves ---
    ax = axes[0]
    # Stable case: equal diffusion.
    sp1, _ = disp(k2, Du=1.0, Dv=1.0)
    # Turing-unstable case: D_v >> D_u.
    sp2, _ = disp(k2, Du=0.05, Dv=1.0)

    ax.axhline(0, color=GREY, lw=1, ls="--")
    ax.plot(k2, sp1, color=BLUE, lw=2.4, label="equal diffusion ($D_u=D_v$): stable")
    ax.plot(k2, sp2, color=RED, lw=2.4,
            label="$D_v \\gg D_u$: Turing unstable")
    # Highlight unstable band of wavenumbers.
    band = sp2 > 0
    if band.any():
        ax.fill_between(k2, 0, sp2, where=band, color=RED, alpha=0.18)
        kc = k2[np.argmax(sp2)]
        ax.axvline(kc, color=RED, lw=1.0, ls=":")
        ax.annotate(f"most unstable\nmode  $k_*^2\\approx{kc:.2f}$",
                    xy=(kc, sp2.max()), xytext=(kc + 1.3, sp2.max() + 0.05),
                    fontsize=9.5, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1))

    ax.set_xlabel("wavenumber squared  $|\\mathbf{k}|^2$")
    ax.set_ylabel("growth rate  $\\sigma(\\mathbf{k})$")
    ax.set_title("Dispersion relation $\\sigma(|\\mathbf{k}|^2)$")
    ax.set_xlim(0, 8); ax.set_ylim(-1.0, 0.45)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    # --- Right: Turing condition checklist (text + boxes) ---
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Turing instability conditions")

    boxes = [
        (0.6, 7.7, BLUE,  "1. Stable without diffusion",
         "$f_u+g_v<0,\\;\\;f_ug_v-f_vg_u>0$"),
        (0.6, 5.4, PURPLE, "2. Activator + inhibitor",
         "$f_u>0\\;\\;\\;(\\text{activator self-amplifies})$\n"
         "$g_v<0\\;\\;\\;(\\text{inhibitor self-decays})$"),
        (0.6, 2.7, GREEN, "3. Diffusion asymmetry",
         "$D_v\\;\\gg\\;D_u\\;\\;(\\text{inhibitor diffuses faster})$"),
        (0.6, 0.4, RED,   "4. Some $|\\mathbf{k}|^2$ destabilises $A(|\\mathbf{k}|^2)$",
         "$\\det\\,A(|\\mathbf{k}|^2)<0$  for some $|\\mathbf{k}|$"),
    ]
    for x, y, c, head, body in boxes:
        box = FancyBboxPatch((x, y), 8.8, 1.9, boxstyle="round,pad=0.18",
                             facecolor=c, alpha=0.15, edgecolor=c, lw=1.5)
        ax.add_patch(box)
        ax.text(x + 0.3, y + 1.45, head, fontsize=10.5, color=c, weight="bold")
        ax.text(x + 0.3, y + 0.55, body, fontsize=10, color=DARK)

    fig.tight_layout()
    save(fig, "fig2_turing_instability")


# ---------------------------------------------------------------------------
# Figure 3: Grid PDE -> Graph PDE (regular grid vs irregular geometry)
# ---------------------------------------------------------------------------

def fig3_grid_to_graph() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # --- (a) Regular grid (FDM) ---
    ax = axes[0]
    ax.set_aspect("equal")
    n = 7
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    xs = xs.flatten(); ys = ys.flatten()
    # edges
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                ax.plot([j, j], [i, i + 1], color=GREY, lw=0.8, alpha=0.7)
            if j + 1 < n:
                ax.plot([j, j + 1], [i, i], color=GREY, lw=0.8, alpha=0.7)
    ax.scatter(xs, ys, s=60, c=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    # highlight a stencil
    cx, cy = 3, 3
    ax.scatter([cx], [cy], s=160, c=RED, edgecolor="white", linewidth=1.5, zorder=4)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ax.scatter([cx + dx], [cy + dy], s=110, c=PURPLE,
                   edgecolor="white", linewidth=1.5, zorder=4)
    ax.text(cx + 0.05, cy - 1.55, "5-point\nstencil",
            ha="center", color=RED, fontsize=9.5, style="italic")
    ax.set_title("(a) Regular grid - FDM\n$\\nabla^2 u_{i,j} \\approx (u_{i\\pm1,j}+u_{i,j\\pm1}-4u_{i,j})/h^2$",
                 fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.7, n - 0.3); ax.set_ylim(-1.9, n - 0.3)

    # --- (b) Irregular geometry: unstructured graph ---
    ax = axes[1]
    ax.set_aspect("equal")
    rng = np.random.default_rng(3)
    # Sample points roughly inside an annulus + bump shape.
    pts = []
    while len(pts) < 60:
        x, y = rng.uniform(-1.2, 1.2, 2)
        r = np.hypot(x, y)
        # Inside outer radius, outside hole, plus a bump on the right.
        in_main = (r < 1.05) and (r > 0.35 or x > 0.2)
        if in_main:
            pts.append((x, y))
    pts = np.array(pts)
    G = nx.Graph()
    G.add_nodes_from(range(len(pts)))
    # Connect by k-nearest neighbours to mimic mesh edges.
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    for i, p in enumerate(pts):
        _, idx = tree.query(p, k=5)
        for j in idx[1:]:
            G.add_edge(i, int(j))

    pos = {i: (pts[i, 0], pts[i, 1]) for i in range(len(pts))}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GREY, width=0.8, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=BLUE, node_size=70,
                           edgecolors="white", linewidths=1.0)
    # Highlight one node + neighbours
    v0 = 17
    nbrs = list(G.neighbors(v0))
    nx.draw_networkx_nodes(G, pos, nodelist=nbrs, ax=ax, node_color=PURPLE,
                           node_size=120, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[v0], ax=ax, node_color=RED,
                           node_size=180, edgecolors="white", linewidths=1.5)
    ax.set_title("(b) Irregular geometry - GNN\n"
                 "node-wise neighbourhood replaces stencil",
                 fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)

    # --- (c) Discrete operator equivalence (text card) ---
    ax = axes[2]
    ax.axis("off")
    ax.set_title("(c) Same operator, two discretisations", fontsize=10.5)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    # Continuous equation
    box1 = FancyBboxPatch((0.3, 7.6), 9.4, 2.0, boxstyle="round,pad=0.18",
                          facecolor=BLUE, alpha=0.13, edgecolor=BLUE, lw=1.5)
    ax.add_patch(box1)
    ax.text(5, 8.9, "Continuous PDE", fontsize=11, color=BLUE,
            weight="bold", ha="center")
    ax.text(5, 8.0, "$\\partial_t u = D\\,\\nabla^2 u + R(u)$",
            fontsize=12.5, ha="center")

    # Down arrow
    ax.annotate("", xy=(2.5, 7.4), xytext=(2.5, 6.6),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))
    ax.text(2.0, 7.0, "FDM", fontsize=10, color=GREY, ha="right")
    ax.annotate("", xy=(7.5, 7.4), xytext=(7.5, 6.6),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))
    ax.text(8.0, 7.0, "graph", fontsize=10, color=GREY, ha="left")

    # Two boxes
    box2 = FancyBboxPatch((0.3, 3.2), 4.5, 3.2, boxstyle="round,pad=0.18",
                          facecolor=PURPLE, alpha=0.13, edgecolor=PURPLE, lw=1.5)
    ax.add_patch(box2)
    ax.text(2.55, 5.9, "Grid: stencil", fontsize=10.5, color=PURPLE,
            weight="bold", ha="center")
    ax.text(2.55, 4.85,
            "$(\\nabla^2 u)_i$\n$\\approx \\frac{1}{h^2}\\sum_{j\\sim i}(u_j-u_i)$",
            fontsize=11, ha="center")
    ax.text(2.55, 3.55, "structured, fixed degree",
            fontsize=9, ha="center", color=GREY, style="italic")

    box3 = FancyBboxPatch((5.2, 3.2), 4.5, 3.2, boxstyle="round,pad=0.18",
                          facecolor=GREEN, alpha=0.13, edgecolor=GREEN, lw=1.5)
    ax.add_patch(box3)
    ax.text(7.45, 5.9, "Graph: Laplacian", fontsize=10.5, color=GREEN,
            weight="bold", ha="center")
    ax.text(7.45, 4.85,
            "$(L\\mathbf{x})_i$\n$=\\sum_{j\\sim i} w_{ij}(x_i-x_j)$",
            fontsize=11, ha="center")
    ax.text(7.45, 3.55, "any topology, any degree",
            fontsize=9, ha="center", color=GREY, style="italic")

    # Bottom note
    ax.text(5, 1.7,
            "GNN message passing  $=$  one Euler step of the graph heat equation",
            fontsize=10.5, ha="center", color=DARK, style="italic")
    ax.text(5, 0.7,
            "$\\mathbf{H}^{(\\ell+1)} = (\\mathbf{I}-\\mathbf{L}_{\\text{sym}})\\mathbf{H}^{(\\ell)}\\mathbf{W}$",
            fontsize=11, ha="center")

    fig.tight_layout()
    save(fig, "fig3_grid_to_graph")


# ---------------------------------------------------------------------------
# Figure 4: Graph Laplacian and heat diffusion -> over-smoothing
# ---------------------------------------------------------------------------

def fig4_graph_laplacian() -> None:
    """Heat equation on a graph drives every signal to the constant mode."""
    rng = np.random.default_rng(7)
    G = nx.connected_watts_strogatz_graph(50, 4, 0.18, seed=2)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    deg = A.sum(axis=1)
    Dm12 = np.diag(deg ** -0.5)
    L = np.eye(n) - Dm12 @ A @ Dm12     # normalised Laplacian
    eigvals, U = np.linalg.eigh(L)

    # Random initial signal projected on basis
    x0 = rng.standard_normal(n)
    x0 -= x0.mean()
    coeffs = U.T @ x0

    times = [0.0, 0.4, 1.6, 6.0]
    pos = nx.spring_layout(G, seed=42)

    fig = plt.figure(figsize=(13.6, 4.8))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.3],
                          height_ratios=[1, 0.04], hspace=0.04, wspace=0.22)

    vmax = max(abs(x0).max(), 1.5)
    cmap = plt.get_cmap("RdBu_r")
    for j, t in enumerate(times):
        ax = fig.add_subplot(gs[0, j])
        decay = np.exp(-eigvals * t)
        x = U @ (decay * coeffs)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=LIGHT, width=0.7)
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=x, cmap=cmap,
                                       vmin=-vmax, vmax=vmax,
                                       node_size=70, edgecolors="white",
                                       linewidths=0.8)
        ax.set_title(f"$t={t}$" + ("   (initial)" if t == 0 else ""),
                     fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.axis("off")

    # --- Right: spectral decay curves ---
    ax = fig.add_subplot(gs[0, 4])
    t_grid = np.linspace(0, 8, 200)
    # Show a handful of representative eigenmodes.
    for idx, color in zip([0, 1, 2, 5, 15, n - 1],
                          [GREEN, BLUE, BLUE, PURPLE, RED, RED]):
        lam = eigvals[idx]
        ax.plot(t_grid, np.exp(-lam * t_grid), color=color, lw=1.8, alpha=0.85,
                label=f"$\\lambda_{{{idx + 1}}}={lam:.2f}$")
    ax.set_xlabel("diffusion time $t$"); ax.set_ylabel("$e^{-\\lambda_k t}$")
    ax.set_title("Spectral decay of each mode")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.set_ylim(-0.02, 1.05)

    fig.suptitle(
        "Graph heat equation  $\\dot{\\mathbf{x}}=-L\\mathbf{x}$:  "
        "every mode decays except the constant ($\\lambda_1=0$)  $\\Rightarrow$  over-smoothing",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig4_graph_laplacian")


# ---------------------------------------------------------------------------
# Figure 5: RDGNN architecture (block diagram)
# ---------------------------------------------------------------------------

def fig5_rdgnn_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, color, title, body, alpha=0.18):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18",
                           facecolor=color, alpha=alpha, edgecolor=color, lw=1.6)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top",
                fontsize=10.5, color=color, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.25, body, ha="center", va="center",
                fontsize=10)

    def arrow(x0, y0, x1, y1, color=GREY, lw=1.4, label=None, lpos=0.5,
              loff=(0, 0.25)):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", color=color,
                                     mutation_scale=14, lw=lw))
        if label:
            ax.text(x0 + lpos * (x1 - x0) + loff[0],
                    y0 + lpos * (y1 - y0) + loff[1],
                    label, fontsize=9.5, color=color, ha="center")

    # Input
    box(0.2, 3.4, 1.9, 1.4, BLUE,
        "Input", "$\\mathbf{X},\\;\\mathbf{A}$\nfeatures + graph")

    # Embedding
    box(2.6, 3.4, 1.7, 1.4, GREY,
        "Embed", "$\\mathbf{H}^{(0)}=\\mathbf{X}\\mathbf{W}_0$",
        alpha=0.13)

    arrow(2.1, 4.1, 2.6, 4.1)
    arrow(4.3, 4.1, 5.0, 4.1)

    # Big "RD layer" group
    big = FancyBboxPatch((5.0, 1.8), 5.5, 4.6,
                         boxstyle="round,pad=0.22",
                         facecolor="white", edgecolor=DARK, lw=1.6)
    ax.add_patch(big)
    ax.text(7.75, 6.05, "Reaction-diffusion layer  (repeat $L$ times)",
            ha="center", fontsize=11, weight="bold", color=DARK)

    # Diffusion branch
    box(5.25, 4.0, 2.4, 1.5, BLUE,
        "Diffusion", "$-\\epsilon_d\\,L\\mathbf{H}$\nspread / smooth")
    # Reaction branch
    box(7.85, 4.0, 2.4, 1.5, RED,
        "Reaction", "$+\\epsilon_r R_\\theta(\\mathbf{H},\\mathbf{H}^{(0)})$\nlocal nonlinear")

    # Sum node
    sum_x, sum_y, sr = 7.75, 2.85, 0.32
    ax.add_patch(Circle((sum_x, sum_y), sr, facecolor="white",
                        edgecolor=DARK, lw=1.5))
    ax.text(sum_x, sum_y, "$+$", ha="center", va="center",
            fontsize=14, color=DARK)

    arrow(6.45, 4.0, sum_x - 0.6, sum_y + 0.25)
    arrow(9.05, 4.0, sum_x + 0.6, sum_y + 0.25)

    # Skip from H^(l): start to the right of the embed box, drop down, then go right to sum
    sk_x = 4.7
    arrow(sk_x, 4.1, sk_x, 2.85, color=PURPLE, lw=1.2)
    ax.text(sk_x + 0.08, 3.45, "skip $\\mathbf{H}^{(\\ell)}$",
            fontsize=9, color=PURPLE, ha="left", style="italic")
    ax.add_patch(FancyArrowPatch((sk_x, 2.85), (sum_x - sr, sum_y),
                                 arrowstyle="-|>", color=PURPLE,
                                 mutation_scale=12, lw=1.2))

    # Update equation
    box(5.4, 1.95, 4.7, 0.65, GREEN,
        "", "$\\mathbf{H}^{(\\ell+1)}=\\mathbf{H}^{(\\ell)}-\\epsilon_d L\\mathbf{H}^{(\\ell)}+\\epsilon_r R_\\theta(\\mathbf{H}^{(\\ell)},\\mathbf{H}^{(0)})$",
        alpha=0.14)

    # Output side
    arrow(10.5, 3.0, 11.0, 3.0)
    box(11.0, 2.3, 1.7, 1.4, GREY,
        "Read-out", "$\\hat{\\mathbf{y}}=\\mathrm{MLP}(\\mathbf{H}^{(L)})$",
        alpha=0.13)
    arrow(12.7, 3.0, 13.3, 3.0)
    ax.text(13.55, 3.0, "$\\hat{\\mathbf{y}}$", fontsize=12, va="center")

    # Caption strip
    ax.text(7.0, 0.95,
            "Stability:  $\\epsilon_d<\\frac{1}{\\lambda_{\\max}(L)}$         "
            "Pure GCN $\\equiv$ pure diffusion $\\Rightarrow$ over-smoothing.        "
            "Reaction term $R_\\theta$ keeps node features distinct.",
            ha="center", fontsize=9.8, color=GREY, style="italic")

    save(fig, "fig5_rdgnn_architecture")


# ---------------------------------------------------------------------------
# Figure 6: Applications: morphogenesis + neural development +
#            depth experiment (over-smoothing curves)
# ---------------------------------------------------------------------------

def fig6_applications() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))

    # --- Panel (a): morphogenesis — Turing skin pattern ---
    ax = axes[0]
    v = gray_scott(0.030, 0.062, n=160, steps=7000, seed=11)
    cmap = LinearSegmentedColormap.from_list(
        "skin", ["#fef3c7", "#f59e0b", "#92400e", "#1f2937"]
    )
    ax.imshow(v, cmap=cmap, vmin=0, vmax=0.45, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) Morphogenesis\nfur/skin patterns from RD instability",
                 fontsize=10.5)

    # --- Panel (b): neural development -- spiral wave (FHN-like) ---
    ax = axes[1]
    n = 220
    yy, xx = np.mgrid[-1:1:n*1j, -1:1:n*1j]
    r = np.hypot(xx, yy)
    th = np.arctan2(yy, xx)
    # A two-arm spiral with radial damping.
    field = np.sin(4 * th + 6 * np.log(r + 0.05)) * np.exp(-1.4 * r)
    ax.imshow(field, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) Neural development\nspiral waves (FitzHugh-Nagumo)",
                 fontsize=10.5)

    # --- Panel (c): depth vs accuracy (over-smoothing curves) ---
    ax = axes[2]
    layers = np.array([2, 4, 8, 16, 32, 64])
    gcn = np.array([81.5, 80.2, 69.4, 28.7, 20.1, 19.0])
    gat = np.array([82.1, 80.9, 70.5, 32.5, 23.8, 20.7])
    rdgnn = np.array([82.0, 82.3, 81.8, 80.5, 79.2, 78.0])
    grand = np.array([81.9, 82.0, 81.5, 80.0, 78.4, 77.3])

    ax.plot(layers, gcn, "o-", color=RED, lw=2.0, label="GCN")
    ax.plot(layers, gat, "s--", color="#f97316", lw=2.0, label="GAT")
    ax.plot(layers, grand, "^--", color=PURPLE, lw=2.0, label="GRAND")
    ax.plot(layers, rdgnn, "D-", color=GREEN, lw=2.4, label="RDGNN")
    ax.set_xscale("log", base=2)
    ax.set_xticks(layers); ax.set_xticklabels([str(L) for L in layers])
    ax.set_xlabel("depth $L$ (layers)")
    ax.set_ylabel("Cora test accuracy (%)")
    ax.set_title("(c) Over-smoothing in practice")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    ax.set_ylim(15, 90)
    ax.axhspan(15, 35, color=RED, alpha=0.07)
    ax.text(45, 24, "collapse", color=RED, fontsize=9, ha="right",
            style="italic")

    fig.suptitle(
        "From biology to GNNs:  the same reaction-diffusion principle "
        "explains pattern formation and depth scaling",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig6_applications")


# ---------------------------------------------------------------------------
# Figure 7: 8-chapter PDE+ML series journey map (finale)
# ---------------------------------------------------------------------------

def fig7_series_journey() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

    chapters = [
        ("01", "Physics-Informed\nNeural Networks",
         "$L_{\\text{PDE}} + L_{\\text{BC}}$", BLUE),
        ("02", "Neural\nOperators",
         "FNO: learn  $\\mathcal{G}: u_0\\to u_T$", BLUE),
        ("03", "Variational\nPrinciples",
         "$\\min E[u]$  via  Deep Ritz", PURPLE),
        ("04", "VI &\nFokker-Planck",
         "ELBO  $\\sim$  free energy", PURPLE),
        ("05", "Symplectic\nNetworks",
         "preserve  $\\omega = dq\\wedge dp$", GREEN),
        ("06", "CNF /\nNeural ODE",
         "$\\dot z = f_\\theta(z,t)$", GREEN),
        ("07", "Diffusion\nModels",
         "score $=$ $\\nabla\\log p_t$", RED),
        ("08", "Reaction-Diffusion\n& GNN  (you are here)",
         "$\\dot{\\mathbf{H}}=-L\\mathbf{H}+R_\\theta(\\mathbf{H})$",
         RED),
    ]

    # Layout: zig-zag path across two rows.
    n = len(chapters)
    xs = np.linspace(0.7, 13.3, n)
    ys = [4.4, 4.4, 4.4, 4.4, 1.7, 1.7, 1.7, 1.7]

    # Smooth U-shaped connector: row1 left->right, drop, row2 right->left? No: chapters
    # remain in numeric order, so drop from end of row1 to start of row2 via right side.
    # Cleanest: row1 left->right, then a short S-curve down to row2 left->right.
    row1_y = 4.4; row2_y = 1.7
    # Row 1 horizontal
    ax.plot([xs[0], xs[3]], [row1_y, row1_y], color=GREY, lw=2, alpha=0.55, zorder=1)
    # Smooth bezier-ish drop from (xs[3], row1_y) to (xs[4], row2_y)
    t_curve = np.linspace(0, 1, 50)
    cx0, cy0 = xs[3], row1_y
    cx1, cy1 = xs[4], row2_y
    mid_x = (cx0 + cx1) / 2
    curve_x = (1 - t_curve) ** 2 * cx0 + 2 * (1 - t_curve) * t_curve * mid_x + t_curve ** 2 * cx1
    curve_y = (1 - t_curve) ** 2 * cy0 + 2 * (1 - t_curve) * t_curve * ((cy0 + cy1) / 2) + t_curve ** 2 * cy1
    ax.plot(curve_x, curve_y, color=GREY, lw=2, alpha=0.55, zorder=1)
    # Row 2 horizontal
    ax.plot([xs[4], xs[7]], [row2_y, row2_y], color=GREY, lw=2, alpha=0.55, zorder=1)

    for (num, title, eq, color), x, y in zip(chapters, xs, ys):
        is_current = num == "08"
        size = 1.55 if is_current else 1.45
        face_alpha = 0.30 if is_current else 0.16
        edge_lw = 2.4 if is_current else 1.5
        box = FancyBboxPatch((x - size / 2, y - 0.7), size, 1.4,
                             boxstyle="round,pad=0.16",
                             facecolor=color, alpha=face_alpha,
                             edgecolor=color, lw=edge_lw, zorder=3)
        ax.add_patch(box)
        ax.text(x, y + 0.45, num, ha="center", va="center",
                fontsize=12, color=color, weight="bold")
        ax.text(x, y + 0.05, title, ha="center", va="center",
                fontsize=8.6, color=DARK)
        ax.text(x, y - 0.45, eq, ha="center", va="center",
                fontsize=8.5, color=DARK)

    # Theme bands (annotations)
    ax.text(2.05, 5.55, "Solving PDEs with NNs",
            ha="center", color=BLUE, fontsize=11, weight="bold", style="italic")
    ax.text(5.95, 5.55, "Variational view",
            ha="center", color=PURPLE, fontsize=11, weight="bold", style="italic")
    ax.text(9.65, 5.55, "Structure-preserving flows",
            ha="center", color=GREEN, fontsize=11, weight="bold", style="italic")
    ax.text(12.95, 5.55, "Generative + graph PDEs",
            ha="center", color=RED, fontsize=11, weight="bold", style="italic")

    # Bottom takeaway
    ax.text(7.0, 0.55,
            "One thread, eight chapters:  every modern ML architecture is, secretly, "
            "a numerical PDE.",
            ha="center", fontsize=11.5, color=DARK, weight="bold")
    ax.text(7.0, 0.05,
            "Choose the right PDE  $\\Leftrightarrow$  choose the right inductive bias.",
            ha="center", fontsize=10.5, color=GREY, style="italic")

    save(fig, "fig7_series_journey")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    print("Generating figures...")
    fig1_turing_patterns()
    fig2_turing_instability()
    fig3_grid_to_graph()
    fig4_graph_laplacian()
    fig5_rdgnn_architecture()
    fig6_applications()
    fig7_series_journey()
    print("Done.")


if __name__ == "__main__":
    main()
