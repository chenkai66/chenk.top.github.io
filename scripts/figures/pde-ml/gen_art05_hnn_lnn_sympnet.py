#!/usr/bin/env python3
"""pde-ml Art 05: 3-panel comparison — HNN vs LNN vs SympNet (light theme).

Designed so that long equations (Euler-Lagrange) sit BELOW their box, not inside,
to prevent overlap with adjacent boxes.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "/tmp/pde-ml-figs"
os.makedirs(OUT_DIR, exist_ok=True)

FIG_W, FIG_H = 16, 7.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("#fafaf6")
ax.set_facecolor("#fafaf6")
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.axis("off")

ax.text(FIG_W/2, 7.0,
        "Three structure-preserving neural networks — what each one learns",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1a1a2e")

PANEL_X = [3.0, 8.0, 13.0]
PANEL_COLORS = {"HNN": "#2f80ed", "LNN": "#8e44ad", "SympNet": "#27ae60"}
PANEL_SUBTITLES = {
    "HNN": "learn the Hamiltonian",
    "LNN": "learn the Lagrangian",
    "SympNet": "learn the symplectic map",
}

# Each panel: list of 3 boxes -> (top label, sub-label, equation-below?)
PANELS = {
    "HNN": [
        ("$(q,\\,p)$", "phase-space", None),
        ("$H_\\theta(q,p)$", "scalar net", None),
        ("$\\dot z = J\\nabla H_\\theta$", "Hamilton eqs", None),
    ],
    "LNN": [
        ("$(q,\\,\\dot q)$", "config space", None),
        ("$L_\\theta(q,\\dot q)$", "scalar net", None),
        ("Euler–Lagrange", "2nd-order autodiff",
         "$\\ddot q = (\\partial^2_{\\dot q\\dot q} L)^{-1}[\\dots]$"),
    ],
    "SympNet": [
        ("$(q,\\,p)$", "phase-space", None),
        ("$\\phi_K\\circ\\dots\\circ\\phi_1$", "symplectic blocks", None),
        ("$(q',\\,p')$", "discrete-time map", None),
    ],
}

INFO = {
    "HNN":     ("Pros: exact energy conservation",
                "Needs: $(q,p,\\dot q,\\dot p)$ data",
                "Lives in: phase space"),
    "LNN":     ("Pros: only needs $(q,\\dot q,\\ddot q)$",
                "Cost: matrix inverse / Hessian",
                "Lives in: configuration space"),
    "SympNet": ("Pros: discrete symplecticity by design",
                "No ODE solver needed",
                "Building blocks: shears + activations"),
}

BOX_W, BOX_H = 1.5, 0.9
ROW_Y = 4.5

for cx, alg in zip(PANEL_X, ["HNN", "LNN", "SympNet"]):
    color = PANEL_COLORS[alg]
    # heading
    ax.text(cx - 1.6, 6.3, alg, fontsize=15, fontweight="bold", color=color, ha="left")
    ax.text(cx - 1.0, 6.3, "  —  " + PANEL_SUBTITLES[alg], fontsize=12,
            color=color, ha="left")
    # 3 boxes left -> right
    box_centers = [cx + (i - 1) * (BOX_W + 0.45) for i in range(3)]
    for i, (top, sub, eq_below) in enumerate(PANELS[alg]):
        bx = box_centers[i]
        rect = FancyBboxPatch((bx - BOX_W/2, ROW_Y - BOX_H/2), BOX_W, BOX_H,
                              boxstyle="round,pad=0.08",
                              linewidth=1.6, edgecolor=color, facecolor="white")
        ax.add_patch(rect)
        ax.text(bx, ROW_Y + 0.12, top, ha="center", va="center",
                fontsize=11, color=color)
        ax.text(bx, ROW_Y - 0.22, sub, ha="center", va="center",
                fontsize=8.5, color="#4a5568")
        if eq_below is not None:
            ax.text(bx, ROW_Y - BOX_H/2 - 0.35, eq_below,
                    ha="center", va="top", fontsize=9.5, color="#2c3e50")
        # arrow to next
        if i < 2:
            arrow = FancyArrowPatch((bx + BOX_W/2 + 0.02, ROW_Y),
                                    (box_centers[i+1] - BOX_W/2 - 0.02, ROW_Y),
                                    arrowstyle="->", mutation_scale=14,
                                    linewidth=1.4, color=color)
            ax.add_patch(arrow)
    # info card centered below boxes
    p1, p2, p3 = INFO[alg]
    info_y = 2.2
    info_w = 3.4
    rect = FancyBboxPatch((cx - info_w/2, info_y - 0.65), info_w, 1.1,
                          boxstyle="round,pad=0.1",
                          linewidth=1.0, edgecolor=color, facecolor="#f3f7fb")
    ax.add_patch(rect)
    ax.text(cx, info_y + 0.25, p1, ha="center", va="center", fontsize=9, color="#2c3e50")
    ax.text(cx, info_y,         p2, ha="center", va="center", fontsize=9, color="#2c3e50")
    ax.text(cx, info_y - 0.25,  p3, ha="center", va="center", fontsize=9, color="#2c3e50")

out = os.path.join(OUT_DIR, "fig6_hnn_lnn_sympnet.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"OK: {out}")
