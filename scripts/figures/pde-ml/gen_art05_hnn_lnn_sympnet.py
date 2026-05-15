#!/usr/bin/env python3
"""pde-ml Art 05: 3-panel comparison — HNN vs LNN vs SympNet (light theme).

Each panel is rendered in its OWN matplotlib subplot via gridspec, so the
three flow diagrams cannot bleed into one another. Long equations sit
under the relevant box as captions instead of inside it.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

OUT_DIR = "/tmp/pde-ml-figs"
os.makedirs(OUT_DIR, exist_ok=True)

BG = "#fafaf6"
PANEL_COLORS = {"HNN": "#2f80ed", "LNN": "#8e44ad", "SympNet": "#27ae60"}
SUBTITLES = {
    "HNN": "learn the Hamiltonian",
    "LNN": "learn the Lagrangian",
    "SympNet": "learn the symplectic map",
}
PANELS = {
    "HNN": [
        ("$(q,\\,p)$",            "phase-space",   None),
        ("$H_\\theta(q,p)$",      "scalar net",    None),
        ("$\\dot z = J\\nabla H_\\theta$", "Hamilton eqs", None),
    ],
    "LNN": [
        ("$(q,\\,\\dot q)$",      "config space",  None),
        ("$L_\\theta(q,\\dot q)$","scalar net",    None),
        ("Euler–Lagrange",        "2nd-order autodiff",
         "$\\ddot q = (\\partial^2_{\\dot q\\dot q} L)^{-1}[\\dots]$"),
    ],
    "SympNet": [
        ("$(q,\\,p)$",                       "phase-space",       None),
        ("$\\phi_K\\!\\circ\\!\\dots\\!\\circ\\!\\phi_1$", "symplectic blocks", None),
        ("$(q',\\,p')$",                     "discrete-time map", None),
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

fig = plt.figure(figsize=(17, 7.5))
fig.patch.set_facecolor(BG)
gs = GridSpec(1, 3, figure=fig, wspace=0.18, left=0.02, right=0.98, top=0.86, bottom=0.05)

fig.suptitle("Three structure-preserving neural networks — what each one learns",
             color="#1a1a2e", fontsize=14, fontweight="bold", y=0.97)

BOX_W, BOX_H = 1.5, 0.95
ROW_Y = 4.0   # in panel coords

for col, alg in enumerate(["HNN", "LNN", "SympNet"]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 6); ax.set_ylim(0, 7)
    ax.axis("off")
    color = PANEL_COLORS[alg]

    # heading: combined on one line, centered
    ax.text(3.0, 6.3, alg, ha="right", va="center",
            fontsize=15, fontweight="bold", color=color)
    ax.text(3.1, 6.3, "  —  " + SUBTITLES[alg], ha="left", va="center",
            fontsize=12, color=color)

    # 3 boxes left -> right, centered horizontally
    pitch = BOX_W + 0.35
    box_centers = [3.0 + (i - 1) * pitch for i in range(3)]
    for i, (top, sub, eq_below) in enumerate(PANELS[alg]):
        bx = box_centers[i]
        rect = FancyBboxPatch((bx - BOX_W/2, ROW_Y - BOX_H/2), BOX_W, BOX_H,
                              boxstyle="round,pad=0.08",
                              linewidth=1.6, edgecolor=color, facecolor="white")
        ax.add_patch(rect)
        ax.text(bx, ROW_Y + 0.16, top, ha="center", va="center",
                fontsize=10.5, color=color)
        ax.text(bx, ROW_Y - 0.22, sub, ha="center", va="center",
                fontsize=8.5, color="#4a5568")
        if eq_below:
            ax.text(bx, ROW_Y - BOX_H/2 - 0.45, eq_below,
                    ha="center", va="top", fontsize=9.5, color="#2c3e50")
        if i < 2:
            arrow = FancyArrowPatch((bx + BOX_W/2 + 0.04, ROW_Y),
                                    (box_centers[i+1] - BOX_W/2 - 0.04, ROW_Y),
                                    arrowstyle="->", mutation_scale=14,
                                    linewidth=1.4, color=color)
            ax.add_patch(arrow)

    # info card centered below
    p1, p2, p3 = INFO[alg]
    info_y = 1.7
    info_w = 4.6
    info_rect = FancyBboxPatch((3.0 - info_w/2, info_y - 0.7), info_w, 1.3,
                               boxstyle="round,pad=0.1",
                               linewidth=1.0, edgecolor=color, facecolor="#f3f7fb")
    ax.add_patch(info_rect)
    ax.text(3.0, info_y + 0.32, p1, ha="center", va="center",
            fontsize=9.5, color="#2c3e50")
    ax.text(3.0, info_y,         p2, ha="center", va="center",
            fontsize=9.5, color="#2c3e50")
    ax.text(3.0, info_y - 0.32,  p3, ha="center", va="center",
            fontsize=9.5, color="#2c3e50")

out = os.path.join(OUT_DIR, "fig6_hnn_lnn_sympnet.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"OK: {out}")
