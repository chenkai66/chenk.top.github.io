#!/usr/bin/env python3
"""Auto-Scaling Lifecycle (EN)."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withSimplePatchShadow

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)

fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 4)
ax.axis("off")

def draw_box(ax, cx, cy, w, h, color, label, fontsize=11, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    box.set_path_effects([withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, rad=0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=f"arc3,rad={rad}",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)

# --- Scale-out row (top, y=3.0) ---
so_y = 3.0
draw_box(ax, 1.8, so_y, 2.6, 0.7, RED, "CPU > 70%\n3 min", fontsize=10.5, bold=True)
draw_arrow(ax, 3.15, so_y, 4.15, so_y)
draw_box(ax, 5.5, so_y, 2.4, 0.7, PURPLE, "Add N\nInstances", fontsize=10.5)
draw_arrow(ax, 6.75, so_y, 7.55, so_y)
draw_box(ax, 9.0, so_y, 2.4, 0.7, BLUE, "Health Check\n& Route", fontsize=10.5)
draw_arrow(ax, 10.25, so_y, 11.0, so_y)
draw_box(ax, 11.8, so_y, 1.4, 0.7, GREEN, "Ready", fontsize=11, bold=True)

ax.text(6.5, 3.6, "SCALE OUT", ha="center", va="bottom",
        fontsize=10, color=PURPLE, fontweight="bold")

# --- Scale-in row (bottom, y=1.0) ---
si_y = 1.0
draw_box(ax, 1.8, si_y, 2.6, 0.7, GREEN, "CPU < 30%\n10 min", fontsize=10.5, bold=True)
draw_arrow(ax, 3.15, si_y, 4.15, si_y)
draw_box(ax, 5.5, si_y, 2.4, 0.7, AMBER, "Drain\nConnections", fontsize=10.5)
draw_arrow(ax, 6.75, si_y, 7.55, si_y)
draw_box(ax, 9.0, si_y, 2.4, 0.7, BLUE, "Remove\nInstances", fontsize=10.5)
draw_arrow(ax, 10.25, si_y, 11.0, si_y)
draw_box(ax, 11.8, si_y, 1.4, 0.7, RED, "Scaled", fontsize=11, bold=True)

ax.text(6.5, 0.35, "SCALE IN", ha="center", va="bottom",
        fontsize=10, color=AMBER, fontweight="bold")

# Dashed separator
ax.axhline(y=2.0, xmin=0.04, xmax=0.96, color="#d1d5db",
           linestyle="--", lw=1, zorder=1)

ax.set_title("Auto-Scaling Lifecycle",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_cloud_autoscale_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_cloud_autoscale_en.png")
