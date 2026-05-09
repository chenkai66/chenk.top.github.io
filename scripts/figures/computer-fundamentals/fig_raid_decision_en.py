#!/usr/bin/env python3
"""RAID Level Decision Tree (EN)."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.5)
ax.axis("off")


def draw_box(ax, cx, cy, w, h, color, label, fontsize=11, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    box.set_path_effects([pe.withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label="", label_offset=(0, 0.15)):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=9.5,
                color="#374151", fontweight="bold", zorder=4)


# Root: Need redundancy?
draw_box(ax, 2.2, 4.6, 3.0, 0.65, PURPLE, "Need redundancy?", fontsize=12, bold=True)

# No branch -> RAID 0 (top right)
draw_arrow(ax, 3.7, 4.45, 5.8, 4.6, label="No", label_offset=(0, 0.12))
draw_box(ax, 7.5, 4.6, 2.6, 0.55, RED, "RAID 0", fontsize=11, bold=True)
ax.text(9.0, 4.6, "fastest, zero protection", ha="left", va="center",
        fontsize=9.5, color="#6b7280", style="italic", zorder=4)

# Yes branch -> How many disks?
draw_arrow(ax, 2.2, 4.28, 2.2, 3.55, label="Yes", label_offset=(-0.5, 0.0))
draw_box(ax, 2.2, 3.1, 3.0, 0.65, BLUE, "How many disks?", fontsize=11, bold=True)

# 2 disks -> RAID 1
draw_arrow(ax, 3.7, 3.3, 5.8, 3.5, label="2", label_offset=(0, 0.12))
draw_box(ax, 7.5, 3.5, 2.6, 0.55, GREEN, "RAID 1", fontsize=11, bold=True)
ax.text(9.0, 3.5, "mirror", ha="left", va="center",
        fontsize=9.5, color="#6b7280", style="italic", zorder=4)

# 3-5 SSD -> RAID 5
draw_arrow(ax, 3.7, 3.0, 5.8, 2.4, label="3-5 SSD", label_offset=(0, 0.12))
draw_box(ax, 7.5, 2.4, 2.6, 0.55, AMBER, "RAID 5", fontsize=11, bold=True)
ax.text(9.0, 2.4, "single parity", ha="left", va="center",
        fontsize=9.5, color="#6b7280", style="italic", zorder=4)

# 6+ HDD -> RAID 6
draw_arrow(ax, 3.7, 2.85, 5.8, 1.3, label="6+ HDD", label_offset=(0.2, 0.12))
draw_box(ax, 7.5, 1.3, 2.6, 0.55, PURPLE, "RAID 6", fontsize=11, bold=True)
ax.text(9.0, 1.3, "double parity", ha="left", va="center",
        fontsize=9.5, color="#6b7280", style="italic", zorder=4)

ax.set_title("RAID Level Decision Tree",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_raid_decision_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_raid_decision_en.png")
