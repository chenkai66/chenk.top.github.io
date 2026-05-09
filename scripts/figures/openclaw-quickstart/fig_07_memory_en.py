#!/usr/bin/env python3
"""Memory 4-Tier Architecture (EN)."""

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

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def draw_box(ax, cx, cy, w, h, color, label, sub_label="", fontsize=12, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        facecolor=color, ec="none", zorder=3,
    )
    box.set_path_effects([withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if sub_label:
        ax.text(cx, cy + 3.5, label, ha="center", va="center", fontsize=fontsize,
                color="white", fontweight=weight, zorder=4)
        ax.text(cx, cy - 3.5, sub_label, ha="center", va="center", fontsize=9.5,
                color="white", alpha=0.85, zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
                color="white", fontweight=weight, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, rad=0, label="", label_offset=(8, 0)):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=f"arc3,rad={rad}",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="left", va="center",
                fontsize=9.5, color="#6b7280", style="italic")


# Title
ax.text(50, 96, "OpenClaw Memory — Four-Tier Model",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#1f2937")

# Layer 1 (top): Context Window
draw_box(ax, 42, 80, 50, 11, GREEN,
         "Context Window",
         "Current conversation tokens · ephemeral",
         fontsize=12, bold=True)

# Layer 2: MEMORY.md Index
draw_box(ax, 42, 60, 50, 11, BLUE,
         "MEMORY.md Index",
         "40-line pointers · loaded every turn",
         fontsize=12, bold=True)

# Layer 3: memory/ Files
draw_box(ax, 42, 40, 50, 11, AMBER,
         "memory/ Files",
         "Detailed entries · loaded on relevance",
         fontsize=12, bold=True)

# Layer 4 (bottom): Semantic Search
draw_box(ax, 42, 20, 50, 11, PURPLE,
         "Semantic Search (bge-m3)",
         "Vector similarity · used when index fails",
         fontsize=12, bold=True)

# Arrows between layers with labels
draw_arrow(ax, 42, 74.5, 42, 66, label="compaction triggers memoryFlush",
           label_offset=(10, 0))
draw_arrow(ax, 42, 54.5, 42, 46, label="index points to files",
           label_offset=(10, 0))
draw_arrow(ax, 42, 34.5, 42, 26, label="fallback search",
           label_offset=(10, 0))

# Tier numbers on the left
for i, (y, tier) in enumerate([(80, "T1"), (60, "T2"), (40, "T3"), (20, "T4")]):
    ax.text(13, y, tier, ha="center", va="center",
            fontsize=10, color="#9ca3af", fontweight="bold")

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_07_memory_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_07_memory_en.png")
