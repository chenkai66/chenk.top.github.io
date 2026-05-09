#!/usr/bin/env python3
"""Multimodal LLM Pipeline — BLIP-2 (EN)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)

fig, ax = plt.subplots(figsize=(13, 3.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 3.5)
ax.axis("off")

def draw_box(ax, cx, cy, w, h, color, label, fontsize=11, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    box.set_path_effects([patheffects.withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle="arc3,rad=0",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)

# ViT (frozen)
draw_box(ax, 1.5, 1.9, 2.0, 0.8, PURPLE, "ViT\n(frozen)", fontsize=12, bold=True)

# Arrow: ViT -> token representation
draw_arrow(ax, 2.55, 1.9, 3.3, 1.9)

# N visual tokens
draw_box(ax, 4.3, 1.9, 1.8, 0.65, "#6b7280", "[N visual\ntokens]", fontsize=10)

# Arrow: tokens -> Q-Former
draw_arrow(ax, 5.25, 1.9, 5.95, 1.9)

# Q-Former (the learned bridge)
draw_box(ax, 7.0, 1.9, 2.0, 0.8, AMBER, "Q-Former", fontsize=12, bold=True)
# Annotation: learned queries
ax.text(7.0, 1.3, "Learned queries", ha="center", va="top", fontsize=9.5,
        color=AMBER, style="italic")
# Cross-attention annotation (curved arrow above Q-Former)
cross_arrow = FancyArrowPatch((4.3, 2.45), (7.0, 2.55),
                               connectionstyle="arc3,rad=-0.3",
                               arrowstyle="-|>", color=AMBER, lw=1.5,
                               mutation_scale=14, linestyle="--", zorder=2)
ax.add_patch(cross_arrow)
ax.text(5.65, 2.95, "Cross-attention", ha="center", va="bottom",
        fontsize=9, color=AMBER, style="italic")

# Arrow: Q-Former -> K query tokens
draw_arrow(ax, 8.05, 1.9, 8.75, 1.9)

# K query tokens
draw_box(ax, 9.7, 1.9, 1.8, 0.65, "#6b7280", "[K query\ntokens]", fontsize=10)

# Arrow: -> LLM
draw_arrow(ax, 10.65, 1.9, 11.2, 1.9)

# LLM (frozen)
draw_box(ax, 12.0, 1.9, 1.5, 0.8, GREEN, "LLM\n(frozen)", fontsize=12, bold=True)

# Frozen indicators (snowflake-like dots)
for cx in [1.5, 12.0]:
    ax.text(cx, 2.55, "* frozen", ha="center", va="bottom", fontsize=8,
            color="#9ca3af")

ax.set_title("Vision-Language Model Pipeline (BLIP-2)",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_multimodal_pipeline_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_multimodal_pipeline_en.png")
