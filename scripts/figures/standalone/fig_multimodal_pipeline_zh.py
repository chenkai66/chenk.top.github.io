#!/usr/bin/env python3
"""Multimodal LLM Pipeline — BLIP-2 (ZH)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

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

# ViT (冻结)
draw_box(ax, 1.5, 1.9, 2.0, 0.8, PURPLE, "ViT\n(冻结)", fontsize=12, bold=True)

draw_arrow(ax, 2.55, 1.9, 3.3, 1.9)

# N 个视觉 token
draw_box(ax, 4.3, 1.9, 1.8, 0.65, "#6b7280", "[N 个视觉\ntoken]", fontsize=10)

draw_arrow(ax, 5.25, 1.9, 5.95, 1.9)

# Q-Former
draw_box(ax, 7.0, 1.9, 2.0, 0.8, AMBER, "Q-Former", fontsize=12, bold=True)
ax.text(7.0, 1.3, "可学习查询", ha="center", va="top", fontsize=9.5,
        color=AMBER, style="italic")

# Cross-attention
cross_arrow = FancyArrowPatch((4.3, 2.45), (7.0, 2.55),
                               connectionstyle="arc3,rad=-0.3",
                               arrowstyle="-|>", color=AMBER, lw=1.5,
                               mutation_scale=14, linestyle="--", zorder=2)
ax.add_patch(cross_arrow)
ax.text(5.65, 2.95, "交叉注意力", ha="center", va="bottom",
        fontsize=9, color=AMBER, style="italic")

draw_arrow(ax, 8.05, 1.9, 8.75, 1.9)

# K 个查询 token
draw_box(ax, 9.7, 1.9, 1.8, 0.65, "#6b7280", "[K 个查询\ntoken]", fontsize=10)

draw_arrow(ax, 10.65, 1.9, 11.2, 1.9)

# LLM (冻结)
draw_box(ax, 12.0, 1.9, 1.5, 0.8, GREEN, "LLM\n(冻结)", fontsize=12, bold=True)

# Frozen indicators
for cx in [1.5, 12.0]:
    ax.text(cx, 2.55, "* 冻结", ha="center", va="bottom", fontsize=8,
            color="#9ca3af")

ax.set_title("视觉-语言模型管线 (BLIP-2)",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_multimodal_pipeline_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_multimodal_pipeline_zh.png")
