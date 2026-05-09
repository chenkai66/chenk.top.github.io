#!/usr/bin/env python3
"""Map-Reduce Document Synthesis (ZH) — 映射-归约式文档综述。"""

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

def draw_arrow(ax, x1, y1, x2, y2, rad=0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=f"arc3,rad={rad}",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)

# Column 1: Documents
doc_y = [2.6, 1.75, 0.9]
for i, y in enumerate(doc_y):
    draw_box(ax, 1.3, y, 1.8, 0.55, PURPLE, f"文档 {i+1}", fontsize=10.5)

# Column 2: Summaries
for i, y in enumerate(doc_y):
    draw_arrow(ax, 2.25, y, 3.55, y)
    draw_box(ax, 4.6, y, 2.2, 0.55, BLUE, f"单文档摘要 {i+1}", fontsize=10.5)

# Phase labels
ax.text(1.3, 3.2, "映射", ha="center", va="bottom", fontsize=10,
        color=PURPLE, fontweight="bold")
ax.text(4.6, 3.2, "逐文档 LLM 调用", ha="center", va="bottom",
        fontsize=9, color="#9ca3af")

# Converge arrows
for y in doc_y:
    draw_arrow(ax, 5.75, y, 7.3, 1.75, rad=0)

# Merge
draw_box(ax, 8.2, 1.75, 1.8, 0.7, AMBER, "主题\n合并", fontsize=11, bold=True)
ax.text(8.2, 3.2, "归约", ha="center", va="bottom", fontsize=10,
        color=AMBER, fontweight="bold")

# Arrow to final
draw_arrow(ax, 9.15, 1.75, 10.15, 1.75)

# Final synthesis
draw_box(ax, 11.3, 1.75, 2.0, 0.7, GREEN, "最终\n综述", fontsize=11, bold=True)

ax.set_title("映射-归约式文档综述",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_map_reduce_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_map_reduce_zh.png")
