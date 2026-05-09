#!/usr/bin/env python3
"""Tree-of-Thought Decision Tree (ZH) — 24 点游戏探索。"""

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

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
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
    arrow = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)

def draw_note(ax, x, y, text, fontsize=9.5, color="#6b7280"):
    ax.text(x, y, text, ha="left", va="center", fontsize=fontsize,
            color=color, style="italic", zorder=4)

# Root
draw_box(ax, 2.0, 5.0, 2.8, 0.7, PURPLE, "{4, 9, 10, 13}", fontsize=12, bold=True)
ax.text(2.0, 5.55, "根", ha="center", va="bottom", fontsize=10, color="#6b7280", fontweight="bold")

# Branch A (kept)
draw_arrow(ax, 3.4, 5.0, 5.3, 4.6)
draw_box(ax, 6.5, 4.6, 2.0, 0.6, BLUE, "{4, 4, 10}", fontsize=11)
draw_note(ax, 4.0, 4.95, "13 - 9 = 4", fontsize=9.5, color="#374151")
draw_note(ax, 7.65, 4.6, "v=8  (保留)", fontsize=9.5, color=BLUE)

# Branch B (explored)
draw_arrow(ax, 3.4, 4.85, 5.3, 3.4)
draw_box(ax, 6.5, 3.4, 2.0, 0.6, AMBER, "{6, 9, 13}", fontsize=11)
draw_note(ax, 4.0, 3.85, "10 - 4 = 6", fontsize=9.5, color="#374151")
draw_note(ax, 7.65, 3.4, "v=5  (待探索)", fontsize=9.5, color=AMBER)

# Branch C (pruned)
draw_arrow(ax, 3.4, 4.7, 5.3, 2.2)
draw_box(ax, 6.5, 2.2, 2.4, 0.6, RED, "{10, 13, 13}", fontsize=11)
draw_note(ax, 4.0, 2.65, "9 + 4 = 13", fontsize=9.5, color="#374151")
draw_note(ax, 7.85, 2.2, "v=3  (剪枝)", fontsize=9.5, color=RED)
ax.plot([5.35, 7.65], [2.2, 2.2], color=RED, lw=1.5, alpha=0.35, zorder=5)

# Level 2
draw_arrow(ax, 7.5, 4.35, 8.8, 4.6)
draw_box(ax, 9.7, 4.6, 1.5, 0.6, BLUE, "{4, 6}", fontsize=11)
draw_note(ax, 8.1, 4.25, "10 - 4 = 6", fontsize=9.5, color="#374151")
draw_note(ax, 10.55, 4.6, "v=9", fontsize=9.5, color=BLUE)

# Solution
draw_arrow(ax, 10.45, 4.35, 11.0, 3.8)
draw_box(ax, 11.7, 3.6, 1.3, 0.65, GREEN, "24", fontsize=14, bold=True)
draw_note(ax, 10.5, 4.0, "6 * 4", fontsize=9.5, color="#374151")
draw_note(ax, 11.7, 3.1, "解出", fontsize=10, color=GREEN)

ax.set_title("思维树 — 探索 24 点游戏",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_tot_tree_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_tot_tree_zh.png")
