#!/usr/bin/env python3
"""MCP 服务器通信流程 (ZH)."""

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
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

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
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

def draw_arrow(ax, x1, y1, x2, y2, rad=0, label="", label_offset=(0, 0.15)):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=f"arc3,rad={rad}",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9.5, color="#6b7280", style="italic")

# Boxes
draw_box(ax, 2.2, 3.5, 2.6, 0.9, PURPLE, "Claude Code", fontsize=12, bold=True)
draw_box(ax, 6.0, 3.5, 2.8, 0.9, BLUE, "MCP 进程", fontsize=12)
draw_box(ax, 9.8, 3.5, 2.2, 0.9, AMBER, "真实浏览器", fontsize=12, bold=True)
draw_box(ax, 2.2, 1.2, 2.4, 0.9, GREEN, "工具结果", fontsize=12, bold=True)
draw_box(ax, 9.8, 1.2, 2.4, 0.9, RED, "页面快照", fontsize=12)

# Arrows
draw_arrow(ax, 3.55, 3.5, 4.55, 3.5, label="stdio", label_offset=(0, 0.22))
draw_arrow(ax, 7.45, 3.5, 8.65, 3.5, label="指令", label_offset=(0, 0.22))
draw_arrow(ax, 9.8, 3.0, 9.8, 1.7)
draw_arrow(ax, 8.55, 1.2, 3.45, 1.2, label="返回给 Agent", label_offset=(0, 0.22))
draw_arrow(ax, 2.2, 3.0, 2.2, 1.7)

ax.set_title("MCP 服务器通信流程",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_mcp_arch_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_mcp_arch_zh.png")
