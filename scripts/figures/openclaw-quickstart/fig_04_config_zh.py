#!/usr/bin/env python3
"""OpenClaw Config Hierarchy (ZH)."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withSimplePatchShadow

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def draw_box(ax, cx, cy, w, h, color, label, fontsize=11, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.5,rounding_size=1.5",
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


# Title
ax.text(50, 96, "openclaw.json — 配置结构层级",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#1f2937")

# Root box
draw_box(ax, 50, 80, 18, 8, PURPLE, "openclaw.json", fontsize=12, bold=True)

# Level 1 branches
branches = [
    (8,  "agent",     BLUE),
    (22, "providers", AMBER),
    (36, "tools",     GREEN),
    (50, "memory",    PURPLE),
    (64, "security",  RED),
    (78, "channels",  BLUE),
    (92, "cron",      GREEN),
]

sub_items = {
    "agent":     "name, model",
    "providers": "dashscope, openai",
    "tools":     "exec, web_search",
    "memory":    "flush, search",
    "security":  "denied_paths",
    "channels":  "telegram, dingtalk",
    "cron":      "schedule, skill",
}

for bx, name, color in branches:
    draw_box(ax, bx, 55, 13, 7, color, name, fontsize=10.5, bold=True)
    draw_arrow(ax, 50, 76, bx, 59)
    draw_box(ax, bx, 35, 13, 7, color + "99", sub_items[name], fontsize=9.5)
    draw_arrow(ax, bx, 51.5, bx, 39)

ax.text(50, 18, "每个节对应 JSON 文件中的一个顶层键",
        ha="center", va="center", fontsize=9.5, color="#6b7280", style="italic")

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_04_config_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_04_config_zh.png")
