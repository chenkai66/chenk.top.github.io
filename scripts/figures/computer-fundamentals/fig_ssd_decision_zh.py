#!/usr/bin/env python3
"""存储介质选择指南 (ZH)."""

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

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

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


# Root
draw_box(ax, 2.2, 4.6, 3.6, 0.65, PURPLE, "单盘需要 > 4 TB？", fontsize=12, bold=True)

# Yes -> HDD
draw_arrow(ax, 4.0, 4.45, 5.8, 4.6, label="是", label_offset=(0, 0.12))
draw_box(ax, 7.5, 4.6, 2.6, 0.55, RED, "HDD", fontsize=12, bold=True)
ax.text(9.0, 4.6, "成本优先", ha="left", va="center",
        fontsize=9.5, color="#6b7280", style="italic", zorder=4)

# No -> SSD
draw_arrow(ax, 2.2, 4.28, 2.2, 3.55, label="否", label_offset=(-0.4, 0.0))
draw_box(ax, 2.2, 3.1, 2.0, 0.65, BLUE, "SSD", fontsize=12, bold=True)

# Old motherboard -> SATA SSD
draw_arrow(ax, 3.2, 3.3, 5.8, 3.5, label="旧主板，无 M.2", label_offset=(0, 0.12))
draw_box(ax, 7.5, 3.5, 2.6, 0.55, AMBER, "SATA SSD", fontsize=11, bold=True)

# Boot/games -> NVMe Gen 3
draw_arrow(ax, 3.2, 3.0, 5.8, 2.4, label="系统盘 / 游戏", label_offset=(0, 0.12))
draw_box(ax, 7.5, 2.4, 2.6, 0.55, GREEN, "NVMe Gen 3", fontsize=11, bold=True)
ax.text(9.0, 2.4, "性价比之选", ha="left", va="center",
        fontsize=9.5, color=GREEN, fontweight="bold", zorder=4)

# Data centre / AI -> NVMe Gen 5
draw_arrow(ax, 3.2, 2.85, 5.8, 1.3, label="数据中心 / AI", label_offset=(0.2, 0.12))
draw_box(ax, 7.5, 1.3, 2.6, 0.55, PURPLE, "NVMe Gen 5", fontsize=11, bold=True)
ax.text(9.0, 1.3, "最大带宽", ha="left", va="center",
        fontsize=9.5, color="#6b7280", style="italic", zorder=4)

ax.set_title("存储介质选择指南",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_ssd_decision_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_ssd_decision_zh.png")
