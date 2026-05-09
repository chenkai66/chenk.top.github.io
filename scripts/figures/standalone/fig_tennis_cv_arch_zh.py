#!/usr/bin/env python3
"""Tennis CV System — 4-Layer Architecture (ZH)."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
ax.axis("off")

layers = [
    (4.8, PURPLE, "应用层", ["可视化", "数据分析", "战术报表", "裁判辅助"]),
    (3.6, BLUE,   "业务层", ["事件检测", "战术分析", "历史对比", "实时推送"]),
    (2.4, AMBER,  "算法层", ["检测+追踪", "姿态", "球追踪"]),
    (1.2, RED,    "硬件层", ["摄像头", "GPU", "边缘设备", "网络"]),
]

row_h = 0.85
label_w = 2.2
content_x0 = 2.8
content_w = 9.6

for y, color, label, components in layers:
    lbox = FancyBboxPatch(
        (0.3, y - row_h / 2), label_w, row_h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    lbox.set_path_effects([pe.withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(lbox)
    ax.text(0.3 + label_w / 2, y, label, ha="center", va="center",
            fontsize=12, color="white", fontweight="bold", zorder=4)

    n = len(components)
    gap = 0.15
    box_w = (content_w - gap * (n - 1)) / n
    for i, comp in enumerate(components):
        bx = content_x0 + i * (box_w + gap)
        cbox = FancyBboxPatch(
            (bx, y - row_h / 2), box_w, row_h,
            boxstyle="round,pad=0.15,rounding_size=0.25",
            facecolor=color, ec="none", alpha=0.55, zorder=3,
        )
        cbox.set_path_effects([pe.withSimplePatchShadow(**SHADOW_KW)])
        ax.add_patch(cbox)
        ax.text(bx + box_w / 2, y, comp, ha="center", va="center",
                fontsize=10.5, color="#1f2937", fontweight="medium", zorder=4)

ax.set_title("网球 CV 系统 — 四层架构",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_tennis_cv_arch_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_tennis_cv_arch_zh.png")
