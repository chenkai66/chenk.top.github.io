#!/usr/bin/env python3
"""多模态大模型迭代优化环 (ZH)."""

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

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.5)
ax.axis("off")


def box(cx, cy, w, h, color, label, fs=10.5, bold=False):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    b.set_path_effects([pe.withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(b)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold" if bold else "normal",
            zorder=4, linespacing=1.3)


def arrow(x1, y1, x2, y2, rad=0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}", **ARROW_KW, zorder=2,
    )
    ax.add_patch(a)


colors = [PURPLE, BLUE, GREEN, RED, AMBER, BLUE, PURPLE]
labels = [
    "选择\n基座模型",
    "跑\n基线",
    "构建\n私有评测",
    "识别 Top-3\n失败模式",
    "设计\n修复方案",
    "添加\n针对性数据",
    "选择更优\n架构",
]

positions = [
    (1.8, 3.3),
    (4.6, 3.3),
    (7.4, 3.3),
    (10.6, 3.3),
    (10.6, 1.3),
    (6.5, 1.3),
    (2.5, 1.3),
]

bw, bh = 2.2, 0.85

for i, (cx, cy) in enumerate(positions):
    box(cx, cy, bw, bh, colors[i], labels[i], fs=10.5, bold=(i == 0))

for i in range(3):
    x1 = positions[i][0] + bw / 2
    x2 = positions[i + 1][0] - bw / 2
    y = positions[i][1]
    arrow(x1, y, x2, y)

arrow(positions[3][0], positions[3][1] - bh / 2,
      positions[4][0], positions[4][1] + bh / 2)

arrow(positions[4][0] - bw / 2, positions[4][1],
      positions[5][0] + bw / 2, positions[5][1])
arrow(positions[5][0] - bw / 2, positions[5][1],
      positions[6][0] + bw / 2, positions[6][1])

arrow(positions[6][0], positions[6][1] + bh / 2,
      positions[0][0], positions[0][1] - bh / 2)

for i, (cx, cy) in enumerate(positions):
    ax.text(cx - bw / 2 + 0.15, cy + bh / 2 - 0.1, str(i + 1),
            fontsize=8, color="white", fontweight="bold", zorder=5, alpha=0.7)

ax.set_title("多模态大模型迭代优化环",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_mllm_iteration_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_mllm_iteration_zh.png")
