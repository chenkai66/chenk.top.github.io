#!/usr/bin/env python3
"""记忆类型分类决策 (ZH)."""

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

def draw_arrow(ax, x1, y1, x2, y2, rad=0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=f"arc3,rad={rad}",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)

# Root node
draw_box(ax, 2.0, 2.5, 2.6, 0.8, PURPLE, "用户说了一句话", fontsize=12, bold=True)

# Decision questions and memory type targets
branches = [
    (4.2, "关于用户自己？",     8.5, 4.2, GREEN,  "user",            "用户偏好 / 身份"),
    (3.5, "纠正 Agent 行为？", 8.5, 3.5, AMBER,  "feedback / lesson", "行为修正 / 教训"),
    (2.5, "项目当前状态？",     8.5, 2.5, BLUE,   "project",          "项目进度 / 状态"),
    (1.6, "事实性信息？",       8.5, 1.6, PURPLE, "reference",        "URL / 配置 / 凭证"),
    (0.8, "都不是",             8.5, 0.8, RED,    "不存储",            "忽略"),
]

for y_q, question, bx, by, color, mem_type, desc in branches:
    # Question text
    ax.text(5.0, y_q, question, ha="center", va="center",
            fontsize=10.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f3f4f6", ec="#d1d5db", lw=0.8))
    # Arrow from question to memory box
    draw_arrow(ax, 6.2, y_q, 7.45, by)
    # Memory type box
    draw_box(ax, bx, by, 2.0, 0.55, color, mem_type, fontsize=10.5, bold=True)
    # Description
    ax.text(10.6, by, desc, ha="left", va="center",
            fontsize=9.5, color="#6b7280")

# Arrows from root to each question
for y_q, *_ in branches:
    draw_arrow(ax, 3.35, 2.5, 3.85, y_q, rad=0)

ax.set_title("记忆类型分类决策",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_memory_decision_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_memory_decision_zh.png")
