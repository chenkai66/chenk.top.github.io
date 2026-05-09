#!/usr/bin/env python3
"""负载均衡算法对比 (ZH)."""

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withSimplePatchShadow

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
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

algorithms = [
    ("轮询",         GREEN,  "A, B, C, A, B, C, ...",          "依次分配，最简单"),
    ("加权轮询",     BLUE,   "A(3), B(1) -> A, A, A, B, ...", "按权重比例分配"),
    ("最少连接",     PURPLE, "挑未完成请求数最少的后端",        "适合长连接场景"),
    ("最短响应时间", AMBER,  "挑延迟 EWMA 最低的后端",         "感知后端性能"),
    ("两选一 (P2C)", RED,    "随机挑 2 后端，选负载轻的",       "低开销 + 均衡"),
]

n = len(algorithms)
row_h = 0.72
gap = 0.18
total_h = n * row_h + (n - 1) * gap
y_start = (5.0 + total_h) / 2 - row_h / 2  # vertically center

for i, (name, color, pattern, desc) in enumerate(algorithms):
    cy = y_start - i * (row_h + gap)

    # Algorithm name box
    draw_box(ax, 1.8, cy, 2.8, row_h, color, name, fontsize=11, bold=True)

    # Pattern illustration
    ax.text(5.8, cy, pattern, ha="left", va="center", fontsize=10,
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f9fafb", ec="#e5e7eb", lw=0.8))

    # Description
    ax.text(11.5, cy, desc, ha="left", va="center",
            fontsize=9.5, color="#6b7280")

ax.set_title("负载均衡算法对比",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_lb_algorithms_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_lb_algorithms_zh.png")
