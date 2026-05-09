#!/usr/bin/env python3
"""弹性伸缩生命周期 — 水平时间线 (ZH)."""
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withSimplePatchShadow

BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
ARROW_C = "#9ca3af"
SHADOW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
BOX_KW = dict(boxstyle="round,pad=0.5,rounding_size=1.5", ec="none")

fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.5)
ax.axis("off")
ax.set_title("弹性伸缩生命周期", fontsize=14, fontweight="bold",
             pad=12, color="#1e293b")

# --- Timeline backbone ---
ax.plot([1, 13], [2.8, 2.8], color="#cbd5e1", lw=3, zorder=0)
ax.text(4.2, 4.9, "扩容阶段 (Scale Out)", fontsize=12, fontweight="bold",
        color=RED, ha="center")
ax.text(10.5, 4.9, "缩容阶段 (Scale In)", fontsize=12, fontweight="bold",
        color=GREEN, ha="center")
ax.plot([7.3, 7.3], [0.3, 4.7], color="#e2e8f0", lw=1.5, ls="--")

# --- 扩容 boxes ---
so_boxes = [
    (0.5, 3.4, 3.0, 0.9, RED,    "指标触发\nCPU > 70% 持续 3 分钟"),
    (4.0, 3.4, 3.0, 0.9, AMBER,  "伸缩组添加\nN 台实例"),
    (0.5, 1.2, 3.0, 0.9, BLUE,   "负载均衡健康检查\n开始路由流量"),
]
for bx_data in so_boxes:
    x, y, w, h, color, label = bx_data
    box = FancyBboxPatch((x, y), w, h, facecolor=color, **BOX_KW)
    box.set_path_effects([withSimplePatchShadow(**SHADOW)])
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white", linespacing=1.3)

ax.annotate("", xy=(4.0 - 0.08, 3.85), xytext=(0.5 + 3.0 + 0.08, 3.85),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2, mutation_scale=16))
ax.annotate("", xy=(2.0, 1.2 + 0.9 + 0.08), xytext=(5.5, 3.4 - 0.08),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2, mutation_scale=16,
                            connectionstyle="arc3,rad=0.3"))

# --- 缩容 boxes ---
si_boxes = [
    (7.8, 3.4, 3.2, 0.9, GREEN,  "指标恢复\nCPU < 30% 持续 10 分钟"),
    (11.5, 3.4, 2.2, 0.9, PURPLE, "伸缩组移除\n多余实例"),
    (7.8, 1.2, 3.2, 0.9, BLUE,   "优先排空连接\n再释放实例"),
]
for bx_data in si_boxes:
    x, y, w, h, color, label = bx_data
    box = FancyBboxPatch((x, y), w, h, facecolor=color, **BOX_KW)
    box.set_path_effects([withSimplePatchShadow(**SHADOW)])
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white", linespacing=1.3)

ax.annotate("", xy=(11.5 - 0.08, 3.85), xytext=(7.8 + 3.2 + 0.08, 3.85),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2, mutation_scale=16))
ax.annotate("", xy=(9.4, 1.2 + 0.9 + 0.08), xytext=(12.6, 3.4 - 0.08),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2, mutation_scale=16,
                            connectionstyle="arc3,rad=0.3"))

ax.text(2.0, 0.3, "阈值: CPU > 70%", fontsize=9.5, color="#64748b",
        ha="center", style="italic")
ax.text(9.4, 0.3, "阈值: CPU < 30%", fontsize=9.5, color="#64748b",
        ha="center", style="italic")

fig.savefig("fig_autoscaling_zh.png", dpi=160, bbox_inches="tight",
            facecolor=BG, pad_inches=0.1)
plt.close()
print("saved fig_autoscaling_zh.png")
