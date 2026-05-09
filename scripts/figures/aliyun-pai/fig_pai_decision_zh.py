#!/usr/bin/env python3
"""PAI 产品决策树 (ZH)."""
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

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_title("PAI 产品决策树", fontsize=14, fontweight="bold",
             pad=12, color="#1e293b")


def draw_box(x, y, w, h, color, label, fontsize=11):
    box = FancyBboxPatch((x, y), w, h, facecolor=color, **BOX_KW)
    box.set_path_effects([withSimplePatchShadow(**SHADOW)])
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", linespacing=1.3)
    return (x, y, w, h)


def arrow_right(src, dst):
    x1 = src[0] + src[2] + 0.08
    x2 = dst[0] - 0.08
    y1 = src[1] + src[3] / 2
    y2 = dst[1] + dst[3] / 2
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2,
                                mutation_scale=16))


def arrow_down(src, dst):
    cx = src[0] + src[2] / 2
    y1 = src[1] - 0.05
    y2 = dst[1] + dst[3] + 0.05
    ax.annotate("", xy=(cx, y2), xytext=(cx, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2,
                                mutation_scale=16))


# Root
root = draw_box(0.3, 7.6, 3.8, 0.8, PURPLE, "你在构建什么?")

# --- Branch 1: 表格 ML ---
b1 = draw_box(0.3, 5.8, 3.8, 0.8, BLUE, "表格 ML\n回归 / 分类 / 聚类")
arrow_down(root, b1)

b1a = draw_box(5.0, 6.8, 3.5, 0.7, GREEN, "数据在 MaxCompute?\n→ Designer", 10)
b1b = draw_box(5.0, 5.8, 3.5, 0.7, AMBER, "数据在 OSS / 外部?\n→ DLC + 自定义代码", 10)
b1c = draw_box(5.0, 4.8, 3.5, 0.7, RED,   "快速试验?\n→ QuickStart AutoML", 10)
arrow_right(b1, b1a)
arrow_right(b1, b1b)
arrow_right(b1, b1c)

ax.text(4.6, 7.15, "是", fontsize=9, color=GREEN, fontweight="bold", ha="center")
ax.text(4.6, 6.15, "外部数据", fontsize=9, color=AMBER, fontweight="bold", ha="center")
ax.text(4.6, 5.15, "快速尝试", fontsize=9, color=RED, fontweight="bold", ha="center")

# --- Branch 2: 深度学习 ---
b2 = draw_box(0.3, 3.4, 3.8, 0.8, BLUE, "深度学习\nCV / NLP / 语音")
arrow_down(b1, b2)

b2a = draw_box(5.0, 3.4, 3.5, 0.7, RED, "始终 → DLC\n(GPU, 自定义代码)", 10)
arrow_right(b2, b2a)

# --- Branch 3: LLM ---
b3 = draw_box(0.3, 1.6, 3.8, 0.8, BLUE, "大模型微调\n/ 推理部署")
arrow_down(b2, b3)

b3a = draw_box(5.0, 1.6, 3.5, 0.7, AMBER, "始终 → PAI-EAS\n+ DLC", 10)
arrow_right(b3, b3a)

# legend
legend_x = 9.5
ax.text(legend_x, 7.8, "产品对应关系", fontsize=10.5, fontweight="bold", color="#1e293b")
for i, (c, lbl) in enumerate([
    (GREEN, "Designer (可视化拖拽建模)"),
    (AMBER, "DLC / EAS (代码驱动, GPU)"),
    (RED,   "QuickStart AutoML (零代码)"),
]):
    yy = 7.2 - i * 0.55
    box = FancyBboxPatch((legend_x, yy), 0.4, 0.35, facecolor=c,
                         boxstyle="round,pad=0.1,rounding_size=0.5", ec="none")
    ax.add_patch(box)
    ax.text(legend_x + 0.6, yy + 0.17, lbl, fontsize=9.5, va="center", color="#475569")

fig.savefig("fig_pai_decision_zh.png", dpi=160, bbox_inches="tight",
            facecolor=BG, pad_inches=0.1)
plt.close()
print("saved fig_pai_decision_zh.png")
