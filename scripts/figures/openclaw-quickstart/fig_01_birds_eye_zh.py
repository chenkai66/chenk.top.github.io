"""
fig_01_birds_eye_zh.py
Alt: OpenClaw 架构鸟瞰：聊天平台 - Channel Adapter - Agent Runtime - LLM
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withSimplePatchShadow

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]

# --- Design tokens ---
BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
ARROW_COLOR = "#9ca3af"
SHADOW_KWARGS = dict(offset=(1.2, -1.2), shadow_rgbFace="#d1d5db", alpha=0.35)
BOXSTYLE = "round,pad=0.15,rounding_size=0.15"
BOXSTYLE_SM = "round,pad=0.1,rounding_size=0.1"

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

# Title
ax.text(6, 6.55, "OpenClaw 架构鸟瞰",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1f2937")


def make_box(ax, x, y, w, h, label, color, fontsize=12, sublabel=None, sublabel_fs=9, style=BOXSTYLE):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        facecolor="white", edgecolor=color, linewidth=2, zorder=5,
    )
    box.set_boxstyle(style)
    box.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
    ax.add_patch(box)
    cx = x + w / 2
    cy = y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.25, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=6)
        ax.text(cx, cy - 0.25, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color="#6b7280", zorder=6)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=6)
    return x + w, cy


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2, mutation_scale=18),
                zorder=3)


# --- Block 1: 聊天平台 ---
b1_x, b1_y, b1_w, b1_h = 0.3, 2.0, 2.0, 2.6
box1 = mpatches.FancyBboxPatch(
    (b1_x, b1_y), b1_w, b1_h,
    boxstyle=BOXSTYLE,
    facecolor="white", edgecolor=BLUE, linewidth=2, zorder=5,
)
box1.set_boxstyle(BOXSTYLE)
box1.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
ax.add_patch(box1)
b1_cx = b1_x + b1_w / 2
b1_cy = b1_y + b1_h / 2
ax.text(b1_cx, b1_cy + 0.7, "聊天平台", ha="center", va="center",
        fontsize=13, fontweight="bold", color=BLUE, zorder=6)
for i, name in enumerate(["Telegram", "钉钉", "微信"]):
    ax.text(b1_cx, b1_cy + 0.15 - i * 0.4, name, ha="center", va="center",
            fontsize=10, color="#6b7280", zorder=6)

# --- Block 2: Channel Adapter ---
b2_x, b2_y, b2_w, b2_h = 3.0, 2.5, 1.8, 1.6
make_box(ax, b2_x, b2_y, b2_w, b2_h, "Channel\nAdapter", AMBER, fontsize=12)
b2_cx = b2_x + b2_w / 2
b2_cy = b2_y + b2_h / 2

# --- Block 3: Agent Runtime (large, with sub-components) ---
b3_x, b3_y, b3_w, b3_h = 5.5, 1.5, 3.2, 3.6
box3 = mpatches.FancyBboxPatch(
    (b3_x, b3_y), b3_w, b3_h,
    boxstyle="round,pad=0.2,rounding_size=0.2",
    facecolor="#f9fafb", edgecolor=RED, linewidth=2.5, zorder=3,
)
box3.set_boxstyle("round,pad=0.2,rounding_size=0.2")
box3.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
ax.add_patch(box3)
b3_cx = b3_x + b3_w / 2
ax.text(b3_cx, b3_y + b3_h - 0.35, "Agent Runtime", ha="center", va="center",
        fontsize=14, fontweight="bold", color=RED, zorder=6)

# Sub-components
sub_w, sub_h = 2.2, 0.6
sub_x = b3_x + (b3_w - sub_w) / 2
make_box(ax, sub_x, 3.8, sub_w, sub_h, "工具", PURPLE, fontsize=11, style=BOXSTYLE_SM)
make_box(ax, sub_x, 3.0, sub_w, sub_h, "技能", GREEN, fontsize=11, style=BOXSTYLE_SM)
make_box(ax, sub_x, 2.2, sub_w, sub_h, "记忆", BLUE, fontsize=11, style=BOXSTYLE_SM)

b3_cy = b3_y + b3_h / 2

# --- Block 4: LLM ---
b4_x, b4_y, b4_w, b4_h = 9.4, 2.0, 2.2, 2.6
box4 = mpatches.FancyBboxPatch(
    (b4_x, b4_y), b4_w, b4_h,
    boxstyle=BOXSTYLE,
    facecolor="white", edgecolor=PURPLE, linewidth=2, zorder=5,
)
box4.set_boxstyle(BOXSTYLE)
box4.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
ax.add_patch(box4)
b4_cx = b4_x + b4_w / 2
b4_cy = b4_y + b4_h / 2
ax.text(b4_cx, b4_cy + 0.7, "LLM", ha="center", va="center",
        fontsize=14, fontweight="bold", color=PURPLE, zorder=6)
for i, name in enumerate(["通义千问", "Claude", "GPT"]):
    ax.text(b4_cx, b4_cy + 0.15 - i * 0.4, name, ha="center", va="center",
            fontsize=10, color="#6b7280", zorder=6)

# --- Arrows ---
arrow(ax, b1_x + b1_w, b1_cy, b2_x, b2_cy)
arrow(ax, b2_x + b2_w, b2_cy, b3_x, b3_cy)
arrow(ax, b3_x + b3_w, b3_cy, b4_x, b4_cy)

fig.savefig("/Users/kchen/Desktop/Project/skilltest/fig_01_birds_eye_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Done: fig_01_birds_eye_zh.png")
