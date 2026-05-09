"""Heartbeat vs Cron — 两种调度原语 (ZH)"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withSimplePatchShadow

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]

# --- design tokens ---
BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
TITLE_COLOR = "#1f2937"
SHADOW = withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
BOX_KW = dict(boxstyle="round,pad=0.5,rounding_size=1.5", ec="none")

def make_box(ax, x, y, w, h, color, label, fontsize=12):
    box = mpatches.FancyBboxPatch((x, y), w, h, **BOX_KW, facecolor=color)
    box.set_path_effects([SHADOW])
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold")

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 96, "Heartbeat vs Cron — 两种调度原语",
        ha="center", va="top", fontsize=14, fontweight="bold", color=TITLE_COLOR)

# Dividing line
ax.plot([50, 50], [5, 88], color="#d1d5db", lw=1.5, ls="--", zorder=0)

# ============= LEFT — Heartbeat =============
col_left = 25
make_box(ax, col_left - 16, 74, 32, 10, GREEN, "Heartbeat（心跳巡逻）", fontsize=12)
ax.text(col_left, 69, "[定时器驱动]", ha="center", va="center",
        fontsize=9.5, color="#6b7280", style="italic")

items_left = [
    "按定时器运行（每 N 分钟）",
    "读 HEARTBEAT.md 获取指令",
    "检查环境后决定动作",
    "适合：巡检、监控、响应式",
]
y_start = 60
for i, txt in enumerate(items_left):
    y = y_start - i * 13
    box = mpatches.FancyBboxPatch((col_left - 18, y - 4), 36, 9, **BOX_KW,
                                   facecolor="#d1fae5", alpha=0.9)
    box.set_path_effects([SHADOW])
    ax.add_patch(box)
    ax.text(col_left, y + 0.5, txt, ha="center", va="center",
            fontsize=10.5, color="#065f46", fontweight="medium")

# ============= RIGHT — Cron =============
col_right = 75
make_box(ax, col_right - 14, 74, 28, 10, BLUE, "Cron（定时任务）", fontsize=12)
ax.text(col_right, 69, "[表达式调度]", ha="center", va="center",
        fontsize=9.5, color="#6b7280", style="italic")

items_right = [
    "按 cron 表达式调度",
    "触发指定的 Skill",
    "执行任务并推送结果",
    "适合：报告、简报、主动式",
]
for i, txt in enumerate(items_right):
    y = y_start - i * 13
    box = mpatches.FancyBboxPatch((col_right - 18, y - 4), 36, 9, **BOX_KW,
                                   facecolor="#dbeafe", alpha=0.9)
    box.set_path_effects([SHADOW])
    ax.add_patch(box)
    ax.text(col_right, y + 0.5, txt, ha="center", va="center",
            fontsize=10.5, color="#1e3a5f", fontweight="medium")

plt.savefig("fig_08_hb_vs_cron_zh.png", dpi=160, bbox_inches="tight",
            facecolor=BG, pad_inches=0.1)
plt.close()
print("saved fig_08_hb_vs_cron_zh.png")
