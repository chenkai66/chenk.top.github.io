"""国内即时通讯渠道选型 (ZH)"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withSimplePatchShadow

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]

BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
TITLE_COLOR = "#1f2937"
SHADOW = withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)
BOX_KW = dict(boxstyle="round,pad=0.5,rounding_size=1.5", ec="none")

def make_box(ax, x, y, w, h, color, label, fontsize=12, text_color="white"):
    box = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h, **BOX_KW, facecolor=color)
    box.set_path_effects([SHADOW])
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(**ARROW_KW))

def note_text(ax, x, y, text):
    ax.text(x, y, text, fontsize=9.5, color="#6b7280", style="italic", ha="center", va="top")

def difficulty(ax, x, y, level):
    labels = {"Easy": "简单", "Medium": "中等", "Hard": "困难"}
    colors = {"Easy": GREEN, "Medium": AMBER, "Hard": RED}
    c = colors.get(level, "#9ca3af")
    badge = mpatches.FancyBboxPatch((x - 4, y - 2.5), 8, 5,
                                     boxstyle="round,pad=0.3,rounding_size=0.8",
                                     facecolor=c, ec="none", alpha=0.85)
    ax.add_patch(badge)
    ax.text(x, y, labels.get(level, level), ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold")

fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 97, "国内即时通讯渠道选型",
        ha="center", va="top", fontsize=14, fontweight="bold", color=TITLE_COLOR)

# Start
make_box(ax, 13, 55, 22, 10, PURPLE, "用户在哪里\n聊天？", fontsize=11)

# DingTalk (top, y=85)
arrow(ax, 24, 60, 42, 85)
make_box(ax, 47, 85, 14, 8, "#6b7280", "钉钉", fontsize=11)
arrow(ax, 54, 85, 68, 85)
make_box(ax, 76, 85, 22, 8, GREEN, "Stream 模式\n（推荐）", fontsize=10)
note_text(ax, 76, 77, "无需公网 IP")
difficulty(ax, 92, 85, "Easy")

# WeCom (middle, y=55)
arrow(ax, 24, 55, 38, 55)
make_box(ax, 47, 55, 14, 8, "#6b7280", "企业微信", fontsize=11)

arrow(ax, 54, 58, 63, 63)
ax.text(58.5, 64, "内部\n员工", fontsize=9, color="#6b7280", ha="center", va="bottom")
make_box(ax, 76, 63, 20, 8, BLUE, "企微应用\n机器人", fontsize=10)
difficulty(ax, 91, 63, "Medium")

arrow(ax, 54, 52, 63, 45)
ax.text(58.5, 42, "外部\n客户", fontsize=9, color="#6b7280", ha="center", va="top")
make_box(ax, 76, 45, 20, 8, AMBER, "企微客服\n号", fontsize=10)
difficulty(ax, 91, 45, "Medium")

# WeChat personal (bottom, y=22)
arrow(ax, 24, 50, 42, 22)
make_box(ax, 47, 22, 18, 8, "#6b7280", "微信个人\n号", fontsize=11)
arrow(ax, 56, 22, 64, 22)
make_box(ax, 76, 22, 22, 8, AMBER, "WorkBuddy\n桌面桥接", fontsize=10)
note_text(ax, 76, 14, "需要常驻桌面")
difficulty(ax, 92, 22, "Hard")

plt.savefig("fig_09_picker_zh.png", dpi=160, bbox_inches="tight",
            facecolor=BG, pad_inches=0.1)
plt.close()
print("saved fig_09_picker_zh.png")
