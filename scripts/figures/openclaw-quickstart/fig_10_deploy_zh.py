"""生产环境部署架构 (ZH)"""
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

def arrow_down(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(**ARROW_KW))

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 97, "生产环境部署架构",
        ha="center", va="top", fontsize=14, fontweight="bold", color=TITLE_COLOR)

cx = 50

# Layer 1: Internet
make_box(ax, cx, 85, 18, 7, "#6b7280", "互联网", fontsize=12)
arrow_down(ax, cx, 81.5, 76)

# Layer 2: nginx
make_box(ax, cx, 72, 32, 7, BLUE, "nginx（反向代理 + TLS）", fontsize=11)
arrow_down(ax, cx, 68.5, 63)

# Layer 3: pm2
make_box(ax, cx, 59, 28, 7, AMBER, "pm2（进程管理器）", fontsize=11)
arrow_down(ax, cx, 55.5, 50)

# Layer 4: OpenClaw Gateway
gw_y, gw_h = 38, 18
gw_box = mpatches.FancyBboxPatch((cx - 22, gw_y - gw_h/2), 44, gw_h,
                                  **BOX_KW, facecolor=PURPLE, alpha=0.15)
gw_box.set_path_effects([SHADOW])
ax.add_patch(gw_box)

gw_border = mpatches.FancyBboxPatch((cx - 22, gw_y - gw_h/2), 44, gw_h,
                                     **BOX_KW, facecolor="none", edgecolor=PURPLE, lw=2)
ax.add_patch(gw_border)

ax.text(cx, gw_y + 7, "OpenClaw 网关", ha="center", va="center",
        fontsize=13, fontweight="bold", color=PURPLE)

sub_w, sub_h = 12, 6
sub_y = gw_y - 3
make_box(ax, cx - 14, sub_y, sub_w, sub_h, BLUE, "渠道层", fontsize=10)
make_box(ax, cx, sub_y, sub_w, sub_h, GREEN, "Agent\n循环", fontsize=10)
make_box(ax, cx + 14, sub_y, sub_w, sub_h, PURPLE, "记忆层", fontsize=10)

arrow_down(ax, cx, gw_y - gw_h/2, 20)

# Layer 5: LLM
make_box(ax, cx, 16, 30, 7, "#6b7280", "LLM 服务（DashScope）", fontsize=11)

# acme.sh annotation
acme_x = 85
acme_y = 72
ax.text(acme_x, acme_y, "acme.sh", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#059669",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#d1fae5", edgecolor="#059669", lw=1.5))

ax.annotate("", xy=(cx + 16, 72), xytext=(acme_x - 6, acme_y),
            arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.5,
                           linestyle="dashed", mutation_scale=14))
ax.text(76, 76, "自动续期\n证书", fontsize=9.5, color="#059669",
        ha="center", va="bottom", style="italic")

plt.savefig("fig_10_deploy_zh.png", dpi=160, bbox_inches="tight",
            facecolor=BG, pad_inches=0.1)
plt.close()
print("saved fig_10_deploy_zh.png")
