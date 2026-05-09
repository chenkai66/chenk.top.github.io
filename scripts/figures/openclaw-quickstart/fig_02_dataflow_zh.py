"""
fig_02_dataflow_zh.py
Alt: 运行时数据流：TUI - Gateway - LLM Provider，Gateway 同时调用本地系统
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

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

# Title
ax.text(6, 6.55, "运行时数据流",
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
        ax.text(cx, cy + 0.22, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=6)
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color="#6b7280", zorder=6)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=6)
    return cx, cy


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2, mutation_scale=18),
                zorder=3)


# --- TUI ---
tui_x, tui_y, tui_w, tui_h = 0.8, 3.3, 2.2, 1.4
tui_cx, tui_cy = make_box(ax, tui_x, tui_y, tui_w, tui_h,
                           "TUI", BLUE, fontsize=14, sublabel="(终端)")

# --- Gateway (central, larger) ---
gw_x, gw_y, gw_w, gw_h = 4.4, 2.8, 3.2, 2.4
gw_box = mpatches.FancyBboxPatch(
    (gw_x, gw_y), gw_w, gw_h,
    boxstyle="round,pad=0.2,rounding_size=0.2",
    facecolor="#f9fafb", edgecolor=RED, linewidth=2.5, zorder=5,
)
gw_box.set_boxstyle("round,pad=0.2,rounding_size=0.2")
gw_box.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
ax.add_patch(gw_box)
gw_cx = gw_x + gw_w / 2
gw_cy = gw_y + gw_h / 2
ax.text(gw_cx, gw_cy + 0.5, "Gateway", ha="center", va="center",
        fontsize=15, fontweight="bold", color=RED, zorder=6)
ax.text(gw_cx, gw_cy, "(网关)", ha="center", va="center",
        fontsize=11, color="#6b7280", zorder=6)
ax.text(gw_cx, gw_cy - 0.45, "路由 / 会话 / Agent", ha="center", va="center",
        fontsize=10, color="#9ca3af", zorder=6)

# --- LLM Provider ---
llm_x, llm_y, llm_w, llm_h = 9.0, 3.3, 2.6, 1.4
llm_cx, llm_cy = make_box(ax, llm_x, llm_y, llm_w, llm_h,
                            "LLM Provider", PURPLE, fontsize=12)

# --- 本地系统 (below gateway) ---
loc_x, loc_y, loc_w, loc_h = 4.4, 0.6, 3.2, 1.4
loc_cx, loc_cy = make_box(ax, loc_x, loc_y, loc_w, loc_h,
                            "本地系统", GREEN, fontsize=13,
                            sublabel="文件 / Shell / 网络")

# --- Arrows ---
# TUI -> Gateway (bidirectional)
arrow(ax, tui_x + tui_w, tui_cy + 0.1, gw_x, gw_cy + 0.1)
arrow(ax, gw_x, gw_cy - 0.1, tui_x + tui_w, tui_cy - 0.1)

# Gateway -> LLM (bidirectional)
arrow(ax, gw_x + gw_w, gw_cy + 0.1, llm_x, llm_cy + 0.1)
arrow(ax, llm_x, llm_cy - 0.1, gw_x + gw_w, gw_cy - 0.1)

# Gateway -> Local System (bidirectional, vertical)
arrow(ax, gw_cx + 0.15, gw_y, gw_cx + 0.15, loc_y + loc_h)
arrow(ax, gw_cx - 0.15, loc_y + loc_h, gw_cx - 0.15, gw_y)

# Label on arrows
ax.text((tui_x + tui_w + gw_x) / 2, tui_cy + 0.35, "请求/响应",
        ha="center", va="center", fontsize=9, color="#9ca3af", zorder=6)
ax.text((gw_x + gw_w + llm_x) / 2, gw_cy + 0.35, "推理调用",
        ha="center", va="center", fontsize=9, color="#9ca3af", zorder=6)
ax.text(gw_cx + 0.55, (gw_y + loc_y + loc_h) / 2, "工具调用",
        ha="left", va="center", fontsize=9, color="#9ca3af", zorder=6)

fig.savefig("/Users/kchen/Desktop/Project/skilltest/fig_02_dataflow_zh.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Done: fig_02_dataflow_zh.png")
