"""
fig_02_topology_en.py
Alt: Architecture topology: terminal -> tui -> gateway -> agent loop / skills index / tool registry -> LLM provider
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withSimplePatchShadow

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
ax.text(6, 6.55, "OpenClaw Architecture Topology",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1f2937")


def make_box(ax, x, y, w, h, label, color, fontsize=11, sublabel=None, style=BOXSTYLE):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        facecolor="white", edgecolor=color, linewidth=2, zorder=5,
    )
    box.set_boxstyle(style)
    box.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.18, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=6)
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center", fontsize=8,
                color="#6b7280", zorder=6)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=6)
    return cx, cy


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2, mutation_scale=18),
                zorder=3)


# --- Boxes ---
# Terminal
t_cx, t_cy = make_box(ax, 0.3, 2.8, 1.4, 1.0, "Terminal", BLUE, sublabel="(User)")

# TUI
tui_cx, tui_cy = make_box(ax, 2.5, 2.8, 1.2, 1.0, "TUI", AMBER)

# Gateway (large container)
gw_x, gw_y, gw_w, gw_h = 4.5, 1.5, 3.5, 3.6
gw_box = mpatches.FancyBboxPatch(
    (gw_x, gw_y), gw_w, gw_h,
    boxstyle="round,pad=0.2,rounding_size=0.2",
    facecolor="#f9fafb", edgecolor=RED, linewidth=2.5, zorder=3,
)
gw_box.set_boxstyle("round,pad=0.2,rounding_size=0.2")
gw_box.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
ax.add_patch(gw_box)
ax.text(gw_x + gw_w / 2, gw_y + gw_h - 0.35, "Gateway",
        ha="center", va="center", fontsize=14, fontweight="bold", color=RED, zorder=6)

# Sub-components inside gateway
sub_w, sub_h = 2.5, 0.65
sub_x = gw_x + (gw_w - sub_w) / 2
make_box(ax, sub_x, 3.95, sub_w, sub_h, "Agent Loop", PURPLE, fontsize=10, style=BOXSTYLE_SM)
make_box(ax, sub_x, 3.05, sub_w, sub_h, "Skills Index", GREEN, fontsize=10, style=BOXSTYLE_SM)
make_box(ax, sub_x, 2.15, sub_w, sub_h, "Tool Registry", BLUE, fontsize=10, style=BOXSTYLE_SM)

gw_cx = gw_x + gw_w / 2
gw_cy = gw_y + gw_h / 2

# LLM Provider
llm_x, llm_y, llm_w, llm_h = 8.8, 2.2, 2.8, 2.2
llm_box = mpatches.FancyBboxPatch(
    (llm_x, llm_y), llm_w, llm_h,
    boxstyle=BOXSTYLE,
    facecolor="white", edgecolor=BLUE, linewidth=2, zorder=5,
)
llm_box.set_boxstyle(BOXSTYLE)
llm_box.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
ax.add_patch(llm_box)
llm_cx = llm_x + llm_w / 2
llm_cy = llm_y + llm_h / 2
ax.text(llm_cx, llm_cy + 0.55, "LLM Provider", ha="center", va="center",
        fontsize=12, fontweight="bold", color=BLUE, zorder=6)
for i, name in enumerate(["DashScope", "Anthropic", "OpenAI"]):
    ax.text(llm_cx, llm_cy + 0.1 - i * 0.35, name, ha="center", va="center",
            fontsize=9, color="#6b7280", zorder=6)

# --- Arrows ---
arrow(ax, 0.3 + 1.4, t_cy, 2.5, tui_cy)
arrow(ax, 2.5 + 1.2, tui_cy, gw_x, gw_cy)
arrow(ax, gw_x + gw_w, gw_cy, llm_x, llm_cy)

fig.savefig("/Users/kchen/Desktop/Project/skilltest/fig_02_topology_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Done: fig_02_topology_en.png")
