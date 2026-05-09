#!/usr/bin/env python3
"""Channel Routing Architecture (EN)."""

import matplotlib.pyplot as plt
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

fig, ax = plt.subplots(figsize=(14, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def draw_box(ax, cx, cy, w, h, color, label, fontsize=11, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        facecolor=color, ec="none", zorder=3,
    )
    box.set_path_effects([withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, rad=0, label="", label_offset=(0, 4)):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=f"arc3,rad={rad}",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9.5, color="#6b7280", style="italic")


# Title
ax.text(50, 96, "OpenClaw Channel Routing",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#1f2937")

# Left: 3 channel boxes stacked vertically
draw_box(ax, 12, 72, 18, 10, BLUE, "Telegram\npolling", fontsize=10.5, bold=True)
draw_box(ax, 12, 50, 18, 10, GREEN, "DingTalk\nStream", fontsize=10.5, bold=True)
draw_box(ax, 12, 28, 18, 10, AMBER, "WeChat\nWorkBuddy", fontsize=10.5, bold=True)

# Center: Gateway
draw_box(ax, 45, 50, 20, 30, PURPLE, "Gateway", fontsize=12, bold=True)
ax.text(45, 38, "message queue", ha="center", va="center",
        fontsize=9.5, color="white", style="italic", alpha=0.8, zorder=4)

# Right: Agent -> LLM
draw_box(ax, 70, 50, 16, 12, BLUE, "Pi Agent", fontsize=11, bold=True)
draw_box(ax, 92, 50, 14, 12, RED, "LLM\nProvider", fontsize=10.5, bold=True)

# Arrows: channels -> gateway
draw_arrow(ax, 21, 72, 35, 58, rad=0.15)
draw_arrow(ax, 21, 50, 35, 50)
draw_arrow(ax, 21, 28, 35, 42, rad=-0.15)

# Gateway -> Agent
draw_arrow(ax, 55, 50, 62, 50, label="dispatch")

# Agent -> LLM
draw_arrow(ax, 78, 50, 85, 50, label="inference")

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_05_channels_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_05_channels_en.png")
