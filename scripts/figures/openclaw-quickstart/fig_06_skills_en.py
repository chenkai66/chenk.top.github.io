#!/usr/bin/env python3
"""Skill Composition Pipeline (EN)."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.patheffects import withSimplePatchShadow

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
GRAY = "#6b7280"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)

fig, ax = plt.subplots(figsize=(14, 4))
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


def draw_small_box(ax, cx, cy, w, h, label, fontsize=9):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor="#d1d5db", ec="none", zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="#374151", fontweight="normal", zorder=4)


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
ax.text(50, 96, "Morning Briefing — Skill Composition",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#1f2937")

# Dashed outline for the morning-briefing scope
outline = Rectangle((22, 15), 52, 68, linewidth=1.5, edgecolor="#d1d5db",
                     facecolor="none", linestyle="--", zorder=1)
ax.add_patch(outline)
ax.text(48, 80, "morning-briefing skill", ha="center", va="center",
        fontsize=10.5, color="#9ca3af", style="italic")

# Cron trigger
draw_box(ax, 10, 50, 11, 10, GREEN, "Cron\n(7 am)", fontsize=10.5, bold=True)

# Path A: today-calendar -> gcalcli
draw_box(ax, 35, 65, 14, 9, BLUE, "today-calendar", fontsize=10, bold=True)
draw_small_box(ax, 52, 65, 10, 7, "gcalcli", fontsize=9)

# Path B: summarize-headlines -> Playwright MCP -> HN
draw_box(ax, 35, 35, 16, 9, BLUE, "summarize-\nheadlines", fontsize=9.5, bold=True)
draw_box(ax, 55, 35, 14, 9, AMBER, "Playwright\nMCP", fontsize=9.5)
draw_small_box(ax, 70, 35, 7, 7, "HN", fontsize=9)

# Compose message
draw_box(ax, 82, 50, 12, 10, GREEN, "Compose\nmessage", fontsize=10, bold=True)

# Telegram output
draw_box(ax, 95, 50, 9, 10, BLUE, "Telegram", fontsize=9.5, bold=True)

# Arrows
# Cron -> morning-briefing area (splits)
draw_arrow(ax, 15.5, 55, 28, 65, rad=0.1)
draw_arrow(ax, 15.5, 45, 27, 35, rad=-0.1)

# Path A arrows
draw_arrow(ax, 42, 65, 47, 65)

# Path B arrows
draw_arrow(ax, 43, 35, 48, 35)
draw_arrow(ax, 62, 35, 66.5, 35)

# Merge into compose
draw_arrow(ax, 57, 65, 76, 55, rad=0.1)
draw_arrow(ax, 73.5, 35, 76, 45, rad=-0.1)

# Compose -> Telegram
draw_arrow(ax, 88, 50, 90.5, 50)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_06_skills_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_06_skills_en.png")
