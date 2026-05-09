#!/usr/bin/env python3
"""Observability Log Pipeline (EN)."""

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

fig, ax = plt.subplots(figsize=(14, 4.2))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.2)
ax.axis("off")

def draw_box(ax, cx, cy, w, h, color, label, fontsize=11, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    box.set_path_effects([withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, rad=0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=f"arc3,rad={rad}",
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)

stages = [
    (1.5,  PURPLE, "App stdout",           "Source"),
    (4.0,  BLUE,   "Fluent Bit /\nFilebeat", "Shipper"),
    (6.7,  AMBER,  "Kafka",                "Buffer"),
    (9.4,  RED,    "Logstash /\nFluentd",   "Processor"),
    (12.2, GREEN,  "Elasticsearch\n/ Loki", "Store"),
]

cy = 1.6
for cx, color, label, role in stages:
    draw_box(ax, cx, cy, 2.1, 0.9, color, label, fontsize=10.5)
    ax.text(cx, cy + 0.72, role, ha="center", va="bottom",
            fontsize=9.5, color="#6b7280", fontweight="bold")

# Arrows between stages
arrow_pairs = [(2.6, 2.9), (5.1, 5.6), (7.8, 8.3), (10.5, 11.1)]
for x1, x2 in arrow_pairs:
    draw_arrow(ax, x1, cy, x2, cy)

# Dashboard branch from Store
draw_box(ax, 12.2, 3.5, 2.1, 0.55, GREEN, "Kibana / Grafana", fontsize=9.5)
draw_arrow(ax, 12.2, 2.1, 12.2, 3.2)
ax.text(12.2, 2.7, "visualize", ha="center", va="center",
        fontsize=8.5, color="#6b7280", style="italic")

ax.set_title("Observability Log Pipeline",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_log_pipeline_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_log_pipeline_en.png")
