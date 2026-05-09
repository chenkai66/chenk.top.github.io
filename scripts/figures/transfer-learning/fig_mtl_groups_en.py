#!/usr/bin/env python3
"""Multi-Task Learning — Task Grouping (EN)."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)

fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 4)
ax.axis("off")


def box(cx, cy, w, h, color, label, fs=11, bold=False):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    b.set_path_effects([pe.withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(b)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold" if bold else "normal", zorder=4)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 connectionstyle="arc3,rad=0", **ARROW_KW, zorder=2))


def cluster_bg(x, y, w, h, color):
    r = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        facecolor=color, ec=color, linewidth=1.5, alpha=0.07, zorder=1,
    )
    ax.add_patch(r)


# Row geometry
rows = [
    {"y": 3.1, "color": PURPLE, "label": "Group 1",
     "tasks": ["Detect", "Segment", "Edges"], "encoder": "Encoder A"},
    {"y": 2.0, "color": BLUE, "label": "Group 2",
     "tasks": ["Depth", "Normals"], "encoder": "Encoder B"},
    {"y": 0.9, "color": GREEN, "label": "Group 3",
     "tasks": ["Pose", "Caption"], "encoder": "Encoder C"},
]

bw, bh = 1.6, 0.55  # task box size
ew, eh = 2.0, 0.6   # encoder box size
enc_cx = 11.0        # encoder center x
task_x0 = 2.5        # first task center x
task_gap = 2.0       # gap between task centers

for row in rows:
    y = row["y"]
    c = row["color"]
    tasks = row["tasks"]
    n = len(tasks)
    # cluster background
    cluster_left = task_x0 - bw / 2 - 0.3
    cluster_right = enc_cx + ew / 2 + 0.3
    cluster_bg(cluster_left, y - bh / 2 - 0.2,
               cluster_right - cluster_left, bh + 0.4, c)
    # group label
    ax.text(0.4, y, row["label"], fontsize=10.5, color=c,
            fontweight="bold", va="center")
    # task boxes
    for i, t in enumerate(tasks):
        tx = task_x0 + i * task_gap
        box(tx, y, bw, bh, c, t, fs=11)
    # arrows from each task to encoder
    for i in range(n):
        tx = task_x0 + i * task_gap + bw / 2
        arrow(tx, y, enc_cx - ew / 2, y)
    # encoder box
    box(enc_cx, y, ew, eh, c, row["encoder"], fs=12, bold=True)

ax.set_title("Multi-Task Learning — Task Grouping",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_mtl_groups_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_mtl_groups_en.png")
