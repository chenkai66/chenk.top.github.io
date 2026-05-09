#!/usr/bin/env python3
"""Multimodal LLM Improvement Cycle (EN)."""

import numpy as np
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

fig, ax = plt.subplots(figsize=(13, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.5)
ax.axis("off")


def box(cx, cy, w, h, color, label, fs=10.5, bold=False):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    b.set_path_effects([pe.withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(b)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold" if bold else "normal",
            zorder=4, linespacing=1.3)


def arrow(x1, y1, x2, y2, rad=0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}", **ARROW_KW, zorder=2,
    )
    ax.add_patch(a)


# Loop layout — 6 steps in a rounded rectangle cycle
# Top row (left to right): choose base -> baseline -> build eval -> identify failures
# Bottom row (right to left): design fix -> add data -> pick better arch
# then back up to "choose base"

colors = [PURPLE, BLUE, GREEN, RED, AMBER, BLUE]
labels = [
    "Choose\nBase Model",
    "Run\nBaseline",
    "Build\nPrivate Eval",
    "Identify Top-3\nFailure Modes",
    "Design\nFix",
    "Add Targeted\nData",
]

# Positions: top row 4 boxes, bottom row 2 boxes
positions = [
    (1.8, 3.3),   # choose base (top-left)
    (4.6, 3.3),   # baseline
    (7.4, 3.3),   # build eval
    (10.6, 3.3),  # identify failures (top-right)
    (10.6, 1.3),  # design fix (bottom-right)
    (6.5, 1.3),   # add data (bottom-center)
]
# "pick better arch" goes bottom-left, loops back up
labels.append("Pick Better\nArchitecture")
colors.append(PURPLE)
positions.append((2.5, 1.3))

bw, bh = 2.2, 0.85

for i, (cx, cy) in enumerate(positions):
    box(cx, cy, bw, bh, colors[i], labels[i], fs=10.5, bold=(i == 0))

# Arrows: top row left-to-right
for i in range(3):
    x1 = positions[i][0] + bw / 2
    x2 = positions[i + 1][0] - bw / 2
    y = positions[i][1]
    arrow(x1, y, x2, y)

# Arrow: top-right down to bottom-right
arrow(positions[3][0], positions[3][1] - bh / 2,
      positions[4][0], positions[4][1] + bh / 2)

# Arrows: bottom row right-to-left
arrow(positions[4][0] - bw / 2, positions[4][1],
      positions[5][0] + bw / 2, positions[5][1])
arrow(positions[5][0] - bw / 2, positions[5][1],
      positions[6][0] + bw / 2, positions[6][1])

# Arrow: bottom-left up to top-left (closing the loop)
arrow(positions[6][0], positions[6][1] + bh / 2,
      positions[0][0], positions[0][1] - bh / 2)

# Step numbers
for i, (cx, cy) in enumerate(positions):
    ax.text(cx - bw / 2 + 0.15, cy + bh / 2 - 0.1, str(i + 1),
            fontsize=8, color="white", fontweight="bold", zorder=5, alpha=0.7)

ax.set_title("Multimodal LLM Improvement Cycle",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_mllm_iteration_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_mllm_iteration_en.png")
