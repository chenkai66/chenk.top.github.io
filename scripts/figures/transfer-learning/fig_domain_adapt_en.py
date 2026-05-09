#!/usr/bin/env python3
"""Domain Adaptation — Decision Guide (EN)."""

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

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")


def box(cx, cy, w, h, color, label, fs=10.5, bold=False, wrap=False):
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


def arrow(x1, y1, x2, y2, label="", rad=0, label_offset=(0, 0.15)):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}", **ARROW_KW, zorder=2,
    )
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=9.5, color="#6b7280",
                ha="center", va="center", fontweight="bold", zorder=5)


# ── Question 1 ──
box(3.0, 6.2, 4.5, 0.65, PURPLE,
    "1. Do you have target-domain labels?", fs=11, bold=True)

# Yes branch
arrow(5.25, 6.2, 7.8, 6.2, label="yes", label_offset=(0, 0.2))
box(10.0, 6.2, 3.6, 0.65, GREEN,
    "Semi-supervised DA\nfine-tune + importance weighting", fs=10)

# No branch -> Q2
arrow(3.0, 5.87, 3.0, 5.15, label="no", label_offset=(0.3, 0))

# ── Question 2 ──
box(3.0, 4.7, 3.6, 0.65, PURPLE,
    "2. Where is the shift?", fs=11, bold=True)

# Covariate shift
arrow(4.8, 4.7, 7.8, 5.0, label="", rad=-0.15)
ax.text(6.5, 5.15, "covariate shift", fontsize=9.5, color="#6b7280",
        ha="center", style="italic")
box(10.0, 5.0, 3.6, 0.6, BLUE, "DANN / DAN", fs=11)
ax.text(10.0, 4.62, "P(X) differs", fontsize=9, color="#9ca3af", ha="center")

# Label shift
arrow(4.8, 4.6, 7.8, 4.0, label="", rad=0)
ax.text(6.5, 4.48, "label shift", fontsize=9.5, color="#6b7280",
        ha="center", style="italic")
box(10.0, 4.0, 3.6, 0.6, AMBER, "Importance Weighting", fs=11)
ax.text(10.0, 3.62, "P(Y) differs", fontsize=9, color="#9ca3af", ha="center")

# Concept drift
arrow(4.8, 4.5, 7.8, 3.05, label="", rad=0.15)
ax.text(6.5, 3.55, "concept drift", fontsize=9.5, color="#6b7280",
        ha="center", style="italic")
box(10.0, 3.05, 3.6, 0.6, RED, "Continual Learning", fs=11)

# No branch -> Q3
arrow(3.0, 4.37, 3.0, 2.55, label="next", label_offset=(0.4, 0))

# ── Question 3 ──
box(3.0, 2.1, 5.0, 0.65, PURPLE,
    "3. Can you access source data at adaptation?", fs=10.5, bold=True)

# Yes
arrow(5.5, 2.1, 7.8, 2.1, label="yes", label_offset=(0, 0.2))
box(10.0, 2.1, 3.6, 0.6, BLUE, "Standard UDA", fs=11)

# No
arrow(3.0, 1.77, 3.0, 1.15)
ax.text(3.35, 1.5, "no", fontsize=9.5, color="#6b7280", fontweight="bold")
box(5.5, 0.85, 4.2, 0.6, RED,
    "Source-free DA (SHOT, etc.)", fs=11)
arrow(3.0, 0.85, 3.4, 0.85)

ax.set_title("Domain Adaptation — Decision Guide",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_domain_adapt_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_domain_adapt_en.png")
