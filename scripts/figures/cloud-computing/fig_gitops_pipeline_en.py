#!/usr/bin/env python3
"""GitOps Deployment Pipeline (EN)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#fdfcf9"
RED = "#e85d4a"
AMBER = "#f5a834"
PURPLE = "#8b5cf6"
BLUE = "#3b82f6"
GREEN = "#10b981"
SHADOW_KW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")

def draw_box(ax, cx, cy, w, h, color, label, fontsize=11, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    box.set_path_effects([patheffects.withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, zorder=4, linespacing=1.3)

def draw_arrow(ax, x1, y1, x2, y2, rad=0, color="#9ca3af"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=f"arc3,rad={rad}",
                            arrowstyle="-|>", color=color, lw=2,
                            mutation_scale=18, zorder=2)
    ax.add_patch(arrow)

# Row 1 (top): git push -> CI Pipeline
draw_box(ax, 1.5, 3.8, 2.0, 0.7, PURPLE, "git push", fontsize=12, bold=True)

draw_arrow(ax, 2.55, 3.8, 3.4, 3.8)

# CI Pipeline — big box with sub-steps
draw_box(ax, 5.5, 3.8, 3.8, 0.8, BLUE, "CI: test, build, scan,\nsign image, push", fontsize=10.5, bold=True)

# Row 2 (middle): CI updates manifest repo
draw_arrow(ax, 5.5, 3.35, 5.5, 2.85)
draw_box(ax, 5.5, 2.3, 3.4, 0.7, AMBER, "Update Manifest Repo\n(image tag bump)", fontsize=10.5, bold=True)

# Row 3 (bottom): ArgoCD detects and syncs
draw_arrow(ax, 5.5, 1.9, 5.5, 1.45)
draw_box(ax, 5.5, 0.9, 3.4, 0.7, GREEN, "ArgoCD detects diff\n& syncs cluster", fontsize=10.5, bold=True)

# Right side: Cluster icon
draw_arrow(ax, 7.25, 0.9, 8.8, 0.9)
draw_box(ax, 10.0, 0.9, 2.2, 0.7, GREEN, "K8s Cluster", fontsize=11, bold=True)

# Annotations on the right side
ax.text(10.0, 0.35, "live workloads updated", ha="center", va="top",
        fontsize=9, color="#6b7280", style="italic")

# Phase labels on the left
ax.text(0.3, 3.8, "1", ha="center", va="center", fontsize=16,
        color=PURPLE, fontweight="bold", alpha=0.4)
ax.text(0.3, 2.3, "2", ha="center", va="center", fontsize=16,
        color=AMBER, fontweight="bold", alpha=0.4)
ax.text(0.3, 0.9, "3", ha="center", va="center", fontsize=16,
        color=GREEN, fontweight="bold", alpha=0.4)

# Dashed separation line for "GitOps boundary"
ax.axhline(y=1.65, xmin=0.05, xmax=0.95, color="#d1d5db",
           linestyle="--", lw=1, zorder=1)
ax.text(12.2, 1.75, "GitOps\nboundary", ha="center", va="bottom",
        fontsize=8.5, color="#9ca3af", style="italic")

ax.set_title("GitOps Deployment Pipeline",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_gitops_pipeline_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_gitops_pipeline_en.png")
