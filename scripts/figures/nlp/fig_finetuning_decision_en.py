#!/usr/bin/env python3
"""Fine-Tuning Strategy Decision Tree (EN)."""

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

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
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


def note(x, y, text, fs=9.5, color="#6b7280", ha="center"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center",
            style="italic", zorder=5)


# ── Q1: Need best score? ──
box(2.5, 5.3, 4.0, 0.6, PURPLE,
    "Need the absolute best score?", fs=11, bold=True)

arrow(4.5, 5.3, 7.3, 5.3)
ax.text(5.9, 5.5, "yes", fontsize=9.5, color="#6b7280", fontweight="bold")
box(9.5, 5.3, 3.8, 0.6, GREEN,
    "Full Fine-Tuning", fs=11, bold=True)
note(9.5, 4.9, "(if you can afford it)", fs=9)

arrow(2.5, 5.0, 2.5, 4.35)
ax.text(2.8, 4.7, "no", fontsize=9.5, color="#6b7280", fontweight="bold")

# ── Q2: Have GPU memory? ──
box(2.5, 3.9, 3.6, 0.6, PURPLE,
    "Have GPU memory?", fs=11, bold=True)

# Yes -> LoRA
arrow(4.3, 3.9, 7.3, 4.2, rad=-0.1)
ax.text(5.8, 4.3, "yes", fontsize=9.5, color="#6b7280", fontweight="bold")
box(9.5, 4.2, 3.8, 0.6, BLUE,
    "LoRA r=16 on q/k/v/o", fs=10.5)

# No -> QLoRA
arrow(4.3, 3.75, 7.3, 3.2, rad=0.1)
ax.text(5.8, 3.3, "no", fontsize=9.5, color="#6b7280", fontweight="bold")
box(9.5, 3.2, 3.8, 0.6, AMBER,
    "QLoRA r=16\nNF4 + paged AdamW", fs=10)

arrow(2.5, 3.6, 2.5, 2.75)

# ── Q3: < 100 examples? ──
box(2.5, 2.3, 3.8, 0.6, PURPLE,
    "Less than 100 examples?", fs=11, bold=True)

arrow(4.4, 2.3, 7.3, 2.3)
ax.text(5.8, 2.5, "yes", fontsize=9.5, color="#6b7280", fontweight="bold")
box(9.5, 2.3, 3.8, 0.6, RED,
    "Few-shot Prompting\n(skip fine-tuning)", fs=10)

arrow(2.5, 2.0, 2.5, 1.35)
ax.text(2.8, 1.7, "no", fontsize=9.5, color="#6b7280", fontweight="bold")

# ── Q4: Latency constraint? ──
box(2.5, 0.9, 4.0, 0.6, PURPLE,
    "Production latency constraint?", fs=11, bold=True)

arrow(4.5, 0.9, 7.3, 0.9)
ax.text(5.9, 1.1, "yes", fontsize=9.5, color="#6b7280", fontweight="bold")
box(9.5, 0.9, 3.8, 0.6, AMBER,
    "Distill into\nSmaller Model", fs=10.5)

ax.set_title("Fine-Tuning Strategy Decision Tree",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_finetuning_decision_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_finetuning_decision_en.png")
