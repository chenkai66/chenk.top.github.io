#!/usr/bin/env python3
"""Transformer Encoder-Decoder Architecture (EN)."""

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

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")


def draw_box(ax, cx, cy, w, h, color, label, fontsize=11, bold=False):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor=color, ec="none", zorder=3,
    )
    box.set_path_effects([pe.withSimplePatchShadow(**SHADOW_KW)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label="", label_offset=(0, 0.15), rad=0):
    cs = f"arc3,rad={rad}"
    arrow = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=cs,
                            **ARROW_KW, zorder=2)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=9.5,
                color="#374151", fontweight="bold", zorder=4)


# === ENCODER PATH (top row, y=3.8) ===
enc_y = 3.6

draw_box(ax, 1.2, enc_y, 2.0, 0.6, PURPLE, "src tokens", fontsize=10.5, bold=True)
draw_arrow(ax, 2.2, enc_y, 3.0, enc_y)
draw_box(ax, 4.0, enc_y, 2.0, 0.6, BLUE, "Embed + PE", fontsize=10.5)
draw_arrow(ax, 5.0, enc_y, 5.8, enc_y)
draw_box(ax, 7.2, enc_y, 2.4, 0.6, AMBER, "Encoder x N", fontsize=11, bold=True)
draw_arrow(ax, 8.4, enc_y, 9.3, enc_y)
draw_box(ax, 10.5, enc_y, 2.2, 0.6, GREEN, "Encoder Output", fontsize=10.5, bold=True)

# === K, V connection (vertical from encoder output to decoder cross-attn) ===
dec_y = 1.4
draw_arrow(ax, 10.5, enc_y - 0.3, 10.5, dec_y + 0.7, label="K, V", label_offset=(0.4, 0.0))

# === DECODER PATH (bottom row, y=1.4) ===
draw_box(ax, 1.2, dec_y, 2.0, 0.6, PURPLE, "tgt tokens", fontsize=10.5, bold=True)
draw_arrow(ax, 2.2, dec_y, 3.0, dec_y)
draw_box(ax, 4.0, dec_y, 2.0, 0.6, BLUE, "Embed + PE", fontsize=10.5)
draw_arrow(ax, 5.0, dec_y, 5.8, dec_y)
draw_box(ax, 7.2, dec_y, 2.4, 0.6, RED, "Decoder x N", fontsize=11, bold=True)
draw_arrow(ax, 8.4, dec_y, 9.3, dec_y)
draw_box(ax, 10.2, dec_y, 1.4, 0.6, BLUE, "Linear", fontsize=10.5)
draw_arrow(ax, 10.9, dec_y, 11.3, dec_y)
draw_box(ax, 12.1, dec_y, 1.4, 0.6, GREEN, "Softmax", fontsize=10.5)
draw_arrow(ax, 12.8, dec_y, 13.2, dec_y)
ax.text(13.5, dec_y, "output", ha="center", va="center", fontsize=10.5,
        color="#374151", fontweight="bold", zorder=4)

# === Annotation for decoder internals ===
ax.text(7.2, dec_y - 0.55, "masked self-attn + cross-attn",
        ha="center", va="center", fontsize=9.5, color="#6b7280",
        style="italic", zorder=4)

# Dashed box around cross-attention connection
from matplotlib.patches import FancyBboxPatch as FBP
highlight = FBP(
    (8.9, dec_y - 0.15), 3.2, enc_y - dec_y + 0.3,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    facecolor="none", ec=AMBER, lw=1.5, ls="--", zorder=1,
)
ax.add_patch(highlight)
ax.text(10.5, (enc_y + dec_y) / 2, "cross-attention",
        ha="center", va="center", fontsize=9, color=AMBER,
        fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, ec="none"))

ax.set_title("Transformer Encoder-Decoder Architecture",
             fontsize=14, fontweight="bold", color="#1f2937", pad=12)

plt.savefig("/Users/kchen/Desktop/Project/skilltest/fig_transformer_arch_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
plt.close()
print("Saved fig_transformer_arch_en.png")
