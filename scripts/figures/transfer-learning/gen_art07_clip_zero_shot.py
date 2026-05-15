#!/usr/bin/env python3
"""TL Art 07: CLIP zero-shot pipeline + predicted class probabilities (light theme).

Long math expressions for cosine similarity and softmax are placed BELOW the box
as captions so they don't overflow into adjacent boxes.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

OUT_DIR = "/tmp/tl-figs"
os.makedirs(OUT_DIR, exist_ok=True)

fig = plt.figure(figsize=(16, 7.5))
fig.patch.set_facecolor("#fafaf6")
gs = GridSpec(1, 5, figure=fig, wspace=0.35)
ax = fig.add_subplot(gs[0, :3])
ax_bar = fig.add_subplot(gs[0, 3:])

ax.set_facecolor("#fafaf6")
ax.set_xlim(0, 12); ax.set_ylim(0, 7.5)
ax.axis("off")
ax.set_title("CLIP zero-shot: build a classifier on the fly from text prompts",
             color="#1a1a2e", fontsize=12, pad=12, loc="center")

BLUE   = "#2f80ed"
PURPLE = "#8e44ad"
GREEN  = "#27ae60"
ORANGE = "#e67e22"
GRAY   = "#7f8c8d"

def draw_box(cx, cy, w, h, top, sub, color, fill="white"):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=1.6, edgecolor=color, facecolor=fill)
    ax.add_patch(rect)
    if top:
        ax.text(cx, cy + (0.13 if sub else 0), top, ha="center", va="center",
                fontsize=11, color=color, fontweight="bold")
    if sub:
        ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                fontsize=8.5, color="#4a5568")

def arrow(x1, y1, x2, y2, color="#7f8c8d"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="->", mutation_scale=12,
                        linewidth=1.2, color=color)
    ax.add_patch(a)

# ---- top row: image branch ----
y_top = 5.4
draw_box(1.0, y_top, 1.4, 0.9, "image", "", GRAY)
draw_box(3.2, y_top, 1.6, 0.9, "image\nencoder", "ViT / ResNet", BLUE)
draw_box(5.5, y_top, 1.6, 0.9, "$I$", "image embedding", BLUE)
arrow(1.0 + 0.7, y_top, 3.2 - 0.8, y_top, BLUE)
arrow(3.2 + 0.8, y_top, 5.5 - 0.8, y_top, BLUE)

# ---- bottom row: text branch ----
y_bot = 2.1
prompt_text = '"a photo of a dog"\n"a photo of a cat"\n"a photo of a zebra"'
prompt_w = 2.4
rect = FancyBboxPatch((1.0 - prompt_w/2, y_bot - 0.85), prompt_w, 1.7,
                      boxstyle="round,pad=0.08",
                      linewidth=1.6, edgecolor=PURPLE, facecolor="white")
ax.add_patch(rect)
ax.text(1.0, y_bot + 0.25, prompt_text, ha="center", va="center",
        fontsize=8.5, color=PURPLE)
ax.text(1.0, y_bot - 0.6, "prompt template", ha="center", va="center",
        fontsize=8.5, color="#4a5568")

draw_box(3.2, y_bot, 1.6, 0.9, "text\nencoder", "Transformer", PURPLE)
draw_box(5.5, y_bot, 1.6, 0.9, "$T_1,\\ldots,T_K$", "text embeddings", PURPLE)
arrow(1.0 + prompt_w/2, y_bot, 3.2 - 0.8, y_bot, PURPLE)
arrow(3.2 + 0.8, y_bot, 5.5 - 0.8, y_bot, PURPLE)

# ---- right column: cosine similarity + softmax ----
y_mid = (y_top + y_bot) / 2

draw_box(8.2, y_mid, 1.7, 1.2, "cosine\nsimilarity", "", GREEN)
draw_box(10.7, y_mid, 1.7, 1.2, "softmax", "", ORANGE)
arrow(5.5 + 0.8, y_top - 0.05, 8.2 - 0.85, y_mid + 0.3, GREEN)
arrow(5.5 + 0.8, y_bot + 0.05, 8.2 - 0.85, y_mid - 0.3, GREEN)
arrow(8.2 + 0.85, y_mid, 10.7 - 0.85, y_mid, ORANGE)

# captions BELOW boxes (avoid overflow)
ax.text(8.2, y_mid - 1.0, r"$I\cdot T_k\,/\,\|I\|\,\|T_k\|$",
        ha="center", va="top", fontsize=10, color="#2c3e50")
ax.text(10.7, y_mid - 1.0, r"$\div\tau$,  then  $\arg\max$",
        ha="center", va="top", fontsize=10, color="#2c3e50")

# ---- bar chart: predicted class probs ----
classes = ["zebra", "horse", "dog", "cat", "tiger"]
probs = [0.52, 0.22, 0.07, 0.05, 0.14]
colors = [ORANGE if p == max(probs) else BLUE for p in probs]
y_pos = list(range(len(classes)))[::-1]
bars = ax_bar.barh(y_pos, probs, color=colors, edgecolor="white", height=0.65)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(classes, fontsize=11, color="#2c3e50")
ax_bar.set_xlim(0, 1.0)
ax_bar.set_xlabel(r"$P(c\mid x)$ via softmax$(I\cdot T_c\,/\,\tau)$",
                  fontsize=10, color="#2c3e50")
ax_bar.set_title("Predicted class probabilities", fontsize=11, color="#1a1a2e", pad=10)
ax_bar.set_facecolor("#fafaf6")
for spine in ("top", "right"):
    ax_bar.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax_bar.spines[spine].set_color("#cbd2d9")
ax_bar.tick_params(colors="#2c3e50")
for bar, p in zip(bars, probs):
    ax_bar.text(p + 0.015, bar.get_y() + bar.get_height()/2, f"{p:.2f}",
                va="center", fontsize=10, color="#2c3e50")

out = os.path.join(OUT_DIR, "fig5_clip_zero_shot.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"OK: {out}")
