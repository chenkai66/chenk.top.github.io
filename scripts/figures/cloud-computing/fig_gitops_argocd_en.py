#!/usr/bin/env python3
"""GitOps with ArgoCD/Flux — horizontal flow with branch-down (EN)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withSimplePatchShadow

BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
ARROW_C = "#9ca3af"
SHADOW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
BOX_KW = dict(boxstyle="round,pad=0.5,rounding_size=1.5", ec="none")

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("GitOps with ArgoCD / Flux", fontsize=14, fontweight="bold",
             pad=12, color="#1e293b")

# top row: Developer → CI builds image → CI updates manifest
top_boxes = [
    (0.5,  3.6, 2.8, 0.8, PURPLE, "Developer\nCommit"),
    (4.2,  3.6, 2.8, 0.8, BLUE,   "CI Builds\nContainer Image"),
    (7.9,  3.6, 3.2, 0.8, AMBER,  "CI Updates Manifest\nin Config Repo"),
]
# bottom row: ArgoCD watches → Cluster reconciles
bot_boxes = [
    (7.9,  1.6, 3.2, 0.8, RED,    "ArgoCD / Flux\nWatches Repo"),
    (7.9,  0.2, 3.2, 0.8, GREEN,  "Cluster Reconciles\nto Desired State"),
]

for x, y, w, h, color, label in top_boxes + bot_boxes:
    box = FancyBboxPatch((x, y), w, h, facecolor=color, **BOX_KW)
    box.set_path_effects([withSimplePatchShadow(**SHADOW)])
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", linespacing=1.4)

# horizontal arrows in top row
for i in range(len(top_boxes) - 1):
    x1 = top_boxes[i][0] + top_boxes[i][2] + 0.05
    x2 = top_boxes[i + 1][0] - 0.05
    y_mid = top_boxes[i][1] + top_boxes[i][3] / 2
    ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2,
                                mutation_scale=18))

# vertical arrows: manifest → ArgoCD → Cluster
pairs = [(top_boxes[2], bot_boxes[0]), (bot_boxes[0], bot_boxes[1])]
for src, dst in pairs:
    cx = src[0] + src[2] / 2
    y1 = src[1] - 0.05
    y2 = dst[1] + dst[3] + 0.05
    ax.annotate("", xy=(cx, y2), xytext=(cx, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2,
                                mutation_scale=18))

fig.savefig("fig_gitops_argocd_en.png", dpi=160, bbox_inches="tight",
            facecolor=BG, pad_inches=0.1)
plt.close()
print("saved fig_gitops_argocd_en.png")
