"""
fig_01_quadrant_en.py
Alt: Where OpenClaw sits in the agent landscape: hosted vs self-hosted, chat-app vs your-surface
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withSimplePatchShadow

# --- Design tokens ---
BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
ARROW_COLOR = "#9ca3af"
SHADOW_KWARGS = dict(offset=(1.2, -1.2), shadow_rgbFace="#d1d5db", alpha=0.35)
# Use small pad/rounding for data-coordinate patches
BOXSTYLE = "round,pad=0.02,rounding_size=0.03"

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.set_aspect("equal")
ax.axis("off")

# Title
ax.text(0, 1.18, "Agent Landscape: Where Does OpenClaw Fit?",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1f2937")

# Draw quadrant lines
ax.axhline(0, color="#e5e7eb", lw=1.2, ls="--", zorder=1)
ax.axvline(0, color="#e5e7eb", lw=1.2, ls="--", zorder=1)

# Axis arrows
ax.annotate("", xy=(1.15, 0), xytext=(-1.15, 0),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2, mutation_scale=18))
ax.annotate("", xy=(0, 1.08), xytext=(0, -1.08),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2, mutation_scale=18))

# Axis labels
ax.text(-1.15, -0.08, "Hosted", ha="center", va="top", fontsize=11,
        fontweight="bold", color="#6b7280")
ax.text(1.15, -0.08, "Self-hosted", ha="center", va="top", fontsize=11,
        fontweight="bold", color="#6b7280")
ax.text(0.08, -1.08, "Chat App", ha="left", va="center", fontsize=11,
        fontweight="bold", color="#6b7280")
ax.text(0.08, 1.08, "Your Surface", ha="left", va="center", fontsize=11,
        fontweight="bold", color="#6b7280")

# Product positions: (x, y, label, color, w, h, fontsize, border_lw)
products = [
    # Top-left: Hosted + Your Surface
    (-0.55, 0.55, "Coze", AMBER, 0.32, 0.16, 11, 1.8),
    (-0.55, 0.30, "Dify", AMBER, 0.32, 0.16, 11, 1.8),
    # Bottom-left: Hosted + Chat App
    (-0.55, -0.40, "ChatGPT", BLUE, 0.36, 0.16, 11, 1.8),
    (-0.55, -0.65, "Claude", PURPLE, 0.32, 0.16, 11, 1.8),
    # Bottom-right: Self-hosted + Chat App
    (0.55, -0.40, "Ollama", RED, 0.32, 0.16, 11, 1.8),
    (0.55, -0.65, "LM Studio", RED, 0.38, 0.16, 11, 1.8),
    # Top-right: Self-hosted + Your Surface — OpenClaw highlighted
    (0.55, 0.45, "OpenClaw", GREEN, 0.48, 0.22, 15, 3.0),
]

for (x, y, label, color, w, h, fs, blw) in products:
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=BOXSTYLE,
        facecolor="white", edgecolor=color, linewidth=blw, zorder=5,
    )
    box.set_boxstyle(BOXSTYLE)
    box.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=color, zorder=6)

# Quadrant background labels
quad_labels = [
    (-0.55, 0.90, "Hosted + Your Surface"),
    (0.55, 0.90, "Self-hosted + Your Surface"),
    (-0.55, -0.90, "Hosted + Chat App"),
    (0.55, -0.90, "Self-hosted + Chat App"),
]
for (x, y, txt) in quad_labels:
    ax.text(x, y, txt, ha="center", va="center", fontsize=8, color="#b0b5bc", style="italic")

fig.savefig("/Users/kchen/Desktop/Project/skilltest/fig_01_quadrant_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Done: fig_01_quadrant_en.png")
