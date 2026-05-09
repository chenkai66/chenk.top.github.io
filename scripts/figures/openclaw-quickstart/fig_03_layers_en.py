"""
fig_03_layers_en.py
Alt: The six layers of an OpenClaw agent
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withSimplePatchShadow

# --- Design tokens ---
BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
ARROW_COLOR = "#9ca3af"
SHADOW_KWARGS = dict(offset=(1.2, -1.2), shadow_rgbFace="#d1d5db", alpha=0.35)
BOXSTYLE = "round,pad=0.15,rounding_size=0.15"

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

# Title
ax.text(6, 6.65, "The Six Layers of an OpenClaw Agent",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1f2937")

layers = [
    ("Channels", "Telegram  |  DingTalk  |  WeChat  |  Web", BLUE),
    ("Gateway", "HTTP/WebSocket  |  Rate Limiting  |  Auth", AMBER),
    ("Router + Sessions + Pi Agent", "", RED),
    ("Tools + Skills", "", PURPLE),
    ("Memory + ContextEngine", "", GREEN),
    ("LLM Providers", "DashScope  |  Anthropic  |  OpenAI", BLUE),
]

layer_h = 0.72
gap = 0.15
total_h = len(layers) * layer_h + (len(layers) - 1) * gap
start_y = (6.2 - total_h) / 2 + 0.1
layer_w = 9.0
layer_x = (12 - layer_w) / 2

for i, (name, sub, color) in enumerate(layers):
    y = start_y + (len(layers) - 1 - i) * (layer_h + gap)
    box = mpatches.FancyBboxPatch(
        (layer_x, y), layer_w, layer_h,
        boxstyle=BOXSTYLE,
        facecolor="white", edgecolor=color, linewidth=2, zorder=5,
    )
    box.set_boxstyle(BOXSTYLE)
    box.set_path_effects([withSimplePatchShadow(**SHADOW_KWARGS)])
    ax.add_patch(box)

    cx = layer_x + layer_w / 2
    cy = y + layer_h / 2

    if sub:
        ax.text(cx, cy + 0.12, name, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color, zorder=6)
        ax.text(cx, cy - 0.16, sub, ha="center", va="center",
                fontsize=9, color="#6b7280", zorder=6)
    else:
        ax.text(cx, cy, name, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color, zorder=6)

    # Layer number badge
    badge_x = layer_x - 0.45
    badge_y = cy
    ax.text(badge_x, badge_y, str(i + 1), ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=7,
            bbox=dict(boxstyle="circle,pad=0.3", facecolor=color, edgecolor=color))

    # Arrow between layers (downward)
    if i < len(layers) - 1:
        arrow_y_start = y
        arrow_y_end = y - gap
        ax.annotate("", xy=(cx, arrow_y_end + 0.02), xytext=(cx, arrow_y_start - 0.02),
                     arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2, mutation_scale=18),
                     zorder=3)

fig.savefig("/Users/kchen/Desktop/Project/skilltest/fig_03_layers_en.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Done: fig_03_layers_en.png")
