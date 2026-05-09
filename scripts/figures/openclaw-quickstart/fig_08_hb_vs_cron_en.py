"""Heartbeat vs Cron — Two Scheduling Primitives (EN)"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withSimplePatchShadow

# --- design tokens ---
BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
TITLE_COLOR = "#1f2937"
SHADOW = withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
ARROW_KW = dict(arrowstyle="-|>", color="#9ca3af", lw=2, mutation_scale=18)
BOX_KW = dict(boxstyle="round,pad=0.5,rounding_size=1.5", ec="none")

def make_box(ax, x, y, w, h, color, label, fontsize=12, alpha=1.0):
    box = mpatches.FancyBboxPatch((x, y), w, h, **BOX_KW, facecolor=color, alpha=alpha)
    box.set_path_effects([SHADOW])
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", wrap=True)
    return box

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 96, "Heartbeat vs Cron — Two Scheduling Primitives",
        ha="center", va="top", fontsize=14, fontweight="bold", color=TITLE_COLOR)

# --- Dividing line ---
ax.plot([50, 50], [5, 88], color="#d1d5db", lw=1.5, ls="--", zorder=0)

# ============= LEFT COLUMN — Heartbeat (green) =============
col_left = 25  # center x

# Header box
make_box(ax, col_left - 14, 74, 28, 10, GREEN, "Heartbeat", fontsize=13)

# Clock metaphor
ax.text(col_left, 69, "[Timer-based]", ha="center", va="center",
        fontsize=9.5, color="#6b7280", style="italic")

items_left = [
    "Runs on a timer (every N minutes)",
    "Reads HEARTBEAT.md for instructions",
    "Checks environment, decides action",
    "Best for: patrols, monitoring, reactive",
]

y_start = 60
for i, txt in enumerate(items_left):
    y = y_start - i * 13
    box = mpatches.FancyBboxPatch((col_left - 18, y - 4), 36, 9, **BOX_KW,
                                   facecolor="#d1fae5", alpha=0.9)
    box.set_path_effects([SHADOW])
    ax.add_patch(box)
    ax.text(col_left, y + 0.5, txt, ha="center", va="center",
            fontsize=10.5, color="#065f46", fontweight="medium")

# ============= RIGHT COLUMN — Cron (blue) =============
col_right = 75

make_box(ax, col_right - 14, 74, 28, 10, BLUE, "Cron", fontsize=13)

ax.text(col_right, 69, "[Schedule-based]", ha="center", va="center",
        fontsize=9.5, color="#6b7280", style="italic")

items_right = [
    "Runs on a schedule (cron expression)",
    "Triggers a specific Skill",
    "Executes task, sends result",
    "Best for: reports, briefings, proactive",
]

for i, txt in enumerate(items_right):
    y = y_start - i * 13
    box = mpatches.FancyBboxPatch((col_right - 18, y - 4), 36, 9, **BOX_KW,
                                   facecolor="#dbeafe", alpha=0.9)
    box.set_path_effects([SHADOW])
    ax.add_patch(box)
    ax.text(col_right, y + 0.5, txt, ha="center", va="center",
            fontsize=10.5, color="#1e3a5f", fontweight="medium")

plt.savefig("fig_08_hb_vs_cron_en.png", dpi=160, bbox_inches="tight",
            facecolor=BG, pad_inches=0.1)
plt.close()
print("saved fig_08_hb_vs_cron_en.png")
