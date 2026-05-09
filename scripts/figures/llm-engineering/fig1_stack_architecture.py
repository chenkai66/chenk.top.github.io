#!/usr/bin/env python3
"""End-to-end LLM serving stack — 3-row architecture diagram."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withSimplePatchShadow

# --- Design tokens ---
BG = '#fdfcf9'
RED, AMBER, PURPLE, BLUE, GREEN = '#e85d4a', '#f5a834', '#8b5cf6', '#3b82f6', '#10b981'
CYAN, PINK, GRAY = '#06b6d4', '#ec4899', '#94a3b8'
ARROW_COLOR = '#9ca3af'
SHADOW_COLOR = '#d1d5db'

# Use a large coordinate space so pad=0.5 / rounding_size=1.5 are small
# relative to the boxes.
COORD_W, COORD_H = 70, 40
BOX_W, BOX_H = 10, 5          # inner rect passed to FancyBboxPatch
# Visual box is BOX_W+1, BOX_H+1 = 11, 6 (pad adds 0.5 each side)
VIS_W = BOX_W + 1.0
VIS_H = BOX_H + 1.0
H_GAP = 2.0                    # gap between visual edges

FIG_W, FIG_H = 14, 8


def make_box(ax, cx, cy, label, color, w=BOX_W, h=BOX_H):
    """Draw a rounded box with inner rect (w, h) centred at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=3,
    )
    patch.set_boxstyle("round,pad=0.5,rounding_size=1.5")
    patch.set_path_effects([
        withSimplePatchShadow(offset=(1.2, -1.2),
                              shadow_rgbFace=SHADOW_COLOR, alpha=0.35)
    ])
    ax.add_patch(patch)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)
    return cx, cy


def draw_arrow(ax, x0, y0, x1, y1, label=None):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-|>', color=ARROW_COLOR, lw=2,
        mutation_scale=18, zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.5, label, fontsize=7, color='#64748b',
                ha='center', va='center', zorder=5)


def centered_xs(n, vis_w=VIS_W, gap=H_GAP, coord_w=COORD_W):
    """Return n x-centres so the row is horizontally centred."""
    total = n * vis_w + (n - 1) * gap
    x_start = (coord_w - total) / 2 + vis_w / 2
    return [x_start + i * (vis_w + gap) for i in range(n)]


# --- Figure ---
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, COORD_W)
ax.set_ylim(0, COORD_H)
ax.set_aspect('equal')
ax.axis('off')

# --- Title & subtitle ---
ax.text(COORD_W / 2, COORD_H - 1.5, 'End-to-end LLM serving stack',
        ha='center', va='top', fontsize=16, fontweight='bold', color='#1e293b')
ax.text(COORD_W / 2, COORD_H - 4.5,
        'Edge  ->  API gateway  ->  model gateway (router/cache/quotas)  ->  model servers  ->  cold-storage',
        ha='center', va='top', fontsize=9, color='#64748b')

# === Top row ===
y_top = 30
top_labels = [
    ("Client / SDK", GRAY),
    ("CDN + WAF", CYAN),
    ("API Gateway\n(authn, rate limit)", BLUE),
    ("Model Gateway\n(routing, caching,\nfallback)", PURPLE),
    ("Tracing\n+ Metrics", GRAY),
]
top_xs = centered_xs(len(top_labels))
top_pos = []
for i, (label, color) in enumerate(top_labels):
    pos = make_box(ax, top_xs[i], y_top, label, color)
    top_pos.append(pos)

# Arrows 1->2->3->4->5
for i in range(len(top_pos) - 1):
    x0 = top_pos[i][0] + VIS_W / 2
    x1 = top_pos[i + 1][0] - VIS_W / 2
    draw_arrow(ax, x0, y_top, x1, y_top)

# === Middle row ===
y_mid = 19
mid_labels = [
    ("vLLM / SGLang\nreplicas (GPU)", GREEN),
    ("TensorRT-LLM\n(low-latency)", GREEN),
    ("CPU fallback\n(rare path)", AMBER),
    ("3rd-party API\n(Claude, OpenAI)", PINK),
]
mid_xs = centered_xs(len(mid_labels))
mid_pos = []
for i, (label, color) in enumerate(mid_labels):
    pos = make_box(ax, mid_xs[i], y_mid, label, color)
    mid_pos.append(pos)

# Arrows from Model Gateway down to each middle box
gw_x, gw_y = top_pos[3]
for mx, my in mid_pos:
    draw_arrow(ax, gw_x, gw_y - VIS_H / 2, mx, my + VIS_H / 2)

# === Bottom row ===
y_bot = 8
bot_labels = [
    ("Object store\n(prompt caches,\nmodels)", GRAY),
    ("Postgres\n(usage, billing)", AMBER),
    ("Vector DB\n(retrieval)", PINK),
]
bot_xs = centered_xs(len(bot_labels))
bot_pos = []
for i, (label, color) in enumerate(bot_labels):
    pos = make_box(ax, bot_xs[i], y_bot, label, color)
    bot_pos.append(pos)

# Arrows: vLLM -> Object store, CPU fallback -> Postgres, 3rd-party -> Vector DB
connections = [(0, 0), (2, 1), (3, 2)]
for mid_i, bot_i in connections:
    mx = mid_pos[mid_i][0]
    bx = bot_pos[bot_i][0]
    draw_arrow(ax, mx, y_mid - VIS_H / 2, bx, y_bot + VIS_H / 2)

# --- Save ---
out = '/Users/kchen/Desktop/Project/skilltest/fig1_stack_architecture.png'
fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'Saved: {out}')
