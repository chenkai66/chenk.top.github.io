#!/usr/bin/env python3
"""On-call escalation flow — horizontal flow with decision branch."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.patheffects import withSimplePatchShadow

# --- Design tokens ---
BG = '#fdfcf9'
RED, AMBER, PURPLE, BLUE, GREEN = '#e85d4a', '#f5a834', '#8b5cf6', '#3b82f6', '#10b981'
CYAN, PINK, GRAY = '#06b6d4', '#ec4899', '#94a3b8'
ARROW_COLOR = '#9ca3af'
SHADOW_COLOR = '#d1d5db'

# Large coordinate space so pad=0.5 / rounding_size=1.5 work well
COORD_W, COORD_H = 70, 40
BOX_W, BOX_H = 10, 5
VIS_W = BOX_W + 1.0   # 11
VIS_H = BOX_H + 1.0   # 6
H_GAP = 2.5
DIAMOND_HALF = 3.5     # half-diagonal of diamond

FIG_W, FIG_H = 14, 8


def make_box(ax, cx, cy, label, color, w=BOX_W, h=BOX_H):
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


def make_diamond(ax, cx, cy, label, color, half=DIAMOND_HALF):
    verts = [(cx, cy + half), (cx + half, cy), (cx, cy - half),
             (cx - half, cy), (cx, cy + half)]
    poly = Polygon(verts, closed=True, facecolor=color, edgecolor='white',
                   linewidth=1.5, zorder=3)
    poly.set_path_effects([
        withSimplePatchShadow(offset=(1.2, -1.2),
                              shadow_rgbFace=SHADOW_COLOR, alpha=0.35)
    ])
    ax.add_patch(poly)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white', zorder=4)
    return cx, cy


def draw_arrow(ax, x0, y0, x1, y1, label=None, label_offset=(0, 0)):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-|>', color=ARROW_COLOR, lw=2,
        mutation_scale=18, zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=8.5, color='#475569',
                ha='center', va='center', fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc=BG, ec='none', alpha=0.9))


# --- Figure ---
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, COORD_W)
ax.set_ylim(0, COORD_H)
ax.set_aspect('equal')
ax.axis('off')

# --- Title & subtitle ---
ax.text(COORD_W / 2, COORD_H - 1.5, 'On-call escalation flow',
        ha='center', va='top', fontsize=16, fontweight='bold', color='#1e293b')
ax.text(COORD_W / 2, COORD_H - 4.5,
        'Alert  ->  ack  ->  mitigate  ->  escalate.  Most incidents are resolved at L1.',
        ha='center', va='top', fontsize=9, color='#64748b')

# --- Layout plan ---
# Main row: 3 boxes + 1 diamond centred at y=25
# Yes branch: Postmortem above diamond (y=33)
# No branch: L2 below diamond (y=17), L3 below L2 (y=9)
# All branch boxes share the diamond's x to keep things tidy.

y_main = 23

# Centre the 4 elements (3 boxes + diamond) in the coordinate space.
# Shift left a bit so the branch column (Postmortem / L2 / L3) fits on the right.
n_main = 4
total_w = n_main * VIS_W + (n_main - 1) * H_GAP
x_start = (COORD_W - total_w) / 2 + VIS_W / 2 - 3  # shift left 3 units
main_xs = [x_start + i * (VIS_W + H_GAP) for i in range(n_main)]

main_specs = [
    ("Alert fires\n(SLO burn)", RED),
    ("L1 on-call\nack < 5min", AMBER),
    ("Run runbook\n(restart, rollback,\ndrain region)", BLUE),
]
positions = []
for i, (label, color) in enumerate(main_specs):
    pos = make_box(ax, main_xs[i], y_main, label, color)
    positions.append(pos)

# Decision diamond
d_pos = make_diamond(ax, main_xs[3], y_main, "Mitigated?", CYAN)
positions.append(d_pos)

# Main flow arrows 1->2->3
for i in range(2):
    x0 = positions[i][0] + VIS_W / 2
    x1 = positions[i + 1][0] - VIS_W / 2
    draw_arrow(ax, x0, y_main, x1, y_main)
# Arrow box3 -> diamond
draw_arrow(ax, positions[2][0] + VIS_W / 2, y_main,
           positions[3][0] - DIAMOND_HALF, y_main)

# --- Yes branch (up from diamond) ---
y_yes = y_main + 8
x_yes = main_xs[3]
make_box(ax, x_yes, y_yes, "Postmortem\n(within 48h)", GREEN)
draw_arrow(ax, main_xs[3], y_main + DIAMOND_HALF,
           x_yes, y_yes - VIS_H / 2,
           label='yes', label_offset=(2.5, 0))

# --- No branch (down from diamond) ---
y_no = y_main - 8
x_no = main_xs[3]
make_box(ax, x_no, y_no, "L2 escalate\n(model team)", PURPLE)
draw_arrow(ax, main_xs[3], y_main - DIAMOND_HALF,
           x_no, y_no + VIS_H / 2,
           label='no', label_offset=(2.5, 0))

# --- L3 box (below L2) ---
y_l3 = y_no - 8
make_box(ax, x_no, y_l3, "L3 incident\ncommander", RED)
draw_arrow(ax, x_no, y_no - VIS_H / 2, x_no, y_l3 + VIS_H / 2)

# --- Bottom caption ---
ax.text(COORD_W / 2, 1.5,
        'SLO burn alert > paging > self-healing first; only escalate to model team for inference-quality regressions.',
        ha='center', va='bottom', fontsize=8.5, color='#64748b', style='italic')

# --- Save ---
out = '/Users/kchen/Desktop/Project/skilltest/fig6_oncall_escalation.png'
fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'Saved: {out}')
