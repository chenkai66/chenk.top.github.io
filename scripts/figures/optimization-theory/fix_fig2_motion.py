#!/usr/bin/env python3
"""Regenerate fig2_motion_cheatsheet.png with proper spacing (no text overlap)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f8f8f8')

# Colors
C_GREEN = '#4CAF50'
C_BLUE = '#2196F3'
C_ORANGE = '#FF9800'
C_RED = '#F44336'
C_PURPLE = '#9C27B0'
C_GRAY = '#607D8B'
C_BG = '#ffffff'
KEY_H = 0.7
KEY_W = 0.8

def draw_key(x, y, text, color, fontsize=13):
    rect = FancyBboxPatch((x - KEY_W/2, y - KEY_H/2), KEY_W, KEY_H,
                          boxstyle="round,pad=0.05", linewidth=1.5,
                          edgecolor=color, facecolor=C_BG)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontfamily='monospace', fontweight='bold', color=color)

def draw_label(x, y, text, fontsize=9, color='#444'):
    ax.text(x, y, text, ha='center', va='top', fontsize=fontsize, color=color)

# Title
ax.text(7, 9.6, 'Vim Motion Cheat Sheet', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#333')

# === Row 1: char-level (green) ===
y1 = 8.5
ax.text(1.2, y1, 'char', ha='right', va='center', fontsize=11, color=C_GRAY, fontstyle='italic')
for i, (k, lbl) in enumerate([('h','left'), ('j','down'), ('k','up'), ('l','right')]):
    x = 2.2 + i * 1.1
    draw_key(x, y1, k, C_GREEN)
    draw_label(x, y1 - 0.5, lbl)

# word-level (green)
ax.text(6.8, y1, 'word', ha='right', va='center', fontsize=11, color=C_GRAY, fontstyle='italic')
for i, (k, lbl) in enumerate([('w','next start'), ('b','prev start'), ('e','word end')]):
    x = 7.8 + i * 1.3
    draw_key(x, y1, k, C_GREEN)
    draw_label(x, y1 - 0.5, lbl)

# line-level (green)
ax.text(11.8, y1, 'line', ha='right', va='center', fontsize=11, color=C_GRAY, fontstyle='italic')
for i, (k, lbl) in enumerate([('0','start'), ('^','non-blank'), ('$','end')]):
    x = 12.0 + i * 0.9
    draw_key(x, y1, k, C_GREEN, fontsize=12)
    draw_label(x, y1 - 0.5, lbl)

# === Row 2: find on line (blue) ===
y2 = 7.1
ax.text(1.2, y2, 'find', ha='right', va='center', fontsize=11, color=C_GRAY, fontstyle='italic')
for i, (k, lbl) in enumerate([('f','char fwd'), ('F','char back'), ('t','till fwd'), (';','repeat'), (',','reverse')]):
    x = 2.2 + i * 1.2
    draw_key(x, y2, k, C_BLUE)
    draw_label(x, y2 - 0.5, lbl)

# === Row 3: file & screen (blue) ===
y3 = 5.7
ax.text(1.2, y3, 'file &\nscreen', ha='right', va='center', fontsize=10, color=C_GRAY, fontstyle='italic')
for i, (k, lbl) in enumerate([('g g','top'), ('G','bottom'), ('nG','line n')]):
    x = 2.2 + i * 1.2
    draw_key(x, y3, k, C_BLUE, fontsize=11)
    draw_label(x, y3 - 0.5, lbl)

# screen navigation (orange)
for i, (k, lbl) in enumerate([('⌃F','pg down'), ('⌃B','pg up'), ('⌃D','half dn'), ('⌃U','half up')]):
    x = 6.5 + i * 1.3
    draw_key(x, y3, k, C_ORANGE, fontsize=11)
    draw_label(x, y3 - 0.5, lbl)

# === Row 4: search (red + orange) ===
y4 = 4.3
ax.text(1.2, y4, 'search', ha='right', va='center', fontsize=11, color=C_GRAY, fontstyle='italic')
for i, (k, lbl) in enumerate([('/','fwd'), ('?','back'), ('n','next'), ('N','prev')]):
    x = 2.2 + i * 1.0
    draw_key(x, y4, k, C_RED)
    draw_label(x, y4 - 0.5, lbl)

for i, (k, lbl) in enumerate([('*','word fwd'), ('#','word back'), ('%','bracket')]):
    x = 6.5 + i * 1.3
    draw_key(x, y4, k, C_ORANGE)
    draw_label(x, y4 - 0.5, lbl)

# === Grammar section at bottom ===
y5 = 2.6
rect_bg = FancyBboxPatch((0.5, 1.0), 13, 2.4, boxstyle="round,pad=0.1",
                         linewidth=0, facecolor='#EDE7F6')
ax.add_patch(rect_bg)

ax.text(7, 3.15, 'Grammar:   operator  +  [count]  +  motion',
        ha='center', va='center', fontsize=14, fontweight='bold', color=C_PURPLE)

examples = [
    ('3w', '=  next 3 words'),
    ('5j', '=  move 5 lines down'),
    ('d2}', '=  delete 2 paragraphs'),
]
for i, (cmd, desc) in enumerate(examples):
    x = 2.5 + i * 4.0
    ax.text(x, 2.3, cmd, ha='center', va='center', fontsize=13,
            fontfamily='monospace', fontweight='bold', color=C_PURPLE)
    ax.text(x + 0.5, 2.3, desc, ha='left', va='center', fontsize=11, color='#555')

ax.text(7, 1.4, 'Rule of thumb: any motion can follow an operator (d / c / y) to operate on exactly that range.',
        ha='center', va='center', fontsize=10, color=C_PURPLE, fontstyle='italic')

plt.tight_layout(pad=0.3)
plt.savefig('/tmp/fig2_motion_cheatsheet.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('fig2 saved')
