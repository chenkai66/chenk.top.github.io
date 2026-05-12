#!/usr/bin/env python3
"""Regenerate fig5_workflows.png with proper containment (no overflow)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10.5)
ax.axis('off')
fig.patch.set_facecolor('#f8f8f8')

C_PURPLE = '#5C6BC0'
C_LIGHT = '#E8EAF6'
C_MID = '#C5CAE9'
C_TIP = '#EDE7F6'
C_CMD = '#F5F5F5'

# Title
ax.text(7, 10.1, 'Two Workflows You Will Use Every Day', ha='center', va='center',
        fontsize=19, fontweight='bold', color='#333')
ax.text(7, 9.7, 'left: project-wide search & replace   |   right: edit multiple files at once',
        ha='center', va='center', fontsize=10, color=C_PURPLE, fontstyle='italic')

# Outer box
outer = FancyBboxPatch((0.3, 0.3), 13.4, 9.1, boxstyle="round,pad=0.15",
                       linewidth=1.5, edgecolor='#ccc', facecolor='white')
ax.add_patch(outer)

def draw_workflow(x_start, width, title, steps, tip_title, tip_lines):
    x_mid = x_start + width / 2
    # Panel background
    panel = FancyBboxPatch((x_start, 0.5), width, 8.8, boxstyle="round,pad=0.1",
                           linewidth=1, edgecolor=C_MID, facecolor=C_LIGHT, alpha=0.3)
    ax.add_patch(panel)

    # Panel title
    ax.text(x_mid, 9.0, title, ha='center', va='center',
            fontsize=13, fontweight='bold', color=C_PURPLE, fontstyle='italic')

    # Steps
    step_y = 8.3
    for i, (cmd, desc) in enumerate(steps):
        y = step_y - i * 1.35

        # Circle number
        circle = plt.Circle((x_start + 0.6, y), 0.25, color=C_PURPLE, zorder=3)
        ax.add_patch(circle)
        ax.text(x_start + 0.6, y, str(i+1), ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', zorder=4)

        # Command box
        cmd_box = FancyBboxPatch((x_start + 1.1, y - 0.3), width - 1.5, 0.6,
                                boxstyle="round,pad=0.05", linewidth=1,
                                edgecolor='#ddd', facecolor=C_CMD)
        ax.add_patch(cmd_box)
        ax.text(x_start + 1.4, y, cmd, ha='left', va='center',
                fontsize=11, fontfamily='monospace', color='#333')

        # Description
        ax.text(x_start + 1.1, y - 0.5, desc, ha='left', va='top',
                fontsize=8.5, color='#666')

    # Tip box at bottom
    tip_y = 1.0
    tip_box = FancyBboxPatch((x_start + 0.2, 0.6), width - 0.4, 0.9,
                            boxstyle="round,pad=0.05", linewidth=0,
                            facecolor=C_TIP, alpha=0.8)
    ax.add_patch(tip_box)
    ax.text(x_start + 0.5, 1.35, tip_title, ha='left', va='center',
            fontsize=9, fontweight='bold', color=C_PURPLE)
    for j, line in enumerate(tip_lines):
        ax.text(x_start + 0.5, 1.05 - j * 0.22, line, ha='left', va='center',
                fontsize=8, color='#555')

# Workflow A: Search & Replace
draw_workflow(0.5, 6.2, 'Workflow A -- Search & Replace',
    [
        ('/oldName', 'search to verify the match before changing anything'),
        ('n    N', 'step through hits with n (next) and N (prev)'),
        (':%s//newName/gc', 'empty pattern reuses last search; gc = global + confirm'),
        ('y  n  a  q', 'y = yes, n = skip, a = all, q = quit replacement'),
        (':noh', "clear the leftover highlight when you're done"),
    ],
    'Tip:',
    ['narrow first  ->  :10,40s/foo/bar/g',
     'trust nothing -- always run with c (confirm) on the first pass']
)

# Workflow B: Multi-file Editing
draw_workflow(7.3, 6.2, 'Workflow B -- Multi-file Editing',
    [
        (':e  src/api.py', 'open file 1 in current window (becomes a buffer)'),
        (':vsp src/main.py', 'vertical split with file 2 -- side by side'),
        (':sp  README.md', 'horizontal split with file 3 above current'),
        ('Ctrl-w  h j k l', 'jump between windows in the four directions'),
        (':ls   :b api', 'list buffers, switch to one by name fragment'),
    ],
    'Mental model:',
    ['buffer = file in memory  |  window = view onto a buffer',
     'tabs are layouts of windows -- not files like in a browser']
)

plt.tight_layout(pad=0.2)
plt.savefig('/tmp/fig5_workflows.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('fig5 saved')
