#!/usr/bin/env python3
"""Regenerate fig7_template_library.png with proper title spacing (no text overlap)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('#fafafa')

# Title (top, with breathing room)
ax.text(7.5, 8.55, 'Prompt Template Library: same five blocks, swapped per task',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#222')
ax.text(7.5, 8.05, 'Reusing the same skeleton makes evaluation, caching, and version control dramatically easier.',
        ha='center', va='center', fontsize=10, color='#666', fontstyle='italic')

# Six template boxes in 2x3 grid
TEMPLATES = [
    {
        'pos': (0.5, 4.3, 4.5, 3.0),
        'color': '#5C6BC0',
        'bg': '#E8EAF6',
        'title': 'Extraction',
        'body': 'Schema: {name, date, amount}\nRules: dates as ISO 8601;\n  skip fields not in source.\nReturn JSON only.',
        'tag': 'role + task + format',
    },
    {
        'pos': (5.25, 4.3, 4.5, 3.0),
        'color': '#9C27B0',
        'bg': '#F3E5F5',
        'title': 'Classification',
        'body': "Labels: {bug, billing,\n  feature, other}\nIf unsure -> 'other' with\nconfidence < 0.5.",
        'tag': 'role + task + format',
    },
    {
        'pos': (10.0, 4.3, 4.5, 3.0),
        'color': '#26A69A',
        'bg': '#E0F2F1',
        'title': 'Retrieval-Augmented QA',
        'body': "Use ONLY the passages below.\nCite as [#1], [#2].\nIf insufficient: say\n'no evidence in context'.",
        'tag': 'role + task + format',
    },
    {
        'pos': (0.5, 0.6, 4.5, 3.0),
        'color': '#FF9800',
        'bg': '#FFF3E0',
        'title': 'Code Generation',
        'body': 'Language: Python 3.11.\nConstraints: no I/O,\n  pure function, type hints,\n  include 3 doctests.',
        'tag': 'role + task + format',
    },
    {
        'pos': (5.25, 0.6, 4.5, 3.0),
        'color': '#78909C',
        'bg': '#ECEFF1',
        'title': 'Summarization',
        'body': 'Output: 5 bullets, <=15\n  words each.\nPreserve numbers verbatim.\nNo opinions.',
        'tag': 'role + task + format',
    },
    {
        'pos': (10.0, 0.6, 4.5, 3.0),
        'color': '#37474F',
        'bg': '#F5F5F5',
        'title': 'Creative Rewrite',
        'body': 'Voice: warm, second-person.\nReading level: grade 8.\nKeep all factual claims;\nvary sentence length.',
        'tag': 'role + task + format',
    },
]

for tpl in TEMPLATES:
    x, y, w, h = tpl['pos']
    # Outer rounded box
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          linewidth=1.8, edgecolor=tpl['color'], facecolor=tpl['bg'])
    ax.add_patch(rect)

    # Title at top of box
    ax.text(x + 0.3, y + h - 0.35, tpl['title'], ha='left', va='center',
            fontsize=12, fontweight='bold', color=tpl['color'])

    # Body — left aligned, monospace, well below the title
    ax.text(x + 0.3, y + h - 1.05, tpl['body'], ha='left', va='top',
            fontsize=9, fontfamily='monospace', color='#333', linespacing=1.5)

    # Tag at bottom right
    tag_w, tag_h = 1.6, 0.32
    tag_x = x + w - tag_w - 0.25
    tag_y = y + 0.2
    tag_rect = FancyBboxPatch((tag_x, tag_y), tag_w, tag_h, boxstyle="round,pad=0.04",
                              linewidth=1, edgecolor=tpl['color'], facecolor='white', alpha=0.9)
    ax.add_patch(tag_rect)
    ax.text(tag_x + tag_w / 2, tag_y + tag_h / 2, tpl['tag'],
            ha='center', va='center', fontsize=7.5, color=tpl['color'])

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/fig7_template_library.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('fig7 saved')
