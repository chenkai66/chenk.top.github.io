#!/usr/bin/env python3
"""Art 05: HNN vs LNN vs SympNet architecture comparison."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3", "gray": "#484f58"}

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.patch.set_facecolor(DARK_BG)

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=C["white"], alpha=0.25, lw=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', color=C["white"], fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, color=C["white"]):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

configs = [
    {"title": "HNN", "color": C["blue"],
     "boxes": [("(q, p)", 0.5, 0.9), ("MLP", 0.5, 0.7), (r"$H_\theta$", 0.5, 0.5),
               (r"$\frac{\partial H}{\partial p},\,-\frac{\partial H}{\partial q}$", 0.5, 0.3),
               (r"$(\dot{q}, \dot{p})$", 0.5, 0.1)],
     "pros": ["Energy exact", "Symplectic ODE"],
     "cons": ["Needs (q,p) split", "No dissipation"]},
    {"title": "LNN", "color": C["green"],
     "boxes": [("(q, q')", 0.5, 0.9), ("MLP", 0.5, 0.7), (r"$L_\theta(q,\dot{q})$", 0.5, 0.5),
               (r"$\frac{d}{dt}\frac{\partial L}{\partial\dot{q}} = \frac{\partial L}{\partial q}$", 0.5, 0.3),
               (r"$\ddot{q}$", 0.5, 0.1)],
     "pros": ["Generalized coords", "Auto constraints"],
     "cons": ["Hessian inversion", "Slower training"]},
    {"title": "SympNet", "color": C["purple"],
     "boxes": [("(q, p)", 0.5, 0.9), ("Shear 1", 0.5, 0.72), ("Shear 2", 0.5, 0.54),
               ("... Shear K", 0.5, 0.36),
               (r"$(q', p')$", 0.5, 0.1)],
     "pros": ["Exact symplectic", "Fast (no ODE)"],
     "cons": ["Fixed time step", "No energy guarantee"]},
]

for ax, cfg in zip(axes, configs):
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 1); ax.set_ylim(-0.15, 1.05)
    ax.axis('off')
    ax.set_title(cfg["title"], color=cfg["color"], fontsize=16, fontweight='bold', pad=10)
    for text, bx, by in cfg["boxes"]:
        draw_box(ax, bx, by, 0.7, 0.1, text, cfg["color"], fontsize=9)
    positions = [b[2] for b in cfg["boxes"]]
    for i in range(len(positions)-1):
        draw_arrow(ax, 0.5, positions[i]-0.05, 0.5, positions[i+1]+0.05, cfg["color"])
    pros_text = '\n'.join([f'+ {p}' for p in cfg["pros"]])
    cons_text = '\n'.join([f'- {c}' for c in cfg["cons"]])
    ax.text(0.5, -0.05, pros_text, ha='center', va='top', color=C["green"], fontsize=8, family='monospace')
    ax.text(0.5, -0.05 - 0.02*len(cfg["pros"]) - 0.02, cons_text, ha='center', va='top',
            color=C["orange"], fontsize=8, family='monospace')

plt.tight_layout()
plt.savefig('/tmp/pde-ml-figs/fig8_lnn_architecture.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("OK: fig8_lnn_architecture.png")
