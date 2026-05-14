#!/usr/bin/env python3
"""Art 01: FDM vs FEM stencil comparison diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3", "gray": "#484f58"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)

# --- Left: FDM 5-point stencil ---
ax1.set_facecolor(DARK_BG)
ax1.set_xlim(-0.5, 4.5); ax1.set_ylim(-0.5, 4.5)
for i in range(5):
    for j in range(5):
        ax1.plot(i, j, 'o', color=C["gray"], markersize=6, zorder=1)
cx, cy = 2, 2
ax1.plot(cx, cy, 'o', color=C["blue"], markersize=14, zorder=3)
for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
    ax1.plot(cx+dx, cy+dy, 'o', color=C["orange"], markersize=10, zorder=2)
    ax1.annotate('', xy=(cx+dx*0.7, cy+dy*0.7), xytext=(cx+dx*0.3, cy+dy*0.3),
                 arrowprops=dict(arrowstyle='->', color=C["orange"], lw=2))

ax1.text(cx, cy-0.35, '$u_{i,j}$', ha='center', va='top', color=C["blue"], fontsize=12, fontweight='bold')
ax1.text(cx+1, cy+0.25, '$u_{i+1,j}$', ha='center', color=C["orange"], fontsize=9)
ax1.text(cx-1, cy+0.25, '$u_{i-1,j}$', ha='center', color=C["orange"], fontsize=9)
ax1.text(cx, cy+1.25, '$u_{i,j+1}$', ha='center', color=C["orange"], fontsize=9)
ax1.text(cx, cy-1.35, '$u_{i,j-1}$', ha='center', color=C["orange"], fontsize=9)

ax1.set_title('Finite Difference (FDM)\n5-point stencil on uniform grid', color=C["white"], fontsize=13, pad=10)
ax1.text(2, -0.45, r'$\nabla^2 u \approx \frac{u_{i+1}+u_{i-1}+u_{j+1}+u_{j-1}-4u_{i,j}}{h^2}$',
         ha='center', color=C["green"], fontsize=11)
ax1.axis('off')

# --- Right: FEM triangular mesh ---
ax2.set_facecolor(DARK_BG)
from matplotlib.tri import Triangulation
np.random.seed(42)
pts_x = np.concatenate([np.linspace(0, 4, 5), np.array([0.5, 1.5, 2.5, 3.5]),
                         np.array([1, 2, 3]), np.array([0.5, 1.5, 2.5, 3.5]),
                         np.linspace(0, 4, 5)])
pts_y = np.concatenate([np.zeros(5), np.ones(4), np.ones(3)*2, np.ones(4)*3, np.ones(5)*4])
pts_x += np.random.randn(len(pts_x))*0.1
pts_y += np.random.randn(len(pts_y))*0.1
tri = Triangulation(pts_x, pts_y)
ax2.triplot(tri, color=C["gray"], linewidth=0.8, alpha=0.6)
ax2.plot(pts_x, pts_y, 'o', color=C["gray"], markersize=4)

hi = 10
ax2.plot(pts_x[hi], pts_y[hi], 'o', color=C["blue"], markersize=14, zorder=3)
neighbors = set()
for s in range(len(tri.triangles)):
    t = tri.triangles[s]
    if hi in t:
        for v in t:
            if v != hi:
                neighbors.add(v)
                ax2.plot([pts_x[hi], pts_x[v]], [pts_y[hi], pts_y[v]], '-', color=C["orange"], lw=2, zorder=2)
for n in neighbors:
    ax2.plot(pts_x[n], pts_y[n], 'o', color=C["orange"], markersize=8, zorder=2)

ax2.fill([pts_x[hi]-0.3, pts_x[hi]+0.5, pts_x[hi]-0.1], [pts_y[hi]-0.1, pts_y[hi]+0.1, pts_y[hi]+0.6],
         alpha=0.15, color=C["purple"])
ax2.set_title('Finite Element (FEM)\nUnstructured mesh + basis functions', color=C["white"], fontsize=13, pad=10)
ax2.text(2, -0.6, r'$\int_\Omega \nabla\phi_i \cdot \nabla u\,dx = \int_\Omega f\phi_i\,dx$',
         ha='center', color=C["green"], fontsize=11)
ax2.axis('off')
ax2.set_xlim(-0.5, 4.5); ax2.set_ylim(-0.8, 4.5)

plt.tight_layout()
plt.savefig('/tmp/pde-ml-figs/fig8_fdm_fem_stencil.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("OK: fig8_fdm_fem_stencil.png")
