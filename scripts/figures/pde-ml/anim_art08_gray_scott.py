#!/usr/bin/env python3
"""Anim Art 08: Gray-Scott reaction-diffusion pattern emergence."""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

DARK_BG = "#0d1117"
C = {"white": "#e6edf3"}

N = 128
F, k = 0.035, 0.065
Du, Dv = 0.16, 0.08
dt = 1.0

u = np.ones((N, N))
v = np.zeros((N, N))
r = N // 4
u[N//2-r:N//2+r, N//2-r:N//2+r] = 0.50
v[N//2-r:N//2+r, N//2-r:N//2+r] = 0.25
np.random.seed(42)
u += 0.02 * np.random.randn(N, N)
v += 0.02 * np.random.randn(N, N)

def laplacian(Z):
    return np.roll(Z,1,0) + np.roll(Z,-1,0) + np.roll(Z,1,1) + np.roll(Z,-1,1) - 4*Z

snapshot_interval = 200
total_steps = 12000
n_snapshots = total_steps // snapshot_interval

snapshots = []
for step in range(total_steps):
    Lu = laplacian(u)
    Lv = laplacian(v)
    uvv = u * v * v
    u += dt * (Du * Lu - uvv + F * (1 - u))
    v += dt * (Dv * Lv + uvv - (F + k) * v)
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    if step % snapshot_interval == 0:
        snapshots.append(v.copy())
        print(f"  Frame {len(snapshots)}/{n_snapshots}...")

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis('off')

im = ax.imshow(snapshots[0], cmap='inferno', interpolation='bilinear', vmin=0, vmax=0.35)
title = ax.set_title('Gray-Scott: Spot Formation', color=C["white"], fontsize=14, pad=10)
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color=C["white"], fontsize=11, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=DARK_BG, alpha=0.8, edgecolor='none'))

def animate(i):
    im.set_data(snapshots[i])
    step_text.set_text(f'Step {i * snapshot_interval}')
    return [im, step_text]

ani = animation.FuncAnimation(fig, animate, frames=len(snapshots), interval=130, blit=True)
plt.tight_layout()
ani.save('/tmp/pde-ml-figs/anim_gray_scott.gif', writer=animation.PillowWriter(fps=8), dpi=100, savefig_kwargs={'facecolor': DARK_BG})
print("OK: anim_gray_scott.gif")
