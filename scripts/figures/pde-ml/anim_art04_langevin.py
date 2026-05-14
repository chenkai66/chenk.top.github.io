#!/usr/bin/env python3
"""Anim Art 04: Langevin dynamics — 200 particles sampling a double-well potential."""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3"}

np.random.seed(42)
N_particles = 200
N_frames = 80
dt = 0.05
sigma = np.sqrt(2 * dt)

V = lambda x: 0.25 * (x**2 - 4)**2
grad_V = lambda x: x * (x**2 - 4)

x_particles = np.random.randn(N_particles) * 0.3

trajectories = [x_particles.copy()]
for _ in range(N_frames - 1):
    x_particles = x_particles - grad_V(x_particles) * dt + sigma * np.random.randn(N_particles)
    trajectories.append(x_particles.copy())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1.5, 1]})
fig.patch.set_facecolor(DARK_BG)
for ax in (ax1, ax2):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=C["white"])
    for spine in ax.spines.values():
        spine.set_color(C["white"])

x_grid = np.linspace(-4, 4, 300)
V_grid = V(x_grid)
gibbs = np.exp(-V_grid)
gibbs /= np.trapz(gibbs, x_grid)

ax1.plot(x_grid, V_grid, color=C["purple"], lw=2, alpha=0.5, label='$V(x)$')
scatter = ax1.scatter([], [], c=C["blue"], s=15, alpha=0.6, zorder=5)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-0.5, 5)
ax1.set_ylabel('$V(x)$', color=C["white"], fontsize=12)
ax1.set_title('Langevin Dynamics in Double-Well Potential', color=C["white"], fontsize=13)
ax1.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=10)

bins = np.linspace(-4, 4, 40)
ax2.plot(x_grid, gibbs, '--', color=C["green"], lw=2, alpha=0.7, label=r'Gibbs $\propto e^{-V}$')
hist_bars = ax2.bar(bins[:-1], np.zeros(len(bins)-1), width=np.diff(bins), align='edge',
                    color=C["orange"], alpha=0.6, edgecolor='none')
ax2.set_xlim(-4, 4)
ax2.set_ylim(0, 0.6)
ax2.set_xlabel('$x$', color=C["white"], fontsize=12)
ax2.set_ylabel('Density', color=C["white"], fontsize=12)
ax2.set_title('Histogram vs Gibbs Equilibrium', color=C["white"], fontsize=13)
ax2.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=10)

time_text = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, color=C["white"], fontsize=11, va='top')

def init():
    scatter.set_offsets(np.empty((0, 2)))
    time_text.set_text('')
    return [scatter, time_text] + list(hist_bars)

def animate(i):
    pts = trajectories[i]
    y_pts = V(pts)
    scatter.set_offsets(np.column_stack([pts, y_pts]))
    counts, _ = np.histogram(pts, bins=bins, density=True)
    for bar, h in zip(hist_bars, counts):
        bar.set_height(h)
    time_text.set_text(f't = {i * dt:.2f}')
    return [scatter, time_text] + list(hist_bars)

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=N_frames, interval=100, blit=True)
plt.tight_layout()
ani.save('/tmp/pde-ml-figs/anim_langevin_sampling.gif', writer=animation.PillowWriter(fps=10), dpi=100, savefig_kwargs={'facecolor': DARK_BG})
print("OK: anim_langevin_sampling.gif")
