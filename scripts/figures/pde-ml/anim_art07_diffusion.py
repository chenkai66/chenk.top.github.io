#!/usr/bin/env python3
"""Anim Art 07: Forward diffusion process — structured data dissolving into noise."""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3"}

np.random.seed(42)

theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
r = 1 + 0.5 * np.cos(5 * theta)
x0 = r * np.cos(theta)
y0 = r * np.sin(theta)
data = np.column_stack([x0, y0])

N_frames = 60
beta_min, beta_max = 0.02, 0.5
betas = np.linspace(beta_min, beta_max, N_frames)

frames_data = [data.copy()]
current = data.copy()
for beta in betas[1:]:
    noise = np.random.randn(*current.shape)
    current = np.sqrt(1 - beta) * current + np.sqrt(beta) * noise
    frames_data.append(current.copy())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(DARK_BG)
for ax in (ax1, ax2):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=C["white"])
    for spine in ax.spines.values():
        spine.set_color(C["white"])

scatter = ax1.scatter([], [], c=C["blue"], s=10, alpha=0.6)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_xlabel('$x_1$', color=C["white"], fontsize=12)
ax1.set_ylabel('$x_2$', color=C["white"], fontsize=12)
ax1.set_title('Forward Diffusion: $x_0 \\to x_T$', color=C["white"], fontsize=13)
ax1.set_aspect('equal')

bins = np.linspace(-4, 4, 50)
ax2.set_xlim(-4, 4)
ax2.set_ylim(0, 0.6)
ax2.set_xlabel('$x_1$', color=C["white"], fontsize=12)
ax2.set_ylabel('Density', color=C["white"], fontsize=12)
ax2.set_title('Marginal Distribution $\\to \\mathcal{N}(0, I)$', color=C["white"], fontsize=13)

x_gauss = np.linspace(-4, 4, 200)
gauss = np.exp(-x_gauss**2 / 2) / np.sqrt(2 * np.pi)
ax2.plot(x_gauss, gauss, '--', color=C["green"], lw=2, alpha=0.7, label='$\\mathcal{N}(0,1)$')
hist_bars = ax2.bar(bins[:-1], np.zeros(len(bins)-1), width=np.diff(bins), align='edge',
                    color=C["purple"], alpha=0.6, edgecolor='none', label='$p(x_1^{(t)})$')
ax2.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=10)

time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color=C["white"], fontsize=11, va='top')
beta_text = ax1.text(0.98, 0.95, '', transform=ax1.transAxes, color=C["orange"], fontsize=10, va='top', ha='right')

def init():
    scatter.set_offsets(np.empty((0, 2)))
    time_text.set_text('')
    beta_text.set_text('')
    return [scatter, time_text, beta_text] + list(hist_bars)

def animate(i):
    pts = frames_data[i]
    scatter.set_offsets(pts)
    alpha = max(0.2, 1.0 - i / N_frames)
    scatter.set_alpha(alpha)
    color_t = C["blue"] if i < N_frames // 2 else C["purple"]
    scatter.set_facecolors(color_t)
    counts, _ = np.histogram(pts[:, 0], bins=bins, density=True)
    for bar, h in zip(hist_bars, counts):
        bar.set_height(h)
    time_text.set_text(f't = {i}/{N_frames}')
    beta_text.set_text(f'$\\beta$ = {betas[min(i, len(betas)-1)]:.3f}')
    return [scatter, time_text, beta_text] + list(hist_bars)

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=N_frames, interval=100, blit=True)
plt.tight_layout()
ani.save('/tmp/pde-ml-figs/anim_forward_diffusion.gif', writer=animation.PillowWriter(fps=10), dpi=100, savefig_kwargs={'facecolor': DARK_BG})
print("OK: anim_forward_diffusion.gif")
