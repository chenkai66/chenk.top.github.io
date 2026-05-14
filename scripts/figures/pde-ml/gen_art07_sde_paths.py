#!/usr/bin/env python3
"""Art 07: SDE particle trajectories + histogram evolution."""
import matplotlib.pyplot as plt
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3"}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1.2, 1]})
fig.patch.set_facecolor(DARK_BG)

# Ornstein-Uhlenbeck: dX = -theta*X dt + sigma dW
np.random.seed(42)
theta, sigma = 1.0, 0.8
dt = 0.01
N_steps = 600
N_paths = 25
t = np.arange(N_steps) * dt

paths = np.zeros((N_paths, N_steps))
paths[:, 0] = np.random.randn(N_paths) * 2

for i in range(1, N_steps):
    paths[:, i] = paths[:, i-1] - theta*paths[:, i-1]*dt + sigma*np.sqrt(dt)*np.random.randn(N_paths)

# Top: sample paths
ax1.set_facecolor(DARK_BG)
for j in range(N_paths):
    ax1.plot(t, paths[j], color=C["blue"], alpha=0.3, lw=0.8)
ax1.axhline(0, color=C["green"], ls='--', lw=1, alpha=0.5)
mean_path = paths.mean(axis=0)
ax1.plot(t, mean_path, color=C["orange"], lw=2.5, label='Ensemble mean')
std_path = paths.std(axis=0)
ax1.fill_between(t, mean_path-std_path, mean_path+std_path, alpha=0.15, color=C["orange"])
ax1.set_ylabel('$X_t$', color=C["white"], fontsize=12)
ax1.set_title('Ornstein-Uhlenbeck SDE: $dX_t = -\\theta X_t\\,dt + \\sigma\\,dW_t$',
              color=C["white"], fontsize=13)
ax1.tick_params(colors=C["white"])
for spine in ax1.spines.values(): spine.set_color(C["white"])
ax1.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"])

# Bottom: histogram snapshots
ax2.set_facecolor(DARK_BG)
times_idx = [0, 50, 150, 500]
colors_hist = [C["purple"], C["blue"], C["orange"], C["green"]]
bins = np.linspace(-4, 4, 40)
for ti, col in zip(times_idx, colors_hist):
    ax2.hist(paths[:, ti], bins=bins, density=True, alpha=0.4, color=col, label=f't={t[ti]:.1f}')

# Exact stationary: N(0, sigma^2/(2*theta))
x_grid = np.linspace(-4, 4, 200)
var_stat = sigma**2 / (2*theta)
ax2.plot(x_grid, np.exp(-x_grid**2/(2*var_stat))/np.sqrt(2*np.pi*var_stat),
         color=C["white"], lw=2, ls='--', label=f'Stationary $N(0,{var_stat:.2f})$')

ax2.set_xlabel('$x$', color=C["white"], fontsize=12)
ax2.set_ylabel('Density', color=C["white"], fontsize=12)
ax2.set_title('Marginal Distribution Converges to Gibbs Equilibrium', color=C["white"], fontsize=13)
ax2.tick_params(colors=C["white"])
for spine in ax2.spines.values(): spine.set_color(C["white"])
ax2.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=9, ncol=3)

plt.tight_layout()
plt.savefig('/tmp/pde-ml-figs/fig8_sde_particle_trajectories.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("OK: fig8_sde_particle_trajectories.png")
