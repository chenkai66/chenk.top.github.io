#!/usr/bin/env python3
"""Anim Art 03: Wasserstein gradient flow — density evolving to Gibbs equilibrium."""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor(DARK_BG)
for ax in (ax1, ax2):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=C["white"])
    for spine in ax.spines.values():
        spine.set_color(C["white"])

x = np.linspace(-4, 4, 400)

V = 0.5 * (x**2 - 1)**2
gibbs = np.exp(-V)
gibbs /= np.trapz(gibbs, x)

rho0 = np.exp(-0.5 * (x - 2)**2 / 0.3**2)
rho0 /= np.trapz(rho0, x)

N_frames = 60
rhos = [rho0.copy()]
rho = rho0.copy()
dx = x[1] - x[0]
dt_flow = 0.04
for _ in range(N_frames - 1):
    log_rho = np.log(rho + 1e-30)
    grad_log_rho = np.gradient(log_rho, dx)
    grad_V = np.gradient(V, dx)
    flux = rho * (grad_log_rho + grad_V)
    div_flux = np.gradient(flux, dx)
    rho = rho + dt_flow * div_flux
    rho = np.maximum(rho, 1e-30)
    rho /= np.trapz(rho, x)
    rhos.append(rho.copy())

kl_values = []
for rho in rhos:
    kl = np.trapz(rho * np.log((rho + 1e-30) / (gibbs + 1e-30)), x)
    kl_values.append(max(kl, 1e-6))

line_rho, = ax1.plot([], [], color=C["blue"], lw=2.5, label=r'$\rho_t(x)$')
line_gibbs, = ax1.plot(x, gibbs, '--', color=C["green"], lw=2, alpha=0.7, label=r'Gibbs $\propto e^{-V}$')
line_V, = ax1.plot(x, V * 0.3, color=C["purple"], lw=1, alpha=0.4, label=r'$V(x)$ (scaled)')
ax1.set_xlim(-4, 4)
ax1.set_ylim(0, 1.2)
ax1.set_xlabel('$x$', color=C["white"], fontsize=12)
ax1.set_ylabel('Density', color=C["white"], fontsize=12)
ax1.set_title('Density Evolution', color=C["white"], fontsize=13)
ax1.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=9)

line_kl, = ax2.plot([], [], color=C["orange"], lw=2.5)
ax2.set_xlim(0, N_frames)
ax2.set_ylim(0, max(kl_values) * 1.1)
ax2.set_xlabel('Iteration', color=C["white"], fontsize=12)
ax2.set_ylabel('KL Divergence', color=C["white"], fontsize=12)
ax2.set_title('Free Energy Decay', color=C["white"], fontsize=13)

time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color=C["white"], fontsize=11, va='top')

def init():
    line_rho.set_data([], [])
    line_kl.set_data([], [])
    time_text.set_text('')
    return line_rho, line_kl, time_text

def animate(i):
    line_rho.set_data(x, rhos[i])
    line_kl.set_data(range(i + 1), kl_values[:i + 1])
    time_text.set_text(f't = {i}')
    return line_rho, line_kl, time_text

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=N_frames, interval=100, blit=True)
plt.tight_layout()
ani.save('/tmp/pde-ml-figs/anim_gradient_flow.gif', writer=animation.PillowWriter(fps=10), dpi=100, savefig_kwargs={'facecolor': DARK_BG})
print("OK: anim_gradient_flow.gif")
