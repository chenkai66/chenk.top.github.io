#!/usr/bin/env python3
"""Art 06: Spiral ODE fitting — true trajectory + Neural ODE fit + vector field."""
import matplotlib.pyplot as plt
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "white": "#e6edf3"}

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

# True spiral (damped oscillator)
t = np.linspace(0, 4*np.pi, 500)
r = 2 * np.exp(-0.1*t)
x_true = r * np.cos(t)
y_true = r * np.sin(t)

# "Neural ODE fit" — slightly perturbed
np.random.seed(42)
x_fit = x_true + 0.05*np.sin(3*t)*np.exp(-0.05*t)
y_fit = y_true + 0.05*np.cos(5*t)*np.exp(-0.05*t)

# Vector field
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 16), np.linspace(-2.5, 2.5, 16))
rr = np.sqrt(xx**2 + yy**2) + 0.01
ux = -0.1*xx - yy/rr * 0.5
uy = -0.1*yy + xx/rr * 0.5
speed = np.sqrt(ux**2 + uy**2)
ax.quiver(xx, yy, ux/speed, uy/speed, speed, cmap='cool', alpha=0.3, scale=25, width=0.004)

ax.plot(x_true, y_true, color=C["blue"], lw=3, label='True trajectory', zorder=3)
ax.plot(x_fit, y_fit, '--', color=C["orange"], lw=2, label='Neural ODE fit', zorder=4)

# Mark start/end
ax.plot(x_true[0], y_true[0], 'o', color=C["green"], markersize=12, zorder=5, label='Start')
ax.plot(x_true[-1], y_true[-1], 's', color=C["green"], markersize=10, zorder=5, label='End')

# Observation points
idx = np.linspace(0, len(t)-1, 20, dtype=int)
ax.scatter(x_true[idx], y_true[idx], color=C["white"], s=20, zorder=6, alpha=0.6, label='Observations')

ax.set_xlabel('$x_1$', color=C["white"], fontsize=13)
ax.set_ylabel('$x_2$', color=C["white"], fontsize=13)
ax.set_title('Neural ODE: Spiral Trajectory Fitting', color=C["white"], fontsize=14)
ax.tick_params(colors=C["white"])
for spine in ax.spines.values(): spine.set_color(C["white"])
ax.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=10)
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig('/tmp/pde-ml-figs/fig8_spiral_ode_fit.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("OK: fig8_spiral_ode_fit.png")
