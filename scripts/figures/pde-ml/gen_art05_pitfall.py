#!/usr/bin/env python3
"""Art 05: Energy drift pitfalls — vanilla MLP vs HNN+Euler vs HNN+leapfrog."""
import matplotlib.pyplot as plt
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3"}

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

t = np.linspace(0, 50, 1000)

np.random.seed(3)
mlp_err = 0.01 * np.exp(0.06*t) + 0.005*np.cumsum(np.random.randn(1000))*0.01
hnn_euler = 0.005 * t * (1 + 0.1*np.sin(0.5*t)) + 0.001
hnn_leapfrog = 0.002 * np.sin(0.3*t) + 0.003*np.sin(1.7*t) + 0.001

ax.semilogy(t, np.abs(mlp_err)+1e-4, color=C["orange"], lw=2.5, label='Vanilla MLP (no structure)')
ax.semilogy(t, np.abs(hnn_euler)+1e-4, color=C["purple"], lw=2, label='HNN + Euler integrator')
ax.semilogy(t, np.abs(hnn_leapfrog)+1e-4, color=C["green"], lw=2, label='HNN + Leapfrog integrator')

ax.axhspan(1e-3, 5e-2, alpha=0.05, color=C["green"])
ax.axhspan(5e-2, 1e1, alpha=0.05, color=C["orange"])

ax.text(45, 3, 'DIVERGES', color=C["orange"], fontsize=11, fontweight='bold', ha='center')
ax.text(45, 0.15, 'Linear drift', color=C["purple"], fontsize=10, ha='center')
ax.text(45, 0.004, 'Bounded oscillation', color=C["green"], fontsize=10, ha='center')

ax.set_xlabel('Time', color=C["white"], fontsize=12)
ax.set_ylabel('|Relative Energy Error|', color=C["white"], fontsize=12)
ax.set_title('Pitfall: Structure-Preserving Architecture Needs Structure-Preserving Integrator',
             color=C["white"], fontsize=13)
ax.tick_params(colors=C["white"])
for spine in ax.spines.values(): spine.set_color(C["white"])
ax.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=11, loc='upper left')
ax.set_ylim(1e-4, 20)

plt.tight_layout()
plt.savefig('/tmp/pde-ml-figs/fig9_pitfall_energy_drift.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("OK: fig9_pitfall_energy_drift.png")
