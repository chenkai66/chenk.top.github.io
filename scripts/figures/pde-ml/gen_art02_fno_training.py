#!/usr/bin/env python3
"""Art 02: FNO training loss curve + prediction vs exact."""
import matplotlib.pyplot as plt
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor(DARK_BG)

# Left: Training loss curve
np.random.seed(7)
epochs = np.arange(1, 501)
loss = 0.5 * np.exp(-epochs/60) + 0.002 + 0.01*np.random.randn(500)*np.exp(-epochs/100)
loss = np.maximum(loss, 0.001)
ax1.set_facecolor(DARK_BG)
ax1.semilogy(epochs, loss, color=C["blue"], lw=1.5, alpha=0.9)
ax1.semilogy(epochs, 0.5*np.exp(-epochs/60)+0.002, color=C["orange"], lw=2, ls='--', label='trend')
ax1.axhline(y=0.002, color=C["green"], ls=':', lw=1, alpha=0.7, label='noise floor')
ax1.set_xlabel('Epoch', color=C["white"], fontsize=11)
ax1.set_ylabel('Relative L2 Loss', color=C["white"], fontsize=11)
ax1.set_title('FNO Training Convergence', color=C["white"], fontsize=13)
ax1.tick_params(colors=C["white"])
for spine in ax1.spines.values(): spine.set_color(C["white"])
ax1.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"])

# Right: Prediction vs Exact
x = np.linspace(0, 1, 256)
exact = -np.sin(np.pi * x) / (1 + np.exp(-(x-0.5)*25))
pred = exact + 0.015*np.sin(8*np.pi*x)*np.exp(-((x-0.5)/0.15)**2)
ax2.set_facecolor(DARK_BG)
ax2.plot(x, exact, color=C["blue"], lw=2.5, label='Exact (Cole-Hopf)')
ax2.plot(x, pred, '--', color=C["orange"], lw=2, label='FNO prediction')
ax2.fill_between(x, exact, pred, alpha=0.15, color=C["orange"])
ax2.set_xlabel('$x$', color=C["white"], fontsize=12)
ax2.set_ylabel('$u(x, t=1)$', color=C["white"], fontsize=12)
ax2.set_title('Burgers Equation at $t = 1$', color=C["white"], fontsize=13)
ax2.tick_params(colors=C["white"])
for spine in ax2.spines.values(): spine.set_color(C["white"])
ax2.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"])

plt.tight_layout()
plt.savefig('/tmp/pde-ml-figs/fig8_fno_training.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("OK: fig8_fno_training.png")
