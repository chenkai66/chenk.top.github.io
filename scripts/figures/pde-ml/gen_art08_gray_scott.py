#!/usr/bin/env python3
"""Art 08: Gray-Scott reaction-diffusion patterns (2x2 grid)."""
import matplotlib.pyplot as plt
import numpy as np

DARK_BG = "#0d1117"
C = {"white": "#e6edf3"}

def gray_scott_sim(F, k, Du=0.16, Dv=0.08, N=128, dt=1.0, steps=8000):
    u = np.ones((N, N))
    v = np.zeros((N, N))
    r = N // 4
    u[N//2-r:N//2+r, N//2-r:N//2+r] = 0.50
    v[N//2-r:N//2+r, N//2-r:N//2+r] = 0.25
    np.random.seed(42)
    u += 0.02 * np.random.randn(N, N)
    v += 0.02 * np.random.randn(N, N)

    def laplacian(Z):
        return (np.roll(Z,1,0) + np.roll(Z,-1,0) + np.roll(Z,1,1) + np.roll(Z,-1,1) - 4*Z)

    for _ in range(steps):
        Lu = laplacian(u)
        Lv = laplacian(v)
        uvv = u * v * v
        u += dt * (Du * Lu - uvv + F * (1 - u))
        v += dt * (Dv * Lv + uvv - (F + k) * v)
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
    return v

configs = [
    ("Spots\n$F=0.035,\\, k=0.065$", 0.035, 0.065),
    ("Stripes\n$F=0.025,\\, k=0.055$", 0.025, 0.055),
    ("Labyrinth\n$F=0.029,\\, k=0.057$", 0.029, 0.057),
    ("Holes\n$F=0.039,\\, k=0.058$", 0.039, 0.058),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.patch.set_facecolor(DARK_BG)

for ax, (label, F, k) in zip(axes.flat, configs):
    print(f"  Simulating {label.split(chr(10))[0]}...")
    v = gray_scott_sim(F, k, steps=10000)
    ax.imshow(v, cmap='inferno', interpolation='bilinear')
    ax.set_title(label, color=C["white"], fontsize=12, pad=8)
    ax.axis('off')

fig.suptitle('Gray-Scott Reaction-Diffusion: Four Turing Morphologies',
             color=C["white"], fontsize=15, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/tmp/pde-ml-figs/fig8_gray_scott_patterns.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("OK: fig8_gray_scott_patterns.png")
