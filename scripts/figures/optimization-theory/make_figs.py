import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Style
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
})
C_BLUE = "#2E5BFF"
C_ORANGE = "#D97706"
C_GREEN = "#059669"
C_PURPLE = "#7C3AED"
C_GREY = "#475569"

# =========================================================
# FIG 8: Linear scaling rule — LR vs batch size
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)

# Left: max stable LR vs batch
B = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
eta_linear = 1e-3 * (B / 256)             # ideal linear scaling, anchored at 256
eta_sqrt   = 1e-3 * np.sqrt(B / 256)      # alternative sqrt rule
eta_obs    = eta_linear * (1 - 0.55 * (B / 8192) ** 1.6)  # diminishing returns at huge B

ax = axes[0]
ax.plot(B, eta_linear, "--", color=C_GREY, lw=2, label="Linear rule  η ∝ B")
ax.plot(B, eta_sqrt,   ":",  color=C_PURPLE, lw=2, label="Sqrt rule   η ∝ √B")
ax.plot(B, eta_obs,    "-",  color=C_BLUE,   lw=2.6, marker="o", ms=6, label="Empirical (with warmup)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(B)
ax.set_xticklabels([str(b) for b in B], fontsize=9, rotation=0)
ax.set_xlabel("Batch size B")
ax.set_ylabel("Maximum stable peak LR  η_max")
ax.set_title("Linear scaling rule and its limits")
ax.legend(loc="upper left", framealpha=0.95)

# Right: gradient noise vs batch
B2 = np.linspace(16, 8192, 400)
noise = 1.0 / np.sqrt(B2)
ax2 = axes[1]
ax2.plot(B2, noise, color=C_ORANGE, lw=2.6, label=r"Gradient SE $\propto 1/\sqrt{B}$")
# Annotate two regimes
ax2.fill_between(B2, 0, noise, where=(B2 < 256), color=C_ORANGE, alpha=0.10)
ax2.fill_between(B2, 0, noise, where=(B2 >= 256), color=C_BLUE,   alpha=0.08)
ax2.axvline(256, color=C_GREY, ls="--", lw=1)
ax2.text(80,  0.18, "Noise-dominated\n(small η, high variance)", fontsize=10, color=C_GREY)
ax2.text(900, 0.06, "Curvature-dominated\n(can scale η up)",     fontsize=10, color=C_GREY)
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Batch size B")
ax2.set_ylabel("Mini-batch gradient noise (relative)")
ax2.set_title("Why bigger batch ⇒ larger LR is safe")
ax2.legend(loc="upper right", framealpha=0.95)

plt.tight_layout()
plt.savefig("/tmp/fig8.png", dpi=180, bbox_inches="tight")
plt.close()
print("fig8 saved")

# =========================================================
# FIG 9: Training loss with vs without warmup + grad-norm
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)

steps = np.arange(0, 1000)

# Loss curves - simulated
np.random.seed(7)
def smooth_noise(n, sigma=0.04, alpha=0.85):
    x = np.zeros(n); v = 0.0
    for i in range(n):
        v = alpha * v + (1 - alpha) * np.random.randn() * sigma
        x[i] = v
    return x

# Without warmup: spike then partial recovery, higher final loss
loss_nowarm = 4.5 * np.exp(-steps / 350) + 1.6 + smooth_noise(len(steps), 0.06)
loss_nowarm[5:55] += 2.5 * np.exp(-(steps[5:55] - 25) ** 2 / 250)   # divergence spike
loss_nowarm[55:] += 0.35  # never fully recovers

# With warmup: smooth descent
loss_warm = 4.5 * np.exp(-steps / 280) + 1.30 + smooth_noise(len(steps), 0.04)

ax = axes[0]
ax.plot(steps, loss_nowarm, color=C_ORANGE, lw=2.0, label="No warmup")
ax.plot(steps, loss_warm,   color=C_BLUE,   lw=2.0, label="Linear warmup (5%)")
ax.axvspan(0, 50, color=C_BLUE, alpha=0.10)
ax.text(25, 6.3, "warmup\nwindow", ha="center", fontsize=9, color=C_BLUE)
ax.set_xlabel("Training step")
ax.set_ylabel("Training loss")
ax.set_title("Effect of warmup on early-training stability")
ax.legend(loc="upper right", framealpha=0.95)
ax.set_ylim(1.0, 7.0)

# Right: grad norm spikes
gn_nowarm = 2.0 + 0.5 * np.exp(-steps / 400) + 0.2 * smooth_noise(len(steps), 0.6, alpha=0.7)
gn_nowarm[10:40] += 8 * np.exp(-(steps[10:40] - 22) ** 2 / 80)   # huge spike
gn_warm   = 2.0 + 0.4 * np.exp(-steps / 400) + 0.18 * smooth_noise(len(steps), 0.5, alpha=0.7)

ax2 = axes[1]
ax2.plot(steps, gn_nowarm, color=C_ORANGE, lw=1.8, label="No warmup")
ax2.plot(steps, gn_warm,   color=C_BLUE,   lw=1.8, label="With warmup")
ax2.axhline(1.0, color=C_GREY, ls=":", lw=1.2, label="clip threshold = 1.0")
ax2.set_xlabel("Training step")
ax2.set_ylabel("Gradient norm")
ax2.set_title("Gradient-norm spikes that warmup prevents")
ax2.legend(loc="upper right", framealpha=0.95)
ax2.set_ylim(0, 12)

plt.tight_layout()
plt.savefig("/tmp/fig9.png", dpi=180, bbox_inches="tight")
plt.close()
print("fig9 saved")
