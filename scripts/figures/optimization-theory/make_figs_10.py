"""Generate 5 matplotlib figures for SGD/variance reduction article."""
import numpy as np
import matplotlib.pyplot as plt

# Color palette
C_BLUE = "#2E5BFF"
C_ORANGE = "#D97706"
C_GREEN = "#059669"
C_PURPLE = "#7C3AED"
C_GRAY = "#475569"

DPI = 180

# ---------------------------------------------------------------
# Figure 1: SGD vs full GD trajectories on a 2D quadratic
# ---------------------------------------------------------------
def fig1_trajectories():
    np.random.seed(7)
    # quadratic f(x) = 0.5 x^T A x with A diag(10, 1) — ill-conditioned
    A = np.diag([10.0, 1.0])
    grad = lambda x: A @ x

    # SGD with noise
    def run_sgd(x0, eta, steps, sigma):
        xs = [x0.copy()]
        x = x0.copy()
        for _ in range(steps):
            noise = sigma * np.random.randn(2)
            g = grad(x) + noise
            x = x - eta * g
            xs.append(x.copy())
        return np.array(xs)

    def run_gd(x0, eta, steps):
        xs = [x0.copy()]
        x = x0.copy()
        for _ in range(steps):
            x = x - eta * grad(x)
            xs.append(x.copy())
        return np.array(xs)

    x0 = np.array([1.6, 1.4])
    gd = run_gd(x0, 0.18, 25)
    sgd = run_sgd(x0, 0.05, 200, sigma=2.5)

    # contours
    xx = np.linspace(-1.7, 1.9, 200)
    yy = np.linspace(-1.5, 1.7, 200)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = 0.5 * (10 * XX**2 + YY**2)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=DPI)
    ax.contour(XX, YY, ZZ, levels=20, colors="#cbd5e1", linewidths=0.7)

    ax.plot(sgd[:, 0], sgd[:, 1], color=C_ORANGE, lw=1.0, alpha=0.85,
            label="SGD (noisy)", zorder=2)
    ax.scatter(sgd[::5, 0], sgd[::5, 1], color=C_ORANGE, s=10, alpha=0.6, zorder=2)

    ax.plot(gd[:, 0], gd[:, 1], color=C_BLUE, lw=2.4, marker="o",
            markersize=4.5, label="Full GD (deterministic)", zorder=3)

    ax.scatter([0], [0], color=C_GREEN, s=180, marker="*",
               zorder=5, label=r"$x^\star$ (optimum)")
    ax.scatter([x0[0]], [x0[1]], color=C_GRAY, s=60, marker="s",
               zorder=4, label=r"$x_0$")

    ax.set_xlabel(r"$x_1$ (high-curvature direction)", fontsize=11)
    ax.set_ylabel(r"$x_2$ (low-curvature direction)", fontsize=11)
    ax.set_title("SGD vs Full GD trajectories on an ill-conditioned quadratic",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_xlim(-1.7, 1.9)
    ax.set_ylim(-1.5, 1.7)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig("/tmp/fig1.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print("fig1 saved")

# ---------------------------------------------------------------
# Figure 2: Convergence rate comparison (log-log)
# ---------------------------------------------------------------
def fig2_convergence_rates():
    T = np.logspace(0, 5, 200)

    # Suboptimality curves (illustrative scaling)
    sgd = 1.0 / np.sqrt(T)             # O(1/sqrt T)
    full_gd = (1 - 1/30.0) ** T         # geometric, kappa=30
    full_gd = np.maximum(full_gd, 1e-12)
    svrg_passes = T / 50.0              # one "epoch" per 50 grads
    svrg = 0.5 ** svrg_passes
    svrg = np.maximum(svrg, 1e-12)
    katyusha = 0.3 ** svrg_passes
    katyusha = np.maximum(katyusha, 1e-12)

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=DPI)
    ax.loglog(T, sgd, color=C_ORANGE, lw=2.4,
              label=r"SGD: $O(1/\sqrt{T})$")
    ax.loglog(T, full_gd, color=C_BLUE, lw=2.4,
              label=r"Full GD: $O((1-1/\kappa)^T)$, $\kappa=30$")
    ax.loglog(T, svrg, color=C_GREEN, lw=2.4,
              label=r"SVRG: linear in #epochs")
    ax.loglog(T, katyusha, color=C_PURPLE, lw=2.4, ls="--",
              label=r"Katyusha (accelerated VR)")

    ax.set_xlabel("Gradient evaluations $T$ (log scale)", fontsize=11)
    ax.set_ylabel(r"Suboptimality $f(x_T) - f^\star$ (log scale)", fontsize=11)
    ax.set_title("Convergence rates: SGD vs Full GD vs Variance Reduction",
                 fontsize=13)
    ax.set_ylim(1e-12, 2)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig("/tmp/fig2.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print("fig2 saved")

# ---------------------------------------------------------------
# Figure 3: Variance reduction visualization (gradient arrows)
# ---------------------------------------------------------------
def fig3_variance_arrows():
    np.random.seed(11)
    # True gradient direction at point x
    true_grad = np.array([1.0, 0.4])

    # SGD: large variance around true grad
    n_arrows = 24
    sgd_noise_scale = 0.55
    sgd_arrows = true_grad + sgd_noise_scale * np.random.randn(n_arrows, 2)

    # SVRG: tiny variance around true grad (control variate)
    svrg_noise_scale = 0.07
    svrg_arrows = true_grad + svrg_noise_scale * np.random.randn(n_arrows, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=DPI)

    for ax, arrows, color, title, scale in [
        (axes[0], sgd_arrows, C_ORANGE, "SGD: high variance", sgd_noise_scale),
        (axes[1], svrg_arrows, C_GREEN, "SVRG: variance suppressed", svrg_noise_scale),
    ]:
        for v in arrows:
            ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        alpha=0.45, lw=1.2))
        # True gradient (mean)
        ax.annotate("", xy=(true_grad[0], true_grad[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=C_BLUE,
                                    lw=3.0))
        ax.scatter([0], [0], color="black", s=40, zorder=5)
        ax.text(true_grad[0]+0.05, true_grad[1]+0.05,
                r"$\nabla f(x)$", color=C_BLUE, fontsize=12, fontweight="bold")
        ax.set_xlim(-1.3, 2.6)
        ax.set_ylim(-1.3, 2.0)
        ax.axhline(0, color="#cbd5e1", lw=0.6)
        ax.axvline(0, color="#cbd5e1", lw=0.6)
        ax.set_title(f"{title}\n(noise std $\\approx$ {scale})", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("white")

    fig.suptitle("Stochastic gradient samples: SGD vs SVRG control-variate estimator",
                 fontsize=13, y=1.02)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig("/tmp/fig3.png", dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("fig3 saved")

# ---------------------------------------------------------------
# Figure 4: Mini-batch noise vs batch size + critical batch size
# ---------------------------------------------------------------
def fig4_minibatch():
    B = np.logspace(0, 4, 200)
    sigma2 = 1.0
    var = sigma2 / B  # variance ~ 1/B

    # Effective speedup curve: ideal linear + saturation
    B_crit = 256.0
    speedup_ideal = B
    speedup_actual = B / (1 + B / B_crit)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), dpi=DPI)

    # Left: variance vs B
    ax = axes[0]
    ax.loglog(B, var, color=C_BLUE, lw=2.4, label=r"$\sigma^2 / B$")
    ax.set_xlabel("Batch size $B$ (log scale)", fontsize=11)
    ax.set_ylabel(r"Gradient variance (log scale)", fontsize=11)
    ax.set_title("Mini-batch reduces variance linearly in $B$", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_facecolor("white")

    # Right: speedup with critical batch size
    ax = axes[1]
    ax.loglog(B, speedup_ideal, color=C_GRAY, lw=1.8, ls="--",
              label="Ideal linear speedup")
    ax.loglog(B, speedup_actual, color=C_ORANGE, lw=2.6,
              label="Actual (linear scaling rule)")
    ax.axvline(B_crit, color=C_PURPLE, lw=1.5, ls=":",
               label=f"Critical batch $B^\\star \\approx {int(B_crit)}$")
    ax.set_xlabel("Batch size $B$ (log scale)", fontsize=11)
    ax.set_ylabel("Per-step learning-rate scaling (log)", fontsize=11)
    ax.set_title("Linear scaling rule saturates beyond $B^\\star$", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_facecolor("white")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig("/tmp/fig4.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print("fig4 saved")

# ---------------------------------------------------------------
# Figure 5: Total gradient evals to reach epsilon (bar chart)
# ---------------------------------------------------------------
def fig5_complexity_bars():
    # Realistic regime: n = 10000, kappa = 1000, epsilon = 1e-4
    n = 10000.0
    kappa = 1000.0
    eps = 1e-4
    log_inv_eps = np.log(1.0 / eps)

    methods = ["Full GD", "SGD", "SVRG / SAGA", "Katyusha"]
    # Complexities (prefactors illustrative; we drop constants)
    full_gd = n * kappa * log_inv_eps
    sgd_cost = (kappa ** 2) / eps          # O(kappa^2 / eps)
    svrg = (n + kappa) * log_inv_eps
    katy = (n + np.sqrt(n * kappa)) * log_inv_eps

    costs = np.array([full_gd, sgd_cost, svrg, katy])
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]

    fig, ax = plt.subplots(figsize=(11, 6.0), dpi=DPI)
    bars = ax.bar(methods, costs, color=colors, edgecolor="white", lw=1.5)

    ax.set_yscale("log")
    ax.set_ylabel("Total gradient evaluations to reach $\\epsilon=10^{-4}$ (log scale)",
                  fontsize=11)
    ax.set_title(f"Gradient-evaluation complexity: $n={int(n)}$, $\\kappa={int(kappa)}$",
                 fontsize=13)

    # Annotate complexity formulas above bars
    formulas = [
        r"$O(n\kappa \log(1/\epsilon))$",
        r"$O(\kappa^2 / \epsilon)$",
        r"$O((n+\kappa) \log(1/\epsilon))$",
        r"$O((n+\sqrt{n\kappa}) \log(1/\epsilon))$",
    ]
    for bar, c, formula in zip(bars, costs, formulas):
        ax.text(bar.get_x() + bar.get_width() / 2, c * 1.5,
                f"{formula}\n$\\approx${c:.1e}",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylim(1e6, costs.max() * 30)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig("/tmp/fig5.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print("fig5 saved")


if __name__ == "__main__":
    fig1_trajectories()
    fig2_convergence_rates()
    fig3_variance_arrows()
    fig4_minibatch()
    fig5_complexity_bars()
    print("All 5 figures done.")
