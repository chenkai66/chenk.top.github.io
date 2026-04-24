#!/usr/bin/env python3
"""
PDE+ML Part 06 - Continuous Normalizing Flows and Neural ODEs.

Generates 7 figures and writes them to BOTH the EN and ZH asset folders:
  - source/_posts/en/pde-ml/06-Continuous-Normalizing-Flows/
  - source/_posts/zh/pde-ml/06-连续归一化流与Neural-ODE/

Run from anywhere:
    python 06-cnf-neural-ode.py

Style: seaborn-v0_8-whitegrid, dpi=150
Palette: blue #2563eb, purple #7c3aed, green #10b981, red #ef4444
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal, gaussian_kde

# Shared style ----------------------------------------------------------------
import sys
from pathlib import Path as _StylePath
sys.path.insert(0, str(_StylePath(__file__).parent.parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Color palette (from shared _style)
BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
RED = COLORS["danger"]
GRAY = COLORS["gray"]
ORANGE = COLORS["warning"]


DPI = 150
RNG = np.random.default_rng(0)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "pde-ml" / "06-Continuous-Normalizing-Flows"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "pde-ml" / "06-连续归一化流与Neural-ODE"

EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to BOTH EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        path = d / name
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers: a hand-crafted "trained CNF" velocity field (analytic, no PyTorch)
# ---------------------------------------------------------------------------

def crescent_samples(n: int, rng=RNG) -> np.ndarray:
    """A crescent (two moons) target distribution."""
    n1 = n // 2
    n2 = n - n1
    theta1 = rng.uniform(0, np.pi, n1)
    x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    x1 += 0.08 * rng.standard_normal(x1.shape)
    theta2 = rng.uniform(0, np.pi, n2)
    x2 = np.stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)], axis=1)
    x2 += 0.08 * rng.standard_normal(x2.shape)
    return np.vstack([x1, x2]) - np.array([0.5, 0.25])


def linear_flow(z0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation flow (Flow Matching style optimal transport)."""
    return (1.0 - t) * z0 + t * x1


# ---------------------------------------------------------------------------
# Figure 1: Density transformation simple -> complex via continuous flow
# ---------------------------------------------------------------------------

def fig1_density_transformation() -> None:
    print("Figure 1: density transformation along a continuous flow")
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))

    n = 4000
    z0 = RNG.standard_normal((n, 2))
    x1 = crescent_samples(n)
    # Pair samples by nearest neighbour for smoother visual
    order = np.argsort(np.arctan2(z0[:, 1], z0[:, 0]))
    x1 = x1[np.argsort(np.arctan2(x1[:, 1], x1[:, 0]))]
    z0 = z0[order]

    times = np.linspace(0.0, 1.0, 5)
    titles = [r"$t=0$ (base $\mathcal{N}(0,I)$)",
              r"$t=0.25$",
              r"$t=0.5$",
              r"$t=0.75$",
              r"$t=1$ (target)"]
    cmaps = ["Blues", "BuPu", "Purples", "RdPu", "Reds"]

    for ax, t, title, cmap in zip(axes, times, titles, cmaps):
        z = linear_flow(z0, x1, t)
        # KDE over a grid for a smooth density field
        kde = gaussian_kde(z.T, bw_method=0.18)
        grid_x, grid_y = np.mgrid[-2.5:2.5:120j, -2.0:2.0:120j]
        positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
        density = kde(positions).reshape(grid_x.shape)

        ax.imshow(density.T, extent=[-2.5, 2.5, -2.0, 2.0], origin="lower",
                  cmap=cmap, aspect="auto")
        ax.contour(grid_x, grid_y, density, levels=6,
                   colors="white", linewidths=0.6, alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.0, 2.0)

    fig.suptitle(
        r"Continuous flow: $\partial_t\rho + \nabla\!\cdot(\rho v_\theta)=0$"
        r"  reshapes a Gaussian into a complex target",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    save(fig, "fig1_density_transformation.png")


# ---------------------------------------------------------------------------
# Figure 2: Neural ODE architecture (ResNet -> Neural ODE)
# ---------------------------------------------------------------------------

def fig2_neural_ode_vs_resnet() -> None:
    print("Figure 2: ResNet (discrete) vs Neural ODE (continuous)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: ResNet ---
    ax = axes[0]
    ax.set_title("ResNet: discrete depth", fontsize=13)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_xticks([])
    ax.set_yticks([])

    layer_x = np.linspace(1, 9, 6)
    for i, x in enumerate(layer_x):
        box = FancyBboxPatch((x - 0.4, 2.5), 0.8, 2.0,
                             boxstyle="round,pad=0.05",
                             linewidth=1.5, edgecolor=BLUE,
                             facecolor="white")
        ax.add_patch(box)
        ax.text(x, 3.5, f"$h_{i}$", ha="center", va="center", fontsize=11)
        if i < len(layer_x) - 1:
            ax.annotate("", xy=(layer_x[i + 1] - 0.45, 3.5),
                        xytext=(x + 0.45, 3.5),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
            ax.text((x + layer_x[i + 1]) / 2, 4.05,
                    f"$+f_{i}$", ha="center", fontsize=9, color=PURPLE)
            # skip arc
            arc_x = np.linspace(x + 0.45, layer_x[i + 1] - 0.45, 30)
            arc_y = 5.2 + 0.4 * np.sin(np.linspace(0, np.pi, 30))
            ax.plot(arc_x, arc_y, color=GREEN, lw=1.2, ls="--", alpha=0.85)

    ax.text(5, 1.6, r"$h_{l+1}=h_l+f_l(h_l)$  (Euler step, $\Delta t=1$)",
            ha="center", fontsize=11, color=BLUE)
    ax.text(5, 0.7, "Memory: $O(L)$ activations stored",
            ha="center", fontsize=10, color=RED)
    ax.text(5, 6.4, "Each block has its own parameters",
            ha="center", fontsize=10, color=GRAY)

    # --- Right: Neural ODE ---
    ax = axes[1]
    ax.set_title("Neural ODE: continuous depth", fontsize=13)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Continuous trajectory
    t = np.linspace(0, 1, 200)
    traj_x = 1 + 8 * t
    traj_y = 3.5 + 0.8 * np.sin(2.5 * np.pi * t) * np.exp(-1.5 * t)
    ax.plot(traj_x, traj_y, color=BLUE, lw=2.5, label=r"$h(t)$")

    # solver checkpoints (adaptive)
    cps = [0.0, 0.12, 0.27, 0.42, 0.55, 0.7, 0.82, 0.92, 1.0]
    for tt in cps:
        x = 1 + 8 * tt
        y = 3.5 + 0.8 * np.sin(2.5 * np.pi * tt) * np.exp(-1.5 * tt)
        ax.plot(x, y, "o", color=PURPLE, markersize=7, zorder=3)

    ax.annotate(r"$h(0)$", xy=(1, 3.5), xytext=(0.2, 4.6),
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.annotate(r"$h(T)$", xy=(9, 3.5 + 0.8 * np.sin(2.5 * np.pi) * np.exp(-1.5)),
                xytext=(9.0, 4.8),
                arrowprops=dict(arrowstyle="->", color=GRAY))

    ax.text(5, 1.6, r"$\dfrac{dh}{dt}=f_\theta(h(t),t)$  (single network)",
            ha="center", fontsize=11, color=BLUE)
    ax.text(5, 0.7, "Memory: $O(1)$ via adjoint method",
            ha="center", fontsize=10, color=GREEN)
    ax.text(5, 6.4, "Adaptive solver chooses step count",
            ha="center", fontsize=10, color=GRAY)

    fig.suptitle("From discrete residual blocks to a continuous-time ODE",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_neural_ode_vs_resnet.png")


# ---------------------------------------------------------------------------
# Figure 3: Adjoint sensitivity method (forward + reverse trajectory)
# ---------------------------------------------------------------------------

def fig3_adjoint_method() -> None:
    print("Figure 3: adjoint sensitivity method")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- Left: forward + reverse trajectories on a 2D vector field ---
    ax = axes[0]
    # Spiral vector field: dx/dt = -y - 0.1 x ; dy/dt = x - 0.1 y
    grid = np.linspace(-2.5, 2.5, 22)
    X, Y = np.meshgrid(grid, grid)
    U = -Y - 0.1 * X
    V = X - 0.1 * Y
    M = np.hypot(U, V)
    ax.quiver(X, Y, U / M, V / M, M, cmap="Blues", scale=35, width=0.0035,
              alpha=0.55, pivot="middle")

    def f(t, z): return [-z[1] - 0.1 * z[0], z[0] - 0.1 * z[1]]

    sol = solve_ivp(f, [0, 6], [2.0, 0.0], t_eval=np.linspace(0, 6, 400))
    ax.plot(sol.y[0], sol.y[1], color=BLUE, lw=2.4,
            label=r"Forward $h(t)$,  $0\to T$")
    ax.plot(sol.y[0][::-1], sol.y[1][::-1], color=RED, lw=1.6, ls="--",
            label=r"Reverse $h(t)$ + adjoint $a(t)$,  $T\to 0$")
    ax.plot(sol.y[0, 0], sol.y[1, 0], "o", color=GREEN, ms=10, zorder=4,
            label="$h(0)$")
    ax.plot(sol.y[0, -1], sol.y[1, -1], "s", color=PURPLE, ms=10, zorder=4,
            label="$h(T)$")
    ax.set_title("Adjoint method: forward + reverse on the same ODE",
                 fontsize=12)
    ax.set_xlabel("$h_1$"); ax.set_ylabel("$h_2$")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.7, 2.7)

    # --- Right: memory comparison bar chart ---
    ax = axes[1]
    Ls = np.array([10, 50, 100, 500, 1000])
    standard = Ls.astype(float) * 1.0   # arbitrary unit (one activation per step)
    adjoint = np.ones_like(Ls).astype(float)

    width = 0.38
    xs = np.arange(len(Ls))
    ax.bar(xs - width / 2, standard, width, label="Standard backprop  $O(L)$",
           color=RED, alpha=0.85, edgecolor="white")
    ax.bar(xs + width / 2, adjoint, width, label="Adjoint  $O(1)$",
           color=GREEN, alpha=0.9, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L={L}" for L in Ls])
    ax.set_ylabel("Activation memory (relative units, log scale)")
    ax.set_title("Memory cost vs solver steps L", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    for i, (s, a) in enumerate(zip(standard, adjoint)):
        ax.text(i - width / 2, s * 1.15, f"{int(s)}", ha="center",
                fontsize=8, color=RED)
        ax.text(i + width / 2, a * 1.15, "1", ha="center",
                fontsize=8, color=GREEN)

    fig.suptitle(r"Adjoint sensitivity: $\dot a=-a^\top\partial_h f_\theta$,"
                 r"  $\nabla_\theta\mathcal{L}=-\!\int_T^0 a^\top\partial_\theta f_\theta\,dt$",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_adjoint_method.png")


# ---------------------------------------------------------------------------
# Figure 4: FFJORD trace estimation (Hutchinson)
# ---------------------------------------------------------------------------

def fig4_ffjord_trace() -> None:
    print("Figure 4: FFJORD Hutchinson trace estimation")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- Left: variance vs number of Hutchinson samples ---
    ax = axes[0]
    rng = np.random.default_rng(42)
    d = 64
    # random Jacobian with known trace
    A = rng.standard_normal((d, d)) / np.sqrt(d)
    true_tr = float(np.trace(A))

    Ks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    n_trials = 400
    means = []
    stds = []
    for K in Ks:
        ests = []
        for _ in range(n_trials):
            eps = rng.standard_normal((K, d))
            est = np.mean(np.einsum("kd,de,ke->k", eps, A, eps))
            ests.append(est)
        ests = np.array(ests)
        means.append(ests.mean())
        stds.append(ests.std())
    means = np.array(means); stds = np.array(stds)

    ax.fill_between(Ks, means - stds, means + stds,
                    color=PURPLE, alpha=0.25, label=r"$\pm 1\sigma$ over 400 trials")
    ax.plot(Ks, means, "o-", color=PURPLE, lw=2,
            label="Hutchinson mean estimate")
    ax.axhline(true_tr, color=GREEN, lw=2, ls="--",
               label=f"True trace = {true_tr:.2f}")
    ax.plot(Ks, true_tr + stds[0] / np.sqrt(Ks), color=BLUE, lw=1.2, ls=":")
    ax.plot(Ks, true_tr - stds[0] / np.sqrt(Ks), color=BLUE, lw=1.2, ls=":",
            label=r"$1/\sqrt{K}$ envelope")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Hutchinson samples $K$")
    ax.set_ylabel(r"$\widehat{\mathrm{tr}}(\partial f/\partial z)$")
    ax.set_title(f"Variance shrinks as $1/\\sqrt{{K}}$  (d={d})", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # --- Right: cost comparison: full Jacobian vs Hutchinson ---
    ax = axes[1]
    ds = np.array([4, 16, 64, 256, 1024])
    full_cost = ds ** 2
    hutch_cost = ds * 4   # K=4 jvp's, each O(d)

    width = 0.38
    xs = np.arange(len(ds))
    ax.bar(xs - width / 2, full_cost, width,
           label=r"Full Jacobian:  $O(d^2)$ AD calls",
           color=RED, alpha=0.85, edgecolor="white")
    ax.bar(xs + width / 2, hutch_cost, width,
           label=r"Hutchinson ($K{=}4$):  $O(Kd)$",
           color=GREEN, alpha=0.9, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d={d}" for d in ds])
    ax.set_ylabel("Floating-point work (relative, log scale)")
    ax.set_title("Per-step divergence cost vs dimension d", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)

    fig.suptitle(r"FFJORD: $\nabla\!\cdot f=\mathbb{E}_\epsilon[\,\epsilon^\top(\partial f/\partial z)\,\epsilon\,]$,"
                 r"  unbiased and $O(d)$",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_ffjord_trace.png")


# ---------------------------------------------------------------------------
# Figure 5: Flow Matching — paired interpolation paths
# ---------------------------------------------------------------------------

def fig5_flow_matching() -> None:
    print("Figure 5: Flow Matching - paired interpolation paths")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- Left: linear-interpolation paths between paired noise/data ---
    ax = axes[0]
    n = 22
    rng = np.random.default_rng(7)
    z0 = rng.standard_normal((n, 2)) * 1.1
    z1 = crescent_samples(n * 30, rng=rng)
    # pick n random target points
    z1 = z1[rng.choice(z1.shape[0], n, replace=False)]

    ts = np.linspace(0, 1, 40)
    for i in range(n):
        path = np.outer(1 - ts, z0[i]) + np.outer(ts, z1[i])
        ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=1.0, alpha=0.55)

    ax.scatter(z0[:, 0], z0[:, 1], color=GREEN, s=45, zorder=3,
               edgecolor="white", lw=1.0, label=r"$z_0\sim\mathcal{N}(0,I)$")
    ax.scatter(z1[:, 0], z1[:, 1], color=RED, s=45, zorder=3,
               edgecolor="white", lw=1.0, label=r"$z_1\sim p_{\mathrm{data}}$")
    # arrows showing target velocities
    for i in range(0, n, 3):
        ax.annotate("", xy=z1[i], xytext=z0[i],
                    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4,
                                    alpha=0.8))
    ax.set_title(r"Conditional path $z_t=(1-t)z_0+t\,z_1$,"
                 r"  target $u_t^*=z_1-z_0$",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.7, 2.7)
    ax.set_xticks([]); ax.set_yticks([])

    # --- Right: training-loss curves CNF vs Flow Matching (illustrative) ---
    ax = axes[1]
    iters = np.arange(1, 4001)
    # Two qualitative curves: FM converges faster and lower (variance-free target)
    cnf = 3.5 * np.exp(-iters / 1500.0) + 0.45 + 0.06 * RNG.standard_normal(iters.shape)
    fm = 2.8 * np.exp(-iters / 350.0) + 0.18 + 0.03 * RNG.standard_normal(iters.shape)
    cnf = np.maximum(cnf, 0.42 + 0.04 * RNG.standard_normal(iters.shape))
    fm = np.maximum(fm, 0.16 + 0.02 * RNG.standard_normal(iters.shape))

    ax.plot(iters, cnf, color=RED, lw=1.6, alpha=0.85,
            label=r"CNF (NLL via $\int \nabla\!\cdot f\,dt$)")
    ax.plot(iters, fm, color=GREEN, lw=1.6, alpha=0.95,
            label="Flow Matching (regression on $u_t^*$)")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (illustrative)")
    ax.set_title("Flow Matching trains faster and more stably", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    fig.suptitle(r"Flow Matching: $\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{t,z_0,z_1}\,\|v_\theta(z_t,t)-(z_1-z_0)\|^2$,"
                 r"  no divergence at training time",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_flow_matching.png")


# ---------------------------------------------------------------------------
# Figure 6: Time-continuous depth — compare ResNet steps vs adaptive ODE
# ---------------------------------------------------------------------------

def fig6_continuous_depth() -> None:
    print("Figure 6: time-continuous depth visualisation")
    fig, ax = plt.subplots(figsize=(11, 6))

    # Underlying continuous trajectory
    ts = np.linspace(0, 1, 400)
    h_true = np.sin(2 * np.pi * ts) * np.exp(-1.5 * ts) + 0.4 * ts

    ax.plot(ts, h_true, color=BLUE, lw=2.6, label=r"True $h(t)$ (Neural ODE solution)")

    # Coarse Euler: 4 steps
    cs = np.array([0, 0.25, 0.5, 0.75, 1.0])
    h_cs = np.sin(2 * np.pi * cs) * np.exp(-1.5 * cs) + 0.4 * cs
    ax.plot(cs, h_cs, "o--", color=RED, lw=1.6, ms=8,
            label="ResNet, fixed depth $L=4$ (coarse)")

    # Medium: 8 steps
    ms_ = np.linspace(0, 1, 9)
    h_ms = np.sin(2 * np.pi * ms_) * np.exp(-1.5 * ms_) + 0.4 * ms_
    ax.plot(ms_, h_ms, "s--", color=ORANGE, lw=1.4, ms=7, alpha=0.85,
            label="ResNet, fixed depth $L=8$")

    # Adaptive: nodes denser where dh/dt is large
    speed = np.abs(np.gradient(h_true, ts))
    cumulative = np.cumsum(speed); cumulative /= cumulative[-1]
    targets = np.linspace(0, 1, 12)
    adaptive_idx = np.searchsorted(cumulative, targets)
    adaptive_idx = np.clip(adaptive_idx, 0, len(ts) - 1)
    ax.plot(ts[adaptive_idx], h_true[adaptive_idx], "D", color=PURPLE, ms=8,
            label="Adaptive solver checkpoints (dopri5)")

    # Highlight error patches between coarse Euler segments and true
    for i in range(len(cs) - 1):
        t0, t1 = cs[i], cs[i + 1]
        mask = (ts >= t0) & (ts <= t1)
        seg_t = ts[mask]
        seg_h = h_true[mask]
        # coarse linear interp
        line = h_cs[i] + (h_cs[i + 1] - h_cs[i]) * (seg_t - t0) / (t1 - t0)
        ax.fill_between(seg_t, line, seg_h, color=RED, alpha=0.10)

    ax.set_xlabel(r"Continuous depth $t\in[0,1]$")
    ax.set_ylabel(r"State $h(t)$")
    ax.set_title("Discrete depth (ResNet) vs continuous depth (Neural ODE)",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig6_continuous_depth.png")


# ---------------------------------------------------------------------------
# Figure 7: Density estimation on 2D toy data (CNF-style)
# ---------------------------------------------------------------------------

def fig7_density_estimation() -> None:
    print("Figure 7: density estimation on 2D toy data")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    # Target: two interleaving moons
    n = 4000
    target = crescent_samples(n)

    # --- (a) Real samples ---
    ax = axes[0]
    ax.scatter(target[:, 0], target[:, 1], s=4, color=BLUE, alpha=0.55)
    ax.set_title("(a) Target samples (two moons)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.0, 2.0)

    # --- (b) Empirical density ---
    grid_x, grid_y = np.mgrid[-2.5:2.5:160j, -2.0:2.0:160j]
    pos = np.vstack([grid_x.ravel(), grid_y.ravel()])
    kde_emp = gaussian_kde(target.T, bw_method=0.18)
    dens_emp = kde_emp(pos).reshape(grid_x.shape)

    ax = axes[1]
    ax.imshow(dens_emp.T, extent=[-2.5, 2.5, -2.0, 2.0], origin="lower",
              cmap="Blues", aspect="auto")
    ax.contour(grid_x, grid_y, dens_emp, levels=8,
               colors="white", linewidths=0.5, alpha=0.7)
    ax.set_title("(b) Empirical density", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # --- (c) CNF-fitted density: simulate by transporting Gaussian via FM map ---
    rng = np.random.default_rng(123)
    z0 = rng.standard_normal((n, 2))
    # use the empirical samples as targets — paired by angle for an OT-like map
    pair_target = target[np.argsort(np.arctan2(target[:, 1], target[:, 0]))]
    z0 = z0[np.argsort(np.arctan2(z0[:, 1], z0[:, 0]))]
    cnf_samples = linear_flow(z0, pair_target, 1.0)
    kde_cnf = gaussian_kde(cnf_samples.T, bw_method=0.20)
    dens_cnf = kde_cnf(pos).reshape(grid_x.shape)

    ax = axes[2]
    ax.imshow(dens_cnf.T, extent=[-2.5, 2.5, -2.0, 2.0], origin="lower",
              cmap="Purples", aspect="auto")
    ax.contour(grid_x, grid_y, dens_cnf, levels=8,
               colors="white", linewidths=0.5, alpha=0.7)
    ax.set_title("(c) CNF-fitted density", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # --- (d) CNF generated samples + a few ODE trajectories ---
    ax = axes[3]
    gen_samples = cnf_samples
    ax.scatter(gen_samples[:, 0], gen_samples[:, 1], s=4,
               color=PURPLE, alpha=0.55, label="Generated")
    # show a few full trajectories
    idx_show = rng.choice(n, 12, replace=False)
    ts_show = np.linspace(0, 1, 30)
    for i in idx_show:
        path = np.outer(1 - ts_show, z0[i]) + np.outer(ts_show, pair_target[i])
        ax.plot(path[:, 0], path[:, 1], color=GREEN, lw=0.8, alpha=0.7)
    ax.scatter(z0[idx_show, 0], z0[idx_show, 1], color=GREEN, s=30,
               edgecolor="white", lw=0.8, zorder=3)
    ax.set_title("(d) Generated samples + ODE flow", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.0, 2.0)

    fig.suptitle("Density estimation on 2D toy data: target / empirical KDE / CNF fit / generative flow",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    save(fig, "fig7_density_estimation.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Output dirs:\n  {EN_DIR}\n  {ZH_DIR}\n")
    fig1_density_transformation()
    fig2_neural_ode_vs_resnet()
    fig3_adjoint_method()
    fig4_ffjord_trace()
    fig5_flow_matching()
    fig6_continuous_depth()
    fig7_density_estimation()
    print("\nDone. All figures written to EN and ZH asset folders.")


if __name__ == "__main__":
    main()
