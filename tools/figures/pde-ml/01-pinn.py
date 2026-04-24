#!/usr/bin/env python3
"""
PDE+ML Chapter 01 - Physics-Informed Neural Networks
Figure generation script.

Generates 7 figures and writes them to BOTH the EN and ZH asset folders:
  - source/_posts/en/pde-ml/01-Physics-Informed-Neural-Networks/
  - source/_posts/zh/pde-ml/01-物理信息神经网络/

Run from anywhere:
    python 01-pinn.py

Style: seaborn-v0_8-whitegrid, dpi=150
Palette: blue #2563eb, purple #7c3aed, green #10b981, red #ef4444

The script does NOT require PyTorch.  Where PINN behaviour is illustrated
we use surrogate solutions: closed-form references (heat, Burgers via
Cole-Hopf), or stylised loss/error curves derived from the published
literature (Wang & Perdikaris 2021, Krishnapriyan et al. 2021).  This
keeps the figures cheap and reproducible, while remaining faithful to
the qualitative behaviour reported for true PINNs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
RED = "#ef4444"
GRAY = "#6b7280"
ORANGE = "#f59e0b"
LIGHT = "#e5e7eb"

DPI = 150

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "pde-ml" / "01-Physics-Informed-Neural-Networks"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "pde-ml" / "01-物理信息神经网络"

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
# Figure 1: PINN architecture
# ---------------------------------------------------------------------------

def fig1_architecture() -> None:
    """Schematic of a PINN: MLP -> u_hat -> autodiff -> physics loss."""
    print("Figure 1: PINN architecture")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Inputs (x, t)
    ax.add_patch(mpatches.Circle((0.6, 4.0), 0.32, color=BLUE, ec="black", lw=1.2))
    ax.add_patch(mpatches.Circle((0.6, 2.0), 0.32, color=BLUE, ec="black", lw=1.2))
    ax.text(0.6, 4.0, r"$x$", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(0.6, 2.0, r"$t$", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(0.6, 5.2, "inputs", ha="center", fontsize=10, color=GRAY)

    # Hidden layers
    layer_x = [2.4, 3.6, 4.8]
    n_nodes = 5
    for lx in layer_x:
        ys = np.linspace(1.0, 5.0, n_nodes)
        for y in ys:
            ax.add_patch(mpatches.Circle((lx, y), 0.22, facecolor="white", ec=PURPLE, lw=1.2))
        # Connections from previous layer
        if lx == layer_x[0]:
            prev_pts = [(0.6, 4.0), (0.6, 2.0)]
        else:
            prev_pts = [(layer_x[layer_x.index(lx) - 1], y) for y in np.linspace(1.0, 5.0, n_nodes)]
        for px, py in prev_pts:
            for y in ys:
                ax.plot([px, lx], [py, y], color=PURPLE, lw=0.4, alpha=0.35)
    ax.text(3.6, 5.6, "MLP   $u_\\theta(x,t)$", ha="center", fontsize=11, color=PURPLE)

    # Output
    ax.add_patch(mpatches.Circle((6.2, 3.0), 0.34, color=GREEN, ec="black", lw=1.2))
    ax.text(6.2, 3.0, r"$\hat u$", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    for y in np.linspace(1.0, 5.0, n_nodes):
        ax.plot([layer_x[-1], 6.2], [y, 3.0], color=PURPLE, lw=0.4, alpha=0.35)

    # Autodiff box
    auto = FancyBboxPatch((6.9, 2.2), 1.6, 1.6,
                          boxstyle="round,pad=0.05", linewidth=1.4,
                          edgecolor=ORANGE, facecolor="#fff7ed")
    ax.add_patch(auto)
    ax.text(7.7, 3.45, "autodiff", ha="center", fontsize=10, color=ORANGE, fontweight="bold")
    ax.text(7.7, 2.95, r"$\partial_t \hat u$", ha="center", fontsize=11)
    ax.text(7.7, 2.55, r"$\partial_x^2 \hat u$", ha="center", fontsize=11)

    # Loss boxes
    loss_specs = [
        (9.3, 4.6, r"PDE residual" + "\n" + r"$\mathcal{L}_r=\|\partial_t\hat u-\nu\partial_x^2\hat u\|^2$", BLUE),
        (9.3, 3.0, r"Boundary" + "\n" + r"$\mathcal{L}_b=\|\hat u-g\|^2_{\partial\Omega}$", PURPLE),
        (9.3, 1.4, r"Initial / data" + "\n" + r"$\mathcal{L}_i=\|\hat u-u_0\|^2$", RED),
    ]
    for cx, cy, txt, color in loss_specs:
        ax.add_patch(FancyBboxPatch((cx - 0.95, cy - 0.55), 1.9, 1.1,
                                    boxstyle="round,pad=0.05", linewidth=1.3,
                                    edgecolor=color, facecolor="white"))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8.5)
        ax.add_patch(FancyArrowPatch((8.5, 3.0), (cx - 0.95, cy),
                                     arrowstyle="->", mutation_scale=12, color=color, lw=1.0))

    ax.add_patch(FancyArrowPatch((6.55, 3.0), (6.95, 3.0),
                                 arrowstyle="->", mutation_scale=14, color="black", lw=1.2))

    ax.text(5.5, 0.45, r"total loss   $\mathcal{L}=\lambda_r\mathcal{L}_r+\lambda_b\mathcal{L}_b+\lambda_i\mathcal{L}_i$",
            ha="center", fontsize=11, color="black",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT, edgecolor="none"))

    fig.suptitle("PINN architecture: MLP + automatic differentiation + physics-informed loss",
                 fontsize=12, y=0.99)
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2: Loss decomposition during training
# ---------------------------------------------------------------------------

def fig2_loss_decomposition() -> None:
    """Loss components vs iteration; emphasises imbalance + balanced runs."""
    print("Figure 2: loss decomposition")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    rng = np.random.default_rng(0)
    it = np.arange(1, 12001)

    # Imbalanced run: residual decays slowly, boundary stalls high
    L_r_im = 1.5 * (it ** -0.42) + 1e-3
    L_b_im = 0.6 * np.exp(-it / 9000) + 1e-2
    L_i_im = 0.9 * (it ** -0.55) + 5e-4
    for arr in (L_r_im, L_b_im, L_i_im):
        arr *= np.exp(0.05 * rng.standard_normal(it.size))

    ax = axes[0]
    ax.loglog(it, L_r_im, color=BLUE, lw=1.4, label=r"$\mathcal{L}_r$ (PDE residual)")
    ax.loglog(it, L_b_im, color=PURPLE, lw=1.4, label=r"$\mathcal{L}_b$ (boundary)")
    ax.loglog(it, L_i_im, color=RED, lw=1.4, label=r"$\mathcal{L}_i$ (initial / data)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Naive equal weights:  boundary loss stalls", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(1e-4, 5)

    # Balanced run: all curves decay together
    L_r_bal = 1.5 * (it ** -0.55) + 5e-5
    L_b_bal = 1.0 * (it ** -0.60) + 5e-5
    L_i_bal = 0.9 * (it ** -0.62) + 3e-5
    for arr in (L_r_bal, L_b_bal, L_i_bal):
        arr *= np.exp(0.05 * rng.standard_normal(it.size))

    ax = axes[1]
    ax.loglog(it, L_r_bal, color=BLUE, lw=1.4, label=r"$\mathcal{L}_r$")
    ax.loglog(it, L_b_bal, color=PURPLE, lw=1.4, label=r"$\mathcal{L}_b$")
    ax.loglog(it, L_i_bal, color=RED, lw=1.4, label=r"$\mathcal{L}_i$")
    ax.set_xlabel("iteration")
    ax.set_title("Adaptive weighting (NTK-balanced)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(1e-4, 5)

    fig.suptitle("Loss decomposition: data + PDE residual + boundary loss",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_loss_decomposition.png")


# ---------------------------------------------------------------------------
# Figure 3: Burgers equation — predicted vs exact (Cole-Hopf reference)
# ---------------------------------------------------------------------------

def burgers_cole_hopf(x: np.ndarray, t: float, nu: float) -> np.ndarray:
    """Exact Burgers solution for u(x,0)= -sin(pi x), x in [-1,1]."""
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        def num_int(eta):
            return np.sin(np.pi * (xi - eta)) * np.exp(
                -np.cos(np.pi * (xi - eta)) / (2 * np.pi * nu)
            ) * np.exp(-eta ** 2 / (4 * nu * t))

        def den_int(eta):
            return np.exp(
                -np.cos(np.pi * (xi - eta)) / (2 * np.pi * nu)
            ) * np.exp(-eta ** 2 / (4 * nu * t))

        # Integrate over a wide enough window
        L = 8.0 * np.sqrt(nu * t) + 2.0
        num, _ = quad(num_int, -L, L, limit=200)
        den, _ = quad(den_int, -L, L, limit=200)
        out[i] = -num / den
    return out


def fig3_burgers_pred_vs_exact() -> None:
    """PINN predicted u(x,t) vs Cole-Hopf reference for Burgers nu=0.01/pi."""
    print("Figure 3: Burgers prediction vs exact")
    nu = 0.01 / np.pi
    x = np.linspace(-1.0, 1.0, 161)
    times = [0.25, 0.50, 0.75]

    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0),
                             gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})

    # Pseudocolor of full field
    Nx, Nt = 161, 81
    xs = np.linspace(-1.0, 1.0, Nx)
    ts = np.linspace(0.05, 0.99, Nt)
    U = np.zeros((Nt, Nx))
    for j, tt in enumerate(ts):
        U[j, :] = burgers_cole_hopf(xs, float(tt), nu)
    pcm = axes[0].pcolormesh(xs, ts, U, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$t$")
    axes[0].set_title(r"Reference $u(x,t)$,  $\nu=0.01/\pi$", fontsize=10)
    fig.colorbar(pcm, ax=axes[0], pad=0.02)
    for tt in times:
        axes[0].axhline(tt, color="black", lw=0.6, ls="--", alpha=0.6)

    for k, tt in enumerate(times):
        u_exact = burgers_cole_hopf(x, tt, nu)
        # Surrogate "PINN" prediction = exact + small structured noise + slight shock smearing
        smear = np.where(np.abs(u_exact) > 0.6, 0.04, 0.0)
        u_pred = u_exact + 0.012 * rng.standard_normal(x.size) + smear * np.sign(-u_exact) * 0.5
        ax = axes[k + 1]
        ax.plot(x, u_exact, color="black", lw=1.6, label="exact")
        ax.plot(x, u_pred, color=BLUE, lw=1.2, ls="--", label="PINN")
        ax.set_xlabel(r"$x$")
        ax.set_title(rf"$t={tt}$", fontsize=10)
        ax.set_ylim(-1.15, 1.15)
        if k == 0:
            ax.set_ylabel(r"$u$")
            ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(r"PINN solving Burgers' equation $u_t+uu_x=\nu u_{xx}$: predicted vs exact",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    save(fig, "fig3_burgers_pred_vs_exact.png")


# ---------------------------------------------------------------------------
# Figure 4: convergence — with vs without PDE loss
# ---------------------------------------------------------------------------

def fig4_convergence_with_without_pde() -> None:
    """Test error vs iteration; PDE loss yields better generalisation with few labels."""
    print("Figure 4: convergence with vs without PDE loss")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    rng = np.random.default_rng(2)
    it = np.arange(1, 8001)

    err_data = 0.40 * (it ** -0.18) + 0.04
    err_pinn = 0.40 * (it ** -0.55) + 0.0035
    err_data *= np.exp(0.04 * rng.standard_normal(it.size))
    err_pinn *= np.exp(0.05 * rng.standard_normal(it.size))

    ax.loglog(it, err_data, color=RED, lw=1.6,
              label="data loss only (50 noisy labels)")
    ax.loglog(it, err_pinn, color=BLUE, lw=1.6,
              label="data + PDE residual (PINN)")

    ax.fill_between(it, err_pinn * 0.7, err_pinn * 1.3, color=BLUE, alpha=0.12)
    ax.fill_between(it, err_data * 0.7, err_data * 1.3, color=RED, alpha=0.12)

    ax.set_xlabel("iteration")
    ax.set_ylabel(r"relative $L^2$ test error")
    ax.set_title("Adding the PDE residual breaks the data bottleneck", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(2e-3, 1.0)

    ax.annotate("plateau:\nlimited by label noise", xy=(4000, err_data[3999]),
                xytext=(900, 0.18), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.annotate("physics keeps\ndriving error down", xy=(6000, err_pinn[5999]),
                xytext=(150, 0.012), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

    fig.tight_layout()
    save(fig, "fig4_convergence.png")


# ---------------------------------------------------------------------------
# Figure 5: inverse problem — parameter discovery
# ---------------------------------------------------------------------------

def fig5_inverse_parameter_discovery() -> None:
    """Recover diffusivity nu* from sparse noisy data; show convergence."""
    print("Figure 5: inverse problem — parameter discovery")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    nu_true = 0.10
    rng = np.random.default_rng(3)

    # Left: synthetic noisy observations on top of analytic heat solution
    x = np.linspace(0, 1, 200)
    t_obs = 0.15
    u_clean = np.exp(-nu_true * np.pi ** 2 * t_obs) * np.sin(np.pi * x)
    x_obs = rng.uniform(0, 1, 30)
    u_obs = (np.exp(-nu_true * np.pi ** 2 * t_obs) * np.sin(np.pi * x_obs)
             + 0.025 * rng.standard_normal(30))

    ax = axes[0]
    ax.plot(x, u_clean, color="black", lw=1.5, label=rf"true $u(x,t={t_obs})$")
    ax.scatter(x_obs, u_obs, color=RED, s=28, zorder=3, label="noisy observations (n=30)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.set_title(rf"Sparse observations of heat eq. ($\nu^\star={nu_true}$)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    # Right: parameter trajectory
    it = np.arange(0, 6001)
    nu_traj = nu_true + (0.45 - nu_true) * np.exp(-it / 800.0)
    nu_traj += 0.005 * np.exp(-it / 2500.0) * rng.standard_normal(it.size)

    ax = axes[1]
    ax.plot(it, nu_traj, color=BLUE, lw=1.5, label=r"learned $\nu(\theta)$")
    ax.axhline(nu_true, color=GREEN, lw=1.5, ls="--", label=rf"true $\nu^\star={nu_true}$")
    ax.fill_between(it, nu_traj - 0.005, nu_traj + 0.005, color=BLUE, alpha=0.15)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\nu$")
    ax.set_title("Parameter trajectory during joint training", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.0, 0.5)

    final_err = abs(nu_traj[-1] - nu_true) / nu_true * 100
    ax.text(0.55, 0.20, rf"final relative error: {final_err:.2f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, edgecolor="none"))

    fig.suptitle("Inverse problem: recover hidden PDE parameter from sparse data",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_inverse_problem.png")


# ---------------------------------------------------------------------------
# Figure 6: failure modes (gradient pathology + multi-scale spectral bias)
# ---------------------------------------------------------------------------

def fig6_failure_modes() -> None:
    """Two known failure modes: gradient imbalance, and spectral bias."""
    print("Figure 6: failure modes")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: gradient norm imbalance per layer (Wang & Perdikaris 2021 style)
    layers = np.arange(1, 9)
    grad_r = 1.0 * 1.7 ** (layers - 1)        # PDE-residual grads explode
    grad_b = 0.05 * np.ones_like(layers)      # boundary grads tiny
    width = 0.35
    ax = axes[0]
    ax.bar(layers - width / 2, grad_r, width=width, color=BLUE,
           label=r"$\|\nabla_\theta\mathcal{L}_r\|$")
    ax.bar(layers + width / 2, grad_b, width=width, color=RED,
           label=r"$\|\nabla_\theta\mathcal{L}_b\|$")
    ax.set_yscale("log")
    ax.set_xlabel("layer index")
    ax.set_ylabel("gradient norm  (log)")
    ax.set_title("Gradient pathology: residual grads dominate boundary grads",
                 fontsize=10.5)
    ax.legend(loc="upper left", fontsize=9)

    # Right: spectral bias on a multi-scale target
    x = np.linspace(0, 1, 600)
    target = np.sin(np.pi * x) + 0.5 * np.sin(8 * np.pi * x) + 0.3 * np.sin(20 * np.pi * x)
    learned = (1.00 * np.sin(np.pi * x)
               + 0.30 * np.sin(8 * np.pi * x)
               + 0.05 * np.sin(20 * np.pi * x))
    ax = axes[1]
    ax.plot(x, target, color="black", lw=1.6, label="target")
    ax.plot(x, learned, color=BLUE, lw=1.4, ls="--", label="PINN after 20k iter")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.set_title("Spectral bias: high-frequency content learned last",
                 fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("PINN failure modes: gradient pathology & multi-scale spectral bias",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_failure_modes.png")


# ---------------------------------------------------------------------------
# Figure 7: comparison — PINN vs FEM vs Neural Operator
# ---------------------------------------------------------------------------

def fig7_pinn_vs_fem_vs_neural_operator() -> None:
    """Qualitative comparison across four axes."""
    print("Figure 7: PINN vs FEM vs Neural Operator")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: radar chart (manual, polar)
    categories = ["mesh-free", "high-d", "complex\ngeometry",
                  "speed (1 PDE)", "generalises\nto new PDEs", "accuracy\nguarantees"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    scores = {
        "FEM":              [1, 1, 4, 4, 1, 5],
        "PINN":             [5, 4, 5, 2, 2, 2],
        "Neural Operator":  [5, 4, 4, 5, 5, 2],
    }
    colors = {"FEM": GRAY, "PINN": BLUE, "Neural Operator": PURPLE}

    ax = axes[0]
    ax.remove()
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    for label, vals in scores.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=colors[label], lw=1.8, label=label)
        ax.fill(angles, v, color=colors[label], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color=GRAY)
    ax.set_ylim(0, 5.5)
    ax.set_title("Capability profile  (5 = best)", fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=9)

    # Right: cost-vs-accuracy scatter
    ax = axes[1]
    methods = [
        ("FEM (high res.)",          1.0e3, 1e-5, GRAY),
        ("FEM (low res.)",           5.0e1, 1e-2, GRAY),
        ("PINN (vanilla)",           1.5e3, 5e-3, BLUE),
        ("PINN (NTK + adaptive)",    3.0e3, 5e-4, BLUE),
        ("DeepONet  (per query)",    1.0e0, 1e-2, PURPLE),
        ("FNO  (per query)",         1.0e0, 5e-3, PURPLE),
    ]
    for name, t, err, c in methods:
        ax.scatter(t, err, color=c, s=80, edgecolor="black", lw=0.6, zorder=3)
        ax.annotate(name, (t, err), textcoords="offset points",
                    xytext=(8, 6), fontsize=8.5, color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("inference / solve time  (s, log)")
    ax.set_ylabel(r"relative $L^2$ error (log)")
    ax.set_title("Accuracy-cost trade-off", fontsize=11)
    ax.set_xlim(0.3, 1e4)
    ax.set_ylim(1e-6, 1e-1)

    fig.suptitle("PINN vs FEM vs Neural Operator", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_pinn_vs_fem_vs_no.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_architecture()
    fig2_loss_decomposition()
    fig3_burgers_pred_vs_exact()
    fig4_convergence_with_without_pde()
    fig5_inverse_parameter_discovery()
    fig6_failure_modes()
    fig7_pinn_vs_fem_vs_neural_operator()
    print("All figures generated.")


if __name__ == "__main__":
    main()
