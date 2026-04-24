#!/usr/bin/env python3
"""
PDE+ML Chapter 03 - Variational Principles and Optimization
Figure generation script.

Generates 7 figures and writes them to BOTH the EN and ZH asset folders:
  - source/_posts/en/pde-ml/03-Variational-Principles/
  - source/_posts/zh/pde-ml/03-变分原理与优化/

Figures:
  fig1_functional_minimization     Brachistochrone / least action: a family
                                   of candidate paths and the minimizing one
  fig2_euler_lagrange              Geometric construction of Euler-Lagrange
                                   from a perturbation y + eps eta
  fig3_wasserstein_gradient_flow   JKO trajectory of a 1-D density toward a
                                   target: a Wasserstein gradient flow
  fig4_mean_field_limit            Two-layer NN training: empirical particle
                                   distribution converging to the mean-field
                                   density as width m grows
  fig5_energy_landscape_3d         3-D non-convex energy landscape with the
                                   gradient-flow trajectory
  fig6_variational_autoencoder     ELBO decomposition (reconstruction loss
                                   vs KL regularization) along training
  fig7_hamiltonian_flow            Symplectic vs dissipative flows on a
                                   pendulum / harmonic oscillator phase plane

Run from anywhere:
    python 03-variational.py

Style: seaborn-v0_8-whitegrid, dpi=150
Palette: blue #2563eb, purple #7c3aed, green #10b981, red #ef4444
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D)
from scipy.integrate import solve_ivp

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
RNG = np.random.default_rng(42)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "pde-ml" / "03-Variational-Principles"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "pde-ml" / "03-变分原理与优化"

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
# Figure 1: Functional minimization - the brachistochrone / least action
# ---------------------------------------------------------------------------

def fig1_functional_minimization() -> None:
    """A family of candidate descent curves with the cycloid as minimizer.

    We plot several admissible curves between A=(0,0) and B=(pi, 2),
    compute the descent time T[y] for each (frictionless slide under
    gravity), and highlight the cycloid as the minimizer.
    """
    print("Figure 1: functional minimization (brachistochrone)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Endpoints chosen so that the cycloid through them has parameter R=1.
    A = (0.0, 0.0)
    B = (np.pi, 2.0)

    # -- Cycloid (true minimizer): x = R(theta - sin theta), y = R(1 - cos theta)
    R = 1.0
    theta = np.linspace(0.0, np.pi, 400)
    x_cycl = R * (theta - np.sin(theta))
    y_cycl = R * (1.0 - np.cos(theta))

    def descent_time(x: np.ndarray, y: np.ndarray) -> float:
        """Approximate frictionless slide time under gravity, g = 1."""
        # T = integral sqrt((1 + (dy/dx)^2) / (2 g y)) dx, with y > 0.
        # Use trapezoid; protect against y -> 0 at the start.
        eps = 1e-6
        y_safe = np.maximum(y, eps)
        # Re-parameterise by arc length to avoid dy/dx blow-ups: ds/sqrt(2 g y)
        ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        y_mid = 0.5 * (y_safe[:-1] + y_safe[1:])
        return float(np.sum(ds / np.sqrt(2.0 * y_mid)))

    # -- Candidate curves between A and B, sorted by descent time
    x_grid = np.linspace(A[0], B[0], 400)

    # Straight line
    y_line = np.linspace(A[1], B[1], 400)

    # A "too flat" parabola: y = (B[1]/B[0]^2) x^2
    y_par_flat = (B[1] / B[0] ** 2) * x_grid ** 2

    # A "too steep" curve: y = B[1] * (x / B[0]) ** 0.4
    y_steep = B[1] * (x_grid / B[0]) ** 0.4

    # A circular arc bulging downward through A and B
    # Centre on perpendicular bisector of AB; pick a radius slightly
    # larger than half the chord length.
    chord_mid = (0.5 * (A[0] + B[0]), 0.5 * (A[1] + B[1]))
    chord_len = np.hypot(B[0] - A[0], B[1] - A[1])
    arc_R = 0.6 * chord_len
    # Perpendicular direction pointing "down" (positive y here)
    nx = -(B[1] - A[1]) / chord_len
    ny = (B[0] - A[0]) / chord_len
    if ny < 0:
        nx, ny = -nx, -ny
    sag = np.sqrt(max(arc_R ** 2 - (chord_len / 2.0) ** 2, 0.0))
    centre = (chord_mid[0] + sag * nx, chord_mid[1] + sag * ny)
    angles_A = np.arctan2(A[1] - centre[1], A[0] - centre[0])
    angles_B = np.arctan2(B[1] - centre[1], B[0] - centre[0])
    # Take the short arc on the "bulge down" side
    a_arc = np.linspace(angles_A, angles_B, 400)
    x_arc = centre[0] + arc_R * np.cos(a_arc)
    y_arc = centre[1] + arc_R * np.sin(a_arc)

    candidates = [
        ("Straight line", x_grid, y_line, GRAY),
        ("Steep then flat", x_grid, y_steep, ORANGE),
        ("Shallow parabola", x_grid, y_par_flat, GREEN),
        ("Circular arc", x_arc, y_arc, PURPLE),
        ("Cycloid (minimizer)", x_cycl, y_cycl, RED),
    ]

    # ---- Left panel: physical curves (y axis flipped, gravity downward)
    ax = axes[0]
    for name, x, y, c in candidates:
        T = descent_time(x, y)
        lw = 3.0 if "Cycloid" in name else 1.7
        alpha = 1.0 if "Cycloid" in name else 0.9
        ax.plot(x, y, color=c, lw=lw, alpha=alpha,
                label=f"{name}  (T = {T:.3f})")
    ax.scatter([A[0], B[0]], [A[1], B[1]], color="black", zorder=6, s=70)
    ax.annotate("A", A, textcoords="offset points", xytext=(-12, -8),
                fontsize=13, fontweight="bold")
    ax.annotate("B", B, textcoords="offset points", xytext=(8, 0),
                fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$  (gravity $\\downarrow$)", fontsize=12)
    ax.set_title("Candidate descent paths", fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # ---- Right panel: bar chart of descent times
    ax = axes[1]
    names = [c[0] for c in candidates]
    times = [descent_time(c[1], c[2]) for c in candidates]
    colors = [c[3] for c in candidates]
    bars = ax.barh(names, times, color=colors, alpha=0.85, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(t + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Descent time $T[y]$  (smaller is better)", fontsize=12)
    ax.set_title("Functional values:  $T[y] = \\int \\sqrt{1+y'^2}/\\sqrt{2gy}\\,dx$",
                 fontsize=12)
    ax.set_xlim(0, max(times) * 1.18)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Functional minimization: the path of least time",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_functional_minimization.png")


# ---------------------------------------------------------------------------
# Figure 2: Euler-Lagrange derivation (perturbation picture)
# ---------------------------------------------------------------------------

def fig2_euler_lagrange() -> None:
    """Geometric picture behind the Euler-Lagrange derivation.

    Left panel: extremal y(x) plus several perturbed competitors
                y(x) + eps eta(x), with the variation eta vanishing
                at the boundary.
    Right panel: J(eps) versus eps along one perturbation, showing
                 J'(0) = 0 (extremum condition) for the true extremal.
    """
    print("Figure 2: Euler-Lagrange perturbation picture")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    a, b = 0.0, 1.0
    x = np.linspace(a, b, 400)

    # Extremal: choose y(x) = sin(pi x) (extremises a Dirichlet-type J on [0,1])
    y0 = np.sin(np.pi * x)
    eta = np.sin(2.0 * np.pi * x) * (x * (1.0 - x))  # vanishes at endpoints

    ax = axes[0]
    eps_values = [-0.6, -0.3, 0.0, 0.3, 0.6]
    cmap = plt.get_cmap("coolwarm")
    for i, eps in enumerate(eps_values):
        y = y0 + eps * eta
        if eps == 0.0:
            ax.plot(x, y, color=RED, lw=3.0, label=r"extremal $y(x)$", zorder=5)
        else:
            color = cmap(0.5 + 0.5 * eps / max(abs(min(eps_values)), abs(max(eps_values))))
            ax.plot(x, y, color=color, lw=1.4, alpha=0.85,
                    label=fr"$y + {eps:+.1f}\eta$")

    # Perturbation eta on a secondary axis-style overlay
    ax.fill_between(x, y0, y0 + 0.6 * eta, color=PURPLE, alpha=0.10,
                    label=r"variation envelope")
    # Mark boundary values
    ax.scatter([a, b], [0, 0], color="black", zorder=6, s=60)
    ax.annotate(r"$y(a)$ fixed", (a, 0), textcoords="offset points",
                xytext=(6, -16), fontsize=11)
    ax.annotate(r"$y(b)$ fixed", (b, 0), textcoords="offset points",
                xytext=(-78, -16), fontsize=11)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title(r"$y(x) + \varepsilon\,\eta(x),\ \ \eta(a)=\eta(b)=0$",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

    # ---- Right panel: J(eps) for J[y] = int (1/2) (y')^2 dx (Dirichlet energy)
    ax = axes[1]

    def J_of(y: np.ndarray) -> float:
        dy = np.gradient(y, x)
        return 0.5 * np.trapz(dy ** 2, x)

    eps_grid = np.linspace(-1.0, 1.0, 121)
    J_grid = np.array([J_of(y0 + e * eta) for e in eps_grid])
    J0 = J_of(y0)
    ax.plot(eps_grid, J_grid, color=BLUE, lw=2.4, label=r"$J(\varepsilon)$")
    ax.axhline(J0, color=GRAY, ls="--", alpha=0.6)
    ax.axvline(0.0, color=GRAY, ls="--", alpha=0.6)
    ax.scatter([0.0], [J0], color=RED, s=90, zorder=5,
               label=r"extremum: $J'(0) = 0$")

    # Tangent line at eps = 0 (slope ~ 0)
    slope = (J_grid[61] - J_grid[59]) / (eps_grid[61] - eps_grid[59])
    tangent = J0 + slope * eps_grid
    ax.plot(eps_grid, tangent, color=RED, ls=":", lw=1.6, alpha=0.8,
            label=fr"tangent at $\varepsilon=0$  (slope $\approx {slope:+.2e}$)")

    ax.set_xlabel(r"$\varepsilon$", fontsize=12)
    ax.set_ylabel(r"$J[y + \varepsilon\eta]$", fontsize=12)
    ax.set_title(r"First variation vanishes  $\Rightarrow$  Euler-Lagrange",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper center")

    fig.suptitle("Euler-Lagrange equation: from a perturbation argument",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_euler_lagrange.png")


# ---------------------------------------------------------------------------
# Figure 3: Wasserstein gradient flow (JKO scheme on a 1-D density)
# ---------------------------------------------------------------------------

def fig3_wasserstein_gradient_flow() -> None:
    """Wasserstein gradient flow of a 1-D density toward a target Gaussian.

    We simulate the Fokker-Planck equation
        d_t rho = div( rho grad V ) + Delta rho
    with V(x) = (x - mu*)^2 / 2, whose stationary solution is the
    Gaussian N(mu*, 1).  The PDE is precisely the Wasserstein gradient
    flow of the free energy
        F[rho] = int V rho dx + int rho log rho dx.
    We then plot snapshots of rho(t, .) and the trajectory of the mean.
    """
    print("Figure 3: Wasserstein gradient flow (Fokker-Planck)")
    fig = plt.figure(figsize=(13.5, 5.5))
    ax_left = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)

    # 1-D Fokker-Planck on a periodic-free domain via finite differences
    L = 12.0
    N = 400
    x = np.linspace(-L / 2, L / 2, N)
    dx = x[1] - x[0]
    mu_star = 1.5
    V = 0.5 * (x - mu_star) ** 2
    dV = (x - mu_star)

    # Initial density: a sharp Gaussian far from the target
    rho = np.exp(-(x + 2.5) ** 2 / (2 * 0.25 ** 2))
    rho /= np.trapz(rho, x)

    dt = 1.5e-3
    n_steps = 5000
    snapshot_times = [0.0, 0.5, 1.5, 4.0, 7.5]
    snapshots = []
    means = []
    times_recorded = []

    def laplacian(u: np.ndarray) -> np.ndarray:
        out = np.zeros_like(u)
        out[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        return out

    def divergence(j: np.ndarray) -> np.ndarray:
        out = np.zeros_like(j)
        out[1:-1] = (j[2:] - j[:-2]) / (2 * dx)
        return out

    t = 0.0
    next_snap = 0
    if next_snap < len(snapshot_times):
        snapshots.append(rho.copy())
        next_snap += 1
    means.append(float(np.trapz(x * rho, x)))
    times_recorded.append(t)

    for step in range(n_steps):
        flux = rho * dV          # rho * grad V
        rhs = divergence(flux) + laplacian(rho)
        rho = rho + dt * rhs
        rho = np.clip(rho, 0.0, None)
        rho /= np.trapz(rho, x)
        t += dt
        if next_snap < len(snapshot_times) and t >= snapshot_times[next_snap]:
            snapshots.append(rho.copy())
            next_snap += 1
        if step % 25 == 0:
            means.append(float(np.trapz(x * rho, x)))
            times_recorded.append(t)

    # Stationary Gaussian for reference
    rho_star = np.exp(-(x - mu_star) ** 2 / 2.0) / np.sqrt(2 * np.pi)

    # ---- Left: density snapshots
    cmap = cm.get_cmap("viridis")
    for i, snap in enumerate(snapshots):
        c = cmap(i / max(1, len(snapshots) - 1))
        ax_left.plot(x, snap, color=c, lw=2.2,
                     label=fr"$t = {snapshot_times[i]:.1f}$")
    ax_left.plot(x, rho_star, color=RED, lw=2.0, ls="--",
                 label=r"target $\rho_\ast$")
    ax_left.axvline(mu_star, color=RED, alpha=0.25, ls=":")
    ax_left.set_xlabel("$x$", fontsize=12)
    ax_left.set_ylabel(r"density $\rho(t, x)$", fontsize=12)
    ax_left.set_title(r"JKO trajectory in $\mathcal{P}_2(\mathbb{R})$",
                      fontsize=13)
    ax_left.set_xlim(-5, 5)
    ax_left.legend(fontsize=10, loc="upper right")

    # ---- Right: free energy and W_2 distance vs time (proxy: |mean - mu*|)
    times_recorded = np.array(times_recorded)
    means = np.array(means)
    # Free energy along the trajectory: re-evaluate with stored snapshots
    snap_times_arr = np.array(snapshot_times)
    F_snap = []
    for snap in snapshots:
        F = (np.trapz(V * snap, x)
             + np.trapz(snap * np.log(np.maximum(snap, 1e-12)), x))
        F_snap.append(F)
    F_snap = np.array(F_snap)
    F_star = (np.trapz(V * rho_star, x)
              + np.trapz(rho_star * np.log(rho_star), x))

    ax_right.plot(snap_times_arr, F_snap - F_star,
                  color=BLUE, lw=2.4, marker="o",
                  label=r"$F[\rho_t] - F[\rho_\ast]$")
    ax_right.plot(times_recorded, np.abs(means - mu_star),
                  color=PURPLE, lw=2.0, alpha=0.9,
                  label=r"$|\mathbb{E}\rho_t - \mu_\ast|$  (mean drift)")
    ax_right.set_xlabel("time $t$", fontsize=12)
    ax_right.set_ylabel("dissipation along the flow", fontsize=12)
    ax_right.set_yscale("log")
    ax_right.set_title("Energy dissipation: " r"$\dot F = -\|\nabla \log\rho + \nabla V\|_\rho^2$",
                       fontsize=12)
    ax_right.legend(fontsize=10)

    fig.suptitle("Wasserstein gradient flow of the free energy $F[\\rho]$",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_wasserstein_gradient_flow.png")


# ---------------------------------------------------------------------------
# Figure 4: Mean-field limit of a two-layer neural network
# ---------------------------------------------------------------------------

def fig4_mean_field_limit() -> None:
    """Empirical particle distribution converges as width m grows.

    We train a two-layer ReLU network of width m to fit a fixed target
    function (a small sum of bumps).  Three widths are compared; the
    histograms of first-layer weights at three training snapshots
    illustrate convergence to a smooth mean-field density as m -> infinity.
    """
    print("Figure 4: mean-field limit (width sweep)")
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    np.random.seed(0)

    # Target: f*(x) = sin(pi x) + 0.5 sin(3 pi x), x in [-1, 1]
    n_data = 64
    x_np = np.linspace(-1.0, 1.0, n_data).reshape(-1, 1)
    y_np = np.sin(np.pi * x_np) + 0.5 * np.sin(3 * np.pi * x_np)
    X = torch.tensor(x_np, dtype=torch.float32)
    Y = torch.tensor(y_np, dtype=torch.float32)

    class TwoLayerNet(nn.Module):
        def __init__(self, m: int) -> None:
            super().__init__()
            self.m = m
            self.W = nn.Parameter(torch.randn(m) * 1.5)       # input weights
            self.b = nn.Parameter(torch.randn(m) * 0.5)       # input biases
            self.a = nn.Parameter(torch.randn(m) / np.sqrt(m))  # output

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (n, 1)
            h = torch.relu(x * self.W + self.b)               # (n, m)
            return (h * self.a).sum(dim=1, keepdim=True)

    widths = [20, 200, 2000]
    snapshot_epochs = [0, 200, 1500]
    snapshots = {m: [] for m in widths}
    losses = {m: [] for m in widths}

    for m in widths:
        model = TwoLayerNet(m)
        # Mean-field scaling: lr ~ 1/sqrt(m) keeps the per-particle drift O(1).
        lr = 0.05 / np.sqrt(m)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        snapshots[m].append((0, model.W.detach().cpu().numpy().copy()))
        losses[m].append((0, float(loss_fn(model(X), Y).item())))

        for epoch in range(1, max(snapshot_epochs) + 1):
            opt.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            opt.step()
            if epoch in snapshot_epochs:
                snapshots[m].append((epoch, model.W.detach().cpu().numpy().copy()))
            if epoch % 50 == 0 or epoch == 1:
                losses[m].append((epoch, float(loss.item())))

    # Plot: 3 rows (widths) x 4 cols (3 snapshots + loss curve)
    fig, axes = plt.subplots(len(widths), 4, figsize=(15, 3.4 * len(widths)))
    snap_color = [BLUE, PURPLE, GREEN]

    for i, m in enumerate(widths):
        for j, (epoch, w) in enumerate(snapshots[m]):
            ax = axes[i, j]
            # Histogram (density) of first-layer weights
            ax.hist(w, bins=40, density=True, color=snap_color[j],
                    alpha=0.55, edgecolor="white")
            # KDE overlay (only meaningful when there is enough variation)
            if w.std() > 1e-6:
                xs = np.linspace(w.min() - 0.3, w.max() + 0.3, 300)
                # Cheap Gaussian KDE
                bw = 1.06 * w.std() * len(w) ** (-1 / 5)
                kde = np.mean(
                    np.exp(-0.5 * ((xs[:, None] - w[None, :]) / bw) ** 2),
                    axis=1,
                ) / (bw * np.sqrt(2 * np.pi))
                ax.plot(xs, kde, color=RED, lw=2.0, label="KDE")
            ax.set_title(fr"$m = {m}$,  epoch ${epoch}$", fontsize=11)
            ax.set_xlabel(r"first-layer weight $w$", fontsize=10)
            if j == 0:
                ax.set_ylabel(r"empirical density $\rho_t^m(w)$", fontsize=10)
            ax.set_xlim(-5, 5)
            if j == 0:
                ax.legend(fontsize=9, loc="upper right")

        # Loss curve in the rightmost column
        ax = axes[i, 3]
        ep, lo = zip(*losses[m])
        ax.plot(ep, lo, color=snap_color[i], lw=2.3,
                label=fr"$m = {m}$")
        ax.set_yscale("log")
        ax.set_xlabel("epoch", fontsize=10)
        ax.set_ylabel("training MSE", fontsize=10)
        ax.set_title("loss decay", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Mean-field limit: empirical weight distribution $\\rho_t^m \\to \\rho_t$"
        " as width $m \\to \\infty$",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, "fig4_mean_field_limit.png")


# ---------------------------------------------------------------------------
# Figure 5: Energy landscape (3D) with gradient-flow trajectory
# ---------------------------------------------------------------------------

def fig5_energy_landscape() -> None:
    """A non-convex 2-D energy landscape with a gradient-flow trajectory."""
    print("Figure 5: 3-D energy landscape with trajectory")
    fig = plt.figure(figsize=(13.5, 5.5))

    # Energy: a tilted double-well-style surface
    def E(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            (x ** 2 - 1) ** 2
            + 0.7 * y ** 2
            + 0.6 * np.sin(2.0 * x) * np.cos(2.0 * y)
            + 0.15 * x
        )

    def grad_E(z: np.ndarray) -> np.ndarray:
        x, y = z
        gx = 4 * x * (x ** 2 - 1) + 1.2 * np.cos(2 * x) * np.cos(2 * y) + 0.15
        gy = 1.4 * y - 1.2 * np.sin(2 * x) * np.sin(2 * y)
        return np.array([gx, gy])

    xs = np.linspace(-1.8, 1.8, 240)
    ys = np.linspace(-1.6, 1.6, 240)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = E(XX, YY)

    # Gradient-flow trajectory (continuous-time ODE)
    sol = solve_ivp(
        lambda t, z: -grad_E(z),
        t_span=(0, 20.0),
        y0=[-1.55, 1.45],
        t_eval=np.linspace(0, 20.0, 600),
        method="RK45",
    )
    traj = sol.y
    traj_z = E(traj[0], traj[1]) + 0.05  # lift slightly above the surface

    # ---- 3-D surface
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3.plot_surface(
        XX, YY, ZZ, cmap="viridis", alpha=0.85,
        linewidth=0, antialiased=True, rstride=4, cstride=4,
    )
    ax3.plot(traj[0], traj[1], traj_z, color=RED, lw=3.0,
             label="gradient flow")
    ax3.scatter(traj[0, 0], traj[1, 0], traj_z[0], color=BLUE, s=70,
                label="start")
    ax3.scatter(traj[0, -1], traj[1, -1], traj_z[-1], color=GREEN, s=80,
                label="basin minimum")
    ax3.set_xlabel("$\\theta_1$")
    ax3.set_ylabel("$\\theta_2$")
    ax3.set_zlabel(r"$E(\theta)$")
    ax3.view_init(elev=32, azim=-58)
    ax3.set_title("Non-convex energy landscape", fontsize=13)
    ax3.legend(fontsize=10, loc="upper left")

    # ---- 2-D contour with trajectory and energy time-series
    ax2 = fig.add_subplot(2, 2, 2)
    cf = ax2.contourf(XX, YY, ZZ, levels=24, cmap="viridis", alpha=0.85)
    ax2.contour(XX, YY, ZZ, levels=24, colors="white",
                linewidths=0.4, alpha=0.5)
    ax2.plot(traj[0], traj[1], color=RED, lw=2.4, label="gradient flow")
    ax2.scatter(traj[0, 0], traj[1, 0], color=BLUE, s=70, zorder=5)
    ax2.scatter(traj[0, -1], traj[1, -1], color=GREEN, s=80, zorder=5)
    ax2.set_xlabel("$\\theta_1$"); ax2.set_ylabel("$\\theta_2$")
    ax2.set_title("Top-down view", fontsize=12)
    plt.colorbar(cf, ax=ax2, fraction=0.046, pad=0.04)

    ax_e = fig.add_subplot(2, 2, 4)
    ax_e.plot(sol.t, E(traj[0], traj[1]), color=BLUE, lw=2.2)
    ax_e.set_xlabel("time $t$", fontsize=11)
    ax_e.set_ylabel(r"$E(\theta(t))$", fontsize=11)
    ax_e.set_title(r"Energy dissipation: $\dot E = -\|\nabla E\|^2 \leq 0$",
                   fontsize=11)

    fig.suptitle("Gradient flow on a non-convex energy landscape",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_energy_landscape.png")


# ---------------------------------------------------------------------------
# Figure 6: Variational autoencoder - ELBO decomposition
# ---------------------------------------------------------------------------

def fig6_variational_autoencoder() -> None:
    """Train a small VAE on synthetic 2-D data; plot ELBO components.

    The variational lower bound
        ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    is itself a *variational principle*: maximising it is equivalent to
    minimising the KL between q and the true posterior.  The figure shows
    how the reconstruction term and the KL regularizer trade off during
    training, plus a 2-D latent map.
    """
    print("Figure 6: Variational autoencoder (ELBO decomposition)")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(7)
    np.random.seed(7)

    # Synthetic 2-D data: a mixture of Gaussians on a circle (4 modes)
    n_per = 256
    centres = np.array([[1.5, 0.0], [-1.5, 0.0], [0.0, 1.5], [0.0, -1.5]])
    data = np.vstack([c + 0.25 * np.random.randn(n_per, 2) for c in centres])
    np.random.shuffle(data)
    X = torch.tensor(data, dtype=torch.float32)

    class VAE(nn.Module):
        def __init__(self, in_dim: int = 2, h: int = 32, z: int = 2) -> None:
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(in_dim, h), nn.Tanh(),
                                     nn.Linear(h, h), nn.Tanh())
            self.fc_mu = nn.Linear(h, z)
            self.fc_logvar = nn.Linear(h, z)
            self.dec = nn.Sequential(nn.Linear(z, h), nn.Tanh(),
                                     nn.Linear(h, h), nn.Tanh(),
                                     nn.Linear(h, in_dim))

        def encode(self, x: torch.Tensor):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparam(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparam(mu, logvar)
            return self.dec(z), mu, logvar, z

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    n_epochs = 800
    batch = 256
    rec_history, kl_history, elbo_history = [], [], []
    n = X.shape[0]
    for ep in range(n_epochs):
        idx = torch.randperm(n)[:batch]
        xb = X[idx]
        recon, mu, logvar, _ = model(xb)
        rec_loss = F.mse_loss(recon, xb, reduction="mean") * xb.shape[1]
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
        loss = rec_loss + kl
        opt.zero_grad(); loss.backward(); opt.step()
        rec_history.append(float(rec_loss.item()))
        kl_history.append(float(kl.item()))
        elbo_history.append(-(rec_history[-1] + kl_history[-1]))

    # ---- Plot
    fig = plt.figure(figsize=(14, 5.5))

    # Panel (a): training curves
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(rec_history, color=BLUE, lw=2.0,
             label=r"reconstruction $-\mathbb{E}_q[\log p(x|z)]$")
    ax1.plot(kl_history, color=PURPLE, lw=2.0,
             label=r"KL$(q(z|x)\,\Vert\,p(z))$")
    ax1.plot(np.array(rec_history) + np.array(kl_history),
             color=RED, lw=2.0, label="negative ELBO")
    ax1.set_xlabel("epoch", fontsize=11)
    ax1.set_ylabel("loss", fontsize=11)
    ax1.set_title("ELBO decomposition", fontsize=12)
    ax1.set_yscale("log")
    ax1.legend(fontsize=9, loc="upper right")

    # Panel (b): data and reconstructions
    ax2 = fig.add_subplot(1, 3, 2)
    with torch.no_grad():
        recon, mu, logvar, z = model(X)
    recon_np = recon.numpy()
    ax2.scatter(data[:, 0], data[:, 1], s=14, color=BLUE, alpha=0.55,
                label="data $x$")
    ax2.scatter(recon_np[:, 0], recon_np[:, 1], s=14, color=RED, alpha=0.55,
                label=r"reconstruction $\hat x$")
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    ax2.set_title("data vs reconstruction", fontsize=12)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_aspect("equal")

    # Panel (c): latent space coloured by data cluster
    ax3 = fig.add_subplot(1, 3, 3)
    cluster = np.argmin(
        np.linalg.norm(data[:, None, :] - centres[None, :, :], axis=2),
        axis=1,
    )
    z_np = z.numpy()
    palette = [BLUE, PURPLE, GREEN, RED]
    for k in range(4):
        mask = cluster == k
        ax3.scatter(z_np[mask, 0], z_np[mask, 1], s=14,
                    color=palette[k], alpha=0.7, label=f"mode {k+1}")
    # Draw the standard-normal prior contour
    th = np.linspace(0, 2 * np.pi, 200)
    for r in (1.0, 2.0):
        ax3.plot(r * np.cos(th), r * np.sin(th),
                 color=GRAY, lw=1.0, ls="--", alpha=0.6)
    ax3.set_xlabel("$z_1$"); ax3.set_ylabel("$z_2$")
    ax3.set_title(r"latent $z \sim q(z|x)$ vs prior $p(z) = \mathcal{N}(0, I)$",
                  fontsize=11)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.set_aspect("equal")

    fig.suptitle("Variational autoencoder: ELBO is a variational principle",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_variational_autoencoder.png")


# ---------------------------------------------------------------------------
# Figure 7: Hamiltonian flow vs gradient flow on a phase plane
# ---------------------------------------------------------------------------

def fig7_hamiltonian_flow() -> None:
    """Symplectic (Hamiltonian) flow vs gradient flow of the same energy.

    Use the harmonic oscillator H(q, p) = 0.5 (q^2 + p^2):
      - Hamiltonian flow: dq/dt =  H_p,  dp/dt = -H_q  (energy preserving)
      - Gradient flow:    dq/dt = -H_q,  dp/dt = -H_p  (energy dissipating)
    Show the vector fields and a few orbits of each.
    """
    print("Figure 7: Hamiltonian vs gradient flow")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    qs = np.linspace(-2.2, 2.2, 26)
    ps = np.linspace(-2.2, 2.2, 26)
    Q, P = np.meshgrid(qs, ps)
    H = 0.5 * (Q ** 2 + P ** 2)

    # ---- Hamiltonian flow: rotation
    U_h = P
    V_h = -Q
    speed_h = np.hypot(U_h, V_h)
    ax = axes[0]
    cf = ax.contourf(Q, P, H, levels=18, cmap="Blues", alpha=0.55)
    ax.contour(Q, P, H, levels=8, colors="white", linewidths=0.6)
    ax.streamplot(Q, P, U_h, V_h, color=BLUE, density=1.2,
                  linewidth=1.0, arrowsize=1.0)

    # Closed orbits
    th = np.linspace(0, 2 * np.pi, 400)
    for r, c in [(0.6, RED), (1.2, PURPLE), (1.8, GREEN)]:
        ax.plot(r * np.cos(th), r * np.sin(th), color=c, lw=2.4,
                label=fr"orbit $H = {0.5 * r**2:.2f}$")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.set_xlabel("$q$  (position)", fontsize=12)
    ax.set_ylabel("$p$  (momentum)", fontsize=12)
    ax.set_title(r"Hamiltonian flow:  $\dot q = H_p,\ \dot p = -H_q$",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="$H(q,p)$")

    # ---- Gradient flow: contraction toward origin
    U_g = -Q
    V_g = -P
    ax = axes[1]
    cf = ax.contourf(Q, P, H, levels=18, cmap="Reds", alpha=0.55)
    ax.contour(Q, P, H, levels=8, colors="white", linewidths=0.6)
    ax.streamplot(Q, P, U_g, V_g, color=RED, density=1.2,
                  linewidth=1.0, arrowsize=1.0)
    # Trajectories
    for q0, p0, c in [(2.0, 1.5, BLUE), (-1.5, 2.0, PURPLE),
                      (1.7, -1.7, GREEN)]:
        sol = solve_ivp(lambda t, z: [-z[0], -z[1]],
                        t_span=(0, 5.0),
                        y0=[q0, p0], t_eval=np.linspace(0, 5, 200))
        ax.plot(sol.y[0], sol.y[1], color=c, lw=2.4,
                label=fr"orbit from $({q0:.1f}, {p0:.1f})$")
        ax.scatter(q0, p0, color=c, s=60, zorder=5)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.set_xlabel("$q$", fontsize=12); ax.set_ylabel("$p$", fontsize=12)
    ax.set_title(r"Gradient flow of $H$:  $\dot q = -H_q,\ \dot p = -H_p$",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="$H(q,p)$")

    fig.suptitle(
        "Two flows on the same energy: symplectic (preserves $H$) vs gradient (dissipates $H$)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "fig7_hamiltonian_flow.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"EN target: {EN_DIR}")
    print(f"ZH target: {ZH_DIR}")
    print()
    fig1_functional_minimization()
    fig2_euler_lagrange()
    fig3_wasserstein_gradient_flow()
    fig4_mean_field_limit()
    fig5_energy_landscape()
    fig6_variational_autoencoder()
    fig7_hamiltonian_flow()
    print("\nAll 7 figures generated.")


if __name__ == "__main__":
    main()
