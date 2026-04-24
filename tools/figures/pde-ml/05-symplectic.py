#!/usr/bin/env python3
"""
PDE and Machine Learning, Part 5 - Symplectic Geometry and Structure-Preserving Networks.
Figure generation script.

Generates 7 figures and writes them to BOTH the EN and ZH asset folders:
  - source/_posts/en/pde-ml/05-Symplectic-Geometry/
  - source/_posts/zh/pde-ml/05-辛几何与保结构网络/

Run from anywhere:
    python 05-symplectic.py

Style: seaborn-v0_8-whitegrid, dpi=150
Palette: blue #2563eb, purple #7c3aed, green #10b981, red #ef4444
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
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
DARK = COLORS["ink"]


DPI = 150

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "pde-ml" / "05-Symplectic-Geometry"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "pde-ml" / "05-辛几何与保结构网络"

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
# Hamiltonian dynamics helpers
# ---------------------------------------------------------------------------

def pendulum_rhs(t, y, g_over_l=1.0):
    q, p = y
    return [p, -g_over_l * np.sin(q)]


def pendulum_H(q, p, g_over_l=1.0):
    return 0.5 * p**2 + g_over_l * (1.0 - np.cos(q))


def harmonic_rhs(t, y, omega2=1.0):
    q, p = y
    return [p, -omega2 * q]


def harmonic_H(q, p, omega2=1.0):
    return 0.5 * (p**2 + omega2 * q**2)


def explicit_euler(rhs, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        dy = np.array(rhs(t[k], y[k]))
        y[k + 1] = y[k] + h * dy
    return y


def symplectic_euler_pendulum(y0, t, g_over_l=1.0):
    """Symplectic Euler for separable H = p^2/2 + V(q)."""
    y = np.zeros((len(t), 2))
    y[0] = y0
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        q, p = y[k]
        # update p first using grad_q V(q) = sin(q)
        p_new = p - h * g_over_l * np.sin(q)
        q_new = q + h * p_new
        y[k + 1] = [q_new, p_new]
    return y


def leapfrog_pendulum(y0, t, g_over_l=1.0):
    """Stormer-Verlet leapfrog for the pendulum."""
    y = np.zeros((len(t), 2))
    y[0] = y0
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        q, p = y[k]
        p_half = p - 0.5 * h * g_over_l * np.sin(q)
        q_new = q + h * p_half
        p_new = p_half - 0.5 * h * g_over_l * np.sin(q_new)
        y[k + 1] = [q_new, p_new]
    return y


# ---------------------------------------------------------------------------
# FIG 1: Hamiltonian flow preserves volume in phase space
# ---------------------------------------------------------------------------

def fig1_hamiltonian_flow():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: phase-space orbits coloured by energy
    ax = axes[0]
    qq = np.linspace(-3.4, 3.4, 220)
    pp = np.linspace(-2.6, 2.6, 200)
    Q, P = np.meshgrid(qq, pp)
    H = pendulum_H(Q, P)
    levels = [0.15, 0.45, 0.85, 1.4, 1.95, 2.4, 3.0]
    cs = ax.contour(Q, P, H, levels=levels, colors=[BLUE, BLUE, PURPLE, PURPLE,
                                                     ORANGE, RED, RED],
                    linewidths=1.6, alpha=0.85)
    ax.contourf(Q, P, H, levels=30, cmap="Blues", alpha=0.18)

    # Add direction arrows along one orbit
    sol = solve_ivp(pendulum_rhs, (0, 2 * np.pi), [1.5, 0.0], dense_output=True,
                    rtol=1e-9, atol=1e-11)
    ts = np.linspace(0, 2 * np.pi, 14)
    pts = sol.sol(ts)
    for i in range(0, len(ts) - 1, 2):
        ax.annotate("", xy=(pts[0, i + 1], pts[1, i + 1]),
                    xytext=(pts[0, i], pts[1, i]),
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5,
                                    alpha=0.75))

    ax.set_xlabel(r"position $q$", fontsize=11)
    ax.set_ylabel(r"momentum $p$", fontsize=11)
    ax.set_title("Phase-space orbits of the pendulum\n(level sets of $H$)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-2.6, 2.6)

    # --- Right: a phase-space "blob" gets transported but keeps its area
    ax = axes[1]
    # initial blob: a small disk
    n_pts = 220
    theta = np.linspace(0, 2 * np.pi, n_pts)
    r = 0.28
    blob0 = np.array([1.6 + r * np.cos(theta), 0.0 + r * np.sin(theta)])

    snapshots = [0.0, 1.0, 2.4, 4.2]
    colors = [BLUE, PURPLE, ORANGE, RED]
    labels = ["$t=0$", "$t=1.0$", "$t=2.4$", "$t=4.2$"]

    for snap_t, col, lbl in zip(snapshots, colors, labels):
        pts = np.zeros_like(blob0)
        for j in range(blob0.shape[1]):
            sol = solve_ivp(harmonic_rhs, (0, snap_t), blob0[:, j],
                            rtol=1e-10, atol=1e-12)
            pts[:, j] = sol.y[:, -1]
        ax.fill(pts[0], pts[1], color=col, alpha=0.32, edgecolor=col,
                linewidth=1.6, label=lbl)

    # Background level sets of harmonic H (concentric circles)
    qq = np.linspace(-2.4, 2.4, 200)
    pp = np.linspace(-2.4, 2.4, 200)
    Q, P = np.meshgrid(qq, pp)
    Hh = harmonic_H(Q, P)
    ax.contour(Q, P, Hh, levels=[0.4, 1.0, 1.8, 2.8], colors=GRAY,
               linewidths=0.8, alpha=0.6)

    ax.set_xlabel(r"position $q$", fontsize=11)
    ax.set_ylabel(r"momentum $p$", fontsize=11)
    ax.set_title("Liouville's theorem: the flow preserves\nphase-space volume",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_aspect("equal")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)

    fig.suptitle("Hamiltonian flow in phase space",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_hamiltonian_flow.png")


# ---------------------------------------------------------------------------
# FIG 2: Symplectic vs non-symplectic integrator (long-term energy drift)
# ---------------------------------------------------------------------------

def fig2_integrator_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Long simulation of the pendulum
    T = 80.0
    h = 0.05
    t = np.arange(0.0, T + h, h)
    y0 = [1.0, 0.0]

    y_eul = explicit_euler(pendulum_rhs, y0, t)
    y_sym = symplectic_euler_pendulum(y0, t)
    y_lf = leapfrog_pendulum(y0, t)

    # Reference
    sol_ref = solve_ivp(pendulum_rhs, (0, T), y0, t_eval=t,
                        rtol=1e-11, atol=1e-13)

    H_eul = pendulum_H(y_eul[:, 0], y_eul[:, 1])
    H_sym = pendulum_H(y_sym[:, 0], y_sym[:, 1])
    H_lf = pendulum_H(y_lf[:, 0], y_lf[:, 1])
    H_ref = pendulum_H(sol_ref.y[0], sol_ref.y[1])
    H0 = H_ref[0]

    # --- Left: phase-space trajectories
    ax = axes[0]
    ax.plot(sol_ref.y[0], sol_ref.y[1], color=GRAY, lw=1.3, alpha=0.7,
            label="reference (RK45 high-order)")
    ax.plot(y_eul[:, 0], y_eul[:, 1], color=RED, lw=1.0, alpha=0.85,
            label="explicit Euler")
    ax.plot(y_sym[:, 0], y_sym[:, 1], color=GREEN, lw=1.0, alpha=0.85,
            label="symplectic Euler")
    ax.plot(y_lf[:, 0], y_lf[:, 1], color=BLUE, lw=1.0, alpha=0.85,
            label="leapfrog (Stormer-Verlet)")
    ax.set_xlabel(r"$q$", fontsize=11)
    ax.set_ylabel(r"$p$", fontsize=11)
    ax.set_title(f"Phase-space trajectory ($T={int(T)}$ s, $h={h}$)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_aspect("equal")

    # --- Right: relative energy drift
    ax = axes[1]
    ax.axhline(0.0, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.plot(t, (H_eul - H0) / H0, color=RED, lw=1.4,
            label="explicit Euler (drifts)")
    ax.plot(t, (H_sym - H0) / H0, color=GREEN, lw=1.4,
            label="symplectic Euler (bounded)")
    ax.plot(t, (H_lf - H0) / H0, color=BLUE, lw=1.4,
            label="leapfrog (bounded, $O(h^2)$)")
    ax.set_xlabel("time $t$", fontsize=11)
    ax.set_ylabel(r"relative energy error $(H - H_0)/H_0$", fontsize=11)
    ax.set_title("Long-term energy: symplectic methods stay bounded",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_yscale("symlog", linthresh=1e-3)

    fig.tight_layout()
    save(fig, "fig2_integrator_comparison.png")


# ---------------------------------------------------------------------------
# FIG 3: Hamiltonian Neural Network architecture
# ---------------------------------------------------------------------------

def fig3_hnn_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, label, color, fc=None, lw=1.6, fontsize=10.5):
        if fc is None:
            fc = color
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.04,rounding_size=0.12",
                              facecolor=fc, edgecolor=color,
                              linewidth=lw, alpha=0.95)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=DARK)

    def arrow(x1, y1, x2, y2, color=DARK, lw=1.8, style="->"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, mutation_scale=15,
                                     color=color, lw=lw))

    # Input
    box(0.2, 2.6, 1.6, 1.3, r"$(q,\ p)$", BLUE,
        fc="#dbeafe", fontsize=13)
    ax.text(1.0, 1.95, "phase-space\nstate", ha="center", va="top",
            fontsize=9, color=GRAY)

    # MLP block
    box(2.4, 1.9, 3.2, 2.6, "", PURPLE, fc="#ede9fe")
    ax.text(4.0, 4.25, "MLP $H_\\theta$", ha="center", va="center",
            fontsize=12, fontweight="bold", color=PURPLE)
    # neurons
    for i, yc in enumerate([3.2, 2.7, 2.2]):
        for xc in [3.0, 3.6, 4.2, 4.8]:
            ax.add_patch(plt.Circle((xc, yc), 0.11, color=PURPLE,
                                    alpha=0.55))
    ax.text(4.0, 1.7, "tanh / softplus\nactivation", ha="center", va="top",
            fontsize=9, color=GRAY)

    # Scalar H
    box(6.2, 2.6, 1.7, 1.3, r"$H_\theta(q,p)$", PURPLE,
        fc="#fbeaff", fontsize=13)
    ax.text(7.05, 1.95, "scalar energy", ha="center", va="top",
            fontsize=9, color=GRAY)

    # Autodiff block
    box(8.6, 2.6, 1.9, 1.3, "autodiff\n$\\nabla_q H,\\ \\nabla_p H$",
        ORANGE, fc="#fff4e0", fontsize=10.5)

    # Output dynamics
    box(11.1, 2.6, 1.7, 1.3, r"$\dot q = +\partial_p H$" + "\n" +
        r"$\dot p = -\partial_q H$", GREEN, fc="#dcfce7", fontsize=10)
    ax.text(11.95, 1.95, "Hamilton\nequations",
            ha="center", va="top", fontsize=9, color=GRAY)

    # Arrows
    arrow(1.8, 3.25, 2.4, 3.25)
    arrow(5.6, 3.25, 6.2, 3.25)
    arrow(7.9, 3.25, 8.6, 3.25)
    arrow(10.5, 3.25, 11.1, 3.25)

    # Loss feedback (training)
    ax.text(6.5, 5.5, "Training loss",
            ha="center", fontsize=11, fontweight="bold", color=RED)
    box(4.8, 5.0, 4.0, 0.7,
        r"$\mathcal{L}=\| \partial_p H_\theta - \dot q\|^2 + \| \partial_q H_\theta + \dot p\|^2$",
        RED, fc="#fee2e2", fontsize=10.5)
    arrow(11.95, 4.6, 8.8, 5.0, color=RED, lw=1.4, style="->")
    arrow(4.8, 5.0, 4.0, 4.5, color=RED, lw=1.4, style="->")

    # Bottom note: by construction
    ax.text(6.5, 0.55,
            "By construction: $\\frac{dH_\\theta}{dt} = \\nabla H^{\\!T} J\\,\\nabla H = 0$  -  energy is conserved exactly",
            ha="center", fontsize=11, color=GREEN, fontweight="bold",
            bbox=dict(facecolor="#dcfce7", edgecolor=GREEN,
                      boxstyle="round,pad=0.4"))

    ax.set_title("Hamiltonian Neural Network (HNN) - learn the energy, not the vector field",
                 fontsize=13, fontweight="bold", pad=14)

    save(fig, "fig3_hnn_architecture.png")


# ---------------------------------------------------------------------------
# FIG 4: Leapfrog integrator visualization
# ---------------------------------------------------------------------------

def fig4_leapfrog():
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)

    # --- Left: leapfrog stencil schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.5, 6.2)
    ax.set_ylim(-0.4, 2.8)
    ax.axis("off")

    # Time axis
    ax.annotate("", xy=(6.0, 0.0), xytext=(-0.3, 0.0),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5))
    ax.text(6.05, -0.15, "time", fontsize=11)

    # q-line and p-line labels
    ax.text(-0.4, 2.0, r"$q$", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(-0.4, 1.0, r"$p$", fontsize=14, color=PURPLE, fontweight="bold")

    # nodes
    q_xs = [0.5, 2.5, 4.5]
    q_labels = [r"$q_k$", r"$q_{k+1}$", r"$q_{k+2}$"]
    p_xs = [1.5, 3.5, 5.5]
    p_labels = [r"$p_{k+1/2}$", r"$p_{k+3/2}$", r"$p_{k+5/2}$"]

    for x, lbl in zip(q_xs, q_labels):
        ax.add_patch(plt.Circle((x, 2.0), 0.16, color=BLUE, alpha=0.85))
        ax.text(x, 2.45, lbl, ha="center", fontsize=11, color=BLUE,
                fontweight="bold")
        ax.plot([x, x], [0.05, -0.05], color=DARK, lw=1.2)
        ax.text(x, -0.27, f"$t={int((x-0.5)/2)}h$" if x == 0.5 else
                (f"$t={int((x-0.5)/2)}h$"), ha="center", fontsize=9,
                color=GRAY)

    for x, lbl in zip(p_xs, p_labels):
        ax.add_patch(plt.Circle((x, 1.0), 0.16, color=PURPLE, alpha=0.85))
        ax.text(x, 0.6, lbl, ha="center", fontsize=11, color=PURPLE,
                fontweight="bold")

    # arrows: q_k -> p_{k+1/2}
    def arr(x1, y1, x2, y2, c):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="->", mutation_scale=12,
                                     color=c, lw=1.5, alpha=0.8))
    # kick (V'(q) updates p)
    arr(0.5, 1.85, 1.4, 1.15, RED)
    arr(2.5, 1.85, 3.4, 1.15, RED)
    arr(4.5, 1.85, 5.4, 1.15, RED)
    # drift (p updates q)
    arr(1.5, 1.15, 2.4, 1.85, GREEN)
    arr(3.5, 1.15, 4.4, 1.85, GREEN)

    # legend
    ax.plot([], [], color=RED, lw=1.8, label=r"kick: $p \leftarrow p - \frac{h}{2}V'(q)$")
    ax.plot([], [], color=GREEN, lw=1.8, label=r"drift: $q \leftarrow q + h\,p$")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    ax.set_title("Leapfrog stencil: $q$ and $p$ live on staggered time grids",
                 fontsize=12, fontweight="bold")

    # --- Right: leapfrog vs RK4 vs explicit Euler on harmonic oscillator
    ax = fig.add_subplot(gs[0, 1])
    T = 30.0
    h = 0.18
    t = np.arange(0, T, h)
    y0 = [1.0, 0.0]

    y_eul = explicit_euler(harmonic_rhs, y0, t)
    y_lf = np.zeros((len(t), 2))
    y_lf[0] = y0
    for k in range(len(t) - 1):
        q, p = y_lf[k]
        p_half = p - 0.5 * h * q
        q_new = q + h * p_half
        p_new = p_half - 0.5 * h * q_new
        y_lf[k + 1] = [q_new, p_new]

    # RK4
    def rk4_step(rhs, y, h):
        k1 = np.array(rhs(0, y))
        k2 = np.array(rhs(0, y + 0.5 * h * k1))
        k3 = np.array(rhs(0, y + 0.5 * h * k2))
        k4 = np.array(rhs(0, y + h * k3))
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    y_rk4 = np.zeros((len(t), 2))
    y_rk4[0] = y0
    for k in range(len(t) - 1):
        y_rk4[k + 1] = rk4_step(harmonic_rhs, y_rk4[k], h)

    H_eul = harmonic_H(y_eul[:, 0], y_eul[:, 1])
    H_lf = harmonic_H(y_lf[:, 0], y_lf[:, 1])
    H_rk4 = harmonic_H(y_rk4[:, 0], y_rk4[:, 1])
    H0 = 0.5

    ax.plot(t, (H_eul - H0) / H0, color=RED, lw=1.5, label="explicit Euler")
    ax.plot(t, (H_rk4 - H0) / H0, color=ORANGE, lw=1.5, label="RK4 (non-symplectic)")
    ax.plot(t, (H_lf - H0) / H0, color=BLUE, lw=1.5, label="leapfrog")
    ax.axhline(0, color=GRAY, lw=0.7, ls="--")
    ax.set_xlabel("time $t$", fontsize=11)
    ax.set_ylabel(r"relative energy error", fontsize=11)
    ax.set_title("Harmonic oscillator: leapfrog energy is bounded",
                 fontsize=12, fontweight="bold")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig4_leapfrog.png")


# ---------------------------------------------------------------------------
# FIG 5: Pendulum - HNN preserves energy, vanilla NN drifts (mock training)
# ---------------------------------------------------------------------------

def fig5_pendulum_hnn_vs_nn():
    """
    We don't actually train networks here; we emulate the qualitative behaviour
    that has been documented in Greydanus et al. (2019) and reproduced widely:

    - Vanilla NN: learns f_theta directly. Tiny systematic bias in f_theta
      destroys symplecticity, so energy drifts (often *down* once dissipation-
      like errors dominate, or *up* if gain-like errors dominate). We mimic
      this with a mild non-conservative perturbation of the true vector field.
    - HNN: learns H_theta. Even with a small bias on H_theta the dynamics
      remain Hamiltonian, so energy is bounded and oscillates around a slightly
      shifted level.

    The reference is the high-accuracy solver.
    """
    rng = np.random.default_rng(0)
    T = 25.0
    h = 0.02
    t = np.arange(0, T, h)
    y0 = [1.4, 0.0]

    # Reference
    sol = solve_ivp(pendulum_rhs, (0, T), y0, t_eval=t,
                    rtol=1e-11, atol=1e-13)
    q_ref, p_ref = sol.y
    H_ref = pendulum_H(q_ref, p_ref)
    H0 = H_ref[0]

    # Vanilla NN: f_theta = J grad H + small dissipation-like term
    eps = 0.012
    y_nn = np.zeros((len(t), 2))
    y_nn[0] = y0
    for k in range(len(t) - 1):
        q, p = y_nn[k]
        # true Hamiltonian field
        dq = p
        dp = -np.sin(q)
        # learned bias: slow energy gain (e.g. learned f over-predicts |p|)
        dq_bias = eps * p
        dp_bias = -eps * np.sin(q) * 0.0 + eps * 0.6 * p
        y_nn[k + 1, 0] = q + h * (dq + dq_bias)
        y_nn[k + 1, 1] = p + h * (dp + dp_bias)
    H_nn = pendulum_H(y_nn[:, 0], y_nn[:, 1])

    # HNN: dynamics from a slightly biased H_theta integrated symplectically.
    delta = 1.04   # learned H differs from true H by a 4% scaling
    y_hnn = np.zeros((len(t), 2))
    y_hnn[0] = y0
    for k in range(len(t) - 1):
        q, p = y_hnn[k]
        # H_theta = 0.5 p^2 + delta*(1 - cos q); dH/dq = delta sin q, dH/dp = p
        p_half = p - 0.5 * h * delta * np.sin(q)
        q_new = q + h * p_half
        p_new = p_half - 0.5 * h * delta * np.sin(q_new)
        y_hnn[k + 1] = [q_new, p_new]
    H_hnn = pendulum_H(y_hnn[:, 0], y_hnn[:, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- (a) Trajectories q(t)
    ax = axes[0]
    ax.plot(t, q_ref, color=GRAY, lw=1.4, label="reference")
    ax.plot(t, y_nn[:, 0], color=RED, lw=1.3, alpha=0.9, label="vanilla NN")
    ax.plot(t, y_hnn[:, 0], color=BLUE, lw=1.3, alpha=0.9, label="HNN")
    ax.set_xlabel("time $t$", fontsize=11)
    ax.set_ylabel(r"angle $q(t)$", fontsize=11)
    ax.set_title("Pendulum trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)

    # --- (b) Phase portrait
    ax = axes[1]
    ax.plot(q_ref, p_ref, color=GRAY, lw=1.3, alpha=0.9, label="reference")
    ax.plot(y_nn[:, 0], y_nn[:, 1], color=RED, lw=1.0, alpha=0.85,
            label="vanilla NN")
    ax.plot(y_hnn[:, 0], y_hnn[:, 1], color=BLUE, lw=1.0, alpha=0.85,
            label="HNN")
    ax.set_xlabel(r"$q$", fontsize=11)
    ax.set_ylabel(r"$p$", fontsize=11)
    ax.set_title("Phase-space trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    ax.set_aspect("equal")

    # --- (c) Energy over time
    ax = axes[2]
    ax.axhline(0, color=GRAY, lw=0.7, ls="--")
    ax.plot(t, (H_nn - H0) / H0, color=RED, lw=1.4,
            label=r"vanilla NN: $\Delta E / E_0$")
    ax.plot(t, (H_hnn - H0) / H0, color=BLUE, lw=1.4,
            label=r"HNN: $\Delta E / E_0$")
    ax.set_xlabel("time $t$", fontsize=11)
    ax.set_ylabel(r"relative energy error", fontsize=11)
    ax.set_title("Energy: vanilla NN drifts, HNN is bounded",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)

    fig.suptitle("Pendulum: HNN conserves energy by construction, a vanilla NN does not",
                 fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig5_pendulum_hnn_vs_nn.png")


# ---------------------------------------------------------------------------
# FIG 6: Symplectic vs Lagrangian neural network (architecture comparison)
# ---------------------------------------------------------------------------

def fig6_hnn_lnn_sympnet():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    def draw_pipeline(ax, title, color, blocks, note):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=color,
                     pad=10)
        n = len(blocks)
        spacing = 9.0 / n
        ys = [4.5 - i * 0.0 for i in range(n)]
        for i, (txt, sub) in enumerate(blocks):
            x = 0.5 + i * spacing
            rect = FancyBboxPatch((x, 3.6), spacing - 0.4, 1.4,
                                  boxstyle="round,pad=0.04,rounding_size=0.12",
                                  facecolor="white", edgecolor=color, lw=1.6)
            ax.add_patch(rect)
            ax.text(x + (spacing - 0.4) / 2, 4.55, txt, ha="center",
                    va="center", fontsize=10.5, fontweight="bold",
                    color=DARK)
            ax.text(x + (spacing - 0.4) / 2, 3.95, sub, ha="center",
                    va="center", fontsize=8.5, color=GRAY)
            if i < n - 1:
                ax.add_patch(FancyArrowPatch(
                    (x + spacing - 0.35, 4.3),
                    (x + spacing + 0.05, 4.3),
                    arrowstyle="->", mutation_scale=12,
                    color=color, lw=1.6))

        ax.text(5.0, 1.8, note, ha="center", va="center", fontsize=10,
                color=DARK,
                bbox=dict(facecolor="#f9fafb", edgecolor=color,
                          boxstyle="round,pad=0.5"))

    # HNN
    draw_pipeline(
        axes[0],
        "HNN  -  learn the Hamiltonian",
        BLUE,
        [
            (r"$(q,p)$", "phase-space"),
            (r"$H_\theta(q,p)$", "scalar net"),
            (r"$\dot z = J\,\nabla H_\theta$", "Hamilton eqs"),
        ],
        ("Pros: exact energy conservation\n"
         "Needs: $(q,p,\\dot q,\\dot p)$ data\n"
         "Lives in: phase space"),
    )

    # LNN
    draw_pipeline(
        axes[1],
        "LNN  -  learn the Lagrangian",
        PURPLE,
        [
            (r"$(q,\dot q)$", "config space"),
            (r"$L_\theta(q,\dot q)$", "scalar net"),
            (r"Euler-Lagrange:  $\ddot q = (\partial^2_{\dot q\dot q}L)^{-1}[\dots]$",
             "2nd-order autodiff"),
        ],
        ("Pros: only needs $(q,\\dot q,\\ddot q)$\n"
         "Cost: matrix inverse / Hessian\n"
         "Lives in: configuration space"),
    )

    # SympNet
    draw_pipeline(
        axes[2],
        "SympNet  -  learn the symplectic map",
        GREEN,
        [
            (r"$(q,p)$", "phase-space"),
            (r"$\phi_K \!\circ\! \cdots \!\circ\! \phi_1$",
             "symplectic blocks"),
            (r"$(q',p')$", "discrete-time map"),
        ],
        ("Pros: discrete symplecticity by design\n"
         "No ODE solver needed\n"
         "Building blocks: shears + activations"),
    )

    fig.suptitle("Three structure-preserving neural networks - what each one learns",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_hnn_lnn_sympnet.png")


# ---------------------------------------------------------------------------
# FIG 7: Hamiltonian deep learning - applications landscape
# ---------------------------------------------------------------------------

def fig7_applications():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Center hub
    cx, cy = 7.0, 4.4
    hub = FancyBboxPatch((cx - 1.7, cy - 0.7), 3.4, 1.4,
                         boxstyle="round,pad=0.06,rounding_size=0.18",
                         facecolor="#1f2937", edgecolor=DARK, lw=2.0)
    ax.add_patch(hub)
    ax.text(cx, cy + 0.2, "Hamiltonian /", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(cx, cy - 0.25, "Symplectic deep learning", ha="center", va="center",
            color="white", fontsize=11.5, fontweight="bold")

    # Applications around the hub
    apps = [
        ("Molecular dynamics",
         "long-time MD without\nenergy drift",
         BLUE,    1.6, 7.4),
        ("Robotics & control",
         "learned dynamics for\nplanning, RL",
         PURPLE, 12.4, 7.4),
        ("Celestial mechanics",
         "stable n-body and\norbital integrators",
         GREEN,   1.6, 4.4),
        ("Plasma & accelerators",
         "particle-in-cell with\nlearned fields",
         ORANGE, 12.4, 4.4),
        ("Hamiltonian Monte Carlo",
         "better proposals via\nlearned $H$",
         RED,     1.6, 1.4),
        ("Fluid & climate",
         "structure-preserving\nreduced-order models",
         "#0ea5e9", 12.4, 1.4),
    ]

    for name, desc, col, x, y in apps:
        rect = FancyBboxPatch((x - 1.6, y - 0.85), 3.2, 1.7,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor="white", edgecolor=col, lw=1.8)
        ax.add_patch(rect)
        ax.text(x, y + 0.4, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col)
        ax.text(x, y - 0.25, desc, ha="center", va="center",
                fontsize=9, color=DARK)
        # connector
        ax.add_patch(FancyArrowPatch(
            (x + (1.6 if x < cx else -1.6), y),
            (cx + (-1.7 if x < cx else 1.7), cy + (0.4 if y > cy + 0.5 else
                                                    -0.4 if y < cy - 0.5 else 0.0)),
            arrowstyle="-", color=col, lw=1.4, alpha=0.7))

    # Top banner
    ax.text(7.0, 8.1,
            "Where structure-preserving networks pay off",
            ha="center", fontsize=13.5, fontweight="bold", color=DARK)

    save(fig, "fig7_applications.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating figures for PDE+ML Part 5: Symplectic Geometry")
    print(f"  EN -> {EN_DIR}")
    print(f"  ZH -> {ZH_DIR}")
    print()
    fig1_hamiltonian_flow()
    fig2_integrator_comparison()
    fig3_hnn_architecture()
    fig4_leapfrog()
    fig5_pendulum_hnn_vs_nn()
    fig6_hnn_lnn_sympnet()
    fig7_applications()
    print("\nDone. 7 figures saved to both EN and ZH asset folders.")


if __name__ == "__main__":
    main()
