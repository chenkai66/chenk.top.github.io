"""
Figure generation for ODE Chapter 06: Linear Systems and the Matrix Exponential.

Outputs six figures into BOTH the EN and ZH asset folders:
  fig1_matrix_exponential_series.png   -- Partial sums of e^{At} converging
  fig2_phase_portrait_zoo.png          -- Full classification of 2D linear systems
  fig3_eigenvalue_decomposition.png    -- Geometric meaning: rotate, scale, rotate back
  fig4_trace_determinant_plane.png     -- Stability map in (tr A, det A) space
  fig5_coupled_oscillator_modes.png    -- Normal modes and beats
  fig6_repeated_eigenvalue_shear.png   -- Generalized eigenvector / degenerate node

Run:
    python 06-systems.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import odeint
from scipy.linalg import expm

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.9,
        "grid.color": "#e2e8f0",
        "grid.alpha": 0.6,
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
RED = "#ef4444"
GREY = "#64748b"

REPO = Path(__file__).resolve().parents[3]
EN_DIR = REPO / "source/_posts/en/ode/06-power-series"
ZH_DIR = REPO / "source/_posts/zh/ode/06-线性微分方程组"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 -- Convergence of the matrix exponential series
# ---------------------------------------------------------------------------
def fig1_matrix_exponential_series() -> None:
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])  # rotation generator -> e^{At} is rotation
    ts = np.linspace(0, 2 * np.pi, 400)

    truth = np.array([expm(A * t) @ np.array([1.0, 0.0]) for t in ts])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: partial sums applied to x0 = (1,0)
    ax = axes[0]
    x0 = np.array([1.0, 0.0])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
    for k, color in zip([1, 2, 4, 8, 16], colors):
        traj = []
        for t in ts:
            S = np.eye(2)
            term = np.eye(2)
            for j in range(1, k + 1):
                term = term @ (A * t) / j
                S = S + term
            traj.append(S @ x0)
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.6, label=f"$N={k}$")
    ax.plot(truth[:, 0], truth[:, 1], color=RED, lw=2.2, ls="--", label=r"$e^{At}\mathbf{x}_0$")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"Partial sums $\sum_{k=0}^{N}(At)^k/k!$ acting on $(1,0)$")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)

    # Right: pointwise error vs N for fixed t
    ax = axes[1]
    t_fixed = np.pi
    Ns = np.arange(1, 25)
    errs = []
    truth_t = expm(A * t_fixed)
    for N in Ns:
        S = np.eye(2)
        term = np.eye(2)
        for j in range(1, N + 1):
            term = term @ (A * t_fixed) / j
            S = S + term
        errs.append(np.linalg.norm(S - truth_t, ord=2))
    ax.semilogy(Ns, errs, "o-", color=BLUE, lw=2, markersize=5)
    ax.set_xlabel("Number of terms $N$")
    ax.set_ylabel(r"$\|S_N(A,t)-e^{At}\|_2$  at $t=\pi$")
    ax.set_title("Spectral-norm error decays super-exponentially")
    ax.axhline(1e-15, color=GREY, ls=":", lw=1)

    fig.suptitle(
        r"Matrix exponential as a power series: $e^{At}=\sum_{k=0}^{\infty}(At)^k/k!$",
        fontsize=13.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig1_matrix_exponential_series.png")


# ---------------------------------------------------------------------------
# Figure 2 -- Phase portrait zoo
# ---------------------------------------------------------------------------
def fig2_phase_portrait_zoo() -> None:
    cases = [
        ("Stable node",     np.array([[-2.0, 0.0], [0.0, -0.7]])),
        ("Unstable node",   np.array([[ 1.6, 0.0], [0.0,  0.6]])),
        ("Saddle point",    np.array([[-1.0, 0.0], [0.0,  1.4]])),
        ("Stable spiral",   np.array([[-0.4, 1.2], [-1.2, -0.4]])),
        ("Unstable spiral", np.array([[ 0.4, 1.2], [-1.2,  0.4]])),
        ("Center",          np.array([[ 0.0, 1.0], [-1.0,  0.0]])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for ax, (name, A) in zip(axes.flat, cases):
        # Streamlines for direction field
        Y, X = np.mgrid[-3:3:25j, -3:3:25j]
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y
        speed = np.sqrt(U ** 2 + V ** 2)
        ax.streamplot(
            X, Y, U, V,
            color=speed, cmap="Blues", density=1.2, linewidth=0.9, arrowsize=0.9,
        )

        # Selected trajectories
        t = np.linspace(0, 6, 400)
        t_back = np.linspace(0, -6, 400)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x0 = [2.6 * np.cos(angle), 2.6 * np.sin(angle)]
            for tt in (t, t_back):
                sol = odeint(lambda x, _t: A @ x, x0, tt)
                ax.plot(sol[:, 0], sol[:, 1], color=PURPLE, lw=0.9, alpha=0.55)

        # Eigenvector lines (real eigenvalues only)
        eigvals, eigvecs = np.linalg.eig(A)
        for lam, v in zip(eigvals, eigvecs.T):
            if np.isreal(lam):
                v = np.real(v)
                v = v / np.linalg.norm(v)
                line = np.array([-3 * v, 3 * v])
                color = GREEN if np.real(lam) < 0 else RED
                ax.plot(line[:, 0], line[:, 1], color=color, lw=2, alpha=0.8)

        eig_str = ", ".join(
            f"{lam.real:+.2f}{'' if abs(lam.imag) < 1e-9 else f'{lam.imag:+.2f}i'}"
            for lam in eigvals
        )
        ax.set_title(f"{name}\n$\\lambda$: {eig_str}", fontsize=10.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.plot(0, 0, "o", color="black", markersize=5, zorder=5)

    fig.suptitle(
        "Six canonical 2D linear systems — eigenvalues determine the portrait",
        fontsize=13.5, y=1.005,
    )
    fig.tight_layout()
    save(fig, "fig2_phase_portrait_zoo.png")


# ---------------------------------------------------------------------------
# Figure 3 -- Geometric eigendecomposition: A = P D P^{-1}
# ---------------------------------------------------------------------------
def fig3_eigenvalue_decomposition() -> None:
    # A non-trivial diagonalizable matrix
    A = np.array([[1.5, 0.7], [0.4, 1.1]])
    eigvals, P = np.linalg.eig(A)
    D = np.diag(eigvals)
    P_inv = np.linalg.inv(P)

    # Unit circle and its image under each step
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    step1 = P_inv @ circle             # rotate into eigenbasis
    step2 = D @ step1                  # scale along eigen-axes
    step3 = P @ step2                  # rotate back == A @ circle

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    snapshots = [
        (circle,  "Input: unit circle",                        BLUE),
        (step1,   r"After $P^{-1}$: aligned with eigen-axes",  PURPLE),
        (step2,   r"After $D$: stretched by $\lambda_i$",      GREEN),
        (step3,   r"After $P$: image $A\,\mathbf{x}$",         RED),
    ]
    for ax, (curve, title, color) in zip(axes, snapshots):
        ax.plot(curve[0], curve[1], color=color, lw=2.2)
        ax.fill(curve[0], curve[1], color=color, alpha=0.12)

        # show eigenvectors as reference
        for lam, v in zip(eigvals, P.T):
            v = np.real(v) / np.linalg.norm(np.real(v))
            ax.annotate(
                "", xy=(2.0 * v[0], 2.0 * v[1]), xytext=(-2.0 * v[0], -2.0 * v[1]),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1, ls=":"),
            )
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)

    fig.suptitle(
        r"Eigendecomposition $A = P D P^{-1}$ as three geometric steps "
        f"(λ = {eigvals[0]:.2f}, {eigvals[1]:.2f})",
        fontsize=13.5, y=1.04,
    )
    fig.tight_layout()
    save(fig, "fig3_eigenvalue_decomposition.png")


# ---------------------------------------------------------------------------
# Figure 4 -- Trace-determinant stability plane
# ---------------------------------------------------------------------------
def fig4_trace_determinant_plane() -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    tr = np.linspace(-4, 4, 400)
    parabola = tr ** 2 / 4

    # Background regions
    ax.fill_between(tr, parabola, 6, where=(tr < 0),
                    color=GREEN, alpha=0.18, label="Stable spiral")
    ax.fill_between(tr, parabola, 6, where=(tr > 0),
                    color=RED, alpha=0.18, label="Unstable spiral")
    ax.fill_between(tr, 0, parabola, where=(tr < 0),
                    color=GREEN, alpha=0.32, label="Stable node")
    ax.fill_between(tr, 0, parabola, where=(tr > 0),
                    color=RED, alpha=0.32, label="Unstable node")
    ax.fill_between(tr, -6, 0, color=GREY, alpha=0.18, label="Saddle (det < 0)")

    # Parabola = repeated eigenvalues; det=0 axis; tr=0 axis
    ax.plot(tr, parabola, color=PURPLE, lw=2, label=r"$\Delta=0$ (repeated)")
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1, ls="--")

    # tr=0, det>0 line = centers
    ax.plot([0, 0], [0, 6], color=BLUE, lw=2.5, label="Center (tr=0, det>0)")

    # Place sample matrices
    samples = [
        (-2.0,  3.0, "stable\nspiral"),
        ( 2.0,  3.0, "unstable\nspiral"),
        (-3.0,  1.5, "stable\nnode"),
        ( 3.0,  1.5, "unstable\nnode"),
        ( 0.5, -1.0, "saddle"),
        ( 0.0,  2.0, "center"),
    ]
    for t_, d_, label in samples:
        ax.plot(t_, d_, "o", color="black", markersize=7, zorder=5)
        ax.annotate(label, (t_, d_), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight="bold")

    ax.set_xlim(-4, 4); ax.set_ylim(-2, 6)
    ax.set_xlabel(r"trace $\tau = \mathrm{tr}\,A = \lambda_1+\lambda_2$")
    ax.set_ylabel(r"determinant $\delta = \det A = \lambda_1\lambda_2$")
    ax.set_title("Stability map in the trace–determinant plane")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig4_trace_determinant_plane.png")


# ---------------------------------------------------------------------------
# Figure 5 -- Coupled oscillators: normal modes and beats
# ---------------------------------------------------------------------------
def fig5_coupled_oscillator_modes() -> None:
    # m=1, k=k'=1, coupling kappa
    k, kappa = 1.0, 0.15
    omega_s = np.sqrt(k)                   # symmetric: in-phase
    omega_a = np.sqrt(k + 2 * kappa)       # antisymmetric: out-of-phase

    def coupled(X, t):
        x1, v1, x2, v2 = X
        a1 = -k * x1 + kappa * (x2 - x1)
        a2 = -k * x2 + kappa * (x1 - x2)
        return [v1, a1, v2, a2]

    t = np.linspace(0, 70, 4000)

    sol_sym = odeint(coupled, [1, 0,  1, 0], t)   # symmetric mode
    sol_ant = odeint(coupled, [1, 0, -1, 0], t)   # antisymmetric mode
    sol_beat = odeint(coupled, [1, 0,  0, 0], t)  # only mass 1 displaced -> beats

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(t, sol_sym[:, 0], color=BLUE, lw=1.6, label=r"$x_1$")
    ax.plot(t, sol_sym[:, 2], color=RED, lw=1.6, ls="--", label=r"$x_2$")
    ax.set_title(rf"Symmetric mode  $\omega_s={omega_s:.3f}$  (masses move together)")
    ax.set_ylabel("displacement")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-1.4, 1.4)

    ax = axes[1]
    ax.plot(t, sol_ant[:, 0], color=BLUE, lw=1.6, label=r"$x_1$")
    ax.plot(t, sol_ant[:, 2], color=RED, lw=1.6, ls="--", label=r"$x_2$")
    ax.set_title(rf"Antisymmetric mode  $\omega_a={omega_a:.3f}$  (masses move opposite)")
    ax.set_ylabel("displacement")
    ax.set_ylim(-1.4, 1.4)

    ax = axes[2]
    ax.plot(t, sol_beat[:, 0], color=BLUE, lw=1.4, label=r"$x_1$")
    ax.plot(t, sol_beat[:, 2], color=RED, lw=1.4, label=r"$x_2$")
    # beat envelope
    env = np.cos((omega_a - omega_s) / 2 * t)
    ax.plot(t, env, color=GREY, lw=1.0, ls=":", alpha=0.9)
    ax.plot(t, -env, color=GREY, lw=1.0, ls=":", alpha=0.9)
    ax.set_title("Generic initial condition: energy oscillates between masses (beats)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("displacement")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-1.4, 1.4)

    fig.tight_layout()
    save(fig, "fig5_coupled_oscillator_modes.png")


# ---------------------------------------------------------------------------
# Figure 6 -- Repeated eigenvalues: shear and degenerate node
# ---------------------------------------------------------------------------
def fig6_repeated_eigenvalue_shear() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    # Left: defective matrix with lambda = -1 (one eigenvector along x-axis)
    A_def = np.array([[-1.0, 1.0], [0.0, -1.0]])
    Y, X = np.mgrid[-3:3:25j, -3:3:25j]
    U = A_def[0, 0] * X + A_def[0, 1] * Y
    V = A_def[1, 0] * X + A_def[1, 1] * Y
    ax = axes[0]
    ax.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2),
                  cmap="Purples", density=1.3, linewidth=0.9)
    t = np.linspace(0, 8, 500)
    t_back = np.linspace(0, -3, 300)
    for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        x0 = [2.5 * np.cos(angle), 2.5 * np.sin(angle)]
        for tt in (t, t_back):
            sol = odeint(lambda x, _t: A_def @ x, x0, tt)
            ax.plot(sol[:, 0], sol[:, 1], color=BLUE, lw=0.9, alpha=0.6)
    # Eigenvector v = (1,0); generalized w = (0,1)
    ax.annotate("", xy=(2.7, 0), xytext=(-2.7, 0),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=2.4))
    ax.text(2.5, -0.35, r"eigenvector $\mathbf{v}$", color=GREEN, fontsize=10, fontweight="bold")
    ax.annotate("", xy=(0, 2.7), xytext=(0, -2.7),
                arrowprops=dict(arrowstyle="-", color=RED, lw=2.4, ls="--"))
    ax.text(0.15, 2.4, r"generalized $\mathbf{w}$", color=RED, fontsize=10, fontweight="bold")
    ax.set_title(r"Defective: $\lambda=-1$ repeated, single eigenvector  →  degenerate node")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.plot(0, 0, "o", color="black", zorder=5)

    # Right: components of the special solution x2(t) = e^{lam t}(t v + w)
    ax = axes[1]
    lam = -1.0
    v = np.array([1.0, 0.0])
    w = np.array([0.0, 1.0])
    ts = np.linspace(0, 6, 400)
    x1 = np.exp(lam * ts)[:, None] * v
    x2 = np.exp(lam * ts)[:, None] * (ts[:, None] * v + w)

    ax.plot(ts, x1[:, 0], color=GREEN, lw=2,
            label=r"$\mathbf{x}_1=e^{\lambda t}\mathbf{v}$ (component 1)")
    ax.plot(ts, x2[:, 0], color=BLUE, lw=2,
            label=r"$\mathbf{x}_2$ component 1: $t\,e^{\lambda t}$")
    ax.plot(ts, x2[:, 1], color=RED, lw=2, ls="--",
            label=r"$\mathbf{x}_2$ component 2: $e^{\lambda t}$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("component value")
    ax.set_title(r"Two independent solutions: a $te^{\lambda t}$ term appears")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Repeated eigenvalue with missing eigenvector: shear flow and the polynomial-times-exponential solution",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig6_repeated_eigenvalue_shear.png")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")
    fig1_matrix_exponential_series()
    print("  fig1 done")
    fig2_phase_portrait_zoo()
    print("  fig2 done")
    fig3_eigenvalue_decomposition()
    print("  fig3 done")
    fig4_trace_determinant_plane()
    print("  fig4 done")
    fig5_coupled_oscillator_modes()
    print("  fig5 done")
    fig6_repeated_eigenvalue_shear()
    print("  fig6 done")
    print("All figures saved.")


if __name__ == "__main__":
    main()
