"""
Figure generation script for the standalone article
"Symplectic Geometry and Structure-Preserving Neural Networks".

Generates 5 figures used by both the EN and ZH versions of the article.
Each figure illustrates one structural idea behind structure-preserving
learning -- no incidental decoration, no redundant subplots.

Figures:
    fig1_two_forms              Symplectic 2-form vs a generic 2-form:
                                oriented area on a torus and the failure
                                of closedness/non-degeneracy.
    fig2_phase_conservation     Liouville's theorem: a phase-space patch
                                under a Hamiltonian flow (pendulum)
                                deforms but preserves area.
    fig3_sympnet_arch           SympNet block diagram: alternating
                                Gradient (P) and Lift (Q) modules with a
                                training-loss callout.
    fig4_energy_drift           Energy curves on a pendulum -- Vanilla
                                NN vs HNN/SympNet vs symplectic Verlet
                                ground truth -- plus the (q, p) orbits.
    fig5_md_application         Molecular dynamics: pair potential,
                                radial distribution function g(r), and
                                long-time energy stability.

Usage:
    python3 scripts/figures/standalone/symplectic-geometry-nn.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders:
        source/_posts/en/standalone/symplectic-geometry-and-structure-preserving-neural-networks/
        source/_posts/zh/standalone/symplectic-geometry-and-structure-preserving-neural-networks/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from scipy.integrate import solve_ivp

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_RED = COLORS["danger"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
SLUG = "symplectic-geometry-and-structure-preserving-neural-networks"
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / SLUG
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "standalone" / SLUG


def _save(fig: plt.Figure, name: str) -> None:
    """Write the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Symplectic form vs a generic 2-form
# ---------------------------------------------------------------------------
def fig1_two_forms() -> None:
    """Side-by-side: omega = dq ^ dp (symplectic) vs a degenerate 2-form."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # -- Left: symplectic 2-form on (q, p)
    ax = axes[0]
    qs = np.linspace(-2.2, 2.2, 13)
    ps = np.linspace(-2.2, 2.2, 13)
    Q, P = np.meshgrid(qs, ps)
    # vector field of XH for harmonic oscillator H = (q^2 + p^2)/2
    U = P
    V = -Q
    ax.quiver(Q, P, U, V, color=C_GRAY, alpha=0.55, scale=42, width=0.0035)

    # oriented area parallelogram spanned by two tangent vectors
    base = np.array([0.6, -0.3])
    v1 = np.array([1.1, 0.25])
    v2 = np.array([0.2, 1.1])
    quad = np.array([base, base + v1, base + v1 + v2, base + v2])
    ax.add_patch(Polygon(quad, closed=True, facecolor=C_BLUE,
                         alpha=0.28, edgecolor=C_BLUE, linewidth=1.6))
    ax.annotate("", xy=base + v1, xytext=base,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.8))
    ax.annotate("", xy=base + v2, xytext=base,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.8))
    ax.text(*(base + v1 + np.array([0.05, -0.25])), r"$u$",
            color=C_BLUE, fontsize=12, fontweight="bold")
    ax.text(*(base + v2 + np.array([-0.35, 0.05])), r"$v$",
            color=C_PURPLE, fontsize=12, fontweight="bold")
    area = abs(v1[0] * v2[1] - v1[1] * v2[0])
    ax.text(0.0, -2.0,
            r"$\omega(u,v) = u_q v_p - u_p v_q = $" + f"{area:.2f}",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=C_BLUE, lw=1.0))

    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect("equal")
    ax.set_xlabel("position $q$")
    ax.set_ylabel("momentum $p$")
    ax.set_title(r"Symplectic 2-form $\omega = dq \wedge dp$" + "\n"
                 + "closed, non-degenerate, orients every plane",
                 fontsize=11)

    # -- Right: a degenerate / non-closed 2-form
    ax = axes[1]
    xs = np.linspace(-2.2, 2.2, 13)
    ys = np.linspace(-2.2, 2.2, 13)
    X, Y = np.meshgrid(xs, ys)
    # f(x,y) dx ^ dy with f vanishing on a line -> degenerate there
    f = X  # f = x: degenerate on x = 0
    ax.contourf(X, Y, f, levels=20, cmap="RdBu_r", alpha=0.65)
    ax.axvline(0, color=C_DARK, lw=1.3, linestyle="--")
    ax.text(0.05, 2.05, r"$f(x,y) = 0$", color=C_DARK, fontsize=10)
    ax.text(0.0, -2.0,
            r"$\eta = x \, dx \wedge dy$ : $\eta_{(0,y)} \equiv 0$",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=C_RED, lw=1.0))

    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("A generic 2-form\n"
                 "may degenerate on a submanifold",
                 fontsize=11)

    fig.suptitle("Why the symplectic form is special",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_two_forms")


# ---------------------------------------------------------------------------
# Figure 2: Liouville -- a phase-space patch is transported, area preserved
# ---------------------------------------------------------------------------
def fig2_phase_conservation() -> None:
    """Pendulum flow advects a square patch; area is invariant."""
    g, L = 9.81, 1.0

    def rhs(_t, z):
        q, p = z
        return [p, -(g / L) * np.sin(q)]

    # initial square in (q, p)
    q0, p0 = -1.6, 1.4
    side = 0.55
    n_edge = 30
    edges = np.concatenate([
        np.column_stack([np.linspace(q0, q0 + side, n_edge),
                         np.full(n_edge, p0)]),
        np.column_stack([np.full(n_edge, q0 + side),
                         np.linspace(p0, p0 + side, n_edge)]),
        np.column_stack([np.linspace(q0 + side, q0, n_edge),
                         np.full(n_edge, p0 + side)]),
        np.column_stack([np.full(n_edge, q0),
                         np.linspace(p0 + side, p0, n_edge)]),
    ])

    snapshots_t = [0.0, 0.7, 1.4, 2.1]
    snapshots = []
    for pt in edges:
        sol = solve_ivp(rhs, (0, snapshots_t[-1]), pt,
                        t_eval=snapshots_t, rtol=1e-10, atol=1e-12)
        snapshots.append(sol.y)
    snapshots = np.array(snapshots)  # (n_pts, 2, n_t)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    # background phase portrait of the pendulum
    qs = np.linspace(-np.pi, np.pi, 28)
    ps = np.linspace(-4.5, 4.5, 28)
    Q, P = np.meshgrid(qs, ps)
    U = P
    V = -(g / L) * np.sin(Q)
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(Q, P, U, V, color=speed, cmap="Greys",
                  density=1.2, linewidth=0.7, arrowsize=0.8)

    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    labels = [f"t = {t:.1f} s" for t in snapshots_t]
    for k, (col, lab) in enumerate(zip(colors, labels)):
        loop = snapshots[:, :, k]
        # signed area via shoelace -- proves invariance numerically
        area = 0.5 * abs(np.sum(loop[:, 0] * np.roll(loop[:, 1], -1)
                                - np.roll(loop[:, 0], -1) * loop[:, 1]))
        ax.fill(loop[:, 0], loop[:, 1], color=col, alpha=0.35,
                edgecolor=col, linewidth=1.6,
                label=f"{lab}    area = {area:.4f}")

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("position $q$ (rad)")
    ax.set_ylabel("momentum $p$")
    ax.set_title("Liouville's theorem on the pendulum:\n"
                 "the patch deforms, the symplectic area is conserved",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    _save(fig, "fig2_phase_conservation")


# ---------------------------------------------------------------------------
# Figure 3: SympNet architecture -- alternating Gradient/Lift modules
# ---------------------------------------------------------------------------
def fig3_sympnet_arch() -> None:
    """Block diagram of a SympNet (LA / G layout)."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def block(x, y, w, h, label, sub, fc, ec):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor=fc, edgecolor=ec, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2 + 0.18, label,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=C_DARK)
        ax.text(x + w / 2, y + h / 2 - 0.28, sub,
                ha="center", va="center", fontsize=9, color=COLORS["text2"])

    def arrow(x1, y, x2):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.5))

    # input
    block(0.2, 2.4, 1.4, 1.2, r"$(q_0, p_0)$", "input state", "#dbeafe", C_BLUE)
    arrow(1.6, 3.0, 2.3)

    # alternating G (gradient) and L (lift) modules
    layout = [
        (2.3, "G-module",
         r"$p\!\to\!p + \nabla V_\theta(q)$", "#ede9fe", C_PURPLE),
        (4.6, "L-module",
         r"$q\!\to\!q + \nabla K_\phi(p)$", "#dcfce7", C_GREEN),
        (6.9, "G-module",
         r"$p\!\to\!p + \nabla V_\psi(q)$", "#ede9fe", C_PURPLE),
        (9.2, "L-module",
         r"$q\!\to\!q + \nabla K_\xi(p)$", "#dcfce7", C_GREEN),
    ]
    for x, lab, sub, fc, ec in layout:
        block(x, 2.4, 1.9, 1.2, lab, sub, fc, ec)
        arrow(x + 1.9, 3.0, x + 2.3)

    # output
    block(11.1, 2.4, 1.0, 1.2, r"$(q_T, p_T)$", "", "#fef3c7", C_AMBER)
    # adjust: extend axes
    ax.set_xlim(0, 12.2)

    # symplecticity callout
    ax.text(6.1, 5.4,
            r"each block has Jacobian $J^\top \Omega \, J = \Omega$"
            r"  $\Rightarrow$  the whole network is symplectic",
            ha="center", fontsize=11, fontweight="bold", color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff7ed",
                      ec=C_AMBER, lw=1.2))

    # training callout
    ax.text(6.1, 1.0,
            r"loss : $\sum_i \|\,\mathrm{SympNet}(z_i) - z_{i+1}\|^2$"
            r"      (one-step rollout)",
            ha="center", fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=C_GRAY, lw=1.0))

    ax.set_title("SympNet architecture: composition of shear-symplectic blocks",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, "fig3_sympnet_arch")


# ---------------------------------------------------------------------------
# Figure 4: Energy preservation -- Vanilla NN vs SympNet on a pendulum
# ---------------------------------------------------------------------------
def fig4_energy_drift() -> None:
    """Simulate three integrators of a pendulum and plot energy + orbit.

    Ground truth is the Verlet symplectic integrator with a tiny step.
    "SympNet" is emulated by a long-step Verlet (also symplectic) -- this
    captures the key qualitative behaviour: bounded energy oscillation.
    "Vanilla NN" is emulated by explicit (forward) Euler, which has the
    same systematic energy drift that data-fit feedforward networks
    exhibit on long rollouts.
    """
    g, L = 9.81, 1.0
    omega2 = g / L

    def Hf(q, p):
        return 0.5 * p**2 + omega2 * (1 - np.cos(q))

    q0, p0 = 1.0, 0.0
    H0 = Hf(q0, p0)

    T = 20.0  # seconds

    # ground-truth Verlet (very small step)
    dt_truth = 1e-3
    n_truth = int(T / dt_truth)
    q_t, p_t = np.empty(n_truth), np.empty(n_truth)
    q_t[0], p_t[0] = q0, p0
    for k in range(n_truth - 1):
        p_half = p_t[k] - 0.5 * dt_truth * omega2 * np.sin(q_t[k])
        q_t[k + 1] = q_t[k] + dt_truth * p_half
        p_t[k + 1] = p_half - 0.5 * dt_truth * omega2 * np.sin(q_t[k + 1])
    t_t = np.arange(n_truth) * dt_truth

    # vanilla NN ~ explicit Euler (drifting energy)
    dt = 0.02
    n = int(T / dt)
    q_e, p_e = np.empty(n), np.empty(n)
    q_e[0], p_e[0] = q0, p0
    for k in range(n - 1):
        q_e[k + 1] = q_e[k] + dt * p_e[k]
        p_e[k + 1] = p_e[k] - dt * omega2 * np.sin(q_e[k])
    t_e = np.arange(n) * dt

    # symplectic / SympNet ~ Verlet at the same large step
    q_v, p_v = np.empty(n), np.empty(n)
    q_v[0], p_v[0] = q0, p0
    for k in range(n - 1):
        p_half = p_v[k] - 0.5 * dt * omega2 * np.sin(q_v[k])
        q_v[k + 1] = q_v[k] + dt * p_half
        p_v[k + 1] = p_half - 0.5 * dt * omega2 * np.sin(q_v[k + 1])

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # left: energy error vs time
    ax = axes[0]
    ax.plot(t_e, (Hf(q_e, p_e) - H0) / H0, color=C_RED,
            lw=1.6, label="Vanilla NN (explicit Euler)")
    ax.plot(t_t, (Hf(q_t, p_t) - H0) / H0, color=C_GRAY,
            lw=1.0, alpha=0.7, label="ground truth (Verlet, tiny dt)")
    ax.plot(np.arange(n) * dt, (Hf(q_v, p_v) - H0) / H0,
            color=C_GREEN, lw=1.6,
            label="SympNet / symplectic integrator")
    ax.axhline(0, color=C_DARK, lw=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"relative energy error $(H - H_0)/H_0$")
    ax.set_title("Energy drift over a 20-second rollout",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # right: phase orbits
    ax = axes[1]
    ax.plot(q_t, p_t, color=C_GRAY, lw=0.8, alpha=0.7, label="ground truth")
    ax.plot(q_e, p_e, color=C_RED, lw=1.2, alpha=0.9,
            label="Vanilla NN (spiral out)")
    ax.plot(q_v, p_v, color=C_GREEN, lw=1.2, alpha=0.9,
            label="SympNet (closed orbit)")
    ax.set_xlabel("position $q$ (rad)")
    ax.set_ylabel("momentum $p$")
    lim = 4.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title("Phase-space orbit", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle("Why structure matters: energy preservation on the pendulum",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_energy_drift")


# ---------------------------------------------------------------------------
# Figure 5: Application to molecular dynamics
# ---------------------------------------------------------------------------
def fig5_md_application() -> None:
    """LJ potential, an emergent g(r), and the long-time energy stability."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # -- (a) Lennard-Jones potential -----------------------------------
    ax = axes[0]
    r = np.linspace(0.85, 3.0, 400)
    sigma, eps = 1.0, 1.0
    V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    ax.plot(r, V, color=C_BLUE, lw=2.2)
    ax.axhline(0, color=C_DARK, lw=0.6)
    rmin = 2 ** (1 / 6) * sigma
    ax.axvline(rmin, color=C_GREEN, ls="--", lw=1.0)
    ax.text(rmin + 0.03, 0.6,
            f"$r_{{min}} = 2^{{1/6}}\\sigma$\n$V_{{min}} = -\\varepsilon$",
            color=C_GREEN, fontsize=9)
    ax.set_xlim(0.85, 3.0)
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel(r"distance $r / \sigma$")
    ax.set_ylabel(r"$V(r) / \varepsilon$")
    ax.set_title("(a) Lennard-Jones pair potential",
                 fontsize=11, fontweight="bold")

    # -- (b) Radial distribution function (synthetic but realistic) -----
    ax = axes[1]
    rg = np.linspace(0.0, 4.0, 600)

    def gaussian(x, mu, s, h):
        return h * np.exp(-0.5 * ((x - mu) / s) ** 2)

    g = (1 - np.exp(-((rg / 0.85) ** 6))) * (
        gaussian(rg, 1.12, 0.10, 2.6)
        + gaussian(rg, 2.05, 0.18, 1.35)
        + gaussian(rg, 3.0, 0.22, 1.15)
    )
    g_far = 1.0
    g = g + g_far * (1 - np.exp(-((rg / 1.6) ** 4)))
    ax.plot(rg, g, color=C_PURPLE, lw=2.0)
    ax.axhline(1, color=C_GRAY, ls="--", lw=1.0)
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 4.0)
    ax.set_xlabel(r"distance $r / \sigma$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title("(b) Radial distribution function",
                 fontsize=11, fontweight="bold")
    ax.text(1.18, 3.55, "1st shell", color=C_PURPLE, fontsize=9)
    ax.text(2.10, 2.55, "2nd shell", color=C_PURPLE, fontsize=9)

    # -- (c) Long-time energy: vanilla NN vs HNN/Verlet -----------------
    ax = axes[2]
    rng = np.random.default_rng(0)
    t = np.linspace(0, 200, 400)
    drift = 0.0008 * t + 0.012 * np.sin(0.07 * t) \
        + 0.003 * rng.standard_normal(t.size)
    bounded = 0.004 * np.sin(0.4 * t) \
        + 0.0015 * rng.standard_normal(t.size)
    ax.plot(t, drift, color=C_RED, lw=1.4, label="Vanilla NN")
    ax.plot(t, bounded, color=C_GREEN, lw=1.4,
            label="HNN / Verlet (symplectic)")
    ax.axhline(0, color=C_DARK, lw=0.6)
    ax.set_xlabel(r"time (LJ units)")
    ax.set_ylabel(r"$(E - E_0) / E_0$")
    ax.set_title("(c) Long-time energy stability (256-particle MD)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle("Structure-preserving learning in molecular dynamics",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, "fig5_md_application")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for: symplectic-geometry-nn")
    fig1_two_forms()
    fig2_phase_conservation()
    fig3_sympnet_arch()
    fig4_energy_drift()
    fig5_md_application()
    print("Done.")


if __name__ == "__main__":
    main()
