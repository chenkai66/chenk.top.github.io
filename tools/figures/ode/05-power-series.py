"""
Figure generation script for ODE Chapter 05:
"Power Series Solutions and Special Functions".

Generates six figures used in BOTH the EN and ZH versions of the article.
Each figure illuminates one core idea: the geometry of convergence, why
the Frobenius extension is needed, and the four families of special
functions (Bessel, Legendre, Hermite, Airy) that arise in physics.

Figures:
    fig1_radius_of_convergence    Complex-plane disk showing how the radius
                                  of convergence of a power series at an
                                  ordinary point is bounded by the nearest
                                  singularity of the coefficient functions.
    fig2_frobenius_vs_taylor      Side-by-side: a plain Taylor series fails
                                  near a regular singular point while the
                                  Frobenius ansatz y = x^r * sum(a_k x^k)
                                  succeeds.
    fig3_bessel_first_kind        First-kind Bessel functions J_0..J_4 on
                                  [0, 20], with their first zeros marked
                                  (the resonant frequencies of a circular
                                  drumhead).
    fig4_drumhead_modes           Two-dimensional vibration modes of a
                                  circular drum: J_n(k_{n,m} r) cos(n phi)
                                  for the first few (n, m) -- the physical
                                  meaning of Bessel zeros.
    fig5_legendre_polynomials     Legendre polynomials P_0..P_5 on [-1, 1]
                                  with the orthogonality interval shaded.
    fig6_qho_wavefunctions        Quantum harmonic oscillator wavefunctions
                                  psi_n built from Hermite polynomials,
                                  stacked by energy and overlaid on the
                                  parabolic potential.

Style:
    seaborn-v0_8-whitegrid, dpi=150, palette
        primary blue   #2563eb
        purple         #7c3aed
        green          #10b981
        red            #ef4444
        amber          #f59e0b
    so that the visual identity matches the rest of the chenk-site
    mathematics articles.

Usage:
    python3 scripts/figures/ode/05-power-series.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay consistent across languages. Parent folders
    are created if missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.special import airy, eval_hermite, eval_legendre, jn_zeros, jv

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

COLOR_BLUE = COLORS["primary"]
COLOR_PURPLE = COLORS["accent"]
COLOR_GREEN = COLORS["success"]
COLOR_RED = COLORS["danger"]
COLOR_AMBER = COLORS["warning"]
COLOR_SLATE = COLORS["text2"]

PALETTE = [COLOR_BLUE, COLOR_PURPLE, COLOR_GREEN, COLOR_RED, COLOR_AMBER]

DPI = 150

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "ode" / "05-laplace-transform"
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ode"
    / "05-级数解法与特殊函数"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to both EN and ZH asset folders at production DPI."""
    for folder in (EN_DIR, ZH_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig1: radius of convergence on the complex plane
# ---------------------------------------------------------------------------


def fig1_radius_of_convergence() -> None:
    """Disk of convergence bounded by nearest singularity in the complex plane.

    Toy ODE: y'' + 1/(1 + x^2) y' + ... has coefficient with singularities
    at x = +- i.  Around the ordinary point x_0 = 0, the series solution is
    therefore guaranteed to converge in |x| < 1.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # Disk of guaranteed convergence
    disk = Circle(
        (0, 0),
        1.0,
        facecolor=COLOR_BLUE,
        edgecolor=COLOR_BLUE,
        alpha=0.12,
        linewidth=2.0,
    )
    ax.add_patch(disk)
    ax.add_patch(
        Circle(
            (0, 0),
            1.0,
            facecolor="none",
            edgecolor=COLOR_BLUE,
            linewidth=2.2,
            linestyle="--",
        )
    )

    # Singularities at +- i
    ax.scatter(
        [0, 0],
        [1, -1],
        s=140,
        color=COLOR_RED,
        zorder=5,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.annotate(
        r"singularity  $x = i$",
        xy=(0, 1),
        xytext=(0.55, 1.25),
        fontsize=11,
        color=COLOR_RED,
        arrowprops=dict(arrowstyle="-", color=COLOR_RED, lw=0.8),
    )
    ax.annotate(
        r"singularity  $x = -i$",
        xy=(0, -1),
        xytext=(0.55, -1.35),
        fontsize=11,
        color=COLOR_RED,
        arrowprops=dict(arrowstyle="-", color=COLOR_RED, lw=0.8),
    )

    # Expansion point
    ax.scatter(
        [0],
        [0],
        s=120,
        color=COLOR_PURPLE,
        zorder=6,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.annotate(
        r"expansion point  $x_0 = 0$",
        xy=(0, 0),
        xytext=(-1.85, -0.35),
        fontsize=11,
        color=COLOR_PURPLE,
    )

    # Radius arrow
    arrow = FancyArrowPatch(
        (0, 0),
        (1.0, 0),
        arrowstyle="->",
        mutation_scale=18,
        color=COLOR_SLATE,
        linewidth=2.0,
    )
    ax.add_patch(arrow)
    ax.text(0.45, 0.10, r"$R = 1$", fontsize=13, color=COLOR_SLATE)

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$\mathrm{Re}\,x$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Im}\,x$", fontsize=12)
    ax.set_title(
        "Radius of Convergence Reaches the Nearest Singularity",
        fontsize=13,
        pad=10,
    )

    fig.tight_layout()
    _save(fig, "fig1_radius_of_convergence")


# ---------------------------------------------------------------------------
# fig2: Frobenius ansatz versus plain Taylor series
# ---------------------------------------------------------------------------


def fig2_frobenius_vs_taylor() -> None:
    """Why the Frobenius extension is needed at a regular singular point.

    Demo ODE: 4 x^2 y'' + y = 0, indicial roots r = 1/2, 1/2.
    True solution branch: y = sqrt(x).  A plain power series sum a_k x^k
    cannot represent sqrt(x); Frobenius series x^{1/2} sum a_k x^k can.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    x = np.linspace(0, 1.5, 500)
    true = np.sqrt(np.maximum(x, 0))

    # Best degree-N Taylor polynomial of sqrt(x) about x = 1, evaluated near 0:
    # we instead show that *any* polynomial through the origin with finite
    # derivative cannot match the vertical tangent of sqrt(x).
    taylor_low = x  # a_1 x
    taylor_mid = x - 0.5 * x**2 + 0.5 * x**3  # arbitrary polynomial
    frobenius = np.sqrt(np.maximum(x, 0))

    axL = axes[0]
    axL.plot(x, true, color="black", linewidth=2.4, label=r"true $y=\sqrt{x}$")
    axL.plot(
        x,
        taylor_low,
        color=COLOR_RED,
        linewidth=2.0,
        linestyle="--",
        label=r"Taylor: $y=x$",
    )
    axL.plot(
        x,
        taylor_mid,
        color=COLOR_AMBER,
        linewidth=2.0,
        linestyle="--",
        label=r"Taylor: $x-\frac{1}{2} x^2+\frac{1}{2} x^3$",
    )
    axL.set_title("Plain Taylor Series Misses the Cusp", fontsize=12)
    axL.set_xlabel(r"$x$", fontsize=11)
    axL.set_ylabel(r"$y(x)$", fontsize=11)
    axL.set_xlim(0, 1.5)
    axL.set_ylim(-0.05, 1.4)
    axL.legend(fontsize=10, loc="lower right")

    axR = axes[1]
    axR.plot(x, true, color="black", linewidth=2.4, label=r"true $y=\sqrt{x}$")
    axR.plot(
        x,
        frobenius,
        color=COLOR_BLUE,
        linewidth=2.4,
        linestyle=":",
        label=r"Frobenius: $x^{1/2}\sum a_k x^k$",
    )
    axR.set_title("Frobenius Ansatz Captures It", fontsize=12)
    axR.set_xlabel(r"$x$", fontsize=11)
    axR.set_xlim(0, 1.5)
    axR.legend(fontsize=10, loc="lower right")

    fig.suptitle(
        r"Regular singular point at $x=0$:  $4x^2 y''+y=0$,  indicial root $r=\frac{1}{2}$",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig2_frobenius_vs_taylor")


# ---------------------------------------------------------------------------
# fig3: Bessel functions of the first kind with their zeros
# ---------------------------------------------------------------------------


def fig3_bessel_first_kind() -> None:
    """J_0..J_4 on [0, 20] with the first zeros highlighted."""
    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    x = np.linspace(0.0, 20.0, 2000)
    for n in range(5):
        ax.plot(
            x,
            jv(n, x),
            color=PALETTE[n],
            linewidth=2.0,
            label=rf"$J_{{{n}}}(x)$",
        )

    # Mark first 4 zeros of J_0 -- these set the modes of a circular drum
    zeros_j0 = jn_zeros(0, 4)
    ax.scatter(
        zeros_j0,
        np.zeros_like(zeros_j0),
        s=70,
        color=COLOR_BLUE,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    for k, z in enumerate(zeros_j0, start=1):
        ax.annotate(
            rf"$j_{{0,{k}}}$",
            xy=(z, 0),
            xytext=(z, -0.18),
            ha="center",
            fontsize=9,
            color=COLOR_BLUE,
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.55, 1.05)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$J_n(x)$", fontsize=12)
    ax.set_title(
        "Bessel Functions of the First Kind  (zeros set the drumhead modes)",
        fontsize=13,
        pad=10,
    )
    ax.legend(fontsize=10, ncol=5, loc="upper right")

    fig.tight_layout()
    _save(fig, "fig3_bessel_first_kind")


# ---------------------------------------------------------------------------
# fig4: 2D drumhead vibration modes
# ---------------------------------------------------------------------------


def fig4_drumhead_modes() -> None:
    """Vibration modes of a unit circular drum:
    u_{n,m}(r, phi) = J_n(j_{n,m} r) * cos(n phi).
    """
    modes = [(0, 1), (0, 2), (1, 1), (2, 1)]  # (n, m)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))

    r = np.linspace(0, 1, 220)
    phi = np.linspace(0, 2 * np.pi, 360)
    R, PHI = np.meshgrid(r, phi)
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)

    for ax, (n, m) in zip(axes, modes):
        zeros = jn_zeros(n, m)
        k = zeros[m - 1]
        U = jv(n, k * R) * np.cos(n * PHI)
        # Normalise for symmetric colour scale
        vmax = np.max(np.abs(U))
        cs = ax.contourf(
            X,
            Y,
            U,
            levels=21,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.contour(
            X,
            Y,
            U,
            levels=[0],
            colors="black",
            linewidths=0.7,
            linestyles="--",
        )
        # Boundary
        circle = Circle(
            (0, 0),
            1.0,
            facecolor="none",
            edgecolor="black",
            linewidth=1.6,
        )
        ax.add_patch(circle)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"mode (n={n}, m={m})", fontsize=11)

    fig.suptitle(
        "Vibration Modes of a Circular Drumhead  "
        r"$u_{n,m}(r,\varphi)=J_n(j_{n,m}\,r)\cos(n\varphi)$",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig4_drumhead_modes")


# ---------------------------------------------------------------------------
# fig5: Legendre polynomials with orthogonality interval
# ---------------------------------------------------------------------------


def fig5_legendre_polynomials() -> None:
    """P_0..P_5 on [-1, 1] with the orthogonality interval shaded."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    x = np.linspace(-1, 1, 600)
    for n in range(6):
        color = PALETTE[n % len(PALETTE)]
        ax.plot(
            x,
            eval_legendre(n, x),
            color=color,
            linewidth=2.0,
            label=rf"$P_{{{n}}}(x)$",
        )

    # Orthogonality interval shading
    ax.axvspan(-1, 1, color=COLOR_BLUE, alpha=0.05)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.15, 1.25)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$P_n(x)$", fontsize=12)
    ax.set_title(
        r"Legendre Polynomials  (orthogonal on $[-1,1]$ with weight $w=1$)",
        fontsize=13,
        pad=10,
    )
    ax.legend(fontsize=10, ncol=3, loc="lower right")

    fig.tight_layout()
    _save(fig, "fig5_legendre_polynomials")


# ---------------------------------------------------------------------------
# fig6: quantum harmonic oscillator eigenstates
# ---------------------------------------------------------------------------


def fig6_qho_wavefunctions() -> None:
    """Hermite-based wavefunctions of the quantum harmonic oscillator,
    stacked at their eigenenergies on top of the parabolic potential.
    """
    from math import factorial

    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    x = np.linspace(-5, 5, 800)
    V = 0.5 * x**2  # potential in units where omega = m = hbar = 1

    ax.plot(x, V, color=COLOR_SLATE, linewidth=1.6, label=r"potential $V(x)=\frac{1}{2} x^2$")

    n_max = 5
    scale = 0.55  # visual scale for psi atop the potential
    for n in range(n_max):
        norm = 1.0 / np.sqrt(2**n * factorial(n)) * (1.0 / np.pi) ** 0.25
        psi = norm * eval_hermite(n, x) * np.exp(-(x**2) / 2)
        E = n + 0.5
        color = PALETTE[n % len(PALETTE)]
        ax.plot(x, scale * psi + E, color=color, linewidth=2.0)
        ax.fill_between(x, E, scale * psi + E, color=color, alpha=0.18)
        ax.axhline(E, color=color, linewidth=0.7, linestyle="--", alpha=0.6)
        ax.text(
            -4.85,
            E + 0.07,
            rf"$n={n},\ E_n={n}+\frac{{1}}{{2}}$",
            fontsize=10,
            color=color,
        )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.2, n_max + 0.6)
    ax.set_xlabel(r"position $x$", fontsize=12)
    ax.set_ylabel(r"energy / wavefunction (offset)", fontsize=12)
    ax.set_title(
        r"Quantum Harmonic Oscillator: Hermite Eigenstates $\psi_n(x)$",
        fontsize=13,
        pad=10,
    )
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    _save(fig, "fig6_qho_wavefunctions")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    fig1_radius_of_convergence()
    fig2_frobenius_vs_taylor()
    fig3_bessel_first_kind()
    fig4_drumhead_modes()
    fig5_legendre_polynomials()
    fig6_qho_wavefunctions()
    print(f"Wrote 6 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
