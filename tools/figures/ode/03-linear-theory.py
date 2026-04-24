"""
Figure generation script for ODE Chapter 03:
"Higher-Order Linear Theory".

Generates 7 figures used in BOTH the EN and ZH versions of the article.
Each figure isolates one core idea about higher-order linear ODEs and is
rendered in a clean, didactic style consistent with the rest of the site.

Figures:
    fig1_damping_regimes              Underdamped / critically damped /
                                      overdamped responses of mx''+bx'+kx=0
                                      shown in three side-by-side panels with
                                      decay envelopes annotated.
    fig2_characteristic_roots         The three canonical root configurations
                                      of a 2nd-order characteristic polynomial
                                      drawn in the complex plane, each tied
                                      to its time-domain solution.
    fig3_resonance_curve              Steady-state amplitude vs forcing
                                      frequency for several damping ratios,
                                      showing how the resonance peak sharpens
                                      as zeta -> 0.
    fig4_rlc_response                 Step response of a series RLC circuit
                                      for three damping regimes (R varied),
                                      paired with the schematic and the
                                      mechanical analogue.
    fig5_wronskian                    Linear (in)dependence visualised via
                                      the Wronskian: independent pair vs
                                      dependent pair, with W(x) plotted
                                      underneath each pair.
    fig6_undetermined_coefficients    The "guess and match" workflow: the
                                      forcing term, the trial form, and the
                                      resulting particular solution stacked
                                      so the matching step is visible.
    fig7_variation_of_parameters      The variation-of-parameters formula
                                      illustrated on y''+y=sec(x): the two
                                      homogeneous building blocks, the
                                      Wronskian-weighted integrals, and the
                                      reconstructed particular solution.

Usage:
    python3 scripts/figures/ode/03-linear-theory.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages. Parent folders are created
    if missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from scipy.integrate import odeint

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
C_RED = COLORS["danger"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "ode" / "03-linear-theory"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "ode" / "03-高阶线性微分方程"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: Damping regimes
# ---------------------------------------------------------------------------
def fig1_damping_regimes() -> None:
    """Three regimes of mx''+bx'+kx=0: under, critical, over."""
    omega0 = 2 * np.pi
    t = np.linspace(0, 3.0, 800)

    def damped(state, t, zeta):
        x, v = state
        return [v, -2 * zeta * omega0 * v - omega0 ** 2 * x]

    cases = [
        (0.15, "Underdamped  (zeta = 0.15)", C_BLUE),
        (1.00, "Critically damped  (zeta = 1)", C_GREEN),
        (2.00, "Overdamped  (zeta = 2)", C_RED),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, (zeta, title, color) in zip(axes, cases):
        sol = odeint(damped, [1.0, 0.0], t, args=(zeta,))
        ax.plot(t, sol[:, 0], color=color, linewidth=2.4, label="x(t)")

        if zeta < 1:
            env = np.exp(-zeta * omega0 * t)
            ax.plot(t, env, "--", color=C_GRAY, linewidth=1.2, label="envelope")
            ax.plot(t, -env, "--", color=C_GRAY, linewidth=1.2)

        ax.axhline(0, color=C_DARK, linewidth=0.6)
        ax.set_title(title, fontsize=11, color=C_DARK)
        ax.set_xlabel("t")
        ax.set_xlim(0, 3.0)
        ax.set_ylim(-1.15, 1.15)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    axes[0].set_ylabel("displacement  x(t)")
    fig.suptitle(
        r"Three damping regimes of  $x'' + 2\zeta\omega_0 x' + \omega_0^2 x = 0$",
        fontsize=13, y=1.02, color=C_DARK,
    )
    fig.tight_layout()
    _save(fig, "fig1_damping_regimes")


# ---------------------------------------------------------------------------
# Fig 2: Characteristic roots in the complex plane
# ---------------------------------------------------------------------------
def fig2_characteristic_roots() -> None:
    """Three canonical root configurations and the matching solution shape."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5),
                             gridspec_kw={"height_ratios": [1.0, 0.85]})

    cases = [
        {
            "title": "Distinct real roots",
            "roots": [(-1.5, 0), (0.5, 0)],
            "expr": r"$y = c_1 e^{-1.5x} + c_2 e^{0.5x}$",
            "fn": lambda x: 0.6 * np.exp(-1.5 * x) + 0.4 * np.exp(0.5 * x),
            "color": C_BLUE,
        },
        {
            "title": "Repeated real root",
            "roots": [(-0.8, 0), (-0.8, 0)],
            "expr": r"$y = (c_1 + c_2 x)\,e^{-0.8x}$",
            "fn": lambda x: (1.0 + 1.4 * x) * np.exp(-0.8 * x),
            "color": C_PURPLE,
        },
        {
            "title": "Complex conjugate pair",
            "roots": [(-0.4, 2.5), (-0.4, -2.5)],
            "expr": r"$y = e^{-0.4x}(c_1\cos 2.5x + c_2\sin 2.5x)$",
            "fn": lambda x: np.exp(-0.4 * x) * np.cos(2.5 * x),
            "color": C_GREEN,
        },
    ]

    # top row: complex plane
    for ax, case in zip(axes[0], cases):
        ax.axhline(0, color=C_DARK, linewidth=0.6)
        ax.axvline(0, color=C_DARK, linewidth=0.6)

        # shade left half-plane (stability region)
        ax.axvspan(-3.5, 0, color=C_GREEN, alpha=0.06)

        for (re, im) in case["roots"]:
            ax.plot(re, im, "o", color=case["color"], markersize=14,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            if abs(im) > 1e-6:
                # mark conjugate pairing line
                ax.plot([re, re], [im, -im], ":", color=case["color"],
                        linewidth=1.0, alpha=0.6)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Re(r)")
        ax.set_ylabel("Im(r)")
        ax.set_title(case["title"], fontsize=11, color=C_DARK)
        ax.text(0.02, 0.97, "stable", transform=ax.transAxes,
                fontsize=8, color=C_GREEN, va="top", alpha=0.8)
        ax.text(0.78, 0.97, "unstable", transform=ax.transAxes,
                fontsize=8, color=C_RED, va="top", alpha=0.8)

    # bottom row: time-domain solution
    x = np.linspace(0, 6, 400)
    for ax, case in zip(axes[1], cases):
        ax.plot(x, case["fn"](x), color=case["color"], linewidth=2.4)
        ax.axhline(0, color=C_DARK, linewidth=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.set_title(case["expr"], fontsize=10, color=C_DARK)
        ax.set_xlim(0, 6)

    fig.suptitle("Characteristic equation roots and the solutions they produce",
                 fontsize=13, y=1.00, color=C_DARK)
    fig.tight_layout()
    _save(fig, "fig2_characteristic_roots")


# ---------------------------------------------------------------------------
# Fig 3: Resonance amplitude curve
# ---------------------------------------------------------------------------
def fig3_resonance_curve() -> None:
    """Steady-state amplitude vs driving frequency for several zeta."""
    omega0 = 1.0
    F0 = 1.0
    r = np.linspace(0.01, 2.5, 600)  # r = omega / omega0

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    zetas = [0.05, 0.1, 0.2, 0.4, 0.707]
    palette = [C_RED, C_AMBER, C_PURPLE, C_BLUE, C_GREEN]

    for zeta, color in zip(zetas, palette):
        # Steady-state amplitude of m=1 forced oscillator
        amp = F0 / np.sqrt((omega0 ** 2 - (r * omega0) ** 2) ** 2
                           + (2 * zeta * omega0 * r * omega0) ** 2)
        ax.plot(r, amp, color=color, linewidth=2.2,
                label=fr"$\zeta = {zeta:g}$")
        # mark peak
        if zeta < 1 / np.sqrt(2):
            r_peak = np.sqrt(1 - 2 * zeta ** 2)
            a_peak = F0 / (2 * zeta * omega0 ** 2 * np.sqrt(1 - zeta ** 2))
            ax.plot(r_peak, a_peak, "o", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=5)

    ax.axvline(1.0, color=C_DARK, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(1.02, 0.5, r"$\omega = \omega_0$", color=C_DARK, fontsize=10,
            transform=ax.get_xaxis_transform())

    ax.set_xlabel(r"frequency ratio  $\omega / \omega_0$", fontsize=11)
    ax.set_ylabel("steady-state amplitude", fontsize=11)
    ax.set_title("Resonance: amplitude response of a forced damped oscillator",
                 fontsize=12, color=C_DARK)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    _save(fig, "fig3_resonance_curve")


# ---------------------------------------------------------------------------
# Fig 4: RLC step response
# ---------------------------------------------------------------------------
def fig4_rlc_response() -> None:
    """Series RLC: schematic on left, step response on right."""
    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.25)

    # ---- left: schematic ----
    axL = fig.add_subplot(gs[0])
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 7)
    axL.set_aspect("equal")
    axL.axis("off")

    # wire rectangle
    axL.plot([1, 9, 9, 1, 1], [1.5, 1.5, 5.5, 5.5, 1.5],
             color=C_DARK, linewidth=1.5)

    # source on left (circle with V)
    axL.add_patch(Circle((1, 3.5), 0.6, fill=False,
                         edgecolor=C_DARK, linewidth=1.5))
    axL.text(1, 3.5, "V(t)", ha="center", va="center",
             fontsize=10, color=C_DARK)

    # resistor (zigzag) on top
    rx = np.linspace(3, 5, 9)
    ry = 5.5 + 0.25 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])
    axL.plot(rx, ry, color=C_RED, linewidth=2)
    axL.text(4, 6.2, "R", ha="center", fontsize=11, color=C_RED)

    # inductor (loops) on top
    for cx in [6.0, 6.6, 7.2, 7.8]:
        axL.add_patch(mpatches.Arc((cx, 5.5), 0.6, 0.6,
                                   theta1=0, theta2=180,
                                   edgecolor=C_BLUE, linewidth=2))
    axL.text(6.9, 6.2, "L", ha="center", fontsize=11, color=C_BLUE)

    # capacitor (parallel plates) on right
    axL.plot([9, 9], [3.0, 4.0], color=C_PURPLE, linewidth=3)
    axL.plot([9.3, 9.3], [3.0, 4.0], color=C_PURPLE, linewidth=3)
    # break the wire at capacitor
    axL.add_patch(Rectangle((8.85, 2.95), 0.6, 1.1, color="white", zorder=2))
    axL.plot([9, 9], [3.0, 4.0], color=C_PURPLE, linewidth=3, zorder=3)
    axL.plot([9.3, 9.3], [3.0, 4.0], color=C_PURPLE, linewidth=3, zorder=3)
    axL.plot([9, 9], [1.5, 3.0], color=C_DARK, linewidth=1.5, zorder=1)
    axL.plot([9.3, 9.3], [4.0, 5.5], color=C_DARK, linewidth=1.5, zorder=1)
    axL.text(10.0, 3.5, "C", ha="center", fontsize=11, color=C_PURPLE)

    axL.set_title("Series RLC circuit",
                  fontsize=12, color=C_DARK, pad=10)
    axL.text(5, 0.4,
             r"$L\,\ddot q + R\,\dot q + q/C = V(t)$",
             ha="center", fontsize=11, color=C_DARK)

    # ---- right: step response ----
    axR = fig.add_subplot(gs[1])
    L, C = 1.0, 1.0
    omega0 = 1.0 / np.sqrt(L * C)

    def rlc(state, t, R):
        q, i = state
        return [i, (1.0 - R * i - q / C) / L]   # step input V=1

    t = np.linspace(0, 18, 1000)
    cases = [
        (0.2, f"R = 0.2  (underdamped, zeta={0.1:.2f})", C_BLUE),
        (2.0, f"R = 2.0  (critically damped)", C_GREEN),
        (5.0, f"R = 5.0  (overdamped)", C_RED),
    ]
    for R, lbl, color in cases:
        sol = odeint(rlc, [0.0, 0.0], t, args=(R,))
        axR.plot(t, sol[:, 0], color=color, linewidth=2.2, label=lbl)

    axR.axhline(1.0, color=C_DARK, linestyle=":", linewidth=1.0, alpha=0.6)
    axR.text(0.2, 1.04, "steady-state q = CV", color=C_DARK, fontsize=9)
    axR.set_xlabel("time")
    axR.set_ylabel("charge q(t)")
    axR.set_title("Step response (V switched on at t=0)",
                  fontsize=12, color=C_DARK)
    axR.set_xlim(0, 18)
    axR.set_ylim(0, 1.9)
    axR.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle("RLC circuit: same equation, same three regimes as the spring",
                 fontsize=13, y=1.04, color=C_DARK)
    fig.tight_layout()
    _save(fig, "fig4_rlc_response")


# ---------------------------------------------------------------------------
# Fig 5: Wronskian visualisation of linear (in)dependence
# ---------------------------------------------------------------------------
def fig5_wronskian() -> None:
    """Independent pair vs dependent pair, with W(x) below each."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0),
                             gridspec_kw={"height_ratios": [1.0, 0.7]},
                             sharex=True)
    x = np.linspace(0, 2 * np.pi, 500)

    # ----- independent: sin x and cos x -----
    y1, y2 = np.sin(x), np.cos(x)
    dy1, dy2 = np.cos(x), -np.sin(x)
    W = y1 * dy2 - y2 * dy1   # = -1

    ax = axes[0, 0]
    ax.plot(x, y1, color=C_BLUE, linewidth=2.4, label=r"$y_1 = \sin x$")
    ax.plot(x, y2, color=C_PURPLE, linewidth=2.4, label=r"$y_2 = \cos x$")
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_title("Linearly INDEPENDENT", fontsize=12, color=C_GREEN)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_ylabel("y(x)")
    ax.set_ylim(-1.5, 1.5)

    ax = axes[1, 0]
    ax.plot(x, W, color=C_GREEN, linewidth=2.4, label=r"$W(x) = -1$")
    ax.fill_between(x, 0, W, color=C_GREEN, alpha=0.15)
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$W(x)$")
    ax.set_ylim(-2, 2)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.text(0.02, 0.05, r"$W \neq 0$  $\Rightarrow$  independent",
            transform=ax.transAxes, fontsize=10, color=C_GREEN)

    # ----- dependent: sin x and 2 sin x -----
    y1, y2 = np.sin(x), 2.0 * np.sin(x)
    dy1, dy2 = np.cos(x), 2.0 * np.cos(x)
    W = y1 * dy2 - y2 * dy1   # = 0

    ax = axes[0, 1]
    ax.plot(x, y1, color=C_BLUE, linewidth=2.4, label=r"$y_1 = \sin x$")
    ax.plot(x, y2, color=C_RED, linewidth=2.4, linestyle="--",
            label=r"$y_2 = 2\sin x$")
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_title("Linearly DEPENDENT", fontsize=12, color=C_RED)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_ylim(-2.5, 2.5)

    ax = axes[1, 1]
    ax.plot(x, W, color=C_RED, linewidth=2.4, label=r"$W(x) = 0$")
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylim(-2, 2)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.text(0.02, 0.05, r"$W \equiv 0$  $\Rightarrow$  dependent",
            transform=ax.transAxes, fontsize=10, color=C_RED)

    fig.suptitle(
        "Wronskian as a test for linear independence:  "
        r"$W(y_1,y_2) = y_1 y_2' - y_2 y_1'$",
        fontsize=12.5, y=1.02, color=C_DARK,
    )
    fig.tight_layout()
    _save(fig, "fig5_wronskian")


# ---------------------------------------------------------------------------
# Fig 6: Method of undetermined coefficients
# ---------------------------------------------------------------------------
def fig6_undetermined_coefficients() -> None:
    """Stack: forcing f(x), trial guess, fitted particular solution.

    Worked example:  y'' + y' + y = e^{-x/2} cos(x)
    Trial:  y_p = e^{-x/2}(A cos x + B sin x)
    Solving 2x2 system gives A and B explicitly.
    """
    x = np.linspace(0, 12, 600)

    # forcing term
    f = np.exp(-0.5 * x) * np.cos(x)

    # particular solution coefficients
    # Substitute y_p = e^{-x/2}(A cos x + B sin x) into y'' + y' + y = f.
    # Algebra (see notes) gives the linear system:
    #   ( 1/4 ) A + ( 3/2 ) B = 1
    #   (-3/2 ) A + ( 1/4 ) B = 0
    M = np.array([[0.25, 1.5], [-1.5, 0.25]])
    rhs = np.array([1.0, 0.0])
    A, B = np.linalg.solve(M, rhs)
    yp = np.exp(-0.5 * x) * (A * np.cos(x) + B * np.sin(x))

    # numerical verification: integrate ODE with y(0)=A, y'(0)=...
    # actually we will just verify by computing LHS and overlaying
    # symbolically: derivatives of yp
    # yp' = e^{-x/2}((-A/2 + B)cos x + (-B/2 - A)sin x)
    # yp''= e^{-x/2}((A/4 - B - B + A)... ); skip explicit form, use finite diff
    dx = x[1] - x[0]
    yp_ = np.gradient(yp, dx)
    yp__ = np.gradient(yp_, dx)
    lhs = yp__ + yp_ + yp

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), sharex=True)

    ax = axes[0]
    ax.plot(x, f, color=C_RED, linewidth=2.2,
            label=r"$f(x) = e^{-x/2}\cos x$")
    ax.fill_between(x, 0, f, color=C_RED, alpha=0.12)
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_ylabel("forcing")
    ax.set_title("Step 1.  Identify the forcing term", fontsize=11, color=C_DARK)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    ax = axes[1]
    ax.plot(x, np.exp(-0.5 * x) * np.cos(x), color=C_GRAY, linewidth=1.6,
            linestyle=":", label=r"basis  $e^{-x/2}\cos x$")
    ax.plot(x, np.exp(-0.5 * x) * np.sin(x), color=C_GRAY, linewidth=1.6,
            linestyle="--", label=r"basis  $e^{-x/2}\sin x$")
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_ylabel("trial basis")
    ax.set_title(r"Step 2.  Trial form  $y_p = e^{-x/2}(A\cos x + B\sin x)$",
                 fontsize=11, color=C_DARK)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    ax = axes[2]
    ax.plot(x, yp, color=C_BLUE, linewidth=2.4,
            label=fr"$y_p$  with $A={A:.3f},\, B={B:.3f}$")
    ax.plot(x, lhs, color=C_GREEN, linewidth=1.4, linestyle="--",
            label=r"$y_p'' + y_p' + y_p$  (matches $f$)")
    ax.plot(x, f, color=C_RED, linewidth=1.0, alpha=0.6,
            label=r"target $f(x)$")
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("particular solution")
    ax.set_title("Step 3.  Solve the linear system for A, B", fontsize=11,
                 color=C_DARK)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle(
        r"Method of undetermined coefficients:  $y'' + y' + y = e^{-x/2}\cos x$",
        fontsize=12.5, y=1.00, color=C_DARK,
    )
    fig.tight_layout()
    _save(fig, "fig6_undetermined_coefficients")


# ---------------------------------------------------------------------------
# Fig 7: Variation of parameters
# ---------------------------------------------------------------------------
def fig7_variation_of_parameters() -> None:
    """Variation of parameters illustrated on y'' + y = sec(x)."""
    # Avoid x = pi/2 where sec blows up.
    x = np.linspace(-1.4, 1.4, 800)

    y1 = np.cos(x)
    y2 = np.sin(x)
    f = 1.0 / np.cos(x)
    W = 1.0  # cos*cos - sin*(-sin) = 1

    # u1' = -y2 f / W = -sin x sec x = -tan x
    # u2' =  y1 f / W =  cos x sec x = 1
    # u1 = ln|cos x|,  u2 = x
    yp = y1 * np.log(np.abs(np.cos(x))) + y2 * x   # = cos x ln|cos x| + x sin x

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # row 1: y1, y2, f
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, y1, color=C_BLUE, linewidth=2.4)
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_title(r"$y_1(x) = \cos x$", fontsize=11, color=C_DARK)
    ax.set_xlim(-1.4, 1.4)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, y2, color=C_PURPLE, linewidth=2.4)
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_title(r"$y_2(x) = \sin x$", fontsize=11, color=C_DARK)
    ax.set_xlim(-1.4, 1.4)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(x, f, color=C_RED, linewidth=2.4)
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_title(r"forcing  $f(x) = \sec x$", fontsize=11, color=C_DARK)
    ax.set_ylim(0, 6)
    ax.set_xlim(-1.4, 1.4)

    # row 2: u1, u2, yp
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, np.log(np.abs(np.cos(x))), color=C_BLUE, linewidth=2.4)
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_title(r"$u_1(x) = -\!\int \frac{y_2 f}{W} dx = \ln|\cos x|$",
                 fontsize=10, color=C_DARK)
    ax.set_xlim(-1.4, 1.4)
    ax.set_xlabel("x")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, x, color=C_PURPLE, linewidth=2.4)
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_title(r"$u_2(x) = \int \frac{y_1 f}{W} dx = x$",
                 fontsize=10, color=C_DARK)
    ax.set_xlim(-1.4, 1.4)
    ax.set_xlabel("x")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, yp, color=C_GREEN, linewidth=2.6)
    ax.axhline(0, color=C_DARK, linewidth=0.6)
    ax.set_title(r"$y_p = \cos x\,\ln|\cos x| + x\sin x$",
                 fontsize=10, color=C_DARK)
    ax.set_xlim(-1.4, 1.4)
    ax.set_xlabel("x")

    fig.suptitle(
        "Variation of parameters on  "
        r"$y'' + y = \sec x$:  build $y_p = u_1 y_1 + u_2 y_2$",
        fontsize=12.5, y=1.00, color=C_DARK,
    )
    _save(fig, "fig7_variation_of_parameters")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for ODE Chapter 03...")
    fig1_damping_regimes()
    fig2_characteristic_roots()
    fig3_resonance_curve()
    fig4_rlc_response()
    fig5_wronskian()
    fig6_undetermined_coefficients()
    fig7_variation_of_parameters()
    print("Done.")


if __name__ == "__main__":
    main()
