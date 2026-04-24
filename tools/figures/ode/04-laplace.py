"""
Figure generation script for ODE Chapter 04: "The Laplace Transform".

Generates 6 figures used in BOTH the EN and ZH versions of the article.
Each figure conveys one core idea about the Laplace transform / control
analysis, with a consistent typography, palette and DPI across the series.

Figures:
    fig1_step_impulse_kernel    The step u(t), Dirac delta delta(t), and the
                                Laplace probe e^{-st} that turns time-domain
                                signals into the s-domain.
    fig2_pole_zero_responses    Pole locations in the complex s-plane mapped
                                directly to their characteristic impulse
                                responses (decay, growth, damped oscillation,
                                sustained oscillation, growing oscillation).
    fig3_partial_fractions      Inverse Laplace via partial fractions: a
                                single rational F(s) = (3s+5)/((s+1)(s+2))
                                decomposes into two simple poles, and the
                                corresponding time-domain components add up
                                to the full y(t).
    fig4_transfer_function      Step response and Bode magnitude/phase of
                                three second-order systems (under-damped,
                                critically damped, over-damped) -- the same
                                physical content seen in time and frequency.
    fig5_resonance_buildup      Forced undamped oscillator at resonance:
                                amplitude grows linearly as t/(2 omega_0).
                                Compares forcing on-resonance vs slightly
                                off-resonance to make the t-factor visible.
    fig6_pid_control            PID closed-loop step response of a second-
                                order plant. Shows P-only (steady-state
                                error), PI (eliminates error, slower), and
                                tuned PID (fast, no overshoot).

Usage:
    python3 scripts/figures/ode/04-laplace.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages. Parent folders are created
    if missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy import signal

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
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]

DPI = 150

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": C_DARK,
    "axes.linewidth": 0.8,
    "xtick.color": C_DARK,
    "ytick.color": C_DARK,
    "grid.color": C_LIGHT,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 10,
})

# ---------------------------------------------------------------------------
# Output paths -- script writes to BOTH language asset folders
# ---------------------------------------------------------------------------
ROOT = Path("/Users/kchen/Desktop/Project/chenk-site/source/_posts")
EN_DIR = ROOT / "en" / "ode" / "04-constant-coefficients"
ZH_DIR = ROOT / "zh" / "ode" / "04-拉普拉斯变换"

EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIRS = [EN_DIR, ZH_DIR]


def save(fig: plt.Figure, name: str) -> None:
    """Save the same PNG into every output directory."""
    for d in OUT_DIRS:
        path = d / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig1: step, delta, and the Laplace probe e^{-st}
# ---------------------------------------------------------------------------
def fig1_step_impulse_kernel() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    t = np.linspace(-0.5, 4.0, 600)

    # Unit step u(t)
    ax = axes[0]
    u = np.where(t >= 0, 1.0, 0.0)
    ax.plot(t, u, color=C_BLUE, lw=2.4)
    ax.scatter([0], [0], s=55, facecolor="white",
               edgecolor=C_BLUE, lw=1.6, zorder=5)
    ax.scatter([0], [1], s=55, color=C_BLUE, zorder=5)
    ax.axhline(0, color=C_DARK, lw=0.6)
    ax.axvline(0, color=C_DARK, lw=0.6)
    ax.set_title("Unit step  $u(t)$\n$\\mathcal{L}\\{u\\} = 1/s$")
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.3, 1.6)
    ax.set_xlabel("$t$")
    ax.set_ylabel("amplitude")

    # Dirac delta -- drawn as an arrow plus a Gaussian "shape"
    ax = axes[1]
    ax.axhline(0, color=C_DARK, lw=0.6)
    ax.axvline(0, color=C_DARK, lw=0.6)
    g = np.exp(-((t - 0.0) ** 2) / 0.005)
    g = g / g.max() * 0.9
    ax.fill_between(t, 0, g, color=C_PURPLE, alpha=0.18)
    ax.annotate(
        "", xy=(0, 1.4), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=2.4,
                        mutation_scale=18),
    )
    ax.text(0.18, 1.35, r"area $= 1$", color=C_PURPLE, fontsize=11)
    ax.set_title("Dirac impulse  $\\delta(t)$\n$\\mathcal{L}\\{\\delta\\} = 1$")
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.3, 1.7)
    ax.set_xlabel("$t$")
    ax.set_yticks([])

    # The Laplace probe e^{-st}: family of decaying exponentials
    ax = axes[2]
    tt = np.linspace(0, 4, 400)
    for s, c, lab in [(0.5, C_GREEN, "$s = 0.5$"),
                      (1.0, C_BLUE, "$s = 1.0$"),
                      (2.0, C_PURPLE, "$s = 2.0$")]:
        ax.plot(tt, np.exp(-s * tt), color=c, lw=2.2, label=lab)
    ax.set_title("Laplace probe  $e^{-st}$\nlarger $s$ $\\Rightarrow$ stronger decay")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$e^{-st}$")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    fig.suptitle(
        r"Building blocks of the Laplace transform: "
        r"$F(s) = \int_0^\infty f(t)\,e^{-st}\,dt$",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig1_step_impulse_kernel")


# ---------------------------------------------------------------------------
# fig2: pole locations in the s-plane <-> impulse responses
# ---------------------------------------------------------------------------
def fig2_pole_zero_responses() -> None:
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])

    # ---- s-plane ----
    ax_s.axhline(0, color=C_DARK, lw=0.7)
    ax_s.axvline(0, color=C_DARK, lw=0.7)
    ax_s.fill_betweenx([-3, 3], -3.2, 0, color=C_GREEN, alpha=0.07)
    ax_s.fill_betweenx([-3, 3], 0, 3.2, color=C_RED, alpha=0.07)
    ax_s.text(-2.9, 2.55, "stable\n(LHP)", color=C_GREEN, fontsize=10,
              fontweight="bold")
    ax_s.text(1.6, 2.55, "unstable\n(RHP)", color=C_RED, fontsize=10,
              fontweight="bold")

    poles = [
        (-1.5, 0.0, C_BLUE,   "A: real, stable"),
        (+1.0, 0.0, C_RED,    "B: real, unstable"),
        (-0.5, +1.6, C_PURPLE, "C: complex, damped"),
        (-0.5, -1.6, C_PURPLE, None),
        (0.0, +2.0, C_AMBER,  "D: pure imaginary"),
        (0.0, -2.0, C_AMBER,  None),
        (+0.4, +1.6, C_GREEN, "E: complex, growing"),
        (+0.4, -1.6, C_GREEN, None),
    ]
    for re, im, c, lab in poles:
        ax_s.scatter(re, im, s=140, marker="x", color=c, lw=2.6,
                     label=lab, zorder=4)
    ax_s.set_xlim(-3.2, 3.2)
    ax_s.set_ylim(-3, 3)
    ax_s.set_xlabel(r"Re($s$)")
    ax_s.set_ylabel(r"Im($s$)")
    ax_s.set_title("Pole locations in the $s$-plane")
    ax_s.legend(loc="lower right", fontsize=9)
    ax_s.set_aspect("equal", adjustable="box")

    # ---- corresponding impulse responses ----
    t = np.linspace(0, 6, 800)
    cases = [
        ("A: $e^{-1.5t}$ decays",          np.exp(-1.5 * t),                                   C_BLUE),
        ("B: $e^{1.0t}$ blows up",         np.exp(1.0 * t),                                    C_RED),
        ("C: $e^{-0.5t}\\cos(1.6t)$",       np.exp(-0.5 * t) * np.cos(1.6 * t),                 C_PURPLE),
        ("D: $\\cos(2t)$ sustained",        np.cos(2 * t),                                      C_AMBER),
        ("E: $e^{0.4t}\\cos(1.6t)$",        np.exp(0.4 * t) * np.cos(1.6 * t),                  C_GREEN),
    ]
    for lab, y, c in cases:
        ax_t.plot(t, y, color=c, lw=2.0, label=lab)

    ax_t.axhline(0, color=C_DARK, lw=0.6)
    ax_t.set_xlim(0, 6)
    ax_t.set_ylim(-6, 8)
    ax_t.set_xlabel("$t$")
    ax_t.set_ylabel("response")
    ax_t.set_title("Impulse responses determined by pole location")
    ax_t.legend(loc="upper left", fontsize=9, ncol=1)

    fig.suptitle(
        "Stability is geometry: the half-plane that holds the poles "
        "decides the long-time fate of the system",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig2_pole_zero_responses")


# ---------------------------------------------------------------------------
# fig3: partial fraction decomposition turns inverse-Laplace into table lookup
# ---------------------------------------------------------------------------
def fig3_partial_fractions() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Y(s) = (3s + 5) / ((s+1)(s+2))
    #      = 2/(s+1) + 1/(s+2)
    # y(t) = 2 e^{-t} + e^{-2t}
    t = np.linspace(0, 5, 500)
    y1 = 2 * np.exp(-t)
    y2 = 1 * np.exp(-2 * t)
    y = y1 + y2

    ax = axes[0]
    ax.plot(t, y1, color=C_BLUE, lw=2.2,
            label=r"$2e^{-t}$  from pole at $s=-1$")
    ax.plot(t, y2, color=C_PURPLE, lw=2.2,
            label=r"$e^{-2t}$  from pole at $s=-2$")
    ax.plot(t, y, color=C_DARK, lw=2.6,
            label=r"$y(t) = 2e^{-t} + e^{-2t}$")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("$t$")
    ax.set_ylabel("amplitude")
    ax.set_title("Inverse Laplace = sum of simple modes")
    ax.legend(loc="upper right")

    # Schematic of the decomposition
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    box_kw = dict(ha="center", va="center", fontsize=12.5,
                  bbox=dict(boxstyle="round,pad=0.55",
                            facecolor="white", edgecolor=C_DARK, lw=1.0))

    ax.text(5.0, 8.5,
            r"$Y(s) = \dfrac{3s+5}{(s+1)(s+2)}$",
            color=C_DARK, **box_kw)

    ax.annotate("", xy=(2.5, 5.6), xytext=(4.4, 7.7),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))
    ax.annotate("", xy=(7.5, 5.6), xytext=(5.6, 7.7),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))

    ax.text(2.5, 4.9, r"$\dfrac{2}{s+1}$", color=C_BLUE, **box_kw)
    ax.text(7.5, 4.9, r"$\dfrac{1}{s+2}$", color=C_PURPLE, **box_kw)

    ax.annotate("", xy=(2.5, 2.6), xytext=(2.5, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))
    ax.annotate("", xy=(7.5, 2.6), xytext=(7.5, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))
    ax.text(4.55, 3.5, "table lookup", color=C_GRAY,
            fontsize=10, ha="center", style="italic")

    ax.text(2.5, 1.9, r"$2e^{-t}$", color=C_BLUE, **box_kw)
    ax.text(7.5, 1.9, r"$e^{-2t}$", color=C_PURPLE, **box_kw)

    ax.text(5.0, 0.55, r"$y(t) = 2e^{-t} + e^{-2t}$",
            color=C_DARK, **box_kw)
    ax.annotate("", xy=(5.0, 1.1), xytext=(2.9, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))
    ax.annotate("", xy=(5.0, 1.1), xytext=(7.1, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))

    ax.set_title("Partial-fraction workflow", fontsize=12)

    fig.suptitle(
        r"Partial fractions: split $Y(s)$ into table-lookup pieces, "
        r"then sum the time-domain modes",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig3_partial_fractions")


# ---------------------------------------------------------------------------
# fig4: second-order transfer function -- step response + Bode plot
# ---------------------------------------------------------------------------
def fig4_transfer_function() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    omega_n = 1.0
    zetas = [0.2, 1.0, 2.0]
    labels = [r"$\zeta=0.2$  under-damped",
              r"$\zeta=1.0$  critically damped",
              r"$\zeta=2.0$  over-damped"]
    colors = [C_BLUE, C_PURPLE, C_GREEN]

    # Step responses
    ax = axes[0]
    t = np.linspace(0, 18, 600)
    for z, lab, c in zip(zetas, labels, colors):
        sys = signal.TransferFunction([omega_n ** 2],
                                      [1, 2 * z * omega_n, omega_n ** 2])
        _, y = signal.step(sys, T=t)
        ax.plot(t, y, color=c, lw=2.2, label=lab)
    ax.axhline(1.0, color=C_DARK, lw=0.7, ls="--", alpha=0.6)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("$t$")
    ax.set_ylabel("output")
    ax.set_title(r"Step response of $H(s)=\frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}$")
    ax.legend(loc="lower right", fontsize=9)

    # Bode magnitude
    ax_mag = axes[1]
    ax_phase = axes[2]
    w = np.logspace(-1.5, 1.5, 500)
    for z, lab, c in zip(zetas, labels, colors):
        sys = signal.TransferFunction([omega_n ** 2],
                                      [1, 2 * z * omega_n, omega_n ** 2])
        w_, mag, phase = signal.bode(sys, w=w)
        ax_mag.semilogx(w_, mag, color=c, lw=2.2, label=lab)
        ax_phase.semilogx(w_, phase, color=c, lw=2.2, label=lab)

    ax_mag.set_title("Bode magnitude")
    ax_mag.set_xlabel(r"$\omega$ (rad/s)")
    ax_mag.set_ylabel("magnitude (dB)")
    ax_mag.axvline(omega_n, color=C_DARK, lw=0.7, ls="--", alpha=0.5)
    ax_mag.text(omega_n * 1.05, ax_mag.get_ylim()[1] * 0.85,
                r"$\omega_n$", color=C_DARK, fontsize=10)

    ax_phase.set_title("Bode phase")
    ax_phase.set_xlabel(r"$\omega$ (rad/s)")
    ax_phase.set_ylabel("phase (deg)")
    ax_phase.axvline(omega_n, color=C_DARK, lw=0.7, ls="--", alpha=0.5)
    ax_phase.set_yticks([0, -45, -90, -135, -180])

    fig.suptitle(
        "The same system, two windows: time-domain step and frequency-domain Bode",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig4_transfer_function")


# ---------------------------------------------------------------------------
# fig5: resonance -- amplitude grows linearly when forcing matches omega_0
# ---------------------------------------------------------------------------
def fig5_resonance_buildup() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))

    omega0 = 1.0
    t = np.linspace(0, 40, 4000)

    # On-resonance: omega = omega_0
    # y'' + omega0^2 y = cos(omega t),  y(0)=y'(0)=0
    # Solution at resonance:  y = t/(2*omega0) * sin(omega0 t)
    y_on = t / (2 * omega0) * np.sin(omega0 * t)

    ax = axes[0]
    ax.plot(t, y_on, color=C_BLUE, lw=1.4, label=r"$y(t)$")
    ax.plot(t, t / (2 * omega0), color=C_RED, lw=1.6, ls="--",
            label=r"envelope $\pm t/(2\omega_0)$")
    ax.plot(t, -t / (2 * omega0), color=C_RED, lw=1.6, ls="--")
    ax.set_xlim(0, 40)
    ax.set_ylim(-22, 22)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(r"On-resonance: $\omega = \omega_0 = 1$  (linear growth)")
    ax.legend(loc="upper left", fontsize=9)

    # Off-resonance: bounded beat pattern
    # General solution to y'' + omega0^2 y = cos(omega t), zero ICs:
    # y = (cos(omega t) - cos(omega0 t)) / (omega0^2 - omega^2)
    omega = 1.1
    y_off = (np.cos(omega * t) - np.cos(omega0 * t)) / (omega0 ** 2 - omega ** 2)

    ax = axes[1]
    ax.plot(t, y_off, color=C_PURPLE, lw=1.4, label=r"$y(t)$")
    bound = 2.0 / abs(omega0 ** 2 - omega ** 2)
    ax.axhline(bound, color=C_GREEN, lw=1.6, ls="--",
               label=r"bounded by $2/|\omega_0^2-\omega^2|$")
    ax.axhline(-bound, color=C_GREEN, lw=1.6, ls="--")
    ax.set_xlim(0, 40)
    ax.set_ylim(-1.3 * bound, 1.3 * bound)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(r"Off-resonance: $\omega = 1.1 \neq \omega_0$  (beats, bounded)")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        r"Why resonance is special: $Y(s) = s/(s^2+\omega_0^2)^2$ "
        r"has a repeated pole, giving a $t$-factor in $y(t)$",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig5_resonance_buildup")


# ---------------------------------------------------------------------------
# fig6: PID closed-loop step response on a second-order plant
# ---------------------------------------------------------------------------
def fig6_pid_control() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.0))

    # Plant: G(s) = 1 / (s^2 + s + 1)
    plant_num = [1.0]
    plant_den = [1.0, 1.0, 1.0]

    def closed_loop(Kp: float, Ki: float, Kd: float):
        """Form unity-feedback closed loop with PID controller and the plant.

        C(s) = Kd s + Kp + Ki/s = (Kd s^2 + Kp s + Ki) / s
        Open loop  L(s) = C(s) G(s)
                       = (Kd s^2 + Kp s + Ki) / (s * (s^2 + s + 1))
        Closed loop T(s) = L / (1 + L)
                       = (Kd s^2 + Kp s + Ki)
                         / (s^3 + s^2 + s + Kd s^2 + Kp s + Ki)
        """
        num = [Kd, Kp, Ki]
        den = [1.0,
               1.0 + Kd,
               1.0 + Kp,
               Ki]
        return signal.TransferFunction(num, den)

    t = np.linspace(0, 18, 1500)
    cases = [
        ("P only  ($K_p=2$)",                 closed_loop(2.0, 0.0, 0.0),  C_AMBER),
        ("PI  ($K_p=2,\\ K_i=1$)",            closed_loop(2.0, 1.0, 0.0),  C_PURPLE),
        ("PID  ($K_p=3,\\ K_i=1.5,\\ K_d=2$)", closed_loop(3.0, 1.5, 2.0),  C_BLUE),
    ]

    for lab, sys, c in cases:
        _, y = signal.step(sys, T=t)
        ax.plot(t, y, color=c, lw=2.4, label=lab)

    ax.axhline(1.0, color=C_DARK, lw=0.8, ls="--", alpha=0.7,
               label="reference $r(t) = 1$")
    ax.fill_between(t, 0.95, 1.05, color=C_GREEN, alpha=0.12,
                    label=r"$\pm 5\%$ band")

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("$t$")
    ax.set_ylabel("output")
    ax.set_title(
        r"Closed-loop step response, plant $G(s)=\dfrac{1}{s^2+s+1}$"
    )
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "P leaves a steady-state error; I removes it; D damps the overshoot",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig6_pid_control")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating ODE chapter 04 figures (Laplace transform)...")
    print(f"  EN dir: {EN_DIR}")
    print(f"  ZH dir: {ZH_DIR}")
    fig1_step_impulse_kernel()
    fig2_pole_zero_responses()
    fig3_partial_fractions()
    fig4_transfer_function()
    fig5_resonance_buildup()
    fig6_pid_control()
    print("Done.")


if __name__ == "__main__":
    main()
