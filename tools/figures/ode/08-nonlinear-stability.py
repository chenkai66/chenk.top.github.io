"""
Chapter 8: Nonlinear Systems and Phase Portraits
Generates 5 figures: Lyapunov stability, Lotka-Volterra, linearization, limit cycles, competition.

Figures:
  fig1_lyapunov_stability.png    -- Lyapunov function level sets shrinking to equilibrium
  fig2_lotka_volterra.png        -- Predator-prey: time series + closed orbits
  fig3_linearization.png         -- Nonlinear vs linearized phase portraits side-by-side
  fig4_limit_cycles.png          -- Van der Pol limit cycles for several mu
  fig5_competition_outcomes.png  -- Four outcomes of competition model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

BLUE   = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN  = COLORS["success"]
RED    = COLORS["danger"]

OUT_DIRS = [
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/08-nonlinear-stability',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/08-非线性系统与相图',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save(fig, name):
    for d in OUT_DIRS:
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print(f'  saved {name}')


# ---------------------------------------------------------------------------
# fig1: Lyapunov stability illustration
# ---------------------------------------------------------------------------
def fig1_lyapunov_stability():
    print('Building fig1_lyapunov_stability...')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # System: x_dot = -x + 0.3*y - 0.2*x*y^2
    #         y_dot = -2*y - 0.1*x^2*y
    def sys(s, t):
        x, y = s
        return [-x + 0.3*y - 0.2*x*y**2, -2*y - 0.1*x**2*y]

    x = np.linspace(-2.5, 2.5, 200)
    y = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(x, y)
    V = X**2 + 0.5 * Y**2

    ax = axes[0]
    levels = [0.05, 0.2, 0.5, 1.0, 1.8, 2.8, 4.0]
    cs = ax.contour(X, Y, V, levels=levels, cmap='viridis', linewidths=1.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    ax.contourf(X, Y, V, levels=levels, cmap='viridis', alpha=0.18)

    t = np.linspace(0, 8, 1500)
    starts = [(2.0, 1.5), (-1.8, 2.0), (1.6, -2.1), (-2.2, -1.0), (0.8, 2.2)]
    cols = [BLUE, PURPLE, GREEN, RED, '#fb923c']
    for s0, c in zip(starts, cols):
        sol = odeint(sys, s0, t)
        ax.plot(sol[:, 0], sol[:, 1], color=c, lw=2.0, alpha=0.95)
        ax.plot(sol[0, 0], sol[0, 1], 'o', color=c, ms=8,
                markeredgecolor='white', mew=1.2)
    ax.plot(0, 0, '*', color='black', ms=16,
            markeredgecolor='white', mew=1.0, zorder=20)

    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12); ax.set_ylabel('y', fontsize=12)
    ax.set_title(r'Trajectories cross level sets of $V$ inward',
                 fontsize=12, fontweight='bold')

    # Right: V(t) decay
    ax = axes[1]
    for s0, c in zip(starts, cols):
        sol = odeint(sys, s0, t)
        Vt = sol[:, 0]**2 + 0.5 * sol[:, 1]**2
        ax.semilogy(t, Vt, color=c, lw=2.0,
                    label=f'start ({s0[0]:.1f}, {s0[1]:.1f})')
    ax.set_xlabel('time  $t$', fontsize=12)
    ax.set_ylabel(r'$V(\mathbf{x}(t))$  (log scale)', fontsize=12)
    ax.set_title(r'$\dot V \leq 0$  =>  asymptotic stability',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(r'Lyapunov Stability:  level sets of $V(x,y)=x^2+\frac{1}{2}y^2$ shrink toward equilibrium',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig1_lyapunov_stability.png')


# ---------------------------------------------------------------------------
# fig2: Lotka-Volterra
# ---------------------------------------------------------------------------
def fig2_lotka_volterra():
    print('Building fig2_lotka_volterra...')
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    a, b, c, d = 1.0, 0.1, 0.5, 0.075
    def lv(s, t):
        x, y = s
        return [a*x - b*x*y, -c*y + d*x*y]

    # Time series
    ax = axes[0]
    t = np.linspace(0, 60, 4000)
    sol = odeint(lv, [10, 5], t)
    ax.plot(t, sol[:, 0], color=BLUE, lw=2.2, label='Prey  $x(t)$')
    ax.plot(t, sol[:, 1], color=RED, lw=2.2, label='Predator  $y(t)$')
    ax.fill_between(t, 0, sol[:, 0], color=BLUE, alpha=0.10)
    ax.fill_between(t, 0, sol[:, 1], color=RED, alpha=0.10)
    ax.set_xlabel('time', fontsize=12); ax.set_ylabel('population', fontsize=12)
    ax.set_title('Population oscillations  (predator lags prey)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    # Phase portrait with vector field
    ax = axes[1]
    xx = np.linspace(0.1, 18, 28)
    yy = np.linspace(0.1, 14, 22)
    XX, YY = np.meshgrid(xx, yy)
    U = a*XX - b*XX*YY
    V = -c*YY + d*XX*YY
    spd = np.sqrt(U**2 + V**2) + 1e-9
    ax.streamplot(XX, YY, U, V, color=spd, cmap='Greens',
                  density=1.1, linewidth=0.8, arrowsize=0.8)

    t2 = np.linspace(0, 30, 3000)
    starts = [(c/d, a/b + 1), (c/d, a/b + 3),
              (c/d, a/b + 5), (c/d + 4, a/b)]
    cols = [BLUE, PURPLE, GREEN, RED]
    for s0, col in zip(starts, cols):
        sol = odeint(lv, s0, t2)
        ax.plot(sol[:, 0], sol[:, 1], color=col, lw=1.8, alpha=0.95)

    ax.plot(c/d, a/b, '*', color='black', ms=16,
            markeredgecolor='white', mew=1.2, zorder=20,
            label=f'center ({c/d:.1f}, {a/b:.1f})')
    ax.set_xlim(0, 18); ax.set_ylim(0, 14)
    ax.set_xlabel('Prey  $x$', fontsize=12); ax.set_ylabel('Predator  $y$', fontsize=12)
    ax.set_title('Closed orbits around the center  (energy-like conservation)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(r"Lotka-Volterra:  $\dot x = ax - bxy,\ \dot y = -cy + dxy$",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig2_lotka_volterra.png')


# ---------------------------------------------------------------------------
# fig3: Linearization comparison
# ---------------------------------------------------------------------------
def fig3_linearization():
    print('Building fig3_linearization...')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # Nonlinear system: damped pendulum near (0,0)
    g = 0.25; w0 = 1.0
    def nonlinear(s, t):
        x, y = s
        return [y, -g*y - w0**2 * np.sin(x)]

    # Linearization at origin: y' = -g y - w0^2 x
    def linear(s, t):
        x, y = s
        return [y, -g*y - w0**2 * x]

    for ax, fn, title in [(axes[0], nonlinear, 'Nonlinear  $\\ddot\\theta + \\gamma\\dot\\theta + \\omega_0^2\\sin\\theta = 0$'),
                          (axes[1], linear,    'Linearized at origin  $\\ddot\\theta + \\gamma\\dot\\theta + \\omega_0^2\\theta = 0$')]:
        x = np.linspace(-3, 3, 28)
        y = np.linspace(-2.5, 2.5, 22)
        X, Y = np.meshgrid(x, y)
        if fn is nonlinear:
            U = Y; V = -g*Y - w0**2 * np.sin(X)
        else:
            U = Y; V = -g*Y - w0**2 * X
        spd = np.sqrt(U**2 + V**2) + 1e-9
        ax.streamplot(X, Y, U, V, color=spd, cmap='Blues',
                      density=1.1, linewidth=0.8, arrowsize=0.8)

        t = np.linspace(0, 25, 3000)
        starts = [(0.3, 0), (1.0, 0), (2.0, 0), (-1.5, 1.0), (2.5, -1.5)]
        cols = [BLUE, PURPLE, GREEN, RED, '#fb923c']
        for s0, col in zip(starts, cols):
            sol = odeint(fn, s0, t)
            ax.plot(sol[:, 0], sol[:, 1], color=col, lw=1.6, alpha=0.95)

        ax.plot(0, 0, 'o', color=GREEN, ms=10,
                markeredgecolor='white', mew=1.2, zorder=20)
        ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel(r'$\theta$', fontsize=12)
        ax.set_ylabel(r'$\omega$', fontsize=12)
        ax.set_title(title, fontsize=11, fontweight='bold')

    fig.suptitle('Hartman-Grobman:  qualitatively equivalent near a hyperbolic equilibrium',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig3_linearization.png')


# ---------------------------------------------------------------------------
# fig4: Van der Pol limit cycles
# ---------------------------------------------------------------------------
def fig4_limit_cycles():
    print('Building fig4_limit_cycles...')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mus = [0.5, 1.5, 4.0]
    cols = [BLUE, PURPLE, RED]

    for ax, mu, col in zip(axes, mus, cols):
        def vdp(s, t, mu=mu):
            x, y = s
            return [y, mu * (1 - x**2) * y - x]

        # Vector field
        x = np.linspace(-4, 4, 26)
        y = np.linspace(-7, 7, 22)
        X, Y = np.meshgrid(x, y)
        U = Y
        V = mu * (1 - X**2) * Y - X
        spd = np.sqrt(U**2 + V**2) + 1e-9
        ax.streamplot(X, Y, U, V, color=spd, cmap='Greys',
                      density=1.0, linewidth=0.7, arrowsize=0.7)

        t = np.linspace(0, 60, 8000)
        # Outside-in
        sol = odeint(vdp, [3.5, 3.5], t)
        ax.plot(sol[:, 0], sol[:, 1], color=COLORS["muted"], lw=0.7, alpha=0.7)
        # Inside-out
        sol = odeint(vdp, [0.05, 0.05], t)
        ax.plot(sol[:, 0], sol[:, 1], color=COLORS["muted"], lw=0.7, alpha=0.7)
        # Highlight the limit cycle: long-time tail
        sol = odeint(vdp, [2.0, 0.0], np.linspace(0, 200, 20000))
        ax.plot(sol[-3000:, 0], sol[-3000:, 1], color=col, lw=2.6, alpha=0.95,
                label='limit cycle')

        ax.plot(0, 0, 'X', color=RED, ms=10,
                markeredgecolor='white', mew=1.2, zorder=20)
        ax.set_xlim(-4, 4); ax.set_ylim(-7, 7)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(fr'$\mu = {mu}$',
                     fontsize=12, fontweight='bold', color=col)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(r"Van der Pol Limit Cycles:  $\ddot x - \mu(1-x^2)\dot x + x = 0$",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig4_limit_cycles.png')


# ---------------------------------------------------------------------------
# fig5: Competition model -- four outcomes
# ---------------------------------------------------------------------------
def fig5_competition_outcomes():
    print('Building fig5_competition_outcomes...')
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Parameters define the four classical regimes
    K1, K2, r1, r2 = 1.0, 1.0, 1.0, 1.0
    cases = [
        ('Species 1 wins',     0.5, 1.5),  # a12<1, a21>1
        ('Species 2 wins',     1.5, 0.5),
        ('Stable coexistence', 0.5, 0.5),
        ('Bistability',        1.5, 1.5),
    ]

    for ax, (title, a12, a21) in zip(axes.flat, cases):
        def comp(s, t, a12=a12, a21=a21):
            x, y = s
            return [r1*x*(1 - (x + a12*y) / K1),
                    r2*y*(1 - (y + a21*x) / K2)]

        xx = np.linspace(0, 1.7, 22)
        yy = np.linspace(0, 1.7, 22)
        XX, YY = np.meshgrid(xx, yy)
        U = r1*XX*(1 - (XX + a12*YY) / K1)
        V = r2*YY*(1 - (YY + a21*XX) / K2)
        spd = np.sqrt(U**2 + V**2) + 1e-9
        ax.streamplot(XX, YY, U, V, color=spd, cmap='Purples',
                      density=1.0, linewidth=0.7, arrowsize=0.7)

        # Nullclines
        x_line = np.linspace(0, 1.7, 200)
        y_null_x = (K1 - x_line) / a12   # x' = 0 (excluding x=0)
        y_null_y = K2 - a21 * x_line     # y' = 0 (excluding y=0)
        ax.plot(x_line, y_null_x, color=BLUE, lw=2.2, label="$x'=0$ nullcline")
        ax.plot(x_line, y_null_y, color=RED, lw=2.2, label="$y'=0$ nullcline")

        # Sample trajectories
        t = np.linspace(0, 30, 3000)
        rng = np.random.default_rng(2)
        for _ in range(8):
            s0 = rng.uniform(0.05, 1.5, 2)
            sol = odeint(comp, s0, t)
            ax.plot(sol[:, 0], sol[:, 1], color=GREEN, lw=1.0, alpha=0.7)

        # Equilibria markers
        eqs = [(0, 0), (K1, 0), (0, K2)]
        # Coexistence eq if exists (a12*a21 != 1)
        det = 1 - a12 * a21
        if abs(det) > 1e-6:
            xs = (K1 - a12 * K2) / det
            ys = (K2 - a21 * K1) / det
            if 0 < xs < 1.7 and 0 < ys < 1.7:
                eqs.append((xs, ys))
        for (ex, ey) in eqs:
            ax.plot(ex, ey, 'o', color='black', ms=8,
                    markeredgecolor='white', mew=1.2, zorder=20)

        ax.set_xlim(0, 1.7); ax.set_ylim(0, 1.7)
        ax.set_xlabel('Species 1  $x$', fontsize=11)
        ax.set_ylabel('Species 2  $y$', fontsize=11)
        ax.set_title(fr'{title}   $\alpha_{{12}}={a12},\ \alpha_{{21}}={a21}$',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Lotka-Volterra Competition:  four qualitative outcomes',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    save(fig, 'fig5_competition_outcomes.png')


if __name__ == '__main__':
    fig1_lyapunov_stability()
    fig2_lotka_volterra()
    fig3_linearization()
    fig4_limit_cycles()
    fig5_competition_outcomes()
    print('\nChapter 8 figures complete.')
