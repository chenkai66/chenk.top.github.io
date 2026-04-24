"""
Chapter 11: Numerical Methods for ODEs -- Euler, RK4, adaptive step, stiffness, implicit.

Figures (saved to BOTH the EN and ZH asset folders):
  fig1_euler_vs_rk4.png             -- accuracy comparison on y' = -2y for several h
  fig2_error_convergence.png        -- log-log convergence: Euler O(h), Heun O(h^2), RK4 O(h^4)
  fig3_adaptive_step_size.png       -- RK45 adaptive stepping on a sharp solution
  fig4_stiffness.png                -- van der Pol (mu=1000): explicit blows up, implicit handles it
  fig5_implicit_vs_explicit.png     -- stability regions in the complex plane
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp

plt.style.use('seaborn-v0_8-whitegrid')

BLUE   = '#2563eb'
PURPLE = '#7c3aed'
GREEN  = '#10b981'
RED    = '#ef4444'

OUT_DIRS = [
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/11-numerical-methods',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/11-数值方法',
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
# Local solvers (kept self-contained on purpose -- so the script does not
# depend on the chapter code samples)
# ---------------------------------------------------------------------------
def euler(f, x0, y0, x_end, h):
    xs = [x0]; ys = [y0]
    x, y = x0, y0
    while x < x_end - 1e-12:
        h_eff = min(h, x_end - x)
        y = y + h_eff * f(x, y)
        x = x + h_eff
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def heun(f, x0, y0, x_end, h):
    xs = [x0]; ys = [y0]
    x, y = x0, y0
    while x < x_end - 1e-12:
        h_eff = min(h, x_end - x)
        k1 = f(x, y)
        k2 = f(x + h_eff, y + h_eff * k1)
        y = y + h_eff * (k1 + k2) / 2
        x = x + h_eff
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def rk4(f, x0, y0, x_end, h):
    xs = [x0]; ys = [y0]
    x, y = x0, y0
    while x < x_end - 1e-12:
        h_eff = min(h, x_end - x)
        k1 = f(x, y)
        k2 = f(x + h_eff/2, y + h_eff*k1/2)
        k3 = f(x + h_eff/2, y + h_eff*k2/2)
        k4 = f(x + h_eff, y + h_eff*k3)
        y = y + h_eff * (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x + h_eff
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# fig1: Euler vs RK4 on y' = -2y
# ---------------------------------------------------------------------------
def fig1_euler_vs_rk4():
    print('Building fig1_euler_vs_rk4...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    f = lambda x, y: -2 * y
    x_exact = np.linspace(0, 3, 400)
    y_exact = np.exp(-2 * x_exact)

    # Left: Euler at several h
    ax = axes[0]
    ax.plot(x_exact, y_exact, color='black', lw=2.4, label='exact $e^{-2x}$')
    palette = [RED, PURPLE, GREEN, BLUE]
    for h, c in zip([0.5, 0.25, 0.1, 0.05], palette):
        xn, yn = euler(f, 0, 1.0, 3.0, h)
        ax.plot(xn, yn, '-o', color=c, ms=4, lw=1.2,
                label=f'Euler  $h={h}$')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y(x)$')
    ax.set_title('Forward Euler: large $h$ overshoots and oscillates',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.4, 1.2)

    # Right: RK4 at the same large h
    ax = axes[1]
    ax.plot(x_exact, y_exact, color='black', lw=2.4, label='exact $e^{-2x}$')
    for h, c in zip([0.5, 0.25, 0.1, 0.05], palette):
        xn, yn = rk4(f, 0, 1.0, 3.0, h)
        ax.plot(xn, yn, '-o', color=c, ms=4, lw=1.2,
                label=f'RK4  $h={h}$')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y(x)$')
    ax.set_title('RK4: even $h=0.5$ tracks the exact curve closely',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Euler vs RK4 on $\\dot{y} = -2y$,  $y(0)=1$",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig1_euler_vs_rk4.png')


# ---------------------------------------------------------------------------
# fig2: error convergence (log-log) -- Euler O(h), Heun O(h^2), RK4 O(h^4)
# ---------------------------------------------------------------------------
def fig2_error_convergence():
    print('Building fig2_error_convergence...')
    fig, ax = plt.subplots(figsize=(10, 6.2))

    f = lambda x, y: -2 * y
    y_true = np.exp(-2 * 3.0)
    hs = np.logspace(-3.5, -0.3, 12)

    err_e, err_h, err_r = [], [], []
    for h in hs:
        _, ye = euler(f, 0, 1.0, 3.0, h);  err_e.append(abs(ye[-1] - y_true))
        _, yh = heun(f, 0, 1.0, 3.0, h);   err_h.append(abs(yh[-1] - y_true))
        _, yr = rk4(f, 0, 1.0, 3.0, h);    err_r.append(abs(yr[-1] - y_true))

    err_e = np.array(err_e); err_h = np.array(err_h); err_r = np.array(err_r)
    ax.loglog(hs, err_e, 'o-', color=RED,    lw=1.8, ms=8, label='Forward Euler')
    ax.loglog(hs, err_h, 's-', color=PURPLE, lw=1.8, ms=8, label="Heun (improved Euler)")
    ax.loglog(hs, err_r, '^-', color=BLUE,   lw=1.8, ms=8, label='RK4')

    # Reference slopes
    ax.loglog(hs, hs * (err_e[0] / hs[0]) * 0.7,
              ls=':', color=RED, lw=1.0, label=r'slope 1  ($\mathcal{O}(h)$)')
    ax.loglog(hs, hs**2 * (err_h[0] / hs[0]**2) * 0.7,
              ls=':', color=PURPLE, lw=1.0, label=r'slope 2  ($\mathcal{O}(h^2)$)')
    ax.loglog(hs, hs**4 * (err_r[0] / hs[0]**4) * 0.7,
              ls=':', color=BLUE, lw=1.0, label=r'slope 4  ($\mathcal{O}(h^4)$)')

    # Floor where round-off dominates RK4
    ax.axhline(1e-14, color='gray', lw=0.6, ls='--')
    ax.text(hs[0]*0.7, 2e-14, 'machine precision floor',
            fontsize=8, color='gray')

    ax.set_xlabel('step size $h$', fontsize=11)
    ax.set_ylabel(r'global error  $|y_N - y(3)|$', fontsize=11)
    ax.set_title('Convergence of one-step methods on $\\dot{y}=-2y$\n'
                 'log-log slopes match the theoretical orders',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    save(fig, 'fig2_error_convergence.png')


# ---------------------------------------------------------------------------
# fig3: adaptive step size on a sharp solution
# ---------------------------------------------------------------------------
def fig3_adaptive_step_size():
    print('Building fig3_adaptive_step_size...')
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5),
                             gridspec_kw={'height_ratios': [1.4, 1.0]},
                             sharex=True)

    # A sharp Gaussian-driven ODE: y' = -2*y + 100*exp(-(t-1)**2 / 0.005)
    def f(t, y):
        return [-2 * y[0] + 100 * np.exp(-(t - 1.0)**2 / 0.005)]

    sol = solve_ivp(f, [0, 3], [0.0], method='RK45',
                    rtol=1e-6, atol=1e-9, dense_output=True)
    tt = np.linspace(0, 3, 1500)
    yy = sol.sol(tt)[0]

    ax = axes[0]
    ax.plot(tt, yy, color=BLUE, lw=2.2, label='solution')
    ax.scatter(sol.t, sol.y[0], color=RED, s=22, zorder=5,
               label=f'RK45 step locations  ({len(sol.t)} steps)')
    ax.axvspan(0.85, 1.15, color=GREEN, alpha=0.10,
               label='steep region (small $h$)')
    ax.set_ylabel('$y(t)$', fontsize=11)
    ax.set_title('Adaptive RK45 places steps where the solution moves fast',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[1]
    dt = np.diff(sol.t)
    ax.semilogy(sol.t[:-1], dt, '-o', color=PURPLE, lw=1.0, ms=4)
    ax.axvspan(0.85, 1.15, color=GREEN, alpha=0.10)
    ax.set_xlabel('$t$', fontsize=11)
    ax.set_ylabel('step size $h_n$  (log)', fontsize=11)
    ax.set_title('Step size shrinks ~ 100x in the steep region, then expands again',
                 fontsize=11, fontweight='bold')

    fig.tight_layout()
    save(fig, 'fig3_adaptive_step_size.png')


# ---------------------------------------------------------------------------
# fig4: stiffness -- van der Pol mu=1000
# ---------------------------------------------------------------------------
def fig4_stiffness():
    print('Building fig4_stiffness...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8),
                             gridspec_kw={'width_ratios': [1.2, 1.0]})

    # van der Pol with large mu is the textbook stiff problem
    mu = 1000.0
    def vdp(t, y):
        return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

    t_span = [0, 3000]
    y0 = [2.0, 0.0]

    sol_bdf = solve_ivp(vdp, t_span, y0, method='BDF',
                        rtol=1e-6, atol=1e-9, dense_output=True)
    sol_rk = solve_ivp(vdp, t_span, y0, method='RK45',
                       rtol=1e-3, atol=1e-6, max_step=5.0)

    ax = axes[0]
    tt = np.linspace(0, 3000, 4000)
    ax.plot(tt, sol_bdf.sol(tt)[0], color=BLUE, lw=1.6,
            label=f'BDF (implicit)  --  {len(sol_bdf.t)} steps')
    ax.plot(sol_rk.t, sol_rk.y[0], color=RED, lw=0.8, alpha=0.7,
            label=f'RK45 (explicit)  --  {len(sol_rk.t)} steps')
    ax.set_xlabel('$t$'); ax.set_ylabel('$y(t)$')
    ax.set_title(r'Van der Pol  $\ddot{y} - \mu(1-y^2)\dot{y} + y = 0$,  '
                 fr'$\mu={int(mu)}$',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[1]
    bars = ax.bar(['BDF\n(implicit)', 'RK45\n(explicit)'],
                  [len(sol_bdf.t), len(sol_rk.t)],
                  color=[BLUE, RED], edgecolor='black', linewidth=1.0,
                  width=0.55)
    for b, v in zip(bars, [len(sol_bdf.t), len(sol_rk.t)]):
        ax.text(b.get_x() + b.get_width()/2, v * 1.04,
                f'{v}', ha='center', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylabel('# integration steps  (log)', fontsize=11)
    ax.set_title('Step count: implicit method needs ~100x fewer steps',
                 fontsize=11, fontweight='bold')

    fig.suptitle('Stiff problem: explicit methods choke, implicit methods breeze through',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig4_stiffness.png')


# ---------------------------------------------------------------------------
# fig5: stability regions in complex plane
# ---------------------------------------------------------------------------
def fig5_implicit_vs_explicit():
    print('Building fig5_implicit_vs_explicit...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))

    # Build a fine grid in complex plane
    re = np.linspace(-5, 3, 600)
    im = np.linspace(-4, 4, 600)
    Re, Im = np.meshgrid(re, im)
    z = Re + 1j * Im

    # Stability functions R(z) = y_{n+1}/y_n  for the test ODE y' = lambda y
    R_euler = 1 + z
    R_heun  = 1 + z + z**2 / 2
    R_rk4   = 1 + z + z**2/2 + z**3/6 + z**4/24
    R_beul  = 1 / (1 - z)
    R_trap  = (1 + z/2) / (1 - z/2)

    # Left panel: explicit methods (bounded regions)
    ax = axes[0]
    ax.axhline(0, color='black', lw=0.6); ax.axvline(0, color='black', lw=0.6)
    for R, color, name in [
        (R_euler, RED,    'Forward Euler'),
        (R_heun,  PURPLE, 'Heun (RK2)'),
        (R_rk4,   BLUE,   'RK4'),
    ]:
        cs = ax.contour(Re, Im, np.abs(R), levels=[1.0],
                        colors=[color], linewidths=2.2)
        ax.contourf(Re, Im, np.abs(R) <= 1.0,
                    levels=[0.5, 1.5], colors=[color], alpha=0.18)
        ax.plot([], [], color=color, lw=2.2, label=name)
    ax.set_xlim(-4.5, 1.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mathrm{Re}(h\lambda)$', fontsize=11)
    ax.set_ylabel(r'$\mathrm{Im}(h\lambda)$', fontsize=11)
    ax.set_title('Explicit methods: small bounded stability regions\n'
                 '(stiff problem $\\Rightarrow$ tiny $h$ required)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    # Right panel: implicit methods
    ax = axes[1]
    ax.axhline(0, color='black', lw=0.6); ax.axvline(0, color='black', lw=0.6)
    for R, color, name in [
        (R_beul,  GREEN,  'Backward Euler'),
        (R_trap,  BLUE,   'Trapezoidal'),
    ]:
        ax.contourf(Re, Im, np.abs(R) <= 1.0,
                    levels=[0.5, 1.5], colors=[color], alpha=0.22)
        ax.plot([], [], color=color, lw=2.2, label=name)
    # Annotate A-stability: entire left half-plane
    ax.axvspan(-5, 0, color='gray', alpha=0.05)
    ax.text(-3.7, 3.0, 'entire left half-plane is stable\n(A-stable)',
            fontsize=10, ha='center', color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray'))
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mathrm{Re}(h\lambda)$', fontsize=11)
    ax.set_ylabel(r'$\mathrm{Im}(h\lambda)$', fontsize=11)
    ax.set_title('Implicit methods: stability covers all of LHP\n'
                 '(any $h$ works for stiff dissipative problems)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    fig.suptitle('Stability regions:  $|R(h\\lambda)| \\leq 1$  in the complex plane',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig5_implicit_vs_explicit.png')


if __name__ == '__main__':
    fig1_euler_vs_rk4()
    fig2_error_convergence()
    fig3_adaptive_step_size()
    fig4_stiffness()
    fig5_implicit_vs_explicit()
    print('\nChapter 11 figures complete.')
