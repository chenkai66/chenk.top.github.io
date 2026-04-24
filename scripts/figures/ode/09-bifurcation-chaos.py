"""
Chapter 9: Bifurcation and Chaos -- Lorenz, butterfly effect, Lyapunov exponents.

Figures:
  fig1_lorenz_attractor.png            -- 3D Lorenz butterfly with two viewpoints
  fig2_sensitivity_initial_conditions.png  -- Two trajectories diverge exponentially
  fig3_bifurcation_diagram.png         -- Logistic-map bifurcation cascade (period-doubling -> chaos)
  fig4_butterfly_effect.png            -- Multi-trajectory ensemble with shaded divergence cone
  fig5_lyapunov_exponents.png          -- Lyapunov exponent spectrum vs rho for Lorenz
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import odeint, solve_ivp

plt.style.use('seaborn-v0_8-whitegrid')

BLUE   = '#2563eb'
PURPLE = '#7c3aed'
GREEN  = '#10b981'
RED    = '#ef4444'

OUT_DIRS = [
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/09-bifurcation-chaos',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/09-混沌理论与洛伦兹系统',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save(fig, name):
    for d in OUT_DIRS:
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print(f'  saved {name}')


def lorenz(s, t, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# ---------------------------------------------------------------------------
# fig1: Lorenz attractor in 3D, two viewpoints, color by time
# ---------------------------------------------------------------------------
def fig1_lorenz_attractor():
    print('Building fig1_lorenz_attractor...')
    fig = plt.figure(figsize=(14, 6.5))

    t = np.linspace(0, 60, 18000)
    sol = odeint(lorenz, [1.0, 1.0, 1.0], t)
    n = len(t)

    for idx, (elev, azim, title) in enumerate([
        (28, -65, 'Classic butterfly view'),
        (60, 30,  'Top-down view (two wings)'),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        # Color by time progression using viridis
        seg_n = 200
        for i in range(0, n - seg_n, seg_n):
            seg = sol[i:i + seg_n + 1]
            color = plt.cm.viridis(i / n)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color=color, lw=0.45, alpha=0.85)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        # equilibria C+/-
        beta = 8/3; rho = 28
        r = np.sqrt(beta * (rho - 1))
        ax.scatter([r, -r], [r, -r], [rho - 1, rho - 1],
                   color=RED, s=60, marker='X',
                   edgecolor='white', linewidth=1.0, zorder=20)
        ax.scatter([0], [0], [0], color=PURPLE, s=60, marker='o',
                   edgecolor='white', linewidth=1.0, zorder=20)

    fig.suptitle(r'Lorenz Attractor   $\sigma=10,\ \rho=28,\ \beta=8/3$',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    save(fig, 'fig1_lorenz_attractor.png')


# ---------------------------------------------------------------------------
# fig2: sensitivity to initial conditions
# ---------------------------------------------------------------------------
def fig2_sensitivity_initial_conditions():
    print('Building fig2_sensitivity_initial_conditions...')
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.35, wspace=0.25)

    t = np.linspace(0, 35, 14000)
    sol1 = odeint(lorenz, [1.0,        1.0, 1.0], t)
    sol2 = odeint(lorenz, [1.0 + 1e-8, 1.0, 1.0], t)

    # Top: 3D attractor with two trajectories
    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    ax3d.plot(sol1[:, 0], sol1[:, 1], sol1[:, 2], color=BLUE,
              lw=0.45, alpha=0.85, label=r'start $(1,1,1)$')
    ax3d.plot(sol2[:, 0], sol2[:, 1], sol2[:, 2], color=RED,
              lw=0.45, alpha=0.85,
              label=r'start $(1+10^{-8},\,1,\,1)$')
    ax3d.view_init(elev=25, azim=-60)
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    ax3d.set_title('Two trajectories starting $10^{-8}$ apart',
                   fontsize=12, fontweight='bold')
    ax3d.legend(loc='upper left', fontsize=10)

    # Bottom-left: x(t) overlay
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, sol1[:, 0], color=BLUE, lw=0.9, label='trajectory 1')
    ax.plot(t, sol2[:, 0], color=RED,  lw=0.9, ls='--', label='trajectory 2')
    ax.set_xlabel('time'); ax.set_ylabel('x(t)')
    ax.set_title('x components diverge after a few Lyapunov times',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Bottom-right: separation distance, log scale
    ax = fig.add_subplot(gs[1, 1])
    diff = np.linalg.norm(sol1 - sol2, axis=1)
    ax.semilogy(t, diff + 1e-16, color=PURPLE, lw=1.4)
    # Fit exponential growth in early regime
    mask = (t > 1) & (t < 18)
    fit = np.polyfit(t[mask], np.log(diff[mask] + 1e-16), 1)
    lam = fit[0]
    ax.semilogy(t[mask], np.exp(np.polyval(fit, t[mask])),
                color=GREEN, lw=2.0, ls='--',
                label=fr'$\sim e^{{\lambda t}},\ \lambda \approx {lam:.2f}$')
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('time'); ax.set_ylabel('separation  $|\\Delta\\mathbf{x}|$')
    ax.set_title('Exponential divergence  (largest Lyapunov exponent)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Butterfly Effect:  exponential sensitivity to initial conditions',
                 fontsize=14, fontweight='bold', y=1.00)
    save(fig, 'fig2_sensitivity_initial_conditions.png')


# ---------------------------------------------------------------------------
# fig3: bifurcation diagram (logistic map -- canonical period-doubling)
# ---------------------------------------------------------------------------
def fig3_bifurcation_diagram():
    print('Building fig3_bifurcation_diagram...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={'width_ratios': [2, 1]})

    # Logistic map x_{n+1} = r x_n (1 - x_n)
    R = np.linspace(2.5, 4.0, 1600)
    n_warm = 1500
    n_keep = 200
    rs_acc, xs_acc = [], []
    for r in R:
        x = 0.4
        for _ in range(n_warm):
            x = r * x * (1 - x)
        for _ in range(n_keep):
            x = r * x * (1 - x)
            rs_acc.append(r); xs_acc.append(x)

    ax = axes[0]
    ax.scatter(rs_acc, xs_acc, s=0.18, c=xs_acc, cmap='viridis',
               alpha=0.55, edgecolors='none')
    ax.set_xlim(2.5, 4.0); ax.set_ylim(0, 1)
    ax.set_xlabel('parameter  $r$', fontsize=12)
    ax.set_ylabel('long-term  $x$', fontsize=12)
    ax.set_title('Logistic map bifurcation cascade  (period-doubling route to chaos)',
                 fontsize=12, fontweight='bold')

    # Annotate Feigenbaum points (approx)
    for r_v, lbl in [(3.0, 'period 2'), (3.449, 'period 4'),
                     (3.544, 'period 8'), (3.5699, 'chaos onset')]:
        ax.axvline(r_v, color='red', lw=0.6, ls='--', alpha=0.7)
        ax.text(r_v, 0.02, lbl, rotation=90, fontsize=8,
                color='red', va='bottom', ha='right')

    # Right panel: zoom on the chaotic region
    ax = axes[1]
    R2 = np.linspace(3.55, 4.0, 1200)
    rs2, xs2 = [], []
    for r in R2:
        x = 0.4
        for _ in range(n_warm):
            x = r * x * (1 - x)
        for _ in range(n_keep):
            x = r * x * (1 - x)
            rs2.append(r); xs2.append(x)
    ax.scatter(rs2, xs2, s=0.18, c=xs2, cmap='plasma',
               alpha=0.55, edgecolors='none')
    ax.set_xlim(3.55, 4.0); ax.set_ylim(0, 1)
    ax.set_xlabel('parameter  $r$', fontsize=12)
    ax.set_title('Zoom: chaotic windows + periodic islands',
                 fontsize=12, fontweight='bold')

    fig.tight_layout()
    save(fig, 'fig3_bifurcation_diagram.png')


# ---------------------------------------------------------------------------
# fig4: butterfly effect ensemble
# ---------------------------------------------------------------------------
def fig4_butterfly_effect():
    print('Building fig4_butterfly_effect...')
    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw={'height_ratios': [1.2, 1.0]})

    t = np.linspace(0, 25, 8000)
    n_traj = 30
    rng = np.random.default_rng(11)
    eps = 1e-3
    base = np.array([1.0, 1.0, 1.0])
    sols = []
    for _ in range(n_traj):
        s0 = base + eps * rng.standard_normal(3)
        sols.append(odeint(lorenz, s0, t))
    sols = np.array(sols)  # (n_traj, n_t, 3)

    # Top: x(t) bundle
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, n_traj))
    for i in range(n_traj):
        ax.plot(t, sols[i, :, 0], color=cmap[i], lw=0.65, alpha=0.75)
    ax.set_xlabel('time', fontsize=11)
    ax.set_ylabel('x(t)', fontsize=11)
    ax.set_title(fr'30 trajectories starting within $\varepsilon = {eps}$  '
                 'of $(1,1,1)$:  predictable -> divergent -> incoherent',
                 fontsize=12, fontweight='bold')

    # Bottom: ensemble spread (std dev) over time
    spread = sols.std(axis=0).mean(axis=1)  # average std across components
    ax = axes[1]
    ax.semilogy(t, spread + 1e-12, color=RED, lw=1.8)
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.fill_between(t, 1e-3, 1.0, color=GREEN, alpha=0.10,
                    label='predictable regime')
    ax.fill_between(t, 1.0, 25.0, color=RED, alpha=0.10,
                    label='incoherent regime')
    ax.set_xlabel('time', fontsize=11)
    ax.set_ylabel('ensemble spread  (log)', fontsize=11)
    ax.set_title('Predictability horizon:  spread crosses system scale',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(1e-4, 25)
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Butterfly Effect:  ensemble forecasting view',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    save(fig, 'fig4_butterfly_effect.png')


# ---------------------------------------------------------------------------
# fig5: Lyapunov exponent (largest) vs rho for Lorenz
# ---------------------------------------------------------------------------
def fig5_lyapunov_exponents():
    print('Building fig5_lyapunov_exponents...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # Estimate the largest LE using the standard renormalization method
    sigma, beta = 10.0, 8/3
    rhos = np.linspace(0.5, 50, 60)
    lams = []
    for rho in rhos:
        def f(s, t, rho=rho):
            x, y, z = s
            return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]
        # Warm up
        s = np.array([1.0, 1.0, 1.0])
        s = odeint(f, s, np.linspace(0, 30, 1500))[-1]
        s2 = s + np.array([1e-8, 0, 0])

        d0 = np.linalg.norm(s2 - s)
        T = 0.5     # renormalization interval
        N = 200
        log_sum = 0.0
        for _ in range(N):
            tt = np.linspace(0, T, 50)
            s_new  = odeint(f, s,  tt)[-1]
            s2_new = odeint(f, s2, tt)[-1]
            d = np.linalg.norm(s2_new - s_new)
            d = max(d, 1e-300)
            log_sum += np.log(d / d0)
            # rescale s2 along separation direction
            s = s_new
            s2 = s_new + (s2_new - s_new) * (d0 / d)
        lam = log_sum / (N * T)
        lams.append(lam)

    lams = np.array(lams)
    ax = axes[0]
    ax.axhline(0, color='gray', lw=0.8)
    pos = lams > 0
    ax.plot(rhos, lams, color='black', lw=1.0, alpha=0.5)
    ax.scatter(rhos[~pos], lams[~pos], color=GREEN, s=22,
               label=r'$\lambda_1 \leq 0$  (regular)')
    ax.scatter(rhos[ pos], lams[ pos], color=RED, s=22,
               label=r'$\lambda_1 > 0$  (chaotic)')
    for r_v, lbl in [(1.0, r'$\rho=1$ pitchfork'),
                     (24.74, r'$\rho\approx 24.74$ Hopf'),
                     (28.0, r'$\rho=28$')]:
        ax.axvline(r_v, color='purple', lw=0.6, ls='--', alpha=0.7)
        ax.text(r_v, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 0.5,
                lbl, rotation=90, fontsize=8, color='purple', ha='right')
    ax.set_xlabel(r'$\rho$', fontsize=12)
    ax.set_ylabel(r'largest Lyapunov exponent  $\lambda_1$', fontsize=12)
    ax.set_title('Lorenz:  $\\lambda_1(\\rho)$  -- chaos for $\\rho \\gtrapprox 25$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    # Right panel: Kaplan-Yorke dimension cartoon for rho=28 and full spectrum
    ax = axes[1]
    spectrum = [0.91, 0.0, -14.57]
    bars = ax.bar(['$\\lambda_1$', '$\\lambda_2$', '$\\lambda_3$'],
                  spectrum, color=[RED, '#888', BLUE],
                  edgecolor='black', linewidth=1.2)
    ax.axhline(0, color='black', lw=1.0)
    for bar, val in zip(bars, spectrum):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (0.6 if val >= 0 else -1.2),
                f'{val:+.2f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('exponent value', fontsize=12)
    ax.set_ylim(-16, 4)
    DKY = 2 + (spectrum[0] + spectrum[1]) / abs(spectrum[2])
    ax.set_title(f'Lyapunov spectrum at $\\rho=28$\n'
                 f'Kaplan-Yorke dim $D_{{KY}} \\approx {DKY:.3f}$  (fractal)',
                 fontsize=11, fontweight='bold')
    sum_lam = sum(spectrum)
    ax.text(0.02, 0.02,
            fr'$\sum \lambda_i = {sum_lam:+.2f} < 0$  =>  volume contracts',
            transform=ax.transAxes, fontsize=10, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray'))

    fig.suptitle('Quantifying Chaos:  Lyapunov exponents',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig5_lyapunov_exponents.png')


if __name__ == '__main__':
    fig1_lorenz_attractor()
    fig2_sensitivity_initial_conditions()
    fig3_bifurcation_diagram()
    fig4_butterfly_effect()
    fig5_lyapunov_exponents()
    print('\nChapter 9 figures complete.')
