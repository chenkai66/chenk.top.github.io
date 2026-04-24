"""
Chapter 7: Stability Theory & Phase Plane Analysis
Generates 5 figures showcasing 2D phase portraits, classification, and stability tools.

Figures:
  fig1_phase_portraits.png            -- Six canonical 2D linear phase portraits
  fig2_trajectory_vector_field.png    -- Damped pendulum + LV trajectories on vector field
  fig3_eigenvalue_classification.png  -- Trace-determinant plane with regions
  fig4_lyapunov_function.png          -- Lyapunov function level sets and decay
  fig5_bifurcation_normal_forms.png   -- Saddle-node, transcritical, pitchfork, Hopf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import odeint

plt.style.use('seaborn-v0_8-whitegrid')

BLUE   = '#2563eb'
PURPLE = '#7c3aed'
GREEN  = '#10b981'
RED    = '#ef4444'

# Output directories: write to BOTH EN and ZH asset folders
OUT_DIRS = [
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/07-systems-and-phase-plane',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/07-稳定性理论',
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
# fig1: Six canonical phase portraits
# ---------------------------------------------------------------------------
def fig1_phase_portraits():
    print('Building fig1_phase_portraits...')
    cases = [
        ('Stable Spiral',   np.array([[-0.3,  1.0], [-1.0, -0.3]]), BLUE),
        ('Unstable Spiral', np.array([[ 0.3,  1.0], [-1.0,  0.3]]), RED),
        ('Stable Node',     np.array([[-1.0,  0.0], [ 0.0, -2.0]]), GREEN),
        ('Saddle Point',    np.array([[ 1.0,  0.0], [ 0.0, -1.0]]), PURPLE),
        ('Center',          np.array([[ 0.0,  1.0], [-1.0,  0.0]]), BLUE),
        ('Degenerate Node', np.array([[-1.0,  1.0], [ 0.0, -1.0]]), GREEN),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    t_fwd = np.linspace(0, 6, 600)
    t_bwd = np.linspace(0, -6, 600)

    for ax, (title, A, col) in zip(axes.flat, cases):
        # Vector field
        x = np.linspace(-2.5, 2.5, 22)
        y = np.linspace(-2.5, 2.5, 22)
        X, Y = np.meshgrid(x, y)
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y
        speed = np.sqrt(U**2 + V**2) + 1e-9
        ax.streamplot(X, Y, U, V, color=speed, cmap='Greys',
                      density=0.9, linewidth=0.8, arrowsize=0.8)

        # Trajectories
        rng = np.random.default_rng(7)
        for _ in range(8):
            x0 = rng.uniform(-2, 2, 2)
            sol_f = odeint(lambda s, t, A=A: A @ s, x0, t_fwd)
            ax.plot(sol_f[:, 0], sol_f[:, 1], color=col, lw=1.4, alpha=0.9)
            sol_b = odeint(lambda s, t, A=A: A @ s, x0, t_bwd)
            ax.plot(sol_b[:, 0], sol_b[:, 1], color=col, lw=0.7,
                    alpha=0.4, linestyle='--')

        ax.plot(0, 0, 'o', color='black', ms=6, zorder=10)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold', color=col)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])

    fig.suptitle('Canonical 2D Phase Portraits  (six linear types)',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    save(fig, 'fig1_phase_portraits.png')


# ---------------------------------------------------------------------------
# fig2: Damped pendulum + Lotka-Volterra on vector field
# ---------------------------------------------------------------------------
def fig2_trajectory_vector_field():
    print('Building fig2_trajectory_vector_field...')
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # --- Damped pendulum
    gamma, w0 = 0.25, 1.0
    def pend(s, t):
        x, y = s
        return [y, -gamma * y - w0**2 * np.sin(x)]

    ax = axes[0]
    th = np.linspace(-3 * np.pi, 3 * np.pi, 30)
    om = np.linspace(-3.2, 3.2, 18)
    TH, OM = np.meshgrid(th, om)
    U = OM
    V = -gamma * OM - w0**2 * np.sin(TH)
    spd = np.sqrt(U**2 + V**2) + 1e-9
    ax.streamplot(TH, OM, U, V, color=spd, cmap='Blues',
                  density=1.2, linewidth=0.8, arrowsize=0.8)

    t = np.linspace(0, 40, 4000)
    starts = [(-2.7 * np.pi, 2.4), (-1.6 * np.pi, 0.5),
              (-0.4 * np.pi, 1.5), (0.6 * np.pi, 2.6),
              (1.5 * np.pi, -1.0), (2.4 * np.pi, -2.5)]
    cols = [BLUE, PURPLE, GREEN, RED, BLUE, PURPLE]
    for s0, c in zip(starts, cols):
        sol = odeint(pend, s0, t)
        ax.plot(sol[:, 0], sol[:, 1], color=c, lw=1.2, alpha=0.9)

    for n in range(-2, 3):
        if n % 2 == 0:
            ax.plot(n * np.pi, 0, 'o', color=GREEN, ms=10,
                    markeredgecolor='white', mew=1.5, zorder=10)
        else:
            ax.plot(n * np.pi, 0, 'X', color=RED, ms=11,
                    markeredgecolor='white', mew=1.5, zorder=10)

    ax.set_xlim(-3 * np.pi, 3 * np.pi)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel(r'$\theta$', fontsize=12)
    ax.set_ylabel(r'$\omega$', fontsize=12)
    ax.set_title('Damped Pendulum:  saddles (X) and stable foci (o)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_xticklabels([r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$'])

    # --- Lotka-Volterra
    a, b, c, d = 1.1, 0.4, 0.4, 0.1
    def lv(s, t):
        x, y = s
        return [a * x - b * x * y, -c * y + d * x * y]

    ax = axes[1]
    xx = np.linspace(0.05, 10, 25)
    yy = np.linspace(0.05, 6, 18)
    XX, YY = np.meshgrid(xx, yy)
    U = a * XX - b * XX * YY
    V = -c * YY + d * XX * YY
    spd = np.sqrt(U**2 + V**2) + 1e-9
    ax.streamplot(XX, YY, U, V, color=spd, cmap='Greens',
                  density=1.2, linewidth=0.8, arrowsize=0.8)

    t = np.linspace(0, 40, 4000)
    for r, col in zip([0.5, 1.0, 1.6, 2.4], [BLUE, PURPLE, GREEN, RED]):
        sol = odeint(lv, [c/d + r, a/b], t)
        ax.plot(sol[:, 0], sol[:, 1], color=col, lw=1.4, alpha=0.9,
                label=f'orbit {r:.1f}')

    ax.plot(c/d, a/b, '*', color='black', ms=14, zorder=10,
            markeredgecolor='white', mew=1.0, label='center')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_xlabel('Prey  $x$', fontsize=12)
    ax.set_ylabel('Predator  $y$', fontsize=12)
    ax.set_title('Lotka-Volterra:  closed orbits around a center',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig.suptitle('Trajectories overlaid on vector fields',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig2_trajectory_vector_field.png')


# ---------------------------------------------------------------------------
# fig3: Trace-determinant classification
# ---------------------------------------------------------------------------
def fig3_eigenvalue_classification():
    print('Building fig3_eigenvalue_classification...')
    fig, ax = plt.subplots(figsize=(11, 7.5))

    tau = np.linspace(-4, 4, 500)
    parabola = tau**2 / 4

    ax.fill_between(tau, parabola, 6, where=(tau < 0),
                    color=BLUE, alpha=0.18, label='Stable spiral')
    ax.fill_between(tau, parabola, 6, where=(tau > 0),
                    color=RED, alpha=0.18, label='Unstable spiral')
    ax.fill_between(tau, 0, parabola, where=(tau < 0),
                    color=GREEN, alpha=0.20, label='Stable node')
    ax.fill_between(tau, 0, parabola, where=(tau > 0),
                    color='#fb923c', alpha=0.20, label='Unstable node')
    ax.fill_between(tau, -3, 0, color=PURPLE, alpha=0.15, label='Saddle')

    ax.plot(tau, parabola, color='black', lw=1.8,
            label=r'$\tau^2 = 4\Delta$  (degenerate)')
    ax.axhline(0, color='black', lw=1.0)
    ax.axvline(0, color='black', lw=1.0)
    ax.plot([0, 0], [0, 6], color=BLUE, lw=2.5,
            label=r'$\tau=0,\,\Delta>0$  (center)')

    annotations = [
        (-2.5, 4.0, 'Stable\nSpiral',   BLUE),
        ( 2.5, 4.0, 'Unstable\nSpiral', RED),
        (-2.5, 0.6, 'Stable Node',      GREEN),
        ( 2.5, 0.6, 'Unstable Node',    '#fb923c'),
        ( 0.0,-1.5, 'Saddle Point',     PURPLE),
        ( 0.2, 5.4, 'Center',           BLUE),
    ]
    for x, y, txt, col in annotations:
        ax.text(x, y, txt, ha='center', va='center', fontsize=11,
                fontweight='bold', color=col,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=col, lw=1.5, alpha=0.92))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 6)
    ax.set_xlabel(r'Trace  $\tau = \mathrm{tr}(A)$', fontsize=12)
    ax.set_ylabel(r'Determinant  $\Delta = \det(A)$', fontsize=12)
    ax.set_title('Trace-Determinant Plane:  classification of 2D linear systems',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95, ncol=2)
    fig.tight_layout()
    save(fig, 'fig3_eigenvalue_classification.png')


# ---------------------------------------------------------------------------
# fig4: Lyapunov function + decay
# ---------------------------------------------------------------------------
def fig4_lyapunov_function():
    print('Building fig4_lyapunov_function...')
    fig = plt.figure(figsize=(13.5, 5.6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    # System: x' = -x + y, y' = -2y  (asymptotically stable origin)
    def sys(s, t):
        x, y = s
        return [-x + y, -2 * y]

    # V(x,y) = x^2 + 0.5*y^2 -- Lyapunov function
    x = np.linspace(-2, 2, 80)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    V = X**2 + 0.5 * Y**2

    ax1.plot_surface(X, Y, V, cmap='viridis', alpha=0.78,
                     edgecolor='none', antialiased=True)
    ax1.contour(X, Y, V, levels=8, cmap='viridis', offset=0,
                linewidths=0.9, alpha=0.7)

    t = np.linspace(0, 5, 400)
    starts = [(1.8, -1.5), (-1.6, 1.2), (1.5, 1.5)]
    cols = [BLUE, PURPLE, GREEN]
    for s0, c in zip(starts, cols):
        sol = odeint(sys, s0, t)
        Vt = sol[:, 0]**2 + 0.5 * sol[:, 1]**2
        ax1.plot(sol[:, 0], sol[:, 1], Vt, color=c, lw=2.2)
        ax1.plot(sol[:, 0], sol[:, 1], 0, color=c, lw=1.0,
                 alpha=0.5, linestyle='--')

    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_zlabel('V(x,y)')
    ax1.set_title(r'Lyapunov surface  $V=x^2+\frac{1}{2}y^2$' +
                  '\n trajectories descend the bowl',
                  fontsize=11, fontweight='bold')

    # Right panel: V(t) decays monotonically
    for s0, c in zip(starts, cols):
        sol = odeint(sys, s0, t)
        Vt = sol[:, 0]**2 + 0.5 * sol[:, 1]**2
        ax2.plot(t, Vt, color=c, lw=2.2,
                 label=f'start ({s0[0]:.1f}, {s0[1]:.1f})')

    ax2.set_xlabel('time  $t$', fontsize=12)
    ax2.set_ylabel(r'$V(\mathbf{x}(t))$', fontsize=12)
    ax2.set_title(r'$V$ decreases monotonically  ($\dot V \leq 0$)',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_yscale('log')

    fig.suptitle("Lyapunov's Direct Method:  proving stability without solving the ODE",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig4_lyapunov_function.png')


# ---------------------------------------------------------------------------
# fig5: Four bifurcation normal forms
# ---------------------------------------------------------------------------
def fig5_bifurcation_normal_forms():
    print('Building fig5_bifurcation_normal_forms...')
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    # 1) Saddle-node:  x_dot = r + x^2  =>  x* = +/- sqrt(-r)
    ax = axes[0, 0]
    r_neg = np.linspace(-2, 0, 200)
    ax.plot(r_neg, +np.sqrt(-r_neg), color=GREEN, lw=2.5, label='stable')
    ax.plot(r_neg, -np.sqrt(-r_neg), color=RED, lw=2.5, ls='--', label='unstable')
    ax.axhline(0, color='gray', lw=0.6)
    ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_xlabel('parameter  $r$'); ax.set_ylabel(r'equilibrium  $x^*$')
    ax.set_title(r'Saddle-Node  $\dot x = r + x^2$', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # 2) Transcritical:  x_dot = r x - x^2  =>  x*=0, x*=r
    ax = axes[0, 1]
    rr = np.linspace(-2, 2, 200)
    # x*=0: stable for r<0, unstable for r>0
    ax.plot(rr[rr <= 0], np.zeros_like(rr[rr <= 0]), color=GREEN, lw=2.5)
    ax.plot(rr[rr >= 0], np.zeros_like(rr[rr >= 0]), color=RED, lw=2.5, ls='--')
    # x*=r: opposite
    ax.plot(rr[rr <= 0], rr[rr <= 0], color=RED, lw=2.5, ls='--')
    ax.plot(rr[rr >= 0], rr[rr >= 0], color=GREEN, lw=2.5)
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_xlabel('parameter  $r$'); ax.set_ylabel(r'equilibrium  $x^*$')
    ax.set_title(r'Transcritical  $\dot x = rx - x^2$', fontsize=12, fontweight='bold')

    # 3) Pitchfork (supercritical):  x_dot = r x - x^3
    ax = axes[1, 0]
    r_pos = np.linspace(0, 2, 200)
    ax.plot(rr[rr <= 0], np.zeros_like(rr[rr <= 0]), color=GREEN, lw=2.5)
    ax.plot(rr[rr >= 0], np.zeros_like(rr[rr >= 0]), color=RED, lw=2.5, ls='--')
    ax.plot(r_pos, +np.sqrt(r_pos), color=GREEN, lw=2.5)
    ax.plot(r_pos, -np.sqrt(r_pos), color=GREEN, lw=2.5)
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlim(-2, 2); ax.set_ylim(-1.7, 1.7)
    ax.set_xlabel('parameter  $r$'); ax.set_ylabel(r'equilibrium  $x^*$')
    ax.set_title(r'Pitchfork  $\dot x = rx - x^3$', fontsize=12, fontweight='bold')

    # 4) Hopf bifurcation: limit cycle radius sqrt(mu) for mu>0
    ax = axes[1, 1]
    mu_vals = [-0.4, 0.0, 0.4, 0.8]
    cols = [GREEN, '#888', PURPLE, RED]
    theta = np.linspace(0, 2*np.pi, 200)
    for mu, c in zip(mu_vals, cols):
        if mu > 0:
            R = np.sqrt(mu)
            ax.plot(R * np.cos(theta), R * np.sin(theta), color=c,
                    lw=2.2, label=fr'$\mu={mu:+.1f}$  cycle  $R={R:.2f}$')
        else:
            ax.plot(0, 0, 'o', color=c, ms=10,
                    label=fr'$\mu={mu:+.1f}$  stable focus')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(r'Hopf:  $\dot r = \mu r - r^3$,  $\dot\theta = 1$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Bifurcations:  four normal forms',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    save(fig, 'fig5_bifurcation_normal_forms.png')


if __name__ == '__main__':
    fig1_phase_portraits()
    fig2_trajectory_vector_field()
    fig3_eigenvalue_classification()
    fig4_lyapunov_function()
    fig5_bifurcation_normal_forms()
    print('\nChapter 7 figures complete.')
