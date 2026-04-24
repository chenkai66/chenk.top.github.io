"""
Chapter 10: Bifurcation Theory -- saddle-node, transcritical, pitchfork, Hopf, normal forms.

Figures (saved to BOTH the EN and ZH asset folders):
  fig1_saddle_node_bifurcation.png   -- vector field flow + bifurcation diagram for x' = mu - x^2
  fig2_transcritical_bifurcation.png -- exchange of stability for x' = mu*x - x^2
  fig3_pitchfork_bifurcation.png     -- supercritical vs subcritical pitchfork (with hysteresis)
  fig4_hopf_bifurcation.png          -- phase portraits + 3D paraboloid for limit-cycle birth
  fig5_normal_forms_overview.png     -- summary panel of all four codimension-1 normal forms
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

OUT_DIRS = [
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/10-bifurcation-theory',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/10-分岔理论',
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
# fig1: saddle-node bifurcation x' = mu - x^2
# ---------------------------------------------------------------------------
def fig1_saddle_node():
    print('Building fig1_saddle_node_bifurcation...')
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)

    # Left: phase line for several mu
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(-2.2, 2.2, 400)
    mus = [-0.6, -0.1, 0.0, 0.4, 1.0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(mus)))
    for mu, c in zip(mus, cmap):
        ax.plot(x, mu - x**2, color=c, lw=1.8, label=fr'$\mu={mu:+.1f}$')
        if mu > 0:
            xs = np.sqrt(mu)
            ax.plot( xs, 0, 'o', color=c, ms=9, mec='white', mew=1.2)
            ax.plot(-xs, 0, 's', color=c, ms=9, mec='white', mew=1.2)
        elif mu == 0:
            ax.plot(0, 0, 'D', color=c, ms=9, mec='white', mew=1.2)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('$x$'); ax.set_ylabel(r'$\dot{x} = \mu - x^2$')
    ax.set_title('Phase function for varying $\\mu$\n'
                 '(circle = stable, square = unstable, diamond = semi-stable)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower center', ncol=5, fontsize=9)
    ax.set_ylim(-3, 1.2)

    # Right: bifurcation diagram + flow arrows
    ax = fig.add_subplot(gs[0, 1])
    mu_pos = np.linspace(0, 2.0, 200)
    ax.plot(mu_pos, np.sqrt(mu_pos), color=BLUE, lw=3, label='stable branch')
    ax.plot(mu_pos, -np.sqrt(mu_pos), color=RED, lw=3, ls='--',
            label='unstable branch')
    # Arrows showing flow direction at fixed mu values
    for mu_v in [0.4, 1.0, 1.6]:
        xs = np.sqrt(mu_v)
        # below stable branch: arrow up
        ax.annotate('', xy=(mu_v, xs - 0.05), xytext=(mu_v, xs - 0.6),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
        # above stable branch: arrow down
        ax.annotate('', xy=(mu_v, xs + 0.05), xytext=(mu_v, xs + 0.6),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
        # below unstable branch: arrow down
        ax.annotate('', xy=(mu_v, -xs - 0.6), xytext=(mu_v, -xs - 0.05),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5,
                                    alpha=0.6))
    ax.scatter([0], [0], color='black', s=110, zorder=5,
               edgecolor='white', linewidth=1.5)
    ax.annotate('saddle-node\n(fold) point', xy=(0, 0), xytext=(-0.7, 1.0),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.axvspan(-1.0, 0.0, color='gray', alpha=0.10)
    ax.text(-0.5, -1.7, 'no equilibria', fontsize=10, ha='center',
            color='gray', fontstyle='italic')
    ax.set_xlim(-1.0, 2.0); ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel(r'parameter $\mu$', fontsize=11)
    ax.set_ylabel(r'equilibrium $x^*$', fontsize=11)
    ax.set_title('Bifurcation diagram: $\\dot{x} = \\mu - x^2$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Saddle-node bifurcation: two equilibria collide and annihilate',
                 fontsize=13, fontweight='bold', y=1.02)
    save(fig, 'fig1_saddle_node_bifurcation.png')


# ---------------------------------------------------------------------------
# fig2: transcritical bifurcation x' = mu*x - x^2
# ---------------------------------------------------------------------------
def fig2_transcritical():
    print('Building fig2_transcritical_bifurcation...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={'width_ratios': [1.0, 1.1]})

    # Left: time series showing exchange of stability
    ax = axes[0]
    t = np.linspace(0, 8, 600)
    for mu, color, ls in [(-0.8, BLUE, '-'), (0.8, RED, '-')]:
        for x0 in [0.05, 0.3, 0.6, 0.9, 1.2]:
            sol = odeint(lambda x, t: mu * x - x**2, x0, t).ravel()
            ax.plot(t, sol, color=color, lw=0.9, alpha=0.8)
    # Single representative stable line per regime
    ax.axhline(0, color=BLUE, lw=2.4, alpha=0.4)
    ax.text(7.6, -0.07, r'$\mu<0$: $x^*=0$ stable', color=BLUE,
            fontsize=10, ha='right')
    ax.text(7.6, 0.85, r'$\mu>0$: $x^*=\mu$ stable', color=RED,
            fontsize=10, ha='right')
    ax.set_xlabel('time $t$'); ax.set_ylabel('$x(t)$')
    ax.set_title('Trajectories converge to whichever branch is stable',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(-0.2, 1.4)

    # Right: bifurcation diagram (the X shape)
    ax = axes[1]
    mu = np.linspace(-1.5, 1.5, 400)
    # Branch x* = 0
    ax.plot(mu[mu <= 0], np.zeros(np.sum(mu <= 0)),
            color=BLUE, lw=3, label=r'$x^*=0$ stable')
    ax.plot(mu[mu >= 0], np.zeros(np.sum(mu >= 0)),
            color=RED, lw=3, ls='--', label=r'$x^*=0$ unstable')
    # Branch x* = mu
    ax.plot(mu[mu <= 0], mu[mu <= 0],
            color=RED, lw=3, ls='--', label=r'$x^*=\mu$ unstable')
    ax.plot(mu[mu >= 0], mu[mu >= 0],
            color=BLUE, lw=3, label=r'$x^*=\mu$ stable')
    ax.scatter([0], [0], color='black', s=130, zorder=10,
               edgecolor='white', linewidth=1.5)
    ax.annotate('exchange of\nstability', xy=(0, 0), xytext=(0.45, -1.0),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.axvline(0, color='gray', lw=0.6, ls=':')
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'parameter $\mu$', fontsize=11)
    ax.set_ylabel(r'equilibrium $x^*$', fontsize=11)
    ax.set_title('Bifurcation diagram: $\\dot{x} = \\mu x - x^2$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Transcritical bifurcation: two branches swap stability at $\\mu=0$',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig2_transcritical_bifurcation.png')


# ---------------------------------------------------------------------------
# fig3: pitchfork bifurcations -- super and sub
# ---------------------------------------------------------------------------
def fig3_pitchfork():
    print('Building fig3_pitchfork_bifurcation...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Supercritical: x' = mu*x - x^3
    ax = axes[0]
    mu = np.linspace(-1.5, 1.5, 400)
    # Trivial branch
    ax.plot(mu[mu <= 0], np.zeros(np.sum(mu <= 0)),
            color=BLUE, lw=3)
    ax.plot(mu[mu >= 0], np.zeros(np.sum(mu >= 0)),
            color=RED, lw=3, ls='--')
    # Bifurcating arms
    mp = mu[mu >= 0]
    ax.plot(mp, np.sqrt(mp), color=BLUE, lw=3, label='stable')
    ax.plot(mp, -np.sqrt(mp), color=BLUE, lw=3)
    ax.plot([], [], color=RED, lw=3, ls='--', label='unstable')
    ax.scatter([0], [0], color='black', s=110, zorder=10,
               edgecolor='white', linewidth=1.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$\mu$', fontsize=11)
    ax.set_ylabel(r'$x^*$', fontsize=11)
    ax.set_title('Supercritical pitchfork  $\\dot{x} = \\mu x - x^3$\n'
                 '("soft" symmetry breaking)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    # Subcritical: x' = mu*x + x^3 (with x^5 stabilization for hysteresis)
    # Use x' = mu*x + x^3 - x^5 to show full hysteresis loop
    ax = axes[1]
    # Solve for steady states of mu*x + x^3 - x^5 = 0  -> x^4 - x^2 - mu = 0 (x!=0)
    # let u = x^2, u^2 - u - mu = 0  -> u = (1 +/- sqrt(1+4mu))/2
    mu_full = np.linspace(-0.30, 1.0, 600)
    valid = mu_full >= -0.25
    u_plus  = (1 + np.sqrt(np.maximum(0, 1 + 4*mu_full))) / 2
    u_minus = (1 - np.sqrt(np.maximum(0, 1 + 4*mu_full))) / 2
    # outer branch (stable big amplitude)
    x_outer = np.sqrt(np.maximum(0, u_plus))
    # inner branch (unstable, exists for -1/4 <= mu <= 0)
    inner_mask = (mu_full >= -0.25) & (mu_full <= 0)
    x_inner = np.sqrt(np.maximum(0, u_minus))
    # trivial
    ax.plot(mu_full[mu_full <= 0], np.zeros(np.sum(mu_full <= 0)),
            color=BLUE, lw=3)
    ax.plot(mu_full[mu_full > 0], np.zeros(np.sum(mu_full > 0)),
            color=RED, lw=3, ls='--')
    # outer stable
    ax.plot(mu_full[valid],  x_outer[valid], color=BLUE, lw=3)
    ax.plot(mu_full[valid], -x_outer[valid], color=BLUE, lw=3)
    # inner unstable
    ax.plot(mu_full[inner_mask],  x_inner[inner_mask],
            color=RED, lw=3, ls='--')
    ax.plot(mu_full[inner_mask], -x_inner[inner_mask],
            color=RED, lw=3, ls='--')
    # bifurcation points
    ax.scatter([0, -0.25, -0.25], [0, np.sqrt(0.5), -np.sqrt(0.5)],
               color='black', s=90, zorder=10,
               edgecolor='white', linewidth=1.5)
    # hysteresis arrows
    ax.annotate('', xy=(0.02, 1.05), xytext=(0.02, 0.05),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2))
    ax.annotate('', xy=(-0.27, 0.05), xytext=(-0.27, np.sqrt(0.5) - 0.02),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2))
    ax.text(0.05, 0.55, 'sudden\njump up', fontsize=9, color=PURPLE,
            fontweight='bold')
    ax.text(-0.55, 0.35, 'jump\ndown', fontsize=9, color=PURPLE,
            fontweight='bold')
    ax.set_xlim(-0.7, 1.0); ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel(r'$\mu$', fontsize=11)
    ax.set_ylabel(r'$x^*$', fontsize=11)
    ax.set_title('Subcritical pitchfork  $\\dot{x}=\\mu x + x^3 - x^5$\n'
                 '("hard" jump + hysteresis loop)',
                 fontsize=11, fontweight='bold')

    fig.suptitle('Pitchfork bifurcations: symmetric splitting, two flavors',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig3_pitchfork_bifurcation.png')


# ---------------------------------------------------------------------------
# fig4: Hopf bifurcation -- phase portraits + 3D paraboloid
# ---------------------------------------------------------------------------
def fig4_hopf():
    print('Building fig4_hopf_bifurcation...')
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.35,
                          wspace=0.30)

    omega = 1.0
    def hopf(s, t, mu):
        x, y = s
        r2 = x*x + y*y
        return [mu*x - omega*y - x*r2, omega*x + mu*y - y*r2]

    t = np.linspace(0, 50, 4000)
    mus = [-0.4, 0.0, 0.4]
    titles = [r'$\mu=-0.4$  stable focus',
              r'$\mu=0$  marginal',
              r'$\mu=+0.4$  stable limit cycle']
    initial = [(1.5, 0), (-1.5, 0.3), (0.05, 0.05), (1.0, -1.2), (-0.8, 1.0)]

    for j, (mu, title) in enumerate(zip(mus, titles)):
        ax = fig.add_subplot(gs[0, j])
        for x0, y0 in initial:
            sol = odeint(hopf, [x0, y0], t, args=(mu,))
            ax.plot(sol[:, 0], sol[:, 1], color=BLUE, lw=0.7, alpha=0.65)
            ax.plot(x0, y0, 'o', color=GREEN, ms=4, mec='white', mew=0.6)
        if mu > 0:
            theta = np.linspace(0, 2*np.pi, 200)
            r = np.sqrt(mu)
            ax.plot(r*np.cos(theta), r*np.sin(theta),
                    color=RED, lw=3, label=fr'limit cycle  $r=\sqrt{{\mu}}={r:.2f}$')
            ax.legend(loc='lower right', fontsize=9)
        ax.plot(0, 0, 'X', color='black', ms=10, mec='white', mew=1.0,
                zorder=10)
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title(title, fontsize=11, fontweight='bold')

    # Bottom: 3D paraboloid showing limit-cycle radius vs mu
    ax3d = fig.add_subplot(gs[1, :], projection='3d')
    mu_grid = np.linspace(-0.5, 1.0, 60)
    theta = np.linspace(0, 2*np.pi, 80)
    M, T_ = np.meshgrid(mu_grid, theta)
    R = np.sqrt(np.maximum(0, M))
    X = R * np.cos(T_); Y = R * np.sin(T_)
    surf = ax3d.plot_surface(M, X, Y, cmap='viridis', alpha=0.55,
                             linewidth=0, antialiased=True, edgecolor='none')
    # Trivial axis (origin) -- stable then unstable
    ax3d.plot(mu_grid[mu_grid <= 0], np.zeros(np.sum(mu_grid <= 0)),
              np.zeros(np.sum(mu_grid <= 0)),
              color=BLUE, lw=4, label='origin stable')
    ax3d.plot(mu_grid[mu_grid >= 0], np.zeros(np.sum(mu_grid >= 0)),
              np.zeros(np.sum(mu_grid >= 0)),
              color=RED, lw=4, ls='--', label='origin unstable')
    ax3d.set_xlabel(r'$\mu$'); ax3d.set_ylabel('$x$'); ax3d.set_zlabel('$y$')
    ax3d.set_title('Hopf paraboloid: limit-cycle amplitude $r=\\sqrt{\\mu}$ '
                   'grows continuously after bifurcation',
                   fontsize=11, fontweight='bold')
    ax3d.view_init(elev=22, azim=-65)
    ax3d.legend(loc='upper left', fontsize=9)

    fig.suptitle('Hopf bifurcation: birth of a limit cycle from a stable focus',
                 fontsize=13, fontweight='bold', y=0.99)
    save(fig, 'fig4_hopf_bifurcation.png')


# ---------------------------------------------------------------------------
# fig5: normal forms overview (4 codimension-1 bifurcations side by side)
# ---------------------------------------------------------------------------
def fig5_normal_forms():
    print('Building fig5_normal_forms_overview...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.42, wspace=0.30)

    mu = np.linspace(-1.5, 1.5, 400)

    # (1) saddle-node x' = mu - x^2
    ax = axes[0, 0]
    mu_pos = mu[mu >= 0]
    ax.plot(mu_pos, np.sqrt(mu_pos), color=BLUE, lw=3, label='stable')
    ax.plot(mu_pos, -np.sqrt(mu_pos), color=RED, lw=3, ls='--', label='unstable')
    ax.scatter([0], [0], color='black', s=80, zorder=10, edgecolor='white')
    ax.axvspan(-1.5, 0, color='gray', alpha=0.10)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$\mu$'); ax.set_ylabel(r'$x^*$')
    ax.set_title('Saddle-node:  $\\dot{x} = \\mu - x^2$\n'
                 'creation/annihilation',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    # (2) transcritical
    ax = axes[0, 1]
    ax.plot(mu[mu <= 0], np.zeros(np.sum(mu <= 0)), color=BLUE, lw=3)
    ax.plot(mu[mu >= 0], np.zeros(np.sum(mu >= 0)), color=RED, lw=3, ls='--')
    ax.plot(mu[mu <= 0], mu[mu <= 0], color=RED, lw=3, ls='--')
    ax.plot(mu[mu >= 0], mu[mu >= 0], color=BLUE, lw=3)
    ax.scatter([0], [0], color='black', s=80, zorder=10, edgecolor='white')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$\mu$'); ax.set_ylabel(r'$x^*$')
    ax.set_title('Transcritical:  $\\dot{x} = \\mu x - x^2$\n'
                 'exchange of stability',
                 fontsize=11, fontweight='bold')

    # (3) supercritical pitchfork
    ax = axes[1, 0]
    ax.plot(mu[mu <= 0], np.zeros(np.sum(mu <= 0)), color=BLUE, lw=3)
    ax.plot(mu[mu >= 0], np.zeros(np.sum(mu >= 0)), color=RED, lw=3, ls='--')
    mp = mu[mu >= 0]
    ax.plot(mp,  np.sqrt(mp), color=BLUE, lw=3)
    ax.plot(mp, -np.sqrt(mp), color=BLUE, lw=3)
    ax.scatter([0], [0], color='black', s=80, zorder=10, edgecolor='white')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$\mu$'); ax.set_ylabel(r'$x^*$')
    ax.set_title('Pitchfork:  $\\dot{x} = \\mu x - x^3$\n'
                 'symmetry breaking',
                 fontsize=11, fontweight='bold')

    # (4) Hopf -- amplitude vs mu
    ax = axes[1, 1]
    mp = mu[mu >= 0]
    ax.plot(mu[mu <= 0], np.zeros(np.sum(mu <= 0)), color=BLUE, lw=3,
            label='focus stable')
    ax.plot(mp, np.zeros_like(mp), color=RED, lw=3, ls='--',
            label='focus unstable')
    ax.fill_between(mp, -np.sqrt(mp), np.sqrt(mp), color=GREEN, alpha=0.20,
                    label='limit cycle')
    ax.plot(mp,  np.sqrt(mp), color=GREEN, lw=3)
    ax.plot(mp, -np.sqrt(mp), color=GREEN, lw=3)
    ax.scatter([0], [0], color='black', s=80, zorder=10, edgecolor='white')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$\mu$'); ax.set_ylabel(r'amplitude $r$')
    ax.set_title('Hopf:  $\\dot{r} = \\mu r - r^3$\n'
                 'birth of oscillation',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Codimension-1 normal forms: the four canonical bifurcations',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig5_normal_forms_overview.png')


if __name__ == '__main__':
    fig1_saddle_node()
    fig2_transcritical()
    fig3_pitchfork()
    fig4_hopf()
    fig5_normal_forms()
    print('\nChapter 10 figures complete.')
