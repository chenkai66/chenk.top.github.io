"""
Chapter 13: Introduction to Partial Differential Equations.

Figures (5):
  fig1_heat_equation_evolution.png  -- 1D heat equation: profile decay + spacetime + Fourier modes
  fig2_wave_propagation.png         -- 1D wave equation: d'Alembert split + standing wave + spacetime
  fig3_laplace_equation.png         -- 2D Laplace: heatmap + isolines + maximum principle
  fig4_separation_of_variables.png  -- Separation procedure visualised: eigenmodes -> superposition
  fig5_pde_classification.png       -- Discriminant plane + canonical examples (parabolic / hyperbolic / elliptic)

Standalone: writes the same five PNGs into both EN and ZH asset folders.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.linalg import solve_banded

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
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/13-pde-introduction',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/13-偏微分方程引论',
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
# fig1: Heat equation u_t = alpha u_xx with Dirichlet BCs
#   Crank-Nicolson, then plot (a) snapshots, (b) spacetime heatmap, (c) modes
# ---------------------------------------------------------------------------
def fig1_heat_equation_evolution():
    print('Building fig1_heat_equation_evolution...')
    L = 1.0
    alpha = 0.05
    Nx = 121
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    T_end = 5.0
    Nt = 1200
    dt = T_end / Nt

    # initial: triangle-ish with a bump
    u = np.zeros(Nx)
    u[(x > 0.15) & (x < 0.55)] = 1.0
    u[(x > 0.65) & (x < 0.85)] = 0.6
    u[0] = 0.0; u[-1] = 0.0

    r = alpha * dt / dx**2
    # Crank-Nicolson: (I - r/2 L) u^{n+1} = (I + r/2 L) u^n
    # interior tridiagonal
    n_in = Nx - 2
    main_A =  (1.0 + r) * np.ones(n_in)
    off_A  = -(r / 2.0) * np.ones(n_in - 1)
    main_B =  (1.0 - r) * np.ones(n_in)
    off_B  =  (r / 2.0) * np.ones(n_in - 1)

    # banded representation for solve_banded (l=u=1)
    ab = np.zeros((3, n_in))
    ab[0, 1:] = off_A
    ab[1, :]  = main_A
    ab[2, :-1] = off_A

    snaps_t = [0.0, 0.05, 0.2, 0.6, 1.5, 4.0]
    snaps = {snaps_t[0]: u.copy()}
    field = np.zeros((Nt + 1, Nx)); field[0] = u
    times = np.linspace(0, T_end, Nt + 1)
    for n in range(Nt):
        ui = u[1:-1]
        rhs = main_B * ui + np.concatenate(([0], off_B * ui[:-1])) \
                          + np.concatenate((off_B * ui[1:], [0]))
        new = solve_banded((1, 1), ab, rhs)
        u[1:-1] = new
        field[n + 1] = u
        for st in snaps_t:
            if st > 0 and abs(times[n + 1] - st) < dt / 2:
                snaps[st] = u.copy()

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32)

    # (a) snapshots
    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, len(snaps_t)))
    for c, t_s in zip(cmap, snaps_t):
        if t_s in snaps:
            ax.plot(x, snaps[t_s], color=c, lw=2.0, label=f't = {t_s:.2f}')
    ax.set_xlabel('position  x', fontsize=11)
    ax.set_ylabel('temperature  u(x, t)', fontsize=11)
    ax.set_title('Heat equation: profile decays toward equilibrium',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_xlim(0, L); ax.set_ylim(-0.05, 1.15)

    # (b) spacetime heatmap
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(field, aspect='auto', origin='lower', cmap='inferno',
                   extent=[0, L, 0, T_end], vmin=0, vmax=1.0)
    ax.set_xlabel('position  x', fontsize=11)
    ax.set_ylabel('time  t', fontsize=11)
    ax.set_title(r'Spacetime  $u(x,t)$  (smoothing forward in time)',
                 fontsize=12, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('u', fontsize=10)

    # (c) Fourier mode decay
    ax = fig.add_subplot(gs[0, 2])
    t_axis = np.linspace(0, T_end, 200)
    for n_mode, color in zip([1, 2, 3, 5, 10],
                             [BLUE, PURPLE, GREEN, RED, '#888']):
        decay = np.exp(-alpha * (n_mode * np.pi / L) ** 2 * t_axis)
        ax.plot(t_axis, decay, color=color, lw=2.0, label=f'n = {n_mode}')
    ax.set_xlabel('time  t', fontsize=11)
    ax.set_ylabel(r'mode amplitude  $\exp(-\alpha (n\pi/L)^2 t)$', fontsize=11)
    ax.set_title('High modes die first  (selective smoothing)',
                 fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1.2)
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(r'Heat equation  $u_t = \alpha\,u_{xx}$  '
                 r'with Dirichlet BCs ($\alpha = 0.05$)',
                 fontsize=13.5, fontweight='bold', y=1.02)
    save(fig, 'fig1_heat_equation_evolution.png')


# ---------------------------------------------------------------------------
# fig2: Wave equation -- d'Alembert split, standing wave, spacetime cone
# ---------------------------------------------------------------------------
def fig2_wave_propagation():
    print('Building fig2_wave_propagation...')
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05],
                          hspace=0.42, wspace=0.32)

    c = 1.0
    L = 4.0
    x = np.linspace(-L, L, 800)

    # --- (a) d'Alembert: f(x) = bump
    def f(z):
        return np.exp(-(z) ** 2 * 8.0)

    ax = fig.add_subplot(gs[0, 0])
    ts = [0.0, 0.6, 1.2, 1.8]
    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(ts)))
    for t_v, color in zip(ts, cmap):
        u = 0.5 * (f(x - c * t_v) + f(x + c * t_v))
        ax.plot(x, u + (ts.index(t_v) * 0.0), color=color, lw=2.2,
                label=f't = {t_v:.1f}')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('u(x, t)', fontsize=11)
    ax.set_title("d'Alembert: pulse splits into right + left", fontsize=12,
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # --- (b) standing wave (string fixed at both ends)
    ax = fig.add_subplot(gs[0, 1])
    Lstr = 1.0
    xs = np.linspace(0, Lstr, 400)
    n_mode = 3; omega = n_mode * np.pi * c
    ts = np.linspace(0, 2 * np.pi / omega, 7)[:6]
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, len(ts)))
    for t_v, color in zip(ts, cmap):
        u = np.sin(n_mode * np.pi * xs / Lstr) * np.cos(omega * t_v)
        ax.plot(xs, u, color=color, lw=2.0, alpha=0.9)
    # node markers
    for k in range(n_mode + 1):
        ax.scatter([k / n_mode], [0], color=RED, s=50, zorder=5,
                   edgecolor='white', linewidth=1.2)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('u(x, t)', fontsize=11)
    ax.set_title(f'Standing wave  (mode n = {n_mode}, fixed ends)\n'
                 'red dots = nodes (always at rest)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, Lstr); ax.set_ylim(-1.2, 1.25)

    # --- (c) spacetime cone (light cone) for d'Alembert
    ax = fig.add_subplot(gs[0, 2])
    Tmax = 2.5
    tt = np.linspace(0, Tmax, 300)
    xx = np.linspace(-L, L, 600)
    XX, TT = np.meshgrid(xx, tt)
    U = 0.5 * (f(XX - c * TT) + f(XX + c * TT))
    im = ax.imshow(U, aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[-L, L, 0, Tmax], vmin=-1, vmax=1)
    # light cone
    ax.plot([0, c * Tmax], [0, Tmax], color='black', lw=1.2, ls='--')
    ax.plot([0, -c * Tmax], [0, Tmax], color='black', lw=1.2, ls='--')
    ax.text(c * Tmax * 0.5, Tmax * 0.55, ' x = +ct',
            color='black', fontsize=10, rotation=58)
    ax.text(-c * Tmax * 0.55, Tmax * 0.55, 'x = -ct',
            color='black', fontsize=10, rotation=-58)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('t', fontsize=11)
    ax.set_title('Spacetime  u(x,t):  finite propagation speed',
                 fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # --- (d) finite-difference numerical: snapshots
    Nx = 401
    x2 = np.linspace(-L, L, Nx)
    dx = x2[1] - x2[0]
    sigma = 0.7  # CFL = c dt / dx
    dt = sigma * dx / c
    Nt = int(2.5 / dt)

    u_prev = f(x2)
    u = u_prev.copy()
    snaps = [(0, u_prev.copy())]
    targets = [int(0.5 / dt), int(1.0 / dt), int(1.5 / dt), int(2.0 / dt)]
    for n in range(1, Nt + 1):
        u_new = np.zeros_like(u)
        u_new[1:-1] = (2 * (1 - sigma**2) * u[1:-1] - u_prev[1:-1] +
                       sigma**2 * (u[2:] + u[:-2]))
        # absorbing-ish boundaries
        u_new[0] = 0; u_new[-1] = 0
        u_prev, u = u, u_new
        if n in targets:
            snaps.append((n * dt, u.copy()))

    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, len(snaps)))
    for (t_v, uu), color in zip(snaps, cmap):
        ax.plot(x2, uu, color=color, lw=1.8, label=f't = {t_v:.2f}')
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('u(x, t)', fontsize=11)
    ax.set_title(f'FD scheme  (CFL $\\sigma = {sigma}$, stable)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # --- (e) CFL violation
    sigma_bad = 1.05
    dt = sigma_bad * dx / c
    Nt = int(0.5 / dt)
    u_prev = f(x2)
    u = u_prev.copy()
    for n in range(1, Nt + 1):
        u_new = np.zeros_like(u)
        u_new[1:-1] = (2 * (1 - sigma_bad**2) * u[1:-1] - u_prev[1:-1] +
                       sigma_bad**2 * (u[2:] + u[:-2]))
        u_new[0] = 0; u_new[-1] = 0
        u_prev, u = u, u_new
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x2, u, color=RED, lw=1.5)
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('u(x, t)', fontsize=11)
    ax.set_title(f'CFL violation  ($\\sigma = {sigma_bad}$)\n'
                 'numerical blow-up: amplitude $\\to \\infty$',
                 fontsize=12, fontweight='bold')

    # --- (f) characteristics
    ax = fig.add_subplot(gs[1, 2])
    for x0 in np.linspace(-3, 3, 9):
        ax.plot([x0, x0 + c * Tmax], [0, Tmax], color=BLUE, lw=0.9, alpha=0.7)
        ax.plot([x0, x0 - c * Tmax], [0, Tmax], color=RED,  lw=0.9, alpha=0.7)
    ax.set_xlim(-L, L); ax.set_ylim(0, Tmax)
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('t', fontsize=11)
    ax.set_title('Characteristics  $x \\pm ct = $ const',
                 fontsize=12, fontweight='bold')
    ax.text(-3.5, 2.0, 'right-going\n(blue)', color=BLUE, fontsize=9)
    ax.text( 1.5, 2.0, 'left-going\n(red)',   color=RED,  fontsize=9)

    fig.suptitle(r'Wave equation  $u_{tt} = c^2\,u_{xx}$  '
                 r'(c = 1):  propagation, standing waves, CFL',
                 fontsize=13.5, fontweight='bold', y=1.00)
    save(fig, 'fig2_wave_propagation.png')


# ---------------------------------------------------------------------------
# fig3: Laplace equation -- Jacobi solve on a square with mixed BCs
# ---------------------------------------------------------------------------
def fig3_laplace_equation():
    print('Building fig3_laplace_equation...')
    N = 91
    x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((N, N))

    # Boundary: top hot bump, bottom 0, sides 0
    u[-1, :] = np.exp(-((x - 0.5) / 0.18) ** 2) * 1.0
    u[0, :]  = 0.0
    u[:, 0]  = 0.0
    u[:, -1] = 0.0

    # Gauss-Seidel
    for it in range(8000):
        u_old = u.copy()
        u[1:-1, 1:-1] = 0.25 * (u[1:-1, 2:] + u[1:-1, :-2] +
                                u[2:, 1:-1] + u[:-2, 1:-1])
        # restore boundary
        u[-1, :] = np.exp(-((x - 0.5) / 0.18) ** 2)
        u[0, :]  = 0.0
        u[:, 0]  = 0.0
        u[:, -1] = 0.0
        if np.max(np.abs(u - u_old)) < 1e-7:
            break
    print(f'    Jacobi/GS iterations = {it}')

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.32)

    # (a) heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(u, origin='lower', cmap='inferno', extent=[0, 1, 0, 1])
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('y', fontsize=11)
    ax.set_title(r'Solution to  $\nabla^2 u = 0$  (top: hot bump)',
                 fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # (b) isolines + gradient arrows
    ax = fig.add_subplot(gs[0, 1])
    cs = ax.contour(X, Y, u, levels=12, cmap='viridis', linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    Gy, Gx = np.gradient(u, y, x)
    sk = 8
    ax.quiver(X[::sk, ::sk], Y[::sk, ::sk], -Gx[::sk, ::sk], -Gy[::sk, ::sk],
              color=BLUE, alpha=0.7, scale=8)
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('y', fontsize=11)
    ax.set_title('Equipotentials  +  $-\\nabla u$  (heat-flux direction)',
                 fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

    # (c) maximum principle illustration: scatter of values at random interior pts
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(0)
    pts_x = rng.uniform(0.05, 0.95, 600)
    pts_y = rng.uniform(0.05, 0.95, 600)
    pts_u = []
    for xv, yv in zip(pts_x, pts_y):
        i = int(yv * (N - 1)); j = int(xv * (N - 1))
        pts_u.append(u[i, j])
    pts_u = np.array(pts_u)
    bd_max = u[-1, :].max()
    ax.scatter(pts_x, pts_u, c=pts_u, cmap='inferno', s=20, alpha=0.85,
               edgecolor='none')
    ax.axhline(bd_max, color=RED, lw=1.4, ls='--',
               label=f'max on boundary = {bd_max:.2f}')
    ax.axhline(0, color=BLUE, lw=1.4, ls='--', label='min on boundary = 0.00')
    ax.set_xlabel('x of interior sample', fontsize=11)
    ax.set_ylabel('u value', fontsize=11)
    ax.set_title('Maximum principle:  interior values stay between\n'
                 'boundary min and boundary max',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(r'Laplace equation  $u_{xx} + u_{yy} = 0$  '
                 r'on the unit square  (Gauss-Seidel)',
                 fontsize=13.5, fontweight='bold', y=1.02)
    save(fig, 'fig3_laplace_equation.png')


# ---------------------------------------------------------------------------
# fig4: separation of variables visualised
# ---------------------------------------------------------------------------
def fig4_separation_of_variables():
    print('Building fig4_separation_of_variables...')
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0],
                          width_ratios=[1.0, 1.0, 1.1],
                          hspace=0.55, wspace=0.32)

    L = 1.0
    x = np.linspace(0, L, 400)
    alpha = 0.05

    # --- row 1: spatial eigenmodes X_n
    ax = fig.add_subplot(gs[0, :2])
    for n_mode, color in zip([1, 2, 3, 4], [BLUE, PURPLE, GREEN, RED]):
        ax.plot(x, np.sin(n_mode * np.pi * x / L), color=color, lw=2.0,
                label=fr'$X_{n_mode}(x) = \sin({n_mode}\pi x/L)$')
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlim(0, L); ax.set_ylim(-1.2, 1.3)
    ax.set_xlabel('x', fontsize=11)
    ax.set_title('Spatial eigenmodes  $X_n$:  satisfy $X(0)=X(L)=0$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # --- row 1 right: temporal factors T_n (decay rates)
    ax = fig.add_subplot(gs[0, 2])
    t = np.linspace(0, 4, 200)
    for n_mode, color in zip([1, 2, 3, 4], [BLUE, PURPLE, GREEN, RED]):
        ax.plot(t, np.exp(-alpha * (n_mode * np.pi / L) ** 2 * t),
                color=color, lw=2.0, label=f'n = {n_mode}')
    ax.set_xlabel('t', fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.2)
    ax.set_title('Temporal factors  $T_n(t) = e^{-\\alpha(n\\pi/L)^2 t}$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # --- row 2: tensor product u_n = X_n T_n at three times
    times = [0.0, 0.5, 2.0]
    for col, t_v in enumerate(times):
        ax = fig.add_subplot(gs[1, col])
        for n_mode, color in zip([1, 2, 3, 4], [BLUE, PURPLE, GREEN, RED]):
            ax.plot(x, np.sin(n_mode * np.pi * x / L) *
                    np.exp(-alpha * (n_mode * np.pi / L) ** 2 * t_v),
                    color=color, lw=1.8, alpha=0.9)
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_xlim(0, L); ax.set_ylim(-1.2, 1.3)
        ax.set_title(f'$u_n(x,t) = X_n(x)\\,T_n(t)$  at t = {t_v:.1f}',
                     fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('mode amplitude', fontsize=10)

    # --- row 3: superposition recovering an initial bump
    ax = fig.add_subplot(gs[2, :])

    # initial condition: square pulse on [0.3, 0.7]
    f = np.where((x > 0.3) & (x < 0.7), 1.0, 0.0)
    # Fourier sine coefficients
    N_terms = 60
    bn = np.zeros(N_terms + 1)
    for n in range(1, N_terms + 1):
        bn[n] = 2 / L * np.trapz(f * np.sin(n * np.pi * x / L), x)

    for K, color, ls in zip([3, 8, 30], [BLUE, PURPLE, GREEN], ['-', '-', '-']):
        approx = np.zeros_like(x)
        for n in range(1, K + 1):
            approx += bn[n] * np.sin(n * np.pi * x / L)
        ax.plot(x, approx, color=color, lw=2.0, ls=ls, label=f'sum of {K} modes')
    ax.plot(x, f, color='black', lw=1.4, ls='--', label='exact initial f(x)')
    ax.set_xlim(0, L); ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('u(x, 0)', fontsize=11)
    ax.set_title('Superposition  $u(x,0) = \\sum b_n\\,X_n(x)$  '
                 '(more modes -> sharper recovery; Gibbs at jumps)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    fig.suptitle('Separation of variables for the heat equation:\n'
                 'eigenmodes -> tensor product -> Fourier superposition',
                 fontsize=13.5, fontweight='bold', y=1.00)
    save(fig, 'fig4_separation_of_variables.png')


# ---------------------------------------------------------------------------
# fig5: PDE classification (parabolic / hyperbolic / elliptic)
# ---------------------------------------------------------------------------
def fig5_pde_classification():
    print('Building fig5_pde_classification...')
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0],
                          hspace=0.45, wspace=0.32)

    # --- top-left: discriminant plane in (B^2, AC) -- show three regions
    ax = fig.add_subplot(gs[0, 0])
    AC = np.linspace(-2, 2, 400)
    B2 = np.linspace(0, 4, 400)
    AAC, BB2 = np.meshgrid(AC, B2)
    Delta = BB2 - AAC
    Z = np.sign(Delta).astype(float)
    Z[np.abs(Delta) < 0.05] = 0.0
    cmap = plt.cm.RdBu_r
    ax.contourf(AAC, BB2, Z, levels=[-1.5, -0.5, 0.5, 1.5],
                colors=['#cfe2ff', '#fff3cd', '#f5c2c7'])
    ax.plot(AC, AC, color='black', lw=2.0)
    ax.text(-1.4, 2.5, 'Hyperbolic\n$B^2 - AC > 0$\n(waves)',
            fontsize=11, fontweight='bold', color=RED, ha='left')
    ax.text(0.6, 0.55, 'Parabolic\n$B^2 - AC = 0$\n(diffusion)',
            fontsize=10, fontweight='bold', color='#b08900', ha='left',
            rotation=42)
    ax.text(1.05, 0.45, 'Elliptic\n$B^2 - AC < 0$\n(equilibrium)',
            fontsize=11, fontweight='bold', color=BLUE, ha='left')
    ax.set_xlabel('AC', fontsize=11); ax.set_ylabel(r'$B^2$', fontsize=11)
    ax.set_title('Discriminant $\\Delta = B^2 - AC$  determines type',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-2, 2); ax.set_ylim(0, 4)

    # --- top-middle: parabolic sample (heat equation cone of influence)
    ax = fig.add_subplot(gs[0, 1])
    L = 4.0; T = 3.0
    Nx = 200; Nt = 200
    x = np.linspace(-L, L, Nx)
    t = np.linspace(0.001, T, Nt)
    XX, TT = np.meshgrid(x, t)
    # delta-source at (0,0): fundamental solution
    G = 1 / np.sqrt(4 * np.pi * 0.4 * TT) * np.exp(-XX**2 / (4 * 0.4 * TT))
    G = np.clip(G, 0, 3)
    im = ax.imshow(G, aspect='auto', origin='lower', cmap='inferno',
                   extent=[-L, L, 0, T])
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('t', fontsize=11)
    ax.set_title('Parabolic:  heat from a $\\delta$-source\n'
                 '(infinite propagation speed, Gaussian spread)',
                 fontsize=11.5, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # --- top-right: hyperbolic sample (light cone)
    ax = fig.add_subplot(gs[0, 2])
    XX, TT = np.meshgrid(x, t)
    c = 1.0
    # d'Alembert with bump
    f = lambda z: np.exp(-z**2 * 6.0)
    U = 0.5 * (f(XX - c * TT) + f(XX + c * TT))
    im = ax.imshow(U, aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[-L, L, 0, T], vmin=-1, vmax=1)
    ax.plot([0, c * T], [0, T], color='black', lw=1.2, ls='--')
    ax.plot([0, -c * T], [0, T], color='black', lw=1.2, ls='--')
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('t', fontsize=11)
    ax.set_title('Hyperbolic:  pulse from $(0,0)$\n'
                 r'(finite speed; light cone $|x|\leq ct$)',
                 fontsize=11.5, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # --- bottom row: 3 canonical PDEs side by side as cards
    info = [
        ('Heat equation', r'$u_t = \alpha\,u_{xx}$',
         ['parabolic', 'irreversible', 'smooths sharp data',
          'IVP + 2 BCs', 'fundamental solution = Gaussian',
          r'modes decay as $e^{-\alpha (n\pi/L)^2 t}$'],
         '#fff3cd', '#b08900'),
        ('Wave equation', r'$u_{tt} = c^2\,u_{xx}$',
         ['hyperbolic', 'reversible in time', 'preserves singularities',
          'IVP needs $u, u_t$ + 2 BCs',
          r"d'Alembert: $u = \frac{1}{2}[f(x-ct) + f(x+ct)]$",
          r'CFL  $c\,\Delta t / \Delta x \leq 1$'],
         '#f5c2c7', RED),
        ('Laplace equation', r'$u_{xx} + u_{yy} = 0$',
         ['elliptic', 'no time evolution', 'maximum principle',
          'BVP only (no IC)', 'mean-value property',
          'extrema only on boundary'],
         '#cfe2ff', BLUE),
    ]
    for col, (title, eq, bullets, bg, accent) in enumerate(info):
        ax = fig.add_subplot(gs[1, col])
        ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0.02, 0.04), 0.96, 0.94,
                                   facecolor=bg, edgecolor=accent,
                                   linewidth=2.0))
        ax.text(0.5, 0.88, title, fontsize=14, fontweight='bold',
                color=accent, ha='center')
        ax.text(0.5, 0.74, eq, fontsize=14, ha='center')
        for i, b in enumerate(bullets):
            ax.text(0.07, 0.62 - i * 0.085, '- ' + b, fontsize=10,
                    color='black')

    fig.suptitle('PDE classification:  parabolic   /   hyperbolic   /   elliptic',
                 fontsize=14, fontweight='bold', y=1.00)
    save(fig, 'fig5_pde_classification.png')


if __name__ == '__main__':
    fig1_heat_equation_evolution()
    fig2_wave_propagation()
    fig3_laplace_equation()
    fig4_separation_of_variables()
    fig5_pde_classification()
    print('\nChapter 13 figures complete.')
