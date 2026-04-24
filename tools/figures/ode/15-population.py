"""
Chapter 15: Population Dynamics.

Figures (5):
  fig1_lotka_volterra.png        -- predator-prey: time series + phase portrait + conserved quantity
  fig2_competition_model.png     -- 2-species competition: 4 outcomes (coexist / exclude X 2 / bistable)
  fig3_allee_effect.png          -- Allee growth-rate curves + bistability / extinction-threshold flow
  fig4_age_structured_leslie.png -- Leslie matrix: spectrum, age distribution, growth trajectory
  fig5_metapopulation.png        -- Levins metapopulation + spatial Fisher-KPP traveling wave

Standalone: writes the same five PNGs into both EN and ZH asset folders.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/15-population-dynamics',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/15-种群动力学',
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
# fig1: Lotka-Volterra predator-prey
# ---------------------------------------------------------------------------
def fig1_lotka_volterra():
    print('Building fig1_lotka_volterra...')
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)

    a, b, d, g = 1.0, 0.1, 0.05, 0.5
    # equilibrium
    x_eq = g / d; y_eq = a / b

    def lv(t, s):
        x, y = s
        return [a * x - b * x * y, d * x * y - g * y]

    t = np.linspace(0, 60, 4000)
    sol = solve_ivp(lv, (0, 60), [10.0, 2.0], t_eval=t, rtol=1e-9)

    # (a) time series with hare-lynx feel
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sol.t, sol.y[0], color=GREEN, lw=2.2, label='prey  x')
    ax.plot(sol.t, sol.y[1], color=RED,   lw=2.2, label='predator  y')
    ax.axhline(x_eq, color=GREEN, lw=0.8, ls=':', alpha=0.6)
    ax.axhline(y_eq, color=RED,   lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('time', fontsize=11); ax.set_ylabel('population', fontsize=11)
    ax.set_title('Predator-prey oscillations\n(Lotka-Volterra; equilibrium dotted)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    # (b) phase portrait with multiple closed orbits + nullclines
    ax = fig.add_subplot(gs[0, 1])
    starts = [(2, 1), (5, 1), (10, 2), (20, 2), (30, 2)]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(starts)))
    for s0, color in zip(starts, cmap):
        s = solve_ivp(lv, (0, 60), list(s0), t_eval=np.linspace(0, 60, 4000),
                      rtol=1e-9)
        ax.plot(s.y[0], s.y[1], color=color, lw=1.6, alpha=0.9)
    # nullclines
    ax.axvline(x_eq, color='black', lw=1.0, ls='--', alpha=0.6,
               label=f'$y$-nullcline  $x = {x_eq:.0f}$')
    ax.axhline(y_eq, color='black', lw=1.0, ls=':', alpha=0.6,
               label=f'$x$-nullcline  $y = {y_eq:.0f}$')
    ax.scatter([x_eq], [y_eq], color='black', s=80, zorder=6,
               edgecolor='white', linewidth=1.2,
               label='center  $(\\gamma/\\delta,\\, \\alpha/\\beta)$')
    # arrows: vector field
    Xg, Yg = np.meshgrid(np.linspace(0.5, 35, 18), np.linspace(0.5, 18, 14))
    U = a * Xg - b * Xg * Yg
    V = d * Xg * Yg - g * Yg
    M = np.sqrt(U**2 + V**2); M[M == 0] = 1
    ax.quiver(Xg, Yg, U / M, V / M, color='gray', alpha=0.45, scale=35)
    ax.set_xlabel('prey  x', fontsize=11); ax.set_ylabel('predator  y', fontsize=11)
    ax.set_title('Phase portrait:  family of closed orbits  (neutrally stable)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 35); ax.set_ylim(0, 18)

    # (c) conserved quantity H along orbits
    ax = fig.add_subplot(gs[1, 0])
    def H(x, y):
        return d * x - g * np.log(x) + b * y - a * np.log(y)
    Hvals = H(sol.y[0], sol.y[1])
    ax.plot(sol.t, Hvals, color=PURPLE, lw=2.0)
    ax.set_xlabel('time', fontsize=11)
    ax.set_ylabel(r'$H = \delta x - \gamma\ln x + \beta y - \alpha\ln y$',
                  fontsize=11)
    ax.set_title('Conserved quantity $H$  (numerical drift only)',
                 fontsize=12, fontweight='bold')

    # (d) Holling Type II + paradox of enrichment hint
    ax = fig.add_subplot(gs[1, 1])
    K_vals = [80, 200, 500]
    for K, color in zip(K_vals, [BLUE, PURPLE, RED]):
        a2, b2 = 1.0, 0.1; h = 0.5; e = 0.05; m = 0.4
        def lv2(t, s, K=K):
            x, y = s
            holling = a2 * x / (1 + a2 * h * x)
            return [b2 * x * (1 - x / K) - holling * y,
                    e * holling * y - m * y]
        s = solve_ivp(lv2, (0, 250), [20, 5],
                      t_eval=np.linspace(0, 250, 5000), rtol=1e-9)
        ax.plot(s.y[0], s.y[1], color=color, lw=1.5, alpha=0.85,
                label=f'K = {K}')
    ax.set_xlabel('prey  x', fontsize=11); ax.set_ylabel('predator  y', fontsize=11)
    ax.set_title('Paradox of enrichment:  larger K  ->  larger limit cycle',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Lotka-Volterra predator-prey dynamics',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig1_lotka_volterra.png')


# ---------------------------------------------------------------------------
# fig2: 2-species competition -- 4 canonical outcomes
# ---------------------------------------------------------------------------
def fig2_competition_model():
    print('Building fig2_competition_model...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    K1, K2, r1, r2 = 100, 100, 0.5, 0.5
    cases = [
        ('Stable coexistence',
         {'a12': 0.5, 'a21': 0.5}, axes[0, 0]),
        ('Species 1 wins',
         {'a12': 0.5, 'a21': 1.5}, axes[0, 1]),
        ('Species 2 wins',
         {'a12': 1.5, 'a21': 0.5}, axes[1, 0]),
        ('Bistable  (founder effect)',
         {'a12': 1.5, 'a21': 1.5}, axes[1, 1]),
    ]

    for title, p, ax in cases:
        a12 = p['a12']; a21 = p['a21']

        def comp(t, s):
            N1, N2 = s
            return [r1 * N1 * (1 - (N1 + a12 * N2) / K1),
                    r2 * N2 * (1 - (N2 + a21 * N1) / K2)]

        # nullclines
        N = np.linspace(0, 150, 300)
        # dN1/dt = 0  <=>  N2 = (K1 - N1) / a12
        n1_null = (K1 - N) / a12
        # dN2/dt = 0  <=>  N2 = K2 - a21 N1
        n2_null = K2 - a21 * N
        ax.plot(N, n1_null, color=BLUE, lw=2.2, label='$dN_1/dt = 0$')
        ax.plot(N, n2_null, color=RED,  lw=2.2, label='$dN_2/dt = 0$')

        # vector field
        Xg, Yg = np.meshgrid(np.linspace(2, 140, 16), np.linspace(2, 140, 16))
        U = r1 * Xg * (1 - (Xg + a12 * Yg) / K1)
        V = r2 * Yg * (1 - (Yg + a21 * Xg) / K2)
        M = np.sqrt(U**2 + V**2); M[M == 0] = 1
        ax.quiver(Xg, Yg, U / M, V / M, color='gray', alpha=0.45, scale=35)

        # multiple trajectories
        starts = [(20, 100), (100, 20), (60, 60), (10, 10), (130, 130)]
        cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(starts)))
        for s0, color in zip(starts, cmap):
            s = solve_ivp(comp, (0, 200), list(s0),
                          t_eval=np.linspace(0, 200, 1500), rtol=1e-9)
            ax.plot(s.y[0], s.y[1], color=color, lw=1.6, alpha=0.85)
            ax.scatter([s0[0]], [s0[1]], color=color, s=18,
                       edgecolor='white', linewidth=0.8, zorder=5)

        # mark equilibria
        for px, py in [(K1, 0), (0, K2)]:
            ax.scatter([px], [py], color='black', s=60, zorder=6,
                       edgecolor='white', linewidth=1.0)
        # interior equilibrium (if exists)
        denom = 1 - a12 * a21
        if abs(denom) > 1e-3:
            n1s = (K1 - a12 * K2) / denom
            n2s = (K2 - a21 * K1) / denom
            if 0 < n1s < 150 and 0 < n2s < 150:
                ax.scatter([n1s], [n2s], color='black', s=120, marker='*',
                           zorder=7, edgecolor='white', linewidth=1.0)

        ax.set_xlim(0, 150); ax.set_ylim(0, 150)
        ax.set_xlabel('$N_1$', fontsize=11); ax.set_ylabel('$N_2$', fontsize=11)
        ax.set_title(f'{title}\n$\\alpha_{{12}} = {a12},\\ \\alpha_{{21}} = {a21}$',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Lotka-Volterra competition:  four outcomes\n'
                 r'(coexistence requires $\alpha_{12} < K_1/K_2$ AND '
                 r'$\alpha_{21} < K_2/K_1$)',
                 fontsize=13.5, fontweight='bold', y=1.00)
    fig.tight_layout()
    save(fig, 'fig2_competition_model.png')


# ---------------------------------------------------------------------------
# fig3: Allee effect
# ---------------------------------------------------------------------------
def fig3_allee_effect():
    print('Building fig3_allee_effect...')
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.28)

    r = 0.5; K = 100; A = 25  # Allee threshold

    N = np.linspace(0, 130, 400)
    # logistic
    f_log = r * N * (1 - N / K)
    # Allee (strong): dN/dt = r N (1 - N/K)(N/A - 1)
    f_allee = r * N * (1 - N / K) * (N / A - 1)
    # weak Allee: dN/dt = r N (1 - N/K)(N/A) ; growth rate dampened at low N but no extinction
    f_weak = r * N * (1 - N / K) * (N / (N + A))

    # (a) per-capita growth rate vs N
    ax = fig.add_subplot(gs[0, 0])
    pc_log = r * (1 - N / K)
    pc_allee = r * (1 - N / K) * (N / A - 1)
    pc_weak  = r * (1 - N / K) * (N / (N + A))
    ax.plot(N, pc_log, color=BLUE, lw=2.2, label='logistic')
    ax.plot(N, pc_weak, color=GREEN, lw=2.2, label='weak Allee')
    ax.plot(N, pc_allee, color=RED, lw=2.2, label='strong Allee')
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(A, color='gray', lw=1.0, ls='--')
    ax.text(A + 2, -0.15, f'A = {A}\nthreshold', fontsize=9, color='gray')
    ax.set_xlabel('N', fontsize=11)
    ax.set_ylabel('per-capita growth  $\\dot N / N$', fontsize=11)
    ax.set_title('Per-capita growth rate vs density',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(-0.6, 0.6)

    # (b) dN/dt vs N
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(N, f_log,   color=BLUE,  lw=2.2, label='logistic')
    ax.plot(N, f_weak,  color=GREEN, lw=2.2, label='weak Allee')
    ax.plot(N, f_allee, color=RED,   lw=2.2, label='strong Allee')
    ax.axhline(0, color='black', lw=0.8)
    # mark equilibria
    for x_eq, color in [(0, RED), (A, RED), (K, RED)]:
        ax.scatter([x_eq], [0], s=60, color=color, edgecolor='white',
                   linewidth=1.0, zorder=5)
    ax.set_xlabel('N', fontsize=11)
    ax.set_ylabel('dN/dt', fontsize=11)
    ax.set_title('Population growth rate (strong Allee has 3 equilibria)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    # (c) trajectories under strong Allee from various N0
    ax = fig.add_subplot(gs[1, 0])
    def allee(t, y, r=r, K=K, A=A):
        N = y[0]
        return [r * N * (1 - N / K) * (N / A - 1)]
    t = np.linspace(0, 100, 1000)
    starts = [5, 15, 22, 28, 40, 70, 110]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(starts)))
    for N0, color in zip(starts, cmap):
        s = solve_ivp(allee, (0, 100), [N0], t_eval=t, rtol=1e-9)
        ax.plot(s.t, s.y[0], color=color, lw=2.0, label=f'$N_0 = {N0}$')
    ax.axhline(K, color='gray', lw=1.0, ls=':')
    ax.axhline(A, color=RED,   lw=1.0, ls='--')
    ax.text(85, K + 3, f'K = {K}',  color='gray', fontsize=9)
    ax.text(85, A + 3, f'A = {A}',  color=RED,   fontsize=9)
    ax.set_xlabel('time', fontsize=11); ax.set_ylabel('N(t)', fontsize=11)
    ax.set_title(f'Strong Allee:  N below A goes extinct, N above A -> K',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='center right', fontsize=8, ncol=2)
    ax.set_ylim(0, 130)

    # (d) phase line / potential V(N) such that dN/dt = -V'(N)
    ax = fig.add_subplot(gs[1, 1])
    Ngrid = np.linspace(0, 130, 400)
    f = r * Ngrid * (1 - Ngrid / K) * (Ngrid / A - 1)
    # numerical antiderivative with sign flip
    V = -np.cumsum(f) * (Ngrid[1] - Ngrid[0])
    ax.plot(Ngrid, V, color=PURPLE, lw=2.4)
    # mark equilibria
    for x_eq, lab in [(0, 'extinct (stable)'),
                      (A, 'unstable threshold'),
                      (K, 'carrying capacity (stable)')]:
        idx = np.argmin(abs(Ngrid - x_eq))
        ax.scatter([x_eq], [V[idx]], s=80, color=RED, edgecolor='white',
                   linewidth=1.0, zorder=5)
        ax.annotate(lab, xy=(x_eq, V[idx]),
                    xytext=(x_eq + 5, V[idx] + 30),
                    fontsize=9, color='black',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax.set_xlabel('N', fontsize=11)
    ax.set_ylabel('potential  V(N)  with  dN/dt = $-V\'(N)$', fontsize=11)
    ax.set_title('Energy landscape:  two basins separated by a barrier',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(V.min() - 50, V.max() + 100)

    fig.suptitle(r'Allee effect:  small populations have negative growth',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig3_allee_effect.png')


# ---------------------------------------------------------------------------
# fig4: Age-structured Leslie matrix
# ---------------------------------------------------------------------------
def fig4_age_structured_leslie():
    print('Building fig4_age_structured_leslie...')
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.32)

    # 5 age classes: juveniles, sub-adults, adults1, adults2, seniors
    F = np.array([0.0, 0.5, 2.5, 1.5, 0.2])           # fertility per class
    P = np.array([0.6, 0.7, 0.85, 0.5])               # survival to next class
    L = np.zeros((5, 5))
    L[0, :] = F
    for i in range(4):
        L[i + 1, i] = P[i]

    eigvals, eigvecs = np.linalg.eig(L)
    idx = np.argmax(np.abs(eigvals))
    lam = float(np.real(eigvals[idx]))
    stable = np.real(eigvecs[:, idx])
    stable = np.abs(stable / stable.sum())

    # (a) Leslie matrix as heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(L, cmap='viridis')
    for i in range(5):
        for j in range(5):
            txt = f'{L[i, j]:.2f}'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if L[i, j] < 0.6 else 'black',
                    fontsize=10)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f'age {i}' for i in range(5)])
    ax.set_yticklabels([f'age {i}' for i in range(5)])
    ax.set_title(f'Leslie matrix L  (top row = fertility, '
                 f'subdiagonal = survival)\n'
                 f'dominant eigenvalue  $\\lambda = {lam:.3f}$  '
                 f'({"growth" if lam > 1 else "decline"})',
                 fontsize=11.5, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # (b) eigenvalue spectrum on complex plane
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(np.real(eigvals), np.imag(eigvals), s=80, color=BLUE,
               edgecolor='white', linewidth=1.2, zorder=5)
    ax.scatter([np.real(eigvals[idx])], [np.imag(eigvals[idx])], s=180,
               color=RED, marker='*', zorder=6, edgecolor='white', linewidth=1.2)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', lw=0.8, ls='--')
    ax.axhline(0, color='black', lw=0.6); ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('Re', fontsize=11); ax.set_ylabel('Im', fontsize=11)
    ax.set_title('Eigenvalues  (red star = dominant -> long-run growth rate)',
                 fontsize=11.5, fontweight='bold')
    ax.set_aspect('equal')

    # (c) trajectory: total population growing geometrically
    n0 = np.array([1000, 0, 0, 0, 0])
    T = 30
    n = np.zeros((T + 1, 5)); n[0] = n0
    for t in range(T):
        n[t + 1] = L @ n[t]
    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, 5))
    for i, color in enumerate(cmap):
        ax.plot(range(T + 1), n[:, i], color=color, lw=2.0,
                label=f'age {i}')
    ax.plot(range(T + 1), n.sum(axis=1), color='black', lw=2.2, ls='--',
            label='total')
    # geometric-growth fit: total ~ N0 * lam^t for late t
    fit_t = np.arange(15, 31)
    ax.plot(fit_t, n[fit_t].sum(axis=1)[0] * lam ** (fit_t - fit_t[0]),
            color=RED, lw=1.4, ls=':', label=fr'$\propto \lambda^t$')
    ax.set_yscale('log')
    ax.set_xlabel('year  t', fontsize=11)
    ax.set_ylabel('count (log)', fontsize=11)
    ax.set_title('Trajectory:  geometric growth at rate  $\\lambda$',
                 fontsize=11.5, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, ncol=2)

    # (d) stable age distribution (eigenvector) vs initial
    ax = fig.add_subplot(gs[1, 1])
    width = 0.38
    ages = np.arange(5)
    init = n0 / n0.sum()
    final = n[-1] / n[-1].sum()
    ax.bar(ages - width / 2, init,   width, color=BLUE,
           label='initial', edgecolor='white')
    ax.bar(ages + width / 2, stable, width, color=RED,
           label='stable (eigenvector)', edgecolor='white')
    ax.scatter(ages, final, color='black', s=80, zorder=5,
               label='at t = 30')
    ax.set_xticks(ages); ax.set_xticklabels([f'age {i}' for i in range(5)])
    ax.set_ylabel('fraction of population', fontsize=11)
    ax.set_title('Stable age distribution  (Perron eigenvector)',
                 fontsize=11.5, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'Age-structured population:  Leslie matrix model',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig4_age_structured_leslie.png')


# ---------------------------------------------------------------------------
# fig5: Metapopulation -- Levins model + Fisher-KPP traveling wave
# ---------------------------------------------------------------------------
def fig5_metapopulation():
    print('Building fig5_metapopulation...')
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    # ---- (a) Levins model dp/dt = c p (1 - p) - e p
    ax = fig.add_subplot(gs[0, 0])
    es = [0.05, 0.1, 0.2, 0.4]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(es)))
    c = 0.3
    t = np.linspace(0, 80, 800)
    for e, color in zip(es, cmap):
        def lev(t, y, e=e):
            p = y[0]
            return [c * p * (1 - p) - e * p]
        s = solve_ivp(lev, (0, 80), [0.05], t_eval=t, rtol=1e-9)
        p_eq = max(0, 1 - e / c)
        ax.plot(s.t, s.y[0], color=color, lw=2.0,
                label=f'e = {e}, $p^* = {p_eq:.2f}$')
        ax.axhline(p_eq, color=color, lw=0.6, ls=':')
    ax.set_xlabel('time', fontsize=11)
    ax.set_ylabel('occupied fraction  p', fontsize=11)
    ax.set_title(f'Levins metapopulation  (c = {c})\n'
                 r'persistence requires $c > e$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='center right', fontsize=9)
    ax.set_ylim(0, 1.0)

    # ---- (b) p_eq vs e/c (extinction threshold)
    ax = fig.add_subplot(gs[0, 1])
    ratios = np.linspace(0, 2, 300)
    p_eq = np.maximum(0, 1 - ratios)
    ax.plot(ratios, p_eq, color=BLUE, lw=2.4)
    ax.fill_between(ratios, 0, p_eq, color=BLUE, alpha=0.15,
                    label='persistence')
    ax.axvline(1.0, color=RED, lw=1.4, ls='--', label='extinction threshold')
    ax.fill_between([1.0, 2.0], 0, 1, color=RED, alpha=0.10,
                    label='regional extinction')
    ax.set_xlabel('ratio  e / c', fontsize=11)
    ax.set_ylabel('equilibrium occupied fraction  $p^*$', fontsize=11)
    ax.set_title('Extinction threshold of the Levins model',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 2); ax.set_ylim(0, 1.05)

    # ---- (c) Fisher-KPP traveling wave: snapshots
    ax = fig.add_subplot(gs[1, 0])
    L = 200; Nx = 800
    x = np.linspace(0, L, Nx); dx = x[1] - x[0]
    D = 1.0; r = 0.5; K = 1.0
    # initial: localized
    u = np.where(x < 5, 1.0, 0.0)
    dt = 0.4 * dx**2 / (2 * D)
    Nt = int(80 / dt)
    snaps_t = [0, 20, 40, 60, 80]
    snaps = {0: u.copy()}
    for n in range(1, Nt + 1):
        lap = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        u = u + dt * (D * lap + r * u * (1 - u / K))
        # zero-flux boundaries (mirror)
        u[0] = u[1]; u[-1] = u[-2]
        for tt in snaps_t:
            if tt > 0 and abs(n * dt - tt) < dt / 2:
                snaps[tt] = u.copy()
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(snaps_t)))
    for tt, color in zip(snaps_t, cmap):
        if tt in snaps:
            ax.plot(x, snaps[tt], color=color, lw=2.0, label=f't = {tt}')
    c_min = 2 * np.sqrt(D * r)
    ax.set_xlabel('position  x', fontsize=11)
    ax.set_ylabel('density  N(x, t)', fontsize=11)
    ax.set_title(f'Fisher-KPP wave  (D = {D},\\ r = {r})\n'
                 f'asymptotic speed  $c_{{\\min}} = 2\\sqrt{{Dr}} = {c_min:.2f}$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, L); ax.set_ylim(-0.05, 1.15)

    # ---- (d) front position vs t (linear -> measured speed)
    ax = fig.add_subplot(gs[1, 1])
    # rerun and track 0.5 contour position
    u = np.where(x < 5, 1.0, 0.0)
    fronts = []; ts = []
    for n in range(1, Nt + 1):
        lap = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        u = u + dt * (D * lap + r * u * (1 - u / K))
        u[0] = u[1]; u[-1] = u[-2]
        if n % 50 == 0:
            mask = u > 0.5
            if mask.any():
                fronts.append(x[mask][-1])
                ts.append(n * dt)
    fronts = np.array(fronts); ts = np.array(ts)
    ax.plot(ts, fronts, color=PURPLE, lw=2.2, label='front position (level 0.5)')
    # fit
    fit = np.polyfit(ts[len(ts)//3:], fronts[len(ts)//3:], 1)
    ax.plot(ts, np.polyval(fit, ts), color=RED, lw=1.5, ls='--',
            label=fr'fit  $c \approx {fit[0]:.2f}$')
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel('time  t', fontsize=11)
    ax.set_ylabel('front position', fontsize=11)
    ax.set_title(f'Linear front motion -> measured speed matches  '
                 f'$2\\sqrt{{Dr}} = {c_min:.2f}$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Spatial population dynamics:  metapopulation + invasion fronts',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig5_metapopulation.png')


if __name__ == '__main__':
    fig1_lotka_volterra()
    fig2_competition_model()
    fig3_allee_effect()
    fig4_age_structured_leslie()
    fig5_metapopulation()
    print('\nChapter 15 figures complete.')
