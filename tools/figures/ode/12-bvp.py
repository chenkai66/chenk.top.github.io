"""
Chapter 12: Boundary Value Problems -- shooting, finite difference, eigenvalues, Sturm-Liouville.

Figures (saved to BOTH the EN and ZH asset folders):
  fig1_shooting_method.png        -- iterative shots converging to the right boundary
  fig2_finite_difference.png      -- discretization grid + tridiagonal matrix sparsity
  fig3_eigenvalue_problem.png     -- first 5 eigenmodes of -y'' = lambda y on [0, pi]
  fig4_sturm_liouville.png        -- Schrodinger / quantum harmonic oscillator eigenfunctions
  fig5_bvp_solution_methods.png   -- four BVP solution strategies on the same problem
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.optimize import brentq

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
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/12-boundary-value-problems',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/12-边值问题',
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
# fig1: Shooting method -- iterative trajectories converging
# ---------------------------------------------------------------------------
def fig1_shooting_method():
    print('Building fig1_shooting_method...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6),
                             gridspec_kw={'width_ratios': [1.4, 1.0]})

    # Linear BVP: y'' + y = 0, y(0) = 0, y(pi/2) = 1  -> y = sin(x)
    a, b = 0.0, np.pi / 2
    alpha, beta = 0.0, 1.0
    def rhs(x, Y, _s=None):
        return [Y[1], -Y[0]]

    # A bunch of trial slopes
    slopes = [0.3, 0.6, 1.4, 2.0, 1.0]   # last one hits target
    colors = [RED, RED, RED, RED, GREEN]
    labels = ['guess $s=0.3$', 'guess $s=0.6$', 'guess $s=1.4$',
              'guess $s=2.0$', 'converged $s=1.0$']

    ax = axes[0]
    x_dense = np.linspace(a, b, 300)
    for s, c, lbl in zip(slopes, colors, labels):
        sol = solve_ivp(rhs, [a, b], [alpha, s], dense_output=True,
                        rtol=1e-8, atol=1e-10)
        y = sol.sol(x_dense)[0]
        lw = 2.4 if c == GREEN else 1.0
        alpha_v = 1.0 if c == GREEN else 0.55
        ax.plot(x_dense, y, color=c, lw=lw, alpha=alpha_v, label=lbl)
        ax.plot(b, sol.sol(b)[0], 'o', color=c, ms=7, mec='white', mew=0.8)

    ax.axhline(beta, color='black', lw=1.0, ls='--', alpha=0.7)
    ax.scatter([b], [beta], color='black', s=130, marker='*',
               zorder=10, label=f'target  $y(\\pi/2)={beta}$')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y(x)$')
    ax.set_xlim(a, b * 1.05)
    ax.set_title("Shooting at $y'' + y = 0$,  $y(0)=0$, $y(\\pi/2)=1$",
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8.5)

    # Right: residual F(s) = y(b; s) - beta
    ax = axes[1]
    s_grid = np.linspace(-0.5, 2.5, 60)
    F = []
    for s in s_grid:
        sol = solve_ivp(rhs, [a, b], [alpha, s], rtol=1e-8, atol=1e-10)
        F.append(sol.y[0, -1] - beta)
    F = np.array(F)
    ax.plot(s_grid, F, color=BLUE, lw=2.0)
    ax.axhline(0, color='black', lw=0.8)
    ax.scatter([1.0], [0.0], color=GREEN, s=130, zorder=10,
               edgecolor='white', linewidth=1.5, label='root  $s^*=1$')
    for s in [0.3, 0.6, 1.4, 2.0]:
        sol = solve_ivp(rhs, [a, b], [alpha, s], rtol=1e-8, atol=1e-10)
        ax.scatter([s], [sol.y[0, -1] - beta], color=RED, s=45, zorder=8,
                   edgecolor='white', linewidth=0.8)
    ax.set_xlabel('initial slope guess $s$', fontsize=11)
    ax.set_ylabel(r'residual $F(s) = y(\pi/2; s) - 1$', fontsize=11)
    ax.set_title('Shooting reduces BVP to root-finding $F(s)=0$',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Shooting method: aim, miss, correct, repeat',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig1_shooting_method.png')


# ---------------------------------------------------------------------------
# fig2: Finite difference -- grid + tridiagonal matrix
# ---------------------------------------------------------------------------
def fig2_finite_difference():
    print('Building fig2_finite_difference...')
    fig = plt.figure(figsize=(14, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25)

    # Solve y'' = -pi^2 sin(pi x), y(0) = y(1) = 0  -> exact y = sin(pi x)
    N = 40
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    n = N - 1
    main = -2 * np.ones(n) / h**2
    off  = np.ones(n - 1) / h**2
    rhs = -np.pi**2 * np.sin(np.pi * x[1:-1])
    # Solve tridiagonal
    from scipy.linalg import solve_banded
    ab = np.zeros((3, n))
    ab[0, 1:] = off
    ab[1, :]  = main
    ab[2, :-1] = off
    y_int = solve_banded((1, 1), ab, rhs)
    y = np.concatenate([[0], y_int, [0]])

    # Left: solution + grid + stencil
    ax = fig.add_subplot(gs[0, 0])
    xx = np.linspace(0, 1, 400)
    ax.plot(xx, np.sin(np.pi * xx), color='black', lw=2.4,
            label='exact  $\\sin(\\pi x)$')
    ax.plot(x, y, 'o', color=BLUE, ms=7, mec='white', mew=0.7,
            label=f'finite difference  ($N={N}$)')
    # Highlight a stencil
    i0 = 12
    for j in [-1, 0, 1]:
        ax.scatter(x[i0 + j], y[i0 + j], s=180, facecolor='none',
                   edgecolor=RED, linewidth=2.0, zorder=8)
    ax.annotate('3-point stencil\n$y_{i-1}, y_i, y_{i+1}$',
                xy=(x[i0], y[i0]), xytext=(0.35, 0.25),
                fontsize=10, color=RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED))
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y(x)$')
    ax.set_title("Discretizing $y''=f$ on a uniform grid",
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower center', fontsize=10)

    # Right: tridiagonal sparsity pattern
    ax = fig.add_subplot(gs[0, 1])
    n_show = 12
    A = np.zeros((n_show, n_show))
    np.fill_diagonal(A, -2)
    np.fill_diagonal(A[1:], 1); np.fill_diagonal(A[:, 1:], 1)
    cmap = plt.cm.RdBu
    im = ax.imshow(A, cmap=cmap, vmin=-2.5, vmax=2.5)
    for i in range(n_show):
        for j in range(n_show):
            if A[i, j] != 0:
                ax.text(j, i, f'{int(A[i,j]):+d}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if abs(A[i,j]) > 1.5 else 'black')
    ax.set_xticks(range(n_show)); ax.set_yticks(range(n_show))
    ax.set_xticklabels([f'$y_{{{i+1}}}$' for i in range(n_show)],
                       fontsize=8, rotation=45)
    ax.set_yticklabels([f'eq $i={i+1}$' for i in range(n_show)],
                       fontsize=8)
    ax.set_title("Tridiagonal matrix  $h^2 A$  (only 3 non-zeros per row)\n"
                 "solvable in $\\mathcal{O}(N)$ time",
                 fontsize=11, fontweight='bold')
    ax.set_aspect('equal')

    fig.suptitle('Finite difference: continuous BVP $\\to$ sparse linear system',
                 fontsize=13, fontweight='bold', y=1.02)
    save(fig, 'fig2_finite_difference.png')


# ---------------------------------------------------------------------------
# fig3: Eigenvalue problem -- first 5 modes of -y'' = lambda y on [0, pi]
# ---------------------------------------------------------------------------
def fig3_eigenvalue_problem():
    print('Building fig3_eigenvalue_problem...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8),
                             gridspec_kw={'width_ratios': [1.3, 1.0]})

    # Discretize -y'' = lambda y, y(0)=y(pi)=0
    N = 400
    h = np.pi / N
    n = N - 1
    main = (2 / h**2) * np.ones(n)
    off  = (-1 / h**2) * np.ones(n - 1)
    eigvals_num, eigvecs = eigh_tridiagonal(main, off, select='i',
                                            select_range=(0, 4))
    x_int = np.linspace(h, np.pi - h, n)
    x = np.concatenate([[0], x_int, [np.pi]])

    ax = axes[0]
    palette = [BLUE, PURPLE, GREEN, RED, COLORS["warning"]]
    x_exact = np.linspace(0, np.pi, 300)
    for k in range(5):
        # Normalize: pin sign so first peak is positive
        v = eigvecs[:, k]
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        v_full = np.concatenate([[0], v, [0]])
        # scale to unit max for nice plotting
        v_full = v_full / np.max(np.abs(v_full))
        ax.plot(x, v_full + 2.5 * k, color=palette[k], lw=2.0,
                label=fr'$\lambda_{k+1}={eigvals_num[k]:.3f}$ '
                      fr'(exact $={(k+1)**2}$)')
        # exact sin((k+1) x), normalized
        y_ex = np.sin((k + 1) * x_exact)
        ax.plot(x_exact, y_ex + 2.5 * k, color='black', lw=0.7,
                ls='--', alpha=0.6)
        ax.axhline(2.5 * k, color='gray', lw=0.4)
    ax.set_xlabel('$x$'); ax.set_ylabel('eigenmode (offset for clarity)')
    ax.set_title("First five eigenmodes of $-y'' = \\lambda y$\n"
                 "$y(0)=y(\\pi)=0$  ($n={N}$ grid)",
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8.5)

    # Right: eigenvalue spectrum vs n^2
    ax = axes[1]
    eigvals_all, _ = eigh_tridiagonal(main, off, select='i',
                                      select_range=(0, 19))
    n_idx = np.arange(1, 21)
    ax.plot(n_idx, n_idx**2, color='black', lw=1.6, ls='--',
            label='exact  $\\lambda_n = n^2$')
    ax.plot(n_idx, eigvals_all, 'o', color=BLUE, ms=7, mec='white', mew=0.7,
            label=f'numerical  ($N={N}$)')
    ax.set_xlabel('mode index $n$'); ax.set_ylabel('$\\lambda_n$')
    ax.set_title('Eigenvalue spectrum: agrees with $n^2$ for low modes,\n'
                 'systematic error grows for higher modes',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    fig.tight_layout()
    save(fig, 'fig3_eigenvalue_problem.png')


# ---------------------------------------------------------------------------
# fig4: Sturm-Liouville -- quantum harmonic oscillator
# ---------------------------------------------------------------------------
def fig4_sturm_liouville():
    print('Building fig4_sturm_liouville...')
    fig, ax = plt.subplots(figsize=(11, 7.0))

    # -psi'' + x^2 psi = E psi  (units where hbar = m = omega = 1)
    # Eigenvalues E_n = 2n + 1
    L = 8.0
    N = 2000
    x = np.linspace(-L, L, N)
    h = x[1] - x[0]
    main = (2 / h**2) + x**2
    off  = -np.ones(N - 1) / h**2
    E_vals, E_vecs = eigh_tridiagonal(main, off, select='i',
                                      select_range=(0, 4))

    # potential
    ax.plot(x, x**2, color='black', lw=1.6, alpha=0.7,
            label='potential  $V(x)=x^2$')

    palette = [BLUE, PURPLE, GREEN, RED, COLORS["warning"]]
    for n in range(5):
        psi = E_vecs[:, n]
        # normalize int |psi|^2 dx = 1
        psi = psi / np.sqrt(np.trapz(psi**2, x))
        # sign convention
        if psi[np.argmax(np.abs(psi))] < 0:
            psi = -psi
        # plot psi shifted to its energy level, scaled for visibility
        ax.fill_between(x, E_vals[n], E_vals[n] + 1.6 * psi,
                        color=palette[n], alpha=0.25)
        ax.plot(x, E_vals[n] + 1.6 * psi, color=palette[n], lw=1.8,
                label=fr'$n={n}$,  $E={E_vals[n]:.3f}$  '
                      fr'(exact $={2*n+1}$)')
        ax.axhline(E_vals[n], color=palette[n], lw=0.6, ls=':')

    ax.set_xlim(-5, 5); ax.set_ylim(-0.5, 11)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('energy $E$  (eigenfunctions overlaid)', fontsize=11)
    ax.set_title('Quantum harmonic oscillator: a Sturm-Liouville eigenproblem\n'
                 r"$-\psi'' + x^2\psi = E\psi$  on $\mathbb{R}$,  "
                 r'$\psi \to 0$ at infinity',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    save(fig, 'fig4_sturm_liouville.png')


# ---------------------------------------------------------------------------
# fig5: BVP solution methods -- four strategies on the same Bratu problem
# ---------------------------------------------------------------------------
def fig5_bvp_solution_methods():
    print('Building fig5_bvp_solution_methods...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8),
                             gridspec_kw={'width_ratios': [1.2, 1.0]})

    # Bratu:  y'' + lambda*exp(y) = 0,  y(0)=y(1)=0,  lambda = 2 (lower branch)
    lam = 2.0

    # (1) Shooting
    def rhs(x, Y):
        return [Y[1], -lam * np.exp(Y[0])]
    def F(s):
        sol = solve_ivp(rhs, [0, 1], [0, s], rtol=1e-10, atol=1e-12)
        return sol.y[0, -1]
    s_star = brentq(F, 0.1, 5.0)
    sol_sh = solve_ivp(rhs, [0, 1], [0, s_star], dense_output=True,
                       rtol=1e-10, atol=1e-12)
    xx = np.linspace(0, 1, 300)
    y_sh = sol_sh.sol(xx)[0]

    # (2) Finite difference (Newton iteration on the nonlinear system)
    N = 80
    h = 1.0 / N
    x_fd = np.linspace(0, 1, N + 1)
    y = np.zeros(N + 1)
    # Newton
    for it in range(30):
        # interior residual: (y_{i+1} - 2y_i + y_{i-1})/h^2 + lam*exp(y_i) = 0
        F_int = (y[2:] - 2*y[1:-1] + y[:-2]) / h**2 + lam * np.exp(y[1:-1])
        if np.max(np.abs(F_int)) < 1e-12:
            break
        # Jacobian (tridiagonal):
        diag = -2 / h**2 + lam * np.exp(y[1:-1])
        off  = np.ones(N - 2) / h**2
        from scipy.linalg import solve_banded
        ab = np.zeros((3, N - 1))
        ab[0, 1:] = off
        ab[1, :]  = diag
        ab[2, :-1] = off
        delta = solve_banded((1, 1), ab, -F_int)
        y[1:-1] += delta
    y_fd = y

    # (3) Collocation via solve_bvp
    def fun(x, Y): return np.vstack([Y[1], -lam * np.exp(Y[0])])
    def bc(Ya, Yb): return np.array([Ya[0], Yb[0]])
    x_init = np.linspace(0, 1, 10)
    Y_init = np.zeros((2, x_init.size))
    Y_init[0] = 0.5 * np.sin(np.pi * x_init)
    sol_bvp = solve_bvp(fun, bc, x_init, Y_init, tol=1e-10)
    y_bvp = sol_bvp.sol(xx)[0]

    # Plot all on same axes (they should overlap)
    ax = axes[0]
    ax.plot(xx, y_sh, color=BLUE, lw=3.5, alpha=0.6,
            label='shooting  +  brentq')
    ax.plot(x_fd, y_fd, 'o', color=PURPLE, ms=5, mec='white', mew=0.6,
            label=f'finite difference + Newton  ($N={N}$)')
    ax.plot(xx, y_bvp, color=GREEN, lw=1.6, ls='--',
            label='scipy.integrate.solve_bvp  (collocation)')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y(x)$')
    ax.set_title('Bratu problem  $y\'\' + 2 e^{y} = 0$,  $y(0)=y(1)=0$\n'
                 'three independent methods agree to plotting accuracy',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower center', fontsize=9.5)

    # Right: max difference between methods
    ax = axes[1]
    # interpolate finite difference onto xx
    y_fd_interp = np.interp(xx, x_fd, y_fd)
    methods = ['shoot vs FD', 'shoot vs solve_bvp', 'FD vs solve_bvp']
    diffs = [np.max(np.abs(y_sh - y_fd_interp)),
             np.max(np.abs(y_sh - y_bvp)),
             np.max(np.abs(y_fd_interp - y_bvp))]
    bars = ax.bar(methods, diffs, color=[BLUE, GREEN, PURPLE],
                  edgecolor='black', linewidth=1.0, width=0.55)
    ax.set_yscale('log')
    for b, v in zip(bars, diffs):
        ax.text(b.get_x() + b.get_width()/2, v * 1.4,
                f'{v:.2e}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'max $|y_{\rm A} - y_{\rm B}|$  (log)', fontsize=11)
    ax.set_title('Pairwise discrepancy between methods',
                 fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)

    fig.suptitle('BVP solution strategies: shooting, finite difference, collocation',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig5_bvp_solution_methods.png')


if __name__ == '__main__':
    fig1_shooting_method()
    fig2_finite_difference()
    fig3_eigenvalue_problem()
    fig4_sturm_liouville()
    fig5_bvp_solution_methods()
    print('\nChapter 12 figures complete.')
