"""
Chapter 14: Epidemic Models and Epidemiology.

Figures (5):
  fig1_sir_model.png            -- SIR dynamics + phase portrait + final-size
  fig2_r0_sensitivity.png       -- Effect of R0 on peak height/timing + final size
  fig3_vaccination_effect.png   -- Herd immunity threshold + vaccination scenarios
  fig4_seir_variant.png         -- SEIR vs SIR + effect of latent period sigma
  fig5_covid_example.png        -- COVID-style: asymptomatic split + Re(t) + intervention timeline

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
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/14-epidemiology',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/14-传染病模型与流行病学',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save(fig, name):
    for d in OUT_DIRS:
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print(f'  saved {name}')


# ---------- core models ----------
def sir(t, y, beta, gamma, N=1.0):
    S, I, R = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]


def seir(t, y, beta, sigma, gamma, N=1.0):
    S, E, I, R = y
    return [-beta * S * I / N,
            beta * S * I / N - sigma * E,
            sigma * E - gamma * I,
            gamma * I]


def covid(t, y, beta, sigma, gamma, p, kappa, N=1.0):
    S, E, Ia, Is, R = y
    inf_force = beta * (Is + kappa * Ia) / N
    return [-inf_force * S,
            inf_force * S - sigma * E,
            p * sigma * E - gamma * Ia,
            (1 - p) * sigma * E - gamma * Is,
            gamma * (Ia + Is)]


def final_size(R0, S_inf_init=0.99):
    """Solve S_inf = S0 * exp(-R0 (1 - S_inf)) by fixed-point."""
    s = 0.5
    for _ in range(200):
        s = S_inf_init * np.exp(-R0 * (1 - s))
    return s


# ---------------------------------------------------------------------------
# fig1: classic SIR -- time series + phase portrait + cumulative + final size
# ---------------------------------------------------------------------------
def fig1_sir_model():
    print('Building fig1_sir_model...')
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)

    R0 = 2.5
    gamma = 1 / 10
    beta = R0 * gamma
    N = 1.0
    y0 = [N - 1e-3, 1e-3, 0.0]
    t = np.linspace(0, 200, 2000)
    sol = solve_ivp(sir, (0, 200), y0, args=(beta, gamma, N),
                    t_eval=t, dense_output=True, rtol=1e-8)
    S, I, R = sol.y

    # (a) time series
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, S, color=BLUE,   lw=2.2, label='S  susceptible')
    ax.plot(t, I, color=RED,    lw=2.2, label='I  infectious')
    ax.plot(t, R, color=GREEN,  lw=2.2, label='R  recovered')
    # peak marker
    i_peak = I.argmax()
    ax.scatter([t[i_peak]], [I[i_peak]], color='black', s=60, zorder=5,
               edgecolor='white', linewidth=1.2)
    ax.annotate(f'peak  $I^* = {I[i_peak]:.3f}$\n at  $t \\approx {t[i_peak]:.0f}$ d',
                xy=(t[i_peak], I[i_peak]),
                xytext=(t[i_peak] + 18, I[i_peak] + 0.12),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                fontsize=10)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('fraction of population', fontsize=11)
    ax.set_title(f'SIR dynamics  ($R_0 = {R0},\\ 1/\\gamma = {1/gamma:.0f}$ d)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='center right', fontsize=10)
    ax.set_xlim(0, 200); ax.set_ylim(0, 1.05)

    # (b) phase plane S vs I; nullcline + R0/N threshold
    ax = fig.add_subplot(gs[0, 1])
    Sg = np.linspace(0.01, 1.0, 220)
    Ig = np.linspace(0.0, 0.5, 220)
    SS, II = np.meshgrid(Sg, Ig)
    dI = beta * SS * II / N - gamma * II
    ax.contourf(SS, II, dI, levels=[-1, 0, 1], colors=['#cfe2ff', '#f5c2c7'],
                alpha=0.55)
    for R0_v, color in zip([1.5, 2.5, 4.0], [GREEN, RED, PURPLE]):
        b = R0_v * gamma
        s = solve_ivp(sir, (0, 250), y0, args=(b, gamma, N),
                      t_eval=np.linspace(0, 250, 1500), rtol=1e-8)
        ax.plot(s.y[0], s.y[1], color=color, lw=2.0, label=f'$R_0 = {R0_v}$')
    ax.axvline(1 / R0, color='black', lw=1.2, ls='--',
               label=f'$S = 1/R_0 = {1/R0:.2f}$  ($I$ peaks)')
    ax.set_xlabel('S', fontsize=11); ax.set_ylabel('I', fontsize=11)
    ax.set_title('Phase plane:  trajectories spiral toward S-axis',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)

    # (c) cumulative incidence (1 - S)
    ax = fig.add_subplot(gs[1, 0])
    cum = 1 - S
    ax.plot(t, cum, color=PURPLE, lw=2.2, label='cumulative infected  $1 - S$')
    fs = 1 - final_size(R0, y0[0])
    ax.axhline(fs, color=RED, lw=1.4, ls='--',
               label=f'final size  $\\approx {fs:.2f}$')
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('fraction ever infected', fontsize=11)
    ax.set_title('Cumulative incidence -> final-size attractor',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.05)

    # (d) final-size relation: S_infty(R0)
    ax = fig.add_subplot(gs[1, 1])
    R0s = np.linspace(0.5, 5.0, 250)
    Sinf = np.array([final_size(r) for r in R0s])
    ax.plot(R0s, Sinf, color=BLUE, lw=2.2, label='$S_\\infty$')
    ax.plot(R0s, 1 - Sinf, color=RED, lw=2.2, label='$1 - S_\\infty$  (ever infected)')
    ax.axvline(1.0, color='black', lw=1.2, ls='--', label='threshold  $R_0 = 1$')
    ax.set_xlabel(r'$R_0$', fontsize=11)
    ax.set_ylabel('final fraction', fontsize=11)
    ax.set_title('Final-size relation  $S_\\infty = S_0 e^{-R_0(1-S_\\infty)}$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='center right', fontsize=10)
    ax.set_xlim(0.5, 5); ax.set_ylim(0, 1.05)

    fig.suptitle('SIR model  --  the workhorse of mathematical epidemiology',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig1_sir_model.png')


# ---------------------------------------------------------------------------
# fig2: R0 sensitivity -- peak vs R0, peak time vs R0, family of curves
# ---------------------------------------------------------------------------
def fig2_r0_sensitivity():
    print('Building fig2_r0_sensitivity...')
    fig = plt.figure(figsize=(15, 5.8))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    gamma = 1 / 10
    N = 1.0
    y0 = [N - 1e-3, 1e-3, 0.0]
    t = np.linspace(0, 220, 2200)

    R0_list = [1.2, 1.6, 2.0, 2.5, 3.5, 5.0]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(R0_list)))

    # (a) family of I(t)
    ax = fig.add_subplot(gs[0, 0])
    for R0, color in zip(R0_list, cmap):
        b = R0 * gamma
        s = solve_ivp(sir, (0, 220), y0, args=(b, gamma, N), t_eval=t,
                      rtol=1e-8)
        ax.plot(t, s.y[1], color=color, lw=2.0, label=f'$R_0 = {R0}$')
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('I(t)', fontsize=11)
    ax.set_title('Higher $R_0$  ->  bigger and earlier peak',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # (b) peak height + peak time vs R0
    ax = fig.add_subplot(gs[0, 1])
    R0_dense = np.linspace(1.05, 6.0, 60)
    peaks = []; tpeaks = []
    for R0 in R0_dense:
        b = R0 * gamma
        s = solve_ivp(sir, (0, 400), y0, args=(b, gamma, N),
                      t_eval=np.linspace(0, 400, 4000), rtol=1e-9)
        peaks.append(s.y[1].max())
        tpeaks.append(s.t[s.y[1].argmax()])
    peaks = np.array(peaks); tpeaks = np.array(tpeaks)
    ax.plot(R0_dense, peaks, color=RED, lw=2.2, label='peak  $I^*$')
    # analytical: I_max = 1 - 1/R0 - ln(R0)/R0   (for S0=1)
    ana = 1 - 1 / R0_dense - np.log(R0_dense) / R0_dense
    ax.plot(R0_dense, ana, color='black', lw=1.2, ls='--',
            label=r'analytical  $1 - 1/R_0 - \ln R_0 / R_0$')
    ax.set_xlabel(r'$R_0$', fontsize=11)
    ax.set_ylabel('peak fraction infected', fontsize=11, color=RED)
    ax.tick_params(axis='y', labelcolor=RED)
    ax.set_xlim(1, 6); ax.set_ylim(0, 0.6)
    ax.legend(loc='upper left', fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(R0_dense, tpeaks, color=BLUE, lw=2.2)
    ax2.set_ylabel('time of peak  (days)', fontsize=11, color=BLUE)
    ax2.tick_params(axis='y', labelcolor=BLUE)
    ax2.grid(False)
    ax.set_title('Peak size  &  peak timing  vs.  $R_0$',
                 fontsize=12, fontweight='bold')

    # (c) final size + herd immunity threshold
    ax = fig.add_subplot(gs[0, 2])
    R0_dense = np.linspace(0.5, 6.0, 250)
    Sinf = np.array([final_size(r) for r in R0_dense])
    HIT = np.where(R0_dense > 1, 1 - 1 / R0_dense, 0)
    ax.fill_between(R0_dense, 0, HIT, color=GREEN, alpha=0.18,
                    label='herd-immunity threshold')
    ax.plot(R0_dense, 1 - Sinf, color=RED, lw=2.2,
            label='final fraction infected')
    ax.plot(R0_dense, HIT, color=GREEN, lw=2.0, ls='--',
            label='HIT  $1 - 1/R_0$')
    ax.axvline(1.0, color='black', lw=1.0, ls=':')
    ax.set_xlabel(r'$R_0$', fontsize=11)
    ax.set_ylabel('fraction', fontsize=11)
    ax.set_title('Final size  vs.  herd-immunity threshold',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0.5, 6); ax.set_ylim(0, 1.05)

    fig.suptitle(r'Sensitivity to $R_0 = \beta / \gamma$',
                 fontsize=14, fontweight='bold', y=1.02)
    save(fig, 'fig2_r0_sensitivity.png')


# ---------------------------------------------------------------------------
# fig3: vaccination -- compare unvaccinated, partial, herd-threshold scenarios
# ---------------------------------------------------------------------------
def fig3_vaccination_effect():
    print('Building fig3_vaccination_effect...')
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.28)

    R0 = 3.0
    gamma = 1 / 10
    beta = R0 * gamma
    N = 1.0
    HIT = 1 - 1 / R0
    t = np.linspace(0, 200, 2000)

    # (a) family of I(t) for different pre-vaccination coverage v
    ax = fig.add_subplot(gs[0, 0])
    coverages = [0.0, 0.20, 0.40, HIT, 0.85]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(coverages)))
    for v, color in zip(coverages, cmap):
        S0 = (1 - v) - 1e-3; R0_init = v
        s = solve_ivp(sir, (0, 200), [S0, 1e-3, R0_init],
                      args=(beta, gamma, N), t_eval=t, rtol=1e-9)
        lab = f'v = {v:.2f}'
        if abs(v - HIT) < 1e-3:
            lab += f'  (= HIT  ${1 - 1/R0:.2f}$)'
        ax.plot(t, s.y[1], color=color, lw=2.0, label=lab)
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('I(t)', fontsize=11)
    ax.set_title(f'Vaccination crushes the curve  ($R_0 = {R0}$)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # (b) peak vs coverage, with HIT marked
    ax = fig.add_subplot(gs[0, 1])
    vs = np.linspace(0, 0.95, 60)
    peaks = []; finals = []
    for v in vs:
        S0 = (1 - v) - 1e-3
        s = solve_ivp(sir, (0, 400), [S0, 1e-3, v],
                      args=(beta, gamma, N),
                      t_eval=np.linspace(0, 400, 4000), rtol=1e-9)
        peaks.append(s.y[1].max())
        finals.append(s.y[2][-1] - v)
    ax.plot(vs, peaks, color=RED, lw=2.2, label='peak fraction infected')
    ax.plot(vs, finals, color=BLUE, lw=2.2, label='additional fraction infected')
    ax.axvline(HIT, color=GREEN, lw=1.6, ls='--',
               label=f'HIT  $1 - 1/R_0 = {HIT:.2f}$')
    ax.fill_between([HIT, 1.0], 0, 1, color=GREEN, alpha=0.10)
    ax.set_xlabel('vaccination coverage  v', fontsize=11)
    ax.set_ylabel('fraction', fontsize=11)
    ax.set_title('Peak  &  total infected  vs.  coverage',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 0.95); ax.set_ylim(0, 1.0)

    # (c) HIT vs R0  (table-style chart with example diseases)
    ax = fig.add_subplot(gs[1, 0])
    R0_axis = np.linspace(1, 18, 400)
    HIT_axis = 1 - 1 / R0_axis
    ax.plot(R0_axis, HIT_axis, color=BLUE, lw=2.4)
    ax.fill_between(R0_axis, 0, HIT_axis, color=BLUE, alpha=0.10)
    diseases = [
        ('Influenza',    1.5, '#888'),
        ('COVID (Wild)', 2.5, BLUE),
        ('SARS',         3.0, PURPLE),
        ('COVID (Delta)',5.0, '#b08900'),
        ('Smallpox',     6.0, '#c08'),
        ('Mumps',       10.0, GREEN),
        ('Measles',     15.0, RED),
    ]
    for name, R0v, color in diseases:
        h = 1 - 1 / R0v
        ax.scatter([R0v], [h], s=80, color=color, zorder=5,
                   edgecolor='white', linewidth=1.3)
        ax.annotate(f'{name}\n$R_0 = {R0v}$, HIT = {h:.0%}',
                    xy=(R0v, h), xytext=(R0v + 0.4, h - 0.18),
                    fontsize=8.5, color=color)
    ax.set_xlabel(r'$R_0$', fontsize=11)
    ax.set_ylabel('herd-immunity threshold  $1 - 1/R_0$', fontsize=11)
    ax.set_title('How much of the population must be immune?',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(1, 18); ax.set_ylim(0, 1.0)

    # (d) imperfect vaccine: effective coverage v * VE
    ax = fig.add_subplot(gs[1, 1])
    VEs = [0.5, 0.7, 0.9, 1.0]
    R0_axis = np.linspace(1.05, 8, 400)
    for VE, color in zip(VEs, [RED, '#b08900', GREEN, BLUE]):
        v_req = (1 - 1 / R0_axis) / VE
        v_req = np.clip(v_req, 0, 1.05)
        ax.plot(R0_axis, v_req, color=color, lw=2.0, label=f'VE = {VE:.0%}')
    ax.axhline(1.0, color='gray', lw=1.0, ls=':')
    ax.fill_between(R0_axis, 1.0, 1.2, color=RED, alpha=0.12,
                    label='infeasible')
    ax.set_xlabel(r'$R_0$', fontsize=11)
    ax.set_ylabel('required coverage  $v$', fontsize=11)
    ax.set_title(r'Imperfect vaccines:  $v \geq (1 - 1/R_0)/\mathrm{VE}$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(1, 8); ax.set_ylim(0, 1.2)

    fig.suptitle('Vaccination and herd immunity',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig3_vaccination_effect.png')


# ---------------------------------------------------------------------------
# fig4: SEIR variant -- compare with SIR, vary sigma
# ---------------------------------------------------------------------------
def fig4_seir_variant():
    print('Building fig4_seir_variant...')
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.28)

    N = 1.0
    R0 = 3.0
    gamma = 1 / 7
    beta = R0 * gamma
    t = np.linspace(0, 200, 2000)

    # (a) SEIR all four compartments
    sigma = 1 / 5
    sol = solve_ivp(seir, (0, 200), [N - 1e-3, 0, 1e-3, 0],
                    args=(beta, sigma, gamma, N), t_eval=t, rtol=1e-9)
    ax = fig.add_subplot(gs[0, 0])
    labs = ['S  susceptible', 'E  exposed', 'I  infectious', 'R  recovered']
    for i, (color, lab) in enumerate(zip([BLUE, PURPLE, RED, GREEN], labs)):
        ax.plot(t, sol.y[i], color=color, lw=2.2, label=lab)
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('fraction', fontsize=11)
    ax.set_title(f'SEIR  (latent  $1/\\sigma = {1/sigma:.0f}$ d, '
                 f'infectious  $1/\\gamma = {1/gamma:.0f}$ d)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='center right', fontsize=10)
    ax.set_ylim(0, 1.05)

    # (b) compare SEIR vs SIR
    ax = fig.add_subplot(gs[0, 1])
    sol_sir = solve_ivp(sir, (0, 200), [N - 1e-3, 1e-3, 0],
                        args=(beta, gamma, N), t_eval=t, rtol=1e-9)
    ax.plot(t, sol_sir.y[1], color=RED, lw=2.4, label='SIR  $I$')
    ax.plot(t, sol.y[2],     color=PURPLE, lw=2.4, label='SEIR  $I$')
    ax.plot(t, sol.y[1],     color=PURPLE, lw=1.4, ls='--', label='SEIR  $E$')
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('fraction', fontsize=11)
    ax.set_title('SEIR delays and slightly lowers the peak vs SIR',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    # (c) varying latent period
    ax = fig.add_subplot(gs[1, 0])
    inv_sigmas = [0.5, 2, 5, 10, 20]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(inv_sigmas)))
    for inv_s, color in zip(inv_sigmas, cmap):
        sg = 1 / inv_s
        sol = solve_ivp(seir, (0, 250), [N - 1e-3, 0, 1e-3, 0],
                        args=(beta, sg, gamma, N),
                        t_eval=np.linspace(0, 250, 2500), rtol=1e-9)
        ax.plot(sol.t, sol.y[2], color=color, lw=2.0,
                label=f'$1/\\sigma = {inv_s:.1f}$ d')
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('I(t)', fontsize=11)
    ax.set_title('Longer latent period  ->  later, broader peak',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # (d) growth rate r vs sigma (analytical comparison: r_SIR vs r_SEIR)
    ax = fig.add_subplot(gs[1, 1])
    sigma_vals = np.linspace(0.05, 2.0, 200)
    # Linear analysis at S=1: SIR r = beta - gamma; SEIR r is positive root of
    #   r^2 + (sigma + gamma) r + sigma (gamma - beta) = 0
    a, b_co, c_co = 1.0, sigma_vals + gamma, sigma_vals * (gamma - beta)
    disc = b_co**2 - 4 * a * c_co
    r_seir = (-b_co + np.sqrt(disc)) / (2 * a)
    r_sir  = beta - gamma
    ax.plot(1 / sigma_vals, r_seir, color=PURPLE, lw=2.2, label='SEIR growth rate')
    ax.axhline(r_sir, color=RED, lw=2.0, ls='--', label='SIR growth rate')
    # doubling time
    ax2 = ax.twinx()
    ax2.plot(1 / sigma_vals, np.log(2) / r_seir, color=GREEN, lw=1.6,
             ls=':', label='doubling time')
    ax.set_xlabel(r'latent period  $1/\sigma$  (days)', fontsize=11)
    ax.set_ylabel('initial growth rate  r  (per day)', fontsize=11,
                  color=PURPLE)
    ax2.set_ylabel('doubling time  ln(2)/r  (days)', fontsize=11, color=GREEN)
    ax.tick_params(axis='y', labelcolor=PURPLE)
    ax2.tick_params(axis='y', labelcolor=GREEN)
    ax.set_xlim(0.5, 20); ax.set_ylim(0, max(r_sir, r_seir.max()) * 1.1)
    ax2.grid(False)
    ax.set_title('Latent period slows initial spread\n'
                 r'(SEIR $r$ $\to$ SIR $r$ as $\sigma \to \infty$)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('SEIR model:  adding a latent period to SIR',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig4_seir_variant.png')


# ---------------------------------------------------------------------------
# fig5: COVID-style: asymptomatic split + intervention timeline + Re(t)
# ---------------------------------------------------------------------------
def fig5_covid_example():
    print('Building fig5_covid_example...')
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.28)

    N = 1.0
    sigma = 1 / 4         # 4-day latent
    gamma = 1 / 7         # 7-day infectious
    p = 0.4               # 40% asymptomatic
    kappa = 0.5           # asymp ~50% as infectious

    # piecewise beta for interventions
    def beta_t(t):
        if t < 30:        return 0.6      # baseline R0 ~ 4.2
        elif t < 60:      return 0.18     # lockdown
        elif t < 120:     return 0.30     # partial relaxation
        else:             return 0.40     # variant + relaxation

    def covid_t(t, y):
        return covid(t, y, beta_t(t), sigma, gamma, p, kappa, N)

    y0 = [N - 1e-4, 0, 0, 1e-4, 0]
    sol = solve_ivp(covid_t, (0, 220), y0, t_eval=np.linspace(0, 220, 2200),
                    rtol=1e-9, max_step=0.5)
    S, E, Ia, Is, R = sol.y
    t = sol.t
    I_total = Ia + Is

    # (a) compartments
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, Is, color=RED,    lw=2.0, label='$I_s$  symptomatic')
    ax.plot(t, Ia, color=PURPLE, lw=2.0, label='$I_a$  asymptomatic')
    ax.plot(t, E,  color='#b08900', lw=2.0, label='E  exposed')
    ax.plot(t, R,  color=GREEN,  lw=2.0, label='R  recovered')
    # shade interventions
    ax.axvspan(0,   30, color='#fff3cd', alpha=0.4)
    ax.axvspan(30,  60, color='#cfe2ff', alpha=0.5)
    ax.axvspan(60, 120, color='#fff3cd', alpha=0.3)
    ax.axvspan(120, 220, color='#f5c2c7', alpha=0.3)
    for x_v, lab in [(15, 'baseline'), (45, 'lockdown'),
                     (90, 'partial relax'), (170, 'variant')]:
        ax.text(x_v, 0.32, lab, ha='center', fontsize=9, color='black')
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('fraction', fontsize=11)
    ax.set_title('COVID-style model:  asymptomatic + intervention timeline',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 0.4)

    # (b) effective reproduction number Re(t)
    Re = np.array([beta_t(tt) * (1 - p + kappa * p) / gamma * S[i]
                   for i, tt in enumerate(t)])
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, Re, color=BLUE, lw=2.4)
    ax.axhline(1.0, color=RED, lw=1.4, ls='--', label='$R_e = 1$')
    ax.fill_between(t, Re, 1.0, where=(Re > 1), color=RED, alpha=0.15,
                    label='growing')
    ax.fill_between(t, Re, 1.0, where=(Re < 1), color=GREEN, alpha=0.15,
                    label='shrinking')
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel(r'$R_e(t)$', fontsize=11)
    ax.set_title('Effective reproduction number  $R_e(t) = R_0(t)\\,S(t)$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, max(Re) * 1.1)

    # (c) reported (only symptomatic) vs true (Is + Ia)
    ax = fig.add_subplot(gs[1, 0])
    reported = Is * 0.7   # also assume 70% of symptomatic detected
    ax.plot(t, I_total, color=PURPLE, lw=2.2, label='true infectious  $I_s + I_a$')
    ax.plot(t, Is, color=RED, lw=2.0, label='all symptomatic  $I_s$')
    ax.plot(t, reported, color='black', lw=1.6, ls='--', label='reported (~ 0.7 $I_s$)')
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('fraction', fontsize=11)
    ax.set_title('Reporting iceberg:  reported << true prevalence',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # (d) cumulative + counterfactual no-intervention
    def covid_const(t, y):
        return covid(t, y, 0.6, sigma, gamma, p, kappa, N)
    sol_nb = solve_ivp(covid_const, (0, 220), y0,
                       t_eval=np.linspace(0, 220, 2200), rtol=1e-9)
    cum  = 1 - S
    cum0 = 1 - sol_nb.y[0]
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, cum * 100, color=BLUE, lw=2.4,
            label='with interventions')
    ax.plot(sol_nb.t, cum0 * 100, color=RED, lw=2.4, ls='--',
            label='no interventions  (counterfactual)')
    saved = (cum0[-1] - cum[-1]) * 100
    ax.text(120, 30, f'cases averted  $\\approx$  {saved:.0f}% of population',
            fontsize=11, color='black',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray'))
    ax.set_xlabel('time  (days)', fontsize=11)
    ax.set_ylabel('cumulative infected  (%)', fontsize=11)
    ax.set_title('Counterfactual:  measuring lives saved',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('COVID-style extension:  asymptomatic transmission, '
                 'time-varying $R_e$, intervention impact',
                 fontsize=14, fontweight='bold', y=0.995)
    save(fig, 'fig5_covid_example.png')


if __name__ == '__main__':
    fig1_sir_model()
    fig2_r0_sensitivity()
    fig3_vaccination_effect()
    fig4_seir_variant()
    fig5_covid_example()
    print('\nChapter 14 figures complete.')
