"""
Chapter 16: Fundamentals of Control Theory.

Figures:
  fig1_pid_step_response.png    -- P, PI, PD, PID step responses (open-loop tuning intuition)
  fig2_root_locus.png           -- Root locus showing closed-loop poles vs gain K
  fig3_bode_plot.png            -- Bode plot with gain margin / phase margin annotations
  fig4_state_space.png          -- State-space block diagram + LQR pole placement comparison
  fig5_feedback_loop.png        -- Closed-loop feedback architecture (plant + controller + sensor)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from scipy.signal import lti, step, bode, place_poles
from scipy.linalg import solve_continuous_are
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
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/16-control-theory',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/16-控制理论基础',
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
# fig1: PID step responses
# ---------------------------------------------------------------------------
def fig1_pid_step_response():
    """Compare P, PI, PD, PID closed-loop responses for a 2nd-order plant."""
    print('Building fig1_pid_step_response...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # Plant: G(s) = 1 / (s^2 + 0.6 s + 1) -- underdamped
    # Closed-loop response with PID controller via time-domain simulation
    # (we simulate the loop with scipy.integrate to avoid implementing
    # discrete approximations of the controller as a transfer function)
    def closed_loop(Kp, Ki, Kd, T=20.0, n=4000):
        t = np.linspace(0, T, n)
        dt = t[1] - t[0]
        r = 1.0  # step reference
        x1 = 0.0  # output y
        x2 = 0.0  # dy/dt
        integ = 0.0
        prev_e = r - x1
        ys = []
        for _ in t:
            e = r - x1
            integ += e * dt
            de = (e - prev_e) / dt
            prev_e = e
            u = Kp * e + Ki * integ + Kd * de
            # plant: y'' + 0.6 y' + y = u
            x2_new = x2 + dt * (u - 0.6 * x2 - x1)
            x1_new = x1 + dt * x2
            x1, x2 = x1_new, x2_new
            ys.append(x1)
        return t, np.array(ys)

    # Left plot: progressive controller types
    ax = axes[0]
    cases = [
        ('No control (open loop)', 0.0, 0.0, 0.0, 'gray'),
        ('P  (Kp=2)',              2.0, 0.0, 0.0, BLUE),
        ('PI (Kp=2, Ki=1.5)',      2.0, 1.5, 0.0, PURPLE),
        ('PD (Kp=2, Kd=0.7)',      2.0, 0.0, 0.7, GREEN),
        ('PID (Kp=2, Ki=1.5, Kd=0.7)', 2.0, 1.5, 0.7, RED),
    ]
    for label, Kp, Ki, Kd, color in cases:
        if Kp == 0 and Ki == 0 and Kd == 0:
            # open loop step input directly
            sys = lti([1], [1, 0.6, 1])
            t_out, y = step(sys, T=np.linspace(0, 20, 4000))
            ax.plot(t_out, y, color=color, lw=2.0, ls='--', label=label)
        else:
            t, y = closed_loop(Kp, Ki, Kd)
            ax.plot(t, y, color=color, lw=2.0, label=label)
    ax.axhline(1.0, color='black', lw=0.8, ls=':')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Output  $y(t)$', fontsize=11)
    ax.set_title('Closed-loop step response  (plant $1/(s^2+0.6s+1)$)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 20); ax.set_ylim(-0.05, 1.6)

    # Right plot: trade-off: P gain too high -> oscillation
    ax = axes[1]
    for Kp, color in [(0.5, BLUE), (2.0, PURPLE), (5.0, GREEN), (10.0, RED)]:
        t, y = closed_loop(Kp, 0.0, 0.0)
        ax.plot(t, y, color=color, lw=1.8, label=f'$K_p={Kp}$')
    ax.axhline(1.0, color='black', lw=0.8, ls=':', label='setpoint')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Output  $y(t)$', fontsize=11)
    ax.set_title('P-only:  high gain reduces error but causes ringing',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 20); ax.set_ylim(-0.1, 2.0)

    fig.suptitle('PID Controller Tuning  --  the three terms shape transient response',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    save(fig, 'fig1_pid_step_response.png')


# ---------------------------------------------------------------------------
# fig2: Root locus
# ---------------------------------------------------------------------------
def fig2_root_locus():
    """Root locus of L(s) = K / (s (s+2)(s+5))."""
    print('Building fig2_root_locus...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # Open-loop transfer function L(s) = K * num/den
    # Closed loop char poly: den + K * num = 0
    num = np.array([1.0])
    poles_OL = np.array([0.0, -2.0, -5.0])
    den = np.poly(poles_OL)

    K_vals = np.linspace(0.001, 80, 2000)
    locus = []
    for K in K_vals:
        char = den.copy()
        char[-len(num):] += K * num
        locus.append(np.roots(char))
    locus = np.array(locus)  # (n_K, 3)

    ax = axes[0]
    # Plot each branch separately by sorting at each step
    branches = locus.copy()
    # naive sort by closeness: at each K, match each new root to nearest previous
    for i in range(1, len(K_vals)):
        prev = branches[i - 1]
        curr = locus[i].copy()
        order = []
        used = [False, False, False]
        for p in prev:
            dists = [abs(p - c) if not used[j] else np.inf
                     for j, c in enumerate(curr)]
            j = int(np.argmin(dists)); used[j] = True
            order.append(j)
        branches[i] = curr[order]

    colors = [BLUE, PURPLE, GREEN]
    for b in range(3):
        ax.plot(branches[:, b].real, branches[:, b].imag,
                color=colors[b], lw=1.6, alpha=0.85,
                label=f'branch {b+1}')

    # Open-loop poles (X), and asymptote intersection
    ax.scatter(poles_OL.real, poles_OL.imag, marker='x',
               color='black', s=140, lw=2.5, zorder=5,
               label='open-loop poles')
    # Asymptote centroid: sigma_a = (sum poles - sum zeros) / (n_p - n_z)
    sigma_a = (sum(poles_OL) - 0) / (3 - 0)
    ax.axvline(0, color='gray', lw=0.6, ls=':')
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    # Stability boundary
    ax.axvline(0, color=RED, lw=1.2, alpha=0.4, label='stability boundary')
    ax.fill_betweenx([-15, 15], 0, 5, color=RED, alpha=0.05)
    ax.text(2.5, 12, 'unstable\n(Re$>0$)', color=RED, fontsize=10,
            ha='center', va='top', fontweight='bold')
    ax.text(-7, 12, 'stable\n(Re$<0$)', color=GREEN, fontsize=10,
            ha='center', va='top', fontweight='bold')
    # Find K_critical -- where branches cross imag axis
    # For L(s) = K / s(s+2)(s+5), Routh -> K_critical = 70
    K_crit = 70.0
    omega_c = np.sqrt(10.0)
    ax.scatter([0, 0], [omega_c, -omega_c], marker='o',
               s=80, edgecolor=RED, facecolor='white', lw=2.0, zorder=6,
               label=fr'imag-axis crossing  $K\approx{K_crit:.0f}$')
    ax.set_xlim(-8, 5); ax.set_ylim(-13, 13)
    ax.set_xlabel('Real axis', fontsize=11)
    ax.set_ylabel('Imag axis', fontsize=11)
    ax.set_title(r'Root locus of  $L(s) = \dfrac{K}{s(s+2)(s+5)}$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')

    # Right plot: closed-loop step response at three K values
    ax = axes[1]
    for K, color in [(10.0, BLUE), (40.0, PURPLE), (70.0, RED)]:
        char = den.copy()
        char[-1] += K
        sys = lti([K], char)
        t_out, y = step(sys, T=np.linspace(0, 12, 4000))
        ax.plot(t_out, y, color=color, lw=1.8, label=f'$K={K:.0f}$')
    ax.axhline(1.0, color='black', lw=0.8, ls=':')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Output', fontsize=11)
    ax.set_title('Closed-loop step response moves through stability boundary',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 12); ax.set_ylim(-0.5, 2.5)

    fig.suptitle('Root Locus:  closed-loop poles trace continuous curves as $K$ varies',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    save(fig, 'fig2_root_locus.png')


# ---------------------------------------------------------------------------
# fig3: Bode plot
# ---------------------------------------------------------------------------
def fig3_bode_plot():
    """Bode magnitude/phase with gain margin and phase margin annotated."""
    print('Building fig3_bode_plot...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 7),
                             gridspec_kw={'width_ratios': [1.4, 1.0]})

    # Stable plant L(s) = 100 / (s (s+1)(s+10))
    sys = lti([100.0], np.poly([0, -1, -10]))
    w = np.logspace(-2, 3, 2000)
    w, mag, phase = bode(sys, w=w)

    # Find crossover frequencies
    # gain crossover: |L|=1 i.e. mag (dB) = 0
    idx_gc = np.argmin(np.abs(mag))
    wg, pm = w[idx_gc], 180 + phase[idx_gc]
    # phase crossover: phase = -180
    idx_pc = np.argmin(np.abs(phase + 180))
    wp, gm = w[idx_pc], -mag[idx_pc]

    # Magnitude
    ax = axes[0, 0]
    ax.semilogx(w, mag, color=BLUE, lw=2.0)
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
    ax.axvline(wg, color=GREEN, lw=1.0, ls=':')
    ax.axvline(wp, color=RED,   lw=1.0, ls=':')
    # gain margin annotation
    ax.annotate('', xy=(wp, 0), xytext=(wp, mag[idx_pc]),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.6))
    ax.text(wp * 1.4, mag[idx_pc] / 2,
            fr'GM $\approx {gm:+.1f}$ dB', color=RED, fontsize=11,
            fontweight='bold')
    ax.set_ylabel('Magnitude (dB)', fontsize=11)
    ax.set_title(r'Bode magnitude  --  $L(s) = 100 / (s(s+1)(s+10))$',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(w[0], w[-1])

    # Phase
    ax = axes[1, 0]
    ax.semilogx(w, phase, color=PURPLE, lw=2.0)
    ax.axhline(-180, color='black', lw=0.8, ls='--', alpha=0.6)
    ax.axvline(wg, color=GREEN, lw=1.0, ls=':')
    ax.axvline(wp, color=RED,   lw=1.0, ls=':')
    ax.annotate('', xy=(wg, -180), xytext=(wg, phase[idx_gc]),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.6))
    ax.text(wg * 1.4, (phase[idx_gc] - 180) / 2,
            fr'PM $\approx {pm:+.1f}^\circ$', color=GREEN, fontsize=11,
            fontweight='bold')
    ax.set_xlabel(r'Frequency  $\omega$  (rad/s)', fontsize=11)
    ax.set_ylabel('Phase (deg)', fontsize=11)
    ax.set_xlim(w[0], w[-1])

    # Right: Nyquist-style polar view (re vs im) showing -1 point
    ax = axes[0, 1]
    re = 10 ** (mag / 20) * np.cos(np.deg2rad(phase))
    im = 10 ** (mag / 20) * np.sin(np.deg2rad(phase))
    ax.plot(re, im, color=BLUE, lw=1.6, label=r'$L(j\omega)$')
    ax.plot(re, -im, color=BLUE, lw=1.6, ls=':', alpha=0.6,
            label=r'mirror (negative $\omega$)')
    ax.scatter([-1], [0], color=RED, s=80, marker='X', zorder=5,
               label=r'$-1+0j$')
    circ = Circle((0, 0), 1.0, fill=False, ls='--', color='gray', lw=1.0)
    ax.add_patch(circ)
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlim(-3.5, 1.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Re $L(j\omega)$', fontsize=11)
    ax.set_ylabel(r'Im $L(j\omega)$', fontsize=11)
    ax.set_title('Nyquist:  encirclement of $-1$ = instability',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)

    # Stability "speedometer"
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Metric', 'Value', 'Healthy?'],
        [r'Gain crossover $\omega_g$', f'{wg:.2f} rad/s', '--'],
        [r'Phase crossover $\omega_p$', f'{wp:.2f} rad/s', '--'],
        ['Gain margin (GM)', f'{gm:+.1f} dB', 'GM>6 dB' + (' ok' if gm > 6 else ' marginal')],
        ['Phase margin (PM)', f'{pm:+.1f} deg', 'PM>30 deg' + (' ok' if pm > 30 else ' marginal')],
    ]
    tbl = ax.table(cellText=table_data, loc='center',
                   cellLoc='left', colWidths=[0.42, 0.30, 0.28])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(3):
        tbl[(0, j)].set_facecolor('#e0e7ff'); tbl[(0, j)].set_text_props(weight='bold')

    fig.suptitle('Frequency-Domain Stability:  gain margin, phase margin, Nyquist',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig3_bode_plot.png')


# ---------------------------------------------------------------------------
# fig4: state-space representation -- visualization + LQR vs pole-placement
# ---------------------------------------------------------------------------
def fig4_state_space():
    """State-space block diagram + LQR design example for inverted pendulum."""
    print('Building fig4_state_space...')
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2],
                          hspace=0.35, wspace=0.35)

    # ---- (a) state-space block diagram (top-left + top-mid) ----
    ax = fig.add_subplot(gs[0, 0:2])
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')

    def block(ax, x, y, w, h, label, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                             linewidth=1.8, edgecolor=color,
                             facecolor=color, alpha=0.18)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=11, fontweight='bold')

    def arrow(ax, x1, y1, x2, y2, label='', offset=(0, 0.25)):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='->', mutation_scale=18,
                                     color='black', lw=1.4))
        if label:
            ax.text((x1 + x2) / 2 + offset[0], (y1 + y2) / 2 + offset[1],
                    label, ha='center', fontsize=10, style='italic')

    # u -> B -> + -> integrator x -> C -> y
    block(ax, 1.5, 2.2, 1.0, 1.0, r'$B$', BLUE)
    block(ax, 4.5, 2.2, 1.4, 1.0, r'$\int dt$', GREEN)
    block(ax, 7.5, 2.2, 1.0, 1.0, r'$C$', PURPLE)
    block(ax, 4.7, 0.4, 1.0, 1.0, r'$A$', RED)

    arrow(ax, 0.3, 2.7, 1.5, 2.7, r'$u$')
    arrow(ax, 2.5, 2.7, 3.6, 2.7)  # to summing junction
    # summing junction
    summ = Circle((3.9, 2.7), 0.18, edgecolor='black', facecolor='white', lw=1.4)
    ax.add_patch(summ); ax.text(3.9, 2.7, '+', ha='center', va='center', fontsize=12)
    arrow(ax, 4.1, 2.7, 4.5, 2.7)
    arrow(ax, 5.9, 2.7, 7.5, 2.7, r'$\dot x \to x$', offset=(0, 0.3))
    arrow(ax, 8.5, 2.7, 11.0, 2.7, r'$y$')
    # feedback Ax
    ax.plot([6.5, 6.5], [2.2, 0.9], color='black', lw=1.2)
    arrow(ax, 6.5, 0.9, 5.7, 0.9)
    arrow(ax, 4.7, 0.9, 3.9, 0.9)
    ax.plot([3.9, 3.9], [0.9, 2.5], color='black', lw=1.2)
    ax.add_patch(FancyArrowPatch((3.9, 2.5), (3.9, 2.55),
                                 arrowstyle='->', mutation_scale=14, color='black'))

    ax.text(6.0, 4.4, r'$\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}, \quad '
                      r'\mathbf{y} = C\mathbf{x}$',
            fontsize=14, ha='center', fontweight='bold')

    # ---- (b) controllability / observability summary card ----
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    rows = [
        ['Test', 'Built from', 'Rank cond.'],
        ['Controllability', r'$[B,\,AB,\,\ldots,\,A^{n-1}B]$', r'rank $= n$'],
        ['Observability',   r'$[C;\,CA;\,\ldots;\,CA^{n-1}]$', r'rank $= n$'],
        ['Stability',       r'eigvals$(A)$',                   r'all Re $<0$'],
    ]
    tbl = ax.table(cellText=rows, loc='center', cellLoc='left',
                   colWidths=[0.32, 0.38, 0.30])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
    for j in range(3):
        tbl[(0, j)].set_facecolor('#ede9fe')
        tbl[(0, j)].set_text_props(weight='bold')

    # ---- (c) inverted pendulum-on-cart: LQR design demo (bottom row) ----
    # Linearised about upright: x = [pos, vel, theta, dtheta]
    M, m, l, g, b = 1.0, 0.2, 0.5, 9.81, 0.1
    A = np.array([
        [0, 1, 0, 0],
        [0, -b/M, -m*g/M, 0],
        [0, 0, 0, 1],
        [0, b/(M*l), (M+m)*g/(M*l), 0],
    ])
    B = np.array([[0], [1/M], [0], [-1/(M*l)]])

    Q = np.diag([10.0, 1.0, 100.0, 1.0])
    R = np.array([[1.0]])
    P = solve_continuous_are(A, B, Q, R)
    K_lqr = np.linalg.solve(R, B.T @ P)

    # Pole placement: pick desired poles
    K_pp = place_poles(A, B, [-1, -2, -3, -4]).gain_matrix

    def simulate(K, T=8.0, n=2000, x0=None):
        if x0 is None:
            x0 = [0.0, 0.0, 0.2, 0.0]   # 0.2 rad initial tilt
        t = np.linspace(0, T, n)

        def f(x, _):
            u = -(K @ x)
            return (A @ x + (B.flatten() * u)).tolist()
        return t, odeint(f, x0, t)

    ax_th = fig.add_subplot(gs[1, 0])
    ax_x  = fig.add_subplot(gs[1, 1])
    ax_u  = fig.add_subplot(gs[1, 2])

    for K, color, label in [(K_lqr, BLUE, 'LQR  $u=-Kx$'),
                            (K_pp,  RED,  'pole placement $\\{-1,-2,-3,-4\\}$')]:
        t, x = simulate(K)
        u = -(x @ K.T).flatten()
        ax_th.plot(t, np.rad2deg(x[:, 2]), color=color, lw=2.0, label=label)
        ax_x.plot(t,  x[:, 0],             color=color, lw=2.0, label=label)
        ax_u.plot(t,  u,                   color=color, lw=2.0, label=label)
    for ax_, ttl, ylbl in [(ax_th, 'Pendulum angle', r'$\theta$  (deg)'),
                           (ax_x,  'Cart position',  r'$x$  (m)'),
                           (ax_u,  'Control effort', r'$u$  (N)')]:
        ax_.axhline(0, color='black', lw=0.8, ls=':')
        ax_.set_xlabel('Time (s)'); ax_.set_ylabel(ylbl)
        ax_.set_title(ttl, fontsize=11, fontweight='bold')
        ax_.legend(loc='best', fontsize=8)

    fig.suptitle('State-Space Representation  +  LQR vs Pole-Placement on inverted pendulum',
                 fontsize=13, fontweight='bold', y=1.00)
    save(fig, 'fig4_state_space.png')


# ---------------------------------------------------------------------------
# fig5: feedback loop architecture
# ---------------------------------------------------------------------------
def fig5_feedback_loop():
    """Closed-loop feedback diagram + sensitivity to disturbances."""
    print('Building fig5_feedback_loop...')
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35)

    # Top: large block diagram
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')

    def block(x, y, w, h, label, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                             linewidth=2.0, edgecolor=color,
                             facecolor=color, alpha=0.18)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=11, fontweight='bold')

    def junction(x, y, sign='+'):
        c = Circle((x, y), 0.22, edgecolor='black', facecolor='white', lw=1.6)
        ax.add_patch(c)
        ax.text(x, y, sign, ha='center', va='center', fontsize=12,
                fontweight='bold')

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='->', mutation_scale=18,
                                     color='black', lw=1.4))

    # r -> sum -> Controller -> sum(disturb) -> Plant -> y
    #                 ^                                      |
    #                 |--------- Sensor <--------------------|
    block(2.7, 3.0, 1.8, 1.0, r'Controller $C(s)$', BLUE)
    block(7.0, 3.0, 1.8, 1.0, r'Plant $G(s)$',      GREEN)
    block(4.9, 0.6, 1.8, 1.0, r'Sensor $H(s)$',     PURPLE)

    # signals
    ax.text(0.3, 3.55, r'reference $r$', fontsize=11)
    arr(1.4, 3.5, 1.95, 3.5)
    junction(2.18, 3.5, '-+'); ax.text(2.18, 3.95, '+', fontsize=10)
    ax.text(1.92, 3.0, '-', fontsize=11, color=RED, fontweight='bold')
    arr(2.4, 3.5, 2.7, 3.5)

    # error e
    ax.text(2.55, 3.85, 'error  $e$', fontsize=10, style='italic', color=RED)

    arr(4.5, 3.5, 5.4, 3.5)
    ax.text(4.95, 3.85, 'control  $u$', fontsize=10, style='italic')
    junction(5.65, 3.5, '+')
    ax.text(5.65, 4.6, r'disturbance $d$', fontsize=10, color=RED)
    arr(5.65, 4.45, 5.65, 3.75)
    arr(5.9, 3.5, 7.0, 3.5)

    arr(8.8, 3.5, 11.5, 3.5)
    ax.text(11.7, 3.55, r'output $y$', fontsize=11, fontweight='bold')

    # noise on measurement
    ax.text(9.2, 1.2, r'noise $n$', fontsize=10, color=RED)
    junction(8.2, 1.1, '+')
    arr(8.8, 1.1, 9.0, 1.1)
    # feedback path: y -> sensor -> sum
    ax.plot([10.5, 10.5], [3.5, 1.1], color='black', lw=1.4)
    arr(10.5, 1.1, 6.7, 1.1)
    # sensor block
    arr(4.9, 1.1, 4.4, 1.1)
    # back to summing junction (connect to bottom of summer at 2.18, 3.5)
    ax.plot([4.4, 2.18], [1.1, 1.1], color='black', lw=1.4)
    ax.plot([2.18, 2.18], [1.1, 3.28], color='black', lw=1.4)
    ax.add_patch(FancyArrowPatch((2.18, 3.28), (2.18, 3.32),
                                 arrowstyle='->', mutation_scale=14, color='black'))

    ax.set_title('Closed-Loop Feedback Architecture  --  '
                 r'$T(s) = \dfrac{C\,G}{1 + C\,G\,H}$,  '
                 r'sensitivity $S = \dfrac{1}{1+CGH}$',
                 fontsize=13, fontweight='bold', loc='center', pad=20)

    # Bottom-left: open loop vs closed loop response to disturbance
    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(0, 30, 4000)
    dt = t[1] - t[0]
    # plant: y' = -y + u + d
    # disturbance d(t) = step at t=10
    d = np.where(t > 10, 0.5, 0.0)
    # Open loop u=1 (constant feedforward to track y=1)
    y_ol = np.zeros_like(t)
    y = 0.0
    for i, ti in enumerate(t):
        y += dt * (-y + 1.0 + d[i])
        y_ol[i] = y
    # Closed loop: PI controller
    Kp, Ki = 3.0, 2.0
    integ = 0.0
    y_cl = np.zeros_like(t)
    y = 0.0
    for i, ti in enumerate(t):
        e = 1.0 - y
        integ += e * dt
        u = Kp * e + Ki * integ
        y += dt * (-y + u + d[i])
        y_cl[i] = y

    ax.plot(t, y_ol, color=RED,  lw=2.0, label='open loop')
    ax.plot(t, y_cl, color=BLUE, lw=2.0, label='closed loop (PI)')
    ax.axhline(1.0, color='black', lw=0.8, ls=':')
    ax.axvline(10,  color='gray',  lw=0.8, ls=':')
    ax.text(10.2, 0.35, 'disturbance enters', fontsize=10, color='gray')
    ax.set_xlabel('Time'); ax.set_ylabel('Output  $y$')
    ax.set_title('Feedback rejects the disturbance; open loop drifts',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    # Bottom-right: sensitivity vs frequency for two designs
    ax = fig.add_subplot(gs[1, 1])
    w = np.logspace(-2, 2, 400)
    s = 1j * w
    # Plant 1/(s+1), controllers C1=1, C2=10, C3=10/s
    for C, color, label in [(1 + 0*s, BLUE, '$C=1$ (poor)'),
                            (10 + 0*s, GREEN, '$C=10$ (better)'),
                            (10 / s, RED,  '$C=10/s$ (integral)')]:
        L = C * 1.0 / (s + 1)
        S = 1.0 / (1.0 + L)
        ax.loglog(w, np.abs(S), color=color, lw=2.0, label=label)
    ax.axhline(1.0, color='black', lw=0.8, ls=':')
    ax.set_xlabel(r'$\omega$ (rad/s)'); ax.set_ylabel(r'$|S(j\omega)|$')
    ax.set_title('Sensitivity  $S = 1/(1+L)$:  smaller is better at low $\\omega$',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(1e-3, 3)

    fig.suptitle('Feedback:  why connecting output back to input is the central idea',
                 fontsize=13, fontweight='bold', y=1.00)
    save(fig, 'fig5_feedback_loop.png')


if __name__ == '__main__':
    fig1_pid_step_response()
    fig2_root_locus()
    fig3_bode_plot()
    fig4_state_space()
    fig5_feedback_loop()
    print('\nChapter 16 figures complete.')
