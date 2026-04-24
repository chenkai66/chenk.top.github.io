"""
Chapter 17: Physics and Engineering Applications.

Figures:
  fig1_pendulum_motion.png      -- small angle vs full nonlinear pendulum (time + phase)
  fig2_rlc_circuit.png          -- RLC underdamped/critical/overdamped + resonance curve
  fig3_planetary_orbits.png     -- Kepler orbits with varying eccentricity + Newton 1/r^2 vs 1/r^3
  fig4_structural_vibration.png -- 2-DOF building model: mode shapes + frequency response
  fig5_fluid_flow.png           -- Poiseuille profile + Reynolds-number transition curve
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from scipy.integrate import solve_ivp, odeint
from scipy.signal import lti, bode

plt.style.use('seaborn-v0_8-whitegrid')

BLUE   = '#2563eb'
PURPLE = '#7c3aed'
GREEN  = '#10b981'
RED    = '#ef4444'

OUT_DIRS = [
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/17-physics-engineering-applications',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/17-物理与工程应用',
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
# fig1: pendulum -- small angle vs full nonlinear
# ---------------------------------------------------------------------------
def fig1_pendulum_motion():
    print('Building fig1_pendulum_motion...')
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.35, wspace=0.30)

    g, L = 9.81, 1.0
    omega0 = np.sqrt(g / L)

    def full(t, s):
        th, om = s
        return [om, -(g/L) * np.sin(th)]
    def linear(t, s):
        th, om = s
        return [om, -(g/L) * th]

    T = 8.0
    t_eval = np.linspace(0, T, 4000)
    amplitudes = [0.1, 0.5, 1.0, 1.5, 2.5]   # rad
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(amplitudes)))

    # (a) time series: full vs linear at one large amplitude
    ax = fig.add_subplot(gs[0, 0])
    th0 = 1.5  # ~86 degrees
    sol_f = solve_ivp(full,   (0, T), [th0, 0], t_eval=t_eval, rtol=1e-10)
    sol_l = solve_ivp(linear, (0, T), [th0, 0], t_eval=t_eval, rtol=1e-10)
    ax.plot(sol_l.t, sol_l.y[0], color=BLUE, lw=2.0, ls='--',
            label=r'small-angle $\ddot\theta = -\omega_0^2\theta$')
    ax.plot(sol_f.t, sol_f.y[0], color=RED,  lw=2.0,
            label=r'full $\ddot\theta = -\omega_0^2\sin\theta$')
    ax.set_xlabel('Time (s)'); ax.set_ylabel(r'$\theta$ (rad)')
    ax.set_title(fr'Large initial angle $\theta_0={th0:.1f}$ rad: linear model loses phase',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # (b) period vs amplitude
    ax = fig.add_subplot(gs[0, 1])
    th0_arr = np.linspace(0.05, 3.0, 80)
    T_full, T_lin = [], []
    for th0 in th0_arr:
        # estimate period of full pendulum by integrating to first full cycle
        sol = solve_ivp(full, (0, 30), [th0, 0],
                        t_eval=np.linspace(0, 30, 30000), rtol=1e-10)
        # find zero crossings of theta from positive to negative side passing back
        th = sol.y[0]
        # period: second zero-up-crossing relative to start
        crossings = np.where((th[:-1] > 0) & (th[1:] <= 0))[0]
        if len(crossings) >= 1:
            # half period at first descending crossing; full period = 2 * t_half
            t_half = sol.t[crossings[0]]
            T_full.append(2 * t_half)
        else:
            T_full.append(np.nan)
        T_lin.append(2 * np.pi / omega0)
    ax.plot(th0_arr, T_lin, color=BLUE, lw=2.0, ls='--',
            label='small angle  $T = 2\\pi/\\omega_0$')
    ax.plot(th0_arr, T_full, color=RED, lw=2.0,
            label='full pendulum  (elliptic integral)')
    ax.axvline(np.pi, color='gray', lw=0.8, ls=':')
    ax.text(np.pi + 0.05, 4.5, r'$\theta_0=\pi$  (vertical)',
            color='gray', fontsize=9)
    ax.set_xlabel(r'amplitude  $\theta_0$  (rad)')
    ax.set_ylabel(r'period  $T$  (s)')
    ax.set_title('Pendulum period grows with amplitude (anisochronism)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(1.5, 7)

    # (c) phase portrait: librations + separatrix + rotations
    ax = fig.add_subplot(gs[1, :])
    TH = np.linspace(-2.2*np.pi, 2.2*np.pi, 600)
    OM = np.linspace(-7, 7, 400)
    THg, OMg = np.meshgrid(TH, OM)
    # Energy E = 0.5 omega^2 + omega0^2 (1 - cos theta)
    E = 0.5 * OMg**2 + omega0**2 * (1 - np.cos(THg))
    Esep = 2 * omega0**2
    levels = sorted(list(np.linspace(0.1, 1.6, 8) * Esep) + [Esep])
    cs = ax.contour(THg, OMg, E, levels=levels,
                    colors=['#888']*len(levels), linewidths=1.0, alpha=0.7)
    # separatrix in red
    ax.contour(THg, OMg, E, levels=[Esep], colors=[RED], linewidths=2.5)
    # equilibria
    for k in [-2, -1, 0, 1, 2]:
        ax.scatter(2*k*np.pi, 0, color=GREEN, s=80, zorder=5,
                   edgecolor='white', linewidth=1.5)  # stable
        ax.scatter((2*k+1)*np.pi, 0, color=PURPLE, s=80, marker='X', zorder=5,
                   edgecolor='white', linewidth=1.5)  # saddle
    ax.set_xlim(-2.2*np.pi, 2.2*np.pi); ax.set_ylim(-7, 7)
    ax.set_xlabel(r'$\theta$  (rad)'); ax.set_ylabel(r'$\dot\theta$  (rad/s)')
    ax.set_title('Pendulum phase portrait:  librations (closed) | separatrix (red) | rotations (open)',
                 fontsize=12, fontweight='bold')
    ax.text(0,    -6.2, 'libration', ha='center', fontsize=11, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(np.pi*1.4, 5.0, 'rotation (CCW)', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(-np.pi*1.4, -5.5, 'rotation (CW)', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('Pendulum:  the canonical nonlinear ODE  '
                 r'$\ddot\theta + (g/L)\sin\theta = 0$',
                 fontsize=13, fontweight='bold', y=1.00)
    save(fig, 'fig1_pendulum_motion.png')


# ---------------------------------------------------------------------------
# fig2: RLC circuit
# ---------------------------------------------------------------------------
def fig2_rlc_circuit():
    print('Building fig2_rlc_circuit...')
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    # Three damping cases for L=1, C=1, R varies
    L, C = 1.0, 1.0
    omega_n = 1 / np.sqrt(L * C)

    # (a) circuit schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    # source
    ax.add_patch(Circle((1.0, 3), 0.5, fill=False, lw=2, edgecolor='black'))
    ax.text(1.0, 3, 'V', fontsize=14, ha='center', va='center', fontweight='bold')
    # wires
    ax.plot([1.5, 3.0], [3, 3], 'k-', lw=2)
    ax.plot([1.0, 1.0], [3.5, 5.0], 'k-', lw=2)
    ax.plot([1.0, 9.0], [5.0, 5.0], 'k-', lw=2)
    ax.plot([9.0, 9.0], [5.0, 3.0], 'k-', lw=2)
    # R
    ax.add_patch(Rectangle((3.0, 2.7), 1.4, 0.6, fill=False,
                           edgecolor=BLUE, lw=2))
    ax.text(3.7, 3.0, 'R', fontsize=12, ha='center', va='center',
            fontweight='bold', color=BLUE)
    # L (inductor: zigzag)
    ax.plot([4.4, 5.0], [3, 3], 'k-', lw=2)
    for i in range(4):
        ax.add_patch(Circle((5.1 + 0.2*i, 3.05), 0.12, fill=False,
                            lw=2, edgecolor=GREEN))
    ax.plot([5.9, 6.5], [3, 3], 'k-', lw=2)
    ax.text(5.5, 3.55, 'L', fontsize=12, ha='center', color=GREEN, fontweight='bold')
    # C (parallel plates)
    ax.plot([6.5, 7.4], [3, 3], 'k-', lw=2)
    ax.plot([7.4, 7.4], [2.5, 3.5], color=RED, lw=2)
    ax.plot([7.7, 7.7], [2.5, 3.5], color=RED, lw=2)
    ax.plot([7.7, 9.0], [3, 3], 'k-', lw=2)
    ax.text(7.55, 3.85, 'C', fontsize=12, ha='center', color=RED, fontweight='bold')

    ax.text(5.0, 0.8,
            r'$L\ddot q + R\dot q + q/C = V(t)$' + '\n'
            r'$\omega_n = 1/\sqrt{LC}, \quad \zeta = \frac{R}{2}\sqrt{C/L}$',
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f3f4f6',
                      edgecolor='gray'))
    ax.set_title('Series RLC circuit', fontsize=12, fontweight='bold')

    # (b) step responses for three damping regimes
    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 25, 4000)
    cases = [
        (0.2, 'underdamped  $\\zeta=0.1$',  RED),
        (2.0, 'critical    $\\zeta=1.0$',   GREEN),
        (5.0, 'overdamped  $\\zeta=2.5$',   BLUE),
    ]
    for R, label, color in cases:
        zeta = R/2 * np.sqrt(C/L)
        sys = lti([1], [L, R, 1/C])  # output is q; step input V=1
        from scipy.signal import step as step_resp
        t_out, q = step_resp(sys, T=t)
        ax.plot(t_out, q, color=color, lw=2.0, label=label)
    ax.axhline(1.0, color='black', lw=0.8, ls=':')
    ax.set_xlabel('Time'); ax.set_ylabel('Charge $q(t)$')
    ax.set_title('Step response by damping ratio $\\zeta$',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    # (c) frequency response (resonance curve)
    ax = fig.add_subplot(gs[1, 0])
    w = np.logspace(-1, 1, 400)
    for R, label, color in cases:
        sys = lti([1], [L, R, 1/C])
        _, mag, _ = bode(sys, w=w)
        ax.semilogx(w, 10**(mag/20), color=color, lw=2.0, label=label)
    ax.axvline(omega_n, color='black', lw=0.8, ls='--', alpha=0.6)
    ax.text(omega_n*1.05, 6, r'$\omega_n=1$ rad/s', fontsize=9)
    ax.set_xlabel(r'$\omega$  (rad/s)'); ax.set_ylabel(r'$|H(j\omega)|$')
    ax.set_title('Resonance: low damping -> sharp peak (high Q)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 7)

    # (d) sweep over R: amplitude, phase
    ax = fig.add_subplot(gs[1, 1])
    Rvals = np.linspace(0.05, 4.0, 100)
    Q = 1.0 / Rvals * np.sqrt(L/C)
    bw = Rvals / L  # 3-dB bandwidth approx
    ax.plot(Rvals, Q,  color=BLUE,  lw=2.0, label=r'$Q$  (peak sharpness)')
    ax.plot(Rvals, bw, color=RED,   lw=2.0, label=r'bandwidth $\Delta\omega$')
    ax.set_xlabel('Resistance R')
    ax.set_title('Damping vs Q-factor and bandwidth',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 12)

    fig.suptitle('RLC Circuit:  electrical analog of a damped oscillator',
                 fontsize=13, fontweight='bold', y=1.00)
    save(fig, 'fig2_rlc_circuit.png')


# ---------------------------------------------------------------------------
# fig3: planetary orbits
# ---------------------------------------------------------------------------
def fig3_planetary_orbits():
    print('Building fig3_planetary_orbits...')
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    GM = 1.0

    def kepler(t, s, n_pow=2.0):
        x, y, vx, vy = s
        r = np.sqrt(x*x + y*y)
        ax_ = -GM * x / r ** (n_pow + 1)
        ay_ = -GM * y / r ** (n_pow + 1)
        return [vx, vy, ax_, ay_]

    # (a) Orbits with varying eccentricity
    ax = fig.add_subplot(gs[:, 0])
    cases = [
        (1.00, 'circle  e=0',     GREEN),
        (0.85, 'ellipse e=0.18',  BLUE),
        (0.70, 'ellipse e=0.51',  PURPLE),
        (0.55, 'ellipse e=0.85',  RED),
    ]
    for v0, label, color in cases:
        sol = solve_ivp(kepler, (0, 18), [1.0, 0.0, 0.0, v0],
                        t_eval=np.linspace(0, 18, 6000),
                        rtol=1e-10, atol=1e-12)
        ax.plot(sol.y[0], sol.y[1], color=color, lw=1.6, label=label)
    ax.scatter([0], [0], color='gold', s=300, edgecolor='orange',
               linewidth=1.5, zorder=10, label='Sun')
    ax.set_xlabel('x (AU)'); ax.set_ylabel('y (AU)')
    ax.set_title("Kepler orbits  --  initial position $(1,0)$, varying $v_y$",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 1.6); ax.set_ylim(-1.7, 1.7)

    # (b) Inverse-square (closed orbits) vs inverse-cube (open spirals)
    ax = fig.add_subplot(gs[0, 1])
    sol_2 = solve_ivp(kepler, (0, 25), [1.0, 0, 0, 0.7],
                      t_eval=np.linspace(0, 25, 6000), args=(2.0,),
                      rtol=1e-10, atol=1e-12)
    sol_3 = solve_ivp(kepler, (0, 5),  [1.0, 0, 0, 0.7],
                      t_eval=np.linspace(0, 5, 6000), args=(3.0,),
                      rtol=1e-10, atol=1e-12)
    ax.plot(sol_2.y[0], sol_2.y[1], color=BLUE, lw=1.4,
            label=r'$1/r^2$ : closed ellipse (Bertrand)')
    ax.plot(sol_3.y[0], sol_3.y[1], color=RED, lw=1.4,
            label=r'$1/r^3$ : spiral collapse')
    ax.scatter([0], [0], color='gold', s=200, edgecolor='orange',
               linewidth=1.5, zorder=10)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Why our universe is special (Bertrand theorem)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_aspect('equal')

    # (c) Energy and angular momentum conservation (numerical check)
    ax = fig.add_subplot(gs[1, 1])
    sol = solve_ivp(kepler, (0, 30), [1.0, 0, 0, 0.7],
                    t_eval=np.linspace(0, 30, 6000),
                    rtol=1e-10, atol=1e-12)
    x, y, vx, vy = sol.y
    r = np.sqrt(x*x + y*y)
    E = 0.5 * (vx*vx + vy*vy) - GM/r
    Lang = x * vy - y * vx
    ax.plot(sol.t, (E - E[0]) / abs(E[0]), color=BLUE, lw=1.6,
            label=r'energy $\Delta E / |E_0|$')
    ax.plot(sol.t, (Lang - Lang[0]) / abs(Lang[0]), color=RED, lw=1.6,
            label=r'ang.\ momentum $\Delta L / |L_0|$')
    ax.axhline(0, color='black', lw=0.8, ls=':')
    ax.set_xlabel('Time'); ax.set_ylabel('relative drift')
    ax.set_title('Conservation laws hold to numerical precision',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    fig.suptitle('Planetary Orbits:  Newton + ODE = Kepler\'s laws',
                 fontsize=13, fontweight='bold', y=1.00)
    save(fig, 'fig3_planetary_orbits.png')


# ---------------------------------------------------------------------------
# fig4: structural vibration (2-DOF building)
# ---------------------------------------------------------------------------
def fig4_structural_vibration():
    print('Building fig4_structural_vibration...')
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          hspace=0.40, wspace=0.40)

    # 2-storey shear building
    m1, m2 = 1.0, 1.0
    k1, k2 = 100.0, 100.0
    M = np.array([[m1, 0], [0, m2]])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    eigvals, eigvecs = np.linalg.eig(np.linalg.solve(M, K))
    order = np.argsort(eigvals.real)
    eigvals = eigvals.real[order]
    eigvecs = eigvecs[:, order]
    omegas = np.sqrt(eigvals)

    # (a) building schematic + mode shapes
    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.5, 5.5); ax.axis('off')
    # ground
    ax.plot([-1.5, 1.5], [0, 0], 'k-', lw=2)
    for x in np.linspace(-1.4, 1.4, 8):
        ax.plot([x, x - 0.15], [0, -0.25], 'k-', lw=1.0)
    # columns + masses
    storeys = [(2.0, m1), (4.0, m2)]
    # mode 1 deformed (in-phase)
    s1 = eigvecs[:, 0] / max(abs(eigvecs[:, 0])) * 0.6
    s2 = eigvecs[:, 1] / max(abs(eigvecs[:, 1])) * 0.6
    # rest position columns
    for off, color, label, signs in [(-0.7, GREEN, f'mode 1  $\\omega={omegas[0]:.2f}$', s1),
                                     (0.7, RED,  f'mode 2  $\\omega={omegas[1]:.2f}$', s2)]:
        for i, (h, m) in enumerate(storeys):
            x_top = off + signs[i] * 0.4
            x_bot = off + (signs[i-1] * 0.4 if i > 0 else 0)
            ax.plot([x_bot, x_top], [storeys[i-1][0] if i > 0 else 0, h],
                    color=color, lw=2.5)
            # mass block
            box = Rectangle((x_top - 0.25, h - 0.15), 0.5, 0.3,
                            facecolor=color, alpha=0.5, edgecolor='black', lw=1.4)
            ax.add_patch(box)
        ax.text(off, 5.0, label, ha='center', fontsize=10, fontweight='bold',
                color=color)

    ax.set_title('2-storey building:  mode shapes',
                 fontsize=12, fontweight='bold')

    # (b) free-vibration time response (initial 2nd-storey kick)
    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 6, 4000)
    # Use modal superposition
    x0 = np.array([0, 1.0])  # initial displacement
    v0 = np.array([0, 0])
    Phi = eigvecs
    # modal initial conditions
    q0 = np.linalg.solve(Phi, x0)
    qd0 = np.linalg.solve(Phi, v0)
    # mode responses
    q = (q0[None, :] * np.cos(omegas[None, :] * t[:, None]) +
         qd0[None, :] / omegas[None, :] * np.sin(omegas[None, :] * t[:, None]))
    x = q @ Phi.T
    ax.plot(t, x[:, 0], color=BLUE, lw=1.6, label='floor 1')
    ax.plot(t, x[:, 1], color=RED,  lw=1.6, label='floor 2')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('displacement')
    ax.set_title('Free vibration after initial kick',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # (c) frequency response with damping
    ax = fig.add_subplot(gs[1, 1])
    C_mat = 0.05 * (M + K * 0.005)  # small Rayleigh damping
    w_arr = np.linspace(0.5, 25, 500)
    H1 = []; H2 = []
    for w in w_arr:
        Z = -w*w * M + 1j * w * C_mat + K
        F = np.array([0, 1.0])
        X = np.linalg.solve(Z, F)
        H1.append(abs(X[0])); H2.append(abs(X[1]))
    ax.plot(w_arr, H1, color=BLUE, lw=2.0, label='floor 1')
    ax.plot(w_arr, H2, color=RED,  lw=2.0, label='floor 2')
    for o in omegas:
        ax.axvline(o, color='gray', lw=0.7, ls='--')
        ax.text(o, ax.get_ylim()[1]*0.0 + 0.01,
                fr'$\omega={o:.1f}$', rotation=90, fontsize=8,
                color='gray', va='bottom')
    ax.set_xlabel(r'forcing $\omega$ (rad/s)')
    ax.set_ylabel('amplitude')
    ax.set_title('Two resonance peaks at modal frequencies',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # (d) Tacoma-style growth: undamped resonance
    ax = fig.add_subplot(gs[0, 2])
    omega_drive = omegas[1]
    t = np.linspace(0, 60, 6000)
    # SDOF resonance for visual: y'' + small*y' + omega^2 y = sin(omega t)
    sig = 0.05
    sol = odeint(lambda y, ti: [y[1], -sig*y[1] - omega_drive**2*y[0] + np.sin(omega_drive*ti)],
                 [0, 0], t)
    ax.plot(t, sol[:, 0], color=PURPLE, lw=1.0)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('amplitude')
    ax.set_title('Resonant forcing -> growing amplitude\n(Tacoma Narrows logic)',
                 fontsize=11, fontweight='bold')

    # (e) tuned mass damper effect
    ax = fig.add_subplot(gs[1, 2])
    # SDOF with TMD: main mass M=1, k=100; TMD m=0.05, k=5
    Mt = np.diag([1.0, 0.05])
    Kt = np.array([[100 + 5, -5], [-5, 5]])
    H_no, H_with = [], []
    w_arr = np.linspace(5, 16, 500)
    for w in w_arr:
        # without TMD
        Z = np.array([[-w*w + 100]])
        H_no.append(1 / abs(Z[0, 0]))
        # with TMD
        Z2 = -w*w * Mt + 0.5j * w * np.eye(2) + Kt
        F = np.array([1.0, 0.0])
        X = np.linalg.solve(Z2, F)
        H_with.append(abs(X[0]))
    ax.plot(w_arr, H_no,   color=RED,  lw=2.0, label='without TMD')
    ax.plot(w_arr, H_with, color=GREEN, lw=2.0, label='with TMD')
    ax.set_xlabel(r'$\omega$  (rad/s)'); ax.set_ylabel('main-mass amplitude')
    ax.set_title('Tuned Mass Damper splits one resonance into two smaller ones',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Structural Vibration:  modal analysis of multi-DOF systems',
                 fontsize=13, fontweight='bold', y=1.00)
    save(fig, 'fig4_structural_vibration.png')


# ---------------------------------------------------------------------------
# fig5: fluid flow
# ---------------------------------------------------------------------------
def fig5_fluid_flow():
    print('Building fig5_fluid_flow...')
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    # (a) Poiseuille velocity profile (parabolic)
    ax = fig.add_subplot(gs[0, 0])
    R = 1.0
    r = np.linspace(-R, R, 200)
    for u_max, color in [(1.0, BLUE), (0.7, GREEN), (0.4, RED)]:
        u = u_max * (1 - (r/R)**2)
        # plot as horizontal flow profile
        ax.fill_betweenx(r, 0, u, color=color, alpha=0.25)
        ax.plot(u, r, color=color, lw=2.0,
                label=fr'$u_\max={u_max}$')
        # arrows
        for ri in np.linspace(-0.85, 0.85, 9):
            ui = u_max * (1 - (ri/R)**2)
            ax.annotate('', xy=(ui, ri), xytext=(0, ri),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.5, lw=1.0))
    ax.axhline( R, color='black', lw=2.5)
    ax.axhline(-R, color='black', lw=2.5)
    ax.set_xlabel('velocity  $u$'); ax.set_ylabel('radius  $r$')
    ax.set_title('Poiseuille flow:  parabolic velocity profile',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-0.05, 1.2)

    # (b) Pressure-flow relation (linear regime)
    ax = fig.add_subplot(gs[0, 1])
    dp = np.linspace(0, 10, 100)
    Q_lam = dp * 1.0  # Q = (pi R^4)/(8 mu L) * dp -- linear
    # Show transition: at high Re, becomes Q ~ dp^0.5 (turbulent loss)
    Q_turb = np.where(dp > 4,
                      1.0 * 4 + 1.4 * np.sqrt(np.maximum(dp - 4, 0.0)),
                      dp * 1.0)
    ax.plot(dp, Q_lam, color=BLUE, lw=2.0,
            label=r'laminar  $Q \propto \Delta p$')
    ax.plot(dp, Q_turb, color=RED, lw=2.0, ls='--',
            label=r'turbulent  $Q \propto \sqrt{\Delta p}$')
    ax.axvline(4, color='gray', lw=0.8, ls=':')
    ax.text(4.1, 9, 'transition  Re$\\approx$2300',
            fontsize=9, color='gray')
    ax.set_xlabel(r'$\Delta p$'); ax.set_ylabel('flow rate Q')
    ax.set_title(r'Hagen-Poiseuille linearity breaks at high Re',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    # (c) damped pendulum-of-fluid: spring-mass-dashpot analogue
    # Show vortex shedding frequency vs Re
    ax = fig.add_subplot(gs[1, 0])
    Re = np.logspace(1.5, 7, 300)
    St = np.where(Re < 200, 0.21 * (1 - 21.2/np.maximum(Re, 30)),
                  np.where(Re < 1e5, 0.21,
                           0.21 + 0.05 * np.log10(Re/1e5)))
    ax.semilogx(Re, St, color=PURPLE, lw=2.0)
    ax.axhline(0.21, color='gray', lw=0.8, ls=':')
    ax.fill_betweenx([0, 0.5], 1e1, 200,    color=GREEN, alpha=0.10,
                     label='laminar')
    ax.fill_betweenx([0, 0.5], 200, 3e5,    color=BLUE,  alpha=0.10,
                     label='subcritical')
    ax.fill_betweenx([0, 0.5], 3e5, 1e7,    color=RED,   alpha=0.10,
                     label='supercritical / turbulent')
    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel(r'Strouhal number St $= f D / U$')
    ax.set_title('Vortex shedding behind a cylinder',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.4)
    ax.legend(loc='upper right', fontsize=8)

    # (d) Time evolution: drag on a settling sphere (Stokes vs Newton)
    ax = fig.add_subplot(gs[1, 1])
    g = 9.81
    rho_p, rho_f = 2500, 1000
    mu = 1e-3
    R_arr = [1e-4, 1e-3, 5e-3]
    t = np.linspace(0, 5, 4000)
    for R, color in zip(R_arr, [BLUE, GREEN, RED]):
        m = (4/3)*np.pi*R**3 * rho_p
        Vp = (4/3)*np.pi*R**3
        Fb = (rho_p - rho_f) * Vp * g
        # Stokes drag F_d = 6 pi mu R v
        c1 = 6 * np.pi * mu * R
        v_term = Fb / c1
        v = v_term * (1 - np.exp(-c1/m * t))
        ax.plot(t, v, color=color, lw=2.0,
                label=fr'R={R*1000:.1f} mm,  $v_t={v_term:.2f}$ m/s')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('velocity (m/s)')
    ax.set_title('Falling sphere reaches terminal velocity (Stokes regime)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Fluid Mechanics:  ODEs from steady flow to drag and vortex shedding',
                 fontsize=13, fontweight='bold', y=1.00)
    save(fig, 'fig5_fluid_flow.png')


if __name__ == '__main__':
    fig1_pendulum_motion()
    fig2_rlc_circuit()
    fig3_planetary_orbits()
    fig4_structural_vibration()
    fig5_fluid_flow()
    print('\nChapter 17 figures complete.')
