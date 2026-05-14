#!/usr/bin/env python3
"""Anim Art 05: Symplectic vs Euler — pendulum phase portrait comparison."""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

DARK_BG = "#0d1117"
C = {"blue": "#58a6ff", "orange": "#f78166", "green": "#7ee787", "purple": "#d2a8ff", "white": "#e6edf3"}

dt = 0.1
N_steps = 300
q0, p0 = 1.5, 0.0

def exact_pendulum(q0, p0, dt, N):
    qs, ps = [q0], [p0]
    q, p = q0, p0
    dt_fine = dt / 100
    for _ in range(N):
        for _ in range(100):
            p -= np.sin(q) * dt_fine
            q += p * dt_fine
        qs.append(q)
        ps.append(p)
    return np.array(qs), np.array(ps)

def euler_pendulum(q0, p0, dt, N):
    qs, ps = [q0], [p0]
    q, p = q0, p0
    for _ in range(N):
        q_new = q + p * dt
        p_new = p - np.sin(q) * dt
        q, p = q_new, p_new
        qs.append(q)
        ps.append(p)
    return np.array(qs), np.array(ps)

def leapfrog_pendulum(q0, p0, dt, N):
    qs, ps = [q0], [p0]
    q, p = q0, p0
    for _ in range(N):
        p_half = p - 0.5 * dt * np.sin(q)
        q = q + dt * p_half
        p = p_half - 0.5 * dt * np.sin(q)
        qs.append(q)
        ps.append(p)
    return np.array(qs), np.array(ps)

q_exact, p_exact = exact_pendulum(q0, p0, dt, N_steps)
q_euler, p_euler = euler_pendulum(q0, p0, dt, N_steps)
q_leap, p_leap = leapfrog_pendulum(q0, p0, dt, N_steps)

H_exact = 0.5 * p_exact**2 - np.cos(q_exact)
H_euler = 0.5 * p_euler**2 - np.cos(q_euler)
H_leap = 0.5 * p_leap**2 - np.cos(q_leap)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(DARK_BG)
for ax in (ax1, ax2):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=C["white"])
    for spine in ax.spines.values():
        spine.set_color(C["white"])

ax1.set_xlim(-4, 4)
ax1.set_ylim(-3, 3)
ax1.set_xlabel('$q$ (angle)', color=C["white"], fontsize=12)
ax1.set_ylabel('$p$ (momentum)', color=C["white"], fontsize=12)
ax1.set_title('Phase Portrait: Pendulum', color=C["white"], fontsize=13)

trail_exact, = ax1.plot([], [], color=C["blue"], lw=1.5, alpha=0.6)
trail_euler, = ax1.plot([], [], color=C["orange"], lw=1.5, alpha=0.6)
trail_leap, = ax1.plot([], [], color=C["green"], lw=1.5, alpha=0.6)
dot_exact, = ax1.plot([], [], 'o', color=C["blue"], ms=8, zorder=5, label='Exact')
dot_euler, = ax1.plot([], [], 'o', color=C["orange"], ms=8, zorder=5, label='Euler (spirals out)')
dot_leap, = ax1.plot([], [], 'o', color=C["green"], ms=8, zorder=5, label='Leapfrog (bounded)')
ax1.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=9, loc='upper left')

t_arr = np.arange(N_steps + 1) * dt
ax2.set_xlim(0, N_steps * dt)
ax2.set_ylim(min(H_exact.min(), H_leap.min()) - 0.3, H_euler.max() * 1.1 + 0.3)
ax2.set_xlabel('Time', color=C["white"], fontsize=12)
ax2.set_ylabel('$H(q,p)$', color=C["white"], fontsize=12)
ax2.set_title('Hamiltonian (Energy) Conservation', color=C["white"], fontsize=13)
ax2.axhline(H_exact[0], color=C["white"], ls=':', lw=1, alpha=0.3)

line_H_exact, = ax2.plot([], [], color=C["blue"], lw=2, label='Exact')
line_H_euler, = ax2.plot([], [], color=C["orange"], lw=2, label='Euler')
line_H_leap, = ax2.plot([], [], color=C["green"], lw=2, label='Leapfrog')
ax2.legend(facecolor=DARK_BG, edgecolor=C["white"], labelcolor=C["white"], fontsize=9)

step = 5

def init():
    for l in [trail_exact, trail_euler, trail_leap, dot_exact, dot_euler, dot_leap,
              line_H_exact, line_H_euler, line_H_leap]:
        l.set_data([], [])
    return (trail_exact, trail_euler, trail_leap, dot_exact, dot_euler, dot_leap,
            line_H_exact, line_H_euler, line_H_leap)

def animate(frame):
    i = frame * step
    trail_exact.set_data(q_exact[:i+1], p_exact[:i+1])
    trail_euler.set_data(q_euler[:i+1], p_euler[:i+1])
    trail_leap.set_data(q_leap[:i+1], p_leap[:i+1])
    dot_exact.set_data([q_exact[i]], [p_exact[i]])
    dot_euler.set_data([q_euler[i]], [p_euler[i]])
    dot_leap.set_data([q_leap[i]], [p_leap[i]])
    line_H_exact.set_data(t_arr[:i+1], H_exact[:i+1])
    line_H_euler.set_data(t_arr[:i+1], H_euler[:i+1])
    line_H_leap.set_data(t_arr[:i+1], H_leap[:i+1])
    return (trail_exact, trail_euler, trail_leap, dot_exact, dot_euler, dot_leap,
            line_H_exact, line_H_euler, line_H_leap)

n_frames = N_steps // step
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=100, blit=True)
plt.tight_layout()
ani.save('/tmp/pde-ml-figs/anim_symplectic_vs_euler.gif', writer=animation.PillowWriter(fps=10), dpi=100, savefig_kwargs={'facecolor': DARK_BG})
print("OK: anim_symplectic_vs_euler.gif")
