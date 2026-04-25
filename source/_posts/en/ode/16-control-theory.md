---
title: "Ordinary Differential Equations (16): Fundamentals of Control Theory"
date: 2024-03-12 09:00:00
tags:
  - Ordinary Differential Equations
  - Control Theory
  - PID Control
  - State Space
  - Feedback Systems
  - Stability
  - Python
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 16
  total: 18
lang: en
mathjax: true
description: "Learn how differential equations power control systems. Cover transfer functions, PID controllers, root locus, Bode plots, state-space methods, controllability, observability, pole placement, LQR optimal control, and observer design with Python examples."
disableNunjucks: true
series_order: 16
---

**When you steer a car you constantly correct based on lane position. A thermostat compares room temperature with the setpoint and adjusts a heater. A rocket gimbal nudges its thrust vector to keep the booster vertical.** Strip away the hardware and the same idea remains: *measure, compare, act*. Control theory is the mathematics of that loop -- and its native language is the ordinary differential equation.

This chapter shows how the entire ODE toolkit -- Laplace transforms (Ch 4), linear systems (Ch 6), eigenvalue stability (Ch 7), nonlinear stability (Ch 8) -- collapses into a single unified discipline whose job is no longer to *describe* dynamics, but to *design* them.

## What You Will Learn

- Open-loop vs. closed-loop control: why feedback is the central idea
- Transfer functions and how they convert ODEs into algebra
- PID controllers and Ziegler-Nichols tuning
- Root locus: how closed-loop poles move with controller gain
- Bode plots, gain margin, phase margin
- State-space representation of MIMO systems
- Controllability, observability, and the rank tests
- Pole placement and LQR optimal control on an inverted pendulum
- Luenberger observers and the separation principle

## Prerequisites

- Chapter 4 -- Laplace transforms (where transfer functions come from)
- Chapter 6 -- Linear systems and matrix exponential
- Chapter 7 -- Eigenvalue criteria for linear stability

---

## 1. Open Loop vs Closed Loop

Suppose we want a heater to bring a room from 15 deg C up to 22 deg C.

**Open loop**: send a fixed power $u(t)$ for a fixed time. If the window is open, or the outside is colder than expected, you arrive at the wrong temperature -- with no recourse.

**Closed loop**: measure the temperature, compute the *error* $e(t) = r(t) - y(t)$ between the reference $r$ and the measured output $y$, and choose $u(t)$ as a function of $e$. The system *self-corrects* against modelling errors and disturbances.

![Closed-loop feedback architecture with disturbance and noise paths.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/16-control-theory/fig5_feedback_loop.png)
*Closed-loop architecture. The reference $r$ enters the summing junction; the controller $C(s)$ acts on the error $e = r - y_{\text{measured}}$; the plant $G(s)$ produces the output $y$, which is measured (with noise $n$) and fed back. Feedback rejects the disturbance $d$ that breaks the open-loop response.*

Two consequences of feedback we will derive precisely:

- **Disturbance rejection** -- a step disturbance no longer drives a steady-state error to infinity.
- **Sensitivity reduction** -- relative variations in the plant $G$ affect the closed-loop transfer function $T = CG/(1+CGH)$ by a factor $S = 1/(1+CGH)$, the *sensitivity function*.

---

## 2. Transfer Functions

For a linear time-invariant (LTI) system with input $u(t)$ and output $y(t)$, the transfer function

$$
G(s) \;=\; \frac{Y(s)}{U(s)}
$$

is the Laplace ratio at zero initial conditions. ODE differentiation becomes multiplication by $s$, integration becomes division -- algebra replaces calculus.

### First-order plant

$$
\tau\dot y + y = K u \quad\Longleftrightarrow\quad G(s) = \frac{K}{\tau s + 1}.
$$

Step response: $y(t) = K(1 - e^{-t/\tau})$. Time constant $\tau$ governs how quickly the system tracks a setpoint change.

### Second-order plant

$$
\ddot y + 2\zeta\omega_n \dot y + \omega_n^2 y = K\omega_n^2 u
\;\;\Longleftrightarrow\;\;
G(s) = \frac{K\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}.
$$

The damping ratio $\zeta$ controls the *qualitative* shape of the response:

| $\zeta$ | Pole structure | Step response |
|---|---|---|
| $0$ | $\pm j\omega_n$ | undamped sine |
| $0 < \zeta < 1$ | complex conjugate, Re $<0$ | underdamped (overshoot + ringing) |
| $1$ | real double pole | critically damped (fastest non-overshoot) |
| $> 1$ | two real poles | overdamped (sluggish) |

These four cases recur everywhere -- circuits, mechanical systems, biology -- because every linear system *near a stable equilibrium* looks like this.

---

## 3. PID Controllers

The most widely deployed controller in industry, by an enormous margin. The control law:

$$
u(t) \;=\; K_p\,e(t) \;+\; K_i\!\int_0^t e(\tau)\,d\tau \;+\; K_d\,\dot e(t),
\qquad e = r - y.
$$

Each term has a clean physical role:

- **Proportional ($K_p$)** -- a "spring" pulling the output toward the setpoint. Fast but cannot eliminate steady-state error against a constant disturbance.
- **Integral ($K_i$)** -- a "memory" that builds while the error is non-zero. Drives steady-state error to *exactly* zero, but adds a pole at the origin (slows response and risks overshoot).
- **Derivative ($K_d$)** -- a "damper" that responds to *predicted* future error. Improves transient behaviour but amplifies high-frequency measurement noise.

![PID step responses on a 2nd-order plant, plus the effect of raising the proportional gain.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/16-control-theory/fig1_pid_step_response.png)
*Left: closed-loop step response with progressively richer controllers (no control -> P -> PI -> PD -> PID). Without control, the underdamped plant rings forever and never tracks; PID nails the setpoint with no steady-state error and minimal overshoot. Right: the classic P-only trade-off -- raising $K_p$ speeds the response and shrinks steady-state error, but ringing eventually dominates.*

### Ziegler-Nichols tuning (a starting point)

A pragmatic recipe that requires no model:

1. Set $K_i = K_d = 0$. Increase $K_p$ until the closed-loop response sustains an oscillation -- record the **ultimate gain** $K_u$ and the **oscillation period** $T_u$.
2. PID gains:

| Controller | $K_p$ | $K_i$ | $K_d$ |
|---|---|---|---|
| P | $0.5\,K_u$ | -- | -- |
| PI | $0.45\,K_u$ | $1.2\,K_p / T_u$ | -- |
| PID | $0.6\,K_u$ | $2\,K_p / T_u$ | $K_p T_u / 8$ |

```python
import numpy as np

class PIDController:
    """Discrete-time PID with optional derivative-on-measurement and
    anti-windup clamping. Use compute(error, dt) each sampling period."""
    def __init__(self, Kp, Ki, Kd, u_min=-np.inf, u_max=np.inf):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_min, self.u_max = u_min, u_max
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        deriv = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        u = self.Kp*error + self.Ki*self.integral + self.Kd*deriv
        # anti-windup: only integrate when not saturated
        if u > self.u_max:
            self.integral -= error * dt
            u = self.u_max
        elif u < self.u_min:
            self.integral -= error * dt
            u = self.u_min
        return u
```

---

## 4. Root Locus -- where the closed-loop poles go

For a unity-feedback loop with open-loop transfer function $K\,L(s)$, the closed-loop characteristic equation is

$$
1 + K\,L(s) = 0.
$$

The **root locus** plots the closed-loop pole positions in the complex plane as the gain $K$ sweeps from $0$ to $\infty$. It tells you, *visually*, the trade-off between speed (poles further left) and damping (poles closer to the real axis).

![Root locus of $K/(s(s+2)(s+5))$ and the corresponding step responses for three gains.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/16-control-theory/fig2_root_locus.png)
*Left: three branches of the locus depart from the open-loop poles at $0, -2, -5$. Two of them eventually cross into the right half-plane near $K \approx 70$ -- the gain at which the closed loop loses stability. Right: closed-loop step responses as $K$ rises through that boundary -- damped, lightly damped, and finally an undamped oscillation right at $K = 70$.*

Two facts to memorize:

- The locus has $n_p - n_z$ asymptotes departing the centroid $\sigma_a = (\sum p_i - \sum z_i)/(n_p - n_z)$ at angles $(2k+1)\pi/(n_p - n_z)$.
- Where the locus crosses the imaginary axis, the loop has poles $\pm j\omega_c$ -- exactly the *neutral* stability point. Routh-Hurwitz delivers $K_{\text{crit}}$ algebraically.

---

## 5. Bode Plots and Stability Margins

The frequency response $L(j\omega)$ -- the same transfer function evaluated on the imaginary axis -- has two natural readouts: magnitude (in dB) and phase (in degrees) versus $\omega$ on a log scale.

![Bode magnitude and phase, Nyquist contour, and stability summary for $L(s)=100/(s(s+1)(s+10))$.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/16-control-theory/fig3_bode_plot.png)
*The **gain crossover** $\omega_g$ is where $|L|$ crosses unity (0 dB); the **phase crossover** $\omega_p$ is where the phase crosses $-180^\circ$. The two robustness numbers are:*
- *Gain margin (GM): how much we can scale $L$ before instability -- read off the magnitude at $\omega_p$.*
- *Phase margin (PM): how much extra phase lag we can tolerate at $\omega_g$ before the system loses stability.*

Rules of thumb: **GM > 6 dB** (a factor of 2) and **PM > 30 deg** (preferably > 45) for a robust design. The Nyquist contour on the right makes the geometric picture explicit: stability requires that the contour does *not* encircle the critical point $-1 + 0j$.

---

## 6. State-Space Representation

For multi-input multi-output (MIMO) systems we drop the transfer-function fiction and write the dynamics directly:

$$
\dot{\mathbf x} = A\mathbf x + B\mathbf u, \qquad
\mathbf y = C\mathbf x + D\mathbf u.
$$

Here $\mathbf x \in \mathbb R^n$ is the **state vector** -- enough information to predict the future given $\mathbf u(t)$. $A$ is the dynamics matrix, $B$ couples the inputs in, $C$ extracts the measured outputs, and $D$ is direct feedthrough (often zero).

![State-space block diagram, the controllability/observability tests, and an LQR vs pole-placement comparison on the inverted pendulum.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/16-control-theory/fig4_state_space.png)
*Top: the universal block diagram and the two key rank tests. Bottom: balancing an inverted pendulum on a cart from a $0.2$-rad initial tilt. LQR (blue) chooses the gains by minimising a cost; pole placement (red) chooses gains to put the closed-loop eigenvalues at $\{-1,-2,-3,-4\}$. Both stabilise -- but LQR uses much less control effort.*

### Stability, controllability, observability

Three rank conditions tell you everything about whether you can design a controller at all:

| Test | Built from | Holds iff |
|---|---|---|
| Stability | eigenvalues of $A$ | all have Re < 0 |
| Controllability | $\mathcal C = [B,\; AB,\; \ldots,\; A^{n-1}B]$ | rank $\mathcal C = n$ |
| Observability | $\mathcal O = [C;\; CA;\; \ldots;\; CA^{n-1}]$ | rank $\mathcal O = n$ |

Controllability says you can drive the state to *any* target with a suitable input. Observability says you can deduce the full state from the output history. They are dual properties under $A \leftrightarrow A^T,\,B \leftrightarrow C^T$.

---

## 7. Pole Placement and LQR

If $(A, B)$ is controllable, the state-feedback law $\mathbf u = -K\mathbf x$ produces closed-loop dynamics $\dot{\mathbf x} = (A - BK)\mathbf x$. We can choose $K$ to put the eigenvalues of $A - BK$ wherever we like.

```python
import numpy as np
from scipy.signal import place_poles

A = np.array([[0, 1, 0, 0],
              [0, -0.1, -1.96, 0],
              [0, 0, 0, 1],
              [0, 0.2, 23.5, 0]])  # inverted pendulum, linearised
B = np.array([[0], [1], [0], [-2]])

K_pp = place_poles(A, B, [-1, -2, -3, -4]).gain_matrix
print('Pole-placement gain K =', K_pp)
print('Closed-loop eigenvalues =', np.linalg.eigvals(A - B @ K_pp))
```

**LQR** (linear quadratic regulator) instead chooses $K$ to *minimise* a quadratic cost

$$
J \;=\; \int_0^\infty \bigl(\mathbf x^T Q \mathbf x + \mathbf u^T R \mathbf u\bigr)\,dt,
$$

trading state error against control effort. The solution comes from the **algebraic Riccati equation**

$$
A^T P + P A - P B R^{-1} B^T P + Q = 0, \qquad K = R^{-1} B^T P.
$$

```python
from scipy.linalg import solve_continuous_are

Q = np.diag([10, 1, 100, 1])      # penalise position and angle heavily
R = np.array([[1.0]])              # modest control penalty
P = solve_continuous_are(A, B, Q, R)
K_lqr = np.linalg.solve(R, B.T @ P)
```

Compared to pole placement, LQR almost always uses less actuator effort to achieve the same disturbance rejection -- and its gains are guaranteed to be stabilising whenever $(A, B)$ is stabilisable and $(A, \sqrt{Q})$ is detectable.

---

## 8. Observers and the Separation Principle

State feedback presupposes you can *measure* every component of $\mathbf x$. In reality you only see $\mathbf y$. A **Luenberger observer** estimates the unmeasured state from the measured output:

$$
\dot{\hat{\mathbf x}} \;=\; A\hat{\mathbf x} + B\mathbf u + L\bigl(\mathbf y - C\hat{\mathbf x}\bigr).
$$

The estimation error $\tilde{\mathbf x} = \mathbf x - \hat{\mathbf x}$ obeys $\dot{\tilde{\mathbf x}} = (A - LC)\tilde{\mathbf x}$. If $(A, C)$ is observable, we can place the eigenvalues of $A - LC$ wherever we like -- pick them faster than the controller poles so that estimation settles before the controller acts.

The miraculous **separation principle** says that the closed-loop poles of *controller + observer* are exactly the union of the controller poles (eigenvalues of $A - BK$) and the observer poles (eigenvalues of $A - LC$). You can design the two pieces independently.

---

## 9. Worked Example: Inverted Pendulum on a Cart

The cart-pendulum has linearised state $\mathbf x = [x,\; \dot x,\; \theta,\; \dot\theta]^T$ with

$$
A = \begin{pmatrix}
0 & 1 & 0 & 0 \\
0 & -b/M & -mg/M & 0 \\
0 & 0 & 0 & 1 \\
0 & b/(ML) & (M+m)g/(ML) & 0
\end{pmatrix},
\quad
B = \begin{pmatrix} 0 \\ 1/M \\ 0 \\ -1/(ML) \end{pmatrix}.
$$

Look at $\mathbf{eigvals}(A)$: one of them sits in the right half-plane (the open-loop pendulum falls). We compute $K_{\text{LQR}}$ from the Riccati equation (above), then simulate $\dot{\mathbf x} = (A - BK)\mathbf x$ from a $0.2$-rad initial tilt. The bottom row of the state-space figure shows the angle decaying smoothly to zero, the cart settling to the origin, and the actuator effort dropping to a small steady push -- the full design loop in fewer than 30 lines of Python.

---

## 10. The Modern Picture

What we have done in this chapter is replay the ODE story with a **goal**. Stability theory (Ch 7-8) said *whether* a system equilibrium survives small perturbations; control theory says *how to engineer* the equilibrium and the convergence rate to it. The same Laplace transform that gave us closed-form solutions in Chapter 4 now gives us frequency-domain robustness margins. The same matrix exponential from Chapter 6 now powers state-space simulation and observer design.

The frontier extends beyond what we covered:

- **Robust control** -- $H_\infty$, $\mu$-synthesis: certify performance under bounded modelling error.
- **Adaptive and model-predictive control (MPC)** -- the controller updates online using a moving-horizon optimisation.
- **Nonlinear control** -- feedback linearisation, sliding mode, control Lyapunov functions.
- **Reinforcement learning** -- the controller is *learned* from experience rather than designed; ties back to Chapter 18's Neural ODEs.

---

## Summary

| Concept | Equation / tool | Designs |
|---|---|---|
| Transfer function | $G(s) = Y/U$ via Laplace | classical compensators, lead/lag |
| PID | $u = K_p e + K_i \int e + K_d \dot e$ | 90% of industrial loops |
| Root locus | $1 + KL(s) = 0$ | gain selection, stability boundary |
| Bode / Nyquist | $L(j\omega)$ | gain & phase margins |
| State space | $\dot x = Ax + Bu$ | MIMO, modern control |
| Pole placement | $u = -Kx$ | dictate closed-loop poles |
| LQR | Riccati equation | optimal $K$ |
| Observer | $\dot{\hat x} = A\hat x + Bu + L(y - C\hat x)$ | estimate hidden states |

Control theory **promotes ODEs from description to design**. We no longer just predict how a system will move -- we tell it where to go.

---

## References

- Ogata, *Modern Control Engineering*, Pearson (2010).
- Franklin, Powell & Emami-Naeini, *Feedback Control of Dynamic Systems*, Pearson (2015).
- Astrom & Murray, *Feedback Systems*, Princeton (2008). (Free PDF.)
- Skogestad & Postlethwaite, *Multivariable Feedback Control*, Wiley (2005).

---

**Previous Chapter**: [Chapter 15: Population Dynamics](/en/ode-chapter-15-population-dynamics/)

**Next Chapter**: [Chapter 17: Physics and Engineering Applications](/en/ode-chapter-17-physics-engineering-applications/)

*This is Part 16 of the 18-part series on Ordinary Differential Equations.*
