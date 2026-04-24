---
title: "ODE Chapter 7: Stability Theory"
date: 2024-08-08 09:00:00
tags:
  - ODE
  - Stability
  - Lyapunov Theory
  - Phase Space
  - Bifurcation
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 7
  total: 18
lang: en
mathjax: true
description: "Will a bridge survive the wind? Will an ecosystem recover from a shock? Stability theory answers these questions using Lyapunov functions, linearization, and bifurcation analysis."
disableNunjucks: true
---

**A small push hits a system. Does it return to rest, drift away, or break entirely?** That single question decides whether bridges survive storms, ecosystems recover from droughts, and economies bounce back from crises. Stability theory answers it -- and it does so *without ever solving the differential equation*. We will learn to read the destiny of a system off the geometry of its phase plane.

## What You Will Learn

- Three precise notions: Lyapunov stable, asymptotically stable, unstable
- Linearization via the Jacobian and the Hartman-Grobman theorem
- Lyapunov's direct method -- proving stability with energy-like functions
- LaSalle's invariance principle for borderline cases
- Trace-determinant classification of all 2D linear systems
- Four canonical bifurcations: saddle-node, transcritical, pitchfork, Hopf
- Worked applications: pendulum, predator-prey, inverted pendulum control

## Prerequisites

- Chapter 6: linear systems, eigenvalues, phase portraits
- Multivariable calculus: partial derivatives, Jacobian matrix

---

## A Visual Tour Before the Theory

Stability is, at heart, a *geometric* statement about how trajectories move in phase space. Six pictures tell the entire story of 2D linear systems.

![Six canonical 2D phase portraits: stable spiral, unstable spiral, stable node, saddle, center, degenerate node.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/07-systems-and-phase-plane/fig1_phase_portraits.png)
*Six canonical 2D phase portraits. Solid lines are forward orbits; dashed lines are backward orbits. The eigenvalues of the linearization completely determine which picture you get.*

For nonlinear systems, the same six pictures still appear -- but only locally, near each equilibrium. The damped pendulum and the Lotka-Volterra predator-prey model both show this beautifully:

![Damped pendulum (left) and Lotka-Volterra (right) trajectories overlaid on their vector fields.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/07-systems-and-phase-plane/fig2_trajectory_vector_field.png)
*Damped pendulum: stable foci alternate with saddles along the$\theta$-axis. Lotka-Volterra: closed orbits encircle a non-hyperbolic center.*

---

## Stability Defined Precisely

Consider $\mathbf{x}' = \mathbf{f}(\mathbf{x})$ with equilibrium $\mathbf{x}^*$ (so $\mathbf{f}(\mathbf{x}^*) = \mathbf{0}$).

**Lyapunov stable.** For every $\varepsilon > 0$, there exists $\delta > 0$ such that
$$\|\mathbf{x}(0) - \mathbf{x}^*\| < \delta \;\Longrightarrow\; \|\mathbf{x}(t) - \mathbf{x}^*\| < \varepsilon \;\;\text{for all } t > 0.$$
*Intuition: nearby trajectories stay nearby forever.*

**Asymptotically stable.** Lyapunov stable **and** $\mathbf{x}(t) \to \mathbf{x}^*$ as $t \to \infty$.
*Intuition: nearby trajectories not only stay nearby but eventually return.*

**Unstable.** Not Lyapunov stable.

The **basin of attraction** is the set of all initial conditions that converge to $\mathbf{x}^*$. Asymptotic stability is a *local* property; the basin tells you how local.

> **Why two definitions?** A center (closed orbits) is Lyapunov stable but not asymptotically stable -- trajectories stay close but never settle. The Lotka-Volterra model is the classic example.

---

## Linearization: The Jacobian Method

Near equilibrium $\mathbf{x}^*$, Taylor-expand and keep the linear part:
$$\mathbf{x}' \;\approx\; J(\mathbf{x} - \mathbf{x}^*), \qquad J_{ij} = \frac{\partial f_i}{\partial x_j}\bigg|_{\mathbf{x}^*}.$$

### Hartman-Grobman theorem

If every eigenvalue of $J$ has **nonzero real part** (a *hyperbolic* equilibrium), then the nonlinear system is locally **topologically equivalent** to its linearization $\mathbf{u}' = J\mathbf{u}$.

- All $\operatorname{Re}(\lambda) < 0$: asymptotically stable
- Any $\operatorname{Re}(\lambda) > 0$: unstable
- Purely imaginary eigenvalues: **linearization fails** -- use Lyapunov methods

### Example: damped pendulum

$$\theta'' + \gamma\theta' + \omega_0^2\sin\theta = 0$$

| Equilibrium | Jacobian | Verdict |
|-----------|----------|---------|
| $(0,0)$ hanging | $\begin{pmatrix}0 & 1 \\ -\omega_0^2 & -\gamma\end{pmatrix}$ | Both eigenvalues have $\operatorname{Re}<0$ when $\gamma>0$: **stable focus** |
| $(\pi,0)$ inverted | $\begin{pmatrix}0 & 1 \\ \omega_0^2 & -\gamma\end{pmatrix}$ | One positive eigenvalue: **saddle, unstable** |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

gamma, omega0 = 0.3, 1.0
def pendulum(X, t):
    x, y = X
    return [y, -gamma*y - omega0**2*np.sin(x)]

fig, ax = plt.subplots(figsize=(10, 6))
t = np.linspace(0, 50, 2000)
for x0 in np.linspace(-3*np.pi, 3*np.pi, 15):
    for y0 in np.linspace(-3, 3, 5):
        sol = odeint(pendulum, [x0, y0], t)
        ax.plot(sol[:,0], sol[:,1], 'b-', linewidth=0.3, alpha=0.5)
for n in range(-2, 3):
    ax.plot(n*np.pi, 0, 'go' if n%2==0 else 'rx', markersize=10)
ax.set(xlabel='theta', ylabel='omega', title='Damped pendulum phase portrait')
ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
```

---

## Lyapunov's Direct Method

### The big idea

Stability without solving the ODE. Construct an *energy-like* scalar function $V(\mathbf{x})$ and show it decreases along trajectories.

**Requirements.**

1. $V(\mathbf{x}^*) = 0$ and $V(\mathbf{x}) > 0$ otherwise (positive definite)
2. $\dot V = \nabla V \cdot \mathbf{f}(\mathbf{x}) \leq 0$ along orbits

**Stability theorems.**

| Sign of $\dot V$ | Conclusion |
|---|---|
| $\dot V \leq 0$ | $\mathbf{x}^*$ Lyapunov stable |
| $\dot V < 0$ except at $\mathbf{x}^*$ | Asymptotically stable |
| $V > 0,\ \dot V > 0$ | Unstable (Chetaev) |

### Why it works -- the picture

Trajectories cross level sets of $V$ inward. Since $V$ has a minimum at $\mathbf{x}^*$, they cannot escape arbitrarily-small level sets, and (with strict descent) they slide all the way to the bottom.

![Lyapunov surface as a bowl with trajectories descending into the basin; right panel shows V(t) decaying.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/07-systems-and-phase-plane/fig4_lyapunov_function.png)
*Left: trajectories slide down the bowl $V(x,y) = x^2 + \tfrac12 y^2$. Right: $V$ along each trajectory decays monotonically -- the geometric proof of asymptotic stability.*

### How to find $V$

- **Physical energy:** kinetic + potential for mechanical systems
- **Quadratic forms:** $V = \mathbf{x}^T P \mathbf{x}$ where $P$ solves the Lyapunov equation $A^T P + PA = -Q$
- **Trial:** start with $V = x^2 + y^2$, compute $\dot V$, adjust coefficients

### Example: pendulum energy

$$V(\theta, \omega) = \tfrac{1}{2}\omega^2 + (1 - \cos\theta), \qquad \dot V = -\gamma\omega^2 \leq 0.$$

The hanging position is stable. LaSalle's principle (next) upgrades this to asymptotic.

---

## LaSalle's Invariance Principle

Sometimes $\dot V \leq 0$ but vanishes on a whole set, not just at $\mathbf{x}^*$. Standard Lyapunov only gives stability, not attraction.

**Theorem.** Let $E = \{\mathbf{x} : \dot V = 0\}$ and $M$ be the *largest invariant subset* of $E$. Every bounded trajectory approaches $M$.

If $M = \{\mathbf{x}^*\}$, then $\mathbf{x}^*$ is asymptotically stable.

For the damped pendulum, $\dot V = 0$ requires $\omega = 0$. But on the line $\omega = 0$ the dynamics force $\dot\omega = -\omega_0^2 \sin\theta \neq 0$ unless also $\theta = 0$. So $M = \{(0,0)\}$ and we get asymptotic stability for free.

---

## Trace-Determinant Classification

For $\mathbf{x}' = A\mathbf{x}$ in 2D, set $\tau = \operatorname{tr}(A)$ and $\Delta = \det(A)$. Then $\lambda_{1,2} = \tfrac12(\tau \pm \sqrt{\tau^2 - 4\Delta})$, and:

| Region | Type |
|---|---|
| $\Delta < 0$ | Saddle |
| $\Delta > 0,\ \tau < 0,\ \tau^2 > 4\Delta$ | Stable node |
| $\Delta > 0,\ \tau > 0,\ \tau^2 > 4\Delta$ | Unstable node |
| $\Delta > 0,\ \tau < 0,\ \tau^2 < 4\Delta$ | Stable spiral |
| $\Delta > 0,\ \tau > 0,\ \tau^2 < 4\Delta$ | Unstable spiral |
| $\Delta > 0,\ \tau = 0$ | Center |
| $\tau^2 = 4\Delta$ | Degenerate / improper node |

A single picture compresses this entire table:

![Trace-determinant plane shaded into stable/unstable regions for spirals, nodes, saddles, centers.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/07-systems-and-phase-plane/fig3_eigenvalue_classification.png)
*Trace-determinant plane. The parabola $\tau^2 = 4\Delta$ separates spirals (above) from nodes (below). The positive $\Delta$-axis is the line of centers.*

---

## Bifurcations: When the Picture Itself Changes

Slowly turn a parameter knob $r$. Equilibria can be born, die, or swap stability. Four "normal forms" capture every codimension-1 bifurcation locally.

![Four canonical bifurcation diagrams: saddle-node, transcritical, pitchfork, Hopf.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/07-systems-and-phase-plane/fig5_bifurcation_normal_forms.png)
*Solid green: stable equilibria. Dashed red: unstable. Hopf panel shows the limit-cycle radius $\sqrt{\mu}$ growing from zero.*

| Bifurcation | Normal form | What happens |
|---|---|---|
| Saddle-node | $\dot x = r + x^2$ | Two equilibria collide and annihilate at $r = 0$ |
| Transcritical | $\dot x = rx - x^2$ | Two equilibria pass through each other and exchange stability |
| Pitchfork | $\dot x = rx - x^3$ | One equilibrium splits into three (symmetry breaking) |
| Hopf | complex eigenvalues cross $i\mathbb{R}$ | A stable focus loses stability and a limit cycle appears |

Hopf is the mechanism behind every self-sustained oscillation in nature -- from heartbeats to the pulsing of variable stars.

---

## Application 1: Lotka-Volterra Predator-Prey

$$x' = ax - bxy, \qquad y' = -cy + dxy$$

The non-trivial equilibrium $(c/d,\ a/b)$ has Jacobian eigenvalues $\pm i\sqrt{ac}$ -- a **center**. The Hartman-Grobman theorem does *not* apply (eigenvalues are imaginary), but a conserved quantity
$$H(x,y) = dx - c\ln x + by - a\ln y$$
makes every orbit closed. The system has periodic population cycles (right panel of fig 2).

## Application 2: Inverted Pendulum Control

The inverted equilibrium is a saddle. Linear feedback $u = -K\mathbf{x}$ shifts the closed-loop eigenvalues into the open left half-plane, converting the saddle into a stable focus.

```python
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint
import matplotlib.pyplot as plt

g, l, m, M_cart = 9.81, 0.5, 0.1, 1.0
A = np.array([[0,1,0,0],[0,0,-m*g/M_cart,0],
              [0,0,0,1],[0,0,(M_cart+m)*g/(M_cart*l),0]])
B = np.array([[0],[1/M_cart],[0],[-1/(M_cart*l)]])

Q = np.diag([10, 1, 100, 1])
R = np.array([[0.1]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("Closed-loop eigenvalues:", np.linalg.eigvals(A - B @ K))

t = np.linspace(0, 5, 500)
sol = odeint(lambda X, t: (A @ X + B @ (-K @ X)).flatten(),
             [0,0,0.2,0], t)

plt.figure(figsize=(8, 4))
plt.plot(t, sol[:,2]*180/np.pi, 'b-', linewidth=2)
plt.xlabel('Time (s)'); plt.ylabel('Angle (degrees)')
plt.title('Inverted pendulum stabilized by LQR')
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
```

---

## Summary

| Concept | Key Point |
|---|---|
| Lyapunov stability | Nearby trajectories stay nearby |
| Asymptotic stability | Nearby trajectories converge to equilibrium |
| Linearization | Jacobian eigenvalues determine local fate (if hyperbolic) |
| Lyapunov function | Energy-like scalar that proves stability without integration |
| LaSalle's principle | Upgrades $\dot V \leq 0$ to asymptotic stability via invariant sets |
| Trace-determinant plane | Single picture classifying every 2D linear system |
| Bifurcations | Saddle-node, transcritical, pitchfork, Hopf -- four ways the picture changes |

---

## Exercises

**Basic.**

1. Determine stability for: (a) $x' = -x,\ y' = -2y$; (b) $x' = y,\ y' = -\sin x - 0.5y$.
2. Use $V = x^2 + y^2$ to analyze $x' = -x + y^2,\ y' = -y$.
3. Find all bifurcation points of $\dot x = rx - x^3$.

**Advanced.**

4. Prove total energy is a Lyapunov function for the damped pendulum, then apply LaSalle.
5. Analyze the Van der Pol oscillator: show the origin is unstable but a stable limit cycle exists.
6. For Lotka-Volterra competition, derive the conditions for coexistence vs. exclusion.

**Programming.**

7. Auto-classify 2D linear-system equilibria from trace and determinant; reproduce fig 3.
8. Animate the Hopf bifurcation: sweep $\mu$ and watch the limit cycle grow.

---

## References

- Strogatz, *Nonlinear Dynamics and Chaos*, CRC Press (2015)
- Khalil, *Nonlinear Systems*, Prentice Hall (2002)
- Guckenheimer & Holmes, *Nonlinear Oscillations, Dynamical Systems, and Bifurcations*, Springer (1983)
- Perko, *Differential Equations and Dynamical Systems*, Springer (2001)

---

**Series Navigation**

| | |
|---|---|
| **Previous** | [Chapter 6: Linear Systems](/en/ode-chapter-06-power-series/) |
| **Current** | Chapter 7: Stability Theory |
| **Next** | [Chapter 8: Nonlinear Systems and Phase Portraits](/en/ode-chapter-08-nonlinear-stability/) |
