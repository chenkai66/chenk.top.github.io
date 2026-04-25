---
title: "ODE Chapter 8: Nonlinear Systems and Phase Portraits"
date: 2023-10-28 09:00:00
tags:
  - ODE
  - Nonlinear Systems
  - Phase Portraits
  - Lotka-Volterra
  - Stability Analysis
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 8
  total: 18
lang: en
mathjax: true
description: "Step beyond linearity: predator-prey oscillations, competition exclusion, Van der Pol limit cycles, Hamiltonian systems, and the Poincare-Bendixson theorem -- the full toolkit for nonlinear 2D dynamics."
disableNunjucks: true
series_order: 8
---

**The real world is nonlinear.** Predator-prey cycles, heartbeat rhythms, neuron firing -- none of these can be captured by linear equations. When superposition fails, the world acquires *new* behaviors: limit cycles, multiple equilibria, bistability, hysteresis. This chapter gives you the geometric and analytic tools to read those behaviors directly off a 2D phase portrait.

## What You Will Learn

- Why nonlinear systems are *fundamentally* different from linear ones
- Lyapunov stability visualized: level sets, bowls, and basins
- Linearization vs. the full nonlinear picture (Hartman-Grobman in action)
- Lotka-Volterra predator-prey: closed orbits and conserved quantities
- Competition models: four canonical outcomes
- Van der Pol oscillator and the geometry of limit cycles
- Gradient and Hamiltonian systems
- Poincare-Bendixson: why 2D systems cannot be chaotic

## Prerequisites

- Chapter 6: linear systems, phase portrait classification
- Chapter 7: stability, linearization, Lyapunov functions

---

## From Linear to Nonlinear

Linear systems obey **superposition**: if $\mathbf{x}_1$ and $\mathbf{x}_2$ are solutions, so is $c_1\mathbf{x}_1 + c_2\mathbf{x}_2$. This is the engine that powers the entire toolkit of Chapters 1-6 -- exponential ansatz, eigenvectors, fundamental matrices.

Nonlinear systems break this rule and pay the price -- closed-form solutions vanish. But they get something priceless in return:

- **Multiple equilibria**, each with its own stability type
- **Limit cycles** -- isolated, stable periodic orbits (impossible in linear systems)
- **Bistability and hysteresis** -- memory of initial conditions
- **Sensitive dependence** -- chaos, in 3D and beyond (Chapter 9)

Almost every interesting system in physics, biology, chemistry, neuroscience, and economics is nonlinear.

---

## Lyapunov Stability, Visualized

A Lyapunov function $V(\mathbf{x})$ is a scalar that decreases along trajectories ($\dot V \leq 0$). Geometrically, level sets of $V$ form a nested family of "bowls" around the equilibrium, and trajectories cross them inward.

![Lyapunov level sets shrinking toward the origin; trajectories cross inward; V(t) decays.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/08-nonlinear-stability/fig1_lyapunov_stability.png)
*Left: trajectories from five different starting points all spiral inward, threading the colored level sets of $V(x,y) = x^2 + \tfrac12 y^2$. Right: $V(\mathbf{x}(t))$ decays monotonically (log scale) -- direct geometric proof that the origin is asymptotically stable.*

Once you see Lyapunov stability as "trajectories falling down a bowl", every theorem becomes obvious:

- $\dot V \leq 0$: trajectories never go uphill -> they stay in the smallest bowl that contains the start (stability).
- $\dot V < 0$ strictly: they keep falling -> they reach the bottom (asymptotic stability).
- $\dot V > 0$: trajectories climb out -> instability.

LaSalle generalizes: when $\dot V$ vanishes on a set, trajectories settle on the largest *invariant subset* of that set.

---

## Phase Portraits and Nullclines

For $x' = f(x,y),\ y' = g(x,y)$:

- The **$x$-nullcline** is the curve $f(x,y) = 0$. There $\dot x = 0$, so trajectories cross it **vertically**.
- The **$y$-nullcline** is the curve $g(x,y) = 0$. Trajectories cross it **horizontally**.
- **Equilibria** sit at intersections of the two nullclines.
- The signs of $f$ and $g$ in each region tell you the direction of flow.

Nullcline analysis is the cheapest way to sketch a phase portrait by hand.

---

## Linearization: Local Truth, Global Surprise

Near a hyperbolic equilibrium, the Jacobian's eigenvalues completely determine the local picture (Hartman-Grobman). But far from the equilibrium, *all bets are off*. The damped pendulum makes this dramatic:

![Side-by-side phase portraits: full nonlinear pendulum vs its linearization at the origin.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/08-nonlinear-stability/fig3_linearization.png)
*Left: the true pendulum has periodic equilibria at every $n\pi$, separatrices, and rotating regimes. Right: its linearization $\ddot\theta + \gamma\dot\theta + \omega_0^2\theta = 0$ shows only a single stable spiral. Locally identical near the origin, globally entirely different.*

| Linear-eigenvalue type | Local equilibrium | Stability (nonlinear) |
|---|---|---|
| $\lambda_1 < \lambda_2 < 0$ (real) | Stable node | Asymptotically stable |
| $0 < \lambda_1 < \lambda_2$ (real) | Unstable node | Unstable |
| $\lambda_1 < 0 < \lambda_2$ | Saddle | Unstable |
| $\alpha \pm \beta i,\ \alpha < 0$ | Stable spiral | Asymptotically stable |
| $\alpha \pm \beta i,\ \alpha > 0$ | Unstable spiral | Unstable |
| $\pm \beta i$ | Center | **Inconclusive** -- nonlinear terms decide |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def damped(state, t, b):
    x, v = state
    return [v, -x - b*v]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
t = np.linspace(0, 30, 1000)
for ax, (b, title) in zip(axes, [(0,'Undamped'), (0.3,'Underdamped'), (2.0,'Overdamped')]):
    for r in [0.5, 1, 1.5, 2]:
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            sol = odeint(damped, [r*np.cos(theta), r*np.sin(theta)], t, args=(b,))
            ax.plot(sol[:,0], sol[:,1], 'b-', linewidth=0.7, alpha=0.6)
    ax.plot(0, 0, 'ro', markersize=6); ax.set_title(title)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## Lotka-Volterra Predator-Prey

### The model

$$x' = \alpha x - \beta xy, \qquad y' = \delta xy - \gamma y$$

- $x$: prey population, $y$: predator population
- Trivial equilibrium $(0,0)$: saddle
- Coexistence equilibrium $(\gamma/\delta,\ \alpha/\beta)$: Jacobian eigenvalues $\pm i\sqrt{\alpha\gamma}$ (a center)

The conserved quantity
$$H(x,y) = \delta x - \gamma\ln x + \beta y - \alpha\ln y$$
makes every orbit closed. Time-series and phase-plane look like this:

![Lotka-Volterra: time-series oscillation (left) and closed orbits encircling the center (right).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/08-nonlinear-stability/fig2_lotka_volterra.png)
*Left: predator (red) lags prey (blue) by a quarter period -- the textbook ecological cycle. Right: a family of nested closed orbits around the center $(c/d,\ a/b)$. Different starting conditions live on different orbits forever.*

### The cycle in words

1. Abundant prey -> predators thrive -> predator population rises.
2. Many predators -> prey depleted -> prey crashes.
3. Few prey -> predators starve -> predator population falls.
4. Few predators -> prey rebounds -> back to step 1.

### Limitations

- **Structurally unstable.** The slightest extra term destroys the closed orbits.
- **No carrying capacity.** Prey grow unbounded if predators are absent.
- **Ignores space and time delays.**

These flaws drove the development of the more realistic models in the next section.

---

## Competition Model: Four Outcomes

$$\begin{aligned} x' &= r_1 x\!\left(1 - \frac{x + \alpha_{12}y}{K_1}\right), \\ y' &= r_2 y\!\left(1 - \frac{y + \alpha_{21}x}{K_2}\right). \end{aligned}$$

The product $\alpha_{12}\,\alpha_{21}$ -- the strength of mutual interference -- determines which of four pictures you get.

![Four competition phase portraits with nullclines: species 1 wins, species 2 wins, coexistence, bistability.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/08-nonlinear-stability/fig5_competition_outcomes.png)
*Blue and red lines are the nullclines. Black dots are equilibria. Grey curves are sample trajectories. The geometry of nullcline intersections decides the long-term fate of every initial condition.*

| Regime | Condition | Outcome |
|---|---|---|
| Weak interference | $\alpha_{12} < 1,\ \alpha_{21} < 1$ | Stable coexistence |
| Strong interference | $\alpha_{12} > 1,\ \alpha_{21} > 1$ | Bistability -- winner depends on starting populations |
| Asymmetric | $\alpha_{12} < 1,\ \alpha_{21} > 1$ | Species 1 wins |
| Asymmetric | $\alpha_{12} > 1,\ \alpha_{21} < 1$ | Species 2 wins |

This is **competitive exclusion** in mathematical clothing.

---

## Van der Pol: Limit Cycles from Nonlinear Damping

$$x'' - \mu(1 - x^2)x' + x = 0$$

The genius is in the damping coefficient $-\mu(1 - x^2)$:

- Inside $|x| < 1$: damping is *negative* -- the system pumps energy in.
- Outside $|x| > 1$: damping is *positive* -- energy bleeds out.

Trajectories from inside grow, trajectories from outside shrink, and both settle on a single **stable limit cycle** -- an isolated periodic orbit that attracts everything in its basin.

![Van der Pol limit cycles for mu = 0.5, 1.5, 4.0; cycles get sharper as mu increases.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/08-nonlinear-stability/fig4_limit_cycles.png)
*Three values of $\mu$. Grey curves are transients spiralling toward the cycle. Coloured curves are the limit cycle itself. As $\mu$ grows, the cycle deforms from near-sinusoidal into the famous "relaxation oscillation" with sharp jumps.*

**Where this shows up:** heartbeat pacemaker cells, neuron action potentials, electronic oscillator circuits, geyser eruptions, business cycles.

---

## Gradient and Hamiltonian Systems: Two Special Worlds

### Gradient systems: $\mathbf{x}' = -\nabla V$

- Trajectories follow the steepest descent of $V$.
- $\dot V = -|\nabla V|^2 \leq 0$ -- the potential always decreases.
- **No periodic orbits possible** (you can't go around a hill and end up lower).
- Machine learning's gradient descent is the discrete cousin.

### Hamiltonian systems: $x' = \dfrac{\partial H}{\partial y},\ y' = -\dfrac{\partial H}{\partial x}$

- $H$ is conserved along every trajectory ($\dot H = 0$).
- Phase-space volume is preserved (Liouville's theorem).
- Orbits are level curves of $H$.
- The undamped pendulum is a textbook example.

These two worlds sit at opposite extremes of the dissipation spectrum.

---

## Poincare-Bendixson: 2D Cannot Be Chaotic

**Theorem (Poincare-Bendixson).** A bounded trajectory of a smooth 2D continuous system that does not approach any equilibrium must converge to a **closed orbit**.

In words: in 2D, the only long-term behaviors are *equilibrium* or *periodic*. There is no room for chaos.

The Jordan curve theorem is the secret here -- a closed orbit divides the plane in two, trapping the trajectory. Add a third dimension and the trajectory can escape *over* the orbit, opening the door to chaos (Chapter 9).

**Bendixson's criterion (no closed orbits).** If $\partial f/\partial x + \partial g/\partial y$ has constant non-zero sign in a simply-connected region, no closed orbit lies inside it.

---

## Numerical Methods Quick Reference

| Method | Order | Notes |
|---|---|---|
| Euler | $O(h)$ | Simple but inaccurate |
| Improved Euler (Heun) | $O(h^2)$ | Average of two slopes |
| RK4 | $O(h^4)$ | Best accuracy/cost ratio |
| RK45 (Dormand-Prince) | adaptive | Default in `scipy.integrate.solve_ivp` |

```python
import numpy as np

def rk4_step(f, x, t, h):
    k1 = h * np.array(f(x, t))
    k2 = h * np.array(f(x + k1/2, t + h/2))
    k3 = h * np.array(f(x + k2/2, t + h/2))
    k4 = h * np.array(f(x + k3,    t + h))
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6
```

---

## Summary

| Concept | Key Point |
|---|---|
| Nonlinearity | Superposition fails; richer dynamics |
| Lyapunov visualization | Trajectories cross level sets inward |
| Linearization | Locally accurate near hyperbolic equilibria; globally only suggestive |
| Lotka-Volterra | Closed orbits from a conserved quantity |
| Competition | Four outcomes via nullcline geometry |
| Van der Pol | Limit cycle from sign-changing damping |
| Gradient systems | No periodic orbits |
| Hamiltonian systems | Energy conserved; orbits are level curves |
| Poincare-Bendixson | 2D rules out chaos |

---

## Exercises

**Conceptual.**

1. Why does superposition fail for $y' = y^2$? Give a concrete counterexample.
2. Why are 2D continuous systems forbidden from being chaotic?
3. Sketch by hand the phase portrait of $x' = y,\ y' = -x + x^3 - 0.2y$ (Duffing).

**Computational.**

4. For $x' = x - x^3,\ y' = -y$: find every equilibrium and classify each.
5. Prove $V = x^2 + y^2$ is a Lyapunov function for $x' = -x + y^2,\ y' = -y$.
6. Verify $H = \delta x - \gamma\ln x + \beta y - \alpha\ln y$ is conserved for Lotka-Volterra.

**Programming.**

7. Reproduce the four competition regimes in fig 5 and shade each basin of attraction.
8. Numerically estimate the Van der Pol period $T(\mu)$ for $\mu \in \{0.1, 0.5, 1, 3, 10\}$.
9. Compare Euler vs. RK4 accuracy for the Van der Pol equation -- find the $\mu$ at which Euler breaks down.

---

## References

- Strogatz, *Nonlinear Dynamics and Chaos*, Westview Press (2015)
- Murray, *Mathematical Biology I*, Springer (2002)
- Hirsch, Smale, & Devaney, *Differential Equations, Dynamical Systems, and Chaos*, Academic Press (2012)
- Verhulst, *Nonlinear Differential Equations and Dynamical Systems*, Springer (1996)

---

**Series Navigation**

