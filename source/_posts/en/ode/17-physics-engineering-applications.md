---
title: "Ordinary Differential Equations (17): Physics and Engineering Applications"
date: 2024-09-17 09:00:00
tags:
  - Ordinary Differential Equations
  - Pendulum
  - RLC Circuit
  - Kepler Orbit
  - Vibration Analysis
  - Fluid Mechanics
  - Python
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 17
  total: 18
lang: en
mathjax: true
description: "See ODEs in action across physics and engineering. Walk through the nonlinear pendulum, RLC circuit and resonance, Kepler orbits and conservation laws, multi-DOF structural vibration with tuned mass dampers, and fluid flow from Poiseuille to vortex shedding -- all with full Python simulations."
disableNunjucks: true
---

**Differential equations are not a pure mathematical game -- they are the language for understanding the physical world.** From celestial motion to circuit response, from a swinging pendulum to vortex shedding behind a bridge cable, every dynamical system "speaks" ODE.

This chapter is a deliberate tour through five canonical applications. Each one will pay back the entire ODE toolkit we built in chapters 1-16: phase planes, eigenvalues, Laplace transforms, modal analysis, conservation laws, numerical integration, control. None of the examples is a "toy" -- they are all genuine working physics, written tightly so that the structure remains visible.

## What You Will Learn

- The nonlinear pendulum: small-angle linearisation vs the full equation, period vs amplitude, the separatrix
- RLC circuits: the three damping regimes, resonance, the Q-factor, and tuning
- Kepler orbits: gravity, eccentricity, conservation of energy and angular momentum, Bertrand's theorem
- Multi-DOF structural vibration: modal analysis, resonance, tuned mass dampers (the Tacoma lesson)
- Fluid mechanics: Poiseuille flow, Reynolds number, vortex shedding, Stokes settling

## Prerequisites

- Chapters 1-3 -- first- and second-order linear theory
- Chapter 6-7 -- linear systems and phase portraits
- Chapter 8 -- nonlinear stability and energy methods
- Chapter 11 -- numerical methods (used throughout)

---

## 1. The Nonlinear Pendulum -- the Hello World of nonlinear ODEs

A point mass on a rigid rod of length $L$ in gravity $g$ obeys

$$
\ddot\theta + \frac{g}{L}\sin\theta = 0.
$$

For small $\theta$ we replace $\sin\theta \approx \theta$ and recover simple harmonic motion with period $T_0 = 2\pi\sqrt{L/g}$ -- *independent* of amplitude. That isochronism is the small-angle miracle Galileo exploited.

The full equation is *not* isochronous: the period depends on amplitude. The exact answer involves the complete elliptic integral of the first kind:

$$
T(\theta_0) = \frac{4}{\omega_0}\,K\!\bigl(\sin(\theta_0/2)\bigr), \qquad \omega_0 = \sqrt{g/L}.
$$

![Pendulum: large-angle vs linearised time histories, period vs amplitude, and the global phase portrait with libration / separatrix / rotation regions.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig1_pendulum_motion.png)
*Top-left: from the same large initial angle ($\theta_0 = 1.5$ rad), the linear and full equations agree for one half-period and then drift apart -- the linear model loses *phase* first, then amplitude. Top-right: $T(\theta_0)$ from numerical integration; below 0.5 rad the small-angle constant period is essentially perfect, but as $\theta_0 \to \pi$ the period diverges because the unstable equilibrium has been reached. Bottom: the full phase portrait $(\theta, \dot\theta)$. Inside the separatrix (red) the pendulum swings (libration); outside it goes over the top (rotation). The pattern repeats $2\pi$-periodically along $\theta$.*

```python
from scipy.integrate import solve_ivp
import numpy as np

g, L = 9.81, 1.0

def pendulum(t, s):
    theta, omega = s
    return [omega, -(g/L)*np.sin(theta)]

# librations vs rotations: vary initial angular velocity at theta=0
for v0 in [1.5, 4.0, 6.5, 7.5]:    # last two go over the top
    sol = solve_ivp(pendulum, (0, 10), [0.0, v0],
                    t_eval=np.linspace(0, 10, 2000),
                    rtol=1e-10, atol=1e-12)
    # examine sol.y -- you will see closed loops or monotone theta
```

**Why this matters.** The pendulum is the simplest non-trivial nonlinear conservative system, and its phase portrait already contains every concept of Chapters 7-8: stable/unstable equilibria, separatrices, period-energy relations, integrability. Every other oscillator -- molecular vibrations, Josephson junctions, plasma waves -- borrows this skeleton.

---

## 2. RLC Circuits -- the same equation, in copper

A series resistor-inductor-capacitor loop driven by voltage $V(t)$ obeys, for charge $q$ on the capacitor:

$$
L\ddot q + R\dot q + \frac{1}{C}\,q = V(t).
$$

Compare with the mass-spring-damper $m\ddot x + c\dot x + kx = F(t)$. They are the *same* equation -- so all our intuition about damping ratio and natural frequency carries over directly.

Define $\omega_n = 1/\sqrt{LC}$ (natural frequency) and $\zeta = (R/2)\sqrt{C/L}$ (damping ratio). The transfer function from $V$ to $q$ is

$$
H(s) = \frac{1/L}{s^2 + (R/L)\,s + 1/(LC)},
$$

with the same three step-response regimes from Chapter 16: underdamped, critical, overdamped.

![Series RLC circuit: schematic, step responses, frequency response, and the Q-factor / bandwidth trade-off.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig2_rlc_circuit.png)
*Top-left: the canonical series RLC. Top-right: step responses for three damping ratios; the underdamped case rings around the steady value, critical lands fastest without overshoot, overdamped is sluggish. Bottom-left: $|H(j\omega)|$ has a sharp resonant peak when $\zeta$ is small. Bottom-right: as $R$ rises, the Q factor $Q = (1/R)\sqrt{L/C}$ falls and the 3-dB bandwidth $\Delta\omega = R/L$ widens -- a fundamental trade between selectivity and speed.*

The peak amplitude of an undamped oscillator driven at resonance grows linearly in time -- a phenomenon the Tacoma Narrows Bridge made historic in 1940 (more on that in section 4).

---

## 3. Planetary Orbits -- where Newton meets Kepler

Newton's gravitational law gives, in two dimensions,

$$
\ddot{\mathbf r} \;=\; -\frac{GM\,\mathbf r}{|\mathbf r|^3}.
$$

The conserved quantities are total energy $E = \tfrac12|\mathbf v|^2 - GM/|\mathbf r|$ and angular momentum $\mathbf L = \mathbf r \times \mathbf v$. Together they prove Kepler's three laws -- without solving the ODE in closed form.

```python
from scipy.integrate import solve_ivp
import numpy as np

GM = 1.0
def kepler(t, s):
    x, y, vx, vy = s
    r = np.hypot(x, y)
    return [vx, vy, -GM*x/r**3, -GM*y/r**3]

# circle, ellipse, near-parabolic
for v0 in [1.0, 0.85, 0.55]:
    sol = solve_ivp(kepler, (0, 15), [1, 0, 0, v0],
                    t_eval=np.linspace(0, 15, 4000),
                    method='DOP853', rtol=1e-10)
    # sol.y[0:2] traces the orbit
```

![Kepler orbits at different initial speeds; inverse-square vs inverse-cube collapse; numerical conservation of energy and angular momentum.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig3_planetary_orbits.png)
*Left: starting at $(1,0)$ with varying tangential velocity, the orbit transitions from a circle to a thin ellipse -- exactly the family Kepler observed. Top-right: a single conceptual experiment -- replace $1/r^2$ with $1/r^3$, and the orbit no longer closes; it spirals into the centre (consistent with **Bertrand's theorem**: only $1/r^2$ and the harmonic potential give closed orbits universally). Bottom-right: a high-order numerical integrator preserves both conserved quantities to roughly $10^{-7}$ over many revolutions.*

**Bertrand's theorem** is one of those quietly amazing facts: out of all spherically symmetric potentials, exactly two ($\propto 1/r$ and $\propto r^2$) produce closed orbits for every bound trajectory. That is *why* a stable Solar System with closed planetary loops exists. Mathematics first, astronomy second.

---

## 4. Structural Vibration -- buildings, bridges, tuned mass dampers

A multi-storey building is, to first approximation, a chain of masses connected by stiff springs (the columns). For two storeys with masses $m_1, m_2$ and stiffnesses $k_1, k_2$:

$$
M\ddot{\mathbf x} + C\dot{\mathbf x} + K\mathbf x = \mathbf F(t),
\qquad
M = \begin{pmatrix} m_1 & 0 \\ 0 & m_2 \end{pmatrix},
\;
K = \begin{pmatrix} k_1+k_2 & -k_2 \\ -k_2 & k_2 \end{pmatrix}.
$$

**Modal analysis.** The eigenproblem $K\,\boldsymbol\phi = \omega^2 M\,\boldsymbol\phi$ yields natural frequencies $\omega_i$ and mode shapes $\boldsymbol\phi_i$. Decomposing $\mathbf x = \sum_i q_i(t)\,\boldsymbol\phi_i$ decouples the equations into $n$ independent SDOF oscillators -- the multi-degree-of-freedom problem becomes $n$ scalar problems.

![2-DOF building: mode shapes, free vibration, frequency response, resonant growth, and the effect of a tuned mass damper.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig4_structural_vibration.png)
*Left: the two mode shapes -- mode 1 is in-phase (whole building sways), mode 2 is out-of-phase (storeys move opposite each other). Middle column: time response after kicking floor 2 (top), and frequency response with two clean resonance peaks (bottom). Right column: top -- under resonant forcing without damping the amplitude grows linearly in time, the canonical *Tacoma Narrows* picture; bottom -- adding a small **tuned mass damper** (TMD) tuned to the offending mode splits the single tall peak into two small adjacent peaks, the standard skyscraper-stabilisation trick (the 660-ton sphere atop Taipei 101).*

The pattern is universal: whether the system is a violin string, a wing, a turbine blade or the Millennium Bridge, the steps are *form $M, C, K$ -> solve eigenproblem -> decouple via modes -> add damping where needed*.

---

## 5. Fluid Mechanics -- from steady pipes to vortex shedding

### Steady pipe flow (Poiseuille)

For incompressible viscous flow in a long circular pipe, the steady Navier-Stokes equations reduce to an ODE in the radial coordinate:

$$
\frac{1}{r}\frac{d}{dr}\!\Bigl(r\,\frac{du}{dr}\Bigr) = -\frac{1}{\mu}\frac{dp}{dz},
$$

with boundary conditions $u(R) = 0$ (no-slip) and $u(0)$ finite. The solution is the parabolic profile $u(r) = u_{\max}(1 - r^2/R^2)$, and total volume flow $Q = \pi R^4 \Delta p / (8\mu L)$ -- the Hagen-Poiseuille law. The fourth-power dependence on radius explains why arteries dilate to drop blood pressure: doubling radius eightfold-improves flow at the same pressure gradient.

### Reynolds number and the laminar-turbulent transition

The dimensionless number $\mathrm{Re} = \rho U L / \mu$ measures the ratio of inertial to viscous effects. Below $\mathrm{Re}\sim 2300$ flow is laminar and the linear $Q \propto \Delta p$ holds; above it the same pressure drop produces less flow because turbulent eddies dissipate energy.

### Vortex shedding (the Strouhal number)

Behind a bluff body in steady flow, alternating vortices shed at frequency $f$ such that the **Strouhal number** $\mathrm{St} = f D / U$ is roughly constant ($\approx 0.21$) over a wide range of Re. When that frequency matches a structural mode, you get the Tacoma resonance again.

### Settling sphere (Stokes drag)

A small dense sphere falling in viscous fluid satisfies

$$
m\dot v = (\rho_p - \rho_f) V g - 6\pi\mu R\,v,
$$

a first-order linear ODE with terminal velocity $v_t = (\rho_p - \rho_f) V g / (6\pi\mu R)$ and time constant $\tau = m/(6\pi\mu R)$.

![Fluid mechanics composite: parabolic Poiseuille profile, laminar/turbulent flow-rate scaling, Strouhal number across Re regimes, and Stokes settling.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig5_fluid_flow.png)
*Top-left: parabolic velocity profile from Hagen-Poiseuille. Top-right: $Q$ vs $\Delta p$ shows the linearity of laminar flow breaking into a $\sqrt{\Delta p}$ scaling once turbulence dominates. Bottom-left: Strouhal number across a huge Reynolds-number range -- the plateau near $0.21$ in the subcritical regime is the reason cables and chimneys oscillate at predictable frequencies. Bottom-right: spheres of three radii falling under gravity reach terminal velocity exponentially, with $v_t$ scaling as $R^2$.*

---

## 6. The Common Thread

Every domain in this chapter speaks the same five-step grammar:

1. **Write Newton's / Kirchhoff's / Navier-Stokes law** for an infinitesimal element.
2. **Reduce to ODEs** by symmetry, dimensional analysis, or projection onto modes.
3. **Identify the equilibrium** and linearise; find natural frequencies and damping ratios.
4. **Add forcing**, look for resonance, and decide whether to add damping or *de-tune* the source.
5. **Integrate numerically** when the nonlinearity bites (large amplitude, turbulence, multi-body).

Master this loop and a vast portion of physics and engineering is unified.

| Domain | Governing ODE | Key dimensionless number |
|---|---|---|
| Mechanics | $\ddot\theta + (g/L)\sin\theta = 0$ | -- |
| Electrical | $L\ddot q + R\dot q + q/C = V(t)$ | $Q = (1/R)\sqrt{L/C}$ |
| Orbital | $\ddot{\mathbf r} = -GM\mathbf r/|\mathbf r|^3$ | eccentricity $e$ |
| Structural | $M\ddot x + C\dot x + Kx = F$ | modal damping $\zeta_i$ |
| Fluid (pipe) | $\nabla \cdot \boldsymbol\sigma = 0$ -> radial ODE | Re |
| Fluid (wake) | -- (empirical) | St $\approx 0.21$ |

---

## Summary

Differential equations are the **lingua franca of physics**. You spent 16 chapters building the language; this chapter showed how a handful of canonical models -- pendulum, RLC, Kepler, MDOF building, pipe flow -- carry you from undergraduate physics to professional engineering practice. None of them required new mathematics. They required us to *recognise the structure* and reach for the right tool.

The next chapter, the finale, looks beyond this classical menu: Neural ODEs, stochastic and fractional equations, and the questions that connect ODE theory to modern machine learning.

---

## References

- Kreyszig, *Advanced Engineering Mathematics*, Wiley (2011).
- Taylor, *Classical Mechanics*, University Science Books (2005).
- Goldstein, Poole & Safko, *Classical Mechanics*, Pearson (2002).
- Den Hartog, *Mechanical Vibrations*, Dover (1985).
- Acheson, *Elementary Fluid Dynamics*, Oxford (1990).

---

**Previous Chapter**: [Chapter 16: Control Theory](/en/ode-chapter-16-control-theory/)

**Next Chapter**: [Chapter 18: Frontiers and Series Summary](/en/ode-chapter-18-advanced-topics-summary/)

*This is Part 17 of the 18-part series on Ordinary Differential Equations.*
