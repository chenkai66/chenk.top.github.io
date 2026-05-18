---
title: "Ordinary Differential Equations (17): Physics and Engineering Applications"
date: 2024-03-29 09:00:00
tags:
  - Ordinary Differential Equations
  - Pendulum
  - RLC Circuit
  - Kepler Orbit
  - Vibration Analysis
  - Fluid Mechanics
  - Python
categories: Ordinary Differential Equations
series: ode
lang: en
mathjax: true
description: "See ODEs in action across physics and engineering. Walk through the nonlinear pendulum, RLC circuit and resonance, Kepler orbits and conservation laws, multi-DOF structural vibration with tuned mass dampers, and fluid flow from Poiseuille to vortex shedding -- all with full Python simulations."
disableNunjucks: true
series_order: 17
series_total: 18
translationKey: "ode-17"
---
**Differential equations are not a pure mathematical game — they are the language for understanding the physical world.** From celestial motion to circuit response, from a swinging pendulum to vortex shedding behind a bridge cable, every dynamical system "speaks" ODE.

This chapter is a deliberate tour through five canonical applications. Each one will pay back the entire ODE toolkit we built in chapters 1-16: phase planes, eigenvalues, Laplace transforms, modal analysis, conservation laws, numerical integration, control. None of the examples is a "toy" — they are all genuine working physics, written tightly so that the structure remains visible.

![Ordinary Differential Equations (17): Physics and Engineering Applications — Chapter overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/illustration_1.png)

---

## What You Will Learn

- The nonlinear pendulum: small-angle linearisation vs the full equation, period vs amplitude, the separatrix
- RLC circuits: the three damping regimes, resonance, the Q-factor, and tuning
- Kepler orbits: gravity, eccentricity, conservation of energy and angular momentum, Bertrand's theorem
- Multi-DOF structural vibration: modal analysis, resonance, tuned mass dampers (the Tacoma lesson)
- Fluid mechanics: Poiseuille flow, Reynolds number, vortex shedding, Stokes settling

## Prerequisites

- Chapters 1-3 — first- and second-order linear theory
- [Chapter 6](/en/ode/06-power-series/)-7 — linear systems and phase portraits
- [Chapter 8](/en/ode/08-nonlinear-stability/) — nonlinear stability and energy methods
- [Chapter 11](/en/ode/11-numerical-methods/) — numerical methods (used throughout)

---

## The Nonlinear Pendulum — the Hello World of nonlinear ODEs

A point mass on a rigid rod of length $L$ in gravity $g$ obeys
$$\ddot\theta + \frac{g}{L}\sin\theta = 0.$$
For small $\theta$ we replace $\sin\theta \approx \theta$ and recover simple harmonic motion with period $T_0 = 2\pi\sqrt{L/g}$ — *independent* of amplitude. That isochronism is the small-angle miracle Galileo exploited.

The full equation is *not* isochronous: the period depends on amplitude. The exact answer involves the complete elliptic integral of the first kind:
$$T(\theta_0) = \frac{4}{\omega_0}\,K\!\bigl(\sin(\theta_0/2)\bigr), \qquad \omega_0 = \sqrt{g/L}.$$
![Pendulum: large-angle vs linearised time histories, period vs amplitude, and the global phase portrait with libration / separatrix / rotation regions.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig1_pendulum_motion.png)
*Top-left: from the same large initial angle ($\theta_0 = 1.5$ rad), the linear and full equations agree for one half-period and then drift apart — the linear model loses *phase* first, then amplitude. Top-right: $T(\theta_0)$ from numerical integration; below 0.5 rad the small-angle constant period is essentially perfect, but as $\theta_0 \to \pi$ the period diverges because the unstable equilibrium has been reached. Bottom: the full phase portrait $(\theta, \dot\theta)$. Inside the separatrix (red) the pendulum swings (libration); outside it goes over the top (rotation). The pattern repeats $2\pi$-periodically along $\theta$.*

```python
from scipy.integrate import solve_ivp
import numpy as np

g, L = 9.81, 1.0

def pendulum(t, s):
    theta, omega = s
    return [omega, -(g/L)*np.sin(theta)]

for v0 in [1.5, 4.0, 6.5, 7.5]:    # last two go over the top
    sol = solve_ivp(pendulum, (0, 10), [0.0, v0],
                    t_eval=np.linspace(0, 10, 2000),
                    rtol=1e-10, atol=1e-12)
    # examine sol.y -- you will see closed loops or monotone theta
```

**Why this matters.** The pendulum is the simplest non-trivial nonlinear conservative system, and its phase portrait already contains every concept of Chapters 7-8: stable/unstable equilibria, separatrices, period-energy relations, integrability. Every other oscillator — molecular vibrations, Josephson junctions, plasma waves — borrows this skeleton.

---

## RLC Circuits — the same equation, in copper

![Ordinary Differential Equations (17): Physics and Engineering Applications — Chapter summary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/illustration_2.png)

A series resistor-inductor-capacitor loop driven by voltage $V(t)$ obeys, for charge $q$ on the capacitor:
$$L\ddot q + R\dot q + \frac{1}{C}\,q = V(t).$$
Compare with the mass-spring-damper $m\ddot x + c\dot x + kx = F(t)$. They are the *same* equation — so all our intuition about damping ratio and natural frequency carries over directly.

Define $\omega_n = 1/\sqrt{LC}$ (natural frequency) and $\zeta = (R/2)\sqrt{C/L}$ (damping ratio). The transfer function from $V$ to $q$ is
$$H(s) = \frac{1/L}{s^2 + (R/L)\,s + 1/(LC)},$$
with the same three step-response regimes from [Chapter 16](/en/ode/16-control-theory/): underdamped, critical, overdamped.

![Series RLC circuit: schematic, step responses, frequency response, and the Q-factor / bandwidth trade-off.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig2_rlc_circuit.png)
*Top-left: the canonical series RLC. Top-right: step responses for three damping ratios; the underdamped case rings around the steady value, critical lands fastest without overshoot, overdamped is sluggish. Bottom-left: $|H(j\omega)|$ has a sharp resonant peak when $\zeta$ is small. Bottom-right: as $R$ rises, the Q factor $Q = (1/R)\sqrt{L/C}$ falls and the 3-dB bandwidth $\Delta\omega = R/L$ widens — a fundamental trade between selectivity and speed.*

The peak amplitude of an undamped oscillator driven at resonance grows linearly in time — a phenomenon the Tacoma Narrows Bridge made historic in 1940 (more on that in section 4).

---

## Planetary Orbits — where Newton meets Kepler

Newton's gravitational law gives, in two dimensions,
$$\ddot{\mathbf r} \;=\; -\frac{GM\,\mathbf r}{|\mathbf r|^3}.$$
The conserved quantities are total energy $E = \tfrac12|\mathbf v|^2 - GM/|\mathbf r|$ and angular momentum $\mathbf L = \mathbf r \times \mathbf v$. Together they prove Kepler's three laws — without solving the ODE in closed form.

```python
from scipy.integrate import solve_ivp
import numpy as np

GM = 1.0
def kepler(t, s):
    x, y, vx, vy = s
    r = np.hypot(x, y)
    return [vx, vy, -GM*x/r**3, -GM*y/r**3]

for v0 in [1.0, 0.85, 0.55]:
    sol = solve_ivp(kepler, (0, 15), [1, 0, 0, v0],
                    t_eval=np.linspace(0, 15, 4000),
                    method='DOP853', rtol=1e-10)
    # sol.y[0:2] traces the orbit
```

![Kepler orbits at different initial speeds; inverse-square vs inverse-cube collapse; numerical conservation of energy and angular momentum.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig3_planetary_orbits.png)
*Left: starting at $(1,0)$ with varying tangential velocity, the orbit transitions from a circle to a thin ellipse — exactly the family Kepler observed. Top-right: a single conceptual experiment — replace $1/r^2$ with $1/r^3$, and the orbit no longer closes; it spirals into the centre (consistent with **Bertrand's theorem**: only $1/r^2$ and the harmonic potential give closed orbits universally). Bottom-right: a high-order numerical integrator preserves both conserved quantities to roughly $10^{-7}$ over many revolutions.*

**Bertrand's theorem** is one of those quietly amazing facts: out of all spherically symmetric potentials, exactly two ($\propto 1/r$ and $\propto r^2$) produce closed orbits for every bound trajectory. That is *why* a stable Solar System with closed planetary loops exists. Mathematics first, astronomy second.

---

## Structural Vibration — buildings, bridges, tuned mass dampers

A multi-storey building is, to first approximation, a chain of masses connected by stiff springs (the columns). For two storeys with masses $m_1, m_2$ and stiffnesses $k_1, k_2$:
$$
M\ddot{\mathbf x} + C\dot{\mathbf x} + K\mathbf x = \mathbf F(t),
\qquad
M = \begin{pmatrix} m_1 & 0 \\ 0 & m_2 \end{pmatrix},
\;
K = \begin{pmatrix} k_1+k_2 & -k_2 \\ -k_2 & k_2 \end{pmatrix}.
$$
**Modal analysis.** The eigenproblem $K\,\boldsymbol\phi = \omega^2 M\,\boldsymbol\phi$ yields natural frequencies $\omega_i$ and mode shapes $\boldsymbol\phi_i$. Decomposing $\mathbf x = \sum_i q_i(t)\,\boldsymbol\phi_i$ decouples the equations into $n$ independent SDOF oscillators — the multi-degree-of-freedom problem becomes $n$ scalar problems.

![2-DOF building: mode shapes, free vibration, frequency response, resonant growth, and the effect of a tuned mass damper.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig4_structural_vibration.png)
*Left: the two mode shapes — mode 1 is in-phase (whole building sways), mode 2 is out-of-phase (storeys move opposite each other). Middle column: time response after kicking floor 2 (top), and frequency response with two clean resonance peaks (bottom). Right column: top — under resonant forcing without damping the amplitude grows linearly in time, the canonical *Tacoma Narrows* picture; bottom — adding a small **tuned mass damper** (TMD) tuned to the offending mode splits the single tall peak into two small adjacent peaks, the standard skyscraper-stabilisation trick (the 660-ton sphere atop Taipei 101).*

The pattern is universal: whether the system is a violin string, a wing, a turbine blade or the Millennium Bridge, the steps are *form $M, C, K$ -> solve eigenproblem -> decouple via modes -> add damping where needed*.

---

## Fluid Mechanics — from steady pipes to vortex shedding

### Steady pipe flow (Poiseuille)

For incompressible viscous flow in a long circular pipe, the steady Navier-Stokes equations reduce to an ODE in the radial coordinate:
$$\frac{1}{r}\frac{d}{dr}\!\Bigl(r\,\frac{du}{dr}\Bigr) = -\frac{1}{\mu}\frac{dp}{dz},$$
with boundary conditions $u(R) = 0$ (no-slip) and $u(0)$ finite. The solution is the parabolic profile $u(r) = u_{\max}(1 - r^2/R^2)$, and total volume flow $Q = \pi R^4 \Delta p / (8\mu L)$ — the Hagen-Poiseuille law. The fourth-power dependence on radius explains why arteries dilate to drop blood pressure: doubling radius eightfold-improves flow at the same pressure gradient.

### Reynolds number and the laminar-turbulent transition

The dimensionless number $\mathrm{Re} = \rho U L / \mu$ measures the ratio of inertial to viscous effects. Below $\mathrm{Re}\sim 2300$ flow is laminar and the linear $Q \propto \Delta p$ holds; above it the same pressure drop produces less flow because turbulent eddies dissipate energy.

### Vortex shedding (the Strouhal number)

Behind a bluff body in steady flow, alternating vortices shed at frequency $f$ such that the **Strouhal number** $\mathrm{St} = f D / U$ is roughly constant ($\approx 0.21$) over a wide range of Re. When that frequency matches a structural mode, you get the Tacoma resonance again.

### Settling sphere (Stokes drag)

A small dense sphere falling in viscous fluid satisfies
$$m\dot v = (\rho_p - \rho_f) V g - 6\pi\mu R\,v,$$
a first-order linear ODE with terminal velocity $v_t = (\rho_p - \rho_f) V g / (6\pi\mu R)$ and time constant $\tau = m/(6\pi\mu R)$.

![Fluid mechanics composite: parabolic Poiseuille profile, laminar/turbulent flow-rate scaling, Strouhal number across Re regimes, and Stokes settling.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/17-physics-engineering-applications/fig5_fluid_flow.png)
*Top-left: parabolic velocity profile from Hagen-Poiseuille. Top-right: $Q$ vs $\Delta p$ shows the linearity of laminar flow breaking into a $\sqrt{\Delta p}$ scaling once turbulence dominates. Bottom-left: Strouhal number across a huge Reynolds-number range — the plateau near $0.21$ in the subcritical regime is the reason cables and chimneys oscillate at predictable frequencies. Bottom-right: spheres of three radii falling under gravity reach terminal velocity exponentially, with $v_t$ scaling as $R^2$.*

---

## The Common Thread

Every domain in this chapter speaks the same five-step grammar:

1. **Write Newton's / Kirchhoff's / Navier-Stokes law** for an infinitesimal element.
2. **Reduce to ODEs** by symmetry, dimensional analysis, or projection onto modes.
3. **Identify the equilibrium** and linearise; find natural frequencies and damping ratios.
4. **Add forcing**, look for resonance, and decide whether to add damping or *de-tune* the source.
5. **Integrate numerically** when the nonlinearity bites (large amplitude, turbulence, multi-body).

Master this loop and a vast portion of physics and engineering is unified.

| Domain | Governing ODE | Key dimensionless number |
|---|---|---|
| Mechanics | $\ddot\theta + (g/L)\sin\theta = 0$ | — |
| Electrical | $L\ddot q + R\dot q + q/C = V(t)$ | $Q = (1/R)\sqrt{L/C}$ |
| Orbital | $\ddot{\mathbf r} = -GM\mathbf r/\lvert \mathbf r\rvert^3$ | eccentricity $e$ |
| Structural | $M\ddot x + C\dot x + Kx = F$ | modal damping $\zeta_i$ |
| Fluid (pipe) | $\nabla \cdot \boldsymbol\sigma = 0$ -> radial ODE | Re |
| Fluid (wake) | — (empirical) | St $\approx 0.21$ |

---

## Worked Example with Units: Designing a Tuned RLC Bandpass

A symbolic equation is mathematics; an equation with units attached is engineering. Design a series RLC bandpass with centre frequency $f_0 = 1\,\text{kHz}$ and bandwidth $\Delta f = 100\,\text{Hz}$.

**Step 1: spec to coefficients.**
$$ \omega_0 = 2\pi f_0 = 6283\ \text{rad/s},\quad Q = \frac{f_0}{\Delta f} = 10. $$
For $L\ddot q + R\dot q + q/C = V(t)$, $\omega_0 = 1/\sqrt{LC}$ and $Q = \omega_0 L/R$.

**Step 2: pick components.** Three unknowns, three constraints — but we have only two physical specs, so fix $L = 10\,\text{mH}$ (a stocked inductor value).
$$ C = \frac{1}{\omega_0^2 L} = \frac{1}{(6283)^2 \cdot 0.01} \approx 2.53\,\mu\text{F}. $$

$$ R = \frac{\omega_0 L}{Q} = \frac{6283 \cdot 0.01}{10} = 6.28\,\Omega. $$
**Step 3: check units.** $\omega_0^2 L C = (\text{s}^{-2})(\text{H})(\text{F}) = (\text{s}^{-2})(\text{V}\cdot\text{s/A})(\text{A}\cdot\text{s/V}) = 1$. Dimensionless, correct.

**Step 4: physical reading.** $Q = 10$ means after the source disconnects, oscillation amplitude takes about $Q/\pi \approx 3.2$ periods to fall by $1/e$ — about 3.2 ms of ringing. In math language, damping ratio $\zeta = 1/(2Q) = 0.05$, deep underdamped.

**Step 5: energy view.** At resonance, peak inductor energy $\tfrac12 L I_{\max}^2$ swaps periodically with peak capacitor energy $\tfrac12 C V_{\max}^2$, with $\pi/Q$ of it lost to $R$ each cycle. Higher $Q$ means narrower, sharper peak.

Five steps and four numbers turn an abstract second-order ODE into something I can put on a breadboard.

## Limit Cases: Small Parameters Make Equations Degenerate

The nonlinear pendulum $\ddot\theta + (g/\ell)\sin\theta = 0$ degenerates to the harmonic oscillator as $|\theta| \to 0$, with amplitude-independent period $2\pi\sqrt{\ell/g}$. Standard small-angle story.

What is more useful in the ML era is the other direction: **parameters going to 0 or infinity often make the equation worse for numerical solvers, not better**.

**Case 1: high Reynolds number.** As viscosity $\nu \to 0$ in Navier-Stokes, the equation formally reduces to Euler, but in practice almost all smooth solutions destabilise into turbulence. Numerically, small $\nu$ blows up the grid Péclet number; you must switch to upwind schemes or shock capturing.

**Case 2: stiff spring limit.** Spring constant $k \to \infty$ effectively constrains the mass to a manifold. Direct integration demands $h \sim 1/\sqrt k$ — unaffordable. The physical fix: rewrite as a constrained ODE/DAE with Lagrange multipliers. `scipy.integrate.solve_ivp` with `LSODA` or `BDF` survives moderate stiffness; beyond that you need Hairer-Wanner-style implicit symplectic methods.

**Case 3: low Mach number.** Compressible flow at $M \to 0$ has acoustic and convective timescales separated by $1/M$. Explicit CFL is dominated by sound waves and wastes 99% of the work. Low-Mach preconditioning is a core trick of industrial CFD.

Lesson: **a parameter going to zero rarely simplifies the numerics**. This is the contradiction Neural Operators in PDE-ML chapter 2 are trying to dodge.

## Data-Driven Modelling: SINDy and Koopman

[Section 6](#structural-vibration--buildings-bridges-tuned-mass-dampers) made the case that one ODE skeleton describes five phenomena. In real engineering, when a new system shows up the equation may not be derivable. You go from data to equation.

**SINDy (Sparse Identification of Nonlinear Dynamics).** Brunton et al. (2016). Given a time series $\{x(t_i)\}$, numerically differentiate to get $\dot x$. Build a feature library $\Theta(x) = [1, x, x^2, \sin x, \dots]$ and solve
$$ \dot x = \Theta(x)\,\xi,\quad \min \|\dot x - \Theta(x)\xi\|_2^2 + \lambda \|\xi\|_1. $$
Most components of $\xi$ collapse to zero; the survivors give you the equation form. With `pysindy`, 1000 samples of Lorenz data recover the three RHS terms with their coefficients $\sigma, \rho, \beta$ — provided the library includes the correct basis.

**Koopman operator methods.** Lift a nonlinear map $x_{t+1} = F(x_t)$ to a linear operator $\mathcal{K}\,g = g \circ F$ on observable functions. Combined with DMD (Dynamic Mode Decomposition), you read off dominant oscillation modes and decay constants. Schmid's *DMD with Control* has been applied to PIV data of Kármán vortex streets.

**When to use which:**

- Clean data, interpretable variables, want an equation → SINDy.
- Noisy data, want forecasting/control, equation form irrelevant → Koopman/DMD.
- Both → SINDy on a Koopman-invariant subspace (Otto & Rowley, 2019).

## Summary

Differential equations are the **lingua franca of physics**. You spent 16 chapters building the language; this chapter showed how a handful of canonical models — pendulum, RLC, Kepler, MDOF building, pipe flow — carry you from undergraduate physics to professional engineering practice. None of them required new mathematics. They required us to *recognise the structure* and reach for the right tool.

The next chapter, the finale, looks beyond this classical menu: Neural ODEs, stochastic and fractional equations, and the questions that connect ODE theory to modern machine learning.

---

## References

- Kreyszig, *Advanced Engineering Mathematics*, Wiley (2011).
- Taylor, *Classical Mechanics*, University Science Books (2005).
- Goldstein, Poole & Safko, *Classical Mechanics*, Pearson (2002).
- Den Hartog, *Mechanical Vibrations*, Dover (1985).
- Acheson, *Elementary Fluid Dynamics*, Oxford (1990).

---

**Previous Chapter**: [Chapter 16: Control Theory](/en/ode/16-control-theory/)

**Next Chapter**: [Chapter 18: Frontiers and Series Summary](/en/ode/18-advanced-topics-summary/)

*This is Part 17 of the 18-part series on Ordinary Differential Equations.*
