---
title: "ODE Chapter 6: Linear Systems and the Matrix Exponential"
date: 2024-08-04 09:00:00
tags:
  - ODE
  - Linear Systems
  - Matrix Exponential
  - Phase Space
  - Coupled Oscillators
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 6
  total: 18
lang: en
mathjax: true
description: "When multiple variables interact, you need systems of ODEs. Learn the matrix exponential, eigenvalue-based solutions, phase portrait classification, and applications to coupled oscillators and RLC circuits."
disableNunjucks: true
---
**One equation describes one quantity. The world is rarely that obliging.** Predator and prey populations push each other up and down. Currents and voltages in an RLC network oscillate together. Chemical species in a reaction network feed into one another. The moment two unknowns share an equation, you have a *system*, and a single $y'=ay$ is no longer enough.

The miracle of the linear case is this: the scalar formula $y(t)=e^{at}y_0$ generalizes verbatim once you learn what $e^{At}$ means for a *matrix* $A$. Linear algebra and ODEs fuse into one object — the matrix exponential — and its eigenstructure tells you everything about the long-term behavior, the geometry of the flow, and the physics of normal modes and beats.

## What You Will Learn

- Writing ODE systems in matrix form $\mathbf{x}'=A\mathbf{x}$ and why this is more than notation
- The matrix exponential $e^{At}$: its definition, its properties, and *three* ways to compute it
- The eigenvalue method, complex eigenvalues, and what to do when eigenvectors run out
- The full classification of 2D phase portraits and the trace–determinant stability map
- Non-homogeneous systems and Duhamel's formula
- Coupled oscillators: normal modes, beats, and energy transfer

## Prerequisites

- Linear algebra: eigenvalues, eigenvectors, change of basis
- Chapter 3: second-order linear ODEs (every such equation becomes a 2D system)

---

## 1. From Two Coupled Equations to One Vector Equation

Consider a deliberately simple ecology model: $x(t)$ is a rabbit population in suitable units, $y(t)$ a wolf population. Rabbits multiply on their own and are eaten by wolves; wolves grow only when rabbits are present:

$$x' = 2x - y, \qquad y' = x + 0.5\,y.$$

Stack the unknowns into a vector $\mathbf{x}=(x,y)^\top$ and the coefficients into a matrix:

$$\mathbf{x}' = A\mathbf{x}, \qquad A = \begin{pmatrix} 2 & -1 \\ 1 & 0.5 \end{pmatrix}.$$

This is not just notation. It is a *change of viewpoint*: a single trajectory $\mathbf{x}(t)$ in the plane replaces two coupled time series. Geometry takes over from algebra.

The scalar ODE $y'=ay$ has solution $y=e^{at}y_0$. Brute analogy suggests

$$\mathbf{x}(t) = e^{At}\,\mathbf{x}_0,$$

but this is only meaningful once we say what *the exponential of a matrix* is. That is the central object of the chapter.

---

## 2. The Matrix Exponential

### 2.1 Definition by power series

Mimicking $e^x = \sum x^k/k!$:

$$e^{At} \;=\; I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots \;=\; \sum_{k=0}^{\infty} \frac{(At)^k}{k!}.$$

This series converges in operator norm for *every* square matrix $A$ and every $t\in\mathbb{R}$, because $\|(At)^k/k!\| \le (\|A\|t)^k/k!$ and the scalar series converges.

![Partial sums of the matrix exponential applied to a vector, converging to the true rotation; the spectral-norm error decays super-exponentially in the number of terms.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/06-power-series/fig1_matrix_exponential_series.png)
*Figure 1. Left: partial sums $\sum_{k=0}^N (At)^k/k!\;\mathbf{x}_0$ for $A$ a rotation generator, converging to the unit circle. Right: spectral-norm error decays super-exponentially with the number of terms — this is why low-degree polynomial approximants like Padé work so well.*

### 2.2 The three properties you actually use

Once the series is in hand, three facts do almost all the work:

| Property | Why it matters |
|---|---|
| $e^{A\cdot 0} = I$ | Initial condition is satisfied automatically |
| $\dfrac{d}{dt}e^{At} = Ae^{At}$ | This is *exactly* what makes $\mathbf{x}(t)=e^{At}\mathbf{x}_0$ solve $\mathbf{x}'=A\mathbf{x}$ |
| $e^{At}e^{As}=e^{A(t+s)}$ | The flow is a one-parameter group; in particular $(e^{At})^{-1}=e^{-At}$ |

A subtle warning: $e^{A+B} = e^A e^B$ holds **only** when $AB=BA$. This is the matrix version of the fact that $\sin(x+y)\ne \sin x+\sin y$ — non-commutativity has consequences.

### 2.3 Three ways to compute it in practice

1. **Power series**, truncated. Cheap conceptually, terrible numerically when $\|At\|$ is large.
2. **Eigendecomposition.** If $A$ is diagonalizable as $A=PDP^{-1}$ with $D=\mathrm{diag}(\lambda_i)$, then
   $$e^{At} = P\,\mathrm{diag}(e^{\lambda_1 t},\dots,e^{\lambda_n t})\,P^{-1}.$$
   This is the structural formula every theoretical argument leans on.
3. **Padé with scaling and squaring.** The industrial method — used by `scipy.linalg.expm`, MATLAB's `expm`, etc. Compute $e^{At/2^s}$ from a Padé rational approximant, then square $s$ times. Robust for large $\|At\|$.

In code:

```python
import numpy as np
from scipy.linalg import expm

A = np.array([[2.0, -1.0], [1.0, 0.5]])
print(expm(A * 0.5))        # exp((0.5)A) -- uses scaling-and-squaring Padé

vals, P = np.linalg.eig(A)
expm_via_eig = P @ np.diag(np.exp(vals * 0.5)) @ np.linalg.inv(P)
print(expm_via_eig.real)    # same matrix (modulo floating-point noise)
```

---

## 3. The Eigenvalue Method

The eigendecomposition formula has a clean re-statement that avoids matrix exponentials altogether.

> **Theorem.** If $A$ has eigenpairs $(\lambda_i,\mathbf{v}_i)$ with linearly independent $\mathbf{v}_i$, the general solution of $\mathbf{x}'=A\mathbf{x}$ is
> $\mathbf{x}(t) = c_1 e^{\lambda_1 t}\mathbf{v}_1 + c_2 e^{\lambda_2 t}\mathbf{v}_2 + \cdots + c_n e^{\lambda_n t}\mathbf{v}_n.$

The proof is a one-liner: for each eigenpair, $\frac{d}{dt}\bigl(e^{\lambda t}\mathbf{v}\bigr)=\lambda e^{\lambda t}\mathbf{v}=A\bigl(e^{\lambda t}\mathbf{v}\bigr)$. Linearity finishes the job.

### 3.1 Geometry: eigenvectors are the natural axes

The decomposition $A=PDP^{-1}$ has a simple geometric reading. Apply $A$ to a vector by:

1. **$P^{-1}$**: re-express the vector in the eigenbasis;
2. **$D$**: stretch along each eigen-axis by the corresponding $\lambda_i$;
3. **$P$**: re-express the stretched vector in the original basis.

The flow $e^{At}$ does the same thing but with $e^{\lambda_i t}$ stretches.

![Geometric eigendecomposition: the unit circle goes to its image under A by rotating into the eigenbasis, scaling along each eigenvector, then rotating back.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/06-power-series/fig3_eigenvalue_decomposition.png)
*Figure 2. The factorization $A=PDP^{-1}$ as three geometric steps. Eigenvectors are the natural axes in which a linear map (and its flow) is just diagonal scaling.*

### 3.2 Complex eigenvalues give rotations

Real matrices can have complex eigenvalues, and they always come in conjugate pairs $\lambda = \alpha\pm\beta i$ with conjugate eigenvectors $\mathbf{v}=\mathbf{a}\pm i\mathbf{b}$. Taking real and imaginary parts of $e^{\lambda t}\mathbf{v}$ produces two real solutions:

$$\mathbf{x}_1(t) = e^{\alpha t}(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t), \qquad \mathbf{x}_2(t) = e^{\alpha t}(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t).$$

Read this geometrically: $\beta$ is the angular frequency of rotation in the $(\mathbf a,\mathbf b)$-plane, and $\alpha$ controls whether the orbit spirals outward ($\alpha>0$), inward ($\alpha<0$), or stays on a closed curve ($\alpha=0$).

---

## 4. Phase Portraits in 2D

For $\mathbf{x}'=A\mathbf{x}$ on the plane, the eigenvalues $\lambda_1,\lambda_2$ determine *everything* about the local geometry near the origin. The taxonomy is short and worth memorizing.

| Eigenvalues | Portrait | Stability |
|---|---|---|
| $\lambda_1<\lambda_2<0$ (real) | **Stable node** — all orbits enter the origin | Asymptotically stable |
| $0<\lambda_1<\lambda_2$ (real) | **Unstable node** — all orbits flee | Unstable |
| $\lambda_1<0<\lambda_2$ (real) | **Saddle** — two stable, two unstable directions | Unstable |
| $\alpha\pm\beta i$, $\alpha<0$ | **Stable spiral** | Asymptotically stable |
| $\alpha\pm\beta i$, $\alpha>0$ | **Unstable spiral** | Unstable |
| $\pm\beta i$ (purely imaginary) | **Center** — closed orbits | Lyapunov stable, *not* asymptotic |
| $\lambda_1=\lambda_2$, two eigenvectors | **Star node** | Sign of $\lambda$ |
| $\lambda_1=\lambda_2$, one eigenvector | **Degenerate node** | Sign of $\lambda$ |

![Phase portraits of the six canonical 2D linear systems, with eigenvector lines colored by stability and arrows showing the direction field.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/06-power-series/fig2_phase_portrait_zoo.png)
*Figure 3. The phase portrait zoo. Green lines mark stable eigen-directions, red mark unstable. The eigenvalues alone determine which picture you get — the rest is just rotation and rescaling of these six prototypes.*

### 4.1 The trace–determinant trick

You don't always need the eigenvalues themselves; for a $2\times 2$ matrix the characteristic polynomial is

$$\lambda^2 - \tau\lambda + \delta = 0, \qquad \tau = \mathrm{tr}\,A = \lambda_1+\lambda_2, \quad \delta = \det A = \lambda_1\lambda_2,$$

so the discriminant is $\Delta = \tau^2 - 4\delta$. The position of $(\tau,\delta)$ in the plane already classifies the equilibrium:

- $\delta < 0$ → real eigenvalues of opposite sign → **saddle**.
- $\delta > 0,\ \Delta > 0$ → real same-sign eigenvalues → **node** (sign of $\tau$ gives stability).
- $\delta > 0,\ \Delta < 0$ → complex eigenvalues → **spiral** (sign of $\tau$ gives stability).
- $\delta > 0,\ \tau = 0$ → purely imaginary → **center**.
- The parabola $\Delta = 0$ is the locus of repeated eigenvalues.

![Stability map in the trace-determinant plane: parabola Delta=0 separates nodes from spirals, the axis det=0 marks saddles, and tr=0 with det>0 marks centers.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/06-power-series/fig4_trace_determinant_plane.png)
*Figure 4. The stability map. A single point $(\tau,\delta)$ tells you the equilibrium type without ever computing the eigenvalues. Centers — the line $\tau=0,\ \delta>0$ — are non-generic: an arbitrarily small perturbation pushes them into spirals.*

---

## 5. Repeated Eigenvalues and Generalized Eigenvectors

The classification table above quietly hides a complication. A repeated eigenvalue $\lambda$ with algebraic multiplicity $2$ may have either *two* linearly independent eigenvectors (rare — this happens iff $A=\lambda I$ on that subspace; the result is a **star node**) or only *one* (the generic defective case — a **degenerate node**).

In the defective case, the eigenvalue method gives only one solution $e^{\lambda t}\mathbf{v}$. The trick: solve

$$(A - \lambda I)\,\mathbf{w} = \mathbf{v}$$

for a **generalized eigenvector** $\mathbf{w}$. Then

$$\mathbf{x}_2(t) \;=\; e^{\lambda t}\bigl(t\,\mathbf{v} + \mathbf{w}\bigr)$$

is a second, linearly independent solution. The polynomial-times-exponential growth of the $t$ factor is the algebraic fingerprint of degeneracy.

This is exactly the Jordan-block phenomenon: $A$ acts like $\lambda I + N$ where $N$ is nilpotent, and $e^{(\lambda I + N)t} = e^{\lambda t}(I + tN + \tfrac12 t^2 N^2 + \cdots)$. The series terminates because $N$ is nilpotent, leaving a polynomial in $t$.

![A defective 2x2 system: the flow shears toward the single eigen-direction, and the second independent solution acquires a t e^{lambda t} component.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/06-power-series/fig6_repeated_eigenvalue_shear.png)
*Figure 5. Left: the defective system $\dot{\mathbf{x}}=\bigl(\begin{smallmatrix}-1&1\\0&-1\end{smallmatrix}\bigr)\mathbf{x}$. The single eigen-direction (green) is the only one orbits can come in along; the generalized direction (red dashed) is where the shear $t\mathbf v$ term lives. Right: the components of the two basis solutions — note the rising-then-decaying $t e^{\lambda t}$ shape.*

---

## 6. Non-Homogeneous Systems: Duhamel's Formula

Add a forcing term:

$$\mathbf{x}' = A\mathbf{x} + \mathbf{g}(t), \qquad \mathbf{x}(0)=\mathbf{x}_0.$$

The solution is the matrix version of variation of parameters:

$$\boxed{\;\mathbf{x}(t) \;=\; e^{At}\mathbf{x}_0 \;+\; \int_0^t e^{A(t-\tau)}\,\mathbf{g}(\tau)\,d\tau.\;}$$

This is **Duhamel's formula**. Read the integrand $e^{A(t-\tau)}\mathbf{g}(\tau)$ this way: at time $\tau$ the forcing kicks the system by $\mathbf{g}(\tau)d\tau$, and that kick then propagates freely under the flow $e^{A(t-\tau)}$ for the remaining time. The full state is the superposition of all such delayed responses — a continuous-time impulse response.

In control theory the same formula appears with $\mathbf{g}(t) = B\mathbf u(t)$:

$$\mathbf{x}(t) = e^{At}\mathbf{x}_0 + \int_0^t e^{A(t-\tau)} B\,\mathbf u(\tau)\,d\tau,$$

and we are one short step from controllability and the matrix $\bigl[B,\,AB,\,A^2B,\dots\bigr]$.

---

## 7. Application: Coupled Oscillators, Normal Modes, and Beats

Two equal masses on a line, each tied to a wall by a spring of stiffness $k$ and to each other by a coupling spring of stiffness $\kappa$:

```
Wall —/\/\/— [m] —/\/\/— [m] —/\/\/— Wall
       k        κ        k
```

Newton's laws give

$$\ddot x_1 = -k\,x_1 + \kappa(x_2 - x_1), \qquad \ddot x_2 = -k\,x_2 + \kappa(x_1 - x_2).$$

A clean change of variables is $u=\tfrac{x_1+x_2}{\sqrt 2}$, $v=\tfrac{x_1-x_2}{\sqrt 2}$. The equations *decouple*:

$$\ddot u = -k\,u, \qquad \ddot v = -(k+2\kappa)\,v.$$

These are the **normal modes**: an in-phase mode at angular frequency $\omega_s=\sqrt{k}$ (the coupling spring never stretches) and an out-of-phase mode at $\omega_a=\sqrt{k+2\kappa}$ (the coupling spring is doing extra work). Every motion is a superposition of the two.

The drama happens when the modes are nearly degenerate ($\kappa \ll k$): displacing only mass 1 excites both modes with equal amplitude, and their slow relative dephasing causes energy to slosh entirely from mass 1 into mass 2 and back. These are **beats** — the audible rise and fall when two slightly mistuned tuning forks sound together.

![Three time-series stacked: pure symmetric mode, pure antisymmetric mode, and the beat pattern arising when only one mass is initially displaced.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/06-power-series/fig5_coupled_oscillator_modes.png)
*Figure 6. Symmetric mode (top): both masses move in phase at $\omega_s$. Antisymmetric mode (middle): they move out of phase at the higher frequency $\omega_a$. Generic initial condition (bottom): the two modes are slightly mistuned and their interference produces beats — energy moves periodically from one mass to the other under the dotted envelope $\cos\bigl(\tfrac{\omega_a-\omega_s}{2}t\bigr)$.*

This is the same mathematics as Rabi oscillations between two quantum states, the swap of energy in coupled LC circuits, and the synchronization of weakly coupled pendulum clocks.

---

## 8. Stability: The Eigenvalue Criterion

For the linear flow $\mathbf{x}'=A\mathbf{x}$, the long-time behavior is decided entirely by the spectrum of $A$:

> **Theorem.**
> - All $\mathrm{Re}(\lambda) < 0\ \Longrightarrow\ e^{At}\to 0$, and the origin is **asymptotically stable**.
> - Any $\mathrm{Re}(\lambda) > 0\ \Longrightarrow$ origin is **unstable**.
> - The borderline $\mathrm{Re}(\lambda) = 0$ requires a closer look (centers, Jordan blocks).

The *liouville formula* sharpens this: $\det \Phi(t) = \det \Phi(0)\,\exp\bigl(\int_0^t \mathrm{tr}\,A\,d\tau\bigr)$. So $\mathrm{tr}\,A < 0$ means phase-space *volumes* contract (a dissipative system), $\mathrm{tr}\,A = 0$ means volumes are preserved (the Hamiltonian case), and $\mathrm{tr}\,A > 0$ means they expand.

Chapter 7 will lift this entire theory to nonlinear systems via linearization at fixed points — the **Hartman–Grobman** theorem says that, away from the borderline cases, the local picture of a nonlinear system *is* the picture of its Jacobian.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Vector form $\mathbf{x}'=A\mathbf{x}$ | Solution is $e^{At}\mathbf{x}_0$ |
| Matrix exponential | Power series; computed via eigendecomposition or Padé scaling-and-squaring |
| Eigenvalue method | $\sum c_k e^{\lambda_k t}\mathbf{v}_k$ — linear combinations of eigenmodes |
| Complex eigenvalues | Real solutions are $e^{\alpha t}\bigl(\cos\beta t,\ \sin\beta t\bigr)$-style — spirals |
| Repeated, defective | Generalized eigenvector adds a $te^{\lambda t}$ solution |
| Phase portraits | Eigenvalues $\to$ node / spiral / saddle / center; trace-det map summarizes |
| Non-homogeneous | Duhamel: $\mathbf{x}=e^{At}\mathbf{x}_0+\int_0^t e^{A(t-\tau)}\mathbf{g}(\tau)d\tau$ |
| Stability | All $\mathrm{Re}(\lambda)<0\ \Leftrightarrow$ asymptotically stable |

---

## Exercises

**Basic**

1. Compute $e^{At}$ for $A=\bigl(\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\bigr)$ from the power series *and* by eigendecomposition. Verify they agree.
2. Solve $\mathbf{x}'=\bigl(\begin{smallmatrix}1&2\\0&3\end{smallmatrix}\bigr)\mathbf{x}$ with $\mathbf{x}(0)=(1,1)^\top$.
3. Classify the origin for (a) $A=\bigl(\begin{smallmatrix}-1&2\\-2&-1\end{smallmatrix}\bigr)$ and (b) $A=\bigl(\begin{smallmatrix}1&0\\0&-2\end{smallmatrix}\bigr)$.

**Advanced**

4. Prove $\det e^A = e^{\mathrm{tr}\,A}$. *Hint:* upper-triangularize $A$ over $\mathbb C$ (Schur decomposition).
5. Find the normal-mode frequencies of three equal masses in a line, each connected to its neighbour and to the walls by springs of stiffness $k$. Identify the symmetric, antisymmetric, and "breathing" modes.
6. Give a $2\times 2$ counterexample to $e^{A+B}=e^A e^B$ when $AB\ne BA$.

**Programming**

7. Write a phase-portrait plotter that, given any $2\times 2$ matrix, automatically labels the equilibrium type using the trace–determinant test.
8. Simulate the coupled-oscillator system for varying coupling $\kappa$ and plot the beat *period* $T_{\text{beat}} = 2\pi/(\omega_a - \omega_s)$ as a function of $\kappa$. Compare against the small-$\kappa$ approximation.

---

## References

- Hirsch, Smale, & Devaney, *Differential Equations, Dynamical Systems, and an Introduction to Chaos*, Academic Press (2012)
- Strang, *Linear Algebra and Learning from Data*, Wellesley-Cambridge (2019)
- Moler & Van Loan, "Nineteen Dubious Ways to Compute the Exponential of a Matrix," *SIAM Review* 45(1) (2003)
- Perko, *Differential Equations and Dynamical Systems*, Springer (2001)

---

**Series Navigation**

| | |
|---|---|
| **Previous** | [Chapter 5: Power Series and Special Functions](/en/ode-chapter-05-laplace-transform/) |
| **Current** | Chapter 6: Linear Systems and the Matrix Exponential |
| **Next** | [Chapter 7: Stability Theory](/en/ode-chapter-07-systems-and-phase-plane/) |
