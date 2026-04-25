---
title: "ODE Chapter 3: Higher-Order Linear Theory"
date: 2023-08-04 09:00:00
tags:
  - ODE
  - Higher-Order ODE
  - Characteristic Equation
  - Spring Oscillation
  - RLC Circuit
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 3
  total: 18
lang: en
mathjax: true
description: "From springs to RLC circuits, the full theory of higher-order linear ODEs: superposition, the Wronskian, characteristic equations, undetermined coefficients, variation of parameters, and the resonance phenomenon."
disableNunjucks: true
series_order: 3
---

**A first-order ODE has memory of one number; a second-order ODE has memory of two.** That tiny extra degree of freedom is what lets the same equation describe a plucked guitar string, the suspension of your car, the L-C tank circuit inside an FM radio, and the swaying of a tall building in the wind. In every case the same three regimes appear -- oscillate, return-with-a-touch-of-overshoot, or crawl back -- and the same algebraic gadget, the *characteristic equation*, predicts which one happens.

This chapter builds the entire toolkit. We will derive it once, prove the structural theorems, then keep meeting the same picture in different clothing.

## What you will learn

- Why second-order shows up the moment Newton's law meets a restoring force
- The structural theorems: superposition, the Wronskian, $y = y_h + y_p$
- Constant-coefficient homogeneous equations via the characteristic polynomial (real, repeated, complex roots)
- Damping ratio $\zeta$ and the under/critical/over trichotomy
- Forced response by **undetermined coefficients** (when $f$ is "nice")
- **Variation of parameters** (when it is not)
- Resonance and how to read the amplitude curve
- The same equation as an RLC circuit, and how to convert any $n$-th order ODE into a first-order system

## Prerequisites

- [Chapter 2: First-order methods](/en/ode-chapter-02-first-order-methods/) -- separable equations, integrating factors
- Linear algebra fluency at the level of $2\times 2$ determinants and complex numbers

---

## 1. Why second-order is the natural unit

Any time a system has both **inertia** (kinetic content, momentum, current in an inductor) and a **restoring force** (a spring, gravity, a capacitor's voltage), Newton's $F = ma$ produces a second derivative on the left and a position on the right:

$$
m\,\ddot x \;=\; -\,k x \;-\; b\,\dot x \;+\; F_\text{ext}(t).
$$

Dividing by $m$ and adopting the standard normalisation gives the equation we will keep returning to:

$$
\boxed{\;\ddot x + 2\zeta\omega_0\,\dot x + \omega_0^2\,x \;=\; f(t)\;}
$$

with **natural frequency** $\omega_0 = \sqrt{k/m}$ and **damping ratio** $\zeta = b/(2\sqrt{mk})$. Almost every example in this chapter is a special case.

> **What second order really buys you.** A first-order ODE $\dot x = F(x,t)$ at time $t$ is determined by the single number $x(t)$. A second-order ODE needs *two* initial conditions, $x(0)$ and $\dot x(0)$ -- you can specify position and velocity independently, so the system can store and exchange two kinds of "stuff" (kinetic and potential energy, voltage and current, etc.). Oscillation is the visible signature of that exchange.

---

## 2. Structure of the solution space

### 2.1 Standard form

An $n$-th order linear ODE looks like

$$
y^{(n)} + p_{n-1}(x)\,y^{(n-1)} + \cdots + p_1(x)\,y' + p_0(x)\,y \;=\; g(x).
$$

It is **homogeneous** when $g \equiv 0$, otherwise **non-homogeneous**.

### 2.2 Three theorems that organise everything

**(T1) Superposition.** If $y_1, y_2$ both solve the homogeneous equation, so does $c_1 y_1 + c_2 y_2$ for any constants $c_1, c_2$. (Linearity of derivatives -- write it out.)

**(T2) Dimension of the homogeneous solution space.** An $n$-th order linear homogeneous ODE on an interval where the coefficients $p_i$ are continuous has *exactly* $n$ linearly independent solutions $y_1, \dots, y_n$, and every solution is

$$
y_h(x) \;=\; c_1 y_1(x) + c_2 y_2(x) + \cdots + c_n y_n(x).
$$

This is the existence-uniqueness theorem in disguise: the $n$ initial conditions $y(x_0), y'(x_0), \dots, y^{(n-1)}(x_0)$ pick out a unique element of an $n$-dimensional vector space.

**(T3) Non-homogeneous structure.** The general solution of $L[y] = g$ is

$$
y \;=\; y_h \;+\; y_p,
$$

where $y_p$ is *any one* particular solution. The reason is that if $y, \tilde y$ are two solutions, $L[y - \tilde y] = 0$, so they differ by a homogeneous solution.

### 2.3 The Wronskian: a determinant test for independence

How do we *check* that $n$ candidate solutions are linearly independent? Differentiate them $n-1$ times and form the **Wronskian**:

$$
W(y_1, \dots, y_n)(x) \;=\; \det
\begin{pmatrix}
y_1 & y_2 & \cdots & y_n \\
y_1' & y_2' & \cdots & y_n' \\
\vdots & \vdots & & \vdots \\
y_1^{(n-1)} & y_2^{(n-1)} & \cdots & y_n^{(n-1)}
\end{pmatrix}.
$$

For $n = 2$: $W(y_1, y_2) = y_1 y_2' - y_2 y_1'$.

**The test.** If $W(x_0) \neq 0$ at *any one point* $x_0$ in the interval, the solutions are linearly independent (and Abel's identity then forces $W \neq 0$ everywhere on the interval).

**Worked example.** Take $y_1 = \sin x, y_2 = \cos x$:
$$
W = \sin x \cdot (-\sin x) - \cos x \cdot \cos x = -1 \neq 0.
$$
Independent everywhere -- they form a basis for solutions of $y'' + y = 0$.

**A dependent example.** $y_1 = \sin x, y_2 = 2\sin x$ gives $W = 2\sin x\cos x - 2\sin x\cos x = 0$ identically. They are scalar multiples; the dimension of the span is one, not two.

![Wronskian as a test for linear independence: independent pair (W is a non-zero constant) versus a dependent pair (W is identically zero).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/03-linear-theory/fig5_wronskian.png)
*Wronskian as a test for linear independence. Left: $\sin x$ and $\cos x$ are independent and $W \equiv -1$. Right: $\sin x$ and $2\sin x$ are scalar multiples and $W \equiv 0$.*

---

## 3. Constant coefficients: the characteristic equation

### 3.1 The trick

For

$$
a_n y^{(n)} + a_{n-1} y^{(n-1)} + \cdots + a_1 y' + a_0 y \;=\; 0
$$

guess $y = e^{rx}$. Each derivative pulls down one factor of $r$, so the equation reduces to the **characteristic polynomial**

$$
P(r) \;\equiv\; a_n r^n + a_{n-1} r^{n-1} + \cdots + a_1 r + a_0 \;=\; 0.
$$

Every root $r$ contributes a building block. The map *roots $\to$ basis solutions* is the entire content of this section.

### 3.2 The three cases

**Case 1: Distinct real roots $r_1, \dots, r_n$.**
$$
y \;=\; c_1 e^{r_1 x} + c_2 e^{r_2 x} + \cdots + c_n e^{r_n x}.
$$
*Example.* $y'' - 5y' + 6y = 0$ gives $(r-2)(r-3) = 0$, so $y = c_1 e^{2x} + c_2 e^{3x}$.

**Case 2: Repeated root $r$ of multiplicity $k$.** A multiplicity-$k$ root only gives one exponential, but the missing $k-1$ basis functions come from multiplying by $x$:
$$
y \;=\; (c_1 + c_2 x + c_3 x^2 + \cdots + c_k x^{k-1})\,e^{rx}.
$$
The reason: when $P(r)$ has a double root, the operator $L$ factors as $(D - r)^2 \cdot Q(D)$, and $(D-r)^2[x e^{rx}] = 0$ by direct calculation.

*Example.* $y'' - 4y' + 4y = 0$ gives $(r-2)^2 = 0$, so $y = (c_1 + c_2 x)e^{2x}$.

**Case 3: Complex conjugate pair $r = \alpha \pm i\beta$.** Real coefficients force complex roots to come in conjugate pairs. The real-valued basis is
$$
y \;=\; e^{\alpha x}\bigl(c_1 \cos\beta x + c_2 \sin\beta x\bigr).
$$
*Why this works.* The complex pair contributes $C_1 e^{(\alpha+i\beta)x} + C_2 e^{(\alpha-i\beta)x}$, and Euler's formula $e^{i\beta x} = \cos\beta x + i\sin\beta x$ rearranges this into the real form. Concretely, $\alpha$ controls the exponential envelope and $\beta$ controls the oscillation rate.

*Example.* $y'' + 2y' + 5y = 0$ gives $r = -1 \pm 2i$, so $y = e^{-x}(c_1\cos 2x + c_2\sin 2x)$ -- a sinusoid trapped inside a shrinking exponential envelope.

### 3.3 Reading roots geometrically

The location of the roots in the complex plane *is* the qualitative behaviour:

- **Left half-plane** ($\Re(r) < 0$): solution decays. Stable.
- **Right half-plane** ($\Re(r) > 0$): solution grows. Unstable.
- **Imaginary axis** ($\Re(r) = 0$): pure oscillation, marginal.
- **Off-axis** ($\Im(r) \neq 0$): oscillatory component at frequency $|\Im(r)|$.

![Three canonical configurations of characteristic roots in the complex plane and the matching time-domain solutions.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/03-linear-theory/fig2_characteristic_roots.png)
*The three canonical root configurations. Top: positions in the complex $r$-plane (the green region marks the stable left half-plane). Bottom: the matching time-domain solution. Repeated roots produce the $x e^{rx}$ "kink" at the origin; complex pairs produce a damped sinusoid.*

---

## 4. Damped oscillation: the trichotomy in pictures

Consider the canonical second-order homogeneous equation in normalised form:

$$
\ddot x + 2\zeta\omega_0\,\dot x + \omega_0^2\,x \;=\; 0.
$$

Its characteristic equation $r^2 + 2\zeta\omega_0 r + \omega_0^2 = 0$ has discriminant $4\omega_0^2(\zeta^2 - 1)$. The sign of $\zeta^2 - 1$ determines which of the three cases we land in:

| Regime | Condition | Roots | Behaviour |
|---|---|---|---|
| **Underdamped** | $0 < \zeta < 1$ | $-\zeta\omega_0 \pm i\omega_d$, $\omega_d = \omega_0\sqrt{1-\zeta^2}$ | Oscillation under a decaying envelope $e^{-\zeta\omega_0 t}$ |
| **Critically damped** | $\zeta = 1$ | $-\omega_0$ (double) | Fastest non-oscillatory return; $(c_1 + c_2 t)e^{-\omega_0 t}$ |
| **Overdamped** | $\zeta > 1$ | two real, both negative | Sum of two decaying exponentials, slow return |

![Three damping regimes side by side: underdamped oscillation under an exponential envelope, critical damping, and a slow overdamped relaxation.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/03-linear-theory/fig1_damping_regimes.png)
*Three damping regimes of $\ddot x + 2\zeta\omega_0 \dot x + \omega_0^2 x = 0$. The dashed envelope on the left panel is $\pm e^{-\zeta\omega_0 t}$ -- the imaginary part of the roots controls the oscillation; the real part controls the envelope decay.*

> **Engineering aside.** Car suspensions are tuned to $\zeta \approx 0.6\text{--}0.7$. That is just below critical damping: the car settles fast, but with a small overshoot you do not feel. Pure $\zeta = 1$ would feel "dead" because the response is sluggish near the end; pure $\zeta = 0.1$ would let you bounce for a city block after every pothole. Door closers, by contrast, are deliberately overdamped ($\zeta > 1$) to avoid slamming.

---

## 5. Non-homogeneous: the method of undetermined coefficients

### 5.1 The recipe

To solve $L[y] = f(x)$ with constant coefficients:

1. Solve $L[y] = 0$ to get $y_h$.
2. Guess a $y_p$ whose form mirrors $f$.
3. Substitute the guess, equate coefficients to determine the unknown constants.
4. Combine: $y = y_h + y_p$.

### 5.2 The guessing table

| Forcing $f(x)$ | Trial $y_p$ |
|---|---|
| $e^{\alpha x}$ | $A e^{\alpha x}$ |
| $\cos\beta x$ or $\sin\beta x$ | $A\cos\beta x + B\sin\beta x$ |
| polynomial of degree $n$ | polynomial of degree $n$ |
| $e^{\alpha x}\cos\beta x$ | $e^{\alpha x}(A\cos\beta x + B\sin\beta x)$ |
| product / sum of the above | corresponding product / sum |

**The resonance correction.** If your trial form is *itself a homogeneous solution*, multiply it by $x$ (or $x^2$ for a double resonance). Otherwise the substitution gives $0 = f$ -- a contradiction.

### 5.3 A worked example end-to-end

Solve $y'' + y' + y = e^{-x/2}\cos x$.

**Homogeneous part.** Roots of $r^2 + r + 1 = 0$ are $r = -\tfrac12 \pm \tfrac{\sqrt 3}{2} i$, so

$$
y_h \;=\; e^{-x/2}\bigl(c_1\cos\tfrac{\sqrt 3}{2}x + c_2\sin\tfrac{\sqrt 3}{2}x\bigr).
$$

**Trial form.** The forcing is $e^{-x/2}\cos x$ -- the exponential envelope matches $y_h$ but the inner frequency $\beta = 1$ does *not* match $\sqrt 3/2$, so there is no resonance and we may try

$$
y_p \;=\; e^{-x/2}(A\cos x + B\sin x).
$$

**Solving for $A, B$.** Substituting and collecting $\cos$ and $\sin$ terms gives the linear system $\tfrac14 A + \tfrac32 B = 1,\ -\tfrac32 A + \tfrac14 B = 0$, whose solution is $A = \tfrac{2}{37},\ B = \tfrac{24}{37}$.

![The undetermined-coefficients workflow as three stacked panels: forcing, trial basis, fitted particular solution.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/03-linear-theory/fig6_undetermined_coefficients.png)
*Method of undetermined coefficients in three steps. (1) Identify the forcing. (2) Pick a trial form spanned by the same exponentials and trig functions. (3) Solve a small linear system; the green dashed curve verifies that $y_p'' + y_p' + y_p$ reproduces $f$ exactly.*

### 5.4 Resonance: when the trial form lies inside $y_h$

Force a frictionless oscillator at exactly its natural frequency:

$$
\ddot x + \omega_0^2 x \;=\; F_0\cos\omega_0 t.
$$

The naive trial $A\cos\omega_0 t + B\sin\omega_0 t$ already solves the homogeneous equation, so we multiply by $t$:

$$
x_p \;=\; \frac{F_0}{2\omega_0}\,t\,\sin\omega_0 t.
$$

The amplitude grows linearly with time -- energy is pumped in every cycle and never removed. Add even a sliver of damping and the unbounded growth becomes a finite peak at $\omega_r = \omega_0\sqrt{1 - 2\zeta^2}$, with peak amplitude $\propto 1/\zeta$.

![Steady-state amplitude versus driving frequency for several damping ratios; the resonance peak sharpens dramatically as zeta tends to zero.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/03-linear-theory/fig3_resonance_curve.png)
*Steady-state amplitude as a function of driving frequency, for $\zeta = 0.05, 0.1, 0.2, 0.4, 1/\sqrt 2$. As $\zeta \to 0$ the peak grows without bound (the undamped resonance limit). For $\zeta > 1/\sqrt 2$ the peak disappears entirely -- the system has no resonance.*

---

## 6. Variation of parameters: when guessing fails

### 6.1 Why we need it

The undetermined-coefficients table only contains exponentials, polynomials, sines, cosines, and their products. For $f(x) = \sec x, \tan x, \ln x, e^{x^2}$, ... we need a method that works for any continuous forcing. **Variation of parameters** is that method.

### 6.2 The formula (second order)

Given $y'' + p(x)y' + q(x)y = f(x)$ and a basis $y_1, y_2$ for the homogeneous equation, look for a particular solution of the form $y_p = u_1(x)\,y_1(x) + u_2(x)\,y_2(x)$ -- promoting the "constants" of the homogeneous solution to functions (hence the name). Imposing the constraint $u_1' y_1 + u_2' y_2 = 0$ to keep things tractable, substitution yields the linear system

$$
\begin{pmatrix} y_1 & y_2 \\ y_1' & y_2' \end{pmatrix}
\begin{pmatrix} u_1' \\ u_2' \end{pmatrix} =
\begin{pmatrix} 0 \\ f \end{pmatrix},
$$

whose determinant is the Wronskian $W$. Cramer's rule then gives the closed-form

$$
\boxed{\;y_p \;=\; -\,y_1 \int \frac{y_2\,f}{W}\,dx \;+\; y_2 \int \frac{y_1\,f}{W}\,dx\;}.
$$

### 6.3 Worked example: $y'' + y = \sec x$

Homogeneous basis: $y_1 = \cos x,\ y_2 = \sin x,\ W = 1$. The integrands simplify nicely:

$$
\frac{y_2\,f}{W} = \sin x\,\sec x = \tan x, \qquad
\frac{y_1\,f}{W} = \cos x\,\sec x = 1.
$$

Integrating,

$$
y_p \;=\; -\cos x\int\tan x\,dx + \sin x\int 1\,dx
       \;=\; \cos x\,\ln|\cos x| + x\,\sin x.
$$

![Variation of parameters on y''+y = sec(x): the homogeneous basis, the antiderivatives u1 and u2, and the reconstructed particular solution.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/03-linear-theory/fig7_variation_of_parameters.png)
*Variation of parameters on $y'' + y = \sec x$. Top row: the homogeneous basis $y_1, y_2$ and the singular forcing $\sec x$. Bottom row: the antiderivatives $u_1(x) = \ln|\cos x|$ and $u_2(x) = x$, and the assembled particular solution $y_p = \cos x \ln|\cos x| + x\sin x$.*

> **When to use which.** Undetermined coefficients is faster when it applies: you guess and solve a small linear system. Variation of parameters always works (with continuous $f$) but costs you two integrals -- which may themselves be hard. For exponential / trig / polynomial forcings, prefer undetermined coefficients; for everything else, reach for variation of parameters.

---

## 7. RLC circuits: the same equation in disguise

A series resistor-inductor-capacitor circuit with applied voltage $V(t)$ obeys Kirchhoff's voltage law:

$$
L\,\ddot q \;+\; R\,\dot q \;+\; \frac{q}{C} \;=\; V(t),
$$

where $q(t)$ is the charge on the capacitor and $\dot q = i(t)$ is the current. Match coefficients with the mechanical normal form:

| Mechanical | Electrical |
|---|---|
| mass $m$ | inductance $L$ |
| damping $b$ | resistance $R$ |
| stiffness $k$ | inverse capacitance $1/C$ |
| displacement $x$ | charge $q$ |
| forcing $F(t)$ | voltage $V(t)$ |

The standard parameters become $\omega_0 = 1/\sqrt{LC}$ (resonant angular frequency), $\zeta = \tfrac{R}{2}\sqrt{C/L}$ (damping ratio), and $Q = 1/(2\zeta)$ (quality factor; high $Q$ means a sharp resonance peak). The entire under/critical/over trichotomy carries over verbatim.

![Series RLC schematic and three step responses showing the same under, critical, over trichotomy as the spring.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/03-linear-theory/fig4_rlc_response.png)
*A series RLC circuit (left) and its step response when $V$ is switched on at $t = 0$ (right). The same three regimes appear; only the units have changed. Tuning radios is exactly the underdamped case with very high $Q$.*

---

## 8. Reduction to a first-order system

Numerical solvers (and the matrix theory of [Chapter 6](/en/ode-chapter-06-systems/)) want first-order systems. Any $n$-th order ODE can be flattened by introducing one variable per derivative. For

$$
y'' + a\,y' + b\,y = f(x), \qquad y_1 := y,\ y_2 := y',
$$

the equivalent system is

$$
\begin{pmatrix} y_1' \\ y_2' \end{pmatrix} \;=\;
\begin{pmatrix} 0 & 1 \\ -b & -a \end{pmatrix}
\begin{pmatrix} y_1 \\ y_2 \end{pmatrix} +
\begin{pmatrix} 0 \\ f(x) \end{pmatrix}.
$$

The eigenvalues of the matrix $\bigl(\begin{smallmatrix} 0 & 1 \\ -b & -a \end{smallmatrix}\bigr)$ are *exactly* the roots of the characteristic polynomial $r^2 + ar + b$ -- so everything we said about characteristic roots is just two-dimensional linear algebra in disguise. We will exploit this fully in Chapter 6.

---

## 9. Summary

### Method selector

| Equation | First tool to try |
|---|---|
| Constant-coefficient homogeneous | Characteristic equation |
| Constant-coefficient non-homogeneous, "nice" $f$ | Characteristic equation + undetermined coefficients |
| Variable-coefficient or arbitrary continuous $f$ | Find $y_h$ first, then variation of parameters |
| Anything you want to integrate numerically | Reduce to a first-order system |

### Roots-to-solutions cheat sheet

| Root | Contribution to the basis |
|---|---|
| Distinct real $r$ | $e^{rx}$ |
| Real $r$, multiplicity $k$ | $e^{rx},\ x e^{rx},\ \dots,\ x^{k-1} e^{rx}$ |
| Complex pair $\alpha \pm i\beta$ | $e^{\alpha x}\cos\beta x,\ e^{\alpha x}\sin\beta x$ |

### Big ideas to keep

- A second-order ODE is the natural language of *inertia plus restoring force*.
- The location of the characteristic roots in $\mathbb{C}$ already tells you the qualitative behaviour.
- The Wronskian is a determinant that detects whether candidate solutions span the full $n$-dimensional solution space.
- Resonance is what happens when your trial form *is* a homogeneous solution; multiplying by $x$ both fixes the algebra and explains the unbounded growth physically.
- Mechanical and electrical second-order systems are the *same equation*; the same intuition transfers without modification.

---

## Exercises

**Warm-up.**

1. General solution of $y'' - 3y' + 2y = 0$.
2. General solution of $y'' - 4y' + 4y = 0$ (repeated root).
3. Solve the IVP $y'' + y = 0,\ y(0) = 1,\ y'(0) = 0$.
4. Solve $y'' + 4y = \sin 2x$ and identify the resonance correction.

**Practice.**

5. Use variation of parameters to find a particular solution of $y'' + y = \tan x$ on $(-\pi/2, \pi/2)$.
6. An RLC circuit has $R = 100\,\Omega,\ L = 0.5\,\text{H},\ C = 10\,\mu\text{F}$. Compute $\omega_0$, $\zeta$, and $Q$, and decide which regime it is in.
7. Show that the steady-state amplitude of $\ddot x + 2\zeta\omega_0\dot x + \omega_0^2 x = F_0\cos\omega t$ is $A(\omega) = F_0/\sqrt{(\omega_0^2 - \omega^2)^2 + (2\zeta\omega_0\omega)^2}$, and that for $\zeta < 1/\sqrt 2$ the peak occurs at $\omega_r = \omega_0\sqrt{1 - 2\zeta^2}$.

**Programming.**

8. Reproduce the damping-regimes figure: plot step responses for $\zeta = 0.1, 0.3, 0.7, 1.0, 2.0$ with $\omega_0 = 2\pi$, using `scipy.integrate.solve_ivp`.
9. Simulate a *double pendulum* (a non-linear coupled second-order system) and show numerically that two trajectories with initial conditions differing by $10^{-6}$ diverge after a few seconds. This is your first taste of [Chapter 9](/en/ode-chapter-09-bifurcation-chaos/) (chaos).

---

## References

- Boyce & DiPrima, *Elementary Differential Equations and Boundary Value Problems*, Wiley (2012). The standard treatment, theorem-by-theorem.
- Kreyszig, *Advanced Engineering Mathematics*, Wiley (2011). Strong on the engineering applications, especially RLC and vibrations.
- Nagle, Saff & Snider, *Fundamentals of Differential Equations*, Pearson (2017). Clean exposition of variation of parameters.
- Strogatz, *Nonlinear Dynamics and Chaos* (1994). Chapters 4 and 5 give the geometric reading of linear stability that anticipates [Chapter 7](/en/ode-chapter-07-systems-and-phase-plane/).
- MIT OpenCourseWare 18.03, *Differential Equations* -- video lectures by Arthur Mattuck.

---

**Series Navigation**

