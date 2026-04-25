---
title: "ODE Chapter 4: The Laplace Transform"
date: 2023-08-21 09:00:00
tags:
  - ODE
  - Laplace Transform
  - Transfer Function
  - Control Systems
  - Signal Processing
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 4
  total: 18
lang: en
mathjax: true
description: "The engineer's secret weapon: turn differential equations into algebra with the Laplace transform. Learn the key properties, partial fractions, transfer functions, and PID control basics."
disableNunjucks: true
series_order: 4
---

**The Laplace transform turns calculus into algebra.** Instead of grinding through integration, guessing trial solutions, and bolting on initial conditions at the end, you transform the entire ODE — equation, forcing, and initial data — into a single polynomial equation in a complex variable $s$. You solve it like a high-school problem, then transform back. Along the way, the *shape* of the solution becomes geometry: poles in the left half of the complex plane decay, poles on the right blow up, poles on the imaginary axis ring forever. This chapter develops that picture from first principles and connects it to the engineering tools — transfer functions, Bode plots, PID control — that turned the Laplace transform into the lingua franca of dynamics.

## What you will learn

- The definition and the intuition behind $e^{-st}$ as a "decay probe"
- The differentiation property — the key that converts an ODE into algebra
- A small transform table you can use to invert almost everything in this course
- Partial fraction decomposition for distinct, repeated, and complex poles
- Transfer functions, the pole–zero picture, and the geometric stability criterion
- Step and impulse responses, and how they encode the system completely
- Bode plots and the time vs frequency duality
- PID control: how P, I, and D each fix one weakness of the others

## Prerequisites

- Chapters 1–3: first- and second-order ODEs and the superposition principle
- Partial fractions from algebra/calculus
- Complex numbers basics ($a + bi$, magnitude, phase)
- Familiarity with the integral $\int_0^\infty e^{-st}\,dt$

---

## 1. Why the Laplace transform exists

Consider an RC circuit driven by a voltage source:

$$RC\,V_c'(t) + V_c(t) = V_s(t), \qquad V_c(0) = V_0.$$

If $V_s$ is a constant, you can solve this with an integrating factor in two lines. But the moment $V_s$ becomes a switched-on step, a brief impulse, a ramp, or some piecewise waveform from a real signal generator, the elementary methods grow tedious. You end up with a forest of integration constants, side-condition matching, and case analysis.

The Laplace transform offers a single workflow that handles all of these:

1. **Transform** both sides of the ODE — derivatives become polynomials in $s$, and the initial conditions are absorbed automatically.
2. **Solve** the resulting algebraic equation for $Y(s)$.
3. **Inverse transform** back to $y(t)$ using a small table.

The bookkeeping disappears. What is left is a clean separation: *who* drives the system (the transform of the input), *what* the system does to it (the transfer function), and *how* it started (the initial conditions, already woven in).

![Building blocks of the Laplace transform: the unit step, the Dirac impulse, and the decaying probe e^{-st}.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/04-constant-coefficients/fig1_step_impulse_kernel.png)
*The Laplace transform integrates a signal $f(t)$ against the probe $e^{-st}$. The step and the impulse are the two canonical inputs you will see throughout the chapter.*

---

## 2. Definition and the core transform table

### 2.1 The forward transform

$$F(s) = \mathcal{L}\{f(t)\} = \int_0^\infty f(t)\,e^{-st}\,dt.$$

**Intuition.** Think of $e^{-st}$ as a probe that, for each complex frequency $s$, asks: *how much of $f$ survives if I weight it by an exponential that decays at rate $\operatorname{Re}(s)$?* When $s$ is large and positive, the probe sees only the behaviour of $f$ near $t = 0$. When $\operatorname{Re}(s)$ is small, it sees the long-term tail. The full function $F(s)$ is the answer at every $s$ at once — a fingerprint of $f$.

### 2.2 A working transform table

| $f(t)$ | $F(s) = \mathcal{L}\{f(t)\}$ | Region of convergence |
|---|---|---|
| $1$ (or $u(t)$) | $1/s$ | $\operatorname{Re}(s) > 0$ |
| $t^n$ | $n!/s^{n+1}$ | $\operatorname{Re}(s) > 0$ |
| $e^{at}$ | $1/(s-a)$ | $\operatorname{Re}(s) > a$ |
| $\sin\omega t$ | $\omega/(s^2+\omega^2)$ | $\operatorname{Re}(s) > 0$ |
| $\cos\omega t$ | $s/(s^2+\omega^2)$ | $\operatorname{Re}(s) > 0$ |
| $e^{at}\sin\omega t$ | $\omega/((s-a)^2+\omega^2)$ | $\operatorname{Re}(s) > a$ |
| $e^{at}\cos\omega t$ | $(s-a)/((s-a)^2+\omega^2)$ | $\operatorname{Re}(s) > a$ |
| $\delta(t)$ (impulse) | $1$ | all $s$ |
| $u(t-a)$ (delayed step) | $e^{-as}/s$ | $\operatorname{Re}(s) > 0$ |

This is enough to invert nearly every problem in the chapter.

### 2.3 Two derivations from the definition

**$\mathcal{L}\{1\}$.** Direct integration:

$$\int_0^\infty e^{-st}\,dt = \left[-\frac{1}{s}\,e^{-st}\right]_0^\infty = \frac{1}{s}, \qquad \operatorname{Re}(s) > 0.$$

**$\mathcal{L}\{e^{at}\}$.** Combine the exponentials before integrating:

$$\int_0^\infty e^{at}\,e^{-st}\,dt = \int_0^\infty e^{-(s-a)t}\,dt = \frac{1}{s-a}, \qquad \operatorname{Re}(s) > a.$$

The same trick — fold $e^{at}$ into the kernel — is what makes the *frequency-shift* property below trivially obvious.

---

## 3. The properties that do all the work

### 3.1 Linearity

$$\mathcal{L}\{a f + b g\} = a F(s) + b G(s).$$

### 3.2 Differentiation — the key to the whole subject

$$\mathcal{L}\{f'(t)\} = sF(s) - f(0),$$
$$\mathcal{L}\{f''(t)\} = s^2 F(s) - s f(0) - f'(0),$$
$$\mathcal{L}\{f^{(n)}(t)\} = s^n F(s) - s^{n-1} f(0) - \cdots - f^{(n-1)}(0).$$

**Why this matters.** Differentiation in $t$ becomes multiplication by $s$, and the initial conditions appear *as part of the formula*, not as side constraints to be matched later. An $n$-th order linear ODE turns into an $n$-th degree polynomial equation in $s$ that already knows about $y(0), y'(0), \dots$.

The proof is one integration by parts:

$$\int_0^\infty f'(t)\,e^{-st}\,dt = \left[f(t)\,e^{-st}\right]_0^\infty + s\int_0^\infty f(t)\,e^{-st}\,dt = sF(s) - f(0),$$

provided $f(t)\,e^{-st}\to 0$ at infinity, which is the meaning of the region of convergence.

### 3.3 Frequency shift

$$\mathcal{L}\{e^{at}f(t)\} = F(s-a).$$

Multiplying by $e^{at}$ in the time domain shifts the entire transform by $a$. This is why every "damped" entry in the table is just an undamped entry with $s\to s-a$.

### 3.4 Time shift (delay)

$$\mathcal{L}\{f(t-a)\,u(t-a)\} = e^{-as}F(s), \qquad a > 0.$$

A delay in time becomes a multiplicative phase $e^{-as}$ in the $s$-domain. Use the gate $u(t-a)$ to make sure you are transforming the *delayed-and-truncated* signal, not the original.

### 3.5 Convolution

$$\mathcal{L}\{(f * g)(t)\} = F(s)\,G(s), \qquad (f * g)(t) = \int_0^t f(\tau)\,g(t-\tau)\,d\tau.$$

In the time domain, the response of an LTI system to an input is a convolution; in the $s$-domain it is just a product. This is the algebraic statement that makes block diagrams work.

### 3.6 Final value theorem

$$\lim_{t\to\infty} f(t) = \lim_{s\to 0} sF(s),$$

provided the limit exists and all poles of $sF(s)$ lie in the open left half-plane. Use it to read steady-state values straight off $Y(s)$, without inverting.

---

## 4. Solving ODEs: the workflow in two examples

### 4.1 First-order with an exponential forcing

Solve $y' + 2y = e^{-t}$, $y(0) = 1$.

**Transform.** Apply $\mathcal{L}$ to both sides. Using the differentiation property,

$$sY(s) - 1 + 2Y(s) = \frac{1}{s+1}.$$

**Solve algebraically.** Group and divide:

$$(s+2)Y(s) = 1 + \frac{1}{s+1}, \qquad Y(s) = \frac{1}{s+2} + \frac{1}{(s+1)(s+2)}.$$

**Partial fractions.** Decompose $\dfrac{1}{(s+1)(s+2)} = \dfrac{1}{s+1} - \dfrac{1}{s+2}$, so

$$Y(s) = \frac{1}{s+2} + \frac{1}{s+1} - \frac{1}{s+2} = \frac{1}{s+1}.$$

**Invert.** Read $\mathcal{L}^{-1}\{1/(s+1)\} = e^{-t}$ from the table.

$$\boxed{\; y(t) = e^{-t}.\;}$$

**Verify.** $y' + 2y = -e^{-t} + 2e^{-t} = e^{-t}$, and $y(0) = 1$. Done.

### 4.2 Second-order with resonance

Solve $y'' + \omega_0^2\,y = \cos\omega_0 t$, with $y(0) = y'(0) = 0$.

**Transform.** Using $\mathcal{L}\{y''\} = s^2 Y - sy(0) - y'(0)$ and $\mathcal{L}\{\cos\omega_0 t\} = s/(s^2+\omega_0^2)$,

$$s^2 Y + \omega_0^2 Y = \frac{s}{s^2 + \omega_0^2}, \qquad Y(s) = \frac{s}{(s^2 + \omega_0^2)^2}.$$

**Invert.** This is a tabulated inverse,

$$y(t) = \frac{t}{2\omega_0}\,\sin\omega_0 t.$$

The crucial feature is the explicit factor of $t$. Algebraically, it comes from the **repeated** pole pair at $s = \pm j\omega_0$: a higher-multiplicity pole produces a polynomial-times-sinusoid in time. Physically, it means the amplitude grows without bound — this is **resonance**.

![On-resonance forcing produces linear growth; off-resonance forcing produces a bounded beat pattern.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/04-constant-coefficients/fig5_resonance_buildup.png)
*Repeated poles in $Y(s)$ are the algebraic fingerprint of resonance: the inverse transform picks up a $t$ factor and the response grows linearly.*

---

## 5. Partial fractions: the only technique you really need

Once you have $Y(s)$ as a rational function, almost all of the inversion work is splitting it into a sum of pieces that match table entries.

### 5.1 Distinct real poles

$$\frac{P(s)}{(s-a)(s-b)} = \frac{A}{s-a} + \frac{B}{s-b}.$$

Use the **cover-up rule**: $A$ equals $P(s)/(s-b)$ evaluated at $s = a$. (Geometrically, you "cover" the $(s-a)$ factor and plug in the pole.)

### 5.2 Repeated poles

$$\frac{P(s)}{(s-a)^3} = \frac{A_1}{s-a} + \frac{A_2}{(s-a)^2} + \frac{A_3}{(s-a)^3}.$$

Each multiplicity-$k$ pole produces a term of the form $t^{k-1}\,e^{at}/(k-1)!$ in the time domain.

### 5.3 Complex conjugate poles

For an irreducible quadratic factor, complete the square:

$$\frac{B s + C}{(s-\alpha)^2 + \beta^2}\;\xrightarrow{\;\mathcal{L}^{-1}\;}\; e^{\alpha t}\big(B \cos\beta t + D \sin\beta t\big),$$

where $D = (C + \alpha B)/\beta$ after rewriting the numerator as $B(s-\alpha) + (C + \alpha B)$.

![Inverse-Laplace as a sum of simple modes: split the rational F(s) into table-lookup pieces, then add the time-domain components.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/04-constant-coefficients/fig3_partial_fractions.png)
*$Y(s) = \dfrac{3s+5}{(s+1)(s+2)} = \dfrac{2}{s+1} + \dfrac{1}{s+2}$, so $y(t) = 2 e^{-t} + e^{-2t}$. Each pole contributes one mode, and they superpose.*

---

## 6. Transfer functions and the geometry of stability

### 6.1 Definition

For a linear time-invariant (LTI) system relating input $u(t)$ to output $y(t)$, define

$$H(s) \;=\; \frac{Y(s)}{U(s)}\quad\text{at zero initial conditions.}$$

$H(s)$ is the **transfer function**. It depends only on the system, not on what you feed in or how it started.

**Example — RC low-pass filter.** From $RC\,V_c' + V_c = V_s$ with $V_c(0) = 0$,

$$H(s) = \frac{1}{RCs + 1} = \frac{1}{\tau s + 1}, \qquad \tau = RC.$$

A single real pole at $s = -1/\tau$.

### 6.2 Poles, zeros, and stability

- **Zeros** are values of $s$ where $H(s) = 0$ (numerator roots).
- **Poles** are values of $s$ where $H(s) \to \infty$ (denominator roots).

The pole picture decides everything about the *unforced* response. Each pole contributes one mode to $\mathcal{L}^{-1}\{H(s)\}$:

| Pole location | Time-domain mode | Behaviour |
|---|---|---|
| Real, $s = -a$ with $a > 0$ | $e^{-at}$ | decays to zero |
| Real, $s = a > 0$ | $e^{at}$ | grows without bound — **unstable** |
| Complex pair $\alpha \pm j\beta$, $\alpha < 0$ | $e^{\alpha t}(\cos\beta t,\sin\beta t)$ | damped oscillation |
| Pure imaginary $\pm j\beta$ | $\cos\beta t,\,\sin\beta t$ | undamped oscillation |
| Complex pair, $\alpha > 0$ | $e^{\alpha t}(\cos\beta t,\sin\beta t)$ | growing oscillation — unstable |

**Geometric stability criterion.** The system is asymptotically stable if and only if every pole of $H(s)$ lies strictly in the left half-plane $\operatorname{Re}(s) < 0$.

![The complex s-plane with sample poles in each region, paired with the impulse responses they produce.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/04-constant-coefficients/fig2_pole_zero_responses.png)
*The real part of a pole sets the decay rate; the imaginary part sets the oscillation frequency. Stability is just "are the poles in the left half-plane?"*

### 6.3 Step and impulse responses

The two canonical probes deserve their own names.

- **Impulse response.** $h(t) = \mathcal{L}^{-1}\{H(s)\}$ — what comes out when the input is a Dirac delta.
- **Step response.** $s(t) = \mathcal{L}^{-1}\{H(s)/s\}$ — what comes out when the input is a unit step.
- **Identity.** $h(t) = s'(t)$. The two encode the same information; a step is the integral of a delta.

Once you know $h(t)$, the response to *any* input is the convolution $y(t) = (h * u)(t)$. Equivalently, $Y(s) = H(s)\,U(s)$.

---

## 7. Two windows on the same system: time and frequency

Substituting $s = j\omega$ in $H(s)$ gives the **frequency response** $H(j\omega)$. The magnitude $|H(j\omega)|$ tells you how much each sinusoidal frequency is amplified; the phase $\arg H(j\omega)$ tells you how much it is delayed. Plotted on a log–log scale, these are the **Bode plots**.

For the canonical second-order system

$$H(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2},$$

the damping ratio $\zeta$ controls everything. With $\zeta < 1$ the poles are complex and the step response overshoots and rings; with $\zeta = 1$ they coalesce into a double real pole at $-\omega_n$ and the response rises as fast as it can without overshoot; with $\zeta > 1$ the poles split apart on the real axis and the response is sluggish.

![Step response and Bode plots for under-, critically, and over-damped second-order systems.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/04-constant-coefficients/fig4_transfer_function.png)
*Three views, one system. Time-domain overshoot, frequency-domain peaking, and pole geometry are three faces of the same dimensionless number $\zeta$.*

```python
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# H(s) = 1 / (s^2 + 0.5 s + 1) -- under-damped second-order
sys = signal.TransferFunction([1.0], [1.0, 0.5, 1.0])

t, y = signal.step(sys)
w, mag, phase = signal.bode(sys)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].plot(t, y);            axes[0].set_title("Step response")
axes[1].semilogx(w, mag);      axes[1].set_title("Bode magnitude (dB)")
axes[2].semilogx(w, phase);    axes[2].set_title("Bode phase (deg)")
for ax in axes: ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
```

---

## 8. PID control: each term fixes what the others cannot

The PID controller is the workhorse of industrial control. It produces an actuation $u(t)$ from the tracking error $e(t) = r(t) - y(t)$:

$$u(t) = K_p\,e(t) + K_i\!\int_0^t e(\tau)\,d\tau + K_d\,\frac{de}{dt}.$$

In the $s$-domain,

$$C(s) = K_p + \frac{K_i}{s} + K_d\,s.$$

Each term covers a specific failure mode of the others.

| Term | What it does | Strength | Failure mode |
|---|---|---|---|
| **P** (proportional) | Reacts to the present error | Fast initial correction | Leaves a steady-state error |
| **I** (integral) | Accumulates past error | Drives steady-state error to zero | Slows the loop, can cause oscillation |
| **D** (derivative) | Predicts where the error is going | Damps overshoot | Amplifies measurement noise |

Closing the loop with the PID controller around a plant $G(s)$ produces

$$T(s) = \frac{C(s)\,G(s)}{1 + C(s)\,G(s)}.$$

Tuning $K_p, K_i, K_d$ moves the closed-loop poles around the $s$-plane. The art is to push them deep into the left half-plane (for stability and speed) without making the imaginary parts too large (which would cause ringing).

![Closed-loop step responses with P-only, PI, and tuned PID controllers driving a lightly damped second-order plant.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/04-constant-coefficients/fig6_pid_control.png)
*P alone leaves a steady-state offset. Adding I removes the offset but slows the loop and adds overshoot. Adding D damps the overshoot back out. The full PID lands inside the $\pm 5\%$ band quickly and stays there.*

---

## 9. Python practice: symbolic and numerical

### 9.1 Symbolic transforms with SymPy

```python
from sympy import (symbols, laplace_transform, inverse_laplace_transform,
                   exp, sin, apart)

s, t = symbols("s t")
a, omega = symbols("a omega", positive=True)

print("L{1}        =", laplace_transform(1, t, s))
print("L{exp(-at)} =", laplace_transform(exp(-a*t), t, s))
print("L{sin(wt)}  =", laplace_transform(sin(omega*t), t, s))

F = (3*s + 5) / ((s + 1) * (s + 2))
print("partial fractions:", apart(F, s))
print("L^{-1}{F}        =", inverse_laplace_transform(F, s, t))
```

### 9.2 Pole–zero analysis with SciPy

```python
import numpy as np
from scipy import signal

# H(s) = (s + 2) / (s^2 + 3 s + 2)
sys = signal.TransferFunction([1, 2], [1, 3, 2])

poles = np.roots([1, 3, 2])
zeros = np.roots([1, 2])
stable = all(p.real < 0 for p in poles)

print(f"poles  = {poles}")
print(f"zeros  = {zeros}")
print(f"stable = {stable}")
```

---

## 10. Summary

### The five-step workflow

1. **Transform** both sides of the ODE; absorb the initial conditions.
2. **Solve** algebraically for $Y(s)$.
3. **Decompose** $Y(s)$ into partial fractions.
4. **Invert** term by term using the table.
5. **Verify** by substituting into the original ODE.

### The properties to memorise

| Property | Formula | Where it earns its keep |
|---|---|---|
| Differentiation | $\mathcal{L}\{f'\} = sF(s) - f(0)$ | Turns ODEs into algebra |
| Frequency shift | $\mathcal{L}\{e^{at}f\} = F(s-a)$ | Damped oscillators, exponential forcing |
| Time shift | $\mathcal{L}\{f(t-a)u(t-a)\} = e^{-as}F(s)$ | Delays and switched inputs |
| Convolution | $\mathcal{L}\{f * g\} = F(s)\,G(s)$ | Block diagrams, system response |
| Final value | $\lim_{t\to\infty} f = \lim_{s\to 0} sF(s)$ | Steady-state without inverting |

### The one-sentence picture

Linear time-invariant systems live naturally in the $s$-plane: their poles are their genome, the left half-plane is "stable", and the operations you do in time become arithmetic in $s$.

---

## Exercises

**Basic**

1. Find $\mathcal{L}\{t^2 e^{-3t}\}$ and $\mathcal{L}\{e^{2t}\sin 3t\}$ using the frequency-shift property.
2. Invert $F(s) = \dfrac{2s+1}{(s+1)(s+3)}$.
3. Solve $y' + 3y = e^{-2t}$ with $y(0) = 1$ using Laplace transforms; verify your answer.
4. Prove the differentiation property $\mathcal{L}\{f'\} = sF(s) - f(0)$ from the definition by integration by parts, and state precisely the condition you need at $t \to \infty$.

**Advanced**

5. Solve $y'' + 4y = \delta(t)$ with $y(0) = y'(0) = 0$, and explain why the impulse response equals $\tfrac{1}{2}\sin 2t$.
6. For $H(s) = \dfrac{10}{s^2 + 2s + 10}$, find the poles, the impulse response, and the step response. Identify $\zeta$ and $\omega_n$.
7. Use the final value theorem to find $\lim_{t\to\infty} y(t)$ when $Y(s) = \dfrac{5}{s(s^2+3s+2)}$, and check that the theorem's hypotheses are satisfied.

**Programming**

8. Plot Bode magnitude and phase for $H(s) = \dfrac{100}{s^2 + 10s + 100}$. Estimate the resonant peak from the plot and compare with $\omega_n\sqrt{1 - 2\zeta^2}$.
9. Simulate the step response of a system with pure time delay, $H(s) = \dfrac{e^{-s}}{s+1}$, using SciPy's `lsim` after expanding the delay with a Padé approximation.

---

## References

- Kreyszig, E. *Advanced Engineering Mathematics* (10th ed.), Wiley (2011) — Chapter 6 is the standard textbook treatment of the Laplace transform.
- Ogata, K. *Modern Control Engineering* (5th ed.), Pearson (2010) — pole–zero analysis, Bode plots, and PID tuning in depth.
- Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2nd ed.), Prentice Hall (1997) — the engineering-signals viewpoint and convolution.
- Strang, G. *Differential Equations and Linear Algebra*, Wellesley-Cambridge (2014) — a more geometric introduction.
- SciPy: [`scipy.signal`](https://docs.scipy.org/doc/scipy/reference/signal.html) — Python tools for transfer functions, step/impulse responses, and Bode plots.

---

**Series Navigation**

