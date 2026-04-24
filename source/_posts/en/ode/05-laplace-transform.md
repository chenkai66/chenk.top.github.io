---
title: "ODE Chapter 5: Power Series and Special Functions"
date: 2024-07-31 09:00:00
tags:
  - ODE
  - Series Solutions
  - Bessel Functions
  - Legendre Polynomials
  - Frobenius Method
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 5
  total: 18
lang: en
mathjax: true
description: "When elementary functions fail, power series step in. Learn the Frobenius method and meet the special functions of physics: Bessel, Legendre, Hermite, and Airy functions -- with Python visualizations."
disableNunjucks: true
---
**Some ODEs have no solutions in terms of familiar functions.** The Bessel equation, the Legendre equation, the Airy equation -- all arise naturally in physics (heat conduction in cylinders, gravitational fields of planets, quantum tunneling). Their solutions *define* entirely new functions. This chapter shows you how to find them using power series, why the Frobenius extension is forced upon us at singular points, and why the same handful of "special functions" keeps appearing across physics and engineering.

## What You Will Learn

- Power series solutions at ordinary points and the role of the radius of convergence
- Regular singular points and the Frobenius method
- Bessel functions: vibrating drums and cylindrical heat conduction
- Legendre polynomials: spherical harmonics and multipole expansions
- Hermite polynomials: the quantum harmonic oscillator
- Airy functions: turning points in quantum mechanics
- The Sturm-Liouville framework that unifies all of the above

## Prerequisites

- Taylor series and radius of convergence
- Chapters 1-3: second-order linear ODEs
- Basic complex numbers (we will use the complex plane to read off the radius of convergence)

---

## 1. Why Power Series?

You are studying heat conduction inside a cylinder and, after separation of variables, arrive at
$$x^2 y'' + xy' + (x^2 - n^2)y = 0.$$
This is the **Bessel equation**. None of our previous methods (characteristic equations for constant coefficients, integrating factors, variation of parameters) reach it because the coefficients are *functions of $x$*, not constants. Worse, the equation is singular at $x = 0$ -- precisely where the cylinder's axis lies and precisely where the answer must be physically well-defined.

**The strategy.** Postulate a series ansatz, plug it in, and let the differential equation generate its own coefficients through a recurrence. We will need *two* flavours of series:

1. Plain Taylor series $\sum a_k (x - x_0)^k$ at **ordinary points**.
2. Frobenius series $(x - x_0)^r \sum a_k (x - x_0)^k$ at **regular singular points**, where the exponent $r$ is itself determined by the equation.

The same machinery unlocks Bessel, Legendre, Hermite, Airy, Chebyshev, Laguerre, hypergeometric -- the entire pantheon of classical special functions.

---

## 2. Power Series at Ordinary Points

### 2.1 The method

Write the equation in standard form
$$y'' + P(x)\,y' + Q(x)\,y = 0.$$

A point $x_0$ is **ordinary** if both $P$ and $Q$ are analytic there. The recipe is then:

1. Verify $x_0$ is ordinary.
2. Postulate $y = \sum_{k=0}^{\infty} a_k (x - x_0)^k$.
3. Differentiate term by term and substitute into the ODE.
4. Collect coefficients of each power $(x - x_0)^k$ and equate to zero.
5. Solve the resulting **recurrence** for $a_k$ in terms of $a_0$ and $a_1$ (the two free parameters that match the two initial conditions).

A theorem of Fuchs guarantees that the resulting series converges in any open disk centred at $x_0$ that contains no singularity of $P$ or $Q$ -- and *the radius extends all the way out to the nearest singularity in the complex plane*. This is why complex analysis matters even when the ODE itself is real-valued.

![Disk of convergence around an ordinary point reaches out to the nearest singularity in the complex plane.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/05-laplace-transform/fig1_radius_of_convergence.png)
*Around an ordinary point $x_0$, the power-series solution converges in the largest disk that excludes every singularity of the coefficient functions. For an equation with $P(x) = 1/(1+x^2)$, the singularities sit at $x = \pm i$, so the radius of convergence around $x_0 = 0$ is $R = 1$.*

### 2.2 Warm-up: rediscovering $e^x$

Solve $y' = y$ by power series. Let $y = \sum a_k x^k$, then $y' = \sum (k+1) a_{k+1} x^k$. Matching coefficients gives
$$(k+1)\,a_{k+1} = a_k, \qquad a_k = \frac{a_0}{k!}.$$
So
$$y = a_0 \sum_{k=0}^{\infty} \frac{x^k}{k!} = a_0\,e^x.$$
The series method "rediscovered" the exponential. The lesson: even when we already know the answer, the recurrence is a mechanical procedure that cannot fail.

### 2.3 The Airy equation

$$y'' - x\,y = 0.$$
This appears whenever a wave (light, quantum probability, water) meets a turning point where the local wavenumber smoothly changes from real to imaginary -- the prototype example is quantum tunnelling near a linear potential barrier.

Both $P(x) = 0$ and $Q(x) = -x$ are entire, so $x = 0$ is ordinary. Substituting $y = \sum a_k x^k$ and matching powers gives the recurrence
$$a_{k+2} = \frac{a_{k-1}}{(k+1)(k+2)} \qquad (k \geq 1), \qquad a_2 = 0.$$
The coefficients split into two independent chains starting from $a_0$ and $a_1$, and the resulting two linearly independent solutions are the **Airy functions** $\text{Ai}(x)$ and $\text{Bi}(x)$. Airy decays exponentially for $x > 0$ (the classically forbidden region) and oscillates for $x < 0$ (the classically allowed region) -- a crisp visualisation of quantum tunnelling.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

x = np.linspace(-15, 5, 1000)
Ai, Aip, Bi, Bip = airy(x)

plt.figure(figsize=(10, 5))
plt.plot(x, Ai, 'b-', linewidth=2, label='Ai(x)')
plt.plot(x, Bi, 'r-', linewidth=2, label='Bi(x)')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.xlabel('x'); plt.ylabel('y')
plt.title("Airy Functions: Solutions of y'' - xy = 0")
plt.legend(); plt.ylim(-0.6, 1.2)
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()
```

---

## 3. Regular Singular Points and the Frobenius Method

### 3.1 Why a plain Taylor series can fail

Consider the toy ODE
$$4 x^2 y'' + y = 0.$$
By inspection, $y = \sqrt{x}$ is a solution -- but $\sqrt{x}$ has a vertical tangent at the origin and is *not* analytic there, so no series of the form $\sum a_k x^k$ can ever reproduce it. Some "fractional power" of $x$ has to enter the ansatz.

![A plain Taylor series cannot capture a vertical tangent at the origin; the Frobenius ansatz can.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/05-laplace-transform/fig2_frobenius_vs_taylor.png)
*Left: any polynomial through the origin has a finite slope at $x = 0$ and so can never match the cusp of $\sqrt{x}$. Right: the Frobenius ansatz $y = x^{1/2}\sum a_k x^k$ reproduces the cusp exactly. The indicial equation for $4x^2 y'' + y = 0$ has the double root $r = 1/2$.*

### 3.2 Regular singular points

The point $x_0$ is a **regular singular point** of $y'' + P(x) y' + Q(x) y = 0$ when $P$ or $Q$ is singular at $x_0$, but the milder products
$$(x - x_0)\,P(x), \qquad (x - x_0)^2\,Q(x)$$
are *both* analytic at $x_0$. Roughly: the singularity is at most a simple pole of $P$ and at most a double pole of $Q$.

### 3.3 The Frobenius ansatz

Try the generalised series
$$y = (x - x_0)^r \sum_{k=0}^{\infty} a_k (x - x_0)^k, \qquad a_0 \neq 0,$$
where the exponent $r$ is to be determined. Substituting into the ODE and equating the lowest power of $(x - x_0)$ gives the **indicial equation**
$$r(r - 1) + p_0\,r + q_0 = 0,$$
where $p_0 = \lim_{x \to x_0}(x-x_0) P(x)$ and $q_0 = \lim_{x \to x_0}(x-x_0)^2 Q(x)$. Its two roots $r_1 \geq r_2$ govern how to assemble two independent solutions:

| Case | Two independent solutions |
|------|---------------------------|
| $r_1 - r_2 \notin \mathbb{Z}$ | Two Frobenius series, one for each root. |
| $r_1 = r_2$ (repeated root) | One Frobenius series and a second containing $\ln(x-x_0)$. |
| $r_1 - r_2 \in \mathbb{Z}_{>0}$ | One Frobenius series; the second may or may not need a logarithmic term. |

The recurrence on $a_k$, obtained from the higher-order coefficient matches, is then unwound term by term.

---

## 4. Bessel Functions

### 4.1 The Bessel equation

$$x^2 y'' + x y' + (x^2 - n^2) y = 0.$$
The three places this equation appears in physics:

- **Vibrating circular drumheads** -- the radial part of $\nabla^2 u + k^2 u = 0$ in polar coordinates.
- **Cylindrical waveguides** -- electromagnetic modes inside a metal pipe.
- **Heat diffusion in a cylinder** -- transient cooling of a metal rod.

The point $x = 0$ is a regular singular point with $p_0 = 1$ and $q_0 = -n^2$, so the indicial equation $r^2 - n^2 = 0$ gives $r = \pm n$.

### 4.2 The Bessel function of the first kind

Taking $r = n$ (with $n \geq 0$) and solving the recurrence yields
$$\boxed{\; J_n(x) = \sum_{m=0}^{\infty} \frac{(-1)^m}{m!\,\Gamma(m+n+1)} \left(\frac{x}{2}\right)^{2m+n}.\;}$$

Three properties we will use repeatedly:

- **Recurrence:** $\displaystyle J_{n-1}(x) + J_{n+1}(x) = \frac{2n}{x}\,J_n(x)$ -- lets us slide between orders.
- **Asymptotic behaviour:** $\displaystyle J_n(x) \approx \sqrt{\frac{2}{\pi x}}\cos\!\left(x - \frac{n\pi}{2} - \frac{\pi}{4}\right)$ as $x \to \infty$ -- so $J_n$ behaves eventually like a damped cosine.
- **Zeros:** the positive zeros $j_{n,1} < j_{n,2} < \dots$ of $J_n$ are spaced approximately $\pi$ apart and *quantise the resonant frequencies of a circular membrane*.

![Bessel functions of the first kind, with the first zeros of $J_0$ marked.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/05-laplace-transform/fig3_bessel_first_kind.png)
*The first five Bessel functions $J_0,\dots,J_4$. The marked zeros $j_{0,1} \approx 2.405$, $j_{0,2} \approx 5.520$, $\ldots$ are the radial wavenumbers of the symmetric vibration modes of a unit circular drum.*

### 4.3 From Bessel zeros to drumhead modes

For a unit circular drum, separation of variables in polar coordinates gives normal modes
$$u_{n,m}(r,\varphi,t) = J_n(j_{n,m}\,r)\,\cos(n\varphi)\,\cos(c\,j_{n,m}\,t),$$
where $j_{n,m}$ is the $m$-th positive zero of $J_n$. The integer $n$ counts the number of *angular* nodal lines, and $m$ counts the number of *radial* nodal circles (including the boundary).

![Two-dimensional vibration modes of a circular drumhead, built from Bessel functions and angular cosines.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/05-laplace-transform/fig4_drumhead_modes.png)
*Snapshots of four normal modes of a unit circular drum. Dashed black curves are nodal lines (where the membrane stays still). Mode $(0,1)$ is the fundamental "breathing" mode; $(0,2)$ adds a radial node; $(1,1)$ adds an angular node; $(2,1)$ adds two angular nodes.*

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros

x = np.linspace(0.01, 20, 1000)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for n in range(5):
    ax1.plot(x, jv(n, x), linewidth=2, label=f'$J_{n}(x)$')
ax1.axhline(0, color='k', linewidth=0.5)
ax1.set_xlabel('x'); ax1.set_title('Bessel Functions of the First Kind')
ax1.legend(fontsize=9); ax1.set_ylim(-0.5, 1.1); ax1.grid(True, alpha=0.3)

for n in range(4):
    zeros = jn_zeros(n, 5)
    ax2.scatter(zeros, [n]*5, s=80, label=f'$J_{n}$ zeros')
ax2.set_xlabel('x'); ax2.set_ylabel('Order n')
ax2.set_title('Zeros of Bessel Functions (Drum Modes)')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
```

### 4.4 Bessel function of the second kind

For the negative root $r = -n$ when $n$ is a non-negative integer, the recurrence breaks down: a logarithmic term is forced. The resulting linearly independent partner is the **Bessel function of the second kind** $Y_n(x)$, which behaves as $Y_n(x) \sim -(2/\pi)(n-1)!\,(x/2)^{-n}$ near the origin and so blows up there. Physical solutions on a domain that *includes* the axis (a solid drum, a solid wire) must therefore use only $J_n$; problems on an *annular* domain (a coaxial cable) need both $J_n$ and $Y_n$.

---

## 5. Legendre Polynomials

### 5.1 The Legendre equation

$$(1 - x^2)\,y'' - 2x\,y' + n(n+1)\,y = 0.$$
This appears every time you separate variables in spherical coordinates: the angular part of the Laplacian, hydrogen-atom wavefunctions, gravitational and electrostatic multipole expansions, antenna radiation patterns. Here $x$ stands in for $\cos\theta$, so the natural domain is $[-1,1]$.

The points $x = \pm 1$ are regular singular points (where the leading coefficient $1-x^2$ vanishes), while $x = 0$ is ordinary.

### 5.2 Why polynomials?

The recurrence at the ordinary point $x = 0$ is
$$a_{k+2} = \frac{k(k+1) - n(n+1)}{(k+1)(k+2)}\,a_k.$$
When $n$ is a *non-negative integer*, the numerator vanishes at $k = n$ and the series **truncates** to a polynomial. The other linearly independent solution remains an infinite series and diverges at $x = \pm 1$, so demanding regularity at the poles of the sphere selects the polynomial solutions and quantises the parameter $n$.

Normalised so that $P_n(1) = 1$:

| $n$ | $P_n(x)$ |
|-----|-----------|
| 0 | $1$ |
| 1 | $x$ |
| 2 | $\frac{1}{2}(3x^2 - 1)$ |
| 3 | $\frac{1}{2}(5x^3 - 3x)$ |
| 4 | $\frac{1}{8}(35x^4 - 30x^2 + 3)$ |

### 5.3 Rodrigues' formula and orthogonality

A compact closed form:
$$P_n(x) = \frac{1}{2^n n!}\,\frac{d^n}{dx^n}(x^2 - 1)^n.$$
And the central identity that makes Legendre series possible:
$$\int_{-1}^{1} P_m(x)\,P_n(x)\,dx = \frac{2}{2n+1}\,\delta_{mn}.$$
Any reasonable function on $[-1,1]$ admits a **Legendre expansion** $f(x) = \sum_{n=0}^{\infty} c_n P_n(x)$ with $c_n = \frac{2n+1}{2}\int_{-1}^{1} f(x) P_n(x)\,dx$ -- the basis of the multipole expansion in electromagnetism and gravitation.

![Legendre polynomials on $[-1,1]$ with the orthogonality interval shaded.](./05-laplace-transform/fig5_legendre_polynomials.png)
*Legendre polynomials $P_0,\dots,P_5$. They all pass through $(1,1)$ by normalisation, and through $(-1, (-1)^n)$ by parity. Every $P_n$ has exactly $n$ simple zeros inside $(-1,1)$ -- the roots used by Gauss-Legendre quadrature.*

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

x = np.linspace(-1, 1, 500)
plt.figure(figsize=(8, 5))
for n in range(6):
    Pn = legendre(n)
    plt.plot(x, Pn(x), linewidth=2, label=f'$P_{n}(x)$')
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel('x'); plt.title('Legendre Polynomials')
plt.legend(fontsize=9); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 6. Hermite Polynomials and the Quantum Harmonic Oscillator

### 6.1 The Hermite equation

$$y'' - 2x\,y' + 2n\,y = 0.$$
For non-negative integer $n$, the polynomial solutions are the **Hermite polynomials** $H_n(x)$:
$$H_0 = 1, \quad H_1 = 2x, \quad H_2 = 4x^2 - 2, \quad H_3 = 8x^3 - 12x, \ldots$$
They satisfy the orthogonality $\int_{-\infty}^{\infty} H_m(x) H_n(x) e^{-x^2} dx = \sqrt{\pi}\,2^n n!\,\delta_{mn}$ -- with the Gaussian weight that will become the wavefunction envelope in a moment.

### 6.2 The quantum harmonic oscillator

Schrödinger's equation for a particle in a parabolic potential, written in dimensionless variables, is
$$-\tfrac{1}{2}\psi'' + \tfrac{1}{2} x^2 \psi = E\,\psi.$$
Substituting $\psi(x) = H_n(x) e^{-x^2/2}$ reduces it to the Hermite equation, and demanding that $\psi$ remain *normalisable* forces $n$ to be a non-negative integer. The eigenfunctions and eigenvalues are
$$\psi_n(x) = \frac{1}{\sqrt{2^n n!}\,\pi^{1/4}}\,H_n(x)\,e^{-x^2/2}, \qquad E_n = n + \tfrac{1}{2}\quad(\text{in units of }\hbar\omega).$$
Two physical lessons fall out for free:

- **Energy quantisation.** Only the integer values $n = 0, 1, 2, \ldots$ give normalisable solutions, so the energy ladder is discrete.
- **Zero-point energy.** Even the ground state has $E_0 = \tfrac{1}{2}\hbar\omega \neq 0$ -- a quantum particle in a parabolic well *cannot* sit perfectly still.

![Hermite-based wavefunctions of the quantum harmonic oscillator, stacked at their eigenenergies.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/05-laplace-transform/fig6_qho_wavefunctions.png)
*The first six eigenstates $\psi_n$ of the quantum harmonic oscillator, drawn at their eigenenergies $E_n = n + \tfrac{1}{2}$ on top of the parabolic potential. The number of nodes of $\psi_n$ equals $n$, a fingerprint of the underlying Hermite polynomial.*

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial

x = np.linspace(-4, 4, 500)
plt.figure(figsize=(8, 6))
for n in range(5):
    Hn = hermite(n)
    norm = 1 / np.sqrt(2**n * factorial(n)) * (1/np.pi)**0.25
    psi = norm * Hn(x) * np.exp(-x**2 / 2)
    plt.plot(x, psi + n, linewidth=2, label=f'n={n}')
    plt.fill_between(x, n, psi + n, alpha=0.2)
    plt.axhline(n, color='gray', linewidth=0.5, linestyle='--')

plt.xlabel('Position x'); plt.ylabel('Wavefunction + energy offset')
plt.title('Quantum Harmonic Oscillator Wavefunctions')
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
```

---

## 7. The Sturm-Liouville Unifying Framework

All the families above are eigenfunctions of self-adjoint **Sturm-Liouville problems**:
$$\frac{d}{dx}\!\left[p(x)\,\frac{dy}{dx}\right] + \bigl[q(x) + \lambda\,w(x)\bigr]\,y = 0,$$
with appropriate boundary conditions. Different choices of $p$, $q$, $w$, and the interval reproduce each special function:

| Function | $p(x)$ | $w(x)$ | Interval |
|----------|--------|--------|----------|
| Legendre $P_n$ | $1 - x^2$ | $1$ | $[-1, 1]$ |
| Bessel $J_n$ | $x$ | $x$ | $[0, a]$ |
| Hermite $H_n$ | $e^{-x^2}$ | $e^{-x^2}$ | $(-\infty, \infty)$ |
| Laguerre $L_n$ | $x e^{-x}$ | $e^{-x}$ | $[0, \infty)$ |
| Chebyshev $T_n$ | $\sqrt{1 - x^2}$ | $1/\sqrt{1 - x^2}$ | $[-1, 1]$ |

Three properties are inherited by every entry:

1. **Real eigenvalues** -- the operator is symmetric under the weight $w$.
2. **Orthogonality** -- $\int y_m y_n\,w\,dx = 0$ for $m \neq n$.
3. **Completeness** -- generic functions admit a *generalised Fourier series* in the eigenfunctions.

This is why "expand in spherical harmonics", "expand in Bessel modes", "expand in Hermite states" all feel the same: they *are* the same theorem with different weights.

---

## 8. Python: Using Special Functions

```python
from scipy import special
import numpy as np

# Bessel
print(f"J_0(5) = {special.jv(0, 5):.6f}")
print(f"First 5 zeros of J_0: {special.jn_zeros(0, 5)}")

# Legendre
print(f"P_3(0.5) = {special.eval_legendre(3, 0.5):.6f}")

# Hermite
print(f"H_4(1.0) = {special.eval_hermite(4, 1.0):.1f}")

# Airy
Ai, Aip, Bi, Bip = special.airy(2.0)
print(f"Ai(2) = {Ai:.6f}")
```

The figures in this article were produced by `scripts/figures/ode/05-power-series.py`; rerun it after pulling to regenerate every PNG into both the EN and ZH asset folders.

---

## Summary

| Concept | Key idea |
|---------|----------|
| Power series at ordinary points | Assume $y = \sum a_k x^k$, derive a recurrence for $a_k$ |
| Radius of convergence | Reaches the nearest singularity of $P$ or $Q$ in $\mathbb{C}$ |
| Regular singular points | Frobenius method: $y = x^r \sum a_k x^k$ |
| Indicial equation | Quadratic in $r$ that fixes the leading exponent |
| Bessel functions | Cylindrical geometry; zeros set drum-mode frequencies |
| Legendre polynomials | Spherical geometry; basis for multipole expansion |
| Hermite polynomials | Quantum harmonic oscillator wavefunctions |
| Sturm-Liouville theory | One framework, many orthogonal eigenfunctions |

---

## Exercises

**Basic**

1. Use power series to solve $y'' + y = 0$ near $x = 0$. Verify you recover $\sin x$ and $\cos x$.
2. Show that $x = 0$ is a regular singular point of $2x^2 y'' + 3x y' - (1 + x) y = 0$ and find its indicial equation.
3. Compute $P_4(x)$ and $P_5(x)$ from Rodrigues' formula and check the recurrence $(n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)$.

**Advanced**

4. For the Airy equation $y'' = xy$, write the recurrence and compute the first 8 non-zero coefficients of each independent solution.
5. Prove the Bessel recurrence $J_{n-1}(x) + J_{n+1}(x) = (2n/x) J_n(x)$ from the series for $J_n$.
6. Expand $f(r) = 1 - r^2$ on $[0, 1]$ as a Fourier-Bessel series in $J_0$ and verify the first few coefficients numerically.

**Programming**

7. Implement the series for $J_n(x)$ from scratch and compare with `scipy.special.jv` over $x \in [0, 30]$ for $n = 0, 1, 2$.
8. Re-create the drumhead figure for modes $(n, m) \in \{(0,1),(0,2),(1,1),(2,1),(3,1)\}$ and animate one period in time.
9. Plot the first 10 quantum harmonic oscillator probability densities $|\psi_n|^2$ on top of the parabolic potential, marking the classical turning points $x = \pm\sqrt{2n+1}$.

---

## References

- Arfken, Weber & Harris, *Mathematical Methods for Physicists*, Academic Press (2012)
- Bender & Orszag, *Advanced Mathematical Methods*, Springer (1999)
- NIST Digital Library of Mathematical Functions: [dlmf.nist.gov](https://dlmf.nist.gov/)

---

**Series Navigation**

| | |
|---|---|
| **Previous** | [Chapter 4: The Laplace Transform](/en/ode-chapter-04-constant-coefficients/) |
| **Current** | Chapter 5: Power Series and Special Functions |
| **Next** | [Chapter 6: Linear Systems and Phase Plane](/en/ode-chapter-06-power-series/) |
