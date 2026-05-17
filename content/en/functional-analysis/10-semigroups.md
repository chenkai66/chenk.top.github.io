---
title: "Functional Analysis (10): Semigroups of Operators — Evolution Equations in Infinite Dimensions"
date: 2021-10-19 09:00:00
tags:
  - functional-analysis
  - semigroups
  - pde
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "C₀-semigroups provide the abstract framework for evolution equations — the Hille-Yosida theorem characterizes which operators generate well-posed dynamics."
disableNunjucks: true
series_order: 10
series_total: 12
translationKey: "functional-analysis-10"
---

Consider the simplest ordinary differential equation: $u'(t) = au(t)$, $u(0) = u_0$, where $a$ is a real number. The solution is $u(t) = e^{at}u_0$. The family of maps $T(t): u_0 \mapsto e^{at}u_0$ forms a one-parameter group satisfying $T(0) = I$, $T(t+s) = T(t)T(s)$, and $T(t) \to I$ as $t \to 0$.

Now replace the scalar $a$ by a bounded operator $A$ on a Banach space $X$. The exponential $e^{tA} = \sum_{n=0}^\infty (tA)^n / n!$ converges in operator norm, and $T(t) = e^{tA}$ solves the abstract Cauchy problem $u'(t) = Au(t)$, $u(0) = u_0$, for every initial condition $u_0 \in X$.

But the operators that arise in PDE — the Laplacian, wave operators, Schrodinger operators — are *unbounded*. The power series for $e^{tA}$ diverges. We cannot simply exponentiate. Yet the physical intuition is clear: the heat equation, the wave equation, and the Schrodinger equation all describe time evolution, and we expect the map $u_0 \mapsto u(t)$ to be a well-defined bounded operator for each $t \ge 0$.

The theory of $C_0$-semigroups resolves this tension. It identifies the precise conditions under which an unbounded operator $A$ generates a family of bounded operators $\{T(t)\}_{t \ge 0}$ that evolve solutions forward in time.

---

## From Finite ODE to Infinite-Dimensional Evolution

### The finite-dimensional picture

In $\mathbb{R}^n$, the system $u'(t) = Au(t)$ with $A \in \mathbb{R}^{n \times n}$ has the matrix exponential solution $u(t) = e^{tA}u_0$. The key properties are:

![Evolution semigroup and generator](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/10-semigroups/fa_fig4_big_theorems.png)


1. **Semigroup property:** $e^{(t+s)A} = e^{tA}e^{sA}$ for all $t, s \ge 0$.
2. **Strong continuity:** $e^{tA} \to I$ as $t \to 0$ (in any matrix norm).
3. **Generator recovery:** $A = \lim_{t \to 0^+} (e^{tA} - I)/t$.
4. **Growth bound:** $\|e^{tA}\| \le Me^{\omega t}$ for constants $M, \omega$ depending on $A$.

The infinite-dimensional theory abstracts these properties, asking: given an operator $A$ (possibly unbounded), when does there exist a family of bounded operators $\{T(t)\}_{t \ge 0}$ satisfying properties 1-4? And if so, how do we construct it?

### Why "semigroup" and not "group"?

For the heat equation $u_t = \Delta u$, solutions smooth out as time progresses forward but cannot generally be continued backward in time (the backward heat equation is ill-posed). The evolution operators $T(t)$ exist only for $t \ge 0$, forming a *semigroup* rather than a group. This asymmetry reflects the irreversibility encoded in parabolic equations.

For the Schrodinger equation $iu_t = -\Delta u$, the evolution is reversible — $T(t)$ extends to a group. But the general theory focuses on semigroups, which encompass both cases.

---

## Strongly Continuous (C₀) Semigroups: Definition and Basic Properties

**Definition.** A **$C_0$-semigroup** (or strongly continuous semigroup) on a Banach space $X$ is a family $\{T(t)\}_{t \ge 0}$ of bounded linear operators $T(t) \in B(X)$ satisfying:

1. $T(0) = I$ (identity operator),
2. $T(t+s) = T(t)T(s)$ for all $t, s \ge 0$ (semigroup property),
3. $\lim_{t \to 0^+} T(t)x = x$ for every $x \in X$ (strong continuity at $0$).

The "$C_0$" stands for "class zero," Hille's original notation for continuity at the origin.

**Remark on terminology.** A *semigroup* in the algebraic sense is a set with an associative binary operation. The operators $\{T(t)\}$ form a semigroup under composition: $T(t) \circ T(s) = T(t+s)$. The parameter $t \ge 0$ makes it a "one-parameter semigroup." The adjective "strongly continuous" (or $C_0$) specifies the topology in which continuity holds — pointwise convergence on vectors in $X$, not convergence in operator norm.

The distinction between strong and uniform continuity is significant: a $C_0$-semigroup is uniformly continuous ($\|T(t) - I\| \to 0$) if and only if its generator $A$ is bounded. Since we are interested in unbounded generators (differential operators), we must work with the weaker notion of strong continuity.

**Proposition (uniform boundedness on compacts).** If $\{T(t)\}_{t \ge 0}$ is a $C_0$-semigroup, then for every $\tau > 0$ there exists $M_\tau \ge 1$ such that $\|T(t)\| \le M_\tau$ for all $t \in [0, \tau]$.

*Proof.* By strong continuity, for each $x \in X$ the orbit $\{T(t)x : t \in [0,1]\}$ is bounded (as the continuous image of a compact set in the strong topology of $X$, this follows from the continuity $t \mapsto T(t)x$). By the uniform boundedness principle (Banach-Steinhaus), $\sup_{t \in [0,1]} \|T(t)\| = M < \infty$. For general $\tau$, write $t = n + s$ with $n \in \mathbb{N}$ and $0 \le s < 1$, then $\|T(t)\| \le M^{n+1} \le M^{\tau + 1}$. $\square$

**Corollary (exponential growth bound).** There exist constants $M \ge 1$ and $\omega \in \mathbb{R}$ such that $\|T(t)\| \le Me^{\omega t}$ for all $t \ge 0$.

The infimum of all admissible $\omega$ is called the **growth bound** $\omega_0$ of the semigroup. When $\omega_0 \le 0$, the semigroup does not grow; when $\omega_0 < 0$, it decays exponentially.

**Special classes:**
- **Contraction semigroup:** $\|T(t)\| \le 1$ for all $t \ge 0$ (equivalently, $M = 1$, $\omega = 0$).
- **Uniformly continuous semigroup:** $\lim_{t \to 0^+} \|T(t) - I\| = 0$ (convergence in operator norm, not just pointwise). These are precisely the semigroups generated by bounded operators: $T(t) = e^{tA}$ with $A \in B(X)$. They are too restrictive for PDE applications.

### Basic examples

**Example 1 (Translation semigroup).** On $X = L^p(\mathbb{R})$ for $1 \le p < \infty$, define $(T(t)f)(x) = f(x+t)$. This is a $C_0$-semigroup of isometries. The generator turns out to be $Af = f'$ with domain $\mathcal{D}(A) = W^{1,p}(\mathbb{R})$.

**Example 2 (Heat semigroup).** On $X = L^2(\mathbb{R}^n)$, define

$$
(T(t)f)(x) = \frac{1}{(4\pi t)^{n/2}} \int_{\mathbb{R}^n} e^{-|x-y|^2/(4t)} f(y) \, dy, \quad t > 0,
$$

and $T(0) = I$. This is a $C_0$-semigroup of contractions. Its generator is the Laplacian $A = \Delta$ with domain $\mathcal{D}(A) = H^2(\mathbb{R}^n)$.

**Example 3 (Multiplication semigroup).** On $L^2(\mathbb{R})$, let $q: \mathbb{R} \to \mathbb{C}$ be measurable with $\text{Re}(q) \le \omega$, and define $(T(t)f)(x) = e^{tq(x)}f(x)$. This is a $C_0$-semigroup with $\|T(t)\| \le e^{\omega t}$. The generator is the multiplication operator $Af = qf$ with its natural domain.

**Example 4 (Poisson semigroup).** On $L^2(\mathbb{R}^n)$, the Poisson semigroup $(T(t)f)(x) = c_n \int_{\mathbb{R}^n} \frac{t}{(|x-y|^2 + t^2)^{(n+1)/2}} f(y)\,dy$ solves the Laplace equation in the upper half-space. Its generator is $-(-\Delta)^{1/2}$, the negative of the fractional Laplacian of order $1/2$. This example shows that semigroups naturally arise from operators that are not standard differential operators.

### Non-examples

Not every family of bounded operators parameterized by $t \ge 0$ is a $C_0$-semigroup. The family $T(t) = I$ for $t = 0$ and $T(t) = 0$ for $t > 0$ satisfies the semigroup property but not strong continuity. The "backward heat semigroup" (attempting to define $T(t)f = $ solution of the backward heat equation at time $-t$) fails because the backward heat equation is ill-posed — the operators do not exist as bounded maps on $L^2$.

---

## The Generator and Its Domain

**Definition.** The **(infinitesimal) generator** of a $C_0$-semigroup $\{T(t)\}_{t \ge 0}$ is the operator $A$ defined by

$$
Ax = \lim_{t \to 0^+} \frac{T(t)x - x}{t},
$$

with domain $\mathcal{D}(A) = \{x \in X : \text{the above limit exists}\}$.

**Theorem (properties of the generator).** Let $A$ be the generator of a $C_0$-semigroup $\{T(t)\}$.

1. $\mathcal{D}(A)$ is dense in $X$.
2. $A$ is a closed operator.
3. For $x \in \mathcal{D}(A)$, the map $t \mapsto T(t)x$ is continuously differentiable and $\frac{d}{dt}T(t)x = AT(t)x = T(t)Ax$.
4. For $x \in \mathcal{D}(A)$, $T(t)x \in \mathcal{D}(A)$ for all $t \ge 0$.
5. The generator uniquely determines the semigroup: if two $C_0$-semigroups have the same generator, they are identical.

*Proof of (1).* For $x \in X$, define $x_t = \frac{1}{t}\int_0^t T(s)x \, ds$. The integral exists as a Bochner integral in $X$. Then $x_t \to x$ as $t \to 0^+$ (by strong continuity). We claim $x_t \in \mathcal{D}(A)$:

$$
\frac{T(h)x_t - x_t}{h} = \frac{1}{t}\left(\frac{1}{h}\int_0^t [T(s+h) - T(s)]x \, ds\right) = \frac{1}{t}\left(\frac{1}{h}\int_t^{t+h} T(s)x \, ds - \frac{1}{h}\int_0^h T(s)x \, ds\right).
$$

As $h \to 0^+$, the first integral converges to $T(t)x$ and the second to $x$, giving $Ax_t = \frac{1}{t}(T(t)x - x)$. Since $x_t \to x$, the domain $\mathcal{D}(A)$ is dense. $\square$

*Proof of (3).* For $x \in \mathcal{D}(A)$:

$$
\frac{T(t+h)x - T(t)x}{h} = T(t)\frac{T(h)x - x}{h} \to T(t)Ax \quad \text{as } h \to 0^+.
$$

The right derivative is $T(t)Ax$. Since $T(t)x \in \mathcal{D}(A)$ (provable by a similar argument), we also get the left derivative, establishing continuous differentiability. The identity $AT(t)x = T(t)Ax$ follows from the commutativity of $T(t)$ with $A$ on $\mathcal{D}(A)$. $\square$

*Proof of (2): closedness.* Suppose $x_n \in \mathcal{D}(A)$, $x_n \to x$, and $Ax_n \to y$. For any $t > 0$:

$$
T(t)x_n - x_n = \int_0^t T(s)Ax_n \, ds.
$$

Taking $n \to \infty$: $T(t)x - x = \int_0^t T(s)y \, ds$ (justified by uniform boundedness of $T(s)$ on $[0, t]$ and the dominated convergence theorem for Bochner integrals). Dividing by $t$ and sending $t \to 0^+$:

$$
\frac{T(t)x - x}{t} = \frac{1}{t}\int_0^t T(s)y \, ds \to y,
$$

so $x \in \mathcal{D}(A)$ and $Ax = y$. Hence $A$ is closed. $\square$

*Proof of (5): uniqueness.* This is a consequence of the identity $T(t)x - x = \int_0^t T(s)Ax \, ds$ for $x \in \mathcal{D}(A)$. If two semigroups $T_1(t)$ and $T_2(t)$ share the same generator $A$, then for $x \in \mathcal{D}(A)$, the function $s \mapsto T_1(t-s)T_2(s)x$ is differentiable (in $s$) with derivative zero, hence constant. Evaluating at $s = 0$ and $s = t$ gives $T_1(t)x = T_2(t)x$. By density of $\mathcal{D}(A)$, $T_1(t) = T_2(t)$ for all $t$. $\square$

### The resolvent of the generator

The generator $A$ and its resolvent are intimately connected to the semigroup via the **Laplace transform identity**: for $\text{Re}(\lambda) > \omega_0$ (the growth bound),

$$
R(\lambda, A)x = (\lambda I - A)^{-1}x = \int_0^\infty e^{-\lambda t}T(t)x \, dt \quad \text{for all } x \in X.
$$

This is a Bochner integral in $X$, converging absolutely since $\|e^{-\lambda t}T(t)x\| \le Me^{(\omega - \text{Re}(\lambda))t}\|x\|$. The identity shows that the resolvent is the Laplace transform of the semigroup — a connection that underlies the entire Hille-Yosida theory.

From this identity one derives the resolvent estimate $\|R(\lambda, A)^n\| \le M/(\text{Re}(\lambda) - \omega)^n$ for $\text{Re}(\lambda) > \omega$ and $n \ge 1$, which appears in the general Hille-Yosida theorem.

### The abstract Cauchy problem

Property (3) means that $u(t) = T(t)u_0$ solves the **abstract Cauchy problem**

$$
\begin{cases} u'(t) = Au(t), & t > 0, \\ u(0) = u_0, \end{cases}
$$

for every initial condition $u_0 \in \mathcal{D}(A)$. The solution is *classical* (continuously differentiable in $X$ and satisfies the equation pointwise in $t$) precisely when $u_0 \in \mathcal{D}(A)$. For $u_0 \in X \setminus \mathcal{D}(A)$, the orbit $T(t)u_0$ is a *mild solution* — continuous but not differentiable at $t = 0$.

The concept of mild solutions is important because in many applications, the initial data is not smooth enough to lie in $\mathcal{D}(A)$. For the heat equation, an initial temperature distribution in $L^2$ (say, a step function) does not lie in $H^2$, but the heat semigroup still produces a well-defined $L^2$-valued function $t \mapsto T(t)u_0$ that is continuous for $t \ge 0$ and smooth for $t > 0$.

### Inhomogeneous equations

The abstract Cauchy problem can be extended to inhomogeneous equations: $u'(t) = Au(t) + f(t)$, $u(0) = u_0$. The solution is given by the **variation of constants formula** (also called Duhamel's formula):

$$
u(t) = T(t)u_0 + \int_0^t T(t-s)f(s)\,ds.
$$

This is the semigroup analogue of the familiar formula for first-order linear ODE. The integral term accounts for the accumulated effect of the forcing function $f$, propagated by the semigroup.

---

## The Hille-Yosida Generation Theorem

The fundamental question of semigroup theory is: *which operators generate $C_0$-semigroups?* The Hille-Yosida theorem gives a complete answer for contraction semigroups.

**Theorem (Hille-Yosida, 1948).** A linear operator $A$ on a Banach space $X$ is the generator of a $C_0$-semigroup of contractions if and only if:

1. $A$ is closed and densely defined.
2. The resolvent set $\rho(A)$ contains $(0, \infty)$, and for every $\lambda > 0$,

$$
\|(\lambda I - A)^{-1}\| \le \frac{1}{\lambda}.
$$

More generally, $A$ generates a $C_0$-semigroup with $\|T(t)\| \le Me^{\omega t}$ if and only if $A$ is closed, densely defined, $(\omega, \infty) \subset \rho(A)$, and

$$
\|(\lambda I - A)^{-n}\| \le \frac{M}{(\lambda - \omega)^n} \quad \text{for all } \lambda > \omega, \; n \ge 1.
$$

**Proof outline (contraction case, sufficiency).** The proof constructs the semigroup via the **Yosida approximation**. For $\lambda > 0$, define the bounded operator

$$
A_\lambda = \lambda A R(\lambda, A) = \lambda^2 R(\lambda, A) - \lambda I,
$$

where $R(\lambda, A) = (\lambda I - A)^{-1}$ is the resolvent. Each $A_\lambda$ is bounded, so $T_\lambda(t) = e^{tA_\lambda}$ is a well-defined uniformly continuous semigroup.

**Step 1 (Yosida approximation converges on the domain).** For $x \in \mathcal{D}(A)$, $A_\lambda x \to Ax$ as $\lambda \to \infty$. This follows from $A_\lambda x = \lambda R(\lambda, A)Ax$ and the fact that $\lambda R(\lambda, A) \to I$ strongly (a standard resolvent identity argument).

**Step 2 (Uniform bound).** The contraction resolvent estimate $\|\lambda R(\lambda, A)\| \le 1$ implies $\|T_\lambda(t)\| \le e^{t\|A_\lambda\|}$. A sharper computation using $A_\lambda = \lambda^2 R(\lambda, A) - \lambda I$ and $\|\lambda R(\lambda, A)\| \le 1$ gives:

$$
\|T_\lambda(t)\| = \|e^{t(\lambda^2 R(\lambda,A) - \lambda I)}\| \le e^{-\lambda t} e^{\lambda^2 t \|R(\lambda,A)\|} \le e^{-\lambda t} e^{\lambda t} = 1.
$$

So each $T_\lambda(t)$ is a contraction.

**Step 3 (Convergence).** For $x \in \mathcal{D}(A)$:

$$
T_\lambda(t)x - T_\mu(t)x = \int_0^t \frac{d}{ds}[T_\lambda(t-s)T_\mu(s)x] \, ds = \int_0^t T_\lambda(t-s)T_\mu(s)(A_\mu x - A_\lambda x) \, ds.
$$

Using the contraction bound and Step 1, $\|T_\lambda(t)x - T_\mu(t)x\| \le t\|A_\mu x - A_\lambda x\| \to 0$ as $\lambda, \mu \to \infty$. By density of $\mathcal{D}(A)$ and the uniform bound, $T_\lambda(t)x$ converges for all $x \in X$, uniformly on compact time intervals.

**Step 4 (Verification).** Define $T(t)x = \lim_{\lambda \to \infty} T_\lambda(t)x$. One verifies that $T(t)$ is a $C_0$-contraction semigroup and that its generator is $A$. $\square$

**Significance.** The Hille-Yosida theorem is the *existence and uniqueness theorem for linear evolution equations in Banach spaces*. It reduces the problem of solving an infinite-dimensional ODE to verifying resolvent estimates — an algebraic/analytic condition that can be checked in concrete cases.

### Historical note

Einar Hille (1948) and Kosaku Yosida (1948) independently proved this theorem. Hille's approach used Laplace transforms and complex analysis; Yosida's used the approximation scheme described above (now called the Yosida approximation). The two approaches are complementary: Hille's is elegant for proving necessity, while Yosida's construction is more explicit and generalizes better.

The theorem has been extended in many directions: to semigroups on locally convex spaces, to bi-continuous semigroups, to integrated semigroups (which handle operators that are not densely defined), and to nonlinear semigroups (the Crandall-Liggett theorem for accretive operators in Banach spaces).

---

## Lumer-Phillips Theorem and Dissipative Operators

The Hille-Yosida resolvent estimates can be difficult to verify directly. The Lumer-Phillips theorem provides a more geometric criterion using the concept of dissipativity.

**Definition.** An operator $A$ on a Banach space $X$ is **dissipative** if for every $x \in \mathcal{D}(A)$, there exists $x^* \in J(x)$ (the duality set: $x^* \in X^*$ with $\langle x, x^* \rangle = \|x\|^2 = \|x^*\|^2$) such that

$$
\text{Re}\,\langle Ax, x^* \rangle \le 0.
$$

On a Hilbert space $H$, this simplifies to $\text{Re}\,\langle Ax, x \rangle \le 0$ for all $x \in \mathcal{D}(A)$.

**Intuition.** Dissipativity means the operator "dissipates energy." For the heat equation, $\text{Re}\,\langle \Delta u, u \rangle = -\|\nabla u\|^2 \le 0$, reflecting the physical fact that heat diffusion reduces temperature gradients.

**Theorem (Lumer-Phillips).** Let $A$ be a densely defined operator on a Banach space $X$. Then $A$ generates a $C_0$-semigroup of contractions if and only if:

1. $A$ is dissipative, and
2. $\text{Range}(\lambda I - A) = X$ for some (equivalently, all) $\lambda > 0$.

*Proof sketch.* Necessity: the contraction property implies dissipativity via a direct computation. Sufficiency: dissipativity implies $\|(\lambda I - A)x\| \ge \lambda \|x\|$ for all $x \in \mathcal{D}(A)$ and $\lambda > 0$ (this is the key estimate). Combined with the range condition, this gives $(\lambda I - A)^{-1}$ exists on all of $X$ with $\|(\lambda I - A)^{-1}\| \le 1/\lambda$, which is exactly the Hille-Yosida condition. One also verifies closedness using the surjectivity of $\lambda I - A$. $\square$

**Example: Verifying Lumer-Phillips for the Laplacian.** On $L^2(\Omega)$, the Laplacian $A = \Delta$ with domain $\mathcal{D}(A) = H^2(\Omega) \cap H_0^1(\Omega)$ is dissipative: $\text{Re}\,\langle \Delta u, u \rangle = -\|\nabla u\|^2 \le 0$. The range condition holds because $(\lambda I - \Delta)u = f$ is an elliptic equation with unique solution for every $f \in L^2$ and $\lambda > 0$ (by the Lax-Milgram theorem, which we will prove in Article 12). So Lumer-Phillips gives a contraction semigroup — exactly the heat semigroup on $\Omega$.

This example illustrates a typical workflow: (1) identify the operator and its natural domain, (2) verify dissipativity (often by an integration-by-parts computation), (3) verify the range condition (often by solving an elliptic equation), (4) conclude existence of the semigroup.

**Corollary.** On a Hilbert space, if $A$ is densely defined, symmetric ($\langle Ax, y \rangle = \langle x, Ay \rangle$), and non-positive ($\langle Ax, x \rangle \le 0$), then $A$ generates a $C_0$-contraction semigroup if and only if the range of $\lambda I - A$ is dense for some $\lambda > 0$.

---

## Application: The Heat Equation via Semigroups

Let us work through the heat equation on a bounded domain to see the theory in action.

**Problem.** Let $\Omega \subset \mathbb{R}^n$ be bounded with smooth boundary. Solve

$$
\begin{cases} u_t = \Delta u & \text{in } \Omega \times (0, \infty), \\ u = 0 & \text{on } \partial\Omega \times (0, \infty), \\ u(0) = u_0 & \text{in } \Omega. \end{cases}
$$

**Step 1: Set up the operator.** Let $X = L^2(\Omega)$ and define $A = \Delta$ (the Laplacian) with domain $\mathcal{D}(A) = H^2(\Omega) \cap H_0^1(\Omega)$ (the Friedrichs extension from the previous article). We know $A$ is self-adjoint and non-positive: $\langle Au, u \rangle = -\|\nabla u\|^2 \le 0$.

**Step 2: Verify the Lumer-Phillips conditions.**

- *Dissipativity:* $\text{Re}\,\langle Au, u \rangle = -\|\nabla u\|^2 \le 0$. Check.
- *Range condition:* For $\lambda > 0$, the equation $(\lambda I - A)u = f$ reads $\lambda u - \Delta u = f$ with $u \in H^2 \cap H_0^1$. This is an elliptic BVP. By elliptic theory (Lax-Milgram, which we will prove in Article 12), for every $f \in L^2(\Omega)$ there exists a unique $u \in H^2(\Omega) \cap H_0^1(\Omega)$ solving this equation. So $\text{Range}(\lambda I - A) = L^2(\Omega)$. Check.

**Step 3: Apply Lumer-Phillips.** The operator $A = \Delta$ (with Dirichlet domain) generates a $C_0$-contraction semigroup $\{T(t)\}_{t \ge 0}$ on $L^2(\Omega)$.

**Step 4: Interpret.** The solution to the heat equation is $u(t) = T(t)u_0$. For $u_0 \in \mathcal{D}(A) = H^2 \cap H_0^1$, this is a classical solution: $u \in C^1((0,\infty); L^2) \cap C([0,\infty); H^2)$, satisfying $u_t = \Delta u$ in $L^2$ for all $t > 0$. For $u_0 \in L^2(\Omega)$, it is a mild solution.

**Step 5: Further properties.** The spectral theorem for the self-adjoint operator $A$ gives eigenvalues $-\lambda_k$ with $0 < \lambda_1 \le \lambda_2 \le \cdots \to \infty$ and orthonormal eigenfunctions $\{e_k\}$. The semigroup has the explicit representation

$$
T(t)u_0 = \sum_{k=1}^\infty e^{-\lambda_k t} \langle u_0, e_k \rangle e_k.
$$

This immediately shows:
- **Exponential decay:** $\|T(t)u_0\| \le e^{-\lambda_1 t}\|u_0\|$ — the solution decays at a rate determined by the first eigenvalue.
- **Smoothing:** For $t > 0$, the rapid decay of $e^{-\lambda_k t}$ as $k \to \infty$ means $T(t)u_0$ is in $\mathcal{D}(A^n)$ for every $n$, hence is infinitely smooth. The heat semigroup converts rough initial data into smooth solutions instantaneously.
- **Irreversibility:** The series cannot be run backward ($t < 0$) because $e^{-\lambda_k t} \to \infty$ — confirming the ill-posedness of the backward heat equation.

### The wave equation: a group example

For contrast, consider the wave equation $u_{tt} = \Delta u$ on $\Omega$ with Dirichlet boundary conditions. Rewriting as a first-order system $U' = AU$ where $U = (u, u_t)^T$ and $A = \begin{pmatrix} 0 & I \\ \Delta & 0 \end{pmatrix}$, the operator $A$ generates a $C_0$-*group* (not just a semigroup) on $H_0^1(\Omega) \times L^2(\Omega)$. The group property $U(t)U(-t) = I$ reflects the time-reversibility of the wave equation.

Unlike the heat semigroup, the wave group does not smooth initial data: singularities propagate at finite speed (Huygens' principle). The wave equation preserves the $H^1 \times L^2$ energy $\|u(t)\|_{H^1}^2 + \|u_t(t)\|_{L^2}^2$ exactly, while the heat equation dissipates energy monotonically. This stark difference between parabolic and hyperbolic evolution is captured cleanly by the semigroup/group distinction.

### Beyond Hilbert spaces: the Schrodinger group

The Schrodinger equation $iu_t = -\Delta u$ generates a unitary group $U(t) = e^{it\Delta}$ on $L^2(\mathbb{R}^n)$ (by Stone's theorem, which we will prove in Article 12). Like the wave group, it is reversible; like the heat semigroup, its generator is the Laplacian. The difference lies in the factor $i$: the Schrodinger group preserves $L^2$ norms (probability conservation) but does not dissipate energy or smooth data in the $L^2$ sense (though it does have dispersive estimates in $L^p$).

### Analytic semigroups

An important subclass of $C_0$-semigroups consists of **analytic semigroups**: those for which $t \mapsto T(t)$ extends to an analytic function on a sector $\{z \in \mathbb{C} : |\arg z| < \theta\}$ for some $\theta > 0$. The heat semigroup is analytic (the heat kernel makes sense for complex time in a sector), while the wave group and Schrodinger group are not.

Analytic semigroups enjoy additional regularity: $T(t)x \in \mathcal{D}(A^n)$ for all $t > 0$, $n \ge 1$, and $x \in X$ (not just $x \in \mathcal{D}(A)$). This is the abstract manifestation of the instantaneous smoothing property of parabolic equations. A generator $A$ produces an analytic semigroup if and only if the resolvent $(\lambda I - A)^{-1}$ satisfies $\|\lambda(\lambda I - A)^{-1}\| \le M$ in a larger sector $\{|\arg(\lambda - \omega)| < \pi/2 + \theta\}$ (not just the right half-plane).

### Perturbation theory

One of the most useful features of semigroup theory is its perturbation results.

**Theorem (Bounded perturbation).** If $A$ generates a $C_0$-semigroup and $B \in B(X)$ is a bounded operator, then $A + B$ also generates a $C_0$-semigroup.

This is the infinite-dimensional analogue of the fact that $e^{t(A+B)}$ is well-defined when $B$ is bounded. The proof uses the Dyson series (perturbation expansion) or the Duhamel formula. In applications, this allows one to treat lower-order terms (bounded perturbations of the principal part) without redoing the entire generation argument.

**Theorem (Relatively bounded perturbation, Kato).** If $A$ generates a contraction semigroup and $B$ is $A$-bounded with relative bound less than 1 (meaning $\|Bx\| \le a\|x\| + b\|Ax\|$ for all $x \in \mathcal{D}(A)$ with $b < 1$), then $A + B$ also generates a $C_0$-semigroup.

Kato's perturbation theorem is the workhorse of mathematical physics. It is used to establish self-adjointness and semigroup generation for Schrodinger operators $-\Delta + V$ where the potential $V$ is unbounded (e.g., the Coulomb potential $V(x) = -e^2/|x|$). The idea is that the singular potential, while unbounded, grows slower than the Laplacian and hence can be treated as a "relatively bounded" perturbation.

### Asymptotic behavior

For many applications, the long-time behavior of the semigroup is as important as its existence. Key results include:

- **Exponential stability:** If $\omega_0 < 0$, then $\|T(t)\| \to 0$ exponentially. This corresponds to decay of solutions — physically, to energy dissipation. For the heat semigroup on a bounded domain, $\omega_0 = -\lambda_1$ where $\lambda_1$ is the first eigenvalue of $-\Delta$.

- **Spectral mapping theorem (partial):** For analytic semigroups, $\sigma(T(t)) \setminus \{0\} = e^{t\sigma(A)}$. This connects the spectrum of the semigroup to the spectrum of its generator. For general $C_0$-semigroups, only the weaker inclusion $e^{t\sigma(A)} \subset \sigma(T(t))$ holds; the full spectral mapping theorem can fail.

- **Ergodic theorems:** For bounded semigroups, the Cesaro mean $\frac{1}{t}\int_0^t T(s)\,ds$ converges strongly to a projection onto $\ker(A)$ as $t \to \infty$. This is the continuous analogue of the mean ergodic theorem for operators.

---

## What's Next

We have seen how $C_0$-semigroups provide a rigorous framework for evolution equations, with the Hille-Yosida and Lumer-Phillips theorems characterizing which operators generate well-posed dynamics. The heat equation illustrated the theory in a concrete PDE setting, revealing exponential decay, instantaneous smoothing, and irreversibility as natural consequences of the spectral properties of the generator.

In the next article, we develop the theory of **distributions and Sobolev spaces** — the function spaces that underlie the domains of differential operators and make weak solutions to PDE rigorous. These spaces are the natural habitat for the operators and semigroups we have studied.

---

*This is Part 10 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 9 — Unbounded Operators](/en/functional-analysis/09-unbounded-operators/)*

*Next: [Part 11 — Distributions and Sobolev Spaces](/en/functional-analysis/11-distributions-sobolev/)*
