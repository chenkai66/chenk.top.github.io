---
title: "Functional Analysis (9): Unbounded Operators — When Boundedness Fails"
date: 2021-10-17 09:00:00
tags:
  - functional-analysis
  - unbounded-operators
  - quantum-mechanics
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "Differential operators like d/dx are unbounded but essential — closed operators, domains, and the distinction between symmetric and self-adjoint resolve the difficulties."
disableNunjucks: true
series_order: 9
series_total: 12
translationKey: "functional-analysis-9"
---

Throughout our journey in functional analysis, we have worked almost exclusively with bounded operators: continuous linear maps $T: X \to Y$ satisfying $\|Tx\| \le M\|x\|$ for some constant $M$ and all $x$. The spectral theorem for compact self-adjoint operators, the Banach-Steinhaus theorem, the open mapping theorem — all of these relied, at some point, on the operator being bounded.

But the most important operators in mathematical physics are *not* bounded. The momentum operator $-i\hbar \frac{d}{dx}$ in quantum mechanics, the Laplacian $-\Delta$ in PDE theory, and multiplication by an unbounded function on $L^2$ all fail to satisfy any bound of the form $\|Tx\| \le M\|x\|$. These operators are defined only on a *dense subspace* of the Hilbert space, not on the entire space. Handling them requires new definitions, new care with domains, and a precise distinction between "symmetric" and "self-adjoint" that has no analogue in the bounded case.

This article develops the theory of unbounded operators on Hilbert spaces, culminating in the spectral theorem for unbounded self-adjoint operators.

---

## Why Unbounded Operators Matter

### Quantum observables

In quantum mechanics, the state of a particle on the real line is a unit vector $\psi \in L^2(\mathbb{R})$. The position observable is represented by the multiplication operator

![Bounded vs unbounded operators](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_fig3_operators.png)


$$
(Q\psi)(x) = x\psi(x),
$$

and the momentum observable by the differential operator

$$
(P\psi)(x) = -i\hbar \frac{d\psi}{dx}.
$$

Neither operator is bounded. For position: take $\psi_n(x) = n^{1/2}\chi_{[n, n+1/n]}(x)$, normalized in $L^2$. Then $\|Q\psi_n\|^2 \ge n^2 \|\psi_n\|^2$, so no finite bound $M$ works. For momentum: the derivative of a highly oscillatory function $e^{inx}$ (suitably truncated) grows like $n$, again defeating any uniform bound.

The Heisenberg uncertainty principle $\Delta Q \cdot \Delta P \ge \hbar/2$ depends essentially on the non-commutativity $[Q, P] = i\hbar I$, which is *impossible* for two bounded operators on a Hilbert space (the trace of $[A, B]$ is zero for bounded operators on a finite-dimensional space, and the infinite-dimensional analogue via Wielandt's theorem shows $[A, B] = iI$ has no bounded solutions). Unbounded operators are therefore not a pathology to be avoided but a fundamental feature of quantum theory.

### Differential operators

In PDE theory, the central object is often a differential operator like the Laplacian

$$
-\Delta u = -\sum_{j=1}^n \frac{\partial^2 u}{\partial x_j^2}.
$$

On $L^2(\Omega)$ for a bounded domain $\Omega \subset \mathbb{R}^n$, this is unbounded: the eigenfunctions of $-\Delta$ (with Dirichlet boundary conditions) have eigenvalues $\lambda_k \to \infty$, so no uniform bound $\|-\Delta u\| \le M\|u\|$ is possible.

Understanding well-posedness of PDE, spectral asymptotics, and the relationship between classical and weak solutions all require a rigorous treatment of unbounded operators.

### Multiplication operators

A simple but instructive class of unbounded operators consists of multiplication operators. On $L^2(\mathbb{R})$, define $(M_g f)(x) = g(x)f(x)$ for a measurable function $g: \mathbb{R} \to \mathbb{C}$. When $g$ is bounded, $M_g$ is a bounded operator. But when $g$ is unbounded — for instance $g(x) = x$ (the position operator) or $g(x) = x^2$ — the operator $M_g$ is unbounded and can only be defined on the domain $\mathcal{D}(M_g) = \{f \in L^2 : gf \in L^2\}$.

Multiplication operators serve as the "model case" for the general theory. The spectral theorem will eventually tell us that every self-adjoint operator is unitarily equivalent to a multiplication operator. Understanding their properties — closedness, spectra, domains — provides intuition for the abstract theory.

---

## Domains, Graphs, and Closed Operators

### The role of the domain

An **unbounded operator** on a Hilbert space $H$ is a linear map $T: \mathcal{D}(T) \to H$, where $\mathcal{D}(T) \subseteq H$ is a linear subspace called the **domain** of $T$. We always require $\mathcal{D}(T)$ to be *dense* in $H$ (otherwise the adjoint, defined below, does not make sense).

The critical point is that the *same formal expression* can define different operators depending on the domain. Consider $T = -id/dx$ on $L^2([0,1])$. We could choose:

- $\mathcal{D}_1 = C_c^\infty((0,1))$, smooth functions compactly supported in the open interval;
- $\mathcal{D}_2 = \{f \in H^1([0,1]) : f(0) = f(1) = 0\}$, the Sobolev space with Dirichlet boundary conditions;
- $\mathcal{D}_3 = \{f \in H^1([0,1]) : f(0) = f(1)\}$, periodic boundary conditions;
- $\mathcal{D}_4 = H^1([0,1])$, no boundary conditions at all.

These are four *different* operators. They have different spectral properties, different adjoints, and different physical interpretations. The lesson: **for unbounded operators, the domain is part of the definition**.

### The graph of an operator

The **graph** of $T$ is the subset of $H \times H$ defined by

$$
\Gamma(T) = \{(x, Tx) : x \in \mathcal{D}(T)\}.
$$

This is a linear subspace of $H \times H$. We say $T$ is **closed** if $\Gamma(T)$ is closed in $H \times H$ (with the product topology). Equivalently, $T$ is closed if whenever $(x_n)$ is a sequence in $\mathcal{D}(T)$ with $x_n \to x$ and $Tx_n \to y$, we have $x \in \mathcal{D}(T)$ and $Tx = y$.

Closedness is the correct replacement for continuity in the unbounded setting. A bounded operator on a Banach space is closed (its graph is closed because it is continuous), but an unbounded operator need not be. However, many important unbounded operators are closed, and those that are not can often be *closed* — extended to a closed operator.

**Definition.** An operator $T$ is **closable** if the closure $\overline{\Gamma(T)}$ in $H \times H$ is itself the graph of a (single-valued) operator. The resulting operator $\bar{T}$ is called the **closure** of $T$.

An equivalent condition: $T$ is closable if and only if whenever $x_n \in \mathcal{D}(T)$, $x_n \to 0$, and $Tx_n \to y$, we must have $y = 0$. (Otherwise the closure of the graph would contain both $(0, y)$ and $(0, 0)$ for some $y \ne 0$, making it not a function.)

### Examples

**Example 1.** The multiplication operator $(Mf)(x) = g(x)f(x)$ on $L^2(\mathbb{R})$, where $g: \mathbb{R} \to \mathbb{C}$ is measurable, with domain

$$
\mathcal{D}(M) = \{f \in L^2(\mathbb{R}) : gf \in L^2(\mathbb{R})\}.
$$

This operator is always closed. To see this: suppose $f_n \to f$ in $L^2$ and $gf_n \to h$ in $L^2$. By passing to a subsequence, $f_n(x) \to f(x)$ a.e., so $g(x)f_n(x) \to g(x)f(x)$ a.e. But $gf_n \to h$ in $L^2$ implies (after passing to a further subsequence) $g(x)f_n(x) \to h(x)$ a.e. Hence $h = gf$ a.e., so $f \in \mathcal{D}(M)$ and $Mf = h$.

**Example 2.** The operator $T = -id/dx$ on $L^2(\mathbb{R})$ with domain $\mathcal{D}(T) = H^1(\mathbb{R})$ (the Sobolev space of functions with one $L^2$ derivative). This is closed: convergence $f_n \to f$ in $L^2$ together with $f_n' \to g$ in $L^2$ implies (by the definition of weak derivatives) that $f' = g$ in the distributional sense, so $f \in H^1$ and $Tf = g$.

**Example 3.** The same differential expression $-id/dx$ on $\mathcal{D}(T) = C_c^\infty(\mathbb{R})$ is *not* closed (the domain is too small — sequences in $C_c^\infty$ can converge to $H^1$ functions that are not smooth). However, it is closable, and its closure is exactly the operator on $H^1(\mathbb{R})$ from Example 2.

### Core properties of closed operators

Several important properties follow directly from the definition of closedness.

**Proposition.** If $T$ is closed and $S$ is bounded with $\mathcal{D}(T) \subseteq \mathcal{D}(S)$, then $T + S$ is closed on $\mathcal{D}(T)$.

*Proof.* Suppose $x_n \in \mathcal{D}(T)$, $x_n \to x$, and $(T + S)x_n \to y$. Since $S$ is bounded, $Sx_n \to Sx$. Hence $Tx_n = (T + S)x_n - Sx_n \to y - Sx$. By closedness of $T$, $x \in \mathcal{D}(T)$ and $Tx = y - Sx$, so $(T + S)x = y$. $\square$

This is useful for perturbation theory: adding a bounded perturbation to a closed operator preserves closedness.

**Proposition.** If $T$ is closed and injective, then $T^{-1}$ (defined on $\text{Range}(T)$) is also closed.

*Proof.* The graph of $T^{-1}$ is $\{(Tx, x) : x \in \mathcal{D}(T)\}$, which is obtained from the graph of $T$ by swapping coordinates — a continuous operation that preserves closedness. $\square$

### The closed graph theorem revisited

Recall the closed graph theorem: if $T: X \to Y$ is a closed operator between Banach spaces and $\mathcal{D}(T) = X$ (the entire space), then $T$ is bounded. Equivalently: **an unbounded operator cannot be both closed and everywhere defined**. This explains why unbounded operators must have proper domains.

The contrapositive perspective is equally illuminating: if we know an operator is unbounded, the closed graph theorem tells us it cannot be defined on all of $X$ and still be closed. The domain restriction is not an artifact of our construction but a mathematical necessity.

---

## The Adjoint of an Unbounded Operator

For bounded operators $T \in B(H)$, the adjoint $T^*$ is the unique bounded operator satisfying $\langle Tx, y \rangle = \langle x, T^*y \rangle$ for all $x, y \in H$. For unbounded operators, the definition requires more care.

**Definition.** Let $T: \mathcal{D}(T) \to H$ be a densely defined operator. The **adjoint** $T^*$ is defined as follows. The domain $\mathcal{D}(T^*)$ consists of all $y \in H$ such that the map $x \mapsto \langle Tx, y \rangle$ is continuous on $\mathcal{D}(T)$ (with respect to the norm of $H$). Since $\mathcal{D}(T)$ is dense, this continuous linear functional extends uniquely to all of $H$, so by Riesz representation there exists a unique $z \in H$ with

$$
\langle Tx, y \rangle = \langle x, z \rangle \quad \text{for all } x \in \mathcal{D}(T).
$$

We set $T^*y = z$.

Key properties:
1. $T^*$ is always a closed operator (even if $T$ is not).
2. $\mathcal{D}(T^*)$ may or may not be dense.
3. If $T$ is closable, then $T^* $ is densely defined, and $\bar{T} = T^{**}$.
4. If $S \subset T$ (meaning $\mathcal{D}(S) \subset \mathcal{D}(T)$ and $Sx = Tx$ on $\mathcal{D}(S)$), then $T^* \subset S^*$. That is, *enlarging the domain of an operator shrinks the domain of its adjoint*.

Property 4 is crucial and counterintuitive. It means that choosing a larger domain for a differential operator gives a *smaller* domain for the adjoint. This interplay between operator and adjoint domains is the heart of boundary-value problems.

**Example.** Consider $T = -id/dx$ on $L^2([0,1])$ with different domains:

- If $\mathcal{D}(T) = C_c^\infty((0,1))$ (compactly supported smooth functions vanishing near the boundary), then integration by parts gives $\langle Tf, g \rangle = \langle f, -ig' \rangle$ for *all* $g \in H^1([0,1])$ with no boundary condition needed (since $f$ vanishes at the endpoints). So $\mathcal{D}(T^*) = H^1([0,1])$, the full Sobolev space with no boundary restriction.

- If $\mathcal{D}(T) = \{f \in H^1([0,1]) : f(0) = f(1) = 0\}$, integration by parts gives $\langle Tf, g \rangle = \langle f, -ig' \rangle + i[\overline{f}g]_0^1 = \langle f, -ig' \rangle$ (boundary terms vanish because $f = 0$ at the endpoints). So $T^*$ acts as $-ig'$ with $\mathcal{D}(T^*) = H^1([0,1])$, again with no boundary restriction.

- If $\mathcal{D}(T) = H^1([0,1])$ (the "maximal" domain), integration by parts gives boundary terms $i[\overline{f(1)}g(1) - \overline{f(0)}g(0)]$ that must vanish for *all* $f \in H^1$. This forces $g(0) = g(1) = 0$. So $\mathcal{D}(T^*) = \{g \in H^1([0,1]) : g(0) = g(1) = 0\}$ — the Dirichlet condition.

Observe the "duality" of boundary conditions: Dirichlet conditions on $T$ correspond to no boundary conditions on $T^*$, and vice versa.

### Proof that $T^*$ is always closed

We promised that $T^*$ is closed for any densely defined $T$. Here is the proof.

*Proof.* Suppose $y_n \in \mathcal{D}(T^*)$, $y_n \to y$, and $T^*y_n \to z$. We need to show $y \in \mathcal{D}(T^*)$ and $T^*y = z$. For any $x \in \mathcal{D}(T)$:

$$
\langle Tx, y \rangle = \lim_{n \to \infty} \langle Tx, y_n \rangle = \lim_{n \to \infty} \langle x, T^*y_n \rangle = \langle x, z \rangle.
$$

So the map $x \mapsto \langle Tx, y \rangle$ equals $x \mapsto \langle x, z \rangle$, which is certainly continuous. Hence $y \in \mathcal{D}(T^*)$ and $T^*y = z$. $\square$

This proof is surprisingly simple. The key observation is that closedness of $T^*$ follows from the continuity of the inner product — a purely Hilbert space phenomenon. This is one of many places where Hilbert space structure simplifies operator theory.

### The graph characterization of the adjoint

There is an elegant reformulation of the adjoint in terms of graphs. Define the unitary operator $J: H \times H \to H \times H$ by $J(x, y) = (-y, x)$. Then

$$
\Gamma(T^*) = [J(\Gamma(T))]^\perp,
$$

where the orthogonal complement is taken in $H \times H$. This identity shows at a glance that $\Gamma(T^*)$ is closed (being the orthogonal complement of a set) and connects the adjoint to the geometric structure of the product Hilbert space.

---

## Symmetric vs Self-Adjoint: The Crucial Distinction

### Definitions

**Definition.** A densely defined operator $T$ on $H$ is **symmetric** (or Hermitian) if

$$
\langle Tx, y \rangle = \langle x, Ty \rangle \quad \text{for all } x, y \in \mathcal{D}(T).
$$

Equivalently, $T \subset T^*$ (meaning $\mathcal{D}(T) \subset \mathcal{D}(T^*)$ and $T^*x = Tx$ for $x \in \mathcal{D}(T)$).

**Definition.** A densely defined operator $T$ is **self-adjoint** if $T = T^*$, meaning both $T \subset T^*$ (symmetry) and $\mathcal{D}(T^*) \subset \mathcal{D}(T)$ (equality of domains).

In the bounded case, the distinction evaporates: a symmetric bounded operator is automatically self-adjoint (both are defined on all of $H$, so domain equality is trivial). For unbounded operators, the distinction is *the* central issue.

### Why the distinction matters

The spectral theorem — the foundation for quantum mechanics and much of PDE theory — holds for **self-adjoint** operators, not merely symmetric ones. A symmetric operator that is not self-adjoint may have:

- Complex eigenvalues (its spectrum can include all of $\mathbb{C}$),
- No spectral decomposition,
- No functional calculus.

**Example (symmetric but not self-adjoint).** Consider $T = -id/dx$ on $L^2([0,1])$ with domain $\mathcal{D}(T) = \{f \in H^1([0,1]) : f(0) = f(1) = 0\}$. Integration by parts shows $T$ is symmetric. But we computed above that $T^* = -id/dx$ on $\mathcal{D}(T^*) = H^1([0,1])$, which is strictly larger than $\mathcal{D}(T)$. So $T \subsetneq T^*$: $T$ is symmetric but not self-adjoint.

What are the eigenvalues of $T^*$? Solving $-ig' = \lambda g$ with $g \in H^1([0,1])$ (no boundary condition) gives $g(x) = ce^{i\lambda x}$. This is in $L^2([0,1])$ for *every* $\lambda \in \mathbb{C}$. So $\sigma(T^*) = \mathbb{C}$ — the spectrum is the entire complex plane.

### Deficiency indices and von Neumann's theory

Given a closed symmetric operator $T$ on $H$, define the **deficiency subspaces**

$$
\mathcal{N}_+ = \ker(T^* - iI), \quad \mathcal{N}_- = \ker(T^* + iI),
$$

and the **deficiency indices**

$$
n_+ = \dim \mathcal{N}_+, \quad n_- = \dim \mathcal{N}_-.
$$

**Theorem (von Neumann).** Let $T$ be a closed symmetric operator with deficiency indices $(n_+, n_-)$.

1. $T$ is self-adjoint if and only if $n_+ = n_- = 0$.
2. $T$ has self-adjoint extensions if and only if $n_+ = n_-$.
3. When $n_+ = n_- = n < \infty$, the self-adjoint extensions of $T$ are in one-to-one correspondence with unitary maps $U: \mathcal{N}_+ \to \mathcal{N}_-$. Each such extension $T_U$ has domain

$$
\mathcal{D}(T_U) = \mathcal{D}(T) \oplus \{(\varphi + U\varphi) : \varphi \in \mathcal{N}_+\}
$$

and acts as $T_U(\psi + \varphi + U\varphi) = T\psi + i\varphi - iU\varphi$ for $\psi \in \mathcal{D}(T)$, $\varphi \in \mathcal{N}_+$.

**Application.** For $T = -id/dx$ on $\{f \in H^1([0,1]) : f(0) = f(1) = 0\}$, one computes $\mathcal{N}_+ = \text{span}\{e^{-x}\}$ and $\mathcal{N}_- = \text{span}\{e^{x}\}$, so $n_+ = n_- = 1$. The self-adjoint extensions are parameterized by unitary maps $U: \mathbb{C} \to \mathbb{C}$, i.e., by $e^{i\theta}$ for $\theta \in [0, 2\pi)$. Each value of $\theta$ gives a different boundary condition — the case $\theta = 0$ recovers periodic boundary conditions $f(0) = f(1)$. Different extensions have different spectra and different physical interpretations.

### Essentially self-adjoint operators

**Definition.** A symmetric operator $T$ is **essentially self-adjoint** if its closure $\bar{T}$ is self-adjoint. Equivalently, $T$ has a *unique* self-adjoint extension.

This is the physicist's paradise: if you can show the operator is essentially self-adjoint on your favorite convenient domain (say $C_c^\infty$), then there is exactly one self-adjoint extension and no ambiguity about which operator you mean. Several useful criteria exist:

**Theorem.** A symmetric operator $T$ is essentially self-adjoint if and only if $n_+ = n_- = 0$ for $\bar{T}$.

**Theorem (Nelson's criterion).** If $T$ is symmetric and has a dense set of analytic vectors (vectors $v$ such that $\sum \|T^n v\| t^n / n! < \infty$ for some $t > 0$), then $T$ is essentially self-adjoint.

**Example (essential self-adjointness of the Laplacian).** On $L^2(\mathbb{R}^n)$, the operator $T = -\Delta$ with domain $\mathcal{D}(T) = C_c^\infty(\mathbb{R}^n)$ is essentially self-adjoint. The key is that Gaussian functions $e^{-a|x|^2}$ (for $a > 0$) are analytic vectors for $-\Delta$, and their span is dense in $L^2$. Nelson's criterion then guarantees essential self-adjointness. The unique self-adjoint extension has domain $H^2(\mathbb{R}^n)$.

This example illustrates a common strategy in mathematical physics: work on the convenient domain $C_c^\infty$ (where integration by parts is easy and boundary terms vanish), prove essential self-adjointness, and conclude that the unique self-adjoint extension is the "correct" operator without needing to specify its domain explicitly.

---

## The Spectral Theorem for Unbounded Self-Adjoint Operators

The spectral theorem generalizes to unbounded self-adjoint operators, but the formulation requires projection-valued measures rather than a simple eigenvalue decomposition.

**Theorem (Spectral theorem, projection-valued measure form).** Let $T$ be a self-adjoint operator on a Hilbert space $H$. There exists a unique projection-valued measure $E$ on the Borel subsets of $\mathbb{R}$ such that

$$
T = \int_{-\infty}^{\infty} \lambda \, dE(\lambda),
$$

in the sense that

$$
\mathcal{D}(T) = \left\{x \in H : \int_{-\infty}^{\infty} \lambda^2 \, d\|E(\lambda)x\|^2 < \infty \right\}
$$

and for $x \in \mathcal{D}(T)$, $y \in H$,

$$
\langle Tx, y \rangle = \int_{-\infty}^{\infty} \lambda \, d\langle E(\lambda)x, y \rangle.
$$

The spectrum of $T$ is the support of $E$: the smallest closed set $S \subset \mathbb{R}$ with $E(\mathbb{R} \setminus S) = 0$.

**Consequences.**

1. The spectrum of a self-adjoint operator is *real* (confirming the physical requirement that observables have real measurement values).
2. For any bounded Borel function $f: \mathbb{R} \to \mathbb{C}$, we can define $f(T) = \int f(\lambda) \, dE(\lambda)$, giving a **functional calculus** for unbounded self-adjoint operators.
3. The spectral theorem provides a "multiplication operator form": every self-adjoint operator is unitarily equivalent to a multiplication operator on some $L^2$ space. Precisely, there exists a measure space $(X, \mu)$, a measurable function $g: X \to \mathbb{R}$, and a unitary $U: H \to L^2(X, \mu)$ such that $UTU^{-1}$ is multiplication by $g$.

**Connection to the bounded case.** For bounded self-adjoint operators, the integral runs over a compact interval $[\alpha, \beta] \subset \mathbb{R}$ and we recover the spectral theorem from earlier articles. The unbounded version is strictly more general — the integration domain is all of $\mathbb{R}$, and the domain restriction on $T$ encodes precisely the integrability condition on $\lambda^2$.

**Proof strategy.** The standard approach uses the **Cayley transform**. Define $V = (T - iI)(T + iI)^{-1}$. One shows:
1. $V$ is a well-defined isometry from $H$ onto $H$ (using self-adjointness to show $T \pm iI$ are bijective).
2. $V$ is unitary, and $I - V$ has dense range.
3. Apply the spectral theorem for unitary operators (which follows from the bounded case) to $V$.
4. Pull back to $T$ using the inverse Cayley transform $T = i(I + V)(I - V)^{-1}$.

This beautifully reduces the unbounded case to the bounded one.

### The spectral theorem for multiplication operators

As a sanity check and concrete illustration, let us verify the spectral theorem for a multiplication operator $M_g$ on $L^2(X, \mu)$ with real-valued $g$. The projection-valued measure is simply

$$
E(B)f = \chi_{g^{-1}(B)} \cdot f \quad \text{for Borel sets } B \subset \mathbb{R},
$$

where $\chi_{g^{-1}(B)}$ is the characteristic function of the preimage $g^{-1}(B)$. Then

$$
\int \lambda \, dE(\lambda)f = \int \lambda \, d(\chi_{g^{-1}((-\infty, \lambda])} \cdot f) = g \cdot f = M_g f,
$$

which is exactly the original operator. The domain condition $\int \lambda^2 \, d\|E(\lambda)f\|^2 < \infty$ becomes $\int |g(x)|^2|f(x)|^2 \, d\mu(x) < \infty$, which is precisely $gf \in L^2$ — the natural domain of $M_g$.

The spectrum of $M_g$ is the essential range of $g$ (the set of $\lambda \in \mathbb{R}$ such that $\mu(g^{-1}((\lambda - \epsilon, \lambda + \epsilon))) > 0$ for all $\epsilon > 0$). Eigenvalues of $M_g$ correspond to level sets $\{x : g(x) = \lambda\}$ with positive measure.

---

## The Laplacian as a Model: Friedrichs Extension

The Friedrichs extension provides a canonical way to construct a self-adjoint operator from a symmetric operator that is bounded below.

**Setup.** Let $\Omega \subset \mathbb{R}^n$ be a bounded open set with smooth boundary. Consider the Laplacian $T_0 = -\Delta$ on $\mathcal{D}(T_0) = C_c^\infty(\Omega)$. Integration by parts gives

$$
\langle -\Delta u, u \rangle = \int_\Omega |\nabla u|^2 \, dx \ge 0 \quad \text{for all } u \in C_c^\infty(\Omega),
$$

so $T_0$ is symmetric and non-negative ($\langle T_0 u, u \rangle \ge 0$). But $T_0$ is *not* self-adjoint — its domain is too small.

**Friedrichs extension procedure.**

1. Define the quadratic form $q(u) = \langle T_0 u, u \rangle + \|u\|^2 = \|\nabla u\|^2 + \|u\|^2$ on $\mathcal{D}(T_0) = C_c^\infty(\Omega)$.

2. Complete $C_c^\infty(\Omega)$ with respect to the norm $\|u\|_q = q(u)^{1/2}$. This completion is precisely $H_0^1(\Omega)$, the Sobolev space of functions with one $L^2$ derivative and zero boundary values.

3. For each $f \in L^2(\Omega)$, the functional $v \mapsto \langle f, v \rangle$ is continuous on $H_0^1(\Omega)$ (by Cauchy-Schwarz and the fact that $\|v\|_{L^2} \le \|v\|_{H_0^1}$). By the Riesz representation theorem applied to the Hilbert space $(H_0^1, q)$, there exists a unique $u \in H_0^1(\Omega)$ with

$$
q(u, v) = \langle f, v \rangle \quad \text{for all } v \in H_0^1(\Omega),
$$

where $q(u, v) = \int_\Omega \nabla u \cdot \nabla \bar{v} \, dx + \int_\Omega u\bar{v} \, dx$.

4. Define $T_F u = f$ (equivalently, $T_F u = -\Delta u$ in the weak sense) with $\mathcal{D}(T_F) = \{u \in H_0^1(\Omega) : -\Delta u \in L^2(\Omega) \text{ in the distributional sense}\}$.

**Theorem (Friedrichs).** The operator $T_F$ is self-adjoint, non-negative, and extends $T_0$. It is called the **Friedrichs extension** of $T_0$.

For the Laplacian, the Friedrichs extension corresponds precisely to Dirichlet boundary conditions: $\mathcal{D}(T_F) = H^2(\Omega) \cap H_0^1(\Omega)$ (by elliptic regularity). Other self-adjoint extensions of $T_0$ correspond to other boundary conditions (Neumann, Robin, etc.), but the Friedrichs extension is distinguished by having the smallest domain among all non-negative self-adjoint extensions.

**Significance.** The Friedrichs extension is the standard tool for turning "formal" differential operators into honest self-adjoint operators. Whenever you see "$-\Delta$ with Dirichlet boundary conditions" in a PDE textbook, the rigorous meaning is the Friedrichs extension.

### Comparison with other self-adjoint extensions

The Friedrichs extension is not the only self-adjoint extension of $T_0 = -\Delta|_{C_c^\infty(\Omega)}$, but it enjoys a special minimality property. Among all non-negative self-adjoint extensions $T$ of $T_0$, the Friedrichs extension $T_F$ satisfies $\mathcal{D}(T_F^{1/2}) \subseteq \mathcal{D}(T^{1/2})$ (it has the smallest form domain). In terms of the ordering of self-adjoint operators, $T_F$ is the *largest* non-negative self-adjoint extension: $T \le T_F$ for all other non-negative self-adjoint extensions $T$.

The **Krein extension** $T_K$ is the smallest non-negative self-adjoint extension ($T_K \le T$ for all others), corresponding to Neumann-type boundary conditions. All other non-negative self-adjoint extensions lie between $T_K$ and $T_F$ in the operator ordering.

For the Laplacian on a bounded domain:
- Friedrichs extension $\leftrightarrow$ Dirichlet boundary conditions ($u|_{\partial\Omega} = 0$).
- Krein extension $\leftrightarrow$ generalized Neumann conditions.
- Other extensions $\leftrightarrow$ Robin boundary conditions ($\partial u/\partial n + \alpha u|_{\partial\Omega} = 0$), or more exotic conditions.

This connection between abstract operator extension theory and concrete boundary conditions is one of the most satisfying aspects of the subject.

---

## What's Next

We have seen that unbounded operators demand careful attention to domains, that the distinction between symmetric and self-adjoint is crucial for the spectral theorem, and that tools like the Friedrichs extension provide canonical self-adjoint realizations of differential operators.

In the next article, we turn to **semigroups of operators**, which provide the abstract framework for evolution equations: how does a system governed by an unbounded operator (like the heat equation or the Schrodinger equation) evolve in time? The Hille-Yosida theorem gives a complete characterization of which operators generate well-posed dynamics.

---

*This is Part 9 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 8 — Spectral Theory](/en/functional-analysis/08-spectral-theory/)*

*Next: [Part 10 — Semigroups of Operators](/en/functional-analysis/10-semigroups/)*
