---
title: "Functional Analysis (3): Hilbert Spaces — Geometry in Infinite Dimensions"
date: 2021-10-05 09:00:00
tags:
  - functional-analysis
  - hilbert-spaces
  - inner-product
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "Inner products give infinite-dimensional spaces geometric structure — orthogonality, projections, and the Riesz representation theorem make Hilbert spaces the analyst's paradise."
disableNunjucks: true
series_order: 3
series_total: 12
translationKey: "functional-analysis-3"
---

Banach spaces give us completeness and a norm, but they lack the one ingredient that makes Euclidean geometry possible: *angles*. Without a notion of perpendicularity, there is no way to project a vector onto a subspace, no way to decompose a signal into orthogonal components, no Fourier series. Hilbert spaces restore all of this by equipping a complete vector space with an **inner product**, and the consequences are remarkably powerful.

This article develops the theory of Hilbert spaces from the inner product axioms through to the Riesz Representation Theorem, which characterizes every continuous linear functional as an inner product — a result with no analogue in general Banach spaces.

---

## Inner Products and the Parallelogram Law

### The inner product axioms

Let $H$ be a vector space over $\mathbb{F}$ (where $\mathbb{F} = \mathbb{R}$ or $\mathbb{C}$). An **inner product** on $H$ is a map $\langle \cdot, \cdot \rangle : H \times H \to \mathbb{F}$ satisfying:

![Orthogonal projection onto a closed subspace in Hilbert space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_fig2_projection.png)


1. **Conjugate symmetry:** $\langle x, y \rangle = \overline{\langle y, x \rangle}$ for all $x, y \in H$.
2. **Linearity in the first argument:** $\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle$.
3. **Positive definiteness:** $\langle x, x \rangle \geq 0$, with equality if and only if $x = 0$.

The convention here is "linear in the first argument" (the physics convention reverses this). In the real case, conjugate symmetry reduces to ordinary symmetry: $\langle x, y \rangle = \langle y, x \rangle$.

From these axioms, we obtain a norm via $\|x\| = \sqrt{\langle x, x \rangle}$. That this is indeed a norm (in particular, that the triangle inequality holds) follows from the Cauchy-Schwarz inequality, which we prove below. A **Hilbert space** is an inner product space that is **complete** with respect to this induced norm — that is, every Cauchy sequence converges.

**Example 1.** The space $\ell^2$ of square-summable sequences with inner product

$$\langle x, y \rangle = \sum_{n=1}^{\infty} x_n \overline{y_n}$$

is the prototypical separable Hilbert space. Completeness follows from the Riesz-Fischer theorem.

**Example 2.** The Lebesgue space $L^2[a,b]$ with inner product

$$\langle f, g \rangle = \int_a^b f(t)\overline{g(t)}\, dt$$

is a Hilbert space. This is the natural setting for Fourier analysis — and, more broadly, for any problem where one wants to measure the "energy" of a function.

**Example 2b.** The **Sobolev space** $W^{1,2}[0,1]$ consists of functions $f \in L^2[0,1]$ whose weak derivative $f'$ also belongs to $L^2[0,1]$. With the inner product

$$\langle f, g \rangle_{W^{1,2}} = \int_0^1 f(t)\overline{g(t)}\, dt + \int_0^1 f'(t)\overline{g'(t)}\, dt,$$

this is a Hilbert space. It arises naturally in the variational formulation of boundary value problems: the weak form of $-u'' = f$ on $[0,1]$ lives in $W^{1,2}_0[0,1]$ (functions vanishing at the boundary).

### The Cauchy-Schwarz inequality

The single most important inequality in all of analysis:

$$|\langle x, y \rangle| \leq \|x\| \cdot \|y\|$$

with equality if and only if $x$ and $y$ are linearly dependent. The proof is a short computation: for any $\lambda \in \mathbb{F}$,

$$0 \leq \|x - \lambda y\|^2 = \|x\|^2 - 2\operatorname{Re}(\lambda \langle y, x \rangle) + |\lambda|^2 \|y\|^2.$$

Choosing $\lambda = \langle x, y \rangle / \|y\|^2$ (when $y \neq 0$) and simplifying yields the result. For the case $y = 0$, both sides are zero and the inequality is trivial.

The Cauchy-Schwarz inequality has immediate consequences. First, it shows that $\|x + y\| \leq \|x\| + \|y\|$ (the triangle inequality for the induced norm), confirming that $\|x\| = \sqrt{\langle x, x \rangle}$ is indeed a norm. Second, it guarantees that the inner product is **continuous** as a function of both arguments: if $x_n \to x$ and $y_n \to y$ in norm, then $\langle x_n, y_n \rangle \to \langle x, y \rangle$. This continuity is essential for working with limits and series in Hilbert spaces.

### The polarization identity

The inner product can be recovered from the norm alone. In the real case:

$$\langle x, y \rangle = \frac{1}{4}\left(\|x + y\|^2 - \|x - y\|^2\right).$$

In the complex case:

$$\langle x, y \rangle = \frac{1}{4}\sum_{k=0}^{3} i^k \|x + i^k y\|^2.$$

These are the **polarization identities**. They show that angles are encoded in distances — if you know all pairwise distances, you can reconstruct all inner products. The polarization identity is also the key tool in the proof of the von Neumann-Jordan theorem.

### The parallelogram law

Every inner product norm satisfies the **parallelogram law**:

$$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2.$$

This identity has a geometric reading: in a parallelogram, the sum of the squares of the diagonals equals the sum of the squares of all four sides.

The remarkable converse — the **von Neumann-Jordan theorem** — states that if a Banach space satisfies the parallelogram law, then there exists a unique inner product that generates the norm (recovered via the polarization identity). This is precisely why **Hilbert spaces are more than Banach spaces**: the parallelogram law is an extra structural constraint that unlocks geometry.

**Why $L^p$ for $p \neq 2$ is not a Hilbert space.** Consider $f = \mathbf{1}_{[0,1/2]}$ and $g = \mathbf{1}_{(1/2,1]}$ in $L^p[0,1]$. Then $\|f\|_p = \|g\|_p = 2^{-1/p}$, $\|f+g\|_p = 1$, $\|f-g\|_p = 1$. The parallelogram law requires $1 + 1 = 2(2^{-2/p}) + 2(2^{-2/p}) = 4 \cdot 2^{-2/p}$, i.e., $2^{2/p} = 2$, which forces $p = 2$.

---

## Orthogonality and Orthogonal Complements

### Orthogonality

Two elements $x, y \in H$ are **orthogonal**, written $x \perp y$, if $\langle x, y \rangle = 0$. A set $S \subseteq H$ is an **orthogonal set** if every pair of distinct elements is orthogonal; it is **orthonormal** if additionally $\|e\| = 1$ for each $e \in S$.

The **orthogonal complement** of a subset $S \subseteq H$ is

$$S^{\perp} = \{x \in H : \langle x, s \rangle = 0 \text{ for all } s \in S\}.$$

Key properties: $S^{\perp}$ is always a closed subspace (even if $S$ is not — the intersection of closed sets obtained from continuity of the inner product), $(S^{\perp})^{\perp} \supseteq \overline{\operatorname{span}}(S)$, and in a Hilbert space equality holds: $(S^{\perp})^{\perp} = \overline{\operatorname{span}}(S)$.

The **Pythagorean theorem** extends to Hilbert spaces: if $x_1, \ldots, x_n$ are pairwise orthogonal, then

$$\left\|\sum_{k=1}^n x_k\right\|^2 = \sum_{k=1}^n \|x_k\|^2.$$

This generalizes to infinite orthogonal sums when the series converges. The Pythagorean theorem is the engine behind Parseval's identity and the energy decomposition that makes Hilbert spaces so useful in signal processing and physics.

### The Projection Theorem

This is the geometric heart of Hilbert space theory.

**Theorem (Orthogonal Projection).** Let $M$ be a **closed convex subset** of a Hilbert space $H$, and let $x \in H$. Then there exists a unique $m_0 \in M$ such that

$$\|x - m_0\| = \inf_{m \in M} \|x - m\| =: d(x, M).$$

When $M$ is a closed **subspace**, this nearest point $m_0$ is characterized by the condition $x - m_0 \perp M$, and we get the orthogonal decomposition $H = M \oplus M^{\perp}$.

**Proof sketch.** Let $d = \inf_{m \in M} \|x - m\|$ and choose a minimizing sequence $(m_n)$ with $\|x - m_n\| \to d$.

*Existence.* Apply the parallelogram law to $x - m_n$ and $x - m_k$:

$$\|( x - m_n) + (x - m_k)\|^2 + \|(x - m_n) - (x - m_k)\|^2 = 2\|x - m_n\|^2 + 2\|x - m_k\|^2.$$

The left side contains $\|2x - m_n - m_k\|^2 = 4\|x - \frac{m_n + m_k}{2}\|^2 \geq 4d^2$ (since $\frac{m_n + m_k}{2} \in M$ by convexity) and $\|m_n - m_k\|^2$. Thus:

$$\|m_n - m_k\|^2 \leq 2\|x - m_n\|^2 + 2\|x - m_k\|^2 - 4d^2 \to 0.$$

So $(m_n)$ is Cauchy. By completeness of $H$ and closedness of $M$, the limit $m_0 = \lim m_n \in M$, and $\|x - m_0\| = d$.

*Uniqueness.* If $m_0$ and $m_0'$ both achieve the minimum, the same parallelogram argument gives $\|m_0 - m_0'\|^2 \leq 2d^2 + 2d^2 - 4d^2 = 0$.

*Orthogonality (when $M$ is a subspace).* For any $m \in M$ and $\lambda \in \mathbb{F}$, $m_0 + \lambda m \in M$, so

$$d^2 \leq \|x - m_0 - \lambda m\|^2 = d^2 - 2\operatorname{Re}(\lambda \langle m, x - m_0 \rangle) + |\lambda|^2\|m\|^2.$$

Choosing $\lambda = t\langle x - m_0, m \rangle / \|m\|^2$ for small $t > 0$ forces $\langle x - m_0, m \rangle = 0$. $\blacksquare$

**Remark on the role of the parallelogram law.** The crucial step in the proof — showing that the minimizing sequence is Cauchy — uses the parallelogram law. This is why the projection theorem, in its full generality, requires a Hilbert space (or at least an inner product). In a general Banach space, the nearest point to a closed convex set may not exist (in non-reflexive spaces) or may not be unique. For example, in $\ell^\infty$, the nearest point from $x = (2, 0, 0, \ldots)$ to the subspace $M = \{y : y_1 = 0\}$ is any point of the form $(0, y_2, 0, \ldots)$ with $|y_2| \leq 2$. The failure of uniqueness stems precisely from the absence of the parallelogram law.

**The orthogonal projection operator.** When $M$ is a closed subspace, the map $P: H \to H$ sending $x$ to its nearest point $m_0 \in M$ is called the **orthogonal projection onto $M$**. It satisfies:

- $P$ is linear and bounded with $\|P\| = 1$ (unless $M = \{0\}$).
- $P^2 = P$ (idempotent: projecting twice gives the same result).
- $P^* = P$ (self-adjoint: $\langle Px, y \rangle = \langle x, Py \rangle$).
- $\operatorname{Range}(P) = M$ and $\ker(P) = M^{\perp}$.

Conversely, any bounded linear operator satisfying $P^2 = P$ and $P^* = P$ is the orthogonal projection onto its range. Orthogonal projections are the building blocks of the spectral theorem for self-adjoint operators.

**Example 3 (Best approximation in $L^2$).** Let $M = \operatorname{span}\{1, t, t^2\} \subset L^2[0,1]$ and $f(t) = e^t$. The best quadratic approximation to $e^t$ in the $L^2$ norm is the orthogonal projection of $f$ onto $M$: the unique polynomial $p(t) = a_0 + a_1 t + a_2 t^2$ satisfying $\langle e^t - p(t), t^k \rangle = 0$ for $k = 0, 1, 2$. Expanding these three conditions gives a $3 \times 3$ linear system (the **normal equations**):

$$\begin{pmatrix} \langle 1, 1 \rangle & \langle t, 1 \rangle & \langle t^2, 1 \rangle \\ \langle 1, t \rangle & \langle t, t \rangle & \langle t^2, t \rangle \\ \langle 1, t^2 \rangle & \langle t, t^2 \rangle & \langle t^2, t^2 \rangle \end{pmatrix} \begin{pmatrix} a_0 \\ a_1 \\ a_2 \end{pmatrix} = \begin{pmatrix} \langle e^t, 1 \rangle \\ \langle e^t, t \rangle \\ \langle e^t, t^2 \rangle \end{pmatrix},$$

where $\langle t^j, t^k \rangle = \int_0^1 t^{j+k}\, dt = \frac{1}{j+k+1}$ (this is the **Hilbert matrix**) and $\langle e^t, t^k \rangle = \int_0^1 t^k e^t\, dt$ can be computed by integration by parts. The solution gives the least-squares polynomial fit, and the $L^2$ error is $\|f - p\|^2 = \|f\|^2 - \|p\|^2$ by the Pythagorean theorem. This approach is the mathematical foundation of linear regression in statistics.

---

## Orthonormal Systems and Bessel's Inequality

### Gram-Schmidt and orthonormal sequences

Given any countable linearly independent set $\{v_1, v_2, \ldots\}$ in a Hilbert space, the **Gram-Schmidt process** produces an orthonormal sequence $\{e_1, e_2, \ldots\}$ with $\operatorname{span}\{e_1, \ldots, e_n\} = \operatorname{span}\{v_1, \ldots, v_n\}$ for every $n$:

$$\tilde{e}_n = v_n - \sum_{k=1}^{n-1} \langle v_n, e_k \rangle e_k, \qquad e_n = \frac{\tilde{e}_n}{\|\tilde{e}_n\|}.$$

The process works because at each step, $\tilde{e}_n$ is the component of $v_n$ orthogonal to $\operatorname{span}\{e_1, \ldots, e_{n-1}\}$ — which is exactly the projection theorem in action. As long as $v_n$ is not in the span of the previous vectors (i.e., the set is linearly independent), we have $\tilde{e}_n \neq 0$ and the normalization is well-defined.

**Stability warning.** In finite-dimensional numerical computation, the classical Gram-Schmidt process is notoriously unstable: rounding errors accumulate and the output vectors lose orthogonality. The **modified Gram-Schmidt** algorithm (which re-orthogonalizes against each $e_k$ sequentially rather than all at once) is numerically superior. For infinite-dimensional theoretical purposes, however, the classical version is perfectly rigorous.

### Fourier coefficients and Bessel's inequality

Let $\{e_n\}_{n=1}^{\infty}$ be an orthonormal sequence in $H$ (not necessarily a basis). For any $x \in H$, the **Fourier coefficients** are $\hat{x}_n = \langle x, e_n \rangle$, and **Bessel's inequality** states:

$$\sum_{n=1}^{\infty} |\langle x, e_n \rangle|^2 \leq \|x\|^2.$$

**Proof.** For any finite $N$, let $S_N = \sum_{n=1}^{N} \langle x, e_n \rangle e_n$. Then:

$$0 \leq \|x - S_N\|^2 = \|x\|^2 - 2\operatorname{Re}\sum_{n=1}^{N} |\langle x, e_n \rangle|^2 + \sum_{n=1}^{N} |\langle x, e_n \rangle|^2 = \|x\|^2 - \sum_{n=1}^{N} |\langle x, e_n \rangle|^2.$$

Since this holds for all $N$ and the partial sums are non-decreasing, the series converges and the inequality follows. $\blacksquare$

The proof reveals something more: the partial sums $S_N$ form the orthogonal projection of $x$ onto $\operatorname{span}\{e_1, \ldots, e_N\}$, and $\|x - S_N\|^2 = \|x\|^2 - \sum_{n=1}^{N}|\langle x, e_n \rangle|^2$ measures the "unexplained energy" after projecting onto the first $N$ directions. As $N$ grows, this residual decreases, and whether it vanishes depends on whether the orthonormal system is complete.

Bessel's inequality tells us that the Fourier coefficients of any vector are square-summable, regardless of whether the orthonormal system is "large enough" to capture the entire vector. In particular, for any orthonormal system, at most countably many Fourier coefficients can be non-zero (since a convergent series of positive terms has only countably many non-zero terms).

**Example 4 (Fourier coefficients in $L^2[-\pi, \pi]$).** The system $\{e_n\}_{n \in \mathbb{Z}}$ with $e_n(t) = \frac{1}{\sqrt{2\pi}} e^{int}$ is orthonormal in $L^2[-\pi, \pi]$. For $f(t) = t$, the Fourier coefficients are:

$$\hat{f}_n = \frac{1}{\sqrt{2\pi}} \int_{-\pi}^{\pi} t\, e^{-int}\, dt.$$

For $n \neq 0$, integration by parts gives $\hat{f}_n = \frac{(-1)^{n+1}}{n} \cdot \sqrt{2\pi}\, i$, and $\hat{f}_0 = 0$ by symmetry. Bessel's inequality then gives:

$$\sum_{n \neq 0} \frac{2\pi}{n^2} \leq \int_{-\pi}^{\pi} t^2\, dt = \frac{2\pi^3}{3},$$

which simplifies to $\sum_{n=1}^{\infty} \frac{1}{n^2} \leq \frac{\pi^2}{3}$. We will sharpen this to equality momentarily.

---

## Orthonormal Bases and Parseval's Identity

### Orthonormal bases in Hilbert spaces

An orthonormal set $\{e_\alpha\}_{\alpha \in A}$ is an **orthonormal basis** (also called a **complete orthonormal system**) if any of the following equivalent conditions holds:

1. $\overline{\operatorname{span}}\{e_\alpha : \alpha \in A\} = H$ (the closed linear span is all of $H$).
2. $\langle x, e_\alpha \rangle = 0$ for all $\alpha \in A$ implies $x = 0$.
3. Every $x \in H$ satisfies the **Parseval identity** (in the separable case, $A = \mathbb{N}$):

$$\|x\|^2 = \sum_{n=1}^{\infty} |\langle x, e_n \rangle|^2.$$

The equivalence of these conditions is a fundamental structural result. Note that (3) upgrades Bessel's inequality to equality precisely when the orthonormal system is complete. The equivalence is proved as follows: $(1) \Rightarrow (2)$: if $\langle x, e_\alpha \rangle = 0$ for all $\alpha$, then $x \perp \overline{\operatorname{span}}\{e_\alpha\} = H$, so $x \perp x$, giving $x = 0$. $(2) \Rightarrow (3)$: the partial sums $S_N = \sum_{n=1}^N \langle x, e_n \rangle e_n$ converge (by Bessel) to some $y \in H$, and $\langle x - y, e_n \rangle = 0$ for all $n$, so $x = y$ by (2), and Parseval follows from the Pythagorean theorem. $(3) \Rightarrow (1)$: Parseval implies every $x$ is the norm limit of its partial sums, which lie in $\operatorname{span}\{e_n\}$.

**Existence.** Every Hilbert space has an orthonormal basis. The proof uses Zorn's lemma: consider the collection of all orthonormal sets, partially ordered by inclusion. Every chain has an upper bound (its union), so there exists a maximal orthonormal set. Maximality is equivalent to completeness.

### The separable case

A Hilbert space is **separable** if it has a countable dense subset. Equivalently, it admits a countable orthonormal basis. This is the case that arises most often in applications.

**Theorem (Fourier Expansion).** If $\{e_n\}_{n=1}^{\infty}$ is an orthonormal basis for a separable Hilbert space $H$, then every $x \in H$ has the convergent expansion:

$$x = \sum_{n=1}^{\infty} \langle x, e_n \rangle\, e_n,$$

where convergence is in the norm of $H$. The Fourier coefficients $\langle x, e_n \rangle$ are the unique scalars achieving this representation.

### Fourier series as a Hilbert space phenomenon

The classical Fourier series is simply the Fourier expansion in $L^2[-\pi, \pi]$ with respect to the orthonormal basis $\{e_n(t) = \frac{1}{\sqrt{2\pi}} e^{int}\}_{n \in \mathbb{Z}}$. The completeness of this system — equivalently, Parseval's identity — is equivalent to the statement that the trigonometric polynomials are dense in $L^2[-\pi, \pi]$.

Proving that this particular orthonormal system is complete is a non-trivial result. One approach uses the Stone-Weierstrass theorem: trigonometric polynomials form a self-adjoint subalgebra of $C[-\pi, \pi]$ that separates points, so they are uniformly dense in $C[-\pi, \pi]$. Since $C[-\pi, \pi]$ is dense in $L^2[-\pi, \pi]$, the trigonometric system is complete. An alternative proof uses the Fejer kernel and Cesaro summability.

This perspective reveals that Fourier analysis is not a collection of ad hoc tricks, but rather the theory of orthogonal expansions in the Hilbert space $L^2$. Convergence of Fourier series in the $L^2$ norm is *automatic* once completeness is established — no Dirichlet kernel gymnastics required. (Pointwise convergence, by contrast, is a much harder question that requires entirely different techniques.)

**Example 5 (Parseval and $\zeta(2)$).** Returning to $f(t) = t$ on $[-\pi, \pi]$, we computed the Fourier coefficients above. Since the trigonometric system is an orthonormal basis, Parseval's identity gives equality:

$$\frac{2\pi^3}{3} = \int_{-\pi}^{\pi} t^2\, dt = \sum_{n \in \mathbb{Z}} |\hat{f}_n|^2 = \sum_{n \neq 0} \frac{2\pi}{n^2} = 4\pi \sum_{n=1}^{\infty} \frac{1}{n^2}.$$

Therefore $\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$, which is Euler's solution to the Basel problem. The computation was a three-line consequence of Hilbert space theory.

---

## The Riesz Representation Theorem

This theorem is arguably the most important single result in Hilbert space theory.

**Theorem (Riesz-Frechet).** Let $H$ be a Hilbert space and $\varphi: H \to \mathbb{F}$ a continuous linear functional (i.e., $\varphi \in H^*$). Then there exists a **unique** $y \in H$ such that

$$\varphi(x) = \langle x, y \rangle \quad \text{for all } x \in H,$$

and $\|\varphi\|_{H^*} = \|y\|_H$.

In other words, $H^* \cong H$ via the conjugate-linear isometric isomorphism $y \mapsto \langle \cdot, y \rangle$. Every Hilbert space is **self-dual**.

**Proof.** If $\varphi = 0$, take $y = 0$. Otherwise, $M = \ker \varphi$ is a closed subspace with $M \neq H$.

*Step 1: Find a vector orthogonal to the kernel.* Since $M$ is proper and closed, $M^{\perp} \neq \{0\}$ (by the projection theorem: take any $x_0 \notin M$ and let $z = x_0 - Px_0$ where $P$ is the orthogonal projection onto $M$; then $z \neq 0$ and $z \in M^{\perp}$). Moreover, $\varphi(z) \neq 0$ because $z \notin M = \ker \varphi$.

*Step 2: Construct the representing element.* For any $x \in H$, the vector

$$x - \frac{\varphi(x)}{\varphi(z)} z$$

lies in $\ker \varphi = M$ (verify: $\varphi$ applied to it gives $\varphi(x) - \varphi(x) = 0$). Therefore it is orthogonal to $z$:

$$\left\langle x - \frac{\varphi(x)}{\varphi(z)} z,\, z \right\rangle = 0.$$

Expanding:

$$\langle x, z \rangle = \frac{\varphi(x)}{\varphi(z)} \|z\|^2,$$

so

$$\varphi(x) = \frac{\overline{\varphi(z)}}{\|z\|^2} \langle x, z \rangle = \left\langle x,\, \frac{\varphi(z)}{\|z\|^2} z \right\rangle.$$

Setting $y = \frac{\varphi(z)}{\|z\|^2} z$ gives $\varphi(x) = \langle x, y \rangle$ for all $x$.

*Step 3: Uniqueness.* If $\langle x, y \rangle = \langle x, y' \rangle$ for all $x$, then $\langle x, y - y' \rangle = 0$ for all $x$; taking $x = y - y'$ gives $\|y - y'\| = 0$.

*Step 4: Isometry.* By Cauchy-Schwarz, $|\varphi(x)| = |\langle x, y \rangle| \leq \|x\|\|y\|$, so $\|\varphi\| \leq \|y\|$. Taking $x = y$ gives $\varphi(y) = \|y\|^2$, so $\|\varphi\| \geq \|y\|$. $\blacksquare$

### Significance

The Riesz Representation Theorem has far-reaching consequences:

1. **Self-duality.** Every Hilbert space is reflexive (in fact, much more: the dual space is isometrically isomorphic to the original, not just the bidual).
2. **Adjoint operators.** For a bounded linear operator $T: H \to H$, the map $x \mapsto \langle Tx, y \rangle$ is a continuous linear functional in $x$ for each fixed $y$. By the Riesz theorem, there exists a unique $T^*y \in H$ with $\langle Tx, y \rangle = \langle x, T^*y \rangle$. This defines the **Hilbert space adjoint** $T^*$, the foundation of spectral theory.
3. **Weak convergence.** A sequence $(x_n)$ converges weakly to $x$ if $\langle x_n, y \rangle \to \langle x, y \rangle$ for all $y \in H$. The Riesz theorem ensures this is the same as convergence against all functionals.
4. **Quantum mechanics.** States in quantum mechanics are unit vectors in a Hilbert space, and observables correspond to self-adjoint operators. The Riesz theorem underpins the bra-ket formalism: $\langle \psi |$ is the functional corresponding to $|\psi\rangle$ via the isomorphism $H \cong H^*$.

### An application: the Lax-Milgram theorem

A powerful consequence of the Riesz theorem that is central to the theory of partial differential equations:

**Theorem (Lax-Milgram).** Let $H$ be a Hilbert space and $a: H \times H \to \mathbb{F}$ a **continuous coercive bilinear form** (i.e., $|a(x,y)| \leq C\|x\|\|y\|$ and $a(x,x) \geq \alpha \|x\|^2$ for some $\alpha > 0$). Then for every $\varphi \in H^*$, there exists a unique $u \in H$ such that $a(u, v) = \varphi(v)$ for all $v \in H$.

The proof uses the Riesz theorem twice: first to represent $\varphi$ as an inner product, then to show that $a(u, \cdot)$ for fixed $u$ is also a continuous linear functional. The map $u \mapsto$ "the Riesz representer of $a(u, \cdot)$" defines a bounded linear operator, and coercivity implies it is invertible.

**Example 7 (Weak formulation of Poisson's equation).** Consider $-\Delta u = f$ on a bounded domain $\Omega \subset \mathbb{R}^n$ with $u = 0$ on $\partial\Omega$. The weak formulation seeks $u \in H^1_0(\Omega)$ such that

$$a(u, v) = \int_\Omega \nabla u \cdot \nabla v\, dx = \int_\Omega fv\, dx = \varphi(v) \quad \forall v \in H^1_0(\Omega).$$

The bilinear form $a$ is continuous (by Cauchy-Schwarz) and coercive (by the Poincare inequality). Lax-Milgram immediately gives existence and uniqueness of the weak solution — no explicit construction needed.

---

## $\ell^2$ as the Universal Separable Hilbert Space

We have seen that every element of a separable Hilbert space can be expanded in an orthonormal basis as $x = \sum_{n=1}^{\infty} c_n e_n$ where $c_n = \langle x, e_n \rangle$ and $\sum |c_n|^2 = \|x\|^2 < \infty$. This means the map

$$U: H \to \ell^2, \qquad x \mapsto (\langle x, e_1 \rangle, \langle x, e_2 \rangle, \ldots)$$

is a **unitary isomorphism** (a bijective linear map preserving the inner product).

**Theorem.** All separable infinite-dimensional Hilbert spaces are unitarily isomorphic to $\ell^2$.

This is a striking structural rigidity: up to isomorphism, there is only **one** separable infinite-dimensional Hilbert space. The spaces $L^2[0,1]$, $L^2(\mathbb{R})$, the Hardy space $H^2(\mathbb{D})$, and the Sobolev space $W^{1,2}$ are all "the same" as abstract Hilbert spaces — they differ only in which concrete functions their elements represent and which orthonormal basis is natural for the application at hand.

This universality is both a blessing and a limitation. It means that any abstract theorem about $\ell^2$ automatically applies to $L^2[0,1]$, $H^2(\mathbb{D})$, etc. But it also means that the interesting features of these spaces — smoothness conditions, boundary behavior, growth rates — are invisible to the Hilbert space structure alone and must be captured by additional data (a specific choice of basis, a specific norm equivalence, or additional algebraic structure like a multiplication operation).

**Example 6 (Unitary equivalence in action).** Consider $L^2[0, 2\pi]$ with orthonormal basis $e_n(t) = \frac{1}{\sqrt{2\pi}} e^{int}$. The Fourier transform $\mathcal{F}: f \mapsto (\hat{f}_n)_{n \in \mathbb{Z}}$ is a unitary map from $L^2[0, 2\pi]$ to $\ell^2(\mathbb{Z})$. Parseval's identity is precisely the statement that $\mathcal{F}$ is an isometry:

$$\|f\|_{L^2}^2 = \int_0^{2\pi} |f(t)|^2\, dt = \sum_{n \in \mathbb{Z}} |\hat{f}_n|^2 = \|\hat{f}\|_{\ell^2}^2.$$

This is why we can study convolution operators, regularity questions, and PDE problems interchangeably in the "physical" domain $L^2$ or the "frequency" domain $\ell^2$, whichever is more convenient.

### Non-separable Hilbert spaces

For completeness: non-separable Hilbert spaces exist (e.g., $\ell^2(I)$ for an uncountable index set $I$) and are classified by the cardinality of their orthonormal bases. Two Hilbert spaces are unitarily isomorphic if and only if their orthonormal bases have the same cardinality — this cardinal number is called the **Hilbert dimension**. The separable case corresponds to Hilbert dimension $\aleph_0$.

Non-separable Hilbert spaces arise in certain areas of mathematical physics (e.g., the GNS construction in algebraic quantum field theory can produce non-separable spaces) and abstract harmonic analysis (e.g., $L^2$ of a non-$\sigma$-compact locally compact group). However, in most concrete applications — quantum mechanics, signal processing, PDE theory — the relevant Hilbert spaces are separable.

### A summary of the hierarchy

To keep perspective on where we are in the landscape of spaces:

| Structure | Space type | Key property |
|---|---|---|
| Vector space + norm | Normed space | Distances, convergence |
| Complete normed space | Banach space | Limits of Cauchy sequences exist |
| Banach space + parallelogram law | Hilbert space | Inner product, orthogonality, projections |
| Hilbert space + separability | Separable Hilbert space | Countable orthonormal basis, unitarily isomorphic to $\ell^2$ |

Each level adds structure and unlocks more powerful theorems. The price for the additional structure is a smaller class of spaces — but the theorems that hold are correspondingly stronger.

---

## What's Next

With the geometry of Hilbert spaces in hand, we now have a space where projections, orthogonal decompositions, and Fourier expansions all work cleanly. But we have been studying individual elements and their representations. The next step is to study the **functionals** and **operators** that act on these spaces.

In the next article, we turn to **dual spaces and the Hahn-Banach theorem**. We will leave the comfort of Hilbert spaces (where every functional is an inner product) and confront the general Banach space setting, where the existence of non-trivial continuous linear functionals is far from obvious. The Hahn-Banach theorem resolves this by showing that functionals can always be extended from subspaces to the whole space — a result that launched modern duality theory.

---

*This is Part 3 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 2 — Normed and Banach Spaces](/en/functional-analysis/02-normed-and-banach/)*

*Next: [Part 4 — Dual Spaces and Hahn-Banach](/en/functional-analysis/04-dual-spaces-hahn-banach/)*
