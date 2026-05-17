---
title: "Inner Product Spaces and Hilbert Spaces"
date: 2021-03-08 09:00:00
tags:
  - functional-analysis
  - hilbert-spaces
  - inner-products
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 2
series_total: 6
translationKey: "functional-analysis-2"
description: "Hilbert spaces bring geometry to infinite dimensions --- orthogonality, projections, and Fourier series all live here."
---

## Geometry in infinite dimensions

Banach spaces have distance and linear structure, but no notion of angle. You can't talk about orthogonality in $\ell^1$ or $C[0,1]$ (with sup norm) in any natural way. To get geometry --- perpendicularity, projections, least-squares approximation --- you need an inner product.

The intuition from $\mathbb{R}^n$ is right: an inner product lets you define the angle between two vectors as $\cos\theta = \langle x, y \rangle / (\|x\|\|y\|)$. Two vectors are orthogonal when this is zero. Projecting $x$ onto a subspace means finding the closest point, and the error is perpendicular to the subspace. All of this generalizes perfectly to infinite dimensions, provided the space is complete.

I find it remarkable that the passage from Banach to Hilbert --- adding just one structural axiom (the parallelogram law) --- gives you so much extra power: orthogonal decomposition, the Riesz representation, self-duality. The geometry that inner products provide is not a luxury; it's what makes the theory tractable.

![Orthogonal projection onto a subspace](/images/functional-analysis/fig02_projection.png)

## Inner product spaces

An **inner product** on a vector space $X$ over $\mathbb{C}$ is a map $\langle \cdot, \cdot \rangle: X \times X \to \mathbb{C}$ satisfying:

$$\langle x, x \rangle \ge 0, \quad \langle x, x \rangle = 0 \iff x = 0,$$

$$\langle x, y \rangle = \overline{\langle y, x \rangle},$$

$$\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle.$$

Every inner product induces a norm: $\|x\| = \sqrt{\langle x, x \rangle}$. But not every norm comes from an inner product --- and this is where the story gets interesting.

**The parallelogram law test.** A norm $\|\cdot\|$ comes from an inner product if and only if:

$$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2 \quad \forall\, x, y.$$

This identity says: the sum of the squares of the diagonals of a parallelogram equals the sum of the squares of its sides. If the norm satisfies this, the **polarization identity** recovers the inner product:

$$\langle x, y \rangle = \frac{1}{4}\left(\|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2\right).$$

**Example 1 (Which $\ell^p$ has an inner product?).** The $\ell^2$ norm satisfies the parallelogram law. The $\ell^1$ and $\ell^\infty$ norms don't. Take $x = (1, 0, 0, \ldots)$ and $y = (0, 1, 0, \ldots)$ in $\ell^1$: $\|x+y\|_1^2 + \|x-y\|_1^2 = 4 + 4 = 8$, but $2\|x\|_1^2 + 2\|y\|_1^2 = 2 + 2 = 4$. The law fails. Among all $\ell^p$ spaces, only $p = 2$ gives an inner product space. This is why $\ell^2$ is geometrically special.

**Example 2.** The space $L^2[0,1]$ with:

$$\langle f, g \rangle = \int_0^1 f(t)\overline{g(t)}\, dt.$$

This is the inner product that makes Fourier analysis work. The functions $e^{2\pi i n t}$ are orthogonal under this pairing.

**Example 3.** The Sobolev inner product on $H^1[0,1]$:

$$\langle f, g \rangle_{H^1} = \int_0^1 f\bar{g}\, dt + \int_0^1 f'\bar{g}'\, dt.$$

This gives a different Hilbert space structure on a subspace of $L^2$, incorporating derivative information.

## The Cauchy-Schwarz inequality

The single most important inequality in all of analysis:

$$|\langle x, y \rangle| \le \|x\|\,\|y\|,$$

with equality if and only if $x$ and $y$ are linearly dependent.

*Proof.* For $y = 0$, both sides are zero. For $y \ne 0$, set $\alpha = \langle x, y \rangle / \|y\|^2$. Then:

$$0 \le \|x - \alpha y\|^2 = \langle x - \alpha y, x - \alpha y \rangle = \|x\|^2 - \frac{|\langle x, y \rangle|^2}{\|y\|^2}.$$

Rearranging gives $|\langle x, y \rangle|^2 \le \|x\|^2 \|y\|^2$. Equality holds iff $x = \alpha y$. $\square$

This single inequality implies:
1. The triangle inequality for the induced norm (so it's actually a norm).
2. The inner product is continuous in both variables.
3. We can define angles: $\cos\theta = \text{Re}\,\langle x, y \rangle / (\|x\|\,\|y\|)$.

In $L^2$, Cauchy-Schwarz becomes $|\int fg| \le (\int |f|^2)^{1/2}(\int |g|^2)^{1/2}$. In probability, it gives $|\text{Cov}(X,Y)| \le \text{SD}(X)\text{SD}(Y)$.

## Hilbert spaces

A **Hilbert space** is a complete inner product space:

$$\text{Hilbert space} = \text{Banach space with inner product}.$$

Both $\ell^2$ and $L^2[0,1]$ are Hilbert spaces. The space of finite sequences with the $\ell^2$ norm is an inner product space but not Hilbert (not complete).

**Key non-examples.** The spaces $\ell^p$ for $p \ne 2$ are Banach but not Hilbert. The space $C[0,1]$ with the sup norm is Banach but not Hilbert (parallelogram law fails). Having an inner product is a very special property.

## Orthogonality and projections

Vectors $x, y$ are **orthogonal** (written $x \perp y$) if $\langle x, y \rangle = 0$. A set $S$ is **orthonormal** if its elements are pairwise orthogonal unit vectors.

The key geometric theorem of Hilbert space theory:

**Theorem (Orthogonal projection).** Let $M$ be a closed subspace of a Hilbert space $H$. For every $x \in H$, there exists a unique $m_0 \in M$ minimizing the distance:

$$\|x - m_0\| = \inf_{m \in M} \|x - m\| =: d(x, M).$$

Moreover, $m_0$ is characterized by the orthogonality condition $(x - m_0) \perp M$: that is, $\langle x - m_0, m \rangle = 0$ for all $m \in M$.

*Proof.* Let $d = \inf_{m \in M} \|x - m\|$. Take a minimizing sequence $(m_n)$ with $\|x - m_n\| \to d$. Apply the parallelogram law to $u = m_n - x$ and $v = m_k - x$:

$$\|m_n - m_k\|^2 = 2\|m_n - x\|^2 + 2\|m_k - x\|^2 - \|m_n + m_k - 2x\|^2.$$

Since $(m_n + m_k)/2 \in M$ (subspace!), $\|m_n + m_k - 2x\|^2 = 4\|(m_n + m_k)/2 - x\|^2 \ge 4d^2$. So:

$$\|m_n - m_k\|^2 \le 2\|m_n - x\|^2 + 2\|m_k - x\|^2 - 4d^2 \to 0.$$

The sequence is Cauchy. Completeness gives $m_0 = \lim m_n \in M$ (closed subspace). For the orthogonality characterization: if $v \in M$, expand $\|x - m_0 - tv\|^2 \ge d^2$ for all $t \in \mathbb{R}$ and optimize over $t$. $\square$

This fails in general Banach spaces. In $\ell^1$, closest points in subspaces need not be unique. The parallelogram law --- hence the inner product --- is essential for uniqueness.

**Corollary (Orthogonal decomposition).** $H = M \oplus M^\perp$ where $M^\perp = \{x \in H : \langle x, m \rangle = 0 \ \forall m \in M\}$.

Every $x$ splits uniquely as $x = P_M x + (x - P_M x)$ with $P_M x \in M$ and $(x - P_M x) \in M^\perp$. The map $P_M: H \to H$ is the **orthogonal projection** onto $M$. It satisfies $P_M^2 = P_M$, $P_M^* = P_M$, and $\|P_M\| = 1$ (unless $M = \{0\}$).

**Example 4 (Least-squares in $L^2$).** To find the best polynomial approximation of degree $\le n$ to a function $f \in L^2[0,1]$, project $f$ onto the closed subspace $M = \{$polynomials of degree $\le n\}$. The projection $P_M f$ is the unique polynomial minimizing $\int_0^1 |f - p|^2 dt$, and the error $f - P_M f$ is orthogonal to every polynomial of degree $\le n$.

## Orthonormal bases and Fourier series

An orthonormal set $\{e_\alpha\}_{\alpha \in A}$ in $H$ is an **orthonormal basis** (or complete orthonormal system) if the only vector orthogonal to all $e_\alpha$ is zero. Equivalently: the closed linear span of $\{e_\alpha\}$ is all of $H$.

**Theorem (Bessel's inequality).** For any orthonormal set $\{e_n\}$ and any $x \in H$:

$$\sum_n |\langle x, e_n \rangle|^2 \le \|x\|^2.$$

**Theorem (Parseval's identity).** If $\{e_n\}_{n=1}^\infty$ is an orthonormal basis of $H$, then for every $x \in H$:

$$x = \sum_{n=1}^\infty \langle x, e_n \rangle\, e_n, \quad \|x\|^2 = \sum_{n=1}^\infty |\langle x, e_n \rangle|^2.$$

The coefficients $c_n = \langle x, e_n \rangle$ are the **Fourier coefficients** of $x$ with respect to the basis. The first equation says every element is determined by its Fourier coefficients; the second says the norm decomposes as a sum of squared coefficients.

*Proof sketch.* Let $S_N = \sum_{n=1}^N c_n e_n$. Then $S_N = P_M x$ where $M = \text{span}(e_1, \ldots, e_N)$, and $\|x - S_N\|^2 = \|x\|^2 - \sum_{n=1}^N |c_n|^2$ (Pythagorean theorem). As $N \to \infty$, completeness of the basis forces $\|x - S_N\| \to 0$. $\square$

**Example 5 (Classical Fourier series).** In $L^2[0, 2\pi]$, the functions $e_n(t) = \frac{1}{\sqrt{2\pi}} e^{int}$ form an orthonormal basis. Parseval's identity becomes:

$$\frac{1}{2\pi}\int_0^{2\pi} |f(t)|^2\, dt = \sum_{n=-\infty}^{\infty} |\hat{f}(n)|^2,$$

which is the classical Parseval theorem. The completeness of this system (the fact that the exponentials form a basis) is a deep theorem --- equivalent to the density of trigonometric polynomials in $L^2$.

**Example 6 (Haar wavelets).** The Haar system on $[0,1]$ is another orthonormal basis of $L^2[0,1]$, with different approximation properties: Haar partial sums converge for functions with jumps, where Fourier partial sums exhibit Gibbs phenomenon. Same abstract framework, different concrete basis, different applications.

## The Riesz representation theorem

**Theorem (Riesz-Frechet).** Every continuous linear functional $f: H \to \mathbb{C}$ on a Hilbert space has the form $f(x) = \langle x, y \rangle$ for a unique $y \in H$, with $\|f\| = \|y\|$.

*Proof.* If $f = 0$, take $y = 0$. Otherwise, $\ker f$ is a closed subspace of codimension 1 (closed because $f$ is continuous). By the orthogonal decomposition, $H = \ker f \oplus (\ker f)^\perp$, and $(\ker f)^\perp$ is one-dimensional. Pick $z \in (\ker f)^\perp$ with $f(z) = 1$. For any $x \in H$:

$$x = \underbrace{(x - f(x)z)}_{\in \ker f} + f(x)z.$$

Taking the inner product with $z$: $\langle x, z \rangle = f(x)\|z\|^2$, so $f(x) = \langle x, z/\|z\|^2 \rangle$. Set $y = z/\|z\|^2$. Then $\|f\| = \|y\|$ by Cauchy-Schwarz (with equality achieved at $x = y$). $\square$

This identifies $H^* \cong H$ (antilinearly). The dual of a Hilbert space is itself. No other Banach space has this self-duality property (except finite-dimensional ones). It's why Hilbert spaces are the "nicest" infinite-dimensional spaces --- you never need to leave the space to talk about functionals.

**Consequence.** Every bounded sesquilinear form $a: H \times H \to \mathbb{C}$ can be written as $a(x,y) = \langle Ax, y \rangle$ for a unique bounded operator $A$. This connects forms to operators --- the starting point for the Lax-Milgram theorem (Part 6).

## Separability and classification

A Hilbert space is **separable** if it has a countable orthonormal basis (equivalently, if it has a countable dense subset).

**Theorem.** All separable infinite-dimensional Hilbert spaces are isometrically isomorphic to $\ell^2$.

*Proof.* If $\{e_n\}$ is a countable orthonormal basis, the map $x \mapsto (\langle x, e_n \rangle)_{n=1}^\infty$ is an isometric isomorphism from $H$ onto $\ell^2$ (by Parseval). $\square$

This is remarkable. $L^2[0,1]$, $L^2(\mathbb{R})$, the Hardy space $H^2(\mathbb{D})$, the Sobolev space $H^1[0,1]$ --- they're all isomorphic to $\ell^2$ as Hilbert spaces. The differences show up in which operators act naturally on them, which subspaces correspond to meaningful function classes, and which bases are computationally useful.

Non-separable Hilbert spaces exist (e.g., $\ell^2(A)$ for uncountable $A$) but they rarely appear in applications. For this series, "Hilbert space" means separable unless stated otherwise.

## What's next

With Hilbert space geometry in hand, the next step is to study the maps between spaces: bounded linear operators. These are the infinite-dimensional analogues of matrices, and they behave in ways that would horrify anyone used to finite-dimensional linear algebra.

---

*This is Part 2 of [Functional Analysis](/en/series/functional-analysis/) (6 parts).
Previous: [Part 1 --- Metric Spaces, Normed Spaces, and Banach Spaces](/en/functional-analysis/01-metric-and-normed-spaces/) · Next: [Part 3 --- Bounded Linear Operators and Functionals](/en/functional-analysis/03-bounded-operators/)*
