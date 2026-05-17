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

# Hilbert Spaces — Geometry in Infinite Dimensions

## Why I Like Hilbert Spaces More Than Banach Spaces

If a Banach space is a normed space that has agreed to be complete, a Hilbert space is a Banach space that has further agreed to admit angles. That extra agreement is what restores almost all of finite-dimensional geometry — orthogonality, projection, the right-angle Pythagoras identity — to the infinite-dimensional setting. In return, the structure is rigid enough that every separable Hilbert space looks like exactly one model, $\ell^2$. There is essentially one infinite-dimensional Hilbert space up to isomorphism, and everything we ever do in the theory amounts to picking a basis and counting coordinates.

The price of admission is a single extra axiom: an inner product. The reward, geometric and computational, is enormous. Linear regression, Fourier series, quantum mechanics, signal processing, the energy method in PDE — every one of these is a Hilbert-space argument in disguise. The inner product gives them all a single grammar.

## Inner Products and Hilbert Spaces

Let $\mathcal{H}$ be a vector space over $\mathbb{C}$ (the real case is similar with conjugate signs removed). An **inner product** is a function $\langle \cdot, \cdot \rangle : \mathcal{H} \times \mathcal{H} \to \mathbb{C}$ satisfying for all $x, y, z \in \mathcal{H}$ and $\alpha \in \mathbb{C}$:

1. $\langle x, x \rangle \geq 0$, with equality iff $x = 0$ (positive definiteness).
2. $\langle x, y \rangle = \overline{\langle y, x \rangle}$ (conjugate symmetry).
3. $\langle \alpha x + z, y \rangle = \alpha \langle x, y \rangle + \langle z, y \rangle$ (linearity in the first argument).

Conjugate symmetry then forces *conjugate*-linearity in the second argument: $\langle x, \alpha y \rangle = \overline{\alpha} \langle x, y \rangle$. (Some authors put linearity in the second slot; the choice is irrelevant once everyone agrees.)

The inner product induces a norm by $\|x\| = \langle x, x \rangle^{1/2}$. It is positive definite by axiom 1, homogeneous because $\langle \alpha x, \alpha x \rangle = |\alpha|^2 \langle x, x \rangle$, and the triangle inequality follows from the Cauchy-Schwarz inequality, which is the central lemma of the subject.

**Cauchy-Schwarz.** For all $x, y \in \mathcal{H}$, $|\langle x, y \rangle| \leq \|x\| \|y\|$, with equality iff $x, y$ are linearly dependent.

*Proof.* If $y = 0$, both sides vanish. Otherwise, set $\lambda = \langle x, y \rangle / \|y\|^2$. Expand $0 \leq \|x - \lambda y\|^2 = \|x\|^2 - 2 \mathrm{Re}(\overline{\lambda} \langle x, y \rangle) + |\lambda|^2 \|y\|^2$, simplify with the value of $\lambda$, and the inequality drops out. $\square$

A **pre-Hilbert space** is a vector space with an inner product. A **Hilbert space** is a pre-Hilbert space that is complete in the induced norm.

![Inner product geometry: angle and length](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_v2_03_1_inner_product.png)

### Examples

- $\mathbb{C}^n$ with $\langle x, y \rangle = \sum_i x_i \overline{y_i}$ (the standard Hermitian inner product).
- $\ell^2$ with $\langle x, y \rangle = \sum_n x_n \overline{y_n}$. The Cauchy-Schwarz inequality on $\mathbb{C}^n$ passes to a limit to give convergence of the sum.
- $L^2(\Omega, \mu)$ with $\langle f, g \rangle = \int_\Omega f \overline{g}\, d\mu$ for any measure space $(\Omega, \mu)$. This is the Hilbert space of every quantum mechanic and PDE analyst.
- The Sobolev space $H^1(\Omega) = W^{1,2}(\Omega)$ with $\langle f, g \rangle_{H^1} = \int (f \overline{g} + \nabla f \cdot \overline{\nabla g})$.

### Worked numerical example

In $L^2[0,1]$, take $f(t) = 1$ and $g(t) = t$. Then $\|f\|_2^2 = 1$, $\|g\|_2^2 = \int_0^1 t^2\,dt = 1/3$, $\langle f, g \rangle = \int_0^1 t\,dt = 1/2$. Cauchy-Schwarz: $|\langle f, g \rangle| = 1/2 \leq \|f\| \cdot \|g\| = 1 \cdot 1/\sqrt{3} \approx 0.577$. The angle between $f$ and $g$ is $\theta$ with $\cos \theta = (1/2)/(1/\sqrt{3}) = \sqrt{3}/2$, so $\theta = \pi/6 = 30°$. So in the geometry of $L^2$, the constant function and the identity function meet at a $30$ degree angle. This is what I mean by "angles in function spaces": once we have an inner product, the language of geometry transports verbatim.

## The Parallelogram Identity

The single algebraic identity that distinguishes inner-product norms from generic norms is the **parallelogram law**:
$$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2.$$
Geometrically: the sum of squared diagonals of a parallelogram equals the sum of squared sides. Proof: expand both sides using $\|u\|^2 = \langle u, u \rangle$.

The remarkable converse is the **Jordan-von Neumann theorem**: a norm satisfies the parallelogram law iff it comes from an inner product. The inner product is then recovered by the **polarization identity**:
$$\langle x, y \rangle = \tfrac{1}{4}\big( \|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2 \big).$$
So I can detect "Hilbertness" of a Banach space by checking a single algebraic identity. The check is concrete: take $x = (1,0)$ and $y = (0,1)$ in $\ell^p_2$. Then $\|x+y\|_p^2 + \|x-y\|_p^2 = 2 \cdot 2^{2/p}$, while $2\|x\|^2 + 2\|y\|^2 = 4$. Equality forces $2^{2/p} = 2$, i.e. $p = 2$. So among the $\ell^p$ family, only $\ell^2$ is Hilbert. The $\ell^p$ family with $p \neq 2$ is forever stuck being only Banach.

### Why this matters

The parallelogram law is the algebraic shadow of *roundness* of the unit ball. Any norm coming from an inner product gives a strictly convex, smooth ball — no corners, no flats. This is what unlocks unique best approximation: in a Hilbert space, every closed convex set has a unique closest point, no exceptions. In a Banach space without strict convexity, that uniqueness fails. The whole power of Hilbert space theory traces back to this single geometric property.

## Orthogonality and the Pythagorean Theorem

Two vectors $x, y \in \mathcal{H}$ are **orthogonal** if $\langle x, y \rangle = 0$, written $x \perp y$. A subset $S \subseteq \mathcal{H}$ is orthogonal if its members are pairwise orthogonal, and **orthonormal** if additionally each has norm $1$.

The Pythagorean theorem holds in full generality: if $x \perp y$, then $\|x + y\|^2 = \|x\|^2 + \|y\|^2$. By induction, for an orthogonal family $\{x_1, \ldots, x_n\}$, $\|\sum x_i\|^2 = \sum \|x_i\|^2$.

Given a subspace $M \subseteq \mathcal{H}$, its **orthogonal complement** is $M^\perp = \{ y : \langle y, x \rangle = 0 \text{ for all } x \in M \}$. The orthogonal complement is always closed (being the intersection of zero-sets of continuous functionals $x \mapsto \langle x, y \rangle$ for $y \in M$).

## Orthogonal Projection

Let $M \subseteq \mathcal{H}$ be a closed subspace. The **orthogonal projection** $P_M : \mathcal{H} \to M$ is defined by: for each $x \in \mathcal{H}$, $P_M x$ is the unique element of $M$ closest to $x$. The closeness comes from the parallelogram law: take a sequence $(y_n) \subseteq M$ with $\|x - y_n\| \to d(x, M)$, apply the parallelogram law to $x - y_n$ and $x - y_m$, and conclude that $(y_n)$ is Cauchy. Completeness gives a limit $y^* \in M$ (since $M$ is closed), uniqueness from strict convexity.

The defining feature of the projection is the orthogonality relation $x - P_M x \perp M$. Any element of $M$ closer to $x$ would contradict the minimality, and the difference is then orthogonal to $M$ for the same reason.

![Orthogonal projection of a vector onto a closed subspace](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_v2_03_2_orthogonal_proj.png)

This projection has all the properties one expects: it is bounded (with $\|P_M\| = 1$ when $M \neq 0$), idempotent ($P_M^2 = P_M$), and self-adjoint ($\langle P_M x, y \rangle = \langle x, P_M y \rangle$). Conversely, any bounded operator on $\mathcal{H}$ that is self-adjoint and idempotent is an orthogonal projection onto its range.

The orthogonal decomposition theorem follows: $\mathcal{H} = M \oplus M^\perp$ for any closed subspace $M$. Every $x$ writes uniquely as $x = P_M x + (x - P_M x)$ with the two pieces orthogonal. As a corollary, $(M^\perp)^\perp = M$ for closed $M$ — a far-from-obvious fact whose Banach-space analog requires Hahn-Banach.

### Numerical example: least squares

Take $\mathcal{H} = \mathbb{R}^3$ and let $M$ be the plane spanned by $u = (1, 0, 0)$ and $v = (0, 1, 0)$. To project $x = (3, 4, 5)$ onto $M$: $P_M x = \langle x, u \rangle u + \langle x, v \rangle v = 3 \cdot u + 4 \cdot v = (3, 4, 0)$. The residual $x - P_M x = (0, 0, 5)$ is orthogonal to $M$, as expected.

This is exactly the linear regression formula. Given data points $(x_i, y_i)$, the best-fitting line $y = ax + b$ in the least-squares sense is the orthogonal projection of the vector $y \in \mathbb{R}^n$ onto the two-dimensional subspace spanned by $(1, 1, \ldots, 1)$ and $(x_1, \ldots, x_n)$. The least-squares "solution" is just an orthogonal projection in disguise. The same machinery generalizes to least-squares fitting in $L^2$ (best polynomial approximation, best Fourier approximation, best wavelet approximation): all are orthogonal projections in some Hilbert space.

### Why this matters

In any Banach space without an inner product, the closest-point projection might not exist or might not be unique. Even if it exists, it need not be linear. The fact that the projection in a Hilbert space is *linear and continuous* is precisely what makes Hilbert space the right setting for variational methods, optimization, and any algorithm involving "the closest function with property P." Without strict convexity, such "closest function" arguments break down.

## Orthonormal Bases and Fourier Coefficients

Let $\mathcal{H}$ be a separable Hilbert space. An **orthonormal sequence** $(e_n)_{n \geq 1}$ in $\mathcal{H}$ is **complete** (or a **Hilbert basis**) if $\overline{\mathrm{span}}\{e_n\} = \mathcal{H}$. By Gram-Schmidt, every separable Hilbert space has an orthonormal basis (as we will see below).

For an orthonormal basis $(e_n)$ and any $x \in \mathcal{H}$, define the **Fourier coefficients** $c_n = \langle x, e_n \rangle$. The basic results:

- (Bessel's inequality) $\sum_n |c_n|^2 \leq \|x\|^2$.
- (Parseval) The orthonormal sequence is a Hilbert basis iff $\sum_n |c_n|^2 = \|x\|^2$ for every $x$. In that case, $x = \sum_n c_n e_n$ in norm.
- (Plancherel) $\langle x, y \rangle = \sum_n c_n(x) \overline{c_n(y)}$.

![Orthonormal basis and Fourier coefficients](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_v2_03_3_orthonormal_basis.png)

The map $x \mapsto (c_n)_{n \geq 1}$ is a unitary (i.e., norm-preserving and surjective) isomorphism $\mathcal{H} \to \ell^2$. So every separable infinite-dimensional Hilbert space is isomorphic to $\ell^2$. There is, up to isomorphism, exactly one such space.

### The trigonometric basis of $L^2[0, 2\pi]$

The functions $e_n(t) = e^{int}/\sqrt{2\pi}$ for $n \in \mathbb{Z}$ form an orthonormal basis of $L^2[0, 2\pi]$. The Fourier coefficients are $c_n = \frac{1}{\sqrt{2\pi}}\int_0^{2\pi} f(t) e^{-int}\,dt$, and Parseval reads
$$\|f\|_2^2 = \sum_{n \in \mathbb{Z}} |c_n|^2.$$
The classical Fourier series convergence question is: in what sense does $\sum c_n e_n$ converge to $f$? In $L^2$, the answer is *always*, by Parseval. In $L^p$ for $p \neq 2$, the question is hard (Carleson's theorem for $p=2$ pointwise a.e. convergence is a deep result, and for $p=1$ pointwise convergence can fail spectacularly on a positive-measure set per Kolmogorov). The Hilbert structure is what makes the convergence question trivial.

![Trigonometric basis of L^2[0,1]](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_v2_03_6_l2_basis.png)

### Numerical example

Consider $f(t) = t$ on $[0, 2\pi]$, viewed in $L^2[0, 2\pi]$. The Fourier coefficients (with the $e_n = e^{int}/\sqrt{2\pi}$ basis) are $c_n = -\sqrt{2\pi}/(in)$ for $n \neq 0$, and $c_0 = \sqrt{2\pi} \cdot \pi / \sqrt{2\pi} = \pi\sqrt{2\pi}$. Parseval: $\|f\|_2^2 = \int_0^{2\pi} t^2 \,dt = 8\pi^3/3$. The sum $\sum |c_n|^2 = 2\pi^3 + \sum_{n \neq 0} 2\pi/n^2 = 2\pi^3 + 4\pi \cdot \pi^2/6 \cdot 2/2$. Working through: $2\pi^3 + 2\pi (\pi^2/3) = 2\pi^3 + 2\pi^3/3 = 8\pi^3/3$. The identity holds.

This is one of those numerical confirmations that does no real work but is reassuring: the same formula that you compute classically (Parseval's identity for Fourier series) is the same as Parseval as a Hilbert-space theorem.

![Parseval's identity: norm equals sum of squared coefficients](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_v2_03_5_parseval.png)

## Gram-Schmidt and Constructing Bases

Given any sequence $(v_n)_{n \geq 1}$ of linearly independent vectors in $\mathcal{H}$, the **Gram-Schmidt process** constructs an orthonormal sequence $(e_n)$ with the same span at every step:
$$u_1 = v_1,\quad e_1 = u_1/\|u_1\|;\quad u_{n+1} = v_{n+1} - \sum_{k=1}^{n} \langle v_{n+1}, e_k \rangle e_k,\quad e_{n+1} = u_{n+1} / \|u_{n+1}\|.$$
At each step, $u_{n+1}$ is the residual of $v_{n+1}$ after projecting onto $\mathrm{span}\{e_1, \ldots, e_n\}$, then normalized. The procedure works as long as the $v_n$ are linearly independent.

![Gram-Schmidt orthogonalization process applied to a finite set](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_v2_03_7_gram_schmidt.png)

### Numerical example

In $L^2[-1, 1]$, take $v_n(t) = t^{n-1}$. Gram-Schmidt produces (after normalization) the **Legendre polynomials**: $P_0(t) = 1/\sqrt{2}$, $P_1(t) = \sqrt{3/2}\, t$, $P_2(t) = \sqrt{5/8}(3t^2 - 1)$, $\ldots$. Each $P_n$ is an orthogonal projection of $t^n$ onto the orthogonal complement of $\mathrm{span}\{1, t, \ldots, t^{n-1}\}$, then normalized. The Legendre polynomials are an orthonormal basis of $L^2[-1, 1]$, so any function in $L^2[-1, 1]$ admits a "Legendre expansion" — the same idea as Fourier series with a different basis.

Different bases have different convergence behavior: Legendre polynomials are excellent for smooth functions on $[-1, 1]$ but have nothing special on $[0, 2\pi]$; Fourier series are natural for periodic functions; Haar wavelets are natural for piecewise-defined functions. Choice of basis is half the art of doing applied analysis in a Hilbert space.

### Why this matters

Gram-Schmidt is constructive — given any countable spanning set, it produces an orthonormal basis, which immediately gives Fourier coefficients and Parseval. So separability (countable dense set) plus inner product gives orthonormal basis automatically. Many existence proofs in Hilbert space theory reduce to "apply Gram-Schmidt to a countable dense set" without further argument.

## The Riesz Representation Theorem

The single most useful theorem in Hilbert space theory:

**Riesz representation theorem.** Let $\varphi: \mathcal{H} \to \mathbb{C}$ be a continuous linear functional on a Hilbert space. Then there is a unique $y \in \mathcal{H}$ with $\varphi(x) = \langle x, y \rangle$ for all $x \in \mathcal{H}$. Moreover, $\|\varphi\| = \|y\|$.

In words: every continuous linear functional on a Hilbert space is given by inner product with some vector. The dual of $\mathcal{H}$ is naturally isometric (anti-)isomorphic to $\mathcal{H}$ itself.

![Riesz representation theorem: every continuous functional comes from an inner product](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_v2_03_4_riesz.png)

*Proof sketch.* If $\varphi = 0$, take $y = 0$. Otherwise $\ker \varphi$ is a closed proper subspace, so $(\ker \varphi)^\perp$ contains a unit vector $z$ (orthogonal complement is non-trivial because $\ker \varphi$ is a proper closed subspace of codimension $1$). Set $y = \overline{\varphi(z)} z$. For any $x \in \mathcal{H}$, write $x = (x - \alpha z) + \alpha z$ where $\alpha = \langle x, z \rangle$ makes the first piece in $\ker \varphi$ (a calculation). Then $\varphi(x) = \alpha \varphi(z)$ and $\langle x, y \rangle = \alpha \overline{\varphi(z)} \cdot \overline{\overline{1}} = \alpha \varphi(z)$. They match. $\square$

### Numerical example

In $\ell^2$, consider the functional $\varphi(x) = x_1 + x_2/2 + x_3/4 + \cdots = \sum_n x_n / 2^{n-1}$. By Riesz, $\varphi(x) = \langle x, y \rangle$ for some $y \in \ell^2$. Reading off, $y = (1, 1/2, 1/4, \ldots)$, with $\|y\|_2^2 = \sum 1/4^{n-1} = 4/3$. So $\|\varphi\| = \|y\|_2 = 2/\sqrt{3}$. Sanity check: by Cauchy-Schwarz, $|\varphi(x)| \leq \|x\|_2 \|y\|_2 = (2/\sqrt{3}) \|x\|_2$, and equality is attained at $x = y$, confirming $\|\varphi\| = 2/\sqrt{3}$.

### Why this matters

Riesz says the dual of a Hilbert space is *itself*. So duality, which is a delicate and abstract construction in general Banach spaces (Article 4 will spend most of its time on the Hahn-Banach theorem just to *produce* enough functionals), is trivially solved in Hilbert spaces. Every linear functional is an inner product. This is why so many calculations in PDE and quantum mechanics flow effortlessly: when you need a linear functional, you just write down a vector and pair against it.

A second consequence: every bounded sesquilinear form $b(x, y)$ on $\mathcal{H} \times \mathcal{H}$ has the form $b(x, y) = \langle T x, y \rangle$ for a unique bounded operator $T$. This is how variational formulations of PDEs (the Lax-Milgram theorem, the heart of finite element method) reduce to operator-theoretic problems — Article 12 expands on this.

## Adjoints and Self-Adjoint Operators

Let $T: \mathcal{H} \to \mathcal{H}$ be bounded. The **adjoint** $T^*$ is defined by $\langle T x, y \rangle = \langle x, T^* y \rangle$ for all $x, y$. Riesz guarantees $T^*$ exists and is bounded with $\|T^*\| = \|T\|$. Properties: $(S+T)^* = S^* + T^*$, $(\alpha T)^* = \overline{\alpha} T^*$, $(ST)^* = T^* S^*$, $(T^*)^* = T$.

An operator $T$ is **self-adjoint** (or Hermitian) if $T = T^*$. Self-adjoint operators in Hilbert space play the role of real symmetric matrices in finite dimensions. Their spectrum is real, they admit a spectral decomposition (Article 8), and they generate unitary groups (Article 10). Quantum-mechanical observables are self-adjoint operators.

A **unitary** operator satisfies $T^* T = T T^* = I$, equivalently is an isometric isomorphism. The Fourier transform on $L^2(\mathbb{R})$ is a unitary operator; it diagonalizes translation, which is why it solves the heat equation, the wave equation, the Schrödinger equation in elementary cases.

A **normal** operator satisfies $T^* T = T T^*$. Normal operators include self-adjoint and unitary as special cases, and they are the maximal class for which the spectral theorem holds.

### Numerical example

The shift $S: \ell^2 \to \ell^2$, $S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$, has adjoint $S^*(x_1, x_2, \ldots) = (x_2, x_3, \ldots)$ (the backward shift). Check: $\langle S x, y \rangle = \sum_{n \geq 2} x_{n-1} \overline{y_n} = \sum_{n \geq 1} x_n \overline{y_{n+1}} = \langle x, S^* y \rangle$. The shift is not self-adjoint or normal: $S^* S = I$ (shift right, then left, gives the identity), but $S S^* x = (0, x_2, x_3, \ldots) \neq x$ in general. So the shift exhibits a dissymmetry that is impossible in finite dimensions where $S^* S = I$ implies $S S^* = I$.

## Weak Convergence in Hilbert Space (a Preview)

Article 5 spends a lot of time on weak topologies, but the Hilbert-space case is so clean it is worth previewing here. A sequence $(x_n) \subset \mathcal{H}$ **converges weakly** to $x$, written $x_n \rightharpoonup x$, if $\langle x_n, y \rangle \to \langle x, y \rangle$ for every $y \in \mathcal{H}$. By Riesz, this is the same as: $\varphi(x_n) \to \varphi(x)$ for every continuous linear functional $\varphi$.

Weak convergence is strictly weaker than norm convergence. The standard basis $(e_n) \subset \ell^2$ has $\langle e_n, y \rangle = y_n \to 0$ for any fixed $y \in \ell^2$ (since the coefficients of $y$ are square-summable, hence vanish), so $e_n \rightharpoonup 0$. But $\|e_n\| = 1$ for every $n$, so $(e_n)$ does not converge to $0$ in norm.

The **Banach-Alaoglu theorem** (Article 5) implies, in the Hilbert-space setting, that every bounded sequence has a weakly convergent subsequence. This is the substitute for compactness of the closed unit ball — recall that in infinite dimensions the norm-closed ball is *not* compact, but in the weak topology it is. The "weak compactness" of bounded sets is the lever that makes variational methods work: minimizing sequences for energy functionals can be assumed to have weak limits, and the limits are the desired minimizers.

A subtle but important point: in $\mathcal{H}$, weak convergence plus norm-convergence of norms implies norm convergence. That is, $x_n \rightharpoonup x$ and $\|x_n\| \to \|x\|$ imply $\|x_n - x\| \to 0$. This is sometimes called the *Radon-Riesz property* or *Kadec-Klee property*. The proof: $\|x_n - x\|^2 = \|x_n\|^2 - 2\mathrm{Re}\langle x_n, x \rangle + \|x\|^2 \to \|x\|^2 - 2\|x\|^2 + \|x\|^2 = 0$. So in Hilbert space, the gap between norm and weak convergence collapses precisely when the norms behave correctly.

## The Polarization Identity in Action

The polarization identity reduces inner-product calculations to norm calculations. A corollary that I find genuinely useful: a bounded operator on a *complex* Hilbert space is determined by the diagonal sesquilinear form $x \mapsto \langle T x, x \rangle$. If $\langle T x, x \rangle = \langle S x, x \rangle$ for all $x$, then $T = S$. Proof: by polarization, $\langle T x, y \rangle$ is determined by $\langle T(x \pm y), (x \pm y) \rangle$ and $\langle T(x \pm i y), (x \pm i y) \rangle$, all of which equal the corresponding diagonal expressions for $S$.

This is *false* in real Hilbert space: a rotation by $90°$ in $\mathbb{R}^2$ has $\langle T x, x \rangle = 0$ for all $x$ (because $T x \perp x$ in this case), but $T \neq 0$. The complex case has more redundancy and that redundancy fixes the operator.

Concretely, this means a complex operator is **self-adjoint** iff $\langle T x, x \rangle$ is real for every $x$. (Take adjoint of the equation $\langle T x, x \rangle = \overline{\langle T x, x \rangle}$ and apply polarization.) This characterization is genuinely useful in checking self-adjointness without writing out the adjoint explicitly. For example, the Laplace operator $-\Delta$ on a suitable domain in $L^2(\Omega)$ has $\langle -\Delta f, f \rangle = \int |\nabla f|^2 \geq 0$, real, so $-\Delta$ is self-adjoint (modulo questions about domains, taken up in Article 9).

## Direct Sums and Tensor Products

Two Hilbert spaces $\mathcal{H}_1, \mathcal{H}_2$ admit a **direct sum** $\mathcal{H}_1 \oplus \mathcal{H}_2$ — pairs $(x_1, x_2)$ with inner product $\langle (x_1, x_2), (y_1, y_2) \rangle = \langle x_1, y_1 \rangle_{\mathcal{H}_1} + \langle x_2, y_2 \rangle_{\mathcal{H}_2}$. The direct sum is a Hilbert space (completeness of components implies completeness of pairs). The decomposition $\mathcal{H} = M \oplus M^\perp$ for a closed subspace is a special case.

The **tensor product** $\mathcal{H}_1 \otimes \mathcal{H}_2$ requires more care. Algebraically, take the vector space spanned by formal symbols $x_1 \otimes x_2$ modulo bilinearity, equip it with the inner product $\langle x_1 \otimes x_2, y_1 \otimes y_2 \rangle = \langle x_1, y_1 \rangle \langle x_2, y_2 \rangle$, and then complete. The result is a Hilbert space, and one identifies $\ell^2 \otimes \ell^2 \cong \ell^2(\mathbb{N} \times \mathbb{N})$ and $L^2(\Omega_1) \otimes L^2(\Omega_2) \cong L^2(\Omega_1 \times \Omega_2)$ canonically.

Tensor products of Hilbert spaces are the basis of multi-particle quantum mechanics — the state space of $n$ identical particles is the symmetric (or antisymmetric) tensor power of the one-particle Hilbert space. They are also how multivariate Fourier analysis reduces to univariate Fourier analysis: the Fourier transform on $L^2(\mathbb{R}^d) \cong L^2(\mathbb{R})^{\otimes d}$ is the tensor product of $d$ copies of the one-dimensional Fourier transform.

### Worked numerical example

In $L^2(\mathbb{R}^2)$, the function $f(x, y) = e^{-(x^2 + y^2)/2}$ factors as $f_1(x) \cdot f_2(y)$ with $f_1(x) = f_2(x) = e^{-x^2/2}$. The function lives in the tensor product $L^2(\mathbb{R}) \otimes L^2(\mathbb{R})$, identified with $L^2(\mathbb{R}^2)$. Its norm squared in $L^2(\mathbb{R}^2)$ is $\int e^{-(x^2 + y^2)}\,dx\,dy = \pi$. The tensor norm computes as $\|f_1\|_{L^2}^2 \cdot \|f_2\|_{L^2}^2 = \sqrt{\pi} \cdot \sqrt{\pi} = \pi$. The two norms agree, illustrating the unitary identification.

## Continuity of the Inner Product

The inner product $\langle \cdot, \cdot \rangle : \mathcal{H} \times \mathcal{H} \to \mathbb{C}$ is continuous as a map of two variables. Specifically, if $x_n \to x$ and $y_n \to y$ in norm, then $\langle x_n, y_n \rangle \to \langle x, y \rangle$. The proof: $|\langle x_n, y_n \rangle - \langle x, y \rangle| \leq |\langle x_n - x, y_n \rangle| + |\langle x, y_n - y \rangle| \leq \|x_n - x\| \|y_n\| + \|x\| \|y_n - y\|$, which tends to $0$ since $(\|y_n\|)$ is bounded.

The same need not be true if convergence is replaced by *weak* convergence. If $x_n \rightharpoonup x$ and $y_n \rightharpoonup y$, the inner products $\langle x_n, y_n \rangle$ may not converge — take $x_n = y_n = e_n$ in $\ell^2$, both converging weakly to $0$, but $\langle e_n, e_n \rangle = 1$ for every $n$, not converging to $\langle 0, 0 \rangle = 0$. So the inner product is jointly continuous in the norm topology but only separately continuous in the weak topology. This is one of the small but important subtleties when working with weak convergence.

## Spectral Theorem in Finite Dimensions: Preview of Article 8

Self-adjoint operators on a finite-dimensional Hilbert space have an orthonormal basis of eigenvectors with real eigenvalues — the classical spectral theorem of linear algebra. In infinite dimensions, the eigenvalue/eigenvector picture breaks down: a generic self-adjoint operator has a *spectrum* (Article 8) that may include continuous parts with no eigenvectors. The right generalization replaces eigenvalue decomposition $T = \sum \lambda_i P_i$ with a spectral measure: $T = \int \lambda \, dE(\lambda)$ where $E$ is a projection-valued measure on the spectrum.

The simplest non-trivial example: the multiplication operator $M f(t) = t f(t)$ on $L^2[0,1]$ is self-adjoint with spectrum $[0,1]$ but no eigenvectors at all. There is no $f \in L^2$ with $t f(t) = \lambda f(t)$ except in the sense of distributions (which would force $f$ supported at the single point $\lambda$, hence $f = 0$ in $L^2$). The spectrum of $M$ is "purely continuous." Article 8 will explain how to read this off from the spectral measure $E([\alpha, \beta]) f = \mathbb{1}_{[\alpha,\beta]}(t) f(t)$, which projects onto the part of $f$ with "frequency" in $[\alpha, \beta]$.

## Reproducing Kernel Hilbert Spaces

A subclass of Hilbert spaces deserves a paragraph because of their importance in machine learning, statistics, and PDE. A **reproducing kernel Hilbert space** (RKHS) on a set $\Omega$ is a Hilbert space $\mathcal{H}$ of functions $f: \Omega \to \mathbb{C}$ such that for every $x \in \Omega$, the evaluation functional $\delta_x: f \mapsto f(x)$ is bounded. By Riesz, $\delta_x$ is represented by some $K_x \in \mathcal{H}$: $f(x) = \langle f, K_x \rangle$ for all $f \in \mathcal{H}$. The two-variable function $K(x, y) = K_x(y) = \langle K_x, K_y \rangle$ is the **reproducing kernel**.

Examples:

- $\ell^2$ with the standard inner product is an RKHS over $\mathbb{N}$, kernel $K(m, n) = \delta_{mn}$.
- $L^2[0,1]$ is *not* an RKHS — point evaluation is not even well-defined, since $L^2$ functions are equivalence classes a.e. So an RKHS must consist of genuine functions, not equivalence classes.
- The Sobolev space $H^1[0,1]$ on a bounded interval *is* an RKHS, with a kernel that is essentially Green's function for $-d^2/dt^2 + 1$.
- The Bargmann-Fock space of entire functions with Gaussian-weighted $L^2$ norm has a beautiful reproducing kernel $K(z, w) = e^{z \overline{w}}$.

In machine learning, the kernel trick is exactly the use of an RKHS: a positive-definite kernel $K(x, y)$ corresponds to a unique RKHS, and the seemingly nonlinear "kernel methods" are linear methods in this Hilbert space. Support vector machines, Gaussian process regression, kernel PCA — all are linear algebra in some RKHS. The deep theorem behind this (Moore-Aronszajn) is exactly the construction outlined above.

### Numerical example

The kernel $K(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$ (the Gaussian RBF kernel) defines an RKHS of functions on $\mathbb{R}^d$. For two points $x, y$, the inner product of the corresponding feature maps is $K(x, y)$. So the "feature space" is implicitly defined by the kernel, and computations stay in the original space — the famous trick that lets SVMs do nonlinear classification with linear-time algorithms.


## Direct-Sum Hilbert Spaces in Action

Direct sums are not just bookkeeping — they are how we build complex Hilbert spaces from simple ones, and they make many constructions transparent.

### The Hardy-Hilbert space and its decomposition

The Hardy space $H^2$ on the unit disk is the closed subspace of $L^2$ on the unit circle consisting of functions whose negative Fourier coefficients vanish. There is a canonical orthogonal decomposition $L^2(\partial \mathbb{D}) = H^2 \oplus \overline{H^2_0}$ where $\overline{H^2_0}$ is the conjugate of the Hardy space with the constant term removed. The projection onto $H^2$ is the **Riesz projection**, defined on Fourier series by truncating to non-negative indices.

The Riesz projection is bounded on $L^2$ — it is just the orthogonal projection onto a closed subspace, hence has norm $1$. On $L^p$ for $p \neq 2$, the boundedness is the M. Riesz theorem, with norms growing like $1/\sin(\pi/p)$. The proof requires complex analysis and is one of the classical results of harmonic analysis on the circle. The Hilbert-space case is, by contrast, a transparent application of orthogonal decomposition.

### Operator block decomposition

Given a closed subspace $M \subseteq \mathcal{H}$ with $\mathcal{H} = M \oplus M^\perp$, every bounded operator $T \in B(\mathcal{H})$ can be written as a $2 \times 2$ block matrix
$$T = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$$
where $A: M \to M$, $B: M^\perp \to M$, $C: M \to M^\perp$, $D: M^\perp \to M^\perp$. The operator $T$ leaves $M$ invariant iff $C = 0$, and is reduced by $M$ (leaves $M$ and $M^\perp$ both invariant) iff $B = C = 0$.

This block-diagonal decomposition is the basis of the spectral theorem for normal operators: every normal $T$ is unitarily equivalent to a direct integral of multiplication operators on a measure space. Concretely, $T$ acts on each spectral subspace of itself, and the decomposition into spectral subspaces gives a block diagonal form.

## Adjoints in Hilbert Space: The Cleanest Adjoint

For a bounded operator $T: \mathcal{H} \to \mathcal{K}$ between Hilbert spaces, the **adjoint** $T^*: \mathcal{K} \to \mathcal{H}$ is the unique bounded operator with $\langle T x, y \rangle_{\mathcal{K}} = \langle x, T^* y \rangle_{\mathcal{H}}$ for all $x \in \mathcal{H}, y \in \mathcal{K}$. Existence: for fixed $y$, the map $x \mapsto \langle T x, y \rangle$ is a bounded linear functional, so by Riesz it equals $\langle x, z \rangle$ for a unique $z \in \mathcal{H}$, and we set $T^* y = z$.

The Hilbert-space adjoint differs from the Banach-space dual operator (Article 4) only by the implicit Riesz isomorphism that identifies $\mathcal{H}^* = \mathcal{H}$. So the Hilbert-space adjoint is always defined on the *same kind* of space as the operator, while the Banach-space adjoint is defined between dual spaces. This makes the Hilbert calculus considerably cleaner.

### Properties

- $\|T^*\| = \|T\|$ (compute both as suprema of $|\langle T x, y \rangle|$).
- $\|T^* T\| = \|T\|^2$ — the **C\*-identity**, which makes $B(\mathcal{H})$ a $C^*$-algebra.
- $T^{**} = T$.
- $(\alpha S + \beta T)^* = \bar\alpha S^* + \bar\beta T^*$, $(S T)^* = T^* S^*$.
- $\ker(T^*) = \mathrm{Range}(T)^\perp$, so $\overline{\mathrm{Range}(T)} = \ker(T^*)^\perp$.

The C\*-identity is the structural property that distinguishes $B(\mathcal{H})$ from a generic Banach algebra. It is the input to the Gelfand-Naimark theorem (every commutative C\*-algebra is $C(K)$ for some compact Hausdorff space $K$) and the spectral theorem for normal operators (Article 8).

### Special operators

Defined via the adjoint:

- **Self-adjoint** (or Hermitian): $T^* = T$. Spectrum is real. Models observables in quantum mechanics.
- **Skew-adjoint** (or anti-Hermitian): $T^* = -T$. Spectrum is purely imaginary. Generators of unitary one-parameter groups (Article 10).
- **Normal**: $T^* T = T T^*$. Spectrum is a compact subset of $\mathbb{C}$. Includes self-adjoint and unitary operators.
- **Unitary**: $T^* T = T T^* = I$. Spectrum lies on the unit circle. Hilbert-space isometries that are surjective.
- **Positive**: $T = T^*$ and $\langle T x, x \rangle \geq 0$ for all $x$. Spectrum lies in $[0, \infty)$. Has a unique positive square root.
- **Projection**: $P^2 = P$ and $P^* = P$. Spectrum $\subseteq \{0, 1\}$. Geometric: orthogonal projection onto a closed subspace.

The classification of operators by adjoint properties is the analogue of the matrix classification (symmetric, orthogonal, normal, etc.) and most of the theorems carry over with the same intuition.

### Numerical example

The Volterra integral operator $V: L^2[0, 1] \to L^2[0, 1]$ defined by $V f(t) = \int_0^t f(s)\,ds$ has adjoint $V^* g(s) = \int_s^1 g(t)\,dt$, computed by Fubini: $\langle V f, g \rangle = \int_0^1 \int_0^t f(s)\,ds\,g(t)\,dt = \int_0^1 f(s) \int_s^1 g(t)\,dt\,ds = \langle f, V^* g \rangle$. The operator $V$ is *not* self-adjoint ($V \neq V^*$), and its spectrum turns out to be $\{0\}$ — the Volterra operator is a quasinilpotent.

The product $V^* V$ is self-adjoint and positive, with spectrum $[0, 4/\pi^2]$ (the eigenvalues coming from $\sin((n + 1/2)\pi t)$). The square root $(V^* V)^{1/2}$ — which exists by spectral calculus — is an explicit positive operator, and the polar decomposition $V = U |V|$ with $|V| = (V^* V)^{1/2}$ and $U$ a partial isometry exhibits Volterra as the product of a positive operator and an isometry. This polar-decomposition structure generalizes the matrix singular-value decomposition to Hilbert space.


## Looking Ahead

Hilbert spaces give us a calculus where geometry imports faithfully from $\mathbb{R}^n$. The unit ball is round, projections are linear, the dual is the space itself, and any continuous functional comes from an inner product. Every separable Hilbert space is isometrically $\ell^2$.

Most of analysis is not Hilbert-space analysis. The scattering of $L^p$ spaces for $p \neq 2$, the spaces of continuous functions, the dual spaces of measures — all are Banach but not Hilbert. The next article confronts the question: what is a continuous linear functional on a generic Banach space, and how do we know enough of them exist? The answer is the Hahn-Banach theorem, the most-used theorem of functional analysis after Riesz.

In the next article, we turn to **dual spaces and the Hahn-Banach theorem**. We will leave the comfort of Hilbert spaces (where every functional is an inner product) and confront the general Banach space setting, where the existence of non-trivial continuous linear functionals is far from obvious. The Hahn-Banach theorem resolves this by showing that functionals can always be extended from subspaces to the whole space — a result that launched modern duality theory.

---

*This is Part 3 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 2 — Normed and Banach Spaces](/en/functional-analysis/02-normed-and-banach/)*

*Next: [Part 4 — Dual Spaces and Hahn-Banach](/en/functional-analysis/04-dual-spaces-hahn-banach/)*
