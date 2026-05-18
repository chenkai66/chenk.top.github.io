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

## Inner Products and the Geometry They Create


![Orthogonal projection in Hilbert space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fa03_projection.png)

![Inner product geometry: angle and projection](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/03_inner_product.png)


If a Banach space is a normed space that has agreed to be complete, a Hilbert space is a Banach space that has further agreed to admit angles. That extra agreement — an inner product — is what restores almost all of finite-dimensional geometry to the infinite-dimensional setting. Orthogonality, projection, the Pythagorean theorem, the notion of "closest point in a subspace" — all come back unchanged. The price of admission is a single axiom; the reward, geometric and computational, is enormous.

Let $\mathcal{H}$ be a vector space over $\mathbb{C}$. An **inner product** is a function $\langle \cdot, \cdot \rangle : \mathcal{H} \times \mathcal{H} \to \mathbb{C}$ satisfying for all $x, y, z \in \mathcal{H}$ and $\alpha \in \mathbb{C}$:

1. $\langle x, x \rangle \geq 0$, with equality iff $x = 0$ (positive definiteness).
2. $\langle x, y \rangle = \overline{\langle y, x \rangle}$ (conjugate symmetry).
3. $\langle \alpha x + z, y \rangle = \alpha \langle x, y \rangle + \langle z, y \rangle$ (linearity in the first argument).

Conjugate symmetry forces conjugate-linearity in the second argument: $\langle x, \alpha y \rangle = \overline{\alpha} \langle x, y \rangle$. Some authors put linearity in the second slot instead; the choice is a convention, not a theorem, and I follow the physics convention of first-slot linearity. The inner product induces a norm by $\|x\| = \langle x, x \rangle^{1/2}$, and the triangle inequality follows from the Cauchy-Schwarz inequality, which is the single most important inequality in all of Hilbert-space theory.

**Cauchy-Schwarz inequality.** For all $x, y \in \mathcal{H}$, $|\langle x, y \rangle| \leq \|x\| \|y\|$, with equality iff $x$ and $y$ are linearly dependent.

*Proof.* If $y = 0$, both sides vanish. Otherwise, set $\lambda = \langle x, y \rangle / \|y\|^2$ and expand $0 \leq \|x - \lambda y\|^2 = \|x\|^2 - |\langle x, y \rangle|^2 / \|y\|^2$. Rearranging gives the inequality. The geometric insight: we are projecting $x$ onto the line through $y$ and observing that the residual $x - \lambda y$ is orthogonal to $y$, hence has non-negative norm-squared. Equality holds precisely when the residual vanishes, i.e., $x$ lies on the line through $y$. $\square$

From Cauchy-Schwarz, the triangle inequality follows: $\|x + y\|^2 = \|x\|^2 + 2\text{Re}\langle x,y\rangle + \|y\|^2 \leq \|x\|^2 + 2\|x\|\|y\| + \|y\|^2 = (\|x\| + \|y\|)^2$. So every inner product space is a normed space, and a **Hilbert space** is an inner-product space that is complete in the induced norm.

The examples are the workhorses of analysis. The space $\ell^2$ consists of sequences $(x_n)$ with $\sum |x_n|^2 < \infty$, equipped with $\langle x, y \rangle = \sum x_n \overline{y_n}$. The space $L^2(\Omega, \mu)$ consists of (equivalence classes of) square-integrable functions with $\langle f, g \rangle = \int f \overline{g}\, d\mu$; this is where quantum mechanics, Fourier analysis, and PDE theory all live. The Sobolev space $H^1(\Omega) = W^{1,2}(\Omega)$ has $\langle f, g \rangle_{H^1} = \int (f\overline{g} + \nabla f \cdot \overline{\nabla g})$, encoding both function values and derivatives. The Hardy space $H^2(\mathbb{D})$ of holomorphic functions on the unit disk with square-summable Taylor coefficients rounds out the standard collection.

![Inner product geometry: angle and length](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/03-hilbert-spaces/fa_v2_03_1_inner_product.png)

**Worked example.** In $L^2[0,1]$, take $f(t) = 1$ and $g(t) = t$. Then $\|f\|^2 = 1$, $\|g\|^2 = \int_0^1 t^2\,dt = 1/3$, and $\langle f, g \rangle = \int_0^1 t\,dt = 1/2$. Cauchy-Schwarz: $|\langle f, g\rangle| = 1/2 \leq \|f\|\cdot\|g\| = 1/\sqrt{3} \approx 0.577$, confirmed. The angle between $f$ and $g$ satisfies $\cos\theta = \frac{\langle f,g\rangle}{\|f\|\|g\|} = \frac{1/2}{1/\sqrt{3}} = \frac{\sqrt{3}}{2}$, giving $\theta = \pi/6$. The constant function and the identity function meet at 30 degrees in $L^2$ geometry. This is not a metaphor or a loose analogy — once we have an inner product, angles in function spaces are as real and computable as angles in $\mathbb{R}^3$.

The **parallelogram law** $\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2$ characterizes inner-product norms: a norm satisfies this identity if and only if it comes from an inner product (the Jordan-von Neumann theorem, 1935). The inner product is recovered by the **polarization identity** $\langle x, y \rangle = \tfrac{1}{4}(\|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2)$. A quick diagnostic: in $\ell^p_2$ with the standard basis vectors $x = (1,0)$, $y = (0,1)$, the parallelogram law reads $2 \cdot 2^{2/p} = 4$, forcing $p = 2$. So among the $\ell^p$ spaces, only $\ell^2$ is Hilbert. The entire $L^p$ family for $p \neq 2$ is stuck being merely Banach — complete normed spaces, but without the geometric richness of angles and orthogonality.

Why does this matter for applications? Linear regression minimizes $\|y - X\beta\|^2$ in a Hilbert space — that is an orthogonal projection. Fourier series expand functions in an orthonormal basis of $L^2$ — that is Hilbert-space geometry. Quantum mechanics postulates that states live in a Hilbert space and observables are self-adjoint operators. Signal processing decomposes signals into frequency components — orthogonal decomposition in $L^2$. The energy method in PDE uses the $H^1$ inner product to extract a priori estimates. Kernel methods in machine learning operate in a reproducing kernel Hilbert space. Every one of these is a Hilbert-space argument wearing domain-specific clothing. The inner product gives them a single grammar.


## Orthogonality, Projections, and the Closest-Point Property

Two vectors $x, y \in \mathcal{H}$ are **orthogonal**, written $x \perp y$, if $\langle x, y \rangle = 0$. The Pythagorean theorem carries over unchanged: if $x \perp y$, then $\|x + y\|^2 = \|x\|^2 + \|y\|^2$. For $n$ pairwise orthogonal vectors, $\|\sum_{k=1}^n x_k\|^2 = \sum_{k=1}^n \|x_k\|^2$. This extends to convergent infinite sums: if $(x_k)$ are pairwise orthogonal and $\sum \|x_k\|^2 < \infty$, the series $\sum x_k$ converges and its norm-squared equals the sum of the individual norm-squares. Infinite-dimensional Pythagoras is not a generalization one has to prove carefully from scratch — it follows from the finite case by continuity of the norm and completeness.

![Projection theorem: closest point in subspace](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/03_orthogonal_projection.png)


The geometric heart of Hilbert space theory is the **projection theorem**: for any closed subspace $M \subseteq \mathcal{H}$ and any $x \in \mathcal{H}$, there exists a unique $m_0 \in M$ minimizing $\|x - m\|$ over $m \in M$, and the minimizer is characterized by the orthogonality condition $(x - m_0) \perp M$. The space decomposes as $\mathcal{H} = M \oplus M^\perp$ where $M^\perp = \{y \in \mathcal{H} : \langle y, m \rangle = 0 \text{ for all } m \in M\}$.

*Why the closest-point property works — and why it needs completeness.* Take a minimizing sequence $(m_n)$ with $d_n = \|x - m_n\| \to d = \inf_{m \in M}\|x - m\|$. Apply the parallelogram law to $u = x - m_n$ and $v = x - m_k$: we get $\|u + v\|^2 + \|u - v\|^2 = 2\|u\|^2 + 2\|v\|^2$. Since $(m_n + m_k)/2 \in M$ (convexity), $\|u + v\|/2 = \|x - (m_n+m_k)/2\| \geq d$. Substituting: $\|m_n - m_k\|^2 = 2d_n^2 + 2d_k^2 - 4\|x - (m_n+m_k)/2\|^2 \leq 2d_n^2 + 2d_k^2 - 4d^2 \to 0$. So $(m_n)$ is Cauchy, hence converges in $M$ (completeness). Without the parallelogram law — in a Banach space that is not Hilbert — this argument collapses, and indeed closest points in closed subspaces may fail to be unique.

The **orthogonal projection** $P_M : \mathcal{H} \to M$ maps each $x$ to its closest point in $M$. It satisfies $P_M^2 = P_M$ (projecting twice is the same as projecting once), $P_M^* = P_M$ (the projection is self-adjoint), $\|P_M\| \leq 1$, and $\text{Range}(P_M) = M$. Conversely, every bounded self-adjoint idempotent is the orthogonal projection onto its range. This bijection between closed subspaces and self-adjoint idempotents is one of the structural pillars of operator theory on Hilbert spaces.

**Worked example: Fourier approximation as projection.** In $L^2[-\pi, \pi]$, let $M_N = \text{span}\{e^{int} : |n| \leq N\}$ — the trigonometric polynomials of degree at most $N$. The orthogonal projection of $f$ onto $M_N$ is the $N$-th partial sum of the Fourier series: $P_{M_N} f = \sum_{|n| \leq N} \hat{f}(n) e^{int}$ where $\hat{f}(n) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(t) e^{-int}\,dt$. The error $f - P_{M_N} f$ is orthogonal to every $e^{int}$ with $|n| \leq N$ — this is precisely the statement that the Fourier coefficients of the error vanish for $|n| \leq N$. The $L^2$ approximation theory of Fourier series is nothing but the projection theorem applied to nested subspaces $M_1 \subset M_2 \subset \cdots$ whose union is dense in $L^2$.

Take a concrete function: $f(t) = |t|$ on $[-\pi, \pi]$. Its Fourier coefficients are $\hat{f}(0) = \pi/2$ and $\hat{f}(n) = -\frac{2}{\pi n^2}$ for odd $n$, zero for even $n \neq 0$. The projection onto $M_1$ gives $P_{M_1}f = \frac{\pi}{2} - \frac{2}{\pi}\cos t$ — the best trigonometric polynomial of degree 1 in the $L^2$ sense. The error norm satisfies $\|f - P_{M_1}f\|^2 = \|f\|^2 - |c_0|^2 - 2|c_1|^2 = \frac{\pi^2}{3} - \frac{\pi^2}{4} - \frac{8}{\pi^2} \approx 0.81 - 0.81 = 0.014$ (approximately). The projection theorem guarantees this is the smallest possible $L^2$ error among all linear combinations of $1$, $\cos t$, $\sin t$.

The projection theorem fails in Banach spaces that are not Hilbert. In $C[0,1]$ with the sup norm, the best polynomial approximation (Chebyshev approximation) exists and is unique, but it is characterized by an equioscillation condition, not by orthogonality — there is no inner product. In $L^1$, closest points in closed subspaces may not be unique at all. The projection theorem is genuinely a Hilbert-space luxury, and it is the single feature that makes Hilbert spaces the analyst's paradise.


## Orthonormal Bases, Parseval, and the Classification Theorem

An **orthonormal system** in $\mathcal{H}$ is a collection $\{e_\alpha\}_{\alpha \in A}$ with $\langle e_\alpha, e_\beta \rangle = \delta_{\alpha\beta}$. It is a **basis** (complete orthonormal system) if its closed linear span equals $\mathcal{H}$, equivalently if $\langle x, e_\alpha \rangle = 0$ for all $\alpha$ forces $x = 0$. The cardinality of a maximal orthonormal system is an invariant called the **Hilbert dimension** of the space.

![Animation: Fourier series building up](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/03_fourier_approx.gif)


![Fourier series in L^2: orthonormal basis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/03_fourier_basis.png)


For **separable** Hilbert spaces — those with a countable dense subset — the Hilbert dimension is countable, and we index the basis as $(e_n)_{n=1}^\infty$. The foundational classification theorem says: every separable infinite-dimensional Hilbert space is isometrically isomorphic to $\ell^2$. There is essentially one separable Hilbert space, and every calculation in $L^2$, in Sobolev spaces, in Hardy spaces, is secretly a calculation in $\ell^2$ once a basis is chosen. The space is rigid; all the interesting mathematics lives in the operators acting on it.

**Bessel's inequality and Parseval's identity.** For any $x \in \mathcal{H}$ and orthonormal system $(e_n)$, the Fourier coefficients $c_n = \langle x, e_n \rangle$ satisfy Bessel's inequality $\sum |c_n|^2 \leq \|x\|^2$. When $(e_n)$ is a basis, equality holds — this is **Parseval's identity**: $\|x\|^2 = \sum_{n=1}^\infty |c_n|^2$, and the expansion $x = \sum c_n e_n$ converges in norm. Parseval is the infinite-dimensional Pythagorean theorem: the norm-squared of a vector equals the sum of squares of its coordinates.

The convergence of the Fourier expansion $x = \sum \langle x, e_n \rangle e_n$ is unconditional — it converges regardless of the ordering of terms. This is a consequence of Bessel's inequality: since $\sum |c_n|^2 < \infty$, for any $\varepsilon > 0$, all but finitely many terms have $|c_n|^2 < \varepsilon$. The partial sums form a Cauchy net, and completeness gives convergence.

The **Gram-Schmidt process** constructs orthonormal systems from linearly independent sets. Starting from $\{v_1, v_2, \ldots\}$, set $e_1 = v_1/\|v_1\|$ and iteratively $e_n = (v_n - \sum_{k<n}\langle v_n, e_k\rangle e_k)/\|...\|$. The process preserves the span at each step: $\text{span}\{e_1, \ldots, e_n\} = \text{span}\{v_1, \ldots, v_n\}$.

**Worked example: classical orthonormal bases.** The monomials $\{1, t, t^2, \ldots\}$ in $L^2[-1,1]$ are linearly independent but not orthogonal. Gram-Schmidt produces (up to normalization) the Legendre polynomials: $P_0 = 1/\sqrt{2}$, $P_1 = t\sqrt{3/2}$, $P_2 = (3t^2-1)\sqrt{5/8}$. The exponentials $\{e^{int}/(2\pi)^{1/2}\}_{n \in \mathbb{Z}}$ form the Fourier basis of $L^2[-\pi,\pi]$. The Hermite functions $h_n(x) = c_n H_n(x) e^{-x^2/2}$ form an orthonormal basis of $L^2(\mathbb{R})$ — they are the eigenfunctions of the quantum harmonic oscillator. Each basis is tailored to a problem: Legendre for polynomial approximation on intervals, Fourier for periodic phenomena, Hermite for the Schrodinger equation with quadratic potential.

**Bessel's inequality gives a practical test for completeness.** If $(e_n)$ is an orthonormal system and we want to verify it is a basis, we check whether Parseval's identity holds for all $x$ — equivalently, whether $\|x - \sum_{n=1}^N \langle x, e_n\rangle e_n\| \to 0$ for all $x$. For the Fourier system on $L^2[-\pi,\pi]$, this is exactly the statement that the Fourier series of every $L^2$ function converges to it in $L^2$ norm — a theorem proved by Riesz and Fischer in 1907, which was one of the early triumphs of Lebesgue integration theory.

The convergence need not be pointwise — there exist $L^2$ functions whose Fourier series diverge at some points (Carleson's theorem gives pointwise a.e. convergence, but that is a much harder result). The norm convergence in $L^2$ is a different and weaker statement: it says the $L^2$ norm of the difference between $f$ and its $N$-th partial sum tends to zero. This is exactly what Parseval gives us, and it is all that the Hilbert-space theory guarantees.

A non-trivial consequence of separability: $L^2[0,1]$ and $\ell^2$ are isometrically isomorphic. Any orthonormal basis $(e_n)$ of $L^2[0,1]$ defines an isomorphism $U: L^2 \to \ell^2$ by $Uf = (\langle f, e_n\rangle)_{n=1}^\infty$. The map $U$ preserves the inner product (by Parseval) and is surjective (given $(c_n) \in \ell^2$, the series $\sum c_n e_n$ converges in $L^2$). So the Fourier basis, the Legendre basis, and the Haar wavelet basis all give different "coordinates" on the same abstract Hilbert space. The choice of basis is a modeling decision, not a mathematical one.


## The Riesz Representation Theorem and Its Consequences

The most structurally important theorem about Hilbert spaces is the **Riesz representation theorem** (Riesz-Frechet): every continuous linear functional $\varphi : \mathcal{H} \to \mathbb{C}$ has the form $\varphi(x) = \langle x, y_\varphi \rangle$ for a unique $y_\varphi \in \mathcal{H}$, with $\|\varphi\| = \|y_\varphi\|$.

![Riesz representation theorem](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/03_riesz_representation.png)


*Proof.* If $\varphi = 0$, take $y_\varphi = 0$. Otherwise, $M = \ker(\varphi)$ is a closed hyperplane (codimension 1). Its orthogonal complement $M^\perp$ is one-dimensional; pick $z \in M^\perp$ with $\varphi(z) = 1$ (normalizing). For any $x \in \mathcal{H}$, write $x = (x - \varphi(x) z) + \varphi(x) z$. The first term is in $M$ (check: $\varphi(x - \varphi(x)z) = \varphi(x) - \varphi(x) = 0$). So $\langle x, z \rangle = \langle \varphi(x) z, z \rangle = \varphi(x)\|z\|^2$, giving $\varphi(x) = \langle x, z/\|z\|^2 \rangle$. Set $y_\varphi = z/\|z\|^2$. Uniqueness: if $\langle x, y_1 \rangle = \langle x, y_2 \rangle$ for all $x$, then $\langle x, y_1 - y_2 \rangle = 0$ for all $x$, so $y_1 = y_2$ (take $x = y_1 - y_2$). $\square$

The theorem establishes a conjugate-linear isometric isomorphism $\mathcal{H}^* \cong \mathcal{H}$. Hilbert spaces are **self-dual**: the dual space is (conjugate-linearly) the space itself. This is a dramatic simplification compared to general Banach spaces, where duals can be unrecognizable: $(\ell^1)^* = \ell^\infty$, $(c_0)^* = \ell^1$, $(L^1)^* = L^\infty$. In Hilbert space, there is no conceptual gap between vectors and linear functionals — the inner product identifies them.

Self-duality immediately gives **reflexivity**: the canonical embedding $J: \mathcal{H} \to \mathcal{H}^{**}$ is surjective. Reflexivity is the prerequisite for weak compactness of the unit ball (Article 5), which in turn is the engine of variational methods. So the chain is: inner product $\Rightarrow$ self-duality $\Rightarrow$ reflexivity $\Rightarrow$ weak compactness $\Rightarrow$ existence of minimizers for energy functionals.

**Application: the Lax-Milgram theorem.** Let $a: \mathcal{H} \times \mathcal{H} \to \mathbb{C}$ be a bounded sesquilinear form ($|a(u,v)| \leq M\|u\|\|v\|$) that is coercive ($\text{Re}\,a(u,u) \geq \alpha\|u\|^2$ for some $\alpha > 0$). Then for every $F \in \mathcal{H}^*$, there exists a unique $u \in \mathcal{H}$ with $a(u,v) = F(v)$ for all $v$, and $\|u\| \leq \|F\|/\alpha$.

*Proof sketch.* For fixed $u$, the map $v \mapsto a(u,v)$ is a bounded linear functional, so by Riesz it equals $\langle Au, v \rangle$ for a unique bounded operator $A$. Coercivity gives $\alpha\|u\|^2 \leq \text{Re}\langle Au, u\rangle \leq \|Au\|\|u\|$, so $\|Au\| \geq \alpha\|u\|$ — the operator $A$ is bounded below. Bounded below plus dense range (from coercivity again, by a standard argument) gives invertibility. Then $u = A^{-1}y_F$ where $F(\cdot) = \langle \cdot, y_F \rangle$ by Riesz.

**Worked example: the Dirichlet problem.** Consider $-\Delta u = f$ on a bounded domain $\Omega$ with $u|_{\partial\Omega} = 0$. The weak formulation: find $u \in H^1_0(\Omega)$ with $a(u,v) = \int_\Omega \nabla u \cdot \nabla v = \int_\Omega f v = F(v)$ for all $v \in H^1_0$. The form $a$ is bounded (Cauchy-Schwarz on gradients) and coercive (Poincare inequality: $\|\nabla u\|_{L^2}^2 \geq C\|u\|_{H^1}^2$ on $H^1_0$). Lax-Milgram gives unique existence. The PDE existence theorem is a one-line corollary of Riesz representation. This is the standard pattern in elliptic PDE: formulate weakly, verify coercivity, invoke Lax-Milgram. The hard work is in the Poincare inequality and the function-space setup, not in the abstract existence argument.


## Adjoint Operators and the Algebra $B(\mathcal{H})$

For a bounded operator $T: \mathcal{H} \to \mathcal{H}$, the **adjoint** $T^*$ is defined by $\langle Tx, y \rangle = \langle x, T^*y \rangle$ for all $x, y$. Existence: for fixed $y$, the functional $x \mapsto \langle Tx, y \rangle$ is bounded (by $\|T\|\|y\|$), so by Riesz it equals $\langle x, z \rangle$ for unique $z$; define $T^*y = z$. The assignment $y \mapsto T^*y$ is linear, and $\|T^*\| = \|T\|$.

![Orthogonal decomposition H = M + M^perp](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/03_orthogonal_complement.png)


The fundamental structural identity is the **$C^*$-identity**: $\|T^*T\| = \|T\|^2$. This makes the algebra $B(\mathcal{H})$ of bounded operators a $C^*$-algebra — the starting point for the abstract spectral theory of Article 8 and the Gelfand-Naimark theorem.

The taxonomy of operators, classified by their relationship to the adjoint, mirrors the finite-dimensional matrix classification:

- **Self-adjoint** ($T = T^*$): spectrum is real, eigenvectors for distinct eigenvalues are orthogonal. Models observables in quantum mechanics. Examples: multiplication operators $M_f$ with real-valued $f$; the Laplacian $-\Delta$ on suitable domains.
- **Unitary** ($T^*T = TT^* = I$): isometric isomorphisms, spectrum on the unit circle. The Fourier transform $\mathcal{F}: L^2(\mathbb{R}) \to L^2(\mathbb{R})$. Time-evolution operators $e^{-iHt}$ in quantum mechanics.
- **Normal** ($T^*T = TT^*$): the maximal class admitting a spectral decomposition. Includes self-adjoint and unitary as special cases.
- **Positive** ($T = T^*$ and $\langle Tx, x\rangle \geq 0$): spectrum in $[0,\infty)$, has a unique positive square root $T^{1/2}$.
- **Projection** ($P = P^2 = P^*$): orthogonal projection onto a closed subspace. Spectrum $\subseteq \{0, 1\}$.

The identity $\ker(T^*) = \text{Range}(T)^\perp$ connects kernels and ranges. Taking orthogonal complements: $\overline{\text{Range}(T)} = \ker(T^*)^\perp$. This is the operator-theoretic rank-nullity theorem and is crucial for Fredholm theory (Article 7).

**Worked example: the shift operator.** The right shift $S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$ on $\ell^2$ has adjoint $S^*(x_1, x_2, \ldots) = (x_2, x_3, \ldots)$ (the left shift). Verification: $\langle Sx, y \rangle = \sum_{n \geq 2} x_{n-1}\overline{y_n} = \sum_{n \geq 1} x_n\overline{y_{n+1}} = \langle x, S^*y \rangle$. Now $S^*S = I$ (shift right then left recovers the original), but $SS^*x = (0, x_2, x_3, \ldots) \neq x$ in general. The shift is an isometry ($\|Sx\| = \|x\|$) but not unitary (not surjective). This is impossible in finite dimensions where an isometry of $\mathbb{C}^n$ to itself is automatically surjective. The shift is the canonical example witnessing the failure of this implication in infinite dimensions.

**Worked example: the Volterra operator.** Define $Vf(t) = \int_0^t f(s)\,ds$ on $L^2[0,1]$. Computing the adjoint via Fubini: $\langle Vf, g\rangle = \int_0^1 g(t)\int_0^t f(s)\,ds\,dt = \int_0^1 f(s)\int_s^1 g(t)\,dt\,ds = \langle f, V^*g\rangle$, so $V^*g(s) = \int_s^1 g(t)\,dt$. The operator $V$ is not self-adjoint ($V \neq V^*$), not normal, and has spectrum $\{0\}$ — it is quasinilpotent. Yet $V \neq 0$. This shows that compact operators (Article 7) can have trivial spectrum without being zero, a phenomenon impossible for self-adjoint operators.

The **polar decomposition** $T = U|T|$ with $|T| = (T^*T)^{1/2}$ and $U$ a partial isometry extends the matrix SVD to infinite dimensions. For the Volterra operator, this gives a factorization into a positive part (capturing "how much" $V$ stretches) and an isometric part (capturing "in which direction").


## Weak Convergence, Tensor Products, and Direct Sums

A sequence $(x_n) \subset \mathcal{H}$ **converges weakly** to $x$, written $x_n \rightharpoonup x$, if $\langle x_n, y \rangle \to \langle x, y \rangle$ for every $y \in \mathcal{H}$. By Riesz, this is the same as convergence against all bounded functionals. Weak convergence is strictly weaker than norm convergence: the standard basis $(e_n) \subset \ell^2$ satisfies $e_n \rightharpoonup 0$ (since $\langle e_n, y \rangle = y_n \to 0$ for any $y \in \ell^2$, as $\sum|y_n|^2 < \infty$ forces $y_n \to 0$) but $\|e_n\| = 1 \not\to 0$.

![Bessel inequality and Parseval identity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/03_bessel_parseval.png)


The **Banach-Alaoglu theorem** in the Hilbert setting: every bounded sequence has a weakly convergent subsequence. This is the infinite-dimensional substitute for the Bolzano-Weierstrass theorem. The unit ball, which is not norm-compact, is weakly compact — and this weak compactness is the engine that drives variational methods. To find minimizers of energy functionals, one extracts a weakly convergent subsequence from a minimizing sequence and shows the functional is lower semicontinuous in the weak topology.

A useful fact specific to Hilbert spaces (the **Radon-Riesz property**): $x_n \rightharpoonup x$ and $\|x_n\| \to \|x\|$ together imply $\|x_n - x\| \to 0$. The proof is a one-liner: $\|x_n - x\|^2 = \|x_n\|^2 - 2\text{Re}\langle x_n, x\rangle + \|x\|^2 \to \|x\|^2 - 2\|x\|^2 + \|x\|^2 = 0$. This gives a clean criterion for upgrading weak convergence to strong convergence.

Another important example of weak convergence: in $L^2[0, 2\pi]$, the sequence $f_n(t) = \sin(nt)$ converges weakly to zero. The Riemann-Lebesgue lemma gives $\int_0^{2\pi} g(t)\sin(nt)\,dt \to 0$ for every $g \in L^2$. But $\|f_n\|_2 = \sqrt{\pi}$ for all $n$. The oscillations average out against every fixed test function, yet the energy stays constant — this is what "weak convergence without strong convergence" looks like physically. Rapid oscillations are invisible to weak topology; only their amplitude envelope matters.

The inner product is jointly continuous in the norm topology: $x_n \to x$ and $y_n \to y$ imply $\langle x_n, y_n\rangle \to \langle x, y\rangle$. But it is only *separately* continuous in the weak topology: $x_n \rightharpoonup x$ gives $\langle x_n, y\rangle \to \langle x, y\rangle$ for fixed $y$, but if also $y_n \rightharpoonup y$, the products $\langle x_n, y_n\rangle$ need not converge to $\langle x, y\rangle$. Counterexample: $x_n = y_n = e_n$ in $\ell^2$ gives $\langle e_n, e_n\rangle = 1$ but $\langle 0, 0\rangle = 0$. This failure of joint weak continuity is a recurrent source of difficulty in nonlinear PDE — passing to the limit in products of weakly convergent sequences requires additional compactness or compensated-compactness arguments.

**Direct sums.** The orthogonal direct sum $\mathcal{H}_1 \oplus \mathcal{H}_2$ has elements $(x_1, x_2)$ with inner product $\langle (x_1,x_2), (y_1,y_2)\rangle = \langle x_1,y_1\rangle_1 + \langle x_2,y_2\rangle_2$. The decomposition $\mathcal{H} = M \oplus M^\perp$ is the prototypical example. Block-matrix representations of operators relative to direct-sum decompositions are the foundation of spectral theory: a normal operator is "diagonalized" when expressed as a direct sum of multiplication operators on spectral subspaces.

**Tensor products.** The completed tensor product $\mathcal{H}_1 \otimes \mathcal{H}_2$ is built by defining $\langle x_1 \otimes x_2, y_1 \otimes y_2\rangle = \langle x_1, y_1\rangle\langle x_2, y_2\rangle$ on simple tensors and extending by linearity and completion. The fundamental identifications: $\ell^2 \otimes \ell^2 \cong \ell^2(\mathbb{N} \times \mathbb{N})$ and $L^2(\Omega_1) \otimes L^2(\Omega_2) \cong L^2(\Omega_1 \times \Omega_2)$. In quantum mechanics, the state space of a bipartite system is $\mathcal{H}_A \otimes \mathcal{H}_B$; entanglement means precisely that a state cannot be written as a simple tensor $\psi_A \otimes \psi_B$ but requires a genuine sum $\sum c_k \psi_A^{(k)} \otimes \psi_B^{(k)}$. The tensor product structure is also why multivariate Fourier analysis reduces to iterated univariate transforms: the Fourier transform on $L^2(\mathbb{R}^d) \cong L^2(\mathbb{R})^{\otimes d}$ is the tensor product of $d$ copies of the one-dimensional transform.

**Worked example.** In $L^2(\mathbb{R}^2)$, the Gaussian $f(x,y) = e^{-(x^2+y^2)/2}$ factors as $f_1 \otimes f_2$ with $f_1(x) = f_2(x) = e^{-x^2/2}$. Its $L^2(\mathbb{R}^2)$ norm squared is $\int e^{-(x^2+y^2)}\,dx\,dy = \pi$. The tensor norm gives $\|f_1\|^2 \cdot \|f_2\|^2 = \sqrt{\pi} \cdot \sqrt{\pi} = \pi$. The function $g(x,y) = xe^{-(x^2+y^2)/2}$ does not factor — it requires the expansion $g = (\text{something in }x) \otimes (\text{something in }y)$ which is impossible since $g(x,y)/g(x',y)$ depends on $y$ in general. Non-factorizable functions in $L^2(\mathbb{R}^2)$ correspond to "entangled" states in the quantum interpretation.


## Reproducing Kernel Hilbert Spaces

A **reproducing kernel Hilbert space** (RKHS) on a set $\Omega$ is a Hilbert space $\mathcal{H}$ of functions $f: \Omega \to \mathbb{C}$ in which point evaluation is a bounded functional: $|f(x)| \leq C_x \|f\|_{\mathcal{H}}$ for every $x \in \Omega$. By Riesz, each evaluation functional $\delta_x$ is represented by an element $K_x \in \mathcal{H}$: $f(x) = \langle f, K_x\rangle$. The two-variable function $K(x,y) = K_x(y) = \langle K_x, K_y\rangle$ is the **reproducing kernel**.

Not every Hilbert space of functions is an RKHS. The space $L^2[0,1]$ fails: its elements are equivalence classes modulo null sets, so $f(x)$ is undefined for a single point $x$. But $H^1[0,1]$ (the Sobolev space on a bounded interval) is an RKHS: by the Sobolev embedding theorem, $H^1[0,1] \hookrightarrow C[0,1]$ continuously, so point evaluation is bounded. Its kernel is related to Green's function for $-d^2/dx^2 + 1$.

The **Moore-Aronszajn theorem** provides the converse: every positive-definite kernel $K$ (meaning $\sum_{i,j} c_i\overline{c_j} K(x_i, x_j) \geq 0$ for all finite selections) determines a unique RKHS with reproducing kernel $K$. The construction: the RKHS is the completion of $\text{span}\{K_x : x \in \Omega\}$ in the inner product $\langle K_x, K_y\rangle = K(x,y)$.

The kernel trick in machine learning exploits this theory directly. Given a positive-definite kernel $K$, the feature map $\Phi: x \mapsto K_x$ embeds data into the RKHS. Inner products in the feature space are computed as $\langle \Phi(x), \Phi(y)\rangle = K(x,y)$ — no explicit construction of the (possibly infinite-dimensional) feature space needed. Support vector machines find a hyperplane in the RKHS, which corresponds to a nonlinear decision boundary in the original space. Gaussian process regression computes conditional expectations in an RKHS. Kernel PCA performs principal component analysis in the feature space. All are linear methods in a Hilbert space, made computationally tractable by the reproducing property.

**Worked example.** The Gaussian RBF kernel $K(x,y) = \exp(-\|x-y\|^2/(2\sigma^2))$ on $\mathbb{R}^d$ defines an infinite-dimensional RKHS of smooth functions. For the dataset $\{x_1, \ldots, x_n\} \subset \mathbb{R}^d$, the Gram matrix $G_{ij} = K(x_i, x_j)$ is positive-definite (as guaranteed by Moore-Aronszajn). An SVM trains by solving a quadratic program in terms of $G$ alone — never touching the infinite-dimensional feature space explicitly. The Hilbert-space geometry (projections, orthogonality, closest points) operates behind the scenes, coordinated by the $n \times n$ kernel matrix.

Another important example: the **Bargmann-Fock space** of entire functions $f(z) = \sum a_n z^n$ with $\|f\|^2 = \sum n! |a_n|^2 < \infty$, which has kernel $K(z,w) = e^{z\overline{w}}$. This space appears in quantum optics (coherent states), complex geometry (Bergman kernels), and random matrix theory (correlation functions of Gaussian analytic functions). The reproducing property $f(z) = \langle f, K_z\rangle$ with $K_z(w) = e^{\overline{z}w}$ gives a resolution of identity: $\frac{1}{\pi}\int |K_z\rangle\langle K_z| e^{-|z|^2}\,dA(z) = I$, the over-completeness relation for coherent states.


## The Spectral Preview and Why Operators Matter More Than Spaces

Since every separable Hilbert space is $\ell^2$, the space itself carries no information — it is a blank canvas. All the interesting mathematics lives in the operators painted on it. A self-adjoint operator $T$ on a finite-dimensional Hilbert space has an orthonormal basis of eigenvectors with real eigenvalues — the spectral theorem of linear algebra. In infinite dimensions, the eigenvalue/eigenvector picture breaks down for generic operators, but a residue survives: the spectrum.

The simplest infinite-dimensional example illustrating the breakdown: the multiplication operator $Mf(t) = tf(t)$ on $L^2[0,1]$. It is self-adjoint ($\langle Mf, g\rangle = \int tf\overline{g} = \langle f, Mg\rangle$) with spectrum $[0,1]$, but it has *no eigenvectors whatsoever*. The equation $tf(t) = \lambda f(t)$ forces $f(t) = 0$ for $t \neq \lambda$, so $f = 0$ in $L^2$. The spectrum is "purely continuous" — every point in $[0,1]$ is a spectral value but none is an eigenvalue. Article 8 will explain how to read this off from a spectral measure $E$ satisfying $M = \int_0^1 t\,dE(t)$, where $E([a,b])f = \mathbf{1}_{[a,b]}(t)f(t)$ projects onto functions supported in $[a,b]$.

For a compact self-adjoint operator, on the other hand, the spectrum behaves like a matrix's: it consists of eigenvalues accumulating only at zero, each with a finite-dimensional eigenspace, plus possibly zero itself. The integral operator $Kf(t) = \int_0^1 k(t,s)f(s)\,ds$ with a symmetric continuous kernel $k$ is compact and self-adjoint, and its spectral theorem gives $Kf = \sum \lambda_n \langle f, e_n\rangle e_n$ with $\lambda_n \to 0$ — the operator decomposes into a countable sum of rank-one projections, exactly like a diagonal matrix with entries tending to zero. Article 7 develops this in full.

The conceptual hierarchy, then: Article 3 (this one) sets up the stage. Articles 4-6 develop the tools (duality, weak topologies, bounded operators). Article 7 puts operators that "almost" live in finite dimensions (compact operators) through the spectral machine. Article 8 handles the general case, requiring measure-theoretic spectral theory. The space $\mathcal{H}$ stays the same throughout — it is always $\ell^2$ in disguise. The operators change, and their spectral properties encode the physics, the geometry, and the analysis.

A final remark on why this matters computationally. Every numerical method for PDE — finite elements, spectral methods, Galerkin approximations — is a projection method: approximate the solution by projecting it onto a finite-dimensional subspace $V_h \subset \mathcal{H}$. The error is $\|u - P_{V_h}u\|$, which by the projection theorem equals the distance from $u$ to $V_h$. The theory of approximation (Jackson theorems, Bramble-Hilbert lemma) estimates this distance in terms of the regularity of $u$ and the mesh size $h$. The entire convergence theory of finite elements is Hilbert-space geometry — projections, orthogonality, and the Cea lemma (which says the Galerkin solution is the quasi-optimal projection under coercivity). Once one sees finite-element theory as projection theory in $H^1$, the convergence rates stop being mysterious and become consequences of approximation properties of polynomial subspaces.


## What's Next

Hilbert spaces are the analyst's paradise — self-dual, reflexive, with a geometry that imports faithfully from finite dimensions. But most of analysis lives outside this paradise. The $L^p$ spaces for $p \neq 2$, the space of continuous functions, the space of measures — all are Banach but not Hilbert. The next article asks: on a general Banach space, how do we know enough continuous linear functionals exist? The answer is the Hahn-Banach theorem, the most-used theorem in functional analysis after Riesz.

---

*This is Part 3 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 2 — Normed and Banach Spaces](/en/functional-analysis/02-normed-and-banach/)*

*Next: [Part 4 — Dual Spaces and Hahn-Banach](/en/functional-analysis/04-dual-spaces-hahn-banach/)*
