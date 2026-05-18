---
title: "Functional Analysis (8): Spectral Theory — Decomposing Operators"
date: 2021-10-15 09:00:00
tags:
  - functional-analysis
  - spectral-theory
  - operators
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "The spectrum generalizes eigenvalues to infinite dimensions — the spectral theorem for bounded self-adjoint operators and continuous functional calculus give us a complete decomposition."
disableNunjucks: true
series_order: 8
series_total: 12
translationKey: "functional-analysis-8"
---

When I first saw the word "spectrum" used for an operator I assumed it was a fancy synonym for "set of eigenvalues." That is the right intuition for matrices and for compact operators, and it is exactly what one wants in introductory linear algebra. The trouble is that it is wrong as soon as the operator is not compact. The position operator $(Mf)(x) = x f(x)$ on $L^2[0, 1]$ has no eigenvalues: any eigenfunction would have to satisfy $x f(x) = \lambda f(x)$ a.e., which forces $f = 0$ everywhere away from a single point, hence $f = 0$ in $L^2$. And yet the operator is clearly not invertible, since $\lambda I - M$ is multiplication by $x - \lambda$, which fails to be boundedly invertible whenever $\lambda \in [0, 1]$.

So we need a notion of "spectral value" that is broader than eigenvalue. The idea is to define the spectrum as the set of $\lambda$ for which $\lambda I - T$ fails to be a bounded invertible operator, *for any reason*. This makes the spectrum a property of the operator's invertibility structure, not just its eigenvector structure, and it works uniformly for compact and non-compact operators alike. The full reward — the **spectral theorem for bounded self-adjoint operators** — promotes the diagonalization picture from compact to general self-adjoint operators by replacing finite or countable diagonals with continuous integrals against a *spectral measure*. This article is the unhurried walk through that reward.

## The Spectrum, Defined

Let $T \in B(X)$ be a bounded operator on a complex Banach space $X$. The **resolvent set** is

![Spectrum decomposition: point, continuous, residual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig08_spectrum_parts.png)

$$ \rho(T) = \{\lambda \in \mathbb{C} : \lambda I - T \text{ is bijective with bounded inverse}\}. $$

The **spectrum** is the complement: $\sigma(T) = \mathbb{C} \setminus \rho(T)$. By the open mapping theorem, $\lambda I - T$ being bijective is enough to guarantee bounded inverse, so the second condition is automatic. The spectrum is therefore the set of $\lambda$ for which $\lambda I - T$ fails to be either injective or surjective.

This binary distinction (injective / surjective) lets us refine the spectrum into pieces.

![Decomposition of the spectrum: point, continuous, residual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/08-spectral-theory/fa_v2_08_1_spectrum_decomp.png)

- **Point spectrum** $\sigma_p(T) = \{\lambda : \lambda I - T \text{ is not injective}\}$ — the eigenvalues, in the usual sense.
- **Continuous spectrum** $\sigma_c(T) = \{\lambda : \lambda I - T \text{ is injective with dense range, but not surjective}\}$ — no eigenvector, but $\lambda I - T$ is "almost" surjective.
- **Residual spectrum** $\sigma_r(T) = \{\lambda : \lambda I - T \text{ is injective, range not dense}\}$ — no eigenvector, and the range misses a substantial part of the space.

These three sets are disjoint and their union is $\sigma(T)$. For self-adjoint operators on Hilbert space, the residual spectrum is empty — a small but useful fact.

A few examples are worth concrete computation.

**Example 1 (matrix).** For $A \in M_n(\mathbb{C})$, the spectrum is the eigenvalues, all in $\sigma_p$. There is no continuous or residual spectrum: in finite dimensions, injective implies surjective. The whole "three pieces" story collapses to the matrix eigenvalue story.

**Example 2 (compact operator).** Article 7 told us that for compact $T$, every nonzero spectral value is an eigenvalue. So $\sigma(T) \setminus \{0\} \subset \sigma_p(T)$, with possibly $0 \in \sigma_c(T)$ or $\sigma_r(T)$. The Volterra operator on $L^2[0, 1]$ has $\sigma = \{0\}$, with $0 \in \sigma_c$.

**Example 3 (multiplication on $L^2[0, 1]$).** $(M f)(x) = x f(x)$. For $\lambda \in [0, 1]$, $\lambda I - M$ is multiplication by $x - \lambda$, which is injective (since $x - \lambda \neq 0$ except on a null set) but not surjective (the image is $\{g \in L^2 : g(x)/(x - \lambda) \in L^2\}$, a proper subset). For $\lambda \in [0, 1]$ in the interior, the image is dense, so $\lambda \in \sigma_c$. For $\lambda \notin [0, 1]$, $\lambda I - M$ is invertible (multiplication by $1/(x-\lambda) \in L^\infty$), so $\lambda \in \rho$. Conclusion: $\sigma(M) = [0, 1]$, all continuous spectrum, no eigenvalues.

The third example is the prototype non-compact self-adjoint operator. It shows in stark form why "find the eigenvalues" is the wrong question for general operators: this operator has no eigenvalues, but its spectrum is a whole interval. The right question is "describe the spectral measure," and the answer for $M$ is "Lebesgue measure on $[0, 1]$, in disguise."

## The Resolvent and Its Analyticity

For $\lambda \in \rho(T)$, define the **resolvent**

![Resolvent function and its analytic properties](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig08_resolvent.png)

$$ R(\lambda; T) = (\lambda I - T)^{-1}. $$

The resolvent is the technical workhorse of spectral theory. Its first virtue: as a function of $\lambda$, it is operator-valued and analytic on $\rho(T)$.

![The resolvent set is open and the resolvent is analytic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/08-spectral-theory/fa_v2_08_2_resolvent.png)

**Theorem.** $\rho(T)$ is open, and $\lambda \mapsto R(\lambda; T)$ is analytic from $\rho(T)$ to $B(X)$, in the sense that it has a convergent power series expansion at every point of $\rho(T)$.

*Proof.* Fix $\lambda_0 \in \rho(T)$. For $\lambda$ near $\lambda_0$, the formal Neumann series

$$ R(\lambda; T) = R(\lambda_0; T) \sum_{n=0}^\infty (\lambda_0 - \lambda)^n R(\lambda_0; T)^n $$

converges in operator norm whenever $|\lambda - \lambda_0| < \|R(\lambda_0; T)\|^{-1}$ — the Neumann series argument. So $\lambda$ is in $\rho(T)$ as well, with the resolvent given by the series, hence analytic. $\square$

A standard consequence is the **first resolvent identity**: $R(\lambda) - R(\mu) = (\mu - \lambda) R(\lambda) R(\mu)$, valid for $\lambda, \mu \in \rho(T)$. It is the operator analog of the partial-fraction identity $1/(\lambda - t) - 1/(\mu - t) = (\mu - \lambda)/((\lambda - t)(\mu - t))$, and it is what makes the resolvent useful in contour integration.

**The spectrum is non-empty and bounded.** For any $T \in B(X)$ on a complex Banach space, $\sigma(T) \neq \emptyset$ and $\sigma(T) \subset \{\lambda : |\lambda| \leq \|T\|\}$. The bound is the Neumann series argument: for $|\lambda| > \|T\|$, the series $\sum (T/\lambda)^n / \lambda$ converges to $R(\lambda; T)$. Non-emptiness uses that if $\sigma(T) = \emptyset$, then $R(\lambda; T)$ would be entire and bounded, hence constant by Liouville, but it goes to zero as $|\lambda| \to \infty$, hence is identically zero — contradiction. The non-emptiness of the spectrum is thus a complex-analytic theorem that has no analog in real Banach spaces.

## Spectral Radius

Define the **spectral radius** $r(T) = \sup\{|\lambda| : \lambda \in \sigma(T)\}$. The neat fact is that this geometric quantity equals an analytic limit:

![Spectral radius formula](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig08_spectral_radius.png)

![Spectral radius formula r(T) = lim ||T^n||^(1/n)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/08-spectral-theory/fa_v2_08_3_spectral_radius.png)

**Spectral radius formula.** $r(T) = \lim_{n \to \infty} \|T^n\|^{1/n}$.

The limit exists by Fekete's lemma applied to the sub-multiplicative sequence $\|T^n\|$. For $|\lambda| > r(T)$, the Neumann series $\sum T^n / \lambda^{n+1}$ converges (root test), giving $\lambda \in \rho(T)$. The opposite direction comes from the analyticity of the resolvent: it must have a singularity somewhere on the circle $|\lambda| = r(T)$, which forces $\sigma(T)$ to touch that circle.

**Why this is shocking.** The left side is purely about the spectrum (a geometric property of the operator). The right side is purely about iterating $T$ (an analytic property). They are equal. In particular: an operator with $\|T^n\|^{1/n} \to 0$ is *quasinilpotent*, with spectrum $\{0\}$. The Volterra operator has this property. Conversely, an operator with spectral radius zero behaves, asymptotically, like a contraction: $\|T^n\| \to 0$, eventually.

Try a small numerical example. Take the $3 \times 3$ matrix

$$ A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}. $$

Then $A^2$ has a single $1$ in the top-right corner, $A^3 = 0$. So $\|A^n\| = 0$ for $n \geq 3$, and $r(A) = 0$. The spectrum of $A$ is $\{0\}$ (it is nilpotent). Now take $B = A + 0.01 \cdot I$. The spectrum is $\{0.01\}$, $\|B^n\|^{1/n} \to 0.01$. The eigenvalue equals the asymptotic growth rate. The spectral radius formula in three lines.

## Self-Adjoint Operators on Hilbert Space

From here on, $H$ is a complex Hilbert space and $T \in B(H)$ is self-adjoint, meaning $T = T^*$. The world becomes much friendlier.

![Self-adjoint operators have real spectrum](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig08_selfadjoint_spectrum.png)

![Spectrum of a self-adjoint operator lies on the real line](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/08-spectral-theory/fa_v2_08_4_self_adjoint.png)

**The spectrum of a self-adjoint operator is real.** For $\lambda = a + ib \in \mathbb{C}$ with $b \neq 0$, the operator $T - \lambda I$ satisfies

$$ \|(T - \lambda I) x\|^2 = \|(T - aI)x\|^2 + b^2 \|x\|^2 \geq b^2 \|x\|^2, $$

so $T - \lambda I$ is bounded below, hence injective with closed range. By self-adjointness, $\text{range}(T - \lambda I)^\perp = \ker(T - \overline{\lambda} I)$, and the same argument shows this kernel is trivial. So the range is all of $H$, $T - \lambda I$ is invertible, and $\lambda \in \rho(T)$.

The spectrum of a self-adjoint operator therefore lies on $\mathbb{R}$. In fact, it lies in $[-\|T\|, \|T\|]$, and (a finer estimate) in $[m, M]$ where $m = \inf_{\|x\|=1} \langle Tx, x \rangle$ and $M = \sup_{\|x\|=1} \langle Tx, x\rangle$, with $m, M \in \sigma(T)$.

**Residual spectrum is empty.** If $\lambda I - T$ has dense range, the same self-adjointness identity shows it is also injective, hence the whole spectrum is point or continuous. No third type. This dichotomy is what makes the spectral theorem for self-adjoint operators so clean: every spectral value is either an eigenvalue (with eigenvector) or a continuous spectrum value (with "approximate eigenvectors": Weyl sequences $x_n$ with $\|x_n\| = 1$ and $(T - \lambda I)x_n \to 0$).

## The Continuous Functional Calculus

This is the key idea, and it is the cleanest one in spectral theory. Given $T = T^*$ bounded with spectrum $\sigma(T) \subset [m, M] \subset \mathbb{R}$, we want to *apply functions to* $T$. For polynomials this is trivial: $p(T) = \sum a_n T^n$. For power series with radius of convergence exceeding $\|T\|$, the same. But what about the function $f(t) = e^t$ on the spectrum? Or $f(t) = \sqrt{t}$ on $[0, M]$ for positive $T$? Or $f(t) = \mathbf{1}_{(\lambda_0, \infty)}(t)$, the indicator of an interval?

![Functional calculus: applying functions to operators](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig08_functional_calculus.png)

The clean answer is the **continuous functional calculus**.

![Functional calculus: applying functions to operators](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/08-spectral-theory/fa_v2_08_6_func_calc.png)

**Theorem (Continuous Functional Calculus).** Let $T = T^* \in B(H)$. There is a unique map $\Phi: C(\sigma(T)) \to B(H)$ such that

1. $\Phi$ is a $*$-algebra homomorphism: $\Phi(fg) = \Phi(f)\Phi(g)$, $\Phi(\overline{f}) = \Phi(f)^*$, $\Phi(1) = I$, $\Phi(t) = T$.
2. $\Phi$ is an isometry: $\|\Phi(f)\| = \|f\|_{C(\sigma(T))} = \sup_{\lambda \in \sigma(T)} |f(\lambda)|$.
3. $\sigma(\Phi(f)) = f(\sigma(T))$ (spectral mapping theorem).

The construction goes through the polynomial case first: define $\Phi(p) = p(T)$ for polynomials, verify $\|p(T)\| = \|p\|_{C(\sigma(T))}$ (this is what self-adjointness gives), then extend by density to continuous functions via Stone-Weierstrass. The end product is a way to define $f(T)$ for any $f$ continuous on the spectrum, with all the algebraic and norm properties one would want.

In particular, we have $e^T$, $\sqrt{T}$ (for positive $T$), $|T| = (T^*T)^{1/2}$ for general $T$, and a range of useful operator-functions. The continuous functional calculus is what makes applied operator theory possible: anywhere one wants to compute $f(T)$ for a self-adjoint $T$, this gives a clean way to do it.

A small numerical instance. Take $T = \text{diag}(1, 2, 3)$ on $\mathbb{C}^3$. Then $\sigma(T) = \{1, 2, 3\}$, and for $f$ continuous, $\Phi(f) = \text{diag}(f(1), f(2), f(3))$ — the functional calculus reduces to applying $f$ to the eigenvalues. For an operator with continuous spectrum, the same idea works but the "diagonal" is replaced with a multiplication operator on a function space. This is essentially the content of the spectral theorem.

## The Spectral Theorem for Bounded Self-Adjoint Operators

There are several equivalent formulations. The two I find most useful are the **multiplication operator form** and the **spectral measure form**.

![Spectral theorem: decomposition via projection-valued measure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig08_spectral_theorem.png)

![Spectral theorem for bounded self-adjoint operators on Hilbert space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/08-spectral-theory/fa_v2_08_5_spectral_thm.png)

**Theorem (Spectral Theorem, Multiplication Form).** Let $T \in B(H)$ be self-adjoint. There exist a measure space $(\Omega, \mu)$, a unitary $U: H \to L^2(\Omega, \mu)$, and a bounded measurable function $h: \Omega \to \mathbb{R}$ such that

$$ U T U^{-1} = M_h, $$

where $M_h$ is multiplication by $h$. In words: every bounded self-adjoint operator is unitarily equivalent to a multiplication operator on some $L^2$ space.

This is the right generalization of "every Hermitian matrix is diagonalizable." For matrices, $\Omega$ is finite (the index set of eigenvalues, with multiplicity), $\mu$ is counting measure, and $M_h$ is the diagonal matrix. For general bounded self-adjoint operators, $\Omega$ may be a continuum and $\mu$ a continuous measure, but the structural picture is the same: in the right basis, $T$ is multiplication by a real-valued function.

**Theorem (Spectral Theorem, Spectral Measure Form).** Let $T \in B(H)$ be self-adjoint. There exists a unique projection-valued measure $E$ on the Borel sets of $\sigma(T)$ such that

$$ T = \int_{\sigma(T)} \lambda \, dE(\lambda), $$

where the integral is interpreted weakly: $\langle T x, y \rangle = \int \lambda \, d \langle E(\lambda) x, y \rangle$. The measure $E$ assigns to each Borel set $B \subset \sigma(T)$ an orthogonal projection $E(B)$, with $E(\emptyset) = 0$, $E(\sigma(T)) = I$, and countable additivity for disjoint unions.

This is the form that physicists love: it is the operator-theoretic content of "an observable has a spectrum, and a measurement projects onto an eigenspace." The projection-valued measure $E$ is what tells you, for each Borel set $B$, the orthogonal projection onto "the part of the state living in spectral region $B$."

For compact self-adjoint operators (article 7), $E$ is a sum of finite-rank projections at the eigenvalues. For multiplication by $x$ on $L^2[0, 1]$, $E(B)$ is multiplication by $\mathbf{1}_B$ — the projection onto functions supported on $B$. Both cases are special instances of the same theorem.

## Examples to Internalize

**Example A (multiplication on $L^2$).** $(Mf)(x) = m(x) f(x)$ on $L^2(\Omega, \mu)$ for a real-valued bounded measurable $m$. Spectrum is the **essential range** of $m$:

$$ \sigma(M) = \{\lambda : \mu(\{x : |m(x) - \lambda| < \varepsilon\}) > 0 \text{ for all } \varepsilon > 0\}. $$

The spectral measure is $E(B) = M_{\mathbf{1}_{m^{-1}(B)}}$, multiplication by the indicator of the preimage of $B$. This is the "model" example to which every self-adjoint operator is unitarily equivalent.

![Spectra of classical operators: shifts, multiplication, integral operators](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/08-spectral-theory/fa_v2_08_7_examples.png)

**Example B (right shift on $\ell^2$).** $S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$. Not self-adjoint: $S^*$ is the left shift $(x_1, x_2, \ldots) \mapsto (x_2, x_3, \ldots)$. Spectrum: $\sigma(S) = \overline{\mathbb{D}} = \{|\lambda| \leq 1\}$. Point spectrum of $S$: empty (no $\ell^2$ eigenvector). Point spectrum of $S^*$: the open disk $\mathbb{D}$, with eigenvectors $(1, \lambda, \lambda^2, \ldots)$ for $|\lambda| < 1$. The asymmetry between $S$ and $S^*$ is a textbook illustration that point spectrum is not a self-adjointness-flavored quantity.

**Example C (Laplacian on $L^2(\mathbb{R})$, technically unbounded but instructive).** $\Delta f = f''$. Via Fourier transform, $\Delta$ becomes multiplication by $-|\xi|^2$ on $L^2(\mathbb{R})$. Spectrum: $\sigma(\Delta) = (-\infty, 0]$, all continuous spectrum. There are no $L^2$ eigenfunctions. The "eigenfunctions" $e^{i\xi x}$ are not in $L^2$, they are *generalized* eigenfunctions. This is the prototype of how Fourier analysis is spectral theory in disguise.

**Example D (integral operator with smoothing kernel on $L^2[0, 1]$).** $K(x, y) = e^{-|x-y|^2}$, a Gaussian kernel. The operator is compact and self-adjoint, so eigenvalues form a sequence going to zero. The eigenfunctions can be approximated numerically by discretizing on a fine grid and diagonalizing the resulting matrix. The eigenvalues decay roughly exponentially, matching the smoothness of the kernel.

These examples are worth building intuition around. Almost any self-adjoint operator one meets in mathematical physics is a variation on one of A, B, C, D, or a combination.

## A Numerical Example to Anchor the Picture

Consider the bounded self-adjoint operator $(T f)(x) = (1 - x^2) f(x) + \int_{-1}^1 K(x, y) f(y) \, dy$ on $L^2[-1, 1]$, with $K(x, y)$ a small Hilbert-Schmidt kernel. The first part is multiplication by $1 - x^2$, with continuous spectrum $[0, 1]$. The second part is a compact self-adjoint operator. The full operator has both a continuous spectrum (from the multiplication part) and possibly a discrete eigenvalue set (perturbations from the compact part).

Numerically, one discretizes the interval into $N = 1000$ points, builds the resulting $1000 \times 1000$ symmetric matrix, and diagonalizes. The eigenvalues cluster densely on $[0, 1]$ (approximating the continuous spectrum) and possibly have a few outliers (approximating discrete eigenvalues). As $N \to \infty$, the cluster of eigenvalues fills out $[0, 1]$ at a rate predicted by the spectral density of the multiplication operator, and the outliers stabilize. This is the qualitative picture of operator spectra: in the limit, multiplication operators give continuous spectrum, compact operators give discrete eigenvalues, and combinations give a mix. The spectral measure form of the theorem is the structural statement that captures both regimes.

![Animation: spectrum shifting under perturbation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/gif08_spectrum_shift.gif)

Concretely: for the multiplication operator $M_g$ on $L^2[0, 1]$ with $g(x) = x$, the spectral measure $E(B) = M_{\mathbf{1}_{B}}$ is multiplication by the indicator of $B$, and $\langle E(B) f, f \rangle = \int_B |f|^2 \, dx$. This *is* the Lebesgue measure of $B$ weighted by $|f|^2$. The eigenfunctions, in the strict sense, do not exist; the right replacement is the spectral measure.

## The Decomposition $\sigma_{ac} \cup \sigma_{sc} \cup \sigma_{pp}$ in Practice

The decomposition $\sigma(T) = \sigma_{ac}(T) \cup \sigma_{sc}(T) \cup \sigma_{pp}(T)$ — absolutely continuous, singular continuous, pure point — is what comes out of the spectral measure form once one applies the Lebesgue decomposition theorem to the projection-valued measure. Most operators of physical interest have $\sigma_{sc} = \emptyset$, but counterexamples exist (random Schrödinger operators with sparse potentials, almost-Mathieu operators at certain parameter values), and these are subtle and interesting.

Why does the decomposition matter? In quantum mechanics, the parts have physical meaning. **Pure point** spectrum corresponds to bound states (electrons trapped near a nucleus, particles in a confining potential). **Absolutely continuous** spectrum corresponds to scattering states (free particles, particles that can escape to infinity). **Singular continuous** spectrum is exotic but real: it corresponds to states that are neither bound nor scattering, with anomalous transport properties. The proof that ordinary atoms have only point + absolutely continuous spectrum (no singular continuous) is a deep theorem in mathematical physics (the RAGE theorem and its descendants), and it took decades to establish.

The continuous functional calculus extends to a **Borel functional calculus**, allowing $f(T)$ for any bounded Borel function $f$ on the spectrum — in particular for indicator functions, which gives back the projection-valued spectral measure $E(B) = \mathbf{1}_B(T)$. So the continuous and Borel functional calculi together give the spectral theorem; conversely, the spectral theorem implies them by integration against $E$. The whole story is a tight three-way equivalence.

## Computing Spectra: A Practitioner's Catalog

Some operators come up so often that knowing their spectra by heart is useful. A short catalog.

**Identity, $I$:** $\sigma(I) = \{1\}$. Trivial, but worth stating. The identity has a single eigenvalue.

**Diagonal multiplication on $\ell^2$:** $T(x_1, x_2, \ldots) = (\lambda_1 x_1, \lambda_2 x_2, \ldots)$ with $\lambda_n$ bounded. Spectrum is the closure of $\{\lambda_n\}$. Each $\lambda_n$ is an eigenvalue. Limit points of $\{\lambda_n\}$ are continuous spectrum.

**Multiplication by $g$ on $L^2(\Omega, \mu)$:** spectrum is the **essential range** of $g$. Pure point spectrum at the values $\lambda$ where $\mu(g^{-1}(\{\lambda\})) > 0$ (so $g$ is constant on a set of positive measure); continuous spectrum elsewhere.

**Right shift on $\ell^2$:** $\sigma(S) = \overline{\mathbb{D}}$, point spectrum empty, residual spectrum the open disk $\mathbb{D}$, continuous spectrum the unit circle. The residual-spectrum mass on $\mathbb{D}$ goes away when we take the adjoint $S^*$ (left shift): then $\mathbb{D}$ becomes point spectrum.

**Discrete Laplacian on $\ell^2(\mathbb{Z})$:** $(\Delta x)_n = x_{n+1} + x_{n-1} - 2 x_n$. Via the Fourier transform $\ell^2(\mathbb{Z}) \cong L^2([-\pi, \pi])$, this is multiplication by $2\cos\theta - 2$ for $\theta \in [-\pi, \pi]$. So the spectrum is $[-4, 0]$, all continuous, no eigenvalues. The picture matches what physicists call the "tight-binding" model band structure.

**Continuous Laplacian $\Delta$ on $L^2(\mathbb{R})$:** unbounded, but instructive: spectrum $(-\infty, 0]$ via Fourier transform.

**Volterra integral operator** $(V f)(x) = \int_0^x f(y) \, dy$ on $L^2[0, 1]$: compact, $\sigma(V) = \{0\}$. As an aside, the $n$-th iterate has norm $\|V^n\| = 1/n!$, so $\|V^n\|^{1/n} \to 0$ confirms $r(V) = 0$ via the spectral radius formula.

**Toeplitz operator** $T_g$ with continuous symbol $g$ on the unit circle, acting on the Hardy space $H^2$: spectrum is the curve $g(\mathbb{T})$ together with all points the curve winds around (this is a remarkable result of Brown and Halmos, 1964). Toeplitz operators are essentially "Fourier multipliers acting on positive frequencies," and their spectral theory is a major subject in operator theory and harmonic analysis. The Hardy-Hilbert kernel $(Hf)(x) = \int_0^\infty f(y)/(x+y)\,dy$ on $L^2(0, \infty)$ is bounded with operator norm exactly $\pi$ (Hilbert's inequality), with continuous spectrum $[0, \pi]$ that one can compute explicitly via Mellin transform.

These examples are not just trivia. They are the building blocks of intuition: when faced with a new operator, one should ask which of these it resembles. Most operators in practice are perturbations of these models, or combinations.

## Spectral Theory in Numerical Linear Algebra

A practical aside. The whole apparatus of spectral theory has direct counterparts in numerical linear algebra. The QR algorithm computes eigenvalues by iterating a shifted similarity transformation; the underlying convergence proof uses the spectral mapping theorem and rate estimates from the spectral gap. The Lanczos algorithm computes eigenvalues of large symmetric matrices by building a Krylov subspace and exploiting orthogonality; the analysis uses the Rayleigh quotient and Courant-Fischer min-max. ARPACK, the standard library for large eigenvalue problems, is essentially Lanczos plus shift-and-invert tricks justified by spectral mapping.

When one studies operator spectra and computes them numerically, the same structural theorems govern both. The error analysis of finite-dimensional approximations to infinite-dimensional spectral problems is the subject of **spectral approximation theory** (Chatelin, Anselone), and it is one of the cleanest applications of operator theory to scientific computing. The take-home message: there is no firewall between operator theory and numerical analysis. The same theorems are used on both sides; only the implementation details differ.

## Why This Matters: Quantum Observables

In quantum mechanics, observables (energy, momentum, position) are self-adjoint operators on a Hilbert space. The spectrum of an observable is exactly the set of possible measurement outcomes. The spectral measure $E$ encodes the probability distribution of outcomes: in state $\psi$, the probability of measuring an outcome in Borel set $B$ is $\langle E(B) \psi, \psi \rangle$. This is not an analogy — it is the literal mathematical foundation of quantum mechanics, formulated by von Neumann in 1932.

The mystery of why eigenvalues of an operator should correspond to physical measurement outcomes has a structural answer: the mathematical structure of "self-adjoint operator on a Hilbert space" was reverse-engineered from the empirical observation that physical observables have real-valued outcomes with definite probabilities. Spectral theory is, in this sense, a piece of physics formulated in mathematical language. The continuous functional calculus tells you how to compute $f(\hat H)$, where $\hat H$ is the Hamiltonian, and that includes $e^{-it\hat H}$ — the time evolution operator. Spectral theory is what makes the Schrödinger equation actually solvable in any nontrivial sense.

Article 12 returns to this in detail. For now: spectral theory is the linear-algebraic infrastructure of quantum mechanics, and it is also the right setting for a vast amount of PDE.

## The Spectral Mapping Theorem and Its Uses

A small but useful statement: $\sigma(f(T)) = f(\sigma(T))$ for any continuous $f$ on $\sigma(T)$. So the spectrum of $T^2$ is $\{\lambda^2 : \lambda \in \sigma(T)\}$, the spectrum of $e^T$ is $\{e^\lambda : \lambda \in \sigma(T)\}$, and so on. The polynomial case is direct (factor $f - \mu$ as $\prod (z - \lambda_j)$, and $f(T) - \mu I = \prod (T - \lambda_j I)$ is invertible iff every factor is). The continuous case follows by approximation.

This is what lets us compute spectra of operators built from $T$ via algebraic operations or functional calculus. If $T \geq 0$ (positive self-adjoint), then $T^{1/2}$ is well-defined and self-adjoint, with $\sigma(T^{1/2}) = \sqrt{\sigma(T)}$. If $T = T^*$ with spectrum in $[m, M]$, then $(T - mI)/(M - m)$ has spectrum in $[0, 1]$, normalizing the operator.

## Why "Spectrum"? A Historical Aside

The word "spectrum" was Hilbert's coinage. In his 1906 lectures on integral equations, he noticed that the eigenvalues of certain symmetric integral kernels formed a "spectrum" reminiscent of the discrete lines of atomic emission spectra. The physical spectrum and the operator spectrum then evolved together: by 1925, when Heisenberg formulated matrix mechanics, the eigenvalues of energy operators were quite literally the observed spectral lines of atoms. The mathematical name for the structure preceded the physical interpretation by two decades, but the two became indistinguishable. "Spectrum" is one of the rare cases where a mathematical term and its physical referent are not just analogous but historically continuous.

## Spectral Projections and the Riesz Functional Calculus

Before unbounded operators, one more tool: the **Riesz functional calculus** for general bounded operators (not necessarily self-adjoint). For $T \in B(X)$ on a Banach space and $f$ holomorphic on a neighborhood of $\sigma(T)$, one defines

$$ f(T) = \frac{1}{2\pi i} \oint_\Gamma f(\lambda) R(\lambda; T) \, d\lambda, $$

where $\Gamma$ is a contour enclosing $\sigma(T)$ in the domain of $f$. The integral is operator-valued and converges by the analyticity of the resolvent.

In particular, taking $f = \mathbf{1}_U$ for a clopen set $U \subset \sigma(T)$ (i.e., $U$ is a connected component of the spectrum), one gets a **Riesz spectral projection**

$$ P_U = \frac{1}{2\pi i} \oint_{\Gamma_U} R(\lambda; T) \, d\lambda, $$

where $\Gamma_U$ encloses only the part of the spectrum in $U$. The projection $P_U$ commutes with $T$ and decomposes $X$ into the direct sum of the $T$-invariant subspaces $\text{range}(P_U)$ and $\text{range}(I - P_U)$, on each of which the spectrum of $T$ is reduced to either $U$ or $\sigma(T) \setminus U$. This is the operator-theoretic version of "isolating an eigenvalue" or "splitting off an invariant subspace."

For self-adjoint operators on a Hilbert space, the Riesz functional calculus and the continuous (or Borel) functional calculus agree where they overlap; the Borel calculus extends further, since we can apply non-holomorphic functions like indicators of arbitrary Borel sets. For general operators, the Riesz calculus is the strongest tool available, and it is what enables results like the Jordan canonical form for compact operators and the structure theory of operator semigroups.

A small worked example. Consider the matrix $A = \text{diag}(1, 2, 3, 4) + \varepsilon N$, where $N$ is some nilpotent perturbation and $\varepsilon$ is small. The Riesz projection $P_1$ associated with the spectral component near $\lambda = 1$ is approximately $\text{diag}(1, 0, 0, 0)$ for small $\varepsilon$, perturbed by an $O(\varepsilon)$ correction computable by the contour integral. This is how perturbation theory in quantum mechanics (Rayleigh-Schrödinger) is rigorously set up — the spectral projections of the unperturbed operator are deformed analytically as the perturbation is turned on, and the projections track the eigenvalues continuously as long as no level crossings occur.

## A Detour Through Spectral Theory of Normal Operators

Self-adjoint operators are a special case of **normal operators**: $T$ is normal if $T T^* = T^* T$. Unitary operators are normal ($U U^* = U^* U = I$); self-adjoint operators are trivially normal; positive operators are normal. The spectral theorem extends without serious modification to bounded normal operators: there is a projection-valued measure $E$ on the (now possibly complex) spectrum, with $T = \int \lambda \, dE(\lambda)$, and a continuous functional calculus $f \mapsto f(T)$ for $f \in C(\sigma(T))$.

The most useful corollary: every unitary operator $U$ on a Hilbert space is unitarily equivalent to multiplication by $e^{i\theta}$ on some $L^2$ space. Specifically, $U$ has spectrum on the unit circle, and the spectral measure on the circle gives a model. This is the right setting for thinking about Fourier transforms, lattice translations on $\ell^2(\mathbb{Z})$, and time evolution in quantum mechanics — all of these are unitary operators, and all of them are diagonalized by the spectral theorem for normal operators.

Why is this worth flagging? Because the most useful operators in physics are typically either self-adjoint (observables) or unitary (symmetries, time evolution), and both fall under the normal-operator spectral theorem. The non-self-adjoint, non-normal operators are computationally useful (transfer operators, dissipative semigroup generators) but their spectral theory is fundamentally messier — defective eigenvectors, generalized eigenspaces, Jordan structure in infinite dimensions. The normal case is where the spectral theory is at its cleanest.

A small numerical example. Take the unitary operator $U: \ell^2(\mathbb{Z}/N\mathbb{Z}) \to \ell^2(\mathbb{Z}/N\mathbb{Z})$ given by the shift $(Ux)_n = x_{n-1}$. This is the discrete Fourier transform of multiplication by $e^{2\pi i k/N}$, $k = 0, 1, \ldots, N-1$. The eigenvalues of $U$ are precisely the $N$-th roots of unity $e^{2\pi i k/N}$. The Fourier transform diagonalizes the shift, and that is the cleanest possible illustration of the spectral theorem for normal operators in finite dimensions. The infinite-dimensional case (continuous Fourier transform, Pontryagin duality) is a vast generalization but the geometric picture is the same.

## A Reading-Order Note

For someone learning this material for the first time, I recommend the following order: (1) the multiplication-operator form of the spectral theorem, because it is concrete; (2) the spectral measure form, as a refinement; (3) the continuous functional calculus, working examples until the algebraic identities become familiar; (4) the spectral mapping theorem, which then becomes a corollary; (5) Riesz projections, only when needed. Reed and Simon (Volume I) handle this in roughly this order. Rudin (Real and Complex Analysis) goes through the spectral theorem in a more abstract form via Gelfand-Naimark, which is conceptually elegant but slower to bring on tangible operators. The two routes meet eventually, but it is worth knowing both.

The single insight that took me longest to absorb was that "$T$ has continuous spectrum at $\lambda$" does not mean "$\lambda$ is an eigenvalue of a slightly perturbed operator." It means something stronger: there are unit vectors $x_n$ with $(T - \lambda) x_n \to 0$ but no convergent subsequence of $x_n$. These approximate eigenvectors, or *Weyl sequences*, are the right replacement for eigenvectors in the continuous-spectrum case. Multiplication by $x$ on $L^2[0, 1]$ has Weyl sequences at every $\lambda \in [0, 1]$: take $x_n = \sqrt{n} \mathbf{1}_{[\lambda - 1/(2n), \lambda + 1/(2n)] \cap [0, 1]}$. They are unit vectors, $(M - \lambda) x_n \to 0$ in $L^2$, but no subsequence converges. The operator is "almost diagonalized" near $\lambda$ but not actually diagonalized — and that is exactly what continuous spectrum captures.

## What's Next, and Why

The bounded self-adjoint case is the cleanest scenario, but it leaves out almost everything that matters in physics. Differentiation operators, the Laplacian, the Schrödinger Hamiltonian — these are all *unbounded*, defined only on dense subdomains of $L^2$. The next article extends spectral theory to unbounded self-adjoint operators, using the closed graph and a careful definition of self-adjointness via the adjoint operator $T^*$ and its domain.

The new technical complications are domains: an unbounded operator is a pair $(T, D(T))$ where $D(T)$ is a dense subspace and $T: D(T) \to H$ is linear. Self-adjointness is no longer a single equation $T = T^*$ but a domain equation $D(T) = D(T^*)$ together with $Tx = T^*x$ on the common domain. Symmetric operators (where $T \subset T^*$) need not be self-adjoint, and the Friedrichs extension and von Neumann deficiency-index theory come in to control which symmetric operators have self-adjoint extensions.

Once we have unbounded self-adjoint operators, the spectral theorem extends with minor modifications: there is still a projection-valued measure on the spectrum (now possibly unbounded as a subset of $\mathbb{R}$), and the operator is still $\int \lambda \, dE(\lambda)$, with $D(T) = \{x : \int \lambda^2 \, d\langle E(\lambda) x, x\rangle < \infty\}$. The functional calculus extends similarly. Everything in this article carries over, with care about domains.

The reward is that we can finally talk about Schrödinger operators, the heat semigroup, momentum and position observables, and the rest of mathematical physics. Domains are a small price. The conceptual lesson of this article — spectrum equals the obstruction to invertibility, and self-adjoint operators are unitarily equivalent to multiplication operators — survives the transition to unbounded operators with only minor edits. Once one has internalized this, the rest of operator-theoretic mathematical physics becomes accessible. The structure of the spectrum encodes the structure of the operator, and the functional calculus turns "applying $f$ to the operator" into a routine computation rather than a conceptual leap. In a sense everything we will do for the next four articles is variations on this theme: extending the calculus to more general operators, using it to write down explicit formulas for evolution equations, and reading off physical and analytic information from spectral data. The unifying viewpoint is that an operator's spectrum, together with its spectral measure, contains all the structural information one would want; everything else is a specialization or computational consequence.

---
