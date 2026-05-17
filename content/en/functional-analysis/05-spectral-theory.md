---
title: "Spectral Theory of Compact Operators"
date: 2021-03-29 09:00:00
tags:
  - functional-analysis
  - spectral-theory
  - compact-operators
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 5
series_total: 6
translationKey: "functional-analysis-5"
description: "Compact operators behave like infinite matrices --- their spectrum is discrete, countable, and clusters only at zero."
---

## From eigenvalues to spectrum

In finite dimensions, every linear operator $A: \mathbb{C}^n \to \mathbb{C}^n$ has $n$ eigenvalues (counting multiplicity). The matrix is understood once you know its eigenvalues and eigenspaces. In infinite dimensions, the situation is more subtle: an operator might have no eigenvalues at all, yet still have a rich "spectrum" that controls its behavior.

The gap between eigenvalues and spectrum is the central theme of this chapter. In finite dimensions they coincide; in infinite dimensions they diverge dramatically, and the spectrum (not just the eigenvalues) is what matters.

I think of the spectrum as measuring "how close an operator is to being non-invertible at each $\lambda$." An eigenvalue is an exact obstruction (the kernel is nontrivial); the continuous spectrum represents approximate obstructions (the operator is injective but not boundedly invertible). Both matter for the functional calculus.

![Spectrum decomposition diagram](/images/functional-analysis/fig05_spectrum.png)

## The spectrum

For a bounded operator $T$ on a Banach space $X$, the **resolvent set** is:

$$\rho(T) = \{\lambda \in \mathbb{C} : (T - \lambda I) \text{ is bijective with bounded inverse}\}.$$

The **spectrum** is the complement:

$$\sigma(T) = \mathbb{C} \setminus \rho(T).$$

Note: bijectivity alone isn't enough --- we need the inverse to be bounded. In a Banach space, the open mapping theorem tells us that a bijective bounded operator automatically has bounded inverse. So for operators on Banach spaces, $\lambda \in \rho(T)$ iff $T - \lambda I$ is bijective.

The spectrum splits into three disjoint parts:

$$\sigma_p(T) = \{\lambda : T - \lambda I \text{ is not injective}\} \quad \text{(point spectrum --- eigenvalues)},$$

$$\sigma_c(T) = \{\lambda : T - \lambda I \text{ is injective, has dense (but not closed) range}\} \quad \text{(continuous spectrum)},$$

$$\sigma_r(T) = \{\lambda : T - \lambda I \text{ is injective, range not dense}\} \quad \text{(residual spectrum)}.$$

In finite dimensions, $\sigma(T) = \sigma_p(T)$ always (non-injectivity = non-surjectivity by rank-nullity). In infinite dimensions, all three pieces can be nonempty simultaneously.

**Basic structural facts:**
1. $\sigma(T)$ is always nonempty (for complex Banach spaces --- proved using Liouville's theorem applied to the operator-valued analytic function $\lambda \mapsto (T - \lambda I)^{-1}$).
2. $\sigma(T)$ is compact, contained in $\{|\lambda| \le \|T\|\}$.
3. The **spectral radius** is $r(T) = \sup_{\lambda \in \sigma(T)} |\lambda| = \lim_{n \to \infty} \|T^n\|^{1/n}$ (Gelfand's formula).

**Example 1: Right shift on $\ell^2$.** The operator $S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$. If $Sx = \lambda x$, then $0 = \lambda x_1$, $x_1 = \lambda x_2$, $x_2 = \lambda x_3$, etc. For $|\lambda| < 1$, this gives $x_n = x_1/\lambda^{n-1}$ which isn't in $\ell^2$ unless $x_1 = 0$. For $|\lambda| \ge 1$, same conclusion. So $\sigma_p(S) = \emptyset$ --- no eigenvalues.

Yet $\sigma(S) = \overline{\mathbb{D}}$, the closed unit disk. Why? For $|\lambda| < 1$, the range of $S - \lambda I$ is not dense: if $y = (1, 0, 0, \ldots)$ were in $\overline{\text{range}(S - \lambda I)}$, one can show $y$ would need to equal $(S - \lambda I)x$ for some $x$, leading to $x_n = -\lambda^{-(n+1)}$, not in $\ell^2$. A careful argument shows these $\lambda$ are in $\sigma_r(S)$.

**Example 2: Multiplication operator.** On $L^2[0,1]$, define $(M_t f)(s) = s \cdot f(s)$. This has $\sigma(M_t) = [0,1]$ (the essential range of the multiplier), $\sigma_p(M_t) = \emptyset$ (no eigenvalues), and $\sigma_c(M_t) = [0,1]$. The spectrum is purely continuous --- every point in $[0,1]$ is in the spectrum because $M_t - \lambda I$ fails to have bounded inverse, but no point is an eigenvalue.

## Compact operators

An operator $T \in B(X, Y)$ is **compact** if the image of the unit ball $\overline{T(B_X)}$ is compact in $Y$. Equivalently: every bounded sequence $(x_n)$ has a subsequence such that $(Tx_{n_k})$ converges.

Key structural properties:
- Every finite-rank operator ($\dim \text{range}(T) < \infty$) is compact.
- The compact operators $K(X)$ form a closed two-sided ideal in $B(X)$: if $T$ is compact, $A, B$ bounded, then $ATB$ is compact.
- On a Hilbert space, $T$ is compact iff $T$ is the norm-limit of finite-rank operators.
- $I$ is compact iff $\dim X < \infty$.
- If $T$ is compact and $\lambda \ne 0$, then $\dim \ker(T - \lambda I) < \infty$ (eigenspaces are finite-dimensional).

**Example 3: Hilbert-Schmidt operators.** On $L^2[0,1]$, define:

$$(Tf)(s) = \int_0^1 k(s,t) f(t)\, dt$$

where $k \in L^2([0,1]^2)$. This is compact (in fact, it's the norm-limit of the finite-rank operators obtained by truncating the singular value expansion of $k$). The condition $k \in L^2$ is equivalent to $\sum_n \sigma_n^2 < \infty$ where $\sigma_n$ are the singular values.

**Example 4: Diagonal operators.** On $\ell^2$, define $D_\lambda x = (\lambda_1 x_1, \lambda_2 x_2, \ldots)$ where $\lambda_n \in \mathbb{C}$. Then $D_\lambda$ is compact iff $\lambda_n \to 0$. The eigenvalues are $\{\lambda_n\}$, and the spectrum is $\overline{\{\lambda_n\}} = \{\lambda_n\} \cup \{0\}$ (if infinitely many $\lambda_n$ are distinct).

## The Riesz-Schauder theorem

**Theorem (Spectral theorem for compact operators).** Let $T$ be a compact operator on an infinite-dimensional Banach space $X$. Then:

1. $0 \in \sigma(T)$ (since $T$ cannot be invertible --- otherwise $I = T^{-1}T$ would be compact, contradicting $\dim X = \infty$).
2. $\sigma(T) \setminus \{0\}$ consists entirely of eigenvalues (no continuous or residual spectrum away from zero).
3. The nonzero eigenvalues form an at most countable set, with $0$ as the only possible accumulation point.
4. Each nonzero eigenvalue has finite multiplicity.

This is the closest infinite-dimensional analogue of the finite-dimensional spectral theorem. Compact operators have "discrete" spectra (except possibly at $0$), just like matrices.

*Proof of (2).* Suppose $\lambda \ne 0$ and $T - \lambda I$ is injective. We show $T - \lambda I$ is surjective (hence $\lambda \notin \sigma(T)$). The key tool is:

**Riesz's Lemma.** If $M$ is a proper closed subspace of a normed space $X$, then for every $\varepsilon > 0$ there exists $x$ with $\|x\| = 1$ and $d(x, M) > 1 - \varepsilon$.

Using this: if $\text{range}(T - \lambda I)$ were a proper closed subspace, define $M_n = \text{range}((T - \lambda I)^n)$ --- a strictly decreasing sequence of closed subspaces. By Riesz's lemma, find unit vectors $x_n \in M_n$ with $d(x_n, M_{n+1}) > 1/2$. For $n > m$: $Tx_n - Tx_m = \lambda x_n + [(T - \lambda I)x_n - Tx_m]$ where the bracketed term is in $M_{n+1}$. So $\|Tx_n - Tx_m\| \ge |\lambda|/2$, contradicting compactness of $T$ (the bounded sequence $(x_n)$ has no subsequence with convergent $T$-image). $\square$

*Proof of (4).* If $\ker(T - \lambda I)$ were infinite-dimensional, the closed unit ball in $\ker(T - \lambda I)$ would be bounded, and $T$ acts on it as $\lambda I$ (scaling). If $\dim \ker(T - \lambda I) = \infty$, we can find an infinite orthonormal sequence $(e_n)$ in the kernel (or use Riesz's lemma in the Banach case). Then $\|Te_n - Te_m\| = |\lambda|\|e_n - e_m\| \ge |\lambda| \cdot c > 0$, contradicting compactness. $\square$

## The spectral theorem for compact self-adjoint operators

On a Hilbert space, compact self-adjoint operators have the cleanest possible structure:

**Theorem.** Let $T$ be a compact self-adjoint operator on a separable Hilbert space $H$ with $\dim H = \infty$. Then:
- All eigenvalues are real.
- Eigenvectors for distinct eigenvalues are orthogonal.
- There exists an orthonormal sequence $(e_n)$ of eigenvectors with eigenvalues $(\lambda_n)$, $\lambda_n \to 0$, such that:

$$Tx = \sum_{n=1}^\infty \lambda_n \langle x, e_n \rangle\, e_n \quad \forall x \in H.$$

The operator is completely "diagonalized" --- it's the infinite-dimensional analogue of the spectral theorem for real symmetric matrices: $A = Q\Lambda Q^T$.

*Proof sketch.* First show that $\|T\| = \sup_{\|x\|=1} |\langle Tx, x \rangle|$ for self-adjoint $T$ (this uses polarization). Achieve the supremum: since $T$ is compact, the unit ball maps to a compact set, so there exists $e_1$ with $Te_1 = \lambda_1 e_1$ where $|\lambda_1| = \|T\|$. Then $T$ maps $\{e_1\}^\perp$ to itself (self-adjointness), and $T|_{\{e_1\}^\perp}$ is still compact and self-adjoint. Induct. $\square$

**Example 5.** The integral operator $(Tf)(s) = \int_0^1 \min(s,t) f(t)\, dt$ on $L^2[0,1]$ is compact, self-adjoint, and positive. Its eigenvalues are:

$$\lambda_n = \frac{1}{(n - 1/2)^2\pi^2}, \quad n = 1, 2, 3, \ldots$$

with eigenfunctions $e_n(t) = \sqrt{2}\sin((n-1/2)\pi t)$. This operator is the Green's function (inverse) of $-d^2/dt^2$ with boundary conditions $f(0) = f'(1) = 0$. The spectral decomposition of the compact operator gives you the eigenfunction expansion for the differential equation.

**Example 6 (Mercer's theorem).** If $T$ is a positive compact self-adjoint integral operator on $L^2[0,1]$ with continuous kernel $k(s,t)$, then:

$$k(s,t) = \sum_{n=1}^\infty \lambda_n e_n(s) e_n(t),$$

converging uniformly. The kernel decomposes as a sum of rank-1 terms --- this is the continuous analogue of the eigendecomposition $A = \sum \lambda_i v_i v_i^T$.

## Fredholm alternative

**Theorem (Fredholm alternative).** For a compact operator $T$ on a Banach space $X$ and $\lambda \ne 0$, exactly one of the following holds:

(a) The equation $(T - \lambda I)x = y$ has a unique solution for every $y \in X$.

(b) The homogeneous equation $(T - \lambda I)x = 0$ has a nontrivial solution.

There is no middle ground. Either the inhomogeneous equation is uniquely solvable for all right-hand sides, or the homogeneous equation has nontrivial solutions (and then solvability of the inhomogeneous equation requires compatibility conditions on $y$).

*Proof.* This follows from the Riesz-Schauder theorem: either $\lambda \notin \sigma(T)$ (case a, $T - \lambda I$ is bijective) or $\lambda \in \sigma_p(T)$ (case b, since the only spectrum away from $0$ is point spectrum). $\square$

**Application (Integral equations).** Consider $f(s) - \lambda \int_0^1 k(s,t)f(t)\,dt = g(s)$ where $k$ is a Hilbert-Schmidt kernel. The Fredholm alternative says: either this has a unique solution for every $g \in L^2$, or the homogeneous version ($g = 0$) has a nontrivial solution. The "resonance" values of $\lambda$ (where the homogeneous equation is solvable) are exactly the eigenvalues of the integral operator --- they form a discrete set accumulating only at $0$.

## Beyond compact: a glimpse at the general theory

For non-compact bounded operators, the spectrum can be much wilder:

- **Bilateral shift** on $\ell^2(\mathbb{Z})$: $(Ux)_n = x_{n-1}$. This is unitary, so $\sigma(U) \subseteq \{|\lambda| = 1\}$. In fact $\sigma(U) = \{|\lambda| = 1\}$, the entire unit circle, purely continuous spectrum with no eigenvalues.

- **Multiplication by $t$** on $L^2[0,1]$: $\sigma(M_t) = [0,1]$, a continuous interval. No eigenvalues, no residual spectrum.

- **Weighted shift**: $(Tx)_n = w_n x_{n-1}$ where $w_n$ are weights. The spectrum depends intricately on the asymptotics of $w_n$.

The full spectral theorem for normal operators on Hilbert spaces replaces the eigenvalue sum with an integral against a **projection-valued measure**:

$$T = \int_{\sigma(T)} \lambda\, dE(\lambda),$$

where $E$ is a resolution of the identity. This is the definitive generalization of matrix diagonalization --- but it requires measure theory on the spectrum rather than just sums over eigenvalues.

## Singular values and the trace class

For compact operators on Hilbert spaces, the **singular values** $s_n(T)$ are the eigenvalues of $(T^*T)^{1/2}$, arranged in decreasing order. They measure "how much the operator stretches" in each independent direction.

An operator is **trace class** if $\sum s_n(T) < \infty$; it's **Hilbert-Schmidt** if $\sum s_n(T)^2 < \infty$. These are important subclasses:
- Trace class $\subset$ Hilbert-Schmidt $\subset$ compact operators.
- For trace class operators, the trace $\text{tr}(T) = \sum \langle Te_n, e_n \rangle$ is well-defined (independent of the orthonormal basis).
- Hilbert-Schmidt operators correspond exactly to integral operators with $L^2$ kernels.

## What's next

The final chapter connects functional analysis to PDE: distributions generalize functions, Sobolev spaces give the right framework for weak solutions, and the abstract machinery we've built --- completeness, dual spaces, compact operators --- becomes the engine that drives existence and regularity theory.

---

*This is Part 5 of [Functional Analysis](/en/series/functional-analysis/) (6 parts).
Previous: [Part 4 --- The Big Four Theorems](/en/functional-analysis/04-big-theorems/) · Next: [Part 6 --- Distributions and Sobolev Spaces](/en/functional-analysis/06-distributions-sobolev/)*
