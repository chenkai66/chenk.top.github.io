---
title: "Functional Analysis (7): Compact Operators — The Bridge to Finite Dimensions"
date: 2021-10-13 09:00:00
tags:
  - functional-analysis
  - compact-operators
  - spectral-theory
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "Compact operators are limits of finite-rank operators and inherit much finite-dimensional spectral behavior — the Fredholm alternative and spectral theorem for compact self-adjoint operators."
disableNunjucks: true
series_order: 7
series_total: 12
translationKey: "functional-analysis-7"
---

I owe my fondness for compact operators to a small embarrassment. As an undergraduate I assumed that infinite-dimensional linear algebra would feel exotic everywhere. It does not. There is a wide and well-mapped suburb of operator theory in which everything one learned about symmetric matrices -- eigenvalues, orthogonal eigenvectors, the spectral decomposition -- comes back almost unchanged, just with eigenvalues tailing off to zero instead of a finite list. That suburb is the world of compact operators, and the price of admission is a single condition: the operator must squeeze the unit ball into a relatively compact set. Once that condition is met, nearly everything follows: the spectrum is countable, nonzero eigenvalues are isolated with finite-dimensional eigenspaces, the Fredholm alternative holds, and integral equations of the second kind become as tractable as linear systems. The line between matrices and infinite-dimensional operators ceases to be a wall and becomes a permeable membrane.

The condition "maps bounded sets to precompact sets" sounds abstract, but it is a precise way of saying *this operator behaves nearly like a matrix*. Geometrically: while a generic bounded operator can map the unit ball to something as wild as the unit ball itself (no compactness gained), a compact operator must map it to something "essentially finite-dimensional" -- a set that, for any tolerance $\varepsilon$, can be covered by finitely many $\varepsilon$-balls. The operator cannot spread mass uniformly across infinitely many directions; it must concentrate its effect on finitely many directions (up to an arbitrarily small error). This concentration is what makes spectral decomposition possible -- the operator is "almost" a finite-rank operator, and finite-rank operators are just matrices.

![Spectrum of compact operator clusters at zero](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig07_spectrum_compact.png)


A terminological note: Riesz (1918) called these operators *vollstetig* (totally continuous). The older English translation "completely continuous" still appears in pre-1970 books. The modern definition (maps bounded sets to precompact sets) is the clean one; the older definition (maps weakly convergent sequences to norm-convergent ones) is equivalent on reflexive spaces but differs on $\ell^1$. I use the modern definition throughout.

![Compact operator: maps the unit ball to a precompact set](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_1_compact_def.png)

## Definition, Examples, and the Approximation by Finite Rank


![Spectrum of a compact operator: eigenvalues accumulate at 0](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fa07_compact_spectrum.png)

![Animation: finite-rank operators converging to compact operator](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/gif07_compact_approx.gif)


![Finite-rank approximation of compact operators](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig07_finite_rank_approx.png)


An operator $T: X \to Y$ between Banach spaces is **compact** if $\overline{T(B_X)}$ is compact in $Y$, equivalently if every bounded sequence $(x_n)$ has a subsequence $(x_{n_k})$ with $(Tx_{n_k})$ convergent. The operator does the work of compactification that the ambient space (in infinite dimensions) declined to do. Every clever argument in this article reduces to extracting a convergent subsequence of images at exactly the right moment.

**Finite-rank operators** -- those with $\dim(\text{Range}(T)) < \infty$ -- are automatically compact (bounded sets in finite-dimensional spaces have compact closure by Heine-Borel). The space $K(X, Y)$ of compact operators is a closed subspace of $B(X, Y)$ and a two-sided ideal: if $T$ is compact and $S$ is bounded, then $ST$ and $TS$ are compact. The proof for closedness: if $T_n \to T$ in norm with each $T_n$ compact, then for any $\varepsilon > 0$, choose $n$ with $\|T - T_n\| < \varepsilon/3$. The precompact set $T_n(B_X)$ has a finite $\varepsilon/3$-net, which serves as a finite $\varepsilon$-net for $T(B_X)$ (by the triangle inequality). Total boundedness of $T(B_X)$ follows, giving precompactness.

![Finite-rank operators are compact](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_2_finite_rank.png)

The key structural fact for Hilbert spaces: **every compact operator on a Hilbert space is the norm-limit of finite-rank operators.** Proof: let $(e_n)$ be an orthonormal basis, $P_n$ the projection onto $\text{span}\{e_1, \ldots, e_n\}$. Then $P_n T$ is finite-rank and $\|T - P_n T\| \to 0$ (because $T(B_H)$ is precompact, and precompact sets in a Hilbert space can be approximated by their projections onto finite-dimensional subspaces). This is the **approximation property** of Hilbert spaces. It holds in $\ell^p$ and $L^p$ for all $p$, but Enflo (1973) showed it fails in some exotic Banach spaces.

**Worked example: integral operators.** The operator $Kf(x) = \int_0^1 k(x,y) f(y)\,dy$ on $L^2[0,1]$ with continuous kernel $k \in C([0,1]^2)$ is compact. One proof: the image of the unit ball consists of functions that are uniformly bounded ($|Kf(x)| \leq \|k\|_\infty$) and equicontinuous ($|Kf(x_1) - Kf(x_2)| \leq \omega_k(|x_1-x_2|)$ where $\omega_k$ is the modulus of continuity of $k$). By Arzela-Ascoli, the image is precompact in $C[0,1]$, hence in $L^2$.

A second proof via Hilbert-Schmidt: if $k \in L^2([0,1]^2)$, then $K$ is **Hilbert-Schmidt** with $\|K\|_{HS}^2 = \int\int |k(x,y)|^2\,dx\,dy < \infty$. Hilbert-Schmidt operators are compact (they are limits of finite-rank operators obtained by truncating the kernel's $L^2$ expansion). For continuous $k$ on $[0,1]^2$, $\|k\|_{L^2} \leq \|k\|_\infty < \infty$, confirming compactness.

**Worked example: the Volterra operator.** $Vf(x) = \int_0^x f(y)\,dy$ on $L^2[0,1]$. The kernel $k(x,y) = \mathbf{1}_{y \leq x}$ is in $L^2([0,1]^2)$ (with $\|k\|_{L^2}^2 = 1/2$), so $V$ is Hilbert-Schmidt, hence compact. Its spectrum is $\{0\}$ -- it has no nonzero eigenvalues (the equation $\int_0^x f = \lambda f$ forces $f' = f/\lambda$ with $f(0) = 0$, giving $f = 0$). So $V$ is a nonzero compact operator with trivial spectrum -- showing that the spectral theorem for compact operators requires self-adjointness in an essential way.

The non-self-adjoint case is genuinely different: $V$ has the same norm as its adjoint ($\|V\| = \|V^*\| = 2/\pi$, computed from the singular values), but its spectral radius is zero while its norm is positive. The operator is quasinilpotent -- all powers $V^n$ have $\|V^n\|^{1/n} \to 0$. One can verify directly: $V^n f(x) = \int_0^x \frac{(x-y)^{n-1}}{(n-1)!} f(y)\,dy$, so $\|V^n\| \leq 1/n!$ and $\|V^n\|^{1/n} \leq (n!)^{-1/n} \to 0$. This makes $V$ the canonical example of a compact operator that is "spectrally trivial" yet "dynamically nontrivial" -- it has no eigenvalues, yet it acts nontrivially on every nonzero vector.

The lack of a spectral decomposition for non-normal compact operators is not just an inconvenience -- it reflects genuine complexity. The theory of non-normal operators (in particular, the theory of invariant subspaces and the still-open invariant subspace problem for compact operators on general Banach spaces) is vastly harder than the normal case. The spectral theorem for compact self-adjoint operators is a gift of symmetry that non-self-adjoint operators simply do not share.


## The Spectral Theorem for Compact Self-Adjoint Operators

The centerpiece of compact operator theory is the spectral decomposition, which says that compact self-adjoint operators behave exactly like real diagonal matrices with entries tending to zero.

![Compact operator maps bounded set to relatively compact set](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig07_compact_operator.png)


**Theorem.** Let $T: H \to H$ be a compact self-adjoint operator on a separable Hilbert space. Then there exists an orthonormal system $(e_n)$ of eigenvectors with real eigenvalues $(\lambda_n)$ satisfying $\lambda_n \to 0$, such that $T = \sum_n \lambda_n \langle \cdot, e_n \rangle e_n$. The eigenvalues are the only nonzero spectral values, each has finite multiplicity, and they accumulate only at zero.

*Proof sketch.* The key step is showing $T$ has at least one eigenvector. Since $T$ is self-adjoint, $\|T\| = \sup_{\|x\|=1} |\langle Tx, x\rangle|$, and by compactness this supremum is attained at some $e_1$ with $Te_1 = \lambda_1 e_1$ where $|\lambda_1| = \|T\|$. (Compactness converts the supremum over the non-compact unit sphere into an achieved maximum.) Now restrict $T$ to $\{e_1\}^\perp$ (which $T$ leaves invariant by self-adjointness) and repeat. The eigenvalues form a sequence tending to zero because the norms $\|T|_{\{e_1,...,e_n\}^\perp}\|$ decrease -- if they did not tend to zero, the subsequence $(e_n/\lambda_n)$ would be bounded with $T(e_n/\lambda_n) = e_n$ having no convergent subsequence, contradicting compactness.

The spectral decomposition gives a complete diagonalization: relative to the orthonormal basis extending $(e_n)$ by a basis of $\ker(T)$, the operator $T$ is a diagonal matrix with entries $\lambda_1, \lambda_2, \ldots, 0, 0, \ldots$. This is the infinite-dimensional version of the finite-dimensional theorem that every real symmetric matrix is orthogonally diagonalizable. The finite-dimensional theory stops at "diagonal matrix"; the infinite-dimensional theory adds one piece: "with diagonal entries converging to zero." That convergence to zero is the content of compactness, and it is the only new ingredient.

An important corollary: the nonzero eigenvalues of a compact self-adjoint operator can accumulate only at zero. They cannot cluster at any nonzero point. This is because if $\lambda \neq 0$ were an accumulation point of eigenvalues, the corresponding eigenvectors $e_n$ with $Te_n = \lambda_n e_n$ (and $\lambda_n \to \lambda$) would form an orthonormal sequence with $\|Te_n\| = |\lambda_n| \geq |\lambda|/2 > 0$ for large $n$. But $(e_n)$ is bounded and $(Te_n)$ would then have no convergent subsequence (the vectors are orthogonal with norms bounded away from zero), contradicting compactness. So the spectrum of a compact self-adjoint operator is a sequence converging to zero (plus possibly zero itself) -- exactly what a "diagonal matrix with entries going to zero" would have.

**Worked example: Mercer's theorem.** If $k(x,y) = k(y,x)$ is a continuous positive-definite kernel on $[0,1]$, the integral operator $Kf = \int k(\cdot, y)f(y)\,dy$ is compact, self-adjoint, and positive. The spectral theorem gives $k(x,y) = \sum_n \lambda_n e_n(x) e_n(y)$ with uniform convergence (Mercer's theorem). This is the basis of kernel methods in machine learning: the kernel matrix $K_{ij} = k(x_i, x_j)$ has eigenvalues that are the coefficients $\lambda_n$ evaluated at the data points, and the eigenvectors of the Gram matrix approximate the eigenfunctions $e_n$. The spectral decay rate of the kernel determines the effective dimensionality of the RKHS (Article 3).

For the Gaussian RBF kernel $k(x,y) = e^{-|x-y|^2/(2\sigma^2)}$, the eigenvalues decay super-exponentially (faster than any geometric sequence), which is why Gaussian kernels give rise to infinite-dimensional feature spaces that are effectively very high-dimensional but not truly infinite for computational purposes.

The min-max characterization of eigenvalues (Courant-Fischer) extends from matrices to compact self-adjoint operators: $\lambda_n = \min_{\dim V = n-1} \max_{x \perp V, \|x\|=1} \langle Tx, x\rangle$. This variational characterization is the basis of Rayleigh-Ritz methods for computing eigenvalues numerically: approximate the infinite-dimensional space by a finite-dimensional subspace and compute eigenvalues of the resulting matrix. The convergence of Rayleigh-Ritz approximations is guaranteed by the min-max principle -- the $n$-th approximate eigenvalue converges to the true $n$-th eigenvalue as the approximation space grows. This is why finite-element eigenvalue computations work, and why Galerkin methods give provably convergent eigenvalue approximations for elliptic operators with compact resolvent.

![Classical compact operators: integral operators on L^2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_7_compact_examples.png)


## The Fredholm Alternative

The **Fredholm alternative** for compact operators is the infinite-dimensional version of the statement that a square matrix equation $Ax = b$ either has a unique solution (when $A$ is invertible) or the homogeneous equation $Ax = 0$ has nontrivial solutions (when $A$ is singular) -- with no third possibility.

![Fredholm alternative: either unique solution or finite-dimensional kernel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig07_fredholm_alternative.png)


**Theorem (Fredholm alternative).** Let $T: X \to X$ be compact and $\lambda \neq 0$. Then exactly one of the following holds:

(A) $\ker(\lambda I - T) = \{0\}$, in which case $\lambda I - T$ is bijective and $(\lambda I - T)^{-1}$ is bounded. The equation $\lambda x - Tx = y$ has a unique solution for every $y$.

(B) $\ker(\lambda I - T) \neq \{0\}$, in which case $\lambda$ is an eigenvalue with finite-dimensional eigenspace. The equation $\lambda x - Tx = y$ is solvable iff $y \perp \ker(\lambda I - T^*)$ (finitely many compatibility conditions).

The proof combines three ingredients. *First*, $\ker(\lambda I - T)$ is finite-dimensional (because $T$ acts as $\lambda I$ on it, so bounded sequences have convergent subsequences only if the space is finite-dimensional). *Second*, $\text{Range}(\lambda I - T)$ is closed -- this uses a standard splitting argument with compactness. *Third*, the Fredholm index $\dim\ker(\lambda I - T) - \dim\text{coker}(\lambda I - T) = 0$ (it is invariant under compact perturbations and equals zero for $T = 0$).

**Why this matters for integral equations.** Equations of the form $\lambda f - Kf = g$ (second-kind Fredholm equations), where $K$ is an integral operator with a nice kernel, fall under exactly this framework. Either there is a unique solution for every right-hand side $g$ (generic case), or the homogeneous equation has finitely many solutions and there are finitely many compatibility conditions on $g$. There is no continuous spectrum, no "almost solvable" regime. This is what makes second-kind integral equations dramatically nicer than first-kind ones ($Kf = g$, where the range of $K$ may not be closed and inversion is ill-posed).

**Worked example.** Consider $\lambda f(x) - x\int_0^1 y f(y)\,dy = g(x)$ on $L^2[0,1]$. The operator $Tf(x) = x\int_0^1 yf(y)\,dy$ is rank-1 (its image is the span of the function $x$). The eigenvalue equation $Tf = \mu f$ forces $f = cx/\mu$ for some constant $c$, and substituting gives $c = c/(3\mu)$, so $\mu = 1/3$. Thus $T$ has one nonzero eigenvalue $\mu = 1/3$ with eigenfunction $f(x) = x$. By Fredholm: for $\lambda \neq 0$ and $\lambda \neq 1/3$, the equation has a unique solution. At $\lambda = 1/3$, solvability requires $g \perp \ker(T^* - \frac{1}{3}I)$. Since $T^*h(y) = y\int_0^1 xh(x)\,dx$ (same structure), the eigenfunction of $T^*$ at $1/3$ is also $h(y) = y$. So the compatibility condition is $\int_0^1 y\,g(y)\,dy = 0$.

The elegance of Fredholm theory is that the entire solvability analysis of the integral equation reduces to a finite-dimensional eigenvalue computation. The operator $T$ is infinite-dimensional, but its spectral behavior at any nonzero $\lambda$ is controlled by a finite-dimensional kernel. This is the sense in which compact operators are "essentially finite-dimensional" -- not that they ARE finite-rank, but that their failure to be invertible is always finite-dimensional in nature.

A more physically motivated example: the Neumann series. For $\|\lambda^{-1}T\| < 1$ (i.e., $|\lambda| > \|T\|$), the resolvent $(\lambda I - T)^{-1} = \lambda^{-1}\sum_{n=0}^\infty (\lambda^{-1}T)^n$ converges geometrically. This is the "Born series" in scattering theory -- each term represents a higher-order scattering event. For $|\lambda| \leq \|T\|$, the series may diverge, but the Fredholm alternative still guarantees that the resolvent exists except at the (at most countably many) eigenvalues. The spectrum of a compact operator is thus entirely composed of eigenvalues (plus possibly zero) -- there is no continuous spectrum, no residual spectrum at nonzero points.


## Singular Values, Hilbert-Schmidt, and Trace Class

For non-self-adjoint compact operators, the spectral theorem does not directly apply, but the **singular value decomposition** provides a complete structural description. For any compact $T: H \to H$, the operator $T^*T$ is compact, self-adjoint, and positive. Applying the spectral theorem to $T^*T$ gives eigenvalues $s_1^2 \geq s_2^2 \geq \ldots \geq 0$ tending to zero. The **singular values** are $s_n = \sqrt{\lambda_n(T^*T)}$, and the SVD is:

![Singular value decomposition for compact operators](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig07_svd.png)


![Hilbert-Schmidt operator as matrix with square-summable entries](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig07_hilbert_schmidt.png)


$$T = \sum_n s_n \langle \cdot, v_n \rangle u_n$$

where $(v_n)$ are eigenvectors of $T^*T$ and $u_n = Tv_n/s_n$. This is the infinite-dimensional twin of the matrix SVD, and it gives a complete description of "how $T$ stretches and rotates space." The vectors $v_n$ are the "input directions" (right singular vectors), $u_n$ are the "output directions" (left singular vectors), and $s_n$ is the "stretching factor" along the $n$-th direction. A compact operator stretches along countably many orthogonal directions, with stretching factors converging to zero -- it compresses most of the space into negligible dimensions.

The SVD is the natural tool for understanding what information a compact operator preserves and what it destroys. If $s_n$ decays rapidly (exponentially), the operator effectively maps into a low-dimensional space -- most of the information in the input is lost. If $s_n$ decays slowly (polynomially), the operator preserves more information but still ultimately compresses. The "effective rank" of a compact operator -- the number of singular values above a threshold $\varepsilon$ -- quantifies its information capacity and is the key parameter in numerical low-rank approximation.

The singular values control quantitative properties:

- $\|T\|_{op} = s_1$ (operator norm = largest singular value).
- $\|T\|_{HS} = (\sum s_n^2)^{1/2}$ (Hilbert-Schmidt norm).
- $\|T\|_1 = \sum s_n$ (trace norm / nuclear norm).

![Hilbert-Schmidt operators with their integral kernel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_5_hilbert_schmidt.png)

**Hilbert-Schmidt operators** are those with $\sum s_n^2 < \infty$, equivalently $\sum_n \|Te_n\|^2 < \infty$ for any orthonormal basis. They form a Hilbert space with inner product $\langle S, T\rangle_{HS} = \sum \langle Se_n, Te_n\rangle$. For integral operators, $\|K\|_{HS} = \|k\|_{L^2}$ -- the operator is Hilbert-Schmidt iff the kernel is square-integrable. This is a stunningly clean correspondence.

**Trace-class operators** have $\sum s_n < \infty$. The **trace** $\text{tr}(T) = \sum \langle Te_n, e_n\rangle$ is well-defined, basis-independent, and satisfies $\text{tr}(AB) = \text{tr}(BA)$. **Lidskii's theorem** says the trace equals the sum of eigenvalues: $\text{tr}(T) = \sum \lambda_n$. In quantum mechanics, density matrices (mixed states) are positive trace-class operators with $\text{tr}(\rho) = 1$, and the von Neumann entropy $S(\rho) = -\text{tr}(\rho\log\rho)$ requires trace-class structure.

**The hierarchy is strict.** The diagonal operator $D$ on $\ell^2$ with entries $d_n$:
- $d_n = 1/n$: compact (entries $\to 0$), Hilbert-Schmidt ($\sum 1/n^2 = \pi^2/6$), NOT trace class ($\sum 1/n = \infty$).
- $d_n = 1/n^2$: trace class ($\sum 1/n^2 < \infty$), with $\text{tr}(D) = \pi^2/6$.
- $d_n = 1/\sqrt{n}$: compact, NOT Hilbert-Schmidt ($\sum 1/n = \infty$).

Each level has a different geometric meaning, and the distinction matters in applications: Hilbert-Schmidt is the "generic" compact operator (most integral operators with reasonable kernels), trace class is the "well-behaved" compact operator (needed for traces, determinants, and quantum statistical mechanics).

The duality between these operator ideals mirrors the sequence-space duality $\ell^1 \subset \ell^2 \subset c_0$: trace class operators form the predual of $B(H)$ (every normal linear functional on $B(H)$ is $T \mapsto \text{tr}(\rho T)$ for a unique trace-class $\rho$), Hilbert-Schmidt operators form a Hilbert space, and compact operators form the "vanishing at infinity" ideal (singular values tend to zero). The $C^*$-algebraic structure of $B(H)$ -- with compact operators as the unique closed two-sided ideal in any infinite-dimensional Hilbert space -- makes these classifications intrinsic rather than basis-dependent.

In quantum information theory, the trace-norm distance $\|\rho - \sigma\|_1 = \text{tr}|\rho - \sigma|$ between density matrices quantifies distinguishability of quantum states (it equals twice the maximum success probability of distinguishing $\rho$ from $\sigma$ minus one). The Hilbert-Schmidt distance $\|\rho - \sigma\|_{HS}$ is computationally easier but physically less meaningful. The operator norm $\|\rho - \sigma\|_{op}$ gives the maximum eigenvalue difference. Each norm captures a different operational meaning, and the hierarchy trace $\geq$ HS $\geq$ operator reflects the ordering of physical distinguishability criteria.


## Compact Operators and Inverse Problems

A practical application that shows why compact operators are simultaneously useful and dangerous. Many inverse problems take the form: given measured data $g$, recover the underlying signal $f$ from $Tf = g$, where $T$ is a compact forward operator. Examples: deblurring (convolution with a PSF), X-ray tomography (Radon transform), heat equation backward in time (evolution operator).

The compactness of $T$ is **bad news for inversion**. Since the singular values $s_n \to 0$, the formal inverse $T^{-1}g = \sum s_n^{-1}\langle g, u_n\rangle v_n$ amplifies high-frequency components catastrophically: noise in $g$ at frequency $n$ is amplified by $1/s_n \to \infty$. This is **ill-posedness** -- small errors in data produce large errors in reconstruction.

The standard fix is **Tikhonov regularization**: replace $T^{-1}$ by $(T^*T + \alpha I)^{-1}T^*$ for some $\alpha > 0$. In the SVD basis:

$$T_\alpha^{-1} g = \sum_n \frac{s_n}{s_n^2 + \alpha} \langle g, u_n\rangle v_n.$$

The filter factor $s_n/(s_n^2 + \alpha) \approx 1/s_n$ for $s_n \gg \sqrt{\alpha}$ (trust the data in well-resolved directions) and $\approx s_n/\alpha \to 0$ for $s_n \ll \sqrt{\alpha}$ (dampen the noise in poorly-resolved directions). The parameter $\alpha$ controls the resolution-stability tradeoff.

This is not merely an engineering trick -- it has a precise operator-theoretic interpretation. Tikhonov regularization solves $\min_f \|Tf - g\|^2 + \alpha\|f\|^2$, which by Lax-Milgram (Article 3) has a unique solution in any Hilbert space. As $\alpha \to 0$, the regularized solution converges to the minimum-norm least-squares solution of $Tf = g$ (when the equation is consistent). The convergence rate depends on the source condition (how smooth $f$ is relative to the singular-value decay of $T$) and is the subject of regularization theory.

The slogan: **compact operators smooth, and smoothing is hard to invert**. The singular value decay rate of $T$ quantifies the degree of ill-posedness. Mildly ill-posed problems (polynomial decay $s_n \sim n^{-\alpha}$, like numerical differentiation with $\alpha = 1$) can be regularized effectively -- the achievable resolution degrades polynomially with noise level. Severely ill-posed problems (exponential decay $s_n \sim e^{-cn}$, like backward heat equation or analytic continuation) resist regularization -- the achievable resolution degrades only logarithmically with noise level, meaning even modest noise destroys most information.

Modern approaches to inverse problems (compressed sensing, total-variation regularization, deep learning priors) all operate within this SVD framework. They differ in the choice of penalty (sparsity in a wavelet basis, bounded variation, learned neural-network prior) but share the structural diagnosis: compact operators erase information about high-frequency components, and reconstruction requires prior assumptions to replace what was lost. The spectral theorem for compact self-adjoint operators, applied to $T^*T$, is the mathematical infrastructure underlying the entire field of inverse problems.


## Compactness Criteria and the Rellich-Kondrachov Embedding

How does one prove a specific operator is compact in practice? Three standard routes:

**(a) Arzela-Ascoli.** Show the image of the unit ball is uniformly bounded and equicontinuous. Works for integral operators with continuous kernels mapping into $C(K)$.

**(b) Norm limit of finite-rank operators.** Find a sequence of explicit finite-rank approximants converging in operator norm. For Hilbert-Schmidt operators, truncate the kernel expansion.

**(c) Compact embedding.** Show $T$ factors through a compact inclusion. If $T: X \to Z$ factors as $T = J \circ S$ where $S: X \to Y$ is bounded and $J: Y \to Z$ is a compact inclusion, then $T$ is compact.

Route (c) is the workhorse of PDE applications, and it connects compact operator theory to Sobolev embedding theory. The basic paradigm: regularity gains compactness. A function with one more derivative than strictly needed for $L^p$ membership lives in a "smaller" space ($W^{1,p}$ vs $L^p$), and the inclusion from the smaller space to the larger is compact. The extra regularity prevents oscillation at arbitrarily fine scales, which is exactly the equicontinuity condition that Arzela-Ascoli uses.

The **Rellich-Kondrachov theorem** states: for bounded Lipschitz $\Omega \subset \mathbb{R}^n$, the inclusion $W^{1,p}(\Omega) \hookrightarrow L^q(\Omega)$ is compact for $q < p^* = np/(n-p)$ when $p < n$ (the critical Sobolev exponent). In particular, $W^{1,p} \hookrightarrow L^p$ compactly. For $p > n$, the inclusion $W^{1,p} \hookrightarrow C(\bar\Omega)$ is compact (Morrey's inequality gives Holder continuity, and Arzela-Ascoli closes the argument).

The proof for $W^{1,2}(\Omega) \hookrightarrow L^2(\Omega)$ on a bounded domain proceeds by contradiction + Fourier analysis: if the embedding were not compact, there would exist a sequence $(u_n)$ bounded in $H^1$ with no $L^2$-convergent subsequence. By Banach-Alaoglu, extract $u_n \rightharpoonup u$ weakly in $H^1$. The claim is that $u_n \to u$ strongly in $L^2$ -- which uses the Fourier characterization of $H^1$ (the Fourier coefficients $\hat u_n(k)$ satisfy $\sum |k|^2|\hat u_n(k)|^2 \leq C$, so for high frequencies $|k| > N$, $\sum_{|k|>N} |\hat u_n(k)|^2 \leq C/N^2 \to 0$ uniformly in $n$, and for low frequencies a diagonal argument gives pointwise convergence of Fourier coefficients). This sketch works cleanly on the torus; the general bounded domain requires cutoffs and extension operators but the idea is identical.

In PDE existence theory, this manifests as follows. Consider the Dirichlet problem $-\Delta u = f$ on $\Omega$ with $u|_{\partial\Omega} = 0$. The solution operator $f \mapsto u$ maps $L^2(\Omega) \to H^2(\Omega) \cap H^1_0(\Omega)$, and the inclusion $H^2 \hookrightarrow L^2$ is compact (by Rellich-Kondrachov applied twice: $H^2 \hookrightarrow H^1$ compact, $H^1 \hookrightarrow L^2$ compact). So the resolvent $(-\Delta)^{-1}: L^2 \to L^2$ is a compact operator. Its spectral theorem gives the eigenvalues $1/\lambda_n$ where $\lambda_n$ are the Dirichlet eigenvalues of $-\Delta$ (with $\lambda_n \to \infty$, ensuring $1/\lambda_n \to 0$ as required for compactness). The eigenfunctions form a complete orthonormal system in $L^2(\Omega)$.

This connection -- compact resolvent implies discrete spectrum -- is one of the most important structural facts in spectral theory of differential operators. It applies whenever the domain $\Omega$ is bounded: the Laplacian (Dirichlet, Neumann, or mixed) on a bounded domain has compact resolvent and hence discrete spectrum. On unbounded domains, the resolvent may fail to be compact (the spectrum may have a continuous part), which is the mathematical content of "particles can escape to infinity."

The physical interpretation is clean: on a bounded domain, the system is "confined" (like a particle in a box), energy levels are quantized, and the resolvent is compact. On an unbounded domain, the particle can propagate freely at high energies, the spectrum has a continuous part, and the resolvent is not compact. The transition from discrete to continuous spectrum -- say, in the hydrogen atom (discrete below zero, continuous above) -- corresponds to the resolvent being compact only when restricted to a spectral subspace, and the dividing energy is the ionization threshold.

For numerical computation: the Galerkin method for eigenvalues of $-\Delta$ on $\Omega$ works by projecting onto a finite-dimensional subspace $V_h$ (e.g., piecewise polynomials on a mesh of size $h$). The computed eigenvalues $\lambda_{n,h}$ converge to the true eigenvalues $\lambda_n$ as $h \to 0$, with error bounds $|\lambda_{n,h} - \lambda_n| \leq C_n h^{2k}$ where $k$ is the polynomial degree. The compactness of the resolvent guarantees that no spurious eigenvalues appear in the limit -- every cluster point of numerical eigenvalues is a true eigenvalue, and every true eigenvalue is a cluster point. This is the spectral pollution-free property of conforming Galerkin methods for compact-resolvent operators, and it depends crucially on compactness.

**Worked example.** On $\Omega = (0, \pi)$, the Dirichlet Laplacian has eigenvalues $\lambda_n = n^2$ and eigenfunctions $e_n(x) = \sqrt{2/\pi}\sin(nx)$. The resolvent $(-\Delta)^{-1}$ maps $f = \sum c_n e_n$ to $u = \sum (c_n/n^2) e_n$. Its singular values are $1/n^2$, so it is trace class ($\sum 1/n^2 = \pi^2/6$) and in particular compact. The trace $\text{tr}((-\Delta)^{-1}) = \sum 1/n^2 = \pi^2/6$ relates to the heat trace: $\text{tr}(e^{t\Delta}) = \sum e^{-n^2 t}$, and for small $t$, $\text{tr}(e^{t\Delta}) \sim |\Omega|/(4\pi t)^{1/2}$ -- the Weyl law, connecting eigenvalue asymptotics to domain geometry.

The Weyl asymptotic law $\lambda_n \sim c_d n^{2/d}$ (for the Dirichlet Laplacian on a $d$-dimensional domain) determines the singular-value decay of the resolvent: $s_n((-\Delta)^{-1}) \sim n^{-2/d}$. In dimension $d = 1$, the resolvent is trace class ($\sum n^{-2} < \infty$). In $d = 2$, it is Hilbert-Schmidt but not trace class ($\sum n^{-1}$ diverges but $\sum n^{-2}$ converges -- wait, in 2D the Weyl law gives $\lambda_n \sim cn$, so $s_n \sim 1/n$ and $\sum 1/n$ diverges). In $d = 3$, the resolvent is compact but not Hilbert-Schmidt ($\lambda_n \sim cn^{2/3}$, so $s_n \sim n^{-2/3}$, and $\sum n^{-4/3} < \infty$ -- actually it IS Hilbert-Schmidt in 3D). The precise relationship between domain dimension, Weyl asymptotics, and Schatten-class membership of the resolvent is a beautiful interface between spectral geometry and operator ideals.


## What's Next

The spectral theorem for compact self-adjoint operators gives a complete diagonalization -- but it only works for compact operators. Most operators in mathematical physics (differential operators, multiplication operators, the position and momentum operators of quantum mechanics) are not compact. Their spectra can have continuous parts, their "eigenvectors" may not be genuine Hilbert-space elements (think of plane waves $e^{ikx}$ for the momentum operator -- not in $L^2$), and the sum $\sum \lambda_n \langle \cdot, e_n\rangle e_n$ must be replaced by an integral $\int \lambda\,dE(\lambda)$ against a spectral measure.

The next article promotes spectral theory to the general bounded self-adjoint case. The compact case reappears as the special instance where the spectral measure is purely atomic (a sum of point masses at the eigenvalues). The multiplication operator $Mf(t) = tf(t)$ on $L^2[0,1]$ exemplifies the opposite extreme: purely continuous spectrum with no eigenvalues at all, spectral measure absolutely continuous with respect to Lebesgue. Between these extremes live operators with mixed spectrum -- discrete eigenvalues below an essential-spectrum threshold, and continuous spectrum above -- which is exactly the structure of quantum-mechanical Hamiltonians (bound states below the ionization energy, scattering states above). The spectral theorem unifies all these cases into a single framework.

---

*This is Part 7 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 6 — Bounded Operators](/en/functional-analysis/06-bounded-operators/)*

*Next: [Part 8 — Spectral Theory](/en/functional-analysis/08-spectral-theory/)*
