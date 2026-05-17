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

I owe my fondness for compact operators to a small embarrassment. As an undergraduate I assumed, the way one assumes the sun will rise, that infinite-dimensional linear algebra would feel exotic everywhere. It does not. There is a wide and well-mapped suburb of operator theory in which everything one learned about symmetric matrices — eigenvalues, orthogonal eigenvectors, the spectral decomposition — comes back almost unchanged, just with a sequence of eigenvalues tailing off to zero instead of a finite list. That suburb is the world of compact operators, and the price of admission is a single condition: the operator must squeeze the unit ball into a relatively compact set.

That condition sounds technical and it is, but it is also a precise way of saying *this operator behaves nearly like a matrix*. Once that is true, almost everything follows: the spectrum is countable, nonzero eigenvalues are isolated and have finite-dimensional eigenspaces, the Fredholm alternative holds, and integral equations of the second kind become tractable. The line between matrices and infinite-dimensional operators ceases to be a wall and becomes a permeable membrane. This article is about that membrane.

![Compact operator: maps the unit ball to a precompact set](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_1_compact_def.png)

## Why "compact" Is the Right Word

In a metric space, a set is compact if every sequence has a convergent subsequence. In a normed space, the unit ball is compact if and only if the space is finite-dimensional — that is the entire content of Riesz's lemma, and it is one of the disappointments of functional analysis. The unit ball, the most natural object in the room, refuses to be compact unless we already live in finite dimensions.

So one of two things must give. Either we restrict attention to finite-dimensional spaces (and abandon Hilbert spaces, $L^2$, integral equations), or we restrict attention to operators that *create* compactness where the ambient space refuses to provide it. The second is the deal compact operators offer: even though $B_X$ is not compact, $T(B_X)$ shall be precompact. We sacrifice generality of the operator and recover the convergence-of-subsequences property we wanted in the first place.

Concretely, $T: X \to Y$ is **compact** if $\overline{T(B_X)}$ is compact in $Y$, equivalently, every bounded sequence $(x_n) \subset X$ has a subsequence $(x_{n_k})$ such that $(Tx_{n_k})$ converges in $Y$. The operator does the work of compactification that the space declined to do. Whenever I have used compact operators in practice — eigenfunction expansions for an integral kernel, Galerkin approximations, the Rellich-Kondrachov embedding inside elliptic regularity — what I have actually been using is this guarantee that bounded sequences have convergent images. Almost every clever argument in this article reduces to extracting a convergent subsequence at exactly the right moment.

A small terminological history. Riesz, in 1918, called these operators *vollstetig* (totally continuous), and the older English translation "completely continuous" still appears in books written before about 1970. The name refers to a related but technically distinct property: mapping weakly convergent sequences to norm-convergent ones. On reflexive spaces, the two properties coincide; on $\ell^1$, which is not reflexive, they part ways. The modern definition in terms of the unit ball is cleaner and is the one I will use throughout.

## Finite Rank Is the Easy Case

A linear map $T: X \to Y$ is **finite-rank** if $\dim(\text{range}(T)) < \infty$. Such a $T$ is automatically compact: bounded sets in a finite-dimensional space have compact closure (Heine-Borel applied to a vector space which, equipped with any norm, is homeomorphic to $\mathbb{R}^n$). So finite-rank $\subset$ compact, and the question of how much room there is in the gap between them is the substance of the theory.

![Finite-rank operators are compact](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_2_finite_rank.png)

The basic algebraic facts about $K(X, Y)$, the space of compact operators, all rest on the same idea: if you keep only those operators whose images of $B_X$ are precompact, that property is preserved by limits and by composition with bounded operators on either side.

1. **$K(X, Y)$ is a closed subspace of $B(X, Y)$.** If $T_n \to T$ in operator norm and each $T_n$ is compact, fix $\varepsilon > 0$ and choose $n$ with $\|T - T_n\| < \varepsilon/3$. The set $T_n(B_X)$ is precompact, so it has a finite $\varepsilon/3$-net $\{y_1, \ldots, y_N\}$. For any $x \in B_X$, pick $j$ with $\|T_n x - y_j\| < \varepsilon/3$; then $\|Tx - y_j\| \leq \|Tx - T_n x\| + \|T_n x - y_j\| < 2\varepsilon/3$. So $T(B_X)$ has a finite $\varepsilon$-net, hence is totally bounded, hence precompact (in a complete space).

2. **$K(X, Y)$ is a two-sided ideal.** If $T \in K(X, Y)$ and $S \in B(Y, Z)$, then $S \circ T$ maps $B_X$ first to a precompact set, then continuously into $Z$ — and continuous images of precompact sets are precompact. The analogous argument handles $T \circ R$ for $R \in B(W, X)$.

3. **In a Hilbert space, every compact operator is a norm-limit of finite-rank operators.** This is the so-called approximation property; it holds in $\ell^p$ and $L^p$ as well, but Per Enflo's 1973 counterexample shows it fails in some Banach spaces. For Hilbert spaces the proof is direct: given an orthonormal basis $\{e_n\}$, let $P_N$ be projection onto $\text{span}(e_1, \ldots, e_N)$. Then $T_N = P_N T$ is finite-rank, and one shows $T_N \to T$ in operator norm using that $T(B_X)$ is precompact, hence almost lives in some finite-dimensional subspace.

A small numerical example will fix ideas. Consider $T: \ell^2 \to \ell^2$ defined by $T(x_1, x_2, x_3, \ldots) = (x_1, x_2/2, x_3/3, \ldots)$, a diagonal operator with entries $1/n$. The truncations $T_N$ that zero everything past coordinate $N$ are finite-rank, and a direct computation gives $\|T - T_N\| = 1/(N+1) \to 0$. So $T$ is compact. The same argument generalizes: any diagonal operator on $\ell^2$ with entries $\lambda_n \to 0$ is compact, and the rate of decay of $\lambda_n$ governs how fast finite-rank approximations converge.

## Why This Matters: Integral Equations

Why did Fredholm and Riesz care about any of this? Because in the early 1900s, the most pressing source of new linear problems came from PDE — and specifically from rewriting boundary value problems as integral equations. If you take a problem like $-u'' = f$ on $[0, 1]$ with $u(0) = u(1) = 0$, invert the differential operator using the Green's function, and recast the boundary value problem as

$$ u(x) = \int_0^1 G(x, y) f(y) \, dy, $$

then you have replaced a differential operator (unbounded, hard) with an integral operator (bounded, often compact, much easier). When the original problem includes a perturbation, you end up with an integral equation of the form $u - K u = g$, where $K$ is compact. The Fredholm alternative — which I will state and prove later — tells you exactly when this equation is solvable. That is what motivated the abstraction in the first place. Compact operators are the right linear-algebraic shadow of integral operators.

## Examples That Will Reappear

**Example 1 (continuous kernel on $C[0,1]$).** Let $K \in C([0,1] \times [0,1])$ and define $(Tf)(x) = \int_0^1 K(x, y) f(y) \, dy$ on $C([0, 1])$ with the sup norm. Then $T$ is compact. Proof: for $\|f\|_\infty \leq 1$,

$$ |Tf(x_1) - Tf(x_2)| \leq \sup_y |K(x_1, y) - K(x_2, y)|, $$

which goes to zero uniformly in $f$ as $|x_1 - x_2| \to 0$, by uniform continuity of $K$ on the compact square. So $T(B)$ is uniformly bounded and equicontinuous; Arzela-Ascoli does the rest.

**Example 2 (Hilbert-Schmidt kernel on $L^2$).** If $K \in L^2([0,1]^2)$, the integral operator $(Tf)(x) = \int K(x, y) f(y) \, dy$ on $L^2[0, 1]$ is compact. The slick proof: approximate $K$ in $L^2$ by simple functions $K_n$ supported on finite unions of rectangles; the resulting $T_n$ are finite-rank, and a direct calculation gives $\|T - T_n\|_{op} \leq \|K - K_n\|_{L^2}$. So $T$ is a norm-limit of finite-rank operators, hence compact.

**Example 3 (multiplication is not compact).** The operator $M_g f(x) = g(x) f(x)$ on $L^2[0,1]$ for $g \in L^\infty$ is bounded with $\|M_g\| = \|g\|_\infty$, but it is almost never compact. Take $g(x) = x$. The functions $f_n(x) = \sqrt{n} \mathbf{1}_{[1 - 1/n, 1]}$ are unit vectors in $L^2$, $M_g f_n$ converges weakly to zero, but $\|M_g f_n\|_2 \to 1$, so no subsequence converges in norm. The "non-compactness" of $M_g$ is what gives multiplication operators their continuous spectrum, a story for the next article.

These three examples mark the basic territory: integral operators with reasonably regular kernels are compact, finite-rank operators are compact, but multiplication operators are not. Almost every compactness argument in PDE traces back to one of the first two.

## Spectral Theorem for Compact Self-Adjoint Operators

Now to the result that pays for the whole machinery. Let $H$ be a Hilbert space and $T: H \to H$ compact and self-adjoint, meaning $\langle Tx, y \rangle = \langle x, Ty \rangle$. The theorem says $T$ behaves exactly like a symmetric matrix with eigenvalues going to zero.

![Spectral theorem for compact self-adjoint operators](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_3_spectral_compact.png)

**Theorem.** Let $T \in K(H)$ be self-adjoint. Then:

1. The spectrum $\sigma(T)$ is countable, with $0$ as the only possible accumulation point.
2. Every nonzero $\lambda \in \sigma(T)$ is an eigenvalue with finite-dimensional eigenspace.
3. Eigenspaces for distinct eigenvalues are orthogonal.
4. There is an orthonormal basis $\{e_n\}$ for $\overline{\text{range}(T)} = (\ker T)^\perp$ consisting of eigenvectors of $T$, and

$$ Tx = \sum_n \lambda_n \langle x, e_n \rangle e_n \quad \text{for every } x \in H, $$

where $\lambda_n \to 0$ if there are infinitely many.

*Proof sketch.* Step 1: eigenvalues are real, because $\lambda \|e\|^2 = \langle Te, e \rangle = \overline{\langle Te, e \rangle} = \overline{\lambda} \|e\|^2$. Step 2: eigenvectors for distinct eigenvalues are orthogonal, by the same self-adjointness trick. Step 3, the heart of the proof: at least one of $\pm \|T\|$ is an eigenvalue. Set $\mu = \sup_{\|x\|=1} |\langle Tx, x \rangle|$, which equals $\|T\|$ for self-adjoint $T$ (use polarization). Pick $x_n$ with $\langle Tx_n, x_n \rangle \to \pm \mu$ and call this limit $\lambda$. Then

$$ \|Tx_n - \lambda x_n\|^2 = \|Tx_n\|^2 - 2\lambda \langle Tx_n, x_n \rangle + \lambda^2 \leq 2\mu^2 - 2\lambda \langle Tx_n, x_n\rangle \to 0, $$

so $Tx_n - \lambda x_n \to 0$. By compactness, pass to a subsequence with $Tx_{n_k} \to y$. Then $x_{n_k} \to y/\lambda$ (assuming $\lambda \neq 0$, which is fine if $T \neq 0$), and $T(y/\lambda) = y$, so $y$ is an eigenvector with eigenvalue $\lambda$.

Step 4: iterate. Restrict $T$ to the orthogonal complement of the first eigenspace, which is $T$-invariant by self-adjointness and still compact, and apply Step 3 again. Continue. Either the process terminates (finite-dimensional range) or you generate orthonormal eigenvectors $e_n$ with eigenvalues $|\lambda_n|$ decreasing. The compactness of $T$ forces $\lambda_n e_n = T e_n$ to have a convergent subsequence; since $\{e_n\}$ is orthonormal, $\|e_n - e_m\| = \sqrt{2}$ for $n \neq m$, so the only way $T e_n$ can converge along a subsequence is if $\lambda_n \to 0$. $\square$

The proof is short because it reuses the same trick three times: extract a subsequence using compactness, exploit self-adjointness to keep things real and orthogonal, push through.

**Why is this remarkable?** In finite dimensions, every symmetric matrix is diagonalizable; this is the bread and butter of first-year linear algebra. In infinite dimensions, a generic bounded self-adjoint operator has *no eigenvalues at all* — multiplication by $x$ on $L^2[0, 1]$ is the standard example, with continuous spectrum $[0, 1]$ and an empty point spectrum. Compactness is exactly what restores diagonalizability. A compact self-adjoint operator is, structurally, an infinite diagonal matrix with diagonal tending to zero; nothing more, nothing less.

**Min-max characterization.** Order the positive eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots > 0$. The Courant-Fischer principle says

$$ \lambda_n = \max_{\dim V = n} \min_{x \in V, \|x\|=1} \langle Tx, x \rangle. $$

This is the basis of every numerical method for computing eigenvalues by minimizing Rayleigh quotients, and it is the abstract statement behind the variational principles physicists use to estimate ground-state energies. When you read someone bounding the lowest eigenvalue of a Schrödinger operator from above by plugging in a trial wavefunction, that is min-max in action.

**A worked numerical example.** Take $K(x, y) = \min(x, y) - xy$ on $[0, 1]^2$ — the covariance of the Brownian bridge. The associated integral operator on $L^2[0, 1]$ is compact and self-adjoint. Setting up the eigenvalue equation $Tf = \lambda f$ and differentiating twice, one finds $-f'' = (1/\lambda) f$ with $f(0) = f(1) = 0$. The solutions are $f_n(x) = \sqrt{2} \sin(n\pi x)$ with eigenvalues $\lambda_n = 1/(n^2 \pi^2)$. Numerically, $\lambda_1 \approx 0.1013$, $\lambda_2 \approx 0.0253$, $\lambda_3 \approx 0.0113$, with $\sum \lambda_n = 1/6$ (matching $\int_0^1 K(x, x) \, dx = \int_0^1 x(1-x) \, dx = 1/6$, by Mercer's theorem). This is also where the **Karhunen-Loève expansion** of the Brownian bridge comes from in probability, so the small mathematical detour pays a dividend in stochastic analysis.

## Eigenvalue Decay and Mercer's Theorem

The eigenvalues of a compact self-adjoint operator have to accumulate at zero, but the rate of decay is itself a meaningful quantity. For the Brownian bridge above, $\lambda_n \asymp n^{-2}$. For an operator with smoother kernel, the decay is faster — for an analytic kernel, exponentially fast. The decay rate is what governs the practical question of *how many terms you need in the eigenfunction expansion to approximate the operator to a given accuracy*.

![Eigenvalues of a compact operator accumulating only at zero](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_6_eigenvalue_decay.png)

**Mercer's theorem** is the analytic statement that ties this all together. If $K$ is continuous, symmetric, and positive semidefinite on $[0, 1]^2$ — meaning $\iint K(x, y) f(x) f(y) \, dx \, dy \geq 0$ — then the spectral expansion of the integral operator gives

$$ K(x, y) = \sum_{n=1}^\infty \lambda_n e_n(x) e_n(y), $$

with the convergence absolute and uniform on $[0, 1]^2$. This is the analytic version of "$K$ equals $\sum \lambda_n |e_n\rangle \langle e_n|$" you would write in Dirac notation, and it is the technical foundation of the **kernel methods** used in machine learning: every positive definite kernel arises from a feature map $\phi: x \mapsto (\sqrt{\lambda_n} e_n(x))_n$ into a Hilbert space, and the inner product in feature space is exactly $K(x, y)$. Reproducing kernel Hilbert spaces are Mercer's theorem, professionalized.

## The Fredholm Alternative

This is the second crown jewel. It is the infinite-dimensional version of "for a square matrix $A$, either $Ax = b$ has a unique solution for every $b$, or the homogeneous equation $Ax = 0$ has nontrivial solutions and $Ax = b$ is solvable iff $b$ lies in the orthogonal complement of $\ker A^T$." The Fredholm alternative says the same thing for operators of the form $\lambda I - T$, where $T$ is compact and $\lambda \neq 0$.

![Fredholm alternative for compact perturbations of identity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_4_fredholm.png)

**Theorem (Fredholm Alternative).** Let $T: H \to H$ be compact on a Hilbert space (the result also holds in Banach spaces with appropriate dual replacements), and let $\lambda \neq 0$. Exactly one of the following holds:

**(A)** $(\lambda I - T)x = y$ has a unique solution for every $y$, and $(\lambda I - T)^{-1}$ is bounded.

**(B)** $(\lambda I - T)x = 0$ has nontrivial solutions, $\dim \ker(\lambda I - T) < \infty$, and $(\lambda I - T)x = y$ is solvable if and only if $y \perp \ker(\overline{\lambda} I - T^*)$.

Three claims in the proof. *First*, $\ker(\lambda I - T)$ is finite-dimensional: on this subspace $T$ acts as multiplication by $\lambda$, so its action on the unit ball is the ball of radius $|\lambda|$; if the kernel were infinite-dimensional, this ball would not be precompact, contradicting compactness of $T$. *Second*, $\text{range}(\lambda I - T)$ is closed, by the standard splitting argument on whether the relevant sequence is bounded, applying compactness to extract a subsequence in either case. *Third*, the **Fredholm index** $\dim \ker(\lambda I - T) - \dim \text{coker}(\lambda I - T)$ is zero — it is invariant under compact perturbations and clearly equals zero for $T = 0$, so it equals zero for every compact $T$. So the kernel and cokernel have equal dimensions, and either both are zero (Case A) or both are positive (Case B).

The reason this is structurally important: integral equations of the second kind, $\lambda f - Kf = g$ with $K$ compact, fall under exactly this scenario. Either there is a unique solution for every right-hand side, or the homogeneous equation has finitely many solutions and there are finitely many compatibility conditions on $g$. There is no third option, no continuous spectrum, no "almost solvable" cases. This is what makes second-kind integral equations dramatically nicer than first-kind ones (where $K f = g$ requires $g$ to lie in the range of $K$, which is generally not closed).

**A worked example.** Consider $\lambda f(x) - x \int_0^1 y f(y) \, dy = g(x)$ on $L^2[0, 1]$. The integral operator $Tf = x \int yf \, dy$ is rank-1 (its image is the line $\mathbb{R} \cdot x$). The eigenvalue equation $Tf = \lambda f$ forces $f(x) = (c/\lambda) x$ for some constant $c = \int y f(y) \, dy$, and substituting gives $c = (c/\lambda) \int y^2 \, dy = c/(3\lambda)$, so $\lambda^2 = 1/3$, $\lambda = \pm 1/\sqrt{3}$. For every $\lambda \notin \{0, \pm 1/\sqrt{3}\}$ the equation has a unique solution; at $\lambda = \pm 1/\sqrt{3}$ we are in case (B), and the equation is solvable only for $g$ orthogonal to a one-dimensional subspace. The whole behavior of an integral equation is encoded in two real numbers.

## Examples Beyond Self-Adjoint: Singular Values

Most compact operators we meet in practice are not self-adjoint. The replacement is the **singular value decomposition**. For any compact $T: H \to H$, the operator $T^* T$ is compact, self-adjoint, and positive, so by the spectral theorem above we can diagonalize it: $T^* T e_n = s_n^2 e_n$ with $s_n \geq 0$ decreasing to zero. The numbers $s_n$ are the **singular values** of $T$, and we have

$$ T = \sum_n s_n \langle \cdot, v_n \rangle u_n, $$

where $v_n$ are the eigenvectors of $T^*T$ and $u_n = T v_n / s_n$ when $s_n \neq 0$. This is the infinite-dimensional twin of the matrix SVD.

![Classical compact operators: integral operators on L^2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_7_compact_examples.png)

The singular values control everything quantitative about $T$:

- $\|T\|_{op} = s_1$ (the largest singular value).
- $\|T\|_{HS} = (\sum s_n^2)^{1/2}$ (Hilbert-Schmidt or Frobenius norm).
- $\|T\|_1 = \sum s_n$ (trace norm or nuclear norm).

Each of these is a Banach-space norm on a subset of compact operators, and they form a nested hierarchy: trace class $\subsetneq$ Hilbert-Schmidt $\subsetneq$ compact. Most mathematical-physics calculations with operators are best phrased in terms of these norms, because they correspond to physical quantities — Hilbert-Schmidt norm to total $L^2$ mass of the kernel, trace norm to expected total magnitude.

## Hilbert-Schmidt and Trace Class

Two specific subclasses deserve their own page.

**Hilbert-Schmidt operators.** $T \in B(H)$ is Hilbert-Schmidt if $\|T\|_{HS}^2 = \sum_n \|T e_n\|^2 < \infty$ for some (equivalently, every) orthonormal basis. Properties: $\|T\|_{op} \leq \|T\|_{HS}$, every Hilbert-Schmidt operator is compact, the HS operators form a Hilbert space themselves with inner product $\langle S, T \rangle_{HS} = \sum_n \langle S e_n, T e_n \rangle$. For integral operators on $L^2$, $\|T\|_{HS}$ equals $\|K\|_{L^2}$ — the kernel is square-integrable iff the operator is Hilbert-Schmidt. This is a stunningly clean correspondence, and it is why $L^2$ kernels are the right setting for many calculations.

![Hilbert-Schmidt operators with their integral kernel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_v2_07_5_hilbert_schmidt.png)

**Trace class.** $T \in B(H)$ is trace class if $\sum_n \langle |T| e_n, e_n \rangle < \infty$, where $|T| = (T^* T)^{1/2}$. Equivalently, $\sum s_n < \infty$. The trace $\text{tr}(T) = \sum_n \langle T e_n, e_n \rangle$ is then well-defined and basis-independent, and **Lidskii's theorem** says it equals $\sum \lambda_n$ where $\lambda_n$ are the eigenvalues of $T$ counted with multiplicity. The cyclic property $\text{tr}(AB) = \text{tr}(BA)$ holds whenever $A$ is trace class and $B$ is bounded.

A small example to clarify the hierarchy. The diagonal operator on $\ell^2$ with entries $1/n$ is compact (entries go to zero), Hilbert-Schmidt ($\sum 1/n^2 = \pi^2/6 < \infty$), but **not** trace class ($\sum 1/n$ diverges). The operator with entries $1/n^2$ is trace class, with trace $\pi^2/6$. The hierarchy is strict and each level has a different geometric meaning.

**Why mathematical physics cares.** In quantum statistical mechanics, the density matrix $\rho$ describing a mixed state is a positive trace-class operator with $\text{tr}(\rho) = 1$. The von Neumann entropy is $S(\rho) = -\text{tr}(\rho \log \rho)$, well-defined precisely because $\rho$ is trace class. In quantum information theory, this operator-theoretic framework is not optional — it is the language. The fact that compact operators give us a clean spectral decomposition, that trace class gives us a finite trace, and that Hilbert-Schmidt gives us an $L^2$-like inner product on operators is what makes the whole quantum-information dictionary work.

## Compactness in PDE: The Rellich-Kondrachov Embedding

I want to flag one PDE-flavored compactness result, because it shows up everywhere in elliptic regularity. The **Rellich-Kondrachov theorem** states that for a bounded domain $\Omega \subset \mathbb{R}^n$ with Lipschitz boundary, the inclusion $W^{1, p}(\Omega) \hookrightarrow L^p(\Omega)$ is compact for $1 \leq p < \infty$. In words, sequences bounded in the Sobolev norm have $L^p$-convergent subsequences. This is a structural fact about how regularity gains compactness, and it is what allows Galerkin approximations to converge, weak solutions to elliptic equations to exist, and the Lax-Milgram framework to have teeth.

I will return to this in article 12. For now: every time you see "by Rellich, we may pass to a subsequence converging strongly," the underlying machinery is the same compactness we have been studying — just applied to inclusion maps between Sobolev spaces.

## A Numerical Excursion: Diagonalizing a 5x5 Matrix as a Toy Compact Operator

Before proclaiming the chasm between finite and infinite dimensions bridged, it is worth running through a small numerical example by hand. Consider the symmetric matrix

$$ A = \begin{pmatrix} 4 & 1 & 0 & 0 & 0 \\ 1 & 3 & 1 & 0 & 0 \\ 0 & 1 & 2 & 1 & 0 \\ 0 & 0 & 1 & 1 & 1 \\ 0 & 0 & 0 & 1 & 0 \end{pmatrix}, $$

a tridiagonal matrix with diagonal $(4, 3, 2, 1, 0)$. Its eigenvalues, computed numerically, are approximately $\{4.541, 3.265, 2.0, 0.735, -0.541\}$ — they interlace the diagonal entries, by Cauchy's interlacing theorem, and the smallest one even slipped below zero. Now imagine the same construction, but extended to an infinite tridiagonal operator on $\ell^2$ with diagonal $(1, 1/2, 1/3, \ldots) \to 0$ and off-diagonal entries also tending to zero. The operator is compact, the eigenvalues likewise tend to zero, and the spectral theorem gives us an orthonormal basis of eigenvectors. The picture is *exactly the same as the matrix case*, only the index set extends to infinity. The compact-operator story is the matrix story, told at infinity.

The reason it is worth doing this exercise is that one's intuition for compact operators should always be calibrated against finite tridiagonal or pentadiagonal matrices. Whenever a statement about compact self-adjoint operators sounds startling, drop down to the finite case and check what it says about a $5 \times 5$ symmetric matrix. Almost always the finite version is something everyone knows, and the infinite version is just the corresponding matrix statement plus "and the eigenvalues go to zero." The whole cleverness is in the proof that this is enough.

## A Word on the Approximation Property

I mentioned that on $\ell^p$ and $L^p$ spaces, every compact operator is the operator-norm limit of finite-rank operators. This is the **approximation property**. It holds in Hilbert spaces (trivially, by truncating to finite-dimensional projections of an orthonormal basis), and it holds in $\ell^p$, $L^p$, and most spaces one meets in practice. But it does not hold in every Banach space.

Per Enflo's 1973 construction of a separable Banach space without the approximation property was a tour de force — a long and difficult example, refined later by Davie, Figiel-Johnson, Szankowski, and others. The example matters because it shows that the seemingly innocent "compact equals limit of finite-rank" is not a theorem of pure soft analysis; it depends on a structural feature of the space that some Banach spaces lack. Operators on Enflo's space can still be compact in the sense of mapping bounded sets to precompact sets, but they cannot all be approximated by finite-rank operators in operator norm.

For us, this is a footnote rather than a working concern: every space that arises in PDE, in mathematical physics, or in machine learning has the approximation property. But it is good to know that the theorem about Hilbert space generalization is not as cheap as it looks.

## Compact Operators on Specific Spaces, Briefly

Different ambient spaces give compact operators slightly different flavors.

- **On $\ell^p$ for $1 \leq p < \infty$.** Compact operators are limits of finite-rank operators (approximation property). Compact operators on $\ell^2$ correspond to "infinite matrices" $(a_{ij})$ whose singular values tend to zero — a beautiful but slightly slippery characterization, since not every double-indexed sequence of complex numbers yields a bounded operator.
- **On $C([0, 1])$.** Compact operators include integral operators with continuous kernels (by Arzela-Ascoli, our Example 1) but also less obvious examples like Volterra operators $(Vf)(x) = \int_0^x K(x, y) f(y) \, dy$ with continuous $K$, which have spectrum $\{0\}$ and yet are nonzero — a striking instance of a compact operator with no nonzero eigenvalues.
- **On $L^p(\Omega)$.** The Rellich-Kondrachov theorem makes inclusions of Sobolev spaces compact, which is the workhorse of elliptic theory. Without it, half the existence theorems for elliptic PDE collapse.

The Volterra operator deserves a small spotlight. On $L^2[0, 1]$, $V f(x) = \int_0^x f(y) \, dy$. It is compact (it is even Hilbert-Schmidt: its kernel is $K(x, y) = \mathbf{1}_{y \leq x}$, with $\|K\|_{L^2}^2 = 1/2$). Its only spectral value is $0$: $\sigma(V) = \{0\}$. But $0$ is not an eigenvalue (since $\int_0^x f = 0$ for all $x$ forces $f = 0$). So $V$ is a compact operator whose spectrum is a single point that is not an eigenvalue — a sharp warning that the spectral theorem for compact self-adjoint operators uses self-adjointness in an essential way. Take it away and you can lose all the nonzero eigenvalues. The Volterra operator is the canonical "compact but not normal" example, and it shows up whenever one wants to construct a counterexample in operator theory.

## Summary of the Hierarchy

| Class | Definition | Spectral characterization |
|---|---|---|
| Finite-rank | $\dim(\text{range}) < \infty$ | Finitely many nonzero eigenvalues |
| Trace class | $\sum s_n < \infty$ | $\sum |\lambda_n| < \infty$ |
| Hilbert-Schmidt | $\sum s_n^2 < \infty$ | $\sum |\lambda_n|^2 < \infty$ |
| Compact | $\overline{T(B_H)}$ compact | $\lambda_n \to 0$ |
| Bounded | $\|T\| < \infty$ | $\sigma(T)$ bounded |

Each inclusion is strict. The spectral characterization on the right-hand side is for self-adjoint operators; for general compact operators, replace $\lambda_n$ with singular values $s_n$.

## Compact Operators in Inverse Problems

A practical detour. Many inverse problems in physics and imaging take the form: given data $g$, recover the underlying $f$ from the equation $T f = g$, where $T$ is a compact operator (a forward model: the way the world maps unknown $f$ to measured $g$). The compactness is bad news in disguise: because $T$ smooths things out (singular values $s_n \to 0$), inverting $T$ amplifies high-frequency components of the data — whichever components $T$ damped most. Specifically, if $T = \sum s_n \langle \cdot, v_n \rangle u_n$, then formally $T^{-1} g = \sum s_n^{-1} \langle g, u_n \rangle v_n$, and the $1/s_n$ factors blow up. Small noise in $g$ becomes huge noise in $f$. This is **ill-posedness**, and it is the reason X-ray reconstruction, deblurring, and MRI inverse problems are difficult.

The standard fix is **regularization**: replace $T^{-1}$ with $(T^* T + \alpha I)^{-1} T^*$ for some $\alpha > 0$ (Tikhonov regularization), which has a clean spectral interpretation as $\sum \frac{s_n}{s_n^2 + \alpha} \langle g, u_n \rangle v_n$. The factor $s_n / (s_n^2 + \alpha)$ approximates $1/s_n$ for $s_n \gg \sqrt{\alpha}$ and tames it to roughly $s_n / \alpha$ for $s_n \ll \sqrt{\alpha}$. So we trust the directions where $T$ is well-behaved and dampen the directions where it is not. Choosing $\alpha$ is the central tradeoff in the field. None of this would be possible without the singular value decomposition, which is a direct consequence of the spectral theorem for compact self-adjoint operators applied to $T^* T$.

If you want one slogan: **compact operators are smoothing operators, and smoothing operators are hard to invert**. Most of the modern theory of regularization, from Tikhonov in the 1960s through Donoho's wavelet shrinkage in the 1990s through deep image priors today, is operator theory wearing applied clothes.

## Compactness via Total Boundedness, in Practice

How does one show in practice that a particular operator is compact? Three workhorse criteria, in increasing generality:

**(a) Show the image of the unit ball is uniformly bounded and equicontinuous.** This is Arzela-Ascoli, and it works for integral operators with continuous kernels, integral operators with kernels in $C^k$, and many convolution operators on $C(K)$ for $K$ compact. The key calculation is a modulus-of-continuity estimate showing $|Tf(x_1) - Tf(x_2)| \to 0$ uniformly in $f$ as $x_1 \to x_2$.

**(b) Show $T$ is the operator-norm limit of finite-rank operators.** This works for any operator on a Hilbert space (or on any space with the approximation property) once you have a candidate sequence of finite-rank approximants. For integral operators, the standard move is to approximate the kernel: write $K = \sum_n c_n \phi_n(x) \psi_n(y)$ as a Hilbert-Schmidt expansion, and truncate to the first $N$ terms.

**(c) Show $T(B_X)$ has a finite $\varepsilon$-net for every $\varepsilon > 0$.** This is total boundedness, and it is the root definition of relative compactness in a complete metric space. Most of the time this is harder to verify directly than (a) or (b), but in some cases — operators between sequence spaces with a nice basis structure — it is the easiest route.

In practice (a) is what one uses for $C(K)$ targets and (b) for $L^2$ targets. Almost every compact operator I have encountered in PDE work was proven compact via Arzela-Ascoli or via a Hilbert-Schmidt kernel estimate. The third route is reserved for theorists.

## What's Next, and Why

The spectral theorem for compact self-adjoint operators is satisfying but it has a hard ceiling: it only works for compact operators. Most operators in mathematical physics are not compact — multiplication operators, differential operators, the position operator in quantum mechanics. None of them have the nice eigenvalue picture above. The next article promotes spectral theory to the general bounded self-adjoint case, replacing eigenvalues by a *spectral measure* and sums by integrals. The compact case will reappear as the special situation in which the spectral measure is purely atomic, a discrete sum of point masses. Most operators we care about are not so accommodating.

The conceptual move from this article to the next is: stop asking "what are the eigenvalues" and start asking "what is the spectrum, and how is the operator built out of multiplication operators on the spectrum." The answer, in three words, is *the functional calculus*. Once one has it, the difference between a compact operator (with eigenvalues going to zero) and a multiplication operator (with continuous spectrum) becomes the difference between a discrete and a continuous spectral measure — two species of the same animal. The next article is the taxonomic exercise that puts compact operators in their proper place.

There is a meta-lesson worth stating before moving on. The reason functional analysis can promote facts from finite to infinite dimensions, again and again, is that it identifies the *minimal structural condition* under which each finite-dimensional fact survives. For "diagonalizable symmetric matrix", the minimal condition is "compact self-adjoint operator." For "every linear map between finite-dimensional spaces is bounded," the minimal condition is "between Banach spaces with a closed graph." For "every continuous function on a compact set has a maximum," the minimal condition is "lower-semicontinuous coercive function on a reflexive Banach space, in the weak topology." Each of these is a translation, and the translation is rarely free. Compactness is the price one pays to keep the spectral theorem for symmetric matrices alive in the infinite-dimensional world. It is a fair price. And once paid, it gives back more than was asked: not just spectral diagonalization but the whole apparatus of singular values, trace class, Hilbert-Schmidt operators, Mercer's theorem, the Fredholm index, and the Rellich-Kondrachov compactness — each of which has paid its own dividends in pure and applied mathematics for the past century.

---

*This is Part 7 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 6 — Bounded Operators](/en/functional-analysis/06-bounded-operators/)*

*Next: [Part 8 — Spectral Theory](/en/functional-analysis/08-spectral-theory/)*
