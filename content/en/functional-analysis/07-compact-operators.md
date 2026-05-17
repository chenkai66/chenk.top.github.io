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

In the last article, we saw how completeness constrains operators between Banach spaces. Now we narrow our focus to a class of operators that behaves almost like matrices: **compact operators**. These are operators that map the (non-compact) unit ball to a relatively compact set — in other words, they "compress" infinite-dimensional sets into something nearly finite-dimensional.

The reward for studying compact operators is a spectral theory that looks remarkably like the eigenvalue theory of finite-dimensional linear algebra. The **spectral theorem for compact self-adjoint operators** gives a complete orthonormal diagonalization, and the **Fredholm alternative** — the infinite-dimensional version of the statement that for a square matrix, either $Ax = b$ has a unique solution or $Ax = 0$ has nontrivial solutions — governs the solvability of integral equations.

This article develops the theory from definition to the two main theorems, with applications to integral equations throughout.

**A word on terminology.** The term "compact operator" (or "completely continuous operator," an older term still found in some texts) reflects the historical development of the theory from the study of integral equations by Fredholm, Hilbert, and Riesz in the early 20th century. The Fredholm theory of integral equations was one of the first major successes of functional analysis, demonstrating the power of abstracting finite-dimensional linear algebra to infinite-dimensional settings. The modern definition in terms of relatively compact images of bounded sets is due to Riesz.

---

## Motivation: Approximation by Finite Rank

Recall that a linear map $T: X \to Y$ between normed spaces is **finite-rank** if $\dim(\text{range}(T)) < \infty$. Finite-rank operators are the simplest bounded operators — they essentially reduce any problem to finite-dimensional linear algebra.

But most interesting operators are not finite-rank. Consider the integral operator

$$(Tf)(x) = \int_0^1 K(x,y) f(y) \, dy$$

on $L^2([0,1])$ with a continuous kernel $K$. The range of $T$ is infinite-dimensional in general, so $T$ is not finite-rank. Yet $T$ has much better behavior than a generic bounded operator: the image $T(B)$ of the unit ball is not just bounded but **equicontinuous** (by continuity of $K$), so the Arzela-Ascoli theorem gives relative compactness.

This observation motivates the following definition.

**Historical context.** Compact operators were first studied systematically by Frigyes Riesz (1918) and later by Schauder. The name "compact" refers to the fact that the operator maps the unit ball to a set with compact closure. In older literature, the term "completely continuous" was used, which actually denotes a slightly different (though related) property: mapping weakly convergent sequences to norm-convergent sequences. For operators on reflexive spaces, the two notions coincide, but in general they differ.

---


![Compact operators spectral decomposition](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/07-compact-operators/fa_fig3_operators.png)

## Definition and First Properties

**Definition.** Let $X$ and $Y$ be Banach spaces. A bounded linear operator $T: X \to Y$ is **compact** if $T(B_X)$ is relatively compact in $Y$ (i.e., $\overline{T(B_X)}$ is compact).

Equivalently, $T$ is compact if every bounded sequence $(x_n)$ in $X$ has a subsequence $(x_{n_k})$ such that $(Tx_{n_k})$ converges in $Y$.

Denote the set of compact operators from $X$ to $Y$ by $K(X,Y)$ (or $\mathcal{K}(X,Y)$).

**Basic properties:**

1. Every finite-rank operator is compact (the image of the unit ball lies in a finite-dimensional subspace, where bounded sets are relatively compact).

2. $K(X,Y)$ is a **closed subspace** of $B(X,Y)$. Proof: if $T_n \to T$ in operator norm and each $T_n$ is compact, then for any $\varepsilon > 0$, choose $n$ with $\|T - T_n\| < \varepsilon/3$. Since $T_n(B_X)$ is relatively compact, it has a finite $\varepsilon/3$-net $\{y_1, \ldots, y_N\}$. For any $x \in B_X$, choose $y_j$ with $\|T_n x - y_j\| < \varepsilon/3$. Then $\|Tx - y_j\| < 2\varepsilon/3 < \varepsilon$. So $T(B_X)$ has a finite $\varepsilon$-net for each $\varepsilon$, giving relative compactness.

3. **Compact operators form a two-sided ideal:** if $T \in K(X,Y)$, $S \in B(Y,Z)$, and $R \in B(W,X)$, then $ST \in K(X,Z)$ and $TR \in K(W,Y)$.

4. In a Hilbert space $H$, every compact operator is the operator-norm limit of finite-rank operators. (This fails in some Banach spaces — Enflo's counterexample of a space without the approximation property.)

**Example 1 (Integral operators with continuous kernels).** Let $K \in C([0,1]^2)$ and define $T: C([0,1]) \to C([0,1])$ by $(Tf)(x) = \int_0^1 K(x,y)f(y)\,dy$. The set $T(B_{C([0,1])})$ is uniformly bounded and equicontinuous:

$$|Tf(x_1) - Tf(x_2)| \leq \int_0^1 |K(x_1,y) - K(x_2, y)| |f(y)| \, dy \leq \sup_y |K(x_1,y) - K(x_2,y)|,$$

which tends to zero uniformly over $\|f\|_\infty \leq 1$ as $|x_1 - x_2| \to 0$ (by uniform continuity of $K$ on a compact set). By the Arzela-Ascoli theorem, $T(B)$ is relatively compact.

**Example 2 (Hilbert-Schmidt operators).** Let $K \in L^2([0,1]^2)$ and define $T: L^2([0,1]) \to L^2([0,1])$ as above. Then $T$ is compact. To see this, approximate $K$ in $L^2$ by continuous (or finite-rank) kernels $K_n$, giving finite-rank operators $T_n$ with $\|T - T_n\| \leq \|K - K_n\|_{L^2} \to 0$.

More generally, if $\{e_n\}$ is an orthonormal basis for a Hilbert space $H$, and $T \in B(H)$ satisfies $\sum_{n} \|Te_n\|^2 < \infty$, then $T$ is called a **Hilbert-Schmidt operator** and is compact. The quantity $\|T\|_{HS} = (\sum_n \|Te_n\|^2)^{1/2}$ is the Hilbert-Schmidt norm, independent of the choice of basis.

**Example 3 (Diagonal operators and compactness).** Let $T: \ell^2 \to \ell^2$ be a diagonal operator, $Te_n = \lambda_n e_n$. Then $T$ is compact if and only if $\lambda_n \to 0$. The "if" direction: the finite-rank operators $T_N(e_n) = \lambda_n e_n$ for $n \leq N$, $T_N(e_n) = 0$ for $n > N$, satisfy $\|T - T_N\| = \sup_{n > N} |\lambda_n| \to 0$. The "only if" direction: if $\lambda_n \not\to 0$, there exists $\varepsilon > 0$ and a subsequence with $|\lambda_{n_k}| \geq \varepsilon$. Then $\|Te_{n_j} - Te_{n_k}\|^2 = |\lambda_{n_j}|^2 + |\lambda_{n_k}|^2 \geq 2\varepsilon^2$ for $j \neq k$, so $(Te_n)$ has no convergent subsequence.

This example gives a complete picture: compact operators on $\ell^2$ are "almost diagonal" in the sense that their singular values tend to zero.

**Non-example: the identity operator.** In an infinite-dimensional space, the identity operator $I: X \to X$ is **never** compact. The image of the unit ball is the unit ball itself, which is not compact (Riesz's lemma). This is the most fundamental non-example and explains why compactness is a genuine restriction.

**Non-example: invertible compact operators.** If $T$ is compact and invertible (i.e., $T^{-1} \in B(X)$), then $I = T^{-1}T$ is compact (as a composition of a bounded operator with a compact one), contradicting the previous non-example. Therefore, **no compact operator on an infinite-dimensional space is invertible**. This implies that $0 \in \sigma(T)$ for every compact operator on an infinite-dimensional space.

**Compact operators and weak convergence.** An equivalent characterization of compact operators on Hilbert spaces ties back to the weak topology from Article 5: a bounded operator $T: H \to H$ is compact if and only if $x_n \rightharpoonup 0$ implies $\|Tx_n\| \to 0$. This says compact operators convert weak convergence to norm convergence — they "upgrade" the mode of convergence. The proof of the forward direction uses the fact that compact operators map bounded sets to relatively compact sets, and in a relatively compact set, weak convergence and norm convergence coincide.

---

## Schauder's Theorem and the Adjoint

**Theorem (Schauder).** Let $X$ and $Y$ be Banach spaces and $T \in B(X,Y)$. Then $T$ is compact if and only if its adjoint $T^*: Y^* \to X^*$ is compact.

**Proof sketch.** ($\Rightarrow$) Suppose $T$ is compact. Let $(g_n)$ be a bounded sequence in $Y^*$. We need to show $(T^* g_n)$ has a convergent subsequence in $X^*$. The set $L = \overline{T(B_X)}$ is compact in $Y$. The restrictions $g_n|_L$ form a uniformly bounded, equicontinuous family on the compact metric space $L$ (equicontinuity follows from $|g_n(y_1) - g_n(y_2)| \leq \|g_n\| \cdot \|y_1 - y_2\| \leq M\|y_1 - y_2\|$). By Arzela-Ascoli, there is a subsequence $g_{n_k}|_L$ converging uniformly on $L$. Then for $\|x\| \leq 1$:

$$\|T^* g_{n_j} - T^* g_{n_k}\| = \sup_{\|x\| \leq 1} |g_{n_j}(Tx) - g_{n_k}(Tx)| \leq \sup_{y \in L} |g_{n_j}(y) - g_{n_k}(y)| \to 0.$$

So $(T^* g_{n_k})$ is Cauchy in $X^*$, hence convergent.

($\Leftarrow$) If $T^*$ is compact, then $T^{**}: X^{**} \to Y^{**}$ is compact (by the forward direction). The restriction of $T^{**}$ to $X \hookrightarrow X^{**}$ is $T$, and the image lies in $Y \hookrightarrow Y^{**}$ by continuity. A refinement of this argument (using that $T^{**}$ maps $B_{X^{**}}$ to a relatively compact subset of $Y^{**}$, intersecting with $Y$) completes the proof. $\square$

---

## The Spectral Theorem for Compact Self-Adjoint Operators

This is the crown jewel of the theory. Let $H$ be a Hilbert space (over $\mathbb{R}$ or $\mathbb{C}$) and $T: H \to H$ a compact self-adjoint operator ($T = T^*$, meaning $\langle Tx, y \rangle = \langle x, Ty \rangle$ for all $x, y$).

**Theorem (Spectral Theorem for Compact Self-Adjoint Operators).** Let $T: H \to H$ be a compact self-adjoint operator on a Hilbert space $H$. Then:

1. The spectrum of $T$ is a countable set (finite or a sequence converging to $0$), with $0$ as the only possible accumulation point.
2. Every nonzero $\lambda \in \sigma(T)$ is an eigenvalue with finite-dimensional eigenspace.
3. Eigenspaces for distinct eigenvalues are orthogonal.
4. There exists an orthonormal basis $\{e_n\}$ for $(\ker T)^\perp$ consisting of eigenvectors of $T$:

$$Tx = \sum_n \lambda_n \langle x, e_n \rangle e_n \quad \text{for all } x \in H,$$

where $\lambda_n \to 0$ (if the set of eigenvalues is infinite).

**Proof.**

*Step 1: Eigenvalues are real.* If $T e = \lambda e$ with $e \neq 0$, then $\lambda \|e\|^2 = \langle \lambda e, e \rangle = \langle Te, e \rangle = \langle e, Te \rangle = \overline{\lambda}\|e\|^2$, so $\lambda = \overline{\lambda}$.

*Step 2: Eigenvectors for distinct eigenvalues are orthogonal.* If $Te_1 = \lambda_1 e_1$ and $Te_2 = \lambda_2 e_2$ with $\lambda_1 \neq \lambda_2$, then $\lambda_1 \langle e_1, e_2 \rangle = \langle Te_1, e_2 \rangle = \langle e_1, Te_2 \rangle = \lambda_2 \langle e_1, e_2 \rangle$, giving $(\lambda_1 - \lambda_2)\langle e_1, e_2 \rangle = 0$, so $\langle e_1, e_2 \rangle = 0$.

*Step 3: $\|T\|$ or $-\|T\|$ is an eigenvalue.* Define $\mu = \sup_{\|x\| = 1} |\langle Tx, x \rangle|$. For self-adjoint $T$, one can show $\mu = \|T\|$ (this requires verifying that $\|T\| = \sup_{\|x\|=1} |\langle Tx, x\rangle|$ for self-adjoint operators, which follows from the polarization identity). Choose $(x_n)$ with $\|x_n\| = 1$ and $\langle Tx_n, x_n \rangle \to \pm\mu$. Set $\lambda = \pm\mu$ (whichever the sequence approaches). Then:

$$\|Tx_n - \lambda x_n\|^2 = \|Tx_n\|^2 - 2\lambda \langle Tx_n, x_n \rangle + \lambda^2 \leq 2\mu^2 - 2\lambda \langle Tx_n, x_n \rangle \to 0.$$

Since $T$ is compact and $(x_n)$ is bounded, pass to a subsequence with $Tx_{n_k} \to y$. Then $x_{n_k} = \frac{1}{\lambda}(Tx_{n_k} - (Tx_{n_k} - \lambda x_{n_k})) \to y/\lambda$, and $Ty = \lambda y$ with $\|y\| = \mu \neq 0$ (if $T \neq 0$).

*Step 4: Iterative construction.* Set $\lambda_1$ as the eigenvalue from Step 3, with eigenspace $E_1$. The restriction $T|_{E_1^\perp}$ is again compact and self-adjoint (self-adjointness of $T$ ensures $T(E_1^\perp) \subset E_1^\perp$). Repeat Step 3 on $E_1^\perp$ to find $\lambda_2$ with $|\lambda_2| \leq |\lambda_1|$. Continue inductively. Either the process terminates (finite number of eigenvalues) or we get a sequence $\lambda_n \to 0$ (since the eigenvectors $e_n$ are orthonormal and $Te_n = \lambda_n e_n$; compactness of $T$ requires $\lambda_n e_n = Te_n$ to have a convergent subsequence from the bounded set $\{e_n\}$, which forces $\lambda_n \to 0$).

*Step 5: Completeness.* Let $V = \overline{\text{span}\{e_n\}}$. Then $T|_{V^\perp}$ is compact and self-adjoint with $\|T|_{V^\perp}\| = 0$ (all eigenvalues have been exhausted), so $T|_{V^\perp} = 0$, meaning $V^\perp = \ker T$. Hence $\{e_n\}$ is an orthonormal basis for $(\ker T)^\perp$. $\square$

**Why this theorem is remarkable.** In finite dimensions, every symmetric matrix is diagonalizable — this is taught in first courses on linear algebra. In infinite dimensions, a generic bounded self-adjoint operator cannot be diagonalized (its spectrum may have no eigenvalues at all, as we will see in Article 8). The spectral theorem for compact self-adjoint operators says that compactness is precisely the condition that rescues diagonalizability: compact self-adjoint operators behave **exactly** like infinite diagonal matrices with entries tending to zero.

**The min-max characterization of eigenvalues.** The eigenvalues of a compact self-adjoint operator $T$ on a Hilbert space can be characterized variationally. Ordering the positive eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots > 0$ (with multiplicities), we have the **Courant-Fischer min-max principle**:

$$\lambda_n = \max_{\dim V = n} \min_{x \in V, \|x\| = 1} \langle Tx, x \rangle = \min_{\dim V = n-1} \max_{x \perp V, \|x\| = 1} \langle Tx, x \rangle.$$

This characterization is fundamental in numerical analysis (it justifies Rayleigh-Ritz methods for computing eigenvalues) and in physics (it provides a variational principle for energy levels in quantum mechanics).

**Example 4 (Mercer's theorem).** If $K: [0,1] \times [0,1] \to \mathbb{R}$ is a continuous, symmetric, positive semi-definite kernel (meaning $\int \int K(x,y)f(x)f(y)\,dx\,dy \geq 0$ for all $f$), then the spectral decomposition of the associated integral operator gives:

$$K(x,y) = \sum_{n=1}^\infty \lambda_n e_n(x) e_n(y),$$

where the convergence is absolute and uniform. This is **Mercer's theorem**, and it establishes the connection between positive definite kernels and feature maps that is central to kernel methods in machine learning. The reproducing kernel Hilbert space (RKHS) associated with $K$ is precisely the range of $T^{1/2}$ equipped with the inner product $\langle T^{1/2}f, T^{1/2}g \rangle_{RKHS} = \langle f, g \rangle_{L^2}$ (restricted to $(\ker T)^\perp$).

**Example 3 (Diagonalizing a Hilbert-Schmidt operator).** Consider $T: L^2([0,1]) \to L^2([0,1])$ with kernel $K(x,y) = \min(x,y) - xy$. This is the covariance kernel of the Brownian bridge. It is symmetric and continuous, hence $T$ is a compact self-adjoint operator. The eigenvalue equation $Tf = \lambda f$ becomes the boundary value problem $-f'' = \mu f$, $f(0) = f(1) = 0$ (where $\mu = 1/\lambda$), with solutions $f_n(x) = \sqrt{2}\sin(n\pi x)$ and $\lambda_n = 1/(n^2\pi^2)$. The spectral decomposition is:

$$(Tf)(x) = \sum_{n=1}^\infty \frac{1}{n^2 \pi^2} \langle f, \sqrt{2}\sin(n\pi \cdot) \rangle \sqrt{2}\sin(n\pi x).$$

---

## The Fredholm Alternative

The Fredholm alternative is the infinite-dimensional generalization of the fundamental theorem of linear algebra for square matrices: $Ax = b$ is solvable if and only if $b$ is orthogonal to the null space of $A^T$.

**Theorem (Fredholm Alternative).** Let $T: H \to H$ be a compact operator on a Hilbert space $H$ (or more generally on a Banach space), and let $\lambda \neq 0$. Then exactly one of the following holds:

**(A)** The equation $(\lambda I - T)x = y$ has a unique solution for every $y \in H$. In this case, $(\lambda I - T)^{-1}$ is bounded.

**(B)** The equation $(\lambda I - T)x = 0$ has nontrivial solutions. In this case, $\dim \ker(\lambda I - T) < \infty$, and the equation $(\lambda I - T)x = y$ is solvable if and only if $y \perp \ker(\overline{\lambda} I - T^*)$.

**Proof sketch for the Hilbert space case.**

*Claim 1: $\ker(\lambda I - T)$ is finite-dimensional.* The restriction of $T$ to $\ker(\lambda I - T)$ is $\lambda I$, which maps the unit ball to a ball of radius $|\lambda|$. If $\ker(\lambda I - T)$ were infinite-dimensional, this ball would not be relatively compact, contradicting compactness of $T$.

*Claim 2: $\text{range}(\lambda I - T)$ is closed.* Suppose $(\lambda I - T)x_n \to y$. We may assume $x_n \perp \ker(\lambda I - T)$ (by projecting). If $(x_n)$ is bounded, compactness of $T$ gives a subsequence with $Tx_{n_k} \to z$, so $x_{n_k} = \frac{1}{\lambda}((\lambda I - T)x_{n_k} + Tx_{n_k}) \to \frac{1}{\lambda}(y + z)$. If $(x_n)$ is unbounded, normalize: $u_n = x_n/\|x_n\|$. Then $(\lambda I - T)u_n \to 0$, and compactness gives $Tu_{n_k} \to w$, so $u_{n_k} \to w/\lambda$, and $(\lambda I - T)(w/\lambda) = 0$, contradicting $u_n \perp \ker(\lambda I - T)$ and $\|u_n\| = 1$.

*Claim 3: The alternative.* Either $\ker(\lambda I - T) = \{0\}$ (Case A) or $\ker(\lambda I - T) \neq \{0\}$ (Case B). In Case A, $\lambda I - T$ is injective with closed range. A dimension-counting argument (using the Fredholm index, which equals zero for $\lambda I - T$ with $T$ compact) shows the range is all of $H$. In Case B, the solvability condition $y \perp \ker(\overline{\lambda}I - T^*)$ follows from the closed range theorem: $\text{range}(\lambda I - T) = \ker(\overline{\lambda} I - T^*)^\perp$. $\square$

**Example 4 (Application to integral equations).** Consider the Fredholm equation of the second kind:

$$\lambda f(x) - \int_0^1 K(x,y) f(y) \, dy = g(x).$$

If $T$ is the integral operator with kernel $K$, this is $(\lambda I - T)f = g$. The Fredholm alternative says: either this equation has a unique solution for every $g$, or the homogeneous equation $\lambda f = Tf$ has finitely many linearly independent solutions, and the inhomogeneous equation is solvable only when $g$ satisfies finitely many compatibility conditions.

For the specific kernel $K(x,y) = xy$, the operator $T$ is rank-1: $(Tf)(x) = x \int_0^1 yf(y)\,dy$. The eigenvalue equation $Tf = \lambda f$ gives $\lambda f(x) = cx$ where $c = \int_0^1 yf(y)\,dy$, so $f(x) = (c/\lambda)x$ and $\lambda = \int_0^1 y \cdot (c/\lambda) y \, dy = c/(3\lambda)$, giving $\lambda^2 = 1/3$, so $\lambda = \pm 1/\sqrt{3}$. For any $\lambda \neq \pm 1/\sqrt{3}$, the equation $(\lambda I - T)f = g$ has a unique solution.

**The resolvent formula for Case (A).** When $(\lambda I - T)^{-1}$ exists and $T$ is an integral operator with kernel $K$, one can express the solution of $\lambda f - Tf = g$ using the **resolvent kernel**: there exists $R_\lambda(x,y)$ such that $f(x) = \frac{1}{\lambda}g(x) + \frac{1}{\lambda}\int_0^1 R_\lambda(x,y) g(y)\,dy$. The resolvent kernel can be computed iteratively via the Neumann series $R_\lambda = \sum_{n=1}^\infty \lambda^{-n} K_n$, where $K_n$ is the $n$-th iterated kernel ($K_1 = K$, $K_{n+1}(x,y) = \int K(x,z)K_n(z,y)\,dz$). This series converges for $|\lambda| > r(T)$, where $r(T)$ is the spectral radius.

**The Fredholm index.** For $\lambda \neq 0$ and $T$ compact, the operator $\lambda I - T$ is a **Fredholm operator**: it has finite-dimensional kernel, closed range, and finite-dimensional cokernel. The **Fredholm index** is defined as:

$$\text{ind}(\lambda I - T) = \dim \ker(\lambda I - T) - \dim \ker(\overline{\lambda}I - T^*).$$

A fundamental property of compact perturbations of the identity is that the Fredholm index is **zero**: $\dim \ker(\lambda I - T) = \dim \ker(\overline{\lambda}I - T^*)$. This is the deep reason why the Fredholm alternative has its either-or form: in Case (A), both dimensions are zero; in Case (B), both are equal and positive. The number of compatibility conditions for solvability of $(\lambda I - T)x = y$ equals the number of linearly independent solutions of $(\lambda I - T)x = 0$.

---

## Hilbert-Schmidt and Trace Class Operators

Two important subclasses of compact operators have quantitative "size" measures that go beyond the operator norm.

**Hilbert-Schmidt operators.** $T \in B(H)$ is Hilbert-Schmidt if for some (equivalently, every) orthonormal basis $\{e_n\}$:

$$\|T\|_{HS}^2 = \sum_n \|Te_n\|^2 < \infty.$$

Properties:
- $\|T\| \leq \|T\|_{HS}$ (so Hilbert-Schmidt operators are bounded).
- Hilbert-Schmidt operators are compact.
- The Hilbert-Schmidt operators form a Hilbert space themselves, with inner product $\langle S, T \rangle_{HS} = \sum_n \langle Se_n, Te_n \rangle$.
- For integral operators on $L^2(\Omega)$, $\|T\|_{HS} = \|K\|_{L^2(\Omega \times \Omega)}$.

**Trace class operators.** $T \in B(H)$ is trace class if $\sum_n \langle |T| e_n, e_n \rangle < \infty$, where $|T| = (T^*T)^{1/2}$. The **trace** is then well-defined:

$$\text{tr}(T) = \sum_n \langle Te_n, e_n \rangle,$$

independent of the choice of orthonormal basis.

Properties:
- Trace class $\subset$ Hilbert-Schmidt $\subset$ compact operators.
- If $T$ is compact with eigenvalues $\lambda_n$ (counted with multiplicity), then $T$ is trace class if and only if $\sum |\lambda_n| < \infty$, and in this case $\text{tr}(T) = \sum \lambda_n$ (**Lidskii's theorem**).
- $\text{tr}(AB) = \text{tr}(BA)$ whenever $A$ is trace class and $B$ is bounded.

**Example 5.** The operator $T$ on $\ell^2$ defined by $T(e_n) = \frac{1}{n} e_n$ is compact (diagonal with eigenvalues $1/n \to 0$), Hilbert-Schmidt ($\sum 1/n^2 < \infty$), but not trace class ($\sum 1/n = \infty$). The operator $S(e_n) = \frac{1}{n^2} e_n$ is trace class with $\text{tr}(S) = \pi^2/6$.

**Singular values and the Schmidt decomposition.** For a compact operator $T$ on a Hilbert space $H$ (not necessarily self-adjoint), the **singular values** are the eigenvalues of $|T| = (T^*T)^{1/2}$, arranged in decreasing order: $s_1(T) \geq s_2(T) \geq \cdots \geq 0$, $s_n(T) \to 0$. The operator $T$ admits a **singular value decomposition** (Schmidt decomposition):

$$T = \sum_n s_n(T) \langle \cdot, v_n \rangle u_n,$$

where $\{v_n\}$ and $\{u_n\}$ are orthonormal sequences in $H$ (the right and left singular vectors). This is the infinite-dimensional generalization of the SVD of a matrix.

The singular values determine the "size" of the operator:
- $\|T\| = s_1(T)$ (the operator norm equals the largest singular value).
- $\|T\|_{HS} = (\sum s_n(T)^2)^{1/2}$ (Hilbert-Schmidt norm).
- $\|T\|_1 = \sum s_n(T)$ (trace norm / nuclear norm).

**Schatten classes.** The **Schatten $p$-class** $\mathcal{S}_p$ consists of compact operators $T$ with $\sum s_n(T)^p < \infty$. The Schatten $p$-norm is $\|T\|_p = (\sum s_n(T)^p)^{1/p}$. Important cases: $\mathcal{S}_1$ = trace class, $\mathcal{S}_2$ = Hilbert-Schmidt, $\mathcal{S}_\infty$ = compact operators (with operator norm). These classes satisfy Holder-type inequalities: if $T \in \mathcal{S}_p$ and $S \in \mathcal{S}_q$ with $1/p + 1/q = 1/r$, then $TS \in \mathcal{S}_r$.

**Application to statistical mechanics.** In quantum statistical mechanics, the density matrix (describing mixed quantum states) is a positive trace-class operator $\rho$ with $\text{tr}(\rho) = 1$. The von Neumann entropy is $S(\rho) = -\text{tr}(\rho \log \rho) = -\sum_n \lambda_n \log \lambda_n$, where $\lambda_n$ are the eigenvalues (by the spectral theorem for compact self-adjoint operators). The trace-class condition ensures this is well-defined. The compact operator framework is thus the mathematical foundation of quantum information theory.

---

## Summary of the Hierarchy

| Class | Definition | Spectral characterization |
|---|---|---|
| Finite-rank | $\dim(\text{range}) < \infty$ | Finitely many nonzero eigenvalues |
| Trace class | $\sum \langle \|T\| e_n, e_n \rangle < \infty$ | $\sum |\lambda_n| < \infty$ |
| Hilbert-Schmidt | $\sum \|Te_n\|^2 < \infty$ | $\sum |\lambda_n|^2 < \infty$ |
| Compact | $\overline{T(B_H)}$ compact | $\lambda_n \to 0$ |
| Bounded | $\|T\| < \infty$ | $\sigma(T)$ bounded |

Each inclusion is strict.

**The approximation property.** A Banach space $X$ has the **approximation property** if every compact operator on $X$ can be approximated (in operator norm) by finite-rank operators. All Hilbert spaces and all $L^p$ spaces have the approximation property. Per Enflo (1973) constructed the first example of a Banach space failing the approximation property — a remarkable result showing that the neat "compact = limit of finite-rank" characterization is specific to well-behaved spaces. The question of which spaces have the approximation property remains an active area of research in Banach space theory.

**Compact operators on specific spaces.** Let us catalog the behavior:
- On $\ell^p$ ($1 \leq p < \infty$): every compact operator is the limit of finite-rank operators. The compact operators on $\ell^2$ can be identified with "infinite matrices" $(a_{ij})$ such that the corresponding operator has singular values tending to zero.
- On $C([0,1])$: compact operators include integral operators with continuous kernels (Arzela-Ascoli), but also more exotic operators.
- On $L^p(\Omega)$: compact embedding theorems (Rellich-Kondrachov) show that the inclusion $W^{1,p}(\Omega) \hookrightarrow L^p(\Omega)$ is compact for bounded $\Omega$. This is the foundation of elliptic regularity theory.

---

## What's Next

The spectral theorem for compact self-adjoint operators is beautiful but limited — it only applies to compact operators, which are a very special (though important) class. In the next article, we develop the **spectral theory of general bounded operators**: the spectrum $\sigma(T)$, the resolvent, the spectral radius formula, and the full spectral theorem for bounded self-adjoint operators via the continuous functional calculus. This will take us from the discrete eigenvalue picture of compact operators to the continuous spectral decomposition that governs quantum mechanics and operator algebras.

The key conceptual shift will be from "diagonalization" (expressing $T$ as a sum $\sum \lambda_n P_n$) to "integration against a spectral measure" (expressing $T$ as $\int \lambda \, dE(\lambda)$). The compact case will appear as the special case where the spectral measure is purely atomic — supported on a countable set of points with finite-rank projections. The general case allows the spectral measure to spread continuously across an interval, corresponding to operators with no eigenvalues at all.

---

*This is Part 7 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 6 — Bounded Operators](/en/functional-analysis/06-bounded-operators/)*

*Next: [Part 8 — Spectral Theory](/en/functional-analysis/08-spectral-theory/)*
