---
title: "Normed Spaces and Banach Spaces"
date: 2021-10-03 09:00:00
tags:
  - functional-analysis
  - banach-spaces
  - normed-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 2
series_total: 12
translationKey: "functional-analysis-02"
description: "Norm axioms, classical examples, equivalence of norms in finite dimensions, completeness and why it matters, Schauder bases, quotient spaces, and the role of separability."
---

If metric spaces are the bare landscape of "distance," then normed spaces are what you get when you add algebraic structure: a vector space where the distance comes from a norm. This is where functional analysis properly begins, because the interplay between the linear structure and the metric structure is what makes the subject sing. And within normed spaces, the Banach spaces --- the complete ones --- are where almost all of the deep theorems live.

In this article we build up the theory of normed spaces from the axioms, work through the classical examples that will accompany us for the rest of the series, prove that norms in finite dimensions are all equivalent (and explain why this fails spectacularly in infinite dimensions), and then turn to completeness. I want to convince you that completeness is not a mild technical convenience but an essential structural property without which the major theorems of functional analysis simply do not hold.

## Norms and normed spaces

A **normed space** is a pair $(X, \|\cdot\|)$ where $X$ is a vector space over $\mathbb{K}$ (either $\mathbb{R}$ or $\mathbb{C}$) and $\|\cdot\|: X \to [0, \infty)$ is a function satisfying:

1. **Positive definiteness:** $\|x\| = 0$ if and only if $x = 0$.
2. **Absolute homogeneity:** $\|\alpha x\| = |\alpha|\|x\|$ for all $\alpha \in \mathbb{K}$, $x \in X$.
3. **Triangle inequality:** $\|x + y\| \leq \|x\| + \|y\|$ for all $x, y \in X$.

Every norm induces a metric via $d(x,y) = \|x - y\|$, so normed spaces are metric spaces. But not every metric on a vector space comes from a norm. The norm-induced metric has two special properties: it is *translation-invariant* ($d(x+z, y+z) = d(x,y)$) and *absolutely homogeneous* ($d(\alpha x, \alpha y) = |\alpha|d(x,y)$). These ensure that the metric "respects" the vector space operations.

A **seminorm** satisfies axioms 2 and 3 but replaces 1 with the weaker condition $\|x\| \geq 0$. Seminorms appear naturally in many contexts (e.g., the $L^p$ "norm" before passing to equivalence classes is a seminorm on the space of measurable functions).

It is worth noting what the norm axioms *do not* require. There is no inner product, no notion of orthogonality, no Pythagorean theorem. Normed spaces are strictly more general than inner product spaces. A norm comes from an inner product if and only if it satisfies the **parallelogram law**: $\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2$ for all $x, y$. This is an easy exercise in one direction (expand the inner products) and a deep result in the other (the Jordan-von Neumann theorem: the parallelogram law implies the existence of an inner product inducing the norm). Most of the spaces we will meet do *not* satisfy the parallelogram law, which is why we work in the more general normed space setting.

**Bounded linear maps.** A linear map $T: X \to Y$ between normed spaces is **bounded** if there exists $C \geq 0$ such that $\|Tx\| \leq C\|x\|$ for all $x \in X$. The smallest such $C$ is the **operator norm** $\|T\| = \sup_{\|x\|\leq 1}\|Tx\|$. For linear maps, boundedness is equivalent to continuity (and even to continuity at a single point). The space $B(X, Y)$ of all bounded linear maps from $X$ to $Y$, with the operator norm, is itself a normed space; it is Banach whenever $Y$ is Banach. This fact --- that the target's completeness is inherited by the operator space --- will be used repeatedly.


![Unit balls in different norms on R^2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_fig1_unit_balls.png)

## The classical examples

The examples in this section are not mere illustrations --- they are the building blocks of the entire subject. Master them and you will be well-equipped for everything that follows.

**The $\ell^p$ spaces.** For $1 \leq p < \infty$, the space $\ell^p$ consists of all sequences $x = (x_1, x_2, \ldots)$ of scalars with
$$
\|x\|_p = \left(\sum_{n=1}^{\infty} |x_n|^p\right)^{1/p} < \infty.
$$
For $p = \infty$,
$$
\ell^\infty = \\{x = (x_n) : \sup_n |x_n| < \infty\\}, \quad \|x\|_\infty = \sup_n |x_n|.
$$

That $\|\cdot\|_p$ satisfies the triangle inequality is Minkowski's inequality, which is itself a consequence of Holder's inequality: for $1/p + 1/q = 1$,
$$
\sum_{n=1}^\infty |x_n y_n| \leq \|x\|_p \|y\|_q.
$$

The proof of Holder's inequality rests on Young's inequality $ab \leq a^p/p + b^q/q$ for $a,b \geq 0$, applied pointwise to the normalized sequences $x_n/\|x\|_p$ and $y_n/\|y\|_q$.

Important subspaces of $\ell^\infty$:
- $c$ = the space of convergent sequences (with the sup norm).
- $c_0$ = the space of sequences converging to zero.

Both $c$ and $c_0$ are closed subspaces of $\ell^\infty$, hence Banach spaces in their own right.

**The $L^p$ spaces.** For a measure space $(\Omega, \Sigma, \mu)$ and $1 \leq p \leq \infty$, the space $L^p(\Omega, \mu)$ consists of (equivalence classes of) measurable functions with
$$
\|f\|_p = \left(\int_\Omega |f|^p \, d\mu\right)^{1/p} < \infty \quad (p < \infty), \qquad \|f\|_\infty = \operatorname{ess\,sup} |f|.
$$

The identification of functions that agree almost everywhere is essential: without it, $\|\cdot\|_p$ is only a seminorm. The $L^p$ spaces are the functional-analytic backbone of modern PDE theory, probability, and harmonic analysis.

**$C[a,b]$ and its relatives.** The space $C[a,b]$ of continuous functions on a closed interval, with the supremum norm $\|f\|_\infty = \max_{t \in [a,b]}|f(t)|$, is a Banach space. More generally, for a compact Hausdorff space $K$, $C(K)$ with the sup norm is a Banach space. The completeness follows directly from the fact that a uniform limit of continuous functions is continuous.

One can also equip $C[a,b]$ with the $L^p$ norm $\|f\|_p = (\int_a^b |f(t)|^p \, dt)^{1/p}$. This is a perfectly good norm, but the resulting space is *not* complete: there exist Cauchy sequences of continuous functions whose $L^p$-limit is not continuous. The completion of $(C[a,b], \|\cdot\|_p)$ is precisely $L^p[a,b]$.

**$BV[a,b]$ --- the space of bounded variation.** A function $f: [a,b] \to \mathbb{R}$ has bounded variation if
$$
V_a^b(f) = \sup \sum_{k=1}^n |f(t_k) - f(t_{k-1})| < \infty,
$$
where the supremum is over all partitions $a = t_0 < t_1 < \cdots < t_n = b$. The space $BV[a,b]$ with the norm $\|f\| = |f(a)| + V_a^b(f)$ is a Banach space. Functions of bounded variation are important because they can be written as the difference of two increasing functions (Jordan decomposition), and they serve as the natural integrators for the Riemann-Stieltjes integral. The dual of $C[a,b]$ turns out to be the space of signed Borel measures on $[a,b]$, which by the Riesz representation theorem can be identified with a quotient of $BV[a,b]$.

**Sobolev spaces (a preview).** For $\Omega \subseteq \mathbb{R}^n$ open and $k \in \mathbb{N}$, $1 \leq p \leq \infty$, the Sobolev space $W^{k,p}(\Omega)$ consists of $L^p$ functions whose weak derivatives up to order $k$ are also in $L^p$. The norm is
$$
\|f\|_{W^{k,p}} = \sum_{|\alpha| \leq k} \|D^\alpha f\|_p.
$$
These spaces are Banach (as closed subspaces of products of $L^p$ spaces). We mention them here because they are the natural domains for differential operators in PDE theory. The special case $W^{k,2} = H^k$ is a Hilbert space, which we will meet again when we discuss unbounded operators.

## Equivalence of norms in finite dimensions

One of the most striking facts in linear algebra is that all norms on a finite-dimensional vector space are equivalent:

**Theorem.** Let $X$ be a finite-dimensional vector space over $\mathbb{K}$ with $\dim X = n$. If $\|\cdot\|_a$ and $\|\cdot\|_b$ are any two norms on $X$, then there exist constants $0 < c \leq C < \infty$ such that
$$
c\|x\|_a \leq \|x\|_b \leq C\|x\|_a \quad \text{for all } x \in X.
$$

**Proof sketch.** Fix a basis $e_1, \ldots, e_n$ and use it to identify $X$ with $\mathbb{K}^n$. Consider the $\ell^1$ norm $\|x\|_1 = \sum |x_k|$ and an arbitrary norm $\|\cdot\|$. The triangle inequality and homogeneity give
$$
\|x\| = \left\|\sum_{k=1}^n x_k e_k\right\| \leq \sum_{k=1}^n |x_k| \|e_k\| \leq M \|x\|_1,
$$
where $M = \max_k \|e_k\|$. So every norm is bounded above by a multiple of $\|\cdot\|_1$.

For the lower bound, consider the unit sphere $S = \\{x \in \mathbb{K}^n : \|x\|_1 = 1\\}$ in the $\ell^1$ norm. The function $x \mapsto \|x\|$ is continuous with respect to $\|\cdot\|_1$ (by the upper bound just established), and $S$ is compact (since we are in finite dimensions). Since $\|x\| > 0$ on $S$ (because $x \neq 0$ on $S$ and $\|\cdot\|$ is a norm), the continuous function $\|\cdot\|$ attains a positive minimum $m > 0$ on $S$. By homogeneity, $\|x\| \geq m\|x\|_1$ for all $x$.

Since any two norms are each equivalent to $\|\cdot\|_1$, they are equivalent to each other.

**Why this fails in infinite dimensions.** The proof uses compactness of the unit sphere, which is equivalent (by the Riesz lemma) to finite-dimensionality. On $\ell^p$ for $p \neq q$, the norms $\|\cdot\|_p$ and $\|\cdot\|_q$ are not equivalent. The sequences $e_n = (0, \ldots, 0, 1, 0, \ldots)$ satisfy $\|e_n\|_p = 1$ for every $p$, so the norms agree on the basis vectors, but on the vector $x_N = (1, 1/2, 1/3, \ldots, 1/N, 0, 0, \ldots)$, the $\ell^1$ and $\ell^2$ norms grow at different rates as $N \to \infty$.

**The Riesz lemma.** This is the key tool that connects finite-dimensionality to compactness. It states: if $M$ is a proper closed subspace of a normed space $X$ and $0 < \theta < 1$, then there exists $x \in X$ with $\|x\| = 1$ and $d(x, M) = \inf_{m \in M}\|x - m\| \geq \theta$. In finite dimensions we could take $\theta = 1$ (attaining the distance), but in infinite dimensions we generally cannot.

Using the Riesz lemma, one can show that the closed unit ball of a normed space is compact if and only if the space is finite-dimensional. The proof constructs an infinite $\theta$-separated sequence ($\|x_n - x_m\| \geq \theta$ for $n \neq m$) by repeatedly applying the lemma to the span of previous vectors. Such a sequence has no convergent subsequence, so the ball is not sequentially compact.

**Consequence.** In finite dimensions, all topological and metric notions (convergence, completeness, continuity of linear maps) are norm-independent. In infinite dimensions, the choice of norm is a crucial modeling decision.

## Banach spaces: completeness

A normed space that is complete --- every Cauchy sequence converges --- is called a **Banach space**, in honor of Stefan Banach, who systematically developed the theory in his 1932 monograph.

All of our main examples are Banach spaces:
- $\ell^p$ for $1 \leq p \leq \infty$ (the proof for $\ell^p$ uses the completeness of $\mathbb{K}$ and a diagonalization argument).
- $L^p(\Omega, \mu)$ for $1 \leq p \leq \infty$ (the Riesz-Fischer theorem).
- $C(K)$ for compact Hausdorff $K$ (uniform limit of continuous functions is continuous).
- $BV[a,b]$ (Helly's selection theorem provides the key compactness).

**Completeness of $\ell^p$ (proof).** Let $(x^{(k)})_{k=1}^\infty$ be a Cauchy sequence in $\ell^p$, where $x^{(k)} = (x_n^{(k)})_{n=1}^\infty$. For each fixed $n$, the sequence $(x_n^{(k)})_{k=1}^\infty$ is Cauchy in $\mathbb{K}$ (since $|x_n^{(k)} - x_n^{(j)}| \leq \|x^{(k)} - x^{(j)}\|_p$), so it converges to some $x_n \in \mathbb{K}$. Set $x = (x_1, x_2, \ldots)$. We need to show $x \in \ell^p$ and $\|x^{(k)} - x\|_p \to 0$.

Given $\varepsilon > 0$, choose $K$ so that $\|x^{(k)} - x^{(j)}\|_p < \varepsilon$ for $k, j \geq K$. For any finite $N$:
$$
\left(\sum_{n=1}^N |x_n^{(k)} - x_n^{(j)}|^p\right)^{1/p} \leq \|x^{(k)} - x^{(j)}\|_p < \varepsilon.
$$
Letting $j \to \infty$ (so $x_n^{(j)} \to x_n$):
$$
\left(\sum_{n=1}^N |x_n^{(k)} - x_n|^p\right)^{1/p} \leq \varepsilon.
$$
Since this holds for all $N$, we get $\|x^{(k)} - x\|_p \leq \varepsilon$ for $k \geq K$. In particular, $x = (x - x^{(K)}) + x^{(K)} \in \ell^p$ (sum of two $\ell^p$ elements), and $x^{(k)} \to x$ in $\ell^p$.

**Completeness of $L^p$ (the Riesz-Fischer theorem).** The proof for $L^p$ is more subtle because we work with equivalence classes of functions. The standard approach uses the following lemma: a normed space is complete if and only if every absolutely convergent series converges. Given a Cauchy sequence $(f_n)$ in $L^p$, pass to a subsequence $(f_{n_k})$ with $\|f_{n_{k+1}} - f_{n_k}\|_p < 2^{-k}$. The telescoping series $f_{n_1} + \sum_k (f_{n_{k+1}} - f_{n_k})$ converges absolutely in $L^p$ norm. One then shows (using the monotone convergence theorem) that the pointwise limit exists a.e. and equals an $L^p$ function to which the subsequence converges in norm. Since the original Cauchy sequence has a convergent subsequence, the whole sequence converges.

## Why completeness matters

Completeness is not an aesthetic preference; it is the structural property that makes the major theorems of functional analysis work. Here are two fundamental reasons.

**Absolute convergence implies convergence.** In a Banach space, a series $\sum_{n=1}^\infty x_n$ converges whenever $\sum_{n=1}^\infty \|x_n\| < \infty$. (In fact, this property is *equivalent* to completeness.) This means we can manipulate infinite series with the same confidence as in $\mathbb{R}$. In a non-complete normed space, absolutely convergent series can fail to converge, making even basic constructions unreliable.

*Proof:* Let $S_N = \sum_{n=1}^N x_n$. For $M > N$, $\|S_M - S_N\| = \|\sum_{n=N+1}^M x_n\| \leq \sum_{n=N+1}^M \|x_n\| \to 0$ as $N, M \to \infty$ (because the tail of a convergent series tends to zero). So $(S_N)$ is Cauchy, hence convergent by completeness.

**Fixed point theorems.** The Banach contraction mapping principle --- the most fundamental fixed point theorem in analysis --- requires completeness. If $T: X \to X$ is a contraction ($\|Tx - Ty\| \leq q\|x-y\|$ for some $q < 1$) on a complete metric space, then $T$ has a unique fixed point. The proof constructs the fixed point as the limit of the iterates $x, Tx, T^2x, \ldots$; without completeness, this Cauchy sequence might not converge.

Applications of the contraction mapping principle include:
- The Picard-Lindelof theorem on existence and uniqueness of ODE solutions.
- The inverse function theorem and the implicit function theorem.
- The existence of solutions to integral equations of the form $f(x) = g(x) + \lambda \int K(x,t) f(t) \, dt$ for small $|\lambda|$.

**The big three theorems.** Completeness is a hypothesis (or consequence) in each of the three pillars:
- The *Baire category theorem* requires completeness (Article 3).
- The *open mapping theorem* and *closed graph theorem* require both the domain and codomain to be Banach.
- The *uniform boundedness principle* requires the domain to be Banach (the codomain can be any normed space).

Without completeness, each of these fails. A well-known counterexample: on $c_{00}$ (finitely supported sequences, a non-complete subspace of $\ell^2$), the coordinate functionals $e_n^*(x) = x_n$ are each bounded with norm 1, and they are pointwise bounded on every $x \in c_{00}$ (since each $x$ has finite support). But the norms $\|e_n^*\| = 1$ are already uniformly bounded, so this particular example does not demonstrate the failure. A subtler example is needed: consider the partial sum operators $S_N(f) = \sum_{n=1}^N \hat{f}(n) e^{inx}$ on $(C[-\pi,\pi], \|\cdot\|_\infty)$, which is a Banach space --- the uniform boundedness principle applies and shows that $\|S_N\| \to \infty$, which implies the existence of continuous functions whose Fourier series diverges. If we tried to run this argument on a non-complete subspace, the conclusion would not follow.

**Neumann series.** Here is a direct application of completeness to operator theory that illustrates its computational power. If $X$ is a Banach space and $T \in B(X)$ with $\|T\| < 1$, then $I - T$ is invertible and
$$
(I - T)^{-1} = \sum_{n=0}^\infty T^n.
$$
The series converges absolutely: $\sum \|T^n\| \leq \sum \|T\|^n = 1/(1-\|T\|) < \infty$. In a Banach space, absolute convergence implies convergence, so the series converges to some $S \in B(X)$. Checking $(I-T)S = S(I-T) = I$ is a formal computation. This is the Banach space analogue of the geometric series $1/(1-z) = \sum z^n$ for $|z| < 1$. In a non-complete normed space, the series might not converge even though it converges absolutely, and the inverse might not exist in the space.

## Schauder bases

In finite-dimensional linear algebra, every vector space has a basis (by Zorn's lemma in general, or by explicit construction in concrete cases), and every vector can be written uniquely as a finite linear combination of basis vectors. In infinite-dimensional Banach spaces, we need infinite series, which requires a notion of convergence.

A sequence $(e_n)_{n=1}^\infty$ in a Banach space $X$ is a **Schauder basis** if every $x \in X$ has a unique representation
$$
x = \sum_{n=1}^\infty \alpha_n e_n
$$
where the series converges in the norm of $X$. The coefficient functionals $e_n^*: X \to \mathbb{K}$, defined by $e_n^*(x) = \alpha_n$, are automatically continuous (this is a non-trivial consequence of the Banach-Steinhaus theorem).

**Examples:**
- The standard basis $(e_n)$ is a Schauder basis for $\ell^p$ ($1 \leq p < \infty$) and $c_0$, but *not* for $\ell^\infty$ (the constant sequence $(1,1,1,\ldots)$ is not a norm-convergent series in the standard basis).
- The trigonometric system $\\{e^{inx}\\}_{n \in \mathbb{Z}}$ is a Schauder basis for $L^p[-\pi, \pi]$ when $1 < p < \infty$ (Carleson-Hunt theorem for $L^2$, Marcel Riesz theorem for $L^p$), but *not* for $L^1$ (the Fourier series of an $L^1$ function can diverge in $L^1$).
- The Haar system is a Schauder basis for $L^p[0,1]$ for all $1 \leq p < \infty$.

A Banach space with a Schauder basis is necessarily separable (the finite rational linear combinations of basis vectors form a countable dense set). The converse was a famous open problem: does every separable Banach space have a Schauder basis? Per Enflo answered this negatively in 1973, constructing a separable Banach space with no Schauder basis.

**Unconditional bases and conditional bases.** A Schauder basis $(e_n)$ is **unconditional** if the series $\sum \alpha_n e_n$ converges unconditionally (i.e., every rearrangement converges) for every $x = \sum \alpha_n e_n \in X$. The standard basis of $\ell^p$ ($1 \leq p < \infty$) is unconditional. The trigonometric system in $L^p$ for $p \neq 2$ is conditional (not unconditional) --- a deep fact related to the Khintchine inequality and Paley's theorem. A celebrated result of Gowers and Maurey (1993) shows that there exist Banach spaces in which *every* basis is conditional; their construction also produced the first example of an infinite-dimensional Banach space in which every bounded operator is a scalar multiple of the identity plus a compact operator.

**Hamel bases versus Schauder bases.** Every vector space has a *Hamel basis* (by Zorn's lemma): a set $B$ such that every vector is a *finite* linear combination of elements of $B$. In infinite-dimensional Banach spaces, Hamel bases are necessarily uncountable (by the Baire category theorem: if $\\{e_n\\}$ were a countable Hamel basis, then $X = \bigcup_N \operatorname{span}\\{e_1, \ldots, e_N\\}$ would be a countable union of nowhere dense sets, contradicting Baire). So Hamel bases and Schauder bases are genuinely different objects: Hamel bases use finite sums, Schauder bases use convergent infinite series.

## Quotient spaces and direct sums

Two standard constructions produce new Banach spaces from old ones.

**Quotient spaces.** Let $X$ be a Banach space and $M$ a closed subspace. The quotient space $X/M = \\{x + M : x \in X\\}$ has a natural norm:
$$
\|x + M\| = \inf_{m \in M} \|x + m\| = d(x, M).
$$

**Theorem.** If $X$ is a Banach space and $M$ is closed, then $X/M$ is a Banach space.

*Proof.* We verify completeness. Let $(x_n + M)$ be a Cauchy sequence in $X/M$. By passing to a subsequence (which suffices for completeness), assume $\|x_{n+1} + M - (x_n + M)\| < 2^{-n}$. Choose representatives: pick $m_1 \in M$ with $\|(x_2 - x_1) + m_1\| < 2^{-1}$, then $m_2 \in M$ with $\|(x_3 - x_2) + m_2\| < 2^{-2}$, etc. Set $y_1 = x_1$ and $y_{n+1} = x_{n+1} + (m_1 + \cdots + m_n)$. Then $y_{n+1} + M = x_{n+1} + M$ and $\|y_{n+1} - y_n\| < 2^{-n}$, so $(y_n)$ is Cauchy in $X$, hence converges to some $y$. Then $x_n + M = y_n + M \to y + M$.

**Quotient spaces matter** because many important constructions produce them naturally: the space $L^p$ is a quotient of the space of $p$-integrable functions by the subspace of functions equal to zero almost everywhere. The first isomorphism theorem for Banach spaces says that if $T: X \to Y$ is a bounded surjection, then $Y \cong X/\ker T$ (as topological vector spaces).

**Direct sums.** Given Banach spaces $X$ and $Y$, the direct sum $X \oplus_p Y$ is the set $X \times Y$ with the norm
$$
\|(x, y)\|_p = \begin{cases} (\|x\|^p + \|y\|^p)^{1/p}, & 1 \leq p < \infty, \\\ \max(\|x\|, \|y\|), & p = \infty. \end{cases}
$$
All these norms give equivalent topologies (since we are adding two coordinates, which is a finite-dimensional choice), and the resulting space is Banach if both $X$ and $Y$ are.

More generally, for a sequence of Banach spaces $(X_n)$, the $\ell^p$-direct sum is
$$
\left(\bigoplus_{n=1}^\infty X_n\right)_p = \left\\{(x_n) : x_n \in X_n,\; \sum_{n=1}^\infty \|x_n\|^p < \infty\right\\},
$$
with the obvious norm. This construction allows us to build complicated Banach spaces from simpler pieces.

## Separable and non-separable spaces

A normed space is **separable** if it contains a countable dense subset. Separability is a measure of "size" that has deep structural consequences.

**Separable examples:**
- $\ell^p$ for $1 \leq p < \infty$ (the sequences with finitely many nonzero rational entries are countable and dense).
- $c_0$ (same argument).
- $L^p(\mathbb{R}^n)$ for $1 \leq p < \infty$ (step functions with rational heights on rational intervals are dense).
- $C[a,b]$ (polynomials with rational coefficients, by Weierstrass approximation).

**Non-separable examples:**
- $\ell^\infty$ is not separable. To see this, for each subset $S \subseteq \mathbb{N}$, let $\mathbf{1}_S$ be the characteristic function of $S$. If $S \neq T$, then $\|\mathbf{1}_S - \mathbf{1}_T\|_\infty = 1$. The open balls $B(\mathbf{1}_S, 1/3)$ are uncountably many disjoint open sets, so no countable set can intersect all of them.
- $L^\infty[0,1]$ is not separable (a similar uncountable family of well-separated functions can be constructed from characteristic functions of disjoint subsets of positive measure --- or more precisely, the argument uses an uncountable collection of measurable subsets no two of which agree up to a null set).
- The space $B(H)$ of bounded operators on an infinite-dimensional Hilbert space is not separable in the operator norm (consider the uncountable family of projections onto one-dimensional subspaces).

**Why separability matters:**
- In a separable Banach space, the weak$^*$ topology on bounded subsets of $X^*$ is metrizable. Combined with the Banach-Alaoglu theorem (the unit ball of $X^*$ is weak$^*$ compact), this gives sequential weak$^*$ compactness, which is essential in many existence proofs.
- A Banach space is separable if and only if every bounded sequence has a weakly convergent subsequence (Eberlein-Smulian, for the reflexive case; more generally, if $X^*$ is separable then the unit ball of $X$ is weakly metrizable, and Alaoglu's theorem gives sequential compactness).
- Separability is related to the existence of Schauder bases (every space with a Schauder basis is separable, but not conversely).
- In probability theory, separability of the underlying function space is often needed to ensure that suprema of stochastic processes are measurable.
- Many constructions in analysis (e.g., Gram-Schmidt orthogonalization, construction of conditional expectations) require a countable dense set.

## What's next

We now have the vocabulary of normed and Banach spaces, along with a library of examples that will serve as test cases and counterexamples throughout the series. The key takeaways: norms give us a way to measure size that is compatible with the linear structure; in finite dimensions all norms are equivalent, but in infinite dimensions the choice of norm is a fundamental modeling decision; completeness --- the Banach property --- is not optional but essential for the deep theorems to hold.

The next article (Article 3) introduces the **Baire category theorem** and its three great consequences: the *uniform boundedness principle* (Banach-Steinhaus theorem), the *open mapping theorem*, and the *closed graph theorem*. These are the power tools of Banach space theory --- they take completeness as input and produce sweeping structural conclusions about bounded linear operators. The uniform boundedness principle says that pointwise boundedness of a family of operators implies uniform boundedness (in the operator norm). The open mapping theorem says that a surjective bounded linear map between Banach spaces is automatically an open map. The closed graph theorem says that a closed linear map between Banach spaces is automatically bounded. Each of these results is genuinely surprising --- it extracts a global conclusion from seemingly local hypotheses --- and each relies crucially on completeness. Once you have them in hand, the theory begins to feel remarkably powerful.

---

*This is Part 2 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 1 — Metric Spaces](/en/functional-analysis/01-metric-spaces/)*

*Next: [Part 3 — Hilbert Spaces](/en/functional-analysis/03-hilbert-spaces/)*
