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
translationKey: "functional-analysis-2"
description: "Norm axioms, classical examples, equivalence of norms in finite dimensions, completeness and why it matters, Schauder bases, quotient spaces, and the role of separability."
---

## Why a Norm Is More Than a Metric Wearing a Hat

![Animation: l^p unit ball morphing as p changes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/02_lp_morph.gif)

In Article 1, the metric was a free-standing function on a set with no algebraic structure. That generality bought us topology and completeness, but it gave nothing back to the algebra. The moment I am willing to assume the underlying set is a vector space, a more rigid object becomes available: a **norm**, a single nonnegative function on the space whose induced metric $d(x,y) = \|x - y\|$ is *translation-invariant* and *positively homogeneous*.

Translation invariance — $d(x + z, y + z) = d(x, y)$ — sounds like a clerical detail, but it is what makes the metric blind to where the origin lives, and that blindness is what lets me compose linear maps with metric arguments. Positive homogeneity — $\|\alpha x\| = |\alpha| \|x\|$ — gives me a quantitative scaling rule that pure metrics lack. Together, these turn a metric vector space into the natural home for *bounded linear operators*, the protagonists of every subsequent article.

The strange thing about working in a normed space rather than a generic metric space is how much more rigid the geometry becomes, even though I have only added a couple of axioms. For example, every closed subspace of a normed space inherits a normed structure (this is obvious in a metric space too), but in the normed setting the *quotient* by a closed subspace is also a normed space — there is no analogous quotient construction in the pure metric world. Linear functionals exist in the algebraic sense in any vector space, but in a normed space we can ask whether they are continuous, and the answer turns into a deep theory (Article 4). This article builds the basic vocabulary.

## The Norm Axioms

Let $V$ be a vector space over $\mathbb{R}$ or $\mathbb{C}$. A **norm** on $V$ is a function $\|\cdot\|: V \to [0, \infty)$ satisfying for all $x, y \in V$ and scalars $\alpha$:

![Norm equivalence in finite dimensions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/02_norm_equivalence.png)

1. $\|x\| = 0 \iff x = 0$ (positive definiteness).
2. $\|\alpha x\| = |\alpha|\, \|x\|$ (positive homogeneity).
3. $\|x + y\| \leq \|x\| + \|y\|$ (triangle inequality).

A **normed space** is a vector space with a norm. The norm induces a metric $d(x, y) = \|x - y\|$, which automatically satisfies the four metric-space axioms (a calculation), so every normed space is a metric space. A **Banach space** is a normed space that is complete in this metric.

Notice how few axioms there are. Compared to an inner product (Article 3) the norm is austere — there is no axiom forcing the parallelogram law, no axiom forcing the unit ball to be round. As a result, most normed spaces are not Hilbert, and most of the geometric intuition that comes from $\mathbb{R}^n$ has to be checked rather than assumed.

### Classical Examples

- $\mathbb{R}^n$ with $\|x\|_p = \big(\sum |x_i|^p\big)^{1/p}$ for $1 \leq p < \infty$, and $\|x\|_\infty = \max_i |x_i|$.
- The sequence spaces $\ell^p = \{ x = (x_n) : \sum |x_n|^p < \infty \}$ with $\|x\|_p = (\sum|x_n|^p)^{1/p}$, and $\ell^\infty$ of bounded sequences with $\|x\|_\infty = \sup |x_n|$.
- The continuous functions $C[a,b]$ with $\|f\|_\infty = \sup_{t \in [a,b]} |f(t)|$.
- The Lebesgue spaces $L^p[a,b] = \{ f : \int_a^b |f|^p < \infty \}$ modulo equality almost everywhere, with $\|f\|_p = \big(\int |f|^p\big)^{1/p}$.
- The bounded continuous functions $C_b(\mathbb{R})$ with sup norm.
- The space $c_0$ of sequences converging to $0$, with sup norm.
- The space of bounded linear operators $B(X, Y)$ between normed spaces, with operator norm $\|T\| = \sup_{\|x\| \leq 1} \|Tx\|$.

The Minkowski inequality (which is the triangle inequality for $\|\cdot\|_p$) and the Hölder inequality ($\sum |x_n y_n| \leq \|x\|_p \|y\|_q$ for $1/p + 1/q = 1$) are the workhorses for proving these are norms. Both are proved by reducing to the convexity of $t \mapsto t^p$ on $[0, \infty)$.

![Unit balls in the l1, l2, and l-infinity norms on R^2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_v2_02_1_unit_balls_lp.png)

### Worked numerical example

Take $x = (1, 2, 2) \in \mathbb{R}^3$. Then $\|x\|_1 = 5$, $\|x\|_2 = \sqrt{1 + 4 + 4} = 3$, $\|x\|_\infty = 2$. The general inequalities $\|x\|_\infty \leq \|x\|_2 \leq \|x\|_1$ check out: $2 \leq 3 \leq 5$. The relations $\|x\|_2 \leq \sqrt{n}\|x\|_\infty$ (here $\sqrt{3} \cdot 2 \approx 3.46 \geq 3$) and $\|x\|_1 \leq \sqrt{n}\|x\|_2$ (here $\sqrt{3} \cdot 3 \approx 5.2 \geq 5$) hold because all $\ell^p$ norms on $\mathbb{R}^n$ are equivalent up to factors of $n^{1/p - 1/q}$. This equivalence is the next theorem; in infinite dimensions it fails.

## All Norms on a Finite-Dimensional Space Are Equivalent

Two norms $\|\cdot\|_a$ and $\|\cdot\|_b$ on the same vector space $V$ are **equivalent** if there exist $C_1, C_2 > 0$ with
$$C_1 \|x\|_a \leq \|x\|_b \leq C_2 \|x\|_a \quad \text{for all } x \in V.$$
Equivalence of norms is precisely equivalence of induced topologies: a sequence converges in one iff it converges in the other.

![Banach (complete) vs incomplete normed space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/02_banach_vs_incomplete.png)

**Theorem.** Any two norms on a finite-dimensional vector space are equivalent.

*Sketch of proof.* Let $V$ be $n$-dimensional with basis $e_1, \ldots, e_n$, and define $\|x\|_2 = \big(\sum |x_i|^2\big)^{1/2}$ in coordinates. Let $\|\cdot\|$ be any other norm. Writing $x = \sum x_i e_i$ and using the triangle inequality and homogeneity gives $\|x\| \leq \sum |x_i| \|e_i\| \leq C_2 \|x\|_2$ for $C_2 = (\sum \|e_i\|^2)^{1/2}$ (Cauchy-Schwarz). For the lower bound, $\|\cdot\|: (V, \|\cdot\|_2) \to \mathbb{R}$ is continuous (by the upper bound) and the unit sphere $S = \{ x : \|x\|_2 = 1 \}$ is compact in $\|\cdot\|_2$ (closed and bounded in finite dimensions). So $\|\cdot\|$ attains a minimum $C_1 > 0$ on $S$, and homogeneity gives $\|x\| \geq C_1 \|x\|_2$ on all of $V$. $\square$

The proof leans on compactness of the unit sphere — the same compactness that fails in infinite dimensions. So the equivalence of all norms is genuinely a finite-dimensional theorem.

![All norms on a finite-dimensional space are equivalent](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_v2_02_2_norm_equiv.png)

### Why this matters

In finite dimensions, you can pick whichever norm makes a calculation easy. The topology, the convergent sequences, the closed sets, the continuous maps — all are the same regardless of choice. This is why introductory linear algebra never asks which norm you mean: any norm gives the standard topology of $\mathbb{R}^n$. The first thing to internalize about infinite dimensions is that this freedom *evaporates*. Choosing $L^2$ versus $L^1$ versus $L^\infty$ on $C[0,1]$ produces three genuinely different topologies, three genuinely different completions, and three genuinely different theorems. Picking the wrong norm is one of the most common ways graduate students lose hours.

A second consequence: every linear map between finite-dimensional normed spaces is continuous. Indeed, in coordinates the map is given by a matrix, all matrix entries are bounded by the operator norm, and equivalence of norms means continuity of one set of coordinate functions implies continuity of any other. In infinite dimensions, linear maps can be discontinuous. The differentiation operator $D: C^1[0,1] \to C[0,1]$, $f \mapsto f'$, is bounded *from $C^1$ to $C$* but if you put the sup norm on its domain — pretending to define $D: C[0,1] \to C[0,1]$ on the dense subspace $C^1[0,1]$ — it is unbounded: $\sin(n t)$ has sup norm $1$, but its derivative has sup norm $n$. Article 9 will revisit unbounded operators in detail.

## The $\ell^p$ Hierarchy and Inclusion Chains

For sequence spaces, the inclusion chain $\ell^1 \subset \ell^2 \subset \cdots \subset \ell^\infty$ holds (with strict inclusions). The reasoning is short: if $\sum |x_n|^p$ converges, then $|x_n| \to 0$, so $|x_n| \leq 1$ eventually, so $|x_n|^q \leq |x_n|^p$ for $q \geq p$, and the tail of $\sum |x_n|^q$ is dominated by $\sum |x_n|^p$. The constant prefactor needed makes the inclusion continuous: $\|x\|_q \leq \|x\|_p$ for $q \geq p$ when $x \in \ell^p$.

![Schauder basis approximation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/02_schauder_basis.png)

For Lebesgue function spaces on a *finite measure space* (such as $L^p[0,1]$), the inclusion goes the opposite direction: $L^q[0,1] \subset L^p[0,1]$ when $q \geq p$, by Hölder's inequality applied to $|f|^p \cdot 1$. On a non-finite measure space (like $\mathbb{R}$), neither inclusion holds — the function $1/(1 + |t|)$ is in $L^p(\mathbb{R})$ for $p > 1$ but not for $p = 1$, while a function blowing up like $|t|^{-1/p}$ near $0$ belongs to no $L^q$ for $q > p$. So $L^p$ inclusions are subtle and depend on the measure. For sequence spaces, where the counting measure is infinite but every singleton has measure $1$, only the "decay at infinity" direction matters, hence the simple chain.

![Inclusion chain l^1 ⊂ l^2 ⊂ ... ⊂ l^infinity for sequence spaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_v2_02_3_lp_chain.png)

### Numerical example

Take the harmonic-style sequence $x_n = 1/n$. Then $\sum 1/n = \infty$, so $x \notin \ell^1$. But $\sum 1/n^2 = \pi^2/6$, so $x \in \ell^2$ with $\|x\|_2 = \pi/\sqrt{6} \approx 1.283$. And $\sup |x_n| = 1$, so $x \in \ell^\infty$ with $\|x\|_\infty = 1$. The example shows where $\ell^2 \setminus \ell^1$ and $\ell^\infty \setminus \ell^2$ separate. To find an element of $\ell^\infty \setminus \ell^2$, the constant sequence $1, 1, 1, \ldots$ works: $\|\cdot\|_\infty = 1$ but $\sum 1 = \infty$.

The worked inclusions are essentially the only "free" structure on the $\ell^p$ family. The norms are inequivalent: an isometric copy of $\ell^1$ inside $\ell^2$ does not coincide with the topology of $\ell^2$, etc. The $\ell^p$ family is a beautiful one-parameter laboratory for testing whether a property of Banach spaces is general or specific to a particular geometry.

### Worked Numerical Example

Take the geometric sequence $x = (2^{-n})_{n \geq 0} = (1, 1/2, 1/4, 1/8, \ldots)$. I will compute its norm across four different $\ell^p$ spaces to verify the inclusion chain $\ell^1 \subset \ell^2 \subset \ell^4 \subset \ell^\infty$ and the monotonicity $\|x\|_q \leq \|x\|_p$ for $q > p$.

For $p=1$: $\|x\|_1 = \sum_{n=0}^\infty 2^{-n} = \frac{1}{1 - 1/2} = 2$.
For $p=2$: $\|x\|_2 = \big(\sum_{n=0}^\infty 4^{-n}\big)^{1/2} = \big(\frac{1}{1 - 1/4}\big)^{1/2} = \sqrt{4/3} \approx 1.1547$.
For $p=4$: $\|x\|_4 = \big(\sum_{n=0}^\infty 16^{-n}\big)^{1/4} = \big(\frac{1}{1 - 1/16}\big)^{1/4} = (16/15)^{1/4} \approx 1.0164$.
For $p=\infty$: $\|x\|_\infty = \sup_{n \geq 0} 2^{-n} = 1$.

The chain $1 \leq 1.0164 \leq 1.1547 \leq 2$ holds exactly as the theory predicts. The ratios also quantify how tight the inclusions are for this specific vector: $\|x\|_2 / \|x\|_1 \approx 0.577$, while $\|x\|_\infty / \|x\|_4 \approx 0.984$. The sequence decays fast enough that higher $p$ norms barely penalize the tail, pushing the value toward the sup norm. If I slow the decay to $y_n = (n+1)^{-2}$, the gap widens: $\|y\|_1 = \pi^2/6 \approx 1.645$, $\|y\|_2 = (\pi^4/90)^{1/2} \approx 1.040$, $\|y\|_\infty = 1$. The hierarchy is not just set containment; it is a quantitative filtration controlled by tail decay rates.

## Banach Spaces: When Completeness Lands

A normed space is a **Banach space** if it is complete in the metric induced by its norm. The completeness is a matter of *fact about a specific norm*, not just about the underlying vector space. Adding a norm structure to a Banach space and a non-Banach space sometimes produces the same vector space — for instance, $\ell^p$ as a vector space is a subspace of $\ell^\infty$, and they are very different objects when topologized.

![Banach space: a normed space that is complete with respect to its norm](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_v2_02_4_banach_complete.png)

### Examples that are Banach

- $\mathbb{R}^n$, $\mathbb{C}^n$ with any norm (finite-dimensional, hence complete).
- $C[a, b]$ with the sup norm. A Cauchy sequence is uniformly Cauchy, hence converges uniformly to a continuous function.
- $\ell^p$ for $1 \leq p \leq \infty$. Cauchy sequences converge term by term; the limit lies back in $\ell^p$ by Fatou's lemma.
- $L^p[a, b]$ for $1 \leq p \leq \infty$ — the Lebesgue completeness theorem (proved using Lebesgue's dominated convergence).
- $C_0(\mathbb{R}^n)$, continuous functions vanishing at infinity, with sup norm.
- Bounded operators $B(X, Y)$ when $Y$ is Banach.

### Examples that are *not* Banach

- $C[0,1]$ with the $L^1$-style metric $\int |f - g|$ (the example from Article 1). Completion is $L^1[0,1]$.
- The space of polynomials with sup norm — countable-dimensional, not closed in $C[0,1]$, completion is $C[0,1]$.
- The space of compactly supported continuous functions on $\mathbb{R}$ with sup norm — completion is $C_0(\mathbb{R})$.

### A trick characterization

There is a slick test: a normed space $V$ is a Banach space iff every absolutely convergent series in $V$ converges. (A series $\sum x_n$ is absolutely convergent if $\sum \|x_n\| < \infty$.) The proof is two lines: a Cauchy sequence has subsequences with $\|x_{n_{k+1}} - x_{n_k}\| < 2^{-k}$, and then $x_{n_K} = x_{n_1} + \sum_{k=1}^{K-1}(x_{n_{k+1}} - x_{n_k})$ is a partial sum of an absolutely convergent series; if those converge, the original Cauchy sequence converges (because Cauchy + a convergent subsequence implies convergent). Conversely, any absolutely convergent series has Cauchy partial sums, so completeness gives a limit. This is one of the most useful equivalent definitions when proving completeness of a specific space — checking that absolutely convergent series converge is often easier than handling Cauchy sequences directly.

## Schauder Bases

In a finite-dimensional normed space, every basis (in the algebraic sense) is also a "topological" basis: every vector is a finite linear combination of basis elements. In an infinite-dimensional Banach space, an algebraic (Hamel) basis is enormous — by the Baire theorem, an infinite-dimensional Banach space cannot have a countable Hamel basis — and useless for analysis.

![Absolute vs conditional convergence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/02_series_convergence.png)

The right notion for separable Banach spaces is a **Schauder basis**: a sequence $(e_n) \subset V$ such that every $x \in V$ has a unique representation $x = \sum_{n=1}^\infty c_n e_n$ as a *convergent series* (in the norm of $V$). The convergence is a topological condition, not an algebraic one, and it is what distinguishes Schauder bases from Hamel bases.

![Schauder basis: a sequence whose finite combinations are dense](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_v2_02_5_schauder.png)

### Examples

- The standard unit vectors $e_n = (0, \ldots, 0, 1, 0, \ldots)$ form a Schauder basis of $\ell^p$ for $1 \leq p < \infty$, but *not* of $\ell^\infty$ (the constant sequence $1$ has no Schauder expansion in this basis, because the partial sums truncate to vectors with sup norm $1$ from $1$).
- The trigonometric system $\{1, \cos(n t), \sin(n t)\}_{n \geq 1}$ is a Schauder basis (in fact, an orthonormal basis) of $L^2[0, 2\pi]$. It is *not* a Schauder basis of $L^p[0, 2\pi]$ for $p \neq 2, p > 1$ — the partial sums of Fourier series do not converge to $f$ in $L^p$ norm for general $f$ when $p = 1$ or $p = \infty$.
- The Haar system is a Schauder basis of every $L^p[0,1]$ for $1 \leq p < \infty$.

The existence of a Schauder basis implies separability (the finite rational combinations of basis vectors are dense), but the converse is false — Per Enflo's 1973 example of a separable Banach space without a Schauder basis settled a long-standing question.

### Numerical example

In $\ell^2$, take $x = (1/n)_{n \geq 1}$. The Schauder expansion in the standard basis is $x = \sum_{n=1}^\infty (1/n) e_n$. The partial sums are $S_N = (1, 1/2, 1/3, \ldots, 1/N, 0, 0, \ldots)$, and $\|x - S_N\|_2^2 = \sum_{n=N+1}^\infty 1/n^2 \to 0$ as $N \to \infty$. Coefficients are unique because if $\sum c_n e_n = 0$ then taking the inner product with $e_m$ forces $c_m = 0$.

### Why this matters

Schauder bases let me reduce questions about an entire infinite-dimensional Banach space to questions about coefficient sequences. Many of the most concrete things one can do in functional analysis — Fourier analysis, wavelet decomposition, finite-dimensional approximation in numerics — are exactly Schauder-basis arguments. The Schauder basis property of an orthonormal basis in a separable Hilbert space (Article 3) is the cleanest version, and historically the model for the general definition.

## Equivalent Norms in Practice

Norms on the same vector space are *equivalent* if they have the same Cauchy sequences, equivalently the same convergent sequences, equivalently the same topology. In finite dimensions all norms are equivalent (above); in infinite dimensions they are usually not.

![Comparing balls of equivalent norms on R^2 by inclusion](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_v2_02_6_equiv_norms.png)

A few observations.

- If $\|\cdot\|_a$ and $\|\cdot\|_b$ are equivalent on $V$, then $V$ is complete with respect to one iff with respect to the other. So "equivalent" is a strong relation: it preserves the Banach property.
- If $V$ is a Banach space under $\|\cdot\|_a$ and $\|\cdot\|_b$, and $\|x\|_b \leq C \|x\|_a$ for all $x$ (one-sided bound), then by the open mapping theorem (Article 6) the reverse bound $\|x\|_a \leq C' \|x\|_b$ holds for some $C'$, so the norms are *automatically* equivalent. This is one of those mildly surprising results whose proof has to wait until we have the open mapping theorem.
- The norms $\|f\|_\infty$ and $\|f\|_1$ on $C[0,1]$ are not equivalent. The sequence $f_n(t) = t^n$ has $\|f_n\|_\infty = 1$ but $\|f_n\|_1 = 1/(n+1) \to 0$.

### Numerical example

On $\mathbb{R}^2$, the inequalities $\|x\|_\infty \leq \|x\|_2 \leq \|x\|_1 \leq \sqrt{2} \|x\|_2 \leq 2 \|x\|_\infty$ hold pointwise. So all three norms are equivalent, and the equivalence constants are not far from $1$. As $n$ grows, the constants in $\mathbb{R}^n$ grow like $n^{1/p - 1/q}$, but they remain finite for any fixed $n$. Send $n \to \infty$ and the constants blow up; this is the precise sense in which finite-dimensional equivalence fails on $\ell^p$.

## Quotient Spaces

Let $V$ be a normed space and $W \subset V$ a closed subspace. The **quotient space** $V/W$ consists of equivalence classes $x + W$ under the relation $x \sim y \iff x - y \in W$. The natural norm on $V/W$ is
$$\|x + W\|_{V/W} = \inf_{w \in W} \|x - w\| = d(x, W).$$
This is genuinely a norm — positive definiteness uses that $W$ is *closed* — and the projection $\pi: V \to V/W$, $\pi(x) = x + W$, is a continuous linear surjection with $\|\pi\| \leq 1$.

If $V$ is a Banach space, then $V/W$ is a Banach space — a Cauchy sequence in $V/W$ can be lifted (using absolute convergence) to a Cauchy sequence in $V$ whose limit projects to the desired limit in the quotient.

### Example

In $L^1(\mathbb{R})$, consider the closed subspace $W$ of functions whose integral is zero. The quotient $L^1(\mathbb{R}) / W$ is one-dimensional, isomorphic to $\mathbb{R}$ via $\overline{f} \mapsto \int f$. More structurally, the quotient construction lets me kill any unwanted closed subspace; in PDE, the Sobolev space $H^1(\Omega) / \mathbb{R}$ (constants modded out) is the right space for variational problems with Neumann boundary conditions.

### Why this matters

Quotients are the way I deal with subspaces that I want to ignore. In the theory of operators, $V / \ker(T)$ identifies elements of $V$ that $T$ sends to the same place. The first isomorphism theorem then says $V / \ker(T) \cong \mathrm{Range}(T)$ via the induced map $\widetilde T: V/\ker(T) \to \mathrm{Range}(T)$. When $T$ is bounded with closed range, $\widetilde T$ is a bicontinuous isomorphism — Article 6 will exploit this.

### Worked Numerical Example

I will compute a quotient norm explicitly in a finite-dimensional setting where the geometry is visible. Let $V = \mathbb{R}^3$ equipped with the $\ell^1$ norm $\|v\|_1 = |v_1| + |v_2| + |v_3|$. Let $W = \{(t, t, t) : t \in \mathbb{R}\}$ be the diagonal subspace. Take the coset represented by $x = (1, 0, 0)$. The quotient norm is defined as $\|x + W\|_{V/W} = \inf_{w \in W} \|x - w\|_1$.

Parametrize $w = (t, t, t)$. Then $x - w = (1 - t, -t, -t)$, and the objective function is $f(t) = |1 - t| + 2|t|$. I minimize this piecewise linear function by checking the critical regions:
- If $t < 0$, $f(t) = (1 - t) - 2t = 1 - 3t > 1$.
- If $0 \leq t \leq 1$, $f(t) = (1 - t) + 2t = 1 + t$, which attains its minimum $1$ at $t = 0$.
- If $t > 1$, $f(t) = (t - 1) + 2t = 3t - 1 > 2$.

The infimum is exactly $1$, achieved at $t = 0$. So $\|(1,0,0) + W\|_{V/W} = 1$. Notice that the minimizer is unique here, but if I change $x$ to $(1, 1, 0)$, the objective becomes $|1-t| + |1-t| + |t| = 2|1-t| + |t|$, which is minimized on the entire interval $t \in [0, 1]$, yielding the same quotient norm $1$. The $\ell^1$ geometry produces flat valleys in the distance function, and the quotient norm faithfully records the depth of the valley, not the location of the minimizer.

## Sequence Spaces $c_0$, $c$, and $\ell^\infty$

Three closely related sequence spaces deserve their own paragraph:

- $\ell^\infty$: bounded sequences, with $\|x\|_\infty = \sup |x_n|$.
- $c$: convergent sequences, with $\|x\|_\infty = \sup |x_n|$.
- $c_0$: sequences converging to $0$, with $\|x\|_\infty = \sup |x_n|$.

All three are Banach spaces under the sup norm. The inclusions $c_0 \subset c \subset \ell^\infty$ are all strict and all closed, since the limit of a uniformly convergent sequence of $c_0$-sequences is in $c_0$, etc.

![Comparison of c_0, c, l^infinity sequence spaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_v2_02_7_seq_spaces.png)

The space $\ell^\infty$ is *not* separable (mentioned in Article 1). For each $A \subseteq \mathbb{N}$, the indicator $\mathbb{1}_A \in \ell^\infty$, and $\|\mathbb{1}_A - \mathbb{1}_B\|_\infty = 1$ for $A \neq B$. There are uncountably many such indicators, so $\ell^\infty$ has an uncountable $1$-separated subset and cannot have a countable dense subset. Both $c$ and $c_0$ *are* separable, with the finitely supported rational sequences as a countable dense set. Separability matters — it is the regularity that lets us do duality nicely (Article 4).

The standard basis $(e_n)$ is a Schauder basis of $c_0$ and of $\ell^p$ for $1 \leq p < \infty$, but not of $c$ (the constant sequence $1$ is in $c$ but its Schauder expansion in $(e_n)$ does not converge in sup norm). For $c$, the right Schauder basis is $\{1, e_1, e_2, \ldots\}$ — the constant $1$ plus the standard basis vectors. This adjustment is one of those small, irritating reminders that the choice of basis is a delicate art.

## Why Completeness is the Dividing Line

Here is the main theorem-pattern that runs through Banach space theory: completeness is what lets us trade pointwise hypotheses for global conclusions.

- **Banach-Steinhaus** (Article 6): pointwise boundedness of a family of operators implies uniform boundedness.
- **Open mapping theorem** (Article 6): a surjective bounded linear map between Banach spaces is automatically open.
- **Closed graph theorem** (Article 6): a linear map between Banach spaces with closed graph is automatically bounded.
- **Hahn-Banach** (Article 4): any continuous linear functional on a subspace extends to the whole space without enlarging its norm.

Of these, only Hahn-Banach is truly purely algebraic (it does not require completeness). The other three crucially require it. In the rest of the series, completeness is the assumption that gets cashed in over and over, often invisibly in the background.

Banach spaces are also the right setting for *operator algebras*, because $B(X)$ is itself a Banach space when $X$ is — and even an algebra under composition, with $\|S T\| \leq \|S\| \|T\|$ (a Banach algebra). Spectral theory (Articles 8, 9) takes place in this setting.

## Bounded Linear Operators: The Operator Norm

A linear map $T: X \to Y$ between normed spaces is **bounded** if there exists a constant $C \geq 0$ with $\|T x\|_Y \leq C \|x\|_X$ for all $x \in X$. The smallest such $C$ is the **operator norm**:
$$\|T\| = \sup_{x \neq 0} \frac{\|T x\|_Y}{\|x\|_X} = \sup_{\|x\|_X \leq 1} \|T x\|_Y = \sup_{\|x\|_X = 1} \|T x\|_Y.$$
The three formulations agree by homogeneity. A linear operator is bounded iff it is continuous iff it is continuous at $0$. (For linear maps these three conditions are equivalent.) So "bounded" and "continuous" are interchangeable for linear operators between normed spaces.

The space $B(X, Y)$ of bounded linear operators is a normed space under the operator norm, and it is complete whenever $Y$ is — a Cauchy sequence of operators is pointwise Cauchy, so converges pointwise to some $T$, and the operator norm bound passes to the limit. So $B(X, Y)$ is a Banach space whenever $Y$ is.

### Numerical example

The shift operator $S: \ell^2 \to \ell^2$ defined by $S(x_1, x_2, x_3, \ldots) = (0, x_1, x_2, \ldots)$ is bounded with $\|S\| = 1$. To see this, $\|S x\|_2^2 = \sum_{n \geq 2} |x_{n-1}|^2 = \|x\|_2^2$, so $\|S x\|_2 = \|x\|_2$ for every $x$, hence $\|S\| = 1$. The same operator is an isometry but not surjective, since the first coordinate of $S x$ is always $0$. This is the kind of structural fact that has no analog in finite dimensions, where every isometry of a finite-dimensional space onto itself is automatically surjective.

The differentiation operator $D: C^1[0,1] \to C[0,1]$, $f \mapsto f'$, is bounded if I put the natural $C^1$ norm $\|f\|_{C^1} = \|f\|_\infty + \|f'\|_\infty$ on the domain: $\|D f\|_\infty = \|f'\|_\infty \leq \|f\|_{C^1}$, so $\|D\| \leq 1$, and the inequality is sharp on functions like $f(t) = \sin(2\pi t) / 2\pi$. But if I put the sup norm on $C^1[0,1]$ (without the derivative term), $D$ is unbounded — the family $f_n(t) = \sin(n \pi t) / n$ has sup norm $1/n \to 0$ but $D f_n = \pi \cos(n\pi t)$ has sup norm $\pi$, refusing to vanish. So the same algebraic operator is bounded under one norm choice and unbounded under another. The pairing of operator and norm is irreducible.

### Worked Numerical Example

The operator norm depends heavily on the choice of norm on the domain and codomain. Consider the matrix $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ acting on $\mathbb{R}^2$. I will compute $\|A\|$ when the space carries the $\ell^1$ norm versus the $\ell^\infty$ norm.

For $A: (\mathbb{R}^2, \|\cdot\|_1) \to (\mathbb{R}^2, \|\cdot\|_1)$, the operator norm equals the maximum absolute column sum. Column 1 sums to $|1| + |3| = 4$. Column 2 sums to $|2| + |4| = 6$. So $\|A\|_1 = 6$. Verification: take $x = (0, 1)$. $\|x\|_1 = 1$. $Ax = (2, 4)$, $\|Ax\|_1 = 6$. The bound is attained.

For $A: (\mathbb{R}^2, \|\cdot\|_\infty) \to (\mathbb{R}^2, \|\cdot\|_\infty)$, the operator norm equals the maximum absolute row sum. Row 1 sums to $|1| + |2| = 3$. Row 2 sums to $|3| + |4| = 7$. So $\|A\|_\infty = 7$. Verification: take $x = (1, 1)$. $\|x\|_\infty = 1$. $Ax = (3, 7)$, $\|Ax\|_\infty = 7$. The bound is attained.

The same linear map has norm $6$ in one geometry and $7$ in another. If I switch to the Euclidean norm $\ell^2$, the norm is the largest singular value: $\sqrt{\lambda_{\max}(A^T A)}$. $A^T A = \begin{pmatrix} 10 & 14 \\ 14 & 20 \end{pmatrix}$, eigenvalues are $15 \pm \sqrt{221} \approx 29.866, 0.134$. The norm is $\sqrt{29.866} \approx 5.465$. Three norms, three different numbers. The operator norm is not an intrinsic property of the matrix; it is a property of the matrix paired with the specific metric geometry of the space.

## Bounded vs Algebraic: A Cautionary Tale

Linearity alone does not buy continuity in infinite dimensions. Concretely, on the algebraic dual of $\ell^2$ (the linear functionals not required to be continuous) one can use the axiom of choice to extend a linear functional that does whatever you want on a Hamel basis. The axiom-of-choice extension is rarely continuous unless you start with continuous data.

A constructive example: take $V = \ell^2_{fin}$, the dense subspace of $\ell^2$ consisting of finitely supported sequences. Define $\varphi: V \to \mathbb{R}$ by $\varphi(x) = \sum_n n x_n$ (a finite sum since $x$ is finitely supported). On the basis vectors $e_n$, $\varphi(e_n) = n$, but $\|e_n\|_2 = 1$, so $\varphi$ is unbounded on the unit ball. The functional $\varphi$ is linear and well-defined, but discontinuous. It does not extend continuously to all of $\ell^2$. The contrast with the previous paragraph is that for *bounded* operators between Banach spaces, continuity is automatic — but bounded is the input hypothesis, not a deduction.

This distinction is why every theorem about operators and functionals in this series starts with "let $T$ be bounded" or "let $T$ be continuous." Without that, there is essentially no theory.

## Closed Subspaces Are Not Automatic

In a metric space, every closed subset is "automatically" closed in the natural sense. In an *infinite-dimensional* normed space, the analogous claim for subspaces is more subtle. A linear subspace need not be closed; if it is finite-dimensional it is automatically closed (a consequence of finite-dimensional equivalence of norms), but countable-dimensional subspaces are usually dense and not closed.

For example, the subspace of polynomials inside $C[0,1]$ (sup norm) is dense (by Weierstrass) but not closed; its closure is $C[0,1]$. The subspace $c_{00}$ of finitely supported sequences inside $\ell^2$ is dense but not closed.

A closed subspace of a Banach space is itself a Banach space (the induced norm is complete because Cauchy sequences in the subspace are Cauchy in the ambient space, converge there, and the limit is in the subspace by closedness). This is what makes closed subspaces the "subobjects" of choice in the category of Banach spaces.

### Why this matters

The "closed range theorem" for bounded operators (Article 4) classifies when the range of an operator is closed, and the answer involves the dual operator. Closedness of subspaces is also exactly what controls quotients (above) and is the input for the open mapping theorem. As a working rule: every theorem in functional analysis that mentions a subspace assumes it is closed, even when the assumption is buried in the language.

## A Geometric Aside: The Shape of the Unit Ball

The unit ball $B = \{ x : \|x\| \leq 1 \}$ of a normed space encodes everything about the norm — by homogeneity, the norm is the *Minkowski functional* of the ball, $\|x\| = \inf\{ t > 0 : x/t \in B \}$. So studying norms on a vector space and studying balanced convex absorbing sets are the same activity.

The shape of the ball reveals the "personality" of the space.

- $\ell^1$ has a sharp polyhedral ball — corners at the unit vectors. The dual $\ell^\infty$ has a cubical ball, sharp on faces. The duality between corners and faces is part of why $(\ell^1)^* = \ell^\infty$.
- $\ell^p$ for $1 < p < \infty$ has a smooth and *strictly convex* ball: no flat faces, no corners. Strict convexity means $\|x\| = \|y\| = 1$ and $\|x + y\|/2 = 1$ force $x = y$. This is the geometric content of *uniqueness of best approximation* in such spaces — every closed convex set has a unique closest point. Strict convexity fails in $\ell^1$ (the unit segment from $e_1$ to $e_2$ lies on the boundary of the ball) and in $\ell^\infty$.
- $\ell^2$ has the roundest ball, an actual sphere — and this is why $\ell^2$ is the unique $\ell^p$ space for which a parallelogram law holds (Article 3).

When a normed space has the property that the dual operator-norm ball is also strictly convex (which boils down to *smoothness* of the original ball), things are very pleasant. The $\ell^p$ spaces for $1 < p < \infty$ have these properties; that is one reason analysis prefers them and avoids $\ell^1, \ell^\infty$ when possible.

### Numerical example

Compare the boundary of the unit ball of $\ell^1$ in $\mathbb{R}^2$ — the diamond $|x| + |y| = 1$ — with the boundary of the $\ell^2$ ball, the circle $x^2 + y^2 = 1$. At the point $(1/2, 1/2) \cdot \sqrt{2} = (1/\sqrt{2}, 1/\sqrt{2})$ on the $\ell^2$ unit circle, the tangent line is unique (a smooth point). At the corresponding $\ell^1$ vertex $(1, 0)$, *infinitely many* lines are "supporting hyperplanes" — every line through $(1, 0)$ with slope between $-1$ and $1$ (exclusive) passes through the half-plane $\{x + y < 1\}$ near $(1,0)$ and so supports the ball. This profusion of supporting hyperplanes is exactly what produces non-unique best approximations and non-uniqueness in many optimization problems posed in $\ell^1$.

## Reflexivity, Separability, Density: Three Cheap Words That Do A Lot

Banach space theory has a small vocabulary of structural adjectives that get used over and over: *reflexive*, *separable*, *uniformly convex*. They are easy to define but it takes a while to develop a feel for what each one *buys* you.

- **Separable.** A Banach space is *separable* if it has a countable dense subset. $\ell^p$ for $1 \leq p < \infty$ is separable (finitely supported rational sequences); $\ell^\infty$ is not. $L^p[0,1]$ for $1 \leq p < \infty$ is separable (rational-coefficient polynomials); $L^\infty[0,1]$ is not. Separability is the *regularity* hypothesis that makes "extract a convergent subsequence" arguments work without nets — sequential compactness is enough.
- **Reflexive.** A Banach space is *reflexive* if the canonical embedding $J: X \to X^{**}$ is surjective (Article 4). $\ell^p$ and $L^p$ for $1 < p < \infty$ are reflexive; $\ell^1, \ell^\infty, L^1, L^\infty, c_0, C[K]$ are not. Reflexivity buys *weak compactness of bounded sets*, which is what makes the direct method of the calculus of variations work.
- **Uniformly convex.** A Banach space is *uniformly convex* if for every $\varepsilon > 0$ there exists $\delta > 0$ with $\|x\| = \|y\| = 1$ and $\|x - y\| > \varepsilon$ implies $\|(x+y)/2\| < 1 - \delta$. Geometrically: the unit ball is "uniformly round" with no flat patches. Uniform convexity implies reflexivity (by a theorem of Milman), and it gives "best approximation" theorems with quantitative rates. $\ell^p$ and $L^p$ for $1 < p < \infty$ are uniformly convex; the others mentioned above are not.

These three properties form a hierarchy: uniformly convex implies reflexive, both imply some weak compactness, separable is sometimes orthogonal to all of these (e.g. $C[0,1]$ is separable but not reflexive). The most-used combination in PDE is *separable + reflexive*, which is what reflexive Sobolev spaces $W^{k,p}$ for $1 < p < \infty$ deliver.

### A practical example

In a variational problem like minimizing a Dirichlet energy, you typically have:
1. *A separable reflexive function space* (e.g. $H^1_0(\Omega)$).
2. *A coercive convex energy* whose sublevel sets are bounded.
3. *A convex constraint set* (closed in norm).

The argument: take a minimizing sequence (bounded by coercivity), extract a weakly convergent subsequence (by reflexivity, on a bounded set, with metrizability of the weak topology by separability), pass to the limit using lower semicontinuity (which is automatic for convex norm-continuous functionals on a reflexive space). The minimizer exists.

Strip any one of the three properties and the argument breaks.

## Distance to a Subspace and Best Approximation

Given a closed subspace $W \subseteq X$ and a point $x \in X$, the **distance** is $d(x, W) = \inf_{w \in W} \|x - w\|$. The infimum is attained — there exists $w^* \in W$ achieving it — when $X$ is *reflexive* and $W$ is closed (the closed unit ball of $W$ is weakly compact, and minimizing $\|x - w\|$ over $W$ becomes a minimization on a bounded weakly compact set with weakly l.s.c. functional).

The infimum is *uniquely* attained when $X$ is *strictly convex* — every closed convex set has a unique closest point. For non-strictly-convex spaces ($\ell^1, \ell^\infty$, $C[K]$), best approximations may be non-unique.

### Numerical example

In $L^2[0, 1]$, find the best polynomial approximation of degree $\leq 2$ to $f(t) = t^3$. Best in $L^2$ means orthogonal projection onto the subspace $\{1, t, t^2\}$. Computing the Gram matrix $\big(\int_0^1 t^{i+j}\,dt\big)_{i,j=0}^2$ — the $3\times 3$ Hilbert matrix — and the right-hand side $\big(\int_0^1 t^3 \cdot t^j\,dt\big)_{j=0}^2 = (1/4, 1/5, 1/6)$, then solving the linear system, yields the best approximation $p^*(t) = 1/20 - 3t/5 + 3t^2/2$ (after computation). The residual $\|t^3 - p^*\|_{L^2}$ comes out to $1/20\sqrt{7} \approx 0.0189$.

For comparison, in the sup norm on $C[0,1]$, the best polynomial approximation of degree $\leq 2$ to $t^3$ is *not* the $L^2$ projection. It is given by Chebyshev approximation (Article 4), and the answer is different — the sup-norm best approximation equioscillates between $-c$ and $+c$ at four points of $[0,1]$. The norm-dependence of best approximation is one of the most concrete reasons different $L^p$ choices lead to different theorems.

### Worked Numerical Example

I will compute the distance from a point to a line in $\mathbb{R}^2$ under two different norms to demonstrate how strict convexity controls uniqueness. Let $W = \{(t, t) : t \in \mathbb{R}\}$ and $x = (1, 0)$.

Under the $\ell^2$ norm, the distance is the orthogonal projection. The projection of $(1,0)$ onto the line $y=x$ is $(1/2, 1/2)$. The distance is $\|(1,0) - (1/2, 1/2)\|_2 = \|(1/2, -1/2)\|_2 = \sqrt{1/4 + 1/4} = 1/\sqrt{2} \approx 0.7071$. The minimizer $t = 1/2$ is unique. The round unit ball forces a single contact point.

Under the $\ell^1$ norm, I minimize $g(t) = \|(1-t, -t)\|_1 = |1-t| + |t|$. For any $t \in [0, 1]$, $g(t) = (1-t) + t = 1$. For $t < 0$, $g(t) = 1 - 2t > 1$. For $t > 1$, $g(t) = 2t - 1 > 1$. The distance is exactly $1$, but it is achieved by every point on the segment from $(0,0)$ to $(1,1)$. The best approximation is not a single vector; it is an entire continuum of vectors. The diamond-shaped unit ball has a flat edge parallel to the subspace, so the expanding ball touches the line along a whole face. If you are running a numerical optimization routine in $\ell^1$, your solver will return whichever point the algorithm stumbles into first. The mathematics does not pick a winner.

## Strictly Convex, Smooth, Uniformly Convex: A Geometric Glossary

The unit ball of a Banach space encodes the entire norm. Three conditions on its shape are particularly important.

- **Strictly convex.** $\|x\| = \|y\| = 1$ and $x \neq y$ implies $\|(x+y)/2\| < 1$. Geometrically: the unit sphere has no flat segments. Equivalent to: every closed convex set has at most one closest point to any given point.
- **Smooth.** Every point of the unit sphere has a unique supporting hyperplane (Article 4). Equivalent to: the norm is Gateaux-differentiable at every nonzero point. Strict convexity of $X$ corresponds (under reflexivity) to smoothness of $X^*$ and vice versa, by a duality theorem.
- **Uniformly convex.** Defined above; quantifies "no flat patches" with a uniform modulus.

The $\ell^p$ family, $1 < p < \infty$, is the canonical example of all three properties. The boundary cases $p = 1$ and $p = \infty$ fail all three: $\ell^1$ has corners (vertices) and $\ell^\infty$ has flat faces. These geometric pathologies translate into theorem failures: in $\ell^1$, best approximation can be non-unique (the supporting hyperplane at a vertex is non-unique); in $\ell^\infty$, weak-norm convergence does not control norm convergence (Schur property is the closest substitute).

The "preferred" Banach spaces of analysis — the ones in which most theorems work cleanly — are uniformly convex. Hilbert space is the limiting case where uniform convexity is encoded in the parallelogram law itself.

### Worked Numerical Example

Strict convexity demands that the midpoint of any two distinct unit vectors lies strictly inside the unit ball. I will test this condition on $\ell^1$ and $\ell^4$ in $\mathbb{R}^2$ using the same pair of vectors.

Take $u = (1, 0)$ and $v = (0, 1)$. In any $\ell^p$ norm, $\|u\|_p = 1$ and $\|v\|_p = 1$. Their midpoint is $m = (u+v)/2 = (1/2, 1/2)$.

In $\ell^1$: $\|m\|_1 = |1/2| + |1/2| = 1$. The midpoint sits exactly on the boundary. The condition $\|(u+v)/2\| < 1$ fails. The unit sphere contains the entire line segment connecting $u$ and $v$. This flat face is the geometric signature of non-strict convexity.

In $\ell^4$: $\|m\|_4 = \big(|1/2|^4 + |1/2|^4\big)^{1/4} = (2/16)^{1/4} = (1/8)^{1/4} = 8^{-0.25} \approx 0.5946$. The midpoint drops deep into the interior. The inequality $\|m\|_4 < 1$ holds decisively. If I push the vectors closer, say $u = (1, 0)$ and $v = (\cos \theta, \sin \theta)$ with $\theta = 0.1$, the $\ell^4$ midpoint norm is $\approx 0.9987$, still strictly less than $1$. The boundary curves inward everywhere. This curvature is what guarantees that optimization problems posed in $\ell^4$ have unique solutions, while the same problems in $\ell^1$ require tie-breaking rules or regularization to pick a single answer.

## Counterexample: Why the Definition Cannot Be Weakened

The definition of a Banach space requires completeness: every Cauchy sequence must converge to a limit inside the space. If I drop completeness and work in a mere normed space, the absolutely convergent series test fails, and bounded operators refuse to extend. I will demonstrate this with $C[0,1]$ equipped with the $L^1$ norm $\|f\|_1 = \int_0^1 |f(t)|\,dt$.

Define a sequence of continuous functions $(f_n)_{n \geq 3}$ by
$$
f_n(t) = \begin{cases}
0 & 0 \leq t \leq \frac{1}{2} - \frac{1}{n}, \\
\frac{n}{2}\big(t - (\frac{1}{2} - \frac{1}{n})\big) & \frac{1}{2} - \frac{1}{n} < t < \frac{1}{2}, \\
1 & \frac{1}{2} \leq t \leq 1.
\end{cases}
$$
Each $f_n$ is a continuous ramp that climbs from $0$ to $1$ over an interval of width $1/n$. For $m > n$, the difference $f_m - f_n$ is nonzero only on $[\frac{1}{2} - \frac{1}{n}, \frac{1}{2}]$. The area between the two ramps is bounded by the area of a triangle of base $1/n$ and height $1$, so $\|f_m - f_n\|_1 \leq \frac{1}{2n}$. Given $\varepsilon > 0$, pick $N > 1/(2\varepsilon)$; then $\|f_m - f_n\|_1 < \varepsilon$ for all $m, n \geq N$. The sequence is Cauchy in $(C[0,1], \|\cdot\|_1)$.

If the space were Banach, $(f_n)$ would converge to some continuous $f \in C[0,1]$. But pointwise, $f_n(t) \to 0$ for $t < 1/2$ and $f_n(t) \to 1$ for $t \geq 1/2$. The $L^1$ limit is the step function $\chi_{[1/2, 1]}$, which is discontinuous. No continuous function can serve as the limit, because if $\|f_n - g\|_1 \to 0$ for some $g \in C[0,1]$, then $g$ must equal the step function almost everywhere, forcing a jump discontinuity. The space has a hole exactly where the limit should land.

This failure breaks the series characterization of completeness. Consider the telescoping series $\sum_{k=3}^\infty (f_{k+1} - f_k)$. The norms satisfy $\|f_{k+1} - f_k\|_1 \leq \frac{1}{2k}$, so $\sum \|f_{k+1} - f_k\|_1$ diverges logarithmically, but I can thin the sequence to $n_j = 2^j$ to force absolute convergence: $\|f_{2^{j+1}} - f_{2^j}\|_1 \leq 2^{-(j+1)}$, making $\sum \|f_{2^{j+1}} - f_{2^j}\|_1 \leq 1/2 < \infty$. The series is absolutely convergent in the normed space, yet its partial sums $f_{2^J}$ converge to a discontinuous function. The series does not converge inside $C[0,1]$. Completeness is not a decorative assumption; it is the structural guarantee that absolutely convergent series actually sum to something in the space. Without it, the calculus of infinite sums collapses.

## Why I Care

I first internalized the weight of completeness during a graduate numerical analysis assignment in 2018. The task was to implement a gradient descent scheme for a convex energy functional on a space of functions over $[0,1]$. I coded the iterates as piecewise linear interpolants and used the sup norm $\|f\|_\infty$ as my stopping criterion, because it felt natural: if the maximum pointwise change drops below $10^{-6}$, the solution has stabilized.

The code ran for four hours. The residual stagnated at $0.04$. The iterates developed increasingly sharp spikes near $t = 0.3$, and the derivative estimates oscillated wildly between $\pm 800$. I checked the step size, the line search, the discretization density. Everything was textbook. I finally printed the $H^1$ seminorm $\int |f_n'|^2$ of the iterates. It was converging smoothly to $12.7$. The sequence was Cauchy in the Sobolev norm, but not in the sup norm. I was measuring distance with a ruler that the space did not respect. The minimizing sequence was trying to converge to a function with a kink, which lives in $H^1$ but not in $C^1$. My sup-norm metric was blind to the actual convergence happening in the energy space, and the algorithm was chasing a ghost.

I switched the stopping criterion to $\|f_{n+1} - f_n\|_{L^2} + \|f_{n+1}' - f_n'\|_{L^2} < 10^{-5}$. The solver terminated in 38 iterations. The spikes vanished. The limit function matched the analytical weak solution to four decimal places. That afternoon burned the lesson into me: completeness and norm choice are not pedantic decorations. They dictate whether your iterative process has a floor to land on or falls through a hole in the topology. If the space is not complete under the metric you are actually computing, your algorithm is mathematically homeless.

## Common Pitfall

A persistent misconception among students is that a linear operator defined on a dense subspace of a Banach space automatically extends to a bounded operator on the whole space. The reasoning usually goes: the subspace is dense, so every point is a limit of subspace elements; linearity preserves limits; therefore the extension exists and is continuous. This is false. Density plus linearity buys nothing unless the operator is already bounded on the dense subspace.

Take $X = \ell^2$ and let $D = c_{00}$ be the subspace of finitely supported sequences. $D$ is dense in $\ell^2$. Define $T: D \to \mathbb{R}$ by $T(x) = \sum_{n=1}^\infty n x_n$. The sum is finite for every $x \in D$, so $T$ is well-defined and linear. Now consider the sequence $x^{(k)} = \sum_{n=1}^k \frac{1}{n^2} e_n$. Each $x^{(k)}$ lies in $D$. The sequence converges in $\ell^2$ to $x = (1/n^2)_{n \geq 1}$, since $\|x - x^{(k)}\|_2^2 = \sum_{n=k+1}^\infty n^{-4} \to 0$. In fact, $\|x^{(k)}\|_2 \leq (\sum_{n=1}^\infty n^{-4})^{1/2} = (\pi^4/90)^{1/2} \approx 1.041$, so the sequence stays inside a small ball.

Apply $T$ to the sequence: $T(x^{(k)}) = \sum_{n=1}^k n \cdot \frac{1}{n^2} = \sum_{n=1}^k \frac{1}{n}$. This is the harmonic series. $T(x^{(k)}) \to \infty$ as $k \to \infty$. The operator maps a convergent, bounded sequence to an unbounded divergent sequence. No continuous extension to $\ell^2$ exists, because any extension $\widetilde{T}$ would have to satisfy $\widetilde{T}(x) = \lim T(x^{(k)}) = \infty$, which is not a real number. The operator is unbounded on the unit ball of $D$, and density does not rescue it.

The correct theorem requires a hypothesis students frequently skip: if $T: D \to Y$ is linear and *bounded* (i.e., $\|Tx\|_Y \leq C \|x\|_X$ for all $x \in D$), and $Y$ is complete, then $T$ extends uniquely to a bounded operator on $\overline{D}$. Boundedness is the ticket. Without it, density is just a topological fact with no analytical leverage.

## Looking Ahead

Banach spaces give us everything we need to do *operator analysis*: norm, completeness, bounded linear maps, quotients, Schauder bases, and inequalities like Hölder and Minkowski. But there is one more axiom we have not yet imposed: an *inner product*, the abstraction of dot product. An inner product makes available the parallelogram law, orthogonality, and projections — the geometry of $\mathbb{R}^n$ resurrected in infinite dimensions. The next article introduces Hilbert spaces and shows what happens when the unit ball is round.

The next article (Article 3) introduces the **Baire category theorem** and its three great consequences: the *uniform boundedness principle* (Banach-Steinhaus theorem), the *open mapping theorem*, and the *closed graph theorem*. These are the power tools of Banach space theory --- they take completeness as input and produce sweeping structural conclusions about bounded linear operators. The uniform boundedness principle says that pointwise boundedness of a family of operators implies uniform boundedness (in the operator norm). The open mapping theorem says that a surjective bounded linear map between Banach spaces is automatically an open map. The closed graph theorem says that a closed linear map between Banach spaces is automatically bounded. Each of these results is genuinely surprising --- it extracts a global conclusion from seemingly local hypotheses --- and each relies crucially on completeness. Once you have them in hand, the theory begins to feel remarkably powerful.

---

### Specific Questions Ahead

We now have the vocabulary of normed spaces, the machinery of completeness, and the operator norm to control linear maps. The geometry is functional, but it remains blunt. I can measure lengths and distances, but I cannot yet measure angles. I can project onto subspaces in strictly convex spaces, but the construction requires solving an optimization problem from scratch each time. The next article introduces the inner product, which upgrades a Banach space to a Hilbert space and restores the full geometric intuition of $\mathbb{R}^n$ to infinite dimensions.

With the current toolkit in hand, you are equipped to ask precise questions that the inner product will answer:
1. How do I define orthogonality in a space where the unit ball is not a sphere, and why does the parallelogram law single out exactly one norm that admits it?
2. When does a closed subspace $W \subset X$ admit a complementary subspace $W^\perp$ such that $X = W \oplus W^\perp$ algebraically and topologically?
3. Why does the dual space $X^*$ of a Hilbert space look exactly like $X$ itself, while the dual of $L^1$ or $C[0,1]$ is a completely different object?
4. How can I construct the orthogonal projection onto a closed convex set without running an iterative minimization algorithm?

You already know why completeness matters: it guarantees that Cauchy sequences land somewhere. You know why strict convexity matters: it forces best approximations to be unique. You know how the operator norm controls continuity. Hilbert spaces package all three properties into a single algebraic identity. The inner product $\langle x, y \rangle$ generates the norm via $\|x\| = \sqrt{\langle x, x \rangle}$, and the Cauchy-Schwarz inequality bridges algebra and topology immediately.

The centerpiece of the next article is the **Riesz Representation Theorem**. It states that for every continuous linear functional $f$ on a Hilbert space $H$, there exists a unique vector $y \in H$ such that $f(x) = \langle x, y \rangle$ for all $x \in H$, and $\|f\| = \|y\|$. This theorem collapses the abstract dual space back into the original space. It turns duality from a separate analytical construction into a geometric rotation. Once you have Riesz, the Lax-Milgram theorem, spectral decompositions, and weak formulations of PDEs stop being mysterious analytical tricks and become direct consequences of orthogonal geometry. The next step is to see how a single bilinear form rebuilds the entire space.
