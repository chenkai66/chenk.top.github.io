---
title: "Metric Spaces, Normed Spaces, and Banach Spaces"
date: 2021-03-01 09:00:00
tags:
  - functional-analysis
  - metric-spaces
  - banach-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 1
series_total: 6
translationKey: "functional-analysis-1"
description: "From metric spaces to Banach spaces: completeness is what separates the useful from the pathological."
---

## Why infinite dimensions need new tools

In $\mathbb{R}^n$, every linear map is continuous, every closed bounded set is compact, and all norms are equivalent. None of these survive the jump to infinite dimensions. Functional analysis exists because infinite-dimensional spaces are fundamentally wilder than finite-dimensional ones, and we need machinery to tame them.

The starting point is deceptively simple: put a notion of distance on a set, then see what follows.

Why can't we just "do linear algebra carefully"? Because the arguments that work in $\mathbb{R}^n$ rely (often silently) on compactness of the unit ball. In infinite dimensions the unit ball is never compact in the norm topology, and this single failure cascades into dozens of pathologies: sequences of operators can converge pointwise but not uniformly, closed subspaces need not be complemented, and the dual space can be strictly larger than you expect. The abstractions below aren't mathematical tourism — they're load-bearing walls.


![Unit balls in l1, l2, and l-infinity norms](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-and-normed-spaces/fa_fig1_unit_balls.png)

## Metric spaces

A **metric space** is a pair $(X, d)$ where $d: X \times X \to [0, \infty)$ satisfies:

$$d(x, y) = 0 \iff x = y, \quad d(x, y) = d(y, x), \quad d(x, z) \le d(x, y) + d(y, z).$$

The triangle inequality is doing the heavy lifting. It forces the topology to behave: open balls are actually open, limits are unique, and sequences can't converge to two different points.

**Example 1.** The space $C[0,1]$ of continuous functions on $[0,1]$ with the sup metric:

$$d(f, g) = \sup_{t \in [0,1]} |f(t) - g(t)|.$$

Convergence here means uniform convergence. A Cauchy sequence of continuous functions converges to a continuous function — this space is complete.

**Example 2.** The same set $C[0,1]$ with the $L^1$ metric:

$$d_1(f, g) = \int_0^1 |f(t) - g(t)|\, dt.$$

Now convergence is weaker. Cauchy sequences can "converge" to discontinuous functions that aren't in $C[0,1]$. The space is incomplete under this metric.

Same set, different metric, completely different analytic properties. The metric matters.

**Example 3 (Discrete metric).** On any set $X$, define $d(x,y) = 1$ for $x \ne y$. This is a metric (check the axioms), and it makes every subset open. The space is complete (every Cauchy sequence is eventually constant), but the topology is useless for analysis — there's no notion of "approximation" here. This shows that not every metric space is analytically interesting; the structure of the metric itself determines what you can do.

## Normed spaces

A metric space has distance but no algebraic structure. A **normed space** is a vector space $X$ over $\mathbb{R}$ (or $\mathbb{C}$) equipped with a norm $\|\cdot\|: X \to [0, \infty)$ satisfying:

$$\|x\| = 0 \iff x = 0, \quad \|\alpha x\| = |\alpha|\,\|x\|, \quad \|x + y\| \le \|x\| + \|y\|.$$

Every norm induces a metric via $d(x, y) = \|x - y\|$, but not every metric comes from a norm. The norm adds homogeneity and translation invariance — the geometry looks the same everywhere.

**Non-example: metrics that don't come from norms.** The metric $d(f,g) = \int_0^1 |f(t) - g(t)| / (1 + |f(t) - g(t)|)\, dt$ on measurable functions metrizes convergence in measure, but it doesn't come from a norm — you can check that homogeneity fails. The space $L^0$ of measurable functions with this metric is a complete topological vector space but not a normed space.

**The $\ell^p$ spaces.** For $1 \le p < \infty$, define:

$$\ell^p = \left\{ (x_n)_{n=1}^\infty : \sum_{n=1}^\infty |x_n|^p < \infty \right\}, \quad \|x\|_p = \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p}.$$

For $p = \infty$:

$$\ell^\infty = \left\{ (x_n) : \sup_n |x_n| < \infty \right\}, \quad \|x\|_\infty = \sup_n |x_n|.$$

That $\|\cdot\|_p$ actually satisfies the triangle inequality is Minkowski's inequality, whose proof requires Holder's inequality as a lemma. This is not a trivial verification — it already uses the duality between $\ell^p$ and $\ell^q$.

These are all different spaces with genuinely different geometries. The unit ball in $\ell^1$ has corners. The unit ball in $\ell^2$ is round. The unit ball in $\ell^\infty$ is a cube. In finite dimensions this is cosmetic; in infinite dimensions it changes which operators are bounded, which functionals are continuous, and which theorems apply.

**Theorem (Norm equivalence fails).** In infinite-dimensional normed spaces, non-equivalent norms exist. Specifically, $\|x\|_1$ and $\|x\|_\infty$ on $\ell^1$ are not equivalent — there's no constant $C$ with $\|x\|_1 \le C\|x\|_\infty$ for all $x \in \ell^1$.

*Proof sketch.* Take $x^{(n)} = (1, 1, \ldots, 1, 0, 0, \ldots)$ with $n$ ones. Then $\|x^{(n)}\|_1 = n$ but $\|x^{(n)}\|_\infty = 1$. No finite $C$ works. $\square$

The finite-dimensional theorem that all norms are equivalent relies on compactness of the unit sphere. In infinite dimensions the unit sphere is not compact, and the theorem fails catastrophically.

## Completeness and Banach spaces

A normed space where every Cauchy sequence converges is called a **Banach space**:

$$\text{Banach space} = \text{complete normed space}.$$

Completeness is not optional — it's the dividing line between spaces where analysis works and spaces where it doesn't. Fixed point theorems, the Baire category theorem, the open mapping theorem — all require completeness.

**Theorem (Completeness of $\ell^p$).** For $1 \le p \le \infty$, the space $\ell^p$ is a Banach space.

*Proof sketch for $\ell^p$, $p < \infty$.* Let $(x^{(k)})$ be Cauchy in $\ell^p$. For each fixed $n$, the sequence $(x^{(k)}_n)_k$ is Cauchy in $\mathbb{R}$ (since $|x^{(k)}_n - x^{(m)}_n| \le \|x^{(k)} - x^{(m)}\|_p$), so it converges to some $x_n$. Define $x = (x_n)$. To show $x \in \ell^p$ and $\|x^{(k)} - x\|_p \to 0$: fix $\varepsilon > 0$, choose $K$ so $\|x^{(k)} - x^{(m)}\|_p < \varepsilon$ for $k, m \ge K$. For any finite $N$, $\sum_{n=1}^N |x^{(k)}_n - x^{(m)}_n|^p < \varepsilon^p$. Let $m \to \infty$: $\sum_{n=1}^N |x^{(k)}_n - x_n|^p \le \varepsilon^p$. Since this holds for all $N$, we get $\|x^{(k)} - x\|_p \le \varepsilon$. $\square$

**Example 4.** The space $C[0,1]$ with the sup norm is Banach (uniform limit of continuous functions is continuous). With the $L^1$ norm, it's not — its completion is $L^1[0,1]$.

**An equivalent condition: absolute convergence implies convergence.** A normed space $X$ is Banach if and only if every absolutely convergent series converges: $\sum \|x_n\| < \infty \implies \sum x_n$ converges in $X$. This is often more convenient for checking completeness than working with Cauchy sequences directly. The proof: partial sums of an absolutely convergent series form a Cauchy sequence; conversely, given a Cauchy sequence $(y_k)$, pass to a subsequence with $\|y_{k_{j+1}} - y_{k_j}\| < 2^{-j}$ and telescope.

## A non-example: why completion matters

Consider the space $c_{00}$ of sequences that are eventually zero, with the $\ell^2$ norm. This is a normed space but not Banach: the sequence $x^{(n)} = (1, 1/2, 1/3, \ldots, 1/n, 0, 0, \ldots)$ is Cauchy (check it), but its limit $(1, 1/2, 1/3, \ldots) \notin c_{00}$.

Every normed space has a completion — a minimal Banach space containing it as a dense subspace. The completion of $c_{00}$ under $\|\cdot\|_2$ is $\ell^2$. This is the infinite-dimensional analogue of completing $\mathbb{Q}$ to get $\mathbb{R}$.

**The completion construction.** Given an incomplete normed space $X$, form the set of all Cauchy sequences in $X$, modulo the equivalence relation $(x_n) \sim (y_n)$ iff $\|x_n - y_n\| \to 0$. Define addition and scalar multiplication componentwise, and set $\|[(x_n)]\| = \lim \|x_n\|$ (this limit exists since $|\|x_n\| - \|x_m\|| \le \|x_n - x_m\| \to 0$). The result $\hat{X}$ is a Banach space, and the map $x \mapsto [(x, x, x, \ldots)]$ embeds $X$ isometrically as a dense subspace. This is exactly how one constructs $\mathbb{R}$ from $\mathbb{Q}$ — the same idea at a higher level of abstraction.

**Why bother with the abstract construction?** In practice, we usually identify the completion concretely: $c_{00}$ completes to $\ell^2$, $C[0,1]$ under $L^p$ norm completes to $L^p[0,1]$, the space of step functions under $L^1$ completes to $L^1$. But the abstract construction guarantees existence and uniqueness (up to isometric isomorphism) without needing to guess the answer.

## The Baire category theorem: a first payoff

**Theorem (Baire).** In a complete metric space, the countable intersection of dense open sets is dense.

Equivalently: a complete metric space cannot be written as a countable union of nowhere-dense sets (sets whose closure has empty interior). A set that is a countable union of nowhere-dense sets is called **meager** (or first category). Baire's theorem says complete metric spaces are not meager in themselves.

*Proof sketch.* Let $U_1, U_2, \ldots$ be dense open sets. We must show $\bigcap U_n$ is dense, meaning it meets every nonempty open set $V$. Since $U_1$ is dense, $V \cap U_1$ is nonempty and open; pick a closed ball $\bar{B}(x_1, r_1) \subseteq V \cap U_1$ with $r_1 < 1$. Since $U_2$ is dense, $B(x_1, r_1) \cap U_2$ is nonempty; pick $\bar{B}(x_2, r_2) \subseteq B(x_1, r_1) \cap U_2$ with $r_2 < 1/2$. Continue inductively: $\bar{B}(x_n, r_n) \subseteq B(x_{n-1}, r_{n-1}) \cap U_n$ with $r_n < 1/n$. The centers $(x_n)$ form a Cauchy sequence (they live in nested balls of shrinking radius). By completeness, $x_n \to x$, and $x \in \bar{B}(x_n, r_n)$ for all $n$, so $x \in \bigcap U_n \cap V$. $\square$

**Why this matters for functional analysis.** The Baire category theorem is the engine behind three of the four "big theorems" in Chapter 4: uniform boundedness, open mapping, and closed graph. Each one works by showing that if the conclusion fails, the space would be a countable union of "thin" (nowhere-dense) sets, contradicting Baire. Without completeness, these theorems fail.

**Application 1: Nowhere-differentiable continuous functions.** There exist continuous functions on $[0,1]$ that are nowhere differentiable. In fact, "most" continuous functions (in the Baire category sense) are nowhere differentiable — the differentiable ones form a meager set in $C[0,1]$. More precisely, for each point $t$ and each bound $M$, the set of $f \in C[0,1]$ with $|f(t+h) - f(t)|/|h| \le M$ for some small $h$ is nowhere dense. Taking the countable union over rational $t$ and integer $M$, the "somewhere-differentiable" functions are meager. Weierstrass had to construct a specific example; Baire tells us that generic continuous functions are monsters.

**Application 2: Condensation of singularities.** If you have a sequence of continuous linear functionals $f_n$ on a Banach space with $\sup_n |f_n(x)| = \infty$ for each $x$ in some dense set, Baire implies $\sup_n |f_n(x)| = \infty$ on a residual (comeager) set. Singular behavior that happens at some points actually happens at "most" points.

## Connections: completeness in applications

Completeness is not just a technical convenience — it appears throughout applied mathematics and physics.

**Signal processing.** The space $L^2(\mathbb{R})$ of square-integrable signals is a Banach space (in fact a Hilbert space). Completeness guarantees that Fourier series and wavelet expansions actually converge to elements of the space. Without completeness, you could have a perfectly good orthonormal expansion whose limit "falls out" of the space.

**Quantum mechanics.** States live in a Hilbert space $\mathcal{H}$ (complete inner product space). The completeness is physically necessary: a sequence of states converging in norm must converge to a state. If the space were incomplete, time evolution could push a state outside the space of allowed states.

**Numerical analysis.** The Banach fixed point theorem (contraction mapping theorem) requires completeness of the underlying space. It guarantees convergence of iterative methods: Newton's method, Picard iteration for ODEs, and most numerical PDE solvers rely on this. The theorem states that if $T: X \to X$ is a contraction ($\|Tx - Ty\| \le q\|x - y\|$ with $q < 1$) on a complete metric space, then $T$ has a unique fixed point, and the iteration $x_{n+1} = Tx_n$ converges to it from any starting point.

## What's next

We have spaces with distance and linear structure. The next step is to add geometry — angles, orthogonality, projections. That means inner products, which leads to Hilbert spaces: the infinite-dimensional analogues of Euclidean space, and the natural home of quantum mechanics and Fourier analysis.

---

*This is Part 1 of the [Functional Analysis](/en/series/functional-analysis/) series (6 articles).*

*Next: [Part 2 — Hilbert Spaces](/en/functional-analysis/02-hilbert-spaces/)*
