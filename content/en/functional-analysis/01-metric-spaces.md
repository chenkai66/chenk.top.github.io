---
title: "Functional Analysis (1): Metric Spaces — Distance, Convergence, and Completeness"
date: 2021-10-01 09:00:00
tags:
  - functional-analysis
  - metric-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 1
series_total: 12
translationKey: "functional-analysis-1"
description: "From the real line to infinite-dimensional function spaces: why completeness is the dividing line."
---

## Why I Had to Stop Trusting My Finite-Dimensional Intuition

The first thing graduate analysis did to me was take away my picture. Up to that point, "distance" had always been the length of an arrow drawn from the origin to a point — Pythagoras, three coordinates, done. Then somebody asked me how far two functions are from each other and the arrow disappeared.

![Unit balls in different metrics on R^2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/01_metric_balls.png)

The trouble is that calculus on $\mathbb{R}^n$ piggybacks on a structure we never had to name. The Euclidean distance gives us convergence, convergence gives us continuity, continuity gives us derivatives and integrals, and the loop closes because $\mathbb{R}^n$ is *complete* — every Cauchy sequence has a limit inside the space. Strip away any of those pieces and the calculus collapses. So when functional analysis asks me to do calculus on a space of functions, I cannot just import the Euclidean recipe. I need a definition of distance that survives the move to infinite dimensions, and a notion of completeness that does not silently assume I am in $\mathbb{R}^n$.

A concrete example pins this down. Take $C[0,1]$, the continuous real-valued functions on $[0,1]$. Try the "obvious" generalization of Euclidean distance: $d(f,g)^2 = \sum_{n=1}^{\infty} (f_n - g_n)^2$ for some basis expansion. The series may not converge. Or pick the equally innocent $d(f,g) = \int_0^1 |f-g|\,dt$. That works as a metric, but Cauchy sequences in this metric escape from $C[0,1]$ — their limits are merely integrable, not continuous. The space leaks. Metric spaces and the notion of completeness exist precisely so that I can talk about which spaces leak and which do not.

There is a second thing finite-dimensional intuition gets wrong, and it is more subtle. In $\mathbb{R}^n$, all the natural metrics are equivalent: a sequence converges in one of them iff it converges in any other. In infinite dimensions this fails spectacularly. The *same* sequence of functions can converge under one perfectly reasonable metric and diverge wildly under another. So when somebody says a sequence converges, the next question is *in what metric*, and the answer changes which theorems apply. The whole edifice of functional analysis is built on respecting this distinction.

![The four axioms of a metric space visualized](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-spaces/fa_v2_01_1_metric_axioms.png)

## The Four Axioms, Stripped to the Bone

A **metric space** is a pair $(X, d)$ where $X$ is a set and $d: X \times X \to \mathbb{R}$ is a function such that for every $x, y, z \in X$:

![Cauchy sequences: complete vs incomplete spaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/01_cauchy_sequence.png)

1. $d(x, y) \geq 0$ (non-negativity).
2. $d(x, y) = 0 \iff x = y$ (positive definiteness).
3. $d(x, y) = d(y, x)$ (symmetry).
4. $d(x, z) \leq d(x, y) + d(y, z)$ (triangle inequality).

Three of those are bookkeeping; the triangle inequality is where all the work happens. It is the only axiom that connects three points, and it is precisely what makes "distance" propagate from local information to global information. Without it, an open ball would not even be open in any useful sense: I could have $x$ close to $y$ and $y$ close to $z$ but $x$ arbitrarily far from $z$, and continuity would be a hallucination.

A useful sanity check is to write down a function on $\mathbb{R}^2$ that satisfies axioms 1-3 but not 4 and watch the consequences. Take $d(x,y) = (x_1-y_1)^2 + (x_2-y_2)^2$ — the squared Euclidean distance. It is non-negative, definite, and symmetric. But for $x=(0,0)$, $y=(1,0)$, $z=(2,0)$, we get $d(x,z)=4$, $d(x,y)+d(y,z)=1+1=2$, and the inequality goes the wrong way. Squaring breaks the metric property and creates exactly the pathology I described above. The square root in the Euclidean distance is not aesthetic — it is doing real work.

### Worked Numerical Example

Take $X = \mathbb{R}^2$ with the Euclidean metric $d_2(x,y) = \sqrt{(x_1-y_1)^2 + (x_2-y_2)^2}$. Set $x = (0,0)$, $y = (3,0)$, $z = (3,4)$. Then $d_2(x,y) = 3$, $d_2(y,z) = 4$, and $d_2(x,z) = 5$. The triangle inequality reads $5 \leq 3 + 4 = 7$, with slack 2. Now switch to the taxicab metric $d_1(x,y) = |x_1-y_1| + |x_2-y_2|$. Same three points: $d_1(x,y) = 3$, $d_1(y,z) = 4$, $d_1(x,z) = 7$. The triangle inequality becomes $7 \leq 3 + 4 = 7$ — saturated, no slack. The reason is geometric: under $d_1$ the path $x \to y \to z$ is the *only* shortest path, while under $d_2$ the diagonal cheats by a factor of $\sqrt{2}$. The same axioms accommodate both, which is exactly the point of working with the abstract definition.

For the supremum metric $d_\infty(x,y) = \max_i |x_i - y_i|$, the same three points give $d_\infty(x,y) = 3$, $d_\infty(y,z) = 4$, $d_\infty(x,z) = 4$. Triangle inequality: $4 \leq 3 + 4 = 7$. The discrepancies between $d_1$, $d_2$, $d_\infty$ on the same three points — $7$, $5$, $4$ — quantify how much each metric "spreads out" distances. In $\mathbb{R}^n$ they are all bounded by each other up to factors of $\sqrt{n}$, so they generate the same topology, but in infinite dimensions there is no such bound and the topologies genuinely diverge.

### Why this matters

The axioms are deliberately weak. They have to be, because I want to plug in:

- $d_p(x,y) = \big(\sum_{i=1}^n |x_i - y_i|^p\big)^{1/p}$ on $\mathbb{R}^n$ for any $1 \leq p \leq \infty$;
- the discrete metric $d(x,y) = 1 - \delta_{xy}$ on any set, where every two distinct points are exactly distance $1$ apart;
- the supremum metric $d_\infty(f,g) = \sup_{t} |f(t) - g(t)|$ on bounded functions;
- the integral metric $d_1(f,g) = \int |f - g|$ on integrable functions;
- the Hausdorff metric $d_H(A, B) = \max\{\sup_{a \in A} d(a, B), \sup_{b \in B} d(b, A)\}$ on closed subsets of a metric space, which makes the space of compact sets into a metric space;
- the edit distance on strings, with no algebraic structure at all.

A theorem proved from the four axioms applies to all of these simultaneously. That is the leverage I am buying. The price is that I have to give up coordinate-based proofs and learn to argue using only $d$.

## Convergence and Open Sets

A sequence $(x_n) \subset X$ **converges** to $x \in X$, written $x_n \to x$, if $d(x_n, x) \to 0$ in $\mathbb{R}$. The definition is so familiar in $\mathbb{R}^n$ that it can mask an important point: convergence is metric-dependent. The same sequence of functions can converge in one metric and diverge in another, and this is not a pathology — it is the entire reason I bother distinguishing metrics.

![Completion of a metric space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/01_completeness.png)

The **open ball** of radius $r$ around $x$ is $B(x, r) = \{ y \in X : d(x,y) < r \}$. A set $U \subseteq X$ is **open** if every point of $U$ is the center of some open ball contained in $U$. This generates a topology, and convergence in the metric agrees with convergence in this topology, so I get topology and metric for the price of one definition. A set is **closed** if its complement is open, equivalently if it contains the limit of every convergent sequence of its elements — the metric makes the topological notion of closure (smallest closed set containing $A$) coincide with the sequential notion (limits of sequences in $A$).

![Open balls under Euclidean, taxicab, and supremum metrics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-spaces/fa_v2_01_2_open_balls.png)

What changes is the *shape* of the balls. In $\mathbb{R}^2$, the unit ball under $d_2$ is a disk, under $d_1$ a square rotated $45°$, under $d_\infty$ an axis-aligned square. The same set of open sets emerges (the metrics are equivalent in finite dimensions, as we will prove next article), but the geometric "feel" is different, and in infinite dimensions the equivalence breaks. The shape of the unit ball is what carries the convexity and reflexivity properties of the space, and a major theme of Article 2 is that the geometry of unit balls in different norms is what distinguishes Hilbert spaces (round balls) from generic Banach spaces (potentially very flat balls).

A continuous map between metric spaces $f: (X, d_X) \to (Y, d_Y)$ is one for which preimages of open sets are open, equivalently: for every $\varepsilon > 0$ and every $x \in X$ there exists $\delta > 0$ such that $d_X(x, x') < \delta$ implies $d_Y(f(x), f(x')) < \varepsilon$. Continuity in the metric sense is automatically continuity in the topological sense, but the metric formulation buys me **uniform continuity** ($\delta$ depending only on $\varepsilon$, not on $x$) and **Lipschitz continuity** ($d_Y(f(x), f(x')) \leq L \cdot d_X(x, x')$). These are quantitative notions absent from pure topology, and they are what let me prove rate-of-convergence theorems.

### Worked Numerical Example
Take $f_n(t) = t^n$ on $[0,1]$. Under the supremum metric $d_\infty(f,g) = \sup_t |f(t)-g(t)|$, the distance to the zero function is $d_\infty(f_n, 0) = 1$ for every $n$. The sequence does not converge to $0$. Switch to the integral metric $d_1(f,g) = \int_0^1 |f-g|\,dt$. Compute $d_1(f_n, 0) = \int_0^1 t^n\,dt = 1/(n+1)$. For $n=9$, $d_1 = 0.1$. For $n=99$, $d_1 = 0.01$. For $n=999$, $d_1 = 0.001$. The numbers shrink to zero linearly in $1/n$. The sequence converges to $0$ in $(C[0,1], d_1)$ but stays exactly distance $1$ away in $(C[0,1], d_\infty)$. The open ball $B_{d_1}(0, 0.05)$ contains $f_{99}$, while $B_{d_\infty}(0, 0.5)$ contains none of them. Convergence is not a property of the sequence alone; it is a property of the pair (sequence, metric).

## Cauchy Sequences and the Hidden Hypothesis

A sequence $(x_n)$ is **Cauchy** if for every $\varepsilon > 0$ there exists $N$ such that $d(x_m, x_n) < \varepsilon$ for all $m, n \geq N$. The eyes-glaze-over phrasing hides the actual content: the terms get arbitrarily close *to each other*, with no reference to a candidate limit.

![Banach contraction mapping theorem](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/01_banach_fixed_point.png)

Every convergent sequence is Cauchy (triangle inequality, again: $d(x_m, x_n) \leq d(x_m, x) + d(x, x_n) \to 0$). The converse — every Cauchy sequence is convergent — is the defining property of a **complete** metric space. It is *not* a free gift. It is an extra hypothesis that has to be earned space by space.

![Cauchy sequences: convergent vs non-convergent in incomplete spaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-spaces/fa_v2_01_3_cauchy_seq.png)

### A Cauchy Sequence That Does Not Converge

Take the rationals $\mathbb{Q}$ with $d(x,y) = |x-y|$. Define
$$a_1 = 1,\quad a_{n+1} = \tfrac{1}{2}\big(a_n + 2/a_n\big).$$
This is the Newton iteration for $\sqrt{2}$. Each $a_n$ is rational. Numerically: $a_1=1$, $a_2=1.5$, $a_3 \approx 1.4167$, $a_4 \approx 1.4142157$, $a_5 \approx 1.4142136$. The differences $|a_{n+1} - a_n|$ shrink quadratically, so the sequence is Cauchy in the obvious sense. But the limit $\sqrt{2}$ is irrational. The sequence is escaping from $\mathbb{Q}$. To $\mathbb{Q}$, this Cauchy sequence has *no limit*, full stop.

That is exactly the kind of leak completeness rules out. The reals $\mathbb{R}$ are constructed precisely to plug holes like this — every Cauchy sequence of reals converges to a real. This is not a theorem about $\mathbb{R}$; it is the *definition* by which we build $\mathbb{R}$ from $\mathbb{Q}$, either via Dedekind cuts or via Cauchy-sequence equivalence classes (which is the construction we will generalize below).

A second example, closer to functional analysis. Consider $C[0,1]$ with the metric $d_1(f,g) = \int_0^1 |f-g|\,dt$. Define
$$f_n(t) = \begin{cases} 0, & 0 \leq t \leq 1/2,\\ n(t - 1/2), & 1/2 < t < 1/2 + 1/n,\\ 1, & 1/2 + 1/n \leq t \leq 1.\end{cases}$$
Each $f_n$ is continuous. A short calculation gives $d_1(f_n, f_m) \leq |1/n - 1/m|/2$, so $(f_n)$ is Cauchy. The pointwise limit is the indicator $\mathbb{1}_{[1/2, 1]}$, which is discontinuous and not in $C[0,1]$. Like $\mathbb{Q}$, the space $C[0,1]$ leaks Cauchy sequences when measured in the integral metric.

### Why this matters

Completeness is the algebraic license to do limit-based arguments. Want to define the integral as a limit of Riemann sums? You need the limit to exist somewhere. Want to solve a differential equation by Picard iteration? You need the iterates to converge to a solution *inside the function space you started in*. Want to define a derivative via difference quotients of operators? You need the resulting operator to live in the same operator space. Without completeness, you have a ladder with the top rung sawed off: you can climb forever but never arrive.

The completeness of a space is *metric-dependent*, not just space-dependent. The space $C[0,1]$ is complete under the supremum metric $d_\infty(f,g) = \sup_t |f-g|$ (the sequence above does not converge uniformly, so it is not Cauchy in $d_\infty$), but incomplete under $d_1$. The question is never just "is this complete?" — it is "is this complete under that metric?"

### Worked Numerical Example
Work in $X = (0, 1)$ with the standard metric $d(x,y) = |x-y|$. Define $x_n = 1/n$. Compute pairwise distances: $d(x_{10}, x_{20}) = |0.1 - 0.05| = 0.05$. $d(x_{100}, x_{200}) = |0.01 - 0.005| = 0.005$. $d(x_{1000}, x_{2000}) = 0.0005$. For any $\varepsilon = 10^{-k}$, choosing $N = 10^k$ guarantees $d(x_m, x_n) < \varepsilon$ for all $m,n \geq N$. The sequence is Cauchy by direct arithmetic. The candidate limit is $0$, but $0 \notin (0,1)$. The distances between terms collapse to zero while the sequence drifts toward a hole in the space. The Cauchy condition detects internal coherence; it cannot detect whether the destination exists inside $X$.

## The Completion of a Metric Space

The good news: every metric space $(X, d)$ has a **completion** $(\widehat X, \widehat d)$ — a complete metric space containing an isometric, dense copy of $X$. The construction is canonical. Take all Cauchy sequences in $X$, declare two of them equivalent if their pointwise distance tends to zero, and define the distance between equivalence classes as the limit of pointwise distances. This is essentially how Cantor builds $\mathbb{R}$ from $\mathbb{Q}$.

![Completing the rationals to the reals via Cauchy sequences](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-spaces/fa_v2_01_4_completion.png)

In more detail. Let $\mathcal{C}$ be the set of all Cauchy sequences in $X$. Define $(x_n) \sim (y_n)$ iff $d(x_n, y_n) \to 0$. This is an equivalence relation (reflexivity and symmetry are obvious; transitivity uses the triangle inequality). Define $\widehat X = \mathcal{C}/{\sim}$ and
$$\widehat d\big([(x_n)], [(y_n)]\big) = \lim_{n \to \infty} d(x_n, y_n).$$
The limit exists because $(d(x_n, y_n))$ is a Cauchy sequence in $\mathbb{R}$ (by the triangle inequality, $|d(x_n, y_n) - d(x_m, y_m)| \leq d(x_n, x_m) + d(y_n, y_m)$), and $\mathbb{R}$ is complete. The map $X \to \widehat X$ sending $x$ to the class of the constant sequence $(x, x, x, \ldots)$ is an isometric embedding, and its image is dense in $\widehat X$.

Completeness of $\widehat X$ requires a diagonal argument. Given a Cauchy sequence $(\widehat{\xi}^{(k)})$ of equivalence classes, pick a representative $(\xi^{(k)}_n)$ for each. Choose $n_k$ large enough that $d(\xi^{(k)}_n, \xi^{(k)}_{n_k}) < 1/k$ for all $n \geq n_k$, and form the diagonal sequence $y_k = \xi^{(k)}_{n_k}$. A bit of bookkeeping shows $(y_k)$ is Cauchy in $X$, and its equivalence class in $\widehat X$ is the limit of $(\widehat{\xi}^{(k)})$.

The completion has a universal property: any uniformly continuous map from $X$ to a complete metric space extends uniquely to $\widehat X$. This is the categorical reason completion shows up everywhere, from $L^p$ spaces (the completion of continuous functions in the $L^p$ norm) to $p$-adic numbers (the completion of $\mathbb{Q}$ under the $p$-adic absolute value $|p^k m/n|_p = p^{-k}$ for $\gcd(m, p) = \gcd(n, p) = 1$).

### Worked Example: completing $C[0,1]$ in the $L^1$ norm

Take the sequence $(f_n)$ from above. It is Cauchy in $(C[0,1], d_1)$. Its equivalence class in the completion is what we call the indicator function $\mathbb{1}_{[1/2,1]}$, identified with any other Cauchy sequence converging to it pointwise almost everywhere. The completion of $C[0,1]$ under $d_1$ is the space $L^1[0,1]$ — Lebesgue-integrable functions modulo equality almost everywhere. The discontinuous indicator is a perfectly fine element there.

A subtler point: the elements of $L^1[0,1]$ are *equivalence classes*, not functions. You cannot ask for the value of an $L^1$ function at a point — that question is not even well-posed, because two equivalent functions can disagree on any prescribed measure-zero set. This is the price of completeness in the $L^1$ norm: I gain a complete space at the cost of pointwise evaluation. In Article 3 we will see Hilbert spaces ($L^2$ in particular) inherit the same trade-off.

The lesson: every reasonable function space you have ever met — the $L^p$ spaces, the Sobolev spaces, the Hardy spaces — is the completion of a more concrete space (continuous functions, or smooth compactly supported functions) in some specific norm. Completeness is not optional decoration; it is the price of admission for limits.

### Worked Numerical Example
Take $\mathbb{Q}$ with $d(x,y)=|x-y|$. Consider two Cauchy sequences approximating $\sqrt{2}$: $a_n$ from Newton's method ($1, 1.5, 1.41666..., 1.414215...$) and $b_n$ from decimal truncation ($1, 1.4, 1.41, 1.414, 1.4142$). Compute termwise distances: $|a_1-b_1|=0$, $|a_2-b_2|=0.1$, $|a_3-b_3| \approx 0.00666$, $|a_4-b_4| \approx 0.000215$, $|a_5-b_5| \approx 0.000013$. The distance sequence tends to $0$. By the completion construction, $[(a_n)] = [(b_n)]$ in $\widehat{\mathbb{Q}}$. They define the same point. Now take $c_n = 1.5, 1.42, 1.415, 1.4143, ...$ (rounding up). $|a_n - c_n|$ also tends to $0$. All three sequences belong to the same equivalence class. The real number $\sqrt{2}$ is not a single sequence; it is the entire class of rational sequences whose mutual distances vanish. The completion turns this vanishing distance into an actual point.

## The Baire Category Theorem

Once a metric space is complete, a remarkable rigidity result kicks in. The **Baire Category Theorem** says: in a complete metric space, the countable intersection of dense open sets is still dense. Equivalently, a complete metric space cannot be written as a countable union of *nowhere-dense* sets (sets whose closure has empty interior).

![Baire category theorem: a complete metric space is not a countable union of nowhere-dense sets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-spaces/fa_v2_01_5_baire.png)

The proof is short and feels like a magic trick. Suppose $\{U_n\}$ is a countable family of dense open sets, and let $V$ be any non-empty open set. I want to show $V \cap \bigcap_n U_n \neq \emptyset$. Since $U_1$ is dense and open, $V \cap U_1$ is non-empty and open, so it contains a closed ball $\overline{B(x_1, r_1)}$ with $r_1 < 1$. Since $U_2$ is dense and open, $B(x_1, r_1) \cap U_2$ contains a closed ball $\overline{B(x_2, r_2)}$ with $r_2 < 1/2$. Iterate, halving the radius each time. The centers $(x_n)$ form a Cauchy sequence — for $m \geq n$, $x_m \in \overline{B(x_n, r_n)}$, so $d(x_n, x_m) \leq r_n < 2^{-n+1}$. Completeness gives a limit point $x^*$. By construction $x^* \in \overline{B(x_n, r_n)}$ for every $n$, hence $x^* \in V \cap \bigcap_n U_n$. Done.

The proof uses completeness in exactly one place: to conclude that the Cauchy sequence of centers has a limit. Without completeness the construction goes through but produces nothing. This is what makes Baire a *consequence* of completeness rather than an axiom.

### Consequences

**Banach-Steinhaus / Uniform Boundedness Principle.** If $\{T_\alpha\}$ is a family of bounded linear operators from a Banach space $X$ to a normed space $Y$, and $\sup_\alpha \|T_\alpha x\| < \infty$ for every $x$, then $\sup_\alpha \|T_\alpha\| < \infty$. Pointwise boundedness implies uniform boundedness. The proof partitions $X$ into closed sets $F_n = \{ x : \|T_\alpha x\| \leq n \text{ for all } \alpha \}$; their union is $X$; Baire forces one of them to have non-empty interior; that interior gives the uniform bound. (Article 6 spells this out.)

**Existence of nowhere-differentiable continuous functions.** The set of continuous functions on $[0,1]$ that are differentiable at *some* point can be written as a countable union of nowhere-dense sets in $(C[0,1], d_\infty)$. Hence, by Baire, this union is *not* all of $C[0,1]$, and the complement is dense. Most continuous functions, in a precise topological sense, fail to be differentiable anywhere.

**Banach's open-mapping theorem and closed-graph theorem.** Both follow from Baire applied to the image or graph of the operator under consideration. Article 6 walks through them carefully.

### Why this matters

Baire is the engine that drives every "automatic" theorem in functional analysis: the uniform boundedness principle, the open mapping theorem, the closed graph theorem (all in Article 6). Each takes a "pointwise" hypothesis and concludes a "uniform" conclusion. The trick is always the same — partition the space into closed sets defined by the failure of the conclusion, observe their union is the whole space, and conclude one of them must have non-empty interior.

A second use is the production of generic objects. The set of continuous nowhere-differentiable functions on $[0,1]$ is the *complement* of a countable union of nowhere-dense sets in $C[0,1]$, hence a dense $G_\delta$. So in a precise topological sense, a generic continuous function fails to be differentiable anywhere. The functions you actually compute with are atypical. This is one of those theorems that should reset your defaults: smooth functions are the exception, not the rule.

## The Banach Fixed-Point Theorem

The clean computational consequence of completeness is the **contraction mapping theorem**. Let $(X, d)$ be a complete metric space and $T: X \to X$ a contraction: there exists $0 \leq \lambda < 1$ with
$$d(T x, T y) \leq \lambda \, d(x, y) \quad \text{for all } x, y \in X.$$
Then $T$ has a unique fixed point $x^*$, and starting from any $x_0 \in X$, the iterates $x_{n+1} = T x_n$ satisfy $d(x_n, x^*) \leq \lambda^n d(x_0, x^*) \leq \frac{\lambda^n}{1-\lambda} d(x_1, x_0)$.

![Banach fixed-point iteration converging to a unique fixed point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-spaces/fa_v2_01_6_fixed_point.png)

The proof is the cleanest in the subject. First, $d(x_{n+1}, x_n) \leq \lambda \, d(x_n, x_{n-1}) \leq \cdots \leq \lambda^n d(x_1, x_0)$, so by the geometric series the iterates form a Cauchy sequence: for $m > n$,
$$d(x_n, x_m) \leq \sum_{k=n}^{m-1} d(x_k, x_{k+1}) \leq \frac{\lambda^n}{1-\lambda} d(x_1, x_0).$$
Completeness gives a limit $x^*$. Continuity of $T$ (built into the contraction inequality) gives $T x^* = \lim T x_n = \lim x_{n+1} = x^*$. Uniqueness drops out of the contraction inequality applied to two fixed points $x^*, y^*$: $d(x^*, y^*) = d(Tx^*, Ty^*) \leq \lambda d(x^*, y^*)$, forcing $d(x^*, y^*) = 0$ since $\lambda < 1$. Every step uses completeness exactly once.

### Numerical example

Take $X = [1, 2] \subset \mathbb{R}$ (complete because closed in $\mathbb{R}$), and $T x = \tfrac{1}{2}(x + 2/x)$ — Newton's iteration for $\sqrt{2}$, restricted to $[1,2]$. A short calculation shows $|T'(x)| = |1/2 - 1/x^2|$, which equals $1/2$ at $x=1$, $1/4$ at $x=2$, and $0$ at $x=\sqrt{2}$. So on $[1,2]$, $|T'(x)| \leq 1/2$, hence $T$ is a contraction with $\lambda = 1/2$. Starting from $x_0 = 1$:
$x_1 = 1.5$, $x_2 \approx 1.41667$, $x_3 \approx 1.41422$, $x_4 \approx 1.41421$. The unique fixed point is $\sqrt{2} \approx 1.41421356$. The convergence is in fact quadratic for Newton — error squares each step — much faster than the linear bound the theorem promises, but the theorem already guarantees there is something to converge to. The point is that completeness is what *guarantees* a limit exists; the rate of convergence is an additional structure I read off from the specific operator.

### Picard-Lindelöf as a fixed-point theorem

Consider the ODE $y'(t) = F(t, y(t))$ with $y(t_0) = y_0$, where $F: \mathbb{R}^2 \to \mathbb{R}$ is Lipschitz in $y$ with constant $L$. By integration this is equivalent to $y(t) = y_0 + \int_{t_0}^t F(s, y(s))\,ds$. Define the operator $T$ on $C[t_0, t_0 + h]$ by
$$(Ty)(t) = y_0 + \int_{t_0}^t F(s, y(s))\,ds.$$
For $h$ small enough, $T$ is a contraction in the supremum metric on $C[t_0, t_0+h]$ with $\lambda = Lh$. Banach's theorem produces the unique fixed point — which is the unique solution of the ODE on $[t_0, t_0+h]$. Existence and uniqueness, both falling out of completeness of $C[t_0, t_0+h]$ under the sup metric. This is the cleanest application of fixed-point theory to a non-trivial classical theorem, and a preview of how I will solve PDEs in later articles.

### Why this matters

The fixed-point theorem is how I solve almost every existence problem in this series. The Picard-Lindelöf theorem on existence of solutions to ODEs is a fixed-point argument in $C[a, b]$. The implicit function theorem in Banach spaces is a fixed-point argument. Solutions of integral equations of the form $f = g + Kf$ exist as fixed points of $T f = g + K f$ whenever $K$ is contractive. The Hartman-Grobman theorem on linearization of dynamical systems uses a fixed-point argument in a function space. Every time I solve $T x = x$ by iteration and bound the rate of convergence, I am cashing in the same theorem.

### Worked Numerical Example
Let $X = [0, 1]$ and $T(x) = \cos(x)$. The derivative is $T'(x) = -\sin(x)$. On $[0,1]$, $|T'(x)| \leq \sin(1) \approx 0.84147$. So $T$ is a contraction with $\lambda = 0.84147$. Start at $x_0 = 0.5$. Iterate: $x_1 = \cos(0.5) \approx 0.87758$, $x_2 = \cos(0.87758) \approx 0.63901$, $x_3 = \cos(0.63901) \approx 0.80269$, $x_4 \approx 0.69478$. The theoretical error bound after $n=4$ steps is $d(x_4, x^*) \leq \frac{\lambda^4}{1-\lambda} d(x_1, x_0) \approx \frac{0.501}{0.15853} \times 0.37758 \approx 1.19$. The actual fixed point is $x^* \approx 0.739085$. The true error is $|0.69478 - 0.739085| \approx 0.0443$. The bound is loose but valid. The contraction constant $\lambda < 1$ forces the error envelope to shrink geometrically, and completeness guarantees the envelope closes on a point inside $[0,1]$.

## Compactness in Metric Spaces

The last piece I need before moving to normed spaces is **compactness**. A subset $K \subseteq X$ is compact if every open cover has a finite subcover. In a metric space this turns out to be equivalent to several other conditions, and that equivalence is what makes compactness so powerful.

![Equivalent characterizations of compactness in metric spaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-spaces/fa_v2_01_7_compactness.png)

**Theorem.** For a subset $K$ of a metric space $X$, the following are equivalent:

1. Every open cover of $K$ has a finite subcover (covering compactness).
2. Every sequence in $K$ has a convergent subsequence with limit in $K$ (sequential compactness).
3. $K$ is complete and **totally bounded**: for every $\varepsilon > 0$, $K$ can be covered by finitely many balls of radius $\varepsilon$.

The implication $(2) \Leftrightarrow (3)$ says that compactness in a metric space is "completeness plus a finiteness condition." In $\mathbb{R}^n$, total boundedness is the same as boundedness (Heine-Borel), so compact sets are exactly the closed bounded sets. In infinite dimensions, *boundedness is no longer enough*: the closed unit ball of $C[0,1]$ is not compact. Article 5 will revisit this and find the right replacement (weak-* compactness via Banach-Alaoglu).

Why does the equivalence $(1) \Leftrightarrow (2)$ require the metric? In a general topological space these are different — sequential compactness is weaker. But in a metric space, every point has a countable neighborhood base (the balls of radius $1/n$), and that countability lets the diagonal argument convert sequence-level data into cover-level data. The metric is what makes the equivalence work.

### Numerical example

In $\mathbb{R}$, the closed interval $[0, 1]$ is compact. Cover it by the open intervals $(k/n - 1/n, k/n + 1/n)$ for $k = 0, 1, \ldots, n$ — these $n+1$ sets cover $[0,1]$. So total boundedness is concrete. Now take the closed unit ball in $\ell^2$, the space of square-summable sequences. The standard basis vectors $e_n = (0, \ldots, 0, 1, 0, \ldots)$ all sit in the unit ball, and $\|e_n - e_m\|_2 = \sqrt{2}$ for $n \neq m$. No subsequence is Cauchy, so no subsequence converges. The ball is bounded but not compact. The dimension matters.

A more striking example: in the space of continuous functions $C[0,1]$ with sup metric, the sequence $f_n(t) = \sin(n\pi t)$ has $\|f_n\|_\infty = 1$ for every $n$, so it sits in the unit ball. But $\|f_n - f_m\|_\infty = 2$ for $n \neq m$ in any range where the sines are out of phase. The ball is not even sequentially compact. A theorem of F. Riesz makes this precise: the closed unit ball of a normed space is compact iff the space is finite-dimensional. The presence of compactness in infinite dimensions requires giving up something — either passing to a weaker topology (weak compactness, Article 5) or restricting to a smaller class of operators (compact operators, Article 7).

### Arzelà-Ascoli

Specializing to $C[K]$ for $K$ a compact metric space, the Arzelà-Ascoli theorem characterizes compact subsets:

**Theorem (Arzelà-Ascoli).** A subset $\mathcal{F} \subseteq C[K]$ has compact closure iff $\mathcal{F}$ is pointwise bounded and equicontinuous: for every $\varepsilon > 0$ there exists $\delta > 0$ such that $d(s,t) < \delta$ implies $|f(s) - f(t)| < \varepsilon$ for *every* $f \in \mathcal{F}$ (the same $\delta$ works for all $f$).

Total boundedness is the abstract notion; equicontinuity plus pointwise boundedness is its concrete form for $C[K]$. The condition rules out wild oscillation (like $\sin(n\pi t)$) by requiring uniform control across the family. Arzelà-Ascoli is the workhorse compactness theorem in classical analysis, and many of the existence theorems we will see — for solutions of ODEs, for minimizers of functionals, for limits of approximation schemes — go through it.

### Why this matters

Compactness is the topological substitute for "finite." Continuous functions on compact sets attain their extrema (Weierstrass), are uniformly continuous, and are bounded. Compactness lets me extract convergent subsequences and so converts existence questions ("does some maximizer exist?") into routine arguments. The whole calculus of variations, the whole theory of weak solutions of PDEs, and most of operator spectral theory cash in compactness at some critical step.

### Worked Numerical Example
Test total boundedness with $\varepsilon = 0.2$. In $[0,1] \subset \mathbb{R}$, place balls of radius $0.2$ at centers $0.1, 0.3, 0.5, 0.7, 0.9$. Five balls cover the interval exactly. The covering number $N(0.2) = 5$ is finite. Now take the closed unit ball $B$ in $\ell^2$. Consider the standard basis vectors $e_1, e_2, e_3, e_4, e_5$. Pairwise distances are $\|e_i - e_j\|_2 = \sqrt{1^2 + (-1)^2} = \sqrt{2} \approx 1.414$. A ball of radius $0.2$ has diameter $0.4$. It can contain at most one $e_k$. To cover just these five points requires five disjoint balls. Since $\ell^2$ contains infinitely many such vectors, $N(0.2)$ is infinite. The unit ball is bounded (radius $1$) but fails total boundedness at $\varepsilon = 0.2$. Compactness collapses in infinite dimensions because boundedness no longer controls the covering number.

## Separability and Density

One last topological notion before I move on. A metric space is **separable** if it has a countable dense subset. The rationals are dense in the reals, so $\mathbb{R}$ is separable. The polynomials with rational coefficients are dense in $C[0,1]$ (Weierstrass approximation), so $C[0,1]$ is separable. The classical sequence spaces $\ell^p$ for $1 \leq p < \infty$ are separable, with the finitely supported rational sequences as a countable dense subset. The space $\ell^\infty$ of bounded sequences is *not* separable — for any countable family of sequences, a diagonal construction produces a bounded sequence at distance $\geq 1$ from each, so no countable family is dense.

Separability matters because it decides whether I can do "constructive" approximation. In a separable space I can hope to enumerate a basis and approximate every element by finite linear combinations. In a non-separable space, no countable enumeration will reach everything. Article 2 will discuss Schauder bases as the right notion of basis for separable Banach spaces; Article 3 will build orthonormal bases for separable Hilbert spaces. Non-separable spaces exist and matter (e.g., $L^\infty$, the space of bounded measurable functions), but the bulk of the theory we will develop assumes separability.

A clean theorem combining the ideas of this article: a metric space is separable iff it has a countable basis for its topology iff every open cover has a countable subcover (Lindelöf property). For complete metric spaces, separability is equivalent to being homeomorphic to a subset of the Hilbert cube $[0,1]^{\mathbb{N}}$. So separability is roughly "no bigger than the continuum, in a structured way," and it is exactly the regularity condition we want for any space that pretends to extend $\mathbb{R}^n$.

## A Working Catalog of Metrics You Will Actually Meet

Theory is one thing; the catalog of metrics that show up in everyday analysis is another. A short list of the metrics I find myself reaching for, with the property that distinguishes each.

- **Discrete metric.** $d(x, y) = 0$ if $x = y$ and $1$ otherwise. Every set is open, every set is closed, the only convergent sequences are eventually constant. Useful as a sanity check (does my proof secretly assume connectivity?) and almost nothing else.
- **Euclidean / $\ell^2$ metric.** $d(x, y) = \big(\sum (x_i - y_i)^2\big)^{1/2}$. The default; the only $\ell^p$ metric whose ball is round; the only one for which "rotation" makes sense as an isometry.
- **Taxicab / $\ell^1$ metric.** $d(x, y) = \sum |x_i - y_i|$. The natural metric for sparsity-promoting optimization; produces polytopal balls with corners at the unit vectors. Geometrically penalizes "diagonal" movement compared to $\ell^2$.
- **Sup / $\ell^\infty$ / Chebyshev metric.** $d(x, y) = \max_i |x_i - y_i|$. Cubical balls; the natural metric on $C[a,b]$ when "uniform approximation" is the goal; the one used for max-norm error bounds in numerics.
- **Hamming metric on $\{0,1\}^n$.** $d(x, y) = \#\{i : x_i \neq y_i\}$. Discrete, finite-valued, used in coding theory and error-correcting codes. The unit ball is a Hamming ball, the discrete analogue of an $\ell^1$ ball.
- **Edit distance on strings.** Smallest number of single-character insertions, deletions, and substitutions converting one string to another. Used in spell-check and bioinformatics. Triangle inequality holds because compositions of edits are still edits.
- **Hausdorff distance on closed bounded sets.** $d_H(A, B) = \max(\sup_{a \in A} d(a, B), \sup_{b \in B} d(b, A))$. Two sets are close if every point of one is close to some point of the other. The right metric for set-valued limits, image processing, geometric measure theory.
- **Wasserstein / earth-mover's distance on probability measures.** $W_p(\mu, \nu) = \inf_{\pi} \big(\int d(x,y)^p\,d\pi(x,y)\big)^{1/p}$ over couplings $\pi$ with marginals $\mu, \nu$. Captures how much "mass" must be moved to turn $\mu$ into $\nu$. Central in optimal transport, with deep connections to PDE and machine learning.

These are not exotic. They are, individually, the natural metric of some specific application. Functional analysis is the language that lets you switch between them while keeping arguments portable. The inequalities that compare them — for example, on a finite-dimensional space, the standard $\ell^p$ norms differ by factors that depend on dimension but stay finite for any fixed $n$ — are the unsung utility theorems of the subject.

### Why pointwise convergence is *not* a metric topology

A diagnostic example. On $C[0, 1]$, "pointwise convergence" — $f_n \to f$ if $f_n(t) \to f(t)$ for every $t$ — is a perfectly natural topological notion, but it is *not* induced by any metric. The proof is short: any metric topology is first-countable (the open balls of rational radii around a point form a countable neighborhood basis), but the pointwise convergence topology on $C[0,1]$ fails first-countability — there is no countable neighborhood basis at $0$. So pointwise convergence is genuinely outside the metric framework, and any theorem that wants to use it has to step into the (more general, more cumbersome) world of topological vector spaces.

This is one of the structural reasons functional analysis is biased toward metric and normed settings: most "natural" topologies turn out to be metric, and the few that don't (pointwise convergence, weak topology in the non-separable case) require the heavier machinery of Article 5.

## A Quick Test of Completeness Intuition

Three quick true/false items to calibrate. (Answers below.)

1. The space $C^1[0, 1]$ of continuously differentiable functions, with the sup metric $d_\infty$, is complete.
2. The space $\mathbb{Q}^n$ with Euclidean distance is complete.
3. The space of polynomials on $[0, 1]$ with $L^2$ inner product is complete.

Answers: (1) **No** — a uniform limit of $C^1$ functions need not be $C^1$ (only continuous), so $C^1$ is not closed in $C[0,1]$ under the sup metric. The right metric for $C^1$ is $d(f, g) = \|f - g\|_\infty + \|f' - g'\|_\infty$, in which it is complete. (2) **No** — $\mathbb{Q}^n$ is dense in $\mathbb{R}^n$ but not closed; Cauchy sequences of rationals converge to irrationals. (3) **No** — polynomials are dense in $L^2[0,1]$ (Weierstrass) but not closed; the completion is the whole of $L^2[0,1]$.

These three together encode the central rule: completeness depends jointly on the *space* and the *metric*. Choosing the wrong metric on a perfectly nice space gives an incomplete object whose completion may be mysterious or unwelcome.

## Connectedness, Path-Connectedness, and Why They Matter Less

A metric space is **connected** if it is not the disjoint union of two non-empty open sets. It is **path-connected** if any two points are joined by a continuous path. Path-connected implies connected; the converse fails (the topologist's sine curve is the standard counterexample).

These properties matter much less in functional analysis than they do in topology proper, because most of the spaces we care about — Banach spaces, function spaces, Sobolev spaces — are linear and hence path-connected via straight-line paths. The exceptions are quotients, projective spaces, and other algebraic constructions where the linearity is broken; there connectivity becomes a non-trivial input.

Where connectivity *does* matter is in the spectral theory of operators. A bounded operator with disconnected spectrum can be decomposed by the Riesz functional calculus into "spectral parts," and the decomposition is what powers the Jordan-canonical-form-style classification of compact operators (Article 7). So connectivity in the *spectrum* (a subset of $\mathbb{C}$) is the place where this metric-space concept earns its keep, even when connectivity in the underlying Banach space is automatic.

## Looking Ahead

Metric spaces give me distance and convergence, completeness lets me actually compute limits, Baire and Banach fixed-point provide leverage, and compactness recovers finite-dimensional intuition under controlled hypotheses. None of this required any algebraic structure on $X$. In the next article I add a vector space structure compatible with the metric — a *norm* — and the theory immediately tightens: I can talk about linear maps, closed subspaces, finite-dimensional approximation, and the very particular completeness that makes a normed space a Banach space. The metric framework is general enough to host edit distance and the discrete metric; the normed framework will be specific enough to support actual analysis.

One last meta-point. Notice how each tool in this article cashed in completeness exactly once, in subtly different ways. Cauchy sequences in the construction of completion: completeness is the conclusion. Baire: completeness produces a non-empty intersection. Contraction mapping: completeness produces the limit of iterates. Compactness in metric spaces (via $(3)$): completeness is half the definition. The definition of complete metric space is so concentrated that essentially every theorem we will prove that goes beyond first-year topology can be traced back to it. When you get stuck on a proof in functional analysis, asking "where am I using completeness?" is the question that most often unblocks you.

---

## Counterexample: Why the Definition Cannot Be Weakened
The Banach Fixed-Point Theorem requires a uniform contraction constant $\lambda < 1$. A natural weakening is to demand only strict contractivity: $d(Tx, Ty) < d(x, y)$ for all $x \neq y$, without a global $\lambda$. This breaks the theorem completely.

Take $X = [0, \infty)$ with the Euclidean metric (complete). Define $T(x) = \sqrt{x^2 + 1}$. The derivative is $T'(x) = x / \sqrt{x^2 + 1}$. For every $x \geq 0$, $0 \leq T'(x) < 1$. By the mean value theorem, $|T(x) - T(y)| < |x - y|$ for all distinct $x, y$. The map strictly shrinks every distance. Yet $T$ has no fixed point: solving $\sqrt{x^2+1} = x$ gives $x^2+1 = x^2$, which is $1=0$. 

Run the iteration from $x_0 = 0$: $x_1 = 1$, $x_2 = \sqrt{2} \approx 1.414$, $x_3 = \sqrt{3} \approx 1.732$, $x_4 = 2$, $x_n = \sqrt{n}$. The distances between successive terms shrink: $d(x_n, x_{n+1}) = \sqrt{n+1} - \sqrt{n} = 1/(\sqrt{n+1} + \sqrt{n})$, which tends to $0$. But the sequence drifts to infinity. The local contraction rate $T'(x)$ approaches $1$ as $x \to \infty$, so there is no uniform $\lambda$ bounding the shrinkage globally. Without that uniform bound, the geometric series argument in the proof diverges, the Cauchy estimate fails, and completeness has nothing to catch. The hypothesis $\lambda < 1$ is not a technical convenience; it is the brake that stops the iterates from sliding off to infinity.

## Why I Care
I first met completeness as a topology exercise and filed it away as abstract bookkeeping. It stopped being abstract during a numerical analysis project in my third year. I was coding a Picard iteration to solve $y'(t) = -y(t)^2$ with $y(0)=1$. I discretized the function space by truncating to polynomials of degree $5$, reasoning that smooth solutions should live there. The first three iterates looked perfect. By step six, the coefficients blew up to $10^4$ and the plot oscillated wildly. I spent two days halving step sizes and checking quadrature rules.

My advisor looked at the divergence, then at my basis truncation, and said: "You are iterating in an incomplete subspace. The exact solution is $1/(1+t)$, a rational function. Your polynomial space has a hole exactly where the limit wants to sit. The iteration is trying to converge to something your basis cannot represent, so it compensates by exploding the coefficients." I switched to a Chebyshev spectral discretization, which effectively works in the $L^2$ completion. The same iteration converged in eight steps to machine precision. Completeness ceased to be a definition and became a numerical stability requirement. If your space leaks, your code diverges.

## Common Pitfall
Beginners routinely assume that "Cauchy in $C[0,1]$ implies the limit is continuous." This confuses the space with the metric. The statement is true under $d_\infty$, but false under $d_1(f,g) = \int_0^1 |f-g|\,dt$.

Take $f_n(t) = \min(1, nt)$ on $[0,1]$. Each $f_n$ is continuous, rising linearly from $0$ to $1$ on $[0, 1/n]$ and staying at $1$ thereafter. Compute the $L^1$ distance between terms for $m > n$: the functions differ only on $[0, 1/n]$, and the area between them is exactly $\frac{1}{2}(1/n - 1/m)$. For $n=100, m=200$, $d_1(f_n, f_m) = 0.0025$. For $n=1000, m=2000$, $d_1 = 0.00025$. The sequence is Cauchy in $(C[0,1], d_1)$. 

The pointwise limit is the step function $\mathbb{1}_{(0,1]}$, which jumps at $0$. The $L^1$ limit is the same step function. It is not continuous. The sequence has no limit inside $C[0,1]$ when measured by $d_1$. The pitfall is treating "Cauchy" as a property of the sequence alone. It is a property of the pair $(X, d)$. If you change the metric, you change which sequences are Cauchy and which limits exist. Always specify the metric before claiming convergence or completeness.

## What's next

With metric spaces providing the foundation of distance and convergence, we are ready to add algebraic structure. In the next article, we equip our spaces with a norm — a single function that simultaneously gives us distance and a vector space structure — and discover Banach spaces, where completeness and linearity combine to produce the powerful theorems that make functional analysis tick.

### Specific Questions Ahead
Metric spaces give us distance and convergence, but they lack algebraic structure. I can measure how far two functions are, but I cannot add them, scale them, or talk about linear operators. The next article adds a vector space structure compatible with the metric. This produces normed spaces, and when completeness is present, Banach spaces. You are now equipped to read it because you already know how to handle Cauchy sequences, how completions plug holes, and why convergence depends on the chosen metric. Those are the exact mechanisms that make normed spaces work.

The next article answers four concrete questions:
1. How do we define a norm so that the induced metric $d(x,y) = \|x-y\|$ automatically respects vector addition and scalar multiplication?
2. Why are all norms on $\mathbb{R}^n$ equivalent, and exactly where does the proof break when the dimension becomes infinite?
3. How do we construct the dual space $X^*$ of bounded linear functionals, and why does completeness of $X$ guarantee completeness of $X^*$ regardless of whether $X$ is complete?
4. When can we extend a bounded linear functional from a subspace to the whole space without increasing its norm?

We will answer the last question with the Hahn-Banach Theorem. It is the first major result that requires the norm structure rather than just the metric. The theorem guarantees that dual spaces are never trivial, which in turn lets us separate convex sets, characterize weak convergence, and build the entire machinery of duality. The proof uses Zorn's Lemma to push a functional outward one dimension at a time while preserving the norm bound. You will see exactly why the triangle inequality and absolute homogeneity of the norm are the constraints that make the extension possible. Once Hahn-Banach is in place, the geometry of the unit ball stops being a picture and becomes a computational tool.
