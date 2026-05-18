---
title: "Functional Analysis (4): Dual Spaces and the Hahn-Banach Theorem — Taming Linear Functionals"
date: 2021-10-07 09:00:00
tags:
  - functional-analysis
  - hahn-banach
  - dual-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "The Hahn-Banach theorem guarantees enough continuous linear functionals exist to separate points — the foundation for duality theory in functional analysis."
disableNunjucks: true
series_order: 4
series_total: 12
translationKey: "functional-analysis-4"
---

## Why You Cannot Skip This Article

Up to now, the theory has been about spaces and the elements that live in them. This article changes the perspective: it asks what you can say about a vector $x$ by *measuring* $x$ against a family of test functionals. The shift from "vectors" to "vectors plus functionals" is what turns Banach spaces into a serviceable analogue of finite-dimensional linear algebra. In finite dimensions, every linear functional is continuous and the dual space is the same dimension as the original — so there is nothing to prove. In infinite dimensions, continuity is a real constraint, and the existence of enough continuous functionals to separate points or extend partial data is *not* obvious. The Hahn-Banach theorem is what guarantees this, and it is the result that makes functional analysis possible.

![Dual space: functionals as hyperplanes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/04_dual_space.png)

A working analyst uses Hahn-Banach the way a working algebraist uses Zorn's lemma: invisibly, dozens of times a day, never proving it from scratch. The point of this article is to produce the theorem cleanly and inspect a few of its standard consequences — the geometric form, the existence of supporting hyperplanes, the canonical embedding into the bidual. Article 5 will then put the dual to work in the form of weak topologies.

## The Dual Space

Let $X$ be a normed space over $\mathbb{R}$ or $\mathbb{C}$. The **dual space** $X^*$ is the space of bounded (equivalently, continuous) linear functionals $\varphi: X \to \mathbb{C}$, equipped with the **dual norm**
$$\|\varphi\|_{X^*} = \sup_{\|x\| \leq 1} |\varphi(x)|.$$
Under this norm, $X^*$ is a Banach space — even when $X$ itself is not (a Cauchy sequence of functionals is pointwise Cauchy, the limit defines a linear functional, and the boundedness passes to the limit by uniform Cauchy-ness).

![Hahn-Banach separation theorem](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/04_hahn_banach_separation.png)

So the dual is automatically a Banach space, regardless of the original space's completeness. This is one of the structural niceties that makes dual constructions so popular: forming the dual *upgrades* incomplete normed spaces to complete ones.

### Classical dualities

Some dual identifications you should know:

- $(\ell^p)^* = \ell^q$ for $1 < p < \infty$, $1/p + 1/q = 1$, via $y \mapsto \varphi_y(x) = \sum y_n x_n$.
- $(\ell^1)^* = \ell^\infty$, same formula.
- $(\ell^\infty)^* \supsetneq \ell^1$ — the dual of $\ell^\infty$ contains the finitely additive measures on $\mathbb{N}$, strictly larger than $\ell^1$ (Banach limits are the classic non-$\ell^1$ examples).
- $(c_0)^* = \ell^1$.
- $(L^p[\Omega])^* = L^q[\Omega]$ for $1 \leq p < \infty$, $1/p + 1/q = 1$ (with the convention $1/\infty = 0$, so $(L^1)^* = L^\infty$).
- $(C[K])^* = M[K]$, the space of finite signed Borel measures on the compact metric space $K$ (the Riesz-Markov theorem).
- $(\mathcal{H})^* = \mathcal{H}$ for Hilbert spaces, by Riesz (Article 3).

![Geometric interpretation of the dual space as hyperplanes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_1_dual_geom.png)

The pattern $(\ell^p)^* = \ell^q$ is so clean it almost looks like a coincidence, but it is forced by Hölder's inequality. The pairing $\langle x, y \rangle = \sum x_n y_n$ between $\ell^p$ and $\ell^q$ is bounded with $|\langle x, y \rangle| \leq \|x\|_p \|y\|_q$, and Hölder's inequality is sharp on appropriately chosen vectors. The argument generalizes verbatim to $L^p$ on any measure space.

### Numerical example

In $\ell^2$, take $y = (1, 1/2, 1/3, \ldots, 1/n, 0, 0, \ldots)$ for $n = 4$, so $y = (1, 1/2, 1/3, 1/4, 0, \ldots)$. The dual functional $\varphi_y(x) = \sum y_k x_k$ has norm $\|\varphi_y\|_{(\ell^2)^*} = \|y\|_2 = \sqrt{1 + 1/4 + 1/9 + 1/16} = \sqrt{205/144} \approx 1.193$. By Cauchy-Schwarz this norm is attained at $x = y / \|y\|_2$, which gives $\varphi_y(x) = \|y\|_2$. The duality is tight — Cauchy-Schwarz is the saturation case.

## The Hahn-Banach Theorem (Analytic Form)

**Theorem (Hahn-Banach, real version).** Let $X$ be a real vector space, $p: X \to \mathbb{R}$ a sublinear functional ($p(x + y) \leq p(x) + p(y)$ and $p(\alpha x) = \alpha p(x)$ for $\alpha \geq 0$), and $\varphi_0: M \to \mathbb{R}$ a linear functional on a subspace $M \subseteq X$ with $\varphi_0(x) \leq p(x)$ for all $x \in M$. Then $\varphi_0$ extends to a linear functional $\varphi: X \to \mathbb{R}$ with $\varphi(x) \leq p(x)$ for all $x \in X$.

![Hahn-Banach extension](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/04_hahn_banach_extension.png)

The complex version: replace sublinear by *seminorm* (so $p(\alpha x) = |\alpha| p(x)$), require $|\varphi_0(x)| \leq p(x)$ on $M$, and the extension satisfies $|\varphi(x)| \leq p(x)$ on $X$.

The version most often quoted: any bounded linear functional on a subspace of a normed space extends to a bounded linear functional on the whole space without enlarging its norm. This is just the seminorm case with $p(x) = \|\varphi_0\|_M \cdot \|x\|$.

![Hahn-Banach extension of a bounded functional from a subspace to the whole space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_2_hb_extension.png)

### Sketch of proof

Step 1 (single-step extension). Given $\varphi_0$ on $M$ and $x_0 \notin M$, set $M' = M + \mathbb{R} x_0$. Any extension is determined by the value $c = \varphi(x_0)$. The constraint $\varphi(m + tx_0) \leq p(m + tx_0)$ for all $m \in M$ and $t \in \mathbb{R}$ becomes (after some manipulation of cases $t > 0, t < 0$) two inequalities $A \leq c \leq B$ where $A = \sup_{m \in M} (\varphi_0(m) - p(m - x_0))$ and $B = \inf_{m \in M} (p(m + x_0) - \varphi_0(m))$. Sublinearity guarantees $A \leq B$ (a calculation), so a valid $c \in [A, B]$ exists.

Step 2 (Zorn's lemma). Order the partial extensions by inclusion of domains and graphs, take a maximal one. The maximal extension must be defined on all of $X$, else step 1 produces a strictly larger extension, contradicting maximality. $\square$

The use of Zorn's lemma is unavoidable in general; the theorem fails in ZF without choice. However, for *separable* normed spaces, Hahn-Banach can be proved without choice — pick a countable dense subset and extend one direction at a time, using only the countable-step version of step 1. This subtlety rarely matters in practice.

### Why this matters

Hahn-Banach lets me do three things that would otherwise be impossible. (i) **Extend** linear functionals from subspaces to the whole space — needed any time I have data on a subset and want a coherent global object. (ii) **Separate** points: there is a continuous functional $\varphi$ with $\varphi(x_0) = \|x_0\|$ and $\|\varphi\| = 1$, by extending the functional $\alpha x_0 \mapsto \alpha \|x_0\|$ on $\mathbb{R} x_0$. So $X^*$ has enough functionals to detect every nonzero element of $X$. (iii) **Compute norms** as $\|x\| = \sup_{\|\varphi\| \leq 1} |\varphi(x)|$, the dual norm of the dual norm, recovering the original norm.

## Geometric Hahn-Banach: Separation of Convex Sets

The "geometric" or "separation" form of Hahn-Banach is more useful in optimization and probability.

![Reflexive spaces: X = X**](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/04_reflexive.png)

**Theorem (Geometric Hahn-Banach).** Let $X$ be a real normed space, $A, B \subseteq X$ disjoint, non-empty, convex sets. (i) If $A$ is open, there exist $\varphi \in X^*$ and $\alpha \in \mathbb{R}$ with $\varphi(a) < \alpha \leq \varphi(b)$ for all $a \in A$, $b \in B$. (ii) If $A$ is closed and $B$ is compact, there exists $\varphi \in X^*$ and $\alpha < \beta$ with $\varphi(a) \leq \alpha < \beta \leq \varphi(b)$ for all $a \in A$, $b \in B$ — *strict* separation.

In words: any two disjoint convex sets can be separated by a hyperplane, with strict separation if one of the sets is closed and the other is compact (specifically separated by a slab).

![Hahn-Banach geometric form: separating two convex sets by a hyperplane](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_3_separation.png)

### Sketch of proof

Use the Minkowski functional $p_A(x) = \inf\{t > 0 : x \in tA\}$ of an open convex set $A$ containing $0$ (if not, translate). $p_A$ is sublinear, and $p_A(x) \leq 1$ iff $x \in A$ (open ball test). Take any $b \in B$ and consider the line $\mathbb{R}(b - a_0)$ for $a_0 \in A$; define a linear functional on this line that is $1$ at $b - a_0$ and bound it by $p_A$. Hahn-Banach extends this to all of $X$. The extended functional separates $A$ from $\{b\}$, and refining slightly (using the gap between $A$ and the closure plus compactness if needed) gives strict separation. $\square$

### Why this matters

The geometric form is the basis of every duality argument in optimization. Convex programming relies on the fact that an infeasible system $A x = b, x \geq 0$ corresponds to a separating hyperplane, and the hyperplane gives a "Farkas-type" certificate of infeasibility. The minimax theorems of game theory are theorems about separating convex sets (the saddle point of a game is the meeting point of two convex hulls). The Choquet integral representation of points in compact convex sets — every point of a compact convex set in a Banach space is the integral of a probability measure on the extreme points — is a deep application of separation.

### Worked Numerical Example

Take $X = \mathbb{R}^2$ equipped with the $\ell^1$ norm $\|x\|_1 = |x_1| + |x_2|$. Let $C = \{x \in \mathbb{R}^2 : x_1 + 2x_2 \leq 2, x_1 \geq 0, x_2 \geq 0\}$, a closed convex triangle with vertices $(0,0), (2,0), (0,1)$. Pick the exterior point $x_0 = (3, 2)$. The distance from $x_0$ to $C$ in $\ell^1$ is attained at the vertex $(2,0)$: $d(x_0, C) = |3-2| + |2-0| = 3$.

The geometric Hahn-Banach theorem guarantees a functional $\varphi \in X^*$ with $\|\varphi\|_\infty = 1$ that separates $x_0$ from $C$ with a gap of exactly $3$. The dual norm is $\|\varphi\|_\infty = \max(|\varphi_1|, |\varphi_2|)$. Try $\varphi(x) = x_1 + x_2$. Then $\|\varphi\|_\infty = 1$. Evaluate on the target: $\varphi(x_0) = 3 + 2 = 5$. Evaluate on $C$: since $x_1, x_2 \geq 0$ and $x_1 + 2x_2 \leq 2$, we have $\varphi(x) = x_1 + x_2 \leq x_1 + 2x_2 \leq 2$. Thus $\sup_{c \in C} \varphi(c) = 2$ (attained at $(2,0)$ and $(0,1)$). The separation inequality reads $\varphi(c) \leq 2 < 5 = \varphi(x_0)$, and the gap $5 - 2 = 3$ matches the computed distance exactly. The hyperplane $\{x : x_1 + x_2 = 3.5\}$ cleanly slices between the triangle and the point.

## Supporting Hyperplanes

A particular case of geometric Hahn-Banach: a closed convex set $C$ in a Banach space $X$ has a supporting hyperplane at every boundary point. That is, for every $x_0 \in \partial C$, there exists $\varphi \in X^*$ with $\varphi(x_0) = \sup_{c \in C} \varphi(c)$.

![Supporting hyperplane to a convex set at a boundary point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_7_supporting.png)

In the language of convex analysis, $\varphi$ belongs to the **subdifferential** of the indicator function of $C$ at $x_0$. The subdifferential is the dual object that records all "supporting directions" at the point.

### Numerical example

Take $C = \{(x_1, x_2) \in \mathbb{R}^2 : x_1^2 + x_2^2 \leq 1\}$, the unit disk. At the boundary point $(\cos\theta, \sin\theta)$, the unique supporting hyperplane is the tangent line, with normal $\varphi(x) = \cos\theta \cdot x_1 + \sin\theta \cdot x_2$. So the supporting hyperplanes at each boundary point are unique — the disk is *smooth*.

Now take $C = \{ x : |x_1| + |x_2| \leq 1 \}$, the unit $\ell^1$ ball. At a vertex like $(1, 0)$, *infinitely many* supporting hyperplanes exist: any $\varphi(x) = a x_1 + b x_2$ with $a = 1$ and $|b| \leq 1$ supports $C$ at $(1,0)$, since $\sup_{c \in C}(c_1 + b c_2) = 1$ for $|b| \leq 1$ (attained at $(1,0)$). So vertices have non-unique supporting hyperplanes — a corner has a "fan" of supports.

This non-uniqueness is exactly the geometric reason $\ell^1$ minimization can have non-unique solutions; the LASSO regression and compressed-sensing literatures spend a lot of energy diagnosing when the solution *is* unique.

## The Bidual and Reflexivity

The dual space $X^*$ is itself a Banach space, so it has its own dual $X^{**} = (X^*)^*$, called the **bidual**. There is a canonical embedding $J: X \to X^{**}$ defined by $(Jx)(\varphi) = \varphi(x)$ for $\varphi \in X^*$. This embedding is well-defined (the linear map $\varphi \mapsto \varphi(x)$ is bounded with norm $\leq \|x\|$), linear, and isometric — the latter using Hahn-Banach to find a $\varphi$ with $|\varphi(x)| = \|x\|$ and $\|\varphi\| = 1$.

![Annihilator of a subspace](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/04_annihilator.png)

A Banach space is **reflexive** if $J$ is surjective, i.e., $X = X^{**}$ canonically. Reflexivity is a strong property; it is preserved under taking closed subspaces, quotients, and finite products, and it implies many compactness and regularity results.

![Canonical embedding V to V** and the meaning of reflexivity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_4_reflexive.png)

### Examples

- All finite-dimensional spaces are reflexive (trivially).
- Hilbert spaces are reflexive: $\mathcal{H}^* = \mathcal{H}$ by Riesz, so $\mathcal{H}^{**} = \mathcal{H}^* = \mathcal{H}$.
- $\ell^p$ and $L^p$ for $1 < p < \infty$ are reflexive: $((L^p)^*)^* = (L^q)^* = L^p$.
- $\ell^1, L^1, c_0, C[K]$ are *not* reflexive. The bidual of $c_0$ is $\ell^\infty$, and $\ell^\infty$ is strictly larger than $c_0$.

### Why reflexivity matters

Reflexivity is equivalent to weak compactness of the closed unit ball — a major theorem of Eberlein and Šmulian (a hint of which appears in Article 5). So in a reflexive space, every bounded sequence has a weakly convergent subsequence — the strongest possible compactness short of norm compactness. This is why minimization arguments in $L^p$ for $1 < p < \infty$ work: take a minimizing sequence, extract a weakly convergent subsequence (by reflexivity), use lower semicontinuity of the norm to pass to the limit. The same approach in $L^1$ or $L^\infty$ fails because of non-reflexivity, and a more delicate argument (compactness in measure, weak-* limits in the dual of $C_0$) is needed.

## Adjoints (Dual) of Bounded Operators

Given a bounded linear operator $T: X \to Y$ between Banach spaces, the **adjoint** (or dual) operator $T^*: Y^* \to X^*$ is defined by $T^*\varphi = \varphi \circ T$, i.e. $(T^*\varphi)(x) = \varphi(T x)$. The adjoint is bounded with $\|T^*\| = \|T\|$ — the upper bound is immediate, and the matching lower bound uses Hahn-Banach to find functionals that almost achieve the norm of $T x$.

Adjoints in general Banach spaces have the formal properties expected from linear algebra: $(S+T)^* = S^* + T^*$, $(\lambda T)^* = \lambda T^*$, $(ST)^* = T^* S^*$, $(T^*)^* = T^{**}$ (which equals $T$ when both spaces are reflexive, identifying $X^{**}$ with $X$). The relationship between the kernel and range:
$$\ker(T^*) = \mathrm{Range}(T)^\perp,\quad \overline{\mathrm{Range}(T)} = \ker(T^*)^\perp,$$
where $\perp$ takes annihilators in the appropriate dual or pre-dual. The closed range theorem gives a finer relation: the range of $T$ is closed in $Y$ iff the range of $T^*$ is closed in $X^*$, and in that case both equal the annihilator of the kernel of the other.

### Numerical example

Take $T: \ell^1 \to \ell^\infty$, $T x = (x_1, x_1 + x_2, x_1 + x_2 + x_3, \ldots)$ — partial sums of a sequence, viewed as a bounded operator. The functional $\varphi_n \in (\ell^\infty)^*$ given by $\varphi_n(y) = y_n$ has norm $1$. Then $T^* \varphi_n \in (\ell^1)^* = \ell^\infty$ is the functional $x \mapsto x_1 + \cdots + x_n$, represented by the bounded sequence $(1, 1, \ldots, 1, 0, 0, \ldots)$ with $n$ ones. As $n \to \infty$, $\|T^*\varphi_n\|_{\ell^\infty} = 1$ remains bounded, illustrating $\|T^*\| \leq \|T\|$ — and in fact $\|T\| = \|T^*\| = 1$ (since the partial sum of sequences with $\ell^1$-norm at most $1$ has $\ell^\infty$-norm at most $1$).

### Worked Numerical Example

Consider the right-shift operator $T: \ell^1 \to \ell^1$ defined by $T(x_1, x_2, x_3, \ldots) = (0, x_1, x_2, \ldots)$. Clearly $\|T\| = 1$. The dual space is $(\ell^1)^* = \ell^\infty$. Take a specific functional $\varphi \in \ell^\infty$ given by the sequence $\varphi = (1, 1/2, 1/4, 1/8, \ldots)$, so $\varphi_n = 2^{-(n-1)}$. Its dual norm is $\|\varphi\|_\infty = 1$.

The adjoint $T^*: \ell^\infty \to \ell^\infty$ acts by precomposition: $(T^*\varphi)(x) = \varphi(Tx)$. Compute explicitly:
$$\varphi(Tx) = \sum_{n=1}^\infty \varphi_n (Tx)_n = \varphi_1 \cdot 0 + \sum_{n=2}^\infty \varphi_n x_{n-1} = \sum_{k=1}^\infty \varphi_{k+1} x_k.$$
Thus $T^*\varphi$ is represented by the shifted sequence $(\varphi_2, \varphi_3, \varphi_4, \ldots) = (1/2, 1/4, 1/8, \ldots)$. The norm drops: $\|T^*\varphi\|_\infty = 1/2$. This matches the operator inequality $\|T^*\varphi\| \leq \|T\| \|\varphi\| = 1 \cdot 1$. If we instead pick $\psi = (1, 1, 1, \ldots) \in \ell^\infty$, then $T^*\psi = (1, 1, 1, \ldots)$ and $\|T^*\psi\|_\infty = 1$, showing the adjoint norm $\|T^*\|$ actually equals $\|T\| = 1$, attained on constant sequences. The calculation verifies the formal identity $(T^*\varphi)_k = \varphi_{k+1}$ and the norm preservation at the operator level.

## Annihilators and Pre-Annihilators

Given a subset $A \subseteq X$, the **annihilator** is
$$A^\perp = \{ \varphi \in X^* : \varphi(a) = 0 \text{ for all } a \in A \}.$$
This is a closed subspace of $X^*$. Dually, given $B \subseteq X^*$,
$$^\perp B = \{ x \in X : \varphi(x) = 0 \text{ for all } \varphi \in B \}$$
is a closed subspace of $X$. The two operations satisfy $^\perp(A^\perp) = \overline{\mathrm{span}(A)}$ for $A \subseteq X$, by Hahn-Banach: any element of $X$ not in $\overline{\mathrm{span}(A)}$ can be separated from $\mathrm{span}(A)$ by a continuous functional vanishing on $A$.

This duality between subspaces and their annihilators is the foundation of every "Fredholm alternative" type theorem. The classical statement: $T x = y$ has a solution iff $\varphi(y) = 0$ for every $\varphi$ with $T^* \varphi = 0$. For closed-range operators (e.g. Fredholm operators, Article 7), the characterization is exact and computable.

### Why this matters

Annihilator duality reduces "where can $T x = y$ be solved?" to "what are the elements of $\ker(T^*)$?" — a question about a different operator on a different space. In PDE this is everyday: the inhomogeneous equation $L u = f$ has a solution iff $f$ is orthogonal to $\ker(L^*)$, where $L^*$ is the formal adjoint of the differential operator (the boundary terms in integration by parts giving the right notion of adjoint). This is sometimes called the "solvability condition" or "compatibility condition."

### Worked Numerical Example

Work in $X = \mathbb{R}^4$ with the Euclidean norm, identifying $X^*$ with $\mathbb{R}^4$ via the dot product. Let $M = \mathrm{span}\{v_1, v_2\}$ where $v_1 = (1, 2, 0, 1)$ and $v_2 = (0, 1, 1, 2)$. The annihilator $M^\perp \subset X^*$ consists of vectors $w$ orthogonal to both $v_1$ and $v_2$. Solve the linear system:
$$w_1 + 2w_2 + w_4 = 0, \quad w_2 + w_3 + 2w_4 = 0.$$
Row reduction yields $w_3 = -w_2 - 2w_4$ and $w_1 = -2w_2 - w_4$. Choosing free variables $(w_2, w_4) = (1, 0)$ gives $u_1 = (-2, 1, -1, 0)$. Choosing $(0, 1)$ gives $u_2 = (-1, 0, -2, 1)$. So $M^\perp = \mathrm{span}\{u_1, u_2\}$.

Now compute the pre-annihilator $^\perp(M^\perp) = \{x \in X : x \cdot u_1 = 0, x \cdot u_2 = 0\}$. The conditions are $-2x_1 + x_2 - x_3 = 0$ and $-x_1 - 2x_3 + x_4 = 0$. Solving this system recovers exactly the original span: $x_2 = 2x_1 + x_3$ and $x_4 = x_1 + 2x_3$. Parameterizing by $x_1 = s, x_3 = t$ gives $x = s(1, 2, 0, 1) + t(0, 1, 1, 2) = s v_1 + t v_2$. Thus $^\perp(M^\perp) = M$. The double-annihilator operation is an exact involution on closed subspaces, and the $2 \times 4$ matrix calculation confirms it with zero numerical slack.

## $L^p$ Duality Spelled Out

For completeness, the duality $(L^p)^* = L^q$ for $1 \leq p < \infty$, $1/p + 1/q = 1$, deserves a careful statement.

**Theorem.** For each $g \in L^q$, the functional $\varphi_g(f) = \int f g$ belongs to $(L^p)^*$, with $\|\varphi_g\|_{(L^p)^*} = \|g\|_{L^q}$. The map $g \mapsto \varphi_g$ is an isometric isomorphism $L^q \to (L^p)^*$.

The Hölder inequality gives $\|\varphi_g\| \leq \|g\|_{L^q}$. The reverse inequality uses an explicit choice of $f$: take $f = |g|^{q-1} \mathrm{sgn}(g) / \|g\|_{L^q}^{q/p}$, normalized so $\|f\|_{L^p} = 1$. Then $\varphi_g(f) = \|g\|_{L^q}$, exhibiting equality.

![Duality (l^p)* = l^q for conjugate exponents](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_6_lp_dual.png)

For surjectivity, given $\varphi \in (L^p)^*$, we need to construct $g \in L^q$ representing it. The Radon-Nikodym theorem provides $g$ as the density of an absolutely continuous measure built from $\varphi$. The case $p = 1$ requires $\sigma$-finiteness of the measure; otherwise the duality can fail.

The case $p = \infty$ is where the pattern breaks. $(L^\infty)^*$ is *strictly larger* than $L^1$ — it contains finitely additive set functions that are not measures. This is one of the structural nuisances of $L^\infty$, and the practical consequence is that $L^\infty$ is not reflexive and many compactness arguments fail there.

### Numerical example

In $L^p[0,1]$, take $f(t) = t$ and consider the functional $\varphi(g) = \int_0^1 g(t) f(t)\,dt = \int_0^1 t \, g(t)\,dt$ acting on $g \in L^p$ (so $f$ plays the role of $g$ in the duality, but with the symbols swapped — $\varphi \in (L^p)^*$ is determined by $f \in L^q$). For $p = 2$, $q = 2$, $\|\varphi\|_{(L^2)^*} = \|f\|_{L^2} = \big(\int_0^1 t^2\,dt\big)^{1/2} = 1/\sqrt{3} \approx 0.577$. Sanity check: by Cauchy-Schwarz, $|\varphi(g)| \leq \|g\|_{L^2} \cdot 1/\sqrt{3}$ for $\|g\|_{L^2} = 1$. Equality at $g(t) = t \sqrt{3}$, which has $L^2$ norm $1$ and $\varphi(g) = \sqrt{3} \int_0^1 t^2\,dt = 1/\sqrt{3}$. The duality is tight.

## A Subtle Use of Hahn-Banach: Banach Limits

A classical and counterintuitive application of Hahn-Banach: there exists a bounded linear functional $L: \ell^\infty(\mathbb{N}) \to \mathbb{R}$ — a **Banach limit** — extending $\lim$ on the subspace of convergent sequences, with $\|L\| = 1$, and translation-invariant: $L((x_2, x_3, \ldots)) = L((x_1, x_2, \ldots))$.

The construction: define a sublinear functional $p(x) = \limsup_{n} \frac{1}{n} \sum_{k=1}^n x_k$ on $\ell^\infty$ (the upper Cesàro average). On the subspace of convergent sequences, $p$ agrees with $\lim$. Hahn-Banach extends $\lim$ to a functional $L: \ell^\infty \to \mathbb{R}$ with $L(x) \leq p(x)$. A bit of work shows $L$ is translation-invariant, with $\|L\| = 1$.

The Banach limit is *not* unique (different Hahn-Banach extensions give different Banach limits) and *not* explicitly definable (no formula for $L$ exists; the construction needs the axiom of choice via Zorn). On the bounded sequence $(0, 1, 0, 1, \ldots)$ — non-convergent in the classical sense — every Banach limit gives $L = 1/2$, by averaging plus translation invariance. So Banach limits give convergence values to all bounded sequences, with the cost that the value depends on which extension you picked.

This is a bizarre but useful object. It is what populates $(\ell^\infty)^* \setminus \ell^1$ — the part of the dual of $\ell^\infty$ that does not come from $\ell^1$ vectors. It is also a clean example of an existence proof in functional analysis that has no constructive analog.

## Weak vs Strong Topology (a Preview of Article 5)

The dual space introduces a new topology on the original space: the **weak topology**, the coarsest topology making all dual functionals continuous. A net $x_\alpha \to x$ weakly iff $\varphi(x_\alpha) \to \varphi(x)$ for every $\varphi \in X^*$. Norm convergence implies weak convergence; the converse is false in infinite dimensions.

![Weak topology vs strong topology compared](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_5_weak_strong.png)

Weak convergence is what you should think of as "convergence of moments" or "convergence of averages." A typical example: in $L^2[0, 2\pi]$, the sequence $f_n(t) = \sin(n t)$ does *not* converge in norm (its norm is $\sqrt{\pi}$ for every $n$), but it converges weakly to $0$ (by the Riemann-Lebesgue lemma: $\int g(t) \sin(nt)\,dt \to 0$ for every $g \in L^2$). The high-frequency oscillation cancels out under integration but does not cancel out in norm. Weak convergence sees the cancellation; norm convergence does not.

The theorem that makes weak topology useful is **Banach-Alaoglu**: the closed unit ball of $X^*$ is compact in the weak-* topology (the dual analogue). When $X$ is reflexive, the closed unit ball of $X$ is weakly compact, by Eberlein-Šmulian. These are the workhorse compactness results of variational analysis. Article 5 will prove them.

## A Concrete Application: Best Approximation in $C[K]$

Let $K$ be a compact metric space and consider the problem: given $f \in C[K]$ and a closed subspace $M \subset C[K]$, find an element of $M$ closest to $f$ in sup norm.

A direct minimizing-sequence approach is delicate in $C[K]$ because the unit ball is not weakly compact (the space is not reflexive). But Hahn-Banach gives an elegant alternative — *duality* of best approximation.

**Theorem.** $d(f, M) = \sup\{ |\varphi(f)| : \varphi \in M^\perp, \|\varphi\| \leq 1 \}$, where $M^\perp \subseteq (C[K])^*$ is the annihilator.

The right-hand side is a maximization over the closed unit ball of the dual space $(M^\perp)$, viewed as a subset of $(C[K])^*$. By Banach-Alaoglu (Article 5), the closed unit ball of $(C[K])^*$ is weak-* compact, and so is its closed subset $M^\perp \cap \overline{B}(0, 1)$. Continuous functions on a compact set attain their supremum, so the sup is achieved by some functional $\varphi^* \in M^\perp$. The duality has converted "find the best approximation" into "find the optimal certifying functional," which is often easier.

This trick is the basis of Chebyshev approximation theory. The functional $\varphi^*$ in question, by Riesz-Markov, is a finite signed measure on $K$ — and a theorem of Markov says it is supported on at most $\dim M + 1$ points (the Chebyshev alternation theorem in disguise). For polynomial approximation on $[a, b]$ this gives the classical Chebyshev equioscillation: the best polynomial approximation oscillates around $f$ at $\geq n+2$ points.

### Numerical example

Approximate $f(t) = t^4$ by polynomials of degree $\leq 2$ on $[-1, 1]$ in sup norm. The best approximation is $p^*(t) = t^2 - 1/8$ (this can be derived from Chebyshev polynomial theory: the best uniform approximation of $t^4$ on $[-1, 1]$ by degree-$\leq 2$ polynomials is the truncation in the Chebyshev basis). The error $f - p^* = t^4 - t^2 + 1/8$ equioscillates at $5$ points in $[-1, 1]$: $\pm 1, \pm 1/\sqrt{2}, 0$, with alternating signs and amplitude $1/8$. So $d(f, M) = 1/8$, attained by an explicit minimax pairing — the dual functional $\varphi^*$ is a discrete measure supported at these $5$ points with appropriate signs.

## The Bipolar Theorem and Closed Convex Hulls

A closed convex set $C$ in a Banach space $X$ containing the origin is determined by its **polar**: $C^\circ = \{ \varphi \in X^* : \varphi(x) \leq 1 \text{ for all } x \in C \}$. Symmetrically, $(C^\circ)^\circ = \{ x \in X : \varphi(x) \leq 1 \text{ for all } \varphi \in C^\circ \}$, and the **bipolar theorem** says $(C^\circ)^\circ = C$ for closed convex $C$ containing $0$ (where the bipolar is taken with respect to the canonical pairing of $X$ and $X^*$).

The bipolar theorem is a direct consequence of Hahn-Banach geometric form: any point not in $C$ can be separated from $C$ by a continuous functional, which lives in $C^\circ$ and witnesses the failure of the bipolar inclusion. So the polar/bipolar duality faithfully represents closed convex sets by their "supporting hyperplane data."

Convex analysis and optimization are mostly about working with this duality. The Fenchel-Legendre transform of a convex function is exactly the polar applied to the epigraph, and the resulting Fenchel duality theorem reduces minimization of $f + g$ to maximization of $-f^* - g^*$ over the dual variables, where $f^*$ is the conjugate. Many of the cleanest algorithms in modern optimization (proximal methods, ADMM, mirror descent) are bookkeeping on the polar duality.

### Worked Numerical Example

Let $X = \mathbb{R}^2$ with the standard pairing. Take the convex set $C = \{(x_1, x_2) : x_1 \geq 0, x_2 \geq 0, x_1 + x_2 \leq 2\}$. This is a right triangle containing the origin. The polar is $C^\circ = \{\varphi \in \mathbb{R}^2 : \varphi_1 x_1 + \varphi_2 x_2 \leq 1 \ \forall x \in C\}$. Testing the vertices of $C$ gives necessary conditions: $\varphi \cdot (0,0) = 0 \leq 1$, $\varphi \cdot (2,0) = 2\varphi_1 \leq 1 \implies \varphi_1 \leq 1/2$, $\varphi \cdot (0,2) = 2\varphi_2 \leq 1 \implies \varphi_2 \leq 1/2$. Since $x_1, x_2 \geq 0$ in $C$, any $\varphi$ with negative components only makes the dot product smaller, so the constraints are exactly $\varphi_1 \leq 1/2$ and $\varphi_2 \leq 1/2$. Thus $C^\circ = \{\varphi : \varphi_1 \leq 1/2, \varphi_2 \leq 1/2\}$, an unbounded quadrant shifted to $(1/2, 1/2)$.

Now compute the bipolar $(C^\circ)^\circ = \{x : \varphi_1 x_1 + \varphi_2 x_2 \leq 1 \ \forall \varphi \in C^\circ\}$. If $x_1 < 0$, pick $\varphi_1 \to -\infty$ to violate the bound, so $x_1 \geq 0$. Similarly $x_2 \geq 0$. With $x_1, x_2 \geq 0$, the supremum of $\varphi \cdot x$ over $C^\circ$ occurs at the corner $\varphi = (1/2, 1/2)$, giving $(1/2)x_1 + (1/2)x_2 \leq 1$, or $x_1 + x_2 \leq 2$. We recover exactly $C$. The bipolar theorem holds with explicit vertex-by-vertex verification: the polar encodes the supporting half-spaces, and the bipolar reconstructs the original intersection.

## A Word on When the Dual Is "Bigger" Than Expected

I have mentioned that $(\ell^\infty)^* \supsetneq \ell^1$ and $(L^\infty)^* \supsetneq L^1$. The exact source is the same in both cases: the dual contains *finitely additive* set functions that are not measures. The space $\mathrm{ba}(\mathcal{A})$ of bounded finitely additive set functions on a $\sigma$-algebra $\mathcal{A}$ is what completes $L^1$ inside $(L^\infty)^*$.

For most working purposes you can pretend $(L^\infty)^* = L^1$ and lose nothing, because the only $L^\infty$-functionals that are "natural" or "constructive" turn out to be in $L^1$. The pathological extras come from Hahn-Banach extensions of finitely-supported limit-like operations and are not detectable without using the axiom of choice. So the sharp statement "$(L^p)^* = L^q$ for $1 \leq p < \infty$" is the working version, and the non-equality at $p = \infty$ is the price of choice.

A related curiosity: $(c_0)^* = \ell^1$, and $(\ell^1)^* = \ell^\infty$, but $(\ell^\infty)^* \neq c_0$. So $c_0$ is not reflexive — its bidual is $\ell^\infty$, which contains $c_0$ properly. The chain $c_0 \subsetneq c_0^{**} = \ell^\infty$ is the cleanest concrete example of non-reflexivity.

## A Tactical Summary

When you see a problem in Banach-space functional analysis, here is the typical Hahn-Banach playbook:

- *Need to extend a partial functional to the whole space?* Hahn-Banach analytic form.
- *Need to separate a point from a closed convex set?* Hahn-Banach geometric form.
- *Need to characterize the closure of a subspace?* Use $\overline{M} = {}^\perp(M^\perp)$.
- *Need to solve $T x = y$?* Apply the Fredholm alternative: $T x = y$ has a solution iff $y \in \ker(T^*)^\perp$.
- *Need to compute the norm of a vector dually?* $\|x\| = \sup_{\|\varphi\| \leq 1} |\varphi(x)|$.
- *Need to set up convex optimization?* Use Fenchel-Rockafellar duality, which is a packaged version of polarity.

Most of these moves are nearly mechanical once you have internalized the dual viewpoint. Hahn-Banach is in the background for all of them, providing the existence of the functionals you need.

## Goldstine and the Density of $X$ in $X^{**}$

A finer point of bidual theory. The canonical embedding $J: X \to X^{**}$ is isometric, and $J(X)$ is a closed subspace of $X^{**}$ (the image of an isometry is closed). When $X$ is non-reflexive, $J(X) \subsetneq X^{**}$. How much "extra" is $X^{**}$ compared to $X$?

**Theorem (Goldstine).** $J(X)$ is dense in $X^{**}$ in the weak-\* topology. More precisely, the unit ball $J(B_X)$ is weak-\* dense in the unit ball of $X^{**}$.

So in the weak-\* topology, every element of $X^{**}$ is approximated by elements of $X$ — the original space, while smaller, is still "weak-\* topologically dense" in its bidual. This is what makes weak-\* convergence the workhorse compactness in non-reflexive settings: bounded sequences in $X^{**}$ have weak-\* convergent subsequences (by Banach-Alaoglu in $X^{**}$, identifying it as the dual of $X^*$), and the limits are weak-\* limits of elements of $X$ — possibly outside $J(X)$, but approximable from it.

The proof of Goldstine is itself a Hahn-Banach argument: if $J(B_X)$ were not weak-\* dense in $B_{X^{**}}$, there would be a weak-\* continuous functional separating them, and weak-\* continuous functionals on $X^{**}$ are exactly elements of $X^*$. The separation contradicts the calculation that $J(B_X)$ has supporting functionals in every direction of $X^*$.

## A Detailed Worked Use of Hahn-Banach: Proving Riesz Representation for $C[K]^*$

The Riesz-Markov-Kakutani theorem identifies $C[K]^*$ with the space $M[K]$ of finite signed Borel measures on a compact metric space $K$. The full proof uses Hahn-Banach in a substantial way; let me sketch how.

Step 1. A bounded linear functional $\varphi$ on $C[K]$ extends, by Hahn-Banach, to a bounded linear functional $\widetilde\varphi$ on the larger space $B(K)$ of bounded Borel-measurable functions on $K$, with the same norm. (One uses a sublinear majorant; the boundedness of $\varphi$ gives the right majorant.)

Step 2. The extension $\widetilde\varphi$ on $B(K)$ defines a finitely additive set function $\mu(E) = \widetilde\varphi(\mathbb{1}_E)$ on Borel sets. The boundedness of $\widetilde\varphi$ translates to the total variation of $\mu$ being at most $\|\varphi\|$.

Step 3. Show $\mu$ is *countably* additive. This uses the regularity of $\mu$ (inner approximation by compact sets, outer by open sets) plus the original hypothesis that $\widetilde\varphi$ comes from a continuous linear functional on $C[K]$. The key is that $\mathbb{1}_{K_1 \cup K_2 \cup \cdots}$ is approximated in some sense by continuous functions when the $K_n$ are nested compact sets.

Step 4. Verify $\widetilde\varphi(f) = \int f\,d\mu$ for all $f \in C[K]$. This is by approximation of continuous $f$ by simple functions and continuity of $\widetilde\varphi$ on $B(K)$.

The role of Hahn-Banach is purely Step 1: extend the functional from $C[K]$ to $B(K)$. Without that extension, there is no candidate for the measure. The non-uniqueness of Hahn-Banach extensions does not matter here, because Steps 3 and 4 force the extension to be the unique one given by integration against the constructed $\mu$. So the construction is canonical even though the intermediate Hahn-Banach extension is not.

This is a typical pattern: Hahn-Banach is used existentially, to provide a place to start; the rest of the argument forces the construction to land on a particular object.

## Quotient Duality and the First Isomorphism Theorem

For a closed subspace $M \subseteq X$, the dual of the quotient is the annihilator of $M$ in $X^*$:
$$(X / M)^* \cong M^\perp.$$
The isomorphism is concrete: a functional on $X / M$ is the same as a functional on $X$ that vanishes on $M$. The norms agree (the quotient norm on the left equals the operator norm of the annihilator functional on the right).

Dually, the dual of a closed subspace is a quotient:
$$M^* \cong X^* / M^\perp.$$
This is more subtle and uses Hahn-Banach: every functional on $M$ extends (non-uniquely) to a functional on $X$, and two extensions differ by an element of $M^\perp$, so the extension is unique modulo $M^\perp$. The norms agree because Hahn-Banach extends without enlarging the norm.

These two duality identities are the Banach-space versions of the "first isomorphism theorem" for groups or rings: subspaces and quotients dualize each other, and the corresponding short exact sequences "$0 \to M \to X \to X/M \to 0$" dualize to "$0 \to M^\perp \to X^* \to M^* \to 0$".

The applications are constant in operator theory. To analyze the range of $T: X \to Y$, identify the *closure* of the range as $\ker(T^*)^\perp$ via the previous identities. To analyze the cokernel $Y / \overline{\mathrm{Range}(T)}$, recognize it as $\ker(T^*)$ by quotient duality. The "Fredholm alternative" of Article 7 — for $T - \lambda I$ with $T$ compact — is exactly this duality applied to a particular family of operators.

### Worked Numerical Example

Take $X = \mathbb{R}^3$ with the $\ell^2$ norm, and $M = \mathrm{span}\{(1, 1, 1)\}$. The quotient space $X/M$ consists of equivalence classes $[x] = x + M$. The quotient norm is $\|[x]\|_{X/M} = \inf_{m \in M} \|x - m\|_2$, which is the Euclidean distance from $x$ to the line $t(1,1,1)$. For $x = (3, 0, 0)$, the orthogonal projection onto $M$ is $(1, 1, 1)$, so the distance is $\|(2, -1, -1)\|_2 = \sqrt{6} \approx 2.449$. Thus $\|[(3,0,0)]\|_{X/M} = \sqrt{6}$.

Now consider the functional $\psi$ on $X/M$ defined by $\psi([x]) = x_1 - x_2$. This is well-defined because $(x_1+t) - (x_2+t) = x_1 - x_2$. The dual norm is $\|\psi\|_{(X/M)^*} = \sup_{[x] \neq 0} |x_1 - x_2| / \|[x]\|_{X/M}$. By quotient duality, this equals the norm of any extension to $X$ that vanishes on $M$. The functional $\Phi(x) = x_1 - x_2$ already vanishes on $(1,1,1)$, so $\Phi \in M^\perp$. Its norm in $X^* \cong \mathbb{R}^3$ is $\|(1, -1, 0)\|_2 = \sqrt{2} \approx 1.414$. Check the quotient formula: for $x = (3,0,0)$, $|\psi([x])| = 3$. Ratio $3 / \sqrt{6} = \sqrt{1.5} \approx 1.225 < \sqrt{2}$. The supremum is attained at $x = (1, -1, 0)$, where $\|[x]\| = \sqrt{2}$ and $|\psi([x])| = 2$, giving ratio $\sqrt{2}$. The isomorphism $(X/M)^* \cong M^\perp$ preserves norms exactly, verified by orthogonal projection geometry.

## Generalized Banach Limits and Cesàro Summability

The Banach-limit construction (above) extends to give a useful tool in summation theory. Given a sequence $(x_n) \in \ell^\infty$, ordinary Cesàro summability says $\frac{1}{N} \sum_{n=1}^N x_n$ converges. Banach limits agree with the Cesàro limit when the latter exists, and extend it to *any* bounded sequence. The price is non-uniqueness: different Banach-limit constructions give different values on non-Cesàro-summable sequences.

A clean theorem: a bounded sequence $(x_n)$ has the same value under *every* Banach limit iff $(x_n)$ is **almost convergent**, where almost convergence means $\frac{1}{N} \sum_{n=k+1}^{k+N} x_n$ converges as $N \to \infty$ uniformly in $k$. So Banach-limit-uniqueness is equivalent to a quantitative averaged convergence condition. A sequence like $(0, 1, 0, 1, \ldots)$ is almost convergent to $1/2$ and has the same Banach-limit value $1/2$ regardless of which Banach limit we use. A sequence with more pathological behavior (built from indicators of Bohr-positive sets, say) has Banach-limit values that depend on the choice.

Almost convergence sits between Cesàro convergence and bounded convergence — it is a way of recovering "limit" semantics for sequences that fail to converge in any pointwise sense. The functional analytic content is purely Hahn-Banach: the existence of a single translation-invariant bounded linear extension of $\lim$ produces almost-convergence as the canonical "best regularization."

## A Fixed-Point Connection: Markov-Kakutani

A classical fixed-point theorem in convex analysis that uses Hahn-Banach machinery:

**Theorem (Markov-Kakutani).** Let $K$ be a non-empty compact convex subset of a Hausdorff topological vector space, and let $\mathcal{F}$ be a commuting family of continuous affine maps $K \to K$. Then there exists a common fixed point $x \in K$ with $T x = x$ for every $T \in \mathcal{F}$.

The proof: for each $T \in \mathcal{F}$, the Cesàro average $A_n^T(x) = \frac{1}{n}(x + T x + \cdots + T^{n-1} x)$ takes $K$ into $K$ (by convexity), and a compactness-cluster argument finds a point fixed by every $A_n^T$ in the limit. The commuting hypothesis lets us coordinate fixed-point arguments across different operators.

Applications include the existence of *invariant means* on amenable groups, the existence of Banach limits (already discussed) as a special case where $\mathcal{F}$ is generated by the shift, and the existence of *Haar measures* on compact groups via averaging arguments. Each application is morally a Hahn-Banach extension theorem in fixed-point disguise.

## Beyond Banach: Locally Convex Spaces

Hahn-Banach in its full strength holds in *locally convex topological vector spaces* — vector spaces with a Hausdorff topology generated by a separating family of seminorms. This generality covers Schwartz functions, distributions, holomorphic functions, and many other natural function spaces that are not Banach.

The geometric form generalizes too: in a locally convex space, two disjoint convex sets — one open, one arbitrary — can be separated by a continuous linear functional. The supporting hyperplane theorem holds at boundary points of any closed convex set.

What does *not* generalize are the more quantitative consequences (closed range theorem, exact dual norms in some cases). The locally convex setting is where Hahn-Banach is most fundamental and most flexible, and where it has the cleanest applications in distribution theory and harmonic analysis on locally compact groups.

## Counterexample: Why the Definition Cannot Be Weakened

The geometric Hahn-Banach theorem promises strict separation of two disjoint convex sets $A$ and $B$ only when one is closed and the other is compact. If both are merely closed and convex in an infinite-dimensional space, strict separation can fail catastrophically: the distance between them can be zero, making any separating hyperplane collapse to a non-strict boundary.

Consider $X = \ell^2$. Define two sets:
$$A = \{x \in \ell^2 : x_{2k} = 0 \text{ for all } k \geq 1\},$$
$$B = \{x \in \ell^2 : x_{2k} = 2^{-k}(x_{2k-1} + 1) \text{ for all } k \geq 1\}.$$
Both are closed affine subspaces, hence convex. They are disjoint: if $x \in A \cap B$, then $0 = 2^{-k}(x_{2k-1} + 1)$ forces $x_{2k-1} = -1$ for all $k$, but the sequence $(-1, 0, -1, 0, \ldots)$ is not in $\ell^2$. So $A \cap B = \varnothing$.

Compute the distance $d(A, B) = \inf\{\|a - b\|_2 : a \in A, b \in B\}$. For any $b \in B$, the closest point in $A$ is obtained by zeroing out the even coordinates of $b$, so $a_{2k-1} = b_{2k-1}$ and $a_{2k} = 0$. The squared distance is
$$\|a - b\|_2^2 = \sum_{k=1}^\infty |b_{2k}|^2 = \sum_{k=1}^\infty 2^{-2k} |b_{2k-1} + 1|^2.$$
We can make this arbitrarily small while keeping $b \in \ell^2$. Choose a truncated sequence $b^{(N)}$ with $b^{(N)}_{2k-1} = -1$ for $k \leq N$ and $0$ for $k > N$. Then $b^{(N)} \in \ell^2$ (finitely many nonzeros), and the distance squared becomes $\sum_{k=N+1}^\infty 2^{-2k} = \frac{4^{-(N+1)}}{1 - 1/4} = \frac{1}{3 \cdot 4^N}$. As $N \to \infty$, $d(A, B) \to 0$.

Since the distance is zero, no continuous linear functional $\varphi$ and scalars $\alpha < \beta$ can satisfy $\varphi(a) \leq \alpha < \beta \leq \varphi(b)$. Any separating hyperplane would require a positive gap. The compactness hypothesis in the strict separation theorem is not decorative; it prevents exactly this asymptotic grazing behavior that infinite dimensions allow.

## Why I Care

I first internalized the Hahn-Banach theorem during a qualifying exam problem on sparse signal recovery. The question asked to prove that a specific vector $x_0 \in \mathbb{R}^n$ is the unique minimizer of an $\ell^1$-regularized least squares problem under a restricted isometry condition. I spent forty minutes trying to construct a dual certificate explicitly: writing down a vector $w$ in the row space of the measurement matrix, bounding its $\ell^\infty$ norm on the complement of the support, and wrestling with triangle inequalities that refused to close. My scratch paper was a mess of component-wise estimates.

Then I remembered the geometric form. The problem was not about coordinates; it was about separating a point from a convex set. The $\ell^1$ ball is a polytope. The measurement constraints define an affine subspace. Uniqueness of the minimizer is equivalent to the affine subspace touching the $\ell^1$ ball at exactly one vertex and otherwise staying outside. Instead of building $w$ entry by entry, I invoked Hahn-Banach to guarantee a supporting hyperplane at that vertex. The hyperplane's normal vector *is* the dual certificate. The existence is automatic once the geometric configuration is verified. I rewrote the proof in six lines: define the convex sets, check disjointness of the relative interior, apply separation, read off the normal. The examiner circled the Hahn-Banach invocation and wrote "this is the point."

That moment killed my habit of brute-forcing functional constructions. Hahn-Banach is not a lemma you prove; it is a permission slip to assume the functional you need exists, provided the convex geometry allows it. I stopped fighting coordinates and started drawing sets.

## Common Pitfall

A persistent misconception is that Hahn-Banach extensions are unique. Students see the norm-preserving extension theorem, assume the extended functional is canonical, and proceed to write $\varphi$ as if it were a well-defined function of the subspace data. It is not. Uniqueness fails whenever the dual norm lacks strict convexity, which happens in almost every space used in applications.

Take $X = \mathbb{R}^2$ with the $\ell^\infty$ norm $\|(x, y)\|_\infty = \max(|x|, |y|)$. Let $M = \{(t, t) : t \in \mathbb{R}\}$ be the diagonal subspace. Define $\varphi_0$ on $M$ by $\varphi_0(t, t) = t$. The norm of $\varphi_0$ is $\sup_{t \neq 0} |t| / |t| = 1$. We want extensions $\varphi(x, y) = \alpha x + \beta y$ to all of $\mathbb{R}^2$ with $\|\varphi\|_1 = |\alpha| + |\beta| = 1$ (since $(\ell^\infty)^* = \ell^1$) that agree with $\varphi_0$ on $M$. Agreement forces $\alpha t + \beta t = t$ for all $t$, so $\alpha + \beta = 1$. The norm constraint is $|\alpha| + |\beta| = 1$.

Solve $\alpha + \beta = 1$ and $|\alpha| + |\beta| = 1$. Any pair $(\alpha, 1-\alpha)$ with $\alpha \in [0, 1]$ works. For $\alpha = 0$, $\varphi(x, y) = y$. For $\alpha = 1$, $\varphi(x, y) = x$. For $\alpha = 1/2$, $\varphi(x, y) = (x+y)/2$. All three have dual norm $1$, all extend $\varphi_0$, and they give different values on $(1, 0)$: $0$, $1$, and $1/2$ respectively. The extension is wildly non-unique.

Uniqueness holds if and only if the dual space $X^*$ is strictly convex (rotund). $\ell^1$ is not strictly convex; its unit ball has flat faces. Whenever your space has an $\ell^1$ or $\ell^\infty$ component, or involves $L^1/L^\infty$, expect multiple norm-preserving extensions. Hahn-Banach guarantees existence, not canonicity. If your proof relies on "the" extension, you have a gap.

## Looking Ahead

We now have the dual space $X^*$ as a real, populated object. Hahn-Banach provides enough functionals to separate points, extend partial data, support convex sets, and characterize norms by duality. The bidual embedding lets us classify spaces by reflexivity. The next big move is to put a topology on $X$ generated by the dual functionals — the weak topology — and on $X^*$ by the embedding into the bidual — the weak-* topology. Both are coarser than the norm topology and admit compactness theorems that fail in the norm topology. Article 5 walks through the weak topologies and the Banach-Alaoglu theorem, and then we have all the tools to attack operator theory in earnest.

These results transform functional analysis from the study of individual operators into a theory with powerful automatic regularity properties — properties that have no analogue in finite dimensions because they are trivially true there. Where the Hahn-Banach theorem guaranteed the existence of enough functionals, the next three theorems will constrain the behavior of families of operators in ways that are impossible to anticipate from finite-dimensional linear algebra alone.

---

### Specific Questions Ahead

You now have the dual space $X^*$ as a populated, computable object. You can extend functionals, separate convex sets, identify annihilators, and recognize when a space embeds into its bidual. The machinery is assembled, but it is currently static. The next step is to make it dynamic by changing the topology.

Article 5 shifts from the norm topology to the weak and weak-* topologies. The norm topology is too fine for infinite-dimensional compactness; closed bounded sets are almost never compact. The dual space provides a coarser topology that recovers compactness at the cost of losing metric structure. You are equipped for this shift because you already know how functionals act on vectors, how the bidual embedding works, and how Banach-Alaoglu will use the product topology on a Cartesian product of disks.

The next article answers these specific questions:
1. Why does a bounded sequence in $L^2[0, 1]$ always have a weakly convergent subsequence, while the same sequence may have no norm-convergent subsequence? We will compute the weak limit of $f_n(t) = \sin(2\pi n t)$ explicitly and show why $\|f_n\|_2 \not\to 0$ but $f_n \rightharpoonup 0$.
2. How does the weak-* topology on $X^*$ differ from the weak topology on $X^*$, and why does the distinction vanish exactly when $X$ is reflexive? We will track the convergence of evaluation functionals in $\ell^1 = (c_0)^*$ to see the difference in action.
3. What is the precise statement and proof of the Banach-Alaoglu theorem, and how does Tychonoff's theorem on product compactness translate into functional analysis? We will build the homeomorphism between the dual unit ball and a closed subset of $\prod_{x \in B_X} \overline{D}(0, \|x\|)$ step by step.
4. How do you prove lower semicontinuity of the norm under weak convergence, and why is this the key to existence proofs in calculus of variations? We will work through a concrete minimization problem where norm convergence fails but weak convergence plus convexity delivers a minimizer.

The central result is the Banach-Alaoglu theorem: the closed unit ball of $X^*$ is compact in the weak-* topology. This is not a curiosity; it is the engine that drives existence theory for PDEs, optimization, and probability. Once you internalize weak compactness, you stop looking for convergent subsequences in norm and start extracting them in duality. Article 5 builds the topology, proves the compactness, and shows you how to use it without getting lost in net convergence.
