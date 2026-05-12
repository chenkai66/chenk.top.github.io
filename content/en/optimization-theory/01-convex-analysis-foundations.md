---
title: 'Optimization (1): Convex Analysis Foundations'
date: 2022-09-14 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
series: optimization-theory
series_order: 1
lang: en
mathjax: true
description: "The geometric and analytic toolkit that unlocks the rest of the series: convex sets, convex functions, the conjugate (Fenchel) transform, subgradients, and the indicator/support function pair. Includes complete proofs of Jensen's inequality, the projection theorem, and the subdifferential of basic norms."
disableNunjucks: true
translationKey: "optim-01"
---

This article is the foundation the rest of the series is built on. Almost every result we will prove later — convergence rates of gradient descent, Lagrangian duality, the proximal operator, even the analysis of stochastic methods — relies on a small set of facts about convex sets and convex functions. We will derive all of them from scratch.

If you only remember three things from this article, make it these:

- A set is convex iff it contains every line segment between its points; a function is convex iff its **epigraph** is a convex set.
- The **conjugate** $f^*$ is the Legendre transform generalized — it converts pointwise inequalities into linear ones and is the bridge between primal and dual problems.
- The **subdifferential** $\partial f(x)$ is the right notion of "gradient" for non-smooth convex functions; it is non-empty whenever $x$ lies in the relative interior of $\mathrm{dom}(f)$.

## What you will learn

1. Convex sets, convex hulls, the projection theorem (with proof).
2. Convex functions and the four equivalent characterizations (definition, epigraph, first-order condition, second-order condition).
3. Operations that preserve convexity — pointwise sup, composition, perspective.
4. The conjugate function $f^*$ and Fenchel--Young inequality.
5. Subgradients and the subdifferential calculus.
6. Worked computations: $\partial \|x\|_1$, $\partial \|x\|_2$, $\partial \max\{0, x\}$.

## Prerequisites

Linear algebra (inner products, norms), basic real analysis (limits, continuity, supremum), and multivariable calculus (gradients, Hessians). No prior optimization background required.

---

## 1. Convex sets

### 1.1 Definition and basic examples

A set $C \subseteq \mathbb{R}^n$ is **convex** if for every $x, y \in C$ and every $\lambda \in [0, 1]$,
$$
\lambda x + (1 - \lambda) y \in C.
$$
Geometrically: the segment connecting any two points of $C$ stays inside $C$.

Examples worth knowing by heart:

- Affine subspaces $\{x : Ax = b\}$ and halfspaces $\{x : a^\top x \leq b\}$.
- Norm balls $\{x : \|x\| \leq r\}$ for any norm.
- The positive semidefinite cone $\mathbf{S}^n_+ = \{X \in \mathbb{R}^{n \times n} : X = X^\top, X \succeq 0\}$.
- The probability simplex $\Delta^{n-1} = \{p \in \mathbb{R}^n_+ : \sum_i p_i = 1\}$.

A surprising one: the set of *invertible* matrices is **not** convex. Consider $X = I$ and $Y = -I$; their midpoint is the zero matrix.

![Convex set vs non-convex set: a set is convex iff every line segment between two of its points stays inside.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig1_convex_set.png)

### 1.2 Operations that preserve convexity

The following constructions take convex inputs to convex outputs. Each is a one-line check from the definition; you should be able to write the proofs without looking.

| Operation                             | Statement                                                                  |
| ------------------------------------- | -------------------------------------------------------------------------- |
| Intersection                          | $\bigcap_{i \in I} C_i$ is convex if each $C_i$ is.                        |
| Affine image                          | $A C + b = \{Ax + b : x \in C\}$ is convex.                                |
| Cartesian product                     | $C_1 \times C_2$ is convex.                                                |
| Sum                                   | $C_1 + C_2 = \{x + y : x \in C_1, y \in C_2\}$ is convex.                  |
| Inverse affine image                  | $\{x : Ax + b \in C\}$ is convex.                                          |

The intersection rule is the most useful in practice — it is the reason a polyhedron $\{x : Ax \leq b\}$ is convex (it is an intersection of halfspaces) and why the feasible set of a convex optimization problem is convex.

### 1.3 The projection theorem

A theorem we will use repeatedly:

> **Projection theorem.** Let $C \subseteq \mathbb{R}^n$ be a non-empty closed convex set and $y \in \mathbb{R}^n$. There exists a unique point $\pi_C(y) \in C$ minimizing $\|x - y\|_2$ over $x \in C$. Moreover, $z = \pi_C(y)$ if and only if
> $$
> \langle y - z, x - z \rangle \leq 0 \quad \text{for all } x \in C.
> $$

**Proof of existence.** Let $d = \inf_{x \in C} \|x - y\|_2$ and pick a sequence $\{x_k\} \subseteq C$ with $\|x_k - y\|_2 \to d$. We show $\{x_k\}$ is Cauchy. By the parallelogram identity applied to $x_k - y$ and $x_m - y$,
$$
\|x_k - x_m\|_2^2 = 2 \|x_k - y\|_2^2 + 2 \|x_m - y\|_2^2 - 4 \left\| \tfrac{x_k + x_m}{2} - y \right\|_2^2.
$$
Since $C$ is convex, $\frac{x_k + x_m}{2} \in C$, so $\|\frac{x_k + x_m}{2} - y\|_2 \geq d$. Thus
$$
\|x_k - x_m\|_2^2 \leq 2 \|x_k - y\|_2^2 + 2 \|x_m - y\|_2^2 - 4 d^2 \to 0
$$
as $k, m \to \infty$. The sequence converges to some $z$, which lies in $C$ because $C$ is closed and satisfies $\|z - y\|_2 = d$.

**Proof of uniqueness.** Suppose $z_1$ and $z_2$ both achieve the minimum. By the same parallelogram argument with $x_k = z_1$ and $x_m = z_2$:
$$
\|z_1 - z_2\|_2^2 \leq 2 d^2 + 2 d^2 - 4 d^2 = 0,
$$
so $z_1 = z_2$.

**Proof of the variational inequality.** Suppose $z = \pi_C(y)$. For any $x \in C$ and $\lambda \in (0, 1]$, the point $z + \lambda (x - z) = (1 - \lambda) z + \lambda x \in C$, so
$$
\|y - z\|_2^2 \leq \|y - z - \lambda (x - z)\|_2^2 = \|y - z\|_2^2 - 2 \lambda \langle y - z, x - z \rangle + \lambda^2 \|x - z\|_2^2.
$$
Rearranging and dividing by $\lambda$:
$$
2 \langle y - z, x - z \rangle \leq \lambda \|x - z\|_2^2.
$$
Letting $\lambda \to 0^+$ gives the inequality. The converse is a similar calculation: assume the inequality holds for all $x \in C$ and expand $\|x - y\|_2^2 = \|(x - z) - (y - z)\|_2^2 \geq \|y - z\|_2^2$. $\blacksquare$

The projection theorem has a beautiful geometric reading: $\pi_C(y)$ is the point of $C$ where the segment $y \to z$ meets $C$ at an angle of at most $90^\circ$ to every other direction inside $C$.

![Projection of $y$ onto a closed convex set $C$: $z = \pi_C(y)$ is the unique closest point, and the residual $y - z$ makes a non-acute angle with every direction $x - z$ pointing into $C$.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig2_projection.png)

---

## 2. Convex functions

### 2.1 Four equivalent characterizations

Let $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ with **effective domain** $\mathrm{dom}(f) = \{x : f(x) < +\infty\}$ assumed convex. The following are equivalent and we will use them interchangeably:

**Definition.** For all $x, y \in \mathrm{dom}(f)$ and $\lambda \in [0, 1]$:
$$
f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y). \tag{D}
$$

**Epigraph characterization.** The set $\mathrm{epi}(f) = \{(x, t) \in \mathbb{R}^{n+1} : f(x) \leq t\}$ is a convex subset of $\mathbb{R}^{n+1}$.

**First-order condition (assumes $f$ differentiable).** For all $x, y \in \mathrm{dom}(f)$:
$$
f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle. \tag{F1}
$$

**Second-order condition (assumes $f$ twice differentiable).** For all $x \in \mathrm{int}(\mathrm{dom}(f))$:
$$
\nabla^2 f(x) \succeq 0. \tag{F2}
$$

Equivalence of (D) and the epigraph: $(\lambda x + (1 - \lambda) y, \lambda s + (1 - \lambda) t) \in \mathrm{epi}(f)$ for any $(x, s), (y, t) \in \mathrm{epi}(f)$ is exactly inequality (D) when we take $s = f(x), t = f(y)$.

(D) $\Rightarrow$ (F1): rearrange (D) as
$$
\frac{f(x + \lambda (y - x)) - f(x)}{\lambda} \leq f(y) - f(x).
$$
Letting $\lambda \to 0^+$ on the left yields the directional derivative $\langle \nabla f(x), y - x \rangle$.

(F1) $\Rightarrow$ (F2): set $y = x + t v$ for small $t$ and a direction $v$. Then (F1) says
$$
f(x + tv) \geq f(x) + t \langle \nabla f(x), v \rangle.
$$
Taylor expansion gives $f(x + tv) = f(x) + t \langle \nabla f(x), v \rangle + \tfrac{t^2}{2} v^\top \nabla^2 f(x) v + o(t^2)$. Comparing yields $v^\top \nabla^2 f(x) v \geq 0$.

(F2) $\Rightarrow$ (D): integrate twice. Specifically, for the line $g(\lambda) = f((1 - \lambda) x + \lambda y)$, $g''(\lambda) = (y - x)^\top \nabla^2 f((1 - \lambda) x + \lambda y) (y - x) \geq 0$, so $g$ is convex on $[0, 1]$, which is exactly (D).

![Two equivalent views of convexity: (left) the first-order condition says the tangent at any point is a global lower bound on $f$; (right) the epigraph $\mathrm{epi}(f)$ is itself a convex set in $\mathbb{R}^{n+1}$.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig3_convex_function.png)

### 2.2 Strict and strong convexity

- **Strictly convex**: (D) holds with strict inequality whenever $x \neq y$ and $\lambda \in (0, 1)$.
- **$\mu$-strongly convex** (for $\mu > 0$): $f - \frac{\mu}{2} \|x\|_2^2$ is convex. Equivalently,
  $$
  f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2} \|y - x\|_2^2.
  $$

Strong convexity is to convexity what "uniformly continuous" is to continuous — it gives a quantitative gap and is what makes optimization rates concrete. We unpack it in detail in article 02.

### 2.3 Examples

You should be able to verify each of these with the second-order condition (or directly):

| $f(x)$                        | Convex? | Justification                                                    |
| ----------------------------- | ------- | ---------------------------------------------------------------- |
| $\|x\|_p$ for $p \geq 1$      | Yes     | Triangle inequality + positive homogeneity.                      |
| $\log(1 + e^x)$ (softplus)    | Yes     | $f''(x) = \frac{e^x}{(1 + e^x)^2} > 0$.                          |
| $-\log \det X$ on $\mathbf{S}^n_{++}$ | Yes | Hessian is $X^{-1} \otimes X^{-1}$; PSD.                         |
| $x \log x$ on $\mathbb{R}_+$  | Yes     | $f''(x) = 1/x > 0$.                                              |
| $x^4$                         | Strictly convex | $f''(x) = 12 x^2 \geq 0$, vanishes only at $x = 0$ (not strong). |
| $\frac{1}{2} x^\top Q x$, $Q \succ 0$ | $\lambda_{\min}(Q)$-strongly convex | $\nabla^2 f = Q$.        |

### 2.4 Operations preserving convexity

| Operation                                | Convexity preserved?                                                  |
| ---------------------------------------- | --------------------------------------------------------------------- |
| Non-negative weighted sum                | Yes ($\sum_i w_i f_i$ with $w_i \geq 0$).                             |
| Pointwise supremum                       | $\sup_{i \in I} f_i$ is convex (epigraph is intersection of epigraphs). |
| Composition with affine map              | $g(x) = f(Ax + b)$ inherits convexity.                                |
| Composition $g(x) = h(f(x))$             | Convex if $h$ is convex non-decreasing and $f$ is convex.             |
| Perspective $g(x, t) = t f(x/t)$ for $t > 0$ | Convex.                                                          |
| Partial minimization                     | $g(x) = \inf_y f(x, y)$ is convex if $f$ is jointly convex.           |

The pointwise supremum rule is the secret weapon: it explains why the **support function** $\sigma_C(y) = \sup_{x \in C} \langle y, x \rangle$ is convex regardless of $C$, and why the maximum of convex functions is convex.

---

## 3. The conjugate function

For any $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ (not necessarily convex), define the **conjugate** (or **Legendre--Fenchel transform**):
$$
f^*(y) = \sup_{x \in \mathbb{R}^n} \big[ \langle y, x \rangle - f(x) \big].
$$

The conjugate $f^*$ is **always convex**, even if $f$ is not — it is the pointwise supremum of affine functions of $y$.

### 3.1 Geometric reading

For a fixed slope $y$, $f^*(y)$ is the largest value of $\langle y, x \rangle - f(x)$. Equivalently, the affine function $x \mapsto \langle y, x \rangle - f^*(y)$ is the highest affine minorant of $f$ with slope $y$. So $f^*$ tracks, for each slope, how far the supporting hyperplane sits below the graph.

![Geometric meaning of the conjugate: for slope $y$, the highest affine minorant of $f$ with that slope is $x \mapsto y\,x - f^*(y)$; the value $f^*(y)$ measures the vertical drop from the origin to where this line crosses the $y$-axis.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig4_conjugate.png)

### 3.2 Fenchel--Young inequality

Directly from the definition:
$$
f(x) + f^*(y) \geq \langle x, y \rangle. \tag{FY}
$$
Equality holds iff $y \in \partial f(x)$ (introduced below). This is the convex-analytic generalization of the AM-GM and Young inequalities; setting $f(x) = \frac{1}{p} |x|^p$ for $p > 1$ and computing $f^*$ recovers the classical Young inequality $\frac{|x|^p}{p} + \frac{|y|^q}{q} \geq xy$ with $\frac{1}{p} + \frac{1}{q} = 1$.

### 3.3 Worked conjugates

Memorize these — we use them as building blocks throughout the series.

| $f(x)$                                            | $f^*(y)$                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------- |
| $\frac{1}{2} \|x\|_2^2$                           | $\frac{1}{2} \|y\|_2^2$ (self-conjugate).                                  |
| $\frac{1}{2} x^\top Q x$, $Q \succ 0$             | $\frac{1}{2} y^\top Q^{-1} y$.                                             |
| Indicator $\iota_C$                               | Support function $\sigma_C(y) = \sup_{x \in C} \langle y, x \rangle$.      |
| $\|x\|$ (any norm)                                | $\iota_{B^\circ}(y) = \begin{cases} 0 & \|y\|_* \leq 1 \\ +\infty & \text{else} \end{cases}$ where $\|\cdot\|_*$ is the dual norm. |
| $e^x$                                             | $y \log y - y$ for $y > 0$ (entropy-like).                                 |
| $\log(1 + e^x)$                                   | $y \log y + (1 - y) \log(1 - y)$ for $y \in [0, 1]$.                       |
| $-\log x$ on $x > 0$                              | $-1 - \log(-y)$ on $y < 0$.                                                |

The pair (norm, indicator of dual unit ball) explains the entire LASSO duality story we develop in article 06.

### 3.4 Biconjugate and why convexity is closure under $(f^*)^*$

The biconjugate $f^{**} = (f^*)^*$ satisfies $f^{**} \leq f$ pointwise. The two are equal exactly when $f$ is convex and lower semicontinuous; in that case $f^{**} = f$. So the operation "take the conjugate twice" is the smallest convex closed function below $f$ — the **convex envelope** of $f$. This is why people sometimes write convex relaxations as $f^{**}$.

---

## 4. Subgradients

### 4.1 The subdifferential

For convex $f$, the **subdifferential** at $x$ is
$$
\partial f(x) = \{g \in \mathbb{R}^n : f(y) \geq f(x) + \langle g, y - x \rangle \text{ for all } y\}.
$$
A vector $g$ in $\partial f(x)$ is a **subgradient** at $x$. Compare with (F1): when $f$ is differentiable, $\partial f(x) = \{\nabla f(x)\}$, a singleton.

When $f$ is not differentiable, $\partial f(x)$ can be a non-singleton set. The point of subgradients is that *every* convex function has them at every interior point of its domain — even if it has no derivative — and the "$g = 0$" condition replaces "$\nabla f = 0$" as the optimality criterion:
$$
x^\star \in \arg\min f \iff 0 \in \partial f(x^\star).
$$

### 4.2 Existence and basic calculus

> **Theorem.** If $f$ is convex and $x \in \mathrm{relint}(\mathrm{dom}(f))$, then $\partial f(x)$ is non-empty.

This follows from the existence of a supporting hyperplane to the convex set $\mathrm{epi}(f)$ at $(x, f(x))$ (the supporting hyperplane theorem). When $x$ is on the boundary of the domain, $\partial f(x)$ may be empty — for example, $f(x) = -\sqrt{x}$ on $[0, \infty)$ has $\partial f(0) = \emptyset$.

Calculus rules — these are the workhorses of proximal methods:

| Rule                              | Statement                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------- |
| Sum                               | $\partial(f + g)(x) = \partial f(x) + \partial g(x)$ if $\mathrm{relint}\,\mathrm{dom}\,f \cap \mathrm{relint}\,\mathrm{dom}\,g \neq \emptyset$. |
| Affine composition                | $\partial(f \circ A)(x) = A^\top \partial f(Ax)$ when $\mathrm{relint}\,\mathrm{dom}\,f \cap \mathrm{range}(A) \neq \emptyset$. |
| Pointwise max                     | $\partial \max_i f_i(x) = \mathrm{conv} \bigcup_{i \in I(x)} \partial f_i(x)$ where $I(x) = \{i : f_i(x) = \max_j f_j(x)\}$. |
| Conjugate equivalence             | $g \in \partial f(x) \iff x \in \partial f^*(g) \iff f(x) + f^*(g) = \langle x, g \rangle$. |

### 4.3 Worked examples

**Example 1: $f(x) = |x|$ on $\mathbb{R}$.**
$$
\partial f(x) = \begin{cases} \{1\} & x > 0 \\ \{-1\} & x < 0 \\ [-1, 1] & x = 0. \end{cases}
$$
At $x = 0$, every slope between $-1$ and $1$ defines a tangent line that lies below $|x|$ everywhere.

![Subgradients of $|x|$ at the kink: every slope in $[-1, 1]$ produces a line through the origin that stays below $|x|$, so $\partial f(0) = [-1, 1]$.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig5_subgradient.png)

**Example 2: $f(x) = \|x\|_1$ on $\mathbb{R}^n$.**

Componentwise the same: $g \in \partial \|\cdot\|_1(x)$ iff
$$
g_i = \begin{cases} \mathrm{sign}(x_i) & x_i \neq 0 \\ \in [-1, 1] & x_i = 0. \end{cases}
$$
This is exactly the structure ISTA exploits to produce sparse solutions.

**Example 3: $f(x) = \|x\|_2$ on $\mathbb{R}^n$.**
$$
\partial \|\cdot\|_2(x) = \begin{cases} \{x / \|x\|_2\} & x \neq 0 \\ \{g : \|g\|_2 \leq 1\} & x = 0. \end{cases}
$$

**Example 4: hinge loss $f(x) = \max\{0, 1 - x\}$.**
$$
\partial f(x) = \begin{cases} \{0\} & x > 1 \\ \{-1\} & x < 1 \\ [-1, 0] & x = 1. \end{cases}
$$
The point $x = 1$ is the kink — at this margin boundary, the loss can be assigned any slope in $[-1, 0]$, which is what lets SVM duality go through cleanly.

### 4.4 Optimality: from $\nabla f = 0$ to $0 \in \partial f$

The result we will rely on most: if $f$ is convex, then $x^\star \in \arg\min f$ iff $0 \in \partial f(x^\star)$. The forward direction is (D) applied to $y = x^\star$ and arbitrary $x$. For the reverse, $0 \in \partial f(x^\star)$ means $f(y) \geq f(x^\star)$ for all $y$.

For constrained problems $\min_{x \in C} f(x)$ with $C$ closed convex and $f$ convex, the optimality condition becomes:
$$
0 \in \partial f(x^\star) + N_C(x^\star),
$$
where $N_C(x^\star) = \{g : \langle g, y - x^\star \rangle \leq 0 \ \forall y \in C\}$ is the **normal cone** of $C$ at $x^\star$. We will revisit this with KKT in article 08.

---

## 5. Putting it together: a worked problem

Consider LASSO:
$$
\min_x F(x) := \tfrac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1,
$$
with $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, $\lambda > 0$.

The first term is convex (quadratic with PSD Hessian $A^\top A$); the second is convex (norm). So $F$ is convex. By the optimality condition,
$$
0 \in A^\top (Ax^\star - b) + \lambda \, \partial \|\cdot\|_1(x^\star).
$$
Componentwise, defining $r = A^\top (Ax^\star - b)$:
$$
r_i = -\lambda \, \mathrm{sign}(x^\star_i) \text{ if } x^\star_i \neq 0, \quad r_i \in [-\lambda, \lambda] \text{ if } x^\star_i = 0.
$$
This is the celebrated **KKT condition for LASSO** — it tells us that any coordinate with $|r_i| < \lambda$ must have $x^\star_i = 0$, i.e. why $\ell_1$ regularization produces sparsity. We rederive it as a fixed point of the soft-threshold operator in article 06.

---

## 7. Summary

| Concept           | What it lets you do                                                            |
| ----------------- | ------------------------------------------------------------------------------ |
| Convex set        | Talk about feasibility and projection cleanly.                                  |
| Convex function   | Use any of the four equivalent characterizations to verify convexity quickly.   |
| Conjugate $f^*$   | Cross between primal and dual, derive Young-type inequalities, build duals.    |
| Subgradient       | Replace $\nabla f$ when $f$ is non-smooth; state optimality as $0 \in \partial f$. |
| Normal cone       | State optimality on a constrained domain; bridge to KKT.                        |

The next article makes this concrete: we show that adding two extra assumptions — Lipschitz smoothness and strong convexity — gives quantitative convergence rates for gradient descent.

## Where the story continues

- Article 02 builds smoothness and strong convexity on top of these foundations and proves the convergence of GD in three regimes.
- Article 06 uses subgradients and the conjugate to derive ISTA, FISTA, and ADMM.
- Article 08 generalizes the optimality condition $0 \in \partial f$ to constrained problems via the KKT system.

## Further reading

- Boyd & Vandenberghe, *Convex Optimization*, Ch. 2--3 — the canonical reference for sets and functions.
- Rockafellar, *Convex Analysis* — the deeper monograph; everything about subgradients and conjugates is here.
- Bertsekas, *Convex Optimization Theory*, Ch. 1 — concise and self-contained.
- Nesterov, *Lectures on Convex Optimization*, Ch. 1 — the modern algorithmic angle.
