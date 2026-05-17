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

## Normed spaces

A metric space has distance but no algebraic structure. A **normed space** is a vector space $X$ over $\mathbb{R}$ (or $\mathbb{C}$) equipped with a norm $\|\cdot\|: X \to [0, \infty)$ satisfying:

$$\|x\| = 0 \iff x = 0, \quad \|\alpha x\| = |\alpha|\,\|x\|, \quad \|x + y\| \le \|x\| + \|y\|.$$

Every norm induces a metric via $d(x, y) = \|x - y\|$, but not every metric comes from a norm. The norm adds homogeneity and translation invariance — the geometry looks the same everywhere.

**The $\ell^p$ spaces.** For $1 \le p < \infty$, define:

$$\ell^p = \left\{ (x_n)_{n=1}^\infty : \sum_{n=1}^\infty |x_n|^p < \infty \right\}, \quad \|x\|_p = \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p}.$$

For $p = \infty$:

$$\ell^\infty = \left\{ (x_n) : \sup_n |x_n| < \infty \right\}, \quad \|x\|_\infty = \sup_n |x_n|.$$

These are all different spaces with genuinely different geometries. The unit ball in $\ell^1$ has corners. The unit ball in $\ell^2$ is round. The unit ball in $\ell^\infty$ is a cube. In finite dimensions this is cosmetic; in infinite dimensions it changes which operators are bounded, which functionals are continuous, and which theorems apply.

**Theorem (Norm equivalence fails).** In infinite-dimensional normed spaces, non-equivalent norms exist. Specifically, $\|x\|_1$ and $\|x\|_\infty$ on $\ell^1$ are not equivalent — there's no constant $C$ with $\|x\|_1 \le C\|x\|_\infty$ for all $x \in \ell^1$.

*Proof sketch.* Take $x^{(n)} = (1, 1, \ldots, 1, 0, 0, \ldots)$ with $n$ ones. Then $\|x^{(n)}\|_1 = n$ but $\|x^{(n)}\|_\infty = 1$. No finite $C$ works. $\square$

## Completeness and Banach spaces

A normed space where every Cauchy sequence converges is called a **Banach space**:

$$\text{Banach space} = \text{complete normed space}.$$

Completeness is not optional — it's the dividing line between spaces where analysis works and spaces where it doesn't. Fixed point theorems, the Baire category theorem, the open mapping theorem — all require completeness.

**Theorem (Completeness of $\ell^p$).** For $1 \le p \le \infty$, the space $\ell^p$ is a Banach space.

*Proof sketch for $\ell^p$, $p < \infty$.* Let $(x^{(k)})$ be Cauchy in $\ell^p$. For each fixed $n$, the sequence $(x^{(k)}_n)_k$ is Cauchy in $\mathbb{R}$ (since $|x^{(k)}_n - x^{(m)}_n| \le \|x^{(k)} - x^{(m)}\|_p$), so it converges to some $x_n$. Define $x = (x_n)$. To show $x \in \ell^p$ and $\|x^{(k)} - x\|_p \to 0$: fix $\varepsilon > 0$, choose $K$ so $\|x^{(k)} - x^{(m)}\|_p < \varepsilon$ for $k, m \ge K$. For any finite $N$, $\sum_{n=1}^N |x^{(k)}_n - x^{(m)}_n|^p < \varepsilon^p$. Let $m \to \infty$: $\sum_{n=1}^N |x^{(k)}_n - x_n|^p \le \varepsilon^p$. Since this holds for all $N$, we get $\|x^{(k)} - x\|_p \le \varepsilon$. $\square$

**Example 3.** The space $C[0,1]$ with the sup norm is Banach (uniform limit of continuous functions is continuous). With the $L^1$ norm, it's not — its completion is $L^1[0,1]$.

## A non-example: why completion matters

Consider the space $c_{00}$ of sequences that are eventually zero, with the $\ell^2$ norm. This is a normed space but not Banach: the sequence $x^{(n)} = (1, 1/2, 1/3, \ldots, 1/n, 0, 0, \ldots)$ is Cauchy (check it), but its limit $(1, 1/2, 1/3, \ldots) \notin c_{00}$.

Every normed space has a completion — a minimal Banach space containing it as a dense subspace. The completion of $c_{00}$ under $\|\cdot\|_2$ is $\ell^2$. This is the infinite-dimensional analogue of completing $\mathbb{Q}$ to get $\mathbb{R}$.

## The Baire category theorem: a first payoff

**Theorem (Baire).** In a complete metric space, the countable intersection of dense open sets is dense.

Equivalently: a complete metric space cannot be written as a countable union of nowhere-dense sets. This sounds abstract, but it's the engine behind three of the four big theorems in Chapter 4.

**Application.** There exist continuous functions on $[0,1]$ that are nowhere differentiable. In fact, "most" continuous functions (in the Baire category sense) are nowhere differentiable — the differentiable ones form a meager set in $C[0,1]$.

## What's next

We have spaces with distance and linear structure. The next step is to add geometry — angles, orthogonality, projections. That means inner products, which leads to Hilbert spaces: the infinite-dimensional analogues of Euclidean space, and the natural home of quantum mechanics and Fourier analysis.
