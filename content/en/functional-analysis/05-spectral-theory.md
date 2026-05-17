---
title: "Spectral Theory of Compact Operators"
date: 2021-03-29 09:00:00
tags:
  - functional-analysis
  - spectral-theory
  - compact-operators
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 5
series_total: 6
translationKey: "functional-analysis-5"
description: "Compact operators behave like infinite matrices — their spectrum is discrete, countable, and clusters only at zero."
---

## From eigenvalues to spectrum

In finite dimensions, every linear operator $A: \mathbb{C}^n \to \mathbb{C}^n$ has $n$ eigenvalues (with multiplicity). The matrix is understood once you know its eigenvalues and eigenspaces. In infinite dimensions, the situation is more subtle: an operator might have no eigenvalues at all, yet still have a rich "spectrum" that controls its behavior.

## The spectrum

For a bounded operator $T$ on a Banach space $X$, the **resolvent set** is:

$$\rho(T) = \{\lambda \in \mathbb{C} : (T - \lambda I) \text{ is bijective with bounded inverse}\}.$$

The **spectrum** is the complement:

$$\sigma(T) = \mathbb{C} \setminus \rho(T).$$

The spectrum splits into three parts:

$$\sigma_p(T) = \{\lambda : T - \lambda I \text{ is not injective}\} \quad \text{(point spectrum / eigenvalues)},$$
$$\sigma_c(T) = \{\lambda : T - \lambda I \text{ is injective, has dense range, but not surjective}\} \quad \text{(continuous spectrum)},$$
$$\sigma_r(T) = \{\lambda : T - \lambda I \text{ is injective, range not dense}\} \quad \text{(residual spectrum)}.$$

In finite dimensions, $\sigma(T) = \sigma_p(T)$ always. In infinite dimensions, all three pieces can be nonempty.

**Example 1: Right shift on $\ell^2$.** The operator $S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$ has no eigenvalues (if $Sx = \lambda x$, then $0 = \lambda x_1$ and $x_n = \lambda x_{n+1}$, forcing $x = 0$ for $|\lambda| \le 1$, and $x \notin \ell^2$ for $|\lambda| > 1$). Yet $\sigma(S) = \overline{D}$, the closed unit disk. Every $\lambda$ with $|\lambda| < 1$ is in $\sigma_r(S)$ — the range of $S - \lambda I$ isn't dense.

**Basic facts.** The spectrum is always:
- Nonempty (for operators on complex Banach spaces — proved using Liouville's theorem for operator-valued analytic functions).
- Compact, contained in the disk $\{|\lambda| \le \|T\|\}$.
- The spectral radius is $r(T) = \sup_{\lambda \in \sigma(T)} |\lambda| = \lim_{n \to \infty} \|T^n\|^{1/n}$.

## Compact operators

An operator $T \in B(X, Y)$ is **compact** if the image of the unit ball $\overline{T(B_X)}$ is compact in $Y$. Equivalently: every bounded sequence has a subsequence whose image converges.

Key properties:
- Finite-rank operators are compact.
- The compact operators $K(X)$ form a closed two-sided ideal in $B(X)$.
- $T$ compact iff $T$ is a norm-limit of finite-rank operators (true for Hilbert spaces; for general Banach spaces this is the approximation property).
- The identity $I$ is compact iff the space is finite-dimensional.

**Example 2: Integral operators.** On $L^2[0,1]$, define:

$$(Tf)(s) = \int_0^1 k(s,t) f(t)\, dt$$

where $k \in L^2([0,1]^2)$. This Hilbert-Schmidt operator is compact. The kernel function $k$ plays the role of the "matrix entries" $a_{ij}$, and the integral replaces the sum.

## The Riesz-Schauder theorem (spectral theory of compact operators)

**Theorem.** Let $T$ be a compact operator on an infinite-dimensional Banach space $X$. Then:

1. $0 \in \sigma(T)$ (always — since $I$ is not compact, $T$ can't be invertible... unless $X$ is finite-dimensional).
2. $\sigma(T) \setminus \{0\}$ consists entirely of eigenvalues (point spectrum only — no continuous or residual spectrum away from $0$).
3. The set of nonzero eigenvalues is at most countable, with $0$ as the only possible accumulation point.
4. Each nonzero eigenvalue has finite multiplicity (the eigenspace is finite-dimensional).

This is the closest infinite-dimensional analogue of the finite-dimensional spectral theorem. Compact operators behave like "finite-dimensional operators plus a small perturbation."

*Proof sketch of (2) and (4).* For $\lambda \ne 0$: if $T - \lambda I$ is injective, show it's surjective (hence $\lambda \notin \sigma(T)$). The key tool is the **Riesz lemma**: if the range of $T - \lambda I$ were a proper closed subspace, we could find unit vectors far from it, and compactness of $T$ would produce a convergent subsequence contradicting the properness. For finite multiplicity: the eigenspace $\ker(T - \lambda I)$ is a subspace on which $T$ acts as $\lambda I$; if it were infinite-dimensional, the unit ball in it couldn't have a compact image under $T$ (since scaling by $\lambda$ doesn't make it compact). $\square$

## The spectral theorem for compact self-adjoint operators

On a Hilbert space, self-adjoint compact operators have the cleanest possible structure:

**Theorem.** Let $T$ be a compact self-adjoint operator on a separable Hilbert space $H$. Then there exists an orthonormal sequence $(e_n)$ of eigenvectors with real eigenvalues $(\lambda_n)$ such that $\lambda_n \to 0$ and:

$$Tx = \sum_{n=1}^\infty \lambda_n \langle x, e_n \rangle\, e_n \quad \forall x \in H.$$

The operator is "diagonalized" by its eigenvectors. This is the infinite-dimensional analogue of the spectral theorem for symmetric matrices.

**Example 3.** The operator $(Tf)(s) = \int_0^1 \min(s,t) f(t)\, dt$ on $L^2[0,1]$ is compact and self-adjoint. Its eigenvalues are $\lambda_n = 1/((n-1/2)^2\pi^2)$ with eigenfunctions $e_n(t) = \sqrt{2}\sin((n-1/2)\pi t)$. This operator is the inverse of $-d^2/dt^2$ with boundary conditions $f(0) = f'(1) = 0$ — spectral theory of compact operators gives you eigenfunction expansions for differential equations.

## Fredholm alternative

**Theorem (Fredholm alternative).** For a compact operator $T$ on a Banach space and $\lambda \ne 0$, exactly one of the following holds:

(a) $(T - \lambda I)x = y$ has a unique solution for every $y$ (i.e., $\lambda \notin \sigma(T)$).

(b) $(T - \lambda I)x = 0$ has a nontrivial solution (i.e., $\lambda$ is an eigenvalue).

There's no middle ground — either the equation is uniquely solvable for all right-hand sides, or the homogeneous equation has nontrivial solutions. This is the infinite-dimensional version of the fact that a square matrix is either invertible or has a nontrivial null space.

**Application.** Consider the integral equation $f(s) - \lambda \int_0^1 k(s,t)f(t)\,dt = g(s)$. The Fredholm alternative says: either this has a unique solution for every $g$, or the homogeneous version has a nontrivial solution. No other behavior is possible. This is the theoretical foundation of numerical methods for integral equations.

## Beyond compact: a glimpse

For general bounded operators, the spectrum can be much wilder:
- Multiplication by $t$ on $L^2[0,1]$: $\sigma(M_t) = [0,1]$, purely continuous spectrum, no eigenvalues.
- The bilateral shift on $\ell^2(\mathbb{Z})$: $\sigma = \{|\lambda| = 1\}$, the unit circle, purely continuous spectrum.

The full spectral theorem (for normal operators on Hilbert spaces) replaces the eigenvalue sum with an integral against a spectral measure. But that's a story for a graduate course.

## What's next

The final chapter connects functional analysis to PDE: distributions generalize functions, Sobolev spaces give the right framework for weak solutions, and the abstract machinery we've built (completeness, dual spaces, compact operators) becomes the engine that drives existence and regularity theory.
