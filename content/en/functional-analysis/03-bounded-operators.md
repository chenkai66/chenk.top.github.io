---
title: "Bounded Linear Operators and Functionals"
date: 2021-03-15 09:00:00
tags:
  - functional-analysis
  - bounded-operators
  - dual-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 3
series_total: 6
translationKey: "functional-analysis-3"
description: "Bounded operators are the morphisms of functional analysis — continuity and linearity conspire to produce surprising rigidity."
---

## Operators as the objects of study

In finite dimensions, once you pick a basis, every linear map is a matrix. You can write it down, compute its eigenvalues, and go home. In infinite dimensions, "writing down" an operator is usually impossible, and the spectral theory is vastly richer. The first step: figure out which linear maps are well-behaved.

## Bounded linear operators

Let $X, Y$ be normed spaces. A linear map $T: X \to Y$ is **bounded** if:

$$\|T\| := \sup_{\|x\| \le 1} \|Tx\| < \infty.$$

Equivalently: there exists $C > 0$ such that $\|Tx\| \le C\|x\|$ for all $x$.

**Theorem.** For a linear map between normed spaces, bounded $\iff$ continuous $\iff$ continuous at $0$.

*Proof.* Bounded $\Rightarrow$ continuous: $\|Tx - Ty\| = \|T(x-y)\| \le \|T\|\,\|x-y\|$, so $T$ is Lipschitz. Continuous at $0$ $\Rightarrow$ bounded: if $T$ is unbounded, find $x_n$ with $\|x_n\| = 1$ and $\|Tx_n\| > n$. Then $x_n/n \to 0$ but $\|T(x_n/n)\| > 1$, contradicting continuity at $0$. $\square$

In finite dimensions, every linear map is bounded. In infinite dimensions, unbounded linear operators exist and are important (differential operators in PDE, for instance), but they require delicate domain considerations. For now, we focus on bounded ones.

## The operator norm and $B(X, Y)$

The space $B(X, Y)$ of all bounded linear operators from $X$ to $Y$ is itself a normed space under the operator norm. Moreover:

**Theorem.** If $Y$ is Banach, then $B(X, Y)$ is Banach (regardless of whether $X$ is complete).

*Proof sketch.* Let $(T_n)$ be Cauchy in $B(X, Y)$. For each $x$, $(T_n x)$ is Cauchy in $Y$ (since $\|T_n x - T_m x\| \le \|T_n - T_m\|\,\|x\|$). Define $Tx = \lim T_n x$. Linearity is clear. Boundedness: $\|T_n\|$ is Cauchy in $\mathbb{R}$, hence bounded by some $M$. So $\|Tx\| = \lim \|T_n x\| \le M\|x\|$. Convergence in operator norm: standard $\varepsilon/3$ argument. $\square$

**Example 1: The right shift.** On $\ell^2$, define $S(x_1, x_2, x_3, \ldots) = (0, x_1, x_2, \ldots)$. Then $\|S\| = 1$, $S$ is an isometry (it preserves norms), but $S$ is not surjective. No finite-dimensional isometry fails to be surjective — this is a purely infinite-dimensional phenomenon.

**Example 2: The left shift.** $L(x_1, x_2, x_3, \ldots) = (x_2, x_3, \ldots)$. Again $\|L\| = 1$. Note $LS = I$ but $SL \ne I$ (information is lost). The shifts are adjoints of each other.

**Example 3: Multiplication operators.** On $L^2[0,1]$, define $(M_\varphi f)(t) = \varphi(t) f(t)$ for $\varphi \in L^\infty[0,1]$. Then $\|M_\varphi\| = \|\varphi\|_\infty$. These are the "diagonal" operators — the spectral theorem says every normal operator looks like one of these (on the right measure space).

## The dual space

The most important special case: $Y = \mathbb{C}$ (or $\mathbb{R}$). A bounded linear functional is a bounded linear map $f: X \to \mathbb{C}$. The **dual space** is:

$$X^* = B(X, \mathbb{C}).$$

By the theorem above, $X^*$ is always Banach (since $\mathbb{C}$ is complete), even if $X$ isn't.

**Identifying duals — the $\ell^p$ story.** For $1 \le p < \infty$, the dual of $\ell^p$ is $\ell^q$ where $1/p + 1/q = 1$:

$$(\ell^p)^* \cong \ell^q, \quad f(x) = \sum_{n=1}^\infty x_n y_n \text{ for unique } y \in \ell^q.$$

The isomorphism is isometric: $\|f\| = \|y\|_q$. For $p = 2$, we get $(\ell^2)^* \cong \ell^2$ — that's the Riesz representation theorem. For $p = 1$, the dual is $\ell^\infty$.

But $(\ell^\infty)^* \ne \ell^1$. The dual of $\ell^\infty$ is strictly larger — it contains "Banach limits" and other exotic functionals that can't be written as convergent series. This asymmetry is a key feature of infinite-dimensional analysis.

## Weak convergence

The dual space gives us a weaker notion of convergence. A sequence $(x_n)$ in $X$ **converges weakly** to $x$ (written $x_n \rightharpoonup x$) if:

$$f(x_n) \to f(x) \quad \forall f \in X^*.$$

Strong (norm) convergence implies weak convergence, but not conversely.

**Example 4.** In $\ell^2$, the standard basis vectors $e_n \rightharpoonup 0$ weakly (since $\langle e_n, y \rangle = y_n \to 0$ for any $y \in \ell^2$), but $\|e_n\| = 1 \not\to 0$. The unit sphere in infinite dimensions is weakly "floppy" — sequences on it can converge weakly to the origin.

This never happens in finite dimensions. In $\mathbb{R}^n$, weak and strong convergence coincide. The distinction is purely infinite-dimensional, and it matters enormously for variational problems (minimizing sequences converge weakly but not strongly).

## Adjoint operators

For $T \in B(X, Y)$, the **adjoint** $T^*: Y^* \to X^*$ is defined by:

$$(T^* g)(x) = g(Tx), \quad g \in Y^*, \ x \in X.$$

This is the "transpose" generalized. Always $\|T^*\| = \|T\|$.

For Hilbert spaces, the Riesz theorem lets us identify the adjoint as an operator $T^*: H \to H$ satisfying $\langle Tx, y \rangle = \langle x, T^* y \rangle$. The right shift and left shift on $\ell^2$ are adjoints: $S^* = L$.

**Self-adjoint operators** ($T = T^*$) have real spectrum — the infinite-dimensional analogue of symmetric matrices having real eigenvalues. They're the operators of quantum mechanics (observables).

## Compact operators: a preview

An operator $T: X \to Y$ is **compact** if it maps bounded sets to relatively compact sets (sets with compact closure). Equivalently, every bounded sequence $(x_n)$ has a subsequence such that $(Tx_{n_k})$ converges.

In finite dimensions, every bounded operator is compact. In infinite dimensions, compact operators are "almost finite-dimensional" — they can be approximated by finite-rank operators.

**Example 5.** On $\ell^2$, the diagonal operator $T(x_1, x_2, \ldots) = (x_1, x_2/2, x_3/3, \ldots)$ is compact. The identity $I: \ell^2 \to \ell^2$ is not compact (the sequence $e_n$ is bounded but has no convergent subsequence since $\|e_n - e_m\| = \sqrt{2}$).

Compact operators form a closed two-sided ideal in $B(H)$. They're the subject of Chapter 5.

## What's next

We've described the players: spaces and operators. Now comes the power — the four fundamental theorems that make the whole theory work. Hahn-Banach says dual spaces are rich. Open mapping says surjective operators are well-behaved. Closed graph says you can check boundedness indirectly. Uniform boundedness says pointwise bounds give uniform bounds. Each one relies on completeness in an essential way.
