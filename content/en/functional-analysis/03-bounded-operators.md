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
description: "Bounded operators are the morphisms of functional analysis --- continuity and linearity conspire to produce surprising rigidity."
---

## Operators as the objects of study

In finite dimensions, once you pick a basis, every linear map is a matrix. You can write it down, compute its eigenvalues, and go home. In infinite dimensions, "writing down" an operator is usually impossible, and the spectral theory is vastly richer. The first step: figure out which linear maps are well-behaved.

The key insight is that in infinite dimensions, linearity alone doesn't buy you continuity. A linear map between finite-dimensional normed spaces is automatically continuous (because it can be represented as a matrix, and matrices act by bounded operations on coordinates). In infinite dimensions, this fails spectacularly --- and the linear maps that are continuous form a proper subclass that we call "bounded operators."

![Operator norm visualization](/images/functional-analysis/fig03_operator_norm.png)

## Bounded linear operators

Let $X, Y$ be normed spaces. A linear map $T: X \to Y$ is **bounded** if:

$$\|T\| := \sup_{\|x\| \le 1} \|Tx\| < \infty.$$

Equivalently: there exists $C > 0$ such that $\|Tx\| \le C\|x\|$ for all $x$. The infimum of all such $C$ is the operator norm $\|T\|$.

The operator norm has equivalent characterizations:

$$\|T\| = \sup_{\|x\| = 1} \|Tx\| = \sup_{x \ne 0} \frac{\|Tx\|}{\|x\|} = \inf\{C \ge 0 : \|Tx\| \le C\|x\| \ \forall x\}.$$

**Theorem (Bounded = Continuous).** For a linear map between normed spaces, the following are equivalent:
1. $T$ is bounded.
2. $T$ is continuous.
3. $T$ is continuous at a single point (e.g., at $0$).

*Proof.* $(1) \Rightarrow (2)$: $\|Tx - Ty\| = \|T(x-y)\| \le \|T\|\,\|x-y\|$, so $T$ is Lipschitz, hence uniformly continuous. $(2) \Rightarrow (3)$: trivial. $(3) \Rightarrow (1)$: suppose $T$ is continuous at $0$ but unbounded. Then there exist $x_n$ with $\|x_n\| = 1$ and $\|Tx_n\| > n$. Set $y_n = x_n/n$; then $y_n \to 0$ but $\|Ty_n\| = \|Tx_n\|/n > 1$. This contradicts continuity at $0$. $\square$

In finite dimensions, every linear map is bounded (proof: the unit sphere is compact, and a continuous function on a compact set is bounded --- but wait, we haven't yet shown the map is continuous! The real argument is that the map is determined by its action on a finite basis, and sums of finitely many bounded terms are bounded). In infinite dimensions, unbounded linear operators exist and are important (differential operators, unbounded observables in quantum mechanics), but they require domain restrictions and form the subject of a later course.

## The operator norm and $B(X, Y)$

The space $B(X, Y)$ of all bounded linear operators from $X$ to $Y$ is itself a normed space under the operator norm. The norm satisfies the submultiplicativity property: if $S \in B(Y, Z)$ and $T \in B(X, Y)$, then:

$$\|ST\| \le \|S\|\,\|T\|.$$

This makes $B(X) = B(X,X)$ a **Banach algebra** (an algebra with a submultiplicative norm).

**Theorem (Completeness of $B(X,Y)$).** If $Y$ is Banach, then $B(X, Y)$ is Banach, regardless of whether $X$ is complete.

*Proof.* Let $(T_n)$ be Cauchy in $B(X, Y)$. For each $x \in X$, the sequence $(T_n x)$ is Cauchy in $Y$ (since $\|T_n x - T_m x\| \le \|T_n - T_m\|\,\|x\|$). Since $Y$ is complete, define $Tx = \lim_{n} T_n x$. Linearity of $T$ is immediate from linearity of limits. For boundedness: the Cauchy sequence $(\|T_n\|)$ in $\mathbb{R}$ is bounded by some $M$, so $\|Tx\| = \lim \|T_n x\| \le M\|x\|$. For convergence in operator norm: given $\varepsilon > 0$, choose $N$ with $\|T_n - T_m\| < \varepsilon$ for $n, m \ge N$. Then for all $\|x\| \le 1$: $\|T_n x - T_m x\| < \varepsilon$. Letting $m \to \infty$: $\|T_n x - Tx\| \le \varepsilon$. Taking the sup over $\|x\| \le 1$: $\|T_n - T\| \le \varepsilon$. $\square$

This theorem is important: it says the space of operators inherits completeness from the target space, not the domain. Even if $X$ is an incomplete normed space, $B(X, Y)$ is Banach whenever $Y$ is.

## Fundamental examples

**Example 1: The right shift.** On $\ell^2$, define $S(x_1, x_2, x_3, \ldots) = (0, x_1, x_2, \ldots)$. Then $\|Sx\|_2 = \|x\|_2$ for all $x$, so $\|S\| = 1$ and $S$ is an **isometry**. But $S$ is not surjective: the vector $(1, 0, 0, \ldots)$ is not in the range. No finite-dimensional isometry can fail to be surjective --- this is purely infinite-dimensional.

**Example 2: The left shift.** $L(x_1, x_2, x_3, \ldots) = (x_2, x_3, \ldots)$. Again $\|L\| = 1$ (surjective but not isometric --- $Le_1 = 0$). Note $LS = I$ but $SL \ne I$: the left shift followed by the right shift loses the first component. Left inverses and right inverses can differ --- something impossible for square matrices.

**Example 3: Multiplication operators.** On $L^2[0,1]$, define $(M_\varphi f)(t) = \varphi(t) f(t)$ for $\varphi \in L^\infty[0,1]$. Then $\|M_\varphi\| = \|\varphi\|_\infty$ (the essential supremum). These are the "diagonal" operators --- the spectral theorem says every normal operator on a separable Hilbert space is unitarily equivalent to a multiplication operator on some $L^2$ space.

**Example 4: Integral operators.** On $L^2[0,1]$, define $(Tf)(s) = \int_0^1 k(s,t) f(t)\, dt$ for a kernel $k \in L^2([0,1]^2)$. By Cauchy-Schwarz, $|Tf(s)|^2 \le \int |k(s,t)|^2 dt \cdot \int |f(t)|^2 dt$, so $\|Tf\|_2^2 \le \|k\|_2^2 \|f\|_2^2$. This gives $\|T\| \le \|k\|_{L^2([0,1]^2)}$. These "Hilbert-Schmidt" operators are always compact (more on this in Part 5).

**Example 5: The Volterra operator.** $(Vf)(s) = \int_0^s f(t)\, dt$ on $L^2[0,1]$. This is bounded with $\|V\| = 2/\pi$ (a non-trivial calculation). Unlike most integral operators, $V$ has no eigenvalues at all --- $Vf = \lambda f$ implies $f' = f/\lambda$, giving $f = ce^{t/\lambda}$, but then $(Vf)(0) = 0$ forces $c = 0$. The spectrum is $\{0\}$, purely continuous. This is impossible in finite dimensions.

## The dual space

The most important special case: $Y = \mathbb{C}$ (or $\mathbb{R}$). A bounded linear functional is a bounded linear map $f: X \to \mathbb{C}$. The **dual space** is:

$$X^* = B(X, \mathbb{C}).$$

By the completeness theorem above, $X^*$ is always Banach (since $\mathbb{C}$ is complete), even if $X$ isn't. The dual space is always well-behaved.

**Identifying duals --- the $\ell^p$ story.** For $1 \le p < \infty$, the dual of $\ell^p$ is $\ell^q$ where $1/p + 1/q = 1$:

$$(\ell^p)^* \cong \ell^q, \quad f(x) = \sum_{n=1}^\infty x_n y_n \text{ for unique } y \in \ell^q.$$

The isomorphism is isometric: $\|f\| = \|y\|_q$.

*Proof sketch for $(\ell^1)^* \cong \ell^\infty$.* Given $f \in (\ell^1)^*$, define $y_n = f(e_n)$. Then $|y_n| = |f(e_n)| \le \|f\|\|e_n\|_1 = \|f\|$, so $y \in \ell^\infty$ with $\|y\|_\infty \le \|f\|$. Conversely, for $x = \sum x_n e_n \in \ell^1$: $f(x) = \sum x_n y_n$ and $|f(x)| \le \|y\|_\infty \|x\|_1$, giving $\|f\| \le \|y\|_\infty$. Combined: $\|f\| = \|y\|_\infty$. $\square$

For $p = 2$: $(\ell^2)^* \cong \ell^2$ --- the Riesz representation theorem from Part 2.

But $(\ell^\infty)^* \ne \ell^1$. The dual of $\ell^\infty$ is strictly larger --- it contains "Banach limits" and other exotic functionals that can't be written as convergent series. Similarly, $(L^\infty)^* \supsetneq L^1$. This asymmetry is a key feature of infinite-dimensional analysis: duality is not always symmetric.

## Weak convergence

The dual space gives us a weaker notion of convergence. A sequence $(x_n)$ in $X$ **converges weakly** to $x$ (written $x_n \rightharpoonup x$) if:

$$f(x_n) \to f(x) \quad \forall f \in X^*.$$

Strong (norm) convergence implies weak convergence (since $|f(x_n) - f(x)| \le \|f\|\|x_n - x\|$), but not conversely.

**Example 6.** In $\ell^2$, the standard basis vectors $e_n \rightharpoonup 0$ weakly (since $\langle e_n, y \rangle = y_n \to 0$ for any $y \in \ell^2$ by Bessel's inequality), but $\|e_n\| = 1 \not\to 0$. The unit sphere in infinite dimensions is weakly "floppy" --- sequences on it can converge weakly to the origin.

This never happens in finite dimensions: there, weak and strong convergence coincide (because the dual is finite-dimensional, so testing against finitely many functionals suffices). The distinction matters enormously for variational problems: minimizing sequences in the calculus of variations typically converge weakly but not strongly.

**Theorem (Weak sequential completeness).** In a reflexive Banach space, every bounded sequence has a weakly convergent subsequence. (This is the Eberlein-Smulian theorem applied to reflexive spaces.)

## Adjoint operators

For $T \in B(X, Y)$, the **(Banach space) adjoint** $T^*: Y^* \to X^*$ is defined by:

$$(T^* g)(x) = g(Tx), \quad g \in Y^*, \ x \in X.$$

This is the infinite-dimensional "transpose." Always $\|T^*\| = \|T\|$ (proved using the Hahn-Banach theorem --- Part 4).

For Hilbert spaces, the Riesz representation identifies $H^* \cong H$, and we get the **Hilbert space adjoint** $T^*: H \to H$ satisfying:

$$\langle Tx, y \rangle = \langle x, T^* y \rangle \quad \forall x, y \in H.$$

The right shift and left shift on $\ell^2$ are Hilbert adjoints: $S^* = L$ (verify: $\langle Sx, y \rangle = \sum_{n=1}^\infty x_n \overline{y_{n+1}} = \langle x, Ly \rangle$).

**Self-adjoint operators** ($T = T^*$) satisfy $\langle Tx, y \rangle = \langle x, Ty \rangle$ for all $x, y$. They have real spectrum --- the infinite-dimensional analogue of symmetric matrices having real eigenvalues. In quantum mechanics, observables are self-adjoint operators, and their spectral values are the possible measurement outcomes.

**Normal operators** ($TT^* = T^*T$) include self-adjoint and unitary operators as special cases. The spectral theorem applies to all normal operators.

## Compact operators: a preview

An operator $T: X \to Y$ is **compact** if it maps bounded sets to relatively compact sets (sets with compact closure). Equivalently, every bounded sequence $(x_n)$ has a subsequence such that $(Tx_{n_k})$ converges.

In finite dimensions, every bounded operator is compact (bounded sets in finite-dimensional spaces have compact closure). In infinite dimensions, compact operators are "almost finite-dimensional" --- they can be approximated by finite-rank operators (operators whose range is finite-dimensional).

**Example 7.** On $\ell^2$, the diagonal operator $D_\lambda(x_1, x_2, \ldots) = (\lambda_1 x_1, \lambda_2 x_2, \ldots)$ where $\lambda_n \to 0$ is compact. Specifically, $T(x_1, x_2, \ldots) = (x_1, x_2/2, x_3/3, \ldots)$ is compact. Its finite-rank truncations $T_N(x_1, \ldots) = (x_1, x_2/2, \ldots, x_N/N, 0, 0, \ldots)$ converge to $T$ in operator norm: $\|T - T_N\| = 1/(N+1) \to 0$.

**Non-example.** The identity $I: \ell^2 \to \ell^2$ is not compact: the sequence $e_n$ is bounded but has no convergent subsequence (since $\|e_n - e_m\| = \sqrt{2}$ for $n \ne m$). The failure of the identity to be compact is equivalent to the space being infinite-dimensional.

Compact operators form a closed two-sided ideal in $B(X)$: if $T$ is compact and $S$ is bounded, both $ST$ and $TS$ are compact. They are the subject of the spectral theory in Part 5.

## What's next

We've described the players: spaces and operators. Now comes the power --- the four fundamental theorems that make the whole theory work. Hahn-Banach says dual spaces are rich. Open mapping says surjective operators are well-behaved. Closed graph says you can check boundedness indirectly. Uniform boundedness says pointwise bounds give uniform bounds. Each one relies on completeness in an essential way.

---

*This is Part 3 of [Functional Analysis](/en/series/functional-analysis/) (6 parts).
Previous: [Part 2 --- Inner Product Spaces and Hilbert Spaces](/en/functional-analysis/02-hilbert-spaces/) · Next: [Part 4 --- The Big Four Theorems](/en/functional-analysis/04-big-theorems/)*
