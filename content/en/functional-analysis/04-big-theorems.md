---
title: "The Big Four — Hahn-Banach, Open Mapping, Closed Graph, Uniform Boundedness"
date: 2021-03-22 09:00:00
tags:
  - functional-analysis
  - hahn-banach
  - open-mapping-theorem
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 4
series_total: 6
translationKey: "functional-analysis-4"
description: "Four theorems that power all of functional analysis — each exploiting completeness in a different way."
---

## The backbone of the theory

Functional analysis has four foundational theorems. Three of them (open mapping, closed graph, uniform boundedness) use the Baire category theorem and require completeness. The fourth (Hahn-Banach) is algebraic and works without completeness but needs the axiom of choice (or Zorn's lemma). Together they form the structural backbone — almost every deep result in the subject traces back to one of these four.

## Hahn-Banach theorem

**Theorem (Hahn-Banach, extension form).** Let $X$ be a normed space, $M \subseteq X$ a subspace, and $f: M \to \mathbb{R}$ a bounded linear functional with $\|f\| = C$. Then there exists an extension $F: X \to \mathbb{R}$ with $F|_M = f$ and $\|F\| = C$.

The complex version works too: extend real and imaginary parts separately using $\text{Im}\,f(x) = -\text{Re}\,f(ix)$.

*Proof idea.* Extend one dimension at a time. If $M \subsetneq X$, pick $x_0 \notin M$ and define $F$ on $M + \mathbb{R}x_0$ by choosing $F(x_0) = c$. The constraint $|F(m + tx_0)| \le C\|m + tx_0\|$ forces $c$ to lie in a closed interval (computed from the values $f(m)/t$ and norms). The interval is nonempty by the existing bound on $f$. Iterate transfinitely (Zorn's lemma). $\square$

**Consequences that matter:**

1. **Separation.** For any $x \ne 0$ in a normed space, there exists $f \in X^*$ with $f(x) = \|x\|$ and $\|f\| = 1$. The dual space separates points — it's rich enough to "see" every nonzero vector.

2. **$X$ embeds in $X^{**}$.** The canonical map $J: X \to X^{**}$ defined by $(Jx)(f) = f(x)$ is an isometric embedding. If $J$ is surjective, $X$ is called **reflexive**. The spaces $\ell^p$ for $1 < p < \infty$ are reflexive; $\ell^1$ and $\ell^\infty$ are not.

3. **Existence of functionals.** Given a closed subspace $M$ and $x_0 \notin M$, there exists $f \in X^*$ with $f|_M = 0$ and $f(x_0) \ne 0$. Dual spaces detect the difference between subspaces.

**Example 1.** On $c_0$ (sequences converging to 0), the functional $f(x) = x_1$ defined on the subspace $\text{span}(e_1)$ extends to the evaluation functional on all of $c_0$, with norm 1. In fact, $(c_0)^* \cong \ell^1$ — every functional on $c_0$ is represented by an absolutely summable sequence.

## Uniform boundedness principle (Banach-Steinhaus)

**Theorem.** Let $X$ be a Banach space, $Y$ a normed space, and $\{T_\alpha\}_{\alpha \in A} \subseteq B(X, Y)$ a family of bounded operators. If $\sup_\alpha \|T_\alpha x\| < \infty$ for every $x \in X$, then $\sup_\alpha \|T_\alpha\| < \infty$.

Pointwise boundedness implies uniform boundedness. This is genuinely surprising — individual vectors tell you about the worst case over all vectors.

*Proof sketch (via Baire).* Define $F_n = \{x \in X : \sup_\alpha \|T_\alpha x\| \le n\}$. Each $F_n$ is closed (as an intersection of closed sets). By hypothesis, $X = \bigcup F_n$. Since $X$ is Banach (hence Baire), some $F_N$ has nonempty interior: there exist $x_0$ and $r > 0$ with $B(x_0, r) \subseteq F_N$. For $\|x\| \le 1$: $T_\alpha x = T_\alpha(x_0 + rx)/r - T_\alpha(x_0)/r$, giving $\|T_\alpha x\| \le 2N/r$. Hence $\sup_\alpha \|T_\alpha\| \le 2N/r$. $\square$

**Example 2: Divergence of Fourier series.** The partial sum operators $S_n f = \sum_{k=-n}^n \hat{f}(k) e^{ikt}$ on $C[-\pi, \pi]$ satisfy $\|S_n\| \to \infty$ (the Lebesgue constants grow like $\log n$). By Banach-Steinhaus, there must exist $f \in C[-\pi, \pi]$ whose Fourier series diverges at some point. This is du Bois-Reymond's theorem, proved painlessly from the abstract principle.

## Open mapping theorem

**Theorem (Banach, Schauder).** If $X, Y$ are Banach spaces and $T: X \to Y$ is a surjective bounded linear operator, then $T$ is an open map (sends open sets to open sets).

**Corollary (Bounded inverse).** If $T: X \to Y$ is a bijective bounded linear operator between Banach spaces, then $T^{-1}$ is automatically bounded. You don't need to check continuity of the inverse — surjectivity + bijectivity + completeness force it.

*Proof sketch.* Must show $T(B_X(0,1))$ contains a ball around $0$ in $Y$. Since $T$ is surjective, $Y = \bigcup_n T(B_X(0, n)) = \bigcup_n n\, T(B_X(0,1))$. By Baire, the closure $\overline{T(B_X(0,1))}$ has nonempty interior. Translation/scaling gives: $\overline{T(B_X(0,1))} \supseteq B_Y(0, \delta)$ for some $\delta > 0$. Then a "geometric series" argument (iteratively approximating $y$ by images of elements in shrinking balls) shows $T(B_X(0,2)) \supseteq B_Y(0, \delta)$. $\square$

**Example 3.** The identity map $I: (C[0,1], \|\cdot\|_\infty) \to (C[0,1], \|\cdot\|_1)$ is bounded (since $\|f\|_1 \le \|f\|_\infty$), bijective, but its inverse is unbounded (take $f_n(t) = t^n$: $\|f_n\|_\infty = 1$ but $\|f_n\|_1 = 1/(n+1) \to 0$). This doesn't contradict the theorem because $(C[0,1], \|\cdot\|_1)$ is not Banach. Completeness of the codomain is essential.

## Closed graph theorem

**Theorem.** Let $X, Y$ be Banach spaces and $T: X \to Y$ linear. If the graph $\{(x, Tx) : x \in X\}$ is closed in $X \times Y$, then $T$ is bounded.

A closed graph means: whenever $x_n \to x$ and $Tx_n \to y$, then $y = Tx$. This is strictly weaker than continuity (which requires $Tx_n \to Tx$ whenever $x_n \to x$, without assuming the limit of $Tx_n$ exists).

*Proof.* The graph $G = \{(x, Tx)\}$ is a closed subspace of the Banach space $X \times Y$, hence Banach. The projection $\pi_1: G \to X$ given by $(x, Tx) \mapsto x$ is bounded and bijective. By the open mapping theorem, $\pi_1^{-1}: X \to G$ is bounded. Since $T = \pi_2 \circ \pi_1^{-1}$ (where $\pi_2$ is the bounded projection onto $Y$), $T$ is bounded. $\square$

**Example 4.** The differentiation operator $D: C^1[0,1] \to C[0,1]$ with $Df = f'$. Is the graph closed in $C[0,1] \times C[0,1]$ (both with sup norm)? If $f_n \to f$ uniformly and $f_n' \to g$ uniformly, then $f$ is differentiable with $f' = g$ (standard calculus theorem). So yes, the graph is closed. But $D$ is unbounded ($\|t^n\|_\infty = 1$ but $\|(t^n)'\|_\infty = n$). No contradiction: $C^1[0,1]$ with the sup norm is not a Banach space (it's not closed in $C[0,1]$). The theorem requires the domain to be Banach with its own topology.

## How they interconnect

The four theorems aren't independent:

- Closed graph $\Leftrightarrow$ open mapping (each implies the other quickly).
- Both use Baire, hence need completeness of both domain and codomain.
- Uniform boundedness uses Baire, needs completeness of the domain only.
- Hahn-Banach is algebraic/geometric — no completeness needed, but needs Zorn's lemma.

The pattern: completeness converts qualitative information (pointwise bounds, surjectivity, graph closure) into quantitative information (uniform bounds, open maps, operator bounds).

## What's next

Armed with these tools, we can tackle spectral theory — the study of "eigenvalues" for operators on infinite-dimensional spaces. Compact operators have the cleanest spectral theory, closest to the finite-dimensional case, and that's where we'll start.
