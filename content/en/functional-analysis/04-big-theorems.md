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


![The four pillars of functional analysis built on Baire category](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-big-theorems/fa_fig4_big_theorems.png)

## Hahn-Banach theorem

**Theorem (Hahn-Banach, extension form).** Let $X$ be a normed space, $M \subseteq X$ a subspace, and $f: M \to \mathbb{R}$ a bounded linear functional with $\|f\| = C$. Then there exists an extension $F: X \to \mathbb{R}$ with $F|_M = f$ and $\|F\| = C$.

The complex version works too: extend real and imaginary parts separately using $\text{Im}\,f(x) = -\text{Re}\,f(ix)$.

*Proof idea.* Extend one dimension at a time. If $M \subsetneq X$, pick $x_0 \notin M$ and define $F$ on $M + \mathbb{R}x_0$ by choosing $F(x_0) = c$. The constraint $|F(m + tx_0)| \le C\|m + tx_0\|$ forces $c$ to lie in a closed interval (computed from the values $f(m)/t$ and norms). The interval is nonempty by the existing bound on $f$. Iterate transfinitely (Zorn's lemma). $\square$

**Consequences that matter:**

1. **Separation of points.** For any $x \ne 0$ in a normed space, there exists $f \in X^*$ with $f(x) = \|x\|$ and $\|f\| = 1$. The dual space separates points — it's rich enough to "see" every nonzero vector.

2. **$X$ embeds in $X^{**}$.** The canonical map $J: X \to X^{**}$ defined by $(Jx)(f) = f(x)$ is an isometric embedding. If $J$ is surjective, $X$ is called **reflexive**. The spaces $\ell^p$ for $1 < p < \infty$ are reflexive; $\ell^1$ and $\ell^\infty$ are not.

3. **Existence of functionals.** Given a closed subspace $M$ and $x_0 \notin M$, there exists $f \in X^*$ with $f|_M = 0$ and $f(x_0) \ne 0$. Dual spaces detect the difference between subspaces.

**Hahn-Banach separation form.** If $A$ and $B$ are disjoint convex sets in a normed space with $A$ open, there exists $f \in X^*$ and $\alpha \in \mathbb{R}$ with $\text{Re}\,f(a) < \alpha \le \text{Re}\,f(b)$ for all $a \in A$, $b \in B$. A hyperplane separates them. This geometric form is the foundation of convex optimization and duality theory in mathematical programming.

**Example 1.** On $c_0$ (sequences converging to 0), the functional $f(x) = x_1$ defined on the subspace $\text{span}(e_1)$ extends to the evaluation functional on all of $c_0$, with norm 1. In fact, $(c_0)^* \cong \ell^1$ — every functional on $c_0$ is represented by an absolutely summable sequence.

**A warning: non-uniqueness of extensions.** The Hahn-Banach extension is generally not unique. On $\ell^\infty$, the functional $f(x) = \lim_{n} x_n$ defined on the subspace $c$ of convergent sequences has norm 1. It extends to a functional on all of $\ell^\infty$ (a "Banach limit"), but there are uncountably many such extensions — you cannot write down an explicit one without the axiom of choice. This non-constructive aspect is intrinsic: in models of set theory without choice, Hahn-Banach can fail.

## Uniform boundedness principle (Banach-Steinhaus)

**Theorem.** Let $X$ be a Banach space, $Y$ a normed space, and $\{T_\alpha\}_{\alpha \in A} \subseteq B(X, Y)$ a family of bounded operators. If $\sup_\alpha \|T_\alpha x\| < \infty$ for every $x \in X$, then $\sup_\alpha \|T_\alpha\| < \infty$.

Pointwise boundedness implies uniform boundedness. This is genuinely surprising — individual vectors tell you about the worst case over all vectors.

*Proof sketch (via Baire).* Define $F_n = \{x \in X : \sup_\alpha \|T_\alpha x\| \le n\}$. Each $F_n$ is closed (as an intersection of closed sets). By hypothesis, $X = \bigcup F_n$. Since $X$ is Banach (hence Baire), some $F_N$ has nonempty interior: there exist $x_0$ and $r > 0$ with $B(x_0, r) \subseteq F_N$. For $\|x\| \le 1$: $T_\alpha x = T_\alpha(x_0 + rx)/(r) - T_\alpha(x_0)/(r)$... more carefully, $\|T_\alpha(rx)\| = \|T_\alpha(x_0 + rx) - T_\alpha(x_0)\| \le 2N$. Hence $\sup_\alpha \|T_\alpha\| \le 2N/r$. $\square$

**Example 2: Divergence of Fourier series.** The partial sum operators $S_n f = \sum_{k=-n}^n \hat{f}(k) e^{ikt}$ on $C[-\pi, \pi]$ satisfy $\|S_n\| \to \infty$ (the Lebesgue constants grow like $\log n$). By Banach-Steinhaus, there must exist $f \in C[-\pi, \pi]$ whose Fourier series diverges at some point. This is du Bois-Reymond's theorem, proved painlessly from the abstract principle.

**Example 3: Condensation of singularities.** If $\{f_n\}$ is a sequence of continuous linear functionals on a Banach space $X$ with $\sup_n |f_n(x)| = \infty$ for each $x$ in a dense subset, then $\sup_n |f_n(x)| = \infty$ for "most" $x$ (a residual set). The singular behavior isn't isolated — it's generic. This is why pointwise convergence failures in Fourier analysis tend to be pervasive rather than exceptional.

## Open mapping theorem

**Theorem (Banach, Schauder).** If $X, Y$ are Banach spaces and $T: X \to Y$ is a surjective bounded linear operator, then $T$ is an open map (sends open sets to open sets).

*Proof sketch.* Must show $T(B_X(0,1))$ contains a ball around $0$ in $Y$. Since $T$ is surjective, $Y = \bigcup_n T(B_X(0, n)) = \bigcup_n n\, T(B_X(0,1))$. By Baire, the closure $\overline{T(B_X(0,1))}$ has nonempty interior. Translation/scaling gives: $\overline{T(B_X(0,1))} \supseteq B_Y(0, \delta)$ for some $\delta > 0$. Then a "geometric series" argument (iteratively approximating $y$ by images of elements in shrinking balls) shows $T(B_X(0,2)) \supseteq B_Y(0, \delta)$. $\square$

**Corollary (Bounded inverse theorem).** If $T: X \to Y$ is a bijective bounded linear operator between Banach spaces, then $T^{-1}$ is automatically bounded. You don't need to check continuity of the inverse — surjectivity + bijectivity + completeness force it.

This is remarkable. In general topology, a continuous bijection need not have a continuous inverse (think of the identity from a finer topology to a coarser one). The bounded inverse theorem says that between Banach spaces, the topology is so rigid that a bounded bijection must be a homeomorphism. It's a consequence of the open mapping theorem: if $T$ is open and bijective, then $T^{-1}$ is continuous.

**Corollary (Isomorphism by equivalence of norms).** If $\|\cdot\|_1$ and $\|\cdot\|_2$ are two complete norms on a vector space $X$ with $\|x\|_2 \le C\|x\|_1$ for all $x$ (one dominates the other), then they are equivalent: $\|x\|_1 \le C'\|x\|_2$ for some $C'$. Apply the bounded inverse theorem to the identity map from $(X, \|\cdot\|_1)$ to $(X, \|\cdot\|_2)$.

**Example 4: Why completeness is essential.** The identity map $I: (C[0,1], \|\cdot\|_\infty) \to (C[0,1], \|\cdot\|_1)$ is bounded (since $\|f\|_1 \le \|f\|_\infty$), bijective, but its inverse is unbounded (take $f_n(t) = t^n$: $\|f_n\|_\infty = 1$ but $\|f_n\|_1 = 1/(n+1) \to 0$). This doesn't contradict the theorem because $(C[0,1], \|\cdot\|_1)$ is not Banach.

## Closed graph theorem

**Theorem.** Let $X, Y$ be Banach spaces and $T: X \to Y$ linear. If the graph $\{(x, Tx) : x \in X\}$ is closed in $X \times Y$, then $T$ is bounded.

A closed graph means: whenever $x_n \to x$ and $Tx_n \to y$, then $y = Tx$. This is strictly weaker than continuity (which requires $Tx_n \to Tx$ whenever $x_n \to x$, without assuming the limit of $Tx_n$ exists).

*Proof.* The graph $G = \{(x, Tx)\}$ is a closed subspace of the Banach space $X \times Y$, hence Banach. The projection $\pi_1: G \to X$ given by $(x, Tx) \mapsto x$ is bounded and bijective. By the open mapping theorem, $\pi_1^{-1}: X \to G$ is bounded. Since $T = \pi_2 \circ \pi_1^{-1}$ (where $\pi_2$ is the bounded projection onto $Y$), $T$ is bounded. $\square$

**Example 5: Differentiation is closed but unbounded.** The differentiation operator $D: C^1[0,1] \to C[0,1]$ with $Df = f'$. If $f_n \to f$ uniformly and $f_n' \to g$ uniformly, then $f' = g$ (standard calculus theorem). So the graph is closed in $C[0,1] \times C[0,1]$. But $D$ is unbounded ($\|t^n\|_\infty = 1$ but $\|(t^n)'\|_\infty = n$). No contradiction: $C^1[0,1]$ with the sup norm is not a Banach space (it's not closed in $C[0,1]$). The theorem requires the domain to be complete.

## How they interconnect

The four theorems aren't independent:

- Closed graph $\Leftrightarrow$ open mapping (each implies the other quickly).
- Both use Baire, hence need completeness of both domain and codomain.
- Uniform boundedness uses Baire, needs completeness of the domain only.
- Hahn-Banach is algebraic/geometric — no completeness needed, but needs Zorn's lemma.

The pattern: completeness converts qualitative information (pointwise bounds, surjectivity, graph closure) into quantitative information (uniform bounds, open maps, operator bounds). This is perhaps the deepest theme in functional analysis — complete spaces are rigid, and rigidity gives you theorems for free.

## Applications: why practitioners care

**PDE existence via Hahn-Banach.** The standard proof that the Poisson equation $-\Delta u = f$ has a weak solution in $H^1_0(\Omega)$ uses the Riesz representation theorem (a consequence of orthogonal projection, which is the Hilbert space version of Hahn-Banach geometry). More generally, existence results for linear PDE often boil down to: construct a functional on a subspace, extend it, then represent it.

**Numerical stability via uniform boundedness.** In numerical analysis, if a sequence of approximation operators $(P_n)$ (interpolation, quadrature, finite elements) satisfies $\sup_n \|P_n\| = \infty$, then Banach-Steinhaus guarantees a "bad" input where the approximations diverge. This is the Lax equivalence theorem's backbone: for a consistent numerical scheme, stability ($\sup_n \|P_n\| < \infty$) is equivalent to convergence.

**Automatic continuity in Banach algebras.** Every homomorphism from a Banach algebra onto a semisimple Banach algebra is automatically continuous (Johnson's theorem). The proof uses the closed graph theorem crucially — you verify that the graph of the homomorphism is closed (using semisimplicity to control the separating ideal), then the theorem gives boundedness for free. No explicit norm estimate is needed.

**Inverse problems.** The bounded inverse theorem tells you that if an operator equation $Tx = y$ has a unique solution for each $y$ (bijectivity), and $T$ is bounded, then the solution depends continuously on the data — the inverse problem is well-posed. When the inverse is unbounded (necessarily meaning the codomain isn't complete or $T$ isn't surjective), the problem is ill-posed and regularization is needed. This classification drives the entire field of inverse problems in imaging, geophysics, and medical tomography.

## What's next

Armed with these tools, we can tackle spectral theory — the study of "eigenvalues" for operators on infinite-dimensional spaces. Compact operators have the cleanest spectral theory, closest to the finite-dimensional case, and that's where we'll start.

---

*This is Part 4 of the [Functional Analysis](/en/series/functional-analysis/) series (6 articles).*

*Previous: [Part 3 — Bounded Linear Operators](/en/functional-analysis/03-bounded-operators/)*

*Next: [Part 5 — Spectral Theory](/en/functional-analysis/05-spectral-theory/)*
