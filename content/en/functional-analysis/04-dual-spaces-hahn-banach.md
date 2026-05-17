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

In the previous article, the Riesz Representation Theorem gave us a complete picture of the dual of a Hilbert space: every continuous linear functional is an inner product, and the dual is isometrically isomorphic to the space itself. This is beautiful, but also special. Most Banach spaces are not Hilbert spaces, and without an inner product, even the question "does there exist a non-zero continuous linear functional?" becomes non-trivial.

The **Hahn-Banach theorem** is the tool that resolves this. It guarantees that continuous linear functionals can be extended from subspaces to the whole space without increasing their norms, and it implies that the dual space of any normed space is **rich enough** to separate points. From this single theorem flows the entire theory of duality, the study of weak topologies, and a host of existence results that pervade modern analysis.

---

## The Dual Space $X^*$

### Definition

Let $X$ be a normed space over $\mathbb{F}$ (where $\mathbb{F} = \mathbb{R}$ or $\mathbb{C}$). The **dual space** $X^*$ is the set of all continuous (equivalently, bounded) linear functionals $\varphi: X \to \mathbb{F}$, equipped with the operator norm:

![Hahn-Banach separation in a normed space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/04-dual-spaces-hahn-banach/fa_fig2_projection.png)


$$\|\varphi\|_{X^*} = \sup_{\|x\| \leq 1} |\varphi(x)|.$$

Since $\mathbb{F}$ is complete, $X^*$ is always a Banach space — even when $X$ itself is not complete. This is an elementary but important point: the dual of any normed space is automatically a Banach space.

The equivalence between continuity and boundedness for linear functionals deserves emphasis. A linear functional $\varphi: X \to \mathbb{F}$ is continuous if and only if $\ker \varphi$ is closed — and a non-zero linear functional with a dense kernel cannot be continuous. In infinite dimensions, there always exist discontinuous linear functionals (their construction requires the axiom of choice), but these pathological objects play no role in analysis. The dual space $X^*$ collects only the well-behaved ones.

### Why the dual matters

The dual space is the stage on which duality theory plays out. In finite dimensions, $X^*$ is isomorphic to $X$ and the distinction is cosmetic. In infinite dimensions, however, $X$ and $X^*$ can differ dramatically — and understanding the dual is the key to understanding the geometry of the original space. Weak convergence, reflexivity, the bidual, and all of weak-star compactness depend on the relationship between $X$ and $X^*$.

### Examples of dual spaces

**Example 1 (Finite dimensions).** If $X = \mathbb{R}^n$ with the Euclidean norm, then $X^* \cong \mathbb{R}^n$ via $\varphi \leftrightarrow a$ where $\varphi(x) = \sum a_i x_i$. More generally, $(\mathbb{R}^n, \|\cdot\|_p)^* \cong (\mathbb{R}^n, \|\cdot\|_q)$ where $\frac{1}{p} + \frac{1}{q} = 1$.

**Example 2 (The duals of $\ell^p$).** For $1 \leq p < \infty$, the dual $(\ell^p)^*$ is isometrically isomorphic to $\ell^q$ where $\frac{1}{p} + \frac{1}{q} = 1$ (with the convention $q = \infty$ when $p = 1$). The isomorphism sends $y = (y_n) \in \ell^q$ to the functional

$$\varphi_y(x) = \sum_{n=1}^{\infty} x_n y_n.$$

That $\|\varphi_y\| = \|y\|_q$ follows from Holder's inequality (for "$\leq$") and a careful choice of $x$ (for "$\geq$"). Specifically, for the reverse inequality, take $x_n = |y_n|^{q-1} \operatorname{sgn}(y_n) / \|y\|_q^{q/p}$ when $1 < p < \infty$. Then $\|x\|_p = 1$ and $\varphi_y(x) = \|y\|_q$.

That every functional on $\ell^p$ arises this way (surjectivity) is the deeper part. Here is a sketch for $1 < p < \infty$: given $\varphi \in (\ell^p)^*$, define $y_n = \varphi(e_n)$ where $e_n$ is the $n$-th standard basis vector. For any finite $N$, take $x = \sum_{n=1}^N |y_n|^{q-2}\overline{y_n}\, e_n / \left(\sum_{n=1}^N |y_n|^q\right)^{1/p}$ (assuming the sum is non-zero). Then $\|x\|_p = 1$ and $\varphi(x) = \left(\sum_{n=1}^N |y_n|^q\right)^{1/q}$. Since $|\varphi(x)| \leq \|\varphi\|$, we get $\left(\sum_{n=1}^N |y_n|^q\right)^{1/q} \leq \|\varphi\|$ for all $N$, proving $y \in \ell^q$. Continuity and density of finite sequences then show $\varphi = \varphi_y$.

**Example 3 (The dual of $C[a,b]$).** The Riesz-Markov representation theorem identifies $C[a,b]^*$ with the space of signed Borel measures of bounded variation on $[a,b]$: every $\varphi \in C[a,b]^*$ is represented by

$$\varphi(f) = \int_a^b f\, d\mu$$

for a unique regular signed measure $\mu$, with $\|\varphi\| = |\mu|([a,b])$. This is far richer than point evaluations — delta measures, absolutely continuous measures, and singular measures all contribute. For instance, the point evaluation $\varphi(f) = f(c)$ corresponds to the Dirac measure $\delta_c$, and the integral functional $\varphi(f) = \int_a^b f(t) w(t)\, dt$ corresponds to the absolutely continuous measure $d\mu = w\, dt$.

### The dual of $\ell^\infty$ and a warning

The dual of $\ell^\infty$ is **not** $\ell^1$. Rather, $(\ell^\infty)^*$ is the space of finitely additive signed measures on $\mathbb{N}$ (or equivalently, the space $\text{ba}(\mathbb{N})$ of bounded finitely additive set functions). This space is much larger than $\ell^1$ and contains exotic objects like **Banach limits** — positive linear functionals $L$ on $\ell^\infty$ with $L(Sx) = L(x)$ (where $S$ is the left shift) and $\liminf x_n \leq L(x) \leq \limsup x_n$.

The existence of Banach limits cannot be proved constructively — it requires Hahn-Banach (and hence the axiom of choice). This example illustrates how the Hahn-Banach theorem produces objects that we know must exist but cannot write down explicitly. It also serves as a warning that dual spaces can be significantly more complex than the original space — and that the "$(\ell^p)^* = \ell^q$" pattern breaks down at $p = \infty$.

---

## The Hahn-Banach Theorem: Analytic Form

### Historical context

The theorem was proved independently by Hans Hahn (1927) and Stefan Banach (1929), though special cases were known earlier (e.g., Eduard Helly's 1912 theorem on moment problems). It is one of the three pillars of functional analysis (alongside the Uniform Boundedness Principle and the Open Mapping Theorem), and unlike the other two, it does not require completeness — it holds for any normed space, not just Banach spaces.

The Hahn-Banach theorem is also notable for its axiomatic status: in the real case, it can be proved using only Zorn's lemma (equivalently, the axiom of choice), but it does *not* imply the axiom of choice. In fact, the Hahn-Banach theorem is strictly weaker than the axiom of choice — it follows from the ultrafilter lemma, which is itself weaker than full AC. This makes Hahn-Banach accessible in constructive mathematics contexts where full AC is unavailable.

### Statement

**Theorem (Hahn-Banach, real version).** Let $X$ be a real vector space and $p: X \to \mathbb{R}$ a **sublinear functional** (i.e., $p(\lambda x) = \lambda p(x)$ for $\lambda \geq 0$ and $p(x + y) \leq p(x) + p(y)$). Let $M \subseteq X$ be a subspace and $f: M \to \mathbb{R}$ a linear functional satisfying $f(x) \leq p(x)$ for all $x \in M$.

Then there exists a linear functional $F: X \to \mathbb{R}$ extending $f$ (i.e., $F|_M = f$) such that $F(x) \leq p(x)$ for all $x \in X$.

The **normed space version** is an immediate corollary: if $X$ is a normed space, $M$ a subspace, and $f \in M^*$, then $f$ extends to $F \in X^*$ with $\|F\| = \|f\|$ (take $p(x) = \|f\| \cdot \|x\|$).

### Proof via Zorn's lemma

The proof has two conceptual steps: a one-dimensional extension, then a maximal extension via Zorn.

**Step 1: One-dimensional extension.** Suppose $M$ is a proper subspace and $x_0 \in X \setminus M$. We want to extend $f$ to $M_1 = M + \mathbb{R} x_0$ by defining $F(m + t x_0) = f(m) + t\alpha$ for some $\alpha \in \mathbb{R}$, while maintaining $F \leq p$.

For $t > 0$: $f(m) + t\alpha \leq p(m + tx_0)$, i.e., $\alpha \leq \frac{p(m + tx_0) - f(m)}{t} = p(m/t + x_0) - f(m/t)$, so $\alpha \leq \inf_{u \in M} [p(u + x_0) - f(u)]$.

For $t < 0$: similarly, $\alpha \geq \sup_{v \in M} [f(v) - p(v - x_0)]$.

The key inequality is: for all $u, v \in M$,

$$f(v) - p(v - x_0) = f(u) + f(v - u) - p(v - x_0) \leq f(u) + p(u - v + v - x_0) - p(v - x_0)$$

Wait — more directly: $f(v) - f(u) = f(v - u) \leq p(v - u) = p((v - x_0) + (x_0 - u)) \leq p(v - x_0) + p(-(u - x_0))$, so

$$f(v) - p(v - x_0) \leq f(u) + p(u + x_0) - 2f(u) + f(u) = f(u) + p(u+x_0) - f(u)?$$

Let me state this more cleanly. For $u, v \in M$:

$$f(u) + f(v) = f(u + v) \leq p(u + v) = p((u + x_0) + (v - x_0)) \leq p(u + x_0) + p(v - x_0).$$

Therefore $f(v) - p(v - x_0) \leq p(u + x_0) - f(u)$, establishing

$$\sup_{v \in M} [f(v) - p(v - x_0)] \leq \inf_{u \in M} [p(u + x_0) - f(u)].$$

Any $\alpha$ between these bounds gives a valid one-dimensional extension. Note that the bounds are real numbers (not $\pm\infty$), so there is always room to choose $\alpha$.

**Step 2: Zorn's lemma.** Consider the partially ordered set of all pairs $(N, g)$ where $N$ is a subspace containing $M$, $g: N \to \mathbb{R}$ is linear, $g|_M = f$, and $g \leq p$ on $N$. Order by extension: $(N_1, g_1) \leq (N_2, g_2)$ if $N_1 \subseteq N_2$ and $g_2|_{N_1} = g_1$. Every chain has an upper bound (take the union of domains and the consistent functional — consistency is guaranteed by the chain condition). By Zorn's lemma, a maximal element $(N_0, g_0)$ exists. If $N_0 \neq X$, Step 1 produces a strict extension, contradicting maximality. Therefore $N_0 = X$. $\blacksquare$

**Remark on the proof structure.** The proof has a characteristic "one step + Zorn" pattern that appears repeatedly in abstract analysis. The hard work is in the one-dimensional extension (the algebraic core); Zorn's lemma then bootstraps this to the full result. This same pattern appears in the proof that every vector space has a basis, that every ideal is contained in a maximal ideal, and many other existence results in algebra and analysis.

### The complex case

The complex version follows from the real one via a trick due to Bohnenblust and Sobczyk. If $f: M \to \mathbb{C}$ is $\mathbb{C}$-linear, write $f = u + iv$ where $u = \operatorname{Re} f$. Then $u$ is $\mathbb{R}$-linear, and one recovers $f$ from $u$ via $f(x) = u(x) - iu(ix)$. Extend $u$ by the real Hahn-Banach, then reconstruct the complex extension. The norm is preserved because $|f(x)| = f(x) \cdot e^{-i\theta}$ for some $\theta$, and $|f(x)| = u(e^{-i\theta} x) \leq p(e^{-i\theta}x) = p(x)$.

---

## Consequences: Separation, Extension, and Norm-Witnessing Functionals

The Hahn-Banach theorem has three immediate corollaries that are used constantly throughout functional analysis.

### Corollary 1: Functionals that witness the norm

**For every $x_0 \in X$, there exists $\varphi \in X^*$ with $\|\varphi\| = 1$ and $\varphi(x_0) = \|x_0\|$.**

*Proof.* Define $f: \mathbb{F} x_0 \to \mathbb{F}$ by $f(\lambda x_0) = \lambda \|x_0\|$. Then $\|f\| = 1$ on the one-dimensional subspace $\operatorname{span}\{x_0\}$. Extend by Hahn-Banach. $\blacksquare$

This seemingly simple result has a profound consequence: **$X^*$ separates points of $X$**. If $x \neq y$, apply the corollary to $x_0 = x - y$ to get a functional that distinguishes them. Without Hahn-Banach, we would have no guarantee that enough continuous linear functionals exist — and indeed, in some exotic topological vector spaces (that are not locally convex), the dual space can be trivial ($X^* = \{0\}$).

**Example: witnessing the norm in $\ell^1$.** Take $x_0 = (1, -1/2, 1/3, -1/4, \ldots) \in \ell^1$ with $\|x_0\|_1 = \sum_{n=1}^\infty 1/n = \infty$ — wait, this does not converge. Let us instead take $x_0 = (1, 0, 0, \ldots)$. The norm-witnessing functional is $\varphi(x) = x_1$, which corresponds to $y = (1, 0, 0, \ldots) \in \ell^\infty$ under the identification $(\ell^1)^* = \ell^\infty$. Indeed $\|\varphi\| = 1$ and $\varphi(x_0) = 1 = \|x_0\|_1$. For a more interesting example, take $x_0 = (1/2, 1/4, 1/8, \ldots) \in \ell^1$ with $\|x_0\|_1 = 1$. The witnessing functional is $\varphi_y$ with $y = (1, 1, 1, \ldots) \in \ell^\infty$: $\varphi_y(x_0) = \sum 1/2^n = 1 = \|x_0\|_1$ and $\|y\|_\infty = 1$.

### Corollary 2: The norm via duality

$$\|x\| = \sup_{\varphi \in X^*, \|\varphi\| \leq 1} |\varphi(x)|.$$

The "$\leq$" direction is trivial (from the definition of operator norm). The "$\geq$" direction follows from Corollary 1: the norm-witnessing functional achieves the supremum.

This formula is the starting point for the **weak topology** on $X$: the coarsest topology making all $\varphi \in X^*$ continuous. The Hahn-Banach theorem guarantees this topology is Hausdorff.

### Corollary 3: Extension of functionals

If $M$ is a closed subspace of $X$ and $f \in M^*$, then $f$ extends to $F \in X^*$ with $\|F\| = \|f\|$. This is the literal content of the normed-space Hahn-Banach theorem, and it is essential for constructing functionals with prescribed behavior on subspaces.

**Example 4 (Constructing a functional with prescribed values).** In $C[0,1]$, let $M = \{f : f(0) = f(1)\}$. Define $\varphi: M \to \mathbb{R}$ by $\varphi(f) = f(0)$. This is bounded with $\|\varphi\|_{M^*} = 1$. By Hahn-Banach, $\varphi$ extends to a functional $\Phi \in C[0,1]^*$ with $\|\Phi\| = 1$. By the Riesz-Markov theorem, $\Phi$ is represented by a measure $\mu$ on $[0,1]$ with $|\mu|([0,1]) = 1$.

---

## Geometric Hahn-Banach: Separating Convex Sets

The analytic Hahn-Banach theorem has a geometric counterpart that is often more intuitive and equally powerful.

### Separation of convex sets

**Theorem (Geometric Hahn-Banach).** Let $X$ be a real normed space, $A$ and $B$ non-empty convex subsets of $X$ with $A \cap B = \emptyset$.

1. **Separation:** If $A$ is open, there exist $\varphi \in X^*$ and $\alpha \in \mathbb{R}$ such that $\varphi(a) < \alpha \leq \varphi(b)$ for all $a \in A$, $b \in B$.
2. **Strict separation:** If $A$ is compact, $B$ is closed, and $X$ is locally convex, then there exist $\varphi \in X^*$ and $\alpha_1 < \alpha_2$ such that $\varphi(a) \leq \alpha_1 < \alpha_2 \leq \varphi(b)$ for all $a \in A$, $b \in B$.

The proof of (1) reduces to separating the origin from the open convex set $A - B$, which is accomplished by constructing the **Minkowski functional** $p_{A-B}$ and applying the analytic Hahn-Banach theorem with $p = p_{A-B}$ and $f$ defined on a one-dimensional subspace.

More precisely, the **Minkowski functional** (or **gauge**) of a convex set $C$ containing the origin is defined as

$$p_C(x) = \inf\{t > 0 : x \in tC\}.$$

When $C$ is open and convex with $0 \in C$, $p_C$ is a sublinear functional, and $C = \{x : p_C(x) < 1\}$. To separate $0$ from the open convex set $U = A - B$ (which does not contain $0$ since $A \cap B = \emptyset$), pick any $x_0 \in U$ and define $f$ on $\mathbb{R} x_0$ by $f(tx_0) = t$. Then $f(x_0) = 1 > p_U(x_0)$ for suitable normalization, and extending $f$ via Hahn-Banach gives the separating functional.

### Geometric intuition

In $\mathbb{R}^n$, separation by a hyperplane is visually obvious: if two convex bodies do not overlap, you can slide a flat plane between them. The geometric Hahn-Banach theorem says this works in infinite dimensions too, even though infinite-dimensional geometry can be highly counterintuitive (e.g., the unit ball is not compact, closed bounded convex sets need not have extreme points in general).

**Example 5 (Supporting hyperplane).** Let $C \subseteq X$ be a closed convex set and $x_0 \notin C$. By geometric Hahn-Banach, there exists $\varphi \in X^*$ and $\alpha$ such that $\varphi(x_0) > \alpha \geq \varphi(c)$ for all $c \in C$. The hyperplane $\{x : \varphi(x) = \alpha\}$ separates $x_0$ from $C$. This is the infinite-dimensional version of the **supporting hyperplane theorem**.

---

## Duals of Classical Spaces: $(\ell^p)^* = \ell^q$ and $C[a,b]^* = $ Measures

### The Riesz representation for $L^p$ spaces

For $1 \leq p < \infty$ and a measure space $(\Omega, \Sigma, \mu)$, the dual of $L^p(\mu)$ is isometrically isomorphic to $L^q(\mu)$ where $\frac{1}{p} + \frac{1}{q} = 1$:

$$(L^p(\mu))^* \cong L^q(\mu).$$

The proof for $1 < p < \infty$ uses the Radon-Nikodym theorem. Given $\varphi \in (L^p)^*$, define a set function $\nu(A) = \varphi(\mathbf{1}_A)$ for measurable sets $A$ with finite measure. Then $\nu$ is a signed measure that is absolutely continuous with respect to $\mu$ (because $\|\mathbf{1}_A\|_p = \mu(A)^{1/p} \to 0$ implies $\nu(A) = \varphi(\mathbf{1}_A) \to 0$). By the Radon-Nikodym theorem, $d\nu = g\, d\mu$ for some measurable $g$. The delicate part is showing that $g \in L^q$ and that $\varphi(f) = \int fg\, d\mu$ for all $f \in L^p$, not just for simple functions. This is accomplished by approximating an arbitrary $f \in L^p$ by simple functions and using the continuity of $\varphi$.

The case $p = 1$ is subtler: $(L^1)^* = L^\infty$ requires the additional hypothesis that the measure space is $\sigma$-finite. Without $\sigma$-finiteness, the Radon-Nikodym theorem may fail, and the identification breaks down.

### The Riesz-Markov theorem

For the space $C(K)$ of continuous functions on a compact Hausdorff space $K$, the dual is the space of regular Borel measures:

$$C(K)^* \cong \mathcal{M}(K),$$

where $\mathcal{M}(K)$ is the space of signed regular Borel measures with the total variation norm. This result, the **Riesz-Markov-Kakutani representation theorem**, is deeper than the $L^p$ duality and connects functional analysis to measure theory in a fundamental way. It tells us that the "natural" notion of generalized function on a compact space is not a distribution (as it would be for Sobolev spaces) but a measure — a perspective that is central to probability theory, where probability measures on compact spaces are exactly the positive norm-one elements of $C(K)^*$.

**Worked Example: Computing the dual norm.** Consider $\varphi: C[0,1] \to \mathbb{R}$ defined by $\varphi(f) = \int_0^{1/2} f(t)\, dt - \int_{1/2}^{1} f(t)\, dt$. The representing measure is $\mu = \mathbf{1}_{[0,1/2]} \cdot \lambda - \mathbf{1}_{(1/2,1]} \cdot \lambda$ where $\lambda$ is Lebesgue measure. The total variation is $|\mu|([0,1]) = \int_0^{1/2} 1\, dt + \int_{1/2}^1 1\, dt = 1$, so $\|\varphi\| = 1$.

To verify directly without invoking the Riesz-Markov theorem: $|\varphi(f)| \leq \int_0^{1/2} |f| + \int_{1/2}^1 |f| \leq \|f\|_\infty$, so $\|\varphi\| \leq 1$. For the reverse, take $f_\epsilon$ to be the continuous function that equals $+1$ on $[0, 1/2 - \epsilon]$, equals $-1$ on $[1/2 + \epsilon, 1]$, and is linear on $[1/2 - \epsilon, 1/2 + \epsilon]$. Then $\|f_\epsilon\|_\infty = 1$ and $\varphi(f_\epsilon) = 1 - 2\epsilon \to 1$ as $\epsilon \to 0$. The supremum is $1$ but is not attained by any continuous function (only by the discontinuous $\operatorname{sgn}(1/2 - t)$) — a concrete instance of the phenomenon that suprema over non-compact sets need not be achieved.

---

## Reflexive Spaces and the Bidual

### The canonical embedding

For any normed space $X$, there is a natural map $J: X \to X^{**}$ (the **bidual** or **double dual**) defined by:

$$J(x)(\varphi) = \varphi(x) \quad \text{for } \varphi \in X^*.$$

This map is called the **canonical embedding**. It is always:

- **Linear:** $J(\alpha x + \beta y) = \alpha J(x) + \beta J(y)$.
- **Isometric:** $\|J(x)\|_{X^{**}} = \|x\|_X$ (this is exactly Corollary 2 from the Hahn-Banach consequences).

In particular, $J$ is injective, so $X$ can be identified with a closed subspace of $X^{**}$.

### Reflexive spaces

A Banach space $X$ is **reflexive** if the canonical embedding $J: X \to X^{**}$ is **surjective** — that is, every element of $X^{**}$ comes from evaluation at some point of $X$. In this case, $J$ is an isometric isomorphism and we write $X \cong X^{**}$.

**Reflexive examples:**
- Every Hilbert space is reflexive (immediate from the Riesz theorem: $H^* \cong H$, so $H^{**} \cong H^* \cong H$).
- $L^p(\mu)$ is reflexive for $1 < p < \infty$ (since $(L^p)^* = L^q$ and $(L^q)^* = L^p$).
- Every finite-dimensional space is reflexive.

**Non-reflexive examples:**
- $L^1(\mu)$ is not reflexive: $(L^1)^* = L^\infty$, but $(L^\infty)^*$ is much larger than $L^1$.
- $C[0,1]$ is not reflexive: its dual is the space of measures, whose dual is far larger than $C[0,1]$.
- $\ell^1$ is not reflexive: $(\ell^1)^* = \ell^\infty$, and $(\ell^\infty)^*$ properly contains $\ell^1$. The canonical embedding $J: \ell^1 \to (\ell^\infty)^*$ sends $x = (x_n)$ to the functional $\Phi \mapsto \sum x_n \Phi(e_n)$, but there are elements of $(\ell^\infty)^*$ (like Banach limits) that do not arise this way.
- $c_0$ (sequences converging to zero) is not reflexive: $(c_0)^* = \ell^1$, $(\ell^1)^* = \ell^\infty \neq c_0$. Note the interesting chain: $c_0 \subsetneq \ell^\infty = (c_0)^{**}$, so $c_0$ "grows" upon double dualization.

### James's theorem

A deep result of Robert C. James (1964) characterizes reflexivity in terms of the attainment of suprema:

**Theorem (James).** A Banach space $X$ is reflexive if and only if every continuous linear functional on $X$ attains its supremum on the closed unit ball.

One direction is easy: if $X$ is reflexive, the unit ball of $X$ is weakly compact (by the Banach-Alaoglu theorem applied to $X^{**}$ and pulled back), and continuous functions on compact sets attain their suprema. The converse is the hard part and was one of the most significant results in Banach space theory.

James's theorem provides a concrete geometric criterion for reflexivity and connects the abstract notion of the bidual to the tangible question of whether optimization problems have solutions.

**Example 6: $c_0$ is not reflexive, witnessed by a functional.** Consider $c_0 = \{x \in \ell^\infty : x_n \to 0\}$ with the sup norm, and the functional $\varphi \in (c_0)^* \cong \ell^1$ defined by $\varphi(x) = \sum_{n=1}^\infty x_n / 2^n$. Then $\|\varphi\| = \sum 1/2^n = 1$. But $\sup_{\|x\|_\infty \leq 1, x \in c_0} \varphi(x) = 1$ is not attained: any $x \in c_0$ with $\|x\|_\infty \leq 1$ satisfies $\varphi(x) = \sum x_n/2^n < \sum 1/2^n = 1$ unless $x_n = 1$ for all $n$, but the constant sequence $(1,1,1,\ldots) \notin c_0$. By James's theorem, this non-attainment witnesses the non-reflexivity of $c_0$.

### The bidual as a completion of duality

The chain $X \hookrightarrow X^{**} \hookrightarrow X^{****} \hookrightarrow \cdots$ stabilizes at the first step for reflexive spaces. For non-reflexive spaces, each embedding is proper, and the transfinite iteration of duality produces ever-larger spaces. This hierarchy is central to the classification theory of Banach spaces.

### A preview: weak and weak-star topologies

The dual space also gives rise to important topologies beyond the norm topology:

- The **weak topology** on $X$ is the coarsest topology making every $\varphi \in X^*$ continuous. A net $(x_\alpha)$ converges weakly to $x$ if $\varphi(x_\alpha) \to \varphi(x)$ for all $\varphi \in X^*$. Weak convergence is weaker than norm convergence: in $\ell^2$, the standard basis vectors $e_n \rightharpoonup 0$ weakly (since $\langle e_n, y \rangle = y_n \to 0$ for any $y \in \ell^2$), but $\|e_n\| = 1 \not\to 0$.

- The **weak-star topology** on $X^*$ is the coarsest topology making the evaluation maps $\varphi \mapsto \varphi(x)$ continuous for each $x \in X$. The **Banach-Alaoglu theorem** (proved in a later article) states that the closed unit ball of $X^*$ is compact in the weak-star topology. This is the source of most compactness arguments in infinite-dimensional analysis and is the reason dual spaces play such a central role.

Both topologies owe their existence and utility to the Hahn-Banach theorem: without enough functionals to separate points, neither topology would be Hausdorff, and the entire framework would collapse.

---

## What's Next

The Hahn-Banach theorem told us that continuous linear functionals exist in abundance. But what can we say about **families** of bounded linear operators? The next article addresses the three pillars of Banach space theory that govern the global behavior of operators:

- The **Uniform Boundedness Principle** (Banach-Steinhaus): a family of operators that is pointwise bounded is uniformly bounded.
- The **Open Mapping Theorem**: a surjective bounded operator between Banach spaces maps open sets to open sets.
- The **Closed Graph Theorem**: an everywhere-defined operator with a closed graph is automatically bounded.

These results transform functional analysis from the study of individual operators into a theory with powerful automatic regularity properties — properties that have no analogue in finite dimensions because they are trivially true there. Where the Hahn-Banach theorem guaranteed the existence of enough functionals, the next three theorems will constrain the behavior of families of operators in ways that are impossible to anticipate from finite-dimensional linear algebra alone.

---

*This is Part 4 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 3 — Hilbert Spaces](/en/functional-analysis/03-hilbert-spaces/)*

*Next: [Part 5 — Weak Topologies](/en/functional-analysis/05-weak-topologies/)*
