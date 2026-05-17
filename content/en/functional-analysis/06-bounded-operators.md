---
title: "Functional Analysis (6): Bounded Linear Operators and the Big Theorems"
date: 2021-10-11 09:00:00
tags:
  - functional-analysis
  - operators
  - open-mapping
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "The Uniform Boundedness Principle, Open Mapping Theorem, and Closed Graph Theorem — three consequences of completeness that constrain how operators can behave."
disableNunjucks: true
series_order: 6
series_total: 12
translationKey: "functional-analysis-6"
---

Every branch of mathematics studies not just objects but the maps between them. In linear algebra, the objects are vector spaces and the maps are linear transformations. In functional analysis, the objects are Banach spaces, and the natural maps — **bounded linear operators** — carry the full force of infinite-dimensional geometry.

What makes the theory of operators on Banach spaces remarkable is the degree to which **completeness alone** constrains the possible behavior of linear maps. Three theorems — the Uniform Boundedness Principle, the Open Mapping Theorem, and the Closed Graph Theorem — all flow from the Baire Category Theorem and together form the backbone of linear functional analysis. Without these results, the theory would be far less rigid, and many naturally defined operators would need laborious case-by-case analysis to show continuity.

This article proves all three, works through their most important applications, and exhibits the counterexamples showing why completeness is indispensable.

---

## The Space $B(X,Y)$ and the Operator Norm

**Definition.** Let $X$ and $Y$ be normed spaces. A linear map $T: X \to Y$ is **bounded** if there exists $C \geq 0$ such that $\|Tx\|_Y \leq C\|x\|_X$ for all $x \in X$. The **operator norm** is

$$\|T\| = \sup_{\|x\| \leq 1} \|Tx\| = \sup_{\|x\| = 1} \|Tx\| = \sup_{x \neq 0} \frac{\|Tx\|}{\|x\|}.$$

![Bounded vs unbounded operators on normed spaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/06-bounded-operators/fa_fig3_operators.png)


The space of all bounded linear operators from $X$ to $Y$ is denoted $B(X,Y)$ (or $\mathcal{L}(X,Y)$). When $Y = \mathbb{R}$ (or $\mathbb{C}$), this is just the dual space $X^*$.

**Proposition.** For a linear map $T: X \to Y$, the following are equivalent:
1. $T$ is bounded.
2. $T$ is continuous.
3. $T$ is continuous at a single point.

**Proof.** $(1 \Rightarrow 2)$: $\|Tx - Ty\| = \|T(x-y)\| \leq \|T\| \cdot \|x-y\|$, so $T$ is Lipschitz. $(2 \Rightarrow 3)$ is trivial. $(3 \Rightarrow 1)$: If $T$ is continuous at $0$, then $T^{-1}(B_Y(0,1))$ contains some ball $B_X(0, \delta)$. For any $x$ with $\|x\| = 1$, the vector $\delta x/2$ lies in this ball, so $\|T(\delta x/2)\| < 1$, giving $\|Tx\| < 2/\delta$. Thus $\|T\| \leq 2/\delta$. $\square$

**Theorem.** If $Y$ is a Banach space, then $B(X,Y)$ is a Banach space.

The proof is standard: if $(T_n)$ is Cauchy in operator norm, then $(T_n x)$ is Cauchy in $Y$ for each $x$; define $Tx = \lim T_n x$; verify linearity and boundedness; show $\|T_n - T\| \to 0$.

**Remark on the role of completeness of $Y$.** This result requires $Y$ to be complete. If $Y$ is merely a normed space, $B(X, Y)$ may not be complete. This is one reason why working with Banach spaces (rather than mere normed spaces) is essential: it ensures that the operator space itself is well-behaved.

**The algebra structure.** When $X = Y$, the space $B(X) = B(X, X)$ is not just a Banach space but a **Banach algebra**: it is closed under composition, $\|ST\| \leq \|S\|\|T\|$ (submultiplicativity), and the identity operator $I$ serves as the multiplicative identity with $\|I\| = 1$. The invertible operators form an open subset of $B(X)$: if $\|I - T\| < 1$, then $T$ is invertible with $T^{-1} = \sum_{n=0}^\infty (I - T)^n$ (the **Neumann series**). This algebraic structure will become central when we study spectral theory in Article 8.

**Example 1 (Multiplication operator).** Let $g \in L^\infty([0,1])$ and define $M_g: L^2([0,1]) \to L^2([0,1])$ by $M_g f = gf$. Then

$$\|M_g f\|_2 = \left(\int |g|^2 |f|^2\right)^{1/2} \leq \|g\|_\infty \|f\|_2,$$

so $\|M_g\| \leq \|g\|_\infty$. In fact $\|M_g\| = \|g\|_\infty$ (by choosing $f$ supported where $|g|$ is close to its essential supremum).

**Example 2 (Integral operator).** Let $K \in L^2([0,1]^2)$ and define $T: L^2([0,1]) \to L^2([0,1])$ by

$$(Tf)(x) = \int_0^1 K(x,y) f(y) \, dy.$$

By the Cauchy-Schwarz inequality,

$$|Tf(x)|^2 \leq \int_0^1 |K(x,y)|^2 \, dy \cdot \int_0^1 |f(y)|^2 \, dy,$$

so $\|Tf\|_2^2 \leq \|K\|_{L^2}^2 \|f\|_2^2$. Hence $T$ is bounded with $\|T\| \leq \|K\|_{L^2([0,1]^2)}$.

**Example 3 (Shift operator).** The right shift $S: \ell^2 \to \ell^2$ defined by $S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$ is a bounded operator with $\|S\| = 1$ (it is an isometry: $\|Sx\| = \|x\|$). Its adjoint (left shift) $S^*: \ell^2 \to \ell^2$ defined by $S^*(x_1, x_2, \ldots) = (x_2, x_3, \ldots)$ also has norm 1 but is not an isometry ($S^* e_1 = 0$). This pair will be important in spectral theory (Article 8).

---

## The Baire Category Theorem

Before proving the three big theorems, we need the key topological ingredient.

**Theorem (Baire Category Theorem).** In a complete metric space, the intersection of countably many dense open sets is dense. Equivalently, a complete metric space cannot be written as a countable union of nowhere dense sets.

A set is **nowhere dense** if its closure has empty interior. A set that is a countable union of nowhere dense sets is called **meager** (or of the first category). The Baire Category Theorem says a complete metric space is **non-meager** — it cannot be covered by countably many "thin" sets.

**Proof sketch.** Let $U_1, U_2, \ldots$ be dense open sets and $V$ a nonempty open set. We must show $V \cap \bigcap_n U_n \neq \emptyset$. Since $U_1$ is dense, $V \cap U_1$ is nonempty and open; choose a closed ball $\overline{B}(x_1, r_1) \subset V \cap U_1$ with $r_1 < 1$. Since $U_2$ is dense, the open ball $B(x_1, r_1) \cap U_2$ is nonempty; choose $\overline{B}(x_2, r_2) \subset B(x_1, r_1) \cap U_2$ with $r_2 < 1/2$. Continue inductively: $\overline{B}(x_{n+1}, r_{n+1}) \subset B(x_n, r_n) \cap U_{n+1}$, $r_n < 1/n$. The sequence $(x_n)$ is Cauchy (since $x_m \in B(x_n, r_n)$ for $m > n$ and $r_n \to 0$), and the limit $x = \lim x_n$ lies in $V \cap \bigcap_n U_n$. $\square$

---

## The Uniform Boundedness Principle (Banach-Steinhaus)

**Theorem (Uniform Boundedness Principle).** Let $X$ be a Banach space, $Y$ a normed space, and $\{T_\alpha\}_{\alpha \in A} \subset B(X,Y)$ a family of bounded linear operators. If

$$\sup_{\alpha \in A} \|T_\alpha x\| < \infty \quad \text{for every } x \in X,$$

then

$$\sup_{\alpha \in A} \|T_\alpha\| < \infty.$$

In other words, a family of operators that is **pointwise bounded** is automatically **uniformly bounded** in operator norm.

**Proof.** For each $n \in \mathbb{N}$, define the closed set

$$F_n = \{x \in X : \|T_\alpha x\| \leq n \text{ for all } \alpha \in A\} = \bigcap_{\alpha \in A} T_\alpha^{-1}(\overline{B}_Y(0, n)).$$

By hypothesis, $X = \bigcup_{n=1}^\infty F_n$. Since $X$ is a Banach space (complete metric space), the Baire Category Theorem implies that some $F_N$ has nonempty interior: there exist $x_0 \in X$ and $r > 0$ such that $B(x_0, r) \subset F_N$.

For any $x$ with $\|x\| \leq 1$, we have $x_0 + rx/2 \in B(x_0, r) \subset F_N$ and $x_0 \in F_N$. Therefore:

$$\|T_\alpha(rx/2)\| = \|T_\alpha(x_0 + rx/2) - T_\alpha(x_0)\| \leq \|T_\alpha(x_0 + rx/2)\| + \|T_\alpha(x_0)\| \leq 2N.$$

Hence $\|T_\alpha x\| \leq 4N/r$ for all $\alpha$ and all $\|x\| \leq 1$, giving $\sup_\alpha \|T_\alpha\| \leq 4N/r$. $\square$

**Application 1: Convergence of operator sequences.** If $T_n \in B(X,Y)$ with $X$ Banach, and $T_n x \to Tx$ for each $x \in X$, then:
- $\sup_n \|T_n\| < \infty$ (by UBP).
- $T$ is bounded with $\|T\| \leq \liminf_n \|T_n\|$.

This is remarkable: pointwise convergence of operators on a Banach space **automatically** gives a bounded limit.

**Application 2: The Condensation of Singularities.** The UBP can be used in "contrapositive mode" to show that pathologies happen **generically**. For example:

*There exists a continuous $2\pi$-periodic function whose Fourier series diverges at $0$.*

**Proof sketch.** The $n$-th partial sum of the Fourier series of $f$ at $0$ is given by $S_n(f) = \int_{-\pi}^{\pi} f(t) D_n(t) \, dt$, where $D_n$ is the Dirichlet kernel. This defines a bounded linear functional $S_n: C([-\pi, \pi]) \to \mathbb{R}$, and one can compute $\|S_n\| = \frac{1}{2\pi}\int_{-\pi}^{\pi} |D_n(t)| \, dt \sim \frac{4}{\pi^2} \log n \to \infty$. If $S_n(f)$ converged for every $f$, the UBP would give $\sup_n \|S_n\| < \infty$, a contradiction. So there exists $f$ for which $(S_n(f))$ is unbounded — divergence of the Fourier series. $\square$

**Application 3: Weakly bounded implies norm bounded.** A subset $A$ of a Banach space $X$ is **weakly bounded** if $\sup_{x \in A} |f(x)| < \infty$ for every $f \in X^*$. By the UBP applied to the family $\{\hat{x}\}_{x \in A} \subset X^{**}$ (viewing elements of $X$ as functionals on $X^*$), we get $\sup_{x \in A} \|x\| < \infty$. So weakly bounded sets are norm bounded.

---

## The Open Mapping Theorem

**Theorem (Open Mapping Theorem / Banach-Schauder).** Let $X$ and $Y$ be Banach spaces and $T: X \to Y$ a bounded linear **surjection**. Then $T$ is an open map: for every open set $U \subset X$, the image $T(U)$ is open in $Y$.

**Proof.** It suffices to show that $T(B_X(0,1))$ contains a ball centered at the origin in $Y$.

*Step 1: $T(B_X(0,1))$ has nonempty interior.* Since $T$ is surjective, $Y = \bigcup_{n=1}^\infty T(B_X(0, n)) = \bigcup_{n=1}^\infty n \cdot T(B_X(0,1))$. By the Baire Category Theorem, some $n \cdot \overline{T(B_X(0,1))}$ has nonempty interior, so $\overline{T(B_X(0,1))}$ has nonempty interior. By a symmetry argument (the set is symmetric about the origin: if $y \in T(B_X(0,1))$, then $-y = T(-x) \in T(B_X(0,1))$), and convex (if $y_1 = Tx_1$, $y_2 = Tx_2$ with $\|x_i\| < 1$, then $ty_1 + (1-t)y_2 = T(tx_1 + (1-t)x_2)$ with $\|tx_1 + (1-t)x_2\| < 1$), the interior of the closure contains a ball centered at the origin: $B_Y(0, \delta) \subset \overline{T(B_X(0,1))}$ for some $\delta > 0$.

*Step 2: From closure to actual image.* We show $B_Y(0, \delta/2) \subset T(B_X(0,1))$. Pick any $y$ with $\|y\| < \delta/2$. Since $y \in \overline{T(B_X(0, 1/2))}$ (by scaling Step 1), there exists $x_1$ with $\|x_1\| < 1/2$ and $\|y - Tx_1\| < \delta/4$.

Now $y - Tx_1 \in \overline{T(B_X(0, 1/4))}$, so there exists $x_2$ with $\|x_2\| < 1/4$ and $\|y - Tx_1 - Tx_2\| < \delta/8$.

Continue: at step $n$, find $x_n$ with $\|x_n\| < 2^{-n}$ and $\|y - \sum_{k=1}^n Tx_k\| < \delta \cdot 2^{-(n+1)}$.

The series $x = \sum_{n=1}^\infty x_n$ converges absolutely in $X$ (since $\sum \|x_n\| < 1$) and $\|x\| < 1$. By continuity of $T$: $Tx = \sum Tx_n = y$. Hence $y \in T(B_X(0,1))$. $\square$

**Corollary (Bounded Inverse Theorem / Banach's Isomorphism Theorem).** If $T: X \to Y$ is a bounded linear bijection between Banach spaces, then $T^{-1}$ is bounded.

**Proof.** $T$ is an open map by the Open Mapping Theorem, so $T^{-1}$ is continuous, hence bounded. $\square$

This corollary is powerful: it says that for Banach spaces, algebraic invertibility of a bounded operator **automatically** implies topological invertibility. You never need to prove separately that the inverse is continuous.

**Example 3 (Equivalent norms).** If $\|\cdot\|_1$ and $\|\cdot\|_2$ are two complete norms on a vector space $X$, and $\|x\|_2 \leq C\|x\|_1$ for all $x$, then the norms are equivalent: $\|x\|_1 \leq C'\|x\|_2$ for some $C'$. Proof: the identity map $(X, \|\cdot\|_1) \to (X, \|\cdot\|_2)$ is a bounded bijection between Banach spaces.

**Example 4 (Projections).** If $X$ is a Banach space and $X = M \oplus N$ as a direct sum of closed subspaces, then the projection $P: X \to M$ (along $N$) is bounded. Proof: $P$ has a closed graph (graph of $P$ = $\{(m+n, m) : m \in M, n \in N\}$, closed since $M, N$ are closed), so the Closed Graph Theorem (below) gives boundedness. Alternatively: the map $(m, n) \mapsto m + n$ from $M \times N$ (with product norm) to $X$ is a bounded bijection, so its inverse is bounded, and $P$ is bounded.

---

## The Closed Graph Theorem

**Definition.** The **graph** of a linear map $T: X \to Y$ is $\text{Graph}(T) = \{(x, Tx) : x \in X\} \subset X \times Y$. We say $T$ has a **closed graph** if $\text{Graph}(T)$ is closed in $X \times Y$ (with the product topology, equivalently the norm $\|(x,y)\| = \|x\| + \|y\|$).

**Theorem (Closed Graph Theorem).** Let $X$ and $Y$ be Banach spaces. A linear map $T: X \to Y$ is bounded if and only if its graph is closed.

**Proof.** The forward direction is clear: if $T$ is bounded (hence continuous) and $(x_n, Tx_n) \to (x, y)$, then $Tx_n \to Tx$ by continuity, so $y = Tx$ and $(x, y) \in \text{Graph}(T)$.

For the converse, assume $\text{Graph}(T)$ is closed. Then $\text{Graph}(T)$ is a closed subspace of the Banach space $X \times Y$, hence a Banach space. The map $\pi_1: \text{Graph}(T) \to X$ defined by $\pi_1(x, Tx) = x$ is a bounded linear bijection. By the Bounded Inverse Theorem, $\pi_1^{-1}: X \to \text{Graph}(T)$ is bounded, i.e., there exists $C$ such that $\|x\| + \|Tx\| \leq C\|x\|$. Hence $\|Tx\| \leq (C-1)\|x\|$, and $T$ is bounded. $\square$

**When is the Closed Graph Theorem useful?** When you have a linear map that is "naturally defined" and you need to show it is bounded, but estimating the operator norm directly is hard. Instead, you just check: if $x_n \to x$ and $Tx_n \to y$, is $y = Tx$? This is often much easier.

**Example 5 (Differentiation operator, used correctly).** Let $X = C^1([0,1])$ with the norm $\|f\| = \|f\|_\infty + \|f'\|_\infty$, and let $Y = C([0,1])$ with $\|g\| = \|g\|_\infty$. The differentiation operator $D: X \to Y$ defined by $Df = f'$ satisfies $\|Df\| = \|f'\|_\infty \leq \|f\|_X$, so $D$ is bounded with $\|D\| \leq 1$. But on the space $C^1([0,1])$ with only the sup-norm $\|f\|_\infty$, differentiation is **not** bounded — the graph is not closed in this weaker topology.

**Example 6 (Using the Closed Graph Theorem for multiplication operators).** Let $g: [0,1] \to \mathbb{R}$ be a measurable function, and suppose the multiplication operator $M_g: L^2([0,1]) \to L^2([0,1])$, $M_g f = gf$, maps $L^2$ into $L^2$. We claim $M_g$ is bounded.

The closed graph approach: suppose $f_n \to f$ in $L^2$ and $gf_n \to h$ in $L^2$. We need $h = gf$. Pass to a subsequence $f_{n_k} \to f$ a.e. Then $gf_{n_k} \to gf$ a.e. But also $gf_{n_k} \to h$ in $L^2$, hence (passing to a further subsequence) $gf_{n_k} \to h$ a.e. Thus $h = gf$ a.e., and the graph is closed. By the Closed Graph Theorem, $M_g$ is bounded.

One can then show (by a separate argument) that $M_g$ bounded implies $g \in L^\infty([0,1])$ — so the Closed Graph Theorem gives us that $g$ must be essentially bounded, just from the assumption that $M_g$ maps $L^2$ to $L^2$.

**Relation between the three theorems.** While we presented the Uniform Boundedness Principle, Open Mapping Theorem, and Closed Graph Theorem as independent results, they are in fact closely related. Here is the logical dependency structure:

- The **Baire Category Theorem** is the common foundation.
- The **Open Mapping Theorem** can be deduced from the Baire Category Theorem (as we did above).
- The **Bounded Inverse Theorem** is an immediate corollary of the Open Mapping Theorem.
- The **Closed Graph Theorem** follows from the Bounded Inverse Theorem (as we showed).
- The **Uniform Boundedness Principle** can be proved independently from Baire, or can also be derived from the Closed Graph Theorem (by considering the operator $(x, \alpha) \mapsto \sum \alpha_i T_i x$ on an appropriate product space).

Conversely, one can start from the Uniform Boundedness Principle and derive the Open Mapping Theorem and Closed Graph Theorem, though this requires more work. The conceptual unity lies in the Baire Category Theorem: all three results exploit the fact that a complete metric space cannot be expressed as a countable union of nowhere dense sets.

---

## Applications: Automatic Continuity

The three theorems above reveal a remarkable feature of Banach spaces: many operations that are merely linear turn out to be automatically continuous, with no additional work.

**Theorem (Hellinger-Toeplitz).** Let $H$ be a Hilbert space and $T: H \to H$ a linear map satisfying $\langle Tx, y \rangle = \langle x, Ty \rangle$ for all $x, y \in H$ (i.e., $T$ is everywhere-defined and symmetric). Then $T$ is bounded.

**Proof.** We use the Closed Graph Theorem. Suppose $x_n \to x$ and $Tx_n \to y$. For any $z \in H$:

$$\langle y, z \rangle = \lim_n \langle Tx_n, z \rangle = \lim_n \langle x_n, Tz \rangle = \langle x, Tz \rangle = \langle Tx, z \rangle.$$

Since this holds for all $z$, we have $y = Tx$. The graph is closed, so $T$ is bounded. $\square$

This has profound consequences for quantum mechanics: any symmetric (Hermitian) operator defined on **all** of a Hilbert space must be bounded. Unbounded self-adjoint operators in quantum mechanics (like the momentum operator $-i\hbar \frac{d}{dx}$ or the position operator $\hat{x}$) can only exist on proper dense subdomains — they cannot be defined on the entire Hilbert space while remaining symmetric. This is one of the key reasons that the mathematical foundations of quantum mechanics require careful attention to operator domains, a subject we will touch on when we discuss unbounded operators later in this series.

**Another application: Continuous inverse of Fourier transform.** The Fourier transform $\mathcal{F}: L^2(\mathbb{R}) \to L^2(\mathbb{R})$ is a bounded bijection (this is a consequence of the Plancherel theorem). By the Bounded Inverse Theorem, $\mathcal{F}^{-1}$ is automatically bounded. We get the Plancherel isometry "for free" from the abstract theory, though of course the concrete proof that $\mathcal{F}$ is surjective still requires analysis.

**Application: Closed range theorem and solvability.** The Closed Range Theorem (due to Banach and later refined) states: for $T \in B(X, Y)$ with $X, Y$ Banach, the range of $T$ is closed if and only if the range of $T^*$ is closed. Moreover, $\overline{\text{range}(T)} = \ker(T^*)^\perp$ and $\overline{\text{range}(T^*)} = \ker(T)^\perp$ (where annihilators replace orthogonal complements in the Banach space setting). This theorem ties together the three big theorems and provides a systematic approach to solvability questions: the equation $Tx = y$ is solvable if and only if $y$ annihilates $\ker(T^*)$.

**Application: Principle of condensation of singularities in detail.** The UBP proof that the Fourier series of some continuous function diverges deserves a more detailed exposition, as it illustrates a powerful technique.

The $n$-th partial sum of the Fourier series at $x = 0$ defines a functional $S_n \in C([-\pi, \pi])^*$:

$$S_n(f) = \sum_{k=-n}^{n} \hat{f}(k) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(t) D_n(t) \, dt,$$

where $D_n(t) = \sum_{k=-n}^n e^{ikt} = \frac{\sin((n+1/2)t)}{\sin(t/2)}$ is the Dirichlet kernel. The norm of $S_n$ is the Lebesgue constant:

$$\|S_n\| = \frac{1}{2\pi}\int_{-\pi}^{\pi} |D_n(t)| \, dt = L_n.$$

A classical computation shows $L_n = \frac{4}{\pi^2}\log n + O(1)$, so $L_n \to \infty$. If $S_n(f)$ were bounded for every continuous $f$, the UBP would give $\sup_n L_n < \infty$, contradiction. Hence the set $\{f \in C[-\pi, \pi] : \sup_n |S_n(f)| = \infty\}$ is nonempty. In fact, by the Baire Category Theorem, this set is **residual** (a dense $G_\delta$) — the "generic" continuous function has a divergent Fourier series at $0$. This is Carleson's theorem's complement: while the Fourier series of $L^2$ functions converge almost everywhere (Carleson, 1966), pointwise divergence is typical for continuous functions at individual points.

---

## Counterexamples: What Fails Without Completeness

The completeness hypothesis in all three theorems is essential, not a technical convenience. The following counterexamples make this concrete, and understanding them is as important as understanding the theorems themselves — they delineate exactly where the theory applies and where it does not.

**Counterexample 1 (UBP fails without completeness of $X$).** Let $X$ be the space of finitely supported sequences with the sup-norm (a non-complete subspace of $c_0$). Define $T_n: X \to \mathbb{R}$ by $T_n(x) = n \cdot x_n$. Then $\|T_n\| = n \to \infty$, but for any fixed $x \in X$ (which has finite support), $T_n(x) = 0$ for all large $n$. So $\sup_n |T_n(x)| < \infty$ for each $x$, but $\sup_n \|T_n\| = \infty$.

**Counterexample 2 (Open Mapping fails without completeness of $Y$).** Let $X = Y = c_{00}$ (finitely supported sequences) with the $\ell^1$ norm, and let $T: X \to Y$ be the identity map viewed from $(c_{00}, \|\cdot\|_1)$ to $(c_{00}, \|\cdot\|_2)$. Since $\|x\|_2 \leq \|x\|_1$ on $c_{00}$, $T$ is bounded. It is a bijection, but $T^{-1}$ is unbounded (consider $x = (1/\sqrt{n}, \ldots, 1/\sqrt{n}, 0, \ldots)$ with $n$ nonzero terms: $\|x\|_1 = \sqrt{n}$, $\|x\|_2 = 1$). The Open Mapping Theorem fails because $Y$ (with the $\ell^2$ norm) is not complete.

**Counterexample 3 (Closed Graph fails without completeness).** Consider the differentiation operator $D: C^1([0,1]) \to C([0,1])$ where $C^1([0,1])$ carries only the sup-norm (not the $C^1$ norm). Then $D$ is unbounded, yet its graph is closed: if $f_n \to f$ uniformly and $f_n' \to g$ uniformly, then $f$ is differentiable with $f' = g$ (by the uniform convergence theorem for derivatives). The issue is that $(C^1([0,1]), \|\cdot\|_\infty)$ is not a Banach space.

**Deeper failure: a non-complete space where every linear functional is continuous.** In a Banach space, the Hahn-Banach theorem guarantees an abundance of continuous linear functionals. But there exist non-complete metrizable topological vector spaces (e.g., $L^p$ for $0 < p < 1$) where the only continuous linear functional is the zero functional. In such spaces, the dual space is trivial, and the notion of "weak topology" collapses. The theory of this article is specific to Banach spaces (or at least locally convex spaces) for good reason.

**Historical note.** The Uniform Boundedness Principle was proved by Banach and Steinhaus in 1927. The Open Mapping Theorem was proved by Banach in 1929 (and independently by Schauder around the same time). The Closed Graph Theorem, as stated here for Banach spaces, also appears in Banach's 1932 monograph *Theorie des operations lineaires*. The realization that all three results flow from the Baire Category Theorem (itself proved by Baire in 1899) came gradually and gave the subject its modern form.

---

## A Unified Perspective

All three theorems can be seen as manifestations of a single principle: **in complete spaces, algebraic structure constrains topological behavior far more than one might expect.**

| Theorem | Input | Conclusion |
|---|---|---|
| Uniform Boundedness | Pointwise bounded family | Uniformly bounded |
| Open Mapping | Surjective bounded linear map | Open map (hence homeomorphism if bijective) |
| Closed Graph | Linear map with closed graph | Bounded (hence continuous) |

Each converts a "soft" algebraic hypothesis into a "hard" analytic conclusion, and each fails without completeness.

These three theorems, together with the Hahn-Banach theorem from Article 4, are sometimes called the **four pillars of functional analysis**. Hahn-Banach is algebraic-geometric in nature (it works even in non-complete spaces and non-metrizable topologies), while the three theorems of this article are topological-metric, fundamentally relying on the Baire Category Theorem.

**When do you use which theorem?** A practical guide:

- **Uniform Boundedness Principle:** when you have a family of operators (or functionals) and need to show the norms are uniformly bounded. Typical setting: you know pointwise bounds and want uniform bounds.
- **Open Mapping Theorem / Bounded Inverse Theorem:** when you have a bijective bounded operator and need to show the inverse is bounded. Typical setting: you have two Banach space norms on the same space and one dominates the other.
- **Closed Graph Theorem:** when you have a linear map defined by some natural formula and need to show it is bounded, but direct norm estimates are hard. Typical setting: the map is "obviously" well-defined and the graph closure is easy to check.

---

## What's Next

With the three big theorems established, we move to a more specific and beautiful class of operators: **compact operators**. These are operators that map bounded sets to relatively compact sets — they are the "closest to finite-dimensional" among all bounded operators and enjoy a spectral theory that closely mirrors the eigenvalue decomposition of matrices. The Fredholm alternative and the spectral theorem for compact self-adjoint operators await.

The progression of this series so far has been: spaces (Articles 1-3), duality (Article 4), compactness via weak topologies (Article 5), general operator theory (this article), and now we specialize to compact operators (Article 7) where the spectral theory becomes especially concrete and powerful. Each level of specialization — from bounded operators to compact operators to self-adjoint compact operators — buys us stronger structural results, culminating in a theory that is virtually indistinguishable from finite-dimensional linear algebra.

---

*This is Part 6 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 5 — Weak Topologies](/en/functional-analysis/05-weak-topologies/)*

*Next: [Part 7 — Compact Operators](/en/functional-analysis/07-compact-operators/)*
