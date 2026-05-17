---
title: "Functional Analysis (11): Distributions and Sobolev Spaces — Generalized Solutions"
date: 2021-10-21 09:00:00
tags:
  - functional-analysis
  - distributions
  - sobolev-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "Distributions extend the notion of function to handle derivatives that don't exist classically — Sobolev spaces provide the right setting for weak solutions to PDE."
disableNunjucks: true
series_order: 11
series_total: 12
translationKey: "functional-analysis-11"
---

Consider the wave equation on the real line. A traveling wave $u(x,t) = f(x - ct)$ solves $u_{tt} = c^2 u_{xx}$ for any twice-differentiable profile $f$. But physical waves can have sharp fronts — a step function profile, for instance, which is not differentiable anywhere. Can we still say such a function "solves" the wave equation? Classical calculus says no. Distribution theory says yes.

The Dirac delta "function" $\delta(x)$, defined by the property $\int \delta(x)\varphi(x)\,dx = \varphi(0)$, is another fundamental object that resists classical treatment. It is not a function in any traditional sense — no measurable function can satisfy this integral identity. Yet it appears throughout physics and engineering as the idealization of a point source, a point mass, or an instantaneous impulse.

Laurent Schwartz's theory of distributions (1950) provides a rigorous framework that accommodates both of these situations. The key idea is elegant: instead of trying to assign pointwise values to generalized "functions," we define them by how they act on smooth test functions. This shift in perspective — from pointwise evaluation to duality — is one of the great triumphs of functional analysis.

Sobolev spaces, introduced by Sergei Sobolev in the 1930s, build on this foundation. They are Banach spaces (often Hilbert spaces) consisting of functions with a prescribed number of weak derivatives in $L^p$. These spaces are the natural domains for differential operators and the correct setting for existence and regularity theory for PDE.

---

## Why Classical Derivatives Aren't Enough

### The motivating problems

**Problem 1: Weak solutions of PDE.** Consider the equation $-u'' = f$ on $(0,1)$ with $u(0) = u(1) = 0$. If $f$ is continuous, classical solutions exist and are $C^2$. But what if $f \in L^2(0,1)$ — say $f$ is the indicator function of an interval? No classical ($C^2$) solution exists. Yet multiplying by a test function $\varphi \in C_c^\infty(0,1)$ and integrating by parts gives

![Sobolev space embedding chain](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/11-distributions-sobolev/fa_fig6_sobolev.png)


$$
\int_0^1 u'\varphi' \, dx = \int_0^1 f\varphi \, dx \quad \text{for all } \varphi \in C_c^\infty(0,1).
$$

A function $u$ satisfying this integral identity is a **weak solution**. It need not be twice differentiable; $u \in H^1_0(0,1)$ (one weak derivative in $L^2$, vanishing at the boundary) suffices. Making this precise requires Sobolev spaces.

**Problem 2: The Dirac delta.** In electrostatics, the potential of a point charge at the origin satisfies $-\Delta \phi = \delta$. The right-hand side is not a function. To make this equation rigorous, we need a space that contains $\delta$ — the space of distributions.

**Problem 3: Fourier analysis.** The Fourier transform of the constant function $1$ is (formally) $\delta$. A rigorous theory of the Fourier transform on $L^2$ (via Plancherel) does not cover distributions, yet extending Fourier analysis to tempered distributions is essential for PDE and signal processing.

### The conceptual shift

The classical approach: a function $f$ is defined by its pointwise values $f(x)$.

The distributional approach: a generalized function $f$ is defined by its action on test functions: $\varphi \mapsto \langle f, \varphi \rangle = \int f\varphi \, dx$.

Two locally integrable functions that differ on a set of measure zero define the same distribution (their integrals against test functions are identical). This means distributions automatically quotient out null sets — a feature, not a bug, since $L^p$ spaces already do the same.

### The topology on $\mathcal{D}'(\Omega)$

The space of distributions carries a natural topology: the **weak-* topology**, where $u_j \to u$ in $\mathcal{D}'(\Omega)$ if $\langle u_j, \varphi \rangle \to \langle u, \varphi \rangle$ for every $\varphi \in \mathcal{D}(\Omega)$. This is the topology of pointwise convergence on test functions. It is weaker than the strong topology (uniform convergence on bounded sets of test functions) but sufficient for most purposes.

A key result: every distribution is the limit (in $\mathcal{D}'$) of a sequence of smooth functions. More precisely, if $u \in \mathcal{D}'(\Omega)$ and $\rho_\epsilon$ is a standard mollifier, then $u * \rho_\epsilon \to u$ in $\mathcal{D}'$ as $\epsilon \to 0$ (where the convolution is defined appropriately). This density of smooth functions in $\mathcal{D}'$ is analogous to the density of $C_c^\infty$ in $L^p$, but holds in a much more general setting.

---

## Test Functions D(Omega) and the Space of Distributions D'(Omega)

### The space of test functions

Let $\Omega \subseteq \mathbb{R}^n$ be open. The space of **test functions** is

$$
\mathcal{D}(\Omega) = C_c^\infty(\Omega),
$$

the space of infinitely differentiable functions with compact support in $\Omega$.

**Topology on $\mathcal{D}(\Omega)$.** A sequence $\varphi_j \to 0$ in $\mathcal{D}(\Omega)$ if:
1. There exists a compact set $K \subset \Omega$ such that $\text{supp}(\varphi_j) \subset K$ for all $j$.
2. For every multi-index $\alpha$, $\partial^\alpha \varphi_j \to 0$ uniformly on $K$.

This is *not* a norm topology — it is an inductive limit topology, the finest locally convex topology making all inclusion maps $C_c^\infty(K) \hookrightarrow C_c^\infty(\Omega)$ continuous. The precise details of this topology are technically demanding but rarely needed in applications; what matters is the sequential characterization above.

**Existence of test functions.** A standard construction: let $\rho(x) = Ce^{-1/(1-|x|^2)}$ for $|x| < 1$ and $\rho(x) = 0$ for $|x| \ge 1$, where $C$ is chosen so that $\int \rho = 1$. Then $\rho \in C_c^\infty(\mathbb{R}^n)$, and $\rho_\epsilon(x) = \epsilon^{-n}\rho(x/\epsilon)$ is a smooth mollifier supported in $B(0, \epsilon)$.

### Distributions

**Definition.** A **distribution** on $\Omega$ is a continuous linear functional on $\mathcal{D}(\Omega)$. The space of all distributions is denoted $\mathcal{D}'(\Omega)$.

Continuity means: if $\varphi_j \to 0$ in $\mathcal{D}(\Omega)$, then $\langle u, \varphi_j \rangle \to 0$. Equivalently, for every compact $K \subset \Omega$, there exist $C > 0$ and $N \in \mathbb{N}$ such that

$$
|\langle u, \varphi \rangle| \le C \sum_{|\alpha| \le N} \sup_K |\partial^\alpha \varphi| \quad \text{for all } \varphi \in C_c^\infty(K).
$$

The smallest such $N$ that works for all compact $K$ (if it exists globally) is the **order** of the distribution. Not all distributions have finite order.

### Embedding of functions into distributions

Every locally integrable function $f \in L^1_{\text{loc}}(\Omega)$ defines a distribution via

$$
\langle T_f, \varphi \rangle = \int_\Omega f(x)\varphi(x) \, dx.
$$

The map $f \mapsto T_f$ is injective (if $\int f\varphi = 0$ for all test functions $\varphi$, then $f = 0$ a.e.) and continuous. We identify $f$ with $T_f$ and write $\langle f, \varphi \rangle$ for the pairing.

This embedding is the bridge between classical and distributional analysis: every $L^p$ function, every continuous function, every measurable locally bounded function is a distribution. But distributions are strictly more general.

### Key examples

**The Dirac delta.** $\langle \delta, \varphi \rangle = \varphi(0)$. This is a distribution of order 0. It cannot be represented by any locally integrable function.

**The principal value distribution.** $\langle \text{p.v.}\frac{1}{x}, \varphi \rangle = \lim_{\epsilon \to 0^+} \int_{|x| > \epsilon} \frac{\varphi(x)}{x} \, dx$. This is a distribution of order 1.

**Derivatives of delta.** $\langle \delta^{(k)}, \varphi \rangle = (-1)^k \varphi^{(k)}(0)$. These are distributions of order $k$.

### The support of a distribution

A distribution $u \in \mathcal{D}'(\Omega)$ is said to vanish on an open set $U \subset \Omega$ if $\langle u, \varphi \rangle = 0$ for all $\varphi \in C_c^\infty(U)$. The **support** of $u$ is the complement of the largest open set on which $u$ vanishes:

$$
\text{supp}(u) = \Omega \setminus \bigcup \{U \text{ open} : u|_U = 0\}.
$$

For the Dirac delta, $\text{supp}(\delta) = \{0\}$. A fundamental structural theorem characterizes distributions with point support:

**Theorem.** If $u \in \mathcal{D}'(\mathbb{R}^n)$ has $\text{supp}(u) = \{0\}$, then $u$ is a finite linear combination of derivatives of $\delta$: $u = \sum_{|\alpha| \le N} c_\alpha \partial^\alpha \delta$ for some $N$ and constants $c_\alpha$.

This result shows that the only "singularities" that can be concentrated at a single point are those generated by delta functions and their derivatives — no exotic new objects lurk at a point.

---

## Operations on Distributions: Derivatives, Multiplication, Convolution

### Distributional derivatives

The key operation. For a smooth function $f$, integration by parts gives

$$
\int f'\varphi \, dx = -\int f\varphi' \, dx \quad \text{for all } \varphi \in C_c^\infty.
$$

This motivates:

**Definition.** The **distributional derivative** $\partial^\alpha u$ of a distribution $u \in \mathcal{D}'(\Omega)$ is defined by

$$
\langle \partial^\alpha u, \varphi \rangle = (-1)^{|\alpha|} \langle u, \partial^\alpha \varphi \rangle \quad \text{for all } \varphi \in \mathcal{D}(\Omega).
$$

**Every distribution is infinitely differentiable** in the distributional sense. This is a dramatic departure from classical analysis, where differentiability is a restrictive condition.

**Examples.**

1. **Heaviside function.** Let $H(x) = \mathbf{1}_{[0,\infty)}(x)$. Then $\langle H', \varphi \rangle = -\langle H, \varphi' \rangle = -\int_0^\infty \varphi'(x) \, dx = \varphi(0) = \langle \delta, \varphi \rangle$. So $H' = \delta$ in the distributional sense — the derivative of the step function is the delta function.

2. **Absolute value.** $|x|' = \text{sgn}(x)$ and $|x|'' = 2\delta$ (both in the distributional sense). The second derivative of $|x|$, which doesn't exist classically at the origin, is twice the Dirac delta.

3. **$\log|x|$ in one dimension.** The distributional derivative is $\text{p.v.}\frac{1}{x}$, the principal value distribution.

### Multiplication by smooth functions

If $a \in C^\infty(\Omega)$ and $u \in \mathcal{D}'(\Omega)$, define $\langle au, \varphi \rangle = \langle u, a\varphi \rangle$. This is well-defined since $a\varphi \in C_c^\infty(\Omega)$ whenever $\varphi$ is.

**Warning.** Multiplication of two arbitrary distributions is *not* defined in general. The product $\delta \cdot \delta$ has no canonical meaning, and attempts to define it lead to the difficulties of renormalization in quantum field theory. Schwartz's impossibility theorem makes this precise: there is no associative, commutative multiplication on $\mathcal{D}'$ that extends the pointwise product of continuous functions and satisfies the Leibniz rule.

### Convolution

If $u \in \mathcal{D}'(\mathbb{R}^n)$ has compact support and $\varphi \in C^\infty(\mathbb{R}^n)$, the convolution $u * \varphi$ is defined and is $C^\infty$. More generally, convolution with a compactly supported distribution is always well-defined:

$$
\langle u * v, \varphi \rangle = \langle u(x), \langle v(y), \varphi(x+y) \rangle \rangle.
$$

Key properties:
- $\delta * f = f$ for any distribution $f$ (the delta function is the convolution identity).
- $\partial^\alpha(u * v) = (\partial^\alpha u) * v = u * (\partial^\alpha v)$ (derivatives can be shifted between the factors).
- Mollification: $f * \rho_\epsilon \in C^\infty$ for any distribution $f$ with compact support, and $f * \rho_\epsilon \to f$ in $\mathcal{D}'$ as $\epsilon \to 0$.

### Tempered distributions and the Fourier transform

The **Schwartz space** $\mathcal{S}(\mathbb{R}^n)$ consists of smooth functions whose derivatives all decay faster than any polynomial: $\sup_x |x^\alpha \partial^\beta \varphi(x)| < \infty$ for all multi-indices $\alpha, \beta$. Its dual $\mathcal{S}'(\mathbb{R}^n)$ is the space of **tempered distributions**.

The Fourier transform extends to an isomorphism $\mathcal{F}: \mathcal{S}' \to \mathcal{S}'$ via duality: $\langle \hat{u}, \varphi \rangle = \langle u, \hat{\varphi} \rangle$. This gives meaning to the Fourier transform of polynomials, $\delta$, and many other objects that fall outside the scope of $L^1$ or $L^2$ Fourier analysis.

**Key formulas in $\mathcal{S}'$:**
- $\hat{\delta} = 1$ (the Fourier transform of the delta function is the constant function 1).
- $\hat{1} = (2\pi)^n \delta$ (the Fourier transform of the constant function 1 is $(2\pi)^n$ times delta).
- $\widehat{\partial^\alpha u} = (i\xi)^\alpha \hat{u}$ (differentiation becomes multiplication by polynomials — valid for all tempered distributions, not just functions).

These identities are the foundation of the Fourier-analytic approach to PDE: solving $-\Delta u = f$ becomes, after Fourier transform, $|\xi|^2 \hat{u} = \hat{f}$, so $\hat{u} = \hat{f}/|\xi|^2$. The difficulty is purely in the inverse transform (and the behavior at $\xi = 0$), not in the algebraic step.

### Fundamental solutions

A **fundamental solution** (or Green's function) for a linear differential operator $L$ is a distribution $E$ satisfying $LE = \delta$. For the Laplacian in $\mathbb{R}^n$ ($n \ge 3$):

$$
E(x) = \frac{1}{n(n-2)\omega_n|x|^{n-2}},
$$

where $\omega_n$ is the volume of the unit ball. In $\mathbb{R}^2$, $E(x) = -\frac{1}{2\pi}\log|x|$; in $\mathbb{R}^1$, $E(x) = -\frac{1}{2}|x|$.

The solution to $-\Delta u = f$ (for suitable $f$) is then $u = E * f$ — convolution with the fundamental solution. This is the distributional incarnation of the classical potential-theoretic approach to electrostatics.

For the heat equation $(\partial_t - \Delta)u = 0$, the fundamental solution is the heat kernel $K(x, t) = (4\pi t)^{-n/2}e^{-|x|^2/(4t)}$ for $t > 0$, which we encountered in the semigroup article. The distributional perspective clarifies why this kernel appears: it is the unique tempered distribution satisfying $(\partial_t - \Delta)K = \delta(x)\delta(t)$.

---

## Sobolev Spaces W^{k,p} and H^k

### Definition

**Definition.** For $k \in \mathbb{N}_0$, $1 \le p \le \infty$, and $\Omega \subseteq \mathbb{R}^n$ open, the **Sobolev space** $W^{k,p}(\Omega)$ is

$$
W^{k,p}(\Omega) = \{u \in L^p(\Omega) : \partial^\alpha u \in L^p(\Omega) \text{ for all } |\alpha| \le k\},
$$

where $\partial^\alpha u$ is the distributional derivative. The norm is

$$
\|u\|_{W^{k,p}} = \left(\sum_{|\alpha| \le k} \|\partial^\alpha u\|_{L^p}^p\right)^{1/p} \quad (1 \le p < \infty),
$$

with the obvious modification for $p = \infty$.

**Notation.** $H^k(\Omega) = W^{k,2}(\Omega)$. These are Hilbert spaces with inner product $\langle u, v \rangle_{H^k} = \sum_{|\alpha| \le k} \langle \partial^\alpha u, \partial^\alpha v \rangle_{L^2}$.

**Definition.** $W_0^{k,p}(\Omega)$ is the closure of $C_c^\infty(\Omega)$ in $W^{k,p}(\Omega)$. Intuitively, these are Sobolev functions that "vanish at the boundary" in a generalized sense. Similarly, $H_0^k(\Omega) = W_0^{k,2}(\Omega)$.

### Completeness

**Theorem.** $W^{k,p}(\Omega)$ is a Banach space. For $p = 2$, $H^k(\Omega)$ is a Hilbert space.

*Proof.* Let $(u_j)$ be a Cauchy sequence in $W^{k,p}$. For each $|\alpha| \le k$, the sequence $(\partial^\alpha u_j)$ is Cauchy in $L^p(\Omega)$. By completeness of $L^p$, there exist functions $u_\alpha \in L^p$ with $\partial^\alpha u_j \to u_\alpha$ in $L^p$. Set $u = u_0$ (the limit of $u_j$ itself in $L^p$). We claim $\partial^\alpha u = u_\alpha$ in the distributional sense. For any test function $\varphi$:

$$
\langle \partial^\alpha u, \varphi \rangle = (-1)^{|\alpha|}\langle u, \partial^\alpha \varphi \rangle = (-1)^{|\alpha|}\lim_j \langle u_j, \partial^\alpha \varphi \rangle = \lim_j \langle \partial^\alpha u_j, \varphi \rangle = \langle u_\alpha, \varphi \rangle.
$$

So $u \in W^{k,p}$ and $u_j \to u$ in $W^{k,p}$. $\square$

### Fractional and negative Sobolev spaces

For $s \in \mathbb{R}$ (not necessarily an integer), one can define $H^s(\mathbb{R}^n)$ via the Fourier transform:

$$
H^s(\mathbb{R}^n) = \{u \in \mathcal{S}'(\mathbb{R}^n) : (1 + |\xi|^2)^{s/2}\hat{u} \in L^2(\mathbb{R}^n)\},
$$

with norm $\|u\|_{H^s} = \|(1 + |\xi|^2)^{s/2}\hat{u}\|_{L^2}$. For $s < 0$, $H^s$ contains distributions that are not functions. The space $H^{-k}(\Omega)$ is naturally identified with the dual of $H_0^k(\Omega)$.

The Dirac delta $\delta \in H^s(\mathbb{R}^n)$ if and only if $s < -n/2$ (since $\hat{\delta} = 1$, and $(1 + |\xi|^2)^{s/2} \in L^2$ iff $s < -n/2$).

### Density and approximation

A fundamental property of Sobolev spaces is that smooth functions are dense:

**Theorem (Meyers-Serrin).** $C^\infty(\Omega) \cap W^{k,p}(\Omega)$ is dense in $W^{k,p}(\Omega)$ for $1 \le p < \infty$.

When $\Omega$ has sufficiently regular boundary, even $C^\infty(\overline{\Omega})$ is dense in $W^{k,p}(\Omega)$. This approximation property is essential for proving many results: one first establishes the result for smooth functions (where classical calculus applies) and then extends by density.

### Poincare inequality

**Theorem (Poincare inequality).** Let $\Omega \subset \mathbb{R}^n$ be bounded and connected. There exists $C = C(\Omega, p)$ such that for all $u \in W_0^{1,p}(\Omega)$:

$$
\|u\|_{L^p(\Omega)} \le C\|\nabla u\|_{L^p(\Omega)}.
$$

This inequality says that for functions vanishing on the boundary, the $L^p$ norm is controlled by the gradient alone. It is the key ingredient in showing that $\|\nabla u\|_{L^p}$ is an equivalent norm on $W_0^{1,p}(\Omega)$, and it appears in every application of the Lax-Milgram theorem to elliptic PDE (as we will see in the next article).

The Poincare constant $C$ depends on the geometry of $\Omega$ — specifically, on its diameter and shape. For a ball of radius $R$, $C \sim R$.

A variant, the **Poincare-Wirtinger inequality**, holds for functions without boundary conditions: $\|u - \bar{u}\|_{L^p} \le C\|\nabla u\|_{L^p}$ where $\bar{u} = \frac{1}{|\Omega|}\int_\Omega u$ is the mean value. This is the relevant inequality for Neumann problems, where the solution is determined only up to an additive constant.

---

## Sobolev Embedding Theorems

The Sobolev embedding theorems answer a fundamental question: if a function has $k$ derivatives in $L^p$, what can we say about its pointwise regularity?

### Sobolev inequality (Gagliardo-Nirenberg-Sobolev)

**Theorem.** Let $1 \le p < n$ and define $p^* = np/(n-p)$ (the **Sobolev conjugate exponent**). Then there exists $C = C(n, p)$ such that for all $u \in W^{1,p}(\mathbb{R}^n)$,

$$
\|u\|_{L^{p^*}} \le C\|\nabla u\|_{L^p}.
$$

Consequently, $W^{1,p}(\mathbb{R}^n) \hookrightarrow L^{p^*}(\mathbb{R}^n)$ continuously.

**Interpretation.** Having one derivative in $L^p$ gives membership in a *better* $L^q$ space (with $q = p^* > p$). The gain in integrability is determined by the dimension $n$: higher dimensions give less improvement.

### Morrey's inequality

**Theorem.** Let $p > n$. Then there exists $C = C(n, p)$ such that for all $u \in W^{1,p}(\mathbb{R}^n)$,

$$
\|u\|_{C^{0,\gamma}(\mathbb{R}^n)} \le C\|u\|_{W^{1,p}},
$$

where $\gamma = 1 - n/p$ and $C^{0,\gamma}$ is the space of Holder continuous functions with exponent $\gamma$.

**Interpretation.** When $p > n$, having one derivative in $L^p$ guarantees *continuity* (and even Holder continuity). This is the threshold: $p < n$ gives integrability improvement; $p > n$ gives regularity.

### The critical case and general embeddings

For $kp < n$: $W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$ for $q \le np/(n - kp)$.

For $kp = n$: $W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$ for all $q < \infty$ (but not $L^\infty$ in general — this is the borderline case, where logarithmic corrections appear).

For $kp > n$: $W^{k,p}(\Omega) \hookrightarrow C^{m,\gamma}(\overline{\Omega})$ where $m = k - \lfloor n/p \rfloor - 1$ and $\gamma$ depends on the fractional part. In this regime, Sobolev functions are classically differentiable — derivatives exist pointwise, not just in the distributional sense.

### Rellich-Kondrachov compactness theorem

**Theorem.** Let $\Omega \subset \mathbb{R}^n$ be bounded with Lipschitz boundary and $1 \le p < n$. Then the embedding $W^{1,p}(\Omega) \hookrightarrow L^q(\Omega)$ is **compact** for $1 \le q < p^*$.

**Significance.** This is the compactness result that powers variational methods in PDE. The idea: a bounded sequence in $W^{1,p}$ has a subsequence converging in $L^q$ (for subcritical $q$). This is the infinite-dimensional analogue of the Bolzano-Weierstrass theorem, replacing "bounded and closed" by "bounded in a stronger norm."

**Application.** Proving existence of eigenvalues for the Laplacian on bounded domains reduces (via the Rellich-Kondrachov theorem) to showing that the resolvent of $-\Delta$ is a compact operator, then applying the spectral theorem for compact operators from earlier in this series.

### Failure of compactness at the critical exponent

The Rellich-Kondrachov theorem fails at the critical exponent $q = p^*$: the embedding $W^{1,p}(\Omega) \hookrightarrow L^{p^*}(\Omega)$ is continuous but *not* compact. This failure has profound consequences for nonlinear PDE. The existence of extremals for the Sobolev inequality (functions that attain equality) was established by Aubin and Talenti, and the associated variational problem exhibits concentration phenomena: minimizing sequences can "concentrate" at a point, losing compactness. Understanding this concentration-compactness phenomenon (Lions, 1984) is central to modern nonlinear analysis.

### Extension theorems

The embedding theorems stated above for $\mathbb{R}^n$ extend to domains $\Omega$ under appropriate regularity assumptions on $\partial\Omega$.

**Theorem (Sobolev extension).** Let $\Omega \subset \mathbb{R}^n$ be bounded with Lipschitz boundary. There exists a bounded linear operator $E: W^{k,p}(\Omega) \to W^{k,p}(\mathbb{R}^n)$ such that $Eu|_\Omega = u$ for all $u \in W^{k,p}(\Omega)$.

This extension operator allows us to reduce problems on domains to problems on $\mathbb{R}^n$, where Fourier-analytic tools are available. The Lipschitz regularity assumption on $\partial\Omega$ is essentially sharp: for domains with cusps or fractal boundaries, extension may fail.

### Sobolev spaces on manifolds

The Sobolev space framework extends to compact Riemannian manifolds $(M, g)$ via coordinate charts and partitions of unity. If $\{(U_\alpha, \phi_\alpha)\}$ is an atlas and $\{\chi_\alpha\}$ a subordinate partition of unity, then $u \in H^k(M)$ if and only if $(\chi_\alpha u) \circ \phi_\alpha^{-1} \in H^k(\phi_\alpha(U_\alpha))$ for each $\alpha$. The Sobolev embedding and compactness theorems carry over with $n = \dim M$.

This extension is essential for studying PDE on curved spaces — the Laplace-Beltrami operator on a Riemannian manifold is the natural generalization of the Euclidean Laplacian, and its analysis requires Sobolev spaces on $M$.

### The role of Sobolev spaces in the functional analysis program

Looking back over the series, Sobolev spaces occupy a central position in the architecture of functional analysis applied to PDE:

- They are the **domains** of differential operators treated as unbounded operators on $L^2$ (Article 9).
- They provide the **Hilbert space setting** for the Lax-Milgram theorem and variational formulations (Article 12).
- The **embedding theorems** connect the abstract Sobolev regularity ($u \in H^k$) to classical regularity ($u \in C^m$), bridging weak and strong solutions.
- The **compactness theorems** (Rellich-Kondrachov) provide the compactness needed for spectral theory, variational methods, and fixed-point arguments.
- The **trace theorems** give rigorous meaning to boundary conditions.

Without Sobolev spaces, the functional-analytic approach to PDE would collapse: there would be no complete spaces in which to seek solutions, no embedding results to extract regularity, and no trace results to impose boundary conditions. Their development in the 1930s-1950s was the key missing piece that allowed the program of Hilbert and Riesz to reach its full potential.

---

## Trace Theorems and Boundary Values

A function in $W^{1,p}(\Omega)$ is defined only up to a set of measure zero. The boundary $\partial\Omega$ has measure zero in $\mathbb{R}^n$, so pointwise restriction to the boundary is not well-defined in the Lebesgue sense. Yet boundary conditions like $u|_{\partial\Omega} = 0$ (Dirichlet) or $\partial u / \partial n|_{\partial\Omega} = g$ (Neumann) are essential for PDE.

The **trace theorem** resolves this by showing that restriction to the boundary extends to a continuous operation on Sobolev spaces.

**Theorem (Trace theorem).** Let $\Omega \subset \mathbb{R}^n$ be bounded with Lipschitz boundary. There exists a unique bounded linear operator

$$
\gamma_0: W^{1,p}(\Omega) \to L^p(\partial\Omega) \quad (1 \le p < \infty)
$$

such that $\gamma_0 u = u|_{\partial\Omega}$ for all $u \in C^\infty(\overline{\Omega})$. Moreover:

1. $\gamma_0$ is surjective onto $W^{1-1/p, p}(\partial\Omega)$ (a fractional Sobolev space on the boundary).
2. $\ker \gamma_0 = W_0^{1,p}(\Omega)$.

**Interpretation.** Part 2 gives a precise characterization of $W_0^{1,p}$: it consists of exactly those Sobolev functions whose trace (boundary value) is zero. This makes rigorous the statement "functions in $H_0^1(\Omega)$ vanish on $\partial\Omega$."

**For higher-order Sobolev spaces,** there are higher-order trace operators $\gamma_j u = \partial^j u / \partial n^j|_{\partial\Omega}$ (normal derivatives of order $j$), and the kernel of the map $(\gamma_0, \gamma_1, \ldots, \gamma_{k-1}): W^{k,p}(\Omega) \to \prod_{j=0}^{k-1} W^{k-j-1/p, p}(\partial\Omega)$ is $W_0^{k,p}(\Omega)$.

### Trace inequalities and applications

The trace theorem comes with a quantitative estimate: $\|\gamma_0 u\|_{L^p(\partial\Omega)} \le C\|u\|_{W^{1,p}(\Omega)}$. This **trace inequality** is essential for formulating and solving boundary value problems.

For Neumann problems ($\partial u/\partial n = g$ on $\partial\Omega$), the trace theorem for normal derivatives gives meaning to the boundary condition: $g$ must lie in the trace space $W^{-1/p', p'}(\partial\Omega)$ (the dual of the trace space for the adjoint exponent). The compatibility condition $\int_\Omega f = \int_{\partial\Omega} g$ (for the Neumann problem $-\Delta u = f$, $\partial u/\partial n = g$) is a consequence of the divergence theorem applied in the distributional sense.

### Capacity and fine properties

Beyond the trace theorem, Sobolev functions possess remarkable fine properties. A function $u \in W^{1,p}(\Omega)$ is defined up to sets of zero $p$-capacity (which are thinner than sets of zero Lebesgue measure). For $p > n$, every set of zero capacity has zero Hausdorff dimension, which is why Morrey's embedding gives pointwise continuity. For $p \le n$, the exceptional sets are more subtle, and understanding them requires potential theory.

The **precise representative** of a Sobolev function is defined at almost every point (in the capacity sense) by taking limits of averages: $u^*(x) = \lim_{r \to 0} \frac{1}{|B(x,r)|}\int_{B(x,r)} u$. This precise representative agrees with $u$ almost everywhere and provides the canonical pointwise interpretation of Sobolev functions.

---

## What's Next

We have built the distributional and Sobolev-space foundations needed for modern PDE theory: distributions give meaning to derivatives that don't exist classically, Sobolev spaces provide Banach/Hilbert settings for weak formulations, embedding theorems connect integrability to regularity, and trace theorems handle boundary conditions rigorously.

In the final article of this series, we put everything together. The **Lax-Milgram theorem** gives existence and uniqueness for elliptic boundary value problems. **Variational methods** convert PDE into minimization problems. **Stone's theorem** connects self-adjoint operators to quantum dynamics. The abstract machinery of functional analysis meets its most important applications.

---

*This is Part 11 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 10 — Semigroups of Operators](/en/functional-analysis/10-semigroups/)*

*Next: [Part 12 — Applications: PDE and QM](/en/functional-analysis/12-applications-pde-qm/)*
