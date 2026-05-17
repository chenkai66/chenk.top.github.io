---
title: "Unbounded Operators: When Boundedness Fails"
date: 2021-10-17 09:00:00
tags:
  - functional-analysis
  - unbounded-operators
  - spectral-theory
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 9
series_total: 12
translationKey: "functional-analysis-9"
description: "Closed operators, the distinction between symmetric and self-adjoint, deficiency indices, Friedrichs extension, the spectral theorem for unbounded self-adjoint operators, and Stone's theorem."
---

Up to this point in the series we have enjoyed a comfortable assumption: our operators are bounded. Bounded operators are tidy objects --- they live on the whole space, they are continuous, and their spectra sit inside a nice compact disc. But the moment you write down any serious differential equation, that comfort evaporates. The Laplacian $-\Delta$ on $L^2(\mathbb{R}^n)$ does not send every $L^2$ function to an $L^2$ function; you need some differentiability. The momentum operator $-i\frac{d}{dx}$ on $L^2(\mathbb{R})$ demands that its inputs be at least absolutely continuous. These operators are *unbounded*, and the theory needed to handle them is simultaneously more delicate and more powerful than anything we have built so far.

This article is the gateway into that theory. We will define closed operators and explain why closedness is the right substitute for continuity. We will dissect the subtle but critical distinction between symmetric and self-adjoint operators --- a distinction that, when ignored, leads to genuine mathematical errors. We will develop the tools (deficiency indices, Friedrichs extension) that tell us when and how an operator can be *made* self-adjoint. Finally, we will state the spectral theorem in the unbounded case and see how Stone's theorem connects self-adjoint operators to one-parameter unitary groups. Throughout, the Laplacian and Schrodinger operator will serve as our primary examples.

## Why unbounded operators are unavoidable

Before diving into the formalism, it is worth pausing to ask: why not just stick to bounded operators? The answer comes from physics and PDE theory. Consider the simplest quantum-mechanical observable: the position operator $Q$ on $L^2(\mathbb{R})$, defined by $(Qf)(x) = xf(x)$. If $f \in L^2(\mathbb{R})$, there is no guarantee that $xf(x)$ is still in $L^2$ --- think of $f(x) = (1+|x|)^{-1}$, which is in $L^2(\mathbb{R})$ but $xf(x)$ is not. So $Q$ cannot be defined on all of $L^2(\mathbb{R})$. Similarly, the momentum operator $P = -i\frac{d}{dx}$ requires differentiability, and the kinetic energy $P^2 = -\frac{d^2}{dx^2}$ requires twice differentiability. None of these can be bounded operators on $L^2$.

More fundamentally, the Hellinger-Toeplitz theorem states that a symmetric operator defined on *all* of a Hilbert space must be bounded. So any symmetric unbounded operator is *forced* to have a restricted domain. The theory of unbounded operators is not an optional generalization; it is a necessity for anyone who wants to do quantum mechanics or PDE theory in a rigorous Hilbert space framework.


![Bounded vs unbounded operators](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_fig3_operators.png)

## Closed operators and their graphs

An *unbounded operator* on a Hilbert space $H$ is a linear map $T: \mathcal{D}(T) \to H$, where the *domain* $\mathcal{D}(T)$ is a linear subspace of $H$ (typically dense, but not the whole space). The word "unbounded" is a misnomer: it does not mean the operator is unbounded on every vector, only that we do not assume a finite operator norm. Some unbounded operators happen to be bounded on their domain, but we deliberately allow the possibility that $\sup_{\|x\|=1, x \in \mathcal{D}(T)} \|Tx\| = \infty$.

Because $T$ is not defined everywhere and is not continuous, the usual topological tools break down. The replacement is the *graph* of $T$:
$$
\Gamma(T) = \\{(x, Tx) : x \in \mathcal{D}(T)\\} \subseteq H \oplus H.
$$

We say $T$ is **closed** if $\Gamma(T)$ is a closed subspace of $H \oplus H$ (with the product topology). Concretely, $T$ is closed if and only if whenever $(x_n)$ is a sequence in $\mathcal{D}(T)$ with $x_n \to x$ and $Tx_n \to y$, we have $x \in \mathcal{D}(T)$ and $Tx = y$.

Why is closedness the right condition? A bounded operator defined on a dense subspace extends uniquely by continuity to the whole space; closedness is not an issue. For unbounded operators we cannot extend to the whole space, but closedness gives us the next best thing: the graph is a well-behaved geometric object in $H \oplus H$, and many of the tools of bounded operator theory (resolvents, spectra, functional calculus) carry over to closed operators with only mild modifications.

**The closed graph theorem revisited.** Recall the closed graph theorem for bounded operators: if $T: H \to H$ is defined on all of $H$ and has a closed graph, then $T$ is bounded. For unbounded operators the contrapositive is instructive: if $T$ is unbounded, its graph cannot be simultaneously closed and defined on all of $H$. This is why unbounded operators *must* have restricted domains.

An operator $T$ is **closable** if the closure $\overline{\Gamma(T)}$ of its graph in $H \oplus H$ is itself the graph of an operator. When $T$ is closable, we write $\bar{T}$ for the operator whose graph is $\overline{\Gamma(T)}$ and call it the **closure** of $T$. Not every densely defined operator is closable --- if there is a sequence $x_n \to 0$ with $Tx_n \to y \neq 0$, then $\overline{\Gamma(T)}$ contains both $(0, 0)$ and $(0, y)$, so it is not the graph of a function. But symmetric operators (defined below) are always closable, which is one reason the symmetric/self-adjoint framework is so important.

**Example: differentiation.** Let $H = L^2[0,1]$ and $T = -i\frac{d}{dx}$ with domain $\mathcal{D}(T) = C^1[0,1]$. This operator is closable. Its closure has domain
$$
\mathcal{D}(\bar{T}) = \\{f \in L^2[0,1] : f \text{ is absolutely continuous and } f' \in L^2[0,1]\\}.
$$
Note that we have not imposed boundary conditions; the closure inherits whatever conditions the original domain imposes, which in this case is none.

## Symmetric operators versus self-adjoint operators

Here lies the subtlest and most consequential distinction in the theory of unbounded operators. Let $T$ be a densely defined operator on a Hilbert space $H$. The **adjoint** $T^*$ is defined as follows: $y \in \mathcal{D}(T^*)$ if and only if there exists $z \in H$ such that
$$
\langle Tx, y \rangle = \langle x, z \rangle \quad \text{for all } x \in \mathcal{D}(T).
$$
When such $z$ exists it is unique (because $\mathcal{D}(T)$ is dense), and we set $T^* y = z$.

We say $T$ is **symmetric** (or Hermitian) if $T \subseteq T^*$, meaning $\mathcal{D}(T) \subseteq \mathcal{D}(T^*)$ and $Tx = T^*x$ for all $x \in \mathcal{D}(T)$. Equivalently,
$$
\langle Tx, y \rangle = \langle x, Ty \rangle \quad \text{for all } x, y \in \mathcal{D}(T).
$$

We say $T$ is **self-adjoint** if $T = T^*$, meaning $\mathcal{D}(T) = \mathcal{D}(T^*)$ and $Tx = T^*x$ for all $x$ in this common domain.

The difference is entirely in the domains. A symmetric operator satisfies the formal adjoint relation on its domain, but $T^*$ might have a strictly larger domain. A self-adjoint operator satisfies the relation and has no room for $T^*$ to be defined on any additional vectors. This may sound like a pedantic distinction, but it has profound consequences:

1. **The spectral theorem holds for self-adjoint operators, not merely symmetric ones.** A symmetric operator need not have any spectral decomposition.
2. **A symmetric operator can have many self-adjoint extensions, exactly one, or none.** The deficiency indices (defined below) tell us which case we are in.
3. **Stone's theorem connects one-parameter unitary groups to self-adjoint operators, not symmetric ones.** If you want to generate a unitary time evolution $e^{itT}$, you need $T$ to be self-adjoint.

**Counterexample: symmetric but not self-adjoint.** Let $H = L^2[0,1]$ and define $T = -i\frac{d}{dx}$ with domain
$$
\mathcal{D}(T) = \\{f \in H^1[0,1] : f(0) = f(1) = 0\\}.
$$
Then $T$ is symmetric: integration by parts gives
$$
\langle Tf, g \rangle = -i\int_0^1 f'(x)\overline{g(x)}\,dx = -i\left[f\overline{g}\right]_0^1 + i\int_0^1 f(x)\overline{g'(x)}\,dx = \langle f, Tg \rangle,
$$
where the boundary term vanishes because $f(0)=f(1)=g(0)=g(1)=0$. But $T^*$ is the same differential expression $-i\frac{d}{dx}$ on the larger domain $H^1[0,1]$ (no boundary conditions), so $T \subsetneq T^*$. The operator $T$ is symmetric but not self-adjoint.

This operator has self-adjoint extensions parametrized by $\theta \in [0, 2\pi)$: define $T_\theta$ on the domain $\{f \in H^1[0,1] : f(1) = e^{i\theta}f(0)\}$. Each choice of $\theta$ gives a different self-adjoint operator with a different spectrum.

## Von Neumann deficiency indices

Given a closed symmetric operator $T$, how do we determine its self-adjoint extensions? The answer is given by von Neumann's theory of deficiency indices.

**Definition.** The *deficiency subspaces* of a closed symmetric operator $T$ are
$$
\mathcal{N}_+ = \ker(T^* - iI), \qquad \mathcal{N}_- = \ker(T^* + iI).
$$
The *deficiency indices* are $n_+ = \dim \mathcal{N}_+$ and $n_- = \dim \mathcal{N}_-$.

The intuition is as follows. If $T$ were self-adjoint, then $T - iI$ and $T + iI$ would both be surjective (because $\pm i$ would be in the resolvent set --- self-adjoint operators have real spectrum). The deficiency subspaces measure how far $\operatorname{ran}(T - iI)$ and $\operatorname{ran}(T + iI)$ are from being all of $H$.

**Von Neumann's theorem.** Let $T$ be a closed, densely defined, symmetric operator with deficiency indices $(n_+, n_-)$. Then:
1. $T$ is self-adjoint if and only if $n_+ = n_- = 0$.
2. $T$ has self-adjoint extensions if and only if $n_+ = n_-$.
3. When $n_+ = n_- = n$, the self-adjoint extensions of $T$ are in bijection with the unitary operators $U: \mathcal{N}_+ \to \mathcal{N}_-$. Each such $U$ determines an extension $T_U$ with domain
$$
\mathcal{D}(T_U) = \mathcal{D}(T) \dotplus \\{(\xi + U\xi) : \xi \in \mathcal{N}_+\\}
$$
(where $\dotplus$ denotes algebraic direct sum inside $\mathcal{D}(T^*)$) and $T_U$ acts as $T^*$ on this domain.

When $n_+ = n_- = n < \infty$, the set of self-adjoint extensions is parametrized by $U(n)$, the unitary group of an $n$-dimensional space. In our differentiation example, $n_+ = n_- = 1$, and $U(1) \cong S^1$, giving the one-parameter family of extensions $T_\theta$ described above.

When $n_+ \neq n_-$, the operator has no self-adjoint extensions at all. This is not just a theoretical curiosity --- it happens for the momentum operator on $L^2[0, \infty)$ with domain $\{f \in H^1[0,\infty) : f(0)=0\}$. Here $n_+ = 0$ and $n_- = 1$ (or vice versa, depending on convention), so no self-adjoint extension exists.

**Computing deficiency indices in practice.** For our differentiation example $T = -id/dx$ on $\{f \in H^1[0,1]: f(0)=f(1)=0\}$, the deficiency equations are $T^*\phi = \pm i\phi$, i.e., $-i\phi' = \pm i\phi$, giving $\phi' = \mp\phi$, so $\phi(x) = Ce^{\mp x}$. Both solutions are in $L^2[0,1]$ (and in $H^1[0,1]$ with no boundary conditions, which is the domain of $T^*$), so $\dim\mathcal{N}_+ = \dim\mathcal{N}_- = 1$. The self-adjoint extensions are parametrized by unitary maps $U: \mathbb{C} \to \mathbb{C}$, i.e., by $e^{i\theta}$, which is exactly the one-parameter family $T_\theta$ we described earlier.

For the half-line operator $T = -id/dx$ on $\{f \in H^1[0,\infty): f(0)=0\}$, the equation $-i\phi' = i\phi$ gives $\phi(x) = Ce^{-x} \in L^2[0,\infty)$, while $-i\phi' = -i\phi$ gives $\phi(x) = Ce^{x} \notin L^2[0,\infty)$. So $n_+ = 1$, $n_- = 0$, and there is no self-adjoint extension. Physically, this corresponds to the fact that a free particle on a half-line with an absorbing boundary at the origin does not have a well-defined momentum observable.

## The Friedrichs extension

Among all self-adjoint extensions (when they exist), there is often a canonical "best" choice. For *semibounded* operators, this is the **Friedrichs extension**.

**Definition.** A symmetric operator $T$ is *bounded below* (or semibounded) if there exists $c \in \mathbb{R}$ such that
$$
\langle Tx, x \rangle \geq c\|x\|^2 \quad \text{for all } x \in \mathcal{D}(T).
$$

The Friedrichs extension proceeds as follows. Without loss of generality assume $c > 0$ (shift $T$ if needed). The form $\mathfrak{t}(x,y) = \langle Tx, y \rangle$ is a densely defined, positive, symmetric sesquilinear form. Its closure $\bar{\mathfrak{t}}$ has a domain $\mathcal{D}[\bar{\mathfrak{t}}]$ that is a Hilbert space under the inner product $\mathfrak{t}(x,y) + \langle x, y \rangle$. By the Riesz representation theorem (applied to the inclusion $\mathcal{D}[\bar{\mathfrak{t}}] \hookrightarrow H$), there exists a unique self-adjoint operator $T_F$ with $\mathcal{D}(T_F) \subseteq \mathcal{D}[\bar{\mathfrak{t}}]$ such that
$$
\bar{\mathfrak{t}}(x, y) = \langle T_F x, y \rangle \quad \text{for all } x \in \mathcal{D}(T_F),\; y \in \mathcal{D}[\bar{\mathfrak{t}}].
$$
This $T_F$ is the Friedrichs extension.

**Key properties:**
- $T_F$ extends $T$ (i.e., $T \subseteq T_F$).
- $T_F$ is the unique self-adjoint extension of $T$ with $\mathcal{D}(T_F) \subseteq \mathcal{D}[\bar{\mathfrak{t}}]$.
- Among all self-adjoint extensions of $T$ that are bounded below by $c$, the Friedrichs extension has the largest lower bound.
- The Friedrichs extension is the "hardest" boundary condition in the sense that it corresponds to Dirichlet boundary conditions for differential operators.

**Example: Laplacian with Dirichlet conditions.** Let $\Omega \subseteq \mathbb{R}^n$ be a bounded open set with smooth boundary. Define $T = -\Delta$ on $\mathcal{D}(T) = C_c^\infty(\Omega)$. This is a positive symmetric operator. Its Friedrichs extension is the Dirichlet Laplacian $-\Delta_D$, defined on $H^2(\Omega) \cap H^1_0(\Omega)$. The form domain is $H^1_0(\Omega)$, and the operator domain consists of those $H^1_0$ functions that are also in $H^2$.

## Essential self-adjointness

An important special case arises when a symmetric operator has a unique self-adjoint extension. In this situation the physics is unambiguous: there is only one way to make the operator self-adjoint, so no boundary condition needs to be chosen.

**Definition.** A symmetric operator $T$ is **essentially self-adjoint** if its closure $\bar{T}$ is self-adjoint. Equivalently, $T$ has deficiency indices $(0,0)$ (after taking the closure, $T$ is already self-adjoint).

Essential self-adjointness means that the operator, though not self-adjoint on its original domain, is "close enough" that there is no ambiguity. The unique self-adjoint extension is the closure $\bar{T}$.

**Criteria for essential self-adjointness.** Several useful sufficient conditions exist:

1. **(Range criterion)** $T$ is essentially self-adjoint if and only if $\operatorname{ran}(T + iI)$ and $\operatorname{ran}(T - iI)$ are both dense in $H$.

2. **(Nelson's analytic vector theorem)** If $T$ is symmetric and has a dense set of analytic vectors (vectors $x$ for which $\sum_{n=0}^\infty \frac{t^n}{n!}\|T^n x\| < \infty$ for some $t > 0$), then $T$ is essentially self-adjoint.

3. **(Kato-Rellich theorem)** If $A$ is self-adjoint, $B$ is symmetric with $\mathcal{D}(A) \subseteq \mathcal{D}(B)$, and there exist $a < 1$ and $b \geq 0$ with $\|Bx\| \leq a\|Ax\| + b\|x\|$ for all $x \in \mathcal{D}(A)$, then $A + B$ is self-adjoint on $\mathcal{D}(A)$.

4. **(Rellich's criterion for Schrodinger operators)** $-\Delta + V$ on $C_c^\infty(\mathbb{R}^n)$ is essentially self-adjoint if $V \in L^2_{\text{loc}}(\mathbb{R}^n)$ is bounded below. More refined criteria (Kato's theorem) allow local singularities: for $n \leq 3$, $V \in L^2_{\text{loc}}$ suffices if $V$ is bounded below.

These criteria are not interchangeable; each applies in different situations. Nelson's theorem is particularly useful in quantum field theory, while Kato-Rellich is the workhorse for Schrodinger operators.

## The spectral theorem for unbounded self-adjoint operators

The spectral theorem extends from bounded to unbounded self-adjoint operators, but the formulation requires more care because the spectrum is no longer bounded.

**Theorem (Spectral theorem, unbounded case).** Let $T$ be a self-adjoint operator on a Hilbert space $H$. There exists a unique projection-valued measure $E$ on the Borel $\sigma$-algebra of $\mathbb{R}$ such that
$$
T = \int_{-\infty}^{\infty} \lambda \, dE(\lambda),
$$
where the integral converges in the strong operator topology, and the domain of $T$ is
$$
\mathcal{D}(T) = \left\\{x \in H : \int_{-\infty}^{\infty} \lambda^2 \, d\|E(\lambda)x\|^2 < \infty \right\\}.
$$

This means that for any $x \in \mathcal{D}(T)$ and $y \in H$,
$$
\langle Tx, y \rangle = \int_{-\infty}^{\infty} \lambda \, d\langle E(\lambda)x, y \rangle.
$$

The **functional calculus** extends to measurable functions: for any Borel function $f: \mathbb{R} \to \mathbb{C}$, we define
$$
f(T) = \int_{-\infty}^{\infty} f(\lambda) \, dE(\lambda)
$$
on the domain $\mathcal{D}(f(T)) = \{x \in H : \int |f(\lambda)|^2 \, d\|E(\lambda)x\|^2 < \infty\}$.

**Proof strategy.** The standard proof goes through the Cayley transform. Define the *Cayley transform* of $T$ by
$$
U = (T - iI)(T + iI)^{-1}.
$$
Because $T$ is self-adjoint, $T \pm iI$ are bijections from $\mathcal{D}(T)$ to $H$, and $U$ is a unitary operator on $H$. Moreover, $1$ is not an eigenvalue of $U$ (if $Ux = x$, then $(T-iI)y = (T+iI)y$ for $y = (T+iI)^{-1}x$, giving $-iy = iy$, so $y=0$). Conversely, any unitary $U$ with $1 \notin \sigma_p(U)$ arises as the Cayley transform of a unique self-adjoint operator.

Apply the spectral theorem for unitary operators (which follows from the spectral theorem for bounded normal operators) to write $U = \int_{|z|=1} z \, dF(z)$. The Mobius transformation $z \mapsto i\frac{1+z}{1-z}$ converts this into the spectral representation $T = \int \lambda \, dE(\lambda)$ on $\mathbb{R}$.

**Spectrum.** The spectrum of a self-adjoint operator is always a subset of $\mathbb{R}$. It decomposes into:
- The **point spectrum** $\sigma_p(T)$: eigenvalues.
- The **absolutely continuous spectrum** $\sigma_{ac}(T)$: related to scattering states in quantum mechanics.
- The **singular continuous spectrum** $\sigma_{sc}(T)$: exotic but physically relevant (e.g., almost-Mathieu operators).

The essential spectrum $\sigma_{ess}(T)$ is the set of $\lambda \in \sigma(T)$ that are either accumulation points of the spectrum or eigenvalues of infinite multiplicity. Weyl's theorem states that $\sigma_{ess}(T)$ is invariant under compact perturbations.

## Stone's theorem and one-parameter unitary groups

Self-adjoint operators and strongly continuous one-parameter unitary groups are two sides of the same coin. This is the content of Stone's theorem, which is fundamental to quantum mechanics (where $e^{-iHt}$ is the time evolution operator for Hamiltonian $H$).

**Definition.** A *strongly continuous one-parameter unitary group* is a map $U: \mathbb{R} \to B(H)$ such that:
- Each $U(t)$ is unitary.
- $U(0) = I$ and $U(s+t) = U(s)U(t)$ for all $s, t \in \mathbb{R}$.
- For each $x \in H$, $t \mapsto U(t)x$ is continuous.

**Stone's theorem.** There is a bijection between self-adjoint operators on $H$ and strongly continuous one-parameter unitary groups on $H$, given by:
- *Forward direction:* If $A$ is self-adjoint, then $U(t) = e^{itA} := \int e^{it\lambda} \, dE(\lambda)$ is a strongly continuous one-parameter unitary group.
- *Reverse direction:* If $\\{U(t)\\}$ is a strongly continuous one-parameter unitary group, then its generator
$$
A = \lim_{t \to 0} \frac{U(t) - I}{it}
$$
(defined on the domain where this limit exists in the strong sense) is a self-adjoint operator, and $U(t) = e^{itA}$.

**Proof sketch (forward direction).** Given the spectral decomposition $A = \int \lambda \, dE(\lambda)$, define $U(t) = \int e^{it\lambda}\,dE(\lambda)$. The group property follows from $e^{is\lambda}e^{it\lambda} = e^{i(s+t)\lambda}$. Unitarity follows from $|e^{it\lambda}|=1$. Strong continuity follows from dominated convergence: $\|U(t)x - x\|^2 = \int |e^{it\lambda}-1|^2\,d\|E(\lambda)x\|^2 \to 0$ as $t \to 0$.

**Proof sketch (reverse direction).** Given $\\{U(t)\\}$, the key step is showing that the domain $\mathcal{D}(A)$ of the generator is dense. This uses an averaging trick: for $x \in H$ and $\varphi \in C_c^\infty(\mathbb{R})$, define $x_\varphi = \int \varphi(t) U(t)x\,dt$. Then $x_\varphi \in \mathcal{D}(A)$ (differentiate under the integral), and by choosing $\varphi$ to approximate a delta function, $x_\varphi \to x$. Symmetry of $A$ follows from the fact that each $U(t)$ is unitary. The hard part is proving $A$ is self-adjoint, not merely symmetric; this follows from showing that $\operatorname{ran}(A \pm iI) = H$ using the Fourier--Laplace transform of $t \mapsto U(t)x$.

**Physical interpretation.** In quantum mechanics, observables are self-adjoint operators. The time evolution under Hamiltonian $H$ is $\psi(t) = e^{-iHt}\psi(0)$. Stone's theorem guarantees that this evolution is well-defined, unitary (probability-preserving), and determined by the self-adjointness of $H$. If $H$ is merely symmetric, we do not get a unique unitary evolution, and the physics is ill-defined.

## The Laplacian and the Schrodinger operator

The Laplacian $-\Delta$ is the canonical example of an unbounded self-adjoint operator, and the Schrodinger operator $-\Delta + V$ is its physically motivated generalization.

**The Laplacian on $\mathbb{R}^n$.** Define $T = -\Delta$ on $C_c^\infty(\mathbb{R}^n)$. This is a positive symmetric operator on $L^2(\mathbb{R}^n)$. It is essentially self-adjoint, and its unique self-adjoint extension (also denoted $-\Delta$) has domain $H^2(\mathbb{R}^n)$. The proof of essential self-adjointness is cleanest via the Fourier transform: $-\Delta$ becomes multiplication by $|\xi|^2$ in Fourier space, which is manifestly self-adjoint on $\{f \in L^2 : |\xi|^2 \hat{f} \in L^2\} = H^2(\mathbb{R}^n)$.

The spectrum of $-\Delta$ on $L^2(\mathbb{R}^n)$ is $[0, \infty)$, purely absolutely continuous (no eigenvalues). This is visible from the Fourier picture: the multiplication operator $M_{|\xi|^2}$ has spectrum $[0, \infty)$ with no point spectrum.

**The Laplacian on a bounded domain $\Omega$.** Here the situation is more interesting because different boundary conditions yield different self-adjoint extensions.

- *Dirichlet Laplacian* $-\Delta_D$: domain $H^2(\Omega) \cap H^1_0(\Omega)$. This is the Friedrichs extension of $-\Delta|_{C_c^\infty(\Omega)}$. Its spectrum is a sequence of eigenvalues $0 < \lambda_1 \leq \lambda_2 \leq \cdots \to \infty$, and the eigenfunctions form an orthonormal basis for $L^2(\Omega)$.

- *Neumann Laplacian* $-\Delta_N$: domain $\{u \in H^2(\Omega) : \partial u / \partial \nu = 0 \text{ on } \partial\Omega\}$. Also self-adjoint, with eigenvalues $0 = \mu_0 \leq \mu_1 \leq \cdots \to \infty$.

These are genuinely different operators with different spectra. The Dirichlet Laplacian has a spectral gap ($\lambda_1 > 0$), while the Neumann Laplacian always has $0$ as an eigenvalue (the constant function).

**Schrodinger operators.** Let $V: \mathbb{R}^n \to \mathbb{R}$ be a potential. The Schrodinger operator $H = -\Delta + V$ (formally) describes a quantum particle in the potential $V$. The key question is: for which $V$ is $H$ self-adjoint (or essentially self-adjoint) on a natural domain?

The Kato-Rellich theorem provides the main answer in physically relevant cases. We need $V$ to be a "small perturbation" of $-\Delta$ in the sense of relative boundedness.

**Theorem (Kato).** Let $n \leq 3$ and $V \in L^2(\mathbb{R}^n) + L^\infty(\mathbb{R}^n)$ (meaning $V = V_1 + V_2$ with $V_1 \in L^2$, $V_2 \in L^\infty$). Then $-\Delta + V$ is self-adjoint on $H^2(\mathbb{R}^n)$ and essentially self-adjoint on $C_c^\infty(\mathbb{R}^n)$.

This covers the Coulomb potential $V(x) = -Z/|x|$ in $\mathbb{R}^3$, which is physically the most important case (hydrogen atom). The Coulomb potential is in $L^2(\mathbb{R}^3) + L^\infty(\mathbb{R}^3)$: take $V_1 = V \cdot \mathbf{1}_{|x|<1}$ (which is in $L^2(\mathbb{R}^3)$ for $n=3$) and $V_2 = V \cdot \mathbf{1}_{|x|\geq 1}$ (which is bounded).

**The hydrogen atom.** For $H = -\Delta - 1/|x|$ on $L^2(\mathbb{R}^3)$, the spectrum consists of:
- Point spectrum: eigenvalues $E_n = -1/(4n^2)$ for $n = 1, 2, 3, \ldots$, each with multiplicity $n^2$.
- Absolutely continuous spectrum: $[0, \infty)$.
- No singular continuous spectrum.

The eigenvalues accumulate at $0$, and the eigenfunctions are the hydrogen orbitals familiar from chemistry. The continuous spectrum corresponds to scattering (ionized) states. This spectral structure was computed by Schrodinger in 1926 and constitutes one of the great triumphs of quantum mechanics.

## Further results and the road ahead

Several important results round out the theory of unbounded operators. Let me mention a few without full proofs.

**The RAGE theorem (Ruelle-Amrein-Georgescu-Enss).** This theorem connects the spectral decomposition of a self-adjoint operator $H$ to the dynamical behavior of $e^{-iHt}$:
- If $\psi$ is in the continuous spectral subspace of $H$, then $e^{-iHt}\psi$ "escapes to infinity" in the sense that for any compact operator $K$, $\frac{1}{T}\int_0^T \|Ke^{-iHt}\psi\|^2\,dt \to 0$.
- If $\psi$ is in the point spectral subspace, then $e^{-iHt}\psi$ remains "localized."

**Kato's inequality and diamagnetic inequality.** These tools extend essential self-adjointness results to magnetic Schrodinger operators $(-i\nabla - A)^2 + V$, where $A$ is a magnetic vector potential.

**Mourre theory.** A sophisticated framework for proving absence of singular continuous spectrum, based on positivity of commutators $i[H, A] \geq c > 0$ (in a suitable sense) on spectral subspaces.

**Weyl's asymptotic formula.** For the Dirichlet Laplacian on a bounded domain $\Omega \subseteq \mathbb{R}^n$, the eigenvalue counting function satisfies
$$
N(\lambda) := \#\\{k : \lambda_k \leq \lambda\\} \sim \frac{\omega_n}{(2\pi)^n} |\Omega| \cdot \lambda^{n/2} \quad \text{as } \lambda \to \infty,
$$
where $\omega_n$ is the volume of the unit ball in $\mathbb{R}^n$. This is a deep result connecting spectral theory to geometry ("Can you hear the shape of a drum?").

## What's next

We have entered the world of unbounded operators and seen that it is both richer and more treacherous than the bounded theory. The central lesson is that *domains matter*: the same formal expression (like $-i d/dx$ or $-\Delta$) can define many different self-adjoint operators depending on the domain, and each choice has physical consequences.

In the next article, we turn to **compact operators and the Fredholm alternative** (Article 10). Compact operators sit at the opposite extreme from unbounded operators --- they are, in a precise sense, the "smallest" operators in $B(H)$. Yet they play a central role in the theory of integral equations, in the study of eigenvalue problems, and as building blocks for the spectral theory of more general operators. We will prove the spectral theorem for compact self-adjoint operators (a warmup that we actually postponed until now) and develop the Fredholm theory that tells us when equations of the form $(I - K)x = y$ have solutions.

---

*This is Part 9 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 8 — Spectral Theory](/en/functional-analysis/08-spectral-theory/)*

*Next: [Part 10 — Semigroups of Operators](/en/functional-analysis/10-semigroups/)*
