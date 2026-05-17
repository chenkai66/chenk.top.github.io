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

Two articles ago I was talking about how spectral theory is the linear-algebraic infrastructure of quantum mechanics. The trouble is that nearly every operator a physicist actually cares about — the position operator, the momentum operator, the Laplacian, the Schrödinger Hamiltonian — is *not bounded*. They are not defined on the whole Hilbert space. They are densely defined, with domains that depend on the regularity or decay of the input function. None of the previous spectral apparatus applies directly. We need to extend it.

The extension is delicate. With unbounded operators, simply writing down "$T = T^*$" no longer makes sense, because the two sides have different domains. There is a real distinction between *symmetric* operators (where $\langle T x, y \rangle = \langle x, T y \rangle$ on the common domain) and *self-adjoint* operators (where additionally the domain of $T$ equals the domain of $T^*$). For bounded operators these notions coincide; for unbounded ones they fall apart in subtle ways, and entire books have been written about the difference. The reward for handling this carefully is that the spectral theorem, the functional calculus, and Stone's theorem all extend — and we get to actually do quantum mechanics rigorously. This article is a walk through the technical landscape.

## Domains, Domains, Domains

An **unbounded operator** on a Hilbert space $H$ is a linear map $T: D(T) \to H$ where the **domain** $D(T)$ is a (typically dense) linear subspace of $H$. The map need not be defined on all of $H$; the domain is part of the data. Two operators with the same formula but different domains are different operators.

![Unbounded operators with their dense but proper domain](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_v2_09_1_unbounded_domain.png)

Concrete example. Consider $T = -i d/dx$ on $L^2[0, 1]$. There are several reasonable choices of domain:

- $D_{\max} = \{f \in L^2 : f \text{ absolutely continuous, } f' \in L^2\}$, with no boundary conditions.
- $D_{\text{Dir}} = \{f \in D_{\max} : f(0) = f(1) = 0\}$, with Dirichlet boundary conditions.
- $D_{\text{per}} = \{f \in D_{\max} : f(0) = f(1)\}$, with periodic boundary conditions.
- $D_{C^\infty_c} = $ compactly supported smooth functions in $(0, 1)$, the "minimal" domain.

These four choices give *four different operators*, even though the formula $-i d/dx$ is the same. They have different spectra, different self-adjointness properties, and different physical interpretations. Periodic boundary conditions give a self-adjoint operator with discrete spectrum $\{2\pi n : n \in \mathbb{Z}\}$. Dirichlet conditions give a symmetric but not self-adjoint operator. The minimal domain gives a symmetric operator with two-dimensional defect, and the maximal domain gives an operator that is not even symmetric.

This sensitivity to domain choice is what makes unbounded operator theory annoying and what makes it interesting. The choice of domain encodes physical content (boundary conditions, decay at infinity), and the same differential expression can describe several physically distinct systems depending on the domain. Almost every paradox in mathematical physics where someone "computes the spectrum two ways and gets different answers" is really a story about implicit domain choices.

## Closed Operators and the Closed Graph

The fundamental regularity property for unbounded operators is *closedness*. The graph of $T$ is

$$ G(T) = \{(x, Tx) : x \in D(T)\} \subset H \times H. $$

We say $T$ is **closed** if $G(T)$ is a closed subspace of $H \times H$. Equivalently, $T$ is closed iff: whenever $x_n \in D(T)$, $x_n \to x$, and $T x_n \to y$, we have $x \in D(T)$ and $T x = y$.

![Closed operator: graph is closed in the product topology](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_v2_09_2_closed_op.png)

Closed is weaker than continuous (= bounded) but stronger than just "linear." For bounded operators on a Banach space, the closed graph theorem says closed = continuous. For unbounded operators, closedness is a substantive condition that allows quite a bit of analysis to work — for example, the spectrum of a closed operator is well-defined as a subset of $\mathbb{C}$.

**Numerical-flavored example.** Take $T f = f'$ on $L^2[0, 1]$ with $D(T) = C^1[0, 1]$. The sequence $f_n(x) = x^{1/2 + 1/n}$ is in $D(T)$ and converges in $L^2$ to $f(x) = x^{1/2}$, while $T f_n = (1/2 + 1/n) x^{-1/2 + 1/n}$ converges in $L^2$ to $g(x) = (1/2) x^{-1/2}$. But $f \notin C^1[0, 1]$, so $T$ defined this way is not closed. The right move is to take its **closure**: the smallest closed extension. For a closable operator (one whose graph is closed in the limit), this is the closed operator obtained by including all limits $(x, y)$ such that $(x_n, T x_n) \to (x, y)$. The closure of $d/dx$ on $C^1$ is $d/dx$ on the Sobolev space $H^1$, which is closed.

A symmetric operator is always closable (its closure is also symmetric), but not every linear operator with a dense domain is closable. The pathological cases involve the closure of the graph containing $(0, y)$ for some nonzero $y$, which would force the closure to map $0$ to two different things. We will assume closability throughout.

## The Adjoint and the Domain $D(T^*)$

The adjoint of an unbounded operator is more delicate than the bounded case. Define

$$ D(T^*) = \{y \in H : x \mapsto \langle T x, y \rangle \text{ is bounded on } D(T)\}, $$

and for $y \in D(T^*)$, $T^* y$ is the unique vector with $\langle T x, y \rangle = \langle x, T^* y \rangle$ for all $x \in D(T)$. The existence of $T^* y$ uses Riesz representation applied to the bounded functional $x \mapsto \langle T x, y \rangle$.

The adjoint is *always* a closed operator, even when $T$ is not. But $D(T^*)$ depends sensitively on $D(T)$, and the larger $D(T)$ is, the smaller $D(T^*)$ tends to be.

A subtle but useful identity: $G(T^*) = \{(y, z) : \langle z, x \rangle - \langle y, T x \rangle = 0 \text{ for all } x \in D(T)\}$, which is the orthogonal complement (under a specific symplectic-flavored pairing) of $\{(x, -T x) : x \in D(T)\} \subset H \times H$. The closedness of $T^*$ is then immediate from this characterization.

## Symmetric vs Self-Adjoint

This is the pivotal distinction. A densely defined operator $T$ is **symmetric** if $\langle T x, y \rangle = \langle x, T y \rangle$ for all $x, y \in D(T)$. Equivalently: $T \subset T^*$, meaning $D(T) \subset D(T^*)$ and $T = T^*$ on $D(T)$.

The operator $T$ is **self-adjoint** if $T = T^*$ as operators, meaning $D(T) = D(T^*)$ and the actions agree.

![Symmetric vs self-adjoint operators: domains of T and T*](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_v2_09_3_sym_vs_sa.png)

Symmetric is a much weaker condition than self-adjoint. For bounded operators they agree, but for unbounded operators a symmetric operator can have many different self-adjoint extensions, or none at all. The spectral theorem and the functional calculus require *self*-adjointness; mere symmetry is not enough.

**Why does this matter?** Because of Stone's theorem (next article), self-adjoint operators generate one-parameter unitary groups: $T = T^*$ implies $e^{-itT}$ is a unitary group on $H$ for all $t \in \mathbb{R}$. Symmetric but not self-adjoint operators do not generate such groups, and the corresponding "time evolution" is ambiguous. In quantum mechanics, this means: for an observable to have a well-defined associated unitary symmetry (energy and time translation, momentum and space translation), the observable must be self-adjoint, not merely symmetric. This is a non-negotiable physical constraint.

The classical example of a symmetric but not self-adjoint operator is $T = -i d/dx$ on $L^2[0, 1]$ with domain $C^\infty_c(0, 1)$ (compactly supported smooth functions). $T$ is symmetric: integration by parts works without boundary terms. But $T^*$ is the same differential operator with a *larger* domain (functions with $f(0)$ and $f(1)$ not necessarily zero), so $D(T) \subsetneq D(T^*)$. The closure $\overline{T}$ is also symmetric but not self-adjoint, with $\overline{T} \neq T^{**}$. To get a self-adjoint operator, one has to specify boundary conditions: periodic, or $e^{i\theta}$-twisted ($f(1) = e^{i\theta} f(0)$). Different boundary conditions give different self-adjoint extensions.

## Deficiency Indices and the von Neumann Theorem

How do we tell whether a symmetric operator has self-adjoint extensions, and how many? The answer is the von Neumann theory of deficiency indices.

For a closed symmetric operator $T$, define the **deficiency subspaces**

$$ \mathcal{N}_\pm = \ker(T^* \mp i I) = \text{range}(T \pm i I)^\perp. $$

The dimensions $n_\pm = \dim \mathcal{N}_\pm$ are the **deficiency indices**. The von Neumann theorem says:

- $T$ is self-adjoint iff $n_+ = n_- = 0$.
- $T$ has self-adjoint extensions iff $n_+ = n_-$. The extensions are parameterized by unitary maps $\mathcal{N}_+ \to \mathcal{N}_-$.
- If $n_+ \neq n_-$, $T$ has no self-adjoint extensions.

For $-i d/dx$ on $C^\infty_c(0, 1)$, $n_+ = n_- = 1$ (the deficiency subspaces are spanned by $e^{\pm x}$ restricted to $[0, 1]$, or more precisely the unique solutions to $\mp f' = i f$ that are in $L^2$). So there is a one-parameter family of self-adjoint extensions, parameterized by $U(1)$, corresponding to the boundary conditions $f(1) = e^{i\theta} f(0)$.

For $-i d/dx$ on $C^\infty_c(0, \infty)$, the situation is asymmetric: only one of $\mathcal{N}_\pm$ is nontrivial, so $n_+ \neq n_-$, and there are no self-adjoint extensions. The momentum operator on the half-line is fundamentally not self-adjoint, regardless of boundary conditions. Physicists say this is because there is no self-adjoint momentum operator for a particle on a half-line, only a self-adjoint Hamiltonian. This is a real physical statement, not just a technicality.

## The Friedrichs Extension

For a particularly important class of symmetric operators — the **semibounded** ones, with $\langle T x, x \rangle \geq c \|x\|^2$ for some $c \in \mathbb{R}$ — there is a canonical self-adjoint extension, the **Friedrichs extension**.

![Friedrichs extension turning a semibounded symmetric operator into a self-adjoint one](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_v2_09_5_friedrichs.png)

The construction: start with the quadratic form $q(x) = \langle T x, x \rangle$ on $D(T)$. Complete $D(T)$ in the norm $\|x\|_q = \sqrt{q(x) + (1 - c)\|x\|^2}$ to get a Hilbert space $V$ with $V \subset H$ continuously. The Friedrichs extension $T_F$ is then the operator with domain $\{x \in V : q(\cdot, x) \text{ is bounded on } V \text{ in } H\text{-norm}\}$, defined via the Riesz representation theorem applied to the bounded form. This extension is self-adjoint and has the same lower bound as $T$.

For the Laplacian $-\Delta$ on $C^\infty_c(\Omega)$ for a bounded domain $\Omega \subset \mathbb{R}^n$, the Friedrichs extension is $-\Delta$ with **Dirichlet boundary conditions** ($f|_{\partial \Omega} = 0$), with domain $H^1_0(\Omega) \cap H^2(\Omega)$. The associated quadratic form is $q(f) = \int_\Omega |\nabla f|^2$, and the Friedrichs extension corresponds to "minimizing the quadratic form among functions vanishing on the boundary." This is the right self-adjoint extension for the Dirichlet Laplacian, the one used in PDE and physics.

The Friedrichs extension is conceptually beautiful because it shows that for semibounded operators, the choice of domain is forced by the variational principle. The "physical" boundary conditions emerge automatically from the quadratic form, not from any extra choice. This is why elliptic boundary value problems work out so cleanly — the Lax-Milgram framework, which we will develop in article 12, is essentially the Friedrichs extension done one-shot.

## A Worked Example: The Particle in a Box

For tangibility, let me work through one example end-to-end. Consider the Hamiltonian $H = -\frac{1}{2} d^2/dx^2$ on $L^2[0, 1]$, the kinetic energy operator for a quantum particle confined to the unit interval. The differential expression is the same in every case; the physics depends entirely on what we choose for the domain (i.e., what boundary conditions).

**Dirichlet boundary conditions** (infinite well): $D(H) = \{f \in H^2[0, 1] : f(0) = f(1) = 0\}$. This is the Friedrichs extension of $H$ on $C^\infty_c(0, 1)$, since the quadratic form is $q(f) = (1/2) \int_0^1 |f'|^2$ and the natural completion gives $H^1_0[0, 1]$. The eigenfunctions are $\phi_n(x) = \sqrt{2} \sin(n\pi x)$ with eigenvalues $E_n = n^2 \pi^2 / 2$. Pure point spectrum, no continuous part.

**Neumann boundary conditions**: $D(H) = \{f \in H^2[0, 1] : f'(0) = f'(1) = 0\}$. Eigenfunctions $\phi_n(x) = \sqrt{2} \cos(n\pi x)$ for $n \geq 1$ together with $\phi_0 = 1$, eigenvalues $E_n = n^2 \pi^2 / 2$ for $n \geq 0$. Same essential spectrum as Dirichlet but with a zero eigenvalue corresponding to the constant function.

**Periodic boundary conditions**: $D(H) = \{f \in H^2 : f(0) = f(1), f'(0) = f'(1)\}$. Eigenfunctions $e^{2\pi i n x}$ for $n \in \mathbb{Z}$, eigenvalues $E_n = (2\pi n)^2/2$. Each nonzero $E_n$ has multiplicity 2 (from $\pm n$), zero eigenvalue is simple.

**Robin boundary conditions** $f'(0) = \alpha f(0)$, $f'(1) = -\alpha f(1)$: a one-parameter family of self-adjoint extensions parameterized by $\alpha \in \mathbb{R}$, with spectra varying continuously in $\alpha$.

The four extensions give four different physical systems, each correct in its own context. Same differential operator, four different spectra, four different time evolutions. This is what "domains encode physics" means in concrete terms.

## When the Particle Lives on a Half-Line

A more pathological example. Consider $T = -i d/dx$ on $L^2(0, \infty)$ with domain $C^\infty_c(0, \infty)$. The deficiency indices are $n_+ = 1$ (the function $e^{-x}$ is in $\ker(T^* - i)$) and $n_- = 0$ (no $L^2$ function satisfies $-i f' = -i f$ on $(0, \infty)$, since $e^x$ is not in $L^2$). So $n_+ \neq n_-$, and the von Neumann theorem says $T$ has no self-adjoint extensions.

Physically: there is no momentum operator for a particle confined to the half-line. Translation invariance is broken by the boundary at $x = 0$, and momentum is the generator of translations, so there is no observable corresponding to momentum on the half-line. Every quantum mechanics textbook handles this implicitly by working on the full line and projecting, or by working with even/odd extensions, but the underlying functional-analytic obstruction is real.

The Hamiltonian on the half-line, on the other hand, *does* have self-adjoint extensions: the operator $-d^2/dx^2$ on $C^\infty_c(0, \infty)$ has deficiency indices $(1, 1)$ — both $e^{-(1+i)x/\sqrt{2}}$ and $e^{-(1-i)x/\sqrt{2}}$ are in $L^2$ — and the self-adjoint extensions form a one-parameter family parameterized by boundary conditions $\cos\alpha \cdot f(0) - \sin\alpha \cdot f'(0) = 0$. So the half-line has Hamiltonians but no momentum operator, and that asymmetry is physically real.

## A Concrete Spectral Computation: The Harmonic Oscillator, in Detail

The harmonic oscillator $H = -\frac{1}{2} d^2/dx^2 + \frac{1}{2} x^2$ on $L^2(\mathbb{R})$ deserves a worked-through diagonalization, both because the explicit eigenfunctions are useful and because the operator-theoretic technique generalizes.

Define the **annihilation** and **creation** operators

$$ a = \frac{1}{\sqrt{2}}(x + d/dx), \qquad a^* = \frac{1}{\sqrt{2}}(x - d/dx). $$

A direct computation gives $[a, a^*] = I$ and $H = a^* a + 1/2$ (or $a a^* - 1/2$). The vacuum state $\phi_0(x) = \pi^{-1/4} e^{-x^2/2}$ satisfies $a \phi_0 = 0$, hence $H \phi_0 = (1/2) \phi_0$. Apply $a^*$ repeatedly: $\phi_n = (a^*)^n \phi_0 / \sqrt{n!}$ are orthonormal eigenfunctions with $H \phi_n = (n + 1/2) \phi_n$. They are the Hermite functions, normalized.

The spectral measure is $E = \sum_{n=0}^\infty |\phi_n\rangle \langle \phi_n| \delta_{n + 1/2}$, atomic on $\{1/2, 3/2, 5/2, \ldots\}$. The functional calculus gives, for any continuous $f$,

$$ f(H) = \sum_{n=0}^\infty f(n + 1/2) \, |\phi_n\rangle \langle \phi_n|, $$

and in particular $e^{-i t H}$ is a unitary group on $L^2(\mathbb{R})$ with explicit kernel via Mehler's formula. This explicit diagonalization is the reason the harmonic oscillator is the workhorse example of every quantum mechanics course; it also generalizes (via Bargmann transform, coherent states, Hermite expansion) into the analysis of pseudodifferential operators and microlocal analysis. From the abstract spectral theorem to the explicit Hermite expansion, every step is concrete.

## The Spectral Theorem for Unbounded Self-Adjoint Operators

The good news: once one has a self-adjoint operator, the entire spectral theory of article 8 applies, with the modification that the spectrum can now be unbounded.

**Theorem.** Let $T$ be self-adjoint on a Hilbert space $H$ (possibly unbounded). There exists a unique projection-valued measure $E$ on the Borel sets of $\mathbb{R}$ such that

$$ T = \int_\mathbb{R} \lambda \, dE(\lambda), $$

with $D(T) = \{x : \int_\mathbb{R} \lambda^2 \, d \langle E(\lambda) x, x \rangle < \infty\}$. The support of $E$ is exactly $\sigma(T)$.

The spectral measure form, the multiplication operator form, and the functional calculus all extend with the appropriate domain bookkeeping. The Borel functional calculus gives $f(T)$ for any bounded Borel function on $\sigma(T)$, defined via $f(T) = \int f(\lambda) \, dE(\lambda)$. For unbounded $f$, one specifies the domain $D(f(T)) = \{x : \int |f(\lambda)|^2 d \langle E(\lambda) x, x \rangle < \infty\}$.

This extension is what makes spectral analysis of differential operators possible. The Laplacian, the Schrödinger operator, the Dirac operator — all are self-adjoint on appropriate domains, and all have spectral measures one can in principle compute. In practice, computing the spectral measure explicitly is hard outside of special cases (free Hamiltonians, harmonic oscillator, hydrogen atom), but the structural existence is what allows abstract arguments to proceed.

## The Laplacian on $L^2(\mathbb{R}^n)$

The standard example. Let $T = -\Delta = -\sum \partial^2/\partial x_j^2$ on $L^2(\mathbb{R}^n)$, with domain $C^\infty_c(\mathbb{R}^n)$. The closure (and self-adjoint extension) has domain $H^2(\mathbb{R}^n)$, the Sobolev space of $L^2$ functions whose distributional second derivatives are in $L^2$.

![The Laplacian as an unbounded self-adjoint operator on L^2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_v2_09_4_laplacian.png)

Via the Fourier transform $\mathcal{F}: L^2(\mathbb{R}^n) \to L^2(\mathbb{R}^n)$, the Laplacian transforms to multiplication by $|\xi|^2$:

$$ \widehat{(-\Delta f)}(\xi) = |\xi|^2 \hat f(\xi). $$

So the Fourier transform unitarily diagonalizes $-\Delta$ as a multiplication operator. The spectrum is $[0, \infty)$, all continuous (since multiplication by $|\xi|^2$ on $L^2(\mathbb{R}^n)$ has no eigenfunctions — no $L^2$ function is a delta function on a level set of $|\xi|^2$).

The spectral measure is, in this representation, $E(B) = \mathcal{F}^{-1} M_{\mathbf{1}_{|\xi|^2 \in B}} \mathcal{F}$. The "eigenfunctions" $e^{i\xi \cdot x}$ are not $L^2$ — they are *generalized eigenfunctions* in the sense of distributions. The proper formulation is via the spectral measure.

This is the rigorous basis for everything physicists do with plane waves. When they expand a wavefunction in a Fourier integral, they are using the spectral resolution of $-\Delta$. The fact that the plane waves are not in $L^2$ is the mathematical content of "particles with definite momentum are not normalizable states."

## Discrete vs. Essential Spectrum

For unbounded self-adjoint operators, the spectrum splits in a slightly different way than for bounded ones. Define:

- $\sigma_d(T)$ = **discrete spectrum** = isolated eigenvalues of finite multiplicity.
- $\sigma_{ess}(T)$ = **essential spectrum** = $\sigma(T) \setminus \sigma_d(T)$.

![Essential spectrum vs discrete spectrum](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_v2_09_6_essential.png)

The essential spectrum is invariant under compact perturbations: $\sigma_{ess}(T + K) = \sigma_{ess}(T)$ for any compact self-adjoint $K$. This is **Weyl's theorem**, and it is the foundation of perturbation theory for differential operators.

The discrete spectrum, on the other hand, can change under perturbations. In quantum mechanics, $\sigma_d(H)$ corresponds to the bound states of the system (the discrete energy levels), and $\sigma_{ess}(H)$ to the scattering or continuous-spectrum states (free states at high energy). For a Schrödinger operator $-\Delta + V$ on $L^2(\mathbb{R}^n)$ with $V$ decaying at infinity, $\sigma_{ess}(-\Delta + V) = [0, \infty) = \sigma(-\Delta)$ by Weyl's theorem, but the discrete spectrum below zero captures the bound states (electron in a hydrogen atom, etc.).

A worked numerical example: the hydrogen atom Hamiltonian $H = -\Delta - 1/|x|$ on $L^2(\mathbb{R}^3)$. Self-adjoint with domain $H^2(\mathbb{R}^3)$ (the Coulomb potential is short-range enough relative to $-\Delta$ for self-adjointness via Kato-Rellich). Discrete spectrum: $\{-1/(4n^2) : n = 1, 2, 3, \ldots\}$, the famous Bohr energy levels (with multiplicity $n^2$). Essential spectrum: $[0, \infty)$. The picture is exactly what physicists draw: a sequence of discrete energy levels accumulating at zero, then a continuous spectrum above zero.

## A Few Common Confusions Worth Naming

**Confusion 1: "Symmetric implies self-adjoint."** No. Symmetric is much weaker. The Hellinger-Toeplitz theorem says that if a symmetric operator is everywhere defined on a Hilbert space, it is bounded — so a genuinely unbounded symmetric operator is *necessarily* defined only on a proper subspace, and the question of self-adjointness becomes nontrivial.

**Confusion 2: "The closure of a symmetric operator is self-adjoint."** No. The closure of a symmetric operator is symmetric, but in general not self-adjoint. The deficiency indices may be nonzero. To get a self-adjoint extension one has to *enlarge* the domain (via boundary conditions or Friedrichs), not just close it.

**Confusion 3: "The spectrum is the same for all self-adjoint extensions."** No. Different self-adjoint extensions have different spectra. The Dirichlet, Neumann, and periodic Laplacians on $[0, 1]$ all have different spectra, even though they share the same differential expression.

**Confusion 4: "Adjoint and Hermitian conjugate are the same thing."** For matrices and bounded operators, yes. For unbounded operators, the adjoint is more subtle because the domain has to be defined carefully. The "Hermitian conjugate" in physics typically refers to the formal differential adjoint (matching boundary terms via integration by parts), which equals the operator-theoretic adjoint only on a specific domain.

**Confusion 5: "Every densely defined symmetric operator has self-adjoint extensions."** False, as the half-line momentum example showed. The deficiency indices $(n_+, n_-)$ must be equal.

These confusions are not pedantic. Each has been the source of genuine mistakes in physics papers; the literature on the "self-adjointness problem" for various Hamiltonians, especially in QED and quantum field theory, is full of subtle errors that came from mishandling domains. Functional analysis is, in this respect, less a luxury than a quality-control procedure.

## Trotter Product Formula and Applications

A useful tool that uses unbounded operators directly: the **Trotter product formula**. If $A$ and $B$ are self-adjoint and $A + B$ is essentially self-adjoint on $D(A) \cap D(B)$, then

$$ e^{-it(A+B)} = \lim_{n \to \infty} \left( e^{-itA/n} e^{-itB/n} \right)^n, $$

with the limit in the strong operator topology. This factorizes the time evolution of a sum of operators into alternating evolutions of each one. In quantum mechanics, where typically $A = -\Delta/2$ (kinetic) and $B = V$ (potential), the formula says: time evolution can be approximated by alternating "free propagation" and "potential pickup."

This has both theoretical and computational uses. On the theory side, Trotter's formula plus the Feynman-Kac formula give the Wiener-process representation of the heat semigroup with potential, which is the analytic foundation of stochastic methods in quantum field theory. On the computational side, the formula is the basis of **split-step methods** in numerical PDE — for the Schrödinger equation, alternating linear (Fourier-space) and nonlinear (real-space) updates, a workhorse algorithm in optical fiber simulation, BEC dynamics, and many other areas.

The formula's existence depends on a self-adjointness fact: $A + B$ being essentially self-adjoint on $D(A) \cap D(B)$, plus a careful resolvent estimate. For Schrödinger operators with reasonable potentials, the conditions are met by the Kato-Rellich theorem. The whole apparatus stands or falls with the underlying self-adjointness theory we have been building.

## Quantum Mechanics: Position, Momentum, and the Hamiltonian

Let me catalog the canonical operators in QM, with their domains.

![Position, momentum, and Hamiltonian operators in quantum mechanics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/09-unbounded-operators/fa_v2_09_7_qm_examples.png)

**Position operator $X$ on $L^2(\mathbb{R})$:** $(X f)(x) = x f(x)$, with $D(X) = \{f \in L^2 : x f \in L^2\}$. Self-adjoint, spectrum $\mathbb{R}$, all continuous. The spectral measure is multiplication by indicators: $E(B) f = \mathbf{1}_B f$. Eigenfunctions don't exist in $L^2$; the formal "delta function eigenstates" $\delta(x - a)$ are distributions, not $L^2$ functions.

**Momentum operator $P$ on $L^2(\mathbb{R})$:** $P = -i d/dx$, with domain $H^1(\mathbb{R})$. Self-adjoint via Fourier transform: $\hat P = M_\xi$, multiplication by $\xi$. Spectrum $\mathbb{R}$, all continuous. The "eigenfunctions" $e^{i\xi x}$ are not $L^2$.

**Commutator relation:** $[X, P] = X P - P X = i I$, on the common domain $D(XP) \cap D(PX) = \{f : f, x f, f', x f', \in L^2\}$, which contains the Schwartz space $\mathcal{S}(\mathbb{R})$. This is the canonical commutation relation, the algebraic heart of quantum mechanics. A subtle point: this commutation cannot be realized by *bounded* operators on a Hilbert space (taking traces would give $0 = \text{tr}([X, P]) = i \text{tr}(I)$, impossible). It only makes sense for unbounded operators.

**Harmonic oscillator Hamiltonian:** $H = -\frac{1}{2} d^2/dx^2 + \frac{1}{2} x^2$ on $L^2(\mathbb{R})$. Self-adjoint with domain $\{f \in H^2 : x^2 f \in L^2\}$. Spectrum: $\{n + 1/2 : n = 0, 1, 2, \ldots\}$ — pure point spectrum, the famous "$\hbar \omega (n + 1/2)$" energy levels. Eigenfunctions: the Hermite functions $H_n(x) e^{-x^2/2}$, which form an orthonormal basis of $L^2(\mathbb{R})$. This is one of the cleanest self-adjoint operators in quantum mechanics, with explicitly diagonalizable spectrum.

The harmonic oscillator is the model example of a self-adjoint operator with discrete spectrum. The hydrogen atom Hamiltonian above is the model example with discrete spectrum at low energies and continuous spectrum at high energies. Together they cover most of what one needs in introductory QM.

## Why This Matters: Existence of Time Evolution

The most important consequence of self-adjointness is **Stone's theorem**: a closed densely defined operator $T$ generates a strongly continuous one-parameter unitary group $U(t) = e^{-itT}$ if and only if $T$ is self-adjoint. We will prove this in article 12 in the form of "every one-parameter unitary group has a self-adjoint generator." The reverse implication is what the spectral theorem gives us: $e^{-itT} = \int e^{-it\lambda} dE(\lambda)$ defines a unitary group whenever $T = T^*$.

For an observable $A$ in quantum mechanics, $e^{-i s A}$ is the unitary symmetry generated by $A$: time evolution from the Hamiltonian, space translation from momentum, rotation from angular momentum. The mathematical statement that observables are self-adjoint is equivalent to the physical statement that they generate continuous symmetries. This is not a coincidence; it is the operator-theoretic content of Noether's theorem, in disguise.

If one tries to use a merely symmetric operator, one finds that $e^{-itT}$ cannot be defined as a unitary for all $t$ — the group property fails. Self-adjointness is precisely the condition for a sensible time evolution to exist. This is why so much of mathematical physics is about proving that a particular differential operator is self-adjoint on a particular domain.

## Numerical Spectral Computation: A Brief Note

In practice, spectra of unbounded self-adjoint operators are computed by truncation and discretization. For a Schrödinger operator on $\mathbb{R}^n$, one truncates to a large box $[-L, L]^n$, discretizes on a grid, and diagonalizes the resulting matrix. The discrete eigenvalues converge to the discrete eigenvalues of the original operator (under mild assumptions), and the essential spectrum is approximated by clusters of densely packed numerical eigenvalues. The convergence theory is the subject of **spectral approximation theory**, and the key technical tool is the **Weyl criterion**: $\lambda \in \sigma_{ess}(T)$ iff there exists a Weyl sequence $x_n \in D(T)$ with $\|x_n\| = 1$, $x_n \rightharpoonup 0$ weakly, and $(T - \lambda) x_n \to 0$ in norm. This condition is exactly what survives discretization.

Software packages like SLEPc and ARPACK provide the numerical infrastructure. The mathematical infrastructure is the spectral theorem for unbounded self-adjoint operators applied to the discretized problem and a careful limit argument as the discretization is refined.

## Practical Self-Adjointness Tests

How does one prove a particular operator is self-adjoint? A few standard tools.

**(a) Closed and symmetric with $\text{range}(T \pm i I) = H$.** This is the basic criterion: $T$ is self-adjoint iff $T$ is closed, symmetric, and the range condition holds. Equivalently, iff the deficiency indices $n_\pm = 0$.

**(b) Symmetric with a self-adjoint extension via Friedrichs.** For semibounded symmetric operators, the Friedrichs extension exists canonically and is self-adjoint.

**(c) Kato-Rellich theorem.** If $T_0$ is self-adjoint and $V$ is symmetric with $D(V) \supset D(T_0)$ and $\|V f\| \leq a \|T_0 f\| + b \|f\|$ for some $a < 1$, then $T_0 + V$ is self-adjoint on $D(T_0)$. This is the workhorse theorem for proving that Schrödinger operators $-\Delta + V$ are self-adjoint, for a wide class of potentials $V$.

**(d) Stone's theorem applied in reverse.** If a one-parameter unitary group $U(t)$ is given (e.g., from a physical symmetry), its generator is automatically self-adjoint, by Stone's theorem.

In practice, (c) covers most cases of physical interest. The Coulomb potential $V(x) = -1/|x|$ on $\mathbb{R}^3$ satisfies the Kato-Rellich hypothesis with $a < 1$ (this requires a Sobolev embedding argument), so $-\Delta - 1/|x|$ is self-adjoint on the Sobolev space $H^2(\mathbb{R}^3)$. The same logic handles a wide range of atomic and molecular Hamiltonians.

## A Worked Example: Self-Adjointness of $-\Delta + V$

Let me run through one of the most useful self-adjointness theorems explicitly. Suppose we want to show that the Schrödinger operator $H = -\Delta + V$ is self-adjoint on $H^2(\mathbb{R}^3)$, the standard domain of the kinetic energy operator $-\Delta$, when $V$ is, say, the Coulomb potential $V(x) = -1/|x|$.

The Kato-Rellich theorem says: if $T_0$ is self-adjoint on $D(T_0)$, $V$ is symmetric on a domain containing $D(T_0)$, and $\|V f\| \leq a \|T_0 f\| + b \|f\|$ for all $f \in D(T_0)$ with some constants $a < 1$ and $b \geq 0$, then $T_0 + V$ is self-adjoint on $D(T_0)$. The Coulomb potential satisfies this with $T_0 = -\Delta$, $a < 1$, by the **Hardy inequality**

$$ \int_{\mathbb{R}^3} \frac{|f|^2}{|x|^2} dx \leq 4 \int_{\mathbb{R}^3} |\nabla f|^2 dx, $$

which after Cauchy-Schwarz gives $\| f/|x| \|_{L^2} \leq 2 \|\nabla f\|_{L^2}$, and a Sobolev inequality bounds $\|\nabla f\|_{L^2}$ by $\|\Delta f\|_{L^2}$ up to lower-order terms. Putting these together gives the Kato-Rellich estimate with $a = 0$ and a small $b$, which is more than enough.

Conclusion: the hydrogen atom Hamiltonian is self-adjoint on $H^2(\mathbb{R}^3)$. From this, the spectral theorem gives the spectrum (Bohr levels plus continuous part), and Stone's theorem gives the time evolution $e^{-itH}$. The whole physics of the hydrogen atom — energy levels, transition rules, time evolution — flows from this single self-adjointness result. Without Kato-Rellich, one is stuck.

For multi-electron atoms, similar techniques work for the Schrödinger Hamiltonian as long as the potential is bounded by a small multiple of $\sqrt{-\Delta}$ in operator-norm sense. The technique extends, but the technicalities multiply. By the time one is doing quantum chemistry with several heavy nuclei, the self-adjointness theory is genuinely complicated. But the principle is unchanged: get the domain right, prove self-adjointness, then spectral theorem and time evolution.

## What's Next, and Why

The next article puts unbounded self-adjoint operators to work, by constructing the **one-parameter semigroups** they generate via the functional calculus. The key result is the Hille-Yosida theorem, which characterizes the generators of strongly continuous semigroups. For the unitary case (self-adjoint generator), this is Stone's theorem; for the contraction case (dissipative generator), it is the full Hille-Yosida theorem. These semigroups solve initial-value problems for evolution equations: heat equation, wave equation, Schrödinger equation, Fokker-Planck. The semigroup approach is the right framework for time-dependent PDE, and it is the natural sequel to the static spectral theory we have built.

Unbounded operator theory is, in the end, the bridge between the static structure of an operator (its spectrum, its self-adjointness) and the dynamics it generates (the semigroup, the time evolution). With both halves in hand, one can finally do mathematical physics rigorously: write down a Schrödinger equation, identify its Hamiltonian as a self-adjoint operator on a specific domain, exponentiate via spectral theorem, and study the resulting unitary group. Each step depends on careful domain bookkeeping and the spectral theorem, but the resulting framework is robust and applies to a vast range of physical systems. For PDE, the same machinery handles parabolic and hyperbolic evolution problems, with semigroup contraction estimates replacing unitarity. The story is the same — only the choice of generator changes. By article 12 we will be applying all of this to elliptic PDE and quantum dynamics in earnest.

The conceptual lesson of this article: domains are not annoying technicalities, they are physical content. The same differential expression with different boundary conditions is a different operator, with different spectrum and different physics. Get the domains right, and the spectral machinery flows; get them wrong, and everything breaks. Once one has internalized this, the rest of mathematical physics becomes a sequence of careful domain identifications followed by spectral computations.

A final remark. Functional analysis becomes hardest at exactly the point where the abstract framework meets the concrete differential operators: the choice of domain, the question of self-adjointness, the deficiency indices, the Friedrichs extension. Beyond this point, everything is downhill — the spectral theorem applies, the functional calculus exists, the time evolution makes sense. The domain step is the hard step. Spend the time on it; the rest follows. Reed and Simon's first two volumes are the gold-standard reference, and they spend essentially their entire second volume on these issues. Once one has read those volumes the unbounded-operator literature stops being mysterious; until then it is full of phrases that sound technical and turn out to be load-bearing. The investment is worth it: there is no other way to make rigorous sense of quantum mechanics, and there is no other way to handle elliptic and parabolic PDE in their natural Hilbert-space setting. Get the domains right, and the rest of mathematical physics opens up.

---

*This is Part 9 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 8 — Spectral Theory](/en/functional-analysis/08-spectral-theory/)*

*Next: [Part 10 — Semigroups](/en/functional-analysis/10-semigroups/)*
