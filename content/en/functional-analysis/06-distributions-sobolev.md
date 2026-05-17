---
title: "Distributions and Sobolev Spaces — Where Analysis Meets PDE"
date: 2021-04-05 09:00:00
tags:
  - functional-analysis
  - distributions
  - sobolev-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 6
series_total: 6
translationKey: "functional-analysis-6"
description: "Distributions extend differentiation to non-smooth functions; Sobolev spaces make PDE theory rigorous."
---

## The problem with classical derivatives

Consider the PDE $-u'' = f$ on $[0,1]$. If $f$ is continuous, classical solutions exist and are $C^2$. But what if $f$ is an $L^2$ function — or worse, a point mass $\delta(x - x_0)$? The classical derivative doesn't apply, yet physically meaningful solutions exist. We need a framework where "derivative" makes sense for non-smooth objects.

The answer: stop asking what $u'(x)$ is at each point. Instead, define derivatives through their action on smooth test functions.

## Test functions and distributions

Let $\Omega \subseteq \mathbb{R}^n$ be open. The **test function space** is:

$$\mathcal{D}(\Omega) = C_c^\infty(\Omega) = \{f \in C^\infty(\Omega) : \text{supp}(f) \text{ is compact in } \Omega\}.$$

A **distribution** is a continuous linear functional on $\mathcal{D}(\Omega)$:

$$u: \mathcal{D}(\Omega) \to \mathbb{R}, \quad \varphi \mapsto \langle u, \varphi \rangle,$$

where continuity means: if $\varphi_n \to 0$ in $\mathcal{D}$ (supports stay in a fixed compact set, all derivatives converge uniformly), then $\langle u, \varphi_n \rangle \to 0$.

The space of distributions is $\mathcal{D}'(\Omega)$.

**Every locally integrable function is a distribution.** If $f \in L^1_{\text{loc}}(\Omega)$, define:

$$\langle f, \varphi \rangle = \int_\Omega f(x)\varphi(x)\, dx.$$

So distributions generalize functions. But there are distributions that aren't functions.

**Example 1: The Dirac delta.** $\langle \delta, \varphi \rangle = \varphi(0)$. This is a distribution (linear and continuous on $\mathcal{D}$), but there's no function $f$ with $\int f\varphi\, dx = \varphi(0)$ for all test functions. The delta "function" is genuinely a distribution, not a function.

## Distributional derivatives

The **distributional derivative** of $u \in \mathcal{D}'$ is defined by:

$$\langle u', \varphi \rangle = -\langle u, \varphi' \rangle \quad \forall \varphi \in \mathcal{D}.$$

The minus sign comes from integration by parts (boundary terms vanish since $\varphi$ has compact support). Higher derivatives:

$$\langle D^\alpha u, \varphi \rangle = (-1)^{|\alpha|} \langle u, D^\alpha \varphi \rangle.$$

Every distribution is infinitely differentiable (as a distribution). This is the power of the theory: differentiation never fails.

**Example 2: Derivative of the Heaviside function.** Let $H(x) = \mathbf{1}_{x > 0}$. Then:

$$\langle H', \varphi \rangle = -\langle H, \varphi' \rangle = -\int_0^\infty \varphi'(x)\, dx = \varphi(0) = \langle \delta, \varphi \rangle.$$

So $H' = \delta$ in the distributional sense. The "derivative of a step function is a delta function" is now a rigorous statement.

**Example 3.** The function $f(x) = |x|$ on $\mathbb{R}$ has distributional derivative $f'(x) = \text{sgn}(x)$ (the sign function), and $f'' = 2\delta$ in the sense of distributions. Repeated differentiation of non-smooth functions produces increasingly singular distributions.

## Sobolev spaces

Distributions let us differentiate anything, but for PDE we need spaces with both integrability and differentiability built in. Enter Sobolev spaces.

**Definition.** For $k \in \mathbb{N}$ and $1 \le p \le \infty$, the **Sobolev space** is:

$$W^{k,p}(\Omega) = \{u \in L^p(\Omega) : D^\alpha u \in L^p(\Omega) \text{ for all } |\alpha| \le k\},$$

with norm:

$$\|u\|_{W^{k,p}} = \left(\sum_{|\alpha| \le k} \|D^\alpha u\|_{L^p}^p\right)^{1/p}.$$

Here $D^\alpha u$ is the distributional derivative. The special case $p = 2$ is written $H^k(\Omega) = W^{k,2}(\Omega)$ — a Hilbert space.

**Theorem.** $W^{k,p}(\Omega)$ is a Banach space. $H^k(\Omega)$ is a Hilbert space with inner product:

$$\langle u, v \rangle_{H^k} = \sum_{|\alpha| \le k} \int_\Omega D^\alpha u \cdot D^\alpha v\, dx.$$

*Proof sketch.* If $(u_n)$ is Cauchy in $W^{k,p}$, then $(D^\alpha u_n)$ is Cauchy in $L^p$ for each $|\alpha| \le k$. Since $L^p$ is complete, $D^\alpha u_n \to v_\alpha$ in $L^p$. Check that $v_\alpha$ is the distributional derivative of $v_0 = \lim u_n$: $\langle v_\alpha, \varphi \rangle = \lim \langle D^\alpha u_n, \varphi \rangle = \lim (-1)^{|\alpha|}\langle u_n, D^\alpha \varphi \rangle = (-1)^{|\alpha|}\langle v_0, D^\alpha \varphi \rangle$. $\square$

## Sobolev embeddings

The key structural result: trading derivatives for integrability (or even continuity).

**Theorem (Sobolev embedding, 1D case).** If $u \in H^1(0,1)$, then $u$ is (equal a.e. to) a continuous function, and:

$$\|u\|_{L^\infty} \le C\|u\|_{H^1}.$$

One derivative of $L^2$ regularity in one dimension buys you continuity. In higher dimensions:

**Theorem (Sobolev, general).** If $\Omega \subseteq \mathbb{R}^n$ is bounded with smooth boundary, $kp > n$, then $W^{k,p}(\Omega) \hookrightarrow C(\overline{\Omega})$ (continuous embedding).

If $kp < n$, you get embedding into $L^q$ with $1/q = 1/p - k/n$ (Sobolev conjugate exponent).

**Example 4.** In $\mathbb{R}^3$, $H^1 = W^{1,2}$ embeds into $L^6$ (since $1/6 = 1/2 - 1/3$) but not into $L^\infty$. You need $H^2$ to embed into $C^0$ in 3D. The critical exponent depends on the dimension.

## Weak solutions of PDE

The payoff. Consider the boundary value problem:

$$-\Delta u = f \text{ in } \Omega, \quad u = 0 \text{ on } \partial\Omega.$$

Multiply by $v \in H^1_0(\Omega)$ (functions in $H^1$ vanishing on the boundary) and integrate by parts:

$$\int_\Omega \nabla u \cdot \nabla v\, dx = \int_\Omega f v\, dx \quad \forall v \in H^1_0(\Omega).$$

A **weak solution** is $u \in H^1_0(\Omega)$ satisfying this identity. No second derivatives of $u$ appear — we only need $u \in H^1$.

**Theorem (Lax-Milgram).** Let $H$ be a Hilbert space, $a: H \times H \to \mathbb{R}$ bilinear, bounded, and coercive ($a(u,u) \ge \alpha\|u\|^2$ for some $\alpha > 0$). Then for every $f \in H^*$, there exists a unique $u \in H$ with $a(u,v) = f(v)$ for all $v$.

*Proof.* By Riesz representation, $a(u, \cdot) = \langle Au, \cdot \rangle$ for some bounded operator $A$. Coercivity gives $\|Au\| \ge \alpha\|u\|$, so $A$ is injective with closed range. The bound $|a(u,v)| \le M\|u\|\,\|v\|$ gives $\|A\| \le M$. To show surjectivity: if $w \perp \text{range}(A)$, then $0 = \langle Aw, w \rangle = a(w,w) \ge \alpha\|w\|^2$, so $w = 0$. $\square$

Applied to the Dirichlet problem: $a(u,v) = \int \nabla u \cdot \nabla v$ is coercive on $H^1_0$ (by the Poincare inequality), so a unique weak solution exists for every $f \in L^2$.

## The big picture

Here's the chain of ideas:
1. **Distributions** extend differentiation to non-smooth objects.
2. **Sobolev spaces** give Banach/Hilbert structure to spaces of "weakly differentiable" functions.
3. **Embedding theorems** relate weak differentiability to classical regularity.
4. **Variational formulations** turn PDE into equations in Hilbert spaces.
5. **Lax-Milgram/Riesz** (abstract functional analysis) gives existence and uniqueness.
6. **Regularity theory** (elliptic regularity) bootstraps weak solutions back to classical ones.

Functional analysis provides steps 4-5. The entire modern PDE existence theory rests on the foundations we've built in this series.

## Where to go from here

This series covered the core of functional analysis: spaces (metric, normed, Banach, Hilbert), maps (bounded operators, functionals, compact operators), structure theorems (the big four), spectral theory, and the connection to PDE through distributions and Sobolev spaces.

The subject branches in many directions from here: unbounded operators and semigroups (quantum mechanics, evolution equations), operator algebras ($C^*$-algebras, von Neumann algebras), nonlinear functional analysis (fixed point theory, degree theory), microlocal analysis (wavefront sets, pseudodifferential operators). Each direction uses the same core machinery — completeness, duality, compactness — applied to increasingly sophisticated settings.

The common thread: infinite-dimensional analysis requires replacing pointwise arguments with structural ones. You can't compute with individual elements of $L^2$ (they're equivalence classes, not functions). You work with norms, inner products, duality pairings, and operator estimates. The abstraction isn't for aesthetics — it's because the concrete objects don't exist at the level of detail you'd need for pointwise arguments.
