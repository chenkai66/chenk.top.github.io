---
title: "Smooth Manifolds, Tangent Bundles, and Connections"
date: 2021-05-29 09:00:00
tags:
  - differential-geometry
  - manifolds
  - connections
  - riemannian-geometry
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 5
series_total: 6
translationKey: "differential-geometry-5"
description: "Abstract manifolds, tangent spaces, and connections generalize surface geometry to any dimension."
---

So far we studied curves and surfaces embedded in $\mathbb{R}^3$. But many geometric objects — spacetime in general relativity, configuration spaces in mechanics, the space of all rotations — have no natural ambient space. We need an intrinsic framework that makes sense without any reference to a surrounding Euclidean world. That framework is the smooth manifold, and it generalizes everything from the previous chapters to arbitrary dimension.

## Smooth manifolds

A **smooth manifold** of dimension $n$ is a topological space $M$ (Hausdorff, second-countable) equipped with an atlas of charts $\{(U_\alpha, \varphi_\alpha)\}$ where each $\varphi_\alpha: U_\alpha \to \mathbb{R}^n$ is a homeomorphism onto an open set, and on overlaps the transition maps $\varphi_\beta \circ \varphi_\alpha^{-1}$ are smooth ($C^\infty$).

The definition formalizes "locally looks like $\mathbb{R}^n$, with a consistent notion of smoothness." A manifold is a space where calculus works locally, even though the global structure may be topologically complex.

**Examples:**
- $\mathbb{R}^n$ itself (one chart: the identity). The trivial case.
- The sphere $S^n$: two charts suffice (stereographic projections from north and south poles). $S^2$ is the surface we have been studying, now viewed intrinsically.
- The torus $T^2 = S^1 \times S^1$: locally flat, globally has a hole.
- Real projective space $\mathbb{RP}^n$: the space of lines through the origin in $\mathbb{R}^{n+1}$. It is a manifold because each line can be described by $n$ coordinates (ratios of homogeneous coordinates).
- The group $SO(3)$ of rotations in 3-space: a 3-dimensional manifold (parametrized by Euler angles, for instance). It is also a Lie group — a manifold with a group structure.


![Charts, overlap regions, and transition maps on a manifold](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/05-manifolds-and-connections/dg_fig5_manifold.png)

## Tangent vectors as derivations

On a surface in $\mathbb{R}^3$, tangent vectors were literally vectors in the ambient space, lying in the tangent plane. On an abstract manifold with no ambient space, we need a different definition.

A **tangent vector** at $p \in M$ is a derivation: a linear map $v: C^\infty(M) \to \mathbb{R}$ satisfying the Leibniz rule:

$$v(fg) = f(p)\,v(g) + g(p)\,v(f).$$

This captures the idea of "directional derivative at $p$." In local coordinates $(x^1, \ldots, x^n)$, the partial derivatives $\partial/\partial x^i|_p$ form a basis for the tangent space $T_pM$. A general tangent vector is:

$$v = v^i \frac{\partial}{\partial x^i}\bigg|_p$$

(Einstein summation convention: repeated upper-lower index pairs are summed.) The tangent space $T_pM$ is an $n$-dimensional real vector space — one such space at each point of $M$.

**Example.** On $S^2$ with coordinates $(\theta,\phi)$, a tangent vector at a point might be $v = 3\,\partial/\partial\theta + 2\,\partial/\partial\phi$. It acts on functions: $v(f) = 3\,\partial f/\partial\theta + 2\,\partial f/\partial\phi$.

## The tangent bundle

The **tangent bundle** is the disjoint union of all tangent spaces:

$$TM = \bigsqcup_{p \in M} T_pM = \{(p, v) : p \in M,\, v \in T_pM\}.$$

It is itself a smooth manifold of dimension $2n$: locally, a point in $TM$ is described by $n$ coordinates for the base point plus $n$ coordinates for the tangent vector. A **vector field** $X$ on $M$ is a smooth section of the tangent bundle: a smooth map $X: M \to TM$ with $X(p) \in T_pM$ for all $p$.

In coordinates: $X = X^i(x)\,\partial/\partial x^i$, where the $X^i$ are smooth functions.

Geometrically, the tangent bundle is the space of all possible "states of motion" on $M$ — a point plus a velocity. For a surface, $TM$ is four-dimensional (two coordinates for position, two for the tangent vector). For spacetime ($n = 4$), $TM$ has dimension 8.

The tangent bundle has a natural projection $\pi: TM \to M$ sending $(p, v) \mapsto p$. The fiber over each point $\pi^{-1}(p) = T_pM$ is a vector space. This makes $TM$ a **vector bundle** — a family of vector spaces parametrized smoothly by a base manifold. Other important bundles include the cotangent bundle $T^*M$ (dual spaces of 1-forms), tensor bundles, and the frame bundle (bases for each tangent space).

## Riemannian metrics

A **Riemannian metric** $g$ on $M$ is a smoothly varying inner product on each tangent space:

$$g_p: T_pM \times T_pM \to \mathbb{R}$$

positive-definite, symmetric, bilinear at each $p$, and varying smoothly with $p$. In coordinates:

$$g = g_{ij}\,dx^i \otimes dx^j$$

where $(g_{ij}(p))$ is a positive-definite symmetric $n\times n$ matrix at each point.

For a surface with the first fundamental form: $g_{11} = E$, $g_{12} = g_{21} = F$, $g_{22} = G$. The Riemannian metric is the direct generalization of the first fundamental form to arbitrary dimension.

A manifold equipped with a Riemannian metric is a **Riemannian manifold** $(M, g)$. On it we can measure:
- **Lengths:** $L[\gamma] = \int_a^b \sqrt{g_{ij}\dot{x}^i\dot{x}^j}\,dt$.
- **Angles:** $\cos\theta = g(u,v)/(|u|\,|v|)$.
- **Volumes:** $\text{vol} = \int \sqrt{\det(g_{ij})}\,dx^1\cdots dx^n$.

**Example (hyperbolic plane).** The upper half-plane $\{(x,y): y > 0\}$ with metric $ds^2 = (dx^2+dy^2)/y^2$ has constant Gaussian curvature $K = -1$. Distances grow as you approach the $x$-axis ($y \to 0$), making the "boundary" infinitely far away. This is the Poincare half-plane model of hyperbolic geometry.

## The problem of differentiation

On $\mathbb{R}^n$, differentiating a vector field is straightforward: differentiate each component. On a manifold, this naive approach fails because tangent vectors at different points live in different vector spaces. The vector $v \in T_pM$ and $w \in T_qM$ (for $p \neq q$) cannot be subtracted or compared — they live in different spaces.

The ordinary derivative of a vector field along a curve gives a vector in $\mathbb{R}^3$ that may not be tangent to the manifold. We need a derivative that stays on the manifold. This requires additional structure: a **connection**.

## Connections (covariant derivatives)

A **connection** (or covariant derivative) on $M$ is an operator $\nabla$ that takes a vector field $X$ and a vector field $Y$ and produces a new vector field $\nabla_X Y$, satisfying:

1. $C^\infty$-linearity in $X$: $\nabla_{fX+gY}Z = f\nabla_X Z + g\nabla_Y Z$
2. $\mathbb{R}$-linearity in $Y$: $\nabla_X(Y+Z) = \nabla_X Y + \nabla_X Z$
3. Leibniz rule: $\nabla_X(fY) = (Xf)Y + f\nabla_X Y$

In coordinates, the connection is determined by the **Christoffel symbols** $\Gamma^k_{ij}$:

$$\nabla_{\partial_i}\partial_j = \Gamma^k_{ij}\,\partial_k.$$

Then for vector fields $X = X^i\partial_i$ and $Y = Y^j\partial_j$:

$$\nabla_X Y = X^i\left(\frac{\partial Y^k}{\partial x^i} + \Gamma^k_{ij}Y^j\right)\partial_k.$$

The quantity in parentheses is the $k$-th component of $\nabla_X Y$. The Christoffel symbols encode the "correction" needed beyond ordinary partial differentiation.

**Example (covariant derivative on the sphere).** On $S^2$ with standard coordinates $(\theta, \phi)$ and metric $g = d\theta^2 + \sin^2\theta\,d\phi^2$, the non-vanishing Christoffel symbols are $\Gamma^\theta_{\phi\phi} = -\sin\theta\cos\theta$ and $\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \cot\theta$. Consider the vector field $Y = \partial/\partial\phi$ (pointing "east"). Its covariant derivative in the $\theta$-direction is:

$$\nabla_{\partial_\theta}(\partial_\phi) = \Gamma^\theta_{\theta\phi}\partial_\theta + \Gamma^\phi_{\theta\phi}\partial_\phi = 0 + \cot\theta\,\partial_\phi.$$

This says: as you move north (increasing $\theta$ toward the pole), the "east" direction rotates relative to the parallel-transport frame — it picks up a $\cot\theta$ correction. This is the Christoffel symbol encoding the convergence of meridians toward the poles.

There are infinitely many possible connections on a manifold. Which one is geometrically natural?

## The Levi-Civita connection

**Theorem (Fundamental theorem of Riemannian geometry).** On a Riemannian manifold $(M, g)$, there exists a unique connection $\nabla$ that is:

1. **Torsion-free:** $\nabla_X Y - \nabla_Y X = [X,Y]$ (the Lie bracket)
2. **Metric-compatible:** $X\langle Y, Z\rangle = \langle\nabla_X Y, Z\rangle + \langle Y, \nabla_X Z\rangle$

This is the **Levi-Civita connection**. Its Christoffel symbols are given by the Koszul formula:

$$\Gamma^k_{ij} = \frac{1}{2}g^{kl}\left(\frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l}\right).$$

*Proof of uniqueness.* Write metric compatibility for all cyclic permutations of $(X,Y,Z)$: three equations. Add the first two, subtract the third. Use torsion-freeness ($\nabla_X Y - \nabla_Y X = [X,Y]$) to eliminate asymmetric terms. This isolates $g(\nabla_X Y, Z)$ as an explicit expression in $g$, $X$, $Y$, $Z$ and their derivatives. Since $g$ is non-degenerate, $\nabla_X Y$ is uniquely determined. Existence follows by verifying that the Koszul formula defines a valid connection satisfying both properties.

The two conditions are geometrically natural. Torsion-freeness means parallel transport does not "twist" — parallelograms close. Metric-compatibility means parallel transport preserves lengths and angles. Together they single out the unique "natural" connection. Any other connection either twists vectors as it transports them, or fails to preserve the metric structure.

**Why uniqueness matters.** On a Riemannian manifold, there is no choice to make: the metric determines the connection, which determines geodesics, parallel transport, and curvature. Everything flows from the single datum $g$.

## Parallel transport

Given a curve $\gamma(t)$ and a vector $v \in T_{\gamma(0)}M$, the **parallel transport** of $v$ along $\gamma$ is the vector field $V(t)$ along $\gamma$ satisfying:

$$\nabla_{\gamma'(t)}V = 0, \quad V(0) = v.$$

In coordinates, this becomes a first-order linear ODE: $\dot{V}^k + \Gamma^k_{ij}\dot{\gamma}^i V^j = 0$. Since linear ODEs have unique solutions, parallel transport is well-defined.

Parallel transport preserves inner products (by metric compatibility): if $V$ and $W$ are both parallel along $\gamma$, then $g(V,W)$ is constant. Lengths and angles are preserved.

On a flat manifold ($\mathbb{R}^n$ with the standard metric), parallel transport is path-independent: the result depends only on endpoints, not the path taken. On a curved manifold, parallel transport depends on the path — and this path-dependence is precisely what curvature measures.

**Example (parallel transport on the sphere).** Transport a tangent vector from the north pole along a meridian to the equator, then along the equator through angle $\alpha$, then back to the north pole along another meridian. The vector rotates by angle $\alpha$. The rotation angle equals the solid angle subtended by the geodesic triangle — which equals $K \cdot \text{area} = (1/R^2)(R^2\alpha) = \alpha$. This geometric phase is called **holonomy**.

## Geodesics revisited

With a connection in hand, geodesics are curves whose tangent vector is parallel-transported along itself:

$$\nabla_{\gamma'}\gamma' = 0.$$

In coordinates: $\ddot{x}^k + \Gamma^k_{ij}\dot{x}^i\dot{x}^j = 0$. The same equations as Chapter 4, now valid on any Riemannian manifold of any dimension.

**Example (geodesics on the hyperbolic plane).** In the Poincare half-plane model with metric $ds^2 = (dx^2+dy^2)/y^2$, geodesics are: (1) vertical lines $x = \text{const}$, and (2) semicircles centered on the $x$-axis. These are the "straight lines" of hyperbolic geometry.

## The curvature tensor

The **Riemann curvature tensor** measures the failure of parallel transport to be path-independent:

$$R(X,Y)Z = \nabla_X\nabla_Y Z - \nabla_Y\nabla_X Z - \nabla_{[X,Y]}Z.$$

If $R = 0$ everywhere, the manifold is flat (locally isometric to $\mathbb{R}^n$). For surfaces, the Riemann tensor reduces to a single function — the Gaussian curvature $K$ — via $R_{1212} = K(EG - F^2)$.

In higher dimensions, $R$ carries much more information. Its contractions give the Ricci tensor $R_{ij}$ and the scalar curvature $R$ — the objects that appear in Einstein's field equations of general relativity.

## What's next

We have all the pieces: manifolds, metrics, connections, geodesics, curvature. The culmination is the Gauss-Bonnet theorem, which links the total curvature of a closed surface to its Euler characteristic — a purely topological invariant. Local differential data, integrated globally, yields topological information. That bridge between geometry and topology is the subject of the final chapter.

---

*This is Part 5 of [Differential Geometry](/en/series/differential-geometry/) (6 parts).
Previous: [Part 4 — Intrinsic Geometry](/en/differential-geometry/04-intrinsic-geometry/) · Next: [Part 6 — The Gauss-Bonnet Theorem](/en/differential-geometry/06-gauss-bonnet/)*
