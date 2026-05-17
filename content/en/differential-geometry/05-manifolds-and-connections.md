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

So far we studied curves and surfaces embedded in $\mathbb{R}^3$. But many geometric objects — spacetime in general relativity, configuration spaces in mechanics, the space of all rotations — have no natural ambient space. We need an intrinsic framework. That framework is the smooth manifold.

## Smooth manifolds

A **smooth manifold** of dimension $n$ is a topological space $M$ equipped with an atlas of charts $\{(U_\alpha, \varphi_\alpha)\}$ where each $\varphi_\alpha: U_\alpha \to \mathbb{R}^n$ is a homeomorphism onto an open set, and on overlaps the transition maps $\varphi_\beta \circ \varphi_\alpha^{-1}$ are smooth ($C^\infty$).

The definition formalizes "locally looks like $\mathbb{R}^n$, with a consistent notion of smoothness." No ambient space needed.

**Examples.**
- $\mathbb{R}^n$ itself (one chart: the identity).
- The sphere $S^n$ (two charts: stereographic projections from north and south poles).
- The torus $T^2 = S^1 \times S^1$.
- Real projective space $\mathbb{RP}^n$: the space of lines through the origin in $\mathbb{R}^{n+1}$.

## Tangent vectors and the tangent space

On a surface in $\mathbb{R}^3$, tangent vectors were literally vectors in the ambient space. On an abstract manifold, we define tangent vectors as **derivations**: linear maps $v: C^\infty(M) \to \mathbb{R}$ satisfying the Leibniz rule $v(fg) = f(p)v(g) + g(p)v(f)$.

In local coordinates $(x^1, \ldots, x^n)$, the tangent space $T_pM$ has basis $\{\partial/\partial x^1|_p, \ldots, \partial/\partial x^n|_p\}$. A tangent vector is:

$$v = v^i \frac{\partial}{\partial x^i}\bigg|_p$$

(Einstein summation convention: repeated indices are summed.)

The tangent space $T_pM$ is an $n$-dimensional real vector space.

## The tangent bundle

The **tangent bundle** is the disjoint union of all tangent spaces:

$$TM = \bigsqcup_{p \in M} T_pM = \{(p, v) : p \in M,\, v \in T_pM\}.$$

It is itself a smooth manifold of dimension $2n$. A **vector field** $X$ on $M$ is a smooth section of the tangent bundle: a smooth map $X: M \to TM$ with $X(p) \in T_pM$ for all $p$.

In coordinates: $X = X^i(x)\,\partial/\partial x^i$.

## Riemannian metrics

A **Riemannian metric** $g$ on $M$ is a smoothly varying inner product on each tangent space. In coordinates:

$$g = g_{ij}\,dx^i \otimes dx^j$$

where $(g_{ij}(p))$ is a positive-definite symmetric matrix at each point. For a surface, this is exactly the first fundamental form: $g_{11} = E$, $g_{12} = g_{21} = F$, $g_{22} = G$.

A manifold equipped with a Riemannian metric is a **Riemannian manifold**. On it we can measure lengths, angles, volumes — all the machinery of the first fundamental form, generalized to any dimension.

## The problem of differentiation

On $\mathbb{R}^n$, differentiating a vector field is straightforward: differentiate each component. On a manifold, this fails because tangent vectors at different points live in different vector spaces. There is no canonical way to compare them.

We need additional structure: a **connection**.

## Connections (covariant derivatives)

A **connection** (or covariant derivative) on $M$ is an operator $\nabla$ that takes a vector field $X$ and a vector field $Y$ and produces a vector field $\nabla_X Y$, satisfying:

1. $\nabla_{fX+gY}Z = f\nabla_X Z + g\nabla_Y Z$ (linear over functions in the first argument)
2. $\nabla_X(Y+Z) = \nabla_X Y + \nabla_X Z$
3. $\nabla_X(fY) = (Xf)Y + f\nabla_X Y$ (Leibniz rule)

In coordinates, the connection is determined by the **Christoffel symbols**:

$$\nabla_{\partial_i}\partial_j = \Gamma^k_{ij}\,\partial_k.$$

There are many possible connections on a manifold. Which one is "natural"?

## The Levi-Civita connection

**Theorem (Fundamental theorem of Riemannian geometry).** On a Riemannian manifold $(M, g)$, there exists a unique connection $\nabla$ that is:

1. **Torsion-free:** $\nabla_X Y - \nabla_Y X = [X,Y]$
2. **Metric-compatible:** $X\langle Y, Z\rangle = \langle\nabla_X Y, Z\rangle + \langle Y, \nabla_X Z\rangle$

This is the **Levi-Civita connection**. Its Christoffel symbols are:

$$\Gamma^k_{ij} = \frac{1}{2}g^{kl}\left(\frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l}\right).$$

*Proof sketch.* Write out metric compatibility for three cyclic permutations of $(X,Y,Z)$, add and subtract appropriately, then use torsion-freeness to isolate $\langle\nabla_X Y, Z\rangle$. Since $g$ is non-degenerate, this determines $\nabla_X Y$ uniquely.

For surfaces in $\mathbb{R}^3$, the Levi-Civita connection coincides with our earlier Christoffel symbols.

## Parallel transport

Given a curve $\gamma(t)$ and a vector $v \in T_{\gamma(0)}M$, the **parallel transport** of $v$ along $\gamma$ is the vector field $V(t)$ along $\gamma$ satisfying:

$$\nabla_{\gamma'(t)}V = 0, \quad V(0) = v.$$

This is a first-order linear ODE, so a unique solution exists. Parallel transport preserves inner products (by metric compatibility) and provides a way to "slide" vectors along curves without twisting them.

On a flat surface, parallel transport is path-independent: the result depends only on endpoints. On a curved surface, it depends on the path — this path-dependence is precisely what curvature measures.

**Example.** Transport a vector around a geodesic triangle on a sphere (say, from the north pole to the equator along a meridian, along the equator by angle $\alpha$, then back to the pole). The vector rotates by angle $\alpha$ — equal to the area of the triangle divided by $R^2$. This is the holonomy of the connection.

## Geodesics revisited

With a connection in hand, geodesics are curves whose tangent vector is parallel-transported along itself:

$$\nabla_{\gamma'}\gamma' = 0.$$

In coordinates: $\ddot{x}^k + \Gamma^k_{ij}\dot{x}^i\dot{x}^j = 0$. The same equations as before, now in full generality.

## What's next

We have all the pieces: manifolds, metrics, connections, curvature. The culmination is the Gauss-Bonnet theorem, which links the total curvature of a surface to its topology — a profound bridge between local geometry and global structure.
