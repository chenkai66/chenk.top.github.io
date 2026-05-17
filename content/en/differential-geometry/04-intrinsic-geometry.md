---
title: "Intrinsic Geometry — Theorema Egregium and Geodesics"
date: 2021-05-22 09:00:00
tags:
  - differential-geometry
  - intrinsic-geometry
  - geodesics
  - theorema-egregium
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 4
series_total: 6
translationKey: "differential-geometry-4"
description: "Gauss's Theorema Egregium and geodesics reveal geometry intrinsic to the surface itself."
---

Here is the central surprise of classical differential geometry: Gaussian curvature, which we defined using the normal vector and the shape operator (extrinsic data), depends only on the first fundamental form (intrinsic data). A surface-dwelling ant can detect curvature without ever leaving the surface.

## Christoffel symbols

The Christoffel symbols encode how the coordinate basis vectors $\mathbf{x}_u$, $\mathbf{x}_v$ change along the surface. They are defined by expressing the tangential components of second derivatives in terms of the basis:

$$\mathbf{x}_{uu} = \Gamma^1_{11}\mathbf{x}_u + \Gamma^2_{11}\mathbf{x}_v + e\mathbf{N}$$
$$\mathbf{x}_{uv} = \Gamma^1_{12}\mathbf{x}_u + \Gamma^2_{12}\mathbf{x}_v + f\mathbf{N}$$
$$\mathbf{x}_{vv} = \Gamma^1_{22}\mathbf{x}_u + \Gamma^2_{22}\mathbf{x}_v + g\mathbf{N}$$

The Christoffel symbols $\Gamma^k_{ij}$ are computed entirely from $E, F, G$ and their partial derivatives. For instance, when $F = 0$:

$$\Gamma^1_{11} = \frac{E_u}{2E}, \quad \Gamma^1_{12} = \frac{E_v}{2E}, \quad \Gamma^1_{22} = \frac{-G_u}{2E}.$$

They are not tensors — they depend on the choice of coordinates. But they are the building blocks of intrinsic geometry.

## Theorema Egregium

**Theorem (Gauss, 1827).** The Gaussian curvature $K$ of a surface depends only on the coefficients $E, F, G$ of the first fundamental form and their first and second partial derivatives.

When $F = 0$, the formula simplifies to:

$$K = -\frac{1}{2\sqrt{EG}}\left[\frac{\partial}{\partial u}\frac{G_u}{\sqrt{EG}} + \frac{\partial}{\partial v}\frac{E_v}{\sqrt{EG}}\right].$$

*Proof sketch.* The Gauss equation relates the Riemann curvature tensor (built from Christoffel symbols and their derivatives) to $K$:

$$R_{1212} = K(EG - F^2)$$

where $R_{1212} = \partial_v\Gamma^1_{11} - \partial_u\Gamma^1_{12} + \Gamma^1_{11}\Gamma^2_{12} - \Gamma^1_{12}\Gamma^2_{11} + \ldots$ (a specific combination). Since $\Gamma^k_{ij}$ are built from $E, F, G$ alone, so is $K$.

**Consequence.** Isometric surfaces have the same Gaussian curvature. You cannot change $K$ without stretching the surface.

**Example.** A cylinder has $K = 0$, same as a plane. You can flatten a cylinder without distortion. But a sphere has $K = 1/R^2 > 0$, so no piece of a sphere can be flattened onto a plane without stretching — this is why flat maps of the Earth inevitably distort distances.

## Geodesics

A **geodesic** is a curve on a surface that is "as straight as possible." Formally, a curve $\gamma(t)$ on $S$ is a geodesic if its acceleration has no tangential component:

$$\nabla_{\gamma'}\gamma' = 0$$

where $\nabla$ denotes the covariant derivative (the tangential component of the ordinary derivative in $\mathbb{R}^3$).

Equivalently, $\gamma$ is a geodesic if and only if its geodesic curvature $\kappa_g = 0$ everywhere. The geodesic curvature is the tangential component of the curvature vector — it measures how much the curve deviates from being a geodesic.

## The geodesic equations

In coordinates, if $\gamma(t) = \mathbf{x}(u(t), v(t))$, the geodesic equations are:

$$u'' + \Gamma^1_{11}(u')^2 + 2\Gamma^1_{12}u'v' + \Gamma^1_{22}(v')^2 = 0$$
$$v'' + \Gamma^2_{11}(u')^2 + 2\Gamma^2_{12}u'v' + \Gamma^2_{22}(v')^2 = 0$$

This is a system of second-order ODEs. Given an initial point and initial direction, there is a unique geodesic — the analogue of "a straight line is determined by a point and a direction."

## Example: geodesics on the sphere

On the sphere of radius $R$, geodesics are great circles (intersections with planes through the center). The shortest path between two points on a sphere follows a great circle arc.

The equator is a geodesic. Meridians are geodesics. But parallels (other than the equator) are not: they have nonzero geodesic curvature because they curve toward the nearest pole.

## Example: geodesics on a cylinder

On the cylinder $\mathbf{x}(u,v) = (\cos u, \sin u, v)$, the Christoffel symbols all vanish (since $E = G = 1$, $F = 0$, all constant). The geodesic equations reduce to $u'' = 0$, $v'' = 0$, giving $u = at + b$, $v = ct + d$ — straight lines in the $(u,v)$ parameter plane.

These correspond to: helices (when both $a,c \neq 0$), straight rulings ($a = 0$), and circles ($c = 0$). Unroll the cylinder flat and they become straight lines — as expected from the isometry with the plane.

## Geodesics as shortest paths

**Theorem.** A shortest curve between two points on a surface (if it exists) is a geodesic.

*Proof sketch.* Apply the calculus of variations to the length functional $L[\gamma] = \int |\gamma'|\,dt$. The Euler-Lagrange equations yield the geodesic equations.

The converse is false: geodesics are only locally length-minimizing. On a sphere, a great circle arc longer than a semicircle is a geodesic but not the shortest path.

## The Gauss-Bonnet preview

Geodesic curvature connects beautifully to topology. For a geodesic triangle $T$ on a surface (a triangle whose sides are geodesics), the angle excess equals the integral of Gaussian curvature:

$$(\alpha_1 + \alpha_2 + \alpha_3) - \pi = \iint_T K\,dA.$$

On a sphere ($K > 0$), triangle angles sum to more than $\pi$. On a hyperbolic surface ($K < 0$), they sum to less. This is the local version of the Gauss-Bonnet theorem.

## What's next

To generalize beyond surfaces in $\mathbb{R}^3$, we need the language of smooth manifolds and connections. This frees us from the ambient space entirely and opens the door to Riemannian geometry in arbitrary dimensions.
