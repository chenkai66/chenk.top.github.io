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

Here is the central surprise of classical differential geometry: Gaussian curvature, which we defined using the normal vector and the shape operator (extrinsic data), depends only on the first fundamental form (intrinsic data). A surface-dwelling ant — equipped with a ruler and protractor but no awareness of the ambient $\mathbb{R}^3$ — can detect Gaussian curvature without ever leaving the surface. This is Gauss's Theorema Egregium, and it changes everything about how we think about geometry.

## Christoffel symbols

Before stating the theorem, we need the Christoffel symbols — the bookkeeping that describes how coordinate basis vectors change as we move along the surface.

The second derivatives $\mathbf{x}_{uu}$, $\mathbf{x}_{uv}$, $\mathbf{x}_{vv}$ can be decomposed into tangential and normal components:

$$\mathbf{x}_{uu} = \Gamma^1_{11}\mathbf{x}_u + \Gamma^2_{11}\mathbf{x}_v + e\mathbf{N}$$
$$\mathbf{x}_{uv} = \Gamma^1_{12}\mathbf{x}_u + \Gamma^2_{12}\mathbf{x}_v + f\mathbf{N}$$
$$\mathbf{x}_{vv} = \Gamma^1_{22}\mathbf{x}_u + \Gamma^2_{22}\mathbf{x}_v + g\mathbf{N}$$

The normal components give the second fundamental form coefficients $e, f, g$. The tangential components are the **Christoffel symbols** $\Gamma^k_{ij}$. They are computed entirely from $E, F, G$ and their partial derivatives. For an orthogonal parametrization ($F = 0$):

$$\Gamma^1_{11} = \frac{E_u}{2E}, \quad \Gamma^1_{12} = \frac{E_v}{2E}, \quad \Gamma^1_{22} = \frac{-G_u}{2E}$$
$$\Gamma^2_{11} = \frac{-E_v}{2G}, \quad \Gamma^2_{12} = \frac{G_u}{2G}, \quad \Gamma^2_{22} = \frac{G_v}{2G}$$

The Christoffel symbols are not tensors — they depend on the choice of coordinates and can be made to vanish at any single point by choosing suitable coordinates (geodesic normal coordinates). But they are the building blocks from which intrinsic geometric quantities are assembled.

In the general case ($F \neq 0$), the Christoffel symbols involve inverting the metric matrix:

$$\Gamma^k_{ij} = \frac{1}{2}g^{kl}\left(\frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l}\right)$$

where $g^{kl}$ denotes the inverse of the metric matrix $(g_{ij}) = \begin{pmatrix}E & F \\ F & G\end{pmatrix}$. This formula — the same one that appears in Riemannian geometry — expresses the fundamental fact that the connection is determined by the metric alone.


![Geodesics on the sphere are great circles](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_fig4_geodesics.png)

## Theorema Egregium

**Theorem (Gauss, 1827).** The Gaussian curvature $K$ of a surface depends only on the coefficients $E, F, G$ of the first fundamental form and their first and second partial derivatives.

For an orthogonal parametrization ($F = 0$), the explicit formula is:

$$K = -\frac{1}{2\sqrt{EG}}\left[\frac{\partial}{\partial u}\left(\frac{G_u}{\sqrt{EG}}\right) + \frac{\partial}{\partial v}\left(\frac{E_v}{\sqrt{EG}}\right)\right].$$

*Proof sketch.* The **Gauss equation** relates the Riemann curvature tensor (constructed from the Christoffel symbols and their derivatives) to the Gaussian curvature:

$$R_{1212} = K(EG - F^2)$$

where $R_{1212}$ is a specific combination of Christoffel symbols and their partial derivatives. Since the $\Gamma^k_{ij}$ are built from $E, F, G$ alone, so is $R_{1212}$, and therefore so is $K$.

More explicitly, the Gauss equation for an orthogonal parametrization reads:

$$K = \frac{1}{2EG}\left[-E_{vv} - G_{uu} + \frac{E_u G_u + E_v^2}{2E} + \frac{G_v E_v + G_u^2}{2G}\right].$$

One can verify this by direct (tedious) computation: start from $K = (eg-f^2)/(EG-F^2)$, express $e, f, g$ in terms of $\mathbf{x}_{uu}\cdot\mathbf{N}$ etc., and eliminate all references to $\mathbf{N}$ using identities derived from $\mathbf{N}\cdot\mathbf{x}_u = 0$, $\mathbf{N}\cdot\mathbf{x}_v = 0$, and $|\mathbf{N}|=1$. The remarkable cancellation that eliminates all dependence on the second fundamental form is what makes the theorem "egregious" (remarkable).

Gauss called this result "egregium" (remarkable/outstanding) because it was completely unexpected. The definition of $K = \kappa_1\kappa_2$ involves the normal vector and the shape operator — objects that depend on how the surface sits in $\mathbb{R}^3$. Yet the final answer depends only on the intrinsic metric.

**Consequence 1.** Isometric surfaces have the same Gaussian curvature at corresponding points. You cannot change $K$ without stretching the surface.

**Consequence 2.** $K$ is a bending invariant. If you bend a surface (without stretching — like rolling paper into a cylinder), the Gaussian curvature at every point is preserved.

## Example: why maps of the Earth distort

A sphere has $K = 1/R^2 > 0$ everywhere. A plane has $K = 0$. By the Theorema Egregium, no isometry between them exists — not even locally. Therefore every flat map of a sphere must distort either distances, angles, or areas (or all three).

The Mercator projection preserves angles (it is conformal) but grossly distorts areas near the poles. The Lambert equal-area projection preserves areas but distorts angles. No projection preserves both, because no flat metric can reproduce the sphere's positive Gaussian curvature.

## Example: the cylinder is flat

A cylinder has $K = 0$, same as a plane. This confirms our earlier observation: a cylinder is locally isometric to the plane (roll/unroll paper). But a sphere with $K = 1/R^2 > 0$ cannot be isometric to a plane — you cannot flatten orange peel without tearing or stretching.

## Parallel transport

Before discussing geodesics, we need the concept of parallel transport — a way to move vectors along curves while keeping them "as constant as possible" on the surface.

Given a curve $\gamma(t)$ on $S$ and a tangent vector $\mathbf{v}_0 \in T_{\gamma(0)}S$, the **parallel transport** of $\mathbf{v}_0$ along $\gamma$ is a vector field $V(t) \in T_{\gamma(t)}S$ satisfying:

$$\frac{DV}{dt} = 0, \quad V(0) = \mathbf{v}_0$$

where $D/dt$ is the covariant derivative (the tangential projection of the ordinary derivative $dV/dt$). In coordinates, if $V = V^1\mathbf{x}_u + V^2\mathbf{x}_v$ and $\gamma(t) = \mathbf{x}(u(t),v(t))$:

$$\dot{V}^k + \Gamma^k_{ij}\dot{\gamma}^i V^j = 0, \quad k = 1,2.$$

This is a first-order linear ODE system — it always has a unique solution. Parallel transport preserves lengths and angles: if $V$ and $W$ are both parallel along $\gamma$, then $I(V,W)$ is constant along the curve.

On a flat surface, parallel transport is path-independent. On a curved surface, it depends on the path — and this path-dependence measures curvature. The classic demonstration: parallel-transport a vector around a closed loop on the sphere. When it returns to the starting point, it has rotated by an angle equal to the enclosed solid angle (i.e., $K$ times the enclosed area). This rotation is the **holonomy** of the loop.

## Geodesics

A **geodesic** is a curve on a surface that is "as straight as possible." Formally, a curve $\gamma(t)$ is a geodesic if its acceleration has no tangential component:

$$\nabla_{\gamma'}\gamma' = 0$$

where $\nabla$ denotes the covariant derivative — the tangential component of the ordinary derivative in $\mathbb{R}^3$. Equivalently, the **geodesic curvature** $\kappa_g = 0$ everywhere along $\gamma$.

Intuitively: if you walk on a surface always looking straight ahead (never turning left or right relative to the surface), your path is a geodesic. A geodesic may curve in $\mathbb{R}^3$, but it never curves within the surface.

## The geodesic equations

In coordinates, if $\gamma(t) = \mathbf{x}(u(t), v(t))$, the geodesic equations are:

$$u'' + \Gamma^1_{11}(u')^2 + 2\Gamma^1_{12}u'v' + \Gamma^1_{22}(v')^2 = 0$$
$$v'' + \Gamma^2_{11}(u')^2 + 2\Gamma^2_{12}u'v' + \Gamma^2_{22}(v')^2 = 0$$

This is a system of second-order nonlinear ODEs. By the existence-uniqueness theorem, given an initial point $p$ and an initial direction $\mathbf{v} \in T_pS$, there is a unique geodesic. This parallels the Euclidean fact: a straight line is determined by a point and a direction.

## Example: geodesics on the sphere

On the sphere of radius $R$, geodesics are great circles — intersections with planes through the center. The shortest path between two points follows a great circle arc.

**Why great circles?** The geodesic equation for the sphere, after some computation, reduces to the condition that $\gamma$ lies in a plane through the origin. Alternatively: a great circle has $\kappa_g = 0$ because its curvature vector (pointing toward the center of the sphere) is purely normal to the surface — no tangential component.

The equator is a geodesic. Every meridian is a geodesic. But a parallel of latitude $\theta_0 \neq \pi/2$ is not: it has geodesic curvature $\kappa_g = \cot\theta_0/R \neq 0$ — it curves toward the nearer pole.

Airline flight paths follow great circle arcs, not lines of constant latitude. A flight from London to Tokyo goes over the Arctic — the great circle route — saving significant distance over the "obvious" path along a parallel.

## Example: geodesics on a cylinder

On the cylinder $\mathbf{x}(u,v) = (\cos u, \sin u, v)$, all Christoffel symbols vanish (since $E = G = 1$, $F = 0$, all constant). The geodesic equations reduce to $u'' = 0$, $v'' = 0$, giving $u = at + b$, $v = ct + d$ — straight lines in the $(u,v)$ parameter plane.

These correspond to three families on the cylinder:
- Helices: both $a, c \neq 0$ — they spiral up the cylinder at constant angle.
- Vertical rulings: $a = 0$ — straight lines parallel to the axis.
- Horizontal circles: $c = 0$ — cross-sections.

Unroll the cylinder flat and all these geodesics become straight lines — as expected from the isometry with the plane. The helix, which looks curved in $\mathbb{R}^3$, is intrinsically straight on the cylinder.

## Example: geodesics on the torus

On the torus, the situation is more complex. The outer equator ($u = 0$) and inner equator ($u = \pi$) are geodesics — they are the "straightest" paths going around the torus in the $v$-direction. Meridians (constant $v$) are also geodesics. But most geodesics on the torus are neither closed nor periodic: they wind around the torus ergodically, never exactly repeating.

## Geodesics as shortest paths

**Theorem.** A shortest curve between two points on a surface (if it exists) is a geodesic.

*Proof sketch.* Apply the calculus of variations to the energy functional $E[\gamma] = \frac{1}{2}\int |\gamma'|^2\,dt$. The Euler-Lagrange equations yield the geodesic equations. (Working with energy rather than length avoids the complication of reparametrization invariance.)

The converse is local only: geodesics are locally length-minimizing but not always globally. On a sphere, a great circle arc shorter than a semicircle is the shortest path. But the longer complementary arc is also a geodesic — just not the shortest one. Beyond the **conjugate point** (the antipodal point), geodesics cease to minimize length.

## The Gauss-Bonnet connection

Geodesic curvature connects local geometry to global topology. For a geodesic triangle $T$ on a surface (a triangle whose sides are geodesics, so $\kappa_g = 0$ on each side), the **angle excess** equals the integral of Gaussian curvature over the interior:

$$(\alpha_1 + \alpha_2 + \alpha_3) - \pi = \iint_T K\,dA.$$

On a sphere ($K > 0$): triangle angles sum to more than $\pi$. The triangle "bulges." On a hyperbolic surface ($K < 0$): triangle angles sum to less than $\pi$. The triangle "pinches." On a flat surface ($K = 0$): angles sum to exactly $\pi$ — Euclidean geometry.

This is the local version of the Gauss-Bonnet theorem, which we will develop fully in Chapter 6. It is the first hint that curvature — a local, differential quantity — carries topological information.

## What's next

To generalize beyond surfaces in $\mathbb{R}^3$, we need the language of smooth manifolds and connections. This frees us from the ambient space entirely and opens the door to Riemannian geometry in arbitrary dimensions — where the "Christoffel symbols and metric" framework scales up without modification.

---

*This is Part 4 of [Differential Geometry](/en/series/differential-geometry/) (6 parts).
Previous: [Part 3 — Gaussian Curvature and the Second Fundamental Form](/en/differential-geometry/03-curvature-of-surfaces/) · Next: [Part 5 — Smooth Manifolds, Tangent Bundles, and Connections](/en/differential-geometry/05-manifolds-and-connections/)*
