---
title: "Gaussian Curvature and the Second Fundamental Form"
date: 2021-05-15 09:00:00
tags:
  - differential-geometry
  - gaussian-curvature
  - second-fundamental-form
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 3
series_total: 6
translationKey: "differential-geometry-3"
description: "The second fundamental form and how Gaussian curvature captures the intrinsic shape of surfaces."
---

The first fundamental form measures distances on a surface. The second fundamental form measures how the surface bends away from its tangent plane. Together they determine the surface completely (up to rigid motion).

## The shape operator

Let $S$ be a regular surface with unit normal $\mathbf{N}$. The **shape operator** (or Weingarten map) at $p$ is:

$$dN_p: T_pS \to T_pS$$

This is the differential of the Gauss map $\mathbf{N}: S \to S^2$. It sends a tangent vector $\mathbf{v}$ to the rate of change of the normal in the direction $\mathbf{v}$.

The shape operator is self-adjoint: $\langle dN_p(\mathbf{v}), \mathbf{w}\rangle = \langle \mathbf{v}, dN_p(\mathbf{w})\rangle$. This means it has real eigenvalues and orthogonal eigenvectors.

## The second fundamental form

The **second fundamental form** is the bilinear form:

$$II(\mathbf{v}, \mathbf{w}) = -\langle dN_p(\mathbf{v}), \mathbf{w}\rangle.$$

In coordinates $(u,v)$, write $II = e\,du^2 + 2f\,du\,dv + g\,dv^2$ where:

$$e = -\mathbf{N}_u \cdot \mathbf{x}_u = \mathbf{N} \cdot \mathbf{x}_{uu}, \quad f = \mathbf{N} \cdot \mathbf{x}_{uv}, \quad g = \mathbf{N} \cdot \mathbf{x}_{vv}.$$

The second equality uses $\mathbf{N} \cdot \mathbf{x}_u = 0$ differentiated.

## Principal curvatures

The eigenvalues $\kappa_1, \kappa_2$ of $dN_p$ (with the sign convention from $II$) are the **principal curvatures**. The corresponding eigenvectors are the **principal directions** — the directions in which the surface bends the most and the least.

At a point where you slice the surface with a normal plane containing direction $\mathbf{v}$, the resulting plane curve has curvature:

$$\kappa_n(\mathbf{v}) = II(\mathbf{v}, \mathbf{v}) / I(\mathbf{v}, \mathbf{v}).$$

This is the **normal curvature** in direction $\mathbf{v}$. The principal curvatures are its extreme values.

**Theorem (Euler).** If $\mathbf{v}$ makes angle $\theta$ with the first principal direction, then

$$\kappa_n(\theta) = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta.$$

*Proof sketch.* In principal coordinates (where the shape operator is diagonal), expand $II(\mathbf{v},\mathbf{v})/I(\mathbf{v},\mathbf{v})$ with $\mathbf{v} = \cos\theta\,\mathbf{e}_1 + \sin\theta\,\mathbf{e}_2$.

## Gaussian and mean curvature

From the principal curvatures, define:

$$K = \kappa_1\kappa_2 \quad \text{(Gaussian curvature)}$$

$$H = \frac{\kappa_1 + \kappa_2}{2} \quad \text{(mean curvature)}$$

In coordinates:

$$K = \frac{eg - f^2}{EG - F^2}, \qquad H = \frac{eG - 2fF + gE}{2(EG - F^2)}.$$

Gaussian curvature is the determinant of the shape operator. Mean curvature is half its trace.

## Geometric meaning of Gaussian curvature

- $K > 0$: the surface is locally convex (elliptic point). Both principal curvatures have the same sign. Think of a sphere or an egg.
- $K < 0$: saddle point (hyperbolic point). Principal curvatures have opposite signs. Think of a saddle or a Pringles chip.
- $K = 0$: the surface is flat in at least one direction (parabolic point). Cylinders and cones have $K = 0$ everywhere.

## Example: the sphere

For a sphere of radius $R$, the Gauss map is $\mathbf{N} = \mathbf{x}/R$ (outward normal). So $dN_p = (1/R)\,\text{Id}$, giving $\kappa_1 = \kappa_2 = 1/R$.

$$K = \frac{1}{R^2}, \qquad H = \frac{1}{R}.$$

Every point is umbilic ($\kappa_1 = \kappa_2$), and the Gaussian curvature is positive and constant.

## Example: the torus

Parametrize the torus by $\mathbf{x}(u,v) = ((R+r\cos u)\cos v,\, (R+r\cos u)\sin v,\, r\sin u)$ where $R > r > 0$.

After computing the metric and second fundamental form coefficients:

$$K = \frac{\cos u}{r(R + r\cos u)}.$$

On the outer equator ($u = 0$): $K = 1/(r(R+r)) > 0$ — elliptic, like a sphere.
On the inner equator ($u = \pi$): $K = -1/(r(R-r)) < 0$ — hyperbolic, like a saddle.
On the top and bottom circles ($u = \pi/2, 3\pi/2$): $K = 0$ — parabolic.

The total curvature of the torus is $\int K\,dA = 0$. We will see why in Chapter 6.

## Example: minimal surfaces

A surface with $H = 0$ everywhere is called a **minimal surface**. It locally minimizes area among surfaces with the same boundary.

The catenoid, parametrized by $\mathbf{x}(u,v) = (\cosh u\cos v,\, \cosh u\sin v,\, u)$, has $H = 0$. Its principal curvatures are $\kappa_1 = 1/\cosh^2 u$ and $\kappa_2 = -1/\cosh^2 u$, equal in magnitude but opposite in sign. Gaussian curvature: $K = -1/\cosh^4 u < 0$ everywhere.

## The fundamental theorem of surfaces

**Theorem (Bonnet).** Let $E, F, G$ and $e, f, g$ be smooth functions on an open set $U \subset \mathbb{R}^2$ with $EG - F^2 > 0$, satisfying the Gauss equation and the Codazzi-Mainardi equations. Then there exists a surface patch $\mathbf{x}: U \to \mathbb{R}^3$ with these as its first and second fundamental form coefficients. The surface is unique up to rigid motion.

*Proof sketch.* The Gauss and Codazzi-Mainardi equations are integrability conditions for the system of PDEs that determine $\mathbf{x}_u$, $\mathbf{x}_v$, and $\mathbf{N}$ from the fundamental forms. By the Frobenius theorem, these conditions are sufficient for existence.

This parallels the fundamental theorem of curves: prescribe the curvature data, get the shape (unique up to rigid motion).

## What's next

A remarkable fact: Gaussian curvature, despite being defined extrinsically (via the normal and the shape operator), turns out to depend only on the first fundamental form. This is Gauss's Theorema Egregium — the gateway to intrinsic geometry.
