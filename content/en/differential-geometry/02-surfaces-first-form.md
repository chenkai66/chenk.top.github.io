---
title: "Surfaces and the First Fundamental Form"
date: 2021-05-08 09:00:00
tags:
  - differential-geometry
  - surfaces
  - first-fundamental-form
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 2
series_total: 6
translationKey: "differential-geometry-2"
description: "How the first fundamental form encodes lengths, angles, and areas on a surface."
---

A surface is a two-dimensional object living in $\mathbb{R}^3$. The first fundamental form is the tool that lets us do metric geometry — measuring distances, angles, areas — directly on the surface, without reference to the ambient space.

## Regular surfaces

A **regular surface** $S \subset \mathbb{R}^3$ is a subset such that for each point $p \in S$, there exists a smooth map $\mathbf{x}: U \to S$ (where $U \subset \mathbb{R}^2$ is open) satisfying:

1. $\mathbf{x}$ is a homeomorphism onto its image.
2. The differential $d\mathbf{x}_q$ is injective for all $q \in U$.

The map $\mathbf{x}(u,v)$ is a **parametrization** (or coordinate patch). Condition 2 means the partial derivatives $\mathbf{x}_u$ and $\mathbf{x}_v$ are linearly independent — the surface has a well-defined tangent plane everywhere.

**Example.** The sphere $S^2$ of radius $R$: parametrize by $\mathbf{x}(\theta, \phi) = (R\sin\theta\cos\phi,\, R\sin\theta\sin\phi,\, R\cos\theta)$ for $\theta \in (0,\pi)$, $\phi \in (0,2\pi)$. This covers everything except a meridian and the poles.

## The tangent plane

At a point $p = \mathbf{x}(u_0, v_0)$, the tangent plane $T_pS$ is spanned by $\mathbf{x}_u$ and $\mathbf{x}_v$. Any tangent vector $\mathbf{w} \in T_pS$ can be written as

$$\mathbf{w} = a\,\mathbf{x}_u + b\,\mathbf{x}_v$$

for some $a, b \in \mathbb{R}$. The unit normal to the surface is

$$\mathbf{N} = \frac{\mathbf{x}_u \times \mathbf{x}_v}{|\mathbf{x}_u \times \mathbf{x}_v|}.$$

## The first fundamental form

The **first fundamental form** (or metric) is the restriction of the dot product in $\mathbb{R}^3$ to the tangent plane. For tangent vectors $\mathbf{w}_1, \mathbf{w}_2 \in T_pS$, it is simply $\langle \mathbf{w}_1, \mathbf{w}_2 \rangle = \mathbf{w}_1 \cdot \mathbf{w}_2$.

In coordinates, if $\mathbf{w} = a\,\mathbf{x}_u + b\,\mathbf{x}_v$, then

$$I(\mathbf{w}) = |\mathbf{w}|^2 = E\,a^2 + 2F\,ab + G\,b^2$$

where the **metric coefficients** are:

$$E = \mathbf{x}_u \cdot \mathbf{x}_u, \quad F = \mathbf{x}_u \cdot \mathbf{x}_v, \quad G = \mathbf{x}_v \cdot \mathbf{x}_v.$$

This is often written as a matrix:

$$\begin{pmatrix} E & F \\ F & G \end{pmatrix}.$$

The first fundamental form encodes all intrinsic metric information about the surface.

## Computing with the metric

### Arc length

A curve $\alpha(t) = \mathbf{x}(u(t), v(t))$ on the surface has length:

$$L = \int_a^b \sqrt{E\,u'^2 + 2F\,u'v' + G\,v'^2}\, dt.$$

### Area

The area of a region $R = \mathbf{x}(D)$ is:

$$A = \iint_D |\mathbf{x}_u \times \mathbf{x}_v|\, du\,dv = \iint_D \sqrt{EG - F^2}\, du\,dv.$$

### Angle between curves

If two curves on the surface meet at a point, with tangent vectors $\mathbf{w}_1 = a_1\mathbf{x}_u + b_1\mathbf{x}_v$ and $\mathbf{w}_2 = a_2\mathbf{x}_u + b_2\mathbf{x}_v$, the angle $\theta$ between them satisfies:

$$\cos\theta = \frac{Ea_1a_2 + F(a_1b_2+a_2b_1) + Gb_1b_2}{\sqrt{Ea_1^2+2Fa_1b_1+Gb_1^2}\,\sqrt{Ea_2^2+2Fa_2b_2+Gb_2^2}}.$$

## Example: the sphere

For the sphere parametrized above, compute:

$$\mathbf{x}_\theta = (R\cos\theta\cos\phi,\, R\cos\theta\sin\phi,\, -R\sin\theta)$$
$$\mathbf{x}_\phi = (-R\sin\theta\sin\phi,\, R\sin\theta\cos\phi,\, 0)$$

The metric coefficients:

$$E = R^2, \quad F = 0, \quad G = R^2\sin^2\theta.$$

So the first fundamental form of the sphere is $ds^2 = R^2\,d\theta^2 + R^2\sin^2\theta\,d\phi^2$. The area element is $\sqrt{EG - F^2} = R^2\sin\theta$, and integrating gives $A = 4\pi R^2$.

The vanishing of $F$ means the coordinate lines ($\theta$-curves and $\phi$-curves, i.e., meridians and parallels) are everywhere orthogonal.

## Example: a surface of revolution

Take a profile curve $(f(u), 0, g(u))$ with $f(u) > 0$ and rotate it about the $z$-axis:

$$\mathbf{x}(u,v) = (f(u)\cos v,\, f(u)\sin v,\, g(u)).$$

Then:

$$E = f'^2 + g'^2, \quad F = 0, \quad G = f^2.$$

If the profile is parametrized by arc length ($f'^2 + g'^2 = 1$), this simplifies to $E = 1$, $F = 0$, $G = f(u)^2$.

**Theorem.** The coordinate curves of a parametrization are orthogonal if and only if $F = 0$.

*Proof.* The $u$-curves have tangent $\mathbf{x}_u$, the $v$-curves have tangent $\mathbf{x}_v$. They are orthogonal iff $\mathbf{x}_u \cdot \mathbf{x}_v = F = 0$.

## Isometries and the intrinsic viewpoint

Two surfaces are **isometric** if there is a diffeomorphism between them that preserves the first fundamental form. Isometric surfaces have identical intrinsic geometry: same lengths, same angles, same areas — even if they look completely different in $\mathbb{R}^3$.

**Theorem.** A flat piece of paper and a cylinder are locally isometric.

*Proof sketch.* The cylinder $\mathbf{x}(u,v) = (\cos u, \sin u, v)$ has $E=1$, $F=0$, $G=1$ — the same as the plane with Cartesian coordinates. Since the metric coefficients agree, the identity map on parameters is an isometry.

You can roll a piece of paper into a cylinder without stretching it. That is the geometric content of this theorem.

## What's next

The first fundamental form tells us about intrinsic measurements. But it says nothing about how the surface curves in space. For that we need the second fundamental form and the shape operator — which lead us to Gaussian and mean curvature.
