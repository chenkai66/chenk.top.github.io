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

A surface is a two-dimensional object living in $\mathbb{R}^3$. The first fundamental form is the tool that lets us do metric geometry — measuring distances, angles, areas — directly on the surface, without reference to the ambient space. Everything in this chapter is about a single $2\times 2$ matrix of functions and the remarkable amount of information it carries.

## Regular surfaces

A **regular surface** $S \subset \mathbb{R}^3$ is a subset such that for each point $p \in S$, there exists a smooth map $\mathbf{x}: U \to S$ (where $U \subset \mathbb{R}^2$ is open) satisfying:

1. $\mathbf{x}$ is a homeomorphism onto its image.
2. The differential $d\mathbf{x}_q$ is injective for all $q \in U$.

The map $\mathbf{x}(u,v)$ is a **parametrization** (or coordinate patch). Condition 2 means the partial derivatives $\mathbf{x}_u$ and $\mathbf{x}_v$ are linearly independent — the surface has a well-defined tangent plane at every point. No self-intersections, no singular points, no edges.

A single parametrization rarely covers the entire surface (the sphere needs at least two patches), so we allow an atlas of overlapping patches with smooth transition maps. But locally, one patch suffices, and that is where we will work.

**Example (sphere).** The sphere $S^2$ of radius $R$: parametrize by $\mathbf{x}(\theta, \phi) = (R\sin\theta\cos\phi,\, R\sin\theta\sin\phi,\, R\cos\theta)$ for $\theta \in (0,\pi)$, $\phi \in (0,2\pi)$. This covers everything except one meridian and the two poles. At the poles, $\mathbf{x}_\phi = 0$, so we need a different chart there.

**Example (graph surface).** If $f: U \to \mathbb{R}$ is smooth, then $\mathbf{x}(u,v) = (u, v, f(u,v))$ parametrizes the graph $z = f(x,y)$. It is always regular because $\mathbf{x}_u \times \mathbf{x}_v = (-f_u, -f_v, 1) \neq 0$.

## Coordinate patches and atlases

The requirement that a single chart cannot always cover an entire surface leads to the notion of an **atlas**: a collection of coordinate patches $\{(U_\alpha, \mathbf{x}_\alpha)\}$ whose images cover $S$. Where two patches overlap, the **transition map** $\mathbf{x}_\beta^{-1} \circ \mathbf{x}_\alpha$ must be smooth — this is what gives the surface a consistent differentiable structure.

For the sphere, stereographic projection from the north pole covers everything except the north pole itself; stereographic projection from the south pole covers everything except the south pole. Together, two charts suffice. The transition map between them (defined on the sphere minus both poles) is the inversion $w \mapsto w/|w|^2$ — smooth and invertible.

The torus $T^2$ can be covered by four rectangular patches (think of the flat square with edges identified — each patch avoids one pair of identified edges). The projective plane $\mathbb{RP}^2$ needs at least three charts.

The key principle: all geometric quantities we define (metric, curvature, geodesics) must be independent of which chart we use. They are properties of the surface, not of the coordinates. Different parametrizations give different coefficient functions $E, F, G$, but the geometric content — lengths, angles, areas — is invariant.


![Surface patch with tangent plane and normal vector](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_fig2_tangent_plane.png)

## The tangent plane

At a point $p = \mathbf{x}(u_0, v_0)$, the tangent plane $T_pS$ is spanned by $\mathbf{x}_u$ and $\mathbf{x}_v$. Any tangent vector $\mathbf{w} \in T_pS$ can be written as

$$\mathbf{w} = a\,\mathbf{x}_u + b\,\mathbf{x}_v$$

for some $a, b \in \mathbb{R}$. The unit normal to the surface is

$$\mathbf{N} = \frac{\mathbf{x}_u \times \mathbf{x}_v}{|\mathbf{x}_u \times \mathbf{x}_v|}.$$

The tangent plane is the best linear approximation to the surface at $p$: the surface deviates from $T_pS$ only at second order. If you zoom in enough, any smooth surface looks flat — the tangent plane captures that flatness.

## The first fundamental form

The **first fundamental form** (or metric) is the restriction of the dot product in $\mathbb{R}^3$ to the tangent plane. For tangent vectors $\mathbf{w}_1, \mathbf{w}_2 \in T_pS$, it is simply $I(\mathbf{w}_1, \mathbf{w}_2) = \mathbf{w}_1 \cdot \mathbf{w}_2$.

In coordinates, if $\mathbf{w} = a\,\mathbf{x}_u + b\,\mathbf{x}_v$, then the squared length is:

$$I(\mathbf{w}) = |\mathbf{w}|^2 = E\,a^2 + 2F\,ab + G\,b^2$$

where the **metric coefficients** are:

$$E = \mathbf{x}_u \cdot \mathbf{x}_u, \quad F = \mathbf{x}_u \cdot \mathbf{x}_v, \quad G = \mathbf{x}_v \cdot \mathbf{x}_v.$$

In matrix form:

$$\begin{pmatrix} E & F \\ F & G \end{pmatrix}$$

This symmetric positive-definite matrix — also written $ds^2 = E\,du^2 + 2F\,du\,dv + G\,dv^2$ — encodes all intrinsic metric information about the surface. Lengths, angles, areas, geodesics, curvature: they all come from here.

The first fundamental form is "intrinsic" because it describes what a creature living on the surface can measure without any awareness of the ambient $\mathbb{R}^3$. A two-dimensional being with a ruler and protractor can determine $E$, $F$, $G$ — by measuring distances between nearby points.

## Computing with the metric

### Arc length

A curve $\alpha(t) = \mathbf{x}(u(t), v(t))$ on the surface has length:

$$L = \int_a^b \sqrt{E\,u'^2 + 2F\,u'v' + G\,v'^2}\, dt.$$

This is just $\int |\alpha'|\,dt$ written in coordinates.

### Area

The area of a region $R = \mathbf{x}(D)$ is:

$$A = \iint_D |\mathbf{x}_u \times \mathbf{x}_v|\, du\,dv = \iint_D \sqrt{EG - F^2}\, du\,dv.$$

The quantity $\sqrt{EG - F^2}$ is the area element — it tells you how much a tiny $du\,dv$ rectangle in parameter space corresponds to actual area on the surface. When $F = 0$ and $E = G = 1$ (an isometric parametrization), area in parameter space equals area on the surface.

**Example (area of a graph surface).** For $z = f(x,y)$ over a domain $D$, we have $E = 1+f_x^2$, $F = f_xf_y$, $G = 1+f_y^2$. The area element is $\sqrt{EG-F^2} = \sqrt{1+f_x^2+f_y^2}$, giving the familiar surface area formula:

$$A = \iint_D \sqrt{1 + f_x^2 + f_y^2}\,dx\,dy.$$

For a nearly flat surface ($|f_x|, |f_y| \ll 1$), this approximates the area of $D$ itself — the surface barely differs from its projection.

### Angle between curves

If two curves on the surface meet at a point, with tangent vectors $\mathbf{w}_1 = a_1\mathbf{x}_u + b_1\mathbf{x}_v$ and $\mathbf{w}_2 = a_2\mathbf{x}_u + b_2\mathbf{x}_v$, the angle $\theta$ between them satisfies:

$$\cos\theta = \frac{I(\mathbf{w}_1, \mathbf{w}_2)}{|\mathbf{w}_1|\,|\mathbf{w}_2|} = \frac{Ea_1a_2 + F(a_1b_2+a_2b_1) + Gb_1b_2}{\sqrt{Ea_1^2+2Fa_1b_1+Gb_1^2}\,\sqrt{Ea_2^2+2Fa_2b_2+Gb_2^2}}.$$

**Theorem.** The coordinate curves of a parametrization are orthogonal if and only if $F = 0$.

*Proof.* The $u$-curves have tangent $\mathbf{x}_u$, the $v$-curves have tangent $\mathbf{x}_v$. They meet at right angles iff $\cos\theta = \mathbf{x}_u \cdot \mathbf{x}_v / (|\mathbf{x}_u||\mathbf{x}_v|) = 0$, which happens iff $F = \mathbf{x}_u \cdot \mathbf{x}_v = 0$.

A parametrization with $F = 0$ everywhere is called **orthogonal**. It is the most convenient choice for computation — many formulas simplify dramatically.

## Example: the sphere

For the sphere of radius $R$ with the standard parametrization $\mathbf{x}(\theta,\phi)$:

$$\mathbf{x}_\theta = (R\cos\theta\cos\phi,\, R\cos\theta\sin\phi,\, -R\sin\theta)$$
$$\mathbf{x}_\phi = (-R\sin\theta\sin\phi,\, R\sin\theta\cos\phi,\, 0)$$

Computing the metric coefficients:

$$E = R^2, \quad F = 0, \quad G = R^2\sin^2\theta.$$

So the first fundamental form of the sphere is:

$$ds^2 = R^2\,d\theta^2 + R^2\sin^2\theta\,d\phi^2.$$

The area element is $\sqrt{EG - F^2} = R^2\sin\theta$. Integrating over $\theta \in [0,\pi]$, $\phi \in [0,2\pi]$ gives the familiar $A = 4\pi R^2$.

The vanishing of $F$ means meridians ($\theta$-curves) and parallels ($\phi$-curves) are everywhere orthogonal. That is obvious geometrically, but the metric makes it precise.

**Example (distance on the sphere).** A curve of constant latitude $\theta_0$ from $\phi = 0$ to $\phi = \phi_1$ has length $L = \int_0^{\phi_1}\sqrt{G}\,d\phi = R\sin\theta_0\cdot\phi_1$. Near the poles ($\theta_0$ small), parallels are short. At the equator ($\theta_0 = \pi/2$), the length is $R\phi_1$ — maximized.

## Example: surface of revolution

Take a profile curve $(f(u), 0, g(u))$ with $f(u) > 0$ and rotate it about the $z$-axis:

$$\mathbf{x}(u,v) = (f(u)\cos v,\, f(u)\sin v,\, g(u)).$$

Then:

$$E = f'^2 + g'^2, \quad F = 0, \quad G = f^2.$$

If the profile is parametrized by arc length ($f'^2 + g'^2 = 1$), this simplifies to $E = 1$, $F = 0$, $G = f(u)^2$.

**Example (torus).** Take $f(u) = R + r\cos u$, $g(u) = r\sin u$ (a circle of radius $r$ centered at distance $R$ from the axis). Then $E = r^2$, $F = 0$, $G = (R+r\cos u)^2$. The metric of the torus is:

$$ds^2 = r^2\,du^2 + (R+r\cos u)^2\,dv^2.$$

The area is $\int_0^{2\pi}\int_0^{2\pi}r(R+r\cos u)\,du\,dv = 4\pi^2 Rr$. Note that the area depends on both the tube radius $r$ and the distance $R$ from the axis.

## Example: the cylinder and the flat metric

The cylinder $\mathbf{x}(u,v) = (\cos u, \sin u, v)$ has $E = 1$, $F = 0$, $G = 1$. Its metric is $ds^2 = du^2 + dv^2$ — identical to the Euclidean metric on the plane. This is the simplest non-trivial example of what comes next.

## Isometries and the intrinsic viewpoint

Two surfaces $S_1$ and $S_2$ are **isometric** if there is a diffeomorphism $\varphi: S_1 \to S_2$ that preserves the first fundamental form: $I_1(\mathbf{v}) = I_2(d\varphi(\mathbf{v}))$ for all tangent vectors $\mathbf{v}$. Isometric surfaces have identical intrinsic geometry: same lengths, same angles, same areas — even if they look completely different in $\mathbb{R}^3$.

**Theorem.** A flat piece of paper (the plane) and a cylinder are locally isometric.

*Proof.* The cylinder $\mathbf{x}(u,v) = (\cos u, \sin u, v)$ has $E = 1$, $F = 0$, $G = 1$ — the same metric as the plane with Cartesian coordinates. The identity map on parameters is an isometry.

You can roll a sheet of paper into a cylinder without stretching or tearing it. That is the geometric content of this theorem. The bending changes how the surface sits in space (its *extrinsic* geometry) but preserves all distances measured within the surface (its *intrinsic* geometry).

**Theorem (rigidity of the sphere).** A sphere is not locally isometric to the plane. Any parametrization of the sphere has $EG - F^2 = R^4\sin^2\theta \neq \text{const}$ (in standard coordinates), so it cannot match the flat metric $EG - F^2 = 1$.

This is why every flat map of the Earth distorts something. The Mercator projection preserves angles ($F = 0$ and the metric is conformal), but it stretches areas near the poles. The equal-area projections preserve area but distort angles. No map does both — because the sphere and the plane are not isometric.

**Example (isometric surfaces with different shapes).** A helicoid and a catenoid are locally isometric — there is a smooth one-parameter family of isometries connecting them (the "associated family"). They look nothing alike in $\mathbb{R}^3$: the helicoid is a ruled surface resembling a spiral staircase, while the catenoid is a surface of revolution shaped like two trumpets. Yet an ant on either surface, measuring only distances and angles, could not tell which one it inhabits.

## Conformal maps

A weaker condition than isometry is **conformality**: a map preserves angles but may scale lengths. A parametrization is **conformal** (or isothermal) if $E = G = \lambda^2(u,v)$ and $F = 0$, so the metric takes the form $ds^2 = \lambda^2(du^2 + dv^2)$. This means the surface looks like the plane up to a position-dependent scaling.

**Theorem.** Every smooth surface admits local conformal (isothermal) coordinates.

This is a deep result — it amounts to solving the Beltrami equation, a PDE. But its existence means we can always find coordinates where the metric is a scalar multiple of the flat metric, which greatly simplifies analysis.

## What the first fundamental form cannot see

The first fundamental form tells us everything intrinsic, but nothing about how the surface bends in space. A plane and a cylinder have the same first fundamental form, yet one is flat and the other is curved (in the ambient sense). To detect this bending, we need the second fundamental form and the shape operator — which measure how the normal vector changes as we move along the surface. That is the subject of the next chapter.

---

*This is Part 2 of [Differential Geometry](/en/series/differential-geometry/) (6 parts).
Previous: [Part 1 — Curves in Space](/en/differential-geometry/01-curves-in-space/) · Next: [Part 3 — Gaussian Curvature and the Second Fundamental Form](/en/differential-geometry/03-curvature-of-surfaces/)*
