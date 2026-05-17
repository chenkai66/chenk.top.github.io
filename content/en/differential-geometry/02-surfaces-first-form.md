---
title: "Surfaces and the First Fundamental Form: Intrinsic Measurements"
date: 2021-11-03 09:00:00
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
series_total: 12
translationKey: "differential-geometry-2"
description: "Regular surfaces, coordinate patches, the tangent plane, and the first fundamental form — how to measure lengths, angles, and areas on a surface without leaving it."
---

In the previous chapter we developed the complete local theory of space curves: two scalar invariants ($\kappa$ and $\tau$) determined the curve up to rigid motion. Curves are one-dimensional, and their geometry was governed by a system of ODEs.

Now we move to surfaces — two-dimensional objects in $\mathbb{R}^3$. The jump from one to two dimensions is qualitatively different. A surface has an internal geometry of its own: distances, angles, and areas can be measured by a creature living on the surface, without any reference to the surrounding space. The tool that makes this possible is the *first fundamental form*, a symmetric bilinear form on the tangent plane that encodes the surface's intrinsic metric. This chapter constructs that tool and uses it to compute lengths and areas on concrete surfaces.

---

## What Is a Surface?

Before giving a formal definition, let us build some intuition. A surface in $\mathbb{R}^3$ is a subset that "looks locally like a piece of $\mathbb{R}^2$." The sphere $x^2 + y^2 + z^2 = 1$ is a surface; so is the torus; so is the graph of any smooth function $z = f(x,y)$. What these have in common is that near any point, you can set up two coordinates that smoothly label the points of the surface — like latitude and longitude on the sphere, or the $(x,y)$ coordinates on a graph.

This idea of "local coordinates" is the starting point. A surface is not defined by a single equation or a single parametrization, but by the existence of enough parametrizations to cover it.

**Definition (Coordinate patch).** A *coordinate patch* (or *parametrization*) of a subset $S \subseteq \mathbb{R}^3$ is a smooth map $\mathbf{x}: U \to S$, where $U \subseteq \mathbb{R}^2$ is open, such that:

1. $\mathbf{x}$ is a homeomorphism onto its image $\mathbf{x}(U) \subseteq S$ (continuous with continuous inverse).
2. The partial derivatives $\mathbf{x}_u = \partial \mathbf{x}/\partial u$ and $\mathbf{x}_v = \partial \mathbf{x}/\partial v$ are linearly independent at every point of $U$.

Condition (2) is the *regularity condition*: it ensures that the image is a genuine two-dimensional surface, not something that collapses to a curve or a point. In coordinates, $\mathbf{x}(u,v) = (x(u,v),\, y(u,v),\, z(u,v))$, and the regularity condition says that the Jacobian matrix has rank 2 everywhere, or equivalently that $\mathbf{x}_u \times \mathbf{x}_v \neq 0$.

**Definition (Regular surface).** A subset $S \subseteq \mathbb{R}^3$ is a *regular surface* if for every point $p \in S$, there exists a coordinate patch $\mathbf{x}: U \to S$ whose image contains $p$.

This means $S$ can be covered by (possibly many overlapping) coordinate patches. The same surface can be parametrized in different ways, just as the same curve could be reparametrized. The challenge of surface theory is to extract quantities that do not depend on the choice of parametrization.

---


![Surface patch with tangent plane and normal vector](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_fig2_tangent_plane.png)

## Regular Surfaces and Coordinate Patches

Let us see the definition in action on several important examples.

**Example 1 (The sphere).** The unit sphere $S^2 = \{(x,y,z) : x^2+y^2+z^2 = 1\}$ can be parametrized by spherical coordinates:

$$\mathbf{x}(\theta, \varphi) = (\sin\theta\cos\varphi,\; \sin\theta\sin\varphi,\; \cos\theta),$$

where $\theta \in (0, \pi)$ and $\varphi \in (0, 2\pi)$. The partial derivatives are:

$$\mathbf{x}_\theta = (\cos\theta\cos\varphi,\; \cos\theta\sin\varphi,\; -\sin\theta),$$
$$\mathbf{x}_\varphi = (-\sin\theta\sin\varphi,\; \sin\theta\cos\varphi,\; 0).$$

One computes $\mathbf{x}_\theta \times \mathbf{x}_\varphi = \sin\theta\,(\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta) = \sin\theta\,\mathbf{x}$. Since $\sin\theta > 0$ for $\theta \in (0,\pi)$, this is nonzero, confirming regularity. However, this parametrization misses the north pole ($\theta = 0$) and the south pole ($\theta = \pi$), and it is not injective along the meridian $\varphi = 0 = 2\pi$. To cover the entire sphere, we need additional patches — for instance, projecting from the north and south poles onto the equatorial plane (stereographic projection provides two patches that together cover all of $S^2$).

**Example 2 (The torus).** The torus with outer radius $R$ and tube radius $r$ ($0 < r < R$) is parametrized by:

$$\mathbf{x}(\theta, \varphi) = ((R + r\cos\theta)\cos\varphi,\; (R + r\cos\theta)\sin\varphi,\; r\sin\theta),$$

where $\theta, \varphi \in (0, 2\pi)$. The parameter $\theta$ goes around the tube and $\varphi$ goes around the central hole. Computing:

$$\mathbf{x}_\theta = (-r\sin\theta\cos\varphi,\; -r\sin\theta\sin\varphi,\; r\cos\theta),$$
$$\mathbf{x}_\varphi = (-(R+r\cos\theta)\sin\varphi,\; (R+r\cos\theta)\cos\varphi,\; 0).$$

The cross product $\mathbf{x}_\theta \times \mathbf{x}_\varphi$ has magnitude $r(R + r\cos\theta)$. Since $R > r > 0$, we have $R + r\cos\theta \geq R - r > 0$, so the parametrization is regular.

**Example 3 (Graph of a function).** If $f: U \to \mathbb{R}$ is smooth, then $S = \{(x, y, f(x,y)) : (x,y) \in U\}$ is a regular surface with the obvious parametrization $\mathbf{x}(u,v) = (u, v, f(u,v))$. Here:

$$\mathbf{x}_u = (1, 0, f_u), \quad \mathbf{x}_v = (0, 1, f_v),$$
$$\mathbf{x}_u \times \mathbf{x}_v = (-f_u, -f_v, 1).$$

This is never zero, so the regularity condition is automatically satisfied for any smooth function $f$. This is why graphs are the simplest examples of regular surfaces.

**Example 4 (Surface of revolution).** Take a curve $\alpha(t) = (r(t), 0, z(t))$ in the $xz$-plane with $r(t) > 0$, and rotate it around the $z$-axis. The resulting surface is:

$$\mathbf{x}(t, \varphi) = (r(t)\cos\varphi,\; r(t)\sin\varphi,\; z(t)),$$

where $\varphi \in (0, 2\pi)$. This includes the sphere (rotating a semicircle), the torus (rotating a circle offset from the axis), the cylinder (rotating a vertical line), and the cone (rotating a slanted line).

The partial derivatives are:

$$\mathbf{x}_t = (r'\cos\varphi,\; r'\sin\varphi,\; z'), \quad \mathbf{x}_\varphi = (-r\sin\varphi,\; r\cos\varphi,\; 0).$$

The cross product gives $\mathbf{x}_t \times \mathbf{x}_\varphi = (-rz'\cos\varphi, -rz'\sin\varphi, rr')$, with magnitude $r\sqrt{(z')^2 + (r')^2}$. Regularity requires $r > 0$ and $\alpha'(t) \neq 0$ (the generating curve is regular).

---

## The Tangent Plane and Normal Vector

At each point of a regular surface, the partial derivatives $\mathbf{x}_u$ and $\mathbf{x}_v$ span a two-dimensional subspace of $\mathbb{R}^3$: the tangent plane.

**Definition (Tangent plane).** Let $S$ be a regular surface with parametrization $\mathbf{x}: U \to S$, and let $p = \mathbf{x}(u_0, v_0)$. The *tangent plane* to $S$ at $p$ is

$$T_p S = \text{span}\{\mathbf{x}_u(u_0, v_0),\, \mathbf{x}_v(u_0, v_0)\}.$$

A tangent vector at $p$ is any vector $w \in T_p S$. It can be written as $w = a\,\mathbf{x}_u + b\,\mathbf{x}_v$ for some scalars $a, b$. Equivalently, a tangent vector is the velocity of a curve on the surface passing through $p$: if $\gamma(t) = \mathbf{x}(u(t), v(t))$ with $\gamma(0) = p$, then $\gamma'(0) = u'(0)\,\mathbf{x}_u + v'(0)\,\mathbf{x}_v \in T_p S$.

The tangent plane is independent of the choice of parametrization. If $\mathbf{y}(s,t)$ is a different parametrization of the same portion of $S$, then $\text{span}\{\mathbf{y}_s, \mathbf{y}_t\} = \text{span}\{\mathbf{x}_u, \mathbf{x}_v\}$ at the corresponding point (this follows from the chain rule and the invertibility of the change-of-coordinates map).

**Definition (Unit normal).** The *unit normal vector* at $p$ is

$$\mathbf{n}(u_0, v_0) = \frac{\mathbf{x}_u \times \mathbf{x}_v}{|\mathbf{x}_u \times \mathbf{x}_v|}(u_0, v_0).$$

Since $\mathbf{x}_u$ and $\mathbf{x}_v$ are linearly independent, $\mathbf{x}_u \times \mathbf{x}_v \neq 0$, and $\mathbf{n}$ is well-defined. It is perpendicular to the tangent plane. The choice of $\mathbf{n}$ or $-\mathbf{n}$ depends on the orientation of the parametrization (swapping $u$ and $v$ flips the sign). A surface is *orientable* if a consistent choice of normal can be made over the entire surface; the Mobius strip is the standard example of a non-orientable surface.

**Example 5 (Normal to the sphere).** For $\mathbf{x}(\theta, \varphi) = (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta)$, we computed $\mathbf{x}_\theta \times \mathbf{x}_\varphi = \sin\theta\,\mathbf{x}$, which has magnitude $\sin\theta$ (since $|\mathbf{x}| = 1$). So:

$$\mathbf{n} = \frac{\sin\theta\,\mathbf{x}}{\sin\theta} = \mathbf{x} = (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta).$$

The outward unit normal to the unit sphere at a point $p$ is the position vector $p$ itself. This is geometrically obvious: the vector from the origin to a point on the sphere points radially outward, perpendicular to the sphere.

**Example 6 (Normal to a graph).** For $\mathbf{x}(u,v) = (u, v, f(u,v))$, we have $\mathbf{x}_u \times \mathbf{x}_v = (-f_u, -f_v, 1)$, so:

$$\mathbf{n} = \frac{(-f_u, -f_v, 1)}{\sqrt{1 + f_u^2 + f_v^2}}.$$

At a critical point of $f$ where $f_u = f_v = 0$, the normal is $(0, 0, 1)$ — pointing straight up. The tangent plane there is horizontal. This matches the intuition that a local maximum or minimum of a function has a horizontal tangent plane.

---

## The First Fundamental Form: Measuring Lengths and Angles on Surfaces

We now come to the central construction of this chapter. Every surface inherits a notion of distance from the ambient $\mathbb{R}^3$ (the distance between two points on the surface is the length of the shortest path on the surface connecting them, not the straight-line distance through space). The first fundamental form is the infinitesimal version of this notion.

**Definition (First fundamental form).** Let $S$ be a regular surface with parametrization $\mathbf{x}(u,v)$. The *first fundamental form* is the restriction of the Euclidean inner product of $\mathbb{R}^3$ to the tangent plane $T_p S$. For tangent vectors $w_1, w_2 \in T_p S$, we define $I_p(w_1, w_2) = w_1 \cdot w_2$ (the ordinary dot product in $\mathbb{R}^3$).

This seems almost trivially simple: we are just taking the dot product. But the power lies in expressing it in terms of the surface coordinates $(u,v)$. If $w = a\,\mathbf{x}_u + b\,\mathbf{x}_v$, then:

$$I(w, w) = w \cdot w = a^2(\mathbf{x}_u \cdot \mathbf{x}_u) + 2ab(\mathbf{x}_u \cdot \mathbf{x}_v) + b^2(\mathbf{x}_v \cdot \mathbf{x}_v).$$

Introducing the standard notation:

$$E = \mathbf{x}_u \cdot \mathbf{x}_u, \quad F = \mathbf{x}_u \cdot \mathbf{x}_v, \quad G = \mathbf{x}_v \cdot \mathbf{x}_v,$$

the first fundamental form becomes:

$$I = E\, du^2 + 2F\, du\, dv + G\, dv^2.$$

In matrix notation, if we represent a tangent vector by its coordinate vector $(du, dv)^T$, then:

$$I = \begin{pmatrix} du & dv \end{pmatrix} \begin{pmatrix} E & F \\ F & G \end{pmatrix} \begin{pmatrix} du \\ dv \end{pmatrix}.$$

The $2 \times 2$ matrix $\begin{pmatrix} E & F \\ F & G \end{pmatrix}$ is the *metric tensor* (or *Gram matrix*) of the surface in coordinates $(u,v)$. It is symmetric and positive definite (since $E > 0$, $G > 0$, and $EG - F^2 = |\mathbf{x}_u \times \mathbf{x}_v|^2 > 0$ by regularity).

The coefficients $E$, $F$, $G$ are functions of $(u,v)$ — they vary from point to point on the surface. The first fundamental form packages all the information needed to measure lengths, angles, and areas on the surface.

**Example 7 (First fundamental form of the plane).** For $\mathbf{x}(u,v) = (u, v, 0)$, we have $\mathbf{x}_u = (1,0,0)$, $\mathbf{x}_v = (0,1,0)$, so $E = 1$, $F = 0$, $G = 1$. The first fundamental form is $I = du^2 + dv^2$, which is just the Euclidean metric. The plane is flat: no distortion.

**Example 8 (First fundamental form of the sphere).** Using spherical coordinates $\mathbf{x}(\theta, \varphi) = (R\sin\theta\cos\varphi, R\sin\theta\sin\varphi, R\cos\theta)$:

$$E = \mathbf{x}_\theta \cdot \mathbf{x}_\theta = R^2(\cos^2\theta\cos^2\varphi + \cos^2\theta\sin^2\varphi + \sin^2\theta) = R^2.$$

$$G = \mathbf{x}_\varphi \cdot \mathbf{x}_\varphi = R^2(\sin^2\theta\sin^2\varphi + \sin^2\theta\cos^2\varphi) = R^2\sin^2\theta.$$

$$F = \mathbf{x}_\theta \cdot \mathbf{x}_\varphi = R^2(-\cos\theta\cos\varphi\sin\theta\sin\varphi + \cos\theta\sin\varphi\sin\theta\cos\varphi) = 0.$$

Therefore:

$$I = R^2\, d\theta^2 + R^2\sin^2\theta\, d\varphi^2.$$

The vanishing of $F$ means that the coordinate lines (meridians and parallels) are orthogonal. The coefficient $G = R^2\sin^2\theta$ depends on $\theta$: near the poles ($\theta$ near 0 or $\pi$), the parallels have small circumference, and $G$ is small. At the equator ($\theta = \pi/2$), $G = R^2$ is maximal. This is why the Mercator projection, which represents the sphere as a rectangle, necessarily distorts areas near the poles.

**Example 9 (First fundamental form of a surface of revolution).** For $\mathbf{x}(t, \varphi) = (r(t)\cos\varphi, r(t)\sin\varphi, z(t))$:

$$E = \mathbf{x}_t \cdot \mathbf{x}_t = (r')^2 + (z')^2, \quad F = \mathbf{x}_t \cdot \mathbf{x}_\varphi = 0, \quad G = \mathbf{x}_\varphi \cdot \mathbf{x}_\varphi = r^2.$$

So $I = [(r')^2 + (z')^2]\, dt^2 + r^2\, d\varphi^2$. If the generating curve is parametrized by arc length (so that $(r')^2 + (z')^2 = 1$), this simplifies to $I = dt^2 + r(t)^2\, d\varphi^2$. The metric depends only on the function $r(t)$ — the "profile" of the surface.

**Example 10 (First fundamental form of the torus).** For the torus $\mathbf{x}(\theta, \varphi) = ((R+r\cos\theta)\cos\varphi, (R+r\cos\theta)\sin\varphi, r\sin\theta)$:

$$E = \mathbf{x}_\theta \cdot \mathbf{x}_\theta = r^2\sin^2\theta\cos^2\varphi + r^2\sin^2\theta\sin^2\varphi + r^2\cos^2\theta = r^2.$$

$$G = \mathbf{x}_\varphi \cdot \mathbf{x}_\varphi = (R+r\cos\theta)^2\sin^2\varphi + (R+r\cos\theta)^2\cos^2\varphi = (R+r\cos\theta)^2.$$

$$F = \mathbf{x}_\theta \cdot \mathbf{x}_\varphi = 0.$$

So $I = r^2\, d\theta^2 + (R + r\cos\theta)^2\, d\varphi^2$. The coefficient $G$ varies: it is largest on the outer equator ($\theta = 0$, $G = (R+r)^2$) and smallest on the inner equator ($\theta = \pi$, $G = (R-r)^2$). This reflects the fact that the outer equator is longer than the inner equator.

**Angle between coordinate curves.** If two curves on the surface meet at a point $p$, the angle $\theta$ between their tangent vectors $w_1, w_2$ satisfies:

$$\cos\theta = \frac{I(w_1, w_2)}{\sqrt{I(w_1, w_1)} \cdot \sqrt{I(w_2, w_2)}} = \frac{w_1 \cdot w_2}{|w_1| \cdot |w_2|}.$$

In particular, the coordinate curves ($u$-curves with $v$ constant, and $v$-curves with $u$ constant) are orthogonal if and only if $F = 0$. This happened in all our examples above. A parametrization with $F = 0$ everywhere is called an *orthogonal parametrization*, and it simplifies many computations. Not every surface admits a global orthogonal parametrization, but locally one always exists (a classical result in surface theory).

---

## Arc Length and Area via the First Form

The first fundamental form transforms all metric computations on the surface into integrals involving $E$, $F$, $G$.

**Arc length.** Let $\gamma(t) = \mathbf{x}(u(t), v(t))$ for $t \in [a, b]$ be a curve on the surface. Its velocity is $\gamma'(t) = u'(t)\,\mathbf{x}_u + v'(t)\,\mathbf{x}_v$, so:

$$|\gamma'(t)|^2 = E\,(u')^2 + 2F\,u'\,v' + G\,(v')^2.$$

The arc length is:

$$L(\gamma) = \int_a^b |\gamma'(t)|\, dt = \int_a^b \sqrt{E\,(u')^2 + 2F\,u'\,v' + G\,(v')^2}\, dt.$$

**Example 11 (Length of a parallel on the sphere).** A parallel of latitude $\theta_0$ on the sphere of radius $R$ is the curve $\gamma(t) = \mathbf{x}(\theta_0, t)$ for $t \in [0, 2\pi]$. Here $u(t) = \theta_0$ (constant), $v(t) = t$, so $u' = 0$, $v' = 1$. With $E = R^2$, $F = 0$, $G = R^2\sin^2\theta_0$:

$$L = \int_0^{2\pi} \sqrt{R^2\sin^2\theta_0}\, dt = 2\pi R\sin\theta_0.$$

At the equator ($\theta_0 = \pi/2$), $L = 2\pi R$. At latitude $\theta_0$ from the north pole, the circumference is $2\pi R \sin\theta_0$, which decreases toward the poles. This is an elementary result, but the derivation via the first fundamental form generalizes immediately to any surface.

**Example 12 (Length of a helix on a cylinder).** The cylinder of radius $a$ is parametrized by $\mathbf{x}(t, \varphi) = (a\cos\varphi, a\sin\varphi, t)$ with $E = 1$, $F = 0$, $G = a^2$. A helix on the cylinder is $\gamma(s) = \mathbf{x}(bs, s)$ (rising by $b$ per radian), so $u' = b$, $v' = 1$, and:

$$|\gamma'|^2 = 1 \cdot b^2 + 0 + a^2 \cdot 1 = b^2 + a^2.$$

The speed is $\sqrt{a^2 + b^2}$, constant — consistent with our calculation from the previous chapter. One full turn ($s \in [0, 2\pi]$) has length $2\pi\sqrt{a^2 + b^2}$.

**Area.** The area of a region $D$ on the surface, corresponding to a domain $D_0 \subseteq U$ in the parameter space, is:

$$\text{Area}(D) = \iint_{D_0} |\mathbf{x}_u \times \mathbf{x}_v|\, du\, dv = \iint_{D_0} \sqrt{EG - F^2}\, du\, dv.$$

The second equality follows from the identity $|\mathbf{x}_u \times \mathbf{x}_v|^2 = |\mathbf{x}_u|^2 |\mathbf{x}_v|^2 - (\mathbf{x}_u \cdot \mathbf{x}_v)^2 = EG - F^2$, which is a consequence of the vector identity $|a \times b|^2 = |a|^2|b|^2 - (a \cdot b)^2$.

The quantity $\sqrt{EG - F^2}$ is the *area element*: it plays the role of the Jacobian determinant for integration on the surface.

**Example 13 (Area of the sphere).** On the sphere of radius $R$, $E = R^2$, $F = 0$, $G = R^2\sin^2\theta$, so $\sqrt{EG - F^2} = R^2\sin\theta$. The area is:

$$\text{Area}(S^2) = \int_0^{2\pi}\int_0^{\pi} R^2\sin\theta\, d\theta\, d\varphi = R^2 \cdot 2\pi \cdot [-\cos\theta]_0^{\pi} = R^2 \cdot 2\pi \cdot 2 = 4\pi R^2.$$

This is the well-known formula for the surface area of a sphere. The derivation from the first fundamental form is clean and generalizes easily.

**Example 14 (Area of the torus).** For the torus with $E = r^2$, $F = 0$, $G = (R+r\cos\theta)^2$:

$$\sqrt{EG - F^2} = r(R + r\cos\theta).$$

$$\text{Area} = \int_0^{2\pi}\int_0^{2\pi} r(R + r\cos\theta)\, d\theta\, d\varphi = 2\pi r \int_0^{2\pi} (R + r\cos\theta)\, d\theta = 2\pi r \cdot 2\pi R = 4\pi^2 Rr.$$

(The $\cos\theta$ term integrates to zero over a full period.) The area of the torus is $4\pi^2 Rr$.

**Example 15 (Area of a graph).** For $z = f(x,y)$ over a region $D_0 \subseteq \mathbb{R}^2$, we have $E = 1 + f_x^2$, $F = f_x f_y$, $G = 1 + f_y^2$, so:

$$EG - F^2 = (1+f_x^2)(1+f_y^2) - f_x^2 f_y^2 = 1 + f_x^2 + f_y^2.$$

The area is:

$$\text{Area} = \iint_{D_0} \sqrt{1 + f_x^2 + f_y^2}\, dx\, dy.$$

This is the standard formula from multivariable calculus for the surface area of a graph. We have derived it as a special case of the general theory.

**Angle formula.** For two tangent vectors $w_1 = a_1 \mathbf{x}_u + b_1 \mathbf{x}_v$ and $w_2 = a_2 \mathbf{x}_u + b_2 \mathbf{x}_v$:

$$\cos\theta = \frac{E a_1 a_2 + F(a_1 b_2 + a_2 b_1) + G b_1 b_2}{\sqrt{(E a_1^2 + 2F a_1 b_1 + G b_1^2)(E a_2^2 + 2F a_2 b_2 + G b_2^2)}}.$$

When $F = 0$ (orthogonal coordinates), this simplifies considerably. In particular, the coordinate curves are orthogonal, and the angle between an arbitrary tangent vector and the $u$-direction is determined solely by the ratio $b/a$ and the coefficients $E, G$.

---

## Isometries and the Intrinsic Viewpoint

The first fundamental form is the complete intrinsic geometry of the surface: any quantity that can be measured without leaving the surface — lengths, angles, areas, geodesic curvature, the Gaussian curvature (as we will see in a later chapter) — is determined by $E$, $F$, $G$ and their derivatives. Two surfaces with the "same" first fundamental form are metrically indistinguishable from the inside.

**Definition (Isometry).** Let $S_1$ and $S_2$ be regular surfaces. A diffeomorphism $\phi: S_1 \to S_2$ is an *isometry* if it preserves the first fundamental form: $I_{S_1}(w, w) = I_{S_2}(d\phi(w), d\phi(w))$ for every tangent vector $w$. In coordinates, if $\phi$ maps $(u,v)$ on $S_1$ to $(s(u,v), t(u,v))$ on $S_2$, then the metric tensors satisfy:

$$\begin{pmatrix} E_1 & F_1 \\ F_1 & G_1 \end{pmatrix} = J^T \begin{pmatrix} E_2 & F_2 \\ F_2 & G_2 \end{pmatrix} J,$$

where $J$ is the Jacobian of the coordinate change.

If an isometry exists between $S_1$ and $S_2$, they are said to be *isometric*. Isometric surfaces have the same intrinsic geometry — a being living on $S_1$ cannot distinguish it from $S_2$ by any internal measurement.

**Definition (Local isometry).** A smooth map $\phi: S_1 \to S_2$ is a *local isometry* if it preserves the first fundamental form but need not be globally bijective. This allows for "wrapping" — the same local geometry can have different global topology.

**Example 16 (Plane and cylinder are locally isometric).** The plane with coordinates $(u,v)$ has $I = du^2 + dv^2$. The cylinder of radius $a$, parametrized by $\mathbf{x}(u,v) = (a\cos(v/a), a\sin(v/a), u)$, has:

$$\mathbf{x}_u = (0, 0, 1), \quad \mathbf{x}_v = (-\sin(v/a), \cos(v/a), 0),$$

so $E = 1$, $F = 0$, $G = 1$. The first fundamental form of the cylinder is $I = du^2 + dv^2$ — identical to the plane.

Therefore the map $(u,v) \mapsto (a\cos(v/a), a\sin(v/a), u)$ is a local isometry from the plane to the cylinder. Geometrically, you can roll a piece of paper into a cylinder without stretching or tearing — the intrinsic metric is preserved. Distances measured along the surface, angles between curves, and areas of regions are the same on the flat paper and on the cylinder.

However, this is only a *local* isometry, not a global one: the plane is simply connected while the cylinder is not (you can walk around the cylinder and return to your starting point). The global topology differs even though the local geometry is identical.

**Example 17 (Catenoid and helicoid are locally isometric).** This is one of the most beautiful results in classical surface theory. The *catenoid* is the surface of revolution obtained by rotating the catenary $r = \cosh z$ around the $z$-axis:

$$\mathbf{x}_C(u, v) = (\cosh u \cos v,\; \cosh u \sin v,\; u).$$

The *helicoid* is a ruled surface (think of a spiral staircase):

$$\mathbf{x}_H(u, v) = (\sinh u \cos v,\; \sinh u \sin v,\; v).$$

Computing the first fundamental forms:

For the catenoid: $E = 1 + \sinh^2 u = \cosh^2 u$, $F = 0$, $G = \cosh^2 u$. So $I_C = \cosh^2 u\,(du^2 + dv^2)$.

For the helicoid: $E = \cosh^2 u$, $F = 0$, $G = \sinh^2 u + 1 = \cosh^2 u$. So $I_H = \cosh^2 u\,(du^2 + dv^2)$.

The first fundamental forms are identical! The catenoid and helicoid are locally isometric, despite looking completely different in $\mathbb{R}^3$: the catenoid is a surface of revolution, while the helicoid is a ruled surface. From the intrinsic point of view, they have exactly the same geometry.

In fact, there is a continuous one-parameter family of surfaces interpolating between the catenoid and the helicoid, all locally isometric to each other. This deformation can be visualized as a physical "bending" of the surface without stretching — like bending a piece of sheet metal. The first fundamental form is preserved throughout the deformation; only the way the surface sits in $\mathbb{R}^3$ changes.

This example powerfully illustrates the distinction between intrinsic and extrinsic geometry. The extrinsic shape (how the surface is embedded in space) is different, but the intrinsic metric (measured by beings living on the surface) is the same. Any property that depends only on the first fundamental form — such as geodesics, areas, or Gaussian curvature — must be the same for the catenoid and the helicoid. (We will verify this for Gaussian curvature in a later chapter: both surfaces have the same Gaussian curvature function.)

**Example 18 (No isometry from sphere to plane).** Consider any map from the sphere to the plane. The sphere of radius $R$ has first fundamental form $I = R^2 d\theta^2 + R^2\sin^2\theta\, d\varphi^2$, while the plane has $I = du^2 + dv^2$. For these to be equal under some coordinate change, we would need the ratio $G/E$ to be constant (since on the plane $G/E = 1$), but on the sphere $G/E = \sin^2\theta$, which varies with $\theta$. More rigorously, the Gaussian curvature of the sphere is $K = 1/R^2 > 0$ while the plane has $K = 0$, and since $K$ is determined by the first fundamental form (Gauss's Theorema Egregium, to be proved in a later chapter), no isometry can exist. This is the mathematical reason why every flat map of the earth necessarily distorts either distances, angles, areas, or some combination thereof.

**Coordinate invariance.** When we change coordinates on the surface — say from $(u,v)$ to $(\bar{u}, \bar{v})$ — the coefficients $E$, $F$, $G$ change according to the transformation law of a $(0,2)$-tensor:

$$\begin{pmatrix} \bar{E} & \bar{F} \\ \bar{F} & \bar{G} \end{pmatrix} = J^T \begin{pmatrix} E & F \\ F & G \end{pmatrix} J, \quad \text{where } J = \begin{pmatrix} u_{\bar{u}} & u_{\bar{v}} \\ v_{\bar{u}} & v_{\bar{v}} \end{pmatrix}.$$

The first fundamental form $I$ itself is unchanged — only its representation in coordinates changes. This is the beginning of tensor calculus on surfaces, which will be developed more systematically when we study abstract manifolds.

---

## What's Next

We have constructed the first fundamental form and shown that it captures the complete intrinsic geometry of a surface: lengths, angles, areas, and isometries. The key examples — sphere, torus, cylinder, catenoid, helicoid — illustrate how the three coefficients $E$, $F$, $G$ encode the metric structure of surfaces with very different shapes.

But the first fundamental form says nothing about *how* the surface bends in the ambient space. A cylinder and a plane have the same first fundamental form, yet they are clearly different objects in $\mathbb{R}^3$. To detect extrinsic bending, we need additional information: the *second fundamental form*, which measures how the unit normal vector $\mathbf{n}$ changes as we move along the surface. From the second fundamental form, we will extract the principal curvatures, the Gaussian curvature $K$, and the mean curvature $H$. The interplay between the first and second fundamental forms is the heart of classical surface theory.

The deepest result in that theory is Gauss's *Theorema Egregium*: the Gaussian curvature $K$, despite being defined using the second fundamental form (which involves the normal vector and hence the ambient space), turns out to depend only on the first fundamental form and its derivatives. The Gaussian curvature is an *intrinsic* invariant. This surprising fact — that something about extrinsic bending is secretly intrinsic — is the philosophical foundation of Riemannian geometry and general relativity, where curvature is defined without any reference to an ambient space.

---

*This is Part 2 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 1 — Curves in Space](/en/differential-geometry/01-curves-in-space/)*

*Next: [Part 3 — Curvature of Surfaces](/en/differential-geometry/03-second-form-curvature/)*
