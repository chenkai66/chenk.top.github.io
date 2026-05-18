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

Curves were a one-dimensional warm-up. The geometry was governed by ODEs, and a single moving frame caught everything interesting. From now on we go up a dimension and the difficulty rises in three directions at once. Tangent vectors get replaced by *tangent planes*. The single arc-length parameter splits into two coordinates $(u, v)$, and reparametrization becomes a $2\times 2$ Jacobian matrix instead of a scalar. And — the real change — we acquire two distinct kinds of geometry: *intrinsic* (what an ant living on the surface can measure) and *extrinsic* (how the surface bends in the surrounding $\mathbb{R}^3$). This article is the intrinsic story. We build the *first fundamental form*, the $2\times 2$ matrix-valued function that lets the ant measure lengths, angles, and areas without ever leaving the surface.

The intrinsic / extrinsic split is going to recur for the next ten chapters, so it is worth pinning down here. A surface is a 2D thing sitting inside a 3D ambient space. Some of its geometry depends on how it sits in $\mathbb{R}^3$ (its bending and twisting); some of it depends only on the surface itself (distances and angles between nearby points on the surface). The first fundamental form captures *exactly the latter*. The next chapter introduces the second fundamental form, which captures the former. Gauss's *Theorema Egregium*, two chapters later, will deliver the punchline: there is one extrinsic-looking quantity (Gaussian curvature) that is in fact intrinsic. But that is for chapter 4. Today we lay the foundation.

![A regular surface patch with its parameterization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_v2_02_1_surface_patch.png)

---

## What is a Surface?


![Tangent plane on a saddle surface with tangent vectors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/02_parametric_surfaces.png)

![Parametric surfaces: sphere, torus, and saddle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/02_parametric_surfaces.png)


The naive definition — "a 2D thing in 3D space" — is fine for intuition but useless for proofs. A precise definition has to say what "smooth" means, how to give the surface coordinates, and how to handle places where coordinates fail.

**Definition (Regular surface, classical).** A subset $S\subseteq\mathbb{R}^3$ is a *regular surface* if for every point $p\in S$ there is an open neighborhood $V\subseteq\mathbb{R}^3$ of $p$, an open set $U\subseteq\mathbb{R}^2$, and a smooth map $\mathbf{x}: U\to V\cap S$ satisfying:
1. $\mathbf{x}$ is a homeomorphism (continuous, with continuous inverse);
2. $\mathbf{x}$ is smooth (each component is $C^\infty$);
3. for every $q\in U$, the differential $d\mathbf{x}_q: \mathbb{R}^2 \to \mathbb{R}^3$ is injective.

The map $\mathbf{x}$ is called a *coordinate chart*, *parametrization*, or *patch*. The third condition — injective differential — is the analog of regularity for curves. Concretely, it says the two partial derivatives $\mathbf{x}_u = \partial\mathbf{x}/\partial u$ and $\mathbf{x}_v = \partial\mathbf{x}/\partial v$ are linearly independent. They span a 2D plane at every point.

**Why this matters.** Patch-by-patch, a surface looks like a smoothly deformed piece of $\mathbb{R}^2$. Calculus on the surface is just calculus on $\mathbb{R}^2$ with an extra layer of bookkeeping (the chart) on top. The $\mathbb{R}^2$ side is where the integrals and partial derivatives live; the surface is where the geometry lives.

### Common examples

**Graph of a function.** $S = \{(u, v, f(u,v)) : (u,v)\in\mathbb{R}^2\}$ where $f$ is smooth. The chart $\mathbf{x}(u,v) = (u, v, f(u,v))$ trivially satisfies all three conditions. *Every* surface is locally a graph (after rotating coordinates), but the global picture often requires multiple charts.

**Sphere of radius $r$.** $S^2_r = \{(x,y,z) : x^2+y^2+z^2 = r^2\}$. No single chart covers it (a topological obstruction we will address in chapter 6). A common patch: spherical coordinates $\mathbf{x}(\theta,\varphi) = (r\sin\varphi\cos\theta, r\sin\varphi\sin\theta, r\cos\varphi)$ for $(\theta,\varphi)\in (0, 2\pi)\times(0,\pi)$. This covers everything except a meridian; you need a second patch (rotated) to cover the rest.

**Cylinder.** $\mathbf{x}(u,v) = (\cos u, \sin u, v)$ for $(u,v)\in (0,2\pi)\times\mathbb{R}$. Covers everything except a single line.

**Torus (donut).** $\mathbf{x}(u,v) = ((R+r\cos v)\cos u, (R+r\cos v)\sin u, r\sin v)$ for $(u,v)\in (0,2\pi)\times(0,2\pi)$. Covers most of the torus; you again need another patch.

The technical sufficiency of these parametrizations as proper "charts" is something I will check explicitly for one of them, the sphere, later in the article. The point right now is to have concrete things in your head.

---

## The Tangent Plane

Given a chart $\mathbf{x}: U\to S$ around a point $p = \mathbf{x}(q)$, the partial derivatives $\mathbf{x}_u(q), \mathbf{x}_v(q)\in\mathbb{R}^3$ are linearly independent (by regularity) and span a 2D subspace of $\mathbb{R}^3$. This subspace is the *tangent plane* $T_pS$.

![Coordinate curves on a torus](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/02_coordinate_curves.png)


**Definition.** The *tangent plane* at $p\in S$ is
$$T_pS = \mathrm{span}\{\mathbf{x}_u(q), \mathbf{x}_v(q)\} \subseteq \mathbb{R}^3.$$

A few things to verify (which I will do in passing): $T_pS$ depends only on $p$, not on the choice of chart. If $\tilde{\mathbf{x}}: \tilde U\to S$ is another chart around $p$, the change of coordinates $\phi = \mathbf{x}^{-1}\circ\tilde{\mathbf{x}}: \tilde U\to U$ is a diffeomorphism between open subsets of $\mathbb{R}^2$, and the chain rule gives $\tilde{\mathbf{x}}_u = \phi_u^1 \mathbf{x}_u + \phi_u^2 \mathbf{x}_v$, etc. The two pairs $\{\mathbf{x}_u, \mathbf{x}_v\}$ and $\{\tilde{\mathbf{x}}_u, \tilde{\mathbf{x}}_v\}$ are related by an invertible linear map (the Jacobian of $\phi$), so they span the same plane.

Geometrically, $T_pS$ is the "best linear approximation" to $S$ at $p$. If you zoom in on the surface near $p$, it looks more and more like its tangent plane.

![Tangent plane and unit normal vector to a surface at a point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_v2_02_2_tangent_plane.png)

The orthogonal complement of $T_pS$ in $\mathbb{R}^3$ is one-dimensional, and is spanned by the *unit normal*
$$\mathbf{n}(p) = \frac{\mathbf{x}_u\times\mathbf{x}_v}{|\mathbf{x}_u\times\mathbf{x}_v|}.$$
The choice of sign (i.e. which of $\pm\mathbf{n}$ to pick) is an orientation. For a connected orientable surface (like a sphere), there are exactly two choices: outward and inward. For a non-orientable surface (like the Möbius strip or the Klein bottle), no global continuous choice exists — a fact we will use later when discussing topology.

### Worked example: tangent plane on the sphere

Let $\mathbf{x}(\theta,\varphi) = (\sin\varphi\cos\theta, \sin\varphi\sin\theta, \cos\varphi)$ on the unit sphere. Then
- $\mathbf{x}_\theta = (-\sin\varphi\sin\theta, \sin\varphi\cos\theta, 0)$,
- $\mathbf{x}_\varphi = (\cos\varphi\cos\theta, \cos\varphi\sin\theta, -\sin\varphi)$.

At the equator $\varphi = \pi/2$, $\theta = 0$: $\mathbf{x} = (1, 0, 0)$, $\mathbf{x}_\theta = (0, 1, 0)$, $\mathbf{x}_\varphi = (0, 0, -1)$. So $T_pS = $ span of $(0,1,0)$ and $(0,0,-1)$, which is the $yz$-plane. The point $p = (1,0,0)$ is the "east" point of the sphere, and the tangent plane there is exactly the $yz$-plane through that point. The unit normal is $\mathbf{x}_\theta\times\mathbf{x}_\varphi/|\cdot| = (-1, 0, 0)$, the inward radial direction (or $(1,0,0)$ if we choose the other orientation). Reassuring: the normal at a point on a sphere is the radial direction. We did not need any of this machinery to know that, but it is good to confirm the formulas are not lying.

---

## The First Fundamental Form

Now we get to the heart of the matter. Given a chart $\mathbf{x}(u,v)$, define three functions on $U$:
$$E = \mathbf{x}_u\cdot\mathbf{x}_u,\qquad F = \mathbf{x}_u\cdot\mathbf{x}_v,\qquad G = \mathbf{x}_v\cdot\mathbf{x}_v.$$

![First fundamental form: measuring area on a surface](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/02_first_form_area.png)


These are the *coefficients of the first fundamental form*. Equivalently, they assemble into a $2\times 2$ matrix:
$$\mathrm{I} = \begin{pmatrix} E & F \\ F & G \end{pmatrix},$$
which is the Gram matrix of the basis $\{\mathbf{x}_u, \mathbf{x}_v\}$ of $T_pS$. Symmetric, and positive-definite (since $\mathbf{x}_u, \mathbf{x}_v$ are linearly independent), with determinant $EG - F^2 > 0$.

![Coefficients E, F, G of the first fundamental form](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_v2_02_3_first_form.png)

The first fundamental form lets us compute the inner product of any two tangent vectors. If $\mathbf{w}_1, \mathbf{w}_2\in T_pS$ are written in the basis $\{\mathbf{x}_u, \mathbf{x}_v\}$ as $\mathbf{w}_i = a_i\mathbf{x}_u + b_i\mathbf{x}_v$, then
$$\mathbf{w}_1\cdot\mathbf{w}_2 = a_1 a_2 E + (a_1 b_2 + a_2 b_1) F + b_1 b_2 G = \begin{pmatrix}a_1 & b_1\end{pmatrix}\mathrm{I}\begin{pmatrix}a_2\\ b_2\end{pmatrix}.$$

In other words, we use $\mathrm{I}$ as a metric on tangent vectors expressed in chart coordinates. The operations of "length", "angle", and "area" all derive from this.

**Why this matters.** The first fundamental form is the *intrinsic metric* of the surface: it is the tool an ant living on the surface uses to measure things. Crucially, two different surfaces (sitting differently in $\mathbb{R}^3$) can have the *same* first fundamental form — meaning the ant cannot tell them apart. We will see this for the cylinder and the plane, which are both flat in the intrinsic sense even though they look very different from outside.

### Length of a curve on the surface

Suppose $\gamma(t) = \mathbf{x}(u(t), v(t))$ is a curve on the surface for $t\in[a,b]$. Then $\gamma'(t) = u'\mathbf{x}_u + v'\mathbf{x}_v$, and
$$|\gamma'(t)|^2 = E (u')^2 + 2 F u' v' + G (v')^2.$$
The length of $\gamma$ is
$$L(\gamma) = \int_a^b\sqrt{E(u')^2 + 2 F u'v' + G(v')^2}\,dt.$$

Notice: this formula uses only $E, F, G$ and the curve's $(u(t), v(t))$. It does not reference the embedding into $\mathbb{R}^3$ in any other way. The ant on the surface, given the metric and the curve, can compute lengths exactly as we do.

![Arc length of a curve on a surface using the first fundamental form](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_v2_02_4_arc_length_surf.png)

### Angle between curves

If two curves $\gamma_1, \gamma_2$ pass through $p$ with tangent vectors $\mathbf{w}_1, \mathbf{w}_2$, the angle between them is
$$\cos\theta = \frac{\mathbf{w}_1\cdot\mathbf{w}_2}{|\mathbf{w}_1||\mathbf{w}_2|},$$
and the inner product is computed using $\mathrm{I}$. Again: only $E, F, G$ and the chart-coordinates of the tangent vectors are needed. Angles are intrinsic.

A useful corollary: a chart is *orthogonal* (the coordinate curves $u = $ const and $v = $ const meet at right angles) if and only if $F\equiv 0$. Spherical coordinates, cylindrical coordinates, and the standard torus parametrization are all orthogonal. Some advanced parametrizations are not.

### Area

The area element on the surface is
$$dS = |\mathbf{x}_u\times\mathbf{x}_v|\,du\,dv = \sqrt{EG - F^2}\,du\,dv.$$

The Lagrange identity $|\mathbf{a}\times\mathbf{b}|^2 = |\mathbf{a}|^2|\mathbf{b}|^2 - (\mathbf{a}\cdot\mathbf{b})^2$ confirms that $|\mathbf{x}_u\times\mathbf{x}_v|^2 = EG - F^2$. So the area of a region $R = \mathbf{x}(D)$ is
$$\mathrm{Area}(R) = \iint_D \sqrt{EG - F^2}\,du\,dv.$$

![Area element dS = sqrt(EG - F^2) du dv](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_v2_02_5_area_element.png)

Once again, intrinsic: the surface ant computes areas using only $E, F, G$.

---

## Worked Examples: Computing $E, F, G$

I will now grind through three standard surfaces and write down the first fundamental form. Doing this once carefully cements the apparatus.

![Metric distortion under different parametrizations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/02_metric_distortion.png)


### The plane

Trivial chart: $\mathbf{x}(u,v) = (u, v, 0)$. Then $\mathbf{x}_u = (1,0,0)$, $\mathbf{x}_v = (0,1,0)$, so
$$E = 1,\quad F = 0,\quad G = 1,\qquad \mathrm{I} = \begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}.$$
The identity matrix. The plane has the Euclidean metric, as expected.

### The cylinder

$\mathbf{x}(u,v) = (\cos u, \sin u, v)$. Then $\mathbf{x}_u = (-\sin u, \cos u, 0)$, $\mathbf{x}_v = (0, 0, 1)$.
$$E = \sin^2 u + \cos^2 u = 1,\quad F = 0,\quad G = 1,\qquad \mathrm{I} = I_2.$$

**The cylinder has the same first fundamental form as the plane.** This is an absolutely critical observation. As far as the intrinsic metric is concerned, the cylinder and the plane are indistinguishable. An ant on the cylinder, equipped only with the metric $\mathrm{I}$, cannot tell whether it is living on flat paper or rolled-up paper. This is the precise sense in which the cylinder is *intrinsically flat*. Of course, externally it bends in $\mathbb{R}^3$ — the second fundamental form (next chapter) will detect that. The first fundamental form does not.

There is a name for this kind of equivalence: two surfaces are *isometric* if they have the same first fundamental form (in some choice of charts). Plane and cylinder are isometric. Plane and sphere are not (as we will see). Plane and saddle are not. Etc.

### The unit sphere

$\mathbf{x}(\theta,\varphi) = (\sin\varphi\cos\theta, \sin\varphi\sin\theta, \cos\varphi)$.
- $\mathbf{x}_\theta = (-\sin\varphi\sin\theta, \sin\varphi\cos\theta, 0)$, so $E = \sin^2\varphi$.
- $\mathbf{x}_\varphi = (\cos\varphi\cos\theta, \cos\varphi\sin\theta, -\sin\varphi)$, so $G = \cos^2\varphi+\sin^2\varphi = 1$.
- $\mathbf{x}_\theta\cdot\mathbf{x}_\varphi = 0$, so $F = 0$.

$$\mathrm{I} = \begin{pmatrix}\sin^2\varphi & 0\\ 0 & 1\end{pmatrix}.$$

This is *not* the identity matrix; it depends on $\varphi$. So the sphere is not isometric to the plane. (This is the rigorous version of the cartographer's frustration: you cannot make a perfectly faithful flat map of the Earth.)

Let us use this to compute something concrete. Equator length: the equator is $\varphi = \pi/2$, $\theta\in[0, 2\pi]$. So $u'(t) = $ doesn't quite apply directly; identify $u = \theta$, $v = \varphi$. Curve: $u = t$, $v = \pi/2$ for $t\in[0, 2\pi]$.
$$L = \int_0^{2\pi}\sqrt{\sin^2(\pi/2)\cdot 1 + 0 + 1\cdot 0}\,dt = \int_0^{2\pi}1\,dt = 2\pi.$$
Equator has length $2\pi$, as expected.

A small circle at latitude $\varphi_0$: $u = t$, $v = \varphi_0$. Length $= \int_0^{2\pi}\sin\varphi_0\,dt = 2\pi\sin\varphi_0$. Smaller circles farther from the equator. Again, exactly what you would expect, but now derived purely from $\mathrm{I}$.

Surface area: $dS = \sqrt{EG - F^2}\,d\theta\,d\varphi = \sin\varphi\,d\theta\,d\varphi$. Total area:
$$\mathrm{Area} = \int_0^{2\pi}\int_0^\pi \sin\varphi\,d\varphi\,d\theta = 2\pi\cdot 2 = 4\pi.$$

The classical $4\pi r^2$ formula (with $r = 1$). Worth flagging: this came out of integrating $|\mathbf{x}_\theta\times\mathbf{x}_\varphi|$, which depends on the chart, but the answer is intrinsic. If we change to a different parametrization the integrand changes but the integral is the same.

### The torus

$\mathbf{x}(u,v) = ((R+r\cos v)\cos u, (R+r\cos v)\sin u, r\sin v)$ for $u, v\in[0, 2\pi)$.
- $\mathbf{x}_u = (-(R+r\cos v)\sin u, (R+r\cos v)\cos u, 0)$, so $E = (R+r\cos v)^2$.
- $\mathbf{x}_v = (-r\sin v\cos u, -r\sin v\sin u, r\cos v)$, so $G = r^2$.
- $\mathbf{x}_u\cdot\mathbf{x}_v = 0$, so $F = 0$.

$$\mathrm{I} = \begin{pmatrix}(R+r\cos v)^2 & 0\\ 0 & r^2\end{pmatrix}.$$

Surface area: $\sqrt{EG-F^2} = r(R+r\cos v)$, so
$$\mathrm{Area} = \int_0^{2\pi}\int_0^{2\pi} r(R+r\cos v)\,du\,dv = 2\pi r\cdot 2\pi R = 4\pi^2 R r.$$

A pleasing closed form. Equator (the outer rim, $v = 0$) has length $2\pi(R+r)$; the inner rim ($v = \pi$) has length $2\pi(R-r)$. The "tube circumference" (a curve of fixed $u$, varying $v$) has length $2\pi r$ — the small circle.

![Sphere, cylinder, and torus shown side by side](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_v2_02_7_classical_surfaces.png)

---

## Change of Coordinates

Suppose I have two charts $\mathbf{x}: U\to S$ and $\tilde{\mathbf{x}}: \tilde U\to S$ overlapping at a point $p$. The transition map $\phi = \mathbf{x}^{-1}\circ\tilde{\mathbf{x}}$ is a diffeomorphism between open subsets of $\mathbb{R}^2$. Write $\phi(u', v') = (u(u', v'), v(u', v'))$. The Jacobian
$$J = \begin{pmatrix}u_{u'} & u_{v'}\\ v_{u'} & v_{v'}\end{pmatrix}$$
relates the two bases of $T_pS$.

![Measuring curve length on a surface](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/02_curve_on_surface.png)


The first fundamental forms transform tensorially: $\tilde{\mathrm{I}} = J^T \mathrm{I} J$. Determinants: $\det\tilde{\mathrm{I}} = (\det J)^2 \det\mathrm{I}$, so $\sqrt{\tilde E\tilde G - \tilde F^2} = |\det J|\sqrt{EG - F^2}$, which is exactly the change-of-variables formula for the area integral. The two charts agree on lengths, angles, and areas.

This transformation rule is the prototype for what later becomes "tensor transformation laws" in general relativity. The first fundamental form is a $(0,2)$-tensor; its components in any two charts are related by the Jacobian of the transition map.

**Why this matters.** Once we are confident the formulas transform correctly under change of chart, we can take a more abstract view: there is a single object — the metric — which is realized as $E, F, G$ in any chart, and the choice of chart is just a coordinate convenience. This is the seed of the tensor calculus we will need for chapter 6 onward.

---

## Isometries and Isometric Surfaces

**Definition.** A diffeomorphism $f: S_1 \to S_2$ between two surfaces is an *isometry* if it preserves the first fundamental form. Concretely: for every $p\in S_1$ and every $\mathbf{w}_1, \mathbf{w}_2\in T_pS_1$,
$$\mathbf{w}_1\cdot\mathbf{w}_2 = (df_p\mathbf{w}_1)\cdot(df_p\mathbf{w}_2).$$

![Isometric surfaces: same metric, different shape](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/02_isometry.png)


Equivalently: in matching charts $\mathbf{x}_1$ on $S_1$ and $\mathbf{x}_2 = f\circ\mathbf{x}_1$ on $S_2$, $E_1 = E_2$, $F_1 = F_2$, $G_1 = G_2$.

**Cylinder and plane.** The map $f(u, v) = (\cos u, \sin u, v)$ from a strip $(0, 2\pi)\times\mathbb{R}\subset \mathbb{R}^2$ to the cylinder is an isometry. Cut the cylinder open along a vertical line and roll it flat: the result is a rectangle. The metric is preserved, even though the embedding into $\mathbb{R}^3$ changes drastically.

**Cone and plane.** Similarly, a cone (sliced open) flattens to a sector of a disk. Cones are also intrinsically flat away from the apex.

**Sphere and plane.** No isometry exists. We will eventually prove this via Gauss's Theorema Egregium: the sphere has positive Gaussian curvature, the plane has zero, and Gaussian curvature is intrinsic. Cartography is doomed.

There is one more flavour of map worth naming.

**Definition.** A map $f$ is *conformal* if it preserves angles (but not necessarily lengths). Equivalently, $f^*\mathrm{I}_2 = \lambda(p)\,\mathrm{I}_1$ for some positive smooth function $\lambda$.

The Mercator projection of the sphere is conformal (angles are preserved — useful for navigation), but not isometric (lengths are distorted, as anyone who has wondered why Greenland looks comically large on a Mercator map knows). The stereographic projection is also conformal.

### Isothermal coordinates

A chart is *isothermal* if it is conformal as a map from flat $\mathbb{R}^2$ to the surface — equivalently, if $E = G$ and $F = 0$ identically:
$$\mathrm{I} = \lambda(u,v)\begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix} = \lambda(u,v)\,I_2.$$

A famous theorem (Korn-Lichtenstein, 1914-16) says every smooth surface admits isothermal coordinates locally. This is non-trivial and uses elliptic PDE theory. The theorem is the gateway from differential geometry to complex analysis: a surface with isothermal coordinates inherits a complex structure, and the theory of Riemann surfaces takes off from there.

![Isothermal coordinates: a conformal parameterization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/02-surfaces-first-form/dg_v2_02_6_isothermal.png)

For the sphere, stereographic projection from the north pole gives isothermal coordinates: in the chart $(u, v)\mapsto$ point on sphere via stereographic inverse, the metric is $4(1+u^2+v^2)^{-2}\,I_2$. Conformal factor $\lambda = 4/(1+u^2+v^2)^2$. Lots of complex analysis on $\mathbb{C}\cup\{\infty\}$ secretly happens on the sphere with this metric.

---

## A Computational Aside: Computing Lengths in the Wild

A pragmatic example. Suppose I want the length of the curve $\theta = t$, $\varphi = t$ on the unit sphere for $t\in[0, \pi/4]$. I will not bother with the embedding — I will use only $\mathrm{I}$.

$E = \sin^2\varphi$, $F = 0$, $G = 1$, with $u = \theta$, $v = \varphi$, so $u' = v' = 1$.

$$L = \int_0^{\pi/4}\sqrt{\sin^2 t + 1}\,dt.$$

This integral is not elementary — it is a complete elliptic integral of the second kind in disguise. Numerically, with $\sin^2 t \in [0, 0.5]$ and $\sqrt{1+\sin^2 t}\in[1, \sqrt{1.5}\approx 1.2247]$, the integrand is between 1 and 1.225, so the answer is between $\pi/4 \approx 0.785$ and $\pi/4 \cdot 1.225 \approx 0.962$. A more careful computation gives $L \approx 0.870$. The point is not the precise number; it is that the computation involved no $\mathbb{R}^3$, just an integrand built from $E$ and $G$.

This is the concrete sense in which "the metric is enough". The ant on the sphere can compute the length of its diagonal walk; the ambient space never enters.

---

## Limits and Non-Examples

A few cautionary notes.

**The chart needs to be a homeomorphism.** A common student mistake: writing $\mathbf{x}(\theta,\varphi)$ for the sphere on $[0,2\pi]\times[0,\pi]$ (closed intervals). Then the map identifies the two endpoints of $\theta$ and is not injective; it is not a homeomorphism. We need open intervals, and we accept that no single chart covers a sphere. The patch-by-patch picture is mandatory.

**Self-intersections.** A parametrized surface $\mathbf{x}: U\to\mathbb{R}^3$ can have a self-intersecting image while still satisfying the regularity condition (linear independence of partials). The image is then an *immersed* surface, but not a *regular* surface in our sense. Regular surface = embedded.

**The first fundamental form is positive-definite.** This is a consequence of $\mathbf{x}_u, \mathbf{x}_v$ being linearly independent. It is not automatic for arbitrary symmetric matrices: $\begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix}$ would have $EG - F^2 = 0$, which violates regularity. Whenever a textbook surface is degenerating ($EG - F^2\to 0$), something is wrong with the chart.

**Pseudo-Riemannian metrics.** In general relativity, the metric on spacetime is *not* positive-definite — it has signature $(-,+,+,+)$. The first fundamental form is positive-definite ("Riemannian"); the spacetime metric is "Lorentzian". The formulas look similar but the geometry is qualitatively different (e.g. there are non-zero "null vectors" with $\mathbf{w}\cdot\mathbf{w} = 0$). We will stay in the positive-definite world for this series.

**Higher dimensions.** Everything we did generalizes: a $k$-dimensional submanifold of $\mathbb{R}^n$ has a first fundamental form (now a $k\times k$ matrix), tangent spaces, isometries, etc. Only the bookkeeping grows. The intrinsic / extrinsic split persists.

---

## What's Next

We have now built the intrinsic story. The first fundamental form is a $2\times 2$ symmetric positive-definite matrix-valued function of the surface coordinates, encoding the metric of the surface. From it we can compute:

- lengths of curves (integrating $\sqrt{E(u')^2 + 2F u'v' + G(v')^2}$);
- angles between curves (using $\mathrm{I}$ as an inner product);
- areas of regions ($\sqrt{EG - F^2}\,du\,dv$);
- isometry equivalence (same $E, F, G$ in matching charts).

What we cannot yet compute is *bending* — how the surface curves in $\mathbb{R}^3$. That requires the *second fundamental form*, a different $2\times 2$ matrix that measures the second-derivative behaviour of $\mathbf{x}$ in the normal direction. The shape operator, principal curvatures, Gaussian curvature, and mean curvature all live in that world.

The next chapter introduces the Gauss map (the map from a surface to the unit sphere sending each point to its unit normal), and from its differential extracts the shape operator. From the shape operator we will read off principal curvatures (the eigenvalues), and from those two numbers we will define Gaussian and mean curvatures. This is the *extrinsic* geometry of surfaces.

After that, in chapter 4, comes the climax of classical surface theory: Gauss's Theorema Egregium, which states that the Gaussian curvature — although defined extrinsically via the second fundamental form — can in fact be computed from the first fundamental form alone. The metric knows the Gaussian curvature. This is the bridge between the intrinsic and extrinsic stories, and it is the conceptual launchpad for the rest of differential geometry.

For now, you should be comfortable computing $E, F, G$ for any explicit parametrization, and using them to compute lengths, angles, and areas. That is the toolkit we will draw on for the next several chapters.

---

## Appendix: Three More Worked Examples

To give the reader more numerical practice before we close, here are three more first-fundamental-form computations, varying in flavour.

### Surface of revolution

Let a profile curve $(\rho(v), z(v))$ be revolved around the $z$-axis. The surface has chart
$$\mathbf{x}(u, v) = (\rho(v)\cos u, \rho(v)\sin u, z(v)),\qquad u\in[0, 2\pi),\ v\in I.$$

Compute:
- $\mathbf{x}_u = (-\rho\sin u, \rho\cos u, 0)$, so $E = \rho^2$.
- $\mathbf{x}_v = (\rho'\cos u, \rho'\sin u, z')$, so $G = (\rho')^2 + (z')^2$.
- $\mathbf{x}_u\cdot\mathbf{x}_v = -\rho\rho'\sin u\cos u + \rho\rho'\cos u\sin u + 0 = 0$, so $F = 0$.

$$\mathrm{I} = \begin{pmatrix}\rho(v)^2 & 0\\ 0 & (\rho')^2 + (z')^2\end{pmatrix}.$$

This is *always* an orthogonal chart ($F = 0$): the meridians ($u = $ const) and parallels ($v = $ const) intersect at right angles. Most "natural" parametrizations of nice surfaces have this property.

If we want the profile curve in arc-length parametrization, $(\rho')^2 + (z')^2 = 1$, and the metric simplifies to
$$\mathrm{I} = \begin{pmatrix}\rho(v)^2 & 0\\ 0 & 1\end{pmatrix}.$$

This is the "warped product" form. Specializing: sphere has $\rho(v) = \sin v$, $z(v) = \cos v$, which (you can check) is arc-length parametrized in $v$. Cylinder has $\rho(v) = $ const, $z(v) = v$. Cone, paraboloid, hyperboloid of one sheet — all surfaces of revolution. The first fundamental form for each is just plug-and-chug.

**Numerical specifics for a paraboloid.** $\rho(v) = v$, $z(v) = v^2/2$ for $v > 0$. Then $\rho' = 1$, $z' = v$, $G = 1 + v^2$. So
$$\mathrm{I} = \begin{pmatrix}v^2 & 0\\ 0 & 1 + v^2\end{pmatrix}.$$
At $v = 1$: $\sqrt{EG - F^2} = \sqrt{v^2(1+v^2)} = v\sqrt{1+v^2} = \sqrt{2}$. Area element at radius $v = 1$ in the chart is $\sqrt{2}\,du\,dv$, slightly more than the planar value $du\,dv$ — because the paraboloid is tilted up at $45^\circ$ there.

### Helicoid

The helicoid is the surface swept out by a horizontal line rotating uniformly about the $z$-axis while translating uniformly along it: $\mathbf{x}(u, v) = (v\cos u, v\sin u, c\, u)$ for $(u, v)\in\mathbb{R}^2$, with $c > 0$ a fixed constant.

- $\mathbf{x}_u = (-v\sin u, v\cos u, c)$, so $E = v^2 + c^2$.
- $\mathbf{x}_v = (\cos u, \sin u, 0)$, so $G = 1$.
- $\mathbf{x}_u\cdot\mathbf{x}_v = -v\sin u\cos u + v\cos u\sin u + 0 = 0$, so $F = 0$.

$$\mathrm{I} = \begin{pmatrix}v^2 + c^2 & 0\\ 0 & 1\end{pmatrix}.$$

Famously (we will prove this in chapter 4), the helicoid is *locally isometric* to the *catenoid* — a surface of revolution generated by the catenary $\rho(v) = c\cosh(v/c)$. Their first fundamental forms differ by a coordinate change. This is a striking example: the helicoid (a screw shape) and the catenoid (a soap film between two rings) look completely different externally, yet are intrinsically the same surface. An ant equipped with only the metric cannot tell whether it is on a helicoid or a catenoid; it can only tell that something near where it is standing has a certain metric.

### Graph of a function

For a graph $\mathbf{x}(u, v) = (u, v, f(u,v))$:
- $\mathbf{x}_u = (1, 0, f_u)$, $E = 1 + f_u^2$.
- $\mathbf{x}_v = (0, 1, f_v)$, $G = 1 + f_v^2$.
- $\mathbf{x}_u\cdot\mathbf{x}_v = f_u f_v$, so $F = f_u f_v$.

$$\mathrm{I} = \begin{pmatrix}1 + f_u^2 & f_u f_v\\ f_u f_v & 1 + f_v^2\end{pmatrix}.$$

$EG - F^2 = (1+f_u^2)(1+f_v^2) - f_u^2 f_v^2 = 1 + f_u^2 + f_v^2$. Area element: $\sqrt{1 + f_u^2 + f_v^2}\,du\,dv$. The familiar formula for surface area of a graph; in calculus class you derived it from a Riemann-sum argument, now it falls out of the first fundamental form.

For $f(u, v) = u^2 + v^2$ (paraboloid as a graph): $f_u = 2u$, $f_v = 2v$, $\sqrt{EG-F^2} = \sqrt{1+4u^2+4v^2}$. Area inside a disk of radius $R$: $\int_0^{2\pi}\int_0^R r\sqrt{1+4r^2}\,dr\,d\theta = 2\pi\cdot\frac{1}{12}((1+4R^2)^{3/2} - 1)$.

For $R = 1$: area $= \pi((5)^{3/2}-1)/6 \approx \pi(11.18 - 1)/6 \approx 5.33$. Compare with the disk area $\pi R^2 = \pi \approx 3.14$. The paraboloid graph has more area than its "shadow" because of the upward tilt — by a factor of about $1.7$. Reasonable.

---

## Appendix: A Lattice of Special Surfaces

Some surfaces deserve names because their first fundamental form has special structure. Knowing these names is part of the trade.

**Flat surface.** A surface with $K = 0$ (Gaussian curvature, defined later). Equivalently, isometric to a piece of the plane. Examples: plane, cylinder, cone (away from apex), tangent developable of any space curve. Flat surfaces are the only ones you can roll out of paper without stretching.

**Surface of revolution.** Generated by revolving a profile curve. Always orthogonal coordinates (meridian / parallel). Includes sphere, paraboloid, hyperboloid, cone, cylinder, torus.

**Ruled surface.** Made up of straight lines. Cylinder, cone, hyperboloid of one sheet, helicoid, Möbius strip, tangent developable. Some ruled surfaces are flat; some are not.

**Minimal surface.** A surface with mean curvature $H = 0$ (defined next chapter). Locally a soap film. Examples: catenoid, helicoid, Scherk's surface, Costa's surface (a celebrated 1980s discovery). Minimal surfaces are the area-minimizers among small perturbations.

**Constant-curvature surface.** $K \equiv $ const. Three flavours by sign: sphere ($K = 1$), plane ($K = 0$), pseudosphere or Beltrami trumpet ($K = -1$, hyperbolic). The hyperbolic case cannot be embedded as a complete surface in $\mathbb{R}^3$ (Hilbert's theorem, 1901), but it can be embedded locally — and the abstract intrinsic geometry of constant negative curvature is the model for non-Euclidean geometry.

**Conformally flat surface.** Admits an isothermal chart, $\mathrm{I} = \lambda\,I_2$. *Every* surface is locally conformally flat (Korn-Lichtenstein). Globally, conformal flatness is a different question.

**Locally Euclidean.** Synonymous with flat. A subtle distinction will appear later: some flat surfaces (like a flat torus) are not isometric to a piece of the plane *globally*, even though they are locally.

These six adjectives — flat, revolution, ruled, minimal, constant-curvature, conformally flat — give a vocabulary for talking about surfaces. We will use most of them.

---

## Appendix: Why "First" Fundamental Form?

The numbering implies a "second" is coming. The naming is historical: Gauss, in his foundational 1827 paper *Disquisitiones generales circa superficies curvas*, organized the data of a surface into two quadratic forms. The first measures intrinsic distances (our $\mathrm{I}$, also written $ds^2$ in older notation). The second measures the bending into ambient space (our $\mathrm{II}$, coming next chapter). Together they determine the surface up to rigid motion — a "fundamental theorem of surfaces" analogous to the fundamental theorem of curves.

Not every $2\times 2$ symmetric pair $(\mathrm{I}, \mathrm{II})$ corresponds to a real surface, however. There are integrability conditions (the *Gauss equation* and the *Codazzi equations*) that any genuine pair must satisfy. Conversely, given a pair satisfying these conditions, there is a unique surface (up to rigid motion) realizing them. We will get there, but first we need the second fundamental form, which is the topic of the very next chapter.

A historical aside on terminology. Gauss did not actually use the matrices $\mathrm{I}$ and $\mathrm{II}$ as such; he wrote out the differentials $E\,du^2 + 2F\,du\,dv + G\,dv^2$ and $L\,du^2 + 2M\,du\,dv + N\,dv^2$ explicitly. The matrix viewpoint came later — partly with Riemann's introduction of higher-dimensional metric tensors in his 1854 *Habilitation* lecture, partly with the subsequent rise of tensor calculus (Ricci-Curbastro and Levi-Civita, late 19th century). By the time Einstein needed differential geometry for general relativity in 1915, the matrix / tensor language was the lingua franca. We use that language because it scales cleanly to higher dimensions; just remember when reading older texts that $E, F, G$ and $L, M, N$ were the original notation.

A computational rule of thumb. When facing an unfamiliar surface, the best path is almost always:
1. Write a chart $\mathbf{x}(u,v)$ explicitly.
2. Compute partials $\mathbf{x}_u, \mathbf{x}_v$.
3. Read off $E, F, G$ from inner products.
4. Read off the unit normal from the cross product.
5. Compute everything else (length, area, angle, second form, curvature) from these inputs.

The discipline of always going through this five-step process is what separates working differential geometers from the rest. The formulas in textbooks for "Gaussian curvature in arbitrary coordinates" or "geodesic equations" can look forbidding, but in any specific case they reduce to plugging in $E, F, G$ and turning a crank. We will do plenty of crank-turning in the chapters ahead.

A final remark on the conceptual structure. The first fundamental form is the differential-geometric reflection of one specific intuition: the *measurement* of distances and angles within a 2D world. Everything else in this article — isometries, isothermal coordinates, change of charts, the metric on classical surfaces — is consequence and elaboration. If you keep that one image in mind (the metric is just the inner product, restricted to tangent planes, expressed in chart coordinates), the rest of this chapter is bookkeeping. The bookkeeping happens to be necessary, because differential geometry is a coordinate-rich subject and you cannot escape it. But the underlying idea is simple, and Gauss was the first to see it clearly. With this groundwork, we are ready to ask the next question: how does a surface bend in space? That requires a different set of formulas, and they are coming up next.

One more numerical sanity check before we close. For the unit sphere, the metric is $\mathrm{I} = \mathrm{diag}(\sin^2\varphi, 1)$, with $\sqrt{EG-F^2} = \sin\varphi$. The total surface area we computed was $4\pi$. Now consider an arbitrary spherical cap of polar angle $\varphi_0$ — the region where $\varphi\in[0,\varphi_0]$. Its area is
$$\int_0^{2\pi}\int_0^{\varphi_0}\sin\varphi\,d\varphi\,d\theta = 2\pi(1 - \cos\varphi_0).$$
For $\varphi_0 = \pi/2$ (a hemisphere), this gives $2\pi$, exactly half of $4\pi$. For $\varphi_0 = \pi/3$ (a "polar ice cap" of $60^\circ$), it gives $2\pi(1 - 0.5) = \pi$, one quarter of the total area. For $\varphi_0 = \pi$, the full sphere, $2\pi(1 - (-1)) = 4\pi$. Internally consistent.

What I love about this calculation is that it never refers to the embedding into $\mathbb{R}^3$ — only to the metric. An ant on the sphere, walking along meridians and integrating, computes the same area. This is the central message of the chapter: *intrinsic geometry is enough for measurement*. The next chapter will show what intrinsic geometry is *not* enough for: detecting how the surface bends in space.

A short closing example to fix one more time the difference between intrinsic and extrinsic. Consider the cylinder again, with metric $\mathrm{I} = I_2$. From inside, this is exactly the plane: parallel meridians and parallel parallels, distances and angles like Euclid's. From outside, it is round: the meridian is a straight vertical line, but the parallel is a circle of radius $1$. The two perspectives give different geometries. The intrinsic one is captured by $\mathrm{I}$; the extrinsic one is captured by quantities we have not yet defined. When we introduce the second fundamental form next chapter, we will be able to say precisely how the cylinder differs from the plane: same $\mathrm{I}$, different $\mathrm{II}$. The plane has zero second form; the cylinder has a non-zero one. And so we will detect the bending without disturbing the metric. That is the whole conceptual content of the next chapter, and the apparatus we are about to build will make it precise.

Take a breath; that is a lot of formulas for one chapter. The good news: chapter 3 builds on this one but does not replace it. The first fundamental form is permanent; we will use it in every subsequent article. The bad news: chapters 3 and 4 will compose the second fundamental form on top of $\mathrm{I}$, and there is more bookkeeping coming. The trick to surviving differential geometry is to keep the conceptual picture clear (intrinsic = $\mathrm{I}$, extrinsic = $\mathrm{II}$, glued by Gauss-Codazzi) while doing the algebra patiently. Onward.

---

*This is Part 2 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 1 — Curves in Space](/en/differential-geometry/01-curves-in-space/)*

*Next: [Part 3 — Curvature of Surfaces](/en/differential-geometry/03-second-form-curvature/)*
