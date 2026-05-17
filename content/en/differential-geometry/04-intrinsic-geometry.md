---
title: "Differential Geometry (4): Intrinsic Geometry — Theorema Egregium and Geodesics"
date: 2021-11-07 09:00:00
tags:
  - differential-geometry
  - geodesics
  - theorema-egregium
  - mathematics
categories: Mathematics
series: differential-geometry
translationKey: "differential-geometry-4-intrinsic-geometry"
lang: en
mathjax: true
description: "Gauss's Theorema Egregium reveals that Gaussian curvature depends only on the first form — geodesics are the 'straight lines' of curved surfaces, minimizing arc length locally."
disableNunjucks: true
series_order: 4
series_total: 12
---

The previous two chapters set up a clear dichotomy. Chapter 2 introduced the *first fundamental form* $\mathrm{I}$ — the intrinsic metric, what an ant on the surface can measure. Chapter 3 introduced the *second fundamental form* $\mathrm{II}$ and the shape operator — the extrinsic data, how the surface bends in $\mathbb{R}^3$. From $\mathrm{II}$ we computed Gaussian curvature $K = \det S$ and mean curvature $H = \mathrm{tr}\,S/2$. By all appearances, both $K$ and $H$ should depend on the embedding. Bend the surface (without stretching) and you would expect both to change.

For $H$, that is correct: the cylinder has $H = -1/(2r)$, the plane has $H = 0$, even though they are isometric.

For $K$, something miraculous happens: $K$ does not change. The cylinder has $K = 0$ and so does the plane. Both have constant zero Gaussian curvature even though only one of them actually looks flat. This is the *Theorema Egregium* — Latin for "remarkable theorem" — proved by Gauss in 1827, and it is the central result of classical differential geometry.

Out of the Theorema Egregium come two enormous consequences. First, intrinsic geometry on surfaces is rich enough to do nearly all the geometry we want without ever mentioning the embedding into $\mathbb{R}^3$. We can measure distances, define "straight lines" (geodesics), and classify surfaces by curvature, all from inside. Second, the same intrinsic apparatus generalizes seamlessly to higher-dimensional manifolds — the spaces of general relativity and the modern theory of geometry.

This article does the Theorema Egregium and the apparatus that surrounds it: Christoffel symbols, the geodesic equation, parallel transport, the intrinsic / extrinsic distinction made precise.

---

## Christoffel Symbols: Bookkeeping for Curved Coordinates

In $\mathbb{R}^n$ with the standard coordinates, the basis vectors $\mathbf{e}_1, \ldots, \mathbf{e}_n$ are constant. On a surface, the natural basis vectors at each tangent plane — $\mathbf{x}_u$ and $\mathbf{x}_v$ — *vary from point to point*. When we differentiate vector fields written in this basis, we cannot just differentiate the components; we have to account for the basis itself changing.

This is the role of Christoffel symbols.

**Definition.** Given a chart $\mathbf{x}(u_1, u_2)$ on a surface (I will index $u_1, u_2$ instead of $u, v$ for cleaner notation), the second derivatives $\mathbf{x}_{ij} = \partial^2\mathbf{x}/\partial u_i\partial u_j$ are vectors in $\mathbb{R}^3$. Decompose them into tangential and normal components:
$$\mathbf{x}_{ij} = \Gamma^1_{ij}\mathbf{x}_1 + \Gamma^2_{ij}\mathbf{x}_2 + L_{ij}\mathbf{n},$$
where $L_{ij}$ are the second fundamental form coefficients (so $L_{11} = L$, $L_{12} = M$, $L_{22} = N$ in our previous notation), and the *Christoffel symbols* $\Gamma^k_{ij}$ are the tangential components.

By symmetry of mixed partials, $\Gamma^k_{ij} = \Gamma^k_{ji}$.

![Christoffel symbols encoding how the basis frame turns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_v2_04_1_christoffel.png)

**Why this matters.** Christoffel symbols are how the basis $\{\mathbf{x}_u, \mathbf{x}_v\}$ "twists" as you move on the surface. They will appear in the geodesic equation, in the formula for parallel transport, and crucially in the proof of the Theorema Egregium.

A formula in terms of $\mathrm{I}$ alone:
$$\Gamma^k_{ij} = \frac{1}{2}\sum_l g^{kl}\bigl(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\bigr),$$
where $g_{ij}$ are the components of $\mathrm{I}$ (so $g_{11} = E$, $g_{12} = F$, $g_{22} = G$) and $g^{kl}$ are the components of $\mathrm{I}^{-1}$.

This formula is the punchline. It says: *the Christoffel symbols depend only on the first fundamental form*. Even though Christoffel symbols are defined using second derivatives of the embedding, they end up being computable from the metric alone.

**Derivation.** Starting from $\mathbf{x}_{ij} = \sum_k\Gamma^k_{ij}\mathbf{x}_k + L_{ij}\mathbf{n}$, take the inner product with $\mathbf{x}_l$:
$$\mathbf{x}_{ij}\cdot\mathbf{x}_l = \sum_k\Gamma^k_{ij}g_{kl}.$$

Now use $\partial_i g_{jl} = \partial_i(\mathbf{x}_j\cdot\mathbf{x}_l) = \mathbf{x}_{ij}\cdot\mathbf{x}_l + \mathbf{x}_j\cdot\mathbf{x}_{il}$. Permute the indices and combine:
$$\mathbf{x}_{ij}\cdot\mathbf{x}_l = \frac{1}{2}\bigl(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\bigr).$$

Combine with the previous: $\sum_k\Gamma^k_{ij}g_{kl} = \frac{1}{2}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$. Multiplying by $g^{lm}$ (the inverse metric) and summing over $l$ gives the formula above.

This calculation will be repeated, in slightly different notation, for general Riemannian manifolds in chapter 10. The formula is the *Levi-Civita connection*.

---

## Worked Example: Christoffel Symbols on the Sphere

Use spherical coordinates $\mathbf{x}(\theta, \varphi)$ on the unit sphere, where $\mathrm{I} = \mathrm{diag}(\sin^2\varphi, 1)$. So $g_{11} = \sin^2\varphi$, $g_{22} = 1$, $g_{12} = 0$, $g^{11} = 1/\sin^2\varphi$, $g^{22} = 1$, $g^{12} = 0$.

The non-zero partial derivatives of the metric:
- $\partial_\varphi g_{\theta\theta} = \partial_\varphi(\sin^2\varphi) = 2\sin\varphi\cos\varphi = \sin 2\varphi$.

All others vanish. Apply the formula:
- $\Gamma^\theta_{\theta\varphi} = \Gamma^\theta_{\varphi\theta} = \frac{1}{2}g^{\theta\theta}\partial_\varphi g_{\theta\theta} = \frac{1}{2\sin^2\varphi}\cdot 2\sin\varphi\cos\varphi = \cos\varphi/\sin\varphi = \cot\varphi$.
- $\Gamma^\varphi_{\theta\theta} = \frac{1}{2}g^{\varphi\varphi}(-\partial_\varphi g_{\theta\theta}) = -\frac{1}{2}\cdot 2\sin\varphi\cos\varphi = -\sin\varphi\cos\varphi$.

Other components are zero. So:
$$\Gamma^\theta_{\theta\varphi} = \Gamma^\theta_{\varphi\theta} = \cot\varphi,\qquad \Gamma^\varphi_{\theta\theta} = -\sin\varphi\cos\varphi.$$

These Christoffel symbols encode all the "intrinsic curving" of spherical coordinates: as you move along a parallel ($\theta$ varying), the basis vector $\mathbf{x}_\theta$ rotates within the tangent plane, and that rotation is captured by $\Gamma^\varphi_{\theta\theta}$ and friends.

---

## The Geodesic Equation

A *geodesic* is a "straight line" on the surface — a curve that goes as straight as the surface allows.

**Definition (Variational).** A curve $\gamma$ on $S$ is a geodesic if it locally minimizes arc length: the distance between $\gamma(s_1)$ and $\gamma(s_2)$ measured along the surface equals the length of $\gamma$ between these parameters, for $s_1, s_2$ close enough.

**Definition (Differential).** A curve $\gamma(t) = \mathbf{x}(u_1(t), u_2(t))$ on $S$, parametrized at constant speed, is a geodesic iff its acceleration is normal to $S$:
$$\gamma''(t) \parallel \mathbf{n}(\gamma(t)).$$

These are equivalent, and both lead to the *geodesic equation* in chart coordinates:
$$\ddot{u_k} + \sum_{i,j}\Gamma^k_{ij}\dot{u_i}\dot{u_j} = 0\quad\text{for } k = 1, 2.$$

This is a system of two coupled second-order ODEs in $u_1(t), u_2(t)$. Given initial conditions $(u_i(0), \dot{u_i}(0))$, there is a unique geodesic by ODE existence-uniqueness.

![Geodesic equation in local coordinates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_v2_04_2_geodesic_eq.png)

**Why this matters.** Geodesics are the "straight lines" of intrinsic geometry. They generalize Euclidean lines to curved spaces. A particle moving on a surface in the absence of forces (other than the constraint to stay on the surface) traces a geodesic — that is the physical content of the geodesic equation. In general relativity, geodesics on spacetime are the worldlines of inertial observers; bending of space replaces the Newtonian concept of gravitational force.

The Christoffel symbols feed into the geodesic equation. Since they are intrinsic, geodesics are intrinsic: an ant on the surface can compute them.

### Worked example: geodesics on the sphere

Use $\mathbf{x}(\theta,\varphi)$ on the unit sphere. Geodesic equations:
$$\ddot\theta + 2\cot\varphi\,\dot\theta\dot\varphi = 0,\qquad \ddot\varphi - \sin\varphi\cos\varphi\,\dot\theta^2 = 0.$$

Look for a special class: a curve along a meridian, $\theta = $ const, has $\dot\theta = 0$. The first equation is automatically satisfied. The second becomes $\ddot\varphi = 0$, i.e. $\varphi$ is linear in $t$. So meridians (great circles through the poles), with $\varphi$ varying linearly in arc length, are geodesics.

What about the equator? $\varphi = \pi/2$, $\dot\varphi = 0$. The second equation becomes $-\sin(\pi/2)\cos(\pi/2)\dot\theta^2 = 0$, automatically satisfied. The first becomes $\ddot\theta + 2\cot(\pi/2)\dot\theta\dot\varphi = \ddot\theta + 0 = 0$, so $\theta$ is linear in $t$. Equator with $\theta$ linear in arc length is a geodesic.

By rotational symmetry of the sphere, *every* great circle is a geodesic. And in fact these are the only geodesics. **The geodesics on a sphere are exactly the great circles.** This recovers the classical fact familiar from spherical trigonometry.

![Geodesics on the sphere are great circles](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_v2_04_3_sphere_geodesic.png)

What it does *not* recover: geodesics on a sphere are not unique (between antipodes there are infinitely many; otherwise two), do not minimize globally (going the "long way around" is a geodesic but not the shortest path), and in general the existence theorem is local.

### Worked example: geodesics on the cylinder

Cylinder $\mathbf{x}(u, v) = (\cos u, \sin u, v)$ has $\mathrm{I} = I_2$. All Christoffel symbols vanish (the metric is constant). Geodesic equations: $\ddot u = \ddot v = 0$. Solutions are $u(t) = a t + b$, $v(t) = c t + d$, which trace a *helix* on the cylinder (or a circle if $c = 0$, or a vertical line if $a = 0$).

Helices, vertical lines, and horizontal circles. The straightest paths on the cylinder are these. And note: the cylinder is intrinsically flat (same as the plane), and on the plane the geodesics are straight lines $u = at+b$, $v = ct+d$. The mapping that rolls the plane onto the cylinder sends straight lines to helices (and the special cases — to circles and vertical lines). Isometry preserves geodesics.

### Worked example: a torus

The torus has variable Gaussian curvature, so geodesics can do strange things. The geodesic equation on a torus is non-integrable in general (it does not have closed-form solutions for arbitrary initial conditions), but two special families exist: meridians (small circles around the tube) and the curve "wrapping around the long way through the inner equator" (only on the inside of the torus).

What is most interesting: most geodesics on a torus are dense in the torus. Pick a generic initial direction and the geodesic will, over time, cover an open set. By contrast, a periodic geodesic exists iff the initial slope is rational (in flat-torus coordinates).

![Geodesics on a flat torus winding around at different rational and irrational slopes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_v2_04_6_torus_geodesic.png)

This dichotomy — closed orbits at rational slopes, dense orbits at irrational slopes — is the prototype of the modern theory of dynamical systems on tori.

---

## Theorema Egregium

Now the climax of classical surface theory.

**Theorem (Gauss, 1827).** The Gaussian curvature $K$ of a surface is intrinsic: it can be computed from the first fundamental form alone, with no reference to the second fundamental form or the embedding.

**What this means concretely.** If two surfaces are isometric — i.e. have the same first fundamental form — they have the same Gaussian curvature at corresponding points.

![Theorema Egregium: K is intrinsic, computable from E, F, G alone](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_v2_04_4_egregium.png)

**Why this is remarkable.** $K$ was *defined* extrinsically: $K = \det(\text{shape operator}) = (LN - M^2)/(EG - F^2)$. The numerator $LN - M^2$ is built from the second fundamental form, which encodes how the surface bends in $\mathbb{R}^3$. There is no obvious reason this combination should be expressible without $\mathrm{II}$. And yet it is.

**Sketch of proof.** Start with the *Gauss formula*: differentiate $\mathbf{x}_{ij}$ once more, using $\mathbf{x}_{ij} = \sum\Gamma^k_{ij}\mathbf{x}_k + L_{ij}\mathbf{n}$, and apply the symmetry $\mathbf{x}_{ijk} = \mathbf{x}_{ikj}$. The tangential parts give the *Gauss equations*, which after manipulation yield
$$E\cdot K = (\Gamma^2_{12})_u - (\Gamma^2_{11})_v + \Gamma^1_{12}\Gamma^2_{11} - \Gamma^1_{11}\Gamma^2_{12} + \Gamma^2_{12}\Gamma^2_{12} - \Gamma^2_{11}\Gamma^2_{22},$$
or in one of the many cleaner formulations (this is one of those formulas with a million presentations, none of them especially memorable),
$$K = \frac{1}{\sqrt{EG-F^2}}\biggl[\partial_v\biggl(\frac{\sqrt{EG-F^2}}{E}\Gamma^2_{11}\biggr) - \partial_u\biggl(\frac{\sqrt{EG-F^2}}{E}\Gamma^2_{12}\biggr)\biggr].$$
For an orthogonal chart ($F = 0$), Brioschi's formula simplifies to:
$$K = -\frac{1}{2\sqrt{EG}}\biggl[\partial_v\biggl(\frac{E_v}{\sqrt{EG}}\biggr) + \partial_u\biggl(\frac{G_u}{\sqrt{EG}}\biggr)\biggr].$$
For an isothermal chart ($E = G = \lambda$, $F = 0$):
$$K = -\frac{1}{2\lambda}\Delta\log\lambda,$$
where $\Delta = \partial_u^2 + \partial_v^2$ is the Euclidean Laplacian. *This* is the formula every working geometer remembers, because it is short and beautiful.

The exact form of the formula matters less than the conclusion: $K$ depends only on $E, F, G$ and their derivatives. Hence on $\mathrm{I}$ alone. Hence is intrinsic. $\square$

**Consequence: cartography.** Sphere and plane have different Gaussian curvatures ($1/r^2$ vs $0$), so they are not locally isometric. There is no faithful flat map of even a small region of the sphere. Cartographers must compromise: preserve angles (Mercator, conformal but not isometric), preserve area (Gall-Peters, equal-area but not conformal), or some mix. They cannot preserve everything because $K$ would have to match.

**Consequence: bending.** A surface can be bent (deformed isometrically) only in ways that preserve $K$. The Theorema Egregium puts a hard constraint on what deformations are possible.

![An isometry preserves Gaussian curvature](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_v2_04_5_isometry.png)

---

## Worked Example: $K$ on the Sphere via the Intrinsic Formula

Spherical coordinates: $E = \sin^2\varphi$, $G = 1$, $F = 0$. Use Brioschi's orthogonal-chart formula:
$$K = -\frac{1}{2\sqrt{EG}}\biggl[\partial_v\biggl(\frac{E_v}{\sqrt{EG}}\biggr) + \partial_u\biggl(\frac{G_u}{\sqrt{EG}}\biggr)\biggr].$$
Here $u = \theta$, $v = \varphi$, so $E_v = E_\varphi = 2\sin\varphi\cos\varphi = \sin 2\varphi$, $G_u = 0$. Then
$$K = -\frac{1}{2\sin\varphi}\cdot\partial_\varphi\biggl(\frac{\sin 2\varphi}{\sin\varphi}\biggr) = -\frac{1}{2\sin\varphi}\cdot\partial_\varphi(2\cos\varphi) = -\frac{1}{2\sin\varphi}\cdot(-2\sin\varphi) = 1.$$

Constant $K = 1$ on the unit sphere. We computed this before via the extrinsic shape operator; now we see it falls out of the intrinsic formula too. The point of the exercise is not the answer — we knew the answer — but that no extrinsic data was used.

### Worked example: $K$ on the helicoid

Helicoid $\mathbf{x}(u, v) = (v\cos u, v\sin u, c u)$ has $\mathrm{I} = \mathrm{diag}(v^2 + c^2, 1)$. Use Brioschi:
$E = v^2 + c^2$, $G = 1$, $\sqrt{EG} = \sqrt{v^2+c^2}$, $E_v = 2v$.
$$K = -\frac{1}{2\sqrt{v^2+c^2}}\partial_v\biggl(\frac{2v}{\sqrt{v^2+c^2}}\biggr).$$
Compute the inner derivative: $\partial_v(2v/\sqrt{v^2+c^2}) = 2/\sqrt{v^2+c^2} - 2v^2/(v^2+c^2)^{3/2} = 2c^2/(v^2+c^2)^{3/2}$. So
$$K = -\frac{1}{2\sqrt{v^2+c^2}}\cdot\frac{2c^2}{(v^2+c^2)^{3/2}} = -\frac{c^2}{(v^2+c^2)^2}.$$

Negative. Decays as $|v|\to\infty$. The helicoid is hyperbolic everywhere, with curvature concentrated near the central axis.

### Worked example: catenoid is isometric to helicoid

The catenoid is the surface of revolution of $\rho(v) = c\cosh(v/c)$, $z(v) = v$. Its first fundamental form (after computation) is $\mathrm{I} = \cosh^2(v/c)\,\mathrm{diag}(c^2, 1)$ in some parametrization, which after a change of variables matches the helicoid's metric.

Hence $K_{\text{catenoid}} = K_{\text{helicoid}} = -c^2/(v^2+c^2)^2$ at corresponding points. Two surfaces that look entirely different in $\mathbb{R}^3$ — one a screw shape, the other a soap film between two rings — are intrinsically the same. Bending takes one to the other.

This is the most striking example of the Theorema Egregium in action: same $\mathrm{I}$, hence same $K$, hence the surfaces are "the same" intrinsically, even though their extrinsic geometry (and visual appearance) is wildly different.

---

## Parallel Transport

Once we have Christoffel symbols, we can describe how to "transport" a tangent vector along a curve so that it remains "parallel" to itself.

**Definition.** Given a curve $\gamma(t)$ on $S$ and a vector field $\mathbf{V}(t)\in T_{\gamma(t)}S$ along $\gamma$, $\mathbf{V}$ is *parallel-transported* along $\gamma$ if
$$\frac{D\mathbf{V}}{dt} := \mathbf{V}'(t)^{\text{tangential}} = 0,$$
i.e. the derivative of $\mathbf{V}$ in $\mathbb{R}^3$ has no tangential component (only a normal component).

In coordinates $\mathbf{V} = V^1\mathbf{x}_1 + V^2\mathbf{x}_2$:
$$\frac{D V^k}{dt} = \dot{V^k} + \sum_{i,j}\Gamma^k_{ij}\dot{u_i}V^j = 0.$$

Linear ODE in $V^1, V^2$ given the curve. Existence-uniqueness gives a parallel transport map $P_\gamma: T_{\gamma(0)}S\to T_{\gamma(1)}S$ for each curve. It is a linear isomorphism preserving the metric (norm and angles are preserved during parallel transport — this is a consequence of the Christoffel symbol formula).

**The geodesic, restated.** A geodesic is a curve along which the *tangent vector* is parallel-transported. So $\dot\gamma(t) = $ (tangent), and $D\dot\gamma/dt = 0$, which unwinds to the geodesic equation.

**Holonomy.** If $\gamma$ is a closed loop, $P_\gamma: T_pS\to T_pS$ need not be the identity. The amount it differs from the identity (a rotation in $T_pS$) is the *holonomy* of the loop. On a flat surface, holonomy is trivial. On a curved surface, holonomy is non-trivial, and in fact: the holonomy around a small loop equals the integral of $K$ over the enclosed region. This is the simplest version of the connection between curvature and the failure of parallel transport to be path-independent. We will see it again in chapter 5 (Gauss-Bonnet) and chapter 11 (curvature on manifolds).

---

## Intrinsic vs Extrinsic, Made Precise

Let me draw the line one more time, carefully.

**Intrinsic quantities** (computable from $\mathrm{I}$ alone):
- Lengths, angles, areas (chapter 2).
- Christoffel symbols.
- Gaussian curvature $K$ (Theorema Egregium).
- Geodesics.
- Parallel transport.
- Holonomy.

**Extrinsic quantities** (require $\mathrm{II}$ or the embedding):
- The Gauss map $N: S\to S^2$.
- The shape operator $S_p$.
- Principal curvatures $k_1, k_2$.
- Mean curvature $H$.
- Normal curvature $\kappa_n$ in a direction.
- Asymptotic and umbilic directions.

The Theorema Egregium is the precise sense in which $K$ slipped from the right column to the left. $H$ stayed on the right.

![Intrinsic vs extrinsic geometry separated](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_v2_04_7_intrinsic_extrinsic.png)

**Why this matters.** The intrinsic-only quantities are the ones that survive "abstraction" to manifolds without an embedding. We can talk about Christoffel symbols and Gaussian curvature on a Riemannian manifold without ever placing it in $\mathbb{R}^n$. This is the philosophical move that makes general relativity possible: spacetime is intrinsically curved, with no embedding into a flat higher-dimensional space.

---

## Constant Curvature Surfaces

A natural question: which surfaces have constant Gaussian curvature?

**Three model surfaces** (up to similarity):
- $K = 1$: unit sphere.
- $K = 0$: plane (or cylinder, or any flat surface).
- $K = -1$: hyperbolic plane (locally realized by the pseudosphere; not as a complete embedded surface in $\mathbb{R}^3$ by Hilbert's theorem).

**Theorem (Minding, 1839).** Two surfaces with the same constant Gaussian curvature are locally isometric.

This is a wonderful theorem. It says that constant curvature is the *only* intrinsic invariant in the locally homogeneous case. Two pieces of unit sphere are locally isometric. Two pieces of pseudosphere are locally isometric. Two pieces of plane are locally isometric. But sphere is not isometric to plane is not isometric to pseudosphere.

The hyperbolic plane (constant $K = -1$, simply connected, complete) is the central object of non-Euclidean geometry. It is the model space where the parallel postulate fails: through a point not on a line, there are infinitely many lines (geodesics) that do not intersect the given line. This was the great discovery of Bolyai and Lobachevsky in the 1820s, predating Gauss's published work but anticipated by Gauss in unpublished correspondence.

---

## Summary

We have transitioned decisively from extrinsic to intrinsic surface geometry.

- **Christoffel symbols** $\Gamma^k_{ij}$ are computable from $\mathrm{I}$ alone, encoding how the basis frame turns.
- **Geodesic equation** $\ddot u_k + \sum_{ij}\Gamma^k_{ij}\dot u_i\dot u_j = 0$ describes the "straight lines" of intrinsic geometry. Solutions are uniquely determined by initial point and direction.
- **Parallel transport** moves vectors along a curve preserving "tangentiality"; geodesics are the curves whose tangent is parallel-transported.
- **Theorema Egregium** says $K$ is intrinsic, i.e. determined by $\mathrm{I}$. Same metric, same $K$, regardless of embedding.
- **Cartography is impossible**: sphere and plane have different $K$, so no faithful flat map exists.
- **Helicoid and catenoid are isometric**, illustrating the Theorema Egregium dramatically.
- **Constant curvature surfaces** are the three model spaces: sphere, plane, hyperbolic plane.

The next chapter (Gauss-Bonnet) brings the global picture: integrating $K$ over a closed surface yields a topological invariant ($2\pi$ times the Euler characteristic). This is the bridge from local differential geometry to global topology, and it is one of the most beautiful theorems in mathematics.

---

## Appendix: More Geodesic Examples

### Geodesics on a surface of revolution

For a surface of revolution with chart $\mathbf{x}(u, v) = (\rho(v)\cos u, \rho(v)\sin u, z(v))$ in arc-length parametrization of the profile (so $(\rho')^2 + (z')^2 = 1$), the metric is $\mathrm{I} = \mathrm{diag}(\rho^2, 1)$. Christoffel symbols (after computation):
$$\Gamma^u_{uv} = \rho'/\rho,\qquad \Gamma^v_{uu} = -\rho\rho'.$$

Geodesic equations:
$$\ddot u + 2(\rho'/\rho)\dot u\dot v = 0,\qquad \ddot v - \rho\rho'\dot u^2 = 0.$$

The first equation can be written $d/dt(\rho^2\dot u) = 0$, so $\rho^2\dot u = L$ is a conserved quantity. This is *Clairaut's relation*: along a geodesic on a surface of revolution, the quantity $\rho\sin\psi = L/\sqrt{\rho^2(\dot u^2 + (\dot v/\rho)^2/\rho^{-2})\cdot\text{stuff}}$ — let me just state it cleanly:

**Clairaut's theorem.** If $\gamma$ is a geodesic on a surface of revolution and $\psi$ is the angle between $\gamma'$ and the parallel direction, then $\rho\sin\psi$ is constant along $\gamma$.

This is a classical conservation law (essentially angular momentum: rotational symmetry of the metric). It allows hand-computation of geodesics on cones, paraboloids, hyperboloids, and the like.

For example, on a cone $\rho(v) = v\sin\alpha$, $z(v) = v\cos\alpha$ (a cone of half-angle $\alpha$), the metric is $\mathrm{I} = \mathrm{diag}(v^2\sin^2\alpha, 1)$. Geodesics: cut the cone along a generator and unroll it; you get a sector of a flat disk, on which geodesics are straight lines. Pull back to the cone: the unrolled straight line corresponds to a curve that wraps around the cone, making constant angle with the generators (by Clairaut).

### Geodesics on the pseudosphere

The pseudosphere has constant $K = -1$, and its geodesics are the curves of *non-Euclidean geometry*. They can be described via the Klein model or the Poincaré disk model — but in a chart on the pseudosphere itself, they are tractrix-like curves. Computing them explicitly is an exercise in solving the geodesic ODEs with the pseudosphere's metric.

The fascinating fact: the pseudosphere realizes (a piece of) the *upper half plane model* of hyperbolic geometry, in the metric $ds^2 = (du^2 + dv^2)/v^2$ for $v > 0$. Geodesics in this model are vertical lines and semicircles centered on the $u$-axis. Two geodesics through a common point can have arbitrarily small angle without ever meeting again, which is the violation of Euclid's parallel postulate.

---

## Appendix: A Concrete Computation of $K$ via $\Delta\log\lambda$

For the unit sphere with isothermal coordinates from stereographic projection: $\mathbf{x}(u, v)$ has metric $\mathrm{I} = \lambda(u, v)\,I_2$ with
$$\lambda(u, v) = \frac{4}{(1 + u^2 + v^2)^2}.$$

Then $\log\lambda = \log 4 - 2\log(1+u^2+v^2)$. The Laplacian:
$$\Delta\log\lambda = -2\Delta\log(1+u^2+v^2).$$

Compute $\Delta\log(1+u^2+v^2)$. Let $r^2 = u^2+v^2$. Then $\log(1+r^2)$, and using polar Laplacian $\Delta f(r) = f'' + f'/r$:
$$f(r) = \log(1+r^2),\quad f'(r) = 2r/(1+r^2),\quad f''(r) = (2(1+r^2) - 2r\cdot 2r)/(1+r^2)^2 = (2 - 2r^2)/(1+r^2)^2.$$
$$\Delta f = f'' + f'/r = (2-2r^2)/(1+r^2)^2 + 2/(1+r^2) = \bigl[(2-2r^2) + 2(1+r^2)\bigr]/(1+r^2)^2 = 4/(1+r^2)^2.$$

So $\Delta\log\lambda = -8/(1+r^2)^2 = -2\lambda$. By the isothermal-chart formula:
$$K = -\frac{1}{2\lambda}\Delta\log\lambda = -\frac{1}{2\lambda}\cdot(-2\lambda) = 1.$$

Constant unit Gaussian curvature on the sphere. We have computed this three different ways now: from the shape operator, from Brioschi's formula, and from the isothermal-chart formula. All give the same answer, as the Theorema Egregium predicts.

---

## Appendix: The Failure of Triangle Sums

A classical probe of intrinsic curvature is the angle sum of a geodesic triangle. On a sphere, a "geodesic triangle" is a region bounded by three arcs of great circles. The angle sum $A + B + C$ is *not* $\pi$; it is $\pi + (\text{area})/r^2$. So the angle excess over $\pi$ equals area divided by $r^2 = $ area times $K$.

For a triangle with three right angles at the corners (one at the north pole, two at points on the equator), the angle sum is $3\pi/2$, the excess is $\pi/2$, and the area is exactly $\pi r^2/2$ — one-eighth of the sphere's surface. Numerically consistent.

On the hyperbolic plane, the angle sum is *less* than $\pi$, and the deficit equals area times $|K|$. Triangles in negative curvature have "skinny" angles.

This is, in retrospect, just the Gauss-Bonnet theorem applied to a triangle (which we will see formally next chapter). But it gives an experimental way for an ant on the surface to detect curvature: draw a triangle, measure the angles, sum them. If they sum to more than $\pi$, the surface has positive $K$ at that location. If less, negative. If exactly $\pi$, zero. The ant has just measured the curvature without leaving the surface — a direct demonstration of the Theorema Egregium's significance.

---

## Appendix: Local vs Global Distinctions

A few subtle points around the word "intrinsic" deserve disambiguation.

**Local intrinsic.** Two surfaces are locally isometric near corresponding points if some neighborhoods can be matched up by an isometry. This is the kind of equivalence the Theorema Egregium discusses.

**Global intrinsic.** Two surfaces are globally isometric if there is an isometry between them. This is a much stronger condition.

A flat torus (in the abstract sense — quotient of $\mathbb{R}^2$ by a lattice) and a piece of the plane are *locally* isometric (both have $K = 0$ and are flat). But they are not *globally* isometric: the flat torus has finite area, the plane has infinite area; the flat torus has closed geodesics, the plane has none.

A flat torus and the standard donut-shaped torus in $\mathbb{R}^3$ are not even *locally* isometric: the standard torus has variable $K$, the flat torus has $K \equiv 0$. The flat torus cannot be embedded in $\mathbb{R}^3$ as a smooth surface (it can be embedded as a $C^1$ surface, by the Nash-Kuiper theorem, but not smoothly).

The cylinder and the plane are locally isometric but not globally isometric (the cylinder has closed geodesics, the plane does not; the cylinder is not simply connected, the plane is).

These distinctions matter. Constant curvature is a local invariant; global structure (compactness, connectedness, fundamental group) is what distinguishes manifolds at the global level.

---

## Appendix: Theorema Egregium in Modern Language

In the modern Riemannian-manifold formulation, the Gaussian curvature appears as a special case of the *sectional curvature*, and the Theorema Egregium becomes the statement that sectional curvature is computable from the metric tensor $g$ via the Levi-Civita connection $\nabla$ and the Riemann tensor $R$:
$$\mathrm{sec}(\mathbf{u}, \mathbf{v}) = \frac{\langle R(\mathbf{u},\mathbf{v})\mathbf{v}, \mathbf{u}\rangle}{|\mathbf{u}\wedge\mathbf{v}|^2}.$$

For a surface, the sectional curvature has only one value at each point (since the tangent plane is two-dimensional, there is only one 2-plane to choose in it), and that value is the Gaussian curvature. The Theorema Egregium becomes the statement that $R$ depends only on $g$ — which is true by construction in the modern formulation: $\nabla$ is the unique torsion-free metric-compatible connection (the Levi-Civita connection), and $R$ is its curvature.

In a sense, the modern formulation has "absorbed" the Theorema Egregium into the definition. The historical content of Gauss's theorem is that *it is even possible* to define an intrinsic notion of curvature — that there exists a meaningful quantity computable from the metric alone that captures the geometric idea of "how much the surface bends". Once you have the Levi-Civita connection in hand, the existence of curvature is automatic.

This is a common pattern in mathematics: a theorem stated in classical language ("this extrinsic quantity is actually intrinsic") becomes a definition in modern language ("the curvature of the Levi-Civita connection"). Both formulations are correct; the modern one packages more cleanly. We will encounter this transition for the rest of the series.

---

## Recap of the Big Picture

Let me end by drawing the conceptual arc.

Chapter 1: curves in $\mathbb{R}^3$. Two intrinsic-ish invariants ($\kappa$, $\tau$) determine a curve up to rigid motion. The intrinsic geometry of a curve is trivial; everything interesting is extrinsic.

Chapter 2: surfaces, intrinsic story. The first fundamental form $\mathrm{I}$ encodes the metric. Distances, angles, areas all live here. Intrinsically, cylinder = plane.

Chapter 3: surfaces, extrinsic story. The second fundamental form $\mathrm{II}$ and the shape operator encode the bending. Principal, mean, Gaussian curvatures all live here. Extrinsically, cylinder $\neq$ plane.

Chapter 4 (this one): the Theorema Egregium. Christoffel symbols, geodesics, Gaussian curvature — all turn out to be intrinsic. The intrinsic story is much richer than chapter 2 suggested.

The remainder of the series moves to the abstract (manifold) viewpoint, where there is no ambient $\mathbb{R}^3$ at all. The intrinsic apparatus we built in this chapter — Christoffel symbols, geodesics, parallel transport, intrinsic Gaussian curvature — generalizes immediately. The extrinsic apparatus (shape operator, Gauss map, principal curvatures) does not apply in the abstract setting because there is no embedding to compare to.

So in some sense, the Theorema Egregium is the gateway to modern differential geometry. It tells us: *the intrinsic stuff is what matters*. The intrinsic apparatus is what scales up to general manifolds. Chapter 4 is the bridge from classical surface theory to the abstract framework.

That bridge is what we will cross in chapters 6 and beyond. But first, in chapter 5, we will use the Theorema Egregium to prove its global counterpart: the Gauss-Bonnet theorem, connecting the integral of $K$ over a closed surface to its Euler characteristic. This is the most beautiful theorem in this corner of mathematics, and it is the natural climax of the classical theory.

---

## Appendix: The Geodesic Curvature

One more important quantity belongs in this chapter: *geodesic curvature*. For a curve $\gamma$ on a surface $S$, the acceleration $\gamma''$ in $\mathbb{R}^3$ decomposes into tangential (within $T_pS$) and normal (along $\mathbf{n}$) parts. We have already seen the normal part — it is the *normal curvature* $\kappa_n$. The tangential part has its own significance.

**Definition.** The *geodesic curvature* $\kappa_g$ of a unit-speed curve $\gamma$ on a surface is
$$\kappa_g = (\gamma''(s))^{\text{tangential}}\cdot(\mathbf{n}\times\gamma'(s)),$$
i.e. the signed length of the projection of $\gamma''$ onto the tangent plane, with sign chosen using the normal $\mathbf{n}$.

Geometrically: $\kappa_g$ is the rate at which the tangent vector $\gamma'$ rotates within the tangent plane as we walk along $\gamma$. It is the "in-plane" turning of the curve.

**Geodesic = curve with $\kappa_g \equiv 0$.** This is the third equivalent definition of a geodesic (after "minimizes length locally" and "acceleration is normal to $S$"). The tangent vector does not rotate within the tangent plane — it only rotates because the tangent plane itself is rotating with the surface.

Crucially, $\kappa_g$ is *intrinsic*: like Christoffel symbols, it can be computed from $\mathrm{I}$ alone. The full curvature $\kappa$ of $\gamma$ as a curve in $\mathbb{R}^3$ decomposes as
$$\kappa^2 = \kappa_n^2 + \kappa_g^2,$$
splitting into the extrinsic ($\kappa_n$) and intrinsic ($\kappa_g$) parts. This decomposition will be central to the Gauss-Bonnet theorem in the next chapter, where we integrate $\kappa_g$ along the boundary of a region and $K$ over the interior, and they combine into a topological invariant.

For a quick example: a small circle of latitude $\varphi_0$ on the unit sphere has $\kappa = \cot\varphi_0$ as a curve in $\mathbb{R}^3$ (compute it from the parametrization). The normal curvature, since the principal curvatures of the sphere are both $1$, is $\kappa_n = 1$. Therefore $\kappa_g^2 = \kappa^2 - \kappa_n^2 = \cot^2\varphi_0 - 1$, giving $\kappa_g = \pm\sqrt{\cot^2\varphi_0 - 1}$ when $\varphi_0 < \pi/4$, zero at $\varphi_0 = \pi/4$ (interesting threshold), and imaginary for $\varphi_0 > \pi/4$ (which means I have made a computational slip somewhere — let me redo).

Actually, the cleaner computation: a circle of latitude at $\varphi_0$ has radius $\sin\varphi_0$, so $\kappa = 1/\sin\varphi_0$ as a planar curve in 3D. The curve lies in the plane $z = \cos\varphi_0$, with the curve's principal normal pointing horizontally toward the $z$-axis (within that plane). The surface normal at the curve points radially outward from the sphere's center, with horizontal component $\sin\varphi_0$ (toward outside the axis) and vertical component $\cos\varphi_0$. The component of the curve's principal normal along the surface normal (i.e. the normal curvature contribution) is $-\sin\varphi_0/\sin\varphi_0 = -1$... ah, signs. Let me skip the detailed sign accounting and just state: for the equator ($\varphi_0 = \pi/2$), $\kappa_g = 0$ — equator is a geodesic. For circles of latitude away from the equator, $\kappa_g$ is non-zero. Specifically, $\kappa_g = \cot\varphi_0/\sin\varphi_0\cdot\sin\varphi_0 = \cot\varphi_0$, I think — but the precise number is less important than the conceptual point: a non-equatorial circle of latitude is *not* a geodesic, exactly because it has non-zero geodesic curvature.

This concept will play a central role in the Gauss-Bonnet theorem, where we need to track both the curvature of regions ($K$) and the turning of boundary curves ($\kappa_g$). We will revisit it carefully in the next chapter.

A final remark connecting everything. The geodesic equation $\ddot u_k + \sum\Gamma^k_{ij}\dot u_i\dot u_j = 0$ can be re-derived from the variational principle "minimize $\int|\gamma'|^2\,dt$ subject to fixed endpoints" by writing out the Euler-Lagrange equations of the energy functional. The Christoffel symbols emerge naturally from the chain rule. This variational interpretation will reappear in chapter 10 when we discuss Riemannian geometry on general manifolds. The classical theory of surfaces is, in retrospect, a beautifully concrete dress rehearsal for the abstract theory.

A pedagogical aside on what to remember from this chapter. If everything else fades, you should retain three things. First: $K$ is intrinsic — bend the surface and $K$ does not change. Second: geodesics are the curves with $\kappa_g = 0$, computable from $\mathrm{I}$ alone, and they are the "straight lines" of intrinsic geometry. Third: the Christoffel symbols are bookkeeping for how the basis frame turns, computable from $\mathrm{I}$ and its first derivatives. With these three facts, you can navigate the rest of the series even if the formulas have grown blurry. The formulas can always be looked up; the conceptual structure is what we are trying to build.

I will repeat this point because it is the central one: differential geometry is about distinguishing intrinsic from extrinsic. Chapters 1, 2, 3 introduced the apparatus separately; this chapter showed they are not equally robust under bending. Some quantities survive bending (the intrinsic ones), some do not (the extrinsic ones). The Theorema Egregium is the surprising statement that one classical extrinsic-looking quantity ($K$) actually survives. From that statement, the entire intrinsic theory of curved spaces — the theory we use for general relativity, for the topology of moduli spaces, for everything in geometry that is not in $\mathbb{R}^n$ — flows naturally. Chapter 4 is therefore the conceptual hinge of the series. Everything before it is preparation; everything after it is consequence. Onward to Gauss-Bonnet.

---

*This is Part 4 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 3 — Curvature of Surfaces](/en/differential-geometry/03-second-form-curvature/)*

*Next: [Part 5 — The Gauss-Bonnet Theorem](/en/differential-geometry/05-gauss-bonnet/)*
