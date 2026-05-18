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

For $H$, that is correct: the cylinder has $H = 1/(2r)$, the plane has $H = 0$, even though they are isometric (you can unroll the cylinder flat without stretching).

For $K$, something miraculous happens: $K$ does not change. The cylinder has $K = 0$ and so does the plane. Both have zero Gaussian curvature even though only one of them looks flat from the outside. This is the *Theorema Egregium* — Latin for "remarkable theorem" — proved by Gauss in 1827, and it is the central result of classical differential geometry.

Out of the Theorema Egregium come two enormous consequences. First, intrinsic geometry on surfaces is rich enough to define "straight lines" (geodesics), measure curvature, and classify surfaces — all without mentioning the embedding. Second, the same intrinsic apparatus generalizes seamlessly to higher-dimensional manifolds where no ambient space even exists. The path from the Theorema Egregium to general relativity is conceptually straight.

---

## Christoffel Symbols: How Coordinates Twist on a Curved Surface

In $\mathbb{R}^n$ with the standard coordinates, the basis vectors $\mathbf{e}_1, \ldots, \mathbf{e}_n$ are constant — they do not change from point to point. On a surface parametrized by $\mathbf{x}(u, v)$, the coordinate basis vectors $\mathbf{x}_u$ and $\mathbf{x}_v$ *do* change as we move along the surface. The tangent plane tilts, stretches, and rotates from point to point. When we differentiate a vector field expressed in this basis, we cannot just differentiate the components — we must also account for the basis itself varying. This bookkeeping is the role of Christoffel symbols.

Given a chart $\mathbf{x}(u_1, u_2)$ on a surface, consider the second derivatives $\mathbf{x}_{ij} = \partial^2\mathbf{x}/\partial u_i\partial u_j$. These are vectors in $\mathbb{R}^3$ that decompose into a part tangent to the surface and a part along the normal:
$$\mathbf{x}_{ij} = \Gamma^1_{ij}\mathbf{x}_1 + \Gamma^2_{ij}\mathbf{x}_2 + L_{ij}\mathbf{n},$$
where $L_{ij}$ are the second fundamental form coefficients (extrinsic — measuring the normal component of the acceleration) and the *Christoffel symbols* $\Gamma^k_{ij}$ are the tangential components (intrinsic — measuring how the basis vectors twist within the surface).

![Christoffel symbols encoding how the basis frame turns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_v2_04_1_christoffel.png)

The fundamental formula, derived by taking the inner product of both sides with $\mathbf{x}_l$ and solving:
$$\Gamma^k_{ij} = \frac{1}{2}\sum_l g^{kl}\bigl(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\bigr),$$
where $g_{ij}$ are the metric components ($g_{11} = E$, $g_{12} = F$, $g_{22} = G$) and $g^{kl}$ are the entries of the inverse metric matrix.

This is the punchline of the entire section: *Christoffel symbols depend only on the first fundamental form and its first derivatives*. Despite being defined via the second derivatives of the embedding $\mathbf{x}$, they are computable from the metric coefficients alone. The normal component $L_{ij}$ carries the extrinsic information; the tangential components $\Gamma^k_{ij}$ are purely intrinsic.

The derivation is instructive. Starting from $\mathbf{x}_{ij}\cdot\mathbf{x}_l = \sum_k \Gamma^k_{ij} g_{kl}$ (dot the decomposition with $\mathbf{x}_l$; the $L_{ij}\mathbf{n}$ term vanishes because $\mathbf{n}\perp\mathbf{x}_l$). Now use the product rule: $\partial_i g_{jl} = \partial_i(\mathbf{x}_j\cdot\mathbf{x}_l) = \mathbf{x}_{ij}\cdot\mathbf{x}_l + \mathbf{x}_j\cdot\mathbf{x}_{il}$. By cycling indices and combining three such identities, we isolate $\mathbf{x}_{ij}\cdot\mathbf{x}_l = \frac{1}{2}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$. Multiplying by $g^{lk}$ (the inverse metric) and summing gives the formula above. The second fundamental form never appeared in the final answer — it dropped out when we projected onto the tangent plane.

Physically, imagine walking on a curved surface while carrying a coordinate grid painted on the surface. As you move, the grid lines curve and spread. The Christoffel symbols quantify this grid distortion: $\Gamma^k_{ij}$ tells you how much the $k$-th basis vector changes when you move in the $i$-th direction, holding the $j$-th coordinate constant. On a flat surface in Cartesian coordinates, all $\Gamma^k_{ij} = 0$ — the grid does not distort. On a sphere in polar coordinates, $\Gamma$'s are nonzero because the meridians converge at the poles.

**Worked example: unit sphere in spherical coordinates.** Take $(\theta, \varphi)$ with $g_{11} = g_{\theta\theta} = \sin^2\varphi$, $g_{22} = g_{\varphi\varphi} = 1$, $g_{12} = 0$. The only nonzero metric derivative is $\partial_\varphi g_{\theta\theta} = 2\sin\varphi\cos\varphi = \sin 2\varphi$. Computing the Christoffel symbols:
- $\Gamma^\theta_{\theta\varphi} = \frac{1}{2}g^{\theta\theta}(\partial_\varphi g_{\theta\theta}) = \frac{1}{2}\cdot\frac{1}{\sin^2\varphi}\cdot 2\sin\varphi\cos\varphi = \cot\varphi$.
- $\Gamma^\varphi_{\theta\theta} = -\frac{1}{2}g^{\varphi\varphi}(\partial_\varphi g_{\theta\theta}) = -\frac{1}{2}\cdot 1 \cdot 2\sin\varphi\cos\varphi = -\sin\varphi\cos\varphi$.
- All other $\Gamma$'s vanish (check by plugging in).

The $\cot\varphi$ term diverges as $\varphi \to 0$ (the north pole), reflecting the coordinate singularity there — not a geometric singularity. The sphere is perfectly smooth at the pole; it is the spherical coordinate system that degenerates. A different chart (stereographic projection) would have bounded Christoffel symbols everywhere in its domain.

---

## The Theorema Egregium: Curvature is Intrinsic

Armed with Christoffel symbols, we can state and understand the central theorem of classical differential geometry.

**Theorem (Gauss, 1827 — Theorema Egregium).** The Gaussian curvature $K$ is expressible entirely in terms of the metric coefficients $g_{ij}$ and their first and second partial derivatives. In particular, $K$ does not depend on the second fundamental form $\mathrm{II}$ or on any information about how the surface is embedded in $\mathbb{R}^3$.

In orthogonal coordinates ($F = 0$), there is a relatively clean formula (Brioschi's formula):
$$K = -\frac{1}{2\sqrt{EG}}\left[\frac{\partial}{\partial u}\left(\frac{1}{\sqrt{EG}}\frac{\partial G}{\partial u}\right) + \frac{\partial}{\partial v}\left(\frac{1}{\sqrt{EG}}\frac{\partial E}{\partial v}\right)\right].$$

In isothermal (conformal) coordinates where $\mathrm{I} = \lambda(u,v)(du^2 + dv^2)$, the formula is particularly elegant:
$$K = -\frac{1}{2\lambda}\Delta\log\lambda,$$
where $\Delta = \partial_u^2 + \partial_v^2$ is the flat Laplacian. This makes the intrinsic nature visually obvious: $K$ is determined by $\lambda$ alone, which is the metric.

The proof of the Theorema Egregium proceeds by comparing two expressions for the quantity $\partial_i\Gamma^k_{jl} - \partial_j\Gamma^k_{il} + \text{quadratic terms in }\Gamma$'s. The equality of mixed third partials $\mathbf{x}_{ijk} = \mathbf{x}_{jik}$ (applied to the decomposition into tangential and normal parts) yields two sets of equations: the tangential part gives the *Gauss equation* (relating $K$ to Christoffel symbols), and the normal part gives the *Codazzi-Mainardi equations* (relating derivatives of $L, M, N$ to Christoffel symbols). The Gauss equation is the Theorema Egregium in formula form.

The key insight about *why* the theorem works: the combination $LN - M^2$ that defines $K$ (in the numerator of $K = (LN - M^2)/(EG - F^2)$) is precisely the combination that appears in the compatibility condition for the surface to exist. The compatibility condition is purely about the intrinsic metric (since it is about whether the embedding equations are consistent), so the combination it produces must also be intrinsic. This is not a coincidence — it is the geometry forcing the algebra.

**Consequences that flow immediately:**

1. *Isometries preserve $K$.* If two surfaces are isometric (same first fundamental form), they have the same Gaussian curvature at corresponding points. The cylinder and the plane both have $K = 0$ — consistent with being isometric. The sphere ($K = 1/r^2$) and the plane ($K = 0$) are not isometric — no distance-preserving map between them exists. This is the mathematical reason why every flat map of the Earth distorts.

2. *Bending preserves $K$.* Bending a surface (deforming it without stretching) is an isometry. So bending cannot change $K$. Roll a flat piece of paper into a cylinder: $K$ stays at zero. Try to bend paper into a sphere: impossible without stretching, because the sphere has $K > 0$.

3. *$H$ is genuinely extrinsic.* The plane has $H = 0$; the cylinder has $H = 1/(2r) \neq 0$. They are isometric, but $H$ differs. Mean curvature detects the embedding; Gaussian curvature does not.

Gauss himself was reportedly surprised by this result. The *definition* of $K$ involves the Gauss map and the shape operator — manifestly extrinsic objects — yet the *value* depends only on the intrinsic metric. He called it "egregium" (remarkable), and the name stuck. It is a rare case in mathematics where the discoverer's excitement is preserved in the theorem's official name.

A historical note: Gauss proved this during his work surveying the Kingdom of Hanover (1821-1825). The practical question was: can you make an accurate flat map of a piece of the Earth's surface? The Theorema Egregium gives the definitive answer: no, because the Earth (approximately a sphere with $K > 0$) cannot be flattened (to a surface with $K = 0$) without distorting distances somewhere. Every cartographic projection — Mercator, Lambert, etc. — introduces distortion, and the Theorema Egregium explains why this is unavoidable.

This has quantifiable implications for navigation and mapmaking. The Mercator projection preserves angles but distorts areas (Greenland appears as large as Africa on a Mercator map, though Africa is 14 times larger). The equal-area projections preserve areas but distort angles. No projection preserves both simultaneously — this follows directly from the Theorema Egregium, since preserving both angles and areas would be an isometry, and no isometry from the sphere to the plane exists. Gauss's surveying work (which produced the triangulation of Hanover) was the practical context that led him to these theoretical insights. The most profound theorem of classical differential geometry emerged from the mundane problem of making accurate maps.

A related consequence for geodesy: the shape of the Earth is determined (locally) by the Gaussian curvature of its surface. If you measure $K$ at every point (by measuring angles of triangles, per the Gauss-Bonnet local theorem), you know the intrinsic geometry completely. Two planets with the same curvature distribution have the same intrinsic geometry — even if one is round and the other is wrinkled — because the Theorema Egregium makes the shape operator irrelevant for intrinsic questions.

**Verification on the isothermal formula.** For the unit sphere with stereographic coordinates, $\lambda = 4/(1 + u^2 + v^2)^2$. Then $\log\lambda = \log 4 - 2\log(1 + r^2)$ where $r^2 = u^2 + v^2$. Computing $\Delta\log(1 + r^2)$: in polar form, $\Delta f(r) = f'' + f'/r$. With $f = \log(1+r^2)$: $f' = 2r/(1+r^2)$, $f'' = (2(1+r^2) - 4r^2)/(1+r^2)^2 = (2 - 2r^2)/(1+r^2)^2$. So $\Delta f = (2-2r^2)/(1+r^2)^2 + 2/(1+r^2) = 4/(1+r^2)^2$. Then $\Delta\log\lambda = -8/(1+r^2)^2 = -2\lambda$. By the isothermal formula: $K = -\frac{1}{2\lambda}(-2\lambda) = 1$. Constant unit curvature on the sphere, computed purely from the metric. No shape operator, no normal vector, no embedding information used.

---

## Geodesics: The Straight Lines of Intrinsic Geometry

With Christoffel symbols in hand, we can define the curves that play the role of "straight lines" on a curved surface. In flat space, a straight line has zero acceleration. On a surface, the analog is a curve whose *tangential* acceleration vanishes — it does not steer within the surface, and any acceleration it has is purely normal (forced by the constraint of staying on the surface).

A curve $\gamma(t) = \mathbf{x}(u_1(t), u_2(t))$ on a surface has an acceleration $\gamma''$ that decomposes into tangential and normal parts. The tangential part is the *covariant derivative* of $\gamma'$ along $\gamma$:
$$\frac{D\gamma'}{dt} = \sum_k\left(\ddot u_k + \sum_{i,j}\Gamma^k_{ij}\dot u_i\dot u_j\right)\mathbf{x}_k.$$

A *geodesic* is a curve for which this tangential acceleration is identically zero:
$$\ddot u_k + \sum_{i,j}\Gamma^k_{ij}\dot u_i\dot u_j = 0, \qquad k = 1, 2.$$

This is a system of two second-order ODEs. By the Picard-Lindelof theorem, given any point $p \in S$ and any tangent vector $\mathbf{v} \in T_pS$, there exists a unique geodesic through $p$ with initial velocity $\mathbf{v}$ (at least for short time). Geodesics are determined by initial point and direction, just like straight lines in Euclidean space.

Three equivalent characterizations, each with different intuitive content:
1. **Zero tangential acceleration.** The curve does not "steer" within the surface. All its acceleration comes from the constraint of staying on the surface (normal acceleration). Imagine a ball bearing sliding on a frictionless surface with no gravity — it traces a geodesic.
2. **Locally length-minimizing.** Among all nearby curves connecting two close points, the geodesic is the shortest. (Warning: this is only local — geodesics can fail to be globally shortest, like the "long way around" on a great circle.)
3. **Zero geodesic curvature.** The geodesic curvature $\kappa_g$ — the in-plane turning rate — vanishes identically. The curve goes "straight" in the sense of not turning within the surface, even though it may curve when viewed from $\mathbb{R}^3$.

On the unit sphere, geodesics are great circles — the curves of maximal radius. On a cylinder, geodesics are helices (including the special cases of straight lines along the axis and circles around the cylinder). On the Poincare disk model of the hyperbolic plane, geodesics are circular arcs perpendicular to the boundary circle (plus diameters).

**Worked example: geodesic equations on the sphere.** With $\Gamma^\theta_{\theta\varphi} = \cot\varphi$ and $\Gamma^\varphi_{\theta\theta} = -\sin\varphi\cos\varphi$ (all others zero), the geodesic equations are:
$$\ddot\theta + 2\cot\varphi\,\dot\theta\dot\varphi = 0, \qquad \ddot\varphi - \sin\varphi\cos\varphi\,\dot\theta^2 = 0.$$

The first equation can be written as $\frac{d}{dt}(\sin^2\varphi\,\dot\theta) = 0$, giving the conservation law $\sin^2\varphi\,\dot\theta = c$ (constant). This is *Clairaut's relation*: along a geodesic on a surface of revolution, $\rho\sin\psi = $ const, where $\rho$ is the distance from the axis of revolution and $\psi$ is the angle between the geodesic and the parallel (circle of latitude). It is a conservation of angular momentum, reflecting the rotational symmetry.

Clairaut's relation immediately tells us qualitative facts about geodesics on the sphere: a geodesic that starts heading "eastward" ($\psi$ near $\pi/2$) at a low latitude ($\rho$ large) will maintain the product $\rho\sin\psi$. As it moves toward the pole ($\rho \to 0$), it must have $\sin\psi \to \infty$... which is impossible, so instead the geodesic turns back before reaching the pole. The only geodesics that reach the pole are the meridians ($\psi = 0$, giving $c = 0$). Great circles indeed.

**More worked examples.** On a cylinder of radius $r$ (with metric $ds^2 = r^2\,d\theta^2 + dz^2$), the Christoffel symbols all vanish (the metric coefficients are constant). The geodesic equations reduce to $\ddot\theta = 0$ and $\ddot z = 0$, so geodesics are curves with $\theta(t) = at + b$ and $z(t) = ct + d$ — helices, with pitch depending on the ratio $c/a$. Special cases: $a = 0$ gives vertical lines along the axis; $c = 0$ gives circles around the cylinder. If you unroll the cylinder flat, every helix becomes a straight line — a concrete demonstration that geodesics are "intrinsic straight lines" preserved under isometry.

On a cone with half-angle $\alpha$, parametrized by $\mathbf{x}(u, v) = (v\sin\alpha\cos u, v\sin\alpha\sin u, v\cos\alpha)$, the metric is $ds^2 = v^2\sin^2\alpha\,du^2 + dv^2$. Clairaut's relation gives $v\sin\alpha\sin\psi = $ const. The geodesics can be understood by unrolling the cone into a flat sector of angle $2\pi\sin\alpha$: geodesics on the cone correspond to straight lines in the sector. When you roll the sector back into a cone, the straight lines become curves that generally do not close up — they spiral around the cone, approaching the apex and then receding. Only specific initial conditions produce closed geodesics.

The variational perspective ties geodesics to physics. The geodesic equation is the Euler-Lagrange equation for the energy functional $E(\gamma) = \frac{1}{2}\int_0^1 |\gamma'|^2\,dt = \frac{1}{2}\int_0^1 \sum_{ij} g_{ij}\dot u_i\dot u_j\,dt$. Minimizing energy (with fixed endpoints) yields the same curves as minimizing length, but with the additional property that the parametrization is proportional to arc length. This variational formulation — geodesics as extremals of an action integral — is exactly the principle of least action in mechanics. A free particle on a curved surface moves along geodesics because it minimizes the kinetic energy integral. The connection between geometry (shortest paths) and physics (least action) is deep and recurs throughout modern mathematical physics.

---

## Parallel Transport and the Geometry of Holonomy

Geodesics are intrinsic "straight lines." But there is a subtler intrinsic concept: carrying a vector along a curve without "rotating" it within the surface. This is *parallel transport*.

Given a curve $\gamma(t)$ on $S$ and an initial tangent vector $\mathbf{v}_0 \in T_{\gamma(0)}S$, we seek a vector field $V(t)$ along $\gamma$ that is "constant" in the intrinsic sense — its covariant derivative along $\gamma$ vanishes:
$$\frac{DV}{dt} = \sum_k\left(\dot V^k + \sum_{i,j}\Gamma^k_{ij}\dot u_i V^j\right)\mathbf{x}_k = 0.$$

This is a linear first-order ODE system in $V^k(t)$, so it has a unique solution for any initial $\mathbf{v}_0$. Parallel transport from $\gamma(0)$ to $\gamma(1)$ is a linear isomorphism $P_\gamma: T_{\gamma(0)}S \to T_{\gamma(1)}S$ that preserves inner products (since the Christoffel symbols come from a metric-compatible connection).

A geodesic is precisely a curve whose tangent vector is parallel-transported along itself: $D\gamma'/dt = 0$. A geodesic "goes straight" because it carries its own direction without rotating.

Now here is the phenomenon that makes curved geometry fundamentally different from flat geometry: *parallel transport around a closed loop does not return the vector to its starting direction*. On a flat surface, carrying a vector around any closed path brings it back unchanged. On a curved surface, the vector comes back rotated. The angle of rotation is the *holonomy* of the loop.

The canonical demonstration: parallel-transport a tangent vector around the boundary of the right-angled spherical triangle (one-eighth of the unit sphere). Start at the north pole with a vector pointing along a meridian (say, toward $(1,0,0)$). Walk south along the meridian to the equator: parallel transport along a geodesic preserves the angle with the geodesic, so the vector stays pointing south. At the equator, turn east and walk a quarter of the equator: the vector maintains its angle with the equator (which is also a geodesic), so it stays pointing south. At the point $(0,1,0)$, walk north back to the pole: again parallel transport along a geodesic. The vector arrives at the pole... but now it points toward $(0,1,0)$, which is $90°$ rotated from the initial direction toward $(1,0,0)$.

The vector has been rotated by $\pi/2$ after traversing a loop enclosing area $\pi/2$. On the unit sphere with $K = 1$, the holonomy equals $K \cdot \text{Area} = 1 \cdot \pi/2 = \pi/2$. This is not a coincidence. For any surface, for a small loop enclosing area $\Delta A$ in a region of approximately constant $K$, the holonomy angle is approximately $K \cdot \Delta A$. This is a precise intrinsic characterization of Gaussian curvature: *$K$ is the holonomy per unit area*.

Foucault's pendulum provides a physical realization. A pendulum swinging at latitude $\varphi$ on Earth has its plane of oscillation parallel-transported along the daily rotation path (a circle of latitude). The enclosed spherical cap has area $2\pi(1 - \cos\varphi)$, and with $K = 1/R^2$ for a sphere of radius $R$, the holonomy per day is $2\pi(1 - \cos\varphi) \cdot K \cdot R^2 = 2\pi(1 - \cos\varphi)$... actually, more precisely, the daily rotation of the pendulum plane is $2\pi\sin\varphi$ (as observed by Foucault), which is the solid angle subtended by the circle of latitude. The point is: the precession of the pendulum is a physical manifestation of holonomy, and holonomy is curvature integrated over area.

The holonomy phenomenon has deep consequences for physics beyond Foucault's pendulum. In quantum mechanics, Berry's geometric phase arises when a quantum state is parallel-transported around a loop in parameter space: the state acquires a phase factor determined by the "curvature" of the parameter-space connection. In gauge theory, the Wilson loop — the holonomy of a gauge connection around a closed path — is a fundamental observable that encodes the field strength. Both are direct generalizations of the surface-level phenomenon we just described: curvature manifests as holonomy, the failure of parallel transport to return to its starting value.

The connection to the Theorema Egregium is now complete. We have three equivalent characterizations of Gaussian curvature, all intrinsic:
1. The formula $K = -\frac{1}{2\lambda}\Delta\log\lambda$ in isothermal coordinates (computed from the metric).
2. The holonomy per unit area (measured by parallel transport around small loops).
3. The angle excess per unit area of geodesic triangles (measured by geodesic triangles).

All three give the same number $K$, all three are computable from $\mathrm{I}$ alone, and all three make the Theorema Egregium geometrically obvious once you accept that parallel transport and geodesics are intrinsic operations. The surprising thing is that this same $K$ also equals the *extrinsic* formula $\det(S) = (LN - M^2)/(EG - F^2)$ involving the shape operator. The Theorema Egregium says: these are the same number. The intrinsic characterizations and the extrinsic formula agree — always.

---

## Isometry, Constant Curvature, and the Three Model Geometries

The Theorema Egregium opens a classification program: since isometries preserve $K$, surfaces with different $K$ are never isometric. What are the surfaces of constant curvature?

**Theorem (Minding, 1839).** Any two surfaces with the same constant Gaussian curvature are locally isometric. That is, small patches of one can be mapped isometrically onto small patches of the other.

This means the local geometry of a surface of constant curvature is completely determined by the value of $K$. There are exactly three cases:

**$K > 0$ (spherical geometry).** The model space is the sphere of radius $1/\sqrt{K}$. Geodesics are great circles. Any two geodesics intersect (there are no "parallels"). The angle sum of a geodesic triangle exceeds $\pi$ by an amount equal to the area times $K$. The total area of the sphere is $4\pi/K$. Every point looks the same as every other (homogeneity).

**$K = 0$ (Euclidean geometry).** The model space is the plane. Geodesics are straight lines. The parallel postulate holds. The angle sum of any triangle is exactly $\pi$. Flat, infinite, familiar.

**$K < 0$ (hyperbolic geometry).** The model space is the hyperbolic plane $\mathbb{H}^2$. Geodesics diverge exponentially. Through any point not on a given geodesic, infinitely many geodesics pass that never intersect the given one (violating Euclid's parallel postulate). The angle sum of a geodesic triangle is less than $\pi$ by an amount equal to the area times $|K|$. There is more area at a given "radius" than in Euclidean geometry — hyperbolic space is "roomier" than flat space.

The hyperbolic plane was the great geometric discovery of the 19th century. Bolyai and Lobachevsky (independently, around 1830) realized that negating the parallel postulate leads to a consistent geometry. The Theorema Egregium and Minding's theorem showed this was not just a logical curiosity but a genuine geometric space — the geometry of surfaces with constant negative curvature. The pseudosphere (tractrix of revolution) provides a concrete piece of this geometry embedded in $\mathbb{R}^3$, though Hilbert (1901) proved no complete smooth embedding exists.

The Poincare disk model represents $\mathbb{H}^2$ as the unit disk with the metric $ds^2 = 4(du^2 + dv^2)/(1 - u^2 - v^2)^2$. The isothermal formula gives $K = -1$. Geodesics are circular arcs perpendicular to the boundary circle, and the boundary itself represents "infinity" (infinitely far away in the hyperbolic metric, though visually finite in the Euclidean picture). The stunning images of M.C. Escher's "Circle Limit" series are tilings of the Poincare disk by congruent hyperbolic polygons.

A concrete pair illustrating the Theorema Egregium: the helicoid and catenoid. Both are minimal surfaces with negative Gaussian curvature, and there exists a one-parameter family of isometric deformations between them. You can physically bend one into the other (without stretching) by a continuous motion. Their $K$ functions are identical at corresponding points — as the Theorema Egregium guarantees — even though one is a helicoidal ramp and the other is a neck-shaped surface of revolution. They look completely different extrinsically but are intrinsically the same surface.

A subtler example: two non-isometric surfaces can have the same $K$ function without being isometric. What makes the helicoid-catenoid pair special is not just that they have the same $K$, but that there is a global isometry between them (an explicit map preserving $\mathrm{I}$). Having the same $K$ is necessary for isometry (by the Theorema Egregium) but not sufficient — the full first fundamental form must match, not just its derived quantity $K$.

The three model spaces also admit beautiful explicit descriptions of their geodesics. On the sphere of radius $R$: every geodesic is a great circle of circumference $2\pi R$; any two geodesics intersect (there are no parallels); the geodesic distance between antipodal points is $\pi R$ (the farthest two points can be on a sphere). On the Euclidean plane: geodesics are infinite straight lines; two geodesics either intersect once or are parallel; there is no maximum distance. On the hyperbolic plane of curvature $-1$: geodesics are infinite and diverge exponentially from each other; through a point not on a given geodesic, infinitely many geodesics pass that never intersect the given one. The exponential divergence of geodesics is measured by the rate at which nearby geodesics separate: on a surface of constant curvature $K$, two initially parallel geodesics at distance $d$ apart diverge as $d(t) \sim d(0)\cosh(t\sqrt{|K|})$ when $K < 0$, remain constant when $K = 0$, and reconverge as $d(t) \sim d(0)\cos(t\sqrt{K})$ when $K > 0$. This is the *Jacobi equation* in action, and it explains why negative curvature makes spaces feel "bigger" (geodesics spread apart) while positive curvature makes them feel "smaller" (geodesics reconverge).

---

## Geodesic Curvature and the Bridge to Gauss-Bonnet

One more intrinsic quantity completes the toolkit: *geodesic curvature*. For a unit-speed curve $\gamma$ on a surface, the acceleration $\gamma''$ in $\mathbb{R}^3$ splits into normal and tangential parts. The normal part is $\kappa_n\mathbf{n}$ (normal curvature, extrinsic). The tangential part defines the geodesic curvature:
$$\kappa_g = (\gamma'')^{\mathrm{tan}} \cdot (\mathbf{n} \times \gamma'),$$
measuring how fast the tangent vector $\gamma'$ rotates *within the tangent plane* as we move along $\gamma$.

A geodesic has $\kappa_g \equiv 0$: its tangent does not rotate in-plane. A circle of latitude on a sphere (other than the equator, which is a geodesic) has nonzero $\kappa_g$ — it is constantly turning within the surface to maintain its latitude. The equator has $\kappa_g = 0$ because it is a great circle; other circles of latitude have $\kappa_g = \cot\varphi$ (where $\varphi$ is the colatitude), measuring their deviation from being geodesics.

The total curvature of $\gamma$ as a space curve satisfies $\kappa^2 = \kappa_n^2 + \kappa_g^2$, partitioning into extrinsic and intrinsic parts. Crucially, $\kappa_g$ is computable from $\mathrm{I}$ alone — it is intrinsic. This makes it available on abstract manifolds where there is no ambient space.

Why emphasize $\kappa_g$? Because it is the boundary ingredient in the Gauss-Bonnet theorem (next chapter):
$$\iint_T K\,dA + \int_{\partial T}\kappa_g\,ds + \sum_i\theta_i = 2\pi.$$

This formula connects three things: the curvature of the region's interior ($K$), the turning of the smooth boundary ($\kappa_g$), and the angular jumps at corners ($\theta_i$). All three are intrinsic. They must conspire to give $2\pi$ — the angle of one full rotation. When $K = 0$ and there are no corners, the boundary must turn through exactly $2\pi$: this is the flat-space theorem that a simple closed curve has winding number $\pm 1$. Nonzero $K$ "uses up" some of this turning budget internally, so the boundary can turn less (on a positively curved surface) or must turn more (on a negatively curved surface) to compensate.

A worked example brings this to life. Take a geodesic disk of radius $r$ on the unit sphere (the set of all points within geodesic distance $r$ of the north pole — this is a spherical cap bounded by a circle of latitude). The boundary is a circle of latitude at colatitude $\varphi = r$. Its geodesic curvature is $\kappa_g = \cot r$ (pointing inward). The integrated geodesic curvature is $\int\kappa_g\,ds = \cot r \cdot 2\pi\sin r = 2\pi\cos r$. The area of the cap is $2\pi(1 - \cos r)$, so $\iint K\,dA = 2\pi(1-\cos r)$ (with $K = 1$). Check: $2\pi(1-\cos r) + 2\pi\cos r = 2\pi$. The local Gauss-Bonnet formula is satisfied. As $r \to 0$, the curvature term shrinks and the boundary turning approaches $2\pi$ (a small circle turns almost all the way around, as in flat space). As $r \to \pi/2$ (a hemisphere), the curvature term gives $2\pi(1 - 0) = 2\pi$ and the boundary turning is $2\pi\cos(\pi/2) = 0$ — the equator is a geodesic ($\kappa_g = 0$), and all the "turning" comes from the interior curvature. As $r \to \pi$ (the whole sphere minus a point), the curvature gives nearly $4\pi$ and the boundary shrinks to a point, contributing negligible turning.

This is the local form of the Gauss-Bonnet theorem, and it encapsulates the entire relationship between local curvature and global turning. The global form — integrating over a closed surface — will be the climax of the classical theory.

---

## What's next

The Gauss-Bonnet theorem takes the local formula above and applies it globally: integrating $K$ over an entire closed surface yields $2\pi\chi(S)$, where $\chi$ is the Euler characteristic — a topological invariant. It is the most beautiful theorem in classical differential geometry, connecting analysis to topology.

---

*This is Part 4 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 3 — Curvature of Surfaces](/en/differential-geometry/03-second-form-curvature/)*

*Next: [Part 5 — The Gauss-Bonnet Theorem](/en/differential-geometry/05-gauss-bonnet/)*
