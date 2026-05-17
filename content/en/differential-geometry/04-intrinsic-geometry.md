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
lang: en
mathjax: true
description: "Gauss's Theorema Egregium reveals that Gaussian curvature depends only on the first form — geodesics are the 'straight lines' of curved surfaces, minimizing arc length locally."
disableNunjucks: true
series_order: 4
series_total: 12
translationKey: "differential-geometry-4"
---

Imagine you are a two-dimensional creature living on a surface, equipped with a ruler and a protractor but no access to the ambient three-dimensional space. What geometric properties can you measure? You can measure distances between points, angles between curves, and areas of regions — everything encoded in the first fundamental form. But can you detect curvature?

The answer, miraculously, is yes. Gauss proved that the Gaussian curvature $K = \kappa_1\kappa_2$ — defined extrinsically as the product of principal curvatures — is in fact completely determined by the first fundamental form $\mathrm{I} = E\,du^2 + 2F\,du\,dv + G\,dv^2$ and its partial derivatives. This is his **Theorema Egregium** ("remarkable theorem"), and it stands as one of the most beautiful results in all of geometry.

This article develops the intrinsic viewpoint: Christoffel symbols, the Theorema Egregium, the compatibility equations of Gauss and Codazzi-Mainardi, and the theory of geodesics.

---

## Intrinsic vs. Extrinsic: What a Surface Dweller Can Measure

To make the distinction precise, we call a property of a surface **intrinsic** if it depends only on the first fundamental form (the metric), and **extrinsic** if it requires knowledge of how the surface sits in $\mathbb{R}^3$.

**Intrinsic quantities:**

![Geodesics on the sphere are great circles](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/04-intrinsic-geometry/dg_fig4_geodesics.png)

- Arc length of curves: $L = \int_a^b \sqrt{E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2}\,dt$
- Angles between curves (via the inner product from $\mathrm{I}$)
- Area of regions: $A = \iint \sqrt{EG - F^2}\,du\,dv$
- Gaussian curvature $K$ (Theorema Egregium)
- Geodesics and geodesic curvature

**Extrinsic quantities:**
- The unit normal $N$
- The second fundamental form coefficients $e, f, g$
- Principal curvatures $\kappa_1, \kappa_2$ individually
- Mean curvature $H$

The intrinsic/extrinsic divide is not just philosophical: it determines which properties are preserved under **isometries** (distance-preserving maps). A cylinder and a plane have different shapes in $\mathbb{R}^3$, but they are locally isometric — you can unroll one onto the other without stretching. Every intrinsic property (including $K = 0$) is preserved; extrinsic properties (like $\kappa_1 = 1/R$ for the cylinder vs. $\kappa_1 = 0$ for the plane) are not.

---

## Christoffel Symbols from the First Fundamental Form

To express intrinsic geometry computationally, we need the **Christoffel symbols** — the correction terms that appear when we differentiate vector fields on a curved surface.

Consider a parametrized surface $\mathbf{r}(u,v)$. The second derivatives $\mathbf{r}_{uu}, \mathbf{r}_{uv}, \mathbf{r}_{vv}$ can be decomposed into tangential and normal components:

$$\mathbf{r}_{uu} = \Gamma^1_{11}\,\mathbf{r}_u + \Gamma^2_{11}\,\mathbf{r}_v + e\,N,$$
$$\mathbf{r}_{uv} = \Gamma^1_{12}\,\mathbf{r}_u + \Gamma^2_{12}\,\mathbf{r}_v + f\,N,$$
$$\mathbf{r}_{vv} = \Gamma^1_{22}\,\mathbf{r}_u + \Gamma^2_{22}\,\mathbf{r}_v + g\,N.$$

The normal components give the second fundamental form coefficients $e, f, g$. The tangential components define the **Christoffel symbols of the second kind** $\Gamma^k_{ij}$.

To find them, take the inner product of both sides with $\mathbf{r}_u$ and $\mathbf{r}_v$. For example, from the first equation:

$$\langle \mathbf{r}_{uu}, \mathbf{r}_u \rangle = \Gamma^1_{11} E + \Gamma^2_{11} F, \qquad \langle \mathbf{r}_{uu}, \mathbf{r}_v \rangle = \Gamma^1_{11} F + \Gamma^2_{11} G.$$

The left-hand sides are computable from the metric: $\langle \mathbf{r}_{uu}, \mathbf{r}_u \rangle = \frac{1}{2}E_u$ and $\langle \mathbf{r}_{uu}, \mathbf{r}_v \rangle = F_u - \frac{1}{2}E_v$. Solving the $2 \times 2$ linear system:

$$\begin{pmatrix} E & F \\ F & G \end{pmatrix} \begin{pmatrix} \Gamma^1_{11} \\ \Gamma^2_{11} \end{pmatrix} = \begin{pmatrix} \frac{1}{2}E_u \\ F_u - \frac{1}{2}E_v \end{pmatrix}.$$

Similarly for $\Gamma^k_{12}$ and $\Gamma^k_{22}$. In each case, the Christoffel symbols are determined entirely by $E, F, G$ and their first partial derivatives — they are intrinsic.

**In orthogonal coordinates** ($F = 0$), the formulas simplify considerably:

$$\Gamma^1_{11} = \frac{E_u}{2E}, \quad \Gamma^2_{11} = -\frac{E_v}{2G}, \quad \Gamma^1_{12} = \frac{E_v}{2E}, \quad \Gamma^2_{12} = \frac{G_u}{2G}, \quad \Gamma^1_{22} = -\frac{G_u}{2E}, \quad \Gamma^2_{22} = \frac{G_v}{2G}.$$

These expressions will be essential for computing geodesics and verifying the Theorema Egregium.

**Example: sphere in spherical coordinates.** For $S^2(R)$ with $E = R^2$, $F = 0$, $G = R^2\sin^2\theta$:

$$\Gamma^1_{11} = 0, \quad \Gamma^2_{11} = 0, \quad \Gamma^1_{12} = 0, \quad \Gamma^2_{12} = \frac{\cos\theta}{\sin\theta}, \quad \Gamma^1_{22} = -\sin\theta\cos\theta, \quad \Gamma^2_{22} = 0.$$

The only nonzero Christoffel symbols are $\Gamma^2_{12} = \cot\theta$ and $\Gamma^1_{22} = -\sin\theta\cos\theta$. These encode the "curving" of the spherical coordinate grid — the meridians spread apart near the equator and converge at the poles, which is captured by $\Gamma^1_{22}$, while $\Gamma^2_{12}$ reflects the angular acceleration needed to follow a parallel circle at non-constant latitude.

**Physical interpretation.** The Christoffel symbols encode what Newtonian mechanics would call "fictitious forces." If a particle moves freely (no external forces) on a surface, its trajectory satisfies the geodesic equations — which involve the Christoffel symbols. On a sphere, a freely moving particle follows a great circle, but in spherical coordinates, the equations of motion involve $\Gamma$ terms that look like "forces." These are the Coriolis and centrifugal terms familiar from rotating reference frames.

---

## Gauss's Theorema Egregium

**Theorem (Gauss, 1827).** The Gaussian curvature $K$ of a regular surface is an intrinsic invariant: it depends only on the coefficients $E, F, G$ of the first fundamental form and their first and second partial derivatives.

### Statement in coordinates

For a surface with $F = 0$ (orthogonal parametrization), the formula takes a particularly elegant form:

$$K = -\frac{1}{2\sqrt{EG}}\left[\frac{\partial}{\partial u}\left(\frac{G_u}{\sqrt{EG}}\right) + \frac{\partial}{\partial v}\left(\frac{E_v}{\sqrt{EG}}\right)\right].$$

This is sometimes called **Gauss's formula** for curvature. For the general case ($F \neq 0$), the expression is more involved but still depends only on $E, F, G$ and their derivatives up to second order.

An equivalent and more compact formulation uses the Christoffel symbols:

$$K = \frac{1}{E}\left(\frac{\partial \Gamma^2_{11}}{\partial v} - \frac{\partial \Gamma^2_{12}}{\partial u} + \Gamma^1_{11}\Gamma^2_{12} - \Gamma^1_{12}\Gamma^2_{11} + \Gamma^2_{11}\Gamma^2_{22} - (\Gamma^2_{12})^2\right).$$

### Proof sketch

The idea is to compare two ways of computing $K$:

1. **Extrinsic definition:** $K = (eg - f^2)/(EG - F^2)$, using the second fundamental form.

2. **Gauss equation:** The compatibility condition for the system $\mathbf{r}_{uu}, \mathbf{r}_{uv}, \mathbf{r}_{vv}$ (requiring $(\mathbf{r}_{uu})_v = (\mathbf{r}_{uv})_u$, i.e., equality of mixed partials) yields an identity relating the Christoffel symbols and their derivatives to the second fundamental form coefficients.

More concretely, we use the Gauss equations (the tangential components of the compatibility condition $\mathbf{r}_{uuv} = \mathbf{r}_{uvu}$). Writing out the tangential part:

$$\frac{\partial \Gamma^k_{11}}{\partial v} - \frac{\partial \Gamma^k_{12}}{\partial u} + \sum_l (\Gamma^l_{11}\Gamma^k_{lv} - \Gamma^l_{12}\Gamma^k_{lu}) = e g_{k \text{-component}} - f f_{k\text{-component}},$$

where the right side involves $e, f, g$ and the metric. But the crucial observation is that the right side equals $K$ times a metric factor. Since the left side involves only Christoffel symbols (hence only the metric), we conclude that $K$ is determined by the metric alone.

A cleaner modern proof uses the **Riemann curvature tensor** $R^l_{ijk}$, defined entirely in terms of Christoffel symbols:

$$R^l_{ijk} = \frac{\partial \Gamma^l_{ik}}{\partial x^j} - \frac{\partial \Gamma^l_{ij}}{\partial x^k} + \Gamma^m_{ik}\Gamma^l_{mj} - \Gamma^m_{ij}\Gamma^l_{mk}.$$

For a 2-dimensional surface, the Riemann tensor has essentially one independent component, and the Gauss equation states

$$R^1_{212} = EK \quad (\text{when } F = 0).$$

Since $R^1_{212}$ is built from Christoffel symbols, $K$ is intrinsic. $\square$

### Significance

The Theorema Egregium has profound consequences:

1. **Isometric invariance of $K$.** If two surfaces are isometric (related by a distance-preserving map), they have the same Gaussian curvature at corresponding points. A sphere ($K = 1/R^2 > 0$) cannot be isometric to any portion of the plane ($K = 0$). This is why every flat map of the Earth must distort distances.

2. **Cylinder $\cong$ plane locally.** A cylinder has $K = 0$, same as a plane, confirming that you can unroll a cylinder without stretching — a fact we know intuitively from rolling paper into a tube.

3. **Rigidity.** While mean curvature $H$ changes under bending (the cylinder and plane have different $H$), Gaussian curvature is preserved. This makes $K$ the fundamental bending invariant.

4. **Cartography.** Every method of projecting the Earth's surface onto a flat map introduces distortion. The Mercator projection preserves angles (it is conformal) but grossly distorts areas near the poles. The equal-area projections preserve area but distort angles. No projection can preserve both simultaneously, because that would require an isometry from a surface with $K = 1/R^2$ to one with $K = 0$ — which the Theorema Egregium forbids.

5. **Gaussian curvature of the hyperbolic plane.** The Poincare disk model of the hyperbolic plane has metric $ds^2 = 4(dx^2+dy^2)/(1-x^2-y^2)^2$. Computing $K$ from this metric using the Gauss formula yields $K = -1$ everywhere. The Theorema Egregium tells us this is a genuine geometric property, not an artifact of the particular model or embedding. The half-plane model $ds^2 = (dx^2+dy^2)/y^2$ also gives $K = -1$, confirming that the two models are isometric (and must be, since they represent the same abstract geometry).

**Historical context.** Gauss proved the Theorema Egregium in his *Disquisitiones generales circa superficies curvas* (1827). He was so struck by the result that he called it "egregium" — remarkable. The proof occupied much of his private research notebooks, and the published version was highly condensed. The result was a major impetus for the development of Riemannian geometry by Riemann in 1854, who generalized the concept of intrinsic curvature to arbitrary dimensions and paved the way for Einstein's general theory of relativity.

---

## The Gauss and Codazzi-Mainardi Equations

The Gauss equation (proved above) is one of two **compatibility conditions** that the first and second fundamental forms must satisfy.

### The Gauss equation

$$K = \frac{eg - f^2}{EG - F^2} = \text{(expression in Christoffel symbols)}.$$

This relates the intrinsic curvature (Christoffel symbols) to the extrinsic curvature (second form coefficients).

### The Codazzi-Mainardi equations

These are the remaining compatibility conditions, obtained from the normal components of $\mathbf{r}_{uuv} = \mathbf{r}_{uvu}$ and $\mathbf{r}_{uvv} = \mathbf{r}_{vvu}$:

$$e_v - f_u = e\Gamma^1_{12} + f(\Gamma^2_{12} - \Gamma^1_{11}) - g\Gamma^2_{11},$$
$$f_v - g_u = e\Gamma^1_{22} + f(\Gamma^2_{22} - \Gamma^1_{12}) - g\Gamma^2_{12}.$$

These equations state that the derivatives of the second fundamental form coefficients are not arbitrary — they are constrained by the Christoffel symbols (hence by the first form).

### The fundamental theorem of surfaces

**Theorem (Bonnet).** Let $E, F, G$ and $e, f, g$ be smooth functions on a simply connected domain $U \subset \mathbb{R}^2$, with $EG - F^2 > 0$. If they satisfy the Gauss equation and the Codazzi-Mainardi equations, then there exists a regular parametrized surface $\mathbf{r}: U \to \mathbb{R}^3$ with these as its first and second fundamental form coefficients. Moreover, this surface is unique up to rigid motions of $\mathbb{R}^3$.

This is the surface-theoretic analogue of the fundamental theorem for curves (which states that curvature and torsion determine a space curve up to rigid motion). Together, the Gauss and Codazzi-Mainardi equations form a complete set of integrability conditions.

**Significance.** Bonnet's theorem says that the geometry of a surface in $\mathbb{R}^3$ is completely determined (up to rigid motion) by its first and second fundamental forms — but not any pair will do. The forms must satisfy the Gauss and Codazzi-Mainardi equations. These equations play the role of "compatibility conditions" in the same way that the equality of mixed partials $f_{xy} = f_{yx}$ is a compatibility condition for a function $f$ to exist with prescribed partial derivatives.

In the language of connections, the Gauss equation says that the curvature of the Levi-Civita connection equals $K$ (times the area form), while the Codazzi-Mainardi equations say that the second fundamental form is a "Codazzi tensor" — its covariant derivative is symmetric. These structures generalize to arbitrary Riemannian manifolds, where the Gauss equation becomes the definition of Riemann curvature and the Codazzi equations generalize to the contracted Bianchi identity.

---

## Geodesics: Definition, Equations, and Examples

Geodesics are the curves on a surface that play the role of "straight lines" — they are the paths of shortest distance (locally) and the paths of zero geodesic curvature.

### Definition

A curve $\gamma(t)$ on a surface $S$ is a **geodesic** if its acceleration vector $\gamma''(t)$ is everywhere perpendicular to $S$. Equivalently, the tangential component of $\gamma''(t)$ vanishes — the curve "does not accelerate within the surface."

In coordinates $\gamma(t) = \mathbf{r}(u(t), v(t))$, the geodesic equations are:

$$\ddot{u} + \Gamma^1_{11}\dot{u}^2 + 2\Gamma^1_{12}\dot{u}\dot{v} + \Gamma^1_{22}\dot{v}^2 = 0,$$
$$\ddot{v} + \Gamma^2_{11}\dot{u}^2 + 2\Gamma^2_{12}\dot{u}\dot{v} + \Gamma^2_{22}\dot{v}^2 = 0.$$

This is a system of two second-order ODEs. By the existence and uniqueness theorem for ODEs, given any point $p \in S$ and any tangent direction $v \in T_pS$, there exists a unique geodesic through $p$ in direction $v$ (at least for a short time).

### Variational characterization

Geodesics are critical points of the **length functional**

$$L[\gamma] = \int_a^b \sqrt{E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2}\,dt$$

among all curves joining two given points. By the Euler-Lagrange equations (applied to the energy functional $\frac{1}{2}\int (E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2)\,dt$, which gives the same critical curves when parametrized by arc length), one recovers the geodesic equations above.

Note the subtlety: geodesics are **locally** length-minimizing. A geodesic from the north pole of a sphere to a nearby point is the shorter great-circle arc, which is genuinely shortest. But continue that geodesic past the south pole, and the arc from north pole through the south pole to a point near the north pole is a geodesic but not length-minimizing — the short arc going directly is shorter.

**Geodesics as "inertial trajectories."** In physics, a geodesic is the trajectory of a free particle — one subject to no forces other than the constraint of staying on the surface. If you roll a marble on a smooth sphere, it follows a great circle. This connects differential geometry to Lagrangian mechanics: the geodesic equations are precisely the Euler-Lagrange equations for the kinetic energy Lagrangian $\mathcal{L} = \frac{1}{2}g_{ij}\dot{x}^i\dot{x}^j$ on the surface. In general relativity, this picture extends to spacetime: freely falling particles follow geodesics of the spacetime metric.

**Constant-speed property.** A geodesic parametrized proportionally to arc length has constant speed: $|\gamma'(t)| = \text{const}$. This follows from differentiating $\langle \gamma', \gamma' \rangle$ and using the geodesic condition (the tangential acceleration vanishes). Thus geodesics are "uniform" — they cover equal arc lengths in equal parameter increments.

### Example: geodesics on the sphere

Parametrize $S^2(R)$ in spherical coordinates: $\mathbf{r}(\theta, \phi) = R(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$. The first fundamental form is

$$E = R^2, \quad F = 0, \quad G = R^2\sin^2\theta.$$

The Christoffel symbols (using the orthogonal-coordinate formulas):

$$\Gamma^1_{11} = 0, \quad \Gamma^2_{11} = 0, \quad \Gamma^1_{12} = 0, \quad \Gamma^2_{12} = \frac{\cos\theta}{\sin\theta}, \quad \Gamma^1_{22} = -\sin\theta\cos\theta, \quad \Gamma^2_{22} = 0.$$

The geodesic equations become:

$$\ddot{\theta} - \sin\theta\cos\theta\,\dot{\phi}^2 = 0,$$
$$\ddot{\phi} + 2\frac{\cos\theta}{\sin\theta}\,\dot{\theta}\dot{\phi} = 0.$$

One can verify that **great circles** satisfy these equations. For instance, a meridian $\phi = \text{const}$ has $\dot{\phi} = 0$ and the equations reduce to $\ddot{\theta} = 0$, i.e., $\theta(t) = at + b$ — uniform motion along a meridian, which is indeed a great circle. More generally, any great circle can be parametrized to satisfy these equations, and these are the only geodesics on the sphere.

**Conclusion:** The geodesics of the sphere are exactly the great circles.

This result has a nice physical interpretation: on a frictionless sphere, a freely sliding bead traces a great circle. Airline routes follow great circles (approximately) because they minimize distance on the Earth's surface.

### Example: geodesics on the cylinder

For the cylinder $x^2+y^2 = R^2$ parametrized by $\mathbf{r}(\theta, z) = (R\cos\theta, R\sin\theta, z)$:

$$E = R^2, \quad F = 0, \quad G = 1.$$

All Christoffel symbols vanish (the metric is flat!). The geodesic equations become $\ddot{\theta} = 0$, $\ddot{z} = 0$, giving $\theta(t) = at+b$, $z(t) = ct+d$. These are **helices** (including the special cases of straight generators when $a=0$ and circles when $c=0$).

When we unroll the cylinder onto a plane, these helices become straight lines — consistent with the cylinder being locally isometric to the plane and geodesics being an intrinsic concept.

### Example: geodesics on a cone

Consider the cone $z = \sqrt{x^2+y^2}$, parametrized by $\mathbf{r}(r,\theta) = (r\cos\theta, r\sin\theta, r)$ for $r > 0$. The metric is $E = 2$, $F = 0$, $G = r^2$. Since $K = 0$ for the cone (away from the apex), we can "unroll" it onto a plane sector with opening angle $\pi\sqrt{2}$. Geodesics on the cone correspond to straight lines in the unrolled sector. When we roll the sector back, these straight lines become curves that may or may not pass through the apex — giving rise to the interesting phenomenon that some pairs of points on the cone are connected by geodesics that wrap around the apex multiple times.

### Example: geodesics on surfaces of revolution

For a surface of revolution $\mathbf{r}(u,v) = (f(v)\cos u, f(v)\sin u, g(v))$ with $f > 0$, the first form is $E = f(v)^2$, $F = 0$, $G = f'(v)^2 + g'(v)^2$. A useful result is **Clairaut's relation**: along any geodesic,

$$f(v)\cos\alpha = \text{const},$$

where $\alpha$ is the angle between the geodesic and the parallel (circle of constant $v$). This gives a first integral of the geodesic equations and allows qualitative analysis of geodesic behavior without solving the full ODE system.

**Proof of Clairaut's relation.** Consider a geodesic $\gamma(s)$ on the surface of revolution, parametrized by arc length. The first component of the geodesic equation gives

$$\frac{d}{ds}\left(f^2 \frac{du}{ds}\right) = 0,$$

since the metric coefficients are independent of $u$ (the surface has rotational symmetry). But $f^2\,du/ds = f^2 \cdot (\cos\alpha)/f = f\cos\alpha$, where $\alpha$ is the angle with the parallel. Therefore $f\cos\alpha$ is constant along the geodesic. $\square$

This is an instance of Noether's theorem: the rotational symmetry of the surface of revolution yields a conserved quantity along geodesics. In Riemannian geometry language, $\partial/\partial u$ is a Killing vector field, and $\langle \gamma', \partial/\partial u \rangle$ is constant along any geodesic.

**Applications of Clairaut's relation.** On a surface of revolution with a "waist" (a parallel circle of minimum radius, like the inner equator of a torus), Clairaut's relation implies that no geodesic tangent to the waist can ever cross to the other side — the constant $f\cos\alpha$ would be violated. This gives a purely geometric explanation for why geodesics on the torus exhibit qualitatively different behaviors depending on their initial direction.

---

## Geodesic Curvature and Its Relationship to Gaussian Curvature

For a curve $\gamma$ on $S$ parametrized by arc length, the curvature vector $\gamma''$ decomposes as

$$\gamma'' = \kappa_g\,(\hat{N} \times \gamma') + \kappa_n\,\hat{N},$$

where $\hat{N}$ is the surface normal, $\kappa_n$ is the normal curvature (from the second form), and $\kappa_g$ is the **geodesic curvature** — the component of acceleration tangent to the surface.

**Definition.** The **geodesic curvature** of a curve $\gamma$ on $S$ (parametrized by arc length) is

$$\kappa_g = \langle \gamma'', N \times \gamma' \rangle.$$

Geodesics are precisely the curves with $\kappa_g = 0$ everywhere.

Geodesic curvature is intrinsic: it can be computed from the first fundamental form and the curve's coordinate functions, without reference to the normal $N$ or the ambient space. The formula in coordinates is

$$\kappa_g = \frac{1}{\sqrt{EG-F^2}}\left[\Gamma^2_{11}\dot{u}^3 + (2\Gamma^2_{12} - \Gamma^1_{11})\dot{u}^2\dot{v} + (\Gamma^2_{22} - 2\Gamma^1_{12})\dot{u}\dot{v}^2 - \Gamma^1_{22}\dot{v}^3 + \dot{u}\ddot{v} - \dot{v}\ddot{u}\right].$$

The intrinsic nature of $\kappa_g$ is essential for the Gauss-Bonnet theorem (Article 5), which relates the integral of $\kappa_g$ along boundary curves, the integral of $K$ over a region, and a topological invariant.

**Example: geodesic curvature of a parallel on a sphere.** On $S^2(R)$, consider the parallel circle at colatitude $\theta_0$ (not a geodesic unless $\theta_0 = \pi/2$, i.e., the equator). Its geodesic curvature is

$$\kappa_g = \frac{\cot\theta_0}{R}.$$

At the equator ($\theta_0 = \pi/2$), $\kappa_g = 0$ — confirming that the equator is a geodesic. As $\theta_0 \to 0$ (approaching the north pole), $\kappa_g \to \infty$ — small circles near a pole are tightly curved within the surface.

**Example: geodesic curvature of a circle on a plane.** A circle of radius $r$ in the Euclidean plane has $\kappa_g = 1/r$ and the integral $\int \kappa_g\,ds = (1/r)(2\pi r) = 2\pi$. This is a special case of the Gauss-Bonnet theorem: for the flat disk enclosed by the circle, $K = 0$ and $\chi = 1$, so $\int K\,dA + \int \kappa_g\,ds = 0 + 2\pi = 2\pi\chi$.

### Parallel transport and holonomy

A closely related concept is **parallel transport**: moving a tangent vector along a curve while keeping it "as constant as possible" within the surface. Formally, a vector field $V(t)$ along a curve $\gamma(t)$ is **parallel** if its covariant derivative vanishes:

$$\frac{DV}{dt} = 0.$$

In coordinates, this becomes

$$\dot{V}^k + \Gamma^k_{ij}\dot{\gamma}^i V^j = 0.$$

Parallel transport preserves lengths and angles (since the covariant derivative is compatible with the metric). However, on a curved surface, parallel transport around a closed loop generally does **not** return a vector to its starting position. The angle of rotation after transporting around a closed loop is called the **holonomy**, and it equals the integral of $K$ over the enclosed region — another manifestation of the Gauss-Bonnet philosophy.

**Example.** Parallel-transport a vector around a spherical triangle (an octant of the sphere, with three right angles). Start with a vector pointing east at the north pole. Transport it south along a meridian — it stays pointing east. Transport it west along the equator — it stays pointing south (parallel transport on a geodesic preserves the angle with the tangent). Transport it back north along another meridian — it arrives at the north pole pointing south, having rotated by $\pi/2$ from its starting direction. The holonomy-curvature relationship is a deep manifestation of the Gauss-Bonnet philosophy: local curvature data (the integral of $K$ over a region) determines a global geometric phenomenon (the rotation of a parallel-transported vector). In gauge theory and mathematical physics, holonomy generalizes to connections on fiber bundles, where it becomes the mathematical foundation for concepts like the Berry phase in quantum mechanics and the Aharonov-Bohm effect in electromagnetism.

### The bridge to Gauss-Bonnet

Consider a geodesic triangle on a surface — a triangle whose sides are geodesic arcs. Let $\alpha_1, \alpha_2, \alpha_3$ be the interior angles. The **local Gauss-Bonnet theorem** (which we'll prove in Article 5) states:

$$\alpha_1 + \alpha_2 + \alpha_3 - \pi = \iint_T K\,dA.$$

The angle excess (or deficit) of a geodesic triangle equals the integral of Gaussian curvature over the triangle. On a sphere ($K > 0$), the angles of a geodesic triangle sum to more than $\pi$; on a saddle surface ($K < 0$), they sum to less than $\pi$. This directly observable phenomenon — measured entirely within the surface — reveals the Gaussian curvature, once again confirming its intrinsic nature.

**Quantitative example.** On a sphere of radius $R$, consider a geodesic triangle consisting of a quarter of the equator and two meridians from the equator to the north pole. The three interior angles are all $\pi/2$, so the angle sum is $3\pi/2$, giving an excess of $\pi/2$. The area of this triangle is $4\pi R^2/8 = \pi R^2/2$, and indeed $K \cdot \text{Area} = (1/R^2)(\pi R^2/2) = \pi/2$, matching the angle excess.

On a surface of constant negative curvature $K = -1$ (the hyperbolic plane), a geodesic triangle with area $A$ has angle sum $\pi - A$. As the triangle grows larger, the angle sum shrinks, approaching zero for a triangle of area $\pi$ (an "ideal triangle" with all vertices at infinity).

---

## What's Next

We have established the intrinsic viewpoint: Gaussian curvature is determined by the metric alone, and geodesics provide the intrinsic notion of "straight line." The next article brings the grand culmination: the **Gauss-Bonnet theorem**, which connects the total Gaussian curvature of a closed surface to its Euler characteristic — a purely topological invariant. This result is the first deep bridge between differential geometry and topology, and it foreshadows the Chern-Gauss-Bonnet theorem, the Atiyah-Singer index theorem, and much of modern geometry.

**Summary of key formulas.** For reference, here are the principal results from this article:

| Concept | Key formula or statement |
|:--------|:------------------------|
| Christoffel symbols | $\Gamma^k_{ij}$ determined by $E, F, G$ and their first derivatives |
| Theorema Egregium | $K = -\dfrac{1}{2\sqrt{EG}}\left[\partial_u\!\left(\dfrac{G_u}{\sqrt{EG}}\right) + \partial_v\!\left(\dfrac{E_v}{\sqrt{EG}}\right)\right]$ (orthogonal coords) |
| Geodesic equations | $\ddot{x}^k + \Gamma^k_{ij}\dot{x}^i\dot{x}^j = 0$ |
| Clairaut relation | $f(v)\cos\alpha = \text{const}$ along geodesics on surfaces of revolution |
| Geodesic curvature | $\kappa_g = 0 \iff$ curve is a geodesic |
| Local Gauss-Bonnet preview | $\sum \alpha_i - \pi = \iint_T K\,dA$ for geodesic triangles |

These tools — Christoffel symbols, the intrinsic curvature formula, geodesic equations, and geodesic curvature — constitute the complete computational framework for intrinsic surface geometry. In the next article, they will converge in the proof of the Gauss-Bonnet theorem.

---

*This is Part 4 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 3 — Curvature of Surfaces](/en/differential-geometry/03-second-form-curvature/)*

*Next: [Part 5 — The Gauss-Bonnet Theorem](/en/differential-geometry/05-gauss-bonnet/)*
