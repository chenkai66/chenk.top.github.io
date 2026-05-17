---
title: "Differential Geometry (10): Riemannian Geometry — Metrics, Connections, and Parallel Transport"
date: 2021-11-19 09:00:00
tags:
  - differential-geometry
  - riemannian-geometry
  - connections
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
description: "A Riemannian metric lets us measure lengths, angles, and volumes on any smooth manifold — the Levi-Civita connection provides the canonical notion of parallel transport and geodesics."
disableNunjucks: true
series_order: 10
series_total: 12
translationKey: "differential-geometry-10"
---

Up to this point in our series, we have studied smooth manifolds equipped with a differentiable structure — charts, tangent vectors, differential forms, and exterior calculus. None of this required a notion of *distance*. We could talk about smooth functions and their derivatives, but not about the length of a curve, the angle between two tangent vectors, or whether a path is "straight." To do geometry in the classical sense — to measure, to compare, to speak of curvature — we need additional structure.

That structure is a **Riemannian metric**: a smoothly varying inner product on each tangent space. With it, every smooth manifold becomes a geometric space. This article introduces Riemannian metrics, the Levi-Civita connection (the canonical way to differentiate vector fields along curves), parallel transport, and geodesics.

The story is motivated by a simple question: on a curved surface like the Earth, what is the "straightest" path between two cities? The answer — a great-circle arc — is a geodesic, and finding it requires knowing the metric. Every statement in general relativity about the motion of planets, the bending of light, or the expansion of the universe rests on the Riemannian (or more precisely, Lorentzian) framework we develop here.

---

## Riemannian metrics: definition, existence, and examples

### Definition

A **Riemannian metric** on a smooth manifold $M$ is a smooth assignment of an inner product to each tangent space. Formally, it is a smooth, symmetric, positive-definite $(0,2)$-tensor field $g$: for each $p \in M$,

![Parallel transport on sphere showing holonomy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/10-riemannian-geometry/dg_fig10_parallel_transport.png)


$$g_p : T_pM \times T_pM \to \mathbb{R}$$

is bilinear, symmetric ($g_p(X, Y) = g_p(Y, X)$), and positive definite ($g_p(X, X) > 0$ for $X \ne 0$). In local coordinates $(x^1, \ldots, x^n)$, the metric is specified by its components

$$g_{ij}(x) = g\left(\frac{\partial}{\partial x^i}, \frac{\partial}{\partial x^j}\right),$$

and we write $g = g_{ij}\, dx^i \otimes dx^j$ (using the Einstein summation convention). The matrix $(g_{ij})$ is symmetric and positive definite at every point.

A smooth manifold equipped with a Riemannian metric is called a **Riemannian manifold**, written $(M, g)$.

### Existence

Every smooth manifold admits a Riemannian metric. The proof uses partitions of unity: cover $M$ by charts, define any inner product on each chart (for instance, pull back the Euclidean metric), and glue them together using a partition of unity. Positive definiteness is preserved because it is a convex condition. This existence result is reassuring — but the *choice* of metric matters enormously, as it determines all the geometry.

**What the metric gives us.** Once we have $g$, we can define:

- **Lengths of curves:** $L(\gamma) = \int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t), \dot{\gamma}(t))}\,dt$.
- **Angles between vectors:** $\cos\theta = \frac{g(X, Y)}{\sqrt{g(X,X)}\sqrt{g(Y,Y)}}$.
- **Volumes:** via the Riemannian volume form $\text{vol}_g = \sqrt{\det g_{ij}}\, dx^1 \wedge \cdots \wedge dx^n$.
- **The musical isomorphisms:** the metric provides a canonical identification between vectors and covectors via $X^\flat = g(X, \cdot)$ and $\alpha^\sharp$ defined by $g(\alpha^\sharp, Y) = \alpha(Y)$. In coordinates, "lowering indices" $v_i = g_{ij}v^j$ and "raising indices" $v^i = g^{ij}v_j$.
- **Gradient, divergence, Laplacian:** the differential-geometric versions of these classical operators all require a metric.

### Examples

**Euclidean space.** $\mathbb{R}^n$ with $g = \delta_{ij}\,dx^i \otimes dx^j$ is the flat Euclidean metric. Geodesics are straight lines, curvature vanishes everywhere.

**The round sphere.** $S^2 \subset \mathbb{R}^3$ inherits a metric from the ambient Euclidean metric by restriction (the *induced metric*). In spherical coordinates $(\theta, \varphi)$ where $\theta \in (0, \pi)$ is the polar angle and $\varphi \in (0, 2\pi)$ the azimuthal angle:

$$g_{S^2} = d\theta \otimes d\theta + \sin^2\theta\, d\varphi \otimes d\varphi.$$

The metric components are $g_{\theta\theta} = 1$, $g_{\varphi\varphi} = \sin^2\theta$, and $g_{\theta\varphi} = 0$. Geodesics on $S^2$ are great circles.

**Hyperbolic space.** The upper half-plane model of the hyperbolic plane $\mathbb{H}^2 = \{(x,y) \in \mathbb{R}^2 : y > 0\}$ carries the metric

$$g_{\mathbb{H}^2} = \frac{dx \otimes dx + dy \otimes dy}{y^2}.$$

This metric has constant negative Gaussian curvature $K = -1$. Geodesics are vertical lines and semicircles centered on the $x$-axis. Distances grow as you approach $y = 0$: the "boundary" is infinitely far away.

**Product metrics.** If $(M_1, g_1)$ and $(M_2, g_2)$ are Riemannian manifolds, the product $M_1 \times M_2$ carries the product metric $g = g_1 + g_2$. For example, the flat torus $T^2 = S^1 \times S^1$ with the product of round metrics on each circle factor is a flat Riemannian manifold (zero curvature everywhere), even though it is topologically a torus.

**Conformal changes.** Given a metric $g$ and a smooth positive function $e^{2f}$, the conformally rescaled metric $\tilde{g} = e^{2f}g$ defines a new Riemannian structure that preserves angles but changes lengths. Conformal geometry is central in complex analysis and string theory.

**Induced metrics on submanifolds.** If $M \subset N$ is an embedded submanifold and $N$ carries a Riemannian metric $g_N$, the **induced metric** (or first fundamental form) on $M$ is $g_M = \iota^*g_N$ where $\iota : M \hookrightarrow N$ is the inclusion. This is how the sphere $S^2$, the torus embedded in $\mathbb{R}^3$, and any surface in Euclidean space gets its metric. The Nash embedding theorem guarantees that every Riemannian manifold can be realized as a submanifold of some $\mathbb{R}^N$ with the induced metric — so the abstract definition of a Riemannian metric is no more general than the induced-metric construction, but it is far more convenient.

**Warped products.** A more general construction than the direct product is the **warped product**: given $(B, g_B)$ and $(F, g_F)$, the warped product $B \times_f F$ has metric $g = g_B + f^2 g_F$ where $f : B \to (0,\infty)$ is smooth. The round metric on $S^n$ can be written as a warped product: $g_{S^n} = dt^2 + \sin^2(t) g_{S^{n-1}}$ for $t \in (0,\pi)$, expressing the sphere as a "family of shrinking spheres" from north pole to south pole. Robertson-Walker metrics in cosmology are warped products of an interval with a space of constant curvature.

### The first fundamental form

For a surface $S \subset \mathbb{R}^3$ parametrized by $\mathbf{r}(u, v)$, the induced metric (first fundamental form) has coefficients

$$E = \mathbf{r}_u \cdot \mathbf{r}_u, \quad F = \mathbf{r}_u \cdot \mathbf{r}_v, \quad G = \mathbf{r}_v \cdot \mathbf{r}_v,$$

so that $ds^2 = E\,du^2 + 2F\,du\,dv + G\,dv^2$. For the unit sphere with $\mathbf{r}(\theta, \varphi) = (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta)$, we get $E = 1$, $F = 0$, $G = \sin^2\theta$, recovering the round metric.

For a surface of revolution $\mathbf{r}(u, v) = (f(u)\cos v, f(u)\sin v, g(u))$ where $f'^2 + g'^2 = 1$ (arc-length parametrization of the profile curve), we get $E = 1$, $F = 0$, $G = f(u)^2$, giving $ds^2 = du^2 + f(u)^2\,dv^2$. This is a warped-product metric. If the profile curve is a circle (torus of revolution), the function $f(u)$ is periodic but changes sign between the inner and outer portions, producing regions of positive and negative curvature.

---

## The Levi-Civita connection: existence and uniqueness

### The problem of differentiation

On $\mathbb{R}^n$, differentiating a vector field is straightforward: compare vectors at nearby points by translating them to the same origin. On a general manifold, tangent spaces at different points are different vector spaces — there is no canonical way to compare vectors at $p$ and $q$. A **connection** provides this missing structure.

### Connections (covariant derivatives)

An **affine connection** (or **covariant derivative**) on $M$ is a map $\nabla : \mathfrak{X}(M) \times \mathfrak{X}(M) \to \mathfrak{X}(M)$, written $(X, Y) \mapsto \nabla_X Y$, satisfying:

1. **$C^\infty(M)$-linearity in $X$:** $\nabla_{fX + gZ} Y = f\nabla_X Y + g\nabla_Z Y$.
2. **$\mathbb{R}$-linearity in $Y$:** $\nabla_X(Y + Z) = \nabla_X Y + \nabla_X Z$.
3. **Leibniz rule in $Y$:** $\nabla_X(fY) = (Xf)Y + f\nabla_X Y$.

In local coordinates, a connection is specified by its **Christoffel symbols** $\Gamma^k_{ij}$:

$$\nabla_{\partial_i} \partial_j = \Gamma^k_{ij}\, \partial_k.$$

There are many connections on any manifold. The Riemannian metric selects a unique, preferred one.

### The fundamental theorem of Riemannian geometry

> **Theorem (Levi-Civita).** On any Riemannian manifold $(M, g)$, there exists a unique connection $\nabla$ that is:
>
> 1. **Metric-compatible:** $Xg(Y,Z) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$ for all vector fields $X, Y, Z$.
> 2. **Torsion-free:** $\nabla_X Y - \nabla_Y X = [X, Y]$ for all $X, Y$.

This unique connection is called the **Levi-Civita connection**.

Metric compatibility means that inner products are preserved under parallel transport (lengths and angles do not change). Torsion-freeness means the connection is "twist-free" — parallelograms close up.

### The Christoffel symbol formula

The Levi-Civita connection's Christoffel symbols are explicitly determined by the metric:

$$\Gamma^k_{ij} = \frac{1}{2} g^{k\ell} \left(\frac{\partial g_{j\ell}}{\partial x^i} + \frac{\partial g_{i\ell}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^\ell}\right),$$

where $g^{k\ell}$ denotes the inverse metric ($g^{k\ell}g_{\ell m} = \delta^k_m$). This is sometimes called the **Koszul formula** in coordinate form.

**Example: the sphere $S^2$.** With coordinates $(\theta, \varphi)$ and metric $g = d\theta^2 + \sin^2\theta\, d\varphi^2$, the nonzero Christoffel symbols are:

$$\Gamma^\theta_{\varphi\varphi} = -\sin\theta\cos\theta, \qquad \Gamma^\varphi_{\theta\varphi} = \Gamma^\varphi_{\varphi\theta} = \cot\theta.$$

All other symbols vanish. These encode how the coordinate basis vectors change as you move on the sphere.

**Example: the hyperbolic plane $\mathbb{H}^2$.** With the upper half-plane model $g = \frac{1}{y^2}(dx^2 + dy^2)$, we have $g_{11} = g_{22} = \frac{1}{y^2}$ and $g_{12} = 0$. The nonzero Christoffel symbols are:

$$\Gamma^x_{xy} = \Gamma^x_{yx} = -\frac{1}{y}, \quad \Gamma^y_{xx} = \frac{1}{y}, \quad \Gamma^y_{yy} = -\frac{1}{y}.$$

Note how the Christoffel symbols diverge as $y \to 0$: the "boundary" of hyperbolic space is infinitely far away, and the connection reflects this by producing stronger and stronger corrections to the flat derivative.

**General pattern.** Computing Christoffel symbols by hand is tedious but mechanical. For a diagonal metric $g_{ij} = g_{ii}\delta_{ij}$ (no summation), the nonzero symbols simplify to:

$$\Gamma^i_{ii} = \frac{1}{2g_{ii}}\frac{\partial g_{ii}}{\partial x^i}, \quad \Gamma^i_{ij} = \Gamma^i_{ji} = \frac{1}{2g_{ii}}\frac{\partial g_{ii}}{\partial x^j} \text{ (no sum)}, \quad \Gamma^i_{jj} = -\frac{1}{2g_{ii}}\frac{\partial g_{jj}}{\partial x^i} \text{ for } i \ne j.$$

This formula covers most metrics encountered in practice (spherical, hyperbolic, cosmological, etc.) and greatly accelerates calculations.

---

## Parallel transport along curves

### Definition

Let $\gamma : [0,1] \to M$ be a smooth curve. A vector field $V(t)$ along $\gamma$ is **parallel** (with respect to $\nabla$) if

$$\nabla_{\dot{\gamma}(t)} V(t) = 0 \quad \text{for all } t \in [0,1].$$

In coordinates, if $\gamma(t) = (x^1(t), \ldots, x^n(t))$ and $V(t) = V^k(t)\,\partial_k$, the parallel transport equation becomes

$$\frac{dV^k}{dt} + \Gamma^k_{ij}(\gamma(t))\, \dot{x}^i(t)\, V^j(t) = 0.$$

This is a system of first-order linear ODEs in the components $V^k(t)$. By standard ODE theory, given any initial vector $V(0) \in T_{\gamma(0)}M$, there exists a unique parallel vector field along $\gamma$. The map $V(0) \mapsto V(1)$ is called **parallel transport** along $\gamma$ from $\gamma(0)$ to $\gamma(1)$.

### Properties

- Parallel transport is a **linear isomorphism** between $T_{\gamma(0)}M$ and $T_{\gamma(1)}M$.
- Because the Levi-Civita connection is metric-compatible, parallel transport is an **isometry**: it preserves inner products, hence lengths and angles.
- Parallel transport **depends on the path** — transporting the same vector along two different paths from $p$ to $q$ generally gives different results.

### Example: parallel transport on the sphere

This is the classic example that builds geometric intuition. Place a tangent vector pointing east at the north pole of $S^2$. Transport it parallel along the meridian to the equator, then along the equator by angle $\alpha$, then back up a meridian to the north pole. When the vector returns, it has rotated by angle $\alpha$ relative to its original direction.

For a triangle with three right angles ($\alpha = \pi/2$), the vector rotates by $90°$. This rotation is a direct manifestation of curvature — on a flat surface, parallel transport around any closed loop returns the vector to its original direction. The amount of rotation equals the integral of the Gaussian curvature over the enclosed region (this is the Gauss-Bonnet theorem in action).

**A detailed computation.** Let us trace the parallel transport more carefully. Start at the north pole $N = (0,0,1)$ with the vector $V_0 = \partial_\varphi$ (pointing east). Travel south along the prime meridian $\varphi = 0$ to the equator. Along this meridian, $\dot\gamma = -\partial_\theta$, and the parallel transport equation gives $\frac{dV^\varphi}{d\theta} + \Gamma^\varphi_{\theta\varphi}\dot\theta V^\varphi = 0$, i.e., $\frac{dV^\varphi}{d\theta} - \cot\theta V^\varphi = 0$. The solution is $V^\varphi(\theta) = V^\varphi(0)\sin\theta$, so at the equator ($\theta = \pi/2$), $V^\varphi = V^\varphi(0)$ — the vector still points east. Now travel along the equator (where $\theta = \pi/2$ is constant) for azimuthal distance $\alpha$. The parallel transport equation on the equator is trivial because $\Gamma^\varphi_{\varphi\varphi} = 0$ and $\Gamma^\theta_{\varphi\varphi} = -\sin(\pi/2)\cos(\pi/2) = 0$, so $V$ remains pointing east. Finally, travel back north along the meridian $\varphi = \alpha$ to the north pole. By the same computation as the first leg (reversed), the vector arrives at the north pole pointing in the direction $\varphi = \alpha$ — it has rotated by angle $\alpha$ relative to its starting direction.

The area of the spherical triangle enclosed is $\alpha$ (the area of a spherical triangle on the unit sphere equals its angle excess, which for a triangle with angles $\pi/2, \pi/2, \alpha$ is $\alpha + \pi/2 + \pi/2 - \pi = \alpha$). So the rotation angle equals the enclosed area times the Gaussian curvature ($K = 1$): $\Delta\phi = K \cdot \text{Area} = \alpha$.

### Path dependence equals curvature

The failure of parallel transport to be path-independent is precisely measured by the **curvature tensor**. For an infinitesimal parallelogram spanned by vectors $X$ and $Y$, the holonomy (the rotation accumulated by parallel transport around the loop) is

$$R(X, Y)V = \nabla_X \nabla_Y V - \nabla_Y \nabla_X V - \nabla_{[X,Y]} V.$$

If $R = 0$ everywhere, parallel transport is path-independent, and the manifold is **flat**. We will develop this fully in Article 11.

---

## Geodesics on Riemannian manifolds

### Definition via the connection

A curve $\gamma : [a,b] \to M$ is a **geodesic** if its tangent vector is parallel along itself:

$$\nabla_{\dot{\gamma}} \dot{\gamma} = 0.$$

In coordinates, the geodesic equation becomes the system of second-order ODEs:

$$\ddot{x}^k + \Gamma^k_{ij}\, \dot{x}^i \dot{x}^j = 0.$$

Geodesics are the manifold generalization of "straight lines" — they are curves with zero acceleration (as measured by the connection). On $\mathbb{R}^n$ with the flat metric, $\Gamma^k_{ij} = 0$, and geodesics are literally straight lines. On the sphere, geodesics are great circles.

### The exponential map

At a point $p \in M$ and a tangent vector $v \in T_pM$, the geodesic $\gamma_v(t)$ with $\gamma_v(0) = p$ and $\dot{\gamma}_v(0) = v$ exists for small $t$ by ODE theory. The **exponential map** is defined by

$$\exp_p(v) = \gamma_v(1),$$

provided the geodesic exists to $t = 1$. For small $v$, $\exp_p$ is a diffeomorphism from a neighborhood of $0 \in T_pM$ to a neighborhood of $p$ in $M$. This gives **normal coordinates** (also called geodesic or Riemann normal coordinates) centered at $p$, in which $\Gamma^k_{ij}(p) = 0$ — the connection "looks flat" at the center.

The exponential map is the right tool for converting tangent space computations into manifold geometry. It is used extensively in Riemannian comparison geometry, Lie group theory, and numerical methods on manifolds.

### Geodesics as length-minimizers

A geodesic locally minimizes the length functional $L(\gamma) = \int_a^b \|\dot{\gamma}(t)\|\,dt$, where $\|\dot{\gamma}\| = \sqrt{g(\dot{\gamma}, \dot{\gamma})}$. More precisely, geodesics are critical points of the energy functional $E(\gamma) = \frac{1}{2}\int_a^b g(\dot{\gamma}, \dot{\gamma})\,dt$, and the geodesic equation is the Euler-Lagrange equation for $E$.

Short geodesics are length-minimizing (among all nearby curves). However, long geodesics may fail to be globally length-minimizing: on the sphere, a great-circle arc longer than a semicircle is a geodesic but not the shortest path between its endpoints.

### The cut locus

The **cut locus** of a point $p$ is the set of points where geodesics from $p$ cease to be globally minimizing. Beyond the cut locus, there exists a shorter geodesic (or multiple geodesics of the same length) connecting $p$ to the point. On $S^n$, the cut locus of the north pole is the south pole — the single antipodal point. On a flat torus $\mathbb{R}^2/\mathbb{Z}^2$, the cut locus of the origin is a square. The exponential map is a diffeomorphism from the set of vectors shorter than the cut distance to the complement of the cut locus in $M$.

### Geodesics and calculus of variations

The geodesic equation can also be derived from a variational principle. The **energy functional** is $E(\gamma) = \frac{1}{2}\int_a^b g(\dot\gamma, \dot\gamma)\,dt$. A curve $\gamma$ is a geodesic if and only if it is a critical point of $E$ among curves with fixed endpoints. The Euler-Lagrange equation for $E$ is exactly $\nabla_{\dot\gamma}\dot\gamma = 0$.

Why use energy instead of length? The length functional $L(\gamma) = \int_a^b \|\dot\gamma\|\,dt$ is reparametrization-invariant, which means its Euler-Lagrange equation has a larger solution space (any reparametrization of a geodesic is a critical point of $L$). The energy functional breaks this symmetry: its critical points are geodesics parametrized proportional to arc length. This makes $E$ analytically cleaner to work with.

---

## The Riemannian distance function and completeness

### Distance

The Riemannian metric induces a distance function on $M$:

$$d(p, q) = \inf_\gamma \int_0^1 \|\dot{\gamma}(t)\|\, dt,$$

where the infimum is over all piecewise-smooth curves $\gamma$ from $p$ to $q$. This satisfies the axioms of a metric space (positivity, symmetry, triangle inequality) and induces the same topology as the manifold topology.

### Completeness

A Riemannian manifold is **geodesically complete** if every geodesic can be extended to all time — equivalently, if $\exp_p(v)$ is defined for all $v \in T_pM$ and all $p$. An open disk in $\mathbb{R}^2$ with the flat metric is not complete (geodesics hit the boundary in finite time), while $\mathbb{R}^n$ and $S^n$ are complete.

### The Hopf-Rinow theorem

> **Theorem (Hopf-Rinow).** For a connected Riemannian manifold $(M, g)$, the following are equivalent:
>
> 1. $(M, d)$ is a complete metric space (every Cauchy sequence converges).
> 2. $M$ is geodesically complete.
> 3. For some $p \in M$, $\exp_p$ is defined on all of $T_pM$.
> 4. Every closed bounded subset of $M$ is compact.
>
> Moreover, if any of these hold, then any two points in $M$ can be joined by a length-minimizing geodesic.

The Hopf-Rinow theorem is the Riemannian analogue of the Heine-Borel theorem. It guarantees that on a complete manifold, shortest paths always exist. Compact manifolds are automatically complete.

**Example.** Hyperbolic space $\mathbb{H}^n$ is complete: geodesics extend to infinite length in both directions (even though the Poincare model makes them look finite, the hyperbolic metric stretches distances near the boundary).

**Example.** The punctured plane $\mathbb{R}^2 \setminus \{0\}$ with the flat metric is **not** complete: the geodesic heading straight toward the origin stops in finite time. Similarly, an open interval $(0,1)$ with the standard metric is not complete. These examples illustrate that completeness is a geometric condition — it depends on the metric, not just the topology.

**Metric completion.** Any incomplete Riemannian manifold can be completed as a metric space (by adding limit points of Cauchy sequences), but the completed space may fail to be a smooth manifold. For instance, the metric completion of the punctured plane $\mathbb{R}^2 \setminus \{0\}$ (with the flat metric) is $\mathbb{R}^2$, which is a smooth manifold. But for the metric completion of a manifold with a conical singularity, the added point is not smooth. Understanding when and how completeness fails is important in general relativity, where incomplete geodesics signal the presence of singularities (like black holes or the Big Bang).

**The Cartan-Hadamard theorem.** If $(M, g)$ is complete and simply connected with $K \le 0$ (non-positive sectional curvature everywhere), then $\exp_p : T_pM \to M$ is a diffeomorphism for any $p$. In particular, $M$ is diffeomorphic to $\mathbb{R}^n$. This is a strong topological conclusion from a curvature condition: simply connected manifolds of non-positive curvature have the simplest possible topology.

---

## Isometries and Killing fields

### Isometries

An **isometry** between Riemannian manifolds $(M, g)$ and $(N, h)$ is a diffeomorphism $\phi : M \to N$ such that $\phi^*h = g$. If $M = N$, we call $\phi$ an isometry of $(M, g)$. The set of all isometries $M \to M$ forms a group $\text{Isom}(M, g)$.

Isometries preserve everything geometric: distances, angles, geodesics, curvature. The Myers-Steenrod theorem states that $\text{Isom}(M, g)$ is a Lie group (with the compact-open topology), and for a compact manifold it is itself compact.

**Examples.**

- $\text{Isom}(\mathbb{R}^n, g_{\text{flat}}) = \mathbb{R}^n \rtimes O(n)$: translations and orthogonal transformations.
- $\text{Isom}(S^n, g_{\text{round}}) = O(n+1)$: the full orthogonal group.
- $\text{Isom}(\mathbb{H}^n, g_{\text{hyp}})$: in dimension 2, this is $\text{PSL}(2, \mathbb{R})$, the group of Mobius transformations preserving the upper half-plane.

### Killing vector fields

A **Killing vector field** $K$ on $(M, g)$ is an infinitesimal isometry: its flow preserves the metric. Formally, the **Lie derivative** of $g$ along $K$ vanishes:

$$\mathcal{L}_K g = 0.$$

In coordinates, this is equivalent to the **Killing equation**:

$$\nabla_i K_j + \nabla_j K_i = 0,$$

where $K_i = g_{ij}K^j$. Killing fields form a Lie algebra (under the Lie bracket of vector fields), which is the Lie algebra of the isometry group.

**Example.** On $S^2$, the rotations about three orthogonal axes give three linearly independent Killing fields — the Lie algebra $\mathfrak{so}(3)$. On $\mathbb{R}^n$, translations and infinitesimal rotations give $n + \binom{n}{2} = \frac{n(n+1)}{2}$ Killing fields.

A Riemannian manifold with the maximum number $\frac{n(n+1)}{2}$ of independent Killing fields is called a **space of maximal symmetry**. These are precisely the spaces of constant sectional curvature: $\mathbb{R}^n$, $S^n$, and $\mathbb{H}^n$.

### The Hodge star and the Laplacian

The Riemannian metric, together with an orientation, defines the **Hodge star operator** $* : \Omega^k(M) \to \Omega^{n-k}(M)$. On an oriented orthonormal coframe $\{e^1, \ldots, e^n\}$, the Hodge star acts by

$$*(e^{i_1} \wedge \cdots \wedge e^{i_k}) = \epsilon_{i_1 \cdots i_k j_1 \cdots j_{n-k}} e^{j_1} \wedge \cdots \wedge e^{j_{n-k}},$$

where $\epsilon$ is the Levi-Civita symbol. In $\mathbb{R}^3$, $*dx = dy \wedge dz$, $*dy = dz \wedge dx$, $*dz = dx \wedge dy$ — these are the familiar relations underlying the cross product.

The Hodge star allows us to define the **codifferential** $\delta = (-1)^{n(k+1)+1} * d * : \Omega^k \to \Omega^{k-1}$ and the **Hodge Laplacian** $\Delta = d\delta + \delta d$. A form is **harmonic** if $\Delta \omega = 0$. The **Hodge theorem** asserts that on a compact oriented Riemannian manifold, every de Rham cohomology class contains a unique harmonic representative. This provides a concrete realization of abstract cohomology classes and connects topology to analysis (elliptic PDE theory).

---

## What's next

We now have the tools to do geometry: a metric to measure, a connection to differentiate, and parallel transport to compare. The natural next question is: **how curved is a Riemannian manifold?** In the next article, we introduce the Riemann curvature tensor and its contractions — Ricci curvature and scalar curvature — which quantify curvature in all its forms. These are the objects that appear in Einstein's field equations and in the great theorems of Riemannian geometry.

### Suggestions for building intuition

The concepts in this article are best internalized through concrete computation. Here are exercises that reward careful work:

1. **Compute Christoffel symbols** for the Poincare disk model of $\mathbb{H}^2$: the open unit disk $\{(u,v) : u^2 + v^2 < 1\}$ with metric $g = \frac{4(du^2 + dv^2)}{(1 - u^2 - v^2)^2}$. Verify that geodesics are arcs of circles meeting the boundary at right angles.

2. **Parallel transport on the cylinder** $S^1 \times \mathbb{R}$ with the product metric: show that parallel transport around a circle at height $z$ returns every vector to itself (the cylinder is flat). Compare this to the sphere, where parallel transport around a latitude circle rotates vectors.

3. **Verify the Hopf-Rinow theorem** for the open unit disk with the Euclidean metric (not complete: geodesics hit the boundary) versus the Poincare disk metric (complete: the hyperbolic metric makes the boundary infinitely far away, so geodesics never reach it in finite time).

4. **Count Killing fields** on the flat torus $T^2 = \mathbb{R}^2/\mathbb{Z}^2$: show that translations in the $x$- and $y$-directions are Killing, but rotations are *not* (they do not preserve the lattice). So $T^2$ has only 2 independent Killing fields, far fewer than the maximum $\frac{2 \cdot 3}{2} = 3$ for a 2-manifold.

These examples illustrate the interplay between the metric, the connection, and the global topology that makes Riemannian geometry so rich.

---

*This is Part 10 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 9 — Integration and Stokes' Theorem](/en/differential-geometry/09-integration-stokes/)*

*Next: [Part 11 — Curvature on Manifolds](/en/differential-geometry/11-curvature-on-manifolds/)*
