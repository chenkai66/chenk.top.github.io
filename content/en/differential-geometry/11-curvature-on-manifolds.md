---
title: "Differential Geometry (11): Curvature in Riemannian Geometry — Riemann, Ricci, and Scalar"
date: 2021-11-21 09:00:00
tags:
  - differential-geometry
  - curvature-tensor
  - ricci
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
description: "The Riemann curvature tensor captures all intrinsic curvature information — its contractions (Ricci and scalar curvature) control volume growth, geodesic deviation, and Einstein's equations."
disableNunjucks: true
series_order: 11
series_total: 12
translationKey: "differential-geometry-11"
---

Curvature is the central concept of Riemannian geometry. Intuitively, it measures how much a space deviates from being flat — how parallel lines converge or diverge, how triangles have angle excess or deficit, how volumes grow differently from Euclidean expectations. In the previous article, we saw that the path-dependence of parallel transport signals the presence of curvature. Now we make this precise.

The **Riemann curvature tensor** encodes all intrinsic curvature information of a Riemannian manifold. Its various contractions — **Ricci curvature** and **scalar curvature** — extract progressively coarser (but more manageable) summaries. These objects are not mere abstractions: Ricci curvature controls volume comparison theorems in geometry, scalar curvature appears in the Hilbert action of general relativity, and sectional curvature classifies the three model geometries.

The plan for this article is as follows. We begin with geometric intuition (curvature as failure of commutativity), then give the precise definition and coordinate formula for the Riemann tensor, study its algebraic and differential symmetries, and define sectional, Ricci, and scalar curvature. We conclude with the classification of constant-curvature spaces and a glimpse of the profound theorems linking curvature to topology.

---

## Curvature as failure of parallel transport to commute

### Geometric motivation

On flat $\mathbb{R}^n$, partial derivatives commute: $\partial_i \partial_j f = \partial_j \partial_i f$. Similarly, covariant derivatives commute on flat space: $\nabla_X \nabla_Y V = \nabla_Y \nabla_X V$ for coordinate vector fields $X, Y$. On a curved manifold, this commutativity fails, and the **failure** is curvature.

Consider two infinitesimally close paths from a point $p$: first move along $X$ then along $Y$, versus first along $Y$ then along $X$. Parallel-transport a vector $V$ along both paths. On a flat manifold, the results agree. On a curved manifold, they differ, and the discrepancy is linear in $V$ — so it defines a linear map on $T_pM$.

More concretely, imagine parallel-transporting a vector around an infinitesimal parallelogram. When the vector returns to the starting point, it has been rotated. The rotation angle (per unit area of the parallelogram) is the curvature. This is exactly what we saw on $S^2$: parallel transport around a spherical triangle rotates vectors by an amount equal to the area of the triangle (times the Gaussian curvature).

---


![Curvature hierarchy: Riemann, Ricci, and scalar](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/11-curvature-on-manifolds/dg_fig11_curvature_hierarchy.png)

## The Riemann curvature tensor $R$

### Definition via the connection

The **Riemann curvature tensor** (or **curvature endomorphism**) is defined by

$$R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z$$

for vector fields $X, Y, Z$. Despite appearances, $R(X, Y)Z$ at a point $p$ depends only on the values of $X, Y, Z$ at $p$ (not on their derivatives) — it is a genuine tensor.

The type of $R$ is $(1,3)$: it takes three vectors and produces one. Equivalently, using the metric to lower an index, we get the $(0,4)$-tensor

$$R(X, Y, Z, W) = g(R(X, Y)Z, W),$$

often written $R_{ijkl}$ in coordinates.

### Coordinate formula

In a coordinate basis $\{\partial_1, \ldots, \partial_n\}$, the components of the Riemann tensor are

$$R^\rho_{\ \sigma\mu\nu} = \partial_\mu \Gamma^\rho_{\nu\sigma} - \partial_\nu \Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}.$$

This formula makes the dependence on the Christoffel symbols explicit: curvature involves first derivatives of $\Gamma$ (hence second derivatives of the metric) plus quadratic terms in $\Gamma$.

**Example: the 2-sphere $S^2$.** Using coordinates $(\theta, \varphi)$ and the Christoffel symbols from Article 10 ($\Gamma^\theta_{\varphi\varphi} = -\sin\theta\cos\theta$, $\Gamma^\varphi_{\theta\varphi} = \cot\theta$), a direct computation gives

$$R^\theta_{\ \varphi\theta\varphi} = \sin^2\theta, \qquad R^\varphi_{\ \theta\varphi\theta} = 1.$$

The fully covariant tensor has $R_{\theta\varphi\theta\varphi} = g_{\theta\theta} R^\theta_{\ \varphi\theta\varphi} = \sin^2\theta$. For the unit sphere, the Gaussian curvature is $K = R_{\theta\varphi\theta\varphi}/(g_{\theta\theta}g_{\varphi\varphi} - g_{\theta\varphi}^2) = \sin^2\theta / \sin^2\theta = 1$, confirming that $S^2$ has constant curvature $+1$.

**Example: hyperbolic plane $\mathbb{H}^2$.** In the upper half-plane model with $g = \frac{1}{y^2}(dx^2 + dy^2)$, a computation using the Christoffel symbols gives $R_{xyxy} = -\frac{1}{y^4}$. The Gaussian curvature is $K = R_{xyxy}/(g_{xx}g_{yy} - g_{xy}^2) = \frac{-1/y^4}{1/y^4} = -1$, confirming constant negative curvature.

**Example: flat torus.** The torus $T^2 = \mathbb{R}^2/\mathbb{Z}^2$ with the flat metric inherited from $\mathbb{R}^2$ has all Christoffel symbols zero, hence $R = 0$ identically. The torus is flat despite being topologically non-trivial. This shows that curvature is a purely geometric (metric) property, not a topological one — though the two are related by deep theorems like Gauss-Bonnet.

**Example: the Schwarzschild metric.** In general relativity, the spacetime geometry outside a spherically symmetric, non-rotating mass $M$ is described by the Schwarzschild metric (in coordinates $(t, r, \theta, \varphi)$):

$$ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right)c^2 dt^2 + \left(1 - \frac{2GM}{c^2 r}\right)^{-1}dr^2 + r^2(d\theta^2 + \sin^2\theta\, d\varphi^2).$$

Computing the Riemann tensor for this metric yields the tidal forces experienced near a massive body. The Ricci tensor vanishes everywhere ($R_{ij} = 0$ — this is a vacuum solution), but the Riemann tensor is nonzero (spacetime is curved). The nonzero curvature is entirely in the Weyl tensor, which describes the tidal stretching and compression experienced by freely falling objects. At the Schwarzschild radius $r_s = 2GM/c^2$, the coordinate singularity can be removed by a change of coordinates (Eddington-Finkelstein or Kruskal-Szekeres), revealing the event horizon of a black hole. At $r = 0$, the Kretschner scalar $R_{ijkl}R^{ijkl} \to \infty$, indicating a genuine curvature singularity.

### Why the Riemann tensor matters

The Riemann tensor is the **complete obstruction to flatness**: a Riemannian manifold is locally isometric to Euclidean space if and only if $R = 0$ everywhere. It also controls:

- **Geodesic deviation:** Two initially parallel geodesics accelerate apart (or together) at a rate determined by $R$. The Jacobi equation $\nabla_{\dot\gamma}^2 J + R(J, \dot\gamma)\dot\gamma = 0$ governs this.
- **Holonomy:** The rotation of parallel transport around a loop is determined by the integral of $R$ over any surface bounded by the loop.
- **Topology:** Via the Gauss-Bonnet theorem and its generalizations, curvature integrals compute topological invariants.

### The Jacobi equation in detail

The **Jacobi equation** (or geodesic deviation equation) is

$$\nabla_{\dot\gamma}^2 J + R(J, \dot\gamma)\dot\gamma = 0,$$

where $J$ is a **Jacobi field** — a vector field along a geodesic $\gamma$ that measures how nearby geodesics separate. If you start a family of geodesics from a point $p$ with slightly different initial velocities, the Jacobi field $J(t)$ tracks the displacement between neighboring geodesics at time $t$.

In constant-curvature spaces, the Jacobi equation has explicit solutions. If $K > 0$ (sphere), $|J(t)| \propto \sin(\sqrt{K}t)$ — nearby geodesics oscillate and reconverge (they meet again at the antipodal point). If $K = 0$ (flat), $|J(t)| \propto t$ — linear separation. If $K < 0$ (hyperbolic), $|J(t)| \propto \sinh(\sqrt{|K|}t)$ — exponential divergence. This trichotomy is the root of all comparison geometry: positive curvature focuses geodesics, negative curvature defocuses them.

**Physical interpretation.** In general relativity, the Jacobi equation governs **tidal forces**. If you are freely falling in a gravitational field, two nearby test particles (following neighboring geodesics) experience relative acceleration proportional to $R(J, \dot\gamma)\dot\gamma$. This is why the Riemann tensor is sometimes called the "tidal tensor" — it describes the stretching and squeezing of matter due to spacetime curvature. Near a black hole, tidal forces become enormous (the Riemann tensor components grow without bound), leading to the phenomenon of spaghettification.

---

## Symmetries of the curvature tensor

The Riemann tensor has $n^4$ components in $n$ dimensions but satisfies powerful symmetry constraints that drastically reduce the number of independent components.

### Algebraic symmetries

For the fully covariant Riemann tensor $R_{ijkl} = g(R(\partial_i, \partial_j)\partial_k, \partial_l)$:

1. **Antisymmetry in the first pair:** $R_{ijkl} = -R_{jikl}$.
2. **Antisymmetry in the second pair:** $R_{ijkl} = -R_{ijlk}$.
3. **Pair symmetry:** $R_{ijkl} = R_{klij}$.
4. **First Bianchi identity (algebraic):** $R_{ijkl} + R_{iklj} + R_{iljk} = 0$.

These reduce the independent components from $n^4$ to $\frac{n^2(n^2-1)}{12}$. In dimension 2, there is only **one** independent component (the Gaussian curvature). In dimension 3, there are 6. In dimension 4 (the dimension of spacetime), there are 20.

**Counting independent components:**

| $n$ | $n^4$ | $\frac{n^2(n^2-1)}{12}$ |
|---|---|---|
| 2 | 16 | 1 |
| 3 | 81 | 6 |
| 4 | 256 | 20 |
| 5 | 625 | 50 |

In general relativity ($n = 4$), the 20 independent components split into: 10 from the Ricci tensor (determined by the matter content via Einstein's equations) and 10 from the Weyl tensor (the "free gravitational field" that propagates as gravitational waves).

### The differential Bianchi identity

The curvature tensor also satisfies a differential identity:

$$\nabla_m R_{ijkl} + \nabla_k R_{ijlm} + \nabla_l R_{ijmk} = 0.$$

This is the **second Bianchi identity** (or differential Bianchi identity). When contracted, it yields the crucial identity

$$\nabla^i G_{ij} = 0,$$

where $G_{ij} = R_{ij} - \frac{1}{2}Rg_{ij}$ is the **Einstein tensor**. This identity — the divergence-freeness of the Einstein tensor — is the mathematical foundation of Einstein's field equations in general relativity, ensuring local conservation of energy-momentum.

---

## Sectional curvature

### Definition

For a 2-dimensional subspace $\sigma = \text{span}(X, Y) \subset T_pM$, the **sectional curvature** is

$$K(\sigma) = K(X, Y) = \frac{R(X, Y, Y, X)}{g(X, X)g(Y, Y) - g(X, Y)^2} = \frac{R(X, Y, Y, X)}{|X \wedge Y|^2}.$$

The denominator is the squared area of the parallelogram spanned by $X$ and $Y$, ensuring $K$ depends only on the plane $\sigma$, not on the choice of basis vectors.

### Relation to Gaussian curvature

If $M$ is 2-dimensional, there is only one 2-plane at each point (the entire tangent plane), and the sectional curvature equals the **Gaussian curvature** $K$. In higher dimensions, the sectional curvature of the plane $\sigma$ equals the Gaussian curvature of the 2-dimensional surface obtained by exponentiating $\sigma$ (the image of $\sigma$ under $\exp_p$).

Sectional curvature determines the full Riemann tensor: if you know $K(\sigma)$ for all 2-planes $\sigma$, you can recover $R_{ijkl}$ completely.

### Constant curvature spaces

A Riemannian manifold has **constant sectional curvature** $\kappa$ if $K(\sigma) = \kappa$ for all 2-planes at all points. The Riemann tensor then takes the simple form

$$R_{ijkl} = \kappa(g_{ik}g_{jl} - g_{il}g_{jk}).$$

The three model spaces of constant curvature are:

| Curvature $\kappa$ | Model space | Geometry |
|---|---|---|
| $\kappa > 0$ | $S^n(\kappa)$ (sphere of radius $1/\sqrt{\kappa}$) | Spherical (elliptic) |
| $\kappa = 0$ | $\mathbb{R}^n$ | Euclidean (flat) |
| $\kappa < 0$ | $\mathbb{H}^n(\kappa)$ (hyperbolic space) | Hyperbolic |

These three geometries have maximal symmetry: their isometry groups have dimension $\frac{n(n+1)}{2}$, the maximum possible. They serve as the "model geometries" against which all other Riemannian manifolds are compared.

---

## Ricci curvature and scalar curvature

### Ricci curvature

The **Ricci curvature tensor** is obtained by contracting (tracing) the Riemann tensor:

$$\text{Ric}(X, Y) = R_{ij} = R^k_{\ ikj} = \sum_{k=1}^n R(e_k, X, Y, e_k),$$

where $\{e_1, \ldots, e_n\}$ is an orthonormal basis. $\text{Ric}$ is a symmetric $(0,2)$-tensor — it takes two vectors and returns a number.

### Geometric meaning of Ricci curvature

Ricci curvature measures the average sectional curvature in a direction. Specifically, if $v$ is a unit vector and $\{v, e_2, \ldots, e_n\}$ is an orthonormal basis, then

$$\text{Ric}(v, v) = \sum_{i=2}^n K(v, e_i).$$

This has a direct geometric consequence: **Ricci curvature controls volume growth.** The volume of a small geodesic ball of radius $r$ centered at $p$ satisfies

$$\frac{\text{Vol}(B_r(p))}{\text{Vol}(B_r^{\text{Eucl}})} = 1 - \frac{R_{ij}v^i v^j}{6(n+2)} r^2 + O(r^4),$$

where $v$ is averaged over directions. Positive Ricci curvature means geodesic balls are *smaller* than Euclidean ones (geodesics converge); negative Ricci curvature means they are *larger* (geodesics diverge).

**Bishop-Gromov volume comparison.** If $\text{Ric} \ge (n-1)\kappa g$ (Ricci curvature bounded below by that of the model space of curvature $\kappa$), then the volume ratio $\text{Vol}(B_r(p))/\text{Vol}(B_r^\kappa)$ is non-increasing in $r$. This is one of the most powerful tools in Riemannian geometry.

**The Myers theorem.** If $\text{Ric} \ge (n-1)\kappa g$ with $\kappa > 0$, then $M$ is compact with diameter $\le \pi/\sqrt{\kappa}$. This is remarkable: a *local* curvature condition forces a *global* topological conclusion. Applying Myers' theorem to $S^n(r)$ (which has $\text{Ric} = \frac{n-1}{r^2}g$), we recover the obvious fact that $S^n(r)$ has diameter $\pi r$.

**The Cheeger-Gromoll splitting theorem.** If $M$ is a complete Riemannian manifold with $\text{Ric} \ge 0$ that contains a line (a geodesic that is minimizing on every segment), then $M$ is isometric to a product $N \times \mathbb{R}$. This shows that non-negative Ricci curvature severely constrains the topology at large scales.

### Scalar curvature

The **scalar curvature** is the trace of the Ricci tensor:

$$R = g^{ij}R_{ij} = \sum_{i=1}^n \text{Ric}(e_i, e_i) = \sum_{i < j} K(e_i, e_j) \cdot 2.$$

Scalar curvature is a single number at each point — the crudest summary of curvature. It measures the leading-order deviation of the volume of small balls from the Euclidean volume:

$$\text{Vol}(B_r(p)) = \text{Vol}(B_r^{\text{Eucl}}) \left(1 - \frac{R(p)}{6(n+2)} r^2 + O(r^4)\right).$$

For $S^n$ of radius $1$, the scalar curvature is $R = n(n-1)$. For $\mathbb{H}^n$ with curvature $-1$, $R = -n(n-1)$.

### Einstein manifolds

A Riemannian manifold is an **Einstein manifold** if the Ricci tensor is proportional to the metric:

$$R_{ij} = \frac{R}{n} g_{ij}.$$

Constant-curvature spaces are Einstein, but the converse is false in dimensions $\ge 4$. Einstein manifolds are critical points of the total scalar curvature functional (the Einstein-Hilbert action) and are central to both Riemannian geometry and physics.

In general relativity, vacuum solutions of Einstein's equations satisfy $R_{ij} = 0$ (Ricci-flat), not $R_{ij} = \lambda g_{ij}$. With a cosmological constant $\Lambda$, vacuum solutions satisfy $R_{ij} = \Lambda g_{ij}$, making them Einstein manifolds. De Sitter space ($\Lambda > 0$) and anti-de Sitter space ($\Lambda < 0$) are the maximally symmetric Einstein manifolds and serve as model spacetimes in cosmology and string theory, respectively.

### The hierarchy

The curvature hierarchy for an $n$-dimensional Riemannian manifold is:

$$\underbrace{R_{ijkl}}_{\frac{n^2(n^2-1)}{12} \text{ components}} \xrightarrow{\text{trace}} \underbrace{R_{ij}}_{{\frac{n(n+1)}{2}} \text{ components}} \xrightarrow{\text{trace}} \underbrace{R}_{1 \text{ component}}.$$

Each level loses information. In dimension 2, $R_{ij}$ and $R$ both determine $R_{ijkl}$ completely. In dimension 3, $R_{ij}$ determines $R_{ijkl}$ (via the decomposition using the Weyl tensor, which vanishes in 3D). Starting in dimension 4, the Weyl tensor $W_{ijkl}$ carries independent information — it is the "traceless" part of the Riemann tensor and governs conformal geometry and gravitational waves.

### The Weyl decomposition

In dimensions $n \ge 3$, the Riemann tensor decomposes uniquely as

$$R_{ijkl} = W_{ijkl} + \frac{1}{n-2}\left(R_{ik}g_{jl} - R_{il}g_{jk} + R_{jl}g_{ik} - R_{jk}g_{il}\right) - \frac{R}{(n-1)(n-2)}(g_{ik}g_{jl} - g_{il}g_{jk}).$$

The **Weyl tensor** $W_{ijkl}$ has all the symmetries of the Riemann tensor and is additionally trace-free: $W^i_{\ jik} = 0$. It is conformally invariant: under a conformal change $\tilde{g} = e^{2f}g$, the Weyl tensor of $\tilde{g}$ equals the Weyl tensor of $g$ (with appropriate index positions). A Riemannian manifold is conformally flat (locally conformal to Euclidean space) if and only if:
- $W = 0$ in dimensions $n \ge 4$,
- the Cotton tensor $C_{ijk} = \nabla_k R_{ij} - \nabla_j R_{ik} - \frac{1}{2(n-1)}(\nabla_k R\, g_{ij} - \nabla_j R\, g_{ik})$ vanishes in dimension $n = 3$.

In physics, the Weyl tensor describes tidal forces and gravitational radiation — the part of the gravitational field that propagates through vacuum, where Ricci curvature vanishes.

---

## Spaces of constant curvature: classification

### The classification theorem

A complete, simply connected Riemannian manifold of constant sectional curvature $\kappa$ is isometric to one of:

- $S^n(1/\sqrt{\kappa})$ if $\kappa > 0$ (the sphere of radius $1/\sqrt{\kappa}$),
- $\mathbb{R}^n$ if $\kappa = 0$,
- $\mathbb{H}^n$ (with metric scaled so curvature is $\kappa$) if $\kappa < 0$.

This is a uniqueness theorem: the curvature constant and the topology (simply connected, complete) pin down the geometry entirely. Any other complete manifold of constant curvature is a quotient of one of these three by a discrete group of isometries acting freely and properly discontinuously.

### Space forms

The quotients are called **space forms**:

- **Positive curvature:** Quotients of $S^n$ include real projective space $\mathbb{RP}^n = S^n/\mathbb{Z}_2$ and lens spaces $L(p,q) = S^3/\mathbb{Z}_p$.
- **Zero curvature:** Quotients of $\mathbb{R}^n$ are the **flat manifolds** (Bieberbach groups). In dimension 2, the only compact flat surfaces are the torus and the Klein bottle.
- **Negative curvature:** Quotients of $\mathbb{H}^n$ give hyperbolic manifolds. In dimension 2, every closed surface of genus $g \ge 2$ admits a hyperbolic metric. In dimension 3, Thurston's geometrization (proved by Perelman) shows that hyperbolic pieces are generically the dominant geometry in the decomposition of 3-manifolds.

### The Gauss-Bonnet theorem

The deepest connection between curvature and topology for surfaces is the **Gauss-Bonnet theorem**: for a compact oriented 2-manifold $M$ without boundary,

$$\int_M K\, dA = 2\pi\chi(M),$$

where $K$ is the Gaussian curvature, $dA$ is the area form, and $\chi(M)$ is the Euler characteristic. For $S^2$, $K = 1$ everywhere, so $\int K\,dA = 4\pi = 2\pi \cdot 2$. For a torus ($\chi = 0$), the total curvature vanishes — regions of positive curvature on the outer edge exactly cancel regions of negative curvature on the inner edge.

The Gauss-Bonnet theorem has a generalization to higher dimensions — the **Chern-Gauss-Bonnet theorem** — which expresses the Euler characteristic as an integral of a polynomial in the Riemann curvature tensor (the Pfaffian of the curvature 2-form). This will reappear when we discuss characteristic classes in the next article.

### Bonnet-Myers and the topology of positive curvature

The Gauss-Bonnet theorem is just the beginning of the deep connections between curvature and topology. Here are some landmark results:

**Synge's theorem.** If $M$ is a compact Riemannian manifold with strictly positive sectional curvature ($K > 0$), then:
- If $n$ is even, $M$ is simply connected.
- If $n$ is odd, $M$ is orientable.

This follows from analyzing the second variation of arc length for closed geodesics: positive curvature forces nearby geodesics to be shorter, which shrinks any closed geodesic to a point.

**The sphere theorem.** If $M$ is a compact, simply connected Riemannian manifold with sectional curvature satisfying $\frac{1}{4} < K \le 1$, then $M$ is homeomorphic to $S^n$. The bound $1/4$ is sharp: $\mathbb{CP}^{n/2}$ has $\frac{1}{4} \le K \le 1$ and is not homeomorphic to a sphere. The differentiable version of the sphere theorem (proving diffeomorphism, not just homeomorphism) was proved by Brendle and Schoen in 2009 using Ricci flow.

**Preissman's theorem.** If $M$ is a compact Riemannian manifold with $K < 0$, then every abelian subgroup of $\pi_1(M)$ is cyclic (isomorphic to $\mathbb{Z}$). This rules out, for instance, the torus $T^n$ (whose fundamental group is $\mathbb{Z}^n$) from carrying a metric of negative curvature.

### Concrete calculations

**Sphere $S^n(r)$ of radius $r$:** Sectional curvature $K = 1/r^2$, Ricci curvature $\text{Ric} = \frac{n-1}{r^2}g$, scalar curvature $R = \frac{n(n-1)}{r^2}$, volume $\text{Vol}(S^n(r)) = r^n \cdot \text{Vol}(S^n(1))$.

**Hyperbolic space $\mathbb{H}^n$:** In the upper half-space model $\{x \in \mathbb{R}^n : x_n > 0\}$ with $g = \frac{1}{x_n^2}(dx_1^2 + \cdots + dx_n^2)$, we get $K = -1$, $\text{Ric} = -(n-1)g$, $R = -n(n-1)$. Volume of balls grows exponentially: $\text{Vol}(B_r) \sim c \cdot e^{(n-1)r}$ for large $r$, in stark contrast with the polynomial growth $\sim r^n$ in Euclidean space.

**The three model geometries in dimension 2.** For surfaces, the three constant-curvature geometries have intuitive characterizations:

| Property | $S^2$ ($K = 1$) | $\mathbb{R}^2$ ($K = 0$) | $\mathbb{H}^2$ ($K = -1$) |
|---|---|---|---|
| Triangle angle sum | $> \pi$ | $= \pi$ | $< \pi$ |
| Circumference of radius-$r$ circle | $2\pi\sin r$ | $2\pi r$ | $2\pi\sinh r$ |
| Area of radius-$r$ disk | $2\pi(1 - \cos r)$ | $\pi r^2$ | $2\pi(\cosh r - 1)$ |
| Parallel lines | None (all lines meet) | Unique parallel | Infinitely many parallels |

The triangle angle sum formula generalizes to the **Gauss-Bonnet formula for geodesic triangles**: $\alpha + \beta + \gamma = \pi + K \cdot \text{Area}(\Delta)$ on a surface of constant curvature $K$. On $S^2$, the excess $\alpha + \beta + \gamma - \pi$ equals the area (times $K$); on $\mathbb{H}^2$, the deficiency $\pi - \alpha - \beta - \gamma$ equals the area (times $|K|$). This provides an intrinsic, measurement-based way to detect curvature: measure the angles of a triangle and compare to $\pi$.

**Comparison theorems.** The three model spaces serve as baselines for general Riemannian manifolds through comparison theorems:

- **Toponogov's comparison theorem:** If sectional curvature $K \ge \kappa$, then geodesic triangles in $M$ are "thinner" than corresponding triangles in the model space of curvature $\kappa$. More precisely, if a geodesic triangle in $M$ has the same side lengths as a triangle in the model space, then its angles are at least as large.
- **Rauch comparison theorem:** If $K \ge \kappa$ (resp. $K \le \kappa$), then Jacobi fields grow at most as fast (resp. at least as fast) as in the model space. This gives precise control over how fast nearby geodesics diverge.

These comparison results are the engine behind many of the deep theorems in Riemannian geometry, including the sphere theorem, the Cheeger finiteness theorem, and Gromov's compactness theorem. They show that controlling curvature from above or below has strong geometric and topological consequences.

---

## What's next

The Riemann curvature tensor and its contractions give us a complete intrinsic description of curvature for Riemannian manifolds. But the framework is richer than just Riemannian geometry. In the final article, we generalize: the tangent bundle is just one vector bundle on $M$, and the Levi-Civita connection is just one connection. **Fiber bundles** and **connections on bundles** provide the mathematical language for gauge theories in physics, and **characteristic classes** extract topological invariants from curvature. This is where differential geometry meets topology and theoretical physics.

### Curvature computations: a practical summary

For reference, here is a recipe for computing curvature from a given metric:

1. **Start with the metric** $g_{ij}$ in coordinates. Compute the inverse metric $g^{ij}$.
2. **Compute Christoffel symbols** using $\Gamma^k_{ij} = \frac{1}{2}g^{k\ell}(\partial_i g_{j\ell} + \partial_j g_{i\ell} - \partial_\ell g_{ij})$.
3. **Compute the Riemann tensor** using $R^\rho_{\ \sigma\mu\nu} = \partial_\mu \Gamma^\rho_{\nu\sigma} - \partial_\nu \Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$.
4. **Compute the Ricci tensor** by contracting: $R_{\sigma\nu} = R^\mu_{\ \sigma\mu\nu}$.
5. **Compute the scalar curvature** by tracing: $R = g^{\sigma\nu}R_{\sigma\nu}$.

In practice, computer algebra systems (Mathematica, SageMath, xAct) handle these computations efficiently. But doing at least one example by hand — say, the sphere or hyperbolic plane — is essential for understanding what the formulas mean geometrically.

### Ricci flow: curvature as a dynamical system

We cannot leave the topic of curvature without mentioning **Ricci flow**, introduced by Richard Hamilton in 1982:

$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij}.$$

This is a heat-type equation for the metric: it evolves the metric in the direction of its Ricci curvature, smoothing out irregularities. Positively curved regions shrink, negatively curved regions expand, and the metric tends toward greater uniformity.

Hamilton showed that Ricci flow on compact 3-manifolds with positive Ricci curvature converges to a constant-curvature metric, proving that such manifolds are diffeomorphic to spherical space forms. Grigori Perelman extended this work dramatically, using Ricci flow with surgery to prove the Poincare conjecture (2003) and Thurston's geometrization conjecture — the complete classification of compact 3-manifolds. This is arguably the greatest application of curvature to topology in the history of mathematics.

### A brief history

The study of curvature goes back to Gauss's *Theorema Egregium* (1827), which proved that the Gaussian curvature of a surface is an intrinsic invariant — it can be computed from the metric alone, without reference to how the surface sits in 3-space. Riemann generalized this to arbitrary dimensions in his 1854 habilitation lecture, introducing the curvature tensor that now bears his name. The Ricci tensor was introduced by Gregorio Ricci-Curbastro and his student Tullio Levi-Civita in the early 1900s as part of developing the "absolute differential calculus" (tensor calculus). Einstein used this calculus to formulate general relativity in 1915, making Ricci curvature the language of gravity. The Bishop-Gromov comparison theorem (1960s) and Hamilton's Ricci flow (1982) brought Ricci curvature to the center of geometric analysis, culminating in Perelman's proof of the Poincare conjecture. The story of curvature is far from finished — the interaction between curvature conditions, topology, and geometric flows remains one of the most active areas of mathematical research.

---

*This is Part 11 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 10 — Riemannian Geometry](/en/differential-geometry/10-riemannian-geometry/)*

*Next: [Part 12 — Fiber Bundles and Physics](/en/differential-geometry/12-bundles-and-physics/)*
