---
title: "The Gauss-Bonnet Theorem — Where Geometry Meets Topology"
date: 2021-06-05 09:00:00
tags:
  - differential-geometry
  - gauss-bonnet
  - topology
  - euler-characteristic
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 6
series_total: 6
translationKey: "differential-geometry-6"
description: "The Gauss-Bonnet theorem connects total curvature to topology via the Euler characteristic."
---

The Gauss-Bonnet theorem is one of the most beautiful results in all of mathematics. It says: integrate the Gaussian curvature over a closed surface, and you get a topological invariant — a number that depends only on the "shape" of the surface in the rubber-sheet sense, not on its specific geometry. Stretch, bend, compress the surface however you like; the total curvature stays the same. This is the bridge between local differential geometry and global topology.

## The local Gauss-Bonnet theorem

Let $R$ be a simply connected region on an oriented surface $S$, bounded by a simple closed piecewise-smooth curve $\partial R$. Suppose $\partial R$ consists of smooth arcs $C_1, \ldots, C_k$ meeting at vertices with exterior angles $\theta_1, \ldots, \theta_k$. Then:

$$\iint_R K\,dA + \int_{\partial R} \kappa_g\,ds + \sum_{i=1}^k \theta_i = 2\pi$$

![Gauss-Bonnet theorem: total curvature equals 2pi times Euler characteristic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-gauss-bonnet/dg_fig6_gauss_bonnet.png)


where $\kappa_g$ is the geodesic curvature of the boundary (positive when the curve turns left) and the boundary is traversed with the region on the left.

This formula unifies three types of "turning": the Gaussian curvature $K$ measures how the surface itself curves; the geodesic curvature $\kappa_g$ measures how the boundary curves on the surface; and the exterior angles $\theta_i$ account for the discrete turning at corners. Their sum is always $2\pi$ — a full turn.

**Example (geodesic triangle).** A triangle $T$ whose sides are geodesics has $\kappa_g = 0$ on each side. The exterior angles are $\theta_i = \pi - \alpha_i$ where $\alpha_i$ are the interior angles. The local Gauss-Bonnet gives:

$$\iint_T K\,dA + 0 + \sum(\pi - \alpha_i) = 2\pi$$

Rearranging:

$$\alpha_1 + \alpha_2 + \alpha_3 = \pi + \iint_T K\,dA.$$

On a sphere ($K > 0$): interior angles sum to more than $\pi$. A geodesic triangle on the sphere "bulges" — its sides bow outward and enclose more angle than a flat triangle.

On the hyperbolic plane ($K < 0$): interior angles sum to less than $\pi$. The triangle is "pinched," and its area is bounded by $\pi/|K|$ no matter how large the sides grow.

On a flat surface ($K = 0$): angles sum to exactly $\pi$. Euclidean geometry.

## Geodesic triangles and area

The relationship between angle excess and area deserves emphasis. On a surface of constant curvature $K$:

$$\text{Area}(T) = \frac{1}{K}(\alpha_1 + \alpha_2 + \alpha_3 - \pi)$$

for a geodesic triangle $T$ (when $K \neq 0$). On the unit sphere ($K = 1$), the area of a geodesic triangle literally equals its angle excess. A geodesic triangle with all right angles ($\alpha_i = \pi/2$) — such as one-eighth of the sphere, bounded by two meridians and the equator — has area $= 3(\pi/2) - \pi = \pi/2$, which is indeed one-eighth of the sphere's total area $4\pi$.

On the hyperbolic plane ($K = -1$), geodesic triangles have angle deficit:

$$\text{Area}(T) = \pi - (\alpha_1 + \alpha_2 + \alpha_3).$$

Since the angles are positive and sum to less than $\pi$, the area is bounded: no geodesic triangle in the hyperbolic plane can have area exceeding $\pi$ (regardless of how long its sides are). As the vertices go to infinity (an "ideal triangle"), all angles approach 0 and the area approaches $\pi$. This is strikingly different from Euclidean geometry, where triangles can have arbitrarily large area.

**Example (geodesic disk).** A geodesic disk of radius $r$ (all points at geodesic distance $\leq r$ from a center) has boundary with geodesic curvature $\kappa_g \approx 1/r$ (for small $r$) and no corners. The local Gauss-Bonnet gives $\iint_D K\,dA + \int_{\partial D}\kappa_g\,ds = 2\pi$. For small $r$, this yields $K(p) \approx 3(2\pi r - L)/({\pi r^3})$ where $L$ is the circumference of the disk. Gaussian curvature can be detected by measuring how circumferences of small circles differ from $2\pi r$.

## The global Gauss-Bonnet theorem

**Theorem.** Let $S$ be a compact oriented surface without boundary. Then:

$$\iint_S K\,dA = 2\pi\chi(S)$$

where $\chi(S)$ is the **Euler characteristic** of $S$.

The Euler characteristic is a topological invariant defined by any triangulation: $\chi = V - E + F$ (vertices minus edges plus faces). For orientable surfaces, $\chi = 2 - 2g$ where $g$ is the genus (number of handles):

- Sphere ($g = 0$): $\chi = 2$, total curvature $= 4\pi$.
- Torus ($g = 1$): $\chi = 0$, total curvature $= 0$.
- Double torus ($g = 2$): $\chi = -2$, total curvature $= -4\pi$.
- Surface of genus $g$: $\chi = 2-2g$, total curvature $= 2\pi(2-2g)$.

*Proof sketch.* Triangulate $S$ into geodesic triangles $T_1, \ldots, T_F$. For each triangle, the local Gauss-Bonnet gives:

$$\iint_{T_j}K\,dA = (\alpha_1^j + \alpha_2^j + \alpha_3^j) - \pi.$$

Sum over all $F$ faces. On the right side: each interior vertex contributes $2\pi$ (angles around it sum to $2\pi$), giving $2\pi V$ total from vertex angles, minus $\pi F$ from the $-\pi$ per triangle. Each edge is shared by two triangles, and the geodesic curvature contributions from shared edges cancel (they are traversed in opposite directions). Careful bookkeeping gives:

$$\iint_S K\,dA = 2\pi V - \pi F + \text{(edge corrections)} = 2\pi(V - E + F) = 2\pi\chi(S).$$

The edge correction: each edge is shared by 2 triangles, and $3F = 2E$ for a triangulation, so $E = 3F/2$. Substituting yields the result.

## Verification: the sphere

For a sphere of radius $R$: $K = 1/R^2$ everywhere, surface area $= 4\pi R^2$. Total curvature:

$$\iint_{S^2} K\,dA = \frac{1}{R^2}\cdot 4\pi R^2 = 4\pi = 2\pi\cdot 2 = 2\pi\chi(S^2).$$

The sphere has $\chi = 2$. Confirmed.

## Verification: the torus

The torus has Gaussian curvature $K = \cos u / (r(R+r\cos u))$. The total curvature:

$$\iint_{T^2}K\,dA = \int_0^{2\pi}\int_0^{2\pi}\frac{\cos u}{r(R+r\cos u)}\cdot r(R+r\cos u)\,du\,dv = \int_0^{2\pi}\int_0^{2\pi}\cos u\,du\,dv = 0.$$

The positive curvature on the outer half and the negative curvature on the inner half cancel exactly. This is required by Gauss-Bonnet since $\chi(T^2) = 0$, regardless of the specific values of $R$ and $r$.

## Why this is remarkable

The left side — $\iint K\,dA$ — is a geometric quantity. It depends on the Riemannian metric. Change the metric (deform the surface) and $K$ changes at every point.

The right side — $2\pi\chi(S)$ — is a topological invariant. It does not change under any continuous deformation. You can inflate the sphere to any size, squash it into an ellipsoid, dimple it — as long as you do not tear it or glue handles, $\chi$ remains 2 and the total curvature remains $4\pi$.

Gauss-Bonnet says: no matter how you redistribute curvature across the surface, its integral is locked in by topology. Push positive curvature from one region and it must pop up elsewhere, maintaining the global constraint.

## Applications to topology

### Topology constrains geometry

A compact surface with $K > 0$ everywhere must have $\chi > 0$, hence must be homeomorphic to a sphere (or $\mathbb{RP}^2$ if non-orientable). There is no closed orientable surface of genus $\geq 1$ with everywhere-positive Gaussian curvature.

Conversely, a compact surface admitting a flat metric ($K = 0$ everywhere) must have $\chi = 0$ — it must be a torus or Klein bottle.

### The genus determines curvature sign

For genus $g \geq 2$, the total curvature $2\pi(2-2g) < 0$, so any such surface must have regions of negative curvature. By the uniformization theorem, it can even be given a metric of constant negative curvature: every compact surface of genus $\geq 2$ carries a hyperbolic metric. The area under this metric is determined by topology alone: $\text{Area} = 4\pi(g-1)$.

This creates a beautiful hierarchy: genus 0 (sphere) carries constant positive curvature; genus 1 (torus) carries flat metrics; genus $\geq 2$ carries constant negative curvature. The topology of the surface dictates the sign of curvature it can uniformly sustain.

### The hairy ball theorem

On $S^2$, every smooth tangent vector field has at least one zero. This follows from the Poincare-Hopf index theorem (a generalization of Gauss-Bonnet for vector fields): the sum of indices of zeros of any vector field on $S$ equals $\chi(S)$. Since $\chi(S^2) = 2 \neq 0$, every vector field must have zeros.

Physically: you cannot comb the hair on a coconut without creating a whorl or a bald spot.

### Angle defect in polyhedra (Descartes' theorem)

For a convex polyhedron (a "polyhedral surface"), Gaussian curvature is concentrated at vertices as discrete angle defects: $K_v = 2\pi - \sum\theta_i$ where $\theta_i$ are the face angles meeting at vertex $v$. Gauss-Bonnet becomes Descartes' theorem: $\sum_v K_v = 4\pi$. For any convex polyhedron (cube, icosahedron, any shape), the total angle defect is $4\pi$.

## Example: the Mobius strip (surfaces with boundary)

The Mobius strip is non-orientable and has boundary, so the global theorem for closed surfaces does not directly apply. But the version with boundary does: for a compact surface $M$ with boundary $\partial M$:

$$\iint_M K\,dA + \int_{\partial M}\kappa_g\,ds = 2\pi\chi(M).$$

A flat Mobius strip (made from a paper strip) has $K = 0$ and $\chi = 0$. Therefore $\int_{\partial M}\kappa_g\,ds = 0$: the total geodesic curvature of the boundary is zero. The boundary winds around twice, and the positive and negative contributions to geodesic curvature cancel.

## Example: the double torus and hyperbolic geometry

A surface of genus 2 has $\chi = -2$, so $\iint K\,dA = -4\pi$. It cannot have $K \geq 0$ everywhere. In fact, by the uniformization theorem, every compact surface of genus $\geq 2$ admits a metric of constant negative curvature $K = -1$, giving area $= 4\pi(g-1)$. The double torus with $K = -1$ has area $4\pi$.

This connects to hyperbolic geometry: the universal cover of a genus-$g$ surface (with constant negative curvature metric) is the hyperbolic plane $\mathbb{H}^2$.

## The Chern-Gauss-Bonnet theorem

The Gauss-Bonnet theorem generalizes to higher even-dimensional manifolds. For a compact oriented Riemannian manifold $M$ of dimension $2n$:

$$\int_M \text{Pf}(\Omega) = (2\pi)^n\chi(M)$$

where $\text{Pf}(\Omega)$ is the Pfaffian of the curvature 2-form. In dimension 2, $\text{Pf}(\Omega) = K\,dA/(2\pi)$ and we recover the classical theorem. In dimension 4, the integrand involves both the Riemann tensor and its contractions.

This generalization is a cornerstone of the Atiyah-Singer index theorem — one of the deepest results of 20th-century mathematics, connecting analysis (elliptic differential operators), geometry (curvature), and topology (characteristic classes).

## Where to go from here

This series covered the classical arc of differential geometry: curves, surfaces, intrinsic geometry, and the bridge to topology. The natural continuations include:

- **Riemannian geometry:** Curvature tensors, Jacobi fields, comparison theorems (Rauch, Toponogov), and the Ricci flow that proved the Poincare conjecture.
- **Lie groups and symmetric spaces:** Manifolds with algebraic structure, where geometry and algebra intertwine.
- **Gauge theory and fiber bundles:** The mathematical framework of modern particle physics, where connections generalize to non-Abelian gauge fields.

The Gauss-Bonnet theorem is not an endpoint. It is a prototype — the first and simplest in a family of index theorems that connect local differential data to global topological invariants. That pattern — local analysis constraining global structure — is perhaps the deepest theme in all of geometry.

---

*This is Part 6 of [Differential Geometry](/en/series/differential-geometry/) (6 parts).
Previous: [Part 5 — Smooth Manifolds, Tangent Bundles, and Connections](/en/differential-geometry/05-manifolds-and-connections/)*
